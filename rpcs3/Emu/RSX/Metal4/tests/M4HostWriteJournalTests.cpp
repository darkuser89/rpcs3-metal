#include "M4GuardedGuestMemory.h"
#include "M4HostWriterGate.h"
#include <array>
#include <atomic>
#include <iostream>
#include <thread>

namespace
{
using namespace metal4;
static_assert(!std::is_copy_constructible_v<guarded_guest_memory> && !std::is_move_constructible_v<guarded_guest_memory>,
	"Watch registrations and monotonic revision ownership must not be cloned");
void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}
template <typename Operation>
void refused(Operation&& operation)
{
	bool failed = false;
	try { operation(); } catch (const std::exception&) { failed = true; }
	require(failed, "Unsafe host completion was accepted");
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex mutex;
  public:
	std::array<std::byte, 4096> bytes{}, other{};
	std::shared_ptr<host_writer_gate> gate;
	bool fail_alias = false;
	unsigned maps = 0, reservations = 0;
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* current = guest_memory_epoch::current(*this)) { operation(*current); return; }
		const auto invoke = [&]
		{
			std::lock_guard lock(mutex);
			guest_memory_epoch epoch(*this, *this);
			operation(*this);
		};
		if (gate)
		{
			auto excluded = gate->acquire([] { std::this_thread::yield(); });
			require(bool(excluded), "Fixture host writer capture refused");
			invoke();
		}
		else invoke();
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		++maps;
		validate_guest_surface_range(range);
		require(range.address < 12288 && range.size <= 4096 - range.address % 4096, "Fixture map crossed a backing");
		return (range.address < 8192 ? bytes.data() : other.data()) + range.address % 4096;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		if (fail_alias) throw std::runtime_error("Alias planning failure");
		return (a.address < 8192) == (b.address < 8192) &&
			overlaps(guest_surface_range{a.address % 4096, a.size}, guest_surface_range{b.address % 4096, b.size});
	}
	void written(guest_surface_range) noexcept override { ++reservations; }
};

void completed_aliases()
{
	auto access = std::make_shared<fixture>();
	guarded_guest_memory memory(access);
	const auto a = memory.watch_surface(guest_byte_coverage({64, 16}));
	const auto alias = memory.watch_surface(guest_byte_coverage({4160, 16}));
	const auto unrelated = memory.watch_surface(guest_byte_coverage({8256, 16}));
	const auto old = memory.read_surface(a), mirror = memory.read_surface(alias), other = memory.read_surface(unrelated);
	{
		auto write = memory.begin_unleased_host_write({4164, 4});
		const auto maps = access->maps;
		refused([&] { (void)memory.read_surface(a); });
		require(access->maps == maps && memory.pending_host_writes() == 1, "Active intent read bytes or disappeared");
		std::fill_n(access->bytes.begin() + 68, 4, std::byte{0x7b});
		write.commit();
		write.commit(); // Retired intent cannot commit/revise twice.
	}
	const auto current = memory.read_surface(a);
	require(current.revision > old.revision && memory.surface_revision(alias) > mirror.revision &&
		current.bytes[4] == std::byte{0x7b} && !memory.pending_host_writes(), "Completed host bytes lost their physical alias revision");
	require(memory.surface_revision(unrelated) == other.revision && access->reservations == 0,
		"Host revision drain changed unrelated memory or invented delayed reservation stores");
	require(!memory.compare_write(a, old.revision, old.bytes), "Stale pre-DMA CAS overwrote completed guest bytes");
	const auto same_bits = memory.surface_revision(a);
	{
		auto write = memory.begin_unleased_host_write({64, 1});
		access->bytes[64] = access->bytes[64];
		write.commit();
	}
	require(memory.surface_revision(a) > same_bits, "Same-bit completed store retained a stale revision");
	const guest_surface_layout pitched{{128, 32}, 4, 2, 1, 16, {}, {}};
	const auto padding = memory.watch_surface(pitched);
	const auto tag = memory.surface_revision(padding);
	{
		auto write = memory.begin_unleased_host_write({136, 4});
		std::fill_n(access->bytes.begin() + 136, 4, std::byte{3});
		write.commit();
	}
	require(memory.surface_revision(padding) == tag, "Padding-only host write invalidated active surface bytes");
	{
		auto write = memory.begin_unleased_host_write({64, 1});
		access->bytes[64] = std::byte{5};
		write.commit();
	}
	memory.notify_mapping_changed({0x80000000, 4096});
	require(memory.read_surface(a).bytes[0] == std::byte{5} && !memory.pending_host_writes(), "Unrelated later mapping blocked a stable completed destination");
	{
		auto write = memory.begin_unleased_host_write({64, 1});
		access->bytes[64] = std::byte{6};
		memory.begin_mapping_change({0x80000000, 4096});
		memory.end_mapping_change({0x80000000, 4096});
		write.commit();
	}
	require(memory.read_surface(a).bytes[0] == std::byte{6} && !memory.pending_host_writes(), "Finished unrelated transition made a stable host store ambiguous");
}

void batches_and_refusal()
{
	auto access = std::make_shared<fixture>();
	guarded_guest_memory memory(access, 16, 4096, 1024, 2);
	const auto a = memory.watch_surface(guest_byte_coverage({64, 4}));
	const auto b = memory.watch_surface(guest_byte_coverage({128, 4}));
	const auto last = memory.surface_revision(b);
	auto first = memory.begin_unleased_host_write(a.range);
	access->bytes[64] = std::byte{1}; first.commit();
	auto second = memory.begin_unleased_host_write(b.range);
	access->bytes[128] = std::byte{2};
	refused([&] { (void)memory.surface_revision(a); });
	refused([&] { (void)memory.begin_unleased_host_write({256, 1}); });
	require(memory.pending_host_writes() == 2, "Failed drain/budget refusal discarded completion records");
	second.commit();
	access->fail_alias = true;
	refused([&] { (void)memory.surface_revision(a); });
	require(memory.pending_host_writes() == 2, "Planning failure retired an unapplied batch");
	access->fail_alias = false;
	require(memory.surface_revision(a) == last + 1 && memory.surface_revision(b) == last + 1 && !memory.pending_host_writes(),
		"Failed batch partially revised watches or successful batch lost one destination");
	{
		auto reused = memory.begin_unleased_host_write({256, 1}); reused.commit();
	}
	(void)memory.surface_revision(a);
	require(!memory.pending_host_writes(), "Successful drain failed to reclaim bounded slots");
	refused([&] { (void)memory.begin_unleased_host_write({0xffffffff, 2}); });
	refused([&] { (void)memory.begin_unleased_host_write({0, 0}); });
	require(!memory.pending_host_writes(), "Invalid pre-store request consumed a slot");
}

void ambiguous_writes()
{
	for (unsigned kind = 0; kind < 4; ++kind)
	{
		auto access = std::make_shared<fixture>();
		guarded_guest_memory memory(access);
		const auto watched = memory.watch_surface(guest_byte_coverage({64, 4}));
		if (kind == 0)
		{
			refused([&]
			{
				auto partial = memory.begin_unleased_host_write(watched.range);
				access->bytes[64] = std::byte{0x6e};
				throw std::runtime_error("Partial DMA store");
			});
		}
		else
		{
			auto write = memory.begin_unleased_host_write(watched.range);
			access->bytes[64] = std::byte{0x6e};
			if (kind == 1) memory.begin_mapping_change({0x80000000, 1});
			if (kind == 2) { memory.begin_mapping_change(watched.range); memory.end_mapping_change(watched.range); }
			write.commit();
			if (kind == 1) memory.end_mapping_change({0x80000000, 1});
			if (kind == 3) { memory.begin_mapping_change(watched.range); memory.end_mapping_change(watched.range); }
		}
		const auto maps = access->maps;
		refused([&] { (void)memory.read_surface(watched); });
		refused([&] { (void)memory.surface_revision(watched); });
		require(memory.pending_host_writes() == 1 && access->maps == maps && access->bytes[64] == std::byte{0x6e},
			"Ambiguous/remapped host write disappeared,read unverified bytes or overwrote dirty intent");
	}
	auto epochs = std::make_shared<mapping_epoch>();
	host_write_journal journal(epochs);
	epochs->begin_change({0, 4096});
	refused([&] { (void)journal.begin_unleased({64, 4}); });
	require(!journal.pending(), "Busy pre-store mapping allocated an intent");
	epochs->end_change({0, 4096});
	refused([&] { host_write_journal invalid(epochs, 0); });
	refused([&] { host_write_journal invalid(epochs, host_write_journal::max_entries + 1); });
	std::optional<host_write_journal::intent> surviving;
	{
		host_write_journal temporary(epochs);
		surviving.emplace(temporary.begin_unleased({64, 4}));
	}
	surviving->commit(); // State/epoch ownership outlives its wrapper,no raw VM owner.
}

void actual_store_ticket_order()
{
	auto access = std::make_shared<fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	guarded_guest_memory memory(access, 16, 4096, 1024, 256);
	const auto watched = memory.watch_surface(guest_byte_coverage({64, 16}));
	const auto before = memory.surface_revision(watched);
	std::array<std::thread, 4> producers;
	std::array<std::exception_ptr, 4> failures;
	std::atomic<unsigned> done{0};
	for (unsigned i = 0; i < producers.size(); ++i)
		producers[i] = std::thread([&, i]
		{
			try
			{
				for (unsigned value = 1; value <= 32; ++value)
				{
					auto ticket = access->gate->admit();
					auto writer = ticket.activate();
					require(bool(writer), "Concurrent host writer activation failed");
					auto completion = memory.begin_unleased_host_write({64 + i * 4, 4});
					std::fill_n(access->bytes.begin() + 64 + i * 4, 4, std::byte(value));
					completion.commit();
				}
			}
			catch (...) { failures[i] = std::current_exception(); }
			++done;
		});
	std::exception_ptr capture_failure;
	try { while (done.load() != 4) { (void)memory.read_surface(watched); std::this_thread::yield(); } }
	catch (...) { capture_failure = std::current_exception(); }
	for (auto& thread : producers) thread.join();
	if (capture_failure) std::rethrow_exception(capture_failure);
	for (auto failure : failures) if (failure) std::rethrow_exception(failure);
	const auto final = memory.read_surface(watched);
	require(final.revision > before && std::all_of(final.bytes.begin(), final.bytes.end(), [](auto byte) { return byte == std::byte{32}; }) &&
		!memory.pending_host_writes(), "Continuous ticket/store/completion/capture ordering lost guest writes");
}
void destination_boundaries()
{
	constexpr std::uintptr_t base = 0x20'0000'0000ull;
	require(!host_write_journal::destination(base, base - 1, 1) && !host_write_journal::destination(base, base + 0x2'0000'0000ull, 1) &&
		!host_write_journal::destination(base, 0, 1) && !host_write_journal::destination(base, base, 0), "Host/null/empty destination entered the guest journal");
	const auto zero = host_write_journal::destination(base, base, 4);
	const auto mirror = host_write_journal::destination(base, base + 0x1'0000'0040ull, 16);
	const auto tail = host_write_journal::destination(base, base + 0x1'ffff'fff0ull, 16);
	require(zero && zero->address == 0 && mirror && mirror->address == 64 && tail && tail->address == 0xfffffff0 && tail->size == 16,
		"Guest zero/sudo mirror/last byte destination conversion failed");
	refused([&] { (void)host_write_journal::destination(base, base + 0xfffffff0, 17); });
	refused([&] { (void)host_write_journal::destination(base, base + 0x1'ffff'ffffull, 2); });
	require(!host_write_journal::destination(std::numeric_limits<std::uintptr_t>::max() - 1, 0, 1), "Wrapped VM reservation accepted an unrelated pointer");
}
} // namespace
int main()
{
	try
	{
		destination_boundaries(); completed_aliases(); batches_and_refusal(); ambiguous_writes(); actual_store_ticket_order();
		std::cout << "Host DMA completions:integer guest/sudo/host/null/tail/wrap destination domains;successful no-fault/same-bit stores revise physical active-byte aliases;stale CAS/padding/unrelated mapping precision;atomic batches and retry after alias planning failure;finite slots/reuse;partial/active/remapped/uncertain stores metadata retained without reads/fixture bytes preserved;128 real threaded stores under continuous writer tickets/captures;owned epoch/intent lifetime;no App or guest VM started\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
