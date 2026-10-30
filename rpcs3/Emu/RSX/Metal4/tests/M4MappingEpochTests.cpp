#include "M4GuardedGuestMemory.h"
#include "M4ScaledImageMapping.h"
#include "../Core/RSXMappingChange.h"
#include <array>
#include <iostream>
#include <mutex>
#include <semaphore>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void refused(F&& operation)
{
	bool failed = false;
	try
	{
		operation();
	}
	catch (const std::exception&)
	{
		failed = true;
	}
	require(failed, "Mapping epoch failed to refuse");
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex mutex;

  public:
	std::array<std::byte, 4096> bytes{};
	std::array<std::byte, 4096> alternate{};
	bool remapped = false;
	unsigned entries = 0, notifications = 0;
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			operation(*current);
			return;
		}
		std::lock_guard lock(mutex);
		++entries;
		guest_memory_epoch epoch(*this, *this);
		operation(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		require(guest_memory_epoch::current(*this) == this, "Mapping fixture escaped exclusion");
		const auto offset = range.address % 4096;
		if (range.address >= 16384 || range.size > 4096 - offset)
			throw std::range_error("Mapping fixture span invalid");
		return (remapped && range.address < 4096 ? alternate.data() : bytes.data()) + offset;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		return overlaps(guest_surface_range{a.address % 4096 + (remapped && a.address < 4096 ? 0x10000u : 0u), a.size},
			guest_surface_range{b.address % 4096 + (remapped && b.address < 4096 ? 0x10000u : 0u), b.size});
	}
	void written(guest_surface_range) noexcept override { ++notifications; }
};
void range_reference()
{
	mapping_epoch epochs;
	require(mapping_epoch::storage_bytes == 8'454'144, "Mapping table lost its fixed storage bound");
	std::vector<std::uint64_t> pages(1 << 20);
	std::uint64_t random = 0x831789;
	const auto next = [&]()
	{
		random = random * 6364136223846793005ull + 1442695040888963407ull;
		return random;
	};
	const auto publish = [&](guest_surface_range range)
	{
		epochs.notify(range);
		const auto stamp = epochs.current();
		if (range.size)
			for (std::size_t page = range.address / 4096; page < (std::uint64_t(range.address) + range.size + 4095) / 4096; ++page)
				pages[page] = stamp;
	};
	const auto check = [&](guest_surface_range range)
	{
		std::uint64_t expected = 0;
		if (range.size)
			for (std::size_t page = range.address / 4096; page < (std::uint64_t(range.address) + range.size + 4095) / 4096; ++page)
				expected = std::max(expected, pages[page]);
		require(epochs.current(range) == expected, "Hierarchical mapping revision differs from independent page reference");
	};
	publish({4095, 2});
	check({0, 1});
	check({4096, 1});
	check({8192, 1});
	publish({0x1fffff, 2});
	check({0x1ff000, 4096});
	check({0x200000, 4096});
	check({0x201000, 4096});
	publish({0, 0x1'0000'0000ull});
	check({0xffffffff, 1});
	check({0, 0x1'0000'0000ull});
	publish({0x800001, 2 * 0x100000 - 2});
	check({0x800000, 0x200000});
	check({0x7fffff, 1});
	for (unsigned i = 0; i < 5000; ++i)
	{
		const auto address = std::uint32_t(next() >> 16);
		const auto size = std::min<std::uint64_t>(1 + (next() % (3 * 0x100000)), 0x1'0000'0000ull - address);
		publish({address, std::size_t(size)});
		check({address, std::size_t(size)});
		const auto query = std::uint32_t(next() >> 16);
		check({query, std::size_t(std::min<std::uint64_t>(next() % 0x200000, 0x1'0000'0000ull - query))});
	}
	const auto before = epochs.current();
	epochs.notify({0xffffffff, 0});
	require(epochs.current() == before && epochs.current({0xffffffff, 0}) == 0, "Empty mapping range invented a revision");
	refused([&] { (void)epochs.current({0xffffffff, 2}); });
	require(epochs.current() == before, "Invalid query poisoned a valid producer");
}

struct mapping_sink
{
	mapping_epoch epochs;
	unsigned begins = 0, ends = 0;
	guest_surface_range last_begin{}, last_end{};
	void on_mapping_change_begin(std::uint32_t address, std::uint32_t size) noexcept
	{
		++begins;
		last_begin = {address, size};
		epochs.begin_change(last_begin);
	}
	void on_mapping_change_end(std::uint32_t address, std::uint32_t size) noexcept
	{
		++ends;
		last_end = {address, size};
		epochs.end_change(last_end);
	}
};

void transition_observers()
{
	mapping_sink sink;
	const auto original = sink.epochs.observe();
	require(original && sink.epochs.unchanged(*original), "Idle mapping observer was refused");
	mapping_epoch foreign;
	require(!foreign.unchanged(*original), "Foreign mapping owner accepted an identity token");
	const auto other_page = sink.epochs.current({8192, 1});
	{
		rsx::mapping_change_scope change(&sink, 64, 16);
		require(!sink.epochs.observe(), "Observer accepted metadata before backing mutation ended");
		require(!sink.epochs.unchanged(*original), "Begin failed to invalidate previous identity observation");
		{
			rsx::mapping_change_scope nested(&sink, 4160, 16);
			require(!sink.epochs.observe(), "Nested mapping transition exposed idle metadata");
		}
		require(!sink.epochs.observe(), "Inner completion exposed unfinished outer transition");
	}
	const auto finished = sink.epochs.observe();
	require(finished && sink.epochs.unchanged(*finished) && !sink.epochs.unchanged(*original), "Finished transition lost serial/ABA rejection");
	require(sink.begins == 2 && sink.ends == 2 && sink.last_end.address == 64 && sink.last_end.size == 16,
		"VM scope did not preserve paired range/lifetime");
	require(sink.epochs.current({8192, 1}) == other_page, "Identity observation invalidated an unrelated page watch");
	mapping_epoch other;
	require(!other.unchanged(*finished), "Foreign mapping owner accepted an identity token");
	sink.epochs.notify({0x8000, 1});
	require(!sink.epochs.unchanged(*finished), "One-shot mapping notice retained an old identity token");
	refused([&]
	{
		rsx::mapping_change_scope partial(&sink, 128, 4);
		throw std::runtime_error("Partially changed mapping");
	});
	require(sink.ends == 3 && sink.last_end.address == 128 && sink.last_end.size == 4 && sink.epochs.observe(),
		"Exception failed to finish mapping transition");
	{
		rsx::mapping_change_scope absent(static_cast<mapping_sink*>(nullptr), 0, 4096);
	}
	mapping_epoch unmatched;
	unmatched.end_change({0, 1});
	require(!unmatched.observe(), "Unmatched mapping end exposed a valid identity");
	unmatched.begin_change({0, 1});
	unmatched.end_change({0, 1});
	require(!unmatched.observe(), "Poisoned transition counter recovered after later callbacks");
	mapping_epoch saturated;
	for (unsigned i = 0; i <= 0xffff; ++i) saturated.begin_change({0, 0});
	for (unsigned i = 0; i <= 0xffff; ++i) saturated.end_change({0, 0});
	require(!saturated.observe(), "Active transition counter wrapped into an idle identity");
}

void concurrent_transitions()
{
	mapping_epoch epochs;
	std::atomic<unsigned> backing{1}, page_flags{1};
	std::binary_semaphore began(0), finish(0);
	const auto original = epochs.observe();
	std::thread producer([&]
	{
		epochs.begin_change({0, 4096});
		backing.store(2, std::memory_order_relaxed);
		began.release();
		finish.acquire();
		page_flags.store(2, std::memory_order_relaxed);
		epochs.end_change({0, 4096});
	});
	began.acquire();
	const bool busy_refused = !epochs.observe() && !epochs.unchanged(*original);
	// Another producer finishes while the first still owns a partial mapping.
	std::thread second([&]
	{
		epochs.begin_change({8192, 4096});
		epochs.end_change({8192, 4096});
	});
	second.join();
	const bool overlap_refused = !epochs.observe();
	finish.release();
	producer.join();
	require(busy_refused, "Concurrent reader accepted half-published backing/page flags");
	require(overlap_refused, "Other producer completion exposed an unfinished mapping");
	const auto stable = epochs.observe();
	require(stable && backing.load(std::memory_order_relaxed) == 2 && page_flags.load(std::memory_order_relaxed) == 2 &&
		epochs.unchanged(*stable), "Stable observation failed to acquire completed atomic mapping metadata");
}
} // namespace
int main()
{
	try
	{
		transition_observers();
		concurrent_transitions();
		range_reference();
		mapping_epoch epochs;
		const auto initial = epochs.current();
		std::array<std::thread, 4> workers;
		for (auto& worker : workers)
			worker = std::thread([&]
			{
				for (unsigned i = 0; i < 10000; ++i)
					epochs.notify({0, 4096});
			});
		for (auto& worker : workers)
			worker.join();
		require(epochs.current() == initial + 40000 && epochs.current({0, 4096}) == epochs.current() && epochs.current({4096, 1}) == 0,
			"Concurrent notifications lost or regressed global/page generations");
		mapping_epoch exhausted(std::numeric_limits<std::uint64_t>::max() - 1);
		exhausted.notify({0, 1});
		refused([&] { (void)exhausted.current(); });
		exhausted.notify({0, 1});
		refused([&] { (void)exhausted.current({0, 1}); });
		mapping_epoch invalid;
		invalid.notify({0xffffffff, 2});
		refused([&] { (void)invalid.current(); });
		invalid.notify({0, 1});
		refused([&] { (void)invalid.current({0, 1}); });
		refused([] { mapping_epoch bad(0); });
		refused([] { mapping_epoch bad(std::numeric_limits<std::uint64_t>::max()); });
		auto provider = std::make_shared<fixture>();
		guarded_guest_memory memory(provider);
		const auto a = memory.watch_surface(guest_byte_coverage({64, 16}));
		const auto alias = memory.watch_surface(guest_byte_coverage({4160, 16}));
		const auto other = memory.watch_surface(guest_byte_coverage({8704, 16}));
		const auto before = memory.read_surface(a);
		const auto mirror_before = memory.surface_revision(alias), other_before = memory.surface_revision(other), generation = memory.mapping_generation();
		const auto entries = provider->entries, notices = provider->notifications;
		provider->execute([&](auto&)
		{
			const auto inside = provider->entries;
			memory.begin_mapping_change(a.range);
			require(!memory.observe_mapping(), "Guarded memory exposed a live VM transition");
			memory.end_mapping_change(a.range);
			require(provider->entries == inside, "Mapping callback recursively entered provider");
		});
		const auto completed_mapping = memory.observe_mapping();
		require(completed_mapping && memory.mapping_unchanged(*completed_mapping), "Guarded memory lost completed VM observation");
		require(provider->entries == entries + 1 && provider->notifications == notices && memory.mapping_generation() == generation + 2, "Mapping callback performed byte writer work");
		require(!memory.compare_write(a, before.revision, before.bytes) && memory.read_surface(a).bytes == before.bytes && provider->notifications == notices, "Identical mapping ABA published stale bytes");
		require(memory.surface_revision(alias) == mirror_before && memory.surface_revision(other) == other_before, "Virtual remap invalidated an unaffected physical mirror or unrelated page");
		const auto fresh = memory.read_surface(a);
		require(memory.compare_write(a, fresh.revision, fresh.bytes).has_value(), "Fresh mapping capture failed");
		require(memory.surface_revision(alias) > mirror_before, "Physical alias byte writes lost their shared revision");
		const auto stable = memory.surface_revision(a);
		require(memory.surface_revision(a) == stable, "Stable mapping invented revisions");
		const auto old_backing = memory.read_surface(a), unchanged_mirror = memory.read_surface(alias);
		provider->execute([&](auto&)
		{
			provider->alternate.fill(std::byte{0x7b});
			provider->remapped = true;
			memory.notify_mapping_changed(a.range);
		});
		require(!memory.compare_write(a, old_backing.revision, old_backing.bytes), "Actual backing replacement accepted old virtual revision");
		const auto replacement = memory.read_surface(a);
		require(std::all_of(replacement.bytes.begin(), replacement.bytes.end(), [](auto byte) { return byte == std::byte{0x7b}; }), "Remap did not capture the replacement backing");
		require(memory.compare_write(a, replacement.revision, replacement.bytes).has_value(), "Replacement backing refused fresh bytes");
		require(memory.surface_revision(alias) == unchanged_mirror.revision && memory.read_surface(alias).bytes == unchanged_mirror.bytes && old_backing.bytes == unchanged_mirror.bytes,
			"Virtual backing replacement changed the old physical mirror or owned snapshot");
		const auto current = memory.read_surface(a), other_current = memory.read_surface(other);
		const std::array<guest_surface_write, 2> batch{{{a, current.revision, current.bytes}, {other, other_current.revision, other_current.bytes}}};
		memory.notify_mapping_changed({0x80000000, 4096});
		require(memory.compare_write_many(batch).has_value(), "Unrelated mapping refused a valid batch");
		const auto batch_a = memory.read_surface(a), batch_other = memory.read_surface(other);
		const std::array<guest_surface_write, 2> stale_batch{{{a, batch_a.revision, batch_a.bytes}, {other, batch_other.revision, batch_other.bytes}}};
		const auto batch_notices = provider->notifications;
		memory.notify_mapping_changed(other.range);
		require(!memory.compare_write_many(stale_batch) && provider->notifications == batch_notices && memory.surface_revision(a) == batch_a.revision, "Last mapped batch member partially published or invalidated first member");
		scaled_image_mapping_epoch::mappings values{};
		values[0] = {64, 16, {}, {}};
		values[1] = {8704, 16, {}, {}};
		const auto query = [values] { return values; };
		const auto global = [&] { return memory.mapping_generation(); };
		const auto ranges = [&](const auto& owned) { return scaled_image_mapping_generation(owned, memory); };
		const scaled_image_mapping_epoch ticket(query, global, ranges);
		memory.notify_mapping_changed({0x80000000, 4096});
		ticket.validate();
		memory.notify_mapping_changed(a.range);
		refused([&] { ticket.validate(); });
		refused([&]
		{
			scaled_image_mapping_epoch malformed([&]
			{
				memory.notify_mapping_changed(a.range);
				return values;
			}, global, ranges);
		});
		bool change = false;
		scaled_image_mapping_epoch query_change([&]
		{
			if (change)
				memory.notify_mapping_changed(a.range);
			return values;
		}, global, ranges);
		change = true;
		refused([&] { query_change.validate(); });
		refused([&] { scaled_image_mapping_epoch absent(query, {}, ranges); });
		// Tile fallback includes its full allocation;disjoint segment lists take
		// precedence over a fictitious linear allocation envelope.
		values[0] = {64, 16, scaled_image_tile_mapping{0x400000, 0x200000, 1024, 0, 1, 0, 0, 0}, {}};
		values[1] = {0, 0, {}, {}};
		const scaled_image_mapping_epoch tile([&] { return values; }, global, ranges);
		memory.notify_mapping_changed({0x5ff000, 4096});
		refused([&] { tile.validate(); });
		values[0].segments = {{0x400000, 0, 0x100000}, {0x800000, 0x100000, 0x100000}};
		const scaled_image_mapping_epoch scatter([values] { return values; }, global, ranges);
		memory.notify_mapping_changed({0x500000, 4096});
		scatter.validate();
		memory.notify_mapping_changed({0x8ff000, 4096});
		refused([&] { scatter.validate(); });
		memory.notify_mapping_changed({0xffffffff, 2});
		refused([&] { (void)memory.surface_revision(a); });
		refused([&] { (void)memory.compare_write(a, stable, before.bytes); });
		std::cout << "Native VM mapping ranges:paired VM scope/exception completion;nested/concurrent transition observers reject half-published metadata,ABA,foreign owners,one-shot notices,unmatched end/count overflow;40000 concurrent lock-free notifications;5000 independent page-oracle updates/10000 random queries plus boundaries/full4GiB;fixed8454144byte table;affected page ABA/stale CAS/last-batch refusal,unrelated batch/physical mirror preserved,physical byte aliases still revised;actual backing replacement keeps old mirror/owned snapshot;owned metadata/tile allocation/disjoint last-segment refusal/gap preservation/query-change/exhaustion poison;no provider reentry/byte store/no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
