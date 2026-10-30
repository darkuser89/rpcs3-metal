#include "M4GuardedGuestMemory.h"
#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Atomic guest batch differs");
}
template <typename F>
void refused(F&& f)
{
	bool failed = false;
	try
	{
		f();
	}
	catch (const std::exception&)
	{
		failed = true;
	}
	require(failed);
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex mutex;

  public:
	std::array<std::byte, 8192> bytes{};
	std::optional<unsigned> deny;
	unsigned notifications = 0;
	void execute(const std::function<void(view&)>& op) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			op(*current);
			return;
		}
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		op(*this);
	}
	std::byte* map(guest_surface_range range, bool write) override
	{
		require(guest_memory_epoch::current(*this) == this);
		const auto address = range.address % bytes.size();
		if (range.address >= 16384 || range.size > bytes.size() - address || (write && deny && address <= *deny && *deny - address < range.size))
			throw std::range_error("Batch mapping denied");
		return bytes.data() + address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		return overlaps(guest_surface_range{a.address % 8192, a.size}, guest_surface_range{b.address % 8192, b.size});
	}
	void written(guest_surface_range) noexcept override { ++notifications; }
};
} // namespace
int main()
{
	try
	{
		auto provider = std::make_shared<fixture>();
		guarded_guest_memory memory(provider);
		const auto a = memory.watch_surface(guest_byte_coverage({64, 16}));
		const auto b = memory.watch_surface(guest_byte_coverage({512, 16}));
		const auto alias = memory.watch_surface(guest_byte_coverage({8256, 16}));
		const std::array<std::byte, 16> first{std::byte{0x11}}, second{std::byte{0x22}};
		const auto writes = [&]
		{ return std::array<guest_surface_write, 2>{{{a, memory.surface_revision(a), first}, {b, memory.surface_revision(b), second}}}; };
		auto old = writes();
		memory.write({512, 1}, std::array{std::byte{0x77}});
		const auto notices = provider->notifications;
		require(!memory.compare_write_many(old) && memory.read_surface(a).bytes == std::vector<std::byte>(16) && provider->notifications == notices);
		auto fresh = writes();
		provider->deny = 512;
		refused([&]
				{ (void)memory.compare_write_many(fresh); });
		require(memory.surface_revision(a) == fresh[0].expected && provider->notifications == notices && provider->bytes[64] == std::byte{0});
		provider->deny.reset();
		const auto alias_before = memory.surface_revision(alias);
		const auto revision = memory.compare_write_many(fresh);
		require(revision && memory.surface_revision(a) == *revision && memory.surface_revision(b) == *revision && memory.surface_revision(alias) > alias_before);
		require(memory.read_surface(a).bytes == std::vector<std::byte>(first.begin(), first.end()) && memory.read_surface(b).bytes == std::vector<std::byte>(second.begin(), second.end()));
		// Freeze every input before the first store,including cross-payload VM aliases.
		provider->execute([&](auto&)
						  {
            const std::array<guest_surface_write,2> swap{{{a,memory.surface_revision(a),{provider->bytes.data()+512,16}},
                {b,memory.surface_revision(b),{provider->bytes.data()+64,16}}}};
            require(bool(memory.compare_write_many(swap))); });
		require(provider->bytes[64] == std::byte{0x22} && provider->bytes[512] == std::byte{0x11});
		const std::array<guest_surface_write, 2> overlapping{{{a, memory.surface_revision(a), first}, {alias, memory.surface_revision(alias), second}}};
		require(bool(memory.compare_write_many(overlapping)) && provider->bytes[64] == std::byte{0x22});
		const auto unchanged = provider->bytes;
		fresh = writes();
		fresh[1].bytes = {};
		refused([&]
				{ (void)memory.compare_write_many(fresh); });
		require(provider->bytes == unchanged);
		guarded_guest_memory bounded(provider, 1024, 31);
		const auto x = bounded.watch_surface(guest_byte_coverage({1024, 16})), y = bounded.watch_surface(guest_byte_coverage({2048, 16}));
		const std::array<guest_surface_write, 2> too_big{{{x, bounded.surface_revision(x), first}, {y, bounded.surface_revision(y), second}}};
		refused([&]
				{ (void)bounded.compare_write_many(too_big); });
		guarded_guest_memory patches(provider, 1024, 64, 1);
		const auto p = patches.watch_surface(guest_byte_coverage({3072, 16})), q = patches.watch_surface(guest_byte_coverage({4096, 16}));
		const std::array<guest_surface_write, 2> too_many{{{p, patches.surface_revision(p), first}, {q, patches.surface_revision(q), second}}};
		refused([&]
				{ (void)patches.compare_write_many(too_many); });
		require(provider->bytes == unchanged && memory.compare_write_many({}) == std::optional<std::uint64_t>{0});
		memory.write(a.range, std::vector<std::byte>(16));
		memory.write(b.range, std::vector<std::byte>(16));
		std::atomic<bool> done = false, torn = false;
		std::thread reader([&]
						   {
            while (!done.load()) provider->execute([&](auto&) { if (provider->bytes[64] != provider->bytes[512]) torn = true; }); });
		try
		{
			for (unsigned i = 0; i < 1000; ++i)
			{
				const std::array<std::byte, 16> value{std::byte(i & 255)};
				const std::array<guest_surface_write, 2> batch{{{a, memory.surface_revision(a), value}, {b, memory.surface_revision(b), value}}};
				require(bool(memory.compare_write_many(batch)));
			}
		}
		catch (...)
		{
			done = true;
			reader.join();
			throw;
		}
		done = true;
		reader.join();
		require(!torn);
		std::cout << "Native atomic guest batch:last-revision/last-map/payload/aggregate-budget/patch-limit refusals preserve every target;cross-payload aliases frozen,physical aliases share revision;1000two-range publications observed without tearing;no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
