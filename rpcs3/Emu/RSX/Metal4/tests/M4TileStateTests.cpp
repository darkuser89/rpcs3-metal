#include "M4TileState.h"
#include "M4GuardedGuestMemory.h"
#include "M4ScaledImageMapping.h"
#include <atomic>
#include <future>
#include <iostream>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
template <typename F>
void refused(F&& operation)
{
	bool failed = false;
	try { operation(); }
	catch (const std::exception&) { failed = true; }
	require(failed, "Native tile state failed to refuse");
}
void oracle()
{
	auto tiles = std::make_unique<tile_state>();
	std::array<tile_descriptor, 15> values{};
	std::array<std::array<std::uint64_t, 65536>, 2> entries{};
	std::uint64_t random = 0x531814, generation = 1;
	const auto next = [&]
	{
		random = random * 6364136223846793005ull + 1442695040888963407ull;
		return random;
	};
	const auto mark = [&](const tile_descriptor& value)
	{
		if (value.bound)
			for (auto i = value.offset >> 16; i < (std::uint64_t(value.offset) + value.size) >> 16; ++i)
				entries[value.location][i] = generation;
	};
	require(tile_state::table_bytes == 1048576, "Tile table exceeded its fixed bound");
	for (unsigned iteration = 0; iteration < 10000; ++iteration)
	{
		const auto index = next() % values.size();
		const auto first = std::uint32_t(next() % 65536);
		const auto count = std::min<unsigned>(1 + next() % 32, 65536 - first);
		const tile_descriptor value{unsigned(next() % 2), first << 16, count << 16, 256, 0, 0, unsigned(next() % 4), bool(next() % 3)};
		if (value != values[index])
		{
			++generation;
			mark(values[index]); mark(value); values[index] = value;
		}
		tiles->update(index, value);
		require(tiles->snapshot() == values && tiles->current() == generation, "Tile snapshot or global version lost a producer");
		const auto address = std::uint32_t(next() >> 16);
		const auto size = std::size_t(std::min<std::uint64_t>(next() % (4 << 20), 0x1'0000'0000ull - address));
		const auto location = unsigned(next() % 2);
		std::uint64_t expected = 0;
		if (size)
			for (auto i = address >> 16; i < (std::uint64_t(address) + size + 65535) / 65536; ++i)
				expected = std::max(expected, entries[location][i]);
		require(tiles->current(location, {address, size}) == expected, "Tile range differs from independent page oracle");
	}
	tiles->update(0, {1, 0xffff0000, 65536, 256, 0, 0, 0, true});
	require(tiles->current(1, {0xffffffff, 1}) == tiles->current(), "Tile top byte lost");
	require(tiles->current(1, {0xffffffff, 0}) == 0, "Empty tile query invented revision");
	refused([&] { (void)tiles->current(1, {0xffffffff, 2}); });
	refused([&] { (void)tiles->current(2, {0, 1}); });
	const auto before = tiles->current();
	refused([&] { tiles->update(15, {}); });
	require(before == tiles->current(), "Invalid index changed the epoch");
	const auto same = tiles->get(0); tiles->update(0, same);
	require(before == tiles->current(), "Identical descriptor invented a transition");
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex mutex;
  public:
	std::array<std::byte, 4096> bytes{};
	unsigned notifications = 0;
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* active = guest_memory_epoch::current(*this)) { operation(*active); return; }
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		operation(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		require(guest_memory_epoch::current(*this) == this, "Tile capture lost provider view");
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Tile fixture map");
		return bytes.data() + range.address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(a, b); }
	void written(guest_surface_range) noexcept override { ++notifications; }
};
void transactions()
{
	auto provider = std::make_shared<fixture>();
	auto memory = std::make_shared<guarded_guest_memory>(provider);
	auto tiles = std::make_shared<tile_state>();
	const auto dispatch = tile_capture_dispatch([provider](const auto& operation) { provider->execute([&](auto&) { operation(); }); }, tiles);
	scaled_image_mapping_epoch::mappings values{};
	values[0] = {0, 16, {}}; values[1] = {64, 16, {}};
	values[0].tiling = scaled_image_memory_mapping::tile_range{0x1fffc0, 128, 1};
	values[1].tiling = scaled_image_memory_mapping::tile_range{0x9fffc0, 128, 1};
	const auto ticket = [&]
	{
		std::optional<scaled_image_mapping_epoch> result;
		checked_cpu_capture(dispatch, [&]
		{
			result.emplace([values] { return values; }, [tiles] { return scaled_image_mapping_versions{9000, 8000, tiles->current()}; },
				[tiles](const auto& owned) { return scaled_image_mapping_versions{9000, 8000, scaled_image_tile_mapping_generation(owned, *tiles)}; });
		});
		return std::move(*result);
	};
	const auto absent = ticket();
	const tile_descriptor source{1, 0x200000, 65536, 256, 0, 0, 0, true};
	tiles->update(0, source); tiles->update(0, {});
	refused([&] { checked_cpu_capture(dispatch, [&] { absent.validate(); }); });
	const auto unrelated = ticket();
	tiles->update(1, {1, 0x500000, 65536, 256, 0, 0, 0, true});
	tiles->update(2, {0, 0x200000, 65536, 256, 0, 0, 0, true});
	checked_cpu_capture(dispatch, [&] { unrelated.validate(); });
	tiles->update(0, source);
	const auto bound = ticket();
	auto changed = source; changed.pitch *= 2;
	tiles->update(0, changed); tiles->update(0, source);
	refused([&] { checked_cpu_capture(dispatch, [&] { bound.validate(); }); });
	const auto destination = ticket();
	tiles->update(3, {1, 0xa00000, 65536, 256, 0, 0, 0, true}); tiles->update(3, {});
	refused([&] { checked_cpu_capture(dispatch, [&] { destination.validate(); }); });
	const auto fresh = ticket();
	const auto coverage = memory->watch_surface(guest_byte_coverage({64, 16}));
	const auto expected = memory->surface_revision(coverage);
	std::array<std::byte, 16> payload; payload.fill(std::byte{0x39});
	std::promise<void> started;
	std::atomic<bool> finished = false;
	std::jthread producer;
	checked_cpu_capture(dispatch, [&]
	{
		fresh.validate();
		producer = std::jthread([&] { started.set_value(); tiles->update(0, changed); finished = true; });
		started.get_future().wait();
		require(!finished, "Tile producer entered while final publication owns its mutex");
		require(memory->compare_write(coverage, expected, payload).has_value(), "Tile transaction failed nested provider CAS");
		require(!finished && provider->notifications == 1, "Tile mutex released before publication finished");
	});
	producer.join();
	require(finished && std::equal(payload.begin(), payload.end(), provider->bytes.begin() + 64), "Tile producer or committed payload lost");
	refused([&] { checked_cpu_capture(dispatch, [&] { fresh.validate(); }); });
	// A high tile counter must not mask changes in either other domain.
	std::uint64_t vm = 1, io = 1;
	const auto global = [&] { return scaled_image_mapping_versions{vm, io, 9000}; };
	const scaled_image_mapping_epoch before_vm([values] { return values; }, global);
	++vm; refused([&] { before_vm.validate(); });
	const scaled_image_mapping_epoch before_io([values] { return values; }, global);
	++io; refused([&] { before_io.validate(); });
	const auto frozen = tiles->snapshot();
	tiles->update(0, source);
	require(frozen[0] == changed, "Owned tile snapshot changed after producer mutation");
	std::weak_ptr<tile_state> weak;
	std::optional<scaled_image_mapping_epoch> owner;
	{
		auto keep = std::make_shared<tile_state>(); weak = keep;
		owner.emplace([values] { return values; }, [keep] { return scaled_image_mapping_versions{1, 1, keep->current()}; });
	}
	require(!weak.expired(), "Pending tile owner died early"); owner->validate(); owner.reset();
	require(weak.expired(), "Tile owner leaked after pending ticket retirement");
	std::function<void()> retained;
	const auto late = tile_capture_dispatch([&](const auto& operation) { retained = operation; }, tiles);
	unsigned calls = 0;
	refused([&] { checked_cpu_capture(late, [&] { ++calls; }); }); retained();
	require(calls == 0, "Retained tile provider executed closed capture");
	refused([&] { (void)tile_capture_dispatch({}, tiles); });
	refused([&] { (void)tile_capture_dispatch(dispatch, {}); });
}
} // namespace
int main()
{
	try
	{
		oracle(); transactions();
		auto tiles = std::make_shared<tile_state>();
		std::atomic<bool> failed = false;
		std::array<std::thread, 4> producers;
		for (unsigned i = 0; i < producers.size(); ++i)
			producers[i] = std::thread([&, i]
			{
				for (unsigned n = 1; n <= 10000; ++n)
				{
					const tile_descriptor value{1, (i + 1) << 16, 65536, n * 256, n, n, n, true};
					tiles->update(i, value);
					for (const auto& current : tiles->snapshot())
						if (current.bound && (current.pitch != current.base * 256 || current.bank != current.base || current.comp != current.base)) failed = true;
				}
			});
		for (auto& producer : producers) producer.join();
		require(!failed && tiles->current() == 40001, "Tile snapshots tore fields or lost concurrent mutations");
		for (const auto value : std::array<tile_descriptor, 4>{{{1, 1, 65536, 256, 0, 0, 0, true}, {1, 0, 0, 256, 0, 0, 0, true}, {2, 0, 65536, 256, 0, 0, 0, true}, {1, 0xffff0000, 131072, 256, 0, 0, 0, true}}})
		{
			auto invalid = std::make_unique<tile_state>(); invalid->update(0, value);
			refused([&] { (void)invalid->current(); }); invalid->update(0, {});
			refused([&] { (void)invalid->current(1, {0, 1}); });
		}
		auto exhausted = std::make_unique<tile_state>(std::array<tile_descriptor, 15>{}, std::numeric_limits<std::uint64_t>::max() - 1);
		exhausted->update(0, {1, 0, 65536, 256, 0, 0, 0, true});
		refused([&] { (void)exhausted->current(); });
		std::cout << "Native tile transactions:10000 independent old/new range-oracle updates/queries;1048576byte fixed table;40000 concurrent mutations/snapshots without torn fields;absent and bound source/last-target ABA refused independently of larger VM/IO stamps;unrelated range/location preserved;VM-before-tile exclusion retained through nested byte CAS,producer resumes afterwards;retained callback cannot execute closed capture;frozen snapshots/owner retirement/three independent domains checked;top/empty/invalid/exhaustion checked;no App/UI/guest VM initialized\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
