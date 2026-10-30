#include "M4IOMappingEpoch.h"
#include "M4ScaledImageMapping.h"
#include "M4TextureMapping.h"
#include <iostream>
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
	try { operation(); }
	catch (const std::exception&) { failed = true; }
	require(failed, "IO mapping epoch failed to refuse");
}
void texture_segments()
{
	unsigned probes = 0;
	const auto identity = [&](std::uint32_t offset, unsigned location, std::size_t size)
	{
		++probes;
		require(location == 0 || size <= 0x100000 - (offset & 0xfffff), "Main texture crossed a logical IO probe boundary");
		return offset;
	};
	require(texture_storage_segments(0xffff0, 1, 0x200030, identity) == std::vector<guest_storage_segment>{{0xffff0, 0, 0x200030}} && probes == 4,
		"Contiguous texture IO did not coalesce physical spans");
	probes = 0;
	require(texture_storage_segments(0xffff0, 0, 0x200030, identity).size() == 1 && probes == 1, "Local texture was split into IO entries");
	const auto top = texture_storage_segments(0xffff0, 1, 64, [](auto offset, auto, auto) { return offset < 0x100000 ? 0xfffffff0u : 0u; });
	require(top == std::vector<guest_storage_segment>{{0xfffffff0, 0, 16}, {0, 16, 48}}, "Disjoint top EA acquired a fictitious wrapped envelope");
	const auto duplicates = texture_storage_segments(0, 1, 0x300000, [](auto, auto, auto) { return 0x800000u; });
	require(duplicates == std::vector<guest_storage_segment>{{0x800000, 0, 0x100000}, {0x800000, 0x100000, 0x100000}, {0x800000, 0x200000, 0x100000}},
		"Repeated physical IO spans lost logical order");
	const auto full = texture_storage_segments(0, 1, 0x1'0000'0000ull, [](auto, auto, auto) { return 0u; });
	require(full.size() == 4096 && full.back().offset == 0xfff00000ull && full.back().size == 0x100000, "Full IO table lost its last texture span");
	refused([&] { (void)texture_storage_segments(0, 2, 1, identity); });
	refused([&] { (void)texture_storage_segments(0, 1, 0, identity); });
	refused([&] { (void)texture_storage_segments(0xffffffff, 1, 2, identity); });
	refused([] { (void)texture_storage_segments(0, 0, 32, [](auto, auto, auto) { return 0xfffffff0u; }); });
	refused([] { (void)texture_storage_segments(0, 1, 1, {}); });
	std::uint64_t random = 0x735167;
	const auto next = [&] { random = random * 6364136223846793005ull + 1; return random; };
	for (unsigned trial = 0; trial < 512; ++trial)
	{
		std::array<std::uint32_t, 16> table{};
		for (auto& entry : table) entry = std::uint32_t(next() % 32) << 20;
		const auto offset = std::uint32_t(next() % (12 << 20));
		const auto size = std::size_t(1 + next() % (3 << 20));
		const auto parts = texture_storage_segments(offset, 1, size, [&](auto logical, auto, auto bytes)
		{
			require(bytes <= 0x100000 - (logical & 0xfffff), "Random texture crossed IO entry");
			return table[logical >> 20] + (logical & 0xfffff);
		});
		std::size_t covered = 0;
		for (const auto& part : parts)
		{
			require(part.offset == covered && part.size, "Random texture has a logical gap");
			for (std::size_t sample = 0; sample < part.size; sample += std::min<std::size_t>(4093, part.size - sample))
			{
				const auto logical = offset + part.offset + sample;
				require(std::uint64_t(part.address) + sample == table[logical >> 20] + (logical & 0xfffff), "Random texture differs from per-byte IO table oracle");
			}
			const auto last = offset + part.offset + part.size - 1;
			require(std::uint64_t(part.address) + part.size - 1 == table[last >> 20] + (last & 0xfffff), "Random texture last byte differs from IO oracle");
			covered += part.size;
		}
		require(covered == size, "Random texture lost storage");
	}
	std::cout << "Native texture segments:512 randomized byte-oracle mappings,coalescing/local/duplicate ordering,4096-entry full4GiB,top-EA disjoint spans and invalid ranges passed\n";
}
void oracle()
{
	io_mapping_epoch epochs;
	std::array<std::uint64_t, 4096> entries{};
	std::uint64_t random = 0x416318;
	const auto next = [&]
	{
		random = random * 6364136223846793005ull + 1442695040888963407ull;
		return random;
	};
	require(io_mapping_epoch::table_bytes == 32768, "IO table exceeded its fixed bound");
	for (unsigned i = 0; i < 10000; ++i)
	{
		const auto first = unsigned(next() % entries.size());
		const auto count = std::min<unsigned>(1 + next() % 16, entries.size() - first);
		epochs.notify({first << 20, std::size_t(count) << 20});
		std::fill_n(entries.begin() + first, count, epochs.current());
		const auto address = std::uint32_t(next() >> 16);
		const auto bytes = std::size_t(std::min<std::uint64_t>(next() % (32 << 20), 0x1'0000'0000ull - address));
		std::uint64_t expected = 0;
		if (bytes)
			for (auto entry = address >> 20; entry < (std::uint64_t(address) + bytes + 0xfffff) / 0x100000; ++entry)
				expected = std::max(expected, entries[entry]);
		require(epochs.current({address, bytes}) == expected, "IO range differs from independent entry oracle");
	}
	epochs.notify({0, 0x1'0000'0000ull});
	require(epochs.current({0xffffffff, 1}) == epochs.current() && epochs.current({0, 0x1'0000'0000ull}) == epochs.current(), "Full4GiB/top IO entry was lost");
	const auto stamp = epochs.current();
	epochs.notify({0xffffffff, 0});
	require(epochs.current() == stamp && epochs.current({0xffffffff, 0}) == 0, "Empty IO event/query invented a revision");
	refused([&] { (void)epochs.current({0xffffffff, 2}); });
	require(epochs.current() == stamp, "Invalid IO query poisoned valid producer state");
}
} // namespace
int main()
{
	try
	{
		texture_segments();
		oracle();
		io_mapping_epoch concurrent;
		std::array<std::thread, 4> workers;
		for (auto& worker : workers)
			worker = std::thread([&]
			{
				for (unsigned i = 0; i < 10000; ++i)
					concurrent.notify({0x300000, 0x100000});
			});
		for (auto& worker : workers) worker.join();
		require(concurrent.current() == 40001 && concurrent.current({0x3fffff, 1}) == 40001 && concurrent.current({0x400000, 1}) == 0, "Concurrent IO stamps lost/regressed or touched unrelated entry");
		io_mapping_epoch exhausted(std::numeric_limits<std::uint64_t>::max() - 1);
		exhausted.notify({0, 0x100000});
		refused([&] { (void)exhausted.current(); });
		exhausted.notify({0, 0x100000});
		refused([&] { (void)exhausted.current({0, 1}); });
		for (const auto range : std::array<guest_surface_range, 3>{{{1, 0x100000}, {0, 1}, {0xfff00000, 0x200000}}})
		{
			io_mapping_epoch invalid;
			invalid.notify(range);
			refused([&] { (void)invalid.current(); });
			invalid.notify({0, 0x100000});
			refused([&] { (void)invalid.current({0, 1}); });
		}
		refused([] { io_mapping_epoch invalid(0); });
		refused([] { io_mapping_epoch invalid(std::numeric_limits<std::uint64_t>::max()); });
		auto io = std::make_shared<io_mapping_epoch>();
		scaled_image_mapping_epoch::mappings values{};
		values[0] = {0x1ffc00, 0x100800, {}, {{0x1ffc00, 0, 1024}, {0x400000, 1024, 0x100000}, {0x600000, 0x100400, 1024}}};
		values[0].io = scaled_image_memory_mapping::io_range{0x1ffc00, 0x100800};
		values[1] = {0x9ffc00, 0x100800, {}, {{0x9ffc00, 0, 1024}, {0xc00000, 1024, 0x100000}, {0xe00000, 0x100400, 1024}}};
		values[1].io = scaled_image_memory_mapping::io_range{0x9ffc00, 0x100800};
		std::uint64_t vm_global = 9000, vm_range = 9000;
		const auto global = [&] { return scaled_image_mapping_versions{vm_global, io->current()}; };
		const auto ranges = [&](const auto& owned)
		{
			return scaled_image_mapping_versions{vm_range, scaled_image_io_mapping_generation(owned, *io)};
		};
		const auto query = [&] { return values; };
		const scaled_image_mapping_epoch before(query, global, ranges);
		const auto identical = values;
		values[0].segments.back().address = 0x800000;
		io->notify({0x300000, 0x100000});
		values = identical;
		io->notify({0x300000, 0x100000});
		require(values == before.original() && vm_range > io->current(), "IO ABA fixture failed to restore metadata or exercise unequal domains");
		refused([&] { before.validate(); });
		const scaled_image_mapping_epoch fresh(query, global, ranges);
		io->notify({0x500000, 0x100000});
		++vm_global; // Unrelated VM mapping does not change the captured range.
		fresh.validate();
		io->notify({0xb00000, 0x100000});
		refused([&] { fresh.validate(); });
		vm_global = vm_range = 1; // Now the IO range counter is larger than VM.
		const scaled_image_mapping_epoch other_domain(query, global, ranges);
		++vm_global;
		++vm_range;
		refused([&] { other_domain.validate(); });
		values[0].io.reset(); values[1].io.reset();
		const scaled_image_mapping_epoch local(query, global, ranges);
		io->notify({0, 0x10000000});
		local.validate();
		values[0].io = scaled_image_memory_mapping::io_range{0x20000, 0x200000};
		const scaled_image_mapping_epoch tile(query, global, ranges);
		io->notify({0x200000, 0x100000});
		refused([&] { tile.validate(); });
		refused([&]
		{
			scaled_image_mapping_epoch changed([&]
			{
				io->notify({0x100000, 0x100000});
				return values;
			}, global, ranges);
		});
		bool change = false;
		const scaled_image_mapping_epoch after_query([&]
		{
			if (change) io->notify({0x100000, 0x100000});
			return values;
		}, global, ranges);
		change = true;
		refused([&] { after_query.validate(); });
		std::weak_ptr<io_mapping_epoch> weak;
		std::optional<scaled_image_mapping_epoch> owned;
		{
			auto keep = std::make_shared<io_mapping_epoch>(); weak = keep;
			owned.emplace([values] { return values; }, [keep] { return scaled_image_mapping_versions{1, keep->current()}; },
				[keep](const auto& capture) { return scaled_image_mapping_versions{1, scaled_image_io_mapping_generation(capture, *keep)}; });
		}
		require(!weak.expired(), "Owned IO generation died before pending ticket");
		owned->validate(); owned.reset();
		require(weak.expired(), "IO generation owner leaked after ticket retirement");
		// Texture dependencies include the complete allocation,not just its first
		// IO entry. Local textures do not depend on unrelated IO transitions.
		auto texture_io = std::make_shared<io_mapping_epoch>();
		const texture_io_revision texture(texture_io, 1, {0xfff00, 0x100300});
		const texture_io_revision local_texture(texture_io, 0, {0xfff00, 0x100300});
		texture_io->notify({0x300000, 0x100000});
		require(texture.current() && local_texture.current(), "Unrelated IO invalidated texture allocation");
		texture_io->notify({0x200000, 0x100000}); texture_io->notify({0x200000, 0x100000});
		require(!texture.current() && local_texture.current(), "Texture last-entry IO ABA was lost or leaked to local texture");
		const texture_io_revision fresh_texture(texture_io, 1, {0xfff00, 0x100300});
		require(fresh_texture.current(), "Fresh texture IO ticket is stale");
		texture_io->notify({1, 1});
		require(local_texture.current(), "Unrelated IO poison invalidated local-only texture");
		refused([&] { (void)fresh_texture.current(); });
		refused([&] { texture_io_revision invalid({}, 2, {0, 1}); });
		refused([&] { texture_io_revision invalid({}, 1, {0xffffffff, 2}); });
		refused([] { texture_io_revision invalid(std::make_shared<io_mapping_epoch>(), 1, {0, 0}); });
		std::weak_ptr<io_mapping_epoch> texture_weak;
		std::optional<texture_io_revision> texture_owner;
		{
			auto keep = std::make_shared<io_mapping_epoch>(); texture_weak = keep;
			texture_owner.emplace(keep, 1, guest_surface_range{0xffffffff, 1});
		}
		require(!texture_weak.expired() && texture_owner->current(), "Texture IO owner died before ticket");
		texture_owner.reset(); require(texture_weak.expired(), "Texture IO owner leaked after ticket");
		std::cout << "Native texture IO ticket:full allocation last-entry ABA/fresh retry,unrelated IO/local independence,poison/invalid/empty/top and owned retirement passed\n";
		std::cout << "Native IO mapping:40000 concurrent notifications;10000 independent entry-oracle updates/queries,32768byte fixed table,full4GiB/top/empty/invalid/alignment/exhaustion without wrap;identical IO ABA and last target entry refused,unrelated IO/VM and local memory preserved;separate VM/IO domains cannot mask either counter,tile allocation IO extent/query-time changes/owned lifetime checked;no App/UI/guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n'; return 1;
	}
}
