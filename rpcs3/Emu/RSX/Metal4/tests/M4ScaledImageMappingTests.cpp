#include "M4RSXScaledImage.h"
#include <iostream>
#include <source_location>

namespace
{
using namespace metal4;
void require(bool v, const std::source_location& where = std::source_location::current())
{
	if (!v)
		throw std::runtime_error("Scaled-image mapping capture differs at line " + std::to_string(where.line()));
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
	require(failed);
}
// Mirrors the fields consumed by the template. The concrete GSRenderer also
// instantiates the probe with actual GCM_context/GcmTileInfo in its object target.
struct tile_info
{
	unsigned location = 1, offset = 0x20000, size = 65536, pitch = 768, comp = 0, base = 0, bank = 2;
	bool bound = true;
};
struct fixture_context
{
	tile_info tile;
	unsigned translation = 0x100000;
	std::optional<scaled_image_tile_mapping> operator()(unsigned io, unsigned, unsigned, std::size_t) const
	{
		if (!tile.bound || tile.location != 1 || io < tile.offset || io - tile.offset >= tile.size)
			return {};
		return scaled_image_tile_mapping{tile.offset + translation, tile.size, tile.pitch, tile.bank, tile.location, tile.offset, tile.comp, tile.base};
	}
};
} // namespace
int main()
{
	try
	{
		fixture_context context;
		unsigned whole_checks = 0;
		const auto translate = [&](unsigned io, unsigned dma, std::size_t size)
		{
			require(dma == 7);
			if (size == 65536)
			{
				require(io == context.tile.offset);
				++whole_checks;
			}
			return io + context.translation;
		};
		const auto query = [&]
		{
			return scaled_image_mapping_epoch::mappings{
				probe_scaled_image_mapping(context, translate, 0x20004, 7, 768, 128),
				probe_scaled_image_mapping(context, translate, 0x20004, 7, 768, 128)};
		};
		const scaled_image_mapping_epoch epoch(query);
		require(epoch.original()[0].address == 0x120304 && epoch.original()[0].size == 128 && whole_checks == 2);
		require(epoch.original()[0].tile->address == 0x120000 && epoch.original()[0].tile->io_offset == 0x20000);
		epoch.validate();
		require(whole_checks == 4);
		const auto original = context;
		for (unsigned field = 0; field < 10; ++field)
		{
			context = original;
			switch (field)
			{
			case 0:
				context.translation += 65536;
				break;
			case 1:
				context.tile.bound = false;
				break;
			case 2:
				context.tile.location = 0;
				break;
			case 3:
				context.tile.offset += 256;
				break;
			case 4:
				context.tile.size *= 2;
				break;
			case 5:
				context.tile.pitch *= 2;
				break;
			case 6:
				context.tile.comp = 1;
				break;
			case 7:
				context.tile.base = 1;
				break;
			case 8:
				context.tile.bank = 3;
				break;
			case 9:
				context.tile.size = 0;
				break;
			}
			refused([&]
					{ epoch.validate(); });
		}
		context = original;
		context.tile.bound = false;
		const scaled_image_mapping_epoch linear(query);
		require(!linear.original()[0].tile);
		context.tile.bound = true;
		refused([&]
				{ linear.validate(); });
		const auto wrong_span = [&](unsigned io, unsigned, std::size_t size)
		{ return io + context.translation + unsigned(size > 1); };
		refused([&]
				{ (void)probe_scaled_image_mapping(context, wrong_span, 0x20004, 7, 768, 128); });
		refused([&]
				{ (void)probe_scaled_image_mapping(context, translate, 0x20004, 7, 0xffff'ffff, 128); });
		refused([&]
				{ (void)probe_scaled_image_mapping(context, translate, 0x20004, 7, 768, 0); });
		refused([&]
				{ (void)scaled_image_mapping_epoch({}); });
		context.tile.bound = false;
		std::vector<unsigned> boundaries;
		const auto contiguous = [&](unsigned io, unsigned, std::size_t)
		{
			if (io % 0x100000 == 0)
				boundaries.push_back(io);
			return io + context.translation;
		};
		(void)probe_scaled_image_mapping(context, contiguous, 0xfff00, 7, 0, 0x200300);
		require(boundaries == std::vector<unsigned>{0x100000, 0x200000, 0x300000});
		for (unsigned wrong : boundaries)
		{
			const auto discontinuous = [&](unsigned io, unsigned, std::size_t)
			{ return io + context.translation + unsigned(io == wrong) * 0x1000000; };
			const auto scattered = probe_scaled_image_mapping(context, discontinuous, 0xfff00, 7, 0, 0x200300);
			require(scattered.segments.size() >= 2);
			require(std::any_of(scattered.segments.begin(), scattered.segments.end(), [&](const auto& segment)
								{ return segment.address == wrong + context.translation + 0x1000000; }));
		}
		refused([&]
				{ (void)probe_scaled_image_mapping(context, contiguous, 0xfffffff0, 7, 0, 32); });
		const auto remapped_entry = [&](unsigned io, unsigned, std::size_t)
		{ return io + context.translation + unsigned(io >= 0x100000) * 0x1000000; };
		const auto advanced = probe_scaled_image_mapping(context, remapped_entry, 0xfff00, 7, 512, 128);
		require(advanced.address == 0x1200100 && advanced.segments[0].address == advanced.address);
		context.tile.bound = true;
		context.tile.size = 2 << 20;
		const auto later_tile_entry = probe_scaled_image_mapping(context, remapped_entry, 0x110000, 7, 0, 128);
		require(later_tile_entry.tile && later_tile_entry.tile->io_offset == 0x20000 && later_tile_entry.segments.size() == 2);
		const auto raw = [](unsigned, unsigned, unsigned, std::size_t) -> std::optional<scaled_image_tile_mapping> { return {}; };
		const auto identity = [](unsigned io, unsigned, std::size_t) { return io + 0x1000000; };
		const auto main = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_LOCATION_MAIN, 512, 0x100100);
		require(main.io && main.io->offset == 0x100100 && main.io->size == 0x100100);
		const auto host = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER, 0, 0x100300);
		require(host.io && host.io->offset == 0xfff00 && host.io->size == 0x100300);
		const auto local = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_LOCATION_LOCAL, 0, 0x100300);
		require(!local.io);
		const auto frame = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_CONTEXT_DMA_MEMORY_FRAME_BUFFER, 0, 0x100300);
		require(!frame.io);
		const auto host_tile = probe_scaled_image_mapping(context, contiguous, 0x110000, CELL_GCM_LOCATION_MAIN, 0, 128);
		require(host_tile.io && host_tile.io->offset == context.tile.offset && host_tile.io->size == context.tile.size);
		const auto tiled_main = probe_scaled_image_mapping(context, contiguous, 0x110000, CELL_GCM_LOCATION_MAIN, 0, 128, true);
		require(tiled_main.tiling && tiled_main.tiling->offset == context.tile.offset && tiled_main.tiling->size == context.tile.size && tiled_main.tiling->location == CELL_GCM_LOCATION_MAIN);
		const auto absent_tile = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER, 512, 128, true);
		require(absent_tile.tiling && absent_tile.tiling->offset == 0x100100 && absent_tile.tiling->size == 128);
		require(!main.tiling && !host.tiling && !probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_LOCATION_LOCAL, 0, 128, true).tiling);
		const auto main_report = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_MAIN, 0, 0x100300);
		require(main_report.io && main_report.io->offset == 0x0e0fff00 && main_report.io->size == 0x100300);
		const auto local_report = probe_scaled_image_mapping(raw, identity, 0xfff00, CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_LOCAL, 0, 128);
		require(!local_report.io);
		refused([&] { (void)probe_scaled_image_mapping(raw, identity, 0xfffff0, CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_MAIN, 0, 32); });
		std::cout << "Native image mapping:owned tile values,whole-allocation IO extent,three1MiB entries gathered with disjoint EA offsets retained;translation/bind/unbind/location/offset/size/pitch/bank/compression/base changes and invalid spans refused;main/host IO extent and full tile allocation retained,main report aperture rebased/bounded,local/framebuffer/local report have no IO dependency;eligible tile absence and full allocation retained,raw DMA/local have no tile dependency;real GCM types compile-only,no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
