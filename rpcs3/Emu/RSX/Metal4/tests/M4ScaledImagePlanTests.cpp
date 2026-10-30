#include "M4RSXScaledImage.h"
#include <iostream>
#include <limits>

namespace
{
void require(bool v)
{
	if (!v)
		throw std::runtime_error("Native scaled-image decoded plan differs");
}
template <typename F>
void refused(F&& f)
{
	bool fail = false;
	try
	{
		f();
	}
	catch (const std::exception&)
	{
		fail = true;
	}
	require(fail);
}
} // namespace
int main()
{
	try
	{
		rsx::blit_src_info source{};
		source.format = rsx::blit_engine::transfer_source_format::a8r8g8b8;
		source.origin = rsx::blit_engine::transfer_origin::center;
		source.width = 8;
		source.height = 4;
		source.pitch = 48;
		source.bpp = 4;
		source.offset_x = 2;
		source.offset_y = 1;
		source.rsx_address = 0x1000;
		source.pixels = reinterpret_cast<u8*>(1);
		rsx::blit_dst_info target{};
		target.format = rsx::blit_engine::transfer_destination_format::r5g6b5;
		target.width = 16;
		target.height = 8;
		target.pitch = 48;
		target.bpp = 2;
		target.rsx_address = 0x4000;
		target.offset_x = 3;
		target.offset_y = 2;
		target.clip_x = 2;
		target.clip_y = 1;
		target.clip_width = 4;
		target.clip_height = 3;
		target.scale_x = target.scale_y = 2;
		target.pixels = reinterpret_cast<u8*>(2);
		auto plan = metal4::capture_scaled_image(source, target, 16, 8, true, 4096);
		require(plan.source.range.address == 0x1038 && plan.source.width == 6 && plan.source.height == 3 && plan.source.row_pitch == 48);
		require(plan.destination.range.address == 0x4066 && plan.destination.width == 4 && plan.destination.height == 3 && plan.destination.row_pitch == 48);
		require(plan.input_width == 8 && plan.scaled_width == 16 && plan.clip_x == 2 && plan.clip_y == 1 && plan.interpolate);
		source.offset_x = 7;
		source.offset_y = 3;
		target.scale_x = target.scale_y = -2;
		plan = metal4::capture_scaled_image(source, target, 16, 8, false, 4096);
		require(plan.source.range.address == 0x1000 && plan.source.width == 8 && plan.source.height == 4 && plan.flip_x && plan.flip_y);
		target.swizzled = true;
		plan = metal4::capture_scaled_image(source, target, 13, 7, false, 4096);
		require(plan.destination.morton && plan.destination.morton->storage == std::array<std::size_t, 3>{16, 8, 1} &&
				plan.destination.morton->origin == std::array<std::size_t, 3>{3, 2, 0} && plan.destination.range.address == 0x4000);
		const auto valid = plan;
		source.offset_x = source.width;
		source.offset_y = source.height;
		const auto edge = metal4::capture_scaled_image(source, target, 16, 8, false, 4096);
		require(edge.source.range.address == source.rsx_address);
		const auto saved_width = target.width;
		target.width = 15;
		refused([&]
				{ (void)metal4::capture_scaled_image(source, target, 16, 8, false, 4096); });
		target.width = saved_width;
		for (unsigned failure = 0; failure < 10; ++failure)
		{
			plan = valid;
			switch (failure)
			{
			case 0:
				plan.clip_x = std::numeric_limits<unsigned>::max();
				break;
			case 1:
				plan.scaled_width = 0;
				break;
			case 2:
				plan.source_format = static_cast<metal4::scaled_image_format>(9);
				break;
			case 3:
				plan.destination.pixel_bytes = 4;
				break;
			case 4:
				plan.input_width = 65537;
				break;
			case 5:
				plan.destination.width = 100;
				break;
			case 6:
				plan.source.row_pitch = 1;
				break;
			case 7:
				plan.source.range.address = 0xffff'ffff;
				break;
			case 8:
				plan.source.morton = metal4::swizzle_region{{8, 4, 2}, {0, 0, 0}, {8, 4, 1}};
				plan.source.range.size = 256;
				break;
			case 9:
				plan.destination.range.size = 0;
				break;
			}
			refused([&]
					{ metal4::validate_scaled_image(plan, 4096); });
		}
		refused([&]
				{ metal4::validate_scaled_image(valid, valid.source.range.size + valid.destination.range.size - 1); });
		for (float value : {0.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
		{
			target.scale_x = value;
			refused([&]
					{ (void)metal4::capture_scaled_image(source, target, 16, 8, false, 4096); });
		}
		const metal4::banked_tile_layout tile{0x10000, 0x40000, 768, 2, 772};
		const auto logical = metal4::plan_guest_surface(tile.address + tile.offset, 8, 4, 4, 48);
		const auto banked = metal4::attach_banked_tile(logical, tile);
		plan = valid;
		plan.source = banked;
		plan.source_tile = tile;
		metal4::validate_scaled_image(plan, 1 << 20);
		const auto coverage = metal4::scaled_image_coverage(banked, plan.source_tile);
		require(coverage.width == tile.size && coverage.height == 1 && coverage.pixel_bytes == 1 && coverage.range.address == tile.address);
		for (std::size_t bytes : {1u, 2u, 4u, 8u, 16u})
		{
			auto wide = tile; wide.offset = static_cast<std::uint32_t>(bytes);
			const auto image = metal4::attach_banked_tile(metal4::plan_guest_surface(wide.address + wide.offset, 8, 4, bytes, wide.pitch), wide);
			metal4::validate_banked_tile(image, wide);
		}
		for (std::size_t bytes : {0u, 3u, 32u})
		{
			auto image = banked; image.pixel_bytes = bytes;
			refused([&] { metal4::validate_banked_tile(image, tile); });
		}
		for (unsigned pitch : {256u, 512u, 768u, 1280u, 1536u, 1792u, 2816u, 3328u, 4096u, 6144u})
			require(metal4::banked_tile_factor(pitch) > 0);
		for (unsigned pitch : {0u, 1u, 255u, 257u, 2304u, 4352u})
			refused([&]
					{ (void)metal4::banked_tile_factor(pitch); });
		for (unsigned failure = 0; failure < 9; ++failure)
		{
			auto bad = tile;
			auto image = banked;
			switch (failure)
			{
			case 0:
				bad.address += 4;
				break;
			case 1:
				bad.size = 0;
				break;
			case 2:
				bad.size -= 1;
				break;
			case 3:
				bad.bank = 4;
				break;
			case 4:
				bad.offset = bad.size;
				break;
			case 5:
				bad.offset = bad.size - 4;
				break;
			case 6:
				image.width = bad.pitch / 4 + 1;
				break;
			case 7:
				image.height = 65537;
				break;
			case 8:
				image.range.address = 0xffff0000;
				break;
			}
			refused([&]
					{ metal4::validate_banked_tile(image, bad); });
		}
		refused([&]
				{ metal4::validate_scaled_image(plan, tile.size + plan.destination.range.size - 1); });
		std::cout << "Native banked plan:1/2/4/8/16-byte geometry with invalid widths refused;10 pitch factorizations,whole-allocation coverage,invalid factor/bank/alignment/range/row/height/budget refused\n";
		std::cout << "Native decoded scaled-image:owned metadata/poisoned guest pointers,offsets/reverse scan/Morton origin/clip/aggregate budget/invalid layout and scale passed;no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
