#pragma once
#include <array>
#include <cstdint>
namespace metal4::test
{
struct framebuffer_layout
{
	std::uint32_t width = 4, height = 4;
	std::array<std::uint32_t, 4> color_addresses{0x1000, 0x2000, 0x3000, 0x4000};
	std::array<std::uint32_t, 4> color_pitch{1, 1, 1, 1}, actual_color_pitch{256, 256, 256, 256};
	std::uint32_t zeta_address = 0x5000, zeta_pitch = 1, actual_zeta_pitch = 256;
	std::uint32_t target = 1, color_format = 8, depth_format = 2, aa_mode = 0, raster_type = 1;
	std::array<std::uint32_t, 2> aa_factors{1, 1};
};
} // namespace metal4::test
