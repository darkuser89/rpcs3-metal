#pragma once

#include "M4SurfaceLayout.h"

#include <algorithm>
#include <array>

namespace metal4
{
struct clear_rectangle
{
	std::size_t x = 0, y = 0, width = 0, height = 0;
};
[[nodiscard]] inline clear_rectangle clip_clear_rectangle(clear_rectangle rectangle, std::size_t width, std::size_t height)
{
	rectangle.x = std::min(rectangle.x, width);
	rectangle.y = std::min(rectangle.y, height);
	rectangle.width = std::min(rectangle.width, width - rectangle.x);
	rectangle.height = std::min(rectangle.height, height - rectangle.y);
	return rectangle;
}

struct color_clear_plan
{
	std::array<float, 4> value;
	std::uint32_t write_mask;
};
// clear_color is raw A:R:G:B, clear_mask has RSX bits R=0x10..A=0x80.
// Clear masks aren't ordinary draw write masks: ABGR clear remaps R/B too.
[[nodiscard]] inline color_clear_plan decode_color_clear(surface_color format, std::uint32_t clear_color, std::uint32_t clear_mask)
{
	(void)color_layout(format);
	std::uint32_t mask = (clear_mask >> 4) & 15;
	const auto a = (clear_color >> 24) & 255, r = (clear_color >> 16) & 255, g = (clear_color >> 8) & 255, b = clear_color & 255;
	color_clear_plan result{{float(r) / 255, float(g) / 255, float(b) / 255, float(a) / 255}, mask};
	switch (format)
	{
	case surface_color::rgba16f:
	case surface_color::rgba32f:
	case surface_color::r32f:
		result.write_mask = 0;
		break; // The existing VK conventional framebuffer clear is NOP.
	case surface_color::rgb565:
		result.value = {float((clear_color >> 11) & 31) / 31, float((clear_color >> 5) & 63) / 63, float(clear_color & 31) / 31, 1};
		result.write_mask = mask & 7;
		break;
	case surface_color::x1rgb5_zero:
	case surface_color::x1rgb5_one:
		result.value = {float((clear_color >> 10) & 31) / 31, float((clear_color >> 5) & 31) / 31, float(clear_color & 31) / 31,
						format == surface_color::x1rgb5_one ? 1.f : 0.f};
		break;
	case surface_color::b8:
		result.value = {float(b) / 255, 0, 0, 1};
		result.write_mask = (mask & 4) ? 1u : 0u;
		break;
	case surface_color::gb8:
		result.value = {float(b) / 255, float(g) / 255, 0, 1};
		result.write_mask = ((mask & 4) ? 1u : 0u) | (mask & 2);
		break;
	case surface_color::abgr8:
	case surface_color::xbgr8_zero:
	case surface_color::xbgr8_one:
		std::swap(result.value[0], result.value[2]);
		result.write_mask = ((mask & 1) << 2) | (mask & 2) | ((mask & 4) >> 2) | (mask & 8);
		break;
	default:
		break;
	}
	return result;
}
} // namespace metal4
