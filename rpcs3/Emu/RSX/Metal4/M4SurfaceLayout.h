#pragma once

#include "M4TextureLayout.h"

#include <cstdint>
#include <variant>

namespace metal4
{

enum class surface_color : std::uint32_t
{
	x1rgb5_zero = 1,
	x1rgb5_one = 2,
	rgb565 = 3,
	xrgb8_zero = 4,
	xrgb8_one = 5,
	argb8 = 8,
	b8 = 9,
	gb8 = 10,
	rgba16f = 11,
	rgba32f = 12,
	r32f = 13,
	xbgr8_zero = 14,
	xbgr8_one = 15,
	abgr8 = 16
};
enum class surface_depth : std::uint32_t
{
	z16_uint = 1,
	z24s8_uint = 2,
	z16_float = 3,
	z24s8_float = 4
};
using surface_format = std::variant<surface_color, surface_depth>;

enum class color_export_remap : std::uint32_t
{
	rgba,
	blue,
	blue_green,
	rgb_zero,
	rgb_one
};

struct surface_color_layout
{
	std::uint32_t pixel_bytes;
	std::uint32_t endian_element_bytes;
	color_export_remap export_remap;
};

[[nodiscard]] inline surface_color_layout color_layout(surface_color format)
{
	switch (format)
	{
	case surface_color::x1rgb5_zero:
		return {2, 2, color_export_remap::rgb_zero};
	case surface_color::x1rgb5_one:
		return {2, 2, color_export_remap::rgb_one};
	case surface_color::rgb565:
		return {2, 2, color_export_remap::rgba};
	case surface_color::xrgb8_zero:
	case surface_color::xbgr8_zero:
		return {4, 4, color_export_remap::rgb_zero};
	case surface_color::xrgb8_one:
	case surface_color::xbgr8_one:
		return {4, 4, color_export_remap::rgb_one};
	case surface_color::argb8:
	case surface_color::abgr8:
		return {4, 4, color_export_remap::rgba};
	case surface_color::b8:
		return {1, 1, color_export_remap::blue};
	case surface_color::gb8:
		return {2, 2, color_export_remap::blue_green};
	// WZYX notation specifies sequential arbitrary data, not reversed RGBA.
	case surface_color::rgba16f:
		return {8, 2, color_export_remap::rgba};
	case surface_color::rgba32f:
		return {16, 4, color_export_remap::rgba};
	case surface_color::r32f:
		return {4, 4, color_export_remap::rgba};
	}
	throw std::invalid_argument("Invalid RSX surface color format");
}

// Input mask: R=1/G=2/B=4/A=8. Returns native logical channel mask after RSX
// format remapping, matching the VK/shared color-utils contract.
[[nodiscard]] inline std::uint32_t color_write_mask(surface_color format, std::uint32_t mask)
{
	if (mask & ~15u)
		throw std::invalid_argument("Invalid RSX surface color write mask");
	(void)color_layout(format);
	switch (format)
	{
	case surface_color::rgb565:
		return mask & 7;
	case surface_color::b8:
		return (mask & 4) ? 1u : 0u;
	case surface_color::gb8:
		return ((mask & 4) ? 1u : 0u) | (mask & 2);
	case surface_color::r32f:
		return mask & 1;
	default:
		return mask;
	}
}

struct surface_byte_region
{
	std::size_t offset = 0;
	std::size_t row_pitch = 0;
};

// This conversion supports pitched single-sample linear color surfaces.
// Swizzled storage/MSAA expansion are separate renderer contracts.
[[nodiscard]] inline texture_buffer_layout surface_byte_layout(surface_color format,
															   std::size_t width, std::size_t height, const surface_byte_region& region)
{
	const auto layout = color_layout(format);
	const auto result = transfer_layout({1, 1, layout.pixel_bytes}, width, height, 1, region.row_pitch);
	if (region.offset > std::numeric_limits<std::size_t>::max() - result.allocation_size)
		throw std::overflow_error("RSX surface byte offset overflow");
	return result;
}

} // namespace metal4
