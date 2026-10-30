#pragma once

#include "M4SurfaceLayout.h"

namespace metal4
{
[[nodiscard]] inline std::size_t surface_guest_bytes(surface_format format)
{
	if (const auto color = std::get_if<surface_color>(&format))
		return color_layout(*color).pixel_bytes;
	switch (std::get<surface_depth>(format))
	{
	case surface_depth::z16_uint:
	case surface_depth::z16_float:
		return 2;
	case surface_depth::z24s8_uint:
	case surface_depth::z24s8_float:
		return 4;
	}
	throw std::invalid_argument("Invalid RSX typeless depth format");
}
struct typeless_layout
{
	std::uint32_t source_width_as_destination;
	texture_buffer_layout source, destination;
};
// Reinterpret each complete BE guest row in destination-format texels before
// nearest scaling. This is a bit transfer, not color/depth numeric conversion.
[[nodiscard]] inline typeless_layout plan_typeless(surface_format source, surface_format destination,
												   std::size_t sw, std::size_t sh, std::size_t dw, std::size_t dh)
{
	const auto a = surface_guest_bytes(source), b = surface_guest_bytes(destination);
	const auto row = checked_texture_product(sw, a);
	if (!sw || !sh || !dw || !dh || row % b)
		throw std::invalid_argument("Typeless transfer requires nonempty regions and complete destination texels per source row");
	const auto width = row / b;
	if (width > std::numeric_limits<std::uint32_t>::max() || sh > std::numeric_limits<std::uint32_t>::max() ||
		dw > std::numeric_limits<std::uint32_t>::max() || dh > std::numeric_limits<std::uint32_t>::max())
		throw std::overflow_error("Typeless nearest extent exceeds uint32");
	return {static_cast<std::uint32_t>(width), transfer_layout({1, 1, a}, sw, sh, 1), transfer_layout({1, 1, b}, dw, dh, 1)};
}
} // namespace metal4
