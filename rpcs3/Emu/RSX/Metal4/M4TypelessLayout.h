#pragma once

#include "M4SurfaceLayout.h"
#include "M4NearestLayout.h"

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
struct typeless_source_window
{
	std::array<std::size_t,2> origin,extent;
	nearest_projection projection;
};
// Align both native source pixels and reinterpreted destination words. The
// codecs pack this enclosing box; the nearest kernel retains the full raster
// phase and skips its at-most-one-pixel boundary padding on either side.
[[nodiscard]] inline typeless_source_window plan_typeless_source_window(surface_format source,surface_format destination,
	std::size_t sw,std::size_t sh,std::size_t dw,std::size_t dh,std::array<bool,3> flip={},nearest_projection projection={})
{
	const auto layout=plan_typeless(source,destination,sw,sh,dw,dh);
	const auto a=surface_guest_bytes(source),b=surface_guest_bytes(destination),alignment=std::max(a,b);
	const auto window=plan_nearest_input_window({layout.source_width_as_destination,static_cast<std::uint32_t>(sh),1},
		{static_cast<std::uint32_t>(dw),static_cast<std::uint32_t>(dh),1},flip,projection);
	const auto begin=std::size_t(window.origin[0])*b/alignment*alignment;
	const auto end=(std::size_t(window.origin[0]+window.extent[0])*b+alignment-1)/alignment*alignment;
	projection=window.projection;
	if ((std::size_t(projection.source_origin[0])*b)%a || projection.source_origin[2] || flip[2])
		throw std::invalid_argument("Typeless cropped source origin is not a complete native 2D pixel");
	projection.source_origin[0]+=static_cast<std::uint32_t>(begin/b);
	projection.source_origin[1]+=window.origin[1];
	return {{begin/a,window.origin[1]},{(end-begin)/a,window.extent[1]},projection};
}
} // namespace metal4
