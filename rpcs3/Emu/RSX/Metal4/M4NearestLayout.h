#pragma once

#include "M4TextureLayout.h"

#include <array>
#include <algorithm>
#include <cstdint>

namespace metal4
{
struct nearest_image
{
	std::array<std::uint32_t, 3> size;
	std::size_t row_pitch = 0, image_pitch = 0;
};
struct nearest_projection
{
	std::array<std::uint32_t, 3> extent{}, offset{};
	// Logical source raster and the origin of the supplied physical window.
	// Zero source_extent selects the complete supplied image.
	std::array<std::uint32_t, 3> source_extent{}, source_origin{};
};
struct nearest_input_window
{
	std::array<std::uint32_t,3> origin{}, extent{};
	nearest_projection projection;
};
// Smallest enclosing physical source box for a destination window. Keep the
// original rational mapping; cropping must not restart its phase or flip.
[[nodiscard]] inline nearest_input_window plan_nearest_input_window(std::array<std::uint32_t,3> source,
	std::array<std::uint32_t,3> destination, std::array<bool,3> flip, nearest_projection projection = {})
{
	if (projection.extent == std::array<std::uint32_t,3>{}) projection.extent=destination;
	if (projection.source_extent == std::array<std::uint32_t,3>{}) projection.source_extent=source;
	nearest_input_window window{{},{},projection};
	for (unsigned axis=0;axis<3;++axis)
	{
		const auto full=projection.source_extent[axis], copy=projection.extent[axis], start=projection.offset[axis];
		const auto base=projection.source_origin[axis];
		if (!full || !copy || !source[axis] || !destination[axis] || start>copy || destination[axis]>copy-start ||
			base>full || source[axis]>full-base)
			throw std::invalid_argument("Native nearest window exceeds its logical source or copy extent");
		const auto coordinate=[&](std::uint32_t index)
		{
			const auto numerator=std::uint64_t(index)*full+full/2;
			const auto quotient=static_cast<std::uint32_t>(numerator/copy);
			return flip[axis]?full-quotient-unsigned(numerator%copy!=0 || (full&1)):quotient;
		};
		const auto a=coordinate(start),b=coordinate(start+destination[axis]-1);
		const auto low=std::min(a,b),high=std::max(a,b);
		if (low<base || high-base>=source[axis])
			throw std::invalid_argument("Native nearest physical source window omits a selected texel");
		window.origin[axis]=low-base;window.extent[axis]=high-low+1;
	}
	return window;
}
struct nearest_layout
{
	texture_buffer_layout source, destination;
	std::size_t texels;
};
[[nodiscard]] inline nearest_layout plan_nearest(nearest_image source, nearest_image destination, std::uint32_t bytes)
{
	if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8 && bytes != 16)
		throw std::invalid_argument("Native nearest blit needs a complete 1/2/4/8/16-byte texel");
	const auto layout = [bytes](nearest_image image)
	{
		return transfer_layout({1, 1, bytes}, image.size[0], image.size[1], image.size[2], image.row_pitch, image.image_pitch);
	};
	return {layout(source), layout(destination), checked_texture_product(checked_texture_product(destination.size[0], destination.size[1]), destination.size[2])};
}
} // namespace metal4
