#pragma once

#include "M4TextureLayout.h"

#include <array>
#include <cstdint>

namespace metal4
{
struct nearest_image
{
	std::array<std::uint32_t, 3> size;
	std::size_t row_pitch = 0, image_pitch = 0;
};
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
