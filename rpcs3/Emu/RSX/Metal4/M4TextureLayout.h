#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace metal4
{

struct texel_block
{
	std::size_t width;
	std::size_t height;
	std::size_t bytes;
};

struct texture_buffer_layout
{
	std::size_t bytes_per_row;
	std::size_t bytes_per_image;
	std::size_t allocation_size;
};

inline std::size_t checked_texture_product(std::size_t a, std::size_t b)
{
	if (b && a > std::numeric_limits<std::size_t>::max() / b)
	{
		throw std::overflow_error("Metal 4 texture transfer size overflow");
	}
	return a * b;
}

// Dimensions describe one mip/slice region. Guest RSX pitches and native Metal
// storage sizes must be converted explicitly before calling this function.
inline texture_buffer_layout transfer_layout(texel_block block, std::size_t width,
	std::size_t height, std::size_t depth, std::size_t row_pitch = 0, std::size_t image_pitch = 0)
{
	if (!block.width || !block.height || !block.bytes || !width || !height || !depth)
	{
		throw std::invalid_argument("Metal 4 texture transfer dimensions must be nonzero");
	}
	const auto columns = width / block.width + (width % block.width != 0);
	const auto rows = height / block.height + (height % block.height != 0);
	const auto minimum_row = checked_texture_product(columns, block.bytes);
	if (!row_pitch) row_pitch = minimum_row;
	if (row_pitch < minimum_row || row_pitch % block.bytes)
	{
		throw std::invalid_argument("Metal 4 texture row pitch is invalid");
	}
	const auto minimum_image = checked_texture_product(row_pitch, rows);
	if (!image_pitch) image_pitch = minimum_image;
	if (image_pitch < minimum_image || image_pitch % block.bytes)
	{
		throw std::invalid_argument("Metal 4 texture image pitch is invalid");
	}
	// Include row padding conservatively; omit unused trailing image padding.
	const auto preceding_images = checked_texture_product(image_pitch, depth - 1);
	if (minimum_image > std::numeric_limits<std::size_t>::max() - preceding_images)
	{
		throw std::overflow_error("Metal 4 texture transfer size overflow");
	}
	return {row_pitch, depth == 1 ? 0 : image_pitch, preceding_images + minimum_image};
}

}
