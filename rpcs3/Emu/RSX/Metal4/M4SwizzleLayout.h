#pragma once

#include "M4TextureLayout.h"

#include <array>
#include <bit>
#include <cstdint>

namespace metal4
{
struct swizzle_region
{
	// Storage dimensions are in pixels/blocks, not bytes. Each dimension must
	// be a power of two. A mip uses its own dimensions and byte base address.
	std::array<std::size_t, 3> storage{1, 1, 1};
	std::array<std::size_t, 3> origin{};
	std::array<std::size_t, 3> size{1, 1, 1};
};

struct swizzle_plan
{
	std::array<std::uint32_t, 3> log2{};
	std::size_t storage_bytes;
	std::size_t block_count;
	texture_buffer_layout linear;
};

[[nodiscard]] inline swizzle_plan plan_swizzle(const swizzle_region& region, std::size_t block_bytes,
	std::size_t row_pitch = 0, std::size_t image_pitch = 0)
{
	if (!std::has_single_bit(block_bytes) || block_bytes > 16)
		throw std::invalid_argument("Native RSX Morton blocks must contain 1, 2, 4, 8 or 16 bytes");
	swizzle_plan plan{};
	std::size_t storage_blocks = 1, active_blocks = 1;
	for (std::size_t axis = 0; axis < 3; ++axis)
	{
		const auto dimension = region.storage[axis];
		if (!std::has_single_bit(dimension) || !region.size[axis] || region.origin[axis] > dimension ||
			region.size[axis] > dimension - region.origin[axis])
			throw std::invalid_argument("Native RSX Morton region exceeds power-of-two storage dimensions");
		plan.log2[axis] = static_cast<std::uint32_t>(std::bit_width(dimension) - 1);
		storage_blocks = checked_texture_product(storage_blocks, dimension);
		active_blocks = checked_texture_product(active_blocks, region.size[axis]);
	}
	plan.storage_bytes = checked_texture_product(storage_blocks, block_bytes);
	plan.block_count = active_blocks;
	plan.linear = transfer_layout({1, 1, block_bytes}, region.size[0], region.size[1], region.size[2], row_pitch, image_pitch);
	return plan;
}

// Compact rectangular Morton order: emit X, Y, Z bits in that order for each
// significance level, skipping an axis once all its dimension bits are used.
// This is the VK GPUDeswizzle/calculate_z_index contract; not tiled/banked RSX
// addressing. Calling with invalid log sums or coordinates throws before shifts.
[[nodiscard]] inline std::size_t morton_index(std::array<std::size_t, 3> position,
	const std::array<std::uint32_t, 3>& log2)
{
	std::uint32_t total = 0;
	for (std::size_t axis = 0; axis < 3; ++axis)
	{
		if (log2[axis] >= std::numeric_limits<std::size_t>::digits ||
			position[axis] >= (std::size_t{1} << log2[axis]) ||
			log2[axis] > std::numeric_limits<std::size_t>::digits - 1 - total)
			throw std::invalid_argument("RSX Morton coordinate or dimension bit count is invalid");
		total += log2[axis];
	}
	std::size_t result = 0;
	std::uint32_t output_bit = 0;
	for (std::uint32_t input_bit = 0; output_bit < total; ++input_bit)
		for (std::size_t axis = 0; axis < 3; ++axis)
			if (input_bit < log2[axis]) result |= ((position[axis] >> input_bit) & 1) << output_bit++;
	return result;
}
}
