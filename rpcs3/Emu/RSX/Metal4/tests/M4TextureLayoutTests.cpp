#include "M4TextureLayout.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace
{

void require(bool condition)
{
	if (!condition) throw std::runtime_error("Texture layout invariant failed");
}

template <typename Exception, typename Function>
void rejects(Function function)
{
	try { function(); }
	catch (const Exception&) { return; }
	throw std::runtime_error("Invalid texture layout was accepted");
}

}

int main()
{
	using namespace metal4;
	// Check padding and partial BC blocks against an independent, enumerated
	// address model. This never creates a Metal device or starts the emulator.
	std::uint32_t random = 0x4d534c34;
	auto next = [&random](std::size_t bound)
	{
		random = random * 1664525u + 1013904223u;
		return static_cast<std::size_t>(random) % bound;
	};
	for (std::size_t trial = 0; trial < 3000; ++trial)
	{
		const std::size_t block_dimension = next(2) ? 4 : 1;
		const texel_block block{block_dimension, block_dimension, std::size_t{1} << next(5)};
		const auto width = next(128) + 1;
		const auto height = next(96) + 1;
		const auto depth = next(9) + 1;
		std::size_t columns = 0;
		for (std::size_t x = 0; x < width; x += block.width) ++columns;
		std::size_t rows = 0;
		for (std::size_t y = 0; y < height; y += block.height) ++rows;
		const auto row_pitch = (columns + next(8)) * block.bytes;
		const auto image_pitch = row_pitch * rows + next(8) * block.bytes;
		const auto layout = transfer_layout(block, width, height, depth, row_pitch, image_pitch);
		require(layout.bytes_per_row == row_pitch);
		require(layout.bytes_per_image == (depth == 1 ? 0 : image_pitch));
		for (std::size_t z = 0; z < depth; ++z)
		{
			for (std::size_t row = 0; row < rows; ++row)
			{
				const auto end = z * image_pitch + row * row_pitch + columns * block.bytes;
				require(end <= layout.allocation_size);
			}
		}
		require(layout.allocation_size == (depth - 1) * image_pitch + rows * row_pitch);
	}
	const auto bc = transfer_layout({4, 4, 16}, 5, 7, 1);
	require(bc.bytes_per_row == 32 && bc.allocation_size == 64 && bc.bytes_per_image == 0);
	const auto limit = std::numeric_limits<std::size_t>::max();
	require(transfer_layout({1, 1, 1}, limit, 1, 1).allocation_size == limit);
	rejects<std::invalid_argument>([] { transfer_layout({4, 4, 16}, 5, 7, 1, 16); });
	rejects<std::invalid_argument>([] { transfer_layout({1, 1, 4}, 1, 1, 1, 5); });
	rejects<std::invalid_argument>([] { transfer_layout({1, 1, 4}, 2, 2, 2, 8, 8); });
	rejects<std::invalid_argument>([] { transfer_layout({0, 1, 4}, 1, 1, 1); });
	rejects<std::invalid_argument>([] { transfer_layout({1, 1, 4}, 0, 1, 1); });
	rejects<std::overflow_error>([] { transfer_layout({1, 1, 4}, limit, 1, 1); });
	rejects<std::overflow_error>([] { transfer_layout({1, 1, 1}, limit, 2, 1); });
	rejects<std::overflow_error>([] { transfer_layout({1, 1, 1}, 1, 1, 3, 1, limit); });
	rejects<std::overflow_error>([] { transfer_layout({1, 1, 1}, 1, 1, 2, 1, limit); });
	std::cout << "Metal 4 texture layout: 3000 address models and boundary checks passed\n";
}
