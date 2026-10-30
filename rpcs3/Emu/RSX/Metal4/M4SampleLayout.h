#pragma once

#include "M4TextureLayout.h"

#include <array>
#include <cstdint>

namespace metal4
{
struct sample_layout
{
	std::uint32_t x, y;
};
[[nodiscard]] inline sample_layout rsx_sample_layout(std::uint32_t samples)
{
	switch (samples)
	{
	case 2:
		return {2, 1};
	case 4:
		return {2, 2};
	default:
		throw std::invalid_argument("RSX sample expansion supports only 2 or 4 samples");
	}
}
[[nodiscard]] inline std::array<std::size_t, 2> sample_expanded_size(std::size_t width, std::size_t height, std::uint32_t samples)
{
	if (!width || !height)
		throw std::invalid_argument("RSX MSAA dimensions must be nonzero");
	const auto layout = rsx_sample_layout(samples);
	return {checked_texture_product(width, layout.x), checked_texture_product(height, layout.y)};
}
[[nodiscard]] inline std::array<std::size_t, 2> expanded_sample_coordinate(std::size_t x, std::size_t y,
																		   std::uint32_t sample, std::uint32_t samples)
{
	const auto layout = rsx_sample_layout(samples);
	if (sample >= samples)
		throw std::out_of_range("RSX sample index exceeds sample count");
	const auto px = checked_texture_product(x, layout.x), py = checked_texture_product(y, layout.y);
	if (px > std::numeric_limits<std::size_t>::max() - sample % layout.x ||
		py > std::numeric_limits<std::size_t>::max() - sample / layout.x)
		throw std::overflow_error("RSX expanded sample coordinate overflow");
	return {px + sample % layout.x, py + sample / layout.x};
}
} // namespace metal4
