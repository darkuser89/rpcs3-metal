#pragma once

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace metal4
{

inline void validate_depth_bounds(float minimum, float maximum)
{
	if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0 || maximum > 1 || minimum > maximum)
	{
		throw std::invalid_argument("Metal 4 stored-depth bounds must be ordered and within [0, 1]");
	}
}

// MSL vertex_id and instance_id are uint, even though Metal's CPU API uses
// NSUInteger. The last valid ID is UINT32_MAX, not UINT32_MAX - 1.
inline void validate_draw_ids(std::size_t first, std::size_t count)
{
	constexpr auto max_id = std::numeric_limits<std::uint32_t>::max();
	if (first > max_id || count > max_id || (count && count - 1 > max_id - first))
	{
		throw std::out_of_range("Metal 4 draw IDs exceed the MSL uint range");
	}
}

inline std::size_t checked_index_bytes(std::size_t allocation, std::size_t offset,
	std::size_t count, std::size_t index_bytes)
{
	if ((index_bytes != 2 && index_bytes != 4) || offset % index_bytes)
	{
		throw std::invalid_argument("Metal 4 index buffer alignment or type is invalid");
	}
	if (offset > allocation || count > (allocation - offset) / index_bytes)
	{
		throw std::out_of_range("Metal 4 indices exceed the buffer allocation");
	}
	return count * index_bytes;
}

inline void validate_scissor(std::size_t width, std::size_t height,
	std::size_t x, std::size_t y, std::size_t scissor_width, std::size_t scissor_height)
{
	if (!width || !height || x > width || y > height ||
		scissor_width > width - x || scissor_height > height - y)
	{
		throw std::out_of_range("Metal 4 scissor exceeds the render target");
	}
}

inline void validate_visibility_offset(std::size_t allocation, std::size_t offset)
{
	if (offset % 8)
	{
		throw std::invalid_argument("Metal 4 visibility offsets require eight-byte alignment");
	}
	if (offset > allocation || 8 > allocation - offset)
	{
		throw std::out_of_range("Metal 4 visibility result exceeds its buffer");
	}
}

}
