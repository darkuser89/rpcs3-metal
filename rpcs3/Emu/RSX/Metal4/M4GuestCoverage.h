#pragma once

#include "M4GuestSurface.h"
#include <utility>

namespace metal4
{
// Geometry is validated by the cursor before using this payload size.
[[nodiscard]] inline std::size_t guest_surface_active_bytes(const guest_surface_layout& layout)
{
	return checked_texture_product(checked_texture_product(layout.width, layout.height), layout.pixel_bytes);
}
// A bounded-memory cursor over sorted, coalesced pixel-byte intervals. Linear
// row padding and Morton storage outside the active box are never included.
// Morton traversal skips disjoint blocks and emits fully contained blocks as
// one range; it doesn't allocate an interval entry per pixel or storage cell.
class guest_surface_byte_cursor final
{
	guest_surface_layout m_layout;
	struct node
	{
		std::array<std::size_t, 3> origin, extent;
		std::size_t first, pixels, level;
	};
	std::array<node, 65> m_stack{};
	std::array<unsigned char, 64> m_axes{};
	std::size_t m_top = 0, m_row = 0, m_column = 0;
	std::optional<guest_surface_range> m_saved;
	[[nodiscard]] std::optional<guest_surface_range> raw_next()
	{
		if (!m_layout.morton)
		{
			if (!m_layout.width || m_row == m_layout.height)
				return {};
			if (m_layout.column_stride)
			{
				const auto address = std::uint64_t(m_layout.range.address) + m_row * m_layout.row_pitch + m_column * m_layout.column_stride;
				if (++m_column == m_layout.width) { m_column = 0; ++m_row; }
				return guest_surface_range{static_cast<std::uint32_t>(address), 1};
			}
			if (m_layout.row_pitch == m_layout.width * m_layout.pixel_bytes)
			{
				m_row = m_layout.height;
				return m_layout.range;
			}
			const auto address = std::uint64_t(m_layout.range.address) + m_row++ * m_layout.row_pitch;
			return guest_surface_range{static_cast<std::uint32_t>(address), m_layout.width * m_layout.pixel_bytes};
		}
		const auto& active = *m_layout.morton;
		while (m_top)
		{
			const auto box = m_stack[--m_top];
			bool disjoint = false, contained = true;
			for (unsigned axis = 0; axis < 3; ++axis)
			{
				const auto end = box.origin[axis] + box.extent[axis], active_end = active.origin[axis] + active.size[axis];
				disjoint |= box.origin[axis] >= active_end || end <= active.origin[axis];
				contained &= box.origin[axis] >= active.origin[axis] && end <= active_end;
			}
			if (disjoint)
				continue;
			if (contained)
			{
				const auto address = std::uint64_t(m_layout.range.address) + box.first * m_layout.pixel_bytes;
				return guest_surface_range{static_cast<std::uint32_t>(address), box.pixels * m_layout.pixel_bytes};
			}
			if (!box.level)
				throw std::logic_error("Guest Morton coverage reached an indivisible partial block");
			const auto axis = m_axes[box.level - 1];
			auto low = box;
			--low.level;
			low.extent[axis] /= 2;
			low.pixels /= 2;
			auto high = low;
			high.origin[axis] += low.extent[axis];
			high.first += low.pixels;
			m_stack[m_top++] = high;
			m_stack[m_top++] = low;
		}
		return {};
	}

  public:
	explicit guest_surface_byte_cursor(const guest_surface_layout& layout) : m_layout(layout)
	{
		validate_guest_surface_range(layout.range);
		if (layout.column_stride)
		{
			if (layout.morton || layout.pixel_bytes != 1) throw std::invalid_argument("Strided guest byte coverage cannot describe Morton pixels");
			const auto expected = plan_guest_byte_matrix(layout.range.address, layout.width, layout.height, layout.column_stride, layout.row_pitch);
			if (expected.range.size != layout.range.size) throw std::invalid_argument("Strided guest byte coverage has inconsistent storage size");
			m_layout.row_pitch = expected.row_pitch;
			return;
		}
		if (!layout.range.size && !layout.width && !layout.morton)
			return;
		const auto linear = guest_surface_linear_layout(layout.width, layout.height, layout.pixel_bytes, layout.row_pitch);
		m_layout.row_pitch = linear.bytes_per_row;
		if (!layout.morton)
		{
			if (linear.allocation_size != layout.range.size)
				throw std::invalid_argument("Guest linear coverage has inconsistent storage size");
			return;
		}
		if (layout.morton->size != std::array<std::size_t, 3>{layout.width, layout.height, 1})
			throw std::invalid_argument("Guest Morton coverage has inconsistent active dimensions");
		const auto plan = plan_swizzle(*layout.morton, layout.pixel_bytes);
		if (plan.storage_bytes != layout.range.size)
			throw std::invalid_argument("Guest Morton coverage has inconsistent storage size");
		std::size_t levels = 0;
		for (unsigned bit = 0; bit < 64; ++bit)
			for (unsigned axis = 0; axis < 3; ++axis)
				if (bit < plan.log2[axis])
					m_axes[levels++] = static_cast<unsigned char>(axis);
		m_stack[0] = {{}, layout.morton->storage, 0, plan.storage_bytes / layout.pixel_bytes, levels};
		m_top = 1;
	}
	[[nodiscard]] std::optional<guest_surface_range> next()
	{
		auto result = m_saved ? std::exchange(m_saved, {}) : raw_next();
		if (!result)
			return {};
		while (const auto following = raw_next())
		{
			if (std::uint64_t(result->address) + result->size == following->address)
				result->size += following->size;
			else
			{
				m_saved = following;
				break;
			}
		}
		return result;
	}
};

[[nodiscard]] inline guest_surface_layout guest_byte_coverage(guest_surface_range range)
{
	validate_guest_surface_range(range);
	return {range, range.size, 1, 1, range.size, {}};
}
[[nodiscard]] inline bool overlaps(const guest_surface_layout& a, const guest_surface_layout& b)
{
	if (!overlaps(a.range, b.range))
		return false;
	guest_surface_byte_cursor ac(a), bc(b);
	auto ar = ac.next(), br = bc.next();
	while (ar && br)
	{
		if (overlaps(*ar, *br))
			return true;
		if (std::uint64_t(ar->address) + ar->size <= br->address)
			ar = ac.next();
		else
			br = bc.next();
	}
	return false;
}
[[nodiscard]] inline bool overlaps(guest_surface_range range, const guest_surface_layout& layout)
{
	return overlaps(guest_byte_coverage(range), layout);
}
} // namespace metal4
