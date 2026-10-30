#pragma once

#include "M4TextureState.h"

#include <span>

namespace metal4
{
inline constexpr std::size_t fragment_texture_method_base = 0x1a00 / 4;
inline constexpr std::size_t fragment_texture_control3_base = 0x1840 / 4;
inline constexpr std::size_t fragment_texture_method_stride = 8;

// Owns the referenced register words. Capture on the RSX render thread while
// method-register writes are excluded; a non-atomic array cannot be safely
// read during concurrent writes. Cache callbacks run only after this copy.
class fragment_texture_snapshot final
{
	std::uint32_t m_mask = 0;
	std::array<fragment_texture_registers, 16> m_units{};

  public:
	[[nodiscard]] std::uint32_t referenced_mask() const noexcept { return m_mask; }
	[[nodiscard]] const fragment_texture_registers& at(std::size_t index) const
	{
		if (index >= m_units.size() || !(m_mask & (1u << index)))
			throw std::out_of_range("Texture unit is absent from the frozen RSX snapshot");
		return m_units[index];
	}
	[[nodiscard]] static fragment_texture_snapshot capture(std::span<const std::uint32_t> methods, std::uint32_t mask)
	{
		if (mask & ~0xffffu)
			throw std::invalid_argument("RSX texture snapshot mask exceeds sixteen units");
		fragment_texture_snapshot snapshot;
		snapshot.m_mask = mask;
		for (unsigned unit = 0; unit < 16; ++unit)
		{
			if (!(mask & (1u << unit)))
				continue;
			const auto base = fragment_texture_method_base + fragment_texture_method_stride * unit;
			// CONTROL3 is a separate contiguous block, not part of the 8-word
			// per-unit texture block. Bounds precede every referenced-unit read.
			if (base + 7 >= methods.size() || fragment_texture_control3_base + unit >= methods.size())
				throw std::out_of_range("RSX method array is too short for the referenced texture snapshot");
			auto& regs = snapshot.m_units[unit];
			regs.offset = methods[base];
			regs.format = methods[base + 1];
			regs.address = methods[base + 2];
			regs.control0 = methods[base + 3];
			regs.control1 = methods[base + 4];
			regs.filter = methods[base + 5];
			regs.image_rect = methods[base + 6];
			regs.border_color = methods[base + 7];
			regs.control3 = methods[fragment_texture_control3_base + unit];
		}
		return snapshot;
	}
};

// rsx::rsx_state exposes this exact host-endian register array publicly.
// Taking it by const reference avoids constructing/copying the whole RSX
// state, including its draw clause and vertex/transform state.
template <typename RsxState>
[[nodiscard]] fragment_texture_snapshot capture_fragment_texture_snapshot(const RsxState& state, std::uint32_t mask)
{
	return fragment_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), mask);
}
} // namespace metal4
