#pragma once

#include "M4TextureState.h"
#include <span>

namespace metal4
{
inline constexpr std::size_t vertex_texture_method_base = 0x900 / 4;
inline constexpr std::size_t vertex_texture_method_stride = 8;
struct vertex_texture_registers
{
	std::uint32_t offset = 0, format = 0, address = 0, control0 = 0;
	std::uint32_t control3 = 0, filter = 0, image_rect = 0, border_color = 0;
	bool operator==(const vertex_texture_registers&) const = default;
	// Reuse native storage/VM/alias resolution, without importing fragment
	// sampler/remap/texel semantics. Vertex border storage is always absent.
	[[nodiscard]] fragment_texture_registers upload_words() const
	{
		return {offset, format | 8u, address, control0 & ~4u, 0xaae4u,
				0x01010000u, image_rect, control3, border_color};
	}
};
class vertex_texture_snapshot final
{
	std::uint32_t m_mask = 0;
	std::array<vertex_texture_registers, 4> m_units{};

  public:
	[[nodiscard]] std::uint32_t referenced_mask() const { return m_mask; }
	[[nodiscard]] const vertex_texture_registers& at(unsigned unit) const
	{
		if (unit >= 4 || !(m_mask & (1u << unit)))
			throw std::out_of_range("Vertex texture unit is absent from snapshot");
		return m_units[unit];
	}
	[[nodiscard]] static vertex_texture_snapshot capture(std::span<const std::uint32_t> words, std::uint32_t mask)
	{
		if (mask & ~15u)
			throw std::invalid_argument("Vertex texture snapshot exceeds four units");
		vertex_texture_snapshot result;
		result.m_mask = mask;
		for (unsigned i = 0; i < 4; ++i)
		{
			if (!(mask & (1u << i)))
				continue;
			const auto base = vertex_texture_method_base + i * vertex_texture_method_stride;
			if (base + 7 >= words.size())
				throw std::out_of_range("RSX method array is too short for vertex texture snapshot");
			result.m_units[i] = {words[base], words[base + 1], words[base + 2], words[base + 3],
								 words[base + 4], words[base + 5], words[base + 6], words[base + 7]};
		}
		return result;
	}
};
[[nodiscard]] inline decoded_texture_state decode_vertex_texture_state(const vertex_texture_registers& words, const texture_cache_state& cache)
{
	decoded_texture_state result;
	result.enabled = words.control0 >> 31;
	if (!result.enabled)
		return result;
	const auto dimension = (words.format >> 4) & 15;
	const bool cube = words.format & 4;
	if (dimension < 1 || dimension > 3 || (cube && dimension != 2) || (words.format & 3) < 1 || (words.format & 3) > 2)
		throw std::invalid_argument("Vertex texture location/dimension is invalid");
	const unsigned width = words.image_rect >> 16, height = dimension == 1 ? 1 : words.image_rect & 65535;
	if (!width || !height || (dimension == 3 && !(words.control3 >> 20)))
		throw std::invalid_argument("Vertex texture has zero logical dimensions");
	result.format = ((words.format >> 8) & 255) & ~0x60u;
	(void)texture_state_detail::features(result.format);
	result.policy.stage = sampler_stage::vertex;
	result.policy.dimension = dimension == 1 ? vertex_texture_dimension::one : dimension == 3 ? vertex_texture_dimension::three
																		   : cube			  ? vertex_texture_dimension::cube
																							  : vertex_texture_dimension::two;
	result.policy.requested_mips = std::min(words.format >> 16, static_cast<unsigned>(std::bit_width(std::max(width, height))));
	result.policy.actual_mips = cache.actual_mips;
	if (!result.policy.requested_mips || cache.actual_mips != result.policy.requested_mips || cache.depth || cache.samples != 1)
		throw std::invalid_argument("Vertex texture needs matching float mip storage and single-sample native metadata");
	result.samples = 1;
	result.sampler.normalized = !((words.format >> 8) & 0x40);
	result.sampler.min_lod = float((words.control0 >> 19) & 4095) / 256;
	result.sampler.max_lod = float((words.control0 >> 7) & 4095) / 256;
	if (result.sampler.max_lod < result.sampler.min_lod)
		throw std::invalid_argument("Vertex texture LOD range is inverted");
	// VK's vertex binding uses nearest,repeat R,no anisotropy/compare/bias,
	// regardless of the unused fragment-style bits or FILTER register.
	for (unsigned i = 0; i < 2; ++i)
	{
		const unsigned wrap = (words.address >> (8 * i)) & 15;
		if (wrap < 1 || wrap > 8)
			throw std::invalid_argument("Vertex texture wrap register is invalid");
		result.sampler.wrap[i] = static_cast<sampler_wrap>(wrap - 1);
	}
	result.sampler.wrap[2] = sampler_wrap::repeat;
	for (unsigned rgba = 0; rgba < 4; ++rgba)
	{
		const auto channel = cache.native_argb_channels[(rgba + 1) % 4];
		if (channel > 5)
			throw std::invalid_argument("Vertex texture native format channels are invalid");
		result.view_channels[rgba] = channel;
		result.sampler.border_color[rgba] = float((words.border_color >> (8 * (3 - (rgba + 1) % 4))) & 255) / 255;
	}
	result.view_border_color = result.sampler.border_color;
	if (std::find(result.sampler.wrap.begin(), result.sampler.wrap.end(), sampler_wrap::border) != result.sampler.wrap.end())
	{
		result.border_emulation = true;
		for (const auto candidate : {std::array<float, 4>{0, 0, 0, 0}, std::array<float, 4>{0, 0, 0, 1}, std::array<float, 4>{1, 1, 1, 1}})
		{
			bool matches = true;
			for (unsigned i = 0; i < 4; ++i)
			{
				const auto channel = result.view_channels[i];
				matches &= result.view_border_color[i] == (channel < 4 ? candidate[channel] : channel == 4 ? 0.f
																										   : 1.f);
			}
			if (matches)
			{
				result.sampler.border_color = candidate;
				result.border_emulation = false;
				break;
			}
		}
	}
	validate_texture_coordinates(cache.coordinates);
	if (cache.coordinates.flags || cache.coordinates.bias != std::array<float, 3>{} ||
		(result.sampler.normalized && cache.coordinates.scale != std::array<float, 3>{1, 1, 1}))
		throw std::invalid_argument("Vertex texture needs native coordinate-transform emulation");
	return result;
}
} // namespace metal4
