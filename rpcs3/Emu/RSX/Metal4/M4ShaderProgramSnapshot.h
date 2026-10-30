#pragma once

#include "M4FragmentMSL.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace metal4
{
inline constexpr std::size_t rsx_vertex_instruction_limit = 544;
inline constexpr std::uint32_t rsx_program_flat = 0x800, rsx_program_instanced_constants = 0x20000;
inline constexpr std::uint32_t rsx_fragment_float32 = 0x40, rsx_fragment_depth_export = 0xe;
[[nodiscard]] inline vertex_texture_dimension decode_program_texture_dimension(unsigned value)
{
	switch (value)
	{
	case 0:
		return vertex_texture_dimension::one;
	case 1:
		return vertex_texture_dimension::two;
	case 2:
		return vertex_texture_dimension::cube;
	case 3:
		return vertex_texture_dimension::three;
	default:
		throw std::invalid_argument("Invalid analysed RSX program texture dimension");
	}
}
// These import already analysed/rebased RSX programs, not raw shader addresses.
// Caller must exclude method/VM mutations while copying the shader bytes and
// metadata. Real VM/protection scheduling remains a renderer responsibility.
class vertex_program_snapshot final
{
	std::vector<std::array<std::uint32_t, 4>> m_code;
	std::vector<std::uint8_t> m_live;
	std::uint32_t m_entry = 0, m_base = 0, m_output_mask = 0, m_control = 0, m_dimensions = 0;

  public:
	template <typename Program>
	[[nodiscard]] static vertex_program_snapshot capture(const Program& program, bool constants_are_current_pass = false)
	{
		if (program.data.empty() || program.data.size() % 4 || program.data.size() / 4 > rsx_vertex_instruction_limit ||
			program.data.size() / 4 > program.instruction_mask.size() || program.base_address >= rsx_vertex_instruction_limit ||
			program.data.size() / 4 > rsx_vertex_instruction_limit - program.base_address || program.entry < program.base_address ||
			program.entry - program.base_address >= program.data.size() / 4)
			throw std::invalid_argument("Analysed RSX vertex program has invalid length/base/entry/mask");
		if ((program.ctrl & rsx_program_instanced_constants) && !constants_are_current_pass)
			throw std::invalid_argument("Native RSX program import requires per-instance constant binding");
		vertex_program_snapshot result;
		result.m_entry = program.entry - program.base_address;
		result.m_base = program.base_address;
		result.m_output_mask = program.output_mask;
		// A clause emitted as separate native draws already owns this pass's
		// complete constants;it needs no host-instance indirection table.
		result.m_control = program.ctrl & (constants_are_current_pass ? ~rsx_program_instanced_constants : 0xffffffffu);
		result.m_dimensions = program.texture_state.texture_dimensions;
		result.m_code.resize(program.data.size() / 4);
		result.m_live.resize(result.m_code.size());
		for (std::size_t i = 0; i < result.m_code.size(); ++i)
		{
			std::copy_n(program.data.begin() + i * 4, 4, result.m_code[i].begin());
			result.m_live[i] = bool(program.instruction_mask.test(i));
		}
		if (!result.m_live[result.m_entry])
			throw std::invalid_argument("Analysed RSX vertex entry is marked dead");
		return result;
	}
	[[nodiscard]] std::span<const std::array<std::uint32_t, 4>> code() const noexcept { return m_code; }
	[[nodiscard]] std::span<const std::uint8_t> live() const noexcept { return m_live; }
	// The analyser span borrows this snapshot. Consume while the owner lives;
	// vertex_shader construction uses it synchronously and retains no span.
	[[nodiscard]] vertex_program_options options(bool point_output) const
	{
		vertex_program_options options;
		options.entry = m_entry;
		options.base_address = m_base;
		options.live_instructions = m_live;
		options.flat_colors = bool(m_control & rsx_program_flat);
		options.point_output = point_output;
		for (unsigned i = 0; i < 4; ++i)
			options.textures[i] = decode_program_texture_dimension((m_dimensions >> (i * 2)) & 3);
		return options;
	}
	void apply(vertex_context& context) const noexcept { context.output_mask = m_output_mask; }
};
class fragment_program_snapshot final
{
	std::vector<std::byte> m_code;
	std::uint32_t m_control = 0, m_texcoord_control = 0, m_color_count = 0, m_dimensions = 0, m_shadow = 0, m_redirected = 0, m_msaa = 0;
	bool m_two_sided = false;

  public:
	template <typename Program>
	[[nodiscard]] static fragment_program_snapshot capture(const Program& program, std::size_t maximum_bytes)
	{
		if (!program.valid || !program.ucode_length || program.ucode_length % 16 || !program.get_data())
			throw std::invalid_argument("Analysed RSX fragment program has invalid code storage");
		if (!maximum_bytes || program.ucode_length > maximum_bytes)
			throw std::length_error("Analysed RSX fragment snapshot exceeds its explicit limit");
		if (program.mrt_buffers_count > 4)
			throw std::invalid_argument("Analysed RSX fragment program exceeds four color outputs");
		fragment_program_snapshot result;
		result.m_code.resize(program.ucode_length);
		std::memcpy(result.m_code.data(), program.get_data(), result.m_code.size());
		result.m_control = program.ctrl;
		result.m_texcoord_control = program.texcoord_control_mask;
		result.m_color_count = program.mrt_buffers_count;
		result.m_two_sided = program.two_sided_lighting;
		result.m_dimensions = program.texture_state.texture_dimensions;
		result.m_shadow = program.texture_state.shadow_textures;
		result.m_redirected = program.texture_state.redirected_textures;
		result.m_msaa = program.texture_state.multisampled_textures;
		return result;
	}
	[[nodiscard]] std::span<const std::byte> code() const noexcept { return m_code; }
	[[nodiscard]] fragment_program_options options(bool allow_output_rounding) const
	{
		fragment_program_options options;
		options.flat_colors = bool(m_control & rsx_program_flat);
		options.float32_exports = bool(m_control & rsx_fragment_float32);
		options.depth_export = bool(m_control & rsx_fragment_depth_export);
		options.color_count = m_color_count;
		options.output_rop.round_to_8bit = allow_output_rounding && bool(m_control & 0x80000);
		options.output_rop.srgb = bool(m_control & 0x100000);
		options.output_rop.polygon_stipple = bool(m_control & 0x800000);
		options.output_rop.stochastic_alpha_coverage = bool(m_control & 0x1000000);
		options.shadow_textures = m_shadow;
		options.redirected_textures = m_redirected;
		options.multisampled_textures = m_msaa;
		for (unsigned i = 0; i < 16; ++i)
			options.textures[i] = decode_program_texture_dimension((m_dimensions >> (i * 2)) & 3);
		return options;
	}
	void apply(vertex_context& vertex, fragment_context& fragment) const noexcept
	{
		vertex.two_sided = fragment.two_sided = m_two_sided;
		fragment.texcoord_2d_mask = m_texcoord_control & 65535;
		fragment.point_sprite_mask = m_texcoord_control >> 16;
		fragment.alpha_test = bool(m_control & 0x400000);
	}
};
} // namespace metal4
