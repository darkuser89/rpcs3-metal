#pragma once

#include "M4DrawTextureEpoch.h"
#include "M4RSXSamplerDescriptor.h"

namespace metal4
{
// The real RSX descriptor arrays and native images share one returned owner.
// Invoke while the renderer excludes method/VM writers. Submission consumes
// textures() without consulting VM/cache again.
class rsx_texture_program_epoch final : public rsx_program_sampler_epoch
{
	draw_texture_epoch m_textures;

  public:
	// Pending jobs expose only CPU sampler metadata. The returned descriptors
	// support program/owned-packet capture while native uploads are in flight;
	// they deliberately carry no native image or completion claim.
	template <typename State, typename Pending>
	[[nodiscard]] static rsx_program_sampler_epoch capture_metadata(const State& state, const Pending& pending)
	{
		const auto& vertex = pending.vertex_words();
		const auto& fragment = pending.fragment_words();
		if (!texture_words_match(vertex, fragment,
			vertex_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), vertex.referenced_mask()),
			capture_fragment_texture_snapshot(state, fragment.referenced_mask())))
			throw std::invalid_argument("Pending sampler metadata differs from the captured registers");
		rsx_program_sampler_epoch epoch;
		for (unsigned unit = 0; unit < 4; ++unit)
			if (vertex.referenced_mask() & (1u << unit))
			{
				const auto words = vertex.at(unit).upload_words();
				epoch.vertex[unit] = std::make_unique<rsx_sampler_descriptor>(words, pending.metadata(true, unit),
					(words.control0 >> 31) ? rsx::texture_format_ex((words.format >> 8) & 255) : rsx::texture_format_ex{});
			}
		for (unsigned unit = 0; unit < 16; ++unit)
			if (fragment.referenced_mask() & (1u << unit))
				epoch.fragment[unit] = std::make_unique<rsx_sampler_descriptor>(fragment.at(unit), pending.metadata(false, unit),
				(fragment.at(unit).control0 >> 31) ? state.fragment_textures[unit].format_ex() : rsx::texture_format_ex{});
		return epoch;
	}
	template <typename State>
	[[nodiscard]] static rsx_texture_program_epoch capture(const State& state, draw_texture_epoch textures)
	{
		const auto& v = textures.vertex_words();
		const auto& f = textures.fragment_words();
		const auto vertex_mask = v.referenced_mask(), fragment_mask = f.referenced_mask();
		textures.validate_words(
			vertex_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), v.referenced_mask()),
			capture_fragment_texture_snapshot(state, f.referenced_mask()));
		rsx_texture_program_epoch epoch;
		epoch.m_textures = std::move(textures);
		for (unsigned unit = 0; unit < 4; ++unit)
			if (vertex_mask & (1u << unit))
			{
				const auto words = epoch.m_textures.vertex_words().at(unit).upload_words();
				epoch.vertex[unit] = std::make_unique<rsx_sampler_descriptor>(words, epoch.m_textures.vertex(unit).metadata,
																			  (words.control0 >> 31) ? rsx::texture_format_ex((words.format >> 8) & 255) : rsx::texture_format_ex{});
			}
		for (unsigned unit = 0; unit < 16; ++unit)
			if (fragment_mask & (1u << unit))
				epoch.fragment[unit] = std::make_unique<rsx_sampler_descriptor>(epoch.m_textures.fragment_words().at(unit),
																				epoch.m_textures.fragment(unit).metadata,
																				(epoch.m_textures.fragment_words().at(unit).control0 >> 31) ? state.fragment_textures[unit].format_ex() : rsx::texture_format_ex{});
		return epoch;
	}
	[[nodiscard]] const draw_texture_epoch& textures() const noexcept { return m_textures; }
};
} // namespace metal4
