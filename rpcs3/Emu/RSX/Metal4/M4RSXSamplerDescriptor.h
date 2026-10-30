#pragma once

#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/RSXThread.h"
#include "M4ProgramSamplerPlan.h"
#include "M4RSXDrawFrontend.h"

namespace metal4
{
// Real RSX metadata contract. Independent native snapshots/views implement
// remap and feedback; this descriptor does not expose mutable render storage.
class rsx_sampler_descriptor final : public rsx::sampled_image_descriptor_base
{
  public:
	rsx_sampler_descriptor(const fragment_texture_registers& words, const texture_cache_state& cache,
						   const rsx::texture_format_ex& format)
	{
		if (!(words.control0 >> 31)) return; // Match VK's empty disabled descriptor.
		const auto plan = program_sampler_plan::capture(words, cache);
		if (format.format_bits != ((words.format >> 8) & 255) || (format.features && format.encoded_remap != words.control1) || format.host_features)
			throw std::invalid_argument("RSX sampler format metadata differs from the frozen method epoch");
		image_type = static_cast<rsx::texture_dimension_extended>(plan.dimension);
		format_class = static_cast<rsx::format_class>(plan.format_class);
		samples = static_cast<u8>(plan.samples);
		std::copy(plan.coordinates.scale.begin(), plan.coordinates.scale.end(), texcoord_xform.scale);
		std::copy(plan.coordinates.bias.begin(), plan.coordinates.bias.end(), texcoord_xform.bias);
		std::copy(plan.coordinates.minimum.begin(), plan.coordinates.minimum.end(), texcoord_xform.clamp_min);
		std::copy(plan.coordinates.maximum.begin(), plan.coordinates.maximum.end(), texcoord_xform.clamp_max);
		texcoord_xform.clamp = bool(plan.coordinates.flags & texture_coordinate_clamp);
		format_ex = format;
	}
	u32 encoded_component_map() const override { return 0; }
};

struct rsx_program_sampler_epoch
{
	std::array<std::unique_ptr<rsx::sampled_image_descriptor_base>, 4> vertex;
	std::array<std::unique_ptr<rsx::sampled_image_descriptor_base>, 16> fragment;
};

// Compile against the actual protected RSX analysis methods. A concrete
// renderer must hold its method/VM epoch across this call and owned capture.
// Resolve callbacks receive frozen words and return independent native-cache
// metadata descriptors;they must retain resources used by subsequent submit.
template <typename RendererBase>
class rsx_program_frontend : public rsx_draw_frontend<RendererBase>
{
  protected:
	using rsx_draw_frontend<RendererBase>::rsx_draw_frontend;
	template <typename Epoch>
	void refresh_owned_programs(const Epoch& epoch)
	{
		for (unsigned unit = 0; unit < 4; ++unit)
			if ((this->current_vp_metadata.referenced_textures_mask & (1u << unit)) && !epoch.vertex[unit])
				throw std::invalid_argument("Missing referenced native vertex sampler descriptor");
		for (unsigned unit = 0; unit < 16; ++unit)
			if ((this->current_fp_metadata.referenced_textures_mask & (1u << unit)) && !epoch.fragment[unit])
				throw std::invalid_argument("Missing referenced native fragment sampler descriptor");
		this->m_graphics_state.set(rsx::vertex_program_state_dirty);
		this->m_graphics_state.set(rsx::fragment_program_state_dirty);
		this->get_current_vertex_program(epoch.vertex);
		this->get_current_fragment_program(epoch.fragment);
	}
	// Joint resolver can return a larger epoch that owns both descriptor arrays
	// and the exact Metal images subsequently passed to the executor.
	template <typename State, typename Resolve>
	[[nodiscard]] auto prepare_owned_programs(const State& state, Resolve&& resolve)
	{
		this->analyse_current_rsx_pipeline();
		const auto vertex = vertex_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), this->current_vp_metadata.referenced_textures_mask);
		const auto fragment = capture_fragment_texture_snapshot(state, this->current_fp_metadata.referenced_textures_mask);
		auto epoch = std::invoke(resolve, vertex, fragment);
		refresh_owned_programs(epoch);
		return epoch;
	}
	template <typename State, typename ResolveVertex, typename ResolveFragment>
	[[nodiscard]] rsx_program_sampler_epoch prepare_owned_programs(const State& state, ResolveVertex&& vertex, ResolveFragment&& fragment)
	{
		this->analyse_current_rsx_pipeline();
		const auto v = vertex_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), this->current_vp_metadata.referenced_textures_mask);
		const auto f = capture_fragment_texture_snapshot(state, this->current_fp_metadata.referenced_textures_mask);
		rsx_program_sampler_epoch epoch;
		for (unsigned unit = 0; unit < 4; ++unit)
			if (v.referenced_mask() & (1u << unit))
			{
				epoch.vertex[unit] = std::invoke(vertex, unit, v.at(unit));
				if (!epoch.vertex[unit])
					throw std::invalid_argument("Missing referenced native vertex sampler descriptor");
			}
		for (unsigned unit = 0; unit < 16; ++unit)
			if (f.referenced_mask() & (1u << unit))
			{
				epoch.fragment[unit] = std::invoke(fragment, unit, f.at(unit));
				if (!epoch.fragment[unit])
					throw std::invalid_argument("Missing referenced native fragment sampler descriptor");
			}
		// Cache class/dimension/transform can change without method writes.
		// Refresh both programs for every independently prepared draw epoch.
		refresh_owned_programs(epoch);
		return epoch;
	}
};
} // namespace metal4
