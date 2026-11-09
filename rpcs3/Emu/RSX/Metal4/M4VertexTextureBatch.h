#pragma once

#include "M4FragmentTextureBatch.h"
#include "M4Sampling.h"
#include "M4TextureStateBinding.h"
#include "M4VertexShader.h"
#include "M4VertexTextureState.h"

namespace metal4
{
inline void configure_vertex_texture_state(vertex_program_options& options, unsigned unit, const decoded_texture_state& state)
{
	options.textures[unit] = state.policy.dimension;
	options.border_textures = (options.border_textures & ~(1u << unit)) | (state.border_emulation ? 1u << unit : 0);
}

// Resolve against the frozen register epoch before writer preparation. Commit
// options/resources only after all units validate;cache/LRU population may
// change on failure. Returned native texture snapshots must stay immutable.
template <typename Resolve>
void apply_vertex_texture_snapshot(const vertex_texture_snapshot& snapshot, Resolve&& resolve,
								   context& ctx, sampler_cache& samplers, vertex_program_options& options, vertex_inputs& inputs,
								   const null_texture_family* nulls = nullptr, border_texture_preparation* preparation = nullptr)
{
	auto candidate_options = options;
	auto candidate_inputs = inputs;
	for (unsigned unit = 0; unit < 4; ++unit)
	{
		if (!(snapshot.referenced_mask() & (1u << unit)))
			continue;
		const auto& words = snapshot.at(unit);
		if (!(words.control0 >> 31))
		{
			const auto state = null_texture_state(candidate_options.textures[unit], sampler_stage::vertex);
			const auto image = null_texture_image(nulls, state.policy.dimension);
			const auto sampler = samplers.acquire(state.sampler, state.policy);
			validate_sampled_texture(ctx, image, sampler.sampler, state.policy.dimension, sampler.parameters);
			configure_vertex_texture_state(candidate_options, unit, state);
			candidate_inputs.border_coverage[unit] = nil;
			candidate_inputs.textures[unit] = image;
			candidate_inputs.set_sampler(unit, sampler);
			continue;
		}
		const auto resource = resolve(unit, words.upload_words());
		const auto state = decode_vertex_texture_state(words, resource.metadata);
		const auto width = words.image_rect >> 16;
		const auto height = state.policy.dimension == vertex_texture_dimension::one ? 1u : words.image_rect & 65535u;
		const auto depth = state.policy.dimension == vertex_texture_dimension::three ? words.control3 >> 20 : 1u;
		if (!resource.texture || resource.texture.width != width || resource.texture.height != height ||
			resource.texture.depth != depth || resource.texture.mipmapLevelCount != state.policy.actual_mips)
			throw std::invalid_argument("Vertex texture native extent/mips require coordinate or storage emulation");
		const bool converted = state.border_emulation && resource.texture.pixelFormat != MTLPixelFormatRGBA32Float;
		const auto view = converted ? samplers.border_color_view(resource.texture, state.view_channels, preparation) : make_register_texture_view(resource.texture, state, &samplers.texture_views());
		auto sampler_state = state.sampler;
		if (state.border_emulation) sampler_state.border_color.fill(1);
		auto sampler = samplers.acquire(sampler_state, state.policy);
		id<MTLTexture> coverage = nil;
		if (state.border_emulation)
		{
			coverage = samplers.border_coverage(resource.texture, preparation);
			sampler.parameters.flags |= sampler_custom_border;
			for (unsigned c = 0; c < 4; ++c)
				sampler.parameters.border_adjust[c] = state.view_border_color[c] - (!converted && state.view_channels[c] == 4 ? 0.f : 1.f);
			validate_border_coverage(ctx, view, coverage, sampler.parameters);
		}
		configure_vertex_texture_state(candidate_options, unit, state);
		candidate_inputs.border_coverage[unit] = coverage;
		validate_sampled_texture(ctx, view, sampler.sampler, state.policy.dimension, sampler.parameters);
		candidate_inputs.textures[unit] = view;
		candidate_inputs.set_sampler(unit, sampler);
	}
	options = std::move(candidate_options);
	inputs = std::move(candidate_inputs);
}
} // namespace metal4
