#pragma once

#include "M4FragmentShader.h"
#include "M4Sampling.h"
#include "M4TextureSnapshot.h"

#include <utility>

namespace metal4
{
struct resolved_fragment_texture
{
	id<MTLTexture> texture = nil;
	texture_cache_state metadata;
};
using null_texture_family = std::array<resolved_fragment_texture, 4>;

[[nodiscard]] inline decoded_texture_state null_texture_state(vertex_texture_dimension dimension, sampler_stage stage)
{
	if (static_cast<unsigned>(dimension) >= 4)
		throw std::invalid_argument("Invalid native null texture dimension");
	decoded_texture_state state;
	state.enabled = true;
	state.policy.dimension = dimension;
	state.policy.stage = stage;
	state.sampler.wrap.fill(sampler_wrap::edge);
	return state;
}

[[nodiscard]] inline id<MTLTexture> null_texture_image(const null_texture_family* family, vertex_texture_dimension dimension)
{
	(void)null_texture_state(dimension, sampler_stage::fragment);
	if (!family)
		throw std::invalid_argument("Disabled referenced texture requires owned native null images");
	const auto image = family->at(static_cast<unsigned>(dimension)).texture;
	// Native RSX 1D sampling uses a height-one 2D image,as for real mip chains.
	constexpr std::array types{MTLTextureType2D, MTLTextureType2D, MTLTextureType3D, MTLTextureTypeCube};
	if (!image || image.textureType != types[static_cast<unsigned>(dimension)] || image.width != 1 || image.height != 1 ||
		image.depth != 1 || image.mipmapLevelCount != 1 || image.sampleCount != 1 || image.pixelFormat != MTLPixelFormatBGRA8Unorm)
		throw std::invalid_argument("Native null texture type/storage is invalid");
	return image;
}

[[nodiscard]] inline decoded_texture_state checked_fragment_texture_state(const fragment_texture_registers& words,
	const texture_cache_state& metadata, const texture_state_overrides& overrides = {})
{
	const auto state = decode_fragment_texture_state(words, metadata, overrides);
	if ((state.coordinates.flags & texture_coordinate_msaa_window) && state.samples == 1)
		throw std::invalid_argument("Multisample window requires a native multisample image");
	const bool sample_raster = state.coordinates.flags & texture_coordinate_sample_raster;
	if ((sample_raster && state.samples != 1) || (!sample_raster && state.samples != 1 && state.samples != 2 && state.samples != 4) ||
		((sample_raster || state.samples > 1) && (
		state.policy.dimension != vertex_texture_dimension::two || metadata.actual_mips != 1 ||
		!supports_direct_msaa_sampling(words, state.redirected))))
		throw std::invalid_argument("Direct MSAA requires a supported single-level native 2D sampling mode");
	if (state.redirected && (state.policy.dimension != vertex_texture_dimension::two || metadata.actual_mips != 1 ||
		(metadata.source_depth_format != 2 && metadata.source_depth_format != 4) ||
		metadata.depth_float != (metadata.source_depth_format == 4)))
		throw std::invalid_argument("Redirected texture requires a single-level native 2D D24S8 snapshot");
	return state;
}

// The cache resolves addresses/layout/resources using the frozen words,
// including OFFSET and CONTROL3 pitch. All requested units are prepared into
// local candidates; a failure publishes neither partial options nor inputs.
// Sampler LRU population is allowed to change on failure; existing input
// owners keep their samplers alive even if a candidate acquisition evicts one.
template <typename Resolve>
void apply_fragment_texture_snapshot(const fragment_texture_snapshot& snapshot, Resolve&& resolve,
									 context& ctx, sampler_cache& samplers, fragment_program_options& options, fragment_inputs& inputs,
									 const texture_state_overrides& overrides = {}, const null_texture_family* nulls = nullptr,
									 border_texture_preparation* preparation = nullptr)
{
	auto candidate_options = options;
	auto candidate_inputs = inputs;
	for (unsigned unit = 0; unit < 16; ++unit)
	{
		if (!(snapshot.referenced_mask() & (1u << unit)))
			continue;
		const auto& words = snapshot.at(unit);
		if (!(words.control0 >> 31))
		{
			const auto state = null_texture_state(candidate_options.textures[unit], sampler_stage::fragment);
			configure_fragment_texture_state(candidate_options, unit, state);
			candidate_inputs.set_texture_state(unit, null_texture_image(nulls, state.policy.dimension), samplers, state);
			validate_sampled_texture(ctx, candidate_inputs.textures[unit], candidate_inputs.samplers[unit], state.policy.dimension, candidate_inputs.sampling[unit]);
			continue;
		}
		const auto resource = resolve(unit, words);
		const auto state = checked_fragment_texture_state(words, resource.metadata, overrides);
		configure_fragment_texture_state(candidate_options, unit, state);
		candidate_inputs.set_texture_state(unit, resource.texture, samplers, state, preparation);
		const auto storage_dimension = state.coordinates.border_atlas[3] & border_atlas_one_dimensional ?
			vertex_texture_dimension::two : state.policy.dimension;
		validate_sampled_texture(ctx, candidate_inputs.textures[unit], candidate_inputs.samplers[unit], storage_dimension,
			candidate_inputs.sampling[unit], state.shadow, state.depth && !state.shadow,
			state.coordinates.border_atlas[3] & border_atlas_cube, state.samples);
	}
	options = std::move(candidate_options);
	inputs = std::move(candidate_inputs);
}
} // namespace metal4
