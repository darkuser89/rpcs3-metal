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

// The cache resolves addresses/layout/resources using the frozen words,
// including OFFSET and CONTROL3 pitch. All requested units are prepared into
// local candidates; a failure publishes neither partial options nor inputs.
// Sampler LRU population is allowed to change on failure; existing input
// owners keep their samplers alive even if a candidate acquisition evicts one.
template <typename Resolve>
void apply_fragment_texture_snapshot(const fragment_texture_snapshot& snapshot, Resolve&& resolve,
									 context& ctx, sampler_cache& samplers, fragment_program_options& options, fragment_inputs& inputs,
									 const texture_state_overrides& overrides = {}, const null_texture_family* nulls = nullptr)
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
		const auto state = decode_fragment_texture_state(words, resource.metadata, overrides);
		if (state.redirected || state.samples != 1)
			throw std::invalid_argument("Frozen native texture binding needs redirected/MSAA sampling emulation");
		configure_fragment_texture_state(candidate_options, unit, state);
		candidate_inputs.set_texture_state(unit, resource.texture, samplers, state);
		validate_sampled_texture(ctx, candidate_inputs.textures[unit], candidate_inputs.samplers[unit], state.policy.dimension,
								 candidate_inputs.sampling[unit], state.shadow, state.depth && !state.shadow && !state.redirected);
	}
	options = std::move(candidate_options);
	inputs = std::move(candidate_inputs);
}
} // namespace metal4
