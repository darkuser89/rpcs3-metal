#pragma once

#include "M4FragmentMSL.h"
#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4Sampler.h"
#include "M4TexelProcessing.h"
#include "M4TextureCoordinates.h"
#include "M4TextureState.h"
#include "M4TextureStateBinding.h"

namespace metal4
{

struct fragment_inputs
{
	std::array<id<MTLTexture>, 16> textures{};
	std::array<id<MTLTexture>, 16> border_coverage{};
	// Samplers must be created via make_sampler()/argumentBufferSupport.
	std::array<id<MTLSamplerState>, 16> samplers{};
	std::array<sampler_parameters, 16> sampling{};
	std::array<texture_coordinates, 16> coordinates{};
	std::array<texel_processing, 16> texels{};
	void set_sampler(std::size_t index, const sampler_binding& binding)
	{
		samplers.at(index) = binding.sampler;
		sampling.at(index) = binding.parameters;
	}
	// Decode from a stable register/cache snapshot first. Acquire validates the
	// effective sampler before changing any input arrays.
	void set_texture_state(std::size_t index, id<MTLTexture> texture, sampler_cache& cache, const decoded_texture_state& state)
	{
		if (index >= textures.size() || !state.enabled)
			throw std::invalid_argument("Invalid/disabled native texture state binding");
		validate_texture_coordinates(state.coordinates);
		validate_texel_processing(state.texels);
		const bool converted = state.border_emulation && !state.depth && !state.shadow && texture.pixelFormat != MTLPixelFormatRGBA32Float;
		const auto view = converted ? cache.border_color_view(texture, state.view_channels) : make_register_texture_view(texture, state, &cache.texture_views());
		auto sampler_state = state.sampler;
		if (state.border_emulation)
			sampler_state.border_color.fill(1);
		auto binding = cache.acquire(sampler_state, state.policy);
		id<MTLTexture> coverage = nil;
		if (state.border_emulation)
		{
			coverage = state.shadow ? cache.border_shadow_view(texture, state.view_border_color[0]) : cache.border_coverage(texture);
			binding.parameters.flags |= sampler_custom_border;
			if (!state.shadow)
				for (unsigned c = 0; c < 4; ++c)
					binding.parameters.border_adjust[c] = state.view_border_color[c] - (state.depth ? (state.depth_remap[c] == depth_sample_channel::zero ? 0.f : 1.f) : !converted && state.view_channels[c] == 4 ? 0.f
																																																				   : 1.f);
		}
		textures[index] = view;
		border_coverage[index] = coverage;
		set_sampler(index, binding);
		coordinates[index] = state.coordinates;
		texels[index] = state.texels;
	}
};

// Compile-time texture declarations must use the same resolved snapshot as
// draw inputs; runtime prepare checks native texture/dimension/compare types.
inline void configure_fragment_texture_state(fragment_program_options& options, std::size_t index, const decoded_texture_state& state)
{
	if (index >= options.textures.size() || !state.enabled)
		throw std::invalid_argument("Invalid/disabled compiled native texture state");
	const auto bit = 1u << index;
	options.border_textures = (options.border_textures & ~bit) | (state.border_emulation ? bit : 0);
	options.textures[index] = state.policy.dimension;
	options.shadow_textures = (options.shadow_textures & ~bit) | (state.shadow ? bit : 0);
	options.depth_textures = (options.depth_textures & ~bit) | (state.depth && !state.shadow && !state.redirected ? bit : 0);
	options.redirected_textures = (options.redirected_textures & ~bit) | (state.redirected ? bit : 0);
	options.multisampled_textures = (options.multisampled_textures & ~bit) | (state.samples > 1 ? bit : 0);
	options.depth_remap[index] = state.depth_remap;
}

class fragment_shader final
{
	friend class shader_cache;
	fragment_shader(const compiler& compiler, decoded_fragment_program image,
					const generated_fragment_program& program, const fragment_program_options& options, id<MTLLibrary> prepared = nil);
	id<MTLLibrary> m_library;
	decoded_fragment_program m_image;
	std::shared_ptr<void> m_constant_identity = std::make_shared<int>(0);
	std::uint32_t m_literal_count = 0;
	std::array<vertex_texture_dimension, 16> m_dimensions;
	std::uint32_t m_texture_mask;
	std::uint32_t m_border_mask;
	std::uint32_t m_shadow_mask;
	std::uint32_t m_depth_mask;
	logic_rop_configuration m_logic_rop;
	blend_rop_configuration m_blend_rop;
	output_rop_configuration m_output_rop;
	std::uint32_t m_half_storage_mask;
	bool m_float32_exports;
	std::uint32_t m_raster_sample_count;
	depth_mirror_configuration m_depth_mirror;

  public:
	class constant_binding final
	{
		friend class fragment_shader;
		std::shared_ptr<void> identity;
		id<MTLBuffer> buffer;
		constant_binding(std::shared_ptr<void> identity, id<MTLBuffer> buffer) : identity(std::move(identity)), buffer(buffer) {}
	};
	fragment_shader(const compiler& compiler, std::span<const std::byte> code,
					const fragment_program_options& options = {});
	[[nodiscard]] MTL4LibraryFunctionDescriptor* function() const;
	[[nodiscard]] std::vector<upload_requirement> uploads() const;
	[[nodiscard]] const logic_rop_configuration& logic_rop() const noexcept
	{
		return m_logic_rop;
	}
	[[nodiscard]] const blend_rop_configuration& blend_rop() const noexcept
	{
		return m_blend_rop;
	}
	[[nodiscard]] const output_rop_configuration& output_rop() const noexcept
	{
		return m_output_rop;
	}
	[[nodiscard]] std::uint32_t half_storage_mask() const noexcept
	{
		return m_half_storage_mask;
	}
	[[nodiscard]] bool float32_exports() const noexcept { return m_float32_exports; }
	[[nodiscard]] std::uint32_t raster_sample_count() const noexcept
	{
		return m_raster_sample_count;
	}
	[[nodiscard]] const depth_mirror_configuration& depth_mirror() const noexcept
	{
		return m_depth_mirror;
	}
	// Refreshes literal bits from the current guest stream; instruction changes
	// reject the old library. Uniforms/literals use completion-owned uploads.
	// Call between context.begin() and submission; tables remain stage-local.
	// current_code must be a stable CPU snapshot for the duration of the call.
	[[nodiscard]] bindings prepare(context& ctx, fragment_context state,
								   std::span<const std::byte> current_code, const fragment_inputs& inputs) const;
	// Capture the validated immutable literal version before starting a draw.
	// Cache eviction/pressure cannot invalidate this owned binding. It may only
	// be consumed by the shader which prepared it; no GPU work is encoded here.
	[[nodiscard]] constant_binding prepare_constants(context& ctx, std::span<const std::byte> current_code) const;
	[[nodiscard]] bindings prepare(context& ctx, fragment_context state,
		const constant_binding& constants, const fragment_inputs& inputs) const;
};

} // namespace metal4
