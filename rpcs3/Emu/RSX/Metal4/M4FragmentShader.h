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
	std::array<id<MTLTexture>, 16> stencil_views{};
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
	void set_texture_state(std::size_t index, id<MTLTexture> texture, sampler_cache& cache, const decoded_texture_state& state,
		border_texture_preparation* preparation = nullptr)
	{
		if (index >= textures.size() || !state.enabled)
			throw std::invalid_argument("Invalid/disabled native texture state binding");
		validate_texture_coordinates(state.coordinates);
		validate_texel_processing(state.texels);
		const bool converted = state.border_emulation && !state.depth && !state.shadow && texture.pixelFormat != MTLPixelFormatRGBA32Float;
		const bool multisampled = state.samples > 1;
		const bool sample_raster = state.coordinates.flags & texture_coordinate_sample_raster;
		const bool msaa_sampling = multisampled || sample_raster;
		const auto view = msaa_sampling ? texture : converted ? cache.border_color_view(texture, state.view_channels, preparation) :
			make_register_texture_view(texture, state, &cache.texture_views());
		auto sampler_state = state.sampler;
		if (state.border_emulation)
			sampler_state.border_color.fill(1);
		if (state.redirected || msaa_sampling)
			sampler_state.wrap.fill(sampler_wrap::edge);
		if (msaa_sampling)
			sampler_state.anisotropy = 1; // Direct sample reads use the shader limit.
		auto policy = state.policy;
		auto transform = state.coordinates;
		const bool atlas = transform.flags & texture_coordinate_border_atlas;
		native_sampler_key effective{};
		if (atlas)
		{
			effective = resolve_sampler(sampler_state, policy);
			auto& modes = transform.border_atlas[3];
			if (effective.min_filter == MTLSamplerMinMagFilterLinear) modes |= border_atlas_min_linear;
			if (effective.mag_filter == MTLSamplerMinMagFilterLinear) modes |= border_atlas_mag_linear;
			if (effective.mip_filter == MTLSamplerMipFilterNearest) modes |= border_atlas_nearest_mip;
			if (effective.mip_filter == MTLSamplerMipFilterLinear) modes |= border_atlas_linear_mip;
			const bool linear = modes & (border_atlas_min_linear | border_atlas_mag_linear);
			sampler_state.min_filter = linear ? sampler_min_filter::linear : sampler_min_filter::nearest;
			sampler_state.mag_filter = linear ? sampler_mag_filter::linear : sampler_mag_filter::nearest;
			sampler_state.min_lod = sampler_state.max_lod = sampler_state.bias = 0;
			sampler_state.anisotropy = 1; // Atlas filtering is performed in guest coordinates.
			policy.actual_mips = 1;
		}
		auto binding = cache.acquire(sampler_state, policy);
		if (atlas)
		{
			binding.parameters.min_lod = effective.min_lod;
			binding.parameters.max_lod = effective.max_lod;
			binding.parameters.bias = effective.bias;
			if (transform.border_atlas_depth)
			{
				binding.parameters.flags |= sampler_volume_border;
				binding.parameters.border_adjust = state.view_border_color;
			}
		}
		id<MTLTexture> coverage = nil;
		id<MTLTexture> stencil = nil;
		if (state.redirected)
		{
			const auto type = multisampled ? MTLTextureType2DMultisample : MTLTextureType2D;
			if (texture.textureType != type || texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
				texture.mipmapLevelCount != 1 || texture.sampleCount != state.samples)
				throw std::invalid_argument("Redirected depth requires a native D24S8 snapshot with both planes");
			const auto identity = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
			stencil = cache.texture_views().get(texture, MTLPixelFormatX32_Stencil8, type,
				NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
		}
		if (state.redirected || msaa_sampling)
		{
			binding.parameters.flags &= sampler_depth_compare | sampler_depth_float;
			if (state.redirected)
				binding.parameters.flags |= sampler_redirected_depth | (state.redirected_float ? sampler_redirected_float : 0);
			if (msaa_sampling)
			{
				binding.parameters.max_lod = state.policy.linear_filtering ? float(state.sampler.anisotropy) : 1.f;
				binding.parameters.flags |= (sample_raster ? sampler_four_samples :
					sampler_multisampled | (state.samples == 4 ? sampler_four_samples : 0)) |
					((state.sampler.compare_function & 7u) << sampler_point_compare_shift);
				if (state.policy.linear_filtering)
				{
					const auto min = state.sampler.min_filter;
					if (min == sampler_min_filter::linear || min == sampler_min_filter::linear_nearest ||
						min == sampler_min_filter::linear_linear || min == sampler_min_filter::convolution)
						binding.parameters.flags |= sampler_msaa_min_linear;
					if (state.sampler.mag_filter != sampler_mag_filter::nearest)
						binding.parameters.flags |= sampler_msaa_mag_linear;
				}
			}
			for (unsigned i = 0; i < 2; ++i)
				binding.parameters.flags |= static_cast<unsigned>(state.sampler.wrap[i]) << (sampler_point_wrap_shift + 3 * i);
			for (unsigned i = 0; i < 4; ++i)
				binding.parameters.flags |= state.view_channels[i] << (sampler_point_channel_shift + 3 * i);
			binding.parameters.border_adjust = state.view_border_color;
		}
		if (state.border_emulation)
		{
			coverage = state.shadow ? cache.border_shadow_view(texture, state.view_border_color[0], preparation) : cache.border_coverage(texture, preparation);
			binding.parameters.flags |= sampler_custom_border;
			if (!state.shadow)
				for (unsigned c = 0; c < 4; ++c)
					binding.parameters.border_adjust[c] = state.view_border_color[c] - (state.depth ? (state.depth_remap[c] == depth_sample_channel::zero ? 0.f : 1.f) : !converted && state.view_channels[c] == 4 ? 0.f
																																																				   : 1.f);
		}
		textures[index] = view;
		border_coverage[index] = coverage;
		stencil_views[index] = stencil;
		set_sampler(index, binding);
		coordinates[index] = transform;
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
	options.stored_repeat_textures = (options.stored_repeat_textures & ~bit) |
		((state.coordinates.flags & texture_coordinate_repeat) ? bit : 0);
	options.stored_mixed_filter_textures = (options.stored_mixed_filter_textures & ~bit) |
		((state.coordinates.flags & texture_coordinate_mixed_filter) ? bit : 0);
	options.stored_border_atlas_textures = (options.stored_border_atlas_textures & ~bit) |
		((state.coordinates.flags & texture_coordinate_border_atlas) ? bit : 0);
	options.textures[index] = state.policy.dimension;
	options.shadow_textures = (options.shadow_textures & ~bit) | (state.shadow ? bit : 0);
	options.depth_textures = (options.depth_textures & ~bit) | (state.depth && !state.shadow && !state.redirected ? bit : 0);
	options.redirected_textures = (options.redirected_textures & ~bit) | (state.redirected ? bit : 0);
	options.multisampled_textures = (options.multisampled_textures & ~bit) | (state.samples > 1 ? bit : 0);
	options.sample_raster_textures = (options.sample_raster_textures & ~bit) |
		((state.coordinates.flags & texture_coordinate_sample_raster) ? bit : 0);
	options.depth_remap[index] = state.depth_remap;
}

class fragment_shader final
{
	friend class shader_cache;
	fragment_shader(const compiler& compiler, decoded_fragment_program image,
					const generated_fragment_program& program, const fragment_program_options& options,
					std::shared_ptr<void> constant_identity, id<MTLLibrary> prepared = nil);
	id<MTLLibrary> m_library;
	decoded_fragment_program m_image;
	std::shared_ptr<void> m_constant_identity;
	std::uint32_t m_literal_count = 0;
	std::array<vertex_texture_dimension, 16> m_dimensions;
	std::uint32_t m_texture_mask;
	std::uint32_t m_border_mask;
	std::uint32_t m_stored_repeat_mask;
	std::uint32_t m_stored_mixed_filter_mask;
	std::uint32_t m_stored_border_atlas_mask;
	std::uint32_t m_shadow_mask;
	std::uint32_t m_depth_mask;
	std::uint32_t m_redirected_mask;
	std::uint32_t m_multisampled_mask;
	std::uint32_t m_sample_raster_mask;
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

  private:
	[[nodiscard]] static constant_binding capture_constants(context& ctx, id<MTLDevice> device,
		const decoded_fragment_program& image, const std::shared_ptr<void>& identity, std::span<const std::byte> current_code);

  public:
	fragment_shader(const compiler& compiler, std::span<const std::byte> code,
					const fragment_program_options& options = {});
	[[nodiscard]] MTL4LibraryFunctionDescriptor* function() const;
	[[nodiscard]] std::vector<upload_requirement> uploads() const;
	[[nodiscard]] static std::vector<upload_requirement> plan_uploads(std::uint32_t texture_mask);
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
