#include "M4FragmentShader.h"
#include "M4Sampling.h"
#include "M4DrawValidation.h"
#include "M4ConstantUpload.h"
#include "M4SampleLayout.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <cmath>
#include <stdexcept>

namespace metal4
{

fragment_shader::fragment_shader(const compiler& compiler, std::span<const std::byte> code,
								 const fragment_program_options& options)
	: fragment_shader(compiler, decode_fragment_program(code), generate_fragment_program(code, options), options, std::make_shared<int>(0))
{
}

fragment_shader::fragment_shader(const compiler& compiler, decoded_fragment_program image,
	const generated_fragment_program& program, const fragment_program_options& options,
	std::shared_ptr<void> constant_identity, id<MTLLibrary> prepared)
	: m_image(std::move(image)), m_constant_identity(std::move(constant_identity)), m_dimensions(options.textures), m_border_mask(options.border_textures), m_stored_repeat_mask(options.stored_repeat_textures), m_stored_mixed_filter_mask(options.stored_mixed_filter_textures), m_stored_border_atlas_mask(options.stored_border_atlas_textures), m_shadow_mask(options.shadow_textures), m_depth_mask(options.depth_textures), m_redirected_mask(options.redirected_textures), m_multisampled_mask(options.multisampled_textures), m_sample_raster_mask(options.sample_raster_textures), m_logic_rop(options.logic_rop), m_blend_rop(options.blend_rop), m_output_rop(options.output_rop), m_half_storage_mask(options.half_storage_mask), m_float32_exports(options.float32_exports), m_raster_sample_count(options.raster_sample_count), m_depth_mirror(options.depth_mirror)
{
	for (const auto& instruction : m_image.instructions)
		if (instruction.literal) ++m_literal_count;
	m_texture_mask = program.texture_mask;
	m_library = prepared ? prepared : compiler.library(program.source, @"RPCS3 native RSX fragment program");
	if (m_library.device != compiler.device() || ![m_library.functionNames containsObject:@"rpcs3_rsx_fragment"])
		throw std::invalid_argument("Native prepared fragment library has a different device or entry");
}

MTL4LibraryFunctionDescriptor* fragment_shader::function() const
{
	return library_function(m_library, @"rpcs3_rsx_fragment");
}

std::vector<upload_requirement> fragment_shader::uploads() const
{
	return plan_uploads(m_texture_mask);
}

std::vector<upload_requirement> fragment_shader::plan_uploads(std::uint32_t texture_mask)
{
	std::vector<upload_requirement> result{{sizeof(fragment_context), 256}, {sizeof(fragment_inputs::sampling), 16}};
	if (texture_mask)
	{
		result.push_back({sizeof(fragment_inputs::coordinates), 16});
		result.push_back({sizeof(fragment_inputs::texels), 16});
	}
	return result;
}

bindings fragment_shader::prepare(context& ctx, fragment_context state,
								  std::span<const std::byte> current_code, const fragment_inputs& inputs) const
{
	return prepare(ctx, state, prepare_constants(ctx, current_code), inputs);
}

fragment_shader::constant_binding fragment_shader::prepare_constants(context& ctx, std::span<const std::byte> current_code) const
{
	return capture_constants(ctx, m_library.device, m_image, m_constant_identity, current_code);
}

fragment_shader::constant_binding fragment_shader::capture_constants(context& ctx, id<MTLDevice> device,
	const decoded_fragment_program& image, const std::shared_ptr<void>& identity, std::span<const std::byte> current_code)
{
	if (device != ctx.device())
		throw std::invalid_argument("Metal 4 fragment constants belong to another device");
	return {identity, ctx.constant_uploads().fragment(image, current_code)};
}

bindings fragment_shader::prepare(context& ctx, fragment_context state,
	const constant_binding& constants, const fragment_inputs& inputs) const
{
	if (constants.identity != m_constant_identity || m_library.device != ctx.device() || state.instruction_budget == 0 || state.instruction_budget > 65536 ||
		state.alpha_function > 7 || state.fog_mode > 6 || (state.texcoord_2d_mask & ~0x3ffu) ||
		(state.point_sprite_mask & ~0x3ffu) || !std::isfinite(state.wpos_scale) ||
		!std::isfinite(state.wpos_bias_x) || !std::isfinite(state.wpos_bias_y) ||
		(state.derivative_sign_y != 1 && state.derivative_sign_y != -1))
	{
		throw std::invalid_argument("Metal 4 RSX fragment context or device is invalid");
	}
	if (state.depth_bounds_enabled > 1 || (state.depth_bounds_enabled && !m_depth_mirror.enabled()))
		throw std::invalid_argument("Fragment stored-depth bounds require a compiled mirror contract");
	if (state.depth_bounds_enabled)
		validate_depth_bounds(state.depth_min, state.depth_max);
	bindings result;
	for (std::size_t i = 0; i < m_dimensions.size(); ++i)
	{
		if (!(m_texture_mask & (1u << i)))
			continue;
		validate_texture_coordinates(inputs.coordinates[i]);
		const auto coordinate_flags = inputs.coordinates[i].flags;
		const bool atlas = coordinate_flags & texture_coordinate_border_atlas;
		const bool cube_atlas = inputs.coordinates[i].border_atlas[3] & border_atlas_cube;
		const auto storage_dimension = inputs.coordinates[i].border_atlas[3] & border_atlas_one_dimensional ?
			vertex_texture_dimension::two : m_dimensions[i];
		const bool redirected = m_redirected_mask & (1u << i);
		const auto samples = sampled_texture_samples(inputs.sampling[i]);
		const bool sample_raster = coordinate_flags & texture_coordinate_sample_raster;
		if (bool(m_multisampled_mask & (1u << i)) != (samples > 1) ||
			bool(m_sample_raster_mask & (1u << i)) != sample_raster || sampled_texture_is_raster(inputs.sampling[i]) != sample_raster)
			throw std::invalid_argument("Compiled MSAA contract differs from bindings");
		validate_sampled_texture(ctx, inputs.textures[i], inputs.samplers[i], storage_dimension,
			inputs.sampling[i], m_shadow_mask & (1u << i), redirected || (m_depth_mask & (1u << i)), cube_atlas, samples);
		if (coordinate_flags & texture_coordinate_msaa_window)
		{
			if (samples == 1)
				throw std::invalid_argument("Multisample window requires a native multisample image");
			const auto image = inputs.textures[i];
			const auto extent = sample_expanded_size(image.width, image.height, samples);
			if (inputs.coordinates[i].maximum[0] > float(extent[0]) || inputs.coordinates[i].maximum[1] > float(extent[1]))
				throw std::invalid_argument("Multisample window exceeds its native sample raster");
		}
		if (redirected != bool(inputs.sampling[i].flags & sampler_redirected_depth))
			throw std::invalid_argument("Compiled redirected depth contract differs from bindings");
		if (redirected)
		{
			const auto source = inputs.textures[i];
			const auto stencil = inputs.stencil_views[i];
			if (!stencil || source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8 ||
				source.mipmapLevelCount != 1 || stencil.pixelFormat != MTLPixelFormatX32_Stencil8 ||
				stencil.parentTexture != source || stencil.textureType != source.textureType || stencil.sampleCount != samples ||
				stencil.width != source.width || stencil.height != source.height)
				throw std::invalid_argument("Redirected stencil view does not match its native depth snapshot");
			result.set_texture(16 + i, stencil);
		}
		if (bool(m_stored_border_atlas_mask & (1u << i)) != atlas)
			throw std::invalid_argument("Compiled fragment stored-border atlas differs from bindings");
		if (atlas)
		{
			const auto& data = inputs.coordinates[i].border_atlas;
			const auto image = inputs.textures[i];
			const bool one_dimensional = data[3] & border_atlas_one_dimensional;
			const auto depth = inputs.coordinates[i].border_atlas_depth;
			if (one_dimensional != (m_dimensions[i] == vertex_texture_dimension::one) ||
				bool(depth) != (m_dimensions[i] == vertex_texture_dimension::three) ||
				cube_atlas != (m_dimensions[i] == vertex_texture_dimension::cube))
				throw std::invalid_argument("Stored-border atlas does not match the guest texture dimension");
			const auto expected = plan_stored_border_atlas(data[0], data[1], data[2], one_dimensional).extent;
			if (image.mipmapLevelCount != 1 || image.width != expected[0] || image.height != expected[1] ||
				image.depth != (depth ? depth : 1u))
				throw std::invalid_argument("Native texture dimensions differ from the stored-border atlas");
		}
		if (bool(m_stored_repeat_mask & (1u << i)) != bool(coordinate_flags & texture_coordinate_repeat) ||
			bool(m_stored_mixed_filter_mask & (1u << i)) != bool(coordinate_flags & texture_coordinate_mixed_filter) ||
			((coordinate_flags & (texture_coordinate_periodic | texture_coordinate_mixed_filter)) &&
				(m_dimensions[i] != vertex_texture_dimension::two || inputs.textures[i].mipmapLevelCount != 1 ||
					inputs.textures[i].width < 3 || inputs.textures[i].height < 3)))
			throw std::invalid_argument("Compiled fragment stored-border contract differs from bindings");
		if (bool(inputs.coordinates[i].border_atlas_depth) != bool(inputs.sampling[i].flags & sampler_volume_border))
			throw std::invalid_argument("Stored volume atlas lacks matching Z border sampling state");
		validate_texel_processing(inputs.texels[i]);
		if (bool(m_border_mask & (1u << i)) != bool(inputs.sampling[i].flags & sampler_custom_border))
			throw std::invalid_argument("Compiled fragment border coverage contract differs from bindings");
		if (m_border_mask & (1u << i))
		{
			validate_border_coverage(ctx, inputs.textures[i], inputs.border_coverage[i], inputs.sampling[i]);
			result.set_texture(16 + i, inputs.border_coverage[i]);
		}
		result.set_texture(i, inputs.textures[i]);
		result.set_sampler(i, inputs.samplers[i]);
	}
	state.constant_count = m_literal_count;
	result.set_buffer(1, constants.buffer, 0);
	const auto uniform = ctx.upload(std::as_bytes(std::span{&state, 1}), 256);
	result.set_buffer(2, uniform.buffer, uniform.offset);
	const auto sampling = ctx.upload(std::as_bytes(std::span(inputs.sampling)), 16);
	result.set_buffer(3, sampling.buffer, sampling.offset);
	if (m_texture_mask)
	{
		const auto coordinates = ctx.upload(std::as_bytes(std::span(inputs.coordinates)), 16);
		result.set_buffer(4, coordinates.buffer, coordinates.offset);
		const auto texels = ctx.upload(std::as_bytes(std::span(inputs.texels)), 16);
		result.set_buffer(5, texels.buffer, texels.offset);
	}
	return result;
}

} // namespace metal4
