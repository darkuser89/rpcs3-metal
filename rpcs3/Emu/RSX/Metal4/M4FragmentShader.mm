#include "M4FragmentShader.h"
#include "M4Sampling.h"
#include "M4DrawValidation.h"
#include "M4ConstantUpload.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <cmath>
#include <stdexcept>

namespace metal4
{

fragment_shader::fragment_shader(const compiler& compiler, std::span<const std::byte> code,
								 const fragment_program_options& options)
	: fragment_shader(compiler, decode_fragment_program(code), generate_fragment_program(code, options), options)
{
}

fragment_shader::fragment_shader(const compiler& compiler, decoded_fragment_program image,
	const generated_fragment_program& program, const fragment_program_options& options, id<MTLLibrary> prepared) : m_image(std::move(image)), m_dimensions(options.textures), m_border_mask(options.border_textures), m_shadow_mask(options.shadow_textures), m_depth_mask(options.depth_textures), m_logic_rop(options.logic_rop), m_blend_rop(options.blend_rop), m_output_rop(options.output_rop), m_half_storage_mask(options.half_storage_mask), m_float32_exports(options.float32_exports), m_raster_sample_count(options.raster_sample_count), m_depth_mirror(options.depth_mirror)
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
	std::vector<upload_requirement> result{{sizeof(fragment_context), 256}, {sizeof(fragment_inputs::sampling), 16}};
	if (m_texture_mask)
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
	if (m_library.device != ctx.device())
		throw std::invalid_argument("Metal 4 fragment constants belong to another device");
	return {m_constant_identity, ctx.constant_uploads().fragment(m_image, current_code)};
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
		validate_sampled_texture(ctx, inputs.textures[i], inputs.samplers[i], m_dimensions[i], inputs.sampling[i], m_shadow_mask & (1u << i), m_depth_mask & (1u << i));
		validate_texture_coordinates(inputs.coordinates[i]);
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
