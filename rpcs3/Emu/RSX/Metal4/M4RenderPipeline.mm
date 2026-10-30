#include "M4RenderPipeline.h"
#include "M4FragmentShader.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <exception>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace metal4
{
namespace
{
MTLCompareFunction compare(compare_operation value)
{
	switch (value)
	{
	case compare_operation::never:
		return MTLCompareFunctionNever;
	case compare_operation::less:
		return MTLCompareFunctionLess;
	case compare_operation::equal:
		return MTLCompareFunctionEqual;
	case compare_operation::less_equal:
		return MTLCompareFunctionLessEqual;
	case compare_operation::greater:
		return MTLCompareFunctionGreater;
	case compare_operation::not_equal:
		return MTLCompareFunctionNotEqual;
	case compare_operation::greater_equal:
		return MTLCompareFunctionGreaterEqual;
	case compare_operation::always:
		return MTLCompareFunctionAlways;
	}
	throw std::invalid_argument("Invalid native comparison function");
}

MTLStencilOperation operation(stencil_operation value)
{
	switch (value)
	{
	case stencil_operation::keep:
		return MTLStencilOperationKeep;
	case stencil_operation::zero:
		return MTLStencilOperationZero;
	case stencil_operation::replace:
		return MTLStencilOperationReplace;
	case stencil_operation::increment_clamp:
		return MTLStencilOperationIncrementClamp;
	case stencil_operation::decrement_clamp:
		return MTLStencilOperationDecrementClamp;
	case stencil_operation::invert:
		return MTLStencilOperationInvert;
	case stencil_operation::increment_wrap:
		return MTLStencilOperationIncrementWrap;
	case stencil_operation::decrement_wrap:
		return MTLStencilOperationDecrementWrap;
	}
	throw std::invalid_argument("Invalid native stencil operation");
}

MTLBlendFactor factor(blend_factor value)
{
	switch (value)
	{
	case blend_factor::zero:
		return MTLBlendFactorZero;
	case blend_factor::one:
		return MTLBlendFactorOne;
	case blend_factor::source_color:
		return MTLBlendFactorSourceColor;
	case blend_factor::one_minus_source_color:
		return MTLBlendFactorOneMinusSourceColor;
	case blend_factor::source_alpha:
		return MTLBlendFactorSourceAlpha;
	case blend_factor::one_minus_source_alpha:
		return MTLBlendFactorOneMinusSourceAlpha;
	case blend_factor::destination_color:
		return MTLBlendFactorDestinationColor;
	case blend_factor::one_minus_destination_color:
		return MTLBlendFactorOneMinusDestinationColor;
	case blend_factor::destination_alpha:
		return MTLBlendFactorDestinationAlpha;
	case blend_factor::one_minus_destination_alpha:
		return MTLBlendFactorOneMinusDestinationAlpha;
	case blend_factor::source_alpha_saturated:
		return MTLBlendFactorSourceAlphaSaturated;
	case blend_factor::constant_color:
		return MTLBlendFactorBlendColor;
	case blend_factor::one_minus_constant_color:
		return MTLBlendFactorOneMinusBlendColor;
	case blend_factor::constant_alpha:
		return MTLBlendFactorBlendAlpha;
	case blend_factor::one_minus_constant_alpha:
		return MTLBlendFactorOneMinusBlendAlpha;
	}
	throw std::invalid_argument("Invalid native blend factor");
}

MTLBlendOperation operation(blend_operation value)
{
	switch (value)
	{
	case blend_operation::add:
		return MTLBlendOperationAdd;
	case blend_operation::subtract:
		return MTLBlendOperationSubtract;
	case blend_operation::reverse_subtract:
		return MTLBlendOperationReverseSubtract;
	case blend_operation::minimum:
		return MTLBlendOperationMin;
	case blend_operation::maximum:
		return MTLBlendOperationMax;
	default:
		throw std::invalid_argument("RSX signed blending requires native shader ROP emulation");
	}
}

MTLStencilDescriptor* stencil_descriptor(const stencil_state& state)
{
	MTLStencilDescriptor* descriptor = [MTLStencilDescriptor new];
	descriptor.stencilCompareFunction = compare(state.compare);
	descriptor.stencilFailureOperation = operation(state.fail);
	descriptor.depthFailureOperation = operation(state.depth_fail);
	descriptor.depthStencilPassOperation = operation(state.pass);
	descriptor.readMask = state.read_mask;
	descriptor.writeMask = state.write_mask;
	return descriptor;
}

// Float fragment exports require a float/normalized color-renderable format.
// These formats support blending on the required Apple7+ family. Integer,
// compressed and depth formats need a different shader/attachment contract.
bool color_renderable(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatA8Unorm:
	case MTLPixelFormatR8Unorm:
	case MTLPixelFormatR8Unorm_sRGB:
	case MTLPixelFormatR8Snorm:
	case MTLPixelFormatR16Unorm:
	case MTLPixelFormatR16Snorm:
	case MTLPixelFormatR16Float:
	case MTLPixelFormatRG8Unorm:
	case MTLPixelFormatRG8Unorm_sRGB:
	case MTLPixelFormatRG8Snorm:
	case MTLPixelFormatB5G6R5Unorm:
	case MTLPixelFormatA1BGR5Unorm:
	case MTLPixelFormatABGR4Unorm:
	case MTLPixelFormatBGR5A1Unorm:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRG16Unorm:
	case MTLPixelFormatRG16Snorm:
	case MTLPixelFormatRG16Float:
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatRGBA8Unorm_sRGB:
	case MTLPixelFormatRGBA8Snorm:
	case MTLPixelFormatBGRA8Unorm:
	case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatRGB10A2Unorm:
	case MTLPixelFormatBGR10A2Unorm:
	case MTLPixelFormatRG11B10Float:
	case MTLPixelFormatRG32Float:
	case MTLPixelFormatRGBA16Unorm:
	case MTLPixelFormatRGBA16Snorm:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatRGBA32Float:
		return true;
	default:
		return false;
	}
}
std::array<std::uint32_t, 4> logic_bits(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatInvalid:
		return {};
	case MTLPixelFormatR8Unorm:
		return {8, 0, 0, 0};
	case MTLPixelFormatA8Unorm:
		return {0, 0, 0, 8};
	case MTLPixelFormatRG8Unorm:
		return {8, 8, 0, 0};
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatBGRA8Unorm:
		return {8, 8, 8, 8};
	case MTLPixelFormatR16Unorm:
		return {16, 0, 0, 0};
	case MTLPixelFormatRG16Unorm:
		return {16, 16, 0, 0};
	case MTLPixelFormatRGBA16Unorm:
		return {16, 16, 16, 16};
	case MTLPixelFormatB5G6R5Unorm:
		return {5, 6, 5, 0};
	case MTLPixelFormatBGR5A1Unorm:
		return {5, 5, 5, 1};
	case MTLPixelFormatA1BGR5Unorm:
		return {5, 5, 5, 1};
	case MTLPixelFormatABGR4Unorm:
		return {4, 4, 4, 4};
	case MTLPixelFormatRGB10A2Unorm:
	case MTLPixelFormatBGR10A2Unorm:
		return {10, 10, 10, 2};
	case MTLPixelFormatR16Float:
	case MTLPixelFormatRG16Float:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRG32Float:
	case MTLPixelFormatRGBA32Float:
	case MTLPixelFormatRG11B10Float:
	case MTLPixelFormatRGBA8Unorm_sRGB:
	case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatR8Unorm_sRGB:
	case MTLPixelFormatRG8Unorm_sRGB:
		return {}; // Vulkan floating-point/sRGB logic ops are passthrough.
	default:
		throw std::invalid_argument("Native shader logic attachment format is not implemented");
	}
}
logic_rop_configuration expected_logic(const render_pipeline_configuration& config)
{
	logic_rop_configuration result;
	if (!config.state.logic_enabled)
		return result;
	result.enabled = true;
	result.operation = config.state.logic_operation;
	result.sample_count = config.state.sample_count;
	for (std::size_t i = 0; i < 4; ++i)
		result.bits[i] = logic_bits(config.colors[i]);
	validate_logic_rop(result);
	return result;
}

bool blend_normalized(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatR16Float:
	case MTLPixelFormatRG16Float:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRG32Float:
	case MTLPixelFormatRGBA32Float:
		return false;
	default:
		if (logic_bits(format) == std::array<std::uint32_t, 4>{})
			throw std::invalid_argument("Native programmable blend format requires a separate encoding contract");
		return true;
	}
}

blend_rop_configuration expected_blend(const render_pipeline_configuration& config)
{
	const auto state = decode_render_state(config.state);
	blend_rop_configuration result;
	if (state.logic_enabled)
		return result;
	for (unsigned i = 0; i < 4; ++i)
	{
		if (!state.colors[i].enabled || (config.native_blend_mask & (1u << i)))
			continue;
		if (config.colors[i] == MTLPixelFormatInvalid)
			throw std::invalid_argument("Native programmable blending addresses an unattached target");
		result.colors[i] = state.colors[i];
		result.colors[i].write_mask = 15;
		result.normalized[i] = blend_normalized(config.colors[i]);
		if (result.normalized[i])
			result.bits[i] = logic_bits(config.colors[i]);
		result.half_float[i] = config.colors[i] == MTLPixelFormatR16Float || config.colors[i] == MTLPixelFormatRG16Float || config.colors[i] == MTLPixelFormatRGBA16Float;
		result.remap[i] = config.color_remap[i];
	}
	if (result.enabled())
		result.sample_count = state.sample_count;
	validate_blend_rop(result);
	return result;
}

bool signed_blend(const blend_state& state)
{
	return state.enabled && (state.operation_rgb >= blend_operation::reverse_subtract_signed ||
							 state.operation_alpha >= blend_operation::reverse_subtract_signed);
}
bool selection_blend(const blend_state& state)
{
	const auto selection = [](blend_operation op) { return op == blend_operation::minimum || op == blend_operation::maximum; };
	return selection(state.operation_rgb) && selection(state.operation_alpha);
}
bool native_export_eligible(const render_pipeline_configuration& config, const render_state& state, unsigned slot, bool float32)
{
	if (float32) return true;
	// A half-rounded source selected against a stored half value is already
	// half. For UNORM8, converting a destination code to half and back cannot
	// cross its quantization boundary. Float32 destinations lack that property.
	return selection_blend(state.colors[slot]) && (config.colors[slot] == MTLPixelFormatRGBA16Float ||
		config.colors[slot] == MTLPixelFormatBGRA8Unorm || config.colors[slot] == MTLPixelFormatRGBA8Unorm);
}
bool native_blend_eligible(const render_pipeline_configuration& config, const render_state& state, unsigned slot)
{
	if (state.logic_enabled || !state.colors[slot].enabled || signed_blend(state.colors[slot]) ||
		(config.shader_blend_mask & (1u << slot)) || config.color_remap[slot] != color_export_remap::rgba ||
		config.state.output_rop.round_to_8bit)
		return false;
	switch (config.colors[slot])
	{
	case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatRGBA8Unorm:
	{
		// RSX saturates the source product, whereas native normalized blending
		// clamps the source before evaluating factors. Keep source-dependent
		// products/factors and dynamic blend constants in the programmable path.
		const auto channel = [](blend_operation op, blend_factor source, blend_factor destination)
		{
			if (op == blend_operation::minimum || op == blend_operation::maximum) return true;
			return (source == blend_factor::zero || source == blend_factor::one) &&
				(destination == blend_factor::zero || destination == blend_factor::one ||
				 destination == blend_factor::destination_color || destination == blend_factor::one_minus_destination_color ||
				 destination == blend_factor::destination_alpha || destination == blend_factor::one_minus_destination_alpha);
		};
		const auto& color = state.colors[slot];
		return channel(color.operation_rgb, color.source_rgb, color.destination_rgb) &&
			channel(color.operation_alpha, color.source_alpha, color.destination_alpha);
	}
	case MTLPixelFormatRGBA16Float:
	{
		// Arithmetic fixed-function half stores differ from the explicit RSX
		// round-to-nearest export across successive primitives. For min/max,
		// rounding the source commutes with selection of an already-half dest.
		return selection_blend(state.colors[slot]);
	}
	case MTLPixelFormatR32Float: case MTLPixelFormatRGBA32Float:
		return true;
	default: return false;
	}
}
} // namespace

void configure_logic_operations(render_pipeline_configuration& config, fragment_program_options& fragment)
{
	if (config.state.logic_enabled) config.native_blend_mask = 0;
	configure_output_rops(config, fragment);
	(void)decode_render_state(config.state);
	config.shader_logic = expected_logic(config);
	fragment.logic_rop = config.shader_logic;
	if (config.state.logic_enabled)
	{
		config.shader_blend = {};
		fragment.blend_rop = {};
	}
	for (unsigned i = 0; i < 4; ++i)
		if (config.colors[i] != MTLPixelFormatInvalid)
			fragment.color_count = std::max(fragment.color_count, i + 1);
}

void configure_shader_blending(render_pipeline_configuration& config, fragment_program_options& fragment)
{
	config.native_blend_mask = 0;
	configure_output_rops(config, fragment);
	config.shader_blend = expected_blend(config);
	fragment.blend_rop = config.shader_blend;
	for (unsigned i = 0; i < 4; ++i)
		if (config.shader_blend.colors[i].enabled)
		{
			fragment.color_count = std::max(fragment.color_count, i + 1);
			fragment.color_remap[i] = config.color_remap[i];
		}
}

void configure_blending(render_pipeline_configuration& config, fragment_program_options& fragment)
{
	config.native_blend_mask = 0;
	const auto state = decode_render_state(config.state);
	for (unsigned i = 0; i < 4; ++i)
		if (native_blend_eligible(config, state, i) && native_export_eligible(config, state, i, fragment.float32_exports)) config.native_blend_mask |= 1u << i;
	configure_output_rops(config, fragment);
	config.shader_blend = expected_blend(config);
	fragment.blend_rop = config.shader_blend;
	fragment.color_remap = config.color_remap;
}

void configure_output_rops(render_pipeline_configuration& config, fragment_program_options& fragment)
{
	config.shader_output = config.state.output_rop;
	fragment.output_rop = config.shader_output;
	fragment.raster_sample_count = config.state.sample_count;
	config.shader_half_storage_mask = 0;
	for (unsigned i = 0; i < 4; ++i)
	{
		if (config.colors[i] == MTLPixelFormatR16Float || config.colors[i] == MTLPixelFormatRG16Float || config.colors[i] == MTLPixelFormatRGBA16Float)
			config.shader_half_storage_mask |= 1u << i;
		if (config.colors[i] != MTLPixelFormatInvalid)
			fragment.color_count = std::max(fragment.color_count, i + 1);
	}
	fragment.half_storage_mask = config.shader_half_storage_mask;
}

MTL4RenderPipelineDescriptor* make_render_descriptor(const render_pipeline_configuration& config,
													 MTL4FunctionDescriptor* vertex, MTL4FunctionDescriptor* fragment)
{
	const auto state = decode_render_state(config.state);
	if (config.native_blend_mask & ~15u) throw std::invalid_argument("Invalid native blend selection mask");
	for (unsigned i = 0; i < 4; ++i)
		if ((config.native_blend_mask & (1u << i)) && !native_blend_eligible(config, state, i))
			throw std::invalid_argument("Native blend selection violates RSX format/ROP semantics");
	if (config.shader_output != config.state.output_rop)
		throw std::invalid_argument("Native shader output ROP contract does not match guest pipeline state");
	validate_depth_mirror(config.depth_mirror);
	if (config.depth_mirror.enabled() && (!config.rasterization || config.depth_mirror.write != state.depth_write))
		throw std::invalid_argument("Native mirror writes differ from effective depth write state");
	if (config.shader_half_storage_mask & ~15u)
		throw std::invalid_argument("Invalid native half storage export mask");
	std::uint32_t expected_half_mask = 0;
	for (unsigned i = 0; i < 4; ++i)
		if (config.colors[i] == MTLPixelFormatR16Float || config.colors[i] == MTLPixelFormatRG16Float || config.colors[i] == MTLPixelFormatRGBA16Float)
			expected_half_mask |= 1u << i;
	if (config.shader_half_storage_mask != expected_half_mask)
		throw std::invalid_argument("Native half attachments require matching explicit shader exports");
	for (unsigned i = 0; i < 4; ++i)
		if ((config.shader_half_storage_mask & (1u << i)) && config.colors[i] != MTLPixelFormatR16Float && config.colors[i] != MTLPixelFormatRG16Float && config.colors[i] != MTLPixelFormatRGBA16Float)
			throw std::invalid_argument("Native half storage export targets a different format");
	if (config.shader_output.polygon_stipple && config.topology != MTLPrimitiveTopologyClassTriangle)
		throw std::invalid_argument("RSX polygon stipple requires a triangle pipeline");
	if (config.shader_output.srgb)
		for (auto format : config.colors)
			if (format == MTLPixelFormatR8Unorm_sRGB || format == MTLPixelFormatRG8Unorm_sRGB || format == MTLPixelFormatRGBA8Unorm_sRGB || format == MTLPixelFormatBGRA8Unorm_sRGB)
				throw std::invalid_argument("RSX shader sRGB conversion requires an unencoded native attachment");
	if (config.shader_logic.enabled && config.shader_logic != expected_logic(config))
		throw std::invalid_argument("Native shader logic contract does not match pipeline formats/state/samples");
	if (config.shader_blend.enabled() && config.shader_blend != expected_blend(config))
		throw std::invalid_argument("Native shader blend contract does not match pipeline formats/state/samples/remaps");
	if (config.shader_blend_mask & ~15u)
		throw std::invalid_argument("Invalid native shader blend mask");
	for (unsigned i = 0; i < 4; ++i)
		if (((config.shader_blend_mask & (1u << i)) || signed_blend(state.colors[i])) && state.colors[i].enabled && !config.shader_blend.colors[i].enabled)
			throw std::invalid_argument("RSX surface blending requires shader ROP semantics");
	if (state.logic_enabled && !config.shader_logic.enabled)
		throw std::invalid_argument("RSX logic/signed blend state requires native shader ROP emulation");
	if (!vertex || (config.rasterization && !fragment) ||
		(config.topology != MTLPrimitiveTopologyClassPoint && config.topology != MTLPrimitiveTopologyClassLine &&
		 config.topology != MTLPrimitiveTopologyClassTriangle && config.topology != MTLPrimitiveTopologyClassUnspecified))
		throw std::invalid_argument("Metal 4 render pipeline functions or topology are invalid");
	MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
	descriptor.vertexFunctionDescriptor = vertex;
	descriptor.fragmentFunctionDescriptor = fragment;
	descriptor.rasterizationEnabled = config.rasterization;
	descriptor.inputPrimitiveTopology = config.topology;
	descriptor.rasterSampleCount = state.sample_count;
	descriptor.alphaToCoverageState = state.alpha_to_coverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
	descriptor.alphaToOneState = state.alpha_to_one ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
	descriptor.colorAttachmentMappingState = MTL4LogicalToPhysicalColorAttachmentMappingStateIdentity;
	for (NSUInteger i = 0; i < config.colors.size(); ++i)
	{
		const auto format = config.colors[i];
		auto target = descriptor.colorAttachments[i];
		if (format == MTLPixelFormatInvalid)
		{
			target.writeMask = MTLColorWriteMaskNone;
			continue;
		}
		if (!color_renderable(format))
			throw std::invalid_argument("Metal 4 color attachment format is incompatible with float exports");
		const auto& source = state.colors[i];
		target.pixelFormat = format;
		const bool native_blend = source.enabled && !config.shader_blend.colors[i].enabled;
		target.blendingState = native_blend ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
		if (native_blend)
		{
			target.sourceRGBBlendFactor = factor(source.source_rgb);
			target.destinationRGBBlendFactor = factor(source.destination_rgb);
			target.sourceAlphaBlendFactor = factor(source.source_alpha);
			target.destinationAlphaBlendFactor = factor(source.destination_alpha);
			target.rgbBlendOperation = operation(source.operation_rgb);
			target.alphaBlendOperation = operation(source.operation_alpha);
		}
		target.writeMask = static_cast<MTLColorWriteMask>(
			((source.write_mask & 1) ? MTLColorWriteMaskRed : 0) |
			((source.write_mask & 2) ? MTLColorWriteMaskGreen : 0) |
			((source.write_mask & 4) ? MTLColorWriteMaskBlue : 0) |
			((source.write_mask & 8) ? MTLColorWriteMaskAlpha : 0));
	}
	if (config.depth_mirror.enabled())
	{
		descriptor.colorAttachments[4].pixelFormat = MTLPixelFormatR32Float;
		descriptor.colorAttachments[4].blendingState = MTL4BlendStateDisabled;
		descriptor.colorAttachments[4].writeMask = config.depth_mirror.write ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
	}
	return descriptor;
}

MTLDepthStencilDescriptor* make_depth_stencil_descriptor(const guest_render_state& guest)
{
	const auto state = decode_render_state(guest);
	MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
	descriptor.depthCompareFunction = compare(state.depth_compare);
	descriptor.depthWriteEnabled = state.depth_write;
	descriptor.frontFaceStencil = stencil_descriptor(state.front);
	descriptor.backFaceStencil = stencil_descriptor(state.back);
	return descriptor;
}

static render_pipeline_bundle prepare_pipeline_state(const compiler& compiler, const render_pipeline_configuration& config)
{
	const auto depth_descriptor = make_depth_stencil_descriptor(config.state);
	const auto device = compiler.device();
	if (![device supportsTextureSampleCount:config.state.sample_count])
		throw std::invalid_argument("Apple GPU does not support the requested raster sample count");
	for (auto format : config.colors)
	{
		if (config.state.sample_count > 1 && !device.supports32BitMSAA &&
			(format == MTLPixelFormatR32Float || format == MTLPixelFormatRG32Float || format == MTLPixelFormatRGBA32Float))
			throw std::invalid_argument("Apple GPU does not support 32-bit float multisampling");
	}
	const auto depth_stencil = [device newDepthStencilStateWithDescriptor:depth_descriptor];
	if (!depth_stencil)
		throw std::runtime_error("Native depth/stencil state creation failed");
	return {nil, depth_stencil, decode_render_state(config.state), config.depth_mirror};
}
static render_pipeline_bundle compile_pipeline(const compiler& compiler, const render_pipeline_configuration& config,
	MTL4FunctionDescriptor* vertex, MTL4FunctionDescriptor* fragment)
{
	const auto descriptor = make_render_descriptor(config, vertex, fragment);
	auto result = prepare_pipeline_state(compiler, config);
	result.pipeline = compiler.render(descriptor);
	return result;
}

render_pipeline_bundle make_render_pipeline(const compiler& compiler, const render_pipeline_configuration& config,
											MTL4FunctionDescriptor* vertex, MTL4FunctionDescriptor* fragment)
{
	if (config.native_blend_mask || config.shader_logic.enabled || config.shader_blend.enabled() || config.shader_output.enabled() || config.shader_half_storage_mask || config.depth_mirror.enabled() || (config.rasterization && config.state.sample_count > 1))
		throw std::invalid_argument("Native shader ROP pipeline requires a verified fragment_shader contract");
	return compile_pipeline(compiler, config, vertex, fragment);
}
static void validate_fragment_contract(const compiler& compiler, const render_pipeline_configuration& config, const fragment_shader& fragment)
{
	const auto state = decode_render_state(config.state);
	for (unsigned i = 0; i < 4; ++i)
		if ((config.native_blend_mask & (1u << i)) && !native_export_eligible(config, state, i, fragment.float32_exports()))
			throw std::invalid_argument("Native RSX blend selection changes fragment export precision");
	if (fragment.logic_rop() != config.shader_logic)
		throw std::invalid_argument("Compiled fragment logic ROP differs from pipeline contract");
	if (fragment.blend_rop() != config.shader_blend)
		throw std::invalid_argument("Compiled fragment blend ROP differs from pipeline contract");
	if (fragment.output_rop() != config.shader_output)
		throw std::invalid_argument("Compiled fragment output ROP differs from pipeline contract");
	if (fragment.half_storage_mask() != config.shader_half_storage_mask)
		throw std::invalid_argument("Compiled fragment half storage exports differ from pipeline contract");
	if (fragment.raster_sample_count() != config.state.sample_count)
		throw std::invalid_argument("Compiled fragment shading sample count differs from pipeline contract");
	if (fragment.depth_mirror() != config.depth_mirror)
		throw std::invalid_argument("Compiled stored-depth mirror differs from pipeline contract");
	if ((config.shader_logic.enabled || config.shader_blend.enabled() || config.depth_mirror.enabled()) && !compiler.device().rasterOrderGroupsSupported)
		throw std::invalid_argument("Native shader ROP requires Apple raster order groups");
}

render_pipeline_bundle make_render_pipeline(const compiler& compiler, const render_pipeline_configuration& config,
											MTL4FunctionDescriptor* vertex, const fragment_shader& fragment)
{
	validate_fragment_contract(compiler, config, fragment);
	return compile_pipeline(compiler, config, vertex, fragment.function());
}

struct render_pipeline_cache::implementation
{
	struct key
	{
		id<MTLLibrary> vertex_library, fragment_library;
		NSString* vertex_name;
		NSString* fragment_name;
		std::array<NSUInteger, 5> raster{};
		std::array<std::array<NSUInteger, 9>, 5> colors{};
		render_state depth;
		std::size_t fingerprint() const
		{
			std::size_t hash = 14695981039346656037ull;
			const auto add = [&](std::size_t value) { hash = (hash ^ value) * 1099511628211ull; };
			add(reinterpret_cast<std::uintptr_t>((__bridge void*)vertex_library));
			add(reinterpret_cast<std::uintptr_t>((__bridge void*)fragment_library));
			add(vertex_name.hash); add(fragment_name.hash);
			for (auto value : raster) add(value);
			for (const auto& color : colors) for (auto value : color) add(value);
			// Depth is checked in full equality; different depth states may share
			// a bucket, and can never substitute for one another.
			return hash;
		}
		bool functions(MTL4LibraryFunctionDescriptor* vertex, MTL4LibraryFunctionDescriptor* fragment) const
		{
			return vertex_library == vertex.library && fragment_library == fragment.library &&
				[vertex_name isEqualToString:vertex.name] && [fragment_name isEqualToString:fragment.name];
		}
		bool operator==(const key& other) const
		{
			return vertex_library == other.vertex_library && fragment_library == other.fragment_library &&
				   [vertex_name isEqualToString:other.vertex_name] && [fragment_name isEqualToString:other.fragment_name] &&
				   raster == other.raster && colors == other.colors && depth == other.depth;
		}
	};
	struct entry
	{
		key description;
		render_pipeline_configuration validated;
		render_pipeline_bundle bundle;
	};
	compiler compile;
	std::list<entry> entries;
	std::unordered_multimap<std::size_t, std::list<entry>::iterator> index;
	void erase_index(std::list<entry>::iterator entry)
	{
		const auto [first, last] = index.equal_range(entry->description.fingerprint());
		for (auto it = first; it != last; ++it)
			if (it->second == entry) { index.erase(it); return; }
	}
	render_pipeline_cache_statistics stats;
	std::shared_ptr<void> identity = std::make_shared<int>(0);
	std::uint64_t generation = 0;
	std::vector<std::weak_ptr<preparation>> pending;
	static key describe(MTL4RenderPipelineDescriptor* descriptor, MTL4LibraryFunctionDescriptor* vp,
		MTL4LibraryFunctionDescriptor* fp, const render_state& decoded)
	{
		key result;
		result.vertex_library = vp.library; result.fragment_library = fp.library;
		result.vertex_name = [vp.name copy]; result.fragment_name = [fp.name copy];
		result.raster = {descriptor.rasterSampleCount, static_cast<NSUInteger>(descriptor.alphaToCoverageState), static_cast<NSUInteger>(descriptor.alphaToOneState),
			static_cast<NSUInteger>(descriptor.rasterizationEnabled), descriptor.inputPrimitiveTopology};
		for (NSUInteger i = 0; i < result.colors.size(); ++i)
		{
			const auto color = descriptor.colorAttachments[i];
			result.colors[i] = {color.pixelFormat, static_cast<NSUInteger>(color.blendingState), color.sourceRGBBlendFactor, color.destinationRGBBlendFactor,
				color.rgbBlendOperation, color.sourceAlphaBlendFactor, color.destinationAlphaBlendFactor, color.alphaBlendOperation, color.writeMask};
		}
		result.depth.depth_compare = decoded.depth_compare; result.depth.depth_write = decoded.depth_write;
		result.depth.front = decoded.front; result.depth.back = decoded.back;
		result.depth.front.reference = result.depth.back.reference = 0;
		return result;
	}
	explicit implementation(const compiler& value) : compile(value) {}
};

struct render_pipeline_cache::preparation
{
	implementation::key description;
	render_pipeline_configuration validated;
	render_pipeline_bundle bundle;
	std::optional<compiler_job> job;
	std::exception_ptr error;
	std::uint64_t generation = 0;
	bool adopted = false;
};

bool render_pipeline_cache::prepared_pipeline::ready() const { return !m_entry || m_entry->error || m_entry->bundle.pipeline || m_entry->job->ready(); }

render_pipeline_cache::render_pipeline_cache(const compiler& compiler, std::size_t capacity)
{
	if (capacity > 4096)
		throw std::invalid_argument("Metal4 render pipeline cache capacity exceeds 4096");
	m_impl = std::make_unique<implementation>(compiler);
	m_impl->stats.capacity = capacity;
}
render_pipeline_cache::~render_pipeline_cache() = default;

render_pipeline_bundle render_pipeline_cache::get(const render_pipeline_configuration& config,
												  MTL4FunctionDescriptor* vertex, const fragment_shader& fragment)
{
	// Validate on hits too. Never let a previously valid native state conceal an
	// incompatible current shader ROP, format or sample contract.
	validate_fragment_contract(m_impl->compile, config, fragment);
	const auto fragment_function = fragment.function();
	const auto decoded = decode_render_state(config.state);
	auto normalized = config;
	normalized.state.front.reference = normalized.state.back.reference = 0;
	if ([vertex class] != [MTL4LibraryFunctionDescriptor class] ||
		[fragment_function class] != [MTL4LibraryFunctionDescriptor class])
	{
		auto result = compile_pipeline(m_impl->compile, config, vertex, fragment_function);
		++m_impl->stats.created;
		++m_impl->stats.bypassed;
		return result;
	}
	const auto vp = static_cast<MTL4LibraryFunctionDescriptor*>(vertex);
	const auto fp = static_cast<MTL4LibraryFunctionDescriptor*>(fragment_function);
	// Exact immutable configuration + retained native library/name identity is
	// a proof of the descriptor validated on the previous successful lookup.
	// Dynamic references are refreshed from decoded state even on this path.
	if (!m_impl->entries.empty())
	{
		const auto& entry = m_impl->entries.front();
		if (entry.validated == normalized && entry.description.functions(vp, fp))
		{
			auto result = entry.bundle;
			result.state = decoded;
			result.depth_mirror = config.depth_mirror;
			++m_impl->stats.fast_hits;
			++m_impl->stats.reused;
			return result;
		}
	}
	const auto descriptor = make_render_descriptor(config, vertex, fragment_function);
	++m_impl->stats.descriptor_lookups;
	for (auto function : {vp, fp})
		if (!function.library || function.library.device != m_impl->compile.device() ||
			!function.name.length || ![function.library.functionNames containsObject:function.name])
			throw std::invalid_argument("Metal4 cached pipeline requires an existing entry on its compiler device");
	implementation::key key = implementation::describe(descriptor, vp, fp, decoded);
	const auto fingerprint = key.fingerprint();
	const auto [first, last] = m_impl->index.equal_range(fingerprint);
	for (auto candidate = first; candidate != last; ++candidate)
	{
		++m_impl->stats.candidates;
		const auto it = candidate->second;
		if (it->description == key)
		{
			auto result = it->bundle;
			result.state = decoded;
			result.depth_mirror = config.depth_mirror;
			it->validated = normalized;
			m_impl->entries.splice(m_impl->entries.begin(), m_impl->entries, it);
			++m_impl->stats.reused;
			return result;
		}
	}
	for (const auto& weak : m_impl->pending)
		if (const auto pending = weak.lock(); pending && !pending->adopted && !pending->error && pending->description == key)
		{
			++m_impl->stats.pending_reused;
			return get(prepared_pipeline(m_impl->identity, pending, decoded, config.depth_mirror));
		}
	auto result = compile_pipeline(m_impl->compile, config, vertex, fragment_function);
	if (!m_impl->stats.capacity) { ++m_impl->stats.created; ++m_impl->stats.bypassed; return result; }
	// Allocate before eviction so a failed compile/allocation preserves old PSOs.
	m_impl->entries.push_front({std::move(key), normalized, result});
	try { m_impl->index.emplace(fingerprint, m_impl->entries.begin()); }
	catch (...) { m_impl->entries.pop_front(); throw; }
	++m_impl->stats.created;
	if (m_impl->entries.size() > m_impl->stats.capacity)
	{
		m_impl->erase_index(std::prev(m_impl->entries.end()));
		m_impl->entries.pop_back();
		++m_impl->stats.evicted;
	}
	m_impl->stats.entries = m_impl->entries.size();
	return result;
}

std::optional<render_pipeline_cache::prepared_pipeline> render_pipeline_cache::prewarm(const render_pipeline_configuration& config,
	MTL4FunctionDescriptor* vertex, const fragment_shader& fragment)
{
	validate_fragment_contract(m_impl->compile, config, fragment);
	const auto fp = fragment.function();
	if ([vertex class] != [MTL4LibraryFunctionDescriptor class] || [fp class] != [MTL4LibraryFunctionDescriptor class]) return {};
	const auto vp = static_cast<MTL4LibraryFunctionDescriptor*>(vertex);
	const auto decoded = decode_render_state(config.state);
	auto normalized = config; normalized.state.front.reference = normalized.state.back.reference = 0;
	if (!m_impl->entries.empty() && m_impl->entries.front().validated == normalized && m_impl->entries.front().description.functions(vp, fp))
	{
		++m_impl->stats.fast_hits; ++m_impl->stats.reused;
		return prepared_pipeline(m_impl->identity, m_impl->entries.front().bundle, decoded, config.depth_mirror);
	}
	const auto descriptor = make_render_descriptor(config, vertex, fp);
	++m_impl->stats.descriptor_lookups;
	for (auto function : {vp, fp})
		if (!function.library || function.library.device != m_impl->compile.device() || !function.name.length ||
			![function.library.functionNames containsObject:function.name]) throw std::invalid_argument("Invalid native pipeline prewarm function");
	const auto key = implementation::describe(descriptor, vp, fp, decoded);
	const auto [first, last] = m_impl->index.equal_range(key.fingerprint());
	for (auto candidate = first; candidate != last; ++candidate)
		if (candidate->second->description == key)
		{
			const auto ready = candidate->second->bundle;
			candidate->second->validated = normalized;
			m_impl->entries.splice(m_impl->entries.begin(), m_impl->entries, candidate->second);
			++m_impl->stats.reused;
			return prepared_pipeline(m_impl->identity, ready, decoded, config.depth_mirror);
		}
	std::erase_if(m_impl->pending, [](const auto& weak) { const auto entry = weak.lock(); return !entry || entry->adopted || entry->error; });
	for (const auto& weak : m_impl->pending)
		if (const auto pending = weak.lock(); pending && pending->description == key)
		{
			++m_impl->stats.pending_reused;
			return prepared_pipeline(m_impl->identity, pending, decoded, config.depth_mirror);
		}
	if (m_impl->pending.size() == 64) { ++m_impl->stats.prewarm_refused; return {}; }
	auto pending = std::make_shared<preparation>();
	pending->description = key; pending->validated = normalized; pending->generation = m_impl->generation;
	pending->bundle = prepare_pipeline_state(m_impl->compile, config);
	pending->job = m_impl->compile.tasks().render(descriptor);
	if (!pending->job) { ++m_impl->stats.prewarm_refused; return {}; }
	m_impl->pending.push_back(pending); ++m_impl->stats.prewarmed;
	return prepared_pipeline(m_impl->identity, std::move(pending), decoded, config.depth_mirror);
}

render_pipeline_bundle render_pipeline_cache::get(const prepared_pipeline& prepared, std::chrono::milliseconds timeout)
{
	if (prepared.m_owner != m_impl->identity || timeout.count() < 0) throw std::invalid_argument("Foreign prepared native pipeline or timeout");
	if (!prepared.m_entry) { auto result = prepared.m_ready; result.state = prepared.m_state; result.depth_mirror = prepared.m_mirror; return result; }
	const auto& entry = prepared.m_entry;
	if (entry->error) std::rethrow_exception(entry->error);
	if (!entry->bundle.pipeline)
	{
		if (!entry->job->ready()) ++m_impl->stats.prewarm_waited;
		try { entry->bundle.pipeline = entry->job->render(timeout); }
		catch (...)
		{
			if (entry->job->ready())
				try { (void)entry->job->render(std::chrono::milliseconds(0)); }
				catch (...) { entry->error = std::current_exception(); entry->job.reset(); }
			throw;
		}
		entry->job.reset(); ++m_impl->stats.created;
	}
	if (!entry->adopted && entry->generation == m_impl->generation && m_impl->stats.capacity)
	{
		const auto fingerprint = entry->description.fingerprint();
		m_impl->entries.push_front({entry->description, entry->validated, entry->bundle});
		try { m_impl->index.emplace(fingerprint, m_impl->entries.begin()); }
		catch (...) { m_impl->entries.pop_front(); throw; }
		if (m_impl->entries.size() > m_impl->stats.capacity)
		{
			m_impl->erase_index(std::prev(m_impl->entries.end())); m_impl->entries.pop_back(); ++m_impl->stats.evicted;
		}
		m_impl->stats.entries = m_impl->entries.size(); entry->adopted = true;
	}
	else if (!entry->adopted) { ++m_impl->stats.bypassed; entry->adopted = true; }
	auto result = entry->bundle; result.state = prepared.m_state; result.depth_mirror = prepared.m_mirror;
	return result;
}
render_pipeline_cache_statistics render_pipeline_cache::statistics() const noexcept
{
	return m_impl->stats;
}
void render_pipeline_cache::clear()
{
	++m_impl->generation; m_impl->pending.clear();
	m_impl->stats.evicted += m_impl->entries.size();
	m_impl->index.clear();
	m_impl->entries.clear();
	m_impl->stats.entries = 0;
}

void render_pipeline_bundle::configure(render_commands& commands, raster_state raster,
									   const bindings& vertex, const bindings& fragment) const
{
	raster.front_stencil_reference = state.front.reference;
	raster.back_stencil_reference = state.back.reference;
	commands.configure(pipeline, depth_stencil, raster, vertex, fragment);
}
} // namespace metal4
