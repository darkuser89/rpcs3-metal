#include "M4DepthBoundsPass.h"
#include "M4DrawValidation.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace metal4
{
namespace
{
depth_mirror_encoding encoding(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatDepth16Unorm:
		return depth_mirror_encoding::unorm16;
	case MTLPixelFormatDepth32Float:
	case MTLPixelFormatDepth32Float_Stencil8:
		return depth_mirror_encoding::float32;
	default:
		throw std::invalid_argument("Stored-depth bounds require a supported native depth format");
	}
}
std::string initialization_shader(bool multisampled)
{
	std::string source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct depth_mirror_vertex { float4 position [[position]]; };
vertex depth_mirror_vertex depth_mirror_init_vertex(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    return {float4(positions[vertex_id], 0, 1)};
}
fragment float4 depth_mirror_init_fragment(depth_mirror_vertex in [[stage_in]],
)metal";
	source += multisampled ? "    texture2d_ms<float, access::read> depth [[texture(0)]], uint sample [[sample_id]]) {\n    return float4(depth.read(uint2(in.position.xy), sample).r);\n}\n" : "    texture2d<float, access::read> depth [[texture(0)]]) {\n    return float4(depth.read(uint2(in.position.xy)).r);\n}\n";
	return source;
}
} // namespace

void configure_depth_bounds_pipeline(render_pipeline_configuration& pipeline, fragment_program_options& fragment, MTLPixelFormat depth)
{
	const auto state = decode_render_state(pipeline.state);
	pipeline.depth_mirror = {encoding(depth), state.depth_write};
	fragment.depth_mirror = pipeline.depth_mirror;
	configure_output_rops(pipeline, fragment);
}

depth_bounds_engine::depth_bounds_engine(context& ctx) : m_compiler(ctx), m_device(ctx.device())
{
	if (!m_device.rasterOrderGroupsSupported)
		throw std::invalid_argument("Native depth mirror requires Raster Order Groups");
	m_single = m_compiler.library(initialization_shader(false), @"RPCS3 native depth mirror initialization");
	m_multisampled = m_compiler.library(initialization_shader(true), @"RPCS3 native per-sample depth mirror initialization");
}
id<MTLRenderPipelineState> depth_bounds_engine::initializer(std::uint32_t samples)
{
	if (samples != 1 && samples != 2 && samples != 4 && samples != 8)
		throw std::invalid_argument("Invalid native depth mirror sample count");
	if (![m_device supportsTextureSampleCount:samples] || (samples > 1 && !m_device.supports32BitMSAA))
		throw std::invalid_argument("Native device cannot multisample the depth mirror");
	if (m_pipelines[samples])
		return m_pipelines[samples];
	auto library = samples == 1 ? m_single : m_multisampled;
	helper_render_configuration config;
	config.samples = samples;
	config.colors[0] = MTLPixelFormatR32Float;
	m_pipelines[samples] = m_compiler.render(library, @"depth_mirror_init_vertex", @"depth_mirror_init_fragment", config);
	return m_pipelines[samples];
}

depth_bounds_pass depth_bounds_engine::prepare(context& ctx, MTL4RenderPassDescriptor* original, const surface_lease& reserved)
{
	if (ctx.device() != m_device || !original || !original.depthAttachment.texture)
		throw std::invalid_argument("Native depth bounds pass requires an actual depth attachment");
	auto depth = original.depthAttachment.texture;
	const auto type = encoding(depth.pixelFormat);
	if (depth.device != m_device || !(depth.usage & MTLTextureUsageShaderRead) || depth.framebufferOnly || depth.storageMode == MTLStorageModeMemoryless || original.renderTargetArrayLength > 1 || original.depthAttachment.depthPlane)
		throw std::invalid_argument("Native depth mirror device/readability/layered contract is invalid");
	if (depth.textureType != MTLTextureType2D && depth.textureType != MTLTextureType2DArray && depth.textureType != MTLTextureType2DMultisample)
		throw std::invalid_argument("Native depth mirror requires a single 2D attachment subresource");
	for (unsigned i = 4; i < 8; ++i)
		if (original.colorAttachments[i].texture)
			throw std::invalid_argument("Depth mirror reserves native color attachments above logical RSX colors");
	if (original.depthAttachment.loadAction != MTLLoadActionLoad && original.depthAttachment.loadAction != MTLLoadActionClear)
		throw std::invalid_argument("Stored-depth bounds cannot assume defined DontCare attachment contents");
	const auto samples = static_cast<std::uint32_t>(depth.sampleCount);
	const auto pipeline = initializer(samples);
	texture_region selected;
	selected.level = original.depthAttachment.level;
	selected.slice = original.depthAttachment.slice;
	const auto dimensions = validate_texture_region(ctx, depth, selected, texture_plane::depth);
	const std::size_t width = dimensions[0], height = dimensions[1];
	if (reserved && reserved->description() != surface_allocation_description{width, height, samples, surface_color::r32f, false})
		throw std::invalid_argument("Reserved depth mirror differs from the attachment dimensions/sample count/format");
	id<MTLTexture> mirror;
	if (reserved)
	{
		mirror = reserved->render_texture(ctx);
		if (mirror.device != m_device)
			throw std::invalid_argument("Reserved depth mirror belongs to another device");
	}
	else
	{
		auto texture_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:width height:height mipmapped:NO];
		texture_descriptor.storageMode = MTLStorageModePrivate;
		texture_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		texture_descriptor.sampleCount = samples;
		if (samples > 1)
			texture_descriptor.textureType = MTLTextureType2DMultisample;
		mirror = make_texture(ctx, texture_descriptor);
	}
	MTL4RenderPassDescriptor* descriptor = [original copy];
	const bool clear_depth = descriptor.depthAttachment.loadAction == MTLLoadActionClear;
	const bool clear_stencil = descriptor.stencilAttachment.texture && descriptor.stencilAttachment.loadAction == MTLLoadActionClear;
	if (clear_depth || clear_stencil)
	{
		// Clear real depth first. In particular D16 native clear conversion must
		// not be replaced by a guessed float value in the mirror.
		auto clear = [MTL4RenderPassDescriptor new];
		clear.depthAttachment = descriptor.depthAttachment;
		clear.depthAttachment.storeAction = MTLStoreActionStore;
		if (descriptor.stencilAttachment.texture)
		{
			clear.stencilAttachment = descriptor.stencilAttachment;
			clear.stencilAttachment.storeAction = MTLStoreActionStore;
		}
		(void)ctx.render(clear);
		ctx.end_encoding();
		descriptor.depthAttachment.loadAction = MTLLoadActionLoad;
		if (clear_stencil)
			descriptor.stencilAttachment.loadAction = MTLLoadActionLoad;
	}
	id<MTLTexture> source = depth;
	if (selected.level || selected.slice || depth.textureType == MTLTextureType2DArray)
	{
		const auto swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
		source = make_texture_view(depth, depth.pixelFormat, MTLTextureType2D, NSMakeRange(selected.level, 1), NSMakeRange(selected.slice, 1), swizzle);
	}
	auto initialize = [MTL4RenderPassDescriptor new];
	initialize.colorAttachments[0].texture = mirror;
	initialize.colorAttachments[0].loadAction = MTLLoadActionDontCare;
	initialize.colorAttachments[0].storeAction = MTLStoreActionStore;
	bindings inputs;
	inputs.set_texture(0, source);
	render_commands draw(ctx, initialize);
	raster_state raster;
	raster.viewport = {0, 0, double(width), double(height), 0, 1};
	raster.scissor = {0, 0, width, height};
	draw.configure(pipeline, nil, raster, {}, inputs);
	draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
	draw.end();
	descriptor.colorAttachments[4].texture = mirror;
	descriptor.colorAttachments[4].loadAction = MTLLoadActionLoad;
	descriptor.colorAttachments[4].storeAction = MTLStoreActionStore;
	return depth_bounds_pass(ctx, descriptor, mirror, type);
}

depth_bounds_pass::depth_bounds_pass(context& ctx, MTL4RenderPassDescriptor* descriptor, id<MTLTexture> mirror, depth_mirror_encoding type) : m_context(ctx), m_descriptor(descriptor), m_mirror(mirror), m_encoding(type), m_samples(static_cast<std::uint32_t>(mirror.sampleCount)) {}
void depth_bounds_pass::configure_pipeline(render_pipeline_configuration& pipeline, fragment_program_options& fragment) const
{
	if (m_ended || pipeline.state.sample_count != m_samples)
		throw std::invalid_argument("Native mirror pass/pipeline sample count differs or pass has ended");
	const auto state = decode_render_state(pipeline.state);
	pipeline.depth_mirror = {m_encoding, state.depth_write};
	fragment.depth_mirror = pipeline.depth_mirror;
	configure_output_rops(pipeline, fragment);
}
void depth_bounds_pass::begin_render()
{
	if (m_ended)
		throw std::logic_error("Native stored-depth mirror pass already ended");
	if (!m_commands)
	{
		m_commands = std::make_unique<render_commands>(m_context, m_descriptor);
		if (m_visibility) m_commands->visibility(m_visibility->first, m_visibility->second);
	}
}
void depth_bounds_pass::visibility(MTLVisibilityResultMode mode, std::size_t offset)
{
	if (m_ended) throw std::logic_error("Native stored-depth mirror pass already ended");
	m_visibility = {mode, offset};
	if (m_commands) m_commands->visibility(mode, offset);
}
void depth_bounds_pass::draw_geometry(const converted_geometry& geometry, const render_pipeline_bundle& pipeline, const fragment_shader& fragment, const bindings& vertex, fragment_context state, std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster, std::size_t instances, std::size_t base_instance, const fragment_shader::constant_binding* constants)
{
	draw_geometry_impl(geometry,nullptr,pipeline,fragment,vertex,state,code,inputs,raster,instances,base_instance,constants);
}
void depth_bounds_pass::draw_geometry(const prepared_geometry& geometry, const render_pipeline_bundle& pipeline, const fragment_shader& fragment, const bindings& vertex, fragment_context state, std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster, std::size_t instances, std::size_t base_instance, const fragment_shader::constant_binding* constants)
{
	draw_geometry_impl(geometry.references(),&geometry,pipeline,fragment,vertex,state,code,inputs,raster,instances,base_instance,constants);
}
void depth_bounds_pass::draw_geometry_impl(const converted_geometry& geometry, const prepared_geometry* prepared, const render_pipeline_bundle& pipeline, const fragment_shader& fragment, const bindings& vertex, fragment_context state, std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster, std::size_t instances, std::size_t base_instance, const fragment_shader::constant_binding* constants)
{
	if (pipeline.depth_mirror.encoding != m_encoding || pipeline.state.sample_count != m_samples || pipeline.depth_mirror.write != pipeline.state.depth_write || fragment.depth_mirror() != pipeline.depth_mirror || fragment.raster_sample_count() != m_samples)
		throw std::invalid_argument("Native stored-depth pass requires matching verified shader/pipeline contracts");
	if (raster.depth_bounds_enabled)
		validate_depth_bounds(raster.depth_min, raster.depth_max);
	state.depth_bounds_enabled = raster.depth_bounds_enabled;
	state.depth_min = raster.depth_min;
	state.depth_max = raster.depth_max;
	const auto bindings = constants ? fragment.prepare(m_context, state, *constants, inputs) : fragment.prepare(m_context, state, code, inputs);
	raster.depth_bounds_enabled = false; // Test actual stored depth in color(4), never incoming depth.
	raster.front_stencil_reference = pipeline.state.front.reference;
	raster.back_stencil_reference = pipeline.state.back.reference;
	begin_render();
	if (prepared) m_commands->draw_geometry(*prepared, pipeline.pipeline, pipeline.depth_stencil, raster, vertex, bindings, instances, base_instance);
	else m_commands->draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vertex, bindings, instances, base_instance);
}
void depth_bounds_pass::end()
{
	begin_render();
	m_commands->end();
	m_ended = true;
}
} // namespace metal4
