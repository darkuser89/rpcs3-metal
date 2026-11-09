#include "M4Draw.h"
#include "M4DrawValidation.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace metal4
{

namespace
{

std::pair<std::size_t, std::size_t> pass_extent(MTL4RenderPassDescriptor* descriptor)
{
	if (!descriptor) throw std::invalid_argument("Metal 4 draw pass requires a descriptor");
	std::size_t width = std::numeric_limits<std::size_t>::max();
	std::size_t height = width;
	bool attached = false;
	auto include = [&](MTLRenderPassAttachmentDescriptor* attachment)
	{
		auto texture = attachment.texture;
		if (!texture) return;
		if (attachment.level >= texture.mipmapLevelCount || attachment.level >= 64)
		{
			throw std::out_of_range("Metal 4 render attachment mip level is invalid");
		}
		width = std::min(width, std::max<std::size_t>(1, texture.width >> attachment.level));
		height = std::min(height, std::max<std::size_t>(1, texture.height >> attachment.level));
		attached = true;
	};
	for (NSUInteger i = 0; i < 8; ++i) include(descriptor.colorAttachments[i]);
	include(descriptor.depthAttachment);
	include(descriptor.stencilAttachment);
	if ((!attached && (!descriptor.renderTargetWidth || !descriptor.renderTargetHeight)) ||
		descriptor.renderTargetWidth > width || descriptor.renderTargetHeight > height)
	{
		throw std::invalid_argument("Metal 4 render target dimensions are invalid");
	}
	return {descriptor.renderTargetWidth ? descriptor.renderTargetWidth : width,
		descriptor.renderTargetHeight ? descriptor.renderTargetHeight : height};
}

void validate_raster(const raster_state& state, std::size_t width, std::size_t height)
{
	if (state.depth_bounds_enabled) validate_depth_bounds(state.depth_min, state.depth_max);
	const auto& v = state.viewport;
	if (!std::isfinite(v.originX) || !std::isfinite(v.originY) ||
		!std::isfinite(v.width) || !std::isfinite(v.height) || v.width <= 0 || v.height <= 0 ||
		!std::isfinite(v.znear) || !std::isfinite(v.zfar) ||
		v.znear < 0 || v.znear > 1 || v.zfar < 0 || v.zfar > 1 ||
		!std::isfinite(state.depth_bias) || !std::isfinite(state.slope_scale) || !std::isfinite(state.bias_clamp))
	{
		throw std::invalid_argument("Metal 4 raster state contains invalid viewport or depth bias");
	}
	for (float value : state.blend_color)
	{
		if (!std::isfinite(value)) throw std::invalid_argument("Metal 4 blend constant must be finite");
	}
	if ((state.cull != MTLCullModeNone && state.cull != MTLCullModeFront && state.cull != MTLCullModeBack) ||
		(state.front_face != MTLWindingClockwise && state.front_face != MTLWindingCounterClockwise) ||
		(state.fill != MTLTriangleFillModeFill && state.fill != MTLTriangleFillModeLines) ||
		(state.depth_clip != MTLDepthClipModeClip && state.depth_clip != MTLDepthClipModeClamp))
	{
		throw std::invalid_argument("Metal 4 raster state enum is invalid");
	}
	const auto& s = state.scissor;
	validate_scissor(width, height, s.x, s.y, s.width, s.height);
}

}

bool supports_native_depth_bounds(id<MTLDevice> device)
{
	return [device supportsFamily:MTLGPUFamilyApple10];
}

render_commands::render_commands(context& ctx, MTL4RenderPassDescriptor* descriptor)
	: render_commands(ctx, descriptor, MTLStageAll, context::render_writes::opaque)
{
}

render_commands::render_commands(context& ctx, MTL4RenderPassDescriptor* descriptor, MTLStages before,
	context::render_writes writes, const bindings* vertex, id<MTLBuffer> geometry, context::shared_guest_pass* shared)
	: m_context(ctx)
{
	const auto [width, height] = pass_extent(descriptor);
	m_width = width;
	m_height = height;
	m_visibility = descriptor.visibilityResultBuffer;
	m_depth_bounds_supported = supports_native_depth_bounds(ctx.device());
	m_full_barrier = before == MTLStageAll && !vertex;
	m_encoder = ctx.render_encoder(descriptor, before, writes, vertex, geometry, shared);
}

void render_commands::materialize_attachments(context& ctx, MTL4RenderPassDescriptor* descriptor)
{
	(void)ctx.render_encoder(descriptor, MTLStageAll, context::render_writes::attachments);
	ctx.end_encoding();
}

void render_commands::draw_prepared_guest(context& ctx, MTL4RenderPassDescriptor* descriptor, const prepared_geometry& geometry,
	id<MTLRenderPipelineState> pipeline, id<MTLDepthStencilState> depth_stencil, const raster_state& state,
	const bindings& vertex, const bindings& fragment, MTLVisibilityResultMode visibility, std::size_t visibility_offset,
	const conditional_draw* conditional, bool retain_guest_pass)
{
	context::shared_guest_pass shared{fragment, visibility, visibility_offset};
	render_commands commands(ctx, descriptor, MTLStageAll, context::render_writes::attachments,
		conditional ? nullptr : &vertex, geometry.buffer(), retain_guest_pass ? &shared : nullptr);
	if (shared.reused ? shared.visibility_changed : visibility != MTLVisibilityResultModeDisabled)
		commands.visibility(visibility, visibility_offset);
	if (conditional) commands.draw_geometry(*conditional, pipeline, depth_stencil, state, vertex, fragment);
	else commands.draw_geometry(geometry, pipeline, depth_stencil, state, vertex, fragment);
	if (!retain_guest_pass) commands.end();
}

void render_commands::draw_fragment_triangle(context& ctx, MTL4RenderPassDescriptor* descriptor, id<MTLRenderPipelineState> pipeline,
	id<MTLDepthStencilState> depth_stencil, const raster_state& state, const bindings& fragment)
{
	// Attachments and fragment resources still wait for all earlier queue
	// stages, including writers behind views or aliasing allocations.
	render_commands commands(ctx, descriptor, MTLStageFragment | MTLStageTile, context::render_writes::attachments);
	commands.configure(pipeline, depth_stencil, state, {}, fragment);
	commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
	commands.end();
}

void render_commands::require_active() const
{
	if (!m_encoder || m_context.current_render_encoder() != m_encoder)
	{
		throw std::logic_error("Metal 4 draw pass is no longer active");
	}
}

void render_commands::configure(id<MTLRenderPipelineState> pipeline, id<MTLDepthStencilState> depth_stencil,
	const raster_state& state, const bindings& vertex, const bindings& fragment)
{
	require_active();
	if (!pipeline || pipeline.device != m_context.device() ||
		(depth_stencil && depth_stencil.device != m_context.device()))
	{
		throw std::invalid_argument("Metal 4 draw pipeline/state must belong to the context device");
	}
	validate_raster(state, m_width, m_height);
	if (state.depth_bounds_enabled && !m_depth_bounds_supported)
		throw std::invalid_argument("Stored-depth bounds require shader ROP emulation on Apple7 through Apple9");
	m_configured = false;
	auto vertex_table = vertex.snapshot(m_context);
	auto fragment_table = fragment.snapshot(m_context);
	m_context.keep_alive(pipeline);
	if (depth_stencil) m_context.keep_alive(depth_stencil);
	[m_encoder setRenderPipelineState:pipeline];
	[m_encoder setDepthStencilState:depth_stencil];
	[m_encoder setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
	[m_encoder setArgumentTable:fragment_table atStages:MTLRenderStageFragment];
	[m_encoder setViewport:state.viewport];
	m_empty_scissor = !state.scissor.width || !state.scissor.height;
	// An empty RSX scissor is a no-op, without passing a zero-size rectangle to Metal.
	if (!m_empty_scissor) [m_encoder setScissorRect:state.scissor];
	[m_encoder setCullMode:state.cull];
	[m_encoder setFrontFacingWinding:state.front_face];
	[m_encoder setTriangleFillMode:state.fill];
	[m_encoder setDepthClipMode:state.depth_clip];
	[m_encoder setDepthBias:state.depth_bias slopeScale:state.slope_scale clamp:state.bias_clamp];
	// The selector exists on older GPU drivers but traps even for [0,1]. Only
	// Apple10+ supports it. Reset every supported draw to avoid stale bounds.
	if (m_depth_bounds_supported)
		[m_encoder setDepthTestMinBound:state.depth_bounds_enabled ? state.depth_min : 0
			maxBound:state.depth_bounds_enabled ? state.depth_max : 1];
	[m_encoder setBlendColorRed:state.blend_color[0] green:state.blend_color[1]
		blue:state.blend_color[2] alpha:state.blend_color[3]];
	[m_encoder setStencilFrontReferenceValue:state.front_stencil_reference backReferenceValue:state.back_stencil_reference];
	m_cull_all_triangles = state.cull_all_triangles;
	m_configured = true;
}

void render_commands::visibility(MTLVisibilityResultMode mode, std::size_t offset)
{
	require_active();
	if (mode != MTLVisibilityResultModeDisabled && mode != MTLVisibilityResultModeBoolean &&
		mode != MTLVisibilityResultModeCounting)
	{
		throw std::invalid_argument("Metal 4 visibility mode is invalid");
	}
	if (mode != MTLVisibilityResultModeDisabled)
	{
		if (!m_visibility) throw std::invalid_argument("Metal 4 pass has no visibility buffer");
		validate_visibility_offset(m_visibility.length, offset);
	}
	else if (offset)
	{
		throw std::invalid_argument("Disabled Metal 4 visibility must have zero offset");
	}
	[m_encoder setVisibilityResultMode:mode offset:offset];
}

bool render_commands::should_draw(MTLPrimitiveType primitive, std::size_t count, std::size_t instances) const
{
	require_active();
	if (!m_configured) throw std::logic_error("Metal 4 draw state has not been configured");
	bool triangle = false;
	switch (primitive)
	{
	case MTLPrimitiveTypePoint:
	case MTLPrimitiveTypeLine:
	case MTLPrimitiveTypeLineStrip:
		break;
	case MTLPrimitiveTypeTriangle:
	case MTLPrimitiveTypeTriangleStrip:
		triangle = true;
		break;
	default:
		throw std::invalid_argument("Metal 4 draw primitive is invalid");
	}
	return count && instances && !m_empty_scissor && !(triangle && m_cull_all_triangles);
}

void render_commands::draw(MTLPrimitiveType primitive, std::size_t first, std::size_t count,
	std::size_t instances, std::size_t base_instance)
{
	validate_draw_ids(first, count);
	validate_draw_ids(base_instance, instances);
	if (!should_draw(primitive, count, instances)) return;
	[m_encoder drawPrimitives:primitive vertexStart:first vertexCount:count
		instanceCount:instances baseInstance:base_instance];
}

void render_commands::draw(const conditional_draw& conditional)
{
	if (!m_full_barrier || !conditional.m_recording.same_batch(m_context.recording_submission()))
		throw std::invalid_argument("Metal 4 conditional draw requires a fully ordered render pass in its recording");
	if (!should_draw(conditional.m_primitive, conditional.m_count, conditional.m_instances)) return;
	m_context.use_resource(conditional.m_arguments);
	if (conditional.m_index_bytes)
	{
		m_context.use_resource(conditional.m_geometry);
		[m_encoder drawIndexedPrimitives:conditional.m_primitive indexType:conditional.m_index_type
			indexBuffer:conditional.m_geometry.gpuAddress + conditional.m_index_offset
			indexBufferLength:conditional.m_index_bytes indirectBuffer:conditional.m_arguments.gpuAddress];
	}
	else
		[m_encoder drawPrimitives:conditional.m_primitive indirectBuffer:conditional.m_arguments.gpuAddress];
}

void render_commands::draw_indexed(MTLPrimitiveType primitive, id<MTLBuffer> indices, std::size_t offset,
	MTLIndexType type, std::size_t count, std::size_t instances, std::ptrdiff_t base_vertex, std::size_t base_instance)
{
	validate_draw_ids(0, count);
	validate_draw_ids(base_instance, instances);
	if (type != MTLIndexTypeUInt16 && type != MTLIndexTypeUInt32)
	{
		throw std::invalid_argument("Metal 4 index type is invalid");
	}
	if (!indices || indices.device != m_context.device())
	{
		throw std::invalid_argument("Metal 4 index buffer must belong to the context device");
	}
	const auto bytes = checked_index_bytes(indices.length, offset, count, type == MTLIndexTypeUInt16 ? 2 : 4);
	if (!should_draw(primitive, count, instances)) return;
	m_context.use_resource(indices);
	[m_encoder drawIndexedPrimitives:primitive indexCount:count indexType:type
		indexBuffer:indices.gpuAddress + offset indexBufferLength:bytes
		instanceCount:instances baseVertex:base_vertex baseInstance:base_instance];
}

void render_commands::end()
{
	require_active();
	m_context.end_encoding();
	m_encoder = nil;
}

void render_commands::draw_geometry(const converted_geometry& geometry, id<MTLRenderPipelineState> pipeline,
	id<MTLDepthStencilState> depth_stencil, const raster_state& state, const bindings& vertex,
	const bindings& fragment, std::size_t instances, std::size_t base_instance)
{
	require_active();
	MTLPrimitiveType primitive;
	std::size_t corners;
	switch (geometry.topology)
	{
	case list_topology::points: primitive = MTLPrimitiveTypePoint; corners = 1; break;
	case list_topology::lines: primitive = MTLPrimitiveTypeLine; corners = 2; break;
	case list_topology::triangles: primitive = MTLPrimitiveTypeTriangle; corners = 3; break;
	default: throw std::invalid_argument("Unknown Metal 4 converted list topology");
	}
	if (geometry.vertices.size() % corners)
	{
		throw std::invalid_argument("Metal 4 converted geometry contains an incomplete primitive");
	}
	validate_draw_ids(0, geometry.vertices.size());
	validate_draw_ids(base_instance, instances);
	if (geometry.vertices.empty() || !instances) return;
	const auto refs = m_context.upload(std::as_bytes(std::span(geometry.vertices)));
	auto vertex_resources = vertex;
	vertex_resources.set_buffer(geometry_binding, refs.buffer, refs.offset);
	configure(pipeline, depth_stencil, state, vertex_resources, fragment);
	draw(primitive, 0, geometry.vertices.size(), instances, base_instance);
}

void render_commands::draw_geometry(const prepared_geometry& geometry, id<MTLRenderPipelineState> pipeline,
	id<MTLDepthStencilState> depth_stencil, const raster_state& state, const bindings& vertex,
	const bindings& fragment, std::size_t instances, std::size_t base_instance)
{
	require_active();
	const auto primitive = geometry.primitive();
	validate_draw_ids(base_instance,instances);
	if (!geometry.count() || !instances) return;
	const auto buffer = geometry.buffer();
	if (!buffer || buffer.device != m_context.device()) throw std::invalid_argument("Native geometry packet has a foreign device");
	auto resources = vertex;
	resources.set_buffer(geometry_binding,buffer);
	configure(pipeline,depth_stencil,state,resources,fragment);
	if (geometry.index_width()) draw_indexed(primitive,buffer,geometry.index_offset(),
		geometry.index_width() == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32,geometry.count(),instances,0,base_instance);
	else draw(primitive,0,geometry.count(),instances,base_instance);
}

void render_commands::draw_geometry(const conditional_draw& conditional, id<MTLRenderPipelineState> pipeline,
	id<MTLDepthStencilState> depth_stencil, const raster_state& state,
	const bindings& vertex, const bindings& fragment)
{
	if (!conditional.m_geometry || conditional.m_geometry.device != m_context.device())
		throw std::invalid_argument("Metal 4 conditional geometry requires a prepared packet on the context device");
	auto resources = vertex;
	resources.set_buffer(geometry_binding, conditional.m_geometry);
	configure(pipeline, depth_stencil, state, resources, fragment);
	draw(conditional);
}

}
