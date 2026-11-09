#pragma once

#include "M4Resources.h"
#include "M4Geometry.h"
#include "M4GeometryBinding.h"

#include <array>
#include <utility>

namespace metal4
{

[[nodiscard]] bool supports_native_depth_bounds(id<MTLDevice> device);

class conditional_draw final
{
	friend class query_pool;
	friend class render_commands;
	submission m_recording;
	id<MTLBuffer> m_arguments = nil, m_geometry = nil;
	MTLPrimitiveType m_primitive;
	std::size_t m_index_offset = 0, m_index_bytes = 0;
	MTLIndexType m_index_type = MTLIndexTypeUInt16;
	std::uint32_t m_count, m_instances;

	conditional_draw(submission recording, MTLPrimitiveType primitive, std::uint32_t count, std::uint32_t instances)
		: m_recording(std::move(recording)), m_primitive(primitive), m_count(count), m_instances(instances) {}
};

struct raster_state
{
	MTLViewport viewport{0, 0, 1, 1, 0, 1};
	MTLScissorRect scissor{0, 0, 1, 1};
	MTLCullMode cull = MTLCullModeNone;
	MTLWinding front_face = MTLWindingCounterClockwise;
	MTLTriangleFillMode fill = MTLTriangleFillModeFill;
	MTLDepthClipMode depth_clip = MTLDepthClipModeClip;
	float depth_bias = 0;
	float slope_scale = 0;
	float bias_clamp = 0;
	// Native bounds testing reads stored attachment depth, not fragment output.
	// Requires Apple10; earlier M GPUs need stored-depth shader ROP emulation.
	// Metal disables its native selector at [0,1]; the RSX executor keeps the
	// mirror for that interval to reject nonfinite/out-of-range stored depth.
	bool depth_bounds_enabled = false;
	float depth_min = 0;
	float depth_max = 1;
	std::array<float, 4> blend_color{};
	std::uint32_t front_stencil_reference = 0;
	std::uint32_t back_stencil_reference = 0;
	// Metal has no front-and-back cull enum. Suppress triangle draws only;
	// points and lines still execute, as required by the RSX cull contract.
	bool cull_all_triangles = false;
};

// Native drawing only: RSX shader translation, vertex fetch, guest primitive
// conversion and draw dependencies must be supplied by the renderer adapter.
// The context owns the encoder. Any context pass switch invalidates this facade;
// further calls fail instead of recording into a stale encoder.
class render_commands final
{
	context& m_context;
	id<MTL4RenderCommandEncoder> m_encoder;
	id<MTLBuffer> m_visibility;
	std::size_t m_width;
	std::size_t m_height;
	bool m_configured = false;
	bool m_empty_scissor = false;
	bool m_cull_all_triangles = false;
	bool m_depth_bounds_supported = false;
	bool m_full_barrier = false;

	void require_active() const;
	[[nodiscard]] bool should_draw(MTLPrimitiveType primitive, std::size_t count, std::size_t instances) const;
	friend class clear_engine;
	friend class depth_bounds_engine;
	friend class depth_bounds_pass;
	friend class guest_texture_resolver;
	friend class presenter;
	friend class draw_executor;
	render_commands(context& ctx, MTL4RenderPassDescriptor* descriptor, MTLStages before,
		context::render_writes writes, const bindings* vertex = nullptr, id<MTLBuffer> geometry = nil, context::shared_guest_pass* shared = nullptr);
	static void materialize_attachments(context& ctx, MTL4RenderPassDescriptor* descriptor);
	// The executor supplies only generated RSX shaders. Their writes are
	// limited to render attachments and the descriptor's visibility buffer.
	static void draw_prepared_guest(context& ctx, MTL4RenderPassDescriptor* descriptor, const prepared_geometry& geometry,
		id<MTLRenderPipelineState> pipeline, id<MTLDepthStencilState> depth_stencil, const raster_state& state,
		const bindings& vertex, const bindings& fragment, MTLVisibilityResultMode visibility, std::size_t visibility_offset,
		const conditional_draw* conditional, bool retain_guest_pass);
	// Trusted helper pipelines generate positions from vertex_id alone.
	// This entry point encodes the whole draw without vertex or index bindings.
	static void draw_fragment_triangle(context& ctx, MTL4RenderPassDescriptor* descriptor, id<MTLRenderPipelineState> pipeline,
		id<MTLDepthStencilState> depth_stencil, const raster_state& state, const bindings& fragment);

public:
	explicit render_commands(context& ctx, MTL4RenderPassDescriptor* descriptor);
	render_commands(const render_commands&) = delete;
	render_commands& operator=(const render_commands&) = delete;
	void configure(id<MTLRenderPipelineState> pipeline, id<MTLDepthStencilState> depth_stencil,
		const raster_state& state, const bindings& vertex, const bindings& fragment);
	void visibility(MTLVisibilityResultMode mode, std::size_t offset = 0);
	void draw(MTLPrimitiveType primitive, std::size_t first, std::size_t count,
		std::size_t instances = 1, std::size_t base_instance = 0);
	void draw(const conditional_draw& conditional);
	// Indices are native-endian and already converted to Metal topology/restart
	// conventions. Buffer bounds do not prove vertex-fetch bounds or guest parity.
	void draw_indexed(MTLPrimitiveType primitive, id<MTLBuffer> indices, std::size_t offset,
		MTLIndexType type, std::size_t count, std::size_t instances = 1,
		std::ptrdiff_t base_vertex = 0, std::size_t base_instance = 0);
	// Pipeline must implement the vertex_reference ABI in buffer(30), including
	// separate evaluation of flat colors. Merely binding an ordinary vertex
	// shader does not implement the guest flat-shading contract.
	void draw_geometry(const converted_geometry& geometry, id<MTLRenderPipelineState> pipeline,
		id<MTLDepthStencilState> depth_stencil, const raster_state& state,
		const bindings& vertex, const bindings& fragment, std::size_t instances = 1,
		std::size_t base_instance = 0);
	// Prepared native indexed/small-array path; no upload or allocation while
	// encoding. Vertex context geometry_count must describe references().
	void draw_geometry(const prepared_geometry& geometry, id<MTLRenderPipelineState> pipeline,
		id<MTLDepthStencilState> depth_stencil, const raster_state& state,
		const bindings& vertex, const bindings& fragment, std::size_t instances = 1,
		std::size_t base_instance = 0);
	// Binds the immutable prepared geometry captured with these indirect arguments.
	void draw_geometry(const conditional_draw& conditional, id<MTLRenderPipelineState> pipeline,
		id<MTLDepthStencilState> depth_stencil, const raster_state& state,
		const bindings& vertex, const bindings& fragment);
	void end();
};

}
