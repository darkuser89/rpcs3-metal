#pragma once

#include "M4Resources.h"
#include "M4Geometry.h"
#include "M4GeometryBinding.h"

#include <array>

namespace metal4
{

[[nodiscard]] bool supports_native_depth_bounds(id<MTLDevice> device);

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

	void require_active() const;
	[[nodiscard]] bool should_draw(MTLPrimitiveType primitive, std::size_t count, std::size_t instances) const;

public:
	explicit render_commands(context& ctx, MTL4RenderPassDescriptor* descriptor);
	render_commands(const render_commands&) = delete;
	render_commands& operator=(const render_commands&) = delete;
	void configure(id<MTLRenderPipelineState> pipeline, id<MTLDepthStencilState> depth_stencil,
		const raster_state& state, const bindings& vertex, const bindings& fragment);
	void visibility(MTLVisibilityResultMode mode, std::size_t offset = 0);
	void draw(MTLPrimitiveType primitive, std::size_t first, std::size_t count,
		std::size_t instances = 1, std::size_t base_instance = 0);
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
	void end();
};

}
