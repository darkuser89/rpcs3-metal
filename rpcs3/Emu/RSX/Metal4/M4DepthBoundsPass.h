#pragma once

#include "M4FragmentShader.h"
#include "M4RenderPipeline.h"
#include "M4Texture.h"
#include "M4SurfacePool.h"
#include <optional>
#include <utility>

namespace metal4
{
class depth_bounds_engine;
// Derive the same shader/pipeline contract before encoding, when only the
// validated native attachment format and effective depth state are available.
void configure_depth_bounds_pipeline(render_pipeline_configuration& pipeline, fragment_program_options& fragment, MTLPixelFormat depth);

// One prepared pass owns a private R32 mirror of the actual depth subresource.
// Color(4) is reserved for the mirror; logical RSX colors remain color(0..3).
// Initialization and draw resource references live until GPU completion.
class depth_bounds_pass final
{
	friend class depth_bounds_engine;
	friend class draw_executor;
	context& m_context;
	MTL4RenderPassDescriptor* m_descriptor;
	id<MTLTexture> m_mirror;
	depth_mirror_encoding m_encoding;
	std::uint32_t m_samples;
	std::unique_ptr<render_commands> m_commands;
	std::optional<std::pair<MTLVisibilityResultMode, std::size_t>> m_visibility;
	bool m_ended = false;

	depth_bounds_pass(context& ctx, MTL4RenderPassDescriptor* descriptor, id<MTLTexture> mirror, depth_mirror_encoding encoding);
	void begin_render(const bindings* guest_vertex = nullptr, id<MTLBuffer> geometry = nil, bool conditional = false);
	// The executor supplies exactly one generated draw with frozen bindings.
	void draw_guest_geometry(const prepared_geometry& geometry, const render_pipeline_bundle& pipeline,
		const fragment_shader& fragment, const bindings& vertex, fragment_context state,
		std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster,
		const fragment_shader::constant_binding* constants, const conditional_draw* conditional = nullptr);
	void draw_geometry_impl(const converted_geometry& geometry, const prepared_geometry* prepared,
		const render_pipeline_bundle& pipeline, const fragment_shader& fragment, const bindings& vertex, fragment_context state,
		std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster,
		std::size_t instances, std::size_t base_instance, const fragment_shader::constant_binding* constants,
		bool guest = false, const conditional_draw* conditional = nullptr);

  public:
	depth_bounds_pass(const depth_bounds_pass&) = delete;
	depth_bounds_pass& operator=(const depth_bounds_pass&) = delete;
	// Derives mirror write behavior from effective native depth-test/write state.
	void configure_pipeline(render_pipeline_configuration& pipeline, fragment_program_options& fragment) const;
	// Applied only when the final draw encoder starts. The depth initializer
	// and any clear materialization never use this visibility result buffer.
	void visibility(MTLVisibilityResultMode mode, std::size_t offset);
	// Uploads bounds from raster_state, then suppresses only the unavailable
	// native bounds selector. Other raster/depth/stencil behavior stays native.
	void draw_geometry(const converted_geometry& geometry, const render_pipeline_bundle& pipeline,
					   const fragment_shader& fragment, const bindings& vertex, fragment_context state,
					   std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster,
					   std::size_t instances = 1, std::size_t base_instance = 0,
					   const fragment_shader::constant_binding* constants = nullptr);
	void draw_geometry(const prepared_geometry& geometry, const render_pipeline_bundle& pipeline,
		const fragment_shader& fragment, const bindings& vertex, fragment_context state,
		std::span<const std::byte> code, const fragment_inputs& inputs, raster_state raster,
		std::size_t instances = 1, std::size_t base_instance = 0,
		const fragment_shader::constant_binding* constants = nullptr);
	void end();
	[[nodiscard]] id<MTLTexture> mirror_texture() const noexcept
	{
		return m_mirror;
	}
};

class depth_bounds_engine final
{
	helper_catalog m_compiler;
	id<MTLDevice> m_device;
	id<MTLLibrary> m_single, m_multisampled;
	std::array<id<MTLRenderPipelineState>, 9> m_pipelines{};
	[[nodiscard]] id<MTLRenderPipelineState> initializer(std::uint32_t samples);

  public:
	explicit depth_bounds_engine(context& ctx);
	// Invoke after context.begin(), outside an existing render facade. Reads
	// actual D16/D32/D32S8 depth, including selected 2D mip/array slice. Clear
	// actions are materialized on the real attachments before reading; depth
	// DontCare and layered/MSAA-array passes need a separate defined policy.
	// An optional pre-reserved R32 lease makes the mirror part of the caller's
	// heap budget. The lease is registered with the command completion lifetime.
	[[nodiscard]] depth_bounds_pass prepare(context& ctx, MTL4RenderPassDescriptor* descriptor, const surface_lease& mirror = {});
};
} // namespace metal4
