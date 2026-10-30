#pragma once

#include "M4DrawTextureEpoch.h"
#include "M4GuestTexture.h"
#include "M4RSXDrawPacket.h"
#include "M4RSXDrawSnapshot.h"
#include "M4SurfaceCache.h"
#include "M4VertexFetch.h"
#include "M4VertexShader.h"
#include "M4Query.h"
#include "M4ShaderCache.h"
#include <algorithm>

namespace metal4
{
class framebuffer_snapshot;
struct versioned_draw_texture_epoch
{
	draw_texture_epoch textures;
	std::vector<guest_texture_revision> revisions;
	// CPU-only; caller excludes every writer/mapping change for the complete
	// validation plus program/vertex/constants capture. Surface operations are
	// serialized on the render thread throughout preparation and submission.
	[[nodiscard]] bool current(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment) const
	{
		if (!textures.matches_words(vertex, fragment))
			return false;
		return std::all_of(revisions.begin(), revisions.end(), [](const auto& revision) { return revision.current(); });
	}
};
// Frozen sampler metadata is available before upload completion so the caller
// can capture programs/geometry/streams/constants in one short CPU epoch.
// No pending native texture can be obtained through this interface.
class versioned_draw_texture_job final
{
	struct state;
	std::shared_ptr<state> m_state;
	explicit versioned_draw_texture_job(std::shared_ptr<state> state);
	friend class draw_executor;
  public:
	~versioned_draw_texture_job();
	versioned_draw_texture_job(versioned_draw_texture_job&&) noexcept;
	versioned_draw_texture_job& operator=(versioned_draw_texture_job&&) noexcept;
	versioned_draw_texture_job(const versioned_draw_texture_job&) = delete;
	versioned_draw_texture_job& operator=(const versioned_draw_texture_job&) = delete;
	[[nodiscard]] const vertex_texture_snapshot& vertex_words() const;
	[[nodiscard]] const fragment_texture_snapshot& fragment_words() const;
	[[nodiscard]] const texture_cache_state& metadata(bool vertex, unsigned unit) const;
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancel_pending() const;
	// CPU-only,inside the same producer epoch as the independent draw capture.
	[[nodiscard]] bool current(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment) const;
	[[nodiscard]] versioned_draw_texture_epoch finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
};
struct draw_targets
{
	std::array<cached_surface_ref, 4> colors{};
	cached_surface_ref depth;
	surface_color register_color_format = surface_color::argb8;
};
struct draw_request
{
	struct wait_dependency
	{
		id<MTLSharedEvent> event = nil;
		std::uint64_t value = 0;
	};
	converted_geometry geometry;
	vertex_fetch_request fetch;
	std::shared_ptr<const std::vector<std::uint32_t>> source_vertices;
	id<MTLBuffer> persistent = nil, transient = nil;
	vertex_inputs vertex_resources;
	vertex_context vertex_uniforms;
	fragment_context fragment_uniforms;
	raster_state raster;
	std::vector<wait_dependency> waits;
	bool alpha_to_coverage = false, alpha_to_one = false;
	bool allow_output_rounding = true, force_disable_blending = false;
	query_pool* queries = nullptr;
};
// Runtime encoder/dependency overrides that are not owned RSX input captures.
// Geometry,packed streams,constants and shader environment come from packet.
struct draw_execution_state
{
	raster_state raster;
	std::vector<draw_request::wait_dependency> waits;
	bool allow_output_rounding = true, force_disable_blending = false;
	query_pool* queries = nullptr;
};
// One native draw boundary: resolve frozen textures before writer preparation,
// then compile/import stages,fetch vertices and encode through SurfaceCache's
// transaction. Uses one instance;per-instance constants and coordinated pixel
// scaling remain frontend contracts.
// Owns no GSRender/thread or VM hooks;the productive caller must capture and
// supply a coherent draw/framebuffer/fetch/constants epoch. Synchronous success
// publishes surface writes;timeout is completed by SurfaceCache::synchronize.
class draw_executor final
{
	context& m_context;
	surface_cache& m_surfaces;
	guest_texture_cache& m_textures;
	compiler m_compiler;
	render_pipeline_cache m_pipelines;
	shader_cache m_shaders;
	sampler_cache m_samplers;
	const bool m_native_depth_bounds_supported;
	void execute_owned(const rsx_draw_packet& packet, const draw_targets& targets, const draw_execution_state& state,
					   framebuffer_extent_policy extent_policy, const draw_texture_epoch* textures = nullptr);
	void execute_framebuffer(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_execution_state& state,
							 const draw_texture_epoch* textures);
	void execute_draw(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request,
					  const draw_texture_epoch* textures);

  public:
	draw_executor(context& context, surface_cache& surfaces, guest_texture_cache& textures);
	[[nodiscard]] shader_cache_statistics vertex_statistics() const noexcept { return m_shaders.vertex_statistics(); }
	[[nodiscard]] shader_cache_statistics fragment_statistics() const noexcept { return m_shaders.fragment_statistics(); }
	[[nodiscard]] render_pipeline_cache_statistics pipeline_statistics() const noexcept { return m_pipelines.statistics(); }
	[[nodiscard]] library_cache_statistics library_statistics() const noexcept { return m_compiler.library_statistics(); }
	[[nodiscard]] compiler_queue_statistics compilation_statistics() const { return m_compiler.tasks().statistics(); }
	[[nodiscard]] draw_texture_epoch prepare_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment);
	[[nodiscard]] versioned_draw_texture_epoch prepare_versioned_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment);
	[[nodiscard]] versioned_draw_texture_job enqueue_versioned_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment,
		std::span<const guest_texture_upload_dependency> dependencies = {});
	void execute(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request);
	void execute(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request, const draw_texture_epoch& textures);
	void execute(const rsx_draw_packet& packet, const draw_targets& targets, const draw_execution_state& state = {});
	void execute(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_execution_state& state = {});
	void execute(const rsx_draw_packet& packet, const draw_targets& targets, const draw_texture_epoch& textures, const draw_execution_state& state = {});
	void execute(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_texture_epoch& textures, const draw_execution_state& state = {});
};
} // namespace metal4
