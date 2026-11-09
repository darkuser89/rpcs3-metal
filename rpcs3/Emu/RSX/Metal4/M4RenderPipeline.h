#pragma once

#include "M4BlendROP.h"
#include "M4DepthBounds.h"
#include "M4Draw.h"
#include "M4LogicOp.h"
#include "M4Pipeline.h"
#include "M4RenderState.h"

namespace metal4
{
struct fragment_program_options;
class fragment_shader;
class prepared_shader_stages;

struct render_pipeline_configuration
{
	guest_render_state state;
	// Logical shader-output order, after guest surface/channel remapping. These
	// must match the render pass; Invalid means an unattached logical output.
	std::array<MTLPixelFormat, 4> colors{};
	// configure_surface_color sets these bits for surfaces whose source/dest
	// alpha semantics require shader ROPs when blending is later enabled.
	std::uint32_t shader_blend_mask = 0;
	// Selected RSX targets using verified fixed-function blend semantics.
	std::uint32_t native_blend_mask = 0;
	logic_rop_configuration shader_logic;
	blend_rop_configuration shader_blend;
	output_rop_configuration shader_output;
	std::uint32_t shader_half_storage_mask = 0;
	depth_mirror_configuration depth_mirror;
	std::array<color_export_remap, 4> color_remap{};
	MTLPrimitiveTopologyClass topology = MTLPrimitiveTopologyClassTriangle;
	bool rasterization = true;
	bool operator==(const render_pipeline_configuration&) const = default;
};

// Descriptor construction does not create a device or execute GPU work.
// Native Metal4 has no depth/stencil pixel-format fields in its pipeline
// descriptor. Depth/stencil formats belong to the render pass attachments.
[[nodiscard]] MTL4RenderPipelineDescriptor* make_render_descriptor(const render_pipeline_configuration& config,
																   MTL4FunctionDescriptor* vertex, MTL4FunctionDescriptor* fragment);
// Derive matching framebuffer-fetch channel widths for shader and pipeline.
void configure_logic_operations(render_pipeline_configuration& config, fragment_program_options& fragment);
// Call after configuring guest surface formats. Uses native framebuffer fetch
// for every enabled blend target, including remapped and signed equations.
void configure_shader_blending(render_pipeline_configuration& config, fragment_program_options& fragment);
// Select native blending where the complete RSX export/format/ROP contract
// permits it; retain programmable ROPs independently on other MRT outputs.
void configure_blending(render_pipeline_configuration& config, fragment_program_options& fragment);
void configure_output_rops(render_pipeline_configuration& config, fragment_program_options& fragment);
[[nodiscard]] MTLDepthStencilDescriptor* make_depth_stencil_descriptor(const guest_render_state& state);

struct render_pipeline_bundle
{
	id<MTLRenderPipelineState> pipeline;
	id<MTLDepthStencilState> depth_stencil;
	render_state state;
	depth_mirror_configuration depth_mirror;
	// Stencil masks live in depth_stencil; the independently masked reference
	// values are encoder state. Always replace both sides on each configure.
	void configure(render_commands& commands, raster_state raster,
				   const bindings& vertex, const bindings& fragment) const;
};

struct render_pipeline_cache_statistics
{
	std::size_t entries = 0, capacity = 0;
	std::uint64_t created = 0, reused = 0, evicted = 0, bypassed = 0;
	std::uint64_t fast_hits = 0, descriptor_lookups = 0, candidates = 0;
	std::uint64_t prewarmed = 0, pending_reused = 0, prewarm_refused = 0, prewarm_waited = 0;
};

// Render-thread LRU for this file's validated descriptor factory only. Native
// states and their libraries stay owned until eviction; encoded submissions
// retain their own states. Capacity bounds entries, not opaque driver bytes.
class render_pipeline_cache final
{
	struct implementation;
	std::unique_ptr<implementation> m_impl;
	struct preparation;

  public:
	explicit render_pipeline_cache(const compiler& compiler, std::size_t capacity = 256);
	~render_pipeline_cache();
	render_pipeline_cache(const render_pipeline_cache&) = delete;
	render_pipeline_cache& operator=(const render_pipeline_cache&) = delete;
	[[nodiscard]] render_pipeline_bundle get(const render_pipeline_configuration& config,
											 MTL4FunctionDescriptor* vertex, const fragment_shader& fragment);
	class prepared_pipeline final
	{
		friend class render_pipeline_cache;
		std::shared_ptr<void> m_owner;
		std::shared_ptr<preparation> m_entry;
		render_pipeline_bundle m_ready{};
		render_state m_state;
		depth_mirror_configuration m_mirror;
		prepared_pipeline(std::shared_ptr<void> owner, std::shared_ptr<preparation> entry,
			render_state state, depth_mirror_configuration mirror)
			: m_owner(std::move(owner)), m_entry(std::move(entry)), m_state(state), m_mirror(mirror) {}
		prepared_pipeline(std::shared_ptr<void> owner, render_pipeline_bundle ready, render_state state, depth_mirror_configuration mirror)
			: m_owner(std::move(owner)), m_ready(ready), m_state(state), m_mirror(mirror) {}
	  public:
		[[nodiscard]] bool ready() const;
	};
  private:
	[[nodiscard]] std::optional<prepared_pipeline> prewarm_ready(const render_pipeline_configuration& config,
		MTL4LibraryFunctionDescriptor* vertex, MTL4LibraryFunctionDescriptor* fragment);
  public:
	// Nonblocking native PSO admission, with at most64 indexed pending owners.
	// Exact effective keys deduplicate even completed but unconsumed jobs. Tickets
	// freeze dynamic stencil references separately and survive clear/eviction.
	[[nodiscard]] std::optional<prepared_pipeline> prewarm(const render_pipeline_configuration& config,
		MTL4FunctionDescriptor* vertex, const fragment_shader& fragment);
	[[nodiscard]] std::optional<prepared_pipeline> prewarm(const render_pipeline_configuration& config,
		const prepared_shader_stages& stages);
	[[nodiscard]] render_pipeline_bundle get(const prepared_pipeline& prepared,
		std::chrono::milliseconds timeout = std::chrono::seconds(30));
	[[nodiscard]] render_pipeline_cache_statistics statistics() const noexcept;
	void clear();
};

// The native compiler builds the render pipeline. Sample counts are checked
// against its Apple device; unsupported configurations throw without fallback.
// Shader ROPs, guest surface conversion, caches and pass matching are separate
// renderer requirements, not inferred from a descriptor compiling successfully.
[[nodiscard]] render_pipeline_bundle make_render_pipeline(const compiler& compiler,
														  const render_pipeline_configuration& config, MTL4FunctionDescriptor* vertex, MTL4FunctionDescriptor* fragment);
// Verifies compiled logic/blend contracts before enabling shader ROPs.
[[nodiscard]] render_pipeline_bundle make_render_pipeline(const compiler& compiler,
														  const render_pipeline_configuration& config, MTL4FunctionDescriptor* vertex, const fragment_shader& fragment);

} // namespace metal4
