#pragma once

#include "M4FragmentShader.h"
#include "M4VertexShader.h"

namespace metal4
{
// Only a validated shader-cache entry can create this immutable function and
// fragment-contract pair. It never materializes a stage or waits for a job.
class prepared_shader_stages final
{
	friend class shader_cache;
	friend class render_pipeline_cache;
	compiler_function m_vertex, m_fragment;
	fragment_program_options m_contract;
	prepared_shader_stages(compiler_function vertex, compiler_function fragment, const fragment_program_options& contract)
		: m_vertex(std::move(vertex)), m_fragment(std::move(fragment)), m_contract(contract) {}
};
struct shader_cache_statistics
{
	std::size_t entries = 0, capacity = 0, host_bytes = 0, host_budget = 0;
	std::uint64_t translated = 0, reused = 0, compiled = 0, evicted = 0, bypassed = 0;
	std::uint64_t lookups = 0, candidates = 0, prepared_hits = 0;
	std::uint64_t prewarmed = 0, prewarm_reused = 0, prewarm_refused = 0, prewarm_waited = 0;
};
// Render-thread caches, independently bounded per stage. Keys own code/options
// and vertex liveness; fragment literals remain dynamic and never key shaders.
// Mask lookup translates/validates on a miss without compiling a Metal library.
// Host quota charges entry/code/source/decoded capacities and shader storage
// up front, excluding allocator/index overhead, driver and externally held storage.
// Index nodes are separately bounded by the entry capacity.
// It is not a global working-set budget. Returned owners survive eviction.
// The separate compiler library-source cache has its own quota.
class shader_cache final
{
	struct implementation;
	std::unique_ptr<implementation> m_impl;

  public:
	// Immutable preflight owners. Retain validated code across eviction/clear,
	// never caller spans. Only the producing cache may consume a ticket.
	template <typename Shader> class prepared_program
	{
		friend class shader_cache;
		std::shared_ptr<void> m_owner, m_entry;
		std::uint32_t m_mask = 0;
		prepared_program(std::shared_ptr<void> owner, std::shared_ptr<void> entry, std::uint32_t mask)
			: m_owner(std::move(owner)), m_entry(std::move(entry)), m_mask(mask) {}
	  public:
		prepared_program() = default;
		[[nodiscard]] std::uint32_t texture_mask() const noexcept { return m_mask; }
	};
	using prepared_vertex = prepared_program<vertex_shader>;
	using prepared_fragment = prepared_program<fragment_shader>;
	[[nodiscard]] prepared_vertex prepare_vertex(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options);
	[[nodiscard]] prepared_fragment prepare_fragment(std::span<const std::byte> code, const fragment_program_options& options);
	// Start native compilation for final options without waiting. A full native
	// queue keeps the validated owner for the ordinary ordered getter. Returned
	// owners bind any option change once; input/literal snapshots remain owned.
	[[nodiscard]] prepared_vertex prewarm_vertex(const prepared_vertex& program, const vertex_program_options& options);
	[[nodiscard]] prepared_fragment prewarm_fragment(const prepared_fragment& program, const fragment_program_options& options);
	[[nodiscard]] bool ready(const prepared_vertex& program) const;
	[[nodiscard]] bool ready(const prepared_fragment& program) const;
	[[nodiscard]] std::optional<prepared_shader_stages> pipeline_stages(const prepared_vertex& vertex,
		const prepared_fragment& fragment) const;
	// Upload current literals from the validated CPU program. The binding
	// belongs to the shader materialized from this exact program and options.
	[[nodiscard]] fragment_shader::constant_binding prepare_constants(context& ctx, const prepared_fragment& program,
		std::span<const std::byte> current_code) const;
	[[nodiscard]] std::shared_ptr<const vertex_shader> vertex(const prepared_vertex& program, const vertex_program_options& options);
	[[nodiscard]] std::shared_ptr<const fragment_shader> fragment(const prepared_fragment& program, const fragment_program_options& options);
	explicit shader_cache(const compiler& compiler, std::size_t capacity = 256, std::size_t host_budget = 16u << 20);
	~shader_cache();
	shader_cache(const shader_cache&) = delete;
	shader_cache& operator=(const shader_cache&) = delete;
	[[nodiscard]] std::uint32_t vertex_mask(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options);
	[[nodiscard]] std::uint32_t fragment_mask(std::span<const std::byte> code, const fragment_program_options& options);
	[[nodiscard]] std::shared_ptr<const vertex_shader> vertex(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options);
	[[nodiscard]] std::shared_ptr<const fragment_shader> fragment(std::span<const std::byte> code, const fragment_program_options& options);
	[[nodiscard]] shader_cache_statistics vertex_statistics() const noexcept;
	[[nodiscard]] shader_cache_statistics fragment_statistics() const noexcept;
	void clear();
};
} // namespace metal4
