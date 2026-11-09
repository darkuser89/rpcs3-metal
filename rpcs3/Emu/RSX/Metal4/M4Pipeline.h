#pragma once

#include "M4Device.h"
#include "M4CompilerQueue.h"

#include <string_view>
#include <array>

namespace metal4
{

struct library_cache_statistics
{
	std::size_t entries = 0, capacity = 0, source_bytes = 0, source_budget = 0;
	std::uint64_t created = 0, reused = 0, evicted = 0, bypassed = 0;
	std::uint64_t candidates = 0;
	std::size_t prepared_source_bytes = 0, prepared_source_budget = 0;
};

class compiler final
{

	id<MTL4Compiler> m_compiler;
	std::shared_ptr<compiler_queue> m_jobs;
	struct library_cache;
	std::shared_ptr<library_cache> m_libraries;
	struct library_preparation;

  public:
	// Render-thread cache keyed by exact MSL bytes under fixed compile options.
	// Labels are diagnostic;the first compilation supplies the cached name.
	// Zero capacity disables caching. Completed and ticket-owned sources
	// have separate quotas. Both exclude opaque driver data.
	explicit compiler(context& ctx, std::size_t library_capacity = 256, std::size_t source_budget = 16u << 20,
		id<MTL4Compiler> native = nil);
	[[nodiscard]] id<MTLDevice> device() const noexcept;
	// Shared bounded native task service. Prewarm callers may admit/poll jobs
	// without waiting; ordinary getters wait only when they need a cold result.
	// Compiler copies share this service; explicit shutdown affects all copies.
	[[nodiscard]] compiler_queue& tasks() const noexcept { return *m_jobs; }
	[[nodiscard]] id<MTLLibrary> library(std::string_view msl, NSString* label = nil) const;
	class prepared_library final
	{
		friend class compiler;
		std::shared_ptr<library_preparation> m_state;
		explicit prepared_library(std::shared_ptr<library_preparation> state) : m_state(std::move(state)) {}
	  public:
		[[nodiscard]] bool ready() const;
		[[nodiscard]] bool failed() const noexcept;
		[[nodiscard]] compiler_function function(NSString* name) const;
		// Host charge for the owned source key/state, excluding native job inputs.
		[[nodiscard]] std::size_t retained_bytes() const noexcept;
	};
	// Render-thread cache access; nonblocking native admission. A full queue
	// or a full ticket source budget returns nullopt without dropping a draw.
	// Tickets own exact source bytes and
	// survive clear/eviction; only this compiler or its copies may consume them.
	[[nodiscard]] std::optional<prepared_library> prewarm_library(std::string_view msl, NSString* label = nil) const;
	[[nodiscard]] id<MTLLibrary> library(const prepared_library& prepared,
		std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
	[[nodiscard]] library_cache_statistics library_statistics() const noexcept;
	void clear_libraries();
	[[nodiscard]] id<MTLComputePipelineState> compute(MTL4ComputePipelineDescriptor* descriptor) const;
	[[nodiscard]] id<MTLRenderPipelineState> render(MTL4RenderPipelineDescriptor* descriptor) const;
};

[[nodiscard]] MTL4LibraryFunctionDescriptor* library_function(id<MTLLibrary> library, NSString* entry);

struct helper_render_configuration
{
	std::array<MTLPixelFormat, 4> colors{};
	std::array<MTLColorWriteMask, 4> masks{MTLColorWriteMaskAll, MTLColorWriteMaskAll, MTLColorWriteMaskAll, MTLColorWriteMaskAll};
	std::uint32_t samples = 1;
	bool operator==(const helper_render_configuration&) const = default;
};

struct helper_catalog_statistics
{
	library_cache_statistics libraries;
	std::size_t capacity = 0, compute_entries = 0, render_entries = 0;
	std::uint64_t compiler_instances = 1;
	std::uint64_t compute_created = 0, compute_reused = 0, compute_evicted = 0, compute_bypassed = 0;
	std::uint64_t render_created = 0, render_reused = 0, render_evicted = 0, render_bypassed = 0;
};

// Fixed native helpers share one compiler/catalogue per device under the same
// safe MSL4 options. Contexts retain it even when constructor-local handles die.
// Calls from different contexts serialize misses so identical jobs compile once.
// This deliberately accepts a closed descriptor profile: default compute state,
// triangle render state with no blending and explicit format/mask/sample keys.
// Dynamic RSX pipelines and custom compiler quotas use compiler separately.
class helper_catalog final
{
	std::shared_ptr<helper_catalog_state> m_state;

  public:
	explicit helper_catalog(context& ctx);
	// Explicit isolated catalogue, useful for constrained owners and quota tests.
	// Capacity is per PSO kind; zero bypasses PSO caching. Library quota is fixed.
	helper_catalog(context& ctx, std::size_t pipeline_capacity);
	[[nodiscard]] id<MTLLibrary> library(std::string_view source, NSString* label = nil) const;
	[[nodiscard]] id<MTLComputePipelineState> compute(id<MTLLibrary> library, NSString* entry) const;
	[[nodiscard]] id<MTLRenderPipelineState> render(id<MTLLibrary> library, NSString* vertex, NSString* fragment,
		const helper_render_configuration& config = {}) const;
	[[nodiscard]] helper_catalog_statistics statistics() const;
	// Externally held libraries/PSOs and in-flight work remain valid after clear.
	void clear();
};

} // namespace metal4
