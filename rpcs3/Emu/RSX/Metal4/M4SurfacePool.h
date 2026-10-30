#pragma once

#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"
#include "M4HeapBudget.h"

#include <variant>

namespace metal4
{
struct surface_pool_state;
struct surface_allocation_description
{
	std::size_t width = 0, height = 0;
	std::uint32_t samples = 1;
	surface_format format = surface_color::argb8;
	// Spatial single-sample backing for native MSAA import/expansion.
	// Single-sample allocations don't need a separate expanded texture.
	bool expanded = false;
	bool operator==(const surface_allocation_description&) const = default;
};

// One exclusive logical owner, with copies allowed while that surface remains
// bound or submitted. This is allocation storage, not guest-address coherence.
// Every checkout needs explicit content initialization. Native texture handles
// must not be used for future work after dropping their allocation lease.
class surface_allocation final : public std::enable_shared_from_this<surface_allocation>
{
	surface_allocation_description m_description;
	id<MTLHeap> m_heap;
	id<MTLTexture> m_render, m_expanded;
	std::size_t m_bytes;
	surface_allocation(surface_allocation_description description, id<MTLHeap> heap,
					   id<MTLTexture> render, id<MTLTexture> expanded);
	friend class surface_pool;
	void use(context& ctx);

  public:
	surface_allocation(const surface_allocation&) = delete;
	surface_allocation& operator=(const surface_allocation&) = delete;
	[[nodiscard]] const surface_allocation_description& description() const noexcept
	{
		return m_description;
	}
	[[nodiscard]] std::size_t reserved_bytes() const noexcept
	{
		return m_bytes;
	}
	// Registers this lease and the backing heap automatically. It stays leased
	// through GPU completion even if all caller handles and the pool disappear.
	// A retained return lets ARC callers release promptly rather than keeping
	// idle native storage alive until an unrelated autorelease-pool drain.
	[[nodiscard]] id<MTLTexture> render_texture(context& ctx) __attribute__((ns_returns_retained));
	[[nodiscard]] id<MTLTexture> expanded_texture(context& ctx) __attribute__((ns_returns_retained));
};
using surface_lease = std::shared_ptr<surface_allocation>;

struct surface_pool_statistics
{
	std::size_t budget = 0, reserved_bytes = 0, idle_bytes = 0;
	std::size_t leased_allocations = 0, idle_allocations = 0;
	std::uint64_t allocations_created = 0, allocations_reused = 0, evictions = 0;
};

class surface_budget_error final : public std::runtime_error
{
  public:
	const std::size_t requested, reserved, budget;
	surface_budget_error(std::size_t request, std::size_t live, std::size_t limit);
};

// Render-thread checkouts; completion callbacks may return their last leases.
// The budget counts actual reserved heap sizes, including in-flight leases
// and retired heaps still retained by external native texture/view handles.
// Evicts only idle allocations in LRU order. Pressure never aliases an active
// surface or waits for GPU work while holding the pool lock. Caller can spill
// guest surfaces, release leases, wait for completion, then retry a checkout.
class surface_pool final
{
	std::shared_ptr<surface_pool_state> m_state;
	[[nodiscard]] surface_lease acquire_once(const surface_allocation_description& description);

  public:
	explicit surface_pool(context& ctx, std::size_t budget, std::shared_ptr<heap_budget> shared_budget = {});
	surface_pool(const surface_pool&) = delete;
	surface_pool& operator=(const surface_pool&) = delete;
	[[nodiscard]] surface_lease acquire(const surface_allocation_description& description);
	[[nodiscard]] surface_pool_statistics statistics() const;
	// Drops oldest idle allocations until their total is <= target bytes.
	// Live caller/command leases remain accounted and untouched.
	[[nodiscard]] std::size_t trim(std::size_t idle_target_bytes = 0);
};
} // namespace metal4
