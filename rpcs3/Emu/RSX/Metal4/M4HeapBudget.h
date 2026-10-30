#pragma once

#include "M4Device.h"
#include <stdexcept>
#include <functional>

namespace metal4
{
struct heap_budget_statistics
{
	std::size_t budget = 0, reserved_bytes = 0, live_heaps = 0, live_buffers = 0;
	std::uint64_t created = 0;
	// Retirement work, not GPU time. Ordinary allocations visit a bounded
	// number of weak owners; pressure and explicit statistics reconcile all.
	std::uint64_t collection_visits = 0, full_collections = 0;
	std::uint64_t recovery_attempts = 0, recovery_successes = 0;
};
enum class native_allocation_kind { heap, buffer, texture };
class native_allocation_error final : public std::runtime_error
{
  public:
	const native_allocation_kind kind;
	const std::size_t requested;
	native_allocation_error(native_allocation_kind kind, std::size_t requested);
};
struct memory_pressure_request
{
	std::size_t requested;
	bool native_failure = false;
};
class heap_budget_error final : public std::runtime_error
{
  public:
	const std::size_t requested, reserved, budget;
	heap_budget_error(std::size_t request, std::size_t live, std::size_t limit);
};
// Share one ledger across native texture caches,surface pools and counters.
// Actual rounded heap/buffer backing remains charged until the native object
// dies,including resources held only by command completion owners.
class heap_budget final
{
	struct impl;
	std::unique_ptr<impl> m_impl;

  public:
	heap_budget(context& ctx, std::size_t bytes);
	~heap_budget();
	heap_budget(const heap_budget&) = delete;
	heap_budget& operator=(const heap_budget&) = delete;
	[[nodiscard]] id<MTLDevice> device() const noexcept;
	[[nodiscard]] heap_budget_statistics statistics() const;
	// Owner-thread callback. Keep the returned registration alive only while
	// its cache owners exist. Foreign threads and VM epochs never reclaim.
	[[nodiscard]] std::shared_ptr<void> register_reclaimer(std::function<void(memory_pressure_request)> callback);
	// Wrap a complete allocation transaction whose local locks/resources have
	// unwound on failure. Nested transactions share ONE retry at the outer edge.
	// Only capacity/native-allocation errors qualify; invalid descriptors do not.
	void with_recovery(const std::function<void()>& allocate);
	[[nodiscard]] id<MTLHeap> allocate(MTLHeapDescriptor* descriptor) __attribute__((ns_returns_retained));
	[[nodiscard]] id<MTLTexture> texture(MTLTextureDescriptor* descriptor) __attribute__((ns_returns_retained));
	[[nodiscard]] id<MTLBuffer> shared_buffer(std::size_t bytes) __attribute__((ns_returns_retained));
	[[nodiscard]] id<MTLBuffer> buffer(std::size_t bytes, MTLStorageMode storage) __attribute__((ns_returns_retained));
};
} // namespace metal4
