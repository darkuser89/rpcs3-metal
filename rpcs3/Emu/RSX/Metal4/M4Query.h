#pragma once

#include "M4Draw.h"
#include "M4HeapBudget.h"

namespace metal4
{
struct visibility_binding
{
	MTLVisibilityResultMode mode;
	std::size_t offset;
};
struct query_pool_statistics
{
	std::size_t live_segments = 0, leased_counters = 0;
	std::uint64_t acquired = 0, initializations = 0, reused_batch = 0, reused_completed = 0, retirement_visits = 0;
};
// Render-thread query lifecycle. Accumulate compatible encoders in one slot,
// and continue that slot after confirmed completion. Pending different batches
// get separate segments. Discard/consume never recycle GPU-owned counter leases.
class query_pool final
{
	struct impl;
	std::unique_ptr<impl> m_impl;

  public:
	query_pool(context& ctx, std::size_t queries = 2048, std::size_t segments = 16384, std::shared_ptr<heap_budget> budget = {});
	~query_pool();
	query_pool(const query_pool&) = delete;
	query_pool& operator=(const query_pool&) = delete;
	[[nodiscard]] std::size_t capacity() const noexcept;
	[[nodiscard]] query_pool_statistics statistics() const;
	void begin(std::uint32_t handle, bool precise);
	void end(std::uint32_t handle);
	void discard(std::uint32_t handle);
	[[nodiscard]] bool ready(std::uint32_t handle) const;
	// Consumes only completed segments,leaving an active query able to gather
	// later draws. Precise results wrap to RSX uint32;boolean results are 0/1.
	[[nodiscard]] std::uint32_t consume(std::uint32_t handle);
	[[nodiscard]] std::optional<visibility_binding> attach(context& ctx, MTL4RenderPassDescriptor* pass);
};
} // namespace metal4
