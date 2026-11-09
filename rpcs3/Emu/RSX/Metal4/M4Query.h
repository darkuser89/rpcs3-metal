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
struct report_query_source
{
	std::uint32_t handle, consumed = 0;
};
struct report_query_snapshot
{
	// Only queries included in this statistic, in retirement order.
	std::span<const report_query_source> sources;
	std::uint32_t retired = 0;
	bool precise = false;
};
// Immutable RSX report value, bound to the queue that consumes it.
class report_predicate final
{
	friend class query_pool;
	friend class draw_executor;
	std::weak_ptr<const std::byte> m_queue;
	std::optional<submission> m_producer;
	id<MTLBuffer> m_value;
	std::optional<bool> m_cpu_result;

	report_predicate(std::weak_ptr<const std::byte> queue, std::optional<submission> producer, id<MTLBuffer> value,
		std::optional<bool> cpu_result = {})
		: m_queue(std::move(queue)), m_producer(std::move(producer)), m_value(value), m_cpu_result(cpu_result) {}
	[[nodiscard]] bool belongs_to(const context& ctx) const;
	[[nodiscard]] bool permits_publication(std::chrono::milliseconds timeout) const;

  public:
	// A CPU-known result can replace predication after successful predecessor completion.
	[[nodiscard]] std::optional<bool> settled_cpu_result(const context& ctx) const;
};
// Render-thread query lifecycle. Accumulate compatible encoders in one slot,
// and continue that slot after confirmed completion. Pending different batches
// get separate segments. Discard/consume never recycle GPU-owned counter leases.
class query_pool final
{
	struct impl;
	std::unique_ptr<impl> m_impl;
	[[nodiscard]] conditional_draw prepare_conditional_geometry(const prepared_geometry& geometry, std::uint32_t instances);
	[[nodiscard]] std::optional<conditional_draw> encode_conditional_draw(std::span<const std::uint32_t> handles,
		const report_query_snapshot* report, conditional_draw draw, std::uint32_t first, std::uint32_t base_instance);
	[[nodiscard]] id<MTLBuffer> encode_reduction(std::span<const std::uint32_t> handles,
		const report_query_snapshot* report, const conditional_draw* draw, std::uint32_t first, std::uint32_t base_instance);

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
	// Snapshots each handle's current consume result on the GPU and draws when
	// any result is nonzero. A pending producer in a different recording
	// requires the caller's fallback. The returned command belongs to this recording.
	[[nodiscard]] std::optional<conditional_draw> encode_conditional_draw(std::span<const std::uint32_t> handles,
		MTLPrimitiveType primitive, std::uint32_t first, std::uint32_t count,
		std::uint32_t instances = 1, std::uint32_t base_instance = 0);
	[[nodiscard]] std::optional<conditional_draw> encode_conditional_draw(std::span<const std::uint32_t> handles,
		const prepared_geometry& geometry, std::uint32_t instances = 1, std::uint32_t base_instance = 0);
	// Includes CPU-retired report counts and consumed intervals without
	// retiring the remaining sources. Unavailable sources defer to the scheduler.
	[[nodiscard]] std::optional<conditional_draw> encode_report_conditional_draw(const report_query_snapshot& report,
		const prepared_geometry& geometry, std::uint32_t instances = 1, std::uint32_t base_instance = 0);
	// Reduces the live handles once. Later draws need only the immutable
	// result, even after RSX has retired or reused every source handle.
	// Pending segments from earlier recordings are observed before commit.
	[[nodiscard]] std::optional<report_predicate> capture_report_predicate(const report_query_snapshot& report);
	// Own the reduction recording and order it after previously submitted work.
	[[nodiscard]] std::optional<report_predicate> submit_report_predicate(const report_query_snapshot& report);
	[[nodiscard]] conditional_draw encode_predicated_draw(const report_predicate& predicate,
		const prepared_geometry& geometry, std::uint32_t instances = 1, std::uint32_t base_instance = 0);
	[[nodiscard]] static upload_requirement predicated_draw_upload() noexcept;
};
} // namespace metal4
