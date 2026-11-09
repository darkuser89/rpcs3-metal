#pragma once

#include "Emu/RSX/RSXZCULL.h"
#include "M4Query.h"

#include <algorithm>
#include <bitset>

namespace metal4
{
struct captured_report_condition
{
	std::vector<report_query_source> sources;
	std::uint32_t retired;
	bool precise;

	[[nodiscard]] report_query_snapshot snapshot() const & { return {sources, retired, precise}; }
	report_query_snapshot snapshot() const && = delete;
};

namespace detail
{
// A terminal ZPASS writer evaluates the statistics slot at its retirement.
// Forwarded writers instead evaluate guest memory and require the CPU path.
[[nodiscard]] inline std::optional<captured_report_condition> capture_report_condition(
	std::span<const rsx::reports::queued_report_write> writers,
	std::span<const rsx::reports::query_stat_counter> statistics, std::uint32_t address,
	std::uint64_t sync_tag, bool precise, std::size_t query_capacity)
{
	if (!address || !sync_tag || query_capacity > rsx::reports::occlusion_query_count)
		return {};
	const auto terminal = std::ranges::find_if(writers, [&](const auto& writer)
	{
		return writer.query && writer.query->sync_tag == sync_tag;
	});
	if (terminal == writers.end() || terminal->sink != address || terminal->forwarder ||
		terminal->type != CELL_GCM_ZPASS_PIXEL_CNT || terminal->counter_tag >= statistics.size() ||
		!terminal->query->pending || !terminal->query->num_draws)
		return {};
	captured_report_condition condition{{}, statistics[terminal->counter_tag].result, precise};
	std::bitset<rsx::reports::occlusion_query_count> captured;
	for (const auto& writer : writers.first(terminal - writers.begin() + 1))
	{
		if (!writer.sink) return {}; // The scheduler stops at an unclaimed writer.
		if (writer.counter_tag != terminal->counter_tag || !writer.query || !writer.query->num_draws ||
			(writer.type != CELL_GCM_ZPASS_PIXEL_CNT && writer.type != CELL_GCM_ZCULL_STATS3))
			continue;
		const auto& query = *writer.query;
		if (!query.pending || query.driver_handle >= query_capacity || captured.test(query.driver_handle))
			return {};
		captured.set(query.driver_handle);
		condition.sources.push_back({query.driver_handle, query.result});
	}
	return condition;
}
} // namespace detail

// Native backend methods of the actual RSX report controller. Base scheduler
// owns report policy/protection; every immediate/queued/aliased store is routed
// through the backend's coherent atomic writer after scaling and MM ordering.
class rsx_report_controller final : public rsx::reports::ZCULL_control
{
	query_pool& m_pool;
	std::function<bool()> m_precise;
	std::function<void(std::uint32_t, std::uint64_t, std::uint32_t)> m_store;
	std::function<void()> m_submit;
	static std::uint32_t handle(rsx::reports::occlusion_query_info* query)
	{
		if (!query)
			throw std::invalid_argument("Native RSX report query is null");
		return query->driver_handle;
	}
	void store_report(vm::addr_t sink, u64 timestamp, u32 value) override
	{
		m_store(sink, timestamp, value);
	}

  public:
	rsx_report_controller(query_pool& pool, std::function<bool()> precise,
						  std::function<void(std::uint32_t, std::uint64_t, std::uint32_t)> store,
						  std::function<void()> submit)
		: m_pool(pool), m_precise(std::move(precise)), m_store(std::move(store)), m_submit(std::move(submit))
	{
		if (!m_precise || !m_store || !m_submit)
			throw std::invalid_argument("Native RSX reports require precise-count policy, coherent atomic writer and submission boundary");
		if (pool.capacity() < m_occlusion_query_data.size())
			throw std::invalid_argument("Native report pool is smaller than RSX scheduler slots");
		for (std::size_t index = 0; index < m_occlusion_query_data.size(); ++index)
			m_occlusion_query_data[index].driver_handle = static_cast<std::uint32_t>(index);
	}
	void begin_occlusion_query(rsx::reports::occlusion_query_info* query) override
	{
		m_pool.begin(handle(query), m_precise());
		query->result = 0;
	}
	void end_occlusion_query(rsx::reports::occlusion_query_info* query) override { m_pool.end(handle(query)); }
	bool check_occlusion_query_status(rsx::reports::occlusion_query_info* query) override
	{
		const auto id = handle(query);
		m_submit(); // Nonblocking: an unsubmitted recording token can never be ready.
		return m_pool.ready(id);
	}
	void get_occlusion_query_result(rsx::reports::occlusion_query_info* query) override
	{
		const auto id = handle(query);
		m_submit(); // Submit the recording before consume waits on its completion.
		query->result += m_pool.consume(id);
	}
	void discard_occlusion_query(rsx::reports::occlusion_query_info* query) override { m_pool.discard(handle(query)); }
	// Capture and encode together, before report retirement can recycle any
	// driver handle. The resulting predicate owns no RSX query pointers.
	[[nodiscard]] std::optional<report_predicate> submit_conditional_predicate(std::uint32_t address, std::uint64_t sync_tag) const
	{
		const auto report = detail::capture_report_condition(m_pending_writes, m_statistics_map,
			address, sync_tag, m_precise(), m_pool.capacity());
		return report ? m_pool.submit_report_predicate(report->snapshot()) : std::nullopt;
	}
};
} // namespace metal4
