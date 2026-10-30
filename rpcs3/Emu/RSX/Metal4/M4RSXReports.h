#pragma once

#include "Emu/RSX/RSXZCULL.h"
#include "M4Query.h"

namespace metal4
{
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
};
} // namespace metal4
