#include "M4Query.h"

#include <cstring>
#include <list>
#include <mutex>
#include <vector>

namespace metal4
{
namespace
{
struct arena
{
	id<MTLBuffer> buffer;
	std::shared_ptr<heap_budget> budget;
	std::mutex mutex;
	std::vector<std::size_t> free;
	std::size_t capacity = 0;
};
struct counter_lease
{
	std::shared_ptr<arena> owner;
	std::size_t index;
	~counter_lease()
	{
		std::lock_guard lock(owner->mutex);
		owner->free.push_back(index); // Capacity reserved at construction.
	}
};
} // namespace
struct query_pool::impl
{
	context& ctx;
	std::shared_ptr<arena> counters = std::make_shared<arena>();
	struct segment
	{
		std::shared_ptr<counter_lease> counter;
		submission completion;
		// Previous batches in this counter are already confirmed. If the
		// current batch is cancelled before commit, their value remains valid.
		bool committed_prefix = false;
	};
	struct query
	{
		bool precise = false, initialized = false;
		bool accumulated_valid = false;
		std::uint32_t accumulated = 0;
		std::list<segment> segments;
		bool listed = false;
		std::list<std::uint32_t>::iterator position;
	};
	std::vector<query> queries;
	std::list<std::uint32_t> occupied;
	query_pool_statistics stats;
	std::optional<std::uint32_t> active;
	explicit impl(context& context) : ctx(context) {}
	query& at(std::uint32_t handle) const
	{
		if (handle >= queries.size())
			throw std::out_of_range("Native query handle exceeds pool");
		return const_cast<query&>(queries[handle]);
	}
	std::shared_ptr<counter_lease> acquire()
	{
		std::lock_guard lock(counters->mutex);
		if (counters->free.empty())
			throw std::runtime_error("Native visibility counter arena exhausted");
		const auto index = counters->free.back();
		auto result = std::make_shared<counter_lease>();
		result->owner = counters;
		result->index = index;
		counters->free.pop_back();
		++stats.acquired;
		return result;
	}
	void unlist(query& query)
	{
		if (query.listed) { occupied.erase(query.position); query.listed = false; }
	}
	void clear_segments(query& query)
	{
		stats.live_segments -= query.segments.size();
		query.segments.clear();
		unlist(query);
	}
	static std::uint64_t value(const segment& segment)
	{
		// A cancelled batch never executes its zero/draw writes. A counter
		// without a confirmed prefix is therefore uninitialized, not zeroed.
		if (!segment.completion.cancelled())
		{
			try { segment.completion.wait(); }
			catch (...) { if (!segment.completion.cancelled()) throw; }
		}
		if (segment.completion.cancelled() && !segment.committed_prefix) return 0;
		std::uint64_t result;
		std::memcpy(&result, static_cast<const std::byte*>(segment.counter->owner->buffer.contents) + segment.counter->index * 8, 8);
		return result;
	}
	void reclaim_query(query& query, bool keep_tail)
	{
		for (auto it = query.segments.begin(); it != query.segments.end();)
		{
			if (keep_tail && std::next(it) == query.segments.end()) break;
			++stats.retirement_visits;
			if (!it->completion.ready()) { ++it; continue; }
			const auto result = value(*it); // Error leaves this segment intact.
			if (query.precise) query.accumulated += static_cast<std::uint32_t>(result);
			else query.accumulated |= result != 0;
			query.accumulated_valid = true;
			it = query.segments.erase(it); --stats.live_segments;
		}
		if (query.segments.empty()) unlist(query);
	}
	void reclaim_completed()
	{
		// Visit only queries with counter work, not the whole handle table.
		for (auto it = occupied.begin(); it != occupied.end();)
		{
			const auto handle = *it++;
			reclaim_query(at(handle), false);
		}
	}
	segment& append(std::uint32_t handle, std::shared_ptr<counter_lease> counter, submission completion)
	{
		auto& query = at(handle);
		query.segments.push_back({std::move(counter), std::move(completion)});
		try
		{
			if (!query.listed) { occupied.push_back(handle); query.position = std::prev(occupied.end()); query.listed = true; }
		}
		catch (...) { query.segments.pop_back(); throw; }
		++stats.live_segments;
		return query.segments.back();
	}
};
query_pool::query_pool(context& ctx, std::size_t queries, std::size_t segments, std::shared_ptr<heap_budget> budget) : m_impl(std::make_unique<impl>(ctx))
{
	if (!queries || queries > 2048 || !segments || segments > 1u << 20)
		throw std::invalid_argument("Native query/counter capacities exceed supported bounds");
	if (budget && budget->device() != ctx.device())
		throw std::invalid_argument("Native query budget belongs to another device");
	m_impl->queries.resize(queries);
	m_impl->counters->budget = std::move(budget);
	m_impl->counters->buffer = m_impl->counters->budget ? m_impl->counters->budget->shared_buffer(segments * 8) : make_buffer(ctx, segments * 8, buffer_storage::shared);
	m_impl->counters->free.reserve(segments);
	m_impl->counters->capacity = segments;
	for (std::size_t index = segments; index; --index)
		m_impl->counters->free.push_back(index - 1);
}
query_pool::~query_pool() = default;
std::size_t query_pool::capacity() const noexcept { return m_impl->queries.size(); }
query_pool_statistics query_pool::statistics() const
{
	auto result = m_impl->stats;
	std::lock_guard lock(m_impl->counters->mutex);
	result.leased_counters = m_impl->counters->capacity - m_impl->counters->free.size();
	return result;
}
void query_pool::begin(std::uint32_t handle, bool precise)
{
	auto& query = m_impl->at(handle);
	if (m_impl->active || !query.segments.empty() || query.accumulated_valid)
		throw std::logic_error("Native query begin conflicts with active/unretired work");
	query.precise = precise;
	query.initialized = true;
	m_impl->active = handle;
}
void query_pool::end(std::uint32_t handle)
{
	(void)m_impl->at(handle);
	if (!m_impl->active || *m_impl->active != handle)
		throw std::logic_error("Ending a non-active native query");
	m_impl->active.reset();
}
void query_pool::discard(std::uint32_t handle)
{
	auto& query = m_impl->at(handle);
	if (m_impl->active == handle)
		m_impl->active.reset();
	m_impl->clear_segments(query);
	query.initialized = false;
	query.accumulated = 0;
	query.accumulated_valid = false;
}
bool query_pool::ready(std::uint32_t handle) const
{
	const auto& query = m_impl->at(handle);
	if (!query.initialized)
		throw std::logic_error("Uninitialized native query");
	for (const auto& segment : query.segments)
		if (!segment.completion.ready())
			return false;
	return true;
}
std::uint32_t query_pool::consume(std::uint32_t handle)
{
	auto& query = m_impl->at(handle);
	if (!query.initialized)
		throw std::logic_error("Uninitialized native query");
	std::uint32_t result = query.accumulated;
	for (const auto& segment : query.segments)
	{
		const auto value = impl::value(segment);
		if (query.precise)
			result += static_cast<std::uint32_t>(value);
		else
			result |= value != 0;
	}
	m_impl->clear_segments(query);
	query.accumulated = 0;
	query.accumulated_valid = false;
	return result;
}
std::optional<visibility_binding> query_pool::attach(context& ctx, MTL4RenderPassDescriptor* pass)
{
	if (&ctx != &m_impl->ctx || !pass)
		throw std::invalid_argument("Native visibility requires its owning context/pass");
	if (!m_impl->active)
		return {};
	if (pass.visibilityResultBuffer)
		throw std::invalid_argument("Native query cannot replace an existing visibility buffer");
	auto& query = m_impl->at(*m_impl->active);
	const auto completion = ctx.recording_submission();
	m_impl->reclaim_query(query, true);
	const auto bind = [&](const auto& segment)
	{
		pass.visibilityResultBuffer = segment.counter->owner->buffer;
		pass.visibilityResultType = MTLVisibilityResultTypeAccumulate;
		return visibility_binding{query.precise ? MTLVisibilityResultModeCounting : MTLVisibilityResultModeBoolean, segment.counter->index * 8};
	};
	if (!query.segments.empty())
	{
		auto& tail = query.segments.back();
		if (tail.completion.same_batch(completion))
		{
			++m_impl->stats.reused_batch;
			return bind(tail);
		}
		if (tail.completion.ready())
		{
			// No blocking wait: ready is immutable. Confirm error status before
			// replacing the sole completion token for this accumulated counter.
			if (!tail.completion.cancelled()) { tail.completion.wait(); tail.committed_prefix = true; }
			ctx.keep_alive(tail.counter);
			if (!tail.committed_prefix)
			{
				fill_buffer(ctx, tail.counter->owner->buffer, tail.counter->index * 8, 8, 0);
				++m_impl->stats.initializations;
			}
			tail.completion = completion;
			++m_impl->stats.reused_completed;
			return bind(tail);
		}
	}
	bool exhausted;
	{
		std::lock_guard lock(m_impl->counters->mutex);
		exhausted = m_impl->counters->free.empty();
	}
	if (exhausted)
		m_impl->reclaim_completed();
	const auto counter = m_impl->acquire();
	ctx.keep_alive(counter);
	const auto offset = counter->index * 8;
	fill_buffer(ctx, counter->owner->buffer, offset, 8, 0);
	++m_impl->stats.initializations;
	const auto& segment = m_impl->append(*m_impl->active, counter, completion);
	return bind(segment);
}
} // namespace metal4
