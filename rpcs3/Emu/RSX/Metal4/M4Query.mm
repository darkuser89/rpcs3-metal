#include "M4Query.h"
#include "M4DrawValidation.h"
#include "M4Pipeline.h"

#include <cstring>
#include <list>
#include <mutex>
#include <vector>

namespace metal4
{
namespace
{
constexpr std::size_t max_conditional_segments = 16384;
struct reduction_range
{
	std::uint32_t first, end, accumulated, precise, consumed;
};
struct reduction_counter
{
	std::uint32_t index, observation, committed_prefix;
};
struct reduction_parameters
{
	MTLDrawIndexedPrimitivesIndirectArguments draw;
	std::uint32_t query_count, indexed, retired, sum_all;
};
static_assert(sizeof(reduction_range) == 20 && sizeof(reduction_counter) == 12 && sizeof(reduction_parameters) == 36);
static_assert(sizeof(MTLDrawPrimitivesIndirectArguments) == 16 && sizeof(MTLDrawIndexedPrimitivesIndirectArguments) == 20);
constexpr std::string_view conditional_shader = R"msl(
#include <metal_stdlib>
using namespace metal;
struct query_range { uint first, end, accumulated, precise, consumed; };
struct query_counter { uint index, observation, committed_prefix; };
struct indexed_arguments { uint count, instances, start; int base_vertex; uint base_instance; };
struct parameters { indexed_arguments draw; uint query_count, indexed, retired, sum_all; };
uint reduce_report(const device ulong* counters, const device query_counter* indices,
    const device query_range* ranges, constant parameters& params, device uint* outcomes)
{
    uint statistic = params.retired;
    for (uint i = 0; i < params.query_count && (params.sum_all || statistic == 0); ++i)
    {
        const query_range range = ranges[i];
        uint result = range.accumulated;
        for (uint j = range.first; j < range.end; ++j)
        {
            const query_counter source = indices[j];
            if (source.observation)
            {
                const uint outcome = outcomes[source.observation];
                if (outcome == 2) // A relevant producer failed after commit.
                {
                    outcomes[0] = 1;
                    return 0;
                }
                if (outcome == 0 && !source.committed_prefix) continue;
            }
            const ulong counter = counters[source.index];
            if (range.precise) result += uint(counter);
            else result |= uint(counter != 0);
        }
        statistic += range.consumed + result;
    }
    return statistic;
}

void write_draw(uint statistic, constant indexed_arguments& draw, uint indexed, device uint* output)
{
    output[0] = draw.count;
    output[1] = statistic ? draw.instances : 0;
    output[2] = draw.start;
    // Metal's indexed ABI inserts base_vertex before base_instance.
    if (indexed)
    {
        output[3] = as_type<uint>(draw.base_vertex);
        output[4] = draw.base_instance;
    }
    else output[3] = draw.base_instance;
}

kernel void rpcs3_conditional_draw(
    const device ulong* counters [[buffer(0)]], const device query_counter* indices [[buffer(1)]],
    const device query_range* ranges [[buffer(2)]], constant parameters& params [[buffer(3)]],
    device uint* output [[buffer(4)]], device uint* outcomes [[buffer(5)]])
{
    write_draw(reduce_report(counters, indices, ranges, params, outcomes), params.draw, params.indexed, output);
}

kernel void rpcs3_capture_report(
    const device ulong* counters [[buffer(0)]], const device query_counter* indices [[buffer(1)]],
    const device query_range* ranges [[buffer(2)]], constant parameters& params [[buffer(3)]],
    device uint& output [[buffer(4)]], device uint* outcomes [[buffer(5)]])
{
    output = reduce_report(counters, indices, ranges, params, outcomes);
}

kernel void rpcs3_predicated_draw(const device uint& predicate [[buffer(0)]],
    constant parameters& params [[buffer(1)]], device uint* output [[buffer(2)]])
{
    write_draw(predicate, params.draw, params.indexed, output);
}
)msl";

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
	id<MTLComputePipelineState> conditional_pipeline, capture_pipeline, predicated_pipeline;
	std::optional<std::uint32_t> active;
	std::optional<submission> cleared_for;
	std::size_t cleared_next = 0, cleared_end = 0;
	explicit impl(context& context) : ctx(context) {}
	id<MTLComputePipelineState> pipeline(__strong id<MTLComputePipelineState>& cached, NSString* function)
	{
		if (!cached)
		{
			helper_catalog compile(ctx);
			const auto library = compile.library(conditional_shader, @"RPCS3 conditional draw");
			cached = compile.compute(library, function);
		}
		return cached;
	}
	id<MTLBuffer> private_buffer(std::size_t bytes)
	{
		return counters->budget ? counters->budget->buffer(bytes, MTLStorageModePrivate)
			: make_buffer(ctx, bytes, buffer_storage::private_gpu);
	}
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
	void initialize(const counter_lease& counter, const submission& recording)
	{
		const auto index = counter.index;
		if (cleared_for && cleared_for->same_batch(recording) && index == cleared_next && index < cleared_end)
			++cleared_next;
		else
		{
			// Clear a bounded run before opening the render pass. Future
			// queries may use the other offsets only in this recording.
			// Count only adjacent free leases in their acquisition order.
			std::size_t count = 1;
			{
				std::lock_guard lock(counters->mutex);
				for (auto it = counters->free.rbegin(); it != counters->free.rend() && count < 16; ++it)
				{
					if (*it != index + count) break;
					++count;
				}
			}
			fill_buffer(ctx, counters->buffer, index * 8, count * 8, 0);
			cleared_for = recording;
			cleared_next = index + 1;
			cleared_end = index + count;
		}
		++stats.initializations;
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

std::optional<conditional_draw> query_pool::encode_conditional_draw(std::span<const std::uint32_t> handles,
	MTLPrimitiveType primitive, std::uint32_t first, std::uint32_t count,
	std::uint32_t instances, std::uint32_t base_instance)
{
	validate_draw_ids(first, count);
	return encode_conditional_draw(handles, nullptr, conditional_draw(m_impl->ctx.recording_submission(), primitive, count, instances),
		first, base_instance);
}

std::optional<conditional_draw> query_pool::encode_conditional_draw(std::span<const std::uint32_t> handles,
	const prepared_geometry& geometry, std::uint32_t instances, std::uint32_t base_instance)
{
	return encode_conditional_draw(handles, nullptr, prepare_conditional_geometry(geometry, instances), 0, base_instance);
}

std::optional<conditional_draw> query_pool::encode_report_conditional_draw(const report_query_snapshot& report,
	const prepared_geometry& geometry, std::uint32_t instances, std::uint32_t base_instance)
{
	return encode_conditional_draw({}, &report, prepare_conditional_geometry(geometry, instances), 0, base_instance);
}

std::optional<report_predicate> query_pool::capture_report_predicate(const report_query_snapshot& report)
{
	auto& ctx = m_impl->ctx;
	const auto recording = ctx.recording_submission();
	const auto value = encode_reduction({}, &report, nullptr, 0, 0);
	if (!value) return {};
	return report_predicate(ctx.queue_identity(), recording, value);
}

std::optional<report_predicate> query_pool::submit_report_predicate(const report_query_snapshot& report)
{
	auto& ctx = m_impl->ctx;
	ctx.validate_begin();
	if (report.sources.size() > capacity())
		throw std::invalid_argument("Metal 4 report requires a bounded set of query handles");
	const auto predecessor = ctx.last_submission();
	// Retired values and CPU-only counters need no GPU reduction. Match the
	// unsigned sum and ordered boolean short circuit of reduce_report().
	std::uint32_t value = report.retired;
	bool cpu_known = true;
	if (report.precise || !value)
		for (const auto& source : report.sources)
		{
			const auto& query = m_impl->at(source.handle);
			if (!query.initialized)
				throw std::logic_error("Uninitialized native conditional query");
			if (!query.segments.empty())
			{
				cpu_known = false;
				break;
			}
			if (report.precise || !value) value += source.consumed + query.accumulated;
		}
	if (cpu_known)
	{
		const auto& budget = m_impl->counters->budget;
		const auto buffer = budget ? budget->shared_buffer(sizeof(value)) : make_buffer(ctx, sizeof(value), buffer_storage::shared);
		std::memcpy(map_buffer(buffer, 0, sizeof(value)).data(), &value, sizeof(value));
		return report_predicate(ctx.queue_identity(), predecessor, buffer, value != 0);
	}
	ctx.begin(predecessor ? &*predecessor : nullptr);
	try
	{
		auto predicate = capture_report_predicate(report);
		if (predicate) (void)ctx.submit();
		else ctx.cancel();
		return predicate;
	}
	catch (...)
	{
		try { ctx.cancel(); } catch (...) {}
		throw;
	}
}

bool report_predicate::belongs_to(const context& ctx) const
{
	return m_value && m_queue.lock() == ctx.queue_identity().lock();
}

std::optional<bool> report_predicate::settled_cpu_result(const context& ctx) const
{
	if (!belongs_to(ctx))
		throw std::invalid_argument("Native draw predicate belongs to another queue");
	if (!m_cpu_result || (m_producer && !m_producer->ready()))
		return {};
	// A known report still has to propagate a failed predecessor.
	if (m_producer) m_producer->wait(std::chrono::milliseconds::zero());
	return m_cpu_result;
}

bool report_predicate::permits_publication(std::chrono::milliseconds timeout) const
{
	// CPU surface authority uses the same immutable GPU result as the draw.
	if (m_producer) m_producer->wait(timeout);
	std::uint32_t value;
	std::memcpy(&value, map_buffer(m_value, 0, sizeof(value)).data(), sizeof(value));
	return value != 0;
}

upload_requirement query_pool::predicated_draw_upload() noexcept
{
	return {sizeof(reduction_parameters), 256};
}

conditional_draw query_pool::encode_predicated_draw(const report_predicate& predicate,
	const prepared_geometry& geometry, std::uint32_t instances, std::uint32_t base_instance)
{
	auto& ctx = m_impl->ctx;
	if (!predicate.belongs_to(ctx))
		throw std::invalid_argument("Metal 4 predicate belongs to another queue");
	validate_draw_ids(base_instance, instances);
	auto draw = prepare_conditional_geometry(geometry, instances);
	const bool indexed = draw.m_index_bytes != 0;
	const auto size = indexed ? sizeof(MTLDrawIndexedPrimitivesIndirectArguments) : sizeof(MTLDrawPrimitivesIndirectArguments);
	const reduction_parameters params{{draw.m_count, instances, 0, 0, base_instance}, 0, indexed, 0, 0};
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}));
	draw.m_arguments = m_impl->private_buffer(size);
	const auto pipeline = m_impl->pipeline(m_impl->predicated_pipeline, @"rpcs3_predicated_draw");
	if (predicate.m_producer) ctx.depend_on_predecessor(*predicate.m_producer);
	bindings resources;
	resources.set_buffer(0, predicate.m_value);
	resources.set_buffer(1, uniforms.buffer, uniforms.offset);
	resources.set_buffer(2, draw.m_arguments);
	const std::array accesses{
		context::accessed_resource{predicate.m_value, 0, sizeof(std::uint32_t)},
		context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size},
		context::accessed_resource{draw.m_arguments, 0, size, true}};
	ctx.checked_dispatch(pipeline, resources, accesses, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
	return draw;
}

conditional_draw query_pool::prepare_conditional_geometry(const prepared_geometry& geometry, std::uint32_t instances)
{
	const auto buffer = geometry.buffer();
	if (!geometry.count() || !buffer || buffer.device != m_impl->ctx.device())
		throw std::invalid_argument("Metal 4 conditional draw requires nonempty prepared geometry on the query device");
	conditional_draw draw(m_impl->ctx.recording_submission(), geometry.primitive(), geometry.count(), instances);
	draw.m_geometry = buffer;
	if (const auto width = geometry.index_width())
	{
		draw.m_index_offset = geometry.index_offset();
		draw.m_index_bytes = checked_index_bytes(buffer.length, draw.m_index_offset, geometry.count(), width);
		draw.m_index_type = width == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
	}
	return draw;
}

std::optional<conditional_draw> query_pool::encode_conditional_draw(std::span<const std::uint32_t> handles,
	const report_query_snapshot* report, conditional_draw draw, std::uint32_t first, std::uint32_t base_instance)
{
	validate_draw_ids(base_instance, draw.m_instances);
	draw.m_arguments = encode_reduction(handles, report, &draw, first, base_instance);
	if (!draw.m_arguments) return {};
	return draw;
}

id<MTLBuffer> query_pool::encode_reduction(std::span<const std::uint32_t> handles, const report_query_snapshot* report,
	const conditional_draw* draw, std::uint32_t first, std::uint32_t base_instance)
{
	const auto source_count = report ? report->sources.size() : handles.size();
	if ((!report && !source_count) || source_count > capacity())
		throw std::invalid_argument("Metal 4 conditional draw requires a bounded set of query handles");
	auto& ctx = m_impl->ctx;
	const auto recording = ctx.recording_submission();
	std::vector<reduction_range> ranges;
	std::vector<reduction_counter> indices;
	std::vector<std::shared_ptr<counter_lease>> leases;
	std::vector<submission> pending;
	bool may_fail = false;
	const auto retired = report ? report->retired : 0;
	const bool sum_all = report && report->precise;
	// A previously confirmed hit terminates a nonprecise report, even if
	// a later query is still in flight or would fail.
	const auto query_count = retired && !sum_all ? 0 : source_count;
	ranges.reserve(query_count);
	for (std::size_t i = 0; i < query_count; ++i)
	{
		const auto source = report ? report->sources[i] : report_query_source{handles[i]};
		const auto& query = m_impl->at(source.handle);
		if (!query.initialized)
			throw std::logic_error("Uninitialized native conditional query");
		const auto begin = static_cast<std::uint32_t>(indices.size());
		for (const auto& segment : query.segments)
		{
			std::uint32_t observation = 0;
			if (!segment.completion.same_batch(recording))
			{
				if (!segment.completion.ready())
				{
					if (draw) return {};
					const auto producer = std::ranges::find_if(pending, [&](const auto& batch)
						{ return batch.same_batch(segment.completion); });
					observation = 2 + static_cast<std::uint32_t>(producer - pending.begin());
					if (producer == pending.end()) pending.push_back(segment.completion);
					may_fail = true;
				}
				else if (segment.completion.cancelled())
				{
					if (!segment.committed_prefix) continue;
				}
				else
				{
					try { segment.completion.wait(); }
					catch (...)
					{
						// A nonprecise report may skip this source if an
						// earlier query hits. Let the scheduler resolve that.
						if (draw)
						{
							if (report && !sum_all) return {};
							throw;
						}
						observation = 1; // Confirmed failure, tested in report order on the GPU.
						may_fail = true;
					}
				}
			}
			if (indices.size() == max_conditional_segments) return {};
			indices.push_back({static_cast<std::uint32_t>(segment.counter->index), observation, segment.committed_prefix});
			leases.push_back(segment.counter);
		}
		ranges.push_back({begin, static_cast<std::uint32_t>(indices.size()), query.accumulated, query.precise, source.consumed});
	}
	const auto pipeline = draw ? m_impl->pipeline(m_impl->conditional_pipeline, @"rpcs3_conditional_draw")
		: m_impl->pipeline(m_impl->capture_pipeline, @"rpcs3_capture_report");
	if (indices.empty()) indices.push_back({}); // Bind a valid buffer for a prefix-only snapshot.
	if (ranges.empty()) ranges.push_back({});
	const bool indexed = draw && draw->m_index_bytes != 0;
	const reduction_parameters params{{draw ? draw->m_count : 0, draw ? draw->m_instances : 0, first, 0, base_instance},
		static_cast<std::uint32_t>(query_count), indexed, retired, sum_all};
	const auto index_data = ctx.upload(std::as_bytes(std::span{indices}), 256);
	const auto range_data = ctx.upload(std::as_bytes(std::span{ranges}), 256);
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
	auto outcomes = uniforms; // Unused by reductions with only confirmed/current sources.
	if (may_fail)
	{
		// Word 0 is written by GPU validation. Word 1 represents a producer
		// that already failed. Remaining words are filled by the submit worker.
		std::vector<std::uint32_t> initial(pending.size() + 2);
		initial[1] = 2;
		outcomes = ctx.upload(std::as_bytes(std::span{initial}));
	}
	const auto counters = m_impl->counters->buffer;
	const auto argument_size = !draw ? sizeof(std::uint32_t) :
		indexed ? sizeof(MTLDrawIndexedPrimitivesIndirectArguments) : sizeof(MTLDrawPrimitivesIndirectArguments);
	const auto& budget = m_impl->counters->budget;
	const auto arguments = draw ? m_impl->private_buffer(argument_size)
		: budget ? budget->shared_buffer(argument_size) : make_buffer(ctx, argument_size, buffer_storage::shared);
	if (may_fail)
	{
		const auto storage = map_buffer(outcomes.buffer, outcomes.offset, outcomes.size);
		for (std::size_t i = 0; i < pending.size(); ++i)
			ctx.depend_on_predecessor(pending[i], storage.data() + (i + 2) * sizeof(std::uint32_t));
		ctx.require_gpu_success(outcomes.buffer, outcomes.offset, @"Conditional report query producer failed");
	}
	for (const auto& lease : leases) ctx.keep_alive(lease);
	bindings resources;
	resources.set_buffer(0, counters);
	resources.set_buffer(1, index_data.buffer, index_data.offset);
	resources.set_buffer(2, range_data.buffer, range_data.offset);
	resources.set_buffer(3, uniforms.buffer, uniforms.offset);
	resources.set_buffer(4, arguments);
	resources.set_buffer(5, outcomes.buffer, outcomes.offset);
	const std::array accesses{
		context::accessed_resource{counters, 0, counters.length},
		context::accessed_resource{index_data.buffer, index_data.offset, index_data.size},
		context::accessed_resource{range_data.buffer, range_data.offset, range_data.size},
		context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size},
		context::accessed_resource{arguments, 0, argument_size, true},
		context::accessed_resource{outcomes.buffer, outcomes.offset, outcomes.size, may_fail}};
	ctx.checked_dispatch(pipeline, resources, accesses, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
	return arguments;
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
	m_impl->initialize(*counter, completion);
	const auto& segment = m_impl->append(*m_impl->active, counter, completion);
	return bind(segment);
}
} // namespace metal4
