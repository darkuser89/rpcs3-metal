#include "M4CompilerQueue.h"

#include <condition_variable>
#include <algorithm>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <dispatch/dispatch.h>

namespace metal4
{
namespace
{
void validate_function(id<MTLDevice> device, MTL4FunctionDescriptor* descriptor)
{
	if ([descriptor isKindOfClass:[MTL4LibraryFunctionDescriptor class]])
	{
		const auto function = static_cast<MTL4LibraryFunctionDescriptor*>(descriptor);
		if (!function.library || function.library.device != device || !function.name.length ||
			![function.library.functionNames containsObject:function.name])
			throw std::invalid_argument("Native Metal4 compiler function is missing or belongs to another device");
	}
}
}
struct compiler_job_state
{
	compiler_job_kind kind;
	std::mutex mutex;
	std::condition_variable changed;
	bool done = false, cancelled = false, active = false;
	id value = nil;
	NSError* error = nil;
	id inputs = nil;
	id<MTL4CompilerTask> task = nil;
	std::string source;
	std::size_t hash = 0;
};

bool compiler_job::ready() const { std::lock_guard lock(m_state->mutex); return m_state->done; }
bool compiler_job::cancelled() const { std::lock_guard lock(m_state->mutex); return m_state->cancelled; }
id compiler_job::result(compiler_job_kind kind, std::chrono::milliseconds timeout) const
{
	if (m_state->kind != kind || timeout.count() < 0) throw std::invalid_argument("Metal4 compiler result kind or timeout is invalid");
	std::unique_lock lock(m_state->mutex);
	if (!m_state->changed.wait_for(lock, timeout, [&] { return m_state->done; }))
		throw std::runtime_error("Timed out waiting for native Metal4 compilation");
	if (m_state->cancelled) throw std::runtime_error("Native Metal4 compilation cancelled before launch");
	if (m_state->error || !m_state->value)
		throw std::runtime_error(m_state->error ? m_state->error.localizedDescription.UTF8String : "Native Metal4 compilation failed without a result");
	return m_state->value;
}
id<MTLLibrary> compiler_job::library(std::chrono::milliseconds timeout) const { return result(compiler_job_kind::library, timeout); }
id<MTLComputePipelineState> compiler_job::compute(std::chrono::milliseconds timeout) const { return result(compiler_job_kind::compute, timeout); }
id<MTLRenderPipelineState> compiler_job::render(std::chrono::milliseconds timeout) const { return result(compiler_job_kind::render, timeout); }

struct compiler_queue::implementation : std::enable_shared_from_this<implementation>
{
	id<MTL4Compiler> native;
	dispatch_queue_t worker = dispatch_queue_create("rpcs3.metal4.compiler.launch", DISPATCH_QUEUE_SERIAL);
	compiler_queue_limits limits;
	mutable std::mutex mutex;
	std::condition_variable changed;
	bool closed = false;
	compiler_queue_statistics stats;
	std::deque<std::shared_ptr<compiler_job_state>> pending;
	std::unordered_multimap<std::size_t, std::weak_ptr<compiler_job_state>> libraries;

	void pump()
	{
		const auto owner = shared_from_this();
		for (;;)
		{
			std::shared_ptr<compiler_job_state> job;
			{
				std::lock_guard lock(mutex);
				if (closed || stats.active == limits.active || pending.empty()) return;
				job = pending.front(); pending.pop_front(); job->active = true;
				--stats.queued; ++stats.active;
				stats.peak_active = std::max(stats.peak_active, stats.active);
			}
			dispatch_async(worker, ^{ @autoreleasepool { owner->start(job); } });
		}
	}
	void finish(const std::shared_ptr<compiler_job_state>& job, id value, NSError* error, bool cancelled = false)
	{
		id inputs;
		id<MTL4CompilerTask> task;
		{
			std::lock_guard lock(job->mutex);
			if (job->done) return;
			job->value = value; job->error = error; job->cancelled = cancelled;
			inputs = job->inputs; job->inputs = nil;
			task = job->task; job->task = nil;
			job->done = true;
			job->changed.notify_all();
		}
		inputs = nil; // Retire copied descriptors outside both locks.
		task = nil;
		{
			std::lock_guard lock(mutex);
			if (job->active) --stats.active;
			stats.source_bytes -= job->source.size();
			if (cancelled) ++stats.cancelled;
			else if (error || !value) ++stats.failed;
			else ++stats.completed;
			if (job->kind == compiler_job_kind::library)
			{
				const auto [first, last] = libraries.equal_range(job->hash);
				for (auto it = first; it != last; ++it)
					if (it->second.lock() == job) { libraries.erase(it); break; }
			}
			std::string().swap(job->source);
			changed.notify_all();
		}
		pump();
	}
	void start(const std::shared_ptr<compiler_job_state>& job)
	{
		bool stopped;
		{
			std::lock_guard lock(mutex);
			stopped = closed;
		}
		if (stopped) { finish(job, nil, nil, true); return; }
		const auto owner = shared_from_this();
		const auto completion_job = job; // Never capture a reference parameter in a late block.
		id<MTL4CompilerTask> task;
		switch (job->kind)
		{
		case compiler_job_kind::library:
		{
			task = [native newLibraryWithDescriptor:job->inputs completionHandler:^(id<MTLLibrary> value, NSError* error) { owner->finish(completion_job, value, error); }];
			break;
		}
		case compiler_job_kind::compute:
		{
			task = [native newComputePipelineStateWithDescriptor:job->inputs compilerTaskOptions:nil completionHandler:^(id<MTLComputePipelineState> value, NSError* error) { owner->finish(completion_job, value, error); }];
			break;
		}
		case compiler_job_kind::render:
		{
			task = [native newRenderPipelineStateWithDescriptor:job->inputs compilerTaskOptions:nil completionHandler:^(id<MTLRenderPipelineState> value, NSError* error) { owner->finish(completion_job, value, error); }];
			break;
		}
		}
		if (!task) { finish(job, nil, nil); return; }
		std::lock_guard lock(job->mutex);
		if (!job->done) job->task = task;
	}
	std::optional<compiler_job> admit(std::shared_ptr<compiler_job_state> job, std::chrono::milliseconds timeout)
	{
		if (timeout.count() < 0 || job->source.size() > limits.source_bytes)
			throw std::invalid_argument("Metal4 compiler admission timeout or source quota is invalid");
		const auto deadline = std::chrono::steady_clock::now() + timeout;
		std::unique_lock lock(mutex);
		for (;;)
		{
			if (closed) throw std::logic_error("Native Metal4 compiler queue is shut down");
			if (job->kind == compiler_job_kind::library)
			{
				const auto [first, last] = libraries.equal_range(job->hash);
				for (auto it = first; it != last; ++it)
					if (auto other = it->second.lock(); other && other->source == job->source)
					{
						std::lock_guard result_lock(other->mutex);
						if (!other->done) { ++stats.deduplicated; return compiler_job(std::move(other)); }
					}
			}
			if (stats.active + stats.queued < limits.jobs && job->source.size() <= limits.source_bytes - stats.source_bytes) break;
			if (!timeout.count()) return {};
			++stats.waiting;
			const auto status = changed.wait_until(lock, deadline);
			--stats.waiting;
			if (status == std::cv_status::timeout) return {};
		}
		pending.push_back(job);
		try { if (job->kind == compiler_job_kind::library) libraries.emplace(job->hash, job); }
		catch (...) { pending.pop_back(); throw; }
		++stats.admitted; ++stats.queued; stats.source_bytes += job->source.size();
		changed.notify_all();
		lock.unlock();
		pump();
		return compiler_job(std::move(job));
	}
	void close()
	{
		{
			std::lock_guard lock(mutex);
			closed = true;
			changed.notify_all();
		}
		for (;;)
		{
			std::shared_ptr<compiler_job_state> job;
			{
				std::lock_guard lock(mutex);
				if (pending.empty()) break;
				job = pending.front(); pending.pop_front(); --stats.queued;
			}
			finish(job, nil, nil, true);
		}
	}
};

compiler_queue::compiler_queue(id<MTL4Compiler> compiler, compiler_queue_limits limits)
{
	if (!compiler.device || !limits.active || limits.active > limits.jobs || limits.jobs > 4096 || !limits.source_bytes)
		throw std::invalid_argument("Native Metal4 compiler queue limits or device are invalid");
	m_impl = std::make_shared<implementation>(); m_impl->native = compiler; m_impl->limits = limits;
}
compiler_queue::~compiler_queue() { m_impl->close(); }

std::optional<compiler_job> compiler_queue::library(std::string_view source, NSString* label, std::chrono::milliseconds timeout)
{
	if (source.empty() || source.size() > m_impl->limits.source_bytes) throw std::invalid_argument("Native Metal4 compiler source is empty or exceeds its job quota");
	auto descriptor = [MTL4LibraryDescriptor new];
	descriptor.source = [[NSString alloc] initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];
	if (!descriptor.source) throw std::invalid_argument("Native Metal4 compiler source must be UTF-8");
	descriptor.name = label;
	auto options = [MTLCompileOptions new];
	options.languageVersion = MTLLanguageVersion4_0;
	options.mathMode = MTLMathModeSafe;
	options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
	options.preserveInvariance = YES;
	descriptor.options = options;
	auto job = std::make_shared<compiler_job_state>(); job->kind = compiler_job_kind::library;
	job->inputs = descriptor; job->source.assign(source); job->hash = std::hash<std::string_view>{}(source);
	return m_impl->admit(std::move(job), timeout);
}
std::optional<compiler_job> compiler_queue::compute(MTL4ComputePipelineDescriptor* descriptor, std::chrono::milliseconds timeout)
{
	if (!descriptor.computeFunctionDescriptor) throw std::invalid_argument("Native Metal4 compute job requires a function");
	validate_function(m_impl->native.device, descriptor.computeFunctionDescriptor);
	MTL4ComputePipelineDescriptor* copy = [descriptor copy];
	copy.computeFunctionDescriptor = descriptor.computeFunctionDescriptor;
	auto job = std::make_shared<compiler_job_state>(); job->kind = compiler_job_kind::compute; job->inputs = copy;
	return m_impl->admit(std::move(job), timeout);
}
std::optional<compiler_job> compiler_queue::render(MTL4RenderPipelineDescriptor* descriptor, std::chrono::milliseconds timeout)
{
	if (!descriptor.vertexFunctionDescriptor) throw std::invalid_argument("Native Metal4 render job requires a vertex function");
	validate_function(m_impl->native.device, descriptor.vertexFunctionDescriptor);
	validate_function(m_impl->native.device, descriptor.fragmentFunctionDescriptor);
	MTL4RenderPipelineDescriptor* copy = [descriptor copy];
	copy.vertexFunctionDescriptor = descriptor.vertexFunctionDescriptor;
	copy.fragmentFunctionDescriptor = descriptor.fragmentFunctionDescriptor;
	auto job = std::make_shared<compiler_job_state>(); job->kind = compiler_job_kind::render; job->inputs = copy;
	return m_impl->admit(std::move(job), timeout);
}
compiler_queue_statistics compiler_queue::statistics() const { std::lock_guard lock(m_impl->mutex); return m_impl->stats; }
bool compiler_queue::shutdown(std::chrono::milliseconds timeout)
{
	if (timeout.count() < 0) throw std::invalid_argument("Native Metal4 compiler shutdown timeout is negative");
	m_impl->close();
	std::unique_lock lock(m_impl->mutex);
	return m_impl->changed.wait_for(lock, timeout, [&]
	{
		const auto& stats = m_impl->stats;
		return stats.admitted == stats.completed + stats.failed + stats.cancelled;
	});
}
} // namespace metal4
