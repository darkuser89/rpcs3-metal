#include "M4Pipeline.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <atomic>
#include <list>
#include <stdexcept>
#include <string>
#include <mutex>
#include <vector>
#include <unordered_map>

namespace metal4
{

struct compiler::library_cache
{
	struct entry
	{
		std::string source;
		std::size_t fingerprint;
		id<MTLLibrary> library;
	};
	std::list<entry> entries;
	std::unordered_multimap<std::size_t, std::list<entry>::iterator> index;
	void erase_index(std::list<entry>::iterator entry)
	{
		const auto [first, last] = index.equal_range(entry->fingerprint);
		for (auto it = first; it != last; ++it)
			if (it->second == entry) { index.erase(it); return; }
	}
	library_cache_statistics stats;
	std::atomic<std::size_t> prepared_source_bytes{0};
	std::uint64_t generation = 0;
	std::vector<std::weak_ptr<library_preparation>> pending;
	id<MTLLibrary> find(std::string_view source)
	{
		const auto [first, last] = index.equal_range(std::hash<std::string_view>{}(source));
		for (auto candidate = first; candidate != last; ++candidate)
		{
			++stats.candidates;
			const auto it = candidate->second;
			if (it->source != source) continue;
			const auto result = it->library;
			entries.splice(entries.begin(), entries, it);
			++stats.reused;
			return result;
		}
		return nil;
	}
	void retain(std::string_view source, id<MTLLibrary> library)
	{
		++stats.created;
		if (!stats.capacity || source.size() > stats.source_budget) { ++stats.bypassed; return; }
		entries.push_front({std::string(source), std::hash<std::string_view>{}(source), library});
		try { index.emplace(entries.front().fingerprint, entries.begin()); }
		catch (...) { entries.pop_front(); throw; }
		while (entries.size() > stats.capacity || stats.source_bytes > stats.source_budget - source.size())
		{
			stats.source_bytes -= entries.back().source.size();
			erase_index(std::prev(entries.end())); entries.pop_back(); ++stats.evicted;
		}
		stats.source_bytes += source.size(); stats.entries = entries.size();
	}
};

struct compiler::library_preparation
{
	std::shared_ptr<library_cache> owner;
	std::uint64_t generation;
	std::string source;
	std::optional<compiler_job> job;
	id<MTLLibrary> resolved = nil;
	bool failed = false;
	std::size_t source_charge = 0;

	~library_preparation()
	{
		if (source_charge)
		{
			std::string{}.swap(source);
			owner->prepared_source_bytes.fetch_sub(source_charge, std::memory_order_relaxed);
		}
	}
};

bool compiler::prepared_library::ready() const { return m_state->resolved || m_state->job->ready(); }
bool compiler::prepared_library::failed() const noexcept { return m_state->failed; }
compiler_function compiler::prepared_library::function(NSString* name) const
{
	return m_state->resolved ? compiler_function::from_library(m_state->resolved, name) : m_state->job->function(name);
}
std::size_t compiler::prepared_library::retained_bytes() const noexcept
{
	return sizeof(library_preparation) + m_state->source.capacity() + 1;
}

compiler::compiler(context& ctx, std::size_t library_capacity, std::size_t source_budget, id<MTL4Compiler> native)
{
	if (library_capacity > 4096 || (library_capacity && !source_budget))
		throw std::invalid_argument("Metal4 library cache capacity/source budget is invalid");
	m_libraries = std::make_shared<library_cache>();
	m_libraries->stats.capacity = library_capacity;
	m_libraries->stats.source_budget = source_budget;
	const auto queue_budget = compiler_queue_limits{}.source_bytes;
	m_libraries->stats.prepared_source_budget = source_budget ? std::min(source_budget, queue_budget) : queue_budget;
	MTL4CompilerDescriptor* descriptor = [MTL4CompilerDescriptor new];
	descriptor.label = @"RPCS3 Metal 4 compiler";
	NSError* error = nil;
	if (native && native.device != ctx.device()) throw std::invalid_argument("Native compiler belongs to another Metal4 context device");
	m_compiler = native ? native : [ctx.device() newCompilerWithDescriptor:descriptor error:&error];
	if (!m_compiler)
	{
		throw std::runtime_error(error ? error.localizedDescription.UTF8String : "Metal 4 compiler creation failed");
	}
	m_jobs = std::make_shared<compiler_queue>(m_compiler);
}

id<MTLDevice> compiler::device() const noexcept
{
	return m_compiler.device;
}

id<MTLLibrary> compiler::library(std::string_view msl, NSString* label) const
{
	@autoreleasepool
	{
		if (msl.empty())
		{
			throw std::invalid_argument("Metal 4 shader source is empty");
		}
		if (const auto cached = m_libraries->find(msl)) return cached;
		for (const auto& weak : m_libraries->pending)
			if (const auto pending = weak.lock(); pending && pending->generation == m_libraries->generation && !pending->resolved && !pending->failed && pending->source == msl)
				return library(prepared_library(pending));
		const auto job = m_jobs->library(msl, label, std::chrono::seconds(30));
		if (!job) throw std::runtime_error("Timed out admitting native Metal4 library compilation");
		const auto library = job->library();
		m_libraries->retain(msl, library);
		return library;
	}
}

std::optional<compiler::prepared_library> compiler::prewarm_library(std::string_view msl, NSString* label) const
{
	if (msl.empty()) throw std::invalid_argument("Metal4 prewarm source is empty");
	if (msl.size() > compiler_queue_limits{}.source_bytes)
		throw std::invalid_argument("Metal4 prewarm source exceeds the native compiler quota");
	if (const auto cached = m_libraries->find(msl))
	{
		auto state = std::make_shared<library_preparation>(); state->owner = m_libraries; state->generation = m_libraries->generation; state->resolved = cached;
		return prepared_library(std::move(state));
	}
	std::erase_if(m_libraries->pending, [&](const auto& weak)
		{ const auto entry = weak.lock(); return !entry || entry->resolved || entry->failed || entry->generation != m_libraries->generation; });
	for (const auto& weak : m_libraries->pending)
		if (const auto pending = weak.lock(); pending && pending->source == msl) return prepared_library(pending);
	// The queue can release its own charge as soon as a job completes. A
	// retained ticket still owns an exact source key, even across clear().
	const auto prepared_bytes = m_libraries->prepared_source_bytes.load(std::memory_order_relaxed);
	if (m_libraries->pending.size() == 64 || msl.size() > m_libraries->stats.prepared_source_budget - prepared_bytes)
		return {};
	auto state = std::make_shared<library_preparation>();
	state->owner = m_libraries; state->generation = m_libraries->generation;
	state->source.assign(msl);
	state->source_charge = state->source.size();
	m_libraries->prepared_source_bytes.fetch_add(state->source_charge, std::memory_order_relaxed);
	state->job = m_jobs->library(state->source, label);
	if (!state->job) return {};
	m_libraries->pending.push_back(state);
	return prepared_library(std::move(state));
}

id<MTLLibrary> compiler::library(const prepared_library& prepared, std::chrono::milliseconds timeout) const
{
	const auto& state = prepared.m_state;
	if (state->owner != m_libraries || timeout.count() < 0)
		throw std::invalid_argument("Metal4 prepared library is foreign or has an invalid timeout");
	if (state->resolved) return state->resolved;
	id<MTLLibrary> library;
	try { library = state->job->library(timeout); }
	catch (...)
	{
		if (state->job->ready())
			try { (void)state->job->library(std::chrono::milliseconds(0)); } catch (...) { state->failed = true; }
		throw;
	}
	if (state->generation == m_libraries->generation)
	{
		if (const auto cached = m_libraries->find(state->source)) state->resolved = cached;
		else { m_libraries->retain(state->source, library); state->resolved = library; }
	}
	else state->resolved = library; // An old ticket cannot repopulate a cleared cache.
	state->job.reset();
	return state->resolved;
}

library_cache_statistics compiler::library_statistics() const noexcept
{
	auto result = m_libraries->stats;
	result.prepared_source_bytes = m_libraries->prepared_source_bytes.load(std::memory_order_relaxed);
	return result;
}

void compiler::clear_libraries()
{
	++m_libraries->generation;
	m_libraries->pending.clear();
	m_libraries->stats.evicted += m_libraries->entries.size();
	m_libraries->index.clear();
	m_libraries->entries.clear();
	m_libraries->stats.entries = m_libraries->stats.source_bytes = 0;
}

id<MTLComputePipelineState> compiler::compute(MTL4ComputePipelineDescriptor* descriptor) const
{
	if (!descriptor || !descriptor.computeFunctionDescriptor)
	{
		throw std::invalid_argument("Metal 4 compute pipeline requires a function descriptor");
	}
	const auto job = m_jobs->compute(descriptor, std::chrono::seconds(30));
	if (!job) throw std::runtime_error("Timed out admitting native Metal4 compute compilation");
	return job->compute();
}

id<MTLRenderPipelineState> compiler::render(MTL4RenderPipelineDescriptor* descriptor) const
{
	if (!descriptor || !descriptor.vertexFunctionDescriptor)
	{
		throw std::invalid_argument("Metal 4 render pipeline requires a vertex function descriptor");
	}
	const auto job = m_jobs->render(descriptor, std::chrono::seconds(30));
	if (!job) throw std::runtime_error("Timed out admitting native Metal4 render compilation");
	return job->render();
}

MTL4LibraryFunctionDescriptor* library_function(id<MTLLibrary> library, NSString* entry)
{
	if (!library || !entry.length)
	{
		throw std::invalid_argument("Metal 4 function requires a library and entry name");
	}
	MTL4LibraryFunctionDescriptor* descriptor = [MTL4LibraryFunctionDescriptor new];
	descriptor.library = library;
	descriptor.name = entry;
	return descriptor;
}

struct helper_catalog_state
{
	struct compute_entry
	{
		id<MTLLibrary> library;
		NSString* name;
		id<MTLComputePipelineState> pipeline;
	};
	struct render_entry
	{
		id<MTLLibrary> library;
		NSString* vertex;
		NSString* fragment;
		helper_render_configuration config;
		id<MTLRenderPipelineState> pipeline;
	};
	std::mutex mutex;
	compiler compile;
	std::list<compute_entry> compute;
	std::list<render_entry> render;
	helper_catalog_statistics stats;
	helper_catalog_state(context& ctx, std::size_t capacity) : compile(ctx)
	{
		stats.capacity = capacity;
	}
};

helper_catalog::helper_catalog(context& ctx)
{
	if (ctx.m_helpers) { m_state = ctx.m_helpers; return; }
	static std::mutex mutex;
	static std::vector<std::weak_ptr<helper_catalog_state>> devices;
	std::lock_guard lock(mutex);
	for (auto it = devices.begin(); it != devices.end();)
	{
		if (auto shared = it->lock())
		{
			if (shared->compile.device() == ctx.device()) { m_state = std::move(shared); break; }
			++it;
		}
		else it = devices.erase(it);
	}
	if (!m_state)
	{
		m_state = std::make_shared<helper_catalog_state>(ctx, 256);
		devices.push_back(m_state);
	}
	ctx.m_helpers = m_state;
}

helper_catalog::helper_catalog(context& ctx, std::size_t capacity)
{
	if (capacity > 4096) throw std::invalid_argument("Metal4 helper pipeline capacity exceeds 4096");
	m_state = std::make_shared<helper_catalog_state>(ctx, capacity);
}

id<MTLLibrary> helper_catalog::library(std::string_view source, NSString* label) const
{
	std::lock_guard lock(m_state->mutex);
	return m_state->compile.library(source, label);
}

id<MTLComputePipelineState> helper_catalog::compute(id<MTLLibrary> library, NSString* entry) const
{
	std::lock_guard lock(m_state->mutex);
	if (!library || library.device != m_state->compile.device() || !entry.length)
		throw std::invalid_argument("Metal4 helper compute requires a library/function on its device");
	for (auto it = m_state->compute.begin(); it != m_state->compute.end(); ++it)
		if (it->library == library && [it->name isEqualToString:entry])
		{
			const auto result = it->pipeline;
			m_state->compute.splice(m_state->compute.begin(), m_state->compute, it);
			++m_state->stats.compute_reused;
			return result;
		}
	NSString* name = [entry copy];
	// A missing function can assert inside Metal instead of yielding NSError.
	// Validate the exported name before handing the fixed profile to the driver.
	if (![library.functionNames containsObject:name])
		throw std::invalid_argument("Metal4 helper compute function is absent from its library");
	auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(library, name);
	const auto result = m_state->compile.compute(descriptor);
	++m_state->stats.compute_created;
	if (!m_state->stats.capacity) { ++m_state->stats.compute_bypassed; return result; }
	m_state->compute.push_front({library, name, result});
	if (m_state->compute.size() > m_state->stats.capacity)
	{
		m_state->compute.pop_back();
		++m_state->stats.compute_evicted;
	}
	return result;
}

id<MTLRenderPipelineState> helper_catalog::render(id<MTLLibrary> library, NSString* vertex, NSString* fragment,
	const helper_render_configuration& config) const
{
	std::lock_guard lock(m_state->mutex);
	if (!library || library.device != m_state->compile.device() || !vertex.length || !fragment.length ||
		(config.samples != 1 && config.samples != 2 && config.samples != 4 && config.samples != 8) ||
		![library.device supportsTextureSampleCount:config.samples])
		throw std::invalid_argument("Metal4 helper render function/device/sample state is invalid");
	for (auto mask : config.masks)
		if (mask & ~MTLColorWriteMaskAll) throw std::invalid_argument("Metal4 helper render write mask is invalid");
	for (auto it = m_state->render.begin(); it != m_state->render.end(); ++it)
		if (it->library == library && [it->vertex isEqualToString:vertex] && [it->fragment isEqualToString:fragment] && it->config == config)
		{
			const auto result = it->pipeline;
			m_state->render.splice(m_state->render.begin(), m_state->render, it);
			++m_state->stats.render_reused;
			return result;
		}
	NSString* v = [vertex copy];
	NSString* f = [fragment copy];
	if (![library.functionNames containsObject:v] || ![library.functionNames containsObject:f])
		throw std::invalid_argument("Metal4 helper render function is absent from its library");
	auto descriptor = [MTL4RenderPipelineDescriptor new];
	descriptor.vertexFunctionDescriptor = library_function(library, v);
	descriptor.fragmentFunctionDescriptor = library_function(library, f);
	descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	descriptor.rasterSampleCount = config.samples;
	for (unsigned i = 0; i < config.colors.size(); ++i)
	{
		descriptor.colorAttachments[i].pixelFormat = config.colors[i];
		descriptor.colorAttachments[i].writeMask = config.masks[i];
		descriptor.colorAttachments[i].blendingState = MTL4BlendStateDisabled;
	}
	const auto result = m_state->compile.render(descriptor);
	++m_state->stats.render_created;
	if (!m_state->stats.capacity) { ++m_state->stats.render_bypassed; return result; }
	m_state->render.push_front({library, v, f, config, result});
	if (m_state->render.size() > m_state->stats.capacity)
	{
		m_state->render.pop_back();
		++m_state->stats.render_evicted;
	}
	return result;
}

helper_catalog_statistics helper_catalog::statistics() const
{
	std::lock_guard lock(m_state->mutex);
	auto result = m_state->stats;
	result.libraries = m_state->compile.library_statistics();
	result.compute_entries = m_state->compute.size();
	result.render_entries = m_state->render.size();
	return result;
}

void helper_catalog::clear()
{
	std::lock_guard lock(m_state->mutex);
	m_state->stats.compute_evicted += m_state->compute.size();
	m_state->stats.render_evicted += m_state->render.size();
	m_state->compute.clear();
	m_state->render.clear();
	m_state->compile.clear_libraries();
}

} // namespace metal4
