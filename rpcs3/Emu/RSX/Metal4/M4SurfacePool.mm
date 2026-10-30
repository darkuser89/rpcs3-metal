#include "M4SurfacePool.h"
#include "M4SampleLayout.h"

#include <algorithm>
#include <list>
#include <mutex>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
struct surface_pool_state
{
	std::mutex mutex;
	id<MTLDevice> device;
	surface_pool_statistics stats;
	std::shared_ptr<heap_budget> shared_budget;
	// Most recently returned allocation first; leased storage is never here.
	std::list<std::unique_ptr<surface_allocation>> idle;
	struct reservation
	{
		__weak id<MTLHeap> heap;
		std::size_t bytes;
	};
	// Retired allocations can still have externally retained texture/view
	// handles. Account backing until the actual native heap object disappears.
	std::list<reservation> reservations;
};

namespace
{
std::size_t align_size(std::size_t size, std::size_t alignment)
{
	if (!alignment)
		throw std::runtime_error("Native surface heap reports zero alignment");
	const auto padding = size % alignment ? alignment - size % alignment : 0;
	if (padding > std::numeric_limits<std::size_t>::max() - size)
		throw std::overflow_error("Native surface heap size overflow");
	return size + padding;
}
MTLTextureDescriptor* descriptor(const surface_allocation_description& description)
{
	if (!description.width || !description.height || (description.samples != 1 && description.samples != 2 && description.samples != 4) ||
		(description.samples == 1 && description.expanded))
		throw std::invalid_argument("Native surface allocation dimensions/samples/expansion are invalid");
	const auto format = std::visit([](auto value)
								   {
									   if constexpr (std::is_same_v<decltype(value), surface_color>)
										   return describe_surface_color(value).format;
									   else
										   return describe_surface_depth(value).format;
								   },
								   description.format);
	MTLTextureDescriptor* result = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:description.width height:description.height mipmapped:NO];
	result.storageMode = MTLStorageModePrivate;
	result.hazardTrackingMode = MTLHazardTrackingModeUntracked;
	result.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	result.sampleCount = description.samples;
	if (description.samples > 1)
		result.textureType = MTLTextureType2DMultisample;
	return result;
}
void collect(surface_pool_state& state)
{
	for (auto it = state.reservations.begin(); it != state.reservations.end();)
		if (!it->heap)
		{
			state.stats.reserved_bytes -= it->bytes;
			it = state.reservations.erase(it);
		}
		else
			++it;
}
void evict(surface_pool_state& state)
{
	const auto bytes = state.idle.back()->reserved_bytes();
	state.idle.pop_back();
	state.stats.idle_bytes -= bytes;
	++state.stats.evictions;
	state.stats.idle_allocations = state.idle.size();
	collect(state);
}
} // namespace

surface_budget_error::surface_budget_error(std::size_t request, std::size_t live, std::size_t limit)
	: std::runtime_error("Native surface budget exhausted; release/spill leases and wait before retrying"), requested(request), reserved(live), budget(limit)
{
}
surface_allocation::surface_allocation(surface_allocation_description description, id<MTLHeap> heap, id<MTLTexture> render, id<MTLTexture> expanded)
	: m_description(description), m_heap(heap), m_render(render), m_expanded(expanded), m_bytes(heap.size)
{
}
void surface_allocation::use(context& ctx)
{
	if (ctx.device() != m_heap.device)
		throw std::invalid_argument("Native surface lease belongs to another device");
	ctx.keep_alive(shared_from_this());
	ctx.use_resource(m_render);
	if (m_expanded)
		ctx.use_resource(m_expanded);
}
id<MTLTexture> surface_allocation::render_texture(context& ctx)
{
	use(ctx);
	return m_render;
}
id<MTLTexture> surface_allocation::expanded_texture(context& ctx)
{
	if (!m_expanded)
		throw std::logic_error("Native surface lease has no expanded MSAA storage");
	use(ctx);
	return m_expanded;
}

surface_pool::surface_pool(context& ctx, std::size_t budget,std::shared_ptr<heap_budget> shared_budget) : m_state(std::make_shared<surface_pool_state>())
{
	if (!budget)
		throw std::invalid_argument("Native surface pool budget must be nonzero");
	m_state->device = ctx.device();
	if(shared_budget&&shared_budget->device()!=ctx.device()) throw std::invalid_argument("Surface/shared heap budget devices differ");
	m_state->shared_budget=std::move(shared_budget);
	m_state->stats.budget = budget;
}
surface_lease surface_pool::acquire(const surface_allocation_description& description)
{
	if (!m_state->shared_budget) return acquire_once(description);
	surface_lease result;
	try
	{
		// The outer transaction owns recovery. Nested ledger allocation cannot
		// invoke a cache callback under this pool's mutex. On failure the whole
		// attempt, including its native heap/placements, unwinds before reclaim.
		m_state->shared_budget->with_recovery([&] { result = acquire_once(description); });
	}
	catch (const heap_budget_error& error)
	{
		throw surface_budget_error(error.requested, error.reserved, error.budget);
	}
	return result;
}
surface_lease surface_pool::acquire_once(const surface_allocation_description& description)
{
	const auto render_descriptor = descriptor(description);
	MTLTextureDescriptor* expanded_descriptor = nil;
	if (description.expanded)
	{
		const auto size = sample_expanded_size(description.width, description.height, description.samples);
		expanded_descriptor = [render_descriptor copy];
		expanded_descriptor.textureType = MTLTextureType2D;
		expanded_descriptor.sampleCount = 1;
		expanded_descriptor.width = size[0];
		expanded_descriptor.height = size[1];
	}
	const auto limit=max_2d_texture_dimension(m_state->device);
	if(render_descriptor.width>limit||render_descriptor.height>limit||
		(expanded_descriptor&&(expanded_descriptor.width>limit||expanded_descriptor.height>limit)))
		throw std::invalid_argument("Native surface dimensions exceed the Apple GPU family limit");
	std::unique_lock lock(m_state->mutex);
	collect(*m_state);
	if (![m_state->device supportsTextureSampleCount:description.samples])
		throw std::invalid_argument("Native surface sample count is unsupported");
	std::unique_ptr<surface_allocation> allocation;
	for (auto it = m_state->idle.begin(); it != m_state->idle.end(); ++it)
		if ((*it)->description() == description)
		{
			allocation = std::move(*it);
			m_state->idle.erase(it);
			m_state->stats.idle_bytes -= allocation->reserved_bytes();
			++m_state->stats.allocations_reused;
			break;
		}
	if (!allocation)
	{
		const auto render_size = [m_state->device heapTextureSizeAndAlignWithDescriptor:render_descriptor];
		if (!render_size.size)
			throw std::invalid_argument("Native surface heap size is invalid");
		std::size_t size = render_size.size, expanded_offset = 0;
		if (expanded_descriptor)
		{
			const auto expanded_size = [m_state->device heapTextureSizeAndAlignWithDescriptor:expanded_descriptor];
			if (!expanded_size.size)
				throw std::invalid_argument("Native expanded surface heap size is invalid");
			expanded_offset = align_size(size, expanded_size.align);
			if (expanded_size.size > std::numeric_limits<std::size_t>::max() - expanded_offset)
				throw std::overflow_error("Native paired surface heap size overflow");
			size = expanded_offset + expanded_size.size;
		}
		const auto make_room = [&](std::size_t request)
		{
			if (request > m_state->stats.budget)
				throw surface_budget_error(request, m_state->stats.reserved_bytes, m_state->stats.budget);
			while (request > m_state->stats.budget - m_state->stats.reserved_bytes && !m_state->idle.empty())
				evict(*m_state);
			if (request > m_state->stats.budget - m_state->stats.reserved_bytes)
				throw surface_budget_error(request, m_state->stats.reserved_bytes, m_state->stats.budget);
		};
		make_room(size);
		MTLHeapDescriptor* heap_descriptor = [MTLHeapDescriptor new];
		heap_descriptor.storageMode = MTLStorageModePrivate;
		heap_descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
		heap_descriptor.type = MTLHeapTypePlacement;
		heap_descriptor.size = size;
		id<MTLHeap> heap;
		if(m_state->shared_budget)
		{
			for(;;)
			{
				try { heap=m_state->shared_budget->allocate(heap_descriptor);break; }
				catch(const heap_budget_error&) { if(m_state->idle.empty()) throw;evict(*m_state); }
			}
		}
		else heap = [m_state->device newHeapWithDescriptor:heap_descriptor];
		if (!heap)
			throw native_allocation_error(native_allocation_kind::heap, size);
		// Heap backing is rounded to device page granularity. Account the actual
		// size and refuse publication if rounding exceeds the remaining budget.
		make_room(heap.size);
		const auto render = [heap newTextureWithDescriptor:render_descriptor offset:0];
		id<MTLTexture> expanded = expanded_descriptor ? [heap newTextureWithDescriptor:expanded_descriptor offset:expanded_offset] : nil;
		if (!render || (expanded_descriptor && !expanded))
			throw native_allocation_error(native_allocation_kind::texture, size);
		allocation = std::unique_ptr<surface_allocation>(new surface_allocation(description, heap, render, expanded));
		m_state->reservations.push_back({heap, heap.size});
		m_state->stats.reserved_bytes += allocation->reserved_bytes();
		++m_state->stats.allocations_created;
	}
	m_state->stats.idle_allocations = m_state->idle.size();
	++m_state->stats.leased_allocations;
	const std::weak_ptr<surface_pool_state> owner = m_state;
	// Shared control-block allocation may fail and invoke its deleter. The
	// deleter returns storage under this mutex, so release it before publishing.
	lock.unlock();
	return surface_lease(allocation.release(), [owner](surface_allocation* pointer) noexcept
						 {
							 std::unique_ptr<surface_allocation> allocation(pointer);
							 if (const auto state = owner.lock())
							 {
								 std::lock_guard lock(state->mutex);
								 --state->stats.leased_allocations;
								 const auto bytes = allocation->reserved_bytes();
								 try
								 {
									 state->idle.push_front(std::move(allocation));
									 state->stats.idle_bytes += bytes;
									 state->stats.idle_allocations = state->idle.size();
								 }
								 catch (...)
								 {
									 // A completion-side metadata allocation failure frees the storage;
									 // never throw across a native GPU feedback block.
									 allocation.reset();
									 collect(*state);
								 }
							 }
						 });
}
surface_pool_statistics surface_pool::statistics() const
{
	std::lock_guard lock(m_state->mutex);
	collect(*m_state);
	return m_state->stats;
}
std::size_t surface_pool::trim(std::size_t idle_target_bytes)
{
	std::lock_guard lock(m_state->mutex);
	collect(*m_state);
	const auto before = m_state->stats.reserved_bytes;
	while (m_state->stats.idle_bytes > idle_target_bytes && !m_state->idle.empty())
		evict(*m_state);
	return before - m_state->stats.reserved_bytes;
}
} // namespace metal4
