#include "M4HeapBudget.h"
#include "M4Texture.h"
#include "M4ExclusiveGuestMemory.h"

#include <bit>
#include <list>
#include <mutex>
#include <thread>

#if !__has_feature(objc_arc)
#error "Native heap accounting requires ARC ownership."
#endif

namespace metal4
{
struct heap_budget::impl
{
	id<MTLDevice> device;
	mutable std::mutex mutex;
	heap_budget_statistics stats;
	struct reclaimer
	{
		std::thread::id owner;
		std::function<void(memory_pressure_request)> callback;
	};
	std::weak_ptr<reclaimer> reclaim;
	struct reservation
	{
		__weak id object;
		std::size_t bytes;
		bool heap;
	};
	std::list<reservation> reservations;
	std::list<reservation>::iterator next = reservations.end();
	void collect(std::size_t count)
	{
		// A cursor visits each owner fairly without rescanning every live
		// allocation. Weak loads remain authoritative for escaped native views
		// and completion owners; delayed collection only overestimates usage.
		count = std::min(count, reservations.size());
		while (count--)
		{
			if (next == reservations.end()) next = reservations.begin();
			const auto it = next++;
			++stats.collection_visits;
			if (!it->object)
			{
				stats.reserved_bytes -= it->bytes;
				if (it->heap) --stats.live_heaps;
				else --stats.live_buffers;
				reservations.erase(it);
			}
		}
	}
	void reconcile()
	{
		++stats.full_collections;
		collect(reservations.size());
	}
	void check(std::size_t bytes)
	{
		if (bytes > stats.budget - stats.reserved_bytes)
		{
			// Never reject an allocation merely because bounded collection has
			// not reached a retired object. No callbacks or GPU waits under lock.
			reconcile();
			if (bytes > stats.budget - stats.reserved_bytes)
				throw heap_budget_error(bytes, stats.reserved_bytes, stats.budget);
		}
	}
};
heap_budget_error::heap_budget_error(std::size_t request, std::size_t live, std::size_t limit)
	: std::runtime_error("Native heap budget exhausted;release retained versions or wait before retry"), requested(request), reserved(live), budget(limit) {}
native_allocation_error::native_allocation_error(native_allocation_kind kind, std::size_t requested)
	: std::runtime_error("Native Metal resource allocation failed"), kind(kind), requested(requested) {}
heap_budget::heap_budget(context& ctx, std::size_t bytes) : m_impl(std::make_unique<impl>())
{
	if (!bytes)
		throw std::invalid_argument("Native shared heap budget must be nonzero");
	m_impl->device = ctx.device();
	m_impl->stats.budget = bytes;
}
heap_budget::~heap_budget() = default;
id<MTLDevice> heap_budget::device() const noexcept { return m_impl->device; }
heap_budget_statistics heap_budget::statistics() const
{
	std::lock_guard lock(m_impl->mutex);
	m_impl->reconcile();
	return m_impl->stats;
}
std::shared_ptr<void> heap_budget::register_reclaimer(std::function<void(memory_pressure_request)> callback)
{
	if (!callback) throw std::invalid_argument("Native memory reclaimer is absent");
	std::lock_guard lock(m_impl->mutex);
	if (!m_impl->reclaim.expired()) throw std::logic_error("Native memory budget already has an owner reclaimer");
	auto registration = std::make_shared<impl::reclaimer>(std::this_thread::get_id(), std::move(callback));
	m_impl->reclaim = registration;
	return registration;
}
void heap_budget::with_recovery(const std::function<void()>& allocate)
{
	struct scope
	{
		const heap_budget* owner;
		scope* previous;
	};
	static thread_local scope* current = nullptr;
	for (auto* active = current; active; active = active->previous)
		if (active->owner == this) { allocate(); return; }
	scope frame{this, current};
	struct restore { scope*& current; scope* previous; ~restore() { current = previous; } } guard{current, current};
	current = &frame;
	const auto reclaim = [&](memory_pressure_request request)
	{
		if (guest_memory_epoch::active()) return false;
		std::shared_ptr<impl::reclaimer> owner;
		{
			std::lock_guard lock(m_impl->mutex);
			if (request.requested > m_impl->stats.budget) return false;
			owner = m_impl->reclaim.lock();
			if (!owner || owner->owner != std::this_thread::get_id()) return false;
			++m_impl->stats.recovery_attempts;
		}
		// The failing operation has unwound all pool/ledger locks and local
		// native objects. A callback may query this same ledger safely.
		owner->callback(request);
		return true;
	};
	try { allocate(); return; }
	catch (const heap_budget_error& error) { if (!reclaim({error.requested, false})) throw; }
	catch (const native_allocation_error& error) { if (!reclaim({error.requested, true})) throw; }
	allocate(); // Exactly one retry, including nested heap/placement calls.
	std::lock_guard lock(m_impl->mutex);
	++m_impl->stats.recovery_successes;
}
id<MTLHeap> heap_budget::allocate(MTLHeapDescriptor* descriptor)
{
	id<MTLHeap> result;
	with_recovery([&]
	{
		if (!descriptor || !descriptor.size || descriptor.storageMode != MTLStorageModePrivate || descriptor.type != MTLHeapTypePlacement || descriptor.hazardTrackingMode != MTLHazardTrackingModeUntracked)
			throw std::invalid_argument("Shared native budget requires private untracked placement heaps");
		std::lock_guard lock(m_impl->mutex);
		m_impl->collect(8);
		m_impl->check(descriptor.size);
		const auto heap = [m_impl->device newHeapWithDescriptor:descriptor];
		if (!heap)
			throw native_allocation_error(native_allocation_kind::heap, descriptor.size);
		m_impl->check(heap.size);
		m_impl->reservations.push_back({heap, heap.size, true});
		m_impl->stats.reserved_bytes += heap.size;
		++m_impl->stats.created;
		++m_impl->stats.live_heaps;
		result = heap;
	});
	return result;
}
id<MTLTexture> heap_budget::texture(MTLTextureDescriptor* descriptor)
{
	id<MTLTexture> texture;
	with_recovery([&]
	{
		if (!descriptor || !descriptor.width || !descriptor.height || !descriptor.depth || !descriptor.arrayLength || !descriptor.mipmapLevelCount || descriptor.sampleCount != 1 || descriptor.storageMode != MTLStorageModePrivate)
			throw std::invalid_argument("Budgeted texture descriptor is invalid");
		if (descriptor.mipmapLevelCount > static_cast<NSUInteger>(std::bit_width(std::max({descriptor.width, descriptor.height, descriptor.depth}))))
			throw std::invalid_argument("Budgeted texture mip count exceeds dimensions");
		if (descriptor.textureType != MTLTextureType3D && (descriptor.width > max_2d_texture_dimension(m_impl->device) || descriptor.height > max_2d_texture_dimension(m_impl->device)))
			throw std::invalid_argument("Budgeted texture dimensions exceed native GPU family limits");
		MTLTextureDescriptor* copy = [descriptor copy];
		copy.hazardTrackingMode = MTLHazardTrackingModeUntracked;
		const auto size = [m_impl->device heapTextureSizeAndAlignWithDescriptor:copy];
		if (!size.size)
			throw std::invalid_argument("Native texture heap requirement is invalid");
		auto hd = [MTLHeapDescriptor new];
		hd.storageMode = MTLStorageModePrivate;
		hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
		hd.type = MTLHeapTypePlacement;
		hd.size = size.size;
		const auto heap = allocate(hd);
		const auto result = [heap newTextureWithDescriptor:copy offset:0];
		if (!result)
			throw native_allocation_error(native_allocation_kind::texture, size.size);
		texture = result;
	});
	return texture;
}
id<MTLBuffer> heap_budget::shared_buffer(std::size_t bytes)
{
	return buffer(bytes, MTLStorageModeShared);
}
id<MTLBuffer> heap_budget::buffer(std::size_t bytes, MTLStorageMode storage)
{
	id<MTLBuffer> result;
	with_recovery([&]
	{
		if (!bytes || bytes > m_impl->device.maxBufferLength || (storage != MTLStorageModeShared && storage != MTLStorageModePrivate))
			throw std::invalid_argument("Budgeted buffer size or storage is invalid");
		std::lock_guard lock(m_impl->mutex);
		m_impl->collect(8);
		m_impl->check(bytes);
		const auto options = storage == MTLStorageModeShared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
		const auto buffer = [m_impl->device newBufferWithLength:bytes options:options];
		if (!buffer)
			throw native_allocation_error(native_allocation_kind::buffer, bytes);
		const auto size = std::max(buffer.length, buffer.allocatedSize);
		m_impl->check(size);
		m_impl->reservations.push_back({buffer, size, false});
		m_impl->stats.reserved_bytes += size;
		++m_impl->stats.created;
		++m_impl->stats.live_buffers;
		result = buffer;
	});
	return result;
}
} // namespace metal4
