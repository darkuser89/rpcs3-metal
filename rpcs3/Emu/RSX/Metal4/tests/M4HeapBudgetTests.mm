#include "M4HeapBudget.h"
#include "M4Resources.h"

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

namespace
{
using namespace metal4;
void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}

void bounded_collection(context& ctx)
{
	heap_budget budget(ctx, 256u << 20);
	std::vector<id<MTLBuffer>> buffers;
	constexpr std::size_t count = 1024;
	std::size_t expected = 0;
	for (std::size_t i = 0; i < count; ++i)
	{
		auto buffer = budget.buffer(4096, i % 2 ? MTLStorageModePrivate : MTLStorageModeShared);
		expected += std::max(buffer.length, buffer.allocatedSize);
		buffers.push_back(buffer);
	}
	const auto live = budget.statistics();
	require(live.live_buffers == count && live.live_heaps == 0 && live.reserved_bytes == expected,
		"Live native buffer accounting differs from actual rounded storage");
	// Includes the explicit statistics reconciliation. A full scan on every
	// allocation would exceed this bound by more than an order of magnitude.
	require(live.collection_visits <= count * 9 && live.full_collections == 1,
		"Ordinary allocation performs unbounded live-reservation scans");
	for (std::size_t i = 0; i < count; i += 2)
	{
		expected -= std::max(buffers[i].length, buffers[i].allocatedSize);
		buffers[i] = nil;
	}
	const auto half = budget.statistics();
	require(half.live_buffers == count / 2 && half.reserved_bytes == expected,
		"Distributed buffer releases were not exactly reconciled");
	buffers.clear();
	require(budget.statistics().reserved_bytes == 0, "Empty ledger retained native buffer charges");
	std::cout << "1024 live shared/private buffers: " << live.collection_visits
		<< " retirement visits including explicit statistics; distributed releases exact\n";
}

void pressure(context& ctx)
{
	std::size_t cost;
	{
		heap_budget probe(ctx, 1u << 20);
		auto buffer = probe.shared_buffer(4096);
		cost = std::max(buffer.length, buffer.allocatedSize);
	}
	constexpr std::size_t count = 32;
	heap_budget budget(ctx, cost * count);
	std::vector<id<MTLBuffer>> buffers;
	for (std::size_t i = 0; i < count; ++i) buffers.push_back(budget.shared_buffer(4096));
	id<MTLBuffer> retained = buffers.back();
	buffers.clear();
	// More dead owners than the ordinary collection slice. No statistics call
	// precedes this request: allocation itself must reconcile before refusing.
	id<MTLBuffer> large = budget.shared_buffer(cost * (count - 8));
	const auto stats = budget.statistics();
	const auto actual = cost + std::max(large.length, large.allocatedSize);
	require(stats.live_buffers == 2 && stats.reserved_bytes == actual && stats.full_collections >= 2,
		"Pressure failed to reclaim retired owners outside the bounded slice");
	bool rejected = false;
	try { (void)budget.shared_buffer(cost * count); }
	catch (const heap_budget_error& error)
	{
		rejected = error.requested == cost * count && error.reserved == actual && error.budget == cost * count;
	}
	require(rejected, "Actual native buffer owners escaped strict budget rejection");
	unsigned reclaim_calls = 0;
	const auto reclaimer = budget.register_reclaimer([&](memory_pressure_request) { ++reclaim_calls; });
	require(!budget.try_shared_buffer(cost * count) && !reclaim_calls &&
		budget.statistics().reserved_bytes == actual, "Optional shared allocation ran recovery or altered live charges");
	retained = nil;
	large = nil;
	{
		const auto optional = budget.try_shared_buffer(4096);
		require(optional && budget.statistics().reserved_bytes == cost && !reclaim_calls,
			"Optional shared allocation did not recover capacity after release");
	}
	require(budget.statistics().reserved_bytes == 0, "Pressure recovery leaked a reservation");
	std::cout << "Full pressure reconciliation, rounded budget rejection, optional allocation without reclamation and release passed\n";
}

void heap_views(context& ctx)
{
	heap_budget budget(ctx, 16u << 20);
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	id<MTLTexture> texture = budget.texture(descriptor);
	const auto cost = texture.heap.size;
	id<MTLTexture> view = [texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
	require(view != nil, "Native texture view allocation failed");
	texture = nil;
	const auto held = budget.statistics();
	require(held.live_heaps == 1 && held.live_buffers == 0 && held.reserved_bytes == cost,
		"Escaped texture view lost its native heap budget charge");
	view = nil;
	require(budget.statistics().reserved_bytes == 0, "Last texture view did not retire its heap charge");
	std::cout << "Retired texture backing remains charged through last native view owner\n";
}

void completion_owners()
{
	context ctx(2);
	heap_budget budget(ctx, 1u << 20);
	id<MTLBuffer> buffer = budget.shared_buffer(4096);
	const auto cost = std::max(buffer.length, buffer.allocatedSize);
	const auto event = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } guard{event};
	ctx.begin();
	ctx.wait_event(event, 1);
	fill_buffer(ctx, buffer, 0, 4096, 0x51);
	auto first = ctx.submit();
	ctx.begin();
	fill_buffer(ctx, buffer, 0, 4096, 0x52);
	auto second = ctx.submit();
	buffer = nil;
	require(!first.ready() && !second.ready() && budget.statistics().reserved_bytes == cost,
		"Pending GPU slots lost their shared native buffer reservation");
	event.signaledValue = 1;
	first.wait();
	second.wait();
	require(budget.statistics().reserved_bytes == 0, "Completed native buffer reservation was not retired");
	buffer = budget.shared_buffer(4096);
	ctx.begin();
	ctx.use_resource(buffer);
	buffer = nil;
	require(budget.statistics().reserved_bytes == cost, "Recording buffer owner was collected early");
	ctx.cancel();
	require(budget.statistics().reserved_bytes == 0, "Cancelled recording retained its buffer reservation");
	std::cout << "Two pending command slots and recording cancellation preserve exact buffer ownership\n";
}

void concurrent_retirement(context& ctx)
{
	heap_budget budget(ctx, 64u << 20);
	std::atomic<unsigned> errors = 0;
	std::vector<std::thread> workers;
	for (unsigned worker = 0; worker < 4; ++worker)
		workers.emplace_back([&]
		{
			@autoreleasepool
			{
				try
				{
					for (unsigned batch = 0; batch < 8; ++batch)
					{
						std::vector<id<MTLBuffer>> owners;
						for (unsigned i = 0; i < 32; ++i) owners.push_back(budget.shared_buffer(4096));
					}
				}
				catch (...) { ++errors; }
			}
		});
	for (auto& worker : workers) worker.join();
	const auto stats = budget.statistics();
	require(!errors && stats.created == 1024 && stats.live_buffers == 0 && stats.reserved_bytes == 0,
		"Concurrent native allocation/retirement corrupted budget accounting");
	std::cout << "1024 native allocations with concurrent last-owner retirement passed\n";
}
}

int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx(1);
			bounded_collection(ctx);
			pressure(ctx);
			heap_views(ctx);
			completion_owners();
			concurrent_retirement(ctx);
		}
		catch (const heap_budget_error& error)
		{
			std::cerr << error.what() << " requested=" << error.requested << " reserved=" << error.reserved << " budget=" << error.budget << '\n';
			return 1;
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
