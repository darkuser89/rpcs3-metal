#include "M4SurfacePool.h"
#include "M4SampleTransfer.h"
#include "M4Clear.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void store(std::span<std::byte> data, std::uint32_t word, unsigned bytes)
{
	for (unsigned byte = 0; byte < bytes; ++byte)
		data[byte] = static_cast<std::byte>(word >> ((bytes - byte - 1) * 8));
}
void roundtrips(metal4::context& ctx)
{
	using namespace metal4;
	surface_pool pool(ctx, 4 * 1024 * 1024);
	surface_color_transfer colors(ctx);
	surface_depth_transfer depths(ctx);
	sample_transfer samples(ctx);
	std::size_t cases = 0;
	for (const bool is_depth : {false, true})
		for (const unsigned count : {1u, 2u, 4u})
			for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			{
				if (is_depth && raw > 4)
					continue;
				@autoreleasepool
				{
					surface_allocation_description description;
					description.width = 8;
					description.height = 6;
					description.samples = count;
					description.expanded = count > 1;
					if (is_depth)
						description.format = static_cast<surface_depth>(raw);
					else
						description.format = static_cast<surface_color>(raw);
					auto lease = pool.acquire(description);
					const auto size = count == 1 ? std::array<std::size_t, 2>{8, 6} : sample_expanded_size(8, 6, count);
					const auto bpp = is_depth ? describe_surface_depth(static_cast<surface_depth>(raw)).guest_bytes : color_layout(static_cast<surface_color>(raw)).pixel_bytes;
					std::vector<std::byte> original(size[0] * size[1] * bpp);
					for (std::size_t pixel = 0; pixel < size[0] * size[1]; ++pixel)
					{
						if (is_depth)
						{
							const auto word = raw == 1 ? 0x3000u + pixel : raw == 3 ? 0x6000u + pixel
																	   : raw == 2	? ((0x600000u + pixel) << 8) | (pixel % 256)
																					: ((0x7e0000u + pixel) << 8) | (pixel % 256);
							store(std::span(original).subspan(pixel * bpp), static_cast<std::uint32_t>(word), bpp);
						}
						else if (raw == 11 || raw == 12 || raw == 13)
						{
							const auto element = raw == 11 ? 2u : 4u;
							for (unsigned c = 0; c < bpp / element; ++c)
								store(std::span(original).subspan(pixel * bpp + c * element), static_cast<std::uint32_t>((raw == 11 ? 0x3400u : 0x3e800000u) + pixel + c), element);
						}
						else
							store(std::span(original).subspan(pixel * bpp), static_cast<std::uint32_t>(0x12345678u + pixel), bpp);
					}
					const auto input = make_upload(ctx, original), output = make_buffer(ctx, original.size(), buffer_storage::shared);
					ctx.begin();
					const auto render = lease->render_texture(ctx), expanded = count > 1 ? lease->expanded_texture(ctx) : render;
					require(render.heap && render.heap == expanded.heap, "Pooled surface isn't backed by its native placement heap");
					if (count > 1)
						require(render.heapOffset + render.allocatedSize <= expanded.heapOffset || expanded.heapOffset + expanded.allocatedSize <= render.heapOffset, "Paired MSAA heap allocations overlap");
					texture_region region;
					region.size = MTLSizeMake(size[0], size[1], 1);
					if (is_depth)
						depths.upload(ctx, static_cast<surface_depth>(raw), input, {}, expanded, region);
					else
						colors.upload(ctx, static_cast<surface_color>(raw), input, {}, expanded, region);
					if (count > 1)
					{
						samples.import(ctx, expanded, render);
						samples.expand(ctx, render, expanded);
					}
					if (is_depth)
						depths.download(ctx, static_cast<surface_depth>(raw), expanded, region, output, {});
					else
						colors.download(ctx, static_cast<surface_color>(raw), expanded, region, output, {});
					lease.reset();
					require(pool.statistics().leased_allocations == 1, "Recording surface lease was returned prematurely");
					ctx.submit().wait();
					require(pool.statistics().leased_allocations == 0, "Completed surface lease wasn't returned");
					const auto actual = map_buffer(output, 0, output.length);
					require(std::equal(actual.begin(), actual.end(), original.begin()), "Native pooled surface roundtrip differs");
					++cases;
				}
				(void)pool.trim();
				require(pool.statistics().reserved_bytes == 0, "Trim didn't release completed pooled heap storage");
			}
	std::cout << cases << " native placement-heap surface roundtrips across 18 formats and 1/2/4 samples passed\n";
}

void lifetimes(metal4::context& ctx)
{
	using namespace metal4;
	const surface_allocation_description description{16, 16, 1, surface_color::argb8, false};
	std::size_t cost = 0;
	{
		surface_pool probe(ctx, 1024 * 1024);
		const auto lease = probe.acquire(description);
		cost = lease->reserved_bytes();
	}
	require(cost > 0, "Native pool heap reservation is zero");
	surface_pool pool(ctx, cost * 2);
	auto first = pool.acquire(description), second = pool.acquire(description);
	require(first.get() != second.get(), "Two active surface leases share an allocation");
	bool rejected = false;
	try
	{
		(void)pool.acquire(description);
	}
	catch (const surface_budget_error& e)
	{
		rejected = e.requested > 0 && e.reserved == cost * 2 && e.budget == cost * 2;
	}
	require(rejected, "Native surface budget ignored active caller leases");
	auto* identity = first.get();
	first.reset();
	const auto before = pool.statistics();
	first = pool.acquire(description);
	require(first.get() == identity && pool.statistics().allocations_reused == before.allocations_reused + 1, "Idle storage wasn't reused exclusively");
	ctx.begin();
	(void)first->render_texture(ctx);
	(void)first->render_texture(ctx);
	first.reset();
	require(pool.statistics().leased_allocations == 2, "Duplicate registration lost a live recording lease");
	ctx.cancel();
	require(pool.statistics().leased_allocations == 1, "Cancel didn't release the last unsubmitted surface lease");
	second.reset();
	require(pool.trim() == cost * 2 && pool.statistics().reserved_bytes == 0, "Idle trim didn't reclaim both actual heap reservations");

	// A retained native texture remains budgeted even after its allocation
	// wrapper is evicted; dropping the handle releases the retired backing.
	id<MTLTexture> escaped = nil;
	@autoreleasepool
	{
		auto owner = pool.acquire(description);
		ctx.begin();
		escaped = owner->render_texture(ctx);
		ctx.cancel();
		owner.reset();
	}
	require(escaped && escaped.heap && escaped.pixelFormat == MTLPixelFormatBGRA8Unorm, "Escaped native texture lost its backing heap");
	(void)pool.trim();
	require(pool.statistics().idle_allocations == 0 && pool.statistics().reserved_bytes == cost, "Retired but externally retained heap disappeared from accounting");
	escaped = nil;
	require(pool.statistics().reserved_bytes == 0, "Retired heap wasn't collected after last native handle release");

	clear_engine clear(ctx);
	surface_color_transfer transfer(ctx);
	const auto output = make_buffer(ctx, 16 * 16 * 4, buffer_storage::shared);
	ctx.begin();
	auto lease = pool.acquire(description);
	clear_request request;
	request.colors[0] = {{lease->render_texture(ctx), 0, 0}, surface_color::argb8};
	request.mask = 0xf0;
	request.color_value = 0x12345678;
	request.rectangle = {0, 0, 16, 16};
	(void)clear.run(ctx, request);
	texture_region region;
	region.size = MTLSizeMake(16, 16, 1);
	transfer.download(ctx, surface_color::argb8, lease->render_texture(ctx), region, output, {});
	std::weak_ptr<surface_allocation> weak = lease;
	lease.reset();
	request.colors[0].image.texture = nil;
	require(!weak.expired(), "Recording commands lost the allocation lease");
	ctx.submit().wait();
	require(weak.expired(), "GPU completion kept its allocation control block alive");
	const auto actual = map_buffer(output, 0, output.length);
	for (std::size_t pixel = 0; pixel < 256; ++pixel)
	{
		const std::array<std::byte, 4> word{std::byte{0x12}, std::byte{0x34}, std::byte{0x56}, std::byte{0x78}};
		require(std::equal(word.begin(), word.end(), actual.begin() + static_cast<std::ptrdiff_t>(pixel * 4)), "Clear/download after caller release differs");
	}
	(void)pool.trim();
	std::cout << "Budget pressure, exclusive reuse, duplicate use, cancel, retired native handles and clear/download completion lifetimes passed\n";
	{
		surface_pool lru(ctx, cost * 2);
		auto a = lru.acquire(description);
		auto* original = a.get();
		a.reset();
		auto bd = description;
		bd.format = surface_color::xrgb8_zero;
		auto b = lru.acquire(bd);
		require(b->reserved_bytes() == cost, "Equal native descriptors reserve different heap sizes");
		b.reset();
		a = lru.acquire(description);
		a.reset();
		auto cd = description;
		cd.format = surface_color::xrgb8_one;
		auto c = lru.acquire(cd);
		require(lru.statistics().evictions == 1, "Pool pressure didn't evict the oldest idle allocation");
		a = lru.acquire(description);
		require(a.get() == original, "LRU eviction discarded recently returned storage");
	}
	std::cout << "Native heap LRU pressure preserves the recently used idle allocation\n";
}

void pending()
{
	using namespace metal4;
	context ctx(1);
	const surface_allocation_description description{16, 16, 1, surface_color::argb8, false};
	std::size_t cost = 0;
	{
		surface_pool probe(ctx, 1024 * 1024);
		auto lease = probe.acquire(description);
		cost = lease->reserved_bytes();
	}
	surface_pool pool(ctx, cost);
	const auto event = [ctx.device() newSharedEvent];
	require(event != nil, "Native shared event allocation failed");
	struct unblock
	{
		id<MTLSharedEvent> event;
		~unblock()
		{
			event.signaledValue = 2;
		}
	} guard{event};
	clear_engine clear(ctx);
	surface_color_transfer transfer(ctx);
	const auto output = make_buffer(ctx, 1024, buffer_storage::shared);
	ctx.begin();
	ctx.wait_event(event, 1);
	auto lease = pool.acquire(description);
	clear_request request;
	request.colors[0] = {{lease->render_texture(ctx), 0, 0}, surface_color::argb8};
	request.mask = 0xf0;
	request.color_value = 0x87654321;
	request.rectangle = {0, 0, 16, 16};
	(void)clear.run(ctx, request);
	texture_region region;
	region.size = MTLSizeMake(16, 16, 1);
	transfer.download(ctx, surface_color::argb8, lease->render_texture(ctx), region, output, {});
	request.colors[0].image.texture = nil;
	std::weak_ptr<surface_allocation> weak = lease;
	lease.reset();
	const auto work = ctx.submit();
	require(!work.ready() && !weak.expired(), "Queue event didn't hold submitted surface work");
	bool timeout = false;
	try
	{
		work.wait(std::chrono::milliseconds(1));
	}
	catch (const std::runtime_error& e)
	{
		timeout = std::string_view(e.what()).find("Timed out") != std::string_view::npos;
	}
	require(timeout, "Blocked native GPU wait didn't time out");
	require(pool.trim() == 0 && pool.statistics().leased_allocations == 1 && pool.statistics().reserved_bytes == cost, "Timeout released in-flight surface storage");
	bool pressure = false;
	try
	{
		(void)pool.acquire(description);
	}
	catch (const surface_budget_error&)
	{
		pressure = true;
	}
	require(pressure, "Pending native surface wasn't charged against the budget");
	event.signaledValue = 1;
	work.wait();
	require(weak.expired() && pool.statistics().leased_allocations == 0, "Unblocked GPU completion didn't return surface lease");
	const auto actual = map_buffer(output, 0, 1024);
	for (std::size_t pixel = 0; pixel < 256; ++pixel)
		require(actual[pixel * 4] == std::byte{0x87} && actual[pixel * 4 + 1] == std::byte{0x65} && actual[pixel * 4 + 2] == std::byte{0x43} && actual[pixel * 4 + 3] == std::byte{0x21}, "Deferred native clear/download output differs");
	ctx.begin();
	ctx.wait_event(event, 2);
	ctx.cancel();
	ctx.begin();
	ctx.submit().wait(); // Cancelled wait must never reach the queue.
	require(pool.trim() == cost, "Unblocked heap wasn't reclaimed");
	std::cout << "Actual pending GPU event, CPU timeout, strict in-flight budget and cancelled queue wait passed\n";

	ctx.begin();
	lease = pool.acquire(description);
	(void)lease->render_texture(ctx);
	weak = lease;
	lease.reset();
	const auto status = make_buffer(ctx, 4, buffer_storage::shared);
	*static_cast<std::uint32_t*>(status.contents) = 1;
	ctx.require_gpu_success(status, 0, @"Intentional surface lease validation failure");
	const auto failed = ctx.submit();
	bool reported = false;
	try
	{
		failed.wait();
	}
	catch (const std::runtime_error&)
	{
		reported = true;
	}
	require(reported && failed.ready() && weak.expired() && pool.statistics().leased_allocations == 0, "Failed GPU feedback didn't release surface storage safely");
	(void)pool.trim();
	require(pool.statistics().reserved_bytes == 0, "Failed GPU heap stayed reserved");
	std::cout << "Failed GPU validation returns completed native surface lease without budget leak\n";
}

void detached()
{
	using namespace metal4;
	const surface_allocation_description description{8, 8, 1, surface_color::argb8, false};
	std::unique_ptr<submission> completion;
	id<MTLBuffer> output = nil;
	id<MTLSharedEvent> event = nil;
	struct unblock
	{
		id<MTLSharedEvent> event = nil;
		~unblock()
		{
			event.signaledValue = 1;
		}
	} guard;
	std::weak_ptr<surface_allocation> weak;
	@autoreleasepool
	{
		context ctx(1);
		surface_pool pool(ctx, 1024 * 1024);
		clear_engine clear(ctx);
		surface_color_transfer transfer(ctx);
		event = [ctx.device() newSharedEvent];
		guard.event = event;
		ctx.begin();
		ctx.wait_event(event, 1);
		auto lease = pool.acquire(description);
		weak = lease;
		clear_request request;
		request.colors[0] = {{lease->render_texture(ctx), 0, 0}, surface_color::argb8};
		request.mask = 0xf0;
		request.color_value = 0xabcdef13;
		request.rectangle = {0, 0, 8, 8};
		(void)clear.run(ctx, request);
		output = make_buffer(ctx, 256, buffer_storage::shared);
		texture_region region;
		region.size = MTLSizeMake(8, 8, 1);
		transfer.download(ctx, surface_color::argb8, lease->render_texture(ctx), region, output, {});
		completion = std::make_unique<submission>(ctx.submit());
	}
	require(!completion->ready() && !weak.expired(), "Destroyed context lost waiting native surface work");
	event.signaledValue = 1;
	completion->wait();
	require(weak.expired(), "Destroyed pool/context left a completed allocation lease alive");
	const auto actual = map_buffer(output, 0, 256);
	for (std::size_t pixel = 0; pixel < 64; ++pixel)
		require(actual[pixel * 4] == std::byte{0xab} && actual[pixel * 4 + 1] == std::byte{0xcd} && actual[pixel * 4 + 2] == std::byte{0xef} && actual[pixel * 4 + 3] == std::byte{0x13}, "Detached pool/context GPU output differs");
	std::cout << "Native surface lease and counter-gated submission survive pool/context destruction until GPU completion\n";
}
void overlapping_submissions()
{
	using namespace metal4;
	context ctx(2);
	surface_pool pool(ctx, 1024 * 1024);
	clear_engine clear(ctx);
	surface_color_transfer transfer(ctx);
	const auto event = [ctx.device() newSharedEvent];
	struct unblock
	{
		id<MTLSharedEvent> event;
		~unblock()
		{
			event.signaledValue = 2;
		}
	} guard{event};
	const surface_allocation_description description{8, 8, 1, surface_color::argb8, false};
	auto lease = pool.acquire(description);
	std::weak_ptr<surface_allocation> weak = lease;
	const std::array<id<MTLBuffer>, 2> outputs{make_buffer(ctx, 256, buffer_storage::shared), make_buffer(ctx, 256, buffer_storage::shared)};
	const auto encode = [&](unsigned index)
	{
		ctx.begin();
		ctx.wait_event(event, index + 1);
		clear_request request;
		request.colors[0] = {{lease->render_texture(ctx), 0, 0}, surface_color::argb8};
		request.mask = 0xf0;
		request.color_value = index ? 0x22334455 : 0x66778899;
		request.rectangle = {0, 0, 8, 8};
		(void)clear.run(ctx, request);
		texture_region region;
		region.size = MTLSizeMake(8, 8, 1);
		transfer.download(ctx, surface_color::argb8, lease->render_texture(ctx), region, outputs[index], {});
		return ctx.submit();
	};
	const auto first = encode(0), second = encode(1);
	lease.reset();
	event.signaledValue = 1;
	first.wait();
	if (second.ready() || weak.expired() || pool.statistics().leased_allocations != 1)
	{
		std::cerr << "Two slots: second ready=" << second.ready() << ", lease expired=" << weak.expired() << ", leased=" << pool.statistics().leased_allocations << ", event=" << event.signaledValue
				  << ", second output=" << unsigned(map_buffer(outputs[1], 0, 4)[0]) << '\n';
		if (second.ready())
			try
			{
				second.wait();
			}
			catch (const std::exception& e)
			{
				std::cerr << e.what() << '\n';
			}
		throw std::runtime_error("First submission released a surface still used by another command slot");
	}
	event.signaledValue = 2;
	second.wait();
	require(weak.expired() && pool.statistics().leased_allocations == 0, "Last command slot didn't return its shared surface lease");
	for (unsigned index = 0; index < 2; ++index)
	{
		const auto actual = map_buffer(outputs[index], 0, 256);
		const std::array<std::byte, 4> expected = index ? std::array{std::byte{0x22}, std::byte{0x33}, std::byte{0x44}, std::byte{0x55}} : std::array{std::byte{0x66}, std::byte{0x77}, std::byte{0x88}, std::byte{0x99}};
		for (std::size_t pixel = 0; pixel < 64; ++pixel)
			require(std::equal(expected.begin(), expected.end(), actual.begin() + static_cast<std::ptrdiff_t>(pixel * 4)), "Two queued native surface writes broke command order");
	}
	(void)pool.trim();
	require(pool.statistics().reserved_bytes == 0, "Multiple command slots leaked native heap storage");
	std::cout << "Shared native surface stays leased through the last of two separately blocked command slots\n";
}
} // namespace
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			metal4::context ctx(1);
			roundtrips(ctx);
			lifetimes(ctx);
			ctx.wait_idle();
			detached();
			pending();
			overlapping_submissions();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
