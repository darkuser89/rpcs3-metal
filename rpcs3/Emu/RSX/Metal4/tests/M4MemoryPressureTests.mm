#include "M4MemoryPressure.h"
#include "M4HostAccessFixture.h"
#include "M4Query.h"
#include <atomic>
#include <iostream>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename Error = std::exception, typename F> void refuses(F&& operation, const char* message)
{
	bool failed = false; try { operation(); } catch (const Error&) { failed = true; }
	require(failed, message);
}
void transaction_boundaries()
{
	context device(1); heap_budget budget(device, 1 << 20);
	unsigned calls = 0, attempts = 0;
	auto registration = budget.register_reclaimer([&](memory_pressure_request request)
	{
		++calls; require(request.native_failure && request.requested == 4096, "Native failure category/size changed");
		(void)budget.statistics(); // Must not hold the ledger lock.
	});
	for (const auto kind : {native_allocation_kind::heap, native_allocation_kind::buffer, native_allocation_kind::texture})
	{
		attempts = 0;
		budget.with_recovery([&] { budget.with_recovery([&]
		{
			if (!attempts++) throw native_allocation_error(kind, 4096);
		}); });
		require(attempts == 2, "Nested native failure did not use exactly one retry");
	}
	require(calls == 3 && budget.statistics().recovery_successes == 3, "Native recovery counter differs");
	attempts = 0;
	refuses<native_allocation_error>([&] { budget.with_recovery([&] { budget.with_recovery([&]
		{ ++attempts; throw native_allocation_error(native_allocation_kind::heap, 4096); }); }); }, "Persistent native failure did not terminate");
	require(attempts == 2 && calls == 4, "Persistent nested native failure retried more than once");
	refuses<std::invalid_argument>([&] { (void)budget.buffer(0, MTLStorageModePrivate); }, "Invalid buffer accepted");
	refuses<std::invalid_argument>([&] { (void)budget.allocate(nil); }, "Invalid heap accepted");
	refuses<std::invalid_argument>([&] { (void)budget.texture(nil); }, "Invalid texture accepted");
	refuses<heap_budget_error>([&] { (void)budget.shared_buffer(2 << 20); }, "Oversized request accepted");
	require(calls == 4, "Invalid/oversized request reclaimed resources");
	auto provider = std::make_shared<test::host_access_fixture>();
	provider->execute([&](auto&) { refuses<native_allocation_error>([&] { budget.with_recovery([]
		{ throw native_allocation_error(native_allocation_kind::buffer, 4096); }); }, "VM epoch recovered allocation"); });
	std::atomic<bool> refused = false;
	std::thread worker([&] { try { budget.with_recovery([] { throw native_allocation_error(native_allocation_kind::buffer, 4096); }); }
		catch (const native_allocation_error&) { refused = true; } }); worker.join();
	require(refused && calls == 4, "Foreign-thread/VM recovery called owner caches");
	registration.reset();
	refuses<native_allocation_error>([&] { budget.with_recovery([] { throw native_allocation_error(native_allocation_kind::heap, 4096); }); }, "Expired owner was invoked");
	require(calls == 4, "Expired registration retained callback");
	id<MTLBuffer> published = nil;
	registration = budget.register_reclaimer([&](memory_pressure_request)
	{
		require(published == nil && budget.statistics().reserved_bytes == 0, "Failed native transaction published or retained partial allocation");
	});
	attempts = 0;
	budget.with_recovery([&]
	{
		const auto candidate = budget.shared_buffer(4096);
		if (!attempts++) throw native_allocation_error(native_allocation_kind::texture, 4096);
		published = candidate;
	});
	require(published != nil && budget.statistics().live_buffers == 1, "Native retry did not publish exactly one allocation");
	published = nil; registration.reset();
	registration = budget.register_reclaimer([](memory_pressure_request) { throw std::logic_error("Injected owner failure"); });
	attempts = 0;
	refuses<std::logic_error>([&] { budget.with_recovery([&] { ++attempts; throw native_allocation_error(native_allocation_kind::heap, 4096); }); }, "Owner failure was swallowed");
	require(attempts == 1, "Owner failure retried allocation");
	registration.reset();
	require(budget.shared_buffer(4096) != nil, "Failed recovery left a stale nested-transaction scope");
	std::cout << "Synthetic heap/buffer/placement failures: one outer retry; persistent/invalid/oversized/VM/foreign-thread/lifetime boundaries passed\n";
}
class memory final : public guest_surface_memory
{
	std::vector<std::byte> bytes = std::vector<std::byte>(16 << 20, std::byte{0x33});
	std::uint64_t version = 1;
	void check(guest_surface_range range) const
	{
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Pressure fixture range");
	}
  public:
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		check(coverage.range); return {version, {bytes.begin() + coverage.range.address, bytes.begin() + coverage.range.address + coverage.range.size}};
	}
	std::uint64_t surface_revision(const guest_surface_layout&) override { return version; }
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage, std::uint64_t expected, std::span<const std::byte> input) override
	{
		check(coverage.range); if (expected != version) return {};
		require(input.size() == coverage.range.size, "Pressure fixture payload");
		guest_surface_byte_cursor cursor(coverage);
		while (const auto range = cursor.next()) std::copy_n(input.begin() + range->address - coverage.range.address, range->size, bytes.begin() + range->address);
		return ++version;
	}
	void write(guest_surface_range range, std::span<const std::byte> input) override
	{
		check(range); require(range.size == input.size(), "Pressure fixture store");
		std::copy(input.begin(), input.end(), bytes.begin() + range.address); ++version;
	}
};
fragment_texture_registers words(unsigned side, unsigned offset = 8 << 20)
{
	fragment_texture_registers r{}; r.offset = offset;
	r.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 8 | 1;
	r.address = (7u << 28) | 0x030303; r.control0 = (1u << 31) | (4095u << 7);
	r.control1 = 0xaae4; r.filter = (1u << 24) | (3u << 16);
	r.image_rect = (side << 16) | side; r.control3 = (1u << 20) | (side * 4);
	return r;
}
struct fixture
{
	context device{1};
	static std::shared_ptr<heap_budget> make_budget(context& device)
	{
		auto result = std::make_shared<heap_budget>(device, 32 << 20); device.set_buffer_budget(result); return result;
	}
	std::shared_ptr<heap_budget> budget = make_budget(device);
	std::shared_ptr<memory> ram = std::make_shared<memory>();
	surface_cache surfaces{device, ram, 32 << 20, budget};
	guest_texture_cache textures{device, ram, [](unsigned offset, unsigned, std::size_t) { return offset; }, 8 << 20, budget};
	memory_pressure_owner pressure{device, surfaces, textures, budget};
	cached_surface_ref surface(unsigned side = 1024, unsigned address = 0)
	{
		return surfaces.acquire({address, {side, side, 1, surface_color::argb8, false}, 0, {}});
	}
	id<MTLBuffer> occupy(std::size_t leave)
	{
		(void)device.trim_upload_pages(); (void)surfaces.reclaim_idle();
		const auto stats = budget->statistics();
		require(stats.budget - stats.reserved_bytes > leave, "Pressure fixture already full");
		return budget->shared_buffer(stats.budget - stats.reserved_bytes - leave);
	}
};
void cross_cache()
{
	{
		std::cout << "Cross-cache surface-to-texture\n";
		fixture f; auto source = f.surface(); f.surfaces.prepare(source);
		require(f.surfaces.status(source).initialized && !f.surfaces.status(source).native_write_epoch, "Surface is not reconstructible");
		// Real texture realization owns its output and several simultaneous
		// staging resources. Leave less than that working set, but enough after
		// the 4-MiB clean surface is actually released.
		id<MTLBuffer> blocker = f.occupy(2 << 20);
		const auto before = f.budget->statistics();
		resolved_fragment_texture image;
		try { image = f.textures.resolve(words(512)); }
		catch (...)
		{
			const auto s = f.surfaces.status(source); const auto b = f.budget->statistics(); const auto p = f.surfaces.memory_statistics();
			std::cerr << "Pressure state initialized=" << s.initialized << " native=" << s.native_write_epoch << " pending=" << s.pending
				<< " ledger=" << b.reserved_bytes << " pool=" << p.reserved_bytes << " idle=" << p.idle_bytes << " attempts=" << b.recovery_attempts << '\n'; throw;
		}
		require(image.texture && f.budget->statistics().recovery_successes > before.recovery_successes && !f.surfaces.status(source).initialized,
			"Texture request did not reclaim reconstructible surface backing");
		blocker = nil; f.surfaces.prepare(source);
		require(f.surfaces.status(source).initialized, "Reclaimed surface failed reconstruction with external handle");
	}
	{
		std::cout << "Cross-cache texture-to-surface\n";
		fixture f;
		{ auto image = f.textures.resolve(words(1024)); require(image.texture != nil, "Pressure texture upload failed"); }
		id<MTLBuffer> blocker = f.occupy(65536);
		const auto before = f.budget->statistics();
		auto target = f.surface(512); f.surfaces.prepare(target);
		require(f.surfaces.status(target).initialized && f.textures.statistics().entries == 0 && f.budget->statistics().recovery_successes > before.recovery_successes,
			"Surface request did not reclaim texture LRU outside pool lock");
		blocker = nil;
	}
	std::cout << "Cross-cache pressure: texture allocation reclaims clean surface; surface allocation reclaims texture LRU; handles reconstruct\n";
}
void native_authority()
{
	for (bool flush : {false, true})
	{
		fixture f; auto target = f.surface(256);
		f.surfaces.clear(target, 0xf0, 0xff13579b, 0, 255, {0, 0, 256, 256});
		if (flush) f.surfaces.flush(target);
		const auto before = f.surfaces.status(target);
		id<MTLBuffer> blocker = f.occupy(4096);
		const auto count = f.budget->statistics().recovery_attempts;
		refuses<heap_budget_error>([&] { (void)f.budget->shared_buffer(65536); }, "Native authority was discarded to make room");
		const auto after = f.surfaces.status(target);
		require(f.budget->statistics().recovery_attempts == count + 1 && after.dirty == before.dirty && after.initialized &&
			after.native_write_epoch == before.native_write_epoch && after.content_generation == before.content_generation,
			"Failed pressure modified native authority or repeated recovery");
		blocker = nil;
	}
	std::cout << "Dirty and clean native precision survive true full-ledger refusal with one recovery and unchanged epochs\n";
}
void queries_and_uploads()
{
	for (unsigned mode = 0; mode < 3; ++mode)
	{
		fixture f; { auto image = f.textures.resolve(words(512)); }
		id<MTLBuffer> blocker = f.occupy(4096);
		const auto before = f.budget->statistics();
		if (mode == 0) { query_pool queries(f.device, 8, 65536, f.budget); require(queries.capacity() == 8, "Query recovery failed"); }
		if (mode == 1)
		{
			f.device.begin(); const std::vector<std::byte> constants(65536, std::byte{0x71});
			try { const auto upload = f.device.upload(constants); require(upload.buffer != nil, "Shader upload recovery failed"); f.device.submit().wait(); }
			catch (...) { f.device.cancel(); throw; }
		}
		if (mode == 2) { const auto scratch = make_buffer(f.device, 65536, buffer_storage::private_gpu); require(scratch != nil, "Private scratch recovery failed"); }
		require(f.budget->statistics().recovery_successes > before.recovery_successes && !f.textures.statistics().entries, "Nontexture allocation did not share cache pressure");
		blocker = nil;
	}
	std::cout << "Query arena,recording shader-constant upload and private scratch buffers share owner recovery\n";
}
void completion_and_views()
{
	fixture f;
	auto image = f.textures.resolve(words(512));
	id<MTLTexture> view = [image.texture newTextureViewWithPixelFormat:image.texture.pixelFormat];
	const auto event = [f.device.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } signal{event};
	f.device.begin(); f.device.wait_event(event, 1); f.device.use_resource(image.texture);
	auto work = f.device.submit(); image.texture = nil;
	id<MTLBuffer> blocker = f.occupy(4096);
	const auto before = f.budget->statistics();
	refuses<heap_budget_error>([&] { (void)f.budget->shared_buffer(65536); }, "Pending native texture charge was evicted");
	require(!f.textures.statistics().entries && f.budget->statistics().reserved_bytes == before.reserved_bytes && !work.ready(),
		"Cache eviction released native completion/view charge early or waited for GPU");
	event.signaledValue = 1; work.wait();
	refuses<heap_budget_error>([&] { (void)f.budget->shared_buffer(65536); }, "Escaped texture view lost its backing charge");
	view = nil;
	const auto recovered = f.budget->shared_buffer(65536);
	require(recovered != nil, "Last native owner release did not restore capacity"); blocker = nil;
	std::cout << "Gated GPU completion and escaped native view retain exact charges after LRU eviction; last release recovers capacity\n";
}
void idle_pins_and_deferred()
{
	{
		fixture f; auto target = f.surface(256); f.surfaces.prepare(target);
		const auto before = f.surfaces.status(target);
		id<MTLBuffer> blocker = f.occupy(4096);
		f.surfaces.cpu_write({2 << 20, 1}, [&]
		{
			refuses<std::logic_error>([&] { (void)f.budget->shared_buffer(65536); }, "Surface CPU-capture scope allowed owner reclaim");
		});
		refuses<heap_budget_error>([&] { f.surfaces.spill(target); }, "Spill source was reclaimed while its snapshot allocation was pending");
		const auto after = f.surfaces.status(target);
		require(after.initialized && !after.spilled && after.content_generation == before.content_generation,
			"Failed spill allocation destroyed its own source backing");
		blocker = nil; f.surfaces.spill(target); f.surfaces.prepare(target);
		require(f.surfaces.status(target).initialized, "Spill retry failed after pressure release");
	}
	{
		fixture f; auto target = f.surface(256); f.surfaces.prepare(target);
		id<MTLBuffer> blocker = f.occupy(4096);
		target.reset(); require(f.surfaces.trim_metadata() == 1 && f.surfaces.memory_statistics().idle_bytes, "Fixture did not leave idle native pool storage");
		const auto buffer = f.budget->shared_buffer(65536);
		require(buffer != nil && !f.surfaces.memory_statistics().idle_bytes, "Shared pressure did not reclaim idle surface storage");
		blocker = nil;
	}
	{
		fixture f; auto target = f.surface(256); f.surfaces.prepare(target);
		const auto before = f.surfaces.status(target);
		id<MTLBuffer> blocker = f.occupy(4096);
		const std::array uses{cached_surface_use{target, false}};
		bool encoded = false;
		refuses<heap_budget_error>([&] { f.surfaces.draw(uses, [&](context& commands, auto)
		{
			encoded = true; (void)make_buffer(commands, 65536, buffer_storage::private_gpu);
		}); }, "Pinned draw source was reclaimed during encoding");
		const auto after = f.surfaces.status(target);
		require(encoded && after.initialized && !after.pending && !after.faulted && after.content_generation == before.content_generation,
			"Failed allocation partially published/corrupted the pinned draw");
		blocker = nil;
	}
	{
		fixture f; auto source = f.surface(256); auto target = f.surface(256, 1 << 20);
		f.surfaces.prepare(source); f.surfaces.defer_copy(source, {0, 0, 256, 256}, target, 0, 0);
		id<MTLBuffer> blocker = f.occupy(4096);
		refuses<heap_budget_error>([&] { (void)f.budget->shared_buffer(65536); }, "Frozen deferred source was reclaimed");
		require(f.surfaces.status(source).initialized && f.surfaces.status(target).deferred_copies == 1, "Pressure changed deferred source/plan");
		blocker = nil; f.surfaces.flush(target);
		require(f.ram->read({0, 256 * 256 * 4}).bytes == f.ram->read({1 << 20, 256 * 256 * 4}).bytes, "Deferred copy changed after pressure recovery");
	}
	std::cout << "Idle pool reclaim,pinned encoding failure without publication,and frozen deferred source retention passed\n";
}
void native_snapshot_pressure()
{
	for (bool hold_view : {false, true})
	{
		fixture f;
		const auto source = f.surface(4);
		f.surfaces.clear(source, 0xf0, 0xff204080, 0, 255, {0, 0, 4, 4});
		const auto epoch = f.surfaces.status(source).native_write_epoch;
		id<MTLTexture> escaped = nil;
		@autoreleasepool
		{
			const auto image = f.textures.resolve(words(4, 0), f.surfaces);
			const auto view = f.surfaces.opaque_snapshot_view(image.texture, image.texture.pixelFormat);
			if (hold_view) escaped = view;
		}
		(void)f.device.trim_upload_pages();
		require(f.surfaces.snapshot_statistics().entries == 1, "Pressure fixture did not retain one native snapshot");
		const auto stats = f.budget->statistics();
		id<MTLBuffer> blocker = f.budget->shared_buffer(stats.budget - stats.reserved_bytes);
		if (hold_view)
		{
			refuses<heap_budget_error>([&] { (void)f.budget->shared_buffer(4096); }, "Escaped snapshot view lost its real heap charge under pressure");
			require(escaped && f.surfaces.snapshot_statistics().entries == 0, "Snapshot reclaimer retained cache owners or destroyed the escaped view");
			escaped = nil;
		}
		const auto recovered = f.budget->shared_buffer(4096);
		require(recovered && f.surfaces.snapshot_statistics().entries == 0 && f.budget->statistics().recovery_attempts == stats.recovery_attempts + 1 &&
			f.surfaces.status(source).dirty && f.surfaces.status(source).native_write_epoch == epoch,
			"Snapshot-pressure recovery lost dirty native authority or retried beyond one outer recovery");
		blocker = nil;
	}
	std::cout << "Native snapshot pressure:full real ledger reclaims cached image/opaque views,retains dirty source;escaped views refuse allocation until last release;one outer recovery passed\n";
}
void local_pool_quota()
{
	context device(1);
	auto budget = std::make_shared<heap_budget>(device, 16 << 20); device.set_buffer_budget(budget);
	std::size_t cost;
	{ surface_pool probe(device, 8 << 20, budget); auto allocation = probe.acquire({256, 256, 1, surface_color::argb8, false}); cost = allocation->reserved_bytes(); }
	auto ram = std::make_shared<memory>();
	surface_cache surfaces(device, ram, cost, budget);
	guest_texture_cache textures(device, ram, [](unsigned offset, unsigned, std::size_t) { return offset; }, 8 << 20, budget);
	memory_pressure_owner pressure(device, surfaces, textures, budget);
	auto first = surfaces.acquire({0, {256, 256, 1, surface_color::argb8, false}, 0, {}});
	auto second = surfaces.acquire({1 << 20, {256, 256, 1, surface_color::argb8, false}, 0, {}});
	surfaces.prepare(first); surfaces.prepare(second);
	require(!surfaces.status(first).initialized && !surfaces.status(first).spilled && surfaces.status(second).initialized,
		"Local surface quota copied reconstructible data into a spill instead of reclaiming it");
	surfaces.prepare(first);
	require(surfaces.status(first).initialized && !surfaces.status(second).initialized, "Local quota recovery did not preserve reconstructible handles");
	std::cout << "Spill source pinning and local-pool clean reclaim without duplicate native spill storage passed\n";
}
void fsr_frame_pressure()
{
	context device(1);
	auto budget=std::make_shared<heap_budget>(device,32u<<20);device.set_buffer_budget(budget);
	auto ram=std::make_shared<memory>();surface_cache surfaces(device,ram,32u<<20,budget);
	guest_texture_cache textures(device,ram,[](unsigned offset,unsigned,std::size_t){return offset;},8u<<20,budget);
	native_frame frame(device);memory_pressure_owner pressure(device,surfaces,textures,budget,nullptr,&frame);
	context foreign;native_frame other(foreign);
	refuses<std::invalid_argument>([&]{memory_pressure_owner invalid(device,surfaces,textures,budget,nullptr,&other);},"Foreign FSR frame accepted by pressure owner");
	auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
	const auto source=make_texture(device,d);d.width=17;d.height=13;d.pixelFormat=MTLPixelFormatBGRA8Unorm;const auto target=make_texture(device,d);
	std::array<std::byte,64> bytes;bytes.fill(std::byte{128});const auto input=make_upload(device,bytes);
	device.begin();buffer_to_texture(device,input,0,source,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)});device.submit().wait();
	frame_policy policy{{4,4},1};policy.stretch=true;policy.fsr=true;
	const present_image image{source,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)}};
	@autoreleasepool {device.begin();frame.encode(image,target,policy);device.submit().wait();}
	(void)device.trim_upload_pages();const auto cached=budget->statistics();require(cached.reserved_bytes>0,"FSR pressure test retained no working images");
	id<MTLBuffer> blocker=budget->shared_buffer(cached.budget-cached.reserved_bytes);
	const auto recovered=budget->shared_buffer(4096);
	require(recovered&&budget->statistics().recovery_successes==cached.recovery_successes+1&&!frame.reclaim_fsr(),"Real full-ledger pressure failed to reclaim FSR frame images in one retry");
	blocker=nil;
	@autoreleasepool {device.begin();frame.encode(image,target,policy);device.submit().wait();}
	require(frame.reclaim_fsr(),"FSR frame did not reconstruct its cache after pressure");
	std::cout<<"FSR native_frame:real full-ledger pressure evicts working images with one recovery,then rebuilds;foreign-frame registration refused\n";
}

void overlay_frame_pressure()
{
	context device(1);
	const auto budget = std::make_shared<heap_budget>(device, 16u << 20);
	device.set_buffer_budget(budget);
	const auto ram = std::make_shared<memory>();
	surface_cache surfaces(device, ram, 16u << 20, budget);
	guest_texture_cache textures(device, ram, [](unsigned offset, unsigned, std::size_t) { return offset; }, 8u << 20, budget);
	native_frame frame(device);
	memory_pressure_owner pressure(device, surfaces, textures, budget, nullptr, &frame);
	const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:8 height:8 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	const auto target = make_texture(device, descriptor);
	overlay_frame overlays;
	overlays.images.push_back({512, 512, 1, false, std::vector<std::uint8_t>(512u * 512u * 4, 255)});
	overlay_draw draw;
	draw.virtual_extent = {8, 8};
	draw.image = 0;
	draw.vertices = {{0, 0, 0, 0}, {8, 0, 1, 0}, {0, 8, 0, 1}, {8, 8, 1, 1}};
	overlays.draws.push_back(std::move(draw));
	const frame_policy policy{{8, 8}, 1};
	const auto render = [&]
	{
		@autoreleasepool
		{
			device.begin();
			frame.encode({}, target, policy, overlays);
			device.submit().wait();
		}
	};
	render();
	(void)device.trim_upload_pages();
	const auto cached = budget->statistics();
	require(cached.live_heaps == 1 && cached.reserved_bytes > 0, "No reusable overlay texture remained under the shared ledger");
	@autoreleasepool
	{
		const auto blocker = budget->shared_buffer(cached.budget - cached.reserved_bytes);
		const auto recovered = budget->shared_buffer(4096);
		const auto after = budget->statistics();
		require(blocker && recovered && after.live_heaps == 0 && after.recovery_successes == cached.recovery_successes + 1,
			"Full-ledger recovery failed to reclaim the ready overlay texture");
	}
	render();
	require(budget->statistics().live_heaps == 1, "Overlay cache did not rebuild after recovery");
	std::cout << "Native overlay pressure: shared ledger reclaimed the ready texture in one retry and rebuilt the cache passed\n";
}
}
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try { transaction_boundaries(); cross_cache(); native_authority(); queries_and_uploads(); completion_and_views(); idle_pins_and_deferred(); native_snapshot_pressure(); local_pool_quota(); fsr_frame_pressure(); overlay_frame_pressure(); }
		catch (const heap_budget_error& error) { std::cerr << error.what() << " requested=" << error.requested << " reserved=" << error.reserved << " budget=" << error.budget << '\n'; return 1; }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
