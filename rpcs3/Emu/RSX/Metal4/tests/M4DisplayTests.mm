#include "M4Display.h"
#include "M4GuestCoverage.h"
#include "M4GuardedGuestMemory.h"
#include "M4MemoryPressure.h"
#include "M4FramebufferSnapshot.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void rejects(F&& fn)
{
	try
	{
		fn();
	}
	catch (const std::exception&)
	{
		return;
	}
	throw std::runtime_error("Invalid display resolve accepted");
}
class memory final : public metal4::guest_surface_memory
{
  public:
	std::vector<std::byte> bytes = std::vector<std::byte>(4096, std::byte{0});
	std::uint64_t version = 1;
	unsigned reads = 0, writes = 0;
	bool conflict = false;
	static metal4::guest_surface_range canonical(metal4::guest_surface_range range)
	{
		if (range.address >= 0x2000 && range.address < 0x3000)
			range.address -= 0x1000;
		return range;
	}
	bool test_storage(const storage_test& test) override
	{
		const storage_lookup lookup = [](auto a, auto b)
		{ return metal4::overlaps(canonical(a), canonical(b)); };
		return test(false, lookup);
	}
	bool test_rebase(const rebase_test& test) override
	{
		const storage_rebase rebase = [](auto source, auto request) -> std::optional<metal4::guest_surface_range>
		{
			const auto a = canonical(source), b = canonical(request);
			if (b.address < a.address || b.size > a.size || std::uint64_t(b.address) - a.address > a.size - b.size)
				return {};
			return metal4::guest_surface_range{source.address + b.address - a.address, b.size};
		};
		const storage_equivalent equal = [](auto a, auto b)
		{ a = canonical(a); b = canonical(b); return a.address == b.address && a.size == b.size; };
		return test(rebase, equal);
	}
	metal4::guest_surface_snapshot read_surface(const metal4::guest_surface_layout& coverage) override
	{
		++reads;
		const auto range = canonical(coverage.range);
		const auto offset = range.address - 0x1000;
		require(range.address >= 0x1000 && offset <= bytes.size() && range.size <= bytes.size() - offset, "Wrong display VM range");
		return {version, {bytes.begin() + offset, bytes.begin() + offset + coverage.range.size}};
	}
	std::uint64_t surface_revision(const metal4::guest_surface_layout&) override { return conflict ? ++version : version; }
	std::optional<std::uint64_t> compare_write(const metal4::guest_surface_layout&, std::uint64_t, std::span<const std::byte>) override
	{
		++writes;
		throw std::runtime_error("Display must not quantize/write guest memory");
	}
	void write(metal4::guest_surface_range, std::span<const std::byte>) override
	{
		++writes;
		throw std::runtime_error("Display must not write guest memory");
	}
};
std::vector<std::byte> render(metal4::context& ctx, metal4::presenter& presenter, const metal4::present_image& image)
{
	using namespace metal4;
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:2 height:2 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	const auto output = make_texture(ctx, descriptor);
	const auto buffer = make_buffer(ctx, 16, buffer_storage::shared);
	ctx.begin();
	presenter.encode(ctx, image, output);
	texture_to_buffer(ctx, output, image.region, buffer, 0);
	ctx.submit().wait();
	const auto span = map_buffer(buffer, 0, 16);
	return {span.begin(), span.end()};
}
void verify(std::span<const std::byte> pixels, std::array<unsigned, 4> bgra)
{
	for (std::size_t i = 0; i < pixels.size(); ++i)
	{
		if (pixels[i] != std::byte(bgra[i % 4]))
			throw std::runtime_error("Display BGRA pixel differs at byte " + std::to_string(i) + ":actual=" +
				std::to_string(std::to_integer<unsigned>(pixels[i])) + ",expected=" + std::to_string(bgra[i % 4]));
	}
}
void fill(memory& ram, unsigned offset, unsigned red = 32)
{
	for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x)
		for (unsigned b = 0; b < 4; ++b) ram.bytes[offset + y * 24 + x * 4 + b] = std::byte(std::array{0u,red,64u,128u}[b]);
	++ram.version;
}
void job_and_cache_cases()
{
	using namespace metal4;
	using namespace std::chrono_literals;
	context ctx(2);
	auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
	auto ram = std::make_shared<memory>(); fill(*ram,0x100);
	const auto translate = [](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; };
	surface_cache surfaces(ctx,ram,8u << 20,budget);
	display_resolver resolver(ctx,ram,translate,budget,4096);
	presenter present(ctx);
	const display_snapshot descriptor{0x100,2,2,24,display_format::xrgb8};
	ctx.begin(); const auto caller = ctx.recording_submission(); const auto before_reads = ram->reads;
	rejects([&] { (void)resolver.prepare(descriptor,surfaces); });
	require(caller.same_batch(ctx.recording_submission()) && !caller.ready() && ram->reads == before_reads,
		"Display preflight cancelled or read through a caller recording"); ctx.cancel();
	const auto gate = [ctx.device() newSharedEvent]; const std::array waits{display_upload_dependency{gate,1}};
	auto first = resolver.prepare(descriptor,surfaces,waits);
	require(!first.ready() && resolver.statistics().uploads == 1 && resolver.statistics().entries == 0 && ram->reads == 1,
		"Display prepare waited for completion or published a pending image");
	const auto reserved = budget->statistics().reserved_bytes;
	rejects([&] { (void)first.finish(1ms); });
	require(!first.ready() && budget->statistics().reserved_bytes == reserved, "Display timeout freed pending native owners");
	auto duplicate = resolver.prepare(descriptor,surfaces);
	auto swapped = descriptor; swapped.format = display_format::xbgr8;
	auto reinterpret = resolver.prepare(swapped,surfaces);
	require(!duplicate.ready() && !reinterpret.ready() && resolver.statistics().uploads == 1 && resolver.statistics().pending_hits == 2 && ram->reads == 1,
		"Pending AV aliases were recaptured or uploaded separately");
	unsigned independent_cpu_work = 0; ++independent_cpu_work;
	require(independent_cpu_work == 1 && !first.ready(), "Display upload prevented independent CPU work");
	gate.signaledValue = 1;
	const auto frozen = first.finish(); const auto same = duplicate.finish(); const auto reverse = reinterpret.finish();
	require(frozen.texture == same.texture && frozen.texture != reverse.texture && resolver.statistics().entries == 1 && resolver.statistics().published == 1,
		"Display completion duplicated its cached storage or AV views");
	verify(render(ctx,present,frozen),{128,64,32,255}); verify(render(ctx,present,reverse),{32,64,128,255});
	const auto creates = budget->statistics().created; const auto encoders = ctx.encoding_statistics();
	for (unsigned i = 0; i < 100; ++i)
		require(resolver.prepare(descriptor,surfaces).finish().texture == frozen.texture, "Unchanged CPU display replaced its opaque view");
	require(resolver.statistics().uploads == 1 && resolver.statistics().hits == 100 && ram->reads == 1 && budget->statistics().created == creates &&
		ctx.encoding_statistics().compute_encoders == encoders.compute_encoders, "Confirmed CPU display hits reread,allocated or encoded");
	fill(*ram,0x100,17);
	rejects([&] { (void)first.finish(); });
	const auto fresh = resolver.prepare(descriptor,surfaces).finish();
	require(fresh.texture != frozen.texture && resolver.statistics().uploads == 2, "Active CPU display write reused stale storage");
	verify(render(ctx,present,fresh),{128,64,17,255}); verify(render(ctx,present,frozen),{128,64,32,255});
	resolver.clear();
	auto pending = resolver.prepare(descriptor,surfaces);
	resolver.clear(); (void)pending.finish();
	require(resolver.statistics().entries == 0, "Display clear permitted late cache republication");
	auto moved = std::move(pending); rejects([&] { (void)pending.ready(); });
	require(moved.ready(), "Moved display job lost its native completion");
	std::jthread foreign([&] { rejects([&] { (void)moved.finish(); }); }); foreign.join();
	display_resolver uncached(ctx,ram,translate,budget,4096,{}, {},{3,1});
	(void)uncached.resolve(descriptor,surfaces); (void)uncached.resolve(descriptor,surfaces);
	require(uncached.statistics().entries == 0 && uncached.statistics().uploads == 2, "Display byte-limit bypass retained oversized backing");
	display_resolver lru(ctx,ram,translate,budget,4096,{}, {},{1,1u << 20});
	(void)lru.resolve(descriptor,surfaces); auto other = descriptor; other.offset += 0x100;
	(void)lru.resolve(other,surfaces); (void)lru.resolve(descriptor,surfaces);
	require(lru.statistics().entries == 1 && lru.statistics().uploads == 3 && lru.statistics().evictions == 2, "Display cache entry limit did not evict by LRU");
	const auto native = surfaces.acquire({0x1100,{2,2,1,surface_color::argb8,false},24,{}});
	surfaces.clear(native,0xf0,0x00556677,0,255,{0,0,2,2});
	const auto native_reads = ram->reads; const auto native_uploads = resolver.statistics().uploads;
	verify(render(ctx,present,resolver.prepare(descriptor,surfaces).finish()),{119,102,85,255});
	require(ram->reads == native_reads && resolver.statistics().uploads == native_uploads && surfaces.status(native).dirty,
		"Cached CPU backbuffer masked a newer native GPU writer or caused VM writeback");
	std::cout << "Display native jobs:unsignalled prepare returns,timeout retains owners,pending AV aliases share one upload/two views;100 confirmed CPU hits without read/allocation/encoding;active writes preserve exact old/new pixels,clear epoch/moved/foreign guards andentry/actual-heap-byte limits passed\n";
	std::cout << "Display native GPU writer takes precedence over an existing VM cache image with unchanged CPU tags,no VM pixel read/writeback passed\n";
}
void native_job_cases()
{
	using namespace metal4;
	using namespace std::chrono_literals;
	for (const auto format : {display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
	for (unsigned odd : {0u,1u})
	{
		context ctx(3); auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		auto ram = std::make_shared<memory>();
		surface_cache surfaces(ctx,ram,8u << 20,budget);
		const auto translate = [](unsigned offset,unsigned,std::size_t) { return offset >= 0x800 ? 0x2000u + offset - 0x800 : 0x1000u + offset; };
		display_resolver resolver(ctx,ram,translate,budget,4096); presenter present(ctx);
		const bool fp16 = format == display_format::rgbx16f;
		const unsigned pitch = (fp16 ? 40 : 24) + odd, bytes = fp16 ? 8 : 4;
		const auto source = surfaces.acquire({0x1100,{4,4,1,fp16 ? surface_color::rgba16f : surface_color::argb8,false},pitch,{}});
		if (!fp16) surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,4,4});
		else
		{
			surfaces.prepare(source,surface_initialization::background);
			const std::array uses{cached_surface_use{source,true}};
			surfaces.draw(uses,[](context& commands,auto images)
			{
				std::array<std::uint16_t,64> pixels;
				for (unsigned b = 0; b < pixels.size(); ++b) pixels[b] = std::array<std::uint16_t,4>{0x3c00,0x3800,0x3400,0}[b % 4];
				const auto input = make_upload(commands,std::as_bytes(std::span(pixels)));
				buffer_to_texture(commands,input,0,images[0],{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)},32);
			});
		}
		const display_snapshot descriptor{0x100 + pitch + bytes,2,2,pitch,format};
		const auto gate = [ctx.device() newSharedEvent], ignored_gate = [ctx.device() newSharedEvent];
		struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
		const std::array waits{display_upload_dependency{gate,1}}, ignored_waits{display_upload_dependency{ignored_gate,77}};
		const auto encoders = ctx.encoding_statistics(); const auto submits = surfaces.transaction_statistics().submitted;
		auto first = resolver.prepare(descriptor,surfaces,waits);
		require(!first.ready() && resolver.statistics().native_submissions == 1 && resolver.statistics().uploads == 0 && ram->reads == 0 && ram->writes == 0 &&
			surfaces.snapshot_statistics().created == 0 && surfaces.snapshot_statistics().entries == 0 && surfaces.transaction_statistics().submitted == submits &&
			ctx.encoding_statistics().compute_encoders == encoders.compute_encoders + 1,
			"Cold native display copied synchronously,read VM or published before its shared event");
		const auto reserved = budget->statistics().reserved_bytes; rejects([&] { (void)first.finish(1ms); });
		require(!first.ready() && budget->statistics().reserved_bytes == reserved, "Native display CPU timeout released pending backing");
		bool metadata_guard_refused = false;
		rejects([&]
		{
			(void)surfaces.snapshot_texture_region(display_coverage(descriptor,translate(descriptor.offset,0,1)),[&](const auto&) -> std::optional<clear_rectangle>
			{
				try { (void)first.finish(5s); } catch (const std::logic_error&) { metadata_guard_refused = true; }
				return {};
			},budget);
		});
		require(metadata_guard_refused && !first.ready(), "Native display waited inside a legacy SurfaceCache CPU metadata callback");
		auto duplicate = resolver.prepare(descriptor,surfaces,ignored_waits);
		auto alias_descriptor = descriptor; alias_descriptor.offset += 0x800;
		auto alias = resolver.prepare(alias_descriptor,surfaces);
		auto alternate_descriptor = descriptor;
		if (!fp16) alternate_descriptor.format = format == display_format::xrgb8 ? display_format::xbgr8 : display_format::xrgb8;
		auto alternate = resolver.prepare(alternate_descriptor,surfaces);
		require(!duplicate.ready() && !alias.ready() && !alternate.ready() && resolver.statistics().native_submissions == 1 &&
			ctx.encoding_statistics().compute_encoders == encoders.compute_encoders + 1 && surfaces.snapshot_statistics().entries == 0,
			"Pending native AV/physical aliases encoded separate copies or completed before their original dependency");
		unsigned cpu_geometry_work = 0; ++cpu_geometry_work;
		require(cpu_geometry_work == 1 && !first.ready(), "Cold native display prevented independent CPU geometry work");
		gate.signaledValue = 1; ctx.wait_idle();
		surfaces.spill(source);
		const auto spill_submits = surfaces.transaction_statistics().submitted, spill_allocations = budget->statistics().created;
		const auto spill_encoders = ctx.encoding_statistics();
		auto from_spill = resolver.prepare(descriptor,surfaces);
		require(from_spill.ready() && surfaces.status(source).spilled && surfaces.transaction_statistics().submitted == spill_submits &&
			budget->statistics().created == spill_allocations && ctx.encoding_statistics().compute_encoders == spill_encoders.compute_encoders &&
			ctx.encoding_statistics().render_encoders == spill_encoders.render_encoders,
			"Pending native display lookup restored an already captured raw-spill source");
		const auto frozen = first.finish(), same = duplicate.finish(), physical = alias.finish(), reinterpreted = alternate.finish();
		require(from_spill.finish().texture == frozen.texture && surfaces.status(source).spilled, "Pending spill display alias lost its captured immutable image");
		require(frozen.texture == same.texture && frozen.texture == physical.texture && (fp16 || frozen.texture != reinterpreted.texture) &&
			frozen.texture.heap == reinterpreted.texture.heap && surfaces.snapshot_statistics().created == 1 && surfaces.snapshot_statistics().entries == 1 &&
			resolver.statistics().native_submissions == 1 && resolver.statistics().entries == 0 && ignored_gate.signaledValue == 0,
			"Native display lost shared views/work,adopted a later dependency or duplicated publication");
		const auto expected = fp16 ? std::array{64u,128u,255u,255u} : format == display_format::xrgb8 ? std::array{51u,34u,17u,255u} : std::array{17u,34u,51u,255u};
		verify(render(ctx,present,frozen),expected);
		verify(render(ctx,present,reinterpreted),fp16 ? expected : std::array{expected[2],expected[1],expected[0],255u});
		const auto allocations = budget->statistics().created, views = surfaces.snapshot_statistics().views_reused;
		const auto ready_encoders = ctx.encoding_statistics();
		for (unsigned i = 0; i < 100; ++i) require(resolver.prepare(descriptor,surfaces).finish().texture == frozen.texture, "Native ready display replaced its opaque view");
		require(resolver.statistics().native_submissions == 1 && ram->reads == 0 && ram->writes == 0 && budget->statistics().created == allocations &&
			ctx.encoding_statistics().compute_encoders == ready_encoders.compute_encoders && surfaces.snapshot_statistics().views_reused == views + 100 && surfaces.status(source).spilled && surfaces.transaction_statistics().submitted == spill_submits,
			"Ready native display allocated,encoded,submitted or touched VM pixels");
	}
	std::cout << "Native display jobs:6 AV-format/even-odd-pitch crop profiles return pending behind one real gate;CPU work/timeout keep owners,duplicate andphysical aliases share one copy/opaque views,late dependencies ignored;pending andready source reuse keeps a real raw spill unrestored;exact AV pixels and100 ready hits per profile without allocation/encoding/VM reads passed\n";
}
void native_generation_and_bound_cases()
{
	using namespace metal4;
	for (unsigned mode = 0; mode < 2; ++mode)
	{
		context ctx(3); auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,budget);
		display_resolver resolver(ctx,ram,[](unsigned o,unsigned,std::size_t) { return 0x1000u + o; },budget,4096); presenter present(ctx);
		const auto source = surfaces.acquire({0x1100,{2,2,1,surface_color::argb8,false},24,{}});
		surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,2,2});
		const auto gate = [ctx.device() newSharedEvent];
		struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
		auto original = resolver.prepare({0x100,2,2,24},surfaces,std::array{display_upload_dependency{gate,1}});
		std::optional<display_upload_job> after_clear;
		if (!mode)
		{
			surfaces.clear_snapshots();
			after_clear.emplace(resolver.prepare({0x100,2,2,24},surfaces));
			require(!after_clear->ready() && resolver.statistics().native_submissions == 2, "Native snapshot clear reused an older pending generation");
		}
		gate.signaledValue = 1;
		if (mode) surfaces.clear(source,0xf0,0x00445566,0,255,{0,0,2,2});
		const auto frozen = original.finish();
		require(surfaces.snapshot_statistics().entries == 0, "Old native display repopulated a cleared/new-writer source key");
		const auto latest = after_clear ? after_clear->finish() : resolver.prepare({0x100,2,2,24},surfaces).finish();
		require(frozen.texture != latest.texture && surfaces.snapshot_statistics().entries == 1 && resolver.statistics().native_submissions == 2,
			"Native display generation/new-writer retry did not produce one fresh ready source");
		verify(render(ctx,present,frozen),{51,34,17,255});
		verify(render(ctx,present,latest),mode ? std::array{102u,85u,68u,255u} : std::array{51u,34u,17u,255u});
	}
	{
		context ctx(2); auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,budget);
		display_resolver resolver(ctx,ram,[](unsigned o,unsigned,std::size_t) { return 0x1000u + o; },budget,4096);
		const auto source = surfaces.acquire({0x1100,{130,1,1,surface_color::argb8,false},520,{}});
		surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,130,1});
		std::vector<display_upload_job> jobs; jobs.reserve(64);
		for (unsigned i = 0; i < 64; ++i) jobs.push_back(resolver.prepare({0x100 + i * 8,2,1,520},surfaces));
		const auto allocations = budget->statistics().created;
		rejects([&] { (void)resolver.prepare({0x100 + 64 * 8,2,1,520},surfaces); });
		require(resolver.statistics().native_submissions == 64 && budget->statistics().created == allocations && ram->reads == 0 && surfaces.snapshot_statistics().entries == 0,
			"Sixty-fifth unconsumed native display escaped admission before its heap allocation");
		jobs.clear(); ctx.wait_idle(); (void)resolver.prepare({0x100 + 64 * 8,2,1,520},surfaces).finish();
		require(resolver.statistics().native_submissions == 65 && surfaces.snapshot_statistics().entries == 1, "Retired native display owners did not restore admission");
	}
	std::cout << "Native display generations:clear forces a fresh pending copy,newer GPU writers keep old captured pixels immutable,no obsolete snapshot key republishes;64 unconsumed native versions bounded before allocation andowner retirement restores admission passed\n";
}
class mapped_provider final : public metal4::exclusive_guest_memory, public metal4::exclusive_guest_memory::view, public std::enable_shared_from_this<mapped_provider>
{
	std::mutex mutex;
  public:
	std::vector<std::byte> data = std::vector<std::byte>(2u << 16);
	unsigned bank = 0;
	bool mapped = true;
	std::optional<std::vector<metal4::storage_identity_span>> storage_identity(metal4::guest_surface_range range) override
	{
		if(!mapped||range.address>65536||range.size>65536-range.address)throw std::out_of_range("Display identity mapping absent");
		return std::vector<metal4::storage_identity_span>{{shared_from_this(),bank*65536+range.address,range}};
	}
	std::byte* map(metal4::guest_surface_range range,bool) override
	{
		if (!mapped || range.address > 65536 || range.size > 65536 - range.address) throw std::out_of_range("Display fixture mapping absent");
		return data.data() + bank * 65536 + range.address;
	}
	bool same_storage(metal4::guest_surface_range a,metal4::guest_surface_range b) override { return metal4::overlaps(a,b); }
	void written(metal4::guest_surface_range) noexcept override {}
	std::shared_ptr<void> retain_backing(std::span<const metal4::guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& work) override
	{
		if (auto* current = metal4::guest_memory_epoch::current(*this)) { work(*current); return; }
		std::lock_guard lock(mutex); metal4::guest_memory_epoch epoch(*this,*this); work(*this);
	}
};
void exact_revision_cases()
{
	using namespace metal4;
	context ctx(2); auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
	auto provider = std::make_shared<mapped_provider>(); auto ram = std::make_shared<guarded_guest_memory>(provider);
	auto tiles = std::make_shared<tile_state>();
	const cpu_capture_dispatch capture = tile_capture_dispatch([provider](const auto& work) { provider->execute([&](auto&) { work(); }); },tiles);
	surface_cache surfaces(ctx,ram,8u << 20,budget,capture);
	display_resolver resolver(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset; },budget,4096,capture,tiles);
	const display_snapshot image{0x100,2,2,24};
	const auto original = resolver.prepare(image,surfaces).finish();
	const auto uploads = resolver.statistics().uploads;
	ram->write({0x110,1},std::array{std::byte{0x77}}); // First row padding only.
	require(resolver.prepare(image,surfaces).finish().texture == original.texture && resolver.statistics().uploads == uploads,
		"Padding-only CPU write invalidated active display bytes");
	tiles->update(1,{0,65536,65536,256,0,0,0,true}); tiles->update(1,{});
	require(resolver.prepare(image,surfaces).finish().texture == original.texture && resolver.statistics().uploads == uploads,
		"Unrelated tile changes invalidated display mapping");
	ram->write({0x100,1},std::array{std::byte{0x55}});
	require(resolver.prepare(image,surfaces).finish().texture != original.texture, "Active CPU byte did not replace display version");
	auto pending = resolver.prepare(image,surfaces);
	provider->execute([&](auto&) { provider->bank = 1; ram->notify_mapping_changed({0,65536}); provider->bank = 0; ram->notify_mapping_changed({0,65536}); });
	rejects([&] { (void)pending.finish(); });
	auto latest = resolver.prepare(image,surfaces); (void)latest.finish();
	tile_descriptor tile{0,0,65536,256,0,0,0,true};
	tiles->update(0,tile); tiles->update(0,{});
	rejects([&] { (void)latest.finish(); });
	auto current = resolver.prepare(image,surfaces); (void)current.finish();
	provider->execute([&](auto&)
	{
		require(current.current(), "CPU-only final display receipt rejected valid nested VM view");
		rejects([&] { (void)current.finish(); });
		rejects([&] { (void)resolver.prepare(image,surfaces); });
	});
	provider->execute([&](auto&) { provider->mapped = false; ram->notify_mapping_changed({0,65536}); });
	rejects([&] { (void)current.finish(); }); rejects([&] { (void)resolver.prepare(image,surfaces); });
	std::cout << "Display exact receipts:real guarded active-byte revision preserves padding hits,replaces active pixels;identical-EA VM mapping ABA,tile bind/unbind ABA andunmap refuse stale jobs;final CPU epoch validates without GPU wait andprepare/finish reject VM exclusion passed\n";
}
void native_exact_revision_cases()
{
	using namespace metal4;
	for (unsigned mode = 0; mode < 6; ++mode)
	{
		context ctx(2); auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			auto provider = std::make_shared<mapped_provider>(); auto ram = std::make_shared<guarded_guest_memory>(provider);
			auto tiles = std::make_shared<tile_state>();
			const cpu_capture_dispatch capture = tile_capture_dispatch([provider](const auto& work) { provider->execute([&](auto&) { work(); }); },tiles);
			surface_cache surfaces(ctx,ram,8u << 20,budget,capture);
			display_resolver resolver(ctx,ram,[](unsigned o,unsigned,std::size_t) { return o; },budget,4096,capture,tiles);
			const auto source = surfaces.acquire({0x100,{2,2,1,surface_color::argb8,false},24,{}});
			surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,2,2});
			const auto gate = [ctx.device() newSharedEvent];
			struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
			auto job = resolver.prepare({0x100,2,2,24},surfaces,std::array{display_upload_dependency{gate,1}});
			require(!job.ready(), "Guarded native display fixture was not pending before producer mutation");
			if (mode == 0) provider->execute([&](auto&) { provider->bank = 1; ram->notify_mapping_changed({0,65536}); provider->bank = 0; ram->notify_mapping_changed({0,65536}); });
			if (mode == 1) { tiles->update(0,{0,0,65536,256,0,0,0,true}); tiles->update(0,{}); }
			if (mode == 2) provider->execute([&](auto&) { provider->mapped = false; ram->notify_mapping_changed({0,65536}); });
			if (mode == 3) ram->write({0x110,1},std::array{std::byte{0x77}});
			if (mode == 4) ram->write({0x100,1},std::array{std::byte{0x77}});
			if (mode == 5) { tiles->update(1,{0,65536,65536,256,0,0,0,true}); tiles->update(1,{}); }
			const bool accepted = mode == 3 || mode == 5;
			if (accepted) provider->execute([&](auto&)
			{
				require(job.current(), "Padding/unrelated tile producer mutation invalidated active native display bytes");
				rejects([&] { (void)job.finish(); });
				rejects([&] { (void)resolver.prepare({0x100,2,2,24},surfaces); });
			});
			gate.signaledValue = 1;
			if (!accepted) rejects([&] { (void)job.finish(); });
			else
			{
				const auto image = job.finish(); presenter present(ctx); verify(render(ctx,present,image),{51,34,17,255});
			}
			require(surfaces.snapshot_statistics().entries == unsigned(accepted) && surfaces.snapshot_statistics().created == unsigned(accepted) &&
				resolver.statistics().uploads == 0 && !guest_memory_epoch::active(),
				"Guarded native display published stale mappings or leaked VM exclusion into completion");
			resolver.clear(); surfaces.clear_snapshots(); (void)ctx.trim_upload_pages();
		}
		require(budget->statistics().reserved_bytes == 0, "Guarded native display mapping cases retained orphan native backing");
	}
	std::cout << "Native display exact receipts:real guarded identical-EA mapping ABA,tile bind/unbind ABA,unmap andactive writes refuse pending images/cache entries;row-padding writes andunrelated tile ABA retain exact native pixels;current is CPU-only,prepare/finish refuse held VM exclusion,all charges retire passed\n";
}
void mixed_guarded_revision_cases()
{
	using namespace metal4;
	for(unsigned mode=0;mode<10;++mode)
	{
		context ctx(3);auto budget=std::make_shared<heap_budget>(ctx,8u<<20);ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			auto provider=std::make_shared<mapped_provider>();auto ram=std::make_shared<guarded_guest_memory>(provider);auto tiles=std::make_shared<tile_state>();
			const cpu_capture_dispatch capture=tile_capture_dispatch([provider](const auto& work){provider->execute([&](auto&){work();});},tiles);
			surface_cache surfaces(ctx,ram,8u<<20,budget,capture);display_resolver resolver(ctx,ram,[](unsigned o,unsigned,std::size_t){return o;},budget,4096,capture,tiles);
			const swizzle_region morton{{4,4,1},{},{3,3,1}};
			framebuffer_surface framebuffer;framebuffer.address=0x100;framebuffer.guest=plan_guest_surface(0x100,3,3,4,0,morton);
			checked_cpu_capture(capture,[&]{map_framebuffer_surface(framebuffer,0x100,0,[](unsigned o,unsigned,std::size_t){return o;},{},tiles,true,4096);});
			cached_surface_description description{0x100,{3,3,1,surface_color::argb8,false},0,morton,{},framebuffer.mapping->state().segments,framebuffer.mapping};
			ram->write({0x134,4},std::array{std::byte{0},std::byte{128},std::byte{96},std::byte{64}});
			const auto source=surfaces.acquire(description);surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,3,3});
			const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};
			const display_snapshot descriptor{0x130,2,1,9};auto job=resolver.prepare(descriptor,surfaces,std::array{display_upload_dependency{gate,1}});
			require(!job.ready()&&resolver.statistics().captured_vm_bytes==4&&resolver.statistics().captured_vm_ranges==1&&resolver.statistics().mixed_casts==1,"Guarded mixed display did not own exactly the missing word");
			std::optional<present_image> original;if(mode==9){gate.signaledValue=1;original=job.finish();}
			if(mode==1)ram->write({0x134,1},std::array{std::byte{0}}); // Same-value VM hole write.
			if(mode==2)ram->write({0x138,1},std::array{std::byte{77}}); // AV padding,also inactive native Morton cell.
			if(mode==3){tiles->update(0,{0,0,65536,1024,8,0,3,true});tiles->update(0,{});}
			if(mode==4)provider->execute([&](auto&){provider->bank=1;ram->notify_mapping_changed({0,65536});provider->bank=0;ram->notify_mapping_changed({0,65536});});
			if(mode==5)ram->write({0x130,1},std::array{std::byte{77}});
			if(mode==6)provider->execute([&](auto&){provider->mapped=false;ram->notify_mapping_changed({0,65536});});
			if(mode==7){tiles->update(1,{0,65536,65536,256,0,0,0,true});tiles->update(1,{});}
			if(mode==8||mode==9)ram->write({0x135,1},std::array{std::byte{32}});
			const bool accepted=mode==0||mode==2||mode==7;
			if(accepted)provider->execute([&](auto&){require(job.current(),"Guarded mixed receipt lost active-byte precision");rejects([&]{(void)job.finish();});rejects([&]{(void)resolver.prepare(descriptor,surfaces);});});
			gate.signaledValue=1;std::optional<present_image> image;
			if(accepted)image=job.finish();else rejects([&]{(void)job.finish();});
			if(mode==1||mode==3||mode==8||mode==9)
			{
				require(framebuffer.mapping->current(),"Hole write/local tag change invalidated native data mapping");
				image=resolver.prepare(descriptor,surfaces).finish();require(surfaces.status(source).dirty&&resolver.statistics().mixed_casts==2,"Fresh mixed display lost native authority orreused stale VM payload");
			}
			if(image)
			{
				presenter present(ctx);const auto pixels=render(ctx,present,*image);const std::array<unsigned,8> expected{51,34,17,255,64,96,mode>=8?32u:128u,255};
				for(unsigned i=0;i<expected.size();++i)require(pixels[i]==std::byte(expected[i]),"Guarded mixed native/VM pixel differs");
				if(original)
				{
					const auto frozen=render(ctx,present,*original);require(frozen[6]==std::byte{128},"Old immutable mixed image changed after VM hole write");
				}
			}
			require(resolver.statistics().entries==unsigned(bool(image))&&!guest_memory_epoch::active(),"Guarded mixed display published stale work orleaked VM exclusion");
			resolver.clear();surfaces.clear_snapshots();(void)ctx.trim_upload_pages();
		}
		require(budget->statistics().reserved_bytes==0,"Guarded mixed display retained old allocations");
	}
	std::cout<<"Mixed display guarded adapter:10 real VM-revision/mapping cases,exact4byte hole capture,native/VM pixels,same-value/pending/ready hole writes,local tile ABA,actual-EA mapping ABA,native writes andunmap refuse stale jobs;padding/unrelated tiles stay valid,fresh hole/tag retry preserves dirty native source,old frozen VM pixels immutable,CPU-epoch guards andall ledger retirement passed\n";
}
void assembly_guarded_revision_cases()
{
	using namespace metal4;
	for(unsigned mode=0;mode<11;++mode)
	{
		context ctx(3);auto budget=std::make_shared<heap_budget>(ctx,8u<<20);ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			auto provider=std::make_shared<mapped_provider>();auto ram=std::make_shared<guarded_guest_memory>(provider);auto tiles=std::make_shared<tile_state>();
			const cpu_capture_dispatch capture=tile_capture_dispatch([provider](const auto& work){provider->execute([&](auto&){work();});},tiles);
			surface_cache surfaces(ctx,ram,8u<<20,budget,capture);display_resolver resolver(ctx,ram,[](unsigned o,unsigned,std::size_t){return o;},budget,4096,capture,tiles);
			std::array<cached_surface_ref,2> native;
			for(unsigned i=0;i<2;++i)
			{
				const unsigned address=i?0x109:0x100;framebuffer_surface fb;fb.address=address;fb.guest=plan_guest_surface(address,1,1,4,4);
				checked_cpu_capture(capture,[&]{map_framebuffer_surface(fb,address,0,[](unsigned o,unsigned,std::size_t){return o;},{},tiles,true,4096);});
				native[i]=surfaces.acquire({address,{1,1,1,surface_color::argb8,false},4,{}, {},fb.mapping->state().segments,fb.mapping});
				surfaces.clear(native[i],0xf0,i?0x00556677:0x00112233,0,255,{0,0,1,1});
			}
			for(unsigned address:{0x104u,0x10du})ram->write({address,4},std::array{std::byte{0},std::byte{128},std::byte{96},std::byte{64}});
			const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};
			const display_snapshot descriptor{0x100,2,2,9};auto job=resolver.prepare(descriptor,surfaces,std::array{display_upload_dependency{gate,1}});
			require(!job.ready()&&resolver.statistics().assembly_casts==1&&resolver.statistics().captured_vm_bytes==8&&resolver.statistics().captured_vm_ranges==2,"Guarded assembly didnotcapture exacttwo holes");
			std::optional<present_image> old;if(mode==9){gate.signaledValue=1;old=job.finish();}
			if(mode==1||mode==9)ram->write({0x105,1},std::array{std::byte{32}});
			if(mode==2)ram->write({0x100,1},std::array{std::byte{55}});
			if(mode==3)ram->write({0x109,1},std::array{std::byte{55}});
			if(mode==4)provider->execute([&](auto&){provider->bank=1;ram->notify_mapping_changed({0x109,4});provider->bank=0;ram->notify_mapping_changed({0x109,4});});
			if(mode==5){tiles->update(0,{0,0,65536,1024,8,0,3,true});tiles->update(0,{});}
			if(mode==6)provider->execute([&](auto&){provider->mapped=false;ram->notify_mapping_changed({0x109,4});});
			if(mode==7)ram->write({0x108,1},std::array{std::byte{55}});
			if(mode==8){tiles->update(1,{0,65536,65536,256,0,0,0,true});tiles->update(1,{});}
			if(mode==10)ram->write({0x104,1},std::array{std::byte{0}});
			const bool accepted=mode==0||mode==7||mode==8;
			if(accepted)provider->execute([&](auto&){require(job.current(),"Guarded assembly lost exactactive receipt");rejects([&]{(void)job.finish();});rejects([&]{(void)resolver.prepare(descriptor,surfaces);});});
			gate.signaledValue=1;std::optional<present_image> image;if(accepted)image=job.finish();else rejects([&]{(void)job.finish();});
			if(mode==1||mode==5||mode==9||mode==10)
			{
				image=resolver.prepare(descriptor,surfaces).finish();require(resolver.statistics().assembly_casts==2&&surfaces.status(native[0]).dirty&&surfaces.status(native[1]).dirty,"Guarded assembly retrylostnative authority");
			}
			if(image)
			{
				presenter present(ctx);const auto actual=render(ctx,present,*image);const std::array<unsigned,16> expected{51,34,17,255,64,96,mode==1||mode==9?32u:128u,255,119,102,85,255,64,96,128,255};
				for(unsigned i=0;i<expected.size();++i)require(actual[i]==std::byte(expected[i]),"Guarded assembly native/VMpixels differ");
				if(old)require(render(ctx,present,*old)[6]==std::byte{128},"Frozen assembly VMpixels changed");
			}
			require(resolver.statistics().entries==unsigned(bool(image))&&!guest_memory_epoch::active(),"Guarded assembly published stalejob orleaked exclusion");resolver.clear();surfaces.clear_snapshots();(void)ctx.trim_upload_pages();
		}
		require(budget->statistics().reserved_bytes==0,"Guarded assembly retained native/VM allocations");
	}
	std::cout<<"Assembly guarded adapter:11 real metadata/revision cases,two native sources+two exact4byte VM holes,independent source writes,mapping ABA/unmap/localtile ABA/same-value andpending-ready VMwrites refuse stalejobs,padding/unrelated tilechanges accepted,freshretry preservesdirty authority,immutable oldpixels,CPU exclusion andallledger retirement passed\n";
}
void failure_and_lifetime_cases()
{
	using namespace metal4;
	using namespace std::chrono_literals;
	for (unsigned mode = 0; mode < 5; ++mode)
	{
		context ctx(1,2,mode == 3 ? 100ms : 5s);
		auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			auto ram = std::make_shared<memory>(); fill(*ram,0x100);
			auto tiles = std::make_shared<tile_state>();
			const auto capture = tile_capture_dispatch([](const auto& work) { work(); },tiles);
			surface_cache surfaces(ctx,ram,8u << 20,budget,capture);
			display_resolver resolver(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; },budget,4096,capture,tiles);
			const auto gate = [ctx.device() newSharedEvent]; const std::array waits{display_upload_dependency{gate,1}};
			const display_snapshot descriptor{0x100,2,2,24};
			auto job = resolver.prepare(descriptor,surfaces,waits);
			require(!job.ready(), "Display negative control was not truly gated");
			if (mode == 0) { fill(*ram,0x100,17); gate.signaledValue = 1; }
			if (mode == 1) { tiles->update(0,{0,0,65536,256,0,0,0,true}); tiles->update(0,{}); gate.signaledValue = 1; }
			if (mode == 2) require(job.cancel_pending(), "Queued display cancellation refused");
			if (mode == 4) ctx.shutdown();
			rejects([&] { (void)job.finish(5s); });
			require(job.ready() && resolver.statistics().entries == 0 && resolver.statistics().published == 0, "Failed display job published an image");
			if (mode < 4)
			{
				ctx.recover_completed_errors();
				auto retry = resolver.prepare(descriptor,surfaces); (void)retry.finish();
				require(resolver.statistics().uploads == 2 && resolver.statistics().entries == 1, "Display retry reused a failed pending token");
			}
			resolver.clear(); (void)ctx.trim_upload_pages();
		}
		require(budget->statistics().reserved_bytes == 0, "Failed/retried display kept orphan native charges");
	}
	context consumer(1); presenter present(consumer);
	std::shared_ptr<heap_budget> ledger;
	std::weak_ptr<memory> weak_memory;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan; id<MTLSharedEvent> gate;
		@autoreleasepool
		{
			context owner(1,2,5s);
			ledger = std::make_shared<heap_budget>(owner,8u << 20); owner.set_buffer_budget(ledger);
			auto ram = std::make_shared<memory>(); fill(*ram,0x100); weak_memory = ram;
			surface_cache surfaces(owner,ram,8u << 20,ledger);
			display_resolver resolver(owner,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; },ledger,4096);
			gate = [owner.device() newSharedEvent];
			const std::array waits{display_upload_dependency{gate,1}};
			orphan.emplace(resolver.prepare({0x100,2,2,24},surfaces,waits));
		}
		require(!orphan->ready() && !weak_memory.expired() && ledger->statistics().reserved_bytes > 0,
			"Pending display did not own its receipt/data beyond resolver/context teardown");
		rejects([&] { (void)orphan->finish(1ms); });
		gate.signaledValue = 1;
		const auto image = orphan->finish(5s);
		orphan.reset(); require(weak_memory.expired(), "Completed display kept its VM provider after job release");
		verify(render(consumer,present,image),{128,64,32,255});
	}
	require(ledger->statistics().reserved_bytes == 0, "Released detached display retained backing after GPU/CPU readback scope");
	std::cout << "Display failures:late CPU write/tile ABA,cancel,deadline,shutdown publish no image;fresh retries skip failed pending jobs;gated upload survives resolver/surface/context destruction with exact rendered pixels,receipt provider retires at job release andall native charges retire passed\n";
}
void native_failure_and_lifetime_cases()
{
	using namespace metal4;
	using namespace std::chrono_literals;
	for (unsigned mode = 0; mode < 5; ++mode)
	{
		context ctx(2,2,mode == 3 ? 100ms : 5s);
		auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			auto ram = std::make_shared<memory>(); fill(*ram,0x100);
			auto tiles = std::make_shared<tile_state>();
			const auto capture = tile_capture_dispatch([](const auto& work) { work(); },tiles);
			surface_cache surfaces(ctx,ram,8u << 20,budget,capture);
			display_resolver resolver(ctx,ram,[](unsigned o,unsigned,std::size_t) { return 0x1000u + o; },budget,4096,capture,tiles);
			const auto source = surfaces.acquire({0x1100,{2,2,1,surface_color::argb8,false},24,{}});
			surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,2,2});
			const auto gate = [ctx.device() newSharedEvent];
			struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
			auto job = resolver.prepare({0x100,2,2,24},surfaces,std::array{display_upload_dependency{gate,1}});
			require(!job.ready() && resolver.statistics().native_submissions == 1, "Native display failure fixture did not remain pending");
			if (mode == 0) { fill(*ram,0x100,17); gate.signaledValue = 1; }
			if (mode == 1) { tiles->update(0,{0,0,65536,256,0,0,0,true}); tiles->update(0,{}); gate.signaledValue = 1; }
			if (mode == 2) require(job.cancel_pending(), "Queued native display cancellation refused");
			if (mode == 4) ctx.shutdown();
			bool refused = false;
			try { (void)job.finish(5s); }
			catch (const surface_conflict_error&) { refused = mode < 2; }
			catch (const std::runtime_error& error)
			{
				const std::string message = error.what();
				refused = mode == 3 ? message.find("deadline expired") != std::string::npos : (mode == 2 || mode == 4) && message.find("cancelled") != std::string::npos;
			}
			require(refused && job.ready() && surfaces.snapshot_statistics().entries == 0 && surfaces.snapshot_statistics().created == 0 &&
				resolver.statistics().entries == 0 && resolver.statistics().published == 0 && resolver.statistics().uploads == 0 && ram->reads == 0 && ram->writes == 0,
				"Failed native display exposed an image,cached a snapshot or touched VM pixels");
			if (mode < 4)
			{
				ctx.recover_completed_errors();
				surfaces.clear(source,0xf0,0x00445566,0,255,{0,0,2,2});
				const auto retry = resolver.prepare({0x100,2,2,24},surfaces).finish();
				presenter present(ctx); verify(render(ctx,present,retry),{102,85,68,255});
				require(resolver.statistics().native_submissions == 2 && surfaces.snapshot_statistics().entries == 1 && ram->reads == 0,
					"Native display retry reused failed work or an obsolete source generation");
			}
			resolver.clear(); surfaces.clear_snapshots(); (void)ctx.trim_upload_pages();
		}
		require(budget->statistics().reserved_bytes == 0, "Failed/retried native display retained native source/target/view charges");
	}
	context consumer(1); presenter present(consumer);
	std::shared_ptr<heap_budget> ledger; std::weak_ptr<memory> weak_memory;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan; id<MTLSharedEvent> gate;
		@autoreleasepool
		{
			context owner(2); ledger = std::make_shared<heap_budget>(owner,8u << 20); owner.set_buffer_budget(ledger);
			auto ram = std::make_shared<memory>(); weak_memory = ram;
			surface_cache surfaces(owner,ram,8u << 20,ledger);
			display_resolver resolver(owner,ram,[](unsigned o,unsigned,std::size_t) { return 0x1000u + o; },ledger,4096);
			const auto source = surfaces.acquire({0x1100,{2,2,1,surface_color::argb8,false},24,{}});
			surfaces.clear(source,0xf0,0x00112233,0,255,{0,0,2,2});
			gate = [owner.device() newSharedEvent];
			orphan.emplace(resolver.prepare({0x100,2,2,24},surfaces,std::array{display_upload_dependency{gate,1}}));
		}
		require(!orphan->ready() && !weak_memory.expired() && ledger->statistics().reserved_bytes > 0,
			"Native display lost its receipt/lease owners beyond context/cache teardown");
		rejects([&] { (void)orphan->finish(1ms); }); gate.signaledValue = 1;
		const auto image = orphan->finish(); orphan.reset();
		require(weak_memory.expired(), "Completed orphan native display kept its CPU service beyond job release");
		verify(render(consumer,present,image),{51,34,17,255});
	}
	require(ledger->statistics().reserved_bytes == 0, "Orphan native display retained source/target charges after final view release");
	{
		context owner(2); auto budget = std::make_shared<heap_budget>(owner,8u << 20); owner.set_buffer_budget(budget);
		@autoreleasepool
		{
			auto ram = std::make_shared<memory>();
			std::unique_ptr<surface_cache> surfaces;
			std::unique_ptr<display_resolver> resolver;
			std::optional<display_upload_job> job;
			bool armed = false, reentry_refused = false;
			const cpu_capture_dispatch capture = [&](const auto& work)
			{
				if (armed)
				{
					armed = false;
					try { (void)job->finish(); } catch (const std::logic_error&) { reentry_refused = true; }
					surfaces->clear_snapshots(); surfaces.reset(); resolver.reset(); job.reset();
				}
				work();
			};
			surfaces = std::make_unique<surface_cache>(owner,ram,8u << 20,budget,capture);
			resolver = std::make_unique<display_resolver>(owner,ram,[](unsigned o,unsigned,std::size_t) { return 0x1000u + o; },budget,4096,capture);
			const auto source = surfaces->acquire({0x1100,{2,2,1,surface_color::argb8,false},24,{}});
			surfaces->clear(source,0xf0,0x00112233,0,255,{0,0,2,2});
			job.emplace(resolver->prepare({0x100,2,2,24,display_format::xbgr8},*surfaces));
			armed = true; const auto image = job->finish();
			require(reentry_refused && !surfaces && !resolver && !job, "Display final capture did not test reentry andpublic-owner destruction");
			verify(render(consumer,present,image),{17,34,51,255});
			(void)owner.trim_upload_pages();
		}
		require(budget->statistics().reserved_bytes == 0, "Display callback-owner destruction retained native charges");
	}
	std::cout << "Native display failures:CPU write/tile ABA/cancel/deadline/shutdown publish no snapshot orimage;fresh native retries exact andVM untouched;gated copy/view/receipt outlives resolver/surfaces/context/pools,another native context renders16exact BGRA bytes,all charges retire;finish survives callback job/cache deletion with frozen AV format andrefuses reentry passed\n";
}
void pressure_and_pending_bound_cases()
{
	using namespace metal4;
	{
		constexpr std::size_t budget_bytes = 2u << 20; // Includes the native 1MiB upload arena.
		context ctx(1); auto budget = std::make_shared<heap_budget>(ctx,budget_bytes); ctx.set_buffer_budget(budget);
		auto ram = std::make_shared<memory>(); fill(*ram,0x100);
		surface_cache surfaces(ctx,ram,budget_bytes,budget);
		guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; },4096,budget);
		display_resolver resolver(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; },budget,4096);
		memory_pressure_owner pressure(ctx,surfaces,textures,budget,&resolver);
		@autoreleasepool { (void)resolver.prepare({0x100,2,2,24},surfaces).finish(); (void)ctx.trim_upload_pages(); }
		require(resolver.statistics().entries == 1 && budget->statistics().reserved_bytes > 0, "Display pressure fixture has no cached allocation");
		@autoreleasepool
		{
			const auto full = make_buffer(ctx,budget_bytes,buffer_storage::shared);
			require(resolver.statistics().entries == 0 && resolver.statistics().evictions == 1 && full.length == budget_bytes,
				"Shared memory pressure did not release the completed CPU backbuffer");
		}
		require(budget->statistics().reserved_bytes == 0, "Display memory-pressure recovery kept retired backing");
		@autoreleasepool
		{
			auto escaped = resolver.prepare({0x100,2,2,24},surfaces).finish(); (void)ctx.trim_upload_pages();
			rejects([&] { (void)make_buffer(ctx,budget_bytes,buffer_storage::shared); });
			require(resolver.statistics().entries == 0 && budget->statistics().reserved_bytes > 0,
				"Display pressure released backing still owned by an escaped AV view");
			escaped.texture = nil;
		}
		require(budget->statistics().reserved_bytes == 0, "Last escaped display release did not restore quota");
	}
	{
		context ctx(1); auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
		auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,budget);
		display_resolver resolver(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; },budget,4096);
		std::vector<display_upload_job> jobs;
		for (unsigned i = 0; i < 64; ++i) jobs.push_back(resolver.prepare({i * 48,2,2,24},surfaces));
		const auto reads = ram->reads; const auto created = budget->statistics().created;
		rejects([&] { (void)resolver.prepare({64 * 48,2,2,24},surfaces); });
		require(resolver.statistics().entries == 0 && resolver.statistics().uploads == 64 && ram->reads == reads && budget->statistics().created == created,
			"Sixty-fifth unconsumed display job escaped bounded admission");
		jobs.clear(); (void)resolver.prepare({64 * 48,2,2,24},surfaces).finish();
		require(resolver.statistics().uploads == 65 && resolver.statistics().entries == 1, "Retired pending owners did not restore display admission");
		display_resolver completed(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u + offset; },budget,4096,{}, {},{64,8u << 20});
		for (unsigned i = 0; i < 65; ++i) (void)completed.prepare({i * 48,2,2,24},surfaces).finish();
		require(completed.statistics().uploads == 65 && completed.statistics().entries == 64 && completed.statistics().evictions == 1,
			"Confirmed cache entries were incorrectly counted as unconsumed pending jobs");
	}
	std::cout << "Display shared-ledger pressure evicts completed cache owner without GPU wait;64 unconsumed versions bounded,65th refused before capture/allocation,owner retirement restores admission passed\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			job_and_cache_cases();
			native_job_cases();
			native_generation_and_bound_cases();
			exact_revision_cases();
			native_exact_revision_cases();
			mixed_guarded_revision_cases();
			assembly_guarded_revision_cases();
			failure_and_lifetime_cases();
			native_failure_and_lifetime_cases();
			pressure_and_pending_bound_cases();
			using namespace metal4;
			context ctx;
			auto ram = std::make_shared<memory>();
			auto budget = std::make_shared<heap_budget>(ctx, 8u << 20);
			surface_cache surfaces(ctx, ram, 8u << 20, budget);
			unsigned translations = 0;
			const auto translate = [&](std::uint32_t offset, unsigned location, std::size_t size)
			{
				++translations;
				require(location == 0 && offset <= 4096 && size <= 4096 - offset, "Wrong local display translation");
				return offset >= 0x800 ? 0x2000 + offset - 0x800 : 0x1000 + offset;
			};
			display_resolver resolver(ctx, ram, translate, budget, 4096);
			presenter presenter(ctx);
			for (const auto format : {display_format::xrgb8, display_format::xbgr8, display_format::rgbx16f})
			for (unsigned pitch : {24u,25u})
			{
				std::fill(ram->bytes.begin(), ram->bytes.end(), std::byte{0xcd});
				++ram->version;
				const std::vector<unsigned> pixel = format == display_format::rgbx16f ? std::vector<unsigned>{0x3c, 0, 0x38, 0, 0x34, 0, 0, 0} : format == display_format::xbgr8 ? std::vector<unsigned>{0, 128, 64, 32}
																																												 : std::vector<unsigned>{0, 32, 64, 128};
				for (unsigned y = 0; y < 2; ++y)
					for (unsigned x = 0; x < 2; ++x)
						for (unsigned b = 0; b < pixel.size(); ++b)
							ram->bytes[0x100 + y * pitch + x * pixel.size() + b] = std::byte(pixel[b]);
				const auto before = ram->bytes;
				const auto image = resolver.resolve({0x100, 2, 2, pitch, format}, surfaces);
				verify(render(ctx, presenter, image), format == display_format::rgbx16f ? std::array<unsigned, 4>{64, 128, 255, 255} : std::array<unsigned, 4>{128, 64, 32, 255});
				require(ram->bytes == before && ram->writes == 0, "Display upload changed guest/padding bytes");
			}
			std::cout << "Three AV formats from even/odd padded big-endian VM rows;X ignored,opaque alpha,VM unchanged passed\n";
			const cached_surface_description description{0x1000, {4, 4, 1, surface_color::argb8, false}, 24, {}};
			const auto target = surfaces.acquire(description);
			surfaces.clear(target, 0xf0, 0x00112233, 0, 0, {0, 0, 4, 4});
			const auto reads = ram->reads;
			const auto before = ram->bytes;
			const auto frozen = resolver.resolve({28, 2, 2, 24, display_format::xrgb8}, surfaces);
			verify(render(ctx, presenter, frozen), {51, 34, 17, 255});
			const auto allocations = budget->statistics().created;
			const auto copies = surfaces.snapshot_statistics().created;
			const auto submissions = surfaces.transaction_statistics().submitted;
			for (unsigned repeat = 0; repeat < 100; ++repeat)
				require(resolver.resolve({28, 2, 2, 24, display_format::xrgb8}, surfaces).texture == frozen.texture,
					"Repeated native display recreated its opaque-alpha view");
			require(surfaces.snapshot_statistics().created == copies && surfaces.transaction_statistics().submitted == submissions &&
				budget->statistics().created == allocations && surfaces.snapshot_statistics().views_reused == 100,
				"Repeated native display allocated/copied/submitted unchanged source pixels");
			const auto reinterpreted = resolver.resolve({28, 2, 2, 24, display_format::xbgr8}, surfaces);
			require(reinterpreted.texture != frozen.texture && surfaces.snapshot_statistics().created == copies &&
				resolver.resolve({28, 2, 2, 24, display_format::xbgr8}, surfaces).texture == reinterpreted.texture,
				"AV reinterpretation did not share its native copy while retaining distinct cached views");
			verify(render(ctx, presenter, reinterpreted), {17, 34, 51, 255});
			require(ram->reads == reads && ram->bytes == before && ram->writes == 0 && surfaces.status(target).dirty, "Native display forced VM upload/writeback");
			surfaces.clear(target, 0xf0, 0x00445566, 0, 0, {0, 0, 4, 4});
			verify(render(ctx, presenter, frozen), {51, 34, 17, 255});
			const auto latest = resolver.resolve({28, 2, 2, 24, display_format::xrgb8}, surfaces);
			verify(render(ctx, presenter, latest), {102, 85, 68, 255});
			const auto prior_reads = ram->reads;
			const auto mirrored = resolver.resolve({0x800 + 28, 2, 2, 24, display_format::xrgb8}, surfaces);
			require(mirrored.texture == latest.texture, "Physical display alias duplicated an unchanged native image/view");
			verify(render(ctx, presenter, mirrored), {102, 85, 68, 255});
			require(ram->reads == prior_reads && ram->writes == 0, "Physical display mirror fell back to VM");
			std::cout << "Physical display mirror preserves dirty native epoch without VM upload passed\n";
			std::cout << "Native display cache:100 identical flips,no snapshot copies/heap allocations/submissions;distinct AV views reuse one image andphysical aliases reuse the exact opaque view passed\n";
			{
				surface_cache segmented(ctx, ram, 8u << 20, budget, [](const auto& operation) { operation(); });
				const std::vector<guest_storage_segment> parts{{0x1600, 0, 16}, {0x2610, 16, 80}};
				const auto source = segmented.acquire({0x1600, {4, 4, 1, surface_color::argb8, false}, 24, {}, {}, parts});
				segmented.clear(source, 0xf0, 0x00112233, 0, 0, {0, 0, 4, 4});
				const auto reads = ram->reads;
				const auto old = resolver.resolve({0x600, 2, 2, 24, display_format::xrgb8}, segmented);
				verify(render(ctx, presenter, old), {51, 34, 17, 255});
				segmented.clear(source, 0xf0, 0x00445566, 0, 0, {0, 0, 4, 4});
				const auto fresh = resolver.resolve({0x600, 2, 2, 24, display_format::xrgb8}, segmented);
				verify(render(ctx, presenter, fresh), {102, 85, 68, 255}); verify(render(ctx, presenter, old), {51, 34, 17, 255});
				require(ram->reads == reads && ram->writes == 0 && segmented.status(source).dirty, "Segmented mirrored native display forced VM conversion");
				std::cout << "Native disjoint physical-mirror display:two source segments,padded crop,immutable old/new epochs,no VM pixel read/writeback passed\n";
			}
			const auto float_target = surfaces.acquire({0x1300, {4, 4, 1, surface_color::rgba16f, false}, 40, {}});
			// RSX conventional clear is NOP on float RTs. Establish genuine
			// native float contents through the cache's draw transaction instead.
			surfaces.prepare(float_target, surface_initialization::background);
			const std::array<cached_surface_use, 1> float_uses{{{float_target, true}}};
			const auto float_draw = [](context& recording, std::span<const id<MTLTexture>> textures)
			{
				MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
				pass.colorAttachments[0].texture = textures[0];
				pass.colorAttachments[0].loadAction = MTLLoadActionClear;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				pass.colorAttachments[0].clearColor = MTLClearColorMake(1, .5, .25, 0);
				(void)recording.render(pass);
			};
			surfaces.draw(float_uses, float_draw);
			const auto float_reads = ram->reads;
			const auto float_image = resolver.resolve({0x300 + 48, 2, 2, 40, display_format::rgbx16f}, surfaces);
			verify(render(ctx, presenter, float_image), {64, 128, 255, 255});
			require(ram->reads == float_reads && surfaces.status(float_target).dirty && ram->writes == 0, "Native float display fell back to VM");
			const auto abgr_target = surfaces.acquire({0x1400, {4, 4, 1, surface_color::abgr8, false}, 24, {}});
			// RSX ABGR clear register remaps R/B, unlike ordinary draw exports.
			surfaces.clear(abgr_target, 0xf0, 0x00332211, 0, 0, {0, 0, 4, 4});
			const auto abgr_image = resolver.resolve({0x400 + 28, 2, 2, 24, display_format::xbgr8}, surfaces);
			verify(render(ctx, presenter, abgr_image), {51, 34, 17, 255});
			const auto abgr_reinterpreted = resolver.resolve({0x400 + 28, 2, 2, 24, display_format::xrgb8}, surfaces);
			verify(render(ctx, presenter, abgr_reinterpreted), {17, 34, 51, 255});
			std::cout << "Native half-float and ABGR display crops,opaque alpha and reverse AV reinterpretation passed\n";
			std::cout << "Dirty native cropped display,no VM quantization,AV byte reinterpretation and immutable old/new epochs passed\n";
			surfaces.clear(target, 0xf0, 0x00112233, 0, 0, {0, 0, 4, 4});
			const auto cast_reads = ram->reads;
			verify(render(ctx,presenter,resolver.resolve({0,2,2,24,display_format::rgbx16f},surfaces)),{0,3,0,255});
			require(ram->reads == cast_reads && ram->writes == 0,"Native ARGB/half display cast used VM pixels");
			const auto prior = translations;
			rejects([&]
					{ (void)resolver.resolve({0, 2, 2, 1}, surfaces); });
			rejects([&]
					{ (void)resolver.resolve({0, 2, 2000, 24}, surfaces); });
			require(prior == translations, "Invalid display reached translation");
			ram->conflict = true;
			rejects([&]
					{ (void)resolver.resolve({0x200, 2, 2, 24}, surfaces); });
			ram->conflict = false;
			require(ram->writes == 0, "Rejected display wrote VM");
			std::cout << "Native ARGB/half display bitcast without VM fallback;invalid plans before translation;revision conflict refused passed\n";
			return 0;
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
