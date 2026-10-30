#include "M4MappingInvalidation.h"
#include "M4SampleTransfer.h"
#include "M4HostAccessFixture.h"
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename Operation> void refuses(Operation&& operation, const char* message)
{
	bool failed = false;
	try { operation(); } catch (const std::exception&) { failed = true; }
	require(failed, message);
}
struct fixture
{
	std::shared_ptr<test::host_access_fixture> provider = std::make_shared<test::host_access_fixture>();
	std::shared_ptr<guarded_guest_memory> memory;
	std::function<void()> outside_capture;
	cpu_capture_dispatch capture = [&](const auto& op) { if (outside_capture) outside_capture(); provider->execute([&](auto&) { op(); }); };
	std::shared_ptr<io_mapping_epoch> io = std::make_shared<io_mapping_epoch>();
	std::shared_ptr<heap_budget> budget;
	surface_cache surfaces;
	guest_texture_cache textures;
	mapping_invalidation_service service;
	explicit fixture(context& device, std::size_t registrations = 1024)
		: memory(std::make_shared<guarded_guest_memory>(provider, registrations)), budget(std::make_shared<heap_budget>(device, 32 << 20)), surfaces(device, memory, 16 << 20, budget, capture),
		  textures(device, memory, [](unsigned offset, unsigned, std::size_t) { return offset; }, 1 << 20, budget, 16, {capture, io}),
		  service(device, surfaces, textures)
	{
		provider->pin_first_alias();
	}
	cached_surface_ref target(std::uint32_t address = 4096, std::size_t pitch = 0)
	{
		return surfaces.acquire({address, {4, 4, 1, surface_color::argb8, false}, pitch, {}});
	}
	void change(bool unmap)
	{
		provider->execute([&](auto&)
		{
			memory->begin_mapping_change({0, 65536});
			if (unmap) provider->unmap_first(); else provider->replace_first();
			memory->end_mapping_change({0, 65536});
		});
	}
	void dirty(const cached_surface_ref& target) { surfaces.clear(target, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 4}); }
};
void old_owner_matrix(context& device)
{
	unsigned count = 0;
	for (auto cause : {mapping_notifications::kind::read, mapping_notifications::kind::unmap})
		for (bool unmap : {false, true}) for (bool alias : {false, true}) for (bool deferred : {false, true})
		{
			fixture f(device);
			const auto target = f.target();
			if (deferred)
			{
				const auto source = f.target(98304); f.dirty(source);
				f.surfaces.defer_copy(source, {0, 0, 4, 4}, target, 0, 0);
				f.surfaces.erase(source);
			}
			else f.dirty(target);
			f.change(unmap);
			const auto replacement = std::vector<std::byte>(64, std::byte{0x6d});
			if (!unmap) f.memory->write({4096, 64}, replacement);
			require(f.service.service({alias ? 0x41000u : 4096u, 64}, cause) == 1, "Native mapping service omitted an original/physical alias output");
			const auto old = f.memory->read({0x41000, 64}).bytes;
			for (unsigned i = 0; i < 64; ++i)
				require(old[i] == std::array{std::byte{255}, std::byte{0x12}, std::byte{0x34}, std::byte{0x56}}[i % 4], "Native service lost old physical native/deferred bytes");
			if (!unmap) require(f.memory->read({4096, 64}).bytes == replacement, "Native mapping service overwrote replacement CPU bytes");
			const auto state = f.surfaces.status(target);
			require(!state.dirty && !state.initialized && !state.faulted && !state.deferred_copies, "Native service retained published old authority");
			require(f.service.service({alias ? 0x41000u : 4096u, 64}, cause) == 0, "Repeated native service republished clean output");
			++count;
		}
	std::cout << count << " original/alias read/unmap remap/absent completed/deferred profiles\n";
}
void selection_and_retry(context& device)
{
	fixture f(device);
	const auto target = f.target(4096, 24); f.dirty(target);
	f.change(false);
	require(f.service.service({0x41010, 8}, mapping_notifications::kind::read) == 0, "Padding-only physical notification selected active output");
	require(f.service.service({0x42000, 16}, mapping_notifications::kind::read) == 0, "Different physical offset selected output");
	require(f.surfaces.status(target).dirty, "Unrelated mapping notification lost dirty authority");
	f.provider->fail_reservation_planning = true;
	refuses([&] { (void)f.service.service({0x41000, 16}, mapping_notifications::kind::unmap); }, "Reservation planning failure was ignored");
	require(f.surfaces.status(target).dirty && !f.surfaces.status(target).faulted, "Failed service discarded/faulted old output");
	f.provider->fail_reservation_planning = false;
	require(f.service.service({0x41000, 16}, mapping_notifications::kind::unmap) == 1, "Native service could not retry reservation planning failure");
	fixture protected_memory(device);
	const auto protected_target = protected_memory.target(); protected_memory.dirty(protected_target);
	protected_memory.provider->execute([&](auto&) { protected_memory.provider->protect_first(false); });
	require(protected_memory.service.service({0x41000, 64}, mapping_notifications::kind::unmap) == 1, "Physical alias selection/publication mapped protected original VA");
	protected_memory.provider->execute([&](auto&) { protected_memory.provider->protect_first(true); });
	// A new CPU write invalidates the physical ticket;selection must still find
	// it and refuse publication rather than silently forget the old output.
	fixture stale(device); const auto stale_target = stale.target(); stale.dirty(stale_target); stale.change(false);
	const std::vector<std::byte> newer(64, std::byte{0x7a}); stale.memory->write({0x41000, 64}, newer);
	refuses([&] { (void)stale.service.service({0x41000, 64}, mapping_notifications::kind::unmap); }, "Stale physical output overwrote newer CPU data");
	require(stale.surfaces.status(stale_target).dirty && stale.memory->read({0x41000, 64}).bytes == newer, "Stale selection lost native authority/new CPU data");
	std::cout << "Active padding/offset selection;protected original;reservation retry;stale CPU refusal\n";
}
void read_precision_profile(context& device, unsigned profile)
{
	fixture f(device);
	cached_surface_description description{4096, {4, 1, 1, surface_depth::z24s8_uint, false}, 0, {}};
	if (profile == 4) description.segments = {{4096, 0, 7}, {65536, 7, 9}};
	const auto target = f.surfaces.acquire(description);
	f.surfaces.prepare(target);
	std::array<float, 4> values{0.123456f, 0.50001f, 0.314159f, 0.876543f};
	const auto input = make_upload(device, std::as_bytes(std::span(values))), output = make_buffer(device, 16, buffer_storage::shared);
	const std::array write{cached_surface_use{target, true}}, read{cached_surface_use{target, false}};
	texture_region region; region.size = MTLSizeMake(4, 1, 1);
	f.surfaces.draw(write, [&](context& commands, auto textures) { buffer_to_texture(commands, input, 0, textures[0], region, 16, 0, texture_plane::depth); });
	const auto epoch = f.surfaces.status(target).native_write_epoch;
	if (profile == 1) f.surfaces.spill(target);
	if (profile == 2 || profile == 3)
	{
		const auto source = f.surfaces.acquire({98304, description.allocation, 0, {}});
		const std::array<float, 4> source_values{0.876543f, 0.314159f, 0.50001f, 0.123456f};
		const auto source_input = make_upload(device, std::as_bytes(std::span(source_values)));
		const std::array source_write{cached_surface_use{source, true}};
		f.surfaces.draw(source_write, [&](context& commands, auto textures) { buffer_to_texture(commands, source_input, 0, textures[0], region, 16, 0, texture_plane::depth); });
		if (profile == 2) { f.surfaces.defer_copy(source, {0, 0, 4, 1}, target, 0, 0); values = source_values; }
		else { f.surfaces.defer_copy(source, {0, 0, 2, 1}, target, 2, 0); values[2] = source_values[0]; values[3] = source_values[1]; }
		f.surfaces.erase(source);
	}
	fragment_texture_registers words;
	words.offset = 98304; words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 8 | 2;
	words.address = (7u << 28) | 0x030303; words.control0 = (1u << 31) | (4095u << 7);
	words.control1 = 0xaae4; words.filter = (1u << 24) | (3u << 16); words.image_rect = (4u << 16) | 4; words.control3 = (1u << 20) | 16;
	(void)f.textures.resolve(words);
	const auto nulls = f.textures.null_images();
	require(f.textures.statistics().entries == 1, "Texture fixture did not create immutable cache entry");
	require(f.service.service({4096, 16}, mapping_notifications::kind::read) == 1, "Current native read did not publish dirty output");
	const auto state = f.surfaces.status(target);
	const bool native_epoch = profile == 2 || profile == 3 ? state.native_write_epoch > epoch : state.native_write_epoch == epoch;
	require(state.initialized && !state.dirty && native_epoch && epoch, "Read notification discarded native precision/provenance");
	f.surfaces.draw(read, [&](context& commands, auto textures) { texture_to_buffer(commands, textures[0], region, output, 0, 16, 0, texture_plane::depth); });
	require(std::memcmp(map_buffer(output, 0, 16).data(), values.data(), 16) == 0, "Read notification quantized native depth bits");
	require(f.textures.statistics().entries == 1 && f.textures.null_images()[0].texture == nulls[0].texture, "Mapping service discarded an unrelated immutable entry or independent null image");
	(void)f.service.service({4096, 16}, mapping_notifications::kind::unmap);
	require(!f.surfaces.status(target).initialized, "Unmap notification retained clean native mirror");
}
void read_precision_and_texture_eviction(context& device)
{
	for (unsigned profile = 0; profile < 5; ++profile) read_precision_profile(device, profile);
	std::cout << "Current read preserves raw native depth bits/epoch;unrelated immutable texture preservation;null image lifetime\n";
	std::cout << "Physical read precision:5 raw Depth32F profiles;completed/raw-spill/full-deferred/partial-native-background/7-byte-disjoint;atomic virtual receipts;no guest quantization of native mirrors\n";
}
fragment_texture_registers texture_words(unsigned offset)
{
	fragment_texture_registers words;
	words.offset = offset; words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 8 | 2;
	words.address = (7u << 28) | 0x030303; words.control0 = (1u << 31) | (4095u << 7);
	words.control1 = 0xaae4; words.filter = (1u << 24) | (3u << 16);
	words.image_rect = (4u << 16) | 4; words.control3 = (1u << 20) | 16;
	return words;
}
void verify_image(context& device, id<MTLTexture> texture, unsigned value)
{
	const auto output = make_buffer(device, 4 * 256, buffer_storage::shared);
	texture_region region; region.size = MTLSizeMake(4, 4, 1);
	device.begin(); texture_to_buffer(device, texture, region, output, 0, 256); device.submit().wait();
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 16; ++x)
		require(static_cast<const unsigned char*>(output.contents)[y * 256 + x] == value, "Immutable/remapped native texture has stale or corrupted bytes");
}
void texture_mapping_selection(context& device)
{
	unsigned profiles = 0;
	for (auto cause : {mapping_notifications::kind::read, mapping_notifications::kind::unmap})
		for (unsigned state = 0; state < 4; ++state) for (bool alias : {false, true})
		{
			fixture f(device);
			f.memory->write({4096, 64}, std::vector<std::byte>(64, std::byte{0x35}));
			const auto original = f.textures.resolve(texture_words(4096));
			const auto unrelated = f.textures.resolve(texture_words(98304));
			if (state == 1 || state == 2) f.change(state == 2);
			if (state == 1) f.memory->write({4096, 64}, std::vector<std::byte>(64, std::byte{0x79}));
			if (state == 3) f.provider->execute([&](auto&) { f.provider->protect_first(false); });
			// No output needs publication. The texture selection itself must
			// retain unrelated hits even with the old VA absent or protected.
			require(f.service.service({alias ? 0x41000u : 4096u, 64}, cause) == 0, "Clean texture notification invented native output");
			require(f.textures.statistics().entries == 1 && f.textures.statistics().invalidations == 1,
				"Physical/original texture selection missed old image or evicted unrelated one");
			const auto before = f.textures.statistics();
			require(f.textures.resolve(texture_words(98304)).texture == unrelated.texture &&
				f.textures.statistics().hits == before.hits + 1 && f.textures.statistics().uploads == before.uploads,
				"Unrelated mapping forced an upload or returned a different native image");
			if (state == 3) f.provider->execute([&](auto&) { f.provider->protect_first(true); });
			verify_image(device, original.texture, 0x35);
			if (state != 2)
			{
				const auto fresh = f.textures.resolve(texture_words(4096));
				require(fresh.texture != original.texture, "Retired/remapped texture returned stale native image");
				verify_image(device, fresh.texture, state == 1 ? 0x79 : 0x35);
			}
			++profiles;
		}
	fixture retry(device);
	const auto a = retry.textures.resolve(texture_words(4096));
	const auto b = retry.textures.resolve(texture_words(98304));
	const auto uploads = retry.textures.statistics().uploads;
	(void)retry.service.service({0x42000, 16}, mapping_notifications::kind::unmap);
	require(retry.textures.statistics().entries == 2, "Different physical offset discarded immutable images");
	retry.provider->fail_physical_query = retry.provider->physical_queries + 2;
	refuses([&] { (void)retry.service.service({0x41000, 64}, mapping_notifications::kind::read); }, "Metadata query failure was ignored");
	require(retry.textures.statistics().entries == 2 && retry.textures.statistics().invalidations == 0, "Failed selection partially retired texture entries");
	retry.provider->fail_physical_query = 0;
	(void)retry.service.service({0x41000, 64}, mapping_notifications::kind::read);
	require(retry.textures.statistics().entries == 1 && retry.textures.resolve(texture_words(98304)).texture == b.texture &&
		retry.textures.statistics().uploads == uploads, "Retried physical selection lost unrelated cache hit");
	for (bool exhausted : {false, true})
	{
		fixture fallback(device, exhausted ? 4 : 1024);
		std::vector<std::shared_ptr<const physical_surface>> occupied;
		if (exhausted)
		{
			const auto coverage = fallback.memory->watch_surface(guest_byte_coverage({8192, 64}));
			const auto revision = fallback.memory->surface_revision(coverage);
			for (unsigned i = 0; i < 4; ++i) occupied.push_back(fallback.memory->capture_physical_surface(coverage, revision));
		}
		else fallback.provider->segments_supported = false;
		(void)fallback.textures.resolve(texture_words(4096));
		(void)fallback.textures.resolve(texture_words(98304));
		require(fallback.textures.statistics().entries == 2, "Physical metadata fallback prevented valid upload");
		(void)fallback.service.service({0x42000, 16}, mapping_notifications::kind::read);
		require(fallback.textures.statistics().entries == 0, "Unsupported/exhausted physical identity guessed non-overlap");
	}
	std::cout << "Unsupported and exhausted physical metadata conservatively evict;valid native uploads survive metadata budget pressure\n";
	std::cout << profiles << " texture mapping profiles:read/unmap,original/alias,live/replaced/absent/protected VA;unrelated native hits survive;no stale remap image;different offset and atomic metadata-failure retry passed\n";
}
void native_snapshot_mapping_versions(context& device)
{
	for (bool unmap : {false, true})
	{
		fixture f(device);
		const auto source = f.target(); f.dirty(source);
		const auto words = texture_words(4096);
		const auto old = f.textures.resolve_versioned(words, f.surfaces);
		const auto copies = f.surfaces.snapshot_statistics().created;
		require(f.textures.resolve_versioned(words, f.surfaces).image.texture == old.image.texture &&
			f.surfaces.snapshot_statistics().created == copies, "Native remap fixture did not cache its original source");
		f.change(unmap);
		bool current = false;
		try { current = old.revision.current(); } catch (const std::exception&) {} // An absent VM mapping also refuses validation.
		require(!current, "Native snapshot source ticket survived a real mapping change");
		require(f.service.service({4096, 64}, mapping_notifications::kind::unmap) == 1,
			"Mapping transition did not retire/publish the old native authority before replacement writes");
		if (unmap)
			refuses([&] { (void)f.textures.resolve_versioned(words, f.surfaces); }, "Unmapped VA returned an old cached native snapshot");
		else
		{
			f.memory->write({4096, 64}, std::vector<std::byte>(64, std::byte{0x79}));
			const auto fresh = f.textures.resolve_versioned(words, f.surfaces);
			require(fresh.image.texture != old.image.texture && fresh.revision.current(), "VA replacement reused the old native snapshot identity");
			verify_image(device, fresh.image.texture, 0x79);
			f.change(false);
			require(!fresh.revision.current(), "Repeated mapping epoch accepted an earlier immutable source ticket");
		}
		const auto output = make_buffer(device, 64, buffer_storage::shared);
		device.begin(); texture_to_buffer(device, old.image.texture,
			{0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, output, 0); device.submit().wait();
		const std::array<std::uint8_t, 4> expected{0x56, 0x34, 0x12, 0xff};
		for (unsigned i = 0; i < 64; ++i) require(static_cast<const std::uint8_t*>(output.contents)[i] == expected[i % 4], "Cached original native pixels changed across remap/unmap");
	}
	std::cout << "Native cached snapshot remap/unmap:real guarded mapping versions reject old tickets andunmapped lookups;replacement VM bytes andretained original64 BGRA bytes remain exact passed\n";
}
void disjoint_and_ticket_metadata(context& device)
{
	fixture f(device);
	cached_surface_description description{4096, {4, 4, 1, surface_color::argb8, false}, 0, {}};
	description.segments = {{4096, 0, 17}, {65536, 17, 47}};
	const auto target = f.surfaces.acquire(description); f.dirty(target); f.change(true);
	require(f.service.service({65536, 47}, mapping_notifications::kind::read) == 1, "Disjoint tail notification missed original active output");
	auto bytes = f.memory->read({0x41000, 17}).bytes;
	const auto tail = f.memory->read({65536, 47}).bytes; bytes.insert(bytes.end(), tail.begin(), tail.end());
	for (unsigned i = 0; i < 64; ++i)
		require(bytes[i] == std::array{std::byte{255}, std::byte{0x12}, std::byte{0x34}, std::byte{0x56}}[i % 4], "Disjoint service lost byte-split native pixels");
	fixture metadata(device), foreign(device);
	const auto layout = metadata.memory->watch_surface(plan_guest_surface(4096, 4, 2, 4, 24));
	const auto revision = metadata.memory->surface_revision(layout);
	const auto ticket = metadata.memory->capture_physical_surface(layout, revision);
	require(ticket && metadata.memory->physical_surface_virtual_current(*ticket, revision) == true, "Metadata-only current watch validation failed");
	metadata.change(true);
	require(metadata.memory->physical_surface_virtual_current(*ticket, revision) == false, "Metadata validation mapped/blessed absent old VA");
	require(metadata.memory->physical_surface_overlaps_current(*ticket, {0x41000, 16}) == true &&
		metadata.memory->physical_surface_overlaps_current(*ticket, {0x41010, 8}) == false, "Physical metadata selection includes padding or misses active old alias");
	refuses([&] { (void)foreign.memory->physical_surface_virtual_current(*ticket, revision); }, "Foreign virtual watch domain accepted");
	refuses([&] { (void)foreign.memory->physical_surface_overlaps_current(*ticket, {4096, 16}); }, "Foreign physical selection domain accepted");
	std::cout << "17-byte disjoint active parts;metadata-only absent VA;physical padding;foreign ticket domains\n";
}
void owner_and_mailbox(context& device)
{
	fixture f(device); const auto target = f.target(); f.dirty(target); f.change(true);
	const auto captures = f.provider->captures;
	bool foreign_refused = false;
	std::thread foreign([&] { try { (void)f.service.service({4096, 64}, mapping_notifications::kind::unmap); } catch (const std::logic_error&) { foreign_refused = true; } });
	foreign.join(); require(foreign_refused && f.provider->captures == captures, "Foreign mapping service entered provider/GPU");
	f.provider->execute([&](auto&) { refuses([&] { (void)f.service.service({4096, 64}, mapping_notifications::kind::unmap); }, "Mapping service ran under VM exclusion"); });
	refuses([&] { (void)f.service.service({0xffffffffu, 2}, mapping_notifications::kind::read); }, "Wrapped mapping service range accepted");
	refuses([&] { (void)f.service.service({4096, 64}, static_cast<mapping_notifications::kind>(9)); }, "Invalid mapping service cause accepted");
	mapping_notifications mailbox; mailbox.bind_owner(); mailbox.notify({4096, 64}, mapping_notifications::kind::unmap);
	unsigned observers = 0;
	refuses([&] { mailbox.drain([&](auto range, auto cause) { (void)f.service.service(range, cause); ++observers; throw std::runtime_error("Platform observer failure after native commit"); }); }, "Mailbox swallowed observer failure");
	require(mailbox.pending() && !f.surfaces.status(target).dirty, "Mailbox failure lost pending observer or reverted native commit");
	const auto reservations = f.provider->reservations;
	require(mailbox.drain([&](auto range, auto cause) { require(f.service.service(range, cause) == 0, "Observer retry republished physical output"); ++observers; }) == 1,
		"Mailbox did not retry failed observer");
	require(!mailbox.pending() && observers == 2 && f.provider->reservations == reservations, "Observer retry repeated physical reservations");
	std::cout << "Foreign-owner/VM-epoch/range/cause guards;actual native mailbox commit then observer failure/retry once\n";
}
void physical_read_races(context& device)
{
	unsigned profiles = 0;
	for (bool deferred : {false, true}) for (unsigned mode = 0; mode < 6; ++mode)
	{
		fixture f(device); const auto target = f.target(); f.dirty(target);
		if (deferred)
		{
			const auto source = f.target(98304); f.dirty(source);
			f.surfaces.defer_copy(source, {0, 0, 4, 4}, target, 0, 0); f.surfaces.erase(source);
		}
		const auto gate = [device.device() newSharedEvent];
		struct release_gate { id<MTLSharedEvent> gate; ~release_gate() { gate.signaledValue = 1; } } release{gate};
		const auto error = make_buffer(device, 4, buffer_storage::shared);
		*static_cast<std::uint32_t*>(error.contents) = 1;
		unsigned recordings = 0; bool injected = false;
		f.outside_capture = [&]
		{
			if (injected) return;
			try { (void)device.recording_submission(); } catch (const std::logic_error&) { return; }
			if (++recordings != (deferred ? 2u : 1u)) return; // Final guest download,after any scratch materialization.
			injected = true;
			if (mode == 0) f.change(false); // Remap after encoding,before physical validation/submit.
			else if (mode == 5) device.require_gpu_success(error, 0, @"Intentional physical readback GPU status failure");
			else { device.wait_event(gate, 1); f.surfaces.set_wait_timeout(std::chrono::milliseconds(1)); }
		};
		bool failed = false;
		try { (void)f.service.service({4096, 64}, mapping_notifications::kind::read); } catch (const std::runtime_error&) { failed = true; }
		require(injected && failed == (mode != 0), "Read service did not reach intended physical GPU boundary");
		if (mode >= 1 && mode <= 4)
		{
			require(f.surfaces.status(target).pending && f.surfaces.status(target).dirty && !f.surfaces.status(target).faulted,
				"Physical readback timeout discarded/faulted native output");
			if (mode == 1 || mode == 2) f.change(mode == 2);
			if (mode == 3) f.provider->execute([&](auto&) { f.provider->protect_first(false); });
			if (mode == 4) f.memory->write({0x41000, 64}, std::vector<std::byte>(64, std::byte{0x7a}));
			gate.signaledValue = 1; f.surfaces.set_wait_timeout(std::chrono::seconds(30));
			failed = false; try { f.surfaces.synchronize(); } catch (const surface_conflict_error&) { failed = true; }
			require(failed == (mode == 4), "Physical read completion did not enforce old CPU revision");
			if (mode == 3) f.provider->execute([&](auto&) { f.provider->protect_first(true); });
		}
		if (mode == 5)
		{
			require(!f.surfaces.status(target).pending && f.surfaces.status(target).dirty && !f.surfaces.status(target).faulted,
				"Read-only GPU status failure faulted/discarded original native output");
			device.recover_completed_errors();
			require(f.service.service({4096, 64}, mapping_notifications::kind::read) == 1, "Physical readback status failure could not retry");
		}
		const auto state = f.surfaces.status(target);
		if (mode == 4)
		{
			require(state.dirty && !state.faulted && f.memory->read({0x41000, 64}).bytes == std::vector<std::byte>(64, std::byte{0x7a}),
				"Failed physical read CAS overwrote CPU data or lost native authority");
		}
		else
		{
			require(!state.dirty && !state.faulted && state.initialized == (mode == 3 || mode == 5), "Read physical commit published wrong native validity");
			const auto old = f.memory->read({0x41000, 64}).bytes;
			for (unsigned i = 0; i < 64; ++i)
				require(old[i] == std::array{std::byte{255}, std::byte{0x12}, std::byte{0x34}, std::byte{0x56}}[i % 4], "Read race lost old physical GPU output");
			if (mode == 0 || mode == 1) require(f.memory->read({4096, 64}).bytes == std::vector<std::byte>(64), "Read race touched replacement CPU allocation");
			if (state.initialized)
				require(state.guest_revision == f.memory->surface_revision(plan_guest_surface(4096, 4, 4, 4)), "Read receipt did not install post-store virtual revision");
		}
		++profiles;
	}
	std::cout << "Physical read races:" << profiles << " completed/deferred profiles;remap after encode;GPUgate timeout/remap/unmap/protected;newer CPU CAS refusal;read-only GPU status failure/retry;post-store watch receipt;replacement bytes preserved\n";
}
void normal_native_completion(context& device)
{
	unsigned profiles = 0;
	for (unsigned operation = 0; operation < 7; ++operation) for (unsigned mode = 0; mode < 5; ++mode)
	{
		fixture f(device); auto tiles = std::make_shared<tile_state>();
		cached_surface_description description{4096, {4, 4, 1, surface_color::argb8, false}, 0, {}};
		description.mapping = capture_surface_mapping(plan_guest_surface(4096, 4, 4, 4), 4096, 1,
			[](unsigned address, unsigned, std::size_t) { return address; }, f.io, tiles, false, 65536);
		description.segments = description.mapping->state().segments;
		const auto target = f.surfaces.acquire(description), source = f.target(98304), other = f.target(69632);
		f.surfaces.clear(source, 0xf0, 0xff89abcd, 0, 255, {0, 0, 4, 4});
		if (operation != 6) f.dirty(target);
		if (operation == 2) f.dirty(other);
		if (operation == 5) f.surfaces.defer_copy(source, {0, 0, 4, 4}, target, 0, 0);
		const auto before = f.surfaces.status(target).native_write_epoch;
		const auto gate = [device.device() newSharedEvent];
		struct release_gate { id<MTLSharedEvent> gate; ~release_gate() { gate.signaledValue = 1; } } release{gate};
		bool injected = false;
		f.outside_capture = [&]
		{
			if (injected) return;
			try { (void)device.recording_submission(); } catch (const std::logic_error&) { return; }
			injected = true; device.wait_event(gate, 1); f.surfaces.set_wait_timeout(std::chrono::milliseconds(1));
		};
		refuses([&]
		{
			switch (operation)
			{
			case 0: f.surfaces.clear(target, 0xf0, 0xff89abcd, 0, 255, {0, 0, 4, 4}); break;
			case 1: f.surfaces.clear(target, 0xf0, 0xff89abcd, 0, 255, {1, 1, 2, 2}); break;
			case 2:
			{
				const std::array uses{cached_surface_use{target, true}, cached_surface_use{other, true}, cached_surface_use{source, false}};
				f.surfaces.draw(uses, [&](context& commands, auto textures)
				{
					texture_region whole; whole.size = MTLSizeMake(4, 4, 1);
					copy_texture(commands, textures[2], whole, textures[0], whole); copy_texture(commands, textures[2], whole, textures[1], whole);
				}); break;
			}
			case 3: f.surfaces.copy(source, {0, 0, 4, 4}, target, 0, 0); break;
			case 4: f.surfaces.blit_typeless(source, {0, 0, 4, 4}, target, {0, 0, 4, 4}); break;
			case 5: f.surfaces.prepare(target); break;
			case 6: f.surfaces.prepare(target, surface_initialization::background); break;
			}
		}, "Native output gate did not suspend completion");
		require(injected && f.surfaces.status(target).pending && !f.surfaces.status(target).faulted, "Native output timeout did not preserve pending GPU work");
		if (mode < 2) f.change(mode == 1);
		if (mode == 0) f.memory->write({4096, 64}, std::vector<std::byte>(64, std::byte{0x6d}));
		if (mode == 2) f.provider->execute([&](auto&)
		{
			f.memory->begin_mapping_change({0, 65536}); f.provider->protect_first(false); f.memory->end_mapping_change({0, 65536});
		});
		if (mode == 3) f.io->notify({0, 1 << 20});
		if (mode == 4) tiles->update(0, tile_descriptor{1, 0, 65536, 256, 0, 0, 0, true});
		gate.signaledValue = 1; f.surfaces.set_wait_timeout(std::chrono::seconds(30)); f.surfaces.synchronize();
		const auto completed = f.surfaces.status(target);
		require(completed.dirty && completed.initialized && !completed.faulted && !completed.pending && !completed.deferred_copies && completed.native_write_epoch > before,
			"Successful native GPU output was lost/faulted by a later virtual mapping change");
		if (operation == 2) require(f.surfaces.status(other).native_write_epoch == completed.native_write_epoch && f.surfaces.status(other).dirty,
			"MRT native completion published only one physical output");
		f.surfaces.flush(target); // Ordinary writeback must also stay physical.
		const auto published = f.surfaces.status(target);
		require(!published.dirty && !published.initialized && !published.faulted, "Detached ordinary writeback retained invalid virtual native mirror");
		const auto bytes = f.memory->read({0x41000, 64}).bytes;
		for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) for (unsigned c = 0; c < 4; ++c)
		{
			const bool old = operation == 1 && (x < 1 || x > 2 || y < 1 || y > 2);
			const auto expected = operation == 6 ? std::array{255u, 0u, 0u, 0u} : old ? std::array{255u, 0x12u, 0x34u, 0x56u} : std::array{255u, 0x89u, 0xabu, 0xcdu};
			require(bytes[(y * 4 + x) * 4 + c] == std::byte(expected[c]), "Physical completion/ordinary writeback lost actual native pixels");
		}
		if (mode == 0) require(f.memory->read({4096, 64}).bytes == std::vector<std::byte>(64, std::byte{0x6d}), "Ordinary writeback touched replacement CPU bytes");
		if (mode == 2) f.provider->execute([&](auto&) { f.provider->protect_first(true); });
		if (operation == 2)
		{
			f.surfaces.flush(other);
			require(f.surfaces.status(other).initialized && !f.surfaces.status(other).dirty && !f.surfaces.status(other).faulted,
				"Current MRT output lost its exact virtual receipt");
		}
		++profiles;
	}
	std::cout << "Normal native completion:" << profiles << " full/partial clear,MRT draw,copy,typeless,deferred,background GPUgate profiles;remap/unmap/protect/IO/tile epochs;physical ordinary writeback;replacement bytes preserved\n";
}
[[noreturn]] void reject_cpu_dispatch() { throw std::runtime_error("Intentional pre-entry CPU dispatcher refusal"); }
void native_completion_dispatch_retry(context& device)
{
	for (bool fail : {false, true})
	{
		fixture f(device); const auto target = f.target(); f.dirty(target);
		const auto before = f.surfaces.status(target);
		const auto gate = [device.device() newSharedEvent];
		struct release_gate { id<MTLSharedEvent> gate; ~release_gate() { gate.signaledValue = 1; } } release{gate};
		const auto error = make_buffer(device, 4, buffer_storage::shared); *static_cast<unsigned*>(error.contents) = 1;
		f.surfaces.set_wait_timeout(std::chrono::milliseconds(1));
		const std::array write{cached_surface_use{target, true}};
		refuses([&] { f.surfaces.draw(write, [&](context& commands, auto)
			{ commands.wait_event(gate, 1); if (fail) commands.require_gpu_success(error, 0, @"Intentional dispatcher-retry GPU error"); }); }, "Dispatcher-retry fixture did not gate native work");
		gate.signaledValue = 1;
		f.surfaces.set_wait_timeout(std::chrono::seconds(30));
		f.outside_capture = reject_cpu_dispatch;
		refuses([&] { f.surfaces.synchronize(); }, "CPU dispatcher refusal was ignored");
		const auto held = f.surfaces.status(target);
		require(held.pending && held.initialized && held.dirty && !held.faulted && held.content_generation == before.content_generation &&
			held.native_write_epoch == before.native_write_epoch, "Pre-entry dispatcher refusal lost pending completion/rollback ownership");
		f.outside_capture = {};
		if (fail)
		{
			refuses([&] { f.surfaces.synchronize(); }, "Failed GPU result became success after dispatcher retry");
			require(f.surfaces.status(target).initialized && !f.surfaces.status(target).spilled && !f.surfaces.status(target).faulted, "Dispatcher retry lost retained native rollback");
			f.surfaces.recover(target);
		}
		else f.surfaces.synchronize();
		f.surfaces.flush(target);
		const auto bytes = f.memory->read({4096, 64}).bytes;
		for (unsigned i = 0; i < bytes.size(); ++i)
			require(bytes[i] == std::array{std::byte{0xff}, std::byte{0x12}, std::byte{0x34}, std::byte{0x56}}[i % 4], "Dispatcher retry damaged prior native contents");
	}
	std::cout << "Native completion dispatcher retry:2 successful/failed GPU profiles;pre-entry rejection retains pending transaction;retry publishes success or exact old rollback\n";
}

void native_rollback_precision(context& device)
{
	unsigned profiles = 0;
	for (bool depth : {false, true}) for (unsigned count : {1u, 2u, 4u}) for (bool disjoint : {false, true})
	{
		fixture f(device);
		cached_surface_description description{4096, {4, 2, count, surface_color::rgba32f, false}, 0, {}};
		if (depth) description.allocation.format = surface_depth::z24s8_uint;
		const auto size = count == 1 ? std::array<std::size_t, 2>{4, 2} : sample_expanded_size(4, 2, count);
		const auto pixel = depth ? 4u : 16u;
		const auto bytes = size[0] * size[1] * pixel;
		if (disjoint) description.segments = {{4096, 0, 7}, {65536, 7, bytes - 7}};
		const auto target = f.surfaces.acquire(description);
		f.surfaces.prepare(target);
		const auto format = depth ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatRGBA32Float;
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:size[0] height:size[1] mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | (depth ? MTLTextureUsageUnknown : MTLTextureUsageShaderWrite);
		const auto expanded = make_texture(device, td);
		std::vector<float> values(bytes / 4), replacement(values.size(), 0.75f);
		for (unsigned i = 0; i < values.size(); ++i) values[i] = std::array{0.123456f, 0.50001f, 0.314159f, 0.876543f}[i % 4];
		const auto input = make_upload(device, std::as_bytes(std::span(values))), next = make_upload(device, std::as_bytes(std::span(replacement)));
		const std::vector<std::byte> stencils(size[0] * size[1], std::byte{0x5b});
		const auto stencil = make_upload(device, stencils), output = make_buffer(device, bytes, buffer_storage::shared), stencil_output = make_buffer(device, stencils.size(), buffer_storage::shared);
		const auto error = make_buffer(device, 4, buffer_storage::shared); *static_cast<unsigned*>(error.contents) = 1;
		const std::array write{cached_surface_use{target, true}}, read{cached_surface_use{target, false}};
		sample_transfer samples(device);
		texture_region region; region.size = MTLSizeMake(size[0], size[1], 1);
		const auto encode = [&](context& commands, auto textures, id<MTLBuffer> buffer)
		{
			const auto texture = count > 1 ? expanded : textures[0];
			buffer_to_texture(commands, buffer, 0, texture, region, size[0] * pixel, 0, depth ? texture_plane::depth : texture_plane::all);
			if (depth) buffer_to_texture(commands, stencil, 0, texture, region, size[0], 0, texture_plane::stencil);
			if (count > 1) samples.import(commands, expanded, textures[0]);
		};
		f.surfaces.draw(write, [&](context& commands, auto textures) { encode(commands, textures, input); });
		const auto before = f.surfaces.status(target);
		const auto before_transaction = f.surfaces.transaction_statistics();
		refuses([&] { f.surfaces.draw(write, [&](context& commands, auto textures)
			{ encode(commands, textures, next); commands.require_gpu_success(error, 0, @"Intentional precision rollback failure"); }); }, "Failed native precision write was accepted");
		const auto rejected = f.surfaces.status(target);
		require(f.surfaces.transaction_statistics().native_versions == before_transaction.native_versions + 1 &&
			f.surfaces.transaction_statistics().raw_backup_submissions == before_transaction.raw_backup_submissions,
			"Precision rollback did not use independent native target version");
		require(!rejected.spilled && rejected.dirty && rejected.initialized && !rejected.faulted && !rejected.pending &&
			rejected.content_generation == before.content_generation && rejected.native_write_epoch == before.native_write_epoch && rejected.guest_revisions == before.guest_revisions,
			"Rollback lost prior native metadata or accepted failed generation");
		f.surfaces.recover(target);
		f.surfaces.draw(read, [&](context& commands, auto textures)
		{
			if (count > 1) samples.expand(commands, textures[0], expanded);
			const auto texture = count > 1 ? expanded : textures[0];
			texture_to_buffer(commands, texture, region, output, 0, size[0] * pixel, 0, depth ? texture_plane::depth : texture_plane::all);
			if (depth) texture_to_buffer(commands, texture, region, stencil_output, 0, size[0], 0, texture_plane::stencil);
		});
		require(std::memcmp(map_buffer(output, 0, bytes).data(), values.data(), bytes) == 0, "Rollback quantized/replaced old native float/sample bits");
		if (depth) require(std::memcmp(map_buffer(stencil_output, 0, stencils.size()).data(), stencils.data(), stencils.size()) == 0, "Rollback lost old native stencil bits");
		++profiles;
	}
	std::cout << "Native rollback precision:" << profiles << " RGBA32F/Depth32F+Stencil8 finite native-version profiles;1/2/4samples;contiguous/7-byte-disjoint;old bits,revision,generation,epoch retained;synthetic main GPUstatus failure\n";
}

void normal_native_refusals(context& device)
{
	unsigned profiles = 0;
	for (unsigned operation : {0u, 1u, 2u}) for (unsigned mode = 0; mode < 4; ++mode)
	{
		fixture f(device); const auto target = f.target(), source = f.target(98304);
		if (operation != 2) f.dirty(target); // Mode2 is VM upload,not native output.
		f.dirty(source);
		const auto before = f.surfaces.status(target).native_write_epoch;
		const auto gate = [device.device() newSharedEvent];
		struct release_gate { id<MTLSharedEvent> gate; ~release_gate() { gate.signaledValue = 1; } } release{gate};
		const auto error = make_buffer(device, 4, buffer_storage::shared); *static_cast<unsigned*>(error.contents) = 1;
		bool injected = false;
		f.outside_capture = [&]
		{
			if (injected) return;
			try { (void)device.recording_submission(); } catch (const std::logic_error&) { return; }
			injected = true;
			if (mode == 2) device.require_gpu_success(error, 0, @"Intentional normal native output status error");
			else if (mode == 3) f.change(false); // Before submission:must refuse even native output.
			else { device.wait_event(gate, 1); f.surfaces.set_wait_timeout(std::chrono::milliseconds(1)); }
		};
		refuses([&]
		{
			if (operation == 0) f.surfaces.clear(target, 0xf0, 0xff89abcd, 0, 255, {0, 0, 4, 4});
			if (operation == 1) f.surfaces.copy(source, {0, 0, 4, 4}, target, 0, 0);
			if (operation == 2) f.surfaces.prepare(target);
		}, "Unsafe native/upload transaction was accepted");
		require(injected, "Native/upload refusal missed intended GPU boundary");
		if (mode < 2)
		{
			require(f.surfaces.status(target).pending, "Native/upload refusal omitted pending GPU gate");
			if (operation == 2) f.change(false);
			else
			{
				const auto address = mode == 1 && operation == 1 ? 98304u : 0x41000u;
				f.memory->write({address, 64}, std::vector<std::byte>(64, std::byte{0x7a}));
			}
			gate.signaledValue = 1; f.surfaces.set_wait_timeout(std::chrono::seconds(30));
			refuses([&] { f.surfaces.synchronize(); }, "Stale physical input/VM upload was committed after completion");
		}
		const auto state = f.surfaces.status(target);
		require(!state.pending && state.faulted == (operation == 2 && mode != 3) && state.native_write_epoch == before,
			"Invalid native/upload input or GPU status falsely published successful metadata");
		if (operation != 2 && mode != 3)
			require(!state.spilled && state.initialized && state.dirty, "Rejected native write lost its prior native version");
		if (mode == 2)
		{
			device.recover_completed_errors();
			if (operation != 2)
			{
				f.surfaces.flush(target);
				const auto bytes = f.memory->read({4096, 64}).bytes;
				for (unsigned i = 0; i < bytes.size(); ++i)
					require(bytes[i] == std::array{std::byte{0xff}, std::byte{0x12}, std::byte{0x34}, std::byte{0x56}}[i % 4], "Native rollback published the failed output instead of prior native bits");
			}
		}
		if (operation != 2 && mode < 2)
		{
			const auto address = mode == 1 && operation == 1 ? 98304u : 0x41000u;
			require(f.memory->read({address, 64}).bytes == std::vector<std::byte>(64, std::byte{0x7a}), "Failed native output overwrote newer CPU data");
		}
		++profiles;
	}
	std::cout << "Normal native refusals:" << profiles << " stale destination/source,VM upload remap,synthetic GPUstatus,pre-submit remap profiles;no false metadata/CPU overwrite\n";
}

}
int main()
{
	@autoreleasepool
	{
		try
		{
			context device{1}; old_owner_matrix(device); selection_and_retry(device); read_precision_and_texture_eviction(device); texture_mapping_selection(device); native_snapshot_mapping_versions(device); disjoint_and_ticket_metadata(device); owner_and_mailbox(device); physical_read_races(device); normal_native_completion(device); normal_native_refusals(device); native_rollback_precision(device); native_completion_dispatch_retry(device);
			std::cout << "Native mapping invalidation service passed;no RPCS3 application/UI/game/guest VM started\n";
			return 0;
		}
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
