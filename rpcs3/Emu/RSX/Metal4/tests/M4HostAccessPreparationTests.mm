#include "M4HostAccessPreparation.h"
#include "M4HostStoreTransaction.h"
#include "M4HostAccessFixture.h"
#include "M4MappingNotifications.h"
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
auto mapping_access(const std::shared_ptr<test::host_access_fixture>& provider)
{
	return [provider]() -> std::shared_ptr<void>
	{
		auto guard = std::make_shared<std::unique_lock<std::mutex>>(provider->mapping_mutex, std::try_to_lock);
		if (!guard->owns_lock()) return {};
		return guard;
	};
}
void gpu_flush_before_store()
{
	context device{1};
	auto provider = std::make_shared<test::host_access_fixture>();
	provider->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(provider);
	const cpu_capture_dispatch capture = [&](const auto& operation) { provider->execute([&](auto&) { operation(); }); };
	surface_cache surfaces(device, memory, 8 << 20, {}, capture);
	const auto input = surfaces.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	const auto target = surfaces.acquire({65536, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	surfaces.clear(input, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 4});
	surfaces.clear(target, 0xf0, 0xffaabbcc, 0, 255, {0, 0, 4, 4});
	require(surfaces.status(input).dirty && surfaces.status(target).dirty && provider->bytes[4097] == std::byte{0},
		"Preparation fixture has no deferred GPU-authoritative bytes");
	const auto watched = memory->watch_surface(guest_byte_coverage({65536, 64}));
	const auto mirror = memory->watch_surface(guest_byte_coverage({0x50000, 64}));
	const auto old = memory->surface_revision(watched), old_mirror = memory->surface_revision(mirror);
	{
		auto ticket = provider->gate->admit({provider->bytes + 4096, provider->bytes + 65552, 16});
		auto writer = ticket.activate();
		require(bool(writer), "GPU host transfer lost its writer ticket");
		prepared_host_transfer prepared;
		{
			auto paused = provider->gate->pause_current_fault();
			require(bool(paused), "GPU host transfer failed to pause");
			auto owner_capture = paused->capture_context();
			require(bool(owner_capture), "GPU host transfer lost capture ownership");
			prepared = prepare_host_transfer(surfaces, *memory, guest_surface_range{65552, 16}, guest_surface_range{4096, 16});
			require(!memory->pending_host_writes() && provider->bytes[4096] == std::byte{255} && provider->bytes[4097] == std::byte{0x12} &&
				provider->bytes[65537] == std::byte{0xaa} && provider->bytes[65553] == std::byte{0xaa},
				"GPU source/destination bytes were not flushed before host-store preparation");
			require(!surfaces.status(input).dirty && !surfaces.status(target).dirty && !surfaces.status(target).initialized,
				"Prepared native mirrors retained dirty/stale authority");
		}
		auto completion = host_store_transaction::prepare(memory, std::move(prepared.destination), std::move(prepared.source), mapping_access(provider));
		require(completion.start(), "Available GPU mapping exclusion was refused");
		std::memcpy(provider->bytes + 65552, provider->bytes + 4096, 8);
		{
			auto store_fault = host_store_transaction::pause_current(*memory);
			require(bool(store_fault), "GPU prefix fault lost the instruction's mapping guard");
			auto writer_fault = provider->gate->pause_current_fault();
			require(bool(writer_fault), "GPU prefix fault lost its host writer");
			{
				auto fault_capture = writer_fault->capture_context();
				require(bool(fault_capture), "GPU prefix fault lost owner capture");
				store_fault->service([&]
				{
					const auto prefix = surfaces.cpu_read({65552, 16});
					require(prefix.bytes[1] == std::byte{0x12} && prefix.bytes[9] == std::byte{0xaa} && memory->pending_host_writes() == 1,
						"GPU fault capture lost the copied prefix,old destination tail or live intent");
					surfaces.prepare(target); // Cache the prefix;final completion must invalidate it.
				});
				const auto queued = prepare_host_transfer(surfaces, *memory, guest_surface_range{70000, 16}, {});
				require(queued.destination && memory->pending_host_writes() == 1,
					"Another paused preparation lost or refused the outstanding prefix intent");
			}
			require(store_fault->resume(), "GPU prefix fault failed stable mapping retry");
		}
		std::memcpy(provider->bytes + 65560, provider->bytes + 4104, 8);
		completion.commit();
	}
	const auto result = memory->read_surface(watched);
	require(result.revision > old && memory->surface_revision(mirror) > old_mirror && !memory->pending_host_writes(),
		"GPU-prepared actual host store did not reconcile physical aliases");
	for (unsigned pixel = 0; pixel < 16; ++pixel)
	{
		const bool copied = pixel >= 4 && pixel < 8;
		require(result.bytes[pixel * 4] == std::byte{255} && result.bytes[pixel * 4 + 1] == (copied ? std::byte{0x12} : std::byte{0xaa}) &&
			result.bytes[pixel * 4 + 2] == (copied ? std::byte{0x34} : std::byte{0xbb}) &&
			result.bytes[pixel * 4 + 3] == (copied ? std::byte{0x56} : std::byte{0xcc}),
			"Host store lost GPU source data or untouched destination old contents");
	}
	surfaces.prepare(target);
	require(surfaces.status(target).initialized && !surfaces.status(target).dirty, "Native destination failed to reload the committed host bytes");
	const auto echo = surfaces.acquire({98304, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	surfaces.copy(target, {0, 0, 4, 4}, echo, 0, 0);
	surfaces.flush(echo);
	require(memory->read({98304, 64}).bytes == result.bytes, "Native GPU mirror retained a fault-prefix snapshot after final completion");
	surfaces.clear(input, 0xf0, 0xff456789, 0, 255, {0, 0, 4, 4});
	std::array<std::byte, 16> output{};
	std::uint64_t read_revision = 0;
	const auto source_watch = memory->watch_surface(guest_byte_coverage({4096, 64}));
	{
		auto ticket = provider->gate->admit({provider->bytes + 4096, output.data(), output.size()});
		auto writer = ticket.activate();
		require(bool(writer), "Source-only host transfer lost its ticket");
		prepared_host_transfer prepared;
		{
			auto paused = provider->gate->pause_current_fault();
			require(bool(paused), "Source-only host transfer failed to pause");
			auto owner_capture = paused->capture_context();
			require(bool(owner_capture), "Source-only host transfer lost capture ownership");
			prepared = prepare_host_transfer(surfaces, *memory, {}, guest_surface_range{4096, 16});
			read_revision = memory->surface_revision(source_watch);
		}
		auto lifetime = host_store_transaction::prepare(memory, {}, std::move(prepared.source), mapping_access(provider));
		require(lifetime.start(), "Available source mapping exclusion was refused");
		std::memcpy(output.data(), provider->bytes + 4096, output.size());
		lifetime.commit();
	}
	require(output[0] == std::byte{255} && output[1] == std::byte{0x45} && output[2] == std::byte{0x67} && output[3] == std::byte{0x89} &&
		!memory->pending_host_writes() && memory->surface_revision(source_watch) == read_revision,
		"Source-only host copy read stale GPU bytes or fabricated a guest-store revision");
	std::cout << "Native Metal4 host preparation:GPU-dirty source/destination flush before actual partial memcpy;VM exclusion;paused prefix recovery and stable retry;native GPU echo reloads final bytes over prefix snapshot;old contents and physical aliases;mirror invalidation/reload;source-only GPU read with no fabricated guest revision;independent physical leases;no App/UI/guest VM started\n";
}
void gpu_alias_after_completed_remap()
{
	context device{1};
	auto provider = std::make_shared<test::host_access_fixture>();
	provider->pin_first_alias();
	provider->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(provider);
	const cpu_capture_dispatch capture = [&](const auto& operation) { provider->execute([&](auto&) { operation(); }); };
	surface_cache surfaces(device, memory, 8 << 20, {}, capture);
	const auto alias = surfaces.acquire({0x41000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	surfaces.prepare(alias); // Keep a native copy of the original zero bytes.
	const auto watched = memory->watch_surface(guest_byte_coverage({0x41000, 64}));
	const auto old = memory->surface_revision(watched);
	auto destination = memory->prepare_host_access({4096, 64}, true);
	std::array<std::byte, 64> payload{};
	for (unsigned pixel = 0; pixel < 16; ++pixel)
	{
		payload[pixel * 4] = std::byte{255}; payload[pixel * 4 + 1] = std::byte{0x65};
		payload[pixel * 4 + 2] = std::byte{0x43}; payload[pixel * 4 + 3] = std::byte{0x21};
	}
	{
		auto ticket = provider->gate->admit(); auto writer = ticket.activate();
		auto store = host_store_transaction::prepare(memory, std::move(destination), {}, mapping_access(provider));
		require(store.start(), "Available completed-remap mapping guard was refused");
		std::memcpy(provider->bytes + 4096, payload.data(), payload.size());
		store.commit();
	}
	provider->execute([&](auto&)
	{
		memory->begin_mapping_change({0, 65536});
		provider->replace_first();
		memory->end_mapping_change({0, 65536});
	});
	const auto echo = surfaces.acquire({65536, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	surfaces.copy(alias, {0, 0, 4, 4}, echo, 0, 0);
	surfaces.flush(echo);
	const auto result = memory->read({65536, 64});
	require(std::equal(result.bytes.begin(), result.bytes.end(), payload.begin()),
		"Native GPU alias retained its pre-store snapshot after completed destination remap");
	require(memory->surface_revision(watched) > old && !memory->pending_host_writes(),
		"Completed remap did not revise the surviving old physical alias");
	require(memory->read({4096, 64}).bytes == std::vector<std::byte>(64),
		"Physical completed-store reconciliation replayed old bytes into replacement guest allocation");
	std::cout << "Native Metal4 physical reconciliation:cached GPU alias reloads completed old physical bytes after destination replacement;GPU echo equals host payload;replacement allocation stays zero;no App/UI/guest VM started\n";
}
void gpu_service_after_mapper_unlock()
{
	context device{1};
	auto provider = std::make_shared<test::host_access_fixture>();
	auto memory = std::make_shared<guarded_guest_memory>(provider);
	const cpu_capture_dispatch capture = [&](const auto& operation) { provider->execute([&](auto&) { operation(); }); };
	surface_cache surfaces(device, memory, 8 << 20, {}, capture);
	const auto target = surfaces.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	surfaces.clear(target, 0xf0, 0xff345678, 0, 255, {0, 0, 4, 4});
	require(surfaces.status(target).dirty && provider->bytes[4097] == std::byte{0}, "Mailbox GPU fixture has no deferred native dirty bytes");
	auto notifications = std::make_unique<mapping_notifications>(); notifications->bind_owner();
	bool captured_callback = false;
	provider->execute([&](auto&)
	{
		notifications->notify({4096, 64}, mapping_notifications::kind::read);
		notifications->drain([&](auto, auto) { captured_callback = true; });
	});
	require(!captured_callback && notifications->pending(), "Native GPU notification service ran under captured VM epoch");
	std::atomic<bool> notified{false}, release_mapping{false};
	std::thread mapper([&]
	{
		std::lock_guard lock(provider->mapping_mutex);
		notifications->notify({4096, 64}, mapping_notifications::kind::read);
		notified = true;
		while (!release_mapping.load()) std::this_thread::yield();
	});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!notified.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
	const bool nonwaiting = notified.load();
	std::exception_ptr failure;
	bool serviced = false;
	try
	{
		if (nonwaiting) notifications->drain([&](auto range, auto cause)
		{
			require(range.address == 4096 && range.size == 4096 && cause == mapping_notifications::kind::read,
				"Native GPU mapping service received incorrect page/kind");
			release_mapping = true; // Mapper finishes before provider capture;GPU waits stay outside its lock.
			surfaces.flush(target);
			serviced = true;
		});
	}
	catch (...) { failure = std::current_exception(); }
	release_mapping = true; mapper.join();
	if (failure) std::rethrow_exception(failure);
	require(nonwaiting && serviced && !notifications->pending() && !surfaces.status(target).dirty &&
		provider->bytes[4096] == std::byte{255} && provider->bytes[4097] == std::byte{0x34} && provider->bytes[4099] == std::byte{0x78},
		"VM-held mapping notification blocked on native GPU service or failed owner writeback after unlock");
	std::cout << "Native Metal4 mapping mailbox:captured owner defers service;mapper notifies under model VM lock without waiting;bound owner flushes GPU-dirty bytes after mapper unlock;exact guest payload,notification retired;no App/UI/guest VM started\n";
}
void gpu_output_to_retained_physical_owner()
{
	context device{1}; clear_engine clear(device); surface_color_transfer codec(device);
	const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	const auto texture = make_texture(device, descriptor);
	const auto output = make_buffer(device, 64, buffer_storage::shared);
	clear_request request{}; request.colors[0] = {{texture, 0, 0}, surface_color::argb8};
	request.mask = 0xf0; request.color_value = 0xff13579b; request.rectangle = {0, 0, 4, 4};
	for (bool unmap : {false, true})
	{
		auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
		guarded_guest_memory memory(provider);
		const auto layout = memory.watch_surface(plan_guest_surface(4096, 4, 4, 4, 16));
		const auto ticket = memory.capture_physical_surface(layout, memory.surface_revision(layout));

		device.begin();
		require(clear.run(device, request).color_targets == 1, "Physical GPU output clear did not write its native target");
		texture_region region{}; region.size = MTLSizeMake(4, 4, 1);
		codec.download(device, surface_color::argb8, texture, region, output, {0, 16});
		device.end_encoding(); device.submit().wait();
		const auto payload = map_buffer(output, 0, 64);
		for (unsigned pixel = 0; pixel < 16; ++pixel)
			require(payload[pixel * 4] == std::byte{255} && payload[pixel * 4 + 1] == std::byte{0x13} &&
				payload[pixel * 4 + 2] == std::byte{0x57} && payload[pixel * 4 + 3] == std::byte{0x9b}, "Physical GPU output codec returned incorrect guest bytes");

		provider->execute([&](auto&)
		{
			memory.begin_mapping_change({0, 65536});
			if (unmap) provider->unmap_first(); else provider->replace_first();
			memory.end_mapping_change({0, 65536});
		});
		const std::vector<std::byte> replacement(64, std::byte{0x5c});
		if (!unmap) memory.write({4096, 64}, replacement);
		const physical_surface_write write{ticket, payload};
		require(memory.compare_write_physical({&write, 1}).has_value() && memory.read({0x41000, 64}).bytes == std::vector<std::byte>(payload.begin(), payload.end()),
			"Native GPU output failed retained physical publication after mapping replacement");
		if (!unmap) require(memory.read({4096, 64}).bytes == replacement, "Native GPU output overwrote newer replacement CPU bytes");
		const auto alias = memory.watch_surface(plan_guest_surface(0x41000, 4, 4, 4, 16));
		const auto stale = memory.capture_physical_surface(alias, memory.surface_revision(alias));
		const std::vector<std::byte> cpu(64, std::byte{0x29}); memory.write({0x41000, 64}, cpu);
		const physical_surface_write old{stale, payload};
		require(!memory.compare_write_physical({&old, 1}) && memory.read({0x41000, 64}).bytes == cpu,
			"Native GPU output replaced newer CPU bytes in retained old allocation");
	}
	std::cout << "Native Metal4 physical publication:GPU clear/guest codec readback publishes only retained old owner after remap/unmap;replacement CPU bytes unchanged;newer old-alias CPU store rejects stale GPU payload;no App/UI/guest VM started\n";
}
void cache_physical_dirty_recovery()
{
	context device{1};
	for (unsigned profile = 0; profile < 10; ++profile)
	{
		auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
		auto memory = std::make_shared<guarded_guest_memory>(provider);
		const cpu_capture_dispatch capture = [&](const auto& operation) { provider->execute([&](auto&) { operation(); }); };
		surface_cache surfaces(device, memory, 8 << 20, {}, capture);
		cached_surface_description description{4096, {4, 4, profile == 4 ? 2u : profile == 5 ? 4u : 1u, surface_color::argb8, false}, profile == 7 ? 20u : 0u, {}};
		if (profile == 6) description.morton = swizzle_region{{8, 8, 1}, {2, 2, 0}, {4, 4, 1}};
		const auto expanded = description.allocation.samples == 1 ? std::array<std::size_t, 2>{4, 4} : sample_expanded_size(4, 4, description.allocation.samples);
		const auto layout = plan_guest_surface(4096, expanded[0], expanded[1], 4, description.row_pitch, description.morton);
		std::vector<std::byte> expected(layout.range.size, std::byte{0x71});
		if (profile == 3) description.segments = {{4096, 0, 17}, {65536, 17, layout.range.size - 17}};
		if (profile == 3)
		{
			memory->write({4096, 17}, std::span(expected).first(17));
			memory->write({65536, layout.range.size - 17}, std::span(expected).subspan(17));
		}
		else memory->write(layout.range, expected);
		const auto target = surfaces.acquire(description);
		const auto alias = surfaces.acquire({0x41000, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		surfaces.prepare(alias); // A clean old-allocation mirror must be retired.
		if (profile >= 8)
		{
			const auto source = surfaces.acquire({98304, {4, 4, 1, surface_color::argb8, false}, 0, {}});
			surfaces.clear(source, 0xf0, 0xff13579b, 0, 255, {0, 0, 4, 4});
			if (profile == 9) { surfaces.defer_copy(source, {0, 0, 4, 4}, target, 0, 0); surfaces.prepare(target); }
			else surfaces.copy(source, {0, 0, 4, 4}, target, 0, 0);
		}
		else surfaces.clear(target, 0xf0, 0xff13579b, 0, 255, profile == 7 ? clear_rectangle{1, 1, 2, 2} : clear_rectangle{0, 0, 4, 4});
		require(surfaces.status(target).dirty, "Physical cache fixture lost native dirty authority");
		if (profile == 2) surfaces.spill(target);
		guest_surface_byte_cursor active(layout);
		while (const auto range = active.next())
			for (std::size_t byte = 0; byte < range->size; ++byte)
			{
				const auto offset = range->address - layout.range.address + byte;
				if (profile == 7)
				{
					const auto y = offset / layout.row_pitch, x = offset % layout.row_pitch / 4;
					if (x < 1 || x >= 3 || y < 1 || y >= 3) continue;
				}
				constexpr std::array value{std::byte{255}, std::byte{0x13}, std::byte{0x57}, std::byte{0x9b}};
				expected[offset] = value[offset % 4];
			}
		provider->execute([&](auto&)
		{
			memory->begin_mapping_change({0, 65536});
			if (profile == 1) provider->unmap_first(); else provider->replace_first();
			memory->end_mapping_change({0, 65536});
		});
		const std::vector<std::byte> replacement(layout.range.size, std::byte{0x5c});
		if (profile != 1) memory->write(layout.range, replacement);
		surfaces.flush_physical(target);
		std::vector<std::byte> actual;
		if (profile == 3)
		{
			actual = memory->read({0x41000, 17}).bytes;
			const auto tail = memory->read({65536, layout.range.size - 17}); actual.insert(actual.end(), tail.bytes.begin(), tail.bytes.end());
		}
		else actual = memory->read({0x41000, layout.range.size}).bytes;
		require(actual == expected, "Native cache physical recovery lost old GPU payload,background,padding or split bytes");
		if (profile != 1) require(memory->read(layout.range).bytes == replacement, "Native cache physical recovery overwrote replacement CPU bytes");
		require(!surfaces.status(target).dirty && !surfaces.status(target).initialized && !surfaces.status(alias).initialized,
			"Physical cache publication retained stale native authority");
		surfaces.flush_physical(target); // Idempotent after successful publication.
	}
	for (bool protected_alias : {false, true})
	{
		auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
		auto memory = std::make_shared<guarded_guest_memory>(provider);
		const cpu_capture_dispatch capture = [&](const auto& operation) { provider->execute([&](auto&) { operation(); }); };
		surface_cache surfaces(device, memory, 8 << 20, {}, capture);
		const auto target = surfaces.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		surfaces.clear(target, 0xf0, 0xff13579b, 0, 255, {0, 0, 4, 4});
		if (protected_alias)
		{
			provider->protect_first(false); surfaces.flush_physical(target); provider->protect_first(true);
			require(memory->read({4096, 64}).bytes[1] == std::byte{0x13}, "Physical cache recovery failed protected current alias");
		}
		else
		{
			provider->execute([&](auto&) { memory->begin_mapping_change({0, 65536}); provider->replace_first(); memory->end_mapping_change({0, 65536}); });
			const std::vector<std::byte> cpu(64, std::byte{0x29}); memory->write({0x41000, 64}, cpu);
			bool refused = false; try { surfaces.flush_physical(target); } catch (const surface_conflict_error&) { refused = true; }
			require(refused && surfaces.status(target).dirty && memory->read({0x41000, 64}).bytes == cpu && memory->read({4096, 64}).bytes == std::vector<std::byte>(64),
				"Native cache accepted stale physical GPU data after newer old-alias CPU store");
		}
	}
	{
		auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
		auto memory = std::make_shared<guarded_guest_memory>(provider);
		const cpu_capture_dispatch capture = [&](const auto& operation) { provider->execute([&](auto&) { operation(); }); };
		surface_cache surfaces(device, memory, 8 << 20, {}, capture);
		const auto source = surfaces.acquire({98304, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		const auto target = surfaces.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		surfaces.clear(source, 0xf0, 0xff13579b, 0, 255, {0, 0, 4, 4});
		surfaces.defer_copy(source, {0, 0, 4, 4}, target, 0, 0);
		provider->execute([&](auto&) { memory->begin_mapping_change({0, 65536}); provider->replace_first(); memory->end_mapping_change({0, 65536}); });
		surfaces.flush_physical(target);
		require(surfaces.status(target).deferred_copies == 0 && !surfaces.status(target).dirty && memory->read({0x41000, 64}).bytes[1] == std::byte{0x13} && memory->read({4096, 64}).bytes == std::vector<std::byte>(64),
			"Physical cache recovery lost a deferred plan or touched replacement bytes");
		bool refused = false;
		surfaces.cancel_deferred_copies(target);
		// A prepublication reservation failure preserves the GPU output and
		// tickets for a second explicit attempt,without repeating guest stores.
		const auto retry = surfaces.acquire({8192, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		surfaces.clear(retry, 0xf0, 0xff2468ac, 0, 255, {0, 0, 4, 4});
		const auto kept = provider->retain_segments({8192, 64});
		provider->execute([&](auto&) { memory->begin_mapping_change({0, 65536}); provider->replace_first(); memory->end_mapping_change({0, 65536}); });
		provider->fail_reservation_planning = true;
		refused = false; try { surfaces.flush_physical(retry); } catch (const std::runtime_error&) { refused = true; }
		require(refused && surfaces.status(retry).dirty && !surfaces.status(retry).faulted && memory->read({8192, 64}).bytes == std::vector<std::byte>(64),
			"Physical cache failed publication consumed GPU authority or modified replacement bytes");
		provider->fail_reservation_planning = false; surfaces.flush_physical(retry);
		require(!surfaces.status(retry).dirty && kept->segments()[0].data()[1] == std::byte{0x24} && kept->segments()[0].data()[3] == std::byte{0xac},
			"Physical cache publication could not retry the old owner after reservation planning failure");
	}
	std::cout << "Native Metal4 cache physical recovery:pre-output typed tickets;clear/copy/materialized-deferred/spill;2x/4x MSAA;origin Morton;partial clear/padding;disjoint storage;old GPU bytes after remap/unmap;replacement CPU bytes preserved;protected aliases;newer old-alias CPU write refuses intact;clean mirrors invalidated;reservation failure retries;unmaterialized deferred publishes old owner;no App/UI/guest VM started\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try { gpu_flush_before_store(); gpu_alias_after_completed_remap(); gpu_service_after_mapper_unlock(); gpu_output_to_retained_physical_owner(); cache_physical_dirty_recovery(); }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
