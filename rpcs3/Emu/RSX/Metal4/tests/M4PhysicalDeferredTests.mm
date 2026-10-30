#include "M4SurfaceCache.h"
#include "M4HostAccessFixture.h"
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void remap(const std::shared_ptr<test::host_access_fixture>& provider, guarded_guest_memory& memory, bool unmap)
{
	provider->execute([&](auto&)
	{
		memory.begin_mapping_change({0, 65536});
		if (unmap) provider->unmap_first(); else provider->replace_first();
		memory.end_mapping_change({0, 65536});
	});
}
void deferred_matrix()
{
	context device{1};
	unsigned count = 0;
	for (bool unmap : {false, true})
		for (unsigned profile = 0; profile < 14; ++profile)
		{
			auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
			auto memory = std::make_shared<guarded_guest_memory>(provider);
			const cpu_capture_dispatch capture = [&](const auto& op) { provider->execute([&](auto&) { op(); }); };
			surface_cache cache(device, memory, 16 << 20, {}, capture);
			cached_surface_description description{4096, {4, 4, profile == 9 ? 2u : profile == 10 ? 4u : 1u, surface_color::argb8, false}, profile == 13 ? 19u : 0u, {}};
			if (profile == 11) description.morton = swizzle_region{{8, 8, 1}, {2, 2, 0}, {4, 4, 1}};
			const auto expanded = description.allocation.samples == 1 ? std::array<std::size_t, 2>{4, 4} : sample_expanded_size(4, 4, description.allocation.samples);
			const auto layout = plan_guest_surface(4096, expanded[0], expanded[1], 4, description.row_pitch, description.morton);
			if (profile == 12) description.segments = {{4096, 0, 17}, {65536, 17, layout.range.size - 17}};
			std::vector<std::byte> expected(layout.range.size, std::byte{0x71});
			if (profile == 12)
			{
				memory->write({4096, 17}, std::span(expected).first(17));
				memory->write({65536, layout.range.size - 17}, std::span(expected).subspan(17));
			}
			else memory->write(layout.range, expected);
			const auto target = cache.acquire(description);
			const auto source = cache.acquire({98304, {4, 4, 1, surface_color::argb8, false}, 0, {}});
			cache.clear(source, 0xf0, 0xff13579b, 0, 255, {0, 0, 4, 4});
			if (profile == 2 || profile == 3 || profile == 6 || profile == 9)
			{
				cache.clear(target, 0xf0, 0xff2468ac, 0, 255, {0, 0, 4, 4});
				guest_surface_byte_cursor active(layout);
				while (const auto range = active.next())
					for (std::size_t i = 0; i < range->size; ++i)
					{
						constexpr std::array bytes{std::byte{255}, std::byte{0x24}, std::byte{0x68}, std::byte{0xac}};
						const auto offset = range->address - layout.range.address + i; expected[offset] = bytes[i % 4];
					}
				if (profile == 3) cache.spill(target);
			}
			if (profile == 4 || profile == 5) cache.prepare(target);
			if (profile == 5)
			{
				std::fill(expected.begin(), expected.end(), std::byte{0x39}); memory->write(layout.range, expected);
			}
			if (profile == 6)
			{
				cache.clear(target, 0xf0, 0xffaabbcc, 0, 255, {0, 0, 2, 2});
				for (std::size_t y = 0; y < 4; ++y) for (std::size_t x = 0; x < 4; ++x)
					if ((x < 2 && y < 2) || (x >= 2 && y >= 2))
					{
						constexpr std::array bytes{std::byte{255}, std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc}};
						for (std::size_t b = 0; b < 4; ++b) expected[y * layout.row_pitch + x * 4 + b] = bytes[b];
					}
				cache.defer_copy(target, {0, 0, 2, 2}, target, 2, 2);
			}
			const bool full = profile == 0 || profile == 7 || profile == 8;
			const clear_rectangle rectangle = full ? clear_rectangle{0, 0, 4, 4} : clear_rectangle{1, 1, 2, 2};
			if (profile == 7)
			{
				cache.defer_copy(source, {0, 0, 1, 1}, target, 0, 0); // Superseded by the full copy below.
			}
			if (profile == 8) cache.defer_blit_typeless(source, rectangle, target, rectangle, {true, false, false});
			else cache.defer_copy(source, rectangle, target, rectangle.x, rectangle.y);
			const auto plans = cache.status(target).deferred_copies;
			require(plans != 0, "Physical deferred fixture materialized before mapping replacement");
			guest_surface_byte_cursor active(layout);
			while (const auto range = active.next())
				for (std::size_t i = 0; i < range->size; ++i)
				{
					const auto offset = range->address - layout.range.address + i;
					std::size_t x, y;
					if (layout.morton)
					{
						// Select exact active Morton bytes by the same public layout
						// cursor,then classify their pixel using the storage address.
						bool found = false; x = y = 0;
						for (std::size_t py = 0; py < 4; ++py) for (std::size_t px = 0; px < 4; ++px)
						{
							auto pixel = layout; pixel.width = pixel.height = 1; pixel.morton->origin[0] += px; pixel.morton->origin[1] += py; pixel.morton->size = {1, 1, 1};
							guest_surface_byte_cursor one(pixel); const auto word = one.next();
							if (offset >= word->address - layout.range.address && offset < word->address - layout.range.address + 4) { x = px; y = py; found = true; }
						}
						require(found, "Morton expected-pixel lookup failed");
					}
					else { x = offset % layout.row_pitch / 4; y = offset / layout.row_pitch; }
					const auto sx = description.allocation.samples > 1 ? x / (expanded[0] / 4) : x;
					const auto sy = description.allocation.samples > 1 ? y / (expanded[1] / 4) : y;
					if (full || (sx >= 1 && sx < 3 && sy >= 1 && sy < 3))
					{
						constexpr std::array bytes{std::byte{255}, std::byte{0x13}, std::byte{0x57}, std::byte{0x9b}};
						expected[offset] = bytes[i % 4];
					}
				}
			remap(provider, *memory, unmap);
			const std::vector<std::byte> replacement(layout.range.size, std::byte{0x5c});
			if (!unmap) memory->write(layout.range, replacement);
			cache.flush_physical(target);
			std::vector<std::byte> actual;
			if (profile == 12)
			{
				actual = memory->read({0x41000, 17}).bytes;
				const auto tail = memory->read({65536, layout.range.size - 17}); actual.insert(actual.end(), tail.bytes.begin(), tail.bytes.end());
			}
			else actual = memory->read({0x41000, layout.range.size}).bytes;
			require(actual == expected, "Physical deferred recovery lost captured source,old background,active bytes or padding");
			if (!unmap) require(memory->read(layout.range).bytes == replacement, "Physical deferred recovery wrote replacement CPU allocation");
			require(!cache.status(target).dirty && !cache.status(target).faulted && cache.status(target).deferred_copies == 0,
				"Physical deferred recovery retained native authority after publication");
			++count;
		}
	std::cout << "Native Metal4 physical deferred:" << count << " remap/unmap profiles;full/partial/ordered/self/typeless;VM/dirty-native/clean-native/stale-clean/raw-spill backgrounds;2x/4x MSAA;origin Morton;17-byte split;odd pitch/padding;replacement CPU bytes preserved;no App/UI/guest VM started\n";
}
void deferred_depth_aspects()
{
	context device{1};
	unsigned count = 0;
	for (bool unmap : {false, true})
		for (unsigned samples : {1u, 2u, 4u})
			for (auto plane : {texture_plane::depth, texture_plane::stencil})
				for (unsigned background = 0; background < 3; ++background)
				{
					auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
					auto memory = std::make_shared<guarded_guest_memory>(provider);
					const cpu_capture_dispatch capture = [&](const auto& op) { provider->execute([&](auto&) { op(); }); };
					surface_cache cache(device, memory, 16 << 20, {}, capture);
					const auto expanded = samples == 1 ? std::array<std::size_t, 2>{4, 4} : sample_expanded_size(4, 4, samples);
					const auto layout = plan_guest_surface(4096, expanded[0], expanded[1], 4, expanded[0] * 4 + 8);
					std::vector<std::byte> expected(layout.range.size, std::byte{0x71});
					for (std::size_t y = 0; y < expanded[1]; ++y) for (std::size_t x = 0; x < expanded[0]; ++x)
						std::fill_n(expected.begin() + y * layout.row_pitch + x * 4, 4, std::byte{0x40});
					memory->write(layout.range, expected);
					std::vector<std::byte> source_bytes(64);
					for (unsigned pixel = 0; pixel < 16; ++pixel)
					{
						std::fill_n(source_bytes.begin() + pixel * 4, 3, std::byte{0x20}); source_bytes[pixel * 4 + 3] = std::byte{0xa9};
					}
					memory->write({98304, 64}, source_bytes);
					const auto target = cache.acquire({4096, {4, 4, samples, surface_depth::z24s8_uint, false}, layout.row_pitch, {}});
					const auto source = cache.acquire({98304, {4, 4, 1, surface_depth::z24s8_uint, false}, 0, {}});
					if (background) cache.prepare(target);
					if (background == 2) cache.spill(target);
					cache.defer_copy(source, {1, 1, 2, 2}, target, 1, 1, plane);
					cache.erase(source); // The deferred plan alone retains its immutable native source.
					remap(provider, *memory, unmap);
					const auto alias = cache.acquire({0x41000, {4, 4, 1, surface_depth::z24s8_uint, false}, 0, {}});
					cache.prepare(alias);
					require(cache.status(alias).initialized, "Physical depth fixture failed to create a clean old-alias mirror");
					const auto replacement = std::vector<std::byte>(layout.range.size, std::byte{0x5c});
					if (!unmap) memory->write(layout.range, replacement);
					cache.flush_physical(target);
					for (std::size_t y = expanded[1] / 4; y < 3 * expanded[1] / 4; ++y)
						for (std::size_t x = expanded[0] / 4; x < 3 * expanded[0] / 4; ++x)
							for (unsigned b = 0; b < 4; ++b)
								if ((plane == texture_plane::depth && b < 3) || (plane == texture_plane::stencil && b == 3))
									expected[y * layout.row_pitch + x * 4 + b] = b == 3 ? std::byte{0xa9} : std::byte{0x20};
					require(memory->read({0x41000, layout.range.size}).bytes == expected, "Physical deferred depth/stencil changed another aspect,background,sample or padding");
					if (!unmap) require(memory->read(layout.range).bytes == replacement, "Physical deferred depth/stencil wrote replacement allocation");
					require(!cache.status(alias).initialized, "Physical deferred publication retained a freshly prepared stale clean alias");
					++count;
				}
	std::cout << "Native Metal4 physical deferred depth:" << count << " Z24S8 remap/unmap profiles;depth-only/stencil-only;VM/clean-native/raw-spill backgrounds;1x/2x/4x samples;source erase lifetime;fresh clean alias invalidation;other aspect/exterior/padding preserved;no App/UI/guest VM started\n";
}
void deferred_failure_and_retention()
{
	for (unsigned mode = 0; mode < 3; ++mode)
	{
		context device{1};
		auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
		auto memory = std::make_shared<guarded_guest_memory>(provider);
		const auto gate = [device.device() newSharedEvent];
		struct release_gate { id<MTLSharedEvent> gate; ~release_gate() { gate.signaledValue = 1; } } release{gate};
		const auto error = make_buffer(device, 4, buffer_storage::shared);
		*static_cast<std::uint32_t*>(error.contents) = 1;
		bool arm = false, injected = false;
		const cpu_capture_dispatch capture = [&](const auto& op)
		{
			// Interpose only at a real recording batch,not at a guessed number
			// of provider calls. Do not touch the context under VM exclusion.
			if (arm)
			{
				try
				{
					(void)device.recording_submission();
					if (mode == 2) device.require_gpu_success(error, 0, @"Intentional deferred scratch GPU status failure");
					else device.wait_event(gate, 1);
					arm = false; injected = true;
				}
				catch (const std::logic_error&) {} // No recording batch at this CPU boundary.
			}
			provider->execute([&](auto&) { op(); });
		};
		surface_cache cache(device, memory, 16 << 20, {}, capture);
		const auto target = cache.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		const auto source = cache.acquire({98304, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		const auto reader = cache.acquire({65536, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		cache.clear(target, 0xf0, 0xff2468ac, 0, 255, {0, 0, 4, 4});
		cache.defer_copy(target, {0, 0, 4, 4}, reader, 0, 0); // Retains the original target's native epoch.
		cache.clear(source, 0xf0, 0xff13579b, 0, 255, {0, 0, 4, 4});
		cache.defer_copy(source, {1, 1, 2, 2}, target, 1, 1);
		const auto before = cache.status(target);
		remap(provider, *memory, false);
		arm = true;
		if (mode != 2) cache.set_wait_timeout(std::chrono::milliseconds(1));
		bool failed = false; try { cache.flush_physical(target); } catch (const std::runtime_error&) { failed = true; }
		require(failed && injected && cache.status(target).deferred_copies == 1 && cache.status(target).dirty &&
			!cache.status(target).faulted && cache.status(target).content_generation == before.content_generation,
			"Deferred scratch timeout/status failure changed old target or consumed its plans");
		if (mode != 2)
		{
			require(cache.status(target).pending, "Deferred timeout lost its submitted scratch allocation");
			if (mode == 1) memory->write({0x41000, 64}, std::vector<std::byte>(64, std::byte{0x29}));
			gate.signaledValue = 1;
			cache.set_wait_timeout(std::chrono::seconds(30));
			failed = false; try { cache.synchronize(); } catch (const surface_conflict_error&) { failed = true; }
			require(failed == (mode == 1), "Deferred completion did not enforce physical CPU revision");
			if (mode == 1) require(cache.status(target).deferred_copies == 1 && !cache.status(target).faulted &&
				cache.status(target).content_generation == before.content_generation && memory->read({0x41000, 64}).bytes == std::vector<std::byte>(64, std::byte{0x29}),
				"Failed deferred physical commit lost CPU data,plans or original native metadata");
		}
		else device.recover_completed_errors();
		cache.flush_physical(reader);
		const auto original = memory->read({65536, 64});
		for (unsigned pixel = 0; pixel < 16; ++pixel)
			require(original.bytes[pixel * 4 + 1] == std::byte{0x24} && original.bytes[pixel * 4 + 3] == std::byte{0xac},
				"Deferred scratch failure/completion modified a captured original native source");
		if (mode != 1)
		{
			cache.flush_physical(target);
			const auto actual = memory->read({0x41000, 64});
			for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x)
				require(actual.bytes[(y * 4 + x) * 4 + 1] == (x >= 1 && x < 3 && y >= 1 && y < 3 ? std::byte{0x13} : std::byte{0x24}),
					"Deferred scratch retry lost old native background or planned copy");
		}
		require(memory->read({4096, 64}).bytes == std::vector<std::byte>(64), "Deferred failure recovery touched replacement VA");
	}
	{
		context device{1};
		auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
		auto memory = std::make_shared<guarded_guest_memory>(provider);
		const cpu_capture_dispatch capture = [&](const auto& op) { provider->execute([&](auto&) { op(); }); };
		surface_cache cache(device, memory, 16 << 20, {}, capture);
		const auto target = cache.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		const auto source = cache.acquire({98304, {4, 4, 1, surface_color::argb8, false}, 0, {}});
		cache.clear(target, 0xf0, 0xff2468ac, 0, 255, {0, 0, 4, 4});
		cache.clear(source, 0xf0, 0xff13579b, 0, 255, {0, 0, 4, 4});
		remap(provider, *memory, false);
		bool refused = false; try { cache.defer_copy(source, {0, 0, 4, 4}, target, 0, 0); } catch (const surface_conflict_error&) { refused = true; }
		require(refused && cache.status(target).dirty && !cache.status(target).deferred_copies,
			"Enqueue after dirty remap replaced the old physical output ticket");
		cache.flush_physical(target);
		require(memory->read({0x41000, 64}).bytes[1] == std::byte{0x24} && memory->read({4096, 64}).bytes == std::vector<std::byte>(64),
			"Rejected dirty enqueue lost its old physical output");
	}
	std::cout << "Native Metal4 physical deferred failures:real GPU-gated timeout and completion;newer old-alias CPU store rejects scratch commit;intentional GPU-status failure retains plans and retries;captured original native reader stays exact;dirty enqueue after remap refuses without replacing old tickets;no App/UI/guest VM started\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try { deferred_matrix(); deferred_depth_aspects(); deferred_failure_and_retention(); }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
