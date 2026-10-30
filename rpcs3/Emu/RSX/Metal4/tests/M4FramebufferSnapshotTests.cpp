#include "M4FramebufferFixtures.h"
#include "M4FramebufferSnapshot.h"
#include <iostream>
using namespace metal4;
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native framebuffer import contract differs");
}
void mapping_tests()
{
	auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
	auto table = std::make_shared<std::array<std::uint32_t, 3>>(std::array<std::uint32_t, 3>{0xfff00000, 0x200000, 0x100000});
	std::function<void()> on_translate;
	const auto translate = [table, &on_translate](std::uint32_t offset, unsigned location, std::size_t size)
	{
		if (on_translate) { auto action = std::move(on_translate); on_translate = {}; action(); }
		if (!location) return offset + 0x100000;
		const auto entry = offset >> 20; require(entry < table->size() && size <= 0x100000 - (offset & 0xfffff));
		return (*table)[entry] + (offset & 0xfffff);
	};
	tile_descriptor tile{1, 0xf0000, 0x40000, 256, 0, 0, 2, true}; tiles->update(0, tile);
	auto geometry = plan_guest_surface(0, 8, 65, 4, 256);
	const auto capture = [&] { return capture_surface_mapping(geometry, 0x100004, 1, translate, io, tiles, true, 64 << 20); };
	auto ticket = capture(); require(ticket->current());
	const auto& mapped = ticket->state();
	require(mapped.pixel_address == 0x200004 && mapped.tile && mapped.tile->address == 0xffff0000 && mapped.tile->offset == 0x10004);
	require(mapped.segments == std::vector<guest_storage_segment>{{0xffff0000, 0, 0x10000}, {0x200000, 0x10000, 0x30000}});
	require(ticket->same_epoch(*capture()));
	io->notify({0x200000, 0x100000}); require(ticket->current() && ticket->same_epoch(*capture()));
	io->notify({0x100000, 0x100000}); io->notify({0x100000, 0x100000}); require(!ticket->current());
	auto renewed = capture(); require(!ticket->same_epoch(*renewed) && renewed->current());
	tiles->update(0, {}); tiles->update(0, tile); require(!renewed->current());
	ticket = capture(); auto unrelated = tile; unrelated.offset = 0x800000; tiles->update(1, unrelated); require(ticket->current());
	on_translate = [&] { io->notify({0x100000, 0x100000}); }; require(!ticket->current());
	on_translate = [&] { tiles->update(0, {}); tiles->update(0, tile); };
	bool refused = false; try { (void)capture(); } catch (const surface_mapping_conflict&) { refused = true; } require(refused);
	tiles->update(0, {});
	auto absent = capture_surface_mapping(geometry, 0x100004, 1, translate, io, tiles, true, 64 << 20);
	tiles->update(0, tile); tiles->update(0, {}); require(!absent->current());
	auto local = capture_surface_mapping(geometry, 0x1000, 0, translate, io, tiles, true, 64 << 20);
	io->notify({1, 1}); require(local->current());
	std::cout << "Native framebuffer mapping:owned whole-tile disjoint topEA/later-origin capture,independent IO/tile ABA,unrelated changes,initial/final query brackets,absence ABA and local foreign-IO poison isolation passed\n";
}
void local_tile_mapping_tests()
{
	auto tiles = std::make_shared<tile_state>(); auto io = std::make_shared<io_mapping_epoch>();
	std::function<void()> on_translate;
	const auto translate = [&](unsigned offset, unsigned, std::size_t)
	{
		if (on_translate) { auto action = std::move(on_translate); on_translate = {}; action(); }
		return offset;
	};
	framebuffer_surface surface; surface.address = 0x100; surface.guest = plan_guest_surface(0x100, 8, 6, 4, 65);
	map_framebuffer_surface(surface, 0x100, 0, translate, io, tiles, true, 4096);
	const auto framebuffer = surface.mapping;
	const auto display = [&] { return capture_surface_mapping(surface.guest, 0x100, 0, translate, io, tiles, false, 4096); };
	auto receipt = display();
	require(framebuffer->current() && receipt->current() && !framebuffer->same_epoch(*receipt));
	unsigned cases = 0;
	for (unsigned comp : {0u, 7u, 8u, 9u, 10u, 11u, 12u}) for (unsigned bank = 0; bank < 4; ++bank) for (unsigned pitch : {256u, 1024u})
	{
		receipt = display();
		tiles->update(0, {0, 0, 0x10000, pitch, comp, cases, bank, true});
		require(!receipt->current() && framebuffer->current());
		map_framebuffer_surface(surface, 0x100, 0, translate, io, tiles, true, 4096);
		require(framebuffer->same_epoch(*surface.mapping) && !surface.mapping->state().tile && surface.mapping->state().logical.size < 4096);
		receipt = display(); require(receipt->current()); ++cases;
	}
	tiles->update(0, {}); require(!receipt->current() && framebuffer->current());
	receipt = display(); const auto local_tag = tile_descriptor{0, 0, 0x10000, 256, 8, 0, 3, true};
	tiles->update(0, local_tag); tiles->update(0, {}); require(!receipt->current() && framebuffer->current());
	on_translate = [&] { tiles->update(0, local_tag); tiles->update(0, {}); };
	require(!framebuffer->current()); // Even transparent queries require a stable producer bracket.
	on_translate = [&] { tiles->update(0, local_tag); tiles->update(0, {}); };
	bool refused = false; try { map_framebuffer_surface(surface, 0x100, 0, translate, io, tiles, true, 4096); } catch (const surface_mapping_conflict&) { refused = true; } require(refused);
	// Opting out of local tags never disables banked main-memory receipts.
	auto main = capture_surface_mapping(surface.guest, 0x100, 1, translate, io, tiles, false, 4096, false);
	tiles->update(1, {1, 0, 0x10000, 256, 0, 0, 1, true}); tiles->update(1, {}); require(!main->current());
	auto exhausted = std::make_shared<tile_state>(std::array<tile_descriptor, 15>{}, std::numeric_limits<std::uint64_t>::max() - 1);
	auto guarded = capture_surface_mapping(surface.guest, 0x100, 0, translate, io, exhausted, false, 4096, false);
	exhausted->update(0, local_tag); refused = false;
	try { (void)guarded->current(); } catch (const std::overflow_error&) { refused = true; } require(refused);
	std::cout << "Local transparent framebuffer mapping:56 compression/bank/pitch descriptors preserve native data epochs;strict AV receipts reject changes/absence ABA;policy identity,main-memory tile ABA,producer brackets andexhaustion guards passed\n";
}
void projection_tests()
{
	unsigned cases = 0;
	for (bool morton : {false, true})
		for (unsigned bytes : {1u, 2u, 4u, 8u, 16u})
			for (unsigned split : {1u, 13u, 31u, 63u})
			{
				const std::optional<swizzle_region> swizzle = morton ? std::optional<swizzle_region>{{{32, 32, 1}, {3, 7, 0}, {8, 5, 1}}} : std::nullopt;
				auto image = plan_guest_surface(0, 8, morton ? 5 : 1080, bytes, morton ? 0 : 256, swizzle);
				std::vector<bool> expected(image.range.size), actual(image.range.size);
				guest_surface_byte_cursor cursor(image);
				while (const auto run = cursor.next()) for (std::size_t i = run->address; i < run->address + run->size; ++i) expected[i] = true;
				const std::vector<guest_storage_segment> segments{{0xffff0000, 0, split}, {0x300000, split, image.range.size - split}};
				image.range.address = segments.front().address;
				const auto parts = project_guest_segments(image, segments, false);
				if (!morton) require(parts.size() <= 4); // Whole row groups do not consume1080watch slots.
				for (const auto& part : parts)
				{
					validate_guest_surface_range(part.coverage.range); guest_surface_byte_cursor active(part.coverage);
					while (const auto run = active.next())
						for (std::size_t i = 0; i < run->size; ++i)
						{
							const auto index = part.offset + run->address - part.coverage.range.address + i;
							require(index < actual.size() && !actual[index]); actual[index] = true;
						}
				}
				require(actual == expected); ++cases;
			}
	require(cases == 40);
	std::cout << "Native framebuffer segmented coverage:40linear1080row/Morton/1-2-4-8-16byte cases,midpixel/midrow IO boundaries,precise active bytes and bounded row-group watches passed\n";
}
void window_tests()
{
	auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
	const auto translate = [](unsigned offset, unsigned location, std::size_t size)
	{
		if (!location) return 0xc0000000u + offset;
		require(location == 1 && size <= 0x100000 - (offset & 0xfffff) && (offset >> 20) <= 1);
		return (offset < 0x100000 ? 0xfff00000u : 0x200000u) + (offset & 0xfffff);
	};
	std::vector<surface_format> formats;
	for (auto value : {surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565, surface_color::xrgb8_zero,
		surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8, surface_color::rgba16f,
		surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero, surface_color::xbgr8_one, surface_color::abgr8}) formats.emplace_back(value);
	for (auto value : {surface_depth::z16_uint, surface_depth::z24s8_uint, surface_depth::z16_float, surface_depth::z24s8_float}) formats.emplace_back(value);
	unsigned count = 0, zero_colors = 0, explicit_zero_depth = 0;
	for (const auto& format : formats)
		for (unsigned aa : {0u, 3u, 4u, 5u})
			for (bool morton : {false, true})
				for (const auto window : {std::array{0u, 0u}, std::array{4u, 0u}, std::array{1u, 1u}})
					for (unsigned location = 0; location < 2; ++location)
					{
						const auto* color = std::get_if<surface_color>(&format);
						const auto bytes = color ? color_layout(*color).pixel_bytes : framebuffer_depth_bytes(std::get<surface_depth>(format));
						const unsigned slot = color ? (count / 2) % 4 : 4, sample_x = aa ? 2 : 1;
						const unsigned raw = 0xffff0, pitch = (4 + (morton ? 0 : 2)) * sample_x * bytes;
						const unsigned expected_offset = raw + pitch * window[1] + bytes * sample_x * window[0];
						const unsigned analysed = translate(raw, location, 1) + pitch * window[1] + bytes * sample_x * window[0];
						test::framebuffer_layout layout; layout.color_addresses.fill(0); layout.actual_color_pitch.fill(0); layout.zeta_address = layout.actual_zeta_pitch = 0;
						layout.aa_mode = aa; layout.aa_factors = {sample_x, aa >= 4 ? 2u : 1u}; layout.raster_type = morton ? 2 : 1;
						if (color)
						{
							layout.color_format = static_cast<unsigned>(*color); layout.target = slot == 0 ? 1 : slot == 1 ? 2 : 0x1f;
							layout.color_addresses[slot] = analysed; layout.actual_color_pitch[slot] = pitch;
						}
						else { layout.target = 0; layout.depth_format = static_cast<unsigned>(std::get<surface_depth>(format)); layout.zeta_address = analysed; layout.zeta_pitch = layout.actual_zeta_pitch = pitch; }
						const auto zero_mask = analysed ? 0u : 1u << slot;
						if (!analysed) { if (color) ++zero_colors; else ++explicit_zero_depth; }
						require(framebuffer_window_zero_colors(layout) == (color ? zero_mask : 0));
						if (!color) require(framebuffer_window_zero_depth(layout, translate(raw, location, 1), window[0], window[1], pitch, 4 * bytes * sample_x) == zero_mask);
						unsigned callbacks = 0;
						const auto snapshot = framebuffer_snapshot::capture(layout, true, {}, 64 << 20, [&](auto& surface)
						{
							require(surface.physical_slot == slot && surface.address == analysed); ++callbacks;
							map_framebuffer_window_surface(surface, raw, location, window[0], window[1], pitch, translate, io, tiles, true, 64 << 20);
						}, zero_mask);
						const auto& surface = color ? *snapshot.colors()[0] : *snapshot.depth();
						require(callbacks == 1 && surface.address == translate(expected_offset, location, 1) && surface.mapping->current());
						const auto& mapping = surface.mapping->state(); require(mapping.logical.address == expected_offset && !mapping.tile && mapping.location == location);
						for (const auto& part : mapping.segments)
							for (std::size_t byte = 0; byte < part.size; ++byte)
								require(part.address + byte == translate(expected_offset + part.offset + byte, location, 1));
						apply_framebuffer_origins(layout, snapshot);
						require((color ? layout.color_addresses[slot] : layout.zeta_address) == surface.address);
						++count;
					}
	require(count == 864 && zero_colors && explicit_zero_depth);
	// A zero depth EA is not sufficient to revive suppressed attachments.
	test::framebuffer_layout depth_layout; depth_layout.color_addresses.fill(0); depth_layout.actual_color_pitch.fill(0);
	depth_layout.zeta_address = 0; depth_layout.zeta_pitch = depth_layout.actual_zeta_pitch = 16;
	require(framebuffer_window_zero_depth(depth_layout, 0xfffffff0, 4, 0, 80, 64) == 16); // Raw80 corrected to16,still accepted.
	require(framebuffer_window_zero_depth(depth_layout, 0xfffffff0, 4, 0, 32, 64) == 0); // Original small-pitch rejection.
	depth_layout.actual_color_pitch[1] = 16; depth_layout.color_addresses[1] = 0;
	require(framebuffer_window_zero_depth(depth_layout, 0xfffffff0, 4, 0, 80, 64) == 0); // Active B alias suppressed depth.
	depth_layout.actual_color_pitch[1] = 0; depth_layout.actual_zeta_pitch = 0;
	require(framebuffer_window_zero_depth(depth_layout, 0xfffffff0, 4, 0, 80, 64) == 0); // Unused depth retained no pitch.
	depth_layout.actual_zeta_pitch = 16;
	require(framebuffer_window_zero_depth(depth_layout, 0xffffffe0, 4, 0, 80, 64) == 0);
	require(framebuffer_window_zero_depth(depth_layout, 0, 4, 0, 80, 64) == 0);
	unsigned refused = 0;
	const auto refusal = [&](auto operation) { try { operation(); } catch (const std::exception&) { ++refused; } };
	refusal([&] { (void)framebuffer_window_origin(0xfffffff0, 4, 0, 256, 4, 1); });
	refusal([&] { (void)framebuffer_window_origin(0, 0, 2, std::numeric_limits<std::size_t>::max(), 4, 1); });
	refusal([&] { (void)framebuffer_window_origin(0, 0, 0, 0, 4, 1); });
	refusal([&] { (void)framebuffer_window_origin(0, 0, 0, 256, 0, 1); });
	refusal([&] { (void)framebuffer_window_origin(0, 0, 0, 256, 4, 3); });
	refusal([&] { (void)framebuffer_snapshot::capture(test::framebuffer_layout{}, true, {}, 64 << 20, {}, 1); });
	refusal([&] { (void)framebuffer_snapshot::capture(test::framebuffer_layout{}, true, {}, 64 << 20, [](auto&) {}, 32); });
	require(refused == 7);
	std::cout << "Native framebuffer logical window:864 color/depth/AA/linear-Morton/local-main captures,physical B/hole/depth slots,topEA reverse/disjoint translation,per-byte segment oracle,zero-EA color/depth presence without reviving alias/unused/bad-pitch depth,raw versus fixed pitch,corrected shared origins and7 overflow/contract refusals passed;productive lifecycle and leases remain open\n";
}
// Independent rectangular Morton oracle: consume the largest remaining
// coordinate bit first, splitting Y before X when their dimensions tie.
std::size_t framebuffer_morton_oracle(std::size_t x, std::size_t y, std::size_t width, std::size_t height)
{
	std::size_t index = 0, cells = width * height;
	while (width > 1 || height > 1)
	{
		cells /= 2;
		if (height > 1 && height >= width)
		{
			height /= 2;
			if (y >= height) { index += cells; y -= height; }
		}
		else
		{
			width /= 2;
			if (x >= width) { index += cells; x -= width; }
		}
	}
	return index;
}
void rounded_morton_tests()
{
	std::vector<surface_format> formats;
	for (auto value : {surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565, surface_color::xrgb8_zero,
		surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8, surface_color::rgba16f,
		surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero, surface_color::xbgr8_one, surface_color::abgr8}) formats.emplace_back(value);
	for (auto value : {surface_depth::z16_uint, surface_depth::z24s8_uint, surface_depth::z16_float, surface_depth::z24s8_float}) formats.emplace_back(value);
	auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
	const auto translate = [](std::uint32_t offset, unsigned location, std::size_t bytes)
	{
		const std::array<unsigned, 4> entries{0xfff00000, 0x200000, 0x100000, 0x400000};
		require(location == 1 && (offset >> 20) < entries.size() && bytes <= 0x100000 - (offset & 0xfffff));
		return entries[offset >> 20] + (offset & 0xfffff);
	};
	unsigned count = 0;
	for (const auto& format : formats)
		for (const auto shape : {std::array{1u, 1u}, std::array{3u, 5u}, std::array{5u, 3u}, std::array{7u, 17u},
			std::array{17u, 7u}, std::array{31u, 1u}, std::array{1u, 31u}, std::array{33u, 65u}})
			for (unsigned aa : {0u, 3u, 4u, 5u})
				for (bool larger : {false, true})
				{
					const auto* color = std::get_if<surface_color>(&format);
					const auto bytes = color ? color_layout(*color).pixel_bytes : framebuffer_depth_bytes(std::get<surface_depth>(format));
					const unsigned sx = aa ? 2 : 1, sy = aa >= 4 ? 2 : 1, raw = 0xffff0;
					const auto pitch = shape[0] * sx * bytes;
					test::framebuffer_layout layout; layout.color_addresses.fill(0); layout.actual_color_pitch.fill(0); layout.zeta_address = layout.actual_zeta_pitch = 0;
					layout.width = shape[0]; layout.height = shape[1]; layout.aa_mode = aa; layout.aa_factors = {sx, sy}; layout.raster_type = 2;
					const auto analysed = static_cast<unsigned>(translate(raw, 1, 1) + bytes * sx);
					if (color) { layout.target = 2; layout.color_format = static_cast<unsigned>(*color); layout.color_addresses[1] = analysed; layout.actual_color_pitch[1] = static_cast<unsigned>(pitch); }
					else { layout.target = 0; layout.depth_format = static_cast<unsigned>(std::get<surface_depth>(format)); layout.zeta_address = analysed; layout.zeta_pitch = layout.actual_zeta_pitch = static_cast<unsigned>(pitch); }
					const std::array<std::size_t, 3> storage{std::size_t(1) << std::bit_width(shape[0] * sx - 1), std::size_t(1) << std::bit_width(shape[1] * sy - 1), 1};
					const auto explicit_storage = larger ? std::optional<std::array<std::size_t, 3>>{{storage[0] * 2, storage[1] * 2, 1}} : std::nullopt;
					const auto expected_storage = explicit_storage.value_or(storage);
					const auto present = analysed ? 0u : color ? 2u : 16u;
					const auto snapshot = framebuffer_snapshot::capture(layout, true, explicit_storage, 64 << 20, [&](auto& surface)
						{ map_framebuffer_window_surface(surface, raw, 1, 1, 0, pitch, translate, io, tiles, true, 64 << 20); }, present);
					const auto& surface = color ? *snapshot.colors()[0] : *snapshot.depth();
					require(surface.morton->storage == expected_storage && surface.morton->size == std::array<std::size_t, 3>{shape[0] * sx, shape[1] * sy, 1});
					require(!surface.row_pitch && surface.guest.range.size == expected_storage[0] * expected_storage[1] * bytes && surface.mapping->current());
					std::vector<bool> expected(surface.guest.range.size), actual(surface.guest.range.size);
					for (unsigned y = 0; y < shape[1] * sy; ++y)
						for (unsigned x = 0; x < shape[0] * sx; ++x)
						{
							const auto first = framebuffer_morton_oracle(x, y, expected_storage[0], expected_storage[1]) * bytes;
							for (std::size_t lane = 0; lane < bytes; ++lane) { require(!expected[first + lane]); expected[first + lane] = true; }
						}
					// Geometry is relative to complete logical storage, not a pretend
					// contiguous top-EA range spanning disjoint physical segments.
					auto relative = surface.guest; relative.range.address = 0; guest_surface_byte_cursor cursor(relative);
					while (const auto run = cursor.next()) for (std::size_t byte = run->address; byte < run->address + run->size; ++byte) { require(!actual[byte]); actual[byte] = true; }
					require(actual == expected); ++count;
				}
	require(count == 1152);
	const auto largest = std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1);
	require(framebuffer_morton_storage({largest, 1, 1}) == std::array{largest, std::size_t(1), std::size_t(1)});
	unsigned refused = 0; const auto refusal = [&](auto action) { try { action(); } catch (const std::exception&) { ++refused; } };
	refusal([&] { (void)framebuffer_morton_storage({0, 1, 1}); });
	refusal([&] { (void)framebuffer_morton_storage({largest + 1, 1, 1}); });
	test::framebuffer_layout layout; layout.width = 7; layout.height = 5; layout.raster_type = 2; layout.zeta_address = 0;
	refusal([&] { (void)framebuffer_snapshot::capture(layout, true, {}, 100); });
	refusal([&] { (void)framebuffer_snapshot::capture(layout, true, std::array<std::size_t, 3>{7, 8, 1}); });
	require(refused == 4);
	std::cout << "Native framebuffer rounded Morton:1152 color/depth/eight rectangular shapes/fourAA/default-rounded-or-explicit-larger captures,logical windows and disjoint reverse IO,independent largest-axis byte coverage oracle,padding exclusion and4 storage/rounding refusals passed;MSAA binding and productive leases remain open\n";
}
int main()
{
	try
	{
		mapping_tests(); local_tile_mapping_tests(); projection_tests(); window_tests(); rounded_morton_tests();
		test::framebuffer_layout layout;
		unsigned profiles = 0;
		for (unsigned format : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			for (unsigned depth = 1; depth <= 4; ++depth)
				for (unsigned target : {0u, 1u, 2u, 0x13u, 0x17u, 0x1fu})
					for (unsigned aa : {0u, 3u, 4u, 5u})
						for (unsigned raster : {1u, 2u})
						{
							layout.color_format = format;
							layout.depth_format = depth;
							layout.target = target;
							layout.aa_mode = aa;
							layout.raster_type = raster;
							layout.aa_factors = {aa ? 2u : 1u, aa >= 4 ? 2u : 1u};
							const auto snapshot = framebuffer_snapshot::capture(layout, true);
							const unsigned mask = target == 2 ? 2 : target == 1	 ? 1
																: target == 0x13 ? 3
																: target == 0x17 ? 7
																: target == 0x1f ? 15
																				 : 0;
							unsigned count = 0;
							for (unsigned physical = 0; physical < 4; ++physical)
								if (mask & (1u << physical))
								{
									const auto& surface = *snapshot.colors()[count++];
									require(surface.physical_slot == physical && surface.address == layout.color_addresses[physical]);
									require(surface.row_pitch == (raster == 1 ? 256 : 0) && bool(surface.morton) == (raster == 2));
									require(surface.guest.width == 4 * layout.aa_factors[0] && surface.guest.height == 4 * layout.aa_factors[1]);
									require(surface.samples == (aa == 0 ? 1 : aa == 3 ? 2
																					  : 4));
								}
							for (; count < 4; ++count)
								require(!snapshot.colors()[count]);
							require(snapshot.depth()->address == 0x5000 && snapshot.depth()->guest.pixel_bytes == (depth == 1 || depth == 3 ? 2 : 4));
							++profiles;
						}
		layout = {};
		layout.target = 0x1f;
		layout.color_addresses[1] = 0;
		layout.color_addresses[3] = 0;
		const auto compact = framebuffer_snapshot::capture(layout, true);
		require(compact.colors()[0]->physical_slot == 0 && compact.colors()[1]->physical_slot == 2 && !compact.colors()[2]);
		layout.raster_type = 2;
		const auto cropped = framebuffer_snapshot::capture(layout, true, std::array<std::size_t, 3>{8, 8, 1});
		require(cropped.colors()[0]->guest.range.size == 256 && cropped.colors()[0]->morton->size == std::array<std::size_t, 3>{4, 4, 1});
		layout.color_addresses.fill(0);
		layout.zeta_address = 0;
		require(cropped.colors()[0]->address == 0x1000 && cropped.depth()->address == 0x5000);
		unsigned rejected = 0;
		const auto refusal = [&](auto action)
		{ try { action(); } catch (const std::exception&) { ++rejected; } };
		layout = {};
		refusal([&]
				{ (void)framebuffer_snapshot::capture(layout, false); });
		const auto bad = [&](auto modify)
		{
			auto value = layout;
			modify(value);
			refusal([&]
					{ (void)framebuffer_snapshot::capture(value, true); });
		};
		bad([](auto& v)
			{ v.width = 0; });
		bad([](auto& v)
			{ v.height = 0; });
		bad([](auto& v)
			{ v.color_format = 6; });
		bad([](auto& v)
			{ v.depth_format = 5; });
		bad([](auto& v)
			{ v.target = 3; });
		bad([](auto& v)
			{ v.aa_mode = 1; });
		bad([](auto& v)
			{ v.aa_factors[0] = 2; });
		bad([](auto& v)
			{ v.raster_type = 0; });
		bad([](auto& v)
			{ v.actual_color_pitch[0] = 15; });
		bad([](auto& v)
			{ v.actual_zeta_pitch = 15; });
		bad([](auto& v)
			{ v.color_addresses[0] = 0xfffffff0; });
		bad([](auto& v)
			{ v.color_addresses.fill(0); v.zeta_address = 0; });
		refusal([&]
				{ (void)framebuffer_snapshot::capture(layout, true, {}, 16); });
		refusal([&]
				{ (void)framebuffer_snapshot::capture(layout, true, std::array<std::size_t, 3>{8, 8, 1}); });
		layout.raster_type = 2;
		refusal([&]
				{ (void)framebuffer_snapshot::capture(layout, true, std::array<std::size_t, 3>{2, 4, 1}); });
		require(rejected == 16);
		std::cout << "Native framebuffer snapshot: " << profiles << " format/depth/target/AA/raster profiles,logical B/hole slots,actual pitches,cropped Morton,ownership and16 refusals passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
