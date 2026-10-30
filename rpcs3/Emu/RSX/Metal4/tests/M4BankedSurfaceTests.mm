#include "Emu/RSX/Common/tiled_dma_copy.hpp"
#include "M4GuardedGuestMemory.h"
#include "M4SurfaceCache.h"
#include "M4FramebufferBinding.h"
#include "M4FramebufferFixtures.h"
#include "M4GuestTexture.h"
#include "M4Display.h"
#include "M4MemoryPressure.h"
#include "M4DrawExecutorFixtures.h"
#include "M4DrawPacketFixtures.h"
#include <cstring>
#include <iostream>
#include <mutex>
#include <bit>
#include <cmath>
#import <AppKit/NSApplication.h>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
// The unchanged shared RSX reference transfers one aligned four-byte word.
// Read/modify/write the selected lane to obtain an independent byte oracle
// for every guest pixel width,including unaligned1/2-byte active origins.
void reference(const guest_surface_layout& image, const banked_tile_layout& tile,
	std::vector<std::byte>& banked, std::vector<std::byte>& linear, bool encode)
{
	for (std::size_t y = 0; y < image.height; ++y)
		for (std::size_t x = 0; x < image.width; ++x)
			for (std::size_t lane = 0; lane < image.pixel_bytes; ++lane)
			{
				auto logical = tile.offset + y * tile.pitch + x * image.pixel_bytes + lane;
				if (image.morton)
				{
					const auto plan = plan_swizzle(*image.morton, image.pixel_bytes);
					logical = tile.offset + morton_index({x + image.morton->origin[0], y + image.morton->origin[1], 0}, plan.log2) * image.pixel_bytes + lane;
				}
				std::array<char, 4> word{};
				const rsx::detiler_config config{0, banked_tile_factor(tile.pitch), tile.pitch / 256, tile.address, tile.size,
					static_cast<std::uint32_t>(logical & ~std::size_t(3)), 0, tile.pitch, tile.bank, 1, 1, 4, 4};
				rsx::tiled_dma_copy(0, 0, config, reinterpret_cast<char*>(banked.data()), word.data(), 1);
				const auto index = (y * image.width + x) * image.pixel_bytes + lane;
				if (encode)
				{
					word[logical & 3] = std::to_integer<char>(linear[index]);
					rsx::tiled_dma_copy(0, 0, config, reinterpret_cast<char*>(banked.data()), word.data(), 0);
				}
				else linear[index] = std::byte(static_cast<unsigned char>(word[logical & 3]));
			}
}
class provider final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<provider>
{
	std::mutex mutex;
	std::vector<std::byte> bytes = std::vector<std::byte>(4 << 20);
  public:
	bool mirror=false;
	guest_surface_range canonical(guest_surface_range range) const
	{
		if(mirror&&range.address>=0x250000&&range.address<0x290000){if(range.size>0x290000-range.address)throw std::out_of_range("Banked display mirror bounds");range.address-=0x200000;}return range;
	}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		return std::vector<storage_identity_span>{{shared_from_this(),canonical(range).address,range}};
	}
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* view = guest_memory_epoch::current(*this)) { operation(*view); return; }
		std::lock_guard lock(mutex); guest_memory_epoch epoch(*this, *this); operation(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		require(guest_memory_epoch::current(*this) == this, "Banked surface VM map escaped provider");
		range=canonical(range);
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Banked surface provider bounds");
		return bytes.data() + range.address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(canonical(a),canonical(b)); }
	void written(guest_surface_range) noexcept override {}
};
// Guard actual pixel capture,not the metadata-only checked_map used when
// guarded_guest_memory retains its current backing lease.
class display_memory final : public guest_surface_memory
{
  public:
	std::shared_ptr<provider> access=std::make_shared<provider>();
	std::shared_ptr<guarded_guest_memory> raw=std::make_shared<guarded_guest_memory>(access);
	std::vector<guest_surface_range> forbidden,reads;
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override {return raw->watch_surface(coverage);}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		const auto range=access->canonical(coverage.range);for(auto active:forbidden)require(!overlaps(active,range),"Display captured GPU-authoritative bank pixels");reads.push_back(range);return raw->read_surface(coverage);
	}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override {return raw->surface_revision(coverage);}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override {return raw->retain_backing(ranges);}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override {return raw->storage_identity(range);}
	bool test_storage(const storage_test& test) override {return raw->test_storage(test);}
	bool test_rebase(const rebase_test& test) override {return raw->test_rebase(test);}
	void write(guest_surface_range range,std::span<const std::byte> bytes) override {raw->write(range,bytes);}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout&,std::uint64_t,std::span<const std::byte>) override {throw std::logic_error("Display wroteback native bank pixels");}
};
// A wrapped top EA plus reverse physical entries. The mirror proves that
// cache overlap planning asks the provider about real storage identities.
class segmented_provider final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<segmented_provider>
{
	std::mutex mutex;
	std::vector<std::byte> bytes = std::vector<std::byte>(4 << 20);
	guest_surface_range physical(guest_surface_range range) const
	{
		validate_guest_surface_range(range);
		if (range.address >= 0xffff0000)
		{
			if (range.size > 0x1'0000'0000ull - range.address) throw std::out_of_range("Top segment bounds");
			return {0x300000 + range.address - 0xffff0000, range.size};
		}
		if (range.address >= 0x200000 && range.address < 0x210000)
		{
			if (range.size > 0x210000 - range.address) throw std::out_of_range("Mirror segment bounds");
			return {0x300000 + range.address - 0x200000, range.size};
		}
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Segment provider bounds");
		return range;
	}
  public:
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		return std::vector<storage_identity_span>{{shared_from_this(),physical(range).address,range}};
	}
	unsigned captures = 0, depth = 0;
	bool fail_tail_write = false;
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* view = guest_memory_epoch::current(*this)) { operation(*view); return; }
		std::lock_guard lock(mutex); guest_memory_epoch epoch(*this, *this); operation(*this);
	}
	cpu_capture_dispatch capture()
	{
		return [this](const auto& operation)
		{
			execute([&](auto&)
			{
				++captures; ++depth;
				try { operation(); } catch (...) { --depth; throw; }
				--depth;
			});
		};
	}
	std::byte* map(guest_surface_range range, bool write) override
	{
		require(guest_memory_epoch::current(*this) == this, "Segment VM map escaped provider");
		if (write && fail_tail_write && range.address == 0x40000) throw std::out_of_range("Last segment write map refused");
		return bytes.data() + physical(range).address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(physical(a), physical(b)); }
	void written(guest_surface_range) noexcept override {}
};
class segmented_memory final : public guest_surface_memory
{
	static std::uint64_t salt(guest_surface_range range)
	{
		return range.address == 0x40000 ? (1ull << 48) : range.address == 0x90000 ? (1ull << 32) : 0;
	}
  public:
	std::shared_ptr<segmented_provider> access = std::make_shared<segmented_provider>();
	std::shared_ptr<guarded_guest_memory> raw = std::make_shared<guarded_guest_memory>(access);
	std::vector<unsigned> read_epochs;
	std::function<void()> on_publish;
	unsigned batch_size = 0;
	std::vector<guest_surface_range> display_forbidden,display_reads;
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override {return raw->storage_identity(range);}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override {return raw->retain_backing(ranges);}
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override
	{
		require(access->depth == 1, "Surface watch lacks common CPU epoch"); return raw->watch_surface(coverage);
	}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		for(auto active:display_forbidden)require(!overlaps(active,coverage.range),"Mapped bankdisplay captured native pixels");display_reads.push_back(coverage.range);
		require(access->depth == 1, "Segment gather lacks common CPU epoch"); read_epochs.push_back(access->captures);
		auto result = raw->read_surface(coverage); result.revision += salt(coverage.range); return result;
	}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override
	{
		require(access->depth == 1, "Segment version query lacks common CPU epoch"); return raw->surface_revision(coverage) + salt(coverage.range);
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage, std::uint64_t expected, std::span<const std::byte> bytes) override
	{
		const guest_surface_write write{coverage, expected, bytes}; return compare_write_many({&write, 1});
	}
	std::optional<std::uint64_t> compare_write_many(std::span<const guest_surface_write> writes) override
	{
		require(access->depth == 1, "Segment scatter lacks common CPU epoch"); batch_size = writes.size();
		if (on_publish) { auto operation = std::move(on_publish); on_publish = {}; operation(); }
		std::vector<guest_surface_write> decoded(writes.begin(), writes.end());
		for (auto& write : decoded) write.expected -= salt(write.coverage.range);
		return raw->compare_write_many(decoded);
	}
	void write(guest_surface_range range, std::span<const std::byte> bytes) override { raw->write(range, bytes); }
	bool test_storage(const storage_test& test) override { return raw->test_storage(test); }
	bool test_rebase(const rebase_test& test) override { return raw->test_rebase(test); }
};
void segmented_cache_case(context& ctx, const std::shared_ptr<heap_budget>& budget, bool morton)
{
	auto memory = std::make_shared<segmented_memory>();
	surface_cache cache(ctx, memory, 64 << 20, budget, memory->access->capture());
	const banked_tile_layout tile{0xffff0000, 0x40000, 768, 2, 0x10000 + (morton ? 20u : 772u)};
	const unsigned width = 8, height = morton ? 5 : 65;
	const auto swizzle = morton ? std::optional<swizzle_region>{{{16, 16, 1}, {3, 7, 0}, {width, height, 1}}} : std::nullopt;
	const std::vector<guest_storage_segment> segments{{0xffff0000, 0, 0x10000}, {0x90000, 0x10000, 0x20000}, {0x40000, 0x30000, 0x10000}};
	auto image = plan_guest_surface(0, width, height, 4, morton ? 0 : tile.pitch, swizzle); image.range = {tile.address, tile.size};
	validate_banked_tile_geometry(image, tile);
	std::vector<std::byte> pixels(width * height * 4);
	for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte(std::array{255u, 20u, 40u, 60u}[i % 4]);
	std::vector<std::byte> encoded(tile.size, std::byte{0xa5}); reference(image, tile, encoded, pixels, true);
	for (const auto& part : segments) memory->write({part.address, part.size}, std::span(encoded).subspan(part.offset, part.size));
	const auto gather = [&]
	{
		std::vector<std::byte> result(tile.size);
		for (const auto& part : segments)
		{
			const auto bytes = memory->raw->read({part.address, part.size}).bytes;
			std::copy(bytes.begin(), bytes.end(), result.begin() + part.offset);
		}
		return result;
	};
	const cached_surface_description description{0x90000 + tile.offset - 0x10000, {width, height, 1, surface_color::argb8, false}, morton ? 0u : tile.pitch, swizzle, tile, segments};
	const auto banked = cache.acquire(description), linear = cache.acquire({0x100000, {width, height, 1, surface_color::argb8, false}, width * 4, {}});
	require(cache.acquire(description) == banked, "Disjoint tile key lost segment identity");
	auto different = description; different.segments.back().address = 0x50000;
	require(cache.acquire(different) != banked, "Disjoint tile key ignored later physical segment");
	auto invalid = description; invalid.address = tile.address + tile.offset; bool refused = false;
	try { (void)cache.acquire(invalid); } catch (const std::invalid_argument&) { refused = true; }
	require(refused, "Disjoint tile accepted a fictitious base-plus-offset pixel EA");
	invalid = description; invalid.segments[1].offset++; refused = false;
	try { (void)cache.acquire(invalid); } catch (const std::invalid_argument&) { refused = true; }
	require(refused, "Disjoint tile accepted a logical storage gap");
	surface_cache legacy(ctx, memory, 64 << 20, budget); refused = false;
	try { (void)legacy.acquire(description); } catch (const std::invalid_argument&) { refused = true; }
	require(refused, "Disjoint tile accepted missing common CPU exclusion");
	memory->read_epochs.clear(); cache.prepare(banked);
	require(memory->read_epochs.size() == 3 && memory->read_epochs[0] == memory->read_epochs[1] && memory->read_epochs[1] == memory->read_epochs[2], "Tile segments were gathered in separate CPU epochs");
	const auto versions = cache.status(banked);
	require(versions.guest_revision == 0 && versions.guest_revisions.size() == 3 && versions.guest_revisions[0] < versions.guest_revisions[1] && versions.guest_revisions[1] < versions.guest_revisions[2], "Disjoint versions lost independent watch domains");
	cache.copy(banked, {0, 0, width, height}, linear, 0, 0); cache.flush(linear);
	require(memory->raw->read({0x100000, pixels.size()}).bytes == pixels, "Top-EA disjoint surface gather decoded incorrect pixels");
	cache.clear(banked, 0xf0, 0xff123456, 0, 255, {0, 0, width, height});
	for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]);
	auto expected = encoded; reference(image, tile, expected, pixels, true);
	cache.prepare_cpu_read({0x70000, 1}); require(cache.status(banked).dirty, "Unmapped tile gap caused a false native writeback");
	cache.prepare_cpu_read({0x20ffff, 1});
	require(cache.cpu_read({0x20ffff, 1}).bytes == std::vector{expected[0xffff]}, "Common-epoch CPU read lost mirror identity");
	require(!cache.status(banked).dirty && memory->batch_size == 3 && gather() == expected, "Mirrored top segment failed to trigger whole-allocation atomic scatter");
	cache.clear(banked, 0xf0, 0xff123456, 0, 255, {0, 0, width, height});
	const auto before = gather();
	memory->on_publish = [&] { memory->write({0xffffffff, 1}, std::array{std::byte{0x77}}); };
	refused = false; try { cache.flush(banked); } catch (const surface_conflict_error&) { refused = true; }
	auto conflicted = before; conflicted[0xffff] = std::byte{0x77};
	require(refused && cache.status(banked).dirty && gather() == conflicted, "First segment conflict under larger unrelated version partially published scatter");
	cache.clear(banked, 0xf0, 0xff123456, 0, 255, {0, 0, width, height});
	memory->access->fail_tail_write = true; refused = false;
	try { cache.flush(banked); } catch (const std::out_of_range&) { refused = true; }
	require(refused && cache.status(banked).dirty && gather() == conflicted, "Last segment mapping failure published earlier segment bytes");
	memory->access->fail_tail_write = false; cache.flush(banked);
	expected = conflicted; reference(image, tile, expected, pixels, true);
	require(gather() == expected && !cache.status(banked).dirty, "Atomic scatter retry lost active pixels or untouched segment padding");
	cache.spill(banked); cache.prepare(banked); cache.defer_copy(banked, {0, 0, width, height}, linear, 0, 0); cache.flush(linear);
	require(memory->raw->read({0x100000, pixels.size()}).bytes == pixels, "Disjoint surface spill/deferred reader damaged pixels");
	cache.defer_copy(linear, {0, 0, width, height}, banked, 0, 0);
	memory->write({0xffffffff, 1}, std::array{std::byte{0x78}}); refused = false;
	try { cache.prepare(banked); } catch (const surface_conflict_error&) { refused = true; }
	require(refused && cache.status(banked).deferred_copies == 1, "Disjoint deferred destination lost first segment revision");
	cache.cancel_deferred_copies(banked);
	std::cout << "Native disjoint banked surface cache:" << (morton ? "Morton:" : "linear:") << "top32-bit EA/reverse segments/later-segment pixel origin,one-epoch gather,independent version vector,physical mirror and gap aliases,atomic three-part scatter,first-part CAS conflict,last-part map failure without partial stores,retry padding,spill/deferred copies passed;productive framebuffer IO/tile tickets remain open\n";
}
void framebuffer_cache_cases(context& ctx, const std::shared_ptr<heap_budget>& budget)
{
	for (unsigned mode = 0; mode < 8; ++mode)
	{
		const auto profile = mode % 4; const bool banked = profile >= 2, morton = profile & 1, window = mode >= 4;
		auto memory = std::make_shared<segmented_memory>(); auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
		const tile_descriptor tile{1, 0xf0000, 0x40000, 768, 0, 0, 2, banked};
		if (banked) tiles->update(0, tile);
		const unsigned raw = window ? 0xffff0 : banked ? 0x100000 + (morton ? 20 : 772) : 0xffff0;
		const unsigned window_x = window ? 4 : 0, window_y = window && !morton ? 1 : 0;
		const unsigned actual_pitch = morton ? 32 : 768;
		const unsigned origin = raw + actual_pitch * window_y + 4 * window_x;
		const auto translate = [](unsigned offset, unsigned location, std::size_t bytes)
		{
			require(location == 1 && (offset >> 20) <= 1 && bytes <= 0x100000 - (offset & 0xfffff), "Framebuffer fixture translation escaped main IO entry");
			return (offset < 0x100000 ? 0xfff00000 : 0x90000) + (offset & 0xfffff);
		};
		test::framebuffer_layout layout; layout.width = 8; layout.height = morton ? 5 : 65; layout.zeta_address = 0;
		layout.color_addresses[0] = window ? static_cast<unsigned>(translate(raw, 1, 1) + actual_pitch * window_y + 4 * window_x) : translate(origin, 1, 1); layout.actual_color_pitch[0] = actual_pitch; layout.raster_type = morton ? 2 : 1;
		const auto storage = morton ? std::optional<std::array<std::size_t, 3>>{{16, 16, 1}} : std::nullopt;
		const auto capture = [&]
		{
			return framebuffer_snapshot::capture(layout, true, storage, 64 << 20, [&](auto& surface)
				{
					if (window) map_framebuffer_window_surface(surface, raw, 1, window_x, window_y, actual_pitch, translate, io, tiles, true, 64 << 20);
					else map_framebuffer_surface(surface, origin, 1, translate, io, tiles, true, 64 << 20);
				}, window && !layout.color_addresses[0] ? 1 : 0);
		};
		const auto framebuffer = capture(); const auto& surface = *framebuffer.colors()[0]; const auto& mapped = surface.mapping->state();
		std::vector<std::byte> pixels(layout.width * layout.height * 4);
		for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte(std::array{255u, 20u, 40u, 60u}[i % 4]);
		std::vector<std::byte> encoded(mapped.logical.size, std::byte{0xa5});
		const auto encode = [&](std::vector<std::byte>& target, std::vector<std::byte>& values)
		{
			if (banked) { reference(surface.guest, *mapped.tile, target, values, true); return; }
			for (unsigned y = 0; y < layout.height; ++y)
				for (unsigned x = 0; x < layout.width; ++x)
				{
					const auto offset = morton ? morton_index({x, y, 0}, plan_swizzle(*surface.morton, 4).log2) * 4 : y * surface.row_pitch + x * 4;
					std::copy_n(values.begin() + (y * layout.width + x) * 4, 4, target.begin() + offset);
				}
		};
		encode(encoded, pixels);
		for (const auto& part : mapped.segments) memory->write({part.address, part.size}, std::span(encoded).subspan(part.offset, part.size));
		const auto gather = [&]
		{
			std::vector<std::byte> result(mapped.logical.size);
			for (const auto& part : mapped.segments)
			{
				const auto bytes = memory->raw->read({part.address, part.size}).bytes;
				std::copy(bytes.begin(), bytes.end(), result.begin() + part.offset);
			}
			return result;
		};
		surface_cache cache(ctx, memory, 64 << 20, budget, tile_capture_dispatch(memory->access->capture(), tiles));
		const auto targets = acquire_framebuffer(framebuffer, cache); const auto target = targets.colors[0];
		require(target && cache.description(target).mapping == surface.mapping && cache.description(target).tile == mapped.tile, "Framebuffer binding dropped owned mapping/tile descriptor");
		require(acquire_framebuffer(capture(), cache).colors[0] == target, "Equivalent framebuffer ticket lost cache reuse");
		const auto linear = cache.acquire({0x100000, {layout.width, layout.height, 1, surface_color::argb8, false}, layout.width * 4, {}});
		cache.copy(target, {0, 0, layout.width, layout.height}, linear, 0, 0); cache.flush(linear);
		require(memory->raw->read({0x100000, pixels.size()}).bytes == pixels, "Mapped framebuffer tile/untiled linear/Morton gather decoded wrong pixels");
		cache.clear(target, 0xf0, 0xff123456, 0, 255, {0, 0, layout.width, layout.height});
		for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]);
		auto expected = encoded; encode(expected, pixels);
		if (!banked && !morton)
		{
			const auto part = std::find_if(mapped.segments.begin(), mapped.segments.end(), [](const auto& value) { return 32 >= value.offset && 32 - value.offset < value.size; });
			require(part != mapped.segments.end(), "Window framebuffer padding lacks actual IO segment");
			memory->write({static_cast<unsigned>(part->address + 32 - part->offset), 1}, std::array{std::byte{0x77}}); expected[32] = std::byte{0x77};
		}
		cache.flush(target); require(gather() == expected && !cache.status(target).dirty, "Mapped framebuffer CAS damaged active pixels or disjoint row padding");
		io->notify({0x200000, 0x100000}); cache.prepare(target); // Unrelated entry stays valid.
		const auto before = gather(); clear_engine clear(ctx); bool refused = false;
		const std::array uses{cached_surface_use{target, true}};
		const auto encode_clear = [&](context& active, std::span<const id<MTLTexture>> textures)
		{
			clear_request request; request.mask = 0xf0; request.color_value = 0xffabcdef;
			request.rectangle = {0, 0, layout.width, layout.height}; request.colors[0] = {{textures[0], 0, 0}, surface_color::argb8};
			(void)clear.run(active, request);
		};
		try
		{
			cache.draw(uses, [&](context& active, auto textures)
			{
				encode_clear(active, textures);
				memory->write({mapped.segments.front().address, 1}, std::array{before[0]});
			});
		}
		catch (const surface_conflict_error&) { refused = true; }
		require(refused && !cache.status(target).faulted && !cache.status(target).pending && !cache.status(target).dirty && gather() == before,
			"Mapped prequeue validation failed to cancel stale unsubmitted commands");
		cache.prepare(target); // Same-value VM store changed only its watch epoch.
		auto gate = [ctx.device() newSharedEvent]; require(bool(gate), "Framebuffer pending fixture lacks a native shared event");
		struct gate_guard { id<MTLSharedEvent> event; ~gate_guard() { [event setSignaledValue:1]; } } unblock{gate};
		cache.set_wait_timeout(std::chrono::milliseconds(0));
		bool timed_out = false;
		try
		{
			cache.draw(uses, [&](context& active, auto textures)
			{
				active.wait_event(gate, 1); encode_clear(active, textures);
			});
		}
		catch (const std::runtime_error&) { timed_out = true; }
		require(timed_out && cache.status(target).pending && !cache.status(target).faulted,
			"Framebuffer fixture did not retain a live queued GPU transaction");
		memory->access->execute([&](auto&)
		{
			if (profile == 1) { auto changed = tile; changed.bound = true; tiles->update(0, changed); tiles->update(0, tile); }
			else if (profile == 2) memory->write({mapped.segments.front().address, 1}, std::array{std::byte{0x78}});
			else { io->notify({0x100000, 0x100000}); io->notify({0x100000, 0x100000}); }
		});
		[gate setSignaledValue:1]; refused = false;
		try { cache.synchronize(std::chrono::seconds(30)); }
		catch (const surface_mapping_conflict&) { refused = profile != 2; }
		catch (const surface_conflict_error&) { refused = profile == 2; }
		auto untouched = before; if (profile == 2) untouched[0] = std::byte{0x78};
		require(refused && !cache.status(target).faulted && !cache.status(target).spilled && cache.status(target).initialized && !cache.status(target).pending && gather() == untouched,
			"Framebuffer GPU completion published an obsolete IO/tile/VM epoch");
	}
	std::cout << "Native framebuffer cache integration:8mapped banked/untiled linear/Morton cases,original plus logical-window disjoint/reverse EA rebasing and analyzed-zero color presence,topEA disjoint gather,owned FramebufferBinding/cache key,clear/writeback/precise row padding,unrelated IO validity;prequeue stale command cancellation,live pending transaction across timeout;IO ABA/tile absence ABA/VM writer after native queue submission refused at GPU completion without guest publication passed;productive backing/output leases and app activation remain open\n";
}
void mapped_native_alias_cases(context& ctx, const std::shared_ptr<heap_budget>& budget)
{
	compiler compiler(ctx);
	const auto library = compiler.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 mapped_vertex(uint i [[vertex_id]]) {
 const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1);
}
fragment float4 mapped_argb(float4 p [[position]],uint s [[sample_id]],constant uint& latest [[buffer(0)]]) {
 const uint x=uint(p.x),y=uint(p.y);
 return float4(float(32+s*16+x*4+y+latest*32)/255,float(64+y*5+s)/255,float(128+x*3+s*7)/255,1);
}
fragment float4 mapped_float(float4 p [[position]],uint s [[sample_id]],constant uint& latest [[buffer(0)]]) {
 const uint x=uint(p.x),y=uint(p.y);
 return float4((latest?.345678926f:.123456791f)+float(s*16+x+y*4)/1024,
  -.987654328f+float(s)/32,12.3456783f+float(x)/16,1.23456788f+float(y)/32);
}
struct mapped_depth_value {float depth [[depth(any)]];uint stencil [[stencil]];};
fragment mapped_depth_value mapped_depth(float4 p [[position]],uint s [[sample_id]],constant uint& latest [[buffer(0)]]) {
 const uint x=uint(p.x),y=uint(p.y);
 return {(latest?.345678926f:.123456791f)+float(s*16+x+y*4)/1024,17+s*31+x+y*7+latest};
}
)msl", @"Mapped individual MSAA sample writer");
	MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
	ds.depthCompareFunction = MTLCompareFunctionAlways;ds.depthWriteEnabled = YES;
	MTLStencilDescriptor* st = [MTLStencilDescriptor new];
	st.stencilCompareFunction = MTLCompareFunctionAlways;st.depthStencilPassOperation = MTLStencilOperationReplace;
	ds.frontFaceStencil = ds.backFaceStencil = st;
	const auto depth_state = [ctx.device() newDepthStencilStateWithDescriptor:ds];
	std::array<std::array<id<MTLRenderPipelineState>, 3>, 3> pipelines{};
	unsigned count = 0;
	for (const unsigned samples : {1u, 2u, 4u})
	for (const unsigned transition : {0u, 1u, 2u, 3u, 4u})
	for (bool banked : {false, true})
		for (bool morton : {false, true})
			for (unsigned kind = 0; kind < 3; ++kind)
			{
				@autoreleasepool
				{
					auto memory = std::make_shared<segmented_memory>(); auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
					const auto capture = tile_capture_dispatch(memory->access->capture(), tiles);
					const tile_descriptor tile{1, 0xf0000, 0x40000, 768, 0, 0, 2, banked}; tiles->update(0, tile);
					const unsigned origin = 0xffff0, bytes = kind == 1 ? 16 : 4;
					const surface_format format = kind == 0 ? surface_format{surface_color::argb8} : kind == 1 ? surface_format{surface_color::rgba32f} : surface_format{surface_depth::z24s8_float};
					const auto swizzle = morton ? std::optional<swizzle_region>{{{8, 8, 1}, {}, {8, 8, 1}}} : std::nullopt;
					const auto geometry = plan_guest_surface(0, 8, 8, bytes, morton ? 0 : 768, swizzle);
					const auto tail_address = std::make_shared<unsigned>(0x90000);
					const auto translate = [tail_address](unsigned offset, unsigned location, std::size_t size)
					{
						require(location == 1 && size <= 0x100000 - (offset & 0xfffff), "Native alias translation crossed an IO entry");
						return (offset < 0x100000 ? 0xfff00000 : *tail_address) + (offset & 0xfffff);
					};
					std::shared_ptr<const surface_mapping_revision> mapping;
					checked_cpu_capture(capture, [&] { mapping = capture_surface_mapping(geometry, origin, 1, translate, io, tiles, true, 64 << 20); });
					const auto& mapped = mapping->state();
					const std::vector<std::byte> initial(mapped.logical.size, std::byte{0xa5});
					for (const auto& part : mapped.segments) memory->write({part.address, part.size}, std::span(initial).subspan(part.offset, part.size));
					auto cache_owner = std::make_unique<surface_cache>(ctx, memory, 64 << 20, budget, capture); auto& cache = *cache_owner;
					guest_texture_cache textures(ctx, memory, translate, 64 << 20, budget, 32, texture_capture_bindings{capture, io});
					const auto factors = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
					const unsigned sample_index = samples == 1 ? 0 : samples == 2 ? 1 : 2;
					if (!pipelines[sample_index][kind])
					{
						auto pd = [MTL4RenderPipelineDescriptor new];
						pd.vertexFunctionDescriptor = library_function(library, @"mapped_vertex");
						pd.fragmentFunctionDescriptor = library_function(library, kind == 0 ? @"mapped_argb" : kind == 1 ? @"mapped_float" : @"mapped_depth");
						pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;pd.rasterSampleCount = samples;
						if (kind != 2) pd.colorAttachments[0].pixelFormat = kind == 0 ? MTLPixelFormatBGRA8Unorm : MTLPixelFormatRGBA32Float;
						pipelines[sample_index][kind] = compiler.render(pd);
					}
					auto source = cache.acquire({mapped.pixel_address, {8 / factors.x, 8 / factors.y, samples, format, samples > 1}, morton ? 0u : 768u, swizzle, mapped.tile, mapped.segments, mapping});
					cache.prepare(source, surface_initialization::background);

					const auto draw = [&](bool latest)
					{
						const std::array uses{cached_surface_use{source, true}};
						cache.draw(uses, [&](context& active, std::span<const id<MTLTexture>> images)
						{
							auto pass = [MTL4RenderPassDescriptor new];
							if (kind == 2)
							{
								pass.depthAttachment.texture = images[0]; pass.depthAttachment.loadAction = MTLLoadActionDontCare; pass.depthAttachment.storeAction = MTLStoreActionStore;
								pass.stencilAttachment.texture = images[0]; pass.stencilAttachment.loadAction = MTLLoadActionDontCare; pass.stencilAttachment.storeAction = MTLStoreActionStore;
							}
							else
							{
								pass.colorAttachments[0].texture = images[0]; pass.colorAttachments[0].loadAction = MTLLoadActionDontCare; pass.colorAttachments[0].storeAction = MTLStoreActionStore;

							}
							bindings fragment;
							const std::uint32_t changed = latest;
							fragment.set_buffer(0, make_upload(active, std::as_bytes(std::span(&changed, 1))));
							render_commands commands(active, pass);
							raster_state raster;raster.viewport = {0, 0, double(8 / factors.x), double(8 / factors.y), 0, 1};
							raster.scissor = {0, 0, 8 / factors.x, 8 / factors.y};
							commands.configure(pipelines[sample_index][kind], kind == 2 ? depth_state : nil, raster, {}, fragment);
							commands.draw(MTLPrimitiveTypeTriangle, 0, 3);commands.end();
						});
					};
					const auto words = [&](clear_rectangle rectangle)
					{
						fragment_texture_registers result;
						const auto delta = morton ? morton_index({rectangle.x, rectangle.y, 0}, plan_swizzle(*swizzle, bytes).log2) * bytes : rectangle.y * 768 + rectangle.x * bytes;
						result.offset = origin + delta;
						result.format = (1u << 16) | ((std::array{0x85u, 0x9bu, 0x91u}[kind] | (morton ? 0u : 0x20u)) << 8) | (2u << 4) | 8 | 2;
						result.control0 = 1u << 31; result.control3 = morton ? 0 : 768;
						result.image_rect = (rectangle.width << 16) | rectangle.height; return result;
					};
					const auto verify = [&](id<MTLTexture> image, bool latest, clear_rectangle rectangle)
					{
						require(image.width == rectangle.width && image.height == rectangle.height && image.sampleCount == 1,
							"Mapped sample snapshot lost expanded/cropped dimensions");
						const unsigned native_bytes = kind == 1 ? 16 : 4;
						const auto output = make_buffer(ctx, image.width * image.height * native_bytes, buffer_storage::shared);
						const auto stencil = make_buffer(ctx, image.width * image.height, buffer_storage::shared);
						texture_region area; area.size = MTLSizeMake(image.width, image.height, 1);
						ctx.begin(); texture_to_buffer(ctx, image, area, output, 0, 0, 0, kind == 2 ? texture_plane::depth : texture_plane::all);
						if (kind == 2) texture_to_buffer(ctx, image, area, stencil, 0, 0, 0, texture_plane::stencil);
						ctx.submit().wait();
						const auto data = map_buffer(output, 0, output.length);
						for (unsigned y = 0; y < rectangle.height; ++y)
							for (unsigned x = 0; x < rectangle.width; ++x)
							{
								const unsigned sx = rectangle.x + x, sy = rectangle.y + y;
								const unsigned px = sx / factors.x, py = sy / factors.y;
								const unsigned sample = sx % factors.x + (sy % factors.y) * factors.x;
								const unsigned pixel = y * rectangle.width + x;
								if (kind == 0)
								{
									const std::array expected{128u + px * 3 + sample * 7, 64u + py * 5 + sample, 32u + sample * 16 + px * 4 + py + unsigned(latest) * 32, 255u};
									for (unsigned lane = 0; lane < 4; ++lane) require(data[pixel * 4 + lane] == std::byte(expected[lane]), "Mapped native MSAA ARGB sample position/channel differs");
								}
								else
								{
									const std::array expected{(latest ? .345678926f : .123456791f) + float(sample * 16 + px + py * 4) / 1024,
										-.987654328f + float(sample) / 32, 12.3456783f + float(px) / 16, 1.23456788f + float(py) / 32};
									for (unsigned lane = 0; lane < (kind == 1 ? 4u : 1u); ++lane)
									{
										std::uint32_t bits; std::memcpy(&bits, data.data() + pixel * native_bytes + lane * 4, 4);
										require(bits == std::bit_cast<std::uint32_t>(expected[lane]), "Mapped native MSAA snapshot changed sample position/native float bits");
									}
									if (kind == 2) require(static_cast<const unsigned char*>(stencil.contents)[pixel] == 17 + sample * 31 + px + py * 7 + unsigned(latest), "Mapped native MSAA stencil sample differs");
								}
							}
					};
					draw(false); const auto reads = memory->read_epochs.size();
					const auto full = words({0, 0, 8, 8});
					const auto frozen = textures.resolve_versioned(full, cache); verify(frozen.image.texture, false, {0, 0, 8, 8});
					const auto rectangle = morton ? clear_rectangle{0, 2, 4, 2} : clear_rectangle{1, 0, 3, 3};
					const auto cropped = textures.resolve_versioned(words(rectangle), cache); verify(cropped.image.texture, false, rectangle);
					draw(true); const auto latest = textures.resolve_versioned(full, cache); verify(latest.image.texture, true, {0, 0, 8, 8}); verify(frozen.image.texture, false, {0, 0, 8, 8});
					require(cache.status(source).dirty && memory->read_epochs.size() == reads && textures.statistics().uploads == 0 && textures.statistics().surface_snapshots == 3,
						"Mapped native RTT snapshot performed guest-format writeback/upload or VM pixel read");
					for (const auto& part : mapped.segments) require(memory->raw->read({part.address, part.size}).bytes == std::vector<std::byte>(part.size, std::byte{0xa5}), "Native RTT snapshot changed guest bytes");
					checked_cpu_capture(capture, [&] { require(frozen.revision.current() && cropped.revision.current() && latest.revision.current(), "Mapped native source ticket is initially stale"); });
					checked_cpu_capture(capture, [&]
					{
						io->notify({0x700000, 0x100000});
						tiles->update(14, tile_descriptor{1, 0x700000, 0x10000, 256, 0, 0, 0, true});
						require(latest.revision.current(), "Unrelated IO/tile update invalidated mapped MSAA snapshot");
					});
					cache.spill(source);
					verify(frozen.image.texture, false, {0, 0, 8, 8});
					source.reset(); cache_owner.reset();
					checked_cpu_capture(capture, [&] { require(latest.revision.current(), "Owned native source ticket lost watches after cache destruction"); });
					checked_cpu_capture(capture, [&]
					{
						if (transition == 0) { auto changed = tile; changed.bound = !tile.bound; tiles->update(0, changed); tiles->update(0, tile); }
						else if (transition == 1)
						{
							const auto& part = banked ? mapped.segments.back() : mapped.segments.front();
							memory->write({static_cast<std::uint32_t>(part.address + (banked ? part.size - 1 : 0)), 1}, std::array{std::byte{0xa5}});
						}
						else if (transition == 2) { io->notify({0x100000, 0x100000}); io->notify({0x100000, 0x100000}); }
						else if (transition == 3) *tail_address = 0x40000; // Actual EA change,without a helpful epoch notification.
						else
						{
							const auto& part = mapped.segments.back();
							memory->raw->notify_mapping_changed({static_cast<std::uint32_t>(part.address + part.size - 1), 1});
						}
					});
					checked_cpu_capture(capture, [&] { require(!latest.revision.current(), "Native RTT final ticket lost source Tile/IO/actual-bank VM dependency"); });
					verify(frozen.image.texture, false, {0, 0, 8, 8});
					verify(cropped.image.texture, false, rectangle); ++count;
				}
			}
	std::cout << "Native mapped RTT aliases:" << count << " banked/untiled linear/Morton ARGB/RGBA32F/D32S8float 1/2/4samples,independent sample_id shader values,whole/odd-cropped sample positions,exact float/depth/stencil,topEA/disjoint segments,no VM pixel read/writeback/upload,unrelated IO/tile stability,spill/cache destruction and five independent TileABA/same-value VM/IOABA/actualEA/VMmapping transitions passed;productive leases and broader composition remain open\n";
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
rsx_draw_packet mapped_sample_packet(unsigned mask)
{
	test::pass_state state;state.registers = test::executor_fixture::registers().registers;
	state.samples = mask;state.multisample = false;
	test::pass_processor processor;
	processor.descriptors[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27), (1u << 31) | (1u << 29)};
	for (float value : {-1.f, -1.f, .5f, 1.f, 3.f, -1.f, .5f, 1.f, -1.f, 3.f, .5f, 1.f})
		for (unsigned shift : {24u, 16u, 8u, 0u}) processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
	for (unsigned index : {0xffffcu, 0u, 4u})
		for (unsigned shift : {24u, 16u, 8u, 0u}) processor.index_bytes.push_back(std::byte(index >> shift));
	test::pass_layout layout;test::pass_metadata metadata;
	auto vertex = test::executor_fixture::vertex_program();auto bytes = test::executor_fixture::program(false);
	test::fragment_program fragment;fragment.pointer = bytes.data();fragment.ucode_length = bytes.size();fragment.ctrl = rsx_fragment_float32;
	const auto packet = rsx_draw_packet::capture(state, processor, layout, metadata, vertex, fragment, 0, provoking_vertex::last,
		[](const auto&, unsigned first, unsigned count)
		{ require(first == 0 && count == 9, "Mapped sample packet lost sparse indexed prefix");return std::pair<std::uint32_t, std::uint32_t>{48, 0}; });
	state.samples = ~mask;state.multisample = true;state.registers.assign(state.registers.size(), 0xffffffffu);
	processor.persistent.assign(48, std::byte{0});processor.index_bytes.assign(12, std::byte{0});
	vertex.data.assign(vertex.data.size(), 0xffffffffu);bytes.assign(bytes.size(), std::byte{0xff});
	return packet;
}
void mapped_owned_framebuffer_cases(context& ctx, const std::shared_ptr<heap_budget>& budget)
{
	unsigned count = 0;
	for (const unsigned mode : {0u, 3u, 4u, 5u})
		for (const bool banked : {false, true})
			for (const bool morton : {false, true})
				for (const bool selective : {false, true})
				{
					@autoreleasepool
					{
						const unsigned samples = mode == 0 ? 1 : mode == 3 ? 2 : 4;
						const unsigned sx = samples == 1 ? 1 : 2, sy = samples == 4 ? 2 : 1, width = 4 * sx, height = 4 * sy;
						const unsigned mask = selective ? samples == 1 ? 0 : 5 : 0xffffffffu;
						const unsigned effective = samples == 1 ? 1 : mask & ((1u << samples) - 1);
						auto memory = std::make_shared<segmented_memory>();auto io = std::make_shared<io_mapping_epoch>();auto tiles = std::make_shared<tile_state>();
						const auto capture = tile_capture_dispatch(memory->access->capture(), tiles);
						const tile_descriptor tile{1, 0xf0000, 0x40000, 768, 0, 0, 2, banked};tiles->update(0, tile);
						constexpr unsigned origin = 0xffff0;
						const auto translate = [](unsigned offset, unsigned location, std::size_t size)
						{
							require(location == 1 && size <= 0x100000 - (offset & 0xfffff), "Mapped owned framebuffer crossed one IO probe");
							return (offset < 0x100000 ? 0xfff00000 : 0x90000) + (offset & 0xfffff);
						};
						test::framebuffer_layout layout;layout.target = 2;layout.color_addresses = {0, translate(origin, 1, 1), 0, 0};layout.zeta_address = 0;
						layout.actual_color_pitch[1] = 768;layout.aa_mode = mode;layout.aa_factors = {sx, sy};layout.raster_type = morton ? 2 : 1;
						std::optional<framebuffer_snapshot> framebuffer;
						checked_cpu_capture(capture, [&]
						{
							framebuffer = framebuffer_snapshot::capture(layout, true, {}, 64 << 20, [&](auto& surface)
							{ map_framebuffer_surface(surface, origin, 1, translate, io, tiles, true, 64 << 20); });
						});
						layout.color_addresses.fill(0);layout.actual_color_pitch.fill(0);layout.aa_factors = {};
						const auto& surface = *framebuffer->colors()[0];const auto& mapped = surface.mapping->state();
						std::vector<std::byte> expected(mapped.logical.size, std::byte{0xa5});
						const auto encode = [&](std::vector<std::byte>& storage, bool changed)
						{
							std::vector<std::byte> pixels(width * height * 4);
							for (unsigned y = 0; y < height; ++y)
								for (unsigned x = 0; x < width; ++x)
								{
									const unsigned sample = x % sx + (y % sy) * sx;
									const unsigned value = changed && (effective & (1u << sample)) ? 0xff4080c0u : 0xff112233u;
									for (unsigned b = 0; b < 4; ++b) pixels[(y * width + x) * 4 + b] = std::byte(value >> ((3 - b) * 8));
								}
							if (banked) { reference(surface.guest, *mapped.tile, storage, pixels, true);return; }
							for (unsigned y = 0; y < height; ++y)
								for (unsigned x = 0; x < width; ++x)
								{
									const auto index = morton ? framebuffer_morton_oracle(x, y, width, height) * 4 : y * 768 + x * 4;
									std::copy_n(pixels.begin() + (y * width + x) * 4, 4, storage.begin() + index);
								}
						};
						encode(expected, false);
						for (const auto& part : mapped.segments) memory->write({part.address, part.size}, std::span(expected).subspan(part.offset, part.size));
						const auto initial = expected;
						const auto gather = [&]
						{
							std::vector<std::byte> bytes(mapped.logical.size);
							for (const auto& part : mapped.segments)
							{
								const auto data = memory->raw->read({part.address, part.size}).bytes;
								std::copy(data.begin(), data.end(), bytes.begin() + part.offset);
							}
							return bytes;
						};
						resolved_fragment_texture held;
						{
							surface_cache cache(ctx, memory, 64 << 20, budget, capture);
							guest_texture_cache textures(ctx, memory, translate, 64 << 20, budget, 32, texture_capture_bindings{capture, io});
							draw_executor executor(ctx, cache, textures);query_pool queries(ctx, 2, 32, budget);
							const auto targets = acquire_framebuffer(*framebuffer, cache);
							const auto packet = mapped_sample_packet(mask);
							draw_execution_state execution;execution.queries = &queries;
							queries.begin(0, true);executor.execute(packet, *framebuffer, execution);queries.end(0);
							require(queries.consume(0) == 16u * std::popcount(effective), "Mapped owned MSAA draw query lost selected native samples");
							const auto reads = memory->read_epochs.size();
							fragment_texture_registers words;words.offset = origin;
							words.format = (1u << 16) | ((morton ? 0x85u : 0xa5u) << 8) | (2u << 4) | 8 | 2;
							words.control0 = 1u << 31;words.control3 = morton ? 0 : 768;words.image_rect = (width << 16) | height;
							const auto image = textures.resolve_versioned(words, cache);held = image.image;
							require(held.texture.width == width && held.texture.height == height && memory->read_epochs.size() == reads &&
								cache.status(targets.colors[0]).dirty && gather() == initial && textures.statistics().uploads == 0,
								"Mapped owned MSAA RTT changed guest bytes,read stale RAM or lost expanded extent");
							checked_cpu_capture(capture, [&] { require(image.revision.current(), "Fresh owned mapped MSAA snapshot has stale source ticket"); });
							encode(expected, true);cache.flush(targets.colors[0]);
							const auto actual = gather();
							// The top-EA entry contains 16 bytes. A 32-byte active MSAA
							// row therefore splits across two physical entries,followed by
							// a separate pitched row group:three CAS parts in one batch.
							const unsigned expected_parts = !banked && !morton && samples > 1 ? 3 : 2;
							if (actual != expected || memory->batch_size != expected_parts || cache.status(targets.colors[0]).dirty)
							{
								const auto difference = std::mismatch(actual.begin(), actual.end(), expected.begin());
								std::cerr << "Mapped owned writeback mismatch:mode=" << mode << " banked=" << banked << " morton=" << morton << " selective=" << selective
									<< " batch=" << memory->batch_size << " expected_parts=" << expected_parts << " segments=" << mapped.segments.size() << " dirty=" << cache.status(targets.colors[0]).dirty;
								if (difference.first != actual.end()) std::cerr << " byte=" << difference.first - actual.begin() << " actual=" << unsigned(*difference.first) << " expected=" << unsigned(*difference.second);
								std::cerr << '\n';throw std::runtime_error("Mapped owned MSAA draw lost atomic segmented writeback/sample words/padding");
							}
							checked_cpu_capture(capture, [&] { require(!image.revision.current(), "Owned mapped MSAA snapshot ticket ignored completed guest writeback"); });
							cache.spill(targets.colors[0]);
						}
						// Old native output stays immutable after actual guest mutation and teardown.
						for (const auto& part : mapped.segments) memory->write({part.address, part.size}, std::vector<std::byte>(part.size, std::byte{0}));
						const auto output = make_buffer(ctx, width * height * 4, buffer_storage::shared);
						ctx.begin();texture_to_buffer(ctx, held.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(width, height, 1)}, output, 0);ctx.submit().wait();
						for (unsigned y = 0; y < height; ++y)
							for (unsigned x = 0; x < width; ++x)
							{
								const unsigned sample = x % sx + (y % sy) * sx;
								const auto values = effective & (1u << sample) ? std::array{192u, 128u, 64u, 255u} : std::array{51u, 34u, 17u, 255u};
								for (unsigned b = 0; b < 4; ++b) require(static_cast<const unsigned char*>(output.contents)[(y * width + x) * 4 + b] == values[b],
									"Mapped owned MSAA texture sample changed after source writeback/spill/VM mutation/cache teardown");
							}
						++count;
					}
				}
	require(count == 32, "Mapped owned MSAA matrix omitted a framebuffer/mask/storage profile");
	std::cout << "Native mapped owned MSAA:32 GPU cases,AA0/3/4/5,B-only banked/untiled linear/Morton,topEA/disjoint IO,poisoned owned RSX packets,full/selective masks with disabled hint and single-sample zero-mask exception,exact query counts,native RTT sample positions,atomic guest words/whole-allocation padding,post-writeback stale tickets and output lifetime through spill/VM mutation/cache teardown passed\n";
}
void rounded_framebuffer_cases(context& ctx, const std::shared_ptr<heap_budget>& budget)
{
	std::vector<surface_format> formats;
	for (auto value : {surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565, surface_color::xrgb8_zero,
		surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8, surface_color::rgba16f,
		surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero, surface_color::xbgr8_one, surface_color::abgr8}) formats.emplace_back(value);
	for (auto value : {surface_depth::z16_uint, surface_depth::z24s8_uint, surface_depth::z16_float, surface_depth::z24s8_float}) formats.emplace_back(value);
	unsigned count = 0;
	for (const auto& format : formats)
		for (bool banked : {false, true})
			for (bool larger : {false, true})
			{
				@autoreleasepool
				{
					const auto* color = std::get_if<surface_color>(&format);
					const auto bytes = color ? color_layout(*color).pixel_bytes : framebuffer_depth_bytes(std::get<surface_depth>(format));
					auto memory = std::make_shared<segmented_memory>(); auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
					const tile_descriptor tile{1, 0xf0000, 0x40000, 768, 0, 0, 2, banked}; if (banked) tiles->update(0, tile);
					const auto translate = [](unsigned offset, unsigned location, std::size_t size)
					{
						require(location == 1 && (offset >> 20) <= 1 && size <= 0x100000 - (offset & 0xfffff), "Rounded framebuffer escaped main IO");
						return (offset < 0x100000 ? 0xfff00000u : 0x90000u) + (offset & 0xfffff);
					};
					test::framebuffer_layout layout; layout.width = 7; layout.height = 5; layout.raster_type = 2;
					layout.color_addresses.fill(0); layout.actual_color_pitch.fill(0); layout.zeta_address = layout.actual_zeta_pitch = 0;
					const auto pitch = static_cast<unsigned>(layout.width * bytes), window_x = static_cast<unsigned>(16 / bytes);
					if (color) { layout.target = 2; layout.color_format = static_cast<unsigned>(*color); layout.actual_color_pitch[1] = pitch; }
					else { layout.target = 0; layout.depth_format = static_cast<unsigned>(std::get<surface_depth>(format)); layout.zeta_pitch = layout.actual_zeta_pitch = pitch; }
					const auto presence = color ? framebuffer_window_zero_colors(layout) : framebuffer_window_zero_depth(layout, 0xfffffff0, window_x, 0, pitch, pitch);
					require(presence == (color ? 2u : 16u), "Rounded framebuffer lost analysed-zero color/depth presence");
					const auto storage = larger ? std::optional<std::array<std::size_t, 3>>{{16, 16, 1}} : std::nullopt;
					const auto framebuffer = framebuffer_snapshot::capture(layout, true, storage, 64 << 20, [&](auto& surface)
						{ map_framebuffer_window_surface(surface, 0xffff0, 1, window_x, 0, pitch, translate, io, tiles, true, 64 << 20); }, presence);
					const auto& surface = color ? *framebuffer.colors()[0] : *framebuffer.depth(); const auto& mapped = surface.mapping->state();
					require(surface.address == 0x90000 && surface.morton->storage == std::array<std::size_t, 3>{larger ? 16u : 8u, larger ? 16u : 8u, 1}, "Rounded framebuffer storage or actual EA differs");
					std::vector<std::byte> pixels(layout.width * layout.height * bytes);
					for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte((i * 29 + i / 17 + 71) & 255);
					std::vector<std::byte> encoded(mapped.logical.size, std::byte{0xa5});
					const auto encode = [&](auto& target, auto& values)
					{
						if (banked) { reference(surface.guest, *mapped.tile, target, values, true); return; }
						for (unsigned y = 0; y < layout.height; ++y) for (unsigned x = 0; x < layout.width; ++x)
						{
							const auto index = framebuffer_morton_oracle(x, y, surface.morton->storage[0], surface.morton->storage[1]) * bytes;
							std::copy_n(values.begin() + (y * layout.width + x) * bytes, bytes, target.begin() + index);
						}
					};
					encode(encoded, pixels); for (const auto& part : mapped.segments) memory->write({part.address, part.size}, std::span(encoded).subspan(part.offset, part.size));
					const auto gather = [&]
					{
						std::vector<std::byte> result(mapped.logical.size);
						for (const auto& part : mapped.segments) { const auto data = memory->raw->read({part.address, part.size}).bytes; std::copy(data.begin(), data.end(), result.begin() + part.offset); }
						return result;
					};
					memory->write({0x100000, pixels.size()}, pixels);
					surface_cache cache(ctx, memory, 64 << 20, budget, tile_capture_dispatch(memory->access->capture(), tiles));
					const auto targets = acquire_framebuffer(framebuffer, cache); const auto source = color ? targets.colors[0] : targets.depth;
					const surface_allocation_description allocation{layout.width, layout.height, 1, format, false};
					const auto baseline = cache.acquire({0x100000, allocation, layout.width * bytes, {}}), expected_linear = cache.acquire({0x110000, allocation, layout.width * bytes, {}}), actual_linear = cache.acquire({0x120000, allocation, layout.width * bytes, {}});
					const clear_rectangle rectangle{0, 0, layout.width, layout.height};
					cache.copy(baseline, rectangle, expected_linear, 0, 0); cache.flush(expected_linear);
					cache.copy(source, rectangle, actual_linear, 0, 0); cache.flush(actual_linear);
					require(memory->raw->read({0x110000, pixels.size()}).bytes == memory->raw->read({0x120000, pixels.size()}).bytes, "Rounded framebuffer native gather differs from independent linear codec");
					if (!banked)
					{
						const auto generation = cache.status(source).content_generation; const auto last = mapped.logical.size - 1;
						const auto part = std::find_if(mapped.segments.begin(), mapped.segments.end(), [&](const auto& entry) { return last >= entry.offset && last - entry.offset < entry.size; });
						require(part != mapped.segments.end(), "Rounded framebuffer padding lacks actual segment");
						memory->write({static_cast<unsigned>(part->address + last - part->offset), 1}, std::array{std::byte{0x6b}}); encoded[last] = std::byte{0x6b}; cache.prepare(source);
						require(cache.status(source).content_generation == generation, "Inactive rounded Morton padding invalidated native content");
					}
					const unsigned mask = color ? 0xf0 : describe_surface_depth(std::get<surface_depth>(format)).stencil ? 3 : 1;
					cache.clear(baseline, mask, 0xff123456, 0x62345678, 255, rectangle); cache.flush(baseline);
					cache.clear(source, mask, 0xff123456, 0x62345678, 255, rectangle); cache.flush(source);
					auto cleared = memory->raw->read({0x100000, pixels.size()}).bytes; auto expected = encoded; encode(expected, cleared);
					require(gather() == expected, "Rounded framebuffer writeback damaged active pixels or inactive storage padding");
					++count;
				}
			}
	require(count == 72, "Rounded framebuffer matrix omitted color/depth/bank/storage");
	std::cout << "Native rounded Morton framebuffers:72 GPU color/depth cases,14color+4depth,7x5 active/default8x8-or-explicit16x16,banked/untiled,logical window with analysed-zero physical B/depth origins,independent linear codec and rectangular byte oracle,native gather/clear/writeback,whole-allocation padding preservation and untiled inactive-write stability passed;productive leases and PS3 hardware parity remain open\n";
}
void banked_display_projection_cases()
{
	std::size_t cases=0,bytes=0;
	for(unsigned base:{0x50000u,0xffff0000u})for(unsigned pitch:{256u,768u,1536u})for(unsigned bank=0;bank<4;++bank)
	for(unsigned bpp:{1u,2u,4u,8u,16u})for(bool morton:{false,true})
	{
		const unsigned width=8,height=morton?5:65;const auto swizzle=morton?std::optional<swizzle_region>{{{16,16,1},{3,7,0},{width,height,1}}}:std::nullopt;
		const banked_tile_layout tile{base,0x40000,pitch,bank,morton?5*bpp:pitch*3+bpp};
		const auto geometry=plan_guest_surface(0,width,height,bpp,morton?0:pitch,swizzle);
		auto image=geometry;image.range={tile.address,tile.size};std::vector<std::byte> mask(tile.size),ones(width*height*bpp,std::byte{1});reference(image,tile,mask,ones,true);
		const auto runs=banked_display_storage_ranges(geometry,tile);std::vector<std::byte> actual(tile.size);
		for(auto run:runs)for(unsigned byte=0;byte<run.size;++byte){require(actual[run.address+byte]==std::byte{0},"Banked projection overlaps itself");actual[run.address+byte]=std::byte{1};++bytes;}
		require(actual==mask,"Banked active projection differs from unchanged shared RSX oracle");
		// Reverse/disjoint actual EAs,including wrapped top32-bit storage.
		const std::vector<guest_storage_segment> segments{{base,0,0x10000},{0x90000,0x10000,0x20000},{0x40000,0x30000,0x10000}};
		const display_storage_source source{geometry,segments,true,1,tile};const auto parts=project_display_storage_source(source);std::vector<std::byte> projected(tile.size);
		for(const auto& part:parts)
		{
			const auto found=std::find_if(segments.begin(),segments.end(),[&](auto p){return part.offset>=p.offset&&part.offset-p.offset<p.size;});require(found!=segments.end(),"Banked projection lost its storage segment");
			require(part.coverage.range.address==found->address+part.offset-found->offset&&part.coverage.range.size<=found->size-(part.offset-found->offset),"Banked projection used fictitious base-plus-offset EA");
			for(unsigned byte=0;byte<part.coverage.range.size;++byte)projected[part.offset+byte]=std::byte{1};
		}
		require(projected==mask,"Banked segmented projection omitted active bytes orselected padding");++cases;
	}
	require(cases==240,"Banked projection matrix incomplete");
	std::cout<<"Banked display CPU projection:"<<cases<<" shared-RSX byte-oracle profiles/"<<bytes<<" active bytes,1-2-4-8-16B/all4banks/threepitchfactors/linear65row andMortoncrop/top32-bit reverse-disjoint segments;exactnative coverage,no padding orfictitious EA passed\n";
	const banked_tile_layout tile{0x10000,0x10000,768,2,4};const auto geometry=plan_guest_surface(0,80,85,4,768);auto image=geometry;image.range={tile.address,tile.size};
	std::vector<std::byte> mask(tile.size),ones(80*85*4,std::byte{1}),actual(tile.size);reference(image,tile,mask,ones,true);
	for(auto run:banked_display_storage_ranges(geometry,tile))std::fill_n(actual.begin()+run.address,run.size,std::byte{1});
	const auto owned=std::count(mask.begin(),mask.end(),std::byte{1});require(actual==mask&&owned<ones.size(),"Banked display assigned out-of-allocation cells physical ownership");
	unsigned guards=0;const auto reject=[&](const auto& work){bool refused=false;try{work();}catch(const std::exception&){refused=true;}require(refused,"Invalid banked display geometry accepted");++guards;};
	reject([&]{auto g=geometry;g.range.address=1;(void)banked_display_storage_ranges(g,tile);});
	reject([&]{auto t=tile;t.offset=5;(void)banked_display_storage_ranges(geometry,t);});
	reject([&]{auto t=tile;t.pitch=769;(void)banked_display_storage_ranges(geometry,t);});
	reject([&]{auto t=tile;t.bank=4;(void)banked_display_storage_ranges(geometry,t);});
	reject([&]{(void)banked_display_storage_ranges(geometry,tile,0);});
	reject([&]{const auto g=plan_guest_surface(0,3,3,4,0,swizzle_region{{4,4,2},{0,0,1},{3,3,1}});(void)banked_display_storage_ranges(g,tile);});
	reject([&]{(void)project_display_storage_source({geometry,{{0x20000,0,tile.size}},true,1,tile});});
	reject([&]{(void)project_display_storage_source({geometry,{{tile.address,0,tile.size-1}},true,1,tile});});
	std::cout<<"Banked display bounds:shared RSX out-of-allocation skip matches"<<owned<<" owned bytes/"<<ones.size()<<" logical bytes;no fabricated native ownership orzero pixels;"<<guards<<" relative-origin/alignment/pitch/bank/runbudget/3D/base/incomplete-segment guards passed\n";
}
constexpr std::string_view display_reader=R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void banked_display_read(texture2d<float,access::read> input [[texture(0)]],device uint4* output [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {
 if(p.x<input.get_width()&&p.y<input.get_height())output[p.y*input.get_width()+p.x]=as_type<uint4>(input.read(p));
}
)metal";
std::vector<std::array<unsigned,4>> banked_display_values(context& ctx,id<MTLComputePipelineState> reader,const present_image& image)
{
	const auto width=image.texture.width,height=image.texture.height;const auto buffer=make_buffer(ctx,width*height*16,buffer_storage::shared);
	ctx.begin();bindings resources;resources.set_texture(0,image.texture);resources.set_buffer(0,buffer,0);ctx.keep_alive(reader);
	const auto table=resources.snapshot(ctx);const auto encoder=ctx.compute();[encoder setComputePipelineState:reader];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(width,height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];ctx.submit().wait();
	const auto bytes=map_buffer(buffer,0,buffer.length);std::vector<std::array<unsigned,4>> values(width*height);std::memcpy(values.data(),bytes.data(),bytes.size());return values;
}
void mapped_banked_display_cases()
{
	std::size_t profiles=0;
	for(bool morton:{false,true})for(unsigned mode=0;mode<8;++mode)
	{
		context ctx{3};auto budget=std::make_shared<heap_budget>(ctx,8u<<20);ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			helper_catalog compile(ctx);const auto library=compile.library(display_reader,@"Mapped bankdisplay reader");const auto reader=compile.compute(library,@"banked_display_read");
			auto memory=std::make_shared<segmented_memory>();auto io=std::make_shared<io_mapping_epoch>();auto tiles=std::make_shared<tile_state>();const auto capture=tile_capture_dispatch(memory->access->capture(),tiles);
			const tile_descriptor tile{1,0xf0000,0x40000,768,0,0,2,true};tiles->update(0,tile);
			const auto translate=[](unsigned offset,unsigned location,std::size_t size)
			{
				if(!location)return offset;require(size<=0x100000-(offset&0xfffff),"Mapped bankdisplay crossed IOentry");return (offset<0x100000?0xfff00000u:0x90000u)+(offset&0xfffff);
			};
			const auto swizzle=morton?std::optional<swizzle_region>{{{16,16,1},{1,1,0},{8,6,1}}}:std::nullopt;
			const auto geometry=plan_guest_surface(0,8,6,4,morton?0:768,swizzle);std::shared_ptr<const surface_mapping_revision> mapping;
			checked_cpu_capture(capture,[&]{mapping=capture_surface_mapping(geometry,0x100000+772,1,translate,io,tiles,true,4u<<20);});const auto mapped=mapping->state();
			std::vector<std::byte> initial(mapped.logical.size,std::byte{0xa7});for(auto part:mapped.segments)memory->write({part.address,part.size},std::span(initial).subspan(part.offset,part.size));
			surface_cache cache(ctx,memory,4u<<20,budget,capture);display_resolver display(ctx,memory,translate,budget,4u<<20,capture,tiles);
			const auto source=cache.acquire({mapped.pixel_address,{4,3,4,surface_color::argb8,false},morton?0u:768u,swizzle,mapped.tile,mapped.segments,mapping});cache.clear(source,0xf0,0x00112233,0,255,{0,0,8,6});
			auto image=geometry;image.range={mapped.tile->address,mapped.tile->size};std::vector<std::byte> expected=initial,mask(initial.size()),pixels(8*6*4),ones(pixels.size(),std::byte{1});
			for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte(std::array{0u,17u,34u,51u}[i%4]);reference(image,*mapped.tile,expected,pixels,true);reference(image,*mapped.tile,mask,ones,true);
			std::size_t selected=0;guest_storage_segment selected_part{};
			for(const auto part:mapped.segments)for(std::size_t offset=part.offset+1;offset+64<part.offset+part.size;++offset)if(mask[offset]!=std::byte{0}){selected=offset-1;selected_part=part;break;}
			require(selected&&selected_part.size,"Mapped bankdisplay lacks acontained AVwindow");const unsigned address=selected_part.address+selected-selected_part.offset;
			for(auto part:project_display_storage_source({geometry,mapped.segments,true,1,mapped.tile}))memory->display_forbidden.push_back(part.coverage.range);
			const display_snapshot descriptor{address,3,3,13};const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};
			auto job=display.prepare(descriptor,cache,std::array{display_upload_dependency{gate,1}});require(!job.ready()&&display.statistics().assembly_casts==1,"Mapped main bankdisplay gate absent");
			bool refused=false;try{(void)job.finish(std::chrono::milliseconds(1));}catch(const std::exception&){refused=true;}require(refused,"Mapped bankdisplay deadline ignored");
			if(mode==1)io->notify({0x700000,0x100000});
			if(mode==2){io->notify({0x100000,0x100000});io->notify({0x100000,0x100000});}
			if(mode==3)memory->raw->notify_mapping_changed({mapped.segments.back().address,4});
			if(mode==4){auto changed=tile;changed.bank^=1;tiles->update(0,changed);tiles->update(0,tile);}
			if(mode==5){const auto tail=mapped.segments.back();memory->write({static_cast<unsigned>(tail.address+tail.size-1),1},std::array{std::byte{0xa7}});}
			if(mode==6){tiles->update(1,{0,0,0x10000,256,0,0,1,true});tiles->update(1,{});} // Source ismain;AV windowlies elsewhere.
			gate.signaledValue=1;const bool accepted=mode==0||mode==1||mode==6||mode==7;
			std::optional<present_image> output;if(accepted){if(mode!=7)output=job.finish();}else{refused=false;try{(void)job.finish();}catch(const std::exception&){refused=true;}require(refused,"Stale mapped main bankdisplay published");}
			const auto verify=[&](const present_image& value,const std::vector<std::byte>& bytes)
			{
				const auto actual=banked_display_values(ctx,reader,value);for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)for(unsigned c=0;c<4;++c)
				{
					const float wanted=c==3?1.f:unsigned(bytes[selected+y*13+x*4+c+1])/255.f;require(std::abs(std::bit_cast<float>(actual[y*3+x][c])-wanted)<=1e-7f,"Mapped bankdisplay differs from independent physical-byte oracle");
				}
			};
			if(output)
			{
				verify(*output,expected);
			}
			if(mode==7)
			{
					ctx.wait_idle(); // Complete old GPU work without publishing its cache key.
					cache.clear(source,0xf0,0x00778899,0,255,{0,0,8,6});for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte(std::array{0u,119u,136u,153u}[i%4]);auto latest=initial;reference(image,*mapped.tile,latest,pixels,true);
					const auto fresh=display.prepare(descriptor,cache).finish();verify(fresh,latest);verify(job.finish(),expected);require(display.prepare(descriptor,cache).finish().texture==fresh.texture,"Late mapped bankdisplay evicted newer writer");
			}
			require(cache.status(source).dirty&&!guest_memory_epoch::active(),"Mapped bankdisplay lostnative authority orCPU exclusion");display.clear();cache.clear_snapshots();(void)ctx.trim_upload_pages();++profiles;
		}
		require(budget->statistics().reserved_bytes==0,"Mapped bankdisplay leaked ledger owners");
	}
	std::cout<<"Mapped main bankdisplay:"<<profiles<<" linear/Morton4sample cases,original main IO/tiletickets/top32-bit disjoint EAsegments,gated timeout,unrelated IO/localtiles accepted;IOABA/mainbankABA/actualEA mappingchange/wholeallocation samevaluewrite refuse stalejobs,new nativewriter/immutableoldpixels cannot evict newreadykey,dirtyauthority/CPU exclusion/allledger retirement passed\n";
}
void banked_display_cases(context& ctx,const std::shared_ptr<heap_budget>& budget)
{
	helper_catalog compile(ctx);const auto library=compile.library(display_reader,@"Banked display test reader");const auto reader=compile.compute(library,@"banked_display_read");
	surface_color_transfer color(ctx);sample_transfer samples(ctx);std::size_t cases=0,channels=0,captured=0;
	const std::array formats{surface_color::x1rgb5_zero,surface_color::x1rgb5_one,surface_color::rgb565,surface_color::xrgb8_zero,surface_color::xrgb8_one,surface_color::argb8,surface_color::b8,surface_color::gb8,surface_color::rgba16f,surface_color::rgba32f,surface_color::r32f,surface_color::xbgr8_zero,surface_color::xbgr8_one,surface_color::abgr8};
	for(auto format:formats)for(unsigned count:{1u,2u,4u})for(unsigned pitch:{256u,768u,1536u})for(unsigned bank=0;bank<4;++bank)for(bool morton:{false,true})
	{
		@autoreleasepool
		{
			auto memory=std::make_shared<display_memory>();auto access=memory->access;access->mirror=true;
			const cpu_capture_dispatch capture=[access](const auto& work){access->execute([&](auto&){work();});};
			surface_cache cache(ctx,memory,4u<<20,budget,capture);display_resolver display(ctx,memory,[](unsigned o,unsigned,std::size_t){return o;},budget,4u<<20,capture);
			const unsigned bpp=color_layout(format).pixel_bytes,width=32/bpp,height=6;const auto scale=count==1?sample_layout{1,1}:rsx_sample_layout(count);
			const auto swizzle=morton?std::optional<swizzle_region>{{{width*2,8,1},{1,1,0},{width,height,1}}}:std::nullopt;
			const banked_tile_layout tile{0x50000,0x40000,pitch,bank,morton?pitch*64+5*bpp:pitch*63+bpp};
			const auto geometry=plan_guest_surface(0,width,height,bpp,morton?0:pitch,swizzle);auto image=geometry;image.range={tile.address,tile.size};
			std::vector<std::byte> initial(tile.size,std::byte{0xa7});memory->write({tile.address,tile.size},initial);
			const auto source=cache.acquire({tile.address+tile.offset,{width/scale.x,height/scale.y,count,format,false},morton?0u:pitch,swizzle,tile});cache.prepare(source,surface_initialization::background);
			std::vector<std::byte> pixels(width*height*bpp);for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte((i*37+i/7+19)%256);
			if(format==surface_color::rgba16f)for(unsigned i=0;i<pixels.size();i+=2){const unsigned word=0x3100+(i/2%11)*0x100;pixels[i]=std::byte(word>>8);pixels[i+1]=std::byte(word);}
			if(format==surface_color::rgba32f||format==surface_color::r32f)for(unsigned i=0;i<pixels.size();i+=4){const unsigned word=0x3e100000+(i/4%7)*0x100000;for(unsigned c=0;c<4;++c)pixels[i+c]=std::byte(word>>(24-c*8));}
			const std::array uses{cached_surface_use{source,true}};cache.draw(uses,[&](context& commands,auto images)
			{
				const auto input=make_upload(commands,pixels);texture_region region;region.size=MTLSizeMake(width,height,1);
				if(count==1)color.upload(commands,format,input,{},images[0],region);
				else
				{
					auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:describe_surface_color(format).format width:width height:height mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
					const auto expanded=make_texture(commands,d);color.upload(commands,format,input,{},expanded,region);samples.import(commands,expanded,images[0]);
				}
			});
			std::vector<std::byte> expected=initial,mask(tile.size),ones(pixels.size(),std::byte{1});reference(image,tile,expected,pixels,true);reference(image,tile,mask,ones,true);
			for(unsigned pos=0;pos<tile.size;)
			{
				if(mask[pos]==std::byte{0}){++pos;continue;}const auto first=pos;while(pos<tile.size&&mask[pos]!=std::byte{0})++pos;memory->forbidden.push_back({tile.address+first,pos-first});
			}
			require(!memory->forbidden.empty(),"Banked display hasno active bytes");const unsigned first=memory->forbidden.front().address-tile.address,offset=first?first-1:first;
			for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
			{
				display.clear();memory->reads.clear();const unsigned avbytes=display_pixel_bytes(av),row=3*avbytes+1;
				const display_snapshot descriptor{tile.address+offset,3,3,row,av};const auto before=display.statistics();const auto output=display.prepare(descriptor,cache).finish();
				const auto verify=[&](const present_image& value)
				{
					const auto buffer=make_buffer(ctx,3*3*16,buffer_storage::shared);ctx.begin();bindings resources;resources.set_texture(0,value.texture);resources.set_buffer(0,buffer,0);ctx.keep_alive(reader);
					const auto table=resources.snapshot(ctx);const auto encoder=ctx.compute();[encoder setComputePipelineState:reader];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(3,3,1) threadsPerThreadgroup:MTLSizeMake(3,3,1)];ctx.submit().wait();
					const auto data=map_buffer(buffer,0,buffer.length);std::array<std::array<unsigned,4>,9> values;std::memcpy(values.data(),data.data(),data.size());
					for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)for(unsigned c=0;c<4;++c)
					{
						const auto at=offset+y*row+x*avbytes;float wanted=1;
						if(c<3){if(av==display_format::rgbx16f){const auto h=std::uint16_t((unsigned(expected[at+c*2])<<8)|unsigned(expected[at+c*2+1]));wanted=float(std::bit_cast<_Float16>(h));}else wanted=unsigned(expected[at+(av==display_format::xrgb8?c+1:3-c)])/255.f;}
						const auto bits=values[y*3+x][c];const float actual=std::bit_cast<float>(bits);const bool equal=std::isnan(wanted)?std::isnan(actual):av==display_format::rgbx16f||c==3?bits==std::bit_cast<unsigned>(wanted):std::abs(actual-wanted)<=1e-7f;
						require(equal,"Banked native display pixels differ from shared RSX bank/raw BE-byte oracle");++channels;
					}
				};verify(output);
				std::size_t missing=0,reads=0;for(unsigned y=0;y<3;++y)for(unsigned byte=0;byte<3*avbytes;++byte)missing+=mask[offset+y*row+byte]==std::byte{0};
				for(auto run:memory->reads)for(unsigned byte=0;byte<run.size;++byte){const auto at=run.address+byte-tile.address,relative=at-offset;require(at>=offset&&mask[at]==std::byte{0}&&relative/row<3&&relative%row<3*avbytes,"Banked display captured native bytes orAV padding");++reads;}
				require(missing==reads&&display.statistics().captured_vm_bytes-before.captured_vm_bytes==missing&&display.statistics().assembly_casts==before.assembly_casts+1,"Banked display exactcapture accounting differs");captured+=missing;
				const auto allocations=budget->statistics().created,encoded=ctx.encoding_statistics().compute_encoders;const auto count_reads=memory->reads.size();
				require(display.prepare(descriptor,cache).finish().texture==output.texture&&budget->statistics().created==allocations&&ctx.encoding_statistics().compute_encoders==encoded&&memory->reads.size()==count_reads,"Warm banked display allocated/encoded/captured");
				auto alias=descriptor;alias.offset+=0x200000;verify(display.prepare(alias,cache).finish());++cases;
			}
			require(cache.status(source).dirty,"Banked display lostdirty authority");memory->forbidden.clear();require(memory->read({tile.address,tile.size}).bytes==initial,"Banked display wroteback native pixels ormodified VMpadding");
		}
	}
	require(cases==3024,"Banked display GPUmatrix incomplete");
	std::cout<<"Native banked display:"<<cases<<" primary+physicalalias profiles/"<<channels<<" channels,14colors/1-2-4samples/all4banks/threepitchfactors/linear64rowcrossing andMortoncrops/threeAV,partial sourcebytes/oddAVpitch;"<<captured<<" exactprimary VMholebytes,no native orpadding reads,warm zeroallocation/capture/encoding,VM unchanged passed\n";
}
void logical_banked_texture_cases(context& ctx,const std::shared_ptr<heap_budget>& budget)
{
	surface_color_transfer colors(ctx);sample_transfer sample_copy(ctx);std::size_t profiles=0,planes=0,bytes=0,holes=0,mixed=0,depth_words=0;
	for(unsigned format:{0x85u,0x86u,0x87u,0x88u,0x8du,0x8eu,0x9bu,0x91u,0x93u})
	for(auto dimension:{vertex_texture_dimension::two,vertex_texture_dimension::cube,vertex_texture_dimension::three})
	for(unsigned pitch:{256u,768u})for(unsigned bank:{0u,3u})for(bool morton:{false,true})for(unsigned count:{1u,4u})
	{
		try { @autoreleasepool
		{
			auto memory=std::make_shared<segmented_memory>();auto io=std::make_shared<io_mapping_epoch>();auto tiles=std::make_shared<tile_state>();
			const auto capture=tile_capture_dispatch(memory->access->capture(),tiles);
			const bool depth=format==0x91||format==0x93;const unsigned bpp=format==0x9b?16:format==0x93?2:4,origin=0x100000-4*bpp;const float native_depth=.345678926f;
			const tile_descriptor tile_desc{1,0xf0000,0x40000,pitch,0,0,bank,true};tiles->update(0,tile_desc);
			const auto tail_address=std::make_shared<unsigned>(0x90000);const auto translate=[tail_address](unsigned offset,unsigned location,std::size_t size)
			{
				require(location==1&&size<=0x100000-(offset&0xfffff),"Logical RTT crossed actual IO entry");
				return (offset<0x100000?0xfff00000u:*tail_address)+(offset&0xfffff);
			};
			const auto swizzle=morton?std::optional<swizzle_region>{{{8,8,1},{1,1,0},{4,4,1}}}:std::nullopt;
			const auto geometry=plan_guest_surface(0,4,4,bpp,morton?0:pitch,swizzle);
			std::shared_ptr<const surface_mapping_revision> mapping;checked_cpu_capture(capture,[&]{mapping=capture_surface_mapping(geometry,origin,1,translate,io,tiles,true,4u<<20);});
			const auto& mapped=mapping->state();const std::vector<std::byte> initial(mapped.logical.size,std::byte{0x11});
			for(const auto& part:mapped.segments)memory->write({part.address,part.size},std::span(initial).subspan(part.offset,part.size));
			surface_cache surfaces(ctx,memory,4u<<20,budget,capture);guest_texture_cache textures(ctx,memory,translate,4u<<20,budget,64,{capture,io});
			const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);const surface_format sf=depth?surface_format{format==0x91?surface_depth::z24s8_float:surface_depth::z16_float}:surface_format{format==0x9b?surface_color::rgba32f:surface_color::argb8};
			const auto source=surfaces.acquire({mapped.pixel_address,{4/factors.x,4/factors.y,count,sf,false},morton?0u:pitch,swizzle,mapped.tile,mapped.segments,mapping});surfaces.prepare(source,surface_initialization::background);
			std::vector<std::byte> pixels(4*4*bpp);
			for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte((i*3+i/7+23)&255);
			if(format==0x9b)for(unsigned i=0;i<pixels.size();i+=4){const unsigned word=0x3e000000+(i/4%4)*0x800000;for(unsigned c=0;c<4;++c)pixels[i+c]=std::byte(word>>(24-c*8));}
			if(depth)for(unsigned i=0;i<pixels.size();i+=bpp){const unsigned bits=std::bit_cast<unsigned>(native_depth),word=format==0x91?((bits>>7)<<8)|53:((bits-(120u<<23))>>11)&65535;for(unsigned c=0;c<bpp;++c)pixels[i+c]=std::byte(word>>((bpp-1-c)*8));}
			surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images)
			{
				if(depth){auto pass=[MTL4RenderPassDescriptor new];pass.depthAttachment.texture=images[0];pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=native_depth;if(format==0x91){pass.stencilAttachment.texture=images[0];pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=53;}(void)commands.render(pass);return;}
				id<MTLTexture> expanded=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat width:4 height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;expanded=make_texture(commands,d);}
				texture_region area;area.size=MTLSizeMake(4,4,1);colors.upload(commands,std::get<surface_color>(sf),make_upload(commands,pixels),{},expanded,area);if(count>1)sample_copy.import(commands,expanded,images[0]);
			});
			auto expected=initial;std::vector<std::byte> mask(initial.size()),logical_mask(initial.size()),ones(pixels.size(),std::byte{1});auto image=geometry;image.range={mapped.tile->address,mapped.tile->size};
			reference(image,*mapped.tile,expected,pixels,true);reference(image,*mapped.tile,mask,ones,true);
			const auto physical_mask=mask;
			// Original logical RTT pixels override this source's raw physical
			// interpretation. The bank reference above remains an independent
			// shared RSX oracle;no production assembly planner builds expected bytes.
			for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)for(unsigned c=0;c<bpp;++c)
			{
				const auto raw=morton?morton_index({x+1,y+1,0},plan_swizzle(*swizzle,bpp).log2)*bpp:y*pitch+x*bpp;
				const auto at=mapped.tile->offset+raw+c;expected.at(at)=pixels[(y*4+x)*bpp+c];mask.at(at)=std::byte{1};logical_mask.at(at)=std::byte{1};
			}
			for(const auto& part:mapped.segments)for(std::size_t at=part.offset,end=at+part.size;at<end;)
			{
				if(mask[at]==std::byte{0}){++at;continue;}const auto first=at;while(at<end&&mask[at]!=std::byte{0})++at;memory->display_forbidden.push_back({static_cast<unsigned>(std::uint64_t(part.address)+first-part.offset),at-first});
			}
			fragment_texture_registers words;words.offset=origin-(depth?bpp:1);words.format=(2u<<16)|((format|0x20u)<<8)|(unsigned(dimension==vertex_texture_dimension::three?3:2)<<4)|(dimension==vertex_texture_dimension::cube?4:0)|8|2;words.control0=1u<<31;words.control3=(dimension==vertex_texture_dimension::three?4u<<20:0)|pitch;words.image_rect=(8u<<16)|8;
			const auto plan=plan_guest_texture(words);const bool vtc=guest_texture_vtc(plan),volume=guest_texture_depth_volume(plan);require(plan.storage_bytes<expected.size()-mapped.tile->offset,"Logical RTT test exceeds tile fixture");
			// A separate VM-only native upload is the decode oracle. It has no
			// cached surfaces or mapping tickets and cannot use the new planner.
			auto reference_memory=std::make_shared<segmented_memory>();for(const auto& part:mapped.segments)reference_memory->write({part.address,part.size},std::span(expected).subspan(part.offset,part.size));
			guest_texture_cache reference_cache(ctx,reference_memory,translate,4u<<20,budget,64,{reference_memory->access->capture(),io});const auto oracle=reference_cache.resolve(words);
			memory->display_reads.clear();const auto result=textures.resolve_versioned(words,surfaces);checked_cpu_capture(capture,[&]{require(result.revision.current()&&surfaces.status(source).dirty,"Logical bank RTT lost native source receipt/authority");});
			std::size_t expected_holes=0,raw_native=0,logical_native=0;
			for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)
			{
				const unsigned row=volume?s.width*8:vtc?s.width*16:depth||plan.codec.block.width==2?s.width*4:s.columns*plan.codec.block.bytes,rows=volume||vtc||plan.codec.block.width==2?s.height:s.rows;
				const auto actual=make_buffer(ctx,rows*1024,buffer_storage::shared),wanted=make_buffer(ctx,rows*1024,buffer_storage::shared);texture_region area;area.level=s.level;area.slice=s.face;area.origin=MTLOriginMake(0,0,z);area.size=MTLSizeMake(s.width,s.height,1);
				ctx.begin();texture_to_buffer(ctx,result.image.texture,area,actual,0,512,0,depth&&!volume?texture_plane::depth:texture_plane::all);texture_to_buffer(ctx,oracle.texture,area,wanted,0,512,0,depth&&!volume?texture_plane::depth:texture_plane::all);if(format==0x91&&!volume){texture_to_buffer(ctx,result.image.texture,area,actual,rows*512,512,0,texture_plane::stencil);texture_to_buffer(ctx,oracle.texture,area,wanted,rows*512,512,0,texture_plane::stencil);}ctx.submit().wait();
				for(unsigned y=0;y<rows;++y)
				{
					if(depth)for(unsigned x=0;x<s.width;++x)
					{
						const auto at=words.offset-mapped.logical.address+s.offset+(z*s.storage_rows+y)*s.row_pitch+x*bpp;bool all_logical=true,all_physical=true;
						for(unsigned c=0;c<bpp;++c){all_logical&=logical_mask.at(at+c)!=std::byte{0};all_physical&=physical_mask.at(at+c)!=std::byte{0}&&logical_mask.at(at+c)==std::byte{0};}
						unsigned actual_bits,wanted_bits;std::memcpy(&actual_bits,static_cast<const std::byte*>(actual.contents)+y*512+x*(volume?8:4),4);std::memcpy(&wanted_bits,static_cast<const std::byte*>(wanted.contents)+y*512+x*(volume?8:4),4);if(all_logical||all_physical)wanted_bits=std::bit_cast<unsigned>(native_depth);require(actual_bits==wanted_bits,"Logical bank RTT depth lost full native precision orVM byte policy");++depth_words;
						if(volume){unsigned stencil,expected_stencil;std::memcpy(&stencil,static_cast<const std::byte*>(actual.contents)+y*512+x*8+4,4);std::memcpy(&expected_stencil,static_cast<const std::byte*>(wanted.contents)+y*512+x*8+4,4);require(stencil==expected_stencil,"Logical bank Depth3D changed numeric stencil");}else if(format==0x91)require(static_cast<const std::byte*>(actual.contents)[rows*512+y*512+x]==static_cast<const std::byte*>(wanted.contents)[rows*512+y*512+x],"Logical bank RTT changed separate stencil");
					}
					else require(!std::memcmp(static_cast<const std::byte*>(actual.contents)+y*512,static_cast<const std::byte*>(wanted.contents)+y*512,row),"Logical bank RTT partial mip/cube/volume/BC/packed native bytes differ");bytes+=row;
				}++planes;
				for(unsigned y=0;y<s.rows;++y)for(unsigned byte=0;byte<s.columns*plan.codec.block.bytes;++byte)
				{
					const auto bb=plan.codec.block.bytes;const auto at=words.offset-mapped.logical.address+s.offset+(vtc&&z<(s.depth&~std::size_t(3))?(z/4)*4*s.row_pitch*s.storage_rows+4*(y*s.row_pitch+(byte/bb)*bb)+(z%4)*bb+byte%bb:(z*s.storage_rows+y)*s.row_pitch+byte);expected_holes+=mask.at(at)==std::byte{0};raw_native+=physical_mask.at(at)!=std::byte{0}&&mask.at(at)!=std::byte{0};logical_native+=mask.at(at)!=std::byte{0}&&physical_mask.at(at)==std::byte{0};
				}
			}
			std::size_t captured=0;for(const auto range:memory->display_reads)captured+=range.size;
			require(captured==expected_holes&&textures.statistics().assembly_vm_bytes==expected_holes,"Logical bank RTT captured more than exact VM holes");holes+=captured;mixed+=raw_native&&logical_native;
			const auto allocations=budget->statistics().created;const auto encoders=ctx.encoding_statistics();const auto reads=memory->display_reads.size();
			require(textures.resolve_versioned(words,surfaces).image.texture==result.image.texture&&budget->statistics().created==allocations&&ctx.encoding_statistics().compute_encoders==encoders.compute_encoders&&ctx.encoding_statistics().render_encoders==encoders.render_encoders&&memory->display_reads.size()==reads,"Warm logical bank RTT recaptured/allocated/encoded");
			memory->display_forbidden.clear();checked_cpu_capture(capture,[&]{for(const auto& part:mapped.segments)require(memory->raw->read({part.address,part.size}).bytes==std::vector<std::byte>(initial.begin()+part.offset,initial.begin()+part.offset+part.size),"Logical bank RTT wrote native pixels to VM");});
			const unsigned transition=profiles%6;
			if(transition==5)surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){auto pass=[MTL4RenderPassDescriptor new];if(depth){pass.depthAttachment.texture=images[0];pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=.25;if(format==0x91){pass.stencilAttachment.texture=images[0];pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=19;}}else{pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(.25,.5,.75,1);}(void)commands.render(pass);});
			else checked_cpu_capture(capture,[&]{const auto& tail=mapped.segments.back();const guest_surface_range last{static_cast<unsigned>(std::uint64_t(tail.address)+tail.size-1),1};if(transition==0){auto changed=tile_desc;changed.bound=false;tiles->update(0,changed);tiles->update(0,tile_desc);}else if(transition==1){io->notify({0x100000,0x100000});io->notify({0x100000,0x100000});}else if(transition==2)memory->write(last,std::array{std::byte{0x11}});else if(transition==3)memory->raw->notify_mapping_changed(last);else *tail_address=0x40000;});
			if(transition==5){const auto fresh=textures.resolve_versioned(words,surfaces);require(fresh.image.texture!=result.image.texture&&textures.resolve_versioned(words,surfaces).image.texture==fresh.image.texture,"Partial logical bank RTT reused old nativewriter cache image");checked_cpu_capture(capture,[&]{require(result.revision.current()&&fresh.revision.current(),"Native writer changed immutable VM input receipt");});}else checked_cpu_capture(capture,[&]{require(!result.revision.current(),"Partial logical bank RTT accepted stale tile/IO/VM/mapping receipt");});++profiles;
		}
		} catch(const std::exception& e){throw std::runtime_error(std::string(e.what())+" format="+std::to_string(format)+" dim="+std::to_string(unsigned(dimension))+" pitch="+std::to_string(pitch)+" bank="+std::to_string(bank)+" morton="+std::to_string(morton)+" samples="+std::to_string(count));}
	}
	require(profiles==432&&mixed&&depth_words,"Logical bank RTT GPUmatrix incomplete");
	std::cout<<"Logical banked RTT:"<<profiles<<" actual mapped topEA/disjoint-IO profiles,"<<planes<<" mip/cube/volume planes,"<<bytes<<" exact native bytes,"<<holes<<" exactVMholebytes,"<<mixed<<" mixed logical/raw profiles,"<<depth_words<<" exact floatdepth words andseparate stencil;ARGB8/RGBA32F/depth sources,BC1/2/3 includingVTC3D,depthcubes/depthvolumes andpaired-pixel targets,1/4samples,two bank/pitch factors,linear/Mortoncrops,432tile/IO/samevalueVM/physicalmapping/actualEA/nativewriter receipt/cache invalidation checks,no nativeVMpixel reads/writeback,warm zeroallocation/capture/encoding passed (VM-only native decode oracle,shared RSX bank byte oracle)\n";
}
struct banked_display_fixture
{
	std::unique_ptr<context> ctx;
	std::shared_ptr<heap_budget> budget;
	std::shared_ptr<display_memory> memory=std::make_shared<display_memory>();
	std::unique_ptr<surface_cache> cache;
	std::unique_ptr<display_resolver> display;
	std::array<cached_surface_ref,2> sources;
	display_snapshot descriptor;
	std::array<unsigned,2> starts;
	std::array<std::vector<std::byte>,2> expected;
	explicit banked_display_fixture(unsigned slots=3):ctx(std::make_unique<context>(slots)),budget(std::make_shared<heap_budget>(*ctx,8u<<20))
	{
		ctx->set_buffer_budget(budget);const auto access=memory->access;const cpu_capture_dispatch capture=[access](const auto& work){access->execute([&](auto&){work();});};
		cache=std::make_unique<surface_cache>(*ctx,memory,4u<<20,budget,capture);display=std::make_unique<display_resolver>(*ctx,memory,[](unsigned o,unsigned,std::size_t){return o;},budget,4u<<20,capture);
		for(unsigned i=0;i<2;++i)
		{
			const banked_tile_layout tile{0x10000+i*0x10000,0x10000,256,1,4};const auto swizzle=i==0?std::optional<swizzle_region>{{{16,8,1},{1,1,0},{8,6,1}}}:std::nullopt;
			const auto geometry=plan_guest_surface(0,8,6,4,i?256:0,swizzle);auto image=geometry;image.range={tile.address,tile.size};std::vector<std::byte> initial(tile.size,std::byte{0xa7});memory->write({tile.address,tile.size},initial);
			sources[i]=cache->acquire({tile.address+tile.offset,{4,3,4,surface_color::argb8,false},i?256u:0u,swizzle,tile});const unsigned word=i?0x00445566:0x00112233;cache->clear(sources[i],0xf0,word,0,255,{0,0,8,6});
			std::vector<std::byte> pixels(8*6*4),ones(pixels.size(),std::byte{1}),mask(tile.size);for(unsigned b=0;b<pixels.size();++b)pixels[b]=std::byte(word>>(24-b%4*8));expected[i]=initial;reference(image,tile,expected[i],pixels,true);reference(image,tile,mask,ones,true);
			const auto first=std::find(mask.begin(),mask.end(),std::byte{1});require(first!=mask.end()&&first!=mask.begin(),"Banked lifetime lacks boundary");starts[i]=unsigned(first-mask.begin())-1;
			for(unsigned position=0;position<tile.size;)
			{
				if(mask[position]==std::byte{0}){++position;continue;}const auto begin=position;while(position<tile.size&&mask[position]!=std::byte{0})++position;memory->forbidden.push_back({tile.address+begin,position-begin});
			}
		}
		descriptor={0x10000+starts[0],3,2,0x10000+starts[1]-starts[0]};require(descriptor.pitch>=12,"Two banked lifetime rows reversed");
	}
};
void verify_banked_fixture(context& ctx,const present_image& image,const std::array<std::vector<std::byte>,2>& expected,const std::array<unsigned,2>& starts,unsigned shifted=0)
{
	helper_catalog compile(ctx);const auto library=compile.library(display_reader,@"Owned bankdisplay reader");const auto reader=compile.compute(library,@"banked_display_read");const auto actual=banked_display_values(ctx,reader,image);
	for(unsigned y=0;y<2;++y)for(unsigned x=0;x<3;++x)for(unsigned c=0;c<4;++c)
	{
		const float wanted=c==3?1.f:unsigned(expected[y][starts[y]+shifted+x*4+c+1])/255.f;require(std::abs(std::bit_cast<float>(actual[y*3+x][c])-wanted)<=1e-7f,"Two banked native output differs from independent RSX byteoracle");
	}
}
void banked_display_lifetime_cases()
{
	{
		banked_display_fixture f;const auto gate=[f.ctx->device() newSharedEvent],other=[f.ctx->device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate},unblock_other{other};
		auto first=f.display->prepare(f.descriptor,*f.cache,std::array{display_upload_dependency{gate,1}});require(!first.ready(),"Two banked pending gate absent");
		const auto created=f.budget->statistics().created;const auto reads=f.memory->reads.size();auto duplicate=f.display->prepare(f.descriptor,*f.cache);require(!duplicate.ready()&&f.budget->statistics().created==created&&f.memory->reads.size()==reads&&f.display->statistics().assembly_casts==1,"Two banked pending assembly didnotshare wholework");
		auto shifted=f.descriptor;shifted.offset+=1;auto independent=f.display->prepare(shifted,*f.cache,std::array{display_upload_dependency{other,1}});require(!independent.ready(),"Two banked independent gate absent");require(first.cancel_pending(),"Two banked pending cancellation failed");other.signaledValue=1;
		bool refused=false;try{(void)duplicate.finish();}catch(const std::exception&){refused=true;}require(refused,"Cancelled bankdisplay published");const auto image=independent.finish();f.ctx->recover_completed_errors();verify_banked_fixture(*f.ctx,image,f.expected,f.starts,1);gate.signaledValue=1;
		const auto fresh=f.display->prepare(f.descriptor,*f.cache).finish();verify_banked_fixture(*f.ctx,fresh,f.expected,f.starts);
		require(f.cache->status(f.sources[0]).dirty&&f.cache->status(f.sources[1]).dirty,"Two bankdisplay sources lostdirty authority");
	}
	@autoreleasepool
	{
		banked_display_fixture f{1};guest_texture_cache textures(*f.ctx,f.memory,[](unsigned o,unsigned,std::size_t){return o;},4u<<20,f.budget);memory_pressure_owner pressure(*f.ctx,*f.cache,textures,f.budget,f.display.get());
		for(unsigned offset:{0u,1u,2u}){ @autoreleasepool{auto descriptor=f.descriptor;descriptor.offset+=offset;(void)f.display->prepare(descriptor,*f.cache).finish();} }
		(void)f.cache->reclaim_idle();const auto before=f.budget->statistics();const auto leave=f.descriptor.pitch*f.descriptor.height+0x10000-1024;
		id<MTLBuffer> blocker=f.budget->shared_buffer(before.budget-before.reserved_bytes-leave);auto descriptor=f.descriptor;descriptor.offset+=3;const auto image=f.display->prepare(descriptor,*f.cache).finish();
		require(f.budget->statistics().recovery_successes>before.recovery_successes&&f.display->statistics().evictions>0&&f.cache->status(f.sources[0]).dirty&&f.cache->status(f.sources[1]).dirty,"Two bankdisplay didnotrecover sharedledger");blocker=nil;verify_banked_fixture(*f.ctx,image,f.expected,f.starts,3);
	}
	std::shared_ptr<heap_budget> ledger;std::weak_ptr<provider> access;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan;std::array<std::vector<std::byte>,2> expected;std::array<unsigned,2> starts;id<MTLSharedEvent> gate;
		{
			auto f=std::make_unique<banked_display_fixture>();expected=f->expected;starts=f->starts;ledger=f->budget;access=f->memory->access;gate=[f->ctx->device() newSharedEvent];orphan.emplace(f->display->prepare(f->descriptor,*f->cache,std::array{display_upload_dependency{gate,1}}));require(!orphan->ready(),"Banked orphan gate absent");f.reset();require(!access.expired(),"Banked orphan lostmetadata/VMowners");
		}
		gate.signaledValue=1;const auto image=orphan->finish();context reader;verify_banked_fixture(reader,image,expected,starts);orphan.reset();require(access.expired(),"Banked orphan provider didnotretire");
	}
	require(ledger->statistics().reserved_bytes==0,"Banked orphan retained native tilebuffers/staging/snapshots/outputs");
	std::cout<<"Two native bankdisplay owners:Morton+linear separate allocations,wholepending sharing,independent windows survive cancellation,actual sharedledger output/rawtile/staging recovery andgated resolver/cache/context destruction;exactRSX pixels,dirtyauthority,provider retirement/zero oldcharges passed\n";
}
void bridges(context& ctx)
{
	surface_banked_transfer bridge(ctx); surface_color_transfer color(ctx); surface_depth_transfer depth(ctx);
	std::vector<surface_format> formats;
	for (auto format : {surface_color::x1rgb5_zero,surface_color::x1rgb5_one,surface_color::rgb565,surface_color::xrgb8_zero,surface_color::xrgb8_one,
		surface_color::argb8,surface_color::b8,surface_color::gb8,surface_color::rgba16f,surface_color::rgba32f,surface_color::r32f,
		surface_color::xbgr8_zero,surface_color::xbgr8_one,surface_color::abgr8}) formats.emplace_back(format);
	for (auto format : {surface_depth::z16_uint,surface_depth::z24s8_uint,surface_depth::z16_float,surface_depth::z24s8_float}) formats.emplace_back(format);
	unsigned cases = 0, array_cases = 0;
	for (const auto& format : formats)
		for (auto pitch : {256u, 768u, 1536u})
			for (unsigned bank = 0; bank < 4; ++bank)
				for (bool morton : {false, true})
				{
					@autoreleasepool
					{
						const auto* c = std::get_if<surface_color>(&format);
						const auto bytes = c ? color_layout(*c).pixel_bytes : describe_surface_depth(std::get<surface_depth>(format)).guest_bytes;
						const auto native = c ? describe_surface_color(*c).format : describe_surface_depth(std::get<surface_depth>(format)).format;
						const std::size_t width = 8, height = morton ? 5 : 65;
						const banked_tile_layout tile{0x50000, 0x40000, pitch, bank, static_cast<std::uint32_t>(morton ? bytes * 5 : pitch * 3 + bytes)};
						const auto swizzle = morton ? std::optional<swizzle_region>{{{16, 16, 1}, {3, 7, 0}, {width, height, 1}}} : std::nullopt;
						const auto image = attach_banked_tile(plan_guest_surface(tile.address + tile.offset, width, height, bytes, morton ? 0 : pitch, swizzle), tile);
						std::vector<std::byte> pixels(width * height * bytes);
						for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte((i * 29 + i / 17 + 71) & 255);
						std::vector<std::byte> encoded(tile.size, std::byte{0xa5}); reference(image, tile, encoded, pixels, true);
						std::vector<std::byte> decoded(pixels.size()); reference(image, tile, encoded, decoded, false);
						const auto guest = make_upload(ctx, encoded), input = make_upload(ctx, decoded);
						const auto output = make_upload(ctx, std::vector<std::byte>(tile.size, std::byte{0x7b}));
						const auto baseline = make_buffer(ctx, pixels.size(), buffer_storage::shared);
						auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native width:width height:height mipmapped:NO];
						descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
						const bool array = bank == 3 && morton;
						if (array)
						{
							descriptor.textureType = MTLTextureType2DArray; descriptor.arrayLength = 3; descriptor.mipmapLevelCount = 2;
							descriptor.width = (width + 2) * 2; descriptor.height = (height + 2) * 2;
						}
						const auto a = make_texture(ctx, descriptor), b = make_texture(ctx, descriptor);
						texture_region region; region.size = MTLSizeMake(width, height, 1);
						if (array) { region.level = 1; region.slice = 2; region.origin = MTLOriginMake(1,1,0); }
						ctx.begin();
						if (c)
						{
							color.upload(ctx, *c, input, {}, a, region); color.download(ctx, *c, a, region, baseline, {});
							const auto before = bridge.statistics();
							const auto tables = ctx.argument_statistics();
							bridge.upload_color(ctx, *c, guest, image, tile, b, region); bridge.download_color(ctx, *c, b, region, output, image, tile);
							const auto after = bridge.statistics();
							const auto encoded = ctx.argument_statistics();
							require(after.dispatches - before.dispatches == 2 && after.texels - before.texels == 2 * width * height &&
								after.active_bytes - before.active_bytes == 2 * width * height * bytes && after.staging_buffers - before.staging_buffers == 2 &&
								after.fused_color_uploads == before.fused_color_uploads + 1 && after.fused_color_downloads == before.fused_color_downloads + 1 &&
								encoded.created + encoded.reused - tables.created - tables.reused == 2,
								"Banked color transfer did not fuse endian conversion into one texel dispatch/one staging owner per direction");
						}
						else
						{
							const auto d = std::get<surface_depth>(format);
							depth.upload(ctx, d, input, {}, a, region); depth.download(ctx, d, a, region, baseline, {});
							const auto before = bridge.statistics();
							bridge.upload_depth(ctx, d, guest, image, tile, b, region); bridge.download_depth(ctx, d, b, region, output, image, tile);
							require(bridge.statistics().dispatches - before.dispatches == 2 && bridge.statistics().texels - before.texels == 2 * width * height &&
								bridge.statistics().active_bytes - before.active_bytes == 2 * width * height * bytes,
								"Banked depth conversion still dispatches once per byte instead of per texel");
						}
						ctx.submit().wait();
						const auto flat = map_buffer(baseline, 0, baseline.length); std::vector<std::byte> expected_pixels(flat.begin(), flat.end());
						std::vector<std::byte> expected(tile.size, std::byte{0x7b}); reference(image, tile, expected, expected_pixels, true);
						const auto actual = map_buffer(output, 0, output.length);
						require(std::equal(actual.begin(), actual.end(), expected.begin()), "Banked surface bridge differs from shared RSX address plus independent linear codec");
						++cases;
						array_cases += array;
					}
				}
	require(cases == 432, "Banked surface matrix omitted a format/bank/pitch/layout");
	require(array_cases == 54, "Banked array/mip/slice/origin matrix omitted a format/pitch");
	std::cout << "Native banked array views:54 color/depth profiles at array slice2,mip1,nonzero origins preserve exact output/padding passed\n";
	std::cout << "Native banked work counters:432 profiles dispatch one thread per texel;336color profiles use one fused endian dispatch/one staging buffer per direction andtwo actual argument-table snapshots per roundtrip passed\n";
	std::cout << "Native banked surface bridge:432 GPU cases,14color+4depth formats,1/2/4/8/16-byte pixels,all4banks,threepitch factors,65row crossings,unaligned byte origins,Morton subregions,complete allocation/padding guards;actual shared RSX byte oracle plus independent native linear codec passed\n";
}
void fused_completion_lifetime()
{
	std::shared_ptr<heap_budget> budget;
	id<MTLSharedEvent> gate = nil;
	id<MTLBuffer> output = nil;
	std::vector<std::byte> expected;
	const auto work = [&]
	{
		@autoreleasepool
		{
			context owner{1};
			budget = std::make_shared<heap_budget>(owner, 8u << 20); owner.set_buffer_budget(budget);
			surface_banked_transfer bridge(owner);
			const banked_tile_layout tile{0x10000, 0x10000, 256, 1, 16};
			const auto image = attach_banked_tile(plan_guest_surface(tile.address + tile.offset, 4, 3, 4, tile.pitch), tile);
			std::vector<std::byte> pixels(4 * 3 * 4);
			for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte((i * 13 + 71) & 255);
			std::vector<std::byte> encoded(tile.size, std::byte{0xa5}); reference(image, tile, encoded, pixels, true);
			expected.assign(tile.size, std::byte{0x7b}); reference(image, tile, expected, pixels, true);
			const auto input = make_upload(owner, encoded); output = make_upload(owner, std::vector<std::byte>(tile.size, std::byte{0x7b}));
			auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:3 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto texture = budget->texture(descriptor);
			gate = [owner.device() newSharedEvent]; require(bool(gate), "Fused banked completion gate is absent");
			owner.begin(); owner.wait_event(gate, 1);
			const texture_region region{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,3,1)};
			bridge.upload_color(owner, surface_color::argb8, input, image, tile, texture, region);
			bridge.download_color(owner, surface_color::argb8, texture, region, output, image, tile);
			require(bridge.statistics().staging_buffers == 2 && bridge.statistics().dispatches == 2, "Fused lifetime fixture lost its two private ranges");
			return owner.submit();
		}
	}();
	@autoreleasepool
	{
		bool timeout = false;
		try { work.wait(std::chrono::milliseconds(1)); } catch (const std::runtime_error&) { timeout = true; }
		require(timeout && !work.ready() && budget->statistics().reserved_bytes > output.length,
			"Destroyed fused banked owners released queued native input/staging/texture charges");
		gate.signaledValue = 1; work.wait();
		require(std::memcmp(output.contents, expected.data(), expected.size()) == 0, "Fused banked pending output lost bytes/padding after owner destruction");
		require(budget->statistics().reserved_bytes == output.length, "Completed fused banked work retained orphan private staging/input heaps");
		output = nil;
	}
	require(budget->statistics().reserved_bytes == 0, "Last fused output owner did not return its quota");
	std::cout << "Native fused banked lifetime:actual gated submission/timeout survives bridge/context/input destruction;entire64KiB output/padding exact andall scratch/heap charges retire at completion passed\n";
}
void cache_case(context& ctx, const std::shared_ptr<heap_budget>& budget)
{
	auto memory = std::make_shared<guarded_guest_memory>(std::make_shared<provider>());
	surface_cache cache(ctx, memory, 64 << 20, budget);
	const banked_tile_layout tile{0x10000, 0x40000, 768, 2, 772};
	const unsigned width = 8, height = 65;
	const auto image = attach_banked_tile(plan_guest_surface(tile.address + tile.offset, width, height, 4, 768), tile);
	std::vector<std::byte> pixels(width * height * 4);
	for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte(std::array{255u, 20u, 40u, 60u}[i % 4]);
	std::vector<std::byte> encoded(tile.size, std::byte{0xa5}); reference(image, tile, encoded, pixels, true); memory->write(image.range, encoded);
	const cached_surface_description description{tile.address + tile.offset, {width, height, 1, surface_color::argb8, false}, 768, {}, tile};
	const auto banked = cache.acquire(description), linear = cache.acquire({0x100000, {width, height, 1, surface_color::argb8, false}, width * 4, {}});
	require(cache.acquire(description) == banked, "Identical banked descriptor missed cache identity");
	auto oversized = description; oversized.tile->size = 128 << 20;
	bool limited = false;
	try { (void)cache.acquire(oversized); } catch (const std::length_error&) { limited = true; }
	require(limited, "Banked cache accepted an unbounded CPU tile snapshot");
	cache.copy(banked, {0, 0, width, height}, linear, 0, 0); cache.flush(linear);
	require(memory->read({0x100000, pixels.size()}).bytes == pixels, "Banked surface cache guest upload decoded wrong pixels");
	cache.clear(banked, 0xf0, 0xff123456, 0, 255, {0, 0, width, height});
	for (unsigned i = 0; i < pixels.size(); ++i) pixels[i] = std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]);
	auto expected = encoded; reference(image, tile, expected, pixels, true);
	cache.prepare_cpu_read({tile.address + tile.size - 1, 1});
	require(!cache.status(banked).dirty && memory->read(image.range).bytes == expected, "Banked cache physical-tail alias did not flush all tile bytes correctly");
	cache.clear(banked, 0xf0, 0xff123456, 0, 255, {0, 0, width, height});
	memory->write({tile.address + tile.size - 1, 1}, std::array{std::byte{0x77}});
	const auto before = memory->read(image.range).bytes; bool refused = false;
	try { cache.flush(banked); } catch (const surface_conflict_error&) { refused = true; }
	require(refused && cache.status(banked).dirty && memory->read(image.range).bytes == before, "Whole-tile conflict failed to retain dirty surface and untouched guest storage");
	cache.clear(banked, 0xf0, 0xff123456, 0, 255, {0, 0, width, height}); cache.flush(banked);
	expected.back() = std::byte{0x77}; require(memory->read(image.range).bytes == expected, "Fresh full-overwrite retry lost unrelated tile bytes");
	cache.spill(banked); cache.prepare(banked); cache.copy(banked, {0, 0, width, height}, linear, 0, 0); cache.flush(linear);
	require(memory->read({0x100000, pixels.size()}).bytes == pixels, "Banked cache spill/restore damaged pixels");
	std::cout << "Native banked surface cache:guest upload/native copy/clear/full-allocation writeback/physical-tail alias,whole-tile VM conflict without partial publication,fresh overwrite/padding preservation,spill/restore passed;productive framebuffer tile binding remains open\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx{1}; auto budget = std::make_shared<heap_budget>(ctx, 64 << 20); ctx.set_buffer_budget(budget);
			require(NSApp==nil,"Banked tests began withApp");banked_display_projection_cases();logical_banked_texture_cases(ctx,budget);banked_display_cases(ctx,budget);mapped_banked_display_cases();banked_display_lifetime_cases();bridges(ctx); fused_completion_lifetime(); cache_case(ctx, budget); segmented_cache_case(ctx, budget, false); segmented_cache_case(ctx, budget, true); framebuffer_cache_cases(ctx, budget); rounded_framebuffer_cases(ctx, budget); mapped_native_alias_cases(ctx, budget); mapped_owned_framebuffer_cases(ctx, budget);require(NSApp==nil,"Banked tests createdApp");
			std::cout << "Native banked surfaces passed;no App/UI/guestVM initialized\n";
		}
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
