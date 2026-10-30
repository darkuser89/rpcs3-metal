#include "M4SurfaceCache.h"
#include "M4HostAccessFixture.h"
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename Error = std::exception, typename F> void refuses(F&& operation, const char* message)
{
	bool failed = false;
	try { operation(); } catch (const Error&) { failed = true; }
	require(failed, message);
}
struct watch_counts { std::size_t live = 0, peak = 0, created = 0, bad_release = 0; bool captured = false; };
class memory_fixture final : public guest_surface_memory
{
	std::vector<std::byte> bytes = std::vector<std::byte>(1 << 20);
	std::vector<std::uint64_t> versions = std::vector<std::uint64_t>(1 << 20);
	std::uint64_t clock = 0;
	void check(guest_surface_range range) const
	{
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Metadata fixture range");
	}
  public:
	std::shared_ptr<watch_counts> watches = std::make_shared<watch_counts>();
	std::size_t fail_watch = 0;
	guest_surface_layout watch_surface(const guest_surface_layout& layout) override
	{
		check(layout.range);
		if (fail_watch && watches->created + 1 == fail_watch) throw std::runtime_error("Injected watch failure");
		auto result = layout;
		result.registration = std::shared_ptr<void>(new std::byte{}, [counts = watches](void* value)
		{
			if (counts->captured || guest_memory_epoch::active()) ++counts->bad_release;
			--counts->live; delete static_cast<std::byte*>(value);
		});
		++watches->created; watches->peak = std::max(watches->peak, ++watches->live);
		return result;
	}
	std::uint64_t surface_revision(const guest_surface_layout& layout) override
	{
		check(layout.range); std::uint64_t result = 0;
		guest_surface_byte_cursor cursor(layout);
		while (const auto range = cursor.next())
			for (std::size_t byte = 0; byte < range->size; ++byte) result = std::max(result, versions[range->address + byte]);
		return result;
	}
	guest_surface_snapshot read_surface(const guest_surface_layout& layout) override
	{
		check(layout.range);
		return {surface_revision(layout), {bytes.begin() + layout.range.address, bytes.begin() + layout.range.address + layout.range.size}};
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& layout, std::uint64_t expected, std::span<const std::byte> data) override
	{
		if (surface_revision(layout) != expected) return {};
		require(data.size() == layout.range.size, "Metadata fixture payload size");
		guest_surface_byte_cursor cursor(layout); ++clock;
		while (const auto range = cursor.next())
		{
			std::copy_n(data.begin() + range->address - layout.range.address, range->size, bytes.begin() + range->address);
			std::fill_n(versions.begin() + range->address, range->size, clock);
		}
		return clock;
	}
	void write(guest_surface_range range, std::span<const std::byte> data) override
	{
		check(range); require(data.size() == range.size, "Metadata fixture store size");
		std::copy(data.begin(), data.end(), bytes.begin() + range.address);
		std::fill_n(versions.begin() + range.address, range.size, ++clock);
	}
	cpu_capture_dispatch capture()
	{
		return [counts = watches](const auto& operation)
		{
			const auto previous = std::exchange(counts->captured, true);
			try { operation(); } catch (...) { counts->captured = previous; throw; }
			counts->captured = previous;
		};
	}
};
cached_surface_description description(std::uint32_t address, std::size_t pitch = 0)
{
	return {address, {4, 4, 1, surface_color::argb8, false}, pitch, {}};
}
void bounded_churn(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(device, memory, 8 << 20, {}, memory->capture(), {16, 16});
	std::weak_ptr<cached_surface> previous;
	for (unsigned i = 0; i < 4096; ++i)
	{
		auto shape = description((i % 1024) * 512);
		shape.allocation.format = i % 2 ? surface_color::argb8 : surface_color::b8;
		const auto item = cache.acquire(shape);
		if (i == 0) previous = item;
		require(cache.acquire(shape) == item, "Normalized descriptor lookup lost identity");
		require(cache.metadata_statistics().entries <= 16 && memory->watches->live <= 16, "Metadata/watch limit exceeded during churn");
	}
	require(previous.expired(), "Old unused descriptor survived churn");
	const auto before = cache.metadata_statistics();
	require(before.descriptor_comparisons == 4096 && before.retired == 4080, "Descriptor lookup scanned unrelated entries or retirement count differs");
	require(cache.trim_metadata() == 16 && memory->watches->live == 0 && !memory->watches->bad_release, "Unused watches did not retire outside capture");
	std::cout << "4096 changing descriptors: <=16 entries/watches,4096 exact hit comparisons,4080 pressure retirements; final watches zero\n";
}
void handles_and_parts(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(device, memory, 8 << 20, {}, memory->capture(), {4, 4});
	std::vector<cached_surface_ref> owners;
	for (unsigned i = 0; i < 4; ++i) owners.push_back(cache.acquire(description(i * 1024)));
	const auto created = memory->watches->created;
	refuses<surface_metadata_budget_error>([&] { (void)cache.acquire(description(8192)); }, "Externally held entries were evicted");
	require(memory->watches->created == created && cache.trim_metadata() == 0, "Refused acquire registered a watch or trim removed handles");
	std::weak_ptr<cached_surface> first = owners[0], second = owners[1];
	(void)cache.acquire(description(0)); // Refresh first; second is now oldest.
	owners[0].reset(); owners[1].reset();
	owners.push_back(cache.acquire(description(8192)));
	require(!first.expired() && second.expired(), "Retirement ignored descriptor access order");
	owners.clear(); require(cache.trim_metadata() == 4, "Released handles did not retire");
	auto segmented = description(4096);
	segmented.segments = {{4096, 0, 17}, {8192, 17, 47}};
	const auto item = cache.acquire(segmented);
	const auto parts = cache.metadata_statistics().watch_parts;
	require(parts > 1 && parts <= 4, "Segment projection fixture lacks multiple watches");
	const auto registrations = memory->watches->created;
	refuses<std::length_error>([&] { auto second_shape = segmented; second_shape.allocation.format = surface_color::xrgb8_zero; (void)cache.acquire(second_shape); }, "Watch-part capacity was not enforced");
	require(memory->watches->created == registrations, "Watch-part refusal happened after registration");
	cache.cpu_write({16384, 1}, [&]
	{
		refuses<std::logic_error>([&] { (void)cache.trim_metadata(); }, "Trim entered a CPU epoch");
		refuses<std::logic_error>([&] { (void)cache.acquire(description(16384)); }, "Acquire entered a CPU epoch");
	});
	require(!memory->watches->bad_release && cache.status(item).guest_revisions.size() == parts, "Watch ownership was damaged");
	std::cout << "External handles/LRU/segmented watch budgets/CPU-epoch refusal passed\n";
}
void ranges_and_precision(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(device, memory, 16 << 20, {}, memory->capture(), {128, 128});
	std::vector<cached_surface_ref> entries;
	for (unsigned i = 0; i < 128; ++i) entries.push_back(cache.acquire(description(i * 1024, 32)));
	const auto selected = entries[73];
	cache.clear(selected, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 4});
	const auto start = cache.metadata_statistics();
	cache.prepare_cpu_read({73 * 1024 + 16, 16}); // Row padding, no active overlap.
	require(cache.status(selected).dirty, "Range index converted row padding to active overlap");
	cache.prepare_cpu_read({73 * 1024, 4});
	require(!cache.status(selected).dirty && cache.status(selected).native_write_epoch, "Exact active read did not flush native output");
	const auto used = cache.metadata_statistics().range_nodes - start.range_nodes;
	require(used < 96, "Direct sparse range query walked the whole descriptor set");
	std::weak_ptr<cached_surface> precise = selected;
	entries.clear();
	require(cache.trim_metadata() == 127 && !precise.expired(), "Native precision or unrelated descriptors retired incorrectly");
	const auto native = cache.acquire(description(73 * 1024, 32));
	require(native == selected && cache.status(native).native_write_epoch, "Clean native precision was reconstructed from guest bytes");
	std::cout << "128 sparse ranges: padding preserved dirty state,active read flushed,interval-node visits=" << used << "; clean native precision retained\n";
}
void unreferenced_precision_and_deferred(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(device, memory, 16 << 20, {}, memory->capture(), {3, 3});
	auto source = cache.acquire(description(4096));
	auto target = cache.acquire(description(8192));
	cache.clear(source, 0xf0, 0xffabcdef, 0, 255, {0, 0, 4, 4});
	cache.flush(source);
	std::weak_ptr<cached_surface> source_watch = source;
	source.reset();
	require(cache.trim_metadata() == 0 && !source_watch.expired(), "Unreferenced clean native precision was discarded");
	source = source_watch.lock();
	cache.defer_copy(source, {0, 0, 4, 4}, target, 0, 0);
	std::weak_ptr<cached_surface> target_watch = target;
	source.reset(); target.reset();
	require(cache.trim_metadata() == 0 && !target_watch.expired(), "Deferred owner/source retired");
	{
		auto dirty = cache.acquire(description(12288));
		cache.clear(dirty, 0xf0, 0xff010203, 0, 255, {0, 0, 4, 4});
	}
	const auto watched = memory->watches->created;
	refuses<surface_metadata_budget_error>([&] { (void)cache.acquire(description(16384)); }, "All protected native owners did not refuse metadata pressure");
	require(memory->watches->created == watched && cache.trim_metadata() == 0, "Protected native owners changed on refusal");
	cache.erase(cache.acquire(description(12288)));
	target = target_watch.lock(); cache.flush(target);
	require(memory->read({8192, 64}).bytes == memory->read({4096, 64}).bytes, "Deferred native copy lost its source lease");
	target.reset();
	std::array<std::byte, 64> replacement{};
	memory->write({4096, 64}, replacement); memory->write({8192, 64}, replacement);
	require(cache.trim_metadata() == 2 && source_watch.expired() && target_watch.expired(), "Stale clean native watches did not retire");
	require(memory->watches->live == 0 && !memory->watches->bad_release, "Native retirement released a watch in capture");
	std::cout << "Unreferenced clean native precision and deferred source/output retained; stale revisions safely retired\n";
}
void interval_oracle(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(device, memory, 32 << 20, {}, memory->capture(), {32, 128});
	std::vector<cached_surface_ref> items;
	std::vector<std::vector<std::uint32_t>> active;
	for (unsigned i = 0; i < 32; ++i)
	{
		auto shape = description(32768 + i * 128, 16 + 16 * (i % 3));
		if (i % 4 == 0) { shape.row_pitch = 0; shape.morton = swizzle_region{{8, 8, 1}, {2, 2, 0}, {4, 4, 1}}; }
		const auto size = shape.morton ? 256 : shape.row_pitch * 4;
		if (i % 4 == 1) shape.segments = {{shape.address, 0, 17}, {65536 + i * 256, 17, size - 17}};
		std::vector<std::uint32_t> expected;
		for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) for (unsigned b = 0; b < 4; ++b)
		{
			std::size_t offset = y * shape.row_pitch + x * 4 + b;
			if (shape.morton)
			{
				std::size_t pixel = 0;
				for (unsigned bit = 0; bit < 3; ++bit) pixel |= (((x + 2) >> bit) & 1u) << (2 * bit) | (((y + 2) >> bit) & 1u) << (2 * bit + 1);
				offset = pixel * 4 + b;
			}
			expected.push_back(!shape.segments.empty() && offset >= 17 ? shape.segments[1].address + offset - 17 : shape.address + offset);
		}
		active.push_back(std::move(expected)); items.push_back(cache.acquire(shape));
	}
	for (const auto& item : items) cache.prepare(item);
	unsigned checked = 0;
	for (unsigned probe = 0; probe < 96; ++probe)
	{
		const auto range = guest_surface_range{probe % 3 == 0 ? 32768 + (probe / 3) * 128 + 17 :
			probe % 3 == 1 ? active[probe / 3][(probe * 13) % 64] : 65536 + (probe / 3) * 256 + 47, probe % 7 + 1};
		cache.cpu_write(range, [] {});
		for (std::size_t i = 0; i < items.size(); ++i)
		{
			const bool overlap = std::any_of(active[i].begin(), active[i].end(), [&](auto byte)
				{ return byte >= range.address && byte < std::uint64_t(range.address) + range.size; });
			require(cache.status(items[i]).initialized == !overlap, "Interval index disagrees with independent active-byte oracle");
			if (overlap) cache.prepare(items[i]);
			++checked;
		}
	}
	items.clear(); require(cache.trim_metadata() == 32, "Interval-tree owner removal failed");
	const auto replacement = cache.acquire(description(0)); cache.prepare(replacement);
	cache.cpu_write({0, 1}, [] {});
	require(!cache.status(replacement).initialized, "Range index retained stale pointers after retirement/rebuild");
	std::cout << checked << " independent interval selections across pitched/Morton/disjoint overlapping images; removal/rebuild passed\n";
}
void mapping_churn(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	auto io = std::make_shared<io_mapping_epoch>(); auto tiles = std::make_shared<tile_state>();
	surface_cache cache(device, memory, 8 << 20, {}, memory->capture(), {8, 8});
	std::weak_ptr<cached_surface> old;
	for (unsigned i = 0; i < 512; ++i)
	{
		io->notify({0, 1 << 20});
		auto shape = description(4096);
		shape.segments = {{4096, 0, 64}};
		shape.mapping = std::make_shared<surface_mapping_revision>([] { return surface_mapping_state{{0, 64}, 1, 4096, {}, {{4096, 0, 64}}}; }, io, tiles, 1);
		const auto item = cache.acquire(shape);
		if (!i) { cache.clear(item, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 4}); cache.flush(item); old = item; }
		// Independently captured equal epochs must hit the same descriptor.
		shape.mapping = std::make_shared<surface_mapping_revision>([] { return surface_mapping_state{{0, 64}, 1, 4096, {}, {{4096, 0, 64}}}; }, io, tiles, 1);
		require(cache.acquire(shape) == item, "Equivalent mapping captures split descriptor identity");
	}
	require(old.expired() && cache.metadata_statistics().entries == 8 && memory->watches->live == 8, "Mapping-epoch churn retained obsolete owners");
	require(cache.invalidate_mapping({4096, 64}, surface_mapping_event::unmap) == 0 && cache.metadata_statistics().entries == 0, "Unmap did not retire unused mapping metadata");
	require(!memory->watches->bad_release, "Mapping owner/watch released inside capture");
	std::cout << "512 IO epochs with equal-capture lookup: <=8 watches; stale clean native owner and unmap metadata retired\n";
}
void partial_registration(context& device)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(device, memory, 8 << 20, {}, memory->capture(), {8, 8});
	auto shape = description(4096); shape.segments = {{4096, 0, 17}, {8192, 17, 47}};
	memory->fail_watch = 2;
	refuses([&] { (void)cache.acquire(shape); }, "Injected partial registration did not fail");
	require(cache.metadata_statistics().entries == 0 && cache.metadata_statistics().watch_parts == 0 && memory->watches->live == 0 && !memory->watches->bad_release, "Partial registration leaked metadata or retired inside capture");
	memory->fail_watch = 0;
	(void)cache.acquire(shape); require(cache.trim_metadata() == 1, "Retry after partial watch failure failed");
}
void pending_and_physical(context& device)
{
	auto provider = std::make_shared<test::host_access_fixture>(); provider->pin_first_alias();
	auto memory = std::make_shared<guarded_guest_memory>(provider, 16);
	const cpu_capture_dispatch capture = [&](const auto& op) { provider->execute([&](auto&) { op(); }); };
	surface_cache cache(device, memory, 16 << 20, {}, capture, {2, 2});
	auto target = cache.acquire(description(4096));
	cache.clear(target, 0xf0, 0xff2468ac, 0, 255, {0, 0, 4, 4});
	// An EA-only index must not discard the second alias of the same backing.
	const auto stats = cache.metadata_statistics();
	cache.prepare_cpu_read({0x41000, 4});
	require(!cache.status(target).dirty && cache.metadata_statistics().physical_candidates > stats.physical_candidates, "Physical alias was lost by virtual range pruning");
	const auto event = [device.device() newSharedEvent];
	struct signal { id<MTLSharedEvent> event; ~signal() { event.signaledValue = 1; } } done{event};
	cache.set_wait_timeout(std::chrono::milliseconds(1));
	std::array uses{cached_surface_use{target, true}};
	refuses([&] { cache.draw(uses, [&](context& commands, auto) { commands.wait_event(event, 1); }); }, "GPU event did not hold pending transaction");
	const auto before = cache.metadata_statistics();
	refuses([&] { (void)cache.trim_metadata(); }, "Trim ignored pending GPU completion");
	refuses([&] { (void)cache.acquire(description(8192)); }, "Acquire ignored pending GPU completion");
	require(cache.metadata_statistics().entries == before.entries && cache.status(target).pending, "Pending metadata changed before completion");
	event.signaledValue = 1; cache.set_wait_timeout(std::chrono::seconds(30)); cache.synchronize();
	provider->execute([&](auto&)
	{
		memory->begin_mapping_change({0, 65536}); provider->replace_first(); memory->end_mapping_change({0, 65536});
	});
	std::weak_ptr<cached_surface> owner = target;
	uses[0].surface.reset();
	target.reset();
	require(cache.trim_metadata() == 0 && !owner.expired(), "Dirty physical authority retired after remap");
	require(cache.invalidate_mapping({4096, 64}, surface_mapping_event::unmap) == 1, "Dirty old physical allocation was not reconciled");
	std::cout << "Physical aliases,old-allocation reconciliation and real pending MTL4 completion protected during metadata retirement\n";
}
void guarded_churn(context& device)
{
	auto provider = std::make_shared<test::host_access_fixture>();
	auto memory = std::make_shared<guarded_guest_memory>(provider, 8);
	const cpu_capture_dispatch capture = [&](const auto& op) { provider->execute([&](auto&) { op(); }); };
	surface_cache cache(device, memory, 8 << 20, {}, capture, {8, 8});
	for (unsigned i = 0; i < 2048; ++i)
	{
		(void)cache.acquire(description((i % 256) * 256));
		require(cache.metadata_statistics().entries <= 8, "Guarded registration churn grew metadata");
	}
	require(cache.trim_metadata() == 8, "Guarded registration trim failed");
	for (unsigned i = 0; i < 8; ++i) (void)cache.acquire(description(i * 1024));
	require(cache.metadata_statistics().entries == 8, "Guarded registration capacity did not recover");
	provider->execute([&](auto&)
	{
		refuses<std::logic_error>([&] { (void)cache.trim_metadata(); }, "Actual VM epoch allowed retirement");
	});
	std::cout << "2048 guarded-provider replacements under an actual 8-watch provider limit; capacity recovered and VM-epoch trim refused\n";
}
}
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			context device(1);
			bounded_churn(device); handles_and_parts(device); ranges_and_precision(device);
			unreferenced_precision_and_deferred(device); interval_oracle(device); mapping_churn(device); partial_registration(device);
			pending_and_physical(device); guarded_churn(device); device.wait_idle();
		}
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
