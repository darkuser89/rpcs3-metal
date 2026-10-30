#include "M4GuardedGuestMemory.h"
#include "M4GuestMemoryAliases.h"
#include "M4GuestMemoryOverlap.h"
#include "M4PhysicalTextureRegion.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
class access final : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex mutex;
	bool active = false;

  public:
	std::array<std::byte, 4096> bytes;
	std::array<std::byte, 4096> alternate{};
	bool remapped = false;
	unsigned calls = 0, notifications = 0;
	std::size_t read_bytes = 0, written_bytes = 0;
	std::optional<unsigned> deny_write;
	std::optional<unsigned> deny_read;
	bool null_map = false;
	std::function<void()> alias_hook;
	access() { bytes.fill(std::byte{0xa5}); }
	std::byte* map(guest_surface_range range, bool write) override
	{
		require(active, "Mapping escaped its exclusive callback");
		const auto physical = range.address % 4096;
		if (range.size > bytes.size() - physical || range.address >= 8192 ||
			(write && deny_write && physical <= *deny_write && *deny_write - physical < range.size) ||
			(!write && deny_read && physical <= *deny_read && *deny_read - physical < range.size))
			throw std::runtime_error("Fixture inaccessible memory");
		(write ? written_bytes : read_bytes) += range.size;
		return null_map ? nullptr : (remapped && range.address < 4096 ? alternate.data() : bytes.data()) + physical;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		require(active, "Alias lookup escaped exclusive callback");
		if (alias_hook)
			alias_hook();
		return overlaps(guest_surface_range{a.address % 4096 + (remapped && a.address < 4096 ? 0x10000u : 0u), a.size},
						guest_surface_range{b.address % 4096 + (remapped && b.address < 4096 ? 0x10000u : 0u), b.size});
	}
	std::optional<guest_surface_range> rebase_storage(guest_surface_range source, guest_surface_range request) override
	{
		require(active, "Rebase escaped exclusive callback");
		return rebase_guest_memory_range(source, request, [&](unsigned block)
										 { return std::uint64_t(remapped && block == 0 ? 0x10000 : block % 4096); }, 4096);
	}
	bool equivalent_storage(guest_surface_range a, guest_surface_range b) override
	{
		require(active, "Rebase proof escaped exclusive callback");
		return equivalent_guest_memory_ranges(a, b, [&](unsigned block)
											  { return std::uint64_t(remapped && block == 0 ? 0x10000 : block % 4096); }, 4096);
	}
	void written(guest_surface_range) noexcept override { ++notifications; }
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* memory = guest_memory_epoch::current(*this))
		{
			operation(*memory);
			return;
		}
		std::lock_guard lock(mutex);
		++calls;
		active = true;
		try
		{
			guest_memory_epoch epoch(*this, *this);
			operation(*this);
		}
		catch (...)
		{
			active = false;
			throw;
		}
		active = false;
	}
};
std::size_t morton_reference(std::array<std::size_t, 3> position, std::array<std::size_t, 3> extent)
{
	std::size_t result = 0, output_bit = 1;
	for (std::size_t bit = 1; bit < extent[0] || bit < extent[1] || bit < extent[2]; bit *= 2)
		for (unsigned axis = 0; axis < 3; ++axis)
			if (bit < extent[axis])
			{
				if (position[axis] & bit)
					result += output_bit;
				output_bit *= 2;
			}
	return result;
}
} // namespace
int main()
{
	try
	{
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory memory(provider);
			const auto first = memory.watch_surface(plan_guest_surface(0, 1, 2, 1, 4));
			const auto later = memory.watch_surface(plan_guest_surface(4096, 1, 2, 1, 4));
			const std::array a{std::byte{1}, std::byte{2}}, b{std::byte{7}, std::byte{8}};
			std::array writes{guest_surface_write{later, memory.surface_revision(later), b, true, 2},
				guest_surface_write{first, memory.surface_revision(first), a, true, 0}};
			require(memory.compare_write_many(writes).has_value() && provider->bytes[0] == b[0] && provider->bytes[4] == b[1],
				"Logical order across interleaved physical aliases followed watch-list order");
			writes[0].expected = memory.surface_revision(later); writes[1].expected = memory.surface_revision(first);
			const auto saved = provider->bytes;
			writes[0].logical_offset = 0;
			bool refused = false;
			try { (void)memory.compare_write_many(writes); } catch (const std::invalid_argument&) { refused = true; }
			require(refused && provider->bytes == saved, "Overlapping logical intervals published an ambiguous prefix");
			writes[0].logical_offset.reset(); refused = false;
			try { (void)memory.compare_write_many(writes); } catch (const std::invalid_argument&) { refused = true; }
			require(refused && provider->bytes == saved, "Mixed publication order modes were accepted");
			writes[0].logical_offset = std::numeric_limits<std::size_t>::max() - 2; refused = false;
			try { (void)memory.compare_write_many(writes); } catch (const std::overflow_error&) { refused = true; }
			require(refused && provider->bytes == saved, "Logical publication extent wrapped");
			unsigned profiles = 0;
			for (unsigned columns : {1u, 2u, 7u}) for (unsigned rows : {1u, 2u, 9u}) for (unsigned stride : {1u, 2u, 3u, 4u})
			{
				const auto shape = memory.watch_surface(plan_guest_byte_matrix(128, columns, rows, stride, columns * stride + 5));
				const auto before = provider->bytes;
				const auto snapshot = memory.read_surface_packed(shape);
				std::vector<std::byte> payload(columns * rows);
				for (unsigned r = 0; r < rows; ++r) for (unsigned c = 0; c < columns; ++c)
				{
					require(snapshot.bytes[r * columns + c] == before[128 + r * shape.row_pitch + c * stride], "Byte matrix packed read differs from row/column oracle");
					payload[r * columns + c] = std::byte((r * 37 + c + stride) % 251);
				}
				const guest_surface_write write{shape, snapshot.revision, payload, true};
				require(memory.compare_write_many({&write, 1}).has_value(), "Byte matrix publication refused");
				auto want = before;
				for (unsigned r = 0; r < rows; ++r) for (unsigned c = 0; c < columns; ++c) want[128 + r * shape.row_pitch + c * stride] = payload[r * columns + c];
				require(provider->bytes == want, "Byte matrix publication modified a column/row gap"); ++profiles;
			}
			std::cout << profiles << " byte-matrix capture/publication profiles preserve exact column/row gaps;logical physical-alias ordering,overlap/mixed-order/wrap refusal passed\n";
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory memory(provider, 16, 8);
			const auto sparse = memory.watch_surface(plan_guest_surface(0, 1, 4, 1, 1024));
			for (unsigned row = 0; row < 4; ++row) provider->bytes[row * 1024] = std::byte(17 + row);
			provider->deny_read = 1; // The storage envelope cannot be mapped.
			provider->deny_write = 1;
			const auto old = memory.read_surface_packed(sparse);
			require(old.bytes == std::vector{std::byte{17}, std::byte{18}, std::byte{19}, std::byte{20}} &&
				provider->read_bytes == 4, "Packed capture read padding or lost ordered active bytes");
			bool refused = false;
			try { (void)memory.read_surface(sparse); }
			catch (const std::length_error&) { refused = true; }
			require(refused, "Dense capture bypassed its storage-size budget");
			memory.write({2, 1}, std::array{std::byte{0x77}});
			require(memory.surface_revision(sparse) == old.revision, "Gap-only write changed packed revision");
			const std::array replacement{std::byte{31}, std::byte{32}, std::byte{33}, std::byte{34}};
			const guest_surface_write write{sparse, old.revision, replacement, true};
			const auto written_before = provider->written_bytes;
			require(memory.compare_write_many({&write, 1}).has_value(), "Packed publication rejected unchanged active bytes");
			require(provider->written_bytes == written_before + 4 && provider->bytes[2] == std::byte{0x77},
				"Packed publication mapped or overwrote a gap");
			for (unsigned row = 0; row < 4; ++row)
				require(provider->bytes[row * 1024] == replacement[row], "Packed publication lost row order");
			const auto saved = provider->bytes;
			require(!memory.compare_write_many({&write, 1}) && provider->bytes == saved, "Stale packed CAS mutated memory");
			provider->deny_read.reset(); provider->deny_write.reset();
			const auto mirror = memory.watch_surface(plan_guest_surface(4096, 1, 4, 1, 1024));
			const auto mirror_revision = memory.surface_revision(mirror);
			memory.write({4096 + 1024, 1}, std::array{std::byte{0x66}});
			require(memory.surface_revision(sparse) > old.revision && memory.surface_revision(mirror) > mirror_revision,
				"Physical alias write missed packed watches");
			const auto before_aba = memory.surface_revision(sparse);
			(void)memory.mutate_mapping({0, 4096}, [&](auto&) noexcept { provider->remapped = true; return true; });
			(void)memory.mutate_mapping({0, 4096}, [&](auto&) noexcept { provider->remapped = false; return true; });
			const guest_surface_write stale{sparse, before_aba, replacement, true};
			require(!memory.compare_write_many({&stale, 1}), "Packed publication accepted same-byte mapping ABA");
			std::cout << "Packed 3073-byte envelope uses4 active bytes within8-byte budget;unmapped gaps,gap writes,physical aliases andVM ABA passed\n";
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory memory(provider, 16, 16);
			const auto first = memory.watch_surface(plan_guest_surface(0, 1, 3, 1, 4));
			const auto last = memory.watch_surface(plan_guest_surface(64, 2, 1, 1));
			const std::array packed{std::byte{1}, std::byte{2}, std::byte{3}};
			const std::array dense{std::byte{4}, std::byte{5}};
			std::array writes{guest_surface_write{first, memory.surface_revision(first), packed, true},
				guest_surface_write{last, memory.surface_revision(last), dense}};
			const auto saved = provider->bytes;
			++writes[1].expected;
			require(!memory.compare_write_many(writes) && provider->bytes == saved, "Late revision failure published packed prefix");
			--writes[1].expected;
			provider->deny_write = 65;
			bool refused = false;
			try { (void)memory.compare_write_many(writes); }
			catch (const std::runtime_error&) { refused = true; }
			require(refused && provider->bytes == saved, "Late mapping failure published packed prefix");
			provider->deny_write.reset();
			require(memory.compare_write_many(writes).has_value(), "Mixed dense/packed atomic transaction failed");
			require(provider->bytes[0] == packed[0] && provider->bytes[4] == packed[1] && provider->bytes[8] == packed[2] &&
				provider->bytes[64] == dense[0] && provider->bytes[65] == dense[1] && provider->bytes[1] == saved[1], "Mixed publication offsets differ");
			// Payloads may alias the guest bytes changed by an earlier batch item.
			const auto alias = memory.watch_surface(plan_guest_surface(4097, 1, 3, 1, 4));
			memory.write({0, 3}, std::array{std::byte{21}, std::byte{22}, std::byte{23}});
			const std::array alias_writes{guest_surface_write{first, memory.surface_revision(first), packed, true},
				guest_surface_write{alias, memory.surface_revision(alias), std::span<const std::byte>{provider->bytes.data(), 3}, true}};
			require(memory.compare_write_many(alias_writes).has_value(), "Packed alias publication failed");
			require(provider->bytes[1] == std::byte{21} && provider->bytes[5] == std::byte{22} && provider->bytes[9] == std::byte{23},
				"Packed batch failed to freeze aliased payload before first mutation");
			const auto overlap = memory.watch_surface(plan_guest_surface(4096, 1, 3, 1, 4));
			const std::array later{std::byte{91}, std::byte{92}, std::byte{93}};
			const std::array overlapping_writes{guest_surface_write{first, memory.surface_revision(first), packed, true},
				guest_surface_write{overlap, memory.surface_revision(overlap), later, true}};
			require(memory.compare_write_many(overlapping_writes).has_value() && provider->bytes[0] == later[0] &&
				provider->bytes[4] == later[1] && provider->bytes[8] == later[2], "Packed physical overlaps lost list-order last writer");
			std::cout << "Packed/dense batch late-CAS andlate-map refusal areatomic;aliased payload freezing passed\n";
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory memory(provider, 16, 8, 3);
			const auto rows = memory.watch_surface(plan_guest_surface(0, 1, 4, 1, 4));
			bool refused_read = false, refused_write = false;
			try { (void)memory.read_surface_packed(rows); }
			catch (const std::length_error&) { refused_read = true; }
			const auto old = memory.read_surface_packed(plan_guest_surface(32, 1, 3, 1, 4));
			const std::array payload{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
			// Obtain its revision without reading or mapping the four active runs.
			const auto dense_snapshot = memory.read_surface_packed(plan_guest_surface(64, 4, 1, 1));
			std::array writes{guest_surface_write{plan_guest_surface(32, 1, 3, 1, 4), old.revision, std::span{payload}.first(3), true},
				guest_surface_write{plan_guest_surface(64, 4, 1, 1), dense_snapshot.revision, payload, true}};
			const auto saved = provider->bytes;
			try { (void)memory.compare_write_many(writes); }
			catch (const std::length_error&) { refused_write = true; }
			require(refused_read && refused_write && provider->bytes == saved, "Packed run budget did not reject before publication");
			guarded_guest_memory limited(provider, 16, 4);
			writes[0].expected = limited.surface_revision(writes[0].coverage);
			writes[1].expected = limited.surface_revision(writes[1].coverage);
			bool refused_budget = false, refused_size = false;
			try { (void)limited.compare_write_many(writes); }
			catch (const std::length_error&) { refused_budget = true; }
			writes[0].bytes = payload;
			try { (void)limited.compare_write_many(std::span{writes}.first(1)); }
			catch (const std::invalid_argument&) { refused_size = true; }
			require(refused_budget && refused_size && provider->bytes == saved, "Packed aggregate budget/payload validation failed");
			std::cout << "Packed capture/publication run limits,aggregate byte budget andpayload-size refusal passed\n";
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory memory(provider);
			const auto layout = memory.watch_surface(plan_guest_surface(0, 2, 2, 4, 16));
			const auto before = provider->calls;
			bool cross_provider_refused = false;
			provider->execute([&](exclusive_guest_memory::view& view)
			{
				require(guest_memory_epoch::current(*provider) == &view, "Nested epoch lost exact outer view");
				const auto old = memory.read_surface(layout);
				memory.write({0, 1}, std::array{std::byte{0x42}});
				require(memory.surface_revision(layout) > old.revision && memory.read_surface(layout).bytes[0] == std::byte{0x42}, "Nested read/write/revision lost one CPU epoch");
				access other;
				try { other.execute([](auto&) {}); }
				catch (const std::logic_error&) { cross_provider_refused = true; }
				bool separate = false;
				std::thread probe([&] { separate = guest_memory_epoch::current(*provider) == nullptr; });
				probe.join();
				require(separate, "Guest epoch view leaked onto another thread");
			});
			require(cross_provider_refused && provider->calls == before + 1 && !guest_memory_epoch::current(*provider), "Nested CPU epoch reacquired exclusion or leaked scope");
			bool failed = false;
			try
			{
				provider->execute([&](auto&) { provider->execute([](auto&) { throw std::range_error("nested capture failure"); }); });
			}
			catch (const std::range_error&) { failed = true; }
			require(failed && !guest_memory_epoch::current(*provider) && memory.read_surface(layout).bytes[0] == std::byte{0x42}, "Exception leaked nested VM epoch or exclusion lock");
			std::cout << "CPU callback epoch reuses exact nested view without relocking;cross-provider refusal,thread isolation and exception unwind passed\n";
		}
		unsigned profiles = 0;
		for (unsigned width : {1u, 2u, 3u, 4u, 7u, 8u})
			for (unsigned height : {1u, 2u, 4u, 8u})
				for (unsigned pixel : {1u, 2u, 4u, 8u})
					for (unsigned padding : {0u, 3u, 16u})
					{
						auto backing = std::make_shared<access>();
						guarded_guest_memory memory(backing);
						const auto layout = plan_guest_surface(64, width, height, pixel, (width + padding) * pixel);
						const auto initial = memory.read_surface(layout);
						require(initial.bytes == std::vector<std::byte>(layout.range.size, std::byte{0xa5}), "Full storage snapshot differs");
						const auto changed = memory.compare_write(layout, initial.revision, std::vector<std::byte>(layout.range.size, std::byte{0x3c}));
						require(changed && *changed > initial.revision && memory.surface_revision(layout) == *changed, "Compare-write failed to publish active revision");
						for (unsigned y = 0; y < height; ++y)
							for (unsigned x = 0; x < layout.row_pitch; ++x)
							{
								const auto offset = y * layout.row_pitch + x;
								if (offset < layout.range.size)
									require(backing->bytes[64 + offset] == (x < width * pixel ? std::byte{0x3c} : std::byte{0xa5}), "Active patch touched linear padding");
							}
						const auto saved = backing->bytes;
						require(!memory.compare_write(layout, initial.revision, initial.bytes) && saved == backing->bytes, "Failed revision compare mutated bytes");
						if (padding && height > 1)
						{
							memory.write({64 + width * pixel, 1}, std::array{std::byte{0x77}});
							require(memory.surface_revision(layout) == *changed, "Padding-only write changed active revision");
						}
						const auto compact = memory.read_surface_packed(layout);
						require(compact.revision == *changed && compact.bytes == std::vector<std::byte>(width * height * pixel, std::byte{0x3c}),
							"Packed linear snapshot differs from independent active-byte count");
						auto payload = compact.bytes;
						for (std::size_t i = 0; i < payload.size(); ++i) payload[i] = std::byte(i % 251);
						const auto before_packed = backing->bytes;
						const guest_surface_write packed_write{layout, compact.revision, payload, true};
						require(memory.compare_write_many({&packed_write, 1}).has_value(), "Packed linear publication failed");
						auto oracle = before_packed;
						for (unsigned row = 0; row < height; ++row)
							for (unsigned byte = 0; byte < width * pixel; ++byte)
								oracle[64 + row * layout.row_pitch + byte] = payload[row * width * pixel + byte];
						require(backing->bytes == oracle, "Packed linear publication differs from independent row oracle");
						++profiles;
					}
		for (unsigned z : {0u, 1u})
			for (unsigned x : {0u, 1u, 3u})
				for (unsigned y : {0u, 1u, 4u})
				{
					auto backing = std::make_shared<access>();
					guarded_guest_memory memory(backing);
					const swizzle_region box{{8, 8, 2}, {x, y, z}, {4, 4, 1}};
					const auto layout = plan_guest_surface(0, 4, 4, 2, 0, box);
					const auto initial = memory.read_surface(layout);
					const auto changed = memory.compare_write(layout, initial.revision, std::vector<std::byte>(layout.range.size, std::byte{0x55}));
					require(bool(changed), "Morton compare-write refused valid revision");
					std::array<bool, 256> expected{};
					for (unsigned row = y; row < y + 4; ++row)
						for (unsigned column = x; column < x + 4; ++column)
						{
							const auto offset = morton_reference({column, row, z}, box.storage) * 2;
							expected[offset] = expected[offset + 1] = true;
						}
					for (unsigned byte = 0; byte < 256; ++byte)
						require(backing->bytes[byte] == (expected[byte] ? std::byte{0x55} : std::byte{0xa5}), "Morton patch touched inactive slice/region");
					const auto compact = memory.read_surface_packed(layout);
					require(compact.bytes == std::vector<std::byte>(32, std::byte{0x55}), "Packed Morton capture includes inactive bytes");
					auto payload = compact.bytes;
					for (unsigned i = 0; i < payload.size(); ++i) payload[i] = std::byte(i + 1);
					const guest_surface_write packed_write{layout, compact.revision, payload, true};
					require(memory.compare_write_many({&packed_write, 1}).has_value(), "Packed Morton publication failed");
					unsigned packed_index = 0;
					for (unsigned byte = 0; byte < 256; ++byte)
						require(backing->bytes[byte] == (expected[byte] ? payload[packed_index++] : std::byte{0xa5}),
							"Packed Morton order/padding differs from independent Morton oracle");
					++profiles;
				}
		unsigned active_alias_cases = 0;
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory aliases(provider);
			const auto pitched = plan_guest_surface(0, 2, 2, 1, 8);
			const auto mirror = plan_guest_surface(4096, 2, 2, 1, 8);
			const auto revision = aliases.surface_revision(pitched);
			require(memory_storage_overlap(aliases, pitched, mirror), "Physical pitched mirror was not selected");
			for (unsigned byte = 0; byte < 16; ++byte)
			{
				const auto calls = provider->calls;
				require(memory_storage_overlap(aliases, guest_surface_range{4096 + byte, 1}, pitched) == (byte % 8 < 2),
						"Physical active overlap included row padding");
				require(provider->calls == calls + 1, "Physical overlap did not use one identity epoch");
				++active_alias_cases;
			}
			const swizzle_region first{{8, 8, 2}, {1, 2, 0}, {4, 4, 1}};
			const swizzle_region second{{8, 8, 2}, {1, 2, 1}, {4, 4, 1}};
			const auto active = plan_guest_surface(0, 4, 4, 1, 0, first);
			const auto other_slice = plan_guest_surface(4096, 4, 4, 1, 0, second);
			require(!memory_storage_overlap(aliases, active, other_slice), "Physical overlap conflated Morton volume slices");
			std::array<bool, 128> expected{};
			for (unsigned y = 2; y < 6; ++y)
				for (unsigned x = 1; x < 5; ++x)
					expected[morton_reference({x, y, 0}, first.storage)] = true;
			for (unsigned byte = 0; byte < 128; ++byte)
			{
				require(memory_storage_overlap(aliases, guest_surface_range{4096 + byte, 1}, active) == expected[byte],
						"Physical active overlap disagreed with independent Morton byte oracle");
				++active_alias_cases;
			}
			require(aliases.surface_revision(pitched) == revision, "Storage identity query changed memory revision");
		}
		auto backing = std::make_shared<access>();
		guarded_guest_memory memory(backing, 16);
		const auto layout = plan_guest_surface(64, 2, 2, 1, 8);
		const auto initial = memory.read_surface(layout);
		const auto mirror = plan_guest_surface(4160, 2, 2, 1, 8);
		const auto alias_revision = memory.surface_revision(mirror);
		memory.write({64, 1}, std::array{std::byte{0x11}});
		memory.write({64, 1}, std::array{std::byte{0xa5}});
		require(memory.surface_revision(mirror) > alias_revision && !memory.compare_write(layout, initial.revision, initial.bytes), "Physical alias/ABA write reused an old revision");
		const auto before_fault = memory.read_surface(layout);
		const auto saved = backing->bytes;
		backing->deny_write = 72;
		bool fault = false;
		try
		{
			(void)memory.compare_write(layout, before_fault.revision, std::vector<std::byte>(layout.range.size, std::byte{0}));
		}
		catch (const std::runtime_error&)
		{
			fault = true;
		}
		require(fault && saved == backing->bytes && memory.surface_revision(layout) == before_fault.revision, "Second-row permission fault caused partial write/revision");
		backing->deny_write.reset();
		backing->null_map = true;
		fault = false;
		try
		{
			(void)memory.read_surface(layout);
		}
		catch (const std::runtime_error&)
		{
			fault = true;
		}
		require(fault, "Absent mapping accepted");
		backing->null_map = false;
		guarded_guest_memory tight(backing, 1, 4096, 1);
		const auto tight_initial = tight.read_surface(layout);
		fault = false;
		try
		{
			(void)tight.compare_write(layout, tight_initial.revision, tight_initial.bytes);
		}
		catch (const std::length_error&)
		{
			fault = true;
		}
		require(fault && saved == backing->bytes, "Patch budget published partial memory");
		fault = false;
		try
		{
			(void)tight.surface_revision(plan_guest_surface(128, 2, 1, 1));
		}
		catch (const std::length_error&)
		{
			fault = true;
		}
		require(fault && tight.surface_revision(layout) == tight_initial.revision,
				"Registration pressure invented an active-byte revision change");
		// An external writer's notification and store share exactly one provider
		// callback. No other observer can see the intervening revision/data state.
		const auto external_revision = memory.surface_revision(layout);
		backing->execute([&](exclusive_guest_memory::view& view)
						 {
			memory.before_write_in_transaction(view, {64, 1});
			*view.map({64, 1}, true) = std::byte{0x42};
			view.written({64, 1}); });
		require(memory.surface_revision(layout) > external_revision, "External guarded write omitted revision");
		const auto saved_external = memory.surface_revision(layout);
		bool unowned = false;
		try { memory.before_write_in_transaction(*backing, {64, 1}); }
		catch (const std::logic_error&) { unowned = true; }
		require(unowned && memory.surface_revision(layout) == saved_external, "Unowned view changed guest revision");
		access different;
		unowned = false;
		try { different.execute([&](auto& view) { memory.before_write_in_transaction(view, {64, 1}); }); }
		catch (const std::logic_error&) { unowned = true; }
		require(unowned && memory.surface_revision(layout) == saved_external, "Different provider view changed guest revision");
		std::cout << "External writer requires its own exact active provider view;unowned/cross-provider notification refused without revision change passed\n";
		{
			auto alias_source = std::make_shared<access>();
			for (unsigned byte = 0; byte < alias_source->bytes.size(); ++byte)
				alias_source->bytes[byte] = std::byte(byte);
			guarded_guest_memory alias_memory(alias_source);
			const auto captured = alias_memory.read_surface(layout);
			const auto result = alias_memory.compare_write(layout, captured.revision,
														   std::span<const std::byte>(alias_source->bytes.data() + 56, layout.range.size));
			require(result && alias_source->bytes[64] == std::byte{56} && alias_source->bytes[65] == std::byte{57} &&
						alias_source->bytes[72] == std::byte{64} && alias_source->bytes[73] == std::byte{65},
					"Aliased payload read bytes modified by an earlier patch");
			alias_memory.write({66, 6}, std::span<const std::byte>(alias_source->bytes.data() + 64, 6));
			require(alias_source->bytes[66] == std::byte{56} && alias_source->bytes[67] == std::byte{57} &&
						alias_source->bytes[70] == std::byte{68} && alias_source->bytes[71] == std::byte{69},
					"Raw overlapping payload was not frozen before mutation");
			guarded_guest_memory byte_limited(alias_source, 1, 1);
			const auto calls = alias_source->calls;
			bool refused = false;
			try
			{
				(void)byte_limited.read_surface(layout);
			}
			catch (const std::length_error&)
			{
				refused = true;
			}
			require(refused && alias_source->calls == calls, "Snapshot byte budget entered exclusive memory");
			refused = false;
			try
			{
				alias_memory.write({0xfffffff0u, 32}, std::array<std::byte, 32>{});
			}
			catch (const std::overflow_error&)
			{
				refused = true;
			}
			require(refused && alias_source->calls == calls, "Wrapped guest range entered memory provider");
		}
		std::atomic<bool> failed = false;
		const auto worker = [&]()
		{
			try
			{
				for (unsigned iteration = 0; iteration < 500; ++iteration)
				{
					const auto snapshot = memory.read_surface(layout);
					const auto result = memory.compare_write(layout, snapshot.revision, std::vector<std::byte>(layout.range.size, std::byte(iteration)));
					if (result)
						require(*result > snapshot.revision, "Concurrent CAS reused a token");
				}
			}
			catch (...)
			{
				failed = true;
			}
		};
		std::thread first(worker), second(worker);
		first.join();
		second.join();
		require(!failed, "Concurrent guarded transactions failed");
		// Canonical identity includes the VM high lock/identity bits and splits
		// ranges at64KiB boundaries. Independent per-byte oracle for small spans.
		unsigned alias_cases = 0;
		const auto lookup = [](std::uint32_t block)
		{
			return block == 0x20000 ? std::uint64_t(0x8000000000010000ull) : block == 0x40000 ? std::uint64_t(0x8000000000010000ull)
																							  : std::uint64_t(block);
		};
		for (unsigned a : {0xfffcu, 0x10000u, 0x1fffcu, 0x20000u, 0x3fffcu, 0x40000u})
			for (unsigned b : {0xfffcu, 0x10000u, 0x1fffcu, 0x20000u, 0x3fffcu, 0x40000u})
				for (unsigned count : {1u, 4u, 8u})
				{
					bool expected = false;
					for (unsigned i = 0; i < count; ++i)
						for (unsigned j = 0; j < count; ++j)
							expected |= (lookup((a + i) & ~65535u) | ((a + i) & 65535u)) ==
										(lookup((b + j) & ~65535u) | ((b + j) & 65535u));
					require(guest_memory_aliases_overlap({a, count}, {b, count}, lookup) == expected, "Canonical physical alias disagrees with byte oracle");
					++alias_cases;
				}
		std::cout << profiles << " guarded linear/Morton memory profiles," << alias_cases << " canonical alias cases,physical alias/ABA,permission/absent-map/patch/registration budget refusals,aliased payloads,external atomic notification and1000 concurrent CAS attempts passed\n";
		unsigned rebase_cases = 0;
		std::array<unsigned, 4> permutation{0, 1, 2, 3};
		do
		{
			const auto block_identity = [&](unsigned block)
			{
				return 0x8000000000000000ull + (block >= 128 && block < 192 ? permutation[(block - 128) / 16] * 16 : block);
			};
			const auto byte_identity = [&](unsigned address)
			{
				return 0x8000000000000000ull + (address < 64 ? address : permutation[(address - 128) / 16] * 16 + address % 16);
			};
			for (unsigned offset = 0; offset < 64; ++offset)
				for (unsigned count : {1u, 4u, 8u, 17u, 33u})
				{
					if (count > 64 - offset)
						continue;
					std::optional<unsigned> expected;
					for (unsigned candidate = 0; candidate + count <= 64; ++candidate)
					{
						bool equal = true;
						for (unsigned byte = 0; byte < count; ++byte)
							equal &= byte_identity(candidate + byte) == byte_identity(128 + offset + byte);
						if (equal)
							expected = candidate;
					}
					const auto result = rebase_guest_memory_range({0, 64}, {128 + offset, count}, block_identity, 16);
					require(bool(result) == bool(expected) && (!result || (result->address == *expected && result->size == count)),
							"Canonical rebase disagreed with independent ordered byte oracle");
					++rebase_cases;
				}
		} while (std::next_permutation(permutation.begin(), permutation.end()));
		const auto duplicate = [](unsigned)
		{ return 0x8000000000000000ull; };
		require(!rebase_guest_memory_range({0, 32}, {128, 1}, duplicate, 16), "Ambiguous duplicate source mirrors accepted");
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory physical(provider);
			fragment_texture_registers words;
			words.control0 = 1u << 31;
			words.format = (1u << 16) | (0xa5u << 8) | 0x2a;
			words.image_rect = (1u << 16) | 2;
			words.control3 = (1u << 20) | 16;
			const auto texture = plan_guest_texture(words);
			const auto source = plan_guest_surface(0, 4, 2, 4, 16);
			const auto request = plan_guest_surface(4100, 1, 2, 4, 16);
			const auto calls = provider->calls;
			const auto rectangle = plan_physical_texture_region(physical, source, surface_color::argb8, texture, request);
			require(rectangle && rectangle->x == 1 && rectangle->y == 0 && rectangle->width == 1 && rectangle->height == 2 &&
						provider->calls == calls + 1,
					"Physical texture crop required unused trailing padding or mixed identity epochs");
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory watched(provider, 2);
			const auto shape = plan_guest_surface(0, 2, 2, 4, 12);
			auto first = watched.watch_surface(shape);
			auto second = watched.watch_surface(shape);
			require(first.registration && first.registration == second.registration, "Duplicate watch did not share ownership");
			const auto revision = watched.surface_revision(first);
			require(watched.surface_revision(second) == revision, "Owned watch changed revision without write");
			auto foreign = first;
			foreign.registration = std::make_shared<std::byte>();
			bool refused = false;
			try
			{
				(void)watched.surface_revision(foreign);
			}
			catch (const std::invalid_argument&)
			{
				refused = true;
			}
			require(refused, "Foreign watch owner accepted");
			auto forged = first;
			forged.registration = std::shared_ptr<void>(first.registration.get(), [](void*) {});
			refused = false;
			try
			{
				(void)watched.surface_revision(forged);
			}
			catch (const std::invalid_argument&)
			{
				refused = true;
			}
			require(refused, "Borrowed pointer masqueraded as owned watch");
			const auto calls = provider->calls;
			std::weak_ptr<void> weak = first.registration;
			first.registration.reset();
			second.registration.reset();
			require(weak.expired() && provider->calls == calls, "Watch destruction accessed provider or retained itself");
			for (unsigned index = 0; index < 1000; ++index)
			{
				const auto owned = watched.watch_surface(plan_guest_surface((index % 200) * 16, 1, 1, 4));
				(void)watched.read_surface(owned);
			}
			auto pinned = watched.watch_surface(shape);
			const auto stable = watched.surface_revision(pinned);
			const auto borrowed = watched.surface_revision(shape); // Legacy caller pins exact tag.
			require(borrowed == stable, "Legacy borrowing changed live revision");
			pinned.registration.reset();
			for (unsigned index = 0; index < 100; ++index)
			{
				const auto owned = watched.watch_surface(plan_guest_surface(128 + (index % 100) * 16, 1, 1, 4));
				(void)watched.surface_revision(owned);
			}
			require(watched.surface_revision(shape) == stable, "Retirement invented a legacy revision change");
			std::cout << "Owned watch sharing/foreign-owner rejection,destructor without VM entry,1000 retirements and legacy stable tags passed\n";
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory watched(provider, 2);
			const auto target = watched.watch_surface(plan_guest_surface(0, 2, 2, 4, 16));
			auto retiring = watched.watch_surface(plan_guest_surface(4, 1, 2, 4, 16));
			const auto revision = watched.surface_revision(target);
			unsigned aliases = 0;
			provider->alias_hook = [&]
			{ if (++aliases == 3) retiring.registration.reset(); };
			const std::vector<std::byte> payload(target.range.size, std::byte{0x71});
			require(watched.compare_write(target, revision, payload).has_value(), "Owner expiration during CAS planning failed");
			provider->alias_hook = {};
			require(!retiring.registration && watched.surface_revision(target) > revision, "Mid-transaction retirement lost live revision");
			std::cout << "Owner expiry during multirow CAS keeps planned entry pointers valid until exclusive transaction ends passed\n";
		}
		{
			auto provider = std::make_shared<access>();
			guarded_guest_memory memory(provider);
			const auto primary = memory.watch_surface(plan_guest_surface(0, 2, 2, 4, 16));
			const auto mirror = memory.watch_surface(plan_guest_surface(4096, 2, 2, 4, 16));
			const auto old = memory.read_surface(primary);
			const auto alias = memory.surface_revision(mirror);
			require(!memory.mutate_mapping({0, 4096}, [](auto&) noexcept
										   { return false; }) &&
						memory.surface_revision(primary) == old.revision,
					"Failed/no-op mapping transaction changed revision");
			require(memory.mutate_mapping({0, 4096}, [&](auto&) noexcept
										  { provider->remapped = true; return true; }),
					"Mapping transaction did not commit");
			const auto current = memory.read_surface(primary);
			require(current.revision > old.revision && current.bytes[0] == std::byte{0} && memory.surface_revision(mirror) == alias,
					"Remap retained old tag/data or invalidated unchanged mirror");
			require(!memory.compare_write(primary, old.revision, old.bytes), "Old mapping epoch wrote into remapped storage");
			const auto before_padding = current.revision;
			(void)memory.mutate_mapping({8, 8}, [](auto&) noexcept
										{ return true; });
			require(memory.surface_revision(primary) == before_padding, "Mapping-only row padding changed active surface epoch");
			provider->deny_read = 8;
			require(memory.surface_revision(primary) == before_padding, "Unmapped padding blocked active-only revision check");
			provider->deny_read = 16;
			bool refused = false;
			try
			{
				(void)memory.surface_revision(primary);
			}
			catch (const std::runtime_error&)
			{
				refused = true;
			}
			require(refused, "Unmapped active row was accepted");
			provider->deny_read.reset();
			(void)memory.mutate_mapping({0, 4096}, [&](auto&) noexcept
										{ provider->remapped = false; return true; });
			require(memory.surface_revision(primary) > before_padding && memory.read_surface(primary).bytes == old.bytes,
					"Remap ABA restored old revision or lost original backing");
			std::cout << "Atomic virtual remap/no-op/ABA,unchanged physical mirror,padding isolation and stale-CAS refusal passed\n";
		}
		std::cout << rebase_cases << " ordered physical rebase cases across24mapping permutations,ambiguous mirror refusal andactive-only trailing-padding crop passed\n";
		std::cout << active_alias_cases << " physical active-byte overlap cases,pitch padding,Morton slice separation andsingle identity epoch passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
