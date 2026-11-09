#include "M4DMA.h"
#include "M4DMAMapping.h"
#include "M4GuardedGuestMemory.h"
#include "M4ScaledImageMapping.h"
#include "M4IOMappingEpoch.h"
#include <iostream>
#include <mutex>
#include <set>
namespace
{
using namespace metal4;
void require(bool v, const char* text)
{
	if (!v)
		throw std::runtime_error(text);
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<fixture>
{
	std::mutex mutex;

  public:
	std::vector<std::byte> bytes = std::vector<std::byte>(16 << 20, std::byte{0xa5});
	std::optional<std::uint32_t> deny;
	bool record_maps = false;
	std::vector<std::pair<guest_surface_range, bool>> mapped;
	unsigned notifications = 0;
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& op) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			op(*current);
			return;
		}
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		op(*this);
	}
	std::byte* map(guest_surface_range r, bool write) override
	{
		require(guest_memory_epoch::current(*this) == this, "DMA map escaped exclusion");
		const auto address = r.address % bytes.size();
		if (r.address >= bytes.size() * 2 || r.size > bytes.size() - address || (write && deny && address <= *deny && *deny - address < r.size))
			throw std::range_error("DMA mapping denied");
		if (record_maps) mapped.emplace_back(r, write);
		return bytes.data() + address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(guest_surface_range{a.address % (16 << 20), a.size}, guest_surface_range{b.address % (16 << 20), b.size}); }
	void written(guest_surface_range) noexcept override { ++notifications; }
};
std::vector<std::byte> expected(const dma_plan& p, std::span<const std::byte> source, std::span<const std::byte> target)
{
	std::vector<std::byte> out(target.begin(), target.end());
	for (unsigned r = 0; r < p.rows; ++r)
		for (unsigned c = 0; c < p.columns; ++c)
			out[std::int64_t(p.destination.first) + std::int64_t(r) * p.destination.pitch + c * p.destination.stride] = source[std::int64_t(p.source.first) + std::int64_t(r) * p.source.pitch + c * p.source.stride];
	return out;
}
std::vector<guest_storage_segment> parts(const dma_region& p)
{
	if (!p.segments.empty())
		return p.segments;
	return {{p.range.address, 0, p.range.size}};
}
void store(guest_surface_memory& m, const dma_region& r, std::span<const std::byte> b)
{
	for (const auto& p : parts(r))
		m.write({p.address, p.size}, b.subspan(p.offset, p.size));
}
std::vector<std::byte> gather(guest_surface_memory& m, const dma_region& r)
{
	std::vector<std::byte> out(r.range.size);
	for (const auto& p : parts(r))
	{
		const auto b = m.read({p.address, p.size}).bytes;
		std::copy(b.begin(), b.end(), out.begin() + p.offset);
	}
	return out;
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx{1};
			auto provider = std::make_shared<fixture>();
			auto memory = std::make_shared<guarded_guest_memory>(provider);
			auto budget = std::make_shared<heap_budget>(ctx, 64 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, memory, 64 << 20, budget);
			guest_texture_cache textures(ctx, memory, [](unsigned a, unsigned, std::size_t)
										 { return a; }, 16 << 20, budget);
			const cpu_capture_dispatch capture = [&](const auto& op)
			{ provider->execute([&](auto&)
								{ op(); }); };
			dma_compute compute(ctx, 8 << 20);
			dma_transfer transfer(ctx, surfaces, textures, memory, capture, 8 << 20);
			const auto verify_packed = [&](const dma_plan& plan, std::span<const std::byte> input, std::span<const std::byte> expected)
			{
				const auto source_layout = plan_dma_active(plan.source, plan.columns, plan.rows);
				const auto target_layout = plan_dma_active(plan.destination, plan.columns, plan.rows);
				std::vector<std::byte> packed_source(source_layout.bytes), packed_target(target_layout.bytes, std::byte{0xcd});
				for (unsigned i = 0; i < source_layout.bytes; ++i) packed_source[i] = input[dma_active_offset(source_layout, i)];
				const auto src = make_upload(ctx, packed_source), dst = make_upload(ctx, packed_target);
				dma_compute compact(ctx, packed_source.size() + packed_target.size());
				ctx.begin(); compact.encode(ctx, plan, src, dst, true); ctx.submit().wait();
				const auto got = map_buffer(dst, 0, packed_target.size());
				for (unsigned i = 0; i < target_layout.bytes; ++i)
					require(got[i] == expected[dma_active_offset(target_layout, i)], "Packed native DMA disagreed with independent dense row/column oracle");
			};
			unsigned cases = 0;
			for (unsigned ss = 1; ss <= 4; ++ss)
				for (unsigned ds = 1; ds <= 4; ++ds)
					for (int sp : {-91, -3, 0, 13, 97})
						for (int dp : {-87, -2, 0, 11, 103})
						{
							@autoreleasepool
							{
								const auto plan = plan_dma({4096, 8192, 1, 2, 17, 5, sp, dp, ss, ds, 0});
								std::vector<std::byte> input(plan.source.range.size), output(plan.destination.range.size, std::byte{0xa5});
								for (unsigned i = 0; i < input.size(); ++i)
									input[i] = std::byte((i * 37 + i / 7 + 73) & 255);
								const auto src = make_upload(ctx, input), dst = make_upload(ctx, output);
								ctx.begin();
								const auto before_dma = ctx.encoding_statistics();
								compute.encode(ctx, plan, src, dst);
								const auto after_dma = ctx.encoding_statistics();
								require(after_dma.checked_dispatches == before_dma.checked_dispatches + 1 &&
									after_dma.dispatch_barriers == before_dma.dispatch_barriers,
									"Independent dense DMA must use checked dispatch without an intra-pass barrier");
								ctx.submit().wait();
								const auto result = map_buffer(dst, 0, output.size());
								const auto want = expected(plan, input, output);
								require(std::equal(result.begin(), result.end(), want.begin()), "Native DMA GPU bytes differ from independent row/column reference");
								verify_packed(plan, input, want);
								store(*memory, plan.source, input); store(*memory, plan.destination, output);
								transfer.execute(plan);
								require(std::equal(want.begin(), want.end(), provider->bytes.begin() + plan.destination.range.address),
									"Productive packed DMA lost signed/overlapping row bytes or modified a gap");
								++cases;
							}
						}
			std::cout << cases << " checked dense DMA dispatches preserve bytes without intra-pass barriers\n";
			for (unsigned stride = 1; stride <= 4; ++stride) for (int pitch : {-1048576, 1048576})
			{
				const auto plan = plan_dma({4096, 0x200000, 1, 2, 4, 2, -19, pitch, 3, stride, 0});
				std::vector<std::byte> input(plan.source.range.size), output(plan.destination.range.size, std::byte{0xb7});
				for (unsigned i = 0; i < input.size(); ++i) input[i] = std::byte(i * 17 + 31);
				const auto active = plan_dma_active(plan.destination, plan.columns, plan.rows);
				require(active.bytes == 8, "Sparse native DMA dispatch includes pitch/stride gaps");
				const auto source = make_upload(ctx, input), target = make_upload(ctx, output);
				ctx.begin(); compute.encode(ctx, plan, source, target); ctx.submit().wait();
				const auto want = expected(plan, input, output);
				const auto got = map_buffer(target, 0, output.size());
				require(std::equal(got.begin(), got.end(), want.begin()), "Sparse active DMA changed gap bytes or lost signed-row data");
				verify_packed(plan, input, want);
			}
			{
				const auto plan = plan_dma({4096, 8192, 1, 2, 7, 0xffffffffu, 0, 0, 3, 4, 0});
				std::vector<std::byte> input(plan.source.range.size), output(plan.destination.range.size, std::byte{0xa5});
				for (unsigned i = 0; i < input.size(); ++i) input[i] = std::byte(i * 13);
				const auto source = make_upload(ctx, input), target = make_upload(ctx, output);
				ctx.begin(); compute.encode(ctx, plan, source, target); ctx.submit().wait();
				for (unsigned c = 0; c < 7; ++c) output[c * 4] = input[c * 3];
				const auto got = map_buffer(target, 0, output.size());
				require(plan_dma_active(plan.destination, plan.columns, plan.rows).bytes == 7 && std::equal(got.begin(), got.end(), output.begin()),
					"UINT_MAX repeated rows failed compact last-writer dispatch");
				verify_packed(plan, input, output);
			}
			{
				const auto plan = plan_dma({0x80000000u, 0, 1, 2, 1, 2, std::numeric_limits<std::int32_t>::min(),
					std::numeric_limits<std::int32_t>::max(), 1, 1, 0});
				const auto src = make_upload(ctx, std::array{std::byte{9}, std::byte{7}});
				const auto dst = make_upload(ctx, std::array{std::byte{0}, std::byte{0}});
				dma_compute compact(ctx, 4);
				ctx.begin(); compact.encode(ctx, plan, src, dst, true); ctx.submit().wait();
				const auto got = map_buffer(dst, 0, 2);
				require(got[0] == std::byte{7} && got[1] == std::byte{9}, "Packed 2GiB-pitch DMA lost signed source order");
				bool refused = false;
				const auto small = make_upload(ctx, std::array{std::byte{0}});
				ctx.begin();
				try { compact.encode(ctx, plan, small, dst, true); }
				catch (const std::invalid_argument&) { refused = true; }
				ctx.cancel();
				require(refused, "Packed native DMA accepted undersized source buffer");
				refused = false; ctx.begin();
				try { compact.encode(ctx, plan, src, small, true); }
				catch (const std::invalid_argument&) { refused = true; }
				ctx.cancel();
				require(refused, "Packed native DMA accepted undersized destination buffer");
			}
			std::cout << "Packed native MSL4 DMA:400 signed/zero/overlap profiles,8 sparse profiles andUINT_MAX rows match independent bytes;2GiB pitch uses2+2 buffer bytes with4-byte budget;undersized buffers refused\n";
			std::cout << "Active native DMA dispatch:8 signed1MiB-gap/stride1..4 profiles preserve every gap byte;UINT_MAX repeated rows dispatch7 bytes with exact results\n";
			{
				const auto sparse = plan_dma({0x100000, 0x600000, 1, 2, 4, 2, 1048576, 1048576, 1, 4, 0});
				std::vector<std::byte> input(sparse.source.range.size, std::byte{0x38}), output(sparse.destination.range.size, std::byte{0xa5});
				for (unsigned row = 0; row < 2; ++row) for (unsigned col = 0; col < 4; ++col) input[row * 1048576 + col] = std::byte(17 + row * 11 + col);
				store(*memory, sparse.source, input); store(*memory, sparse.destination, output);
				auto want = expected(sparse, input, output);
				dma_transfer compact(ctx, surfaces, textures, memory, capture, 16);
				const auto gate = [ctx.device() newSharedEvent];
				struct signal { id<MTLSharedEvent> gate; ~signal() { gate.signaledValue = 1; } } release{gate};
				provider->record_maps = true; provider->mapped.clear();
				bool timed = false;
				try { compact.execute(sparse, std::chrono::milliseconds(1), [&](auto& c) { c.wait_event(gate, 1); }); }
				catch (const std::runtime_error&) { timed = true; }
				provider->record_maps = false;
				require(timed && compact.has_pending(), "Sparse productive DMA did not remain gated");
				std::size_t capture_bytes = 0;
				for (const auto& [range, write] : provider->mapped) { require(!write, "Sparse capture wrote guest bytes"); capture_bytes += range.size; }
				require(capture_bytes == 16, "Sparse capture mapped envelope bytes instead of8 source and8 target revision bytes");
				memory->write({sparse.destination.range.address + 1, 1}, std::array{std::byte{0x79}}); want[1] = std::byte{0x79};
				memory->write({sparse.destination.range.address + 512, 1}, std::array{std::byte{0x73}}); want[512] = std::byte{0x73};
				memory->write({sparse.source.range.address, 1}, std::array{std::byte{0xee}});
				provider->record_maps = true; provider->mapped.clear(); gate.signaledValue = 1; compact.synchronize(); provider->record_maps = false;
				std::size_t published = 0;
				for (const auto& [range, write] : provider->mapped) { require(write, "Sparse publication recaptured guest pixels"); published += range.size; }
				require(published == 8 && std::equal(want.begin(), want.end(), provider->bytes.begin() + sparse.destination.range.address),
					"Sparse productive DMA overwrote CPU gap writes or lost frozen source");
				// Identical relative geometry selects the aligned native MTL4 blit.
				auto blit = sparse; blit.destination = plan_dma_region(0x600000, 1048576, 1, 4, 2);
				const auto gap_surface = surfaces.acquire({0x600080, {4, 1, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(gap_surface, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 1});
				fragment_texture_registers gap_words{};
				gap_words.offset = 0x600100;
				gap_words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 2u | 8u;
				gap_words.control0 = 1u << 31; gap_words.control1 = 0xaae4;
				gap_words.image_rect = (4u << 16) | 1u; gap_words.control3 = (1u << 20) | 16u;
				const auto gap_texture = textures.resolve(gap_words);
				const auto invalidations = textures.statistics().invalidations;
				store(*memory, blit.source, input); compact.execute(blit);
				for (unsigned row = 0; row < 2; ++row) for (unsigned col = 0; col < 4; ++col)
					require(provider->bytes[0x600000 + row * 1048576 + col] == input[row * 1048576 + col], "Sparse packed native blit lost bytes");
				require(surfaces.status(gap_surface).dirty && provider->bytes[0x600080] == std::byte{0xa5} &&
					textures.statistics().invalidations == invalidations && textures.resolve(gap_words).texture == gap_texture.texture,
					"Active DMA flushed or invalidated a native surface/texture solely inside a pitch gap");
				(void)surfaces.cpu_read({0x600080, 16});
				std::cout << "Productive sparse DMA:16-byte aggregate payload budget despite1MiB pitch;16 read-mapped bytes,8 published bytes;gated stride/pitch-gap CPU writes andfrozen source preserved;aligned packed native blit passed\n";
				std::cout << "Active DMA cache coherence preserves GPU-dirty surfaces andimmutable uploaded textures inside pitch gaps without flush/eviction\n";
			}
			{
				auto aliased = plan_dma({0x100000, 0x900000, 1, 2, 5, 3, 19, 3, 2, 4, 0});
				aliased.destination.segments = {{0x900000, 0, 8}, {0x900000, 8, 8}, {0x900000, 16, 7}};
				std::vector<std::byte> input(aliased.source.range.size), output(aliased.destination.range.size, std::byte{0xb7});
				for (unsigned i = 0; i < input.size(); ++i) input[i] = std::byte(i * 7 + 11);
				store(*memory, aliased.source, input); memory->write({0x900000, output.size()}, output);
				const auto logical = expected(aliased, input, output);
				std::set<std::size_t> active;
				for (unsigned row = 0; row < 3; ++row) for (unsigned col = 0; col < 5; ++col) active.insert(row * 3 + col * 4);
				for (auto offset : active) output[offset % 8] = logical[offset];
				transfer.execute(aliased);
				require(std::equal(output.begin(), output.end(), provider->bytes.begin() + 0x900000),
					"Interleaved residue groups/scattered physical aliases lost logical last-writer order");
				const auto matrix = plan_dma({0x100000, 0x800000, 1, 2, 2, 2048, 13, 17, 3, 4, 0});
				input.assign(matrix.source.range.size, std::byte{0x36}); output.assign(matrix.destination.range.size, std::byte{0x59});
				for (unsigned row = 0; row < 2048; ++row) for (unsigned col = 0; col < 2; ++col) input[row * 13 + col * 3] = std::byte(row * 11 + col);
				store(*memory, matrix.source, input); store(*memory, matrix.destination, output);
				const auto want = expected(matrix, input, output); transfer.execute(matrix);
				require(std::equal(want.begin(), want.end(), provider->bytes.begin() + matrix.destination.range.address), "Large byte matrix exhausted per-row watches or lost bytes");
				std::cout << "Productive packed DMA:400 signed/zero/overlap profiles;interleaved groups across3 physical-alias segments preserve logical order;2048 sparse rows use bounded matrix watches passed\n";
			}
			{
				const auto logical = plan_dma({0x200000, 0x800000, 1, 1, 4, 2, 2097152, 2097152, 3, 4, 0});
				std::array<std::uint32_t, 4> io_offsets{0x200000, 0x400000, 0x800000, 0xa00000};
				std::array<std::uint32_t, 4> addresses{0x300000, 0x500000, 0xb00000, 0xd10000};
				auto io = std::make_shared<io_mapping_epoch>();
				const auto query = [&]
				{
					const auto translate = [&](unsigned offset, unsigned, std::size_t count)
					{
						for (unsigned i = 0; i < 4; ++i) if (offset == io_offsets[i] && count <= 4096) return addresses[i];
						throw std::range_error("Unmapped sparse IO gap was translated");
					};
					return scaled_image_mapping_epoch::mappings{
						probe_dma_mapping(logical.source, 4, 2, 1, translate, logical.source.range.address),
						probe_dma_mapping(logical.destination, 4, 2, 1, translate, logical.destination.range.address)};
				};
				const auto ticket = [&]
				{
					std::optional<scaled_image_mapping_epoch> result;
					capture([&] { result.emplace(query, [&] { return scaled_image_mapping_versions{memory->mapping_generation(), io->current()}; },
						[&](const auto& values) { return dma_mapping_versions(values, *memory, *io); }); });
					return std::move(*result);
				};
				auto mapped = logical;
				const auto initial = ticket();
				attach_dma_mapping(mapped.source, initial.original()[0]); attach_dma_mapping(mapped.destination, initial.original()[1]);
				std::vector<std::byte> input(logical.source.range.size, std::byte{0x33}), output(logical.destination.range.size, std::byte{0xa5});
				for (unsigned row = 0; row < 2; ++row) for (unsigned col = 0; col < 4; ++col) input[row * 2097152 + col * 3] = std::byte(17 + row * 19 + col);
				store(*memory, mapped.source, input); store(*memory, mapped.destination, output);
				dma_transfer compact(ctx, surfaces, textures, memory, capture, 16);
				compact.execute(mapped, std::chrono::seconds(30), {}, [owned = ticket()] { owned.validate(); });
				for (unsigned row = 0; row < 2; ++row) for (unsigned col = 0; col < 4; ++col)
					require(provider->bytes[addresses[2 + row] + col * 4] == input[row * 2097152 + col * 3], "Sparse translated native DMA lost active bytes");
				const auto gate = [ctx.device() newSharedEvent];
				struct signal { id<MTLSharedEvent> gate; ~signal() { gate.signaledValue = 100; } } release{gate};
				const auto pause = [&](unsigned value)
				{
					bool timed = false;
					try { compact.execute(mapped, std::chrono::milliseconds(1), [&](auto& c) { c.wait_event(gate, value); }, [owned = ticket()] { owned.validate(); }); }
					catch (const std::runtime_error&) { timed = true; }
					require(timed && compact.has_pending(), "Sparse translated DMA did not remain gated");
				};
				pause(1);
				capture([&] { memory->notify_mapping_changed({0x400000, 4096}); io->notify({0x300000, 1048576}); });
				gate.signaledValue = 1; compact.synchronize();
				for (unsigned row = 0; row < 2; ++row) for (unsigned col = 0; col < 4; ++col)
					require(provider->bytes[addresses[2 + row] + col * 4] == input[row * 2097152 + col * 3], "Inactive VM/IO gap changes refused valid DMA");
				unsigned value = 2;
				for (bool vm : {false, true}) for (unsigned endpoint = 0; endpoint < 4; ++endpoint, ++value)
				{
					store(*memory, mapped.destination, output);
					pause(value);
					const auto before = provider->bytes;
					const auto notifications = provider->notifications;
					capture([&]
					{
						if (vm)
						{
							memory->notify_mapping_changed({addresses[endpoint], 4096});
							memory->notify_mapping_changed({addresses[endpoint], 4096});
						}
						else
						{
							addresses[endpoint] += 4096; io->notify({io_offsets[endpoint], 1048576});
							addresses[endpoint] -= 4096; io->notify({io_offsets[endpoint], 1048576});
						}
					});
					gate.signaledValue = value;
					bool rejected = false;
					try { compact.synchronize(); } catch (const scaled_image_mapping_conflict&) { rejected = true; }
					require(rejected && !compact.has_pending() && provider->bytes == before && provider->notifications == notifications,
						"Sparse mapped DMA missed active VM/IO ABA or published a prefix");
				}
				compact.execute(mapped, std::chrono::seconds(30), {}, [owned = ticket()] { owned.validate(); });
				for (unsigned row = 0; row < 2; ++row) for (unsigned col = 0; col < 4; ++col)
					require(provider->bytes[addresses[2 + row] + col * 4] == input[row * 2097152 + col * 3], "Fresh sparse mapped DMA retry lost bytes");
				std::cout << "Active mapped native DMA:2MiB-pitch unmapped IO gaps,disjoint EAs and16-byte payload budget;GPU-gated unrelated VM/IO gaps accepted;all4 source/target endpoints reject identical VM andIO ABA without prefix;fresh retry passed\n";
			}
			// Linear copy crosses two IO boundaries with disjoint physical entries.
			auto plan = plan_dma({0x1ffc00, 0x9ffc00, 1, 2, 1024, 1026, 1024, 1024, 1, 1, 0});
			plan.source.segments = {{0x1ffc00, 0, 1024}, {0x400000, 1024, 1 << 20}, {0x600000, (1 << 20) + 1024, 1024}};
			plan.destination.segments = {{0x9ffc00, 0, 1024}, {0xc00000, 1024, 1 << 20}, {0xe00000, (1 << 20) + 1024, 1024}};
			std::vector<std::byte> input(plan.source.range.size), output(plan.destination.range.size, std::byte{0xa5});
			for (unsigned i = 0; i < input.size(); ++i)
				input[i] = std::byte((i * 11 + i / 1024) & 255);
			store(*memory, plan.source, input);
			store(*memory, plan.destination, output);
			transfer.execute(plan);
			require(gather(*memory, plan.destination) == input, "Scattered DMA differs");
			const auto event = [ctx.device() newSharedEvent];
			struct release
			{
				id<MTLSharedEvent> e;
				~release() { e.signaledValue = 100; }
			} cleanup{event};
			const auto pending = [&](unsigned value, const std::function<void()>& validate = std::function<void()>{})
			{
				bool timed = false;
				try
				{
					transfer.execute(plan, std::chrono::milliseconds(1), [&](auto& c)
									 { c.wait_event(event, value); }, validate);
				}
				catch (const std::runtime_error&)
				{
					timed = true;
				}
				require(timed && transfer.has_pending(), "DMA fixture needs pending GPU work");
			};
			store(*memory, plan.destination, output);
			pending(1);
			memory->write({plan.destination.segments.back().address, 1}, std::array{std::byte{0x77}});
			const auto before = gather(*memory, plan.destination);
			const auto notices = provider->notifications;
			event.signaledValue = 1;
			bool conflict = false;
			try
			{
				transfer.synchronize();
			}
			catch (const texture_revision_conflict&)
			{
				conflict = true;
			}
			require(conflict && !transfer.has_pending() && gather(*memory, plan.destination) == before && provider->notifications == notices, "Last DMA CAS partially published");
			store(*memory, plan.destination, output);
			pending(2);
			provider->deny = plan.destination.segments.back().address;
			event.signaledValue = 2;
			bool denied = false;
			try
			{
				transfer.synchronize();
			}
			catch (const std::range_error&)
			{
				denied = true;
			}
			provider->deny.reset();
			require(denied && !transfer.has_pending() && gather(*memory, plan.destination) == output, "Last DMA mapping partially published");
			pending(3);
			memory->write({plan.source.segments.back().address, 1}, std::array{std::byte{0xee}});
			event.signaledValue = 3;
			transfer.synchronize();
			require(gather(*memory, plan.destination) == input, "Pending DMA lost frozen source");
			store(*memory, plan.source, input);
			store(*memory, plan.destination, output);
			auto generation = std::make_shared<unsigned>(1);
			const auto old = *generation;
			pending(4, [generation, old]
					{if(*generation!=old)throw std::runtime_error("DMA metadata changed"); });
			++*generation;
			event.signaledValue = 4;
			bool stale = false;
			try
			{
				transfer.synchronize();
			}
			catch (const std::runtime_error&)
			{
				stale = true;
			}
			require(stale && gather(*memory, plan.destination) == output, "DMA metadata change published stale bytes");
			transfer.execute(plan);
			require(gather(*memory, plan.destination) == input, "Fresh DMA retry failed");
			store(*memory, plan.destination, output);
			scaled_image_mapping_epoch::mappings stable{};
			stable[0] = {plan.source.range.address, plan.source.range.size, {}, plan.source.segments};
			stable[1] = {plan.destination.range.address, plan.destination.range.size, {}, plan.destination.segments};
			std::optional<scaled_image_mapping_epoch> observed;
			capture([&]
					{ observed.emplace([stable]
									   { return stable; }, [memory]
									   { return memory->mapping_generation(); }, [memory](const auto& values)
									   { return scaled_image_mapping_generation(values, *memory); }); });
			pending(5, [owned = std::move(*observed)]
					{ owned.validate(); });
			const auto epoch_before = memory->mapping_generation();
			const auto aba_notices = provider->notifications;
			provider->execute([&](auto&)
							  { memory->notify_mapping_changed({plan.source.segments.back().address, 1}); memory->notify_mapping_changed({plan.source.segments.back().address, 1}); });
			event.signaledValue = 5;
			bool aba = false;
			try
			{
				transfer.synchronize();
			}
			catch (const scaled_image_mapping_conflict&)
			{
				aba = true;
			}
			require(aba && !transfer.has_pending() && memory->mapping_generation() == epoch_before + 2 && provider->notifications == aba_notices && gather(*memory, plan.destination) == output, "Identical-value VM mapping ABA published DMA bytes");
			transfer.execute(plan);
			require(gather(*memory, plan.destination) == input, "DMA fresh capture after VM mapping ABA failed");
			// A gap in the logical envelope is not part of any translated segment.
			store(*memory, plan.destination, output);
			capture([&]
			{
				observed.emplace([stable] { return stable; }, [memory] { return memory->mapping_generation(); },
					[memory](const auto& values) { return scaled_image_mapping_generation(values, *memory); });
			});
			pending(6, [owned = std::move(*observed)] { owned.validate(); });
			provider->execute([&](auto&) { memory->notify_mapping_changed({0x300000, 4096}); });
			event.signaledValue = 6;
			transfer.synchronize();
			require(!transfer.has_pending() && gather(*memory, plan.destination) == input, "Unrelated scatter gap refused valid DMA publication");
			// IO and VM are independent revision domains. A larger VM stamp must
			// not hide IO ABA while the actual EA bytes/revisions stay unchanged.
			auto io_values = std::make_shared<scaled_image_mapping_epoch::mappings>(stable);
			auto io_epochs = std::make_shared<io_mapping_epoch>();
			(*io_values)[0].io = scaled_image_memory_mapping::io_range{0x1ffc00, plan.source.range.size};
			(*io_values)[1].io = scaled_image_memory_mapping::io_range{0x9ffc00, plan.destination.range.size};
			provider->execute([&](auto&)
			{
				for (unsigned i = 0; i < 8; ++i)
					memory->notify_mapping_changed({plan.source.segments.front().address, 1});
			});
			const auto io_ticket = [&]
			{
				std::optional<scaled_image_mapping_epoch> result;
				capture([&]
				{
					result.emplace([io_values] { return *io_values; }, [memory, io_epochs]
					{
						return scaled_image_mapping_versions{memory->mapping_generation(), io_epochs->current()};
					}, [memory, io_epochs](const auto& values)
					{
						return scaled_image_mapping_versions{scaled_image_mapping_generation(values, *memory), scaled_image_io_mapping_generation(values, *io_epochs)};
					});
				});
				return std::move(*result);
			};
			const auto refuse_io = [&](unsigned event_value, bool source)
			{
				store(*memory, plan.destination, output);
				const auto original_values = *io_values;
				const auto& last_target = plan.destination.segments.back();
				const auto target_watch = guest_byte_coverage({last_target.address, last_target.size});
				const auto target_revision = memory->surface_revision(target_watch);
				pending(event_value, [owned = io_ticket()] { owned.validate(); });
				const auto notifications = provider->notifications;
				provider->execute([&](auto&)
				{
					auto& last = (*io_values)[source ? 0 : 1].segments.back();
					const auto original = last.address;
					last.address += 0x100000;
					io_epochs->notify({source ? 0x300000u : 0xb00000u, 0x100000});
					last.address = original;
					io_epochs->notify({source ? 0x300000u : 0xb00000u, 0x100000});
				});
				require(*io_values == original_values && memory->surface_revision(target_watch) == target_revision, "IO ABA fixture changed final values or VM revision");
				event.signaledValue = event_value;
				bool refused = false;
				try { transfer.synchronize(); }
				catch (const scaled_image_mapping_conflict&) { refused = true; }
				require(refused && !transfer.has_pending() && provider->notifications == notifications && gather(*memory, plan.destination) == output, "Identical IO ABA published stale native bytes");
			};
			refuse_io(7, true);
			require(scaled_image_mapping_generation(*io_values, *memory) > io_epochs->current(), "IO ABA fixture did not exercise a larger VM counter");
			transfer.execute(plan, std::chrono::seconds(30), {}, [owned = io_ticket()] { owned.validate(); });
			require(gather(*memory, plan.destination) == input, "Fresh native IO generation retry failed");
			store(*memory, plan.destination, output);
			pending(8, [owned = io_ticket()] { owned.validate(); });
			provider->execute([&](auto&) { io_epochs->notify({0x500000, 0x100000}); });
			event.signaledValue = 8;
			transfer.synchronize();
			require(!transfer.has_pending() && gather(*memory, plan.destination) == input, "Unrelated IO entry refused valid native transfer");
			refuse_io(9, false);
			// Later physical segments participate in source flush and target invalidation.
			store(*memory, plan.destination, output);
			const auto dirty = surfaces.acquire({plan.source.segments[1].address, {4, 1, 1, surface_color::argb8, false}, 16, {}});
			surfaces.clear(dirty, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 1});
			provider->execute([&](auto&) { memory->notify_mapping_changed({0x300000, 4096}); });
			const auto mirror = surfaces.acquire({plan.destination.segments[1].address, {4, 1, 1, surface_color::argb8, false}, 16, {}});
			surfaces.prepare(mirror);
			fragment_texture_registers words{};
			words.offset = plan.destination.segments[1].address;
			words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 2u | 8u;
			words.control0 = 1u << 31;
			words.control1 = 0xaae4;
			words.image_rect = (4u << 16) | 1u;
			words.control3 = (1u << 20) | 16u;
			const auto old_texture = textures.resolve(words);
			const auto invalidations = textures.statistics().invalidations;
			transfer.execute(plan);
			require(!surfaces.status(mirror).initialized && textures.statistics().invalidations > invalidations, "DMA later cache mirrors not invalidated");
			const auto retained = make_buffer(ctx, 16, buffer_storage::shared);
			ctx.begin();
			texture_to_buffer(ctx, old_texture.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 1, 1)}, retained, 0);
			ctx.submit().wait();
			for (const auto byte : map_buffer(retained, 0, 16))
				require(byte == std::byte{0xa5}, "DMA changed retained old texture");
			const auto dirty_bytes = gather(*memory, plan.destination);
			for (unsigned i = 0; i < 16; ++i)
				require(dirty_bytes[1024 + i] == std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]), "DMA missed later dirty source");
			for (const auto& part : plan.destination.segments)
			{
				require(memory->read({part.address - 1, 1}).bytes[0] == std::byte{0xa5}, "DMA preceding segment guard changed");
				require(memory->read({part.address + unsigned(part.size), 1}).bytes[0] == std::byte{0xa5}, "DMA following segment guard changed");
			}
			// Same physical allocation via a different virtual mirror is memmove.
			plan = plan_dma({4096, 4097 + (16 << 20), 1, 2, 31, 7, 37, 41, 1, 2, 0});
			input.resize(plan.source.range.size);
			for (unsigned i = 0; i < input.size(); ++i)
				input[i] = std::byte(i * 17);
			store(*memory, plan.source, input);
			const auto old_target = gather(*memory, plan.destination);
			const auto want = expected(plan, input, old_target);
			transfer.execute(plan);
			require(gather(*memory, plan.destination) == want, "DMA physical alias lost snapshot semantics");
			// Destruction before completion must never publish a late write.
			plan = plan_dma({2048, 8192, 1, 2, 31, 3, -47, 43, 3, 4, 0});
			input.assign(plan.source.range.size, std::byte{0x3c});
			output.assign(plan.destination.range.size, std::byte{0xa5});
			store(*memory, plan.source, input);
			store(*memory, plan.destination, output);
			{
				dma_transfer abandoned(ctx, surfaces, textures, memory, capture, 8 << 20);
				bool timed = false;
				try
				{
					abandoned.execute(plan, std::chrono::milliseconds(1), [&](auto& c)
									  { c.wait_event(event, 10); });
				}
				catch (const std::runtime_error&)
				{
					timed = true;
				}
				require(timed && abandoned.has_pending(), "DMA teardown needs pending work");
			}
			event.signaledValue = 10;
			ctx.begin();
			ctx.submit().wait();
			require(gather(*memory, plan.destination) == output, "DMA teardown published late bytes");
			transfer.execute(plan);
			require(gather(*memory, plan.destination) == expected(plan, input, output), "Negative pitched DMA publication differs");
			store(*memory, plan.destination, output);
			bool cancelled = false;
			try
			{
				transfer.execute(plan, std::chrono::seconds(30), [](auto&)
								 { throw std::runtime_error("DMA encode cancelled"); });
			}
			catch (const std::runtime_error&)
			{
				cancelled = true;
			}
			require(cancelled && !transfer.has_pending() && gather(*memory, plan.destination) == output, "DMA encoding failure published bytes");
			const auto frozen = plan;
			transfer.execute(plan, std::chrono::seconds(30), [&](auto&)
							 { plan.columns = 1; });
			require(gather(*memory, frozen.destination) == expected(frozen, input, output), "Caller mutation changed DMA encoding");
			std::cout << "Native DMA:" << cases << " independent GPU signed-pitch/stride/overlapping-row cases;scattered raw copy/last-CAS/last-map atomic refusal/frozen source/metadata refusal/retry/identical-value last-segment VM mapping ABA refusal/unrelated gap and dirty-source preservation/independent VM-IO domains/source-target identical IO ABA refusal/unrelated IO accepted/physical aliases/later dirty source/cache invalidation/old immutable texture/segment guards/negative publication/encode cancellation/owned plan/teardown passed;no App/UI/guest VM initialized\n";
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
