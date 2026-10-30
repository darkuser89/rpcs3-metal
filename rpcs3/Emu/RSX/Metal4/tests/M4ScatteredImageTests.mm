#include "M4GuardedGuestMemory.h"
#include "M4ScaledImage.h"
#include "M4ScaledImageMapping.h"
#include "M4IOMappingEpoch.h"
#include "M4TileState.h"
#include <iostream>
#include <mutex>

namespace
{
using namespace metal4;
void require(bool v, const char* message)
{
	if (!v)
		throw std::runtime_error(message);
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<fixture>
{
	std::mutex mutex;

  public:
	std::vector<std::byte> bytes = std::vector<std::byte>(32 << 20, std::byte{0xa5});
	std::optional<unsigned> denied;
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
		require(guest_memory_epoch::current(*this) == this, "Scatter map escaped CPU epoch");
		const auto address = r.address % bytes.size();
		if (r.address >= bytes.size() * 2 || r.size > bytes.size() - address || (write && denied && address <= *denied && *denied - address < r.size))
			throw std::range_error("Scatter mapping denied");
		return bytes.data() + address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		return overlaps(guest_surface_range{a.address % (32 << 20), a.size}, guest_surface_range{b.address % (32 << 20), b.size});
	}
	void written(guest_surface_range) noexcept override { ++notifications; }
};
std::vector<guest_storage_segment> segments(unsigned first, std::size_t size)
{
	std::vector<guest_storage_segment> out;
	for (std::size_t offset = 0; offset < size;)
	{
		const auto address = offset ? ((4u + 2u * unsigned(out.size() - 1)) << 20) + (first & 0x1000000) : first;
		const auto count = std::min<std::size_t>(size - offset, offset ? 1 << 20 : 0x100000 - (first & 0xfffff));
		out.push_back({address, offset, count});
		offset += count;
	}
	return out;
}
void store(guest_surface_memory& memory, std::span<const guest_storage_segment> parts, std::span<const std::byte> bytes)
{
	for (const auto& p : parts)
		memory.write({p.address, p.size}, bytes.subspan(p.offset, p.size));
}
std::vector<std::byte> gather(guest_surface_memory& memory, std::span<const guest_storage_segment> parts)
{
	std::vector<std::byte> out(parts.back().offset + parts.back().size);
	for (const auto& p : parts)
	{
		const auto bytes = memory.read({p.address, p.size}).bytes;
		std::copy(bytes.begin(), bytes.end(), out.begin() + p.offset);
	}
	return out;
}
std::vector<std::byte> contiguous(context& ctx, scaled_image_compute& compute, const scaled_image_plan& plan, std::span<const std::byte> source, std::span<const std::byte> target)
{
	@autoreleasepool
	{
		const auto input = make_upload(ctx, source), output = make_upload(ctx, target);
		ctx.begin();
		compute.encode(ctx, plan, input, output);
		ctx.submit().wait();
		const auto bytes = map_buffer(output, 0, target.size());
		return {bytes.begin(), bytes.end()};
	}
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
			auto budget = std::make_shared<heap_budget>(ctx, 128 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, memory, 128 << 20, budget);
			guest_texture_cache textures(ctx, memory, [](unsigned a, unsigned, std::size_t)
										 { return a; }, 16 << 20, budget);
			unsigned captures = 0;
			auto tile_epochs = std::make_shared<tile_state>();
			const cpu_capture_dispatch capture = tile_capture_dispatch([&](const auto& op)
			{++captures;provider->execute([&](auto&){op();}); }, tile_epochs);
			scaled_image_compute compute(ctx, 16 << 20);
			scaled_image_transfer transfer(ctx, surfaces, textures, memory, capture, 16 << 20);
			unsigned cases = 0;
			for (unsigned pixel : {2u, 4u})
				for (unsigned kind = 0; kind < 3; ++kind)
					for (bool flip : {false, true})
					{
						@autoreleasepool
						{
							const auto format = pixel == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8;
							guest_surface_layout source, target;
							std::optional<banked_tile_layout> src_tile, dst_tile;
							if (kind == 0)
							{
								const unsigned pitch = 256 * pixel, height = (1 << 20) / pitch + 2;
								source = plan_guest_surface(0x1ffc00, 255, height, pixel, pitch);
								target = plan_guest_surface(0x11ffc00, 255, height, pixel, pitch);
							}
							else if (kind == 1)
							{
								const swizzle_region swizzle{{1024, 1024, 1}, {3, 7, 0}, {256, 513, 1}};
								source = plan_guest_surface(0x1ffc00, 256, 513, pixel, 0, swizzle);
								target = plan_guest_surface(0x11ffc00, 256, 513, pixel, 0, swizzle);
							}
							else
							{
								src_tile = banked_tile_layout{0x1f0000, 2 << 20, 768, 1, 772};
								dst_tile = banked_tile_layout{0x11f0000, 2 << 20, 768, 3, 1540};
								source = attach_banked_tile(plan_guest_surface(src_tile->address + src_tile->offset, 8, 4, pixel, 768), *src_tile);
								target = attach_banked_tile(plan_guest_surface(dst_tile->address + dst_tile->offset, 8, 4, pixel, 768), *dst_tile);
							}
							scaled_image_plan plan{source, target, format, format, unsigned(source.width), unsigned(source.height), unsigned(target.width), unsigned(target.height), 0, 0, flip, false, false, src_tile, dst_tile};
							plan.source_segments = segments(source.range.address, source.range.size);
							plan.destination_segments = segments(target.range.address, target.range.size);
							std::vector<std::byte> input(source.range.size), output(target.range.size, std::byte{0xa5});
							for (unsigned i = 0; i < input.size(); ++i)
								input[i] = std::byte((i * 37 + i / 1024 + 73) & 255);
							store(*memory, plan.source_segments, input);
							store(*memory, plan.destination_segments, output);
							const auto expected = contiguous(ctx, compute, plan, input, output);
							if (kind == 0 && !flip)
							{
								auto independent = output;
								for (unsigned y = 0; y < source.height; ++y)
									std::copy_n(input.begin() + y * source.row_pitch, source.width * pixel, independent.begin() + y * target.row_pitch);
								require(expected == independent, "Identity oracle differs from independent row bytes/padding");
							}
							const auto before = captures;
							transfer.execute(plan);
							require(captures == before + 2 && gather(*memory, plan.destination_segments) == expected, "Scattered raw bytes differ from contiguous native codec");
							for (const auto& part : plan.destination_segments)
							{
								require(memory->read({part.address - 1, 1}).bytes[0] == std::byte{0xa5}, "Scatter changed preceding guard byte");
								require(memory->read({part.address + unsigned(part.size), 1}).bytes[0] == std::byte{0xa5}, "Scatter changed following guard byte");
							}
							++cases;
						}
					}
			const auto source = plan_guest_surface(0x1ffc00, 256, 1026, 4, 1024), target = plan_guest_surface(0x11ffc00, 256, 1026, 4, 1024);
			scaled_image_plan plan{source, target, scaled_image_format::argb8, scaled_image_format::argb8, 256, 1026, 256, 1026};
			plan.source_segments = segments(source.range.address, source.range.size);
			plan.destination_segments = segments(target.range.address, target.range.size);
			std::vector<std::byte> input(source.range.size, std::byte{0x3c}), output(target.range.size, std::byte{0xa5});
			store(*memory, plan.source_segments, input);
			store(*memory, plan.destination_segments, output);
			const auto gate = [ctx.device() newSharedEvent];
			struct release
			{
				id<MTLSharedEvent> event;
				~release() { event.signaledValue = 100; }
			} cleanup{gate};
			const auto pending = [&](unsigned signal, const std::function<void()>& validate = std::function<void()>{})
			{
				bool timeout = false;
				try
				{
					transfer.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
									 { commands.wait_event(gate, signal); }, validate);
				}
				catch (const std::runtime_error&)
				{
					timeout = true;
				}
				require(timeout && transfer.has_pending(), "Scatter fixture lacks pending GPU work");
			};
			pending(1);
			const auto& last = plan.destination_segments.back();
			memory->write({last.address, 1}, std::array{std::byte{0x77}});
			const auto before_conflict = gather(*memory, plan.destination_segments);
			const auto notices = provider->notifications;
			gate.signaledValue = 1;
			bool conflict = false;
			try
			{
				transfer.synchronize();
			}
			catch (const texture_revision_conflict&)
			{
				conflict = true;
			}
			require(conflict && !transfer.has_pending() && provider->notifications == notices && gather(*memory, plan.destination_segments) == before_conflict, "Last scatter CAS partially published earlier segments");
			store(*memory, plan.destination_segments, output);
			pending(2);
			provider->denied = last.address;
			const auto before_deny = provider->notifications;
			gate.signaledValue = 2;
			bool denied = false;
			try
			{
				transfer.synchronize();
			}
			catch (const std::range_error&)
			{
				denied = true;
			}
			provider->denied.reset();
			require(denied && !transfer.has_pending() && provider->notifications == before_deny && gather(*memory, plan.destination_segments) == output, "Last scatter mapping failure partially published");
			transfer.execute(plan);
			require(gather(*memory, plan.destination_segments) == input, "Fresh scatter retry failed");
			store(*memory, plan.destination_segments, output);
			pending(3);
			memory->write({plan.source_segments.back().address, 1}, std::array{std::byte{0xee}});
			gate.signaledValue = 3;
			transfer.synchronize();
			require(gather(*memory, plan.destination_segments) == input, "Pending scatter lost frozen source bytes");
			store(*memory, plan.source_segments, input);
			store(*memory, plan.destination_segments, output);
			auto mappings = std::make_shared<scaled_image_mapping_epoch::mappings>();
			(*mappings)[0] = {source.range.address, source.range.size, {}, plan.source_segments};
			(*mappings)[1] = {target.range.address, target.range.size, {}, plan.destination_segments};
			std::optional<scaled_image_mapping_epoch> epoch;
			capture([&]
					{ epoch.emplace([mappings]
									{ return *mappings; }); });
			pending(4, [owned = std::move(*epoch)]
					{ owned.validate(); });
			(*mappings)[1].segments.back().address += 65536;
			gate.signaledValue = 4;
			bool remapped = false;
			try
			{
				transfer.synchronize();
			}
			catch (const scaled_image_mapping_conflict&)
			{
				remapped = true;
			}
			require(remapped && gather(*memory, plan.destination_segments) == output, "Scatter IO entry remap published stale bytes");
			// The actual native metadata ticket also observes monotonic VM events.
			(*mappings)[1].segments = plan.destination_segments;
			std::optional<scaled_image_mapping_epoch> observed;
			capture([&]
					{ observed.emplace([mappings]
									   { return *mappings; }, [memory]
									   { return memory->mapping_generation(); }, [memory](const auto& values)
									   { return scaled_image_mapping_generation(values, *memory); }); });
			pending(5, [owned = std::move(*observed)]
					{ owned.validate(); });
			const auto epoch_before = memory->mapping_generation();
			const auto notices_before_aba = provider->notifications;
			provider->execute([&](auto&)
							  { memory->notify_mapping_changed({plan.destination_segments.back().address, 1}); memory->notify_mapping_changed({plan.destination_segments.back().address, 1}); });
			require(memory->mapping_generation() == epoch_before + 2, "VM mapping ABA lost a monotonic event");
			gate.signaledValue = 5;
			bool aba = false;
			try
			{
				transfer.synchronize();
			}
			catch (const scaled_image_mapping_conflict&)
			{
				aba = true;
			}
			require(aba && !transfer.has_pending() && provider->notifications == notices_before_aba && gather(*memory, plan.destination_segments) == output, "Identical-value VM mapping ABA published image bytes");
			transfer.execute(plan);
			require(gather(*memory, plan.destination_segments) == input, "Image fresh capture after VM mapping ABA failed");
			store(*memory, plan.destination_segments, output);
			capture([&]
			{
				observed.emplace([mappings] { return *mappings; }, [memory] { return memory->mapping_generation(); },
					[memory](const auto& values) { return scaled_image_mapping_generation(values, *memory); });
			});
			pending(6, [owned = std::move(*observed)] { owned.validate(); });
			provider->execute([&](auto&) { memory->notify_mapping_changed({0x300000, 4096}); });
			gate.signaledValue = 6;
			transfer.synchronize();
			require(!transfer.has_pending() && gather(*memory, plan.destination_segments) == input, "Unrelated scatter gap refused valid image publication");
			// IO and VM are independent revision domains. A larger VM stamp must
			// not hide IO ABA while the actual EA bytes/revisions stay unchanged.
			auto io_epochs = std::make_shared<io_mapping_epoch>();
			(*mappings)[0].io = scaled_image_memory_mapping::io_range{0x1ffc00, plan.source.range.size};
			(*mappings)[1].io = scaled_image_memory_mapping::io_range{0x9ffc00, plan.destination.range.size};
			provider->execute([&](auto&)
			{
				for (unsigned i = 0; i < 8; ++i)
					memory->notify_mapping_changed({plan.source_segments.front().address, 1});
			});
			const auto io_ticket = [&]
			{
				std::optional<scaled_image_mapping_epoch> result;
				capture([&]
				{
					result.emplace([mappings] { return *mappings; }, [memory, io_epochs]
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
				store(*memory, plan.destination_segments, output);
				const auto original_values = *mappings;
				const auto& last_target = plan.destination_segments.back();
				const auto target_watch = guest_byte_coverage({last_target.address, last_target.size});
				const auto target_revision = memory->surface_revision(target_watch);
				pending(event_value, [owned = io_ticket()] { owned.validate(); });
				const auto notifications = provider->notifications;
				provider->execute([&](auto&)
				{
					auto& last = (*mappings)[source ? 0 : 1].segments.back();
					const auto original = last.address;
					last.address += 0x100000;
					io_epochs->notify({source ? 0x300000u : 0xb00000u, 0x100000});
					last.address = original;
					io_epochs->notify({source ? 0x300000u : 0xb00000u, 0x100000});
				});
				require(*mappings == original_values && memory->surface_revision(target_watch) == target_revision, "IO ABA fixture changed final values or VM revision");
				gate.signaledValue = event_value;
				bool refused = false;
				try { transfer.synchronize(); }
				catch (const scaled_image_mapping_conflict&) { refused = true; }
				require(refused && !transfer.has_pending() && provider->notifications == notifications && gather(*memory, plan.destination_segments) == output, "Identical IO ABA published stale native bytes");
			};
			refuse_io(7, true);
			require(scaled_image_mapping_generation(*mappings, *memory) > io_epochs->current(), "IO ABA fixture did not exercise a larger VM counter");
			transfer.execute(plan, std::chrono::seconds(30), {}, [owned = io_ticket()] { owned.validate(); });
			require(gather(*memory, plan.destination_segments) == input, "Fresh native IO generation retry failed");
			store(*memory, plan.destination_segments, output);
			pending(8, [owned = io_ticket()] { owned.validate(); });
			provider->execute([&](auto&) { io_epochs->notify({0x500000, 0x100000}); });
			gate.signaledValue = 8;
			transfer.synchronize();
			require(!transfer.has_pending() && gather(*memory, plan.destination_segments) == input, "Unrelated IO entry refused valid native transfer");
			refuse_io(9, false);
			(*mappings)[0].tiling = scaled_image_memory_mapping::tile_range{0x1ffc00, plan.source.range.size, 1};
			(*mappings)[1].tiling = scaled_image_memory_mapping::tile_range{0x9ffc00, plan.destination.range.size, 1};
			const auto tile_ticket = [&]
			{
				std::optional<scaled_image_mapping_epoch> result;
				capture([&]
				{
					result.emplace([mappings] { return *mappings; }, [memory, io_epochs, tile_epochs]
					{
						return scaled_image_mapping_versions{memory->mapping_generation(), io_epochs->current(), tile_epochs->current()};
					}, [memory, io_epochs, tile_epochs](const auto& values)
					{
						return scaled_image_mapping_versions{scaled_image_mapping_generation(values, *memory), scaled_image_io_mapping_generation(values, *io_epochs), scaled_image_tile_mapping_generation(values, *tile_epochs)};
					});
				});
				return std::move(*result);
			};
			for (unsigned event_value : {10u, 11u})
			{
				store(*memory, plan.destination_segments, output);
				const auto original = *mappings;
				const auto& last_target = plan.destination_segments.back();
				const auto target_watch = guest_byte_coverage({last_target.address, last_target.size});
				const auto revision = memory->surface_revision(target_watch);
				pending(event_value, [owned = tile_ticket()] { owned.validate(); });
				const auto notifications = provider->notifications;
				tile_epochs->update(0, {1, event_value == 10 ? 0x300000u : 0xb00000u, 65536, 256, 0, 0, 0, true});
				tile_epochs->update(0, {});
				require(*mappings == original && memory->surface_revision(target_watch) == revision, "Tile ABA changed final metadata or VM revision");
				gate.signaledValue = event_value;
				bool refused = false;
				try { transfer.synchronize(); }
				catch (const scaled_image_mapping_conflict&) { refused = true; }
				require(refused && !transfer.has_pending() && provider->notifications == notifications && gather(*memory, plan.destination_segments) == output, "Absent-tile ABA published stale native image bytes");
			}
			transfer.execute(plan, std::chrono::seconds(30), {}, [owned = tile_ticket()] { owned.validate(); });
			require(gather(*memory, plan.destination_segments) == input, "Fresh tile generation retry failed");
			store(*memory, plan.destination_segments, output);
			pending(12, [owned = tile_ticket()] { owned.validate(); });
			tile_epochs->update(1, {1, 0x500000, 65536, 256, 0, 0, 0, true});
			tile_epochs->update(2, {0, 0x300000, 65536, 256, 0, 0, 0, true});
			gate.signaledValue = 12;
			transfer.synchronize();
			require(!transfer.has_pending() && gather(*memory, plan.destination_segments) == input, "Unrelated tile range/location refused native image publication");
			// Physical aliases between source and target keep memmove-style snapshots.
			plan.destination = source;
			plan.destination.range.address += 32 << 20;
			plan.destination_segments = plan.source_segments;
			for (auto& p : plan.destination_segments)
				p.address += 32 << 20;
			for (unsigned i = 0; i < input.size(); ++i)
				input[i] = std::byte((i * 11 + i / 1024) & 255);
			plan.flip_x = true;
			store(*memory, plan.source_segments, input);
			const auto alias_expected = contiguous(ctx, compute, plan, input, input);
			transfer.execute(plan);
			require(gather(*memory, plan.destination_segments) == alias_expected, "Scattered physical overlap lost captured source");
			// Dirty native storage in a later segment must flush before capture.
			plan.flip_x = false;
			plan.destination = target;
			plan.destination_segments = segments(target.range.address, target.range.size);
			store(*memory, plan.source_segments, input);
			store(*memory, plan.destination_segments, output);
			const auto dirty = surfaces.acquire({plan.source_segments[1].address, {4, 1, 1, surface_color::argb8, false}, 16, {}});
			surfaces.clear(dirty, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 1});
			provider->execute([&](auto&) { memory->notify_mapping_changed({0x300000, 4096}); });
			const auto target_mirror = surfaces.acquire({plan.destination_segments[1].address, {4, 1, 1, surface_color::argb8, false}, 16, {}});
			surfaces.prepare(target_mirror);
			fragment_texture_registers words{};
			words.offset = plan.destination_segments[1].address;
			words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 2u | 8u;
			words.control0 = 1u << 31;
			words.control1 = 0xaae4;
			words.image_rect = (4u << 16) | 1u;
			words.control3 = (1u << 20) | 16u;
			const auto old_texture = textures.resolve(words);
			const auto invalidations = textures.statistics().invalidations;
			transfer.execute(plan);
			require(!surfaces.status(target_mirror).initialized && textures.statistics().invalidations > invalidations, "Later segment cache mirrors were not invalidated");
			const auto retained = make_buffer(ctx, 16, buffer_storage::shared);
			ctx.begin();
			texture_to_buffer(ctx, old_texture.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 1, 1)}, retained, 0);
			ctx.submit().wait();
			for (const auto byte : map_buffer(retained, 0, 16))
				require(byte == std::byte{0xa5}, "Scatter changed retained old native texture");
			const auto dirty_result = gather(*memory, plan.destination_segments);
			for (unsigned i = 0; i < 16; ++i)
				require(dirty_result[plan.destination_segments[1].offset + i] == std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]), "Scatter missed dirty later-segment source");
			std::cout << "Native scattered image:" << cases << " RGB565/ARGB8 linear/Morton/banked/flip cases preserve all storage and guards;linear identity matches source independently,other codecs compared with existing contiguous GPU path;last CAS/map failure publishes no segment,source freeze/fresh retry/IO-remap refusal/identical-value last-segment VM mapping ABA refusal/unrelated gap and dirty-source preservation/independent VM-IO domains/source-target identical IO ABA refusal/unrelated IO accepted/absent source-target tile ABA refusal/unrelated tile range-location accepted/fresh tile retry/physical aliases/later dirty source/cache invalidation/old immutable image passed;no App/UI/guest VM initialized\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
