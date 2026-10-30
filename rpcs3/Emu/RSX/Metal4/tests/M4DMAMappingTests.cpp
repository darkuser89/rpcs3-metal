#include "M4RSXDMA.h"
#include "M4IOMappingEpoch.h"
#include "M4MappingEpoch.h"
#include <iostream>
#include <set>
#include <source_location>

namespace
{
using namespace metal4;
void require(bool value, const std::source_location& where = std::source_location::current())
{
	if (!value) throw std::runtime_error("Active DMA mapping differs at line " + std::to_string(where.line()));
}
template <typename F> void refused(F&& operation)
{
	bool failed = false;
	try { operation(); } catch (const std::exception&) { failed = true; }
	require(failed);
}
struct memory_versions
{
	mapping_epoch epochs{1000};
	std::uint64_t mapping_generation(guest_surface_range range) const { return epochs.current(range); }
};
}
int main()
{
	try
	{
		unsigned profiles = 0;
		for (unsigned origin : {0x30000u, 0x30ffdu, 0x30fffu})
			for (unsigned columns : {1u, 3u, 17u}) for (unsigned rows : {1u, 2u, 9u}) for (unsigned stride = 1; stride <= 4; ++stride)
				for (int pitch : {-8193, -4096, -4095, -31, -3, 0, 7, 4095, 4096, 8193})
				{
					const auto region = plan_dma_region(origin, pitch, stride, columns, rows);
					std::set<std::uint32_t> expected, pages, observed_pages;
					for (unsigned r = 0; r < rows; ++r) for (unsigned c = 0; c < columns; ++c)
					{
						const auto address = std::uint32_t(std::int64_t(origin) + std::int64_t(r) * pitch + c * stride);
						expected.insert(address); pages.insert(address / 4096);
					}
					for (const auto range : dma_active_pages(region, columns, rows))
						for (auto page = range.address / 4096; page <= (std::uint64_t(range.address) + range.size - 1) / 4096; ++page) observed_pages.insert(page);
					require(pages == observed_pages);
					std::set<std::uint32_t> actual;
					for_each_dma_active_range(region, columns, rows, [&](auto range)
					{
						for (std::size_t i = 0; i < range.size; ++i) require(actual.insert(range.address + std::uint32_t(i)).second);
					});
					require(actual == expected);
					const auto translate = [&](std::uint32_t io, unsigned, std::size_t count)
					{
						require(count && count <= 0x100000 - (io & 0xfffff));
						for (auto page = io / 4096; page <= (std::uint64_t(io) + count - 1) / 4096; ++page) require(pages.contains(page));
						return 0xa0000000u + io;
					};
					const auto mapping = probe_rsx_dma_mapping(region, columns, rows, CELL_GCM_LOCATION_MAIN, translate);
					auto mapped = region; attach_dma_mapping(mapped, mapping);
					validate_dma({mapped, mapped, columns, rows}, expected.size() * 2, true);
					actual.clear();
					for_each_dma_active_range(mapped, columns, rows, [&](auto range)
					{
						for (std::size_t i = 0; i < range.size; ++i) require(actual.insert(range.address - 0xa0000000u + std::uint32_t(i)).second);
					});
					require(actual == expected); ++profiles;
				}
		std::cout << profiles << " independent signed/zero/overlap/page-edge profiles:exact selected pages,logical andtranslated active barriers,no gaps orduplicate bytes passed\n";
		{
			const auto logical = plan_dma_region(0x200000, 1048576, 3, 4, 2);
			unsigned calls = 0;
			const auto translate = [&](unsigned io, unsigned, std::size_t count)
			{
				++calls;
				if (io == 0x200000 && count == 4096) return 0xfffff000u;
				if (io == 0x300000 && count == 10) return 0x1000u;
				throw std::runtime_error("Unmapped pitch gap was probed");
			};
			const auto mapping = probe_rsx_dma_mapping(logical, 4, 2, CELL_GCM_LOCATION_MAIN, translate);
			require(calls == 2 && mapping.segments.size() == 2);
			auto mapped = logical; attach_dma_mapping(mapped, mapping);
			validate_dma({mapped, mapped, 4, 2}, 16, true);
			require(mapped.range.address == logical.range.address && project_dma_active(mapped, 4, 2).size() == 2);
			auto missing = mapped; missing.segments.pop_back();
			refused([&] { validate_dma({missing, mapped, 4, 2}, 16, true); });
			auto overlapping = mapped; overlapping.segments[1].offset = 1;
			refused([&] { validate_dma({overlapping, mapped, 4, 2}, 16, true); });
			unsigned barriers = 0; refused([&] { for_each_dma_active_range(missing, 4, 2, [&](auto) { ++barriers; }); }); require(!barriers);
			std::cout << "Sparse mapping:unmapped1MiB pitch gaps never probed;last4KiB-EA pluslow disjoint EA accepted without fictitious envelope;missing/overlap mappings refuse before barriers\n";
		}
		const auto identity = [](unsigned address, unsigned, std::size_t) { return address; };
		{
			const auto region = plan_dma_region(0xfffff0, 0, 1, 16, 1);
			for (unsigned location : {CELL_GCM_LOCATION_MAIN, CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER})
				require(probe_rsx_dma_mapping(region, 16, 1, location, identity).io->offset == region.range.address);
			for (unsigned location : {CELL_GCM_LOCATION_LOCAL, CELL_GCM_CONTEXT_DMA_MEMORY_FRAME_BUFFER, CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_LOCAL})
				require(!probe_rsx_dma_mapping(region, 16, 1, location, identity).io);
			require(probe_rsx_dma_mapping(region, 16, 1, CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_MAIN, identity).io->offset == 0x0efffff0);
			refused([&] { (void)probe_rsx_dma_mapping(plan_dma_region(0xfffff0, 0, 1, 17, 1), 17, 1, CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_MAIN, identity); });
			const auto extreme = plan_dma_region(0x80000000u, std::numeric_limits<std::int32_t>::min(), 1, 1, 2);
			auto mapped = extreme; attach_dma_mapping(mapped, probe_dma_mapping(extreme, 1, 2, 0, identity)); validate_dma({mapped, mapped, 1, 2}, 4, true);
			const auto repeated = plan_dma_region(0xffffffe0u, 0, 4, 7, 0xffffffffu);
			require(dma_active_pages(repeated, 7, 0xffffffffu).size() == 1);
			unsigned bytes = 0; for_each_dma_active_range(repeated, 7, 0xffffffffu, [&](auto r) { bytes += r.size; }); require(bytes == 7);
			refused([&] { (void)dma_active_pages(plan_dma_region(0, 8192, 1, 1, 4097), 1, 4097); });
			// Three individually sparse group streams combine into adjacent pages.
			require(dma_active_pages(plan_dma_region(0, 4096, 3, 1, 20000), 1, 20000).size() == 1);
			std::cout << "Main/local/host/frame/report IO domains,report aperture,INT_MIN pitch,UINT_MAX repeated rows,range budget andinterleaved page-stream merging passed\n";
		}
		{
			memory_versions memory;
			io_mapping_epoch io;
			const auto plan = plan_dma({0x200000, 0x800000, CELL_GCM_LOCATION_MAIN, CELL_GCM_LOCATION_MAIN, 4, 2, 2097152, 2097152, 3, 4, 0});
			std::function<void()> hook;
			unsigned translated_delta = 0;
			const auto query = [&]
			{
				const auto translate = [&](unsigned address, unsigned, std::size_t size)
				{
					if (hook) { auto op = std::exchange(hook, {}); op(); }
					require((address >> 20) == 2 || (address >> 20) == 4 || (address >> 20) == 8 || (address >> 20) == 10);
					require(size <= 4096);
					return address + translated_delta;
				};
				return scaled_image_mapping_epoch::mappings{
					probe_rsx_dma_mapping(plan.source, 4, 2, CELL_GCM_LOCATION_MAIN, translate),
					probe_rsx_dma_mapping(plan.destination, 4, 2, CELL_GCM_LOCATION_MAIN, translate)};
			};
			const auto ticket = [&]
			{
				return scaled_image_mapping_epoch(query, [&] { return scaled_image_mapping_versions{memory.epochs.current(), io.current()}; },
					[&](const auto& values) { return dma_mapping_versions(values, memory, io); });
			};
			auto stable = ticket();
			memory.epochs.notify({0x300000, 4096}); io.notify({0x300000, 1048576}); stable.validate();
			memory.epochs.notify({0x900000, 4096}); io.notify({0x900000, 1048576}); stable.validate();
			for (unsigned address : {0x200000u, 0x400000u, 0x800000u, 0xa00000u})
			{
				auto before_vm = ticket(); const auto original = query();
				memory.epochs.notify({address, 4096}); memory.epochs.notify({address, 4096});
				require(query() == original); refused([&] { before_vm.validate(); });
				auto before_io = ticket(); io.notify({address, 1048576}); io.notify({address, 1048576});
				require(query() == original && memory.epochs.current() > io.current()); refused([&] { before_io.validate(); });
			}
			auto changed = ticket(); translated_delta = 4096; refused([&] { changed.validate(); }); translated_delta = 0;
			hook = [&] { memory.epochs.notify({0x300000, 4096}); }; refused([&] { (void)ticket(); });
			auto raced = ticket(); hook = [&] { io.notify({0x400000, 1048576}); }; refused([&] { raced.validate(); });
			ticket().validate();
			std::cout << "Active mapping epochs:unrelated VM/IO gap changes accepted;all4 source/target endpoints reject identical-value VM andIO ABA independently;changed translation,initial/during-validation transitions andfresh retry passed\n";
		}
		std::cout << "No RPCS3 App/UI/GSRender/guest VM started\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
