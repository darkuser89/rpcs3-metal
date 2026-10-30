#include "M4DMAPlan.h"
#include <array>
#include <iostream>
#include <set>
namespace
{
using namespace metal4;
void require(bool v)
{
	if (!v)
		throw std::runtime_error("Native DMA plan differs");
}
template <typename F>
void refused(F&& op)
{
	bool failed = false;
	try
	{
		op();
	}
	catch (const std::exception&)
	{
		failed = true;
	}
	require(failed);
}
void check_projection(const dma_region& region, unsigned columns, unsigned rows)
{
	const auto active = plan_dma_active(region, columns, rows);
	const auto parts = project_dma_active(region, columns, rows);
	require(parts.size() <= 3 * active.group_count * std::max<std::size_t>(1, region.segments.size()));
	std::vector<bool> visited(active.bytes);
	for (const auto& part : parts)
	{
		std::size_t packed = part.packed_offset;
		guest_surface_byte_cursor cursor(part.coverage);
		while (const auto range = cursor.next()) for (std::size_t i = 0; i < range->size; ++i)
		{
			const auto logical = part.logical_offset + range->address - part.coverage.range.address + i;
			require(packed < visited.size() && !visited[packed] && dma_active_offset(active, packed) == logical);
			visited[packed++] = true;
			std::size_t address = region.range.address + logical;
			for (const auto& segment : region.segments)
				if (logical >= segment.offset && logical - segment.offset < segment.size) address = segment.address + logical - segment.offset;
			require(address == std::size_t(range->address) + i);
		}
	}
	require(std::all_of(visited.begin(), visited.end(), [](bool v) { return v; }));
}
} // namespace
int main()
{
	try
	{
		unsigned cases = 0;
		for (unsigned columns : {1u, 7u, 31u})
			for (unsigned rows : {1u, 3u, 9u})
				for (unsigned stride = 1; stride <= 4; ++stride)
					for (int pitch : {-131, -3, 0, 13, 137})
					{
						const auto region = plan_dma_region(4096, pitch, stride, columns, rows);
						std::uint32_t lo = 0xffffffff, hi = 0;
						std::set<std::uint32_t> expected;
						for (unsigned r = 0; r < rows; ++r)
							for (unsigned c = 0; c < columns; ++c)
							{
								const auto address = std::uint32_t(std::int64_t(4096) + std::int64_t(r) * pitch + c * stride);
								expected.insert(address);
								lo = std::min(lo, address);
								hi = std::max(hi, address);
							}
						require(region.range.address == lo && region.range.size == std::size_t(hi - lo) + 1 && region.first == 4096 - lo);
						validate_dma({region, region, columns, rows}, 1 << 20);
						const auto active = plan_dma_active(region, columns, rows);
						require(active.bytes == expected.size() && active.group_count <= stride);
						std::set<std::uint32_t> actual;
						for (unsigned i = 0; i < active.bytes; ++i)
							require(actual.insert(region.range.address + dma_active_offset(active, i)).second);
						require(actual == expected);
						check_projection(region, columns, rows);
						refused([&] { (void)dma_active_offset(active, active.bytes); });
						++cases;
					}
		unsigned exhaustive = 0;
		for (unsigned columns = 1; columns <= 12; ++columns)
			for (unsigned rows = 1; rows <= 12; ++rows)
				for (unsigned stride = 1; stride <= 4; ++stride)
					for (int pitch = -24; pitch <= 24; ++pitch)
					{
						const auto region = plan_dma_region(4096, pitch, stride, columns, rows);
						const auto active = plan_dma_active(region, columns, rows);
						std::set<unsigned> expected, actual;
						for (unsigned row = 0; row < rows; ++row) for (unsigned column = 0; column < columns; ++column)
							expected.insert(4096 + int(row) * pitch + column * stride);
						for (unsigned i = 0; i < active.bytes; ++i) require(actual.insert(region.range.address + dma_active_offset(active, i)).second);
						require(expected == actual && actual.size() == active.bytes); ++exhaustive;
						auto scattered = region;
						if (region.range.size > 1)
						{
							const auto split = region.range.size / 2;
							scattered.segments = {{region.range.address, 0, split}, {0x80000, split, region.range.size - split}};
						}
						check_projection(scattered, columns, rows);
					}
		std::cout << exhaustive << " exhaustive small signed-pitch/stride active-byte sets matched independent row/column enumeration\n";
		for (unsigned stride : {0u, 5u, 0xffffffffu})
			refused([&]
					{ (void)plan_dma_region(4096, 4, stride, 3, 5); });
		refused([]
				{ (void)plan_dma_region(0, -1, 1, 1, 2); });
		refused([]
				{ (void)plan_dma_region(0xffffffff, 1, 2, 2, 1); });
		refused([]
				{ (void)plan_dma_region(4096, 0x7fffffff, 4, 0xffffffff, 0xffffffff); });
		refused([]
				{ (void)plan_dma_region(4096, std::numeric_limits<std::int32_t>::min(), 4, 0xffffffff, 0xffffffff); });
		refused([]
				{ (void)plan_dma_region(4096, 1, 1, 0, 5); });
		refused([]
				{ (void)plan_dma_region(4096, 1, 1, 5, 0); });
		const auto end = plan_dma_region(0xffffffff, 0, 1, 1, 0xffffffff);
		require(end.range.size == 1 && plan_dma_active(end, 1, 0xffffffff).bytes == 1);
		const auto sparse = plan_dma_region(0, 1048576, 1, 4, 2);
		require(sparse.range.size == 1048580 && plan_dma_active(sparse, 4, 2).bytes == 8);
		validate_dma({sparse, sparse, 4, 2}, 16, true);
		refused([&] { validate_dma({sparse, sparse, 4, 2}, 15, true); });
		refused([&] { validate_dma({sparse, sparse, 4, 2}, 16); });
		const auto negative = plan_dma_region(0x80000000u, std::numeric_limits<std::int32_t>::min(), 1, 1, 2);
		require(plan_dma_active(negative, 1, 2).bytes == 2);
		validate_dma({negative, negative, 1, 2}, 4, true);
		check_projection(negative, 1, 2);
		const auto matrix = plan_dma_region(0, 4099, 3, 1025, 1025);
		require(project_dma_active(matrix, 1025, 1025).size() <= 3);
		check_projection(matrix, 1025, 1025);
		std::cout << "Active segment projection:28224 split-storage profiles,180 contiguous profiles and1,050,625 strided bytes with<=3 watch matrices;packed offsets,exact EAs andno gaps/duplicates verified\n";
		const auto huge = plan_dma_region(0, 0, 4, 0x40000000, 0xffffffff);
		require(plan_dma_active(huge, 0x40000000, 0xffffffff).bytes == 0x40000000);
		std::cout << "Active DMA residue groups:exact independent byte sets/no duplicate dispatch;1MiB row gap gives8 threads;UINT_MAX repeated rows collapse;INT_MIN pitch and1GiB sparse count checked\n";
		auto plan = plan_dma({4096, 8192, 1, 2, 7, 3, -19, 23, 2, 3, 0});
		validate_dma(plan, 1024);
		refused([&]
				{ validate_dma(plan, 1); });
		++plan.source.first;
		refused([&]
				{ validate_dma(plan, 1024); });
		--plan.source.first;
		plan.destination.segments = {{plan.destination.range.address, 1, plan.destination.range.size}};
		refused([&]
				{ validate_dma(plan, 1024); });
		std::cout << "Native DMA plan:" << cases << " independent signed-row/stride extents;negative/zero/overlapping rows,32bit endpoints/extreme products/invalid dimensions/strides/aggregate budgets/segment mismatch checked;no guest VM initialized\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
