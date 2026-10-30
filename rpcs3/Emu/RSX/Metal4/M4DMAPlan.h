#pragma once

#include "M4GuestSegments.h"
#include <algorithm>
#include <bit>
#include <array>
#include <numeric>
#include <limits>

namespace metal4
{
struct dma_command
{
	std::uint32_t source, destination, source_dma, destination_dma, columns, rows;
	std::int32_t source_pitch, destination_pitch;
	std::uint32_t source_stride, destination_stride, notify;
	bool operator==(const dma_command&) const = default;
};
struct dma_region
{
	guest_surface_range range;
	std::uint32_t first;
	std::int32_t pitch;
	std::uint32_t stride;
	std::vector<guest_storage_segment> segments;
	// Geometry remains in logical RSX space. Segments may omit inactive pages;
	// their addresses are translated EAs and offsets refer to that geometry.
	bool sparse_segments = false;
};
struct dma_plan
{
	dma_region source, destination;
	std::uint32_t columns, rows;
};
inline dma_region plan_dma_region(std::uint32_t origin, std::int32_t pitch, std::uint32_t stride,
								  std::uint32_t columns, std::uint32_t rows)
{
	if (!columns || !rows || !stride || stride > 4)
		throw std::invalid_argument("Native DMA requires nonempty dimensions and byte strides1..4");
	const std::int64_t last_row = std::int64_t(pitch) * (rows - 1);
	const auto first = std::int64_t(origin) + std::min<std::int64_t>(0, last_row);
	const auto end = std::uint64_t(origin) + std::uint64_t(std::max<std::int64_t>(0, last_row)) + std::uint64_t(columns - 1) * stride + 1;
	if (first < 0 || end > 0x1'0000'0000ull)
		throw std::overflow_error("Native DMA signed row extent wraps RSX address space");
	return {{std::uint32_t(first), std::size_t(end - std::uint64_t(first))}, std::uint32_t(std::int64_t(origin) - first), pitch, stride, {}};
}
inline dma_plan plan_dma(const dma_command& command)
{
	return {plan_dma_region(command.source, command.source_pitch, command.source_stride, command.columns, command.rows),
			plan_dma_region(command.destination, command.destination_pitch, command.destination_stride, command.columns, command.rows), command.columns, command.rows};
}
inline void validate_dma_segments(const dma_region& region)
{
	if (!region.sparse_segments)
	{
		validate_guest_segments(guest_byte_coverage(region.range), region.segments);
		return;
	}
	if (region.segments.empty() || region.segments.size() > 4096)
		throw std::invalid_argument("Native sparse DMA mapping segment count is invalid");
	std::size_t end = 0;
	for (const auto& part : region.segments)
	{
		validate_guest_surface_range({part.address, part.size});
		if (!part.size || part.offset < end || part.offset > region.range.size || part.size > region.range.size - part.offset)
			throw std::invalid_argument("Native sparse DMA mapping segments overlap or exceed logical storage");
		end = part.offset + part.size;
	}
}
inline void validate_dma_geometry(const dma_plan& plan)
{
	for (const auto* region : {&plan.source, &plan.destination})
	{
		validate_guest_surface_range(region->range);
		if (region->first > 0xffff'ffffu - region->range.address)
			throw std::overflow_error("Native DMA first row address wraps");
		const auto expected = plan_dma_region(region->range.address + region->first, region->pitch, region->stride, plan.columns, plan.rows);
		if (expected.range.address != region->range.address || expected.range.size != region->range.size || expected.first != region->first)
			throw std::invalid_argument("Native DMA region differs from signed row extent");
		validate_dma_segments(*region);
	}
}
// Each group contains one residue class modulo the byte stride. Rows in
// that class form equal-length intervals with a fixed nonnegative step.
// Overlapping intervals merge; separated intervals retain row-sized runs.
struct dma_active_group
{
	std::uint32_t first = 0, row_step = 0, columns = 0, rows = 0, count = 0, merged = 0;
};
struct dma_active_layout
{
	std::array<dma_active_group, 4> groups{};
	std::uint32_t group_count = 0, bytes = 0, stride = 0;
};
inline dma_active_layout plan_dma_active(const dma_region& region, std::uint32_t columns, std::uint32_t rows)
{
	if (region.first > 0xffff'ffffu - region.range.address)
		throw std::overflow_error("Native active DMA first row wraps");
	const auto expected = plan_dma_region(region.range.address + region.first, region.pitch, region.stride, columns, rows);
	if (expected.range.address != region.range.address ||
		expected.range.size != region.range.size || expected.first != region.first)
		throw std::invalid_argument("Native active DMA layout differs from its row extent");
	const auto pitch = std::uint64_t(region.pitch < 0 ? -std::int64_t(region.pitch) : region.pitch);
	const auto period = region.stride / std::gcd(pitch, std::uint64_t(region.stride));
	dma_active_layout result;
	result.stride = region.stride;
	std::uint64_t total = 0;
	for (std::uint32_t first_row = 0; first_row < std::min<std::uint64_t>(period, rows); ++first_row)
	{
		const auto row_count = (std::uint64_t(rows) - 1 - first_row) / period + 1;
		const auto last_row = first_row + (row_count - 1) * period;
		const auto first = std::int64_t(region.first) + std::int64_t(region.pitch) * std::int64_t(region.pitch < 0 ? last_row : first_row);
		const auto step = row_count > 1 ? pitch * period : 0;
		const bool merged = step / region.stride <= columns;
		const auto count = merged ? columns + (row_count - 1) * (step / region.stride) : row_count * columns;
		if (first < 0 || std::uint64_t(first) > 0xffff'ffffull || step > 0xffff'ffffull || count > 0xffff'ffffull - total)
			throw std::length_error("Native active DMA dispatch exceeds 32-bit byte indexing");
		result.groups[result.group_count++] = {std::uint32_t(first), std::uint32_t(step), columns,
			std::uint32_t(row_count), std::uint32_t(count), std::uint32_t(merged)};
		total += count;
	}
	result.bytes = std::uint32_t(total);
	return result;
}
inline std::uint32_t dma_active_offset(const dma_active_layout& layout, std::uint32_t index)
{
	for (std::uint32_t i = 0; i < layout.group_count; ++i)
	{
		const auto& group = layout.groups[i];
		if (index >= group.count) { index -= group.count; continue; }
		return group.first + (group.merged ? index * layout.stride :
			(index / group.columns) * group.row_step + (index % group.columns) * layout.stride);
	}
	throw std::out_of_range("Native active DMA index exceeds selected bytes");
}
struct dma_storage_part
{
	guest_surface_layout coverage;
	std::size_t packed_offset, logical_offset;
};
// Intersect each stride-residue group with translated segments. At most a
// partial first row, a full row matrix and a partial last row per intersection;
// metadata is independent of row count and of the envelope's inactive bytes.
inline std::vector<dma_storage_part> project_dma_active(const dma_region& region, std::uint32_t columns, std::uint32_t rows)
{
	const auto active = plan_dma_active(region, columns, rows);
	validate_dma_segments(region);
	const guest_storage_segment single{region.range.address, 0, region.range.size};
	const auto segments = region.segments.empty() ? std::span<const guest_storage_segment>{&single, 1} : std::span<const guest_storage_segment>{region.segments};
	std::vector<dma_storage_part> result;
	std::size_t base = 0, covered = 0;
	for (unsigned group = 0; group < active.group_count; ++group)
	{
		const auto& g = active.groups[group];
		const std::size_t cols = g.merged ? g.count : g.columns, count = g.merged ? 1 : g.rows;
		const std::size_t width = (cols - 1) * active.stride + 1, pitch = g.merged ? width : g.row_step;
		const auto end = std::size_t(g.first) + (count - 1) * pitch + width;
		for (const auto& segment : segments)
		{
			const auto lo = std::max<std::size_t>(g.first, segment.offset), hi = std::min(end, segment.offset + segment.size);
			if (lo >= hi) continue;
			const auto add = [&](std::size_t row, std::size_t col, std::size_t ncols, std::size_t nrows)
			{
				if (!ncols || !nrows) return;
				const auto logical = g.first + row * pitch + col * active.stride;
				const auto address = static_cast<std::uint32_t>(segment.address + logical - segment.offset);
				result.push_back({plan_guest_byte_matrix(address, ncols, nrows, active.stride, nrows == 1 ? 0 : pitch),
					base + row * cols + col, logical});
				covered += ncols * nrows;
			};
			const auto partial = [&](std::size_t row)
			{
				const auto start = g.first + row * pitch;
				const auto first_col = lo > start ? (lo - start + active.stride - 1) / active.stride : 0;
				const auto past_col = std::min(cols, (hi - start + active.stride - 1) / active.stride);
				if (first_col < past_col) add(row, first_col, past_col - first_col, 1);
			};
			auto row = (lo - g.first) / pitch;
			const auto last = std::min(count - 1, (hi - 1 - g.first) / pitch);
			if (lo > g.first + row * pitch) { partial(row); ++row; }
			const auto full_end = hi >= std::size_t(g.first) + width ? std::min(count, (hi - g.first - width) / pitch + 1) : 0;
			if (row < full_end) { add(row, 0, cols, full_end - row); row = full_end; }
			if (row <= last) partial(row);
		}
		base += g.count;
	}
	if (covered != active.bytes) throw std::invalid_argument("Native sparse DMA mappings omit active bytes");
	return result;
}
inline void validate_dma(const dma_plan& plan, std::size_t limit, bool packed = false)
{
	validate_dma_geometry(plan);
	const auto source_bytes = packed ? plan_dma_active(plan.source, plan.columns, plan.rows).bytes : plan.source.range.size;
	const auto destination_bytes = packed ? plan_dma_active(plan.destination, plan.columns, plan.rows).bytes : plan.destination.range.size;
	if (source_bytes > limit || destination_bytes > limit - source_bytes ||
		destination_bytes > std::numeric_limits<std::uint32_t>::max())
		throw std::length_error("Native DMA exceeds aggregate snapshot or dispatch budget");
	for (const auto* region : {&plan.source, &plan.destination})
		if (region->sparse_segments) (void)project_dma_active(*region, plan.columns, plan.rows);
}
template <typename Registers>
dma_command capture_dma_command(const Registers& registers, std::uint32_t notify)
{
	return {registers.nv0039_input_offset(), registers.nv0039_output_offset(), registers.nv0039_input_location(), registers.nv0039_output_location(),
			registers.nv0039_line_length(), registers.nv0039_line_count(), std::bit_cast<std::int32_t>(std::uint32_t(registers.nv0039_input_pitch())), std::bit_cast<std::int32_t>(std::uint32_t(registers.nv0039_output_pitch())),
			registers.nv0039_input_format(), registers.nv0039_output_format(), notify};
}
} // namespace metal4
