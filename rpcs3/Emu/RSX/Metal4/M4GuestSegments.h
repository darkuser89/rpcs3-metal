#pragma once

#include "M4GuestCoverage.h"

namespace metal4
{
// Ordered byte ranges in an owned flat storage snapshot. Segment addresses
// are real guest EAs;offset is the corresponding byte offset in that snapshot.
struct guest_storage_segment
{
	std::uint32_t address;
	std::size_t offset, size;
	bool operator==(const guest_storage_segment&) const = default;
};
inline void validate_guest_segments(const guest_surface_layout& image, std::span<const guest_storage_segment> segments)
{
	if (segments.empty())
		return;
	if (segments.size() > 4096 || segments.front().address != image.range.address)
		throw std::invalid_argument("Native image segment origin/count differs from storage");
	std::size_t end = 0;
	for (const auto& segment : segments)
	{
		validate_guest_surface_range({segment.address, segment.size});
		if (!segment.size || segment.offset != end || segment.size > image.range.size - end)
			throw std::invalid_argument("Native image segments must cover storage exactly in order");
		end += segment.size;
	}
	if (end != image.range.size)
		throw std::invalid_argument("Incomplete native image segments");
}
struct guest_storage_part
{
	guest_surface_layout coverage;
	std::size_t offset;
};
// Project active untiled pixels into actual EAs. Whole pitched row groups
// retain one precise watch;partial rows/Morton runs use byte coverages.
// Padding stays outside CAS coverage,including padding at an IO boundary.
inline std::vector<guest_storage_part> project_guest_segments(const guest_surface_layout& image,
	std::span<const guest_storage_segment> segments, bool full_allocation, std::size_t part_limit = 1u << 20)
{
	if (image.column_stride) throw std::invalid_argument("Native image projection does not accept CPU byte matrices");
	validate_guest_segments(image, segments);
	if (segments.empty()) throw std::invalid_argument("Native segmented coverage is empty");
	std::vector<guest_storage_part> result;
	const auto add = [&](guest_surface_layout coverage, std::size_t offset)
	{
		if (result.size() >= part_limit) throw std::length_error("Native segmented active coverage exceeds run budget");
		result.push_back({std::move(coverage), offset});
	};
	if (full_allocation)
	{
		for (const auto& segment : segments) add(guest_byte_coverage({segment.address, segment.size}), segment.offset);
		return result;
	}
	auto logical = image; logical.range.address = 0;
	(void)guest_surface_byte_cursor(logical);
	if (!logical.morton)
	{
		const auto pitch = logical.row_pitch, active = logical.width * logical.pixel_bytes;
		for (const auto& segment : segments)
			for (std::size_t position = segment.offset; position < segment.offset + segment.size;)
			{
				const auto remaining = segment.offset + segment.size - position, column = position % pitch;
				const auto address = static_cast<std::uint32_t>(std::uint64_t(segment.address) + position - segment.offset);
				if (!column && remaining >= pitch)
				{
					const auto rows = remaining / pitch;
					add(plan_guest_surface(address, logical.width, rows, logical.pixel_bytes, pitch), position);
					position += rows * pitch;
				}
				else
				{
					const auto count = std::min(remaining, pitch - column);
					if (column < active) add(guest_byte_coverage({address, std::min(count, active - column)}), position);
					position += count;
				}
			}
		return result;
	}
	guest_surface_byte_cursor cursor(logical);
	std::size_t first = 0;
	while (const auto active = cursor.next())
	{
		const auto end = std::uint64_t(active->address) + active->size;
		while (first < segments.size() && segments[first].offset + segments[first].size <= active->address) ++first;
		for (auto i = first; i < segments.size() && segments[i].offset < end; ++i)
		{
			const auto& segment = segments[i];
			const auto begin = std::max<std::size_t>(active->address, segment.offset);
			const auto count = std::min<std::size_t>(end, segment.offset + segment.size) - begin;
			add(guest_byte_coverage({static_cast<std::uint32_t>(std::uint64_t(segment.address) + begin - segment.offset), count}), begin);
		}
	}
	return result;
}
} // namespace metal4
