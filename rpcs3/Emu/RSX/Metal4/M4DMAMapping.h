#pragma once

#include "M4DMAPlan.h"
#include "M4ScaledImageMapping.h"

namespace metal4
{
// Sorted union of unique logical byte intervals. State stays bounded by the
// four residue groups; repeated rows do not multiply the selected byte count.
class dma_active_byte_cursor final
{
	dma_active_layout m_active;
	std::array<std::uint32_t, 4> m_index{};
	std::optional<guest_surface_range> m_saved;
	std::optional<guest_surface_range> raw_next()
	{
		unsigned chosen = 4;
		std::uint64_t first = 0x1'0000'0000ull;
		for (unsigned i = 0; i < m_active.group_count; ++i)
		{
			const auto& g = m_active.groups[i];
			if (m_index[i] == g.count) continue;
			const auto offset = std::uint64_t(g.first) + (g.merged ? std::uint64_t(m_index[i]) * m_active.stride :
				std::uint64_t(m_index[i] / g.columns) * g.row_step + (m_index[i] % g.columns) * m_active.stride);
			if (offset < first) { first = offset; chosen = i; }
		}
		if (chosen == 4) return {};
		const auto& g = m_active.groups[chosen];
		const auto count = m_active.stride == 1 ? (g.merged ? g.count : g.columns) : 1;
		m_index[chosen] += count;
		return guest_surface_range{std::uint32_t(first), count};
	}
public:
	explicit dma_active_byte_cursor(const dma_region& region, std::uint32_t columns, std::uint32_t rows)
		: m_active(plan_dma_active(region, columns, rows)) {}
	std::optional<guest_surface_range> next()
	{
		auto result = m_saved ? std::exchange(m_saved, {}) : raw_next();
		if (!result) return {};
		while (const auto following = raw_next())
		{
			if (std::uint64_t(result->address) + result->size == following->address) result->size += following->size;
			else { m_saved = following; break; }
		}
		return result;
	}
};

// Mapping granularity is4KiB. Include only pages containing selected bytes;
// column gaps<=3 bytes cannot hide a whole page. Merge four sorted row streams
// before imposing the range budget, so interleaved groups do not inflate it.
inline std::vector<guest_surface_range> dma_active_pages(const dma_region& region, std::uint32_t columns, std::uint32_t rows)
{
	const auto active = plan_dma_active(region, columns, rows);
	struct stream { std::uint64_t first, width, pitch, rows, row = 0; };
	std::array<stream, 4> streams{};
	for (unsigned i = 0; i < active.group_count; ++i)
	{
		const auto& g = active.groups[i];
		auto& s = streams[i];
		s = {std::uint64_t(region.range.address) + g.first,
			(std::uint64_t(g.merged ? g.count : g.columns) - 1) * active.stride + 1, g.row_step, g.merged ? 1 : g.rows};
		if (s.rows > 1 && s.pitch - s.width < 4096)
		{
			s.width += (s.rows - 1) * s.pitch;
			s.rows = 1;
		}
	}
	std::vector<guest_surface_range> result;
	std::optional<std::pair<std::uint64_t, std::uint64_t>> pending;
	const auto emit = [&]
	{
		if (!pending) return;
		const auto first = std::max<std::uint64_t>(pending->first, region.range.address);
		const auto end = std::min(pending->second, std::uint64_t(region.range.address) + region.range.size);
		if (result.size() == 4096) throw std::length_error("Native DMA active mapping exceeds range budget");
		result.push_back({std::uint32_t(first), std::size_t(end - first)});
	};
	for (;;)
	{
		unsigned chosen = 4;
		std::uint64_t first = 0x1'0000'0000ull, end = 0;
		for (unsigned i = 0; i < active.group_count; ++i)
		{
			const auto& s = streams[i];
			if (s.row == s.rows) continue;
			const auto start = s.first + s.row * s.pitch;
			const auto page = start & ~4095ull;
			if (page < first) { chosen = i; first = page; end = (start + s.width + 4095) & ~4095ull; }
		}
		if (chosen == 4) break;
		++streams[chosen].row;
		if (pending && first <= pending->second) pending->second = std::max(pending->second, end);
		else { emit(); pending = {{first, end}}; }
	}
	emit();
	return result;
}

template <typename Translate>
scaled_image_memory_mapping probe_dma_mapping(const dma_region& region, std::uint32_t columns, std::uint32_t rows,
	unsigned dma, const Translate& translate, std::optional<std::uint32_t> io_base = {})
{
	if (!region.segments.empty() || region.sparse_segments) throw std::invalid_argument("Native DMA mapping probe requires logical untransformed geometry");
	scaled_image_memory_mapping result{0, region.range.size, {}};
	if (io_base)
	{
		validate_guest_surface_range({*io_base, region.range.size});
		result.io = scaled_image_memory_mapping::io_range{*io_base, region.range.size};
	}
	for (const auto range : dma_active_pages(region, columns, rows))
		for (std::size_t delta = 0; delta < range.size;)
		{
			const auto logical = std::uint32_t(std::uint64_t(range.address) + delta);
			const auto count = std::min<std::size_t>(range.size - delta, 0x100000 - (logical & 0xfffff));
			const auto address = translate(logical, dma, count);
			validate_guest_surface_range({address, count});
			const auto offset = std::size_t(logical - region.range.address);
			if (!result.segments.empty() && result.segments.back().offset + result.segments.back().size == offset &&
				std::uint64_t(result.segments.back().address) + result.segments.back().size == address) result.segments.back().size += count;
			else
			{
				if (result.segments.size() == 4096) throw std::length_error("Native DMA translated segments exceed budget");
				result.segments.push_back({address, offset, count});
			}
			delta += count;
		}
	result.address = result.segments.front().address;
	return result;
}
inline void attach_dma_mapping(dma_region& region, const scaled_image_memory_mapping& mapped)
{
	if (mapped.size != region.range.size || mapped.segments.empty() || mapped.address != mapped.segments.front().address || mapped.tile || mapped.tiling)
		throw std::invalid_argument("Native DMA mapping differs from logical geometry");
	region.segments = mapped.segments;
	region.sparse_segments = true;
	validate_dma_segments(region);
}
template <typename Memory, typename IO>
scaled_image_mapping_versions dma_mapping_versions(const scaled_image_mapping_epoch::mappings& values, const Memory& memory, const IO& io)
{
	scaled_image_mapping_versions result;
	for (const auto& value : values)
		for (const auto& part : value.segments)
		{
			result.vm = std::max(result.vm, memory.mapping_generation({part.address, part.size}));
			if (value.io) result.io = std::max(result.io, io.current({std::uint32_t(value.io->offset + part.offset), part.size}));
		}
	return result;
}
// Called after geometry/mapping validation. Translate byte-exact logical runs
// into the captured segments without treating unselected page bytes as data.
template <typename Visit>
void for_each_dma_active_range(const dma_region& region, std::uint32_t columns, std::uint32_t rows, const Visit& visit)
{
	(void)project_dma_active(region, columns, rows);
	dma_active_byte_cursor cursor(region, columns, rows);
	std::size_t segment_index = 0;
	while (const auto run = cursor.next())
	{
		if (region.segments.empty()) { visit(guest_surface_range{region.range.address + run->address, run->size}); continue; }
		std::size_t offset = run->address, end = offset + run->size;
		while (offset < end)
		{
			while (segment_index < region.segments.size() && region.segments[segment_index].offset + region.segments[segment_index].size <= offset) ++segment_index;
			if (segment_index == region.segments.size() || region.segments[segment_index].offset > offset) throw std::logic_error("Native DMA active barrier lost mapping");
			const auto& part = region.segments[segment_index];
			const auto count = std::min(end - offset, part.offset + part.size - offset);
			visit(guest_surface_range{std::uint32_t(part.address + offset - part.offset), count});
			offset += count;
		}
	}
}
} // namespace metal4
