#pragma once

#include "M4GuestSegments.h"

#include <algorithm>

namespace metal4
{
// Missing consumed bytes inside the source allocation, in original storage
// order. Row padding is excluded. A null result means some consumed byte is
// outside the owned allocation; empty means the image is fully native.
[[nodiscard]] inline std::optional<std::vector<guest_surface_range>> morton_display_holes(
	const guest_surface_layout& source, const guest_surface_layout& display, std::size_t offset)
{
	if (!source.morton || source.range.address || display.range.address || display.morton || display.column_stride || !display.width || !display.height) return {};
	guest_surface_byte_cursor active(source), requested(display);
	auto available = active.next();
	std::vector<guest_surface_range> holes;
	std::size_t comparisons = 0;
	while (const auto run = requested.next())
	{
		if (run->address > source.range.size || offset > source.range.size - run->address) return {};
		auto position = offset + run->address;
		if (run->size > source.range.size - position) return {};
		const auto end = position + run->size;
		while (position < end)
		{
			if (++comparisons > (1u << 20)) throw std::length_error("Mixed Morton display proof exceeds run budget");
			if (available && std::uint64_t(available->address) + available->size <= position) { available = active.next(); continue; }
			if (available && available->address <= position)
				position = std::min(end, std::size_t(available->address) + available->size);
			else
			{
				const auto next = available ? std::min(end, std::size_t(available->address)) : end;
				holes.push_back({static_cast<std::uint32_t>(position), next - position});
				position = next;
			}
		}
	}
	return holes;
}
// Prove only bytes consumed by linear scanout, not its row padding. Every
// consumed byte must belong to the native source's active Morton box; a raw
// allocation bound alone cannot establish that an unwritten cell has pixels.
[[nodiscard]] inline bool morton_display_bytes_active(const guest_surface_layout& source,
	const guest_surface_layout& display, std::size_t offset)
{
	if (!source.morton || source.range.address || display.range.address || display.morton || display.column_stride || !display.width || !display.height)
		return false;
	guest_surface_byte_cursor active(source), requested(display);
	auto available = active.next();
	std::size_t comparisons = 0;
	while (const auto run = requested.next())
	{
		if (++comparisons > (1u << 20)) throw std::length_error("Morton display proof exceeds run budget");
		if (run->address > source.range.size || offset > source.range.size - run->address) return false;
		const auto start = offset + run->address;
		if (run->size > source.range.size - start) return false;
		while (available && std::uint64_t(available->address) + available->size <= start)
		{
			if (++comparisons > (1u << 20)) throw std::length_error("Morton display proof exceeds run budget");
			available = active.next();
		}
		if (!available || start < available->address || start - available->address > available->size ||
			run->size > available->size - (start - available->address)) return false;
	}
	return true;
}
// Resolve the logical byte offset through one stable physical-identity query.
// Ordered segments may split a texel/AV row. Prove every active byte in both
// address spaces; neither a matching first address nor logical overlap proves
// the remainder. This path does not inverse-map banked main-memory storage.
// Opt-in inactive cells require the caller to capture the exact missing bytes
// and retain a current AV receipt; allocation identity is not pixel authority.
[[nodiscard]] inline std::optional<std::size_t> plan_morton_display_storage(guest_surface_memory& memory,
	const guest_surface_layout& source, std::span<const guest_storage_segment> source_segments,
	const guest_surface_layout& display, std::span<const guest_storage_segment> requested,
	std::optional<std::size_t> logical_anchor = {}, bool allow_inactive = false)
{
	if (!source.morton || source.range.address || display.range.address || source_segments.empty() || requested.empty()) return {};
	auto storage = source; storage.range.address = source_segments.front().address;
	validate_guest_segments(storage, source_segments);
	storage = display; storage.range.address = requested.front().address;
	validate_guest_segments(storage, requested);
	const auto segment_at = [](auto parts, std::size_t offset) -> const guest_storage_segment*
	{
		const auto end = std::upper_bound(parts.begin(), parts.end(), offset,
			[](auto value, const auto& part) { return value < part.offset; });
		if (end == parts.begin()) return nullptr;
		const auto& part = *std::prev(end);
		return offset - part.offset < part.size ? &part : nullptr;
	};
	std::optional<std::size_t> result;
	bool ambiguous = false;
	const auto prove = [&](const guest_surface_memory::storage_rebase& rebase, const guest_surface_memory::storage_equivalent& equal)
	{
		std::size_t comparisons = 0;
		const auto candidate = [&](std::size_t offset)
		{
			if (allow_inactive ? !morton_display_holes(source, display, offset) : !morton_display_bytes_active(source, display, offset)) return;
			guest_surface_byte_cursor cursor(display);
			while (const auto run = cursor.next())
				for (std::size_t position = run->address, remaining = run->size; remaining;)
				{
					if (++comparisons > (1u << 20)) throw std::length_error("Morton display identity proof exceeds run budget");
					const auto* src = segment_at(source_segments, offset + position);
					const auto* dst = segment_at(requested, position);
					if (!src || !dst) return;
					const auto a = offset + position - src->offset, b = position - dst->offset;
					const auto count = std::min({remaining, src->size - a, dst->size - b});
					if (!equal({static_cast<std::uint32_t>(std::uint64_t(src->address) + a), count},
					{static_cast<std::uint32_t>(std::uint64_t(dst->address) + b), count})) return;
					position += count; remaining -= count;
				}
			ambiguous |= result && *result != offset;
			result = offset;
		};
		if (logical_anchor) candidate(*logical_anchor);
		else for (const auto& part : source_segments)
			if (const auto anchor = rebase({part.address, part.size}, {requested.front().address, 1}))
				candidate(part.offset + std::size_t(anchor->address - part.address));
		return bool(result) && !ambiguous;
	};
	(void)memory.test_rebase(prove);
	return ambiguous ? std::nullopt : result;
}
} // namespace metal4
