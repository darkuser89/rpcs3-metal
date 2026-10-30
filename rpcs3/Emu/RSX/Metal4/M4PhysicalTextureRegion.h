#pragma once

#include "M4GuestSegments.h"
#include "M4SurfaceTextureRegion.h"

namespace metal4
{
// Select and prove the translation in one stable identity callback. Source
// unused padding need not contain the texture's trailing row padding;every
// active requested pixel byte must match the native source storage in order.
[[nodiscard]] inline std::optional<clear_rectangle> plan_physical_texture_region(
	guest_surface_memory& memory, const guest_surface_layout& source, surface_format format,
	const guest_texture_layout& texture, const guest_surface_layout& coverage)
{
	(void)guest_surface_byte_cursor(source);
	(void)guest_surface_byte_cursor(coverage);
	if (!coverage.range.size)
		return {};
	std::optional<clear_rectangle> result;
	const auto prove = [&](const guest_surface_memory::storage_rebase& rebase, const guest_surface_memory::storage_equivalent& equal)
	{
		const auto anchor = rebase(source.range, {coverage.range.address, 1});
		if (!anchor)
			return false;
		const auto rectangle = plan_surface_texture_region(source, format, texture,
			{anchor->address, coverage.range.size});
		if (!rectangle)
			return false;
		guest_surface_byte_cursor cursor(coverage);
		const auto source_end = std::uint64_t(source.range.address) + source.range.size;
		while (const auto active = cursor.next())
		{
			const auto translated = std::uint64_t(anchor->address) + active->address - coverage.range.address;
			if (translated < source.range.address || translated >= source_end || active->size > source_end - translated ||
				!equal({static_cast<std::uint32_t>(translated), active->size}, *active))
				return false;
		}
		result = rectangle;
		return true;
	};
	(void)memory.test_rebase(prove);
	return result;
}
// Prove an untiled physical alias or an explicitly anchored logical RTT
// alias across real ordered storage segments. No VM pixel bytes are read.
// source_origin is the logical pixel-storage base within a whole banked tile;
// the caller supplies its captured logical anchor instead of inverse detiling.
[[nodiscard]] inline std::optional<surface_texture_region> plan_segmented_texture_mapping(
	guest_surface_memory& memory, const guest_surface_layout& source, surface_format format,
	std::span<const guest_storage_segment> source_segments, std::size_t source_origin,
	const guest_texture_layout& texture, std::span<const guest_storage_segment> requested,
	std::optional<std::size_t> logical_anchor = {}, bool allow_cast = false, bool allow_partial_pixels = false)
{
	(void)guest_surface_byte_cursor(source);
	if (source.range.address || source_segments.empty() || requested.empty() ||
		texture.dimension != vertex_texture_dimension::two || texture.mips != 1 || texture.layers != 1 || texture.depth != 1 ||
		texture.stored_border || texture.adjusted_pitch || !texture.codec.block.width || !texture.codec.block.height ||
		((texture.codec.block.width != 1 || texture.codec.block.height != 1) && (!allow_cast || texture.swizzled)))
		return {};
	const auto source_size = source_segments.back().offset + source_segments.back().size;
	auto storage = guest_byte_coverage({0, source_size}); storage.range.address = source_segments.front().address;
	validate_guest_segments(storage, source_segments);
	const auto columns = texture.width / texture.codec.block.width + bool(texture.width % texture.codec.block.width);
	const auto rows = texture.height / texture.codec.block.height + bool(texture.height % texture.codec.block.height);
	auto request = plan_guest_surface(0, columns, rows, texture.codec.block.bytes,
		texture.swizzled ? 0 : texture.subresources.at(0).row_pitch,
		texture.swizzled ? std::optional<swizzle_region>{{{texture.width, texture.height, 1}, {}, {texture.width, texture.height, 1}}} : std::nullopt);
	if (request.range.size != texture.storage_bytes) return {};
	auto request_storage = request; request_storage.range.address = requested.front().address;
	validate_guest_segments(request_storage, requested);
	std::optional<surface_texture_region> result;
	bool ambiguous = false;
	const auto segment_at = [](std::span<const guest_storage_segment> parts, std::size_t offset) -> const guest_storage_segment*
	{
		const auto end = std::upper_bound(parts.begin(), parts.end(), offset,
			[](auto value, const auto& part) { return value < part.offset; });
		if (end == parts.begin()) return nullptr;
		const auto& part = *std::prev(end);
		return offset - part.offset < part.size ? &part : nullptr;
	};
	const auto prove = [&](const guest_surface_memory::storage_rebase& rebase, const guest_surface_memory::storage_equivalent& equal)
	{
		std::size_t comparisons = 0;
		const auto candidate = [&](std::size_t anchor)
		{
			if (anchor < source_origin || anchor - source_origin > std::numeric_limits<std::uint32_t>::max()) return;
			const auto rectangle = plan_surface_texture_mapping(source, format, texture,
				{static_cast<std::uint32_t>(anchor - source_origin), texture.storage_bytes}, allow_cast, allow_partial_pixels);
			if (!rectangle) return;
			guest_surface_byte_cursor cursor(request);
			while (const auto active = cursor.next())
				for (std::size_t position = active->address, remaining = active->size; remaining;)
				{
					if (++comparisons > (1u << 20)) throw std::length_error("Native segmented texture proof exceeds run budget");
					if (position > source_size || anchor > source_size - position) return;
					const auto* src = segment_at(source_segments, anchor + position);
					const auto* dst = segment_at(requested, position);
					if (!src || !dst) return;
					const auto src_offset = anchor + position - src->offset, dst_offset = position - dst->offset;
					const auto count = std::min({remaining, src->size - src_offset, dst->size - dst_offset});
					if (!equal({static_cast<std::uint32_t>(std::uint64_t(src->address) + src_offset), count},
						{static_cast<std::uint32_t>(std::uint64_t(dst->address) + dst_offset), count})) return;
					position += count; remaining -= count;
				}
			if (result && (result->pixels.x != rectangle->pixels.x || result->pixels.y != rectangle->pixels.y ||
				result->pixels.width != rectangle->pixels.width || result->pixels.height != rectangle->pixels.height ||
				result->guest_row_offset != rectangle->guest_row_offset)) ambiguous = true;
			result = rectangle;
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
[[nodiscard]] inline std::optional<clear_rectangle> plan_segmented_texture_region(
	guest_surface_memory& memory, const guest_surface_layout& source, surface_format format,
	std::span<const guest_storage_segment> source_segments, std::size_t source_origin,
	const guest_texture_layout& texture, std::span<const guest_storage_segment> requested,
	std::optional<std::size_t> logical_anchor = {}, bool allow_cast = false)
{
	const auto mapping = plan_segmented_texture_mapping(memory, source, format, source_segments, source_origin,
		texture, requested, logical_anchor, allow_cast);
	return mapping ? std::optional{mapping->pixels} : std::nullopt;
}
} // namespace metal4
