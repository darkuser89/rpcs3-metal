#pragma once

#include "M4PhysicalTextureRegion.h"
#include "M4SurfaceCache.h"
#include "M4DisplayStorage.h"

namespace metal4
{
[[nodiscard]] inline std::optional<std::size_t> plan_cached_morton_display_storage(guest_surface_memory& memory,
	const cached_surface_description& source, const guest_surface_layout& display,
	std::span<const guest_storage_segment> requested, std::uint32_t offset, bool allow_inactive = false)
{
	if (!source.morton || source.tile || !std::holds_alternative<surface_color>(source.allocation.format)) return {};
	const auto size = source.allocation.samples == 1 ? std::array{source.allocation.width, source.allocation.height} :
		sample_expanded_size(source.allocation.width, source.allocation.height, source.allocation.samples);
	const auto geometry = plan_guest_surface(0, size[0], size[1], color_layout(std::get<surface_color>(source.allocation.format)).pixel_bytes, 0, source.morton);
	std::optional<std::size_t> anchor;
	if (source.mapping)
	{
		const auto& mapping = source.mapping->state();
		if (!mapping.location && offset >= mapping.logical.address && offset - mapping.logical.address < mapping.logical.size)
			anchor = std::size_t(offset - mapping.logical.address);
	}
	auto segments = source.segments;
	if (segments.empty()) segments.push_back({source.address, 0, geometry.range.size});
	return plan_morton_display_storage(memory, geometry, segments, display, requested, anchor, allow_inactive);
}
// Caller holds the surface cache CPU epoch and validates its mapping ticket.
// Banked RTT lookup uses original logical coordinates;raw RAM uploads keep
// their existing context policy and are never automatically detiled here.
// MSAA rectangles address individual samples in the guest-expanded 2x1/2x2
// raster;the cache snapshots that raster without averaging or VM re-encoding.
[[nodiscard]] inline std::optional<surface_texture_region> plan_cached_texture_mapping(guest_surface_memory& memory,
	const cached_surface_description& source, const guest_texture_layout& texture,
	std::span<const guest_storage_segment> requested, std::optional<surface_format> format = {}, bool allow_cast = false, bool allow_partial_pixels = false)
{
	const auto size = source.allocation.samples == 1 ? std::array{source.allocation.width, source.allocation.height} :
		sample_expanded_size(source.allocation.width, source.allocation.height, source.allocation.samples);
	const auto bytes = std::holds_alternative<surface_color>(source.allocation.format) ? color_layout(std::get<surface_color>(source.allocation.format)).pixel_bytes :
		describe_surface_depth(std::get<surface_depth>(source.allocation.format)).guest_bytes;
	const auto geometry = plan_guest_surface(0, size[0], size[1], bytes, source.row_pitch, source.morton);
	std::optional<std::size_t> anchor;
	const auto origin = source.tile ? source.tile->offset : 0;
	if (source.mapping)
	{
		const auto& mapping = source.mapping->state();
		if (mapping.location == texture.location && texture.offset >= mapping.logical.address &&
			texture.offset - mapping.logical.address < mapping.logical.size)
			anchor = std::size_t(texture.offset - mapping.logical.address);
	}
	if (source.tile && !anchor) return {};
	std::vector<guest_storage_segment> storage = source.segments;
	if (storage.empty()) storage.push_back({source.tile ? source.tile->address : source.address, 0,
		source.tile ? source.tile->size : geometry.range.size});
	return plan_segmented_texture_mapping(memory, geometry, format.value_or(source.allocation.format), storage, origin, texture, requested, anchor, allow_cast, allow_partial_pixels);
}
[[nodiscard]] inline std::optional<clear_rectangle> plan_cached_texture_region(guest_surface_memory& memory,
	const cached_surface_description& source, const guest_texture_layout& texture,
	std::span<const guest_storage_segment> requested, std::optional<surface_format> format = {}, bool allow_cast = false)
{
	const auto mapping = plan_cached_texture_mapping(memory, source, texture, requested, format, allow_cast);
	return mapping ? std::optional{mapping->pixels} : std::nullopt;
}
} // namespace metal4
