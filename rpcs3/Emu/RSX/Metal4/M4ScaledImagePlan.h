#pragma once

#include "M4BankedTile.h"
#include "M4GuestSegments.h"

namespace metal4
{
enum class scaled_image_format : std::uint32_t
{
	rgb565,
	argb8
};
struct scaled_image_plan
{
	guest_surface_layout source, destination;
	scaled_image_format source_format, destination_format;
	// Full source/output affine domain before clipping. Captured source may
	// be a remaining subregion;reads clamp to its active pixel bounds.
	std::uint32_t input_width, input_height, scaled_width, scaled_height;
	std::uint32_t clip_x = 0, clip_y = 0;
	bool flip_x = false, flip_y = false, interpolate = false;
	std::optional<banked_tile_layout> source_tile = {}, destination_tile = {};
	std::vector<guest_storage_segment> source_segments = {}, destination_segments = {};
};
inline std::size_t scaled_image_pixel_bytes(scaled_image_format format)
{
	switch (format)
	{
	case scaled_image_format::rgb565:
		return 2;
	case scaled_image_format::argb8:
		return 4;
	}
	throw std::invalid_argument("Invalid native scaled-image format");
}
inline guest_surface_layout scaled_image_coverage(const guest_surface_layout& image, const std::optional<banked_tile_layout>& tile)
{
	return tile ? guest_byte_coverage(image.range) : image;
}
inline void validate_scaled_image(const scaled_image_plan& plan, std::size_t byte_limit)
{
	validate_guest_segments(plan.source, plan.source_segments);
	validate_guest_segments(plan.destination, plan.destination_segments);
	if (plan.source_tile)
		validate_banked_tile(plan.source, *plan.source_tile);
	else
		(void)guest_surface_byte_cursor(plan.source);
	if (plan.destination_tile)
		validate_banked_tile(plan.destination, *plan.destination_tile);
	else
		(void)guest_surface_byte_cursor(plan.destination);
	for (const auto* layout : {&plan.source, &plan.destination})
		if (layout->morton && (layout->morton->storage[2] != 1 || layout->morton->origin[2] ||
							   layout->morton->storage[0] > 0xffff'ffffull || layout->morton->storage[1] > 0xffff'ffffull))
			throw std::invalid_argument("Native scaled-image Morton storage must fit two-dimensional coordinates");
	if (!plan.input_width || !plan.input_height || !plan.scaled_width || !plan.scaled_height ||
		!plan.source.width || !plan.source.height || !plan.destination.width || !plan.destination.height ||
		plan.input_width > 65536 || plan.input_height > 65536 || plan.scaled_width > 65536 || plan.scaled_height > 65536 ||
		plan.source.width > plan.input_width || plan.source.height > plan.input_height ||
		plan.clip_x > plan.scaled_width || plan.destination.width > plan.scaled_width - plan.clip_x ||
		plan.clip_y > plan.scaled_height || plan.destination.height > plan.scaled_height - plan.clip_y ||
		plan.source.pixel_bytes != scaled_image_pixel_bytes(plan.source_format) ||
		plan.destination.pixel_bytes != scaled_image_pixel_bytes(plan.destination_format))
		throw std::invalid_argument("Invalid native scaled-image dimensions,clip or pixel layout");
	if (plan.source.range.size > byte_limit || plan.destination.range.size > byte_limit - plan.source.range.size)
		throw std::length_error("Native scaled-image source and destination exceed aggregate snapshot budget");
}
} // namespace metal4
