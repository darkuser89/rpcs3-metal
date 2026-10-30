#pragma once

#include "Emu/RSX/Utils/rsx_utils.h"
#include "M4ScaledImageMapping.h"
#include "M4ScaledImagePlan.h"

#include <bit>
#include <cmath>
#include <limits>

namespace metal4
{
// Probe the complete logical/physical allocation through its original IO
// offset. Context and translation are also instantiated against the real RSX
// interfaces;tests supply owned metadata,never a live VM or renderer.
template <typename Lookup, typename Translate>
scaled_image_memory_mapping probe_scaled_image_mapping(const Lookup& lookup, const Translate& translate,
													   std::uint32_t io_offset, unsigned dma, std::uint32_t image_delta, std::size_t image_bytes, bool track_tiles = false)
{
	if (image_delta > 0xffff'ffffull - io_offset)
		throw std::overflow_error("Native scaled-image IO origin wraps");
	const auto logical_io = io_offset + image_delta;
	scaled_image_memory_mapping result{translate(logical_io, dma, 1), image_bytes, {}};
	validate_guest_surface_range({result.address, result.size});
	if (!image_bytes || image_bytes > std::numeric_limits<std::uint32_t>::max())
		throw std::invalid_argument("Native scaled-image mapping extent must fit RSX interface");
	result.tile = lookup(logical_io, dma, result.address, image_bytes);
	const auto mapped_io = result.tile ? result.tile->io_offset : logical_io;
	const auto mapped_size = result.tile ? result.tile->size : image_bytes;
	const auto mapped_address = result.tile ? result.tile->address : result.address;
	if (!mapped_size || mapped_size > 0x1'0000'0000ull - mapped_io)
		throw std::overflow_error("Native scaled-image IO extent wraps");
	if (dma == CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER || dma == CELL_GCM_LOCATION_MAIN)
	{
		result.io = scaled_image_memory_mapping::io_range{mapped_io, mapped_size};
		if (track_tiles) result.tiling = scaled_image_memory_mapping::tile_range{mapped_io, mapped_size, CELL_GCM_LOCATION_MAIN};
	}
	else if (dma == CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_MAIN)
	{
		// get_address places main report offsets in the IO report aperture.
		if (mapped_io >= 0x1000000 || mapped_size > 0x1000000 - mapped_io)
			throw std::out_of_range("Native main report mapping exceeds its IO aperture");
		result.io = scaled_image_memory_mapping::io_range{0x0e000000 + mapped_io, mapped_size};
	}
	// Probe each1MiB IO entry separately. The shared get_address full-span
	// check assumes one linear EA;disjoint entries require owned byte gather.
	for (std::size_t delta = 0; delta < mapped_size;)
	{
		const auto io = static_cast<std::uint32_t>(mapped_io + delta);
		const auto count = std::min<std::size_t>(mapped_size - delta, 0x100000 - (io & 0xfffff));
		const auto address = translate(io, dma, count);
		validate_guest_surface_range({address, count});
		if (!delta && address != mapped_address)
			throw scaled_image_mapping_conflict();
		if (!result.segments.empty() && std::uint64_t(result.segments.back().address) + result.segments.back().size == address)
			result.segments.back().size += count;
		else
			result.segments.push_back({address, delta, count});
		delta += count;
	}
	return result;
}

// Freeze decoded values only:never dereference src/dst.pixels. The shared
// decoder has already normalized fractional origin offsets to integer pixels.
inline scaled_image_plan capture_scaled_image(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst,
											  std::uint32_t output_width, std::uint32_t output_height, bool interpolate, std::size_t limit)
{
	const auto source_format = [&]
	{
		switch (src.format)
		{
		case rsx::blit_engine::transfer_source_format::r5g6b5:
			return scaled_image_format::rgb565;
		case rsx::blit_engine::transfer_source_format::a8r8g8b8:
		case rsx::blit_engine::transfer_source_format::x8r8g8b8:
			return scaled_image_format::argb8;
		default:
			throw std::invalid_argument("Unsupported decoded native scaled-image source format");
		}
	}();
	const auto target_format = [&]
	{
		switch (dst.format)
		{
		case rsx::blit_engine::transfer_destination_format::r5g6b5:
			return scaled_image_format::rgb565;
		case rsx::blit_engine::transfer_destination_format::a8r8g8b8:
			return scaled_image_format::argb8;
		default:
			throw std::invalid_argument("Unsupported decoded native scaled-image destination format");
		}
	}();
	if (!std::isfinite(dst.scale_x) || !std::isfinite(dst.scale_y) || dst.scale_x == 0 || dst.scale_y == 0 ||
		!output_width || !output_height || output_width > 65536 || output_height > 65536 ||
		src.bpp != scaled_image_pixel_bytes(source_format) || dst.bpp != scaled_image_pixel_bytes(target_format) ||
		(src.origin != rsx::blit_engine::transfer_origin::center && src.origin != rsx::blit_engine::transfer_origin::corner))
		throw std::invalid_argument("Invalid decoded native scaled-image scale,origin or byte count");
	const auto scaled_width = std::floor(std::abs(dst.scale_x) * src.width), scaled_height = std::floor(std::abs(dst.scale_y) * src.height);
	if (scaled_width != dst.width || scaled_height != dst.height)
		throw std::invalid_argument("Decoded native scaled-image output extent disagrees with scale");
	const auto normalize = [](std::uint32_t offset, std::uint32_t extent, bool flip)
	{
		if (flip && offset)
		{
			if (offset > extent)
				throw std::invalid_argument("Decoded reverse scaled-image offset exceeds source");
			offset = extent - offset;
			if (offset == 1)
				offset = 0;
		}
		if (offset >= extent)
			throw std::invalid_argument("Decoded scaled-image offset exceeds source");
		return offset;
	};
	const auto x = normalize(src.offset_x, src.width, dst.scale_x < 0), y = normalize(src.offset_y, src.height, dst.scale_y < 0);
	const auto offset_address = [](std::uint32_t base, std::uint64_t offset)
	{
		if (offset > 0xffff'ffffull - base)
			throw std::overflow_error("Native scaled-image guest offset wraps");
		return static_cast<std::uint32_t>(base + offset);
	};
	const auto source_address = offset_address(src.rsx_address, std::uint64_t(y) * src.pitch + std::uint64_t(x) * src.bpp);
	auto source = plan_guest_surface(source_address, src.width - x, src.height - y, src.bpp, src.pitch);
	guest_surface_layout target;
	if (dst.swizzled)
	{
		const swizzle_region morton{{std::bit_ceil(output_width), std::bit_ceil(output_height), 1},
									{dst.offset_x, dst.offset_y, 0},
									{dst.clip_width, dst.clip_height, 1}};
		target = plan_guest_surface(dst.rsx_address, dst.clip_width, dst.clip_height, dst.bpp, 0, morton);
	}
	else
	{
		const auto address = offset_address(dst.rsx_address, std::uint64_t(dst.offset_y) * dst.pitch + std::uint64_t(dst.offset_x) * dst.bpp);
		target = plan_guest_surface(address, dst.clip_width, dst.clip_height, dst.bpp, dst.pitch);
	}
	scaled_image_plan result{std::move(source), std::move(target), source_format, target_format, src.width, src.height, dst.width, dst.height,
							 dst.clip_x, dst.clip_y, dst.scale_x < 0, dst.scale_y < 0, interpolate};
	validate_scaled_image(result, limit);
	return result;
}
} // namespace metal4
