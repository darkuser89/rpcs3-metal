#pragma once

#include "M4GuestCoverage.h"
#include <bit>

namespace metal4
{
// Main-memory RSX banked allocation. Local VRAM tile access is already
// transparent in the emulator. Capture the complete allocation because bank
// permutation can address outside a nominal rectangular byte span.
struct banked_tile_layout
{
	std::uint32_t address, size, pitch, bank, offset;
	bool operator==(const banked_tile_layout&) const = default;
};
inline std::uint32_t banked_tile_factor(std::uint32_t pitch)
{
	if (!pitch || (pitch & 255))
		throw std::invalid_argument("Banked tile pitch must be a positive multiple of 256");
	const auto columns = pitch / 256;
	if (std::has_single_bit(columns))
		return columns;
	for (const unsigned prime : {3, 5, 7, 11, 13})
		if (columns % prime == 0 && std::has_single_bit(columns / prime))
			return columns / prime;
	throw std::invalid_argument("Unsupported banked tile pitch factorization");
}
// Geometry of an owned flat tile buffer. Its base participates in the RSX
// permutation, but a disjoint allocation is not one contiguous guest EA span.
inline void validate_banked_tile_geometry(const guest_surface_layout& image, const banked_tile_layout& tile)
{
	(void)banked_tile_factor(tile.pitch);
	if ((tile.address & 65535) || !tile.size || (tile.size & 65535) || tile.bank > 3 || tile.offset >= tile.size ||
		image.range.address != tile.address || image.range.size != tile.size || !image.width || !image.height || image.width > 65536 || image.height > 65536 ||
		(!std::has_single_bit(image.pixel_bytes) || image.pixel_bytes > 16) || tile.offset % image.pixel_bytes)
		throw std::invalid_argument("Invalid banked tile allocation,bank or image origin");
	(void)guest_surface_byte_cursor(plan_guest_surface(0, image.width, image.height, image.pixel_bytes, image.morton ? 0 : tile.pitch, image.morton));
	if (image.morton)
	{
		if (plan_swizzle(*image.morton, image.pixel_bytes).storage_bytes > tile.size - tile.offset)
			throw std::invalid_argument("Banked Morton image exceeds allocation");
	}
	else if (image.width > (tile.pitch - tile.offset % tile.pitch) / image.pixel_bytes ||
			 image.width * image.pixel_bytes > tile.size - tile.offset ||
			 image.height - 1 > (tile.size - tile.offset - image.width * image.pixel_bytes) / tile.pitch)
		throw std::invalid_argument("Banked image extends beyond a tile row or allocation");
}
inline void validate_banked_tile(const guest_surface_layout& image, const banked_tile_layout& tile)
{
	validate_guest_surface_range(image.range);
	validate_banked_tile_geometry(image, tile);
}
// Attach only to a previously validated logical image. Its shape remains the
// GPU image view;the range changes to the entire physical allocation. CPU
// revision/CAS coverage for this view is deliberately conservative,all bytes.
inline guest_surface_layout attach_banked_tile(guest_surface_layout image, const banked_tile_layout& tile)
{
	(void)guest_surface_byte_cursor(image);
	if (image.range.address < tile.address || std::uint64_t(image.range.address) - tile.address != tile.offset)
		throw std::invalid_argument("Banked image origin disagrees with translated tile address");
	image.range = {tile.address, tile.size};
	validate_banked_tile(image, tile);
	return image;
}
} // namespace metal4
