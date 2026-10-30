#pragma once

#include "M4ClearPlan.h"
#include "M4GuestTextureLayout.h"

namespace metal4
{
// Native channel interpretation for storage-compatible RTT aliases. Equal
// native formats retain the source's X-channel semantics,as VK temporary
// views do. RGBA8 -> ARGB8 is a data cast:swap R/B and retain the raw alpha.
// Same-format source maps also override a target's ordinary upload map;
// format casts use the target map,including its explicit constant alpha.
[[nodiscard]] inline std::optional<std::array<std::uint32_t, 4>> surface_texture_channels(
	surface_format format, const guest_texture_layout& texture)
{
	auto channels = texture.codec.base_argb;
	if (texture.codec.depth)
		return format == surface_format{*texture.codec.depth} ? std::optional{channels} : std::nullopt;
	const auto color = std::get_if<surface_color>(&format);
	if (!color) return {};
	const auto x_alpha = [&](bool zero)
	{
		channels[0] = zero ? 4 : 5;
	};
	switch (texture.format)
	{
	case 0x81: if (*color != surface_color::b8) return {}; break;
	case 0x82: case 0x9d:
		if (*color != surface_color::x1rgb5_zero && *color != surface_color::x1rgb5_one) return {};
		x_alpha(*color == surface_color::x1rgb5_zero); break;
	case 0x84: if (*color != surface_color::rgb565) return {}; break;
	case 0x85: case 0x9e:
		switch (*color)
		{
		case surface_color::argb8: channels[0] = 3; break;
		case surface_color::xrgb8_zero: case surface_color::xrgb8_one:
			x_alpha(*color == surface_color::xrgb8_zero); break;
		case surface_color::abgr8: case surface_color::xbgr8_zero: case surface_color::xbgr8_one:
			for (auto& channel : channels)
				if (channel == 0 || channel == 2) channel ^= 2;
			break;
		default: return {};
		}
		break;
	case 0x8b: case 0x98:
		if (*color != surface_color::gb8) return {};
		channels = {1, 0, 1, 0}; break;
	case 0x9a: if (*color != surface_color::rgba16f) return {}; break;
	case 0x9b: if (*color != surface_color::rgba32f) return {}; break;
	case 0x9c: if (*color != surface_color::r32f) return {}; break;
	default: return {};
	}
	return channels;
}
// An immutable source rectangle plus the guest-byte prefix to discard from
// each packed row. The prefix belongs to the selected source mapping.
struct surface_texture_region
{
	clear_rectangle pixels;
	std::size_t guest_row_offset = 0;
};
// Map active guest bytes to native source pixels without reading VM.
// Linear block targets address guest block columns/rows,including complete
// edge blocks for non-multiple dimensions. Optional partial linear casts
// include the containing edge source pixels and retain a byte
// prefix for GPU pack/crop/decode. Morton subimages still require equal texel
// widths and exact axis address increments,including carry boundaries.
[[nodiscard]] inline std::optional<surface_texture_region> plan_surface_texture_mapping(const guest_surface_layout& source,
	surface_format format, const guest_texture_layout& texture, guest_surface_range range, bool allow_cast = false, bool allow_partial_pixels = false)
{
	if (texture.dimension != vertex_texture_dimension::two || texture.mips != 1 || texture.layers != 1 || texture.depth != 1 ||
		texture.stored_border || texture.adjusted_pitch || !texture.codec.block.width || !texture.codec.block.height ||
		((texture.codec.block.width != 1 || texture.codec.block.height != 1) && (!allow_cast || texture.swizzled)) ||
		range.size != texture.storage_bytes || bool(source.morton) != texture.swizzled || (allow_partial_pixels && !allow_cast) ||
		(!allow_cast && source.pixel_bytes != texture.codec.block.bytes) ||
		range.address < source.range.address || !source.pixel_bytes || texture.subresources.empty() || (!allow_cast && !surface_texture_channels(format, texture)))
		return {};
	// Linear casts preserve active guest bytes per row,not pixel count. A
	// Morton cast may reuse this rectangle only at equal guest texel widths:
	// changing bytes per pixel also changes the interleaved X/Y address bits.
	const auto columns = texture.width / texture.codec.block.width + bool(texture.width % texture.codec.block.width);
	const auto rows = texture.height / texture.codec.block.height + bool(texture.height % texture.codec.block.height);
	const auto row_bytes = checked_texture_product(columns, texture.codec.block.bytes);
	if (!texture.codec.block.bytes || (!allow_partial_pixels && row_bytes % source.pixel_bytes) ||
		(source.morton && source.pixel_bytes != texture.codec.block.bytes)) return {};
	auto width = row_bytes / source.pixel_bytes;
	const auto delta = std::size_t(range.address - source.range.address);
	if (delta >= source.range.size)
		return {};
	std::size_t x, y, prefix = 0;
	if (!source.morton)
	{
		if (!source.row_pitch || source.row_pitch != texture.subresources[0].row_pitch)
			return {};
		const auto byte_x = delta % source.row_pitch;
		const auto active_bytes = checked_texture_product(source.width, source.pixel_bytes);
		if (byte_x > active_bytes || row_bytes > active_bytes - byte_x) return {};
		x = byte_x / source.pixel_bytes;
		y = delta / source.row_pitch;
		prefix = byte_x % source.pixel_bytes;
		if (allow_partial_pixels)
		{
			const auto span = checked_texture_sum(prefix, row_bytes);
			width = span / source.pixel_bytes + bool(span % source.pixel_bytes);
		}
		else if (prefix) return {};
	}
	else
	{
		const auto& storage = *source.morton;
		if (delta % source.pixel_bytes || !std::has_single_bit(texture.width) || !std::has_single_bit(texture.height))
			return {};
		const auto native = plan_swizzle(storage, source.pixel_bytes);
		const auto local = plan_swizzle({{texture.width, texture.height, 1}, {}, {texture.width, texture.height, 1}}, source.pixel_bytes);
		std::array<std::size_t, 3> origin{};
		std::size_t packed_bit = 0;
		for (unsigned bit = 0; bit < std::max({native.log2[0], native.log2[1], native.log2[2]}); ++bit)
			for (unsigned axis = 0; axis < 3; ++axis)
				if (bit < native.log2[axis])
					origin[axis] |= (((delta / source.pixel_bytes) >> packed_bit++) & 1) << bit;
		if (origin[2] != storage.origin[2] || origin[0] < storage.origin[0] || origin[1] < storage.origin[1])
			return {};
		const auto base = morton_index(origin, native.log2);
		// Morton is the sum of disjoint axis bit weights. Comparing each
		// one-dimensional sequence proves the complete rectangle without
		// scanning width*height pixels. Carries can permit unaligned regions
		// when the remaining high storage bits all belong to that axis.
		for (unsigned axis = 0; axis < 2; ++axis)
		{
			const auto extent = axis ? texture.height : texture.width;
			if (extent > storage.storage[axis] - origin[axis])
				return {};
			for (std::size_t pixel = 1; pixel < extent; ++pixel)
			{
				auto position = origin;
				position[axis] += pixel;
				std::array<std::size_t, 3> unit{};
				unit[axis] = pixel;
				if (morton_index(position, native.log2) - base != morton_index(unit, local.log2))
					return {};
			}
		}
		x = origin[0] - storage.origin[0];
		y = origin[1] - storage.origin[1];
	}
	if (x > source.width || width > source.width - x || y > source.height || rows > source.height - y)
		return {};
	return surface_texture_region{{x, y, width, rows}, prefix};
}
// Existing pixel-aligned callers deliberately retain their stricter contract.
[[nodiscard]] inline std::optional<clear_rectangle> plan_surface_texture_region(const guest_surface_layout& source,
	surface_format format, const guest_texture_layout& texture, guest_surface_range range, bool allow_cast = false)
{
	const auto mapping = plan_surface_texture_mapping(source, format, texture, range, allow_cast);
	return mapping ? std::optional{mapping->pixels} : std::nullopt;
}
} // namespace metal4
