#pragma once

#include "M4GuestSurface.h"
#include "M4SampleLayout.h"
#include "M4SurfaceLayout.h"
#include "M4SurfaceMapping.h"

namespace metal4
{
inline std::size_t framebuffer_depth_bytes(surface_depth depth)
{
	switch (depth)
	{
	case surface_depth::z16_uint: case surface_depth::z16_float: return 2;
	case surface_depth::z24s8_uint: case surface_depth::z24s8_float: return 4;
	default: throw std::invalid_argument("Native framebuffer depth format is invalid");
	}
}
// VK cs_deswizzle_3d uses ceil_log2 of the sample-expanded raster. Visible
// dimensions need not be powers of two; inactive cells remain guest padding.
// Explicit larger storage supplied by a caller still takes precedence.
inline std::array<std::size_t, 3> framebuffer_morton_storage(std::array<std::size_t, 3> size)
{
	for (auto& dimension : size)
	{
		if (!dimension || dimension > (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1)))
			throw std::overflow_error("Native framebuffer Morton dimension cannot round to a power of two");
		dimension = std::bit_ceil(dimension);
	}
	return size;
}
struct framebuffer_surface
{
	std::uint32_t address = 0, physical_slot = 0;
	surface_format format = surface_color::argb8;
	std::size_t width = 0, height = 0, row_pitch = 0;
	std::uint32_t samples = 1;
	std::optional<swizzle_region> morton;
	guest_surface_layout guest;
	std::shared_ptr<const surface_mapping_revision> mapping = {};
};
class framebuffer_snapshot final
{
	std::array<std::optional<framebuffer_surface>, 4> m_colors;
	std::optional<framebuffer_surface> m_depth;
	surface_color m_color_format = surface_color::argb8;
	std::uint32_t m_sample_mode = 0;

  public:
	// Input is validated rsx::framebuffer_layout AFTER address/DMA/context
	// analysis and pitch fixups. An optional CPU-only mapping callback attaches
	// owned IO/tile state under caller-provided writer exclusion. No writeback
	// happens here. Explicit presence preserves analysed EA wrap-to-zero;the
	// layout's zero sentinel otherwise remains authoritative. Swizzle storage
	// defaults to the rounded sample raster; callers can supply larger complete
	// sample-expanded dimensions for cropped storage. Visible pixels remain size.
	template <typename Layout>
	[[nodiscard]] static framebuffer_snapshot capture(const Layout& layout, bool layout_valid,
													  std::optional<std::array<std::size_t, 3>> swizzle_storage = {}, std::size_t byte_limit = 512u << 20,
		const std::function<void(framebuffer_surface&)>& map = {}, std::uint32_t zero_address_mask = 0)
	{
		if ((zero_address_mask & ~31u) || (zero_address_mask && !map))
			throw std::invalid_argument("Zero-EA framebuffer origins need explicit presence and a mapping callback");
		if (!layout_valid || !layout.width || !layout.height)
			throw std::invalid_argument("Native framebuffer capture requires validated nonempty layout");
		framebuffer_snapshot result;
		result.m_color_format = static_cast<surface_color>(layout.color_format);
		const auto color_bytes = color_layout(result.m_color_format).pixel_bytes;
		const auto target = static_cast<std::uint32_t>(layout.target);
		std::uint32_t physical_mask;
		switch (target)
		{
		case 0:
			physical_mask = 0;
			break;
		case 1:
			physical_mask = 1;
			break;
		case 2:
			physical_mask = 2;
			break;
		case 0x13:
			physical_mask = 3;
			break;
		case 0x17:
			physical_mask = 7;
			break;
		case 0x1f:
			physical_mask = 15;
			break;
		default:
			throw std::invalid_argument("Native framebuffer target enum is invalid");
		}
		result.m_sample_mode = static_cast<std::uint32_t>(layout.aa_mode);
		std::uint32_t samples = 1, x = 1, y = 1;
		switch (result.m_sample_mode)
		{
		case 0:
			break;
		case 3:
			samples = 2;
			x = 2;
			break;
		case 4:
		case 5:
			samples = 4;
			x = y = 2;
			break;
		default:
			throw std::invalid_argument("Native framebuffer sample mode is invalid");
		}
		if (layout.aa_factors[0] != x || layout.aa_factors[1] != y)
			throw std::invalid_argument("Native framebuffer sample factors disagree with analysed layout");
		const auto raster = static_cast<std::uint32_t>(layout.raster_type);
		if (raster != 1 && raster != 2)
			throw std::invalid_argument("Native framebuffer raster type is invalid");
		if (raster == 1 && swizzle_storage)
			throw std::invalid_argument("Linear framebuffer cannot carry Morton storage");
		std::size_t bytes = 0;
		const auto surface = [&](std::uint32_t address, surface_format format, std::uint32_t slot, std::size_t pixel_bytes, std::size_t pitch)
		{
			framebuffer_surface value;
			value.address = address;
			value.format = format;
			value.physical_slot = slot;
			value.width = layout.width;
			value.height = layout.height;
			value.samples = samples;
			const std::array<std::size_t, 3> size{checked_texture_product(value.width, x), checked_texture_product(value.height, y), 1};
			if (raster == 2)
			{
				const auto storage = swizzle_storage ? *swizzle_storage : framebuffer_morton_storage(size);
				value.morton = swizzle_region{storage, {}, size};
			}
			else
			{
				if (!pitch || pitch < checked_texture_product(size[0], pixel_bytes))
					throw std::invalid_argument("Native analysed framebuffer pitch is too short");
				value.row_pitch = pitch;
			}
			value.guest = plan_guest_surface(map ? 0 : address, size[0], size[1], pixel_bytes, value.row_pitch, value.morton);
			if (map) map(value);
			if (value.guest.range.size > byte_limit - bytes)
				throw std::length_error("Native framebuffer guest storage exceeds capture budget");
			bytes += value.guest.range.size;
			return value;
		};
		unsigned logical = 0;
		for (unsigned slot = 0; slot < 4; ++slot)
			if ((physical_mask & (1u << slot)) && (layout.color_addresses[slot] || (zero_address_mask & (1u << slot))))
				result.m_colors[logical++] = surface(layout.color_addresses[slot], result.m_color_format, slot, color_bytes, layout.actual_color_pitch[slot]);
		if (layout.zeta_address || (zero_address_mask & 16))
		{
			const auto depth = static_cast<surface_depth>(layout.depth_format);
			result.m_depth = surface(layout.zeta_address, depth, 4, framebuffer_depth_bytes(depth), layout.actual_zeta_pitch);
		}
		if (!logical && !result.m_depth)
			throw std::invalid_argument("Native framebuffer has no active surfaces");
		return result;
	}
	[[nodiscard]] const auto& colors() const { return m_colors; }
	[[nodiscard]] const auto& depth() const { return m_depth; }
	[[nodiscard]] surface_color color_format() const { return m_color_format; }
	[[nodiscard]] std::uint32_t sample_mode() const { return m_sample_mode; }
};
inline void map_framebuffer_surface(framebuffer_surface& surface, std::uint32_t offset, unsigned location,
	const std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)>& translate,
	const std::shared_ptr<io_mapping_epoch>& io, const std::shared_ptr<tile_state>& tiles, bool tiling, std::size_t limit)
{
	// GCM_context::get_tiled_memory_region exposes local VRAM transparently;
	// local tile tags do not change framebuffer byte storage. Display receipts
	// still observe those epochs independently before exposing an AV image.
	auto mapping = capture_surface_mapping(surface.guest, offset, location, translate, io, tiles, tiling, limit, false);
	const auto& state = mapping->state();
	if (state.pixel_address != surface.address) throw surface_mapping_conflict();
	surface.guest.range = {state.segments.front().address, state.logical.size};
	surface.mapping = std::move(mapping);
}
// The common layout applies window bytes AFTER translating the original
// surface address. That EA arithmetic is not a logical IO translation across
// disjoint entries. Recompute from captured raw offset,window and actual pitch;
// never infer logical displacement from the analysed physical EA.
inline std::uint32_t framebuffer_window_origin(std::uint32_t raw, std::uint32_t window_x, std::uint32_t window_y,
	std::size_t actual_pitch, std::size_t pixel_bytes, std::uint32_t samples)
{
	if (!actual_pitch || !pixel_bytes || (samples != 1 && samples != 2 && samples != 4))
		throw std::invalid_argument("Native framebuffer window geometry is invalid");
	const auto horizontal = checked_texture_product(checked_texture_product(pixel_bytes, samples == 1 ? 1 : 2), window_x);
	const auto vertical = checked_texture_product(actual_pitch, window_y);
	const auto remaining = std::uint64_t(0xffff'ffffu) - raw;
	if (vertical > remaining || horizontal > remaining - vertical)
		throw std::overflow_error("Native framebuffer logical window origin wraps RSX offsets");
	return static_cast<std::uint32_t>(raw + vertical + horizontal);
}
inline void map_framebuffer_window_surface(framebuffer_surface& surface, std::uint32_t raw, unsigned location,
	std::uint32_t window_x, std::uint32_t window_y, std::size_t actual_pitch,
	const std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)>& translate,
	const std::shared_ptr<io_mapping_epoch>& io, const std::shared_ptr<tile_state>& tiles, bool tiling, std::size_t limit)
{
	const auto origin = framebuffer_window_origin(raw, window_x, window_y, actual_pitch, surface.guest.pixel_bytes, surface.samples);
	if (!translate) throw std::invalid_argument("Native framebuffer window needs IO translation");
	surface.address = translate(origin, location, 1);
	map_framebuffer_surface(surface, origin, location, translate, io, tiles, tiling, limit);
}
// Common analysis assigns actual color pitch only after accepting that slot.
// Thus it proves color presence even when its later additive EA wraps to zero.
// Depth pitch alone is insufficient;the separate helper below accounts for
// every depth-zero path in the common analysed-layout contract.
template <typename Layout>
inline std::uint32_t framebuffer_window_zero_colors(const Layout& layout)
{
	std::uint32_t mask = 0;
	for (unsigned slot = 0; slot < 4; ++slot)
		if (!layout.color_addresses[slot] && layout.actual_color_pitch[slot]) mask |= 1u << slot;
	return mask;
}
// Recover depth presence only for an additive EA wrap-to-zero. Common
// analysis can also clear depth for unused access (no actual pitch),invalid
// small linear pitch,or a surviving color alias. Keep those zero sentinels.
// raw_depth_pitch precedes fixups;using corrected pitch would drop valid
// depth after another attachment shrinks the common framebuffer width.
template <typename Layout>
inline std::uint32_t framebuffer_window_zero_depth(const Layout& layout, std::uint32_t original_depth_address,
	std::uint32_t window_x, std::uint32_t window_y, std::uint32_t raw_depth_pitch, std::size_t minimum_depth_pitch)
{
	if (layout.zeta_address || !layout.actual_zeta_pitch || !original_depth_address) return 0;
	const auto depth = framebuffer_depth_bytes(static_cast<surface_depth>(layout.depth_format));
	const auto depth_delta = std::uint64_t(layout.actual_zeta_pitch) * window_y + std::uint64_t(depth) * layout.aa_factors[0] * window_x;
	if (static_cast<std::uint32_t>(original_depth_address + depth_delta)) return 0;
	if (static_cast<unsigned>(layout.raster_type) == 1 && raw_depth_pitch <= 64 && raw_depth_pitch < minimum_depth_pitch) return 0;
	const auto color = color_layout(static_cast<surface_color>(layout.color_format)).pixel_bytes;
	for (unsigned slot = 0; slot < 4; ++slot)
		if (layout.actual_color_pitch[slot])
		{
			const auto delta = std::uint64_t(layout.actual_color_pitch[slot]) * window_y + std::uint64_t(color) * layout.aa_factors[0] * window_x;
			if (static_cast<std::uint32_t>(layout.color_addresses[slot] - delta) == original_depth_address) return 0;
		}
	return 16;
}
template <typename Layout>
inline void apply_framebuffer_origins(Layout& layout, const framebuffer_snapshot& snapshot)
{
	for (const auto& surface : snapshot.colors())
		if (surface) layout.color_addresses[surface->physical_slot] = surface->address;
	if (snapshot.depth()) layout.zeta_address = snapshot.depth()->address;
}
} // namespace metal4
