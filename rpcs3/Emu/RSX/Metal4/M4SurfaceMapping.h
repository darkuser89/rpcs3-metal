#pragma once

#include "M4BankedTile.h"
#include "M4TextureMapping.h"
#include "M4TileState.h"

namespace metal4
{
struct surface_mapping_state
{
	guest_surface_range logical;
	unsigned location;
	std::uint32_t pixel_address;
	std::optional<banked_tile_layout> tile;
	std::vector<guest_storage_segment> segments;
	bool operator==(const surface_mapping_state& b) const
	{
		return logical.address == b.logical.address && logical.size == b.logical.size && location == b.location &&
			pixel_address == b.pixel_address && tile == b.tile && segments == b.segments;
	}
};
class surface_mapping_conflict final : public std::runtime_error
{
  public:
	surface_mapping_conflict() : std::runtime_error("Native framebuffer IO or tile mapping changed") {}
};
// Construct/query under the same CPU writer exclusion as the VM watches/CAS.
// The owned query retains only immutable geometry and shared epoch owners.
// VM mapping/byte versions remain the cache's independent per-watch vector.
class surface_mapping_revision final
{
	std::function<surface_mapping_state()> m_query;
	std::shared_ptr<io_mapping_epoch> m_io;
	std::shared_ptr<tile_state> m_tiles;
	surface_mapping_state m_state;
	std::uint64_t m_io_version = 0, m_tile_version = 0;
	bool m_observe_tiles;
	[[nodiscard]] auto global() const
	{
		return std::array<std::uint64_t, 2>{m_io ? m_io->current() : 0, m_tiles->current()};
	}
	[[nodiscard]] auto versions() const
	{
		return std::array<std::uint64_t, 2>{m_io ? m_io->current(m_state.logical) : 0,
			m_observe_tiles ? m_tiles->current(m_state.location, m_state.logical) : 0};
	}
  public:
	surface_mapping_revision(std::function<surface_mapping_state()> query, std::shared_ptr<io_mapping_epoch> io,
		std::shared_ptr<tile_state> tiles, unsigned location, bool observe_local_tiles = true)
		: m_query(std::move(query)), m_io(location == 1 ? std::move(io) : nullptr), m_tiles(std::move(tiles)), m_observe_tiles(location == 1 || observe_local_tiles)
	{
		if (!m_query || !m_tiles || location > 1 || (location == 1 && !m_io))
			throw std::invalid_argument("Native framebuffer mapping requires owned IO/tile services");
		const auto initial = global();
		m_state = m_query();
		if (m_state.location != location || !m_state.logical.size) throw std::invalid_argument("Native framebuffer mapping location/extent differs");
		const auto values = versions(); m_io_version = values[0]; m_tile_version = values[1];
		if (global() != initial) throw surface_mapping_conflict();
	}
	[[nodiscard]] const auto& state() const { return m_state; }
	[[nodiscard]] bool current() const
	{
		const auto initial = global();
		const std::array expected{m_io_version, m_tile_version};
		if (versions() != expected) return false;
		const auto mapped = m_query();
		return mapped == m_state && versions() == expected && global() == initial;
	}
	[[nodiscard]] bool same_epoch(const surface_mapping_revision& b) const
	{
		return m_io == b.m_io && m_tiles == b.m_tiles && m_observe_tiles == b.m_observe_tiles && m_io_version == b.m_io_version &&
			m_tile_version == b.m_tile_version && m_state == b.m_state;
	}
	// Immutable bucket identity only; callers must still compare same_epoch.
	// No query/VM access and no ownership changes are needed on a cache hit.
	[[nodiscard]] std::size_t epoch_hash() const noexcept
	{
		std::size_t result = std::hash<const void*>{}(m_io.get());
		const auto add = [&](std::size_t value) { result ^= value + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2); };
		add(std::hash<const void*>{}(m_tiles.get()));
		add(m_observe_tiles);
		add(m_io_version); add(m_tile_version);
		return result;
	}
};
// Capture a complete tile for banked main memory;local tile access is already
// transparent. By default absence is watched too,so a later bind/unbind ABA
// is detected;local framebuffer data may opt out while AV receipts stay strict.
inline std::shared_ptr<const surface_mapping_revision> capture_surface_mapping(const guest_surface_layout& geometry,
	std::uint32_t offset, unsigned location, const std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)>& translate,
	const std::shared_ptr<io_mapping_epoch>& io, const std::shared_ptr<tile_state>& tiles, bool tiling, std::size_t byte_limit,
	bool observe_local_tiles = true)
{
	if (!tiles || !translate || location > 1 || geometry.range.size > byte_limit)
		throw std::invalid_argument("Native framebuffer mapping services/geometry are invalid");
	const auto query = [geometry, offset, location, translate, tiles, tiling, byte_limit]
	{
		surface_mapping_state value{{offset, geometry.range.size}, location, translate(offset, location, 1), {}, {}};
		validate_guest_surface_range(value.logical);
		if (tiling && location == 1)
		{
			std::optional<tile_descriptor> selected;
			for (const auto& tile : tiles->snapshot())
				if (tile.bound && tile.location == location && overlaps(value.logical, guest_surface_range{tile.offset, tile.size}))
				{
					if (selected || offset < tile.offset || offset - tile.offset >= tile.size)
						throw std::invalid_argument("Native framebuffer crosses multiple tile allocations");
					selected = tile;
				}
			if (selected)
			{
				if (selected->comp || selected->size > byte_limit)
					throw std::invalid_argument("Native framebuffer tile compression/budget is unsupported");
				value.logical = {selected->offset, selected->size};
				value.tile = banked_tile_layout{translate(selected->offset, location, 1), selected->size, selected->pitch, selected->bank, offset - selected->offset};
				auto image = geometry; image.range = {value.tile->address, value.tile->size};
				validate_banked_tile_geometry(image, *value.tile);
				if (!geometry.morton && geometry.row_pitch != value.tile->pitch)
					throw std::invalid_argument("Native framebuffer analysed pitch disagrees with banked tile");
			}
		}
		value.segments = texture_storage_segments(value.logical.address, location, value.logical.size, translate);
		const auto delta = std::uint64_t(offset) - value.logical.address;
		const auto origin = std::find_if(value.segments.begin(), value.segments.end(), [&](const auto& part)
			{ return delta >= part.offset && delta - part.offset < part.size; });
		if (origin == value.segments.end() || std::uint64_t(origin->address) + delta - origin->offset != value.pixel_address)
			throw surface_mapping_conflict();
		return value;
	};
	// Main-memory bank mappings always observe tile epochs. Local framebuffer
	// storage is transparent, while display receipts may separately invalidate
	// across descriptor changes without discarding the native pixel authority.
	return std::make_shared<surface_mapping_revision>(query, io, tiles, location, observe_local_tiles);
}
} // namespace metal4
