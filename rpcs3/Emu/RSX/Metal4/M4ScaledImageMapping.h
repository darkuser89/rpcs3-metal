#pragma once

#include "M4BankedTile.h"
#include "M4GuestSegments.h"
#include <algorithm>
#include <array>
#include <functional>

namespace metal4
{
// Owned values only. No GcmTileInfo pointer or guest backing survives capture.
struct scaled_image_tile_mapping
{
	std::uint32_t address, size, pitch, bank, location, io_offset, compression, compression_base;
	bool operator==(const scaled_image_tile_mapping&) const = default;
};
struct scaled_image_memory_mapping
{
	std::uint32_t address;
	std::size_t size;
	std::optional<scaled_image_tile_mapping> tile;
	std::vector<guest_storage_segment> segments = {};
	struct io_range
	{
		std::uint32_t offset;
		std::size_t size;
		bool operator==(const io_range&) const = default;
	};
	std::optional<io_range> io = {};
	struct tile_range
	{
		std::uint32_t offset;
		std::size_t size;
		unsigned location;
		bool operator==(const tile_range&) const = default;
	};
	// Record eligibility even when no tile was initially bound: bind/unbind
	// ABA must invalidate that absence as well as an existing descriptor.
	std::optional<tile_range> tiling = {};
	bool operator==(const scaled_image_memory_mapping&) const = default;
};
// Resolve only owned translated EAs. Banked captures retain the full tile's
// actual segments,which can be disjoint;do not use a fictitious linear span.
template <typename Memory>
std::uint64_t scaled_image_mapping_generation(const std::array<scaled_image_memory_mapping, 2>& values, const Memory& memory)
{
	std::uint64_t result = 0;
	for (const auto& value : values)
	{
		if (!value.segments.empty())
			for (const auto& segment : value.segments)
				result = std::max(result, memory.mapping_generation({segment.address, segment.size}));
		else if (value.tile)
			result = std::max(result, memory.mapping_generation({value.tile->address, value.tile->size}));
		else
			result = std::max(result, memory.mapping_generation({value.address, value.size}));
	}
	return result;
}
template <typename IO>
std::uint64_t scaled_image_io_mapping_generation(const std::array<scaled_image_memory_mapping, 2>& values, const IO& epochs)
{
	std::uint64_t result = 0;
	for (const auto& value : values)
		if (value.io)
			result = std::max(result, epochs.current({value.io->offset, value.io->size}));
	return result;
}
template <typename Tiles>
std::uint64_t scaled_image_tile_mapping_generation(const std::array<scaled_image_memory_mapping, 2>& values, const Tiles& tiles)
{
	std::uint64_t result = 0;
	for (const auto& value : values)
		if (value.tiling)
			result = std::max(result, tiles.current(value.tiling->location, {value.tiling->offset, value.tiling->size}));
	return result;
}
// Independent counters must remain independent. Taking max(vm,io) would hide
// an IO transition whenever the VM counter is already larger. VM-only callers
// can keep returning one scalar;the IO domain then remains explicitly zero.
struct scaled_image_mapping_versions
{
	std::uint64_t vm, io, tile;
	scaled_image_mapping_versions(std::uint64_t vm_value = 0, std::uint64_t io_value = 0, std::uint64_t tile_value = 0)
		: vm(vm_value), io(io_value), tile(tile_value) {}
	bool operator==(const scaled_image_mapping_versions&) const = default;
};
class scaled_image_mapping_conflict final : public std::runtime_error
{
  public:
	scaled_image_mapping_conflict() : std::runtime_error("Native scaled-image IO or tile mapping changed") {}
};
// Query and validate only within the same CPU exclusion contract as memory
// capture/publication. Equality catches changed mappings,not an identical ABA;
// Owned producer generations catch observed VM/IO/tile ABA independently.
// All three producer domains must stay excluded through byte capture/CAS.
// Unobserved producer paths and productive backing leases remain open.
class scaled_image_mapping_epoch final
{
  public:
	using mappings = std::array<scaled_image_memory_mapping, 2>;
	using query = std::function<mappings()>;
	using generation = std::function<scaled_image_mapping_versions()>;
	using range_generation = std::function<scaled_image_mapping_versions(const mappings&)>;

  private:
	query m_query;
	generation m_generation;
	range_generation m_range_generation;
	scaled_image_mapping_versions m_original_generation;
	mappings m_original;
	[[nodiscard]] scaled_image_mapping_versions version() const
	{
		return m_range_generation ? m_range_generation(m_original) : m_generation();
	}

  public:
	explicit scaled_image_mapping_epoch(query probe, generation producer = {}, range_generation ranges = {})
		: m_query(std::move(probe)), m_generation(std::move(producer)), m_range_generation(std::move(ranges))
	{
		if (!m_query || (m_range_generation && !m_generation))
			throw std::invalid_argument("Native scaled-image mapping query/generation is absent");
		// Before translation the relevant EAs are unknown. Check the global stamp
		// across initial capture;later validation uses only the captured ranges.
		const auto initial = m_generation ? m_generation() : scaled_image_mapping_versions{};
		m_original = m_query();
		if (m_generation)
			m_original_generation = m_range_generation ? version() : initial;
		if (m_generation && m_generation() != initial)
			throw scaled_image_mapping_conflict();
	}
	[[nodiscard]] const mappings& original() const noexcept { return m_original; }
	void validate() const
	{
		if ((m_generation && version() != m_original_generation) || m_query() != m_original)
			throw scaled_image_mapping_conflict();
		if (m_generation && version() != m_original_generation)
			throw scaled_image_mapping_conflict();
	}
};
} // namespace metal4
