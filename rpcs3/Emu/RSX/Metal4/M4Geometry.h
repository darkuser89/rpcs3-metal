#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace metal4
{

enum class guest_topology
{
	points, lines, line_loop, line_strip, triangles, triangle_strip,
	triangle_fan, quads, quad_strip, polygon,
};

enum class list_topology { points, lines, triangles };
enum class provoking_vertex { first, last };

// MSL ABI: uint source_vertex; uint flat_vertex; in buffer(30). The vertex program
// fetches position/smooth attributes using vertex, and flat colors using the
// output of flat_vertex. Index rotation alone cannot preserve both quad colors
// and the original diagonal. The RSX adapter must select the provoking rule,
// including the whole-polygon rule, rather than relying on Metal defaults.
struct vertex_reference
{
	std::uint32_t vertex;
	std::uint32_t flat_vertex;
	bool operator==(const vertex_reference&) const = default;
};
static_assert(sizeof(vertex_reference) == 8 && offsetof(vertex_reference, flat_vertex) == 4);
inline constexpr std::size_t geometry_binding = 30;

struct converted_geometry
{
	list_topology topology;
	std::vector<vertex_reference> vertices;
	// Includes flat-color sources that aren't in a triangle's geometric corners.
	std::optional<std::pair<std::uint32_t, std::uint32_t>> source_range;
};

// Stable first-use indexing of converted RSX references. Pair identity keeps
// distinct provoking colors; smooth-only programs can share by source ID.
// The ordered index stream reconstructs every corner, including degenerates.
// Private construction prevents a stale/forged index stream after planning.
class geometry_index_plan final
{
	converted_geometry m_references;
	std::vector<std::byte> m_bytes;
	std::uint32_t m_count = 0, m_index_width = 0;
	geometry_index_plan() = default;
	friend geometry_index_plan plan_indexed_references(const converted_geometry&, bool);
  public:
	[[nodiscard]] const converted_geometry& references() const noexcept { return m_references; }
	[[nodiscard]] std::span<const std::byte> bytes() const noexcept { return m_bytes; }
	[[nodiscard]] std::uint32_t count() const noexcept { return m_count; }
	[[nodiscard]] std::uint32_t index_width() const noexcept { return m_index_width; }
	[[nodiscard]] std::size_t index_offset() const noexcept { return m_references.vertices.size() * sizeof(vertex_reference); }
};
[[nodiscard]] geometry_index_plan plan_indexed_references(const converted_geometry& geometry, bool flat_colors = true);
// Only IDs that require colors from a different vertex; independent of the
// geometric corner interval and including polygon/quad flat-only sources.
[[nodiscard]] std::vector<std::uint32_t> plan_flat_color_sources(std::span<const vertex_reference> references);

// Produces independent lists without hardware restart markers. Restart resets
// strip parity, fans, loops and incomplete primitives at every segment boundary.
// Degenerates and repeated single-vertex point indices are intentionally kept.
// The caller supplies restart in the decoded index domain (16-bit markers must
// be narrowed before this call). Source-range lengths require 64-bit arithmetic.
[[nodiscard]] converted_geometry convert_geometry(guest_topology topology,
	std::span<const std::uint32_t> indices, provoking_vertex provoking,
	std::optional<std::uint32_t> restart = std::nullopt);
[[nodiscard]] converted_geometry convert_array_geometry(guest_topology topology,
	std::size_t first, std::size_t count, provoking_vertex provoking);
[[nodiscard]] std::vector<std::uint32_t> decode_guest_indices(std::span<const std::byte> bytes,
	std::size_t index_width, bool big_endian);

}
