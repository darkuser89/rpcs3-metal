#include "M4Geometry.h"
#include "M4DrawValidation.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <unordered_map>

namespace metal4
{

std::vector<std::uint32_t> plan_flat_color_sources(std::span<const vertex_reference> references)
{
	std::vector<std::uint32_t> ids;
	for (const auto reference : references) if (reference.vertex!=reference.flat_vertex) ids.push_back(reference.flat_vertex);
	std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
	return ids;
}

geometry_index_plan plan_indexed_references(const converted_geometry& geometry, bool flat_colors)
{
	unsigned corners;
	switch (geometry.topology)
	{
	case list_topology::points: corners = 1; break;
	case list_topology::lines: corners = 2; break;
	case list_topology::triangles: corners = 3; break;
	default: throw std::invalid_argument("Native indexed geometry topology is invalid");
	}
	validate_draw_ids(0,geometry.vertices.size());
	if (geometry.vertices.size() % corners) throw std::invalid_argument("Native indexed geometry has an incomplete primitive");
	// Bound source inspection and host/native packet arithmetic before reads.
	if (geometry.vertices.size() > 0x20000000u / sizeof(vertex_reference))
		throw std::length_error("Native geometry reference stream exceeds512MiB");
	geometry_index_plan result;
	result.m_references.topology = geometry.topology;
	result.m_count = static_cast<std::uint32_t>(geometry.vertices.size());
	std::unordered_map<std::uint64_t,std::uint32_t> keys;
	std::vector<std::uint32_t> indices;
	indices.reserve(geometry.vertices.size());
	for (auto reference : geometry.vertices)
	{
		if (!flat_colors) reference.flat_vertex = reference.vertex;
		const auto key = std::uint64_t(reference.vertex) | (std::uint64_t(reference.flat_vertex) << 32);
		const auto [entry,inserted] = keys.try_emplace(key,static_cast<std::uint32_t>(result.m_references.vertices.size()));
		if (inserted)
		{
			result.m_references.vertices.push_back(reference);
			auto& range = result.m_references.source_range;
			if (!range) range = {std::min(reference.vertex,reference.flat_vertex),std::max(reference.vertex,reference.flat_vertex)};
			else { range->first = std::min({range->first,reference.vertex,reference.flat_vertex}); range->second = std::max({range->second,reference.vertex,reference.flat_vertex}); }
		}
		indices.push_back(entry->second);
	}
	if (result.m_references.vertices.size() != geometry.vertices.size())
		// Avoid all-ones native index values; guest restart is already removed.
		result.m_index_width = result.m_references.vertices.size() <= 65535 ? 2 : 4;
	const auto reference_bytes = result.index_offset();
	const auto index_bytes = std::uint64_t(result.m_index_width) * result.m_count;
	if (reference_bytes + index_bytes > 0x20000000ull) throw std::length_error("Native indexed geometry packet exceeds512MiB");
	result.m_bytes.resize(reference_bytes + static_cast<std::size_t>(index_bytes));
	if (reference_bytes) std::memcpy(result.m_bytes.data(),result.m_references.vertices.data(),reference_bytes);
	if (result.m_index_width) for (std::size_t i = 0; i < indices.size(); ++i)
	{
		if (result.m_index_width == 2)
		{
			const auto value = static_cast<std::uint16_t>(indices[i]);
			std::memcpy(result.m_bytes.data() + reference_bytes + i * 2,&value,2);
		}
		else std::memcpy(result.m_bytes.data() + reference_bytes + i * 4,&indices[i],4);
	}
	return result;
}

namespace
{

list_topology native_list(guest_topology topology)
{
	switch (topology)
	{
	case guest_topology::points: return list_topology::points;
	case guest_topology::lines:
	case guest_topology::line_loop:
	case guest_topology::line_strip: return list_topology::lines;
	case guest_topology::triangles:
	case guest_topology::triangle_strip:
	case guest_topology::triangle_fan:
	case guest_topology::quads:
	case guest_topology::quad_strip:
	case guest_topology::polygon: return list_topology::triangles;
	}
	throw std::invalid_argument("Unknown Metal 4 guest topology");
}

void append_segment(converted_geometry& result, guest_topology topology,
	std::span<const std::uint32_t> indices, provoking_vertex provoking)
{
	const auto n = indices.size();
	const bool last = provoking == provoking_vertex::last;
	auto emit = [&](std::span<const std::uint32_t> corners, std::uint32_t flat)
	{
		if (corners.size() > std::numeric_limits<std::uint32_t>::max() - result.vertices.size())
		{
			throw std::length_error("Converted Metal 4 geometry exceeds uint vertex IDs");
		}
		for (const auto corner : corners) result.vertices.push_back({corner, flat});
	};
	auto point = [&](std::uint32_t a) { emit(std::array{a}, a); };
	auto line = [&](std::uint32_t a, std::uint32_t b) { emit(std::array{a, b}, last ? b : a); };
	auto triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t flat)
	{
		emit(std::array{a, b, c}, flat);
	};
	switch (topology)
	{
	case guest_topology::points:
		for (const auto index : indices) point(index);
		break;
	case guest_topology::lines:
		for (std::size_t i = 0; i < n / 2; ++i) line(indices[i * 2], indices[i * 2 + 1]);
		break;
	case guest_topology::line_loop:
	case guest_topology::line_strip:
		for (std::size_t i = 1; i < n; ++i) line(indices[i - 1], indices[i]);
		if (topology == guest_topology::line_loop && n > 1) line(indices.back(), indices.front());
		break;
	case guest_topology::triangles:
		for (std::size_t i = 0; i < n / 3; ++i)
		{
			const auto offset = i * 3;
			triangle(indices[offset], indices[offset + 1], indices[offset + 2], indices[offset + (last ? 2 : 0)]);
		}
		break;
	case guest_topology::triangle_strip:
		for (std::size_t i = 2; i < n; ++i)
		{
			const auto flat = indices[last ? i : i - 2];
			// Preserve the alternating strip winding while emitting triangle lists.
			if (i & 1) triangle(indices[i - 1], indices[i - 2], indices[i], flat);
			else triangle(indices[i - 2], indices[i - 1], indices[i], flat);
		}
		break;
	case guest_topology::triangle_fan:
	case guest_topology::polygon:
		for (std::size_t i = 2; i < n; ++i)
		{
			const auto flat = topology == guest_topology::polygon
				? (last ? indices.back() : indices.front())
				: indices[last ? i : i - 1];
			triangle(indices[0], indices[i - 1], indices[i], flat);
		}
		break;
	case guest_topology::quads:
		for (std::size_t i = 0; i < n / 4; ++i)
		{
			const auto offset = i * 4;
			const auto flat = indices[offset + (last ? 3 : 0)];
			triangle(indices[offset], indices[offset + 1], indices[offset + 2], flat);
			triangle(indices[offset + 2], indices[offset + 3], indices[offset], flat);
		}
		break;
	case guest_topology::quad_strip:
		for (std::size_t pair = 1; pair < n / 2; ++pair)
		{
			const auto offset = (pair - 1) * 2;
			const auto flat = indices[offset + (last ? 3 : 0)];
			triangle(indices[offset], indices[offset + 1], indices[offset + 2], flat);
			triangle(indices[offset + 2], indices[offset + 1], indices[offset + 3], flat);
		}
		break;
	}
}

}

converted_geometry convert_geometry(guest_topology topology, std::span<const std::uint32_t> indices,
	provoking_vertex provoking, std::optional<std::uint32_t> restart)
{
	if (provoking != provoking_vertex::first && provoking != provoking_vertex::last)
	{
		throw std::invalid_argument("Unknown Metal 4 provoking vertex rule");
	}
	validate_draw_ids(0, indices.size());
	converted_geometry result{native_list(topology), {}, std::nullopt};
	std::size_t segment_start = 0;
	for (std::size_t i = 0; i < indices.size(); ++i)
	{
		if (restart && indices[i] == *restart)
		{
			append_segment(result, topology, indices.subspan(segment_start, i - segment_start), provoking);
			segment_start = i + 1;
		}
	}
	append_segment(result, topology, indices.subspan(segment_start), provoking);
	if (!result.vertices.empty())
	{
		std::uint32_t low = std::numeric_limits<std::uint32_t>::max();
		std::uint32_t high = 0;
		for (const auto& ref : result.vertices)
		{
			low = std::min({low, ref.vertex, ref.flat_vertex});
			high = std::max({high, ref.vertex, ref.flat_vertex});
		}
		result.source_range = std::pair{low, high};
	}
	return result;
}

converted_geometry convert_array_geometry(guest_topology topology, std::size_t first,
	std::size_t count, provoking_vertex provoking)
{
	validate_draw_ids(first, count);
	std::vector<std::uint32_t> indices(count);
	for (std::size_t i = 0; i < count; ++i) indices[i] = static_cast<std::uint32_t>(first + i);
	return convert_geometry(topology, indices, provoking);
}

std::vector<std::uint32_t> decode_guest_indices(std::span<const std::byte> bytes,
	std::size_t index_width, bool big_endian)
{
	if ((index_width != 2 && index_width != 4) || bytes.size() % index_width)
	{
		throw std::invalid_argument("Guest Metal 4 indices require complete 16- or 32-bit elements");
	}
	validate_draw_ids(0, bytes.size() / index_width);
	std::vector<std::uint32_t> result(bytes.size() / index_width);
	for (std::size_t i = 0; i < result.size(); ++i)
	{
		std::uint32_t value = 0;
		for (std::size_t byte = 0; byte < index_width; ++byte)
		{
			const auto shift = static_cast<unsigned>((big_endian ? index_width - 1 - byte : byte) * 8);
			value |= std::to_integer<std::uint32_t>(bytes[i * index_width + byte]) << shift;
		}
		result[i] = value;
	}
	return result;
}

}
