#include "M4DrawGeometryPlan.h"

#include <algorithm>
#include <iterator>

namespace metal4
{
guest_topology decode_rsx_topology(std::uint32_t primitive)
{
	if (!primitive || primitive > 10)
		throw std::invalid_argument("Native RSX primitive type is invalid");
	return static_cast<guest_topology>(primitive - 1);
}
namespace
{
std::vector<draw_subrange> checked_ranges(std::uint32_t count, std::span<const draw_subrange> ranges, geometry_plan_limits limits)
{
	if (count > limits.input_elements)
		throw std::length_error("Native draw input count exceeds budget");
	if (ranges.empty())
		return {{0, count}};
	std::uint64_t cursor = 0;
	for (const auto& range : ranges)
	{
		if (range.first != cursor || !range.count || range.count > count - cursor)
			throw std::invalid_argument("Native draw subranges do not partition current command");
		cursor += range.count;
	}
	if (cursor != count)
		throw std::invalid_argument("Native draw subranges omit current command elements");
	return {ranges.begin(), ranges.end()};
}
void append(draw_geometry_plan& plan, converted_geometry geometry, geometry_plan_limits limits)
{
	if (geometry.vertices.size() > limits.converted_vertices - plan.geometry.vertices.size())
		throw std::length_error("Native converted draw exceeds geometry budget");
	plan.geometry.vertices.insert(plan.geometry.vertices.end(), geometry.vertices.begin(), geometry.vertices.end());
}
void finish(draw_geometry_plan& plan, geometry_plan_limits limits)
{
	if (plan.geometry.vertices.empty())
	{
		plan.streams.source_vertices = std::make_shared<const std::vector<std::uint32_t>>();
		return;
	}
	std::uint32_t low = 0xffffffffu, high = 0;
	for (const auto& reference : plan.geometry.vertices)
	{
		low = std::min({low, reference.vertex, reference.flat_vertex});
		high = std::max({high, reference.vertex, reference.flat_vertex});
	}
	auto sources = std::make_shared<std::vector<std::uint32_t>>();
	sources->reserve(plan.geometry.vertices.size() * 2);
	for (const auto& reference : plan.geometry.vertices)
	{
		sources->push_back(reference.vertex);
		sources->push_back(reference.flat_vertex);
	}
	std::sort(sources->begin(),sources->end());
	sources->erase(std::unique(sources->begin(),sources->end()),sources->end());
	if (sources->size() > limits.source_vertices || std::uint64_t(high) - low + 1 > 0xffffffffull)
		throw std::length_error("Native vertex source set exceeds budget or shader envelope");
	plan.geometry.source_range = std::pair{low, high};
	plan.streams.stream_first = plan.streams.first_vertex = low;
	plan.streams.vertex_count = high - low + 1;
	plan.streams.source_vertices = std::move(sources);
}
} // namespace
draw_geometry_plan plan_array_geometry(guest_topology topology, std::uint32_t first, std::uint32_t count,
									   std::span<const draw_subrange> subranges, provoking_vertex provoking, geometry_plan_limits limits)
{
	const auto ranges = checked_ranges(count, subranges, limits);
	if (count && count - 1 > 0xffffffffu - first)
		throw std::overflow_error("Native array vertex range wraps");
	draw_geometry_plan result{convert_geometry(topology, {}, provoking), {}};
	for (const auto& range : ranges)
		append(result, convert_array_geometry(topology, static_cast<std::uint64_t>(first) + range.first, range.count, provoking), limits);
	finish(result, limits);
	return result;
}
draw_geometry_plan plan_indexed_geometry(guest_topology topology, std::span<const std::byte> bytes,
										 std::uint32_t index_width, std::uint32_t base_index, std::optional<std::uint32_t> restart,
										 std::span<const draw_subrange> subranges, provoking_vertex provoking, geometry_plan_limits limits)
{
	if ((index_width != 2 && index_width != 4) || bytes.size() % index_width || bytes.size() / index_width > limits.input_elements)
		throw std::invalid_argument("Native indexed draw has invalid or over-budget elements");
	const auto ranges = checked_ranges(static_cast<std::uint32_t>(bytes.size() / index_width), subranges, limits);
	const auto indices = decode_guest_indices(bytes, index_width, true);
	if (restart && index_width == 2)
		*restart &= 65535;
	draw_geometry_plan result{convert_geometry(topology, {}, provoking), {}};
	for (const auto& range : ranges)
	{
		auto geometry = convert_geometry(topology, std::span(indices).subspan(range.first, range.count), provoking, restart);
		for (auto& reference : geometry.vertices)
		{
			reference.vertex = (reference.vertex + base_index) & 0xfffffu;
			reference.flat_vertex = (reference.flat_vertex + base_index) & 0xfffffu;
		}
		append(result, std::move(geometry), limits);
	}
	finish(result, limits);
	return result;
}
} // namespace metal4
