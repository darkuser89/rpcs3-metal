#include "M4Geometry.h"
#include "M4FlatColorLayout.h"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <set>

namespace
{

void require(bool condition)
{
	if (!condition) throw std::runtime_error("Metal 4 geometry invariant failed");
}

template <typename Function>
void rejects(Function function)
{
	try { function(); }
	catch (const std::invalid_argument&) { return; }
	catch (const std::out_of_range&) { return; }
	throw std::runtime_error("Invalid Metal 4 geometry was accepted");
}

std::size_t expected_corners(metal4::guest_topology topology, std::size_t n)
{
	using enum metal4::guest_topology;
	switch (topology)
	{
	case points: return n;
	case lines: return (n / 2) * 2;
	case line_loop: return n < 2 ? 0 : n * 2;
	case line_strip: return n < 2 ? 0 : (n - 1) * 2;
	case triangles: return (n / 3) * 3;
	case triangle_strip:
	case triangle_fan:
	case polygon: return n < 3 ? 0 : (n - 2) * 3;
	case quads: return (n / 4) * 6;
	case quad_strip: return n < 4 ? 0 : (n / 2 - 1) * 6;
	}
	throw std::runtime_error("Unknown test topology");
}

void check_index_plan(const metal4::converted_geometry& geometry, bool flat)
{
	using namespace metal4;
	const auto plan = plan_indexed_references(geometry,flat);
	require(plan.count() == geometry.vertices.size() && plan.references().topology == geometry.topology);
	require(plan.matches(geometry, flat));
	std::set<std::pair<unsigned,unsigned>> unique;
	for (auto reference : geometry.vertices)
	{
		if (!flat) reference.flat_vertex = reference.vertex;
		unique.insert({reference.vertex,reference.flat_vertex});
	}
	require(plan.references().vertices.size() == unique.size());
	std::set<unsigned> expected_flat;
	for(const auto reference:plan.references().vertices)if(reference.vertex!=reference.flat_vertex)expected_flat.insert(reference.flat_vertex);
	require(plan_flat_color_sources(plan.references().vertices)==std::vector<unsigned>(expected_flat.begin(),expected_flat.end()));
	require(bool(plan.index_width()) == (unique.size() < geometry.vertices.size()));
	require(plan.bytes().size() == unique.size() * 8 + geometry.vertices.size() * plan.index_width());
	for (std::size_t corner = 0; corner < geometry.vertices.size(); ++corner)
	{
		std::uint32_t index = static_cast<unsigned>(corner);
		if (plan.index_width() == 2) { std::uint16_t value; std::memcpy(&value,plan.bytes().data() + plan.index_offset() + corner * 2,2); index = value; }
		if (plan.index_width() == 4) std::memcpy(&index,plan.bytes().data() + plan.index_offset() + corner * 4,4);
		require(index < plan.references().vertices.size());
		auto expected = geometry.vertices[corner]; if (!flat) expected.flat_vertex = expected.vertex;
		require(plan.references().vertices[index] == expected);
		vertex_reference uploaded; std::memcpy(&uploaded,plan.bytes().data() + index * 8,8); require(uploaded == expected);
	}
}

}

int main()
{
	using namespace metal4;
	constexpr std::array topologies{guest_topology::points, guest_topology::lines,
		guest_topology::line_loop, guest_topology::line_strip, guest_topology::triangles,
		guest_topology::triangle_strip, guest_topology::triangle_fan, guest_topology::quads,
		guest_topology::quad_strip, guest_topology::polygon};
	std::size_t cases = 0;
	std::size_t index_cases = 0;
	for (auto topology : topologies)
	{
		for (auto provoking : {provoking_vertex::first, provoking_vertex::last})
		{
			for (std::size_t n = 0; n < 32; ++n)
			{
				std::vector<std::uint32_t> segment;
				for (std::size_t i = 0; i < n; ++i) segment.push_back(static_cast<std::uint32_t>(i + 10));
				const auto converted = convert_geometry(topology, segment, provoking);
				require(converted.vertices.size() == expected_corners(topology, n));
				require(converted.source_range.has_value() == !converted.vertices.empty());
				const std::size_t step = converted.topology == list_topology::points ? 1 :
					converted.topology == list_topology::lines ? 2 : 3;
				for (std::size_t i = 0; i < converted.vertices.size(); ++i)
				{
					const auto ref = converted.vertices[i];
					require(ref.vertex >= 10 && ref.vertex < 10 + n);
					require(ref.flat_vertex >= 10 && ref.flat_vertex < 10 + n);
					require(ref.flat_vertex == converted.vertices[i - i % step].flat_vertex);
					require(ref.vertex >= converted.source_range->first && ref.vertex <= converted.source_range->second);
				}
				// Splitting one draw with leading/consecutive/trailing restart
				// markers must equal two independently assembled segment draws.
				std::vector<std::uint32_t> restarted{99};
				restarted.insert(restarted.end(), segment.begin(), segment.end());
				restarted.insert(restarted.end(), {99, 99});
				restarted.insert(restarted.end(), segment.begin(), segment.end());
				restarted.push_back(99);
				const auto split = convert_geometry(topology, restarted, provoking, 99);
				auto expected = converted.vertices;
				expected.insert(expected.end(), converted.vertices.begin(), converted.vertices.end());
				require(split.vertices == expected);
				for (bool flat : {false,true})
				{
					check_index_plan(converted,flat); check_index_plan(split,flat); index_cases += 2;
				}
				++cases;
			}
		}
	}
	const std::array<std::uint32_t, 4> quad{0, 1, 2, 3};
	const auto q = convert_geometry(guest_topology::quads, quad, provoking_vertex::last);
	require(q.vertices == std::vector<vertex_reference>{{0, 3}, {1, 3}, {2, 3}, {2, 3}, {3, 3}, {0, 3}});
	const auto strip = convert_geometry(guest_topology::triangle_strip, quad, provoking_vertex::last);
	require(strip.vertices == std::vector<vertex_reference>{{0, 2}, {1, 2}, {2, 2}, {2, 3}, {1, 3}, {3, 3}});
	const auto qs = convert_geometry(guest_topology::quad_strip, quad, provoking_vertex::last);
	require(qs.vertices == std::vector<vertex_reference>{{0, 3}, {1, 3}, {2, 3}, {2, 3}, {1, 3}, {3, 3}});
	const auto loop = convert_geometry(guest_topology::line_loop, quad, provoking_vertex::last);
	require(loop.vertices.back() == vertex_reference{0, 0} && loop.vertices[6] == vertex_reference{3, 0});
	const auto fan = convert_geometry(guest_topology::triangle_fan, quad, provoking_vertex::first);
	require(fan.vertices[0].flat_vertex == 1 && fan.vertices[3].flat_vertex == 2);
	const auto polygon = convert_geometry(guest_topology::polygon, quad, provoking_vertex::last);
	for (const auto ref : polygon.vertices) require(ref.flat_vertex == 3);
	const std::array<std::uint32_t, 3> repeated{7, 7, 7};
	const auto points = convert_geometry(guest_topology::points, repeated, provoking_vertex::last);
	require(points.vertices.size() == 3 && points.source_range == std::pair{7u, 7u});
	const auto degenerate = convert_geometry(guest_topology::triangles, repeated, provoking_vertex::last);
	require(degenerate.vertices.size() == 3 && degenerate.source_range == std::pair{7u, 7u});
	check_index_plan(points,true); check_index_plan(degenerate,true);
	for (unsigned count : {65534u,65535u,65536u,65537u})
	{
		converted_geometry boundary{list_topology::points,{},std::pair{0xffffffffu,0xffffffffu}};
		for (unsigned i = 0; i < count; ++i) boundary.vertices.push_back({i,i});
		boundary.vertices.push_back({0,0});
		const auto plan = plan_indexed_references(boundary);
		require(plan.index_width() == (count <= 65535 ? 2 : 4));
		require(plan.references().source_range == std::pair{0u,count - 1});
		check_index_plan(boundary,true);
		boundary.vertices.back() = {1,1};
		require(!plan.matches(boundary, true));
	}
	const converted_geometry pair_identity{list_topology::triangles,{{0xffffffffu,0},{0xffffffffu,1},{0xffffffffu,0}},std::nullopt};
	require(plan_indexed_references(pair_identity,true).references().vertices.size() == 2);
	require(plan_indexed_references(pair_identity,false).references().vertices.size() == 1);
	check_index_plan(pair_identity,true); check_index_plan(pair_identity,false);
	const auto flat_plan = plan_indexed_references(pair_identity, true);
	const auto smooth_plan = plan_indexed_references(pair_identity, false);
	auto changed_pair = pair_identity;
	changed_pair.vertices[1].flat_vertex = 3;
	require(!flat_plan.matches(changed_pair, true) && smooth_plan.matches(changed_pair, false) &&
		!smooth_plan.matches(changed_pair, true));
	changed_pair.topology = list_topology::points;
	require(!smooth_plan.matches(changed_pair, false));
	auto unindexed = convert_array_geometry(guest_topology::points, 0, 3, provoking_vertex::last);
	const auto unindexed_plan = plan_indexed_references(unindexed);
	require(!unindexed_plan.index_width() && unindexed_plan.matches(unindexed, true));
	std::swap(unindexed.vertices.front(), unindexed.vertices.back());
	require(!unindexed_plan.matches(unindexed, true));
	std::cout << "Geometry plan reuse: exact corner order, topology and effective Flat Color sources verified for unindexed, 16-bit and 32-bit packets\n";
	require(plan_flat_color_sources(std::array{vertex_reference{9,0xffffffffu},vertex_reference{0,7},vertex_reference{7,7},vertex_reference{1,7}})==std::vector<unsigned>{7,0xffffffffu});
	require(flat_color_bytes(0,0)==0 && flat_color_bytes(1,84)==80 && !flat_color_bytes(1,83));
	require(!flat_color_bytes(std::numeric_limits<std::size_t>::max(),std::numeric_limits<std::size_t>::max()));
	require(flat_color_bytes(512u*1024*1024/84,512u*1024*1024).has_value() && !flat_color_bytes(512u*1024*1024/84+1,512u*1024*1024));
	rejects([] { (void)plan_indexed_references({list_topology::triangles,{{0,0},{1,1}},std::nullopt}); });
	rejects([] { (void)plan_indexed_references({static_cast<list_topology>(4),{},std::nullopt}); });
	constexpr auto max_id = std::numeric_limits<std::uint32_t>::max();
	const auto end = convert_array_geometry(guest_topology::points, max_id, 1, provoking_vertex::last);
	require(end.vertices == std::vector<vertex_reference>{{max_id, max_id}});
	rejects([] { (void)convert_array_geometry(guest_topology::points, max_id, 2, provoking_vertex::last); });
	rejects([] { (void)convert_geometry(static_cast<guest_topology>(100), {}, provoking_vertex::last); });
	rejects([] { (void)convert_geometry(guest_topology::points, {}, static_cast<provoking_vertex>(100)); });
	// Unaligned guest bytes and explicit endian conversion: no typed pointer cast.
	const std::array<std::byte, 5> encoded{std::byte{0}, std::byte{0x12}, std::byte{0x34}, std::byte{0xab}, std::byte{0xcd}};
	const auto bytes = std::span(encoded).subspan(1);
	require(decode_guest_indices(bytes, 2, true) == std::vector<std::uint32_t>{0x1234, 0xabcd});
	require(decode_guest_indices(bytes, 2, false) == std::vector<std::uint32_t>{0x3412, 0xcdab});
	require(decode_guest_indices(bytes, 4, true) == std::vector<std::uint32_t>{0x1234abcd});
	require(decode_guest_indices(bytes, 4, false) == std::vector<std::uint32_t>{0xcdab3412});
	rejects([&] { (void)decode_guest_indices(bytes, 1, true); });
	rejects([&] { (void)decode_guest_indices(bytes.first(3), 2, true); });
	std::cout << "Metal 4 geometry: " << cases << " topology/restart cases plus flat-source and endian checks passed\n";
	std::cout << "Native reference indexing:" << index_cases << " ordered corner/restart/topology/smooth/flat roundtrips;16/32bit boundaries,pair identity,degenerate/repeated points,untrusted source_range andincomplete topology refusals passed\n";
	std::cout << "Flat color planning:sorted unique differing source IDs inall2560roundtrips,flat-only/maxuint IDs,80byteABI/84bytecombinedbudget/exact-boundary/overflow arithmetic passed\n";
}
