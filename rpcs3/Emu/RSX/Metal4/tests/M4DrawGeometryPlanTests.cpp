#include "M4DrawGeometryPlan.h"

#include <array>
#include <iostream>

using namespace metal4;
namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
std::vector<std::byte> encode(std::span<const std::uint32_t> indices, unsigned width)
{
	std::vector<std::byte> bytes;
	for (const auto index : indices)
		for (unsigned byte = width; byte; --byte)
			bytes.push_back(std::byte(index >> ((byte - 1) * 8)));
	return bytes;
}
struct clause
{
	unsigned primitive = 5, command = 1;
	bool is_immediate_draw = false;
	unsigned first = 10, count = 6;
	std::vector<std::uint32_t> inline_vertex_array;
	std::vector<draw_subrange> ranges;
	bool is_single_draw() const { return ranges.empty(); }
	bool empty() const { return command == 2 ? inline_vertex_array.empty() : count == 0; }
	const auto& get_subranges() const { return ranges; }
	unsigned min_index() const { return first; }
	unsigned get_elements_count() const { return count; }
};
struct state
{
	clause current_draw_clause;
	unsigned width = 0, base = 0, restart = 0;
	bool restart_enable = false;
	unsigned index_type() const { return width; }
	unsigned vertex_data_base_index() const { return base; }
	bool restart_index_enabled() const { return restart_enable; }
	unsigned restart_index() const { return restart; }
};
struct array_command
{
};
struct inline_command
{
};
struct indexed_command
{
	std::span<const std::byte> raw_index_buffer;
};
struct processor
{
	std::vector<std::byte> bytes;
	mutable unsigned calls = 0;
	std::variant<array_command, inline_command, indexed_command> get_draw_command(const state& s) const
	{
		++calls;
		if (s.current_draw_clause.command == 3)
			return indexed_command{bytes};
		if (s.current_draw_clause.command == 2)
			return inline_command{};
		return array_command{};
	}
};
struct block
{
	unsigned attribute_stride = 16;
};
struct layout
{
	std::vector<block*> interleaved_blocks;
};
} // namespace
int main()
{
	try
	{
		unsigned profiles = 0;
		// Independent arithmetic oracle across mask boundaries, unsigned base
		// overflow and raw restart domains. Underlying topology conversion has
		// separate tests; points expose every planner source ID directly.
		for (unsigned width : {2u, 4u})
			for (unsigned base : {0u, 1u, 65535u, 0xfffffu, 0x100000u, 0xfffffff0u, 0xffffffffu})
				for (unsigned first = 0; first < 1024; ++first)
				{
					const std::array<std::uint32_t, 4> indices{first, first + 1, first + 65535, 0xffffffffu};
					const auto result = plan_indexed_geometry(guest_topology::points, encode(indices, width), width, base, {}, {}, provoking_vertex::last);
					require(result.geometry.vertices.size() == 4, "Indexed point count changed");
					unsigned high = 0, low = 0xffffffffu;
					std::vector<unsigned> sources;
					for (unsigned i = 0; i < 4; ++i)
					{
						const std::uint64_t raw = indices[i] & (width == 2 ? 65535u : 0xffffffffu);
						const auto expected = static_cast<unsigned>((raw + base) % 1048576);
						require(result.geometry.vertices[i] == vertex_reference{expected, expected}, "20-bit guest address or index endian mismatch");
						high = std::max(high, expected);
						low = std::min(low, expected); sources.push_back(expected);
					}
					std::sort(sources.begin(),sources.end()); sources.erase(std::unique(sources.begin(),sources.end()),sources.end());
					require(result.streams.stream_first == low && result.streams.first_vertex == low &&
						result.streams.vertex_base_index == 0 && result.streams.vertex_index_offset == 0 && result.streams.vertex_count == high - low + 1 &&
						result.streams.source_vertices && *result.streams.source_vertices == sources,"Sparse source set or original frequency/modulo phase changed");
					++profiles;
				}
		const auto restarted = plan_indexed_geometry(guest_topology::triangle_strip,
													 encode(std::array<std::uint32_t, 9>{0, 1, 2, 0xffff, 3, 4, 5, 6, 0xffff}, 2), 2, 0xfffff,
													 0xffffffffu, {}, provoking_vertex::last);
		require(restarted.geometry.vertices == std::vector<vertex_reference>{
												   {0xfffff, 1}, {0, 1}, {1, 1}, {2, 4}, {3, 4}, {4, 4}, {4, 5}, {3, 5}, {5, 5}},
				"Restart was compared after base mask or strip parity did not reset");
		for (unsigned raw = 1; raw <= 10; ++raw)
		{
			const auto topology = decode_rsx_topology(raw);
			const auto plan = plan_array_geometry(topology, 100, 8, std::array{draw_subrange{0, 4}, draw_subrange{4, 4}}, provoking_vertex::last);
			for (const auto& vertex : plan.geometry.vertices)
				require((vertex.vertex < 104) == (vertex.flat_vertex < 104), "Flat source crossed subdraw barrier");
			const auto points = plan_array_geometry(topology, 0, 0, {}, provoking_vertex::last);
			require(points.geometry.vertices.empty() && !points.geometry.source_range && !points.streams.vertex_count, "Empty plan acquired vertex streams");
		}
		const auto loops = plan_array_geometry(guest_topology::line_loop, 5, 4, std::array{draw_subrange{0, 2}, draw_subrange{2, 2}}, provoking_vertex::last);
		require(loops.geometry.vertices == std::vector<vertex_reference>{{5, 6}, {6, 6}, {6, 5}, {5, 5}, {7, 8}, {8, 8}, {8, 7}, {7, 7}}, "Line loops connected across subrange boundary");
		const auto incomplete = plan_array_geometry(guest_topology::triangles, 7, 6, std::array{draw_subrange{0, 2}, draw_subrange{2, 4}}, provoking_vertex::last);
		require(incomplete.geometry.vertices == std::vector<vertex_reference>{{9, 11}, {10, 11}, {11, 11}}, "Incomplete primitives leaked across subrange boundary");
		const auto polygon = plan_array_geometry(guest_topology::polygon, 9, 4, {}, provoking_vertex::last);
		require(polygon.geometry.vertices[0].flat_vertex == 12 && polygon.streams.first_vertex == 9 && polygon.streams.vertex_count == 4 &&
			*polygon.streams.source_vertices == std::vector<unsigned>{9,10,11,12}, "Polygon flat-only source omitted from capture set");
		state s;
		processor p;
		block b;
		layout l{{&b}};
		const auto array = capture_draw_geometry(s, p, l, provoking_vertex::last);
		require(array.geometry.vertices[0].vertex == 10 && array.streams.first_vertex == 10 && array.streams.vertex_count == 6, "Real-shape array adapter lost clause origin");
		s.current_draw_clause.command = 2;
		s.current_draw_clause.inline_vertex_array.resize(12);
		const auto inlined = capture_draw_geometry(s, p, l, provoking_vertex::last);
		require(inlined.geometry.vertices.size() == 3 && inlined.streams.vertex_count == 3, "Inline byte stride not applied");
		s.current_draw_clause.command = 3;
		s.current_draw_clause.count = 3;
		s.current_draw_clause.is_immediate_draw = true;
		s.current_draw_clause.first = 2;
		s.width = 1;
		s.base = 1;
		p.bytes = encode(std::array<std::uint32_t, 5>{111, 222, 0xfffff, 0, 1}, 4);
		const auto indexed = capture_draw_geometry(s, p, l, provoking_vertex::last);
		require(indexed.geometry.vertices == std::vector<vertex_reference>{{0, 2}, {1, 2}, {2, 2}}, "Immediate pass was not sliced and decoded as big-endian u32");
		unsigned rejected = 0;
		const auto refusal = [&](auto action)
		{
			try
			{
				action();
			}
			catch (const std::exception&)
			{
				++rejected;
			}
		};
		refusal([]
				{ (void)decode_rsx_topology(0); });
		refusal([]
				{ (void)decode_rsx_topology(11); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::points, 0xffffffffu, 2, {}, provoking_vertex::last); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::points, 0, 4, std::array{draw_subrange{1, 4}}, provoking_vertex::last); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::points, 0, 4, std::array{draw_subrange{0, 3}}, provoking_vertex::last); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::points, 0, 4, std::array{draw_subrange{0, 5}}, provoking_vertex::last); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::points, 3, 4, {}, provoking_vertex::last, {3, 8, 8}); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::points, 3, 4, {}, provoking_vertex::last, {4, 3, 8}); });
		refusal([]
				{ (void)plan_array_geometry(guest_topology::triangle_fan, 0, 4, {}, provoking_vertex::last, {4, 4, 5}); });
		refusal([]
				{ (void)plan_indexed_geometry(guest_topology::points, std::array<std::byte, 1>{}, 2, 0, {}, {}, provoking_vertex::last); });
		s.current_draw_clause.is_immediate_draw = false;
		s.width = 2;
		refusal([&]
				{ (void)capture_draw_geometry(s, p, l, provoking_vertex::last); });
		s.width = 0;
		s.current_draw_clause.count = 4;
		refusal([&]
				{ (void)capture_draw_geometry(s, p, l, provoking_vertex::last); });
		s.current_draw_clause.command = 2;
		s.current_draw_clause.inline_vertex_array.resize(13);
		refusal([&]
				{ (void)capture_draw_geometry(s, p, l, provoking_vertex::last); });
		b.attribute_stride = 0;
		refusal([&]
				{ (void)capture_draw_geometry(s, p, l, provoking_vertex::last); });
		s.current_draw_clause.command = 3;
		s.current_draw_clause.is_immediate_draw = true;
		s.current_draw_clause.first = 4;
		s.current_draw_clause.count = 3;
		refusal([&]
				{ (void)capture_draw_geometry(s, p, l, provoking_vertex::last); });
		s.current_draw_clause.count = 0;
		const auto calls = p.calls;
		const auto empty = capture_draw_geometry(s, p, l, provoking_vertex::last);
		require(empty.geometry.vertices.empty() && p.calls == calls, "Empty RSX clause read index VM");
		require(rejected == 15, "Geometry planner refusal missing");
		std::cout << "Native RSX geometry planner: " << profiles << " address profiles, ten topologies, subrange/restart/provoking and array/indexed/inline adapters;15 refusals passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
