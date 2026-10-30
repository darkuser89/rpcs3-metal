#pragma once

#include "M4Geometry.h"
#include "M4VertexDataSnapshot.h"

#include <variant>

namespace metal4
{
struct draw_subrange
{
	std::uint32_t first = 0, count = 0;
};
struct draw_geometry_plan
{
	converted_geometry geometry;
	vertex_stream_range streams;
};
struct geometry_plan_limits
{
	std::uint32_t input_elements = 1u << 20;
	std::uint32_t source_vertices = 1u << 20;
	std::uint32_t converted_vertices = 3u << 20;
};
[[nodiscard]] guest_topology decode_rsx_topology(std::uint32_t primitive);
// Subranges partition the current command, not all passes in the clause.
// Separate conversion resets adjacency, winding, incomplete primitives and
// polygon provoking colors at each explicit barrier. Restart is compared in
// the raw decoded domain before applying the guest 20-bit base-index mask.
[[nodiscard]] draw_geometry_plan plan_array_geometry(guest_topology topology, std::uint32_t first,
													 std::uint32_t count, std::span<const draw_subrange> subranges, provoking_vertex provoking,
													 geometry_plan_limits limits = {});
[[nodiscard]] draw_geometry_plan plan_indexed_geometry(guest_topology topology,
													   std::span<const std::byte> bytes, std::uint32_t index_width, std::uint32_t base_index,
													   std::optional<std::uint32_t> restart, std::span<const draw_subrange> subranges,
													   provoking_vertex provoking, geometry_plan_limits limits = {});

// Real DrawCommands/rsx_state adapter. The caller executes current-pass
// dependencies and owns a coherent state/layout/index-VM epoch. Does not
// advance the clause, collapse instances or provide VM locks/DMA barriers.
template <typename State, typename Processor, typename Layout>
[[nodiscard]] draw_geometry_plan capture_draw_geometry(const State& state, const Processor& processor,
													   const Layout& layout, provoking_vertex provoking, geometry_plan_limits limits = {})
{
	const auto& clause = state.current_draw_clause;
	const auto topology = decode_rsx_topology(static_cast<std::uint32_t>(clause.primitive));
	if (clause.empty())
		return plan_array_geometry(topology, 0, 0, {}, provoking, limits);
	std::vector<draw_subrange> subranges;
	if (static_cast<std::uint32_t>(clause.command) != 2 && !clause.is_single_draw())
		for (const auto& range : clause.get_subranges())
			subranges.push_back({range.first, range.count});
	const auto command = processor.get_draw_command(state);
	const auto visit = [&](const auto& draw) -> draw_geometry_plan
	{
		if constexpr (requires { draw.raw_index_buffer; })
		{
			const auto width = clause.is_immediate_draw ? 4u : (static_cast<std::uint32_t>(state.index_type()) == 0 ? 4u : 2u);
			if (!clause.is_immediate_draw && static_cast<std::uint32_t>(state.index_type()) > 1)
				throw std::invalid_argument("Native RSX index type is invalid");
			auto bytes = draw.raw_index_buffer;
			if (bytes.size() % width)
				throw std::invalid_argument("Native RSX index span contains incomplete elements");
			if (clause.is_immediate_draw)
			{
				// Immediate get_raw_index_array returns the entire push buffer;
				// ordinary VM arrays are already sliced to the current pass.
				const std::uint64_t first = static_cast<std::uint64_t>(clause.min_index()) * width;
				const std::uint64_t length = static_cast<std::uint64_t>(clause.get_elements_count()) * width;
				if (first > bytes.size() || length > bytes.size() - first)
					throw std::out_of_range("Native immediate index pass exceeds push buffer");
				bytes = bytes.subspan(static_cast<std::size_t>(first), static_cast<std::size_t>(length));
			}
			else if (bytes.size() / width != clause.get_elements_count())
				throw std::invalid_argument("Native RSX index span differs from current draw count");
			const auto restart = state.restart_index_enabled() ? std::optional<std::uint32_t>(state.restart_index()) : std::nullopt;
			return plan_indexed_geometry(topology, bytes, width, state.vertex_data_base_index(), restart,
										 subranges, provoking, limits);
		}
		else
		{
			if (static_cast<std::uint32_t>(clause.command) == 2)
			{
				if (layout.interleaved_blocks.empty() || !layout.interleaved_blocks[0] || !layout.interleaved_blocks[0]->attribute_stride)
					throw std::invalid_argument("Native inline draw has no vertex stride");
				const std::uint64_t bytes = static_cast<std::uint64_t>(clause.inline_vertex_array.size()) * 4;
				const auto stride = layout.interleaved_blocks[0]->attribute_stride;
				if (bytes % stride || bytes / stride > std::numeric_limits<std::uint32_t>::max())
					throw std::invalid_argument("Native inline draw contains incomplete vertices");
				return plan_array_geometry(topology, 0, static_cast<std::uint32_t>(bytes / stride), {}, provoking, limits);
			}
			return plan_array_geometry(topology, clause.min_index(), clause.get_elements_count(), subranges, provoking, limits);
		}
	};
	return std::visit(visit, command);
}
} // namespace metal4
