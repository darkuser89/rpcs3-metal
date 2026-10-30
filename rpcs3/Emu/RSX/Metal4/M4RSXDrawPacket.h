#pragma once

#include "M4DrawEnvironment.h"
#include "M4DrawGeometryPlan.h"
#include "M4RSXDrawSnapshot.h"
#include "M4VertexDataSnapshot.h"
#include <functional>

namespace metal4
{
struct draw_capture_limits
{
	std::size_t fragment_bytes = 1u << 20;
	std::size_t vertex_stream_bytes = 64u << 20;
	geometry_plan_limits geometry;
	bool constants_are_current_pass = false;
	vertex_constant_cache* constants = nullptr;
	std::function<vertex_sparse_inputs(const vertex_stream_range&)> sparse_sources;
};
// One owned current-pass input packet. Capture while the renderer excludes
// method/VM writers across ALL operations below, after pass dependencies and
// program/layout analysis. This boundary does not acquire those locks,advance
// the draw clause,resolve framebuffer addresses or snapshot texture VM bytes.
class rsx_draw_packet final
{
	rsx_draw_snapshot m_state;
	draw_environment_snapshot m_environment;
	converted_geometry m_geometry;
	vertex_data_snapshot m_vertices;
	rsx_draw_packet(rsx_draw_snapshot state, draw_environment_snapshot environment,
					converted_geometry geometry, vertex_data_snapshot vertices)
		: m_state(std::move(state)), m_environment(std::move(environment)),
		  m_geometry(std::move(geometry)), m_vertices(std::move(vertices)) {}

  public:
	template <typename State, typename Processor, typename Layout, typename Metadata, typename Vertex, typename Fragment, typename Requirements>
	[[nodiscard]] static rsx_draw_packet capture(const State& state, const Processor& processor, const Layout& layout,
												 const Metadata& metadata, const Vertex& vertex, const Fragment& fragment, std::uint32_t fragment_texture_mask,
												 provoking_vertex provoking, Requirements&& requirements, draw_capture_limits limits = {})
	{
		auto registers = rsx_draw_snapshot::capture(state, vertex, fragment, fragment_texture_mask,
													limits.fragment_bytes, metadata.referenced_textures_mask, limits.constants_are_current_pass);
		auto environment = draw_environment_snapshot::capture(state);
		auto geometry = capture_draw_geometry(state, processor, layout, provoking, limits.geometry);
		if (limits.sparse_sources)
		{
			auto vertices = vertex_data_snapshot::capture_sparse(limits.sparse_sources(geometry.streams),state,
				geometry.streams,limits.vertex_stream_bytes,limits.constants);
			return {std::move(registers),std::move(environment),std::move(geometry.geometry),std::move(vertices)};
		}
		// Explicit reference adapter for old callers without raw source access.
		// Productive RSX frontend always supplies sparse_sources below.
		if (!geometry.geometry.vertices.empty())
		{
			const auto high = geometry.geometry.source_range->second;
			if (high == 0xffffffffu) throw std::length_error("Native legacy vertex prefix cannot represent UINT_MAX");
			geometry.streams = {0,0,high + 1};
		}
		const auto sizes = geometry.geometry.vertices.empty() ? std::pair<std::uint32_t, std::uint32_t>{0, 0} : std::invoke(std::forward<Requirements>(requirements), layout, geometry.streams.stream_first, geometry.streams.vertex_count);
		auto vertices = vertex_data_snapshot::capture(processor, layout, metadata, state, geometry.streams, sizes, limits.vertex_stream_bytes, limits.constants);
		return {std::move(registers), std::move(environment), std::move(geometry.geometry), std::move(vertices)};
	}
	[[nodiscard]] const rsx_draw_snapshot& state() const { return m_state; }
	[[nodiscard]] const draw_environment_snapshot& environment() const { return m_environment; }
	[[nodiscard]] const converted_geometry& geometry() const { return m_geometry; }
	[[nodiscard]] const vertex_data_snapshot& vertices() const { return m_vertices; }
};
} // namespace metal4
