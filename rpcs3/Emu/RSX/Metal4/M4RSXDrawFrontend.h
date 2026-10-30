#pragma once

#include "M4DrawClauseDriver.h"
#include "M4DrawEnvironment.h"
#include "M4DrawGeometryPlan.h"
#include "M4FramebufferSnapshot.h"
#include "M4RSXDrawPacket.h"
#include "M4RSXDrawSnapshot.h"
#include "M4VertexDataSnapshot.h"
#include "M4RSXVertexSources.h"

#include <type_traits>

namespace metal4
{
// Adapter for a real GSRender/rsx::thread-derived renderer. Its analysed
// programs are protected members,so capture must run from the renderer
// instead of a free function inspecting another thread. Does not instantiate
// a renderer,load programs,acquire VM locks or select the application backend.
template <typename RendererBase>
class rsx_draw_frontend : public RendererBase
{
  protected:
	using RendererBase::RendererBase;
	// prepare(pass,flags) refreshes analysed programs/framebuffer and returns
	// an optional epoch with .framebuffer. The epoch's RAII guard must keep
	// method/VM writers excluded through layout analysis,capture and submit.
	// An empty optional skips that range. Caller owns thread::end()/queries.
	template <typename State, typename Layout, typename Prepare, typename Submit>
	std::uint32_t submit_owned_clause(State& state, Layout& layout, Prepare&& prepare, Submit&& submit,
									  provoking_vertex provoking, draw_capture_limits limits = {},
		clause_cleanup_owner cleanup = clause_cleanup_owner::driver)
	{
		limits.constants_are_current_pass = true;
		const auto emit = [&](std::uint32_t pass, std::uint32_t flags)
		{
			const auto epoch = std::invoke(prepare, pass, flags);
			if (!epoch)
				return;
			this->GRAPH_frontend().analyse_inputs_interleaved(layout, this->current_vp_metadata);
			const auto packet = capture_owned_pass(state, layout, provoking, limits);
			if constexpr (std::is_invocable_v<Submit, const rsx_draw_packet&, decltype((epoch->framebuffer)), std::uint32_t, decltype(*epoch)>)
				std::invoke(submit, packet, epoch->framebuffer, pass, *epoch);
			else
				std::invoke(submit, packet, epoch->framebuffer, pass);
		};
		// rsx::vertex_arrays_changed=4;verified against the real clause enum.
		return drive_draw_clause(state.current_draw_clause, this->m_ctx, emit, 4, cleanup);
	}
	template <typename Layout>
	[[nodiscard]] framebuffer_snapshot capture_owned_framebuffer(const Layout& layout, bool validated,
																 std::optional<std::array<std::size_t, 3>> swizzle_storage = {}) const
	{
		return framebuffer_snapshot::capture(layout, validated, swizzle_storage);
	}
	template <typename State, typename Layout>
	[[nodiscard]] rsx_draw_packet capture_owned_pass(const State& state, const Layout& layout,
													 provoking_vertex provoking, draw_capture_limits limits = {})
	{
		if constexpr (requires (Layout& value) { value.alloc_interleaved_block(); })
			limits.sparse_sources = [&](const vertex_stream_range& range)
			{
				return rsx_vertex_sources(this->GRAPH_frontend(),layout,this->current_vp_metadata,state,range);
			};
		const auto requirements = [&](const auto& input_layout, std::uint32_t first, std::uint32_t count)
		{
			return this->calculate_memory_requirements(input_layout, first, count);
		};
		return rsx_draw_packet::capture(state, this->GRAPH_frontend(), layout, this->current_vp_metadata,
										this->current_vertex_program, this->current_fragment_program, this->current_fp_metadata.referenced_textures_mask,
										provoking, requirements, limits);
	}
	template <typename State>
	[[nodiscard]] draw_environment_snapshot capture_owned_environment(const State& state) const
	{
		return draw_environment_snapshot::capture(state);
	}
	template <typename State, typename Layout>
	[[nodiscard]] draw_geometry_plan capture_owned_geometry(const State& state, const Layout& layout,
															provoking_vertex provoking, geometry_plan_limits limits = {})
	{
		return capture_draw_geometry(state, this->GRAPH_frontend(), layout, provoking, limits);
	}
	template <typename State>
	[[nodiscard]] rsx_draw_snapshot capture_owned_draw(const State& state, std::size_t fragment_limit) const
	{
		return rsx_draw_snapshot::capture(state, this->current_vertex_program, this->current_fragment_program,
										  this->current_fp_metadata.referenced_textures_mask, fragment_limit, this->current_vp_metadata.referenced_textures_mask);
	}
	template <typename State, typename Layout>
	[[nodiscard]] vertex_data_snapshot capture_owned_vertex_data(const State& state, const Layout& layout,
																 vertex_stream_range range, std::size_t byte_limit)
	{
		if (range.source_vertices)
			return vertex_data_snapshot::capture_sparse(rsx_vertex_sources(this->GRAPH_frontend(),layout,this->current_vp_metadata,state,range),state,range,byte_limit);
		const auto sizes = this->calculate_memory_requirements(layout, range.stream_first, range.vertex_count);
		return vertex_data_snapshot::capture(this->GRAPH_frontend(), layout, this->current_vp_metadata,
											 state, range, sizes, byte_limit);
	}
};
} // namespace metal4
