#pragma once

#include "M4PipelineSnapshot.h"
#include "M4ShaderProgramSnapshot.h"
#include "M4TextureSnapshot.h"
#include "M4VertexTextureState.h"

namespace metal4
{
// All capture inputs must belong to one analysed draw epoch, with method/VM
// writes excluded throughout this call. Owns program bytes and register words;
// resolved framebuffer/fetch/constants and actual VM hooks are supplied later.
class rsx_draw_snapshot final
{
	pipeline_register_snapshot m_pipeline;
	fragment_texture_snapshot m_textures;
	vertex_texture_snapshot m_vertex_textures;
	vertex_program_snapshot m_vertex;
	fragment_program_snapshot m_fragment;

  public:
	template <typename State, typename Vertex, typename Fragment>
	[[nodiscard]] static rsx_draw_snapshot capture(const State& state, const Vertex& vertex, const Fragment& fragment,
												   std::uint32_t texture_mask, std::size_t maximum_fragment_bytes, std::uint32_t vertex_texture_mask = 0,
												   bool constants_are_current_pass = false)
	{
		rsx_draw_snapshot result;
		result.m_pipeline = capture_pipeline_register_snapshot(state);
		result.m_textures = capture_fragment_texture_snapshot(state, texture_mask);
		result.m_vertex_textures = vertex_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), vertex_texture_mask);
		result.m_vertex = vertex_program_snapshot::capture(vertex, constants_are_current_pass);
		result.m_fragment = fragment_program_snapshot::capture(fragment, maximum_fragment_bytes);
		return result;
	}
	[[nodiscard]] const pipeline_register_snapshot& pipeline() const noexcept { return m_pipeline; }
	[[nodiscard]] const fragment_texture_snapshot& textures() const noexcept { return m_textures; }
	[[nodiscard]] const vertex_texture_snapshot& vertex_textures() const noexcept { return m_vertex_textures; }
	[[nodiscard]] const vertex_program_snapshot& vertex() const noexcept { return m_vertex; }
	[[nodiscard]] const fragment_program_snapshot& fragment() const noexcept { return m_fragment; }
};
} // namespace metal4
