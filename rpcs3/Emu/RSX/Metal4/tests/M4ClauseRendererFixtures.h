#pragma once

#include "M4DrawPacketFixtures.h"
#include "M4RSXDrawFrontend.h"
#include "M4ShaderSnapshotFixtures.h"

namespace metal4::test
{
struct clause_context
{
	std::vector<unsigned> dependencies;
	unsigned cleanups = 0;
};
struct clause_renderer_base
{
	clause_context context;
	clause_context* m_ctx = &context;
	pass_processor processor;
	pass_metadata current_vp_metadata;
	struct
	{
		unsigned referenced_textures_mask = 0;
	} current_fp_metadata;
	vertex_program current_vertex_program;
	fragment_program current_fragment_program;
	auto& GRAPH_frontend() { return processor; }
	std::pair<std::uint32_t, std::uint32_t> calculate_memory_requirements(const pass_layout&, unsigned first, unsigned count)
	{
		if (first || count != 9)
			throw std::logic_error("Clause renderer prefix changed unexpectedly");
		return {48, 0};
	}
};
struct clause_renderer : rsx_draw_frontend<clause_renderer_base>
{
	using rsx_draw_frontend<clause_renderer_base>::submit_owned_clause;
};
} // namespace metal4::test
