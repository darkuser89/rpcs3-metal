#pragma once

#include "M4RenderPipeline.h"
#include "M4ShaderProgramSnapshot.h"

namespace metal4
{
// Import program-derived settings before configuring framebuffer ROPs and
// resolving textures/mirrors. Preserve runtime fetch/constants/viewport/fog
// uniforms; replace only fields owned by analysed program metadata.
inline void apply_shader_program_snapshots(const vertex_program_snapshot& vertex, const fragment_program_snapshot& fragment,
										   bool point_output, bool allow_output_rounding, render_pipeline_configuration& pipeline,
										   vertex_program_options& vertex_options, fragment_program_options& fragment_options,
										   vertex_context& vertex_uniforms, fragment_context& fragment_uniforms)
{
	auto vp = vertex.options(point_output);
	auto fp = fragment.options(allow_output_rounding);
	if (vp.flat_colors != fp.flat_colors)
		throw std::invalid_argument("Analysed RSX vertex/fragment interpolation metadata disagrees");
	auto config = pipeline;
	auto vc = vertex_uniforms;
	auto fc = fragment_uniforms;
	vertex.apply(vc);
	fragment.apply(vc, fc);
	config.state.output_rop = fp.output_rop;
	pipeline = std::move(config);
	vertex_options = std::move(vp);
	fragment_options = std::move(fp);
	vertex_uniforms = std::move(vc);
	fragment_uniforms = std::move(fc);
}
} // namespace metal4
