#pragma once

#include "M4FragmentShader.h"
#include "M4PipelineSnapshot.h"
#include "M4SurfaceColor.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace metal4
{
// Logical attachment slots match fragment output slots. Width/height/scissor
// use a 1:1 guest/native pixel domain; viewport and resolution transforms are
// supplied by the framebuffer adapter. Preserve its sample/A2C/output settings.
// Publish all four outputs only after native state/ROP validation succeeds.
inline void apply_pipeline_register_snapshot(const pipeline_register_snapshot& snapshot,
											 const std::array<std::optional<surface_color>, 4>& formats, surface_color register_color_format, std::size_t width, std::size_t height,
											 render_pipeline_configuration& pipeline, fragment_program_options& fragment, raster_state& raster, fragment_context& uniforms, bool force_disable_blending = false)
{
	if (!width || !height)
		throw std::invalid_argument("Native RSX pipeline binding needs framebuffer dimensions");
	auto candidate = pipeline;
	auto shader = fragment;
	auto dynamic = raster;
	auto constants = uniforms;
	auto state = snapshot.render_state(pipeline.state.sample_count, force_disable_blending);
	state.alpha_to_coverage = pipeline.state.alpha_to_coverage;
	state.alpha_to_one = pipeline.state.alpha_to_one;
	state.output_rop = pipeline.state.output_rop;
	candidate.state = state;
	candidate.colors = {};
	candidate.shader_blend_mask = 0;
	candidate.shader_half_storage_mask = 0;
	candidate.color_remap = {};
	shader.color_count = 0;
	shader.half_storage_mask = 0;
	shader.color_remap = {};
	for (unsigned slot = 0; slot < 4; ++slot)
	{
		if (formats[slot])
			configure_surface_color(candidate, shader, slot, *formats[slot], state.colors[slot].write_mask);
		else
		{
			candidate.state.colors[slot].write_mask = 0;
			candidate.state.colors[slot].enabled = false;
		}
	}
	configure_blending(candidate, shader);
	configure_logic_operations(candidate, shader);
	const auto decoded = decode_render_state(candidate.state);
	dynamic.blend_color = snapshot.blend_color(register_color_format == surface_color::rgba16f);
	constants.blend_color = dynamic.blend_color;
	dynamic.front_stencil_reference = decoded.front.reference;
	dynamic.back_stencil_reference = decoded.back.reference;
	dynamic.front_face = snapshot.clockwise() ? MTLWindingClockwise : MTLWindingCounterClockwise;
	const auto cull = snapshot.cull();
	dynamic.cull = cull == pipeline_cull::front ? MTLCullModeFront : cull == pipeline_cull::back ? MTLCullModeBack
																								 : MTLCullModeNone;
	dynamic.cull_all_triangles = cull == pipeline_cull::all;
	const auto scissor = snapshot.scissor();
	const auto x = std::min<std::size_t>(scissor.x, width), y = std::min<std::size_t>(scissor.y, height);
	dynamic.scissor = {x, y, std::min<std::size_t>(scissor.width, width - x), std::min<std::size_t>(scissor.height, height - y)};
	pipeline = std::move(candidate);
	fragment = std::move(shader);
	raster = std::move(dynamic);
	uniforms = std::move(constants);
}
} // namespace metal4
