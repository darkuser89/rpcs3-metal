#pragma once

#include "M4DrawEnvironment.h"
#include "M4DrawExecutor.h"

namespace metal4
{
inline void apply_draw_environment(const draw_environment_snapshot& snapshot,
								   std::size_t framebuffer_width, std::size_t framebuffer_height, draw_request& request,
								   framebuffer_extent_policy policy = framebuffer_extent_policy::exact, std::uint32_t samples = 1)
{
	snapshot.validate_framebuffer_extent(framebuffer_width, framebuffer_height, policy);
	if (samples != 1 && samples != 2 && samples != 4 && samples != 8)
		throw std::invalid_argument("Native environment raster sample count is invalid");
	snapshot.apply(request.vertex_uniforms, request.fragment_uniforms);
	// VK only installs pSampleMask on a genuinely multisampled pipeline.
	if (samples == 1) request.fragment_uniforms.sample_mask = 0xffffffffu;
	// The viewport maps guest clip coordinates;smaller analysed attachments
	// crop those pixels. Shrinking the viewport here would rescale geometry.
	request.raster.viewport = {0, 0, static_cast<double>(snapshot.width()), static_cast<double>(snapshot.height()), 0, 1};
	request.alpha_to_coverage = snapshot.alpha_to_coverage();
	request.alpha_to_one = snapshot.alpha_to_one();
	request.raster.depth_bounds_enabled = snapshot.depth_bounds_enabled();
	request.raster.depth_min = snapshot.depth_min();
	request.raster.depth_max = snapshot.depth_max();
	request.raster.depth_clip = snapshot.depth_clamp_enabled() ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
	// These are native rasterizer units: Metal derives the constant-bias
	// resolution from the actual D16/D32F attachment. Do not apply the VK
	// NVIDIA D24 correction to Apple hardware or to our D32F-backed Z24.
	const bool polygons = request.geometry.topology == list_topology::triangles;
	request.raster.depth_bias = polygons ? snapshot.polygon_bias() : 0;
	request.raster.slope_scale = polygons ? snapshot.polygon_slope() : 0;
	request.raster.bias_clamp = 0;
}
} // namespace metal4
