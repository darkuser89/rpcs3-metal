#pragma once

#include <array>
#include <cstdint>

namespace metal4::test
{
struct environment_state
{
	std::uint32_t width = 4, height = 4, clip = 0, branch = 0, alpha = 0x207, fog = 0x2601, origin = 0, center = 0, window_height = 4, samples = 65535;
	std::array<float, 3> scale{2, 2, 1}, offset{2, 2, 0};
	float point = 1, near = 0, far = 1, reference = 0, fog0 = 0, fog1 = 0;
	bool multisample = false, null_pattern = false;
	bool alpha_coverage = false, alpha_one = false;
	bool bounds = false;
	float bounds_min = 0, bounds_max = 1;
	bool depth_clamp = false, depth_clip = true, offset_fill = false;
	float polygon_bias = 0, polygon_slope = 0;
	std::array<std::uint32_t, 32> pattern = []
	{
		std::array<std::uint32_t, 32> result;
		result.fill(0xffffffffu);
		return result;
	}();
	auto surface_clip_width() const { return width; }
	auto surface_clip_height() const { return height; }
	auto viewport_scale_x() const { return scale[0]; }
	auto viewport_scale_y() const { return scale[1]; }
	auto viewport_scale_z() const { return scale[2]; }
	auto viewport_offset_x() const { return offset[0]; }
	auto viewport_offset_y() const { return offset[1]; }
	auto viewport_offset_z() const { return offset[2]; }
	auto transform_branch_bits() const { return branch; }
	auto point_size() const { return point; }
	auto clip_min() const { return near; }
	auto clip_max() const { return far; }
	auto clip_planes_mask() const { return clip; }
	auto alpha_func() const { return alpha; }
	auto alpha_ref() const { return reference; }
	auto fog_equation() const { return fog; }
	auto fog_params_0() const { return fog0; }
	auto fog_params_1() const { return fog1; }
	auto shader_window_origin() const { return origin; }
	auto pixel_center() const { return center; }
	auto shader_window_height() const { return window_height; }
	auto msaa_enabled() const { return multisample; }
	auto msaa_sample_mask() const { return samples; }
	auto msaa_alpha_to_coverage_enabled() const { return alpha_coverage; }
	auto msaa_alpha_to_one_enabled() const { return alpha_one; }
	auto depth_bounds_test_enabled() const { return bounds; }
	auto depth_bounds_min() const { return bounds_min; }
	auto depth_bounds_max() const { return bounds_max; }
	auto depth_clamp_enabled() const { return depth_clamp; }
	auto depth_clip_enabled() const { return depth_clip; }
	auto poly_offset_fill_enabled() const { return offset_fill; }
	auto poly_offset_bias() const { return polygon_bias; }
	auto poly_offset_scale() const { return polygon_slope; }
	const void* polygon_stipple_pattern() const { return null_pattern ? nullptr : pattern.data(); }
};
} // namespace metal4::test
