#pragma once

#include "M4FragmentMSL.h"
#include "M4DrawValidation.h"
#include "M4VertexProgram.h"

#include <cstring>
#include <stdexcept>

namespace metal4
{
enum class framebuffer_extent_policy
{
	exact,
	clip_to_framebuffer
};
class draw_environment_snapshot final
{
	std::uint32_t m_width = 0, m_height = 0;
	vertex_context m_vertex;
	fragment_context m_fragment;
	bool m_depth_bounds = false;
	bool m_depth_clamp = false;
	float m_polygon_bias = 0, m_polygon_slope = 0;
	bool m_msaa = false, m_alpha_to_coverage = false, m_alpha_to_one = false;
	float m_depth_min = 0, m_depth_max = 1;
	draw_environment_snapshot() = default;

  public:
	// Caller owns a coherent analysed register epoch. Captures values and the
	// stipple words, without retaining state/processor pointers. Uses 1:1 guest
	// pixels; resolution scaling needs a coordinated framebuffer/scissor policy.
	template <typename State>
	[[nodiscard]] static draw_environment_snapshot capture(const State& state)
	{
		draw_environment_snapshot result;
		result.m_width = state.surface_clip_width();
		result.m_height = state.surface_clip_height();
		if (!result.m_width || !result.m_height)
			throw std::invalid_argument("Native RSX environment has empty clip extent");
		const float half_width = result.m_width / 2.f, half_height = result.m_height / 2.f;
		const float sx = state.viewport_scale_x() / half_width;
		const float sy = state.viewport_scale_y() / half_height;
		const float ox = (state.viewport_offset_x() - half_width) / half_width;
		const float oy = (state.viewport_offset_y() - half_height) / half_height;
		// Row-vector matrix ABI. Metal maps positive NDC Y upward, whereas
		// the guest viewport maps it downward; negate scale AND offset.
		result.m_vertex.scale_offset = {sx, 0, 0, ox, 0, -sy, 0, -oy,
										0, 0, state.viewport_scale_z(), state.viewport_offset_z(), 0, 0, 0, 1};
		result.m_vertex.branch_bits = state.transform_branch_bits();
		result.m_vertex.point_size = state.point_size();
		result.m_vertex.z_near = state.clip_min();
		result.m_vertex.z_far = state.clip_max();
		const std::uint32_t clip = state.clip_planes_mask();
		for (unsigned i = 0; i < 6; ++i)
		{
			const auto op = (clip >> (4 * i)) & 15;
			if (op > 2)
				throw std::invalid_argument("Native RSX user clip operation is invalid");
			if (op)
			{
				result.m_vertex.clip_enable |= 1u << i;
				result.m_vertex.clip_factors[i] = op == 1 ? -1.f : 1.f;
			}
		}
		const auto alpha = static_cast<std::uint32_t>(state.alpha_func());
		if (alpha < 0x200 || alpha > 0x207)
			throw std::invalid_argument("Native RSX alpha function is invalid");
		result.m_fragment.alpha_function = alpha - 0x200;
		result.m_fragment.alpha_reference = state.alpha_ref();
		const auto fog = static_cast<std::uint32_t>(state.fog_equation());
		if (fog == 0x2601)
			result.m_fragment.fog_mode = 1;
		else if (fog >= 0x800 && fog <= 0x804)
			result.m_fragment.fog_mode = fog - 0x800 + 2;
		else
			throw std::invalid_argument("Native RSX fog equation is invalid");
		result.m_fragment.fog_param0 = state.fog_params_0();
		result.m_fragment.fog_param1 = state.fog_params_1();
		const auto origin = static_cast<std::uint32_t>(state.shader_window_origin());
		const auto center = static_cast<std::uint32_t>(state.pixel_center());
		if (origin > 1 || center > 1)
			throw std::invalid_argument("Native RSX shader window is invalid");
		result.m_fragment.wpos_scale = origin ? -1.f : 1.f;
		result.m_fragment.wpos_bias_x = center ? -.5f : 0;
		result.m_fragment.wpos_bias_y = (origin ? static_cast<float>(state.shader_window_height()) : 0.f) - (center ? .5f : 0.f);
		result.m_fragment.derivative_sign_y = 1;
		result.m_msaa = state.msaa_enabled();
		// VK supplies the sample mask independently of the MSAA-enable hint.
		result.m_fragment.sample_mask = state.msaa_sample_mask();
		result.m_alpha_to_coverage = state.msaa_alpha_to_coverage_enabled();
		result.m_alpha_to_one = state.msaa_alpha_to_one_enabled();
		result.m_depth_clamp = state.depth_clamp_enabled() || !state.depth_clip_enabled();
		if (state.poly_offset_fill_enabled())
		{
			result.m_polygon_bias = state.poly_offset_bias();
			result.m_polygon_slope = state.poly_offset_scale();
			if (!std::isfinite(result.m_polygon_bias) || !std::isfinite(result.m_polygon_slope))
				throw std::invalid_argument("Native RSX polygon offset is not finite");
		}
		result.m_depth_bounds = state.depth_bounds_test_enabled();
		if (result.m_depth_bounds)
		{
			result.m_depth_min = state.depth_bounds_min();
			result.m_depth_max = state.depth_bounds_max();
			validate_depth_bounds(result.m_depth_min, result.m_depth_max);
		}
		const auto pattern = state.polygon_stipple_pattern();
		if (!pattern)
			throw std::invalid_argument("Native RSX polygon stipple storage is absent");
		std::memcpy(result.m_fragment.stipple_pattern.data(), pattern, 128);
		return result;
	}
	[[nodiscard]] std::uint32_t width() const { return m_width; }
	[[nodiscard]] std::uint32_t height() const { return m_height; }
	[[nodiscard]] bool msaa_enabled() const { return m_msaa; }
	[[nodiscard]] bool alpha_to_coverage() const { return m_alpha_to_coverage; }
	[[nodiscard]] bool alpha_to_one() const { return m_alpha_to_one; }
	[[nodiscard]] bool depth_bounds_enabled() const { return m_depth_bounds; }
	[[nodiscard]] bool depth_clamp_enabled() const { return m_depth_clamp; }
	[[nodiscard]] float polygon_bias() const { return m_polygon_bias; }
	[[nodiscard]] float polygon_slope() const { return m_polygon_slope; }
	[[nodiscard]] float depth_min() const { return m_depth_min; }
	[[nodiscard]] float depth_max() const { return m_depth_max; }
	void validate_framebuffer_extent(std::size_t width, std::size_t height, framebuffer_extent_policy policy = framebuffer_extent_policy::exact) const
	{
		if (!width || !height || (policy != framebuffer_extent_policy::exact && policy != framebuffer_extent_policy::clip_to_framebuffer) ||
			width > m_width || height > m_height ||
			(policy == framebuffer_extent_policy::exact && (width != m_width || height != m_height)))
			throw std::invalid_argument("Native environment/framebuffer needs coordinated scaling or valid crop policy");
	}
	void apply(vertex_context& vertex, fragment_context& fragment) const
	{
		vertex.scale_offset = m_vertex.scale_offset;
		vertex.branch_bits = m_vertex.branch_bits;
		vertex.point_size = m_vertex.point_size;
		vertex.z_near = m_vertex.z_near;
		vertex.z_far = m_vertex.z_far;
		vertex.clip_enable = m_vertex.clip_enable;
		vertex.clip_factors = m_vertex.clip_factors;
		fragment.alpha_function = m_fragment.alpha_function;
		fragment.alpha_reference = m_fragment.alpha_reference;
		fragment.fog_mode = m_fragment.fog_mode;
		fragment.fog_param0 = m_fragment.fog_param0;
		fragment.fog_param1 = m_fragment.fog_param1;
		fragment.wpos_scale = m_fragment.wpos_scale;
		fragment.wpos_bias_x = m_fragment.wpos_bias_x;
		fragment.wpos_bias_y = m_fragment.wpos_bias_y;
		fragment.derivative_sign_y = m_fragment.derivative_sign_y;
		fragment.sample_mask = m_fragment.sample_mask;
		fragment.stipple_pattern = m_fragment.stipple_pattern;
	}
};
} // namespace metal4
