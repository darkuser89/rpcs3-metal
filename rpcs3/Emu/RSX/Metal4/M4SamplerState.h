#pragma once

#include "M4VertexProgram.h"

namespace metal4
{
// Semantic RSX states: the register adapter must decode guest enum words.
enum class sampler_wrap
{
	repeat,
	mirror,
	edge,
	border,
	clamp,
	mirror_once_edge,
	mirror_once_border,
	mirror_once_clamp
};
enum class sampler_min_filter
{
	nearest,
	linear,
	nearest_nearest,
	linear_nearest,
	nearest_linear,
	linear_linear,
	convolution
};
enum class sampler_mag_filter
{
	nearest,
	linear,
	convolution
};
enum class sampler_stage
{
	fragment,
	vertex
};

struct guest_sampler_state
{
	std::array<sampler_wrap, 3> wrap{sampler_wrap::repeat, sampler_wrap::repeat, sampler_wrap::repeat};
	sampler_min_filter min_filter = sampler_min_filter::nearest;
	sampler_mag_filter mag_filter = sampler_mag_filter::nearest;
	std::uint32_t anisotropy = 1;
	float min_lod = 0, max_lod = 0, bias = 0;
	// Already in native texture-view channel/colorspace order. RSX remap and
	// signed border conversion belong to the texture/register adapter.
	std::array<float, 4> border_color{0, 0, 0, 1};
	bool normalized = true, depth_compare = false, depth_float = false;
	std::uint32_t compare_function = 0x207;
};
struct sampler_texture_policy
{
	sampler_stage stage = sampler_stage::fragment;
	vertex_texture_dimension dimension = vertex_texture_dimension::two;
	std::uint32_t requested_mips = 1, actual_mips = 1;
	bool linear_filtering = true;
};
} // namespace metal4
