#pragma once

#include "M4FragmentProgram.h"
#include "M4VertexProgram.h"
#include "M4SurfaceLayout.h"
#include "M4LogicOp.h"
#include "M4BlendROP.h"
#include "M4DepthBounds.h"
#include "M4DepthSampling.h"

#include <string>

namespace metal4
{

struct fragment_program_options
{
	bool flat_colors = false;
	bool float32_exports = true;
	bool depth_export = false;
	std::uint32_t color_count = 1;
	std::array<color_export_remap, 4> color_remap{};
	logic_rop_configuration logic_rop;
	blend_rop_configuration blend_rop;
	output_rop_configuration output_rop;
	std::uint32_t half_storage_mask = 0;
	// VK forces shading rate 1.0: native MSAA entries must run per sample even
	// when no programmable blend/logic attachment read is required.
	std::uint32_t raster_sample_count = 1;
	depth_mirror_configuration depth_mirror;
	std::array<vertex_texture_dimension, 16> textures = []
	{
		std::array<vertex_texture_dimension, 16> result;
		result.fill(vertex_texture_dimension::two);
		return result;
	}();
	// Native depth comparison: height-one 1D,2D and cube use dedicated MSL
	// depth types and matching sampler metadata. Redirected/MSAA depth and
	// 3D shadow textures still require a separate texture emulation layer.
	std::uint32_t border_textures = 0;
	std::uint32_t shadow_textures = 0;
	// Ordinary 3D depth reads use RG32Float volume R;3D shadows remain explicit
	// emulation work. Ordinary depth reads are distinct from shadow comparisons and redirected
	// packed depth/stencil emulation. Compiled view remaps precede texel controls.
	std::uint32_t depth_textures = 0;
	std::array<depth_sample_remap, 16> depth_remap{};
	std::uint32_t redirected_textures = 0;
	std::uint32_t multisampled_textures = 0;
	bool operator==(const fragment_program_options&) const = default;
};

// CPU/MSL ABI, fragment buffer(2). Mutable inline float4 constants are densely
// packed in fragment buffer(1), in generated constant_offsets order. Values
// are decoded from RSX bytes before upload. One draw uses one context.
struct alignas(16) fragment_context
{
	std::uint32_t constant_count = 0;
	std::uint32_t instruction_budget = 65536;
	std::uint32_t texcoord_2d_mask = 0;
	std::uint32_t point_sprite_mask = 0;
	std::uint32_t two_sided = 0;
	std::uint32_t alpha_test = 0;
	std::uint32_t alpha_function = 7;
	std::uint32_t predicate = 1;
	float alpha_reference = 0;
	float wpos_scale = 1;
	float wpos_bias_x = 0;
	float wpos_bias_y = 0;
	float derivative_sign_y = 1;
	float fog_param0 = 0;
	float fog_param1 = 0;
	// Native enum: 0 raw, 1 linear, 2 exp, 3 exp2, 4 exp_abs,
	// 5 exp2_abs, 6 linear_abs. The guest adapter must translate RSX enums.
	std::uint32_t fog_mode = 0;
	std::uint32_t sample_mask = 0xffffffffu;
	// Depth bounds belong to raster_state: they compare stored depth, which a
	// fragment shader cannot replace by testing its own newly computed depth.
	std::array<std::uint32_t, 3> reserved{};
	// Programmable blending reads these completion-owned values. Fixed-function
	// blending continues to use raster_state::blend_color in the encoder.
	std::array<float, 4> blend_color{};
	// 32 framebuffer rows, bit zero = first pixel. Host words are copied into
	// completion-owned uniforms; callers translate guest endian register data.
	std::array<std::uint32_t, 32> stipple_pattern = []
	{
		std::array<std::uint32_t, 32> rows;
		rows.fill(0xffffffffu);
		return rows;
	}();
	// Only a prepared native mirror pass may populate these from raster_state.
	std::uint32_t depth_bounds_enabled = 0;
	float depth_min = 0, depth_max = 1;
	std::uint32_t depth_reserved = 0;
};
static_assert(sizeof(fragment_context) == 240 && offsetof(fragment_context, alpha_reference) == 32 &&
			  offsetof(fragment_context, sample_mask) == 64 && offsetof(fragment_context, blend_color) == 80 && offsetof(fragment_context, stipple_pattern) == 96 && offsetof(fragment_context, depth_bounds_enabled) == 224);

struct generated_fragment_program
{
	std::string source;
	std::vector<std::size_t> constant_offsets;
	std::uint32_t texture_mask = 0;
	std::uint32_t input_mask = 0;
	bool has_kill_instruction = false;
	// The generated entry can additionally discard on dynamic predicate,
	// alpha tests and runtime faults, even without a KIL instruction.
};

// Native MSL4, no GLSL/SPIR-V/Vulkan translation stage. Physical R/H aliases,
// mutable literal slots and structured flow addresses are preserved. Unknown
// instructions, invalid flow regions and unsupported sampling modes fail on
// the CPU. Full RSX texture/ROP emulation and hardware float parity are open.
[[nodiscard]] generated_fragment_program generate_fragment_program(std::span<const std::byte> bytes,
																   const fragment_program_options& options = {});

} // namespace metal4
