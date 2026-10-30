#pragma once

#include <array>
#include <cstdint>
#include "M4OutputROP.h"

namespace metal4
{

// Raw numeric values are the RSX/GCM enum values, with no Vulkan dependency.
// color write masks below use R=1,G=2,B=4,A=8 after guest format remapping.
struct guest_blend_state
{
	bool enabled = false;
	std::uint32_t source_rgb = 1, destination_rgb = 0;
	std::uint32_t source_alpha = 1, destination_alpha = 0;
	std::uint32_t operation_rgb = 0x8006, operation_alpha = 0x8006;
	std::uint32_t write_mask = 15;
	bool operator==(const guest_blend_state&) const = default;
};

struct guest_stencil_state
{
	std::uint32_t compare = 0x207;
	std::uint32_t fail = 0x1e00, depth_fail = 0x1e00, pass = 0x1e00;
	std::uint32_t read_mask = 255, write_mask = 255, reference = 0;
	bool operator==(const guest_stencil_state&) const = default;
};

struct guest_render_state
{
	std::array<guest_blend_state, 4> colors{};
	bool depth_test = false, depth_write = false;
	std::uint32_t depth_compare = 0x201;
	bool stencil_test = false, two_sided_stencil = false;
	guest_stencil_state front, back;
	bool logic_enabled = false;
	std::uint32_t logic_operation = 0x1503;
	std::uint32_t sample_count = 1;
	bool alpha_to_coverage = false, alpha_to_one = false;
	output_rop_configuration output_rop;
	bool operator==(const guest_render_state&) const = default;
};

enum class compare_operation { never, less, equal, less_equal, greater, not_equal, greater_equal, always };
enum class stencil_operation { keep, zero, replace, increment_clamp, decrement_clamp, invert, increment_wrap, decrement_wrap };
enum class blend_factor
{
	zero, one, source_color, one_minus_source_color, source_alpha, one_minus_source_alpha,
	destination_color, one_minus_destination_color, destination_alpha, one_minus_destination_alpha,
	source_alpha_saturated, constant_color, one_minus_constant_color, constant_alpha, one_minus_constant_alpha
};
enum class blend_operation { add, subtract, reverse_subtract, minimum, maximum, reverse_subtract_signed, add_signed, reverse_add_signed };

struct blend_state
{
	bool enabled = false;
	blend_factor source_rgb = blend_factor::one, destination_rgb = blend_factor::zero;
	blend_factor source_alpha = blend_factor::one, destination_alpha = blend_factor::zero;
	blend_operation operation_rgb = blend_operation::add, operation_alpha = blend_operation::add;
	std::uint32_t write_mask = 15;
	bool operator==(const blend_state&) const = default;
};

struct stencil_state
{
	compare_operation compare = compare_operation::always;
	stencil_operation fail = stencil_operation::keep, depth_fail = stencil_operation::keep, pass = stencil_operation::keep;
	std::uint32_t read_mask = 255, write_mask = 0, reference = 0;
	bool operator==(const stencil_state&) const = default;
};

struct render_state
{
	std::array<blend_state, 4> colors{};
	compare_operation depth_compare = compare_operation::always;
	bool depth_write = false, stencil_test = false;
	stencil_state front, back;
	bool logic_enabled = false;
	std::uint32_t logic_operation = 0x1503;
	std::uint32_t sample_count = 1;
	bool alpha_to_coverage = false, alpha_to_one = false;
	// Signed equations and logic ops have no fixed-function Metal equivalent.
	// The native descriptor factory rejects this state until a matching shader
	// ROP implementation is supplied; it never silently substitutes FUNC_ADD.
	bool requires_shader_rop = false;
	bool operator==(const render_state&) const = default;
};

[[nodiscard]] compare_operation decode_compare(std::uint32_t value, bool reverse = false);
[[nodiscard]] stencil_operation decode_stencil(std::uint32_t value);
[[nodiscard]] blend_factor decode_blend_factor(std::uint32_t value);
[[nodiscard]] blend_operation decode_blend_operation(std::uint32_t value);
[[nodiscard]] render_state decode_render_state(const guest_render_state& guest);

}
