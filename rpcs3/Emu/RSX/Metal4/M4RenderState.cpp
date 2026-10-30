#include "M4RenderState.h"

#include <stdexcept>

namespace metal4
{
compare_operation decode_compare(std::uint32_t value, bool reverse)
{
	// The shared RSX enum adapter accepts both low three-bit and GL encodings.
	value |= 0x200;
	if (value < 0x200 || value > 0x207) throw std::invalid_argument("Invalid RSX comparison function");
	if (reverse)
	{
		if (value == 0x201) value = 0x204;
		else if (value == 0x204) value = 0x201;
		else if (value == 0x203) value = 0x206;
		else if (value == 0x206) value = 0x203;
	}
	return static_cast<compare_operation>(value - 0x200);
}

stencil_operation decode_stencil(std::uint32_t value)
{
	switch (value)
	{
	case 0x1e00: return stencil_operation::keep;
	case 0: return stencil_operation::zero;
	case 0x1e01: return stencil_operation::replace;
	case 0x1e02: return stencil_operation::increment_clamp;
	case 0x1e03: return stencil_operation::decrement_clamp;
	case 0x150a: return stencil_operation::invert;
	case 0x8507: return stencil_operation::increment_wrap;
	case 0x8508: return stencil_operation::decrement_wrap;
	default: throw std::invalid_argument("Invalid RSX stencil operation");
	}
}

blend_factor decode_blend_factor(std::uint32_t value)
{
	switch (value)
	{
	case 0: return blend_factor::zero;
	case 1: return blend_factor::one;
	case 0x300: return blend_factor::source_color;
	case 0x301: return blend_factor::one_minus_source_color;
	case 0x302: return blend_factor::source_alpha;
	case 0x303: return blend_factor::one_minus_source_alpha;
	case 0x304: return blend_factor::destination_alpha;
	case 0x305: return blend_factor::one_minus_destination_alpha;
	case 0x306: return blend_factor::destination_color;
	case 0x307: return blend_factor::one_minus_destination_color;
	case 0x308: return blend_factor::source_alpha_saturated;
	case 0x8001: return blend_factor::constant_color;
	case 0x8002: return blend_factor::one_minus_constant_color;
	case 0x8003: return blend_factor::constant_alpha;
	case 0x8004: return blend_factor::one_minus_constant_alpha;
	default: throw std::invalid_argument("Invalid RSX blend factor");
	}
}

blend_operation decode_blend_operation(std::uint32_t value)
{
	switch (value)
	{
	case 0x8006: return blend_operation::add;
	case 0x8007: return blend_operation::minimum;
	case 0x8008: return blend_operation::maximum;
	case 0x800a: return blend_operation::subtract;
	case 0x800b: return blend_operation::reverse_subtract;
	case 0xf005: return blend_operation::reverse_subtract_signed;
	case 0xf006: return blend_operation::add_signed;
	case 0xf007: return blend_operation::reverse_add_signed;
	default: throw std::invalid_argument("Invalid RSX blend equation");
	}
}

namespace
{
stencil_state stencil(const guest_stencil_state& guest)
{
	return {decode_compare(guest.compare), decode_stencil(guest.fail), decode_stencil(guest.depth_fail),
		decode_stencil(guest.pass), guest.read_mask & 255, guest.write_mask & 255, guest.reference & 255};
}

bool signed_operation(blend_operation operation)
{
	return operation == blend_operation::reverse_subtract_signed || operation == blend_operation::add_signed ||
		operation == blend_operation::reverse_add_signed;
}
}

render_state decode_render_state(const guest_render_state& guest)
{
	if (guest.sample_count != 1 && guest.sample_count != 2 && guest.sample_count != 4 && guest.sample_count != 8)
		throw std::invalid_argument("Invalid Metal 4 raster sample count");
	render_state result;
	result.sample_count = guest.sample_count;
	result.alpha_to_coverage = guest.alpha_to_coverage;
	result.alpha_to_one = guest.alpha_to_one;
	if (guest.depth_test)
	{
		result.depth_compare = decode_compare(guest.depth_compare);
		result.depth_write = guest.depth_write;
	}
	if (guest.stencil_test)
	{
		result.stencil_test = true;
		result.front = stencil(guest.front);
		result.back = guest.two_sided_stencil ? stencil(guest.back) : result.front;
	}
	if (guest.logic_enabled)
	{
		if (guest.logic_operation < 0x1500 || guest.logic_operation > 0x150f)
			throw std::invalid_argument("Invalid RSX logic operation");
		result.logic_enabled = true;
		result.logic_operation = guest.logic_operation;
		result.requires_shader_rop = true;
	}
	for (std::size_t i = 0; i < result.colors.size(); ++i)
	{
		const auto& source = guest.colors[i];
		auto& target = result.colors[i];
		if (source.write_mask & ~15u) throw std::invalid_argument("Invalid color channel write mask");
		target.write_mask = source.write_mask;
		if (!source.enabled || guest.logic_enabled) continue;
		target.enabled = true;
		target.source_rgb = decode_blend_factor(source.source_rgb);
		target.destination_rgb = decode_blend_factor(source.destination_rgb);
		target.source_alpha = decode_blend_factor(source.source_alpha);
		target.destination_alpha = decode_blend_factor(source.destination_alpha);
		target.operation_rgb = decode_blend_operation(source.operation_rgb);
		target.operation_alpha = decode_blend_operation(source.operation_alpha);
		result.requires_shader_rop |= signed_operation(target.operation_rgb) || signed_operation(target.operation_alpha);
	}
	return result;
}
}
