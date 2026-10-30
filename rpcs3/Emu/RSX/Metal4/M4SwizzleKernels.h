#pragma once
#include <string_view>

namespace metal4
{
inline constexpr std::string_view swizzle_kernels = R"msl(
#include <metal_stdlib>
using namespace metal;
struct swizzle_parameters
{
	ulong origin[3], size[3];
	ulong swizzled_offset, linear_offset, row_pitch, image_pitch;
	ulong count, first;
	uint log2[3], block_bytes, swap_width, to_swizzled;
};
ulong rsx_morton(ulong3 position, constant uint* dimensions)
{
	ulong offset = 0;
	uint output_bit = 0;
	const uint count = dimensions[0] + dimensions[1] + dimensions[2];
	for (uint input_bit = 0; output_bit < count; ++input_bit)
		for (uint axis = 0; axis < 3; ++axis)
			if (input_bit < dimensions[axis]) offset |= ((position[axis] >> input_bit) & 1ul) << output_bit++;
	return offset;
}
kernel void rpcs3_swizzle(device uchar* swizzled [[buffer(0)]], device uchar* linear [[buffer(1)]],
	constant swizzle_parameters& p [[buffer(2)]], uint thread_id [[thread_position_in_grid]])
{
	if (ulong(thread_id) >= p.count) return;
	const ulong ordinal = p.first + thread_id;
	const ulong x = ordinal % p.size[0], y = (ordinal / p.size[0]) % p.size[1], z = ordinal / (p.size[0] * p.size[1]);
	const ulong a = p.swizzled_offset + rsx_morton(ulong3(x + p.origin[0], y + p.origin[1], z + p.origin[2]), p.log2) * p.block_bytes;
	const ulong b = p.linear_offset + z * p.image_pitch + y * p.row_pitch + x * p.block_bytes;
	for (uint byte = 0; byte < p.block_bytes; ++byte)
	{
		if (p.to_swizzled) swizzled[a + byte] = linear[b + (byte ^ (p.swap_width - 1))];
		else linear[b + byte] = swizzled[a + (byte ^ (p.swap_width - 1))];
	}
}
)msl";
}
