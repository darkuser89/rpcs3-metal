#pragma once
#include <string_view>

namespace metal4
{
inline constexpr std::string_view surface_depth_kernels = R"msl(
#include <metal_stdlib>
using namespace metal;
struct depth_parameters
{
	ulong guest_offset, depth_offset, stencil_offset;
	ulong guest_pitch, depth_pitch, stencil_pitch;
	ulong width, count, first;
	uint format, gather;
};
uint rsx_depth_load(device const uchar* data, ulong offset, uint width, bool big_endian)
{
	uint result = 0;
	for (uint byte = 0; byte < width; ++byte)
		result |= uint(data[offset + byte]) << (8 * (big_endian ? width - 1 - byte : byte));
	return result;
}
void rsx_depth_store(device uchar* data, ulong offset, uint width, uint value, bool big_endian)
{
	for (uint byte = 0; byte < width; ++byte)
		data[offset + byte] = uchar(value >> (8 * (big_endian ? width - 1 - byte : byte)));
}
kernel void rpcs3_surface_depth(device uchar* guest [[buffer(0)]], device uchar* depth [[buffer(1)]],
	device uchar* stencil [[buffer(2)]], constant depth_parameters& p [[buffer(3)]],
	uint thread_id [[thread_position_in_grid]])
{
	if (ulong(thread_id) >= p.count) return;
	const ulong pixel = p.first + thread_id, x = pixel % p.width, y = pixel / p.width;
	const bool has_stencil = p.format == 2 || p.format == 4;
	const uint guest_width = has_stencil ? 4 : 2, depth_width = p.format == 1 ? 2 : 4;
	const ulong g = p.guest_offset + y * p.guest_pitch + x * guest_width;
	const ulong d = p.depth_offset + y * p.depth_pitch + x * depth_width;
	const ulong s = p.stencil_offset + y * p.stencil_pitch + x;
	if (!p.gather)
	{
		uint raw = rsx_depth_load(guest, g, guest_width, true), value = raw;
		if (p.format == 2) value = as_type<uint>(float(raw >> 8) / 16777215.0f);
		else if (p.format == 3) value = (raw << 11) + (120u << 23);
		else if (p.format == 4) value = (raw >> 8) << 7;
		rsx_depth_store(depth, d, depth_width, value, false);
		// One pixel owns one stencil byte, no atomics or prior zero-fill needed.
		if (has_stencil) stencil[s] = uchar(raw);
	}
	else
	{
		uint value = rsx_depth_load(depth, d, depth_width, false);
		if (p.format == 2)
		{
			float normalized = as_type<float>(value);
			if (isnan(normalized)) normalized = 0;
			value = uint(clamp(normalized, 0.0f, 1.0f) * 16777215.0f);
		}
		else if (p.format == 3) value = ((value - (120u << 23)) >> 11) & 65535u;
		else if (p.format == 4) value = value >> 7;
		if (has_stencil) value = (value << 8) | uint(stencil[s]);
		rsx_depth_store(guest, g, guest_width, value, true);
	}
}
)msl";
}
