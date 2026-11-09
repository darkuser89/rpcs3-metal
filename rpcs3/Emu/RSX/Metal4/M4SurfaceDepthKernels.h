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
uint rsx_depth_expand(uint raw, uint format)
{
	if (format == 2) return as_type<uint>(float(raw >> 8) / 16777215.0f);
	if (format == 3) return (raw << 11) + (120u << 23);
	if (format == 4) return (raw >> 8) << 7;
	return raw;
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
		uint raw = rsx_depth_load(guest, g, guest_width, true), value = rsx_depth_expand(raw, p.format);
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
kernel void rpcs3_surface_depth_patch(device const uchar* guest [[buffer(0)]], device uchar* depth [[buffer(1)]],
	device uchar* stencil [[buffer(2)]], constant depth_parameters& p [[buffer(3)]],
	device const uchar* mask [[buffer(4)]], uint thread_id [[thread_position_in_grid]])
{
	if (ulong(thread_id) >= p.count) return;
	const ulong pixel = p.first + thread_id, x = pixel % p.width, y = pixel / p.width;
	const bool has_stencil = p.format == 2 || p.format == 4;
	const uint guest_width = has_stencil ? 4 : 2, depth_width = p.format == 1 ? 2 : 4;
	const ulong g = p.guest_offset + y * p.guest_pitch + x * guest_width;
	bool depth_write = false;
	for (uint lane = 0; lane < guest_width - uint(has_stencil); ++lane)
		depth_write = depth_write || mask[g + lane] != 0;
	const uint raw = rsx_depth_load(guest, g, guest_width, true);
	if (depth_write)
	{
		uint value = rsx_depth_expand(raw, p.format);
		if (p.format == 2)
		{
			// Keep a patch word in its exact integer gather interval if the
			// division rounds outside it. Ordinary upload keeps its existing
			// rational/truncation contract.
			const uint integer = raw >> 8;
			const uint contracted = uint(as_type<float>(value) * 16777215.0f);
			if (contracted < integer) ++value;
			else if (contracted > integer) --value;
		}
		rsx_depth_store(depth, p.depth_offset + y * p.depth_pitch + x * depth_width, depth_width, value, false);
	}
	if (has_stencil && mask[g + guest_width - 1])
		stencil[p.stencil_offset + y * p.stencil_pitch + x] = uchar(raw);
}
)msl";
}
