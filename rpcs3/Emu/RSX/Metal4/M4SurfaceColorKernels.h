#pragma once

#include <string_view>

namespace metal4
{
inline constexpr std::string_view surface_color_kernels = R"msl(
#include <metal_stdlib>
using namespace metal;
struct surface_color_parameters
{
	ulong source_offset, destination_offset;
	ulong source_pitch, destination_pitch;
	ulong width, pixel_count, first_pixel;
	uint pixel_bytes, element_bytes;
};
kernel void rpcs3_surface_color_bytes(device const uchar* source [[buffer(0)]],
	device uchar* destination [[buffer(1)]], constant surface_color_parameters& p [[buffer(2)]],
	uint thread_id [[thread_position_in_grid]])
{
	if (ulong(thread_id) >= p.pixel_count) return;
	const ulong pixel = p.first_pixel + thread_id;
	const ulong x = pixel % p.width, y = pixel / p.width;
	const ulong input = p.source_offset + y * p.source_pitch + x * p.pixel_bytes;
	const ulong output = p.destination_offset + y * p.destination_pitch + x * p.pixel_bytes;
	for (uint byte = 0; byte < p.pixel_bytes; ++byte)
		destination[output + byte] = source[input + (byte ^ (p.element_bytes - 1))];
}
)msl";
}
