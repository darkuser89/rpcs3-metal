#pragma once

#include <string_view>

namespace metal4
{
inline constexpr std::string_view vertex_fetch_kernels = R"msl(
#include <metal_stdlib>
using namespace metal;

struct rsx_fetch_parameters
{
	uint2 attributes[16];
	ulong persistent_length;
	ulong transient_length;
	uint first_vertex;
	uint vertex_count;
	uint vertex_base_index;
	int vertex_index_offset;
	uint chunk_base;
	uint chunk_count;
	uint attribute_mask;
	uint reserved;
	uint element_counts[16];
};

uint rsx_vertex_word(device const uchar* stream, ulong offset, uint width, bool big_endian)
{
	uint value = 0;
	for (uint i = 0; i < width; ++i)
		value |= uint(stream[offset + i]) << (8 * (big_endian ? width - 1 - i : i));
	return value;
}

float rsx_vertex_signed16(uint value)
{
	return value < 32768 ? float(value) : float(int(value) - 65536);
}

inline float4 rsx_fetch_attribute(device const uchar* persistent, device const uchar* transient,
	constant rsx_fetch_parameters& p, uint source_vertex, uint location)
{
	const uint2 desc = p.attributes[location];
	const uint stride = desc.x & 255;
	const uint frequency = (desc.x >> 8) & 65535;
	const uint type = (desc.x >> 24) & 7;
	const uint count = (desc.x >> 27) & 7;
	if (!(p.attribute_mask & (1u << location)) || !count) return float4(0, 0, 0, 1);
	if (!type || count > 4 || (type == 6 && count != 1)) return float4(0, 0, 0, 1);
	const uint widths[8] = {0, 2, 4, 2, 1, 2, 4, 1};
	const float scales[8] = {1, 32767.5f, 1, 1, 255, 1, 32767, 1};
	ulong index = 0;
	if (frequency && stride)
	{
		if (desc.y & 0x80000000u)
		{
			const long adjusted = long(source_vertex) + p.vertex_index_offset;
			if (adjusted < 0 || adjusted > 4294967295L) return float4(0, 0, 0, 1);
			index = ulong(adjusted) % frequency;
		}
		else
		{
			if (source_vertex < p.vertex_base_index) return float4(0, 0, 0, 1);
			index = ulong(source_vertex - p.vertex_base_index) / frequency;
		}
	}
	const uint width = widths[type];
	ulong offset = desc.y & 0x1fffffffu;
	const bool is_transient = (desc.y & 0x40000000u) != 0;
	const ulong length = is_transient ? p.transient_length : p.persistent_length;
	device const uchar* stream = is_transient ? transient : persistent;
	const uint mode = desc.x >> 30, records = p.element_counts[location];
	if (mode || records)
	{
		if (!records || mode > 2 || stride < count * width + (mode == 2 ? 4u : 0u)) return float4(0,0,0,1);
		const ulong header = mode == 1 ? 4 : 0;
		if (offset > length || header > length - offset || ulong(records) * stride > length - offset - header) return float4(0,0,0,1);
		if (mode == 2)
		{
			uint low = 0, high = records;
			while (low < high)
			{
				const uint middle = low + (high - low) / 2;
				if (ulong(rsx_vertex_word(stream,offset + ulong(middle) * stride,4,false)) < index) low = middle + 1;
				else high = middle;
			}
			if (low == records || ulong(rsx_vertex_word(stream,offset + ulong(low) * stride,4,false)) != index) return float4(0,0,0,1);
			offset += ulong(low) * stride + 4;
		}
		else
		{
			const ulong base = mode == 1 ? rsx_vertex_word(stream,offset,4,false) : 0;
			if (index < base || index - base >= records) return float4(0,0,0,1);
			offset += header + (index - base) * stride;
		}
	}
	else offset += index * stride;
	if (offset > length || ulong(count * width) > length - offset) return float4(0, 0, 0, 1);
	uint4 bits = uint4(0);
	for (uint c = 0; c < count; ++c)
		bits[c] = rsx_vertex_word(stream, offset + c * width, width, (desc.y & 0x20000000u) != 0);
	// Float inputs already contain their final IEEE bits. Even division by1
	// may flush subnormals on the GPU; no normalization arithmetic is needed.
	if (type == 2)
	{
		if (count < 4) bits.w = 0x3f800000u;
		return as_type<float4>(bits);
	}
	float4 value;
	if (type == 1 || type == 5)
	{
		for (uint c = 0; c < 4; ++c) value[c] = rsx_vertex_signed16(bits[c]);
		if (type == 1) value += 0.5f;
	}
	else if (type == 3)
	{
		for (uint c = 0; c < 4; ++c) value[c] = float(as_type<half>(ushort(bits[c])));
	}
	else if (width == 1) value = float4(bits);
	else
	{
		value = float4(rsx_vertex_signed16((bits.x & 2047) << 5),
			rsx_vertex_signed16(((bits.x >> 11) & 2047) << 5),
			rsx_vertex_signed16(((bits.x >> 22) & 1023) << 6), scales[type]);
	}
	if (count < 4) value.w = scales[type];
	return value / scales[type];
}

kernel void rpcs3_fetch_vertices(device const uchar* persistent [[buffer(0)]],
	device const uchar* transient [[buffer(1)]], device float4* output [[buffer(2)]],
	constant rsx_fetch_parameters& p [[buffer(3)]], uint thread_id [[thread_position_in_grid]])
{
	const uint location = thread_id % 16;
	const uint local = thread_id / 16;
	if (local >= p.chunk_count) return;
	const uint ordinal = p.chunk_base + local;
	if (ordinal >= p.vertex_count) return;
	output[ulong(ordinal) * 16 + location] = rsx_fetch_attribute(persistent, transient, p, p.first_vertex + ordinal, location);
}
)msl";
inline constexpr auto vertex_fetch_runtime = vertex_fetch_kernels.substr(0, vertex_fetch_kernels.find("kernel void rpcs3_fetch_vertices"));
}
