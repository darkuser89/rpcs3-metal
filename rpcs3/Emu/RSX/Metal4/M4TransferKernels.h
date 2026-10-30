#pragma once

#include <string_view>

namespace metal4
{

inline constexpr std::string_view transfer_kernels = R"metal(
#include <metal_stdlib>
using namespace metal;

struct copy_parameters
{
    ulong source_offset;
    ulong destination_offset;
    ulong count;
    uint swap_width;
    uint padding;
};

kernel void rpcs3_copy_bytes(device const uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant copy_parameters& params [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (ulong(index) < params.count)
    {
        const ulong source_index = ulong(index) ^ ulong(params.swap_width - 1);
        destination[params.destination_offset + index] = source[params.source_offset + source_index];
    }
}

struct fill_parameters
{
    ulong word_offset;
    ulong count;
    uint value;
    uint padding;
};

kernel void rpcs3_fill_words(device uint* destination [[buffer(0)]],
    constant fill_parameters& params [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (ulong(index) < params.count)
    {
        destination[params.word_offset + index] = params.value;
    }
}

struct e4m12_parameters
{
    ulong source_offset;
    ulong destination_offset;
    ulong count;
    uint expand;
    uint source_big_endian;
    uint destination_big_endian;
    uint padding;
};

kernel void rpcs3_convert_e4m12(device const uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant e4m12_parameters& params [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (ulong(index) >= params.count) return;
    const uint source_width = params.expand ? 2 : 4;
    const uint destination_width = params.expand ? 4 : 2;
    const ulong source_base = params.source_offset + ulong(index) * source_width;
    const ulong destination_base = params.destination_offset + ulong(index) * destination_width;
    uint value = 0;
    for (uint byte = 0; byte < source_width; ++byte)
    {
        const uint shift = (params.source_big_endian ? source_width - 1 - byte : byte) * 8;
        value |= uint(source[source_base + byte]) << shift;
    }
    // Match VKCompute.h's bit operations. In particular, don't use half casts,
    // float arithmetic, saturation or rounding-to-nearest in this codec.
    value = params.expand ? (value << 11) + (120u << 23) : (value - (120u << 23)) >> 11;
    for (uint byte = 0; byte < destination_width; ++byte)
    {
        const uint shift = (params.destination_big_endian ? destination_width - 1 - byte : byte) * 8;
        destination[destination_base + byte] = uchar(value >> shift);
    }
}
struct nearest_parameters
{
    ulong source_offset, destination_offset;
    ulong source_row, source_image, destination_row, destination_image;
    ulong first, count;
    uint source_width, source_height, source_depth;
    uint destination_width, destination_height, destination_depth;
    uint bytes, flip;
};

uint rpcs3_nearest_axis(uint destination, uint source_extent, uint destination_extent, bool flip)
{
    const ulong numerator = ulong(destination) * source_extent + source_extent / 2;
    const uint quotient = uint(numerator / destination_extent);
    // Flip the continuous source coordinate before selecting a texel. At an
    // exact integer boundary, reversing the selected index is off by one.
    const uint ceil_adjustment = uint(numerator % destination_extent != 0 || (source_extent & 1) != 0);
    return flip ? source_extent - quotient - ceil_adjustment : quotient;
}

kernel void rpcs3_nearest_bytes(device const uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]],
    constant nearest_parameters& p [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (ulong(index) >= p.count) return;
    const ulong texel = p.first + index;
    const uint x = uint(texel % p.destination_width);
    const uint y = uint((texel / p.destination_width) % p.destination_height);
    const uint z = uint(texel / (ulong(p.destination_width) * p.destination_height));
    // floor((d + 0.5) * source/destination), without floating-point or
    // overflowing the doubled numerator. All extents are nonzero uint32.
    const uint sx = rpcs3_nearest_axis(x, p.source_width, p.destination_width, bool(p.flip & 1));
    const uint sy = rpcs3_nearest_axis(y, p.source_height, p.destination_height, bool(p.flip & 2));
    const uint sz = rpcs3_nearest_axis(z, p.source_depth, p.destination_depth, bool(p.flip & 4));
    const ulong src = p.source_offset + ulong(sz) * p.source_image + ulong(sy) * p.source_row + ulong(sx) * p.bytes;
    const ulong dst = p.destination_offset + ulong(z) * p.destination_image + ulong(y) * p.destination_row + ulong(x) * p.bytes;
    for (uint byte = 0; byte < p.bytes; ++byte)
        destination[dst + byte] = source[src + byte];
}
)metal";

}
