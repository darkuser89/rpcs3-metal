#pragma once
#include <string_view>
namespace metal4
{
inline constexpr std::string_view dma_kernels = R"metal(
#include <metal_stdlib>
using namespace metal;
struct dma_active_group
{
    uint first, row_step, columns, rows, count, merged;
};
struct dma_parameters
{
    long source_first, destination_first, source_pitch, destination_pitch;
    uint source_stride, destination_stride, columns, rows, bytes, groups, source_groups, packed;
    dma_active_group active[4];
    dma_active_group source_active[4];
};
long dma_floor(long value, long divisor)
{
    const long quotient = value / divisor;
    return quotient - long(value % divisor < 0);
}
long dma_ceil(long value, long divisor) { return -dma_floor(-value, divisor); }
kernel void rpcs3_dma_bytes(device const uchar* source [[buffer(0)]],
    device uchar* destination [[buffer(1)]], constant dma_parameters& p [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= p.bytes) return;
    const uint packed_output = index;
    uint logical = index;
    for (uint group = 0; group < p.groups; ++group)
    {
        const auto g = p.active[group];
        if (logical >= g.count) { logical -= g.count; continue; }
        index = g.first + (g.merged ? logical * p.destination_stride :
            (logical / g.columns) * g.row_step + (logical % g.columns) * p.destination_stride);
        break;
    }
    const long delta = long(index) - p.destination_first;
    const long width = long(p.columns - 1) * p.destination_stride;
    long low = 0, high = long(p.rows) - 1;
    if (p.destination_pitch > 0)
    {
        low = max(low, dma_ceil(delta - width, p.destination_pitch));
        high = min(high, dma_floor(delta, p.destination_pitch));
    }
    else if (p.destination_pitch < 0)
    {
        low = max(low, dma_ceil(-delta, -p.destination_pitch));
        high = min(high, dma_floor(width - delta, -p.destination_pitch));
    }
    else if (delta < 0 || delta > width) return;
    // One invocation owns each destination byte. Overlapping/zero-pitch rows
    // choose the last row in command order;at most stride<=4 congruence probes.
    for (uint candidate = 0; candidate < p.destination_stride && high >= low; ++candidate, --high)
    {
        const long column_bytes = delta - high * p.destination_pitch;
        if (column_bytes < 0 || column_bytes > width || column_bytes % p.destination_stride) continue;
        const long column = column_bytes / p.destination_stride;
        const long input = p.source_first + high * p.source_pitch + column * p.source_stride;
        if (p.packed)
        {
            uint base = 0;
            for (uint group = 0; group < p.source_groups; ++group)
            {
                const auto g = p.source_active[group];
                const long relative = input - g.first;
                if (relative >= 0)
                {
                    const long row = g.merged ? 0 : relative / g.row_step;
                    const long byte = g.merged ? relative : relative % g.row_step;
                    const long slot = g.merged ? byte / p.source_stride : row * g.columns + byte / p.source_stride;
                    if (byte % p.source_stride == 0 && slot < g.count &&
                        (g.merged || (row < g.rows && byte / p.source_stride < g.columns)))
                    {
                        destination[packed_output] = source[base + uint(slot)];
                        return;
                    }
                }
                base += g.count;
            }
            return;
        }
        destination[index] = source[input];
        return;
    }
}
)metal";
}
