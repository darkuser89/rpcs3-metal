#pragma once

#include <string_view>

namespace metal4
{

inline constexpr std::string_view vertex_runtime = R"metal(
#include <metal_stdlib>
using namespace metal;

struct rsx_vertex_context
{
    float4x4 scale_offset;
    uint vertex_base, vertex_count, constant_count, branch_bits;
    float point_size, z_near, z_far;
    uint output_mask;
    uint two_sided, instruction_budget, geometry_count, predicate;
    float clip_factors[6];
    uint clip_enable, flat_count;
};
struct rsx_vertex_reference { uint source_vertex; uint flat_vertex; };
struct rsx_vertex_registers
{
    float4 dst[16];
    float4 temp[64];
    int4 address[2];
    float4 condition[2];
    uint fault;
};

inline rsx_vertex_registers rsx_initialize()
{
    rsx_vertex_registers r;
    for (uint i = 0; i < 16; ++i) r.dst[i] = float4(0, 0, 0, 1);
    for (uint i = 0; i < 64; ++i) r.temp[i] = float4(0);
    for (uint i = 0; i < 2; ++i) { r.address[i] = int4(0); r.condition[i] = float4(0); }
    r.fault = 0;
    return r;
}
inline float4 rsx_attribute(device const float4* attributes, constant rsx_vertex_context& ctx,
    uint source_vertex, long slot, thread uint& fault)
{
    if (source_vertex < ctx.vertex_base || source_vertex - ctx.vertex_base >= ctx.vertex_count || slot < 0 || slot >= 16)
    {
        fault |= 1u;
        return float4(0, 0, 0, 1);
    }
    return attributes[ulong(source_vertex - ctx.vertex_base) * 16 + ulong(slot)];
}
inline float4 rsx_constant(device const float4* constants, constant rsx_vertex_context& ctx,
    long index, thread uint& fault)
{
    if (index < 0 || ulong(index) >= ctx.constant_count)
    {
        fault |= 2u;
        return float4(0);
    }
    return constants[ulong(index)];
}
inline bool4 rsx_condition(float4 value, uint mode)
{
    switch (mode)
    {
    case 1: return value < 0;
    case 2: return value == 0;
    case 3: return value <= 0;
    case 4: return value > 0;
    case 5: return value != 0;
    case 6: return value >= 0;
    case 7: return bool4(true);
    default: return bool4(false);
    }
}
inline int4 rsx_address(float4 value, thread uint& fault)
{
    if (!all(isfinite(value)) || any(value < -2147483648.0f) || any(value >= 2147483648.0f))
    {
        fault |= 32u;
        return int4(0);
    }
    // Match the current Vulkan decoder's conversion. NV ARL's documented
    // floor rule differs for negative fractions; RSX hardware parity is open.
    return int4(value);
}
inline float4 rsx_lit(float4 value)
{
    const float x = max(value.x, 0.0f);
    const float y = max(value.y, 1e-10f);
    return float4(1, x, x > 0 ? exp2(value.w * log2(y)) : 0, 1);
}
inline float4 rsx_zclip(float4 pos, float near_plane, float far_plane)
{
    // Float-only fallback used by the Vulkan path on Apple GPUs. Metal's
    // native vertex stage has no float64 path; exact depth parity remains open.
    float d = pos.z / pos.w;
    if (d < 0 && d >= near_plane) d = 0;
    else if (d > 1 && d <= far_plane)
        d = min(1.0f, 0.99f + 0.01f * (pos.z - near_plane) / (far_plane - near_plane));
    else return pos;
    return float4(pos.xy, d * pos.w, pos.w);
}
)metal";

}
