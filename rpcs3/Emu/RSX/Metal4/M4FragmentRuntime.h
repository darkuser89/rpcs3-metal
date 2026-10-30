#pragma once

#include <string_view>

namespace metal4
{

inline constexpr std::string_view fragment_runtime = R"metal(
#include <metal_stdlib>
using namespace metal;
struct rsx_fragment_context {
    uint constant_count, instruction_budget, texcoord_2d_mask, point_sprite_mask;
    uint two_sided, alpha_test, alpha_function, predicate;
    float alpha_reference, wpos_scale, wpos_bias_x, wpos_bias_y;
    float derivative_sign_y, fog_param0, fog_param1;
    uint fog_mode, sample_mask, reserved[3];
    float4 blend_color;
    uint stipple_pattern[32];
    uint depth_bounds_enabled;
    float depth_min, depth_max;
    uint depth_reserved;
};
struct rsx_fragment_registers {
    uint words[192];
    float4 condition[2];
    int address;
    uint fault;
    bool killed;
};
inline rsx_fragment_registers rsx_fp_initialize() {
    rsx_fragment_registers r;
    for (uint i = 0; i < 192; ++i) r.words[i] = 0;
    r.condition[0] = r.condition[1] = float4(0);
    r.address = 0; r.fault = 0; r.killed = false;
    return r;
}
inline float rsx_fp_half(uint word, uint shift) {
    return float(as_type<half>(ushort(word >> shift)));
}
inline void rsx_fp_store_half(thread uint& word, uint shift, float value) {
    const uint raw = uint(as_type<ushort>(half(value)));
    word = (word & ~(0xffffu << shift)) | (raw << shift);
}
inline float4 rsx_fp_read(thread const rsx_fragment_registers& r, uint index, bool fp16) {
    if (fp16) {
        return float4(rsx_fp_half(r.words[index * 2], 0), rsx_fp_half(r.words[index * 2], 16),
            rsx_fp_half(r.words[index * 2 + 1], 0), rsx_fp_half(r.words[index * 2 + 1], 16));
    }
    return as_type<float4>(uint4(r.words[index * 4], r.words[index * 4 + 1],
        r.words[index * 4 + 2], r.words[index * 4 + 3]));
}
inline float4 rsx_fp_clamp(float4 value, float lo, float hi) {
    return clamp(select(value, float4(0), isnan(value)), lo, hi);
}
inline bool4 rsx_fp_condition(float4 value, uint mode) {
    switch (mode) {
    case 1: return value < 0; case 2: return value == 0;
    case 3: return value <= 0; case 4: return value > 0;
    case 5: return value != 0; case 6: return value >= 0;
    case 7: return bool4(true); default: return bool4(false);
    }
}
inline bool rsx_fp_compare(float a, float b, uint mode) {
    switch (mode) {
    case 1: return a < b; case 2: return a == b;
    case 3: return a <= b; case 4: return a > b;
    case 5: return a != b; case 6: return a >= b;
    case 7: return true; default: return false;
    }
}
inline float4 rsx_fp_constant(device const float4* constants, constant rsx_fragment_context& ctx,
    uint index, thread uint& fault) {
    if (index >= ctx.constant_count) { fault |= 1u; return float4(0); }
    return constants[index];
}
inline float4 rsx_fp_divsq(float4 a, float b) {
    return select(a, a / sqrt(abs(b)), abs(a) > 0);
}
inline float4 rsx_fp_lit(float4 value) {
    float x = max(value.x, 0.0f), y = max(value.y, 1e-10f);
    return float4(1, x, x > 0 ? exp2(value.w * log2(y)) : 0, 1);
}
inline float4 rsx_fp_lif(float4 value) {
    return float4(1, value.y, value.y > 0 ? exp2(value.w) : 0, 1);
}
inline float4 rsx_fp_normalize(float4 value) {
    float3 n = length(value.xyz) > 0 ? normalize(value.xyz) : value.xyz;
    return n.xyzz;
}
inline float4 rsx_fp_linear_srgb(float4 value) {
    float4 low = value * 12.92f;
    float4 high = 1.055f * pow(value, float4(1.0f / 2.4f)) - 0.055f;
    return clamp(select(high, low, value < 0.0031308f), 0.0f, 1.0f);
}
inline float4 rsx_fp_srgb_linear(float4 value) {
    return select(value / 12.92f, pow((value + 0.055f) / 1.055f, float4(2.4f)), value > 0.04045f);
}
inline float4 rsx_fp_fog(float coordinate, constant rsx_fragment_context& ctx) {
    float x = (ctx.fog_mode >= 4) ? abs(coordinate) : coordinate;
    float v = ctx.fog_param1 * x + ctx.fog_param0 - 1.5f;
    float y;
    switch (ctx.fog_mode) {
    case 1: case 6: y = ctx.fog_param1 * x + ctx.fog_param0 - 1.0f; break;
    case 2: case 4: y = exp(11.084f * v); break;
    case 3: case 5: y = exp(-pow(4.709f * v, 2.0f)); break;
    default: return float4(coordinate, 0, 0, 0);
    }
    return float4(coordinate, clamp(y, 0.0f, 1.0f), 0, 0);
}
)metal";

}
