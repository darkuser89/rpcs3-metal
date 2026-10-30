#pragma once

#include <string_view>

namespace metal4
{
inline constexpr std::string_view output_rop_msl = R"metal(
inline float4 rsx_rop_round8(float4 value, bool half_precision) {
    const float4 truncated = as_type<float4>(as_type<uint4>(value) & uint4(0xfffff000u));
    const uint4 raw = uint4(max(floor(fma(truncated, float4(255.0f), float4(0.5f))), float4(0.0f)));
    if (half_precision) return float4(half4(raw) / half4(255.0h));
    return float4(raw) / float4(255.0f);
}
inline float3 rsx_rop_srgb(float3 value) {
    const float3 low = value * 12.92f;
    const float3 high = 1.055f * pow(value, float3(1.0f / 2.4f)) - 0.055f;
    return clamp(select(high, low, value < float3(0.0031308f)), 0.0f, 1.0f);
}
inline bool rsx_rop_alpha_coverage(float alpha, float2 position) {
    // Called VK RSXProgramCommon _rand contract; retain its floating-point
    // behavior rather than replace it with a different integer random stream.
    const float random = fract(sin(dot(position, float2(12.9898f, 78.233f))) * 43758.5453f);
    return alpha > random;
}
)metal";
}
