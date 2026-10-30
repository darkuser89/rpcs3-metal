#pragma once

#include <string_view>

namespace metal4
{
// Mirrors the programmable RSX blend contract: raw source exports blend with
// remapped previous attachment contents before output channel remapping.
inline constexpr std::string_view blend_rop_msl = R"metal(
inline float4 rsx_blend_destination(float4 color, uint remap) {
    switch (remap) {
    case 1: return color.rrrr;
    case 2: return color.rgrg;
    case 3: case 4: return float4(color.rgb, 1.0f);
    default: return color;
    }
}
inline float4 rsx_blend_factor(uint factor, float4 source, float4 destination, float4 blend_constant) {
    switch (factor) {
    case 0: return 0; case 1: return 1;
    case 2: return source; case 3: return 1.0f - source;
    case 4: return source.aaaa; case 5: return 1.0f - source.aaaa;
    case 6: return destination; case 7: return 1.0f - destination;
    case 8: return destination.aaaa; case 9: return 1.0f - destination.aaaa;
    case 10: return float4(float3(min(source.a, 1.0f - destination.a)), 1.0f);
    case 11: return blend_constant; case 12: return 1.0f - blend_constant;
    case 13: return blend_constant.aaaa; case 14: return 1.0f - blend_constant.aaaa;
    default: return 0;
    }
}
inline float rsx_blend_saturate(float value, bool normalized) {
    return normalized ? clamp(value, 0.0f, 1.0f) : value;
}
inline float rsx_blend_signed(float value, bool normalized) {
    if (!normalized) return value;
    const uint byte = uint(clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    const int signed_byte = int(byte) - (byte >= 128u ? 256 : 0);
    return float(signed_byte) / 255.0f;
}
inline float rsx_blend_component(uint operation, float source, float destination,
    float source_factor, float destination_factor, bool normalized) {
    switch (operation) {
    case 3: return min(rsx_blend_saturate(source, normalized), destination);
    case 4: return max(rsx_blend_saturate(source, normalized), destination);
    case 5: return destination - rsx_blend_signed(source, normalized);
    case 6: return rsx_blend_signed(source, normalized) + destination;
    case 7: return rsx_blend_saturate(source, normalized) + rsx_blend_signed(destination, normalized);
    }
    const float s = rsx_blend_saturate(source * source_factor, normalized);
    const float d = destination * destination_factor;
    switch (operation) {
    case 0: return s + d; case 1: return s - d; case 2: return d - s;
    default: return 0;
    }
}
inline float4 rsx_blend_color(float4 source, float4 native_destination, float4 blend_constant,
    uint4 factors, uint2 operations, uint remap, bool normalized) {
    const float4 destination = rsx_blend_destination(native_destination, remap);
    const float4 sf = rsx_blend_factor(factors.x, source, destination, blend_constant);
    const float4 df = rsx_blend_factor(factors.y, source, destination, blend_constant);
    const float4 sa = rsx_blend_factor(factors.z, source, destination, blend_constant);
    const float4 da = rsx_blend_factor(factors.w, source, destination, blend_constant);
    return float4(
        rsx_blend_component(operations.x, source.r, destination.r, sf.r, df.r, normalized),
        rsx_blend_component(operations.x, source.g, destination.g, sf.g, df.g, normalized),
        rsx_blend_component(operations.x, source.b, destination.b, sf.b, df.b, normalized),
        rsx_blend_component(operations.y, source.a, destination.a, sa.a, da.a, normalized));
}
// Tile-local attachment reads can observe pre-store values. Quantize each
// fragment's export explicitly so subsequent primitives read stored precision.
inline float4 rsx_blend_store(float4 color, uint4 bits) {
    for (uint c = 0; c < 4; ++c) {
        if (!bits[c]) continue;
        const float maximum = float((1u << bits[c]) - 1u);
        color[c] = rint(clamp(color[c], 0.0f, 1.0f) * maximum) / maximum;
    }
    return color;
}
)metal";
}
