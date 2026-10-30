#pragma once

#include <string_view>

namespace metal4
{
inline constexpr std::string_view logic_rop_msl = R"metal(
inline uint rsx_logic_word(uint operation, uint source, uint destination) {
    switch (operation) {
    case 0x1500: return 0; case 0x1501: return source & destination;
    case 0x1502: return source & ~destination; case 0x1503: return source;
    case 0x1504: return ~source & destination; case 0x1505: return destination;
    case 0x1506: return source ^ destination; case 0x1507: return source | destination;
    case 0x1508: return ~(source | destination); case 0x1509: return ~(source ^ destination);
    case 0x150a: return ~destination; case 0x150b: return source | ~destination;
    case 0x150c: return ~source; case 0x150d: return ~source | destination;
    case 0x150e: return ~(source & destination); case 0x150f: return 0xffffffffu;
    default: return source;
    }
}
inline float4 rsx_logic_color(float4 source, float4 destination, uint operation, uint4 bits) {
    float4 result = source;
    for (uint channel = 0; channel < 4; ++channel) {
        if (bits[channel] == 0) continue;
        const uint mask = (1u << bits[channel]) - 1u;
        float s = isnan(source[channel]) ? 0.0f : clamp(source[channel], 0.0f, 1.0f);
        float d = isnan(destination[channel]) ? 0.0f : clamp(destination[channel], 0.0f, 1.0f);
        const uint a = uint(rint(s * float(mask))), b = uint(rint(d * float(mask)));
        result[channel] = float(rsx_logic_word(operation, a, b) & mask) / float(mask);
    }
    return result;
}
)metal";
} // namespace metal4
