#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace metal4
{
inline constexpr std::uint32_t texel_alpha_kill = 1, texel_renormalize = 2, texel_channels16 = 4, texel_biased_format = 8;
// Semantic masks use RGBA bits0..3 after native view/remap. Register/format
// adapter supplies them; they are not raw RSX texture-control bit positions.
struct alignas(16) texel_processing
{
	std::uint32_t sign_mask = 0, gamma_mask = 0, expand_mask = 0, bx2_mask = 15;
	std::uint32_t control = 0;
	std::array<std::uint32_t, 3> reserved{};
};
static_assert(sizeof(texel_processing) == 32 && offsetof(texel_processing, control) == 16);
inline void validate_texel_processing(const texel_processing& value)
{
	if ((value.sign_mask | value.gamma_mask | value.expand_mask | value.bx2_mask) & ~15u || (value.control & ~15u) || value.reserved != std::array<std::uint32_t, 3>{})
		throw std::invalid_argument("Invalid native texel processing masks/control");
}
inline constexpr std::string_view texel_processing_msl = R"metal(
struct rsx_texel_processing { uint sign_mask,gamma_mask,expand_mask,bx2_mask; uint control; uint reserved[3]; };
inline float4 rsx_process_texel(float4 value,constant rsx_texel_processing& p,bool expand_instruction,thread bool& killed,thread uint& fault) {
    if ((p.control & 1u) && value.a < 0.000001f) { killed=true; return value; }
    if (p.control & 2u) value=floor(value*255.0f)/255.0f;
    uint gamma=p.gamma_mask, expand=p.expand_mask;
    const bool bx2=expand_instruction && (p.control & 8u);
    if (bx2) { gamma=0; expand|=p.bx2_mask; }
    gamma &= ~p.sign_mask; expand &= ~(p.sign_mask|gamma);
    const uint bits=(p.control & 4u)?16u:8u;
    const float range=(p.control & 4u)?65535.0f:255.0f;
    const float center=(p.control & 4u)?32768.0f:128.0f;
    const float divisor=(p.control & 4u)?32767.0f:127.0f;
    for (uint c=0;c<4;++c) {
        const uint mask=1u<<c;
        if (p.sign_mask & mask) {
            const float q=floor(fma(value[c],range,0.5f));
            if (!(q>=0 && q<4294967296.0f)) { fault|=64u; return value; }
            const uint raw=uint(q)<<(32u-bits);
            value[c]=float(as_type<int>(raw)>>(32u-bits))/divisor;
        } else if (gamma & mask) {
            value[c]=value[c]>0.04045f?pow((value[c]+0.055f)/1.055f,2.4f):value[c]/12.92f;
        } else if (expand & mask) {
            value[c]=bx2?value[c]*2.0f-1.0f:(floor(fma(value[c],range,0.5f))-center)/divisor;
        }
    }
    return value;
}
)metal";
} // namespace metal4
