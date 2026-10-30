#pragma once
#include <string_view>

namespace metal4
{
inline constexpr std::string_view present_kernels = R"msl(
#include <metal_stdlib>
using namespace metal;
struct present_vertex { float4 position [[position]]; float2 uv; };
struct present_parameters {
    float2 origin, extent; uint level, flip_x, flip_y, padding;
    float gamma; uint limited_rgb; uint2 padding2;
    float2 right_origin, right_extent; uint right_level, stereo_mode; uint2 padding3;
    float4 left_matrix[3], right_matrix[3];
    float2 destination_origin, destination_extent;
};
vertex present_vertex m4_present_vertex(uint index [[vertex_id]])
{
    const float2 uv = float2((index << 1) & 2, index & 2);
    return {float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), uv};
}
float4 m4_present_sample(texture2d<float> image, sampler filter, float2 uv,
    float2 origin, float2 extent, uint mip, constant present_parameters& params)
{
    if (params.flip_x) uv.x = 1 - uv.x;
    if (params.flip_y) uv.y = 1 - uv.y;
    // Clamp to selected crop texel centers, preventing linear-filter bleed
    // from adjacent display-buffer pixels at a cropped edge.
    const float2 texel = clamp(origin + uv * extent, origin + 0.5f, origin + extent - 0.5f);
    const float2 size = float2(image.get_width(mip), image.get_height(mip));
    return image.sample(filter, texel / size, level(mip));
}
fragment float4 m4_present_fragment(present_vertex in [[stage_in]],
    texture2d<float> image [[texture(0)]], texture2d<float> right_image [[texture(1)]], sampler filter [[sampler(0)]],
    constant present_parameters& params [[buffer(0)]])
{
    // Pixel centers select the exact half-frame boundary even for odd AV
    // rectangles; interpolated UV rounding must not exchange the center eye.
    float2 uv = (in.position.xy - params.destination_origin) / params.destination_extent;
    bool right = false;
    if (params.stereo_mode == 1) { right = uv.x >= 0.5f; uv.x = uv.x * 2.0f - (right ? 1.0f : 0.0f); }
    if (params.stereo_mode == 2) { right = uv.y >= 0.5f; uv.y = uv.y * 2.0f - (right ? 1.0f : 0.0f); }
    if (params.stereo_mode == 3) right = (uint(in.position.y) & 1u) == 0;
    float4 color;
    if (right) color = m4_present_sample(right_image, filter, uv, params.right_origin, params.right_extent, params.right_level, params);
    else color = m4_present_sample(image, filter, uv, params.origin, params.extent, params.level, params);
    if (params.stereo_mode == 4) {
        const float3 other = m4_present_sample(right_image, filter, uv, params.right_origin, params.right_extent, params.right_level, params).rgb;
        color.rgb = clamp(params.left_matrix[0].rgb * color.r + params.left_matrix[1].rgb * color.g + params.left_matrix[2].rgb * color.b +
            params.right_matrix[0].rgb * other.r + params.right_matrix[1].rgb * other.g + params.right_matrix[2].rgb * other.b, 0.0f, 1.0f);
    }
    if (params.stereo_mode) color.a = 1.0f;
    if (params.gamma != 1.0f) color.rgb = pow(color.rgb, float3(params.gamma));
    // Match the existing VK calibration formula,including its 220 scale.
    if (params.limited_rgb) color.rgb = (color.rgb * 220.0f + 16.0f) / 255.0f;
    return color;
}
)msl";
} // namespace metal4
