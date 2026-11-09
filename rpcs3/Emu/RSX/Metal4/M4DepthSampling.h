#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace metal4
{
// Resolved view channels, in RGBA order. VK depth views replicate the depth
// component; register remaps may replace individual channels by zero or one.
enum class depth_sample_channel : std::uint32_t
{
	depth,
	zero,
	one
};
using depth_sample_remap = std::array<depth_sample_channel, 4>;
inline void validate_depth_sample_remap(const depth_sample_remap& remap)
{
	for (auto channel : remap)
		if (channel > depth_sample_channel::one)
			throw std::invalid_argument("Invalid native depth texture view channel");
}
inline constexpr std::string_view depth_sampling_msl = R"metal(
inline float4 rsx_depth_texel(float depth,uint4 channels,float one=1.0f) {
    return select(select(float4(depth),float4(0),channels==1u),float4(one),channels==2u);
}
)metal";
inline constexpr std::string_view point_sampling_msl = R"metal(
inline float rsx_point_axis(float value, uint wrap) {
    if (wrap == 0u) return fract(value);
    if (wrap == 1u) {
        float phase = value - 2.0f * floor(value * 0.5f);
        return min(phase, 2.0f - phase);
    }
    return wrap >= 5u ? abs(value) : value;
}
inline bool rsx_point_position(float2 coord, uint2 extent, uint flags, thread uint2& pixel) {
    if (!all(isfinite(coord))) return false;
    const uint2 wrap = (uint2(flags) >> uint2(7u, 10u)) & 7u;
    coord = float2(rsx_point_axis(coord.x, wrap.x), rsx_point_axis(coord.y, wrap.y));
    if (any((wrap == 3u) & ((coord < 0.0f) | (coord >= 1.0f)))) return false;
    pixel = min(uint2(clamp(coord, 0.0f, 1.0f) * float2(extent)), extent - 1u);
    return true;
}
inline float4 rsx_point_remap(float4 texel, uint flags, float one=1.0f) {
    const uint4 channels = (uint4(flags) >> uint4(13u, 16u, 19u, 22u)) & 7u;
    const float values[6] = {texel.r, texel.g, texel.b, texel.a, 0.0f, one};
    return float4(values[channels.x], values[channels.y], values[channels.z], values[channels.w]);
}
)metal";
inline constexpr std::string_view redirected_depth_sampling_msl = R"metal(
inline float4 rsx_pack_redirected_depth(float z, uint stencil, uint flags) {
    const uint bits = (flags & 64u) ? as_type<uint>(z) >> 7u :
        uint(clamp(isnan(z) ? 0.0f : z, 0.0f, 1.0f) * 16777215.0f);
    const float4 packed = float4((bits >> 8u) & 255u, bits & 255u, stencil, (bits >> 16u) & 255u) / 255.0f;
    return rsx_point_remap(packed, flags);
}
inline float4 rsx_redirected_depth(depth2d<float> depth, texture2d<uint, access::read> stencil,
    float2 coord, constant rsx_sampler_parameters& state) {
    uint2 pixel;
    if (!rsx_point_position(coord, uint2(depth.get_width(), depth.get_height()), state.flags, pixel)) return state.border_adjust;
    return rsx_pack_redirected_depth(depth.read(pixel), stencil.read(pixel).x, state.flags);
}
)metal";
inline constexpr std::string_view multisample_sampling_msl = R"metal(
struct rsx_multisample_window { uint2 origin; int2 extent; uint2 factors; };
template<typename Image>
inline rsx_multisample_window rsx_sample_raster_bounds(Image image) {
    return {uint2(0), int2(image.get_width(), image.get_height()), uint2(1)};
}
inline rsx_multisample_window rsx_multisample_bounds(texture2d<float, access::read> image, constant rsx_texture_coordinates&) {
    return rsx_sample_raster_bounds(image);
}
inline rsx_multisample_window rsx_multisample_bounds(depth2d<float> image, constant rsx_texture_coordinates&) {
    return rsx_sample_raster_bounds(image);
}
template<typename Image>
inline rsx_multisample_window rsx_multisample_bounds(Image image, constant rsx_texture_coordinates& coords) {
    const uint2 factors(2u, image.get_num_samples() / 2u);
    if (coords.flags & 32768u) {
        const uint2 origin(coords.min_x, coords.min_y), end(coords.max_x, coords.max_y);
        return {origin, int2(end - origin), factors};
    }
    return {uint2(0), int2(uint2(image.get_width(), image.get_height()) * factors), factors};
}
template<typename Image>
inline auto rsx_sample_read(Image image, uint2 pixel, uint sample) { return image.read(pixel, sample); }
template<typename T>
inline auto rsx_sample_read(texture2d<T, access::read> image, uint2 pixel, uint) { return image.read(pixel); }
inline float rsx_sample_read(depth2d<float> image, uint2 pixel, uint) { return image.read(pixel); }
struct rsx_sample_address { uint2 pixel; uint sample; };
inline rsx_sample_address rsx_multisample_address(uint2 expanded, uint2 factors) {
    const uint2 location = expanded % factors;
    return {expanded / factors, location.x + location.y * factors.x};
}
inline bool rsx_multisample_position(float2 coord, uint flags, rsx_multisample_window window,
    thread uint2& pixel, thread uint& sample) {
    uint2 expanded;
    if (!rsx_point_position(coord, uint2(window.extent), flags, expanded)) return false;
    const auto address = rsx_multisample_address(expanded + window.origin, window.factors);
    pixel = address.pixel;
    sample = address.sample;
    return true;
}
inline float3 rsx_multisample_weights(float fraction, float step, float texel) {
    if (abs(step - texel) < 0.000001f) return float3(1, 0, 0);
    if (step < texel) return float3(1.0f - fraction, fraction, 0);
    const float covered = fma(fraction, texel, step) - texel;
    const float next = min(texel, covered);
    const float last = max(0.0f, covered - texel);
    return float3(1.0f - next - last, next, last);
}
inline float rsx_multisample_axis(float coord, float extent, uint wrap) {
    if (wrap == 0u) return fract(coord) * extent;
    if (wrap == 1u) return (coord - 2.0f * floor(coord * 0.5f)) * extent;
    if (wrap == 3u) return clamp(coord * extent, -3.0f, extent);
    return clamp(wrap >= 5u ? abs(coord) : coord, 0.0f, 1.0f) * extent;
}
inline int rsx_multisample_tap_axis(int tap, int extent, uint wrap) {
    if (wrap == 0u) return tap % extent;
    if (wrap == 1u) {
        const int phase = tap % (2 * extent);
        return min(phase, 2 * extent - 1 - phase);
    }
    if (wrap == 3u) return tap;
    return clamp(tap, 0, extent - 1);
}
template<typename Image, typename Value>
inline void rsx_multisample_tap(Image image, int2 tap, rsx_multisample_window window, uint2 wrap,
    float weight, bool first, thread Value& value, thread float& border) {
    if (weight == 0.0f) return;
    tap = int2(rsx_multisample_tap_axis(tap.x, window.extent.x, wrap.x), rsx_multisample_tap_axis(tap.y, window.extent.y, wrap.y));
    if (any(tap < 0) || any(tap >= window.extent)) {
        border += weight;
        return;
    }
    const auto address = rsx_multisample_address(uint2(tap) + window.origin, window.factors);
    const Value contribution = rsx_sample_read(image, address.pixel, address.sample) * weight;
    value = first ? contribution : value + contribution;
}
struct rsx_multisample_filter_control {
    float2 dx, dy;
    float level_or_bias;
    uint mode; // 0: implicit, 1: explicit gradients, 2: explicit level.
};
inline bool rsx_multisample_linear(float2 coord, float2 extent, constant rsx_sampler_parameters& state,
    rsx_multisample_filter_control control) {
    const uint filters = state.flags >> 30u;
    if (filters == 0u || filters == 3u) return filters != 0u;
    const float bias = state.bias + control.level_or_bias;
    bool minifies = bias > 0.0f;
    if (control.mode != 2u) {
        const float2 dx = (control.mode == 1u ? control.dx : dfdx(coord)) * extent;
        const float2 dy = (control.mode == 1u ? control.dy : dfdy(coord)) * extent;
        minifies = max(dot(dx, dx), dot(dy, dy)) > exp2(-2.0f * bias);
    }
    return (filters & (minifies ? 2u : 1u)) != 0u;
}
template<typename Image, typename Value>
inline bool rsx_multisample_sample(Image image, float2 coord, uint flags, rsx_multisample_window window,
    bool linear, float2 derivative, thread Value& value, thread float& border) {
    border = 0.0f;
    if (!linear) {
        uint2 pixel;
        uint sample;
        if (!rsx_multisample_position(coord, flags, window, pixel, sample)) return false;
        value = rsx_sample_read(image, pixel, sample);
        return true;
    }
    if (!all(isfinite(coord)) || !all(isfinite(derivative))) return false;
    const int2 extent = window.extent;
    const float2 texel = 1.0f / float2(extent);
    const uint2 wrap = (uint2(flags) >> uint2(7u, 10u)) & 7u;
    const float2 step = select(derivative, -derivative, (wrap >= 5u) & (coord < 0.0f));
    const float2 scaled(rsx_multisample_axis(coord.x, float(extent.x), wrap.x),
        rsx_multisample_axis(coord.y, float(extent.y), wrap.y));
    const int2 pixel = int2(floor(scaled));
    const float3 horizontal = rsx_multisample_weights(fract(scaled.x), step.x, texel.x);
    const float3 vertical = rsx_multisample_weights(fract(scaled.y), step.y, texel.y);
    // The RSX MSAA footprint uses a third tap in X and folds the last
    // two vertical weights into the second row.
    value = Value(0);
    for (int x = 0; x < 3; ++x) {
        rsx_multisample_tap(image, pixel + int2(x, 0), window, wrap, horizontal[x] * vertical.x, x == 0, value, border);
        rsx_multisample_tap(image, pixel + int2(x, 1), window, wrap, horizontal[x] * (vertical.y + vertical.z), false, value, border);
    }
    return true;
}
template<typename Image, typename Value>
inline bool rsx_multisample_value(Image image, float2 coord, constant rsx_sampler_parameters& state,
    constant rsx_texture_coordinates& coords, rsx_multisample_filter_control control, thread Value& value, thread float& border) {
    const auto window = rsx_multisample_bounds(image, coords);
    const float2 extent = float2(window.extent);
    const uint limit = uint(state.max_lod);
    if (limit > 1u && control.mode != 2u && all(isfinite(coord))) {
        const float2 dx = control.mode == 1u ? control.dx : dfdx(coord);
        const float2 dy = control.mode == 1u ? control.dy : dfdy(coord);
        const float2 tx = dx * extent, ty = dy * extent;
        const float px2 = dot(tx, tx), py2 = dot(ty, ty);
        if (isfinite(px2) && isfinite(py2)) {
            const auto footprint = rsx_texture_footprint(px2, py2, limit, 0.0f);
            if (footprint.taps > 1u) {
                const float count = float(footprint.taps);
                const float2 major = footprint.major_x ? dx : dy;
                const float2 derivative(dx.x / (footprint.major_x ? count : 1.0f),
                    dy.y / (footprint.major_x ? 1.0f : count));
                const bool linear = (state.flags & (1u << 31u)) != 0u;
                value = Value(0);
                border = 0.0f;
                for (uint i = 1u; i <= footprint.taps; ++i) {
                    Value sample;
                    float coverage;
                    const float2 position = coord + major * (float(i) / (count + 1.0f) - 0.5f);
                    if (rsx_multisample_sample(image, position, state.flags, window, linear, derivative, sample, coverage)) {
                        value += sample;
                        border += coverage;
                    } else {
                        border += 1.0f;
                    }
                }
                value /= count;
                border /= count;
                return true;
            }
        }
    }
    const bool linear = rsx_multisample_linear(coord, extent, state, control);
    const float2 derivative = linear ? float2(dfdx(coord.x), dfdy(coord.y)) : float2(0);
    return rsx_multisample_sample(image, coord, state.flags, window, linear, derivative, value, border);
}
template<typename Image>
inline float4 rsx_multisample_color(Image image, float2 coord, constant rsx_sampler_parameters& state,
    constant rsx_texture_coordinates& coords, rsx_multisample_filter_control control) {
    float4 value;
    float border;
    if (!rsx_multisample_value(image, coord, state, coords, control, value, border)) return state.border_adjust;
    const float4 remapped = rsx_point_remap(value, state.flags, 1.0f - border);
    return border == 0.0f ? remapped : remapped + state.border_adjust * border;
}
template<typename Image>
inline float4 rsx_multisample_depth(Image image, float2 coord, constant rsx_sampler_parameters& state,
    constant rsx_texture_coordinates& coords, rsx_multisample_filter_control control, uint4 remap) {
    float value;
    float border;
    if (!rsx_multisample_value(image, coord, state, coords, control, value, border)) return state.border_adjust;
    const float4 remapped = rsx_depth_texel(value, remap, 1.0f - border);
    return border == 0.0f ? remapped : remapped + state.border_adjust * border;
}
template<typename Image>
inline float4 rsx_multisample_shadow(Image image, float2 coord, constant rsx_sampler_parameters& state,
    constant rsx_texture_coordinates& coords, float reference) {
    float z;
    float border;
    if (!rsx_multisample_value(image, coord, state, coords, {}, z, border)) z = state.border_adjust.r;
    else if (border != 0.0f) z += state.border_adjust.r * border;
    return float4(rsx_fp_compare(z, rsx_shadow_reference(reference, state), (state.flags >> 27u) & 7u));
}
)metal";
inline constexpr std::string_view multisample_redirected_depth_msl = R"metal(
template<typename Depth, typename Stencil>
inline float4 rsx_multisample_redirected(Depth depth, Stencil stencil,
    float2 coord, constant rsx_sampler_parameters& state, constant rsx_texture_coordinates& coords) {
    uint2 pixel;
    uint sample;
    if (!rsx_multisample_position(coord, state.flags, rsx_multisample_bounds(depth, coords), pixel, sample)) return state.border_adjust;
    return rsx_pack_redirected_depth(rsx_sample_read(depth, pixel, sample), rsx_sample_read(stencil, pixel, sample).x, state.flags);
}
)metal";
} // namespace metal4
