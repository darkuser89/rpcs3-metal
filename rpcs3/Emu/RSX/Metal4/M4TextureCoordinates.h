#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace metal4
{
inline constexpr std::uint32_t texture_coordinate_clamp = 1;
inline constexpr std::uint32_t texture_coordinate_mirror_x = 2;
inline constexpr std::uint32_t texture_coordinate_mirror_y = 4;
inline constexpr std::uint32_t texture_coordinate_repeat_x = 8;
inline constexpr std::uint32_t texture_coordinate_repeat_y = 16;
inline constexpr std::uint32_t texture_coordinate_mirror_repeat_x = 32;
inline constexpr std::uint32_t texture_coordinate_mirror_repeat_y = 64;
inline constexpr std::uint32_t texture_coordinate_periodic_pixel_bias = 128;
inline constexpr std::uint32_t texture_coordinate_repeat_linear = 256;
inline constexpr std::uint32_t texture_coordinate_mixed_filter = 512;
inline constexpr std::uint32_t texture_coordinate_mixed_min_linear = 1024;
inline constexpr std::uint32_t texture_coordinate_mixed_pixels = 2048;
inline constexpr std::uint32_t texture_coordinate_mixed_point_edge_x = 4096;
inline constexpr std::uint32_t texture_coordinate_mixed_point_edge_y = 8192;
inline constexpr std::uint32_t texture_coordinate_border_atlas = 16384;
// Minimum and maximum hold the sample-raster origin and exclusive end.
inline constexpr std::uint32_t texture_coordinate_msaa_window = 32768;
// A single-level image containing the already laid-out MSAA sample raster.
inline constexpr std::uint32_t texture_coordinate_sample_raster = 65536;
// Atlas modes store X/Y wrap, within-level and mip filters, coordinate
// flags and, for volume images, the three-bit Z wrap value.
inline constexpr std::uint32_t border_atlas_min_linear = 1 << 6;
inline constexpr std::uint32_t border_atlas_mag_linear = 1 << 7;
inline constexpr std::uint32_t border_atlas_nearest_mip = 1 << 8;
inline constexpr std::uint32_t border_atlas_linear_mip = 2 << 8;
inline constexpr std::uint32_t border_atlas_pixels = 1 << 10;
inline constexpr std::uint32_t border_atlas_point_bias = 1 << 11;
inline constexpr std::uint32_t border_atlas_one_dimensional = 1 << 12;
inline constexpr unsigned border_atlas_wrap_z_shift = 13;
inline constexpr std::uint32_t border_atlas_wrap_z_mask = 7 << border_atlas_wrap_z_shift;
inline constexpr std::uint32_t border_atlas_cube = 1 << 16;
inline constexpr unsigned border_atlas_anisotropy_shift = 17;
inline constexpr std::uint32_t border_atlas_anisotropy_mask = 31 << border_atlas_anisotropy_shift;
inline constexpr std::uint32_t texture_coordinate_repeat = texture_coordinate_repeat_x | texture_coordinate_repeat_y;
inline constexpr std::uint32_t texture_coordinate_periodic = texture_coordinate_repeat |
	texture_coordinate_mirror_repeat_x | texture_coordinate_mirror_repeat_y;
struct stored_border_mip_atlas
{
	std::array<std::size_t, 2> extent{};
	std::array<std::array<std::size_t, 2>, 16> origins{};
};
// Base dimensions come from 16-bit guest texture registers. The strip places
// all smaller rings beyond the short side, in increasing logical mip order.
[[nodiscard]] inline stored_border_mip_atlas plan_stored_border_atlas(std::size_t width, std::size_t height, std::size_t mips, bool one_dimensional)
{
	if (!width || !height || width > 65535 || height > 65535 || !mips || mips > 16 ||
		(one_dimensional && height != 1))
		throw std::invalid_argument("Invalid stored-border atlas dimensions");
	if (one_dimensional)
	{
		stored_border_mip_atlas result{{width + 2, mips}, {}};
		for (std::size_t mip = 1; mip < mips; ++mip)
			result.origins[mip][1] = mip;
		return result;
	}
	stored_border_mip_atlas result{{width + 2, height + 2}, {}};
	const std::array base{width, height};
	const unsigned along = width >= height ? 0 : 1;
	const unsigned across = 1 - along;
	const auto strip = result.extent[across];
	std::size_t next = 0;
	for (std::size_t mip = 1; mip < mips; ++mip)
	{
		auto& origin = result.origins[mip];
		origin[along] = next;
		origin[across] = strip;
		next += std::max(base[along] >> mip, std::size_t{1}) + 2;
		result.extent[along] = std::max(result.extent[along], next);
		result.extent[across] = std::max(result.extent[across], strip + std::max(base[across] >> mip, std::size_t{1}) + 2);
	}
	return result;
}
// Native fragment buffer4. Guest/register/cache adapter supplies effective
// scale/bias/subregion clamps; native image dimensions alone cannot infer them.
struct alignas(16) texture_coordinates
{
	std::array<float, 3> scale{1, 1, 1};
	std::uint32_t flags = 0;
	std::array<float, 3> bias{};
	std::uint32_t border_atlas_depth = 0;
	std::array<float, 2> minimum{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
	std::array<float, 2> maximum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
	// Logical base width, height, mip count and addressing/filter modes.
	std::array<std::uint32_t, 4> border_atlas{};
};
static_assert(sizeof(texture_coordinates) == 64 && offsetof(texture_coordinates, border_atlas) == 48 && offsetof(texture_coordinates, flags) == 12 && offsetof(texture_coordinates, bias) == 16 && offsetof(texture_coordinates, border_atlas_depth) == 28 && offsetof(texture_coordinates, minimum) == 32 && offsetof(texture_coordinates, maximum) == 40);
inline void validate_texture_coordinates(const texture_coordinates& value)
{
	constexpr auto mirrors = texture_coordinate_mirror_x | texture_coordinate_mirror_y;
	constexpr auto mixed = texture_coordinate_mixed_min_linear | texture_coordinate_mixed_pixels |
		texture_coordinate_mixed_point_edge_x | texture_coordinate_mixed_point_edge_y;
	constexpr auto auxiliary = texture_coordinate_periodic_pixel_bias | texture_coordinate_repeat_linear |
		texture_coordinate_mixed_filter | mixed;
	if ((value.flags & ~(texture_coordinate_clamp | mirrors | texture_coordinate_periodic | auxiliary | texture_coordinate_border_atlas | texture_coordinate_msaa_window | texture_coordinate_sample_raster)) ||
		((value.flags & texture_coordinate_msaa_window) && value.flags != texture_coordinate_msaa_window) ||
		((value.flags & texture_coordinate_sample_raster) && value.flags != texture_coordinate_sample_raster) ||
		((value.flags & (mirrors | texture_coordinate_periodic)) && !(value.flags & texture_coordinate_clamp)) ||
		((value.flags & texture_coordinate_periodic_pixel_bias) && !(value.flags & texture_coordinate_periodic)) ||
		((value.flags & texture_coordinate_mixed_filter) && !(value.flags & texture_coordinate_clamp)) ||
		((value.flags & mixed) && !(value.flags & texture_coordinate_mixed_filter)) ||
		((value.flags & texture_coordinate_repeat_linear) && !(value.flags & texture_coordinate_repeat)))
		throw std::invalid_argument("Invalid texture coordinate flags");
	const auto& atlas = value.border_atlas;
	if (value.flags & texture_coordinate_border_atlas)
	{
		const bool volume = value.border_atlas_depth != 0;
		const bool one_dimensional = atlas[3] & border_atlas_one_dimensional;
		const bool cube = atlas[3] & border_atlas_cube;
		const auto factor = (atlas[3] & border_atlas_anisotropy_mask) >> border_atlas_anisotropy_shift;
		const auto allowed = 8191u | border_atlas_cube | border_atlas_anisotropy_mask | (volume ? border_atlas_wrap_z_mask : 0u);
		if (value.flags != texture_coordinate_border_atlas || !atlas[0] || !atlas[1] ||
			atlas[0] > 65535 || atlas[1] > 65535 || !atlas[2] || atlas[2] > 16 || value.border_atlas_depth > 2048 ||
			(one_dimensional && (atlas[1] != 1 || volume || cube)) ||
			(cube && (volume || atlas[0] != atlas[1] || (atlas[3] & (border_atlas_pixels | border_atlas_point_bias)))) ||
			(!one_dimensional && !volume && !cube && atlas[2] < 2 && !factor) ||
			(factor != 0 && factor != 2 && factor != 4 && factor != 6 && factor != 8 && factor != 10 && factor != 12 && factor != 16) ||
			(atlas[3] & ~allowed) || (atlas[3] & 768u) == 768u)
			throw std::invalid_argument("Invalid stored-border atlas sampling metadata");
	}
	else if (value.border_atlas_depth || atlas != std::array<std::uint32_t, 4>{})
		throw std::invalid_argument("Atlas metadata requires atlas sampling");
	for (unsigned i = 0; i < 3; ++i)
		if (!std::isfinite(value.scale[i]) || !std::isfinite(value.bias[i]))
			throw std::invalid_argument("Nonfinite texture coordinate transform");
	for (unsigned i = 0; i < 2; ++i)
	{
		if ((value.flags & texture_coordinate_msaa_window) &&
			(value.minimum[i] < 0 || value.maximum[i] > 65536 || value.minimum[i] >= value.maximum[i] ||
			 value.minimum[i] != std::floor(value.minimum[i]) || value.maximum[i] != std::floor(value.maximum[i])))
			throw std::invalid_argument("Invalid native multisample window");
		// Periodic axes store the guest period and the native half-texel
		// size in place of the two clamp bounds.
		const auto axis = value.flags & ((texture_coordinate_mirror_x | texture_coordinate_repeat_x |
			texture_coordinate_mirror_repeat_x) << i);
		const bool repeats = axis & texture_coordinate_periodic;
		if (axis && (axis & (axis - 1)))
			throw std::invalid_argument("Conflicting texture coordinate modes on one axis");
		if (!std::isfinite(value.minimum[i]) || !std::isfinite(value.maximum[i]) ||
			(repeats ? value.minimum[i] <= 0 || value.maximum[i] <= 0 || value.maximum[i] > 1.f / 6 :
				value.minimum[i] > value.maximum[i]))
			throw std::invalid_argument("Invalid texture coordinate clamp range");
	}
}
// Use logical guest dimensions, which may differ from a scaled native image.
[[nodiscard]] inline texture_coordinates normalize_guest_pixels(std::size_t width, std::size_t height = 1, std::size_t depth = 1)
{
	if (!width || !height || !depth)
		throw std::invalid_argument("Guest texture dimensions must be nonzero");
	texture_coordinates value;
	value.scale = {1.f / float(width), 1.f / float(height), 1.f / float(depth)};
	return value;
}
inline constexpr std::string_view texture_coordinates_msl = R"metal(
struct rsx_texture_coordinates {
    float scale_x, scale_y, scale_z; uint flags;
    float bias_x, bias_y, bias_z; uint border_atlas_depth;
    float min_x, min_y, max_x, max_y;
    uint4 border_atlas;
};
inline float rsx_texture_coordinate(float coordinate, constant rsx_texture_coordinates& p) {
    float value=fma(coordinate,p.scale_x,p.bias_x);
    return (p.flags & 1u) ? clamp(value,p.min_x,p.max_x) : value;
}
inline float rsx_periodic_coordinate(float coordinate, float period, float half_texel, bool mirror, bool linear_repeat) {
    float normalized=coordinate/period;
    float folded=mirror ? 1.0f-abs(2.0f*fract(normalized*0.5f)-1.0f) : fract(normalized);
    float value=fma(folded,1.0f-4.0f*half_texel,2.0f*half_texel);
    return linear_repeat ? value : clamp(value,3.0f*half_texel,1.0f-3.0f*half_texel);
}
inline float2 rsx_texture_coordinate(float2 coordinate, constant rsx_texture_coordinates& p,float point_offset=0.0f) {
    coordinate=select(coordinate,abs(coordinate),bool2((p.flags & 2u) != 0u,(p.flags & 4u) != 0u));
    coordinate+=point_offset;
    float2 value=fma(coordinate,float2(p.scale_x,p.scale_y),float2(p.bias_x,p.bias_y));
    if (!(p.flags & 1u)) return value;
    float offset=(p.flags & 128u) ? 0.01f : 0.0f;
    value.x=(p.flags & 40u) ? rsx_periodic_coordinate(coordinate.x+offset,p.min_x,p.max_x,
        (p.flags & 32u)!=0u,(p.flags & 264u)==264u) : clamp(value.x,p.min_x,p.max_x);
    value.y=(p.flags & 80u) ? rsx_periodic_coordinate(coordinate.y+offset,p.min_y,p.max_y,
        (p.flags & 64u)!=0u,(p.flags & 272u)==272u) : clamp(value.y,p.min_y,p.max_y);
    return value;
}
inline float3 rsx_texture_coordinate(float3 coordinate, constant rsx_texture_coordinates& p) {
    return fma(coordinate,float3(p.scale_x,p.scale_y,p.scale_z),float3(p.bias_x,p.bias_y,p.bias_z));
}
)metal";
inline constexpr std::string_view stored_border_repeat_msl = R"metal(
struct rsx_repeat_span { float lower,upper,weight; };
inline rsx_repeat_span rsx_repeat_seam(float coordinate, float half_texel, bool repeat) {
    if (!repeat) return {coordinate,coordinate,0.0f};
    float first=3.0f*half_texel, last=1.0f-first;
    // The periodic seam is half a texel on each side of the logical image.
    if (coordinate<first) return {last,first,saturate((coordinate-half_texel)/(2.0f*half_texel))};
    if (coordinate>last) return {last,first,saturate((coordinate-last)/(2.0f*half_texel))};
    return {coordinate,coordinate,0.0f};
}
inline float4 rsx_repeat_tap(texture2d<float> image,sampler state,float2 uv,bool) {
    return image.sample(state,uv,level(0.0f));
}
inline float rsx_repeat_tap(depth2d<float> image,sampler state,float2 uv,bool) {
    return image.sample(state,uv,level(0.0f));
}
inline float rsx_repeat_tap(depth2d<float> image,sampler state,float2 uv,float reference) {
    return image.sample_compare(state,uv,reference,level(0.0f));
}
template<typename Image,typename Reference>
inline auto rsx_sample_stored_repeat(Image image,sampler state,float2 uv,constant rsx_texture_coordinates& p,Reference reference) {
    // One actual level. Keep sampling explicit in divergent seam branches.
    auto x=rsx_repeat_seam(uv.x,p.max_x,(p.flags & 264u)==264u);
    auto y=rsx_repeat_seam(uv.y,p.max_y,(p.flags & 272u)==272u);
    auto low=rsx_repeat_tap(image,state,float2(x.lower,y.lower),reference);
    if (x.weight>0.0f) low=mix(low,rsx_repeat_tap(image,state,float2(x.upper,y.lower),reference),x.weight);
    if (y.weight>0.0f) {
        auto high=rsx_repeat_tap(image,state,float2(x.lower,y.upper),reference);
        if (x.weight>0.0f) high=mix(high,rsx_repeat_tap(image,state,float2(x.upper,y.upper),reference),x.weight);
        low=mix(low,high,y.weight);
    }
    return low;
}
)metal";
inline constexpr std::string_view stored_border_mixed_filter_msl = R"metal(
template<typename Image>
inline bool rsx_stored_minifies(Image image,constant rsx_texture_coordinates& p,float2 dx,float2 dy,float bias) {
    // Derivatives use the guest interior footprint before wrapping or clamping.
    if (!(p.flags & 2048u)) {
        float2 logical=float2(image.get_width(),image.get_height())-2.0f;
        dx*=logical;
        dy*=logical;
    }
    return max(dot(dx,dx),dot(dy,dy))*exp2(2.0f*bias)>1.0f;
}
inline float4 rsx_stored_point(texture2d<float> image,sampler,float2 pixel,float2,bool) {
    return image.read(uint2(pixel),0);
}
inline float rsx_stored_point(depth2d<float> image,sampler,float2 pixel,float2,bool) {
    return image.read(uint2(pixel),0);
}
inline float rsx_stored_point(depth2d<float> image,sampler state,float2 pixel,float2 extent,float reference) {
    return rsx_repeat_tap(image,state,(pixel+0.5f)/extent,reference);
}
template<typename Image,typename Reference>
inline auto rsx_sample_stored_mixed(Image image,sampler state,float2 coordinate,constant rsx_texture_coordinates& p,Reference reference,bool minifies) {
    bool min_linear=(p.flags & 1024u)!=0u;
    if (minifies==min_linear) return rsx_sample_stored_repeat(image,state,rsx_texture_coordinate(coordinate,p),p,reference);
    // Use an integer read for ordinary point samples. Shadow comparisons
    // still use the native compare function at the selected texel center.
    float2 extent=float2(image.get_width(),image.get_height());
    float point_offset=(p.flags & 2048u) ? 0.01f : 0.0f;
    float2 pixel=floor(rsx_texture_coordinate(coordinate,p,point_offset)*extent);
    if (p.flags & 4096u) pixel.x=clamp(pixel.x,1.0f,extent.x-2.0f);
    if (p.flags & 8192u) pixel.y=clamp(pixel.y,1.0f,extent.y-2.0f);
    return rsx_stored_point(image,state,pixel,extent,reference);
}
)metal";
inline constexpr std::string_view stored_border_atlas_msl = R"metal(
inline rsx_repeat_span rsx_border_atlas_axis(float coordinate,float logical,uint wrap,bool linear,float point_bias) {
    // repeat, mirror, edge, border, clamp, then the three mirror-once modes.
    if (wrap>=5u) coordinate=abs(coordinate);
    coordinate+=point_bias;
    float padded=logical+2.0f;
    if (wrap<=1u) {
        float half_texel=0.5f/padded;
        bool seam=linear && wrap==0u;
        float uv=rsx_periodic_coordinate(coordinate,1.0f,half_texel,wrap==1u,seam);
        auto taps=rsx_repeat_seam(uv,half_texel,seam);
        return {taps.lower*padded,taps.upper*padded,taps.weight};
    }
    bool border=wrap==3u || wrap==6u;
    bool edge_blend=linear && (wrap==4u || wrap==7u);
    float inset=border ? 0.5f : edge_blend ? 1.0f : 1.5f;
    float pixel=clamp(fma(coordinate,logical,1.0f),inset,padded-inset);
    return {pixel,pixel,0.0f};
}
struct rsx_border_xy { rsx_repeat_span x,y; };
inline rsx_border_xy rsx_border_atlas_xy(float2 coordinate,uint4 atlas,uint mip,bool linear) {
    uint2 base=atlas.xy;
    uint2 logical=max(base>>mip,uint2(1));
    bool one_dimensional=(atlas.w & 4096u)!=0u;
    uint2 origin=0u;
    if (one_dimensional) origin.y=mip;
    else if (mip) {
        bool wide=base.x>=base.y;
        uint along=wide ? base.x : base.y;
        uint offset=0u;
        for (uint level=1u;level<mip;++level) offset+=max(along>>level,1u)+2u;
        origin=wide ? uint2(offset,base.y+2u) : uint2(base.x+2u,offset);
    }
    float2 base_scale=(atlas.w & 1024u) ? 1.0f/float2(base) : float2(1.0f);
    float point_bias=!linear && (atlas.w & 2048u) ? 0.01f : 0.0f;
    auto x=rsx_border_atlas_axis(coordinate.x*base_scale.x,float(logical.x),atlas.w & 7u,linear,point_bias*base_scale.x);
    auto y=one_dimensional ? rsx_repeat_span{0.5f,0.5f,0.0f} :
        rsx_border_atlas_axis(coordinate.y*base_scale.y,float(logical.y),(atlas.w>>3u) & 7u,linear,point_bias*base_scale.y);
    x.lower+=float(origin.x); x.upper+=float(origin.x);
    y.lower+=float(origin.y); y.upper+=float(origin.y);
    return {x,y};
}
template<typename Image,typename Reference>
inline auto rsx_border_atlas_level(Image image,sampler state,float2 coordinate,uint4 atlas,
                                  Reference reference,uint mip,bool linear) {
    const auto xy=rsx_border_atlas_xy(coordinate,atlas,mip,linear);
    const auto x=xy.x,y=xy.y;
    float2 extent=float2(image.get_width(),image.get_height());
    float2 lower=float2(x.lower,y.lower);
    if (!linear) return rsx_stored_point(image,state,floor(lower),extent,reference);
    float2 upper=float2(x.upper,y.upper);
    auto low=rsx_repeat_tap(image,state,lower/extent,reference);
    if (x.weight>0.0f) low=mix(low,rsx_repeat_tap(image,state,float2(upper.x,lower.y)/extent,reference),x.weight);
    if (y.weight>0.0f) {
        auto high=rsx_repeat_tap(image,state,float2(lower.x,upper.y)/extent,reference);
        if (x.weight>0.0f) high=mix(high,rsx_repeat_tap(image,state,upper/extent,reference),x.weight);
        low=mix(low,high,y.weight);
    }
    return low;
}
struct rsx_atlas_filter { uint first; float weight; bool linear; };
inline rsx_atlas_filter rsx_border_atlas_filter(uint4 atlas,constant rsx_sampler_parameters& sampling,float lod,bool minifying) {
    lod+=sampling.bias;
    bool linear=(atlas.w & (lod>0.0f || minifying ? 64u : 128u))!=0u;
    uint mip_filter=(atlas.w>>8u) & 3u;
    float level=mip_filter ? clamp(lod,sampling.min_lod,sampling.max_lod) : 0.0f;
    uint first=min(uint(floor(level+(mip_filter==1u ? 0.5f : 0.0f))),atlas.z-1u);
    return {first,mip_filter==2u && first+1u<atlas.z ? fract(level) : 0.0f,linear};
}
template<typename Image,typename Reference>
inline auto rsx_sample_border_atlas(Image image,sampler state,float2 coordinate,
                                  constant rsx_texture_coordinates& p,Reference reference,
                                  constant rsx_sampler_parameters& sampling,float lod,bool minifying=false) {
    const auto filter=rsx_border_atlas_filter(p.border_atlas,sampling,lod,minifying);
    auto value=rsx_border_atlas_level(image,state,coordinate,p.border_atlas,reference,filter.first,filter.linear);
    if (filter.weight>0.0f)
        value=mix(value,rsx_border_atlas_level(image,state,coordinate,p.border_atlas,reference,filter.first+1u,filter.linear),filter.weight);
    return value;
}
)metal";
inline constexpr std::string_view stored_border_volume_msl = R"metal(
struct rsx_border_z { rsx_repeat_span taps; float border; };
inline rsx_border_z rsx_border_volume_z(float coordinate,float logical,uint wrap,bool linear,float point_bias) {
    // A guest volume has stored XY rings. Its Z faces have no stored texel.
    if (wrap<=1u) {
        auto taps=rsx_border_atlas_axis(coordinate,logical,wrap,linear,point_bias);
        taps.lower-=1.0f; taps.upper-=1.0f;
        return {taps,0.0f};
    }
    if (wrap>=5u) coordinate=abs(coordinate);
    float pixel=(coordinate+point_bias)*logical;
    float edge=logical-0.5f;
    float border=0.0f;
    if (wrap==3u || wrap==6u)
        border=linear ? saturate(max(0.5f-pixel,pixel-edge)) : float(pixel<0.0f || pixel>=logical);
    pixel=clamp(pixel,0.5f,edge);
    return {{pixel,pixel,0.0f},border};
}
inline float4 rsx_border_volume_plane(texture3d<float> image,sampler state,rsx_border_xy xy,float z,float3 extent) {
    const auto x=xy.x,y=xy.y;
    auto low=image.sample(state,float3(x.lower,y.lower,z)/extent,level(0.0f));
    if (x.weight>0.0f) low=mix(low,image.sample(state,float3(x.upper,y.lower,z)/extent,level(0.0f)),x.weight);
    if (y.weight>0.0f) {
        auto high=image.sample(state,float3(x.lower,y.upper,z)/extent,level(0.0f));
        if (x.weight>0.0f) high=mix(high,image.sample(state,float3(x.upper,y.upper,z)/extent,level(0.0f)),x.weight);
        low=mix(low,high,y.weight);
    }
    return low;
}
inline float4 rsx_border_volume_level(texture3d<float> image,sampler state,float3 coordinate,
                                      constant rsx_texture_coordinates& p,float4 border,uint mip,bool linear) {
    const auto xy=rsx_border_atlas_xy(coordinate.xy,p.border_atlas,mip,linear);
    const float logical=float(max(p.border_atlas_depth>>mip,1u));
    const float z_scale=(p.border_atlas.w & 1024u) ? 1.0f/float(p.border_atlas_depth) : 1.0f;
    const float point_bias=!linear && (p.border_atlas.w & 2048u) ? 0.01f*z_scale : 0.0f;
    const auto z=rsx_border_volume_z(coordinate.z*z_scale,logical,(p.border_atlas.w>>13u)&7u,linear,point_bias);
    if (z.border>=1.0f) return border;
    const float3 lower=float3(xy.x.lower,xy.y.lower,z.taps.lower);
    if (!linear) return image.read(uint3(floor(lower)),0);
    const float3 extent=float3(image.get_width(),image.get_height(),image.get_depth());
    auto value=rsx_border_volume_plane(image,state,xy,z.taps.lower,extent);
    if (z.taps.weight>0.0f)
        value=mix(value,rsx_border_volume_plane(image,state,xy,z.taps.upper,extent),z.taps.weight);
    return z.border>0.0f ? mix(value,border,z.border) : value;
}
inline float4 rsx_sample_border_atlas(texture3d<float> image,sampler state,float3 coordinate,
                                     constant rsx_texture_coordinates& p,bool,
                                     constant rsx_sampler_parameters& sampling,float lod,bool minifying=false) {
    const auto filter=rsx_border_atlas_filter(p.border_atlas,sampling,lod,minifying);
    auto value=rsx_border_volume_level(image,state,coordinate,p,sampling.border_adjust,filter.first,filter.linear);
    if (filter.weight>0.0f)
        value=mix(value,rsx_border_volume_level(image,state,coordinate,p,sampling.border_adjust,filter.first+1u,filter.linear),filter.weight);
    return value;
}
)metal";
inline constexpr std::string_view stored_border_cube_msl = R"metal(
inline uint rsx_border_cube_face(float3 direction) {
    const float3 magnitude=abs(direction);
    if (magnitude.x>=max(magnitude.y,magnitude.z)) return direction.x<0.0f ? 1u : 0u;
    if (magnitude.y>=magnitude.z) return direction.y<0.0f ? 3u : 2u;
    return direction.z<0.0f ? 5u : 4u;
}
inline float3 rsx_border_cube_axes(float3 direction,uint face) {
    // Cube major-axis projection. The sign of the major component is kept
    // in this transform so its derivative is valid on either face.
    switch (face) {
    case 0: return float3(-direction.z,-direction.y,direction.x);
    case 1: return float3(direction.z,-direction.y,-direction.x);
    case 2: return float3(direction.x,direction.z,direction.y);
    case 3: return float3(direction.x,-direction.z,-direction.y);
    case 4: return float3(direction.x,-direction.y,direction.z);
    default: return float3(-direction.x,-direction.y,-direction.z);
    }
}
inline float4 rsx_border_cube_tap(texture2d_array<float> image,sampler state,float2 uv,uint face,bool) {
    return image.sample(state,uv,face,level(0.0f));
}
inline float rsx_border_cube_tap(depth2d_array<float> image,sampler state,float2 uv,uint face,bool) {
    return image.sample(state,uv,face,level(0.0f));
}
inline float rsx_border_cube_tap(depth2d_array<float> image,sampler state,float2 uv,uint face,float reference) {
    return image.sample_compare(state,uv,face,reference,level(0.0f));
}
inline float4 rsx_border_cube_point(texture2d_array<float> image,sampler,float2 pixel,float2,uint face,bool) {
    return image.read(uint2(pixel),face,0);
}
inline float rsx_border_cube_point(depth2d_array<float> image,sampler,float2 pixel,float2,uint face,bool) {
    return image.read(uint2(pixel),face,0);
}
inline float rsx_border_cube_point(depth2d_array<float> image,sampler state,float2 pixel,float2 extent,uint face,float reference) {
    return rsx_border_cube_tap(image,state,(pixel+0.5f)/extent,face,reference);
}
template<typename Image,typename Reference>
inline auto rsx_border_cube_level(Image image,sampler state,float2 uv,uint face,uint4 atlas,Reference reference,uint mip,bool linear) {
    const auto xy=rsx_border_atlas_xy(uv,atlas,mip,linear);
    const auto x=xy.x,y=xy.y;
    const float2 extent=float2(image.get_width(),image.get_height());
    const float2 lower=float2(x.lower,y.lower);
    if (!linear) return rsx_border_cube_point(image,state,floor(lower),extent,face,reference);
    auto low=rsx_border_cube_tap(image,state,lower/extent,face,reference);
    if (x.weight>0.0f)
        low=mix(low,rsx_border_cube_tap(image,state,float2(x.upper,y.lower)/extent,face,reference),x.weight);
    if (y.weight>0.0f) {
        auto high=rsx_border_cube_tap(image,state,float2(x.lower,y.upper)/extent,face,reference);
        if (x.weight>0.0f)
            high=mix(high,rsx_border_cube_tap(image,state,float2(x.upper,y.upper)/extent,face,reference),x.weight);
        low=mix(low,high,y.weight);
    }
    return low;
}
template<typename Image,typename Reference>
inline auto rsx_sample_border_cube(Image image,sampler state,float3 direction,constant rsx_texture_coordinates& p,
                                  Reference reference,constant rsx_sampler_parameters& sampling,float lod,bool minifying=false) {
    const uint face=rsx_border_cube_face(direction);
    const float3 axes=rsx_border_cube_axes(direction,face);
    const float2 uv=axes.z>0.0f ? saturate(fma(axes.xy/axes.z,0.5f,0.5f)) : float2(0.5f);
    const auto filter=rsx_border_atlas_filter(p.border_atlas,sampling,lod,minifying);
    auto value=rsx_border_cube_level(image,state,uv,face,p.border_atlas,reference,filter.first,filter.linear);
    if (filter.weight>0.0f)
        value=mix(value,rsx_border_cube_level(image,state,uv,face,p.border_atlas,reference,filter.first+1u,filter.linear),filter.weight);
    return value;
}
)metal";
inline constexpr std::string_view anisotropic_footprint_msl = R"metal(
struct rsx_filter_footprint { uint taps; float lod; bool major_x; };
inline rsx_filter_footprint rsx_texture_footprint(float px2,float py2,uint factor,float bias) {
    const bool major_x=px2>=py2;
    const float major2=max(px2,py2);
    if (factor<=1u || !(major2>1.0f)) return {1u,0.5f*log2(major2)+bias,major_x};
    const float major=sqrt(major2);
    const float minor=sqrt(min(px2,py2));
    // Reduce the logical LOD by the number of samples along the longer
    // screen-space derivative.
    const uint taps=uint(min(float(factor),ceil(major/max(minor,1.0f))));
    return {taps,0.5f*log2(major2)-log2(float(taps))+bias,major_x};
}
)metal";
inline constexpr std::string_view stored_border_anisotropic_msl = R"metal(
inline float2 rsx_border_texel_scale(constant rsx_texture_coordinates& p,float2) {
    return (p.border_atlas.w & 1024u) ? float2(1.0f) : float2(p.border_atlas.xy);
}
inline float3 rsx_border_texel_scale(constant rsx_texture_coordinates& p,float3) {
    return (p.border_atlas.w & 1024u) ? float3(1.0f) : float3(float2(p.border_atlas.xy),float(p.border_atlas_depth));
}
template<typename Image,typename Coordinate,typename Reference>
inline auto rsx_sample_border_anisotropic(Image image,sampler state,Coordinate coordinate,
                                          constant rsx_texture_coordinates& p,Reference reference,
                                          constant rsx_sampler_parameters& sampling,Coordinate dx,Coordinate dy,float bias) {
    const Coordinate scale=rsx_border_texel_scale(p,coordinate);
    const Coordinate tx=dx*scale,ty=dy*scale;
    const auto footprint=rsx_texture_footprint(dot(tx,tx),dot(ty,ty),(p.border_atlas.w>>17u)&31u,bias);
    if (footprint.taps==1u)
        return rsx_sample_border_atlas(image,state,coordinate,p,reference,sampling,footprint.lod);
    const Coordinate major=footprint.major_x ? dx : dy;
    auto value=rsx_sample_border_atlas(image,state,coordinate+major*(1.0f/float(footprint.taps+1u)-0.5f),
                                      p,reference,sampling,footprint.lod,true);
    for (uint tap=2u;tap<=footprint.taps;++tap)
        value+=rsx_sample_border_atlas(image,state,coordinate+major*(float(tap)/float(footprint.taps+1u)-0.5f),
                                       p,reference,sampling,footprint.lod,true);
    return value/float(footprint.taps);
}
)metal";
inline constexpr std::string_view stored_border_cube_anisotropic_msl = R"metal(
template<typename Image,typename Reference>
inline auto rsx_sample_border_cube_anisotropic(Image image,sampler state,float3 direction,
                                               constant rsx_texture_coordinates& p,Reference reference,
                                               constant rsx_sampler_parameters& sampling,float3 dx,float3 dy,float bias) {
    const uint face=rsx_border_cube_face(direction);
    const float3 axes=rsx_border_cube_axes(direction,face);
    if (!(axes.z>0.0f)) return rsx_sample_border_cube(image,state,direction,p,reference,sampling,bias);
    const float2 ratio=axes.xy/axes.z;
    const float3 gx=rsx_border_cube_axes(dx,face),gy=rsx_border_cube_axes(dy,face);
    const float scale=0.5f*float(p.border_atlas.x)/axes.z;
    const float2 tx=(gx.xy-ratio*gx.z)*scale,ty=(gy.xy-ratio*gy.z)*scale;
    const auto footprint=rsx_texture_footprint(dot(tx,tx),dot(ty,ty),(p.border_atlas.w>>17u)&31u,bias);
    if (footprint.taps==1u)
        return rsx_sample_border_cube(image,state,direction,p,reference,sampling,footprint.lod);
    const float3 major=footprint.major_x ? dx : dy;
    auto value=rsx_sample_border_cube(image,state,direction+major*(1.0f/float(footprint.taps+1u)-0.5f),
                                     p,reference,sampling,footprint.lod,true);
    for (uint tap=2u;tap<=footprint.taps;++tap)
        value+=rsx_sample_border_cube(image,state,direction+major*(float(tap)/float(footprint.taps+1u)-0.5f),
                                      p,reference,sampling,footprint.lod,true);
    return value/float(footprint.taps);
}
)metal";
} // namespace metal4
