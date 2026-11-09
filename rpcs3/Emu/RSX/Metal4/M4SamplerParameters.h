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
inline constexpr std::uint32_t sampler_opaque_black_border = 1;
inline constexpr std::uint32_t sampler_depth_compare = 2;
inline constexpr std::uint32_t sampler_depth_float = 4;
inline constexpr std::uint32_t sampler_custom_border = 8;
inline constexpr std::uint32_t sampler_volume_border = 16;
inline constexpr std::uint32_t sampler_redirected_depth = 32;
inline constexpr std::uint32_t sampler_redirected_float = 64;
inline constexpr unsigned sampler_point_wrap_shift = 7;
inline constexpr unsigned sampler_point_channel_shift = 13;
inline constexpr std::uint32_t sampler_multisampled = 1u << 25;
inline constexpr std::uint32_t sampler_four_samples = 1u << 26;
inline constexpr unsigned sampler_point_compare_shift = 27;
inline constexpr std::uint32_t sampler_msaa_mag_linear = 1u << 30;
inline constexpr std::uint32_t sampler_msaa_min_linear = 1u << 31;
inline constexpr auto sampler_msaa_linear_mask = sampler_msaa_mag_linear | sampler_msaa_min_linear;
struct alignas(16) sampler_parameters
{
	// Multisample images have no mip levels. In that binding max_lod carries
	// the anisotropy limit instead of a mip clamp.
	float min_lod = 0, max_lod = std::numeric_limits<float>::max(), bias = 0;
	std::uint32_t flags = 0;
	std::array<float, 4> border_adjust{};
};
static_assert(sizeof(sampler_parameters) == 32 && offsetof(sampler_parameters, flags) == 12);
[[nodiscard]] inline unsigned sampled_texture_samples(const sampler_parameters& value)
{
	return !(value.flags & sampler_multisampled) ? 1 : value.flags & sampler_four_samples ? 4 : 2;
}
[[nodiscard]] inline bool sampled_texture_is_raster(const sampler_parameters& value)
{
	// With the multisample bit clear, the count bit denotes a 2D sample raster.
	return (value.flags & (sampler_multisampled | sampler_four_samples)) == sampler_four_samples;
}
inline void validate_sampler_parameters(const sampler_parameters& value)
{
	const bool redirected = value.flags & sampler_redirected_depth;
	const bool msaa_sampling = (value.flags & sampler_multisampled) || sampled_texture_is_raster(value);
	constexpr auto normal_flags = sampler_opaque_black_border | sampler_depth_compare | sampler_depth_float | sampler_custom_border | sampler_volume_border;
	constexpr auto point_flags = ((1u << (sampler_point_channel_shift + 12)) - 1u) & ~normal_flags;
	constexpr auto msaa_flags = sampler_multisampled | sampler_four_samples | (7u << sampler_point_compare_shift) | sampler_msaa_linear_mask;
	const auto allowed = msaa_sampling ? point_flags | msaa_flags | sampler_depth_compare | sampler_depth_float :
		redirected ? point_flags : normal_flags;
	if (!std::isfinite(value.min_lod) || !std::isfinite(value.max_lod) || !std::isfinite(value.bias) || value.min_lod < 0 || value.max_lod < value.min_lod || value.bias < -16 || value.bias > 15.999f ||
		(value.flags & ~allowed) || (!redirected && (value.flags & sampler_redirected_float)) ||
		(redirected && (value.flags & sampler_msaa_linear_mask)) ||
		((value.flags & sampler_depth_float) && !(value.flags & sampler_depth_compare)))
		throw std::invalid_argument("Invalid native shader sampler parameters");
	constexpr std::array anisotropy{1.f, 2.f, 4.f, 6.f, 8.f, 10.f, 12.f, 16.f};
	if (msaa_sampling && (value.min_lod != 0 || (redirected && value.max_lod != 1) ||
		std::find(anisotropy.begin(), anisotropy.end(), value.max_lod) == anisotropy.end()))
		throw std::invalid_argument("Invalid native multisample anisotropy limit");
	if (redirected || msaa_sampling)
		for (unsigned i = 0; i < 4; ++i)
			if (((value.flags >> (sampler_point_channel_shift + 3 * i)) & 7u) > 5)
				throw std::invalid_argument("Invalid point-sampled texture channel");
	for (float channel : value.border_adjust)
		if (!std::isfinite(channel) || (!msaa_sampling && !(value.flags & (sampler_custom_border | sampler_volume_border | sampler_redirected_depth)) && channel != 0))
			throw std::invalid_argument("Invalid native border adjustment");
}
inline constexpr std::string_view sampler_parameters_msl = R"metal(
struct rsx_sampler_parameters { float min_lod; float max_lod; float bias; uint flags; float4 border_adjust; };
inline float4 rsx_border_color(float4 value, float weight, constant rsx_sampler_parameters& parameters) {
    return value + weight * parameters.border_adjust;
}
inline float rsx_shadow_reference(float value, constant rsx_sampler_parameters& parameters) {
    return (parameters.flags & 4u) ? value : min(value, 1.0f);
}
)metal";
} // namespace metal4
