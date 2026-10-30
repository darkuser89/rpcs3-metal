#pragma once

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
struct alignas(16) sampler_parameters
{
	float min_lod = 0, max_lod = std::numeric_limits<float>::max(), bias = 0;
	std::uint32_t flags = 0;
	std::array<float, 4> border_adjust{};
};
static_assert(sizeof(sampler_parameters) == 32 && offsetof(sampler_parameters, flags) == 12);
inline void validate_sampler_parameters(const sampler_parameters& value)
{
	if (!std::isfinite(value.min_lod) || !std::isfinite(value.max_lod) || !std::isfinite(value.bias) || value.min_lod < 0 || value.max_lod < value.min_lod || value.bias < -16 || value.bias > 15.999f || (value.flags & ~(sampler_opaque_black_border | sampler_depth_compare | sampler_depth_float | sampler_custom_border)) || ((value.flags & sampler_depth_float) && !(value.flags & sampler_depth_compare)))
		throw std::invalid_argument("Invalid native shader sampler parameters");
	for (float channel : value.border_adjust)
		if (!std::isfinite(channel) || (!(value.flags & sampler_custom_border) && channel != 0))
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
