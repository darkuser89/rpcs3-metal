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
inline constexpr std::uint32_t texture_coordinate_clamp = 1;
// Native fragment buffer4. Guest/register/cache adapter supplies effective
// scale/bias/subregion clamps; native image dimensions alone cannot infer them.
struct alignas(16) texture_coordinates
{
	std::array<float, 3> scale{1, 1, 1};
	std::uint32_t flags = 0;
	std::array<float, 3> bias{};
	std::uint32_t reserved = 0;
	std::array<float, 2> minimum{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
	std::array<float, 2> maximum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
};
static_assert(sizeof(texture_coordinates) == 48 && offsetof(texture_coordinates, flags) == 12 && offsetof(texture_coordinates, bias) == 16 && offsetof(texture_coordinates, minimum) == 32 && offsetof(texture_coordinates, maximum) == 40);
inline void validate_texture_coordinates(const texture_coordinates& value)
{
	if ((value.flags & ~texture_coordinate_clamp) || value.reserved)
		throw std::invalid_argument("Invalid texture coordinate flags/reserved field");
	for (unsigned i = 0; i < 3; ++i)
		if (!std::isfinite(value.scale[i]) || !std::isfinite(value.bias[i]))
			throw std::invalid_argument("Nonfinite texture coordinate transform");
	for (unsigned i = 0; i < 2; ++i)
		if (!std::isfinite(value.minimum[i]) || !std::isfinite(value.maximum[i]) || value.minimum[i] > value.maximum[i])
			throw std::invalid_argument("Invalid texture coordinate clamp range");
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
    float bias_x, bias_y, bias_z; uint reserved;
    float min_x, min_y, max_x, max_y;
};
inline float rsx_texture_coordinate(float coordinate, constant rsx_texture_coordinates& p) {
    float value=fma(coordinate,p.scale_x,p.bias_x);
    return (p.flags & 1u) ? clamp(value,p.min_x,p.max_x) : value;
}
inline float2 rsx_texture_coordinate(float2 coordinate, constant rsx_texture_coordinates& p) {
    float2 value=fma(coordinate,float2(p.scale_x,p.scale_y),float2(p.bias_x,p.bias_y));
    return (p.flags & 1u) ? clamp(value,float2(p.min_x,p.min_y),float2(p.max_x,p.max_y)) : value;
}
inline float3 rsx_texture_coordinate(float3 coordinate, constant rsx_texture_coordinates& p) {
    return fma(coordinate,float3(p.scale_x,p.scale_y,p.scale_z),float3(p.bias_x,p.bias_y,p.bias_z));
}
)metal";
} // namespace metal4
