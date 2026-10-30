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
inline float4 rsx_depth_texel(float depth,uint4 channels) {
    return select(select(float4(depth),float4(0),channels==1u),float4(1),channels==2u);
}
)metal";
} // namespace metal4
