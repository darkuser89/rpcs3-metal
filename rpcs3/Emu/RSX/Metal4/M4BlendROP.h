#pragma once

#include "M4RenderState.h"
#include "M4SurfaceLayout.h"

#include <algorithm>
#include <stdexcept>

namespace metal4
{
// Shader constants exclude write masks and blend constants: masks remain
// attachment state, while completion-owned fragment uniforms carry constants.
struct blend_rop_configuration
{
	std::uint32_t sample_count = 1;
	std::array<blend_state, 4> colors{};
	std::array<bool, 4> normalized{};
	std::array<std::array<std::uint32_t, 4>, 4> bits{};
	std::array<bool, 4> half_float{};
	std::array<color_export_remap, 4> remap{};
	[[nodiscard]] bool enabled() const noexcept
	{
		return std::any_of(colors.begin(), colors.end(), [](const auto& color)
						   {
							   return color.enabled;
						   });
	}
	bool operator==(const blend_rop_configuration&) const = default;
};

inline void validate_blend_rop(const blend_rop_configuration& config)
{
	if (config.sample_count != 1 && config.sample_count != 2 && config.sample_count != 4 && config.sample_count != 8)
		throw std::invalid_argument("Invalid native blend ROP sample count");
	for (unsigned i = 0; i < 4; ++i)
	{
		if (config.remap[i] > color_export_remap::rgb_one)
			throw std::invalid_argument("Invalid native blend ROP channel remap");
		const auto& c = config.colors[i];
		if (!c.enabled)
			continue;
		for (auto width : config.bits[i])
			if (width > 16)
				throw std::invalid_argument("Invalid blend ROP storage channel width");
		const bool has_bits = std::any_of(config.bits[i].begin(), config.bits[i].end(), [](auto width)
										  {
											  return width != 0;
										  });
		if (has_bits != config.normalized[i] || (has_bits && config.half_float[i]))
			throw std::invalid_argument("Inconsistent blend ROP storage encoding");
		if (c.source_rgb > blend_factor::one_minus_constant_alpha || c.destination_rgb > blend_factor::one_minus_constant_alpha ||
			c.source_alpha > blend_factor::one_minus_constant_alpha || c.destination_alpha > blend_factor::one_minus_constant_alpha ||
			c.operation_rgb > blend_operation::reverse_add_signed || c.operation_alpha > blend_operation::reverse_add_signed)
			throw std::invalid_argument("Invalid native blend ROP operation or factor");
	}
}
} // namespace metal4
