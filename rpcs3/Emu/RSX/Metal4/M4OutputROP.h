#pragma once

namespace metal4
{
struct output_rop_configuration
{
	bool round_to_8bit = false;
	bool srgb = false;
	bool polygon_stipple = false;
	// The VK shader path uses stochastic discard before blend/remap. This is
	// independent of the native multisample alpha-to-coverage pipeline state.
	bool stochastic_alpha_coverage = false;
	[[nodiscard]] bool enabled() const noexcept
	{
		return round_to_8bit || srgb || polygon_stipple || stochastic_alpha_coverage;
	}
	bool operator==(const output_rop_configuration&) const = default;
};
} // namespace metal4
