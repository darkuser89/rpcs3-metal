#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace metal4
{
enum class stereo_mode : std::uint32_t { disabled, side_by_side, over_under, interlaced, anaglyph };
struct stereo_output
{
	stereo_mode mode = stereo_mode::disabled;
	// Columns multiply source R,G,B respectively; the fourth lane is padding.
	std::array<std::array<float, 4>, 3> left{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}}, right{};
	bool operator==(const stereo_output&) const = default;
};
inline void validate_stereo_output(const stereo_output& output)
{
	if (output.mode > stereo_mode::anaglyph) throw std::invalid_argument("Unknown native stereo output mode");
	for (const auto* columns : {&output.left, &output.right})
		for (const auto& column : *columns)
			for (unsigned c = 0; c < 4; ++c)
				if (!std::isfinite(column[c]) || (c == 3 && column[c] != 0))
					throw std::invalid_argument("Invalid native stereo matrix coefficient/padding");
}
template <typename Configuration>
[[nodiscard]] stereo_output capture_stereo_configuration(Configuration& config, bool enabled)
{
	config.update_from_config(enabled);
	using mode = decltype(config.stereo_mode());
	stereo_output output;
	switch (config.stereo_mode())
	{
	case mode::disabled: break;
	case mode::side_by_side: output.mode = stereo_mode::side_by_side; break;
	case mode::over_under: output.mode = stereo_mode::over_under; break;
	case mode::interlaced: output.mode = stereo_mode::interlaced; break;
	case mode::anaglyph_red_green:
	case mode::anaglyph_red_blue:
	case mode::anaglyph_red_cyan:
	case mode::anaglyph_magenta_cyan:
	case mode::anaglyph_trioscopic:
	case mode::anaglyph_amber_blue:
	case mode::anaglyph_custom: output.mode = stereo_mode::anaglyph; break;
	default: throw std::invalid_argument("Unknown RSX stereo configuration");
	}
	const auto& matrices = config.matrices();
	for (unsigned column = 0; column < 3; ++column)
		for (unsigned c = 0; c < 3; ++c)
		{
			output.left[column][c] = matrices.left[column].rgb[c];
			output.right[column][c] = matrices.right[column].rgb[c];
		}
	validate_stereo_output(output);
	return output;
}
} // namespace metal4
