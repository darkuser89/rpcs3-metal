#pragma once
#include <string>

namespace metal4
{
enum class sample_aspect
{
	color,
	depth,
	depth_stencil,
	stencil
};
[[nodiscard]] std::string generate_sample_shader(sample_aspect aspect, bool import_samples);
} // namespace metal4
