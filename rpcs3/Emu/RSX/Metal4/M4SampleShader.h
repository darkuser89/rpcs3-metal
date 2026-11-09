#pragma once
#include <string>
#include <string_view>

namespace metal4
{
enum class sample_aspect
{
	color,
	depth,
	depth_stencil,
	stencil
};
inline constexpr std::string_view sample_projection_msl = R"metal(
struct sample_parameters {
    uint samples_x, samples_y, origin_x, origin_y, width, height, target_x, target_y;
    uint source_width, source_height, copy_width, copy_height, copy_x, copy_y, flip, row_elements;
};
struct sample_address { uint2 pixel; uint sample; };
inline sample_address locate_sample(uint2 pixel, constant sample_parameters& p) {
    const uint2 local = pixel + uint2(p.copy_x, p.copy_y);
    uint2 offset = local;
    if (p.flip || p.source_width != p.copy_width || p.source_height != p.copy_height) {
        offset.x = rpcs3_nearest_axis(local.x, p.source_width, p.copy_width, bool(p.flip & 1u));
        offset.y = rpcs3_nearest_axis(local.y, p.source_height, p.copy_height, bool(p.flip & 2u));
    }
    const uint2 position = offset + uint2(p.origin_x, p.origin_y);
    const uint2 layout(p.samples_x, p.samples_y);
    const uint2 lane = position % layout;
    return {position / layout, lane.x + lane.y * layout.x};
}
)metal";
[[nodiscard]] std::string generate_sample_shader(sample_aspect aspect, bool import_samples,
	bool single_sample_source = false, bool half_color = false);
} // namespace metal4
