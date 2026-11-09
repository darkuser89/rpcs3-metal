#pragma once

#include "M4StereoPlan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace metal4
{
struct frame_extent
{
	std::uint32_t width = 0, height = 0;
	bool operator==(const frame_extent&) const = default;
};
struct frame_rectangle
{
	std::uint32_t x = 0, y = 0, width = 0, height = 0;
	bool operator==(const frame_rectangle&) const = default;
};
struct frame_geometry
{
	double width = 0, height = 0, scale = 1, x = 0, y = 0;
	bool visible = false;
	bool operator==(const frame_geometry&) const = default;
};
struct frame_geometry_statistics
{
	std::uint64_t epoch = 0, layer_transactions = 0, vsync_writes = 0;
};
enum class frame_pacing
{
	unsynchronized,
	synchronized,
	adaptive,
};
[[nodiscard]] inline bool frame_display_sync(frame_pacing pacing)
{
	switch (pacing)
	{
	case frame_pacing::unsynchronized: return false;
	case frame_pacing::synchronized: return true;
	// CAMetalLayer exposes a sync toggle. Keep sync enabled for adaptive
	// requests until a native late-frame presentation path is available.
	case frame_pacing::adaptive: return true;
	}
	throw std::invalid_argument("Invalid native frame pacing mode");
}
struct frame_policy
{
	frame_extent virtual_display;
	double display_aspect = 16.0 / 9;
	bool stretch = false;
	frame_pacing pacing = frame_pacing::synchronized;
	bool linear = true;
	float gamma = 1;
	bool limited_rgb = false;
	stereo_output stereo;
	bool fsr = false;
	float fsr_sharpness = 1;
};
inline frame_extent plan_drawable_extent(double width, double height, double scale, std::uint32_t limit)
{
	if (!std::isfinite(width) || !std::isfinite(height) || !std::isfinite(scale) || width < 0 || height < 0 || scale <= 0 || !limit)
		throw std::invalid_argument("Invalid native frame geometry/scale");
	const double w = std::ceil(width * scale), h = std::ceil(height * scale);
	if (!std::isfinite(w) || !std::isfinite(h) || w > limit || h > limit)
		throw std::out_of_range("Native drawable geometry exceeds device limits");
	return {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
}
inline void validate_frame_policy(const frame_policy& policy)
{
	validate_stereo_output(policy.stereo);
	(void)frame_display_sync(policy.pacing);
	if (!std::isfinite(policy.display_aspect) || policy.display_aspect <= 0 || !policy.virtual_display.width || !policy.virtual_display.height ||
		!std::isfinite(policy.gamma) || policy.gamma <= 0 || !std::isfinite(policy.fsr_sharpness) || policy.fsr_sharpness < 0 || policy.fsr_sharpness > 2)
		throw std::invalid_argument("Invalid native frame AV policy");
}

template <typename Av>
[[nodiscard]] frame_policy capture_frame_policy(const Av& av, bool stretch, frame_pacing pacing, bool linear, bool full_rgb)
{
	const auto size = av.video_frame_size();
	frame_policy policy{{size.width, size.height}, av.get_aspect_ratio(), stretch, pacing, linear, float(av.gamma), !full_rgb};
	validate_frame_policy(policy);
	return policy;
}
// Same two-stage fit as the VK AV path: source into virtual display,then
// virtual AV aspect into the physical drawable. Preserve truncation at each
// integer virtual fit,not an approximate source-aspect-only shortcut.
inline frame_rectangle plan_frame_rectangle(frame_extent source, frame_extent target, const frame_policy& policy)
{
	validate_frame_policy(policy);
	if (!source.width || !source.height || !target.width || !target.height)
		return {};
	if (policy.stretch)
		return {0, 0, target.width, target.height};
	const auto fit = [](frame_extent extent, double aspect)
	{
		frame_rectangle result{0, 0, extent.width, extent.height};
		const double ratio = aspect / (double(extent.width) / extent.height);
		if (ratio > 1)
		{
			result.height = static_cast<std::uint32_t>(extent.height / ratio);
			result.y = (extent.height - result.height) / 2;
		}
		else if (ratio < 1)
		{
			result.width = static_cast<std::uint32_t>(extent.width * ratio);
			result.x = (extent.width - result.width) / 2;
		}
		return result;
	};
	const auto inner = fit(policy.virtual_display, double(source.width) / source.height);
	const auto outer = fit(target, policy.display_aspect);
	const double sx = double(outer.width) / policy.virtual_display.width, sy = double(outer.height) / policy.virtual_display.height;
	const auto x1 = static_cast<std::uint32_t>(inner.x * sx), y1 = static_cast<std::uint32_t>(inner.y * sy);
	const auto x2 = static_cast<std::uint32_t>((std::uint64_t(inner.x) + inner.width) * sx);
	const auto y2 = static_cast<std::uint32_t>((std::uint64_t(inner.y) + inner.height) * sy);
	return {outer.x + x1, outer.y + y1, x2 - x1, y2 - y1};
}
} // namespace metal4
