#pragma once

#include "M4DrawValidation.h"

#include <stdexcept>

namespace metal4
{
enum class depth_bounds_path { disabled, native_test, mirror };
// Metal treats [0,1] as disabling the native selector. RSX still rejects
// nonfinite/out-of-range stored float depth there, so that interval needs the
// mirror even on a device supporting native narrower intervals.
[[nodiscard]] inline depth_bounds_path plan_depth_bounds(bool enabled, float minimum, float maximum, bool native_supported)
{
	if (!enabled) return depth_bounds_path::disabled;
	validate_depth_bounds(minimum,maximum);
	return native_supported && (minimum != 0 || maximum != 1) ? depth_bounds_path::native_test : depth_bounds_path::mirror;
}
enum class depth_mirror_encoding
{
	none,
	unorm16,
	float32
};
struct depth_mirror_configuration
{
	depth_mirror_encoding encoding = depth_mirror_encoding::none;
	bool write = false;
	[[nodiscard]] bool enabled() const noexcept
	{
		return encoding != depth_mirror_encoding::none;
	}
	bool operator==(const depth_mirror_configuration&) const = default;
};
inline void validate_depth_mirror(const depth_mirror_configuration& config)
{
	if ((config.encoding != depth_mirror_encoding::none && config.encoding != depth_mirror_encoding::unorm16 && config.encoding != depth_mirror_encoding::float32) || (!config.enabled() && config.write))
		throw std::invalid_argument("Invalid native stored-depth mirror contract");
}
} // namespace metal4
