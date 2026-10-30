#pragma once

#include "M4StereoPlan.h"
#include "Utilities/stereo_config.h"

namespace metal4
{
// Only copies the existing configuration contract. No shared shader, Vulkan
// object or borrowed configuration/matrix storage reaches the Metal pass.
inline stereo_output capture_rsx_stereo(stereo_config& config, bool enabled)
{
	return capture_stereo_configuration(config, enabled);
}
} // namespace metal4
