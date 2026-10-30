#pragma once

#include "M4Device.h"
#include "M4VertexProgram.h"
#include "M4SamplerParameters.h"

namespace metal4
{

[[nodiscard]] bool float_sampled_texture_format(MTLPixelFormat format);

// Ordinary float or explicitly compiled depth read/comparison sampling.
// Integer, redirected depth and multisample need dedicated emulation.
// Samplers must have argumentBufferSupport,
// as provided by make_sampler(). RSX 1D mip chains use height-one 2D images.
void validate_sampled_texture(context& ctx, id<MTLTexture> texture, id<MTLSamplerState> sampler,
							  vertex_texture_dimension dimension, const sampler_parameters& parameters = {}, bool shadow = false, bool depth = false);

} // namespace metal4
