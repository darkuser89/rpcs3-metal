#pragma once
#include "M4Texture.h"

namespace metal4
{
// Shared preflight for linear and Morton surface transfers. Validates before
// staging conversion is encoded; these APIs support one 2D slice or cube face at a time.
void validate_surface_texture(context& ctx, MTLPixelFormat format, id<MTLTexture> texture, const texture_region& region);
}
