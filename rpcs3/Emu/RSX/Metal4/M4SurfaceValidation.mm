#include "M4SurfaceValidation.h"
#include <algorithm>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
void validate_surface_texture(context& ctx, MTLPixelFormat format, id<MTLTexture> texture, const texture_region& region)
{
	if (!texture || texture.device != ctx.device() || texture.pixelFormat != format || texture.sampleCount != 1 ||
		texture.framebufferOnly || texture.storageMode == MTLStorageModeMemoryless || region.size.depth != 1 || region.origin.z ||
		!region.size.width || !region.size.height || region.level >= texture.mipmapLevelCount ||
		region.level >= std::numeric_limits<std::size_t>::digits ||
		(texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray &&
		 texture.textureType != MTLTextureTypeCube && texture.textureType != MTLTextureTypeCubeArray))
		throw std::invalid_argument("Surface transfer needs a matching single-sample 2D slice or cube face");
	const auto width = std::max<std::size_t>(1, texture.width >> region.level), height = std::max<std::size_t>(1, texture.height >> region.level);
	const auto slices = texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray
		? checked_texture_product(texture.arrayLength, 6) : texture.arrayLength;
	if (region.slice >= slices || region.origin.x > width || region.size.width > width - region.origin.x ||
		region.origin.y > height || region.size.height > height - region.origin.y)
		throw std::out_of_range("Surface mip/slice/region is out of bounds");
}
}
