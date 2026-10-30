#pragma once

#include "M4Texture.h"
#include "M4TextureState.h"

namespace metal4
{
[[nodiscard]] __attribute__((ns_returns_retained)) inline id<MTLTexture> make_register_texture_view(id<MTLTexture> texture, const decoded_texture_state& state,
	texture_view_cache* cache = nullptr)
{
	if (!texture || !state.enabled)
		throw std::invalid_argument("Register texture view requires an active native source");
	// Depth remaps are compiled into MSL. Ordinary float views implement their
	// register remap before sign/gamma/BX2 texel processing.
	if (state.depth || state.view_channels == std::array<std::uint32_t, 4>{0, 1, 2, 3})
		return texture;
	constexpr std::array<MTLTextureSwizzle, 6> channels = {MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha, MTLTextureSwizzleZero, MTLTextureSwizzleOne};
	MTLTextureSwizzleChannels swizzle;
	swizzle.red = channels.at(state.view_channels[0]);
	swizzle.green = channels.at(state.view_channels[1]);
	swizzle.blue = channels.at(state.view_channels[2]);
	swizzle.alpha = channels.at(state.view_channels[3]);
	const auto slices = texture.textureType == MTLTextureTypeCube || texture.textureType == MTLTextureTypeCubeArray ? texture.arrayLength * 6 : texture.arrayLength;
	if (cache)
		return cache->get(texture, texture.pixelFormat, texture.textureType, NSMakeRange(0, texture.mipmapLevelCount), NSMakeRange(0, slices), swizzle);
	return make_texture_view(texture, texture.pixelFormat, texture.textureType, NSMakeRange(0, texture.mipmapLevelCount), NSMakeRange(0, slices), swizzle);
}
} // namespace metal4
