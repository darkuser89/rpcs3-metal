#include "M4Sampling.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <stdexcept>

namespace metal4
{
namespace
{

bool float_sample_format(MTLPixelFormat format)
{
	// An explicit allowlist prevents integer/depth textures from reaching a
	// texture<float> function. Depth format reinterpretation needs the RSX
	// texture cache and is deliberately not inferred by this stage helper.
	switch (format)
	{
	case MTLPixelFormatA8Unorm:
	case MTLPixelFormatR8Unorm:
	case MTLPixelFormatR8Unorm_sRGB:
	case MTLPixelFormatR8Snorm:
	case MTLPixelFormatRG8Unorm:
	case MTLPixelFormatRG8Unorm_sRGB:
	case MTLPixelFormatRG8Snorm:
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatRGBA8Unorm_sRGB:
	case MTLPixelFormatRGBA8Snorm:
	case MTLPixelFormatBGRA8Unorm:
	case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatR16Unorm:
	case MTLPixelFormatR16Snorm:
	case MTLPixelFormatR16Float:
	case MTLPixelFormatRG16Unorm:
	case MTLPixelFormatRG16Snorm:
	case MTLPixelFormatRG16Float:
	case MTLPixelFormatRGBA16Unorm:
	case MTLPixelFormatRGBA16Snorm:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRG32Float:
	case MTLPixelFormatRGBA32Float:
	case MTLPixelFormatB5G6R5Unorm:
	case MTLPixelFormatA1BGR5Unorm:
	case MTLPixelFormatABGR4Unorm:
	case MTLPixelFormatBGR5A1Unorm:
	case MTLPixelFormatRGB10A2Unorm:
	case MTLPixelFormatBGR10A2Unorm:
	case MTLPixelFormatRG11B10Float:
	case MTLPixelFormatRGB9E5Float:
	case MTLPixelFormatBC1_RGBA:
	case MTLPixelFormatBC1_RGBA_sRGB:
	case MTLPixelFormatBC2_RGBA:
	case MTLPixelFormatBC2_RGBA_sRGB:
	case MTLPixelFormatBC3_RGBA:
	case MTLPixelFormatBC3_RGBA_sRGB:
	case MTLPixelFormatBC4_RUnorm:
	case MTLPixelFormatBC4_RSnorm:
	case MTLPixelFormatBC5_RGUnorm:
	case MTLPixelFormatBC5_RGSnorm:
	case MTLPixelFormatBC6H_RGBFloat:
	case MTLPixelFormatBC6H_RGBUfloat:
	case MTLPixelFormatBC7_RGBAUnorm:
	case MTLPixelFormatBC7_RGBAUnorm_sRGB:
		return true;
	default:
		return false;
	}
}

} // namespace

bool float_sampled_texture_format(MTLPixelFormat format)
{
	return float_sample_format(format);
}

void validate_sampled_texture(context& ctx, id<MTLTexture> texture, id<MTLSamplerState> sampler,
							  vertex_texture_dimension dimension, const sampler_parameters& parameters, bool shadow, bool depth,
							  bool cube_atlas, unsigned samples)
{
	validate_sampler_parameters(parameters);
	MTLTextureType expected;
	switch (dimension)
	{
	case vertex_texture_dimension::one:
	case vertex_texture_dimension::two:
		expected = MTLTextureType2D;
		break;
	case vertex_texture_dimension::three:
		expected = MTLTextureType3D;
		break;
	case vertex_texture_dimension::cube:
		expected = cube_atlas ? MTLTextureType2DArray : MTLTextureTypeCube;
		break;
	default:
		throw std::invalid_argument("Unknown Metal 4 sampled texture dimension");
	}
	if (samples != sampled_texture_samples(parameters) || (samples > 1 && dimension != vertex_texture_dimension::two))
		throw std::invalid_argument("Native multisample texture does not match its shader sampler state");
	if (sampled_texture_is_raster(parameters) &&
		(dimension != vertex_texture_dimension::two || cube_atlas || texture.mipmapLevelCount != 1))
		throw std::invalid_argument("MSAA sample raster requires a single-level 2D texture");
	if (samples > 1)
		expected = MTLTextureType2DMultisample;
	if (!texture || !sampler || texture.device != ctx.device() || sampler.device != ctx.device() ||
		texture.textureType != expected || (cube_atlas && (dimension != vertex_texture_dimension::cube || texture.arrayLength != 6)) ||
		texture.sampleCount != samples || texture.framebufferOnly ||
		texture.storageMode == MTLStorageModeMemoryless || !(texture.usage & MTLTextureUsageShaderRead) ||
		!(depth && dimension==vertex_texture_dimension::three ? texture.pixelFormat==MTLPixelFormatRG32Float : (shadow || depth) ? texture.pixelFormat == MTLPixelFormatDepth16Unorm || texture.pixelFormat == MTLPixelFormatDepth32Float || texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 : float_sample_format(texture.pixelFormat)) || (dimension == vertex_texture_dimension::one && texture.height != 1))
	{
		throw std::invalid_argument("Metal 4 RSX texture or sampler is incompatible with its compiled sampling binding");
	}
	if ((shadow && depth) || shadow != bool(parameters.flags & sampler_depth_compare) ||
		(shadow && dimension == vertex_texture_dimension::three) ||
		(shadow && texture.pixelFormat == MTLPixelFormatDepth16Unorm && (parameters.flags & sampler_depth_float)))
		throw std::invalid_argument("Native depth shader requires matching read/comparison/encoding metadata and supported dimension");
	if ((parameters.flags & sampler_opaque_black_border) && (texture.pixelFormat == MTLPixelFormatA1BGR5Unorm || texture.pixelFormat == MTLPixelFormatABGR4Unorm))
		throw std::invalid_argument("Packed A1BGR5/ABGR4 textures require shader emulation for opaque-black borders");
}

} // namespace metal4
