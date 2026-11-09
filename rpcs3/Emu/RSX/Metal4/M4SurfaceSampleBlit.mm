#include "M4SurfaceSampleBlit.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
namespace
{
texture_region expanded_region(texture_region region, std::uint32_t samples)
{
	if (samples == 1)
		return region;
	const auto layout = rsx_sample_layout(samples);
	region.origin.x = checked_texture_product(region.origin.x, layout.x);
	region.origin.y = checked_texture_product(region.origin.y, layout.y);
	region.size.width = checked_texture_product(region.size.width, layout.x);
	region.size.height = checked_texture_product(region.size.height, layout.y);
	return region;
}
id<MTLTexture> sample_window_texture(context& ctx, id<MTLTexture> texture, MTLSize size)
{
	if (size.width > max_2d_texture_dimension(ctx.device()) || size.height > max_2d_texture_dimension(ctx.device()))
		throw std::invalid_argument("Native expanded sample blit exceeds Apple texture dimensions");
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat width:size.width height:size.height mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	return make_texture(ctx, descriptor);
}
} // namespace
surface_sample_blitter::surface_sample_blitter(context& ctx) : m_samples(ctx), m_blits(ctx), m_typeless(ctx) {}
void surface_sample_blitter::nearest(context& ctx, surface_format sf, id<MTLTexture> source, const texture_region& sr,
									 surface_format df, id<MTLTexture> destination, const texture_region& dr, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	if (!source || !destination || source.pixelFormat != surface_native_format(sf) || destination.pixelFormat != surface_native_format(df) ||
		(!typeless && sf != df) || (typeless && plane != texture_plane::all) || flip[2] || sr.size.depth != 1 || dr.size.depth != 1 || sr.origin.z || dr.origin.z)
		throw std::invalid_argument("Native sample blit needs matching surface descriptors and 2D regions");
	for (const auto& texture : {source, destination})
		if ((texture.sampleCount != 1 && texture.sampleCount != 2 && texture.sampleCount != 4) ||
			(texture.sampleCount > 1 && texture.textureType != MTLTextureType2DMultisample) ||
			(texture.sampleCount == 1 && texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray))
			throw std::invalid_argument("Native surface sample blit supports 1/2/4 samples and 2D textures");
	(void)validate_texture_region(ctx, source, sr, plane);
	(void)validate_texture_region(ctx, destination, dr, plane);
	const bool direct = sf == df && source.sampleCount == destination.sampleCount && !flip[0] && !flip[1] && plane == texture_plane::all &&
						sr.size.width == dr.size.width && sr.size.height == dr.size.height && source.width == destination.width && source.height == destination.height;
	if (direct)
	{
		copy_texture(ctx, source, sr, destination, dr);
		return;
	}
	const auto source_region = expanded_region(sr, static_cast<std::uint32_t>(source.sampleCount));
	const auto target_region = expanded_region(dr, static_cast<std::uint32_t>(destination.sampleCount));
	if (typeless)
		(void)plan_typeless(sf, df, source_region.size.width, source_region.size.height, target_region.size.width, target_region.size.height);
	const auto input = source.sampleCount == 1 ? source : sample_window_texture(ctx, source, source_region.size);
	const auto output = destination.sampleCount == 1 ? destination : sample_window_texture(ctx, destination, target_region.size);
	auto input_region = source_region, output_region = target_region;
	if (source.sampleCount > 1)
	{
		m_samples.expand_region(ctx, source, input, {source_region.origin.x, source_region.origin.y}, plane);
		input_region.origin = MTLOriginMake(0, 0, 0);
	}
	if (destination.sampleCount > 1)
		output_region.origin = MTLOriginMake(0, 0, 0);
	const bool full = dr.origin.x == 0 && dr.origin.y == 0 && dr.size.width == destination.width && dr.size.height == destination.height;
	if (typeless)
		m_typeless.nearest(ctx, sf, input, input_region, df, output, output_region, flip);
	else
		m_blits.nearest(ctx, input, input_region, output, output_region, plane, flip);
	if (destination.sampleCount > 1)
	{
		// A partial import loads the selected attachment and discards samples
		// outside this window. Never decode or re-export its untouched background.
		if (full) m_samples.import(ctx, output, destination, plane);
		else m_samples.import_region(ctx, output, destination, {target_region.origin.x, target_region.origin.y}, plane);
	}
}
} // namespace metal4
