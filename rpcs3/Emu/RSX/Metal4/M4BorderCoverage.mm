#include "M4BorderCoverage.h"
#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4Sampling.h"
#include "M4Texture.h"
#include <algorithm>
#include <vector>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif
namespace metal4
{
namespace
{
bool same_shape(id<MTLTexture> a, id<MTLTexture> b)
{
	return a.textureType == b.textureType && a.width == b.width && a.height == b.height &&
		   a.depth == b.depth && a.mipmapLevelCount == b.mipmapLevelCount && a.arrayLength == b.arrayLength;
}
} // namespace
border_coverage_cache::border_coverage_cache(context& ctx, std::shared_ptr<heap_budget> budget, std::size_t capacity)
	: m_context(ctx), m_budget(std::move(budget)), m_capacity(capacity)
{
	if (!m_budget || m_budget->device() != ctx.device() || !capacity || capacity > 4096)
		throw std::invalid_argument("Native border coverage needs a matching budget and bounded capacity");
}
id<MTLTexture> border_coverage_cache::get(id<MTLTexture> source)
{
	if (!source || source.device != m_context.device() || source.sampleCount != 1 || source.arrayLength != 1 ||
		(source.textureType != MTLTextureType2D && source.textureType != MTLTextureType3D && source.textureType != MTLTextureTypeCube))
		throw std::invalid_argument("Native border coverage source shape is unsupported");
	for (auto it = m_images.begin(); it != m_images.end(); ++it)
		if (same_shape(source, *it))
		{
			const auto result = *it;
			m_images.splice(m_images.begin(), m_images, it);
			return result;
		}
	const auto image = build(source, MTLPixelFormatR32Float);
	// Publish only initialized images. Caller/submission owners outlive eviction.
	m_images.push_front(image);
	if (m_images.size() > m_capacity)
		m_images.pop_back();
	return image;
}
void border_coverage_cache::initialize(unsigned kind)
{
	if (!m_clear[kind])
	{
		helper_catalog compile(m_context);
		const auto library = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void zero2(texture2d<float,access::write> image [[texture(0)]], uint2 p [[thread_position_in_grid]]) { image.write(float4(0),p); }
kernel void zero3(texture3d<float,access::write> image [[texture(0)]], uint3 p [[thread_position_in_grid]]) { image.write(float4(0),p); }
kernel void zerocube(texture2d_array<float,access::write> image [[texture(0)]], uint3 p [[thread_position_in_grid]]) { image.write(float4(0),p.xy,p.z); }
kernel void copy2(texture2d<float,access::read> source [[texture(1)]],texture2d<float,access::write> image [[texture(0)]], uint2 p [[thread_position_in_grid]]) { image.write(source.read(p),p); }
kernel void copy3(texture3d<float,access::read> source [[texture(1)]],texture3d<float,access::write> image [[texture(0)]], uint3 p [[thread_position_in_grid]]) { image.write(source.read(p),p); }
kernel void copycube(texture2d_array<float,access::read> source [[texture(1)]],texture2d_array<float,access::write> image [[texture(0)]], uint3 p [[thread_position_in_grid]]) { image.write(source.read(p.xy,p.z),p.xy,p.z); }
)metal",
											 @"RPCS3 border coverage and float snapshot");

		const auto clear = compile.compute(library, std::array<NSString*, 3>{@"zero2", @"zero3", @"zerocube"}[kind]);
		const auto copy = compile.compute(library, std::array<NSString*, 3>{@"copy2", @"copy3", @"copycube"}[kind]);
		m_clear[kind] = clear;
		m_copy[kind] = copy;
	}
}
id<MTLTexture> border_coverage_cache::build(id<MTLTexture> source, MTLPixelFormat format, id<MTLTexture> input)
{
	const unsigned kind = source.textureType == MTLTextureType3D ? 1 : source.textureType == MTLTextureTypeCube ? 2
																												: 0;
	initialize(kind);
	auto descriptor = [MTLTextureDescriptor new];
	descriptor.textureType = source.textureType;
	descriptor.pixelFormat = format;
	descriptor.width = source.width;
	descriptor.height = source.height;
	descriptor.depth = source.depth;
	descriptor.mipmapLevelCount = source.mipmapLevelCount;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
	const auto image = m_budget->texture(descriptor);
	const auto pipeline = input ? m_copy[kind] : m_clear[kind];
	bool begun = false;
	try
	{
		m_context.begin();
		begun = true;
		for (NSUInteger level = 0; level < image.mipmapLevelCount; ++level)
		{
			const auto type = kind == 2 ? MTLTextureType2DArray : image.textureType;
			const auto view = make_texture_view(image, image.pixelFormat, type, NSMakeRange(level, 1), NSMakeRange(0, kind == 2 ? 6 : 1),
												MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha));
			bindings resources;
			resources.set_texture(0, view);
			if (input)
			{
				const auto src = make_texture_view(input, input.pixelFormat, type, NSMakeRange(level, 1), NSMakeRange(0, kind == 2 ? 6 : 1),
												   input.swizzle);
				resources.set_texture(1, src);
			}
			const auto table = resources.snapshot(m_context);
			m_context.keep_alive(pipeline);
			const auto encoder = m_context.compute();
			[encoder setComputePipelineState:pipeline];
			[encoder setArgumentTable:table];
			[encoder dispatchThreads:MTLSizeMake(view.width, view.height, kind == 2 ? 6 : view.depth) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
		}
		m_context.submit().wait();
	}
	catch (...)
	{
		if (begun)
		{
			try
			{
				m_context.cancel();
			}
			catch (...)
			{
			}
		}
		throw;
	}
	return image;
}
id<MTLTexture> border_coverage_cache::color_view(id<MTLTexture> source, std::array<std::uint32_t, 4> channels)
{
	if (!source || source.device != m_context.device() || source.sampleCount != 1 || source.arrayLength != 1 ||
		(source.textureType != MTLTextureType2D && source.textureType != MTLTextureType3D && source.textureType != MTLTextureTypeCube))
		throw std::invalid_argument("Native custom border color snapshot shape is unsupported");
	if (!float_sampled_texture_format(source.pixelFormat) || !(source.usage & MTLTextureUsageShaderRead) ||
		source.framebufferOnly || source.storageMode == MTLStorageModeMemoryless)
		throw std::invalid_argument("Native border color snapshot requires readable ordinary float storage");
	for (auto it = m_colors.begin(); it != m_colors.end(); ++it)
		if (it->source == source && it->channels == channels)
		{
			const auto result = it->image;
			m_colors.splice(m_colors.begin(), m_colors, it);
			return result;
		}
	constexpr std::array<MTLTextureSwizzle, 6> swizzles = {MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha, MTLTextureSwizzleZero, MTLTextureSwizzleOne};
	const auto view = make_texture_view(source, source.pixelFormat, source.textureType, NSMakeRange(0, source.mipmapLevelCount), NSMakeRange(0, source.textureType == MTLTextureTypeCube ? 6 : 1),
										MTLTextureSwizzleChannelsMake(swizzles.at(channels[0]), swizzles.at(channels[1]), swizzles.at(channels[2]), swizzles.at(channels[3])));
	const auto image = build(source, MTLPixelFormatRGBA32Float, view);
	m_colors.push_front({source, channels, image});
	if (m_colors.size() > m_capacity)
		m_colors.pop_back();
	return image;
}
id<MTLTexture> border_coverage_cache::shadow_view(id<MTLTexture> source, float value)
{
	if (!source || source.device != m_context.device() || source.sampleCount != 1 || source.arrayLength != 1 ||
		(source.textureType != MTLTextureType2D && source.textureType != MTLTextureTypeCube) ||
		(source.pixelFormat != MTLPixelFormatDepth16Unorm && source.pixelFormat != MTLPixelFormatDepth32Float && source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) ||
		!std::isfinite(value) || value < 0 || value > 1)
		throw std::invalid_argument("Native shadow border requires supported depth storage and normalized register color");
	const auto format = source.pixelFormat == MTLPixelFormatDepth16Unorm ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float;
	for (auto it = m_shadows.begin(); it != m_shadows.end(); ++it)
		if (it->value == value && it->image.pixelFormat == format && same_shape(source, it->image))
		{
			const auto result = it->image;
			m_shadows.splice(m_shadows.begin(), m_shadows, it);
			return result;
		}
	auto descriptor = [MTLTextureDescriptor new];
	descriptor.textureType = source.textureType;
	descriptor.pixelFormat = format;
	descriptor.width = source.width;
	descriptor.height = source.height;
	descriptor.depth = 1;
	descriptor.mipmapLevelCount = source.mipmapLevelCount;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	const auto image = m_budget->texture(descriptor);
	bool begun = false;
	try
	{
		m_context.begin();
		begun = true;
		for (NSUInteger level = 0; level < image.mipmapLevelCount; ++level)
		{
			if (format == MTLPixelFormatDepth16Unorm)
			{
				// Native D16 clear conversion truncates some fractional values.
				// Upload the exact UNORM code instead; comparisons remain native.
				const auto width = std::max<NSUInteger>(1, image.width >> level), height = std::max<NSUInteger>(1, image.height >> level);
				const auto rows = std::min<NSUInteger>(height, std::max<NSUInteger>(1, (1u << 20) / (width * 2)));
				// Float input times65535 is exact in double; integer ties-even
				// encoding does not depend on the render thread's rounding mode.
				const double scaled = double(value) * 65535;
				auto code = static_cast<std::uint16_t>(scaled);
				const double fraction = scaled - code;
				if (fraction > .5 || (fraction == .5 && (code & 1)))
					++code;
				const std::vector<std::uint16_t> pixels(width * rows, code);
				const auto upload = m_context.upload(std::as_bytes(std::span(pixels)), 256);
				for (NSUInteger face = 0; face < (image.textureType == MTLTextureTypeCube ? 6u : 1u); ++face)
					for (NSUInteger y = 0; y < height; y += rows)
					{
						texture_region region;
						region.level = level;
						region.slice = face;
						region.origin = MTLOriginMake(0, y, 0);
						region.size = MTLSizeMake(width, std::min(rows, height - y), 1);
						buffer_to_texture(m_context, upload.buffer, upload.offset, image, region, width * 2, 0, texture_plane::depth);
					}
				continue;
			}
			for (NSUInteger face = 0; face < (image.textureType == MTLTextureTypeCube ? 6u : 1u); ++face)
			{
				auto pass = [MTL4RenderPassDescriptor new];
				pass.depthAttachment.texture = image;
				pass.depthAttachment.level = level;
				pass.depthAttachment.slice = face;
				pass.depthAttachment.loadAction = MTLLoadActionClear;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
				pass.depthAttachment.clearDepth = value;
				(void)m_context.render(pass);
			}
		}
		m_context.submit().wait();
	}
	catch (...)
	{
		if (begun)
		{
			try
			{
				m_context.cancel();
			}
			catch (...)
			{
			}
		}
		throw;
	}
	m_shadows.push_front({value, image});
	if (m_shadows.size() > m_capacity)
		m_shadows.pop_back();
	return image;
}
void validate_border_coverage(context& ctx, id<MTLTexture> source, id<MTLTexture> coverage, const sampler_parameters& parameters)
{
	validate_sampler_parameters(parameters);
	if (!(parameters.flags & sampler_custom_border) || !coverage || coverage.device != ctx.device() ||
		coverage.pixelFormat != ((parameters.flags & sampler_depth_compare) ? (source.pixelFormat == MTLPixelFormatDepth16Unorm ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float) : MTLPixelFormatR32Float) || coverage.sampleCount != 1 || !(coverage.usage & MTLTextureUsageShaderRead) ||
		!same_shape(source, coverage))
		throw std::invalid_argument("Native border coverage must match source dimensions and mip chain");
}
} // namespace metal4
