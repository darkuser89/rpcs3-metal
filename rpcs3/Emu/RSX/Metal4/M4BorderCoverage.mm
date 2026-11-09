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
struct border_texture_preparation::image
{
	id<MTLTexture> texture;
	std::weak_ptr<const std::byte> queue;
	std::function<void(context&)> initialize;
	std::optional<submission> producer;

	bool reusable()
	{
		if (producer && producer->ready())
		{
			try { producer->wait(); }
			catch (const std::runtime_error&) { return false; }
			producer.reset();
		}
		return true;
	}
};

void border_texture_preparation::encode(context& ctx)
{
	if (m_images.empty()) return;
	const auto current = ctx.recording_submission();
	const auto queue = ctx.queue_identity().lock();
	for (const auto& image : m_images)
	{
		if (image->queue.lock() != queue)
			throw std::invalid_argument("Native border helper belongs to another queue");
		if (image->producer)
		{
			if (!image->producer->same_batch(current))
				ctx.depend_on_predecessor(*image->producer);
		}
		else if (image->initialize)
		{
			image->initialize(ctx);
			image->producer = current;
			image->initialize = {};
		}
	}
}

id<MTLTexture> border_coverage_cache::use(const std::shared_ptr<helper>& image, border_texture_preparation* preparation)
{
	if (!image->initialize && !image->producer) return image->texture;
	if (preparation)
	{
		if (std::ranges::find(preparation->m_images, image) == preparation->m_images.end())
			preparation->m_images.push_back(image);
		return image->texture;
	}
	// Standalone callers receive initialized images. In particular, never
	// wait for an unsubmitted recording that belongs to the caller.
	m_context.validate_begin();
	if (!image->producer)
	{
		border_texture_preparation work;
		work.m_images.push_back(image);
		m_context.begin();
		try
		{
			work.encode(m_context);
			(void)m_context.submit();
		}
		catch (...)
		{
			m_context.cancel();
			throw;
		}
	}
	image->producer->wait();
	image->producer.reset();
	return image->texture;
}

border_coverage_cache::border_coverage_cache(context& ctx, std::shared_ptr<heap_budget> budget, std::size_t capacity)
	: m_context(ctx), m_budget(std::move(budget)), m_capacity(capacity)
{
	if (!m_budget || m_budget->device() != ctx.device() || !capacity || capacity > 4096)
		throw std::invalid_argument("Native border coverage needs a matching budget and bounded capacity");
}
id<MTLTexture> border_coverage_cache::get(id<MTLTexture> source, border_texture_preparation* preparation)
{
	if (!source || source.device != m_context.device() || source.sampleCount != 1 || source.arrayLength != 1 ||
		(source.textureType != MTLTextureType2D && source.textureType != MTLTextureType3D && source.textureType != MTLTextureTypeCube))
		throw std::invalid_argument("Native border coverage source shape is unsupported");
	for (auto it = m_images.begin(); it != m_images.end(); ++it)
		if (same_shape(source, (*it)->texture))
		{
			if (!(*it)->reusable()) { m_images.erase(it); break; }
			const auto result = use(*it, preparation);
			m_images.splice(m_images.begin(), m_images, it);
			return result;
		}
	const auto image = build(source, MTLPixelFormatR32Float);
	const auto result = use(image, preparation);
	m_images.push_front(image);
	if (m_images.size() > m_capacity)
		m_images.pop_back();
	return result;
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
std::shared_ptr<border_coverage_cache::helper> border_coverage_cache::build(id<MTLTexture> source, MTLPixelFormat format, id<MTLTexture> input)
{
	const unsigned kind = source.textureType == MTLTextureType3D ? 1 : source.textureType == MTLTextureTypeCube ? 2 : 0;
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
	return std::make_shared<helper>(helper{image, m_context.queue_identity(), [image, input, kind, pipeline](context& ctx)
	{
		for (NSUInteger level = 0; level < image.mipmapLevelCount; ++level)
		{
			const auto type = kind == 2 ? MTLTextureType2DArray : image.textureType;
			const auto slices = NSMakeRange(0, kind == 2 ? 6 : 1);
			const auto mip = NSMakeRange(level, 1);
			const auto view = make_texture_view(image, image.pixelFormat, type, mip, slices,
				MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha));
			bindings resources;
			resources.set_texture(0, view);
			id<MTLTexture> src = nil;
			if (input)
			{
				src = make_texture_view(input, input.pixelFormat, type, mip, slices, input.swizzle);
				resources.set_texture(1, src);
			}
			// The output view exposes exactly this mip of the new root texture.
			const std::array accesses{context::accessed_resource{image, 0, 0, true, level},
				context::accessed_resource{src}};
			ctx.checked_dispatch(pipeline, resources, std::span{accesses}.first(src ? 2 : 1),
				MTLSizeMake(view.width, view.height, kind == 2 ? 6 : view.depth), MTLSizeMake(8, 8, 1));
		}
	}, {}});
}
id<MTLTexture> border_coverage_cache::color_view(id<MTLTexture> source, std::array<std::uint32_t, 4> channels, border_texture_preparation* preparation)
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
			if (!it->image->reusable()) { m_colors.erase(it); break; }
			const auto result = use(it->image, preparation);
			m_colors.splice(m_colors.begin(), m_colors, it);
			return result;
		}
	constexpr std::array<MTLTextureSwizzle, 6> swizzles = {MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha, MTLTextureSwizzleZero, MTLTextureSwizzleOne};
	const auto view = make_texture_view(source, source.pixelFormat, source.textureType, NSMakeRange(0, source.mipmapLevelCount), NSMakeRange(0, source.textureType == MTLTextureTypeCube ? 6 : 1),
										MTLTextureSwizzleChannelsMake(swizzles.at(channels[0]), swizzles.at(channels[1]), swizzles.at(channels[2]), swizzles.at(channels[3])));
	const auto image = build(source, MTLPixelFormatRGBA32Float, view);
	const auto result = use(image, preparation);
	m_colors.push_front({source, channels, image});
	if (m_colors.size() > m_capacity)
		m_colors.pop_back();
	return result;
}
id<MTLTexture> border_coverage_cache::shadow_view(id<MTLTexture> source, float value, border_texture_preparation* preparation)
{
	if (!source || source.device != m_context.device() || source.sampleCount != 1 || source.arrayLength != 1 ||
		(source.textureType != MTLTextureType2D && source.textureType != MTLTextureTypeCube) ||
		(source.pixelFormat != MTLPixelFormatDepth16Unorm && source.pixelFormat != MTLPixelFormatDepth32Float && source.pixelFormat != MTLPixelFormatDepth32Float_Stencil8) ||
		!std::isfinite(value) || value < 0 || value > 1)
		throw std::invalid_argument("Native shadow border requires supported depth storage and normalized register color");
	const auto format = source.pixelFormat == MTLPixelFormatDepth16Unorm ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float;
	for (auto it = m_shadows.begin(); it != m_shadows.end(); ++it)
		if (it->value == value && it->image->texture.pixelFormat == format && same_shape(source, it->image->texture))
		{
			if (!it->image->reusable()) { m_shadows.erase(it); break; }
			const auto result = use(it->image, preparation);
			m_shadows.splice(m_shadows.begin(), m_shadows, it);
			return result;
		}
	const auto image = build_shadow(source, format, value);
	const auto result = use(image, preparation);
	m_shadows.push_front({value, image});
	if (m_shadows.size() > m_capacity)
		m_shadows.pop_back();
	return result;
}

std::shared_ptr<border_coverage_cache::helper> border_coverage_cache::build_shadow(id<MTLTexture> source, MTLPixelFormat format, float value)
{
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
	id<MTLBuffer> upload = nil;
	NSUInteger strip_texels = 0;
	if (format == MTLPixelFormatDepth16Unorm)
	{
		// Native D16 clears can truncate. Pack the exact ties-even UNORM
		// code once and reuse the immutable strip across every mip and face.
		const double scaled = double(value) * 65535;
		auto code = static_cast<std::uint16_t>(scaled);
		const double fraction = scaled - code;
		if (fraction > .5 || (fraction == .5 && (code & 1))) ++code;
		const auto rows = std::min<NSUInteger>(image.height, std::max<NSUInteger>(1, (1u << 20) / (image.width * sizeof(code))));
		strip_texels = image.width * rows;
		const std::vector<std::uint16_t> pixels(strip_texels, code);
		upload = make_upload(m_context, std::as_bytes(std::span(pixels)));
	}
	return std::make_shared<helper>(helper{image, m_context.queue_identity(), [image, upload, strip_texels, value](context& ctx)
	{
		const auto faces = image.textureType == MTLTextureTypeCube ? 6u : 1u;
		for (NSUInteger level = 0; level < image.mipmapLevelCount; ++level)
		{
			if (upload)
			{
				const auto width = std::max<NSUInteger>(1, image.width >> level);
				const auto height = std::max<NSUInteger>(1, image.height >> level);
				const auto rows = std::min(height, strip_texels / width);
				for (NSUInteger face = 0; face < faces; ++face)
					for (NSUInteger y = 0; y < height; y += rows)
					{
						texture_region region;
						region.level = level;
						region.slice = face;
						region.origin = MTLOriginMake(0, y, 0);
						region.size = MTLSizeMake(width, std::min(rows, height - y), 1);
						buffer_to_texture(ctx, upload, 0, image, region, width * sizeof(std::uint16_t), 0, texture_plane::depth);
					}
				continue;
			}
			for (NSUInteger face = 0; face < faces; ++face)
			{
				auto pass = [MTL4RenderPassDescriptor new];
				pass.depthAttachment.texture = image;
				pass.depthAttachment.level = level;
				pass.depthAttachment.slice = face;
				pass.depthAttachment.loadAction = MTLLoadActionClear;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
				pass.depthAttachment.clearDepth = value;
				(void)ctx.render(pass);
			}
		}
	}, {}});
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
