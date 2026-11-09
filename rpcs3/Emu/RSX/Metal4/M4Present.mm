#include "M4Present.h"
#include "M4PresentKernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace metal4
{
namespace
{
std::size_t format_index(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatBGRA8Unorm:
		return 0;
	case MTLPixelFormatBGRA8Unorm_sRGB:
		return 1;
	case MTLPixelFormatRGBA16Float:
		return 2;
	default:
		throw std::invalid_argument("Unsupported Metal4 display attachment format");
	}
}
bool readable_color(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatRGBA8Unorm_sRGB:
	case MTLPixelFormatBGRA8Unorm:
	case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatRGBA32Float:
		return true;
	default:
		return false;
	}
}
id<MTLTexture> owner(id<MTLTexture> texture)
{
	while (texture.parentTexture)
		texture = texture.parentTexture;
	return texture;
}
struct parameters
{
	float origin[2], extent[2];
	std::uint32_t level, flip_x, flip_y, padding = 0;
	float gamma;
	std::uint32_t limited_rgb, padding2[2]{};
	float right_origin[2], right_extent[2];
	std::uint32_t right_level, stereo_mode, padding3[2]{};
	std::array<std::array<float, 4>, 3> left_matrix, right_matrix;
	float destination_origin[2], destination_extent[2];
};
static_assert(sizeof(parameters) == 192 && offsetof(parameters, gamma) == 32 && offsetof(parameters, right_origin) == 48 &&
	offsetof(parameters, right_level) == 64 && offsetof(parameters, left_matrix) == 80 && offsetof(parameters, right_matrix) == 128 &&
	offsetof(parameters, destination_origin) == 176);
} // namespace

presenter::presenter(context& ctx) : m_device(ctx.device()), m_compiler(ctx)
{
	m_library = m_compiler.library(present_kernels, @"RPCS3 native Metal4 presentation");
	MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
	descriptor.sAddressMode = descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
	descriptor.mipFilter = MTLSamplerMipFilterNearest;
	descriptor.minFilter = descriptor.magFilter = MTLSamplerMinMagFilterNearest;
	m_nearest = make_sampler(ctx, descriptor);
	descriptor.minFilter = descriptor.magFilter = MTLSamplerMinMagFilterLinear;
	m_linear = make_sampler(ctx, descriptor);
}

void presenter::configure_layer(CAMetalLayer* layer, MTLPixelFormat format) const
{
	if (!layer)
		throw std::invalid_argument("Metal4 presentation requires a layer");
	if (format != MTLPixelFormatBGRA8Unorm && format != MTLPixelFormatBGRA8Unorm_sRGB)
		throw std::invalid_argument("Metal4 SDR layer requires a documented BGRA8 drawable format");
	layer.device = m_device;
	layer.pixelFormat = format;
	layer.framebufferOnly = YES;
}

void presenter::validate(context& ctx, const present_image& image, id<MTLTexture> destination) const
{
	validate_stereo_output(image.stereo);
	if (ctx.device() != m_device || !destination || destination.device != m_device || destination.textureType != MTLTextureType2D ||
		destination.sampleCount != 1 || !(destination.usage & MTLTextureUsageRenderTarget) ||
		(image.filter != present_filter::nearest && image.filter != present_filter::linear) || !std::isfinite(image.gamma) || image.gamma <= 0)
		throw std::invalid_argument("Invalid native Metal4 presentation images/filter");
	if (image.stereo.mode != stereo_mode::disabled && !image.right_eye) throw std::invalid_argument("Native stereo output requires both owned eyes");
	(void)format_index(destination.pixelFormat);
	const auto check_source = [&](id<MTLTexture> source, const texture_region& region)
	{
		if (!source || source.device != m_device || source.textureType != MTLTextureType2D || source.sampleCount != 1 ||
			!readable_color(source.pixelFormat) || !(source.usage & MTLTextureUsageShaderRead) || source.framebufferOnly)
			throw std::invalid_argument("Invalid native Metal4 presentation eye");
		(void)validate_texture_region(ctx, source, region);
		const auto a = owner(source), b = owner(destination);
		if (a == b || (a.buffer && a.buffer == b.buffer) || (a.heap && a.heap == b.heap))
			throw std::invalid_argument("Metal4 presentation requires independent source/destination allocations");
	};
	check_source(image.texture, image.region);
	if (image.right_eye) check_source(image.right_eye->texture, image.right_eye->region);
	const auto& rect = image.destination;
	if (!rect.width || !rect.height || rect.x > destination.width || rect.width > destination.width - rect.x ||
		rect.y > destination.height || rect.height > destination.height - rect.y)
		throw std::out_of_range("Metal4 presentation rectangle exceeds destination");
}

id<MTLRenderPipelineState> presenter::pipeline(MTLPixelFormat format)
{
	auto& result = m_pipelines[format_index(format)];
	if (!result)
	{
		helper_render_configuration config;
		config.colors[0] = format;
		result = m_compiler.render(m_library, @"m4_present_vertex", @"m4_present_fragment", config);
	}
	return result;
}

void present_image::depend_on_uploads(context& ctx) const
{
	for (const auto& upload : uploads)
		if (upload) ctx.depend_on_predecessor(*upload);
}

void begin_present_commands(context& ctx, const present_image* image)
{
	const auto last = ctx.last_submission();
	const bool follows_upload = last && image && std::ranges::any_of(image->uploads, [&](const auto& upload)
	{
		return upload && upload->same_batch(*last);
	});
	ctx.begin(follows_upload ? &*last : nullptr);
}

void presenter::encode(context& ctx, const present_image& image, id<MTLTexture> destination)
{
	@autoreleasepool
	{
		validate(ctx, image, destination);
		image.depend_on_uploads(ctx);
		const auto state = pipeline(destination.pixelFormat);
		parameters params{{float(image.region.origin.x), float(image.region.origin.y)},
								{float(image.region.size.width), float(image.region.size.height)},
								static_cast<std::uint32_t>(image.region.level),
								image.flip_x,
								image.flip_y,
								0,
								image.gamma,
								image.limited_rgb,
								{0, 0}};
		const auto right = image.right_eye.value_or(present_eye{image.texture, image.region});
		params.right_origin[0] = float(right.region.origin.x); params.right_origin[1] = float(right.region.origin.y);
		params.right_extent[0] = float(right.region.size.width); params.right_extent[1] = float(right.region.size.height);
		params.right_level = static_cast<std::uint32_t>(right.region.level);
		params.stereo_mode = static_cast<std::uint32_t>(image.stereo.mode);
		params.left_matrix = image.stereo.left; params.right_matrix = image.stereo.right;
		params.destination_origin[0] = float(image.destination.x); params.destination_origin[1] = float(image.destination.y);
		params.destination_extent[0] = float(image.destination.width); params.destination_extent[1] = float(image.destination.height);
		const auto upload = ctx.upload(std::as_bytes(std::span{&params, 1}));
		bindings fragment;
		fragment.set_buffer(0, upload.buffer, upload.offset);
		fragment.set_texture(0, image.texture);
		fragment.set_texture(1, right.texture);
		fragment.set_sampler(0, image.filter == present_filter::nearest ? m_nearest : m_linear);
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = destination;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
		raster_state raster;
		const auto& rect = image.destination;
		raster.viewport = {double(rect.x), double(rect.y), double(rect.width), double(rect.height), 0, 1};
		raster.scissor = rect;
		render_commands::draw_fragment_triangle(ctx, pass, state, nil, raster, fragment);
	}
}

std::optional<submission> presenter::present(context& ctx, const present_image& image, CAMetalLayer* layer)
{
	@autoreleasepool
	{
		if (ctx.device() != m_device || !layer || layer.device != m_device || !layer.framebufferOnly)
			throw std::invalid_argument("Native Metal4 presentation layer/device mismatch");
		(void)format_index(layer.pixelFormat);
		const auto drawable = [layer nextDrawable];
		if (!drawable)
			return std::nullopt;
		validate(ctx, image, drawable.texture);
		(void)pipeline(drawable.texture.pixelFormat);
		begin_present_commands(ctx, &image);
		try
		{
			ctx.target_drawable(drawable);
			encode(ctx, image, drawable.texture);
			return ctx.submit();
		}
		catch (...)
		{
			ctx.cancel();
			throw;
		}
	}
}

void presenter::encode_black(context& ctx, id<MTLTexture> destination) const
{
	@autoreleasepool
	{
		if (ctx.device() != m_device || !destination || destination.device != m_device || destination.textureType != MTLTextureType2D ||
			destination.sampleCount != 1 || !(destination.usage & MTLTextureUsageRenderTarget))
			throw std::invalid_argument("Invalid native black presentation target");
		(void)format_index(destination.pixelFormat);
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = destination;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
		render_commands::materialize_attachments(ctx, pass);
	}
}
} // namespace metal4
