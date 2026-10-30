#include "M4Clear.h"
#include "M4ClearShaders.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
namespace
{
struct alignas(16) parameters
{
	std::array<std::array<float, 4>, 4> color;
	float depth;
	std::array<std::uint32_t, 3> reserved{};
};
static_assert(sizeof(parameters) == 80);
std::uint32_t full_color_mask(surface_color format)
{
	return format == surface_color::rgb565 ? 7u : format == surface_color::b8 ? 1u : format == surface_color::gb8 ? 3u : 15u;
}
MTLColorWriteMask native_mask(std::uint32_t mask)
{
	return static_cast<MTLColorWriteMask>(((mask & 1) ? MTLColorWriteMaskRed : 0) | ((mask & 2) ? MTLColorWriteMaskGreen : 0) |
										  ((mask & 4) ? MTLColorWriteMaskBlue : 0) | ((mask & 8) ? MTLColorWriteMaskAlpha : 0));
}
struct attachment_info
{
	std::size_t width, height;
	std::uint32_t samples;
};
clear_attachment storage(clear_attachment image)
{
	while (image.texture.parentTexture)
	{
		image.level += image.texture.parentRelativeLevel;
		image.slice += image.texture.parentRelativeSlice;
		image.texture = image.texture.parentTexture;
	}
	return image;
}
bool overlaps(clear_attachment a, clear_attachment b)
{
	a = storage(a);
	b = storage(b);
	if (a.texture == b.texture)
		return a.level == b.level && a.slice == b.slice;
	if (!a.texture.heap || a.texture.heap != b.texture.heap)
		return false;
	return a.texture.heapOffset <= b.texture.heapOffset ? b.texture.heapOffset - a.texture.heapOffset < a.texture.allocatedSize : a.texture.heapOffset - b.texture.heapOffset < b.texture.allocatedSize;
}
attachment_info validate(context& ctx, const clear_attachment& image, MTLPixelFormat format)
{
	const auto texture = image.texture;
	if (!texture || texture.device != ctx.device() || texture.pixelFormat != format || !(texture.usage & MTLTextureUsageRenderTarget) ||
		texture.storageMode == MTLStorageModeMemoryless || image.level >= texture.mipmapLevelCount || image.level >= std::numeric_limits<std::size_t>::digits ||
		image.slice >= texture.arrayLength || (texture.textureType != MTLTextureType2D && texture.textureType != MTLTextureType2DArray && texture.textureType != MTLTextureType2DMultisample) ||
		(texture.sampleCount != 1 && texture.sampleCount != 2 && texture.sampleCount != 4))
		throw std::invalid_argument("Native clear attachment format/device/subresource/sample contract is invalid");
	return {std::max<std::size_t>(1, texture.width >> image.level), std::max<std::size_t>(1, texture.height >> image.level), static_cast<std::uint32_t>(texture.sampleCount)};
}
void attachment(MTLRenderPassAttachmentDescriptor* destination, const clear_attachment& source, MTLLoadAction load)
{
	destination.texture = source.texture;
	destination.level = source.level;
	destination.slice = source.slice;
	destination.loadAction = load;
	destination.storeAction = MTLStoreActionStore;
}
} // namespace

clear_engine::clear_engine(context& ctx) : m_device(ctx.device()), m_compiler(ctx)
{
	m_library = m_compiler.library(clear_shaders, @"RPCS3 native masked clear");
}
std::optional<upload_requirement> clear_engine::upload_for(surface_format format, std::uint32_t mask, std::uint32_t stencil_mask, bool full_rectangle)
{
	bool fast = full_rectangle;
	if (const auto color = std::get_if<surface_color>(&format))
	{
		const auto plan = decode_color_clear(*color, 0, mask);
		if (!plan.write_mask) return {};
		fast &= plan.write_mask == full_color_mask(*color);
	}
	else
	{
		const auto depth = describe_surface_depth(std::get<surface_depth>(format));
		const bool write_depth = mask & 1, write_stencil = (mask & 2) && depth.stencil && (stencil_mask & 255);
		if (!write_depth && !write_stencil) return {};
		fast &= (!write_stencil || (stencil_mask & 255) == 255) && !(write_depth && depth.format == MTLPixelFormatDepth16Unorm);
	}
	return fast ? std::optional<upload_requirement>{} : upload_requirement{sizeof(parameters), 256};
}
id<MTLRenderPipelineState> clear_engine::pipeline(const std::array<MTLPixelFormat, 4>& formats,
												  const std::array<std::uint32_t, 4>& masks, std::uint32_t samples, bool depth)
{
	helper_render_configuration config;
	config.colors = formats;
	config.samples = samples;
	for (unsigned slot = 0; slot < 4; ++slot) config.masks[slot] = native_mask(masks[slot]);
	return m_compiler.render(m_library, @"rpcs3_clear_vertex", depth ? @"rpcs3_clear_depth" : @"rpcs3_clear_color", config);
}
id<MTLDepthStencilState> clear_engine::depth_state(bool depth, std::uint32_t mask)
{
	if (mask > 255) throw std::invalid_argument("Native clear stencil mask exceeds eight bits");
	auto& cached = m_depth_states[(depth ? 256 : 0) + mask];
	if (cached) { ++m_depth_reused; return cached; }
	MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
	descriptor.depthCompareFunction = MTLCompareFunctionAlways;
	descriptor.depthWriteEnabled = depth;
	MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
	stencil.stencilCompareFunction = MTLCompareFunctionAlways;
	stencil.depthStencilPassOperation = MTLStencilOperationReplace;
	stencil.writeMask = mask;
	descriptor.frontFaceStencil = descriptor.backFaceStencil = stencil;
	const auto state = [m_device newDepthStencilStateWithDescriptor:descriptor];
	if (!state)
		throw std::runtime_error("Native clear depth/stencil state allocation failed");
	cached = state;
	++m_depth_created;
	return state;
}

clear_result clear_engine::run(context& ctx, const clear_request& request)
{
	if (ctx.device() != m_device)
		throw std::invalid_argument("Native clear engine belongs to another device");
	clear_result result{};
	if (!(request.mask & 0xf3u) || !request.rectangle.width || !request.rectangle.height)
		return result;
	std::array<color_clear_plan, 4> plans{};
	std::array<MTLPixelFormat, 4> formats{};
	std::array<std::uint32_t, 4> masks{};
	std::vector<clear_attachment> images;
	std::size_t width = 0, height = 0;
	std::uint32_t samples = 0;
	const auto include = [&](const clear_attachment& image, attachment_info info)
	{
		for (const auto& previous : images)
			if (overlaps(previous, image))
				throw std::invalid_argument("Clear attachments overlap the same storage");
		images.push_back(image);
		if (!samples)
		{
			width = info.width;
			height = info.height;
			samples = info.samples;
		}
		else if (width != info.width || height != info.height || samples != info.samples)
			throw std::invalid_argument("Clear attachments must have matching dimensions/samples");
	};
	for (std::size_t slot = 0; slot < 4; ++slot)
	{
		const auto& source = request.colors[slot];
		if (!source.image.texture)
			continue;
		plans[slot] = decode_color_clear(source.format, request.color_value, request.mask);
		if (!plans[slot].write_mask)
			continue;
		formats[slot] = describe_surface_color(source.format).format;
		masks[slot] = plans[slot].write_mask;
		include(source.image, validate(ctx, source.image, formats[slot]));
		result.color_targets |= 1u << slot;
	}
	const auto stencil_mask = request.stencil_write_mask & 255;
	surface_depth_descriptor depth{};
	float depth_value = 0;
	double load_depth_value = 0;
	if (request.depth.texture && (request.mask & 3))
	{
		depth = describe_surface_depth(request.depth_format);
		result.depth = (request.mask & 1) != 0;
		result.stencil = (request.mask & 2) && depth.stencil && stencil_mask;
		if (result.depth || result.stencil)
			include(request.depth, validate(ctx, request.depth, depth.format));
		const auto raw = depth.stencil ? request.zstencil_value >> 8 : request.zstencil_value & 65535;
		depth_value = float(raw) / (depth.stencil ? 16777215.0f : 65535.0f);
		// Preserve the double precision of MTL clearDepth. D16 load-clear
		// quantization still requires the draw fallback below.
		load_depth_value = double(raw) / (depth.stencil ? 16777215.0 : 65535.0);
	}
	if (!samples)
		return {};
	const auto rect = clip_clear_rectangle(request.rectangle, width, height);
	if (!rect.width || !rect.height)
		return {};
	const bool full = rect.x == 0 && rect.y == 0 && rect.width == width && rect.height == height;
	// D16 native load-clear conversion truncates below the intended integer
	// on tested Apple hardware, even with a double normalized ratio. The draw
	// path rounds the normalized fragment output correctly; preserve that path.
	bool fast = full && (!result.stencil || stencil_mask == 255) && !(result.depth && depth.format == MTLPixelFormatDepth16Unorm);
	for (std::size_t slot = 0; slot < 4; ++slot)
		if (masks[slot])
			fast &= masks[slot] == full_color_mask(request.colors[slot].format);
	MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
	for (NSUInteger slot = 0; slot < 4; ++slot)
		if (masks[slot])
		{
			attachment(pass.colorAttachments[slot], request.colors[slot].image, fast ? MTLLoadActionClear : MTLLoadActionLoad);
			const auto& color = plans[slot].value;
			pass.colorAttachments[slot].clearColor = MTLClearColorMake(color[0], color[1], color[2], color[3]);
		}
	if (result.depth || result.stencil)
	{
		attachment(pass.depthAttachment, request.depth, fast && result.depth ? MTLLoadActionClear : MTLLoadActionLoad);
		pass.depthAttachment.clearDepth = load_depth_value;
		if (depth.stencil)
		{
			attachment(pass.stencilAttachment, request.depth, fast && result.stencil ? MTLLoadActionClear : MTLLoadActionLoad);
			pass.stencilAttachment.clearStencil = request.zstencil_value & 255;
		}
	}
	if (fast)
	{
		(void)ctx.render(pass);
		ctx.end_encoding();
		return result;
	}
	const auto pipe = pipeline(formats, masks, samples, result.depth);
	const auto ds = depth_state(result.depth, result.stencil ? stencil_mask : 0);
	parameters values{};
	values.depth = depth_value;
	for (std::size_t slot = 0; slot < 4; ++slot)
		values.color[slot] = plans[slot].value;
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&values, 1}), 256);
	bindings fragment;
	fragment.set_buffer(0, uniforms.buffer, uniforms.offset);
	render_commands commands(ctx, pass);
	raster_state raster;
	raster.viewport = {0, 0, double(width), double(height), 0, 1};
	raster.scissor = {rect.x, rect.y, rect.width, rect.height};
	raster.front_stencil_reference = raster.back_stencil_reference = request.zstencil_value & 255;
	commands.configure(pipe, ds, raster, {}, fragment);
	commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
	commands.end();
	return result;
}
} // namespace metal4
