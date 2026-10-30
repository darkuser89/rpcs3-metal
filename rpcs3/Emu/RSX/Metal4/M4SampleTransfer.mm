#include "M4SampleTransfer.h"
#include "M4SurfaceColor.h"
#include "M4Texture.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
namespace
{
sample_aspect aspect(MTLPixelFormat format)
{
	if (format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float)
		return sample_aspect::depth;
	if (format == MTLPixelFormatDepth32Float_Stencil8)
		return sample_aspect::depth_stencil;
	for (const auto raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
		if (describe_surface_color(static_cast<surface_color>(raw)).format == format)
			return sample_aspect::color;
	throw std::invalid_argument("Native MSAA sample transfer format is not a supported RSX surface");
}
id<MTLTexture> owner(id<MTLTexture> texture)
{
	while (texture.parentTexture)
		texture = texture.parentTexture;
	return texture;
}
bool identity(id<MTLTexture> texture)
{
	const auto channels = texture.swizzle;
	return channels.red == MTLTextureSwizzleRed && channels.green == MTLTextureSwizzleGreen &&
		   channels.blue == MTLTextureSwizzleBlue && channels.alpha == MTLTextureSwizzleAlpha;
}
} // namespace

sample_transfer::sample_transfer(context& ctx) : m_compiler(ctx), m_device(ctx.device())
{
	MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
	descriptor.depthCompareFunction = MTLCompareFunctionAlways;
	descriptor.depthWriteEnabled = YES;
	m_depth_state = [m_device newDepthStencilStateWithDescriptor:descriptor];
	MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
	stencil.stencilCompareFunction = MTLCompareFunctionAlways;
	stencil.depthStencilPassOperation = MTLStencilOperationReplace;
	stencil.readMask = stencil.writeMask = 255;
	descriptor.frontFaceStencil = descriptor.backFaceStencil = stencil;
	m_depth_stencil_state = [m_device newDepthStencilStateWithDescriptor:descriptor];
	descriptor.depthWriteEnabled = NO;
	m_stencil_state = [m_device newDepthStencilStateWithDescriptor:descriptor];
	if (!m_depth_state || !m_depth_stencil_state || !m_stencil_state)
		throw std::runtime_error("Native MSAA depth/stencil state allocation failed");
}

id<MTLRenderPipelineState> sample_transfer::pipeline(MTLPixelFormat format, std::uint32_t samples, bool importing, sample_aspect type)
{
	auto& library = m_libraries.at(static_cast<std::size_t>(type) * 2 + importing);
	if (!library) library = m_compiler.library(generate_sample_shader(type, importing), @"RPCS3 native spatial MSAA transfer");
	helper_render_configuration config;
	config.samples = importing ? samples : 1;
	if (type == sample_aspect::color)
		config.colors[0] = format;
	return m_compiler.render(library, @"rpcs3_sample_vertex", @"rpcs3_sample_fragment", config);
}

void sample_transfer::run(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, bool importing, texture_plane plane)
{
	if (ctx.device() != m_device || !multisampled || !expanded || multisampled.device != m_device || expanded.device != m_device ||
		multisampled.textureType != MTLTextureType2DMultisample || expanded.textureType != MTLTextureType2D ||
		expanded.sampleCount != 1 || expanded.pixelFormat != multisampled.pixelFormat ||
		multisampled.framebufferOnly || expanded.framebufferOnly || multisampled.storageMode == MTLStorageModeMemoryless ||
		expanded.storageMode == MTLStorageModeMemoryless || owner(multisampled) == owner(expanded))
		throw std::invalid_argument("Native spatial MSAA transfer texture/device/alias contract is invalid");
	if (!identity(multisampled) || !identity(expanded))
		throw std::invalid_argument("Native sample transfer requires identity channel views");
	const auto a = owner(multisampled), b = owner(expanded);
	if (a.heap && a.heap == b.heap)
	{
		const bool overlaps = a.heapOffset <= b.heapOffset ? b.heapOffset - a.heapOffset < a.allocatedSize : a.heapOffset - b.heapOffset < b.allocatedSize;
		if (overlaps)
			throw std::invalid_argument("Native sample transfer textures alias a heap allocation");
	}
	if (multisampled.sampleCount > 4)
		throw std::invalid_argument("RSX MSAA supports only 2 or 4 samples");
	const auto samples = static_cast<std::uint32_t>(multisampled.sampleCount);
	const auto size = sample_expanded_size(multisampled.width, multisampled.height, samples);
	if (expanded.width != size[0] || expanded.height != size[1])
		throw std::invalid_argument("Expanded MSAA texture dimensions do not match RSX sample raster");
	const auto source = importing ? expanded : multisampled, target = importing ? multisampled : expanded;
	if (!(source.usage & MTLTextureUsageShaderRead) || !(target.usage & MTLTextureUsageRenderTarget))
		throw std::invalid_argument("Native MSAA transfer requires readable source and renderable target");
	auto type = aspect(target.pixelFormat);
	(void)storage_block(target.pixelFormat, target.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && plane == texture_plane::all ? texture_plane::depth : plane);
	if (plane == texture_plane::depth)
		type = sample_aspect::depth;
	else if (plane == texture_plane::stencil)
		type = sample_aspect::stencil;
	const auto render_pipeline = pipeline(target.pixelFormat, samples, importing, type);
	id<MTLTexture> stencil = nil;
	if (type == sample_aspect::depth_stencil || type == sample_aspect::stencil)
	{
		const auto swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
		stencil = make_texture_view(source, MTLPixelFormatX32_Stencil8, source.textureType, NSMakeRange(0, 1), NSMakeRange(0, 1), swizzle);
	}
	const auto layout = rsx_sample_layout(samples);
	const std::array<std::uint32_t, 2> parameters{layout.x, layout.y};
	const auto uniforms = ctx.upload(std::as_bytes(std::span(parameters)), 256);
	bindings fragment;
	fragment.set_buffer(0, uniforms.buffer, uniforms.offset);
	fragment.set_texture(0, type == sample_aspect::stencil ? stencil : source);
	if (type == sample_aspect::depth_stencil)
		fragment.set_texture(1, stencil);
	if (type == sample_aspect::depth || type == sample_aspect::depth_stencil)
	{
		const std::uint32_t zero = 0;
		const auto status = ctx.upload(std::as_bytes(std::span{&zero, 1}), 4);
		fragment.set_buffer(1, status.buffer, status.offset);
		ctx.require_gpu_success(status.buffer, status.offset, @"Native MSAA depth outside finite [0,1] needs an exact alternative transfer path");
	}
	MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
	if (type == sample_aspect::color)
	{
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	}
	else
	{
		if (type != sample_aspect::stencil)
		{
			pass.depthAttachment.texture = target;
			pass.depthAttachment.loadAction = MTLLoadActionDontCare;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
		}
		if (stencil)
		{
			pass.stencilAttachment.texture = target;
			pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
		}
	}
	render_commands commands(ctx, pass);
	raster_state raster;
	raster.viewport = {0, 0, static_cast<double>(target.width), static_cast<double>(target.height), 0, 1};
	raster.scissor = {0, 0, target.width, target.height};
	commands.configure(render_pipeline, type == sample_aspect::color ? nil : type == sample_aspect::stencil ? m_stencil_state
																		 : stencil							? m_depth_stencil_state
																											: m_depth_state,
					   raster, {}, fragment);
	commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
	commands.end();
}
void sample_transfer::expand(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, texture_plane plane)
{
	run(ctx, multisampled, expanded, false, plane);
}
void sample_transfer::import(context& ctx, id<MTLTexture> expanded, id<MTLTexture> multisampled, texture_plane plane)
{
	run(ctx, multisampled, expanded, true, plane);
}
} // namespace metal4
