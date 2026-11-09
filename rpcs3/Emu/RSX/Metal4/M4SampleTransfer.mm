#include "M4SampleTransfer.h"
#include "M4Resources.h"
#include "M4SurfaceColor.h"
#include "M4Texture.h"
#include "M4TransferKernels.h"

#include <algorithm>
#include <sstream>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
namespace
{
struct float_expansion
{
	MTLPixelFormat pixel_format;
	const char* function;
	std::array<const char*, 2> source_type;
	const char* output_type;
	const char* channel;
	texture_plane plane;
};
constexpr std::array float_expansions{
	float_expansion{MTLPixelFormatDepth32Float, "expand_float_depth", {"depth2d_ms<float>", "depth2d<float>"}, "uint", "", texture_plane::depth},
	float_expansion{MTLPixelFormatR32Float, "expand_r32f", {"texture2d_ms<float>", "texture2d<float, access::read>"}, "uint", ".x", texture_plane::all},
	float_expansion{MTLPixelFormatRGBA32Float, "expand_rgba32f", {"texture2d_ms<float>", "texture2d<float, access::read>"}, "uint4", "", texture_plane::all},
	float_expansion{MTLPixelFormatRGBA16Float, "expand_rgba16f", {"texture2d_ms<half>", "texture2d<half, access::read>"}, "ushort4", "", texture_plane::all},
};
std::string float_expand_source()
{
	std::ostringstream out;
	out << "#include <metal_stdlib>\nusing namespace metal;\n" << nearest_axis_msl << sample_projection_msl;
	for (const auto& format : float_expansions)
		for (const bool single : {false, true})
		{
			out << "kernel void " << format.function << (single ? "_single" : "") << "("
				<< format.source_type[single] << " source [[texture(0)]], device " << format.output_type
				<< "* out [[buffer(0)]], constant sample_parameters& p [[buffer(1)]], uint2 pixel [[thread_position_in_grid]]) {\n"
				<< "    const auto input = locate_sample(pixel, p);\n"
				<< "    out[pixel.y * p.row_elements + pixel.x] = as_type<" << format.output_type
				<< ">(source.read(input.pixel" << (single ? "" : ", input.sample") << ")" << format.channel << ");\n}\n";
		}
	return out.str();
}

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

id<MTLRenderPipelineState> sample_transfer::pipeline(MTLPixelFormat format, std::uint32_t samples, bool importing, sample_aspect type, bool single_sample_source)
{
	const auto variant = importing ? 1u : single_sample_source ? 2u : 0u;
	// Keep half words in half registers through the fragment export. Promotion
	// to float and conversion back can quiet NaNs and change their payloads.
	const bool half_color = type == sample_aspect::color && format == MTLPixelFormatRGBA16Float;
	auto& library = m_libraries.at(static_cast<std::size_t>(type) * 3 + variant + (half_color ? 12 : 0));
	if (!library) library = m_compiler.library(generate_sample_shader(type, importing, single_sample_source, half_color), @"RPCS3 native spatial MSAA transfer");
	helper_render_configuration config;
	config.samples = importing ? samples : 1;
	if (type == sample_aspect::color)
		config.colors[0] = format;
	return m_compiler.render(library, @"rpcs3_sample_vertex", @"rpcs3_sample_fragment", config);
}

void sample_transfer::run(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, bool importing, texture_plane plane,
	std::optional<std::array<std::size_t,2>> crop, const texture_region* destination, bool preserve,
	const sample_raster_projection* projection)
{
	const bool single_source = projection && multisampled.sampleCount == 1;
	if (ctx.device() != m_device || !multisampled || !expanded || multisampled.device != m_device || expanded.device != m_device ||
		multisampled.textureType != (single_source ? MTLTextureType2D : MTLTextureType2DMultisample) || expanded.textureType != MTLTextureType2D ||
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
	const auto size = single_source ? std::array{multisampled.width, multisampled.height} :
		sample_expanded_size(multisampled.width, multisampled.height, samples);
	const auto origin=crop.value_or(std::array<std::size_t,2>{});
	texture_region area;
	area.size = MTLSizeMake(expanded.width, expanded.height, 1);
	if (destination)
	{
		area = *destination;
		if (area.level || area.slice || area.origin.z || area.size.depth != 1)
			throw std::invalid_argument("Native MSAA expansion destination must be level-zero 2D");
		(void)validate_texture_region(ctx, expanded, area, plane);
	}
	if (!projection && crop && (origin[0]>size[0]||origin[1]>size[1]||area.size.width>size[0]-origin[0]||area.size.height>size[1]-origin[1]||
		origin[0]>std::numeric_limits<std::uint32_t>::max()||origin[1]>std::numeric_limits<std::uint32_t>::max()))
		throw std::invalid_argument("Native MSAA crop exceeds the expanded sample raster");
	if (!projection && !crop && (expanded.width != size[0] || expanded.height != size[1]))
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
	const auto layout = single_source ? sample_layout{1, 1} : rsx_sample_layout(samples);
	const std::array extent{area.size.width, area.size.height};
	const auto mapping = projection ? *projection : sample_raster_projection{origin, extent, {}, extent};
	if (projection)
		for (unsigned axis = 0; axis < 2; ++axis)
			if (!mapping.source_extent[axis] || mapping.source_origin[axis] > size[axis] ||
				mapping.source_extent[axis] > size[axis] - mapping.source_origin[axis] ||
				mapping.copy_extent[axis] > std::numeric_limits<std::uint32_t>::max() ||
				mapping.copy_offset[axis] > mapping.copy_extent[axis] ||
				extent[axis] > mapping.copy_extent[axis] - mapping.copy_offset[axis])
				throw std::invalid_argument("Native nearest sample projection exceeds its source or copy extent");
	const parameters params{layout.x, layout.y, static_cast<std::uint32_t>(mapping.source_origin[0]),
		static_cast<std::uint32_t>(mapping.source_origin[1]), static_cast<std::uint32_t>(extent[0]), static_cast<std::uint32_t>(extent[1]),
		static_cast<std::uint32_t>(area.origin.x), static_cast<std::uint32_t>(area.origin.y),
		static_cast<std::uint32_t>(mapping.source_extent[0]), static_cast<std::uint32_t>(mapping.source_extent[1]),
		static_cast<std::uint32_t>(mapping.copy_extent[0]), static_cast<std::uint32_t>(mapping.copy_extent[1]),
		static_cast<std::uint32_t>(mapping.copy_offset[0]), static_cast<std::uint32_t>(mapping.copy_offset[1]),
		unsigned(mapping.flip[0]) | (unsigned(mapping.flip[1]) << 1), 0};
	const auto float_format = target.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ?
		MTLPixelFormatDepth32Float : target.pixelFormat;
	const auto expansion = std::ranges::find(float_expansions, float_format, &float_expansion::pixel_format);
	const bool float_words = !importing && type != sample_aspect::stencil && expansion != float_expansions.end();
	if (!float_words || type == sample_aspect::depth_stencil)
		raster_transfer(ctx, source, target, samples, importing, float_words ? sample_aspect::stencil : type, params,
			crop.has_value(), preserve || (importing && crop));
	if (float_words)
		expand_float(ctx, source, target, expansion - float_expansions.begin(), params);
	if(importing){++m_statistics.imports;m_statistics.imported_texels+=expanded.width*expanded.height;if(crop){++m_statistics.cropped_imports;m_statistics.cropped_imported_texels+=expanded.width*expanded.height;}}
	else if(crop || projection){++m_statistics.cropped_expansions;m_statistics.cropped_texels+=area.size.width*area.size.height;}
	else {++m_statistics.full_expansions;m_statistics.full_texels+=expanded.width*expanded.height;}
}

void sample_transfer::expand_float(context& ctx, id<MTLTexture> source, id<MTLTexture> target, std::size_t format,
	parameters params)
{
	const auto& expansion = float_expansions.at(format);
	const bool single = source.sampleCount == 1;
	auto& pipeline = m_float_expanders.at(format * 2 + single);
	if (!pipeline)
	{
		if (!m_float_expand_library)
			m_float_expand_library = m_compiler.library(float_expand_source(), @"RPCS3 native float sample word expansion");
		const auto name = std::string(expansion.function) + (single ? "_single" : "");
		pipeline = m_compiler.compute(m_float_expand_library, [NSString stringWithUTF8String:name.c_str()]);
	}
	const auto element_size = storage_block(target.pixelFormat, expansion.plane).bytes;

	const auto row_pitch = (params.width * element_size + 255) & ~std::size_t(255);
	const auto rows = std::min<std::size_t>(params.height, std::max<std::size_t>(1, (1u << 20) / row_pitch));
	const auto words_buffer = make_buffer(ctx, row_pitch * rows, buffer_storage::private_gpu);
	bindings arguments;
	arguments.set_texture(0, source);
	arguments.set_buffer(0, words_buffer);
	params.row_elements = static_cast<std::uint32_t>(row_pitch / element_size);
	const auto copy_y = params.copy_y;
	for (std::size_t first = 0; first < params.height; first += rows)
	{
		const auto count = std::min<std::size_t>(rows, params.height - first);
		params.copy_y = copy_y + static_cast<std::uint32_t>(first);
		const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
		arguments.set_buffer(1, uniforms.buffer, uniforms.offset);
		const std::array accesses{context::accessed_resource{source},
			context::accessed_resource{words_buffer, 0, row_pitch * count, true},
			context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size}};
		ctx.checked_dispatch(pipeline, arguments, accesses, MTLSizeMake(params.width, count, 1),
			MTLSizeMake(pipeline.threadExecutionWidth, 1, 1));
		texture_region region;
		region.origin = MTLOriginMake(params.target_x, params.target_y + first, 0);
		region.size = MTLSizeMake(params.width, count, 1);
		buffer_to_texture(ctx, words_buffer, 0, target, region, row_pitch, 0, expansion.plane);
	}
}

void sample_transfer::raster_transfer(context& ctx, id<MTLTexture> source, id<MTLTexture> target, unsigned samples,
	bool importing, sample_aspect type, const parameters& params, bool cropped, bool preserve)
{
	const auto render_pipeline = pipeline(target.pixelFormat, samples, importing, type, !importing && source.sampleCount == 1);
	id<MTLTexture> stencil = nil;
	if (type == sample_aspect::depth_stencil || type == sample_aspect::stencil)
	{
		const auto swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
		stencil = make_texture_view(source, MTLPixelFormatX32_Stencil8, source.textureType, NSMakeRange(0, 1), NSMakeRange(0, 1), swizzle);
	}
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
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
	const auto load = preserve ? MTLLoadActionLoad : MTLLoadActionDontCare;
	if (type == sample_aspect::color)
	{
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = load;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	}
	else
	{
		if (type != sample_aspect::stencil)
		{
			pass.depthAttachment.texture = target;
			pass.depthAttachment.loadAction = load;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
		}
		if (stencil)
		{
			pass.stencilAttachment.texture = target;
			pass.stencilAttachment.loadAction = load;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
		}
	}
	raster_state raster;
	raster.viewport = {0, 0, static_cast<double>(target.width), static_cast<double>(target.height), 0, 1};
	raster.scissor = {0, 0, target.width, target.height};
	if(importing&&cropped)
	{
		const auto x = params.origin_x / params.samples_x, y = params.origin_y / params.samples_y;
		const auto end_x = params.origin_x + params.width, end_y = params.origin_y + params.height;
		raster.scissor = {x, y, end_x / params.samples_x + bool(end_x % params.samples_x) - x,
			end_y / params.samples_y + bool(end_y % params.samples_y) - y};
	}
	else if (!importing)
		raster.scissor = {params.target_x, params.target_y, params.width, params.height};
	// Keep full queue coverage around native MSAA copies. A fragment/tile
	// wait alone did not preserve the copied samples in a blit-to-render chain.
	render_commands commands(ctx, pass);
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
void sample_transfer::expand_region(context& ctx,id<MTLTexture> multisampled,id<MTLTexture> expanded,
	std::array<std::size_t,2> origin,texture_plane plane)
{
	run(ctx,multisampled,expanded,false,plane,origin);
}
void sample_transfer::project_into(context& ctx, id<MTLTexture> source, id<MTLTexture> expanded,
	const sample_raster_projection& projection, const texture_region& destination, bool preserve, texture_plane plane)
{
	run(ctx, source, expanded, false, plane, {}, &destination, preserve, &projection);
}
void sample_transfer::import_region(context& ctx,id<MTLTexture> expanded,id<MTLTexture> multisampled,
	std::array<std::size_t,2> origin,texture_plane plane)
{
	run(ctx,multisampled,expanded,true,plane,origin);
}
} // namespace metal4
