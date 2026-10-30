#include "M4SurfaceColor.h"
#include "M4SurfaceColorKernels.h"
#include "M4SurfaceValidation.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>

namespace metal4
{
surface_color_descriptor describe_surface_color(surface_color format)
{
	const auto layout = color_layout(format);
	auto swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen,
		MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
	MTLPixelFormat native;
	switch (format)
	{
	// The guest A1R5G5B5 bit word and native BGR5A1 LE word both have
	// blue in bits 0..4 and alpha in bit15. Endian conversion suffices.
	case surface_color::x1rgb5_zero: case surface_color::x1rgb5_one: native = MTLPixelFormatBGR5A1Unorm; break;
	case surface_color::rgb565: native = MTLPixelFormatB5G6R5Unorm; break;
	case surface_color::xrgb8_zero: case surface_color::xrgb8_one: case surface_color::argb8:
		native = MTLPixelFormatBGRA8Unorm; break;
	case surface_color::xbgr8_zero: case surface_color::xbgr8_one: case surface_color::abgr8:
		native = MTLPixelFormatRGBA8Unorm; break;
	case surface_color::b8:
		native = MTLPixelFormatR8Unorm;
		swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne);
		break;
	case surface_color::gb8:
		native = MTLPixelFormatRG8Unorm;
		swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleGreen);
		break;
	case surface_color::rgba16f: native = MTLPixelFormatRGBA16Float; break;
	case surface_color::rgba32f: native = MTLPixelFormatRGBA32Float; break;
	case surface_color::r32f:
		native = MTLPixelFormatR32Float;
		swizzle = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed);
		break;
	default: throw std::invalid_argument("Invalid RSX surface color format");
	}
	if (layout.export_remap == color_export_remap::rgb_zero) swizzle.alpha = MTLTextureSwizzleZero;
	if (layout.export_remap == color_export_remap::rgb_one) swizzle.alpha = MTLTextureSwizzleOne;
	return {native, swizzle, layout};
}

void configure_surface_color(render_pipeline_configuration& pipeline, fragment_program_options& fragment,
	std::size_t slot, surface_color format, std::uint32_t guest_write_mask)
{
	if (slot >= 4) throw std::out_of_range("RSX surface slot exceeds four color outputs");
	const auto descriptor = describe_surface_color(format);
	const auto mask = color_write_mask(format, guest_write_mask);
	const bool remapped_blend = descriptor.layout.export_remap != color_export_remap::rgba;
	pipeline.colors[slot] = descriptor.format;
	fragment.raster_sample_count = pipeline.state.sample_count;
	pipeline.shader_half_storage_mask = (pipeline.shader_half_storage_mask & ~(1u << slot)) | (format == surface_color::rgba16f ? (1u << slot) : 0);
	fragment.half_storage_mask = (fragment.half_storage_mask & ~(1u << slot)) | (format == surface_color::rgba16f ? (1u << slot) : 0);
	pipeline.state.colors[slot].write_mask = mask;
	pipeline.shader_blend_mask = (pipeline.shader_blend_mask & ~(1u << slot)) | (remapped_blend ? (1u << slot) : 0);
	fragment.color_remap[slot] = descriptor.layout.export_remap;
	pipeline.color_remap[slot] = descriptor.layout.export_remap;
	fragment.color_count = std::max(fragment.color_count, static_cast<std::uint32_t>(slot + 1));
}

namespace
{
struct color_parameters
{
	std::uint64_t source_offset, destination_offset;
	std::uint64_t source_pitch, destination_pitch;
	std::uint64_t width, pixel_count, first_pixel;
	std::uint32_t pixel_bytes, element_bytes;
};
static_assert(sizeof(color_parameters) == 64 && offsetof(color_parameters, pixel_bytes) == 56);

void validate_range(context& ctx, id<MTLBuffer> buffer, const surface_byte_region& region,
	const texture_buffer_layout& layout)
{
	if (!buffer || buffer.device != ctx.device() || region.offset > buffer.length ||
		layout.allocation_size > buffer.length - region.offset)
		throw std::out_of_range("RSX surface byte region exceeds native buffer allocation");
}

std::size_t staging_pitch(surface_color format, const texture_region& region)
{
	const auto row = checked_texture_product(region.size.width, color_layout(format).pixel_bytes);
	if (row > std::numeric_limits<std::size_t>::max() - 255) throw std::overflow_error("Surface staging pitch overflow");
	return (row + 255) & ~std::size_t(255);
}
}

surface_color_transfer::surface_color_transfer(context& ctx) : m_device(ctx.device())
{
	helper_catalog compiler(ctx);
	const auto library = compiler.library(surface_color_kernels, @"RPCS3 native surface color byte conversion");

	m_pipeline = compiler.compute(library, @"rpcs3_surface_color_bytes");
}

void surface_color_transfer::convert(context& ctx, surface_color format, std::size_t width, std::size_t height,
	id<MTLBuffer> source, const surface_byte_region& input, id<MTLBuffer> destination, const surface_byte_region& output) const
{
	if (ctx.device() != m_device) throw std::invalid_argument("Native surface transfer belongs to another Metal device");
	const auto layout = color_layout(format);
	const auto source_layout = surface_byte_layout(format, width, height, input);
	const auto destination_layout = surface_byte_layout(format, width, height, output);
	validate_range(ctx, source, input, source_layout);
	validate_range(ctx, destination, output, destination_layout);
	if (source == destination && input.offset < output.offset + destination_layout.allocation_size &&
		output.offset < input.offset + source_layout.allocation_size)
		throw std::invalid_argument("Surface color conversion source and destination cannot overlap");
	const auto pixels = checked_texture_product(width, height);
	bindings resources;
	resources.set_buffer(0, source);
	resources.set_buffer(1, destination);
	ctx.keep_alive(m_pipeline);
	constexpr std::size_t max_count = 1u << 30;
	for (std::size_t first = 0; first < pixels;)
	{
		const auto count = std::min(pixels - first, max_count);
		const color_parameters params{input.offset, output.offset, source_layout.bytes_per_row, destination_layout.bytes_per_row,
			width, count, first, layout.pixel_bytes, layout.endian_element_bytes};
		const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
		resources.set_buffer(2, uniforms.buffer, uniforms.offset);
		const auto table = resources.snapshot(ctx);
		const auto encoder = ctx.compute();
		ctx.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
		[encoder setComputePipelineState:m_pipeline];
		[encoder setArgumentTable:table];
		[encoder dispatchThreads:MTLSizeMake(count, 1, 1)
			threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, m_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
		first += count;
	}
}

void surface_color_transfer::upload(context& ctx, surface_color format, id<MTLBuffer> guest, const surface_byte_region& input,
	id<MTLTexture> texture, const texture_region& region) const
{
	if (ctx.device() != m_device) throw std::invalid_argument("Native surface transfer belongs to another device");
	validate_surface_texture(ctx, describe_surface_color(format).format, texture, region);
	validate_range(ctx, guest, input, surface_byte_layout(format, region.size.width, region.size.height, input));
	const auto pitch = staging_pitch(format, region);
	const auto size = checked_texture_product(pitch, region.size.height);
	const auto native = make_buffer(ctx, size, buffer_storage::private_gpu);
	convert(ctx, format, region.size.width, region.size.height, guest, input, native, {0, pitch});
	buffer_to_texture(ctx, native, 0, texture, region, pitch);
}

void surface_color_transfer::download(context& ctx, surface_color format, id<MTLTexture> texture, const texture_region& region,
	id<MTLBuffer> guest, const surface_byte_region& output) const
{
	if (ctx.device() != m_device) throw std::invalid_argument("Native surface transfer belongs to another device");
	validate_surface_texture(ctx, describe_surface_color(format).format, texture, region);
	const auto pitch = staging_pitch(format, region);
	const auto size = checked_texture_product(pitch, region.size.height);
	// Validate destination before recording texture reads.
	validate_range(ctx, guest, output, surface_byte_layout(format, region.size.width, region.size.height, output));
	const auto native = make_buffer(ctx, size, buffer_storage::private_gpu);
	texture_to_buffer(ctx, texture, region, native, 0, pitch);
	convert(ctx, format, region.size.width, region.size.height, native, {0, pitch}, guest, output);
}
}
