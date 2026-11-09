#include "M4SurfaceDepth.h"
#include "M4SurfaceDepthKernels.h"
#include "M4SurfaceValidation.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <cstring>

namespace metal4
{
surface_depth_descriptor describe_surface_depth(surface_depth format)
{
	switch (format)
	{
	case surface_depth::z16_uint:
		return {MTLPixelFormatDepth16Unorm, 2, 2, false};
	case surface_depth::z24s8_uint:
		return {MTLPixelFormatDepth32Float_Stencil8, 4, 4, true};
	case surface_depth::z16_float:
		return {MTLPixelFormatDepth32Float, 2, 4, false};
	case surface_depth::z24s8_float:
		return {MTLPixelFormatDepth32Float_Stencil8, 4, 4, true};
	}
	throw std::invalid_argument("Invalid RSX depth surface format");
}

namespace
{
struct depth_parameters
{
	std::uint64_t guest_offset, depth_offset, stencil_offset;
	std::uint64_t guest_pitch, depth_pitch, stencil_pitch;
	std::uint64_t width, count, first;
	std::uint32_t format, gather;
};
static_assert(sizeof(depth_parameters) == 80 && offsetof(depth_parameters, format) == 72);

texture_buffer_layout range(context& ctx, const depth_transfer_plane& plane, std::size_t bytes,
							std::size_t width, std::size_t height)
{
	const auto layout = transfer_layout({1, 1, bytes}, width, height, 1, plane.bytes.row_pitch);
	if (!plane.buffer || plane.buffer.device != ctx.device() || plane.bytes.offset > plane.buffer.length ||
		layout.allocation_size > plane.buffer.length - plane.bytes.offset)
		throw std::out_of_range("Native depth plane byte range exceeds allocation");
	return layout;
}

void no_overlap(const depth_transfer_plane& a, const texture_buffer_layout& al,
				const depth_transfer_plane& b, const texture_buffer_layout& bl)
{
	if (a.buffer == b.buffer && a.bytes.offset < b.bytes.offset + bl.allocation_size &&
		b.bytes.offset < a.bytes.offset + al.allocation_size)
		throw std::invalid_argument("Native depth transfer planes cannot overlap");
}

depth_transfer_plane staging(context& ctx, std::size_t bytes, const texture_region& region)
{
	const auto row = checked_texture_product(region.size.width, bytes);
	if (row > std::numeric_limits<std::size_t>::max() - 255)
		throw std::overflow_error("Depth plane pitch overflow");
	const auto pitch = (row + 255) & ~std::size_t(255);
	return {make_buffer(ctx, checked_texture_product(pitch, region.size.height), buffer_storage::private_gpu), {0, pitch}};
}
} // namespace

surface_depth_transfer::surface_depth_transfer(context& ctx) : m_device(ctx.device())
{
	helper_catalog compiler(ctx);
	const auto library = compiler.library(surface_depth_kernels, @"RPCS3 native depth/stencil surface codec");

	m_pipeline = compiler.compute(library, @"rpcs3_surface_depth");
	m_patch_pipeline = compiler.compute(library, @"rpcs3_surface_depth_patch");
}

void surface_depth_transfer::convert(context& ctx, surface_depth format, std::size_t width, std::size_t height,
									 id<MTLBuffer> guest, surface_byte_region bytes, const depth_transfer_plane& depth, const depth_transfer_plane& stencil, bool gather, id<MTLBuffer> prepared) const
{
	if (ctx.device() != m_device)
		throw std::invalid_argument("Depth transfer belongs to another device");
	const auto metadata = describe_surface_depth(format);
	const depth_transfer_plane guest_plane{guest, bytes};
	const auto gl = range(ctx, guest_plane, metadata.guest_bytes, width, height);
	const auto dl = range(ctx, depth, metadata.depth_bytes, width, height);
	no_overlap(guest_plane, gl, depth, dl);
	texture_buffer_layout sl{};
	if (metadata.stencil)
	{
		sl = range(ctx, stencil, 1, width, height);
		no_overlap(guest_plane, gl, stencil, sl);
		no_overlap(depth, dl, stencil, sl);
	}
	else if (stencil.buffer)
		throw std::invalid_argument("Z16 surface does not have a stencil plane");
	const std::array<std::byte, 16> zero{};
	const auto dummy = !metadata.stencil ?
		(prepared ? upload_slice{prepared, 0, zero.size()} : ctx.upload(zero, 16)) : upload_slice{};
	bindings resources;
	resources.set_buffer(0, guest);
	resources.set_buffer(1, depth.buffer);
	resources.set_buffer(2, metadata.stencil ? stencil.buffer : dummy.buffer, metadata.stencil ? 0 : dummy.offset);
	const auto pixels = checked_texture_product(width, height);
	constexpr std::size_t max_count = 1u << 30;
	for (std::size_t first = 0; first < pixels;)
	{
		const auto count = std::min(pixels - first, max_count);
		const depth_parameters parameters{bytes.offset, depth.bytes.offset, stencil.bytes.offset, gl.bytes_per_row, dl.bytes_per_row,
										  sl.bytes_per_row, width, count, first, static_cast<std::uint32_t>(format), gather ? 1u : 0u};
		const auto uniforms = prepared ? upload_slice{prepared, (first / max_count) * 256, sizeof(parameters)} :
			ctx.upload(std::as_bytes(std::span{&parameters, 1}), 256);
		resources.set_buffer(3, uniforms.buffer, uniforms.offset);
		const std::array accesses{
			context::accessed_resource{guest, bytes.offset, gl.allocation_size, gather},
			context::accessed_resource{depth.buffer, depth.bytes.offset, dl.allocation_size, !gather},
			context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size},
			context::accessed_resource{stencil.buffer, stencil.bytes.offset, sl.allocation_size, !gather}};
		ctx.checked_dispatch(m_pipeline, resources, std::span{accesses}.first(metadata.stencil ? 4 : 3), count);
		first += count;
	}
}

void surface_depth_transfer::scatter(context& ctx, surface_depth format, std::size_t width, std::size_t height,
									 id<MTLBuffer> guest, surface_byte_region bytes, const depth_transfer_plane& depth, const depth_transfer_plane& stencil) const
{
	convert(ctx, format, width, height, guest, bytes, depth, stencil, false);
}
void surface_depth_transfer::gather(context& ctx, surface_depth format, std::size_t width, std::size_t height,
									const depth_transfer_plane& depth, const depth_transfer_plane& stencil, id<MTLBuffer> guest, surface_byte_region bytes) const
{
	convert(ctx, format, width, height, guest, bytes, depth, stencil, true);
}
surface_depth_transfer::transfer_storage surface_depth_transfer::prepare_transfer(context& ctx, surface_depth format,
	id<MTLBuffer> guest, surface_byte_region bytes, const texture_region& region, bool gather) const
{
	if (ctx.device() != m_device) throw std::invalid_argument("Depth transfer belongs to another device");
	const auto metadata = describe_surface_depth(format);
	const auto width = region.size.width, height = region.size.height;
	const auto gl = range(ctx, {guest, bytes}, metadata.guest_bytes, width, height);
	const auto depth = staging(ctx, metadata.depth_bytes, region);
	const auto stencil = metadata.stencil ? staging(ctx, 1, region) : depth_transfer_plane{};
	const auto pixels = checked_texture_product(width, height);
	constexpr std::size_t max_count = 1u << 30;
	static_assert(sizeof(depth_parameters) <= 256);
	const auto uniforms = make_buffer(ctx, checked_texture_product((pixels - 1) / max_count + 1, 256), buffer_storage::shared);
	auto* output = static_cast<std::byte*>(uniforms.contents);
	for (std::size_t first = 0; first < pixels;)
	{
		const auto count = std::min(pixels - first, max_count);
		const depth_parameters params{bytes.offset, depth.bytes.offset, stencil.bytes.offset, gl.bytes_per_row,
			depth.bytes.row_pitch, stencil.bytes.row_pitch, width, count, first, static_cast<std::uint32_t>(format), gather ? 1u : 0u};
		std::memcpy(output, &params, sizeof(params));
		output += 256;
		first += count;
	}
	return {depth, stencil, uniforms};
}

void surface_depth_transfer::upload(context& ctx, surface_depth format, id<MTLBuffer> guest, surface_byte_region bytes,
	id<MTLTexture> texture, const texture_region& region, texture_plane plane) const
{
	upload_impl(ctx, format, guest, bytes, texture, region, nullptr, plane);
}
void surface_depth_transfer::upload_impl(context& ctx, surface_depth format, id<MTLBuffer> guest, surface_byte_region bytes,
									id<MTLTexture> texture, const texture_region& region, const transfer_storage* prepared, texture_plane plane) const
{
	if (ctx.device() != m_device)
		throw std::invalid_argument("Depth transfer belongs to another device");
	const auto metadata = describe_surface_depth(format);
	if (plane != texture_plane::all && plane != texture_plane::depth && (plane != texture_plane::stencil || !metadata.stencil))
		throw std::invalid_argument("Native depth upload selected an unavailable plane");
	validate_surface_texture(ctx, metadata.format, texture, region);
	(void)range(ctx, {guest, bytes}, metadata.guest_bytes, region.size.width, region.size.height);
	const auto depth = prepared ? prepared->depth : staging(ctx, metadata.depth_bytes, region);
	const auto stencil = prepared ? prepared->stencil : metadata.stencil ? staging(ctx, 1, region) : depth_transfer_plane{};
	convert(ctx, format, region.size.width, region.size.height, guest, bytes, depth, stencil, false,
		prepared ? prepared->uniforms : nil);
	if (plane != texture_plane::stencil)
		buffer_to_texture(ctx, depth.buffer, 0, texture, region, depth.bytes.row_pitch, 0, texture_plane::depth);
	if (metadata.stencil && plane != texture_plane::depth)
		buffer_to_texture(ctx, stencil.buffer, 0, texture, region, stencil.bytes.row_pitch, 0, texture_plane::stencil);
}
void surface_depth_transfer::download(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
									  id<MTLBuffer> guest, surface_byte_region bytes) const
{
	download_impl(ctx, format, texture, region, guest, bytes, nullptr);
}
void surface_depth_transfer::download_impl(context& ctx, surface_depth format, id<MTLTexture> texture,
	const texture_region& region, id<MTLBuffer> guest, surface_byte_region bytes, const transfer_storage* prepared) const
{
	if (ctx.device() != m_device)
		throw std::invalid_argument("Depth transfer belongs to another device");
	const auto metadata = describe_surface_depth(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	(void)range(ctx, {guest, bytes}, metadata.guest_bytes, region.size.width, region.size.height);
	const auto depth = prepared ? prepared->depth : staging(ctx, metadata.depth_bytes, region);
	const auto stencil = prepared ? prepared->stencil : metadata.stencil ? staging(ctx, 1, region) : depth_transfer_plane{};
	texture_to_buffer(ctx, texture, region, depth.buffer, 0, depth.bytes.row_pitch, 0, texture_plane::depth);
	if (metadata.stencil)
		texture_to_buffer(ctx, texture, region, stencil.buffer, 0, stencil.bytes.row_pitch, 0, texture_plane::stencil);
	convert(ctx, format, region.size.width, region.size.height, guest, bytes, depth, stencil, true,
		prepared ? prepared->uniforms : nil);
}

void surface_depth_transfer::patch(context& ctx, surface_depth format, id<MTLBuffer> guest, id<MTLBuffer> mask,
	surface_byte_region bytes, id<MTLTexture> texture, const texture_region& region) const
{
	if (ctx.device() != m_device)
		throw std::invalid_argument("Depth patch belongs to another device");
	const auto metadata = describe_surface_depth(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	const auto width = region.size.width, height = region.size.height;
	const auto gl = range(ctx, {guest, bytes}, metadata.guest_bytes, width, height);
	(void)range(ctx, {mask, bytes}, metadata.guest_bytes, width, height);
	const auto depth = staging(ctx, metadata.depth_bytes, region);
	const auto stencil = metadata.stencil ? staging(ctx, 1, region) : depth_transfer_plane{};
	// Retain raw native bits, including low float mantissa bits which do not
	// survive a guest-word round trip. Stencil-only writes never recode depth.
	texture_to_buffer(ctx, texture, region, depth.buffer, 0, depth.bytes.row_pitch, 0, texture_plane::depth);
	if (metadata.stencil)
		texture_to_buffer(ctx, texture, region, stencil.buffer, 0, stencil.bytes.row_pitch, 0, texture_plane::stencil);
	const std::array<std::byte, 16> zero{};
	const auto dummy = metadata.stencil ? upload_slice{} : ctx.upload(zero, 16);
	bindings resources;
	resources.set_buffer(0, guest);
	resources.set_buffer(1, depth.buffer);
	resources.set_buffer(2, metadata.stencil ? stencil.buffer : dummy.buffer, metadata.stencil ? 0 : dummy.offset);
	resources.set_buffer(4, mask);
	const auto pixels = checked_texture_product(width, height);
	constexpr std::size_t max_count = 1u << 30;
	for (std::size_t first = 0; first < pixels;)
	{
		const auto count = std::min(pixels - first, max_count);
		const depth_parameters parameters{bytes.offset, 0, 0, gl.bytes_per_row, depth.bytes.row_pitch,
			stencil.bytes.row_pitch, width, count, first, static_cast<std::uint32_t>(format), 0};
		const auto uniforms = ctx.upload(std::as_bytes(std::span{&parameters, 1}), 256);
		resources.set_buffer(3, uniforms.buffer, uniforms.offset);
		const std::array accesses{context::accessed_resource{guest, bytes.offset, gl.allocation_size},
			context::accessed_resource{depth.buffer, 0, depth.buffer.length, true},
			context::accessed_resource{mask, bytes.offset, gl.allocation_size},
			context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size},
			context::accessed_resource{stencil.buffer, 0, stencil.buffer.length, true}};
		ctx.checked_dispatch(m_patch_pipeline, resources, std::span{accesses}.first(metadata.stencil ? 5 : 4), count);
		first += count;
	}
	buffer_to_texture(ctx, depth.buffer, 0, texture, region, depth.bytes.row_pitch, 0, texture_plane::depth);
	if (metadata.stencil)
		buffer_to_texture(ctx, stencil.buffer, 0, texture, region, stencil.bytes.row_pitch, 0, texture_plane::stencil);
}
} // namespace metal4
