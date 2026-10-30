#include "M4SurfaceSwizzle.h"
#include "M4SurfaceValidation.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
surface_swizzle_transfer::surface_swizzle_transfer(context& ctx)
	: m_device(ctx.device()), m_swizzle(ctx), m_color(ctx), m_depth(ctx)
{
}

id<MTLBuffer> surface_swizzle_transfer::staging(context& ctx, id<MTLBuffer> guest, std::size_t offset,
	const swizzle_region& layout, std::size_t bytes, const texture_region& region) const
{
	if (ctx.device() != m_device || !guest || guest.device != m_device || layout.size[0] != region.size.width ||
		layout.size[1] != region.size.height || layout.size[2] != 1 || region.size.depth != 1)
		throw std::invalid_argument("Morton surface bridge device or active dimensions are invalid");
	const auto plan = plan_swizzle(layout, bytes);
	if (offset > guest.length || plan.storage_bytes > guest.length - offset)
		throw std::out_of_range("Morton surface storage exceeds guest buffer allocation");
	return make_buffer(ctx, plan.linear.allocation_size, buffer_storage::private_gpu);
}

void surface_swizzle_transfer::upload_color(context& ctx, surface_color format, id<MTLBuffer> guest, std::size_t offset,
	const swizzle_region& layout, id<MTLTexture> texture, const texture_region& region) const
{
	const auto metadata = describe_surface_color(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	const auto linear = staging(ctx, guest, offset, layout, metadata.layout.pixel_bytes, region);
	m_swizzle.convert(ctx, guest, offset, linear, 0, layout, metadata.layout.pixel_bytes);
	m_color.upload(ctx, format, linear, {}, texture, region);
}

void surface_swizzle_transfer::download_color(context& ctx, surface_color format, id<MTLTexture> texture, const texture_region& region,
	id<MTLBuffer> guest, std::size_t offset, const swizzle_region& layout) const
{
	const auto metadata = describe_surface_color(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	const auto linear = staging(ctx, guest, offset, layout, metadata.layout.pixel_bytes, region);
	m_color.download(ctx, format, texture, region, linear, {});
	m_swizzle.convert(ctx, guest, offset, linear, 0, layout, metadata.layout.pixel_bytes, 0, 0, true);
}

void surface_swizzle_transfer::upload_depth(context& ctx, surface_depth format, id<MTLBuffer> guest, std::size_t offset,
	const swizzle_region& layout, id<MTLTexture> texture, const texture_region& region) const
{
	const auto metadata = describe_surface_depth(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	const auto linear = staging(ctx, guest, offset, layout, metadata.guest_bytes, region);
	m_swizzle.convert(ctx, guest, offset, linear, 0, layout, metadata.guest_bytes);
	m_depth.upload(ctx, format, linear, {}, texture, region);
}

void surface_swizzle_transfer::download_depth(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
	id<MTLBuffer> guest, std::size_t offset, const swizzle_region& layout) const
{
	const auto metadata = describe_surface_depth(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	const auto linear = staging(ctx, guest, offset, layout, metadata.guest_bytes, region);
	m_depth.download(ctx, format, texture, region, linear, {});
	m_swizzle.convert(ctx, guest, offset, linear, 0, layout, metadata.guest_bytes, 0, 0, true);
}
}
