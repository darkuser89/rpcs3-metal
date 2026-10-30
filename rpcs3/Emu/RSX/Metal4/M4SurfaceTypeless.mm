#include "M4SurfaceTypeless.h"
#include "M4SurfaceValidation.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
MTLPixelFormat surface_native_format(surface_format format)
{
	if (const auto color = std::get_if<surface_color>(&format))
		return describe_surface_color(*color).format;
	return describe_surface_depth(std::get<surface_depth>(format)).format;
}
surface_typeless_transfer::surface_typeless_transfer(context& ctx)
	: m_color(ctx), m_depth(ctx), m_transfer(ctx), m_blits(ctx)
{
}
void surface_typeless_transfer::nearest(context& ctx, surface_format sf, id<MTLTexture> source, const texture_region& sr,
										surface_format df, id<MTLTexture> destination, const texture_region& dr, std::array<bool, 3> flip) const
{
	validate_surface_texture(ctx, surface_native_format(sf), source, sr);
	validate_surface_texture(ctx, surface_native_format(df), destination, dr);
	if (flip[2])
		throw std::invalid_argument("Typeless surface transfers operate on one 2D slice, not a flipped volume");
	const auto layout = plan_typeless(sf, df, sr.size.width, sr.size.height, dr.size.width, dr.size.height);
	if (sf == df)
	{
		m_blits.nearest(ctx, source, sr, destination, dr, texture_plane::all, flip);
		return;
	}
	const auto packed_source = make_buffer(ctx, layout.source.allocation_size, buffer_storage::private_gpu);
	const auto packed_destination = make_buffer(ctx, layout.destination.allocation_size, buffer_storage::private_gpu);
	if (const auto color = std::get_if<surface_color>(&sf))
		m_color.download(ctx, *color, source, sr, packed_source, {0, layout.source.bytes_per_row});
	else
		m_depth.download(ctx, std::get<surface_depth>(sf), source, sr, packed_source, {0, layout.source.bytes_per_row});
	const nearest_image input{{layout.source_width_as_destination, static_cast<std::uint32_t>(sr.size.height), 1}, layout.source.bytes_per_row, 0};
	const nearest_image output{{static_cast<std::uint32_t>(dr.size.width), static_cast<std::uint32_t>(dr.size.height), 1}, layout.destination.bytes_per_row, 0};
	m_transfer.nearest(ctx, packed_source, 0, input, packed_destination, 0, output, static_cast<std::uint32_t>(surface_guest_bytes(df)), flip);
	if (const auto color = std::get_if<surface_color>(&df))
		m_color.upload(ctx, *color, packed_destination, {0, layout.destination.bytes_per_row}, destination, dr);
	else
		m_depth.upload(ctx, std::get<surface_depth>(df), packed_destination, {0, layout.destination.bytes_per_row}, destination, dr);
}
} // namespace metal4
