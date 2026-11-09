#include "M4TextureBlit.h"

#include <vector>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
void texture_blitter::nearest(context& ctx, id<MTLTexture> source, const texture_region& sr,
							  id<MTLTexture> destination, const texture_region& dr, texture_plane plane, std::array<bool, 3> flip,
							  nearest_projection projection) const
{
	if (!source || !destination || source.pixelFormat != destination.pixelFormat || source.sampleCount != destination.sampleCount)
		throw std::invalid_argument("Native nearest texture blit needs matching formats/sample counts");
	const auto size = [](MTLSize value)
	{
		return std::array{value.width, value.height, value.depth};
	};
	const bool whole = (projection.extent == std::array<std::uint32_t,3>{} ||
		projection.extent == std::array<std::uint32_t,3>{static_cast<std::uint32_t>(dr.size.width),static_cast<std::uint32_t>(dr.size.height),static_cast<std::uint32_t>(dr.size.depth)}) &&
		projection.offset == std::array<std::uint32_t,3>{} && projection.source_extent == std::array<std::uint32_t,3>{} && projection.source_origin == std::array<std::uint32_t,3>{};
	if (whole && size(sr.size) == size(dr.size) && !flip[0] && !flip[1] && !flip[2])
	{
		copy_texture(ctx, source, sr, destination, dr, plane);
		return;
	}
	if (source.sampleCount != 1)
		throw std::invalid_argument("Scaled/flipped native MSAA requires a separate sample path");
	(void)validate_texture_region(ctx, source, sr, plane);
	(void)validate_texture_region(ctx, destination, dr, plane);
	const auto narrow=[](MTLSize s)
	{
		if(s.width>UINT32_MAX || s.height>UINT32_MAX || s.depth>UINT32_MAX)throw std::overflow_error("Native nearest extent exceeds uint32");
		return std::array{static_cast<std::uint32_t>(s.width),static_cast<std::uint32_t>(s.height),static_cast<std::uint32_t>(s.depth)};
	};
	const auto window=plan_nearest_input_window(narrow(sr.size),narrow(dr.size),flip,projection);
	projection=window.projection;
	auto selected=sr;selected.origin.x+=window.origin[0];selected.origin.y+=window.origin[1];selected.origin.z+=window.origin[2];
	selected.size=MTLSizeMake(window.extent[0],window.extent[1],window.extent[2]);
	for(unsigned axis=0;axis<3;++axis)projection.source_origin[axis]+=window.origin[axis];
	const auto image = [&](MTLSize dimensions, texel_block block)
	{
		const auto s = size(dimensions);
		for (const auto extent : s)
			if (extent > std::numeric_limits<std::uint32_t>::max())
				throw std::overflow_error("Native nearest extent exceeds uint32");
		const auto minimal = transfer_layout(block, s[0], s[1], s[2]);
		if (minimal.bytes_per_row > std::numeric_limits<std::size_t>::max() - 255)
			throw std::overflow_error("Native nearest staging pitch overflow");
		return nearest_image{{static_cast<std::uint32_t>(s[0]), static_cast<std::uint32_t>(s[1]), static_cast<std::uint32_t>(s[2])},
							 (minimal.bytes_per_row + 255) & ~std::size_t(255),
							 0};
	};
	struct transfer
	{
		texture_plane aspect;
		nearest_image source, destination;
		nearest_layout layout;
		std::uint32_t bytes;
		id<MTLBuffer> input, output;
	};
	std::vector<transfer> transfers;
	const auto prepare = [&](texture_plane aspect)
	{
		const auto block = storage_block(source.pixelFormat, aspect);
		if (block.width != 1 || block.height != 1)
			throw std::invalid_argument("Native nearest scale/flip does not decode compressed blocks");
		const auto a = image(selected.size, block), b = image(dr.size, block);
		const auto bytes = static_cast<std::uint32_t>(block.bytes);
		const auto layout = plan_nearest(a, b, bytes);
		transfers.push_back({aspect, a, b, layout, bytes,
							 make_buffer(ctx, layout.source.allocation_size, buffer_storage::private_gpu),
							 make_buffer(ctx, layout.destination.allocation_size, buffer_storage::private_gpu)});
	};
	if (source.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && plane == texture_plane::all)
	{
		prepare(texture_plane::depth);
		prepare(texture_plane::stencil);
	}
	else
		prepare(plane);
	for (const auto& t : transfers)
		texture_to_buffer(ctx, source, selected, t.input, 0, t.layout.source.bytes_per_row, t.layout.source.bytes_per_image, t.aspect);
	for (const auto& t : transfers)
		m_transfer.nearest(ctx, t.input, 0, t.source, t.output, 0, t.destination, t.bytes, flip, projection);
	for (const auto& t : transfers)
		buffer_to_texture(ctx, t.output, 0, destination, dr, t.layout.destination.bytes_per_row, t.layout.destination.bytes_per_image, t.aspect);
}
} // namespace metal4
