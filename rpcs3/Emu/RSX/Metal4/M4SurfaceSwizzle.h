#pragma once

#include "M4Swizzle.h"
#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"

namespace metal4
{
// Byte permutation precedes/follows guest BE format conversion. Active guest
// storage origin is independent of native texture mip/slice/origin. Downloads
// write only active Morton pixels and preserve every unrelated storage cell.
// These bridges transfer one 2D slice from a 2D/3D Morton volume; not MSAA,
// banked/tiled guest allocations, mip-chain packing, surface cache or VM state.
class surface_swizzle_transfer final
{
	id<MTLDevice> m_device;
	swizzle_engine m_swizzle;
	surface_color_transfer m_color;
	surface_depth_transfer m_depth;
	[[nodiscard]] id<MTLBuffer> staging(context& ctx, id<MTLBuffer> guest, std::size_t offset,
		const swizzle_region& layout, std::size_t bytes, const texture_region& region) const;

public:
	explicit surface_swizzle_transfer(context& ctx);
	void upload_color(context& ctx, surface_color format, id<MTLBuffer> guest, std::size_t offset,
		const swizzle_region& layout, id<MTLTexture> texture, const texture_region& region) const;
	void download_color(context& ctx, surface_color format, id<MTLTexture> texture, const texture_region& region,
		id<MTLBuffer> guest, std::size_t offset, const swizzle_region& layout) const;
	void upload_depth(context& ctx, surface_depth format, id<MTLBuffer> guest, std::size_t offset,
		const swizzle_region& layout, id<MTLTexture> texture, const texture_region& region,
		texture_plane plane = texture_plane::all) const;
	void download_depth(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
		id<MTLBuffer> guest, std::size_t offset, const swizzle_region& layout) const;
};
}
