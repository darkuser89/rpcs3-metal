#pragma once

#include "M4BankedTile.h"
#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"

namespace metal4
{
struct surface_banked_statistics
{
	// Encoding/allocation counters; completion and time are separate.
	std::uint64_t dispatches = 0, texels = 0, active_bytes = 0;
	std::uint64_t staging_buffers = 0, staging_bytes = 0;
	std::uint64_t fused_color_uploads = 0, fused_color_downloads = 0;
};
// Encode-only banked main-memory surface bridge. The complete tile buffer is
// owned by the caller;unrelated bytes survive downloads. No VM access or waits.
// Local VRAM tiling is transparent and must not be attached to this bridge.
class surface_banked_transfer final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;
	surface_depth_transfer m_depth;
	mutable surface_banked_statistics m_statistics;
	[[nodiscard]] id<MTLBuffer> staging(context& ctx, id<MTLBuffer> guest,
		const guest_surface_layout& image, const banked_tile_layout& tile, const texture_region& region, std::size_t pitch = 0) const;
	void convert(context& ctx, const guest_surface_layout& image, const banked_tile_layout& tile,
		id<MTLBuffer> guest, id<MTLBuffer> linear, bool encode, std::size_t pitch = 0, std::uint32_t element_bytes = 1) const;

  public:
	explicit surface_banked_transfer(context& ctx);
	[[nodiscard]] surface_banked_statistics statistics() const noexcept { return m_statistics; }
	void upload_color(context& ctx, surface_color format, id<MTLBuffer> guest, const guest_surface_layout& image,
		const banked_tile_layout& tile, id<MTLTexture> texture, const texture_region& region) const;
	void download_color(context& ctx, surface_color format, id<MTLTexture> texture, const texture_region& region,
		id<MTLBuffer> guest, const guest_surface_layout& image, const banked_tile_layout& tile) const;
	void upload_depth(context& ctx, surface_depth format, id<MTLBuffer> guest, const guest_surface_layout& image,
		const banked_tile_layout& tile, id<MTLTexture> texture, const texture_region& region) const;
	void download_depth(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
		id<MTLBuffer> guest, const guest_surface_layout& image, const banked_tile_layout& tile) const;
};
} // namespace metal4
