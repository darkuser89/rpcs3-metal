#pragma once

#include "M4Texture.h"
#include "M4Transfer.h"

namespace metal4
{
class texture_blitter final
{
	transfer_engine m_transfer;

  public:
	explicit texture_blitter(context& ctx) : m_transfer(ctx) {}
	// Nearest sample at destination texel centers; selected native texel/plane
	// bits are copied unchanged, including f32 NaNs and unclamped depth. Source
	// axes may flip within the positive region. Snapshot all source planes
	// before any destination write, including self/parent-view/heap aliases.
	// Scaling/flips require single samples and uncompressed matching formats.
	// Plain equal-size copies use copy_texture, including its native MSAA path.
	void nearest(context& ctx, id<MTLTexture> source, const texture_region& source_region,
				 id<MTLTexture> destination, const texture_region& destination_region,
				 texture_plane plane = texture_plane::all, std::array<bool, 3> flip = {}, nearest_projection projection = {}) const;
};
} // namespace metal4
