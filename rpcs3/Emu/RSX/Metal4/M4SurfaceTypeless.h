#pragma once

#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"
#include "M4TextureBlit.h"
#include "M4TypelessLayout.h"

namespace metal4
{
[[nodiscard]] MTLPixelFormat surface_native_format(surface_format format);

// Native single-sample 2D surface typeless transfer. Packs source guest-format
// BE storage bits, reinterprets each row as destination-format guest texels,
// then applies center-nearest scaling/flips and unpacks to destination native
// planes. No guest VM publication or CPU readback. Source snapshot precedes
// every destination write, including aliases and equal-native depth classes.
// Matching guest formats use the existing raw native nearest/copy path.
class surface_typeless_transfer final
{
	surface_color_transfer m_color;
	surface_depth_transfer m_depth;
	transfer_engine m_transfer;
	texture_blitter m_blits;

  public:
	explicit surface_typeless_transfer(context& ctx);
	void nearest(context& ctx, surface_format source_format, id<MTLTexture> source, const texture_region& source_region,
				 surface_format destination_format, id<MTLTexture> destination, const texture_region& destination_region,
				 std::array<bool, 3> flip = {}, nearest_projection projection = {},
				 texture_plane destination_plane = texture_plane::all) const;
};
} // namespace metal4
