#pragma once

#include "M4SampleTransfer.h"
#include "M4SurfaceTypeless.h"

namespace metal4
{
// Copy in RSX's spatial sample raster (1x1/2x1/2x2), not sample averaging.
// Coordinates are rendered pixels; each source/destination box is expanded
// with its own sample count before nearest/typeless transfer. Equal unflipped
// MSAA copies still use the bit-exact native texture-copy path. Raster sample
// bridges retain their explicit finite [0,1] depth validation requirement.
// Scratch and expansion cover only the copied windows. Partial imports load
// selected target attachments and preserve exterior samples without reading
// or reconstructing their background in a separate sample raster.
class surface_sample_blitter final
{
	sample_transfer m_samples;
	texture_blitter m_blits;
	surface_typeless_transfer m_typeless;

  public:
	explicit surface_sample_blitter(context& ctx);
	void nearest(context& ctx, surface_format source_format, id<MTLTexture> source, const texture_region& source_region,
				 surface_format destination_format, id<MTLTexture> destination, const texture_region& destination_region,
				 texture_plane plane = texture_plane::all, std::array<bool, 3> flip = {}, bool typeless = false);
};
} // namespace metal4
