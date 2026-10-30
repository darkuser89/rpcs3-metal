#pragma once

#include "M4SurfaceLayout.h"
#include "M4Texture.h"
#include "M4Pipeline.h"
#include "M4Resources.h"

namespace metal4
{
struct surface_depth_descriptor
{
	MTLPixelFormat format;
	std::uint32_t guest_bytes, depth_bytes;
	bool stencil;
};
[[nodiscard]] surface_depth_descriptor describe_surface_depth(surface_depth format);

struct depth_transfer_plane
{
	id<MTLBuffer> buffer = nil;
	surface_byte_region bytes;
};

// Single-sample linear surface codec. Guest words are BE: Z16, or D24 in
// bits31..8 with S8 in bits7..0. Native combined D32F/S8 uses separate planes,
// never a guessed packed 8-byte transfer. Plane offsets/pitches are independent.
// Z16F is RSX E4M12, Z24F is the upper 24 bits of a nonnegative float32 word.
// Normalized Z24 gathers truncate float32(depth)*16777215 as in VK; malformed
// native depth values are explicitly clamped, with NaN->0, avoiding UB casts.
// Call during begin()/submit() recording outside a live render facade. Distinct
// heap aliases, guest swizzles/MSAA expansion and VM coherence are caller-owned.
class surface_depth_transfer final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;
	void convert(context& ctx, surface_depth format, std::size_t width, std::size_t height,
				 id<MTLBuffer> guest, surface_byte_region bytes, const depth_transfer_plane& depth,
				 const depth_transfer_plane& stencil, bool gather) const;

  public:
	explicit surface_depth_transfer(context& ctx);
	void scatter(context& ctx, surface_depth format, std::size_t width, std::size_t height,
				 id<MTLBuffer> guest, surface_byte_region bytes, const depth_transfer_plane& depth,
				 const depth_transfer_plane& stencil = {}) const;
	void gather(context& ctx, surface_depth format, std::size_t width, std::size_t height,
				const depth_transfer_plane& depth, const depth_transfer_plane& stencil,
				id<MTLBuffer> guest, surface_byte_region bytes) const;
	void upload(context& ctx, surface_depth format, id<MTLBuffer> guest, surface_byte_region bytes,
				id<MTLTexture> texture, const texture_region& region) const;
	void download(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
				  id<MTLBuffer> guest, surface_byte_region bytes) const;
};
} // namespace metal4
