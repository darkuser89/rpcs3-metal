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
	id<MTLComputePipelineState> m_patch_pipeline;
	void convert(context& ctx, surface_depth format, std::size_t width, std::size_t height,
				 id<MTLBuffer> guest, surface_byte_region bytes, const depth_transfer_plane& depth,
				 const depth_transfer_plane& stencil, bool gather, id<MTLBuffer> prepared = nil) const;

	friend class guest_texture_resolver;
	friend class surface_banked_transfer;
	struct transfer_storage
	{
		depth_transfer_plane depth, stencil;
		id<MTLBuffer> uniforms;
	};
	[[nodiscard]] transfer_storage prepare_transfer(context& ctx, surface_depth format, id<MTLBuffer> guest,
		surface_byte_region bytes, const texture_region& region, bool gather) const;
	void upload_impl(context& ctx, surface_depth format, id<MTLBuffer> guest, surface_byte_region bytes,
		id<MTLTexture> texture, const texture_region& region, const transfer_storage* prepared,
		texture_plane plane = texture_plane::all) const;
	void download_impl(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
		id<MTLBuffer> guest, surface_byte_region bytes, const transfer_storage* prepared) const;

  public:
	explicit surface_depth_transfer(context& ctx);
	void scatter(context& ctx, surface_depth format, std::size_t width, std::size_t height,
				 id<MTLBuffer> guest, surface_byte_region bytes, const depth_transfer_plane& depth,
				 const depth_transfer_plane& stencil = {}) const;
	void gather(context& ctx, surface_depth format, std::size_t width, std::size_t height,
				const depth_transfer_plane& depth, const depth_transfer_plane& stencil,
				id<MTLBuffer> guest, surface_byte_region bytes) const;
	void upload(context& ctx, surface_depth format, id<MTLBuffer> guest, surface_byte_region bytes,
				id<MTLTexture> texture, const texture_region& region, texture_plane plane = texture_plane::all) const;
	void download(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
				  id<MTLBuffer> guest, surface_byte_region bytes) const;
	// Byte mask uses the same guest offset/pitch as guest. Pixels with no
	// selected depth byte retain raw native depth; stencil follows its own
	// byte mask. Any selected depth byte replaces that pixel's depth from
	// the assembled guest word.
	// Z24U patch writes choose a representable value that gathers to the
	// exact requested integer. Ordinary upload keeps its existing rounding.
	void patch(context& ctx, surface_depth format, id<MTLBuffer> guest, id<MTLBuffer> mask,
		surface_byte_region bytes, id<MTLTexture> texture, const texture_region& region) const;
};
} // namespace metal4
