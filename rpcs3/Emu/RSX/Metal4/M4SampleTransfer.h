#pragma once

#include "M4SampleLayout.h"
#include "M4SampleShader.h"
#include "M4Texture.h"
#include "M4Draw.h"
#include "M4Pipeline.h"

#include <vector>

namespace metal4
{
// Spatial sample expansion/import, not averaging. One call operates on complete
// single-slice 2D level-zero textures, using a 2x1/2x2 RSX expanded raster. Surface codecs
// can then upload/download that ordinary expanded texture as guest bytes.
class sample_transfer final
{
	helper_catalog m_compiler;
	id<MTLDevice> m_device;
	id<MTLDepthStencilState> m_depth_state, m_depth_stencil_state, m_stencil_state;
	std::array<id<MTLLibrary>, 8> m_libraries{};
	[[nodiscard]] id<MTLRenderPipelineState> pipeline(MTLPixelFormat format, std::uint32_t samples,
													  bool importing, sample_aspect aspect);
	void run(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, bool importing, texture_plane plane);

  public:
	explicit sample_transfer(context& ctx);
	// Requires ShaderRead on source, RenderTarget on destination and distinct
	// underlying texture allocations and identity channel views. Native formats match.
	// Invoke outside a live render facade; the transfer owns its render pass.
	// Destination pixels/samples are all overwritten. Completion retains every
	// attachment, shader library/pipeline, argument table and stencil view.
	// Depth outside finite [0,1] cannot be preserved by this native raster path:
	// GPU validation makes submission.wait() fail rather than silently clamping.
	// Such validated work must complete in an offscreen submission before use
	// in a submission that presents a drawable. Exact raw-depth transfer remains
	// an open backend requirement, as does float-color NaN payload parity.
	// Selected depth/stencil transfers attach and read only that plane. Stencil
	// transfers never inspect, validate or rewrite an unrelated depth plane.
	void expand(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, texture_plane plane = texture_plane::all);
	void import(context& ctx, id<MTLTexture> expanded, id<MTLTexture> multisampled, texture_plane plane = texture_plane::all);
};
} // namespace metal4
