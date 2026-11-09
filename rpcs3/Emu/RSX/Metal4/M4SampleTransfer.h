#pragma once

#include "M4SampleLayout.h"
#include "M4SampleShader.h"
#include "M4Texture.h"
#include "M4Draw.h"
#include "M4Pipeline.h"

#include <vector>
#include <optional>

namespace metal4
{
struct sample_transfer_statistics
{
	// Successful encodings,not completed work or elapsed time.
	std::uint64_t full_expansions=0,cropped_expansions=0,full_texels=0,cropped_texels=0,imports=0;
	std::uint64_t imported_texels=0,cropped_imports=0,cropped_imported_texels=0;
};
struct sample_raster_projection
{
	std::array<std::size_t, 2> source_origin, source_extent;
	// Offset and extent refer to the full destination of the guest copy,
	// even when only a window inside that copy will be written.
	std::array<std::size_t, 2> copy_offset, copy_extent;
	std::array<bool, 2> flip{};
};
// Spatial sample expansion/import, not averaging. One call operates on complete
// single-slice 2D level-zero textures, using a 2x1/2x2 RSX expanded raster. Surface codecs
// can then upload/download that ordinary expanded texture as guest bytes.
class sample_transfer final
{
	struct parameters
	{
		std::uint32_t samples_x, samples_y, origin_x, origin_y, width, height, target_x, target_y;
		std::uint32_t source_width, source_height, copy_width, copy_height, copy_x, copy_y, flip, row_elements;
	};
	static_assert(sizeof(parameters) == 64);
	helper_catalog m_compiler;
	id<MTLDevice> m_device;
	id<MTLDepthStencilState> m_depth_state, m_depth_stencil_state, m_stencil_state;
	std::array<id<MTLLibrary>, 24> m_libraries{};
	id<MTLLibrary> m_float_expand_library = nil;
	std::array<id<MTLComputePipelineState>, 8> m_float_expanders{};
	sample_transfer_statistics m_statistics;
	[[nodiscard]] id<MTLRenderPipelineState> pipeline(MTLPixelFormat format, std::uint32_t samples,
													  bool importing, sample_aspect aspect, bool single_sample_source);
	void run(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, bool importing, texture_plane plane,
		std::optional<std::array<std::size_t,2>> crop={}, const texture_region* destination = nullptr, bool preserve = false,
		const sample_raster_projection* projection = nullptr);
	void raster_transfer(context& ctx, id<MTLTexture> source, id<MTLTexture> target, unsigned samples,
		bool importing, sample_aspect type, const parameters& params, bool cropped, bool preserve);
	void expand_float(context& ctx, id<MTLTexture> source, id<MTLTexture> target, std::size_t format,
		parameters params);

  public:
	explicit sample_transfer(context& ctx);
	// Requires ShaderRead on source, RenderTarget on destination and distinct
	// underlying texture allocations and identity channel views. Native formats match.
	// Invoke outside a live render facade; the transfer owns its render pass.
	// Destination pixels/samples are all overwritten. Completion retains every
	// attachment, shader library/pipeline, argument table and stencil view.
	// Float expansion copies the words returned by the typed multisample read.
	// Float-color import keeps half/float storage types through the export,
	// including NaN payloads, signed zero and subnormals.
	// Depth raster transfers still reject values outside finite [0,1] at submission.wait().
	// Such validated work must complete in an offscreen submission before use
	// in a submission that presents a drawable. Raster import does not guarantee
	// arbitrary depth words in a native MSAA target.
	// Selected depth/stencil transfers attach and read only that plane. Stencil
	// transfers never inspect, validate or rewrite an unrelated depth plane.
	void expand(context& ctx, id<MTLTexture> multisampled, id<MTLTexture> expanded, texture_plane plane = texture_plane::all);
	// Expanded sample-raster origin;destination dimensions define the crop.
	// Odd origins/extents may split a native pixel's samples. Only selected
	// source samples/planes are read;the entire destination plane is written.
	void expand_region(context& ctx,id<MTLTexture> multisampled,id<MTLTexture> expanded,
		std::array<std::size_t,2> origin,texture_plane plane=texture_plane::all);
	// Copy a crop into a subrectangle of an expanded image. Preserve samples
	// written by earlier pieces when composing a complete snapshot.
	void project_into(context& ctx, id<MTLTexture> source, id<MTLTexture> expanded,
		const sample_raster_projection& projection, const texture_region& destination, bool preserve,
		texture_plane plane = texture_plane::all);
	void import(context& ctx, id<MTLTexture> expanded, id<MTLTexture> multisampled, texture_plane plane = texture_plane::all);
	// Update only this expanded-raster rectangle in an existing native target.
	// Loads selected target planes and discards untouched boundary samples;
	// samples outside the rectangle and unrelated planes retain their contents.
	void import_region(context& ctx,id<MTLTexture> expanded,id<MTLTexture> multisampled,
		std::array<std::size_t,2> origin,texture_plane plane=texture_plane::all);
	[[nodiscard]] sample_transfer_statistics statistics() const noexcept {return m_statistics;}
};
} // namespace metal4
