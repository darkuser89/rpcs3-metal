#pragma once

#include "M4ClearPlan.h"
#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"

#include <vector>
#include <optional>

namespace metal4
{
struct clear_attachment
{
	id<MTLTexture> texture = nil;
	std::size_t level = 0, slice = 0;
};
struct clear_color_attachment
{
	clear_attachment image;
	surface_color format = surface_color::argb8;
};
struct clear_request
{
	std::array<clear_color_attachment, 4> colors;
	clear_attachment depth;
	surface_depth depth_format = surface_depth::z16_uint;
	std::uint32_t mask = 0;
	std::uint32_t color_value = 0;
	std::uint32_t zstencil_value = 0;
	std::uint32_t stencil_write_mask = 255;
	clear_rectangle rectangle;
};
struct clear_result
{
	std::uint32_t color_targets = 0;
	bool depth = false, stencil = false;
};

// Clears preserve channels/aspects outside the mask and pixels outside clipped
// scissor. Targets must be initialized when preservation is needed; surface
// cache initialization/dirty/VM rules remain the RSX renderer's contract.
class clear_engine final
{
	id<MTLDevice> m_device;
	helper_catalog m_compiler;
	id<MTLLibrary> m_library;
	// Exactly two depth-write choices times 256 effective stencil masks.
	std::array<id<MTLDepthStencilState>, 512> m_depth_states{};
	std::uint64_t m_depth_created = 0, m_depth_reused = 0;
	[[nodiscard]] id<MTLRenderPipelineState> pipeline(const std::array<MTLPixelFormat, 4>& formats,
													  const std::array<std::uint32_t, 4>& masks, std::uint32_t samples, bool depth);
	[[nodiscard]] id<MTLDepthStencilState> depth_state(bool depth, std::uint32_t stencil_mask);

  public:
	explicit clear_engine(context& ctx);
	// Exact uniform admission for the cache's single color/depth target. Full
	// native load clears and format/aspect NOPs need no upload reservation.
	[[nodiscard]] static std::optional<upload_requirement> upload_for(surface_format format,
		std::uint32_t mask, std::uint32_t stencil_mask, bool full_rectangle);
	[[nodiscard]] std::uint64_t depth_states_created() const noexcept { return m_depth_created; }
	[[nodiscard]] std::uint64_t depth_states_reused() const noexcept { return m_depth_reused; }
	// Native Metal4 load clears for full write-through; native fragment and
	// depth/stencil states for partial masks/rectangles. 1/2/4 samples, 2D or
	// selected 2DArray slice/mip; invoke outside a live render facade.
	// Result identifies the aspects actually changed, including format NOPs.
	[[nodiscard]] clear_result run(context& ctx, const clear_request& request);
};
} // namespace metal4
