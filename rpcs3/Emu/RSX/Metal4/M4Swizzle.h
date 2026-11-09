#pragma once

#include "M4SwizzleLayout.h"
#include "M4Pipeline.h"
#include "M4Resources.h"

#include <span>
#include <vector>

namespace metal4
{
class swizzle_engine final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;

	friend class guest_texture_resolver;
	[[nodiscard]] std::vector<id<MTLBuffer>> prepare_conversion(context& ctx,
		const swizzle_region& region, std::size_t block_bytes, bool to_swizzled) const;
	void convert_impl(context& ctx, id<MTLBuffer> swizzled, std::size_t swizzled_offset,
		id<MTLBuffer> linear, std::size_t linear_offset, const swizzle_region& region,
		std::size_t block_bytes, std::size_t row_pitch, std::size_t image_pitch,
		bool to_swizzled, std::uint32_t swap_width, std::span<const id<MTLBuffer>> prepared) const;

public:
	explicit swizzle_engine(context& ctx);
	// Both directions preserve bytes outside the active region, including all
	// row/image padding and other Morton cells. Offsets can be byte-unaligned.
	// swap_width reverses each 1/2/4-byte element within a pixel/block; use 1
	// when chaining with BE surface codecs. Source and destination must not
	// alias, including distinct buffers backed by overlapping heap allocations.
	// Invoke during recording outside a live render facade. One call describes
	// one mip/region; compressed textures use block coordinates supplied by the
	// texture-layout adapter, which is a separate renderer requirement.
	void convert(context& ctx, id<MTLBuffer> swizzled, std::size_t swizzled_offset,
		id<MTLBuffer> linear, std::size_t linear_offset, const swizzle_region& region,
		std::size_t block_bytes, std::size_t row_pitch = 0, std::size_t image_pitch = 0,
		bool to_swizzled = false, std::uint32_t swap_width = 1) const;
};
}
