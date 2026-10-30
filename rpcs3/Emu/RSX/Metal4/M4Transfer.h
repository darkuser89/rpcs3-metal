#pragma once

#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4NearestLayout.h"

namespace metal4
{

class transfer_engine final
{

	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_copy_pipeline;
	id<MTLComputePipelineState> m_fill_pipeline;
	id<MTLComputePipelineState> m_e4m12_pipeline;
	id<MTLComputePipelineState> m_nearest_pipeline;

  public:
	explicit transfer_engine(context& ctx);
	// Supports arbitrary byte offsets; swap_width 2/4 reverses each 16/32-bit
	// element. Source and destination ranges must not overlap.
	void copy(context& ctx, id<MTLBuffer> source, std::size_t source_offset,
			  id<MTLBuffer> destination, std::size_t destination_offset, std::size_t size,
			  std::uint32_t swap_width = 1) const;
	// Repeats a 32-bit word, matching vkCmdFillBuffer semantics.
	void fill_words(context& ctx, id<MTLBuffer> destination, std::size_t offset,
					std::size_t size, std::uint32_t value) const;
	// Bit-exact RSX E4M12<->float32 codec from VKCompute.h. expand=true reads
	// two bytes and writes four per element; false performs the reverse.
	// Endianness is explicit for each side. Odd element counts are supported.
	// Same-buffer overlap is rejected; distinct heap aliases are caller-owned.
	void convert_e4m12(context& ctx, id<MTLBuffer> source, std::size_t source_offset,
					   id<MTLBuffer> destination, std::size_t destination_offset, std::size_t elements,
					   bool expand, bool source_big_endian, bool destination_big_endian) const;
	// Center-based nearest mapping, copying raw texel bytes without numeric
	// conversion. Pitches/offsets are independent; each source axis may flip.
	// Snapshot overlapping sources before use; this kernel requires disjoint
	// source/destination buffer ranges, including caller-owned heap aliases.
	void nearest(context& ctx, id<MTLBuffer> source, std::size_t source_offset, nearest_image source_image,
				 id<MTLBuffer> destination, std::size_t destination_offset, nearest_image destination_image,
				 std::uint32_t texel_bytes, std::array<bool, 3> flip = {}) const;
};

} // namespace metal4
