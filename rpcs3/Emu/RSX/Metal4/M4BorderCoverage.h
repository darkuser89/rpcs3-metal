#pragma once
#include "M4HeapBudget.h"
#include "M4SamplerParameters.h"
#include <list>

namespace metal4
{
// Immutable helpers match every source mip/face/extent. R32F zero images
// measure color/depth-read border weights; constant native depth images keep
// shadow comparisons and their filtering on the GPU. No extra sampler slot.
// Cold creation owns a submission and waits: outside a VM/method lock/draw.
class border_coverage_cache final
{
	context& m_context;
	std::shared_ptr<heap_budget> m_budget;
	std::list<id<MTLTexture>> m_images;
	std::array<id<MTLComputePipelineState>, 3> m_clear{}, m_copy{};
	struct color_entry
	{
		id<MTLTexture> source;
		std::array<std::uint32_t, 4> channels;
		id<MTLTexture> image;
	};
	std::list<color_entry> m_colors;
	struct shadow_entry
	{
		float value;
		id<MTLTexture> image;
	};
	std::list<shadow_entry> m_shadows;
	void initialize(unsigned kind);
	[[nodiscard]] id<MTLTexture> build(id<MTLTexture> source, MTLPixelFormat format, id<MTLTexture> input = nil);
	std::size_t m_capacity;

  public:
	border_coverage_cache(context& ctx, std::shared_ptr<heap_budget> budget, std::size_t capacity);
	[[nodiscard]] id<MTLTexture> get(id<MTLTexture> source);
	// Source must remain immutable while retained by this cache.
	[[nodiscard]] id<MTLTexture> color_view(id<MTLTexture> source, std::array<std::uint32_t, 4> channels);
	// Constant native depth matching source shape/mips/comparison precision.
	[[nodiscard]] id<MTLTexture> shadow_view(id<MTLTexture> source, float value);
};
void validate_border_coverage(context& ctx, id<MTLTexture> source, id<MTLTexture> coverage, const sampler_parameters& parameters);
} // namespace metal4
