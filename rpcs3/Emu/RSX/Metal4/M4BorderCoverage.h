#pragma once
#include "M4HeapBudget.h"
#include "M4SamplerParameters.h"
#include <functional>
#include <list>
#include <vector>

namespace metal4
{
// A draw admits helper allocations before encoding, then records their
// initialization before any shader samples them.
class border_texture_preparation final
{
	struct image;
	std::vector<std::shared_ptr<image>> m_images;
	friend class border_coverage_cache;

  public:
	void encode(context& ctx);
};
// Immutable helpers match every source mip/face/extent. R32F zero images
// measure color/depth-read border weights; constant native depth images keep
// shadow comparisons and their filtering on the GPU. No extra sampler slot.
// Standalone consumers may request a completed helper synchronously.
class border_coverage_cache final
{
	context& m_context;
	std::shared_ptr<heap_budget> m_budget;
	using helper = border_texture_preparation::image;
	std::list<std::shared_ptr<helper>> m_images;
	std::array<id<MTLComputePipelineState>, 3> m_clear{}, m_copy{};
	struct color_entry
	{
		id<MTLTexture> source;
		std::array<std::uint32_t, 4> channels;
		std::shared_ptr<helper> image;
	};
	std::list<color_entry> m_colors;
	struct shadow_entry
	{
		float value;
		std::shared_ptr<helper> image;
	};
	std::list<shadow_entry> m_shadows;
	void initialize(unsigned kind);
	[[nodiscard]] std::shared_ptr<helper> build(id<MTLTexture> source, MTLPixelFormat format, id<MTLTexture> input = nil);
	[[nodiscard]] std::shared_ptr<helper> build_shadow(id<MTLTexture> source, MTLPixelFormat format, float value);
	[[nodiscard]] id<MTLTexture> use(const std::shared_ptr<helper>& image, border_texture_preparation* preparation);
	std::size_t m_capacity;

  public:
	border_coverage_cache(context& ctx, std::shared_ptr<heap_budget> budget, std::size_t capacity);
	[[nodiscard]] id<MTLTexture> get(id<MTLTexture> source, border_texture_preparation* preparation = nullptr);
	// Source must remain immutable while retained by this cache.
	[[nodiscard]] id<MTLTexture> color_view(id<MTLTexture> source, std::array<std::uint32_t, 4> channels, border_texture_preparation* preparation = nullptr);
	// Constant native depth matching source shape/mips/comparison precision.
	[[nodiscard]] id<MTLTexture> shadow_view(id<MTLTexture> source, float value, border_texture_preparation* preparation = nullptr);
};
void validate_border_coverage(context& ctx, id<MTLTexture> source, id<MTLTexture> coverage, const sampler_parameters& parameters);
} // namespace metal4
