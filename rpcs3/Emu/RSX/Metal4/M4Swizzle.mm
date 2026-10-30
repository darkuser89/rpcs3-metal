#include "M4Swizzle.h"
#include "M4SwizzleKernels.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>

namespace metal4
{
namespace
{
struct swizzle_parameters
{
	std::array<std::uint64_t, 3> origin, size;
	std::uint64_t swizzled_offset, linear_offset, row_pitch, image_pitch;
	std::uint64_t count, first;
	std::array<std::uint32_t, 3> log2;
	std::uint32_t block_bytes, swap_width, to_swizzled;
};
static_assert(sizeof(swizzle_parameters) == 120 && offsetof(swizzle_parameters, log2) == 96);

void range(context& ctx, id<MTLBuffer> buffer, std::size_t offset, std::size_t bytes)
{
	if (!buffer || buffer.device != ctx.device() || offset > buffer.length || bytes > buffer.length - offset)
		throw std::out_of_range("Native Morton conversion buffer range exceeds allocation");
}
}

swizzle_engine::swizzle_engine(context& ctx) : m_device(ctx.device())
{
	helper_catalog compiler(ctx);
	const auto library = compiler.library(swizzle_kernels, @"RPCS3 native rectangular Morton conversion");

	m_pipeline = compiler.compute(library, @"rpcs3_swizzle");
}

void swizzle_engine::convert(context& ctx, id<MTLBuffer> swizzled, std::size_t swizzled_offset,
	id<MTLBuffer> linear, std::size_t linear_offset, const swizzle_region& region, std::size_t block_bytes,
	std::size_t row_pitch, std::size_t image_pitch, bool to_swizzled, std::uint32_t swap_width) const
{
	if (ctx.device() != m_device || (swap_width != 1 && swap_width != 2 && swap_width != 4) || block_bytes % swap_width)
		throw std::invalid_argument("Native Morton conversion device or endian element width is invalid");
	const auto plan = plan_swizzle(region, block_bytes, row_pitch, image_pitch);
	range(ctx, swizzled, swizzled_offset, plan.storage_bytes);
	range(ctx, linear, linear_offset, plan.linear.allocation_size);
	if (swizzled == linear && swizzled_offset < linear_offset + plan.linear.allocation_size && linear_offset < swizzled_offset + plan.storage_bytes)
		throw std::invalid_argument("Native Morton source and destination ranges cannot overlap");
	bindings resources;
	resources.set_buffer(0, swizzled);
	resources.set_buffer(1, linear);
	ctx.keep_alive(m_pipeline);
	constexpr std::size_t max_count = 1u << 30;
	for (std::size_t first = 0; first < plan.block_count;)
	{
		const auto count = std::min(plan.block_count - first, max_count);
		const swizzle_parameters params{{region.origin[0], region.origin[1], region.origin[2]},
			{region.size[0], region.size[1], region.size[2]}, swizzled_offset, linear_offset,
			plan.linear.bytes_per_row, plan.linear.bytes_per_image, count, first, plan.log2,
			static_cast<std::uint32_t>(block_bytes), swap_width, to_swizzled ? 1u : 0u};
		const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
		resources.set_buffer(2, uniforms.buffer, uniforms.offset);
		const auto table = resources.snapshot(ctx);
		const auto encoder = ctx.compute();
		ctx.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
		[encoder setComputePipelineState:m_pipeline];
		[encoder setArgumentTable:table];
		[encoder dispatchThreads:MTLSizeMake(count, 1, 1)
			threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, m_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
		first += count;
	}
}
}
