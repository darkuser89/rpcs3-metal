#include "M4Transfer.h"
#include "M4TransferKernels.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <stdexcept>

namespace metal4
{

namespace
{

// Keep grid indices in uint range and chunk boundaries aligned to swap elements.
constexpr std::size_t max_dispatch_count = 1u << 30;

struct copy_parameters
{
	std::uint64_t source_offset;
	std::uint64_t destination_offset;
	std::uint64_t count;
	std::uint32_t swap_width;
	std::uint32_t padding = 0;
};

struct fill_parameters
{
	std::uint64_t word_offset;
	std::uint64_t count;
	std::uint32_t value;
	std::uint32_t padding = 0;
};

struct e4m12_parameters
{
	std::uint64_t source_offset;
	std::uint64_t destination_offset;
	std::uint64_t count;
	std::uint32_t expand;
	std::uint32_t source_big_endian;
	std::uint32_t destination_big_endian;
	std::uint32_t padding = 0;
};
struct nearest_parameters
{
	std::uint64_t source_offset, destination_offset;
	std::uint64_t source_row, source_image, destination_row, destination_image;
	std::uint64_t first, count;
	std::uint32_t source_width, source_height, source_depth;
	std::uint32_t destination_width, destination_height, destination_depth;
	std::uint32_t bytes, flip;
};

static_assert(sizeof(copy_parameters) == 32 && offsetof(copy_parameters, swap_width) == 24);
static_assert(sizeof(fill_parameters) == 24 && offsetof(fill_parameters, value) == 16);
static_assert(sizeof(e4m12_parameters) == 40 && offsetof(e4m12_parameters, expand) == 24);
static_assert(sizeof(nearest_parameters) == 96 && offsetof(nearest_parameters, source_width) == 64);

void validate_buffer(context& ctx, id<MTLBuffer> buffer, std::size_t offset, std::size_t size)
{
	if (!buffer || buffer.device != ctx.device() || offset > buffer.length || size > buffer.length - offset)
	{
		throw std::invalid_argument("Metal 4 transfer buffer or range is invalid");
	}
}

void dispatch(context& ctx, id<MTLComputePipelineState> pipeline, const bindings& resources, std::size_t count)
{
	ctx.keep_alive(pipeline);
	auto table = resources.snapshot(ctx);
	auto encoder = ctx.compute();
	ctx.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
	[encoder setComputePipelineState:pipeline];
	[encoder setArgumentTable:table];
	const auto group_size = std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
	[encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
}

} // namespace

transfer_engine::transfer_engine(context& ctx)
	: m_device(ctx.device())
{
	helper_catalog compiler(ctx);
	auto library = compiler.library(transfer_kernels, @"RPCS3 Metal 4 transfers");

	m_copy_pipeline = compiler.compute(library, @"rpcs3_copy_bytes");
	m_fill_pipeline = compiler.compute(library, @"rpcs3_fill_words");
	m_e4m12_pipeline = compiler.compute(library, @"rpcs3_convert_e4m12");
	m_nearest_pipeline = compiler.compute(library, @"rpcs3_nearest_bytes");
}

void transfer_engine::copy(context& ctx, id<MTLBuffer> source, std::size_t source_offset,
						   id<MTLBuffer> destination, std::size_t destination_offset, std::size_t size, std::uint32_t swap_width) const
{
	if (ctx.device() != m_device)
	{
		throw std::invalid_argument("Metal 4 transfer engine belongs to another device");
	}
	validate_buffer(ctx, source, source_offset, size);
	validate_buffer(ctx, destination, destination_offset, size);
	if ((swap_width != 1 && swap_width != 2 && swap_width != 4) || size % swap_width)
	{
		throw std::invalid_argument("Metal 4 transfer requires complete 1-, 2- or 4-byte elements");
	}
	if (size == 0)
		return;
	if (source == destination && source_offset < destination_offset + size && destination_offset < source_offset + size)
	{
		throw std::invalid_argument("Metal 4 transfer ranges cannot overlap");
	}
	// Distinct buffers may still alias a heap allocation. The transfer owner must
	// not submit overlapping aliases, just as for the Vulkan transfer operation.
	if (swap_width == 1 && !((source_offset | destination_offset | size) & 3))
	{
		copy_buffer(ctx, source, source_offset, destination, destination_offset, size);
		return;
	}
	for (std::size_t position = 0; position < size;)
	{
		const auto count = std::min(size - position, max_dispatch_count);
		const copy_parameters params{source_offset + position, destination_offset + position, count, swap_width};
		auto parameters = ctx.upload(std::as_bytes(std::span(&params, 1)));
		bindings resources;
		resources.set_buffer(0, source);
		resources.set_buffer(1, destination);
		resources.set_buffer(2, parameters.buffer, parameters.offset);
		dispatch(ctx, m_copy_pipeline, resources, count);
		position += count;
	}
}

void transfer_engine::fill_words(context& ctx, id<MTLBuffer> destination, std::size_t offset,
								 std::size_t size, std::uint32_t value) const
{
	if (ctx.device() != m_device)
	{
		throw std::invalid_argument("Metal 4 transfer engine belongs to another device");
	}
	validate_buffer(ctx, destination, offset, size);
	if ((offset | size) & 3)
	{
		throw std::invalid_argument("Metal 4 word fills require four-byte alignment");
	}
	if (size == 0)
		return;
	if ((value & 0xff) * 0x01010101u == value)
	{
		fill_buffer(ctx, destination, offset, size, static_cast<std::uint8_t>(value));
		return;
	}
	const auto word_count = size / 4;
	for (std::size_t position = 0; position < word_count;)
	{
		const auto count = std::min(word_count - position, max_dispatch_count);
		const fill_parameters params{offset / 4 + position, count, value};
		auto parameters = ctx.upload(std::as_bytes(std::span(&params, 1)));
		bindings resources;
		resources.set_buffer(0, destination);
		resources.set_buffer(1, parameters.buffer, parameters.offset);
		dispatch(ctx, m_fill_pipeline, resources, count);
		position += count;
	}
}

void transfer_engine::convert_e4m12(context& ctx, id<MTLBuffer> source, std::size_t source_offset,
									id<MTLBuffer> destination, std::size_t destination_offset, std::size_t elements,
									bool expand, bool source_big_endian, bool destination_big_endian) const
{
	if (ctx.device() != m_device)
	{
		throw std::invalid_argument("Metal 4 transfer engine belongs to another device");
	}
	validate_buffer(ctx, source, source_offset, 0);
	validate_buffer(ctx, destination, destination_offset, 0);
	const std::size_t source_width = expand ? 2 : 4;
	const std::size_t destination_width = expand ? 4 : 2;
	if (elements > (source.length - source_offset) / source_width ||
		elements > (destination.length - destination_offset) / destination_width)
	{
		throw std::out_of_range("Metal 4 depth conversion exceeds the buffer allocation");
	}
	if (!elements)
		return;
	if (source == destination && source_offset < destination_offset + elements * destination_width &&
		destination_offset < source_offset + elements * source_width)
	{
		throw std::invalid_argument("Metal 4 depth conversion ranges cannot overlap");
	}
	for (std::size_t position = 0; position < elements;)
	{
		const auto count = std::min(elements - position, max_dispatch_count);
		const e4m12_parameters params{source_offset + position * source_width,
									  destination_offset + position * destination_width, count,
									  static_cast<std::uint32_t>(expand), static_cast<std::uint32_t>(source_big_endian),
									  static_cast<std::uint32_t>(destination_big_endian)};
		auto parameters = ctx.upload(std::as_bytes(std::span(&params, 1)));
		bindings resources;
		resources.set_buffer(0, source);
		resources.set_buffer(1, destination);
		resources.set_buffer(2, parameters.buffer, parameters.offset);
		dispatch(ctx, m_e4m12_pipeline, resources, count);
		position += count;
	}
}

void transfer_engine::nearest(context& ctx, id<MTLBuffer> source, std::size_t source_offset, nearest_image source_image,
							  id<MTLBuffer> destination, std::size_t destination_offset, nearest_image destination_image,
							  std::uint32_t bytes, std::array<bool, 3> flip) const
{
	if (ctx.device() != m_device)
		throw std::invalid_argument("Native nearest engine belongs to another device");
	const auto layout = plan_nearest(source_image, destination_image, bytes);
	validate_buffer(ctx, source, source_offset, layout.source.allocation_size);
	validate_buffer(ctx, destination, destination_offset, layout.destination.allocation_size);
	if (source == destination && source_offset < destination_offset + layout.destination.allocation_size &&
		destination_offset < source_offset + layout.source.allocation_size)
		throw std::invalid_argument("Native nearest blit buffer ranges cannot overlap");
	for (std::size_t first = 0; first < layout.texels;)
	{
		const auto count = std::min(layout.texels - first, max_dispatch_count);
		const nearest_parameters params{source_offset, destination_offset,
										layout.source.bytes_per_row, layout.source.bytes_per_image, layout.destination.bytes_per_row, layout.destination.bytes_per_image,
										first, count, source_image.size[0], source_image.size[1], source_image.size[2],
										destination_image.size[0], destination_image.size[1], destination_image.size[2], bytes,
										static_cast<std::uint32_t>(flip[0]) | (static_cast<std::uint32_t>(flip[1]) << 1) | (static_cast<std::uint32_t>(flip[2]) << 2)};
		const auto parameters = ctx.upload(std::as_bytes(std::span(&params, 1)));
		bindings resources;
		resources.set_buffer(0, source);
		resources.set_buffer(1, destination);
		resources.set_buffer(2, parameters.buffer, parameters.offset);
		dispatch(ctx, m_nearest_pipeline, resources, count);
		first += count;
	}
}

} // namespace metal4
