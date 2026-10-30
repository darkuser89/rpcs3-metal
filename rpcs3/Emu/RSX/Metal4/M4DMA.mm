#include "M4DMA.h"
#include "M4DMAKernels.h"
namespace metal4
{
namespace
{
struct dma_parameters
{
	std::int64_t source_first, destination_first, source_pitch, destination_pitch;
	std::uint32_t source_stride, destination_stride, columns, rows, bytes, groups, source_groups, packed;
	dma_active_group active[4];
	dma_active_group source_active[4];
};
static_assert(sizeof(dma_active_group) == 24 && sizeof(dma_parameters) == 256);
struct dma_capture
{
	std::vector<std::byte> bytes;
	std::vector<guest_surface_write> parts;
};
dma_capture capture_dma(guest_surface_memory& memory, std::span<const dma_storage_part> parts, std::size_t bytes, bool source)
{
	dma_capture result{std::vector<std::byte>(source ? bytes : 0), {}};
	for (const auto& part : parts)
	{
		auto coverage = memory.watch_surface(part.coverage);
		std::uint64_t revision;
		if (source)
		{
			const auto snapshot = memory.read_surface_packed(coverage);
			if (snapshot.bytes.size() != guest_surface_active_bytes(coverage) || part.packed_offset > bytes || snapshot.bytes.size() > bytes - part.packed_offset)
				throw std::runtime_error("Native packed DMA snapshot size differs");
			std::copy(snapshot.bytes.begin(), snapshot.bytes.end(), result.bytes.begin() + part.packed_offset);
			revision = snapshot.revision;
		}
		else revision = memory.surface_revision(coverage);
		result.parts.push_back({std::move(coverage), revision, {}, true, part.logical_offset});
	}
	return result;
}
} // namespace
dma_compute::dma_compute(context& ctx, std::size_t limit) : m_device(ctx.device()), m_limit(limit)
{
	@autoreleasepool
	{
		if (!limit)
			throw std::invalid_argument("Native DMA requires a finite snapshot budget");
		helper_catalog compile(ctx);
		const auto library = compile.library(dma_kernels, @"RPCS3 native raw/strided DMA");

		m_pipeline = compile.compute(library, @"rpcs3_dma_bytes");
	}
}
void dma_compute::encode(context& ctx, const dma_plan& plan, id<MTLBuffer> source, id<MTLBuffer> destination, bool packed) const
{
	validate_dma(plan, m_limit, packed);
	const auto active = plan_dma_active(plan.destination, plan.columns, plan.rows);
	const auto source_active = packed ? plan_dma_active(plan.source, plan.columns, plan.rows) : dma_active_layout{};
	if (ctx.device() != m_device || !source || !destination || source.device != m_device || destination.device != m_device || source == destination ||
		source.length < (packed ? source_active.bytes : plan.source.range.size) || destination.length < (packed ? active.bytes : plan.destination.range.size))
		throw std::invalid_argument("Native DMA requires distinct adequately sized buffers on its own device");
	// Identical relative byte geometry makes the packed permutation an identity,
	// including overlapping rows. Metal's aligned native blit needs no shader.
	if (packed && plan.source.pitch == plan.destination.pitch && plan.source.stride == plan.destination.stride && !(active.bytes & 3))
	{
		copy_buffer(ctx, source, 0, destination, 0, active.bytes);
		return;
	}
	dma_parameters params{plan.source.first, plan.destination.first, plan.source.pitch, plan.destination.pitch,
								plan.source.stride, plan.destination.stride, plan.columns, plan.rows, active.bytes, active.group_count,
								source_active.group_count, std::uint32_t(packed), {}, {}};
	std::copy(active.groups.begin(), active.groups.end(), params.active);
	std::copy(source_active.groups.begin(), source_active.groups.end(), params.source_active);
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
	bindings resources;
	resources.set_buffer(0, source);
	resources.set_buffer(1, destination);
	resources.set_buffer(2, uniforms.buffer, uniforms.offset);
	const auto table = resources.snapshot(ctx);
	ctx.keep_alive(m_pipeline);
	const auto encoder = ctx.compute();
	ctx.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
	[encoder setComputePipelineState:m_pipeline];
	[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(active.bytes, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, m_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
}
struct dma_transfer::pending
{
	submission work;
	std::vector<guest_surface_write> parts;
	std::vector<dma_storage_part> segments;
	id<MTLBuffer> output;
	std::function<void()> validate;
};
dma_transfer::dma_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures, std::shared_ptr<guest_surface_memory> memory,
						   cpu_capture_dispatch capture, std::size_t limit)
	: m_context(ctx), m_surfaces(surfaces), m_textures(textures), m_memory(std::move(memory)), m_capture(std::move(capture)), m_compute(ctx, limit), m_limit(limit)
{
	if (!m_memory || !m_capture)
		throw std::invalid_argument("Native DMA requires coherent memory and CPU capture");
	m_surfaces.validate_texture_adapter(ctx, *m_memory);
	m_textures.validate_surface_adapter(ctx, m_surfaces);
}
dma_transfer::~dma_transfer() = default;
void dma_transfer::execute(dma_plan plan, std::chrono::milliseconds timeout, const std::function<void(context&)>& dependencies, std::function<void()> validate_metadata)
{
	@autoreleasepool
	{
		if (timeout.count() < 0)
			throw std::invalid_argument("Native DMA timeout must be nonnegative");
		validate_dma(plan, m_limit, true);
		const auto source_parts = project_dma_active(plan.source, plan.columns, plan.rows);
		auto target_parts = project_dma_active(plan.destination, plan.columns, plan.rows);
		const auto source_bytes = plan_dma_active(plan.source, plan.columns, plan.rows).bytes;
		const auto target_bytes = plan_dma_active(plan.destination, plan.columns, plan.rows).bytes;
		synchronize(timeout);
		if (validate_metadata)
			checked_cpu_capture(m_capture, validate_metadata);
		for (const auto* parts : {&source_parts, &std::as_const(target_parts)})
			for (const auto& part : *parts) m_surfaces.prepare_cpu_read_active(part.coverage);
		dma_capture input, output;
		checked_cpu_capture(m_capture, [&]
							{
			if (validate_metadata) validate_metadata();
			input = capture_dma(*m_memory, source_parts, source_bytes, true); output = capture_dma(*m_memory, target_parts, target_bytes, false); });
		const auto source = make_upload(m_context, input.bytes), target = make_buffer(m_context, target_bytes, buffer_storage::shared);
		m_context.begin();
		std::unique_ptr<pending> transaction;
		try
		{
			transaction = std::make_unique<pending>(pending{m_context.recording_submission(), std::move(output.parts), std::move(target_parts), target, std::move(validate_metadata)});
			if (dependencies)
				dependencies(m_context);
			m_compute.encode(m_context, plan, source, target, true);
			transaction->work = m_context.submit();
		}
		catch (...)
		{
			m_context.cancel();
			throw;
		}
		m_pending = std::move(transaction);
		synchronize(timeout);
	}
}
void dma_transfer::synchronize(std::chrono::milliseconds timeout)
{
	@autoreleasepool
	{
		if (timeout.count() < 0)
			throw std::invalid_argument("Native DMA timeout must be nonnegative");
		if (!m_pending)
			return;
		try
		{
			m_pending->work.wait(timeout);
		}
		catch (...)
		{
			if (m_pending->work.ready())
				m_pending.reset();
			throw;
		}
		auto completed = std::move(m_pending);
		if (completed->validate)
			checked_cpu_capture(m_capture, completed->validate);
		std::vector<guest_surface_layout> coverage;
		for (const auto& part : completed->segments)
			coverage.push_back(part.coverage);
		// Resolve Metal properties before entering the CPU publication epoch.
		for (std::size_t i = 0; i < completed->parts.size(); ++i)
			completed->parts[i].bytes = map_buffer(completed->output, completed->segments[i].packed_offset, guest_surface_active_bytes(completed->segments[i].coverage));
		m_surfaces.cpu_write_active(coverage, [&]
		{
			if (completed->validate) completed->validate();
			if (!m_memory->compare_write_many(completed->parts)) throw texture_revision_conflict();
		}, m_capture);
		for (const auto& part : coverage) m_textures.invalidate_active(part);
	}
}
} // namespace metal4
