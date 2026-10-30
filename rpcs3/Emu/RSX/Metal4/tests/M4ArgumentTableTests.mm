#include "M4Pipeline.h"
#include "M4Resources.h"

#include <array>
#include <iostream>
#include <optional>
#include <vector>

namespace
{
using namespace metal4;
void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}
constexpr auto source = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void snapshot(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
    texture2d<float> image [[texture(127)]], sampler filter [[sampler(15)]])
{
    output[0] = input ? input[0] : 0xdec0u;
    output[1] = is_null_texture(image) ? 0xbeefu : uint(image.sample(filter, float2(1.25, 0.5)).r * 4.0f);
}
)msl";

id<MTLComputePipelineState> make_pipeline(context& ctx)
{
	compiler compiler(ctx);
	auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(compiler.library(source), @"snapshot");
	return compiler.compute(descriptor);
}

id<MTL4ArgumentTable> encode(context& ctx, id<MTLComputePipelineState> pipeline,
	id<MTLBuffer> input, std::size_t input_offset, id<MTLBuffer> output, std::size_t output_offset,
	id<MTLTexture> texture, id<MTLSamplerState> sampler)
{
	bindings bindings;
	bindings.set_buffer(0, input, input_offset, input ? 16 : 0);
	bindings.set_buffer(1, output, output_offset);
	bindings.set_texture(127, texture);
	bindings.set_sampler(15, sampler);
	auto table = bindings.snapshot(ctx);
	ctx.keep_alive(pipeline);
	auto encoder = ctx.compute();
	[encoder setComputePipelineState:pipeline];
	[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	return table;
}

void slots_and_reuse()
{
	context ctx(2, 2);
	const auto pipeline = make_pipeline(ctx);
	const std::array<std::uint32_t, 4> words{11, 22, 33, 44};
	const auto input = make_upload(ctx, std::as_bytes(std::span(words)));
	const auto output = make_buffer(ctx, 256, buffer_storage::shared);
	std::fill_n(static_cast<std::uint32_t*>(output.contents), 64, 0u);
	auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float width:2 height:1 mipmapped:NO];
	td.storageMode = MTLStorageModeShared;
	td.usage = MTLTextureUsageShaderRead;
	const auto texture = [ctx.device() newTextureWithDescriptor:td];
	require(texture != nil, "Test texture allocation failed");
	const std::array<float, 2> texels{0.25f, 0.75f};
	[texture replaceRegion:MTLRegionMake2D(0, 0, 2, 1) mipmapLevel:0 withBytes:texels.data() bytesPerRow:8];
	auto sd = [MTLSamplerDescriptor new];
	sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
	sd.sAddressMode = MTLSamplerAddressModeRepeat;
	const auto repeat = make_sampler(ctx, sd);
	sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
	const auto clamp = make_sampler(ctx, sd);
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 2; } } guard{gate};
	std::array<id<MTL4ArgumentTable>, 4> tables;
	std::vector<submission> work;
	for (unsigned slot = 0; slot < 2; ++slot)
	{
		ctx.begin();
		ctx.wait_event(gate, slot + 1);
		for (unsigned i = 0; i < 2; ++i)
			tables[slot * 2 + i] = encode(ctx, pipeline, input, (slot * 2 + i) * 4, output,
				(slot * 2 + i) * 8, texture, i ? clamp : repeat);
		work.push_back(ctx.submit());
	}
	for (unsigned i = 0; i < 4; ++i)
		for (unsigned j = i + 1; j < 4; ++j)
			require(tables[i] != tables[j], "Pending encodings share a mutable argument table");
	require(!work[0].ready() && !work[1].ready(), "Command slots were not held pending");
	gate.signaledValue = 1;
	work[0].wait();
	require(!work[1].ready(), "Second slot did not remain independently gated");
	ctx.begin();
	for (unsigned i = 0; i < 2; ++i)
	{
		const auto reused = encode(ctx, pipeline, nil, 0, output, (4 + i) * 8, nil, nil);
		require(reused == tables[i], "Completed slot did not reuse its own argument table");
	}
	work.push_back(ctx.submit());
	const auto before = ctx.argument_statistics();
	require(before.created == 4 && before.reused == 2 && before.cached == 4 && before.capacity == 4,
		"Native argument table pool allocation/reuse count differs");
	gate.signaledValue = 2;
	work[1].wait();
	work[2].wait();
	const auto* result = static_cast<const std::uint32_t*>(output.contents);
	for (unsigned i = 0; i < 4; ++i)
		require(result[i * 2] == words[i] && result[i * 2 + 1] == (i % 2 ? 3u : 1u),
			"Pending argument snapshots lost buffer offsets, texture IDs or sampler changes");
	for (unsigned i = 4; i < 6; ++i)
		require(result[i * 2] == 0xdec0 && result[i * 2 + 1] == 0xbeef,
			"Recycled table retained old buffer or texture bindings");

	ctx.begin();
	const auto cancelled = ctx.recording_submission();
	std::vector<id<MTL4ArgumentTable>> overflow;
	for (unsigned i = 0; i < 10; ++i)
	{
		auto table = encode(ctx, pipeline, input, 8, output, 128 + i * 8, texture, clamp);
		for (auto old : overflow) require(old != table, "Overflow reused a table within one command buffer");
		overflow.push_back(table);
	}
	ctx.cancel();
	require(cancelled.cancelled() && ctx.argument_statistics().cached == 4 && ctx.argument_statistics().bypassed == 8,
		"Cancelled overflow escaped the bounded reusable table capacity");
	for (unsigned i = 32; i < 52; ++i) require(result[i] == 0, "Cancelled argument table work executed");
	ctx.begin();
	const auto reused_after_cancel = encode(ctx, pipeline, input, 8, output, 48, texture, clamp);
	require(reused_after_cancel == tables[2], "Cancelled slot lost its reusable table");
	ctx.submit().wait();
	require(result[12] == 33 && result[13] == 3, "Cancel/reuse corrupted the next native dispatch");

	const auto error = make_buffer(ctx, 4, buffer_storage::shared);
	*static_cast<std::uint32_t*>(error.contents) = 1;
	ctx.begin();
	(void)encode(ctx, pipeline, input, 12, output, 56, texture, repeat);
	ctx.require_gpu_success(error, 0, @"Intentional completed error for table lifetime");
	bool failed = false;
	try { ctx.submit().wait(); } catch (const std::runtime_error&) { failed = true; }
	require(failed, "GPU error fixture did not report completed validation failure");
	ctx.recover_completed_errors();
	ctx.begin();
	(void)encode(ctx, pipeline, nil, 0, output, 64, nil, nil);
	ctx.submit().wait();
	require(result[16] == 0xdec0 && result[17] == 0xbeef, "Completed-error recovery retained stale table bindings");
	std::cout << "Two gated slots: four distinct tables, completion reuse, changed offsets/samplers, nil slots, bounded overflow, cancellation and completed-error recovery passed\n";
}

void detached_and_disabled()
{
	std::optional<submission> pending;
	id<MTLBuffer> output;
	id<MTLSharedEvent> gate;
	struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 1; } } guard;
	{
		context ctx(1, 1);
		const auto pipeline = make_pipeline(ctx);
		output = make_buffer(ctx, 8, buffer_storage::shared);
		gate = [ctx.device() newSharedEvent];
		guard.gate = gate;
		ctx.begin();
		ctx.wait_event(gate, 1);
		(void)encode(ctx, pipeline, nil, 0, output, 0, nil, nil);
		pending = ctx.submit();
	}
	require(!pending->ready(), "Detached argument-table fixture did not remain pending");
	gate.signaledValue = 1;
	pending->wait();
	const auto* result = static_cast<const std::uint32_t*>(output.contents);
	require(result[0] == 0xdec0 && result[1] == 0xbeef, "Context destruction invalidated pending argument storage");
	context uncached(1, 0);
	for (unsigned i = 0; i < 3; ++i)
	{
		uncached.begin();
		(void)bindings{}.snapshot(uncached);
		uncached.cancel();
	}
	const auto stats = uncached.argument_statistics();
	require(stats.cached == 0 && stats.created == 3 && stats.reused == 0 && stats.bypassed == 3,
		"Disabled argument cache retained reusable tables");
	std::cout << "Pending argument tables survive context destruction; zero capacity bypass remains bounded\n";
}
}

int main()
{
	@autoreleasepool
	{
		try { slots_and_reuse(); detached_and_disabled(); }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
