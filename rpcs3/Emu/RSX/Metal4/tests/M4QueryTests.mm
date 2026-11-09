#include "M4Pipeline.h"
#include "M4Query.h"
#include "M4RSXReports.h"
#include "M4Texture.h"

#include <cstring>
#include <iostream>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void rejects(F&& fn)
{
	try
	{
		fn();
	}
	catch (const std::exception&)
	{
		return;
	}
	throw std::runtime_error("Invalid native query accepted");
}
constexpr std::string_view shader = R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 query_vertex(uint index [[vertex_id]])
{
    float2 uv = float2((index << 1) & 2, index & 2);
    return float4(uv * 2 - 1, 0.5f, 1);
}
fragment float4 query_fragment() { return float4(1); }
fragment float4 query_discard() { discard_fragment(); return float4(1); }
)msl";
struct fixture
{
	metal4::context& ctx;
	id<MTLTexture> color, depth;
	id<MTLRenderPipelineState> normal, kill;
	id<MTLDepthStencilState> pass_depth, reject_depth;
	explicit fixture(metal4::context& context) : ctx(context)
	{
		using namespace metal4;
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		color = make_texture(ctx, descriptor);
		descriptor.pixelFormat = MTLPixelFormatDepth32Float;
		depth = make_texture(ctx, descriptor);
		compiler compiler(ctx);
		const auto library = compiler.library(shader);
		MTL4RenderPipelineDescriptor* pipeline = [MTL4RenderPipelineDescriptor new];
		pipeline.vertexFunctionDescriptor = library_function(library, @"query_vertex");
		pipeline.fragmentFunctionDescriptor = library_function(library, @"query_fragment");
		pipeline.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		pipeline.rasterSampleCount = 1;
		pipeline.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		normal = compiler.render(pipeline);
		pipeline.fragmentFunctionDescriptor = library_function(library, @"query_discard");
		kill = compiler.render(pipeline);
		MTLDepthStencilDescriptor* test = [MTLDepthStencilDescriptor new];
		test.depthCompareFunction = MTLCompareFunctionAlways;
		pass_depth = [ctx.device() newDepthStencilStateWithDescriptor:test];
		test.depthCompareFunction = MTLCompareFunctionLess;
		reject_depth = [ctx.device() newDepthStencilStateWithDescriptor:test];
	}
	void encode(metal4::query_pool& pool, bool reject = false, bool discard = false,
		const std::function<void(id<MTLBuffer>, std::size_t)>& before = {},
		const metal4::conditional_draw* conditional = nullptr)
	{
		using namespace metal4;
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = color;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 1, 0, 1);
		pass.depthAttachment.texture = depth;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.depthAttachment.clearDepth = 0;
		const auto visibility = pool.attach(ctx, pass);
		if (before && visibility) before(pass.visibilityResultBuffer, visibility->offset);
		render_commands commands(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, 4, 4, 0, 1};
		raster.scissor = {0, 0, 4, 4};
		commands.configure(discard ? kill : normal, reject ? reject_depth : pass_depth, raster, {}, {});
		if (visibility)
			commands.visibility(visibility->mode, visibility->offset);
		if (conditional) commands.draw(*conditional);
		else commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
		commands.end();
	}
};
void intervals_and_boundaries()
{
	using namespace metal4;
	context ctx(3); fixture draws(ctx);
	query_pool pool(ctx, 1, 1);
	for (const bool precise : {true, false})
	{
		const auto start = pool.statistics();
		pool.begin(0, precise); ctx.begin();
		for (unsigned i = 0; i < 512; ++i) draws.encode(pool, i % 3 == 0, i % 3 == 1);
		require(pool.statistics().live_segments == 1 && pool.statistics().leased_counters == 1 && !pool.ready(0), "One batch did not retain one pending interval");
		ctx.submit().wait();
		// 170 accepted draws, each covering sixteen pixels.
		require(pool.consume(0) == (precise ? 170 * 16 : 1), "Batched interval mixed accepted/rejected draws incorrectly");
		require(pool.statistics().acquired - start.acquired == 1 && pool.statistics().initializations - start.initializations == 1 &&
			pool.statistics().reused_batch - start.reused_batch == 511, "One batch repeated counter allocation/zeroing");
		for (unsigned i = 0; i < 80; ++i)
		{
			ctx.begin(); draws.encode(pool); ctx.submit().wait();
			if ((i + 1) % 20 == 0) require(pool.consume(0) == (precise ? 320 : 1), "Active partial consume lost a query boundary");
		}
		pool.end(0);
		require(pool.consume(0) == 0 && !pool.statistics().leased_counters && !pool.statistics().live_segments, "Consumed intervals retained query slots");
	}
	std::cout << "512 encoders/one interval/one zero for count and boolean;80 completed draws with four active-consume boundaries passed\n";
	// Seed a real GPU counter below the uint32 boundary, then accumulate two
	// native draws. This verifies the public RSX modulo without billions of draws.
	pool.begin(0, true); ctx.begin();
	const std::uint64_t initial = 0xfffffff0ull;
	const auto seed = make_upload(ctx, std::as_bytes(std::span{&initial, 1}));
	draws.encode(pool, false, false, [&](id<MTLBuffer> buffer, std::size_t offset) { copy_buffer(ctx, seed, 0, buffer, offset, 8); });
	draws.encode(pool); ctx.submit().wait(); pool.end(0);
	require(pool.consume(0) == 16, "Native accumulated count lost uint32 wrap");
	// Cancellation must retain a confirmed prefix but never read a fresh
	// counter whose initializing command was cancelled before execution.
	for (unsigned prefix : {0u, 1u, 3u})
	{
		pool.begin(0, true);
		for (unsigned i = 0; i < prefix; ++i) { ctx.begin(); draws.encode(pool); ctx.submit().wait(); }
		for (unsigned i = 0; i < 3; ++i) { ctx.begin(); draws.encode(pool); ctx.cancel(); }
		require(pool.consume(0) == prefix * 16, "Cancelled interval lost prior prefix or read uninitialized data");
		ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(0);
		require(pool.consume(0) == 16, "Fresh interval after cancellation retained an old count");
	}
	pool.begin(0, true); ctx.begin(); draws.encode(pool); ctx.submit().wait();
	ctx.begin(); draws.encode(pool); ctx.cancel();
	ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(0);
	require(pool.consume(0) == 32, "Reuse after cancelled continuation reset a confirmed prefix");
	for (const bool prefix : {false, true})
	{
		pool.begin(0, true);
		if (prefix) { ctx.begin(); draws.encode(pool); ctx.submit().wait(); }
		const auto gate = [ctx.device() newSharedEvent];
		ctx.begin(); ctx.wait_event(gate, 1); draws.encode(pool); const auto queued = ctx.submit();
		require(queued.cancel_pending(), "Gated query continuation committed before cancellation");
		rejects([&] { queued.wait(); });
		require(pool.consume(0) == (prefix ? 16 : 0), "Queued cancellation lost prefix or read uninitialized counter");
		ctx.recover_completed_errors();
		ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(0);
		require(pool.consume(0) == 16, "Queued cancellation prevented fresh interval reuse");
	}
	std::cout << "Native uint32 wrap and zero/committed-prefix cancellation/reuse passed\n";
}
void pending_error_and_sparse_retirement()
{
	using namespace metal4;
	context ctx(3); fixture draws(ctx); query_pool pool(ctx, 2048, 2);
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } guard{gate};
	pool.begin(2047, true); ctx.begin(); ctx.wait_event(gate, 1); draws.encode(pool); const auto first = ctx.submit();
	ctx.begin(); draws.encode(pool); const auto second = ctx.submit();
	require(pool.statistics().live_segments == 2 && pool.statistics().leased_counters == 2 && !pool.ready(2047), "Different pending batches shared a writable query counter");
	ctx.begin(); rejects([&] { draws.encode(pool); }); ctx.cancel();
	gate.signaledValue = 1; first.wait(); second.wait();
	ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(2047);
	require(pool.consume(2047) == 48, "Completed pending segments lost accumulated visibility");
	const auto start = pool.statistics();
	for (const auto handle : {2046u, 2047u})
	{
		pool.begin(handle, true); ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(handle);
	}
	pool.begin(0, true); ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(0);
	require(pool.statistics().retirement_visits - start.retirement_visits == 2, "Counter pressure scanned unused query handles");
	for (const auto handle : {2046u, 2047u, 0u}) require(pool.consume(handle) == 16, "Sparse completed-query retirement lost a report");
	pool.begin(0, true); ctx.begin(); draws.encode(pool); ctx.submit().wait();
	const auto status = make_buffer(ctx, 4, buffer_storage::shared); *static_cast<std::uint32_t*>(status.contents) = 1;
	ctx.begin(); draws.encode(pool); ctx.require_gpu_success(status, 0, @"Intentional query status failure"); const auto failed = ctx.submit();
	rejects([&] { failed.wait(); });
	rejects([&] { (void)pool.consume(0); });
	ctx.recover_completed_errors(); ctx.begin();
	const std::uint32_t failed_handle[]{0};
	rejects([&] { (void)pool.encode_conditional_draw(failed_handle, MTLPrimitiveTypeTriangle, 0, 3); });
	rejects([&] { draws.encode(pool); }); ctx.cancel();
	pool.discard(0); pool.begin(0, true); ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(0);
	require(pool.consume(0) == 16, "Explicit query error discard did not restore counter reuse");
	std::cout << "Pending batch isolation/exhaustion,2 occupied-handle visits out of2048,and failed-result refusal/discard passed\n";
}
void render_to_buffer_blit()
{
	using namespace metal4;
	context ctx{3};
	fixture draws(ctx);
	query_pool pool(ctx, 1, 2);
	const auto readback = make_buffer(ctx, 8, buffer_storage::shared);
	for (bool separate_batch : {false, true})
		for (bool discard : {false, true})
		{
			ctx.begin();
			pool.begin(0, true);
			id<MTLBuffer> counter = nil;
			std::size_t offset = 0;
			draws.encode(pool, false, discard, [&](id<MTLBuffer> buffer, std::size_t at)
			{
				counter = buffer;
				offset = at;
			});
			pool.end(0);
			std::optional<submission> producer;
			if (separate_batch)
			{
				producer = ctx.submit();
				ctx.begin();
			}
			const auto before = ctx.encoding_statistics();
			copy_buffer(ctx, counter, offset, readback, 0, 8);
			ctx.submit().wait();
			if (producer) producer->wait();
			std::uint64_t raw = 0;
			std::memcpy(&raw, map_buffer(readback, 0, 8).data(), 8);
			require(raw == (discard ? 0u : 16u) && pool.consume(0) == raw, "Render-to-blit visibility dependency lost a native counter");
			const auto after = ctx.encoding_statistics();
			require(after.buffer_blit_queue_barriers == before.buffer_blit_queue_barriers + 1 &&
				after.buffer_blit_barriers == before.buffer_blit_barriers, "Visibility copy did not use the blit-stage queue barrier");
		}
	std::cout << "Native render-to-buffer blits: four exact visibility results, same and separate submission, narrowed consumer stages passed\n";
}
void bounded_counter_preclear()
{
	using namespace metal4;
	context ctx;
	fixture draws(ctx);
	constexpr unsigned count = 33;
	query_pool pool(ctx, count, count);
	for (bool discard : {false, true})
	{
		const auto encoders = ctx.encoding_statistics().compute_encoders;
		const auto initializations = pool.statistics().initializations;
		ctx.begin();
		for (unsigned handle = 0; handle < count; ++handle)
		{
			pool.begin(handle, true);
			draws.encode(pool, false, discard);
			pool.end(handle);
		}
		require(ctx.encoding_statistics().compute_encoders == encoders + 3 &&
			pool.statistics().initializations == initializations + count,
			"33 contiguous counters did not initialize in three bounded blits");
		ctx.submit().wait();
		for (unsigned handle = count; handle > 0; --handle)
			require(pool.consume(handle - 1) == (discard ? 0u : 16u),
				"Counter preclear lost a result or reused a stale value in the next recording");
		require(pool.statistics().leased_counters == 0, "Counter preclear retained unused leases");
	}
	std::cout << "Bounded visibility preclear: 66 exact counters across two recordings, three blits per 33, stale values cleared\n";
}

void conditional_indirect_draw()
{
	using namespace metal4;
	context ctx(3);
	fixture draws(ctx);
	query_pool sources(ctx, 3, 5), observed(ctx, 3, 3);
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { if (event.signaledValue < 2) event.signaledValue = 2; } } guard{gate};
	const std::uint32_t both[]{0, 1}, precise[]{0}, boolean[]{1};
	const auto snapshot = [&](std::span<const std::uint32_t> handles)
	{
		const auto command = sources.encode_conditional_draw(handles, MTLPrimitiveTypeTriangle, 0, 3);
		require(command.has_value(), "Same-recording query did not provide an indirect draw");
		return *command;
	};
	const auto measure = [&](const conditional_draw& command, std::uint32_t handle = 0, bool discard = false)
	{
		observed.begin(handle, true);
		draws.encode(observed, false, discard, {}, &command);
		observed.end(handle);
	};
	const auto colors = make_buffer(ctx, 3 * 256, buffer_storage::shared);
	const auto read_color = [&](unsigned index) { texture_to_buffer(ctx, draws.color, {}, colors, index * 256, 256); };
	const std::uint64_t initial = 0xfffffff0ull;
	const auto seed = make_upload(ctx, std::as_bytes(std::span{&initial, 1}));
	ctx.begin();
	ctx.wait_event(gate, 1);
	sources.begin(0, true);
	draws.encode(sources, false, false, [&](id<MTLBuffer> buffer, std::size_t offset) { copy_buffer(ctx, seed, 0, buffer, offset, 8); });
	sources.end(0);
	sources.begin(1, false);
	draws.encode(sources, false, true);
	sources.end(1);
	const auto zero = snapshot(both);
	sources.discard(1);
	sources.begin(1, false);
	draws.encode(sources);
	sources.end(1);
	const auto visible = snapshot(both);
	measure(zero);
	read_color(0);
	measure(visible, 1);
	read_color(1);
	measure(visible, 2, true);
	read_color(2);
	const auto recorded = ctx.submit();
	require(!recorded.ready(), "Gated conditional work completed prematurely");
	gate.signaledValue = 1;
	recorded.wait();
	require(observed.consume(0) == 0 && observed.consume(1) == 16 && observed.consume(2) == 0,
		"Conditional draws did not preserve zero/nonzero snapshot across handle reuse");
	const std::array<std::uint8_t, 4> clear_pixel{0, 255, 0, 255}, drawn_pixel{255, 255, 255, 255};
	require(std::memcmp(map_buffer(colors, 0, 4).data(), clear_pixel.data(), 4) == 0 &&
		std::memcmp(map_buffer(colors, 256, 4).data(), drawn_pixel.data(), 4) == 0 &&
		std::memcmp(map_buffer(colors, 512, 4).data(), clear_pixel.data(), 4) == 0,
		"Indirect draws changed clear/discard results");
	require(sources.consume(0) == 0 && sources.consume(1) == 1, "GPU reduction altered the CPU reports");
	ctx.begin();
	rejects([&] { measure(visible); });
	observed.discard(0);
	ctx.cancel();

	// The same queue can record a second segment while a producer awaits its
	// gate. Confirm fallback before release, then snapshot the completed and
	// still-recording segments without consuming either.
	sources.begin(0, true);
	ctx.begin();
	ctx.wait_event(gate, 2);
	draws.encode(sources);
	const auto pending = ctx.submit();
	ctx.begin();
	require(!sources.encode_conditional_draw(precise, MTLPrimitiveTypeTriangle, 0, 3),
		"An unresolved foreign recording supplied an indirect command");
	draws.encode(sources, false, false, [&](id<MTLBuffer> buffer, std::size_t offset)
	{
		const std::uint64_t wrap = 0xffffffe0ull;
		const auto value = make_upload(ctx, std::as_bytes(std::span{&wrap, 1}));
		copy_buffer(ctx, value, 0, buffer, offset, 8);
	});
	sources.end(0);
	gate.signaledValue = 2;
	pending.wait();
	measure(snapshot(precise));
	ctx.submit().wait();
	require(observed.consume(0) == 0 && sources.consume(0) == 0,
		"Conditional reduction failed to wrap the sum across multiple segments");

	for (const bool prefix : {false, true})
	{
		sources.begin(1, false);
		if (prefix) { ctx.begin(); draws.encode(sources); ctx.submit().wait(); }
		ctx.begin(); draws.encode(sources); ctx.cancel();
		sources.end(1);
		ctx.begin(); measure(snapshot(boolean)); ctx.submit().wait();
		require(observed.consume(0) == (prefix ? 16 : 0) && sources.consume(1) == std::uint32_t(prefix),
			"GPU reduction mishandled a cancelled counter epoch");
	}

	query_pool retired(ctx, 2, 1);
	ctx.begin(); retired.begin(0, true); draws.encode(retired); retired.end(0); ctx.submit().wait();
	ctx.begin(); retired.begin(1, false); draws.encode(retired, false, true); retired.end(1);
	const auto accumulated = retired.encode_conditional_draw(precise, MTLPrimitiveTypeTriangle, 0, 3);
	require(accumulated.has_value(), "Retired prefix could not be encoded");
	measure(*accumulated); ctx.submit().wait();
	require(observed.consume(0) == 16 && retired.consume(0) == 16 && retired.consume(1) == 0,
		"Reallocated native counter replaced an already retired report prefix");
	std::cout << "Native conditional indirect draws: zero/nonzero, clear/discard, reuse, multi-segment wrap, retired/cancelled prefixes and pending fallback passed\n";
}

void conditional_report_draw()
{
	using namespace metal4;
	context ctx;
	fixture draws(ctx);
	query_pool sources(ctx, 3, 3), observed(ctx, 10, 10);
	const auto geometry = prepared_geometry::prepare(ctx, plan_indexed_references(
		{list_topology::triangles, {{0, 0}, {1, 1}, {2, 2}}, std::nullopt}));
	const auto measure = [&](const report_query_snapshot& report, std::uint32_t target)
	{
		const auto command = sources.encode_report_conditional_draw(report, geometry);
		require(command.has_value(), "Available report did not produce a conditional draw");
		observed.begin(target, true);
		draws.encode(observed, false, false, {}, &*command);
		observed.end(target);
	};
	constexpr std::uint32_t negative_sixteen = 0xfffffff0u;
	const std::uint64_t counter_value = std::uint64_t(negative_sixteen) - 16;
	const auto seed = make_upload(ctx, std::as_bytes(std::span{&counter_value, 1}));
	ctx.begin();
	sources.begin(0, true);
	draws.encode(sources, false, false, [&](id<MTLBuffer> buffer, std::size_t offset) { copy_buffer(ctx, seed, 0, buffer, offset, 8); });
	sources.end(0);
	sources.begin(1, true);
	draws.encode(sources);
	sources.end(1);
	sources.begin(2, false);
	draws.encode(sources);
	sources.end(2);
	const report_query_source pair[]{{0}, {1}}, first[]{{0}}, consumed[]{{1, negative_sixteen}},
		boolean_consumed[]{{2, 0xffffffffu}};
	std::array<rsx::reports::occlusion_query_info, 5> query_records{};
	for (std::uint32_t i = 0; i < query_records.size(); ++i)
	{
		query_records[i].driver_handle = i;
		query_records[i].sync_tag = i + 1;
		query_records[i].num_draws = 1;
		query_records[i].pending = true;
	}
	constexpr std::uint32_t target_address = 0x2000;
	std::array<rsx::reports::queued_report_write, 5> writers{};
	const auto claim = [&](unsigned position, unsigned query, std::uint32_t slot, std::uint32_t address, std::uint32_t type)
	{
		writers[position].query = &query_records[query];
		writers[position].counter_tag = slot;
		writers[position].sink = vm::cast(address);
		writers[position].type = type;
	};
	claim(0, 0, 1, 0x1000, CELL_GCM_ZCULL_STATS3);
	claim(1, 3, 0, 0x1010, CELL_GCM_ZPASS_PIXEL_CNT);
	claim(2, 4, 1, 0x1020, CELL_GCM_ZCULL_STATS2);
	claim(3, 1, 1, target_address, CELL_GCM_ZPASS_PIXEL_CNT);
	claim(4, 2, 1, 0x2010, CELL_GCM_ZPASS_PIXEL_CNT);
	std::array<rsx::reports::query_stat_counter, 2> stats{};
	stats[1].result = 9;
	query_records[1].result = std::uint32_t{0} - 9;
	const auto capture = [&] { return detail::capture_report_condition(writers, stats, target_address, 2, true, sources.capacity()); };
	const auto captured = capture();
	require(captured && captured->retired == 9 && captured->sources.size() == 2 &&
		captured->sources[0].handle == 0 && captured->sources[1].handle == 1 &&
		captured->sources[1].consumed == std::uint32_t{0} - 9,
		"RSX report selection used the wrong slot, retirement order or prefix");
	query_records[1].result = 0;
	stats[1].result = 1;
	measure(captured->snapshot(), 0); // The captured prefixes and two native values cancel modulo uint32.
	writers[3].forwarder = &writers[4];
	require(!capture(), "Forwarded report used GPU counters instead of its guest-memory value");
	writers[3].forwarder = nullptr;
	query_records[1].pending = false;
	require(!capture(), "Retired report reused a live handle without an epoch");
	query_records[1].pending = true;
	query_records[1].driver_handle = 0;
	require(!capture(), "RSX report accepted a duplicate driver handle");
	query_records[1].driver_handle = 1;
	writers[1].sink = {};
	require(!capture(), "RSX report bypassed an unclaimed queue entry");
	measure({pair, 0, false}, 1); // Boolean report stops at the first hit.
	measure({first, 16, true}, 2);
	measure({first, 17, true}, 3);
	measure({consumed, 0, true}, 4);
	measure({consumed, 1, true}, 5);
	measure({boolean_consumed, 0, false}, 6);
	measure({{}, 0, true}, 7);
	measure({{}, 3, true}, 8);
	measure({first, 16, false}, 9);
	ctx.submit().wait();
	const std::uint32_t expected[]{0, 16, 0, 16, 0, 16, 0, 0, 16, 16};
	for (std::uint32_t i = 0; i < std::size(expected); ++i)
		require(observed.consume(i) == expected[i], "Metal 4 report reduction changed the conditional result");
	require(sources.consume(0) == negative_sixteen && sources.consume(1) == 16 && sources.consume(2) == 1,
		"GPU report aggregation consumed or modified the CPU query results");

	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } guard{gate};
	ctx.begin(); ctx.wait_event(gate, 1);
	sources.begin(0, true); draws.encode(sources); sources.end(0);
	const auto producer = ctx.submit();
	ctx.begin();
	require(!sources.encode_report_conditional_draw({first, 16, true}, geometry),
		"Precise report accepted an unresolved foreign producer");
	measure({first, 16, false}, 0);
	const auto consumer = ctx.submit();
	gate.signaledValue = 1;
	producer.wait();
	consumer.wait();
	require(observed.consume(0) == 16 && sources.consume(0) == 16, "Known Boolean report hit depended on a later query");

	const auto error = make_buffer(ctx, sizeof(std::uint32_t), buffer_storage::shared);
	*static_cast<std::uint32_t*>(error.contents) = 1;
	ctx.begin();
	sources.begin(1, true); draws.encode(sources); sources.end(1);
	ctx.require_gpu_success(error, 0, @"Intentional conditional report producer failure");
	const auto failed = ctx.submit();
	rejects([&] { failed.wait(); });
	ctx.recover_completed_errors();
	ctx.begin();
	require(!sources.encode_report_conditional_draw({pair, 0, false}, geometry),
		"Failed Boolean producer bypassed the scheduler fallback");
	rejects([&] { (void)sources.encode_report_conditional_draw({pair, 0, true}, geometry); });
	measure({pair, 1, false}, 0);
	ctx.submit().wait();
	require(observed.consume(0) == 16, "Known Boolean report hit inherited an irrelevant producer failure");
	std::cout << "Native conditional report sums: 10 exact GPU outcomes, RSX queue selection, prefixes, uint32 wrap, Boolean stop and pending/error policy passed\n";
}

void retained_report_predicate()
{
	using namespace metal4;
	context ctx(3);
	fixture draws(ctx);
	query_pool sources(ctx, 1, 1), observed(ctx, 3, 3);
	const auto geometry = prepared_geometry::prepare(ctx, plan_indexed_references(
		{list_topology::triangles, {{0, 0}, {1, 1}, {2, 2}}, std::nullopt}));
	const auto measure = [&](const report_predicate& predicate, std::uint32_t target)
	{
		const auto command = sources.encode_predicated_draw(predicate, geometry);
		observed.begin(target, true);
		draws.encode(observed, false, false, {}, &command);
		observed.end(target);
	};
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { if (!event.signaledValue) event.signaledValue = 1; } } guard{gate};
	const report_query_source handle[]{{0}};
	ctx.begin();
	ctx.wait_event(gate, 1);
	sources.begin(0, true); draws.encode(sources); sources.end(0);
	const auto positive = sources.capture_report_predicate({handle, 0, true});
	const auto negative = sources.capture_report_predicate({{}, 0, true});
	require(positive && negative, "Recording could not capture both native report values");
	measure(*positive, 0);
	const auto producer = ctx.submit();
	ctx.begin();
	measure(*positive, 1);
	measure(*negative, 2);
	const auto consumer = ctx.submit();
	require(!producer.ready() && !consumer.ready(), "An unreleased report dependency completed prematurely");
	gate.signaledValue = 1;
	producer.wait();
	consumer.wait();
	require(observed.consume(0) == 16 && observed.consume(1) == 16 && observed.consume(2) == 0,
		"Same/next-recording predicates did not produce exact visibility");

	sources.discard(0);
	ctx.begin();
	sources.begin(0, true); draws.encode(sources, true); sources.end(0);
	const auto replaced = sources.capture_report_predicate({handle, 0, true});
	require(replaced.has_value(), "Reused report handle could not be captured");
	measure(*positive, 0);
	measure(*replaced, 1);
	ctx.submit().wait();
	require(observed.consume(0) == 16 && observed.consume(1) == 0 && sources.consume(0) == 0,
		"Recycling the only native counter changed an earlier predicate");

	context foreign;
	query_pool foreign_sources(foreign, 1, 1);
	foreign.begin();
	rejects([&] { (void)foreign_sources.encode_predicated_draw(*positive, geometry); });
	foreign.cancel();

	const auto untouched = make_buffer(ctx, 4, buffer_storage::shared);
	const auto status = make_buffer(ctx, 4, buffer_storage::shared);
	*static_cast<std::uint32_t*>(untouched.contents) = 0;
	*static_cast<std::uint32_t*>(status.contents) = 1;
	for (bool gpu_failure : {false, true})
	{
		const auto failure_gate = [ctx.device() newSharedEvent];
		release unblock{failure_gate};
		ctx.begin();
		ctx.wait_event(failure_gate, 1);
		const auto predicate = sources.capture_report_predicate({{}, 1, true});
		require(predicate.has_value(), "Known report prefix was not captured");
		if (gpu_failure) ctx.require_gpu_success(status, 0, @"Intentional predicate GPU failure");
		const auto rejected_producer = ctx.submit();
		ctx.begin();
		measure(*predicate, 0);
		fill_buffer(ctx, untouched, 0, 4, 255);
		const auto rejected_consumer = ctx.submit();
		if (!gpu_failure) require(rejected_producer.cancel_pending(), "Predicate producer committed before cancellation");
		failure_gate.signaledValue = 1;
		rejects([&] { rejected_producer.wait(); });
		rejects([&] { rejected_consumer.wait(); });
		require(rejected_consumer.cancelled() && *static_cast<const std::uint32_t*>(untouched.contents) == 0,
			"Failed report producer allowed dependent GPU work");
		require(observed.consume(0) == 0, "Rejected predicate produced visibility");
		ctx.recover_completed_errors();
		ctx.begin();
		rejects([&] { (void)sources.encode_predicated_draw(*predicate, geometry); });
		ctx.cancel();
	}
	std::cout << "Native retained report predicates: same and pending recordings, query-slot reuse, foreign queue refusal and cancellation/error propagation passed\n";
}

void pending_report_segments()
{
	using namespace metal4;
	context ctx(3);
	fixture draws(ctx);
	query_pool sources(ctx, 2, 4), observed(ctx, 1, 1);
	const auto geometry = prepared_geometry::prepare(ctx, plan_indexed_references(
		{list_topology::triangles, {{0, 0}, {1, 1}, {2, 2}}, std::nullopt}));
	const report_query_source first[]{{0}}, both[]{{0}, {1}};
	const auto capture = [&](const report_query_snapshot& report)
	{
		const auto predicate = sources.capture_report_predicate(report);
		require(predicate.has_value(), "Pending report failed to capture a native predicate");
		const auto draw = sources.encode_predicated_draw(*predicate, geometry);
		observed.begin(0, true);
		draws.encode(observed, false, false, {}, &draw);
		observed.end(0);
	};
	struct gate_guard
	{
		id<MTLSharedEvent> event;
		~gate_guard() { if (!event.signaledValue) event.signaledValue = 1; }
		void open() { event.signaledValue = 1; }
	};
	{
		gate_guard gate{[ctx.device() newSharedEvent]};
		ctx.begin(); ctx.wait_event(gate.event, 1);
		sources.begin(1, true); draws.encode(sources); sources.end(1);
		sources.begin(0, true); draws.encode(sources);
		const auto producer = ctx.submit();
		ctx.begin(); draws.encode(sources); sources.end(0);
		capture({both, 0, true});
		const auto consumer = ctx.submit();
		require(!producer.ready() && !consumer.ready(), "Pending and current query segments were published early");
		gate.open();
		producer.wait(); consumer.wait();
		require(sources.consume(0) == 32 && sources.consume(1) == 16 && observed.consume(0) == 16,
			"Pending and current query segments did not reduce together");
	}
	for (bool prefix : {false, true})
	{
		std::size_t previous_offset = 0;
		ctx.begin(); sources.begin(0, true);
		draws.encode(sources, false, false, [&](id<MTLBuffer>, std::size_t offset) { previous_offset = offset; });
		ctx.submit().wait();
		if (!prefix)
		{
			sources.discard(0);
			sources.begin(0, true);
		}
		gate_guard gate{[ctx.device() newSharedEvent]};
		ctx.begin(); ctx.wait_event(gate.event, 1);
		draws.encode(sources, false, false, [&](id<MTLBuffer>, std::size_t offset)
			{ require(offset == previous_offset, "Cancellation fixture did not reuse the known nonzero native counter"); });
		sources.end(0);
		const auto producer = ctx.submit();
		ctx.begin(); capture({first, 0, true}); const auto consumer = ctx.submit();
		require(producer.cancel_pending(), "Visibility producer committed before cancellation");
		gate.open();
		rejects([&] { producer.wait(); });
		consumer.wait();
		const auto expected = prefix ? 16u : 0u;
		require(sources.consume(0) == expected && observed.consume(0) == expected,
			"Cancelled producer lost its committed prefix or exposed a stale counter");
		ctx.recover_completed_errors();
	}

	const auto error = make_buffer(ctx, 4, buffer_storage::shared);
	const auto readback = make_buffer(ctx, 256, buffer_storage::shared);
	*static_cast<std::uint32_t*>(error.contents) = 1;
	for (bool precise : {false, true})
	{
		ctx.begin(); sources.begin(0, true); draws.encode(sources); sources.end(0); ctx.submit().wait();
		gate_guard gate{[ctx.device() newSharedEvent]};
		ctx.begin(); ctx.wait_event(gate.event, 1);
		sources.begin(1, true); draws.encode(sources); sources.end(1);
		ctx.require_gpu_success(error, 0, @"Intentional pending source failure");
		const auto producer = ctx.submit();
		ctx.begin(); capture({both, 0, precise});
		texture_to_buffer(ctx, draws.color, {}, readback, 0, 256);
		const auto consumer = ctx.submit();
		gate.open();
		rejects([&] { producer.wait(); });
		if (precise)
		{
			rejects([&] { consumer.wait(); });
			const std::array<std::uint8_t, 4> clear{0, 255, 0, 255};
			require(std::memcmp(map_buffer(readback, 0, 4).data(), clear.data(), 4) == 0,
				"Relevant report producer error permitted color output");
			observed.discard(0);
			ctx.recover_completed_errors();
			ctx.begin();
			const auto failed_snapshot = sources.capture_report_predicate({both, 0, true});
			require(failed_snapshot.has_value(), "Completed query error lost the ordered GPU validation path");
			const auto failed_capture = ctx.submit();
			rejects([&] { failed_capture.wait(); });
		}
		else
		{
			consumer.wait();
			require(observed.consume(0) == 16, "Boolean hit was invalidated by an irrelevant later producer error");
		}
		require(sources.consume(0) == 16, "Pending capture consumed the independent CPU counter");
		sources.discard(1);
		ctx.recover_completed_errors();
	}
	std::cout << "Native pending report sources: mixed recordings, stale counter suppression, committed prefix and ordered failure handling passed\n";
}

void submitted_report_predicates()
{
	using namespace metal4;
	using namespace std::chrono_literals;
	context ctx(2, 64, 5s, context::recording_overlap::explicit_predecessor);
	fixture draws(ctx);
	query_pool sources(ctx, 1, 1), observed(ctx, 1, 1);
	const auto geometry = prepared_geometry::prepare(ctx, plan_indexed_references(
		{list_topology::triangles, {{0, 0}, {1, 1}, {2, 2}}, std::nullopt}));
	const report_query_source source[]{{0}};
	require(!ctx.last_submission(), "Empty queue reported a predecessor");
	for (unsigned profile = 0; profile < 3; ++profile)
	{
		const bool visible = profile != 0, cancelled = profile == 2;
		const auto event = [ctx.device() newSharedEvent];
		struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } release_gate{event};
		ctx.begin();
		ctx.wait_event(event, 1);
		sources.begin(0, true);
		draws.encode(sources, !visible);
		sources.end(0);
		const auto producer = ctx.submit();
		const auto predicate = sources.submit_report_predicate({source, 0, true});
		const auto reduction = ctx.last_submission();
		require(predicate && reduction && !reduction->same_batch(producer) && !producer.ready() && !reduction->ready(),
			"Report reduction blocked on the producer or completed before its event");
		bool waited = false;
		try { reduction->wait(1ms); }
		catch (const submission_wait_timeout&) { waited = true; }
		require(waited, "Report was visible before the ordered reduction");
		if (cancelled) require(producer.cancel_pending(), "Could not cancel gated report producer");
		event.signaledValue = 1;
		if (cancelled)
		{
			rejects([&] { producer.wait(5s); });
			rejects([&] { reduction->wait(5s); });
			require(reduction->ready() && producer.cancelled(), "Report cancellation did not settle both submissions");
			sources.discard(0);
			ctx.recover_completed_errors();
		}
		else
		{
			reduction->wait(5s);
			ctx.begin();
			const auto draw = sources.encode_predicated_draw(*predicate, geometry);
			observed.begin(0, true);
			draws.encode(observed, false, false, {}, &draw);
			observed.end(0);
			ctx.submit().wait(5s);
			const auto expected = visible ? 16u : 0u;
			require(sources.consume(0) == expected && observed.consume(0) == expected,
				"Deferred report reduction returned the wrong count or predicate");
		}
	}
	std::cout << "Submitted report predicates: reduction returned before the gated producer; exact zero and sixteen counts and cancellation propagated\n";
}

void cpu_known_report_predicates()
{
	using namespace metal4;
	using namespace std::chrono_literals;
	context ctx(2, 64, 5s, context::recording_overlap::explicit_predecessor);
	fixture draws(ctx);
	query_pool sources(ctx, 2, 1), observed(ctx, 1, 1);
	const auto geometry = prepared_geometry::prepare(ctx, plan_indexed_references(
		{list_topology::triangles, {{0, 0}, {1, 1}, {2, 2}}, std::nullopt}));
	const auto verify = [&](const report_predicate& predicate, unsigned expected)
	{
		ctx.begin();
		const auto draw = sources.encode_predicated_draw(predicate, geometry);
		observed.begin(0, true);
		draws.encode(observed, false, false, {}, &draw);
		observed.end(0);
		ctx.submit().wait(5s);
		require(observed.consume(0) == expected, "CPU-known report changed GPU draw visibility");
	};
	const auto zero = sources.submit_report_predicate({{}, 0, true});
	require(zero && !ctx.last_submission(), "Empty CPU-known report submitted GPU work");
	verify(*zero, 0);

	ctx.begin();
	sources.begin(0, true);
	draws.encode(sources);
	sources.end(0);
	ctx.submit().wait(5s);
	const auto event = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } release_gate{event};
	ctx.begin();
	ctx.wait_event(event, 1);
	sources.begin(1, true);
	draws.encode(sources); // Reclaims source 0 into its CPU accumulator.
	sources.end(1);
	const auto producer = ctx.submit();
	const auto before = ctx.encoding_statistics().compute_encoders;
	const report_query_source accumulated[]{{0, 0}}, wrapped[]{{0, std::uint32_t{0} - 16}}, live[]{{1, 0}};
	const auto hit = sources.submit_report_predicate({accumulated, 0, true});
	const auto wrap_zero = sources.submit_report_predicate({wrapped, 0, true});
	const auto wrap_one = sources.submit_report_predicate({wrapped, 1, true});
	const auto retired_hit = sources.submit_report_predicate({live, 1, false});
	const auto latest = ctx.last_submission();
	require(hit && wrap_zero && wrap_one && retired_hit && latest && latest->same_batch(producer) &&
		!producer.ready() && ctx.encoding_statistics().compute_encoders == before,
		"CPU-known report waited or recorded a redundant reduction");
	event.signaledValue = 1;
	producer.wait(5s);
	verify(*hit, 16);
	verify(*wrap_zero, 0);
	verify(*wrap_one, 16);
	verify(*retired_hit, 16);
	require(sources.consume(0) == 16 && sources.consume(1) == 16, "CPU-known report consumed source counters");
	std::cout << "CPU-known reports: five exact GPU draw outcomes without reduction submissions; gated predecessor, uint32 wrap and retired boolean hit preserved\n";
}
} // namespace
int main(int argc, char** argv)
{
	@autoreleasepool
	{
		try
		{
			if (argc == 2 && std::string_view(argv[1]) == "--conditional-only")
			{
				conditional_indirect_draw();
				conditional_report_draw();
				retained_report_predicate();
				pending_report_segments();
				submitted_report_predicates();
				cpu_known_report_predicates();
				return 0;
			}
			require(argc == 1, "Unknown native query test selector");
			render_to_buffer_blit();
			bounded_counter_preclear();
			conditional_indirect_draw();
			conditional_report_draw();
			retained_report_predicate();
			pending_report_segments();
			submitted_report_predicates();
			cpu_known_report_predicates();
			using namespace metal4;
			intervals_and_boundaries();
			pending_error_and_sparse_retirement();
			context ctx;
			query_pool pool(ctx, 4, 4);
			fixture draws(ctx);
			// Count native allocation size,not just the eight-byte payload.
			const auto probe = make_buffer(ctx, 8, buffer_storage::shared);
			const auto charge = std::max(probe.length, probe.allocatedSize);
			auto budget = std::make_shared<heap_budget>(ctx, charge);
			{
				query_pool charged(ctx, 1, 1, budget);
				const auto stats = budget->statistics();
				require(stats.reserved_bytes == charge && stats.live_buffers == 1 && stats.live_heaps == 0, "Counter allocation not charged at native size");
				try
				{
					query_pool excess(ctx, 1, 1, budget);
					throw std::runtime_error("Counter allocation bypassed shared budget");
				}
				catch (const heap_budget_error& error)
				{
					require(error.reserved == charge && error.budget == charge, "Counter budget failure lost accounting");
				}
				require(budget->statistics().reserved_bytes == charge, "Failed allocation changed counter reservation");
			}
			require(budget->statistics().reserved_bytes == 0 && budget->statistics().live_buffers == 0, "Destroyed counter buffer remains charged");
			{
				auto insufficient = std::make_shared<heap_budget>(ctx, charge - 1);
				try
				{
					query_pool rounded(ctx, 1, 1, insufficient);
					throw std::runtime_error("Rounded counter allocation bypassed budget");
				}
				catch (const heap_budget_error&)
				{
				}
				require(insufficient->statistics().reserved_bytes == 0, "Refused rounded allocation leaked reservation");
			}
			std::cout << "Counter native-size budget,exhaustion,rounded refusal and destruction release passed\n";
			{
				auto descriptor = [MTLHeapDescriptor new];
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
				descriptor.type = MTLHeapTypePlacement;
				descriptor.size = 65536;
				const auto heap_probe = [ctx.device() newHeapWithDescriptor:descriptor];
				require(bool(heap_probe), "Mixed budget heap probe unavailable");
				auto mixed = std::make_shared<heap_budget>(ctx, heap_probe.size + charge);
				query_pool charged(ctx, 1, 1, mixed);
				id<MTLHeap> heap = mixed->allocate(descriptor);
				require(mixed->statistics().live_heaps == 1 && mixed->statistics().live_buffers == 1 &&
							mixed->statistics().reserved_bytes == heap.size + charge,
						"Heap and counter do not share reservation total");
				try
				{
					(void)mixed->shared_buffer(8);
					throw std::runtime_error("Mixed heap/counter budget overcommitted");
				}
				catch (const heap_budget_error&)
				{
				}
				heap = nil;
				require(mixed->statistics().live_heaps == 0 && mixed->statistics().reserved_bytes == charge, "Heap release removed live counter charge");
				std::cout << "Counter and private heap share one budget;heap release preserves live counter charge passed\n";
			}
			pool.begin(0, true);
			ctx.begin();
			draws.encode(pool);
			draws.encode(pool);
			auto observed = ctx.recording_submission();
			require(!observed.ready() && !pool.ready(0), "Recording query published completion");
			ctx.submit().wait();
			pool.end(0);
			require(observed.ready() && !observed.cancelled() && pool.ready(0) && pool.consume(0) == 32, "Precise multiencoder query differs");
			require(pool.consume(0) == 0, "Consumed query counted twice");
			pool.begin(0, false);
			ctx.begin();
			draws.encode(pool);
			draws.encode(pool);
			ctx.submit().wait();
			pool.end(0);
			require(pool.consume(0) == 1, "Boolean query summed hits rather than OR");
			for (const bool discard : {false, true})
			{
				pool.begin(0, true);
				ctx.begin();
				draws.encode(pool, !discard, discard);
				ctx.submit().wait();
				pool.end(0);
				require(pool.consume(0) == 0, "Depth/shader-rejected fragments contributed visibility");
			}
			std::cout << "Native precise32/boolean1 across encoders;depth rejection,fragment discard and single consumption passed\n";
			pool.begin(0, true);
			ctx.begin();
			draws.encode(pool);
			auto cancelled = ctx.recording_submission();
			ctx.cancel();
			pool.end(0);
			require(cancelled.ready() && cancelled.cancelled() && pool.consume(0) == 0, "Cancelled encoding counted visibility");
			rejects([&]
					{ cancelled.wait(); });
			ctx.begin();
			draws.encode(pool);
			ctx.submit().wait();
			require(cancelled.cancelled(), "Reused slot changed old cancellation token");
			pool.discard(0);
			pool.begin(1, true);
			ctx.begin();
			for (const auto handle : {1u, 0u, 2u, 3u})
			{
				if (handle != 1) pool.begin(handle, true);
				draws.encode(pool); pool.end(handle);
			}
			pool.discard(1); pool.begin(1, true);
			rejects([&]
					{ draws.encode(pool); });
			ctx.cancel();
			for (unsigned handle = 0; handle < 4; ++handle) pool.discard(handle);
			pool.begin(1, true);
			ctx.begin();
			draws.encode(pool);
			ctx.submit().wait();
			pool.end(1);
			require(pool.consume(1) == 16, "Counter arena not reusable after cancelled exhaustion");
			std::cout << "Cancelled observed tokens,slot reuse,bounded counter exhaustion and safe reuse passed\n";
			query_pool pressure(ctx, 1, 2);
			pressure.begin(0, true);
			for (unsigned index = 0; index < 40; ++index)
			{
				ctx.begin();
				draws.encode(pressure);
				ctx.submit().wait();
			}
			pressure.end(0);
			require(pressure.consume(0) == 640, "Bounded arena retirement lost active-query accumulation");
			require(pressure.statistics().acquired == 1 && pressure.statistics().initializations == 1 && pressure.statistics().reused_completed == 39,
				"Completed draws repeated slot allocation/zeroing");
			std::cout << "Forty completed draws/640 samples:one counter,one initialization,39 continuations passed\n";
			const auto gate = [ctx.device() newSharedEvent];
			require(bool(gate), "Query gate unavailable");
			auto delayed = std::make_unique<query_pool>(ctx, 1, 1, budget);
			delayed->begin(0, true);
			ctx.begin();
			ctx.wait_event(gate, 1);
			// Match DrawExecutor's draw scope so Metal validation temporaries
			// cannot keep the counter alive beyond submission completion.
			@autoreleasepool
			{
				draws.encode(*delayed);
			}
			const auto pending = ctx.submit();
			delayed->end(0);
			require(!delayed->ready(0), "Pending GPU query appeared ready");
			delayed->discard(0);
			delayed->begin(0, true);
			// End/discard cannot recycle the sole slot while pending GPU owns it.
			MTL4RenderPassDescriptor* dummy = [MTL4RenderPassDescriptor new];
			ctx.begin();
			rejects([&]
					{ (void)delayed->attach(ctx, dummy); });
			ctx.cancel();
			delayed.reset();
			require(budget->statistics().reserved_bytes == charge, "Pending GPU counter released shared reservation early");
			try
			{
				query_pool excess(ctx, 1, 1, budget);
				throw std::runtime_error("Pending counter reservation allowed budget overcommit");
			}
			catch (const heap_budget_error&)
			{
			}
			gate.signaledValue = 1;
			pending.wait();
			require(budget->statistics().reserved_bytes == 0, "Completed discarded query retained counter reservation");
			query_pool reusable(ctx, 1, 1, budget);
			require(budget->statistics().live_buffers == 1, "Completed counter budget cannot be reused");
			std::cout << "Pending discard retains counter lease;GPU work survives query-pool destruction passed\n";
			std::cout << "Shared counter budget remains charged through pending destruction and is reusable after GPU completion passed\n";
			rejects([&]
					{ pool.begin(4, true); });
			pool.begin(2, true);
			rejects([&]
					{ pool.begin(3, true); });
			rejects([&]
					{ pool.end(3); });
			pool.end(2);
			require(pool.consume(2) == 0, "Empty native query was nonzero");
			rejects([&]
					{ (void)ctx.recording_submission(); });
			std::cout << "Invalid handle/nested/wrong-end/unrecorded token and empty queries passed;no RPCS3/app/UI started\n";
			return 0;
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
