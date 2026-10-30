#include "M4Pipeline.h"
#include "M4Query.h"
#include "M4Texture.h"

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
		const std::function<void(id<MTLBuffer>, std::size_t)>& before = {})
	{
		using namespace metal4;
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = color;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
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
		commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
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
	rejects([&] { draws.encode(pool); }); ctx.cancel();
	pool.discard(0); pool.begin(0, true); ctx.begin(); draws.encode(pool); ctx.submit().wait(); pool.end(0);
	require(pool.consume(0) == 16, "Explicit query error discard did not restore counter reuse");
	std::cout << "Pending batch isolation/exhaustion,2 occupied-handle visits out of2048,and failed-result refusal/discard passed\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
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
			draws.encode(*delayed);
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
			// wait() publishes only after completion-owned leases are released.
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
