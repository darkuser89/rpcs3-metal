#include "M4Pipeline.h"
#include "M4Draw.h"
#include "M4Texture.h"
#include "M4Resources.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

// Test-only callback gate around the real native compiler. It never fabricates
// a library/PSO: every result below comes from an actual MTL4CompilerTask.
@interface M4GatedCompiler : NSObject
{
	id<MTL4Compiler> _native;
	std::mutex _mutex;
	std::condition_variable _changed;
	std::deque<dispatch_block_t> _callbacks;
	bool _holding;
	std::atomic<bool> _mainThreadLaunch;
}
- (instancetype)initWithNative:(id<MTL4Compiler>)native;
- (id<MTLDevice>)device;
- (void)deliver:(dispatch_block_t)callback;
- (void)waitForCallbacks:(std::size_t)count;
- (void)releaseCallbacks;
- (bool)mainThreadLaunch;
@end

@implementation M4GatedCompiler
- (instancetype)initWithNative:(id<MTL4Compiler>)native
{
	if ((self = [super init])) { _native = native; _holding = true; _mainThreadLaunch = false; }
	return self;
}
- (id<MTLDevice>)device { return _native.device; }
- (void)deliver:(dispatch_block_t)callback
{
	{
		std::lock_guard lock(_mutex);
		if (_holding) { _callbacks.push_back([callback copy]); _changed.notify_all(); return; }
	}
	callback();
}
- (void)waitForCallbacks:(std::size_t)count
{
	std::unique_lock lock(_mutex);
	if (!_changed.wait_for(lock, std::chrono::seconds(10), [&] { return _callbacks.size() >= count; }))
		throw std::runtime_error("Timed out waiting for real native compiler callbacks");
}
- (void)releaseCallbacks
{
	std::deque<dispatch_block_t> callbacks;
	{ std::lock_guard lock(_mutex); _holding = false; callbacks.swap(_callbacks); }
	for (auto callback : callbacks) callback();
}
- (bool)mainThreadLaunch { return _mainThreadLaunch; }
- (id<MTL4CompilerTask>)newLibraryWithDescriptor:(MTL4LibraryDescriptor*)descriptor completionHandler:(MTLNewLibraryCompletionHandler)completion
{
	if ([NSThread isMainThread]) _mainThreadLaunch = true;
	return [_native newLibraryWithDescriptor:descriptor completionHandler:^(id<MTLLibrary> result, NSError* error)
		{ [self deliver:^{ completion(result, error); }]; }];
}
- (id<MTL4CompilerTask>)newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor*)descriptor compilerTaskOptions:(MTL4CompilerTaskOptions*)options completionHandler:(MTLNewComputePipelineStateCompletionHandler)completion
{
	if ([NSThread isMainThread]) _mainThreadLaunch = true;
	return [_native newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:options completionHandler:^(id<MTLComputePipelineState> result, NSError* error)
		{ [self deliver:^{ completion(result, error); }]; }];
}
- (id<MTL4CompilerTask>)newRenderPipelineStateWithDescriptor:(MTL4PipelineDescriptor*)descriptor compilerTaskOptions:(MTL4CompilerTaskOptions*)options completionHandler:(MTLNewRenderPipelineStateCompletionHandler)completion
{
	if ([NSThread isMainThread]) _mainThreadLaunch = true;
	return [_native newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:options completionHandler:^(id<MTLRenderPipelineState> result, NSError* error)
		{ [self deliver:^{ completion(result, error); }]; }];
}
@end

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refuses(F&& operation, const char* message)
{
	try { operation(); } catch (const std::exception&) { return; }
	throw std::runtime_error(message);
}
const std::string source = R"(
#include <metal_stdlib>
using namespace metal;
kernel void queued_compute(device uint* out [[buffer(0)]]) { out[0]=37; }
kernel void changed_compute(device uint* out [[buffer(0)]]) { out[0]=99; }
vertex float4 queued_vertex(uint i [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[i],0,1); }
fragment float4 queued_fragment() { return float4(.25,.5,.75,1); }
fragment float4 changed_fragment() { return float4(1,0,0,1); }
)";
id<MTL4Compiler> native(context& ctx)
{
	NSError* error = nil;
	const auto result = [ctx.device() newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
	if (!result) throw std::runtime_error(error.localizedDescription.UTF8String);
	return result;
}

void bounded_and_deduplicated(context& ctx)
{
	const auto gate = [[M4GatedCompiler alloc] initWithNative:native(ctx)];
	compiler_queue queue((id<MTL4Compiler>)gate, {2, 4, 8 * source.size()});
	const auto first = queue.library(source);
	require(first.has_value(), "First native compiler job refused");
	std::vector<std::thread> producers;
	std::atomic<unsigned> duplicate_handles = 0;
	for (unsigned i = 0; i < 4; ++i) producers.emplace_back([&]
	{
		for (unsigned n = 0; n < 32; ++n) if (queue.library(source)) ++duplicate_handles;
	});
	for (auto& producer : producers) producer.join();
	const auto second = queue.library(source + "\n//second");
	std::string caller = source + "\n//third";
	const auto third = queue.library(caller), fourth = queue.library(source + "\n//fourth");
	caller.assign(caller.size(), '\xff');
	require(second && third && fourth && !queue.library(source + "\n//overflow"), "Native compiler job capacity was not bounded");
	[gate waitForCallbacks:2];
	const auto stats = queue.statistics();
	require(stats.active == 2 && stats.queued == 2 && stats.admitted == 4 && stats.deduplicated == 128 && duplicate_handles == 128 &&
		stats.source_bytes <= 8 * source.size() && !first->ready(), "Single-flight native jobs lost bounds/owned bytes");
	refuses([&] { (void)first->library(std::chrono::milliseconds(1)); }, "Native compiler timeout released unfinished work");
	refuses([&] { (void)first->compute(std::chrono::milliseconds(0)); }, "Wrong compiler result kind accepted");
	[gate releaseCallbacks];
	const auto library = first->library();
	require(second->library() && third->library() && fourth->library() && queue.shutdown(), "Owned native jobs failed after caller mutation");
	const auto done = queue.statistics();
	require(done.completed == 4 && !done.failed && !done.cancelled && !done.active && !done.queued && !done.source_bytes &&
		done.peak_active == 2 && ![gate mainThreadLaunch] && [library.functionNames containsObject:@"queued_compute"],
		"Native async results/worker completion counters differ");
	std::cout << "128 concurrent exact-source duplicates:four native jobs,two active/two queued,owned UTF8 source,capacity/timeout/kind guards andworker-only launch passed\n";
}

void frozen_pipeline_inputs(context& ctx)
{
	compiler ordinary(ctx);
	const auto library = ordinary.library(source);
	const auto gate = [[M4GatedCompiler alloc] initWithNative:native(ctx)];
	compiler_queue queue((id<MTL4Compiler>)gate, {1, 3, 1 << 20});
	const auto blocker = queue.library(source);
	[gate waitForCallbacks:1];
	auto compute = [MTL4ComputePipelineDescriptor new];
	compute.computeFunctionDescriptor = library_function(library, @"queued_compute");
	const auto compute_job = queue.compute(compute);
	auto render = [MTL4RenderPipelineDescriptor new];
	render.vertexFunctionDescriptor = library_function(library, @"queued_vertex");
	render.fragmentFunctionDescriptor = library_function(library, @"queued_fragment");
	render.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	render.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	const auto render_job = queue.render(render);
	require(compute_job && render_job && queue.statistics().queued == 2, "Pipeline descriptors were not queued behind native blocker");
	compute.computeFunctionDescriptor = library_function(library, @"changed_compute");
	render.fragmentFunctionDescriptor = library_function(library, @"changed_fragment");
	render.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
	[gate releaseCallbacks];
	(void)blocker->library();
	const auto cp = compute_job->compute();
	const auto rp = render_job->render();
	require(queue.shutdown(), "Native pipeline queue did not drain");
	const auto out = make_buffer(ctx, 4, buffer_storage::shared), pixels = make_buffer(ctx, 1024, buffer_storage::shared);
	auto description = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	description.storageMode = MTLStorageModePrivate;
	description.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
	const auto target = make_texture(ctx, description);
	ctx.begin();
	bindings buffers; buffers.set_buffer(0, out);
	const auto table = buffers.snapshot(ctx);
	const auto encoder = ctx.compute();
	ctx.keep_alive(cp);
	[encoder setComputePipelineState:cp]; [encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	auto pass = [MTL4RenderPassDescriptor new];
	pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	render_commands draw(ctx, pass);
	raster_state raster; raster.viewport = {0, 0, 4, 4, 0, 1}; raster.scissor = {0, 0, 4, 4};
	draw.configure(rp, nil, raster, {}, {}); draw.draw(MTLPrimitiveTypeTriangle, 0, 3); draw.end();
	texture_region region; region.size = MTLSizeMake(4, 4, 1);
	texture_to_buffer(ctx, target, region, pixels, 0, 256);
	ctx.submit().wait();
	require(*static_cast<unsigned*>(out.contents) == 37, "Queued compute borrowed mutated function descriptor");
	const auto* bytes = static_cast<const unsigned char*>(pixels.contents);
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x)
		for (unsigned c = 0; c < 4; ++c) require(bytes[y * 256 + x * 4 + c] == std::array{64u, 128u, 191u, 255u}[c], "Queued render borrowed mutated function/attachment descriptor");
	std::cout << "Queued native compute/render descriptors survive caller function/write-mask changes:GPU37 and16 exact RGBA pixels passed\n";
}

void source_quota_and_context_lifetime()
{
	M4GatedCompiler* gate;
	std::optional<compiler_job> retained, cancelled;
	{
		auto ctx = std::make_unique<context>(1);
		gate = [[M4GatedCompiler alloc] initWithNative:native(*ctx)];
		auto queue = std::make_unique<compiler_queue>((id<MTL4Compiler>)gate, compiler_queue_limits{1, 8, 2 * source.size() + 32});
		retained = queue->library(source);
		[gate waitForCallbacks:1];
		cancelled = queue->library(source + "\n//second");
		require(cancelled && !queue->library(source + "\n//third") && queue->library(source),
			"MSL-byte quota did not bound queued inputs independently of the job count or refused deduplication");
		const auto stats = queue->statistics();
		require(stats.active == 1 && stats.queued == 1 && stats.source_bytes == 2 * source.size() + 9,
			"Queued native MSL payload accounting differs");
		ctx.reset();
		queue.reset();
	}
	require(cancelled->ready() && cancelled->cancelled() && !retained->ready(), "Owner/context destruction lost queued/running separation");
	[gate releaseCallbacks];
	const auto library = retained->library();
	context followup(1);
	compiler compile(followup);
	auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(library, @"queued_compute");
	require(compile.compute(descriptor) != nil, "Native library from destroyed owner/context is no longer usable");
	std::cout << "Independent MSL-byte quota/exact pending payload;duplicate at full quota;late real library survives queue/context destruction andbuilds native compute state passed\n";
}

void shutdown_and_failure(context& ctx)
{
	const auto gate = [[M4GatedCompiler alloc] initWithNative:native(ctx)];
	std::optional<compiler_job> first, cancelled;
	{
		auto queue = std::make_unique<compiler_queue>((id<MTL4Compiler>)gate, compiler_queue_limits{1, 2, 1 << 20});
		first = queue->library(source);
		[gate waitForCallbacks:1];
		cancelled = queue->library(source + "\n//queued");
		std::promise<void> entered;
		std::atomic<bool> closed_woke_producer = false;
		std::thread waiter([&]
		{
			entered.set_value();
			try { (void)queue->library(source + "\n//waiting", nil, std::chrono::seconds(5)); }
			catch (const std::logic_error&) { closed_woke_producer = true; }
		});
		entered.get_future().wait();
		for (unsigned i = 0; i < 1000 && !queue->statistics().waiting; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		require(queue->statistics().waiting == 1, "Producer did not enter bounded compiler admission wait");
		require(!queue->shutdown(std::chrono::milliseconds(1)), "Shutdown joined an unfinished native callback");
		waiter.join();
		require(closed_woke_producer && cancelled->ready() && cancelled->cancelled() && !first->ready(), "Shutdown lost queued cancellation or admitted work after closing");
		refuses([&] { (void)cancelled->library(); }, "Cancelled queued native library returned a result");
		refuses([&] { (void)queue->library(source); }, "Closed native queue accepted work");
		queue.reset(); // A late real native callback still owns its completion.
	}
	[gate releaseCallbacks];
	require(first->library() != nil, "Native task/result died with its queue owner");
	compiler_queue queue(native(ctx), {1, 2, 4096});
	for (unsigned i = 0; i < 2; ++i)
	{
		const auto failed = queue.library("#include <metal_stdlib>\nthis is invalid MSL;");
		refuses([&] { (void)failed->library(); }, "Invalid native MSL compilation succeeded");
	}
	const auto good = queue.library(source);
	require(good->library() && queue.shutdown() && queue.statistics().failed == 2 && queue.statistics().completed == 1,
		"Native compile errors poisoned following jobs or failed to release quotas");
	refuses([&] { compiler_queue invalid(nil); }, "Nil native compiler accepted");
	refuses([&] { compiler_queue invalid(native(ctx), {0, 2, 4096}); }, "Zero worker limit accepted");
	compiler_queue invalid_inputs(native(ctx), {1, 2, 4096});
	refuses([&] { (void)invalid_inputs.library(""); }, "Empty source accepted");
	refuses([&] { (void)invalid_inputs.library(std::string(4097, 'x')); }, "Oversized source accepted");
	refuses([&] { (void)invalid_inputs.library(std::string(1, '\xff')); }, "Invalid UTF8 accepted");
	refuses([&] { (void)invalid_inputs.compute(nil); }, "Nil compute descriptor accepted");
	refuses([&] { (void)invalid_inputs.render(nil); }, "Nil render descriptor accepted");
	const auto library = first->library();
	auto missing = [MTL4ComputePipelineDescriptor new];
	missing.computeFunctionDescriptor = library_function(library, @"absent_entry");
	refuses([&] { (void)invalid_inputs.compute(missing); }, "Missing native function entered driver compilation");
	require(!invalid_inputs.statistics().admitted, "Invalid inputs consumed native quota");
	std::cout << "Shutdown cancels queued jobs/wakes blocked admission;running callback survives owner;two real native errors recover quotas;invalid source/descriptors/limits refused passed\n";
}

void prepared_library_contracts(context& ctx)
{
	compiler compile(ctx, 2), foreign(ctx);
	const auto wait_ready = [](const auto& ticket)
	{
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (!ticket.ready())
		{
			if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Prepared native library did not complete");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	};
	std::string caller = source;
	const auto prepared = compile.prewarm_library(caller);
	require(prepared.has_value(), "Prepared native library refused empty queue");
	caller.assign(caller.size(), '\xff');
	wait_ready(*prepared);
	for (unsigned i = 0; i < 100; ++i) require(compile.prewarm_library(source).has_value(), "Ready unconsumed library deduplication refused");
	require(compile.tasks().statistics().admitted == 1 && !compile.library_statistics().created,
		"Ready unconsumed library was compiled/adopted again");
	const auto library = compile.library(source);
	require(library && compile.library(*prepared, std::chrono::milliseconds(0)) == library &&
		compile.tasks().statistics().admitted == 1 && compile.library_statistics().created == 1,
		"Ordinary library getter failed to adopt exact pending source");
	refuses([&] { (void)foreign.library(*prepared); }, "Foreign compiler consumed prepared source");
	const auto old = compile.prewarm_library(source + "\n//clear epoch");
	require(old.has_value(), "Prepared clear-epoch fixture refused");
	compile.clear_libraries();
	compiler copy = compile;
	wait_ready(*old);
	require(copy.library(*old, std::chrono::milliseconds(0)) && !compile.library_statistics().entries,
		"Prepared source owner was invalidated or repopulated a cleared cache");
	std::vector<compiler::prepared_library> failures;
	for (unsigned i = 0; i < 2; ++i)
	{
		const auto before = compile.tasks().statistics().admitted;
		const auto failed = compile.prewarm_library("#include <metal_stdlib>\ninvalid prepared source;");
		require(failed.has_value(), "Prepared native failure fixture refused");
		refuses([&] { (void)compile.library(*failed); }, "Invalid prepared MSL returned a library");
		require(failed->failed() && compile.tasks().statistics().admitted == before + 1,
			"Failed prepared source was cached as a successful request or poisoned retry");
		failures.push_back(*failed);
	}
	compile.clear_libraries();
	std::vector<compiler::prepared_library> pending;
	for (unsigned i = 0; i < 64; ++i)
	{
		const auto text = source + "\n//bounded prepared source " + std::to_string(i);
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		for (;;)
		{
			auto ticket = compile.prewarm_library(text);
			if (ticket) { pending.push_back(*ticket); break; }
			if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Native library prewarm could not admit64 source owners");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	require(!compile.prewarm_library(source + "\n//overflow source owner"), "Prepared native source index exceeded64 owners");
	const auto before = compile.tasks().statistics().admitted;
	for (const auto& ticket : pending) require(compile.library(ticket) != nil, "Prepared source failed during bounded adoption");
	require(compile.tasks().statistics().admitted == before && compile.library_statistics().entries == 2,
		"Prepared source adoption recompiled or exceeded completed LRU budget");
	std::cout << "Prepared native libraries:owned caller bytes,100 ready-unconsumed duplicates/one job,ordinary getter adoption,foreign/copy/clear epochs,two failure retries,64-source window/65th refusal andtwo-entry LRU passed\n";
}

void productive_wrapper(context& ctx)
{
	compiler compile(ctx);
	const auto begin = std::chrono::steady_clock::now();
	const auto library = compile.library(source);
	const auto cold = std::chrono::steady_clock::now();
	for (unsigned i = 0; i < 100; ++i) require(compile.library(source) == library, "Warm compiler cache changed native identity");
	const auto warm = std::chrono::steady_clock::now();
	auto descriptor = [MTL4ComputePipelineDescriptor new]; descriptor.computeFunctionDescriptor = library_function(library, @"queued_compute");
	require(compile.compute(descriptor) != nil, "Productive compute wrapper failed native async result");
	const auto stats = compile.tasks().statistics();
	require(stats.admitted == 2 && compile.library_statistics().created == 1 && compile.library_statistics().reused == 100,
		"Productive compiler missed its native async service/cache");
	std::cout << "Productive compiler native async wrappers:one process-library-cache miss_us=" << std::chrono::duration_cast<std::chrono::microseconds>(cold-begin).count()
		<< ";100 exact-source cache hits_us=" << std::chrono::duration_cast<std::chrono::microseconds>(warm-cold).count() << ";no FPS/device-cache-cold claim\n";
}
}
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			context ctx(1);
			bounded_and_deduplicated(ctx);
			frozen_pipeline_inputs(ctx);
			source_quota_and_context_lifetime();
			shutdown_and_failure(ctx);
			prepared_library_contracts(ctx);
			productive_wrapper(ctx);
			return 0;
		}
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
