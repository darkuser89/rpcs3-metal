#include "M4SurfaceCache.h"
#include "M4HostAccessFixture.h"
#include "M4Query.h"
#include "M4VertexFetch.h"
#include "M4FragmentShader.h"
#include "M4ConstantUpload.h"

#include <bit>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refuses(F&& op, const char* message)
{
	try { op(); } catch (const std::exception&) { return; }
	throw std::runtime_error(message);
}
struct fixture
{
	context& device;
	std::shared_ptr<test::host_access_fixture> provider = std::make_shared<test::host_access_fixture>();
	std::shared_ptr<guarded_guest_memory> memory = std::make_shared<guarded_guest_memory>(provider);
	cpu_capture_dispatch capture = [this](const auto& op) { provider->execute([&](auto&) { op(); }); };
	surface_cache cache;
	cached_surface_description description{4096, {4, 4, 1, surface_color::argb8, false}, 0, {}};
	cached_surface_ref target;
	explicit fixture(context& ctx, std::size_t budget = 8 << 20, std::shared_ptr<heap_budget> ledger = {}) : device(ctx), cache(ctx, memory, budget, std::move(ledger), capture)
	{
		memory->write({4096, 64}, std::vector<std::byte>(64));
		target = cache.acquire(description);
		cache.clear(target, 0xf0, 0xff102030, 0, 255, {0, 0, 4, 4});
		cache.flush(target);
	}
	void clear(unsigned word, unsigned x = 0) { cache.clear(target, 0xf0, word, 0, 255, {x, 0, 1, 1}); }
};

void actual_draws(context& ctx)
{
	fixture f(ctx);
	compiler compile(ctx);
	const auto library = compile.library(R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 batch_vertex(uint i [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1); }
fragment float4 batch_fragment(constant float4& color [[buffer(0)]]) { return color; }
)");
	auto descriptor = [MTL4RenderPipelineDescriptor new];
	descriptor.vertexFunctionDescriptor = library_function(library, @"batch_vertex");
	descriptor.fragmentFunctionDescriptor = library_function(library, @"batch_fragment");
	descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	const auto pipeline = compile.render(descriptor);
	query_pool queries(ctx, 4, 16);
	queries.begin(0, true);
	const auto initial = f.cache.status(f.target);
	const auto before = f.cache.transaction_statistics();
	const auto original = f.memory->read({4096, 64}).bytes;
	const std::array uses{cached_surface_use{f.target, true}};
	const auto gate = [ctx.device() newSharedEvent];
	auto batch = f.cache.batch_draws();
	for (unsigned i = 0; i < 32; ++i)
	{
		require(f.cache.acquire(f.description) == f.target, "Same framebuffer lookup flushed/replaced batch target");
		f.cache.draw(uses, [&](context& commands, auto textures)
		{
			if (!i) commands.wait_event(gate, 1);
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = textures[0];
			pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			const auto query = queries.attach(commands, pass);
			const std::array<float, 4> color{float(i + 1) / 255, float(3 * i + 2) / 255, 128.f / 255, 1};
			const auto constant = commands.upload(std::as_bytes(std::span(color)), 16);
			bindings fragment;
			fragment.set_buffer(0, constant.buffer, constant.offset);
			raster_state raster;
			raster.viewport = {0, 0, 4, 4, 0, 1};
			raster.scissor = {i % 4, (i / 4) % 4, 1, 1};
			render_commands render(commands, pass);
			if (query) render.visibility(query->mode, query->offset);
			render.configure(pipeline, nil, raster, {}, fragment);
			render.draw(MTLPrimitiveTypeTriangle, 0, 3);
			render.end();
		});
		const auto state = f.cache.status(f.target);
		require(state.pending && state.dirty == initial.dirty && state.content_generation == initial.content_generation &&
			f.cache.transaction_statistics().submitted == before.submitted,
			"Draw batch waited/submitted or published metadata before GPU completion");
	}
	queries.end(0);
	require(!queries.ready(0) && f.memory->read({4096, 64}).bytes == original, "Unsubmitted draw batch exposed query/guest output");
	const auto encoded = f.cache.transaction_statistics();
	require(encoded.batched_draws == before.batched_draws + 32 && encoded.native_versions == before.native_versions + 1 &&
		encoded.native_version_blits == before.native_version_blits + 1 && encoded.raw_backup_submissions == before.raw_backup_submissions,
		"Same-target draws did not share one private version/preservation blit");
	f.cache.submit_draw_batch();
	f.cache.submit_draw_batch(); // Polling again must not submit the prefix twice.
	require(!queries.ready(0) && f.cache.status(f.target).pending &&
		f.cache.status(f.target).content_generation == initial.content_generation &&
		f.cache.transaction_statistics().submitted == before.submitted + 1,
		"Query submission boundary waited on its gate or published surface metadata");
	gate.signaledValue = 1;
	const auto samples = queries.consume(0); // Must finish without another surface submit.
	require(f.cache.status(f.target).content_generation == initial.content_generation,
		"Query consumption published unvalidated surface metadata");
	batch.finish();
	const auto done = f.cache.transaction_statistics();
	require(done.submitted == before.submitted + 1 && done.draw_batches == before.draw_batches + 1 &&
		f.cache.status(f.target).content_generation == initial.content_generation + 32 && !f.cache.status(f.target).pending,
		"Draw batch did not publish all ordered generations in one submission");
	require(queries.ready(0) && samples == 32 && queries.statistics().initializations == 1 && queries.statistics().reused_batch == 31,
		"Draw batch visibility lost samples or allocated separate per-draw counters");
	f.cache.flush(f.target);
	const auto result = f.memory->read({4096, 64}).bytes;
	for (unsigned pixel = 0; pixel < 16; ++pixel)
	{
		const unsigned i = 16 + pixel;
		require(result[4 * pixel] == std::byte{255} && unsigned(result[4 * pixel + 1]) == i + 1 &&
			unsigned(result[4 * pixel + 2]) == 3 * i + 2 && unsigned(result[4 * pixel + 3]) == 128,
			"Batched native triangle order/scissor/uniform pixels differ");
	}
	std::cout << "32 real Metal4 triangles:one nonblocking gated query submit,one native target version/initial blit,32 delayed generations,one query counter/32samples andexact ordered per-draw pixels passed\n";
}
void capacity_and_boundaries(context& ctx)
{
	fixture f(ctx);
	const auto before = f.cache.transaction_statistics();
	const auto generation = f.cache.status(f.target).content_generation;
	{
		auto batch = f.cache.batch_draws(4);
		for (unsigned i = 0; i < 10; ++i) f.clear(0xff000000 | i, i % 4);
		require(f.cache.transaction_statistics().draw_batches == before.draw_batches + 2 && f.cache.status(f.target).content_generation == generation + 8,
			"Capacity boundary lost completed prefixes or published final unsubmitted draws");
		batch.finish();
	}
	require(f.cache.transaction_statistics().draw_batches == before.draw_batches + 3 && f.cache.status(f.target).content_generation == generation + 10,
		"Bounded draw batches did not publish exactly4/4/2 operations");
	{
		auto batch = f.cache.batch_draws();
		const auto before_read = f.cache.transaction_statistics();
		f.clear(0xff123456);
		const auto data = f.cache.cpu_read({4096, 4}).bytes;
		require(data == std::vector<std::byte>{std::byte{255}, std::byte{0x12}, std::byte{0x34}, std::byte{0x56}} &&
			f.cache.transaction_statistics().draw_batches == before_read.draw_batches + 1, "CPU read did not finish unsubmitted native output");
		f.clear(0xff234567);
		batch.finish();
	}
	{
		const auto alias = f.cache.acquire({4096, {2, 8, 1, surface_color::argb8, false}, 0, {}});
		auto batch = f.cache.batch_draws();
		const auto old_batches = f.cache.transaction_statistics().draw_batches;
		f.clear(0xff345678);
		f.cache.prepare(alias);
		require(f.cache.transaction_statistics().draw_batches == old_batches + 1 && !f.cache.status(f.target).pending,
			"Aliased input preparation missed native draw batch boundary");
		batch.finish();
	}
	refuses([&] { auto invalid = f.cache.batch_draws(0); }, "Zero-capacity batch accepted");
	refuses([&] { auto invalid = f.cache.batch_draws(257); }, "Unbounded draw batch accepted");
	{
		auto batch = f.cache.batch_draws();
		refuses([&] { auto nested = f.cache.batch_draws(); }, "Nested draw batch accepted");
		batch.finish();
		refuses([&] { batch.finish(); }, "Draw batch finished twice");
	}
	std::cout << "Bounded4/4/2 batches,CPU read/alias publication boundaries,same-scope continuation,capacity/nesting/double-finish guards passed\n";
}
void failures_and_timeout(context& ctx)
{
	fixture f(ctx);
	const auto original = f.memory->read({4096, 64}).bytes;
	const auto initial = f.cache.status(f.target);
	{
		auto batch = f.cache.batch_draws();
		f.clear(0xff123456);
		f.clear(0xff234567, 1);
	} // Cancel all unsubmitted encodings,never publish a successful prefix.
	require(!f.cache.status(f.target).pending && f.cache.status(f.target).content_generation == initial.content_generation,
		"Scope cancellation published an unsubmitted prefix");
	{
		auto batch = f.cache.batch_draws();
		f.clear(0xff123456);
		const std::array use{cached_surface_use{f.target, true}};
		refuses([&] { f.cache.draw(use, [](context&, auto) { throw std::runtime_error("Cancel batch encoding"); }); }, "Throwing encoder was accepted");
	}
	require(!f.cache.status(f.target).pending && f.cache.status(f.target).content_generation == initial.content_generation,
		"Throwing encoder failed to roll back complete unsubmitted batch");
	{
		const std::array<unsigned, 1> bad{1};
		const auto error = make_upload(ctx, std::as_bytes(std::span(bad)));
		auto batch = f.cache.batch_draws();
		f.clear(0xff123456);
		const std::array use{cached_surface_use{f.target, true}};
		f.cache.draw(use, [&](context& commands, auto) { commands.require_gpu_success(error, 0, @"Reject entire draw batch"); });
		refuses([&] { batch.finish(); }, "Failed batch GPU status published native generations");
	}
	require(!f.cache.status(f.target).faulted && f.cache.status(f.target).initialized && !f.cache.status(f.target).pending &&
		f.cache.status(f.target).content_generation == initial.content_generation, "Failed GPU batch did not retain original native authority");
	f.cache.recover(f.target);
	{
		const std::array use{cached_surface_use{f.target, true}};
		f.cache.draw(use, [](context&, auto) {});
		f.cache.flush(f.target);
		require(f.memory->read({4096, 64}).bytes == original, "Cancelled/rejected batch altered original native pixels");
	}
	{
		auto batch = f.cache.batch_draws();
		f.clear(0xff123456);
		const auto token = ctx.recording_submission();
		f.memory->write({4096, 1}, std::array{std::byte{0x7a}});
		refuses([&] { batch.finish(); }, "Stale unsubmitted VM inputs reached native queue");
		require(token.cancelled() && !f.cache.status(f.target).pending && f.memory->read({4096, 1}).bytes[0] == std::byte{0x7a},
			"Pre-submit VM conflict changed CPU bytes or left an uncancelled batch");
		f.cache.prepare(f.target);
	}
	{
		const auto gate = [ctx.device() newSharedEvent];
		struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } cleanup{gate};
		f.cache.set_wait_timeout(std::chrono::milliseconds(1));
		auto batch = f.cache.batch_draws();
		const auto before = f.cache.status(f.target).content_generation;
		const std::array use{cached_surface_use{f.target, true}};
		f.cache.draw(use, [&](context& commands, auto) { commands.wait_event(gate, 1); });
		f.clear(0xff456789);
		unsigned allocations = 0;
		refuses([&] { f.cache.prepare_draw_resources([&] { ++allocations; throw native_allocation_error(native_allocation_kind::buffer, 4096); }); },
			"Admission waited through an unsignalled prefix or retried after timeout");
		require(allocations == 1 && f.cache.status(f.target).pending && f.cache.status(f.target).content_generation == before,
			"Admission timeout retried allocator or published unfinished prefix");
		refuses([&] { batch.finish(); }, "Unsignalled GPU batch did not retain pending completion");
		require(f.cache.status(f.target).pending && f.cache.status(f.target).content_generation == before, "Timeout retired/published unfinished draw batch");
		gate.signaledValue = 1;
		f.cache.set_wait_timeout(std::chrono::seconds(30));
		f.cache.synchronize();
		require(!f.cache.status(f.target).pending && f.cache.status(f.target).content_generation == before + 2,
			"Completed timed-out batch failed to publish both draws once");
	}
	std::cout << "Unsubmitted scope/encoder cancellation,whole-batch GPU rejection,pre-submit VM conflict andgated timeout/retry retain old native pixels andpublish exact generations passed\n";
}
void submitted_physical_ownership(context& ctx)
{
	for (const bool cpu_race : {false, true})
	{
		fixture f(ctx);
		f.provider->pin_first_alias();
		const auto before = f.cache.status(f.target);
		const auto gate = [ctx.device() newSharedEvent];
		struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } cleanup{gate};
		f.cache.set_wait_timeout(std::chrono::milliseconds(1));
		auto batch = f.cache.batch_draws();
		const std::array use{cached_surface_use{f.target, true}};
		f.cache.draw(use, [&](context& commands, auto) { commands.wait_event(gate, 1); });
		f.clear(0xff789abc);
		refuses([&] { batch.finish(); }, "Physical batch did not remain GPU-gated");
		f.provider->execute([&](auto&)
		{
			f.memory->begin_mapping_change({0, 65536});
			f.provider->replace_first();
			f.memory->end_mapping_change({0, 65536});
		});
		f.memory->write({4096, 64}, std::vector<std::byte>(64, std::byte{0x7a}));
		if (cpu_race) f.memory->write({0x41000, 1}, std::array{std::byte{0x6b}});
		gate.signaledValue = 1;
		f.cache.set_wait_timeout(std::chrono::seconds(30));
		if (cpu_race)
		{
			refuses([&] { f.cache.synchronize(); }, "Stale retained physical input published batch output");
			require(f.cache.status(f.target).content_generation == before.content_generation && !f.cache.status(f.target).faulted &&
				f.memory->read({0x41000, 1}).bytes[0] == std::byte{0x6b}, "Physical conflict lost old native metadata or CPU bytes");
		}
		else
		{
			f.cache.synchronize();
			require(f.cache.status(f.target).content_generation == before.content_generation + 2 && f.cache.status(f.target).dirty,
				"Successful old-physical batch was discarded after virtual remap");
			f.cache.flush(f.target);
			require(f.memory->read({0x41000, 4}).bytes == std::vector<std::byte>{std::byte{255}, std::byte{0x78}, std::byte{0x9a}, std::byte{0xbc}},
				"Batch wrote through replacement VA instead of retained original physical allocation");
		}
		require(f.memory->read({4096, 64}).bytes == std::vector<std::byte>(64, std::byte{0x7a}), "Old batch modified replacement allocation bytes");
	}
	std::cout << "Submitted native batches retain original physical ownership across VA replacement;new allocation untouched;old-allocation CPU race refuses whole batch without losing CPU bytes passed\n";
}

void owner_lifetime(context& ctx)
{
	auto f = std::make_unique<fixture>(ctx);
	const auto memory = f->memory;
	const auto original = memory->read({4096, 64}).bytes;
	auto batch = f->cache.batch_draws();
	f->clear(0xffabcdef);
	const auto token = ctx.recording_submission();
	f.reset();
	require(token.cancelled() && memory->read({4096, 64}).bytes == original, "Destroyed batch owner submitted unfinished output");
	refuses([&] { batch.finish(); }, "Batch handle accessed destroyed surface owner");
	std::cout << "Draw batch handle survives destroyed owner without dereference;unsubmitted GPU work cancelled andguest bytes retained passed\n";
}

void resource_admission()
{
	context ctx(1);
	auto ledger = std::make_shared<heap_budget>(ctx, 16 << 20);
	ctx.set_buffer_budget(ledger);
	fixture f(ctx, 16 << 20, ledger);
	vertex_fetch_engine fetch(ctx);
	const std::array uses{cached_surface_use{f.target, true}};
	const auto before = f.cache.transaction_statistics();
	const auto generation = f.cache.status(f.target).content_generation;
	auto batch = f.cache.batch_draws();
	@autoreleasepool
	{
		const auto prefix_only = make_buffer(ctx, 4 << 20, buffer_storage::private_gpu);
		f.cache.draw(uses, [&](context& commands, auto) { commands.use_resource(prefix_only); });
	} // Drain temporary native returns; only completion now owns these four MiB.
	const auto occupied = ledger->statistics();
	// Keep device backing-rounding headroom, still far below the next output.
	id<MTLBuffer> blocker = ledger->shared_buffer(occupied.budget - occupied.reserved_bytes - (64 << 10));
	vertex_fetch_request request;
	request.vertex_count = 8192; // Two-MiB float4[16] output, no guest input reads.
	request.attribute_mask = 0;
	std::optional<vertex_fetch_engine::prepared_fetch> prepared;
	unsigned attempts = 0;
	f.cache.prepare_draw_resources([&]
	{
		++attempts;
		auto candidate = fetch.prepare(ctx, request, nil, nil);
		prepared = std::move(candidate);
	});
	require(attempts == 2 && f.cache.transaction_statistics().draw_resource_retries == before.draw_resource_retries + 1 &&
		f.cache.transaction_statistics().draw_batches == before.draw_batches + 1 &&
		f.cache.status(f.target).content_generation == generation + 1 && !f.cache.status(f.target).pending,
		"Real buffer pressure did not finish exactly one successful prefix before retrying fetch admission");
	blocker = nil;
	const auto readback = make_buffer(ctx, 2 << 20, buffer_storage::shared);
	request.vertex_count = 1; // The admitted request and output extent are immutable.
	f.cache.draw(uses, [&](context& commands, auto)
	{
		const std::array<std::byte, 16> zeros{};
		(void)commands.upload(zeros, 16); // Ensure the small encoding arena already exists.
		const auto allocations = ledger->statistics().created;
		const auto output = fetch.decode(commands, *prepared);
		require(output.length == 2 << 20 && ledger->statistics().created == allocations,
			"Prepared vertex decode allocated another native buffer or borrowed mutable request state");
		copy_buffer(commands, output, 0, readback, 0, readback.length);
	});
	prepared.reset(); // Encoded resources must survive loss of the admission owner.
	batch.finish();
	const auto* words = static_cast<const unsigned*>(readback.contents);
	for (std::size_t i = 0; i < readback.length / sizeof(unsigned); ++i)
		require(words[i] == (i % 4 == 3 ? 0x3f800000u : 0u), "Admitted native vertex fetch output differs");
	require(f.cache.status(f.target).content_generation == generation + 2,
		"Draw following admission retry lost its ordered publication");
	{
		auto next = f.cache.batch_draws();
		f.clear(0xff456789);
		__weak id<MTLBuffer> partial = nil;
		unsigned partial_attempts = 0;
		f.cache.prepare_draw_resources([&]
		{
			if (!partial_attempts++)
			{
				const auto candidate = make_buffer(ctx, 4096, buffer_storage::private_gpu);
				partial = candidate;
				throw native_allocation_error(native_allocation_kind::buffer, 4096);
			}
			require(partial == nil, "Partial native allocation survived admission unwind/autorelease drain");
		});
		require(partial_attempts == 2, "Partial admission allocation did not retry exactly once");
		next.finish();
	}
	std::cout << "Real16-MiB ledger pressure:completed four-MiB prefix releases room for two-MiB admitted fetch;exact524288 GPU words,immutable request andno decode-time buffer allocation passed\n";
}

void upload_admission()
{
	{
		context compact(1);
		auto budget = std::make_shared<heap_budget>(compact, (4 << 20) + (64 << 10));
		compact.set_buffer_budget(budget);
		const std::array<std::byte, 16> bytes{};
		@autoreleasepool { compact.begin(); (void)compact.upload(bytes); compact.submit().wait(); }
		const std::array large_request{upload_requirement{4 << 20, 256}};
		const auto large_plan = compact.prepare_uploads(large_request);
		require(budget->statistics().live_buffers == 1 && budget->statistics().reserved_bytes <= budget->statistics().budget,
			"Unused completed small arena stayed pinned while admitting a larger page without a prefix");
		compact.begin();
		const std::vector<std::byte> payload(4 << 20, std::byte{0x5a});
		compact.with_uploads(large_plan, [&] { (void)compact.upload(payload); });
		compact.submit().wait();
	}
	context ctx(2);
	auto ledger = std::make_shared<heap_budget>(ctx, 16 << 20);
	ctx.set_buffer_budget(ledger);
	const auto readback = make_buffer(ctx, 128, buffer_storage::shared);
	const std::array requirements{upload_requirement{16, 16}, upload_requirement{1 << 20, 256}, upload_requirement{36, 16}};
	const auto plan = ctx.prepare_uploads(requirements);
	refuses([&] { (void)ctx.prepare_uploads(requirements); }, "Two outstanding plans could reserve the same upload bytes");
	const auto allocated = ledger->statistics().created;
	(void)ctx.trim_upload_pages();
	const std::array<std::byte, 16> small{std::byte{0x31}};
	const std::vector<std::byte> large(1 << 20, std::byte{0x52}), tail(36, std::byte{0x73});
	ctx.begin();
	ctx.with_uploads(plan, [&]
	{
		refuses([&] { (void)ctx.submit(); }, "Submit inside admitted uploads accepted");
		refuses([&] { ctx.cancel(); }, "Cancel inside admitted uploads accepted");
		refuses([&] { ctx.with_uploads(plan, [] {}); }, "Nested admitted uploads accepted");
		const auto a = ctx.upload(small, 16), b = ctx.upload(large), c = ctx.upload(tail, 16);
		copy_buffer(ctx, a.buffer, a.offset, readback, 0, 16);
		copy_buffer(ctx, b.buffer, b.offset + large.size() - 16, readback, 16, 16);
		copy_buffer(ctx, c.buffer, c.offset, readback, 32, tail.size());
	});
	require(ledger->statistics().created == allocated, "Admitted encoding allocated a native upload buffer");
	ctx.submit().wait();
	const auto* result = static_cast<const std::byte*>(readback.contents);
	for (unsigned i = 0; i < 16; ++i) require(result[i] == small[i], "Small admitted GPU upload differs");
	for (unsigned i = 16; i < 32; ++i) require(result[i] == std::byte{0x52}, "Large admitted GPU upload differs");
	for (unsigned i = 32; i < 68; ++i) require(result[i] == std::byte{0x73}, "Tail admitted GPU upload differs");
	ctx.begin();
	refuses([&] { ctx.with_uploads(plan, [] {}); }, "Single-use upload reservation reused");
	ctx.cancel();
	// The next slot has no cache yet. Complete it, then return to the original
	// slot to prove that its native arena backing is reused after completion.
	ctx.begin(); ctx.submit().wait();
	const auto reused = ctx.prepare_uploads(requirements);
	require(ledger->statistics().created == allocated, "Completed upload arenas were not reused by admission");
	ctx.begin();
	ctx.with_uploads(reused, [&] { (void)ctx.upload(small, 16); (void)ctx.upload(large); (void)ctx.upload(tail, 16); });
	ctx.cancel();
	const auto before_invalid = ledger->statistics().created;
	refuses([&] { (void)ctx.prepare_uploads({}); }, "Empty upload plan accepted");
	for (const auto request : {upload_requirement{0, 16}, upload_requirement{1, 3}, upload_requirement{ctx.device().maxBufferLength + 1, 16}})
		refuses([&] { (void)ctx.prepare_uploads(std::span{&request, 1}); }, "Invalid upload reservation accepted");
	require(ledger->statistics().created == before_invalid, "Invalid upload plan allocated storage");
	const std::array one{upload_requirement{16, 16}};
	const auto incomplete = ctx.prepare_uploads(one);
	ctx.begin();
	refuses([&] { ctx.with_uploads(incomplete, [] {}); }, "Unconsumed upload reservation accepted");
	ctx.cancel();
	const auto mismatch = ctx.prepare_uploads(one);
	ctx.begin();
	refuses([&] { ctx.with_uploads(mismatch, [&] { (void)ctx.upload(small, 256); }); }, "Mismatched upload alignment accepted");
	(void)ctx.upload(small, 16); // Failure must leave ordinary encoding usable.
	ctx.cancel();
	ctx.begin();
	const auto protected_plan = ctx.prepare_uploads(one);
	context other(1);
	other.begin();
	refuses([&] { other.with_uploads(protected_plan, [] {}); }, "Foreign context accepted upload reservation");
	other.cancel();
	const std::array<std::byte, 16> prerequisite{std::byte{0x8c}};
	const auto early = ctx.upload(prerequisite, 16);
	ctx.with_uploads(protected_plan, [&]
	{
		const auto admitted = ctx.upload(small, 16);
		require(early.buffer != admitted.buffer || early.offset + early.size <= admitted.offset ||
			admitted.offset + admitted.size <= early.offset, "Prerequisite upload overlapped reserved bytes");
		copy_buffer(ctx, early.buffer, early.offset, readback, 80, prerequisite.size());
		copy_buffer(ctx, admitted.buffer, admitted.offset, readback, 96, small.size());
	});
	ctx.submit().wait();
	require(std::equal(prerequisite.begin(), prerequisite.end(), result + 80) &&
		std::equal(small.begin(), small.end(), result + 96), "Prerequisite or admitted upload changed on GPU readback");
	// A reservation from a recording may only cross its own completed boundary.
	ctx.begin();
	(void)ctx.upload(small, 16);
	const auto crossing = ctx.prepare_uploads(one);
	const auto gate = [ctx.device() newSharedEvent];
	ctx.wait_event(gate, 1);
	const auto prefix = ctx.submit();
	ctx.begin();
	refuses([&] { ctx.with_uploads(crossing, [] {}); }, "Unfinished foreign slot borrowed admitted arena storage");
	ctx.cancel();
	gate.signaledValue = 1;
	prefix.wait();
	ctx.begin();
	id<MTLBuffer> transferred = nil;
	ctx.with_uploads(crossing, [&]
	{
		const auto slice = ctx.upload(small, 16);
		transferred = slice.buffer;
		copy_buffer(ctx, slice.buffer, slice.offset, readback, 0, small.size());
	});
	const auto second_gate = [ctx.device() newSharedEvent];
	ctx.wait_event(second_gate, 1);
	const auto second = ctx.submit();
	ctx.begin();
	const auto independent = ctx.upload(small, 16);
	require(independent.buffer != transferred, "Former slot retained transferred arena backing while the new GPU owner was pending");
	ctx.cancel();
	second_gate.signaledValue = 1;
	second.wait();
	for (unsigned i = 0; i < small.size(); ++i) require(result[i] == small[i], "Former slot reuse overwrote transferred pending upload bytes");
	std::cout << "Upload admission:three native arena pages,GPU byte copies,no encoding allocations,completed-page reuse,trim retention andgated cross-slot completion;protected prerequisite bytes and invalid/foreign/nesting/sequence/reuse/submit/cancel guards passed\n";
}

void admission_failure_boundaries(context& ctx)
{
	fixture f(ctx);
	const auto generation = f.cache.status(f.target).content_generation;
	auto batch = f.cache.batch_draws();
	f.clear(0xff123456);
	const auto before = f.cache.transaction_statistics();
	unsigned attempts = 0;
	refuses([&] { f.cache.prepare_draw_resources({}); }, "Empty draw admission callback accepted");
	refuses([&] { f.cache.prepare_draw_resources([&] { ++attempts; throw std::invalid_argument("Invalid admission"); }); }, "Invalid admission retried");
	require(attempts == 1, "Untyped allocation error retried");
	attempts = 0;
	refuses([&] { f.cache.prepare_draw_resources([&] { ++attempts; throw heap_budget_error(4097, 0, 4096); }); }, "Oversized admission retried");
	require(attempts == 1 && f.cache.transaction_statistics().submitted == before.submitted,
		"Invalid/oversized admission submitted an unrelated prefix");
	f.provider->execute([&](auto&)
	{
		refuses([&] { f.cache.prepare_draw_resources([&] { ++attempts; }); }, "Admission ran inside VM exclusion");
		refuses([&] { f.cache.submit_draw_batch(); }, "Report submission ran inside VM exclusion");
	});
	require(attempts == 1 && f.cache.transaction_statistics().submitted == before.submitted,
		"VM refusal executed allocator or submitted prefix");
	attempts = 0;
	refuses([&] { f.cache.prepare_draw_resources([&] { ++attempts; throw native_allocation_error(native_allocation_kind::buffer, 4096); }); }, "Persistent allocation failure accepted");
	require(attempts == 2 && f.cache.transaction_statistics().submitted == before.submitted + 1 &&
		f.cache.status(f.target).content_generation == generation + 1,
		"Persistent admission did not preserve the completed prefix with exactly one retry");
	f.clear(0xff234567);
	batch.finish();
	for (const auto kind : {native_allocation_kind::heap, native_allocation_kind::texture, native_allocation_kind::buffer})
	{
		auto next = f.cache.batch_draws();
		f.clear(0xff345678);
		attempts = 0;
		f.cache.prepare_draw_resources([&] { if (!attempts++) throw native_allocation_error(kind, 4096); });
		require(attempts == 2, "Typed allocation failure failed one prefix retry");
		next.finish();
	}
	attempts = 0;
	refuses([&] { f.cache.prepare_draw_resources([&] { ++attempts; throw native_allocation_error(native_allocation_kind::buffer, 4096); }); }, "No-prefix native failure accepted");
	require(attempts == 1, "Admission retried without any prefix to retire");
	std::cout << "Admission:invalid/oversized/VM refusals without submit;heap/texture/buffer one retry;persistent/no-prefix failure bounds andcontinued ordered draws passed\n";
}

void admission_gpu_failure(context& ctx)
{
	fixture f(ctx);
	const auto initial = f.cache.status(f.target);
	const std::array bad_status{1u};
	const auto bad = make_upload(ctx, std::as_bytes(std::span(bad_status)));
	const std::array uses{cached_surface_use{f.target, true}};
	auto batch = f.cache.batch_draws();
	f.clear(0xffabcdef);
	f.cache.draw(uses, [&](context& commands, auto) { commands.require_gpu_success(bad, 0, @"Admission prefix rejected"); });
	unsigned attempts = 0;
	refuses([&] { f.cache.prepare_draw_resources([&] { ++attempts; throw native_allocation_error(native_allocation_kind::buffer, 4096); }); }, "Failed GPU prefix allowed admission retry");
	require(attempts == 1 && f.cache.status(f.target).content_generation == initial.content_generation &&
		!f.cache.status(f.target).pending && !f.cache.status(f.target).faulted,
		"Failed GPU prefix retried allocator or published failed native output");
	f.cache.recover(f.target);
	batch.finish();
	std::cout << "GPU-failed admission prefix rolls back entire output andrefuses allocation retry passed\n";
}

void admitted_fragment_constants(context& ctx)
{
	fixture f(ctx);
	compiler compile(ctx);
	std::vector<std::byte> code;
	const auto append = [&](unsigned word)
	{
		for (unsigned shift : {8u, 0u, 24u, 16u}) code.push_back(std::byte(word >> shift));
	};
	for (auto word : {(1u << 24) | (15u << 9) | 1u,
		(0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9}) append(word);
	for (float value : {0.25f, 0.5f, 0.75f, 1.f}) append(std::bit_cast<unsigned>(value));
	fragment_shader shader(compile, code);
	auto constants = shader.prepare_constants(ctx, code);
	fragment_shader other(compile, code);
	ctx.begin();
	refuses([&] { (void)other.prepare(ctx, {}, constants, {}); }, "Foreign shader accepted prepared constants");
	ctx.cancel();
	code.assign(code.size(), std::byte{0xff});
	ctx.constant_uploads().clear();
	const auto stats = ctx.constant_uploads().statistics();
	const auto library = compile.library(R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 admitted_vertex(uint i [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1); }
)");
	auto descriptor = [MTL4RenderPipelineDescriptor new];
	descriptor.vertexFunctionDescriptor = library_function(library, @"admitted_vertex");
	descriptor.fragmentFunctionDescriptor = shader.function();
	descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	const auto pipeline = compile.render(descriptor);
	const std::array uses{cached_surface_use{f.target, true}};
	f.cache.draw(uses, [&](context& commands, auto textures)
	{
		const auto inputs = shader.prepare(commands, {}, constants, {});
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = textures[0];
		pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		render_commands render(commands, pass);
		raster_state raster;
		raster.viewport = {0, 0, 4, 4, 0, 1}; raster.scissor = {0, 0, 4, 4};
		render.configure(pipeline, nil, raster, {}, inputs);
		render.draw(MTLPrimitiveTypeTriangle, 0, 3);
		render.end();
	});
	require(ctx.constant_uploads().statistics().created == stats.created &&
		ctx.constant_uploads().statistics().instruction_checks == stats.instruction_checks,
		"Admitted fragment constants repeated layout validation or native upload while encoding");
	f.cache.flush(f.target);
	const auto bytes = f.memory->read({4096, 64}).bytes;
	for (unsigned pixel = 0; pixel < 16; ++pixel)
		require(bytes[4 * pixel] == std::byte{255} && bytes[4 * pixel + 1] == std::byte{64} &&
			bytes[4 * pixel + 2] == std::byte{128} && bytes[4 * pixel + 3] == std::byte{191},
			"Owned fragment constant version lost literal values after caller mutation/cache eviction");
	std::cout << "Prepared RSX fragment literals:shader identity checked;host snapshot destroyed/cache evicted;no encoding-time cache lookup/upload;exact16 pixels passed\n";
}

void mirror_admission(context& ctx)
{
	const surface_allocation_description depth_description{4, 4, 1, surface_depth::z24s8_uint, false};
	std::size_t budget;
	{
		surface_pool probe(ctx, 1 << 20);
		budget = probe.acquire({4, 4, 1, surface_color::argb8, false})->reserved_bytes() +
			2 * probe.acquire(depth_description)->reserved_bytes() + 2 * probe.acquire({4, 4, 1, surface_color::r32f, false})->reserved_bytes();
	}
	fixture f(ctx, budget);
	const auto depth = f.cache.acquire({8192, depth_description, 0, {}});
	f.cache.clear(depth, 3, 0, 0xffffff42, 255, {0, 0, 4, 4});
	f.cache.flush(depth);
	const auto original = f.memory->read({8192, 64}).bytes;
	const auto generation = f.cache.status(depth).content_generation;
	const auto before = f.cache.transaction_statistics();
	const std::array uses{cached_surface_use{depth, true}};
	auto batch = f.cache.batch_draws();
	for (unsigned i = 0; i < 10; ++i)
		f.cache.draw_depth_bounds(uses, 0, [](auto textures)
		{
			auto pass = [MTL4RenderPassDescriptor new];
			pass.depthAttachment.texture = textures[0];
			pass.depthAttachment.loadAction = MTLLoadActionLoad;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
			pass.stencilAttachment.texture = textures[0];
			pass.stencilAttachment.loadAction = MTLLoadActionLoad;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
			return pass;
		}, [](context&, auto, depth_bounds_pass&) {});
	batch.finish();
	const auto after = f.cache.transaction_statistics();
	require(after.draw_resource_retries > before.draw_resource_retries && after.draw_batches > before.draw_batches + 1 &&
		f.cache.status(depth).content_generation == generation + 10 && f.cache.memory_statistics().reserved_bytes <= budget,
		"Bounded mirror quota did not finish native prefixes and reuse completed depth mirrors");
	f.cache.flush(depth);
	require(f.memory->read({8192, 64}).bytes == original, "Mirror admission altered preserved native depth/stencil bytes");
	std::cout << "Ten native depth-mirror passes under two-mirror quota:ordered prefix retirement/completed storage reuse,ten generations andexact preserved depth/stencil bytes passed\n";
}

void pressure_fallback(context& ctx)
{
	std::size_t cost;
	{
		surface_pool probe(ctx, 1 << 20);
		cost = probe.acquire({4, 4, 1, surface_color::argb8, false})->reserved_bytes();
	}
	fixture f(ctx, cost);
	const auto before = f.cache.transaction_statistics();
	{
		auto batch = f.cache.batch_draws();
		for (unsigned i = 0; i < 3; ++i) f.clear(0xff123400 + i, i);
		batch.finish();
	}
	const auto after = f.cache.transaction_statistics();
	require(after.raw_backup_submissions == before.raw_backup_submissions + 3 && after.batched_draws == before.batched_draws,
		"Insufficient private-version quota failed to preserve ordinary bounded fallback");
	f.cache.flush(f.target);
	const auto bytes = f.memory->read({4096, 64}).bytes;
	for (unsigned i = 0; i < 3; ++i) require(bytes[4 * i + 3] == std::byte(i), "Batch fallback lost ordered guest pixels");
	std::cout << "One-native-heap draw batch admission retains bounded raw fallback andexact ordered results passed\n";
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
			actual_draws(ctx);
			capacity_and_boundaries(ctx);
			failures_and_timeout(ctx);
			submitted_physical_ownership(ctx);
			owner_lifetime(ctx);
			resource_admission();
			upload_admission();
			admission_failure_boundaries(ctx);
			admission_gpu_failure(ctx);
			admitted_fragment_constants(ctx);
			mirror_admission(ctx);
			pressure_fallback(ctx);
			ctx.wait_idle();
			return 0;
		}
		catch (const heap_budget_error& e)
		{
			std::cerr << e.what() << " requested=" << e.requested << " reserved=" << e.reserved << " budget=" << e.budget << '\n'; return 1;
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
