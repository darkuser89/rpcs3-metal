#include "M4ConstantUpload.h"
#include "M4HeapBudget.h"
#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4Sampler.h"
#include "M4VertexProgram.h"
#include "M4VertexDataSnapshot.h"

#include <bit>
#include <cstring>
#include <iostream>
#include <optional>
#include <sys/mman.h>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
void append(std::vector<std::byte>& bytes, std::uint32_t word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u}) bytes.push_back(std::byte(word >> shift));
}
std::vector<std::byte> fragment_code(unsigned seed)
{
	std::vector<std::byte> bytes;
	for (unsigned i = 0; i < 3; ++i)
	{
		// A literal in a dead NOP source still occupies a physical block.
		for (auto word : {i == 2 ? 1u : 0u, 2u, 0u, 0u}) append(bytes, word);
		for (auto word : {seed + i, 0x80000000u, 0x7fc12345u + i, 0x7f800000u}) append(bytes, word);
	}
	return bytes;
}
void fragment_versions(context& ctx)
{
	constant_upload_cache cache(ctx, 2, 96);
	auto bytes = fragment_code(1);
	const auto image = decode_fragment_program(bytes);
	const auto first = cache.fragment(image, bytes);
	for (unsigned i = 0; i < 100; ++i)
		require(cache.fragment(image, bytes) == first, "Unchanged literal upload version changed");
	require(cache.statistics().created == 1 && cache.statistics().reused == 100 && cache.statistics().uploaded_bytes == 48,
		"Unchanged fragment literals allocated/uploaded again");
	bytes = fragment_code(2);
	const auto second = cache.fragment(image, bytes);
	require(second != first && static_cast<const unsigned*>(first.contents)[0] == 1 && static_cast<const unsigned*>(second.contents)[0] == 2,
		"Literal update overwrote a previously owned version");
	const auto expected = refresh_fragment_constants(image, bytes);
	require(std::memcmp(second.contents, expected.data(), 48) == 0, "Literal NaN/sign bits differ");
	bytes.insert(bytes.end(), 17, std::byte{0xff});
	require(cache.fragment(image, bytes) == second, "Trailing fragment bytes changed constants");
	unsigned refused = 0;
	const auto before = cache.statistics();
	for (std::size_t prefix = 0; prefix < image.byte_length; ++prefix)
	{
		try { (void)cache.fragment(image, std::span(bytes).first(prefix)); }
		catch (const std::invalid_argument&) { ++refused; }
	}
	for (unsigned word = 0; word < 12; ++word)
	{
		auto invalid = bytes;
		invalid[(word / 4) * 32 + (word % 4) * 4] ^= std::byte{1};
		try { (void)cache.fragment(image, invalid); }
		catch (const std::invalid_argument&) { ++refused; }
	}
	require(refused == 108 && cache.statistics().created == before.created && cache.statistics().evicted == before.evicted,
		"Invalid layout or instruction mutation reached upload cache");
	(void)cache.fragment(image, fragment_code(3));
	require(cache.statistics().entries == 2 && cache.statistics().payload_bytes == 96 && cache.statistics().evicted == 1,
		"Constant version LRU/payload budget failed");
	cache.clear();
	require(static_cast<const unsigned*>(first.contents)[0] == 1 && !cache.statistics().entries, "Clear invalidated external constant owner");
	constant_upload_cache tiny(ctx, 2, 16), disabled(ctx, 0);
	(void)tiny.fragment(image, bytes);
	(void)disabled.fragment(image, bytes);
	(void)disabled.fragment(image, bytes);
	require(tiny.statistics().bypassed == 1 && !tiny.statistics().entries && disabled.statistics().created == 2 && !disabled.statistics().entries,
		"Tiny/disabled constant cache retained versions");
	std::vector<std::byte> flow;
	append(flow, 1u); append(flow, 0); append(flow, 0); append(flow, 0);
	const auto empty = cache.fragment(decode_fragment_program(flow), flow);
	require(empty.length == 16 && static_cast<const unsigned*>(empty.contents)[0] == 0, "Empty constants lack valid zero binding");
	std::cout << "Fragment constants:100 reuse hits,one48-byte upload,dynamic NaN/sign payloads,108 truncation/word refusals,LRU/quota/clear/bypass andempty binding passed\n";
}

// Actual translated RSX ARL a0, v0; MOV o0, c[a0.x]. Every register is
// addressed indirectly by a GPU thread, including all registers468..511.
id<MTLComputePipelineState> indirect_pipeline(compiler& compile)
{
	constexpr unsigned xyzw = (1u << 12) | (2u << 10) | (3u << 8);
	const unsigned input = xyzw | 2, constant = xyzw | 3;
	const std::array<std::array<unsigned, 4>, 2> code{{
		{7u << 10, (13u << 22) | (input >> 9), (input & 511) << 23, 15u << 13},
		{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (constant >> 9), (constant & 511) << 23,
			(15u << 13) | (15u << 17) | (1u << 7) | 3u}
	}};
	auto generated = generate_vertex_program(code);
	require(generated.indexed_constants, "RSX probe does not use indirect constants");
	generated.source += R"(
kernel void inspect_constants(device const float4* attributes [[buffer(0)]],
 device const float4* constants [[buffer(1)]],constant rsx_vertex_context& ctx [[buffer(2)]],
 constant rsx_sampler_parameters* sampling [[buffer(3)]],device uint4* out [[buffer(4)]],
 device uint* faults [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 const auto r = rsx_evaluate(attributes,constants,ctx,id,sampling);
 out[id] = as_type<uint4>(r.dst[0]); faults[id] = r.fault;
}
)";
	const auto library = compile.library(generated.source);
	auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(library, @"inspect_constants");
	return compile.compute(descriptor);
}
void indirect_and_retention()
{
	auto ctx = std::make_unique<context>(2);
	compiler compile(*ctx);
	const auto pipeline = indirect_pipeline(compile);
	std::vector<std::array<float, 4>> attributes(512 * 16);
	vertex_constant_registers first_values{}, second_values{};
	for (unsigned i = 0; i < 512; ++i)
	{
		attributes[i * 16][0] = float(i);
		first_values[i] = {std::bit_cast<unsigned>(float(i) + .25f), 0x80000000u, 0x7fc12345u, 0x3f800000u};
		second_values[i] = {std::bit_cast<unsigned>(float(i) + .5f), 0u, 0x7fc54321u, 0xbf800000u};
	}
	const auto inputs = make_upload(*ctx, std::as_bytes(std::span(attributes)));
	vertex_context state;
	state.vertex_count = state.constant_count = 512;
	const auto uniform = make_upload(*ctx, std::as_bytes(std::span(&state, 1)));
	std::array<sampler_parameters, 4> sampling{};
	const auto samplers = make_upload(*ctx, std::as_bytes(std::span(sampling)));
	const auto gate = [ctx->device() newSharedEvent];
	std::array<id<MTLBuffer>, 2> outputs, faults;
	std::array<std::optional<submission>, 2> submissions;
	for (unsigned pass = 0; pass < 2; ++pass)
	{
		@autoreleasepool
		{
			const auto& values = pass ? second_values : first_values;
			const auto constants = ctx->constant_uploads().words(std::as_bytes(std::span(values)));
			const auto stats = ctx->constant_uploads().statistics();
			for (unsigned hit = 0; hit < 100; ++hit)
				require(ctx->constant_uploads().words(std::as_bytes(std::span(values))) == constants, "Vertex constant buffer identity changed");
			require(ctx->constant_uploads().statistics().uploaded_bytes == stats.uploaded_bytes && ctx->constant_uploads().statistics().reused == stats.reused + 100,
				"Vertex constant hit performed an upload");
			outputs[pass] = make_buffer(*ctx, sizeof(values), buffer_storage::shared);
			faults[pass] = make_buffer(*ctx, 512 * 4, buffer_storage::shared);
			ctx->begin(); ctx->wait_event(gate, 1); ctx->keep_alive(pipeline);
			bindings resources;
			resources.set_buffer(0, inputs); resources.set_buffer(1, constants);
			resources.set_buffer(2, uniform); resources.set_buffer(3, samplers);
			resources.set_buffer(4, outputs[pass]); resources.set_buffer(5, faults[pass]);
			const auto arguments = resources.snapshot(*ctx);
			const auto encoder = ctx->compute();
			[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:arguments];
			[encoder dispatchThreads:MTLSizeMake(512, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
			submissions[pass] = ctx->submit();
		}
	}
	require(!submissions[0]->ready() && !submissions[1]->ready(), "Pending constant versions did not remain gated");
	require(ctx->constant_uploads().statistics().created == 2 && ctx->constant_uploads().statistics().uploaded_bytes == 16384,
		"Two distinct full-register versions did not upload exactly once");
	for (unsigned i = 0; i < 64; ++i)
	{
		const std::array<unsigned, 4> newer{i, 0xdeadbeef, 0x12345678, 0x76543210};
		(void)ctx->constant_uploads().words(std::as_bytes(std::span(newer)));
	}
	require(ctx->constant_uploads().statistics().entries == 64 && ctx->constant_uploads().statistics().evicted == 2,
		"Pending constant versions were not evicted from the bounded cache");
	ctx->constant_uploads().clear();
	ctx.reset(); // Pending command resources, not this cache, retain both versions.
	gate.signaledValue = 1;
	for (unsigned pass = 0; pass < 2; ++pass)
	{
		submissions[pass]->wait();
		const auto& values = pass ? second_values : first_values;
		require(std::memcmp(outputs[pass].contents, values.data(), sizeof(values)) == 0, "Indirect RSX constant read changed bits or used another version");
		for (unsigned i = 0; i < 512; ++i) require(static_cast<const unsigned*>(faults[pass].contents)[i] == 0, "Indirect RSX constant read faulted");
	}
	std::cout << "Two pending native MSL4 RSX ARL/MOV programs:1024 indirect register reads including468..511 preserve exact NaN/signed-zero bits after eviction/clear/context destruction;200 hits upload no bytes passed\n";
}
void budget_and_trim()
{
	context ctx;
	const auto probe = [ctx.device() newBufferWithLength:8192 options:MTLResourceStorageModeShared];
	const auto charge = std::max(probe.length, probe.allocatedSize);
	auto budget = std::make_shared<heap_budget>(ctx, charge);
	ctx.set_buffer_budget(budget);
	vertex_constant_registers values{};
	@autoreleasepool
	{
		(void)ctx.constant_uploads().words(std::as_bytes(std::span(values)));
	}
	require(budget->statistics().reserved_bytes == charge && ctx.constant_uploads().statistics().entries == 1,
		"Immutable constant upload bypassed native heap ledger");
	values[511][3] = 0x7fc12345;
	@autoreleasepool
	{
		const auto replacement = ctx.constant_uploads().words(std::as_bytes(std::span(values)));
		require(static_cast<const unsigned*>(replacement.contents)[2047] == 0x7fc12345 && budget->statistics().reserved_bytes == charge,
			"Constant-pressure retry failed to reclaim old unused version");
	}
	require(ctx.constant_uploads().statistics().entries == 1 && ctx.constant_uploads().statistics().evicted == 1,
		"Pressure retry retained obsolete constant versions");
	@autoreleasepool
	{
		ctx.begin();
		const auto held = ctx.constant_uploads().words(std::as_bytes(std::span(values)));
		ctx.use_resource(held);
		ctx.constant_uploads().clear();
	}
	require(budget->statistics().reserved_bytes == charge, "Cache clear released a recording constant binding");
	ctx.cancel();
	(void)ctx.trim_upload_pages();
	require(!ctx.constant_uploads().statistics().entries && budget->statistics().reserved_bytes == 0,
		"Explicit trim retained unused native constant charge");
	std::cout << "Constant buffers share native heap ledger;one-buffer pressure retry andrecording cancellation/trim release unused versions passed\n";
}
void packed_stream_versions()
{
	context ctx; auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
	@autoreleasepool
	{
		constant_upload_cache cache(ctx,2,48);
		const std::array<std::byte,16> zeros{};
		const auto short_version = cache.packed_bytes(std::span(zeros).first(1)), long_version = cache.packed_bytes(zeros);
		require(short_version != long_version,"Packed stream key omitted exact byte length");
		std::array<std::byte,17> bytes{}; bytes.back() = std::byte{0x7f};
		const auto first = cache.packed_bytes(bytes); const auto before = cache.statistics();
		for (unsigned i = 0; i < 100; ++i) require(cache.packed_bytes(bytes) == first,"Identical packed bytes did not reuse native owner");
		require(cache.statistics().created == before.created && cache.statistics().uploaded_bytes == before.uploaded_bytes && cache.statistics().reused == before.reused + 100,
			"Repeated packed bytes allocated/uploaded again");
		bytes.back() = std::byte{0x42}; const auto newer = cache.packed_bytes(bytes);
		require(newer != first && static_cast<const std::byte*>(first.contents)[16] == std::byte{0x7f} && static_cast<const std::byte*>(newer.contents)[16] == std::byte{0x42},
			"New packed tail byte overwrote an older immutable stream");
		for (unsigned i = 17; i < 32; ++i) require(static_cast<const std::byte*>(newer.contents)[i] == std::byte{},"Packed upload padding is not deterministic zero");
		require(cache.statistics().entries <= 2 && cache.statistics().payload_bytes <= 48 && cache.statistics().evicted > 0,"Packed stream LRU/byte bound exceeded");
		const std::array<std::byte,49> oversized{}; (void)cache.packed_bytes(oversized);
		require(cache.statistics().bypassed == 1 && !cache.packed_bytes({}),"Oversized/empty packed stream retained an invalid owner");
		// A real inaccessible virtual reservation proves preflight reads no
		// oversized source bytes, without physically allocating half a GiB.
		const std::size_t length = 0x20000001u;
		const auto inaccessible = mmap(nullptr,length,PROT_NONE,MAP_PRIVATE | MAP_ANON,-1,0);
		require(inaccessible != MAP_FAILED,"Packed stream bound fixture could not reserve virtual address space");
		struct mapping_scope { void* address; std::size_t size; ~mapping_scope() { munmap(address,size); } } mapping{inaccessible,length};
		const auto allocations = budget->statistics().created; bool refused = false;
		try { (void)cache.packed_bytes({static_cast<const std::byte*>(inaccessible),length}); } catch (const std::length_error&) { refused = true; }
		require(refused && budget->statistics().created == allocations,"Packed source bound allocated native backing before refusal");
		bool full_domain = false;
		try { (void)cache.packed_bytes({static_cast<const std::byte*>(inaccessible),0x20000000u}); }
		catch (const heap_budget_error&) { full_domain = true; }
		require(full_domain && budget->statistics().created == allocations,
			"Packed cache narrowed the existing stream domain or read source before actual native admission");
	}
	(void)ctx.trim_upload_pages(); require(budget->statistics().reserved_bytes == 0,"Packed cache teardown retained unused ledger charge");
	std::cout << "Packed stream cache:exact content/length,100zero-upload hits,tail-byte versions,zero padding,bounded LRU/bytes,bypass/empty;existing512MiB boundary reaches ledger admission,512MiB+1 refuses before reading a real inaccessible source/native allocation;native charges retire passed\n";
}
void packed_stream_orphan()
{
	std::shared_ptr<heap_budget> budget;
	@autoreleasepool
	{
		auto ctx = std::make_unique<context>(2); budget = std::make_shared<heap_budget>(*ctx,8u << 20); ctx->set_buffer_budget(budget);
		const auto gate = [ctx->device() newSharedEvent];
		const auto output = make_buffer(*ctx,32,buffer_storage::shared);
		std::array<std::byte,17> bytes; for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = std::byte(i * 13 + 1);
		const auto original = bytes;
		std::optional<submission> work;
		@autoreleasepool
		{
			const auto stream = ctx->vertex_uploads().packed_bytes(bytes);
			ctx->begin(); ctx->wait_event(gate,1); copy_buffer(*ctx,stream,0,output,0,32); work.emplace(ctx->submit());
		}
		ctx->vertex_uploads().clear(); (void)ctx->trim_upload_pages(); bytes.fill(std::byte{});
		bool timeout = false;
		try { work->wait(std::chrono::milliseconds(1)); } catch (const std::runtime_error&) { timeout = !work->ready(); }
		require(timeout && budget->statistics().reserved_bytes > 0,"Pending packed stream lost immutable backing on timeout/cache trim");
		ctx.reset(); gate.signaledValue = 1; work->wait();
		require(std::memcmp(output.contents,original.data(),17) == 0,"Packed stream did not survive context/cache/source destruction");
		for (unsigned i = 17; i < 32; ++i) require(static_cast<const std::byte*>(output.contents)[i] == std::byte{},"Pending packed upload lost zero padding");
	}
	require(budget->statistics().reserved_bytes == 0,"Packed native orphan retained ledger backing after final GPU/caller release");
	std::cout << "Packed stream event-gated GPU copy:cache clear/trim,CPU timeout,mutable source andcontext destruction retain exact17bytes/zero padding through real completion;all native charges retire passed\n";
}
void packed_stream_pressure()
{
	context ctx;
	const auto probe = [ctx.device() newBufferWithLength:8192 options:MTLResourceStorageModeShared];
	const auto charge = std::max(probe.length,probe.allocatedSize);
	auto budget = std::make_shared<heap_budget>(ctx,charge); ctx.set_buffer_budget(budget);
	std::array<std::byte,8192> bytes{};
	@autoreleasepool { (void)ctx.vertex_uploads().packed_bytes(bytes); }
	require(budget->statistics().reserved_bytes == charge,"Packed native stream bypassed exact one-buffer ledger");
	bytes.back() = std::byte{0x7f};
	@autoreleasepool
	{
		const auto replacement = ctx.vertex_uploads().packed_bytes(bytes);
		require(static_cast<const std::byte*>(replacement.contents)[8191] == bytes.back() && budget->statistics().reserved_bytes == charge,
			"Packed pressure recovery did not retire an old unused stream before replacement");
	}
	require(ctx.vertex_uploads().statistics().entries == 1 && ctx.vertex_uploads().statistics().evicted == 1 && ctx.vertex_uploads().statistics().created == 2,
		"Packed pressure recovery retained stale cache ownership");
	ctx.begin();
	@autoreleasepool
	{
		const auto held = ctx.vertex_uploads().packed_bytes(bytes); ctx.use_resource(held); ctx.vertex_uploads().clear();
		const auto recording = ctx.recording_submission(); bytes.back() = std::byte{0x42}; bool refused = false;
		try { (void)ctx.vertex_uploads().packed_bytes(bytes); } catch (const heap_budget_error&) { refused = true; }
		require(refused && budget->statistics().reserved_bytes == charge && recording.same_batch(ctx.recording_submission()),
			"Packed pressure reused recording storage or altered caller recording before retirement");
	}
	ctx.cancel(); (void)ctx.trim_upload_pages();
	require(budget->statistics().reserved_bytes == 0 && !ctx.vertex_uploads().statistics().entries,"Cancelled packed recording left stale cache/native backing");
	std::cout << "Packed stream real one-buffer pressure:old cache owner retired before replacement,recording-pinned bytes refuse reuse,caller token preserved,cancel/trim returns exact native charge passed\n";
}
}
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			context ctx;
			fragment_versions(ctx);
			indirect_and_retention();
			budget_and_trim();
			packed_stream_versions();
			packed_stream_orphan();
			packed_stream_pressure();
			return 0;
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
