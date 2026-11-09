#include "M4FragmentShader.h"
#include "M4HeapBudget.h"
#include "M4RenderPipeline.h"
#include "M4Texture.h"
#include "M4Transfer.h"
#include "M4VertexFetch.h"
#include "M4VertexShader.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
	if (!condition)
		throw std::runtime_error(message);
}

void buffer_budget_test()
{
	using namespace metal4;
	context probe;
	const auto native = make_buffer(probe, 1024 * 1024, buffer_storage::shared);
	const auto page_charge = std::max(native.length, native.allocatedSize);
	auto budget = std::make_shared<heap_budget>(probe, page_charge);
	@autoreleasepool
	{
		context ctx(1);
		ctx.set_buffer_budget(budget);
		ctx.begin();
		const std::array<std::byte, 16> data{};
		const auto first = ctx.upload(data, 16);
		const auto next = ctx.upload(data, 16);
		require(first.buffer == next.buffer && next.offset == 16 && budget->statistics().reserved_bytes == page_charge,
				"Upload slices do not share one charged native page");
		try
		{
			(void)make_buffer(ctx, 8, buffer_storage::private_gpu);
			throw std::runtime_error("Private intermediate bypassed upload budget");
		}
		catch (const heap_budget_error&)
		{
		}
		ctx.cancel();
		require(budget->statistics().reserved_bytes == page_charge, "Cancel released a reusable upload page charge");
		ctx.begin();
		const auto reused = ctx.upload(data, 16);
		require(reused.buffer == first.buffer && reused.offset == 0 && budget->statistics().live_buffers == 1,
				"Cancelled upload page was not reused within its budget");
		ctx.submit().wait();
		try
		{
			ctx.set_buffer_budget(budget);
			throw std::runtime_error("Context replaced installed buffer ledger");
		}
		catch (const std::logic_error&)
		{
		}
	}
	require(budget->statistics().reserved_bytes == 0, "Destroyed idle context retained upload charge");
	{
		context late;
		const auto prior = make_buffer(late, 8, buffer_storage::shared);
		try
		{
			late.set_buffer_budget(budget);
			throw std::runtime_error("Late ledger ignored preexisting allocation");
		}
		catch (const std::logic_error&)
		{
		}
		require(bool(prior), "Late-allocation probe missing");
	}
	{
		auto pending_budget = std::make_shared<heap_budget>(probe, page_charge * 2);
		std::optional<submission> pending;
		id<MTLSharedEvent> gate;
		std::size_t charge = 0;
		@autoreleasepool
		{
			auto ctx = std::make_unique<context>(1);
			ctx->set_buffer_budget(pending_budget);
			id<MTLBuffer> temporary = make_buffer(*ctx, 1024 * 1024, buffer_storage::private_gpu);
			gate = [ctx->device() newSharedEvent];
			ctx->begin();
			ctx->wait_event(gate, 1);
			const std::array<std::byte, 16> data{};
			(void)ctx->upload(data, 16);
			charge = pending_budget->statistics().reserved_bytes;
			require(pending_budget->statistics().live_buffers == 2, "Pending upload/private buffers not both charged");
			fill_buffer(*ctx, temporary, 0, temporary.length, 0x5a);
			pending = ctx->submit();
			temporary = nil;
			ctx.reset();
		}
		require(!pending->ready() && pending_budget->statistics().reserved_bytes == charge,
				"Pending context destruction released intermediate allocation charge");
		gate.signaledValue = 1;
		pending->wait();
		require(pending_budget->statistics().reserved_bytes == 0, "Completed temporary/upload buffers stayed charged");
	}
	std::cout << "Context upload/private buffers share budget;cancel reuse,late install refusal and pending context destruction passed\n";
}

void upload_pressure_test()
{
	using namespace metal4;
	context probe;
	const auto native = make_buffer(probe, 1024 * 1024, buffer_storage::shared);
	const auto charge = std::max(native.length, native.allocatedSize);
	const std::array<std::byte, 16> data{std::byte{0x42}};
	{
		context ctx(1);
		auto budget = std::make_shared<heap_budget>(ctx, charge);
		ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			ctx.begin();
			(void)ctx.upload(data, 16);
			ctx.cancel();
		}
		const auto replacement = make_buffer(ctx, 1024 * 1024, buffer_storage::private_gpu);
		require(replacement.storageMode == MTLStorageModePrivate && budget->statistics().live_buffers == 1,
				"Cancelled idle upload page prevented private allocation under pressure");
	}
	{
		context ctx(2);
		auto budget = std::make_shared<heap_budget>(ctx, charge);
		ctx.set_buffer_budget(budget);
		@autoreleasepool
		{
			ctx.begin();
			(void)ctx.upload(data, 16);
			ctx.submit().wait();
		}
		ctx.begin();
		(void)ctx.upload(data, 16);
		require(budget->statistics().live_buffers == 1 && ctx.trim_upload_pages() == 0,
				"Completed-slot pressure failed or trimmed used recording page");
		ctx.cancel();
	}
	{
		context ctx(1);
		auto budget = std::make_shared<heap_budget>(ctx, charge);
		ctx.set_buffer_budget(budget);
		upload_slice retained{};
		@autoreleasepool
		{
			ctx.begin();
			retained = ctx.upload(data, 16);
			ctx.cancel();
		}
		require(ctx.trim_upload_pages() == 1 && budget->statistics().reserved_bytes == charge,
				"Trimming removed external upload owner's charge");
		try
		{
			(void)make_buffer(ctx, 1024 * 1024, buffer_storage::private_gpu);
			throw std::runtime_error("External upload owner allowed budget overcommit");
		}
		catch (const heap_budget_error&)
		{
		}
		require(std::memcmp(static_cast<const std::byte*>(retained.buffer.contents) + retained.offset, data.data(), data.size()) == 0,
				"Trimmed external upload slice was corrupted");
		retained.buffer = nil;
		require(budget->statistics().reserved_bytes == 0, "Released external upload owner stayed charged");
		const auto replacement = make_buffer(ctx, 1024 * 1024, buffer_storage::private_gpu);
		require(bool(replacement), "Released external upload budget not reusable");
	}
	{
		context ctx(1);
		auto budget = std::make_shared<heap_budget>(ctx, charge * 2);
		ctx.set_buffer_budget(budget);
		const std::vector<std::byte> full(1024 * 1024);
		@autoreleasepool
		{
			ctx.begin();
			(void)ctx.upload(full, 16);
			(void)ctx.upload(full, 16);
			ctx.cancel();
		}
		ctx.begin();
		const auto first = ctx.upload(data, 16);
		require(ctx.trim_upload_pages() == 1 && budget->statistics().live_buffers == 1,
				"Unused recording page was not reclaimed independently");
		const auto second = ctx.upload(data, 16);
		require(first.buffer == second.buffer && second.offset == 16 &&
					std::memcmp(static_cast<const std::byte*>(first.buffer.contents) + first.offset, data.data(), data.size()) == 0,
				"Active-page trimming corrupted slice or append offset");
		const auto replacement = make_buffer(ctx, 1024 * 1024, buffer_storage::private_gpu);
		require(bool(replacement) && budget->statistics().live_buffers == 2, "Reclaimed recording page did not free budget");
		ctx.cancel();
	}
	{
		context ctx(2);
		auto budget = std::make_shared<heap_budget>(ctx, charge);
		ctx.set_buffer_budget(budget);
		const auto gate = [ctx.device() newSharedEvent];
		std::optional<submission> pending;
		@autoreleasepool
		{
			ctx.begin();
			ctx.wait_event(gate, 1);
			(void)ctx.upload(data, 16);
			pending = ctx.submit();
		}
		ctx.begin();
		try
		{
			(void)ctx.upload(data, 16);
			throw std::runtime_error("Pending upload page recycled under pressure");
		}
		catch (const heap_budget_error&)
		{
		}
		require(!pending->ready() && ctx.trim_upload_pages() == 0 && budget->statistics().reserved_bytes == charge,
				"Trimming waited for or released pending upload storage");
		ctx.cancel();
		gate.signaledValue = 1;
		pending->wait();
		require(ctx.trim_upload_pages() == 1 && budget->statistics().reserved_bytes == 0,
				"Completed pending page cannot be reclaimed");
	}
	std::cout << "Upload pressure reclaims cancelled/completed/unused pages,preserves active slices and never recycles pending GPU pages passed\n";
	std::cout << "Trimmed external upload owner keeps exact bytes and budget charge until last release passed\n";
}

void library_cache_test()
{
	using namespace metal4;
	context ctx;
	const std::string a = "#include <metal_stdlib>\nusing namespace metal;\nkernel void cached_write(device uint* out [[buffer(0)]]) { out[0] = 1; }\n";
	auto b = a, c = a;
	b[b.find("= 1") + 2] = '2';
	c[c.find("= 1") + 2] = '3';
	const auto output = make_buffer(ctx, 4, buffer_storage::shared);
	const auto gate = [ctx.device() newSharedEvent];
	std::optional<submission> pending;
	@autoreleasepool
	{
		auto cached = std::make_unique<compiler>(ctx, 2, 4096);
		const auto old_a = cached->library(a, @"first label");
		const auto old_b = cached->library(b);
		require(cached->library(std::string(a), @"new diagnostic label") == old_a && cached->library_statistics().reused == 1,
				"Exact source cache missed or label changed native identity");
		require(cached->library_statistics().candidates == 1, "Exact source index scanned an unrelated library");
		try
		{
			(void)cached->library("#error deliberate cache failure\n");
			throw std::logic_error("Invalid native source compiled");
		}
		catch (const std::runtime_error&)
		{
		}
		require(cached->library_statistics().entries == 2 && cached->library_statistics().created == 2 &&
					cached->library_statistics().evicted == 0,
				"Compilation failure evicted successful cached libraries");
		(void)cached->library(c);
		require(cached->library_statistics().entries == 2 && cached->library_statistics().source_bytes == a.size() + c.size() &&
					cached->library_statistics().evicted == 1 && cached->library(a) == old_a,
				"Library LRU evicted the recently reused source");
		require(cached->library(b) != old_b && cached->library_statistics().created == 4, "Evicted library was not recompiled");
		cached->clear_libraries();
		require(cached->library_statistics().entries == 0 && cached->library_statistics().source_bytes == 0,
				"Library clear retained cache entries/source charge");
		auto descriptor = [MTL4ComputePipelineDescriptor new];
		descriptor.computeFunctionDescriptor = library_function(old_a, @"cached_write");
		id<MTLComputePipelineState> pipeline = cached->compute(descriptor);
		ctx.begin();
		ctx.wait_event(gate, 1);
		ctx.keep_alive(pipeline);
		bindings resources;
		resources.set_buffer(0, output);
		const auto arguments = resources.snapshot(ctx);
		const auto encoder = ctx.compute();
		[encoder setComputePipelineState:pipeline];
		[encoder setArgumentTable:arguments];
		[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
		pending = ctx.submit();
		pipeline = nil;
		cached.reset();
	}
	require(!pending->ready(), "Compiler cache lifetime gate did not remain pending");
	gate.signaledValue = 1;
	pending->wait();
	require(*static_cast<const std::uint32_t*>(output.contents) == 1,
			"Eviction/clear/compiler destruction corrupted retained native program");
	{
		compiler bytes(ctx, 4, a.size());
		(void)bytes.library(a);
		(void)bytes.library(b);
		require(bytes.library_statistics().entries == 1 && bytes.library_statistics().source_bytes == b.size() &&
					bytes.library_statistics().evicted == 1,
				"Library source byte quota exceeded");
		(void)bytes.library(a + "// oversized source\n");
		require(bytes.library_statistics().entries == 1 && bytes.library_statistics().bypassed == 1,
				"Oversized source evicted bounded cache instead of bypassing");
		compiler uncached(ctx, 0);
		(void)uncached.library(a);
		(void)uncached.library(a);
		require(uncached.library_statistics().entries == 0 && uncached.library_statistics().created == 2 &&
					uncached.library_statistics().reused == 0 && uncached.library_statistics().bypassed == 2,
				"Disabled library cache retained source or skipped compilation");
	}
	std::cout << "Native exact-source library cache,LRU/source quota/failure isolation,bypass and pending GPU survival after clear/compiler destruction passed\n";
}

void transfer_test(metal4::context& ctx)
{
	using namespace metal4;
	transfer_engine transfer(ctx);
	const std::array<std::byte, 13> input{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5},
										  std::byte{6}, std::byte{7}, std::byte{8}, std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12}, std::byte{13}};
	const auto source = make_upload(ctx, input);
	const auto target = make_buffer(ctx, 64, buffer_storage::shared);
	constexpr std::size_t elements = 65537; // All E4M12 encodings and an odd tail.
	std::vector<std::byte> packed(elements * 2);
	for (std::size_t i = 0; i < elements; ++i)
	{
		packed[i * 2] = static_cast<std::byte>(i >> 8);
		packed[i * 2 + 1] = static_cast<std::byte>(i);
	}
	const auto depth_source = make_upload(ctx, packed);
	const auto expanded = make_buffer(ctx, elements * 4, buffer_storage::shared);
	const auto contracted = make_buffer(ctx, elements * 2, buffer_storage::shared);
	ctx.begin();
	fill_buffer(ctx, target, 0, 64, 0xa5);
	transfer.fill_words(ctx, target, 4, 20, 0x12345678);
	transfer.copy(ctx, source, 1, target, 7, 8, 4);
	transfer.copy(ctx, source, 0, target, 32, 12);
	transfer.convert_e4m12(ctx, depth_source, 0, expanded, 0, elements, true, true, false);
	transfer.convert_e4m12(ctx, expanded, 0, contracted, 0, elements, false, false, true);
	ctx.submit().wait();
	std::array<std::byte, 64> expected;
	expected.fill(std::byte{0xa5});
	for (std::size_t i = 4; i < 24; ++i)
		expected[i] = static_cast<std::byte>(0x12345678u >> ((i % 4) * 8));
	for (std::size_t i = 0; i < 8; ++i)
		expected[7 + i] = input[1 + (i / 4) * 4 + 3 - i % 4];
	std::copy_n(input.begin(), 12, expected.begin() + 32);
	const auto actual = map_buffer(target, 0, 64);
	require(std::equal(actual.begin(), actual.end(), expected.begin()), "GPU transfer/fill/endian results differ");
	const auto roundtrip = map_buffer(contracted, 0, elements * 2);
	require(std::equal(roundtrip.begin(), roundtrip.end(), packed.begin()), "GPU E4M12 round-trip differs");
	const auto floats = map_buffer(expanded, 0, elements * 4);
	for (std::size_t i = 0; i < elements; ++i)
	{
		const auto raw = static_cast<std::uint16_t>(i);
		const float expected_float = std::ldexp(1.0f + float(raw % 4096) / 4096.0f, int(raw / 4096) - 7);
		std::uint32_t bits;
		std::memcpy(&bits, floats.data() + i * 4, 4);
		require(bits == std::bit_cast<std::uint32_t>(expected_float), "GPU E4M12 expansion differs from numerical model");
	}
	std::cout << "Native GPU transfers, byte guards and all 65536 E4M12 values plus odd tail passed\n";
}

void append_fragment_word(std::vector<std::byte>& output, std::uint32_t value)
{
	for (const unsigned shift : {8u, 0u, 24u, 16u})
		output.push_back(static_cast<std::byte>(value >> shift));
}

void render_test(metal4::context& ctx)
{
	using namespace metal4;
	compiler compiler(ctx);
	// One RSX MOV o[0], v[0], full mask, TR condition and END. Sources use the
	// physical split operand encoding, independently of the native decoder.
	constexpr std::uint32_t input_xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	const std::array<std::array<std::uint32_t, 4>, 1> vertex_code{{{(63u << 15) | (7u << 10) | (1u << 30),
																	(1u << 22) | (input_xyzw >> 9), (input_xyzw & 511) << 23,
																	(15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
	vertex_shader vertex(compiler, vertex_code);
	std::cout << "Native RSX vertex library compiled\n";
	std::vector<std::byte> fragment_code;
	for (const auto word : {(1u << 24) | (15u << 9) | 1u,
							(0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
		append_fragment_word(fragment_code, word);
	for (const float value : {0.25f, 0.5f, 0.75f, 1.0f})
		append_fragment_word(fragment_code, std::bit_cast<std::uint32_t>(value));
	fragment_shader fragment(compiler, fragment_code);
	std::cout << "Native RSX fragment library compiled\n";
	render_pipeline_configuration pipeline_config;
	pipeline_config.colors[0] = MTLPixelFormatRGBA8Unorm;
	const auto pipeline = make_render_pipeline(compiler, pipeline_config, vertex.function(), fragment.function());
	std::cout << "Native triangle pipeline compiled\n";
	vertex_program_options point_options;
	point_options.point_output = true;
	vertex_shader point_vertex(compiler, vertex_code, point_options);
	auto point_config = pipeline_config;
	point_config.topology = MTLPrimitiveTopologyClassPoint;
	const auto point_pipeline = make_render_pipeline(compiler, point_config, point_vertex.function(), fragment.function());
	require(point_pipeline.pipeline != nil, "Native point pipeline compilation failed");
	std::cout << "Native point pipeline compiled\n";
	auto blend_config = pipeline_config;
	blend_config.state.colors[0].enabled = true;
	for (const auto factor : {0u, 1u, 0x300u, 0x301u, 0x302u, 0x303u, 0x304u, 0x305u,
							  0x306u, 0x307u, 0x308u, 0x8001u, 0x8002u, 0x8003u, 0x8004u})
	{
		auto& state = blend_config.state.colors[0];
		state.source_rgb = state.destination_rgb = state.source_alpha = state.destination_alpha = factor;
		const auto variant = make_render_pipeline(compiler, blend_config, vertex.function(), fragment.function());
		require(variant.pipeline != nil, "Native blend factor pipeline compilation failed");
	}
	blend_config.state.colors[0] = {true, 0x302, 0x303, 1, 0, 0x8006, 0x8006, 9};
	const auto blend_pipeline = make_render_pipeline(compiler, blend_config, vertex.function(), fragment.function());
	std::cout << "Native pipelines for all fifteen blend factors compiled\n";
	vertex_fetch_engine fetch(ctx);
	std::vector<std::byte> packed_vertices;
	// An oversized full-screen triangle. Vertex attributes are PS3 byte order.
	for (const float value : {-1.f, -1.f, 0.5f, 1.f, 3.f, -1.f, 0.5f, 1.f, -1.f, 3.f, 0.5f, 1.f})
	{
		const auto raw = std::bit_cast<std::uint32_t>(value);
		for (const unsigned shift : {24u, 16u, 8u, 0u})
			packed_vertices.push_back(static_cast<std::byte>(raw >> shift));
	}
	const auto guest_stream = make_upload(ctx, packed_vertices);
	const auto geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	const auto readback = make_buffer(ctx, 4096, buffer_storage::shared);
	MTLTextureDescriptor* texture_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
																								  width:16
																								 height:16
																							  mipmapped:NO];
	texture_descriptor.storageMode = MTLStorageModePrivate;
	texture_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	const auto target = make_texture(ctx, texture_descriptor);
	std::cout << "Native render resources created\n";
	ctx.begin(); // Reuses the completed transfer slot and its command allocator.
	vertex_fetch_request request;
	request.vertex_count = 3;
	request.attribute_mask = 1;
	request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
	vertex_inputs vertex_data;
	const auto native_stream = make_buffer(ctx, packed_vertices.size(), buffer_storage::private_gpu);
	copy_buffer(ctx, guest_stream, 0, native_stream, 0, packed_vertices.size());
	const auto before_fetch = ctx.encoding_statistics();
	vertex_data.attributes = fetch.decode(ctx, request, native_stream, nil);
	const auto after_fetch = ctx.encoding_statistics();
	require(after_fetch.checked_dispatches == before_fetch.checked_dispatches + 1 &&
		after_fetch.blit_to_dispatch_barriers == before_fetch.blit_to_dispatch_barriers + 1 &&
		after_fetch.compute_encoders == before_fetch.compute_encoders,
		"Vertex fetch must wait for its stream blit within the same compute encoder");
	vertex_context vertex_state;
	vertex_state.vertex_count = 3;
	const auto vertex_bindings = vertex.prepare(ctx, vertex_state, geometry, vertex_data);
	const auto fragment_bindings = fragment.prepare(ctx, {}, fragment_code, {});
	std::cout << "Native fetch and shader bindings encoded after a checked blit-to-dispatch dependency\n";
	MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
	pass.colorAttachments[0].texture = target;
	pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
	render_commands commands(ctx, pass);
	raster_state raster;
	raster.viewport = {0, 0, 16, 16, 0, 1};
	raster.scissor = {0, 0, 16, 16};
	if (![ctx.device() supportsFamily:MTLGPUFamilyApple10])
	{
		raster.depth_bounds_enabled = true;
		bool rejected = false;
		try
		{
			pipeline.configure(commands, raster, vertex_bindings, fragment_bindings);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Unsupported depth bounds must reject before calling the driver");
		raster.depth_bounds_enabled = false;
	}
	commands.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vertex_bindings, fragment_bindings);
	std::cout << "Native triangle draw encoded\n";
	commands.end();
	texture_region region;
	region.size = MTLSizeMake(16, 16, 1);
	texture_to_buffer(ctx, target, region, readback, 0, 256, 4096);
	std::cout << "Native texture readback encoded\n";
	ctx.submit().wait();
	const auto result = map_buffer(readback, 0, 4096);
	for (std::size_t y = 0; y < 16; ++y)
		for (std::size_t x = 0; x < 16; ++x)
		{
			const auto pixel = result.subspan(y * 256 + x * 4, 4);
			const std::array<std::byte, 4> expected{std::byte{64}, std::byte{128}, std::byte{191}, std::byte{255}};
			if (!std::equal(pixel.begin(), pixel.end(), expected.begin()))
			{
				std::cerr << "Unexpected native pixel " << x << ',' << y << ": ";
				for (auto channel : pixel)
					std::cerr << unsigned(channel) << ' ';
				std::cerr << '\n';
				throw std::runtime_error("Native RSX fetch/vertex/fragment/draw/readback result differs");
			}
		}
	std::cout << "Native GPU RSX fetch/vertex/fragment/pipeline/draw/readback passed for all 256 pixels\n";
	// Reuse the completed slot, refresh only the literal alpha, blend against
	// a known destination, and preserve the masked G/B channels. This also checks
	// that fresh argument tables/uploads don't corrupt previously recorded work.
	fragment_code.resize(fragment_code.size() - 4);
	append_fragment_word(fragment_code, std::bit_cast<std::uint32_t>(0.25f));
	ctx.begin();
	const auto blended_vertex = vertex.prepare(ctx, vertex_state, geometry, vertex_data);
	const auto blended_fragment = fragment.prepare(ctx, {}, fragment_code, {});
	pass.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 0.5);
	render_commands blended_commands(ctx, pass);
	blended_commands.draw_geometry(geometry, blend_pipeline.pipeline, blend_pipeline.depth_stencil,
								   raster, blended_vertex, blended_fragment);
	blended_commands.end();
	texture_to_buffer(ctx, target, region, readback, 0, 256, 4096);
	ctx.submit().wait();
	const auto blended_result = map_buffer(readback, 0, 4096);
	for (std::size_t y = 0; y < 16; ++y)
		for (std::size_t x = 0; x < 16; ++x)
		{
			const auto pixel = blended_result.subspan(y * 256 + x * 4, 4);
			const std::array<std::byte, 4> expected{std::byte{54}, std::byte{102}, std::byte{153}, std::byte{64}};
			require(std::equal(pixel.begin(), pixel.end(), expected.begin()), "Native RGB/alpha blending, channel mask or literal refresh differs");
		}
	std::cout << "Native GPU separate RGB/alpha blending, channel masks, literal refresh and slot reuse passed\n";
}
} // namespace

int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			// Standalone native backend tests; no RPCS3 application, guest process,
			// window, drawable, game or UI subsystem is loaded or started.
			buffer_budget_test();
			upload_pressure_test();
			library_cache_test();
			metal4::context ctx(1);
			auto budget = std::make_shared<metal4::heap_budget>(ctx, 64u << 20);
			ctx.set_buffer_budget(budget);
			transfer_test(ctx);
			render_test(ctx);
			ctx.wait_idle();
			require(budget->statistics().live_buffers > 0, "Integrated transfer/fetch/draw did not charge upload pages");
			std::cout << "Integrated native transfer/vertex-fetch/draw uses shared buffer ledger passed\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << "Native Metal4 GPU test failed: " << error.what() << '\n';
			return 1;
		}
	}
}
