#include "M4FragmentShader.h"
#include "M4RenderPipeline.h"
#include "M4Texture.h"
#include "M4VertexShader.h"
#include "M4VertexTextureBatch.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string_view>

namespace
{
using namespace metal4;
using color = std::array<float, 4>;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void append(std::vector<std::byte>& bytes, unsigned word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		bytes.push_back(std::byte(word >> shift));
}
std::vector<std::byte> program(unsigned opcode, unsigned unit, color coords, bool gradients = false)
{
	std::vector<std::byte> result;
	for (unsigned w : {(opcode << 24) | (unit << 17) | (15u << 9) | 1u, (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, (0xe4u << 9) | (gradients ? 2u : 0u), (0xe4u << 9) | (gradients ? 2u : 0u)})
		append(result, w);
	for (float v : coords)
		append(result, std::bit_cast<unsigned>(v));
	return result;
}
color mip_color(unsigned level) { return {.125f + level * .125f, .25f, .5f - level * .125f, .75f}; }
struct fixture
{
	context ctx{1};
	compiler compile{ctx};
	std::shared_ptr<heap_budget> ledger = std::make_shared<heap_budget>(ctx, 64u << 20);
	sampler_cache cache{ctx, 4, ledger};
	id<MTLLibrary> vertex;
	id<MTLTexture> target;
	id<MTLBuffer> output;
	unsigned cases = 0;
	fixture()
	{
		vertex = compile.library("#include <metal_stdlib>\nusing namespace metal;\nvertex float4 full(uint id [[vertex_id]]) {const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],.5f,1);}");
		auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:2 height:2 mipmapped:NO];
		d.storageMode = MTLStorageModePrivate;
		d.usage = MTLTextureUsageRenderTarget;
		target = make_texture(ctx, d);
		output = make_buffer(ctx, 4096, buffer_storage::shared);
	}
	id<MTLTexture> source(vertex_texture_dimension dimension, MTLPixelFormat format = MTLPixelFormatRGBA32Float)
	{
		auto d = [MTLTextureDescriptor new];
		d.textureType = dimension == vertex_texture_dimension::three ? MTLTextureType3D : dimension == vertex_texture_dimension::cube ? MTLTextureTypeCube
																																	  : MTLTextureType2D;
		d.pixelFormat = format;
		d.width = 4;
		d.height = dimension == vertex_texture_dimension::one ? 1 : 4;
		d.depth = dimension == vertex_texture_dimension::three ? 4 : 1;
		d.mipmapLevelCount = 3;
		d.storageMode = MTLStorageModePrivate;
		d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		const auto image = make_texture(ctx, d);
		ctx.begin();
		for (unsigned level = 0; level < 3; ++level)
			for (unsigned face = 0; face < (dimension == vertex_texture_dimension::cube ? 6 : 1); ++face)
			{
				const auto w = std::max<NSUInteger>(1, image.width >> level), h = std::max<NSUInteger>(1, image.height >> level), z = std::max<NSUInteger>(1, image.depth >> level);
				std::vector<color> pixels(w * h * z, mip_color(level));
				texture_region region;
				region.level = level;
				region.slice = face;
				region.size = MTLSizeMake(w, h, z);
				if (format == MTLPixelFormatRGBA32Float)
					buffer_to_texture(ctx, make_upload(ctx, std::as_bytes(std::span(pixels))), 0, image, region, w * 16, w * h * 16);
				else
				{
					std::vector<std::byte> bytes(pixels.size() * 4);
					for (unsigned p = 0; p < pixels.size(); ++p)
						for (unsigned c = 0; c < 4; ++c)
							bytes[p * 4 + c] = std::byte(unsigned(std::nearbyint(pixels[p][c] * 255)));
					buffer_to_texture(ctx, make_upload(ctx, bytes), 0, image, region, w * 4, w * h * 4);
				}
			}
		ctx.submit().wait();
		return image;
	}
	void zero_mips(id<MTLTexture> image)
	{
		const auto before = ctx.encoding_statistics();
		const auto coverage = cache.border_coverage(image);
		const auto after = ctx.encoding_statistics();
		require(after.checked_dispatches == before.checked_dispatches + 3 &&
			after.dispatch_barriers == before.dispatch_barriers,
			"Independent zeroed coverage mips must use checked dispatch without intra-pass waits");
		require(cache.border_coverage(image) == coverage, "Equal source shape did not reuse coverage image");
		for (unsigned level = 0; level < 3; ++level)
			for (unsigned face = 0; face < (image.textureType == MTLTextureTypeCube ? 6 : 1); ++face)
			{
				const auto w = std::max<NSUInteger>(1, image.width >> level), h = std::max<NSUInteger>(1, image.height >> level), z = std::max<NSUInteger>(1, image.depth >> level);
				texture_region region;
				region.level = level;
				region.slice = face;
				region.size = MTLSizeMake(w, h, z);
				ctx.begin();
				texture_to_buffer(ctx, coverage, region, output, 0, 256, 256 * h);
				ctx.submit().wait();
				for (unsigned dz = 0; dz < z; ++dz)
					for (unsigned y = 0; y < h; ++y)
						for (unsigned x = 0; x < w * 4; ++x)
							require(static_cast<const unsigned char*>(output.contents)[dz * 256 * h + y * 256 + x] == 0, "Coverage mip/face was not zero initialized");
			}
		std::cout << "Three checked zero coverage mips preserve all texels without intra-pass waits\n";
	}
	void draw(vertex_texture_dimension dimension, id<MTLTexture> image, unsigned op, unsigned unit, color coords, bool linear, bool remap, float lod_bias = 0, bool mipmaps = false, unsigned anisotropy = 1, bool mixed = false)
	{
		decoded_texture_state state;
		state.enabled = true;
		state.border_emulation = true;
		state.policy.dimension = dimension;
		state.policy.requested_mips = state.policy.actual_mips = 3;
		state.sampler.wrap.fill(sampler_wrap::border);
		state.sampler.min_filter = linear ? sampler_min_filter::linear : sampler_min_filter::nearest;
		state.sampler.mag_filter = linear ? sampler_mag_filter::linear : sampler_mag_filter::nearest;
		if (mipmaps)
			state.sampler.min_filter = linear ? sampler_min_filter::linear_linear : sampler_min_filter::nearest_nearest;
		state.sampler.max_lod = 2;
		state.sampler.bias = lod_bias;
		state.sampler.anisotropy = anisotropy;
		if (mixed)
			state.sampler.wrap = {sampler_wrap::repeat, sampler_wrap::border, sampler_wrap::edge};
		state.view_border_color = {.2f, .4f, .6f, .3f};
		if (remap)
		{
			state.view_channels = {1, 0, 4, 5};
			state.view_border_color[2] = 0;
			state.view_border_color[3] = 1;
		}
		fragment_inputs inputs;
		inputs.set_texture_state(unit, image, cache, state);
		fragment_program_options options;
		configure_fragment_texture_state(options, unit, state);
		const auto code = program(op, unit, coords, mipmaps && op == 25);
		const auto render = [&](const fragment_inputs& resources, const fragment_program_options& compiled)
		{
			fragment_shader shader(compile, code, compiled);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			const auto pipeline = make_render_pipeline(compile, config, library_function(vertex, @"full"), shader);
			ctx.begin();
			const auto bindings = shader.prepare(ctx, {}, code, resources);
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = target;
			pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			render_commands commands(ctx, pass);
			raster_state raster;
			raster.viewport = {0, 0, 2, 2, 0, 1};
			raster.scissor = {0, 0, 2, 2};
			commands.configure(pipeline.pipeline, pipeline.depth_stencil, raster, {}, bindings);
			commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
			commands.end();
			texture_region region;
			region.size = MTLSizeMake(2, 2, 1);
			texture_to_buffer(ctx, target, region, output, 0, 256);
			ctx.submit().wait();
			color result;
			std::copy_n(static_cast<const float*>(output.contents), 4, result.begin());
			for (unsigned y = 0; y < 2; ++y)
				for (unsigned x = 0; x < 2; ++x)
					for (unsigned c = 0; c < 4; ++c)
						require(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c] == result[c], "Uniform sampling varied across target pixels");
			return result;
		};
		const auto actual = render(inputs, options);
		// Independent native oracle: sample the real source with canonical zero
		// and one borders. Their difference measures its actual interpolation
		// weights, including Apple's finite coordinate precision. The emulated
		// path instead samples a separate zero texture with one sampler only.
		auto native_inputs = inputs;
		auto native_options = options;
		native_options.border_textures = 0;
		auto native_state = state.sampler;
		native_state.border_color.fill(0);
		native_inputs.set_sampler(unit, cache.acquire(native_state, state.policy));
		const auto black = render(native_inputs, native_options);
		native_state.border_color.fill(1);
		native_inputs.set_sampler(unit, cache.acquire(native_state, state.policy));
		const auto white = render(native_inputs, native_options);
		for (unsigned c = 0; c < 4; ++c)
		{
			const float wanted = state.view_channels[c] == 4 ? 0 : state.view_channels[c] == 5 ? 1
																							   : black[c] + (white[c] - black[c]) * state.view_border_color[c];
			if (!std::isfinite(actual[c]) || std::abs(actual[c] - wanted) > 2e-5f)
				throw std::runtime_error("Native custom border mismatch dim=" + std::to_string(unsigned(dimension)) + " op=" + std::to_string(op) + " got=" + std::to_string(actual[c]) + " wanted=" + std::to_string(wanted));
		}
		++cases;
	}
	void float_mips(id<MTLTexture> image)
	{
		for (bool remap : {false, true})
		{
			const std::array<unsigned, 4> channels = remap ? std::array<unsigned, 4>{1, 0, 4, 5} : std::array<unsigned, 4>{0, 1, 2, 3};
			const auto before = ctx.encoding_statistics();
			const auto converted = cache.border_color_view(image, channels);
			const auto after = ctx.encoding_statistics();
			require(after.compute_encoders == before.compute_encoders + 1 &&
				after.checked_dispatches == before.checked_dispatches + 3 &&
				after.dispatch_queue_barriers == before.dispatch_queue_barriers + 1 &&
				after.dispatch_barriers == before.dispatch_barriers,
				"Independent border snapshot mips must share an encoder without intra-pass waits");
			require(cache.border_color_view(image, channels) == converted, "Immutable color snapshot was not cached");
			for (unsigned level = 0; level < 3; ++level)
			{
				const auto w = std::max<NSUInteger>(1, image.width >> level), h = std::max<NSUInteger>(1, image.height >> level);
				texture_region region;
				region.level = level;
				region.size = MTLSizeMake(w, h, 1);
				ctx.begin();
				texture_to_buffer(ctx, converted, region, output, 0, 256);
				ctx.submit().wait();
				auto expected = mip_color(level);
				for (auto& value : expected)
					value = std::nearbyint(value * 255) / 255;
				if (remap)
					expected = {expected[1], expected[0], 0, 1};
				for (unsigned y = 0; y < h; ++y)
					for (unsigned x = 0; x < w; ++x)
						for (unsigned c = 0; c < 4; ++c)
							require(std::abs(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c] - expected[c]) < 2e-6f, "Float color snapshot changed UNORM texels/mips/view remap");
			}
		}
		std::cout << "Six checked border snapshot mips preserve color and remapping without intra-pass waits\n";
	}
	void vertex_and_lifetime(id<MTLTexture> image)
	{
		const auto gate = [ctx.device() newSharedEvent];
		const auto pending = [&]
		{
			sampler_cache local(ctx, 1, ledger);
			std::vector<unsigned> registers(0x2000 / 4);
			const unsigned base = vertex_texture_method_base + 3 * vertex_texture_method_stride;
			registers[base + 1] = (3u << 16) | (0x85u << 8) | (2u << 4) | 1;
			registers[base + 2] = 0x0404;
			registers[base + 3] = 0x80000000u | (512u << 7);
			registers[base + 4] = 16;
			registers[base + 6] = (4u << 16) | 4;
			registers[base + 7] = 0xff404080;
			vertex_inputs input;
			vertex_program_options vo;
			vo.point_output = true;
			apply_vertex_texture_snapshot(vertex_texture_snapshot::capture(registers, 8), [&](unsigned, const fragment_texture_registers&)
										  { texture_cache_state metadata;metadata.actual_mips=3;return resolved_fragment_texture{image,metadata}; }, ctx, local, vo, input);
			require(vo.border_textures == 8 && input.border_coverage[3], "Vertex batch omitted custom border contract");
			constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
			const std::array<std::array<unsigned, 4>, 1> words = {{{(63u << 15) | (7u << 10) | (1u << 30), (25u << 22) | (xyzw >> 9), ((xyzw & 511) << 23) | (3u << 8), (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
			vertex_shader vp(compile, words, vo);
			const auto code = program(1, 0, {.75f, .25f, .5f, 1});
			fragment_shader fp(compile, code);
			render_pipeline_configuration config;
			config.topology = MTLPrimitiveTopologyClassPoint;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			const auto pipeline = make_render_pipeline(compile, config, vp.function(), fp);
			std::array<color, 16> attributes{};
			attributes[0] = {-1, .5f, 0, 0};
			input.attributes = make_upload(ctx, std::as_bytes(std::span(attributes)));
			const auto geometry = convert_array_geometry(guest_topology::points, 0, 1, provoking_vertex::last);
			ctx.begin();
			ctx.wait_event(gate, 1);
			vertex_context uniforms;
			uniforms.vertex_count = 1;
			const auto vb = vp.prepare(ctx, uniforms, geometry, input), fb = fp.prepare(ctx, {}, code, {});
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = target;
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			render_commands commands(ctx, pass);
			raster_state raster;
			raster.viewport = {0, 0, 2, 2, 0, 1};
			raster.scissor = {0, 0, 2, 2};
			commands.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb);
			commands.end();
			texture_region region;
			region.size = MTLSizeMake(2, 2, 1);
			texture_to_buffer(ctx, target, region, output, 0, 256);
			return ctx.submit();
		}();
		require(!pending.ready(), "Custom vertex lifetime draw did not remain pending");
		gate.signaledValue = 1;
		pending.wait();
		for (unsigned y = 0; y < 2; ++y)
			for (unsigned x = 0; x < 2; ++x)
				for (unsigned c = 0; c < 4; ++c)
				{
					const float wanted = x == 1 && y == 0 ? color{.75f, .25f, .5f, 1}[c] : 0;
					require(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c] == wanted, "Custom vertex border did not position point or survive cache destruction");
				}
		std::cout << "Native custom vertex border on unit3 positions exact point;pending GPU retains coverage after sampler/cache/input/shader destruction passed\n";
	}
	void cold_helper_batch()
	{
		const auto input = source(vertex_texture_dimension::cube);
		sampler_cache batch_cache(ctx, 4, ledger);
		const auto depth_source = [&](MTLPixelFormat format)
		{
			auto descriptor = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format size:4 mipmapped:YES];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			return make_texture(ctx, descriptor);
		};
		const auto depth16 = depth_source(MTLPixelFormatDepth16Unorm);
		const auto depth32 = depth_source(MTLPixelFormatDepth32Float);
		const auto collect = [&](border_texture_preparation& work)
		{
			return std::array<id<MTLTexture>, 4>{batch_cache.border_coverage(input, &work),
				batch_cache.border_color_view(input, {1,0,3,2}, &work),
				batch_cache.border_shadow_view(depth16, .5f, &work), batch_cache.border_shadow_view(depth32, .375f, &work)};
		};
		ctx.begin();
		const auto cancelled = ctx.recording_submission();
		border_texture_preparation abandoned;
		const auto old = collect(abandoned);
		require(cancelled.same_batch(ctx.recording_submission()), "Cold border allocation replaced the caller's recording");
		abandoned.encode(ctx);
		ctx.cancel();
		require(cancelled.cancelled(), "Cancelled helper recording did not report cancellation");

		ctx.begin();
		const auto recording = ctx.recording_submission();
		border_texture_preparation preparation;
		const auto before = ctx.encoding_statistics();
		const auto images = collect(preparation);
		for (std::size_t i = 0; i < old.size(); ++i)
			require(images[i] != old[i], "Cancelled border helper was reused as initialized");
		require(recording.same_batch(ctx.recording_submission()) && ctx.encoding_statistics().checked_dispatches == before.checked_dispatches,
			"Cold border helper encoded or submitted while admitting storage");
		preparation.encode(ctx);
		const auto initialized = ctx.encoding_statistics().checked_dispatches;
		border_texture_preparation repeated;
		require(collect(repeated) == images, "Same recording replaced an encoded border helper");
		repeated.encode(ctx);
		require(initialized == before.checked_dispatches + 6 && ctx.encoding_statistics().checked_dispatches == initialized,
			"Checked coverage and color dispatches repeated within one recording");
		for (unsigned kind = 0; kind < images.size(); ++kind)
			for (unsigned mip = 0; mip < 3; ++mip)
				for (unsigned face = 0; face < 6; ++face)
				{
					texture_region region;
					region.level = mip;
					region.slice = face;
					region.size = MTLSizeMake(1, 1, 1);
					const auto offset = (kind * 18 + mip * 6 + face) * 32;
					const auto pixel_bytes = kind == 1 ? 16u : kind == 2 ? 2u : 4u;
					texture_to_buffer(ctx, images[kind], region, output, offset, pixel_bytes, 0,
						kind < 2 ? texture_plane::all : texture_plane::depth);
				}
		const auto completed = ctx.submit();
		require(completed.same_batch(recording), "Helper initialization and reads used separate submissions");
		completed.wait();
		for (unsigned mip = 0; mip < 3; ++mip)
			for (unsigned face = 0; face < 6; ++face)
			{
				const auto texel = mip_color(mip);
				const auto* base = static_cast<const std::byte*>(output.contents) + (mip * 6 + face) * 32;
				const auto read = [&](unsigned kind, auto wanted)
				{
					decltype(wanted) value;
					std::memcpy(&value, base + kind * 18 * 32, sizeof(value));
					require(value == wanted, "Staged border helper contents differ after retry");
				};
				read(0, 0.f);
				read(1, color{texel[1], texel[0], texel[3], texel[2]});
				read(2, std::uint16_t{32768});
				read(3, .375f);
			}
		border_texture_preparation warm;
		require(collect(warm) == images, "Successful helper images were not reused across recordings");
		std::cout << "Cold border helper batch: 72 exact cube face/mip samples for coverage, float color, D16 and D32; cancellation and same-recording reuse passed\n";
	}
	void all()
	{
		cold_helper_batch();
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::three, vertex_texture_dimension::cube})
		{
			const auto image = source(dimension);
			zero_mips(image);
			for (unsigned op : {23u, 24u, 25u, 47u, 49u})
				for (bool linear : {false, true})
					for (bool remap : {false, true})
						for (float u : {-.19f, -.065f, .03f, .5f, .995f, 1.02f, 1.17f})
						{
							draw(dimension, image, op, 15, {u, .03f, .5f, 1}, linear, remap);
						}
			for (unsigned op : {25u, 47u})
				for (float lod : {0.f, .25f, .75f, 1.25f, 2.f})
					for (unsigned anisotropy : {1u, 4u, 16u})
						draw(dimension, image, op, 0, {.03f, .03f, .5f, 1}, true, false, lod, true, anisotropy);
			draw(dimension, image, 23, 15, {-.19f, .03f, .5f, 1}, true, true, 0, false, 1, true);
		}
		const auto normalized = source(vertex_texture_dimension::two, MTLPixelFormatRGBA8Unorm);
		float_mips(normalized);
		for (unsigned op : {23u, 24u, 25u, 47u, 49u})
			for (bool linear : {false, true})
				for (float u : {-.19f, -.065f, .03f, .5f, .995f, 1.02f, 1.17f})
					draw(vertex_texture_dimension::two, normalized, op, 15, {u, .03f, .5f, 1}, linear, true);
		vertex_and_lifetime(source(vertex_texture_dimension::two));
		const auto tiny = std::make_shared<heap_budget>(ctx, 1);
		sampler_cache constrained(ctx, 1, tiny);
		bool rejected = false;
		try
		{
			(void)constrained.border_coverage(normalized);
		}
		catch (const heap_budget_error&)
		{
			rejected = true;
		}
		require(rejected && tiny->statistics().reserved_bytes == 0, "Border helper bypassed actual heap quota");
		require(ledger->statistics().live_heaps > 0, "Coverage images were not charged in heap ledger");
		std::cout << cases << " native RSX custom border draws:1D/2D/3D/cube,TEX/projected/gradient/LOD/bias,nearest/linear/mips/anisotropy/mixed wraps,view remaps and texture15 with no extra sampler slot passed\n";
	}
};
} // namespace
int main(int argc, char** argv)
{
	@autoreleasepool
	{
		try
		{
			fixture f;
			if (argc == 2 && std::string_view(argv[1]) == "--cold-helpers-only") f.cold_helper_batch();
			else { require(argc == 1, "Unknown border test selector"); f.all(); }
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
