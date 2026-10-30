#include "M4Sampler.h"
#include "M4Pipeline.h"
#include "M4Texture.h"
#include "M4Draw.h"
#include "M4FragmentShader.h"
#include "M4RenderPipeline.h"
#include "M4Sampling.h"

#include <bit>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace
{
using namespace metal4;
using color = std::array<float, 4>;
struct alignas(16) request
{
	color uv_lod_reference;
	std::uint32_t explicit_lod = 1;
	std::uint32_t reserved[3]{};
	sampler_parameters sampling{};
};
static_assert(sizeof(request) == 64);
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void descriptor_cache(context& ctx)
{
	sampler_cache cache(ctx, 2);
	guest_sampler_state state;
	state.min_filter = sampler_min_filter::nearest_linear;
	state.max_lod = 6;
	sampler_texture_policy policy;
	policy.requested_mips = policy.actual_mips = 7;
	const auto first = cache.acquire(state, policy);
	for (int i = -64; i <= 63; ++i)
	{
		state.bias = i * 0.25f;
		const auto bound = cache.acquire(state, policy);
		require(bound.sampler == first.sampler && bound.parameters.bias == state.bias &&
			bound.parameters.min_lod == 0 && bound.parameters.max_lod == 6 && bound.parameters.flags == 0,
			"Shader-only bias created a sampler or reused stale parameters");
	}
	// Compare Never is also the native descriptor for a non-shadow binding.
	state.depth_compare = true;
	state.compare_function = 0x200;
	for (const bool floating : {false, true, false})
	{
		state.depth_float = floating;
		const auto bound = cache.acquire(state, policy);
		require(bound.sampler == first.sampler && bound.parameters.bias == state.bias &&
			bound.parameters.flags == (sampler_depth_compare | (floating ? sampler_depth_float : 0)),
			"Shader depth policy was lost or duplicated an identical native descriptor");
	}
	require(cache.statistics().created == 1 && cache.statistics().reused == 131,
		"Shader-only state caused extra backend sampler creation requests");
	state.compare_function = 0x201;
	const auto compare = cache.acquire(state, policy);
	require(cache.statistics().created == 2 && compare.sampler != first.sampler,
		"Changed native compare function reused an incompatible sampler");
	state.depth_float = !state.depth_float;
	require(cache.acquire(state, policy).sampler == compare.sampler && cache.statistics().created == 2,
		"Depth codec split a native comparison sampler");
	state.min_lod = 1;
	const auto clamped = cache.acquire(state, policy);
	require(cache.statistics().created == 3 && cache.statistics().evicted == 1 &&
		cache.statistics().entries == 2 && clamped.parameters.min_lod == 1,
		"Native LOD clamp was omitted from the bounded sampler cache key");
	state.min_lod = 0;
	require(cache.acquire(state, policy).sampler == compare.sampler,
		"Sampler descriptor LRU evicted the recently used comparison sampler");
	std::cout << "128 shader-only bias values and depth policies share one native sampler; compare/clamp keys and LRU remain distinct\n";
}
float addressed(float u, sampler_wrap wrap)
{
	if (wrap == sampler_wrap::repeat)
		return u - std::floor(u);
	if (wrap == sampler_wrap::mirror)
	{
		u -= 2 * std::floor(u / 2);
		return u <= 1 ? u : 2 - u;
	}
	if (wrap == sampler_wrap::mirror_once_edge || wrap == sampler_wrap::mirror_once_border || wrap == sampler_wrap::mirror_once_clamp)
		return std::abs(u);
	return u;
}
color texel(int x, sampler_wrap mode, color border)
{
	if (mode == sampler_wrap::border && (x < 0 || x >= 4))
		return border;
	if (mode == sampler_wrap::repeat)
		x = (x % 4 + 4) % 4;
	x = std::clamp(x, 0, 3);
	return {float(x * 64) / 255, 64.f / 255, 32.f / 255, 1};
}
color spatial(float u, sampler_wrap wrap, bool linear, color border)
{
	u = addressed(u, wrap);
	if (!linear)
		return texel(int(std::floor(u * 4)), wrap, border);
	const float position = u * 4 - .5f;
	const int first = int(std::floor(position));
	const float weight = position - first;
	const auto a = texel(first, wrap, border), b = texel(first + 1, wrap, border);
	color out;
	for (unsigned i = 0; i < 4; ++i)
		out[i] = a[i] * (1 - weight) + b[i] * weight;
	return out;
}
void encode(context& ctx, id<MTLComputePipelineState> pipeline, id<MTLTexture> texture, id<MTLSamplerState> sampler, id<MTLBuffer> input, id<MTLBuffer> output, std::size_t count)
{
	bindings bound;
	bound.set_texture(0, texture);
	bound.set_sampler(0, sampler);
	bound.set_buffer(0, input);
	bound.set_buffer(1, output);
	ctx.keep_alive(pipeline);
	const auto table = bound.snapshot(ctx);
	const auto encoder = ctx.compute();
	ctx.barrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
	[encoder setComputePipelineState:pipeline];
	[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<std::size_t>(count, pipeline.threadExecutionWidth), 1, 1)];
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx(1);
			descriptor_cache(ctx);
			compiler compile(ctx);
			sampler_cache cache(ctx, 4);
			const auto library = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
struct sampling { float min_lod; float max_lod; float bias; uint reserved; float4 border_adjust; };
struct request { float4 uv_lod_reference; uint explicit_lod; uint reserved[3]; sampling state; };
kernel void color_sample(const device request* input [[buffer(0)]], device float4* output [[buffer(1)]],
    texture2d<float> image [[texture(0)]], sampler state [[sampler(0)]], uint i [[thread_position_in_grid]]) {
    const auto p=input[i];
    const float d=exp2(p.uv_lod_reference.z+p.state.bias);
    output[i]=p.explicit_lod ? image.sample(state,p.uv_lod_reference.xy,level(clamp(p.uv_lod_reference.z+p.state.bias,p.state.min_lod,p.state.max_lod))) :
        image.sample(state,p.uv_lod_reference.xy,gradient2d(float2(d/image.get_width(),0),float2(0,d/image.get_height())));
}
kernel void depth_sample(const device request* input [[buffer(0)]], device float4* output [[buffer(1)]],
    depth2d<float> image [[texture(0)]], sampler state [[sampler(0)]], uint i [[thread_position_in_grid]]) {
    output[i]=float4(image.sample_compare(state,input[i].uv_lod_reference.xy,input[i].uv_lod_reference.w,level(0)));
}
struct raster_sample { float4 position [[position]]; float2 uv; };
vertex raster_sample mip_vertex(uint id [[vertex_id]], constant request& parameters [[buffer(0)]]) {
    const float2 positions[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
    const float2 position=positions[id];
    return {float4(position,0,1), (position+1)*0.5f*4*exp2(parameters.uv_lod_reference.z)/64};
}
fragment float4 mip_fragment(raster_sample in [[stage_in]], texture2d<float> image [[texture(0)]],sampler state [[sampler(0)]], constant request& parameters [[buffer(0)]]) {
    return image.sample(state,in.uv,bias(parameters.state.bias));
}
)metal",
												 @"Native RSX sampler regression");
			auto pd = [MTL4ComputePipelineDescriptor new];
			pd.computeFunctionDescriptor = library_function(library, @"color_sample");
			const auto color_pipeline = compile.compute(pd);
			pd.computeFunctionDescriptor = library_function(library, @"depth_sample");
			const auto depth_pipeline = compile.compute(pd);
			auto render_descriptor = [MTL4RenderPipelineDescriptor new];
			render_descriptor.vertexFunctionDescriptor = library_function(library, @"mip_vertex");
			render_descriptor.fragmentFunctionDescriptor = library_function(library, @"mip_fragment");
			render_descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
			render_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
			const auto mip_pipeline = compile.render(render_descriptor);
			auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			td.usage = MTLTextureUsageShaderRead;
			const auto image = make_texture(ctx, td);
			std::vector<std::byte> bytes(64);
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 4; ++x)
				{
					const unsigned offset = (y * 4 + x) * 4;
					bytes[offset] = std::byte(x * 64);
					bytes[offset + 1] = std::byte(y * 64);
					bytes[offset + 2] = std::byte(32);
					bytes[offset + 3] = std::byte(255);
				}
			ctx.begin();
			texture_region whole;
			whole.size = MTLSizeMake(4, 4, 1);
			buffer_to_texture(ctx, make_upload(ctx, bytes), 0, image, whole, 16);
			ctx.submit().wait();
			td.width = td.height = 64;
			td.mipmapLevelCount = 7;
			const auto mip_image = make_texture(ctx, td);
			ctx.begin();
			for (unsigned level = 0; level < 7; ++level)
			{
				const unsigned size = 64 >> level;
				std::vector<std::byte> pixels(size * size * 4);
				for (unsigned p = 0; p < size * size; ++p)
				{
					pixels[p * 4] = std::byte(level * 32);
					pixels[p * 4 + 1] = std::byte(255);
					pixels[p * 4 + 3] = std::byte(255);
				}
				texture_region region;
				region.level = level;
				region.size = MTLSizeMake(size, size, 1);
				buffer_to_texture(ctx, make_upload(ctx, pixels), 0, mip_image, region, size * 4);
			}
			ctx.submit().wait();
			td.width = td.height = 4;
			td.mipmapLevelCount = 1;
			td.pixelFormat = MTLPixelFormatDepth32Float;
			td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto depth = make_texture(ctx, td);
			ctx.begin();
			auto clear = [MTL4RenderPassDescriptor new];
			clear.depthAttachment.texture = depth;
			clear.depthAttachment.loadAction = MTLLoadActionClear;
			clear.depthAttachment.clearDepth = .5;
			clear.depthAttachment.storeAction = MTLStoreActionStore;
			(void)ctx.render(clear);
			ctx.submit().wait();
			td.pixelFormat = MTLPixelFormatRGBA32Float;
			td.usage = MTLTextureUsageRenderTarget;
			const auto mip_target = make_texture(ctx, td);
			unsigned cases = 0;
			float max_trilinear_lod_error = 0;
			auto run = [&](const guest_sampler_state& state, const sampler_texture_policy& policy, id<MTLTexture> texture, std::span<const request> input, std::span<const color> expected, bool compare = false, float tolerance = 2e-6f)
			{
				const auto sampler = cache.acquire(state, policy);
				std::vector<request> requests(input.begin(), input.end());
				for (auto& value : requests)
					value.sampling = sampler.parameters;
				const auto data = make_upload(ctx, std::as_bytes(std::span(requests)));
				const auto output = make_buffer(ctx, input.size() * 16, buffer_storage::shared);
				ctx.begin();
				encode(ctx, compare ? depth_pipeline : color_pipeline, texture, sampler.sampler, data, output, input.size());
				ctx.submit().wait();
				const auto actual = static_cast<const float*>(output.contents);
				for (unsigned i = 0; i < input.size(); ++i)
					for (unsigned c = 0; c < 4; ++c)
						if (!std::isfinite(actual[i * 4 + c]) || std::abs(actual[i * 4 + c] - expected[i][c]) > tolerance)
							throw std::runtime_error("Native sampler differs at case " + std::to_string(cases + i) + " component " + std::to_string(c) + ": actual=" + std::to_string(actual[i * 4 + c]) + " expected=" + std::to_string(expected[i][c]));
				if (tolerance > 2e-6f)
					for (unsigned i = 0; i < input.size(); ++i)
						max_trilinear_lod_error = std::max(max_trilinear_lod_error, std::abs(actual[i * 4] - expected[i][0]) * 255 / 32);
				cases += input.size();
			};
			const std::array coordinates{-2.125f, -1.125f, -.375f, -.125f, .125f, .375f, .625f, .875f, 1.125f, 1.375f, 1.875f, 2.125f, -.0625f, 0.f, .25f, .5f, 1.f, 1.0625f};
			for (const auto wrap : {sampler_wrap::repeat, sampler_wrap::mirror, sampler_wrap::edge, sampler_wrap::border, sampler_wrap::clamp, sampler_wrap::mirror_once_edge, sampler_wrap::mirror_once_border, sampler_wrap::mirror_once_clamp})
				for (const bool linear : {false, true})
					for (const color border : {color{0, 0, 0, 0}, color{0, 0, 0, 1}, color{1, 1, 1, 1}})
					{
						guest_sampler_state state;
						state.wrap = {wrap, sampler_wrap::edge, sampler_wrap::edge};
						state.border_color = border;
						state.min_filter = linear ? sampler_min_filter::linear : sampler_min_filter::nearest;
						state.mag_filter = linear ? sampler_mag_filter::linear : sampler_mag_filter::nearest;
						std::vector<request> input;
						std::vector<color> expected;
						for (float u : coordinates)
						{
							input.push_back({{u, .375f, 0, 0}});
							expected.push_back(spatial(u, wrap, linear, border));
						}
						run(state, {}, image, input, expected);
					}
			sampler_texture_policy mip_policy;
			mip_policy.requested_mips = mip_policy.actual_mips = 7;
			// One native object, two immutable parameter snapshots in the same
			// pending command. Eviction/cache destruction must preserve both jobs.
			{
				const auto gate = [ctx.device() newSharedEvent];
				struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 1; } } guard{gate};
				std::array<id<MTLBuffer>, 2> output{
					make_buffer(ctx, 16, buffer_storage::shared), make_buffer(ctx, 16, buffer_storage::shared)};
				auto work = [&]
				{
					sampler_cache local(ctx, 1);
					guest_sampler_state state;
					state.min_filter = sampler_min_filter::nearest_nearest;
					state.max_lod = 6;
					const auto first = local.acquire(state, mip_policy);
					ctx.begin();
					ctx.wait_event(gate, 1);
					for (unsigned i = 0; i < 2; ++i)
					{
						state.bias = float(i * 2);
						const auto bound = local.acquire(state, mip_policy);
						require(bound.sampler == first.sampler, "Bias changed the native descriptor identity");
						request input{{.375f, .375f, 0, 0}};
						input.sampling = bound.parameters;
						encode(ctx, color_pipeline, mip_image, bound.sampler,
							make_upload(ctx, std::as_bytes(std::span(&input, 1))), output[i], 1);
					}
					require(local.statistics().created == 1, "Bias batch created duplicate samplers");
					state.max_lod = 3;
					(void)local.acquire(state, mip_policy);
					require(local.statistics().evicted == 1, "Pending sampler was not evicted for the lifetime check");
					return ctx.submit();
				}();
				require(!work.ready(), "Sampler parameter snapshots were not held pending");
				gate.signaledValue = 1;
				work.wait();
				for (unsigned i = 0; i < 2; ++i)
					require(std::abs(static_cast<const float*>(output[i].contents)[0] - float(i * 64) / 255) <= 2e-6f,
						"Shared native sampler lost per-binding bias after cache destruction");
				std::cout << "Pending shared sampler preserves separate shader biases after eviction/cache destruction\n";
			}
			for (const bool trilinear : {false, true})
				for (const float bias : {-1.f, -.5f, 0.f, .5f, 1.f})
					for (const auto clamps : {std::array{0.f, 6.f}, std::array{2.f, 4.f}, std::array{1.25f, 3.25f}})
					{
						guest_sampler_state state;
						state.min_filter = trilinear ? sampler_min_filter::nearest_linear : sampler_min_filter::nearest_nearest;
						state.min_lod = clamps[0];
						state.max_lod = clamps[1];
						state.bias = bias;
						std::vector<request> input;
						std::vector<color> expected;
						for (float lod : {-2.f, .25f, .75f, 1.25f, 2.25f, 5.25f, 7.25f})
						{
							input.push_back({{.375f, .375f, lod, 0}, 0});
							const float level = std::clamp(lod + bias, clamps[0], clamps[1]);
							expected.push_back({(trilinear ? level : std::floor(level + .5f)) * 32 / 255, 1, 0, 1});
						}
						const auto sampler = cache.acquire(state, mip_policy);
						for (unsigned i = 0; i < input.size(); ++i)
						{
							auto current = input[i];
							current.sampling = sampler.parameters;
							const auto parameters = make_upload(ctx, std::as_bytes(std::span(&current, 1)));
							const auto output = make_buffer(ctx, 256 * 4, buffer_storage::shared);
							ctx.begin();
							auto pass = [MTL4RenderPassDescriptor new];
							pass.colorAttachments[0].texture = mip_target;
							pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
							pass.colorAttachments[0].storeAction = MTLStoreActionStore;
							bindings vertex, fragment;
							vertex.set_buffer(0, parameters);
							fragment.set_texture(0, mip_image);
							fragment.set_sampler(0, sampler.sampler);
							fragment.set_buffer(0, parameters);
							render_commands draw(ctx, pass);
							raster_state raster;
							raster.viewport = {0, 0, 4, 4, 0, 1};
							raster.scissor = {0, 0, 4, 4};
							draw.configure(mip_pipeline, nil, raster, vertex, fragment);
							draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
							draw.end();
							texture_region region;
							region.size = MTLSizeMake(4, 4, 1);
							texture_to_buffer(ctx, mip_target, region, output, 0, 256);
							ctx.submit().wait();
							for (unsigned y = 0; y < 4; ++y)
								for (unsigned x = 0; x < 4; ++x)
									for (unsigned c = 0; c < 4; ++c)
										if (!std::isfinite(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c]) || std::abs(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c] - expected[i][c]) > (trilinear ? 32.f / (255 * 64) + 2e-6f : 2e-6f))
											throw std::runtime_error("Implicit native sampler LOD/bias differs: bias=" + std::to_string(bias) + " lod=" + std::to_string(input[i].uv_lod_reference[2]) + " clampMin=" + std::to_string(clamps[0]) + " actual=" + std::to_string(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c]) + " expected=" + std::to_string(expected[i][c]));
							if (trilinear)
								max_trilinear_lod_error = std::max(max_trilinear_lod_error, std::abs(static_cast<const float*>(output.contents)[0] - expected[i][0]) * 255 / 32);
							++cases;
						}
						run(state, mip_policy, mip_image, input, expected, false, trilinear ? 32.f / (255 * 64) + 2e-6f : 2e-6f);
						for (auto& value : input)
							value.explicit_lod = 1;
						run(state, mip_policy, mip_image, input, expected);
					}
			for (unsigned op = 0; op < 8; ++op)
			{
				guest_sampler_state state;
				state.depth_compare = true;
				state.compare_function = 0x200 + op;
				std::vector<request> input;
				std::vector<color> expected;
				for (float reference : {.25f, .5f, .75f})
				{
					input.push_back({{.375f, .375f, 0, reference}});
					const bool passes = op == 0 ? false : op == 1 ? .5f < reference
													  : op == 2	  ? .5f == reference
													  : op == 3	  ? .5f <= reference
													  : op == 4	  ? .5f > reference
													  : op == 5	  ? .5f != reference
													  : op == 6	  ? .5f >= reference
																  : true;
					const float value = passes;
					expected.push_back({value, value, value, value});
				}
				run(state, {}, depth, input, expected, true);
			}
			// An actual translated RSX TEXLOD uses the same metadata contract,
			// including sampler bias and clamps ignored by native explicit levels.
			std::vector<std::byte> code;
			for (unsigned word : {(47u << 24) | (15u << 9) | 1u, (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, (0xe4u << 9) | 2u, 0xe4u << 9})
				for (unsigned shift : {8u, 0u, 24u, 16u})
					code.push_back(static_cast<std::byte>(word >> shift));
			for (float value : {.375f, .375f, 0.f, 1.f})
				for (unsigned shift : {8u, 0u, 24u, 16u})
					code.push_back(static_cast<std::byte>(std::bit_cast<unsigned>(value) >> shift));
			fragment_shader translated(compile, code);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			const auto rsx_pipeline = make_render_pipeline(compile, config, library_function(library, @"mip_vertex"), translated);
			const auto geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
			for (const auto lod_state : {std::array{-1.f, 2.f, 4.f}, std::array{2.f, 0.f, 6.f}, std::array{4.f, 0.f, 3.f}})
			{
				guest_sampler_state state;
				state.min_filter = sampler_min_filter::nearest_nearest;
				state.bias = lod_state[0];
				state.min_lod = lod_state[1];
				state.max_lod = lod_state[2];
				fragment_inputs inputs;
				inputs.textures[0] = mip_image;
				inputs.set_sampler(0, cache.acquire(state, mip_policy));
				const request vertex_state{{0, 0, 0, 0}};
				const auto parameters = make_upload(ctx, std::as_bytes(std::span(&vertex_state, 1)));
				const auto output = make_buffer(ctx, 256 * 4, buffer_storage::shared);
				ctx.begin();
				const auto fb = translated.prepare(ctx, {}, code, inputs);
				bindings vb;
				vb.set_buffer(0, parameters);
				auto pass = [MTL4RenderPassDescriptor new];
				pass.colorAttachments[0].texture = mip_target;
				pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				render_commands draw(ctx, pass);
				raster_state raster;
				raster.viewport = {0, 0, 4, 4, 0, 1};
				raster.scissor = {0, 0, 4, 4};
				draw.draw_geometry(geometry, rsx_pipeline.pipeline, rsx_pipeline.depth_stencil, raster, vb, fb);
				draw.end();
				texture_region region;
				region.size = MTLSizeMake(4, 4, 1);
				texture_to_buffer(ctx, mip_target, region, output, 0, 256);
				ctx.submit().wait();
				const float expected = std::floor(std::clamp(.375f + lod_state[0], lod_state[1], lod_state[2]) + .5f) * 32 / 255;
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
						require(std::isfinite(static_cast<const float*>(output.contents)[y * 64 + x * 4]) && std::abs(static_cast<const float*>(output.contents)[y * 64 + x * 4] - expected) <= 2e-6f, "Translated RSX TEXLOD lost sampler shader bias/clamp metadata");
				++cases;
			}
			std::array<texture_coordinates, 6> transforms{};
			transforms[1].scale = {4, 4, 1};
			transforms[1].bias = {-1, -1, 0};
			transforms[2].scale = {-1, 1, 1};
			transforms[2].bias = {1, 0, 0};
			transforms[3].scale = {1, 2, 1};
			transforms[3].bias = {0, -.125f, 0};
			transforms[4].flags = texture_coordinate_clamp;
			transforms[4].minimum = {.6f, .1f};
			transforms[4].maximum = {.8f, .2f};
			transforms[5] = normalize_guest_pixels(4, 4);
			for (unsigned transform = 0; transform < transforms.size(); ++transform)
			{
				const float coordinate = transform == 5 ? 1.5f : .375f;
				std::vector<std::byte> current(code.begin(), code.begin() + 16);
				for (float value : {coordinate, coordinate, 0.f, 1.f})
					for (unsigned shift : {8u, 0u, 24u, 16u})
						current.push_back(static_cast<std::byte>(std::bit_cast<unsigned>(value) >> shift));
				guest_sampler_state state;
				state.wrap = {sampler_wrap::edge, sampler_wrap::edge, sampler_wrap::edge};
				fragment_inputs inputs;
				inputs.textures[0] = image;
				inputs.set_sampler(0, cache.acquire(state));
				inputs.coordinates[0] = transforms[transform];
				const request vertex_state{{0, 0, 0, 0}};
				const auto parameters = make_upload(ctx, std::as_bytes(std::span(&vertex_state, 1)));
				const auto output = make_buffer(ctx, 256 * 4, buffer_storage::shared);
				ctx.begin();
				const auto fb = translated.prepare(ctx, {}, current, inputs);
				bindings vb;
				vb.set_buffer(0, parameters);
				auto pass = [MTL4RenderPassDescriptor new];
				pass.colorAttachments[0].texture = mip_target;
				pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				render_commands draw(ctx, pass);
				raster_state raster;
				raster.viewport = {0, 0, 4, 4, 0, 1};
				raster.scissor = {0, 0, 4, 4};
				draw.draw_geometry(geometry, rsx_pipeline.pipeline, rsx_pipeline.depth_stencil, raster, vb, fb);
				draw.end();
				texture_region region;
				region.size = MTLSizeMake(4, 4, 1);
				texture_to_buffer(ctx, mip_target, region, output, 0, 256);
				ctx.submit().wait();
				float u = std::fma(coordinate, transforms[transform].scale[0], transforms[transform].bias[0]), v = std::fma(coordinate, transforms[transform].scale[1], transforms[transform].bias[1]);
				if (transforms[transform].flags & texture_coordinate_clamp)
				{
					u = std::clamp(u, transforms[transform].minimum[0], transforms[transform].maximum[0]);
					v = std::clamp(v, transforms[transform].minimum[1], transforms[transform].maximum[1]);
				}
				const color expected{float(std::clamp(int(std::floor(u * 4)), 0, 3) * 64) / 255, float(std::clamp(int(std::floor(v * 4)), 0, 3) * 64) / 255, 32.f / 255, 1};
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
						for (unsigned c = 0; c < 4; ++c)
							require(std::isfinite(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c]) && std::abs(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c] - expected[c]) <= 2e-6f, "Translated ordinary RSX texture lost coordinate transform/normalization/clamp metadata");
				++cases;
			}
			guest_sampler_state packed_state;
			packed_state.wrap[0] = sampler_wrap::border;
			const auto packed_binding = cache.acquire(packed_state);
			for (const auto format : {MTLPixelFormatA1BGR5Unorm, MTLPixelFormatABGR4Unorm})
			{
				td.pixelFormat = format;
				td.usage = MTLTextureUsageShaderRead;
				const auto packed = make_texture(ctx, td);
				bool refused = false;
				try
				{
					validate_sampled_texture(ctx, packed, packed_binding.sampler, vertex_texture_dimension::two, packed_binding.parameters);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused, "Incompatible packed-format opaque-black border was not rejected");
			}
			// Between texel centers: accidentally using the replacement linear
			// sampler would return32/255 instead of the retained nearest64/255.
			const request point{{.25f, .375f, 0, 0}};
			bool descriptor_bias_refused = false;
			auto bad_descriptor = [MTLSamplerDescriptor new];
			bad_descriptor.lodBias = 1;
			try
			{
				(void)make_sampler(ctx, bad_descriptor);
			}
			catch (const std::invalid_argument&)
			{
				descriptor_bias_refused = true;
			}
			require(descriptor_bias_refused, "Unverified native descriptor bias was silently accepted");
			const color expected_point{64.f / 255, 64.f / 255, 32.f / 255, 1};
			const auto output = make_buffer(ctx, 16, buffer_storage::shared);
			const auto input = make_upload(ctx, std::as_bytes(std::span(&point, 1)));
			id<MTLSharedEvent> gate = [ctx.device() newSharedEvent];
			struct release_gate
			{
				id<MTLSharedEvent> event;
				~release_gate()
				{
					event.signaledValue = 1;
				}
			} release{gate};
			submission blocked = [&]
			{
				sampler_cache transient(ctx, 1);
				guest_sampler_state state;
				const auto first = transient.acquire(state);
				require(transient.acquire(state).sampler == first.sampler && transient.statistics().reused == 1, "Sampler cache failed reuse");
				ctx.begin();
				ctx.wait_event(gate, 1);
				encode(ctx, color_pipeline, image, first.sampler, input, output, 1);
				state.mag_filter = sampler_mag_filter::linear;
				(void)transient.acquire(state);
				require(transient.statistics().entries == 1 && transient.statistics().created == 2 && transient.statistics().evicted == 1, "Sampler cache did not enforce its LRU capacity");
				return ctx.submit();
			}();
			bool timeout = false;
			try
			{
				blocked.wait(std::chrono::milliseconds(1));
			}
			catch (const std::runtime_error&)
			{
				timeout = true;
			}
			require(timeout && !blocked.ready(), "Sampler lifetime test did not actually block submitted GPU work");
			gate.signaledValue = 1;
			blocked.wait();
			for (unsigned i = 0; i < 4; ++i)
				require(std::abs(static_cast<const float*>(output.contents)[i] - expected_point[i]) <= 2e-6f, "Evicted sampler died with its cache before native GPU use");
			ctx.wait_idle();
			std::cout << "Packed A1BGR5/ABGR4 opaque-black borders and unverified native descriptor bias are rejected before GPU sampling\n";
			std::cout << "Maximum measured trilinear derivative LOD error: " << max_trilinear_lod_error << " (limit1/64); explicit levels use2e-6 channel tolerance\n";
			std::cout << cases << " native sampler values: eight wrap states, nearest/linear, canonical borders, implicit-fragment mip/LOD/bias, reversed depth compares and evicted/destroyed-cache timeout lifetime passed\n";
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
