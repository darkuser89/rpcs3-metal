#include "M4FragmentShader.h"
#include "M4TextureStateBinding.h"
#include "M4VertexTextureBatch.h"

#include <algorithm>
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
const auto identity = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
const auto remap = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue, MTLTextureSwizzleRed, MTLTextureSwizzleOne, MTLTextureSwizzleZero);

id<MTLTexture> source(context& ctx, MTLTextureType type = MTLTextureType2DArray)
{
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:YES];
	descriptor.textureType = type;
	descriptor.arrayLength = type == MTLTextureType2DArray ? 2 : 1;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	const auto texture = make_texture(ctx, descriptor);
	ctx.begin();
	for (unsigned slice = 0; slice < (type == MTLTextureTypeCube ? 6 : descriptor.arrayLength); ++slice)
		for (unsigned mip = 0; mip < texture.mipmapLevelCount; ++mip)
		{
			const auto width = 8u >> mip;
			std::vector<std::uint8_t> pixels(width * width * 4);
			for (unsigned i = 0; i < width * width; ++i)
			{
				pixels[i * 4] = 20 + slice * 10 + mip * 40;
				pixels[i * 4 + 1] = 40;
				pixels[i * 4 + 2] = 80;
				pixels[i * 4 + 3] = 200;
			}
			texture_region region;
			region.slice = slice;
			region.level = mip;
			region.size = MTLSizeMake(width, width, 1);
			buffer_to_texture(ctx, make_upload(ctx, std::as_bytes(std::span(pixels))), 0, texture, region, width * 4);
		}
	ctx.submit().wait();
	return texture;
}

id<MTLComputePipelineState> pipeline(context& ctx)
{
	compiler compile(ctx);
	auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(compile.library(R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void inspect(texture2d<float> image [[texture(0)]], device uint4* out [[buffer(0)]]) {
    out[0] = uint4(round(image.read(uint2(0)) * 255.0f));
}
)msl"), @"inspect");
	return compile.compute(descriptor);
}
void encode(context& ctx, id<MTLComputePipelineState> program, id<MTLTexture> view, id<MTLBuffer> output)
{
	bindings bound;
	bound.set_texture(0, view);
	bound.set_buffer(0, output);
	ctx.keep_alive(program);
	const auto table = bound.snapshot(ctx);
	const auto encoder = ctx.compute();
	[encoder setComputePipelineState:program];
	[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}
void key_and_sampling(context& ctx)
{
	texture_view_cache cache(ctx, 32);
	const auto image = source(ctx);
	const auto program = pipeline(ctx);
	const auto output = make_buffer(ctx, 16, buffer_storage::shared);
	for (unsigned slice = 0; slice < 2; ++slice)
		for (unsigned mip = 0; mip < 3; ++mip)
			for (const bool swizzled : {false, true})
			{
				const auto channels = swizzled ? remap : identity;
				const auto view = cache.get(image, image.pixelFormat, MTLTextureType2D, NSMakeRange(mip, 1), NSMakeRange(slice, 1), channels);
				require(cache.get(image, image.pixelFormat, MTLTextureType2D, NSMakeRange(mip, 1), NSMakeRange(slice, 1), channels) == view,
					"Repeated native view descriptor did not reuse its object");
				ctx.begin();
				encode(ctx, program, view, output);
				ctx.submit().wait();
				const auto red = 20 + slice * 10 + mip * 40;
				const std::array<std::uint32_t, 4> expected = swizzled ? std::array{80u, red, 255u, 0u} : std::array{red, 40u, 80u, 200u};
				require(std::equal(expected.begin(), expected.end(), static_cast<const std::uint32_t*>(output.contents)),
					"Cached view lost its mip/slice/swizzle identity");
			}
	require(cache.statistics().created == 12 && cache.statistics().reused == 12, "Repeated views issued extra native creation requests");
	const auto original = cache.get(image, image.pixelFormat, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
	const auto srgb = cache.get(image, MTLPixelFormatRGBA8Unorm_sRGB, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
	require(srgb != original && srgb.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB, "View format was omitted from its cache key");
	const auto array = cache.get(image, image.pixelFormat, MTLTextureType2DArray, NSMakeRange(0, 2), NSMakeRange(0, 2), identity);
	require(array.textureType == MTLTextureType2DArray && array.mipmapLevelCount == 2 && array.arrayLength == 2,
		"View type or range length was omitted from its key");
	const auto nested = cache.get(original, original.pixelFormat, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), remap);
	require(cache.get(original, original.pixelFormat, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), remap) == nested,
		"View of view was not cached against its exact source");
	const auto other = source(ctx);
	const auto fresh = cache.get(other, other.pixelFormat, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
	require(fresh != original, "New source allocation reused an old native view");
	std::cout << "Native view reuse preserves exact source, format, type, mip/slice ranges and swizzle; 12 GPU values passed\n";
}
void bounds_and_bindings(context& ctx)
{
	const auto image = source(ctx, MTLTextureType2D);
	texture_view_cache cache(ctx, 2);
	const auto get = [&](unsigned mip) { return cache.get(image, image.pixelFormat, MTLTextureType2D, NSMakeRange(mip, 1), NSMakeRange(0, 1), identity); };
	const auto a = get(0), b = get(1);
	require(get(0) == a, "View LRU hit changed identity");
	(void)get(2);
	require(cache.statistics().created == 3 && cache.statistics().entries == 2 && cache.statistics().evicted == 1 && get(0) == a,
		"View LRU did not preserve its recent entry within capacity");
	(void)get(1);
	require(cache.statistics().created == 4 && bool(b), "Evicted view owner did not survive or its key remained cached");
	const auto before = cache.statistics();
	bool invalid = false;
	try { (void)get(9); } catch (const std::out_of_range&) { invalid = true; }
	require(invalid && cache.statistics().created == before.created && cache.statistics().evicted == before.evicted,
		"Invalid view request evicted a valid entry");
	cache.clear();
	require(cache.statistics().entries == 0, "Cleared view cache retained entries");
	texture_view_cache bypass(ctx, 0);
	for (unsigned i = 0; i < 2; ++i)
		(void)bypass.get(image, image.pixelFormat, image.textureType, NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
	require(bypass.statistics().entries == 0 && bypass.statistics().created == 2 && bypass.statistics().bypassed == 2,
		"Disabled view cache retained entries");
	sampler_cache samplers(ctx, 4);
	decoded_texture_state state;
	state.enabled = true;
	state.view_channels = {2, 0, 5, 4};
	fragment_inputs inputs;
	inputs.set_texture_state(0, image, samplers, state);
	const auto first = inputs.textures[0];
	inputs.set_texture_state(1, image, samplers, state);
	require(inputs.textures[1] == first && samplers.texture_views().statistics().created == 1 && samplers.texture_views().statistics().reused == 1,
		"Productive register binding did not use its view cache");
	state.view_channels = {0, 1, 2, 3};
	inputs.set_texture_state(2, image, samplers, state);
	require(inputs.textures[2] == image && samplers.texture_views().statistics().created == 1, "Identity remap unnecessarily created a native view");
	state.depth = true;
	state.view_channels = {2, 0, 5, 4};
	require(make_register_texture_view(image, state, &samplers.texture_views()) == image, "Shader depth remap created a native swizzle view");
	std::vector<std::uint32_t> registers(0x1000 / 4);
	const auto base = vertex_texture_method_base;
	registers[base + 1] = 1 | (2u << 4) | (0xa5u << 8) | (4u << 16);
	registers[base + 2] = 1 | (1u << 8);
	registers[base + 3] = 0x80000000u;
	registers[base + 4] = 32;
	registers[base + 6] = (8u << 16) | 8;
	const auto snapshot = vertex_texture_snapshot::capture(registers, 1);
	vertex_program_options options;
	vertex_inputs vertex;
	const auto resolve = [&](unsigned, const fragment_texture_registers&)
	{
		texture_cache_state metadata;
		metadata.actual_mips = 4;
		metadata.native_argb_channels = {4, 2, 0, 5};
		return resolved_fragment_texture{image, metadata};
	};
	const auto prior = samplers.texture_views().statistics();
	for (unsigned i = 0; i < 2; ++i)
		apply_vertex_texture_snapshot(snapshot, resolve, ctx, samplers, options, vertex);
	require(vertex.textures[0] == first && samplers.texture_views().statistics().created == prior.created &&
		samplers.texture_views().statistics().reused == prior.reused + 2,
		"Productive vertex register remap did not share the existing native view");
	const auto cube = source(ctx, MTLTextureTypeCube);
	state.depth = false;
	const auto cube_view = make_register_texture_view(cube, state, &samplers.texture_views());
	require(cube_view.textureType == MTLTextureTypeCube && make_register_texture_view(cube, state, &samplers.texture_views()) == cube_view,
		"Register cube remap did not retain its six-face view");
	std::cout << "Bounded LRU, failure isolation, zero-capacity bypass, fragment/vertex register bindings, identity/depth bypass and cube remap passed\n";
}
void pending_eviction(context& ctx)
{
	const auto program = pipeline(ctx);
	const auto output = make_buffer(ctx, 16, buffer_storage::shared);
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 1; } } guard{gate};
	std::optional<submission> pending;
	@autoreleasepool
	{
		const auto image = source(ctx);
		texture_view_cache cache(ctx, 1);
		const auto view = cache.get(image, image.pixelFormat, MTLTextureType2D, NSMakeRange(1, 1), NSMakeRange(1, 1), remap);
		ctx.begin();
		ctx.wait_event(gate, 1);
		encode(ctx, program, view, output);
		(void)cache.get(image, image.pixelFormat, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
		require(cache.statistics().evicted == 1, "Pending view fixture did not evict its native object");
		cache.clear();
		pending = ctx.submit();
	}
	require(!pending->ready(), "View lifetime fixture was not actually pending");
	gate.signaledValue = 1;
	pending->wait();
	const std::array<std::uint32_t, 4> expected{80, 70, 255, 0};
	require(std::equal(expected.begin(), expected.end(), static_cast<const std::uint32_t*>(output.contents)),
		"Evicted view lost source/allocation ownership after cache destruction");
	std::cout << "Pending GPU view survives eviction, clear, cache destruction and release of external source/view references\n";
}
}
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx(1);
			key_and_sampling(ctx);
			bounds_and_bindings(ctx);
			pending_eviction(ctx);
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
