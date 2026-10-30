#include "M4SampleTransfer.h"
#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void store(std::span<std::byte> bytes, std::uint32_t value, unsigned count)
{
	for (unsigned byte = 0; byte < count; ++byte)
		bytes[byte] = static_cast<std::byte>(value >> ((count - 1 - byte) * 8));
}
void populate(std::span<std::byte> bytes, unsigned raw, bool depth, std::size_t pixel)
{
	const auto index = static_cast<std::uint32_t>(pixel);
	if (depth)
	{
		const std::array<std::uint32_t, 6> e4m12{0, 0x5000, 0x6000, 0x7000, 0x2345, 0x3456};
		const auto value = raw == 1 ? index % 65536 : raw == 3 ? e4m12[index % e4m12.size()]
												  : raw == 2   ? ((index * 7919 % 0x1000000) << 8) | (index % 256)
															   : (std::bit_cast<std::uint32_t>(float(index % 1025) / 1024.0f) >> 7 << 8) | (index % 256);
		store(bytes, value, (raw == 1 || raw == 3) ? 2 : 4);
	}
	else if (raw <= 3)
		store(bytes, index % 65536, 2);
	else if (raw == 11 || raw == 12 || raw == 13)
	{
		const unsigned width = raw == 11 ? 2 : 4, count = raw == 13 ? 1 : 4;
		constexpr std::array<float, 8> values{-8, -0.5f, 0, 0.125f, 0.25f, 0.5f, 1, 2};
		for (unsigned c = 0; c < count; ++c)
		{
			const auto f = values[(pixel + c) % values.size()];
			const auto value = raw == 11 ? std::uint32_t(std::bit_cast<std::uint16_t>(static_cast<_Float16>(f))) : std::bit_cast<std::uint32_t>(f);
			store(bytes.subspan(c * width), value, width);
		}
	}
	else
		for (std::size_t byte = 0; byte < bytes.size(); ++byte)
			bytes[byte] = static_cast<std::byte>(index * 13 + byte * 41);
}

void run(metal4::context& ctx)
{
	using namespace metal4;
	sample_transfer transfer(ctx);
	surface_color_transfer colors(ctx);
	surface_depth_transfer depths(ctx);
	std::size_t cases = 0;
	for (const unsigned samples : {2u, 4u})
		for (const bool depth : {false, true})
			for (const auto raw : depth ? std::vector<unsigned>{1, 2, 3, 4} : std::vector<unsigned>{1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16})
			{
				const auto color = depth ? surface_color_descriptor{} : describe_surface_color(static_cast<surface_color>(raw));
				const auto ds = depth ? describe_surface_depth(static_cast<surface_depth>(raw)) : surface_depth_descriptor{};
				const auto format = depth ? ds.format : color.format;
				const auto bytes = depth ? ds.guest_bytes : color.layout.pixel_bytes;
				const std::size_t width = !depth && raw <= 3 ? 256 : 13, height = !depth && raw <= 3 ? 128 : 7;
				const auto size = sample_expanded_size(width, height, samples);
				const auto pitch = (size[0] + 3) * bytes;
				std::vector<std::byte> expected(3 + pitch * size[1] + 11, std::byte{0xa7});
				for (std::size_t y = 0; y < size[1]; ++y)
					for (std::size_t x = 0; x < size[0]; ++x)
						populate(std::span(expected).subspan(3 + y * pitch + x * bytes, bytes), raw, depth, y * size[0] + x);
				const auto source = make_upload(ctx, expected), out = make_buffer(ctx, expected.size(), buffer_storage::shared);
				MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
				md.textureType = MTLTextureType2DMultisample;
				md.sampleCount = samples;
				md.storageMode = MTLStorageModePrivate;
				md.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				const auto multisampled = make_texture(ctx, md);
				MTLTextureDescriptor* ed = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:size[0] height:size[1] mipmapped:NO];
				ed.storageMode = MTLStorageModePrivate;
				ed.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				const auto expanded = make_texture(ctx, ed), result = make_texture(ctx, ed);
				texture_region region;
				region.size = MTLSizeMake(size[0], size[1], 1);
				ctx.begin();
				fill_buffer(ctx, out, 0, out.length, 0xa7);
				if (depth)
					depths.upload(ctx, static_cast<surface_depth>(raw), source, {3, pitch}, expanded, region);
				else
					colors.upload(ctx, static_cast<surface_color>(raw), source, {3, pitch}, expanded, region);
				transfer.import(ctx, expanded, multisampled);
				transfer.expand(ctx, multisampled, result);
				if (depth)
					depths.download(ctx, static_cast<surface_depth>(raw), result, region, out, {3, pitch});
				else
					colors.download(ctx, static_cast<surface_color>(raw), result, region, out, {3, pitch});
				ctx.submit().wait();
				const auto actual = map_buffer(out, 0, out.length);
				if (!std::equal(actual.begin(), actual.end(), expected.begin()))
				{
					const auto diff = std::mismatch(actual.begin(), actual.end(), expected.begin());
					std::cerr << "Samples " << samples << " depth " << depth << " format " << raw << " byte " << diff.first - actual.begin()
							  << ": " << unsigned(*diff.first) << " expected " << unsigned(*diff.second) << '\n';
					throw std::runtime_error("Native individual sample import/expansion bytes differ");
				}
				++cases;
				std::cout << samples << "x " << (depth ? "depth " : "color ") << raw << " native sample import/expansion passed\n";
			}
	std::cout << cases << " native MSAA surface cases preserve every sample and byte guard\n";
	ctx.wait_idle();
}

void independent_samples(metal4::context& ctx)
{
	using namespace metal4;
	compiler compiler(ctx);
	sample_transfer transfer(ctx);
	surface_depth_transfer depth_transfer(ctx);
	std::string source = generate_sample_shader(sample_aspect::color, false);
	source += R"msl(
struct independently_written_sample {
	float4 color [[color(0)]];
	float depth [[depth(any)]];
	uint stencil [[stencil]];
};
fragment independently_written_sample independent_fragment(sample_vertex in [[stage_in]],uint sample [[sample_id]]) {
	const uint2 pixel=uint2(in.position.xy);
	return {float4(float(sample+1)/4.0f,float(pixel.x%4)/4.0f,float(pixel.y%4)/4.0f,1),
		float(sample+1)/8.0f,(17+sample*53+pixel.x+pixel.y*7)%256};
}
)msl";
	const auto library = compiler.library(source, @"Independent MSAA sample writer");
	MTLDepthStencilDescriptor* depth_descriptor = [MTLDepthStencilDescriptor new];
	depth_descriptor.depthCompareFunction = MTLCompareFunctionAlways;
	depth_descriptor.depthWriteEnabled = YES;
	MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
	stencil.stencilCompareFunction = MTLCompareFunctionAlways;
	stencil.depthStencilPassOperation = MTLStencilOperationReplace;
	depth_descriptor.frontFaceStencil = depth_descriptor.backFaceStencil = stencil;
	const auto state = [ctx.device() newDepthStencilStateWithDescriptor:depth_descriptor];
	for (const unsigned samples : {2u, 4u})
	{
		constexpr std::size_t width = 13, height = 7;
		const auto size = sample_expanded_size(width, height, samples);
		const auto make = [&](MTLPixelFormat format, bool msaa)
		{
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:msaa ? width : size[0] height:msaa ? height : size[1] mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
			if (msaa)
			{
				descriptor.textureType = MTLTextureType2DMultisample;
				descriptor.sampleCount = samples;
			}
			return make_texture(ctx, descriptor);
		};
		const auto color = make(MTLPixelFormatRGBA32Float, true), depth = make(MTLPixelFormatDepth32Float_Stencil8, true),
				   out_color = make(MTLPixelFormatRGBA32Float, false), out_depth = make(MTLPixelFormatDepth32Float_Stencil8, false);
		MTL4RenderPipelineDescriptor* pd = [MTL4RenderPipelineDescriptor new];
		pd.vertexFunctionDescriptor = library_function(library, @"rpcs3_sample_vertex");
		pd.fragmentFunctionDescriptor = library_function(library, @"independent_fragment");
		pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		pd.rasterSampleCount = samples;
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
		const auto pipeline = compiler.render(pd);
		const auto color_buffer = make_buffer(ctx, size[0] * size[1] * 16, buffer_storage::shared),
				   depth_buffer = make_buffer(ctx, size[0] * size[1] * 4, buffer_storage::shared);
		ctx.begin();
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = color;
		pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.depthAttachment.texture = depth;
		pass.depthAttachment.loadAction = MTLLoadActionDontCare;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.stencilAttachment.texture = depth;
		pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
		pass.stencilAttachment.storeAction = MTLStoreActionStore;
		render_commands commands(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, double(width), double(height), 0, 1};
		raster.scissor = {0, 0, width, height};
		commands.configure(pipeline, state, raster, {}, {});
		commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
		commands.end();
		transfer.expand(ctx, color, out_color);
		transfer.expand(ctx, depth, out_depth);
		texture_region region;
		region.size = MTLSizeMake(size[0], size[1], 1);
		texture_to_buffer(ctx, out_color, region, color_buffer, 0, size[0] * 16);
		depth_transfer.download(ctx, surface_depth::z24s8_float, out_depth, region, depth_buffer, {});
		ctx.submit().wait();
		const auto colors = map_buffer(color_buffer, 0, color_buffer.length), depths = map_buffer(depth_buffer, 0, depth_buffer.length);
		for (std::size_t y = 0; y < size[1]; ++y)
			for (std::size_t x = 0; x < size[0]; ++x)
			{
				const auto px = x / 2, py = samples == 2 ? y : y / 2;
				const unsigned sample = static_cast<unsigned>(x % 2 + (samples == 2 ? 0 : (y % 2) * 2));
				std::array<float, 4> actual;
				std::memcpy(actual.data(), colors.data() + (y * size[0] + x) * 16, 16);
				const std::array<float, 4> expected{float(sample + 1) / 4.0f, float(px % 4) / 4.0f, float(py % 4) / 4.0f, 1};
				require(actual == expected, "Independent MSAA color sample coordinate differs");
				const auto bits = std::bit_cast<std::uint32_t>(float(sample + 1) / 8.0f);
				const auto guest = (bits >> 7 << 8) | static_cast<std::uint32_t>((17 + sample * 53 + px + py * 7) % 256);
				for (unsigned byte = 0; byte < 4; ++byte)
					require(depths[(y * size[0] + x) * 4 + byte] == static_cast<std::byte>(guest >> ((3 - byte) * 8)), "Independent MSAA depth/stencil sample coordinate differs");
			}
		ctx.begin();
		bool rejected = false;
		try
		{
			transfer.expand(ctx, color, out_depth);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Incompatible native MSAA formats were accepted");
		rejected = false;
		try
		{
			transfer.expand(ctx, out_color, out_color);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Invalid native MSAA sample/type/alias contract was accepted");
		const auto reversed = make_texture_view(out_color, MTLPixelFormatRGBA32Float, MTLTextureType2D,
			NSMakeRange(0,1), NSMakeRange(0,1), MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue,MTLTextureSwizzleGreen,MTLTextureSwizzleRed,MTLTextureSwizzleAlpha));
		rejected = false;
		try { transfer.import(ctx,reversed,color); } catch (const std::invalid_argument&) { rejected = true; }
		require(rejected,"Channel-remapped view was accepted as raw native sample input");
		ctx.cancel();
	}
	std::cout << "Independent per-sample writer confirms color/depth/stencil sample placement for 2x and 4x\n";
}

void invalid_depth_values()
{
	using namespace metal4;
	for (const float invalid : {-0.25f, 2.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
	{
		const auto operation = [&]
		{
			// The feedback callback must own its checks and buffers even after
			// this context and sample-transfer object have been destroyed.
			context ctx(1);
			sample_transfer transfer(ctx);
			const std::array<float, 2> values{invalid, 0.5f};
			const auto input = make_upload(ctx, std::as_bytes(std::span(values)));
			MTLTextureDescriptor* single = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:2 height:1 mipmapped:NO];
			single.storageMode = MTLStorageModePrivate;
			single.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto source = make_texture(ctx, single);
			MTLTextureDescriptor* multi = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:1 height:1 mipmapped:NO];
			multi.textureType = MTLTextureType2DMultisample;
			multi.sampleCount = 2;
			multi.storageMode = MTLStorageModePrivate;
			multi.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto target = make_texture(ctx, multi);
			ctx.begin();
			texture_region region;
			region.size = MTLSizeMake(2, 1, 1);
			buffer_to_texture(ctx, input, 0, source, region);
			transfer.import(ctx, source, target);
			return ctx.submit();
		}();
		bool rejected = false;
		try
		{
			operation.wait();
		}
		catch (const std::runtime_error& error)
		{
			rejected = std::string(error.what()).find("outside finite [0,1]") != std::string::npos;
		}
		require(rejected && operation.ready(), "Invalid MSAA depth silently clamped or completion checks lost after destruction");
	}
	std::cout << "Negative/>1/Inf/NaN depth reports GPU submission validation failure after context destruction\n";
}
} // namespace
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			metal4::context ctx(1);
			run(ctx);
			independent_samples(ctx);
			ctx.wait_idle();
			invalid_depth_values();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
