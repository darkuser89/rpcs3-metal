#include "M4SurfaceColor.h"
#include "M4VertexShader.h"
#include "M4FragmentShader.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}

constexpr std::string_view sample_kernel = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void sample_surface(texture2d<float> image [[texture(0)]], sampler sampling [[sampler(0)]],
	device float4* output [[buffer(0)]], uint index [[thread_position_in_grid]])
{
	const uint width = image.get_width(), height = image.get_height();
	if (index >= width * height) return;
	const float2 coordinate = (float2(index % width, index / width) + 0.5f) / float2(width, height);
	output[index] = image.sample(sampling, coordinate, level(0));
}
)msl";

struct fixture { std::uint32_t format, bytes, element; };
constexpr std::array<fixture, 14> formats{{{1,2,2}, {2,2,2}, {3,2,2}, {4,4,4}, {5,4,4}, {8,4,4},
	{9,1,1}, {10,2,2}, {11,8,2}, {12,16,4}, {13,4,4}, {14,4,4}, {15,4,4}, {16,4,4}}};

void store_be(std::span<std::byte> bytes, std::uint32_t value, unsigned width)
{
	for (unsigned i = 0; i < width; ++i) bytes[i] = static_cast<std::byte>(value >> (8 * (width - 1 - i)));
}

std::array<float, 4> pixel_fixture(std::span<std::byte> guest, const fixture& f, std::size_t ordinal)
{
	const auto value = static_cast<std::uint32_t>(ordinal);
	if (f.format <= 3)
	{
		store_be(guest, value, 2);
		if (f.format == 3)
			return {float(value / 2048 % 32) / 31, float(value / 32 % 64) / 63, float(value % 32) / 31, 1};
		return {float(value / 1024 % 32) / 31, float(value / 32 % 32) / 31, float(value % 32) / 31, f.format == 1 ? 0.f : 1.f};
	}
	if (f.format == 11 || f.format == 12 || f.format == 13)
	{
		// Distinct W/Z/Y/X elements, including signed zero and exact binary values.
		constexpr std::array<float, 8> values{-0.f, 0.f, 0.125f, -0.25f, 0.5f, 1.f, 2.f, -8.f};
		std::array<float, 4> expected{};
		const unsigned components = f.format == 13 ? 1 : 4;
		for (unsigned c = 0; c < components; ++c)
		{
			const float current = values[(ordinal + c) % values.size()];
			expected[c] = current;
			const auto bits = f.format == 11 ? std::uint32_t(std::bit_cast<std::uint16_t>(static_cast<_Float16>(current))) : std::bit_cast<std::uint32_t>(current);
			store_be(guest.subspan(c * f.element), bits, f.element);
		}
		if (f.format == 13) expected.fill(expected[0]);
		return expected;
	}
	const unsigned blue = (value * 13 + 17) % 256, green = (value * 7 + 31) % 256,
		red = (value * 3 + 59) % 256, alpha = (value * 5 + 83) % 256;
	if (f.format == 9)
	{
		guest[0] = static_cast<std::byte>(blue);
		return {float(blue)/255, float(blue)/255, float(blue)/255, 1};
	}
	if (f.format == 10)
	{
		guest[0] = static_cast<std::byte>(green); guest[1] = static_cast<std::byte>(blue);
		return {float(blue)/255, float(green)/255, float(blue)/255, float(green)/255};
	}
	guest[0] = static_cast<std::byte>(alpha); guest[2] = static_cast<std::byte>(green);
	guest[1] = static_cast<std::byte>(f.format >= 14 ? blue : red);
	guest[3] = static_cast<std::byte>(f.format >= 14 ? red : blue);
	return {float(red)/255, float(green)/255, float(blue)/255,
		f.format == 4 || f.format == 14 ? 0.f : f.format == 5 || f.format == 15 ? 1.f : float(alpha)/255};
}

void run(metal4::context& ctx)
{
	using namespace metal4;
	compiler compiler(ctx);
	surface_color_transfer conversion(ctx);
	const auto library = compiler.library(sample_kernel, @"Standalone surface sample verification");
	MTL4ComputePipelineDescriptor* compute_descriptor = [MTL4ComputePipelineDescriptor new];
	compute_descriptor.computeFunctionDescriptor = library_function(library, @"sample_surface");
	const auto sample_pipeline = compiler.compute(compute_descriptor);
	MTLSamplerDescriptor* sampler_descriptor = [MTLSamplerDescriptor new];
	sampler_descriptor.minFilter = MTLSamplerMinMagFilterNearest;
	sampler_descriptor.magFilter = MTLSamplerMinMagFilterNearest;
	const auto sampler = make_sampler(ctx, sampler_descriptor);
	for (const auto& f : formats)
	{
		const auto format = static_cast<surface_color>(f.format);
		const std::size_t width = f.format <= 3 ? 256 : 13, height = f.format <= 3 ? 256 : 7;
		const auto source_pitch = (width + 3) * f.bytes, destination_pitch = (width + 5) * f.bytes;
		std::vector<std::byte> source_bytes(3 + source_pitch * height + 29, std::byte{0xa7});
		std::vector<std::array<float, 4>> expected_colors;
		for (std::size_t y = 0; y < height; ++y)
			for (std::size_t x = 0; x < width; ++x)
				expected_colors.push_back(pixel_fixture(std::span(source_bytes).subspan(3 + y * source_pitch + x * f.bytes, f.bytes), f, y * width + x));
		const auto source = make_upload(ctx, source_bytes);
		const auto native = make_buffer(ctx, 7 + destination_pitch * height + 31, buffer_storage::shared);
		const auto guest_result = make_buffer(ctx, 5 + destination_pitch * height + 33, buffer_storage::shared);
		const auto sampled = make_buffer(ctx, width * height * 16, buffer_storage::shared);
		const auto descriptor = describe_surface_color(format);
		MTLTextureDescriptor* texture_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:descriptor.format width:width height:height mipmapped:NO];
		texture_descriptor.storageMode = MTLStorageModePrivate;
		texture_descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
		const auto texture = make_texture(ctx, texture_descriptor);
		const auto view = make_texture_view(texture, descriptor.format, MTLTextureType2D, NSMakeRange(0,1), NSMakeRange(0,1), descriptor.sampling_swizzle);
		texture_region region; region.size = MTLSizeMake(width, height, 1);
		ctx.begin();
		fill_buffer(ctx, native, 0, native.length, 0xd3);
		fill_buffer(ctx, guest_result, 0, guest_result.length, 0xe5);
		conversion.convert(ctx, format, width, height, source, {3, source_pitch}, native, {7, destination_pitch});
		conversion.upload(ctx, format, source, {3, source_pitch}, texture, region);
		conversion.download(ctx, format, texture, region, guest_result, {5, destination_pitch});
		bindings resources;
		resources.set_buffer(0, sampled);
		resources.set_texture(0, view);
		resources.set_sampler(0, sampler);
		const auto table = resources.snapshot(ctx);
		const auto encoder = ctx.compute();
		ctx.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
		ctx.keep_alive(sample_pipeline);
		[encoder setComputePipelineState:sample_pipeline];
		[encoder setArgumentTable:table];
		[encoder dispatchThreads:MTLSizeMake(width * height, 1, 1)
			threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, sample_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
		ctx.submit().wait();
		std::vector<std::byte> expected_native(native.length, std::byte{0xd3}), expected_guest(guest_result.length, std::byte{0xe5});
		for (std::size_t y = 0; y < height; ++y)
			for (std::size_t x = 0; x < width; ++x)
				for (unsigned byte = 0; byte < f.bytes; ++byte)
				{
					// Independent per-element byte reversal, without the kernel's XOR.
					const auto element = byte / f.element, lane = byte % f.element;
					expected_native[7 + y * destination_pitch + x * f.bytes + byte] = source_bytes[3 + y * source_pitch + x * f.bytes + element * f.element + f.element - 1 - lane];
					expected_guest[5 + y * destination_pitch + x * f.bytes + byte] = source_bytes[3 + y * source_pitch + x * f.bytes + byte];
				}
		const auto actual_native = map_buffer(native, 0, native.length), actual_guest = map_buffer(guest_result, 0, guest_result.length);
		require(std::equal(actual_native.begin(), actual_native.end(), expected_native.begin()), "Surface byte conversion/padding/guards differ");
		require(std::equal(actual_guest.begin(), actual_guest.end(), expected_guest.begin()), "Surface native texture round-trip/padding/guards differ");
		const auto sample_bytes = map_buffer(sampled, 0, sampled.length);
		for (std::size_t pixel = 0; pixel < width * height; ++pixel)
		{
			std::array<float, 4> actual;
			std::memcpy(actual.data(), sample_bytes.data() + pixel * 16, 16);
			for (unsigned channel = 0; channel < 4; ++channel)
				if (std::abs(actual[channel] - expected_colors[pixel][channel]) > 0.00001f)
				{
					std::cerr << "Surface format " << f.format << ", pixel " << pixel << ", channel " << channel
						<< ": actual " << actual[channel] << ", expected " << expected_colors[pixel][channel] << '\n';
					throw std::runtime_error("Native sampled packed/color/float channel mapping differs");
				}
		}
		std::cout << "Surface " << f.format << ": byte conversion, texture round-trip and " << width * height << " sampled pixels passed\n";
	}
	ctx.wait_idle();
}

void append_word(std::vector<std::byte>& output, std::uint32_t word)
{
	for (const unsigned shift : {8u, 0u, 24u, 16u}) output.push_back(static_cast<std::byte>(word >> shift));
}

void draw_formats(metal4::context& ctx)
{
	using namespace metal4;
	compiler compiler(ctx);
	surface_color_transfer transfer(ctx);
	constexpr std::uint32_t input_xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	const std::array<std::array<std::uint32_t, 4>, 1> code{{{(63u << 15) | (7u << 10) | (1u << 30),
		(1u << 22) | (input_xyzw >> 9), (input_xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
	vertex_shader vertex(compiler, code);
	std::vector<std::byte> fragment_code;
	for (const auto word : {(1u << 24) | (15u << 9) | 1u,
		(0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9}) append_word(fragment_code, word);
	for (const float value : {0.25f, 0.5f, 0.75f, 0.375f}) append_word(fragment_code, std::bit_cast<std::uint32_t>(value));
	std::array<float, 3 * 16 * 4> attributes{};
	const std::array<std::array<float,4>,3> positions{{{-1,-1,0.5f,1}, {3,-1,0.5f,1}, {-1,3,0.5f,1}}};
	for (std::size_t i = 0; i < 3; ++i) std::copy(positions[i].begin(), positions[i].end(), attributes.begin() + static_cast<std::ptrdiff_t>(i * 64));
	vertex_inputs vertex_data;
	vertex_data.attributes = make_upload(ctx, std::as_bytes(std::span(attributes)));
	vertex_context vertex_state; vertex_state.vertex_count = 3;
	const auto geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	for (const auto& f : formats)
	{
		const auto format = static_cast<surface_color>(f.format);
		render_pipeline_configuration config;
		fragment_program_options options;
		configure_surface_color(config, options, 0, format, 15);
		fragment_shader fragment(compiler, fragment_code, options);
		const auto pipeline = make_render_pipeline(compiler, config, vertex.function(), fragment);
		MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:config.colors[0] width:4 height:4 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		const auto texture = make_texture(ctx, descriptor);
		const auto readback = make_buffer(ctx, 4 * 4 * f.bytes, buffer_storage::shared);
		ctx.begin();
		const auto vs = vertex.prepare(ctx, vertex_state, geometry, vertex_data);
		const auto fs = fragment.prepare(ctx, {}, fragment_code, {});
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = texture;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0);
		render_commands commands(ctx, pass);
		raster_state raster; raster.viewport = {0,0,4,4,0,1}; raster.scissor = {0,0,4,4};
		commands.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vs, fs);
		commands.end();
		texture_region region; region.size = MTLSizeMake(4,4,1);
		transfer.download(ctx, format, texture, region, readback, {});
		ctx.submit().wait();
		std::array<std::byte,16> expected{};
		if (f.format <= 3)
		{
			// Quantized source R=.25,G=.5,B=.75, X=0/1. Explicit expected
			// integer fields, not a round-trip through the implementation.
			const unsigned word = f.format == 3 ? (8u << 11) | (32u << 5) | 23u :
				(8u << 10) | (16u << 5) | 23u | (f.format == 2 ? 0x8000u : 0);
			store_be(expected, word, 2);
		}
		else if (f.format == 9) expected[0] = std::byte{191};
		else if (f.format == 10) { expected[0] = std::byte{128}; expected[1] = std::byte{191}; }
		else if (f.format == 11 || f.format == 12 || f.format == 13)
		{
			const std::array<std::uint16_t,4> half_bits{0x3400, 0x3800, 0x3a00, 0x3600};
			const std::array<std::uint32_t,4> float_bits{0x3e800000, 0x3f000000, 0x3f400000, 0x3ec00000};
			for (unsigned c = 0; c < (f.format == 13 ? 1u : 4u); ++c)
				store_be(std::span(expected).subspan(c * f.element), f.format == 11 ? half_bits[c] : float_bits[c], f.element);
		}
		else
		{
			expected[0] = f.format == 4 || f.format == 14 ? std::byte{0} : f.format == 5 || f.format == 15 ? std::byte{255} : std::byte{96};
			expected[1] = f.format >= 14 ? std::byte{191} : std::byte{64};
			expected[2] = std::byte{128};
			expected[3] = f.format >= 14 ? std::byte{64} : std::byte{191};
		}
		const auto actual = map_buffer(readback, 0, readback.length);
		for (std::size_t pixel = 0; pixel < 16; ++pixel)
			if (!std::equal(expected.begin(), expected.begin() + f.bytes, actual.begin() + static_cast<std::ptrdiff_t>(pixel * f.bytes)))
			{
				std::cerr << "Surface " << f.format << " draw bytes: ";
				for (unsigned b = 0; b < f.bytes; ++b) std::cerr << unsigned(actual[pixel * f.bytes + b]) << '(' << unsigned(expected[b]) << ") ";
				std::cerr << '\n';
				throw std::runtime_error("Native RSX color export/remap/render/download differs");
			}
		std::cout << "Surface " << f.format << ": native RSX fragment export, render and guest bytes passed\n";
	}
	ctx.wait_idle();
}

void raw_float_bits_and_subregions(metal4::context& ctx)
{
	using namespace metal4;
	surface_color_transfer transfer(ctx);
	for (const auto format : {surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::argb8})
	{
		const auto metadata = describe_surface_color(format);
		const auto bpp = metadata.layout.pixel_bytes, element = metadata.layout.endian_element_bytes;
		constexpr std::size_t width = 5, height = 3;
		const auto pitch = (width + 2) * bpp;
		std::vector<std::byte> original(1 + pitch * height + 7, std::byte{0x5a});
		constexpr std::array<std::uint32_t, 8> float_bits{0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0x7f812345,1,0x7fffff};
		constexpr std::array<std::uint16_t, 8> half_bits{0,0x8000,0x7c00,0xfc00,0x7e37,0x7d37,1,0x3ff};
		for (std::size_t y = 0; y < height; ++y)
			for (std::size_t x = 0; x < width; ++x)
				for (unsigned c = 0; c < bpp / element; ++c)
				{
					const auto index = (y * width + x + c) % 8;
					store_be(std::span(original).subspan(1 + y * pitch + x * bpp + c * element), element == 2 ? half_bits[index] : float_bits[index], element);
				}
		const auto input = make_upload(ctx, original), output = make_buffer(ctx, original.size(), buffer_storage::shared);
		MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metadata.format width:16 height:16 mipmapped:NO];
		descriptor.textureType = MTLTextureType2DArray; descriptor.arrayLength = 2; descriptor.mipmapLevelCount = 2;
		descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		const auto texture = make_texture(ctx, descriptor);
		texture_region region; region.level = 1; region.slice = 1; region.origin = MTLOriginMake(2,1,0); region.size = MTLSizeMake(width,height,1);
		ctx.begin();
		fill_buffer(ctx, output, 0, output.length, 0x5a);
		transfer.upload(ctx, format, input, {1,pitch}, texture, region);
		transfer.download(ctx, format, texture, region, output, {1,pitch});
		ctx.submit().wait();
		const auto actual = map_buffer(output, 0, output.length);
		require(std::equal(actual.begin(), actual.end(), original.begin()), "Surface NaN/Inf/-zero/subnormal/mip/slice/region raw bits changed");
	}
	// These checks must fail before conversion encoding; a fresh valid recording
	// can then be cancelled with no driver crash or resource lifetime violation.
	ctx.begin();
	const auto buffer = make_buffer(ctx,64,buffer_storage::shared);
	const auto rejects = [](auto function)
	{
		bool rejected = false;
		try { function(); } catch (const std::invalid_argument&) { rejected = true; }
		catch (const std::out_of_range&) { rejected = true; }
		require(rejected, "Invalid native surface conversion was accepted");
	};
	rejects([&] { transfer.convert(ctx, surface_color::argb8, 2,2,buffer,{},buffer,{}); });
	rejects([&] { transfer.convert(ctx, surface_color::argb8, 2,2,buffer,{60,8},buffer,{0,8}); });
	rejects([&] { transfer.convert(ctx, surface_color::argb8, 2,2,buffer,{0,7},buffer,{32,8}); });
	rejects([&] { transfer.convert(ctx, static_cast<surface_color>(6), 2,2,buffer,{},buffer,{32,8}); });
	ctx.cancel();
	std::cout << "Surface raw NaN/Inf/-zero/subnormal payloads, nonzero mip/slice/origin and rejection cases passed\n";
}
}

int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try { metal4::context context(1); run(context); draw_formats(context); raw_float_bits_and_subregions(context); }
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
