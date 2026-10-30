#include "M4FragmentShader.h"
#include "M4VertexShader.h"
#include "M4VertexFetch.h"
#include "M4SurfaceColor.h"
#include "M4SampleTransfer.h"

#include <algorithm>
#include <bit>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void append(std::vector<std::byte>& code, std::uint32_t word)
{
	for (const unsigned shift : {8u, 0u, 24u, 16u})
		code.push_back(static_cast<std::byte>(word >> shift));
}
std::vector<std::byte> program()
{
	std::vector<std::byte> code;
	for (const unsigned reg : {0u, 2u, 3u, 4u})
	{
		for (const auto word : {(1u << 24) | (reg << 1) | (15u << 9) | (reg == 4 ? 1u : 0u), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(code, word);
		for (const float value : {0.25f, 0.5f, 0.75f, 1.0f})
			append(code, std::bit_cast<std::uint32_t>(value));
	}
	return code;
}
std::uint32_t logic(unsigned op, std::uint32_t source, std::uint32_t destination, unsigned bits)
{
	constexpr std::array<unsigned, 16> truth{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
	std::uint32_t result = 0;
	for (unsigned bit = 0; bit < bits; ++bit)
		if (truth[op] & (1u << (((source >> bit) & 1) * 2 + ((destination >> bit) & 1))))
			result |= 1u << bit;
	return result;
}
std::uint32_t source_word(surface_color format)
{
	switch (format)
	{
	case surface_color::x1rgb5_zero:
		return 0x2217;
	case surface_color::x1rgb5_one:
		return 0xa217;
	case surface_color::rgb565:
		return 0x4417;
	case surface_color::xrgb8_zero:
		return 0x004080bf;
	case surface_color::xrgb8_one:
	case surface_color::argb8:
		return 0xff4080bf;
	case surface_color::xbgr8_zero:
		return 0x00bf8040;
	case surface_color::xbgr8_one:
	case surface_color::abgr8:
		return 0xffbf8040;
	case surface_color::b8:
		return 0xbf;
	case surface_color::gb8:
		return 0x80bf;
	default:
		return 0;
	}
}
std::uint32_t write_bits(surface_color format, unsigned mask)
{
	std::array<std::uint32_t, 4> bits;
	switch (format)
	{
	case surface_color::x1rgb5_zero:
	case surface_color::x1rgb5_one:
		bits = {0x7c00, 0x03e0, 0x001f, 0x8000};
		break;
	case surface_color::rgb565:
		bits = {0xf800, 0x07e0, 0x001f, 0};
		break;
	case surface_color::b8:
		bits = {0, 0, 255, 0};
		break;
	case surface_color::gb8:
		bits = {0, 0xff00, 0x00ff, 0};
		break;
	case surface_color::xbgr8_zero:
	case surface_color::xbgr8_one:
	case surface_color::abgr8:
		bits = {0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000};
		break;
	default:
		bits = {0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000};
		break;
	}
	std::uint32_t result = 0;
	for (unsigned i = 0; i < 4; ++i)
		if (mask & (1u << i))
			result |= bits[i];
	return result;
}
void store(std::span<std::byte> data, std::uint32_t value, unsigned width)
{
	for (unsigned i = 0; i < width; ++i)
		data[i] = static_cast<std::byte>(value >> ((width - 1 - i) * 8));
}
} // namespace
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			context ctx(1);
			compiler compiler(ctx);
			constexpr std::uint32_t xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
			const std::array<std::array<std::uint32_t, 4>, 1> vertex_code{{{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
			vertex_shader vertex(compiler, vertex_code);
			vertex_fetch_engine fetch(ctx);
			std::vector<std::byte> vertices;
			for (const float value : {-1.f, -1.f, 0.5f, 1.f, 3.f, -1.f, 0.5f, 1.f, -1.f, 3.f, 0.5f, 1.f})
				for (const unsigned shift : {24u, 16u, 8u, 0u})
					vertices.push_back(static_cast<std::byte>(std::bit_cast<std::uint32_t>(value) >> shift));
			const auto vertex_data = make_upload(ctx, vertices);
			const auto code = program();
			const auto geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
			surface_color_transfer colors(ctx);
			sample_transfer samples(ctx);
			std::size_t cases = 0;
			for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
				for (const unsigned count : {1u, 2u, 4u})
					for (unsigned op = 0; op < 16; ++op)
					{
						const auto format = static_cast<surface_color>(raw);
						const auto metadata = describe_surface_color(format);
						const bool floating = raw >= 11 && raw <= 13;
						const unsigned slot = op % 4;
						render_pipeline_configuration config;
						config.state.logic_enabled = true;
						config.state.logic_operation = 0x1500 + op;
						config.state.sample_count = count;
						// Logic state must override blending, including on float targets
						// where the logic operation itself is ignored by Vulkan.
						config.state.colors[slot].enabled = true;
						config.state.colors[slot].destination_rgb = 1;
						config.state.colors[slot].destination_alpha = 1;
						fragment_program_options options;
						configure_surface_color(config, options, slot, format, 15);
						configure_logic_operations(config, options);
						fragment_shader fragment(compiler, code, options);
						if (raw == 8 && count == 1 && op == 6)
						{
							bool rejected = false;
							try
							{
								(void)make_render_pipeline(compiler, config, vertex.function(), fragment.function());
							}
							catch (const std::invalid_argument&)
							{
								rejected = true;
							}
							require(rejected, "Unverified function descriptor bypassed logic shader contract");
							auto mismatch = config;
							mismatch.shader_logic.operation = 0x1507;
							rejected = false;
							try
							{
								(void)make_render_pipeline(compiler, mismatch, vertex.function(), fragment);
							}
							catch (const std::invalid_argument&)
							{
								rejected = true;
							}
							require(rejected, "Mismatched compiled logic operation was accepted");
						}
						const std::size_t sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1, width = 8 * sx, height = 4 * sy;
						const auto bytes = metadata.layout.pixel_bytes;
						const auto pitch = width * bytes;
						std::vector<std::byte> original(pitch * height);
						for (std::size_t i = 0; i < original.size(); i += metadata.layout.endian_element_bytes)
							store(std::span(original).subspan(i), floating ? (raw == 11 ? 0xb800 : 0xbf000000) : std::uint32_t(0xa5c37e12 + i * 73), metadata.layout.endian_element_bytes);
						const auto input = make_upload(ctx, original), output = make_buffer(ctx, original.size(), buffer_storage::shared);
						auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metadata.format width:width height:height mipmapped:NO];
						td.storageMode = MTLStorageModePrivate;
						td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
						const auto expanded = make_texture(ctx, td);
						td.width = 8;
						td.height = 4;
						td.sampleCount = count;
						if (count > 1)
							td.textureType = MTLTextureType2DMultisample;
						const auto target = count == 1 ? expanded : make_texture(ctx, td);
						for (const unsigned mask : {0u, 5u, 10u, 15u})
							for (unsigned control = 0; control < (raw == 8 && op == 7 ? 4u : 1u); ++control)
							{
								configure_surface_color(config, options, slot, format, mask);
								const auto pipeline = make_render_pipeline(compiler, config, vertex.function(), fragment);
								ctx.begin();
								texture_region whole;
								whole.size = MTLSizeMake(width, height, 1);
								colors.upload(ctx, format, input, {}, expanded, whole);
								if (count > 1)
									samples.import(ctx, expanded, target);
								vertex_fetch_request request;
								request.vertex_count = 3;
								request.attribute_mask = 1;
								request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
								fragment_context fc;
								if (control == 1)
									fc.predicate = 0;
								if (control == 2)
								{
									fc.alpha_test = 1;
									fc.alpha_function = 0;
								}
								if (control == 3)
									fc.sample_mask = 5;
								vertex_inputs v;
								v.attributes = fetch.decode(ctx, request, vertex_data, nil);
								vertex_context vc;
								vc.vertex_count = 3;
								const auto vb = vertex.prepare(ctx, vc, geometry, v), fb = fragment.prepare(ctx, fc, code, {});
								auto pass = [MTL4RenderPassDescriptor new];
								pass.colorAttachments[slot].texture = target;
								pass.colorAttachments[slot].loadAction = MTLLoadActionLoad;
								pass.colorAttachments[slot].storeAction = MTLStoreActionStore;
								render_commands draw(ctx, pass);
								raster_state raster;
								raster.viewport = {0, 0, 8, 4, 0, 1};
								raster.scissor = {1, 1, 6, 2};
								// Overlapping instances in one native draw test raster-order fetch
								// against primitive writes, not a snapshot taken before the draw.
								draw.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb, 2);
								draw.end();
								if (count > 1)
									samples.expand(ctx, target, expanded);
								colors.download(ctx, format, expanded, whole, output, {});
								ctx.submit().wait();
								auto expected = original;
								const auto enabled = write_bits(format, mask), src = source_word(format);
								for (std::size_t y = sy; y < 3 * sy; ++y)
									for (std::size_t x = sx; x < 7 * sx; ++x)
									{
										const unsigned sample = x % sx + sx * (y % sy);
										if (control == 1 || control == 2 || !(fc.sample_mask & (1u << sample)))
											continue;
										const auto offset = (y * width + x) * bytes;
										if (floating)
										{
											constexpr std::array<std::uint32_t, 4> half{0x3400, 0x3800, 0x3a00, 0x3c00}, full{0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000};
											const unsigned element = metadata.layout.endian_element_bytes;
											for (unsigned channel = 0; channel < bytes / element; ++channel)
												if (mask & (1u << channel))
													store(std::span(expected).subspan(offset + channel * element), raw == 11 ? half[channel] : full[channel], element);
											continue;
										}
										std::uint32_t value = 0;
										for (unsigned byte = 0; byte < bytes; ++byte)
											value = (value << 8) | unsigned(original[offset + byte]);
										for (unsigned instance = 0; instance < 2; ++instance)
											value = (value & ~enabled) | (logic(op, src, value, bytes * 8) & enabled);
										store(std::span(expected).subspan(offset), value, bytes);
									}
								const auto actual = map_buffer(output, 0, output.length);
								if (!std::equal(actual.begin(), actual.end(), expected.begin()))
								{
									const auto diff = std::mismatch(actual.begin(), actual.end(), expected.begin());
									std::cerr << "Logic format=" << raw << " samples=" << count << " op=" << op << " mask=" << mask << " byte=" << diff.first - actual.begin() << " actual=" << unsigned(*diff.first) << " expected=" << unsigned(*diff.second) << '\n';
									throw std::runtime_error("Native per-fragment logic ROP differs from independent packed-word truth tables");
								}
								++cases;
							}
					}
			std::cout << cases << " native RSX logic-ROP draw cases across all 14 color formats, 16 operations, masks and 1/2/4 samples preserve primitive-order framebuffer visibility and float passthrough\n";
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
