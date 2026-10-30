#include "M4FragmentShader.h"
#include "M4VertexShader.h"
#include "M4VertexFetch.h"
#include "M4SurfaceColor.h"
#include "M4SampleTransfer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void append(std::vector<std::byte>& code, unsigned word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		code.push_back(static_cast<std::byte>(word >> shift));
}
std::vector<std::byte> program(float alpha, bool full)
{
	std::vector<std::byte> code;
	for (unsigned word : {(1u << 24) | (1u << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
		append(code, word);
	for (float value : {0.f, 0.f, 0.5f, 0.f})
		append(code, std::bit_cast<unsigned>(value));
	for (unsigned word : {(1u << 24) | (15u << 9) | 1u | (full ? 0u : 128u), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
		append(code, word);
	for (float value : {0.5f, 0.75f, 0.25f, alpha})
		append(code, std::bit_cast<unsigned>(value));
	return code;
}
class fixture
{
	context m_ctx{1};
	compiler m_compiler{m_ctx};
	vertex_fetch_engine m_fetch{m_ctx};
	surface_color_transfer m_colors{m_ctx};
	sample_transfer m_samples{m_ctx};
	std::unique_ptr<vertex_shader> m_vertex;
	id<MTLBuffer> m_vertices;
	converted_geometry m_geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);

  public:
	std::size_t cases = 0;
	double maximum_density_error = 0;
	fixture()
	{
		constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
		const std::array<std::array<unsigned, 4>, 1> code{{{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
		m_vertex = std::make_unique<vertex_shader>(m_compiler, code);
		std::vector<std::byte> bytes;
		for (float value : {-1.f, -1.f, 0.5f, 1.f, 3.f, -1.f, 0.5f, 1.f, -1.f, 3.f, 0.5f, 1.f})
			for (unsigned shift : {24u, 16u, 8u, 0u})
				bytes.push_back(static_cast<std::byte>(std::bit_cast<unsigned>(value) >> shift));
		m_vertices = make_upload(m_ctx, bytes);
	}
	void run(surface_color format, unsigned count, bool full, bool blend, bool filtered, bool colorless)
	{
		render_pipeline_configuration config;
		config.state.sample_count = count;
		config.state.depth_test = true;
		config.state.depth_write = true;
		config.state.depth_compare = 0x207;
		config.state.output_rop.stochastic_alpha_coverage = true;
		config.state.output_rop.polygon_stipple = filtered;
		fragment_program_options options;
		options.float32_exports = full;
		options.depth_export = true;
		options.color_count = colorless ? 0 : 1;
		if (!colorless)
		{
			config.state.colors[0].enabled = blend;
			// Coverage must use pre-blend alpha. Force blended alpha to zero.
			config.state.colors[0].source_alpha = config.state.colors[0].destination_alpha = 0;
			configure_surface_color(config, options, 0, format, 15);
		}
		configure_shader_blending(config, options);
		const auto code = program(0.5f, full);
		fragment_shader fragment(m_compiler, code, options);
		auto mismatch = config;
		mismatch.state.sample_count = count == 1 ? 2 : 1;
		bool rejected = false;
		try { (void)make_render_pipeline(m_compiler, mismatch, m_vertex->function(), fragment); }
		catch (const std::invalid_argument&) { rejected = true; }
		require(rejected, "Stochastic shader accepted a mismatched raster sample count");
		const auto pipeline = make_render_pipeline(m_compiler, config, m_vertex->function(), fragment);
		constexpr unsigned native_width = 128, native_height = 96;
		const unsigned sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1, width = native_width * sx, height = native_height * sy;
		const auto info = describe_surface_color(format);
		const unsigned bytes = info.layout.pixel_bytes;
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:info.format width:width height:height mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		const auto expanded = make_texture(m_ctx, descriptor);
		descriptor.width = native_width;
		descriptor.height = native_height;
		descriptor.sampleCount = count;
		if (count > 1)
			descriptor.textureType = MTLTextureType2DMultisample;
		const auto target = count == 1 ? expanded : make_texture(m_ctx, descriptor);
		descriptor.textureType = MTLTextureType2D;
		descriptor.sampleCount = 1;
		descriptor.pixelFormat = MTLPixelFormatDepth32Float;
		descriptor.width = width;
		descriptor.height = height;
		const auto expanded_depth = make_texture(m_ctx, descriptor);
		descriptor.width = native_width;
		descriptor.height = native_height;
		descriptor.sampleCount = count;
		if (count > 1)
			descriptor.textureType = MTLTextureType2DMultisample;
		const auto depth = count == 1 ? expanded_depth : make_texture(m_ctx, descriptor);
		const std::vector<std::byte> zeros(width * height * bytes);
		const auto input = make_upload(m_ctx, zeros), output = make_buffer(m_ctx, zeros.size(), buffer_storage::shared), depth_output = make_buffer(m_ctx, width * height * 4, buffer_storage::shared);
		std::vector<unsigned char> previous;
		const std::array<float, 10> alphas{0.f, 0.25f, 0.5f, 0.5f, 0.75f, 1.f, 1.25f, -0.25f, std::numeric_limits<float>::quiet_NaN(), 1.f};
		for (unsigned iteration = 0; iteration < alphas.size(); ++iteration)
		{
			fragment_context fc;
			fc.sample_mask = filtered ? 5u : 0xffffffffu;
			fc.alpha_test = iteration == 9;
			fc.alpha_function = 0;
			for (unsigned row = 0; row < 32; ++row)
				fc.stipple_pattern[row] = std::rotl(0x925a6b49u, row);
			m_ctx.begin();
			texture_region whole;
			whole.size = MTLSizeMake(width, height, 1);
			if (!colorless)
			{
				m_colors.upload(m_ctx, format, input, {}, expanded, whole);
				if (count > 1)
					m_samples.import(m_ctx, expanded, target);
			}
			vertex_fetch_request request;
			request.vertex_count = 3;
			request.attribute_mask = 1;
			request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
			vertex_inputs inputs;
			inputs.attributes = m_fetch.decode(m_ctx, request, m_vertices, nil);
			vertex_context vc;
			vc.vertex_count = 3;
			const auto vb = m_vertex->prepare(m_ctx, vc, m_geometry, inputs), fb = fragment.prepare(m_ctx, fc, program(alphas[iteration], full), {});
			auto pass = [MTL4RenderPassDescriptor new];
			if (!colorless)
			{
				pass.colorAttachments[0].texture = target;
				pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			}
			pass.depthAttachment.texture = depth;
			pass.depthAttachment.loadAction = MTLLoadActionClear;
			pass.depthAttachment.clearDepth = 0.125;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
			render_commands draw(m_ctx, pass);
			raster_state raster;
			raster.viewport = {3, 2, 125, 94, 0, 1};
			raster.scissor = {5, 3, 118, 90};
			draw.draw_geometry(m_geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb);
			draw.end();
			if (!colorless)
			{
				if (count > 1)
					m_samples.expand(m_ctx, target, expanded);
				m_colors.download(m_ctx, format, expanded, whole, output, {});
			}
			if (count > 1)
				m_samples.expand(m_ctx, depth, expanded_depth);
			texture_to_buffer(m_ctx, expanded_depth, whole, depth_output, 0, width * 4);
			m_ctx.submit().wait();
			const auto depths = map_buffer(depth_output, 0, depth_output.length), colors = map_buffer(output, 0, output.length);
			std::vector<unsigned char> mask(width * height);
			std::size_t eligible = 0, covered = 0;
			for (unsigned y = 0; y < height; ++y)
				for (unsigned x = 0; x < width; ++x)
				{
					unsigned index = y * width + x, sample = x % sx + sx * (y % sy), px = x / sx, py = y / sy;
					bool allowed = px >= 5 && px < 123 && py >= 3 && py < 93 && (fc.sample_mask & (1u << sample));
					if (filtered)
						allowed = allowed && (fc.stipple_pattern[py % 32] & (1u << (px % 32)));
					unsigned depth_word;
					std::memcpy(&depth_word, depths.data() + index * 4, 4);
					require(depth_word == std::bit_cast<unsigned>(0.125f) || depth_word == std::bit_cast<unsigned>(0.5f), "Coverage corrupted stored depth");
					bool visible = depth_word == std::bit_cast<unsigned>(0.5f);
					mask[index] = visible;
					require(!visible || allowed, "Coverage wrote outside scissor/stipple/sample mask");
					eligible += allowed;
					covered += visible;
					if (!colorless)
					{
						auto pixel = colors.subspan(index * bytes, bytes);
						bool marked = format == surface_color::b8 ? pixel[0] == std::byte{64} : format == surface_color::rgba16f ? pixel[0] == std::byte{0x38} && pixel[1] == std::byte{0}
																																 : pixel[1] == std::byte{128};
						require(marked == visible, "Stochastic discard did not update color and depth together");
						if (!visible)
							require(std::all_of(pixel.begin(), pixel.end(), [](auto v)
												{
													return v == std::byte{0};
												}),
									"Coverage modified rejected color bytes");
					}
				}
			if (iteration == 0 || iteration == 7 || iteration == 8 || iteration == 9)
				require(covered == 0, "Zero/negative/NaN alpha or alpha-test rejection produced coverage");
			if (iteration == 5 || iteration == 6)
				require(covered == eligible, "Unit/greater-than-unit alpha lost eligible coverage");
			if (iteration >= 1 && iteration <= 4)
			{
				double ratio = double(covered) / double(eligible), expected = alphas[iteration];
				maximum_density_error = std::max(maximum_density_error, std::abs(ratio - expected));
				require(std::abs(ratio - expected) < 0.04, "Stochastic alpha coverage density differs from alpha");
				if (iteration > 1)
					for (unsigned i = 0; i < mask.size(); ++i)
						require(!previous[i] || mask[i], "Coverage is not monotonic for increasing alpha");
				if (iteration == 3)
					require(mask == previous, "Equal alpha produced a different coverage mask");
			}
			previous = std::move(mask);
			++cases;
		}
	}
};
} // namespace
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			fixture test;
			for (unsigned count : {1u, 2u, 4u})
				for (bool full : {true, false})
					for (bool filtered : {false, true})
					{
						for (auto format : {surface_color::argb8, surface_color::b8, surface_color::xrgb8_one, surface_color::rgba16f})
							for (bool blend : {false, true})
								test.run(format, count, full, blend, filtered, false);
						test.run(surface_color::argb8, count, full, false, filtered, true);
					}
			std::cout << test.cases << " native stochastic coverage draws verify endpoints/NaN, density, monotonic/deterministic masks, pre-blend/remap alpha, stencil-free color/depth correlation, colorless passes and 1/2/4 samples\n";
			std::cout << "Maximum observed coverage density error: " << test.maximum_density_error << " (limit 0.04); exact masks for endpoints/repeated alpha and exact color/depth correlation\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
