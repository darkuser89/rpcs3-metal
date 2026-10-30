#include "M4DepthBoundsPass.h"
#include "M4VertexShader.h"
#include "M4VertexFetch.h"
#include "M4SurfaceColor.h"
#include "M4SampleTransfer.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

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
std::vector<std::byte> program(float z)
{
	std::vector<std::byte> code;
	for (unsigned reg : {1u, 0u})
	{
		for (unsigned word : {(1u << 24) | (reg << 1) | (15u << 9) | (reg == 0 ? 1u : 0u), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(code, word);
		for (float value : reg ? std::array<float, 4>{0, 0, z, 0} : std::array<float, 4>{0.125f, 0.25f, 0.375f, 0.5f})
			append(code, std::bit_cast<unsigned>(value));
	}
	return code;
}
float read_depth(std::span<const std::byte> data, unsigned offset, MTLPixelFormat format)
{
	if (format == MTLPixelFormatDepth16Unorm)
	{
		std::uint16_t word;
		std::memcpy(&word, data.data() + offset, 2);
		return float(word) / 65535.f;
	}
	float value;
	std::memcpy(&value, data.data() + offset, 4);
	return value;
}
float read_float(std::span<const std::byte> data, unsigned offset)
{
	float value;
	std::memcpy(&value, data.data() + offset, 4);
	return value;
}
class fixture
{
	context m_ctx{1};
	compiler m_compiler{m_ctx};
	depth_bounds_engine m_bounds{m_ctx};
	vertex_fetch_engine m_fetch{m_ctx};
	sample_transfer m_samples{m_ctx};
	std::unique_ptr<vertex_shader> m_vertex;
	id<MTLBuffer> m_vertices;
	converted_geometry m_geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);

  public:
	unsigned cases = 0;
	fixture()
	{
		constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
		const std::array<std::array<unsigned, 4>, 1> code{{{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
		m_vertex = std::make_unique<vertex_shader>(m_compiler, code);
		std::vector<std::byte> bytes;
		for (float value : {-1.f, -1.f, 0.75f, 1.f, 3.f, -1.f, 0.75f, 1.f, -1.f, 3.f, 0.75f, 1.f})
			for (unsigned shift : {24u, 16u, 8u, 0u})
				bytes.push_back(static_cast<std::byte>(std::bit_cast<unsigned>(value) >> shift));
		m_vertices = make_upload(m_ctx, bytes);
	}
	void run(MTLPixelFormat format, unsigned count, unsigned kind)
	{
		constexpr unsigned native_width = 8, native_height = 6;
		const unsigned sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1, width = native_width * sx, height = native_height * sy, pitch = 256;
		const bool stencil = format == MTLPixelFormatDepth32Float_Stencil8;
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		const auto expanded_depth = make_texture(m_ctx, td);
		td.width = native_width;
		td.height = native_height;
		td.sampleCount = count;
		if (count > 1)
			td.textureType = MTLTextureType2DMultisample;
		id<MTLTexture> depth = count == 1 ? expanded_depth : make_texture(m_ctx, td);
		if (kind == 16)
		{
			td.textureType = MTLTextureType2DArray;
			td.sampleCount = 1;
			td.width = native_width * 2;
			td.height = native_height * 2;
			td.mipmapLevelCount = 2;
			td.arrayLength = 3;
			depth = make_texture(m_ctx, td);
			td.mipmapLevelCount = 1;
			td.arrayLength = 1;
		}
		td.textureType = MTLTextureType2D;
		td.sampleCount = 1;
		td.width = width;
		td.height = height;
		td.pixelFormat = MTLPixelFormatRGBA32Float;
		const auto expanded_color = make_texture(m_ctx, td);
		td.width = native_width;
		td.height = native_height;
		td.sampleCount = count;
		if (count > 1)
			td.textureType = MTLTextureType2DMultisample;
		const auto color = count == 1 ? expanded_color : make_texture(m_ctx, td);
		td.textureType = MTLTextureType2D;
		td.sampleCount = 1;
		td.width = width;
		td.height = height;
		td.pixelFormat = MTLPixelFormatR32Float;
		const auto expanded_mirror = make_texture(m_ctx, td);
		const auto before = make_buffer(m_ctx, pitch * height, buffer_storage::shared), after = make_buffer(m_ctx, pitch * height, buffer_storage::shared), mirror_output = make_buffer(m_ctx, pitch * height, buffer_storage::shared), color_output = make_buffer(m_ctx, pitch * height, buffer_storage::shared), stencil_output = make_buffer(m_ctx, pitch * height, buffer_storage::shared), stencil_before = make_buffer(m_ctx, pitch * height, buffer_storage::shared);
		auto original = [MTL4RenderPassDescriptor new];
		original.colorAttachments[0].texture = color;
		original.colorAttachments[0].loadAction = MTLLoadActionClear;
		original.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
		original.colorAttachments[0].storeAction = MTLStoreActionStore;
		original.depthAttachment.texture = depth;
		original.depthAttachment.loadAction = MTLLoadActionClear;
		original.depthAttachment.clearDepth = (kind == 1 || kind == 4) ? 0.75 : 0.25;
		original.depthAttachment.storeAction = MTLStoreActionStore;
		if (stencil)
		{
			original.stencilAttachment.texture = depth;
			original.stencilAttachment.loadAction = MTLLoadActionClear;
			original.stencilAttachment.clearStencil = 17;
			original.stencilAttachment.storeAction = MTLStoreActionStore;
		}
		if (kind == 16)
		{
			original.depthAttachment.level = 1;
			original.depthAttachment.slice = 2;
			if (stencil)
			{
				original.stencilAttachment.level = 1;
				original.stencilAttachment.slice = 2;
			}
		}
		render_pipeline_configuration config;
		config.state.sample_count = count;
		config.state.depth_test = kind != 7;
		config.state.depth_write = kind != 5;
		config.state.depth_compare = kind == 2 ? 0x201 : 0x207;
		config.state.colors[0].enabled = true;
		config.state.colors[0].destination_rgb = 1;
		config.state.colors[0].destination_alpha = 1;
		config.state.stencil_test = stencil;
		config.state.front.compare = (kind == 3 || kind == 4 || kind == 8) ? 0x200 : 0x207;
		config.state.front.fail = 0x1e02;
		config.state.front.depth_fail = 0x1e02;
		config.state.front.pass = 0x1e00;
		fragment_program_options options;
		options.depth_export = kind != 11 && kind != 12 && kind != 13;
		configure_surface_color(config, options, 0, surface_color::rgba32f, 15);
		configure_shader_blending(config, options);
		const auto code = program((kind == 1 || kind == 4) ? 0.25f : 0.75f);
		m_ctx.begin();
		vertex_fetch_request request;
		request.vertex_count = 3;
		request.attribute_mask = 1;
		request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
		vertex_inputs inputs;
		inputs.attributes = m_fetch.decode(m_ctx, request, m_vertices, nil);
		vertex_context vc;
		vc.vertex_count = 3;
		const auto vb = m_vertex->prepare(m_ctx, vc, m_geometry, inputs);
		texture_region region;
		region.size = MTLSizeMake(width, height, 1);
		texture_region selected = region;
		if (kind == 16)
		{
			selected.level = 1;
			selected.slice = 2;
		}
		if (kind == 14 || kind == 15)
		{
			std::vector<std::byte> initial_depth(pitch * height), initial_stencil(pitch * height);
			for (unsigned y = 0; y < height; ++y)
				for (unsigned x = 0; x < width; ++x)
				{
					float value = std::array<float, 4>{0.125f, 0.25f, 0.5f, 0.875f}[(x + y * width) % 4];
					if (kind == 15 && (x + y) % 3 == 0)
						value = std::bit_cast<float>(0x7fc12345u);
					if (format == MTLPixelFormatDepth16Unorm)
					{
						std::uint16_t word = std::uint16_t(std::nearbyint(value * 65535.f));
						std::memcpy(initial_depth.data() + y * pitch + x * 2, &word, 2);
					}
					else
						std::memcpy(initial_depth.data() + y * pitch + x * 4, &value, 4);
					initial_stencil[y * pitch + x] = static_cast<std::byte>(17 + (x + y) % 7);
				}
			const auto upload = make_upload(m_ctx, initial_depth), stencil_upload = make_upload(m_ctx, initial_stencil);
			buffer_to_texture(m_ctx, upload, 0, expanded_depth, region, pitch, 0, texture_plane::depth);
			if (stencil)
				buffer_to_texture(m_ctx, stencil_upload, 0, expanded_depth, region, pitch, 0, texture_plane::stencil);
			if (count > 1)
				m_samples.import(m_ctx, expanded_depth, depth);
			original.depthAttachment.loadAction = MTLLoadActionLoad;
			if (stencil)
				original.stencilAttachment.loadAction = MTLLoadActionLoad;
		}
		auto pass = m_bounds.prepare(m_ctx, original);
		pass.configure_pipeline(config, options);
		fragment_shader fragment(m_compiler, code, options);
		const auto pipeline = make_render_pipeline(m_compiler, config, m_vertex->function(), fragment);
		// Independent expected model starts from the real native cleared bytes,
		// avoiding assumptions about D16 load-action conversion rounding.
		if (count > 1)
			m_samples.expand(m_ctx, depth, expanded_depth);
		texture_to_buffer(m_ctx, kind == 16 ? depth : expanded_depth, kind == 16 ? selected : region, before, 0, pitch, 0, texture_plane::depth);
		if (stencil)
			texture_to_buffer(m_ctx, kind == 16 ? depth : expanded_depth, kind == 16 ? selected : region, stencil_before, 0, pitch, 0, texture_plane::stencil);
		raster_state raster;
		raster.viewport = {0, 0, native_width, native_height, 0, 1};
		raster.scissor = {1, 1, 6, 4};
		raster.depth_bounds_enabled = kind != 6 && kind != 12 && kind != 13;
		raster.depth_min = kind == 10 ? 0.25f : 0.2f;
		raster.depth_max = kind == 10 ? 0.25f : 0.3f;
		if (kind == 12)
			raster.depth_bias = 4096;
		if (kind == 13)
			raster.depth_bias = -4096;
		fragment_context fc;
		if (kind == 8)
		{
			fc.alpha_test = 1;
			fc.alpha_function = 0;
		}
		if (kind == 9)
			fc.sample_mask = 5;
		pass.draw_geometry(m_geometry, pipeline, fragment, vb, fc, code, {}, raster, 2);
		pass.end();
		if (count > 1)
		{
			m_samples.expand(m_ctx, depth, expanded_depth);
			m_samples.expand(m_ctx, color, expanded_color);
			m_samples.expand(m_ctx, pass.mirror_texture(), expanded_mirror);
		}
		texture_to_buffer(m_ctx, kind == 16 ? depth : expanded_depth, kind == 16 ? selected : region, after, 0, pitch, 0, texture_plane::depth);
		texture_to_buffer(m_ctx, count == 1 ? pass.mirror_texture() : expanded_mirror, region, mirror_output, 0, pitch);
		texture_to_buffer(m_ctx, expanded_color, region, color_output, 0, pitch);
		if (stencil)
			texture_to_buffer(m_ctx, kind == 16 ? depth : expanded_depth, kind == 16 ? selected : region, stencil_output, 0, pitch, 0, texture_plane::stencil);
		m_ctx.submit().wait();
		const auto initial = map_buffer(before, 0, before.length), actual = map_buffer(after, 0, after.length), mirrored = map_buffer(mirror_output, 0, mirror_output.length), colors = map_buffer(color_output, 0, color_output.length), stencils = map_buffer(stencil_output, 0, stencil_output.length), initial_stencil = map_buffer(stencil_before, 0, stencil_before.length);
		for (unsigned y = 0; y < height; ++y)
			for (unsigned x = 0; x < width; ++x)
			{
				const unsigned depth_bytes = format == MTLPixelFormatDepth16Unorm ? 2 : 4;
				float stored = read_depth(initial, y * pitch + x * depth_bytes, format), expected = stored;
				unsigned marker = 0, svalue = stencil ? unsigned(initial_stencil[y * pitch + x]) : 17;
				const bool spatial = x / sx >= 1 && x / sx < 7 && y / sy >= 1 && y / sy < 5;
				const unsigned sample = x % sx + sx * (y % sy);
				for (unsigned instance = 0; instance < 2 && spatial; ++instance)
				{
					if (raster.depth_bounds_enabled && !(expected >= raster.depth_min && expected <= raster.depth_max))
						continue;
					if (kind == 8 || !(fc.sample_mask & (1u << sample)))
						continue;
					if (stencil && (kind == 3 || kind == 4))
					{
						++svalue;
						continue;
					}
					float incoming = (kind == 1 || kind == 4) ? 0.25f : 0.75f;
					if (format == MTLPixelFormatDepth16Unorm)
						incoming = std::nearbyint(incoming * 65535.f) / 65535.f;
					if (kind == 2 && !(incoming < expected))
					{
						if (stencil)
							++svalue;
						continue;
					}
					++marker;
					if (kind != 5 && kind != 7)
						expected = incoming;
				}
				const float z = read_depth(actual, y * pitch + x * depth_bytes, format), mirror = read_float(mirrored, y * pitch + x * 4), mark = read_float(colors, y * pitch + x * 16);
				// Bias path checks actual native depth against the live mirror; the
				// unbounded raster depth itself is hardware-generated, not guessed.
				if (kind != 12 && kind != 13)
					require(std::bit_cast<unsigned>(z) == std::bit_cast<unsigned>(expected), "Actual depth differs from independent stored-depth bounds model");
				if ((z != mirror && !(std::isnan(z) && std::isnan(mirror))) || mark != float(marker) * 0.125f || (stencil && unsigned(stencils[y * pitch + x]) != svalue))
				{
					std::cerr << "Bounds format=" << unsigned(format) << " samples=" << count << " kind=" << kind << " x=" << x << " y=" << y << " depth=" << z << " mirror=" << mirror << " mark=" << mark << " expected mark=" << marker * 0.125f << " stencil=" << unsigned(stencils[y * pitch + x]) << " expected stencil=" << svalue << '\n';
					throw std::runtime_error("Native depth mirror/history/late stencil gating differs");
				}
			}
		++cases;
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
			for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
				for (unsigned count : {1u, 2u, 4u})
					for (unsigned kind = 0; kind < 17; ++kind)
					{
						if (kind == 15 && (count > 1 || format == MTLPixelFormatDepth16Unorm))
							continue;
						if (kind == 16 && count > 1)
							continue;
						test.run(format, count, kind);
					}
			std::cout << test.cases << " native stored-depth mirror draws verify live intra-draw bounds, depth/stencil failure ordering, no-write/disabled states, sample masks, alpha rejection, D16/D32/D32S8, load/clear, NaN, mip/array slices and bias consistency\n";
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
