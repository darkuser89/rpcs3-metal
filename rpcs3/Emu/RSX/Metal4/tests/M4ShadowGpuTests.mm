#include "M4FragmentShader.h"
#include "M4RenderPipeline.h"
#include "M4Texture.h"
#include "M4Sampling.h"
#include "M4ShaderCache.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <tuple>

namespace
{
using namespace metal4;
using vector = std::array<float, 4>;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void append(std::vector<std::byte>& bytes, unsigned value)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		bytes.push_back(static_cast<std::byte>(value >> shift));
}
std::vector<std::byte> program(unsigned opcode, vector coordinate, vector gradient_x = {}, vector gradient_y = {}, unsigned unit = 0, bool varying = false)
{
	std::vector<std::byte> code;
	unsigned reg = 1;
	for (auto literal : {coordinate, gradient_x, gradient_y})
	{
		for (unsigned word : {(1u << 24) | (reg << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(code, word);
		for (float value : literal)
			append(code, std::bit_cast<unsigned>(value));
		++reg;
	}
	if (varying)
	{
		// Window-position input -> r4; r1 = r4 * r2 + literal r1.
		for (unsigned word : {(1u << 24) | (4u << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 1u, 0xe4u << 9, 0xe4u << 9})
			append(code, word);
		for (unsigned word : {(4u << 24) | (1u << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 16u, (0xe4u << 9) | 8u, (0xe4u << 9) | 4u})
			append(code, word);
	}
	for (unsigned word : {(opcode << 24) | (unit << 17) | (15u << 9) | 1u, (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 4u, (0xe4u << 9) | 8u, (0xe4u << 9) | 12u})
		append(code, word);
	return code;
}
bool comparison(unsigned op, float depth, float reference)
{
	switch (op)
	{
	case 0:
		return false;
	case 1:
		return depth < reference;
	case 2:
		return depth == reference;
	case 3:
		return depth <= reference;
	case 4:
		return depth > reference;
	case 5:
		return depth != reference;
	case 6:
		return depth >= reference;
	default:
		return true;
	}
}
class fixture
{
	context ctx{1};
	compiler compile{ctx};
	shader_cache custom_shaders{compile};
	render_pipeline_cache custom_pipelines{compile};
	sampler_cache samplers{ctx, 64};
	id<MTLLibrary> vertex_library;
	id<MTLTexture> target;
	id<MTLBuffer> output;
	converted_geometry geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	std::array<std::array<std::unique_ptr<fragment_shader>, 5>, 3> fragments;
	std::array<std::array<std::optional<render_pipeline_bundle>, 5>, 3> pipelines;
	struct depth_pipeline
	{
		std::unique_ptr<fragment_shader> shader;
		render_pipeline_bundle pipeline;
	};
	std::map<std::tuple<vertex_texture_dimension, unsigned, unsigned>, depth_pipeline> depth_pipelines;
	unsigned depth_cases = 0;
	float max_depth_error = 0;
	float max_depth16_filter_error = 0;
	unsigned custom_cases = 0;
	float max_custom_error = 0;
	float max_custom_pcf_error = 0;

  public:
	unsigned cases = 0;
	float max_pcf_error = 0;
	float max_coordinate_pcf_error = 0;
	fixture()
	{
		vertex_library = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
vertex float4 shadow_vertex(uint id [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[id],0,1); }
)metal",
										 @"Native RSX shadow regression vertex");
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageRenderTarget;
		target = make_texture(ctx, td);
		output = make_buffer(ctx, 256 * 4, buffer_storage::shared);
	}
	id<MTLTexture> texture(MTLPixelFormat format, vertex_texture_dimension dimension, bool mipmaps = false, int extended = 0)
	{
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:4 height:dimension == vertex_texture_dimension::one ? 1 : 4 mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		if (dimension == vertex_texture_dimension::cube)
			td.textureType = MTLTextureTypeCube;
		if (mipmaps)
			td.mipmapLevelCount = 3;
		const auto result = make_texture(ctx, td);
		ctx.begin();
		for (unsigned level = 0; level < td.mipmapLevelCount; ++level)
			for (unsigned face = 0; face < (dimension == vertex_texture_dimension::cube ? 6u : 1u); ++face)
			{
				const unsigned width = 4 >> level, height = dimension == vertex_texture_dimension::one ? 1 : width;
				const unsigned bytes = format == MTLPixelFormatDepth16Unorm ? 2 : 4;
				std::vector<std::byte> depth(width * height * bytes), stencil(width * height, std::byte{17});
				for (unsigned y = 0; y < height; ++y)
					for (unsigned x = 0; x < width; ++x)
					{
						const float value = mipmaps ? std::array{.25f, .5f, .75f}[level] : extended > 0								 ? std::array{.5f, 1.125f, 1.5f, 2.f}[x % 4]
																					   : extended < 0								 ? std::array{-.5f, -.125f, .125f, .5f}[x % 4]
																					   : dimension == vertex_texture_dimension::cube ? float(face + 1) / 8
																																	 : std::array{.125f, .375f, .625f, .875f}[(x + y) % 4];
						if (bytes == 2)
						{
							const std::uint16_t word = std::uint16_t(std::nearbyint(value * 65535));
							std::memcpy(depth.data() + (y * width + x) * bytes, &word, 2);
						}
						else
							std::memcpy(depth.data() + (y * width + x) * bytes, &value, 4);
					}
				texture_region region;
				region.slice = face;
				region.level = level;
				region.size = MTLSizeMake(width, height, 1);
				buffer_to_texture(ctx, make_upload(ctx, depth), 0, result, region, width * bytes, 0, texture_plane::depth);
				if (format == MTLPixelFormatDepth32Float_Stencil8)
					buffer_to_texture(ctx, make_upload(ctx, stencil), 0, result, region, width, 0, texture_plane::stencil);
			}
		ctx.submit().wait();
		return result;
	}
	void verify_shadow_helper(id<MTLTexture> source)
	{
		const auto helper = samplers.border_shadow_view(source, .375f);
		require(samplers.border_shadow_view(source, .375f) == helper, "Constant depth helper was not reused");
		require(samplers.border_shadow_view(source, .625f) != helper, "Different depth borders shared a constant helper");
		require(helper.pixelFormat == (source.pixelFormat == MTLPixelFormatDepth16Unorm ? MTLPixelFormatDepth16Unorm : MTLPixelFormatDepth32Float), "Shadow helper lost comparison storage precision");
		for (unsigned level = 0; level < helper.mipmapLevelCount; ++level)
			for (unsigned face = 0; face < (helper.textureType == MTLTextureTypeCube ? 6u : 1u); ++face)
			{
				texture_region region;
				region.level = level;
				region.slice = face;
				region.size = MTLSizeMake(std::max<NSUInteger>(1, helper.width >> level), std::max<NSUInteger>(1, helper.height >> level), 1);
				ctx.begin();
				texture_to_buffer(ctx, helper, region, output, 0, 256, 0, texture_plane::depth);
				ctx.submit().wait();
				for (unsigned y = 0; y < region.size.height; ++y)
					for (unsigned x = 0; x < region.size.width; ++x)
					{
						const auto* data = static_cast<const std::byte*>(output.contents) + y * 256;
						if (helper.pixelFormat == MTLPixelFormatDepth16Unorm)
						{
							std::uint16_t value;
							std::memcpy(&value, data + x * 2, 2);
							require(value == std::uint16_t(std::nearbyint(.375f * 65535)), "Native D16 helper clear changed border encoding");
						}
						else
						{
							float value;
							std::memcpy(&value, data + x * 4, 4);
							require(value == .375f, "Native shadow helper mip/face was not initialized");
						}
					}
			}
	}
	void verify_stencil(id<MTLTexture> texture)
	{
		if (texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
			return;
		for (unsigned level = 0; level < texture.mipmapLevelCount; ++level)
			for (unsigned face = 0; face < (texture.textureType == MTLTextureTypeCube ? 6u : 1u); ++face)
			{
				const auto width = std::max<std::size_t>(1, texture.width >> level), height = std::max<std::size_t>(1, texture.height >> level);
				const auto data = make_buffer(ctx, 256 * height, buffer_storage::shared);
				ctx.begin();
				texture_region region;
				region.level = level;
				region.slice = face;
				region.size = MTLSizeMake(width, height, 1);
				texture_to_buffer(ctx, texture, region, data, 0, 256, 0, texture_plane::stencil);
				ctx.submit().wait();
				for (unsigned y = 0; y < height; ++y)
					for (unsigned x = 0; x < width; ++x)
						require(static_cast<const std::uint8_t*>(data.contents)[y * 256 + x] == 17, "Shadow sampling changed source stencil bits");
			}
	}
	void run(id<MTLTexture> depth, vertex_texture_dimension dimension, unsigned opcode, unsigned compare, bool linear, float reference, float u = .375f, float v = .375f, bool depth_float = false, float lod = 0, float bias = 0, float expected = -1, vector direction = {1, 0, 0, 0}, float min_lod = 0, texture_coordinates coordinates = {}, float tolerance = 2e-6f, texel_processing processing = {})
	{
		const unsigned dim = dimension == vertex_texture_dimension::one ? 0 : dimension == vertex_texture_dimension::two ? 1
																														 : 2;
		const unsigned op = opcode == 23 ? 0 : opcode == 24 ? 1
										   : opcode == 25	? 2
										   : opcode == 47	? 3
															: 4;
		vector coordinate = dimension == vertex_texture_dimension::cube ? vector{direction[0], direction[1], direction[2], reference} : vector{u, v, reference, 1};
		if (opcode == 24)
			for (auto& value : coordinate)
				value *= 2;
		vector gradient_x = opcode == 25 ? vector{std::exp2(lod) / 4, 0, 0, 0} : vector{lod, 0, 0, 0};
		vector gradient_y = opcode == 25 ? vector{0, dimension == vertex_texture_dimension::one ? 0.f : std::exp2(lod) / 4, 0, 0} : vector{};
		if (opcode == 25 && dimension == vertex_texture_dimension::cube)
		{
			gradient_x = {0, 0, std::exp2(lod) / 2, 0};
			gradient_y = {0, std::exp2(lod) / 2, 0, 0};
		}
		const auto code = program(opcode, coordinate, gradient_x, gradient_y);
		if (!fragments[dim][op])
		{
			fragment_program_options options;
			options.shadow_textures = 1;
			options.textures[0] = dimension;
			fragments[dim][op] = std::make_unique<fragment_shader>(compile, code, options);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			pipelines[dim][op] = make_render_pipeline(compile, config, library_function(vertex_library, @"shadow_vertex"), *fragments[dim][op]);
		}
		guest_sampler_state state;
		state.depth_compare = true;
		state.depth_float = depth_float;
		state.compare_function = 0x200 + compare;
		state.wrap = {sampler_wrap::edge, sampler_wrap::edge, sampler_wrap::edge};
		state.min_filter = linear ? sampler_min_filter::linear_linear : sampler_min_filter::nearest_nearest;
		state.mag_filter = linear ? sampler_mag_filter::linear : sampler_mag_filter::nearest;
		state.max_lod = depth.mipmapLevelCount - 1;
		state.min_lod = min_lod;
		state.bias = bias;
		sampler_texture_policy policy;
		policy.dimension = dimension;
		policy.requested_mips = policy.actual_mips = depth.mipmapLevelCount;
		fragment_inputs inputs;
		inputs.textures[0] = depth;
		inputs.coordinates[0] = coordinates;
		inputs.texels[0] = processing;
		inputs.set_sampler(0, samplers.acquire(state, policy));
		ctx.begin();
		const auto binding = fragments[dim][op]->prepare(ctx, {}, code, inputs);
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		if (processing.control)
		{
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(.25, .25, .25, .25);
		}
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		render_commands draw(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, 4, 4, 0, 1};
		raster.scissor = {0, 0, 4, 4};
		const auto& pipeline = *pipelines[dim][op];
		draw.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, {}, binding);
		draw.end();
		texture_region region;
		region.size = MTLSizeMake(4, 4, 1);
		texture_to_buffer(ctx, target, region, output, 0, 256);
		ctx.submit().wait();
		if (expected < 0)
		{
			u = std::fma(u, coordinates.scale[0], coordinates.bias[0]);
			v = std::fma(v, coordinates.scale[1], coordinates.bias[1]);
			if ((coordinates.flags & texture_coordinate_clamp) && dimension != vertex_texture_dimension::cube)
			{
				u = std::clamp(u, coordinates.minimum[0], coordinates.maximum[0]);
				v = std::clamp(v, coordinates.minimum[1], coordinates.maximum[1]);
			}
			const float ref = depth_float ? reference : std::min(reference, 1.f);
			auto test = [&](int x, int y)
			{
				x = std::clamp(x, 0, 3);
				y = dimension == vertex_texture_dimension::one ? 0 : std::clamp(y, 0, 3);
				float value = dimension == vertex_texture_dimension::cube ? .125f : std::array{.125f, .375f, .625f, .875f}[(x + y) % 4];
				if (depth.pixelFormat == MTLPixelFormatDepth16Unorm)
					value = float(std::nearbyint(value * 65535)) / 65535;
				return float(comparison(compare, value, ref));
			};
			if (!linear || dimension == vertex_texture_dimension::cube)
				expected = test(int(std::floor(u * 4)), int(std::floor(v * 4)));
			else
			{
				const float x = u * 4 - .5f, y = dimension == vertex_texture_dimension::one ? 0 : v * 4 - .5f;
				const int ix = int(std::floor(x)), iy = int(std::floor(y));
				const float fx = x - ix, fy = y - iy;
				expected = (test(ix, iy) * (1 - fx) + test(ix + 1, iy) * fx) * (1 - fy) + (test(ix, iy + 1) * (1 - fx) + test(ix + 1, iy + 1) * fx) * fy;
			}
		}
		const auto data = static_cast<const float*>(output.contents);
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				for (unsigned c = 0; c < 4; ++c)
					if (!std::isfinite(data[y * 64 + x * 4 + c]) || std::abs(data[y * 64 + x * 4 + c] - expected) > tolerance)
						throw std::runtime_error("Native RSX shadow mismatch:format=" + std::to_string(depth.pixelFormat) + " dim=" + std::to_string(dim) + " opcode=" + std::to_string(opcode) + " compare=" + std::to_string(compare) + " ref=" + std::to_string(reference) + " float=" + std::to_string(depth_float) + " expected=" + std::to_string(expected) + " actual=" + std::to_string(data[y * 64 + x * 4 + c]));
		++cases;
		if (tolerance > 2e-6f)
			max_coordinate_pcf_error = std::max(max_coordinate_pcf_error, std::abs(data[0] - expected));
		if (linear && depth.mipmapLevelCount > 1)
		{
			max_pcf_error = std::max(max_pcf_error, std::abs(data[0] - expected));
		}
	}
	void run_depth(id<MTLTexture> source, vertex_texture_dimension dimension, unsigned opcode, vector coordinate,
		vector expected, float lod = 0, float bias = 0, float min_lod = 0, bool linear = false,
		depth_sample_remap remap = {}, texel_processing processing = {}, texture_coordinates coordinates = {}, bool killed = false, const decoded_texture_state* register_state = nullptr)
	{
		vector gx = {lod, 0, 0, 0}, gy{};
		if (opcode == 25)
		{
			gx = {std::exp2(lod) / 4, 0, 0, 0};
			gy = {0, dimension == vertex_texture_dimension::one ? 0.f : std::exp2(lod) / 4, 0, 0};
			if (dimension == vertex_texture_dimension::cube)
			{
				gx = {0, 0, std::exp2(lod) / 2, 0};
				gy = {0, std::exp2(lod) / 2, 0, 0};
			}
		}
		if (opcode == 24)
			for (auto& c : coordinate)
				c *= 2;
		const auto code = program(opcode, coordinate, gx, gy);
		unsigned remap_key = 0;
		for (auto c : remap)
			remap_key = remap_key * 3 + static_cast<unsigned>(c);
		const auto key = std::tuple{dimension, opcode, remap_key};
		if (!depth_pipelines.contains(key))
		{
			fragment_program_options options;
			options.depth_textures = 1;
			options.textures[0] = dimension;
			options.depth_remap[0] = remap;
			if (register_state)
				configure_fragment_texture_state(options, 0, *register_state);
			auto shader = std::make_unique<fragment_shader>(compile, code, options);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			auto pipeline = make_render_pipeline(compile, config, library_function(vertex_library, @"shadow_vertex"), *shader);
			depth_pipelines.emplace(key, depth_pipeline{std::move(shader), std::move(pipeline)});
		}
		const auto& bundle = depth_pipelines.at(key);
		guest_sampler_state state;
		state.wrap = {sampler_wrap::edge, sampler_wrap::edge, sampler_wrap::edge};
		state.min_filter = linear ? sampler_min_filter::linear_linear : sampler_min_filter::nearest_nearest;
		state.mag_filter = linear ? sampler_mag_filter::linear : sampler_mag_filter::nearest;
		state.max_lod = source.mipmapLevelCount - 1;
		state.min_lod = min_lod;
		state.bias = bias;
		sampler_texture_policy policy;
		policy.dimension = dimension;
		policy.requested_mips = policy.actual_mips = source.mipmapLevelCount;
		fragment_inputs inputs;
		inputs.textures[0] = source;
		inputs.set_sampler(0, samplers.acquire(state, policy));
		inputs.texels[0] = processing;
		inputs.coordinates[0] = coordinates;
		if (register_state)
			inputs.set_texture_state(0, source, samplers, *register_state);
		ctx.begin();
		const auto binding = bundle.shader->prepare(ctx, {}, code, inputs);
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(.25, .25, .25, .25);
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		render_commands draw(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, 4, 4, 0, 1};
		raster.scissor = {0, 0, 4, 4};
		draw.draw_geometry(geometry, bundle.pipeline.pipeline, bundle.pipeline.depth_stencil, raster, {}, binding);
		draw.end();
		texture_region region;
		region.size = MTLSizeMake(4, 4, 1);
		texture_to_buffer(ctx, target, region, output, 0, 256);
		ctx.submit().wait();
		const bool quantized_filter = linear && source.pixelFormat == MTLPixelFormatDepth16Unorm;
		// D16 filtering need not preserve a full float interpolation result.
		// Keep the CPU interpolation reference and record the native deviation;
		// allow half a D16 step only for these explicit linear-filter cases.
		const float tolerance = quantized_filter ? .5f / 65535 + 2e-6f : 2e-6f;
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				for (unsigned c = 0; c < 4; ++c)
				{
					const float actual = static_cast<const float*>(output.contents)[y * 64 + x * 4 + c];
					const float error = std::abs(actual - (killed ? .25f : expected[c]));
					if (!std::isfinite(actual) || error > tolerance)
						throw std::runtime_error("Native non-comparison depth mismatch:case=" + std::to_string(depth_cases) + " opcode=" + std::to_string(opcode) + " format=" + std::to_string(source.pixelFormat) + " dimension=" + std::to_string(static_cast<unsigned>(dimension)) + " lane=" + std::to_string(c) + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected[c]));
					max_depth_error = std::max(max_depth_error, error);
					if (quantized_filter)
						max_depth16_filter_error = std::max(max_depth16_filter_error, error);
				}
		++depth_cases;
	}
	void depth_all()
	{
		const auto broadcast = [](float v) { return vector{v, v, v, v}; };
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::cube})
			for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				const auto source = texture(format, dimension);
				for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					for (bool linear : {false, true})
						for (unsigned i = 0; i < (dimension == vertex_texture_dimension::cube ? 6u : 4u); ++i)
						{
							static const std::array<vector, 6> directions = {{{1, 0, 0, 1}, {-1, 0, 0, 1}, {0, 1, 0, 1}, {0, -1, 0, 1}, {0, 0, 1, 1}, {0, 0, -1, 1}}};
							const vector coordinate = dimension == vertex_texture_dimension::cube ? directions[i] : vector{(float(i) + .5f) / 4, .125f, 0, 1};
							float expected = dimension == vertex_texture_dimension::cube ? float(i + 1) / 8 : (float(i) + .5f) / 4;
							if (format == MTLPixelFormatDepth16Unorm)
								expected = float(std::nearbyint(expected * 65535)) / 65535;
							run_depth(source, dimension, opcode, coordinate, broadcast(expected), 0, 0, 0, linear);
						}
				verify_stencil(source);
				if (dimension != vertex_texture_dimension::cube)
					for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					{
						const float wanted = dimension == vertex_texture_dimension::one ? .25f : .375f;
						float a = .125f, b = .375f, c = .625f;
						if (format == MTLPixelFormatDepth16Unorm)
						{
							a = float(std::nearbyint(a * 65535)) / 65535;
							b = float(std::nearbyint(b * 65535)) / 65535;
							c = float(std::nearbyint(c * 65535)) / 65535;
						}
						const float expected = format == MTLPixelFormatDepth16Unorm ? (dimension == vertex_texture_dimension::one ? (a + b) / 2 : (a + 2 * b + c) / 4) : wanted;
						run_depth(source, dimension, opcode, {.25f, .25f, 0, 1}, broadcast(expected), -8, 0, 0, true);
					}
				const auto mip = texture(format, dimension, true);
				verify_shadow_helper(mip);
				for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					for (unsigned level = 0; level < 3; ++level)
					{
						float expected = .25f * float(level + 1);
						if (format == MTLPixelFormatDepth16Unorm)
							expected = float(std::nearbyint(expected * 65535)) / 65535;
						const vector coordinate = dimension == vertex_texture_dimension::cube ? vector{1, 0, 0, 1} : vector{.375f, .375f, 0, 1};
						// Explicit LOD/gradients select the level; implicit fetches use
						// sampler minLOD because constant coordinates have zero footprint.
						const float minimum = opcode == 47 || opcode == 25 ? 0.f : float(level);
						run_depth(mip, dimension, opcode, coordinate, broadcast(expected), float(level), 0, minimum);
						if (opcode == 47 || opcode == 25)
							run_depth(mip, dimension, opcode, coordinate, broadcast(expected), float(level) - 1, 1);
					}
				const vector coordinate = dimension == vertex_texture_dimension::cube ? vector{1, 0, 0, 1} : vector{.375f, .375f, 0, 1};
				const auto unorm = [format](float value)
				{
					return format == MTLPixelFormatDepth16Unorm ? float(std::nearbyint(value * 65535)) / 65535 : value;
				};
				run_depth(mip, dimension, 47, coordinate, broadcast((unorm(.25f) + unorm(.5f)) / 2), .5f, 0, 0, true);
				run_depth(mip, dimension, 47, coordinate, broadcast(unorm(.75f)), 9);
				run_depth(mip, dimension, 47, coordinate, broadcast(unorm(.5f)), 0, 0, 1);
				run_depth(mip, dimension, 25, coordinate, broadcast(unorm(.5f)), 0, 0, 1);
				verify_stencil(mip);
			}
		const auto source = texture(MTLPixelFormatDepth32Float_Stencil8, vertex_texture_dimension::two);
		for (unsigned mapping = 0; mapping < 81; ++mapping)
		{
			depth_sample_remap remap;
			vector expected;
			unsigned code = mapping;
			for (unsigned c = 0; c < 4; ++c)
			{
				remap[c] = static_cast<depth_sample_channel>(code % 3);
				expected[c] = code % 3 == 0 ? .125f : code % 3 == 1 ? 0.f : 1.f;
				code /= 3;
			}
			run_depth(source, vertex_texture_dimension::two, 23, {.125f, .125f, 0, 1}, expected, 0, 0, 0, false, remap);
		}
		const depth_sample_remap remap = {depth_sample_channel::depth, depth_sample_channel::one, depth_sample_channel::zero, depth_sample_channel::zero};
		texel_processing processing;
		processing.control = texel_alpha_kill;
		run_depth(source, vertex_texture_dimension::two, 23, {.125f, .125f, 0, 1}, {}, 0, 0, 0, false, remap, processing, {}, true);
		processing.control = 0;
		processing.gamma_mask = 1;
		processing.expand_mask = 2;
		run_depth(source, vertex_texture_dimension::two, 23, {.125f, .125f, 0, 1}, {std::pow((.125f + .055f) / 1.055f, 2.4f), 1, 0, 0}, 0, 0, 0, false, remap, processing);
		processing = {};
		processing.sign_mask = 1;
		run_depth(source, vertex_texture_dimension::two, 23, {.625f, .125f, 0, 1}, {-97.f / 127, .625f, .625f, .625f}, 0, 0, 0, false, {}, processing);
		processing = {};
		processing.control = texel_renormalize;
		run_depth(source, vertex_texture_dimension::two, 23, {.125f, .125f, 0, 1}, broadcast(31.f / 255), 0, 0, 0, false, {}, processing);
		texture_coordinates coordinates;
		coordinates.scale = {-1, 0, 1};
		coordinates.bias = {1, .125f, 0};
		run_depth(source, vertex_texture_dimension::two, 24, {.125f, .5f, 0, 1}, broadcast(.875f), 0, 0, 0, false, {}, {}, coordinates);
		verify_stencil(source);
		for (auto format : {MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			for (int extended : {-1, 1})
			{
				const auto raw = texture(format, vertex_texture_dimension::two, false, extended);
				run_depth(raw, vertex_texture_dimension::two, 47, {extended < 0 ? .125f : .875f, .125f, 0, 1}, broadcast(extended < 0 ? -.5f : 2.f));
				verify_stencil(raw);
			}
		const auto sampler = samplers.acquire({});
		validate_sampled_texture(ctx, source, sampler.sampler, vertex_texture_dimension::two, sampler.parameters, false, true);
		for (unsigned invalid = 0; invalid < 4; ++invalid)
		{
			bool rejected = false;
			try
			{
				auto parameters = sampler.parameters;
				if (invalid == 1)
					parameters.flags |= sampler_depth_compare;
				validate_sampled_texture(ctx, invalid == 2 ? target : source, sampler.sampler, vertex_texture_dimension::two, parameters, invalid == 3, invalid != 0);
			}
			catch (const std::invalid_argument&)
			{
				rejected = true;
			}
			require(rejected, "Depth read type/comparison preflight accepted invalid binding");
		}
		const unsigned first = depth_cases;
		for (auto format : {MTLPixelFormatDepth16Unorm,MTLPixelFormatDepth32Float,MTLPixelFormatDepth32Float_Stencil8})
		{
			const auto image = texture(format,vertex_texture_dimension::two);
			for (unsigned remap_word : {0xaae4u,0xa9e4u,0xa8e4u,0x5555u})
			{
				fragment_texture_registers regs;
				const unsigned raw_format = format == MTLPixelFormatDepth16Unorm ? 0x92 : format == MTLPixelFormatDepth32Float ? 0x93 : 0x91;
				regs.format = (1u << 16) | (raw_format << 8) | (2u << 4) | 1;
				regs.control0 = (1u << 31) | 4; // Alpha-kill selects ordinary read,even with LESS.
				regs.control1 = remap_word;
				regs.filter = (1u << 24) | (3u << 16);
				regs.address = (1u << 28) | 0x030303;
				regs.image_rect = (4u << 16) | 4;
				texture_cache_state metadata;
				metadata.depth = true;
				metadata.depth_float = format != MTLPixelFormatDepth16Unorm;
				const auto decoded = decode_fragment_texture_state(regs,metadata);
				require(decoded.depth && !decoded.shadow,"Alpha-kill depth register state incorrectly selected shadow comparison");
				const float depth = format == MTLPixelFormatDepth16Unorm ? float(std::nearbyint(.125f*65535))/65535 : .125f;
				vector expected{depth,depth,depth,depth};
				if (remap_word == 0xa9e4) expected[3] = 1;
				if (remap_word == 0xa8e4) expected[3] = 0;
				if (remap_word == 0x5555) expected = {1,1,1,1};
				run_depth(image,vertex_texture_dimension::two,23,{.125f,.125f,0,1},expected,0,0,0,false,decoded.depth_remap,decoded.texels,decoded.coordinates,remap_word == 0xa8e4,&decoded);
			}
			verify_stencil(image);
		}
		std::cout << depth_cases-first << " native register->depth declaration/remap->sampler->buffer4/5 draws passed\n";
		std::cout << depth_cases << " native non-comparison depth draws:1D/2D/cube,D16/D32/D32S8,all five fetch opcodes,nearest/linear,LOD/gradient/bias/clamps,81 view remaps,post-remap alpha-kill/gamma/expand,coordinate projection;max error=" << max_depth_error << '\n';
		std::cout << "D16 depth linear-filter max error=" << max_depth16_filter_error << " (separate half-D16-step limit plus2e-6;other depth cases2e-6)\n";
	}
	std::array<vector, 16> custom_render(id<MTLTexture> image, const decoded_texture_state& state, std::span<const std::byte> code)
	{
		fragment_program_options options;
		configure_fragment_texture_state(options, 15, state);
		fragment_inputs inputs;
		inputs.set_texture_state(15, image, samplers, state);
		const auto shader = custom_shaders.fragment(code, options);
		render_pipeline_configuration config;
		config.colors[0] = MTLPixelFormatRGBA32Float;
		const auto pipeline = custom_pipelines.get(config, library_function(vertex_library, @"shadow_vertex"), *shader);
		ctx.begin();
		const auto binding = shader->prepare(ctx, {}, code, inputs);
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(.25, .25, .25, .25);
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		render_commands draw(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, 4, 4, 0, 1};
		raster.scissor = {0, 0, 4, 4};
		draw.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, {}, binding);
		draw.end();
		texture_region region;
		region.size = MTLSizeMake(4, 4, 1);
		texture_to_buffer(ctx, target, region, output, 0, 256);
		ctx.submit().wait();
		std::array<vector, 16> result;
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				std::copy_n(static_cast<const float*>(output.contents) + y * 64 + x * 4, 4, result[y * 4 + x].begin());
		return result;
	}
	void custom_draw(id<MTLTexture> image, decoded_texture_state state, unsigned opcode, vector coordinate, vector expected, bool killed = false)
	{
		if (opcode == 24)
			for (auto& channel : coordinate)
				channel *= 2;
		const auto pixels = custom_render(image, state, program(opcode, coordinate, {9, 0, 0, 0}, {0, 9, 0, 0}, 15));
		const float tolerance = image.pixelFormat == MTLPixelFormatDepth16Unorm && !state.shadow ? .5f / 65535 + 2e-6f : 2e-6f;
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				for (unsigned c = 0; c < 4; ++c)
				{
					const float actual = pixels[y * 4 + x][c];
					const float error = std::abs(actual - (killed ? .25f : expected[c]));
					if (!std::isfinite(actual) || error > tolerance)
						throw std::runtime_error("Custom depth border mismatch case=" + std::to_string(custom_cases) + " format=" + std::to_string(image.pixelFormat) + " shadow=" + std::to_string(state.shadow) + " compare=" + std::to_string(state.sampler.compare_function) + " op=" + std::to_string(opcode) + " lane=" + std::to_string(c) + " got=" + std::to_string(actual) + " wanted=" + std::to_string(expected[c]));
					max_custom_error = std::max(max_custom_error, error);
				}
		++custom_cases;
	}
	void custom_border_all()
	{
		const auto broadcast = [](float value)
		{ return vector{value, value, value, value}; };
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::cube})
			for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				const auto source = texture(format, dimension);
				decoded_texture_state state;
				state.enabled = state.depth = state.border_emulation = true;
				state.policy.dimension = dimension;
				state.sampler.wrap.fill(sampler_wrap::border);
				state.view_border_color = {.375f, 0, 1, .625f};
				state.depth_remap = {depth_sample_channel::depth, depth_sample_channel::zero, depth_sample_channel::one, depth_sample_channel::depth};
				const auto unorm = [format](float v)
				{ return format == MTLPixelFormatDepth16Unorm ? float(std::nearbyint(v * 65535)) / 65535 : v; };
				// Independent CPU tap reference: compare each depth/border tap
				// before interpolation, never compare an interpolated depth.
				const auto reference = [&](bool shadow, unsigned compare, bool linear, float ref, float u, float v, unsigned face)
				{
					vector result{};
					if (!state.sampler.depth_float)
						ref = std::min(ref, 1.f);
					if (shadow && format == MTLPixelFormatDepth16Unorm)
						ref = unorm(std::clamp(ref, 0.f, 1.f));
					const float px = u * 4 - .5f, py = dimension == vertex_texture_dimension::one ? 0 : v * 4 - .5f;
					const int ix = int(std::floor(linear ? px : u * 4)), iy = int(std::floor(linear ? py : v * 4));
					const float fx = linear ? px - ix : 0, fy = linear ? py - iy : 0;
					for (unsigned dy = 0; dy < (linear ? 2u : 1u); ++dy)
						for (unsigned dx = 0; dx < (linear ? 2u : 1u); ++dx)
						{
							const int x = ix + dx, y = dimension == vertex_texture_dimension::one ? 0 : iy + dy;
							const bool outside = dimension != vertex_texture_dimension::cube && (x < 0 || x >= 4 || y < 0 || y >= 4);
							const float weight = (linear ? (dx ? fx : 1 - fx) : 1) * (linear ? (dy ? fy : 1 - fy) : 1);
							const float depth = outside ? (shadow ? unorm(.375f) : .375f) : unorm(dimension == vertex_texture_dimension::cube ? float(face + 1) / 8 : std::array{.125f, .375f, .625f, .875f}[(x + y) % 4]);
							for (unsigned c = 0; c < 4; ++c)
								result[c] += weight * (shadow ? float(comparison(compare, depth, ref)) : outside										  ? state.view_border_color[c]
																									 : state.depth_remap[c] == depth_sample_channel::zero ? 0.f
																									 : state.depth_remap[c] == depth_sample_channel::one  ? 1.f
																																						  : depth);
						}
					return result;
				};
				const std::array<vector, 6> directions = {{{1, 0, 0, 1}, {-1, 0, 0, 1}, {0, 1, 0, 1}, {0, -1, 0, 1}, {0, 0, 1, 1}, {0, 0, -1, 1}}};
				for (bool shadow : {false, true})
					for (unsigned op : {23u, 24u, 25u, 47u, 49u})
					{
						if (shadow && dimension == vertex_texture_dimension::cube && op == 24)
							continue;
						state.shadow = state.sampler.depth_compare = shadow;
						state.sampler.depth_float = shadow && format != MTLPixelFormatDepth16Unorm;
						for (bool linear : {false, true})
						{
							state.sampler.min_filter = linear ? sampler_min_filter::linear : sampler_min_filter::nearest;
							state.sampler.mag_filter = linear ? sampler_mag_filter::linear : sampler_mag_filter::nearest;
							for (unsigned compare = 0; compare < (shadow ? 8u : 1u); ++compare)
								for (float ref : shadow ? std::vector<float>{-.25f, .375f - 1e-5f, .375f - 1e-6f, .375f, .375f + 1e-6f, .375f + 1e-5f, 1.25f} : std::vector<float>{0})
									for (unsigned at = 0; at < (dimension == vertex_texture_dimension::cube ? 6u : 5u); ++at)
									{
										state.sampler.compare_function = 0x200 + compare;
										const float u = std::array{-.25f, -.0625f, .0625f, .375f, 1.0625f}[at % 5];
										const float v = dimension == vertex_texture_dimension::one ? .5f : std::array{.375f, .0625f, -.0625f, .375f, 1.f}[at % 5];
										vector coord = dimension == vertex_texture_dimension::cube ? directions[at] : vector{u, v, ref, 1};
										if (dimension == vertex_texture_dimension::cube)
											coord[3] = shadow ? ref : 1;
										custom_draw(source, state, op, coord, reference(shadow, compare, linear, ref, dimension == vertex_texture_dimension::cube ? .375f : u, dimension == vertex_texture_dimension::cube ? .375f : v, at));
									}
						}
					}
				verify_stencil(source);
				const auto mip = texture(format, dimension, true);
				state.policy.requested_mips = state.policy.actual_mips = 3;
				for (bool shadow : {false, true})
					for (unsigned level = 0; level < 3; ++level)
						for (unsigned op : {23u, 25u, 47u, 49u})
						{
							state.shadow = state.sampler.depth_compare = shadow;
							state.sampler.depth_float = shadow && format != MTLPixelFormatDepth16Unorm;
							state.sampler.compare_function = 0x201;
							state.sampler.min_filter = sampler_min_filter::linear_linear;
							state.sampler.mag_filter = sampler_mag_filter::linear;
							state.sampler.min_lod = state.sampler.max_lod = float(level);
							state.sampler.bias = .5f;
							const float weight = dimension == vertex_texture_dimension::cube ? 0.f : .5f;
							const float d = unorm(.25f * float(level + 1));
							vector wanted;
							for (unsigned c = 0; c < 4; ++c)
								wanted[c] = shadow ? (1 - weight) * float(comparison(1, d, .375f)) + weight * float(comparison(1, .375f, .375f)) : (1 - weight) * (state.depth_remap[c] == depth_sample_channel::zero ? 0.f : state.depth_remap[c] == depth_sample_channel::one ? 1.f
																																																																				: d) +
																																					   weight * state.view_border_color[c];
							custom_draw(mip, state, op, dimension == vertex_texture_dimension::cube ? vector{1, 0, 0, .375f} : vector{0, .5f, .375f, 1}, wanted);
						}
				verify_stencil(mip);
				if (dimension != vertex_texture_dimension::cube)
				{
					state.shadow = state.sampler.depth_compare = true;
					state.sampler.depth_float = format != MTLPixelFormatDepth16Unorm;
					state.sampler.min_lod = 0;
					state.sampler.max_lod = 2;
					bool varying_pcf = false;
					for (unsigned compare : {1u, 3u, 4u, 6u})
						for (float border : {.375f, .625f})
							for (unsigned anisotropy : {1u, 4u, 16u})
								for (float bias : {-.5f, .5f, 1.25f})
									for (vector scale : {vector{.125f, .125f, 0, 0}, vector{.5f, .01f, 0, 0}, vector{2, .25f, 0, 0}})
									{
										state.sampler.compare_function = 0x200 + compare;
										state.sampler.anisotropy = anisotropy;
										state.sampler.bias = bias;
										state.view_border_color.fill(border);
										state.border_emulation = true;
										const auto code = program(23, {-.5f, .375f, .5f, 1}, scale, {}, 15, true);
										const auto actual = custom_render(mip, state, code);
										for (const auto& pixel : actual)
											varying_pcf |= pixel != actual[0];
										// The native 0/1 border with the same comparison
										// result is an independent PCF/LOD/anisotropy oracle.
										const bool pass = comparison(compare, border, .5f);
										const float canonical = comparison(compare, 0, .5f) == pass ? 0 : 1;
										state.sampler.border_color.fill(canonical);
										state.border_emulation = false;
										const auto expected = custom_render(mip, state, code);
										for (unsigned p = 0; p < 16; ++p)
											for (unsigned c = 0; c < 4; ++c)
											{
												const float error = std::abs(actual[p][c] - expected[p][c]);
												max_custom_pcf_error = std::max(max_custom_pcf_error, error);
												// Match the separate empirical PCF precision limit
												// of the coordinate regressions. This does not
												// establish bit-exact native PCF equivalence.
												if (!std::isfinite(actual[p][c]) || actual[p][c] < 0 || actual[p][c] > 1 || error > .5f / 255 + 2e-6f)
													throw std::runtime_error("Varying custom PCF mismatch:format=" + std::to_string(format) + " dim=" + std::to_string(unsigned(dimension)) + " compare=" + std::to_string(compare) + " border=" + std::to_string(border) + " anisotropy=" + std::to_string(anisotropy) + " bias=" + std::to_string(bias) + " scale=" + std::to_string(scale[0]) + "," + std::to_string(scale[1]) + " pixel=" + std::to_string(p) + " actual=" + std::to_string(actual[p][c]) + " expected=" + std::to_string(expected[p][c]));
											}
										++custom_cases;
									}
					state.sampler.anisotropy = 1;
					require(varying_pcf, "Custom PCF fixture did not generate varying pixels/implicit derivatives");
					state.border_emulation = true;
					state.view_border_color = {.375f, 0, 1, .625f};
				}
				if (dimension != vertex_texture_dimension::cube)
				{
					// Native canonical-border controls independently test D16
					// reference quantization,including negative and >1 references.
					state.policy.requested_mips = state.policy.actual_mips = 1;
					state.shadow = state.sampler.depth_compare = true;
					state.sampler.depth_float = format != MTLPixelFormatDepth16Unorm;
					state.sampler.min_lod = state.sampler.max_lod = state.sampler.bias = 0;
					state.sampler.min_filter = sampler_min_filter::nearest;
					state.sampler.mag_filter = sampler_mag_filter::nearest;
					for (float border : {0.f, 1.f})
						for (unsigned compare = 0; compare < 8; ++compare)
							for (float ref : {-.25f, 0.f, 1e-6f, .5f / 65535, 1.5f / 65535, 1.f - .5f / 65535, 1.f - 1e-6f, 1.f, 1.25f})
							{
								state.sampler.compare_function = 0x200 + compare;
								state.sampler.border_color.fill(border);
								state.view_border_color.fill(border);
								const auto code = program(23, {-1, .5f, ref, 1}, {}, {}, 15);
								state.border_emulation = false;
								const auto native = custom_render(source, state, code);
								state.border_emulation = true;
								const auto custom = custom_render(source, state, code);
								for (unsigned p = 0; p < 16; ++p)
									for (unsigned c = 0; c < 4; ++c)
										require(std::isfinite(custom[p][c]) && std::abs(custom[p][c] - native[p][c]) <= 2e-6f, "Custom shadow helper changed native canonical-border comparison precision");
								custom_cases += 2;
							}
					fragment_texture_registers regs;
					regs.format = (1u << 16) | ((format == MTLPixelFormatDepth16Unorm ? 0x92u : format == MTLPixelFormatDepth32Float ? 0x93u
																																	 : 0x91u)
												<< 8) |
								  (2u << 4) | 1;
					regs.control0 = 1u << 31;
					regs.control1 = 0xaae4;
					regs.filter = (1u << 24) | (1u << 16);
					regs.image_rect = (4u << 16) | 4;
					regs.border_color = 0xff606060;
					texture_cache_state metadata;
					metadata.depth = true;
					metadata.depth_float = format != MTLPixelFormatDepth16Unorm;
					for (unsigned compare = 1; compare < 7; ++compare)
						for (float delta : {-1e-6f, 0.f, 1e-6f})
						{
							regs.address = (compare << 28) | 0x040404;
							auto decoded = decode_fragment_texture_state(regs, metadata);
							if (dimension == vertex_texture_dimension::one)
								decoded.policy.dimension = dimension;
							require(decoded.shadow && decoded.border_emulation, "Real depth registers lost custom comparison border");
							const float border = decoded.view_border_color[0], ref = border + delta;
							const float effective = format == MTLPixelFormatDepth16Unorm ? unorm(ref) : ref;
							custom_draw(source, decoded, 24, {-1, .5f, ref, 1}, broadcast(float(comparison(compare, unorm(border), effective))));
						}
					state.border_emulation = true;
					state.view_border_color = {.375f, 0, 1, .625f};
				}
				state.shadow = state.sampler.depth_compare = state.sampler.depth_float = false;
				state.policy.requested_mips = state.policy.actual_mips = 1;
				state.sampler.min_lod = state.sampler.max_lod = state.sampler.bias = 0;
				state.sampler.min_filter = sampler_min_filter::nearest;
				state.sampler.mag_filter = sampler_mag_filter::nearest;
				if (dimension != vertex_texture_dimension::cube)
				{
					state.texels.gamma_mask = 1;
					custom_draw(source, state, 23, {-1, .5f, 0, 1}, {std::pow((.375f + .055f) / 1.055f, 2.4f), 0, 1, .625f});
					state.texels = {};
					state.texels.control = texel_alpha_kill;
					state.view_border_color[3] = 0;
					custom_draw(source, state, 23, {-1, .5f, 0, 1}, broadcast(0), true);
				}
			}
		const auto budget = std::make_shared<heap_budget>(ctx, 1);
		sampler_cache constrained(ctx, 1, budget);
		bool rejected = false;
		try
		{
			(void)constrained.border_shadow_view(texture(MTLPixelFormatDepth32Float, vertex_texture_dimension::two), .375f);
		}
		catch (const heap_budget_error&)
		{
			rejected = true;
		}
		require(rejected && budget->statistics().reserved_bytes == 0, "Constant depth helper bypassed actual heap quota");
		std::cout << custom_cases << " native custom depth-border cases:D16/D32/D32S8,1D/2D/cube,unit15,five fetch ops,eight reversed compares,nearest/PCF,depth remaps,mip clamps,bias,post-border gamma/alpha-kill and unchanged stencil;CPU tap-reference max error=" << max_custom_error << '\n';
		std::cout << "Native constant depth helpers:all mips/faces/encodings,reuse/color keys,actual heap refusal and varying PCF/LOD/anisotropy1/4/16 canonical-reference pixels passed\n";
		std::cout << "Custom varying PCF canonical-reference max error=" << max_custom_pcf_error << " (separate half-UNORM8-step limit plus2e-6;exact/dyadic shadow cases2e-6)\n";
	}
	void all()
	{
		custom_border_all();
		depth_all();
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::cube})
			for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				@autoreleasepool
				{
					const auto depth = texture(format, dimension);
					for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					{
						if (dimension == vertex_texture_dimension::cube && opcode == 24)
							continue;
						for (unsigned compare = 0; compare < 8; ++compare)
							for (bool linear : {false, true})
								for (float reference : {-.25f, .2f, .5f, .8f, 1.f, 1.25f})
									run(depth, dimension, opcode, compare, linear, reference, .25f, .25f);
					}
					verify_stencil(depth);
				}
			}
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two})
			for (auto format : {MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				const auto depth = texture(format, dimension, false, true);
				for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					for (unsigned compare = 0; compare < 8; ++compare)
						for (bool depth_float : {false, true})
						{
							const float ref = 1.25f, value = 1.125f;
							run(depth, dimension, opcode, compare, false, ref, .375f, .125f, depth_float, 0, 0, float(comparison(compare, value, depth_float ? ref : 1.f)));
						}
				verify_stencil(depth);
			}
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two})
			for (auto format : {MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				const auto depth = texture(format, dimension, false, -1);
				for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					for (unsigned compare = 0; compare < 8; ++compare)
						run(depth, dimension, opcode, compare, false, -.25f, .375f, .125f, true, 0, 0, float(comparison(compare, -.125f, -.25f)));
				verify_stencil(depth);
			}
		for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
		{
			const auto cube = texture(format, vertex_texture_dimension::cube);
			const std::array<vector, 6> directions{{{1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0}, {0, 0, 1, 0}, {0, 0, -1, 0}}};
			for (unsigned face = 0; face < 6; ++face)
				for (unsigned compare = 0; compare < 8; ++compare)
				{
					float value = float(face + 1) / 8;
					if (format == MTLPixelFormatDepth16Unorm)
						value = float(std::nearbyint(value * 65535)) / 65535;
					run(cube, vertex_texture_dimension::cube, 23, compare, false, .4f, .375f, .375f, false, 0, 0, float(comparison(compare, value, .4f)), directions[face]);
				}
			verify_stencil(cube);
		}
		// VK redirects all shadow texture opcodes to implicit comparison.
		// Vary operands that would select different mips if accidentally used.
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::cube})
			for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				const auto depth = texture(format, dimension, true);
				for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
				{
					if (dimension == vertex_texture_dimension::cube && opcode == 24)
						continue;
					for (unsigned compare = 0; compare < 8; ++compare)
						for (bool linear : {false, true})
							for (float bias : {-.5f, 0.f, .5f})
								for (float min_lod : {0.f, 1.f, 1.25f, 2.f})
									for (float operand : {0.f, 1.25f, 8.25f})
									{
										auto test = [&](unsigned mip)
										{
											float value = std::array{.25f, .5f, .75f}[mip];
											if (format == MTLPixelFormatDepth16Unorm)
												value = float(std::nearbyint(value * 65535)) / 65535;
											return float(comparison(compare, value, .6f));
										};
										const unsigned low = unsigned(std::floor(min_lod));
										const float weight = min_lod - low;
										const float expected = linear ? test(low) * (1 - weight) + test(std::min(low + 1, 2u)) * weight : test(unsigned(std::floor(min_lod + .5f)));
										run(depth, dimension, opcode, compare, linear, .6f, .375f, .375f, false, operand, bias, expected, {1, 0, 0, 0}, min_lod);
									}
				}
				verify_stencil(depth);
			}
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two})
			for (auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8})
			{
				const auto source = texture(format, dimension);
				std::array<texture_coordinates, 6> transforms{};
				transforms[1].scale = {2, 2, 1};
				transforms[1].bias = {-.25f, -.25f, 0};
				transforms[2].scale = {-1, 1, 1};
				transforms[2].bias = {1, 0, 0};
				transforms[3].flags = texture_coordinate_clamp;
				transforms[3].minimum = {.375f, .125f};
				transforms[3].maximum = {.625f, .375f};
				transforms[4] = normalize_guest_pixels(4, dimension == vertex_texture_dimension::one ? 1 : 4);
				transforms[5].scale = {2, 2, 1};
				transforms[5].bias = {-.125f, -.125f, 0};
				transforms[5].flags = texture_coordinate_clamp;
				transforms[5].minimum = {.125f, .375f};
				transforms[5].maximum = {.875f, .625f};
				for (const auto& transform : transforms)
					for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
						for (unsigned compare = 0; compare < 8; ++compare)
							for (bool linear : {false, true})
								for (float coordinate : {-.125f, .125f, .375f, .625f, 1.5f})
									run(source, dimension, opcode, compare, linear, .5f, coordinate, coordinate, false, 0, 0, -1, {1, 0, 0, 0}, 0, transform, linear ? .5f / 255 + 2e-6f : 2e-6f);
				verify_stencil(source);
			}
		{
			const auto cube = texture(MTLPixelFormatDepth32Float, vertex_texture_dimension::cube);
			texture_coordinates transform;
			transform.scale = {-1, 1, 1};
			transform.flags = texture_coordinate_clamp;
			transform.minimum = {.25f, .25f};
			transform.maximum = {.75f, .75f};
			for (unsigned compare = 0; compare < 8; ++compare)
				run(cube, vertex_texture_dimension::cube, 23, compare, false, .2f, .375f, .375f, false, 0, 0, float(comparison(compare, .25f, .2f)), {1, 0, 0, 0}, 0, transform);
		}
		const auto depth = texture(MTLPixelFormatDepth32Float, vertex_texture_dimension::two);
		texel_processing shadow_processing;
		shadow_processing.control = texel_alpha_kill | texel_renormalize | texel_biased_format;
		shadow_processing.sign_mask = shadow_processing.gamma_mask = shadow_processing.expand_mask = 15;
		for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
			run(depth, vertex_texture_dimension::two, opcode, 1, false, .1f, .375f, .375f, false, 0, 0, 0, {1, 0, 0, 0}, 0, {}, 2e-6f, shadow_processing);
		guest_sampler_state state;
		state.depth_compare = true;
		const auto comparison_sampler = samplers.acquire(state);
		bool refused = false;
		try
		{
			validate_sampled_texture(ctx, depth, comparison_sampler.sampler, vertex_texture_dimension::two, {}, true);
		}
		catch (const std::invalid_argument&)
		{
			refused = true;
		}
		require(refused, "Shadow sampling accepted missing comparison metadata");
		refused = false;
		try
		{
			validate_sampled_texture(ctx, target, comparison_sampler.sampler, vertex_texture_dimension::two, comparison_sampler.parameters, true);
		}
		catch (const std::invalid_argument&)
		{
			refused = true;
		}
		require(refused, "Shadow sampling accepted a color texture");
		refused = false;
		try
		{
			validate_sampled_texture(ctx, depth, comparison_sampler.sampler, vertex_texture_dimension::two, comparison_sampler.parameters, false);
		}
		catch (const std::invalid_argument&)
		{
			refused = true;
		}
		require(refused, "Ordinary sampling accepted depth-comparison bindings");
		state.depth_float = true;
		const auto float_sampler = samplers.acquire(state);
		const auto d16 = texture(MTLPixelFormatDepth16Unorm, vertex_texture_dimension::two);
		refused = false;
		try
		{
			validate_sampled_texture(ctx, d16, float_sampler.sampler, vertex_texture_dimension::two, float_sampler.parameters, true);
		}
		catch (const std::invalid_argument&)
		{
			refused = true;
		}
		require(refused, "D16 accepted float-depth shadow metadata");
		id<MTLSharedEvent> gate = [ctx.device() newSharedEvent];
		struct release_gate
		{
			id<MTLSharedEvent> gate;
			~release_gate()
			{
				gate.signaledValue = 1;
			}
		} release{gate};
		const auto lifetime_output = make_buffer(ctx, 256 * 4, buffer_storage::shared);
		const auto blocked = [&]
		{
			const auto source = texture(MTLPixelFormatDepth32Float, vertex_texture_dimension::two);
			const auto code = program(23, {.375f, .375f, .5f, 1});
			sampler_cache temporary(ctx, 1);
			guest_sampler_state state;
			state.depth_compare = true;
			state.compare_function = 0x201;
			fragment_inputs inputs;
			decoded_texture_state custom;
			custom.enabled = custom.depth = custom.shadow = custom.border_emulation = true;
			custom.sampler = state;
			custom.view_border_color.fill(.25f);
			custom.coordinates.scale = {0, 0, 1};
			custom.coordinates.bias = {-1, .125f, 0};
			inputs.set_texture_state(0, source, temporary, custom);
			fragment_program_options options;
			configure_fragment_texture_state(options, 0, custom);
			fragment_shader shader(compile, code, options);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			const auto pipeline = make_render_pipeline(compile, config, library_function(vertex_library, @"shadow_vertex"), shader);
			ctx.begin();
			ctx.wait_event(gate, 1);
			const auto fb = shader.prepare(ctx, {}, code, inputs);
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = target;
			pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			render_commands draw(ctx, pass);
			raster_state raster;
			raster.viewport = {0, 0, 4, 4, 0, 1};
			raster.scissor = {0, 0, 4, 4};
			draw.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, {}, fb);
			draw.end();
			state.compare_function = 0x204;
			inputs.set_sampler(0, temporary.acquire(state));
			inputs.coordinates[0] = {};
			require(temporary.statistics().evicted == 1, "Shadow lifetime fixture did not evict original comparison sampler");
			texture_region region;
			region.size = MTLSizeMake(4, 4, 1);
			texture_to_buffer(ctx, target, region, lifetime_output, 0, 256);
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
		require(timeout && !blocked.ready(), "Shadow lifetime job was not actually counter-blocked");
		gate.signaledValue = 1;
		blocked.wait();
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 16; ++x)
				require(static_cast<const float*>(lifetime_output.contents)[y * 64 + x] == 1, "Shadow sampler/input/cache/coordinate lifetime changed original comparison after timeout");
		ctx.wait_idle();
		std::cout << "Custom depth border pending GPU retains constant helper after source/sampler/cache/input/shader destruction and sampler eviction passed\n";
		std::cout << "Shadow sampler-minLOD PCF max error: " << max_pcf_error << '\n';
		std::cout << "Coordinate shadow-PCF max error: " << max_coordinate_pcf_error << " (separate half-unorm8-step limit plus2e-6)\n";
		std::cout << cases << " native translated RSX shadow cases:1D/2D/cube,D16/D32/D32S8,five texture operations,eight comparisons,nearest/PCF,projected and integer/float extended reference policies passed\n";
		std::cout << "Shadow sampler mip clamps/ignored shader LOD-gradient-bias operands, six cube faces, unchanged source stencil planes, invalid type/metadata preflight and actual destroyed-source/cache/evicted-sampler timeout lifetime passed\n";
	}
};
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			fixture test;
			test.all();
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
