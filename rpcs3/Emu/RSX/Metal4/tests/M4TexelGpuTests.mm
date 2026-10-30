#include "M4FragmentShader.h"
#include "M4RenderPipeline.h"
#include "M4Texture.h"
#include "M4FragmentTextureBatch.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
using color = std::array<float, 4>;
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
std::vector<std::byte> program(bool bx2 = false, bool predicate = true, unsigned condition = 7, unsigned texture_unit = 0)
{
	std::vector<std::byte> code;
	for (unsigned word : {(23u << 24) | (texture_unit << 17) | (15u << 9) | (bx2 ? 1u << 21 : 0) | 1u, (0xe4u << 9) | ((predicate ? condition : 0u) << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
		append(code, word);
	for (float v : {.5f, .5f, 0.f, 1.f})
		append(code, std::bit_cast<unsigned>(v));
	return code;
}
color reference(color value, const texel_processing& p, bool bx2, bool& killed)
{
	if ((p.control & texel_alpha_kill) && value[3] < 1e-6f)
	{
		killed = true;
		return value;
	}
	if (p.control & texel_renormalize)
		for (auto& c : value)
			c = std::floor(c * 255.f) / 255.f;
	const bool active = bx2 && (p.control & texel_biased_format);
	const unsigned gamma = (active ? 0 : p.gamma_mask) & ~p.sign_mask;
	const unsigned expand = (p.expand_mask | (active ? p.bx2_mask : 0)) & ~(p.sign_mask | gamma);
	const unsigned bits = (p.control & texel_channels16) ? 16 : 8;
	const float range = float((1u << bits) - 1), center = float(1u << (bits - 1)), divisor = center - 1;
	for (unsigned c = 0; c < 4; ++c)
	{
		if (p.sign_mask & (1u << c))
		{
			const unsigned q = unsigned(std::floor(std::fma(value[c], range, .5f))) & ((1u << bits) - 1);
			const int signed_q = q & (1u << (bits - 1)) ? int(q) - int(1u << bits) : int(q);
			value[c] = float(signed_q) / divisor;
		}
		else if (gamma & (1u << c))
			value[c] = value[c] > .04045f ? std::pow((value[c] + .055f) / 1.055f, 2.4f) : value[c] / 12.92f;
		else if (expand & (1u << c))
			value[c] = active ? value[c] * 2 - 1 : (std::floor(std::fma(value[c], range, .5f)) - center) / divisor;
	}
	return value;
}
class fixture
{
	context ctx{1};
	compiler compile{ctx};
	sampler_cache cache{ctx};
	id<MTLLibrary> vertex;
	id<MTLTexture> image, target, depth;
	id<MTLBuffer> output, depth_output;
	std::array<std::unique_ptr<fragment_shader>, 96> shaders;
	std::array<std::optional<render_pipeline_bundle>, 96> pipelines;
	converted_geometry geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);

  public:
	unsigned cases = 0;
	float max_error = 0;
	fixture()
	{
		vertex = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
vertex float4 texel_vertex(uint id [[vertex_id]]){const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],.75,1);}
)metal",
								 @"RSX texel regression vertex");
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:1 height:1 mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		image = make_texture(ctx, td);
		td.width = td.height = 4;
		td.usage = MTLTextureUsageRenderTarget;
		target = make_texture(ctx, td);
		td.pixelFormat = MTLPixelFormatDepth32Float;
		depth = make_texture(ctx, td);
		output = make_buffer(ctx, 1024, buffer_storage::shared);
		depth_output = make_buffer(ctx, 1024, buffer_storage::shared);
	}
	void draw(color source, texel_processing controls, bool bx2 = false, bool predicate = true, MTLPixelFormat source_format = MTLPixelFormatRGBA32Float, unsigned condition = 7, const decoded_texture_state* register_state = nullptr, const color* sample_reference = nullptr, const fragment_inputs* frozen_inputs = nullptr, unsigned texture_unit = 0)
	{
		require(texture_unit < 16,"Texel fixture texture unit exceeds hardware slots");
		require(condition == 7 || condition == 4, "Texel fixture only models ALWAYS or GREATER against initial zero condition registers");
		if (image.pixelFormat != source_format)
		{
			auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source_format width:1 height:1 mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			td.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			image = make_texture(ctx, td);
		}
		std::vector<std::byte> pixels;
		if (source_format == MTLPixelFormatRGBA32Float)
		{
			const auto bytes = std::as_bytes(std::span(&source, 1));
			pixels.assign(bytes.begin(), bytes.end());
		}
		else
		{
			const unsigned bytes = source_format == MTLPixelFormatRGBA16Unorm ? 2 : 1;
			const float range = bytes == 2 ? 65535.f : 255.f;
			pixels.resize(bytes * 4);
			for (unsigned c = 0; c < 4; ++c)
			{
				const unsigned q = unsigned(std::nearbyint(source[c] * range));
				source[c] = float(q) / range;
				if (bytes == 2)
				{
					const std::uint16_t word = q;
					std::memcpy(pixels.data() + c * 2, &word, 2);
				}
				else
					pixels[c] = std::byte(q);
			}
		}
		const auto code = program(bx2, predicate, condition,texture_unit);
		const unsigned index = (bx2 ? 1 : 0) + (predicate ? (condition == 7 ? 0 : 4) : 2) + 6 * texture_unit;
		if (!shaders[index])
		{
			shaders[index] = std::make_unique<fragment_shader>(compile, code);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			config.state.depth_test = true;
			config.state.depth_write = true;
			config.state.depth_compare = 0x207;
			pipelines[index] = make_render_pipeline(compile, config, library_function(vertex, @"texel_vertex"), *shaders[index]);
		}
		fragment_inputs inputs;
		inputs.textures[0] = image;
		inputs.set_sampler(0, cache.acquire({}));
		inputs.texels[0] = controls;
		if (sample_reference)
			source = *sample_reference;
		if (register_state)
		{
			inputs.set_texture_state(0, image, cache, *register_state);
			const auto original = source;
			for (unsigned c = 0; c < 4; ++c)
			{
				const auto channel = register_state->view_channels[c];
				source[c] = channel < 4 ? original[channel] : channel == 4 ? 0.f : 1.f;
			}
		}
		if (frozen_inputs)
			inputs = *frozen_inputs;
		ctx.begin();
		texture_region pixel;
		pixel.size = MTLSizeMake(1, 1, 1);
		buffer_to_texture(ctx, make_upload(ctx, pixels), 0, image, pixel, pixels.size());
		const auto fb = shaders[index]->prepare(ctx, {}, code, inputs);
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(.25, .25, .25, .25);
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.depthAttachment.texture = depth;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.clearDepth = .25;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		render_commands commands(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, 4, 4, 0, 1};
		raster.scissor = {0, 0, 4, 4};
		const auto& pipeline = *pipelines[index];
		commands.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, {}, fb);
		commands.end();
		texture_region region;
		region.size = MTLSizeMake(4, 4, 1);
		texture_to_buffer(ctx, target, region, output, 0, 256);
		texture_to_buffer(ctx, depth, region, depth_output, 0, 256, 0, texture_plane::depth);
		ctx.submit().wait();
		bool killed = false;
		const auto evaluated = predicate ? reference(source, controls, bx2, killed) : color{};
		const auto expected = condition == 7 ? evaluated : color{};
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
			{
				require(static_cast<const float*>(depth_output.contents)[y * 64 + x] == (killed ? .25f : .75f), "Texture alphakill/predicate changed native depth write ordering");
				for (unsigned c = 0; c < 4; ++c)
				{
					const float value = static_cast<const float*>(output.contents)[y * 64 + x * 4 + c], wanted = killed ? .25f : expected[c];
					const float error = std::abs(value - wanted);
					if (!std::isfinite(value) || error > 2e-6f)
						throw std::runtime_error("Native RSX texel processing mismatch case=" + std::to_string(cases) + " lane=" + std::to_string(c) + " actual=" + std::to_string(value) + " expected=" + std::to_string(wanted));
					max_error = std::max(max_error, error);
				}
			}
		++cases;
	}
	void exhaustive(bool sixteen, bool expansion)
	{
		const unsigned count = sixteen ? 65536 : 256;
		std::vector<color> values(count);
		for (unsigned i = 0; i < count; ++i)
			values[i] = {float(i) / float(count - 1), float(i) / float(count - 1), float(i) / float(count - 1), float(i) / float(count - 1)};
		std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
		source += texel_processing_msl;
		source += R"metal(
kernel void process(const device float4* input [[buffer(0)]],constant rsx_texel_processing& control [[buffer(1)]],device float4* output [[buffer(2)]],device uint* status [[buffer(3)]],uint i [[thread_position_in_grid]]){
    bool killed=false;uint fault=0;output[i]=rsx_process_texel(input[i],control,false,killed,fault);status[i]=fault|(killed?128u:0u);
}
)metal";
		const auto library = compile.library(source, @"Exhaustive native texel signed arithmetic");
		auto pd = [MTL4ComputePipelineDescriptor new];
		pd.computeFunctionDescriptor = library_function(library, @"process");
		const auto pipeline = compile.compute(pd);
		texel_processing control;
		if (sixteen)
			control.control |= texel_channels16;
		if (expansion)
			control.expand_mask = 15;
		else
			control.sign_mask = 15;
		const auto input = make_upload(ctx, std::as_bytes(std::span(values))), parameters = make_upload(ctx, std::as_bytes(std::span(&control, 1)));
		const auto output = make_buffer(ctx, count * 16, buffer_storage::shared), status = make_buffer(ctx, count * 4, buffer_storage::shared);
		ctx.begin();
		bindings table;
		table.set_buffer(0, input);
		table.set_buffer(1, parameters);
		table.set_buffer(2, output);
		table.set_buffer(3, status);
		ctx.keep_alive(pipeline);
		const auto arguments = table.snapshot(ctx);
		const auto encoder = ctx.compute();
		[encoder setComputePipelineState:pipeline];
		[encoder setArgumentTable:arguments];
		[encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
		ctx.submit().wait();
		for (unsigned i = 0; i < count; ++i)
		{
			bool killed = false;
			const auto expected = reference(values[i], control, false, killed);
			require(static_cast<const unsigned*>(status.contents)[i] == 0, "Valid unorm texel produced fault/discard");
			for (unsigned c = 0; c < 4; ++c)
				require(std::bit_cast<unsigned>(static_cast<const float*>(output.contents)[i * 4 + c]) == std::bit_cast<unsigned>(expected[c]), "Exhaustive native signed texel arithmetic differs in float bits");
		}
		std::cout << count << " native " << (sixteen ? "16" : "8") << "-bit " << (expansion ? "biased expansion" : "sign extension") << " encodings match exact float bits\n";
	}
	void all()
	{
		for (bool sixteen : {false, true})
			for (bool expansion : {false, true})
				exhaustive(sixteen, expansion);
		for (unsigned mask = 0; mask < 16; ++mask)
			for (unsigned variant = 0; variant < 8; ++variant)
				for (bool bx2 : {false, true})
				{
					texel_processing controls;
					controls.sign_mask = (variant & 1) ? mask : 0;
					controls.gamma_mask = (variant & 2) ? mask : 0;
					controls.expand_mask = (variant & 4) ? mask : 0;
					controls.bx2_mask = mask;
					controls.control = texel_biased_format;
					draw({127.f / 255, 128.f / 255, 1, 0}, controls, bx2);
					controls.control |= texel_channels16 | texel_renormalize;
					draw({.04045f, .5f, .875f, 1}, controls, bx2);
				}
		for (float alpha : {0.f, 5e-7f, 1e-6f, 2e-6f, 1.f})
		{
			texel_processing controls;
			controls.control = texel_alpha_kill;
			controls.expand_mask = 8;
			draw({.5, .25, .75, alpha}, controls);
			draw({.5, .25, .75, alpha}, controls, false, false);
		}
		for (bool bx2 : {false, true})
		{
			texel_processing controls;
			controls.gamma_mask = 15;
			controls.expand_mask = 15;
			controls.bx2_mask = 15;
			draw({.125, .375, .625, .875}, controls, bx2);
		}
		for (float alpha : {0.f, 1.f})
		{
			texel_processing controls;
			controls.control = texel_alpha_kill;
			draw({.5, .25, .75, alpha}, controls, false, true, MTLPixelFormatRGBA32Float, 4);
		}
		for (auto format : {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Unorm})
			for (bool expansion : {false, true})
				for (unsigned mask = 0; mask < 16; ++mask)
				{
					texel_processing controls;
					if (format == MTLPixelFormatRGBA16Unorm)
						controls.control = texel_channels16;
					if (expansion)
						controls.expand_mask = mask;
					else
						controls.sign_mask = mask;
					draw({0, .5f, 1, .75f}, controls, false, true, format);
				}
		ctx.wait_idle();
		std::cout << cases << " translated RSX texel cases:RGBA masks/precedence,8/16-bit,renormalize,gamma,BX2 format gating,alpha-kill-before-conversion,native depth preservation,static NEVER skips and dynamic false predicates evaluate;max error=" << max_error << '\n';
	}
	void register_cases()
	{
		fragment_texture_registers regs;
		regs.format = (1u << 16) | (0x85u << 8) | (2u << 4) | 1;
		regs.address = (7u << 28) | 0x030303;
		regs.control0 = 1u << 31;
		regs.control1 = 0xaae4;
		regs.filter = (1u << 24) | (3u << 16);
		regs.image_rect = (1u << 16) | 1;
		const unsigned first = cases;
		for (unsigned remap : {0xaae4u,0xa954u,0xaa1bu,0xaaaaU,0x0000u,0x5555u})
			for (unsigned sign : {0u,2u,15u})
				for (unsigned gamma : {0u,15u})
				{
					regs.control1 = remap;
					regs.filter = (1u << 24) | (3u << 16) | (sign << 28);
					regs.address = (7u << 28) | (gamma << 20) | (1u << 12) | 0x030303;
					const auto state = decode_fragment_texture_state(regs, {});
					fragment_program_options options;
					configure_fragment_texture_state(options, 0, state);
					require(options.textures[0] == vertex_texture_dimension::two && !(options.depth_textures | options.shadow_textures | options.redirected_textures | options.multisampled_textures), "Ordinary register state configured wrong shader texture binding");
					draw({.5,.25,.75,1}, state.texels, false, true, MTLPixelFormatRGBA8Unorm, 7, &state);
				}
		regs.control1 = 0xaaaa; // All read G:raw alpha1 becomes sampled alpha0.
		regs.filter = (1u << 24) | (3u << 16);
		regs.address = (7u << 28) | 0x030303;
		regs.control0 |= 4;
		const auto killed = decode_fragment_texture_state(regs, {});
		draw({1,0,1,1}, killed.texels, false, true, MTLPixelFormatRGBA8Unorm, 7, &killed);
		regs.control0 &= ~4u;
		regs.control1 = 0xa900; // ARGB:one,A,A,A.
		regs.address = (7u << 28) | 0x040404;
		texture_cache_state outside;
		outside.coordinates.scale = {0,0,1};
		outside.coordinates.bias = {-4,-4,0};
		const auto border = decode_fragment_texture_state(regs,outside);
		const color transparent{};
		draw({1,1,1,1},border.texels,false,true,MTLPixelFormatRGBA8Unorm,7,&border,&transparent);
		std::cout << cases - first << " native register->view->sampler->buffer4/5 texel draws passed\n";
	}
	void snapshot_cases()
	{
		struct rsx_register_fixture { std::array<std::uint32_t,0x10000/4> registers{}; } live;
		const auto write = [&](unsigned unit,unsigned remap)
		{
			const unsigned base = (0x1a00+32*unit)/4;
			live.registers[base] = 0x123400 + unit * 256;
			live.registers[base+1] = (1u<<16) | (0x85u<<8) | (2u<<4) | 1;
			live.registers[base+2] = (7u<<28) | 0x030303;
			live.registers[base+3] = 1u<<31;
			live.registers[base+4] = remap;
			live.registers[base+5] = (1u<<24) | (3u<<16);
			live.registers[base+6] = (1u<<16) | 1;
			live.registers[(0x1840+4*unit)/4] = 256;
		};
		write(0,0xaae4);
		write(3,0xa954); // One,R,R,R.
		const auto snapshot = capture_fragment_texture_snapshot(live,9);
		fragment_program_options options;
		fragment_inputs inputs;
		inputs.coordinates[15].bias = {.125f,.25f,.5f}; // Unreferenced state must survive.
		sampler_cache temporary(ctx,1);
		unsigned callbacks = 0;
		apply_fragment_texture_snapshot(snapshot,[&](unsigned unit,const fragment_texture_registers& words)
			{
				require(words.offset == 0x123400+unit*256 && words.control3 == 256,"Frozen cache resolver lost address/pitch fields");
				if (unit == 0) live.registers[(0x1a10+32*3)/4] = 0x5555; // Mutate future unit during callback.
				++callbacks;
				return resolved_fragment_texture{image,{}};
			},ctx,temporary,options,inputs);
		require(callbacks == 2 && inputs.coordinates[15].bias == std::array<float,3>{.125f,.25f,.5f},"Batch read unreferenced state or published a partial input set");
		const unsigned first = cases;
		const color uploaded{.25f,.5f,.75f,1}, expected{64.f/255,64.f/255,64.f/255,1};
		draw(uploaded,inputs.texels[3],false,true,MTLPixelFormatRGBA8Unorm,7,nullptr,&expected,&inputs,3);
		const auto old_inputs = inputs;
		const auto old_options = options;
		const auto unchanged = [&]
		{
			return inputs.textures == old_inputs.textures && inputs.samplers == old_inputs.samplers &&
				std::memcmp(inputs.sampling.data(),old_inputs.sampling.data(),sizeof(inputs.sampling)) == 0 &&
				std::memcmp(inputs.coordinates.data(),old_inputs.coordinates.data(),sizeof(inputs.coordinates)) == 0 &&
				std::memcmp(inputs.texels.data(),old_inputs.texels.data(),sizeof(inputs.texels)) == 0 &&
				options.textures == old_options.textures && options.depth_remap == old_options.depth_remap &&
				options.depth_textures == old_options.depth_textures && options.shadow_textures == old_options.shadow_textures &&
				options.redirected_textures == old_options.redirected_textures && options.multisampled_textures == old_options.multisampled_textures;
		};
		for (unsigned failure = 0; failure < 6; ++failure)
		{
			write(0,0xaa1b);
			write(3,0x5555);
			live.registers[(0x1a08+32*0)/4] = (7u<<28) | 0x010101; // Candidate sampler evicts prior one.
			if (failure == 1) live.registers[(0x1a04+32*3)/4] = (1u<<16) | (0x80u<<8) | (2u<<4) | 1;
			if (failure == 3) live.registers[(0x1a04+32*3)/4] |= 4; // Cube declaration with an actual 2D resource.
			if (failure == 5) live.registers[(0x1a0c+32*3)/4] = 0;
			const auto failed_snapshot = capture_fragment_texture_snapshot(live,9);
			bool rejected = false;
			try
			{
				apply_fragment_texture_snapshot(failed_snapshot,[&](unsigned unit,const fragment_texture_registers&)
					{
						if (unit == 3 && failure == 0) throw std::runtime_error("Cache resolver failure fixture");
						texture_cache_state metadata;
						if (unit == 3 && failure == 4) metadata.samples = 2;
						return resolved_fragment_texture{unit == 3 && failure == 2 ? nil : image,metadata};
					},ctx,temporary,options,inputs);
			}
			catch (const std::exception&) { rejected = true; }
			require(rejected && unchanged(),"Later-unit cache/decode/resource/sample/disabled failure published partial options or inputs");
			draw(uploaded,inputs.texels[3],false,true,MTLPixelFormatRGBA8Unorm,7,nullptr,&expected,&inputs,3);
		}
		require(temporary.statistics().evicted > 0,"Atomic snapshot test did not evict candidate sampler");
		unsigned empty_calls = 0;
		apply_fragment_texture_snapshot(fragment_texture_snapshot::capture({},0),[&](unsigned,const fragment_texture_registers&) { ++empty_calls; return resolved_fragment_texture{}; },ctx,temporary,options,inputs);
		require(empty_calls == 0 && unchanged(),"Empty snapshot resolved or changed input state");
		std::cout << cases-first << " native frozen-snapshot draws:future-unit mutation,6 failed batch publications with valid old GPU output,sampler eviction,address/pitch,unreferenced/empty masks passed\n";
		const auto gate = [ctx.device() newSharedEvent];
		require(bool(gate),"Snapshot lifetime shared event allocation failed");
		const auto pending_output = make_buffer(ctx,1024,buffer_storage::shared);
		auto pending = [&]
		{
			write(0,0xaae4);
			write(3,0xa954);
			for (unsigned unit : {0u,3u}) live.registers[(0x1a18+32*unit)/4] = (2u<<16) | 1;
			const auto frozen = capture_fragment_texture_snapshot(live,9);
			auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:2 height:1 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto source = make_texture(ctx,descriptor);
			// Finish upload before the blocked draw. Its job must retain the
			// sampled view/parent rather than relying on an in-flight copy owner.
			ctx.begin();
			const std::array<std::uint8_t,8> pixels = {64,128,191,255,192,32,16,255};
			texture_region pixel;
			pixel.size = MTLSizeMake(2,1,1);
			buffer_to_texture(ctx,make_upload(ctx,std::as_bytes(std::span(pixels))),0,source,pixel,8);
			ctx.submit().wait();
			sampler_cache local_samplers(ctx,1);
			fragment_inputs local_inputs;
			fragment_program_options local_options;
			apply_fragment_texture_snapshot(frozen,[&](unsigned,const fragment_texture_registers&)
				{
					texture_cache_state metadata;
					metadata.coordinates.scale = {0,0,1};
					metadata.coordinates.bias = {.25f,.5f,0};
					return resolved_fragment_texture{source,metadata};
				},ctx,local_samplers,local_options,local_inputs);
			const auto code = program(false,true,7,3);
			fragment_shader local_shader(compile,code,local_options);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			const auto local_pipeline = make_render_pipeline(compile,config,library_function(vertex,@"texel_vertex"),local_shader);
			ctx.begin();
			ctx.wait_event(gate,1);
			const auto binding = local_shader.prepare(ctx,{},code,local_inputs);
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = target;
			pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			render_commands commands(ctx,pass);
			raster_state raster;
			raster.viewport = {0,0,4,4,0,1};
			raster.scissor = {0,0,4,4};
			commands.draw_geometry(geometry,local_pipeline.pipeline,local_pipeline.depth_stencil,raster,{},binding);
			commands.end();
			local_inputs.coordinates[3] = {}; // Replacement would select red192,not64.
			local_inputs.texels[3].sign_mask = 15; // Replacement would convert the captured normalized value.
			guest_sampler_state replacement;
			local_inputs.set_sampler(3,local_samplers.acquire(replacement));
			require(local_samplers.statistics().evicted > 0,"Snapshot lifetime fixture did not evict encoded sampler");
			live.registers.fill(0);
			texture_region region;
			region.size = MTLSizeMake(4,4,1);
			texture_to_buffer(ctx,target,region,pending_output,0,256);
			return ctx.submit();
		}();
		bool timed_out = false;
		try { pending.wait(std::chrono::milliseconds(1)); } catch (const std::runtime_error&) { timed_out = true; }
		require(timed_out && !pending.ready(),"Snapshot lifetime GPU job was not actually counter-blocked");
		gate.signaledValue = 1;
		pending.wait();
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				for (unsigned c = 0; c < 4; ++c)
					require(std::abs(static_cast<const float*>(pending_output.contents)[y*64+x*4+c] - (c == 3 ? 1.f : 64.f/255)) < 2e-6f,"Destroyed snapshot/source/view/cache/shader or mutated metadata changed pending GPU result");
		std::cout << "Frozen snapshot native view/source/cache/shader destruction,eviction and buffer4/5 mutation retained original output after real GPU timeout\n";
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
			test.register_cases();
			test.snapshot_cases();
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
