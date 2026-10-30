#include "M4SurfaceDepth.h"
#include "M4SurfaceDepthKernels.h"
#include "M4VertexShader.h"
#include "M4FragmentShader.h"
#include "M4RenderPipeline.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
	if (!condition)
		throw std::runtime_error(message);
}

void descriptors()
{
	using namespace metal4;
	constexpr std::array<MTLPixelFormat, 4> expected{MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float_Stencil8,
													 MTLPixelFormatDepth32Float, MTLPixelFormatDepth32Float_Stencil8};
	for (unsigned i = 1; i <= 4; ++i)
	{
		const auto d = describe_surface_depth(static_cast<surface_depth>(i));
		require(d.format == expected[i - 1] && d.guest_bytes == (i % 2 ? 2u : 4u) &&
					d.depth_bytes == (i == 1 ? 2u : 4u) && d.stencil == (i % 2 == 0),
				"Depth surface descriptor differs");
	}
	for (unsigned i = 0; i < 256; ++i)
	{
		bool valid = true;
		try
		{
			(void)describe_surface_depth(static_cast<surface_depth>(i));
		}
		catch (const std::invalid_argument&)
		{
			valid = false;
		}
		require(valid == (i >= 1 && i <= 4), "Invalid depth format was accepted");
	}
}

std::uint32_t load(std::span<const std::byte> bytes, unsigned count, bool be)
{
	std::uint32_t value = 0;
	for (unsigned b = 0; b < count; ++b)
		value |= std::uint32_t(bytes[b]) << ((be ? count - 1 - b : b) * 8);
	return value;
}
void store(std::span<std::byte> bytes, unsigned count, std::uint32_t value, bool be)
{
	for (unsigned b = 0; b < count; ++b)
		bytes[b] = static_cast<std::byte>(value >> ((be ? count - 1 - b : b) * 8));
}

std::uint32_t float24_pattern(std::size_t index)
{
	// Include positive zero, subnormal, normal, infinity, quiet/signaling NaNs
	// and all upper24 bit fields across a large stride; pure copies preserve bits.
	constexpr std::array<std::uint32_t, 8> special{0, 1, 0x7fff, 0x7f0000, 0x7f8000, 0x7f8123, 0x7fffff, 0x3f8000};
	return index < special.size() ? special[index] : static_cast<std::uint32_t>(index * 7919) & 0xffffff;
}

void codec(metal4::context& ctx, metal4::surface_depth_transfer& transfer, metal4::surface_depth format)
{
	using namespace metal4;
	const auto metadata = describe_surface_depth(format);
	const std::size_t width = format == surface_depth::z24s8_uint ? 4096 : 513;
	const std::size_t height = format == surface_depth::z24s8_uint ? 4096 : 129;
	const auto gp = (width + 3) * metadata.guest_bytes, dp = (width + 5) * metadata.depth_bytes, sp = width + 7;
	std::vector<std::byte> guest_bytes(3 + gp * height + 13, std::byte{0xa7});
	for (std::size_t y = 0; y < height; ++y)
		for (std::size_t x = 0; x < width; ++x)
		{
			const auto pixel = y * width + x;
			const auto value = format == surface_depth::z24s8_float ? (float24_pattern(pixel) << 8) | (static_cast<std::uint32_t>(pixel) % 256) : metadata.stencil ? (static_cast<std::uint32_t>(pixel) << 8) | (static_cast<std::uint32_t>(pixel) % 256)
																																								   : static_cast<std::uint32_t>(pixel) % 65536;
			store(std::span(guest_bytes).subspan(3 + y * gp + x * metadata.guest_bytes), metadata.guest_bytes, value, true);
		}
	const auto guest = make_upload(ctx, guest_bytes);
	const auto depth = make_buffer(ctx, 5 + dp * height + 17, buffer_storage::shared);
	const auto stencil = metadata.stencil ? make_buffer(ctx, 7 + sp * height + 19, buffer_storage::shared) : nil;
	const auto output = make_buffer(ctx, guest_bytes.size(), buffer_storage::shared);
	const auto texture_output = make_buffer(ctx, guest_bytes.size(), buffer_storage::shared);
	MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metadata.format width:width height:height mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
	const auto texture = make_texture(ctx, descriptor);
	texture_region region;
	region.size = MTLSizeMake(width, height, 1);
	ctx.begin();
	fill_buffer(ctx, depth, 0, depth.length, 0xd3);
	if (stencil)
		fill_buffer(ctx, stencil, 0, stencil.length, 0xe5);
	fill_buffer(ctx, output, 0, output.length, 0xa7);
	fill_buffer(ctx, texture_output, 0, texture_output.length, 0xa7);
	const depth_transfer_plane d{depth, {5, dp}}, s{stencil, {7, sp}};
	transfer.scatter(ctx, format, width, height, guest, {3, gp}, d, metadata.stencil ? s : depth_transfer_plane{});
	transfer.gather(ctx, format, width, height, d, metadata.stencil ? s : depth_transfer_plane{}, output, {3, gp});
	transfer.upload(ctx, format, guest, {3, gp}, texture, region);
	transfer.download(ctx, format, texture, region, texture_output, {3, gp});
	ctx.submit().wait();
	const auto actual_d = map_buffer(depth, 0, depth.length), actual_g = map_buffer(output, 0, output.length),
			   actual_texture = map_buffer(texture_output, 0, texture_output.length);
	const auto actual_s = stencil ? map_buffer(stencil, 0, stencil.length) : std::span<std::byte>{};
	std::vector<std::byte> expected_d(depth.length, std::byte{0xd3}), expected_s(stencil.length, std::byte{0xe5});
	auto expected_g = guest_bytes;
	std::size_t quantized_changes = 0;
	for (std::size_t y = 0; y < height; ++y)
		for (std::size_t x = 0; x < width; ++x)
		{
			const auto ga = 3 + y * gp + x * metadata.guest_bytes, da = 5 + y * dp + x * metadata.depth_bytes;
			const auto raw = load(std::span(guest_bytes).subspan(ga), metadata.guest_bytes, true);
			std::uint32_t dbits = raw, encoded = raw;
			if (format == surface_depth::z16_float)
			{
				const auto exponent = static_cast<int>(raw / 4096) - 7;
				dbits = std::bit_cast<std::uint32_t>(std::ldexp(1.0f + float(raw % 4096) / 4096, exponent));
			}
			else if (format == surface_depth::z24s8_float)
				dbits = (raw >> 8) * 128;
			else if (format == surface_depth::z24s8_uint)
			{
				// Independent numerical model uses a double rational rounded to
				// float32. Gather retains the established VK truncation contract.
				const float normalized = static_cast<float>(double(raw >> 8) / 16777215.0);
				dbits = std::bit_cast<std::uint32_t>(normalized);
				const auto contracted = static_cast<std::uint32_t>(normalized * 16777215.0f);
				encoded = (contracted << 8) | (raw % 256);
				quantized_changes += encoded != raw;
			}
			store(std::span(expected_d).subspan(da), metadata.depth_bytes, dbits, false);
			store(std::span(expected_g).subspan(ga), metadata.guest_bytes, encoded, true);
			if (stencil)
				expected_s[7 + y * sp + x] = static_cast<std::byte>(raw);
		}
	if (!std::equal(actual_d.begin(), actual_d.end(), expected_d.begin()))
	{
		const auto mismatch = std::mismatch(actual_d.begin(), actual_d.end(), expected_d.begin());
		std::cerr << "Depth format " << unsigned(format) << " first depth-byte mismatch at " << mismatch.first - actual_d.begin() << '\n';
		throw std::runtime_error("Depth scatter bits or padding differs");
	}
	require(std::equal(actual_g.begin(), actual_g.end(), expected_g.begin()), "Depth gather/guest-padding differs");
	require(std::equal(actual_texture.begin(), actual_texture.end(), expected_g.begin()), "Depth native texture round-trip differs");
	if (stencil)
		require(std::equal(actual_s.begin(), actual_s.end(), expected_s.begin()), "Stencil scatter/padding differs");
	std::cout << "Depth " << unsigned(format) << ": " << width * height << " pixel scatter/gather/native texture checks passed; "
			  << quantized_changes << " normalized truncation changes\n";
}

void rejection(metal4::context& ctx, metal4::surface_depth_transfer& transfer)
{
	using namespace metal4;
	const auto buffer = make_buffer(ctx, 256, buffer_storage::shared);
	ctx.begin();
	const auto rejects = [](auto function)
	{
		bool rejected = false;
		try
		{
			function();
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		catch (const std::out_of_range&)
		{
			rejected = true;
		}
		require(rejected, "Invalid native depth planes accepted");
	};
	rejects([&]
			{ transfer.scatter(ctx, surface_depth::z16_uint, 2, 2, buffer, {}, {buffer, {0, 4}}); });
	rejects([&]
			{ transfer.scatter(ctx, surface_depth::z24s8_uint, 2, 2, buffer, {}, {buffer, {32, 8}}); });
	rejects([&]
			{ transfer.scatter(ctx, surface_depth::z24s8_uint, 2, 2, buffer, {}, {buffer, {32, 8}}, {buffer, {32, 2}}); });
	rejects([&]
			{ transfer.scatter(ctx, surface_depth::z16_uint, 2, 2, buffer, {}, {buffer, {252, 4}}); });
	rejects([&]
			{ transfer.scatter(ctx, surface_depth::z16_uint, 2, 2, buffer, {}, {buffer, {32, 3}}); });
	rejects([&]
			{ transfer.scatter(ctx, surface_depth::z16_uint, 2, 2, buffer, {}, {buffer, {32, 4}}, {buffer, {64, 2}}); });
	rejects([&]
			{ transfer.scatter(ctx, static_cast<surface_depth>(99), 2, 2, buffer, {}, {buffer, {32, 4}}); });
	ctx.cancel();
}

void append_word(std::vector<std::byte>& bytes, std::uint32_t value)
{
	for (const unsigned shift : {8u, 0u, 24u, 16u})
		bytes.push_back(static_cast<std::byte>(value >> shift));
}

void render_depth(metal4::context& ctx, metal4::surface_depth_transfer& transfer)
{
	using namespace metal4;
	compiler compiler(ctx);
	constexpr auto input_xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	const std::array<std::array<std::uint32_t, 4>, 1> vertex_code{{{(63u << 15) | (7u << 10) | (1u << 30),
																	(1u << 22) | (input_xyzw >> 9), (input_xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
	vertex_shader vertex(compiler, vertex_code);
	std::vector<std::byte> fragment_code;
	for (unsigned destination = 0; destination < 2; ++destination)
	{
		for (const auto word : {(1u << 24) | (15u << 9) | (destination << 1) | destination,
								(0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append_word(fragment_code, word);
		for (const float value : destination ? std::array<float, 4>{0, 0, 0.5f, 0} : std::array<float, 4>{1, 0, 0, 1})
			append_word(fragment_code, std::bit_cast<std::uint32_t>(value));
	}
	fragment_program_options options;
	options.depth_export = true;
	fragment_shader fragment(compiler, fragment_code, options);
	const auto geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	std::array<float, 3 * 64> attributes{};
	const std::array<std::array<float, 4>, 3> positions{{{-1, -1, 0.5f, 1}, {3, -1, 0.5f, 1}, {-1, 3, 0.5f, 1}}};
	for (std::size_t i = 0; i < 3; ++i)
		std::copy(positions[i].begin(), positions[i].end(), attributes.begin() + static_cast<std::ptrdiff_t>(i * 64));
	vertex_inputs input;
	input.attributes = make_upload(ctx, std::as_bytes(std::span(attributes)));
	vertex_context state;
	state.vertex_count = 3;
	MTLTextureDescriptor* cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	cd.storageMode = MTLStorageModePrivate;
	cd.usage = MTLTextureUsageRenderTarget;
	const auto color = make_texture(ctx, cd);
	raster_state raster;
	raster.viewport = {0, 0, 4, 4, 0, 1};
	raster.scissor = {0, 0, 4, 4};
	for (const auto format : {surface_depth::z16_uint, surface_depth::z24s8_uint, surface_depth::z16_float, surface_depth::z24s8_float})
	{
		const auto metadata = describe_surface_depth(format);
		const auto raw = format == surface_depth::z16_uint ? 0x4000u : format == surface_depth::z16_float ? 0x5000u
																   : format == surface_depth::z24s8_uint  ? 0x40000055u
																										  : 0x7d000055u;
		std::vector<std::byte> bytes(metadata.guest_bytes * 16);
		for (std::size_t pixel = 0; pixel < 16; ++pixel)
			store(std::span(bytes).subspan(pixel * metadata.guest_bytes), metadata.guest_bytes, raw, true);
		const auto source = make_upload(ctx, bytes), result = make_buffer(ctx, bytes.size(), buffer_storage::shared);
		MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metadata.format width:4 height:4 mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		const auto texture = make_texture(ctx, td);
		texture_region region;
		region.size = MTLSizeMake(4, 4, 1);
		render_pipeline_configuration config;
		config.colors[0] = MTLPixelFormatRGBA8Unorm;
		config.state.depth_test = true;
		config.state.depth_write = true;
		config.state.depth_compare = 0x201; // Incoming .5 fails LESS against stored .25.
		config.state.stencil_test = metadata.stencil;
		config.state.front.compare = 0x202;
		config.state.front.reference = 0x55;
		config.state.front.depth_fail = 0x1e02; // On depth fail: increment stencil.
		const auto first = make_render_pipeline(compiler, config, vertex.function(), fragment.function());
		config.state.depth_compare = 0x204; // .5 succeeds GREATER against .25.
		config.state.front.reference = 0x56;
		config.state.front.depth_fail = 0x1e00;
		config.state.front.pass = 0x150a; // Successful stencil/depth inverts 0x56->0xa9.
		const auto second = make_render_pipeline(compiler, config, vertex.function(), fragment.function());
		config.state.depth_compare = 0x207;
		config.state.depth_write = false;
		config.state.front.compare = 0x200;
		config.state.front.reference = 0x3c;
		config.state.front.fail = 0x1e01;
		config.state.front.write_mask = 15;
		const auto third = make_render_pipeline(compiler, config, vertex.function(), fragment.function());
		config.state.front.compare = 0x202;
		config.state.front.reference = 0xbc;
		config.state.front.read_mask = 15;
		config.state.front.fail = 0x1e00;
		config.state.front.pass = 0x1e03;
		const auto fourth = make_render_pipeline(compiler, config, vertex.function(), fragment.function());
		auto alternate_code = fragment_code;
		alternate_code.resize(alternate_code.size() - 16);
		for (const float value : {0.f, 0.f, 0.75f, 0.f})
			append_word(alternate_code, std::bit_cast<std::uint32_t>(value));
		ctx.begin();
		transfer.upload(ctx, format, source, {}, texture, region);
		const auto vs = vertex.prepare(ctx, state, geometry, input), fs = fragment.prepare(ctx, {}, fragment_code, {});
		const auto alternate_fs = fragment.prepare(ctx, {}, alternate_code, {});
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = color;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
		pass.depthAttachment.texture = texture;
		pass.depthAttachment.loadAction = MTLLoadActionLoad;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		if (metadata.stencil)
		{
			pass.stencilAttachment.texture = texture;
			pass.stencilAttachment.loadAction = MTLLoadActionLoad;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
		}
		render_commands commands(ctx, pass);
		raster.front_stencil_reference = first.state.front.reference;
		raster.back_stencil_reference = first.state.back.reference;
		commands.draw_geometry(geometry, first.pipeline, first.depth_stencil, raster, vs, fs);
		raster.front_stencil_reference = second.state.front.reference;
		raster.back_stencil_reference = second.state.back.reference;
		commands.draw_geometry(geometry, second.pipeline, second.depth_stencil, raster, vs, fs);
		raster.front_stencil_reference = third.state.front.reference;
		raster.back_stencil_reference = third.state.back.reference;
		commands.draw_geometry(geometry, third.pipeline, third.depth_stencil, raster, vs, alternate_fs);
		raster.front_stencil_reference = fourth.state.front.reference;
		raster.back_stencil_reference = fourth.state.back.reference;
		commands.draw_geometry(geometry, fourth.pipeline, fourth.depth_stencil, raster, vs, alternate_fs);
		commands.end();
		transfer.download(ctx, format, texture, region, result, {});
		ctx.submit().wait();
		const auto actual = map_buffer(result, 0, result.length);
		const auto expected = format == surface_depth::z16_uint ? 0x8000u : format == surface_depth::z16_float ? 0x6000u
																		: format == surface_depth::z24s8_uint  ? 0x7fffffabu
																											   : 0x7e0000abu;
		for (std::size_t pixel = 0; pixel < 16; ++pixel)
		{
			const auto value = load(actual.subspan(pixel * metadata.guest_bytes), metadata.guest_bytes, true);
			if (value != expected)
			{
				std::cerr << "Depth draw format " << unsigned(format) << ": " << std::hex << value << " expected " << expected << std::dec << '\n';
				throw std::runtime_error("Native depth fail/pass write and stencil operation differs");
			}
		}
		std::cout << "Depth " << unsigned(format) << ": LESS/GREATER, enabled/disabled depth writes, stencil fail/depth-fail/pass and masks passed\n";
	}
}
} // namespace

namespace
{
void cube_transfers(metal4::context& ctx,metal4::surface_depth_transfer& transfer)
{
	using namespace metal4;unsigned profiles=0,planes=0,guards=0;std::size_t words=0;
	for(auto format:{surface_depth::z16_uint,surface_depth::z16_float,surface_depth::z24s8_uint,surface_depth::z24s8_float})for(unsigned cubes:{1u,2u})
	{
		@autoreleasepool
		{
			const auto metadata=describe_surface_depth(format);auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:metadata.format width:8 height:8 mipmapped:YES];descriptor.textureType=cubes==1?MTLTextureTypeCube:MTLTextureTypeCubeArray;descriptor.arrayLength=cubes;descriptor.mipmapLevelCount=3;descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;const auto texture=make_texture(ctx,descriptor);
			const auto value=[&](unsigned face,unsigned level,unsigned x,unsigned y){const unsigned n=face*997+level*137+y*11+x;return metadata.stencil?(format==surface_depth::z24s8_float?float24_pattern(n%8)<<8:0x34567800u+n*256)|((face*17+level*7+x+y)&255):format==surface_depth::z16_float?(0x6100u+n)&65535:0x1234u+n;};
			for(unsigned face=0;face<cubes*6;++face)for(unsigned level=0;level<3;++level)
			{
				const unsigned side=8>>level,pitch=side*metadata.guest_bytes+metadata.guest_bytes;std::vector<std::byte> input(3+side*pitch+7,std::byte{0x37});for(unsigned y=0;y<side;++y)for(unsigned x=0;x<side;++x)store(std::span(input).subspan(3+y*pitch+x*metadata.guest_bytes),metadata.guest_bytes,value(face,level,x,y),true);
				const auto guest=make_upload(ctx,input);texture_region region;region.slice=face;region.level=level;region.size=MTLSizeMake(side,side,1);ctx.begin();transfer.upload(ctx,format,guest,{3,pitch},texture,region);ctx.submit().wait();
			}
			// All faces/mips were written before any readback,so indexing bugs
			// cannot hide behind immediately reading the same mistaken face.
			for(unsigned face=0;face<cubes*6;++face)for(unsigned level=0;level<3;++level)
			{
				const unsigned side=8>>level;const auto output=make_buffer(ctx,512*side,buffer_storage::shared);texture_region region;region.slice=face;region.level=level;region.size=MTLSizeMake(side,side,1);ctx.begin();texture_to_buffer(ctx,texture,region,output,0,256,0,texture_plane::depth);if(metadata.stencil)texture_to_buffer(ctx,texture,region,output,256*side,256,0,texture_plane::stencil);ctx.submit().wait();const auto data=map_buffer(output,0,output.length);
				for(unsigned y=0;y<side;++y)for(unsigned x=0;x<side;++x){const unsigned v=value(face,level,x,y);const unsigned expected=format==surface_depth::z16_uint?v:format==surface_depth::z16_float?(v<<11)+(120u<<23):format==surface_depth::z24s8_float?(v>>8)<<7:std::bit_cast<unsigned>(float(v>>8)/16777215.f);require(load(data.subspan(y*256+x*metadata.depth_bytes),metadata.depth_bytes,false)==expected,"Depthcube/cubearray face/mip rawdepth alias orbits differ");if(metadata.stencil)require(data[side*256+y*256+x]==std::byte(v),"Depthcube/cubearray stencil face alias");++words;}++planes;
			}
			const auto guest=make_buffer(ctx,256,buffer_storage::shared);texture_region valid;valid.level=1;valid.slice=cubes*6-1;valid.origin=MTLOriginMake(1,1,0);valid.size=MTLSizeMake(2,2,1);const auto reject=[&](texture_region region){const auto before=ctx.encoding_statistics();bool refused=false;try{transfer.upload(ctx,format,guest,{},texture,region);}catch(const std::invalid_argument&){refused=true;}catch(const std::out_of_range&){refused=true;}require(refused&&ctx.encoding_statistics().compute_encoders==before.compute_encoders,"Malformed depthcube region encoded before refusal");++guards;};
			ctx.begin();auto invalid=valid;invalid.slice=cubes*6;reject(invalid);invalid=valid;invalid.level=3;reject(invalid);invalid=valid;invalid.origin.z=1;reject(invalid);invalid=valid;invalid.size.depth=2;reject(invalid);invalid=valid;invalid.origin.x=3;reject(invalid);ctx.cancel();++profiles;
		}
	}
	std::cout<<"Depthcube codecs:"<<profiles<<" Cube/CubeArray four-depth-format profiles,"<<planes<<" faces/mips,"<<words<<" exact native depthwords/separate stencil includingfloat24 specialbits,"<<guards<<" pre-encoding slice/level/Z/extent refusals;all faces initialized before readback passed\n";
}
void plane_boundaries(metal4::context& ctx, metal4::surface_depth_transfer& transfer)
{
	using namespace metal4;
	const std::array<std::uint32_t, 8> raw{0xff800000, 0xbf800000, 0x80000000, 0x7fc12345, 0x7f800000, 0x40000000, 0x3f800000, 0x3f000000};
	const auto depth = make_upload(ctx, std::as_bytes(std::span(raw)));
	const std::array<std::byte, 8> stencil_values{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8}};
	const auto stencil = make_upload(ctx, stencil_values);
	const auto guest = make_buffer(ctx, 32, buffer_storage::shared);
	ctx.begin();
	transfer.gather(ctx, surface_depth::z24s8_uint, 8, 1, {depth, {}}, {stencil, {}}, guest, {});
	ctx.submit().wait();
	const auto actual = map_buffer(guest, 0, 32);
	const std::array<std::uint32_t, 8> expected{0x00000001, 0x00000002, 0x00000003, 0x00000004, 0xffffff05, 0xffffff06, 0xffffff07, 0x7fffff08};
	for (unsigned i = 0; i < 8; ++i)
		require(load(actual.subspan(i * 4), 4, true) == expected[i], "Normalized depth gather NaN/Inf/range handling differs");
	for (const auto format : {surface_depth::z16_uint, surface_depth::z16_float, surface_depth::z24s8_uint, surface_depth::z24s8_float})
	{
		const auto description = describe_surface_depth(format);
		const std::size_t pitch = 8 * description.guest_bytes;
		std::vector<std::byte> bytes(1 + pitch * 3 + 11, std::byte{0x37});
		for (std::size_t y = 0; y < 3; ++y)
			for (std::size_t x = 0; x < 5; ++x)
			{
				const std::uint32_t value = format == surface_depth::z16_uint ? 0x1234 : format == surface_depth::z16_float ? 0x5678 :
					format == surface_depth::z24s8_uint ? 0x345678ab : 0x7d1234ab;
				store(std::span(bytes).subspan(1 + y * pitch + x * description.guest_bytes), description.guest_bytes, value, true);
			}
		const auto source = make_upload(ctx, bytes), output = make_buffer(ctx, bytes.size(), buffer_storage::shared);
		MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:description.format width:16 height:16 mipmapped:NO];
		td.textureType = MTLTextureType2DArray;
		td.arrayLength = 2;
		td.mipmapLevelCount = 2;
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		const auto texture = make_texture(ctx, td);
		texture_region region;
		region.level = 1;
		region.slice = 1;
		region.origin = MTLOriginMake(2, 1, 0);
		region.size = MTLSizeMake(5, 3, 1);
		ctx.begin();
		fill_buffer(ctx, output, 0, output.length, 0x37);
		transfer.upload(ctx, format, source, {1, pitch}, texture, region);
		transfer.download(ctx, format, texture, region, output, {1, pitch});
		ctx.submit().wait();
		const auto result = map_buffer(output, 0, output.length);
		require(std::equal(result.begin(), result.end(), bytes.begin()), "Depth mip/array/origin/unaligned-offset transfer differs");
	}
	std::cout << "Depth normalized malformed floats, mip/slice/origin and unaligned pitched regions passed\n";
}
}

int main(int argc, char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			descriptors();
			if (argc == 2)
			{
				std::ofstream out(argv[1]);
				out << metal4::surface_depth_kernels;
				if (!out)
					throw std::runtime_error("Cannot export native depth kernel");
				std::cout << "Depth descriptors and invalid formats checked without GPU\n";
				return 0;
			}
			metal4::context context(1);
			metal4::surface_depth_transfer transfer(context);
			for (const auto f : {metal4::surface_depth::z16_uint, metal4::surface_depth::z16_float,
								 metal4::surface_depth::z24s8_uint, metal4::surface_depth::z24s8_float})
				codec(context, transfer, f);
			rejection(context, transfer);
			render_depth(context, transfer);
			plane_boundaries(context, transfer);
			cube_transfers(context,transfer);
			context.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
