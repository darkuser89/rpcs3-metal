#include "M4Clear.h"
#include "M4SampleTransfer.h"

#include <algorithm>
#include <bit>
#include <iostream>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void store(std::span<std::byte> data, std::uint32_t value, unsigned bytes)
{
	for (unsigned byte = 0; byte < bytes; ++byte)
		data[byte] = static_cast<std::byte>(value >> ((bytes - 1 - byte) * 8));
}
std::uint32_t color_replacement(unsigned format, unsigned mask, std::uint32_t seed, std::uint32_t value)
{
	std::uint32_t fields = 0;
	if (format <= 3)
	{
		if (mask & 1)
			fields |= format == 3 ? 0xf800 : 0x7c00;
		if (mask & 2)
			fields |= format == 3 ? 0x07e0 : 0x03e0;
		if (mask & 4)
			fields |= 0x001f;
		if ((mask & 8) && format != 3)
		{
			fields |= 0x8000;
			value = format == 2 ? value | 0x8000 : value & 0x7fff;
		}
	}
	else if (format == 9)
	{
		if (mask & 4)
			fields = 255;
	}
	else if (format == 10)
	{
		if (mask & 2)
			fields |= 0xff00;
		if (mask & 4)
			fields |= 255;
	}
	else
	{
		if (mask & 1)
			fields |= 0xff0000;
		if (mask & 2)
			fields |= 0xff00;
		if (mask & 4)
			fields |= 255;
		if (mask & 8)
			fields |= 0xff000000;
	}
	return (seed & ~fields) | (value & fields);
}

void color_cases(metal4::context& ctx)
{
	using namespace metal4;
	clear_engine clear(ctx);
	surface_color_transfer bytes(ctx);
	sample_transfer samples(ctx);
	std::size_t cases = 0;
	for (const unsigned count : {1u, 2u, 4u})
		for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
		{
			const auto format = static_cast<surface_color>(raw);
			const auto description = describe_surface_color(format);
			constexpr std::size_t width = 8, height = 6;
			const auto size = count == 1 ? std::array<std::size_t, 2>{width, height} : sample_expanded_size(width, height, count);
			const auto bpp = description.layout.pixel_bytes;
			const auto pitch = static_cast<std::size_t>(size[0] * bpp);
			std::vector<std::byte> original(pitch * size[1]);
			for (std::size_t pixel = 0; pixel < size[0] * size[1]; ++pixel)
			{
				if (raw == 11 || raw == 12 || raw == 13)
				{
					const unsigned element = raw == 11 ? 2 : 4;
					for (unsigned c = 0; c < bpp / element; ++c)
						store(std::span(original).subspan(pixel * bpp + c * element), raw == 11 ? 0x3800u : 0x3f000000u, element);
				}
				else
					store(std::span(original).subspan(pixel * bpp), 0xa5c37e12u, bpp);
			}
			const auto input = make_upload(ctx, original), output = make_buffer(ctx, original.size(), buffer_storage::shared);
			MTLTextureDescriptor* ed = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:description.format width:size[0] height:size[1] mipmapped:NO];
			ed.storageMode = MTLStorageModePrivate;
			ed.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto expanded = make_texture(ctx, ed), resolved = count == 1 ? expanded : make_texture(ctx, ed);
			MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:description.format width:width height:height mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			td.usage = ed.usage;
			if (count > 1)
			{
				td.textureType = MTLTextureType2DMultisample;
				td.sampleCount = count;
			}
			const auto target = count == 1 ? expanded : make_texture(ctx, td);
			texture_region region;
			region.size = MTLSizeMake(size[0], size[1], 1);
			for (unsigned mask = 0; mask < 16; ++mask)
				for (const bool full : {false, true})
				{
					ctx.begin();
					bytes.upload(ctx, format, input, {}, expanded, region);
					if (count > 1)
						samples.import(ctx, expanded, target);
					clear_request request;
					request.colors[0] = {{target, 0, 0}, format};
					request.mask = mask << 4;
					request.color_value = 0x7b3dc5e9;
					request.rectangle = full ? clear_rectangle{0, 0, width, height} : clear_rectangle{2, 1, 3, 3};
					const auto changed = clear.run(ctx, request);
					if (count > 1)
						samples.expand(ctx, target, resolved);
					bytes.download(ctx, format, resolved, region, output, {});
					ctx.submit().wait();
					auto expected = original;
					for (std::size_t y = 0; y < size[1]; ++y)
						for (std::size_t x = 0; x < size[0]; ++x)
						{
							const auto px = count == 1 ? x : x / 2, py = count == 4 ? y / 2 : y;
							if (!full && (px < 2 || px >= 5 || py < 1 || py >= 4))
								continue;
							if (raw == 11 || raw == 12 || raw == 13)
								continue;
							const auto value = color_replacement(raw, mask, 0xa5c37e12u, 0x7b3dc5e9);
							store(std::span(expected).subspan((y * size[0] + x) * bpp), value, bpp);
						}
					const auto actual = map_buffer(output, 0, output.length);
					if (!std::equal(actual.begin(), actual.end(), expected.begin()))
					{
						const auto error = std::mismatch(actual.begin(), actual.end(), expected.begin());
						std::cerr << "Clear color " << raw << " samples " << count << " mask " << mask << " full " << full << " byte " << error.first - actual.begin()
								  << ": " << unsigned(*error.first) << " expected " << unsigned(*error.second) << '\n';
						throw std::runtime_error("Native full/masked/scissored color clear differs");
					}
					if (raw == 11 || raw == 12 || raw == 13)
						require(!changed.color_targets, "Float framebuffer clear wasn't NOP");
					++cases;
				}
		}
	std::cout << cases << " native full/masked/scissored color clears across 14 formats and 1/2/4 samples passed\n";
}

void depth_cases(metal4::context& ctx)
{
	using namespace metal4;
	clear_engine clear(ctx);
	surface_depth_transfer bytes(ctx);
	sample_transfer samples(ctx);
	std::size_t cases = 0;
	for (const unsigned count : {1u, 2u, 4u})
		for (const unsigned raw : {1u, 2u, 3u, 4u})
		{
			const auto format = static_cast<surface_depth>(raw);
			const auto description = describe_surface_depth(format);
			constexpr std::size_t width = 8, height = 6;
			const auto size = count == 1 ? std::array<std::size_t, 2>{width, height} : sample_expanded_size(width, height, count);
			const auto seed = raw == 1 ? 0x4000u : raw == 3 ? 0x5000u
											   : raw == 2	? 0x400000a5u
															: 0x7d0000a5u;
			const auto bpp = description.guest_bytes;
			std::vector<std::byte> original(size[0] * size[1] * bpp);
			for (std::size_t pixel = 0; pixel < size[0] * size[1]; ++pixel)
				store(std::span(original).subspan(pixel * bpp), seed, bpp);
			const auto input = make_upload(ctx, original), output = make_buffer(ctx, original.size(), buffer_storage::shared);
			MTLTextureDescriptor* ed = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:description.format width:size[0] height:size[1] mipmapped:NO];
			ed.storageMode = MTLStorageModePrivate;
			ed.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto expanded = make_texture(ctx, ed), resolved = count == 1 ? expanded : make_texture(ctx, ed);
			MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:description.format width:width height:height mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			td.usage = ed.usage;
			if (count > 1)
			{
				td.textureType = MTLTextureType2DMultisample;
				td.sampleCount = count;
			}
			const auto target = count == 1 ? expanded : make_texture(ctx, td);
			texture_region region;
			region.size = MTLSizeMake(size[0], size[1], 1);
			const auto zvalue = description.stencil ? 0x800000u : 0x8000u;
			const float normalized = float(zvalue) / (description.stencil ? 16777215.0f : 65535.0f);
			const auto fb = std::bit_cast<std::uint32_t>(normalized);
			const auto cleared_depth = raw == 1 ? zvalue : raw == 3 ? ((fb - 0x3c000000u) / 2048) % 65536
													   : raw == 2	? static_cast<std::uint32_t>(normalized * 16777215.0f)
																	: fb / 128;
			for (unsigned mask = 0; mask < 4; ++mask)
				for (unsigned stencil_mask = 0; stencil_mask < (description.stencil ? 256u : 1u); ++stencil_mask)
					for (const bool full : {false, true})
					{
						ctx.begin();
						bytes.upload(ctx, format, input, {}, expanded, region);
						if (count > 1)
							samples.import(ctx, expanded, target);
						clear_request request;
						request.depth = {target, 0, 0};
						request.depth_format = format;
						request.mask = mask;
						request.zstencil_value = description.stencil ? (zvalue << 8) | 0x3c : zvalue;
						request.stencil_write_mask = stencil_mask;
						request.rectangle = full ? clear_rectangle{0, 0, width, height} : clear_rectangle{2, 1, 3, 3};
						const auto changed = clear.run(ctx, request);
						if (count > 1)
							samples.expand(ctx, target, resolved);
						bytes.download(ctx, format, resolved, region, output, {});
						ctx.submit().wait();
						auto expected = original;
						const auto expected_depth = (mask & 1) ? cleared_depth : (description.stencil ? seed >> 8 : seed);
						const auto expected_stencil = (mask & 2) ? (0xa5u & (~stencil_mask & 255)) | (0x3cu & stencil_mask) : 0xa5u;
						const auto word = description.stencil ? (expected_depth << 8) | expected_stencil : expected_depth;
						for (std::size_t y = 0; y < size[1]; ++y)
							for (std::size_t x = 0; x < size[0]; ++x)
							{
								const auto px = count == 1 ? x : x / 2, py = count == 4 ? y / 2 : y;
								if (full || (px >= 2 && px < 5 && py >= 1 && py < 4))
									store(std::span(expected).subspan((y * size[0] + x) * bpp), word, bpp);
							}
						const auto actual = map_buffer(output, 0, output.length);
						if (!std::equal(actual.begin(), actual.end(), expected.begin()))
						{
							const auto error = std::mismatch(actual.begin(), actual.end(), expected.begin());
							std::cerr << "Clear depth " << raw << " samples " << count << " mask " << mask << " stencil mask " << stencil_mask << " full " << full << " byte " << error.first - actual.begin()
									  << ": " << unsigned(*error.first) << " expected " << unsigned(*error.second) << '\n';
							throw std::runtime_error("Native clear depth/stencil aspect/mask/region/sample preservation differs");
						}
						require(changed.depth == bool(mask & 1), "Depth clear result mask differs");
						require(changed.stencil == bool((mask & 2) && description.stencil && stencil_mask), "Stencil clear result mask differs");
						++cases;
					}
		}
	require(clear.depth_states_created() == 511 && clear.depth_states_reused() > 511,
		"Direct clear state index failed to reuse all 511 reachable depth/mask combinations");
	std::cout << "Clear depth-state direct index:511 reachable states created once; " << clear.depth_states_reused() << " constant-index hits passed\n";
	std::cout << cases << " native depth/stencil clears across four formats, all 256 stencil masks and 1/2/4 samples passed\n";
}

void attachment_cases(metal4::context& ctx)
{
	using namespace metal4;
	clear_engine clear(ctx);
	surface_color_transfer transfer(ctx);
	const std::array<surface_color, 4> formats{surface_color::argb8, surface_color::gb8, surface_color::b8, surface_color::abgr8};
	std::array<id<MTLTexture>, 4> textures{};
	std::array<id<MTLBuffer>, 4> input{}, output{};
	texture_region region;
	region.level = 1;
	region.slice = 1;
	region.size = MTLSizeMake(8, 8, 1);
	for (std::size_t slot = 0; slot < 4; ++slot)
	{
		const auto d = describe_surface_color(formats[slot]);
		MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:d.format width:16 height:16 mipmapped:NO];
		descriptor.textureType = MTLTextureType2DArray;
		descriptor.arrayLength = 2;
		descriptor.mipmapLevelCount = 2;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
		textures[slot] = make_texture(ctx, descriptor);
		std::vector<std::byte> seed(64 * d.layout.pixel_bytes, std::byte{0xa5});
		input[slot] = make_upload(ctx, seed);
		output[slot] = make_buffer(ctx, seed.size(), buffer_storage::shared);
	}
	for (const bool sparse : {false, true})
	{
		ctx.begin();
		clear_request request;
		request.mask = 0xf0;
		request.color_value = 0x173b7dc1;
		request.rectangle = {6, 5, 99, 99}; // Clipped to [6,8)x[5,8), no overflow.
		for (std::size_t slot = 0; slot < 4; ++slot)
		{
			transfer.upload(ctx, formats[slot], input[slot], {}, textures[slot], region);
			if (!sparse || slot == 3)
				request.colors[slot] = {{textures[slot], 1, 1}, formats[slot]};
		}
		const auto result = clear.run(ctx, request);
		require(result.color_targets == (sparse ? 8u : 15u), "Clear result MRT mapping differs");
		for (std::size_t slot = 0; slot < 4; ++slot)
			transfer.download(ctx, formats[slot], textures[slot], region, output[slot], {});
		ctx.submit().wait();
		for (std::size_t slot = 0; slot < 4; ++slot)
		{
			const auto bpp = describe_surface_color(formats[slot]).layout.pixel_bytes;
			const auto actual = map_buffer(output[slot], 0, output[slot].length);
			for (std::size_t y = 0; y < 8; ++y)
				for (std::size_t x = 0; x < 8; ++x)
				{
					std::array<std::byte, 4> expected;
					expected.fill(std::byte{0xa5});
					if (x >= 6 && y >= 5 && (!sparse || slot == 3))
						store(expected, slot == 2 ? 0xc1u : slot == 1 ? 0x7dc1u
																	  : 0x173b7dc1u,
							  bpp);
					require(std::equal(expected.begin(), expected.begin() + bpp, actual.begin() + static_cast<std::ptrdiff_t>((y * 8 + x) * bpp)), "Clear sparse MRT/mip/slice/clipped rectangle differs");
				}
		}
	}
	ctx.begin();
	clear_request empty;
	empty.colors[0] = {{textures[0], 1, 1}, formats[0]};
	empty.mask = 0xf0;
	empty.rectangle = {999, 999, 999, 999};
	require(clear.run(ctx, empty).color_targets == 0, "Out-of-frame clear wasn't empty");
	empty.rectangle = {0, 0, 8, 8};
	empty.colors[0].image.level = 99;
	bool rejected = false;
	try
	{
		(void)clear.run(ctx, empty);
	}
	catch (const std::invalid_argument&)
	{
		rejected = true;
	}
	require(rejected, "Invalid clear mip was accepted");
	empty.colors[0].image.level = 1;
	empty.colors[1] = empty.colors[0];
	rejected = false;
	try
	{
		(void)clear.run(ctx, empty);
	}
	catch (const std::invalid_argument&)
	{
		rejected = true;
	}
	require(rejected, "Aliased clear MRT attachments were accepted");
	const auto view = make_texture_view(textures[0], textures[0].pixelFormat, MTLTextureType2D, NSMakeRange(1, 1), NSMakeRange(1, 1),
										MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha));
	empty.colors[1].image = {view, 0, 0};
	rejected = false;
	try
	{
		(void)clear.run(ctx, empty);
	}
	catch (const std::invalid_argument&)
	{
		rejected = true;
	}
	require(rejected, "Aliased clear view subresources were accepted");
	empty.colors[1].image = {textures[0], 1, 0};
	region.slice = 0;
	transfer.upload(ctx, formats[0], input[0], {}, textures[0], region);
	region.slice = 1;
	transfer.upload(ctx, formats[0], input[0], {}, textures[0], region);
	empty.color_value = 0x173b7dc1;
	require(clear.run(ctx, empty).color_targets == 3, "Disjoint slices of one MRT texture were rejected");
	const std::array<id<MTLBuffer>, 2> slice_output{make_buffer(ctx, 256, buffer_storage::shared), make_buffer(ctx, 256, buffer_storage::shared)};
	for (const unsigned slice : {0u, 1u})
	{
		region.slice = slice;
		transfer.download(ctx, formats[0], textures[0], region, slice_output[slice], {});
	}
	ctx.submit().wait();
	for (const unsigned slice : {0u, 1u})
	{
		const auto actual = map_buffer(slice_output[slice], 0, 256);
		for (std::size_t pixel = 0; pixel < 64; ++pixel)
		{
			std::array<std::byte, 4> expected;
			store(expected, 0x173b7dc1, 4);
			require(std::equal(expected.begin(), expected.end(), actual.begin() + static_cast<std::ptrdiff_t>(pixel * 4)), "Disjoint MRT array slice clear differs");
		}
	}
	std::cout << "Mixed/sparse four-MRT clears, mip/slice, clipping, invalid subresources and parent-view alias guards passed\n";
}

void combined_cases(metal4::context& ctx)
{
	using namespace metal4;
	clear_engine clear(ctx);
	surface_color_transfer colors(ctx);
	surface_depth_transfer depths(ctx);
	MTLTextureDescriptor* cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:4 mipmapped:NO];
	cd.storageMode = MTLStorageModePrivate;
	cd.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	const auto color = make_texture(ctx, cd);
	cd.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
	const auto depth = make_texture(ctx, cd);
	std::vector<std::byte> cs(64), ds(64);
	for (std::size_t pixel = 0; pixel < 16; ++pixel)
	{
		store(std::span(cs).subspan(pixel * 4), 0xa5c37e12, 4);
		store(std::span(ds).subspan(pixel * 4), 0x400000a5, 4);
	}
	const auto ci = make_upload(ctx, cs), di = make_upload(ctx, ds), co = make_buffer(ctx, 64, buffer_storage::shared), dout = make_buffer(ctx, 64, buffer_storage::shared);
	texture_region region;
	region.size = MTLSizeMake(4, 4, 1);
	for (const unsigned mask : {0xf3u, 0x53u, 0x32u})
		for (const bool full : {false, true})
		{
			ctx.begin();
			colors.upload(ctx, surface_color::argb8, ci, {}, color, region);
			depths.upload(ctx, surface_depth::z24s8_uint, di, {}, depth, region);
			clear_request request;
			request.colors[0] = {{color, 0, 0}, surface_color::argb8};
			request.depth = {depth, 0, 0};
			request.depth_format = surface_depth::z24s8_uint;
			request.mask = mask;
			request.color_value = 0x173b7dc1;
			request.zstencil_value = 0x8000003c;
			request.stencil_write_mask = mask == 0x32 ? 0x59 : 255;
			request.rectangle = full ? clear_rectangle{0, 0, 4, 4} : clear_rectangle{1, 1, 2, 2};
			(void)clear.run(ctx, request);
			colors.download(ctx, surface_color::argb8, color, region, co, {});
			depths.download(ctx, surface_depth::z24s8_uint, depth, region, dout, {});
			ctx.submit().wait();
			const auto ca = map_buffer(co, 0, 64), da = map_buffer(dout, 0, 64);
			for (std::size_t y = 0; y < 4; ++y)
				for (std::size_t x = 0; x < 4; ++x)
				{
					const bool changed = full || (x >= 1 && x < 3 && y >= 1 && y < 3);
					const auto cv = changed ? color_replacement(8, mask >> 4, 0xa5c37e12, request.color_value) : 0xa5c37e12;
					const auto zv = changed && (mask & 1) ? 0x800000u : 0x400000u;
					const auto sv = changed && (mask & 2) ? (0xa5u & (~request.stencil_write_mask & 255)) | (0x3cu & request.stencil_write_mask) : 0xa5u;
					std::array<std::byte, 4> expected;
					store(expected, cv, 4);
					require(std::equal(expected.begin(), expected.end(), ca.begin() + static_cast<std::ptrdiff_t>((y * 4 + x) * 4)), "Combined clear color differs");
					store(expected, (zv << 8) | sv, 4);
					require(std::equal(expected.begin(), expected.end(), da.begin() + static_cast<std::ptrdiff_t>((y * 4 + x) * 4)), "Combined clear depth/stencil differs");
				}
		}
	cd.pixelFormat = MTLPixelFormatDepth16Unorm;
	const auto d16 = make_texture(ctx, cd);
	for (const unsigned word : {0u, 1u, 2u, 3u, 0x3fffu, 0x4000u, 0x4001u, 0x7ffeu, 0x7fffu, 0x8000u, 0x8001u, 0xbfffu, 0xc000u, 0xfffdu, 0xfffeu, 0xffffu})
	{
		ctx.begin();
		clear_request request;
		request.depth = {d16, 0, 0};
		request.mask = 1;
		request.zstencil_value = word;
		request.rectangle = {0, 0, 4, 4};
		(void)clear.run(ctx, request);
		depths.download(ctx, surface_depth::z16_uint, d16, region, dout, {});
		ctx.submit().wait();
		const auto actual = map_buffer(dout, 0, 32);
		for (std::size_t pixel = 0; pixel < 16; ++pixel)
			require(unsigned(actual[pixel * 2]) == word / 256 && unsigned(actual[pixel * 2 + 1]) == word % 256, "D16 clear boundary word differs");
	}
	std::cout << "Six combined color/depth/stencil clears and sixteen exact D16 boundary words passed\n";
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
			color_cases(ctx);
			depth_cases(ctx);
			attachment_cases(ctx);
			combined_cases(ctx);
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
