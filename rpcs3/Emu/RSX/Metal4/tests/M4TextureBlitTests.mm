#include "M4TextureBlit.h"
#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"

#include <algorithm>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
std::size_t coordinate(std::size_t i, std::size_t source, std::size_t destination, bool flipped)
{
	// Independent exact affine-coordinate model from the Vulkan blit equations.
	// Retain half-pixel numerator explicitly, flip before integer conversion.
	const auto denominator = 2 * destination;
	auto numerator = (2 * i + 1) * source;
	if (flipped)
		numerator = source * denominator - numerator;
	return numerator / denominator;
}
void buffers(context& ctx)
{
	transfer_engine engine(ctx);
	std::size_t cases = 0;
	for (const unsigned bytes : {1u, 2u, 4u, 8u, 16u})
		for (const unsigned mask : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u})
		{
			nearest_image a{{3, 5, 2}, 6 * bytes, 7 * 6 * bytes}, b{{7, 2, 3}, 12 * bytes, 4 * 12 * bytes};
			const auto layout = plan_nearest(a, b, bytes);
			std::vector<std::byte> source(3 + layout.source.allocation_size + 17), destination(7 + layout.destination.allocation_size + 19, std::byte{0xcc});
			for (std::size_t i = 0; i < source.size(); ++i)
				source[i] = static_cast<std::byte>((i * 13 + 71) % 256);
			const auto input = make_upload(ctx, source), output = make_upload(ctx, destination);
			ctx.begin();
			engine.nearest(ctx, input, 3, a, output, 7, b, bytes, {bool(mask & 1), bool(mask & 2), bool(mask & 4)});
			ctx.submit().wait();
			auto expected = destination;
			for (std::size_t z = 0; z < b.size[2]; ++z)
				for (std::size_t y = 0; y < b.size[1]; ++y)
					for (std::size_t x = 0; x < b.size[0]; ++x)
					{
						const auto from = 3 + coordinate(z, a.size[2], b.size[2], mask & 4) * a.image_pitch +
										  coordinate(y, a.size[1], b.size[1], mask & 2) * a.row_pitch + coordinate(x, a.size[0], b.size[0], mask & 1) * bytes;
						const auto to = 7 + z * b.image_pitch + y * b.row_pitch + x * bytes;
						std::copy_n(source.begin() + from, bytes, expected.begin() + to);
					}
			const auto actual = map_buffer(output, 0, output.length);
			require(std::equal(actual.begin(), actual.end(), expected.begin()), "Nearest buffer 3D/misaligned offset/padding/flip differs");
			++cases;
		}
	// Exercise exact integer sampling boundaries in both orientations. Reversing
	// the index after nearest selection gives a different answer for 4->2.
	for (std::uint32_t src = 1; src <= 17; ++src)
		for (std::uint32_t dst = 1; dst <= 17; ++dst)
			for (const bool flip : {false, true})
			{
				std::vector<std::byte> data(src);
				for (std::uint32_t i = 0; i < src; ++i)
					data[i] = static_cast<std::byte>(i);
				const auto input = make_upload(ctx, data), output = make_buffer(ctx, dst, buffer_storage::shared);
				ctx.begin();
				engine.nearest(ctx, input, 0, {{src, 1, 1}}, output, 0, {{dst, 1, 1}}, 1, {flip, false, false});
				ctx.submit().wait();
				const auto actual = map_buffer(output, 0, dst);
				for (std::size_t i = 0; i < dst; ++i)
					require(actual[i] == static_cast<std::byte>(coordinate(i, src, dst, flip)), "Nearest rational boundary selected the wrong source texel");
				++cases;
			}
	std::cout << cases << " native nearest buffer cases preserve raw bytes, pitches, unaligned offsets, axis flips and rational boundaries\n";
}
std::vector<std::byte> pattern(std::size_t count, unsigned bytes, bool depth_float)
{
	std::vector<std::byte> data(count * bytes);
	for (std::size_t i = 0; i < data.size(); ++i)
		data[i] = static_cast<std::byte>((i * 37 + 91) % 256);
	if (bytes % 4 == 0 || depth_float)
		for (std::size_t i = 0; i < data.size(); i += 4)
		{
			const std::array<std::uint32_t, 6> words{0x7fc12345, 0xff800000, 0x40000000, 1, 0x80000000, 0x3f000007};
			const auto word = words[(i / 4) % words.size()];
			std::copy_n(reinterpret_cast<const std::byte*>(&word), 4, data.begin() + i);
		}
	return data;
}
void texture_case(context& ctx, texture_blitter& blitter, MTLPixelFormat format, bool alias, unsigned mask, unsigned mode, texture_plane plane)
{
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:24 height:20 mipmapped:NO];
	descriptor.textureType = MTLTextureType2DArray;
	descriptor.arrayLength = 2;
	descriptor.mipmapLevelCount = 2;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	const auto texture = make_texture(ctx, descriptor);
	texture_region whole;
	whole.level = 1;
	whole.size = MTLSizeMake(12, 10, 1);
	const auto aspects = format == MTLPixelFormatDepth32Float_Stencil8 ? std::vector{texture_plane::depth, texture_plane::stencil} : std::vector{texture_plane::all};
	struct observation
	{
		texture_plane aspect;
		unsigned bytes;
		std::vector<std::byte> original, initial;
		id<MTLBuffer> input, target, output;
	};
	std::vector<observation> observations;
	for (const auto aspect : aspects)
	{
		const auto bytes = static_cast<unsigned>(storage_block(format, aspect).bytes);
		auto original = pattern(120, bytes, aspect == texture_plane::depth);
		std::vector<std::byte> initial(original.size(), std::byte{0xcc});
		observations.push_back({aspect, bytes, original, initial, make_upload(ctx, original), make_upload(ctx, initial), make_buffer(ctx, original.size(), buffer_storage::shared)});
	}
	ctx.begin();
	for (const auto& o : observations)
	{
		buffer_to_texture(ctx, o.input, 0, texture, whole, 12 * o.bytes, 0, o.aspect);
		if (!alias)
		{
			auto other = whole;
			other.slice = 1;
			buffer_to_texture(ctx, o.target, 0, texture, other, 12 * o.bytes, 0, o.aspect);
		}
	}
	const auto identity = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
	const auto view = make_texture_view(texture, format, MTLTextureType2D, NSMakeRange(1, 1), NSMakeRange(0, 1), identity);
	texture_region sr, dr = whole;
	sr.origin = MTLOriginMake(1, 2, 0);
	dr.slice = alias ? 0 : 1;
	dr.origin = MTLOriginMake(2, 1, 0);
	sr.size = mode == 0 ? MTLSizeMake(5, 3, 1) : mode == 1 ? MTLSizeMake(8, 6, 1)
														   : MTLSizeMake(6, 4, 1);
	dr.size = mode == 0 ? MTLSizeMake(7, 5, 1) : mode == 1 ? MTLSizeMake(3, 2, 1)
														   : MTLSizeMake(6, 4, 1);
	blitter.nearest(ctx, view, sr, texture, dr, plane, {bool(mask & 1), bool(mask & 2), false});
	whole.slice = dr.slice;
	for (const auto& o : observations)
		texture_to_buffer(ctx, texture, whole, o.output, 0, 12 * o.bytes, 0, o.aspect);
	ctx.submit().wait();
	for (const auto& o : observations)
	{
		auto expected = alias ? o.original : o.initial;
		if (plane == texture_plane::all || plane == o.aspect)
			for (std::size_t y = 0; y < dr.size.height; ++y)
				for (std::size_t x = 0; x < dr.size.width; ++x)
				{
					const auto sy = sr.origin.y + coordinate(y, sr.size.height, dr.size.height, mask & 2);
					const auto sx = sr.origin.x + coordinate(x, sr.size.width, dr.size.width, mask & 1);
					const auto from = (sy * 12 + sx) * o.bytes, to = ((y + dr.origin.y) * 12 + x + dr.origin.x) * o.bytes;
					std::copy_n(o.original.begin() + from, o.bytes, expected.begin() + to);
				}
		const auto actual = map_buffer(o.output, 0, o.output.length);
		if (!std::equal(actual.begin(), actual.end(), expected.begin()))
		{
			std::cerr << "Nearest fmt=" << unsigned(format) << " alias=" << alias << " flip=" << mask << " mode=" << mode << " aspect=" << unsigned(o.aspect) << '\n';
			throw std::runtime_error("Native nearest texture raw bits/outside pixels/unselected aspect differs");
		}
	}
}
void textures(context& ctx)
{
	texture_blitter blitter(ctx);
	std::size_t count = 0;
	for (const bool depth : {false, true})
		for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
		{
			if (depth && raw > 4)
				continue;
			const auto format = depth ? describe_surface_depth(static_cast<surface_depth>(raw)).format : describe_surface_color(static_cast<surface_color>(raw)).format;
			const auto planes = format == MTLPixelFormatDepth32Float_Stencil8 ? std::vector{texture_plane::all, texture_plane::depth, texture_plane::stencil} : std::vector{texture_plane::all};
			for (const auto plane : planes)
				for (const bool alias : {false, true})
					for (const unsigned mask : {0u, 1u, 2u, 3u})
						for (const unsigned mode : {0u, 1u, 2u})
						{
							@autoreleasepool
							{
								texture_case(ctx, blitter, format, alias, mask, mode, plane);
							}
							++count;
						}
		}
	std::cout << count << " native nearest texture raw-plane cases across all surface formats preserve NaN/depth/stencil bits, mip/array/parent views, flips and overlapping snapshots\n";
}
void volume(context& ctx)
{
	texture_blitter blitter(ctx);
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:8 height:8 mipmapped:NO];
	descriptor.textureType = MTLTextureType3D;
	descriptor.depth = 6;
	descriptor.storageMode = MTLStorageModePrivate;
	const auto texture = make_texture(ctx, descriptor);
	const auto data = pattern(8 * 8 * 6, 1, false);
	const auto input = make_upload(ctx, data), output = make_buffer(ctx, data.size(), buffer_storage::shared);
	texture_region whole;
	whole.size = MTLSizeMake(8, 8, 6);
	for (const unsigned mask : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u})
	{
		ctx.begin();
		buffer_to_texture(ctx, input, 0, texture, whole, 8, 64);
		texture_region sr, dr;
		sr.origin = MTLOriginMake(1, 2, 1);
		sr.size = MTLSizeMake(3, 5, 2);
		dr.origin = MTLOriginMake(2, 1, 1);
		dr.size = MTLSizeMake(5, 3, 4);
		blitter.nearest(ctx, texture, sr, texture, dr, texture_plane::all, {bool(mask & 1), bool(mask & 2), bool(mask & 4)});
		texture_to_buffer(ctx, texture, whole, output, 0, 8, 64);
		ctx.submit().wait();
		auto expected = data;
		for (std::size_t z = 0; z < 4; ++z)
			for (std::size_t y = 0; y < 3; ++y)
				for (std::size_t x = 0; x < 5; ++x)
				{
					const auto from = (1 + coordinate(z, 2, 4, mask & 4)) * 64 + (2 + coordinate(y, 5, 3, mask & 2)) * 8 + 1 + coordinate(x, 3, 5, mask & 1);
					const auto to = (z + 1) * 64 + (y + 1) * 8 + x + 2;
					expected[to] = data[from];
				}
		const auto actual = map_buffer(output, 0, output.length);
		require(std::equal(actual.begin(), actual.end(), expected.begin()), "Native 3D nearest overlap/flip differs");
	}
	std::cout << "8 native 3D overlap scales with independent x/y/z flips passed\n";
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
			buffers(ctx);
			textures(ctx);
			volume(ctx);
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
