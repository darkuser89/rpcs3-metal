#include "M4SurfaceTypeless.h"

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
const std::array<surface_format, 18> formats{surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565,
											 surface_color::xrgb8_zero, surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8,
											 surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero,
											 surface_color::xbgr8_one, surface_color::abgr8, surface_depth::z16_uint, surface_depth::z24s8_uint,
											 surface_depth::z16_float, surface_depth::z24s8_float};
std::size_t coordinate(std::size_t i, std::size_t source, std::size_t destination, bool flip)
{
	const auto divisor = 2 * destination, value = (2 * i + 1) * source;
	return (flip ? source * divisor - value : value) / divisor;
}
id<MTLTexture> texture(context& ctx, surface_format format, std::size_t width, std::size_t height)
{
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:surface_native_format(format) width:width * 2 height:height * 2 mipmapped:NO];
	descriptor.mipmapLevelCount = 2;
	descriptor.arrayLength = 2;
	descriptor.textureType = MTLTextureType2DArray;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	return make_texture(ctx, descriptor);
}
void matrix(context& ctx)
{
	surface_typeless_transfer typeless(ctx);
	surface_color_transfer color(ctx);
	surface_depth_transfer depth(ctx);
	const auto upload = [&](surface_format format, id<MTLBuffer> input, id<MTLTexture> target, const texture_region& region)
	{
		if (const auto c = std::get_if<surface_color>(&format))
			color.upload(ctx, *c, input, {}, target, region);
		else
			depth.upload(ctx, std::get<surface_depth>(format), input, {}, target, region);
	};
	const auto download = [&](surface_format format, id<MTLTexture> source, const texture_region& region, id<MTLBuffer> target)
	{
		if (const auto c = std::get_if<surface_color>(&format))
			color.download(ctx, *c, source, region, target, {});
		else
			depth.download(ctx, std::get<surface_depth>(format), source, region, target, {});
	};
	std::size_t count = 0;
	for (std::size_t a = 0; a < formats.size(); ++a)
		for (std::size_t b = 0; b < formats.size(); ++b)
			for (const unsigned mode : {0u, 1u, 2u})
				for (const unsigned flip : {0u, 1u, 2u, 3u})
				{
					@autoreleasepool
					{
						const auto sf = formats[a], df = formats[b];
						const auto sb = surface_guest_bytes(sf), db = surface_guest_bytes(df);
						const auto sw = 32 / sb, reinterpreted = 32 / db;
						const auto dw = mode == 0 ? reinterpreted : mode == 1 ? reinterpreted + 3
																			  : std::max<std::size_t>(1, reinterpreted / 2);
						const std::size_t dh = mode == 0 ? 3 : mode == 1 ? 5
																		 : 2;
						const auto source = texture(ctx, sf, sw + 3, 6), destination = texture(ctx, df, dw + 4, 8);
						texture_region source_whole, target_whole;
						source_whole.level = target_whole.level = 1;
						target_whole.slice = 1;
						source_whole.size = MTLSizeMake(sw + 3, 6, 1);
						target_whole.size = MTLSizeMake(dw + 4, 8, 1);
						std::vector<std::byte> original((sw + 3) * 6 * sb), initial((dw + 4) * 8 * db, std::byte{0xcc});
						for (std::size_t i = 0; i < original.size(); ++i)
							original[i] = static_cast<std::byte>((i * 37 + i / 7 + 19) % 256);
						const auto input = make_upload(ctx, original), seed = make_upload(ctx, initial), output = make_buffer(ctx, initial.size(), buffer_storage::shared);
						ctx.begin();
						upload(sf, input, source, source_whole);
						upload(df, seed, destination, target_whole);
						texture_region sr = source_whole, dr = target_whole;
						sr.origin = MTLOriginMake(1, 1, 0);
						sr.size = MTLSizeMake(sw, 3, 1);
						dr.origin = MTLOriginMake(2, 1, 0);
						dr.size = MTLSizeMake(dw, dh, 1);
						typeless.nearest(ctx, sf, source, sr, df, destination, dr, {bool(flip & 1), bool(flip & 2), false});
						download(df, destination, target_whole, output);
						ctx.submit().wait();
						auto expected = initial;
						for (std::size_t y = 0; y < dh; ++y)
							for (std::size_t x = 0; x < dw; ++x)
							{
								const auto from = (1 + coordinate(y, 3, dh, flip & 2)) * (sw + 3) * sb + sb + coordinate(x, reinterpreted, dw, flip & 1) * db;
								const auto to = ((y + 1) * (dw + 4) + x + 2) * db;
								std::copy_n(original.begin() + from, db, expected.begin() + to);
							}
						const auto actual = map_buffer(output, 0, output.length);
						if (!std::equal(actual.begin(), actual.end(), expected.begin()))
						{
							const auto difference = std::mismatch(actual.begin(), actual.end(), expected.begin());
							std::cerr << "Typeless source=" << a << " target=" << b << " mode=" << mode << " flip=" << flip << " byte=" << difference.first - actual.begin() << " actual=" << unsigned(*difference.first) << " expected=" << unsigned(*difference.second) << '\n';
							throw std::runtime_error("Typeless guest storage words, scaling or untouched target pixels differ");
						}
					}
					++count;
				}
	std::cout << count << " native typeless 18x18 format-pair copies/scales/flips preserve BE guest words and exterior array/mip pixels\n";
}
void same_native_depth_alias(context& ctx)
{
	surface_typeless_transfer typeless(ctx);
	surface_depth_transfer depths(ctx);
	const auto value = texture(ctx, surface_depth::z24s8_uint, 8, 6);
	texture_region whole;
	whole.level = 1;
	whole.size = MTLSizeMake(8, 6, 1);
	std::vector<std::byte> original(8 * 6 * 4);
	for (std::size_t i = 0; i < original.size(); i += 4)
	{
		original[i] = std::byte{0x80};
		original[i + 1] = original[i + 2] = std::byte{0};
		original[i + 3] = static_cast<std::byte>(i / 4 * 7);
	}
	const auto input = make_upload(ctx, original), depth = make_buffer(ctx, 8 * 6 * 4, buffer_storage::shared), stencil = make_buffer(ctx, 8 * 6, buffer_storage::shared);
	ctx.begin();
	depths.upload(ctx, surface_depth::z24s8_uint, input, {}, value, whole);
	texture_region sr = whole, dr = whole;
	sr.origin = MTLOriginMake(1, 1, 0);
	sr.size = MTLSizeMake(3, 2, 1);
	dr.origin = MTLOriginMake(2, 2, 0);
	dr.size = MTLSizeMake(5, 3, 1);
	typeless.nearest(ctx, surface_depth::z24s8_uint, value, sr, surface_depth::z24s8_float, value, dr, {true, false, false});
	texture_to_buffer(ctx, value, whole, depth, 0, 32, 0, texture_plane::depth);
	texture_to_buffer(ctx, value, whole, stencil, 0, 8, 0, texture_plane::stencil);
	ctx.submit().wait();
	const auto actual = map_buffer(depth, 0, depth.length), sa = map_buffer(stencil, 0, stencil.length);
	for (std::size_t y = 0; y < 6; ++y)
		for (std::size_t x = 0; x < 8; ++x)
		{
			const bool copied = x >= 2 && x < 7 && y >= 2 && y < 5;
			const auto expected = copied ? 0x40000000u : 0x3f000001u;
			std::uint32_t raw;
			std::copy_n(actual.begin() + (y * 8 + x) * 4, 4, reinterpret_cast<std::byte*>(&raw));
			require(raw == expected, "Equal-native Z24U->Z24F used a raw format shortcut or changed exterior depth");
			const auto index = copied ? (1 + coordinate(y - 2, 2, 3, false)) * 8 + 1 + coordinate(x - 2, 3, 5, true) : y * 8 + x;
			require(sa[y * 8 + x] == static_cast<std::byte>(index * 7), "Aliased typeless depth transfer lost source stencil snapshot");
		}
	std::cout << "Equal-native Z24U/Z24F alias reinterprets packed guest bits instead of copying float depth, preserving exterior depth/stencil\n";
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
			matrix(ctx);
			same_native_depth_alias(ctx);
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
