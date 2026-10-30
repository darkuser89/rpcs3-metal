#include "M4SurfaceSwizzle.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}

std::size_t spatial_index(std::array<std::size_t, 3> position, std::array<std::size_t, 3> size)
{
	std::size_t index = 0;
	while (size != std::array<std::size_t, 3>{1, 1, 1})
	{
		std::size_t axis = 0;
		for (std::size_t i = 1; i < 3; ++i)
			if (size[i] >= size[axis])
				axis = i;
		size[axis] /= 2;
		if (position[axis] >= size[axis])
		{
			position[axis] -= size[axis];
			index += size[0] * size[1] * size[2];
		}
	}
	return index;
}

void permutation(metal4::context& ctx)
{
	using namespace metal4;
	swizzle_engine engine(ctx);
	const std::array<std::array<std::size_t, 3>, 7> dimensions{{{1, 1, 1}, {1, 16, 1}, {64, 1, 1}, {8, 4, 2}, {4, 16, 8}, {16, 8, 1}, {8, 8, 4}}};
	std::size_t cases = 0;
	for (const auto& storage : dimensions)
		for (const std::size_t bytes : {1u, 2u, 4u, 8u, 16u})
			for (const std::uint32_t swap : {1u, 2u, 4u})
			{
				if (swap > bytes)
					continue;
				swizzle_region region;
				region.storage = storage;
				for (std::size_t axis = 0; axis < 3; ++axis)
				{
					region.origin[axis] = storage[axis] > 2 ? 1 : 0;
					region.size[axis] = storage[axis] - region.origin[axis] - (storage[axis] > 4 ? 1 : 0);
				}
				const auto row = (region.size[0] + 3) * bytes, image = (region.size[1] + 2) * row;
				const auto plan = plan_swizzle(region, bytes, row, image);
				std::vector<std::byte> source_s(3 + plan.storage_bytes + 17), source_l(5 + plan.linear.allocation_size + 19);
				for (std::size_t i = 0; i < source_s.size(); ++i)
					source_s[i] = static_cast<std::byte>(i * 31 + 7);
				for (std::size_t i = 0; i < source_l.size(); ++i)
					source_l[i] = static_cast<std::byte>(i * 17 + 23);
				const auto s = make_upload(ctx, source_s), l = make_upload(ctx, source_l);
				const auto out_s = make_buffer(ctx, source_s.size(), buffer_storage::shared), out_l = make_buffer(ctx, source_l.size(), buffer_storage::shared);
				ctx.begin();
				fill_buffer(ctx, out_s, 0, out_s.length, 0xd3);
				fill_buffer(ctx, out_l, 0, out_l.length, 0xe5);
				engine.convert(ctx, s, 3, out_l, 5, region, bytes, row, image, false, swap);
				engine.convert(ctx, out_s, 3, l, 5, region, bytes, row, image, true, swap);
				ctx.submit().wait();
				std::vector<std::byte> expected_s(source_s.size(), std::byte{0xd3}), expected_l(source_l.size(), std::byte{0xe5});
				for (std::size_t z = 0; z < region.size[2]; ++z)
					for (std::size_t y = 0; y < region.size[1]; ++y)
						for (std::size_t x = 0; x < region.size[0]; ++x)
						{
							const auto a = 3 + spatial_index({x + region.origin[0], y + region.origin[1], z + region.origin[2]}, storage) * bytes;
							const auto b = 5 + z * plan.linear.bytes_per_image + y * row + x * bytes;
							for (std::size_t byte = 0; byte < bytes; ++byte)
							{
								const auto reversed = byte / swap * swap + swap - 1 - byte % swap;
								expected_l[b + byte] = source_s[a + reversed];
								expected_s[a + byte] = source_l[b + reversed];
							}
						}
				const auto actual_s = map_buffer(out_s, 0, out_s.length), actual_l = map_buffer(out_l, 0, out_l.length);
				require(std::equal(actual_s.begin(), actual_s.end(), expected_s.begin()), "Native Morton encode/region/guards differs");
				require(std::equal(actual_l.begin(), actual_l.end(), expected_l.begin()), "Native Morton decode/padding/guards differs");
				++cases;
			}
	std::cout << cases << " native GPU Morton 2D/3D, block/endian, partial-region and guard cases passed\n";
}

void surfaces(metal4::context& ctx)
{
	using namespace metal4;
	surface_swizzle_transfer bridge(ctx);
	const swizzle_region layout{{16, 8, 2}, {3, 2, 1}, {5, 3, 1}};
	texture_region region;
	region.level = 1;
	region.slice = 1;
	region.origin = MTLOriginMake(2, 1, 0);
	region.size = MTLSizeMake(5, 3, 1);
	for (unsigned kind = 0; kind < 2; ++kind)
		for (const unsigned raw : kind == 0 ? std::vector<unsigned>{1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16} : std::vector<unsigned>{1, 2, 3, 4})
		{
			const auto color = kind == 0 ? describe_surface_color(static_cast<surface_color>(raw)) : surface_color_descriptor{};
			const auto depth = kind == 1 ? describe_surface_depth(static_cast<surface_depth>(raw)) : surface_depth_descriptor{};
			const auto bpp = kind == 0 ? color.layout.pixel_bytes : depth.guest_bytes;
			const auto format = kind == 0 ? color.format : depth.format;
			std::vector<std::byte> source(3 + 16 * 8 * 2 * bpp + 13, std::byte{0xa7});
			for (std::size_t z = 0; z < 2; ++z)
				for (std::size_t y = 0; y < 8; ++y)
					for (std::size_t x = 0; x < 16; ++x)
					{
						const auto address = 3 + spatial_index({x, y, z}, layout.storage) * bpp;
						for (unsigned byte = 0; byte < bpp; ++byte)
							source[address + byte] = static_cast<std::byte>((x + y * 13 + z * 23) * 7 + byte * 11);
						if (kind == 1)
						{
							// Valid known depth codes with independently varying stencil.
							const auto word = raw == 1 ? 0x1234u : raw == 3 ? 0x5678u
															   : raw == 2	? 0x34567800u | (unsigned(x + y) % 256)
																			: 0x7d123400u | (unsigned(x + y) % 256);
							for (unsigned byte = 0; byte < bpp; ++byte)
								source[address + byte] = static_cast<std::byte>(word >> ((bpp - 1 - byte) * 8));
						}
					}
			const auto input = make_upload(ctx, source), output = make_buffer(ctx, source.size(), buffer_storage::shared);
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:16 height:16 mipmapped:NO];
			descriptor.textureType = MTLTextureType2DArray;
			descriptor.arrayLength = 2;
			descriptor.mipmapLevelCount = 2;
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
			const auto texture = make_texture(ctx, descriptor);
			ctx.begin();
			fill_buffer(ctx, output, 0, output.length, 0xd3);
			if (kind == 0)
			{
				bridge.upload_color(ctx, static_cast<surface_color>(raw), input, 3, layout, texture, region);
				bridge.download_color(ctx, static_cast<surface_color>(raw), texture, region, output, 3, layout);
			}
			else
			{
				bridge.upload_depth(ctx, static_cast<surface_depth>(raw), input, 3, layout, texture, region);
				bridge.download_depth(ctx, static_cast<surface_depth>(raw), texture, region, output, 3, layout);
			}
			ctx.submit().wait();
			std::vector<std::byte> expected(source.size(), std::byte{0xd3});
			for (std::size_t y = 2; y < 5; ++y)
				for (std::size_t x = 3; x < 8; ++x)
				{
					const auto address = 3 + spatial_index({x, y, 1}, layout.storage) * bpp;
					std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(address), bpp, expected.begin() + static_cast<std::ptrdiff_t>(address));
				}
			const auto actual = map_buffer(output, 0, output.length);
			require(std::equal(actual.begin(), actual.end(), expected.begin()), "Morton surface upload/download or untouched cells differs");
		}
	std::cout << "All fourteen color and four depth surfaces passed Morton->native mip/slice/region->Morton preservation\n";
	ctx.begin();
	const auto buffer = make_buffer(ctx, 64, buffer_storage::shared);
	swizzle_engine engine(ctx);
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
		require(rejected, "Invalid Morton GPU range was accepted");
	};
	rejects([&]
			{ engine.convert(ctx, buffer, 0, buffer, 0, {}, 4); });
	rejects([&]
			{ engine.convert(ctx, buffer, 64, buffer, 0, {}, 4); });
	rejects([&]
			{ engine.convert(ctx, buffer, 0, buffer, 32, {}, 4, 0, 0, false, 3); });
	rejects([&]
			{ engine.convert(ctx, buffer, 0, buffer, 32, {}, 1, 0, 0, false, 2); });
	ctx.cancel();
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
			permutation(ctx);
			surfaces(ctx);
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
