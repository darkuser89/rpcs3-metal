#include "M4GuestCoverage.h"

#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
std::size_t reference(std::array<std::size_t, 3> position, std::array<std::size_t, 3> dimensions)
{
	std::size_t output = 0, stride = 1;
	for (std::size_t bit = 1; bit < dimensions[0] || bit < dimensions[1] || bit < dimensions[2]; bit *= 2)
		for (unsigned axis = 0; axis < 3; ++axis)
			if (bit < dimensions[axis])
			{
				if ((position[axis] / bit) % 2)
					output += stride;
				stride *= 2;
			}
	return output;
}
void inspect(const guest_surface_layout& layout, const std::vector<bool>& expected)
{
	std::vector<bool> actual(layout.range.size);
	guest_surface_byte_cursor cursor(layout);
	std::uint64_t previous_end = 0;
	bool first = true;
	while (const auto range = cursor.next())
	{
		require(range->size && range->address >= layout.range.address && std::uint64_t(range->address) + range->size <= std::uint64_t(layout.range.address) + layout.range.size, "Coverage exceeds storage");
		require(first || range->address > previous_end, "Coverage isn't sorted/coalesced/disjoint");
		first = false;
		previous_end = std::uint64_t(range->address) + range->size;
		for (std::size_t byte = 0; byte < range->size; ++byte)
			actual[range->address - layout.range.address + byte] = true;
	}
	require(actual == expected, "Coverage cursor differs from independently enumerated bytes");
}
} // namespace
int main()
{
	using namespace metal4;
	try
	{
		std::size_t models = 0;
		for (const std::size_t sx : {1u, 2u, 4u, 8u})
			for (const std::size_t sy : {1u, 2u, 4u, 8u})
				for (const std::size_t sz : {1u, 2u, 4u})
					for (std::size_t x = 0; x < sx; ++x)
						for (std::size_t y = 0; y < sy; ++y)
							for (std::size_t z = 0; z < sz; ++z)
								for (std::size_t w = 1; w <= sx - x; ++w)
									for (std::size_t h = 1; h <= sy - y; ++h)
										for (const unsigned bpp : {1u, 2u, 4u, 8u, 16u})
										{
											const swizzle_region region{{sx, sy, sz}, {x, y, z}, {w, h, 1}};
											const auto layout = plan_guest_surface(17, w, h, bpp, 0, region);
											std::vector<bool> expected(layout.range.size);
											for (std::size_t py = y; py < y + h; ++py)
												for (std::size_t px = x; px < x + w; ++px)
													for (unsigned byte = 0; byte < bpp; ++byte)
														expected[reference({px, py, z}, {sx, sy, sz}) * bpp + byte] = true;
											inspect(layout, expected);
											++models;
										}
		for (std::size_t w = 1; w < 17; ++w)
			for (std::size_t h = 1; h < 9; ++h)
				for (const unsigned bpp : {1u, 2u, 4u, 8u, 16u})
					for (std::size_t pad = 0; pad < 9; ++pad)
					{
						const auto layout = plan_guest_surface(11, w, h, bpp, (w + pad) * bpp);
						std::vector<bool> expected(layout.range.size);
						for (std::size_t y = 0; y < h; ++y)
							for (std::size_t byte = 0; byte < w * bpp; ++byte)
								expected[y * layout.row_pitch + byte] = true;
						inspect(layout, expected);
						++models;
					}
		for (unsigned bpp : {1u,2u,4u,8u,16u})
			for(unsigned width : {1u,3u,7u})
				for(unsigned padding : {1u,3u,5u})
				{
					const auto pitch=width*bpp+padding;
					const auto layout=plan_guest_surface(13,width,5,bpp,pitch);
					require(layout.row_pitch==pitch && layout.range.size==pitch*5,"Guest pitch was rounded to native texels");
					std::vector<bool> expected(layout.range.size);
					for(unsigned y=0;y<5;++y) for(unsigned b=0;b<width*bpp;++b) expected[y*pitch+b]=true;
					inspect(layout,expected);++models;
				}
		std::size_t pairs = 0;
		for (unsigned a = 0; a < 24; ++a)
			for (unsigned b = 0; b < 24; ++b)
				for (unsigned ap = 8; ap < 20; ap += 4)
					for (unsigned bp = 8; bp < 20; bp += 4)
					{
						const auto aa = plan_guest_surface(a, 2, 3, 4, ap), bb = plan_guest_surface(b, 2, 3, 4, bp);
						bool expected = false;
						for (unsigned byte = 0; byte < 96; ++byte)
						{
							const bool active_a = byte >= a && byte - a < ap * 3 && (byte - a) % ap < 8;
							const bool active_b = byte >= b && byte - b < bp * 3 && (byte - b) % bp < 8;
							expected |= active_a && active_b;
						}
						require(overlaps(aa, bb) == expected, "Pitched active alias differs from byte model");
						++pairs;
					}
		guest_surface_byte_cursor entire(guest_byte_coverage({0, 0x100000000ull}));
		const auto full = entire.next();
		require(full && full->size == 0x100000000ull && !entire.next(), "Contiguous full-VM coverage wasn't one bounded-memory interval");
		const auto huge = plan_guest_surface(0, 32768, 1, 1, 0, swizzle_region{{65536, 65536, 1}, {0, 65535, 0}, {32768, 1, 1}});
		guest_surface_byte_cursor sparse(huge);
		std::size_t bytes = 0, ranges = 0;
		while (const auto range = sparse.next())
		{
			bytes += range->size;
			++ranges;
		}
		require(bytes == 32768 && ranges == 16384, "Sparse coverage inside four-GiB Morton storage differs");
		guest_surface_byte_cursor empty(guest_byte_coverage({0xffffffffu, 0}));
		require(!empty.next(), "Empty byte coverage wasn't empty");
		const auto first = plan_guest_surface(0, 4, 4, 4, 0, swizzle_region{{8, 8, 2}, {1, 1, 0}, {4, 4, 1}});
		const auto second = plan_guest_surface(0, 4, 4, 4, 0, swizzle_region{{8, 8, 2}, {1, 1, 1}, {4, 4, 1}});
		require(!overlaps(first, second) && overlaps(first, first), "Disjoint Morton slices alias");
		std::cout << models << " independently enumerated coverage models, " << pairs << " active alias pairs, disjoint Morton slices and four-GiB sparse bounds passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
