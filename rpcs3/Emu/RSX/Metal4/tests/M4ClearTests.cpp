#include "M4ClearPlan.h"
#include "M4ClearShaders.h"

#include <fstream>
#include <iostream>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("RSX clear plan invariant failed");
}
} // namespace
int main(int argc, char** argv)
{
	using namespace metal4;
	try
	{
		std::size_t models = 0;
		for (unsigned raw = 0; raw < 65536; ++raw)
		{
			for (const auto format : {surface_color::rgb565, surface_color::x1rgb5_zero, surface_color::x1rgb5_one})
			{
				const auto p = decode_color_clear(format, raw, 0xf0);
				const unsigned red = format == surface_color::rgb565 ? raw / 2048 % 32 : raw / 1024 % 32;
				const unsigned green = format == surface_color::rgb565 ? raw / 32 % 64 : raw / 32 % 32;
				const auto red_bits = static_cast<unsigned>(p.value[0] * 31 + 0.5f);
				const auto green_bits = static_cast<unsigned>(p.value[1] * (format == surface_color::rgb565 ? 63 : 31) + 0.5f);
				const auto blue_bits = static_cast<unsigned>(p.value[2] * 31 + 0.5f);
				require(red_bits == red && green_bits == green && blue_bits == raw % 32);
				require(p.value[3] == (format == surface_color::x1rgb5_zero ? 0.f : 1.f));
				++models;
			}
		}
		for (unsigned mask = 0; mask < 16; ++mask)
			for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			{
				const auto p = decode_color_clear(static_cast<surface_color>(raw), 0x173b7dc1, mask * 16);
				const auto expected = raw == 11 || raw == 12 || raw == 13 ? 0u : raw == 3 ? mask % 8
																			 : raw == 9	  ? mask / 4 % 2
																			 : raw == 10  ? (mask / 4 % 2) + (mask / 2 % 2) * 2
																			 : raw >= 14  ? (mask % 2) * 4 + (mask / 2 % 2) * 2 + (mask / 4 % 2) + (mask / 8 % 2) * 8
																						  : mask;
				require(p.write_mask == expected);
				if (raw == 4 || raw == 5 || raw == 8)
					require(p.value == std::array<float, 4>{59.f / 255, 125.f / 255, 193.f / 255, 23.f / 255});
				if (raw >= 14)
					require(p.value == std::array<float, 4>{193.f / 255, 125.f / 255, 59.f / 255, 23.f / 255});
			}
		std::size_t rectangles = 0;
		for (std::size_t fw = 1; fw <= 8; ++fw)
			for (std::size_t fh = 1; fh <= 6; ++fh)
				for (std::size_t x = 0; x <= 9; ++x)
					for (std::size_t y = 0; y <= 7; ++y)
						for (std::size_t w = 0; w <= 10; ++w)
							for (std::size_t h = 0; h <= 8; ++h)
							{
								const auto r = clip_clear_rectangle({x, y, w, h}, fw, fh);
								for (std::size_t py = 0; py < fh; ++py)
									for (std::size_t px = 0; px < fw; ++px)
									{
										const bool expected = px >= x && py >= y && px - x < w && py - y < h;
										const bool actual = px >= r.x && py >= r.y && px - r.x < r.width && py - r.y < r.height;
										require(actual == expected);
									}
								++rectangles;
							}
		const auto maximum = std::numeric_limits<std::size_t>::max();
		const auto r = clip_clear_rectangle({maximum, maximum, maximum, maximum}, 16, 16);
		require(r.x == 16 && r.y == 16 && r.width == 0 && r.height == 0);
		require(decode_color_clear(surface_color::argb8, 0x12345678, 0xfffffff3).write_mask == 15);
		if (argc == 2)
		{
			std::ofstream out(argv[1]);
			out << clear_shaders;
			if (!out)
				throw std::runtime_error("Cannot export clear shaders");
		}
		std::cout << models << " packed clear fixtures and " << rectangles << " independent clipped rectangle models passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
