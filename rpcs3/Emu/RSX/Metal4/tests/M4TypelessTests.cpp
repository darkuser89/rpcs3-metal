#include "M4TypelessLayout.h"

#include <array>
#include <iostream>

using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
int main()
{
	try
	{
		const std::array<surface_format, 18> formats{surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565,
													 surface_color::xrgb8_zero, surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8,
													 surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero,
													 surface_color::xbgr8_one, surface_color::abgr8, surface_depth::z16_uint, surface_depth::z24s8_uint,
													 surface_depth::z16_float, surface_depth::z24s8_float};
		std::size_t count = 0;
		for (const auto a : formats)
			for (const auto b : formats)
				for (std::size_t bytes = 16; bytes <= 80; bytes += 16)
				{
					const auto source_width = bytes / surface_guest_bytes(a), target_width = bytes / surface_guest_bytes(b);
					const auto layout = plan_typeless(a, b, source_width, 3, target_width + 1, 5);
					std::size_t source_bytes = 0;
					for (std::size_t y = 0; y < 3; ++y)
						for (std::size_t x = 0; x < source_width; ++x)
							source_bytes += surface_guest_bytes(a);
					require(layout.source_width_as_destination == target_width && layout.source.allocation_size == source_bytes,
							"Typeless source row reinterprets partial words or crosses row boundaries");
					require(layout.destination.bytes_per_row == (target_width + 1) * surface_guest_bytes(b) &&
								layout.destination.allocation_size == 5 * layout.destination.bytes_per_row,
							"Typeless destination footprint differs");
					++count;
				}
		const auto reject = [](surface_format a, surface_format b, std::size_t sw, std::size_t sh, std::size_t dw, std::size_t dh)
		{
			bool failed = false;
			try
			{
				(void)plan_typeless(a, b, sw, sh, dw, dh);
			}
			catch (const std::exception&)
			{
				failed = true;
			}
			require(failed, "Typeless planner accepted invalid/partial/overflowing storage");
		};
		reject(surface_color::b8, surface_color::rgba32f, 15, 1, 1, 1);
		reject(surface_depth::z16_float, surface_color::argb8, 3, 1, 1, 1);
		reject(surface_color::argb8, surface_depth::z16_uint, 0, 1, 1, 1);
		reject(surface_color::argb8, surface_color::b8, 1, 0, 1, 1);
		reject(static_cast<surface_color>(0), surface_color::b8, 1, 1, 1, 1);
		reject(surface_color::b8, static_cast<surface_depth>(0), 1, 1, 1, 1);
		reject(surface_color::rgba32f, surface_color::b8, std::numeric_limits<std::size_t>::max(), 1, 1, 1);
		reject(surface_color::rgba32f, surface_color::b8, std::uint64_t(0x10000000), 1, 1, 1);
		reject(surface_color::b8, surface_color::b8, 1, 1, std::uint64_t(0x100000000), 1);
		std::cout << count << " typeless whole-row interpretations across all 18x18 surface pairs and partial/overflow boundaries passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
