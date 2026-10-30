#include "M4GuestSurface.h"

#include <iostream>

int main()
{
	using namespace metal4;
	try
	{
		std::size_t cases = 0;
		for (std::size_t a = 0; a < 64; ++a)
			for (std::size_t as = 0; as < 32; ++as)
				for (std::size_t b = 0; b < 64; ++b)
					for (std::size_t bs = 0; bs < 32; ++bs)
					{
						bool expected = false;
						for (std::size_t byte = 0; byte < 96; ++byte)
							expected |= byte >= a && byte - a < as && byte >= b && byte - b < bs;
						if (overlaps({static_cast<std::uint32_t>(a), as}, {static_cast<std::uint32_t>(b), bs}) != expected)
							throw std::runtime_error("Guest surface range overlap differs");
						++cases;
					}
		for (std::size_t width = 1; width < 33; ++width)
			for (std::size_t height = 1; height < 17; ++height)
				for (const unsigned bytes : {1u, 2u, 4u, 8u, 16u})
					for (std::size_t padding = 0; padding < 7; ++padding)
					{
						const auto pitch = (width + padding) * bytes;
						const auto layout = plan_guest_surface(123, width, height, bytes, pitch);
						if (layout.range.address != 123 || layout.range.size != pitch * height || layout.row_pitch != pitch)
							throw std::runtime_error("Guest pitched surface footprint differs");
					}
		const auto maximum = std::numeric_limits<std::uint32_t>::max();
		if (plan_guest_surface(maximum, 1, 1, 1).range.size != 1)
			throw std::runtime_error("Last valid PS3 VM byte was rejected");
		bool rejected = false;
		try
		{
			(void)plan_guest_surface(maximum, 2, 1, 1);
		}
		catch (const std::overflow_error&)
		{
			rejected = true;
		}
		if (!rejected)
			throw std::runtime_error("PS3 surface address wrapped");
		const swizzle_region slice{{16, 8, 4}, {3, 2, 1}, {5, 4, 1}};
		const auto layout = plan_guest_surface(0x1000, 5, 4, 4, 0, slice);
		if (layout.range.size != 16 * 8 * 4 * 4 || !layout.morton)
			throw std::runtime_error("Guest Morton surface footprint differs");
		rejected = false;
		try
		{
			(void)plan_guest_surface(0, 5, 4, 4, 32, slice);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		if (!rejected)
			throw std::runtime_error("Morton surface accepted linear pitch");
		if (overlaps({maximum, 1}, {0, 1}) || !overlaps({maximum - 1, 2}, {maximum, 1}))
			throw std::runtime_error("Guest overlap near VM boundary differs");
		std::cout << cases << " independent range-overlap models, pitched/Morton layouts and VM overflow boundaries passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
