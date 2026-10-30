#include "M4NearestLayout.h"
#include "M4TransferKernels.h"

#include <fstream>
#include <iostream>

using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
int main(int argc, char** argv)
{
	try
	{
		std::size_t cases = 0;
		for (const unsigned bytes : {1u, 2u, 4u, 8u, 16u})
			for (std::uint32_t width = 1; width <= 7; ++width)
				for (std::uint32_t height = 1; height <= 5; ++height)
					for (std::uint32_t depth = 1; depth <= 3; ++depth)
					{
						nearest_image a{{width, height, depth}, (width + 2) * bytes, (height + 3) * (width + 2) * bytes};
						nearest_image b{{height, depth, width}, (height + 4) * bytes, (depth + 1) * (height + 4) * bytes};
						const auto plan = plan_nearest(a, b, bytes);
						std::size_t count = 0, end = 0;
						for (std::size_t z = 0; z < b.size[2]; ++z)
							for (std::size_t y = 0; y < b.size[1]; ++y)
								for (std::size_t x = 0; x < b.size[0]; ++x)
								{
									end = z * b.image_pitch + y * b.row_pitch + (x + 1) * bytes;
									require(end <= plan.destination.allocation_size, "Nearest plan omitted an active texel");
									++count;
								}
						require(count == plan.texels && plan.destination.allocation_size - end == b.row_pitch - b.size[0] * bytes,
								"Nearest plan includes unused final image padding or wrong texel count");
						++cases;
					}
		const nearest_image one{{1, 1, 1}};
		const auto rejected = [&](nearest_image a, nearest_image b, unsigned bytes)
		{
			bool failed = false;
			try
			{
				(void)plan_nearest(a, b, bytes);
			}
			catch (const std::exception&)
			{
				failed = true;
			}
			require(failed, "Nearest plan accepted zero/partial/overflowing storage");
		};
		for (const unsigned bytes : {0u, 3u, 6u, 32u})
			rejected(one, one, bytes);
		rejected({{0, 1, 1}}, one, 1);
		rejected(one, {{1, 0, 1}}, 1);
		rejected(one, {{1, 1, 0}}, 1);
		rejected({{7, 2, 1}, 6, 0}, one, 1);
		rejected(one, {{2, 2, 2}, 8, 8}, 4);
		rejected(one, {{0xffffffffu, 0xffffffffu, 0xffffffffu}}, 16);
		rejected({{1, 2, 2}, std::numeric_limits<std::size_t>::max(), 0}, one, 1);
		const auto large = plan_nearest(one, {{0xffffffffu, 1, 1}}, 16);
		require(large.texels == 0xffffffffu && large.destination.allocation_size == std::uint64_t(0xffffffffu) * 16,
				"Nearest uint32 extent was truncated to 32-bit byte arithmetic");
		if (argc == 2)
			std::ofstream(argv[1]) << transfer_kernels;
		std::cout << cases << " nearest pitched image plans, sparse texel coverage and uint32/size_t overflow boundaries passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
