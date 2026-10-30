#include "M4SwizzleLayout.h"
#include "M4SwizzleKernels.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native Morton layout invariant failed");
}
template <typename Function>
bool accepts(Function function)
{
	try
	{
		function();
		return true;
	}
	catch (const std::invalid_argument&)
	{
		return false;
	}
	catch (const std::overflow_error&)
	{
		return false;
	}
}

// Independent spatial partition model: split the largest remaining axis,
// preferring Z then Y then X on ties. Traverse low child then high child.
// This describes compact Morton order without interleaving coordinate bits.
void partition(std::array<std::size_t, 3> size, std::array<std::size_t, 3> origin,
			   std::vector<std::array<std::size_t, 3>>& order)
{
	if (size == std::array<std::size_t, 3>{1, 1, 1})
	{
		order.push_back(origin);
		return;
	}
	std::size_t axis = 0;
	for (std::size_t i = 1; i < 3; ++i)
		if (size[i] >= size[axis])
			axis = i;
	size[axis] /= 2;
	partition(size, origin, order);
	origin[axis] += size[axis];
	partition(size, origin, order);
}

void addresses()
{
	using namespace metal4;
	std::size_t models = 0;
	for (unsigned w = 0; w < 9; ++w)
		for (unsigned h = 0; h < 8; ++h)
			for (unsigned d = 0; d < 4; ++d)
			{
				swizzle_region region;
				region.storage = {std::size_t{1} << w, std::size_t{1} << h, std::size_t{1} << d};
				region.size = region.storage;
				const auto plan = plan_swizzle(region, 1);
				std::vector<std::array<std::size_t, 3>> order;
				partition(region.storage, {}, order);
				require(order.size() == plan.block_count && plan.storage_bytes == order.size());
				for (std::size_t index = 0; index < order.size(); ++index)
					require(morton_index(order[index], plan.log2) == index);
				models += order.size();
			}
	std::cout << models << " Morton addresses match independent recursive partition model\n";
}

void footprints()
{
	using namespace metal4;
	std::size_t models = 0;
	for (const std::size_t bytes : {1u, 2u, 4u, 8u, 16u})
		for (std::size_t x = 0; x < 8; ++x)
			for (std::size_t y = 0; y < 4; ++y)
				for (std::size_t z = 0; z < 2; ++z)
					for (std::size_t width = 1; width <= 8 - x; ++width)
						for (std::size_t height = 1; height <= 4 - y; ++height)
						{
							const swizzle_region region{{8, 4, 2}, {x, y, z}, {width, height, 1}};
							const auto pitch = (width + 3) * bytes;
							const auto plan = plan_swizzle(region, bytes, pitch);
							std::size_t count = 0;
							for (std::size_t row = 0; row < height; ++row)
								for (std::size_t byte = 0; byte < pitch; ++byte)
									++count;
							require(plan.linear.bytes_per_row == pitch && plan.linear.allocation_size == count &&
									plan.block_count == width * height && plan.storage_bytes == 64 * bytes);
							++models;
						}
	for (const std::size_t bytes : {0u, 3u, 5u, 32u})
		require(!accepts([&]
						 { (void)plan_swizzle({}, bytes); }));
	for (const std::size_t dimension : {0u, 3u, 7u})
	{
		swizzle_region region;
		region.storage[0] = dimension;
		require(!accepts([&]
						 { (void)plan_swizzle(region, 4); }));
	}
	swizzle_region region{{8, 4, 2}, {7, 3, 1}, {2, 1, 1}};
	require(!accepts([&]
					 { (void)plan_swizzle(region, 4); }));
	region.origin = {0, 0, 0};
	region.size = {1, 0, 1};
	require(!accepts([&]
					 { (void)plan_swizzle(region, 4); }));
	region.storage = {std::size_t{1} << 63, 2, 1};
	region.size = {1, 1, 1};
	require(!accepts([&]
					 { (void)plan_swizzle(region, 1); }));
	region.storage = {std::size_t{1} << 63, 1, 1};
	require(plan_swizzle(region, 1).storage_bytes == (std::size_t{1} << 63));
	require(!accepts([&]
					 { (void)plan_swizzle(region, 2); }));
	require(!accepts([]
					 { (void)morton_index({0, 0, 0}, {64, 0, 0}); }));
	require(!accepts([]
					 { (void)morton_index({0, 0, 0}, {32, 32, 0}); }));
	require(!accepts([]
					 { (void)morton_index({2, 0, 0}, {1, 0, 0}); }));
	require(morton_index({(std::size_t{1} << 63) - 1, 0, 0}, {63, 0, 0}) == (std::size_t{1} << 63) - 1);
	region.storage = {8, 4, 2};
	region.size = {8, 4, 2};
	require(!accepts([&]
					 { (void)plan_swizzle(region, 4, 31); }));
	require(!accepts([&]
					 { (void)plan_swizzle(region, 4, 32, 127); }));
	require(!accepts([&]
					 { (void)plan_swizzle(region, 4, std::numeric_limits<std::size_t>::max() - 3); }));
	std::cout << models << " independently enumerated Morton region/pitch footprints passed\n";
}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		addresses();
		footprints();
		if (argc == 2)
		{
			std::ofstream file(argv[1]);
			file << metal4::swizzle_kernels;
			if (!file)
				throw std::runtime_error("Cannot export Morton kernel");
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
