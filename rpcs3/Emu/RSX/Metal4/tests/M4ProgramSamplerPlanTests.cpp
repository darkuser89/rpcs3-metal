#include "M4ProgramSamplerPlan.h"

#include <iostream>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native program sampler metadata differs");
}
} // namespace
int main()
{
	try
	{
		unsigned profiles = 0, rejected = 0;
		for (unsigned dimension : {1u, 2u, 3u, 6u})
			for (unsigned depth = 0; depth < 5; ++depth)
				for (unsigned samples : {1u, 2u, 4u, 8u})
					for (bool clamp : {false, true})
					{
						metal4::fragment_texture_registers words;
						words.control0 = 1u << 31;
						words.format = dimension == 6 ? 0x24 : dimension << 4;
						metal4::texture_cache_state cache;
						cache.source_depth_format = depth;
						cache.depth = depth != 0;
						cache.depth_float = depth >= 3;
						cache.samples = samples;
						cache.coordinates.scale = {.125f, .25f, .5f};
						cache.coordinates.bias = {.1f, .2f, .3f};
						cache.coordinates.minimum = {.15f, .25f};
						cache.coordinates.maximum = {.75f, .85f};
						cache.coordinates.flags = clamp ? metal4::texture_coordinate_clamp : 0;
						const auto result = metal4::program_sampler_plan::capture(words, cache);
						constexpr unsigned expected_class[]{1, 2, 8, 4, 16};
						require(result.dimension == (dimension == 1 ? 0 : dimension == 2 ? 1
																	  : dimension == 3	 ? 3
																						 : 2) &&
								result.format_class == expected_class[depth] && result.samples == samples &&
								result.coordinates.scale == cache.coordinates.scale && result.coordinates.bias == cache.coordinates.bias &&
								result.coordinates.minimum == cache.coordinates.minimum && result.coordinates.maximum == cache.coordinates.maximum &&
								result.coordinates.flags == cache.coordinates.flags);
						cache.coordinates = {};
						require(result.coordinates.scale[0] == .125f && result.coordinates.bias[0] == .1f);
						++profiles;
					}
		const auto reject = [&](auto mutate)
		{
			metal4::fragment_texture_registers words;
			words.control0 = 1u << 31;
			words.format = 0x20;
			metal4::texture_cache_state cache;
			mutate(words, cache);
			bool caught = false;
			try
			{
				(void)metal4::program_sampler_plan::capture(words, cache);
			}
			catch (const std::invalid_argument&)
			{
				caught = true;
			}
			require(caught);
			++rejected;
		};
		metal4::fragment_texture_registers disabled;
		disabled.offset = disabled.format = disabled.address = disabled.control1 = 0xffffffffu;
		metal4::texture_cache_state invalid_unused;
		invalid_unused.samples = invalid_unused.actual_mips = invalid_unused.source_depth_format = 0xffffffffu;
		const auto empty = metal4::program_sampler_plan::capture(disabled, invalid_unused);
		require(empty.dimension == 1 && empty.format_class == 0 && empty.samples == 1 &&
				empty.coordinates.scale == std::array<float, 3>{1, 1, 1});
		std::cout << "Disabled program sampler ignores invalid unused registers/cache and matches empty RSX descriptor passed\n";
		for (unsigned dimension : {0u, 4u, 15u})
			reject([&](auto& w, auto&)
				   { w.format = dimension << 4; });
		for (unsigned dimension : {1u, 3u})
			reject([&](auto& w, auto&)
				   { w.format = (dimension << 4) | 4; });
		for (unsigned samples : {0u, 3u, 256u})
			reject([&](auto&, auto& c)
				   { c.samples = samples; });
		reject([](auto&, auto& c)
			   { c.actual_mips = 0; });
		reject([](auto&, auto& c)
			   { c.source_depth_format = 5; });
		reject([](auto&, auto& c)
			   { c.depth = true; });
		reject([](auto&, auto& c)
			   { c.source_depth_format = 1; });
		reject([](auto&, auto& c)
			   { c.depth_float = true; });
		reject([](auto&, auto& c)
			   { c.depth = true; c.source_depth_format = 3; });
		reject([](auto&, auto& c)
			   { c.host_sign_mask = 1; });
		reject([](auto&, auto& c)
			   { c.host_gamma_mask = 1; });
		reject([](auto&, auto& c)
			   { c.coordinates.flags = 2; });
		reject([](auto&, auto& c)
			   { c.coordinates.bias[1] = std::numeric_limits<float>::quiet_NaN(); });
		reject([](auto&, auto& c)
			   { c.coordinates.minimum[0] = 2; c.coordinates.maximum[0] = 1; });
		std::cout << profiles << " program sampler profiles and " << rejected << " invalid metadata cases passed;owned transforms preserve pre-nearest bias\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
