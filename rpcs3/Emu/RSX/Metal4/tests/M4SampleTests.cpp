#include "M4SampleLayout.h"
#include "M4SampleShader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("RSX sample layout invariant failed");
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
	catch (const std::out_of_range&)
	{
		return false;
	}
	catch (const std::overflow_error&)
	{
		return false;
	}
}
} // namespace
int main(int argc, char** argv)
{
	using namespace metal4;
	try
	{
		std::size_t models = 0;
		for (const unsigned samples : {2u, 4u})
			for (std::size_t width = 1; width <= 65; ++width)
				for (std::size_t height = 1; height <= 33; ++height)
				{
					const auto expected_width = width * 2, expected_height = height * (samples == 2 ? 1 : 2);
					require(sample_expanded_size(width, height, samples) == std::array<std::size_t, 2>{expected_width, expected_height});
					std::vector<bool> seen(expected_width * expected_height);
					// Independently scan every expanded pixel; alternating X within
					// each row selects samples 0/1 or 2/3 in the second sample row.
					for (std::size_t py = 0; py < expected_height; ++py)
						for (std::size_t px = 0; px < expected_width; ++px)
						{
							const auto x = px / 2, y = samples == 2 ? py : py / 2;
							const auto sample = static_cast<unsigned>(px % 2 + (samples == 2 ? 0 : py % 2 * 2));
							const auto coordinate = expanded_sample_coordinate(x, y, sample, samples);
							require(coordinate == std::array<std::size_t, 2>{px, py});
							const auto index = coordinate[1] * expected_width + coordinate[0];
							require(!seen[index]);
							seen[index] = true;
							++models;
						}
				}
		for (const unsigned samples : {0u, 1u, 3u, 8u, 0xffffffffu})
			require(!accepts([&]
							 {
								 (void)rsx_sample_layout(samples);
							 }));
		require(!accepts([]
						 {
							 (void)sample_expanded_size(0, 1, 2);
						 }));
		require(!accepts([]
						 {
							 (void)sample_expanded_size(1, 0, 4);
						 }));
		require(!accepts([]
						 {
							 (void)sample_expanded_size(std::numeric_limits<std::size_t>::max(), 1, 2);
						 }));
		require(!accepts([]
						 {
							 (void)expanded_sample_coordinate(0, 0, 4, 4);
						 }));
		require(!accepts([]
						 {
							 (void)expanded_sample_coordinate(std::numeric_limits<std::size_t>::max(), 0, 0, 2);
						 }));
		require(!accepts([]
						 {
							 (void)generate_sample_shader(static_cast<sample_aspect>(99), false);
						 }));
		unsigned variants = 0;
		if (argc == 2)
			std::filesystem::create_directories(argv[1]);
		for (const auto aspect : {sample_aspect::color, sample_aspect::depth, sample_aspect::depth_stencil, sample_aspect::stencil})
			for (const bool importing : {false, true})
			{
				const auto source = generate_sample_shader(aspect, importing);
				require((source.find("[[sample_id]]") != std::string::npos) == importing);
				require((source.find("[[stencil]]") != std::string::npos) == (aspect == sample_aspect::depth_stencil || aspect == sample_aspect::stencil));
				if (aspect == sample_aspect::stencil)
					require(source.find("depth2d") == std::string::npos && source.find("validation") == std::string::npos && source.find("[[color(0)]]") == std::string::npos);
				if (argc == 2)
				{
					std::ofstream out(std::filesystem::path(argv[1]) / (std::to_string(variants) + ".metal"));
					out << source;
					if (!out)
						throw std::runtime_error("Cannot export sample shader");
				}
				++variants;
			}
		std::cout << models << " independent expanded sample mappings and " << variants << " native sample shaders checked\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
