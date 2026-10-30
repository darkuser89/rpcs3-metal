#pragma once

#include "M4TextureState.h"

namespace metal4
{
// Metadata for rsx::thread program analysis,not a native sampler binding.
// Coordinates are the cache transform BEFORE the register nearest/UN bias.
struct program_sampler_plan
{
	std::uint32_t dimension = 1, format_class = 1, samples = 1;
	texture_coordinates coordinates;

	[[nodiscard]] static program_sampler_plan capture(const fragment_texture_registers& words, const texture_cache_state& cache)
	{
		if (!(words.control0 >> 31))
		{
			program_sampler_plan empty;
			empty.format_class = 0; // Actual RSX descriptor default:undefined storage.
			return empty;
		}
		const auto dimension = (words.format >> 4) & 15;
		const bool cube = words.format & 4;
		if (dimension < 1 || dimension > 3 || (cube && dimension != 2))
			throw std::invalid_argument("Invalid program sampler dimension");
		if (!cache.actual_mips || (cache.samples != 1 && cache.samples != 2 && cache.samples != 4 && cache.samples != 8))
			throw std::invalid_argument("Invalid program sampler mip/sample metadata");
		if (cache.source_depth_format > 4 || cache.depth != bool(cache.source_depth_format) ||
			cache.depth_float != (cache.source_depth_format == 3 || cache.source_depth_format == 4))
			throw std::invalid_argument("Program sampler source depth codec is absent or inconsistent");
		// Current guest cache publishes raw UNORM/float images. Per-channel
		// host conversions need a matching RSX format_ex contract first.
		if (cache.host_sign_mask || cache.host_gamma_mask)
			throw std::invalid_argument("Program sampler host conversion descriptor is not implemented");
		validate_texture_coordinates(cache.coordinates);
		program_sampler_plan value;
		value.dimension = dimension == 1 ? 0 : dimension == 3 ? 3
										   : cube			  ? 2
															  : 1;
		constexpr std::array<std::uint32_t, 5> classes{1, 2, 8, 4, 16};
		value.format_class = classes[cache.source_depth_format];
		value.samples = cache.samples;
		value.coordinates = cache.coordinates;
		return value;
	}
};
} // namespace metal4
