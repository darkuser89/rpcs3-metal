#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace metal4
{

inline constexpr std::size_t fragment_register_words = 48 * 4;

struct fragment_register_location
{
	std::uint32_t word;
	std::uint32_t shift;
	bool half;
};

// Rn.xyzw occupies four consecutive uint32 words. H(2n).xyzw occupies
// the low/high halves of Rn.xy; H(2n+1).xyzw occupies those of Rn.zw.
// Raw instructions have six register-index bits, so only H0..H63 can be
// directly encoded even though the physical bank has room for H0..H95.
[[nodiscard]] constexpr fragment_register_location fragment_location(std::uint32_t reg,
	std::uint32_t channel, bool half)
{
	if (channel >= 4 || reg >= (half ? 64u : 48u))
	{
		throw std::out_of_range("RSX fragment register exceeds the physical or encoded register bank");
	}
	return half ? fragment_register_location{reg * 2 + channel / 2, (channel % 2) * 16, true} :
		fragment_register_location{reg * 4 + channel, 0, false};
}

[[nodiscard]] constexpr std::uint32_t fragment_insert_half(std::uint32_t word, std::uint16_t value,
	std::uint32_t shift)
{
	if (shift != 0 && shift != 16) throw std::invalid_argument("RSX half register shift is invalid");
	return (word & ~(0xffffu << shift)) | (std::uint32_t(value) << shift);
}

}
