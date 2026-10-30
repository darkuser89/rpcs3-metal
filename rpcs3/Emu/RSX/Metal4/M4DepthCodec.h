#pragma once

#include <cstdint>

namespace metal4
{

// Preserve VKCompute.h's RSX E4M12 bit contract, including truncation and
// unsigned wrap for out-of-domain inputs. This is not IEEE binary16, and does
// not apply normalized-depth clamping or reinterpret a Metal Depth16Unorm.
constexpr std::uint32_t expand_e4m12_bits(std::uint16_t value)
{
	return (static_cast<std::uint32_t>(value) << 11) + (120u << 23);
}

constexpr std::uint16_t contract_e4m12_bits(std::uint32_t value)
{
	return static_cast<std::uint16_t>((value - (120u << 23)) >> 11);
}

}
