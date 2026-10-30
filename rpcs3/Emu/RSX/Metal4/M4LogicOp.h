#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

namespace metal4
{
struct logic_rop_configuration
{
	bool enabled = false;
	std::uint32_t operation = 0x1503;
	std::uint32_t sample_count = 1;
	// Native logical RGBA normalized channel widths. Zero denotes an absent
	// channel or a floating-point attachment, where VK logic ops pass through.
	std::array<std::array<std::uint32_t, 4>, 4> bits{};
	bool operator==(const logic_rop_configuration&) const = default;
};
inline void validate_logic_rop(const logic_rop_configuration& configuration)
{
	if (configuration.operation < 0x1500 || configuration.operation > 0x150f)
		throw std::invalid_argument("Invalid RSX shader logic operation");
	if (configuration.sample_count != 1 && configuration.sample_count != 2 && configuration.sample_count != 4 && configuration.sample_count != 8)
		throw std::invalid_argument("Invalid native logic ROP sample count");
	for (const auto& target : configuration.bits)
		for (const auto width : target)
			if (width > 16)
				throw std::invalid_argument("Normalized shader logic channel exceeds sixteen bits");
}
[[nodiscard]] inline std::uint32_t apply_logic_operation(std::uint32_t operation, std::uint32_t source, std::uint32_t destination)
{
	switch (operation)
	{
	case 0x1500:
		return 0;
	case 0x1501:
		return source & destination;
	case 0x1502:
		return source & ~destination;
	case 0x1503:
		return source;
	case 0x1504:
		return ~source & destination;
	case 0x1505:
		return destination;
	case 0x1506:
		return source ^ destination;
	case 0x1507:
		return source | destination;
	case 0x1508:
		return ~(source | destination);
	case 0x1509:
		return ~(source ^ destination);
	case 0x150a:
		return ~destination;
	case 0x150b:
		return source | ~destination;
	case 0x150c:
		return ~source;
	case 0x150d:
		return ~source | destination;
	case 0x150e:
		return ~(source & destination);
	case 0x150f:
		return 0xffffffffu;
	default:
		throw std::invalid_argument("Invalid RSX shader logic operation");
	}
}
} // namespace metal4
