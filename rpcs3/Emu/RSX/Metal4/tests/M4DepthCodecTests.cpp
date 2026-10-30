#include "M4DepthCodec.h"

#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
	using namespace metal4;
	for (std::uint32_t value = 0; value < 65536; ++value)
	{
		// Independent numerical model: E4M12 has exponent bias 7, 12 explicit
		// mantissa bits and the same implicit leading one at exponent zero.
		const auto exponent = static_cast<int>(value / 4096) - 7;
		const auto mantissa = 1.0f + static_cast<float>(value % 4096) / 4096.0f;
		const auto expected = std::bit_cast<std::uint32_t>(std::ldexp(mantissa, exponent));
		const auto packed = static_cast<std::uint16_t>(value);
		const auto bits = expand_e4m12_bits(packed);
		if (bits != expected || contract_e4m12_bits(bits) != packed ||
			contract_e4m12_bits(bits + 2047) != packed)
		{
			throw std::runtime_error("RSX E4M12 numerical/round-trip/truncation check failed");
		}
	}
	// Preserve the Vulkan codec's modulo bit behavior outside its depth domain;
	// a future semantic clamp must be a separate, explicitly chosen operation.
	static_assert(expand_e4m12_bits(0) == 0x3c000000u);
	static_assert(contract_e4m12_bits(0) == 0x8000u);
	static_assert(contract_e4m12_bits(0xffffffffu) == 0x7fffu);
	std::cout << "Metal 4 E4M12 codec: all 65536 encodings, inverse and truncation checks passed\n";
}
