#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace metal4
{
// Raw IEEE color bits plus the same evaluator fault as the reference vertex
// path. MSL uses uint4[4], uint fault and three scalar padding words; bitcasts
// avoid normalization/conversion at the intermediate memory boundary.
struct alignas(16) flat_color_result
{
	std::array<std::array<std::uint32_t,4>,4> colors{};
	std::uint32_t fault = 0;
	std::array<std::uint32_t,3> padding{};
};
static_assert(sizeof(flat_color_result)==80 && offsetof(flat_color_result,fault)==64);
[[nodiscard]] constexpr std::optional<std::size_t> flat_color_bytes(std::size_t count, std::size_t byte_limit)
{
	if (count > std::numeric_limits<std::uint32_t>::max() || count > byte_limit / (sizeof(flat_color_result)+4) ||
		count > std::numeric_limits<std::size_t>::max() / sizeof(flat_color_result)) return std::nullopt;
	return count * sizeof(flat_color_result);
}
} // namespace metal4
