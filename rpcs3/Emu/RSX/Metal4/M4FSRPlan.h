#pragma once

#include "M4FramePlan.h"
#include <array>
#include <bit>
#include <cstddef>
#include <limits>

namespace metal4
{
struct alignas(16) fsr_parameters
{
	std::array<std::array<std::uint32_t, 4>, 4> easu{};
	std::array<std::uint32_t, 4> rcas{};
	std::array<std::uint32_t, 2> origin{}, extent{};
	std::uint32_t level = 0, width = 0, height = 0, padding = 0;
};
static_assert(sizeof(fsr_parameters) == 112 && offsetof(fsr_parameters, rcas) == 64 &&
	offsetof(fsr_parameters, origin) == 80 && offsetof(fsr_parameters, level) == 96);
// FSR1 EASU/RCAS FP32 constants. Coordinates describe the selected mip/crop,
// not a presumed full-size input. Checked against AMD's original setup in tests.
inline fsr_parameters plan_fsr(frame_extent input, frame_rectangle crop, frame_extent output,
	std::uint32_t level, float sharpness)
{
	if (!input.width || !input.height || !crop.width || !crop.height || !output.width || !output.height ||
		crop.x > input.width || crop.width > input.width - crop.x || crop.y > input.height || crop.height > input.height - crop.y ||
		!std::isfinite(sharpness) || sharpness < 0 || sharpness > 2)
		throw std::invalid_argument("Invalid native FSR input/crop/output/sharpness");
	fsr_parameters result;
	const float iw = 1.f / input.width, ih = 1.f / input.height, ow = 1.f / output.width, oh = 1.f / output.height;
	const auto bits = [](float value) { return std::bit_cast<std::uint32_t>(value); };
	result.easu[0] = {bits(crop.width * ow), bits(crop.height * oh),
		bits(.5f * crop.width * ow - .5f + crop.x), bits(.5f * crop.height * oh - .5f + crop.y)};
	result.easu[1] = {bits(iw), bits(ih), bits(iw), bits(-ih)};
	result.easu[2] = {bits(-iw), bits(2 * ih), bits(iw), bits(2 * ih)};
	result.easu[3] = {bits(0.f * iw), bits(4 * ih), 0, 0};
	result.rcas[0] = bits(std::exp2(-sharpness)); // The FP16 packed constant is unused by this FP32 kernel.
	result.origin = {crop.x, crop.y}; result.extent = {crop.width, crop.height};
	result.level = level; result.width = output.width; result.height = output.height;
	return result;
}
inline bool needs_fsr(frame_extent source, frame_extent output) noexcept
{
	return source.width < output.width && source.height < output.height;
}
inline frame_extent fsr_eye_extent(frame_extent output, stereo_mode mode) noexcept
{
	if (mode == stereo_mode::side_by_side) output.width = output.width / 2 + output.width % 2;
	if (mode == stereo_mode::over_under) output.height = output.height / 2 + output.height % 2;
	return output;
}
inline std::size_t fsr_working_bytes(frame_extent output, std::size_t limit)
{
	// Two RGBA16Float images per eye. Native backing padding is charged by the
	// common heap ledger in addition to this logical working-set admission.
	if (!output.width || !output.height || output.width > limit / 16 || output.height > limit / (std::size_t(output.width) * 16))
		throw std::length_error("Native FSR working set exceeds byte budget");
	return std::size_t(output.width) * output.height * 16;
}
} // namespace metal4
