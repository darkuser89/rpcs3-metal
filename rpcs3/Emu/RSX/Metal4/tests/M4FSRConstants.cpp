#include "M4FSRPlan.h"
#define A_CPU 1
#include "3rdparty/GPUOpen/include/ffx_a.h"
#include "3rdparty/GPUOpen/include/ffx_fsr1.h"
#undef A_CPU

namespace metal4::test_fsr
{
std::array<std::array<std::uint32_t,4>,4> amd_constants(frame_extent input,frame_rectangle crop,frame_extent output)
{
	std::array<std::array<std::uint32_t,4>,4> result{};
	FsrEasuConOffset(result[0].data(),result[1].data(),result[2].data(),result[3].data(),
		float(crop.width),float(crop.height),float(input.width),float(input.height),float(output.width),float(output.height),float(crop.x),float(crop.y));
	return result;
}
float amd_sharpness(float stops)
{
	std::array<std::uint32_t,4> result{};FsrRcasCon(result.data(),stops);return std::bit_cast<float>(result[0]);
}
} // namespace metal4::test_fsr
