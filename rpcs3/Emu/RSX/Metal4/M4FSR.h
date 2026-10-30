#pragma once

#include "M4FSRPlan.h"
#include "M4Present.h"
#include "M4HeapBudget.h"

namespace metal4
{
struct fsr_statistics
{
	std::uint64_t created = 0, reused = 0, easu_dispatches = 0, rcas_dispatches = 0, bypassed = 0;
	std::size_t working_bytes = 0, working_limit = 0;
};
// Native FSR1 FP32 computation with RGBA16Float working storage. Exactly one
// reusable pair per eye; all use is confined to one context/queue/render owner.
// Every frame recomputes pixels: texture identity alone is not a content key.
class fsr_upscaler final
{
	struct impl;
	std::unique_ptr<impl> m_impl;
  public:
	explicit fsr_upscaler(context& ctx, std::size_t working_limit = 256u << 20, std::shared_ptr<heap_budget> budget = {});
	~fsr_upscaler();
	fsr_upscaler(const fsr_upscaler&) = delete;
	fsr_upscaler& operator=(const fsr_upscaler&) = delete;
	// Encode in an existing recording, outside a render facade/VM epoch. Both
	// dimensions must grow for EASU/RCAS; otherwise returns the native input.
	// Consume the result before encoding the same eye again: working images
	// are reused in queue order rather than representing immutable frame history.
	[[nodiscard]] present_eye encode(context& ctx, const present_eye& input, frame_extent output, float sharpness, unsigned eye = 0);
	bool clear();
	bool clear_eye(unsigned eye);
	[[nodiscard]] fsr_statistics statistics() const;
};
} // namespace metal4
