#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__arm64e__)
#error "Native Metal4 fault context requires the arm64 Darwin ABI."
#endif

#include "M4ARM64Memory.h"
#include <sys/ucontext.h>

namespace metal4
{
// The native build explicitly targets arm64, not arm64e's opaque signed-pointer
// ABI. FP/LR are separate registers in Darwin's saved state, not x[29]/x[30].
// This adapter does not read the instruction pointer or install a signal hook.
[[nodiscard]] inline std::optional<arm64_memory_state> capture_arm64_context(const ucontext_t& context) noexcept
{
	if (!context.uc_mcontext || context.uc_mcsize < sizeof(*context.uc_mcontext)) return {};
	arm64_memory_state result;
	const auto& source = *context.uc_mcontext;
	for (unsigned i = 0; i < 29; ++i) result.x[i] = source.__ss.__x[i];
	result.x[29] = source.__ss.__fp;
	result.x[30] = source.__ss.__lr;
	result.sp = source.__ss.__sp;
	result.pc = source.__ss.__pc;
	static_assert(sizeof(source.__ns.__v) == sizeof(result.v));
	std::memcpy(result.v.data(), source.__ns.__v, sizeof(result.v));
	result.cpsr = source.__ss.__cpsr;
	result.fpsr = source.__ns.__fpsr;
	result.fpcr = source.__ns.__fpcr;
	return result;
}

// Called on the original stopped thread only after successful render-owner
// memory completion. A changed/mismatched saved context refuses atomically.
// Exception state, signal mask, stack metadata and machine padding stay intact.
[[nodiscard]] inline bool publish_arm64_context(ucontext_t& context,
	const arm64_memory_state& before, const arm64_memory_state& after) noexcept
{
	const auto current = capture_arm64_context(context);
	if (!current || *current != before) return false;
	auto& destination = *context.uc_mcontext;
	for (unsigned i = 0; i < 29; ++i) destination.__ss.__x[i] = after.x[i];
	destination.__ss.__fp = after.x[29];
	destination.__ss.__lr = after.x[30];
	destination.__ss.__sp = after.sp;
	destination.__ss.__pc = after.pc;
	std::memcpy(destination.__ns.__v, after.v.data(), sizeof(after.v));
	destination.__ss.__cpsr = after.cpsr;
	destination.__ns.__fpsr = after.fpsr;
	destination.__ns.__fpcr = after.fpcr;
	return true;
}
} // namespace metal4
