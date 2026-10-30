#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__) || defined(__arm64e__)
#error "Native instruction capture requires Darwin arm64."
#endif

#include <cstdint>
#include <limits>
#include <optional>
#include <mach/mach.h>
#include <mach/mach_vm.h>

namespace metal4
{
// Render-owner fetch,never a signal-handler load from an arbitrary PC. The
// kernel validates the complete copy after executable/readable-region checking.
// Caller must additionally retain immutable code identity throughout the
// original fault and completion;this read alone cannot exclude JIT patching.
[[nodiscard]] inline std::optional<std::uint32_t> capture_darwin_instruction(std::uint64_t pc) noexcept
{
	if (!pc || (pc & 3) || pc > std::numeric_limits<std::uint64_t>::max() - 4) return {};
	mach_vm_address_t address = pc;
	mach_vm_size_t size = 0;
	vm_region_basic_info_data_64_t info{};
	mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
	mach_port_t object = MACH_PORT_NULL;
	const auto status = ::mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
		reinterpret_cast<vm_region_info_t>(&info), &count, &object);
	if (object != MACH_PORT_NULL) (void)::mach_port_deallocate(mach_task_self(), object);
	if (status != KERN_SUCCESS || address > pc || pc - address > size || size - (pc - address) < 4 ||
		(info.protection & (VM_PROT_READ | VM_PROT_EXECUTE)) != (VM_PROT_READ | VM_PROT_EXECUTE)) return {};
	std::uint32_t result = 0;
	mach_vm_size_t copied = 0;
	if (::mach_vm_read_overwrite(mach_task_self(), pc, sizeof(result), reinterpret_cast<mach_vm_address_t>(&result), &copied) != KERN_SUCCESS || copied != sizeof(result)) return {};
	return result;
}
} // namespace metal4
