#pragma once

#include "M4GuestSurface.h"

namespace metal4
{
// A recovered transfer must cover this fault and fit the guest address space.
// This validates extent only;writer/protection transactions and mapping leases
// remain separate contracts. DMA metadata must come from the faulting job.
inline void validate_fault_transfer_range(guest_surface_range range, std::uint32_t fault)
{
	validate_guest_surface_range(range);
	if (!range.size || fault < range.address || std::uint64_t(fault) - range.address >= range.size)
		throw std::invalid_argument("Native DMA fault is outside its transfer byte extent");
}
} // namespace metal4
