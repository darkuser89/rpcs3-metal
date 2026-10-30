#pragma once

#include "M4GuestCoverage.h"
#include <algorithm>

namespace metal4
{
// Same render-owner CPU capture as the rest of fault selection. Original
// registrations still select mirrors after remap;other aliases require exact
// active physical metadata. An unsupported provider cannot imply no overlap.
[[nodiscard]] inline bool memory_physical_fault_overlap(guest_surface_memory& memory, const physical_identity& identity,
	std::span<const std::shared_ptr<void>> registrations, const guest_surface_layout& coverage)
{
	if (coverage.registration && std::any_of(registrations.begin(), registrations.end(), [&](const auto& owner)
		{ return owner && !owner.owner_before(coverage.registration) && !coverage.registration.owner_before(owner); })) return true;
	const auto matches = memory.physical_identity_overlaps_current(identity, coverage);
	if (!matches) throw std::runtime_error("Native fault selection lacks active physical metadata");
	return *matches;
}
// The caller holds one storage_test identity epoch throughout this traversal.
// Physical identity order need not follow virtual Morton/row address order.
[[nodiscard]] inline bool active_storage_overlap(bool direct, const guest_surface_memory::storage_lookup& aliases,
	const guest_surface_layout& a, const guest_surface_layout& b)
{
	if (direct)
		return overlaps(a, b);
	guest_surface_byte_cursor ac(a);
	(void)guest_surface_byte_cursor(b);
	if (!aliases(a.range, b.range))
		return false;
	while (const auto ar = ac.next())
	{
		guest_surface_byte_cursor bc(b);
		while (const auto br = bc.next())
			if (aliases(*ar, *br))
				return true;
	}
	return false;
}
[[nodiscard]] inline bool memory_storage_overlap(guest_surface_memory& memory, const guest_surface_layout& a, const guest_surface_layout& b)
{
	return memory.test_storage([&](bool direct, const guest_surface_memory::storage_lookup& aliases)
		{ return active_storage_overlap(direct, aliases, a, b); });
}
[[nodiscard]] inline bool memory_storage_overlap(guest_surface_memory& memory, guest_surface_range a, const guest_surface_layout& b)
{
	return memory_storage_overlap(memory, guest_byte_coverage(a), b);
}
[[nodiscard]] inline bool memory_storage_overlap(guest_surface_memory& memory, guest_surface_range a, guest_surface_range b)
{
	validate_guest_surface_range(a);
	validate_guest_surface_range(b);
	return memory.test_storage([&](bool, const guest_surface_memory::storage_lookup& aliases) { return aliases(a, b); });
}
} // namespace metal4
