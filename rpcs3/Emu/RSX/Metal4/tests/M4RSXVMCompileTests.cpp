#include "M4RSXVMAccess.h"
#include "stdafx.h"

// Actual CPU suspend/mapping/reservation/address APIs compile-instantiated;
// never initializes VM,constructs a renderer or starts the application.
std::shared_ptr<metal4::guest_surface_memory> make_real_vm_adapter_checked(
	std::function<std::shared_ptr<void>()> non_cpu_guard)
{
	return std::make_shared<metal4::guarded_guest_memory>(
		std::make_shared<metal4::rsx_vm_access>(std::move(non_cpu_guard)));
}
std::uint32_t real_vm_address_checked(std::uint32_t offset, unsigned location, std::size_t size)
{
	return metal4::translate_rsx_vm_range(offset, location, size);
}

// Compile the mapping-commit callback contract against the real provider's
// view type. No-op only;never calls vm::map/unmap or starts guest CPU state.
bool real_vm_mapping_contract_checked(metal4::guarded_guest_memory& memory, metal4::guest_surface_range range)
{
	const auto no_change = [](metal4::exclusive_guest_memory::view&) noexcept { return false; };
	return memory.mutate_mapping(range, no_change);
}
void real_vm_nested_capture_checked(metal4::rsx_vm_access& access, metal4::guest_surface_memory& memory, const metal4::guest_surface_layout& coverage)
{
	access.capture_cpu([&]
	{
		(void)memory.read_surface(coverage);
		(void)memory.surface_revision(coverage);
	});
}
