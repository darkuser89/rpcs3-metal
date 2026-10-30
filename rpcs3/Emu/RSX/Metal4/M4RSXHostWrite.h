#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "The native Metal4 DMA writer bridge requires Apple Silicon."
#endif

#include "Emu/RSX/RSXOffload.h"
#include "Emu/Memory/vm_locking.h"
#include "M4PreparedHostTransfer.h"
#include "M4HostStoreTransaction.h"

namespace metal4
{
struct rsx_host_access
{
	std::optional<guest_surface_range> destination, source;
	[[nodiscard]] explicit operator bool() const noexcept { return destination || source; }
};
[[nodiscard]] inline rsx_host_access describe_rsx_host_access(void* destination, const void* source, std::uint64_t bytes)
{
	const auto base = reinterpret_cast<std::uintptr_t>(vm::g_base_addr);
	return {host_write_journal::destination(base, reinterpret_cast<std::uintptr_t>(destination), bytes),
		host_write_journal::destination(base, reinterpret_cast<std::uintptr_t>(source), bytes)};
}

// The caller owns a live render dispatcher. Preparation runs on that owner
// under the actual DMA producer's paused capture context. Resume the writer
// before arming an instruction, then reacquire nonwaiting mapping exclusion
// only after the raw-source reservation has been acquired by the offloader.
template <typename Dispatch, typename Prepare>
[[nodiscard]] std::optional<host_write_intent> stage_rsx_host_write(rsx::dma_manager& manager,
	std::shared_ptr<guarded_guest_memory> memory, const rsx_host_access& access, bool owner_thread,
	Dispatch&& dispatch, Prepare&& prepare)
{
	if (!access) return {};
	if (!memory) throw std::invalid_argument("Native DMA writer requires guarded memory");
	const bool offloader = manager.is_current_thread();
	if (offloader) manager.set_mem_fault_flag();
	struct recovery
	{
		rsx::dma_manager* manager;
		~recovery() { if (manager) manager->clear_mem_fault_flag(); }
	} cleanup{offloader ? &manager : nullptr};
	prepared_host_transfer prepared;
	{
		auto paused = manager.pause_native_writer_fault();
		if (!paused) throw std::logic_error("Native DMA preparation lost its continuous writer ticket");
		if (!owner_thread) vm::temporary_unlock();
		std::invoke(std::forward<Dispatch>(dispatch), [&]
		{
			auto capture = paused->capture_context();
			if (!capture) throw std::logic_error("Native DMA producer resumed before owner preparation");
			prepared = std::invoke(std::forward<Prepare>(prepare), access);
		});
	}
	return host_store_transaction::prepare(std::move(memory), std::move(prepared.destination), std::move(prepared.source), []() -> std::shared_ptr<void>
	{
		auto guard = vm::host_mapping_guard::try_acquire();
		if (!guard) return {};
		return std::make_shared<vm::host_mapping_guard>(std::move(guard));
	});
}
} // namespace metal4
