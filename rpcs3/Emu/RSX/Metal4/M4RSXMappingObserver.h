#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "The native Metal4 VM observer requires Apple Silicon."
#endif

#include "Emu/Memory/vm_locking.h"
#include "Emu/Memory/vm_mapping_observer.h"
#include "M4GuardedGuestMemory.h"
#include "M4RSXPageProtection.h"

namespace metal4
{
// Install before exposing a real VM memory adapter to producers. Teardown
// runs outside provider/VM capture; writer exclusion drains callbacks before
// releasing their memory owner. This does not install CPU/DMA store hooks.
class rsx_mapping_observer final : public vm::mapping_change_observer
{
	const std::shared_ptr<guarded_guest_memory> m_memory;
	const std::shared_ptr<rsx_page_protection> m_protection;

  public:
	explicit rsx_mapping_observer(std::shared_ptr<guarded_guest_memory> memory,
		std::shared_ptr<rsx_page_protection> protection = {}) : m_memory(std::move(memory)), m_protection(std::move(protection))
	{
		if (!m_memory) throw std::invalid_argument("Native VM observer requires guarded memory");
		const vm::writer_lock lock;
		if (!vm::attach_mapping_observer(*this, lock)) throw std::logic_error("Native VM mapping observer is already installed");
		if (m_protection && !m_protection->bind(lock))
		{
			if (!vm::detach_mapping_observer(*this, lock)) std::terminate();
			throw std::logic_error("Native page protection already belongs to another observer");
		}
	}
	~rsx_mapping_observer()
	{
		const vm::writer_lock lock;
		if (m_protection) m_protection->shutdown(lock);
		if (!vm::detach_mapping_observer(*this, lock)) std::terminate();
	}
	rsx_mapping_observer(const rsx_mapping_observer&) = delete;
	rsx_mapping_observer& operator=(const rsx_mapping_observer&) = delete;
	[[nodiscard]] bool observes(const std::shared_ptr<guarded_guest_memory>& memory) const noexcept { return m_memory == memory; }
	void on_mapping_change_begin(std::uint32_t address, std::uint32_t size) noexcept override
	{
		m_memory->begin_mapping_change({address, size});
	}
	void on_mapping_change_end(std::uint32_t address, std::uint32_t size) noexcept override
	{
		m_memory->end_mapping_change({address, size});
	}
	void on_mapping_change_end_locked(const vm::mapping_change_view& view, std::uint32_t address, std::uint32_t size) noexcept override
	{
		if (m_protection) m_protection->mapping_changed(view, address, size);
		on_mapping_change_end(address, size);
	}
};
} // namespace metal4
