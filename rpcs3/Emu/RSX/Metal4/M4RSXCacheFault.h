#pragma once

#include "M4RSXVMAccess.h"
#include "M4SurfaceCache.h"
#include "M4GuestTexture.h"

namespace metal4
{
class cache_fault_conflict final : public std::runtime_error
{
  public:
	cache_fault_conflict() : std::runtime_error("Native cache fault plan changed before cache preparation completed") {}
};

// Render-owner service. prepare() returns cache preparation only;complete()
// also commits an owned normal ARM instruction under renewed VM exclusion.
// Neither operation permits an unprotected hardware retry. Cache/context
// references must outlive the service and shutdown must drain its callers.
class rsx_cache_fault final
{
	const std::shared_ptr<rsx_vm_access> m_access;
	const std::shared_ptr<guarded_guest_memory> m_memory;
	surface_cache& m_surfaces;
	guest_texture_cache& m_textures;
	const std::thread::id m_owner = std::this_thread::get_id();
	bool m_active = false;
	struct operation_guard
	{
		bool& active;
		explicit operation_guard(bool& value) : active(value)
		{
			if (active) throw std::logic_error("Native cache fault service cannot reenter its current owner task");
			active = true;
		}
		~operation_guard() { active = false; }
	};
  public:
	struct preparation
	{
		std::shared_ptr<const rsx_page_protection::fault_plan> plan;
		std::size_t published_surfaces, invalidated_textures;
		bool writing;
	};
	rsx_cache_fault(context& device, std::shared_ptr<rsx_vm_access> access, std::shared_ptr<guarded_guest_memory> memory,
		surface_cache& surfaces, guest_texture_cache& textures)
		: m_access(std::move(access)), m_memory(std::move(memory)), m_surfaces(surfaces), m_textures(textures)
	{
		if (guest_memory_epoch::active() || !m_access || !m_memory || !m_memory->uses_provider(*m_access))
			throw std::invalid_argument("Native cache fault service requires its actual provider outside VM exclusion");
		m_surfaces.validate_texture_adapter(device, *m_memory);
		m_textures.validate_surface_adapter(device, m_surfaces);
	}
	rsx_cache_fault(const rsx_cache_fault&) = delete;
	rsx_cache_fault& operator=(const rsx_cache_fault&) = delete;
	struct completion
	{
		arm64_memory_state reply;
		std::size_t host_pages, watched_pages, published_surfaces, invalidated_textures;
	};
	// Synchronous,non-reentrant render-owner task from initial permission/
	// mapping capture through all GPU preparation and exact byte/register reply.
	// The dispatcher must keep the original CPU parked until it publishes reply.
	// Safe instruction fetch,signal dispatch and lifecycle are external contracts.
	[[nodiscard]] completion complete(const arm64_memory_request& request, std::uint64_t fault, bool writing)
	{
		if (std::this_thread::get_id() != m_owner || guest_memory_epoch::active())
			throw std::logic_error("Native instruction completion requires its render owner outside VM exclusion");
		operation_guard guard{m_active};
		const auto plan = m_access->capture_instruction(request, fault, writing, *m_memory);
		const auto published = m_surfaces.prepare_physical_fault(*plan->identity, plan->registrations, writing);
		const auto invalidated = writing || published ? m_textures.invalidate_physical_fault(*plan->identity, plan->registrations) : 0;
		const auto reply = m_access->complete_instruction(*plan, *m_memory);
		if (!reply) throw cache_fault_conflict();
		return {*reply, plan->pages.size(), plan->faults.size(), published, invalidated};
	}
	[[nodiscard]] std::optional<preparation> prepare(std::uint32_t address, bool writing)
	{
		if (std::this_thread::get_id() != m_owner || guest_memory_epoch::active())
			throw std::logic_error("Native cache fault preparation requires its render owner outside VM exclusion");
		operation_guard guard{m_active};
		auto plan = m_access->capture_fault(address);
		if (!plan) return {};
		const auto published = m_surfaces.prepare_physical_fault(plan->identity(), plan->registrations(), writing);
		const auto invalidated = writing || published ? m_textures.invalidate_physical_fault(plan->identity(), plan->registrations()) : 0;
		// Reads do not simulate a CPU write or discard valid native precision.
		// A write retires both revision domains after all GPU publication. The
		// real instruction still must revalidate and update reservations/bytes.
		const auto current = writing ? m_access->invalidate_fault(*m_memory, *plan) : m_access->fault_current(*plan);
		if (!current) throw cache_fault_conflict();
		return preparation{std::move(plan), published, invalidated, writing};
	}
};
} // namespace metal4
