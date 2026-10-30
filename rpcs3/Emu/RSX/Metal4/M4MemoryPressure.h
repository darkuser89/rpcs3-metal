#pragma once

#include "M4GuestTexture.h"
#include "M4Display.h"
#include "M4SurfaceCache.h"
#include "M4Frame.h"

namespace metal4
{
// Render-owner recovery for one shared native ledger. Lifetime ends before
// the caches/context. It never waits, encodes, maps guest bytes or spills
// irreplaceable native data into another allocation on unified memory.
class memory_pressure_owner final
{
	context& m_context;
	surface_cache& m_surfaces;
	guest_texture_cache& m_textures;
	display_resolver* m_display;
	native_frame* m_frame;
	std::shared_ptr<heap_budget> m_budget;
	std::shared_ptr<void> m_registration;
	void reclaim(memory_pressure_request request)
	{
		// The surface entry point checks both the VM epoch and its CPU-capture
		// scope before releasing any owner, including reusable upload pages.
		(void)m_surfaces.reclaim_idle();
		(void)m_context.trim_upload_pages();
		const auto needs_room = [&]
		{
			const auto stats = m_budget->statistics();
			return request.native_failure || request.requested > stats.budget - stats.reserved_bytes;
		};
		// Each successful step removes one cache resource owner. No entry is
		// created here, so all passes are bounded by the entry counts at entry.
		while (needs_room() && m_display && m_display->reclaim_one()) {}
		if (needs_room() && m_frame) (void)m_frame->reclaim_fsr();
		while (needs_room() && m_textures.reclaim_one()) {}
		while (needs_room() && m_surfaces.reclaim_clean()) {}
	}
  public:
	memory_pressure_owner(context& device, surface_cache& surfaces, guest_texture_cache& textures, std::shared_ptr<heap_budget> budget, display_resolver* display = nullptr, native_frame* frame = nullptr)
		: m_context(device), m_surfaces(surfaces), m_textures(textures), m_display(display), m_frame(frame), m_budget(std::move(budget))
	{
		if (!m_budget || m_budget->device() != device.device() || surfaces.heap_ledger() != m_budget || textures.heap_ledger() != m_budget || (m_display && m_display->heap_ledger() != m_budget) || (m_frame && &m_frame->m_context != &device))
			throw std::invalid_argument("Native memory pressure owners require one device and shared ledger");
		m_registration = m_budget->register_reclaimer([this](memory_pressure_request request) { reclaim(request); });
	}
	memory_pressure_owner(const memory_pressure_owner&) = delete;
	memory_pressure_owner& operator=(const memory_pressure_owner&) = delete;
};
} // namespace metal4
