#pragma once

#include "M4GuestTexture.h"
#include "M4MappingNotifications.h"
#include "M4SurfaceCache.h"

namespace metal4
{
// Native owner service shared by actual GSRender and headless GPU tests.
// Foreign mapping producers only enqueue in mapping_notifications.
class mapping_invalidation_service final
{
	surface_cache& m_surfaces;
	guest_texture_cache& m_textures;
	const std::thread::id m_owner = std::this_thread::get_id();
	bool m_servicing = false;
  public:
	mapping_invalidation_service(context& device, surface_cache& surfaces, guest_texture_cache& textures)
		: m_surfaces(surfaces), m_textures(textures)
	{
		textures.validate_surface_adapter(device, surfaces);
	}
	std::size_t service(guest_surface_range range, mapping_notifications::kind cause)
	{
		if (m_owner != std::this_thread::get_id() || m_servicing || guest_memory_epoch::active())
			throw std::logic_error("Native mapping service requires its owner outside VM exclusion and recursive dispatch");
		if (cause != mapping_notifications::kind::read && cause != mapping_notifications::kind::unmap)
			throw std::invalid_argument("Invalid native mapping service cause");
		validate_guest_surface_range(range);
		if (!range.size) return 0;
		struct scope { bool& active; ~scope() { active = false; } } guard{m_servicing};
		m_servicing = true;
		const auto selected = m_surfaces.invalidate_mapping(range, cause == mapping_notifications::kind::read ?
			surface_mapping_event::read : surface_mapping_event::unmap);
		// Use upload-time physical tickets; never reconstruct old identity from
		// replaced/protected virtual bytes. Independent images remain cached.
		(void)m_textures.invalidate_mapping(range);
		return selected;
	}
};
} // namespace metal4
