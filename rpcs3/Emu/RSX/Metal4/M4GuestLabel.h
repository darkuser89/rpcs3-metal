#pragma once

#include "M4GuestTexture.h"
#include "M4SurfaceCache.h"

namespace metal4
{
// Flip semaphores intentionally bypass pipeline/DMA waits so guest reset and
// display-queue recovery cannot deadlock. The atomic writer must still update
// guarded revisions/notifications;pending dirty aliases then fail writeback.
inline void store_guest_flip_label(std::uint32_t address, std::uint32_t value,
								   const std::function<void(std::uint32_t, std::uint32_t)>& store)
{
	if ((address & 3u) || !store)
		throw std::invalid_argument("Native flip label requires aligned address and atomic writer");
	validate_guest_surface_range({address, 4});
	store(address, value);
}

// Synchronous native semaphore release. Pending native work, DMA/reports and
// dirty alias writeback must precede the atomic store,including same-value
// releases:the shared RSX caller may return immediately after seeing our data.
inline void release_guest_label(surface_cache& surfaces, guest_texture_cache& textures,
								std::uint32_t address, std::uint32_t value, const std::function<void()>& synchronize_dependencies,
								const std::function<void(std::uint32_t, std::uint32_t)>& store,
								std::chrono::milliseconds timeout = std::chrono::seconds(30))
{
	if ((address & 3u) || !synchronize_dependencies || !store)
		throw std::invalid_argument("Native label requires aligned address,dependency sync and atomic store");
	const guest_surface_range range{address, 4};
	validate_guest_surface_range(range);
	surfaces.synchronize(timeout);
	synchronize_dependencies();
	surfaces.cpu_write(range, [&]
					   { store(address, value); });
	textures.invalidate(range);
}
} // namespace metal4
