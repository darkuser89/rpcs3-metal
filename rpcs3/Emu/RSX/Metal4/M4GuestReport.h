#pragma once

#include "M4GuestTexture.h"
#include "M4SurfaceCache.h"

namespace metal4
{
// Called after the RSX scheduler's report scaling and MM ordering. GPU waits
// and dirty alias writeback finish before the atomic CPU-only VM transaction.
// Pending failure never publishes the report or invalidates native contents.
inline void store_guest_report(surface_cache& surfaces, guest_texture_cache& textures,
							   std::uint32_t address, std::uint64_t timestamp, std::uint32_t value,
							   const std::function<void(std::uint32_t, std::uint64_t, std::uint32_t)>& store,
							   std::chrono::milliseconds timeout = std::chrono::seconds(30))
{
	if ((address & 15u) || !store)
		throw std::invalid_argument("Native report requires aligned address and atomic store");
	const guest_surface_range range{address, 16};
	validate_guest_surface_range(range);
	surfaces.synchronize(timeout);
	surfaces.cpu_write(range, [&]
					   { store(address, timestamp, value); });
	textures.invalidate(range);
}
} // namespace metal4
