#pragma once

#include "M4DrawExecutor.h"
#include "M4FramebufferSnapshot.h"

namespace metal4
{
// SurfaceCache::acquire registers descriptors without guest reads/GPU writes.
// Fully validate capture before this call. Carries owned IO/tile tickets;
// productive VM protection and old-backing/output leases are separate.
inline draw_targets acquire_framebuffer(const framebuffer_snapshot& snapshot, surface_cache& cache)
{
	// Native raster sample count comes from the owned surface geometry. Guest
	// sample masks/A2C/A2One come from the separately owned draw environment;
	// storage expands sample indices without a host averaging resolve.
	draw_targets result;
	result.register_color_format = snapshot.color_format();
	const auto acquire = [&](const framebuffer_surface& surface)
	{
		const auto* mapped = surface.mapping ? &surface.mapping->state() : nullptr;
		return cache.acquire({surface.address, {surface.width, surface.height, surface.samples, surface.format, surface.samples > 1}, surface.row_pitch, surface.morton,
			mapped ? mapped->tile : std::nullopt, mapped ? mapped->segments : std::vector<guest_storage_segment>{}, surface.mapping});
	};
	for (unsigned slot = 0; slot < 4; ++slot)
		if (snapshot.colors()[slot])
			result.colors[slot] = acquire(*snapshot.colors()[slot]);
	if (snapshot.depth())
		result.depth = acquire(*snapshot.depth());
	return result;
}
} // namespace metal4
