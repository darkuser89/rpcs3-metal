#pragma once

#include "M4PreparedHostTransfer.h"
#include "M4SurfaceCache.h"

namespace metal4
{
// Render owner only,inside the paused writer's capture_context. GPU flushes
// precede provider exclusion;alias recheck and both physical leases share one
// CPU epoch. No journal slot is armed until the producer resumes its ticket.
[[nodiscard]] inline prepared_host_transfer prepare_host_transfer(surface_cache& surfaces, guarded_guest_memory& memory,
	std::optional<guest_surface_range> destination, std::optional<guest_surface_range> source)
{
	if (!destination && !source) throw std::invalid_argument("Guest host transfer has no guest access range");
	std::array<guest_surface_range, 2> ranges{};
	std::size_t count = 0;
	if (source) ranges[count++] = *source;
	if (destination) ranges[count++] = *destination;
	prepared_host_transfer prepared;
	// Conservatively invalidate clean source mirrors too. A read-only access
	// creates no guest write revision or reservation notification.
	// Another leased producer may already be paused in a fault while this
	// preparation is queued. Provider capture drains all active writers;publish
	// its conservative prefix revisions without dropping that producer's intent.
	memory.with_host_write_recovery([&]
	{
		surfaces.cpu_write(std::span{ranges}.first(count), [&]
		{
			if (source) prepared.source.emplace(memory.prepare_host_access(*source, false));
			if (destination) prepared.destination.emplace(memory.prepare_host_access(*destination, true));
		});
	});
	return prepared;
}
} // namespace metal4
