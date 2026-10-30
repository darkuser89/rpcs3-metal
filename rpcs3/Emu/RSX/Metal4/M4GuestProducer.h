#pragma once

#include <functional>
#include <stdexcept>

namespace metal4
{
// MM work must complete before draining DMA. A false DMA result means the
// producer is fault-blocked,not complete. Never capture or publish afterward.
template <typename Flush, typename Drain>
void synchronize_guest_producers(Flush&& flush, Drain&& drain)
{
	std::invoke(flush);
	if (!std::invoke(drain))
		throw std::runtime_error("Native guest producer is blocked in DMA fault recovery");
}
} // namespace metal4
