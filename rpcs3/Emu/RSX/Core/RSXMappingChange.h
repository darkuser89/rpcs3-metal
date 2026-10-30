#pragma once

#include <cstdint>

namespace rsx
{
// The VM owns the sink and its writer lock for this entire scope. Pair both
// notifications even if a mapping operation throws after changing some pages.
// End means the transition stopped,not that the requested mapping succeeded.
// Sinks must not allocate,throw,wait or enter a VM provider from either callback.
template <typename Sink>
class mapping_change_scope final
{
	Sink* const m_sink;
	const std::uint32_t m_address, m_size;

  public:
	mapping_change_scope(Sink* sink, std::uint32_t address, std::uint32_t size) noexcept
		: m_sink(sink), m_address(address), m_size(size)
	{
		static_assert(noexcept(m_sink->on_mapping_change_begin(m_address, m_size)));
		static_assert(noexcept(m_sink->on_mapping_change_end(m_address, m_size)));
		if (m_sink) m_sink->on_mapping_change_begin(m_address, m_size);
	}
	~mapping_change_scope()
	{
		if (m_sink) m_sink->on_mapping_change_end(m_address, m_size);
	}
	mapping_change_scope(const mapping_change_scope&) = delete;
	mapping_change_scope& operator=(const mapping_change_scope&) = delete;
};
} // namespace rsx
