#pragma once

#include "M4GuestSurface.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>

namespace metal4
{
// RSX IO mappings use 1MiB entries. Producers and readers share VM exclusion;
// the stamp is reserved before entry publication,not an asynchronous snapshot.
// Owned before installing callbacks;notification never allocates,locks or waits.
class io_mapping_epoch final
{
	static constexpr std::uint64_t poisoned = std::numeric_limits<std::uint64_t>::max();
	static constexpr std::size_t entry_bytes = 1 << 20;
	std::atomic<std::uint64_t> m_generation;
	std::array<std::atomic<std::uint64_t>, 4096> m_entries{};

  public:
	static constexpr std::size_t table_bytes = sizeof(m_entries);
	explicit io_mapping_epoch(std::uint64_t initial = 1) : m_generation(initial)
	{
		static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
		if (!initial || initial == poisoned)
			throw std::invalid_argument("IO mapping epoch requires a nonzero unexhausted generation");
	}
	void notify(guest_surface_range range) noexcept
	{
		if (range.size > 0x1'0000'0000ull - range.address || (range.size && ((range.address | range.size) & (entry_bytes - 1))))
		{
			m_generation.store(poisoned, std::memory_order_release);
			return;
		}
		if (!range.size)
			return;
		auto generation = m_generation.load(std::memory_order_relaxed);
		while (generation != poisoned && !m_generation.compare_exchange_weak(generation, generation + 1, std::memory_order_release, std::memory_order_relaxed))
		{
		}
		if (generation >= poisoned - 1)
			return;
		const auto stamp = generation + 1;
		const auto end = (std::uint64_t(range.address) + range.size) / entry_bytes;
		for (std::size_t entry = range.address / entry_bytes; entry < end; ++entry)
		{
			auto previous = m_entries[entry].load(std::memory_order_relaxed);
			while (previous < stamp && !m_entries[entry].compare_exchange_weak(previous, stamp, std::memory_order_release, std::memory_order_relaxed))
			{
			}
		}
	}
	[[nodiscard]] std::uint64_t current() const
	{
		const auto generation = m_generation.load(std::memory_order_acquire);
		if (generation == poisoned)
			throw std::overflow_error("Native IO mapping epoch is exhausted or invalid");
		return generation;
	}
	[[nodiscard]] std::uint64_t current(guest_surface_range range) const
	{
		validate_guest_surface_range(range);
		(void)current();
		std::uint64_t result = 0;
		if (range.size)
		{
			const auto end = (std::uint64_t(range.address) + range.size + entry_bytes - 1) / entry_bytes;
			for (std::size_t entry = range.address / entry_bytes; entry < end; ++entry)
				result = std::max(result, m_entries[entry].load(std::memory_order_acquire));
		}
		(void)current();
		return result;
	}
};
} // namespace metal4
