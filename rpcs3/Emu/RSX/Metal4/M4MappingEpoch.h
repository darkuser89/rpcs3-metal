#pragma once

#include "M4GuestSurface.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>

namespace metal4
{
// A mapping producer may hold the real VM writer lock. Notification must not
// allocate,reenter VM/CPU suspension or wait on renderer/GPU work. Queries and
// producers MUST share exclusion:the unique stamp is reserved before range
// publication. A stamp maximum is exact under that contract,not an asynchronous
// snapshot. 4KiB pages conservatively cover the notification's edge bytes.
class mapping_epoch final
{
	std::atomic<std::uint64_t> m_generation;
	static constexpr auto poisoned = std::numeric_limits<std::uint64_t>::max();
	// One atomic word covers both transition serial and active producer count.
	// A producer publishes busy BEFORE touching mapping metadata and publishes
	// its final serial AFTER mutation. Nested/concurrent producers cannot expose
	// an idle gap. Saturation is permanent refusal,never wraparound/ABA.
	static constexpr std::uint64_t active_mask = 0xffff, serial_step = 0x10000;
	std::atomic<std::uint64_t> m_transitions{0};
	void transition(bool begin) noexcept
	{
		auto previous = m_transitions.load(std::memory_order_relaxed);
		while (previous != poisoned)
		{
			const auto active = previous & active_mask;
			const bool invalid = begin ? active == active_mask : active == 0;
			const bool exhausted = (previous & ~active_mask) == (poisoned & ~active_mask);
			const auto next = invalid || exhausted ? poisoned :
				((previous & ~active_mask) + serial_step) | (begin ? active + 1 : active - 1);
			if (m_transitions.compare_exchange_weak(previous, next, std::memory_order_acq_rel, std::memory_order_relaxed))
				return;
		}
	}
	static constexpr std::size_t page_count = 0x1'0000'0000ull / 4096;
	static constexpr std::size_t pages_per_region = 256;
	static constexpr std::size_t region_count = page_count / pages_per_region;
	struct region
	{
		std::atomic<std::uint64_t> whole{0}, partial{0};
	};
	// Fixed ownership allocated before installing any producer callback.
	std::unique_ptr<std::atomic<std::uint64_t>[]> m_pages;
	std::unique_ptr<region[]> m_regions;
	static void raise(std::atomic<std::uint64_t>& value, std::uint64_t stamp) noexcept
	{
		auto previous = value.load(std::memory_order_relaxed);
		while (previous < stamp && !value.compare_exchange_weak(previous, stamp, std::memory_order_release, std::memory_order_relaxed))
		{
		}
	}

  public:
	class observation final
	{
		friend class mapping_epoch;
		const mapping_epoch* m_owner;
		std::uint64_t m_serial, m_generation;
		observation(const mapping_epoch* owner, std::uint64_t serial, std::uint64_t generation)
			: m_owner(owner), m_serial(serial), m_generation(generation) {}
	};
	// For asynchronous readers of ATOMIC mapping metadata whose producers all
	// use these paired notifications. VM startup/teardown must be excluded by
	// the owner lifetime protocol before installing any observer. This token is
	// neither a VM lock nor a backing lease;it does not permit byte access.
	// Observe BEFORE reading metadata,then unchanged AFTER reading/using it.
	// An intervening transition,including an unrelated one,refuses this global
	// identity observation without invalidating unrelated per-range watches.
	[[nodiscard]] std::optional<observation> observe() const noexcept
	{
		const auto serial = m_transitions.load(std::memory_order_acquire);
		const auto generation = m_generation.load(std::memory_order_acquire);
		if ((serial & active_mask) || generation == poisoned)
			return {};
		return observation(this, serial, generation);
	}
	[[nodiscard]] bool unchanged(const observation& value) const noexcept
	{
		// Keep the caller's preceding atomic metadata reads before validation,
		// including on Apple ARM's weakly ordered memory model.
		std::atomic_thread_fence(std::memory_order_acquire);
		return value.m_owner == this && m_transitions.load(std::memory_order_acquire) == value.m_serial &&
			m_generation.load(std::memory_order_acquire) == value.m_generation;
	}
	void begin_change(guest_surface_range range) noexcept
	{
		transition(true);
		notify(range);
	}
	void end_change(guest_surface_range range) noexcept
	{
		notify(range);
		transition(false);
	}
	static constexpr std::size_t storage_bytes = page_count * sizeof(std::atomic<std::uint64_t>) + region_count * sizeof(region);
	explicit mapping_epoch(std::uint64_t initial = 1) : m_generation(initial)
	{
		static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
		if (!initial || initial == poisoned)
			throw std::invalid_argument("Mapping epoch requires a nonzero unexhausted generation");
		m_pages = std::make_unique<std::atomic<std::uint64_t>[]>(page_count);
		m_regions = std::make_unique<region[]>(region_count);
	}
	void notify(guest_surface_range range) noexcept
	{
		if (range.size > 0x1'0000'0000ull - range.address)
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
		const auto end = (std::uint64_t(range.address) + range.size + 4095) / 4096;
		for (std::size_t page = range.address / 4096; page < end;)
		{
			const auto index = page / pages_per_region;
			if (page % pages_per_region == 0 && end - page >= pages_per_region)
			{
				raise(m_regions[index].whole, stamp);
				page += pages_per_region;
			}
			else
			{
				const auto part_end = std::min<std::size_t>(end, (index + 1) * pages_per_region);
				for (; page < part_end; ++page)
					raise(m_pages[page], stamp);
				raise(m_regions[index].partial, stamp);
			}
		}
	}
	[[nodiscard]] std::uint64_t current() const
	{
		const auto generation = m_generation.load(std::memory_order_acquire);
		if (generation == poisoned)
			throw std::overflow_error("Native VM mapping epoch is exhausted or invalid");
		return generation;
	}
	// No allocation or VM access. Whole regions skip their 256 page cells;
	// partial queries inspect only their edge pages,never an unrelated page.
	[[nodiscard]] std::uint64_t current(guest_surface_range range) const
	{
		validate_guest_surface_range(range);
		(void)current();
		std::uint64_t result = 0;
		const auto end = range.size ? (std::uint64_t(range.address) + range.size + 4095) / 4096 : range.address / 4096;
		for (std::size_t page = range.address / 4096; page < end;)
		{
			const auto index = page / pages_per_region;
			result = std::max(result, m_regions[index].whole.load(std::memory_order_acquire));
			if (page % pages_per_region == 0 && end - page >= pages_per_region)
			{
				result = std::max(result, m_regions[index].partial.load(std::memory_order_acquire));
				page += pages_per_region;
			}
			else
			{
				const auto part_end = std::min<std::size_t>(end, (index + 1) * pages_per_region);
				for (; page < part_end; ++page)
					result = std::max(result, m_pages[page].load(std::memory_order_acquire));
			}
		}
		(void)current();
		return result;
	}
};
} // namespace metal4
