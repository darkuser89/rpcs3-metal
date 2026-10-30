#pragma once

#include "M4GuestSurface.h"
#include "M4ExclusiveGuestMemory.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <functional>
#include <limits>
#include <stdexcept>
#include <thread>

namespace metal4
{
// Level-triggered page invalidations,not an ordered log of VM incarnations.
// All old native/physical dirty owners must already be retained independently.
// The owner service must reconcile them without accessing a reused guest VA.
// Producers may hold VM exclusion:only fixed-storage lock-free atomics here;
// no allocation,callback,owner lock,provider access or GPU/CPU suspension wait.
class mapping_notifications final
{
  public:
	enum class kind { read, unmap };
	static constexpr std::size_t page_bytes = 4096, page_count = 0x1'0000'0000ull / page_bytes;
	static constexpr std::size_t word_count = page_count / 64, maximum_ranges = page_count * 2;
	struct drain_statistics
	{
		std::uint64_t root_words = 0, summary_words = 0, page_words = 0, ranges = 0;
	};
  private:
	static constexpr std::size_t summary_count = word_count / 64, root_count = summary_count / 64;
	std::array<std::array<std::atomic<std::uint64_t>, summary_count>, 2> m_dirty{};
	std::array<std::array<std::atomic<std::uint64_t>, root_count>, 2> m_roots{};
	drain_statistics m_statistics{}; // Render owner only; producers touch no counters.
	std::array<std::array<std::atomic<std::uint64_t>, word_count>, 2> m_pages{};
	std::atomic<bool> m_pending{false}, m_poisoned{false};
	std::thread::id m_owner;
	bool m_bound = false, m_draining = false;
	static constexpr std::uint64_t mask(std::size_t first, std::size_t count) noexcept
	{
		return (std::numeric_limits<std::uint64_t>::max() >> (64 - count)) << first;
	}
  public:
	mapping_notifications() { static_assert(std::atomic<std::uint64_t>::is_always_lock_free); }
	mapping_notifications(const mapping_notifications&) = delete;
	mapping_notifications& operator=(const mapping_notifications&) = delete;
	void bind_owner()
	{
		if (m_bound) throw std::logic_error("Native mapping notification owner is already bound");
		m_owner = std::this_thread::get_id(); m_bound = true;
	}
	void notify(guest_surface_range range, kind cause) noexcept
	{
		if (!range.size) return;
		if (range.size > 0x1'0000'0000ull - range.address || (cause != kind::read && cause != kind::unmap))
		{
			m_poisoned.store(true, std::memory_order_release);
			m_pending.store(true, std::memory_order_release); return;
		}
		const auto index = cause == kind::read ? 0 : 1;
		auto& words = m_pages[index];
		const auto end = (std::uint64_t(range.address) + range.size + page_bytes - 1) / page_bytes;
		for (std::size_t page = range.address / page_bytes; page < end;)
		{
			const auto first = page % 64, count = std::min<std::size_t>(end - page, 64 - first);
			const auto word = page / 64, group = word / 64;
			words[word].fetch_or(mask(first, count), std::memory_order_release);
			// Publish from leaves towards roots; an owner may observe a harmless
			// redundant summary, but cannot consume a summary before its leaf.
			m_dirty[index][group].fetch_or(std::uint64_t{1} << (word % 64), std::memory_order_release);
			m_roots[index][group / 64].fetch_or(std::uint64_t{1} << (group % 64), std::memory_order_release);
			page += count;
		}
		m_pending.store(true, std::memory_order_release);
	}
	[[nodiscard]] bool pending() const noexcept { return m_pending.load(std::memory_order_acquire); }
	// Statistics are read on the bound render owner, outside drain.
	[[nodiscard]] drain_statistics statistics() const noexcept { return m_statistics; }
	// Only the bound render owner,outside its VM/task/engine locks. A provider
	// may then acquire exclusion;the mapper never waits for this owner service.
	// Read requests precede unmap requests in each bounded pass. Failed callbacks
	// must be retryable/idempotent;their current range and unprocessed bits stay.
	// Notifications arriving during service remain pending for another pass.
	template <typename Service>
	std::size_t drain(Service&& service, std::size_t range_limit = 256)
	{
		if (!m_bound || m_owner != std::this_thread::get_id()) throw std::logic_error("Native mapping service requires its render owner");
		if (!range_limit || range_limit > maximum_ranges) throw std::invalid_argument("Native mapping service range budget is invalid");
		if (m_draining) return 0;
		if (guest_memory_epoch::active()) return 0; // Never run GPU/service work inside an existing VM capture.
		if (m_poisoned.load(std::memory_order_acquire)) throw std::runtime_error("Native mapping notification range is poisoned");
		if (!m_pending.exchange(false, std::memory_order_acq_rel)) return 0;
		struct scope
		{
			bool& active;
			~scope() { active = false; }
		} guard{m_draining};
		m_draining = true;
		std::size_t delivered = 0;
		for (std::size_t cause = 0; cause < 2; ++cause)
			for (std::size_t root = 0; root < root_count; ++root)
			{
				++m_statistics.root_words;
				auto groups = m_roots[cause][root].exchange(0, std::memory_order_acq_rel);
				while (groups)
				{
					const auto group = root * 64 + std::countr_zero(groups);
					++m_statistics.summary_words;
					auto words = m_dirty[cause][group].exchange(0, std::memory_order_acq_rel);
					while (words)
					{
						const auto word = group * 64 + std::countr_zero(words);
						auto& source = m_pages[cause][word];
						++m_statistics.page_words;
						auto bits = source.exchange(0, std::memory_order_acq_rel);
						const auto restore = [&]
						{
							// Return ALL locally claimed, unprocessed summaries. OR
							// merges arrivals without overwriting another producer.
							source.fetch_or(bits, std::memory_order_release);
							m_dirty[cause][group].fetch_or(words, std::memory_order_release);
							m_roots[cause][root].fetch_or(groups, std::memory_order_release);
							m_pending.store(true, std::memory_order_release);
						};
						try
						{
							while (bits)
							{
								if (m_poisoned.load(std::memory_order_acquire)) throw std::runtime_error("Native mapping notification range is poisoned");
								if (delivered == range_limit) { restore(); return delivered; }
								const auto first = std::countr_zero(bits), count = std::countr_one(bits >> first);
								const guest_surface_range range{static_cast<std::uint32_t>((word * 64 + first) * page_bytes), count * page_bytes};
								std::invoke(service, range, cause == 0 ? kind::read : kind::unmap);
								bits &= ~mask(first, count); ++delivered; ++m_statistics.ranges;
							}
						}
						catch (...) { restore(); throw; }
						words &= words - 1;
					}
					groups &= groups - 1;
				}
			}
		return delivered;
	}
};
} // namespace metal4
