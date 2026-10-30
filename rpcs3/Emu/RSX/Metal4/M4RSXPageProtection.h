#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "Native Metal4 page protection requires Apple Silicon."
#endif

#include "Emu/Memory/vm.h"
#include "Emu/Memory/vm_locking.h"
#include "Emu/Memory/vm_mapping_observer.h"
#include "M4GuestSurface.h"
#include "M4PhysicalBacking.h"
#include "M4PhysicalPageIndex.h"
#include "M4PageProtectionPlan.h"
#include "util/vm.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>

namespace metal4
{
// Owned overlays on the CPU/base mapping only. The VM's logical permissions
// and sudo mapping are unchanged. All operations require global VM writer
// exclusion; registration/release additionally require suspended guest CPUs
// and drained host writers, supplied by the concrete VM provider. Cache/fault
// ownership must retire a token before reopening its physical bytes.
//
// Fixed tables and weak allocation identities permit allocation-free mapping
// callbacks, including newly created aliases and constructor-local restore.
// A weak identity cannot retain a VM allocation or confuse a recycled shm/VA.
class rsx_page_protection final
{
	friend class rsx_mapping_observer;
	friend class rsx_vm_access;
  public:
	enum class access { trap_writes, trap_reads_and_writes };
	class token final
	{
		friend class rsx_page_protection;
		std::shared_ptr<const std::byte> m_owner;
		std::uint64_t m_value = 0;
		token(std::shared_ptr<const std::byte> owner, std::uint64_t value) : m_owner(owner), m_value(value) {}
	  public:
		token() = default;
		bool operator==(const token&) const = default;
	};

  private:
	struct watch
	{
		std::uint64_t id = 0;
		std::weak_ptr<utils::shm> memory;
		std::uint64_t offset = 0, size = 0;
		access mode = access::trap_reads_and_writes;
		std::weak_ptr<void> registration;
		bool leased = false;
	};
	struct logical_page
	{
		std::uint8_t flags = 0;
		std::weak_ptr<utils::shm> memory;
		std::uint64_t offset = 0;
	};
  public:
	// Immutable CPU-only receipt. Registration owners and independent kernel
	// mappings survive an owner handoff;no VM allocation handle or guest VA is
	// retained as physical storage. This neither opens CPU pages nor grants an
	// instruction retry. Byte access still requires provider exclusion.
	class fault_plan final
	{
		friend class rsx_page_protection;
		friend class rsx_vm_access;
		struct physical_slice
		{
			std::shared_ptr<void> mapping;
			std::size_t offset, size;
		};
		std::shared_ptr<const std::byte> m_domain;
		std::uint64_t m_generation = 0;
		std::uint32_t m_address = 0;
		std::vector<watch> m_watches;
		std::vector<logical_page> m_pages;
		std::vector<std::shared_ptr<void>> m_registrations;
		std::vector<physical_slice> m_slices;
		std::optional<physical_identity> m_identity;
		fault_plan() = default;
	  public:
		fault_plan(const fault_plan&) = delete;
		fault_plan& operator=(const fault_plan&) = delete;
		[[nodiscard]] std::span<const std::shared_ptr<void>> registrations() const noexcept { return m_registrations; }
		[[nodiscard]] const physical_identity& identity() const noexcept { return *m_identity; }
	};
  private:
	const std::shared_ptr<const std::byte> m_domain = std::make_shared<const std::byte>();
	const std::size_t m_page_size;
	std::vector<watch> m_watches;
	physical_page_index m_index;
	page_protection_plan m_page_plan;
	std::uint64_t m_refreshed_pages = 0;
	mutable std::uint64_t m_alias_visits = 0;
	std::uint64_t m_next = 0, m_generation = 0;
	bool m_bound = false, m_cache_domain_used = false;
	std::atomic<bool> m_failed{false};
	void changed() noexcept
	{
		if (m_generation == std::numeric_limits<std::uint64_t>::max()) m_failed = true;
		else ++m_generation;
	}

	[[nodiscard]] static std::size_t checked_capacity(std::size_t size)
	{
		if (!size || size > 65536) throw std::invalid_argument("Invalid native protection segment budget");
		return size;
	}
	[[nodiscard]] static std::size_t host_page_size()
	{
		const auto size = utils::get_page_size();
		if (size < 4096 || size > 65536 || (size & (size - 1)))
			throw std::runtime_error("Unsupported native VM page granularity");
		return size;
	}
	[[nodiscard]] static bool matches(const watch& value, const vm::backing_segment& part) noexcept
	{
		return value.id && part.memory && !value.memory.owner_before(part.memory) && !part.memory.owner_before(value.memory) &&
			value.offset < part.offset + part.size && part.offset < value.offset + value.size;
	}
	void require_lock(const vm::writer_lock& lock) const
	{
		if (lock.range_lock || !m_bound || m_failed)
			throw std::logic_error("Native protection requires a live owner and global VM writer exclusion");
	}
	template <typename Lookup>
	void refresh_index(std::uint64_t first, std::uint64_t end, Lookup&& lookup) noexcept
	{
		first &= ~4095ull;
		end = std::min<std::uint64_t>((end + 4095) & ~4095ull, 0x1'0000'0000ull);
		for (auto address = first; address < end; address += 4096)
		{
			++m_refreshed_pages;
			const auto a = static_cast<std::uint32_t>(address);
			if (!vm::check_addr(a, vm::page_allocated, 4096))
			{
				if (!m_index.replace(a / 4096, {}, 0)) std::terminate();
				continue;
			}
			const auto part = lookup(a, 4096);
			if (!part.memory || part.size != 4096 || !m_index.replace(a / 4096, part.memory, part.offset)) m_failed = true;
		}
	}
	void dirty(const watch& w) noexcept
	{
		if (w.id) m_alias_visits += m_index.aliases(w.memory, w.offset, w.size,
			[&](std::uint32_t address, std::uint64_t) { m_page_plan.mark(address); });
	}
	template <typename Lookup>
	page_protection_plan::access logical_access(std::uint32_t address, bool live, Lookup&& lookup) noexcept
	{
		page_protection_plan::access result;
		if (!vm::check_addr(address, vm::page_allocated, 4096)) { result.baseline = PROT_NONE; return result; }
		const auto flags = vm::get_addr_flags(address).second;
		result.baseline = flags & vm::page_readable ? (flags & vm::page_writable ? PROT_READ | PROT_WRITE : PROT_READ) : PROT_NONE;
		if (!live) return result;
		const auto part = lookup(address, 4096);
		if (!part.memory || part.size != 4096)
		{
			m_failed = true; result.selected = true; result.overlay = PROT_NONE; return result;
		}
		for (const auto& w : m_watches)
			if (matches(w, part)) { result.selected = true; result.overlay &= w.mode == access::trap_writes ? PROT_READ : PROT_NONE; }
		return result;
	}
	static void apply_protection(std::uint64_t address, std::uint64_t size, int permissions) noexcept
	{
		// Reapply after real mapping/permission mutations even when unchanged.
		// Never resume a VM producer after failure to install the overlay.
		if (::mprotect(vm::g_base_addr + address, size, permissions) != 0) std::terminate();
	}
	template <typename Lookup>
	void synchronize(std::uint64_t first, std::uint64_t end, Lookup&& lookup) noexcept
	{
		const bool live = std::any_of(m_watches.begin(), m_watches.end(), [](const watch& w) { return w.id != 0; });
		m_page_plan.synchronize(first, end, [&](std::uint32_t a) { return logical_access(a, live, lookup); }, apply_protection);
	}
	void synchronize(const vm::writer_lock& lock) noexcept
	{
		const bool live = std::any_of(m_watches.begin(), m_watches.end(), [](const watch& w) { return w.id != 0; });
		m_page_plan.synchronize_dirty([&](std::uint32_t a)
		{
			return logical_access(a, live, [&](std::uint32_t address, std::uint32_t size)
			{ return vm::get_backing_segment(address, size, lock, vm::page_allocated); });
		}, apply_protection);
	}

  public:
	explicit rsx_page_protection(std::size_t segment_capacity = 4096)
		: m_page_size(host_page_size()), m_watches(checked_capacity(segment_capacity)), m_page_plan(m_page_size)
	{
	}
	~rsx_page_protection() { if (m_bound) std::terminate(); }
	rsx_page_protection(const rsx_page_protection&) = delete;
	rsx_page_protection& operator=(const rsx_page_protection&) = delete;
	[[nodiscard]] std::size_t page_size() const noexcept { return m_page_size; }
	[[nodiscard]] bool failed() const noexcept { return m_failed; }
  private:
	[[nodiscard]] bool bind(const vm::writer_lock& lock) noexcept
	{
		// Once cache watches existed,an unobserved shutdown interval may contain
		// CPU stores. Reusing old virtual/physical revisions would be unsafe;
		// require a fresh provider,protection owner and guarded cache domain.
		if (lock.range_lock || m_bound || m_failed || m_cache_domain_used) return false;
		// One bounded initial metadata reconciliation under writer exclusion.
		// Mapping callbacks maintain only the changed logical pages thereafter.
		refresh_index(0, 0x1'0000'0000ull, [&](std::uint32_t a, std::uint32_t n)
		{ return vm::get_backing_segment(a, n, lock, vm::page_allocated); });
		m_bound = !m_failed; return m_bound;
	}
	void shutdown(const vm::writer_lock& lock) noexcept
	{
		if (lock.range_lock || !m_bound) std::terminate();
		changed();
		for (auto& w : m_watches) { dirty(w); w = {}; }
		synchronize(lock); m_bound = false;
	}
  public:
	[[nodiscard]] token protect(guest_surface_range range, access mode, const vm::writer_lock& lock)
	{
		return protect_ranges(std::span<const guest_surface_range>{&range, 1}, mode, lock, {});
	}
  private:
	[[nodiscard]] token protect_ranges(std::span<const guest_surface_range> ranges, access mode,
		const vm::writer_lock& lock, const std::shared_ptr<void>& owner)
	{
		require_lock(lock);
		if (ranges.empty() || (mode != access::trap_writes && mode != access::trap_reads_and_writes))
			throw std::invalid_argument("Invalid native protection coverage or mode");
		const auto available = std::count_if(m_watches.begin(), m_watches.end(), [](const watch& w) { return !w.id; });
		std::vector<watch> prepared;
		std::size_t bytes = 0;
		for (const auto range : ranges)
		{
			validate_guest_surface_range(range);
			if (!range.size || range.size > (64u << 20) - bytes) throw std::invalid_argument("Native protection coverage exceeds byte budget");
			bytes += range.size;
			for (auto cursor = std::uint64_t(range.address), end = cursor + range.size; cursor < end;)
			{
				const auto part = vm::get_backing_segment(static_cast<std::uint32_t>(cursor), static_cast<std::uint32_t>(end - cursor), lock, vm::page_allocated);
				if (!part.memory || !part.size) throw std::runtime_error("Native protection crosses an absent allocation");
				const auto already = [&](const watch& w)
				{
					return owner && w.leased && !w.registration.owner_before(owner) && !owner.owner_before(w.registration) &&
						!w.memory.owner_before(part.memory) && !part.memory.owner_before(w.memory) &&
						w.offset == part.offset && w.size == part.size && w.mode == mode;
				};
				if (!std::any_of(m_watches.begin(), m_watches.end(), already) && !std::any_of(prepared.begin(), prepared.end(), already))
				{
					if (prepared.size() == static_cast<std::size_t>(available)) throw std::length_error("Native protection segment budget exhausted");
					prepared.push_back({0, part.memory, part.offset, part.size, mode, owner, bool(owner)});
				}
				cursor += part.size;
			}
		}
		if (prepared.empty()) return {};
		if (m_next == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Native protection token exhausted");
		// All ranges,physical slices,allocations and capacity checks precede any
		// mutation. One owner can retain old and replacement physical mappings;
		// repeated captures of unchanged active bytes consume no extra slots.
		const auto id = ++m_next;
		for (auto& w : prepared) w.id = id;
		auto next = prepared.begin();
		for (auto& w : m_watches) if (!w.id && next != prepared.end()) { w = *next++; dirty(w); }
		changed();
		synchronize(lock);
		if (m_failed) throw std::runtime_error("Native protection found unresolved allocated VM metadata");
		return token{m_domain, id};
	}
	void protect_registration(std::span<const guest_surface_range> ranges, const std::shared_ptr<void>& owner, const vm::writer_lock& lock)
	{
		if (!owner) throw std::invalid_argument("Native cache watch requires its registration owner");
		(void)protect_ranges(ranges, access::trap_reads_and_writes, lock, owner);
		m_cache_domain_used = true;
	}
	void collect_retired(const vm::writer_lock& lock)
	{
		require_lock(lock);
		bool changed = false;
		for (auto& w : m_watches)
			if (w.id && w.leased && (w.registration.expired() || w.memory.expired())) { dirty(w); w = {}; changed = true; }
		if (changed) { this->changed(); synchronize(lock); }
		if (m_failed) throw std::runtime_error("Native protection retirement found unresolved VM metadata");
	}
  public:
	void release(token id, const vm::writer_lock& lock)
	{
		require_lock(lock);
		if (id.m_owner != m_domain || !id.m_value || std::none_of(m_watches.begin(), m_watches.end(), [&](const watch& w) { return w.id == id.m_value; }))
			throw std::invalid_argument("Foreign or retired native protection token");
		for (auto& w : m_watches) if (w.id == id.m_value) { dirty(w); w = {}; }
		changed();
		synchronize(lock);
		if (m_failed) throw std::runtime_error("Native protection release found unresolved VM metadata");
	}
	// Query on the render owner in a provider epoch. Select every physical
	// watch contributing to the fault's complete native page,including watches
	// on neighbouring logical pages;resolving only the fault byte is unsafe.
	[[nodiscard]] std::vector<token> fault_tokens(std::uint32_t address, const vm::writer_lock& lock) const
	{
		require_lock(lock);
		std::vector<token> result;
		const auto first = std::uint64_t(address) & ~(std::uint64_t(m_page_size) - 1);
		if (!m_page_plan.owned(address)) return result;
		for (auto cursor = first; cursor < first + m_page_size; cursor += 4096)
		{
			const auto part = vm::get_backing_segment(static_cast<std::uint32_t>(cursor), 4096, lock, vm::page_allocated);
			for (const auto& w : m_watches)
				if (matches(w, part) && std::find(result.begin(), result.end(), token{m_domain, w.id}) == result.end()) result.push_back(token{m_domain, w.id});
		}
		return result;
	}
  private:
	[[nodiscard]] std::vector<logical_page> fault_pages(std::uint32_t address, const vm::writer_lock& lock) const
	{
		std::vector<logical_page> result;
		const auto first = std::uint64_t(address) & ~(std::uint64_t(m_page_size) - 1);
		for (auto cursor = first; cursor < first + m_page_size; cursor += 4096)
		{
			const auto flags = vm::get_addr_flags(static_cast<std::uint32_t>(cursor)).second;
			const auto part = vm::get_backing_segment(static_cast<std::uint32_t>(cursor), 4096, lock, vm::page_allocated);
			if ((flags & vm::page_allocated) && (!part.memory || part.size != 4096))
				throw std::runtime_error("Native fault plan cannot identify allocated host-page neighbours");
			result.push_back({flags, part.memory, part.offset});
		}
		return result;
	}
	[[nodiscard]] std::shared_ptr<fault_plan> plan_fault(std::uint32_t address, const vm::writer_lock& lock) const
	{
		require_lock(lock);
		auto tokens = fault_tokens(address, lock);
		if (tokens.empty()) return {};
		auto result = std::shared_ptr<fault_plan>(new fault_plan);
		result->m_domain = m_domain; result->m_generation = m_generation; result->m_address = address;
		result->m_pages = fault_pages(address, lock);
		// Retiring a token later affects ALL its slices,not only the fault byte.
		// Include all tokens of each selected registration,including surviving
		// old aliases and replacement backing armed by subsequent captures.
		bool expanded;
		do
		{
			expanded = false;
			for (const auto& w : m_watches)
			{
				if (!w.id) continue;
				const auto selected = std::find(tokens.begin(), tokens.end(), token{m_domain, w.id}) != tokens.end();
				const auto owner_selected = w.leased && std::any_of(result->m_registrations.begin(), result->m_registrations.end(), [&](const auto& owner)
				{ return !w.registration.owner_before(owner) && !owner.owner_before(w.registration); });
				if (!selected && !owner_selected) continue;
				if (!selected) { tokens.push_back(token{m_domain, w.id}); expanded = true; }
				if (w.leased && !owner_selected)
				{
					auto owner = w.registration.lock();
					if (!owner) throw std::runtime_error("Native fault plan lost a selected registration");
					result->m_registrations.push_back(std::move(owner)); expanded = true;
				}
			}
		} while (expanded);
		for (const auto& w : m_watches)
			if (w.id && std::find(tokens.begin(), tokens.end(), token{m_domain, w.id}) != tokens.end()) result->m_watches.push_back(w);
		return result;
	}
	[[nodiscard]] bool fault_current(const fault_plan& plan, const vm::writer_lock& lock) const
	{
		require_lock(lock);
		if (plan.m_domain != m_domain || plan.m_generation != m_generation || !plan.m_identity) return false;
		const auto pages = fault_pages(plan.m_address, lock);
		if (pages.size() != plan.m_pages.size()) return false;
		for (std::size_t i = 0; i < pages.size(); ++i)
			if (pages[i].flags != plan.m_pages[i].flags || pages[i].offset != plan.m_pages[i].offset ||
				pages[i].memory.owner_before(plan.m_pages[i].memory) || plan.m_pages[i].memory.owner_before(pages[i].memory)) return false;
		return true;
	}
  public:
	struct index_statistics
	{
		std::uint64_t refreshed_logical_pages, visited_index_nodes;
		std::size_t mapped_logical_pages, metadata_bytes;
		page_protection_plan::statistics protection;
	};
	[[nodiscard]] index_statistics statistics(const vm::writer_lock& lock) const
	{
		require_lock(lock);
		return {m_refreshed_pages, m_alias_visits, m_index.size(), m_index.metadata_bytes() + m_page_plan.metadata_bytes(), m_page_plan.counters()};
	}
	template <typename Visit>
	std::size_t physical_aliases(const std::weak_ptr<void>& owner, std::uint64_t offset, std::uint64_t size,
		const vm::writer_lock& lock, Visit&& visit) const
	{
		require_lock(lock);
		const auto visits = m_index.aliases(owner, offset, size, std::forward<Visit>(visit));
		m_alias_visits += visits; return visits;
	}
	void mapping_changed(const vm::mapping_change_view& view, std::uint32_t address, std::uint32_t size) noexcept
	{
		if (!m_bound || !size || std::uint64_t(address) + size > 0x1'0000'0000ull) std::terminate();
		changed(); // Reject ABA even when a mapping is restored to the same shm/flags.
		refresh_index(address, std::uint64_t(address) + size, [&](std::uint32_t a, std::uint32_t n)
		{ return view.lookup(a, n, vm::page_allocated); });
		synchronize(address, std::uint64_t(address) + size, [&](std::uint32_t a, std::uint32_t n)
		{ return view.lookup(a, n, vm::page_allocated); });
	}
};
} // namespace metal4
