#pragma once

#include "M4PhysicalBacking.h"
#include <functional>
#include <atomic>
#include <limits>
#include <optional>
#include <utility>

namespace metal4
{
// A provider excludes guest CPUs,host/DMA writers and mapping changes for
// the complete callback. Callbacks may run on a thread other than the caller.
class exclusive_guest_memory
{
  public:
	class view
	{
	  public:
		virtual ~view() = default;
		[[nodiscard]] virtual std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range) { return {}; }
		[[nodiscard]] virtual std::byte* map(guest_surface_range range, bool write) = 0;
		// Register exact active-byte physical watches in the same suspended
		// capture as cache input. Providers without CPU overlays remain legacy
		// fixtures;the real VM provider owns weak registration lifetimes.
		[[nodiscard]] virtual bool uses_cpu_protection() const noexcept { return false; }
		virtual void watch_ranges(std::span<const guest_surface_range>, const std::shared_ptr<void>&) {}
		// Own physical storage without holding VM/exclusion locks after execute.
		// Release must be safe on a completion thread,after provider destruction.
		// Empty means unsupported and is rejected by guarded_guest_memory.
		[[nodiscard]] virtual std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) { return {}; }
		// Exact immutable slices of one logical range,owned independently of VM
		// allocation handles. Unsupported providers refuse production DMA prep.
		[[nodiscard]] virtual std::shared_ptr<const physical_backing> retain_segments(guest_surface_range) { return {}; }
		// CPU metadata only,inside the same provider exclusion as journal drain.
		// Unsupported/absent/over-budget current backing refuses the whole batch;
		// never compare a retained store against its possibly reused guest VA.
		[[nodiscard]] virtual bool overlaps_backing(const physical_backing& written, guest_surface_range current,
			std::size_t segment_limit)
		{
			const auto backing = retain_segments(current);
			if (!backing || backing->range().address != current.address || backing->range().size != current.size ||
				backing->segments().size() > segment_limit)
				throw std::runtime_error("Physical host reconciliation requires exact bounded current slices");
			return written.overlaps(*backing);
		}
		[[nodiscard]] virtual bool overlaps_identity(const physical_identity& kept, guest_surface_range current,
			std::size_t segment_limit)
		{
			const auto backing = retain_segments(current);
			if (!backing || backing->range().address != current.address || backing->range().size != current.size ||
				backing->segments().size() > segment_limit)
				throw std::runtime_error("Physical revisions require exact bounded current slices");
			return kept.overlaps(*backing);
		}
		// Plan ALL current virtual aliases of the active physical slices before
		// mutation. written() is called only on these still-current aliases under
		// the same exclusion,including protected pages. An empty set means none;
		// unsupported providers must throw rather than omit reservations.
		[[nodiscard]] virtual std::vector<guest_surface_range> physical_write_aliases(
			std::span<const std::shared_ptr<const physical_backing>> written, std::size_t)
		{
			if (written.empty()) return {};
			throw std::logic_error("Provider lacks physical write reservation planning");
		}
		// Include physical aliases,not only equal guest virtual addresses.
		[[nodiscard]] virtual bool same_storage(guest_surface_range a, guest_surface_range b) = 0;
		[[nodiscard]] virtual std::optional<guest_surface_range> rebase_storage(guest_surface_range source, guest_surface_range request)
		{
			validate_guest_surface_range(source);
			validate_guest_surface_range(request);
			if (request.address < source.address || request.size > source.size ||
				std::uint64_t(request.address) - source.address > source.size - request.size)
				return {};
			return request;
		}
		[[nodiscard]] virtual bool equivalent_storage(guest_surface_range a, guest_surface_range b)
		{
			return a.address == b.address && a.size == b.size;
		}
		// Reservation/invalidation notifications must not fail after mutation.
		virtual void written(guest_surface_range range) noexcept = 0;
	};
	virtual ~exclusive_guest_memory() = default;
	virtual void execute(const std::function<void(view&)>& operation) = 0;
};

// Thread-local reuse of the exact view installed by a provider's outer
// callback. Nested reads must not suspend CPUs or recursively lock the VM.
// Providers must consult current() BEFORE acquiring any exclusion locks.
class guest_memory_epoch final
{
	inline static std::atomic<std::uint64_t> s_next{1};
	inline static thread_local std::uint64_t s_identity = 0;
	inline static thread_local const exclusive_guest_memory* s_owner = nullptr;
	inline static thread_local exclusive_guest_memory::view* s_view = nullptr;

  public:
	[[nodiscard]] static bool active() noexcept { return s_owner != nullptr; }
	[[nodiscard]] static std::uint64_t identity(const exclusive_guest_memory& owner) noexcept
	{ return s_owner == &owner ? s_identity : 0; }
	[[nodiscard]] static exclusive_guest_memory::view* current(const exclusive_guest_memory& owner)
	{
		if (s_owner && s_owner != &owner)
			throw std::logic_error("Nested guest memory epochs use different providers");
		return s_view;
	}
	guest_memory_epoch(const exclusive_guest_memory& owner, exclusive_guest_memory::view& memory)
	{
		if (s_owner)
			throw std::logic_error("Guest memory epoch already installed");
		auto identity = s_next.load(std::memory_order_relaxed);
		for (;;)
		{
			if (identity == std::numeric_limits<std::uint64_t>::max())
				throw std::overflow_error("Guest memory epoch identity exhausted");
			if (s_next.compare_exchange_weak(identity, identity + 1, std::memory_order_relaxed)) break;
		}
		s_identity = identity;
		s_owner = &owner;
		s_view = &memory;
	}
	~guest_memory_epoch()
	{
		s_identity = 0;
		s_view = nullptr;
		s_owner = nullptr;
	}
	guest_memory_epoch(const guest_memory_epoch&) = delete;
	guest_memory_epoch& operator=(const guest_memory_epoch&) = delete;
};

} // namespace metal4
