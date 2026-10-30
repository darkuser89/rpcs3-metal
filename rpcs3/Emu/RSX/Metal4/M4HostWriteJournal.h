#pragma once

#include "M4MappingEpoch.h"
#include "M4HostWriteIntent.h"
#include "M4PhysicalBacking.h"
#include <functional>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace metal4
{
// Producers reserve before the actual guest store and commit immediately after
// it,while still holding their host-writer ticket. A provider drains only after
// excluding ALL writers and taking VM mapping exclusion. No GPU work/wait here.
// Ambiguous/partial/remapped records stay owned for future physical recovery.
// Prepared production entries own exact independently mapped physical slices.
// Remap/retry recovery must still be installed before activation. Retention
// preserves old bytes;it does not prove which mapping a retried store touched.
class host_write_journal final
{
	public:
	struct preparation
	{
		guest_surface_range range;
		std::uint64_t mapping;
		std::shared_ptr<const physical_backing> backing;
	};
	struct completion
	{
		guest_surface_range range;
		std::shared_ptr<const physical_backing> backing;
	};
	private:
	enum class phase { free, active, complete, ambiguous };
	struct slot
	{
		phase state = phase::free;
		guest_surface_range range{};
		std::uint64_t mapping = 0;
		std::shared_ptr<const physical_backing> backing;
		std::optional<preparation> source;
		bool partial_pending = true;
	};
	struct state
	{
		std::shared_ptr<mapping_epoch> mappings;
		std::mutex mutex;
		std::atomic<std::uint64_t> revision{1};
		void changed() noexcept
		{
			// Mutations hold mutex. Saturation disables observation reuse forever.
			const auto value = revision.load(std::memory_order_relaxed);
			if (value != std::numeric_limits<std::uint64_t>::max()) revision.store(value + 1, std::memory_order_release);
		}
		std::vector<slot> slots;
		std::vector<completion> scratch;
		state(std::shared_ptr<mapping_epoch> mappings, std::size_t capacity)
			: mappings(std::move(mappings)), slots(capacity)
		{
			scratch.reserve(capacity);
		}
	};
	std::shared_ptr<state> m_state;
	inline static thread_local const state* s_recovery = nullptr;
	static void finish(void* opaque, std::size_t index, bool completed) noexcept
	{
		auto& owner = *static_cast<state*>(opaque);
		std::lock_guard lock(owner.mutex);
		auto& entry = owner.slots[index];
		bool stable = false;
		if (completed)
		{
			try
			{
				// A finished unrelated transition does not change the destination.
				// A fresh global observation brackets the range query,so neither
				// a half-published begin stamp nor an active transition can pass.
				const auto observed = owner.mappings->observe();
				stable = observed && owner.mappings->current(entry.range) == entry.mapping &&
					(!entry.source || owner.mappings->current(entry.source->range) == entry.source->mapping) &&
					owner.mappings->unchanged(*observed);
			}
			catch (...) {} // Poison/exhaustion after the actual store stays ambiguous.
		}
		entry.state = stable ? phase::complete : phase::ambiguous;
		owner.changed();
	}

  public:
	struct observation
	{
		std::uint64_t revision;
		bool recovering;
		bool operator==(const observation&) const = default;
		[[nodiscard]] bool reusable() const noexcept { return revision != std::numeric_limits<std::uint64_t>::max(); }
	};
	// Safe only for metadata reuse under the same provider exclusion as drain.
	[[nodiscard]] observation observe() const noexcept
	{ return {m_state->revision.load(std::memory_order_acquire), s_recovery == m_state.get()}; }
	static constexpr std::size_t max_entries = 4096;
	using intent = host_write_intent;
	// Both joined 4GiB VM mirrors share guest offsets. Compare integer addresses:
	// host GPU/vector pointers are unrelated C++ objects,so pointer subtraction
	// against the VM reservation would not be a valid address-domain test.
	[[nodiscard]] static std::optional<guest_surface_range> destination(std::uintptr_t base, std::uintptr_t pointer, std::uint64_t bytes)
	{
		constexpr std::uint64_t last = 0x1'ffff'ffffull;
		if (!bytes || base > std::numeric_limits<std::uintptr_t>::max() - last || pointer - base > last) return {};
		const guest_surface_range range{static_cast<std::uint32_t>(pointer - base), bytes};
		validate_guest_surface_range(range);
		return range;
	}
	explicit host_write_journal(std::shared_ptr<mapping_epoch> mappings, std::size_t capacity = 1024)
	{
		if (!mappings || !capacity || capacity > max_entries)
			throw std::invalid_argument("Host write journal requires mapping ownership and a finite entry budget");
		m_state = std::make_shared<state>(std::move(mappings), capacity);
	}
	host_write_journal(const host_write_journal&) = delete;
	host_write_journal& operator=(const host_write_journal&) = delete;
	[[nodiscard]] intent begin_unleased(guest_surface_range range)
	{
		return begin_record(range, {}, {});
	}
	// prepare() runs under the provider;begin() only after the producer's
	// paused ticket has resumed. No active journal entry may block another
	// producer's preparation while its writer is still suspended.
	[[nodiscard]] intent begin(preparation destination, std::optional<preparation> source = {})
	{
		const auto valid = [](const preparation& value)
		{
			return value.backing && value.backing->range().address == value.range.address &&
				value.backing->range().size == value.range.size;
		};
		if (!valid(destination) || (source && !valid(*source)))
			throw std::invalid_argument("Prepared host access requires exact physical backing ownership");
		return begin_record(destination.range, std::move(destination), std::move(source));
	}
	void validate_prepared_read(const preparation& source) const
	{
		const auto observation = m_state->mappings->observe();
		if (!source.backing || source.backing->range().address != source.range.address ||
			source.backing->range().size != source.range.size || !observation ||
			m_state->mappings->current(source.range) != source.mapping || !m_state->mappings->unchanged(*observation))
			throw std::runtime_error("Prepared host source changed before the actual read");
	}
	void mark_partial(const host_write_intent& intent)
	{
		std::lock_guard lock(m_state->mutex);
		if (intent.m_owner.get() != m_state.get() || intent.m_slot >= m_state->slots.size() ||
			m_state->slots[intent.m_slot].state != phase::active)
			throw std::logic_error("Native prefix notification lost its active journal intent");
		m_state->slots[intent.m_slot].partial_pending = true;
		m_state->changed();
	}
	// Only on the render owner inside a paused host-writer capture_context.
	// The provider must exclude ALL active writers,CPUs and mapping changes.
	// Leased active stores may publish conservative prefix revisions for fault
	// service,without retiring their intents or claiming final completion.
	template <typename Operation>
	void with_incomplete_recovery(Operation&& operation)
	{
		if (s_recovery && s_recovery != m_state.get()) throw std::logic_error("Nested host recovery uses another journal");
		struct scope
		{
			const state* previous;
			~scope() { s_recovery = previous; }
		} active{std::exchange(s_recovery, m_state.get())};
		std::invoke(std::forward<Operation>(operation));
	}
	private:
	[[nodiscard]] intent begin_record(guest_surface_range range, std::optional<preparation> destination, std::optional<preparation> source)
	{
		validate_guest_surface_range(range);
		if (!range.size) throw std::invalid_argument("Host write intent requires nonempty destination bytes");
		std::lock_guard lock(m_state->mutex);
		for (std::size_t index = 0; index < m_state->slots.size(); ++index)
		{
			auto& entry = m_state->slots[index];
			if (entry.state != phase::free) continue;
			const auto observation = m_state->mappings->observe();
			if (!observation) throw std::runtime_error("Host store begins during an unfinished mapping transition");
			const auto mapping = m_state->mappings->current(range);
			if ((destination && mapping != destination->mapping) ||
				(source && m_state->mappings->current(source->range) != source->mapping))
				throw std::runtime_error("Prepared host access remapped before the actual store");
			if (!m_state->mappings->unchanged(*observation))
				throw std::runtime_error("Host store destination changed during intent preparation");
			entry.range = range;
			entry.mapping = mapping;
			entry.backing = destination ? std::move(destination->backing) : nullptr;
			entry.source = std::move(source);
			entry.state = phase::active;
			m_state->changed();
			return intent(m_state, index, finish);
		}
		throw std::length_error("Host write completion journal entry budget exhausted before store");
	}
	public:
	// Validate every event BEFORE applying any revisions. The callback plans all
	// potentially throwing work before assigning revisions. Failed application
	// retains the entire batch. Explicit paused recovery may publish active
	// prefix revisions,but keeps every active intent and its physical owners.
	template <typename Apply>
	void drain(Apply&& apply)
	{
		std::lock_guard lock(m_state->mutex);
		if (!m_state->mappings->observe())
			throw std::runtime_error("Host completion drain overlaps a mapping transition");
		auto& ranges = m_state->scratch;
		ranges.clear();
		struct clear_scratch
		{
			std::vector<completion>& values;
			~clear_scratch() { values.clear(); }
		} cleanup{ranges}; // Scratch must not retain retired physical allocations.
		for (const auto& entry : m_state->slots)
		{
			if (entry.state == phase::free) continue;
			const bool recovering = entry.state == phase::active && entry.backing && s_recovery == m_state.get();
			if (entry.state != phase::complete && !recovering)
				throw std::runtime_error("Host completion journal retains an unfinished or ambiguous guest store");
			// Commit verified BOTH original mappings before releasing instruction
			// exclusion. A later remap cannot change those completed physical bytes.
			// Unleased and paused/partial entries still need their original mapping.
			const bool physical_complete = entry.state == phase::complete && entry.backing;
			if (!physical_complete && m_state->mappings->current(entry.range) != entry.mapping)
				throw std::runtime_error("Host store destination remapped before physical alias reconciliation");
			if (!physical_complete && entry.source && m_state->mappings->current(entry.source->range) != entry.source->mapping)
				throw std::runtime_error("Host store source remapped before physical alias reconciliation");
			if (!recovering || entry.partial_pending) ranges.push_back({entry.range, entry.backing});
		}
		std::invoke(std::forward<Apply>(apply), std::span<const completion>(ranges));
		bool changed = false;
		for (auto& entry : m_state->slots)
		{
			if (entry.state == phase::complete) { entry = {}; changed = true; }
			else if (entry.state == phase::active && entry.partial_pending) { entry.partial_pending = false; changed = true; }
		}
		if (changed) m_state->changed();
	}
	[[nodiscard]] std::size_t pending() const
	{
		std::lock_guard lock(m_state->mutex);
		std::size_t count = 0;
		for (const auto& entry : m_state->slots) count += entry.state != phase::free;
		return count;
	}
};
} // namespace metal4
