#pragma once

#include "M4GuardedGuestMemory.h"

namespace metal4
{
// Preparation has flushed GPU aliases and leased bytes under the provider.
// start() acquires NONBLOCKING mapping exclusion and arms the journal only
// after the caller acquires its source reservation. The exclusion covers the
// actual instruction through commit. Fault service must drop both locks and
// use paused capture_context before entering a provider;resume revalidates.
class host_store_transaction final
{
	struct state : std::enable_shared_from_this<state>
	{
		std::shared_ptr<guarded_guest_memory> memory;
		std::optional<host_write_journal::preparation> destination, source;
		std::function<std::shared_ptr<void>()> acquire_mapping;
		std::shared_ptr<void> mapping;
		std::optional<host_write_intent> completion;
		state* previous = nullptr;
		bool active = false, paused = false;
	};
	inline static thread_local state* s_current = nullptr;
	static bool start(void* opaque)
	{
		auto& value = *static_cast<state*>(opaque);
		if (s_current || value.active) throw std::logic_error("Nested native memory instructions cannot recursively own the VM lock");
		value.mapping = value.acquire_mapping();
		if (!value.mapping) return false; // Caller releases reservations and re-prepares before bounded retry.
		try
		{
			if (value.destination) value.completion.emplace(value.memory->begin_host_write(*value.destination, value.source));
			else value.completion.emplace(value.memory->begin_host_read(*value.source));
		}
		catch (...) { value.mapping.reset(); throw; }
		value.previous = std::exchange(s_current, &value);
		value.active = true;
		return true;
	}
	static void finish(void* opaque, std::size_t, bool completed) noexcept
	{
		auto& value = *static_cast<state*>(opaque);
		if (value.active)
		{
			// A failed/unresumed fault must never be relabeled as a successful
			// store. The journal retains physical dirty bytes and ambiguity.
			if (completed && !value.paused && value.mapping) value.completion->commit();
			value.completion.reset();
			value.mapping.reset();
			s_current = value.previous;
			value.active = false;
		}
	}
  public:
	enum class resume_result { ready, busy, changed, inactive };
	class fault_pause
	{
		friend class host_store_transaction;
		std::shared_ptr<state> m_state;
		explicit fault_pause(state& value) : m_state(value.shared_from_this())
		{
			if (value.destination) value.memory->mark_host_write_fault(*value.completion);
			value.mapping.reset(); // BEFORE owner/provider or MM ordering work.
			value.paused = true;
		}
	public:
		fault_pause(const fault_pause&) = delete;
		fault_pause& operator=(const fault_pause&) = delete;
		fault_pause(fault_pause&& other) noexcept : m_state(std::move(other.m_state)) {}
		// No automatic retry on unwind. The caller must stop the faulting
		// instruction unless resume succeeded under the original mapping.
		template <typename Operation>
		void service(Operation&& operation) const
		{
			if (!m_state || !m_state->active || !m_state->paused) throw std::logic_error("Native fault service lost its suspended transaction");
			m_state->memory->with_host_write_recovery(std::forward<Operation>(operation));
		}
		[[nodiscard]] resume_result attempt_resume()
		{
			if (!m_state || !m_state->active || !m_state->paused) return resume_result::inactive;
			auto mapping = m_state->acquire_mapping();
			if (!mapping) return resume_result::busy;
			try
			{
				if (m_state->destination) m_state->memory->validate_host_access(*m_state->destination);
				if (m_state->source) m_state->memory->validate_host_access(*m_state->source);
			}
			catch (...) { return resume_result::changed; }
			m_state->mapping = std::move(mapping);
			m_state->paused = false;
			return resume_result::ready;
		}
		[[nodiscard]] bool resume() { return attempt_resume() == resume_result::ready; }
	};
	[[nodiscard]] static host_write_intent prepare(std::shared_ptr<guarded_guest_memory> memory,
		std::optional<host_write_journal::preparation> destination,
		std::optional<host_write_journal::preparation> source,
		std::function<std::shared_ptr<void>()> acquire_mapping)
	{
		if (!memory || (!destination && !source) || !acquire_mapping)
			throw std::invalid_argument("Native host instruction requires memory,prepared bytes and mapping exclusion");
		auto value = std::make_shared<state>();
		value->memory = std::move(memory);
		value->destination = std::move(destination);
		value->source = std::move(source);
		value->acquire_mapping = std::move(acquire_mapping);
		return host_write_intent{std::move(value), 0, finish, start};
	}
	[[nodiscard]] static std::optional<fault_pause> pause_current(const guarded_guest_memory& memory)
	{
		if (!s_current) return {};
		if (s_current->memory.get() != &memory || s_current->paused)
			throw std::logic_error("Native memory fault uses another provider or an already suspended instruction");
		return fault_pause{*s_current};
	}
};
} // namespace metal4
