#pragma once

#include <algorithm>
#include <chrono>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace metal4
{
// Admission tickets cover queued AND immediate host work through the actual
// memory instruction. Captures close admission,drain existing work and retain
// exclusion for their complete CPU callback. This gate covers only writers
// explicitly routed through it;it does not install VM protection/CPU hooks.
class host_writer_gate final
{
  public:
	struct transfer
	{
		const void* source = nullptr;
		void* destination = nullptr;
		std::uint64_t bytes = 0;
	};
  private:
	enum class phase { open, draining, exclusive };
	struct core
	{
		std::mutex mutex;
		std::recursive_mutex captures;
		std::condition_variable changed;
		phase state = phase::open;
		std::thread::id owner;
		std::size_t pending = 0, active = 0, faults = 0;
	};
	struct writer_state
	{
		std::shared_ptr<core> gate;
		transfer extent;
		bool counted = false, active = false, suspended = false, finished = false;
		writer_state(std::shared_ptr<core> gate, transfer extent) : gate(std::move(gate)), extent(extent) {}
		~writer_state()
		{
			std::lock_guard lock(gate->mutex);
			if (counted && !finished) --gate->pending; // Cancel an unstarted queue ticket.
			gate->changed.notify_all();
		}
	};
	std::shared_ptr<core> m_core = std::make_shared<core>();

  public:
	class ticket;
	class fault_pause;
	// Guards are thread-affine and must retire in reverse acquisition order.
	// They must never be retained by GPU submissions or completion handlers.
	class exclusion
	{
		friend class host_writer_gate;
		std::shared_ptr<core> m_core;
		std::unique_lock<std::recursive_mutex> m_serial;
		phase m_previous;
		std::thread::id m_owner;
		exclusion(std::shared_ptr<core> gate, std::unique_lock<std::recursive_mutex> serial,
			phase previous, std::thread::id owner)
			: m_core(std::move(gate)), m_serial(std::move(serial)), m_previous(previous), m_owner(owner) {}
	  public:
		exclusion(exclusion&& other) noexcept
			: m_core(std::exchange(other.m_core, {})), m_serial(std::move(other.m_serial)),
			m_previous(other.m_previous), m_owner(other.m_owner) {}
		exclusion(const exclusion&) = delete;
		exclusion& operator=(const exclusion&) = delete;
		~exclusion()
		{
			if (!m_core) return;
			std::lock_guard lock(m_core->mutex);
			m_core->state = m_previous;
			m_core->owner = m_owner;
			m_core->changed.notify_all();
		}
	};
	class writer_scope
	{
		friend class host_writer_gate;
		friend class ticket;
		inline static thread_local writer_scope* current = nullptr;
		std::shared_ptr<writer_state> m_writer;
		writer_scope* m_previous = nullptr;
		explicit writer_scope(std::shared_ptr<writer_state> writer) : m_writer(std::move(writer))
		{
			if (!m_writer) return;
			auto& gate = *m_writer->gate;
			std::unique_lock lock(gate.mutex);
			if (m_writer->finished || m_writer->active || m_writer->suspended ||
				(gate.state == phase::exclusive && gate.owner == std::this_thread::get_id()))
			{
				lock.unlock();
				m_writer.reset();
				return;
			}
			gate.changed.wait(lock, [&] { return gate.state != phase::exclusive; });
			if (m_writer->finished || m_writer->active || m_writer->suspended)
			{
				lock.unlock(); m_writer.reset(); return;
			}
			m_writer->active = true;
			++gate.active;
			m_previous = std::exchange(current, this);
		}
	  public:
		writer_scope(const writer_scope&) = delete;
		writer_scope& operator=(const writer_scope&) = delete;
		~writer_scope()
		{
			if (!m_writer) return;
			current = m_previous;
			auto& gate = *m_writer->gate;
			std::lock_guard lock(gate.mutex);
			if (m_writer->active) --gate.active;
			if (m_writer->suspended) --gate.faults;
			--gate.pending;
			m_writer->active = m_writer->suspended = false;
			m_writer->finished = true;
			gate.changed.notify_all();
		}
		[[nodiscard]] explicit operator bool() const noexcept { return bool(m_writer); }
	};
	class ticket
	{
		friend class host_writer_gate;
		std::shared_ptr<writer_state> m_writer;
		explicit ticket(std::shared_ptr<writer_state> writer) : m_writer(std::move(writer)) {}
	  public:
		ticket() = default;
		ticket(ticket&&) = default;
		ticket& operator=(ticket&&) = default;
		ticket(const ticket&) = delete;
		[[nodiscard]] explicit operator bool() const noexcept { return bool(m_writer); }
		[[nodiscard]] writer_scope activate() const { return writer_scope{m_writer}; }
	};
	class fault_context
	{
		friend class host_writer_gate;
		friend class fault_pause;
		inline static thread_local writer_state* current = nullptr;
		std::shared_ptr<writer_state> m_writer;
		writer_state* m_previous = nullptr;
		explicit fault_context(std::shared_ptr<writer_state> writer) : m_writer(std::move(writer))
		{
			std::unique_lock lock(m_writer->gate->mutex);
			if (!m_writer->suspended || m_writer->finished)
			{
				lock.unlock(); m_writer.reset(); return;
			}
			m_previous = std::exchange(current, m_writer.get());
		}
	  public:
		fault_context(const fault_context&) = delete;
		fault_context& operator=(const fault_context&) = delete;
		~fault_context() { if (m_writer) current = m_previous; }
		[[nodiscard]] explicit operator bool() const noexcept { return bool(m_writer); }
	};
	class fault_pause
	{
		friend class host_writer_gate;
		std::array<std::shared_ptr<writer_state>, 64> m_writers;
		std::size_t m_count;
		fault_pause(std::array<std::shared_ptr<writer_state>, 64> writers, std::size_t count)
			: m_writers(std::move(writers)), m_count(count) {}
	  public:
		fault_pause(fault_pause&& other) noexcept
			: m_writers(std::move(other.m_writers)), m_count(std::exchange(other.m_count, 0)) {}
		fault_pause(const fault_pause&) = delete;
		fault_pause& operator=(const fault_pause&) = delete;
		~fault_pause()
		{
			if (!m_count) return;
			auto& gate = *m_writers[0]->gate;
			std::unique_lock lock(gate.mutex);
			// Reacquire exclusion BEFORE returning to the faulting instruction.
			// Normal captures still see its pending ticket until that store ends.
			gate.changed.wait(lock, [&] { return gate.state != phase::exclusive; });
			for (std::size_t i = 0; i < m_count; ++i)
			{
				const auto& writer = m_writers[i];
				if (writer->finished) continue;
				--gate.faults; ++gate.active;
				writer->suspended = false;
				writer->active = true;
			}
			gate.changed.notify_all();
		}
		[[nodiscard]] fault_context capture_context() const { return fault_context{m_writers[0]}; }
	};

	host_writer_gate() = default;
	host_writer_gate(const host_writer_gate&) = delete;
	host_writer_gate& operator=(const host_writer_gate&) = delete;
	[[nodiscard]] ticket admit(transfer extent = {nullptr, nullptr, 0}) const
	{
		const auto gate = m_core;
		std::unique_lock lock(gate->mutex);
		const auto own_writer = [&]
		{
			for (auto* scope = writer_scope::current; scope; scope = scope->m_previous)
				if (scope->m_writer->gate == gate && scope->m_writer->active) return true;
			return false;
		};
		// Reentrant callbacks may enqueue while a capture drains their parent.
		// Issuing work inside an exclusive callback is rejected,never self-waited.
		if (gate->state == phase::exclusive && gate->owner == std::this_thread::get_id()) return {};
		gate->changed.wait(lock, [&]
		{
			return gate->state == phase::open ||
				(gate->state == phase::draining && (gate->owner == std::this_thread::get_id() || own_writer()));
		});
		if (gate->pending == std::numeric_limits<std::size_t>::max()) return {};
		auto writer = std::make_shared<writer_state>(gate, extent);
		++gate->pending; writer->counted = true;
		return ticket{std::move(writer)};
	}
	[[nodiscard]] std::optional<transfer> current_transfer() const
	{
		for (auto* scope = writer_scope::current; scope; scope = scope->m_previous)
		{
			const auto writer = scope->m_writer;
			if (writer->gate != m_core) continue;
			std::lock_guard lock(m_core->mutex);
			if (writer->active && writer->extent.bytes) return writer->extent;
		}
		return {};
	}
	[[nodiscard]] std::optional<fault_pause> pause_current_fault() const
	{
		// Bound fault-stack bookkeeping without allocating in the fault path.
		// An over-depth stack is rejected before any writer changes state.
		std::array<std::shared_ptr<writer_state>, 64> blocked;
		std::size_t count = 0;
		std::lock_guard lock(m_core->mutex);
		for (auto* scope = writer_scope::current; scope; scope = scope->m_previous)
		{
			const auto writer = scope->m_writer;
			if (writer->gate != m_core || !writer->active) continue;
			if (count == blocked.size()) return {}; // Reject before changing any scope.
			blocked[count++] = writer;
		}
		if (!count) return {};
		// A callback's parent writer is blocked by the same fault stack too.
		// Pausing only its innermost copy would deadlock owner capture on parent.
		for (std::size_t i = 0; i < count; ++i)
		{
			const auto& writer = blocked[i];
			writer->active = false; writer->suspended = true;
			--m_core->active; ++m_core->faults;
		}
		m_core->changed.notify_all();
		return fault_pause{std::move(blocked), count};
	}
	// Pump runs without the gate mutex;it may service a fault on this caller
	// and acquire a nested fault capture. It must return within its own bound.
	// A normal capture refuses fault-blocked work rather than waiting for itself.
	template <typename Pump>
	[[nodiscard]] std::optional<exclusion> acquire(Pump&& pump,
		std::chrono::milliseconds timeout = std::chrono::seconds(30)) const
	{
		if (timeout.count() < 0) return {};
		const auto gate = m_core;
		{
			std::lock_guard lock(gate->mutex);
			// Check before waiting for another capture's serial lock:that
			// capture may itself be draining this caller's active instruction.
			for (auto* scope = writer_scope::current; scope; scope = scope->m_previous)
				if (scope->m_writer->gate == gate && scope->m_writer->active) return {};
		}
		std::unique_lock serial(gate->captures);
		std::unique_lock lock(gate->mutex);
		const auto previous = gate->state;
		const auto owner = gate->owner;
		if (previous == phase::exclusive)
			return exclusion{gate, std::move(serial), previous, owner};
		const auto fault = fault_context::current;
		const bool recovering = fault && fault->gate == gate && fault->suspended && !fault->finished;
		if (!recovering)
			for (auto* scope = writer_scope::current; scope; scope = scope->m_previous)
				if (scope->m_writer->gate == gate && scope->m_writer->active) return {};
		gate->state = phase::draining;
		gate->owner = std::this_thread::get_id();
		struct rollback
		{
			core& gate;
			std::unique_lock<std::mutex>& lock;
			phase previous;
			std::thread::id owner;
			bool armed = true;
			~rollback()
			{
				if (!armed) return;
				if (!lock.owns_lock()) lock.lock();
				gate.state = previous; gate.owner = owner; gate.changed.notify_all();
			}
		} restore{*gate, lock, previous, owner};
		const auto deadline = std::chrono::steady_clock::now() + timeout;
		while (recovering ? gate->active != 0 : gate->pending != 0)
		{
			if ((!recovering && gate->faults) || (recovering && (!fault->suspended || fault->finished)) ||
				std::chrono::steady_clock::now() >= deadline) return {};
			lock.unlock();
			std::invoke(pump);
			lock.lock();
			gate->changed.wait_for(lock, std::chrono::milliseconds(1));
		}
		if (recovering && (!fault->suspended || fault->finished)) return {};
		gate->state = phase::exclusive;
		restore.armed = false;
		return exclusion{gate, std::move(serial), previous, owner};
	}
	// Real provider captures can arrive while another producer is suspended
	// in owner fault service/preparation. Release capture serialization on that
	// refusal,pump the queued owner job,then retry after the producer resumes.
	// Never wait on this caller's own unserviced active/suspended instruction.
	template <typename Pump>
	[[nodiscard]] std::optional<exclusion> acquire_waiting_faults(Pump&& pump,
		std::chrono::milliseconds timeout = std::chrono::seconds(30)) const
	{
		if (timeout.count() < 0) return {};
		const auto deadline = std::chrono::steady_clock::now() + timeout;
		for (;;)
		{
			const auto remaining = std::max(std::chrono::milliseconds::zero(),
				std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()));
			auto result = acquire(pump, remaining);
			if (result) return result;
			{
				std::lock_guard lock(m_core->mutex);
				// The fault can clear AFTER acquire refused and BEFORE this
				// check. Retry even if faults is now zero;its producer may
				// already be active,or have completed while we released serial.
				if (std::chrono::steady_clock::now() >= deadline) return {};
				for (auto* scope = writer_scope::current; scope; scope = scope->m_previous)
					if (scope->m_writer->gate == m_core && (scope->m_writer->active || scope->m_writer->suspended)) return {};
			}
			std::invoke(pump); // No gate/serial/VM lock held during handoff.
		}
	}
};
} // namespace metal4
