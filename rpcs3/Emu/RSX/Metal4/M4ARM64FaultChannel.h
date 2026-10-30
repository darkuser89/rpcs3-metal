#pragma once

#include "M4DarwinARM64Context.h"
#include <atomic>
#include <functional>
#include <memory>
#include <thread>

namespace metal4
{
// Register each original thread outside signal context. Its stable slot owns
// all request/reply bytes. Signal-side exchange uses only fixed copies and
// lock-free atomics;it never invokes the render queue,VM,GPU or an allocator.
// The render owner must pump drain(),and close before retiring its services.
// This transport does not release/reacquire guest CPU/passive VM locks,install
// a handler or establish an immutable instruction-code lease.
class arm64_fault_channel final
{
	enum phase : unsigned { vacant, idle, capturing, pending, running, completed, unsupported, failed, cancelled };
	struct slot
	{
		std::atomic<unsigned> state{vacant};
		arm64_memory_state before, after;
		std::uint64_t fault = 0;
		bool writing = false;
	};
	struct core
	{
		const std::thread::id owner = std::this_thread::get_id();
		const std::uint64_t first, size;
		const std::size_t capacity;
		std::unique_ptr<slot[]> slots;
		std::atomic<bool> closed{false}, draining{false};
		core(std::uint64_t address, std::uint64_t bytes, std::size_t count)
			: first(address), size(bytes), capacity(count), slots(std::make_unique<slot[]>(count)) {}
	};
	const std::shared_ptr<core> m_core;
	inline static thread_local slot* s_current = nullptr;
	static std::shared_ptr<core> create(std::uint64_t first, std::uint64_t size, std::size_t capacity)
	{
		if (!size || size > std::numeric_limits<std::uint64_t>::max() - first || !capacity || capacity > 256)
			throw std::invalid_argument("Native ARM fault channel requires bounded address and slot ranges");
		return std::make_shared<core>(first, size, capacity);
	}
	static_assert(std::atomic<unsigned>::is_always_lock_free && std::atomic<bool>::is_always_lock_free);
  public:
	enum class outcome { unowned, completed, unsupported, failed, cancelled };
	struct input
	{
		arm64_memory_state before;
		std::uint64_t fault;
		bool writing;
	};
	struct response
	{
		outcome status;
		arm64_memory_state reply{};
	};
	class client final
	{
		friend class arm64_fault_channel;
		const std::shared_ptr<core> m_core;
		slot* const m_slot;
		const bool m_on_owner;
		static slot* attach(const std::shared_ptr<core>& owner)
		{
			if (s_current || owner->closed.load(std::memory_order_acquire))
				throw std::logic_error("Native ARM fault thread already registered or channel closed");
			for (std::size_t i = 0; i < owner->capacity; ++i)
			{
				auto& value = owner->slots[i];
				unsigned empty = vacant;
				if (!value.state.compare_exchange_strong(empty, idle, std::memory_order_acq_rel)) continue;
				if (owner->closed.load(std::memory_order_acquire))
				{
					value.state.store(vacant, std::memory_order_release);
					throw std::logic_error("Native ARM fault registration raced channel close");
				}
				s_current = &value;
				return &value;
			}
			throw std::length_error("Native ARM fault channel slot budget exhausted");
		}
		explicit client(std::shared_ptr<core> owner)
			: m_core(std::move(owner)), m_slot(attach(m_core)), m_on_owner(m_core->owner == std::this_thread::get_id()) {}
	  public:
		client(const client&) = delete;
		client& operator=(const client&) = delete;
		~client()
		{
			// Destruction belongs to the original thread after all its handlers
			// returned. The shared core outlives channel shutdown through this lease.
			if (s_current != m_slot || m_slot->state.load(std::memory_order_acquire) != idle) std::terminate();
			s_current = nullptr;
			m_slot->state.store(vacant, std::memory_order_release);
		}
		[[nodiscard]] response exchange(const arm64_memory_state& before, std::uint64_t fault, bool writing) noexcept
		{
			if (s_current != m_slot || m_on_owner) return {outcome::failed};
			if (fault < m_core->first || fault - m_core->first >= m_core->size) return {outcome::unowned};
			unsigned ready = idle;
			if (!m_slot->state.compare_exchange_strong(ready, capturing, std::memory_order_seq_cst)) return {outcome::failed};
			if (m_core->closed.load(std::memory_order_seq_cst))
			{
				m_slot->state.store(idle, std::memory_order_release);
				return {outcome::cancelled};
			}
			m_slot->before = before; m_slot->fault = fault; m_slot->writing = writing;
			m_slot->state.store(pending, std::memory_order_release);
			unsigned state;
			while ((state = m_slot->state.load(std::memory_order_acquire)) == pending || state == running) asm volatile("yield");
			response result{outcome::failed};
			if (state == completed) result = {outcome::completed, m_slot->after};
			else if (state == unsupported) result.status = outcome::unsupported;
			else if (state == cancelled) result.status = outcome::cancelled;
			m_slot->state.store(idle, std::memory_order_release);
			return result;
		}
		[[nodiscard]] outcome handle(ucontext_t& context, std::uint64_t fault) noexcept
		{
			// Unrelated faults remain unowned even when their context/reason
			// is outside the normal data-abort classes serviced by this channel.
			if (fault < m_core->first || fault - m_core->first >= m_core->size) return outcome::unowned;
			const auto before = capture_arm64_context(context);
			if (!before) return outcome::failed;
			const auto syndrome = context.uc_mcontext->__es.__esr;
			const auto kind = syndrome >> 26;
			if (kind != 0x24 && kind != 0x25) return outcome::unsupported;
			const auto result = exchange(*before, fault, (syndrome & 64) != 0);
			if (result.status != outcome::completed) return result.status;
			return publish_arm64_context(context, *before, result.reply) ? outcome::completed : outcome::failed;
		}
	};
	explicit arm64_fault_channel(std::uint64_t first, std::uint64_t size, std::size_t capacity = 64)
		: m_core(create(first, size, capacity)) {}
	~arm64_fault_channel() { close(); }
	arm64_fault_channel(const arm64_fault_channel&) = delete;
	arm64_fault_channel& operator=(const arm64_fault_channel&) = delete;
	[[nodiscard]] client attach() const { return client{m_core}; }
	[[nodiscard]] bool open() const noexcept { return !m_core->closed.load(std::memory_order_acquire); }
	[[nodiscard]] bool has_pending() const noexcept
	{
		for (std::size_t i = 0; i < m_core->capacity; ++i)
			if (m_core->slots[i].state.load(std::memory_order_acquire) == pending) return true;
		return false;
	}
	// Only the render owner executes callbacks. Input is an owned local copy,
	// not a pointer into the original signal frame. Exceptions become terminal
	// failures;no protected hardware retry is authorized by any non-success.
	template <typename Operation>
	std::size_t drain(Operation&& operation)
	{
		const auto owner = m_core;
		if (std::this_thread::get_id() != owner->owner || owner->draining.exchange(true, std::memory_order_acq_rel))
			throw std::logic_error("Native ARM fault draining requires its non-reentrant render owner");
		struct reset { std::atomic<bool>& value; ~reset() { value.store(false, std::memory_order_release); } } cleanup{owner->draining};
		// Own rvalue callback captures and retire them before publishing idle.
		// In particular,foreign close must join their destructors as well.
		auto work = std::forward<Operation>(operation);
		std::size_t count = 0;
		for (std::size_t i = 0; i < owner->capacity && !owner->closed.load(std::memory_order_acquire); ++i)
		{
			auto& value = owner->slots[i];
			unsigned waiting = pending;
			if (!value.state.compare_exchange_strong(waiting, running, std::memory_order_acquire)) continue;
			const input request{value.before, value.fault, value.writing};
			unsigned result = failed;
			try
			{
				if (const auto reply = std::invoke(work, request)) { value.after = *reply; result = completed; }
				else result = unsupported;
			}
			catch (...) { result = failed; }
			value.state.store(result, std::memory_order_release);
			++count;
		}
		return count;
	}
	void close()
	{
		const auto owner = m_core;
		if (owner->owner == std::this_thread::get_id() && owner->draining.load(std::memory_order_acquire))
			throw std::logic_error("Native ARM fault owner cannot close its running callback");
		// Admission and this scan share one sequentially consistent order.
		// Otherwise close could miss capturing while that producer misses closed.
		owner->closed.store(true, std::memory_order_seq_cst);
		for (std::size_t i = 0; i < owner->capacity; ++i)
		{
			auto& value = owner->slots[i];
			for (;;)
			{
				auto state = value.state.load(std::memory_order_seq_cst);
				if (state == capturing || state == running) { std::this_thread::yield(); continue; }
				if (state == pending && !value.state.compare_exchange_strong(state, cancelled, std::memory_order_acq_rel)) continue;
				break;
			}
		}
		while (owner->draining.load(std::memory_order_acquire)) std::this_thread::yield();
	}
};
} // namespace metal4
