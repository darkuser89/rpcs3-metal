#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace metal4
{
// Synchronous producer -> render-owner handoff. Only drain()/owner invoke
// executes work. Pending timeout/close/wake failure retires captured owners
// without execution; running work is joined before producer return. GPU waits
// inside work therefore need their own bounded completion policy. Producers
// must release VM/passive CPU locks before waiting. This queue does not install
// page protection or make a separately retried guest store transactional.
class render_tasks final
{
	enum class phase
	{
		pending,
		running,
		completed,
		cancelling,
		cancelled
	};
	struct job
	{
		std::mutex mutex;
		std::condition_variable changed;
		phase state = phase::pending;
		std::function<void()> work;
		std::exception_ptr failure;
	};
	struct core
	{
		std::mutex mutex;
		std::condition_variable idle;
		std::deque<std::shared_ptr<job>> queue;
		std::function<void()> wake;
		std::thread::id owner;
		std::size_t running = 0;
		bool bound = false, closed = false;
	};
	std::shared_ptr<core> m_core = std::make_shared<core>();
	std::size_t m_capacity;
	std::chrono::milliseconds m_timeout;
	static void cancel_or_join(const std::shared_ptr<job>& state)
	{
		std::function<void()> retired;
		std::unique_lock lock(state->mutex);
		if (state->state == phase::pending)
		{
			state->state = phase::cancelling;
			retired.swap(state->work);
			lock.unlock();
			retired = {};
			lock.lock();
			state->state = phase::cancelled;
			lock.unlock();
			state->changed.notify_all();
		}
		else if (state->state == phase::running || state->state == phase::cancelling)
			state->changed.wait(lock, [&]
								{ return state->state == phase::completed || state->state == phase::cancelled; });
	}
	// Caller has claimed running under both core/job locks. Work owners retire
	// before completion and idle publication,so close cannot release their VM/
	// native context references while a callback or its destructor still runs.
	static void execute(const std::shared_ptr<core>& owner, const std::shared_ptr<job>& state)
	{
		std::function<void()> operation;
		{
			std::lock_guard lock(state->mutex);
			operation.swap(state->work);
		}
		std::exception_ptr failure;
		try
		{
			operation();
		}
		catch (...)
		{
			failure = std::current_exception();
		}
		operation = {};
		{
			std::lock_guard lock(state->mutex);
			state->failure = failure;
			state->state = phase::completed;
		}
		{
			std::lock_guard lock(owner->mutex);
			--owner->running;
		}
		owner->idle.notify_all();
		state->changed.notify_all();
	}

  public:
	// Wake must be a short nonblocking CPU signal; it runs under the queue lock
	// and must not reenter this queue. Up to capacity pending jobs plus one
	// outer owner callback can exist; nested owner invoke runs synchronously.
	explicit render_tasks(std::function<void()> wake, std::size_t capacity = 64,
						  std::chrono::milliseconds timeout = std::chrono::seconds(30))
		: m_capacity(capacity), m_timeout(timeout)
	{
		if (!wake || !capacity || capacity > 4096 || timeout.count() <= 0 || timeout > std::chrono::minutes(1))
			throw std::invalid_argument("Native render tasks require wake, bounded capacity and positive timeout");
		m_core->wake = std::move(wake);
	}
	~render_tasks() { close(); }
	render_tasks(const render_tasks&) = delete;
	render_tasks& operator=(const render_tasks&) = delete;
	void bind_owner()
	{
		std::lock_guard lock(m_core->mutex);
		if (m_core->bound || m_core->closed)
			throw std::logic_error("Native render-task owner is already bound or closed");
		m_core->owner = std::this_thread::get_id();
		m_core->bound = true;
	}
	[[nodiscard]] bool available() const
	{
		std::lock_guard lock(m_core->mutex);
		return m_core->bound && !m_core->closed;
	}
	[[nodiscard]] bool pending() const
	{
		std::lock_guard lock(m_core->mutex);
		return !m_core->queue.empty();
	}
	void invoke(std::function<void()> work) const
	{
		if (!work)
			throw std::invalid_argument("Cannot dispatch an empty native render task");
		const auto owner = m_core;
		auto state = std::make_shared<job>();
		state->work = std::move(work);
		work = {};
		const auto deadline = std::chrono::steady_clock::now() + m_timeout;
		bool direct = false;
		{
			std::unique_lock lock(owner->mutex);
			if (!owner->bound || owner->closed)
				throw std::logic_error("Native render-task service is unavailable");
			direct = owner->owner == std::this_thread::get_id();
			if (direct)
			{
				state->state = phase::running;
				++owner->running;
			}
			else
			{
				if (owner->queue.size() == m_capacity)
					throw std::length_error("Native render-task queue exhausted");
				owner->queue.push_back(state);
				try
				{
					owner->wake();
				}
				catch (...)
				{
					const auto failure = std::current_exception();
					owner->queue.pop_back();
					lock.unlock();
					cancel_or_join(state);
					std::rethrow_exception(failure);
				}
			}
		}
		if (direct)
			execute(owner, state);
		std::unique_lock lock(state->mutex);
		if (!state->changed.wait_until(lock, deadline, [&]
									   { return state->state == phase::completed || state->state == phase::cancelled; }))
		{
			lock.unlock();
			cancel_or_join(state);
			throw std::runtime_error("Timed out dispatching native render task");
		}
		if (state->state == phase::cancelled)
			throw std::runtime_error("Native render task cancelled before execution");
		const auto failure = state->failure;
		lock.unlock();
		if (failure)
			std::rethrow_exception(failure);
	}
	std::size_t drain(std::size_t maximum = 64)
	{
		const auto owner = m_core;
		{
			std::lock_guard lock(owner->mutex);
			if (!owner->bound || owner->owner != std::this_thread::get_id())
				throw std::logic_error("Only the bound render thread may drain native tasks");
			if (owner->running || owner->closed)
				return 0; // No recursive foreign job execution.
		}
		std::size_t consumed = 0;
		while (consumed < maximum)
		{
			std::shared_ptr<job> state;
			bool claimed = false;
			{
				std::lock_guard lock(owner->mutex);
				if (owner->queue.empty() || owner->closed)
					break;
				state = std::move(owner->queue.front());
				owner->queue.pop_front();
				std::lock_guard task_lock(state->mutex);
				if (state->state == phase::pending)
				{
					state->state = phase::running;
					++owner->running;
					claimed = true;
				}
			}
			++consumed;
			if (claimed)
				execute(owner, state);
		}
		return consumed;
	}
	// May run on another thread and joins active work. Closing/destroying the
	// owner from inside its own callback is forbidden; its resources are live.
	void close()
	{
		const auto owner = m_core;
		std::deque<std::shared_ptr<job>> retired;
		std::function<void()> wake;
		{
			std::unique_lock lock(owner->mutex);
			if (owner->running && owner->owner == std::this_thread::get_id())
				throw std::logic_error("Native render service cannot close inside its owner callback");
			owner->closed = true;
			wake.swap(owner->wake);
			retired.swap(owner->queue);
		}
		for (const auto& state : retired)
			cancel_or_join(state);
		wake = {};
		std::unique_lock lock(owner->mutex);
		owner->idle.wait(lock, [&]
						 { return !owner->running; });
	}
};
} // namespace metal4
