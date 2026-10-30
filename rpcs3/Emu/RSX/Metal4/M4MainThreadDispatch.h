#pragma once

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace metal4
{
// Owns posted closure state,never posts references to stack wake-up objects.
// Pending timeout/post failure cancels before execution. Already-running UI
// work must finish before invoke returns: callers may capture short-lived
// frame references. UI work must never wait on GPU/guest CPU/render-thread work.
class main_thread_dispatch final
{
	enum class phase
	{
		pending,
		running,
		completed,
		cancelled
	};
	struct job
	{
		std::mutex mutex;
		std::condition_variable changed;
		phase state = phase::pending;
		std::function<void()> work;
		std::exception_ptr error;
	};
	std::function<void(std::function<void()>)> m_post;
	std::function<bool()> m_is_main;
	std::chrono::milliseconds m_timeout;
	static void disarm_or_join(const std::shared_ptr<job>& state)
	{
		std::unique_lock lock(state->mutex);
		if (state->state == phase::pending)
		{
			state->state = phase::cancelled;
			state->work = {};
			state->changed.notify_all();
		}
		else if (state->state == phase::running)
			state->changed.wait(lock, [&]
								{ return state->state == phase::completed; });
	}

  public:
	main_thread_dispatch(std::function<void(std::function<void()>)> post, std::function<bool()> is_main,
						 std::chrono::milliseconds timeout = std::chrono::seconds(5))
		: m_post(std::move(post)), m_is_main(std::move(is_main)), m_timeout(timeout)
	{
		if (!m_post || !m_is_main || timeout.count() <= 0 || timeout > std::chrono::minutes(1))
			throw std::invalid_argument("Native frame requires a main-thread dispatcher/probe and positive timeout");
	}
	void invoke(std::function<void()> work) const
	{
		if (!work)
			throw std::invalid_argument("Cannot dispatch an empty frame operation");
		if (m_is_main())
		{
			work();
			return;
		}
		auto state = std::make_shared<job>();
		state->work = std::move(work);
		work = {}; // libc++ may retain a callable in a moved-from std::function.
		const auto run = [state, probe = m_is_main]
		{
			std::function<void()> operation;
			{
				std::lock_guard lock(state->mutex);
				if (state->state != phase::pending)
					return;
				state->state = phase::running;
				operation = std::move(state->work);
				state->work = {};
			}
			std::exception_ptr error;
			try
			{
				if (!probe())
					throw std::logic_error("Frame operation was dispatched to the wrong thread");
				operation();
			}
			catch (...)
			{
				error = std::current_exception();
			}
			operation = {}; // Release closure owners before publishing completion.
			{
				std::lock_guard lock(state->mutex);
				state->error = error;
				state->state = phase::completed;
			}
			state->changed.notify_all();
		};
		const auto deadline = std::chrono::steady_clock::now() + m_timeout;
		try
		{
			m_post(run);
		}
		catch (...)
		{
			const auto error = std::current_exception();
			disarm_or_join(state);
			std::rethrow_exception(error);
		}
		std::unique_lock lock(state->mutex);
		if (!state->changed.wait_until(lock, deadline, [&]
									   { return state->state == phase::completed; }))
		{
			lock.unlock();
			disarm_or_join(state);
			throw std::runtime_error("Timed out dispatching native frame operation");
		}
		const auto error = state->error;
		lock.unlock();
		if (error)
			std::rethrow_exception(error);
	}
};
} // namespace metal4
