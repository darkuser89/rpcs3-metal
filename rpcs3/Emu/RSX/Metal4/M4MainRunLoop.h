#pragma once

#import <Foundation/NSThread.h>
#import <CoreFoundation/CFRunLoop.h>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace metal4
{
// Main-thread event-loop observation, with no timer, posted per-frame job or
// renderer wait. Callbacks must own their data (or lock a weak owner); they may
// finish after invalidation and must never borrow a frame/context/stack object.
class main_run_loop_observer final
{
	struct state
	{
		std::mutex mutex;
		std::shared_ptr<const std::function<void()>> sample;
		std::exception_ptr error;
		bool active = true, running = false;
	};
	std::shared_ptr<state> m_state;
	CFRunLoopObserverRef m_observer = nullptr;
	struct callback_owner
	{
		CFRunLoopObserverRef observer;
		explicit callback_owner(CFRunLoopObserverRef value) : observer(value) { CFRetain(observer); }
		~callback_owner() { CFRelease(observer); }
	};

  public:
	explicit main_run_loop_observer(std::function<void()> sample)
		: m_state(std::make_shared<state>())
	{
		if (![NSThread isMainThread] || !sample)
			throw std::invalid_argument("Native geometry observation requires the main thread and an owned callback");
		m_state->sample = std::make_shared<const std::function<void()>>(std::move(sample));
		const auto owner = m_state;
		m_observer = CFRunLoopObserverCreateWithHandler(kCFAllocatorDefault,
			kCFRunLoopBeforeSources | kCFRunLoopBeforeWaiting, true, 0,
			^(CFRunLoopObserverRef observer, CFRunLoopActivity)
			{
				// Invalidation can remove the CF registration while this callback
				// is running. Keep both the block and its C++ state independently
				// alive before releasing locks or entering the user's operation.
				const callback_owner keep_block(observer);
				const auto retained = owner;
				std::shared_ptr<const std::function<void()>> operation;
				{
					std::lock_guard lock(retained->mutex);
					if (!retained->active || retained->running) return;
					retained->running = true;
					operation = retained->sample;
				}
				std::exception_ptr error;
				try { (*operation)(); } catch (...) { error = std::current_exception(); }
				operation = {};
				{
					std::lock_guard lock(retained->mutex);
					retained->running = false;
					if (error)
					{
						retained->error = error;
						retained->active = false;
						retained->sample = {};
					}
				}
				if (error) CFRunLoopObserverInvalidate(observer);
			});
		if (!m_observer) throw std::runtime_error("Native geometry run-loop observer allocation failed");
		CFRunLoopAddObserver(CFRunLoopGetMain(), m_observer, kCFRunLoopCommonModes);
	}
	~main_run_loop_observer()
	{
		// Core Foundation permits cross-thread invalidation; no NSRunLoop or
		// AppKit operation is performed here. An already-running callback owns
		// its local data and cannot borrow this observer's C++ object.
		{
			std::lock_guard lock(m_state->mutex);
			m_state->active = false;
			m_state->sample = {};
		}
		CFRunLoopObserverInvalidate(m_observer);
		CFRelease(m_observer);
	}
	main_run_loop_observer(const main_run_loop_observer&) = delete;
	main_run_loop_observer& operator=(const main_run_loop_observer&) = delete;
	[[nodiscard]] bool active() const
	{
		std::lock_guard lock(m_state->mutex);
		return m_state->active;
	}
	void rethrow_error() const
	{
		std::exception_ptr error;
		{ std::lock_guard lock(m_state->mutex); error = m_state->error; }
		if (error) std::rethrow_exception(error);
	}
};
} // namespace metal4
