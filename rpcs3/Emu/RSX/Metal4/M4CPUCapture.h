#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace metal4
{
using cpu_capture_dispatch = std::function<void(const std::function<void()>&)>;

// A renderer capture must execute exactly once,on its calling RSX thread,
// before dispatch returns. Retained/foreign/duplicate callbacks cannot run
// the operation later with dangling register/layout/stack references.
inline void checked_cpu_capture(const cpu_capture_dispatch& dispatch, std::function<void()> operation)
{
	if (!dispatch || !operation)
		throw std::invalid_argument("CPU capture requires dispatch and operation");
	struct state
	{
		std::mutex mutex;
		std::thread::id owner = std::this_thread::get_id();
		std::function<void()> operation;
		std::exception_ptr failure;
		bool invoked = false, closed = false, invalid = false;
	};
	auto shared = std::make_shared<state>();
	shared->operation = std::move(operation);
	const auto invoke = [shared]
	{
		std::function<void()> work;
		{
			std::lock_guard lock(shared->mutex);
			if (shared->closed || shared->invoked || shared->owner != std::this_thread::get_id())
			{
				shared->invalid = true;
				return;
			}
			shared->invoked = true;
			work = std::move(shared->operation);
		}
		try
		{
			work();
		}
		catch (...)
		{
			std::lock_guard lock(shared->mutex);
			shared->failure = std::current_exception();
		}
	};
	try
	{
		dispatch(invoke);
	}
	catch (...)
	{
		std::lock_guard lock(shared->mutex);
		shared->closed = true;
		shared->operation = {};
		throw;
	}
	std::exception_ptr failure;
	bool valid;
	{
		std::lock_guard lock(shared->mutex);
		shared->closed = true;
		shared->operation = {};
		failure = shared->failure;
		valid = shared->invoked && !shared->invalid;
	}
	if (failure)
		std::rethrow_exception(failure);
	if (!valid)
		throw std::logic_error("CPU capture did not run exactly once synchronously on its caller thread");
}
} // namespace metal4
