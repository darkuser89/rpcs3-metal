#include "M4MainThreadDispatch.h"

#include <array>
#include <atomic>
#include <deque>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void rejects(F&& fn)
{
	try
	{
		fn();
	}
	catch (const std::exception&)
	{
		return;
	}
	throw std::runtime_error("Invalid dispatch accepted");
}
class event_loop
{
	std::mutex mutex;
	std::condition_variable changed;
	std::deque<std::function<void()>> queue;
	bool stopping = false;
	std::thread worker;
	std::thread::id id;

  public:
	event_loop()
	{
		auto ready = std::make_shared<std::promise<void>>();
		worker = std::thread([this, ready]
							 {
			id = std::this_thread::get_id(); ready->set_value();
			for (;;)
			{
				std::function<void()> work;
				{
					std::unique_lock lock(mutex); changed.wait(lock, [&] { return stopping || !queue.empty(); });
					if (stopping && queue.empty()) return;
					work = std::move(queue.front()); queue.pop_front();
				}
				work();
			} });
		ready->get_future().wait();
	}
	~event_loop()
	{
		{
			std::lock_guard lock(mutex);
			stopping = true;
		}
		changed.notify_all();
		worker.join();
	}
	void post(std::function<void()> work)
	{
		{
			std::lock_guard lock(mutex);
			queue.push_back(std::move(work));
		}
		changed.notify_one();
	}
	bool is_main() const { return id == std::this_thread::get_id(); }
};
struct owner
{
	unsigned& destroyed;
	std::thread::id& thread;
	owner(unsigned& count, std::thread::id& id) : destroyed(count), thread(id) {}
	~owner()
	{
		++destroyed;
		thread = std::this_thread::get_id();
	}
};
} // namespace
int main()
{
	try
	{
		using namespace metal4;
		using namespace std::chrono_literals;
		event_loop loop;
		unsigned posts = 0, value = 0;
		const auto post = [&](std::function<void()> work)
		{ ++posts; loop.post(std::move(work)); };
		main_thread_dispatch dispatcher(post, [&]
										{ return loop.is_main(); }, 2s);
		dispatcher.invoke([&]
						  { ++value; dispatcher.invoke([&] { ++value; }); });
		require(value == 2 && posts == 1, "Nested main-thread call reposted/deadlocked");
		bool caught = false;
		try
		{
			dispatcher.invoke([]
							  { throw std::domain_error("UI operation failed"); });
		}
		catch (const std::domain_error& error)
		{
			caught = std::string(error.what()) == "UI operation failed";
		}
		require(caught, "UI exception type/message lost");
		unsigned destroyed = 0;
		std::thread::id released_on;
		auto retained = std::make_shared<owner>(destroyed, released_on);
		const std::weak_ptr<owner> weak = retained;
		dispatcher.invoke([retained = std::move(retained)] { (void)retained; });
		require(weak.expired() && destroyed == 1 && released_on != std::this_thread::get_id(), "Closure owner released after completion/on caller thread");
		std::cout << "Main-thread nested inline calls,exception transport and closure-owner release before completion passed\n";
		std::vector<std::function<void()>> pending;
		main_thread_dispatch delayed([&](auto work)
									 { pending.push_back(std::move(work)); }, []
									 { return false; }, 5ms);
		rejects([&]
				{ delayed.invoke([&]
								 { ++value; }); });
		const auto old_value = value;
		pending.front()();
		pending.front()();
		require(value == old_value, "Late/double delivery used cancelled caller reference");
		main_thread_dispatch failed_post([&](auto work)
										 { pending.push_back(std::move(work)); throw std::domain_error("post failed"); }, []
										 { return false; });
		caught = false;
		try
		{
			failed_post.invoke([&]
							   { ++value; });
		}
		catch (const std::domain_error& error)
		{
			caught = std::string(error.what()) == "post failed";
		}
		pending.back()();
		require(caught && value == old_value, "Post-after-enqueue failure left live work");
		main_thread_dispatch wrong_thread([](auto work)
										  { work(); work(); }, []
										  { return false; });
		rejects([&]
				{ wrong_thread.invoke([&]
									  { ++value; }); });
		require(value == old_value, "Wrong-thread delivery ran target work");
		std::cout << "Pending timeout,late/double delivery,post-after-enqueue failure and wrong-thread target protection passed\n";
		std::promise<void> entered, release;
		const auto released = release.get_future().share();
		std::atomic<bool> returned = false, timed_out = false, finished = false;
		main_thread_dispatch running([&](auto work)
									 { loop.post(std::move(work)); }, [&]
									 { return loop.is_main(); }, 5ms);
		std::thread caller([&]
						   {
			try { running.invoke([&] { entered.set_value(); released.wait(); finished = true; }); }
			catch (const std::runtime_error&) { timed_out = true; }
			returned = true; });
		const bool began = entered.get_future().wait_for(2s) == std::future_status::ready;
		std::this_thread::sleep_for(25ms);
		const bool premature = returned;
		release.set_value();
		caller.join();
		require(began && !premature && returned && finished && timed_out, "Running timeout returned before target closure finished");
		std::cout << "Running timeout joins operation before releasing caller-owned lifetime passed\n";
		std::atomic<unsigned> count = 0;
		main_thread_dispatch concurrent([&](auto work)
										{ loop.post(std::move(work)); }, [&]
										{ return loop.is_main(); }, 2s);
		std::array<std::thread, 4> callers;
		for (auto& thread : callers)
			thread = std::thread([&]
								 { for (unsigned i = 0; i < 100; ++i) concurrent.invoke([&] { ++count; }); });
		for (auto& thread : callers)
			thread.join();
		require(count == 400, "Concurrent main-thread dispatch lost/duplicated work");
		rejects([]
				{ main_thread_dispatch invalid({}, []
											   { return true; }); });
		rejects([]
				{ main_thread_dispatch invalid([](auto) {}, {}, 1ms); });
		rejects([]
				{ main_thread_dispatch invalid([](auto) {}, []
											   { return true; }, 0ms); });
		rejects([]
				{ main_thread_dispatch invalid([](auto) {}, []
											   { return true; }, 2min); });
		rejects([&]
				{ dispatcher.invoke({}); });
		std::cout << "400 concurrent caller jobs and invalid dispatcher preflight passed;CPU fixture only,no app/UI started\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
