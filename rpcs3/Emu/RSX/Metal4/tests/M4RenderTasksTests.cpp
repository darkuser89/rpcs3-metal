#include "M4RenderTasks.h"

#include <array>
#include <atomic>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native render-task scheduling contract differs");
}
template <typename E, typename F>
void refuses(F&& operation)
{
	bool caught = false;
	try
	{
		operation();
	}
	catch (const E&)
	{
		caught = true;
	}
	require(caught);
}
template <typename F>
void until(F&& predicate)
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!predicate())
	{
		if (std::chrono::steady_clock::now() > deadline)
			throw std::runtime_error("Render-task fixture did not reach barrier");
		std::this_thread::yield();
	}
}
} // namespace
int main()
{
	try
	{
		refuses<std::invalid_argument>([]
									   { render_tasks tasks({}); });
		refuses<std::invalid_argument>([]
									   { render_tasks tasks([] {}, 0); });
		refuses<std::invalid_argument>([]
									   { render_tasks tasks([] {}, 4097); });
		refuses<std::invalid_argument>([]
									   { render_tasks tasks([] {}, 1, std::chrono::milliseconds(0)); });
		refuses<std::invalid_argument>([]
									   { render_tasks tasks([] {}, 1, std::chrono::minutes(2)); });
		const auto owner = std::this_thread::get_id();
		std::atomic<unsigned> wakes = 0;
		render_tasks tasks([&]
						   { ++wakes; });
		require(!tasks.available());
		refuses<std::logic_error>([&]
								  { tasks.invoke([] {}); });
		refuses<std::logic_error>([&]
								  { (void)tasks.drain(); });
		tasks.bind_owner();
		refuses<std::logic_error>([&]
								  { tasks.bind_owner(); });
		refuses<std::invalid_argument>([&]
									   { tasks.invoke({}); });
		unsigned calls = 0;
		tasks.invoke([&]
					 {
			++calls;
			tasks.invoke([&] { ++calls; });
			require(tasks.drain() == 0);
			refuses<std::logic_error>([&] { tasks.close(); }); });
		require(calls == 2 && wakes == 0 && tasks.available());
		refuses<std::range_error>([&]
								  { tasks.invoke([]
												 { throw std::range_error("owner error"); }); });
		std::atomic<bool> wrong_refused = false;
		std::thread wrong([&]
						  { refuses<std::logic_error>([&] { (void)tasks.drain(); }); wrong_refused = true; });
		wrong.join();
		require(wrong_refused);
		{
			auto token = std::make_shared<int>(7);
			std::weak_ptr<int> weak = token;
			std::atomic<bool> retired = false;
			std::thread producer([&, token = std::move(token)]() mutable
								 {
				tasks.invoke([&, token = std::move(token)] { require(std::this_thread::get_id() == owner && *token == 7); ++calls; });
				retired = weak.expired(); });
			until([&]
				  { return tasks.pending(); });
			require(tasks.drain(1) == 1);
			producer.join();
			require(retired && calls == 3 && !tasks.pending());
		}
		{
			std::atomic<bool> failed = false;
			std::thread producer([&]
								 { refuses<std::overflow_error>([&] { tasks.invoke([] { throw std::overflow_error("producer error"); }); }); failed = true; });
			until([&]
				  { return tasks.pending(); });
			(void)tasks.drain();
			producer.join();
			require(failed && tasks.available());
		}
		{
			std::atomic<unsigned> signals = 0;
			render_tasks bounded([&]
								 { ++signals; }, 2);
			bounded.bind_owner();
			std::array<unsigned, 2> order{};
			unsigned next = 0;
			std::thread first([&]
							  { bounded.invoke([&]
											   { order[next++] = 1; }); });
			until([&]
				  { return signals == 1; });
			std::thread second([&]
							   { bounded.invoke([&]
												{ order[next++] = 2; }); });
			until([&]
				  { return signals == 2; });
			std::thread excess([&]
							   { refuses<std::length_error>([&]
															{ bounded.invoke([] {}); }); });
			excess.join();
			require(bounded.drain(0) == 0 && bounded.drain(1) == 1 && bounded.pending());
			require(bounded.drain(1) == 1);
			first.join();
			second.join();
			require(order == std::array<unsigned, 2>{1, 2} && signals == 2);
		}
		{
			render_tasks late([] {}, 1, std::chrono::milliseconds(5));
			late.bind_owner();
			auto token = std::make_shared<int>(1);
			std::weak_ptr<int> weak = token;
			std::atomic<bool> ran = false, retired = false;
			std::thread producer([&, token = std::move(token)]() mutable
								 {
				refuses<std::runtime_error>([&] { late.invoke([&, token = std::move(token)] { ran = true; }); });
				retired = weak.expired(); });
			producer.join();
			require(retired && !ran && late.pending());
			require(late.drain() == 1 && !ran && !late.pending());
		}
		{
			render_tasks running([] {}, 1, std::chrono::milliseconds(100));
			running.bind_owner();
			std::atomic<bool> finished = false, returned = false;
			std::thread producer([&]
								 {
				refuses<std::runtime_error>([&] { running.invoke([&] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); require(!returned); finished = true; }); });
				require(finished); returned = true; });
			until([&]
				  { return running.pending(); });
			(void)running.drain();
			producer.join();
			require(returned);
		}
		{
			render_tasks stopping([] {});
			stopping.bind_owner();
			std::atomic<bool> entered = false, closed = false;
			std::thread producer([&]
								 { stopping.invoke([&]
												   { entered = true; until([&] { return !stopping.available(); }); require(!closed); }); });
			std::thread closer([&]
							   { until([&] { return entered.load(); }); stopping.close(); closed = true; });
			until([&]
				  { return stopping.pending(); });
			(void)stopping.drain();
			producer.join();
			closer.join();
			require(closed);
			refuses<std::logic_error>([&]
									  { stopping.invoke([] {}); });
			refuses<std::logic_error>([&]
									  { stopping.bind_owner(); });
		}
		{
			render_tasks stopping([] {});
			stopping.bind_owner();
			auto token = std::make_shared<int>(1);
			std::weak_ptr<int> weak = token;
			std::atomic<bool> ran = false, retired = false;
			std::thread producer([&, token = std::move(token)]() mutable
								 {
				refuses<std::runtime_error>([&] { stopping.invoke([&, token = std::move(token)] { ran = true; }); });
				retired = weak.expired(); });
			until([&]
				  { return stopping.pending(); });
			stopping.close();
			producer.join();
			require(retired && !ran && !stopping.pending());
			stopping.close();
		}
		{
			render_tasks rejected([]
								  { throw std::range_error("wake failure"); });
			rejected.bind_owner();
			auto token = std::make_shared<int>(1);
			std::weak_ptr<int> weak = token;
			std::thread producer([&, token = std::move(token)]() mutable
								 {
				refuses<std::range_error>([&] { rejected.invoke([token = std::move(token)] { throw std::runtime_error("cancelled work ran"); }); });
				require(weak.expired()); });
			producer.join();
			require(!rejected.pending());
		}
		std::atomic<unsigned> done = 0, executed = 0;
		std::mutex error_mutex;
		std::exception_ptr failure;
		std::array<std::thread, 8> producers;
		for (unsigned id = 0; id < producers.size(); ++id)
			producers[id] = std::thread([&, id]
										{
				try
				{
					for (unsigned i = 0; i < 100; ++i)
					{
						unsigned result = 0;
						tasks.invoke([&, id, i] { require(std::this_thread::get_id() == owner); result = id * 100 + i + 1; ++executed; });
						require(result == id * 100 + i + 1);
					}
				}
				catch (...) { std::lock_guard lock(error_mutex); failure = std::current_exception(); }
				++done; });
		until([&]
			  { (void)tasks.drain(3); return done == producers.size(); });
		for (auto& producer : producers)
			producer.join();
		if (failure)
			std::rethrow_exception(failure);
		require(executed == 800 && !tasks.pending());
		tasks.close();
		require(!tasks.available());
		std::cout << "Native render-task handoff:owner-only FIFO,bounded admission,errors/retirement,late cancellation,running timeout/close join,wake refusal,nesting and800jobs from8producers passed\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
