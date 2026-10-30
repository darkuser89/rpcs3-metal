#include "M4CPUCapture.h"

#include <array>
#include <atomic>
#include <iostream>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Checked CPU capture contract differs");
}
template <typename Exception, typename F>
void refuses(F&& operation)
{
	bool caught = false;
	try
	{
		operation();
	}
	catch (const Exception&)
	{
		caught = true;
	}
	require(caught);
}
} // namespace
int main()
{
	try
	{
		using namespace metal4;
		const cpu_capture_dispatch immediate = [](const auto& operation)
		{ operation(); };
		unsigned calls = 0;
		checked_cpu_capture(immediate, [&]
							{ ++calls; });
		require(calls == 1);
		refuses<std::invalid_argument>([&]
									   { checked_cpu_capture({}, [] {}); });
		refuses<std::invalid_argument>([&]
									   { checked_cpu_capture(immediate, {}); });
		std::function<void()> retained;
		auto owner = std::make_shared<int>(1);
		std::weak_ptr<int> weak = owner;
		refuses<std::logic_error>([&]
								  { checked_cpu_capture([&](const auto& callback)
														{ retained = callback; }, [owner, &calls]
														{ ++calls; }); });
		owner.reset();
		require(weak.expired());
		retained();
		require(calls == 1);
		refuses<std::logic_error>([&]
								  { checked_cpu_capture([](const auto& callback)
														{ callback(); callback(); }, [&]
														{ ++calls; }); });
		require(calls == 2);
		refuses<std::logic_error>([&]
								  { checked_cpu_capture([](const auto& callback)
														{ std::thread foreign(callback); foreign.join(); }, [&]
														{ ++calls; }); });
		require(calls == 2);
		bool dispatched = false;
		refuses<std::range_error>([&]
								  { checked_cpu_capture([&](const auto& callback)
														{ dispatched = true; callback(); dispatched = false; }, []
														{ throw std::range_error("capture failure"); }); });
		require(!dispatched);
		refuses<std::overflow_error>([&]
									 { checked_cpu_capture([&](const auto& callback)
														   { retained = callback; throw std::overflow_error("dispatch failure"); }, [&]
														   { ++calls; }); });
		retained();
		require(calls == 2);
		checked_cpu_capture([&](const auto& callback)
							{ retained = callback; callback(); }, [&]
							{ ++calls; });
		retained();
		require(calls == 3);
		checked_cpu_capture(immediate, [&]
							{ checked_cpu_capture(immediate, [&]
												  { ++calls; }); });
		require(calls == 4);
		std::atomic<unsigned> captures = 0;
		std::array<std::thread, 8> threads;
		for (auto& thread : threads)
			thread = std::thread([&]
								 {
				for (unsigned i = 0; i < 1000; ++i)
				{
					const auto caller = std::this_thread::get_id();
					checked_cpu_capture(immediate, [&] { require(std::this_thread::get_id() == caller); ++captures; });
				} });
		for (auto& thread : threads)
			thread.join();
		require(captures == 8000);
		std::cout << "Checked CPU capture:exactly-once caller affinity,foreign/missing/duplicate/late refusal,owned-operation retirement,error transport,nesting and8000parallel captures passed\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
