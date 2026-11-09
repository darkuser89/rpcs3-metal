#include "M4CPUCapture.h"
#include "M4TransferPublication.h"

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
void completed_publication_contracts()
{
	using namespace metal4;
	struct output { std::shared_ptr<int> bytes=std::make_shared<int>(37); };
	for(unsigned phase:{1u,2u})for(unsigned mode=0;mode<3;++mode)
	{
		auto pending=std::make_unique<output>();std::weak_ptr<int> lifetime=pending->bytes;
		unsigned calls=0,stores=0;std::function<void()> retained;
		const cpu_capture_dispatch dispatch=[&](const auto& action)
		{
			if(++calls==phase)
			{
				retained=action;
				if(mode==0)throw std::range_error("CPU entry refused");
				if(mode==1)return;
				std::thread foreign(action);foreign.join();return;
			}
			action();
		};
		const auto publish=[&](auto& completed,const auto& capture)
		{
			checked_cpu_capture(capture,[&]{require(*completed.bytes==37);});
			checked_cpu_capture(capture,[&]{require(*completed.bytes==37);++stores;});
		};
		refuses<std::exception>([&]{publish_completed_transfer(pending,dispatch,publish);});
		require(pending&&!lifetime.expired()&&!stores);retained();require(!stores);
		publish_completed_transfer(pending,[](const auto& action){action();},publish);
		require(!pending&&lifetime.expired()&&stores==1);retained();require(stores==1);
	}
	for(unsigned mode=0;mode<4;++mode)
	{
		auto pending=std::make_unique<output>();std::weak_ptr<int> lifetime=pending->bytes;unsigned stores=0;
		const cpu_capture_dispatch dispatch=[&](const auto& action)
		{
			action();
			if(mode==1)throw std::range_error("Dispatcher failed after action");
			if(mode==2)action();
		};
		refuses<std::exception>([&]
		{
			publish_completed_transfer(pending,dispatch,[&](auto&,const auto& capture)
			{
				if(mode==3)throw std::range_error("Metadata rejected outside capture");
				checked_cpu_capture(capture,[&]{++stores;if(mode==0)throw std::range_error("Writer failed after prefix");});
			});
		});
		require(!pending&&lifetime.expired()&&stores==unsigned(mode!=3));
	}
	std::unique_ptr<output> absent;
	refuses<std::logic_error>([&]{publish_completed_transfer(absent,[](const auto& action){action();},[](auto&,const auto&){});});
	std::cout<<"Completed transfer publication: six first/second-entry refusal cases retain ownership,late/foreign/dropped callbacks cannot write,retry stores once;four started/duplicate/post-action/metadata failures retire without replay,empty owner refused\n";
}
} // namespace
int main()
{
	try
	{
		using namespace metal4;
		completed_publication_contracts();
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
