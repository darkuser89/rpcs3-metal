#include "M4HostWriterGate.h"

#include <atomic>
#include <iostream>
#include <stdexcept>

namespace
{
using namespace metal4;
using namespace std::chrono_literals;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
bool until(const std::atomic<bool>& flag)
{
	const auto deadline = std::chrono::steady_clock::now() + 5s;
	while (!flag.load())
	{
		if (std::chrono::steady_clock::now() >= deadline) return false;
		std::this_thread::yield();
	}
	return true;
}
void cancelled_and_nested()
{
	host_writer_gate gate;
	require(!gate.pause_current_fault(), "Unrelated thread acquired a fault pause");
	auto ticket = gate.admit();
	require(bool(ticket) && !gate.acquire([] {}, 0ms), "Queued admission did not prevent capture");
	auto moved = std::move(ticket);
	require(!ticket && bool(moved), "Moving ticket changed queue ownership");
	moved = {};
	auto capture = gate.acquire([] {}, 0ms);
	require(bool(capture), "Cancelled ticket leaked capture exclusion");
	require(!gate.admit(), "Capture owner admitted a writer inside exclusion");
	{
		auto nested = gate.acquire([] {}, 0ms);
		require(bool(nested) && !gate.admit(), "Nested capture released outer exclusion");
	}
	require(!gate.admit(), "Nested capture retirement reopened admission");
	capture.reset();
	auto writer_ticket = gate.admit();
	{
		auto writer = writer_ticket.activate();
		require(bool(writer), "Admitted writer failed activation");
		require(!gate.acquire([] {}, 1s), "Active writer captured its own incomplete instruction");
		require(!writer_ticket.activate(), "Same ticket activated twice concurrently");
	}
	require(!writer_ticket.activate(), "Completed ticket was reactivated");
	require(bool(gate.acquire([] {}, 0ms)), "Completed writer remained pending");
}
void continuous_admission()
{
	host_writer_gate gate;
	std::atomic<bool> draining=false, captured=false, release=false, late_started=false, late_done=false;
	std::atomic<bool> failed=false;
	std::atomic<unsigned> bytes=0;
	std::jthread capture_thread, late_thread;
	auto ticket = gate.admit();
	{
		auto writer = ticket.activate();
		capture_thread = std::jthread([&]
		{
			auto capture = gate.acquire([&] { draining = true; }, 5s);
			if (!capture) { failed = true; return; }
			captured = true;
			if (!until(release)) failed = true;
		});
		require(until(draining), "Capture never began draining active work");
		require(!gate.acquire([] {}, 0ms), "Writer waited behind a capture draining its own instruction");
		// Existing callbacks may enqueue nested work while another thread drains.
		auto child = gate.admit();
		{ auto active = child.activate(); require(bool(active), "Draining blocked existing writer's child"); bytes = 1; }
		late_thread = std::jthread([&]
		{
			late_started = true;
			auto late = gate.admit();
			auto active = late.activate();
			if (!active) { failed = true; return; }
			bytes = 2; late_done = true;
		});
		require(until(late_started), "Late producer did not start");
		require(!captured && !late_done && bytes == 1, "Capture/late producer bypassed active writer");
	}
	require(until(captured), "Drained work never yielded capture");
	const auto deadline = std::chrono::steady_clock::now() + 30ms;
	while (std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
	require(!late_done && bytes == 1, "Late producer entered a live capture");
	release = true;
	capture_thread.join(); late_thread.join();
	require(!failed && late_done && bytes == 2, "Late producer did not resume after capture retirement");
	require(bool(gate.acquire([] {}, 0ms)), "Admission test leaked a writer");
}
void fault_instruction_retry()
{
	host_writer_gate gate;
	auto first = gate.admit();
	auto later = gate.admit(); // Queued after the faulting instruction.
	std::atomic<bool> ready=false, service_returned=false, resumed=false, finish=false, failed=false;
	std::atomic<unsigned> bytes=0;
	host_writer_gate::fault_pause* paused = nullptr;
	std::jthread producer([&]
	{
		auto writer = first.activate();
		auto fault = gate.pause_current_fault();
		if (!writer || !fault) { failed = true; ready = true; return; }
		paused = &*fault; ready = true;
		if (!until(service_returned)) { failed = true; return; }
		fault.reset(); // Must reacquire BEFORE the original memory instruction.
		bytes = 0x5a; resumed = true;
		if (!until(finish)) failed = true;
	});
	require(until(ready) && !failed && paused, "Faulting writer did not pause");
	require(!gate.acquire([] {}, 0ms), "Normal capture ignored a fault-blocked instruction");
	{
		auto context = paused->capture_context();
		require(bool(context), "Owner lost paused fault authority");
		auto capture = gate.acquire([] {}, 0ms);
		require(bool(capture), "Fault capture waited for its blocked job/later queue");
		service_returned = true;
		// Deliberately wake the producer before releasing capture. The original
		// instruction must still be unable to touch its destination.
		const auto deadline = std::chrono::steady_clock::now() + 30ms;
		while (std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
		require(!resumed && bytes == 0, "Fault continuation wrote while capture was exclusive");
		capture.reset();
		require(until(resumed) && bytes == 0x5a, "Original instruction did not resume after capture");
		later = {}; // Isolate the still-active original instruction's ticket.
		require(!gate.acquire([] {}, 1ms), "Capture slipped between fault recovery and writer completion");
	}
	finish = true;
	producer.join();
	require(!failed && bool(gate.acquire([] {}, 0ms)), "Fault recovery leaked admission or lost completion");
}
void reentrant_fault_and_metadata()
{
	host_writer_gate gate;
	unsigned source=17, destination=0;
	auto queued = gate.admit({&source, &destination, sizeof(destination)});
	unsigned pumps=0;
	auto capture = gate.acquire([&]
	{
		++pumps;
		auto writer = queued.activate();
		require(bool(writer), "Queued writer could not run during drain");
		const auto extent = gate.current_transfer();
		require(extent && extent->source == &source && extent->destination == &destination && extent->bytes == sizeof(destination),
			"Fault extent did not follow actual active writer");
		auto callback = gate.admit();
		auto inner = callback.activate();
		const auto nested_extent = gate.current_transfer();
		require(nested_extent && nested_extent->bytes == sizeof(destination), "Nested callback hid outer transfer extent");
		auto fault = gate.pause_current_fault();
		require(bool(fault) && !gate.pause_current_fault(), "Fault pause duplicated active writer");
		{
			auto context = fault->capture_context();
			auto owner = gate.acquire([] {}, 0ms);
			require(bool(context) && bool(owner) && !gate.admit(), "Reentrant fault capture failed or admitted writes");
		}
		fault.reset();
		destination = source;
	}, 1s);
	require(bool(capture) && pumps == 1 && destination == source, "Outer drain did not resume after nested fault service");
	require(!gate.current_transfer(), "Completed transfer extent escaped writer scope");
}
void exception_and_lifetime()
{
	host_writer_gate gate;
	auto queued = gate.admit();
	bool rejected=false;
	try { (void)gate.acquire([] { throw std::runtime_error("pump failed"); }); }
	catch (const std::runtime_error&) { rejected=true; }
	queued = {};
	std::atomic<bool> done=false;
	std::jthread producer([&] { auto ticket=gate.admit(); auto writer=ticket.activate(); done=bool(writer); });
	producer.join();
	require(rejected && done && bool(gate.acquire([] {}, 0ms)), "Throwing pump failed to reopen admission");
	host_writer_gate::ticket retained;
	{
		auto temporary = std::make_unique<host_writer_gate>();
		retained = temporary->admit();
	}
	{ auto writer=retained.activate(); require(bool(writer), "Ticket lost storage after manager/gate destruction"); }
}
} // namespace
int main()
{
	try
	{
		cancelled_and_nested();
		continuous_admission();
		fault_instruction_retry();
		reentrant_fault_and_metadata();
		exception_and_lifetime();
		std::cout << "Host writer gate:queued/immediate completion,continuous admission,nested callbacks/captures,fault-owner handoff,reacquisition before original store,normal-fault refusal,exact active transfer metadata,cancellation,pump exception rollback and retained-ticket lifetime passed\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n'; return 1;
	}
}
