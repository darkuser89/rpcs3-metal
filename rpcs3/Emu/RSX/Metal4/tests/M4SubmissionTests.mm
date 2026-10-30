#include "M4Resources.h"
#include "M4HeapBudget.h"

#include <array>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>

namespace
{
using namespace metal4;
using namespace std::chrono_literals;
void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}
void cancelled(const submission& work, const char* diagnostic)
{
	bool rejected = false;
	try { work.wait(5s); }
	catch (const std::runtime_error& error)
	{
		rejected = std::string(error.what()).find(diagnostic) != std::string::npos;
	}
	require(rejected && work.ready() && work.cancelled(), "Uncommitted cancellation did not finish with its expected diagnostic");
	require(!work.cancel_pending(), "Finished token accepted cancellation");
}
void deadline_and_queue()
{
	context ctx(2, 2, 100ms);
	const auto output = make_buffer(ctx, 8, buffer_storage::shared);
	std::memset(output.contents, 0, 8);
	const auto gate = [ctx.device() newSharedEvent];
	const auto native_gate = [ctx.device() newEvent];
	ctx.begin();
	// If cancellation enqueues this native wait prematurely, the later good
	// command never executes. Neither of these events is ever signalled.
	ctx.wait_event(native_gate, 1);
	ctx.wait_event(gate, 1);
	fill_buffer(ctx, output, 0, 4, 0x11);
	auto owner = std::make_shared<int>(1);
	std::weak_ptr<int> weak = owner;
	ctx.keep_alive(owner);
	owner.reset();
	auto bad = ctx.submit();
	ctx.begin();
	fill_buffer(ctx, output, 4, 4, 0x22);
	auto good = ctx.submit();
	require(!weak.expired(), "Queued batch released its owner before worker cleanup");
	cancelled(bad, "deadline expired");
	good.wait(5s);
	require(weak.expired(), "Expired precommit batch retained its owner");
	const auto* words = static_cast<const std::uint32_t*>(output.contents);
	require(words[0] == 0 && words[1] == 0x22222222, "Expired batch executed or poisoned a later native queue commit");
	ctx.recover_completed_errors();
	ctx.begin();
	fill_buffer(ctx, output, 0, 4, 0x33);
	ctx.submit().wait(5s);
	require(words[0] == 0x33333333 && bad.cancelled(), "Cancelled slot reuse changed its old token or failed to execute");
	std::cout << "Never-signalled shared event expires; no native queue wait leaks, later commit and recovered slot execute correctly\n";
}
void explicit_cancel_and_shutdown()
{
	context ctx(3, 2, 5s);
	const auto output = make_buffer(ctx, 12, buffer_storage::shared);
	std::memset(output.contents, 0, 12);
	const auto gate = [ctx.device() newSharedEvent];
	ctx.begin();
	const auto observed = ctx.recording_submission();
	require(!observed.cancel_pending(), "Queued cancellation accepted an active recording");
	ctx.wait_event(gate, 1);
	fill_buffer(ctx, output, 0, 4, 0x11);
	const auto first = ctx.submit();
	bool timeout = false;
	try { first.wait(1ms); } catch (const std::runtime_error&) { timeout = true; }
	require(timeout && !first.ready() && !first.cancelled(), "CPU wait timeout cancelled submitted work");
	ctx.begin();
	fill_buffer(ctx, output, 4, 4, 0x22);
	const auto second = ctx.submit();
	require(second.cancel_pending(), "Queued batch behind a gate could not be cancelled");
	require(first.cancel_pending(), "Waiting batch could not be cancelled");
	cancelled(first, "cancelled before Metal4 commit");
	cancelled(second, "cancelled before Metal4 commit");
	require(observed.cancelled(), "Pre-submit observation lost cancellation state");
	ctx.recover_completed_errors();
	ctx.begin();
	ctx.wait_event(gate, 1);
	fill_buffer(ctx, output, 0, 4, 0x33);
	const auto queued = ctx.submit();
	ctx.begin();
	fill_buffer(ctx, output, 4, 4, 0x44);
	const auto behind = ctx.submit();
	ctx.begin();
	fill_buffer(ctx, output, 8, 4, 0x55);
	const auto recording = ctx.recording_submission();
	const auto before = std::chrono::steady_clock::now();
	ctx.shutdown();
	require(std::chrono::steady_clock::now() - before < 2s, "Shutdown remained stuck in a native shared-event CPU wait");
	ctx.shutdown();
	cancelled(queued, "cancelled before Metal4 commit");
	cancelled(behind, "cancelled before Metal4 commit");
	cancelled(recording, "Recording was cancelled");
	gate.signaledValue = 1;
	const auto* words = static_cast<const std::uint32_t*>(output.contents);
	require(words[0] == 0 && words[1] == 0 && words[2] == 0, "Cancelled work wrote GPU output");
	bool stopped = false;
	try { ctx.begin(); } catch (const std::logic_error&) { stopped = true; }
	require(stopped, "Shutdown allowed a new recording");
	std::cout << "CPU timeout preserves ownership; explicit queued cancellation and idempotent shutdown/join discard no-longer-authorized work\n";
}
void detached_deadline()
{
	std::shared_ptr<heap_budget> budget;
	std::optional<submission> work;
	std::weak_ptr<int> weak;
	@autoreleasepool
	{
		auto ctx = std::make_unique<context>(1, 2, 100ms);
		budget = std::make_shared<heap_budget>(*ctx, 8u << 20);
		ctx->set_buffer_budget(budget);
		const auto gate = [ctx->device() newSharedEvent];
		const auto scratch = make_buffer(*ctx, 4096, buffer_storage::private_gpu);
		ctx->begin();
		ctx->wait_event(gate, 1);
		fill_buffer(*ctx, scratch, 0, 4096, 0x66);
		const std::array<std::byte, 16> data{};
		(void)ctx->upload(data);
		(void)bindings{}.snapshot(*ctx);
		auto owner = std::make_shared<int>(7);
		weak = owner;
		ctx->keep_alive(owner);
		work = ctx->submit();
		ctx.reset();
	}
	cancelled(*work, "deadline expired");
	require(weak.expired() && budget->statistics().reserved_bytes == 0,
		"Destroyed context with a never-signalled event leaked native allocations or leases");
	std::cout << "Detached context reaches its finite deadline and releases temporary buffers, upload pages, argument tables and owner leases\n";
}
void commit_race()
{
	context ctx(1, 2, 1s);
	const auto output = make_buffer(ctx, 4, buffer_storage::shared);
	unsigned cancelled_count = 0;
	for (unsigned i = 0; i < 128; ++i)
	{
		*static_cast<std::uint32_t*>(output.contents) = 0;
		const auto gate = [ctx.device() newSharedEvent];
		ctx.begin();
		ctx.wait_event(gate, 1);
		fill_buffer(ctx, output, 0, 4, 0x77);
		const auto work = ctx.submit();
		gate.signaledValue = 1;
		if (work.cancel_pending())
		{
			cancelled(work, "cancelled before Metal4 commit");
			require(*static_cast<const std::uint32_t*>(output.contents) == 0, "Accepted cancellation raced a native commit");
			++cancelled_count;
			ctx.recover_completed_errors();
		}
		else
		{
			work.wait(5s);
			require(!work.cancelled() && *static_cast<const std::uint32_t*>(output.contents) == 0x77777777,
				"Committed work was partially cancelled or lost its resource lifetime");
		}
	}
	ctx.begin();
	fill_buffer(ctx, output, 0, 4, 0x88);
	const auto complete = ctx.submit();
	complete.wait(5s);
	require(!complete.cancel_pending() && !complete.cancelled(), "Completed native work accepted retroactive cancellation");
	std::cout << "128 signal/cancel races preserve the commit boundary (accepted: " << cancelled_count << ")\n";
}
}

int main()
{
	@autoreleasepool
	{
		try
		{
			deadline_and_queue();
			explicit_cancel_and_shutdown();
			detached_deadline();
			commit_race();
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
