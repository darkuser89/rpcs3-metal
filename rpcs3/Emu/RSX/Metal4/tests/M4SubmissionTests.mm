#include "M4Resources.h"
#include "M4HeapBudget.h"
#include "M4Pipeline.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <optional>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace
{
using namespace metal4;
using namespace std::chrono_literals;
void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}
void buffer_blit_dependencies()
{
	context ctx{2};
	const auto a = make_buffer(ctx, 1024, buffer_storage::private_gpu);
	const auto b = make_buffer(ctx, 1024, buffer_storage::private_gpu);
	const auto out = make_buffer(ctx, 1024, buffer_storage::shared);
	ctx.begin();
	for (unsigned i = 0; i < 64; ++i)
		fill_buffer(ctx, a, i * 4, 4, static_cast<std::uint8_t>(i));
	auto stats = ctx.encoding_statistics();
	require(stats.buffer_blit_queue_barriers == 1 && stats.buffer_blit_barriers_elided == 64 &&
		stats.buffer_blit_barriers == 0, "Disjoint writes synchronized inside the checked blit encoder");
	fill_buffer(ctx, b, 0, 4, 0x71); // The bounded history must retire safely.
	copy_buffer(ctx, a, 0, b, 4, 256);
	fill_buffer(ctx, a, 0, 4, 0x72); // Write after the copy's read.
	copy_buffer(ctx, a, 0, b, 260, 4); // Read after the fill's write.
	fill_buffer(ctx, b, 260, 4, 0x73); // Write after write.
	copy_buffer(ctx, b, 0, out, 0, 264);
	ctx.submit().wait();
	const auto actual = map_buffer(out, 0, 264);
	for (unsigned i = 0; i < actual.size(); ++i)
	{
		const unsigned expected = i < 4 ? 0x71 : i >= 260 ? 0x73 : (i - 4) / 4;
		require(actual[i] == std::byte(expected), "Checked buffer blit broke a range dependency");
	}
	stats = ctx.encoding_statistics();
	require(stats.buffer_blit_barriers == 5 && stats.buffer_blit_alias_barriers == 0 &&
		stats.buffer_blit_barriers_elided == 65, "Private buffer history did not distinguish hazards or its bound");

	helper_catalog helpers(ctx);
	const auto library = helpers.library(R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void blit_dependency(device uint* values [[buffer(0)]]) {
    values[1] = values[0] ^ 0xffffffffu;
}
)metal", @"Native buffer blit stage dependency");
	const auto pipeline = helpers.compute(library, @"blit_dependency");
	ctx.begin();
	fill_buffer(ctx, out, 0, 4, 0x29);
	bindings inputs;
	inputs.set_buffer(0, out);
	const auto table = inputs.snapshot(ctx);
	const auto raw = ctx.compute();
	ctx.keep_alive(pipeline);
	[raw setComputePipelineState:pipeline];
	[raw setArgumentTable:table];
	[raw dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
	copy_buffer(ctx, out, 4, out, 8, 4);
	[raw fillBuffer:out range:NSMakeRange(16, 4) value:0x34];
	copy_buffer(ctx, out, 16, out, 20, 4);
	ctx.submit().wait();
	const auto mixed = map_buffer(out, 0, 24);
	for (unsigned i = 0; i < 4; ++i)
		require(mixed[8 + i] == std::byte{0xd6} && mixed[20 + i] == std::byte{0x34},
			"Raw encoder transition lost dispatch or late blit writes");
	const auto after = ctx.encoding_statistics();
	require(after.checked_blit_raw_transitions == stats.checked_blit_raw_transitions + 1 &&
		after.buffer_blit_alias_barriers == stats.buffer_blit_alias_barriers + 2 &&
		after.compute_encoders == stats.compute_encoders + 1,
		"Exposed encoder did not retain conservative protection for later buffer blits");
	std::cout << "Native checked buffer blits: 65 elided intra-pass barriers, five private dependency/limit barriers, "
		<< "264 exact bytes and raw dispatch/retained encoder hazards passed\n";
}

void aliased_buffer_blits()
{
	const std::size_t page = static_cast<std::size_t>(getpagesize());
	char path[] = "/private/tmp/rpcs3-m4-blit-alias-XXXXXX";
	const int fd = mkstemp(path);
	require(fd >= 0, "Could not create mapped buffer alias fixture");
	unlink(path);
	struct mapped_pages
	{
		int fd;
		std::size_t size;
		void* address = MAP_FAILED;
		~mapped_pages()
		{
			if (address != MAP_FAILED) munmap(address, size);
			close(fd);
		}
	} pages{fd, page * 2};
	require(ftruncate(fd, page) == 0, "Could not size mapped buffer alias fixture");
	pages.address = mmap(nullptr, page * 2, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
	require(pages.address != MAP_FAILED, "Could not reserve neighboring virtual pages");
	const auto a = pages.address;
	const auto b = static_cast<std::byte*>(pages.address) + page;
	require(mmap(a, page, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0) == a &&
		mmap(b, page, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0) == b,
		"Could not create distinct virtual aliases");
	@autoreleasepool
	{
		context ctx{2};
		const auto first = [ctx.device() newBufferWithBytesNoCopy:a length:page options:MTLResourceStorageModeShared deallocator:nil];
		const auto second = [ctx.device() newBufferWithBytesNoCopy:b length:page options:MTLResourceStorageModeShared deallocator:nil];
		const auto mirrored = [ctx.device() newBufferWithBytesNoCopy:a length:page * 2 options:MTLResourceStorageModeShared deallocator:nil];
		const auto out = make_buffer(ctx, page, buffer_storage::shared);
		require(first && second && mirrored && first != second && first.gpuAddress != second.gpuAddress, "Metal did not expose distinct aliases");
		ctx.begin();
		fill_buffer(ctx, first, 0, 16, 0x43);
		copy_buffer(ctx, second, 0, out, 0, 16);
		fill_buffer(ctx, second, 8, 16, 0x56);
		copy_buffer(ctx, first, 0, out, 16, 24);
		ctx.submit().wait();
		const auto actual = map_buffer(out, 0, 40);
		for (unsigned i = 0; i < actual.size(); ++i)
			require(actual[i] == (i < 24 ? std::byte{0x43} : std::byte{0x56}), "Aliased virtual buffer bytes were stale");
		const auto stats = ctx.encoding_statistics();
		require(stats.buffer_blit_queue_barriers == 1 && stats.buffer_blit_barriers_elided == 1 &&
			stats.buffer_blit_alias_barriers == 3, "Aliased shared buffers did not retain alias visibility");

		ctx.begin();
		fill_buffer(ctx, mirrored, 0, 4, 0x69);
		copy_buffer(ctx, mirrored, page, out, 40, 4);
		fill_buffer(ctx, mirrored, page, 4, 0x2a);
		copy_buffer(ctx, mirrored, 0, out, 44, 4);
		ctx.submit().wait();
		const auto same_buffer = map_buffer(out, 40, 8);
		for (unsigned i = 0; i < same_buffer.size(); ++i)
			require(same_buffer[i] == (i < 4 ? std::byte{0x69} : std::byte{0x2a}), "One mirrored buffer was treated as linear backing");
		require(ctx.encoding_statistics().buffer_blit_alias_barriers == stats.buffer_blit_alias_barriers + 3,
			"Disjoint virtual ranges inside a shared alias dropped physical visibility");
	}
	std::cout << "Native checked buffer aliases: separate buffers and mirrored ranges in one buffer, six alias barriers and 48 exact bytes passed\n";
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
			buffer_blit_dependencies();
			aliased_buffer_blits();
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
