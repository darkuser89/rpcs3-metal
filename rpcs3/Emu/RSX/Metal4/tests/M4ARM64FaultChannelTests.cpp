#include "M4ARM64FaultChannel.h"
#include "M4DarwinInstruction.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void await_pending(arm64_fault_channel& channel)
{
	const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!channel.has_pending() && std::chrono::steady_clock::now() < end) std::this_thread::yield();
	require(channel.has_pending(), "Native ARM fault producer did not publish its pending request");
}
void test_owned_slots()
{
	arm64_fault_channel channel{0x1000, 4096, 16};
	std::array<std::atomic<unsigned>, 16> outcomes{};
	std::vector<std::thread> threads;
	for (unsigned i = 0; i < outcomes.size(); ++i) threads.emplace_back([&, i]
	{
		auto client = channel.attach();
		arm64_memory_state before; before.pc = 0x4000; before.x[0] = i; before.v[31][15] = std::byte{static_cast<unsigned char>(i)};
		if (client.exchange(before, 0xfff, false).status != arm64_fault_channel::outcome::unowned) return;
		bool duplicate = false;
		try { auto other = channel.attach(); (void)other; } catch (const std::logic_error&) { duplicate = true; }
		if (!duplicate) return;
		for (unsigned iteration = 0; iteration < 2; ++iteration)
		{
			const auto reply = client.exchange(before, 0x1000 + i, (i & 1) != 0);
			if (reply.status != arm64_fault_channel::outcome::completed || reply.reply.pc != before.pc + 4 ||
				reply.reply.x[0] != i + 1 || reply.reply.v[31][15] != before.v[31][15]) return;
			before.pc = reply.reply.pc;
		}
		outcomes[i].store(1, std::memory_order_release);
	});
	std::size_t completed = 0;
	const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (completed < 32 && std::chrono::steady_clock::now() < end)
	{
		completed += channel.drain([](const auto& request) -> std::optional<arm64_memory_state>
		{
			require(request.fault == 0x1000 + request.before.x[0] && request.writing == ((request.before.x[0] & 1) != 0), "Native ARM slot mixed owned input or direction");
			auto reply = request.before; reply.pc += 4; ++reply.x[0]; return reply;
		});
		std::this_thread::yield();
	}
	channel.close(); for (auto& thread : threads) thread.join();
	require(completed == 32 && std::all_of(outcomes.begin(), outcomes.end(), [](auto& x) { return x.load(std::memory_order_acquire) == 1; }),
		"Native ARM channel lost a concurrent producer,owned reply or reusable slot");
}
void test_failure_and_close()
{
	using outcome = arm64_fault_channel::outcome;
	for (unsigned scenario = 0; scenario < 3; ++scenario)
	{
		arm64_fault_channel channel{0x1000, 4096, 1};
		outcome result = outcome::completed;
		std::thread producer([&] { auto client = channel.attach(); result = client.exchange({}, 0x1000, true).status; });
		await_pending(channel);
		if (scenario == 0) channel.close();
		else channel.drain([&](const auto&) -> std::optional<arm64_memory_state>
		{
			if (scenario == 1) return {};
			throw std::runtime_error("Controlled native fault owner failure");
		});
		producer.join();
		require(result == (scenario == 0 ? outcome::cancelled : scenario == 1 ? outcome::unsupported : outcome::failed),
			"Native ARM channel authorized retry after cancellation,unsupported instruction or owner failure");
	}
	arm64_fault_channel channel{0x1000, 4096, 1};
	std::atomic<bool> closing{false}, closed{false};
	outcome result = outcome::failed;
	std::thread producer([&] { auto client = channel.attach(); result = client.exchange({}, 0x1000, false).status; });
	await_pending(channel);
	std::thread closer([&]
	{
		while (!closing.load(std::memory_order_acquire)) std::this_thread::yield();
		channel.close(); closed.store(true, std::memory_order_release);
	});
	bool joined = false;
	channel.drain([&](const auto& request) -> std::optional<arm64_memory_state>
	{
		closing.store(true, std::memory_order_release);
		while (channel.open()) std::this_thread::yield();
		for (unsigned i = 0; i < 1000; ++i) std::this_thread::yield();
		joined = !closed.load(std::memory_order_acquire);
		bool reentry = false, self_close = false;
		try { channel.drain([](const auto&) -> std::optional<arm64_memory_state> { return {}; }); } catch (const std::logic_error&) { reentry = true; }
		try { channel.close(); } catch (const std::logic_error&) { self_close = true; }
		require(reentry && self_close, "Native ARM channel reentered or closed its own running callback");
		return request.before;
	});
	producer.join(); closer.join();
	require(joined && closed && result == outcome::completed, "Native ARM close failed to join running work before owner retirement");
}
void test_admission_and_lifetime()
{
	for (unsigned i = 0; i < 128; ++i)
	{
		arm64_fault_channel channel{0x1000, 4096, 1};
		std::atomic<bool> ready{false}, begin{false};
		arm64_fault_channel::outcome result = arm64_fault_channel::outcome::completed;
		std::thread producer([&]
		{
			auto client = channel.attach(); ready.store(true, std::memory_order_release);
			while (!begin.load(std::memory_order_acquire)) std::this_thread::yield();
			result = client.exchange({}, 0x1000, false).status;
		});
		while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
		begin.store(true, std::memory_order_release); channel.request_close(); producer.join();
		require(result == arm64_fault_channel::outcome::cancelled, "Native ARM admission raced close and escaped cancellation");
	}
	auto channel = std::make_unique<arm64_fault_channel>(0x1000, 4096, 1);
	std::atomic<bool> ready{false}, begin{false};
	arm64_fault_channel::outcome result = arm64_fault_channel::outcome::completed;
	std::thread producer([&]
	{
		auto client = channel->attach(); ready.store(true, std::memory_order_release);
		while (!begin.load(std::memory_order_acquire)) std::this_thread::yield();
		result = client.exchange({}, 0x1000, false).status;
	});
	while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
	channel.reset(); begin.store(true, std::memory_order_release); producer.join();
	require(result == arm64_fault_channel::outcome::cancelled, "Native ARM registered slot did not outlive closed channel ownership");
}

void test_close_from_running_callback()
{
	using outcome = arm64_fault_channel::outcome;
	arm64_fault_channel channel{0x1000, 4096, 2};
	outcome running = outcome::failed;
	outcome cancelled = outcome::failed;
	outcome later = outcome::failed;
	std::thread first([&]
	{
		auto client = channel.attach();
		running = client.exchange({}, 0x1000, false).status;
	});
	await_pending(channel);
	std::size_t drained = channel.drain([&](const auto& request) -> std::optional<arm64_memory_state>
	{
		std::thread second([&]
		{
			auto client = channel.attach();
			cancelled = client.exchange({}, 0x1001, false).status;
			later = client.exchange({}, 0x1002, false).status;
		});
		await_pending(channel);
		channel.request_close();
		second.join();
		return request.before;
	});
	first.join();
	channel.close();
	require(drained == 1 && running == outcome::completed && cancelled == outcome::cancelled &&
		later == outcome::cancelled && !channel.has_pending(),
		"Native ARM nonblocking close left a parked request or cancelled running work");
	std::cout << "ARM fault failure cancellation: running callback closed admission without waiting; queued and later faults cancelled; 128 admission races passed\n";
}

void test_callback_retirement()
{
	arm64_fault_channel channel{0x1000, 4096, 1};
	std::atomic<bool> callback{false}, retiring{false}, release{false}, closed{false};
	bool joined = false;
	struct lifetime
	{
		std::atomic<bool>& retiring;
		std::atomic<bool>& release;
		~lifetime()
		{
			retiring.store(true, std::memory_order_release);
			while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
		}
	};
	arm64_fault_channel::outcome result = arm64_fault_channel::outcome::failed;
	std::thread producer([&] { auto client = channel.attach(); result = client.exchange({}, 0x1000, false).status; });
	await_pending(channel);
	std::thread closer([&]
	{
		while (!callback.load(std::memory_order_acquire)) std::this_thread::yield();
		channel.close(); closed.store(true, std::memory_order_release);
	});
	std::thread releaser([&]
	{
		while (!retiring.load(std::memory_order_acquire)) std::this_thread::yield();
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
		while (!closed.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
		joined = !closed.load(std::memory_order_acquire);
		release.store(true, std::memory_order_release);
	});
	channel.drain([&, owner = std::shared_ptr<lifetime>(new lifetime{retiring, release})](const auto& request) -> std::optional<arm64_memory_state>
	{
		callback.store(true, std::memory_order_release);
		while (channel.open()) std::this_thread::yield();
		return request.before;
	});
	producer.join(); closer.join(); releaser.join();
	require(joined && result == arm64_fault_channel::outcome::completed, "Native ARM close returned before owned callback destruction");
}

void test_registration_and_fetch()
{
	using outcome = arm64_fault_channel::outcome;
	arm64_fault_channel channel{0x1000, 4096, 1};
	{
		auto client = channel.attach();
		ucontext_t unrelated{};
		require(client.handle(unrelated, 0xfff) == outcome::unowned && client.handle(unrelated, 0x2000) == outcome::unowned &&
			client.handle(unrelated, 0x1000) == outcome::failed, "Native ARM channel claimed an unrelated fault or accepted a malformed owned context");
		require(client.exchange({}, 0x1000, false).status == outcome::failed, "Native ARM owner blocked waiting for its own fault queue");
		bool budget = false;
		std::thread foreign([&] { try { auto other = channel.attach(); (void)other; } catch (const std::length_error&) { budget = true; } }); foreign.join();
		require(budget, "Native ARM fault channel exceeded its fixed slot budget");
		outcome foreign_result = outcome::completed;
		std::thread wrong([&] { foreign_result = client.exchange({}, 0x1000, true).status; }); wrong.join();
		require(foreign_result == outcome::failed, "Native ARM client accepted a foreign original thread");
	}
	std::atomic<arm64_fault_channel::client*> original{nullptr};
	std::atomic<bool> release{false}, foreign_done{false};
	std::thread registered([&]
	{
		auto client = channel.attach(); original.store(&client, std::memory_order_release);
		while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
	});
	while (!original.load(std::memory_order_acquire)) std::this_thread::yield();
	outcome misplaced = outcome::completed;
	std::thread foreign([&]
	{
		misplaced = original.load(std::memory_order_acquire)->exchange({}, 0x1000, false).status;
		foreign_done.store(true, std::memory_order_release);
	});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!foreign_done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
	{
		channel.drain([](const auto& request) -> std::optional<arm64_memory_state> { return request.before; });
		std::this_thread::yield();
	}
	channel.close(); foreign.join(); release.store(true, std::memory_order_release); registered.join();
	require(misplaced == outcome::failed, "Native ARM client accepted another producer's TLS identity");
	bool refused = false; try { auto client = channel.attach(); (void)client; } catch (const std::logic_error&) { refused = true; }
	require(refused, "Native ARM channel registered after shutdown");
	const auto pc = reinterpret_cast<std::uint64_t>(&test_registration_and_fetch);
	const auto instruction = capture_darwin_instruction(pc);
	std::uint32_t known; std::memcpy(&known, reinterpret_cast<const void*>(pc), sizeof(known));
	require(instruction && *instruction == known && !capture_darwin_instruction(0) && !capture_darwin_instruction(pc + 1) &&
		!capture_darwin_instruction(~std::uint64_t{0} - 3), "Darwin instruction capture lost exact executable bytes or accepted an invalid PC");
	const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
	auto* bytes = ::mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	require(bytes != MAP_FAILED, "Cannot create Darwin instruction capture fixture");
	const auto address = reinterpret_cast<std::uint64_t>(bytes);
	const bool data_refused = !capture_darwin_instruction(address);
	const bool protected_ok = ::mprotect(bytes, page, PROT_NONE) == 0;
	const bool protected_refused = !capture_darwin_instruction(address);
	const bool unmapped = ::munmap(bytes, page) == 0;
	require(data_refused && protected_ok && protected_refused && unmapped && !capture_darwin_instruction(address),
		"Darwin instruction capture dereferenced or accepted data,protected or absent memory");
}
} // namespace
void test_arm64_fault_channel()
{
	test_owned_slots(); test_failure_and_close(); test_admission_and_lifetime(); test_close_from_running_callback();
	test_callback_retirement(); test_registration_and_fetch();
	std::cout << "ARM fault channel:32 owned concurrent exchanges,16 registered host producers;fixed slots/TLS identity;owner/foreign/budget refusal;unsupported/exception/pending-cancel/running-close join/reentry;128 admission-close races,registered slot and owned callback-destructor lifetime;owner kernel instruction capture/executable-only/invalid PCs;no App/guest CPU/signal installation\n";
}
#ifdef METAL4_ARM64_FAULT_CHANNEL_STANDALONE
int main()
{
	try { test_arm64_fault_channel(); return 0; }
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
#endif
