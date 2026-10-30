#include "M4HostAccessFixture.h"
#include "M4RenderTasks.h"
#include <atomic>
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
using metal4::test::host_access_fixture;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refused(F&& operation)
{
	bool caught = false;
	try { operation(); } catch (const std::exception&) { caught = true; }
	require(caught, "Unsafe physical host access preparation was accepted");
}
void slices_and_actual_store()
{
	auto access = std::make_shared<host_access_fixture>();
	guarded_guest_memory memory(access);
	const guest_surface_range destination{65528, 16}, source{128, 16};
	const auto watch = memory.watch_surface(guest_byte_coverage(destination));
	const auto alias = memory.watch_surface(guest_byte_coverage({destination.address + 0x40000, destination.size}));
	const auto old = memory.surface_revision(watch), old_alias = memory.surface_revision(alias);
	std::fill_n(access->bytes + source.address, source.size, std::byte{0x5a});
	std::optional<host_write_journal::preparation> target, input, mirror;
	const auto captures = access->captures;
	access->execute([&](auto&)
	{
		target.emplace(memory.prepare_host_access(destination, true));
		input.emplace(memory.prepare_host_access(source, false));
		mirror.emplace(memory.prepare_host_access({destination.address + 0x40000, destination.size}, false));
	});
	require(access->captures == captures + 1 && !memory.pending_host_writes(), "Preparation armed an active slot or recaptured the provider");
	const auto parts = target->backing->segments();
	require(parts.size() == 2 && parts[0].guest.size == 8 && parts[1].guest.size == 8 &&
		parts[0].offset == 65528 && parts[1].offset == 0 && !physical_backing::same_allocation(parts[0], parts[1]) &&
		physical_backing::same_allocation(parts[0], mirror->backing->segments()[0]), "Physical slices lost offsets,allocation split or alias ownership");
	std::weak_ptr<void> first_owner = parts[0].owner;
	mirror.reset();
	{
		auto completed = memory.begin_host_write(std::move(*target), std::move(input));
		target.reset();
		std::memcpy(access->bytes + destination.address, access->bytes + source.address, destination.size);
		completed.commit();
	}
	require(!first_owner.expired(), "Completed intent retired backing before reconciliation");
	require(memory.surface_revision(watch) > old && memory.surface_revision(alias) > old_alias &&
		std::all_of(access->bytes + destination.address, access->bytes + destination.address + destination.size,
			[](auto value) { return value == std::byte{0x5a}; }) && !memory.pending_host_writes(), "Prepared actual store lost bytes or physical alias revisions");
	require(first_owner.expired(), "Successful reconciliation retained dead physical leases");
	access->segments_supported = false;
	refused([&] { (void)memory.prepare_host_access(source, false); });
	require(!memory.pending_host_writes(), "Unsupported provider consumed a journal slot");
}
void remap_retention_and_source_lifetime()
{
	auto access = std::make_shared<host_access_fixture>();
	auto memory = std::make_unique<guarded_guest_memory>(access);
	std::fill_n(access->bytes + 32, 16, std::byte{0x43});
	auto before = memory->prepare_host_access({32, 16}, true);
	std::weak_ptr<void> old = before.backing->segments()[0].owner;
	access->execute([&](auto&)
	{
		memory->begin_mapping_change({0, 65536});
		access->replace_first();
		memory->end_mapping_change({0, 65536});
	});
	std::fill_n(access->bytes + 32, 16, std::byte{0x77});
	require(before.backing->segments()[0].data()[0] == std::byte{0x43}, "Old physical bytes followed a reused guest VA");
	refused([&] { (void)memory->begin_host_write(before); });
	require(!memory->pending_host_writes(), "Stale pre-store mapping consumed an active slot");
	before.backing.reset();
	require(old.expired(), "Refused preparation leaked an independent mapping");
	auto prepared = memory->prepare_host_access({32, 16}, true);
	old = prepared.backing->segments()[0].owner;
	auto* physical = prepared.backing->segments()[0].data();
	{
		auto partial = memory->begin_host_write(std::move(prepared));
		access->bytes[32] = std::byte{0x19}; // A partial store,then VA replacement.
		access->execute([&](auto&)
		{
			memory->begin_mapping_change({0, 65536});
			access->replace_first();
			memory->end_mapping_change({0, 65536});
		});
	}
	refused([&] { (void)memory->read({32, 16}); });
	require(memory->pending_host_writes() == 1 && !old.expired() && physical[0] == std::byte{0x19} && physical[1] == std::byte{0x77} &&
		access->bytes[32] == std::byte{0}, "Ambiguous journal lost old physical dirty bytes or wrote the replacement allocation");
	memory.reset();
	require(old.expired(), "Destroyed ambiguous journal leaked physical owners");
	memory = std::make_unique<guarded_guest_memory>(access);
	access->bytes[32] = std::byte{0x61};
	auto read = memory->prepare_host_access({32, 16}, false);
	old = read.backing->segments()[0].owner;
	physical = read.backing->segments()[0].data();
	auto lifetime = memory->begin_host_read(std::move(read));
	memory.reset(); access.reset();
	require(!old.expired() && physical[0] == std::byte{0x61}, "Read-only source lease depended on provider or VM handle lifetime");
	lifetime.commit();
	require(old.expired(), "Source-only completion retained physical bytes");
}
void paused_preparations()
{
	auto access = std::make_shared<host_access_fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	guarded_guest_memory memory(access);
	const auto watched = memory.watch_surface(guest_byte_coverage({65536, 128}));
	const auto revision = memory.surface_revision(watched);
	render_tasks tasks([] {});
	tasks.bind_owner();
	std::atomic<unsigned> prepared{0}, finished{0};
	std::atomic<bool> resume{false};
	std::array<std::exception_ptr, 2> failures{};
	std::array<std::thread, 2> producers;
	for (unsigned i = 0; i < 2; ++i)
		producers[i] = std::thread([&, i]
		{
			try
			{
				auto ticket = access->gate->admit();
				auto writer = ticket.activate();
				require(bool(writer), "Prepared producer lost writer activation");
				std::optional<host_write_journal::preparation> result;
				{
					auto paused = access->gate->pause_current_fault();
					require(bool(paused), "Prepared producer failed to yield");
					tasks.invoke([&]
					{
						auto context = paused->capture_context();
						require(bool(context), "Preparation lost its paused capture context");
						result.emplace(memory.prepare_host_access({65536 + 64 * i, 64}, true));
						require(!memory.pending_host_writes(), "Paused preparation armed a blocking active journal intent");
					});
					++prepared;
					while (!resume.load()) std::this_thread::yield();
				}
				auto completion = memory.begin_host_write(std::move(*result));
				std::fill_n(access->bytes + 65536 + 64 * i, 64, std::byte(i + 1));
				completion.commit();
			}
			catch (...) { failures[i] = std::current_exception(); }
			++finished;
		});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (prepared.load() < 2 && !finished.load() && std::chrono::steady_clock::now() < deadline)
	{
		tasks.drain();
		std::this_thread::yield();
	}
	resume = true;
	while (finished.load() < 2 && std::chrono::steady_clock::now() < deadline)
	{
		tasks.drain();
		std::this_thread::yield();
	}
	if (finished.load() < 2) tasks.close();
	for (auto& thread : producers) thread.join();
	for (auto failure : failures) if (failure) std::rethrow_exception(failure);
	require(prepared == 2 && memory.surface_revision(watched) > revision && !memory.pending_host_writes() &&
		access->bytes[65536] == std::byte{1} && access->bytes[65600] == std::byte{2}, "Concurrent paused preparations failed to resume/store/reconcile");
}
void invalid_slices()
{
	auto owner = std::make_shared<std::array<std::byte, 64>>();
	refused([&] { (void)physical_backing({0, 8}, {{{1, 8}, owner, 0, 64}}); });
	refused([&] { (void)physical_backing({0, 8}, {{{0, 4}, owner, 0, 64}}); });
	refused([&] { (void)physical_backing({0, 8}, {{{0, 8}, owner, 60, 64}}); });
	refused([&] { (void)physical_backing({0, 8}, {{{0, 8}, {}, 0, 64}}); });
}
} // namespace
int main()
{
	try
	{
		invalid_slices(); slices_and_actual_store(); remap_retention_and_source_lifetime(); paused_preparations();
		std::cout << "Physical host preparation:exact independent allocation slices/offsets/aliases;actual store revisions;two paused owner handoffs before journal activation;stale pre-store refusal;partial/remap dirty physical bytes retained;source close/VA replacement/provider destruction/completion release;no App or guest VM started\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
