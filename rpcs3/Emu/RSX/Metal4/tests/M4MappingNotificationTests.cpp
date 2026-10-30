#include "M4MappingNotifications.h"
#include "M4HostAccessFixture.h"
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <new>

namespace
{
thread_local bool track_allocations = false;
thread_local unsigned allocations = 0;
}
void* operator new(std::size_t bytes)
{
	if (track_allocations) ++allocations;
	if (auto* value = std::malloc(bytes ? bytes : 1)) return value;
	throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }

namespace
{
using namespace metal4;
using kind = mapping_notifications::kind;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refused(F&& operation)
{
	bool caught = false;
	try { operation(); } catch (const std::exception&) { caught = true; }
	require(caught, "Invalid mapping notification operation was accepted");
}
struct event { guest_surface_range range; kind cause; };
void boundaries_and_priority()
{
	auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
	allocations = 0; track_allocations = true;
	box->notify({0, 0}, kind::read);
	box->notify({4095, 2}, kind::unmap);
	box->notify({4096, 1}, kind::unmap); // Coalesced with the preceding pages.
	box->notify({0xffff'ffff, 1}, kind::read);
	track_allocations = false;
	require(allocations == 0 && box->pending(), "VM notification allocated or lost pending wake");
	std::vector<event> events;
	box->drain([&](auto range, auto cause) { events.push_back({range, cause}); });
	require(events.size() == 2 && events[0].cause == kind::read && events[0].range.address == 0xffff'f000 && events[0].range.size == 4096 &&
		events[1].cause == kind::unmap && events[1].range.address == 0 && events[1].range.size == 8192 && !box->pending(),
		"Mapping notification boundary,read-before-unmap or duplicate coalescing failed");
	box->notify({0, 0x1'0000'0000ull}, kind::unmap);
	std::uint64_t cursor = 0;
	box->drain([&](auto range, auto cause)
	{
		require(range.address == cursor && range.size && range.size <= 64 * 4096 && cause == kind::unmap, "Whole-VM notification has a gap,wrap or invalid chunk");
		cursor += range.size;
	}, mapping_notifications::maximum_ranges);
	require(cursor == 0x1'0000'0000ull && !box->pending(), "Whole-VM notification lost its final page");
	refused([&] { box->bind_owner(); });
	refused([&] { box->drain([](auto, auto) {}, 0); });
	refused([&] { box->drain([](auto, auto) {}, mapping_notifications::maximum_ranges + 1); });
}
void bounded_failure_and_reentry()
{
	auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
	for (unsigned page : {0, 2, 64, 128}) box->notify({page * 4096, 1}, kind::unmap);
	std::vector<std::uint32_t> serviced;
	const auto first = box->drain([&](auto range, auto) { serviced.push_back(range.address); }, 1);
	require(first == 1 && serviced == std::vector<std::uint32_t>{0} && box->pending(), "Range budget discarded unprocessed mapping pages");
	refused([&]
	{
		box->drain([&](auto range, auto)
		{
			require(range.address == 2 * 4096, "Failed range was skipped before retry");
			box->notify({3 * 4096, 1}, kind::unmap);
			throw std::runtime_error("Injected owner service failure");
		});
	});
	require(box->pending(), "Failed owner service cleared pending wake");
	bool nested = false;
	box->drain([&](auto range, auto)
	{
		if (range.address == 2 * 4096) require(range.size == 8192, "Failed mapping retry overwrote concurrently published bits");
		serviced.push_back(range.address);
		if (!nested)
		{
			nested = true;
			box->notify({0, 1}, kind::read);
			require(box->drain([](auto, auto) { throw std::runtime_error("Reentrant callback executed"); }) == 0,
				"Reentrant mapping service stole outer work");
		}
	});
	require(serviced == std::vector<std::uint32_t>{0, 2 * 4096, 64 * 4096, 128 * 4096} && box->pending(),
		"Failed mapping service lost concurrent bits or budget continuation");
	unsigned later = 0;
	box->drain([&](auto range, auto cause) { require(range.address == 0 && cause == kind::read, "Reentrant notification changed kind/range"); ++later; });
	require(later == 1 && !box->pending(), "Notification during owner service was lost");
}
void nonwaiting_vm_handoff()
{
	auto box = std::make_unique<mapping_notifications>();
	std::mutex vm_mutex;
	std::atomic<bool> notified{false}, callback_entered{false}, release_vm{false}, serviced{false};
	std::atomic<unsigned> producer_allocations{0};
	std::exception_ptr failure;
	std::thread producer([&]
	{
		std::lock_guard lock(vm_mutex);
		allocations = 0; track_allocations = true;
		box->notify({4096, 16}, kind::unmap);
		track_allocations = false; producer_allocations = allocations;
		notified = true;
		while (!release_vm.load()) std::this_thread::yield();
	});
	std::thread owner([&]
	{
		try
		{
			box->bind_owner();
			while (!notified.load()) std::this_thread::yield();
			box->drain([&](auto, auto)
			{
				callback_entered = true;
				std::lock_guard lock(vm_mutex);
				serviced = true;
			});
		}
		catch (...) { failure = std::current_exception(); callback_entered = true; }
	});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!callback_entered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
	const bool excluded = notified.load() && callback_entered.load() && !serviced.load();
	release_vm = true; producer.join(); owner.join();
	if (failure) std::rethrow_exception(failure);
	require(excluded && serviced && producer_allocations == 0 && !box->pending(),
		"VM-held notification blocked on owner,allocated,or ran service before VM exclusion ended");
	refused([&] { box->drain([](auto, auto) {}); }); // Main thread is not the bound owner.
}
void concurrent_pages()
{
	auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
	std::array<std::vector<bool>, 2> seen, expected;
	for (unsigned i = 0; i < 2; ++i) { seen[i].resize(mapping_notifications::page_count); expected[i].resize(mapping_notifications::page_count); }
	std::array<std::thread, 4> producers;
	std::array<unsigned, 4> producer_allocations{};
	std::atomic<unsigned> done{0};
	for (unsigned thread = 0; thread < 4; ++thread)
	{
		for (unsigned n = 0; n < 1024; ++n) expected[(thread + n) % 2][(thread * 262139 + n * 4093) % mapping_notifications::page_count] = true;
		producers[thread] = std::thread([&, thread]
		{
			allocations = 0; track_allocations = true;
			for (unsigned n = 0; n < 1024; ++n)
				box->notify({static_cast<std::uint32_t>(((thread * 262139 + n * 4093) % mapping_notifications::page_count) * 4096), 1}, (thread + n) % 2 ? kind::unmap : kind::read);
			track_allocations = false; producer_allocations[thread] = allocations; ++done;
		});
	}
	const auto service = [&](auto range, auto cause)
	{
		for (std::uint64_t page = range.address / 4096, end = page + range.size / 4096; page < end; ++page)
		{
			require(page < mapping_notifications::page_count, "Concurrent mailbox published an unrelated page");
			seen[cause == kind::read ? 0 : 1][page] = true;
		}
	};
	while (done.load() != 4) { box->drain(service, 7); std::this_thread::yield(); }
	for (auto& thread : producers) thread.join();
	while (box->pending()) box->drain(service, 7);
	require(seen == expected && std::all_of(producer_allocations.begin(), producer_allocations.end(), [](auto n) { return n == 0; }),
		"Concurrent bounded mapping drains lost pages/kinds or allocated on producers");
}
void repeated_generations()
{
	auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
	const std::array pages{1u, 65u, 4097u, 1048575u};
	std::array<std::atomic<unsigned>, 4> acknowledgements{};
	std::array<std::thread, 4> producers;
	std::atomic<bool> stop{false};
	std::atomic<unsigned> done{0};
	for (unsigned i = 0; i < 4; ++i) producers[i] = std::thread([&, i]
	{
		for (unsigned generation = 1; generation <= 1024 && !stop.load(); ++generation)
		{
			box->notify({pages[i] * 4096, 1}, kind::read);
			while (acknowledgements[i].load() < generation && !stop.load()) std::this_thread::yield();
		}
		++done;
	});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	bool unrelated = false;
	while (done.load() < 4 && std::chrono::steady_clock::now() < deadline)
	{
		box->drain([&](auto range, auto)
		{
			const auto found = std::find(pages.begin(), pages.end(), range.address / 4096);
			if (found == pages.end() || range.size != 4096) unrelated = true;
			else ++acknowledgements[found - pages.begin()];
		}, 1);
		std::this_thread::yield();
	}
	stop = true;
	for (auto& producer : producers) producer.join();
	while (box->pending()) box->drain([](auto, auto) {});
	require(!unrelated && std::all_of(acknowledgements.begin(), acknowledgements.end(), [](const auto& n) { return n.load() == 1024; }),
		"Repeated page generation lost its wake or was delivered twice");
	std::cout << "4096 acknowledged repeated page generations across four bitmap branches passed\n";
}
void sparse_summaries()
{
	auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
	for (unsigned page : {0u, 4095u, 4096u, 262144u, 1048575u})
	{
		const auto before = box->statistics();
		box->notify({page * 4096, 1}, kind::unmap);
		require(box->drain([&](auto range, auto cause)
		{
			require(range.address == page * 4096 && range.size == 4096 && cause == kind::unmap, "Sparse summary selected a wrong leaf");
		}) == 1, "Sparse summary lost its page");
		const auto after = box->statistics();
		require(after.root_words - before.root_words == 8 && after.summary_words - before.summary_words == 1 &&
			after.page_words - before.page_words == 1, "Single notification scanned unrelated page words");
	}
	// Distinct words, summaries and roots must survive both an early budget
	// return and a failed callback, including the root's locally claimed bits.
	const std::array pages{0u, 2u, 64u, 4096u, 262144u, 1048575u};
	for (const auto page : pages) box->notify({page * 4096, 1}, kind::read);
	std::vector<unsigned> delivered;
	refused([&] { box->drain([&](auto, auto) { throw std::runtime_error("Summary restoration test"); }); });
	while (box->pending()) box->drain([&](auto range, auto) { delivered.push_back(range.address / 4096); }, 1);
	require(std::equal(delivered.begin(), delivered.end(), pages.begin(), pages.end()), "Budget/failure lost claimed root or summary bits");
	const auto before = box->statistics();
	box->notify({0, 0x1'0000'0000ull}, kind::read);
	box->drain([](auto, auto) {}, mapping_notifications::maximum_ranges);
	const auto after = box->statistics();
	require(after.root_words - before.root_words == 8 && after.summary_words - before.summary_words == 256 &&
		after.page_words - before.page_words == 16384, "Full-VM summary traversal missed leaves or scanned the other cause");
	std::cout << "Sparse mapping drain:8 root+1 summary+1 page-word exchanges;full4GiB:8+256+16384;cross-root failure/budget restoration passed\n";
}
void poison()
{
	for (bool bad_kind : {false, true})
	{
		auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
		box->notify({4096, 1}, kind::read);
		box->notify(bad_kind ? guest_surface_range{0, 1} : guest_surface_range{0xffff'ffff, 2},
			bad_kind ? static_cast<kind>(9) : kind::unmap);
		bool invoked = false;
		refused([&] { box->drain([&](auto, auto) { invoked = true; }); });
		require(!invoked && box->pending(), "Poisoned mapping range reached service or lost pending state");
	}
}
void captured_owner_defers_service()
{
	auto provider = std::make_shared<test::host_access_fixture>();
	auto box = std::make_unique<mapping_notifications>(); box->bind_owner();
	box->notify({4096, 16}, kind::read);
	bool invoked = false;
	provider->execute([&](auto&)
	{
		require(guest_memory_epoch::active(), "Fixture failed to install its actual provider epoch");
		require(box->drain([&](auto, auto) { invoked = true; }) == 0 && box->pending(),
			"Mapping service executed or lost wake inside provider VM capture");
	});
	require(!invoked && !guest_memory_epoch::active(), "Provider capture ran service or leaked its epoch");
	box->drain([&](auto, auto) { invoked = true; require(!guest_memory_epoch::active(), "Mapping service reentered captured VM"); });
	require(invoked && !box->pending(), "Deferred mapping service did not run after capture release");
}
} // namespace
int main()
{
	try
	{
		boundaries_and_priority(); sparse_summaries(); repeated_generations(); bounded_failure_and_reentry(); poison(); captured_owner_defers_service();
		for (unsigned schedule = 0; schedule < 32; ++schedule) nonwaiting_vm_handoff();
		for (unsigned schedule = 0; schedule < 16; ++schedule) concurrent_pages();
		std::cout << "Native mapping notifications:zero producer allocations;fixed lock-free page bitmap;read/unmap priority,edges/full4GiB,duplicates,bounded passes;failed service retains concurrent/unprocessed bits;owner affinity and nonrecursive service;32VM-held producer/owner schedules;16x4096concurrent notifications spanning all4GiB with exact page/kind coverage;poison refused;no App or guest VM started\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
