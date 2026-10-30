#include "M4HostAccessFixture.h"
#include "M4HostStoreTransaction.h"
#include "M4RenderTasks.h"
#include <atomic>
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
using test::host_access_fixture;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refused(F&& operation)
{
	bool caught = false;
	try { operation(); } catch (const std::exception&) { caught = true; }
	require(caught, "Unsafe staged host instruction or recovery was accepted");
}
auto mapping_access(const std::shared_ptr<host_access_fixture>& provider)
{
	return [provider]() -> std::shared_ptr<void>
	{
		auto guard = std::make_shared<std::unique_lock<std::mutex>>(provider->mapping_mutex, std::try_to_lock);
		if (!guard->owns_lock()) return {};
		return guard;
	};
}
void actual_instruction_exclusion()
{
	auto access = std::make_shared<host_access_fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(access);
	const auto watched = memory->watch_surface(guest_byte_coverage({64, 16}));
	const auto before = memory->surface_revision(watched);
	std::fill_n(access->bytes + 128, 16, std::byte{0x51});
	auto destination = memory->prepare_host_access({64, 16}, true);
	auto source = memory->prepare_host_access({128, 16}, false);
	std::atomic<bool> attempted{false}, excluded{false}, entered{false};
	std::thread mapper;
	{
		auto ticket = access->gate->admit();
		auto writer = ticket.activate();
		auto instruction = host_store_transaction::prepare(memory, destination, source, mapping_access(access));
		require(!memory->pending_host_writes(), "Staged preparation armed the journal before mapping exclusion");
		require(instruction.start(), "Available mapping exclusion was refused");
		std::memcpy(access->bytes + 64, access->bytes + 128, 8);
		mapper = std::thread([&]
		{
			if (access->mapping_mutex.try_lock()) access->mapping_mutex.unlock();
			else excluded = true;
			attempted = true;
			std::lock_guard lock(access->mapping_mutex);
			entered = true;
		});
		while (!attempted.load()) std::this_thread::yield();
		const bool protected_instruction = excluded.load() && !entered.load();
		std::memcpy(access->bytes + 72, access->bytes + 136, 8);
		instruction.commit();
		mapper.join();
		require(protected_instruction, "VM mapping exclusion ended before the actual host instruction completed");
	}
	require(entered && memory->surface_revision(watched) > before && !memory->pending_host_writes() &&
		std::all_of(access->bytes + 64, access->bytes + 80, [](auto byte) { return byte == std::byte{0x51}; }),
		"Actual VM-excluded memcpy lost completion revisions or bytes");
}
void busy_and_cancelled()
{
	auto access = std::make_shared<host_access_fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(access);
	auto prepared = memory->prepare_host_access({64, 16}, true);
	{
		auto cancelled = host_store_transaction::prepare(memory, prepared, {}, mapping_access(access));
	}
	require(!memory->pending_host_writes(), "Unstarted cancellation invented a partial store");
	std::atomic<bool> ready{false}, release{false};
	std::thread mapper([&]
	{
		std::lock_guard lock(access->mapping_mutex);
		ready = true;
		while (!release.load()) std::this_thread::yield();
		memory->begin_mapping_change({0, 65536});
		access->replace_first();
		memory->end_mapping_change({0, 65536});
	});
	while (!ready.load()) std::this_thread::yield();
	bool denied = false, retried = false;
	{
		auto ticket = access->gate->admit();
		auto writer = ticket.activate();
		{
			auto instruction = host_store_transaction::prepare(memory, prepared, {}, mapping_access(access));
			denied = !instruction.start();
		}
		release = true; mapper.join();
		const bool untouched = !memory->pending_host_writes() && access->bytes[64] == std::byte{0};
		std::optional<host_write_intent> instruction;
		{
			auto paused = access->gate->pause_current_fault();
			require(bool(paused), "Busy producer could not yield before fresh preparation");
			auto capture = paused->capture_context();
			require(bool(capture), "Busy producer lost owner re-preparation context");
			auto fresh = memory->prepare_host_access({64, 16}, true);
			instruction.emplace(host_store_transaction::prepare(memory, std::move(fresh), {}, mapping_access(access)));
		}
		retried = instruction->start();
		if (retried) { std::fill_n(access->bytes + 64, 16, std::byte{0x3e}); instruction->commit(); }
		require(denied && untouched, "VM-busy instruction was allowed,blocked or left a dirty journal intent");
	}
	require(retried && memory->read({64, 16}).bytes[0] == std::byte{0x3e} && !memory->pending_host_writes(),
		"Busy producer failed fresh re-preparation/retry after a VM remap");
	access->execute([&](auto&) { memory->notify_mapping_changed({0, 65536}); });
	{
		auto ticket = access->gate->admit();
		auto writer = ticket.activate();
		auto stale = host_store_transaction::prepare(memory, prepared, {}, mapping_access(access));
		refused([&] { (void)stale.start(); });
	}
	require(!memory->pending_host_writes(), "Stale staged start armed an intent");
	// Mapping exclusion rolled back even when journal preparation threw.
	std::thread check([&] { const bool unlocked = access->mapping_mutex.try_lock(); require(unlocked, "Failed staged start leaked mapping exclusion"); access->mapping_mutex.unlock(); });
	check.join();
}
void fault_service(bool remap, bool abort_after_service = false)
{
	auto access = std::make_shared<host_access_fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(access);
	const auto watched = memory->watch_surface(guest_byte_coverage({64, 16}));
	const auto alias = memory->watch_surface(guest_byte_coverage({0x40040, 16}));
	const auto before = memory->surface_revision(watched), before_alias = memory->surface_revision(alias);
	std::fill_n(access->bytes + 128, 16, std::byte{0x63});
	auto destination = memory->prepare_host_access({64, 16}, true);
	auto source = memory->prepare_host_access({128, 16}, false);
	auto* old_bytes = destination.backing->segments()[0].data();
	std::weak_ptr<void> old_owner = destination.backing->segments()[0].owner;
	render_tasks tasks([] {}); tasks.bind_owner();
	std::atomic<bool> done{false}, started{false};
	std::exception_ptr failure;
	std::uint64_t partial_revision = 0;
	std::thread producer([&]
	{
		try
		{
			auto ticket = access->gate->admit();
			auto writer = ticket.activate();
			auto instruction = host_store_transaction::prepare(memory, std::move(destination), std::move(source), mapping_access(access));
			require(instruction.start(), "Available mapping exclusion was refused");
			std::memcpy(access->bytes + 64, access->bytes + 128, 8);
			started = true;
			{
				auto store_fault = host_store_transaction::pause_current(*memory);
				require(bool(store_fault), "Fault did not release the native VM guard");
				auto writer_fault = access->gate->pause_current_fault();
				require(bool(writer_fault), "Fault did not suspend its continuous host ticket");
				tasks.invoke([&]
				{
					auto capture = writer_fault->capture_context();
					require(bool(capture), "Fault owner lost its suspended capture context");
					refused([&] { (void)memory->read_surface(watched); }); // No wildcard permission outside service.
					if (remap)
					{
						access->execute([&](auto&)
						{
							memory->begin_mapping_change({0, 65536});
							access->replace_first();
							memory->end_mapping_change({0, 65536});
						});
						store_fault->service([&] { refused([&] { (void)memory->read_surface(watched); }); });
					}
					else store_fault->service([&]
					{
						const auto prefix = memory->read_surface(watched);
						partial_revision = prefix.revision;
						require(partial_revision > before && memory->surface_revision(alias) > before_alias && memory->pending_host_writes() == 1 &&
							prefix.bytes[0] == std::byte{0x63} && prefix.bytes[7] == std::byte{0x63} && prefix.bytes[8] == std::byte{0},
							"Paused prefix capture lost revisions,partial bytes or retained intent");
						require(memory->surface_revision(watched) == partial_revision, "Repeated fault capture invented another prefix write");
					});
					refused([&] { (void)memory->read_surface(watched); }); // Permission retired after service.
				});
				if (remap) require(!store_fault->resume(), "Remapped instruction resumed its stale guest mapping");
				else if (!abort_after_service) require(store_fault->resume(), "Stable fault instruction failed to reacquire and validate mapping");
			}
			if (!remap && !abort_after_service)
			{
				std::memcpy(access->bytes + 72, access->bytes + 136, 8);
				instruction.commit();
			}
		}
		catch (...) { failure = std::current_exception(); }
		done = true;
	});
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (!started.load() && !done.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
	bool captured = false;
	{
		auto excluded = access->gate->acquire_waiting_faults([&] { tasks.drain(); std::this_thread::yield(); }, std::chrono::seconds(10));
		captured = bool(excluded);
	}
	if (!captured) tasks.close();
	producer.join();
	require(captured, "Normal owner capture did not pump/retry an external paused producer");
	if (failure) std::rethrow_exception(failure);
	if (remap || abort_after_service)
	{
		refused([&] { (void)memory->read_surface(watched); });
		require(memory->pending_host_writes() == 1 && !old_owner.expired() && old_bytes[0] == std::byte{0x63} && old_bytes[8] == std::byte{0},
			"Failed/unresumed fault discarded original dirty physical bytes or claimed completion");
		if (remap) require(access->bytes[64] == std::byte{0}, "Failed retry wrote the replacement allocation");
	}
	else require(memory->surface_revision(watched) > partial_revision && !memory->pending_host_writes() && access->bytes[79] == std::byte{0x63},
		"Retried instruction lost final bytes or post-prefix completion revision");
}
void multiple_prefix_faults()
{
	auto access = std::make_shared<host_access_fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(access);
	const auto watched = memory->watch_surface(guest_byte_coverage({64, 16}));
	auto revision = memory->surface_revision(watched);
	std::fill_n(access->bytes + 128, 16, std::byte{0x42});
	auto destination = memory->prepare_host_access({64, 16}, true);
	auto source = memory->prepare_host_access({128, 16}, false);
	{
		auto ticket = access->gate->admit(); auto writer = ticket.activate();
		auto instruction = host_store_transaction::prepare(memory, std::move(destination), std::move(source), mapping_access(access));
		require(instruction.start(), "Available mapping exclusion was refused");
		for (unsigned prefix = 4; prefix <= 8; prefix += 4)
		{
			std::memcpy(access->bytes + 60 + prefix, access->bytes + 124 + prefix, 4);
			auto store_fault = host_store_transaction::pause_current(*memory);
			require(bool(store_fault), "Repeated prefix fault lost its transaction");
			auto writer_fault = access->gate->pause_current_fault();
			require(bool(writer_fault), "Repeated prefix fault lost its ticket");
			{
				auto capture = writer_fault->capture_context();
				require(bool(capture), "Repeated prefix fault lost capture context");
				store_fault->service([&]
				{
					const auto bytes = memory->read_surface(watched);
					require(bytes.revision > revision && bytes.bytes[prefix - 1] == std::byte{0x42} && bytes.bytes[prefix] == std::byte{0},
						"Later fault prefix kept an earlier partial revision");
					revision = bytes.revision;
				});
			}
			if (prefix == 4)
			{
				std::atomic<bool> ready{false}, release{false};
				std::thread mapper([&]
				{
					std::lock_guard lock(access->mapping_mutex);
					ready = true;
					while (!release.load()) std::this_thread::yield();
				});
				while (!ready.load()) std::this_thread::yield();
				const auto result = store_fault->attempt_resume();
				release = true; mapper.join();
				require(result == host_store_transaction::resume_result::busy, "Transient fault-resume contention was not distinguished from remap");
			}
			require(store_fault->resume(), "Repeated prefix fault could not reacquire mapping");
		}
		std::memcpy(access->bytes + 72, access->bytes + 136, 8);
		instruction.commit();
	}
	require(memory->surface_revision(watched) > revision && !memory->pending_host_writes(), "Repeated prefix final completion lost revision");
}

void source_only()
{
	auto access = std::make_shared<host_access_fixture>();
	access->gate = std::make_shared<host_writer_gate>();
	auto memory = std::make_shared<guarded_guest_memory>(access);
	const auto watched = memory->watch_surface(guest_byte_coverage({128, 16}));
	const auto revision = memory->surface_revision(watched);
	std::fill_n(access->bytes + 128, 16, std::byte{0x2d});
	auto source = memory->prepare_host_access({128, 16}, false);
	std::array<std::byte, 16> bytes{};
	{
		auto ticket = access->gate->admit(); auto writer = ticket.activate();
		auto instruction = host_store_transaction::prepare(memory, {}, std::move(source), mapping_access(access));
		require(instruction.start(), "Available mapping exclusion was refused"); std::memcpy(bytes.data(), access->bytes + 128, 8);
		{
			auto fault = host_store_transaction::pause_current(*memory);
			require(bool(fault), "Source-only fault lost mapping exclusion");
			auto writer_fault = access->gate->pause_current_fault();
			require(bool(writer_fault), "Source-only fault lost host ticket");
			{
				auto capture = writer_fault->capture_context();
				require(bool(capture), "Source-only fault lost owner capture");
				fault->service([&] { require(memory->surface_revision(watched) == revision, "Source-only fault invented a prefix guest write"); });
			}
			require(fault->resume(), "Source-only fault failed stable retry");
		}
		std::memcpy(bytes.data() + 8, access->bytes + 136, 8); instruction.commit();
	}
	require(memory->surface_revision(watched) == revision && !memory->pending_host_writes() && bytes[15] == std::byte{0x2d},
		"Source-only VM exclusion fabricated guest revisions or copied wrong bytes");
}
} // namespace
int main()
{
	try
	{
		actual_instruction_exclusion(); busy_and_cancelled();
		for (unsigned schedule = 0; schedule < 32; ++schedule) fault_service(false);
		fault_service(true); fault_service(false, true); multiple_prefix_faults(); source_only();
		std::cout << "Staged host instruction:nonwaiting mapping exclusion covers real memcpy;busy/stale/unstarted refusal before dirty intent;busy-remap fresh preparation and retry under continuous ticket;fault unlock before synchronous owner capture;normal captures pump external paused work across32 schedules including cleared-fault transition;leased prefix/alias revisions retain intent;scoped recovery;each repeated prefix advances revisions;stable reacquire/retry/final revision;remap/unresumed failure retains old physical dirty bytes and stops replacement writes;source-only no fabricated revision;CPU OS fixtures,no App or guest VM started\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
