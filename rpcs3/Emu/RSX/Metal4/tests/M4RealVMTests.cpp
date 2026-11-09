#include "stdafx.h"
#include "M4RSXVMAccess.h"
#include "M4DrawClauseDriver.h"
#include "Emu/RSX/NV47/FW/draw_call.hpp"
#include "M4RSXMappingObserver.h"
#include "M4HostWriterGate.h"
#include "M4HostStoreTransaction.h"
#include "M4RSXHostWrite.h"
#include "M4RenderTasks.h"
#include "Emu/system_config.h"

#include <iostream>
#include <future>
#include <mach/mach.h>
#include <mach/mach_vm.h>

extern atomic_t<bool> g_headless;
void test_real_vm_cache_watches();
void test_real_vm_cache_faults();
void test_arm64_memory_instructions();
void test_rsx_overlay_icon();

namespace
{
// VM address spaces are reserved by static initializers, before main. Keep
// their failure path headless as well as the explicitly executed test cases.
struct headless_diagnostics
{
	headless_diagnostics() { g_headless = true; }
} diagnostics;
using namespace metal4;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template <typename Operation> void refuses(Operation&& operation, const char* message)
{
	bool failed = false; try { operation(); } catch (const std::exception&) { failed = true; }
	require(failed, message);
}
struct fixture
{
	static constexpr std::uint32_t base = 0x10000000, alias = base + 65536, second = base + 131072;
	std::shared_ptr<vm::block_t> block;
	std::shared_ptr<host_writer_gate> gate = std::make_shared<host_writer_gate>();
	unsigned guard_calls = 0;
	std::shared_ptr<rsx_vm_access> access;
	std::shared_ptr<guarded_guest_memory> memory;
	std::shared_ptr<rsx_mapping_observer> mappings;
	explicit fixture(u64 flags = vm::block_size_64k | vm::bf0_0x2,
		std::function<std::shared_ptr<void>()> external_guard = {}, std::shared_ptr<rsx_page_protection> protection = {})
	{
		block = vm::map(base, 4 * 65536, flags);
		require(bool(block), "Real VM test block could not be mapped");
		access = std::make_shared<rsx_vm_access>([&, external_guard = std::move(external_guard)]() -> std::shared_ptr<void>
		{
			++guard_calls;
			if (external_guard) return external_guard();
			auto guard = gate->acquire([] {}, std::chrono::seconds(1));
			if (!guard) return {};
			return std::make_shared<host_writer_gate::exclusion>(std::move(*guard));
		}, 512u << 20, 1024, protection);
		memory = std::make_shared<guarded_guest_memory>(access);
		mappings = std::make_shared<rsx_mapping_observer>(memory, protection);
	}
	~fixture()
	{
		mappings.reset(); memory.reset(); access.reset();
		if (block) { (void)vm::unmap(base, false, &block); block.reset(); }
	}
	void allocate_pair()
	{
		auto shared = std::make_shared<utils::shm>(65536);
		// Imported allocations take their page-size flags from this argument,
		// unlike private allocations which inherit the block's flags.
		require(block->falloc(base, 65536, &shared, vm::block_size_64k) &&
			block->falloc(alias, 65536, &shared, vm::block_size_64k) && block->falloc(second, 65536),
			"Real VM shared/disjoint allocations failed");
	}
};
int actual_host_protection(std::uint32_t guest)
{
	mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(vm::g_base_addr + guest);
	const auto requested = address;
	mach_vm_size_t size = 0;
	vm_region_basic_info_data_64_t info{};
	mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
	mach_port_t object = MACH_PORT_NULL;
	const auto result = ::mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
		reinterpret_cast<vm_region_info_t>(&info), &count, &object);
	if (object != MACH_PORT_NULL) (void)::mach_port_deallocate(mach_task_self(), object);
	require(result == KERN_SUCCESS && address <= requested && requested - address < size, "Cannot query actual native VM protection");
	return info.protection;
}
void actual_page_protection()
{
	{
	fixture f; f.allocate_pair(); f.mappings.reset();
	auto service = std::make_shared<rsx_page_protection>(4);
	f.mappings = std::make_shared<rsx_mapping_observer>(f.memory, service);
	const auto host_page = service->page_size();
	require(host_page == static_cast<std::size_t>(utils::get_page_size()) && host_page >= 4096, "Native protection used logical instead of actual host page size");
	using mode = rsx_page_protection::access;
	rsx_page_protection::token dirty, clean;
	{
		const vm::writer_lock lock;
		const auto before = service->statistics(lock);
		require(before.refreshed_logical_pages == (1u << 20), "Native protection did not reconcile the address space once on bind");
		dirty = service->protect({fixture::base + 17, 1}, mode::trap_reads_and_writes, lock);
		clean = service->protect({fixture::base + 18, 1}, mode::trap_writes, lock);
		const auto after = service->statistics(lock);
		require(after.refreshed_logical_pages == before.refreshed_logical_pages &&
			after.protection.native_pages - before.protection.native_pages == 4 &&
			after.protection.logical_pages - before.protection.logical_pages == 4 * host_page / 4096 &&
			after.protection.protection_runs - before.protection.protection_runs == 4 &&
			after.visited_index_nodes > before.visited_index_nodes &&
			after.visited_index_nodes - before.visited_index_nodes < 256,
			"Native protection scanned beyond the two physical aliases for each watch");
		std::cout << "Real VM protection index: two watches touch four native pages in four protection calls and visit "
			<< after.visited_index_nodes - before.visited_index_nodes << " index nodes\n";
		const auto tokens = service->fault_tokens(fixture::alias + static_cast<u32>(host_page - 1), lock);
		require(tokens.size() == 2 && std::find(tokens.begin(), tokens.end(), dirty) != tokens.end() &&
			std::find(tokens.begin(), tokens.end(), clean) != tokens.end(), "Native fault selection omitted physical aliases or host-page neighbours");
		refuses([&] { (void)service->protect({fixture::base + 4 * 65536 - 1, 2}, mode::trap_reads_and_writes, lock); },
			"Native protection accepted partly absent allocation");
		refuses([&] { service->release({}, lock); }, "Native protection accepted a foreign token");
		const auto extra1 = service->protect({fixture::base + 19, 1}, mode::trap_writes, lock);
		const auto extra2 = service->protect({fixture::base + 20, 1}, mode::trap_writes, lock);
		refuses([&] { (void)service->protect({fixture::base + 21, 1}, mode::trap_writes, lock); },
			"Native protection exceeded its fixed segment budget");
		service->release(extra1, lock); service->release(extra2, lock);
	}
	require(actual_host_protection(fixture::base) == VM_PROT_NONE && actual_host_protection(fixture::alias) == VM_PROT_NONE &&
		actual_host_protection(fixture::second) == (VM_PROT_READ | VM_PROT_WRITE) &&
		actual_host_protection(fixture::base + static_cast<u32>(host_page)) == (VM_PROT_READ | VM_PROT_WRITE) &&
		vm::check_addr(fixture::base, vm::page_readable | vm::page_writable, 65536),
		"Native physical overlay missed an alias,changed logical VM flags or protected unrelated pages");
	vm::g_sudo_addr[fixture::base + 17] = 0x59;
	std::byte byte{};
	mach_vm_size_t copied = 0;
	require(::mach_vm_read_overwrite(mach_task_self(), reinterpret_cast<mach_vm_address_t>(vm::g_base_addr + fixture::base + 17),
		1, reinterpret_cast<mach_vm_address_t>(&byte), &copied) != KERN_SUCCESS,
		"Actual kernel read bypassed native read protection");
	{
		const vm::writer_lock lock; service->release(dirty, lock);
		refuses([&] { service->release(dirty, lock); }, "Native protection accepted a retired token");
	}
	require(actual_host_protection(fixture::base) == VM_PROT_READ && actual_host_protection(fixture::alias) == VM_PROT_READ &&
		vm::g_base_addr[fixture::alias + 17] == 0x59, "Releasing strong native overlay lost weaker protection or physical bytes");
	copied = 0;
	require(::mach_vm_read_overwrite(mach_task_self(), reinterpret_cast<mach_vm_address_t>(vm::g_base_addr + fixture::base + 17),
		1, reinterpret_cast<mach_vm_address_t>(&byte), &copied) == KERN_SUCCESS && copied == 1 && byte == std::byte{0x59},
		"Kernel protection probe failed its readable positive control");
	// A new alias must be protected by the productive mapping callback,without
	// manually notifying the service and without retaining the allocation owner.
	std::shared_ptr<utils::shm> shared;
	{
		const vm::writer_lock lock; shared = vm::get_backing_segment(fixture::base, 1, lock).memory;
	}
	const auto third = fixture::base + 3 * 65536;
	const auto before_late_map = [&] { const vm::writer_lock lock; return service->statistics(lock); }();
	require(f.block->falloc(third, 65536, &shared, vm::block_size_64k), "Could not map late physical alias");
	const auto after_late_map = [&] { const vm::writer_lock lock; return service->statistics(lock); }();
	require(after_late_map.refreshed_logical_pages - before_late_map.refreshed_logical_pages == 16 &&
		after_late_map.mapped_logical_pages - before_late_map.mapped_logical_pages == 16,
		"Late native alias refreshed beyond the changed 64-KiB mapping");
	shared.reset();
	require(actual_host_protection(third) == VM_PROT_READ, "Productive VM callback exposed a new unprotected physical alias");
	require(vm::page_protect(fixture::alias, 65536, vm::page_allocated, 0, vm::page_writable), "Could not restrict logical alias permissions");
	require(f.block->dealloc(fixture::base) == 65536 && f.block->falloc(fixture::base, 65536), "Could not replace protected guest VA");
	require(actual_host_protection(fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && actual_host_protection(third) == VM_PROT_READ,
		"Native protection attached old physical watch to a replacement VA or lost surviving alias");
	{
		const vm::writer_lock lock; service->release(clean, lock);
	}
	require(actual_host_protection(fixture::alias) == VM_PROT_READ && actual_host_protection(third) == (VM_PROT_READ | VM_PROT_WRITE),
		"Native overlay release broadened current logical VM permissions");
	{
		const vm::writer_lock lock;
		(void)service->protect({third + 17, 1}, mode::trap_reads_and_writes, lock);
	}
	f.mappings.reset();
	require(actual_host_protection(fixture::alias) == VM_PROT_READ && actual_host_protection(third) == (VM_PROT_READ | VM_PROT_WRITE),
		"Native observer shutdown failed to restore live current mappings");
	f.mappings = std::make_shared<rsx_mapping_observer>(f.memory, service);
	{
		const vm::writer_lock lock;
		refuses([&] { service->release(clean, lock); }, "Rebound native observer accepted old protection token");
		(void)service->protect({third + 17, 1}, mode::trap_reads_and_writes, lock);
	}
	// Weak metadata must not trip the actual VM's strict external-shm-owner
	// check when the last physical aliases disappear with a live watch.
	require(bool(vm::unmap(fixture::base, false, &f.block).first), "Whole-block unmap rejected native weak watch identities");
	f.block.reset();
	f.mappings.reset();
	require(!service->failed(), "Native protection latched unresolved allocation metadata");
	std::cout << "Real VM protection index: one initial 1048576-page reconciliation; late alias refreshes 16 logical pages\n";
	{
		const vm::writer_lock lock;
		refuses([&] { (void)service->protect({fixture::base, 1}, mode::trap_writes, lock); }, "Detached native protection remained available");
	}
	}
	{
		fixture f(vm::block_size_4k | vm::preallocated | vm::bf0_0x2); f.mappings.reset();
		auto service = std::make_shared<rsx_page_protection>(2);
		f.mappings = std::make_shared<rsx_mapping_observer>(f.memory, service);
		const auto size = static_cast<u32>(service->page_size());
		for (u32 offset = 0; offset < 2 * size; offset += 4096)
			require(f.block->falloc(fixture::base + offset, 4096), "Could not allocate native subpage fixture");
		rsx_page_protection::token first, second;
		{
			const vm::writer_lock lock;
			first = service->protect({fixture::base + 17, 1}, rsx_page_protection::access::trap_reads_and_writes, lock);
			second = service->protect({fixture::base + size + 17, 1}, rsx_page_protection::access::trap_writes, lock);
		}
		require(actual_host_protection(fixture::base) == VM_PROT_NONE && actual_host_protection(fixture::base + size) == VM_PROT_READ,
			"Native common-backing offsets collapsed disjoint host pages");
		if (size > 4096)
			require(vm::page_protect(fixture::base + 4096, 4096, vm::page_allocated, 0, vm::page_readable | vm::page_writable),
				"Could not restrict logical subpage neighbour");
		{
			const vm::writer_lock lock; service->release(first, lock); service->release(second, lock);
		}
		require(actual_host_protection(fixture::base) == (size > 4096 ? VM_PROT_NONE : VM_PROT_READ | VM_PROT_WRITE) &&
			actual_host_protection(fixture::base + size) == (VM_PROT_READ | VM_PROT_WRITE),
			"Native release broadened hidden logical neighbour or failed disjoint restoration");
		if (size > 4096)
			require(vm::page_protect(fixture::base + 4096, 4096, vm::page_allocated, vm::page_readable | vm::page_writable, 0),
				"Could not restore logical subpage fixture");
	}
	std::cout << "Real VM native protection:actual host-page granularity;NO/RO overlays;kernel read refusal;sudo unchanged;physical aliases and late maps;host-neighbour fault tokens;weak allocation lifetime;VA replacement;current logical permission restoration;whole-block retirement;owned observer shutdown/rebind;fixed budget;common-backing disjoint offsets;mixed logical subpages fail closed;no CPU fault dispatch yet\n";
}
void actual_watch_protection()
{
	{
		auto service = std::make_shared<rsx_page_protection>(8);
		fixture f(vm::block_size_64k | vm::bf0_0x2, {}, service); f.allocate_pair();
		const auto page = static_cast<u32>(service->page_size());
		const auto shape = plan_guest_surface(fixture::base, 4, 2, 4, 2 * page);
		auto first = f.memory->watch_surface(shape), duplicate = f.memory->watch_surface(shape);
		const auto revision = f.memory->surface_revision(first);
		for (unsigned i = 0; i < 32; ++i) require(f.memory->surface_revision(duplicate) == revision, "Idempotent native watch changed its revision");
		require(first.registration == duplicate.registration && actual_host_protection(fixture::base) == VM_PROT_NONE &&
			actual_host_protection(fixture::alias) == VM_PROT_NONE && actual_host_protection(fixture::base + 2 * page) == VM_PROT_NONE,
			"Actual guarded watch did not install physical CPU protection in its capture");
		if (page < 65536) require(actual_host_protection(fixture::base + page) == (VM_PROT_READ | VM_PROT_WRITE),
			"Native active-byte watch protected a whole padding-only host page");
		{
			const vm::writer_lock lock;
			require(service->fault_tokens(fixture::base, lock).size() == 1, "Repeated native registration consumed duplicate protection slots");
		}
		std::weak_ptr<void> lifetime = first.registration;
		const auto captures = f.guard_calls;
		first.registration.reset(); duplicate.registration.reset();
		require(lifetime.expired() && f.guard_calls == captures && actual_host_protection(fixture::base) == VM_PROT_NONE,
			"Watch destructor entered VM,retained its own owner or reopened CPU memory outside exclusion");
		f.access->capture_cpu([] {});
		require(actual_host_protection(fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) &&
			actual_host_protection(fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE), "Actual suspended provider failed to retire expired watch protection");
		auto renewed = f.memory->watch_surface(shape);
		require(f.memory->surface_revision(renewed) > revision, "Native watch reincarnation reused its old revision");
		require(f.block->dealloc(fixture::base) == 65536 && f.block->falloc(fixture::base, 65536), "Could not replace native watched VA");
		require(actual_host_protection(fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && actual_host_protection(fixture::alias) == VM_PROT_NONE,
			"Native registration followed a replacement VA before capture or lost old physical protection");
		(void)f.memory->surface_revision(renewed);
		require(actual_host_protection(fixture::base) == VM_PROT_NONE && actual_host_protection(fixture::alias) == VM_PROT_NONE,
			"Native watch capture failed to rearm replacement bytes while keeping old physical aliases");
		const auto pinned_revision = f.memory->surface_revision(shape);
		std::weak_ptr<void> pinned = renewed.registration;
		renewed.registration.reset(); f.access->capture_cpu([] {});
		require(!pinned.expired() && actual_host_protection(fixture::base) == VM_PROT_NONE &&
			f.memory->surface_revision(shape) == pinned_revision, "Legacy unleased access failed to pin native watch lifetime/tag");
		f.mappings.reset();
		refuses([&] { (void)std::make_shared<rsx_mapping_observer>(f.memory, service); },
			"Retired native cache domain was rebound across an unobserved CPU interval");
		refuses([&] { f.access->capture_cpu([] {}); }, "Retired native provider reopened old cache revisions");
	}
	{
		auto service = std::make_shared<rsx_page_protection>(1);
		fixture f(vm::block_size_64k | vm::bf0_0x2, {}, service); f.allocate_pair();
		const auto shape = plan_guest_surface(fixture::base, 4, 2, 4, 2 * service->page_size());
		refuses([&] { (void)f.memory->watch_surface(shape); }, "Native multirow registration exceeded segment capacity");
		require(actual_host_protection(fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) &&
			actual_host_protection(fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE), "Failed native multirow watch published a partial OS overlay");
		const auto single = f.memory->watch_surface(plan_guest_surface(fixture::base, 1, 1, 4));
		require(single.registration && actual_host_protection(fixture::base) == VM_PROT_NONE, "Failed native watch consumed capacity or left orphan state");
	}
	std::cout << "Real VM watch protection:actual guarded Surface/Texture registration path;active bytes exclude padding host pages;idempotent weak owner;destructor without VM entry;deferred retirement in real CPU-suspended capture;reincarnation;old physical aliases and replacement rearm;legacy stable pin;terminal cache domain after shutdown;multirow budget refusal before any overlay\n";
}
void actual_fault_receipts()
{
	{
	auto service = std::make_shared<rsx_page_protection>(32);
	fixture f(vm::block_size_64k | vm::bf0_0x2, {}, service); f.allocate_pair();
	const auto page = static_cast<u32>(service->page_size());
	f.access->capture_cpu([&] { std::memset(vm::g_sudo_addr + fixture::base, 0x31, 65536); });
	auto rows = f.memory->watch_surface(plan_guest_surface(fixture::base, 4, 2, 4, 2 * page));
	auto neighbour = f.memory->watch_surface(plan_guest_surface(fixture::base + page - 4, 1, 1, 4));
	auto other = f.memory->watch_surface(plan_guest_surface(fixture::second, 1, 1, 4));
	const auto rows_revision = f.memory->surface_revision(rows), neighbour_revision = f.memory->surface_revision(neighbour);
	const auto other_revision = f.memory->surface_revision(other);
	auto rows_physical = f.memory->capture_physical_surface(rows, rows_revision);
	auto neighbour_physical = f.memory->capture_physical_surface(neighbour, neighbour_revision);
	const auto other_physical = f.memory->capture_physical_surface(other, other_revision);
	auto plan = f.access->capture_fault(fixture::alias + page - 1);
	require(plan && plan->registrations().size() == 2 && plan->identity().segments().size() == 3 && f.access->fault_current(*plan),
		"Native fault plan omitted a host-page neighbour or remote row of its token/registration");
	require(!f.access->capture_fault(fixture::second + 2 * page), "Native fault plan claimed an unprotected page");
	refuses([&] { (void)f.access->capture_fault(fixture::alias, 8); }, "Native fault plan exceeded its aggregate receipt byte budget");
	require(f.access->fault_current(*plan) && actual_host_protection(fixture::base) == VM_PROT_NONE,
		"Failed native fault planning mutated its protection domain");
	guarded_guest_memory foreign(f.access);
	refuses([&] { (void)f.access->invalidate_fault(foreign, *plan); }, "Native fault invalidation accepted foreign registrations");
	require(f.memory->surface_revision(rows) == rows_revision && f.memory->physical_surface_current(*rows_physical),
		"Foreign native fault invalidation partially changed revisions");
	require(f.access->invalidate_fault(*f.memory, *plan), "Current native fault receipt was refused before revision invalidation");
	require(f.memory->surface_revision(rows) > rows_revision && f.memory->surface_revision(neighbour) > neighbour_revision &&
		f.memory->surface_revision(other) == other_revision,
		"Native fault invalidation omitted virtual revisions or changed disjoint storage");
	require(!f.memory->physical_surface_current(*rows_physical) && !f.memory->physical_surface_current(*neighbour_physical) &&
		f.memory->physical_surface_current(*other_physical), "Native fault invalidation omitted physical revisions or changed disjoint storage");
	require(actual_host_protection(fixture::base) == VM_PROT_NONE && actual_host_protection(fixture::alias) == VM_PROT_NONE,
		"Revision-only native fault plan opened CPU memory without instruction completion");
	// Same physical identity and final flags must still reject an ABA interval.
	require(vm::page_protect(fixture::alias, 65536, vm::page_allocated, 0, vm::page_writable) &&
		vm::page_protect(fixture::alias, 65536, vm::page_allocated, vm::page_writable, 0) && !f.access->fault_current(*plan),
		"Native fault receipt accepted a mapping/permission ABA interval");
	plan = f.access->capture_fault(fixture::alias);
	const auto new_watch = f.memory->watch_surface(plan_guest_surface(fixture::base + 8, 1, 1, 4));
	require(!f.access->fault_current(*plan), "Native fault receipt ignored an additional protection owner");
	std::weak_ptr<utils::shm> original;
	{ const vm::writer_lock lock; original = vm::get_backing_segment(fixture::base, 4, lock).memory; }
	require(f.block->dealloc(fixture::base) == 65536 && f.block->falloc(fixture::base, 65536), "Could not replace fault receipt VA");
	f.access->capture_cpu([&] { std::memset(vm::g_sudo_addr + fixture::base, 0x72, 65536); });
	(void)f.memory->surface_revision(rows); (void)f.memory->surface_revision(neighbour);
	plan = f.access->capture_fault(fixture::alias);
	require(plan && plan->identity().segments().size() == 7, "Native fault receipt omitted old aliases or replacement owner closure");
	const auto revision = f.memory->surface_revision(rows);
	const auto kept = f.memory->capture_physical_surface(rows, revision);
	require(f.access->invalidate_fault(*f.memory, *plan) && !f.memory->physical_surface_current(*kept),
		"Native fault receipt did not invalidate replacement physical revision cells");
	rows_physical.reset(); neighbour_physical.reset(); rows.registration.reset(); neighbour.registration.reset();
	// Receipt mappings must not keep the original VM shm alive or impede unmap.
	require(f.block->dealloc(fixture::alias) == 65536 && original.expired(),
		"Owned fault receipt retained a strong VM allocation handle during actual deallocation");
	require(!f.access->fault_current(*plan), "Native fault receipt remained valid after actual alias unmap");
	f.access->capture_cpu([&]
	{
		bool old_bytes = false, replacement_bytes = false;
		for (const auto& part : plan->identity().segments())
		{
			const auto owner = part.owner.lock(); require(bool(owner), "Native fault receipt lost independently retained bytes");
			const auto value = *(static_cast<const u8*>(owner.get()) + part.offset);
			old_bytes |= value == 0x31; replacement_bytes |= value == 0x72;
		}
		require(old_bytes && replacement_bytes, "Native fault receipt followed a reused VA instead of owning both physical incarnations");
	});
	}
	{
		auto service = std::make_shared<rsx_page_protection>(4);
		fixture f(vm::block_size_64k | vm::bf0_0x2, {}, service); f.allocate_pair();
		{ const vm::writer_lock lock; (void)service->protect({fixture::base, 4}, rsx_page_protection::access::trap_writes, lock); }
		const auto plan = f.access->capture_fault(fixture::alias);
		require(plan && plan->registrations().empty(), "Manual native protection could not issue an owned physical fault receipt");
		f.mappings.reset(); f.mappings = std::make_shared<rsx_mapping_observer>(f.memory, service);
		require(!f.access->fault_current(*plan) && !f.access->capture_fault(fixture::base),
			"Manual protection shutdown/rebind revived an old fault receipt");
	}
	{
		auto service = std::make_shared<rsx_page_protection>(4);
		fixture f(vm::block_size_64k | vm::bf0_0x2, {}, service); f.allocate_pair();
		auto watch = f.memory->watch_surface(plan_guest_surface(fixture::base, 1, 1, 4));
		std::weak_ptr<void> registration = watch.registration;
		auto plan = f.access->capture_fault(fixture::alias);
		const auto physical = plan->identity().segments().front().owner;
		watch.registration.reset(); f.access->capture_cpu([] {});
		require(!registration.expired() && !physical.expired() && actual_host_protection(fixture::alias) == VM_PROT_NONE,
			"Owned fault receipt failed to pin its selected registration and independent bytes across handoff");
		const auto captures = f.guard_calls;
		std::thread release([plan = std::move(plan)]() mutable { plan.reset(); }); release.join();
		require(registration.expired() && physical.expired() && captures == f.guard_calls && actual_host_protection(fixture::base) == VM_PROT_NONE,
			"Foreign last fault receipt destructor entered VM or retained its owners");
		f.access->capture_cpu([] {});
		require(actual_host_protection(fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) &&
			actual_host_protection(fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE), "Fault receipt retirement failed to restore aliases in the next suspended epoch");
	}
	std::cout << "Real VM fault receipts:owned host-page neighbour and full token/registration closure;bounded independent physical mappings;virtual AND physical invalidation;disjoint revision unchanged;foreign domain refusal;mapping/permission ABA and added owner refusal;old+replacement physical identities;actual unmap without strong shm retention;manual shutdown/rebind rejects receipts;foreign last-owner retirement;OS protection stays armed;no instruction retry claimed\n";
}
void actual_mapping_hook_pairing()
{
	fixture f; f.allocate_pair(); f.mappings.reset();
	struct probe final : vm::mapping_change_observer
	{
		struct event { std::uint32_t address, size; bool begin, allocated, writable; };
		std::shared_ptr<guarded_guest_memory> memory;
		std::array<event, 6> events{};
		unsigned count = 0;
		bool invalid = false;
		explicit probe(std::shared_ptr<guarded_guest_memory> value) : memory(std::move(value))
		{
			const vm::writer_lock lock;
			require(vm::attach_mapping_observer(*this, lock), "Actual VM probe could not install");
		}
		~probe()
		{
			const vm::writer_lock lock;
			if (!vm::detach_mapping_observer(*this, lock)) std::terminate();
		}
		void record(std::uint32_t address, std::uint32_t size, bool begin) noexcept
		{
			if (vm::host_mapping_guard::try_acquire() || memory->observe_mapping()) invalid = true;
			if (count == events.size()) { invalid = true; return; }
			events[count++] = {address, size, begin, vm::check_addr(address, vm::page_allocated, size), vm::check_addr(address, vm::page_writable, size)};
		}
		void on_mapping_change_begin(std::uint32_t address, std::uint32_t size) noexcept override
		{
			memory->begin_mapping_change({address, size}); record(address, size, true);
		}
		void on_mapping_change_end(std::uint32_t address, std::uint32_t size) noexcept override
		{
			record(address, size, false); memory->end_mapping_change({address, size});
		}
	} sink(f.memory);
	const auto address = fixture::base + 3 * 65536;
	require(f.block->falloc(address, 65536) && vm::page_protect(address, 65536, vm::page_allocated, 0, vm::page_writable) &&
		f.block->dealloc(address) == 65536, "Actual VM probe mutations failed");
	require(!sink.invalid && sink.count == 6 && f.memory->observe_mapping().has_value(), "Actual VM hooks were unpaired,outside writer lock or exposed idle mapping epoch");
	constexpr std::array allocated{false, true, true, true, true, false};
	constexpr std::array writable{false, true, true, false, false, false};
	for (unsigned i = 0; i < sink.count; ++i)
		require(sink.events[i].address == address && sink.events[i].size == 65536 && sink.events[i].begin == (i % 2 == 0) &&
			sink.events[i].allocated == allocated[i] && sink.events[i].writable == writable[i],
			"Actual VM hook range/order did not enclose page metadata mutation");
	std::cout << "Real VM hook pairing:actual map/protect/unmap old/new page flags;6ordered callbacks;actual writer lock held;mapping epoch busy throughout callbacks;no renderer\n";
}

void automatic_mapping_delivery()
{
	fixture f; f.allocate_pair();
	const auto layout = f.memory->watch_surface(guest_byte_coverage({fixture::base + 128, 16}));
	const auto revision = f.memory->surface_revision(layout);
	const auto observed = f.memory->observe_mapping();
	require(observed.has_value() && f.mappings->observes(f.memory), "Actual VM mapping observer was not installed");
	refuses([&] { auto duplicate = std::make_shared<rsx_mapping_observer>(f.memory); }, "Second native VM observer replaced the live owner");
	require(f.block->dealloc(fixture::second) == 65536 && f.block->falloc(fixture::second, 65536), "Unrelated actual VM replacement failed");
	require(!f.memory->mapping_unchanged(*observed) && f.memory->surface_revision(layout) == revision,
		"Automatic VM mapping notification missed transition or invalidated unrelated watch");
	const auto before_protect = f.memory->observe_mapping();
	require(before_protect.has_value() && vm::page_protect(fixture::base, 65536, vm::page_allocated, 0, vm::page_writable), "Actual readonly transition failed");
	require(!f.memory->mapping_unchanged(*before_protect) && f.memory->surface_revision(layout) > revision,
		"Actual protection hook did not update native mapping/watch epochs");
	const auto stable = f.memory->observe_mapping();
	require(stable.has_value() && vm::page_protect(fixture::base, 65536, vm::page_allocated, 0, vm::page_writable) &&
		!vm::page_protect(fixture::base + 3 * 65536, 65536, vm::page_allocated, 0, vm::page_writable) &&
		!f.block->falloc(fixture::base, 65536) && f.memory->mapping_unchanged(*stable),
		"No-op or refused VM mutation fabricated native mapping transition");
	const auto before_remap = f.memory->surface_revision(layout);
	require(f.block->dealloc(fixture::base) == 65536 && f.block->falloc(fixture::base, 65536), "Actual native observer remap failed");
	require(f.memory->surface_revision(layout) > before_remap, "Automatic unmap/map ABA retained prior watch revision");
	// Removal cannot release the callback's memory while a real VM writer is
	// active. The foreign thread starts teardown before exclusion is released.
	std::promise<void> entered;
	std::future<void> finished;
	{
		const vm::writer_lock lock;
		finished = std::async(std::launch::async, [observer = std::move(f.mappings), &entered]() mutable
		{
			entered.set_value(); observer.reset();
		});
		entered.get_future().wait();
		require(finished.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout, "Native observer detached outside VM writer exclusion");
	}
	finished.get();
	const auto detached = f.memory->observe_mapping();
	require(detached.has_value() && vm::page_protect(fixture::base, 65536, vm::page_allocated, 0, vm::page_writable) &&
		f.memory->mapping_unchanged(*detached), "Detached native observer remained reachable by actual VM producers");
	f.mappings = std::make_shared<rsx_mapping_observer>(f.memory);
	require(vm::page_protect(fixture::base, 65536, vm::page_allocated, vm::page_writable, 0) && !f.memory->mapping_unchanged(*detached),
		"Native observer could not reinstall after serialized teardown");
	std::cout << "Real automatic VM delivery:map/unmap/protection hooks;precise unrelated watches;ABA;no-op/refusal;exclusive owner;foreign teardown waits for writer;detach/reinstall;no manual begin/end\n";
}

struct mapping_identity_probe final : vm::mapping_change_observer
{
	struct identity { std::uintptr_t owner; u64 offset; u32 bytes; bool readable; };
	struct event { u32 address, size; bool begin; std::array<identity, 3> identities; };
	std::shared_ptr<guarded_guest_memory> memory;
	std::array<event, 16> events{};
	unsigned count = 0;
	bool invalid = false;
	explicit mapping_identity_probe(std::shared_ptr<guarded_guest_memory> value) : memory(std::move(value))
	{
		const vm::writer_lock lock;
		require(vm::attach_mapping_observer(*this, lock), "Actual VM identity probe could not install");
	}
	~mapping_identity_probe()
	{
		const vm::writer_lock lock;
		if (!vm::detach_mapping_observer(*this, lock)) std::terminate();
	}
	void on_mapping_change_begin(u32 address, u32 size) noexcept override { memory->begin_mapping_change({address, size}); }
	void on_mapping_change_end(u32 address, u32 size) noexcept override { memory->end_mapping_change({address, size}); }
	void record(const vm::mapping_change_view& view, u32 address, u32 size, bool begin) noexcept
	{
		if (vm::host_mapping_guard::try_acquire() || memory->observe_mapping() ||
			view.lookup(address, 0, vm::page_allocated).memory || view.lookup(0xfffffffcu, 8, vm::page_allocated).memory)
			invalid = true;
		if (count == events.size()) { invalid = true; return; }
		auto& result = events[count++]; result.address = address; result.size = size; result.begin = begin;
		constexpr std::array addresses{fixture::base, fixture::alias, fixture::second};
		for (unsigned i = 0; i < addresses.size(); ++i)
		{
			const auto part = view.lookup(addresses[i] + 16, 8, vm::page_allocated);
			result.identities[i] = {reinterpret_cast<std::uintptr_t>(part.memory.get()), part.offset, part.size,
				bool(view.lookup(addresses[i] + 16, 8, vm::page_readable).memory)};
		}
	}
	void on_mapping_change_begin_locked(const vm::mapping_change_view& view, u32 address, u32 size) noexcept override
	{
		on_mapping_change_begin(address, size); record(view, address, size, true);
	}
	void on_mapping_change_end_locked(const vm::mapping_change_view& view, u32 address, u32 size) noexcept override
	{
		record(view, address, size, false); on_mapping_change_end(address, size);
	}
};

void actual_mapping_backing_identity()
{
	static_assert(!std::is_default_constructible_v<vm::mapping_change_view> && !std::is_copy_constructible_v<vm::mapping_change_view> &&
		!std::is_move_constructible_v<vm::mapping_change_view>);
	{
		fixture f; f.mappings.reset(); mapping_identity_probe sink(f.memory);
		auto shared = std::make_shared<utils::shm>(65536);
		const auto owner = reinterpret_cast<std::uintptr_t>(shared.get());
		require(f.block->falloc(fixture::base, 65536, &shared, vm::block_size_64k) &&
			f.block->falloc(fixture::alias, 65536, &shared, vm::block_size_64k) && f.block->falloc(fixture::second, 65536),
			"Actual VM identity setup allocations failed");
		require(!sink.invalid && sink.count == 6 && !sink.events[0].identities[0].owner &&
			sink.events[1].identities[0].owner == owner && sink.events[1].identities[0].offset == 16 && sink.events[1].identities[0].bytes == 8 &&
			!sink.events[2].identities[1].owner && sink.events[3].identities[1].owner == owner && sink.events[3].identities[1].offset == 16 &&
			sink.events[5].identities[2].owner && sink.events[5].identities[2].owner != owner,
			"Actual VM map-end callback preceded allocation-table identity publication");
		require(vm::page_protect(fixture::base, 65536, vm::page_allocated, 0, vm::page_readable | vm::page_writable), "Actual VM identity protection failed");
		require(sink.count == 8 && sink.events[6].identities[0].readable && !sink.events[7].identities[0].readable &&
			sink.events[7].identities[0].owner == owner && sink.events[7].identities[0].bytes == 8,
			"Actual VM callback lookup confused protected metadata with readable memory");
		require(f.block->dealloc(fixture::base) == 65536 && f.block->falloc(fixture::base, 65536), "Actual VM identity replacement failed");
		require(sink.count == 12 && sink.events[8].identities[0].owner == owner && !sink.events[9].identities[0].owner &&
			!sink.events[10].identities[0].owner && sink.events[11].identities[0].owner && sink.events[11].identities[0].owner != owner &&
			sink.events[11].identities[1].owner == owner,
			"Actual VM callbacks lost old backing,retained a removed VA or confused replacement with live alias");
		const auto replacement = sink.events[11].identities[0].owner;
		shared.reset(); // Whole-block teardown forbids external strong shm owners.
		require(vm::unmap(fixture::base, false, &f.block).second, "Actual VM whole-block removal failed");
		f.block.reset();
		require(!sink.invalid && sink.count == 14 && sink.events[12].begin && !sink.events[13].begin &&
			sink.events[12].address == fixture::base && sink.events[12].size == 4 * 65536 &&
			sink.events[13].address == fixture::base && sink.events[13].size == 4 * 65536 &&
			sink.events[12].identities[0].owner == replacement && sink.events[12].identities[1].owner == owner && sink.events[12].identities[2].owner &&
			std::all_of(sink.events[13].identities.begin(), sink.events[13].identities.end(), [](const auto& identity) { return !identity.owner; }) &&
			f.memory->observe_mapping().has_value(), "Actual VM whole-block callback did not enclose location/allocation retirement exactly once");
	}
	{
		fixture f(vm::block_size_64k | vm::preallocated | vm::bf0_0x2); f.mappings.reset(); mapping_identity_probe sink(f.memory);
		require(f.block->falloc(fixture::base, 65536) && f.block->falloc(fixture::alias, 65536), "Actual common-backing identity allocations failed");
		require(!sink.invalid && sink.count == 4 && sink.events[3].identities[0].owner &&
			sink.events[3].identities[0].owner == sink.events[3].identities[1].owner &&
			sink.events[3].identities[1].offset - sink.events[3].identities[0].offset == 65536,
			"Actual VM callback metadata merged distinct offsets in a common allocation");
	}
	{
		// Release the external block handle before the real VM close's strict
		// owner checks. Reinitialize VM only after callback/provider teardown.
		struct restore_vm
		{
			bool closed = false;
			~restore_vm() { if (closed) vm::init(); }
		} restore;
		fixture f; f.allocate_pair(); f.mappings.reset(); mapping_identity_probe sink(f.memory);
		f.block.reset(); restore.closed = true; vm::close();
		require(!sink.invalid && sink.count == 8 && sink.events[6].begin && !sink.events[7].begin &&
			sink.events[6].address == fixture::base && sink.events[7].address == fixture::base &&
			sink.events[6].identities[0].owner && sink.events[6].identities[0].owner == sink.events[6].identities[1].owner &&
			sink.events[6].identities[2].owner && std::all_of(sink.events[7].identities.begin(), sink.events[7].identities.end(),
				[](const auto& identity) { return !identity.owner; }) && f.memory->observe_mapping().has_value(),
			"Actual VM close callback lost old allocation identity before block invalidation or remained busy");
	}
	{
		struct restore_vm
		{
			bool loaded = false;
			~restore_vm() { if (loaded) { vm::close(); vm::init(); } }
		} restore;
		fixture f; f.allocate_pair();
		const std::vector<std::byte> first(65536, std::byte{0x55}), second(65536, std::byte{0x73});
		f.memory->write({fixture::base, first.size()}, first); f.memory->write({fixture::second, second.size()}, second);
		utils::serial archive;
		{ const vm::writer_lock lock; vm::save(archive); }
		archive.set_reading_state();
		f.mappings.reset(); mapping_identity_probe sink(f.memory); f.block.reset();
		{
			const vm::writer_lock lock; restore.loaded = true; vm::load(archive);
		}
		require(!sink.invalid && sink.count == 14 && sink.events[9].identities[0].owner &&
			sink.events[11].identities[0].owner == sink.events[11].identities[1].owner &&
			sink.events[13].identities[2].owner && sink.events[13].identities[2].owner != sink.events[13].identities[0].owner &&
			f.memory->read({fixture::base, first.size()}).bytes == first && f.memory->read({fixture::alias, first.size()}).bytes == first &&
			f.memory->read({fixture::second, second.size()}).bytes == second,
			"Actual VM restore callback could not inspect an allocation before its block entered g_locations or lost restored alias bytes");
	}
	std::cout << "Real VM callback identities:map-end after allocation publication;imported physical alias;private replacement;protected metadata/readability;whole-block begin before location removal,end after retirement;one native outer pair;common-backing offsets;actual VMclose before block invalidation;actual VMsave/load with constructor-local backing identity and restored alias bytes;borrowed nonconstructible view;no recursive VM lock\n";
}

void capture_and_exception()
{
	fixture f; f.allocate_pair();
	const auto caller = std::this_thread::get_id();
	f.access->execute([&](exclusive_guest_memory::view& view)
	{
		require(std::this_thread::get_id() == caller && guest_memory_epoch::current(*f.access) == &view, "Real VM capture changed thread or omitted epoch");
		f.access->execute([&](auto& nested) { require(&nested == &view, "Nested real VM capture did not reuse writer view"); });
		require(view.same_storage({fixture::base + 16, 8}, {fixture::alias + 16, 8}) &&
			!view.same_storage({fixture::base + 16, 8}, {fixture::second + 16, 8}), "Real VM alias metadata differs from backing allocation");
		require(!vm::host_mapping_guard::try_acquire(), "Real provider did not hold actual VM writer exclusion");
	});
	require(f.guard_calls == 1 && !guest_memory_epoch::active(), "Nested real provider recursively acquired host gate or leaked epoch");
	refuses([&] { f.access->capture_cpu([] [[noreturn]] { throw std::runtime_error("Intentional capture exception"); }); }, "Real CPU suspend swallowed operation exception");
	require(!guest_memory_epoch::active() && bool(vm::host_mapping_guard::try_acquire()), "Real exception path retained VM writer lock/epoch");
	std::cout << "Real VM capture:nested caller-thread epoch;actual shared/disjoint storage;writer exclusion;exception unwind\n";
}
void physical_aliases_and_atomic_stores()
{
	fixture f; f.allocate_pair();
	const auto layout = f.memory->watch_surface(plan_guest_surface(fixture::base + 128, 4, 2, 1, 16));
	const auto revision = f.memory->surface_revision(layout);
	const auto ticket = f.memory->capture_physical_surface(layout, revision);
	require(ticket && f.memory->physical_surface_overlaps_current(*ticket, {fixture::alias + 128, 4}) == true &&
		f.memory->physical_surface_overlaps_current(*ticket, {fixture::alias + 132, 12}) == false,
		"Real VM physical identity omitted alias or included padding");
	const auto before = vm::reservation_acquire(fixture::base + 128).load();
	const std::vector<std::byte> bytes(32, std::byte{0x53});
	const physical_surface_write write{ticket, bytes};
	const auto receipt = f.memory->compare_write_physical({&write, 1}, {&revision, 1});
	require(receipt && receipt->virtual_revisions.size() == 1 && receipt->virtual_revisions[0] == f.memory->surface_revision(layout), "Real VM physical commit omitted exact virtual receipt");
	require(f.memory->read({fixture::alias + 128, 4}).bytes == std::vector<std::byte>(4, std::byte{0x53}) &&
		f.memory->read({fixture::alias + 132, 12}).bytes == std::vector<std::byte>(12) && vm::reservation_acquire(fixture::base + 128).load() != before,
		"Real VM physical commit lost active bytes,padding or reservation notification");
	f.access->store_label(*f.memory, fixture::base + 256, 0x12345678);
	require(f.memory->read({fixture::alias + 256, 4}).bytes == std::vector<std::byte>{std::byte{0x12}, std::byte{0x34}, std::byte{0x56}, std::byte{0x78}}, "Actual atomic label store lost big-endian bytes");
	f.access->store_report(*f.memory, fixture::base + 272, 0x0123456789abcdef, 0x13579bdf);
	const auto report = f.memory->read({fixture::alias + 272, 16}).bytes;
	constexpr std::array expected{std::byte{0x01}, std::byte{0x23}, std::byte{0x45}, std::byte{0x67}, std::byte{0x89}, std::byte{0xab}, std::byte{0xcd}, std::byte{0xef},
		std::byte{0x13}, std::byte{0x57}, std::byte{0x9b}, std::byte{0xdf}, std::byte{}, std::byte{}, std::byte{}, std::byte{}};
	require(std::equal(report.begin(), report.end(), expected.begin()), "Actual aligned 16-byte atomic report store lost ABI/endian/padding");
	refuses([&] { f.access->store_label(*f.memory, fixture::base + 1, 0); }, "Misaligned real atomic label accepted");
	refuses([&] { f.access->store_report(*f.memory, fixture::base + 8, 0, 0); }, "Misaligned real atomic report accepted");
	std::cout << "Real VM publication:physical active alias/padding;atomic receipt;actual reservations;atomic BE32 label/BE128 report\n";
}
void exact_storage_queries()
{
	fixture f; f.allocate_pair();
	f.access->execute([&](auto& view)
	{
		require(view.equivalent_storage({fixture::base + 4077, 9000}, {fixture::alias + 4077, 9000}) &&
			!view.equivalent_storage({fixture::base + 4077, 9000}, {fixture::alias + 4078, 9000}), "Real equivalent-storage query lost exact byte offsets");
		const auto translated = view.rebase_storage({fixture::base, 65536}, {fixture::alias + 4077, 9000});
		require(translated && translated->address == fixture::base + 4077 && translated->size == 9000,
			"Real alias could not be rebased across page boundaries");
		require(!view.rebase_storage({fixture::base, 2 * 65536}, {fixture::alias + 4077, 9000}), "Duplicate real VM source mirrors produced ambiguous rebase");
		require(!view.same_storage({fixture::base, 65536}, {fixture::second, 65536}) &&
			!view.same_storage({fixture::base + 16, 8}, {fixture::alias + 24, 8}) &&
			!view.same_storage({fixture::base, 0}, {fixture::base, 8}), "Real alias overlap ignored owner,offset or empty interval");
	});
	require(vm::page_protect(fixture::base, 65536, vm::page_allocated, 0, vm::page_readable | vm::page_writable), "Actual VM protection failed");
	f.access->execute([&](auto& view)
	{
		require(view.same_storage({fixture::base + 16, 8}, {fixture::alias + 16, 8}), "Protected real alias disappeared from metadata-only overlap");
		require(!view.equivalent_storage({fixture::base + 16, 8}, {fixture::alias + 16, 8}) &&
			!view.rebase_storage({fixture::base, 65536}, {fixture::alias + 16, 8}), "Protected real source granted readable equivalence/rebase");
		refuses([&] { (void)view.map({fixture::base + 16, 8}, false); }, "Protected real alias granted byte access");
	});
	std::cout << "Real VM aliases:cross-page exact offsets;unique/ambiguous rebase;protected metadata;readable access refused\n";
}
void common_backing_offsets()
{
	fixture f(vm::block_size_64k | vm::preallocated | vm::bf0_0x2);
	require(f.block->falloc(fixture::base, 65536) && f.block->falloc(fixture::alias, 65536), "Actual common-backing allocations failed");
	f.access->execute([&](auto& view)
	{
		const auto first = view.retain_segments({fixture::base + 16, 8}), next = view.retain_segments({fixture::alias + 16, 8});
		require(physical_backing::same_allocation(first->segments().front(), next->segments().front()) &&
			next->segments().front().offset - first->segments().front().offset == 65536, "Actual preallocated common backing omitted distinct offsets");
		require(!view.same_storage({fixture::base + 16, 8}, {fixture::alias + 16, 8}) &&
			!view.equivalent_storage({fixture::base + 16, 8}, {fixture::alias + 16, 8}) &&
			!view.rebase_storage({fixture::base, 65536}, {fixture::alias + 16, 8}), "Same owner with disjoint common-backing offsets became an alias");
	});
	std::cout << "Real VM common backing:independent mapping identity shared;allocation offsets remain disjoint\n";
}
void remap_and_retained_publication()
{
	fixture f; f.allocate_pair();
	const auto layout = f.memory->watch_surface(plan_guest_surface(fixture::base + 128, 4, 2, 1, 16));
	const auto alias_watch = f.memory->watch_surface(plan_guest_surface(fixture::alias + 128, 4, 2, 1, 16));
	const auto revision = f.memory->surface_revision(layout), alias_revision = f.memory->surface_revision(alias_watch);
	const auto ticket = f.memory->capture_physical_surface(layout, revision);
	require(bool(ticket), "Actual old backing was not retained");
	require(f.block->dealloc(fixture::base) == 65536 && f.block->falloc(fixture::base, 65536), "Actual VM replacement allocation failed");
	const std::vector<std::byte> replacement(32, std::byte{0x19}), bytes(32, std::byte{0x64});
	f.memory->write({fixture::base + 128, 32}, replacement);
	require(vm::page_protect(fixture::alias, 65536, vm::page_allocated, 0, vm::page_readable | vm::page_writable), "Actual remaining alias protection failed");
	require(f.memory->physical_surface_overlaps_current(*ticket, {fixture::base + 128, 4}) == false &&
		f.memory->physical_surface_overlaps_current(*ticket, {fixture::alias + 128, 4}) == true, "Actual remap/protection reconciled physical identity incorrectly");
	const auto reservation = vm::reservation_acquire(fixture::alias + 128).load();
	const physical_surface_write write{ticket, bytes};
	const auto receipt = f.memory->compare_write_physical({&write, 1}, {&revision, 1});
	require(receipt && receipt->virtual_revisions.empty() && f.memory->read({fixture::base + 128, 32}).bytes == replacement &&
		vm::reservation_acquire(fixture::alias + 128).load() != reservation, "Actual physical write hit replacement VA,lost protected-alias reservation or returned invalid receipt");
	require(vm::page_protect(fixture::alias, 65536, vm::page_allocated, vm::page_readable | vm::page_writable, 0), "Actual alias permission restore failed");
	require(f.memory->read({fixture::alias + 128, 4}).bytes == std::vector<std::byte>(4, std::byte{0x64}) &&
		f.memory->read({fixture::alias + 132, 12}).bytes == std::vector<std::byte>(12) && f.memory->surface_revision(alias_watch) > alias_revision,
		"Actual old physical alias/padding/revision was not updated");
	const auto latest_revision = f.memory->surface_revision(alias_watch);
	const auto retained = f.memory->capture_physical_surface(alias_watch, latest_revision);
	std::shared_ptr<const physical_backing> independent;
	f.access->execute([&](auto& view) { independent = view.retain_segments({fixture::alias + 128, 32}); });
	require(bool(retained), "Actual remaining alias was not captured");
	require(f.block->dealloc(fixture::alias) == 65536, "Actual last alias unmap failed");
	const std::vector<std::byte> after_unmap(32, std::byte{0x78});
	const physical_surface_write detached{retained, after_unmap};
	require(f.memory->compare_write_physical({&detached, 1}).has_value(), "Independent actual kernel mapping died with last VM allocation handle");
	const auto* raw = independent->segments().front().data();
	require(std::all_of(raw, raw + 4, [](auto byte) { return byte == std::byte{0x78}; }) &&
		std::all_of(raw + 4, raw + 16, [](auto byte) { return byte == std::byte{}; }) &&
		std::all_of(raw + 16, raw + 20, [](auto byte) { return byte == std::byte{0x78}; }) &&
		f.memory->read({fixture::base + 128, 32}).bytes == replacement, "Detached actual physical publication lost bytes/padding or changed replacement mapping");
	refuses([&] { (void)f.memory->read_physical_surface(*retained); }, "Stale actual physical snapshot remained readable after commit");
	require(!f.memory->compare_write_physical({&detached, 1}), "Actual physical ticket allowed duplicate publication");
	std::cout << "Real VM recovery:actual remap/protection;old alias only;reservations;detached kernel owner survives last VM handle\n";
}
void actual_host_instruction()
{
	fixture f; f.allocate_pair();
	const auto watched = f.memory->watch_surface(guest_byte_coverage({fixture::alias + 384, 16}));
	const auto before = f.memory->surface_revision(watched);
	const std::vector<std::byte> source(16, std::byte{0x49});
	f.memory->write({fixture::second + 384, 16}, source);
	const auto destination = f.memory->prepare_host_access({fixture::base + 384, 16}, true);
	const auto origin = f.memory->prepare_host_access({fixture::second + 384, 16}, false);
	{
		auto ticket = f.gate->admit(); auto writer = ticket.activate();
		auto instruction = host_store_transaction::prepare(f.memory, destination, origin, []() -> std::shared_ptr<void>
		{
			auto mapping = vm::host_mapping_guard::try_acquire();
			if (!mapping) return {};
			return std::make_shared<vm::host_mapping_guard>(std::move(mapping));
		});
		require(instruction.start(), "Actual host instruction could not acquire nonwaiting VM mapping guard");
		bool excluded = false;
		std::thread mapper([&] { excluded = !vm::host_mapping_guard::try_acquire(); }); mapper.join();
		require(excluded, "Actual foreign host mapper entered a pending native instruction");
		std::memcpy(vm::g_sudo_addr + fixture::base + 384, vm::g_sudo_addr + fixture::second + 384, 16);
		vm::reservation_update(fixture::base + 384);
		instruction.commit();
		require(bool(vm::host_mapping_guard::try_acquire()), "Actual native instruction retained mapping guard after commit");
	}
	require(f.memory->surface_revision(watched) > before && !f.memory->pending_host_writes() &&
		f.memory->read({fixture::alias + 384, 16}).bytes == source, "Actual host instruction lost physical alias bytes or journal revisions");
	std::cout << "Real VM host instruction:actual memcpy;foreign nonwaiting mapper exclusion;guard release;alias journal drain\n";
}

// Uses the production manager, offload thread, preparation bridge, VM and
// journal. The owner dispatcher prepares CPU memory only; no surface cache or
// renderer exists here, so this does not establish GPU/PPU coherence.
struct actual_dma_fixture
{
	const bool previous_multithreaded = g_cfg.video.multithreaded_rsx.get();
	rsx::dma_manager manager;
	fixture f{vm::block_size_64k | vm::bf0_0x2, [&] { return manager.capture_native_writes(); }};
	render_tasks tasks{[] {}};
	const std::thread::id owner = std::this_thread::get_id();
	unsigned preparations = 0;
	bool fail_preparation = false;
	std::shared_ptr<const rsx::dma_manager::native_host_write_handler> handler;
	actual_dma_fixture()
	{
		f.allocate_pair(); tasks.bind_owner();
		handler = std::make_shared<const rsx::dma_manager::native_host_write_handler>(
			[&](void* destination, const void* source, u64 bytes)
			{
				const auto access = describe_rsx_host_access(destination, source, bytes);
				return stage_rsx_host_write(manager, f.memory, access, std::this_thread::get_id() == owner,
					[&](auto&& operation) { tasks.invoke(std::forward<decltype(operation)>(operation)); },
					[&](const rsx_host_access& ranges)
					{
						require(std::this_thread::get_id() == owner && !manager.is_current_thread(), "Actual DMA preparation escaped owner dispatcher");
						++preparations;
						if (fail_preparation) throw std::runtime_error("Intentional real DMA preparation failure");
						prepared_host_transfer prepared;
						f.access->execute([&](auto&)
						{
							require(!vm::host_mapping_guard::try_acquire(), "Actual DMA capture omitted VM writer exclusion");
							if (ranges.source) prepared.source = f.memory->prepare_host_access(*ranges.source, false);
							if (ranges.destination) prepared.destination = f.memory->prepare_host_access(*ranges.destination, true);
						});
						return prepared;
					});
			});
		require(manager.attach_native_host_write_handler(handler) && !manager.attach_native_host_write_handler(handler), "Actual DMA handler ownership was not exclusive");
		auto other = std::make_shared<const rsx::dma_manager::native_host_write_handler>([](void*, const void*, u64) -> std::optional<host_write_intent> { return {}; });
		require(!manager.detach_native_host_write_handler(other), "Actual DMA detached another callback owner");
	}
	~actual_dma_fixture()
	{
		tasks.close();
		try { manager.join(); } catch (...) {}
		try { (void)manager.detach_native_host_write_handler(handler); } catch (...) { std::terminate(); }
		(void)g_cfg.video.multithreaded_rsx.from_string(previous_multithreaded ? "true" : "false");
	}
	void queued()
	{
		require(g_cfg.video.multithreaded_rsx.from_string("true"), "Actual DMA queue config refused");
		manager.init();
	}
	void finish()
	{
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		for (;;)
		{
			tasks.drain();
			if (manager.sync()) return;
			require(std::chrono::steady_clock::now() < deadline, "Actual DMA completion exceeded test deadline");
			std::this_thread::yield();
		}
	}
};

void actual_dma_stores()
{
	actual_dma_fixture d;
	require(g_cfg.video.multithreaded_rsx.from_string("false"), "Actual DMA immediate config refused");
	std::vector<u8> source(4096);
	for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<u8>(i * 29 + 7);
	d.f.memory->write({fixture::second + 128, source.size()}, std::as_bytes(std::span(source)));
	const auto destination = d.f.memory->watch_surface(guest_byte_coverage({fixture::alias + 128, source.size()}));
	const auto origin = d.f.memory->watch_surface(guest_byte_coverage({fixture::second + 128, source.size()}));
	const auto unrelated = d.f.memory->watch_surface(guest_byte_coverage({fixture::base + 32768, 16}));
	const auto source_revision = d.f.memory->surface_revision(origin), unrelated_revision = d.f.memory->surface_revision(unrelated);
	unsigned operations = 0;
	for (bool queued : {false, true})
	{
		if (queued) d.queued();
		for (bool destination_sudo : {false, true}) for (bool source_sudo : {false, true})
		{
			const u32 bytes = queued ? 4096 : 32;
			const auto before = d.f.memory->surface_revision(destination);
			d.manager.copy((destination_sudo ? vm::g_sudo_addr : vm::g_base_addr) + fixture::base + 128,
				(source_sudo ? vm::g_sudo_addr : vm::g_base_addr) + fixture::second + 128, bytes);
			d.finish(); ++operations;
			require(d.f.memory->surface_revision(destination) > before &&
				d.f.memory->read({fixture::alias + 128, bytes}).bytes == std::vector<std::byte>(std::as_bytes(std::span(source)).begin(), std::as_bytes(std::span(source)).begin() + bytes),
				"Actual raw DMA lost base/sudo bytes or physical alias revision");
		}
		const u32 bytes = queued ? 4096 : 32;
		std::vector<u8> owned(bytes, 0x91);
		const auto before = d.f.memory->surface_revision(destination);
		d.manager.copy(vm::g_sudo_addr + fixture::base + 128, owned, bytes); d.finish(); ++operations;
		require(d.f.memory->surface_revision(destination) > before && d.f.memory->read({fixture::alias + 128, bytes}).bytes == std::vector<std::byte>(bytes, std::byte{0x91}),
			"Actual vector DMA lost owned source or alias publication");
		std::vector<u8> host(bytes);
		const auto read_revision = d.f.memory->surface_revision(destination);
		d.manager.copy(host.data(), vm::g_sudo_addr + fixture::second + 128, bytes); d.finish(); ++operations;
		require(std::equal(host.begin(), host.end(), source.begin()) && d.f.memory->surface_revision(destination) == read_revision,
			"Actual guest-source-only DMA changed destination watches or lost bytes");
		for (const auto primitive : {rsx::primitive_type::line_loop, rsx::primitive_type::triangle_fan, rsx::primitive_type::quads})
		{
			std::vector<u16> expected;
			if (primitive == rsx::primitive_type::line_loop) expected = {0, 1, 2, 3, 4, 5, 6, 7, 0};
			else if (primitive == rsx::primitive_type::triangle_fan) expected = {0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 7};
			else expected = {0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4};
			const auto revision = d.f.memory->surface_revision(destination);
			d.manager.emulate_as_indexed(vm::g_sudo_addr + fixture::base + 128, primitive, 8); d.finish(); ++operations;
			const auto bytes_expected = std::as_bytes(std::span(expected));
			require(d.f.memory->surface_revision(destination) > revision &&
				d.f.memory->read({fixture::alias + 128, bytes_expected.size()}).bytes == std::vector<std::byte>(bytes_expected.begin(), bytes_expected.end()),
				"Actual native index expansion lost index ordering/extent or alias publication");
		}
	}
	std::vector<u8> short_source(8);
	refuses([&] { d.manager.copy(vm::g_sudo_addr + fixture::base, short_source, 9); }, "Actual vector DMA read beyond supplied source");
	require(d.preparations == operations && operations == 18 && !d.f.memory->pending_host_writes() &&
		d.f.memory->surface_revision(origin) == source_revision && d.f.memory->surface_revision(unrelated) == unrelated_revision,
		"Actual DMA preparation count,source/unrelated precision or journal retirement differs");
	d.manager.join();
	refuses([&] { d.manager.copy(vm::g_sudo_addr + fixture::base, short_source, 8); }, "Closed actual DMA manager admitted a writer");
	std::cout << "Real DMA stores:18 actual immediate/queued raw/vector/source-only/index operations;base/sudo combinations;physical alias revisions;owner handoff;actual producer and VM exclusion;short source/closed admission refused\n";
}

void actual_dma_failure()
{
	actual_dma_fixture d; d.queued(); d.fail_preparation = true;
	const auto watch = d.f.memory->watch_surface(guest_byte_coverage({fixture::alias + 128, 8192}));
	const auto revision = d.f.memory->surface_revision(watch);
	std::vector<u8> first(4096, 0x31), second(4096, 0x72);
	d.manager.copy(vm::g_sudo_addr + fixture::base + 128, first, 4096);
	d.manager.copy(vm::g_sudo_addr + fixture::base + 4224, second, 4096);
	std::string failure;
	try { d.finish(); } catch (const std::exception& error) { failure = error.what(); }
	require(failure == "Intentional real DMA preparation failure", "Actual DMA did not deliver first owner preparation error");
	// sync may expose the error before retire has finished cancellation. This
	// capture waits for both the failed active and unstarted queued tickets.
	require(bool(d.manager.capture_native_writes()), "Actual failed DMA retained active or queued writer tickets");
	require(d.preparations == 1 && !d.f.memory->pending_host_writes() && d.f.memory->surface_revision(watch) == revision &&
		d.f.memory->read({fixture::alias + 128, 8192}).bytes == std::vector<std::byte>(8192),
		"Actual failed DMA executed/captured a later job or published bytes/journal");
	std::array<u8, 8> later_source{};
	std::string rejected;
	try { d.manager.copy(vm::g_sudo_addr + fixture::base, later_source.data(), later_source.size()); }
	catch (const std::exception& error) { rejected = error.what(); }
	require(rejected == failure, "Failed actual DMA manager admitted another instruction or replaced its failure");
	std::string joined;
	try { d.manager.join(); } catch (const std::exception& error) { joined = error.what(); }
	require(joined == failure, "Actual DMA join swallowed or replaced worker failure");
	std::cout << "Real DMA failure:owner exception reaches sync/join;unstarted queued job cancelled;all writer tickets drained;bytes/revisions/journal unchanged;future admission refused;nonthrowing destructor\n";
}

void actual_dma_shutdown()
{
	actual_dma_fixture d; d.queued();
	require(d.manager.detach_native_host_write_handler(d.handler), "Actual DMA owner could not detach before replacement");
	std::promise<void> entered, release, joining;
	const auto resumed = release.get_future().share();
	d.handler = std::make_shared<const rsx::dma_manager::native_host_write_handler>(
		[&](void*, const void*, u64) -> std::optional<host_write_intent>
		{
			entered.set_value(); resumed.wait(); return {};
		});
	require(d.manager.attach_native_host_write_handler(d.handler), "Actual DMA callback owner could not reinstall");
	std::array<u8, 8> source{1, 3, 5, 7, 9, 11, 13, 15}, destination{};
	auto writer = std::async(std::launch::async, [&] { d.manager.copy(destination.data(), source.data(), source.size()); });
	entered.get_future().wait();
	auto shutdown = std::async(std::launch::async, [&] { joining.set_value(); d.manager.join(); });
	joining.get_future().wait();
	const bool waited = shutdown.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout;
	release.set_value(); writer.get(); shutdown.get();
	require(waited && destination == source && bool(d.manager.capture_native_writes()),
		"Actual DMA shutdown escaped admitted writer or retained its ticket");
	refuses([&] { d.manager.copy(destination.data(), source.data(), source.size()); }, "Actual DMA shutdown reopened admission");
	std::cout << "Real DMA shutdown:join waits for admitted foreign immediate writer and retained callback;successful bytes;producer ticket drained;closed admission refused\n";
}

}
int main()
{
	// Initialize only VM address-space/allocation bookkeeping. Do not initialize
	// Emulator,Qt application,RSX renderer,PPU/SPU threads,a guest or a game.
	std::cout.setf(std::ios::unitbuf);
	try
	{
		g_headless = true; // Route shared fatal diagnostics to stderr, never a dialog.
		std::cout << "Headless real VM test beginning;no application initialization\n" << std::flush;
		require(!cpu_thread::get_current() && !g_fxo->try_get<rsx::thread>(), "Real VM test unexpectedly has an emulator CPU/renderer");
		const auto created = cpu_thread::g_threads_created.load();
		for (bool first_dependencies : {false, true})
		{
			rsx::draw_clause clause;
			clause.reset(rsx::primitive_type::triangles);
			clause.command = rsx::draw_command::inlined_array;
			clause.inline_vertex_array.push_back(0x12345678);
			unsigned emits = 0;
			require(metal4::drive_draw_clause(clause, static_cast<rsx::context*>(nullptr), [&](unsigned pass, unsigned flags)
			{
				require(!pass && flags == 4 && clause.is_rendering && clause.pass_count() == 1,
					"Actual inline clause lost its single owned pass");
				++emits;
			}, 4, metal4::clause_cleanup_owner::rsx_thread_end, first_dependencies) == 1 && emits == 1 && !clause.is_rendering,
				"Actual inline clause did not normalize before RSX cleanup");
			require(!metal4::owned_clause_requires_draw_completion(clause, true), "Actual inline clause retained a GPU completion boundary");
			clause.post_execute_cleanup(nullptr);
			clause.inline_vertex_array.clear();
			require(clause.empty(), "Actual inline cleanup retained its input buffer");
		}
		for (bool trailing : {false, true})
		{
			rsx::draw_clause clause;
			clause.reset(rsx::primitive_type::triangles);
			clause.command = rsx::draw_command::array;
			clause.append(0,3);
			if (trailing) clause.insert_command_barrier(rsx::vertex_base_modifier_barrier, 0);
			require(metal4::owned_clause_requires_draw_completion(clause, false) == trailing && !clause.is_rendering,
				"Actual trailing RSX dependency lost its GPU completion boundary");
			clause.begin();
			require(clause.get_range().count == 3 && !clause.next() && !clause.is_rendering,
				"Actual completion policy consumed or corrupted a clause range");
		}
		std::cout << "Actual RSX clause completion: two owned-inline first-dependency profiles and two ordinary/trailing-barrier profiles; one pass, cursor normalization and retained trailing wait; no renderer or guest CPU\n";
		vm::init();
		try { actual_mapping_hook_pairing(); automatic_mapping_delivery(); actual_mapping_backing_identity(); actual_page_protection(); actual_watch_protection(); actual_fault_receipts(); capture_and_exception(); physical_aliases_and_atomic_stores(); exact_storage_queries(); common_backing_offsets();
			remap_and_retained_publication(); actual_host_instruction(); actual_dma_stores(); actual_dma_failure(); actual_dma_shutdown(); test_real_vm_cache_watches(); test_real_vm_cache_faults(); test_arm64_memory_instructions(); }
		catch (...) { vm::close(); throw; }
		vm::close();
		test_rsx_overlay_icon();
		require(cpu_thread::g_threads_created.load() == created && !cpu_thread::get_current() && !g_fxo->try_get<rsx::thread>(),
			"Headless real VM test created an emulator CPU or renderer");
		std::cout << "Concrete Metal4 rsx_vm_access executed against actual VM;memory initialized/closed;no RPCS3 App/UI/renderer/guest CPU/game started\n";
		return 0;
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
