#include "stdafx.h"
#include "M4RSXVMAccess.h"
#include "M4RSXMappingObserver.h"
#include "M4HostWriterGate.h"
#include "M4SurfaceCache.h"
#include "M4GuestTexture.h"
#include "M4IOMappingEpoch.h"
#include "M4RSXCacheFault.h"
#include "M4RendererRuntimeRegistration.h"
#include "M4ARM64FaultChannel.h"
#include <signal.h>

#include <iostream>
#include <mach/mach.h>
#include <mach/mach_vm.h>

void test_arm64_fault_channel();

namespace
{
using namespace metal4;
void test_real_vm_arm_completion();
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int protection(std::uint32_t guest)
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
	require(result == KERN_SUCCESS && address <= requested && requested - address < size, "Cannot query actual VM cache protection");
	return info.protection;
}
struct vm_fixture
{
	static constexpr u32 base = 0x10000000, alias = base + 65536, input = base + 131072;
	std::shared_ptr<vm::block_t> block;
	std::shared_ptr<rsx_page_protection> service = std::make_shared<rsx_page_protection>(64);
	std::shared_ptr<host_writer_gate> gate = std::make_shared<host_writer_gate>();
	std::shared_ptr<rsx_vm_access> access;
	std::shared_ptr<guarded_guest_memory> memory;
	std::shared_ptr<rsx_mapping_observer> observer;
	std::function<void()> before_guard;
	vm_fixture()
	{
		block = vm::map(base, 4 * 65536, vm::block_size_64k | vm::bf0_0x2);
		require(bool(block), "Could not map actual VM cache fixture");
		access = std::make_shared<rsx_vm_access>([this]() -> std::shared_ptr<void>
		{
			if (before_guard) before_guard();
			auto exclusion = gate->acquire([] {}, std::chrono::seconds(1));
			if (!exclusion) return {};
			return std::make_shared<host_writer_gate::exclusion>(std::move(*exclusion));
		}, 32u << 20, 64, service);
		memory = std::make_shared<guarded_guest_memory>(access);
		observer = std::make_shared<rsx_mapping_observer>(memory, service);
		auto shared = std::make_shared<utils::shm>(65536);
		require(block->falloc(base, 65536, &shared, vm::block_size_64k) &&
			block->falloc(alias, 65536, &shared, vm::block_size_64k) && block->falloc(input, 65536), "Could not allocate actual VM cache storage");
	}
	~vm_fixture()
	{
		observer.reset(); memory.reset(); access.reset();
		(void)vm::unmap(base, false, &block); block.reset();
	}
};

void test_runtime_registration(context& device, vm_fixture& f, const cpu_capture_dispatch& capture,
	const std::shared_ptr<heap_budget>& budget)
{
	surface_cache surfaces(device, f.memory, 8u << 20, budget, capture);
	guest_texture_cache textures(device, f.memory, [](unsigned offset, unsigned, std::size_t) { return offset; },
		1u << 20, budget, 4, {capture, std::make_shared<io_mapping_epoch>()});
	const auto target = surfaces.acquire({vm_fixture::input, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	unsigned retired = 0;
	std::unique_ptr<rsx_cache_fault> fault;
	for (unsigned scenario = 0; scenario < 4; ++scenario)
	{
		surfaces.clear(target, 0xf0, 0xff123456 + scenario, 0, 255, {0, 0, 4, 4});
		std::string_view expected;
		if (scenario == 1) expected = "Partial runtime bind";
		if (scenario == 2) expected = "Runtime work failed";
		if (scenario == 3) expected = "Runtime shutdown failed after retirement";
		bool flushed = false;
		try
		{
			renderer_runtime_registration registration;
			registration.install([&]
			{
				if (fault)
				{
					const auto prepared = fault->prepare(vm_fixture::input, false);
					flushed = prepared && prepared->published_surfaces == 1 && !surfaces.status(target).dirty;
					fault.reset();
				}
				++retired;
				if (scenario == 3) throw std::runtime_error("Runtime shutdown failed after retirement");
			}, [&]
			{
				fault = std::make_unique<rsx_cache_fault>(device, f.access, f.memory, surfaces, textures);
				if (scenario == 1) throw std::runtime_error("Partial runtime bind");
			});
			if (scenario == 2) throw std::runtime_error("Runtime work failed");
			registration.close();
			registration.close();
			require(expected.empty(), "Expected runtime failure was swallowed");
		}
		catch (const std::runtime_error& error)
		{
			require(error.what() == expected, "Runtime cleanup masked the original failure");
		}
		require(flushed && !fault && retired == scenario + 1, "Runtime did not retire exactly once while its caches were live");
	}
	std::cout << "Metal4 runtime registration: four exit paths retire exactly once with live real-VM fault service; 4 native writebacks\n";
}
} // namespace

// Called only by the separate headless test after vm::init. Construct native
// Metal resources and the actual caches,never a GSRender,App,window or guest CPU.
void test_real_vm_cache_watches()
{
	@autoreleasepool
	{
		using namespace metal4;
		context device{1};
		vm_fixture f;
		const cpu_capture_dispatch capture = [&](const auto& operation) { f.access->capture_cpu(operation); };
		auto budget = std::make_shared<heap_budget>(device, 32u << 20);
		{
			surface_cache surfaces(device, f.memory, 8u << 20, budget, capture);
			auto target = surfaces.acquire({vm_fixture::base, {4, 4, 1, surface_color::argb8, false}, 16, {}});
			require(protection(vm_fixture::base) == VM_PROT_NONE && protection(vm_fixture::alias) == VM_PROT_NONE &&
				protection(vm_fixture::input) == (VM_PROT_READ | VM_PROT_WRITE), "Actual native Surfacecache acquisition did not protect real physical VM aliases");
			surfaces.clear(target, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 4});
			require(surfaces.status(target).dirty && vm::g_sudo_addr[vm_fixture::base] == 0 && protection(vm_fixture::base) == VM_PROT_NONE,
				"Actual native GPU output exposed stale guest bytes or lost its watch");
			auto fault = f.access->capture_fault(vm_fixture::alias);
			require(fault && !guest_memory_epoch::active(), "Actual native cache fault receipt retained VM exclusion across GPU work");
			const auto index_before = [&] { const vm::writer_lock lock; return f.service->statistics(lock); }();
			const auto base_reservation = vm::reservation_acquire(vm_fixture::base).load();
			const auto alias_reservation = vm::reservation_acquire(vm_fixture::alias).load();
			surfaces.flush(target);
			const auto index_after = [&] { const vm::writer_lock lock; return f.service->statistics(lock); }();
			require(index_after.refreshed_logical_pages == index_before.refreshed_logical_pages &&
				index_after.visited_index_nodes > index_before.visited_index_nodes &&
				index_after.visited_index_nodes - index_before.visited_index_nodes < 256 &&
				vm::reservation_acquire(vm_fixture::base).load() != base_reservation &&
				vm::reservation_acquire(vm_fixture::alias).load() != alias_reservation,
				"Actual GPU writeback bypassed the alias index or missed a reservation");
			f.access->capture_cpu([&]
			{
				constexpr std::array<u8, 4> expected{255, 0x12, 0x34, 0x56};
				for (unsigned i = 0; i < 64; ++i)
					require(vm::g_sudo_addr[vm_fixture::base + i] == expected[i % 4] && vm::g_sudo_addr[vm_fixture::alias + i] == expected[i % 4],
						"Actual native GPU flush lost real VM bytes or physical alias output");
			});
			std::cout << "Real VM native writeback used the physical index without a mapping refresh and notified both alias reservations; visited "
				<< index_after.visited_index_nodes - index_before.visited_index_nodes << " index nodes for 64 bytes\n";
			require(!surfaces.status(target).dirty && protection(vm_fixture::base) == VM_PROT_NONE,
				"Actual Surfacecache publication prematurely released input watch lifetime");
			{
				const auto watch = f.memory->watch_surface(plan_guest_surface(vm_fixture::base, 4, 4, 4, 16));
				const auto before = f.memory->surface_revision(watch);
				const auto ticket = f.memory->capture_physical_surface(watch, before);
				require(f.access->fault_current(*fault) && f.access->invalidate_fault(*f.memory, *fault) &&
					f.memory->surface_revision(watch) > before && !f.memory->physical_surface_current(*ticket) &&
					protection(vm_fixture::base) == VM_PROT_NONE, "Actual GPU fault publication failed dual revision retirement or reopened CPU pages");
				surfaces.prepare(target);
			}
			fault.reset();
			surfaces.erase(target);
			f.access->capture_cpu([] {});
			require(protection(vm_fixture::base) == VM_PROT_NONE, "External native Surface owner lost its VM protection after cache erase");
			target.reset();
		}
		// Reuse the completed one-slot context to release completion owners.
		device.begin(); device.cancel();
		f.access->capture_cpu([] {});
		require(protection(vm_fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && protection(vm_fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE),
			"Actual Surfacecache last-owner retirement did not restore real VM aliases");
		{
			surface_cache surfaces(device, f.memory, 8u << 20, budget, capture);
			auto target = surfaces.acquire({vm_fixture::base, {4, 4, 1, surface_color::argb8, false}, 16, {}});
			surfaces.clear(target, 0xf0, 0xffabcdef, 0, 255, {0, 0, 4, 4});
			auto before_remap = f.access->capture_fault(vm_fixture::alias);
			require(f.block->dealloc(vm_fixture::base) == 65536 && f.block->falloc(vm_fixture::base, 65536),
				"Actual GPU fault test could not replace its original VA");
			f.access->capture_cpu([&] { std::memset(vm::g_sudo_addr + vm_fixture::base, 0x73, 65536); });
			require(!f.access->invalidate_fault(*f.memory, *before_remap), "Actual GPU fault publication accepted a pre-remap receipt");
			{
				const auto watch = f.memory->watch_surface(plan_guest_surface(vm_fixture::base, 4, 4, 4, 16));
				auto fault = f.access->capture_fault(vm_fixture::alias);
				require(fault && fault->identity().segments().size() == 2 && !guest_memory_epoch::active(),
					"Actual GPU fault closure lost old/replacement active rows or held VM exclusion");
				surfaces.flush_physical(target);
				f.access->capture_cpu([&]
				{
					constexpr std::array<u8, 4> expected{255, 0xab, 0xcd, 0xef};
					for (unsigned i = 0; i < 64; ++i)
						require(vm::g_sudo_addr[vm_fixture::alias + i] == expected[i % 4] && vm::g_sudo_addr[vm_fixture::base + i] == 0x73,
							"Actual GPU fault flush wrote replacement VA or lost original physical alias bytes");
				});
				const auto revision = f.memory->surface_revision(watch);
				const auto ticket = f.memory->capture_physical_surface(watch, revision);
				require(f.access->invalidate_fault(*f.memory, *fault) && !f.memory->physical_surface_current(*ticket),
					"Actual remapped GPU fault closure failed replacement physical invalidation");
			}
			surfaces.erase(target); target.reset(); before_remap.reset();
		}
		device.begin(); device.cancel(); f.access->capture_cpu([] {});
		require(protection(vm_fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && protection(vm_fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE),
			"Actual remapped GPU fault receipt left orphan watches after all owners retired");
		{
			guest_texture_cache textures(device, f.memory, [](unsigned offset, unsigned, std::size_t) { return offset; },
				1u << 20, budget, 4, {capture, std::make_shared<io_mapping_epoch>()});
			f.access->capture_cpu([&]
			{
				for (unsigned i = 0; i < 64; ++i) vm::g_sudo_addr[vm_fixture::input + i] = std::array<u8, 4>{255, 0x57, 0x68, 0x79}[i % 4];
			});
			fragment_texture_registers words;
			words.offset = vm_fixture::input;
			words.control0 = 1u << 31;
			words.format = (1u << 16) | (0xa5u << 8) | 0x2a;
			words.image_rect = (4u << 16) | 4;
			words.control3 = (1u << 20) | 16;
			const auto uploaded = textures.resolve(words);
			require(uploaded.texture && protection(vm_fixture::input) == VM_PROT_NONE,
				"Actual native Texturecache upload did not protect its real VM snapshot");
			const auto before = textures.statistics().hits;
			const auto cached = textures.resolve(words);
			require(cached.texture == uploaded.texture && textures.statistics().hits > before && protection(vm_fixture::input) == VM_PROT_NONE,
				"Actual native Texturecache hit lost registered snapshot or installed a different resource");
			textures.clear();
		}
		device.begin(); device.cancel();
		f.access->capture_cpu([] {});
		require(protection(vm_fixture::input) == (VM_PROT_READ | VM_PROT_WRITE) && !f.service->failed(),
			"Actual Texturecache retirement retained orphan protection or failed VM metadata");
		test_runtime_registration(device, f, capture, budget);
		std::cout << "Real VM native cache watches:actual Metal4 Surfacecache acquisition/clear/physical flush;fixed ARGB bytes in actual VM aliases;owned fault receipt across GPU flush outside VM exclusion;virtual+physical invalidation;actual GPUdirty VA remap preserves old alias output and replacement bytes;dirty and external-owner watch lifetime;GPU completion retirement;actual native Texturecache upload/hit/clear;real suspended provider releases last registration;no App/UI/GSRender/guest CPU/game\n";
	}
}

// Automatic discovery/service proof,with actual native caches and actual VM.
// This prepares CPU instructions only;it never opens pages or runs a guest CPU.
void test_real_vm_cache_faults()
{
	@autoreleasepool
	{
		using namespace metal4;
		context device{1};
		auto budget = std::make_shared<heap_budget>(device, 32u << 20);
		const auto words = [](u32 address, unsigned height = 4)
		{
			fragment_texture_registers value;
			value.offset = address; value.control0 = 1u << 31;
			value.format = (1u << 16) | (0xa5u << 8) | 0x2a;
			value.image_rect = (4u << 16) | height; value.control3 = (1u << 20) | 16;
			return value;
		};
		{
			vm_fixture f;
			const cpu_capture_dispatch capture = [&](const auto& operation) { f.access->capture_cpu(operation); };
			const auto page = static_cast<u32>(f.service->page_size());
			const auto near = vm_fixture::base + 128, remote = vm_fixture::base + page, far = vm_fixture::input;
			{
				surface_cache surfaces(device, f.memory, 8u << 20, budget, capture);
				guest_texture_cache textures(device, f.memory, [](unsigned offset, unsigned, std::size_t) { return offset; },
					1u << 20, budget, 8, {capture, std::make_shared<io_mapping_epoch>()});
				rsx_cache_fault faults(device, f.access, f.memory, surfaces, textures);
				const auto acquire = [&](u32 address) { return surfaces.acquire({address, {4, 4, 1, surface_color::argb8, false}, 16, {}}); };
				auto first = acquire(vm_fixture::base), second = acquire(near), untouched = acquire(remote), clean = acquire(far);
				surfaces.clear(clean, 0xf0, 0xff425364, 0, 255, {0, 0, 4, 4}); surfaces.flush(clean);
				const auto clean_epoch = surfaces.status(clean).native_write_epoch;
				(void)textures.resolve(words(vm_fixture::base)); (void)textures.resolve(words(near));
				const auto far_image = textures.resolve(words(far)); (void)textures.resolve(words(remote));
				surfaces.clear(first, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				surfaces.clear(second, 0xf0, 0xff445566, 0, 255, {0, 0, 4, 4});
				surfaces.clear(untouched, 0xf0, 0xff778899, 0, 255, {0, 0, 4, 4});
				const auto before = textures.statistics();
				auto read = faults.prepare(vm_fixture::alias + page - 1, false);
				require(read && !read->writing && read->published_surfaces == 2 &&
					f.access->fault_current(*read->plan), "Automatic native read fault omitted a physical neighbour");
				require(read->invalidated_textures == 2, "Automatic native cache fault omitted matching immutable texture retirement");
				require(surfaces.status(first).initialized && surfaces.status(second).initialized && !surfaces.status(first).dirty &&
					!surfaces.status(second).dirty && surfaces.status(first).native_write_epoch && surfaces.status(second).native_write_epoch &&
					surfaces.status(untouched).dirty && surfaces.status(clean).initialized && surfaces.status(clean).native_write_epoch == clean_epoch,
					"Automatic native read fault lost native precision or reconciled disjoint authorities");
				require(textures.statistics().entries == 2 && textures.statistics().invalidations == before.invalidations + 2 &&
					textures.resolve(words(far)).texture == far_image.texture, "Native fault evicted a disjoint immutable texture");
				const auto padding = plan_guest_surface(vm_fixture::base + 64, 4, 2, 4, 256);
				require(f.memory->physical_identity_overlaps_current(read->plan->identity(), padding) == false &&
					f.memory->physical_identity_overlaps_current(read->plan->identity(), plan_guest_surface(vm_fixture::alias, 4, 4, 4, 16)) == true,
					"Fault physical metadata confused active bytes with pitched padding or lost a protected alias");
				const auto check = [&](u32 address, std::array<u8, 4> expected)
				{
					f.access->capture_cpu([&]
					{
						for (unsigned i = 0; i < 64; ++i) require(vm::g_sudo_addr[address + i] == expected[i % 4], "Automatic native fault publication produced wrong actual VM bytes");
					});
				};
				check(vm_fixture::alias, {255, 0x11, 0x22, 0x33}); check(vm_fixture::alias + 128, {255, 0x44, 0x55, 0x66});
				check(remote, {0, 0, 0, 0});
				require(protection(vm_fixture::base) == VM_PROT_NONE && protection(vm_fixture::alias) == VM_PROT_NONE,
					"Native cache read preparation opened CPU pages before instruction completion");
				// All owner/epoch mistakes must refuse before dispatching GPU work.
				bool refused = false;
				f.access->capture_cpu([&] { try { (void)faults.prepare(vm_fixture::base, true); } catch (const std::logic_error&) { refused = true; } });
				require(refused, "Native cache fault service entered GPU preparation from a held VM epoch");
				refused = false;
				std::thread foreign([&] { try { (void)faults.prepare(vm_fixture::base, true); } catch (const std::logic_error&) { refused = true; } }); foreign.join();
				require(refused, "Native cache fault service accepted a foreign render owner");
				try
				{
					auto wrong = std::make_shared<guarded_guest_memory>(f.access);
					rsx_cache_fault invalid(device, f.access, wrong, surfaces, textures); refused = false;
				}
				catch (const std::invalid_argument&) { refused = true; }
				require(refused && !faults.prepare(vm_fixture::base + 3 * 65536, true), "Native cache fault service accepted foreign bindings or an unwatched page");
				surfaces.clear(first, 0xf0, 0xff667788, 0, 255, {0, 0, 4, 4});
				surfaces.clear(second, 0xf0, 0xff99aabb, 0, 255, {0, 0, 4, 4});
				(void)textures.resolve(words(vm_fixture::base)); (void)textures.resolve(words(near));
				auto write = faults.prepare(vm_fixture::alias, true);
				require(write && write->writing && write->published_surfaces == 2 && write->invalidated_textures == 2 &&
					!surfaces.status(first).initialized && !surfaces.status(second).initialized && !surfaces.status(first).native_write_epoch &&
					!surfaces.status(second).native_write_epoch && surfaces.status(untouched).dirty && surfaces.status(clean).initialized,
					"Automatic native write preparation retained mutable mirrors or changed unrelated authorities");
				check(vm_fixture::alias, {255, 0x66, 0x77, 0x88}); check(vm_fixture::alias + 128, {255, 0x99, 0xaa, 0xbb});
				require(protection(vm_fixture::base) == VM_PROT_NONE && protection(vm_fixture::alias) == VM_PROT_NONE,
					"Native cache write preparation exposed a hardware retry window");
				read.reset(); write.reset();
				(void)textures.resolve(words(vm_fixture::base)); (void)textures.resolve(words(near));
				surfaces.clear(first, 0xf0, 0xffab1122, 0, 255, {0, 0, 4, 4});
				surfaces.clear(second, 0xf0, 0xffcd3344, 0, 255, {0, 0, 4, 4});
				require(f.block->dealloc(vm_fixture::base) == 65536 && f.block->falloc(vm_fixture::base, 65536), "Automatic fault test could not replace its dirty VA");
				f.access->capture_cpu([&] { std::memset(vm::g_sudo_addr + vm_fixture::base, 0x73, 65536); });
				// Read publication attempts to preserve native virtual receipts and
				// thereby arms replacement watches. This changes the original plan.
				bool stale = false;
				try { (void)faults.prepare(vm_fixture::alias, false); } catch (const cache_fault_conflict&) { stale = true; }
				require(stale, "Automatic native cache fault accepted a plan changed during GPU publication");
				auto retry = faults.prepare(vm_fixture::alias, false);
				require(retry && retry->published_surfaces == 0 && retry->plan->identity().segments().size() == 4 &&
					surfaces.status(untouched).dirty, "Automatic native cache fault retry lost replacement/old owners or unselected output");
				check(vm_fixture::alias, {255, 0xab, 0x11, 0x22}); check(vm_fixture::alias + 128, {255, 0xcd, 0x33, 0x44});
				check(vm_fixture::base, {0x73, 0x73, 0x73, 0x73});
				retry.reset();
				// A stale second authority must be found before publishing the first.
				surfaces.clear(first, 0xf0, 0xff998877, 0, 255, {0, 0, 4, 4});
				surfaces.clear(second, 0xf0, 0xffaa9876, 0, 255, {0, 0, 4, 4});
				const std::vector<std::byte> newer(64, std::byte{0x5e}); f.memory->write({near, 64}, newer);
				refused = false;
				try { (void)faults.prepare(vm_fixture::base, true); } catch (const surface_conflict_error&) { refused = true; }
				require(refused && surfaces.status(first).dirty && surfaces.status(second).dirty && protection(vm_fixture::base) == VM_PROT_NONE,
					"Native fault preflight lost stale output owners or reopened CPU pages");
				check(vm_fixture::base, {0x73, 0x73, 0x73, 0x73}); check(near, {0x5e, 0x5e, 0x5e, 0x5e});
			}
			device.begin(); device.cancel(); f.access->capture_cpu([] {});
			require(protection(vm_fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && protection(vm_fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE),
				"Automatic cache fault fixture retained orphan watches after final owners retired");
		}
		{
			vm_fixture f;
			const cpu_capture_dispatch capture = [&](const auto& operation) { f.access->capture_cpu(operation); };
			const auto page = static_cast<u32>(f.service->page_size());
			f.access->capture_cpu([&] { std::memset(vm::g_sudo_addr + vm_fixture::base, 0x4b, 65536); });
			{
				surface_cache surfaces(device, f.memory, 8u << 20, budget, capture);
				guest_texture_cache textures(device, f.memory, [](unsigned offset, unsigned, std::size_t) { return offset; },
					1u << 20, budget, 4, {capture, std::make_shared<io_mapping_epoch>()});
				rsx_cache_fault faults(device, f.access, f.memory, surfaces, textures);
				auto source = surfaces.acquire({vm_fixture::input, {4, 2, 1, surface_color::argb8, false}, 16, {}});
				auto target = surfaces.acquire({vm_fixture::base, {4, 2, 1, surface_color::argb8, false}, 2 * page, {}});
				surfaces.clear(source, 0xf0, 0xff2468ac, 0, 255, {0, 0, 4, 2});
				surfaces.defer_copy(source, {0, 0, 4, 2}, target, 0, 0);
				const auto padding_image = textures.resolve(words(vm_fixture::base + page, 1));
				auto prepared = faults.prepare(vm_fixture::alias, true);
				require(prepared && prepared->published_surfaces == 1 && prepared->invalidated_textures == 0 &&
					prepared->plan->identity().segments().size() == 2 && surfaces.status(source).dirty &&
					!surfaces.status(target).dirty && !surfaces.status(target).deferred_copies && !surfaces.status(target).initialized &&
					textures.resolve(words(vm_fixture::base + page, 1)).texture == padding_image.texture,
					"Automatic deferred fault omitted a remote active row or reconciled padding/unrelated source");
				f.access->capture_cpu([&]
				{
					constexpr std::array<u8, 4> expected{255, 0x24, 0x68, 0xac};
					for (unsigned row = 0; row < 2; ++row)
						for (unsigned i = 0; i < 16; ++i)
							require(vm::g_sudo_addr[vm_fixture::alias + row * 2 * page + i] == expected[i % 4], "Automatic deferred fault lost actual aliased GPU output");
					for (unsigned i = 0; i < 32; ++i) require(vm::g_sudo_addr[vm_fixture::base + page + i] == 0x4b, "Automatic fault wrote padding-only host-page bytes");
				});
			}
			device.begin(); device.cancel(); f.access->capture_cpu([] {});
			require(protection(vm_fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && protection(vm_fixture::alias) == (VM_PROT_READ | VM_PROT_WRITE),
				"Deferred automatic fault retirement retained orphan protection");
		}
		test_arm64_fault_channel();
		test_real_vm_arm_completion();
		std::cout << "Real VM cache fault service:automatic multiple physical Surface/Texture alias discovery;read preserves native precision and disjoint mirrors;write drops selected mirrors and both revisions;actual remap old-alias-only GPU output,replacement bytes preserved;changed plan refused then retried;all selected stale tickets preflight before first publication;deferred remote row closure,padding-only texture retained;foreign owner/bindings and held VM epoch refused;protected CPU pages until actual instruction completion;no App/UI/GSRender/guest CPU/game\n";
	}
}

namespace
{
struct vm_native_result
{
	std::array<std::byte, 32> vectors{};
	std::uint64_t first = 0, second = 0, base = 0;
};
vm_native_result execute_vm_instruction(unsigned form, std::uint8_t* address, const std::array<std::byte, 32>& payload)
{
	vm_native_result result;
#define VM_INSTRUCTION(text) asm volatile( \
	"mov x4,%[base]\n" "ldp q2,q3,[%[input]]\n" "mov x2,#0x1234\n" "mov x3,#0x5678\n" \
	text "\n" "stp q2,q3,[%[vectors]]\n" "mov %[first],x2\n" "mov %[second],x3\n" "mov %[after],x4\n" \
	: [first] "=&r" (result.first), [second] "=&r" (result.second), [after] "=&r" (result.base) \
	: [base] "r" (address), [input] "r" (payload.data()), [vectors] "r" (result.vectors.data()) \
	: "x2", "x3", "x4", "v2", "v3", "memory")
	switch (form)
	{
	case 0: VM_INSTRUCTION("ldp q2,q3,[x4],#32"); break;
	case 1: VM_INSTRUCTION("stp q2,q3,[x4],#32"); break;
	case 2: VM_INSTRUCTION("ldrsb x2,[x4],#1"); break;
	case 3: VM_INSTRUCTION("str xzr,[x4]"); break;
	case 4: VM_INSTRUCTION("ldp x2,x3,[x4],#16"); break;
	default: throw std::invalid_argument("Unknown actual VM ARM test form");
	}
#undef VM_INSTRUCTION
	return result;
}
struct vm_fault_exchange
{
	std::atomic<unsigned> phase{0};
	arm64_fault_channel::client* client = nullptr;
	vm_native_result actual;
};
thread_local vm_fault_exchange* current_vm_fault = nullptr;
void controlled_vm_fault(int, siginfo_t* info, void* opaque)
{
	auto* exchange = current_vm_fault;
	auto* context = static_cast<ucontext_t*>(opaque);
	if (!exchange || !exchange->client || !info || !context) ::_exit(90);
	if (exchange->client->handle(*context, reinterpret_cast<std::uint64_t>(info->si_addr)) != arm64_fault_channel::outcome::completed) ::_exit(93);
}
class controlled_vm_signals
{
	struct sigaction m_bus{}, m_segv{};
  public:
	controlled_vm_signals()
	{
		struct sigaction action{};
		action.sa_sigaction = controlled_vm_fault; action.sa_flags = SA_SIGINFO;
		(void)sigemptyset(&action.sa_mask);
		require(::sigaction(SIGBUS, &action, &m_bus) == 0, "Cannot install actual VM ARM test bus handler");
		if (::sigaction(SIGSEGV, &action, &m_segv) != 0)
		{
			(void)::sigaction(SIGBUS, &m_bus, nullptr);
			throw std::runtime_error("Cannot install actual VM ARM test segv handler");
		}
	}
	~controlled_vm_signals()
	{
		if (::sigaction(SIGBUS, &m_bus, nullptr) != 0 || ::sigaction(SIGSEGV, &m_segv, nullptr) != 0) ::_exit(94);
	}
};
rsx_cache_fault::completion run_vm_fault(rsx_cache_fault& faults, unsigned form, u32 address,
	const std::array<std::byte, 32>& payload, vm_native_result& actual)
{
	controlled_vm_signals signals;
	arm64_fault_channel channel{reinterpret_cast<std::uint64_t>(vm::g_base_addr), 0xe0000000ull, 1};
	vm_fault_exchange exchange;
	std::thread producer([&]
	{
		auto client = channel.attach();
		exchange.client = &client;
		current_vm_fault = &exchange;
		exchange.actual = execute_vm_instruction(form, vm::g_base_addr + address, payload);
		current_vm_fault = nullptr;
		exchange.phase.store(3, std::memory_order_release);
	});
	try
	{
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
		while (!channel.has_pending() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
		require(channel.has_pending(), "Actual VM ARM instruction did not publish its owned fault slot");
		std::optional<rsx_cache_fault::completion> completed;
		const auto drained = channel.drain([&](const auto& input) -> std::optional<arm64_memory_state>
		{
			// The original thread is parked in immutable linked test code.
			std::weak_ptr<const void> held;
			completed = faults.complete_parked(input, [&](std::uint64_t)
			{
				auto lease = std::make_shared<const std::byte>();
				held = lease;
				return lease;
			});
			require(completed && held.expired(), "Owner rejected the parked instruction or retained its code lease");
			require(completed->reply.pc == input.before.pc + 4, "Actual VM ARM reply did not complete the exact instruction");
			return completed->reply;
		});
		require(drained == 1 && completed, "Actual VM ARM owner omitted its queued instruction");
		while (exchange.phase.load(std::memory_order_acquire) != 3 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
		require(exchange.phase.load(std::memory_order_acquire) == 3, "Actual VM ARM instruction did not resume from its owned reply");
		producer.join(); actual = exchange.actual;
		return *completed;
	}
	catch (...)
	{
		channel.close(); producer.join();
		throw;
	}
}

void test_real_vm_arm_completion()
{
	context device{1};
	vm_fixture f;
	const cpu_capture_dispatch capture = [&](const auto& operation) { f.access->capture_cpu(operation); };
	auto budget = std::make_shared<heap_budget>(device, 32u << 20);
	const auto page = static_cast<u32>(f.service->page_size());
	const auto first_address = vm_fixture::base + page - 16, second_address = vm_fixture::base + page;
	const auto fault_address = vm_fixture::alias + page - 16;
	{
		surface_cache surfaces(device, f.memory, 8u << 20, budget, capture);
		guest_texture_cache textures(device, f.memory, [](unsigned offset, unsigned, std::size_t) { return offset; },
			1u << 20, budget, 8, {capture, std::make_shared<io_mapping_epoch>()});
		rsx_cache_fault faults(device, f.access, f.memory, surfaces, textures);
		auto first = surfaces.acquire({first_address, {4, 1, 1, surface_color::argb8, false}, 16, {}});
		auto second = surfaces.acquire({second_address, {4, 1, 1, surface_color::argb8, false}, 16, {}});
		auto untouched = surfaces.acquire({vm_fixture::base + 2 * page, {4, 1, 1, surface_color::argb8, false}, 16, {}});
		const auto dirty = [&]
		{
			surfaces.clear(first, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 1});
			surfaces.clear(second, 0xf0, 0xff445566, 0, 255, {0, 0, 4, 1});
		};
		dirty(); surfaces.clear(untouched, 0xf0, 0xff778899, 0, 255, {0, 0, 4, 1});
		std::array<std::byte, 32> expected, payload;
		for (unsigned i = 0; i < 32; ++i)
		{
			expected[i] = std::byte{i < 16 ? std::array<u8, 4>{255, 0x11, 0x22, 0x33}[i % 4] : std::array<u8, 4>{255, 0x44, 0x55, 0x66}[i % 4]};
			payload[i] = std::byte{static_cast<u8>(0x80 + i)};
		}
		vm_native_result actual;
		unsigned refused_captures = 0;
		f.before_guard = [&] { ++refused_captures; };
		const arm64_fault_channel::input unleased{{}, reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address), false};
		require(!faults.complete_parked(unleased, [](std::uint64_t) -> std::shared_ptr<const void> { return {}; }) &&
			!faults.complete_parked(unleased, [](std::uint64_t) { return std::make_shared<const std::byte>(); }) &&
			refused_captures == 0, "Unleased or invalid executable code reached VM capture");
		f.before_guard = {};
		const auto read = run_vm_fault(faults, 0, fault_address, payload, actual);
		require(read.host_pages == 2 && read.watched_pages == 2 && read.published_surfaces == 2 && actual.vectors == expected &&
			actual.base == reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address + 32) && surfaces.status(untouched).dirty,
			"Actual VM ARM pair load omitted a host page,GPU output,writeback or disjoint authority");
		{
			const auto watch = f.memory->watch_surface(plan_guest_surface(first_address, 4, 2, 4, 16));
			const auto revision = f.memory->surface_revision(watch);
			const auto physical = f.memory->capture_physical_surface(watch, revision);
			// No GPU publication in this store:revision/reservation assertions
			// must detect CPU completion itself rather than an earlier GPU flush.
			const auto reservation_first = vm::reservation_acquire(first_address).load();
			const auto reservation_second = vm::reservation_acquire(second_address).load();
			const auto alias_first = vm::reservation_acquire(fault_address).load();
			const auto alias_second = vm::reservation_acquire(fault_address + 16).load();
			const auto write = run_vm_fault(faults, 1, fault_address, payload, actual);
			require(write.host_pages == 2 && write.watched_pages == 2 && write.published_surfaces == 0 &&
				actual.base == reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address + 32) &&
				f.memory->surface_revision(watch) > revision && !f.memory->physical_surface_current(*physical) &&
				!surfaces.status(first).initialized && !surfaces.status(second).initialized && surfaces.status(untouched).dirty,
				"Actual VM ARM pair store lost its precise reply or virtual/physical cache retirement");
			require(vm::reservation_acquire(first_address).load() != reservation_first && vm::reservation_acquire(second_address).load() != reservation_second &&
				vm::reservation_acquire(fault_address).load() != alias_first && vm::reservation_acquire(fault_address + 16).load() != alias_second,
				"Actual VM ARM store omitted a physical-alias reservation notification");
			f.access->capture_cpu([&]
			{
				require(std::memcmp(vm::g_sudo_addr + first_address, payload.data(), payload.size()) == 0 &&
					std::memcmp(vm::g_sudo_addr + fault_address, payload.data(), payload.size()) == 0, "Actual VM ARM store lost canonical or aliased bytes");
			});
		}
		dirty();
		const auto dirty_write = run_vm_fault(faults, 1, fault_address, payload, actual);
		require(dirty_write.published_surfaces == 2 && actual.base == reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address + 32),
			"Actual VM ARM dirty pair store omitted GPU publication or writeback");
		dirty();
		(void)run_vm_fault(faults, 2, fault_address, payload, actual);
		require(actual.first == ~std::uint64_t{0} && actual.base == reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address + 1),
			"Actual VM ARM scalar load lost sign extension or post-index");
		(void)run_vm_fault(faults, 3, fault_address, payload, actual);
		f.access->capture_cpu([&] { for (unsigned i = 0; i < 8; ++i) require(vm::g_sudo_addr[fault_address + i] == 0, "Actual VM ARM XZR store did not write zero"); });
		dirty();
		(void)run_vm_fault(faults, 4, fault_address + 8, payload, actual);
		std::uint64_t low, high; std::memcpy(&low, expected.data() + 8, 8); std::memcpy(&high, expected.data() + 16, 8);
		require(actual.first == low && actual.second == high, "Actual VM ARM scalar pair load lost one side of a host-page crossing");
		require(protection(first_address) == VM_PROT_NONE && protection(second_address) == VM_PROT_NONE &&
			protection(fault_address) == VM_PROT_NONE && protection(fault_address + 16) == VM_PROT_NONE,
			"Actual VM ARM completion opened protected CPU pages");
		// Synthetic requests exercise refusal without allowing a failing native
		// instruction to resume. No cache/GPU publication may precede permissions.
		arm64_memory_state state; state.pc = 0x1000; state.x[4] = reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address);
		const auto request = arm64_memory_request::decode(0xad000c82u, state); // stp q2,q3,[x4]
		require(request && request->size() == 32 && request->writing(), "Actual VM ARM refusal fixture is not the intended pair store");
		unsigned refusal_checks = 0;
		const auto refused = [&](const auto& operation)
		{
			++refusal_checks;
			bool failed = false; try { operation(); } catch (const std::exception&) { failed = true; }
			if (!failed) throw std::runtime_error("Actual VM ARM completion accepted invalid access/owner case " + std::to_string(refusal_checks));
		};
		dirty();
		refused([&] { (void)faults.complete(*request, state.x[4], false); });
		refused([&] { (void)faults.complete(*request, state.x[4] - 1, true); });
		f.access->capture_cpu([&] { refused([&] { (void)faults.complete(*request, state.x[4], true); }); });
		bool foreign_refused = false;
		std::thread foreign([&] { try { (void)faults.complete(*request, state.x[4], true); } catch (const std::logic_error&) { foreign_refused = true; } }); foreign.join();
		require(foreign_refused, "Actual VM ARM completion accepted a foreign render owner");
		require(vm::page_protect(vm_fixture::alias + page, 4096, vm::page_allocated, 0, vm::page_writable), "Cannot restrict second logical VM instruction page");
		refused([&] { (void)faults.complete(*request, state.x[4], true); });
		require(surfaces.status(first).dirty && surfaces.status(second).dirty, "Invalid full-instruction permission caused partial GPU publication");
		require(vm::page_protect(vm_fixture::alias + page, 4096, vm::page_allocated, vm::page_writable, 0), "Cannot restore second logical VM instruction page");
		// Observe an exact permissions ABA after both valid GPU publications.
		// Matching final flags/owners alone must not authorize the old request.
		dirty();
		bool changed = false;
		f.before_guard = [&]
		{
			if (changed || surfaces.status(first).dirty || surfaces.status(second).dirty) return;
			changed = true;
			require(vm::page_protect(vm_fixture::alias, 65536, vm::page_allocated, 0, vm::page_writable) &&
				vm::page_protect(vm_fixture::alias, 65536, vm::page_allocated, vm::page_writable, 0), "Cannot perform instruction mapping ABA");
		};
		refused([&] { (void)faults.complete(*request, state.x[4], true); });
		f.before_guard = {};
		require(changed && std::memcmp(vm::g_sudo_addr + first_address, expected.data(), expected.size()) == 0,
			"Stale instruction ABA wrote CPU payload or lost already valid GPU publications");
		bool reentry_refused = false;
		f.before_guard = [&]
		{
			if (reentry_refused) return;
			try { (void)faults.complete(*request, state.x[4], true); } catch (const std::logic_error&) { reentry_refused = true; }
		};
		(void)faults.complete(*request, state.x[4], true);
		f.before_guard = {};
		require(reentry_refused, "Native cache fault admitted another task during instruction completion");
		// A valid logical store can fault with NO remaining watch:the coarse
		// host-page baseline itself may be NO. Do not infer permissions from
		// watch presence,and do not demand read permission for a store.
		const auto private_address = vm_fixture::input;
		require(vm::page_protect(private_address, 4096, vm::page_allocated, 0, vm::page_readable), "Cannot create write-only logical VM page");
		{
			const vm::writer_lock lock;
			const auto token = f.service->protect({private_address, 1}, rsx_page_protection::access::trap_reads_and_writes, lock);
			f.service->release(token, lock);
		}
		require(protection(private_address) == VM_PROT_NONE && !f.access->capture_fault(private_address), "Write-only coarse VM fixture retained an active watch");
		const auto no_watch_write = run_vm_fault(faults, 1, private_address, payload, actual);
		require(no_watch_write.host_pages == 1 && !no_watch_write.watched_pages && !no_watch_write.published_surfaces &&
			std::memcmp(vm::g_sudo_addr + private_address, payload.data(), payload.size()) == 0 && protection(private_address) == VM_PROT_NONE,
			"Valid write-only instruction without a watch was refused,miswritten or unprotected");
		state.x[4] = reinterpret_cast<std::uint64_t>(vm::g_base_addr + private_address);
		const auto load = arm64_memory_request::decode(0xad400c82u, state); // ldp q2,q3,[x4]
		require(load && !load->writing(), "Write-only VM refusal fixture is not a load");
		refused([&] { (void)faults.complete(*load, state.x[4], false); });
		require(vm::page_protect(private_address, 4096, vm::page_allocated, vm::page_readable, 0), "Cannot restore logical VM read permission");
		if (page > 4096)
		{
			// page_protect expands size to the allocation granularity while
			// preserving the requested4KiB-aligned start. Its shifted64KiB
			// interval therefore needs an allocated trailing neighbour too.
			require(f.block->falloc(vm_fixture::base + 3 * 65536, 65536), "Cannot allocate shifted protection tail");
			require(vm::page_protect(private_address + 4096, 4096, vm::page_allocated, 0, vm::page_readable | vm::page_writable), "Cannot restrict mixed host-page neighbour");
			{
				const vm::writer_lock lock;
				const auto token = f.service->protect({private_address, 1}, rsx_page_protection::access::trap_reads_and_writes, lock);
				f.service->release(token, lock);
			}
			require(protection(private_address) == VM_PROT_NONE && !f.access->capture_fault(private_address), "Mixed neighbour fixture is not unwatched and protected");
			const auto no_watch_read = run_vm_fault(faults, 0, private_address, payload, actual);
			require(!no_watch_read.watched_pages && !no_watch_read.published_surfaces && actual.vectors == payload && protection(private_address) == VM_PROT_NONE,
				"Valid logical instruction with inaccessible neighbour was refused or opened the host page");
			require(vm::page_protect(private_address + 4096, 4096, vm::page_allocated, vm::page_readable | vm::page_writable, 0), "Cannot restore mixed host-page neighbour");
		}
		dirty();
		// Invalid second authority on another host page must refuse before first
		// publication,so per-page GPU preparation would be observably incorrect.
		state.x[4] = reinterpret_cast<std::uint64_t>(vm::g_base_addr + fault_address);
		f.memory->write({second_address, 1}, std::array{std::byte{0x5e}});
		refused([&] { (void)faults.complete(*request, state.x[4], true); });
		require(surfaces.status(first).dirty && surfaces.status(second).dirty && vm::g_sudo_addr[second_address] == 0x5e,
			"Whole-instruction preflight published before discovering another host page's stale output");
	}
	device.begin(); device.cancel(); f.access->capture_cpu([] {});
	require(protection(vm_fixture::base) == (VM_PROT_READ | VM_PROT_WRITE) && !f.service->failed(), "Actual VM ARM test leaked owned protection");
	std::cout << "Real VM ARM completion:" << (page > 4096 ? 8 : 7) << " actual protected native instructions via shared parked-instruction service; unleased or invalid code refused before VM capture; parked host thread,actual Metal4 dirty outputs and real suspended VM;two-host-page combined authority preflight;canonical/alias bytes,scalar/SIMD/pairs/sign/XZR/writeback/PC replies;virtual+physical revisions and all physical-alias reservations;full logical permissions/foreign owner/held epoch refused;all CPU pages remain NO during completion;registered fixed fault-channel slots;handler only copies/spins;owner kernel instruction fetch;test-only signal installation,no App/guest CPU\n";
}
} // namespace
