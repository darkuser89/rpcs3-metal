#include "M4PhysicalPageIndex.h"
#include "M4PageProtectionPlan.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <random>
#include <utility>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

namespace
{
std::atomic<bool> count_allocations{false};
std::atomic<std::size_t> allocations{0};
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename Operation> void refuses(Operation&& operation)
{
	bool failed = false; try { operation(); } catch (const std::invalid_argument&) { failed = true; }
	require(failed, "Invalid physical index input accepted");
}
void mutations_and_oracle()
{
	constexpr unsigned pages = 1024, owner_count = 24;
	physical_page_index index(pages);
	std::array<std::shared_ptr<void>, owner_count> owners;
	for (auto& owner : owners) owner = std::make_shared<int>(7);
	struct mapping { unsigned owner = owner_count; std::uint64_t offset = 0; };
	std::array<mapping, pages> oracle{};
	std::mt19937 random(0x4d340045);
	unsigned queries = 0;
	for (unsigned step = 0; step < 24000; ++step)
	{
		const unsigned page = random() % pages, owner = random() % (owner_count + 1);
		const std::uint64_t offset = (random() % 512) * 4096ull;
		require(index.replace(page, owner == owner_count ? std::shared_ptr<void>{} : owners[owner], offset), "Valid replacement failed");
		oracle[page] = {owner, offset};
		if (step % 13) continue;
		const auto selected = random() % owner_count;
		const std::uint64_t first = random() % (512 * 4096), size = random() % 20000;
		std::vector<std::pair<std::uint32_t, std::uint64_t>> actual, expected;
		const auto visits = index.aliases(owners[selected], first, size, [&](auto address, auto count) { actual.emplace_back(address, count); });
		for (unsigned p = 0; p < pages; ++p)
		{
			const auto& value = oracle[p];
			if (value.owner != selected) continue;
			const auto begin = std::max(first, value.offset), end = std::min(first + size, value.offset + 4096);
			if (begin < end) expected.emplace_back(p * 4096u + begin - value.offset, end - begin);
		}
		std::sort(actual.begin(), actual.end()); std::sort(expected.begin(), expected.end());
		require(actual == expected, "Physical index differs from independent exhaustive page oracle");
		require(visits <= 4 * index.tree_height() + 2 * expected.size(), "Physical query walked unrelated page subtrees");
		++queries;
	}
	for (auto& owner : owners) require(owner.use_count() == 1, "Physical index retained a strong allocation owner");
	std::cout << "Physical page index:24000 remap/unmap/offset updates," << queries << " exact exhaustive-oracle queries across24owners\n";
}
void boundaries_and_lifetime()
{
	refuses([] { physical_page_index index(0); });
	refuses([] { physical_page_index index((1u << 20) + 1); });
	physical_page_index index(8);
	int reused_pointer = 0;
	auto old = std::shared_ptr<void>(&reused_pointer, [](void*) {});
	auto replacement = std::shared_ptr<void>(&reused_pointer, [](void*) {});
	require(index.replace(0, old, 0) && index.replace(1, replacement, 0) && index.replace(2, old, 4096), "Owner fixture failed");
	std::vector<std::pair<std::uint32_t, std::uint64_t>> hits;
	index.aliases(old, 4095, 2, [&](auto a, auto n) { hits.emplace_back(a, n); });
	require(hits == decltype(hits){{4095, 1}, {8192, 1}}, "Offset boundaries or reused source pointers confused");
	require(!index.replace(8, old, 0) && !index.replace(0, old, 1) &&
		!index.replace(0, old, std::numeric_limits<std::uint64_t>::max() & ~4095ull) && index.size() == 3,
		"Invalid index mutation changed its current state");
	refuses([&] { index.aliases(old, std::numeric_limits<std::uint64_t>::max(), 1, [](auto, auto) {}); });
	unsigned count = 0;
	index.aliases(old, 0, 0, [&](auto, auto) { ++count; });
	require(count == 0 && index.replace(0, {}, 0), "Zero query or remove failed");
	std::weak_ptr<void> expired = old; old.reset();
	require(expired.expired(), "Index kept source alive");
	index.aliases(expired, 0, 8192, [&](auto, auto) { ++count; });
	require(count == 0 && index.replace(2, replacement, 0), "Expired source identity was revived");
	for (unsigned n = 0; n < 8; ++n) require(index.replace(n, {}, 0), "Index removal failed");
	require(index.size() == 0 && index.tree_height() == 0, "All-node erase left tree state");
}
void full_domain_and_no_allocations()
{
	physical_page_index index;
	auto owner = std::make_shared<int>(1);
	constexpr unsigned pages = 1u << 20;
	for (unsigned page = 0; page < pages; ++page)
		require(index.replace(page, owner, std::uint64_t(page) * 4096), "Full-domain insertion failed");
	require(index.size() == pages && index.tree_height() <= 21 && owner.use_count() == 1, "Full-domain AVL balance or source lifetime failed");
	std::uint64_t address = 0, bytes = 0;
	unsigned matches = 0;
	const auto visits = index.aliases(owner, 0xffffffffull, 1, [&](auto a, auto n) { address = a; bytes = n; ++matches; });
	require(matches == 1 && address == 0xffffffffull && bytes == 1 && visits <= 21, "4GiB last-byte query scanned or wrapped");
	bool valid = true;
	allocations = 0; count_allocations = true;
	for (unsigned n = 0; n < 10000; ++n)
	{
		const auto page = (n * 7919u) & (pages - 1);
		valid &= index.replace(page, {}, 0);
		valid &= index.replace(page, owner, std::uint64_t(page) * 4096);
		index.aliases(owner, std::uint64_t(page) * 4096 + 17, 7, [](auto, auto) {});
	}
	count_allocations = false;
	require(valid && allocations == 0, "Mapping callback/query allocated heap metadata");
	for (unsigned page = 0; page < pages; ++page)
		require(index.replace((page * 7919u) & (pages - 1), {}, 0), "Permutation deletion failed");
	require(index.size() == 0 && index.tree_height() == 0, "Full-domain permutation deletion corrupted AVL");
	std::cout << "Physical page index:1048576 pages/full4GiB;last-byte query " << visits << " nodes;10000 mutation/query pairs zero heap allocations;permuted full retirement;metadata " << index.metadata_bytes() << " bytes\n";
}
void native_page_unions()
{
	for (const unsigned size : {4096u, 16384u, 65536u})
	{
		page_protection_plan plan(size);
		std::vector<page_protection_plan::access> permissions(4 * size / 4096);
		struct run { std::uint64_t address, size; int permissions; };
		std::vector<run> runs;
		const auto lookup = [&](auto address) { return permissions.at(address / 4096); };
		const auto apply = [&](auto address, auto bytes, auto flags) { runs.push_back({address, bytes, flags}); };
		for (auto& p : permissions) { p.overlay = PROT_READ; p.selected = true; }
		for (unsigned a = 0; a < 2 * size; a += 4096) { plan.mark(a); plan.mark(a + 17); }
		require(plan.pending_pages() == 2, "Dirty native pages were not deduplicated");
		plan.synchronize_dirty(lookup, apply);
		require(runs.size() == 1 && runs[0].address == 0 && runs[0].size == 2 * size && runs[0].permissions == PROT_READ &&
			plan.pending_pages() == 0 && plan.owned(0) && plan.owned(size) && !plan.owned(2 * size), "Targeted protection widened or failed to coalesce adjacent native pages");
		auto counters = plan.counters();
		require(counters.native_pages == 2 && counters.logical_pages == 2 * size / 4096 && counters.protection_runs == 1, "Protection visited the global guest domain");
		// Stronger watch on a logical neighbour wins the complete host page.
		permissions[size / 4096 - 1].overlay = PROT_NONE;
		plan.mark(1); runs.clear(); plan.synchronize_dirty(lookup, apply);
		require(runs.size() == 1 && runs[0].permissions == PROT_NONE, "Host-neighbour watch union granted hidden permissions");
		// Last watch release restores only the intersection of current base
		// rights, including unallocated or inaccessible logical neighbours.
		for (auto& p : permissions) { p.selected = false; p.overlay = PROT_READ | PROT_WRITE; }
		permissions[size / 4096 - 1].baseline = PROT_NONE;
		plan.mark(0); plan.mark(size); runs.clear(); plan.synchronize_dirty(lookup, apply);
		require(runs.size() == 2 && runs[0].permissions == PROT_NONE && runs[1].permissions == (PROT_READ | PROT_WRITE) &&
			!plan.owned(0) && !plan.owned(size), "Retirement lost current baseline rights or unrelated page ownership");
		// Reinstall after the VM changed the underlying mapping, then require
		// another protection callback even with identical overlay permissions.
		permissions[0] = {PROT_READ | PROT_WRITE, PROT_READ, true};
		runs.clear(); plan.synchronize(17, 18, lookup, apply); plan.synchronize(17, 18, lookup, apply);
		require(runs.size() == 2 && runs[0].size == size && runs[1].size == size, "Mapping ABA skipped reapplication");
		refuses([&] { plan.synchronize(1, 0, lookup, apply); });
		refuses([&] { plan.synchronize(0, 0x1'0000'0001ull, lookup, apply); });
		// Last guest byte must form one native page ending exactly at4GiB.
		runs.clear();
		plan.synchronize(0xffffffffull, 0x1'0000'0000ull, [](auto) { return page_protection_plan::access{PROT_READ | PROT_WRITE, PROT_READ, true}; }, apply);
		require(runs.size() == 1 && runs[0].address + runs[0].size == 0x1'0000'0000ull && runs[0].size == size, "Last native page wrapped or exceeded guest domain");
	}
	std::cout << "Native page plan:4/16/64KiB logical models;2 dirty native pages not1M;dedup/contiguous coalescing;union/mixed rights/absent neighbours;retirement;ABA reapply;4GiB bounds\n";
}
void full_dirty_domain()
{
	page_protection_plan plan(4096);
	unsigned calls = 0;
	std::uint64_t bytes = 0;
	allocations = 0; count_allocations = true;
	for (std::uint64_t a = 0; a < 0x1'0000'0000ull; a += 4096) { plan.mark(static_cast<std::uint32_t>(a)); plan.mark(static_cast<std::uint32_t>(a + 17)); }
	plan.synchronize_dirty([](auto) { return page_protection_plan::access{PROT_READ | PROT_WRITE, PROT_READ, true}; },
		[&](auto, auto n, auto) { ++calls; bytes += n; });
	count_allocations = false;
	require(allocations == 0 && calls == 1 && bytes == 0x1'0000'0000ull && plan.pending_pages() == 0 &&
		plan.counters().native_pages == (1u << 20), "Full dirty domain allocated,overflowed or failed run coalescing");
	std::cout << "Native page plan:all1048576 dirty slots with duplicate marks;one4GiB protection run;zero heap allocations during mark/sort/union/apply\n";
}
void indexed_union_oracle()
{
	constexpr unsigned pages = 256, owner_count = 8, watch_count = 8;
	for (const unsigned host_size : {4096u, 16384u, 65536u})
	{
		physical_page_index index(pages);
		page_protection_plan plan(host_size);
		std::array<std::shared_ptr<void>, owner_count> owners;
		for (auto& value : owners) value = std::make_shared<int>(7);
		struct mapping { unsigned owner = owner_count; std::uint64_t offset = 0; int baseline = PROT_NONE; };
		struct watch { unsigned owner = owner_count; std::uint64_t offset = 0, size = 0; int overlay = PROT_NONE; };
		std::array<mapping, pages> mappings{};
		std::array<watch, watch_count> watches{};
		std::vector<int> actual(pages * 4096 / host_size, PROT_NONE);
		std::mt19937 random(0x4550 + host_size);
		const auto lookup = [&](std::uint32_t address)
		{
			const auto& p = mappings.at(address / 4096);
			page_protection_plan::access result{p.baseline};
			if (p.owner == owner_count) return result;
			for (const auto& w : watches)
				if (w.owner == p.owner && w.offset < p.offset + 4096 && p.offset < w.offset + w.size)
				{
					result.selected = true; result.overlay &= w.overlay;
				}
			return result;
		};
		const auto apply = [&](auto address, auto size, auto permissions)
		{
			for (auto p = address / host_size, end = (address + size) / host_size; p < end; ++p) actual.at(p) = permissions;
		};
		const auto mark = [&](const watch& w)
		{
			if (w.owner != owner_count) index.aliases(owners[w.owner], w.offset, w.size, [&](auto address, auto) { plan.mark(address); });
		};
		for (unsigned operation = 0; operation < 5000; ++operation)
		{
			if (operation % 3)
			{
				const unsigned page = random() % pages, owner = random() % (owner_count + 1);
				const auto offset = (random() % 64) * 4096ull;
				const auto baseline = owner == owner_count ? PROT_NONE : random() % 3 == 0 ? PROT_READ : PROT_READ | PROT_WRITE;
				mappings[page] = {owner, offset, baseline};
				require(index.replace(page, owner == owner_count ? std::shared_ptr<void>{} : owners[owner], offset), "Composed mapping update failed");
				// The VM has just replaced permissions underneath the overlay.
				const auto native = page * 4096 / host_size;
				int base = PROT_READ | PROT_WRITE;
				for (unsigned n = native * host_size / 4096; n < (native + 1) * host_size / 4096; ++n) base &= mappings[n].baseline;
				actual[native] = base;
				plan.synchronize(std::uint64_t(page) * 4096, std::uint64_t(page + 1) * 4096, lookup, apply);
			}
			else
			{
				auto& w = watches[random() % watch_count]; mark(w);
				w = {random() % (owner_count + 1), random() % (64 * 4096), 1 + random() % 8192, random() % 2 ? PROT_NONE : PROT_READ};
				mark(w); plan.synchronize_dirty(lookup, apply);
			}
			// Independent exhaustive union: no reverse-index calls, no shared
			// logical_access callback, and no dependence on ownership flags.
			for (unsigned native = 0; native < actual.size(); ++native)
			{
				int expected = PROT_READ | PROT_WRITE;
				for (unsigned p = native * host_size / 4096; p < (native + 1) * host_size / 4096; ++p)
				{
					expected &= mappings[p].baseline;
					if (mappings[p].owner == owner_count) continue;
					for (const auto& w : watches)
					{
						if (w.owner != mappings[p].owner) continue;
						const auto begin = std::max(w.offset, mappings[p].offset);
						const auto end = std::min(w.offset + w.size, mappings[p].offset + 4096);
						if (begin < end) expected &= w.overlay;
					}
				}
				require(actual[native] == expected, "Targeted indexed union differs from full-domain permission oracle");
			}
		}
		for (auto& w : watches) { mark(w); w.owner = owner_count; }
		plan.synchronize_dirty(lookup, apply);
		for (unsigned a = 0; a < pages * 4096; a += host_size) require(!plan.owned(a), "Full watch shutdown retained native ownership");
	}
	std::cout << "Composed indexed protection:15000 mapping/watch/release operations;full-domain independent permission union after each;4/16/64KiB;late aliases/offsets/absent pages;shutdown\n";
}
int kernel_permissions(void* pointer)
{
	mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(pointer);
	const auto requested = address;
	mach_vm_size_t size = 0;
	vm_region_basic_info_data_64_t info{};
	mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
	mach_port_t object = MACH_PORT_NULL;
	const auto result = ::mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
		reinterpret_cast<vm_region_info_t>(&info), &count, &object);
	if (object != MACH_PORT_NULL) (void)::mach_port_deallocate(mach_task_self(), object);
	require(result == KERN_SUCCESS && address <= requested && requested - address < size, "Cannot inspect headless kernel protection");
	return info.protection;
}
void actual_kernel_protection()
{
	const auto size = static_cast<unsigned>(::getpagesize());
	page_protection_plan plan(size);
	physical_page_index index;
	auto owner = std::make_shared<int>(1);
	auto unrelated = std::make_shared<int>(2);
	const auto bytes = 4 * size;
	auto* memory = static_cast<std::byte*>(::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
	require(memory != MAP_FAILED, "Cannot reserve kernel protection fixture");
	const auto cleanup = std::shared_ptr<void>(memory, [=](void* p) { (void)::munmap(p, bytes); });
	char path[] = "/private/tmp/rpcs3-m4-045-alias-XXXXXX";
	const auto fd = ::mkstemp(path);
	require(fd >= 0, "Cannot create independent physical alias fixture");
	(void)::unlink(path);
	const auto file = std::shared_ptr<void>(reinterpret_cast<void*>(std::uintptr_t(fd) + 1), [=](void*) { (void)::close(fd); });
	require(::ftruncate(fd, size) == 0 &&
		::mmap(memory, size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0) == memory &&
		::mmap(memory + size, size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, fd, 0) == memory + size,
		"Cannot establish actual shared physical aliases");
	for (unsigned a = 0; a < bytes; a += 4096)
		require(index.replace(a / 4096, a < 2 * size ? owner : unrelated, a < size ? a : a - size), "Kernel alias fixture failed");
	int overlay = PROT_NONE;
	bool selected = true;
	const auto lookup = [&](auto address) { return page_protection_plan::access{PROT_READ | PROT_WRITE, address < 2 * size ? overlay : PROT_READ | PROT_WRITE, address < 2 * size && selected}; };
	const auto apply = [&](auto address, auto n, auto flags) { require(::mprotect(memory + address, n, flags) == 0, "Native kernel protection failed"); };
	const auto mark = [&] { index.aliases(owner, 17, 1, [&](auto address, auto) { plan.mark(address); }); };
	mark(); plan.synchronize_dirty(lookup, apply);
	require(kernel_permissions(memory) == PROT_NONE && kernel_permissions(memory + size) == PROT_NONE &&
		kernel_permissions(memory + 2 * size) == (VM_PROT_READ | VM_PROT_WRITE), "Exact physical alias selection failed actual kernel protections");
	mach_vm_size_t copied = 0;
	std::array<std::byte, 1> buffer{};
	require(::mach_vm_read_overwrite(mach_task_self(), reinterpret_cast<mach_vm_address_t>(memory), 1,
		reinterpret_cast<mach_vm_address_t>(buffer.data()), &copied) != KERN_SUCCESS, "Kernel read bypassed actual NO overlay");
	overlay = PROT_READ; mark(); plan.synchronize_dirty(lookup, apply);
	require(kernel_permissions(memory) == VM_PROT_READ && kernel_permissions(memory + size) == VM_PROT_READ, "Kernel weaker union failed");
	selected = false; overlay = PROT_READ | PROT_WRITE; mark(); plan.synchronize_dirty(lookup, apply);
	require(kernel_permissions(memory) == (VM_PROT_READ | VM_PROT_WRITE) && kernel_permissions(memory + size) == (VM_PROT_READ | VM_PROT_WRITE), "Kernel retirement did not restore baseline");
	memory[17] = std::byte{0x59};
	require(memory[17] == std::byte{0x59} && memory[size + 17] == std::byte{0x59} && plan.counters().protection_runs == 3,
		"Restored native physical aliases lost shared bytes or coalescing");
	std::cout << "Actual kernel protection:" << size << " byte host pages;two indexed physical aliases;one run perNO/RO/RW transition;unrelated pages untouched;Mach read refused;no guest VM/App\n";
}
}
void* operator new(std::size_t bytes)
{
	if (count_allocations) ++allocations;
	if (auto* result = std::malloc(bytes ? bytes : 1)) return result;
	throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
int main()
{
	try { mutations_and_oracle(); boundaries_and_lifetime(); full_domain_and_no_allocations(); native_page_unions(); full_dirty_domain(); indexed_union_oracle(); actual_kernel_protection(); }
	catch (const std::exception& e) { count_allocations = false; std::cerr << e.what() << '\n'; return 1; }
	std::cout << "Physical page index checks passed;no VM/renderer/App started\n";
}
