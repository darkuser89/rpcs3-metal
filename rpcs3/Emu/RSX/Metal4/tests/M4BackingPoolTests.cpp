#include "M4BackingPool.h"
#include "M4GuardedGuestMemory.h"
#include "M4IndependentBacking.h"

#include <array>
#include <iostream>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
template <typename Exception, typename Operation>
void refuses(Operation&& operation)
{
	bool failed = false;
	try { operation(); } catch (const Exception&) { failed = true; }
	require(failed, "Invalid backing operation was accepted");
}
class provider : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex mutex;
  public:
	std::shared_ptr<std::array<std::byte, 128>> storage = std::make_shared<std::array<std::byte, 128>>();
	unsigned calls = 0;
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* nested = guest_memory_epoch::current(*this)) { operation(*nested); return; }
		std::lock_guard lock(mutex);
		++calls;
		guest_memory_epoch epoch(*this, *this);
		operation(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		require(guest_memory_epoch::current(*this) == this, "Backing map escaped its provider epoch");
		if (range.address > storage->size() || range.size > storage->size() - range.address) throw std::out_of_range("Fixture bounds");
		return storage->data() + range.address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(a, b); }
	void written(guest_surface_range) noexcept override {}
};
class owning_provider final : public provider
{
  public:
	bool empty = false;
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override
	{
		require(guest_memory_epoch::current(*this) == this, "Backing ownership escaped its provider epoch");
		return empty ? std::shared_ptr<void>{} : storage;
	}
};
void budgets_and_owners()
{
	backing_pool pool(64, 2);
	unsigned maps = 0, releases = 0, source_deaths = 0;
	auto source = std::shared_ptr<int>(new int(7), [&](int* p) { ++source_deaths; delete p; });
	const auto map = [&] { ++maps; return std::shared_ptr<void>(new int(9), [&](void* p) { ++releases; delete static_cast<int*>(p); }); };
	auto first = pool.acquire(source, 32, map);
	const auto alias = std::shared_ptr<void>(source, source.get());
	auto same = pool.acquire(alias, 32, map);
	const auto identity = pool.source_identity(first);
	require(!identity.expired() && !identity.owner_before(source) && !source.owner_before(identity) &&
		pool.source_identity({}).expired() && pool.source_identity(source).expired(), "Reverse backing identity confused source and mapping owners");
	require(pool.matches(source, first) && pool.matches(alias, same) && !pool.matches({}, first) &&
		!pool.matches(source, {}) && !pool.matches(std::make_shared<int>(7), first) && maps == 1,
		"Physical ownership lookup allocated or confused independent source owners");
	require(first == same && maps == 1 && pool.retained_bytes() == 32 && source.use_count() == 2, "Aliased owner was duplicated or strongly retained");
	refuses<std::logic_error>([&] { (void)pool.acquire(source, 16, map); });
	auto second_source = std::make_shared<int>(8);
	auto second = pool.acquire(second_source, 32, map);
	refuses<std::length_error>([&] { (void)pool.acquire(std::make_shared<int>(10), 1, map); });
	require(maps == 2 && pool.retained_bytes() == 64, "Capacity refusal allocated backing");
	second.reset();
	refuses<std::runtime_error>([&] { (void)pool.acquire(second_source, 32, []() -> std::shared_ptr<void> { throw std::runtime_error("mapper failure"); }); });
	refuses<std::runtime_error>([&] { (void)pool.acquire(second_source, 32, []() -> std::shared_ptr<void> { return {}; }); });
	refuses<std::length_error>([&] { (void)pool.acquire(second_source, 33, map); });
	require(pool.retained_bytes() == 32 && maps == 2, "Failed mapping or budget refusal leaked its reservation");
	same.reset();
	first.reset();
	require(pool.retained_bytes() == 0 && releases == 2 && source_deaths == 0, "Final-owner release lost its charge or source");
	int reused_address = 0;
	auto previous = std::shared_ptr<void>(&reused_address, [](void*) {});
	auto replacement = std::shared_ptr<void>(&reused_address, [](void*) {});
	auto old_mapping = pool.acquire(previous, 32, map);
	auto new_mapping = pool.acquire(replacement, 32, map);
	const auto old_identity = pool.source_identity(old_mapping), new_identity = pool.source_identity(new_mapping);
	require(!old_identity.owner_before(previous) && !previous.owner_before(old_identity) &&
		!new_identity.owner_before(replacement) && !replacement.owner_before(new_identity) &&
		(old_identity.owner_before(new_identity) || new_identity.owner_before(old_identity)), "Reverse backing identity confused a recycled pointer");
	require(pool.matches(previous, old_mapping) && pool.matches(replacement, new_mapping) &&
		!pool.matches(previous, new_mapping) && !pool.matches(replacement, old_mapping),
		"Physical ownership lookup confused reused source pointers");
	require(old_mapping != new_mapping && pool.retained_bytes() == 64, "Reused source pointer incorrectly reused an old control-block mapping");
	old_mapping.reset(); new_mapping.reset();
	require(pool.retained_bytes() == 0, "Distinct replacement ownership leaked its budget");
	std::cout << "Backing pool deduplicates ownership,refunds failed maps and enforces byte/entry budgets\n";
}
void independent_mapping()
{
	constexpr std::size_t bytes = 65536;
	refuses<std::length_error>([] { (void)independent_backing_charge(0); });
	refuses<std::length_error>([] { (void)independent_backing_charge(std::numeric_limits<std::size_t>::max()); });
	char name[] = "/tmp/rpcs3-metal4-backing-XXXXXX";
	const int fd = ::mkstemp(name);
	require(fd >= 0, "Backing fixture file creation failed");
	(void)::unlink(name);
	require(::ftruncate(fd, bytes) == 0, "Backing fixture file size failed");
	bool closed = false;
	auto source = std::shared_ptr<int>(new int(fd), [&](int* value) { (void)::close(*value); closed = true; delete value; });
	auto* guest = static_cast<std::byte*>(::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
	require(guest != MAP_FAILED, "Guest fixture mapping failed");
	for (std::size_t i = 0; i < bytes; ++i) guest[i] = std::byte((i * 29 + 17) & 255);
	std::shared_ptr<void> lease;
	{
		const auto charge = independent_backing_charge(bytes);
		backing_pool pool(charge);
		refuses<std::runtime_error>([&] { (void)pool.acquire(source, charge, [&] { return map_independent_backing(bytes, [](void*) { return MAP_FAILED; }); }); });
		refuses<std::runtime_error>([&] { (void)pool.acquire(source, charge, [&] { return map_independent_backing(bytes, [](void*) -> void* { throw std::runtime_error("fixed mapper failure"); }); }); });
		require(pool.retained_bytes() == 0, "Failed reserved mapping retained its charge");
		lease = pool.acquire(source, charge, [&]
		{
			return map_independent_backing(bytes, [&](void* reserved)
			{
				require(reinterpret_cast<std::uintptr_t>(reserved) % backing_alignment == 0, "Backing address lacks VM allocation alignment");
				return ::mmap(reserved, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
			});
		});
		require(source.use_count() == 1, "Lease retained the original allocation handle");
		static_cast<std::byte*>(lease.get())[12345] = std::byte{0x5c};
		require(guest[12345] == std::byte{0x5c}, "Independent lease does not reference the same physical bytes");
		source.reset();
		require(closed && pool.retained_bytes() == charge, "Source close prematurely retired the independent mapping");
		require(::munmap(guest, bytes) == 0, "Guest fixture unmap failed");
		auto* replacement = static_cast<std::byte*>(::mmap(guest, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0));
		require(replacement == guest, "Guest fixture address reuse failed");
		std::fill_n(replacement, bytes, std::byte{0x77});
		for (std::size_t i = 0; i < bytes; ++i)
			require(static_cast<std::byte*>(lease.get())[i] == (i == 12345 ? std::byte{0x5c} : std::byte((i * 29 + 17) & 255)), "Old backing was replaced after virtual-address reuse");
		require(::munmap(replacement, bytes) == 0, "Replacement fixture unmap failed");
	}
	std::thread completion([owner = std::move(lease)]() mutable { owner.reset(); });
	completion.join();
	std::cout << "65536 physical bytes survive source close,guest unmap,identical-address replacement,pool destruction and completion-thread release\n";
}
void guarded_ranges()
{
	auto access = std::make_shared<owning_provider>();
	guarded_guest_memory memory(access, 4, 64, 4);
	const std::array ranges{guest_surface_range{0, 16}, guest_surface_range{32, 16}};
	auto lease = memory.retain_backing(ranges);
	std::weak_ptr<void> old = access->storage;
	const auto before = access->calls;
	access->execute([&](auto&) { access->storage = std::make_shared<std::array<std::byte, 128>>(); memory.notify_mapping_changed({0, 128}); });
	require(!old.expired() && access->calls == before + 1, "Range ownership did not survive producer remapping");
	std::thread completion([owner = std::move(lease)]() mutable { owner.reset(); });
	completion.join();
	require(old.expired(), "Old backing was retained after its final consumer");
	const auto calls = access->calls;
	const std::array wrap{guest_surface_range{0xfffffffeu, 4}};
	const std::array oversized{guest_surface_range{0, 65}};
	const std::array excessive{guest_surface_range{0, 1}, guest_surface_range{2, 1}, guest_surface_range{4, 1}, guest_surface_range{6, 1}, guest_surface_range{8, 1}};
	refuses<std::overflow_error>([&] { (void)memory.retain_backing(wrap); });
	refuses<std::length_error>([&] { (void)memory.retain_backing(oversized); });
	refuses<std::length_error>([&] { (void)memory.retain_backing(excessive); });
	require(access->calls == calls, "Invalid backing geometry entered provider exclusion");
	access->empty = true;
	refuses<std::runtime_error>([&] { (void)memory.retain_backing(ranges); });
	guarded_guest_memory unsupported(std::make_shared<provider>());
	refuses<std::runtime_error>([&] { (void)unsupported.retain_backing(ranges); });
	std::cout << "Guarded backing capture owns one provider epoch;remap lifetime,range budgets,absent and unsupported providers checked\n";
}
} // namespace
int main()
{
	try
	{
		budgets_and_owners();
		independent_mapping();
		guarded_ranges();
		return 0;
	}
	catch (const std::exception& failure)
	{
		std::cerr << failure.what() << '\n';
		return 1;
	}
}
