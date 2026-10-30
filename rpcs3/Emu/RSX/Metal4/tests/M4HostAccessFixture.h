#pragma once

#include "M4GuardedGuestMemory.h"
#include "M4HostWriterGate.h"
#include "M4BackingPool.h"
#include "M4IndependentBacking.h"
#include <array>
#include <mutex>
#include <unistd.h>

namespace metal4::test
{
// CPU-only stand-in for VM exclusion,with real file-backed allocations and
// independent kernel mappings. It never creates a guest VM or starts RPCS3.
class host_access_fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view
{
	std::mutex m_mutex;
	std::array<std::shared_ptr<int>, 2> m_sources;
	backing_pool m_pool{8 * independent_backing_charge(65536), 8};
	std::shared_ptr<void> m_first_alias;
	bool m_first_mapped = true, m_first_accessible = true;
	static std::shared_ptr<int> source()
	{
		char name[] = "/tmp/rpcs3-metal4-host-access-XXXXXX";
		const int fd = ::mkstemp(name);
		if (fd < 0) throw std::runtime_error("Physical fixture file creation failed");
		(void)::unlink(name);
		auto owner = std::shared_ptr<int>(new int(fd), [](int* value) { (void)::close(*value); delete value; });
		if (::ftruncate(fd, 65536)) throw std::runtime_error("Physical fixture allocation failed");
		return owner;
	}
	[[nodiscard]] static guest_surface_range logical(guest_surface_range range)
	{
		validate_guest_surface_range(range);
		if (range.address >= 0x40000) range.address -= 0x40000;
		if (range.address >= 131072 || range.size > 131072 - range.address)
			throw std::out_of_range("Physical fixture access crossed the mapping");
		return range;
	}
  public:
	std::shared_ptr<host_writer_gate> gate;
	std::mutex mapping_mutex;
	std::byte* bytes = nullptr;
	unsigned captures = 0, reservations = 0;
	bool segments_supported = true;
	unsigned physical_queries = 0, fail_physical_query = 0;
	bool fail_reservation_planning = false;
	std::array<guest_surface_range, 1024> reservation_ranges{};
	host_access_fixture()
	{
		auto reserved = ::mmap(nullptr, 131072, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
		if (reserved == MAP_FAILED) throw std::runtime_error("Physical fixture address reservation failed");
		bytes = static_cast<std::byte*>(reserved);
		try
		{
			for (unsigned i = 0; i < 2; ++i)
			{
				m_sources[i] = source();
				if (::mmap(bytes + i * 65536, 65536, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, *m_sources[i], 0) != bytes + i * 65536)
					throw std::runtime_error("Physical fixture mapping failed");
			}
		}
		catch (...) { (void)::munmap(bytes, 131072); throw; }
	}
	~host_access_fixture() override { (void)::munmap(bytes, 131072); }
	void replace_first()
	{
		auto replacement = source();
		if (::mmap(bytes, 65536, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, *replacement, 0) != bytes)
			throw std::runtime_error("Physical fixture replacement mapping failed");
		m_sources[0] = std::move(replacement); // Close the old allocation handle.
		m_first_mapped = m_first_accessible = true;
	}
	void unmap_first()
	{
		if (::mmap(bytes, 65536, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) != bytes)
			throw std::runtime_error("Physical fixture unmap failed");
		m_sources[0].reset(); m_first_mapped = m_first_accessible = false;
	}
	void protect_first(bool accessible)
	{
		if (::mprotect(bytes, 65536, accessible ? PROT_READ | PROT_WRITE : PROT_NONE)) throw std::runtime_error("Physical fixture protection failed");
		m_first_accessible = accessible;
	}
	void pin_first_alias()
	{
		execute([&](auto&)
		{
			m_first_alias = retain_segments({0, 65536})->segments()[0].owner;
		});
	}
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* current = guest_memory_epoch::current(*this)) { operation(*current); return; }
		const auto run = [&]
		{
			std::scoped_lock lock(m_mutex, mapping_mutex);
			++captures;
			guest_memory_epoch epoch(*this, *this);
			operation(*this);
		};
		if (gate)
		{
			auto exclusion = gate->acquire([] { std::this_thread::yield(); });
			if (!exclusion) throw std::runtime_error("Physical fixture host exclusion refused");
			run();
		}
		else run();
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		const auto mapped = logical(range);
		if (m_first_alias && range.address >= 0x40000 && mapped.address < 65536)
		{
			if (range.size > 65536 - mapped.address) throw std::out_of_range("Pinned fixture alias crosses an allocation");
			return static_cast<std::byte*>(m_first_alias.get()) + mapped.address;
		}
		if (mapped.address < 65536 && (!m_first_mapped || !m_first_accessible)) throw std::runtime_error("Physical fixture mapping is absent or protected");
		return bytes + mapped.address;
	}
	std::shared_ptr<const physical_backing> retain_segments(guest_surface_range range) override
	{
		if (!segments_supported) return {};
		const auto mapped = logical(range);
		std::vector<physical_backing::segment> parts;
		for (std::uint64_t cursor = 0; cursor < range.size;)
		{
			const auto offset = mapped.address + cursor;
			const auto block = offset / 65536;
			const auto count = std::min(range.size - cursor, 65536 - offset % 65536);
			if (!block && !m_first_mapped && !(m_first_alias && range.address >= 0x40000)) throw std::runtime_error("Physical fixture allocation is absent");
			auto owner = m_first_alias && range.address >= 0x40000 && block == 0 ? m_first_alias :
				m_pool.acquire(m_sources[block], independent_backing_charge(65536), [&]
			{
				return map_independent_backing(65536, [&](void* reserved)
				{
					return ::mmap(reserved, 65536, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, *m_sources[block], 0);
				});
			});
			parts.push_back({{static_cast<std::uint32_t>(range.address + cursor), count}, std::move(owner), offset % 65536, 65536});
			cursor += count;
		}
		return std::make_shared<physical_backing>(range, std::move(parts));
	}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override
	{
		auto owners = std::make_shared<std::vector<std::shared_ptr<const physical_backing>>>();
		for (auto range : ranges) if (range.size) owners->push_back(retain_segments(range));
		return owners;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		if (m_first_alias || !m_first_mapped)
		{
			const auto mapped = logical(a);
			for (std::size_t cursor = 0; cursor < a.size;)
			{
				const auto offset = mapped.address + cursor;
				const auto count = std::min(a.size - cursor, 65536 - offset % 65536);
				if (offset >= 65536 || m_first_mapped || (a.address >= 0x40000 && m_first_alias))
					if (overlaps_backing(*retain_segments({static_cast<std::uint32_t>(a.address + cursor), count}), b, 1 << 20)) return true;
				cursor += count;
			}
			return false;
		}
		return overlaps(logical(a), logical(b));
	}
	bool overlaps_backing(const physical_backing& written, guest_surface_range current, std::size_t limit) override
	{
		if (++physical_queries == fail_physical_query) throw std::runtime_error("Injected physical reconciliation failure");
		const std::array<std::shared_ptr<const physical_backing>, 1> owners{std::make_shared<physical_backing>(written)};
		return matches_current(physical_identity(owners), current, limit);
	}
	bool overlaps_identity(const physical_identity& kept, guest_surface_range current, std::size_t limit) override
	{
		if (++physical_queries == fail_physical_query) throw std::runtime_error("Injected physical reconciliation failure");
		return matches_current(kept, current, limit);
	}
	bool matches_current(const physical_identity& kept, guest_surface_range current, std::size_t limit)
	{
		const auto mapped = logical(current);
		std::size_t runs = 0;
		for (std::size_t cursor = 0; cursor < current.size;)
		{
			if (++runs > limit) throw std::length_error("Physical fixture query budget exceeded");
			const auto block = (mapped.address + cursor) / 65536, offset = (mapped.address + cursor) % 65536;
			const auto count = std::min(current.size - cursor, 65536 - offset);
			const bool pinned = !block && current.address >= 0x40000 && m_first_alias;
			if (block || pinned || m_first_mapped)
				for (const auto& part : kept.segments())
				{
					const bool same = pinned ? !part.owner.owner_before(m_first_alias) && !m_first_alias.owner_before(part.owner) : m_pool.matches(m_sources[block], part.owner);
					if (same && part.offset < offset + count && offset < part.offset + part.size) return true;
				}
			cursor += count;
		}
		return false;
	}
	std::vector<guest_surface_range> physical_write_aliases(std::span<const std::shared_ptr<const physical_backing>> written, std::size_t limit) override
	{
		if (fail_reservation_planning) throw std::runtime_error("Injected physical reservation planning failure");
		std::vector<guest_surface_range> result;
		for (const auto base : {0u, 0x40000u})
			for (unsigned block = 0; block < 2; ++block)
			{
				const bool pinned = !block && base && m_first_alias;
				if (!block && !pinned && !m_first_mapped) continue;
				for (const auto& backing : written)
					for (const auto& part : backing->segments())
					{
						const bool same = pinned ? !part.owner.owner_before(m_first_alias) && !m_first_alias.owner_before(part.owner) : m_pool.matches(m_sources[block], part.owner);
						if (!same) continue;
						if (result.size() == limit) throw std::length_error("Physical fixture reservation budget exceeded");
						result.push_back({static_cast<std::uint32_t>(base + block * 65536 + part.offset), part.guest.size});
					}
			}
		return result;
	}
	void written(guest_surface_range range) noexcept override
	{
		if (reservations < reservation_ranges.size()) reservation_ranges[reservations] = range;
		++reservations;
	}
};
} // namespace metal4::test
