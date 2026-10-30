#pragma once

#include "M4GuestCoverage.h"
#include "M4ExclusiveGuestMemory.h"
#include "M4HostWriteJournal.h"
#include "M4PhysicalSurface.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <type_traits>
#include <unordered_map>

namespace metal4
{
// Exact monotonic revisions for a bounded set of active-byte coverages.
// Provider writers call before_write_in_transaction and mutate in ONE callback.
// Host DMA writers instead hold a continuous exclusion ticket through a bounded
// completion intent. Provider capture excludes them and drains validated events.
// Direct PPU/SPU/fault/partial-store recovery and composite binding remain open.
// Owned layouts retain stable registrations until their final owner expires.
// Retirement happens inside the next exclusive transaction,never in a GPU
// completion/destructor. Legacy unleased reads pin registrations permanently.
class guarded_guest_memory final : public guest_surface_memory
{
  public:
	struct maintenance_statistics
	{
		std::uint64_t runs = 0, reuses = 0, watch_visits = 0, physical_visits = 0, indexed_lookups = 0;
	};
  private:
	maintenance_statistics m_statistics{};
	std::uint64_t m_maintenance_epoch = 0, m_maintenance_mapping = 0;
	host_write_journal::observation m_maintenance_journal{};
	std::shared_ptr<exclusive_guest_memory> m_access;
	std::size_t m_capacity, m_byte_limit, m_run_limit;
	std::uint64_t m_clock = 0, m_observed_mapping = 1;
	std::shared_ptr<mapping_epoch> m_mapping = std::make_shared<mapping_epoch>();
	host_write_journal m_host_writes;
	struct entry
	{
		guest_surface_layout coverage;
		std::uint64_t revision;
		bool permanent = true;
		std::weak_ptr<void> owner;
		std::uint64_t mapping;
		std::shared_ptr<void> pinned_owner = {};
	};
	std::list<entry> m_entries;
	// Weak cells contain only allocation identities and revisions. Tickets own
	// physical mappings;retiring a cell cannot release a mapping or GPU object.
	std::list<std::weak_ptr<physical_surface_state>> m_physical;
	[[nodiscard]] physical_surface_state& owned_physical(const physical_surface& surface) const
	{
		for (const auto& weak : m_physical)
			if (!weak.owner_before(surface.m_state) && !surface.m_state.owner_before(weak)) return *surface.m_state;
		throw std::invalid_argument("Physical surface ticket belongs to another revision domain");
	}
	[[nodiscard]] std::vector<std::shared_ptr<physical_surface_state>> affected_physical(
		exclusive_guest_memory::view& view, guest_surface_range range) const
	{
		std::vector<std::shared_ptr<physical_surface_state>> result;
		for (const auto& weak : m_physical)
			if (auto cell = weak.lock(); cell && view.overlaps_identity(cell->identity, range, m_run_limit)) result.push_back(std::move(cell));
		return result;
	}
	[[nodiscard]] static std::byte* checked_map(exclusive_guest_memory::view& view, guest_surface_range range, bool write)
	{
		const auto pointer = view.map(range, write);
		if (!pointer)
			throw std::runtime_error("Guest memory provider returned an absent mapping");
		return pointer;
	}
	[[nodiscard]] static bool same(const guest_surface_layout& a, const guest_surface_layout& b)
	{
		if (a.range.address != b.range.address || a.range.size != b.range.size || a.width != b.width ||
			a.height != b.height || a.pixel_bytes != b.pixel_bytes || a.row_pitch != b.row_pitch || a.column_stride != b.column_stride || bool(a.morton) != bool(b.morton))
			return false;
		return !a.morton || (a.morton->storage == b.morton->storage && a.morton->origin == b.morton->origin && a.morton->size == b.morton->size);
	}
	struct coverage_hash
	{
		std::size_t operator()(const guest_surface_layout& value) const noexcept
		{
			std::size_t hash = 0;
			const auto mix = [&](std::size_t n) { hash ^= n + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2); };
			mix(value.range.address); mix(value.range.size); mix(value.width); mix(value.height);
			mix(value.pixel_bytes); mix(value.row_pitch); mix(value.column_stride); mix(bool(value.morton));
			if (value.morton)
				for (const auto* part : {&value.morton->storage, &value.morton->origin, &value.morton->size})
					for (auto n : *part) mix(n);
			return hash;
		}
	};
	struct coverage_equal
	{
		bool operator()(const guest_surface_layout& a, const guest_surface_layout& b) const { return same(a, b); }
	};
	std::unordered_map<guest_surface_layout, std::list<entry>::iterator, coverage_hash, coverage_equal> m_index;
	void insert(entry value)
	{
		m_entries.push_front(std::move(value));
		try
		{
			auto key = m_entries.front().coverage;
			key.registration.reset(); // The index must never pin an owned watch.
			if (!m_index.emplace(std::move(key), m_entries.begin()).second)
				throw std::logic_error("Duplicate native watch index insertion");
		}
		catch (...) { m_entries.pop_front(); throw; }
	}
	std::list<entry>::iterator find(exclusive_guest_memory::view& view, const guest_surface_layout& coverage)
	{
		++m_statistics.indexed_lookups;
		auto found = m_index.find(coverage);
		if (found != m_index.end() && !found->second->permanent && found->second->owner.expired())
		{
			prune(view, true);
			found = m_index.find(coverage);
		}
		return found == m_index.end() ? m_entries.end() : found->second;
	}
	[[nodiscard]] std::uint64_t tick()
	{
		if (m_clock == std::numeric_limits<std::uint64_t>::max())
			throw std::overflow_error("Guest memory revision counter exhausted");
		return ++m_clock;
	}
	void prune(exclusive_guest_memory::view& view, bool force = false)
	{
		const auto epoch = guest_memory_epoch::identity(*m_access);
		const auto mapping = m_mapping->current();
		const auto journal = m_host_writes.observe();
		// Provider exclusion is essential. Nonparticipating legacy providers get
		// no identity and always run maintenance. Recovery-mode changes count.
		if (!force && epoch && epoch == m_maintenance_epoch && mapping == m_maintenance_mapping &&
			journal.reusable() && journal == m_maintenance_journal && m_mapping->observe())
		{
			++m_statistics.reuses;
			return;
		}
		++m_statistics.runs;
		m_host_writes.drain([&](std::span<const host_write_journal::completion> writes)
		{
			// This runs only in provider exclusion. Producers publish epochs without
			// touching registrations. Refresh only watches whose virtual pages changed;
			// a physical mirror outside that virtual mapping keeps its backing/tag.
			const auto mapping = m_mapping->current();
			std::vector<std::pair<entry*, std::uint64_t>> remapped;
			std::vector<entry*> changed;
			std::vector<std::shared_ptr<physical_surface_state>> physical;
			const auto live = [](const entry& value) { return value.permanent || !value.owner.expired(); };
			if (mapping != m_observed_mapping)
			{
				for (auto& entry : m_entries)
				{
					if (!live(entry)) continue;
					const auto current = m_mapping->current(entry.coverage.range);
					if (current == entry.mapping)
						continue;
					remapped.emplace_back(&entry, current);
					changed.push_back(&entry);
				}
			}
			for (const auto& write : writes)
			{
				for (const auto& weak : m_physical)
					if (auto cell = weak.lock(); cell && std::find(physical.begin(), physical.end(), cell) == physical.end() &&
						(write.backing ? cell->identity.overlaps(*write.backing) : view.overlaps_identity(cell->identity, write.range, m_run_limit)))
						physical.push_back(std::move(cell));
				for (auto& entry : m_entries)
				{
					if (!live(entry) || std::find(changed.begin(), changed.end(), &entry) != changed.end()) continue;
					guest_surface_byte_cursor cursor(entry.coverage);
					std::size_t runs = 0;
					while (const auto active = cursor.next())
					{
						if (++runs > m_run_limit) throw std::length_error("Host reconciliation exceeds active-byte run budget");
						const bool match = write.backing ? view.overlaps_backing(*write.backing, *active, m_run_limit) :
							view.same_storage(*active, write.range);
						if (match) { changed.push_back(&entry); break; }
					}
				}
			}
			// All mapping/physical/cursor checks and allocations finish before any
			// revision changes. Failed planning keeps tags,epochs and intents intact.
			if (!changed.empty() || !physical.empty())
			{
				const auto revision = tick();
				for (auto* entry : changed) entry->revision = revision;
				for (const auto& cell : physical) cell->revision = revision;
			}
			for (const auto& [entry, current] : remapped) entry->mapping = current;
			m_observed_mapping = mapping;
			for (auto it = m_entries.begin(); it != m_entries.end();)
			{
				++m_statistics.watch_visits;
				if (!live(*it)) { m_index.erase(it->coverage); it = m_entries.erase(it); }
				else ++it;
			}
			m_physical.remove_if([&](const auto& cell) { ++m_statistics.physical_visits; return cell.expired(); });
		});
		m_maintenance_epoch = epoch;
		m_maintenance_mapping = m_mapping->current();
		m_maintenance_journal = m_host_writes.observe();
	}
	void protect_watch(exclusive_guest_memory::view& view, const guest_surface_layout& coverage, const std::shared_ptr<void>& owner)
	{
		if (!view.uses_cpu_protection()) return;
		std::vector<guest_surface_range> ranges;
		guest_surface_byte_cursor cursor(coverage);
		while (const auto active = cursor.next())
		{
			if (ranges.size() == m_run_limit) throw std::length_error("Guest protection exceeds active-byte run budget");
			ranges.push_back(*active);
		}
		if (!ranges.empty()) view.watch_ranges(ranges, owner);
	}
	entry& track(exclusive_guest_memory::view& view, const guest_surface_layout& coverage)
	{
		prune(view);
		if (auto it = find(view, coverage); it != m_entries.end())
		{
			if (coverage.registration)
			{
				const auto owner = it->owner.lock();
				if (!owner || owner.get() != coverage.registration.get() || owner.owner_before(coverage.registration) || coverage.registration.owner_before(owner))
					throw std::invalid_argument("Guest memory watch belongs to a different registration");
			}
			else
			{
				it->permanent = true; // Legacy callers require stable unleased tags.
				if (!it->pinned_owner) it->pinned_owner = it->owner.lock();
				if (!it->pinned_owner) it->owner = it->pinned_owner = std::make_shared<std::byte>();
			}
			protect_watch(view, coverage, coverage.registration ? coverage.registration : it->pinned_owner);
			return *it;
		}
		if (coverage.registration)
			throw std::invalid_argument("Guest memory watch is no longer registered");
		if (m_entries.size() == m_capacity) prune(view, true);
		if (m_entries.size() == m_capacity)
			throw std::length_error("Guest memory coverage-registration budget exhausted");
		const auto revision = tick();
		const auto owner = std::make_shared<std::byte>();
		protect_watch(view, coverage, owner);
		insert({coverage, revision, true, owner, m_mapping->current(coverage.range), owner});
		return m_entries.front();
	}
	void validate(const guest_surface_layout& coverage, bool packed = false) const
	{
		(void)guest_surface_byte_cursor(coverage);
		if ((packed ? guest_surface_active_bytes(coverage) : coverage.range.size) > m_byte_limit)
			throw std::length_error("Guest memory operation exceeds snapshot budget");
	}
	// Plan revisions before touching bytes. Iteration and cursor validation can
	// fail;assignment after successful mutation cannot throw.
	[[nodiscard]] std::vector<entry*> affected(exclusive_guest_memory::view& view, guest_surface_range range)
	{
		std::vector<entry*> result;
		for (auto& entry : m_entries)
		{
			if (!view.same_storage(entry.coverage.range, range))
				continue;
			guest_surface_byte_cursor cursor(entry.coverage);
			while (const auto active = cursor.next())
				if (view.same_storage(*active, range))
				{
					result.push_back(&entry);
					break;
				}
		}
		return result;
	}
	struct patch
	{
		std::byte* destination;
		std::size_t source_offset, size;
		guest_surface_range range;
		std::size_t logical_offset = 0;
	};

  public:
	[[nodiscard]] maintenance_statistics statistics()
	{
		maintenance_statistics result;
		m_access->execute([&](auto&) { result = m_statistics; });
		return result;
	}
	[[nodiscard]] bool uses_provider(const exclusive_guest_memory& access) const noexcept { return m_access.get() == &access; }
	std::optional<bool> physical_identity_overlaps_current(const physical_identity& identity, const guest_surface_layout& coverage) override
	{
		validate(coverage);
		if (identity.segments().size() > m_run_limit) throw std::length_error("Physical selection exceeds identity segment budget");
		bool result = false;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			guest_surface_byte_cursor cursor(coverage);
			std::size_t runs = 0;
			while (const auto active = cursor.next())
			{
				if (++runs > m_run_limit) throw std::length_error("Physical selection exceeds active-byte run budget");
				if (view.overlaps_identity(identity, *active, m_run_limit)) { result = true; break; }
			}
		});
		return result;
	}
	// Fault publication has already completed outside provider exclusion.
	// Invalidate original owners even after a virtual remap,all current active
	// physical aliases,and old retained physical tickets. No bytes or OS rights
	// change here. Every lookup/allocation precedes the single revision commit.
	void invalidate_physical_in_transaction(exclusive_guest_memory::view& view, const physical_identity& identity,
		std::span<const std::shared_ptr<void>> registrations)
	{
		if (guest_memory_epoch::current(*m_access) != &view)
			throw std::logic_error("Physical invalidation requires this memory domain's active provider view");
		prune(view);
		std::vector<entry*> revisions;
		for (const auto& owner : registrations)
		{
			if (!owner) throw std::invalid_argument("Physical invalidation requires live registration owners");
			bool found = false;
			for (auto& entry : m_entries)
				if (!entry.owner.owner_before(owner) && !owner.owner_before(entry.owner))
				{
					found = true;
					if (std::find(revisions.begin(), revisions.end(), &entry) == revisions.end()) revisions.push_back(&entry);
				}
			if (!found) throw std::invalid_argument("Physical invalidation contains a foreign registration");
		}
		for (auto& entry : m_entries)
		{
			if (std::find(revisions.begin(), revisions.end(), &entry) != revisions.end()) continue;
			guest_surface_byte_cursor cursor(entry.coverage);
			std::size_t runs = 0;
			while (const auto active = cursor.next())
			{
				if (++runs > m_run_limit) throw std::length_error("Physical invalidation exceeds active-byte run budget");
				if (view.overlaps_identity(identity, *active, m_run_limit)) { revisions.push_back(&entry); break; }
			}
		}
		std::vector<std::shared_ptr<physical_surface_state>> physical;
		for (const auto& weak : m_physical)
			if (auto cell = weak.lock(); cell && cell->identity.overlaps(identity)) physical.push_back(std::move(cell));
		if (revisions.empty() && physical.empty()) return;
		const auto revision = tick();
		for (auto* entry : revisions) entry->revision = revision;
		for (const auto& cell : physical) cell->revision = revision;
	}
	std::shared_ptr<const physical_surface> capture_physical_surface(const guest_surface_layout& coverage, std::uint64_t expected) override
	{
		validate(coverage);
		std::shared_ptr<const physical_surface> result;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			if (track(view, coverage).revision != expected) throw std::runtime_error("Physical capture lost its virtual input revision");
			if (m_physical.size() == m_capacity) prune(view, true);
			if (m_physical.size() == m_capacity) throw std::length_error("Physical surface registration budget exhausted");
			std::vector<std::shared_ptr<const physical_backing>> parts;
			guest_surface_byte_cursor cursor(coverage);
			std::size_t segments = 0;
			while (const auto active = cursor.next())
			{
				(void)checked_map(view, *active, false);
				auto backing = view.retain_segments(*active);
				if (!backing) return; // Legacy provider cannot support physical recovery.
				if (backing->range().address != active->address || backing->range().size != active->size ||
					backing->segments().size() > m_run_limit - segments) throw std::runtime_error("Physical capture lacks exact bounded active slices");
				segments += backing->segments().size(); parts.push_back(std::move(backing));
			}
			if (parts.empty()) throw std::invalid_argument("Physical surface capture requires active bytes");
			auto cell = std::make_shared<physical_surface_state>(physical_surface_state{physical_identity(parts), tick()});
			result = std::shared_ptr<const physical_surface>(new physical_surface(coverage, std::move(parts), cell));
			m_physical.push_back(cell);
		});
		return result;
	}
	std::optional<guest_surface_snapshot> read_physical_surface(const physical_surface& surface) override
	{
		if (surface.coverage().range.size > m_byte_limit) throw std::length_error("Physical snapshot exceeds byte budget");
		guest_surface_snapshot result{surface.m_expected, std::vector<std::byte>(surface.coverage().range.size)};
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			if (owned_physical(surface).revision != surface.m_expected) throw std::runtime_error("Physical snapshot lost its captured revision");
			for (const auto& backing : surface.m_parts)
				for (const auto& part : backing->segments())
					std::memcpy(result.bytes.data() + part.guest.address - surface.coverage().range.address, part.data(), part.guest.size);
		});
		return result;
	}
	bool physical_surface_current(const physical_surface& surface) override
	{
		bool current = false;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view); current = owned_physical(surface).revision == surface.m_expected;
		});
		return current;
	}
	std::optional<bool> physical_surface_overlaps_current(const physical_surface& surface, guest_surface_range range) override
	{
		validate_guest_surface_range(range);
		bool result = false;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			result = view.overlaps_identity(owned_physical(surface).identity, range, m_run_limit);
		});
		return result;
	}
	std::optional<bool> physical_surface_virtual_current(const physical_surface& surface, std::uint64_t expected) override
	{
		bool result = false;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			result = owned_physical(surface).revision == surface.m_expected && track(view, surface.coverage()).revision == expected;
		});
		return result;
	}
	std::optional<std::uint64_t> compare_write_physical(std::span<const physical_surface_write> writes) override
	{
		const auto publication = compare_write_physical(writes, {});
		if (publication) return publication->revision;
		return {};
	}
	std::optional<physical_surface_publication> compare_write_physical(std::span<const physical_surface_write> writes,
		std::span<const std::uint64_t> expected) override
	{
		if (!expected.empty() && expected.size() != writes.size()) throw std::invalid_argument("Physical publication watch count differs from writes");
		if (writes.empty()) return physical_surface_publication{0, {}};
		if (writes.size() > m_run_limit) throw std::length_error("Physical publication exceeds ticket budget");
		std::size_t total = 0;
		for (const auto& write : writes)
		{
			if (!write.surface || write.bytes.size() != write.surface->coverage().range.size)
				throw std::invalid_argument("Physical publication payload differs from captured storage");
			if (write.bytes.size() > m_byte_limit - total) throw std::length_error("Physical publication exceeds aggregate byte budget");
			total += write.bytes.size();
		}
		physical_surface_publication result{0, std::vector<std::uint64_t>(expected.size())};
		bool published = false;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			std::vector<entry*> virtual_watches;
			virtual_watches.reserve(expected.size());
			for (std::size_t i = 0; i < expected.size(); ++i)
			{
				(void)owned_physical(*writes[i].surface);
				virtual_watches.push_back(&track(view, writes[i].surface->coverage()));
			}
			bool same_virtual = !expected.empty();
			for (std::size_t i = 0; i < expected.size(); ++i) same_virtual &= virtual_watches[i]->revision == expected[i];
			// track() can drain revisions;validate physical tickets after ALL watches.
			for (const auto& write : writes) if (owned_physical(*write.surface).revision != write.surface->m_expected) return;
			std::vector<std::byte> payload;
			payload.reserve(total);
			for (const auto& write : writes) payload.insert(payload.end(), write.bytes.begin(), write.bytes.end());
			std::vector<patch> patches;
			std::vector<std::shared_ptr<const physical_backing>> backings;
			std::size_t offset = 0;
			for (const auto& write : writes)
			{
				for (const auto& backing : write.surface->m_parts)
				{
					backings.push_back(backing);
					for (const auto& part : backing->segments())
					{
						if (patches.size() == m_run_limit) throw std::length_error("Physical publication exceeds patch budget");
						patches.push_back({part.data(), offset + part.guest.address - write.surface->coverage().range.address, part.guest.size, part.guest});
					}
				}
				offset += write.bytes.size();
			}
			std::vector<entry*> revisions;
			for (auto& entry : m_entries)
			{
				guest_surface_byte_cursor cursor(entry.coverage);
				bool changed = false;
				std::size_t runs = 0;
				while (!changed)
				{
					const auto active = cursor.next(); if (!active) break;
					if (++runs > m_run_limit) throw std::length_error("Physical publication exceeds virtual watch run budget");
					for (const auto& backing : backings) if (view.overlaps_backing(*backing, *active, m_run_limit)) { changed = true; break; }
				}
				if (changed) revisions.push_back(&entry);
			}
			// Every receipt must identify a watch this exact physical store updates.
			for (auto* watch : virtual_watches)
				same_virtual &= std::find(revisions.begin(), revisions.end(), watch) != revisions.end();
			std::vector<std::shared_ptr<physical_surface_state>> physical;
			for (const auto& weak : m_physical)
				if (auto cell = weak.lock())
					for (const auto& backing : backings) if (cell->identity.overlaps(*backing)) { physical.push_back(std::move(cell)); break; }
			auto aliases = view.physical_write_aliases(backings, m_run_limit);
			if (aliases.size() > m_run_limit) throw std::length_error("Physical publication exceeds reservation range budget");
			for (const auto range : aliases) validate_guest_surface_range(range);
			std::sort(aliases.begin(), aliases.end(), [](auto a, auto b) { return a.address < b.address; });
			std::size_t used = 0;
			for (const auto range : aliases)
			{
				if (!range.size) continue;
				if (used && std::uint64_t(range.address) <= std::uint64_t(aliases[used - 1].address) + aliases[used - 1].size)
				{
					auto& last = aliases[used - 1]; last.size = std::max(std::uint64_t(last.address) + last.size, std::uint64_t(range.address) + range.size) - last.address;
				}
				else aliases[used++] = range;
			}
			// No mapping lookup,allocation,validation or throw after this tick.
			const auto revision = tick();
			for (const auto& patch : patches) std::memcpy(patch.destination, payload.data() + patch.source_offset, patch.size);
			for (auto* entry : revisions) entry->revision = revision;
			for (const auto& cell : physical) cell->revision = revision;
			for (std::size_t i = 0; i < used; ++i) view.written(aliases[i]);
			result.revision = revision;
			if (same_virtual)
				for (std::size_t i = 0; i < virtual_watches.size(); ++i) result.virtual_revisions[i] = virtual_watches[i]->revision;
			else result.virtual_revisions.clear();
			published = true;
		});
		if (!published) return {};
		return result;
	}
	// Preflight mapping resources before entry. Commit must be noexcept and
	// return true only after changing this virtual range;no recursive adapter
	// calls from mutation. Mapping and revisions share one exclusion epoch.
	// Unchanged physical mirrors outside the virtual range keep their tags.
	template <typename Mutation>
	bool mutate_mapping(guest_surface_range range, Mutation&& mutation)
	{
		static_assert(std::is_nothrow_invocable_r_v<bool, Mutation&, exclusive_guest_memory::view&>);
		validate_guest_surface_range(range);
		if (!range.size)
			throw std::invalid_argument("Mapping transaction requires a nonempty range");
		bool changed = false;
		m_access->execute([&](exclusive_guest_memory::view& view)
						  {
			prune(view);
			std::vector<entry*> revisions;
			const auto virtual_range = guest_byte_coverage(range);
			for (auto& entry : m_entries)
				if (overlaps(entry.coverage, virtual_range)) revisions.push_back(&entry);
			const auto revision = tick();
			changed = std::invoke(mutation, view);
			if (changed) for (auto* entry : revisions) entry->revision = revision; });
		return changed;
	}
	// Called by mapping producers under their own exclusion,including the real
	// VM writer lock. Never enters the provider or touches registration storage.
	void notify_mapping_changed(guest_surface_range range) noexcept { m_mapping->notify(range); }
	// Call only with the corresponding host-writer ticket active. Reserve
	// before the actual store,commit after it;no provider/VM lock is entered.
	// Unleased entry points exercise metadata only,never the native GS path.
	[[nodiscard]] host_write_journal::intent begin_unleased_host_write(guest_surface_range range) { return m_host_writes.begin_unleased(range); }
	[[nodiscard]] host_write_journal::preparation prepare_host_access(guest_surface_range range, bool write)
	{
		validate_guest_surface_range(range);
		if (!range.size || range.size > m_byte_limit) throw std::length_error("Host access exceeds the guest capture byte budget");
		host_write_journal::preparation result{range, 0, {}};
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			(void)checked_map(view, range, write);
			result.backing = view.retain_segments(range);
			if (!result.backing || result.backing->range().address != range.address ||
				result.backing->range().size != range.size || result.backing->segments().size() > m_run_limit)
				throw std::runtime_error("Host access provider did not retain exact bounded physical slices");
			result.mapping = m_mapping->current(range);
		});
		return result;
	}
	[[nodiscard]] host_write_journal::intent begin_host_write(host_write_journal::preparation destination,
		std::optional<host_write_journal::preparation> source = {})
	{
		return m_host_writes.begin(std::move(destination), std::move(source));
	}
	[[nodiscard]] host_write_journal::intent begin_host_read(host_write_journal::preparation source)
	{
		m_host_writes.validate_prepared_read(source);
		return host_write_intent::keep_source(std::move(source.backing));
	}
	// Caller already owns mapping exclusion. This is metadata validation only.
	void validate_host_access(const host_write_journal::preparation& value) const { m_host_writes.validate_prepared_read(value); }
	void mark_host_write_fault(const host_write_intent& intent) { m_host_writes.mark_partial(intent); }
	template <typename Operation>
	void with_host_write_recovery(Operation&& operation)
	{
		m_host_writes.with_incomplete_recovery(std::forward<Operation>(operation));
	}
	[[nodiscard]] std::size_t pending_host_writes() const { return m_host_writes.pending(); }
	// Actual VM producers pair these around the complete backing mutation.
	// Completion observers reject live transitions;page watches remain precise.
	void begin_mapping_change(guest_surface_range range) noexcept { m_mapping->begin_change(range); }
	void end_mapping_change(guest_surface_range range) noexcept { m_mapping->end_change(range); }
	[[nodiscard]] auto observe_mapping() const noexcept { return m_mapping->observe(); }
	[[nodiscard]] bool mapping_unchanged(const mapping_epoch::observation& value) const noexcept { return m_mapping->unchanged(value); }
	[[nodiscard]] std::uint64_t mapping_generation() const { return m_mapping->current(); }
	// Same provider exclusion as capture/publication;this is a page revision,
	// not a backing lease or a concurrently readable mapping snapshot.
	[[nodiscard]] std::uint64_t mapping_generation(guest_surface_range range) const { return m_mapping->current(range); }
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override
	{
		validate(coverage, true);
		if (coverage.registration)
		{
			m_access->execute([&](exclusive_guest_memory::view& view)
							  { (void)track(view, coverage); });
			return coverage;
		}
		auto result = coverage;
		const auto proposed = std::make_shared<std::byte>();
		m_access->execute([&](exclusive_guest_memory::view& view)
						  {
			prune(view);
			if (auto found = find(view, coverage); found != m_entries.end())
			{
				auto& entry = *found;
				result.registration = entry.owner.lock();
				if (!result.registration) { result.registration = proposed; entry.owner = proposed; }
				protect_watch(view, coverage, result.registration);
				return;
			}
			if (m_entries.size() == m_capacity) prune(view, true);
			if (m_entries.size() == m_capacity) throw std::length_error("Guest memory coverage-registration budget exhausted");
			const auto revision = tick();
			protect_watch(view, coverage, proposed);
			insert({coverage, revision, false, proposed, m_mapping->current(coverage.range)});
			result.registration = proposed; });
		return result;
	}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		validate_guest_surface_range(range);
		std::optional<std::vector<storage_identity_span>> result;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view); result = view.storage_identity(range);
		});
		return result;
	}
	bool test_rebase(const rebase_test& test) override
	{
		bool result = false;
		const auto evaluate = [&](exclusive_guest_memory::view& view)
		{
			prune(view);
			const storage_rebase translate = [&](guest_surface_range source, guest_surface_range request)
			{ return view.rebase_storage(source, request); };
			const storage_equivalent equal = [&](guest_surface_range a, guest_surface_range b)
			{ return view.equivalent_storage(a, b); };
			result = test(translate, equal);
		};
		m_access->execute(evaluate);
		return result;
	}
	std::optional<guest_surface_range> rebase_storage(guest_surface_range source, guest_surface_range request) override
	{
		std::optional<guest_surface_range> result;
		m_access->execute([&](exclusive_guest_memory::view& view)
						  { prune(view); result = view.rebase_storage(source, request); });
		return result;
	}
	bool test_storage(const storage_test& test) override
	{
		bool result = false;
		const auto evaluate = [&](exclusive_guest_memory::view& view)
		{
			prune(view);
			const storage_lookup aliases = [&](guest_surface_range a, guest_surface_range b)
			{ return view.same_storage(a, b); };
			result = test(false, aliases);
		};
		m_access->execute(evaluate);
		return result;
	}
	guarded_guest_memory(std::shared_ptr<exclusive_guest_memory> access, std::size_t capacity = 1024,
						 std::size_t byte_limit = 64 << 20, std::size_t run_limit = 1 << 20, std::size_t host_write_limit = 1024)
		: m_access(std::move(access)), m_capacity(capacity), m_byte_limit(byte_limit), m_run_limit(run_limit), m_host_writes(m_mapping, host_write_limit)
	{
		if (!m_access || !capacity || !byte_limit || !run_limit)
			throw std::invalid_argument("Guarded guest memory requires an exclusive provider and finite budgets");
	}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override
	{
		if (ranges.size() > m_run_limit) throw std::length_error("Guest backing request exceeds range budget");
		std::size_t bytes = 0;
		for (const auto range : ranges)
		{
			validate_guest_surface_range(range);
			if (range.size > m_byte_limit - bytes) throw std::length_error("Guest backing request exceeds operation budget");
			bytes += range.size;
		}
		if (!bytes) return {};
		std::shared_ptr<void> owner;
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			for (const auto range : ranges)
				if (range.size)
					(void)checked_map(view, range, false);
			owner = view.retain_backing(ranges);
			if (!owner) throw std::runtime_error("Guest provider returned an absent backing lease");
		});
		return owner;
	}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		validate(coverage);
		guest_surface_snapshot result{0, std::vector<std::byte>(coverage.range.size)};
		m_access->execute([&](exclusive_guest_memory::view& view)
						  {
			prune(view);
			if (coverage.range.size)
				std::memcpy(result.bytes.data(), checked_map(view, coverage.range, false), coverage.range.size);
			result.revision = track(view, coverage).revision; });
		return result;
	}
	guest_surface_snapshot read_surface_packed(const guest_surface_layout& coverage) override
	{
		validate(coverage, true);
		guest_surface_snapshot result{0, std::vector<std::byte>(guest_surface_active_bytes(coverage))};
		m_access->execute([&](exclusive_guest_memory::view& view)
		{
			prune(view);
			guest_surface_byte_cursor cursor(coverage);
			std::size_t offset = 0, runs = 0;
			while (const auto active = cursor.next())
			{
				if (++runs > m_run_limit) throw std::length_error("Packed guest capture exceeds active-byte run budget");
				std::memcpy(result.bytes.data() + offset, checked_map(view, *active, false), active->size);
				offset += active->size;
			}
			result.revision = track(view, coverage).revision;
		});
		return result;
	}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override
	{
		validate(coverage, true);
		std::uint64_t result = 0;
		m_access->execute([&](exclusive_guest_memory::view& view)
						  {
			prune(view);
			guest_surface_byte_cursor cursor(coverage);
			while (const auto active = cursor.next())
			{
				(void)checked_map(view, *active, false);
			}
			result = track(view, coverage).revision; });
		return result;
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage, std::uint64_t expected,
											   std::span<const std::byte> bytes) override
	{
		const guest_surface_write write{coverage, expected, bytes};
		return compare_write_many({&write, 1});
	}
	std::optional<std::uint64_t> compare_write_many(std::span<const guest_surface_write> writes) override
	{
		if (writes.empty())
			return 0;
		std::size_t total = 0;
		const bool ordered = writes.front().logical_offset.has_value();
		if (writes.size() > m_run_limit)
			throw std::length_error("Guest memory batch exceeds patch budget");
		for (const auto& write : writes)
		{
			if (write.logical_offset.has_value() != ordered) throw std::invalid_argument("Guest publication mixes ordered and unordered parts");
			if (ordered && write.coverage.range.size > std::numeric_limits<std::size_t>::max() - *write.logical_offset)
				throw std::overflow_error("Guest publication logical extent wraps");
			validate(write.coverage, write.packed);
			if (write.bytes.size() != (write.packed ? guest_surface_active_bytes(write.coverage) : write.coverage.range.size))
				throw std::invalid_argument("Guest memory compare-write payload differs from storage");
			if (write.bytes.size() > m_byte_limit - total)
				throw std::length_error("Guest memory batch exceeds aggregate byte budget");
			total += write.bytes.size();
		}
		std::optional<std::uint64_t> result;
		m_access->execute([&](exclusive_guest_memory::view& view)
						  {
			for (const auto& write : writes)
				if (track(view, write.coverage).revision != write.expected) return;
			// Freeze the payload while VM writers are excluded. It may alias
			// mapped guest storage;later patches must never read modified bytes.
			std::vector<std::byte> payload;
			payload.reserve(total);
			for (const auto& write : writes) payload.insert(payload.end(), write.bytes.begin(), write.bytes.end());
			std::vector<patch> patches;
			std::vector<entry*> revisions;
			std::vector<std::shared_ptr<physical_surface_state>> physical;
			std::size_t payload_offset = 0;
			for (const auto& write : writes)
			{
				guest_surface_byte_cursor cursor(write.coverage);
				std::size_t packed_offset = 0;
				while (const auto active = cursor.next())
				{
					if (patches.size() == m_run_limit) throw std::length_error("Guest memory compare-write exceeds patch budget");
					const auto offset = write.packed ? packed_offset : active->address - write.coverage.range.address;
					patches.push_back({checked_map(view, *active, true), payload_offset + offset, active->size, *active,
						ordered ? *write.logical_offset + active->address - write.coverage.range.address : 0});
					packed_offset += active->size;
					for (auto* entry : affected(view, *active))
						if (std::find(revisions.begin(), revisions.end(), entry) == revisions.end()) revisions.push_back(entry);
					for (auto& cell : affected_physical(view, *active))
						if (std::find(physical.begin(), physical.end(), cell) == physical.end()) physical.push_back(std::move(cell));
				}
				payload_offset += write.bytes.size();
			}
			if (ordered)
			{
				std::sort(patches.begin(), patches.end(), [](const auto& a, const auto& b) { return a.logical_offset < b.logical_offset; });
				for (std::size_t i = 1; i < patches.size(); ++i)
					if (patches[i - 1].logical_offset + patches[i - 1].size > patches[i].logical_offset)
						throw std::invalid_argument("Guest publication logical active intervals overlap");
			}
			if (patches.empty())
			{
				result = writes[0].expected;
				return;
			}
			const auto revision = tick();
			for (const auto& patch : patches)
				std::memcpy(patch.destination, payload.data() + patch.source_offset, patch.size);
			for (auto* entry : revisions)
				entry->revision = revision;
			for (const auto& cell : physical) cell->revision = revision;
			for (const auto& patch : patches)
				view.written(patch.range);
			result = revision; });
		return result;
	}
	void write(guest_surface_range range, std::span<const std::byte> bytes) override
	{
		validate_guest_surface_range(range);
		if (range.size != bytes.size() || range.size > m_byte_limit)
			throw std::invalid_argument("Guest memory write payload/budget differs from range");
		if (!range.size)
			return;
		m_access->execute([&](exclusive_guest_memory::view& view)
						  {
			prune(view);
			const std::vector<std::byte> payload(bytes.begin(), bytes.end());
			const auto destination = checked_map(view, range, true);
			const auto revisions = affected(view, range);
			const auto physical = affected_physical(view, range);
			const auto revision = tick();
			std::memcpy(destination, payload.data(), payload.size());
			for (auto* entry : revisions)
				entry->revision = revision;
			for (const auto& cell : physical) cell->revision = revision;
			view.written(range); });
	}
	void before_write_in_transaction(exclusive_guest_memory::view& view, guest_surface_range range)
	{
		if (guest_memory_epoch::current(*m_access) != &view)
			throw std::logic_error("External guest write does not own this provider's active view");
		prune(view);
		validate_guest_surface_range(range);
		if (!range.size)
			return;
		const auto revisions = affected(view, range);
		const auto physical = affected_physical(view, range);
		const auto revision = tick();
		for (auto* entry : revisions)
			entry->revision = revision;
		for (const auto& cell : physical) cell->revision = revision;
	}
};
} // namespace metal4
