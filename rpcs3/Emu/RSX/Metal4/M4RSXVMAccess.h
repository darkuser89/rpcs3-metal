#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "The native Metal4 VM bridge requires Apple Silicon."
#endif

#include "Emu/CPU/CPUThread.h"
#include "Emu/Memory/vm.h"
#include "Emu/Memory/vm_locking.h"
#include "Emu/Memory/vm_reservation.h"
#include "Emu/RSX/RSXThread.h"
#include "M4GuardedGuestMemory.h"
#include "M4BackingPool.h"
#include "M4IndependentBacking.h"
#include "M4GuestMemoryAliases.h"
#include "M4ReportAtomic.h"
#include "M4ARM64Memory.h"
#include "M4RSXPageProtection.h"

#include <exception>
#include <thread>
#include "util/vm.hpp"
#include <sys/mman.h>

namespace metal4
{
// Actual VM mapping and short CPU-suspended callback. This does not install
// PPU/SPU/fault/DMA revision hooks. A renderer may activate it only after
// routing all writes through the guarded revision contract.
class rsx_vm_access final : public exclusive_guest_memory
{
	friend class rsx_cache_fault;
	// Private to the continuous render-owner service. Independent byte owners
	// and weak original allocation identities never retain VM shm across GPU work.
	struct instruction_plan
	{
		arm64_memory_request request;
		guest_surface_range range;
		std::uint64_t generation = 0, mapping = 0;
		std::vector<std::pair<std::uint32_t, std::vector<rsx_page_protection::logical_page>>> pages;
		std::vector<std::shared_ptr<const rsx_page_protection::fault_plan>> faults;
		std::shared_ptr<const physical_backing> backing;
		std::optional<physical_identity> identity;
		std::vector<std::shared_ptr<void>> registrations;
		instruction_plan(const arm64_memory_request& value, guest_surface_range bytes) : request(value), range(bytes) {}
	};
	// Must drain/exclude non-CPU DMA/host writers for this entire operation.
	// No permissive default:an absent guard fails before guest memory access.
	std::function<std::shared_ptr<void>()> m_acquire_host_guard;
	backing_pool m_backings;
	const std::shared_ptr<rsx_page_protection> m_protection;
	class mapped_view final : public view
	{
		rsx_vm_access& m_owner;
		const vm::writer_lock& m_lock;
	  public:
		mapped_view(rsx_vm_access& owner, const vm::writer_lock& lock) : m_owner(owner), m_lock(lock) {}
		bool uses_cpu_protection() const noexcept override { return bool(m_owner.m_protection); }
		void watch_ranges(std::span<const guest_surface_range> ranges, const std::shared_ptr<void>& owner) override
		{
			if (m_owner.m_protection) m_owner.m_protection->protect_registration(ranges, owner, m_lock);
		}
		std::shared_ptr<const rsx_page_protection::fault_plan> capture_fault(std::uint32_t address, std::size_t byte_limit)
		{
			if (!m_owner.m_protection) throw std::logic_error("Native fault plan requires the installed protection owner");
			if (!byte_limit || byte_limit > (64u << 20)) throw std::invalid_argument("Invalid native fault receipt byte budget");
			auto plan = m_owner.m_protection->plan_fault(address, m_lock);
			if (!plan) return {};
			std::size_t total = 0;
			// Preflight the whole closure before retaining any independent mapping.
			for (const auto& w : plan->m_watches)
			{
				if (w.size > byte_limit - total) throw std::length_error("Native fault receipts exceed aggregate byte budget");
				total += w.size;
			}
			std::vector<physical_identity::segment> identity;
			for (const auto& w : plan->m_watches)
			{
				const auto source = w.memory.lock();
				if (!source || w.offset > source->size() || w.size > source->size() - w.offset)
					throw std::runtime_error("Native fault receipt lost its original physical allocation");
				auto mapping = m_owner.m_backings.acquire(source, independent_backing_charge(source->size()), [source]
				{
					return map_independent_backing(source->size(), [&](void* reserved) { return source->map(reserved); });
				});
				identity.push_back({mapping, static_cast<std::size_t>(w.offset), static_cast<std::size_t>(w.size)});
				plan->m_slices.push_back({std::move(mapping), static_cast<std::size_t>(w.offset), static_cast<std::size_t>(w.size)});
			}
			plan->m_identity.emplace(std::move(identity));
			return plan;
		}
		bool fault_current(const rsx_page_protection::fault_plan& plan) const
		{
			return m_owner.m_protection && m_owner.m_protection->fault_current(plan, m_lock);
		}
		bool overlaps_backing(const physical_backing& written, guest_surface_range current, std::size_t limit) override
		{
			validate_guest_surface_range(current);
			std::size_t runs = 0;
			for (std::uint64_t cursor = current.address, end = cursor + current.size; cursor < end;)
			{
				if (++runs > limit) throw std::length_error("Real VM physical reconciliation exceeds page budget");
				// Allocated-only metadata includes protected current aliases.
				// Bound each query to one page so an absent neighbour cannot hide
				// a live physical slice. Never map/read/write guest bytes here.
				const auto count = static_cast<u32>(std::min(end - cursor, 4096 - (cursor & 4095)));
				const auto part = vm::get_backing_segment(static_cast<u32>(cursor), count, m_lock, vm::page_allocated);
				if (!part.memory || !part.size)
				{
					if (vm::check_addr(static_cast<u32>(cursor), vm::page_allocated, count))
						throw std::runtime_error("Physical reconciliation found an inaccessible or unresolved allocated VM page");
					cursor += count; // Genuinely unmapped pages have no current alias.
					continue;
				}
				for (const auto& slice : written.segments())
					if (m_owner.m_backings.matches(part.memory, slice.owner) &&
						slice.offset < part.offset + part.size && part.offset < slice.offset + slice.guest.size) return true;
				cursor += part.size;
			}
			return false;
		}
		bool overlaps_identity(const physical_identity& kept, guest_surface_range current, std::size_t limit) override
		{
			validate_guest_surface_range(current);
			std::size_t runs = 0;
			for (std::uint64_t cursor = current.address, end = cursor + current.size; cursor < end;)
			{
				if (++runs > limit) throw std::length_error("Physical revision query exceeds VM page budget");
				const auto count = static_cast<u32>(std::min(end - cursor, 4096 - (cursor & 4095)));
				const auto part = vm::get_backing_segment(static_cast<u32>(cursor), count, m_lock, vm::page_allocated);
				if (!part.memory || !part.size)
				{
					if (vm::check_addr(cursor, vm::page_allocated, count)) throw std::runtime_error("Physical revision query cannot identify allocated VM bytes");
					cursor += count; continue;
				}
				for (const auto& slice : kept.segments())
					if (m_owner.m_backings.matches(part.memory, slice.owner) &&
						slice.offset < part.offset + part.size && part.offset < slice.offset + slice.size) return true;
				cursor += part.size;
			}
			return false;
		}
		std::vector<guest_surface_range> physical_write_aliases(
			std::span<const std::shared_ptr<const physical_backing>> written, std::size_t limit) override
		{
			std::vector<guest_surface_range> aliases;
			if (m_owner.m_protection)
			{
				for (const auto& backing : written)
					for (const auto& slice : backing->segments())
						m_owner.m_protection->physical_aliases(m_owner.m_backings.source_identity(slice.owner),
							slice.offset, slice.guest.size, m_lock, [&](std::uint32_t address, std::uint64_t size)
							{
								if (aliases.size() == limit) throw std::length_error("Physical reservation aliases exceed range budget");
								aliases.push_back({address, static_cast<std::size_t>(size)});
							});
				return aliases;
			}
			std::size_t pages = 0;
			// A provider without the continuous protection/mapping owner has no
			// trusted index. Keep the bounded exhaustive reconciliation for it.
			for (u64 address = 0; address < 0x1'0000'0000ull; address += 4096)
			{
				if (++pages > limit) throw std::length_error("Physical reservation alias scan exceeds VM page budget");
				if (!vm::check_addr(address, vm::page_allocated, 4096)) continue;
				const auto part = vm::get_backing_segment(static_cast<u32>(address), 4096, m_lock, vm::page_allocated);
				if (!part.memory || part.size != 4096) throw std::runtime_error("Physical reservation alias scan found unresolved allocated bytes");
				for (const auto& backing : written)
					for (const auto& slice : backing->segments())
						if (m_owner.m_backings.matches(part.memory, slice.owner))
						{
							const auto first = std::max<u64>(part.offset, slice.offset);
							const auto end = std::min<u64>(part.offset + part.size, slice.offset + slice.guest.size);
							if (first >= end) continue;
							if (aliases.size() == limit) throw std::length_error("Physical reservation aliases exceed range budget");
							aliases.push_back({static_cast<u32>(address + first - part.offset), static_cast<std::size_t>(end - first)});
						}
			}
			return aliases;
		}
		static void require_permissions(guest_surface_range range, u8 flags)
		{
			// The VM's whole-range fast path skips to a64KiB/1MiB boundary
			// based on the first page's allocation-size bit. Logical4KiB
			// protections may differ within that allocation:check EACH page.
			for (std::uint64_t cursor = range.address, end = cursor + range.size; cursor < end;)
			{
				const auto count = static_cast<u32>(std::min(end - cursor, 4096 - (cursor & 4095)));
				if (!vm::check_addr(cursor, flags, count))
					throw std::runtime_error("RSX VM operation references inaccessible guest pages");
				cursor += count;
			}
		}
		std::shared_ptr<const physical_backing> retain_segments(guest_surface_range range) override
		{
			return retain_segments_with_flags(range, vm::page_readable);
		}
		std::shared_ptr<const physical_backing> retain_segments_with_flags(guest_surface_range range, u8 flags)
		{
			validate_guest_surface_range(range);
			if (!range.size || range.size > std::numeric_limits<u32>::max())
				throw std::invalid_argument("Real VM physical slices require a nonempty32bit range");
			require_permissions(range, flags);
			std::vector<physical_backing::segment> segments;
			for (std::uint64_t cursor = range.address, end = cursor + range.size; cursor < end;)
			{
				const auto part = vm::get_backing_segment(static_cast<u32>(cursor), static_cast<u32>(end - cursor), m_lock, flags);
				if (!part.memory || !part.size) throw std::runtime_error("Guest backing capture crosses an absent physical allocation");
				auto owner = m_owner.m_backings.acquire(part.memory, independent_backing_charge(part.memory->size()), [memory = part.memory]
				{
					// Kernel mappings retain bytes after the original shm closes.
					// Never keep strong shm refs across VM block teardown.
					return map_independent_backing(memory->size(), [&](void* reserved) { return memory->map(reserved); });
				});
				segments.push_back({{part.address, part.size}, std::move(owner), part.offset, part.memory->size()});
				cursor += part.size;
			}
			return std::make_shared<physical_backing>(range, std::move(segments));
		}
		std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override
		{
			auto owners = std::make_shared<std::vector<std::shared_ptr<const physical_backing>>>();
			for (const auto range : ranges)
			{
				validate_guest_surface_range(range);
				if (range.size) owners->push_back(retain_segments(range));
			}
			if (owners->empty()) throw std::invalid_argument("Real VM backing lease requires nonempty ranges");
			return owners;
		}

		std::shared_ptr<instruction_plan> capture_instruction(const arm64_memory_request& request,
			std::uint64_t fault, bool writing, guarded_guest_memory& revisions)
		{
			if (!m_owner.m_protection || !revisions.uses_provider(m_owner))
				throw std::logic_error("Native instruction requires its live protected VM provider");
			m_owner.m_protection->require_lock(m_lock);
			const auto base = reinterpret_cast<std::uint64_t>(vm::g_base_addr);
			// Normal RAM only. Raw-SPU/MMIO requires its device semantics rather
			// than a canonical-memory copy. The full instruction must fit.
			if (!request.matches_fault(fault, writing) || request.address() < base ||
				request.address() - base >= 0xe0000000ull || request.size() > 0xe0000000ull - (request.address() - base))
				throw std::invalid_argument("Native ARM instruction is not a matching complete normal VM RAM access");
			const guest_surface_range range{static_cast<std::uint32_t>(request.address() - base), request.size()};
			(void)map(range, writing); // Check all logical permissions BEFORE any GPU publication.
			auto result = std::make_shared<instruction_plan>(request, range);
			result->generation = m_owner.m_protection->m_generation;
			result->mapping = revisions.mapping_generation(range);
			result->backing = retain_segments_with_flags(range, vm::page_allocated);
			std::vector<physical_identity::segment> identity;
			std::size_t total = 0;
			const auto append_identity = [&](physical_identity::segment slice)
			{
				if (std::any_of(identity.begin(), identity.end(), [&](const auto& old)
				{ return old.offset == slice.offset && old.size == slice.size && !old.owner.owner_before(slice.owner) && !slice.owner.owner_before(old.owner); })) return;
				if (slice.size > (64u << 20) - total) throw std::length_error("Native instruction closure exceeds aggregate byte budget");
				total += slice.size; identity.push_back(std::move(slice));
			};
			for (const auto& slice : result->backing->segments()) append_identity({slice.owner, slice.offset, slice.guest.size});
			const auto page = m_owner.m_protection->page_size();
			for (std::uint64_t first = std::uint64_t(range.address) & ~(std::uint64_t(page) - 1), end = std::uint64_t(range.address) + range.size;
				first < end; first += page)
			{
				const auto address = static_cast<std::uint32_t>(first);
				result->pages.emplace_back(address, m_owner.m_protection->fault_pages(address, m_lock));
				if (auto plan = capture_fault(address, 64u << 20))
				{
					for (const auto& slice : plan->identity().segments()) append_identity(slice);
					for (const auto& owner : plan->registrations())
						if (std::none_of(result->registrations.begin(), result->registrations.end(), [&](const auto& old)
						{ return !old.owner_before(owner) && !owner.owner_before(old); })) result->registrations.push_back(owner);
					result->faults.push_back(std::move(plan));
				}
			}
			result->identity.emplace(std::move(identity));
			return result;
		}
		bool instruction_current(const instruction_plan& plan, const guarded_guest_memory& revisions) const
		{
			m_owner.m_protection->require_lock(m_lock);
			if (plan.generation != m_owner.m_protection->m_generation || plan.mapping != revisions.mapping_generation(plan.range)) return false;
			for (const auto& [address, expected] : plan.pages)
			{
				const auto current = m_owner.m_protection->fault_pages(address, m_lock);
				if (current.size() != expected.size()) return false;
				for (std::size_t i = 0; i < current.size(); ++i)
					if (current[i].flags != expected[i].flags || current[i].offset != expected[i].offset ||
						current[i].memory.owner_before(expected[i].memory) || expected[i].memory.owner_before(current[i].memory)) return false;
			}
			for (const auto& fault : plan.faults) if (!fault_current(*fault)) return false;
			return true;
		}
		std::optional<arm64_memory_state> complete_instruction(const instruction_plan& plan, guarded_guest_memory& revisions)
		{
			if (!instruction_current(plan, revisions)) return {};
			auto* bytes = map(plan.range, plan.request.writing());
			if (!plan.request.writing())
			{
				std::array<std::byte, 32> loaded{};
				std::memcpy(loaded.data(), bytes, plan.range.size);
				return plan.request.complete({loaded.data(), plan.range.size});
			}
			const auto payload = plan.request.store_bytes();
			const auto reply = plan.request.complete({});
			const std::array backings{plan.backing};
			const auto aliases = physical_write_aliases(backings, 1u << 20);
			// One preflighted revision commit covers both exact instruction bytes
			// and all cache-registration closures. All allocations/validation and
			// reply computation precede the byte store;no fallible work follows it.
			revisions.invalidate_physical_in_transaction(*this, *plan.identity, plan.registrations);
			std::memcpy(bytes, payload.data(), plan.range.size);
			for (const auto alias : aliases) written(alias);
			return reply;
		}

		// g_shmem is reservation-lock canonicalization,not an exhaustive map of
		// allocation aliases. Resolve exact owners/offsets from live VM metadata.
		// Strong handles live only within this writer-locked view operation.
		using storage_segments = std::vector<vm::backing_segment>;
		static constexpr std::size_t storage_budget = 2u << 20;
		storage_segments storage(guest_surface_range range, u8 flags, std::size_t& budget)
		{
			validate_guest_surface_range(range);
			storage_segments result;
			for (u64 address = range.address, end = address + range.size; address < end;)
			{
				if (!budget) throw std::length_error("Real VM alias query exceeds metadata budget");
				--budget;
				const auto size = static_cast<u32>(std::min(end - address, 4096 - (address & 4095)));
				const auto part = vm::get_backing_segment(static_cast<u32>(address), size, m_lock, flags);
				if (!part.memory || part.size != size)
				{
					if (vm::check_addr(address, flags, size)) throw std::runtime_error("Real VM alias query cannot identify allocated bytes");
					if (flags != vm::page_allocated) return {}; // Full readable coverage required.
					address += size; continue;
				}
				if (!result.empty() && result.back().memory == part.memory &&
					u64{result.back().address} + result.back().size == address &&
					result.back().offset + result.back().size == part.offset &&
					part.size <= std::numeric_limits<u32>::max() - result.back().size)
					result.back().size += part.size;
				else result.push_back(part);
				address += size;
			}
			return result;
		}
		bool equivalent(guest_surface_range a, guest_surface_range b, std::size_t& budget)
		{
			validate_guest_surface_range(a); validate_guest_surface_range(b);
			if (a.size != b.size) return false;
			if (!a.size) return true;
			const auto from = storage(a, vm::page_readable, budget), to = storage(b, vm::page_readable, budget);
			if (from.empty() || to.empty()) return false;
			std::size_t ai = 0, bi = 0; u64 ao = 0, bo = 0;
			while (ai < from.size() && bi < to.size())
			{
				if (from[ai].memory != to[bi].memory || from[ai].offset + ao != to[bi].offset + bo) return false;
				const auto count = std::min(from[ai].size - ao, to[bi].size - bo);
				ao += count; bo += count;
				if (ao == from[ai].size) { ++ai; ao = 0; }
				if (bo == to[bi].size) { ++bi; bo = 0; }
			}
			return ai == from.size() && bi == to.size();
		}
		bool equivalent_storage(guest_surface_range a, guest_surface_range b) override
		{
			auto budget = storage_budget;
			return equivalent(a, b, budget);
		}
		std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
		{
			validate_guest_surface_range(range);
			if (!range.size) throw std::invalid_argument("Storage identity requires nonempty current bytes");
			auto budget = storage_budget;
			const auto parts = storage(range, vm::page_allocated, budget);
			std::vector<storage_identity_span> result; result.reserve(parts.size());
			for (const auto& part : parts)
			{
				std::shared_ptr<void> allocation = part.memory;
				result.push_back({allocation,static_cast<std::size_t>(part.offset),
					{static_cast<u32>(part.address),static_cast<std::size_t>(part.size)}});
			}
			return result;
		}
		std::optional<guest_surface_range> rebase_storage(guest_surface_range source, guest_surface_range request) override
		{
			validate_guest_surface_range(source); validate_guest_surface_range(request);
			if (!request.size || request.size > source.size) return {};
			auto budget = storage_budget;
			const auto target = storage(request, vm::page_readable, budget);
			if (target.empty()) return {};
			const auto origins = storage(source, vm::page_allocated, budget);
			std::optional<guest_surface_range> result;
			for (const auto& part : origins)
			{
				if (part.memory != target.front().memory || target.front().offset < part.offset ||
					target.front().offset - part.offset >= part.size) continue;
				const u64 candidate = part.address + target.front().offset - part.offset;
				if (request.size > u64{source.address} + source.size - candidate) continue;
				const guest_surface_range translated{static_cast<u32>(candidate), request.size};
				if (!equivalent(translated, request, budget)) continue;
				if (result) return {}; // Duplicate mirrors cannot name one source interval.
				result = translated;
			}
			return result;
		}
		bool same_storage(guest_surface_range a, guest_surface_range b) override
		{
			auto budget = storage_budget;
			auto from = storage(a, vm::page_allocated, budget), to = storage(b, vm::page_allocated, budget);
			const auto order = [](const auto& first, const auto& second)
			{
				if (first.memory != second.memory) return std::less<utils::shm*>{}(first.memory.get(), second.memory.get());
				return first.offset < second.offset;
			};
			std::sort(from.begin(), from.end(), order); std::sort(to.begin(), to.end(), order);
			std::size_t ai = 0, bi = 0;
			while (ai < from.size() && bi < to.size())
			{
				if (from[ai].memory == to[bi].memory && from[ai].offset < to[bi].offset + to[bi].size &&
					to[bi].offset < from[ai].offset + from[ai].size) return true;
				if (order(from[ai], to[bi])) ++ai; else ++bi;
			}
			return false;
		}
		std::byte* map(guest_surface_range range, bool write) override
		{
			validate_guest_surface_range(range);
			if (!range.size || range.size > std::numeric_limits<u32>::max())
				throw std::invalid_argument("RSX VM operation range must fit the real32bit access interface");
			require_permissions(range, write ? vm::page_writable : vm::page_readable);
			// Allocator/protection transitions are excluded by vm::writer_lock.
			// SurfaceCache must resolve CPU read/write coherence before entry.
			return reinterpret_cast<std::byte*>(vm::g_sudo_addr + range.address);
		}
		void written(guest_surface_range range) noexcept override
		{
			// Native copies participate in the real reservation notification
			// mechanism. Local no-reservation pages need no timestamp update.
			const auto end = std::uint64_t(range.address) + range.size;
			for (auto address = std::uint64_t(range.address & ~127u); address < end; address += 128)
				if (!(vm::get_addr_flags(static_cast<u32>(address)).second & vm::page_no_reservations))
					vm::reservation_update(static_cast<u32>(address));
		}
	};

	[[nodiscard]] std::shared_ptr<instruction_plan> capture_instruction(const arm64_memory_request& request,
		std::uint64_t fault, bool writing, guarded_guest_memory& revisions)
	{
		std::shared_ptr<instruction_plan> result;
		execute([&](view& memory) { result = static_cast<mapped_view&>(memory).capture_instruction(request, fault, writing, revisions); });
		return result;
	}
	[[nodiscard]] std::optional<arm64_memory_state> complete_instruction(const instruction_plan& plan, guarded_guest_memory& revisions)
	{
		std::optional<arm64_memory_state> result;
		execute([&](view& memory) { result = static_cast<mapped_view&>(memory).complete_instruction(plan, revisions); });
		return result;
	}

  public:
	explicit rsx_vm_access(std::function<std::shared_ptr<void>()> acquire_host_guard,
		std::size_t backing_bytes = 512u << 20, std::size_t backing_entries = 1024,
		std::shared_ptr<rsx_page_protection> protection = {})
		: m_acquire_host_guard(std::move(acquire_host_guard)), m_backings(backing_bytes, backing_entries), m_protection(std::move(protection))
	{
		if (!m_acquire_host_guard)
			throw std::invalid_argument("Real RSX VM access requires non-CPU writer exclusion");
	}
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* memory = guest_memory_epoch::current(*this))
		{
			operation(*memory);
			return;
		}
		if (const auto caller = cpu_thread::get_current();
			caller && (caller->get_class() == thread_class::ppu || caller->get_class() == thread_class::spu))
			throw std::logic_error("Real VM transactions must be dispatched to the host/RSX thread before CPU suspension");
		const auto guard = m_acquire_host_guard();
		if (!guard)
			throw std::runtime_error("Real RSX VM writer-exclusion guard is absent");
		std::exception_ptr failure;
		const auto caller_thread = std::this_thread::get_id();
		bool invoked = false;
		// No exception may escape suspend_work's noexcept dispatcher. Re-throw
		// only after CPU suspension has ended and mapping lock is released.
		auto suspended_operation = [&]() noexcept
		{
			try
			{
				if (invoked || std::this_thread::get_id() != caller_thread)
					throw std::logic_error("Real VM suspension changed capture thread or invoked twice");
				invoked = true;
				vm::writer_lock lock;
				if (m_protection) m_protection->collect_retired(lock);
				mapped_view memory(*this, lock);
				guest_memory_epoch epoch(*this, memory);
				operation(memory);
			}
			catch (...)
			{
				failure = std::current_exception();
			}
		};
		// push(nullptr) waits for an existing workload instead of joining it.
		// It becomes the first workload and executes on this host caller thread.
		// This is stronger than suspend_all(non-null CPU),which may delegate.
		using operation_type = decltype(suspended_operation);
		cpu_thread::suspend_work work{0, false, false, 0, nullptr, &suspended_operation, nullptr,
									  [](void* function, void*)
									  { std::invoke(*static_cast<operation_type*>(function)); }, nullptr};
		(void)work.push(nullptr);
		if (failure)
			std::rethrow_exception(failure);
		if (!invoked)
			throw std::logic_error("Real VM suspension omitted capture");
	}
	// Synchronous CPU-only renderer capture. Guarded memory calls within this
	// callback reuse the suspended transaction's mapping view on its thread.
	// GPU work,UI dispatch and waits are forbidden inside operation.
	void capture_cpu(const std::function<void()>& operation)
	{
		execute([&](view&)
				{ operation(); });
	}
	// Capture on the render owner,then perform GPU flushes outside execute().
	// Receipts retain independent bytes/registrations without keeping shm alive.
	// Revalidation is mandatory after the handoff;neither method opens pages.
	[[nodiscard]] std::shared_ptr<const rsx_page_protection::fault_plan> capture_fault(std::uint32_t address,
		std::size_t byte_limit = 64u << 20)
	{
		std::shared_ptr<const rsx_page_protection::fault_plan> result;
		execute([&](view& memory) { result = static_cast<mapped_view&>(memory).capture_fault(address, byte_limit); });
		return result;
	}
	[[nodiscard]] bool fault_current(const rsx_page_protection::fault_plan& plan)
	{
		bool result = false;
		execute([&](view& memory) { result = static_cast<mapped_view&>(memory).fault_current(plan); });
		return result;
	}
	// After successful native publication,retire both revision domains in one
	// suspended epoch. A stale plan refuses before this invalidation. Protection
	// stays armed:actual CPU instruction completion is a separate required step.
	[[nodiscard]] bool invalidate_fault(guarded_guest_memory& revisions, const rsx_page_protection::fault_plan& plan)
	{
		bool result = false;
		execute([&](view& memory)
		{
			if (!static_cast<mapped_view&>(memory).fault_current(plan)) return;
			revisions.invalidate_physical_in_transaction(memory, plan.identity(), plan.registrations());
			result = true;
		});
		return result;
	}
	void store_label(guarded_guest_memory& revisions, std::uint32_t address, std::uint32_t value)
	{
		if (address & 3u)
			throw std::invalid_argument("Guest semaphore store must be four-byte aligned");
		const guest_surface_range range{address, 4};
		execute([&](view& memory)
				{
			const auto pointer = memory.map(range, true);
			revisions.before_write_in_transaction(memory, range);
			const be_t<u32> encoded = value;
			reinterpret_cast<atomic_t<be_t<u32>>*>(pointer)->release(encoded);
			memory.written(range); });
	}
	void store_report(guarded_guest_memory& revisions, std::uint32_t address, std::uint64_t timestamp, std::uint32_t value)
	{
		static_assert(sizeof(CellGcmReportData) == 16 && alignof(CellGcmReportData) == 16);
		if (address & 15u)
			throw std::invalid_argument("Guest report store must be sixteen-byte aligned");
		const guest_surface_range range{address, sizeof(CellGcmReportData)};
		execute([&](view& memory)
				{
			const auto pointer = memory.map(range, true);
			revisions.before_write_in_transaction(memory, range);
			store_report_atomic(*reinterpret_cast<atomic_t<CellGcmReportData>*>(pointer), timestamp, value);
			memory.written(range); });
	}
};
[[nodiscard]] inline std::uint32_t translate_rsx_vm_range(std::uint32_t offset, unsigned location, std::size_t size)
{
	if (!size || size > std::numeric_limits<u32>::max())
		throw std::invalid_argument("RSX VM translation requires a bounded nonempty range");
	return rsx::get_address(offset, location, static_cast<u32>(size));
}
} // namespace metal4
