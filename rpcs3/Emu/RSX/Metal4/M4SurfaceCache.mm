#include "M4SurfaceCache.h"
#include "M4GuestMemoryOverlap.h"
#include "M4PhysicalSurface.h"
#include "M4ExclusiveGuestMemory.h"
#include "M4ScaledImageNativePlan.h"

#include <algorithm>
#include <list>
#include <unordered_map>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
namespace
{
// Every part owns a real VM watch. Image geometry is deliberately separate;
// bank permutation consumes the gathered flat allocation, never a fake EA.
struct surface_storage
{
	std::vector<guest_storage_part> parts;
	std::size_t size = 0;
};
using physical_tickets = std::vector<std::shared_ptr<const physical_surface>>;
struct surface_storage_snapshot
{
	std::vector<std::uint64_t> revisions;
	std::vector<std::byte> bytes;
};
bool active_storage_overlap(bool direct, const guest_surface_memory::storage_lookup& lookup,
	const surface_storage& a, const surface_storage& b)
{
	for (const auto& ap : a.parts)
		for (const auto& bp : b.parts)
			if (metal4::active_storage_overlap(direct, lookup, ap.coverage, bp.coverage)) return true;
	return false;
}
bool active_storage_overlap(bool direct, const guest_surface_memory::storage_lookup& lookup,
	const guest_surface_layout& a, const surface_storage& b)
{
	for (const auto& bp : b.parts)
		if (metal4::active_storage_overlap(direct, lookup, a, bp.coverage)) return true;
	return false;
}
template <typename A>
bool memory_storage_overlap(guest_surface_memory& memory, const A& a, const surface_storage& b)
{
	return memory.test_storage([&](bool direct, const auto& lookup)
		{ return active_storage_overlap(direct, lookup, a, b); });
}
bool equivalent_storage(guest_surface_memory& memory, const surface_storage& a, const surface_storage& b)
{
	if (a.parts.size() != b.parts.size()) return false;
	return memory.test_rebase([&](const auto&, const guest_surface_memory::storage_equivalent& equal)
	{
		for (std::size_t i = 0; i < a.parts.size(); ++i)
			if (a.parts[i].offset != b.parts[i].offset || !equal(a.parts[i].coverage.range, b.parts[i].coverage.range)) return false;
		return true;
	});
}
struct raw_plane
{
	texture_plane aspect;
	std::size_t offset, pitch;
};
struct raw_surface_snapshot
{
	id<MTLBuffer> buffer = nil;
	std::vector<raw_plane> planes;
};
struct deferred_copy_section
{
	surface_lease source;
	std::uint64_t generation;
	clear_rectangle rectangle;
	clear_rectangle destination;
	texture_plane plane;
	std::array<bool, 3> flip;
	bool typeless;
};
class surface_list_guard
{
	std::vector<cached_surface_ref>& list;
	const std::size_t size;

  public:
	surface_list_guard(std::vector<cached_surface_ref>& list, std::span<const cached_surface_ref> surfaces)
		: list(list), size(list.size()) { list.insert(list.end(), surfaces.begin(), surfaces.end()); }
	surface_list_guard(std::vector<cached_surface_ref>& list, const cached_surface_ref& surface)
		: list(list), size(list.size())
	{
		list.push_back(surface);
	}
	~surface_list_guard()
	{
		list.resize(size);
	}
};
bool same(const swizzle_region& a, const swizzle_region& b)
{
	return a.storage == b.storage && a.origin == b.origin && a.size == b.size;
}
bool same(const cached_surface_description& a, const cached_surface_description& b)
{
	return a.address == b.address && a.allocation == b.allocation && a.row_pitch == b.row_pitch &&
		   bool(a.morton) == bool(b.morton) && (!a.morton || same(*a.morton, *b.morton)) && a.tile == b.tile && a.segments == b.segments &&
		bool(a.mapping) == bool(b.mapping) && (!a.mapping || a.mapping->same_epoch(*b.mapping));
}
std::size_t descriptor_hash(const cached_surface_description& value)
{
	std::size_t result = 0;
	const auto add = [&](std::size_t word) { result ^= word + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2); };
	add(value.address); add(value.row_pitch);
	const auto& allocation = value.allocation;
	add(allocation.width); add(allocation.height); add(allocation.samples); add(allocation.expanded);
	add(allocation.format.index()); std::visit([&](auto format) { add(static_cast<std::size_t>(format)); }, allocation.format);
	add(bool(value.morton));
	if (value.morton)
		for (const auto& axis : {value.morton->storage, value.morton->origin, value.morton->size})
			for (const auto word : axis) add(word);
	add(bool(value.tile));
	if (value.tile)
	{
		add(value.tile->address); add(value.tile->size); add(value.tile->pitch);
		add(value.tile->bank); add(value.tile->offset);
	}
	for (const auto& part : value.segments) { add(part.address); add(part.offset); add(part.size); }
	add(bool(value.mapping)); if (value.mapping) add(value.mapping->epoch_hash());
	return result;
}
std::size_t pixel_bytes(const surface_allocation_description& a)
{
	if (const auto color = std::get_if<surface_color>(&a.format))
		return color_layout(*color).pixel_bytes;
	return describe_surface_depth(std::get<surface_depth>(a.format)).guest_bytes;
}
MTLPixelFormat native_format(const surface_allocation_description& a)
{
	if (const auto color = std::get_if<surface_color>(&a.format))
		return describe_surface_color(*color).format;
	return describe_surface_depth(std::get<surface_depth>(a.format)).format;
}
void validate_surface_blit(const surface_allocation_description& a, clear_rectangle rectangle,
						   const surface_allocation_description& b, clear_rectangle destination, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	if (!typeless && a.format != b.format)
		throw std::invalid_argument("Cached nearest copy requires equal guest format");
	if (rectangle.x > a.width || rectangle.width > a.width - rectangle.x || rectangle.y > a.height || rectangle.height > a.height - rectangle.y ||
		destination.x > b.width || destination.width > b.width - destination.x || destination.y > b.height || destination.height > b.height - destination.y)
		throw std::out_of_range("Cached native copy exceeds surface dimensions");
	if (flip[2])
		throw std::invalid_argument("Cached 2D blit cannot flip a volume axis");
	if (typeless)
	{
		if (plane != texture_plane::all)
			throw std::invalid_argument("Cached typeless transfers require all guest aspects");
		const auto source_size = a.samples == 1 ? std::array{rectangle.width, rectangle.height} : sample_expanded_size(rectangle.width, rectangle.height, a.samples);
		const auto target_size = b.samples == 1 ? std::array{destination.width, destination.height} : sample_expanded_size(destination.width, destination.height, b.samples);
		(void)plan_typeless(a.format, b.format, source_size[0], source_size[1], target_size[0], target_size[1]);
		return;
	}
	(void)storage_block(native_format(a), native_format(a) == MTLPixelFormatDepth32Float_Stencil8 && plane == texture_plane::all ? texture_plane::depth : plane);
}
} // namespace

class cached_surface final
{
  public:
	cached_surface_description description;
	guest_surface_layout layout;
	surface_storage storage;
	surface_lease lease;
	// Independent physical VM mapping,not a writer/VM lock. Keep dirty and
	// deferred output storage alive until writeback,discard or cache teardown.
	std::shared_ptr<void> backing;
	std::shared_ptr<physical_tickets> physical;
	raw_surface_snapshot spill;
	std::vector<deferred_copy_section> deferred;
	std::vector<std::uint64_t> deferred_revision;
	std::uint64_t generation = 0;
	bool initialized = false, dirty = false, faulted = false;
	std::vector<std::uint64_t> revision;
	std::uint64_t last_use = 0;
	// Zero identifies VM-decoded data. Successful native writes publish a
	// cache-wide epoch;flush and bit-exact spill/restore retain that provenance.
	std::uint64_t native_write_epoch = 0;
	void reset_backing() noexcept { backing.reset(); physical.reset(); }
};

surface_conflict_error::surface_conflict_error(guest_surface_range range)
	: std::runtime_error("Guest memory changed while native surface data was dirty; serialize VM access before writeback"), range(range)
{
}

struct surface_validation
{
	cached_surface_ref surface;
	std::vector<std::uint64_t> revisions;
	bool physical = false;
};

struct surface_backing
{
	cached_surface_ref surface;
	std::shared_ptr<void> owner;
	std::shared_ptr<physical_tickets> physical;
};

// Keep replaced GPU/storage owners until CPU publication has left the VM
// transaction. Last lease release may enter the pool/driver and must not run
// while guest writers and mappings are excluded.
struct surface_retirement
{
	cached_surface_ref surface;
	surface_lease allocation;
	id<MTLBuffer> spill;
	std::shared_ptr<void> backing;
	std::shared_ptr<physical_tickets> physical;
	std::vector<surface_lease> deferred;
};

struct surface_cache::impl
{
	context& ctx;
	const std::thread::id owner_thread = std::this_thread::get_id();
	std::shared_ptr<guest_surface_memory> memory;
	const std::size_t snapshot_limit;
	cpu_capture_dispatch capture;
	std::shared_ptr<heap_budget> shared_ledger;
	surface_pool pool;
	surface_color_transfer color;
	surface_depth_transfer depth;
	surface_swizzle_transfer swizzle;
	surface_banked_transfer banked;
	std::shared_ptr<sample_transfer> samples;
	clear_engine clears;
	surface_sample_blitter sample_blits;
	transfer_engine row_transfers;
	std::unique_ptr<depth_bounds_engine> bounds;
	std::vector<cached_surface_ref> surfaces;
	// These indexes do not own surfaces, watches or native allocations.
	std::unordered_map<const cached_surface*, std::size_t> positions;
	std::unordered_multimap<std::size_t, cached_surface*> descriptors;
	struct range_node
	{
		std::uint64_t begin, end, maximum;
		cached_surface* surface;
	};
	std::vector<range_node> ranges;
	bool ranges_dirty = true;
	surface_metadata_statistics metadata;
	surface_transaction_statistics transactions;
	struct snapshot_entry
	{
		std::weak_ptr<cached_surface> source;
		std::uint64_t generation, native_epoch;
		std::vector<std::uint64_t> versions;
		std::shared_ptr<heap_budget> budget;
		clear_rectangle rectangle;
		std::size_t row_offset, bytes;
		id<MTLTexture> texture;
		std::shared_ptr<const surface_texture_revision> revision;
		struct view { MTLPixelFormat format = MTLPixelFormatInvalid; id<MTLTexture> texture = nil; };
		std::array<view, 4> views{};
	};
	std::list<snapshot_entry> snapshots;
	surface_snapshot_statistics snapshot_stats;
	std::shared_ptr<void> snapshot_lifetime = std::make_shared<int>(0);
	std::uint64_t snapshot_generation = 1;
	void evict_snapshot(std::list<snapshot_entry>::iterator it)
	{
		snapshot_stats.bytes -= it->bytes;
		snapshots.erase(it); ++snapshot_stats.evicted;
		snapshot_stats.entries = snapshots.size();
	}
	void prune_snapshots()
	{
		for (auto it = snapshots.begin(); it != snapshots.end();)
		{
			const auto source = it->source.lock();
			if (!source || source->faulted || source->generation != it->generation ||
				source->native_write_epoch != it->native_epoch)
			{ auto stale = it++; evict_snapshot(stale); }
			else ++it;
		}
	}
	std::vector<cached_surface_ref> pinned;
	std::vector<cached_surface_ref> resolving;
	std::uint64_t clock = 0;
	std::uint64_t native_clock = 0;
	// Deferred/cancelled ownership can change a former VM hole without leaving
	// a materialized native source for a later ready-image lookup to discover.
	std::uint64_t texture_topology_epoch = 0;
	[[nodiscard]] std::uint64_t reserve_native_epoch()
	{
		if (native_clock == std::numeric_limits<std::uint64_t>::max())
			throw std::overflow_error("Native surface write epoch exhausted");
		return ++native_clock;
	}
	std::chrono::milliseconds timeout = std::chrono::seconds(30);
	struct rollback_surface
	{
		cached_surface_ref surface;
		cached_surface previous;
		raw_surface_snapshot snapshot;
		bool copy = false;
		surface_lease output;
	};
	struct native_rollback
	{
		std::vector<rollback_surface> surfaces;
		std::optional<submission> backup_work;
		bool preserve_contents = true;
		// A multi-draw commit can invalidate aliases before a later callback
		// fails. Retain their exact pre-batch validity until publication ends.
		std::vector<std::pair<cached_surface_ref, cached_surface>> aliases;
	};
	struct transaction
	{
		submission work;
		std::function<void()> commit;
		std::vector<cached_surface_ref> mutated;
		std::vector<surface_validation> validated;
		std::vector<surface_backing> backings;
		std::vector<surface_retirement> retirements;
		bool trim_pool = false;
		bool native_output = false;
		std::shared_ptr<native_rollback> rollback;
	};
	std::optional<transaction> pending;
	struct recorded_draw_batch
	{
		std::optional<submission> recording;
		std::shared_ptr<native_rollback> rollback = std::make_shared<native_rollback>();
		std::vector<std::function<void()>> commits;
		std::vector<cached_surface_ref> mutated;
		std::vector<surface_validation> validated;
		std::vector<surface_backing> backings;
		std::vector<surface_retirement> retirements;
	};
	std::size_t batch_capacity = 0;
	std::shared_ptr<void> batch_lifetime = std::make_shared<int>(0);
	std::shared_ptr<recorded_draw_batch> recording_batch;
	impl(context& ctx, std::shared_ptr<guest_surface_memory> memory, std::size_t budget, std::shared_ptr<heap_budget> shared_budget, cpu_capture_dispatch capture, surface_metadata_limits limits)
		: ctx(ctx), memory(std::move(memory)), snapshot_limit(budget), capture(std::move(capture)), shared_ledger(shared_budget), pool(ctx, budget, std::move(shared_budget)), color(ctx), depth(ctx), swizzle(ctx), banked(ctx), samples(std::make_shared<sample_transfer>(ctx)), clears(ctx), sample_blits(ctx), row_transfers(ctx)
	{
		if (!this->memory)
			throw std::invalid_argument("Native surface cache requires a VM memory adapter");
		if (!limits.entries || !limits.watch_parts || limits.entries > 65536 || limits.watch_parts > (1u << 20))
			throw std::invalid_argument("Native surface metadata limits are invalid");
		metadata.limits = limits;
	}
	inline static thread_local const impl* captured = nullptr;
	struct capture_scope
	{
		const impl* previous;
		explicit capture_scope(const impl* owner) : previous(std::exchange(captured, owner)) {}
		~capture_scope() { captured = previous; }
	};
	void cpu(const std::function<void()>& operation, const cpu_capture_dispatch& writer_capture = {}) const
	{
		if (captured == this) { operation(); return; }
		const auto invoke = [&] { capture_scope scope(this); operation(); };
		const auto& dispatch = writer_capture ? writer_capture : capture;
		if (dispatch) checked_cpu_capture(dispatch, invoke);
		else invoke(); // Legacy single-watch callers provide writer exclusion.
	}
	void outside_epoch() const
	{
		if (guest_memory_epoch::active() || captured)
			throw std::logic_error("Native surface retirement cannot run inside CPU/VM exclusion");
	}
	void rebuild_ranges()
	{
		if (!ranges_dirty) return;
		ranges.clear();
		for (const auto& surface : surfaces)
			for (const auto& part : surface->storage.parts)
				ranges.push_back({part.coverage.range.address, std::uint64_t(part.coverage.range.address) + part.coverage.range.size, 0, surface.get()});
		std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
		// A static balanced interval tree stored in sorted order. Every middle
		// element owns the maximum end of its complete implicit subtree.
		const auto build = [&](auto&& self, std::size_t first, std::size_t last) -> std::uint64_t
		{
			if (first == last) return 0;
			const auto middle = first + (last - first) / 2;
			return ranges[middle].maximum = std::max({ranges[middle].end, self(self, first, middle), self(self, middle + 1, last)});
		};
		build(build, 0, ranges.size());
		ranges_dirty = false;
	}
	std::vector<cached_surface_ref> range_candidates(std::span<const guest_surface_range> requested)
	{
		rebuild_ranges();
		std::vector<bool> selected(surfaces.size());
		const auto query = [&](auto&& self, std::size_t first, std::size_t last, guest_surface_range range) -> void
		{
			if (first == last) return;
			const auto middle = first + (last - first) / 2;
			const auto& node = ranges[middle];
			++metadata.range_nodes;
			const auto end = std::uint64_t(range.address) + range.size;
			if (node.maximum <= range.address || ranges[first].begin >= end) return;
			self(self, first, middle, range);
			if (node.begin < end && range.address < node.end) selected[positions.at(node.surface)] = true;
			if (node.begin < end) self(self, middle + 1, last, range);
		};
		for (const auto range : requested) { ++metadata.range_queries; if (range.size) query(query, 0, ranges.size(), range); }
		std::vector<cached_surface_ref> result;
		for (std::size_t index = 0; index < selected.size(); ++index)
			if (selected[index]) result.push_back(surfaces[index]);
		return result;
	}
	std::vector<cached_surface_ref> candidates(bool direct, std::span<const guest_surface_range> requested)
	{
		if (direct) return range_candidates(requested);
		// Physical aliases need not share any virtual range. Never discard them
		// using an EA-only index; this fallback is bounded by the metadata cap.
		metadata.physical_candidates += surfaces.size();
		return surfaces;
	}
	void remove(const cached_surface_ref& surface)
	{
		outside_epoch();
		const auto index = positions.at(surface.get());
		const auto bucket = descriptors.equal_range(descriptor_hash(surface->description));
		for (auto i = bucket.first; i != bucket.second; ++i)
			if (i->second == surface.get()) { descriptors.erase(i); break; }
		positions.erase(surface.get());
		metadata.watch_parts -= surface->storage.parts.size();
		// Invalidate raw pointers before the final owner can be released.
		ranges.clear(); ranges_dirty = true;
		surfaces.erase(surfaces.begin() + index);
		for (auto i = index; i < surfaces.size(); ++i) positions.at(surfaces[i].get()) = i;
		metadata.entries = surfaces.size();
		++metadata.retired;
	}
	bool retirable(const cached_surface_ref& surface)
	{
		++metadata.retirement_visits;
		if (surface.use_count() != 1 || surface->dirty || !surface->deferred.empty() || surface->faulted ||
			(surface->lease && surface->lease.use_count() != 1)) return false;
		// Pending transactions/pins/resolving lists own shared handles too.
		if (!surface->initialized || !surface->native_write_epoch) return true;
		// Clean native precision can serve RTT after writeback. Keep it until
		// its mapping or active-byte revision is positively known to be stale.
		bool stale = false;
		cpu([&]
		{
			if (surface->description.mapping && !surface->description.mapping->current()) { stale = true; return; }
			for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
				if (memory->surface_revision(surface->storage.parts[i].coverage) != surface->revision[i]) { stale = true; break; }
		});
		return stale;
	}
	std::size_t retire_to(std::size_t target_entries, std::size_t target_parts)
	{
		outside_epoch();
		if (surfaces.size() <= target_entries && metadata.watch_parts <= target_parts) return 0;
		std::vector<cached_surface*> ordered;
		for (const auto& surface : surfaces) ordered.push_back(surface.get());
		std::stable_sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) { return a->last_use < b->last_use; });
		const auto before = surfaces.size();
		for (const auto* candidate : ordered)
		{
			if (surfaces.size() <= target_entries && metadata.watch_parts <= target_parts) break;
			const auto& surface = surfaces[positions.at(candidate)];
			if (retirable(surface)) remove(surface);
		}
		return before - surfaces.size();
	}
	bool reclaim_clean(const cached_surface_ref& protected_surface = {})
	{
		outside_epoch();
		if (pending || recording_batch) return false;
		cached_surface_ref oldest;
		for (const auto& surface : surfaces)
		{
			if (surface == protected_surface || (!surface->lease && !surface->spill.buffer) || surface->dirty || surface->faulted || !surface->deferred.empty() ||
				(surface->initialized && surface->native_write_epoch) || (surface->lease && surface->lease.use_count() != 1) ||
				std::find(pinned.begin(), pinned.end(), surface) != pinned.end() ||
				std::find(resolving.begin(), resolving.end(), surface) != resolving.end()) continue;
			if (!oldest || surface->last_use < oldest->last_use) oldest = surface;
		}
		if (!oldest) return false;
		oldest->lease.reset();
		oldest->spill = {};
		oldest->reset_backing();
		oldest->initialized = false;
		oldest->native_write_epoch = 0;
		(void)pool.trim();
		return true;
	}
	void reserve_metadata(std::size_t parts)
	{
		if (parts > metadata.limits.watch_parts) throw surface_metadata_budget_error();
		retire_to(metadata.limits.entries - 1, metadata.limits.watch_parts - parts);
		if (surfaces.size() >= metadata.limits.entries || metadata.watch_parts > metadata.limits.watch_parts - parts)
			throw surface_metadata_budget_error();
	}
	static void check_mapping(const cached_surface_ref& surface)
	{
		if (surface->description.mapping && !surface->description.mapping->current()) throw surface_mapping_conflict();
	}
	void validate_in_epoch(std::span<const surface_validation> values) const
	{
		for (const auto& value : values)
		{
			const auto& surface = value.surface;
			if (value.physical)
			{
				if (!surface->physical || surface->physical->size() != surface->storage.parts.size())
					throw std::logic_error("Native surface lacks complete physical tickets");
				for (const auto& ticket : *surface->physical)
					if (!memory->physical_surface_current(*ticket)) throw surface_conflict_error(ticket->coverage().range);
				continue;
			}
			check_mapping(surface);
			if (value.revisions.size() != surface->storage.parts.size()) throw std::logic_error("Surface validation lacks watch versions");
			for (std::size_t i = 0; i < value.revisions.size(); ++i)
				if (memory->surface_revision(surface->storage.parts[i].coverage) != value.revisions[i])
					throw surface_conflict_error(surface->storage.parts[i].coverage.range);
			check_mapping(surface);
		}
	}
	void validate_mappings(std::span<const surface_validation> values) const
	{
		if (!values.empty()) cpu([&] { validate_in_epoch(values); });
	}
	void validate_completion(const transaction& completed) const
	{
		const auto retained_inputs_current = [&]
		{
			// A successful native write belongs to the physical inputs captured
			// before submission. A later VA/IO/tile change cannot turn that valid
			// output into a GPU failure. CPU changes to those physical bytes still
			// refuse;uploads and unknown/failed GPU status never take this path.
			if (!completed.native_output || completed.backings.empty()) return false;
			for (const auto& input : completed.backings)
				if (!input.physical || input.physical->size() != input.surface->storage.parts.size()) return false;
			for (const auto& input : completed.backings)
				for (const auto& ticket : *input.physical)
					if (!memory->physical_surface_current(*ticket)) throw surface_conflict_error(ticket->coverage().range);
			return true;
		};
		// Virtual access and IO/tile policy were checked immediately before
		// submission. Requiring those VAs again after success would lose valid
		// old-allocation output on unmap/protection as well as on remap.
		if (!retained_inputs_current()) validate_in_epoch(completed.validated);
	}
	std::vector<surface_backing> retain_backings(std::span<const surface_validation> values) const
	{
		std::vector<surface_backing> result;
		if (values.empty()) return result;
		result.reserve(values.size());
		cpu([&]
		{
			validate_in_epoch(values);
			for (const auto& value : values)
			{
				if (std::any_of(result.begin(), result.end(), [&](const auto& retained) { return retained.surface == value.surface; })) continue;
				if (value.physical)
				{
					result.push_back({value.surface, value.surface->backing, value.surface->physical});
					continue;
				}
				auto tickets = std::make_shared<physical_tickets>();
				tickets->reserve(value.surface->storage.parts.size());
				std::vector<guest_surface_range> ranges;
				for (const auto& part : value.surface->storage.parts)
				{
					guest_surface_byte_cursor cursor(part.coverage);
					while (const auto range = cursor.next())
					{
						if (ranges.size() == (1u << 20)) throw std::length_error("Surface backing capture exceeds active-range budget");
						ranges.push_back(*range);
					}
				}
				for (std::size_t i = 0; i < value.surface->storage.parts.size(); ++i)
				{
					const auto ticket = memory->capture_physical_surface(value.surface->storage.parts[i].coverage, value.revisions[i]);
					if (ticket) tickets->push_back(ticket);
				}
				if (!tickets->empty() && tickets->size() != value.surface->storage.parts.size())
					throw std::runtime_error("Native surface has only partial physical publication support");
				result.push_back({value.surface, memory->retain_backing(ranges), tickets->empty() ? nullptr : std::move(tickets)});
			}
			validate_in_epoch(values);
		});
		return result;
	}
	static void retain_mutated_backings(const transaction& value) noexcept
	{
		for (const auto& surface : value.mutated)
			for (const auto& retained : value.backings)
				if (retained.surface == surface) { surface->backing = retained.owner; surface->physical = retained.physical; break; }
	}
	static std::vector<surface_retirement> retain_metadata_storage(std::span<const surface_validation> validated,
		std::span<const cached_surface_ref> mutated, std::span<const cached_surface_ref> aliases)
	{
		std::vector<surface_retirement> result;
		const auto retain = [&](const cached_surface_ref& surface)
		{
			if (std::any_of(result.begin(), result.end(), [&](const auto& item) { return item.surface == surface; })) return;
			surface_retirement owner{surface, surface->lease, surface->spill.buffer, surface->backing, surface->physical, {}};
			owner.deferred.reserve(surface->deferred.size());
			for (const auto& section : surface->deferred) owner.deferred.push_back(section.source);
			result.push_back(std::move(owner));
		};
		for (const auto& value : validated) retain(value.surface);
		for (const auto& surface : mutated) retain(surface);
		for (const auto& surface : aliases) retain(surface);
		return result;
	}
	std::vector<std::uint64_t> current(const cached_surface_ref& surface) const
	{
		std::vector<std::uint64_t> result(surface->storage.parts.size());
		cpu([&]
		{
			check_mapping(surface);
			for (std::size_t i = 0; i < result.size(); ++i)
				result[i] = memory->surface_revision(surface->storage.parts[i].coverage);
			check_mapping(surface);
		});
		return result;
	}
	surface_storage_snapshot read(const cached_surface_ref& surface) const
	{
		surface_storage_snapshot result{std::vector<std::uint64_t>(surface->storage.parts.size()), std::vector<std::byte>(surface->storage.size)};
		cpu([&]
		{
			check_mapping(surface);
			for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
			{
				const auto& part = surface->storage.parts[i];
				const auto snapshot = memory->read_surface(part.coverage);
				if (snapshot.bytes.size() != part.coverage.range.size)
					throw std::runtime_error("VM surface snapshot has the wrong byte count");
				std::copy(snapshot.bytes.begin(), snapshot.bytes.end(), result.bytes.begin() + part.offset);
				result.revisions[i] = snapshot.revision;
			}
			check_mapping(surface);
		});
		return result;
	}
	std::vector<std::uint64_t> publish(const cached_surface_ref& surface, std::span<const std::byte> bytes) const
	{
		if (bytes.size() != surface->storage.size || surface->revision.size() != surface->storage.parts.size())
			throw std::logic_error("Native surface publication differs from captured storage");
		std::vector<guest_surface_write> writes;
		writes.reserve(surface->storage.parts.size());
		std::vector<std::uint64_t> result(surface->storage.parts.size());
		for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
		{
			const auto& part = surface->storage.parts[i];
			writes.push_back({part.coverage, surface->revision[i], bytes.subspan(part.offset, part.coverage.range.size)});
		}
		if (!capture)
		{
			// Preserve the original single-watch CAS contract. A separately
			// unlocked version query could accept an intervening CPU write.
			if (writes.size() != 1) throw std::logic_error("Disjoint publication lacks CPU exclusion");
			const auto revision = memory->compare_write(writes[0].coverage, writes[0].expected, writes[0].bytes);
			if (!revision) throw surface_conflict_error(writes[0].coverage.range);
			result[0] = *revision;
			return result;
		}
		cpu([&]
		{
			check_mapping(surface);
			if (!memory->compare_write_many(writes)) throw surface_conflict_error(surface->storage.parts.front().coverage.range);
			// The batch result is provider-defined. Query each watch under the
			// same writer exclusion instead of assuming a common version domain.
			for (std::size_t i = 0; i < result.size(); ++i)
				result[i] = memory->surface_revision(surface->storage.parts[i].coverage);
		});
		return result;
	}
	void check(const cached_surface_ref& surface) const
	{
		if (!surface || !positions.contains(surface.get()))
			throw std::invalid_argument("Native cached surface belongs to another cache");
	}
	// Versioned writes retain untouched old native storage. In-place fallback
	// requires a successful independent read-only backup. Keep replaced leases
	// in rollback/retirements until publication leaves CPU exclusion.
	void reject(transaction& completed, bool backup_trusted) noexcept
	{
		if (completed.rollback)
		{
			for (auto& [surface, previous] : completed.rollback->aliases)
			{
				surface->initialized = previous.initialized;
				surface->spill = std::move(previous.spill);
				surface->native_write_epoch = previous.native_write_epoch;
			}
			retain_mutated_backings(completed);
			for (auto& saved : completed.rollback->surfaces)
			{
				auto& target = *saved.surface;
				auto& previous = saved.previous;
				if (!saved.output && !backup_trusted)
				{
					target.faulted = true;
					target.initialized = false;
					continue;
				}
				// A versioned writer never modified the old native allocation;
				// its contents are independent of the failed command's status.
				target.lease = saved.output ? previous.lease : surface_lease{};
				target.spill = saved.output ? std::move(previous.spill) : std::move(saved.snapshot);
				target.backing = std::move(previous.backing);
				target.physical = std::move(previous.physical);
				target.deferred = std::move(previous.deferred);
				target.deferred_revision = std::move(previous.deferred_revision);
				target.revision = std::move(previous.revision);
				target.generation = previous.generation;
				target.native_write_epoch = previous.native_write_epoch;
				target.initialized = saved.output && previous.initialized;
				target.dirty = previous.dirty;
				target.faulted = previous.faulted;
			}
			return;
		}
		retain_mutated_backings(completed);
		for (const auto& surface : completed.mutated) { surface->faulted = true; surface->initialized = false; }
	}
	void restore_recording(const std::shared_ptr<recorded_draw_batch>& batch) noexcept
	{
		for (const auto& saved : batch->rollback->surfaces)
			if (saved.output) saved.surface->lease = saved.previous.lease;
	}
	void cancel_draw_batch() noexcept
	{
		batch_capacity = 0;
		if (!recording_batch) return;
		auto batch = std::move(recording_batch);
		try { ctx.cancel(); } catch (...) {}
		restore_recording(batch);
		++transactions.cancelled_draw_batches;
	}
	void flush_draw_batch()
	{
		if (!recording_batch) return;
		auto batch = recording_batch;
		try
		{
			ctx.end_encoding();
			validate_mappings(batch->validated);
			// Construct every callback/container before the irreversible submit.
			transaction value{ctx.recording_submission(), [batch]
				{ for (const auto& commit : batch->commits) commit(); },
				batch->mutated, batch->validated, batch->backings, batch->retirements, false, true, batch->rollback};
			value.work = ctx.submit();
			pending.emplace(std::move(value));
			recording_batch.reset();
			++transactions.submitted;
			++transactions.draw_batches;
		}
		catch (...)
		{
			ctx.cancel();
			restore_recording(batch);
			recording_batch.reset();
			++transactions.cancelled_draw_batches;
			throw;
		}
	}
	void synchronize(std::chrono::milliseconds wait, bool flush_recording = true)
	{
		if (flush_recording) flush_draw_batch();
		if (!pending) return;
		std::exception_ptr gpu_error;
		try { pending->work.wait(wait); }
		catch (...)
		{
			if (!pending->work.ready()) throw; // A timeout never retires live work.
			gpu_error = std::current_exception();
		}
		// Never wait for either GPU submission under guest writer exclusion.
		bool backup_trusted = bool(pending->rollback);
		if (backup_trusted && pending->rollback->backup_work)
		{
			try { pending->rollback->backup_work->wait(wait); }
			catch (...) { backup_trusted = false; }
		}
		auto completed = std::move(*pending);
		pending.reset();
		bool finalized = false;
		try
		{
			cpu([&]
			{
				try
				{
					if (gpu_error) std::rethrow_exception(gpu_error);
					validate_completion(completed);
					retain_mutated_backings(completed);
					completed.commit();
					for (const auto& surface : completed.mutated)
						if (!surface->dirty && surface->deferred.empty() && !surface->faulted) surface->reset_backing();
					finalized = true;
				}
				catch (...) { reject(completed, backup_trusted); finalized = true; throw; }
			});
		}
		catch (...)
		{
			// A dispatcher which refused to enter leaves completion available for
			// retry, rather than losing the output or its only rollback owner.
			if (!finalized) pending.emplace(std::move(completed));
			throw;
		}
		completed.commit = {};
		completed.retirements.clear();
		if (completed.trim_pool) (void)pool.trim();
	}
	bool batch_alias(const cached_surface_ref& surface)
	{
		if (!recording_batch) return false;
		for (const auto& other : recording_batch->mutated)
			if (other != surface && memory_storage_overlap(*memory, surface->storage, other->storage)) return true;
		return false;
	}
	void batch_surface_boundary(const cached_surface_ref& surface)
	{
		if (!recording_batch) return;
		if (batch_alias(surface) || (std::find(recording_batch->mutated.begin(), recording_batch->mutated.end(), surface) != recording_batch->mutated.end() && current(surface) != surface->revision))
			synchronize(timeout);
	}
	bool append_draw_batch(const std::function<void()>& encode, std::function<void()>& commit,
		const std::vector<cached_surface_ref>& mutated, std::vector<surface_validation>& validated,
		const std::vector<cached_surface_ref>& aliases, bool preserve_contents)
	{
		if (!batch_capacity || mutated.empty()) return false;
		if (recording_batch && recording_batch->commits.size() == batch_capacity) synchronize(timeout);
		for (const auto& surface : mutated) batch_surface_boundary(surface);
		// An uninitialized full clear needs the ordinary initialization contract.
		for (const auto& surface : mutated)
			if (!surface->initialized || surface->faulted || surface->spill.buffer)
			{
				synchronize(timeout);
				return false;
			}
		if (validated.empty()) for (const auto& surface : mutated) validated.push_back({surface, surface->revision});
		auto backings = retain_backings(validated);
		auto retirements = retain_metadata_storage(validated, mutated, aliases);
		surface_list_guard pins(pinned, mutated);
		std::vector<rollback_surface> additions;
		for (const auto& surface : mutated)
		{
			if (recording_batch && std::any_of(recording_batch->rollback->surfaces.begin(), recording_batch->rollback->surfaces.end(),
				[&](const auto& saved) { return saved.surface == surface; })) continue;
			if (std::any_of(additions.begin(), additions.end(), [&](const auto& saved) { return saved.surface == surface; })) continue;
			rollback_surface saved{surface, *surface, surface->spill, false};
			try { saved.output = pool.acquire(surface->description.allocation); }
			catch (const surface_budget_error&)
			{
				additions.clear();
				synchronize(timeout);
				return false;
			}
			catch (const heap_budget_error&)
			{
				additions.clear();
				synchronize(timeout);
				return false;
			}
			catch (const native_allocation_error&)
			{
				additions.clear();
				synchronize(timeout);
				return false;
			}
			additions.push_back(std::move(saved));
		}
		auto batch = recording_batch ? recording_batch : std::make_shared<recorded_draw_batch>();
		try
		{
			// All owners/callbacks are attached before encoding starts. Any later
			// exception cancels this complete unsubmitted batch and restores leases.
			batch->commits.push_back(std::move(commit));
			batch->validated.insert(batch->validated.end(), validated.begin(), validated.end());
			batch->backings.insert(batch->backings.end(), backings.begin(), backings.end());
			batch->retirements.insert(batch->retirements.end(), retirements.begin(), retirements.end());
			for (const auto& surface : mutated)
				if (std::find(batch->mutated.begin(), batch->mutated.end(), surface) == batch->mutated.end()) batch->mutated.push_back(surface);
			for (const auto& surface : aliases)
				if (std::none_of(batch->rollback->aliases.begin(), batch->rollback->aliases.end(), [&](const auto& item) { return item.first == surface; }))
					batch->rollback->aliases.emplace_back(surface, *surface);
			const auto first_addition = batch->rollback->surfaces.size();
			batch->rollback->surfaces.insert(batch->rollback->surfaces.end(), std::make_move_iterator(additions.begin()), std::make_move_iterator(additions.end()));
			if (!recording_batch)
			{
				ctx.begin();
				batch->recording.emplace(ctx.recording_submission());
				recording_batch = batch;
				ctx.keep_alive(batch);
			}
			for (const auto& input : backings)
			{
				if (input.owner) ctx.keep_alive(input.owner);
				if (input.physical) ctx.keep_alive(input.physical);
			}
			validate_mappings(validated);
			for (std::size_t i = first_addition; i < batch->rollback->surfaces.size(); ++i)
			{
				const auto& saved = batch->rollback->surfaces[i];
				if (preserve_contents)
				{
					texture_region whole;
					whole.size = MTLSizeMake(saved.surface->description.allocation.width, saved.surface->description.allocation.height, 1);
					copy_texture(ctx, saved.previous.lease->render_texture(ctx), whole, saved.output->render_texture(ctx), whole);
					++transactions.native_version_blits;
				}
				saved.surface->lease = saved.output;
				++transactions.native_versions;
			}
			encode();
			++transactions.batched_draws;
			return true;
		}
		catch (...)
		{
			if (recording_batch) { ctx.cancel(); recording_batch.reset(); ++transactions.cancelled_draw_batches; }
			restore_recording(batch);
			throw;
		}
	}
	void execute(const std::function<void()>& encode, std::function<void()> commit,
				 std::vector<cached_surface_ref> mutated = {}, std::vector<surface_validation> validated = {},
				 std::vector<cached_surface_ref> aliases = {}, bool trim_pool = false, bool native_output = false, bool preserve_contents = true, bool batchable = false)
	{
		if (batchable && native_output && append_draw_batch(encode, commit, mutated, validated, aliases, preserve_contents)) return;
		synchronize(timeout);
		if (validated.empty())
			for (const auto& surface : mutated) validated.push_back({surface, surface->revision});
		auto backings = retain_backings(validated);
		auto retirements = retain_metadata_storage(validated, mutated, aliases);
		auto rollback = native_output ? backup_native(mutated, preserve_contents) : nullptr;
		// Allocate the transaction's callbacks and owners before submission.
		@autoreleasepool
		{
			ctx.begin();
			try
			{
				if (rollback)
				{
					ctx.keep_alive(rollback);
					for (auto& saved : rollback->surfaces)
					{
						if (!saved.output) continue;
						texture_region whole;
						const auto& allocation = saved.surface->description.allocation;
						whole.size = MTLSizeMake(allocation.width, allocation.height, 1);
						if (rollback->preserve_contents)
						{
							copy_texture(ctx, saved.previous.lease->render_texture(ctx), whole, saved.output->render_texture(ctx), whole);
							++transactions.native_version_blits;
						}
						saved.surface->lease = saved.output;
					}
				}
				for (const auto& retained : backings)
				{
					if (retained.owner) ctx.keep_alive(retained.owner);
					if (retained.physical) ctx.keep_alive(retained.physical);
				}
				encode();
			}
			catch (...)
			{
				ctx.cancel();
				if (rollback) for (auto& saved : rollback->surfaces) if (saved.output) saved.surface->lease = saved.previous.lease;
				throw;
			}
			try
			{
				ctx.end_encoding();
				validate_mappings(validated);
				const auto work = ctx.submit();
				++transactions.submitted;
				pending.emplace(transaction{work, std::move(commit), std::move(mutated), std::move(validated), std::move(backings), std::move(retirements), trim_pool, native_output, std::move(rollback)});
			}
			catch (...)
			{
				ctx.cancel();
				if (rollback) for (auto& saved : rollback->surfaces) if (saved.output) saved.surface->lease = saved.previous.lease;
				throw;
			}
		}
		synchronize(timeout);
	}
	texture_region region(const cached_surface_ref& surface) const
	{
		texture_region result;
		result.size = MTLSizeMake(surface->layout.width, surface->layout.height, 1);
		return result;
	}
	id<MTLTexture> transfer_texture(const cached_surface_ref& surface, const surface_lease& target = {}) __attribute__((ns_returns_retained))
	{
		const auto& allocation = target ? target : surface->lease;
		return surface->description.allocation.samples > 1 ? allocation->expanded_texture(ctx) : allocation->render_texture(ctx);
	}
	void upload_guest(const cached_surface_ref& surface, id<MTLBuffer> buffer, const surface_lease& target = {})
	{
		const auto& allocation = target ? target : surface->lease;
		const auto texture = transfer_texture(surface, allocation);
		const auto area = region(surface);
		auto pitch = surface->layout.row_pitch;
		if (!surface->description.tile && !surface->layout.morton && pitch % surface->layout.pixel_bytes)
		{
			const auto row = checked_texture_product(surface->layout.width, surface->layout.pixel_bytes);
			const auto packed = make_buffer(ctx, checked_texture_product(row, surface->layout.height), buffer_storage::private_gpu);
			for (std::size_t y = 0; y < surface->layout.height; ++y)
				row_transfers.copy(ctx, buffer, y * pitch, packed, y * row, row);
			buffer = packed;pitch = row;
		}
		if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
		{
			if (surface->description.tile)
				banked.upload_color(ctx, *format, buffer, surface->layout, *surface->description.tile, texture, area);
			else if (surface->layout.morton)
				swizzle.upload_color(ctx, *format, buffer, 0, *surface->layout.morton, texture, area);
			else
				color.upload(ctx, *format, buffer, {0, pitch}, texture, area);
		}
		else
		{
			const auto depth_format = std::get<surface_depth>(surface->description.allocation.format);
			if (surface->description.tile)
				banked.upload_depth(ctx, depth_format, buffer, surface->layout, *surface->description.tile, texture, area);
			else if (surface->layout.morton)
				swizzle.upload_depth(ctx, depth_format, buffer, 0, *surface->layout.morton, texture, area);
			else
				depth.upload(ctx, depth_format, buffer, {0, pitch}, texture, area);
		}
		if (surface->description.allocation.samples > 1)
			samples->import(ctx, texture, allocation->render_texture(ctx));
	}
	void download_guest(const cached_surface_ref& surface, id<MTLBuffer> buffer)
	{
		const auto texture = transfer_texture(surface);
		const auto area = region(surface);
		const auto output = buffer;
		auto pitch = surface->layout.row_pitch;
		const bool repack = !surface->description.tile && !surface->layout.morton && pitch % surface->layout.pixel_bytes;
		if (repack)
		{
			pitch = checked_texture_product(surface->layout.width, surface->layout.pixel_bytes);
			buffer = make_buffer(ctx, checked_texture_product(pitch, surface->layout.height), buffer_storage::private_gpu);
		}
		if (surface->description.allocation.samples > 1)
			samples->expand(ctx, surface->lease->render_texture(ctx), texture);
		if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
		{
			if (surface->description.tile)
				banked.download_color(ctx, *format, texture, area, buffer, surface->layout, *surface->description.tile);
			else if (surface->layout.morton)
				swizzle.download_color(ctx, *format, texture, area, buffer, 0, *surface->layout.morton);
			else
				color.download(ctx, *format, texture, area, buffer, {0, pitch});
		}
		else
		{
			const auto depth_format = std::get<surface_depth>(surface->description.allocation.format);
			if (surface->description.tile)
				banked.download_depth(ctx, depth_format, texture, area, buffer, surface->layout, *surface->description.tile);
			else if (surface->layout.morton)
				swizzle.download_depth(ctx, depth_format, texture, area, buffer, 0, *surface->layout.morton);
			else
				depth.download(ctx, depth_format, texture, area, buffer, {0, pitch});
		}
		if (repack)
			for (std::size_t y = 0; y < surface->layout.height; ++y)
				row_transfers.copy(ctx, buffer, y * pitch, output, y * surface->layout.row_pitch, pitch);
	}
	[[nodiscard]] std::vector<cached_surface_ref> plan_aliases(const cached_surface_ref& surface)
	{
		std::vector<cached_surface_ref> result;
		std::vector<guest_surface_range> requested;
		for (const auto& part : surface->storage.parts) requested.push_back(part.coverage.range);
		memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
							 {
			for (const auto& other : candidates(direct, requested))
				if (other != surface && active_storage_overlap(direct, lookup, surface->storage, other->storage))
					result.push_back(other);
			return false; });
		return result;
	}
	static void invalidate_aliases(const std::vector<cached_surface_ref>& aliases) noexcept
	{
		for (const auto& other : aliases)
		{
			other->initialized = false;
			other->spill = {};
			other->native_write_epoch = 0;
		}
	}
	[[nodiscard]] std::vector<cached_surface_ref> cpu_aliases(const guest_surface_layout& coverage)
	{
		return cpu_aliases(std::span{&coverage, 1});
	}
	[[nodiscard]] std::vector<cached_surface_ref> cpu_aliases(std::span<const guest_surface_layout> coverage, bool native_only = false)
	{
		std::vector<cached_surface_ref> result;
		std::vector<guest_surface_range> requested;
		for (const auto& part : coverage) requested.push_back(part.range);
		memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
							 {
			for (const auto& surface : candidates(direct, requested))
				if ((!native_only || surface->dirty || !surface->deferred.empty() ||
					(surface->initialized && !surface->faulted && (surface->lease || surface->spill.buffer))) &&
					std::any_of(coverage.begin(), coverage.end(), [&](const auto& part)
					{ return active_storage_overlap(direct, lookup, part, surface->storage); }))
					result.push_back(surface);
			return false; });
		return result;
	}
	void flush(const cached_surface_ref& surface)
	{
		check(surface);
		synchronize(timeout);
		if (surface->physical && (surface->dirty || !surface->deferred.empty()))
		{
			flush_physical(surface, true);
			return;
		}
		materialize(surface);
		// Canonical writeback changes the revision of overlapping copy plans.
		// Consume those plans first. Nested writeback while a plan initializes its
		// background skips the resolving target and completes the old owner once.
		if (surface->dirty)
			for (const auto& other : surfaces)
				if (other != surface && !other->deferred.empty() && memory_storage_overlap(*memory, surface->storage, other->storage) &&
					std::find(resolving.begin(), resolving.end(), other) == resolving.end())
					materialize(other);
		if (!surface->dirty)
			return;
		prepare(surface, surface_initialization::guest);
		const auto snapshot = read(surface);
		if (snapshot.revisions != surface->revision)
			throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		if (snapshot.bytes.size() != surface->layout.range.size)
			throw std::runtime_error("VM surface snapshot has the wrong byte count");
		const auto output = make_upload(ctx, snapshot.bytes);
		struct writeback
		{
			id<MTLBuffer> buffer;
			std::span<std::byte> bytes;
		};
		// Capture the shared CPU pointer before entering publication;do not
		// query Metal properties while the VM writer lock is held.
		const writeback destination{output, map_buffer(output, 0, output.length)};
		const auto aliases = plan_aliases(surface);
		execute([&]
				{ download_guest(surface, output); },
				[this, surface, destination, aliases]
				{
					surface->revision = publish(surface, destination.bytes);
					surface->dirty = false;
					surface->reset_backing();
					invalidate_aliases(aliases);
				}, {}, {{surface, surface->revision}}, aliases);
	}
	std::vector<cached_surface_ref> physical_aliases(const cached_surface_ref& surface) const
	{
		std::vector<cached_surface_ref> aliases;
		std::vector<std::shared_ptr<const physical_backing>> parts;
		for (const auto& ticket : *surface->physical)
			parts.insert(parts.end(), ticket->parts().begin(), ticket->parts().end());
		const physical_identity identity(parts);
		for (const auto& other : surfaces)
		{
			if (other == surface) continue;
			if (other->dirty || !other->deferred.empty())
			{
				if (!other->physical) throw std::runtime_error("Physical recovery cannot order untyped dirty aliases");
				for (const auto& a : *surface->physical)
					for (const auto& b : *other->physical)
						if (a->overlaps(*b)) throw surface_conflict_error(a->coverage().range);
			}
			else
			{
				// Typed providers can preserve disjoint clean native precision.
				// Legacy metadata without this query stays conservative.
				for (const auto& part : other->storage.parts)
				{
					const auto overlap = memory->physical_identity_overlaps_current(identity, part.coverage);
					if (!overlap || *overlap) { aliases.push_back(other); break; }
				}
			}
		}
		return aliases;
	}
	std::vector<std::byte> read_physical(const cached_surface_ref& surface) const
	{
		const surface_validation validation{surface, {}, true};
		std::vector<std::byte> old(surface->storage.size);
		cpu([&]
		{
			validate_in_epoch({&validation, 1});
			for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
			{
				const auto snapshot = memory->read_physical_surface(*(*surface->physical)[i]);
				const auto& part = surface->storage.parts[i];
				if (!snapshot || snapshot->bytes.size() != part.coverage.range.size)
					throw std::runtime_error("Physical recovery lacks an exact old-owner snapshot");
				std::copy(snapshot->bytes.begin(), snapshot->bytes.end(), old.begin() + part.offset);
			}
		});
		return old;
	}
	void materialize_physical(const cached_surface_ref& surface)
	{
		if (surface->deferred.empty()) return;
		if (surface->faulted) throw std::runtime_error("Cannot materialize faulted native physical output");
		const surface_validation validation{surface, {}, true};
		validate_mappings({&validation, 1});
		const auto aliases = physical_aliases(surface);
		surface_list_guard pin(pinned, surface);
		std::size_t first = 0;
		bool full = false;
		for (std::size_t i = 0; i < surface->deferred.size(); ++i)
		{
			const auto& section = surface->deferred[i];
			if (section.plane == texture_plane::all && section.destination.x == 0 && section.destination.y == 0 &&
				section.destination.width == surface->description.allocation.width && section.destination.height == surface->description.allocation.height)
			{
				first = i;
				full = true;
			}
		}
		const bool native_background = surface->dirty || surface->revision == surface->deferred_revision;
		const auto spill = !full && native_background ? surface->spill : raw_surface_snapshot{};
		const auto background = !full && native_background && surface->initialized && !spill.buffer ? surface->lease : surface_lease{};
		if (!full && surface->dirty && !spill.buffer && !background)
			throw std::runtime_error("Deferred physical output lost its prior native dirty contents");
		const auto input = !full && !spill.buffer && !background ? make_upload(ctx, read_physical(surface)) : nil;
		// Write into a distinct allocation. Captured sources (including self)
		// and the prior target stay intact on timeout,error or failed CPU commit.
		const auto destination = allocate_transient(surface->description.allocation, surface);
		const auto native_epoch = reserve_native_epoch();
		auto revision = surface->deferred_revision;
		execute([&]
		{
			ctx.keep_alive(destination);
			const auto texture = destination->render_texture(ctx);
			if (input) upload_guest(surface, input, destination);
			else if (spill.buffer)
			{
				const auto transfer = transfer_texture(surface, destination);
				for (const auto& plane : spill.planes)
					buffer_to_texture(ctx, spill.buffer, plane.offset, transfer, region(surface), plane.pitch, 0, plane.aspect);
				if (surface->description.allocation.samples > 1) samples->import(ctx, transfer, texture);
			}
			else if (background)
			{
				texture_region area;
				area.size = MTLSizeMake(surface->description.allocation.width, surface->description.allocation.height, 1);
				copy_texture(ctx, background->render_texture(ctx), area, texture, area);
			}
			for (std::size_t i = first; i < surface->deferred.size(); ++i)
			{
				const auto& section = surface->deferred[i];
				texture_region sr, dr;
				sr.origin = MTLOriginMake(section.rectangle.x, section.rectangle.y, 0);
				dr.origin = MTLOriginMake(section.destination.x, section.destination.y, 0);
				sr.size = MTLSizeMake(section.rectangle.width, section.rectangle.height, 1);
				dr.size = MTLSizeMake(section.destination.width, section.destination.height, 1);
				sample_blits.nearest(ctx, section.source->description().format, section.source->render_texture(ctx), sr,
					surface->description.allocation.format, texture, dr, section.plane, section.flip, section.typeless);
			}
		}, [surface, destination, native_epoch, aliases, revision = std::move(revision)]() mutable
		{
			surface->lease = destination;
			surface->revision = std::move(revision);
			surface->deferred.clear();
			surface->spill = {};
			surface->initialized = true;
			surface->dirty = true;
			surface->native_write_epoch = native_epoch;
			++surface->generation;
			invalidate_aliases(aliases);
		}, {}, {validation}, aliases);
	}
	void flush_physical(const cached_surface_ref& surface, bool preserve_native = false)
	{
		check(surface);
		synchronize(timeout);
		materialize_physical(surface);
		if (!surface->dirty) return;
		if (surface->faulted) throw std::runtime_error("Cannot publish faulted native physical output");
		const surface_validation validation{surface, {}, true};
		validate_mappings({&validation, 1});
		const auto aliases = physical_aliases(surface);
		if (!surface->lease)
		{
			if (!surface->spill.buffer) throw std::runtime_error("Physical dirty surface lost native output and spill");
			allocate(surface);
		}
		if (surface->spill.buffer)
		{
			const auto snapshot = surface->spill;
			execute([&]
			{
				const auto texture = transfer_texture(surface);
				for (const auto& plane : snapshot.planes)
					buffer_to_texture(ctx, snapshot.buffer, plane.offset, texture, region(surface), plane.pitch, 0, plane.aspect);
				if (surface->description.allocation.samples > 1) samples->import(ctx, texture, surface->lease->render_texture(ctx));
			}, [surface] { surface->initialized = true; surface->spill = {}; }, {}, {validation});
		}
		if (!surface->initialized) throw std::runtime_error("Physical dirty surface has no completed native contents");
		const auto output = make_upload(ctx, read_physical(surface));
		const auto bytes = map_buffer(output, 0, output.length);
		std::vector<physical_surface_write> writes;
		writes.reserve(surface->storage.parts.size());
		for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
		{
			const auto& part = surface->storage.parts[i];
			writes.push_back({(*surface->physical)[i], bytes.subspan(part.offset, part.coverage.range.size)});
		}
		execute([&] { download_guest(surface, output); },
			[this, surface, output, writes = std::move(writes), aliases, preserve_native, virtual_revisions = surface->revision]
			{
				(void)output; // Retain the shared readback buffer through pending CPU publication.
				bool retain = preserve_native;
				if (retain)
				{
					try { check_mapping(surface); }
					catch (const surface_mapping_conflict&) { retain = false; }
				}
				auto publication = memory->compare_write_physical(writes, retain ? std::span<const std::uint64_t>(virtual_revisions) : std::span<const std::uint64_t>{});
				if (!publication) throw surface_conflict_error(surface->layout.range);
				// Receipt storage was prepared before guest mutation. Moving it and
				// publishing metadata cannot allocate or query mappings after the store.
				const bool current = !publication->virtual_revisions.empty();
				if (current) surface->revision = std::move(publication->virtual_revisions);
				surface->dirty = false;
				surface->initialized = current;
				surface->spill = {};
				surface->reset_backing();
				invalidate_aliases(aliases);
			}, {}, {validation}, aliases);
	}
	void flush_aliases(const cached_surface_ref& surface)
	{
		for (const auto& other : surfaces)
			if (other != surface && (other->dirty || !other->deferred.empty()) && memory_storage_overlap(*memory, surface->storage, other->storage) &&
				std::find(resolving.begin(), resolving.end(), other) == resolving.end())
				flush(other);
	}
	void preserve_readers(const surface_lease& allocation, const cached_surface_ref& writer)
	{
		if (!allocation)
			return;
		// Keep the allocation identity even if materializing a reader spills the
		// writer or invalidates its logical guest alias. Never retain an MTL handle.
		const auto captured = allocation;
		for (const auto& other : surfaces)
			if (other != writer && std::find(resolving.begin(), resolving.end(), other) == resolving.end() &&
				std::any_of(other->deferred.begin(), other->deferred.end(), [&](const auto& section)
							{ return section.source == captured; }))
				materialize(other);
	}
	void materialize(const cached_surface_ref& surface)
	{
		if (surface->deferred.empty() || std::find(resolving.begin(), resolving.end(), surface) != resolving.end())
			return;
		if (surface->faulted)
			throw std::runtime_error("Deferred native surface is faulted; overwrite or recover before use");
		if (current(surface) != surface->deferred_revision)
			throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		surface_list_guard resolution(resolving, surface);
		surface_list_guard pin(pinned, surface);
		preserve_readers(surface->lease, surface);
		flush_aliases(surface);
		std::size_t first = 0;
		bool full = false;
		for (std::size_t i = 0; i < surface->deferred.size(); ++i)
		{
			const auto& section = surface->deferred[i];
			if (section.plane == texture_plane::all && section.destination.x == 0 && section.destination.y == 0 &&
				section.destination.width == surface->description.allocation.width && section.destination.height == surface->description.allocation.height)
			{
				first = i;
				full = true;
			}
		}
		std::vector<std::uint64_t> revision;
		if (full)
		{
			revision = current(surface);
			allocate(surface);
		}
		else
		{
			prepare(surface, surface_initialization::guest);
			revision = surface->revision;
		}
		const auto aliases = plan_aliases(surface);
		const auto native_epoch = reserve_native_epoch();
		execute([&]
				{
					const auto destination = surface->lease->render_texture(ctx);
					id<MTLTexture> snapshot = nil;
					// Several queued reads of this allocation must all see the captured
					// epoch, including self-copies followed by other destination writes.
					for (std::size_t i = first; i < surface->deferred.size(); ++i)
						if (surface->deferred[i].source == surface->lease)
						{
							auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:destination.pixelFormat width:destination.width height:destination.height mipmapped:NO];
							descriptor.sampleCount = destination.sampleCount;
							descriptor.textureType = destination.textureType;
							descriptor.storageMode = MTLStorageModePrivate;
							descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
							snapshot = make_texture(ctx, descriptor);
							texture_region area;
							area.size = MTLSizeMake(destination.width, destination.height, 1);
							copy_texture(ctx, destination, area, snapshot, area);
							break;
						}
					for (std::size_t i = first; i < surface->deferred.size(); ++i)
					{
						const auto& section = surface->deferred[i];
						const auto source = section.source == surface->lease ? snapshot : section.source->render_texture(ctx);
						texture_region sr, dr;
						sr.origin = MTLOriginMake(section.rectangle.x, section.rectangle.y, 0);
						dr.origin = MTLOriginMake(section.destination.x, section.destination.y, 0);
						sr.size = MTLSizeMake(section.rectangle.width, section.rectangle.height, 1);
						dr.size = MTLSizeMake(section.destination.width, section.destination.height, 1);
						sample_blits.nearest(ctx, section.source->description().format, source, sr, surface->description.allocation.format, destination, dr, section.plane, section.flip, section.typeless);
					} },
				[surface, revision, aliases, native_epoch]
				{
					surface->deferred.clear();
					surface->revision = revision;
					surface->initialized = true;
					surface->dirty = true;
					surface->native_write_epoch = native_epoch;
					surface->spill = {};
					++surface->generation;
					invalidate_aliases(aliases);
				},
				{surface}, {{surface, revision}}, aliases, false, true);
	}
	void allocate(const cached_surface_ref& surface)
	{
		if (surface->lease)
			return;
		surface->lease = allocate_transient(surface->description.allocation, surface);
	}
	surface_lease allocate_transient(const surface_allocation_description& description, const cached_surface_ref& protected_surface = {})
	{
		for (;;)
		{
			try
			{
				return pool.acquire(description);
			}
			catch (const surface_budget_error& error)
			{
				if (error.requested > error.budget)
					throw;
				if (!snapshots.empty()) { evict_snapshot(std::prev(snapshots.end())); continue; }
				if (recording_batch)
				{
					synchronize(timeout);
					++transactions.draw_resource_retries;
					continue;
				}
				// A local pool quota can fail before the shared ledger is entered.
				// Reconstructible data needs no additional raw spill allocation.
				if (reclaim_clean(protected_surface)) continue;
				cached_surface_ref oldest;
				for (const auto& other : surfaces)
					if (other != protected_surface && other->lease && !(other->faulted && other->dirty) &&
						std::find(pinned.begin(), pinned.end(), other) == pinned.end() && (!oldest || other->last_use < oldest->last_use))
						oldest = other;
				if (!oldest)
					throw;
				spill_surface(oldest);
			}
			catch (const heap_budget_error&)
			{
				if (!recording_batch) throw;
				synchronize(timeout);
				(void)ctx.trim_upload_pages();
				++transactions.draw_resource_retries;
			}
			catch (const native_allocation_error&)
			{
				if (!recording_batch) throw;
				synchronize(timeout);
				(void)ctx.trim_upload_pages();
				++transactions.draw_resource_retries;
			}
		}
	}
	raw_surface_snapshot raw_snapshot(const cached_surface_ref& surface, std::size_t limit = std::numeric_limits<std::size_t>::max())
	{
		raw_surface_snapshot result;
		const auto format = native_format(surface->description.allocation);
		const auto aspects = format == MTLPixelFormatDepth32Float_Stencil8 ? std::vector{texture_plane::depth, texture_plane::stencil} : std::vector{std::holds_alternative<surface_depth>(surface->description.allocation.format) ? texture_plane::depth : texture_plane::all};
		std::size_t size = 0;
		for (const auto aspect : aspects)
		{
			const auto block = storage_block(format, aspect);
			const auto row = checked_texture_product(surface->layout.width, block.bytes);
			if (row > std::numeric_limits<std::size_t>::max() - 255)
				throw std::overflow_error("Native surface spill row overflow");
			const auto pitch = (row + 255) & ~std::size_t(255);
			const auto bytes = checked_texture_product(pitch, surface->layout.height);
			if (size > std::numeric_limits<std::size_t>::max() - bytes)
				throw std::overflow_error("Native surface spill size overflow");
			result.planes.push_back({aspect, size, pitch});
			size += bytes;
		}
		if (size > limit) throw std::runtime_error("Native rollback exceeds bounded raw snapshot budget");
		result.buffer = make_buffer(ctx, size, buffer_storage::shared);
		return result;
	}
	std::shared_ptr<native_rollback> backup_native(std::span<const cached_surface_ref> mutated, bool preserve_contents)
	{
		surface_list_guard pins(pinned, mutated);
		auto result = std::make_shared<native_rollback>();
		result->surfaces.reserve(mutated.size());
		result->preserve_contents = preserve_contents;
		std::size_t remaining = snapshot_limit;
		bool copy = false;
		for (const auto& surface : mutated)
		{
			if (std::any_of(result->surfaces.begin(), result->surfaces.end(), [&](const auto& saved) { return saved.surface == surface; })) continue;
			rollback_surface saved{surface, *surface, surface->spill, false};
			if (surface->initialized && !surface->faulted && !saved.snapshot.buffer)
			{
				// Reserve a private native destination. The old lease is the
				// rollback version; no sample expansion/readback is necessary.
				try { saved.output = pool.acquire(surface->description.allocation); }
				catch (const surface_budget_error&) {} // Bounded legacy raw backup under pressure.
				if (saved.output) ++transactions.native_versions;
				else
				{
					saved.snapshot = raw_snapshot(surface, remaining);
					remaining -= saved.snapshot.buffer.length;
					transactions.raw_backup_bytes += saved.snapshot.buffer.length;
					saved.copy = copy = true;
				}
			}
			result->surfaces.push_back(std::move(saved));
		}
		if (!copy) return result;
		// Same ordered Metal4 queue as the write. This read-only submission
		// has its own status and retains all old resources through completion.
		ctx.begin();
		try
		{
			ctx.keep_alive(result);
			for (const auto& saved : result->surfaces)
			{
				if (!saved.copy) continue;
				const auto texture = transfer_texture(saved.surface);
				if (saved.surface->description.allocation.samples > 1)
					samples->expand(ctx, saved.previous.lease->render_texture(ctx), texture);
				for (const auto& plane : saved.snapshot.planes)
					texture_to_buffer(ctx, texture, region(saved.surface), saved.snapshot.buffer, plane.offset, plane.pitch, 0, plane.aspect);
			}
			result->backup_work = ctx.submit();
			++transactions.raw_backup_submissions;
		}
		catch (...) { ctx.cancel(); throw; }
		return result;
	}
	void spill_surface(const cached_surface_ref& surface)
	{
		check(surface);
		synchronize(timeout);
		surface_list_guard pin(pinned, surface);
		if (!surface->lease)
			return;
		if (surface->faulted && surface->dirty)
			throw std::runtime_error("Cannot spill faulted dirty native contents");
		if (surface->faulted || !surface->initialized || (!surface->dirty && current(surface) != surface->revision))
		{
			surface->lease.reset();
			surface->initialized = false;
			surface->spill = {};
			(void)pool.trim();
			return;
		}
		const auto snapshot = raw_snapshot(surface);
		execute([&]
				{
					const auto texture = transfer_texture(surface);
					if (surface->description.allocation.samples > 1)
						samples->expand(ctx, surface->lease->render_texture(ctx), texture);
					for (const auto& plane : snapshot.planes)
						texture_to_buffer(ctx, texture, region(surface), snapshot.buffer, plane.offset, plane.pitch, 0, plane.aspect); },
				[surface, snapshot]
				{
					surface->spill = snapshot;
					surface->lease.reset();
				}, {}, {{surface, surface->revision}}, {}, true);
	}
	void prepare(const cached_surface_ref& surface, surface_initialization initialization)
	{
		check(surface);
		batch_surface_boundary(surface);
		synchronize(timeout, false);
		surface_list_guard pin(pinned, surface);
		if (surface->faulted)
			throw std::runtime_error("Native cached surface is faulted; restore valid backing before use");
		materialize(surface);
		flush_aliases(surface);
		const auto revision = current(surface);
		if (surface->dirty && surface->revision != revision)
			throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		surface->last_use = ++clock;
		if (surface->initialized && surface->lease && surface->revision == revision)
			return;
		if (surface->spill.buffer && surface->revision != revision)
		{
			surface->spill = {};
			surface->initialized = false;
		}
		allocate(surface);
		preserve_readers(surface->lease, surface);
		if (surface->spill.buffer)
		{
			const auto snapshot = surface->spill;
			execute([&]
					{
						const auto texture = transfer_texture(surface);
						for (const auto& plane : snapshot.planes)
							buffer_to_texture(ctx, snapshot.buffer, plane.offset, texture, region(surface), plane.pitch, 0, plane.aspect);
						if (surface->description.allocation.samples > 1)
							samples->import(ctx, texture, surface->lease->render_texture(ctx)); },
					[surface]
					{
						surface->spill = {};
						surface->initialized = true;
					},
					{surface});
			return;
		}
		if (initialization == surface_initialization::background)
		{
			const auto aliases = plan_aliases(surface);
			const auto native_epoch = reserve_native_epoch();
			execute([&]
					{
						MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
						const auto texture = surface->lease->render_texture(ctx);
						if (std::holds_alternative<surface_color>(surface->description.allocation.format))
						{
							pass.colorAttachments[0].texture = texture;
							pass.colorAttachments[0].loadAction = MTLLoadActionClear;
							pass.colorAttachments[0].storeAction = MTLStoreActionStore;
							pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
						}
						else
						{
							pass.depthAttachment.texture = texture;
							pass.depthAttachment.loadAction = MTLLoadActionClear;
							pass.depthAttachment.storeAction = MTLStoreActionStore;
							pass.depthAttachment.clearDepth = 1;
							if (describe_surface_depth(std::get<surface_depth>(surface->description.allocation.format)).stencil)
							{
								pass.stencilAttachment.texture = texture;
								pass.stencilAttachment.loadAction = MTLLoadActionClear;
								pass.stencilAttachment.storeAction = MTLStoreActionStore;
								pass.stencilAttachment.clearStencil = 255;
							}
						}
						(void)ctx.render(pass);
						ctx.end_encoding(); },
					[surface, revision, aliases, native_epoch]
					{
						surface->revision = revision;
						surface->initialized = true;
						surface->dirty = true;
						surface->native_write_epoch = native_epoch;
						++surface->generation;
						invalidate_aliases(aliases);
					},
					{surface}, {{surface, revision}}, aliases, false, true);
			return;
		}
		const auto snapshot = read(surface);
		if (snapshot.bytes.size() != surface->layout.range.size)
			throw std::runtime_error("VM surface snapshot has the wrong byte count");
		const auto input = make_upload(ctx, snapshot.bytes);
		execute([&]
				{ upload_guest(surface, input); },
				[surface, revision = snapshot.revisions]
				{
					surface->revision = revision;
					surface->initialized = true;
					surface->dirty = false;
					surface->reset_backing();
					surface->native_write_epoch = 0;
					++surface->generation;
				},
				{surface}, {{surface, snapshot.revisions}});
	}
};

surface_cache::surface_cache(context& ctx, std::shared_ptr<guest_surface_memory> memory, std::size_t heap_budget, std::shared_ptr<metal4::heap_budget> shared_budget, cpu_capture_dispatch capture, surface_metadata_limits metadata)
	: m_impl(std::make_unique<impl>(ctx, std::move(memory), heap_budget, std::move(shared_budget), std::move(capture), metadata))
{
}
surface_cache::~surface_cache() { m_impl->snapshot_lifetime.reset(); m_impl->cancel_draw_batch(); }
surface_cache::draw_batch::draw_batch(surface_cache& owner, std::size_t capacity)
	: m_owner(&owner), m_lifetime(owner.m_impl->batch_lifetime)
{
	owner.m_impl->outside_epoch();
	if (!capacity || capacity > 256) throw std::invalid_argument("Native draw batch capacity must be in[1,256]");
	if (owner.m_impl->batch_capacity) throw std::logic_error("Native draw batch scopes cannot nest");
	owner.m_impl->synchronize(owner.m_impl->timeout);
	owner.m_impl->batch_capacity = capacity;
}
surface_cache::draw_batch::~draw_batch() { if (m_owner && !m_lifetime.expired()) m_owner->m_impl->cancel_draw_batch(); }
void surface_cache::draw_batch::finish()
{
	if (!m_owner || m_lifetime.expired()) throw std::logic_error("Native draw batch finished or owner destroyed");
	m_owner->m_impl->outside_epoch();
	auto* owner = m_owner;
	m_owner = nullptr;
	owner->m_impl->batch_capacity = 0;
	owner->m_impl->synchronize(owner->m_impl->timeout);
}
cached_surface_ref surface_cache::acquire(const cached_surface_description& description)
{
	m_impl->outside_epoch();
	m_impl->synchronize(m_impl->timeout, false);
	auto normalized = description;
	if (normalized.allocation.samples != 1 && normalized.allocation.samples != 2 && normalized.allocation.samples != 4)
		throw std::invalid_argument("Native cached surface samples are invalid");
	normalized.allocation.expanded = normalized.allocation.samples > 1;
	const auto size = normalized.allocation.samples == 1 ? std::array{normalized.allocation.width, normalized.allocation.height} : sample_expanded_size(normalized.allocation.width, normalized.allocation.height, normalized.allocation.samples);
	const auto limit = max_2d_texture_dimension(m_impl->ctx.device());
	if (size[0] > limit || size[1] > limit)
		throw std::invalid_argument("Cached expanded dimensions exceed Apple GPU family limit");
	const bool segmented = !normalized.segments.empty();
	if ((segmented || normalized.mapping) && !m_impl->capture)
		throw std::invalid_argument("Mapped/disjoint surfaces require a common CPU capture provider");
	if (normalized.tile && !normalized.morton)
	{
		if (normalized.row_pitch && normalized.row_pitch != normalized.tile->pitch)
			throw std::invalid_argument("Banked surface row pitch differs from tile pitch");
		normalized.row_pitch = normalized.tile->pitch;
	}
	auto layout = plan_guest_surface(segmented ? 0 : normalized.address, size[0], size[1], pixel_bytes(normalized.allocation), normalized.row_pitch, normalized.morton);
	if (!layout.morton)
		normalized.row_pitch = layout.row_pitch;
	if (normalized.tile)
	{
		if (normalized.tile->size > m_impl->snapshot_limit) throw std::length_error("Banked surface snapshot exceeds cache byte limit");
		if (segmented)
		{
			layout.range = {normalized.tile->address, normalized.tile->size};
			validate_banked_tile_geometry(layout, *normalized.tile);
			validate_guest_segments(layout, normalized.segments);
			const auto origin = normalized.tile->offset;
			const auto part = std::find_if(normalized.segments.begin(), normalized.segments.end(), [&](const auto& segment)
				{ return origin >= segment.offset && origin - segment.offset < segment.size; });
			if (part == normalized.segments.end() || std::uint64_t(part->address) + origin - part->offset != normalized.address)
				throw std::invalid_argument("Banked surface pixel origin differs from actual storage segment");
		}
		else layout = attach_banked_tile(std::move(layout), *normalized.tile);
	}
	else if (segmented)
	{
		layout.range.address = normalized.address;
		validate_guest_segments(layout, normalized.segments);
		if (layout.range.size > m_impl->snapshot_limit) throw std::length_error("Disjoint surface snapshot exceeds cache byte limit");
	}
	if (normalized.mapping)
	{
		const auto& mapped = normalized.mapping->state();
		if (mapped.pixel_address != normalized.address || mapped.tile != normalized.tile || mapped.segments != normalized.segments || mapped.logical.size != layout.range.size)
			throw std::invalid_argument("Surface descriptor differs from owned framebuffer mapping");
		m_impl->cpu([&] { if (!normalized.mapping->current()) throw surface_mapping_conflict(); });
	}
	const auto hash = descriptor_hash(normalized);
	++m_impl->metadata.descriptor_lookups;
	const auto bucket = m_impl->descriptors.equal_range(hash);
	for (auto i = bucket.first; i != bucket.second; ++i)
	{
		++m_impl->metadata.descriptor_comparisons;
		if (same(i->second->description, normalized))
		{
			i->second->last_use = ++m_impl->clock;
			return m_impl->surfaces[m_impl->positions.at(i->second)];
		}
	}
	m_impl->synchronize(m_impl->timeout);
	auto surface = std::make_shared<cached_surface>();
	surface->description = normalized;
	surface->layout = layout;
	surface->storage.size = layout.range.size;
	if (segmented)
		surface->storage.parts = project_guest_segments(layout, normalized.segments, bool(normalized.tile), m_impl->metadata.limits.watch_parts);
	else surface->storage.parts.push_back({normalized.tile ? guest_byte_coverage(layout.range) : layout, 0});
	surface->revision.resize(surface->storage.parts.size());
	m_impl->reserve_metadata(surface->storage.parts.size());
	m_impl->cpu([&]
	{
		impl::check_mapping(surface);
		for (auto& part : surface->storage.parts) part.coverage = m_impl->memory->watch_surface(part.coverage);
		impl::check_mapping(surface);
	});
	m_impl->surfaces.push_back(surface);
	try
	{
		m_impl->positions.emplace(surface.get(), m_impl->surfaces.size() - 1);
		m_impl->descriptors.emplace(hash, surface.get());
	}
	catch (...)
	{
		m_impl->positions.erase(surface.get());
		m_impl->surfaces.pop_back();
		throw;
	}
	surface->last_use = ++m_impl->clock;
	m_impl->ranges_dirty = true;
	m_impl->metadata.entries = m_impl->surfaces.size();
	m_impl->metadata.watch_parts += surface->storage.parts.size();
	return surface;
}
cached_surface_status surface_cache::status(const cached_surface_ref& surface) const
{
	m_impl->check(surface);
	return {surface->initialized, surface->dirty, bool(surface->spill.buffer), surface->faulted, bool(m_impl->pending || m_impl->recording_batch), surface->revision.size() == 1 ? surface->revision[0] : 0, surface->generation, surface->deferred.size(), surface->native_write_epoch, surface->revision};
}
void surface_cache::prepare(const cached_surface_ref& surface, surface_initialization initialization)
{
	m_impl->prepare(surface, initialization);
}
cached_surface_description surface_cache::description(const cached_surface_ref& surface) const
{
	m_impl->check(surface);
	return surface->description;
}
void surface_cache::clear(const cached_surface_ref& surface, std::uint32_t mask, std::uint32_t color, std::uint32_t zstencil, std::uint32_t stencil_mask, clear_rectangle rectangle)
{
	m_impl->check(surface);
	m_impl->batch_surface_boundary(surface);
	m_impl->synchronize(m_impl->timeout, false);
	const auto clipped = clip_clear_rectangle(rectangle, surface->description.allocation.width, surface->description.allocation.height);
	if (!clipped.width || !clipped.height)
		return;
	bool changed = false, full = false;
	if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
	{
		const auto plan = decode_color_clear(*format, color, mask);
		changed = plan.write_mask != 0;
		full = changed && plan.write_mask == decode_color_clear(*format, color, 0xf0).write_mask;
	}
	else
	{
		const auto depth = describe_surface_depth(std::get<surface_depth>(surface->description.allocation.format));
		changed = (mask & 1) || ((mask & 2) && depth.stencil && (stencil_mask & 255));
		full = (mask & 1) && (!depth.stencil || ((mask & 2) && (stencil_mask & 255) == 255));
	}
	if (!changed)
		return;
	const auto clear_upload = clear_engine::upload_for(surface->description.allocation.format, mask, stencil_mask,
		clipped.x == 0 && clipped.y == 0 && clipped.width == surface->description.allocation.width && clipped.height == surface->description.allocation.height);
	std::optional<upload_plan> uploads;
	if (clear_upload)
		prepare_draw_resources([&] { uploads = m_impl->ctx.prepare_uploads(std::span{&*clear_upload, 1}); });
	m_impl->preserve_readers(surface->lease, surface);
	surface->last_use = ++m_impl->clock;
	full &= clipped.x == 0 && clipped.y == 0 && clipped.width == surface->description.allocation.width && clipped.height == surface->description.allocation.height;
	std::vector<std::uint64_t> revision;
	if (full)
	{
		m_impl->flush_aliases(surface);
		revision = m_impl->current(surface);
		m_impl->allocate(surface);
	}
	else
	{
		m_impl->prepare(surface, surface_initialization::guest);
		revision = surface->revision;
	}
	const auto aliases = m_impl->plan_aliases(surface);
	const auto native_epoch = m_impl->reserve_native_epoch();
	m_impl->execute([&]
					{
						clear_request request;
						request.mask = mask;
						request.color_value = color;
						request.zstencil_value = zstencil;
						request.stencil_write_mask = stencil_mask;
						request.rectangle = clipped;
						if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
							request.colors[0] = {{surface->lease->render_texture(m_impl->ctx), 0, 0}, *format};
						else
						{
							request.depth = {surface->lease->render_texture(m_impl->ctx), 0, 0};
							request.depth_format = std::get<surface_depth>(surface->description.allocation.format);
						}
						if (uploads) m_impl->ctx.with_uploads(*uploads, [&] { (void)m_impl->clears.run(m_impl->ctx, request); });
						else (void)m_impl->clears.run(m_impl->ctx, request); },
					[this, surface, revision, aliases, native_epoch]
					{
						surface->revision = revision;
						surface->initialized = true;
						surface->dirty = true;
						surface->native_write_epoch = native_epoch;
						surface->faulted = false;
						surface->spill = {};
						surface->deferred.clear();
						++surface->generation;
						m_impl->invalidate_aliases(aliases);
					},
					{surface}, {{surface, revision}}, aliases, false, true, !full, true);
}
void surface_cache::draw(std::span<const cached_surface_use> uses, const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode)
{
	draw_impl(uses, encode, false);
}
void surface_cache::draw_feedback(std::span<const cached_surface_use> uses, const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode)
{
	draw_impl(uses, encode, true);
}
void surface_cache::draw_depth_bounds(std::span<const cached_surface_use> uses, std::size_t depth_index,
									  const std::function<MTL4RenderPassDescriptor*(std::span<const id<MTLTexture>>)>& descriptor,
									  const std::function<void(context&, std::span<const id<MTLTexture>>, depth_bounds_pass&)>& encode, bool feedback)
{
	if (depth_index >= uses.size() || !uses[depth_index].write || !descriptor || !encode)
		throw std::invalid_argument("Cached depth bounds require a depth write target and both callbacks");
	m_impl->check(uses[depth_index].surface);
	const auto& allocation = uses[depth_index].surface->description.allocation;
	if (!std::holds_alternative<surface_depth>(allocation.format))
		throw std::invalid_argument("Cached depth bounds target must have a depth format");
	if (!m_impl->bounds)
		m_impl->bounds = std::make_unique<depth_bounds_engine>(m_impl->ctx);
	const surface_allocation_description mirror_description{allocation.width, allocation.height, allocation.samples, surface_color::r32f, false};
	surface_lease mirror;
	draw_impl(uses, [&](context& ctx, auto textures)
			  {
				  MTL4RenderPassDescriptor* pass = descriptor(textures);
				  if (!pass || pass.depthAttachment.texture != textures[depth_index] || pass.depthAttachment.level || pass.depthAttachment.slice)
					  throw std::invalid_argument("Cached depth bounds descriptor must select the supplied depth write texture");
				  auto prepared = m_impl->bounds->prepare(ctx, pass, mirror);
				  encode(ctx, textures, prepared);
				  prepared.end(); }, feedback, &mirror_description, &mirror);
}
std::optional<surface_cache::scaled_image_endpoints> surface_cache::prepare_scaled_image(const scaled_image_plan& plan,
	std::chrono::milliseconds timeout)
{
	m_impl->outside_epoch();
	// The closed native route publishes metadata under the cache's producer
	// exclusion. Legacy callers without that exclusion retain the VM route.
	if(!m_impl->capture)return {};
	prepare_texture_boundary(true);m_impl->synchronize(timeout);
	scaled_image_endpoints result;
	bool complete=true;
	m_impl->cpu([&]
	{
		for(unsigned side=0;side<2&&complete;++side)
		{
			const auto& image=side?plan.destination:plan.source;
			const auto requested=scaled_image_storage(image,side?plan.destination_tile:plan.source_tile,
				side?plan.destination_segments:plan.source_segments);
			const auto projected=project_display_storage_source(requested);
			std::vector<guest_surface_layout> coverage;for(const auto& part:projected)coverage.push_back(part.coverage);
			const auto candidates=m_impl->cpu_aliases(coverage,true);
			const auto dirty=std::count_if(candidates.begin(),candidates.end(),[](const auto& s){return s->dirty||!s->deferred.empty();});
			if(dirty>1){complete=false;break;}
			const auto expected=(side?plan.destination_format:plan.source_format)==scaled_image_format::rgb565?surface_color::rgb565:surface_color::argb8;
			for(const auto& surface:candidates)
			{
				if(dirty&&!surface->dirty&&surface->deferred.empty())continue;
				m_impl->check_mapping(surface);
				if(!surface->initialized||surface->faulted||!surface->deferred.empty()||(!surface->lease&&!surface->spill.buffer))continue;
				if(m_impl->current(surface)!=surface->revision)
				{
					if(surface->dirty)throw surface_conflict_error(surface->storage.parts.front().coverage.range);
					continue;
				}
				if(surface->description.allocation.format!=surface_format{expected})continue;
				const auto& description=surface->description;
				const auto size=description.allocation.samples==1?std::array{description.allocation.width,description.allocation.height}:
					sample_expanded_size(description.allocation.width,description.allocation.height,description.allocation.samples);
				auto geometry=plan_guest_surface(description.tile?description.tile->address:description.address,size[0],size[1],image.pixel_bytes,
					description.morton?0:description.row_pitch,description.morton);
				if(description.tile)geometry.range={description.tile->address,description.tile->size};
				const auto storage=scaled_image_storage(geometry,description.tile,description.segments);
				const auto region=plan_native_scaled_image_region(*m_impl->memory,storage,requested);
				if(!region)continue;
				if(result.surfaces[side]&&result.surfaces[side]->native_write_epoch>=surface->native_write_epoch)continue;
				result.surfaces[side]=surface;result.regions[side]=*region;
			}
			complete=bool(result.surfaces[side]);
		}
	});
	return complete?std::optional{std::move(result)}:std::nullopt;
}
void surface_cache::draw_scaled_image(const scaled_image_endpoints& endpoints,
	const std::function<void(context&,std::span<const id<MTLTexture>>)>& encode,
	std::function<void()> publication_check,std::chrono::milliseconds timeout)
{
	m_impl->outside_epoch();
	const auto previous=m_impl->timeout;
	struct restore_timeout{impl& state;std::chrono::milliseconds value;~restore_timeout(){state.timeout=value;}} restore{*m_impl,previous};
	m_impl->timeout=timeout;
	const std::array uses{cached_surface_use{endpoints.surfaces[0],false},cached_surface_use{endpoints.surfaces[1],true}};
	draw_impl(uses,encode,true,nullptr,nullptr,std::move(publication_check));
}
void surface_cache::draw_impl(std::span<const cached_surface_use> uses, const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode, bool feedback,
							  const surface_allocation_description* transient, surface_lease* reservation,std::function<void()> publication_check)
{
	if(publication_check)m_impl->cpu(publication_check);
	// Sixteen fragment inputs, four vertex inputs, four color targets and depth.
	if (uses.empty() || uses.size() > 25 || !encode || std::count_if(uses.begin(), uses.end(), [](const auto& use)
																	 { return use.write; }) > 5)
		throw std::invalid_argument("Native cache draw needs at most 25 uses/five write targets and an encoder");
	for (std::size_t i = 0; i < uses.size(); ++i)
	{
		m_impl->check(uses[i].surface);
		for (std::size_t j = 0; j < i; ++j)
			if ((feedback ? uses[i].write && uses[j].write : uses[i].write || uses[j].write) && memory_storage_overlap(*m_impl->memory, uses[i].surface->storage, uses[j].surface->storage))
				throw std::invalid_argument("Native cache draw write targets overlap guest storage");
	}
	if (std::any_of(uses.begin(), uses.end(), [](const auto& use) { return !use.write; }))
		m_impl->synchronize(m_impl->timeout);
	for (const auto& use : uses)
		if (use.write)
			m_impl->preserve_readers(use.surface->lease, use.surface);
	std::vector<cached_surface_ref> pins;
	for (const auto& use : uses)
		pins.push_back(use.surface);
	auto previous_pins = m_impl->pinned;
	m_impl->pinned.insert(m_impl->pinned.end(), pins.begin(), pins.end());
	std::vector<surface_lease> snapshots(uses.size());
	try
	{
		// Capture readers before writer preparation can invalidate/reinterpret
		// their guest aliases. All required allocations stay pinned until copied.
		for (const auto& use : uses)
			if (!use.write)
				m_impl->prepare(use.surface, surface_initialization::guest);
		std::vector<std::size_t> copies;
		if (feedback)
			for (std::size_t i = 0; i < uses.size(); ++i)
			{
				if (uses[i].write || !std::any_of(uses.begin(), uses.end(), [&](const auto& other)
												  { return other.write && memory_storage_overlap(*m_impl->memory, uses[i].surface->storage, other.surface->storage); }))
					continue;
				for (std::size_t j = 0; j < i; ++j)
					if (!uses[j].write && uses[j].surface == uses[i].surface)
					{
						snapshots[i] = snapshots[j];
						break;
					}
				if (snapshots[i])
					continue;
				auto description = uses[i].surface->description.allocation;
				description.expanded = false; // Capture native samples, without a second guest transfer backing.
				snapshots[i] = m_impl->allocate_transient(description);
				copies.push_back(i);
			}
		if (!copies.empty())
		{
			std::vector<surface_validation> captured;
			for (const auto i : copies) captured.push_back({uses[i].surface, uses[i].surface->revision});
			m_impl->execute([&]
							{
								for (const auto i : copies)
								{
									texture_region whole;
									const auto& description = uses[i].surface->description.allocation;
									whole.size = MTLSizeMake(description.width, description.height, 1);
									copy_texture(m_impl->ctx, uses[i].surface->lease->render_texture(m_impl->ctx), whole, snapshots[i]->render_texture(m_impl->ctx), whole);
								} },
							[] {}, {}, std::move(captured));
		}
		for (const auto& use : uses)
			if (use.write)
				m_impl->prepare(use.surface, surface_initialization::guest);
		if (transient)
			*reservation = m_impl->allocate_transient(*transient);
	}
	catch (...)
	{
		m_impl->pinned = std::move(previous_pins);
		throw;
	}
	m_impl->pinned = std::move(previous_pins);
	std::vector<cached_surface_ref> written;
	std::vector<cached_surface_ref> aliases;
	for (const auto& use : uses)
		if (use.write)
		{
			written.push_back(use.surface);
			const auto affected = m_impl->plan_aliases(use.surface);
			aliases.insert(aliases.end(), affected.begin(), affected.end());
		}
	const auto native_epoch = written.empty() ? 0 : m_impl->reserve_native_epoch();
	const bool batchable=!publication_check;
	m_impl->execute([&]
					{
						std::vector<id<MTLTexture>> textures;
						for (std::size_t i = 0; i < uses.size(); ++i)
						{
							if (snapshots[i])
							{
								textures.push_back(snapshots[i]->render_texture(m_impl->ctx));
							}
							else
								textures.push_back(uses[i].surface->lease->render_texture(m_impl->ctx));
						}
						encode(m_impl->ctx, textures); },
					[this, written, aliases, native_epoch,check=std::move(publication_check)]
					{
						if(check)check();
						for (const auto& surface : written)
						{
							surface->dirty = true;
							surface->native_write_epoch = native_epoch;
							++surface->generation;
						}
						m_impl->invalidate_aliases(aliases);
					},
					written, [&]
					{
						std::vector<surface_validation> values;
						for (const auto& use : uses) values.push_back({use.surface, use.surface->revision});
						return values;
					}(), aliases, false, !written.empty(), true, batchable);
}
void surface_cache::copy(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
						 std::size_t x, std::size_t y, texture_plane plane)
{
	blit_nearest(source, rectangle, destination, {x, y, rectangle.width, rectangle.height}, plane);
}
void surface_cache::blit_nearest(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
								 clear_rectangle target, texture_plane plane, std::array<bool, 3> flip)
{
	transfer(source, rectangle, destination, target, plane, flip, false);
}
void surface_cache::blit_typeless(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
								  clear_rectangle target, std::array<bool, 3> flip)
{
	transfer(source, rectangle, destination, target, texture_plane::all, flip, true);
}
void surface_cache::transfer(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
							 clear_rectangle target, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	m_impl->check(source);
	m_impl->check(destination);
	m_impl->synchronize(m_impl->timeout);
	const auto& a = source->description.allocation;
	const auto& b = destination->description.allocation;
	if (!rectangle.width || !rectangle.height || !target.width || !target.height)
		return;
	validate_surface_blit(a, rectangle, b, target, plane, flip, typeless);
	m_impl->preserve_readers(destination->lease, destination);
	std::vector<cached_surface_ref> pins{source, destination};
	auto previous_pins = m_impl->pinned;
	m_impl->pinned.insert(m_impl->pinned.end(), pins.begin(), pins.end());
	const bool full = plane == texture_plane::all && target.x == 0 && target.y == 0 && target.width == b.width && target.height == b.height && source != destination;
	std::vector<std::uint64_t> revision;
	try
	{
		m_impl->prepare(source, surface_initialization::guest);
		if (full)
		{
			m_impl->flush_aliases(destination);
			revision = m_impl->current(destination);
			m_impl->allocate(destination);
		}
		else
		{
			m_impl->prepare(destination, surface_initialization::guest);
			revision = destination->revision;
		}
	}
	catch (...)
	{
		m_impl->pinned = std::move(previous_pins);
		throw;
	}
	m_impl->pinned = std::move(previous_pins);
	source->last_use = destination->last_use = ++m_impl->clock;
	const auto aliases = m_impl->plan_aliases(destination);
	const auto native_epoch = m_impl->reserve_native_epoch();
	m_impl->execute([&]
					{
						texture_region sr, dr;
						sr.origin = MTLOriginMake(rectangle.x, rectangle.y, 0);
						dr.origin = MTLOriginMake(target.x, target.y, 0);
						sr.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
						dr.size = MTLSizeMake(target.width, target.height, 1);
						m_impl->sample_blits.nearest(m_impl->ctx, a.format, source->lease->render_texture(m_impl->ctx), sr,
													 b.format, destination->lease->render_texture(m_impl->ctx), dr, plane, flip, typeless); },
					[this, destination, revision, aliases, native_epoch]
					{
						destination->revision = revision;
						destination->initialized = true;
						destination->dirty = true;
						destination->native_write_epoch = native_epoch;
						destination->faulted = false;
						destination->spill = {};
						destination->deferred.clear();
						++destination->generation;
						m_impl->invalidate_aliases(aliases);
					},
					{destination}, {{source, source->revision}, {destination, revision}}, aliases, false, true);
}
void surface_cache::defer_copy(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
							   std::size_t x, std::size_t y, texture_plane plane)
{
	defer_blit_nearest(source, rectangle, destination, {x, y, rectangle.width, rectangle.height}, plane);
}
void surface_cache::defer_blit_nearest(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
									   clear_rectangle target, texture_plane plane, std::array<bool, 3> flip)
{
	defer_transfer(source, rectangle, destination, target, plane, flip, false);
}
void surface_cache::defer_blit_typeless(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
										clear_rectangle target, std::array<bool, 3> flip)
{
	defer_transfer(source, rectangle, destination, target, texture_plane::all, flip, true);
}
void surface_cache::defer_transfer(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
								   clear_rectangle target, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	m_impl->check(source);
	m_impl->check(destination);
	m_impl->synchronize(m_impl->timeout);
	if (!rectangle.width || !rectangle.height || !target.width || !target.height)
		return;
	validate_surface_blit(source->description.allocation, rectangle, destination->description.allocation, target, plane, flip, typeless);
	m_impl->prepare(source, surface_initialization::guest);
	auto revision = m_impl->current(destination);
	if ((destination->dirty && revision != destination->revision) ||
		(!destination->deferred.empty() && revision != destination->deferred_revision))
		throw surface_conflict_error(destination->storage.parts.front().coverage.range);
	const surface_validation validation{destination, revision};
	const auto retained = m_impl->retain_backings({&validation, 1});
	auto plans = destination->deferred;
	plans.push_back({source->lease, source->generation, rectangle, target, plane, flip, typeless});
	// A new deferred owner can cover a former VM hole in an assembled image.
	// Invalidate its source-set identity now,not only when materialized later.
	m_impl->texture_topology_epoch = m_impl->reserve_native_epoch();
	destination->deferred = std::move(plans);
	destination->deferred_revision = std::move(revision);
	destination->backing = retained.front().owner;
	destination->physical = retained.front().physical;
}
void surface_cache::cancel_deferred_copies(const cached_surface_ref& surface)
{
	m_impl->check(surface);
	m_impl->synchronize(m_impl->timeout);
	if(!surface->deferred.empty())m_impl->texture_topology_epoch = m_impl->reserve_native_epoch();
	surface->deferred.clear();
	if (!surface->dirty) surface->reset_backing();
	(void)m_impl->pool.trim();
}
void surface_cache::flush(const cached_surface_ref& surface)
{
	m_impl->flush(surface);
}
void surface_cache::flush_physical(const cached_surface_ref& surface)
{
	m_impl->flush_physical(surface);
}
std::size_t surface_cache::prepare_physical_fault(const physical_identity& identity,
	std::span<const std::shared_ptr<void>> registrations, bool writing)
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native cache fault preparation cannot wait inside VM exclusion");
	m_impl->synchronize(m_impl->timeout);
	std::vector<cached_surface_ref> selected, outputs;
	m_impl->cpu([&]
	{
		for (const auto& surface : m_impl->surfaces)
		{
			bool matches = false;
			if (surface->dirty || !surface->deferred.empty())
			{
				if (!surface->physical || surface->physical->size() != surface->storage.parts.size())
					throw std::runtime_error("Native fault discovery cannot omit untyped dirty output");
				for (const auto& ticket : *surface->physical)
					for (const auto& part : ticket->parts())
						if (identity.overlaps(*part)) matches = true;
			}
			else
				for (const auto& part : surface->storage.parts)
					if (memory_physical_fault_overlap(*m_impl->memory, identity, registrations, part.coverage)) { matches = true; break; }
			if (!matches) continue;
			selected.push_back(surface);
			if (surface->dirty || !surface->deferred.empty()) outputs.push_back(surface);
		}
		// Preflight every selected authority before the first GPU writeback.
		// Later host writes can still make a CAS fail;already valid publications
		// remain committed and all remaining owners/protection survive for retry.
		for (const auto& surface : outputs)
		{
			if (surface->faulted) throw std::runtime_error("Native fault discovery found faulted dirty output");
			const surface_validation validation{surface, {}, true};
			m_impl->validate_in_epoch({&validation, 1});
			(void)m_impl->physical_aliases(surface);
		}
	});
	for (const auto& surface : outputs) m_impl->flush_physical(surface, !writing);
	if (writing)
		for (const auto& surface : selected)
		{
			surface->initialized = false;
			surface->spill = {};
			surface->native_write_epoch = 0;
			surface->reset_backing();
		}
	return outputs.size();
}
std::size_t surface_cache::invalidate_mapping(guest_surface_range range, surface_mapping_event cause)
{
	validate_guest_surface_range(range);
	if (cause != surface_mapping_event::read && cause != surface_mapping_event::unmap)
		throw std::invalid_argument("Invalid native surface mapping event");
	if (guest_memory_epoch::active()) throw std::logic_error("Native mapping recovery cannot run inside VM exclusion");
	if (!range.size) return 0;
	m_impl->synchronize(m_impl->timeout);
	const auto coverage = guest_byte_coverage(range);
	std::vector<cached_surface_ref> selected;
	m_impl->cpu([&]
	{
		for (const auto& surface : m_impl->surfaces)
		{
			if (!surface->dirty && surface->deferred.empty()) continue;
			bool matches = std::any_of(surface->storage.parts.begin(), surface->storage.parts.end(),
				[&](const auto& part) { return overlaps(coverage, part.coverage); });
			if (!matches)
			{
				if (!surface->physical) throw std::runtime_error("Native mapping selection lacks retained physical tickets");
				for (const auto& ticket : *surface->physical)
				{
					const auto alias = m_impl->memory->physical_surface_overlaps_current(*ticket, range);
					if (!alias) throw std::runtime_error("Native mapping selection lacks current allocation metadata");
					if (*alias) { matches = true; break; }
				}
			}
			if (matches) selected.push_back(surface);
		}
	});
	for (const auto& surface : selected)
		m_impl->flush_physical(surface, cause == surface_mapping_event::read);
	if (cause == surface_mapping_event::unmap)
	{
		// Clean mirrors need no old-VA lookup. Conservatively retire all of them
		// only after every selected output was reconciled successfully. Resource
		// release stays outside provider exclusion;dirty/deferred owners survive.
		for (const auto& surface : m_impl->surfaces)
			if (!surface->dirty && surface->deferred.empty())
			{
				surface->initialized = false;
				surface->spill = {};
				surface->native_write_epoch = 0;
				surface->reset_backing();
			}
	}
	const auto count = selected.size();
	selected.clear();
	if (cause == surface_mapping_event::unmap) m_impl->retire_to(0, 0);
	return count;
}
void surface_cache::spill(const cached_surface_ref& surface)
{
	m_impl->spill_surface(surface);
}
void surface_cache::recover(const cached_surface_ref& surface)
{
	m_impl->check(surface);
	m_impl->synchronize(m_impl->timeout);
	if (surface->faulted && surface->dirty && !surface->spill.buffer)
		throw std::runtime_error("Faulted dirty surface has no recoverable backing");
	m_impl->ctx.recover_completed_errors();
	if (surface->faulted)
	{
		surface->faulted = false;
		surface->initialized = false;
	}
}
void surface_cache::erase(const cached_surface_ref& surface)
{
	m_impl->outside_epoch();
	m_impl->flush(surface);
	surface->lease.reset();
	surface->spill = {};
	surface->reset_backing();
	m_impl->remove(surface);
	(void)m_impl->pool.trim();
}
void surface_cache::prepare_texture_boundary(bool gpu_work)
{
	if (m_impl->owner_thread != std::this_thread::get_id()) throw std::logic_error("Native texture boundary belongs to another render thread");
	m_impl->outside_epoch();
	if (m_impl->recording_batch)
	{
		const auto& recording = m_impl->recording_batch->recording;
		if (!recording || !recording->same_batch(m_impl->ctx.recording_submission()))
			throw std::logic_error("Native surface batch no longer owns the current recording");
		if (!gpu_work) return;
		m_impl->flush_draw_batch();
	}
	m_impl->ctx.validate_begin();
}
void surface_cache::validate_texture_adapter(context& ctx, const guest_surface_memory& memory) const
{
	if (&ctx != &m_impl->ctx || &memory != m_impl->memory.get())
		throw std::invalid_argument("Native texture/surface caches require the same context and VM adapter");
}
void surface_cache::prepare_texture_read_boundary(std::span<const guest_surface_layout> coverage)
{
	prepare_texture_boundary(false);
	if (coverage.empty() || coverage.size() > (1u << 20))
		throw std::invalid_argument("Native texture read coverage count is invalid");
	for (const auto& part : coverage) { validate_guest_surface_range(part.range); (void)guest_surface_byte_cursor(part); }
	bool hazard = false;
	if (const auto batch = m_impl->recording_batch)
		m_impl->cpu([&]
		{
			m_impl->memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
			{
				for (const auto& surface : batch->mutated)
					for (const auto& part : coverage)
						if (active_storage_overlap(direct, lookup, part, surface->storage)) hazard = true;
				return false;
			});
			// A remapped VA need not identify the allocation retained by a draw.
			// Unknown physical identity cannot justify skipping its boundary.
			for (const auto& backing : batch->backings)
				if (!hazard && backing.physical && std::find(batch->mutated.begin(), batch->mutated.end(), backing.surface) != batch->mutated.end())
					for (const auto& ticket : *backing.physical)
						for (const auto& part : coverage)
						{
							guest_surface_byte_cursor cursor(part);
							while (const auto run = cursor.next())
							{
								const auto overlap = m_impl->memory->physical_surface_overlaps_current(*ticket, *run);
								if (!overlap || *overlap) { hazard = true; break; }
							}
						}
		});
	// Already submitted transactions still publish through normal completion.
	// An unrelated,unsubmitted draw recording remains owned by its batch.
	m_impl->synchronize(m_impl->timeout, hazard);
}
struct prepared_surface_snapshot::state
{
	surface_cache::impl* cache = nullptr;
	std::weak_ptr<void> lifetime;
	std::uint64_t cache_generation = 0;
	std::thread::id thread = std::this_thread::get_id();
	context* context_owner = nullptr;
	cached_surface_ref source;
	std::weak_ptr<cached_surface> authority;
	std::uint64_t authority_generation = 0, authority_epoch = 0;
	surface_lease allocation;
	std::vector<surface_backing> backings;
	std::shared_ptr<sample_transfer> samples;
	std::list<surface_cache::impl::snapshot_entry> candidate;
	id<MTLTexture> image;
	cached_surface_description description;
	clear_rectangle rectangle{};
	surface_format format;
	std::size_t prefix = 0;
	std::shared_ptr<const surface_texture_revision> receipt;
	std::optional<submission> encoded;
	bool copy = false, complete = false, published = false;
	std::uint64_t views_created = 0, views_reused = 0;
	void outside() const
	{
		if (thread != std::this_thread::get_id() || guest_memory_epoch::active() || surface_cache::impl::captured)
			throw std::logic_error("Native snapshot encoding/publication requires its owner outside CPU exclusion");
	}
};
prepared_surface_snapshot::prepared_surface_snapshot(std::shared_ptr<state> state) : m_state(std::move(state)) {}
id<MTLTexture> prepared_surface_snapshot::texture() const { return m_state->image; }
surface_format prepared_surface_snapshot::format() const { return m_state->format; }
std::size_t prepared_surface_snapshot::row_offset() const { return m_state->prefix; }
const cached_surface_description& prepared_surface_snapshot::source_description() const { return m_state->description; }
bool prepared_surface_snapshot::same_content(const prepared_surface_snapshot& other) const
{
	const auto& a = *m_state; const auto& b = *other.m_state;
	const auto source = a.authority.lock();
	return source && source == b.authority.lock() && a.cache == b.cache && a.cache_generation == b.cache_generation &&
		a.authority_generation == b.authority_generation && a.authority_epoch == b.authority_epoch &&
		a.format == b.format && a.prefix == b.prefix && a.rectangle.x == b.rectangle.x && a.rectangle.y == b.rectangle.y &&
		a.rectangle.width == b.rectangle.width && a.rectangle.height == b.rectangle.height;
}
bool prepared_surface_snapshot::matches_source(const cached_surface_ref& source,bool require_cache_epoch) const
{
	const auto& state=*m_state;state.outside();const auto alive=state.lifetime.lock();
	return alive && source && state.authority.lock()==source && state.cache->positions.contains(source.get()) &&
		(!require_cache_epoch || state.cache->snapshot_generation==state.cache_generation) &&
		!source->faulted && source->generation==state.authority_generation && source->native_write_epoch==state.authority_epoch &&
		source->initialized && source->deferred.empty();
}
bool prepared_surface_snapshot::cache_native_epoch(std::uint64_t epoch) const
{
	m_state->outside();const auto alive=m_state->lifetime.lock();return alive && m_state->cache->native_clock==epoch;
}
bool prepared_surface_snapshot::source_current(bool require_cache_epoch) const {return matches_source(m_state->authority.lock(),require_cache_epoch);}
std::shared_ptr<const surface_texture_revision> prepared_surface_snapshot::revision() const { return m_state->receipt; }
bool prepared_surface_snapshot::needs_copy() const { return m_state->copy; }
void prepared_surface_snapshot::validate_outside() const { m_state->outside(); }
id<MTLTexture> prepared_surface_snapshot::opaque_view(MTLPixelFormat format)
{
	auto& state = *m_state; state.outside();
	if (!state.image || state.image.textureType != MTLTextureType2D || state.image.sampleCount != 1)
		throw std::invalid_argument("Prepared opaque view requires a single-sample 2D native snapshot");
	const auto make = [&]
	{
		return make_texture_view(state.image,format,MTLTextureType2D,NSMakeRange(0,1),NSMakeRange(0,1),
			MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed,MTLTextureSwizzleGreen,MTLTextureSwizzleBlue,MTLTextureSwizzleOne));
	};
	const auto get = [&](auto& views, std::uint64_t& created, std::uint64_t& reused)
	{
		for (const auto& view : views) if (view.texture && view.format == format) { ++reused; return view.texture; }
		const auto image = make();
		for (auto& view : views) if (!view.texture) { view = {format,image}; ++created; break; }
		return image;
	};
	if (!state.candidate.empty()) return get(state.candidate.front().views,state.views_created,state.views_reused);
	if (const auto alive = state.lifetime.lock())
		for (auto& entry : state.cache->snapshots)
			if (entry.texture == state.image) return get(entry.views,state.cache->snapshot_stats.views_created,state.cache->snapshot_stats.views_reused);
	return make();
}
void prepared_surface_snapshot::encode(context& ctx)
{
	auto& state = *m_state; state.outside();
	if (&ctx != state.context_owner) throw std::invalid_argument("Prepared native snapshot belongs to another context");
	if (!state.copy) return;
	const auto recording = ctx.recording_submission();
	if (state.encoded)
	{
		if (state.encoded->same_batch(recording)) return;
		if (!state.encoded->ready() || !state.encoded->cancelled()) throw std::logic_error("Prepared native snapshot already belongs to a submission");
	}
	ctx.keep_alive(m_state);
	for (const auto& input : state.backings)
	{
		if (input.owner) ctx.keep_alive(input.owner);
		if (input.physical) ctx.keep_alive(input.physical);
	}
	const auto& rectangle = state.candidate.front().rectangle;
	texture_region target; target.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
	auto input = target; input.origin = MTLOriginMake(rectangle.x, rectangle.y, 0);
	const auto native = state.samples ? state.allocation->expanded_texture(ctx) : state.allocation->render_texture(ctx);
	if (state.samples) state.samples->expand(ctx, state.allocation->render_texture(ctx), native);
	copy_texture(ctx, native, input, state.image, target);
	state.encoded = recording;
}
void prepared_surface_snapshot::completed()
{
	auto& state = *m_state; state.outside();
	if (state.copy)
	{
		if (!state.encoded || !state.encoded->ready()) throw std::logic_error("Native snapshot completion has not succeeded");
		state.encoded->wait(std::chrono::milliseconds(1));
	}
	state.complete = true;
}
void prepared_surface_snapshot::publish()
{
	auto& state = *m_state; state.outside();
	if (!state.copy || state.published) return;
	if (!state.complete) throw std::logic_error("Native snapshot cannot publish before successful completion");
	state.published = true;
	const auto alive = state.lifetime.lock();
	if (!alive || state.cache->snapshot_generation != state.cache_generation) return;
	auto& cache = *state.cache;
	++cache.snapshot_stats.created;
	const auto& first = state.candidate.front();
	// A later native writer may keep an older job's immutable image valid,
	// but that job must not reinsert an obsolete ready-cache content key.
	if (state.source->faulted || state.source->generation != first.generation || state.source->native_write_epoch != first.native_epoch) return;
	const auto limits = cache.snapshot_stats.limits;
	if (!limits.entries || first.bytes > limits.bytes) { ++cache.snapshot_stats.bypassed; return; }
	for (const auto& ready : cache.snapshots)
		if (ready.source.lock() == state.source && ready.budget == first.budget && ready.generation == first.generation && ready.native_epoch == first.native_epoch &&
			ready.row_offset == first.row_offset && ready.rectangle.x == first.rectangle.x && ready.rectangle.y == first.rectangle.y &&
			ready.rectangle.width == first.rectangle.width && ready.rectangle.height == first.rectangle.height)
			return; // Another completed owner already published this content key.
	while (cache.snapshots.size() >= limits.entries || cache.snapshot_stats.bytes > limits.bytes - first.bytes)
		cache.evict_snapshot(std::prev(cache.snapshots.end()));
	cache.snapshot_stats.bytes += first.bytes;
	cache.snapshot_stats.views_created += state.views_created; cache.snapshot_stats.views_reused += state.views_reused;
	cache.snapshots.splice(cache.snapshots.begin(),state.candidate);
	cache.snapshot_stats.entries = cache.snapshots.size();
}
std::shared_ptr<prepared_surface_snapshot> prepared_surface_snapshot::settled() const
{
	const auto& original = *m_state;
	original.outside();
	if (original.copy && (!original.complete || !original.published))
		throw std::logic_error("Native snapshot execution owners cannot retire before confirmed publication");
	auto compact = std::make_shared<state>();
	compact->image = original.image;
	compact->format = original.format;
	compact->prefix = original.prefix;
	compact->description = original.description;
	compact->rectangle = original.rectangle;
	compact->receipt = original.receipt;
	compact->context_owner = original.context_owner;
	compact->cache = original.cache;
	compact->lifetime = original.lifetime;
	compact->cache_generation = original.cache_generation;
	compact->authority = original.authority;
	compact->authority_generation = original.authority_generation;
	compact->authority_epoch = original.authority_epoch;
	compact->complete = true;
	return std::shared_ptr<prepared_surface_snapshot>(new prepared_surface_snapshot(std::move(compact)));
}
bool prepared_surface_snapshot::cache_current() const
{
	const auto& state = *m_state;
	state.outside();
	const auto alive = state.lifetime.lock();
	const auto source = state.authority.lock();
	return alive && source && state.cache->snapshot_generation == state.cache_generation && state.cache->positions.contains(source.get()) &&
		!source->faulted && source->generation == state.authority_generation && source->native_write_epoch == state.authority_epoch;
}

id<MTLTexture> surface_cache::snapshot_texture(guest_surface_range range,
											   const std::function<bool(const cached_surface_description&)>& accepts, const std::shared_ptr<heap_budget>& budget)
{
	if (!accepts)
		throw std::invalid_argument("Native surface texture snapshot requires a validator");
	const auto mapping = [&](const cached_surface_description& description) -> std::optional<clear_rectangle>
	{
		if (!accepts(description))
			return {};
		const auto& allocation = description.allocation;
		const auto size = allocation.samples == 1 ? std::array{allocation.width, allocation.height} :
			sample_expanded_size(allocation.width, allocation.height, allocation.samples);
		return clear_rectangle{0, 0, size[0], size[1]};
	};
	return snapshot_texture_region(guest_byte_coverage(range), mapping, budget);
}
id<MTLTexture> surface_cache::snapshot_texture_region(const guest_surface_layout& coverage,
													  const std::function<std::optional<clear_rectangle>(const cached_surface_description&)>& region, const std::shared_ptr<heap_budget>& budget)
{
	return snapshot_texture_regions({&coverage, 1}, region, budget);
}
id<MTLTexture> surface_cache::snapshot_texture_regions(std::span<const guest_surface_layout> coverage,
	const std::function<std::optional<clear_rectangle>(const cached_surface_description&)>& region,
	const std::shared_ptr<heap_budget>& budget, std::shared_ptr<const surface_texture_revision>* source_revision, bool allow_guest_fallback, std::optional<surface_format>* source_format)
{
	if (source_revision) source_revision->reset();
	if (source_format) source_format->reset();
	if (!region) throw std::invalid_argument("Native surface texture snapshot needs a validator");
	const auto mapping = [&](const cached_surface_description& source) -> std::optional<surface_texture_region>
	{
		const auto pixels = region(source);
		return pixels ? std::optional{surface_texture_region{*pixels, 0}} : std::nullopt;
	};
	return snapshot_texture_mapping(coverage, mapping, budget, source_revision, allow_guest_fallback, source_format);
}
id<MTLTexture> surface_cache::snapshot_texture_mapping(std::span<const guest_surface_layout> coverage,
	const std::function<std::optional<surface_texture_region>(const cached_surface_description&)>& region,
	const std::shared_ptr<heap_budget>& budget, std::shared_ptr<const surface_texture_revision>* source_revision, bool allow_guest_fallback,
	std::optional<surface_format>* source_format, std::optional<std::size_t>* source_row_offset,bool defer_unsupported)
{
	if (source_revision) source_revision->reset();
	if (source_format) source_format->reset();
	if (source_row_offset) source_row_offset->reset();
	const auto prepared = prepare_texture_mapping(coverage,region,budget,allow_guest_fallback,{},true,defer_unsupported);
	if (!prepared) return nil;
	if (prepared->needs_copy())
	{
		const auto& state = *prepared->m_state;
		m_impl->execute([&] { prepared->encode(m_impl->ctx); },[] {},{},{{state.source,state.source->revision}});
		prepared->completed();
		m_impl->cpu([&] { if (!prepared->revision()->current()) throw surface_conflict_error(state.source->storage.parts.front().coverage.range); });
		prepared->publish();
	}
	if (source_revision) *source_revision = prepared->revision();
	if (source_format) *source_format = prepared->format();
	if (source_row_offset) *source_row_offset = prepared->row_offset();
	return prepared->texture();
}
std::uint64_t surface_cache::texture_source_epoch() const
{
	m_impl->outside_epoch();return m_impl->native_clock;
}
std::uint64_t surface_cache::texture_source_topology_epoch() const
{
	m_impl->outside_epoch();return m_impl->texture_topology_epoch;
}
std::vector<cached_surface_ref> surface_cache::texture_sources(std::span<const guest_surface_layout> coverage)
{
	prepare_texture_read_boundary(coverage);
	std::vector<cached_surface_ref> result;
	m_impl->cpu([&]
	{
		result = m_impl->cpu_aliases(coverage,true);
		// Retained dirty physical output must never be mistaken for VM bytes
		// just because its original source VA was replaced or unmapped.
		for (const auto& surface : m_impl->surfaces)
			if (surface->physical && (surface->dirty || !surface->deferred.empty()) &&
				std::find(result.begin(),result.end(),surface)==result.end())
				for (const auto& ticket : *surface->physical)
				{
					bool found = false;
					for (const auto& part : coverage)
					{
						guest_surface_byte_cursor cursor(part);
						while (const auto run = cursor.next())
						{
							const auto match = m_impl->memory->physical_surface_overlaps_current(*ticket,*run);
							if (!match) throw std::runtime_error("Native display lacks physical source identity");
							if (*match) { found=true;break; }
						}
						if (found) break;
					}
					if (found) { result.push_back(surface);break; }
				}
		std::erase_if(result,[&](const auto& source)
		{
			if (source->dirty || !source->deferred.empty())
			{
				m_impl->check_mapping(source);
				if (m_impl->current(source)!=(source->deferred.empty()?source->revision:source->deferred_revision))
					throw surface_conflict_error(source->storage.parts.front().coverage.range);
				return false;
			}
			if (!source->native_write_epoch || source->faulted) return true;
			if (source->description.mapping && !source->description.mapping->current()) return true;
			return m_impl->current(source) != source->revision;
		});
	});
	if (result.size()>64) throw std::length_error("Native display source bound exceeded");
	return result;
}
std::shared_ptr<prepared_surface_snapshot> surface_cache::prepare_texture_mapping(std::span<const guest_surface_layout> coverage,
	const std::function<std::optional<surface_texture_region>(const cached_surface_description&)>& region,
	const std::shared_ptr<heap_budget>& budget, bool allow_guest_fallback, std::span<const std::shared_ptr<prepared_surface_snapshot>> pending, bool allow_new,
	bool defer_unsupported, const cached_surface_ref& source_only)
{
	m_impl->outside_epoch();
	if (pending.size() > 64) throw std::length_error("Native snapshot preparation exceeds its batch bound");
	if (coverage.empty() || coverage.size() > (1u << 20)) throw std::invalid_argument("Native texture coverage count is invalid");
	for (const auto& part : coverage) { validate_guest_surface_range(part.range); (void)guest_surface_byte_cursor(part); }
	if (!region || !budget || budget->device() != m_impl->ctx.device())
		throw std::invalid_argument("Native surface texture snapshot needs a validator and matching heap budget");
	prepare_texture_read_boundary(coverage);
	if (source_only) m_impl->check(source_only);
	const auto candidates = source_only ? std::vector<cached_surface_ref>{source_only} : m_impl->cpu_aliases(coverage, true);
	const auto map_region = [&](const cached_surface_ref& surface)
	{
		std::optional<surface_texture_region> value;
		m_impl->cpu([&]
		{
			m_impl->check_mapping(surface); value = region(surface->description); m_impl->check_mapping(surface);
		});
		return value;
	};
	cached_surface_ref source;
	clear_rectangle selected{};
	std::size_t row_offset = 0;
	bool ambiguous_clean_epoch = false;
	for (const auto& surface : candidates)
		if (surface->dirty || !surface->deferred.empty())
		{
			const auto mapping = map_region(surface);
			if (source || !mapping)
			{
				if (allow_guest_fallback || defer_unsupported) return {};
				throw std::invalid_argument("GPU surface texture overlap requires an unsupported alias conversion");
			}
			source = surface;
			selected = mapping->pixels;
			row_offset = mapping->guest_row_offset;
		}
	// VM writeback synchronizes guest storage but does not replace native
	// depthfloat precision. Prefer a compatible initialized backing while its
	// active-byte revision remains valid. Dirty/deferred aliases above always
	// take priority, including unsupported overlaps that must never fall back.
	if (!source)
		for (const auto& surface : candidates)
			if (surface->initialized && !surface->faulted && (surface->lease || surface->spill.buffer) &&
				m_impl->current(surface) == surface->revision)
			{
				const auto mapping = map_region(surface);
				if (!mapping)
					continue;
				if (source)
				{
					if (surface->native_write_epoch < source->native_write_epoch)
						continue;
					if (surface->native_write_epoch == source->native_write_epoch)
					{
						const auto& a = source->description;
						const auto& b = surface->description;
						const bool same_shape = a.allocation == b.allocation && a.row_pitch == b.row_pitch && a.tile == b.tile &&
												(!a.tile || a.address == b.address) &&
												bool(a.morton) == bool(b.morton) && (!a.morton || same(*a.morton, *b.morton)) &&
												selected.x == mapping->pixels.x && selected.y == mapping->pixels.y && selected.width == mapping->pixels.width && selected.height == mapping->pixels.height &&
												row_offset == mapping->guest_row_offset;
						const bool equivalent_vm = !surface->native_write_epoch && same_shape &&
							equivalent_storage(*m_impl->memory, source->storage, surface->storage);
						ambiguous_clean_epoch |= !equivalent_vm;
						continue;
					}
				}
				ambiguous_clean_epoch = false;
				source = surface;
				selected = mapping->pixels;
				row_offset = mapping->guest_row_offset;
			}
	if (ambiguous_clean_epoch)
		throw std::invalid_argument("Native clean surface texture aliases lack a unique write epoch");
	if (!source)
		return {};
	const auto& allocation = source->description.allocation;
	const auto size = allocation.samples == 1 ? std::array{allocation.width, allocation.height} :
		sample_expanded_size(allocation.width, allocation.height, allocation.samples);
	if (row_offset >= surface_guest_bytes(allocation.format))
		throw std::out_of_range("Native texture snapshot row prefix exceeds source guest texel");
	if (!selected.width || !selected.height || selected.x > size[0] || selected.width > size[0] - selected.x ||
		selected.y > size[1] || selected.height > size[1] - selected.y)
		throw std::out_of_range("Native texture snapshot rectangle exceeds source pixels");
	surface_list_guard pin(m_impl->pinned, source);
	const auto make_revision = [&]
	{
		std::vector<guest_surface_layout> watched;
		for (const auto& part : source->storage.parts) watched.push_back(part.coverage);
		return std::make_shared<surface_texture_revision>(m_impl->memory, std::move(watched), source->revision, source->description.mapping);
	};
	const auto existing = [&]() -> std::shared_ptr<prepared_surface_snapshot>
	{
		// Ready images and privately owned copies no longer require their
		// source allocation to be restored from a raw spill. Deferred content
		// still needs materialization before its new version can be selected.
		if (!source->initialized || source->faulted || !source->deferred.empty()) return {};
		for (auto it = m_impl->snapshots.begin(); it != m_impl->snapshots.end(); ++it)
			if (it->source.lock() == source && it->budget == budget && it->generation == source->generation &&
				it->native_epoch == source->native_write_epoch && it->row_offset == row_offset &&
				it->rectangle.x == selected.x && it->rectangle.y == selected.y &&
				it->rectangle.width == selected.width && it->rectangle.height == selected.height)
			{
				// Our own writeback changes VM receipts, not the native content
				// version. Reloads/writers already change generation/epoch in prepare.
				std::shared_ptr<const surface_texture_revision> revision = it->versions == source->revision ? it->revision : make_revision();
				bool valid = false;
				m_impl->cpu([&] { valid = revision->current(); });
				if (!valid) return {};
				it->versions = source->revision; it->revision = std::move(revision);
				const auto texture = it->texture;
				m_impl->snapshots.splice(m_impl->snapshots.begin(), m_impl->snapshots, it);
				++m_impl->snapshot_stats.reused;
				source->last_use = ++m_impl->clock;
				auto state = std::make_shared<prepared_surface_snapshot::state>();
				state->image = texture; state->format = allocation.format; state->prefix = row_offset; state->receipt = m_impl->snapshots.front().revision;
				state->description = source->description;
				state->rectangle = selected;
				state->authority = source; state->authority_generation = source->generation; state->authority_epoch = source->native_write_epoch;
				state->complete = true; state->context_owner = &m_impl->ctx;
				state->cache = m_impl.get(); state->lifetime = m_impl->snapshot_lifetime; state->cache_generation = m_impl->snapshot_generation;
				return std::shared_ptr<prepared_surface_snapshot>(new prepared_surface_snapshot(std::move(state)));
			}
		for (const auto& previous : pending)
		{
			if (!previous || !previous->needs_copy()) continue;
			const auto& state = *previous->m_state;
			if (state.candidate.empty()) continue;
			const auto& item = state.candidate.front();
			if (state.source == source && state.cache == m_impl.get() && state.cache_generation == m_impl->snapshot_generation && state.context_owner == &m_impl->ctx && item.budget == budget && item.generation == source->generation &&
				item.native_epoch == source->native_write_epoch && item.versions == source->revision && item.row_offset == row_offset &&
				item.rectangle.x == selected.x && item.rectangle.y == selected.y && item.rectangle.width == selected.width && item.rectangle.height == selected.height)
			{
				bool valid = false;
				m_impl->cpu([&] { valid = state.receipt->current(); });
				if (!valid) return {};
				source->last_use = ++m_impl->clock;
				return previous;
			}
		}
		return {};
	};
	m_impl->prune_snapshots();
	if (const auto hit = existing()) return hit;
	m_impl->prepare(source, surface_initialization::guest);
	m_impl->prune_snapshots();
	if (const auto hit = existing()) return hit;
	if (!allow_new) throw std::length_error("Native snapshot pending-owner bound exceeded before allocation");
	auto state = std::make_shared<prepared_surface_snapshot::state>();
	state->cache = m_impl.get(); state->lifetime = m_impl->snapshot_lifetime; state->cache_generation = m_impl->snapshot_generation;
	state->source = source; state->allocation = source->lease; state->format = allocation.format; state->prefix = row_offset;
	state->description = source->description;
	state->rectangle = selected;
	state->authority = source; state->authority_generation = source->generation; state->authority_epoch = source->native_write_epoch;
	state->context_owner = &m_impl->ctx; state->receipt = make_revision(); state->copy = true;
	state->backings = m_impl->retain_backings(std::array{surface_validation{source,source->revision}});
	if (allocation.samples > 1) state->samples = m_impl->samples;
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native_format(allocation) width:selected.width height:selected.height mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	for (;;)
	{
		std::exception_ptr failure;
		@autoreleasepool { try { state->image = budget->texture(descriptor); } catch (...) { failure = std::current_exception(); } }
		if (!failure) break;
		try { std::rethrow_exception(failure); }
		catch (const heap_budget_error&)
		{
			if (m_impl->pool.trim()) continue;
			if (m_impl->snapshots.empty()) throw;
			m_impl->evict_snapshot(std::prev(m_impl->snapshots.end()));
		}
	}
	state->candidate.push_front({source,source->generation,source->native_write_epoch,source->revision,budget,selected,row_offset,state->image.heap.size,state->image,state->receipt});
	m_impl->cpu([&] { if (!state->receipt->current()) throw surface_conflict_error(source->storage.parts.front().coverage.range); });
	return std::shared_ptr<prepared_surface_snapshot>(new prepared_surface_snapshot(std::move(state)));
}
guest_surface_snapshot surface_cache::cpu_read(guest_surface_range range)
{
	prepare_cpu_read(range);
	guest_surface_snapshot result{};
	m_impl->cpu([&] { result = m_impl->memory->read(range); });
	return result;
}
void surface_cache::prepare_cpu_read(guest_surface_range range)
{
	prepare_cpu_read_active(guest_byte_coverage(range));
}
void surface_cache::prepare_cpu_read_active(const guest_surface_layout& coverage)
{
	(void)guest_surface_byte_cursor(coverage);
	m_impl->synchronize(m_impl->timeout);
	const auto aliases = m_impl->cpu_aliases(coverage);
	for (const auto& surface : aliases)
		if (surface->dirty || !surface->deferred.empty())
			m_impl->flush(surface);
}
void surface_cache::cpu_write(guest_surface_range range, std::span<const std::byte> bytes)
{
	validate_guest_surface_range(range);
	if (bytes.size() != range.size)
		throw std::invalid_argument("CPU surface write byte count differs from range");
	const std::vector<std::byte> payload(bytes.begin(), bytes.end());
	cpu_write(range, [&]
			  { m_impl->memory->write(range, payload); });
}
void surface_cache::cpu_write(guest_surface_range range, const std::function<void()>& store)
{
	cpu_write(std::span<const guest_surface_range>{&range, 1}, store);
}
void surface_cache::cpu_write(std::span<const guest_surface_range> ranges, const std::function<void()>& store,
	const cpu_capture_dispatch& capture)
{
	std::vector<guest_surface_layout> coverage;
	coverage.reserve(ranges.size());
	for (const auto range : ranges) coverage.push_back(guest_byte_coverage(range));
	cpu_write_active(coverage, store, capture);
}
void surface_cache::cpu_write_active(std::span<const guest_surface_layout> coverage, const std::function<void()>& store,
	const cpu_capture_dispatch& capture)
{
	for (const auto& part : coverage) (void)guest_surface_byte_cursor(part);
	if (!store)
		throw std::invalid_argument("External surface writer is absent");
	m_impl->synchronize(m_impl->timeout);
	std::vector<cached_surface_ref> aliases;
	for (const auto& part : coverage)
		for (const auto& surface : m_impl->cpu_aliases(part))
			if (std::find(aliases.begin(), aliases.end(), surface) == aliases.end())
				aliases.push_back(surface);
	for (const auto& surface : aliases)
		if (surface->dirty || !surface->deferred.empty())
			m_impl->flush(surface);
	std::vector<surface_retirement> retirements;
	m_impl->cpu([&]
	{
		// Mapping transitions can introduce another physical alias after the
		// preceding flush. Never wait for that alias's GPU work under exclusion.
		for (const auto& part : coverage)
			for (const auto& surface : m_impl->cpu_aliases(part))
				if (std::find(aliases.begin(), aliases.end(), surface) == aliases.end()) aliases.push_back(surface);
		for (const auto& surface : aliases)
			if (surface->dirty || !surface->deferred.empty())
				throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		retirements = impl::retain_metadata_storage({}, {}, aliases);
		std::vector<surface_validation> before_store;
		for (const auto& surface : aliases) before_store.push_back({surface, m_impl->current(surface)});
		const auto invalidate = [&]() noexcept
		{
			for (const auto& surface : aliases)
			{
				surface->initialized = false;
				surface->spill = {};
				surface->dirty = false;
				surface->reset_backing();
				surface->faulted = false;
				surface->native_write_epoch = 0;
			}
		};
		// A throwing producer may already have written a prefix. Preserve a
		// flushed native mirror only if its mapping and all watch versions
		// still match;otherwise guest bytes are authoritative before unlock.
		try { store(); }
		catch (...)
		{
			bool unchanged = false;
			try { m_impl->validate_in_epoch(before_store); unchanged = true; }
			catch (...) {}
			if (!unchanged) invalidate();
			throw;
		}
		invalidate();
	}, capture);
}
void surface_cache::synchronize(std::chrono::milliseconds timeout)
{
	m_impl->synchronize(timeout);
}
void surface_cache::prepare_draw_resources(const std::function<void()>& allocate)
{
	m_impl->outside_epoch();
	if (!allocate) throw std::invalid_argument("Native draw resource admission requires an allocator");
	const auto attempt = [&] { @autoreleasepool { allocate(); } };
	try { attempt(); return; }
	catch (const heap_budget_error& error)
	{
		if (error.requested > error.budget || (!m_impl->recording_batch && !m_impl->pending)) throw;
	}
	catch (const native_allocation_error&)
	{
		if (!m_impl->recording_batch && !m_impl->pending) throw;
	}
	// Never discard a successful prefix just to make room for the next draw.
	// GPU errors, VM conflicts and timeouts propagate without an allocation retry.
	m_impl->synchronize(m_impl->timeout);
	(void)m_impl->ctx.trim_upload_pages();
	++m_impl->transactions.draw_resource_retries;
	attempt();
}
void surface_cache::submit_draw_batch()
{
	m_impl->outside_epoch();
	m_impl->flush_draw_batch();
}
void surface_cache::set_wait_timeout(std::chrono::milliseconds timeout)
{
	if (timeout.count() < 0)
		throw std::invalid_argument("Native cache wait timeout must be nonnegative");
	m_impl->timeout = timeout;
}
surface_transaction_statistics surface_cache::transaction_statistics() const noexcept
{
	return m_impl->transactions;
}
surface_snapshot_statistics surface_cache::snapshot_statistics() const noexcept { return m_impl->snapshot_stats; }
void surface_cache::set_snapshot_limits(surface_snapshot_limits limits)
{
	m_impl->outside_epoch();
	if (limits.entries > 4096 || (limits.entries && !limits.bytes))
		throw std::invalid_argument("Native snapshot cache limits are invalid");
	m_impl->snapshot_stats.limits = limits;
	while (m_impl->snapshots.size() > limits.entries || m_impl->snapshot_stats.bytes > limits.bytes)
		m_impl->evict_snapshot(std::prev(m_impl->snapshots.end()));
}
void surface_cache::clear_snapshots()
{
	m_impl->outside_epoch();
	if (m_impl->snapshot_generation == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Native snapshot cache generation exhausted");
	++m_impl->snapshot_generation;
	while (!m_impl->snapshots.empty()) m_impl->evict_snapshot(std::prev(m_impl->snapshots.end()));
}
id<MTLTexture> surface_cache::opaque_snapshot_view(id<MTLTexture> texture, MTLPixelFormat format)
{
	m_impl->outside_epoch();
	if (!texture || texture.device != m_impl->ctx.device() || texture.textureType != MTLTextureType2D || texture.sampleCount != 1)
		throw std::invalid_argument("Opaque snapshot view requires a native 2D image on this device");
	const auto make = [&]
	{
		return make_texture_view(texture, format, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1),
			MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleOne));
	};
	for (auto& entry : m_impl->snapshots)
		if (entry.texture == texture)
		{
			for (const auto& view : entry.views)
				if (view.texture && view.format == format) { ++m_impl->snapshot_stats.views_reused; return view.texture; }
			const auto result = make();
			for (auto& view : entry.views)
				if (!view.texture) { view = {format, result}; break; }
			++m_impl->snapshot_stats.views_created;
			return result;
		}
	return make();
}
surface_pool_statistics surface_cache::memory_statistics() const
{
	return m_impl->pool.statistics();
}
std::shared_ptr<heap_budget> surface_cache::heap_ledger() const
{
	return m_impl->shared_ledger;
}
std::size_t surface_cache::reclaim_idle()
{
	m_impl->outside_epoch();
	clear_snapshots();
	return m_impl->pool.trim();
}
bool surface_cache::reclaim_clean()
{
	return m_impl->reclaim_clean();
}

surface_metadata_statistics surface_cache::metadata_statistics() const
{
	return m_impl->metadata;
}
std::size_t surface_cache::trim_metadata(std::size_t target_entries)
{
	m_impl->outside_epoch();
	m_impl->synchronize(m_impl->timeout);
	return m_impl->retire_to(target_entries, m_impl->metadata.limits.watch_parts);
}
} // namespace metal4
