#pragma once

#include "M4Clear.h"
#include "M4DepthBoundsPass.h"
#include "M4GuestCoverage.h"
#include "M4GuestSurface.h"
#include "M4GuestSegments.h"
#include "M4CPUCapture.h"
#include "M4SurfaceMapping.h"
#include "M4SampleTransfer.h"
#include "M4SurfacePool.h"
#include "M4SurfaceSampleBlit.h"
#include "M4SurfaceSwizzle.h"
#include "M4SurfaceBanked.h"
#include "M4SurfaceTypeless.h"
#include "M4SurfaceTextureRevision.h"
#include "M4SurfaceTextureRegion.h"
#include "M4TextureBlit.h"
#include "M4DisplayAssembly.h"
#include "M4ScaledImageNativePlan.h"

#include <functional>

namespace metal4
{
// Texture preparation reached an operation with side effects while the
// draw's immutable GPU report was false.
struct predicated_texture_suppressed {};

struct cached_surface_description
{
	std::uint32_t address = 0;
	surface_allocation_description allocation;
	std::size_t row_pitch = 0;
	std::optional<swizzle_region> morton;
	std::optional<banked_tile_layout> tile = {};
	// Complete allocation in logical storage order. Untiled watches cover
	// active pixels;banked watches cover all bytes. Empty means one contiguous
	// allocation. Addresses are actual EAs,not tile-base offsets.
	std::vector<guest_storage_segment> segments = {};
	std::shared_ptr<const surface_mapping_revision> mapping = {};
};
enum class surface_initialization
{
	guest,
	background
};
enum class surface_mapping_event { read, unmap };
class cached_surface;
struct scaled_image_plan;
using cached_surface_ref = std::shared_ptr<cached_surface>;
struct cached_surface_status
{
	bool initialized, dirty, spilled, faulted, pending;
	std::uint64_t guest_revision;
	std::uint64_t content_generation;
	std::size_t deferred_copies;
	std::uint64_t native_write_epoch = 0;
	// Independent watch versions;guest_revision is available only for a
	// single watch (zero for disjoint storage). Never reduce these with max.
	std::vector<std::uint64_t> guest_revisions = {};
};
class surface_conflict_error final : public std::runtime_error
{
  public:
	const guest_surface_range range;
	explicit surface_conflict_error(guest_surface_range range);
};
struct cached_surface_use
{
	cached_surface_ref surface;
	bool write = false;
};

struct surface_metadata_limits
{
	std::size_t entries = 256;
	std::size_t watch_parts = 512;
};
struct surface_metadata_statistics
{
	surface_metadata_limits limits;
	std::size_t entries = 0, watch_parts = 0;
	std::uint64_t descriptor_lookups = 0, descriptor_comparisons = 0;
	std::uint64_t range_queries = 0, range_nodes = 0, physical_candidates = 0;
	std::uint64_t retirement_visits = 0, retired = 0;
};
class surface_metadata_budget_error final : public std::length_error
{
  public:
	surface_metadata_budget_error() : std::length_error("Native surface metadata budget exhausted; remaining owners are protected") {}
};

// Render-thread cache. Methods own begin/submit; invoke outside a live facade.
// Operations synchronize before publishing metadata or VM bytes. A timeout
// preserves the pending transaction; synchronize() finishes it after its GPU
// work completes. GPU jobs keep native leases even if this cache is destroyed.
// Remappable providers also retain independent physical backing through GPU
// completion and while dirty/deferred output exists. These owners do not lock
// guest writers;revision checks/CAS and productive protection hooks still apply.
// Successful native output may complete against its captured physical inputs
// after a virtual remap. It remains dirty until physical publication;replacement
// VAs receive no output. Stale physical inputs and actual GPU errors still refuse.
// With CPU capture,final revisions,guest writeback and cache metadata publish
// in one provider transaction. GPU waits and replaced-resource retirement run
// outside that transaction. Legacy callers must supply their own exclusion.
// This is a VM adapter boundary, not yet an RPCS3 VM/protection-hook binding.
struct surface_transaction_statistics
{
	std::uint64_t submitted = 0, native_versions = 0, native_version_blits = 0;
	std::uint64_t raw_backup_submissions = 0, raw_backup_bytes = 0;
	std::uint64_t batched_draws = 0, draw_batches = 0, cancelled_draw_batches = 0;
	std::uint64_t draw_resource_retries = 0;
};

struct surface_snapshot_limits
{
	std::size_t entries = 64, bytes = 64u << 20;
};
struct surface_snapshot_statistics
{
	surface_snapshot_limits limits;
	std::size_t entries = 0, bytes = 0;
	std::uint64_t created = 0, reused = 0, evicted = 0, bypassed = 0;
	std::uint64_t views_created = 0, views_reused = 0;
};
// Closed backend preparation owner. Pending images are visible only to the
// native batch encoders; public cache APIs still return completed snapshots.
class prepared_surface_snapshot final
{
	struct state;
	std::shared_ptr<state> m_state;
	explicit prepared_surface_snapshot(std::shared_ptr<state> state);
	friend class surface_cache;
	friend class guest_texture_cache;
	friend class guest_texture_resolver;
	friend class versioned_guest_texture_job;
	friend class display_resolver;
	friend class display_upload_job;
	[[nodiscard]] id<MTLTexture> texture() const;
	[[nodiscard]] surface_format format() const;
	[[nodiscard]] std::size_t row_offset() const;
	// Immutable source geometry only, without an allocation or execution lease.
	[[nodiscard]] const cached_surface_description& source_description() const;
	// Independent immutable copies may contain the same native epoch/rectangle.
	// Metadata-only identity; callers still validate the current VM receipt.
	[[nodiscard]] bool same_content(const prepared_surface_snapshot& other) const;
	[[nodiscard]] bool matches_source(const cached_surface_ref& source, bool require_cache_epoch = true) const;
	[[nodiscard]] bool source_current(bool require_cache_epoch=true) const;
	[[nodiscard]] bool cache_native_epoch(std::uint64_t epoch) const;
	[[nodiscard]] bool cache_texture_sources(std::uint64_t epoch, std::uint64_t topology,
		std::span<const guest_surface_layout> coverage,
		std::span<const std::shared_ptr<prepared_surface_snapshot>> retained) const;
	[[nodiscard]] std::shared_ptr<const surface_texture_revision> revision() const;
	[[nodiscard]] bool needs_copy() const;
	[[nodiscard]] id<MTLTexture> opaque_view(MTLPixelFormat format) __attribute__((ns_returns_retained));
	void validate_outside() const;
	void encode(context& ctx);
	void completed();
	void publish();
	// Compact confirmed image/receipt owner, without source allocation/backing
	// leases or a completed command token. Construct only after publication.
	[[nodiscard]] std::shared_ptr<prepared_surface_snapshot> settled() const;
	[[nodiscard]] bool cache_current() const;
};

class surface_cache final
{
	friend class prepared_surface_snapshot;
	friend class guest_texture_cache;
	friend class versioned_guest_texture_job;
	friend class display_resolver;
	friend class draw_executor;
	friend class scaled_image_transfer;
	struct impl;
	std::unique_ptr<impl> m_impl;
	// Preserve our own recording for CPU-only epochs; close it before helpers
	// that need new GPU work. Foreign caller recordings remain forbidden.
	void prepare_texture_boundary(bool gpu_work);
	// Preserve unrelated work for immutable reads. A revision-tracked consumer
	// can also cross a submitted batch when storage independence is proved.
	// Returns true if an existing texture capture must be prepared again.
	bool prepare_texture_read_boundary(std::span<const guest_surface_layout> coverage, bool revision_tracked = false);
	[[nodiscard]] std::shared_ptr<prepared_surface_snapshot> prepare_texture_mapping(std::span<const guest_surface_layout> coverage,
		const std::function<std::optional<surface_texture_region>(const cached_surface_description&)>& region,
		const std::shared_ptr<heap_budget>& budget, bool allow_guest_fallback,
		std::span<const std::shared_ptr<prepared_surface_snapshot>> pending = {}, bool allow_new = true,
		bool defer_unsupported = false, const cached_surface_ref& source_only = {}, const draw_publication& permit = {},
		bool revision_tracked = false, clear_rectangle* multisample_window = nullptr, bool allow_full_multisample = true);
	[[nodiscard]] std::vector<cached_surface_ref> texture_sources(std::span<const guest_surface_layout> coverage, bool revision_tracked = false);
	[[nodiscard]] std::uint64_t texture_source_epoch() const;
	[[nodiscard]] std::uint64_t texture_source_topology_epoch() const;
	void transfer(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
				  clear_rectangle destination_rectangle, texture_plane plane, std::array<bool, 3> flip, bool typeless);
	void defer_transfer(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
						clear_rectangle destination_rectangle, texture_plane plane, std::array<bool, 3> flip, bool typeless);
	struct scaled_image_input_view
	{
		id<MTLTexture> image = nil;
		clear_rectangle pixels;
	};
	void draw_impl(std::span<const cached_surface_use> surfaces,
				   const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode, bool snapshot_feedback,
				   const surface_allocation_description* transient = nullptr, surface_lease* reservation = nullptr,
				   std::function<void()> publication_check = {}, std::span<const cached_surface_ref> full_overwrites = {}, bool scaled_inputs = false,
					   bool frozen_scaled_inputs = false, bool version_scaled_donor = false,
					   draw_publication publish_draw = {}, const submission* predecessor = nullptr,
					   const std::function<void(context&)>& before_loads = {},
					   std::span<const std::vector<clear_rectangle>> input_windows = {},
					   std::vector<std::vector<scaled_image_input_view>>* input_views = nullptr, bool defer_scaled_completion = false,
					   std::function<void()> after_scaled_commit = {});
	struct scaled_image_endpoints
	{
		std::array<cached_surface_ref,2> surfaces;
		std::array<clear_rectangle,2> regions;
		// Disjoint native windows covering one packed output. Paired
		// rectangles retain canonical and native sample-raster coordinates.
		std::vector<cached_surface_ref> outputs;
		std::vector<scaled_image_output_region> output_regions;
		std::vector<scaled_image_output_byte_region> output_byte_regions;
		std::vector<surface_format> output_formats;
		std::vector<cached_surface_ref> cold_outputs;
		bool frozen_inputs = false;
		bool ram_only = false;
		bool version_target = false;
		bool prepare_version_targets = false;
		std::optional<std::uint64_t> ownership_epoch;
		// A packed input assembled from uniquely ordered native byte authorities.
		// Every selected source remains a validated draw reader.
		std::vector<cached_surface_ref> inputs;
		std::vector<cached_surface_description> descriptions;
		display_storage_assembly assembly;
		// Only bytes missing from every native authority. Their watched CPU
		// epoch must survive preparation,timeouts and final output publication.
		std::vector<std::byte> input_bytes;
		std::vector<display_storage_copy> input_patches;
		std::shared_ptr<const surface_texture_revision> input_revision;
	};
	[[nodiscard]] std::optional<scaled_image_endpoints> prepare_scaled_image(const scaled_image_plan& plan,
		std::chrono::milliseconds timeout, std::size_t writer_limit, std::size_t byte_limit);
	void draw_scaled_image(const scaled_image_endpoints& endpoints,
		std::span<const std::vector<clear_rectangle>> input_windows,
		const std::function<void(context&,std::span<const id<MTLTexture>>,std::span<const std::vector<scaled_image_input_view>>)>& encode,
		std::function<void()> publication_check,std::chrono::milliseconds timeout,
		const std::function<void(context&)>& before_loads = {}, bool defer_completion = false,
		std::function<void()> after_commit = {});

  public:
	surface_cache(context& ctx, std::shared_ptr<guest_surface_memory> memory, std::size_t heap_budget, std::shared_ptr<metal4::heap_budget> shared_budget = {}, cpu_capture_dispatch capture = {}, surface_metadata_limits metadata = {});
	~surface_cache();
	enum class predicated_draw_completion { synchronous, deferred };
	// One owner-thread command batch. Draws reuse private native target versions;
	// metadata becomes visible only after successful GPU/VM validation. Readback,
	// alias/texture/producer boundaries flush automatically. Never submit/begin or
	// cancel this native context directly while the scope is active.
	class draw_batch final
	{
		surface_cache* m_owner;
		std::weak_ptr<void> m_lifetime;
		explicit draw_batch(surface_cache& owner, std::size_t capacity, predicated_draw_completion completion);
		friend class surface_cache;
	  public:
		~draw_batch();
		draw_batch(const draw_batch&) = delete;
		draw_batch& operator=(const draw_batch&) = delete;
		void submit(); // Close the scope and leave completion to the surface cache.
		void finish(); // A GPU timeout retains submitted work for synchronize().
	};
	// Opening the scope allows preparation while previous GPU work is in flight.
	// Surface access and finish() perform the necessary completion checks.
	// A deferred predicated draw submits its own transaction inside the scope.
	[[nodiscard]] draw_batch batch_draws(std::size_t capacity = 64,
		predicated_draw_completion completion = predicated_draw_completion::synchronous)
	{
		return draw_batch(*this, capacity, completion);
	}
	// Allocation-only transaction outside VM exclusion. On a typed allocation
	// failure, finish an existing prefix and retry once. The callback must unwind
	// partial allocations and publish its outputs only after complete success.
	void prepare_draw_resources(const std::function<void()>& allocate);
	// Query/report consumers must submit recorded counters before polling/waiting.
	// Does not wait for completion or publish surface metadata.
	void submit_draw_batch();
	// Retire completed surface work without flushing a recording or waiting for the GPU.
	[[nodiscard]] bool poll_submitted();
	surface_cache(const surface_cache&) = delete;
	surface_cache& operator=(const surface_cache&) = delete;
	// A descriptor hit returns a handle without reading or publishing its contents.
	[[nodiscard]] cached_surface_ref acquire(const cached_surface_description& description);
	[[nodiscard]] cached_surface_status status(const cached_surface_ref& surface) const;
	[[nodiscard]] cached_surface_description description(const cached_surface_ref& surface) const;
	void prepare(const cached_surface_ref& surface, surface_initialization initialization = surface_initialization::guest);
	void clear(const cached_surface_ref& surface, std::uint32_t mask, std::uint32_t color, std::uint32_t zstencil,
			   std::uint32_t stencil_mask, clear_rectangle rectangle);
	// Copy in rendered pixel coordinates, with equal guest format and region
	// size. Different sample counts map through the expanded sample raster.
	// Full all-aspect overwrites may
	// initialize an untouched destination without loading guest pixels first.
	void copy(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
			  std::size_t destination_x, std::size_t destination_y, texture_plane plane = texture_plane::all);
	void blit_nearest(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
					  clear_rectangle destination_rectangle, texture_plane plane = texture_plane::all, std::array<bool, 3> flip = {});
	// Reinterpret BE guest storage words in the destination format, then nearest
	// scale in each surface's expanded 1x1/2x1/2x2 sample raster. All aspects;
	// rows must contain complete target guest texels. Full destination overwrites
	// skip its prior guest upload. Raster bridges reject unsupported raw depth.
	void blit_typeless(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
					   clear_rectangle destination_rectangle, std::array<bool, 3> flip = {});
	// Captures the prepared source's current native allocation/content epoch;
	// destination copies execute in order when its contents are needed. Native
	// allocation leases survive source spill, erase and descriptor replacement.
	// Writers materialize outstanding readers before modifying that allocation.
	// A full overwrite supersedes queued destination contents on GPU success.
	// Native writes take a bounded raw snapshot first. Confirmed backup success
	// permits exact rollback after rejected completion; unknown/failed backup
	// status retains the fault contract. No guest-format round trip is used.
	void defer_copy(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
					std::size_t destination_x, std::size_t destination_y, texture_plane plane = texture_plane::all);
	void defer_blit_nearest(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
							clear_rectangle destination_rectangle, texture_plane plane = texture_plane::all, std::array<bool, 3> flip = {});
	void defer_blit_typeless(const cached_surface_ref& source, clear_rectangle source_rectangle, const cached_surface_ref& destination,
							 clear_rectangle destination_rectangle, std::array<bool, 3> flip = {});
	// Cancel only unexecuted copy plans; retain the destination's prior backing.
	void cancel_deferred_copies(const cached_surface_ref& surface);
	// Callback encodes native draws using the supplied textures in logical slot
	// order. Handles must not escape for future work: keep cached_surface_refs
	// and prepare/register their current allocation on every use instead.
	void draw(std::span<const cached_surface_use> surfaces,
			  const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode,
			  draw_publication publish_draw = {}, const submission* predecessor = nullptr);
	// Explicit barrier-boundary feedback: read-only entries overlapping a write
	// target receive immutable native snapshots of their prepared contents. Add
	// separate read/write entries for the same surface. Duplicate read entries
	// share one pooled snapshot. Snapshot heaps count against the same budget
	// as pinned targets and stay leased through GPU completion. The callback's
	// entire command batch sees that epoch;
	// per-fragment framebuffer fetch and intra-batch draw visibility are separate
	// contracts. Call once per required RSX texture barrier/draw boundary.
	void draw_feedback(std::span<const cached_surface_use> surfaces,
					   const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode);
	// Reserves a pooled R32 mirror before starting the draw transaction. The
	// descriptor callback selects the supplied depth write target (level/slice
	// zero); encode uses a prepared pass, which this method ends automatically.
	// Cancellation/timeout use the same metadata and completion rules as draw.
	void draw_depth_bounds(std::span<const cached_surface_use> surfaces, std::size_t depth_index,
						   const std::function<MTL4RenderPassDescriptor*(std::span<const id<MTLTexture>>)>& descriptor,
						   const std::function<void(context&, std::span<const id<MTLTexture>>, depth_bounds_pass&)>& encode,
						   bool snapshot_feedback = false, draw_publication publish_draw = {}, const submission* predecessor = nullptr);
	// Typed providers use physical publication and an atomic virtual-watch
	// receipt even for ordinary writeback. Untyped legacy providers retain CAS.
	void flush(const cached_surface_ref& surface);
	// Materialize deferred plans into a separate native allocation and publish
	// native dirty contents through the physical tickets
	// captured before the GPU write,even after the original VA was unmapped.
	// Never touches replacement allocations. Stale CPU data,faulted output
	// and ambiguous dirty aliases refuse intact.
	void flush_physical(const cached_surface_ref& surface);
	// Render-owner fault preparation outside VM exclusion. Discover retained
	// old outputs and current active aliases by physical identity/registration,
	// preflight all selected dirty tickets,then publish every selected output.
	// Writes discard affected clean mirrors;reads keep valid native precision.
	// Returns the number of dirty/deferred outputs published. No CPU page opens.
	[[nodiscard]] std::size_t prepare_physical_fault(const physical_identity& identity,
		std::span<const std::shared_ptr<void>> registrations, bool writing);
	// Owner-thread service outside VM exclusion. Select original active ranges
	// and surviving physical aliases before publication. Unmap uses old physical
	// owners only;read keeps native precision using virtual-watch receipts from
	// the same physical commit. No normal mutable virtual GPUflush after a probe.
	// A conflict preserves remaining dirty/deferred owners for a later retry.
	[[nodiscard]] std::size_t invalidate_mapping(guest_surface_range range, surface_mapping_event cause);
	// Native raw-plane spill preserves the actual native format bits, not a
	// lossy guest re-encoding. MSAA uses individual sample expansion/import.
	void spill(const cached_surface_ref& surface);
	// Failed uploads can reload guest data; failed native writes need an intact
	// spill snapshot or a subsequent full overwrite. Never discard dirty data.
	void recover(const cached_surface_ref& surface);
	void erase(const cached_surface_ref& surface);
	// Texture adapters must share the same render context and atomic VM source.
	void validate_texture_adapter(context& ctx, const guest_surface_memory& memory) const;
	// Capture GPU-dirty/deferred or revision-valid clean contents into independent
	// immutable heap storage. No VM writeback or guest-format quantization.
	// Returns nil without a usable native backing. Every dirty/deferred overlap
	// must match; unsupported/ambiguous sources throw before any VM fallback.
	// This is a draw-boundary snapshot, not intra-draw framebuffer visibility.
	// Clean native precision survives VM writeback and native raw spill/restore;
	// incompatible or revision-stale clean sources do not prevent VM fallback.
	[[nodiscard]] id<MTLTexture> snapshot_texture(guest_surface_range range,
												  const std::function<bool(const cached_surface_description&)>& accepts,
												  const std::shared_ptr<heap_budget>& budget) __attribute__((ns_returns_retained));
	// A region mapper supplies native pixel coordinates for the requested guest
	// active-byte coverage. Padding/exterior overlaps do not select sources.
	// Rectangles use the guest-expanded sample raster for MSAA surfaces. Native
	// 2-/4-sample expansion and the copy complete before publishing the immutable
	// snapshot;no guest-codec roundtrip or averaging. Depth expansion inherits
	// sample_transfer's finite [0,1] restriction and validates before publication.
	// The independently allocated snapshot has exactly the mapped dimensions.
	[[nodiscard]] id<MTLTexture> snapshot_texture_region(const guest_surface_layout& coverage,
														 const std::function<std::optional<clear_rectangle>(const cached_surface_description&)>& region,
														 const std::shared_ptr<heap_budget>& budget) __attribute__((ns_returns_retained));
	// Ordered actual coverage for disjoint requests. Mapping callbacks execute
	// under this cache's CPU capture;GPU copies and waits remain outside it.
	// Explicit guest fallback returns nil for composition/conversion gaps;the
	// caller must flush all actual aliases before reading guest bytes. Selected
	// source format is published with its ticket only after successful completion.
	[[nodiscard]] id<MTLTexture> snapshot_texture_regions(std::span<const guest_surface_layout> coverage,
		const std::function<std::optional<clear_rectangle>(const cached_surface_description&)>& region,
		const std::shared_ptr<heap_budget>& budget,
		std::shared_ptr<const surface_texture_revision>* source_revision = nullptr, bool allow_guest_fallback = false,
		std::optional<surface_format>* source_format = nullptr) __attribute__((ns_returns_retained));
	// Byte casts can snapshot enclosing edge pixels and publish the selected
	// row prefix alongside format/revision,only after final GPU/CPU validation.
	// Optional outputs reset on entry and remain empty on nil/failure.
	[[nodiscard]] id<MTLTexture> snapshot_texture_mapping(std::span<const guest_surface_layout> coverage,
		const std::function<std::optional<surface_texture_region>(const cached_surface_description&)>& region,
		const std::shared_ptr<heap_budget>& budget,
		std::shared_ptr<const surface_texture_revision>* source_revision = nullptr, bool allow_guest_fallback = false,
		std::optional<surface_format>* source_format = nullptr, std::optional<std::size_t>* source_row_offset = nullptr,
		bool defer_unsupported = false, const draw_publication& permit = {}) __attribute__((ns_returns_retained));
	[[nodiscard]] guest_surface_snapshot cpu_read(guest_surface_range range);
	// Resolve native dirty/deferred overlaps without allocating a CPU snapshot.
	// Capture bytes later inside the caller's common CPU epoch.
	void prepare_cpu_read(guest_surface_range range);
	void prepare_cpu_read_active(const guest_surface_layout& coverage);
	void prepare_cpu_read_active(const guest_surface_layout& coverage, const draw_publication& permit);
	void cpu_write(guest_surface_range range, std::span<const std::byte> bytes);
	// Flush native aliases before capture;store and mirror invalidation then
	// share one CPU epoch,including a throwing store that may have written a
	// prefix. Writer updates the same provider's revisions/reservations and
	// reuses its active epoch;no GPU work or nonrecursive redispatch in store.
	void cpu_write(guest_surface_range range, const std::function<void()>& store);
	// One atomic producer store spanning disjoint physical IO entries. Alias
	// planning/dirty flush completes before store. An optional writer capture
	// supplies the same memory provider's exclusion for a transfer adapter;
	// otherwise the cache's configured capture/caller-exclusion contract applies.
	void cpu_write(std::span<const guest_surface_range> ranges, const std::function<void()>& store,
		const cpu_capture_dispatch& capture = {});
	void cpu_write_active(std::span<const guest_surface_layout> coverage, const std::function<void()>& store,
		const cpu_capture_dispatch& capture = {});
	void synchronize(std::chrono::milliseconds timeout = std::chrono::seconds(30));
	void set_wait_timeout(std::chrono::milliseconds timeout);
	[[nodiscard]] surface_pool_statistics memory_statistics() const;
	[[nodiscard]] surface_transaction_statistics transaction_statistics() const noexcept;
	// Immutable native copies keyed by the selected source's complete content
	// version and rectangle. Limits count cache owners/actual texture heap bytes;
	// escaped snapshots and native completion owners retain their own charges.
	[[nodiscard]] surface_snapshot_statistics snapshot_statistics() const noexcept;
	void set_snapshot_limits(surface_snapshot_limits limits);
	void clear_snapshots();
	// Display's closed 2D/one-level opaque-alpha profile shares the snapshot
	// entry's bounded views. Noncached inputs get an independently owned view.
	[[nodiscard]] id<MTLTexture> opaque_snapshot_view(id<MTLTexture> texture, MTLPixelFormat format) __attribute__((ns_returns_retained));
	[[nodiscard]] std::shared_ptr<heap_budget> heap_ledger() const;
	// Owner pressure only: no GPU wait/encoding, provider query or metadata
	// retirement. Release idle storage, then one oldest reconstructible clean
	// backing. Preserve dirty/deferred, native precision and active leases.
	std::size_t reclaim_idle();
	bool reclaim_clean();
	[[nodiscard]] surface_metadata_statistics metadata_statistics() const;
	// Completion first, then oldest eligible descriptions. External handles,
	// dirty/deferred owners and valid native precision are never evicted. The
	// target is best effort; acquire enforces both hard limits before watches.
	// All ownership release occurs outside CPU/VM exclusion.
	std::size_t trim_metadata(std::size_t target_entries = 0);
};
} // namespace metal4
