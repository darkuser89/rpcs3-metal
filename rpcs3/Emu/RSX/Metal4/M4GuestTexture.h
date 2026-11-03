#pragma once

#include "M4FragmentTextureBatch.h"
#include "M4GuestTextureLayout.h"
#include "M4HeapBudget.h"
#include "M4SurfaceDepth.h"
#include "M4SurfaceColor.h"
#include "M4Swizzle.h"
#include "M4Transfer.h"
#include "M4CPUCapture.h"
#include "M4TextureMapping.h"
#include "M4SurfaceTextureRevision.h"
#include "M4DisplayAssembly.h"
#include "M4SurfaceBanked.h"
#include "M4TextureAssembly.h"

#include <functional>
#include <list>
#include <utility>

namespace metal4
{
class surface_cache;
struct guest_texture_batch_owner;
class prepared_surface_snapshot;

class texture_revision_conflict final : public std::runtime_error
{
  public:
	texture_revision_conflict() : std::runtime_error("Guest texture changed during native preparation") {}
};
class guest_texture_revision final
{
	std::shared_ptr<guest_surface_memory> m_memory;
	std::vector<guest_surface_layout> m_coverage;
	std::vector<guest_storage_segment> m_segments;
	std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> m_translate;
	std::uint32_t m_offset;
	std::size_t m_storage_bytes;
	unsigned m_location;
	std::vector<std::uint64_t> m_revisions;
	texture_io_revision m_io;
	std::shared_ptr<const surface_texture_revision> m_surface_revision;
	guest_texture_revision(std::shared_ptr<guest_surface_memory> memory, std::vector<guest_surface_layout> coverage, std::vector<guest_storage_segment> segments,
		std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate,
		std::uint32_t offset, std::size_t storage_bytes, unsigned location, std::vector<std::uint64_t> revisions, texture_io_revision io)
		: m_memory(std::move(memory)), m_coverage(std::move(coverage)), m_segments(std::move(segments)), m_translate(std::move(translate)),
		m_offset(offset), m_storage_bytes(storage_bytes), m_location(location), m_revisions(std::move(revisions)), m_io(std::move(io)) {}
	friend class guest_texture_cache;

  public:
	// CPU-only. Validate all tickets together inside the final CPU epoch.
	// Observed logical IO epochs also catch identical-address ABA. All IO
	// producers must stay excluded through this complete validation.
	[[nodiscard]] bool current() const
	{
		if (!m_memory || !m_translate || (m_surface_revision && !m_surface_revision->current()) || !m_io.current() ||
			texture_storage_segments(m_offset, m_location, m_storage_bytes, m_translate) != m_segments || m_coverage.size() != m_revisions.size())
			return false;
		for (std::size_t i = 0; i < m_coverage.size(); ++i)
			if (m_memory->surface_revision(m_coverage[i]) != m_revisions[i]) return false;
		return m_io.current() && (!m_surface_revision || m_surface_revision->current());
	}
};
struct versioned_guest_texture
{
	resolved_fragment_texture image;
	guest_texture_revision revision;
};
// CPU metadata and receipts may be consumed while native work is pending.
// Images remain private until successful completion and common validation.
// The job owns its capture services; cache/context teardown cannot dangle them.
class versioned_guest_texture_job final
{
	struct state;
	std::shared_ptr<state> m_state;
	explicit versioned_guest_texture_job(std::shared_ptr<state> state);
	friend class guest_texture_cache;
  public:
	~versioned_guest_texture_job();
	versioned_guest_texture_job(versioned_guest_texture_job&&) noexcept;
	versioned_guest_texture_job& operator=(versioned_guest_texture_job&&) noexcept;
	versioned_guest_texture_job(const versioned_guest_texture_job&) = delete;
	versioned_guest_texture_job& operator=(const versioned_guest_texture_job&) = delete;
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancel_pending() const;
	[[nodiscard]] const texture_cache_state& metadata(std::size_t index) const;
	// CPU-only; the caller excludes the same VM/IO/surface producers.
	[[nodiscard]] bool current() const;
	[[nodiscard]] std::vector<versioned_guest_texture> finish(std::chrono::milliseconds timeout = std::chrono::seconds(30));
};
// CPU-owned VM version. No guest pointers, watches or Metal objects survive
// capture. Realization may follow VM writes/unmapping and resolver teardown.
// GPU-dirty surface aliases must be resolved separately before VM capture.
class captured_guest_texture final
{
	guest_texture_layout m_plan;
	texture_cache_state m_metadata;
	std::vector<std::byte> m_bytes;
	captured_guest_texture(guest_texture_layout plan, texture_cache_state metadata, std::vector<std::byte> bytes)
		: m_plan(std::move(plan)), m_metadata(metadata), m_bytes(std::move(bytes)) {}
	friend class guest_texture_resolver;
	friend class guest_texture_cache;

  public:
	captured_guest_texture(const captured_guest_texture&) = delete;
	captured_guest_texture& operator=(const captured_guest_texture&) = delete;
	captured_guest_texture(captured_guest_texture&&) = default;
	captured_guest_texture& operator=(captured_guest_texture&&) = default;
	[[nodiscard]] const texture_cache_state& metadata() const noexcept { return m_metadata; }
	[[nodiscard]] std::size_t storage_bytes() const noexcept { return m_bytes.size(); }
};
struct guest_texture_upload_statistics
{
	// Encoding counters, not successful-completion or elapsed-time counters.
	// Scratch excludes the immutable upload buffer and native depth-codec planes.
	std::uint64_t subresources = 0, direct_blits = 0, direct_bytes = 0;
	std::uint64_t staging_buffers = 0, staging_bytes = 0;
	std::uint64_t conversion_dispatches = 0, conversion_blocks = 0, converted_bytes = 0;
	std::uint64_t depth_subresources = 0;
	std::uint64_t vtc_array_images = 0, vtc_decode_dispatches = 0, vtc_texels = 0;
	std::uint64_t depth_volume_subresources = 0, depth_volume_texels = 0, depth_volume_plane_packs = 0;
};
struct guest_texture_upload_dependency
{
	id<MTLEvent> event;
	std::uint64_t value;
};
// A submitted batch owns its immutable native sources and destinations until
// actual completion, independently of the resolver, CPU snapshots and context.
// Pending or failed images cannot escape. This is a captured-version upload,
// not a cache/VM revision receipt: consumers must validate their own tickets.
class guest_texture_upload_job final
{
	struct state;
	std::unique_ptr<state> m_state;
	guest_texture_upload_job(submission work, std::vector<resolved_fragment_texture> images);
	friend class guest_texture_resolver;

  public:
	~guest_texture_upload_job();
	guest_texture_upload_job(guest_texture_upload_job&&) noexcept;
	guest_texture_upload_job& operator=(guest_texture_upload_job&&) noexcept;
	guest_texture_upload_job(const guest_texture_upload_job&) = delete;
	guest_texture_upload_job& operator=(const guest_texture_upload_job&) = delete;
	// Ready includes failed/cancelled completion; only finish() publishes images.
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancel_pending() const;
	// Timeout leaves all ownership intact; retry is allowed after completion.
	// A CPU/VM exclusion must not be held while waiting.
	[[nodiscard]] std::vector<resolved_fragment_texture> finish(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
};
// Immutable upload resolution,not yet an alias/coherence texture cache.
// Owns begin/submit; call outside a live encoding transaction. Address mapper
// receives masked OFFSET and normalized local/main location. Main spans are
// probed separately at every logical 1MiB IO boundary.
class guest_texture_resolver final
{
	[[nodiscard]] static MTLPixelFormat native_format(guest_texture_storage value);
	context& m_context;
	std::shared_ptr<guest_surface_memory> m_memory;
	std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> m_translate;
	std::size_t m_limit;
	cpu_capture_dispatch m_capture;
	std::shared_ptr<heap_budget> m_budget;
	surface_depth_transfer m_depth;
	surface_color_transfer m_color;
	std::unique_ptr<swizzle_engine> m_display_swizzle;
	std::unique_ptr<surface_banked_transfer> m_display_banked;
	id<MTLComputePipelineState> m_conversion;
	id<MTLComputePipelineState> m_vtc_decode;
	id<MTLComputePipelineState> m_depth_volume_decode, m_depth_volume_interleave;
	id<MTLComputePipelineState> m_display_hole_scatter;
	id<MTLComputePipelineState> m_display_storage_copy;
	guest_texture_upload_statistics m_upload_statistics;
	struct source
	{
		guest_texture_layout plan;
		std::vector<guest_storage_segment> segments;
		std::vector<guest_surface_layout> coverage;
	};
	struct display_byte_patch
	{
		std::uint64_t target, payload, size;
	};
	struct texture_assembly_input;
	struct display_assembly_input;
	struct native_cast_input
	{
		guest_texture_layout plan, packed;
		surface_format format;
		id<MTLTexture> source;
		texture_cache_state metadata;
		std::size_t row_offset;
		std::optional<swizzle_region> display_morton = {};
		std::vector<display_byte_patch> display_vm_patches = {};
		std::vector<std::byte> display_vm_bytes = {};
		clear_rectangle source_rectangle = {};
		std::shared_ptr<texture_assembly_input> assembly = {};
	};
	struct texture_assembly_slice
	{
		struct native_patch {id<MTLTexture> texture;clear_rectangle source,target;};
		texture_assembly_plane plane;
		std::shared_ptr<prepared_surface_snapshot> native;
		surface_texture_region region{};
		std::optional<native_cast_input> cast;
		std::optional<captured_guest_texture> vm;
		std::vector<std::shared_ptr<prepared_surface_snapshot>> partial_native;
		std::vector<display_storage_source> partial_layouts;
		display_storage_assembly partial_plan;
		std::shared_ptr<display_assembly_input> bytes;
		std::vector<std::array<std::uint8_t,4>> pixel_map;
		std::vector<native_patch> precision;
		std::array<std::uint32_t,4> channels{},output_map{};
		id<MTLRenderPipelineState> normalizer=nil;
	};
	struct texture_assembly_input
	{
		source input;
		std::vector<std::uint64_t> revisions;
		std::vector<texture_assembly_slice> slices;
		std::vector<std::shared_ptr<prepared_surface_snapshot>> native;
		std::shared_ptr<const surface_texture_revision> receipt;
		texture_cache_state metadata;
		std::uint64_t native_epoch=0,topology_epoch=0;
		std::size_t source_bytes=0;
	};
	void prepare_texture_assembly(texture_assembly_input& input);
	void encode_texture_assembly(const texture_assembly_input& input,id<MTLTexture> texture);
	struct display_buffer_copy { std::uint64_t target, source, ordinal, size; };
	struct display_assembly_native
	{
		surface_format format;
		id<MTLTexture> texture;
		guest_surface_layout geometry;
		std::vector<display_buffer_copy> copies;
		std::size_t tight_pitch;
		std::optional<banked_tile_layout> tile;
	};
	struct display_assembly_input
	{
		guest_texture_layout plan;
		std::vector<display_assembly_native> native;
		std::vector<display_byte_patch> vm_patches;
		std::vector<std::byte> vm_bytes;
		std::size_t raw_bytes=0,linear_bytes=0;
	};
	void prepare_display_helpers(bool morton,bool holes,bool assembly,bool banked=false);
	void encode_display_holes(id<MTLBuffer> target,std::span<const display_byte_patch> patches,std::span<const std::byte> bytes);
	[[nodiscard]] display_assembly_input prepare_display_assembly(const guest_texture_layout& plan,
		std::span<const std::shared_ptr<prepared_surface_snapshot>> native,std::span<const display_storage_source> sources,
		const display_storage_assembly& assembly,std::vector<std::byte> vm_bytes);
	void encode_display_assembly(const display_assembly_input& input,id<MTLTexture> output);
	void cpu_capture(const std::function<void()>& operation) const;
	void watch(source& input) const;
	[[nodiscard]] bool current(const source& input, const std::vector<std::uint64_t>& revisions) const;
	[[nodiscard]] source prepare(const fragment_texture_registers& words) const;
	[[nodiscard]] captured_guest_texture capture(const fragment_texture_registers& words, const source& input, std::vector<std::uint64_t>& revisions);
	[[nodiscard]] id<MTLTexture> allocate(const guest_texture_layout& plan) __attribute__((ns_returns_retained));
	void encode_upload(const guest_texture_layout& plan, id<MTLBuffer> guest, id<MTLTexture> texture);
	void encode_vtc(const guest_texture_layout& plan, id<MTLBuffer> guest, id<MTLTexture> texture);
	void encode_depth_volume(const guest_texture_layout& plan,id<MTLBuffer> guest,id<MTLTexture> texture);
	void encode_depth_volume_plane(id<MTLTexture> source,const texture_region& region,id<MTLTexture> target,unsigned level,std::size_t z);
	[[nodiscard]] native_cast_input prepare_cast(const guest_texture_layout& plan, surface_format format,
		id<MTLTexture> source, texture_cache_state metadata, std::size_t row_offset,clear_rectangle source_rectangle={}) const;
	[[nodiscard]] native_cast_input prepare_morton_display_cast(const guest_texture_layout& plan, surface_format format,
		id<MTLTexture> source, const swizzle_region& layout, std::size_t offset,
		std::vector<display_byte_patch> patches = {}, std::vector<std::byte> bytes = {});
	void encode_cast(const native_cast_input& input, id<MTLTexture> texture);
	[[nodiscard]] guest_texture_upload_job enqueue_mixed(std::span<const captured_guest_texture> snapshots,
		std::span<const native_cast_input> casts, std::span<const guest_texture_upload_dependency> dependencies,
		std::span<const std::shared_ptr<prepared_surface_snapshot>> native = {});
	[[nodiscard]] resolved_fragment_texture cast_surface(const guest_texture_layout& plan, surface_format format,
		id<MTLTexture> source, texture_cache_state metadata, std::size_t row_offset);
	[[nodiscard]] resolved_fragment_texture upload(const captured_guest_texture& snapshot, id<MTLTexture> texture);
	[[nodiscard]] resolved_fragment_texture upload(const fragment_texture_registers& words, const source& input, std::vector<std::uint64_t>& revisions);
	friend class guest_texture_cache;
	friend class versioned_guest_texture_job;
	friend class display_resolver;

  public:
	guest_texture_resolver(context& ctx, std::shared_ptr<guest_surface_memory> memory,
						   std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate, std::size_t maximum_snapshot_bytes, std::shared_ptr<heap_budget> budget = {}, cpu_capture_dispatch capture = {});
	[[nodiscard]] resolved_fragment_texture resolve(const fragment_texture_registers& words);
	// CPU-only after resolver construction: translate, watch and atomically
	// copy one VM version. No allocation/encoding/submission/GPU wait occurs.
	[[nodiscard]] captured_guest_texture capture(const fragment_texture_registers& words);
	// Uses only owned captured data; does not translate/read/watch/check VM.
	// Call after releasing CPU-producer/mapping guards, outside encoding.
	// This intentionally realizes the captured version even if VM changed.
	[[nodiscard]] resolved_fragment_texture realize(const captured_guest_texture& snapshot);
	static constexpr std::size_t maximum_upload_images = 64;
	static constexpr std::size_t maximum_upload_source_bytes = 256u << 20;
	// Preflight every input before begin/allocation. One native submission, no
	// per-image wait. begin() can wait for a previous in-flight slot's reuse.
	// Requires the same buffer/texture ledger, outside CPU epochs/recording.
	// Copies all sources before return; captures may then be destroyed/changed.
	[[nodiscard]] guest_texture_upload_job enqueue_uploads(std::span<const captured_guest_texture> snapshots,
		std::span<const guest_texture_upload_dependency> dependencies = {});
	[[nodiscard]] guest_texture_upload_statistics upload_statistics() const noexcept { return m_upload_statistics; }
};

// Both fields are required together. Native GSRender supplies the same
// provider used by guarded memory and its owned observed IO producer epochs.
// CPU metadata capture/validation is synchronous;GPU preparation is outside.
struct texture_capture_bindings
{
	cpu_capture_dispatch capture;
	std::shared_ptr<io_mapping_epoch> io;
};
struct guest_texture_cache_statistics
{
	std::size_t entries = 0;
	std::uint64_t hits = 0, uploads = 0, invalidations = 0, evictions = 0, surface_snapshots = 0, null_initializations = 0;
	std::size_t cast_entries = 0, cast_bytes = 0;
	std::uint64_t casts_created = 0, casts_reused = 0;
	// Submitted native VM/cast batches versus successfully published images.
	std::uint64_t upload_batches = 0, coalesced_uploads = 0, coalesced_casts = 0;
	std::uint64_t assemblies_created=0,assemblies_reused=0,assembly_native_planes=0,assembly_vm_planes=0,assembly_vm_bytes=0;
};
// Render-thread VM-revision/LRU cache of immutable read-only GPU snapshots.
// A shared heap ledger is mandatory;eviction drops cache references only.
// Callers must not write returned textures. Use the surface-aware overload
// when GPU surfaces may overlap any actual guest storage segment.
class guest_texture_cache final
{
	friend class versioned_guest_texture_job;
	guest_texture_resolver m_resolver;
	std::size_t m_capacity;
	texture_capture_bindings m_access;
	struct entry
	{
		guest_texture_resolver::source input;
		std::vector<std::uint64_t> revisions;
		resolved_fragment_texture resource;
		std::vector<std::shared_ptr<const physical_surface>> physical;
		std::shared_ptr<guest_texture_resolver::texture_assembly_input> assembly={};
	};
	std::list<entry> m_entries;
	struct pending_upload
	{
		guest_texture_resolver::source input;
		std::vector<std::uint64_t> revisions;
		captured_guest_texture snapshot;
	};
	struct pending_textures
	{
		std::vector<pending_upload> vm;
		std::vector<guest_texture_resolver::native_cast_input> casts;
		std::vector<std::shared_ptr<prepared_surface_snapshot>> native;
	};
	struct pending_image
	{
		std::size_t index;
		bool cast;
	};
	void check_pending_bytes(const pending_textures& pending, std::size_t extra) const;
	[[nodiscard]] std::optional<resolved_fragment_texture> resolve_assembly(const fragment_texture_registers& words,
		const guest_texture_resolver::source& input,surface_cache& surfaces,std::shared_ptr<const surface_texture_revision>* receipt,
		pending_textures* pending,std::optional<pending_image>* pending_index);
	void publish_assembly(const std::shared_ptr<guest_texture_resolver::texture_assembly_input>& assembly,const resolved_fragment_texture& image);
	[[nodiscard]] std::function<bool()> publication_guard() const;
	struct cast_entry
	{
		__weak id<MTLTexture> source;
		surface_format format;
		std::uint32_t target_format;
		std::size_t width, height, row_offset, bytes;
		id<MTLTexture> texture;
	};
	std::list<cast_entry> m_casts;
	void evict_cast(std::list<cast_entry>::iterator it);
	guest_texture_cache_statistics m_stats;
	null_texture_family m_null_images{};
	std::shared_ptr<guest_texture_batch_owner> m_batch_owner;
	[[nodiscard]] resolved_fragment_texture resolve_impl(const fragment_texture_registers& words, surface_cache* surfaces,
		std::shared_ptr<const surface_texture_revision>* source_revision = nullptr, const std::function<void()>& after_flush = {},
		pending_textures* pending = nullptr, std::optional<pending_image>* pending_index = nullptr);
	[[nodiscard]] versioned_guest_texture resolve_versioned_impl(const fragment_texture_registers& words, surface_cache* surfaces,
		pending_textures* pending = nullptr, std::optional<pending_image>* pending_index = nullptr);

  public:
	guest_texture_cache(context& ctx, std::shared_ptr<guest_surface_memory> memory,
						std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate,
						std::size_t maximum_snapshot_bytes, std::shared_ptr<heap_budget> budget, std::size_t capacity = 256, texture_capture_bindings access = {});
	~guest_texture_cache();
	guest_texture_cache(const guest_texture_cache&) = delete;
	guest_texture_cache& operator=(const guest_texture_cache&) = delete;
	[[nodiscard]] resolved_fragment_texture resolve(const fragment_texture_registers& words);
	[[nodiscard]] std::shared_ptr<heap_budget> heap_ledger() const { return m_resolver.m_budget; }
	// Immutable typed transparent-black images;no guest address/VM access.
	// Creates all four dimensions before publication,outside recording.
	[[nodiscard]] null_texture_family null_images();
	[[nodiscard]] bool null_images_ready() const noexcept { return bool(m_null_images[0].texture); }
	// Single-mip 2D aliases/subregions use bounded immutable native snapshots
	// before consulting VM tags,including expanded MSAA samples. GPU casts reuse
	// exact native source identities under per-kind entry and heap-byte limits.
	// Other uncompressed casts pack/decode on GPU after byte-address proof;
	// different guest texel widths require linear storage. Other overlaps fail.
	// Revision-valid clean contiguous native backings preserve precision after
	// writeback. Disjoint sources flush actual overlaps then gather guest bytes.
	[[nodiscard]] resolved_fragment_texture resolve(const fragment_texture_registers& words, surface_cache& surfaces);
	// Render-thread preparation outside CPU epochs. The ticket guards the VM
	// and translation version spanning native snapshot/upload preparation.
	[[nodiscard]] versioned_guest_texture resolve_versioned(const fragment_texture_registers& words, surface_cache& surfaces);
	// Captures all sources and submits one mixed native batch, without waiting
	// for that batch. Up to64 unconsumed jobs per cache; source/image bounds
	// remain those of enqueue_uploads. Call finish after independent CPU work.
	[[nodiscard]] versioned_guest_texture_job prepare_versioned_batch(std::span<const fragment_texture_registers> words,
		surface_cache& surfaces, std::span<const guest_texture_upload_dependency> dependencies = {});
	// Prepare every immutable surface snapshot and VM source before recording.
	// Coalesce pending VM versions/casts; submit all misses together, then
	// validate ALL VM/IO/surface receipts before publishing any new entry.
	// One completion wait, never one per VM image or native codec cast.
	// Dependencies gate new uploads/casts; ready images need no new work.
	// Cold native snapshot copies/expansions share the batch. Deferred/spilled
	// source materialization and prior surface transactions may still wait first.
	// Requires one installed buffer/texture ledger,outside VM exclusion. Ready
	// reads may preserve an unrelated SurfaceCache-owned recording;GPU misses
	// and pending write aliases close it. Foreign recordings remain forbidden.
	[[nodiscard]] std::vector<versioned_guest_texture> resolve_versioned_batch(std::span<const fragment_texture_registers> words,
		surface_cache& surfaces, std::span<const guest_texture_upload_dependency> dependencies = {},
		std::chrono::milliseconds timeout = std::chrono::seconds(30));
	[[nodiscard]] guest_texture_cache_statistics statistics() const noexcept;
	// VM cache invalidation preserves independent null images,so prepared
	// guest epochs never need to rebuild them under a method/VM guard.
	void clear();
	// Drops one oldest immutable cache owner. Escaped/native completion owners
	// keep their actual heap charge until their final release.
	bool reclaim_one();
	// Explicit teardown/pressure only,outside a held method/VM capture epoch.
	void clear_null_images();
	void invalidate(guest_surface_range range);
	void invalidate_active(const guest_surface_layout& coverage);
	// Old physical tickets select current aliases without mapping old guest VAs.
	// Missing/budget-limited physical metadata conservatively retires that entry.
	[[nodiscard]] std::size_t invalidate_mapping(guest_surface_range range);
	// Plan exact current aliases and original registration owners in one CPU
	// capture;retire immutable GPU resources only after leaving that capture.
	[[nodiscard]] std::size_t invalidate_physical_fault(const physical_identity& identity,
		std::span<const std::shared_ptr<void>> registrations);
	void validate_surface_adapter(context& context, surface_cache& surfaces) const;
};
} // namespace metal4
