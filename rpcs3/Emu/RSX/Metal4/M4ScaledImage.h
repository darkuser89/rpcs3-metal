#pragma once

#include "M4CPUCapture.h"
#include "M4GuestTexture.h"
#include "M4ScaledImagePlan.h"
#include "M4ScaledImageNativePlan.h"
#include "M4SurfaceCache.h"

namespace metal4
{
class scaled_image_compute final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;
	id<MTLComputePipelineState> m_gather;
	std::size_t m_limit;

  public:
	explicit scaled_image_compute(context& ctx, std::size_t byte_limit);
	// Encode only:source/destination are independent owned storage snapshots.
	// Writes active destination pixels;all padding/Morton exterior survives.
	void encode(context& ctx, const scaled_image_plan& plan, id<MTLBuffer> source, id<MTLBuffer> destination) const;
	void gather(context& ctx,id<MTLBuffer> source,id<MTLBuffer> destination,
		std::span<const display_storage_copy> copies,std::size_t input) const;
};
// Render-owner transfer with recoverable pending GPU work. Suitable native
// RGB565/ARGB8 authorities stay GPU-resident:private guest-word snapshots feed
// the exact scale/filter kernel and native versions publish in the cache CPU
// epoch. Overlapping reads use immutable feedback snapshots;1/2/4 samples keep
// their individual expanded pixels. Missing native destinations are registered
// as exact active output rasters,without uploading VM background. A full-write
// transaction publishes their first authority only on success;pitch/Morton/tile
// exterior stays in guest storage until an actual observation/writeback.
// Missing RAM inputs use a coherent guest upload into an exact native source
// raster;the result stays dirty on GPU. Deferred endpoints materialize their
// captured native sections before use;full rectangular unions need no guest
// background. Partial copies preserve native/spilled or coherent RAM background.
// Inputs from disjoint native color/depth byte authorities are packed on
// GPU,including guest-format reinterpretation. Sources remain validated draw
// readers;odd pitches,Morton andbanked storage retain their active byte order.
// Only missing input bytes use coherent RAM snapshots;their owned revision
// receipts remain mandatory through final publication and pending timeouts.
// Depth/stencil packing uses the established guest-word codecs,not color
// normalization. MSAA depth retains the finite [0,1] sample-transfer contract.
// Assembly packs only selected native texels in bounded linear rectangles;
// native Morton/bank allocation size is not the packed snapshot size. Sparse
// consumers share texels without reading padding or allocating a whole tile.
// MSAA inputs expand only selected sample-raster rectangles. MSAA output
// imports update only selected native samples,retaining the native background.
// Up to 64 disjoint compatible native targets can cover the exact
// destination through bounded window lists. Each keeps its sample order and
// storage mapping with transactional rollback; retained readers use target versions.
// Sources which are themselves selected output authorities retain old leases
// in the same transaction; distinct overlapping authority owners stay guarded.
// Missing output bytes can receive fully written cold B8 authorities without VM
// background upload. Existing color authorities can receive byte-scattered writes
// through GPU word downloads/gathers/imports, preserving all unselected lanes.
// Per-target guest-word codecs retain different native word sizes. Depth byte
// patches retain untouched raw depth/stencil planes, including native precision.
// Overlapping/unordered owners still need broader assembly. MSAA depth keeps
// the finite [0,1] import contract; arbitrary raw MSAA depth remains unsupported.
// Unsupported combinations
// retain the coherent VM snapshot/CAS route. Banked VM coverage is conservative.
// No retained guest pointers,no CPU image processing or legacy backend.
class scaled_image_transfer final
{
	struct pending;
	context& m_context;
	surface_cache& m_surfaces;
	guest_texture_cache& m_textures;
	std::shared_ptr<guest_surface_memory> m_memory;
	cpu_capture_dispatch m_capture;
	scaled_image_compute m_compute;
	surface_color_transfer m_colors;
	std::unique_ptr<surface_depth_transfer> m_depth;
	sample_transfer m_samples;
	std::size_t m_limit;
	std::size_t m_writer_limit;
	std::unique_ptr<pending> m_pending;
	std::uint64_t m_native_completed=0,m_vm_completed=0;
	void execute_impl(scaled_image_plan plan, std::chrono::milliseconds timeout,
		const std::function<void(context&)>& dependencies, std::function<void()> validate_metadata, bool defer_native_completion);
	void retire_native();
	void encode_native(const scaled_image_plan& plan,std::span<const clear_rectangle> regions,
		context& ctx,std::span<const id<MTLTexture>> images,
		std::span<const cached_surface_description> inputs={},std::span<const std::vector<scaled_image_input_region>> input_regions={},
		std::span<const std::byte> input_bytes={},std::span<const display_storage_copy> input_patches={},
		std::span<const scaled_image_output_region> output_regions={},
		std::span<const scaled_image_output_byte_region> output_byte_regions={},
		std::span<const surface_format> output_formats={},std::size_t output_count=1,
		std::span<const std::vector<surface_cache::scaled_image_input_view>> input_views={});

  public:
	scaled_image_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures,
						  std::shared_ptr<guest_surface_memory> memory, cpu_capture_dispatch capture, std::size_t byte_limit,
						  std::size_t writer_limit = scaled_image_max_writers);
	~scaled_image_transfer();
	// Optional dependencies encode native prerequisites only;never VM/CPU/UI
	// work. A timeout after output submission retains its transaction for
	// synchronize()/teardown. Known native target spill/deferred loads share
	// that transaction and its rollback. Other source/target preparation is
	// SurfaceCache-owned:finish that prefix before retrying execute(). Prior queued
	// copies may materialize;the current scaled output is never published by
	// preparation alone.
	// Copy the plan before callbacks. An owned metadata validator runs in both
	// CPU epochs and before potentially flushing cache preparation. It is held
	// across timeout until publication or teardown.
	// It must be CPU-only and own everything needed by a later synchronize().
	void execute(scaled_image_plan plan, std::chrono::milliseconds timeout = std::chrono::seconds(30),
				 const std::function<void(context&)>& dependencies = {}, std::function<void()> validate_metadata = {});
	// Submit native output without waiting for its completion. Every target has
	// a private version until GPU/metadata validation; observers retain the cache
	// coherence boundary. Prior work and preparation may still wait. The coherent
	// VM fallback keeps execute()'s synchronous behavior. At most one owned job.
	void enqueue(scaled_image_plan plan, std::chrono::milliseconds timeout = std::chrono::seconds(30),
		const std::function<void(context&)>& dependencies = {}, std::function<void()> validate_metadata = {});
	// Retire ready native work without flushing a recording or waiting for GPU
	// completion. An older timed-out VM fallback still needs synchronize().
	[[nodiscard]] bool poll();
	void synchronize(std::chrono::milliseconds timeout = std::chrono::seconds(30));
	[[nodiscard]] bool has_pending() const noexcept { return bool(m_pending); }
	[[nodiscard]] std::uint64_t native_completed() const noexcept { return m_native_completed; }
	[[nodiscard]] std::uint64_t vm_completed() const noexcept { return m_vm_completed; }
	[[nodiscard]] sample_transfer_statistics native_sample_statistics() const noexcept {return m_samples.statistics();}
};
} // namespace metal4
