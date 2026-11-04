#pragma once

#include "M4CPUCapture.h"
#include "M4GuestTexture.h"
#include "M4ScaledImagePlan.h"
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
// Overlapping dirty owners and multi-owner outputs still require
// broader assembly. Unsupported combinations
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
	std::unique_ptr<surface_swizzle_transfer> m_swizzle;
	std::unique_ptr<surface_banked_transfer> m_banked;
	std::size_t m_limit;
	std::unique_ptr<pending> m_pending;
	std::uint64_t m_native_completed=0,m_vm_completed=0;
	void encode_native(const scaled_image_plan& plan,std::span<const clear_rectangle> regions,
		context& ctx,std::span<const id<MTLTexture>> images,
		std::span<const cached_surface_description> inputs={},std::span<const display_storage_copy> copies={},
		std::span<const std::byte> input_bytes={},std::span<const display_storage_copy> input_patches={});

  public:
	scaled_image_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures,
						  std::shared_ptr<guest_surface_memory> memory, cpu_capture_dispatch capture, std::size_t byte_limit);
	~scaled_image_transfer();
	// Optional dependencies encode native prerequisites only;never VM/CPU/UI
	// work. A timeout after output submission retains its transaction for
	// synchronize()/teardown. Earlier source/target preparation is SurfaceCache-
	// owned:finish that prefix before explicitly retrying execute(). Prior queued
	// copies may materialize;the current scaled output is never published by
	// preparation alone.
	// Copy the plan before callbacks. An owned metadata validator runs in both
	// CPU epochs and before potentially flushing cache preparation. It is held
	// across timeout until publication or teardown.
	// It must be CPU-only and own everything needed by a later synchronize().
	void execute(scaled_image_plan plan, std::chrono::milliseconds timeout = std::chrono::seconds(30),
				 const std::function<void(context&)>& dependencies = {}, std::function<void()> validate_metadata = {});
	void synchronize(std::chrono::milliseconds timeout = std::chrono::seconds(30));
	[[nodiscard]] bool has_pending() const noexcept { return bool(m_pending); }
	[[nodiscard]] std::uint64_t native_completed() const noexcept { return m_native_completed; }
	[[nodiscard]] std::uint64_t vm_completed() const noexcept { return m_vm_completed; }
};
} // namespace metal4
