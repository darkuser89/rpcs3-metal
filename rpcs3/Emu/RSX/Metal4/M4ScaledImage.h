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
	std::size_t m_limit;

  public:
	explicit scaled_image_compute(context& ctx, std::size_t byte_limit);
	// Encode only:source/destination are independent owned storage snapshots.
	// Writes active destination pixels;all padding/Morton exterior survives.
	void encode(context& ctx, const scaled_image_plan& plan, id<MTLBuffer> source, id<MTLBuffer> destination) const;
};
// Synchronous render-owner transfer with recoverable pending GPU work. Capture
// both snapshots in one CPU epoch;GPU scale/format/filter/swizzle runs outside
// exclusion. Publish destination active bytes with revision CAS in a new CPU
// epoch. Banked allocations use conservative whole-allocation CAS coverage.
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
	sample_transfer m_samples;
	std::size_t m_limit;
	std::unique_ptr<pending> m_pending;
	std::uint64_t m_native_completed=0,m_vm_completed=0;
	void encode_native(const scaled_image_plan& plan,std::span<const clear_rectangle> regions,
		context& ctx,std::span<const id<MTLTexture>> images);

  public:
	scaled_image_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures,
						  std::shared_ptr<guest_surface_memory> memory, cpu_capture_dispatch capture, std::size_t byte_limit);
	~scaled_image_transfer();
	// Optional dependencies encode native prerequisites only;never VM/CPU/UI
	// work. A timeout retains the transaction for synchronize()/teardown.
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
