#pragma once
#include "M4CPUCapture.h"
#include "M4DMAPlan.h"
#include "M4GuestTexture.h"
#include "M4SurfaceCache.h"
namespace metal4
{
class dma_compute final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;
	std::size_t m_limit;

  public:
	dma_compute(context& ctx, std::size_t limit);
	// Packed buffers use the unique-byte group order from plan_dma_active.
	void encode(context& ctx, const dma_plan& plan, id<MTLBuffer> source, id<MTLBuffer> destination, bool packed = false) const;
};
// Frozen source bytes provide memmove semantics,including physical aliases.
// GPU work occurs outside CPU capture;all target segments publish atomically.
class dma_transfer final
{
	struct pending;
	context& m_context;
	surface_cache& m_surfaces;
	guest_texture_cache& m_textures;
	std::shared_ptr<guest_surface_memory> m_memory;
	cpu_capture_dispatch m_capture;
	dma_compute m_compute;
	std::size_t m_limit;
	std::unique_ptr<pending> m_pending;

  public:
	dma_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures,
				 std::shared_ptr<guest_surface_memory> memory, cpu_capture_dispatch capture, std::size_t limit);
	~dma_transfer();
	void execute(dma_plan plan, std::chrono::milliseconds timeout = std::chrono::seconds(30),
				 const std::function<void(context&)>& dependencies = {}, std::function<void()> validate_metadata = {});
	void synchronize(std::chrono::milliseconds timeout = std::chrono::seconds(30));
	[[nodiscard]] bool has_pending() const noexcept { return bool(m_pending); }
};
} // namespace metal4
