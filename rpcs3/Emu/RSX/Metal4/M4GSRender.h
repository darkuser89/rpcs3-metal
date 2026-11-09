#pragma once

#include "Emu/RSX/GSRender.h"
#include "M4CPUCapture.h"
#include "M4Display.h"
#include "M4DrawExecutor.h"
#include "M4Frame.h"
#include "M4GuardedGuestMemory.h"
#include "M4MappingNotifications.h"
#include "M4IOMappingEpoch.h"
#include "M4RSXTextureEpoch.h"
#include "M4RenderTasks.h"
#include "M4TileState.h"

#include <atomic>
#include <mutex>

namespace metal4
{
struct memory_fault_request
{
	std::uint32_t address;
	bool writing;
	// Exact faulting-job byte extent,including u16 index expansion. Services
	// must not wait for this blocked offloader itself;extent is not a lease.
	std::optional<guest_surface_range> dma_transfer_range;
};
// Platform/VM integration must supply these concrete services before renderer
// activation. No default unprotected VM or legacy GL/VK presentation path.
class rsx_mapping_observer;
class rsx_vm_access;
class arm64_fault_channel;
struct renderer_arm64_fault_bindings
{
	std::shared_ptr<rsx_vm_access> vm;
	// Grant a lease only for an original parked PC known to be immutable
	// since the fault, and keep its executable identity until completion.
	std::function<std::shared_ptr<const void>(std::uint64_t pc)> retain_instruction;
	// Publish the channel to the platform bridge. The bridge owns producer
	// registration, passive-lock release, signal routing and render wakeups.
	// Publication itself must not wait for a fault to be handled.
	std::function<void(std::shared_ptr<const arm64_fault_channel>)> install_channel;
};
struct renderer_bindings
{
	std::shared_ptr<guarded_guest_memory> memory;
	std::shared_ptr<rsx_mapping_observer> vm_mapping_observer;
	// Synchronously run a short CPU-only capture while excluding all writers
	// and mapping transitions. Must execute exactly once on the calling RSX thread.
	// The real VM provider's capture_cpu supplies nested guarded view reuse.
	cpu_capture_dispatch capture_epoch;
	// Atomic big-endian label store inside the same guarded memory provider.
	std::function<void(std::uint32_t, std::uint32_t)> store_label;
	// Atomic sixteen-byte timestamp/value/zero-padding report in the same provider.
	std::function<void(std::uint32_t, std::uint64_t, std::uint32_t)> store_report;
	// Register render-thread cache/context owners with fault/DMA dispatch.
	std::function<void(context&, surface_cache&, guest_texture_cache&)> bind_runtime;
	// Runs only on the render owner,after foreign CPU/VM passive unlock. This
	// is fault recovery,not a substitute for transactional producer write hooks.
	std::function<bool(const memory_fault_request&)> memory_fault;
	// Optional platform cleanup after the built-in native cache service,on the
	// render owner outside VM/task/engine locks. Must be idempotent on retry and
	// must never protect/write a reused guest VA as if it were the old backing.
	std::function<void(const utils::address_range32&, rsx::invalidation_cause)> memory_invalidated;
	// Service-specific work only; renderer always runs the common RSX task pump.
	std::function<void(rsx::FIFO::state)> local_task;
	// Retire and drain every borrowed runtime reference, including after a
	// partially failed bind. Report errors only after references are retired.
	std::function<void()> shutdown;
	std::optional<renderer_arm64_fault_bindings> arm64_faults;
	std::size_t heap_bytes = 512u << 20, snapshot_bytes = 64u << 20;
	bool writer_and_mapping_hooks_installed = false;
};

// Actual GSRender subclass. Built/checked without constructing it or starting
// RPCS3. Application factory/platform/protection service installation remains
// external until its contracts are implemented and verified.
class gs_renderer : public rsx_program_frontend<GSRender>
{
	struct runtime;
	renderer_bindings m_bindings;
	std::shared_ptr<tile_state> m_tiles;
	std::shared_ptr<io_mapping_epoch> m_io_mapping = std::make_shared<io_mapping_epoch>();
	render_tasks m_tasks;
	mapping_notifications m_mapping_notifications;
	std::unique_ptr<runtime> m_runtime;
	std::atomic<bool> m_failed = false;
	std::mutex m_instruction_channel_mutex;
	std::shared_ptr<arm64_fault_channel> m_instruction_channel;
	static utils::serial* validate_bindings(utils::serial* archive, const renderer_bindings& bindings);
	void fail_renderer() noexcept;
	void set_instruction_channel(std::shared_ptr<arm64_fault_channel> channel);
	[[nodiscard]] std::optional<framebuffer_snapshot> current_framebuffer(rsx::framebuffer_creation_context mode);
	[[nodiscard]] runtime& device_runtime();
	void service_mapping_notifications(std::size_t range_limit = 256);

  public:
	gs_renderer(utils::serial* archive, renderer_bindings bindings);
	~gs_renderer() override;
	u64 get_cycles() override;
	void on_init_thread() override;
	void on_exit() override;
	void begin() override;
	void end() override;
	void begin_conditional_rendering(const std::vector<rsx::reports::occlusion_query_info*>& sources) override;
	void end_conditional_rendering() override;
	void clear_surface(u32 mask) override;
	void flip(const rsx::display_flip_info_t& info) override;
	bool on_access_violation(u32 address, bool writing) override;
	std::optional<host_write_journal::intent> on_native_host_write(void* destination, const void* source, u64 bytes) override;
	void on_io_mapping_changed(u32 offset, u32 size) noexcept override;
	bool defer_memory_map_notification(u32 address, u32 size) noexcept override;
	bool defer_memory_unmap_notification(u32 address, u32 size) noexcept override;
	void update_tile(u32 index, const std::function<void(GcmTileInfo&)>& mutation) override;
	bool is_tile_bound(u32 index) const override;
	void on_invalidate_memory_range(const utils::address_range32& range, rsx::invalidation_cause cause) override;
	void do_local_task(rsx::FIFO::state state) override;
	void on_semaphore_acquire_wait() override;
	bool release_GCM_label(u32 type, u32 address, u32 value) override;
	bool release_GCM_flip_label(u32 address, u32 value) override;
	bool requires_native_buffer_transfer() const override { return true; }
	bool buffer_transfer(u32 notify) override;
	bool requires_native_scaled_image() const override { return true; }
	bool scaled_image_from_memory(const rsx::blit_src_info& source, const rsx::blit_dst_info& destination, bool interpolate) override;
};
} // namespace metal4
