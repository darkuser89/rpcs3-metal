#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__) || !defined(__OBJC__)
#error "The native Metal 4 backend requires Objective-C++ on Apple Silicon."
#endif

// Foundation transitively imports Carbon's global Vector128 typedef,which
// conflicts with RPCS3's Vector128 concept in real renderer translation units.
// Scope the SDK spelling change to these imports;never rename RSX code.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 M4CarbonVector128
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#pragma pop_macro("Vector128")
#import <TargetConditionals.h>

#if !TARGET_OS_OSX
#error "The native Metal 4 backend requires macOS."
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace metal4
{

// CPU readiness only. Callers still record the original native queue wait
// and validate its device; opaque/nil events cannot prove readiness here.
[[nodiscard]] inline bool shared_event_reached(id<MTLEvent> event, std::uint64_t value)
{
	return [event conformsToProtocol:@protocol(MTLSharedEvent)] &&
		static_cast<id<MTLSharedEvent>>(event).signaledValue >= value;
}

struct completion_state;
// A CPU wait deadline is distinct from a completed GPU submission failure.
// The token may finish immediately after the deadline;owners still choose
// when to retry/finalize the retained transaction.
class submission_wait_timeout final : public std::runtime_error
{
  public:
	submission_wait_timeout():std::runtime_error("Timed out waiting for Metal 4 submission"){}
};
struct upload_plan_state;
class heap_budget;
class helper_catalog;
struct helper_catalog_state;
class constant_upload_cache;
class bindings;
class guest_texture_resolver;
struct texture_region;
enum class texture_plane;

struct upload_slice
{
	id<MTLBuffer> buffer;
	std::size_t offset;
	std::size_t size;
};

struct upload_requirement
{
	std::size_t size;
	std::size_t alignment = 256;
};

// An immutable, single-use reservation of native upload storage. Holding it
// protects the buffers against cache trimming before the draw is encoded.
class upload_plan final
{
	friend class context;
	std::shared_ptr<upload_plan_state> m_state;
	explicit upload_plan(std::shared_ptr<upload_plan_state> state) : m_state(std::move(state)) {}
};

struct argument_table_statistics
{
	std::size_t cached = 0, capacity = 0;
	std::uint64_t created = 0, reused = 0, bypassed = 0;
};
struct command_encoding_statistics
{
	std::uint64_t compute_encoders = 0, render_encoders = 0;
	std::uint64_t buffer_blit_queue_barriers = 0, buffer_blit_barriers = 0;
	std::uint64_t buffer_blit_alias_barriers = 0, buffer_blit_barriers_elided = 0;
	std::uint64_t texture_blit_queue_barriers = 0, texture_blit_barriers = 0;
	std::uint64_t texture_blit_alias_barriers = 0, texture_blit_barriers_elided = 0;
	std::uint64_t checked_blit_raw_transitions = 0;
	std::uint64_t checked_dispatches = 0, dispatch_queue_barriers = 0;
	std::uint64_t dispatch_barriers = 0, dispatch_barriers_elided = 0;
	std::uint64_t blit_to_dispatch_barriers = 0, dispatch_to_blit_barriers = 0;
	std::uint64_t full_render_barriers = 0, fragment_render_barriers = 0;
	std::uint64_t guest_draw_full_barriers = 0, guest_draw_fragment_barriers = 0;
};

// A completion remains valid even after its command slot has been reused.
class submission final
{

	std::shared_ptr<completion_state> m_state;

	explicit submission(std::shared_ptr<completion_state> state);
	friend class context;

public:

	// Immutable recording/completion identity, independent of slot reuse.
	[[nodiscard]] bool same_batch(const submission& other) const noexcept { return m_state == other.m_state; }
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancelled() const;
	// Requests cancellation only while queued before the native commit point.
	// True guarantees this batch cannot execute. wait() still observes cleanup
	// and reports cancellation; GPU work past that point cannot be cancelled.
	[[nodiscard]] bool cancel_pending() const;
	// A timeout never permits resetting an allocator or releasing GPU resources.
	void wait(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
};

// Encoding and submission calls belong to one render thread. Completion callbacks
// only touch separately synchronized state and never access the context itself.
class context final
{

	struct impl;
	std::unique_ptr<impl> m_impl;
	std::unique_ptr<constant_upload_cache> m_constants;
	std::unique_ptr<constant_upload_cache> m_vertex_uploads;
	std::unique_ptr<constant_upload_cache> m_geometry_uploads;
	friend class helper_catalog;
	friend class guest_texture_resolver;
	friend class surface_color_transfer;
	friend class surface_depth_transfer;
	friend class sample_transfer;
	friend class surface_banked_transfer;
	friend class swizzle_engine;
	friend class transfer_engine;
	friend class dma_compute;
	friend class scaled_image_compute;
	friend class vertex_fetch_engine;
	friend class fsr_upscaler;
	friend class border_coverage_cache;
	friend class border_texture_preparation;
	friend class versioned_guest_texture_job;
	friend class vertex_shader;
	friend class render_commands;
	friend class draw_executor;
	friend struct present_image;
	void end_guest_render_pass();
	friend class query_pool;
	friend class surface_cache;
	[[nodiscard]] bool can_begin_after(const submission& predecessor) const;
	[[nodiscard]] std::optional<upload_plan> plan_uploads(std::span<const upload_requirement> requirements, bool allow_recovery);
	// Same-queue dependency. The worker requires success unless the caller
	// supplies a retained, shared uint32 slot for the query outcome.
	void depend_on_predecessor(const submission& producer, std::byte* query_outcome = nullptr);
	friend class depth_bounds_pass;
	friend void copy_buffer(context&, id<MTLBuffer>, std::size_t, id<MTLBuffer>, std::size_t, std::size_t);
	friend void fill_buffer(context&, id<MTLBuffer>, std::size_t, std::size_t, std::uint8_t);
	friend void buffer_to_texture(context&, id<MTLBuffer>, std::size_t, id<MTLTexture>,
		const texture_region&, std::size_t, std::size_t, texture_plane);
	friend void texture_to_buffer(context&, id<MTLTexture>, const texture_region&, id<MTLBuffer>,
		std::size_t, std::size_t, std::size_t, texture_plane);
	friend void copy_texture(context&, id<MTLTexture>, const texture_region&, id<MTLTexture>,
		const texture_region&, texture_plane);
	[[nodiscard]] id<MTL4ComputeCommandEncoder> compute_encoder(MTLStages before);
	enum class render_writes { opaque, attachments };
	struct shared_guest_pass
	{
		const bindings& fragment;
		MTLVisibilityResultMode visibility;
		std::size_t visibility_offset;
		bool reused = false;
		bool visibility_changed = false;
	};
	[[nodiscard]] bool render_reads_wait_needed(const bindings& inputs, id<MTLBuffer> geometry,
		id<MTLBuffer> owned_visibility) const;
	[[nodiscard]] id<MTL4RenderCommandEncoder> render_encoder(MTL4RenderPassDescriptor* descriptor, MTLStages before,
		render_writes writes, const bindings* vertex = nullptr, id<MTLBuffer> geometry = nil, shared_guest_pass* shared = nullptr);
	static constexpr std::size_t max_dispatch_accesses = 16;
	struct accessed_resource
	{
		id<MTLResource> resource;
		std::size_t offset = 0, size = 0; // Zero for a texture access.
		bool writes = false;
		// Optional proven subresource coordinates, relative to the named texture.
		std::optional<std::size_t> mip = {}, slice = {};
		MTLRegion box = {}; // Zero width means the whole selected subresource.
	};
	[[nodiscard]] id<MTL4ComputeCommandEncoder> checked_blit(accessed_resource source, accessed_resource destination);
	[[nodiscard]] id<MTL4ComputeCommandEncoder> checked_compute(MTLStages stage, std::span<const accessed_resource> resources, bool texture_blit);
	void checked_dispatch(id<MTLComputePipelineState> pipeline, const bindings& arguments, std::span<const accessed_resource> resources,
		MTLSize grid, MTLSize group);
	void checked_dispatch(id<MTLComputePipelineState> pipeline, const bindings& arguments,
		std::span<const accessed_resource> resources, std::size_t threads);
	// Shared device/options catalogue; native feedback retains individual PSOs.
	std::shared_ptr<helper_catalog_state> m_helpers;
	const std::shared_ptr<const std::byte> m_identity = std::make_shared<const std::byte>();

public:

	enum class recording_overlap { unrestricted, explicit_predecessor };
	explicit context(std::uint32_t max_in_flight = 3, std::size_t argument_tables_per_slot = 64,
		std::chrono::milliseconds event_wait_timeout = std::chrono::seconds(30),
		recording_overlap overlap = recording_overlap::unrestricted);
	~context();
	context(const context&) = delete;
	context& operator=(const context&) = delete;
	context(context&&) = delete;
	context& operator=(context&&) = delete;

	[[nodiscard]] id<MTLDevice> device() const noexcept;
	// Opaque context/queue identity, independent of recycled object addresses.
	// Weak observers do not keep a context, queue or GPU resource alive.
	[[nodiscard]] std::weak_ptr<const std::byte> queue_identity() const noexcept { return m_identity; }
	// Install once,before the first buffer allocation or recording. All native
	// resource-helper buffers and retained upload pages share this ledger.
	void set_buffer_budget(std::shared_ptr<heap_budget> budget);
	[[nodiscard]] std::shared_ptr<heap_budget> buffer_ledger() const noexcept;
	[[nodiscard]] id<MTLBuffer> allocate_buffer(std::size_t bytes, MTLStorageMode storage) __attribute__((ns_returns_retained));
	// Drops cached constant versions, unused recording pages and pages from completed/cancelled slots.
	// Never waits for GPU work;external owners can keep their budget charge.
	[[nodiscard]] std::size_t trim_upload_pages();
	// Context-local immutable constant uploads; also released by trimming.
	[[nodiscard]] constant_upload_cache& constant_uploads();
	[[nodiscard]] constant_upload_cache& vertex_uploads();
	[[nodiscard]] constant_upload_cache& geometry_uploads();
	// Exclusive table for this encoding, with all binding slots cleared. Only
	// completed/cancelled slots may reuse tables. Excess tables live until this
	// submission completes but are not retained in the bounded reusable pool.
	[[nodiscard]] id<MTL4ArgumentTable> argument_table();
	[[nodiscard]] argument_table_statistics argument_statistics() const noexcept;
	[[nodiscard]] command_encoding_statistics encoding_statistics() const noexcept;
	// CPU-only preflight for preparation that may later begin encoding. Does
	// not wait for an in-flight slot or change a caller's current recording.
	void validate_begin() const;
	// The last submitted work, excluding any open recording.
	[[nodiscard]] std::optional<submission> last_submission() const;
	// In explicit-predecessor mode, only work with this successful-
	// completion dependency can record ahead of the previous submission.
	void begin(const submission* predecessor = nullptr);
	// The context owns encoder lifetimes; callers must not call endEncoding.
	[[nodiscard]] id<MTL4ComputeCommandEncoder> compute();
	[[nodiscard]] id<MTL4RenderCommandEncoder> render(MTL4RenderPassDescriptor* descriptor);
	// Borrowed encoder, or nil after a pass switch, submit, or cancel.
	[[nodiscard]] id<MTL4RenderCommandEncoder> current_render_encoder() const noexcept;
	void end_encoding();
	// Synchronizes commands within the current pass; masks must match its stages.
	void barrier(MTLStages after, MTLStages before, bool resource_alias = false);
	// Metal 4 command buffers don't retain referenced resources automatically.
	// Register every indirect resource as well as every attachment and buffer.
	void use_resource(id<MTLResource> resource);
	void keep_alive(id object);
	// Completion-owned leases keep pooled allocations unavailable for reuse
	// until GPU feedback. Cancellation releases unsubmitted leases immediately.
	void keep_alive(std::shared_ptr<void> owner);
	// A shared uint32 status written by GPU validation. Nonzero becomes a
	// submission error after completion, checked by wait(). Register before
	// submitting; validated work must complete before drawable presentation.
	void require_gpu_success(id<MTLBuffer> status, std::size_t offset, NSString* message);
	// Defers a native queue wait until submit, before the whole command buffer.
	// Cancel discards it. Signals must come from CPU or another queue, since a
	// later signal on this same queue cannot satisfy a preceding queue wait.
	// Shared counters gate a serial submit worker via native CPU shared-event
	// waits before Metal4 commit: native GPU queue waits were observed to pass
	// unsignaled counters. The render thread remains asynchronous. Submitted
	// waiting work keeps all leases, even after context destruction or CPU timeout.
	// Shared-event gating has a separate finite deadline from submit() (30s by
	// default). Expiry cancels the uncommitted batch, releasing the serial worker.
	// Native non-shared event waits, once committed, cannot be cancelled here.
	void wait_event(id<MTLEvent> event, std::uint64_t value);
	// Copies CPU data into slot-owned shared storage. Storage is reused only
	// after the GPU has completed this slot's previous submission.
	[[nodiscard]] upload_slice upload(std::span<const std::byte> bytes, std::size_t alignment = 256);
	// Allocation-only admission before a surface transaction. Lists the exact
	// upload sequence; one unconsumed reservation per context. Existing unused
	// arena space is reused. Encoding may cross
	// a completed batch boundary, but cannot borrow an unfinished foreign slot.
	[[nodiscard]] upload_plan prepare_uploads(std::span<const upload_requirement> requirements);
	// All upload() calls in this callback consume the reserved sequence without
	// allocating another buffer. Mismatch, nesting and reuse are rejected. The
	// callback must not begin, submit or cancel command buffers itself.
	void with_uploads(const upload_plan& plan, const std::function<void()>& encode);
	// Select before encoding commands that write to this drawable.
	void target_drawable(id<CAMetalDrawable> drawable);
	// Observes the currently recording batch before submit. Cancel completes
	// this token as cancelled;it never aliases a later reused command slot.
	[[nodiscard]] submission recording_submission();
	[[nodiscard]] submission submit();
	void cancel();
	void wait_idle();
	// Render-thread shutdown: cancels recording and queued batches, joins the
	// CPU submit worker, and refuses new recordings. Already committed GPU work
	// retains its resources through feedback. Repeated shutdown is harmless.
	// Destruction alone allows already submitted batches their bounded gate wait.
	void shutdown();
	// Acknowledges already completed failed slots so fresh work can be encoded.
	// Existing submission tokens retain their errors; pending work is untouched.
	void recover_completed_errors();
};

}
