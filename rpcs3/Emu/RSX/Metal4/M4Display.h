#pragma once

#include "M4DisplaySnapshot.h"
#include "M4Present.h"
#include "M4SurfaceCache.h"
#include "M4CPUCapture.h"
#include "M4GuestTexture.h"

namespace metal4
{
struct display_cache_state;
struct display_source_version;
struct display_upload_dependency
{
	id<MTLEvent> event;
	std::uint64_t value;
};
struct display_cache_limits
{
	std::size_t entries = 3, bytes = 64u << 20;
};
struct display_cache_statistics
{
	std::size_t entries = 0, bytes = 0;
	std::uint64_t uploads = 0, hits = 0, pending_hits = 0, published = 0, evictions = 0, native_snapshots = 0;
	std::uint64_t native_submissions = 0;
	std::uint64_t native_casts = 0, native_cast_hits = 0;
	std::uint64_t mixed_casts = 0, captured_vm_bytes = 0, captured_vm_ranges = 0;
	std::uint64_t assembly_casts = 0, assembly_hits = 0;
};
// finish validates the owned source receipt after native success, outside
// CPU/VM exclusion, on its preparing thread.
class display_upload_job final
{
	struct state;
	std::shared_ptr<state> m_state;
	display_format m_format;
	display_upload_job(std::shared_ptr<state> state, display_format format);
	[[nodiscard]] static present_image owned_image(const state& state, display_format format);
	friend class display_resolver;
	friend class display_frame_job;
	friend struct display_cache_state;
  public:
	~display_upload_job();
	display_upload_job(display_upload_job&&) noexcept;
	display_upload_job& operator=(display_upload_job&&) noexcept;
	display_upload_job(const display_upload_job&) = delete;
	display_upload_job& operator=(const display_upload_job&) = delete;
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancel_pending() const;
	// CPU-only; caller holds the same producer/VM epoch as display capture.
	[[nodiscard]] bool current() const;
	[[nodiscard]] present_image finish(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
};
// One owned AV frame: separate jobs/receipts for each physical eye, followed
// by joint final validation in one producer epoch. No half-frame accessor.
class display_frame_job final
{
	struct state;
	std::shared_ptr<state> m_state;
	explicit display_frame_job(std::shared_ptr<state> state);
	friend class display_resolver;
  public:
	~display_frame_job();
	display_frame_job(display_frame_job&&) noexcept;
	display_frame_job& operator=(display_frame_job&&) noexcept;
	display_frame_job(const display_frame_job&) = delete;
	display_frame_job& operator=(const display_frame_job&) = delete;
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancel_pending() const;
	// Caller owns the same CPU/VM epoch as the AV snapshot.
	[[nodiscard]] bool current() const;
	// Validate both immutable eyes in that epoch. The returned image carries
	// the submissions needed to encode GPU consumers safely.
	[[nodiscard]] present_image image_for_encoding() const;
	[[nodiscard]] present_image finish(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
};
// Native surface epochs take precedence over CPU backbuffers. VM versions
// own active-byte watches and optional tile/mapping receipts; immutable uploads
// and cached images are admitted under the shared ledger. Linear native MSAA
// uses individual expanded samples; AV storage casts pack/crop/decode on GPU.
// Local tile tags leave storage transparent; AV receipts still watch changes.
// Morton color sources prove every requested byte before GPU packing into
// their original storage order and decoding the linear AV window.
// Missing cells within that allocation use only owned captured VM fragments;
// native bytes and AV row padding are never read for that composition.
class display_resolver final
{
	context& m_context;
	std::shared_ptr<guest_surface_memory> m_memory;
	std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> m_translate;
	std::shared_ptr<heap_budget> m_budget;
	std::size_t m_limit;
	cpu_capture_dispatch m_capture;
	std::shared_ptr<tile_state> m_tiles;
	std::shared_ptr<display_cache_state> m_cache;
	guest_texture_resolver m_upload;
	struct upload_batch;
	[[nodiscard]] std::shared_ptr<display_upload_job::state> prepare_impl(const display_snapshot& image, surface_cache& surfaces,
		upload_batch& batch, bool require_ledger);
	[[nodiscard]] std::shared_ptr<display_upload_job::state> prepare_assembly(const std::shared_ptr<display_source_version>& version,
		surface_cache& surfaces,const guest_texture_layout& plan,upload_batch& batch);

  public:
	// Capture/translation callbacks retain their services for outstanding jobs.
	display_resolver(context& ctx, std::shared_ptr<guest_surface_memory> memory,
					 std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate,
					 std::shared_ptr<heap_budget> budget, std::size_t snapshot_limit,
					 cpu_capture_dispatch capture = {}, std::shared_ptr<tile_state> tiles = {}, display_cache_limits limits = {});
	// Resolves native aliases first,then matching completed/pending VM versions.
	// New VM uploads and native copies return after submit,without an immediate
	// completion wait. Source materialization/previous transactions can wait first.
	// Requires the installed shared buffer/texture ledger. No borrowed VM data.
	// Dependencies gate new work; reused versions retain their original work.
	[[nodiscard]] display_upload_job prepare(const display_snapshot& image, surface_cache& surfaces,
		std::span<const display_upload_dependency> dependencies = {});
	[[nodiscard]] display_frame_job prepare_frame(const display_snapshot& image, surface_cache& surfaces,
		std::span<const display_upload_dependency> dependencies = {});
	[[nodiscard]] present_image resolve(const display_snapshot& image, surface_cache& surfaces);
	[[nodiscard]] display_cache_statistics statistics() const;
	void clear();
	bool reclaim_one();
	[[nodiscard]] std::shared_ptr<heap_budget> heap_ledger() const noexcept { return m_budget; }
};
} // namespace metal4
