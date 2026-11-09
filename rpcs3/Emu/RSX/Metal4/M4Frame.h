#pragma once

#include "M4FramePlan.h"
#include "M4Overlay.h"

#include <functional>
#include <vector>

@class NSView;

namespace metal4
{
struct captured_frame
{
	frame_extent extent;
	std::vector<std::uint8_t> bgra;
};

class frame_capture_job final
{
	friend class native_frame;
	frame_extent m_extent;
	std::size_t m_row_pitch;
	id<MTLBuffer> m_readback;
	submission m_work;
	frame_capture_job(frame_extent extent, std::size_t row_pitch, id<MTLBuffer> readback, submission work);

  public:
	frame_capture_job(frame_capture_job&&) noexcept = default;
	frame_capture_job& operator=(frame_capture_job&&) noexcept = default;
	[[nodiscard]] bool ready() const;
	// Retire a completed capture and check its submission without copying pixels.
	bool discard_if_ready();
	[[nodiscard]] std::optional<captured_frame> take_if_ready();
};

// Render-thread owner. AppKit attach/update/detach run only on the main thread;
// caller dispatches them without holding guest-CPU/VM locks. Never creates a
// window or NSApplication. An unattached frame cannot acquire a drawable.
class native_frame final
{
	friend class memory_pressure_owner;
	friend class frame_geometry_test_access;
	friend class frame_capture_test_access;
	struct acquisition_ticket
	{
		CAMetalLayer* layer;
		std::uint64_t epoch;
	};
	[[nodiscard]] std::optional<acquisition_ticket> prepare_acquisition(frame_pacing pacing);
	[[nodiscard]] bool current_acquisition(const acquisition_ticket& ticket) const;
	[[nodiscard]] std::optional<present_image> prepare_image(const std::optional<present_image>& image,
		id<MTLTexture> target, const frame_policy& policy);
	void encode_prepared(const std::optional<present_image>& image, id<MTLTexture> target);
	[[nodiscard]] overlay_renderer::prepared_frame prepare_overlays(const overlay_frame& frame,
		const std::optional<present_image>& image, id<MTLTexture> target);
	[[nodiscard]] id<MTLTexture> make_overlay_backbuffer(const std::optional<present_image>& image, id<MTLTexture> target);
	bool reclaim_overlay_image();
	void encode_overlays(const overlay_renderer::prepared_frame& frame, const std::optional<present_image>& image, id<MTLTexture> target);
	[[nodiscard]] std::optional<frame_capture_job> encode_capture(const std::optional<present_image>& image,
		frame_extent extent, MTLPixelFormat format, const overlay_renderer::prepared_frame* overlays = nullptr);
	struct impl;
	std::unique_ptr<impl> m_impl;
	context& m_context;
	presenter m_presenter;

  public:
	explicit native_frame(context& ctx);
	~native_frame();
	native_frame(const native_frame&) = delete;
	native_frame& operator=(const native_frame&) = delete;
	void attach(NSView* view);
	void update_from_view();
	void detach();
	// Layer metadata can be checked without creating a view/window. Main thread
	// only;geometry publication and acquisition are synchronized internally.
	void update_geometry(double width, double height, double scale, bool visible);
	void update_geometry(const frame_geometry& geometry);
	[[nodiscard]] frame_geometry_statistics geometry_statistics() const;
	[[nodiscard]] frame_extent extent() const;
	[[nodiscard]] bool attached() const;
	[[nodiscard]] CAMetalLayer* layer() const;
	// Encode the same output policy offscreen without acquiring a drawable.
	void encode(const std::optional<present_image>& image, id<MTLTexture> target, const frame_policy& policy,
		const overlay_frame& overlays = {});
	// Render thread only;up to three outstanding submissions,waits outside the
	// geometry lock. Uses actual acquired texture dimensions after any resize.
	// An empty capture slot requests one optional readback in the same submission.
	[[nodiscard]] bool present(const std::optional<present_image>& image, const frame_policy& policy,
		std::optional<frame_capture_job>* capture = nullptr, const std::function<overlay_frame(frame_extent)>& overlays = {},
		bool capture_overlays = false);
	void drain();
	// Render-owner cache release; submitted work retains its own resources.
	bool reclaim_fsr();
};
} // namespace metal4
