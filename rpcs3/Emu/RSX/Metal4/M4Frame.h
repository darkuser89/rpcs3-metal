#pragma once

#include "M4FramePlan.h"
#include "M4Present.h"

@class NSView;

namespace metal4
{
// Render-thread owner. AppKit attach/update/detach run only on the main thread;
// caller dispatches them without holding guest-CPU/VM locks. Never creates a
// window or NSApplication. An unattached frame cannot acquire a drawable.
class native_frame final
{
	friend class memory_pressure_owner;
	friend class frame_geometry_test_access;
	struct acquisition_ticket
	{
		CAMetalLayer* layer;
		std::uint64_t epoch;
	};
	[[nodiscard]] std::optional<acquisition_ticket> prepare_acquisition(bool vsync);
	[[nodiscard]] bool current_acquisition(const acquisition_ticket& ticket) const;
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
	void encode(const std::optional<present_image>& image, id<MTLTexture> target, const frame_policy& policy);
	// Render thread only;up to three outstanding submissions,waits outside the
	// geometry lock. Uses actual acquired texture dimensions after any resize.
	[[nodiscard]] bool present(const std::optional<present_image>& image, const frame_policy& policy);
	void drain();
	// Render-owner cache release; submitted work retains its own resources.
	bool reclaim_fsr();
};
} // namespace metal4
