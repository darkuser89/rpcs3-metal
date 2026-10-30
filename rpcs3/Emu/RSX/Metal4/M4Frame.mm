#include "M4Frame.h"
#include "M4FSR.h"
#include "M4MainRunLoop.h"

#import <AppKit/NSView.h>
#import <AppKit/NSWindow.h>
#import <QuartzCore/CATransaction.h>

#include <deque>
#include <dispatch/dispatch.h>
#include <mutex>

namespace metal4
{
namespace
{
void require_main()
{
	if (![NSThread isMainThread])
		throw std::logic_error("Metal4 frame geometry/attachment requires the main thread");
}
} // namespace
struct native_frame::impl
{
	// The observer only locks a weak geometry owner. It never retains or
	// accesses the renderer, context, FSR cache or pending submissions.
	struct geometry
	{
		mutable std::mutex mutex;
		CAMetalLayer* layer = [CAMetalLayer layer];
		__weak NSView* view = nil;
		CALayer* root = nil;
		bool alive = true, attached = false, visible = false, enabled_backing = false, vsync = true;
		frame_extent extent;
		std::optional<frame_geometry> current;
		frame_geometry_statistics stats;
		std::exception_ptr error;
		std::uint64_t binding_epoch = 0;
		std::uint32_t limit = 0;

		void publish(const frame_geometry& value, std::optional<std::uint64_t> binding = {})
		{
			if (!std::isfinite(value.x) || !std::isfinite(value.y))
				throw std::invalid_argument("Invalid native frame bounds origin");
			const auto size = plan_drawable_extent(value.width, value.height, value.scale, limit);
			std::lock_guard lock(mutex);
			if (!alive || (binding && (!attached || *binding != binding_epoch))) return;
			if (current == value && visible == value.visible && !error) return;
			const bool layer_changed = !current || current->width != value.width || current->height != value.height ||
				current->scale != value.scale || current->x != value.x || current->y != value.y;
			if (layer_changed)
			{
				[CATransaction begin];
				[CATransaction setDisableActions:YES];
				layer.frame = CGRectMake(value.x, value.y, value.width, value.height);
				layer.bounds = CGRectMake(0, 0, value.width, value.height);
				layer.contentsScale = value.scale;
				layer.drawableSize = CGSizeMake(size.width, size.height);
				[CATransaction commit];
				++stats.layer_transactions;
			}
			current = value;
			extent = size;
			visible = value.visible;
			error = {};
			++stats.epoch;
		}
		void update_from_view()
		{
			require_main();
			NSView* source;
			std::uint64_t binding;
			{
				std::lock_guard lock(mutex);
				source = view;
				if (!alive || !attached || !source || source.layer != root)
					throw std::logic_error("Native frame view/backing changed;detach before rebinding");
				binding = binding_epoch;
			}
			const auto bounds = source.bounds;
			NSWindow* window = source.window;
			const double scale = window ? window.backingScaleFactor : 1;
			publish({bounds.size.width, bounds.size.height, scale, bounds.origin.x, bounds.origin.y,
				!source.hiddenOrHasHiddenAncestor && window && window.isVisible}, binding);
		}
		void observe() noexcept
		{
			@autoreleasepool
			{
				{
					std::lock_guard lock(mutex);
					if (!alive || !attached) return;
				}
				try { update_from_view(); }
				catch (...)
				{
					std::lock_guard lock(mutex);
					if (!alive || !attached) return;
					// Block acquisition until a successful fresh observation. Errors
					// cross to the renderer as owned exception state, never through CF.
					if (!error) ++stats.epoch;
					error = std::current_exception();
					visible = false;
				}
			}
		}
	};
	std::shared_ptr<geometry> gui = std::make_shared<geometry>();
	std::unique_ptr<main_run_loop_observer> observer;
	std::deque<submission> pending;
	std::unique_ptr<fsr_upscaler> fsr;
};
native_frame::native_frame(context& ctx) : m_impl(std::make_unique<impl>()), m_context(ctx), m_presenter(ctx)
{
	auto& gui = *m_impl->gui;
	gui.limit = static_cast<std::uint32_t>(max_2d_texture_dimension(ctx.device()));
	m_presenter.configure_layer(gui.layer);
	gui.layer.maximumDrawableCount = 3;
	gui.layer.allowsNextDrawableTimeout = YES;
	gui.layer.presentsWithTransaction = NO;
	gui.vsync = gui.layer.displaySyncEnabled;
	CGColorSpaceRef color = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
	if (!color)
		throw std::runtime_error("Metal4 SDR color-space allocation failed");
	gui.layer.colorspace = color;
	CGColorSpaceRelease(color);
}
// Normal shutdown still joins the main-thread detach. The fallback invalidates
// observation without waiting for UI work. A running callback has only its own
// geometry owner; queued layer removal has no renderer/context reference.
native_frame::~native_frame()
{
	CAMetalLayer* orphan = nil;
	{
		std::lock_guard lock(m_impl->gui->mutex);
		m_impl->gui->alive = false;
		m_impl->gui->visible = false;
		++m_impl->gui->stats.epoch;
		if (m_impl->gui->attached) orphan = m_impl->gui->layer;
	}
	m_impl->observer.reset();
	if (orphan)
		dispatch_async(dispatch_get_main_queue(), ^{ [orphan removeFromSuperlayer]; });
}
void native_frame::attach(NSView* view)
{
	require_main();
	if (!view)
		throw std::invalid_argument("Metal4 frame requires an existing NSView");
	const auto gui = m_impl->gui;
	{
		std::lock_guard lock(gui->mutex);
		if (gui->attached)
			throw std::logic_error("Metal4 frame already attached");
		gui->enabled_backing = !view.wantsLayer;
		view.wantsLayer = YES;
		if (!view.layer)
		{
			if (gui->enabled_backing) view.wantsLayer = NO;
			throw std::runtime_error("Native frame view has no layer backing");
		}
		gui->view = view;
		gui->root = view.layer;
		[gui->root addSublayer:gui->layer];
		gui->attached = true;
		++gui->binding_epoch;
		++gui->stats.epoch;
	}
	try
	{
		update_from_view();
		const std::weak_ptr<impl::geometry> weak = gui;
		m_impl->observer = std::make_unique<main_run_loop_observer>([weak]
		{
			if (const auto owner = weak.lock()) owner->observe();
		});
	}
	catch (...)
	{
		detach();
		throw;
	}
}
void native_frame::update_from_view()
{
	m_impl->gui->update_from_view();
}
void native_frame::update_geometry(double width, double height, double scale, bool visible)
{
	update_geometry({width, height, scale, 0, 0, visible});
}
void native_frame::update_geometry(const frame_geometry& geometry)
{
	require_main();
	m_impl->gui->publish(geometry);
}
frame_geometry_statistics native_frame::geometry_statistics() const
{
	std::lock_guard lock(m_impl->gui->mutex);
	return m_impl->gui->stats;
}
void native_frame::detach()
{
	require_main();
	m_impl->observer.reset();
	auto& gui = *m_impl->gui;
	std::lock_guard lock(gui.mutex);
	[gui.layer removeFromSuperlayer];
	NSView* view = gui.view;
	if (view && gui.enabled_backing && view.layer == gui.root && !gui.root.sublayers.count)
		view.wantsLayer = NO;
	const bool changed = gui.attached || gui.visible || bool(gui.error);
	gui.view = nil;
	gui.root = nil;
	gui.attached = gui.visible = gui.enabled_backing = false;
	gui.error = {};
	++gui.binding_epoch;
	if (changed) ++gui.stats.epoch;
}
frame_extent native_frame::extent() const
{
	std::lock_guard lock(m_impl->gui->mutex);
	return m_impl->gui->extent;
}
bool native_frame::attached() const
{
	std::lock_guard lock(m_impl->gui->mutex);
	return m_impl->gui->attached && m_impl->gui->view != nil;
}
CAMetalLayer* native_frame::layer() const
{
	std::lock_guard lock(m_impl->gui->mutex);
	return m_impl->gui->layer;
}
void native_frame::encode(const std::optional<present_image>& image, id<MTLTexture> target, const frame_policy& policy)
{
	@autoreleasepool
	{
		validate_frame_policy(policy);
		if (!policy.fsr || !image) (void)reclaim_fsr();
		if (!target || target.device != m_context.device() || target.textureType != MTLTextureType2D || target.sampleCount != 1 ||
			!(target.usage & MTLTextureUsageRenderTarget))
			throw std::invalid_argument("Invalid native frame target");
		if (image)
		{
			(void)validate_texture_region(m_context, image->texture, image->region);
			if (image->region.size.width > UINT32_MAX || image->region.size.height > UINT32_MAX)
				throw std::out_of_range("Native frame source extent exceeds integer planning domain");
			const auto rect = plan_frame_rectangle({static_cast<std::uint32_t>(image->region.size.width), static_cast<std::uint32_t>(image->region.size.height)},
												   {static_cast<std::uint32_t>(target.width), static_cast<std::uint32_t>(target.height)}, policy);
			if (rect.width && rect.height)
			{
				auto prepared = *image;
				prepared.destination = {rect.x, rect.y, rect.width, rect.height};
				prepared.filter = policy.linear ? present_filter::linear : present_filter::nearest;
				prepared.gamma = policy.gamma;
				prepared.limited_rgb = policy.limited_rgb;
				prepared.stereo = policy.stereo;
				// Validate both original eyes and destination aliases before any
				// scaling can replace the original source allocation identities.
				m_presenter.validate(m_context, prepared, target);
				if (policy.fsr)
				{
					if (!m_impl->fsr) m_impl->fsr = std::make_unique<fsr_upscaler>(m_context);
					const auto output = fsr_eye_extent({rect.width, rect.height}, policy.stereo.mode);
					const auto left = m_impl->fsr->encode(m_context, {prepared.texture, prepared.region}, output, policy.fsr_sharpness);
					prepared.texture = left.texture; prepared.region = left.region;
					if (policy.stereo.mode != stereo_mode::disabled)
						prepared.right_eye = m_impl->fsr->encode(m_context, *prepared.right_eye, output, policy.fsr_sharpness, 1);
					else (void)m_impl->fsr->clear_eye(1);
					prepared.filter = present_filter::linear;
				}
				m_presenter.encode(m_context, prepared, target);
				return;
			}
		}
		m_presenter.encode_black(m_context, target);
	}
}
std::optional<native_frame::acquisition_ticket> native_frame::prepare_acquisition(bool vsync)
{
	std::lock_guard lock(m_impl->gui->mutex);
	auto& gui = *m_impl->gui;
	if (gui.error) std::rethrow_exception(gui.error);
	if (!gui.alive || !gui.attached || !gui.view || !gui.visible || !gui.extent.width || !gui.extent.height) return {};
	if (gui.vsync != vsync)
	{
		gui.layer.displaySyncEnabled = vsync;
		gui.vsync = vsync;
		++gui.stats.vsync_writes;
	}
	return acquisition_ticket{gui.layer, gui.stats.epoch};
}
bool native_frame::current_acquisition(const acquisition_ticket& ticket) const
{
	std::lock_guard lock(m_impl->gui->mutex);
	const auto& gui = *m_impl->gui;
	if (gui.error) std::rethrow_exception(gui.error);
	return gui.alive && gui.attached && gui.view && gui.visible && gui.extent.width && gui.extent.height &&
		ticket.layer == gui.layer && ticket.epoch == gui.stats.epoch;
}
bool native_frame::present(const std::optional<present_image>& image, const frame_policy& policy)
{
	@autoreleasepool
	{
		validate_frame_policy(policy);
		const auto ticket = prepare_acquisition(policy.vsync);
		if (!ticket) return false;
		while (!m_impl->pending.empty() && m_impl->pending.front().ready())
		{
			m_impl->pending.front().wait();
			m_impl->pending.pop_front();
		}
		if (m_impl->pending.size() >= 3)
		{
			m_impl->pending.front().wait();
			m_impl->pending.pop_front();
		}
		const auto drawable = [ticket->layer nextDrawable];
		if (!drawable)
			return false;
		if (!current_acquisition(*ticket)) return false;
		m_context.begin();
		try
		{
			m_context.target_drawable(drawable);
			encode(image, drawable.texture, policy);
			m_impl->pending.push_back(m_context.submit());
			return true;
		}
		catch (...)
		{
			m_context.cancel();
			throw;
		}
	}
}
void native_frame::drain()
{
	@autoreleasepool
	{
		while (!m_impl->pending.empty())
		{
			m_impl->pending.front().wait();
			m_impl->pending.pop_front();
		}
	}
}
bool native_frame::reclaim_fsr()
{
	return m_impl->fsr && m_impl->fsr->clear();
}
} // namespace metal4
