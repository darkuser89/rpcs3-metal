#pragma once

#include "Emu/RSX/GSFrameBase.h"
#include "Emu/emu_callbacks.h"
#include "M4Frame.h"
#include "M4MainThreadDispatch.h"

namespace metal4
{
inline main_thread_dispatch make_rsx_frame_dispatch()
{
	const auto callback = g_emu_callbacks.call_from_main_thread;
	if (!callback)
		throw std::invalid_argument("RPCS3 main-thread callback is not installed");
	const auto post = [callback](std::function<void()> operation)
	{
		// Direct installed GUI callback: no emulation-stop filter and no borrowed
		// wake-up address. Our owned job handles completion/cancellation/errors.
		callback(std::move(operation), nullptr);
	};
	const auto probe = []
	{ return bool([NSThread isMainThread]); };
	return main_thread_dispatch(post, probe);
}

// Actual GSFrameBase handle contract on macOS is void* -> NSView. These calls
// never create a view/application/window. invoke joins running closures before
// returning,so renderer-owned frame and GSFrame references cannot outlive it.
class rsx_frame_service final
{
	native_frame& m_frame;
	main_thread_dispatch m_dispatch;

  public:
	explicit rsx_frame_service(native_frame& frame) : m_frame(frame), m_dispatch(make_rsx_frame_dispatch()) {}
	void initialize(GSFrameBase* platform)
	{
		if (!platform)
			throw std::invalid_argument("Metal4 renderer has no GSFrame platform object");
		const auto attach = [&]
		{
			const auto handle = platform->handle();
			if (!handle)
				throw std::runtime_error("GSFrame has no native NSView handle");
			m_frame.attach((__bridge NSView*)handle);
		};
		m_dispatch.invoke(attach);
	}
	void shutdown()
	{
		const auto detach = [&]
		{ m_frame.detach(); };
		m_dispatch.invoke(detach);
	}
};
} // namespace metal4
