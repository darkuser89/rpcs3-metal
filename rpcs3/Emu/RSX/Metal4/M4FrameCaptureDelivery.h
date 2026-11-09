#pragma once

#include "M4Frame.h"
#include <exception>
#include <utility>

namespace metal4
{
template <typename Action, typename Report>
bool try_capture_consumer(const char* operation, Action action, const Report& report)
{
	try { return action(); }
	catch (...)
	{
		report(operation, std::current_exception());
		return false;
	}
}

template <typename Mode, typename ReadMode, typename Consumer, typename Report>
bool can_deliver_frame_video(Mode recording, const ReadMode& read_mode, Consumer& consumer, const Report& report)
{
	return recording == read_mode() &&
		try_capture_consumer("video readiness", [&] { return consumer.can_consume_frame(); }, report) &&
		recording == read_mode();
}

template <typename Mode, typename ReadMode, typename Consumer, typename RetryScreenshot, typename Report>
void collect_frame_capture(std::optional<frame_capture_job>& capture, bool& capture_screenshot,
	std::optional<Mode>& capture_recording, ReadMode read_mode, Consumer& consumer, RetryScreenshot retry_screenshot,
	const Report& report)
{
	if (!capture) return;
	// Observe changes even before GPU completion. Keep the readback slot until
	// it retires so an abandoned video delivery cannot expand the GPU backlog.
	if (capture_recording && *capture_recording != read_mode())
		capture_recording.reset();
	if (!capture->ready()) return;

	auto job = std::exchange(capture, std::nullopt);
	const bool screenshot = std::exchange(capture_screenshot, false);
	const auto recording = std::exchange(capture_recording, std::nullopt);
	const bool video = recording && can_deliver_frame_video(*recording, read_mode, consumer, report);
	if (!screenshot && !video)
	{
		job->discard_if_ready();
		return;
	}

	std::optional<captured_frame> pixels;
	try
	{
		pixels = job->take_if_ready();
	}
	catch (...)
	{
		if (screenshot) retry_screenshot();
		throw;
	}
	const auto [width, height] = pixels->extent;
	if (screenshot && !try_capture_consumer("screenshot delivery", [&]
	{
		auto screenshot_pixels = video ? pixels->bgra : std::move(pixels->bgra);
		consumer.take_screenshot(std::move(screenshot_pixels), width, height, true);
		return true;
	}, report))
		retry_screenshot();
	// The CPU copy and screenshot callback can take time. Recheck the
	// recording mode and consumer readiness at the video handoff as well.
	if (video && can_deliver_frame_video(*recording, read_mode, consumer, report))
		(void)try_capture_consumer("video delivery", [&]
		{
			consumer.present_frame(std::move(pixels->bgra), width * 4, width, height, true);
			return true;
		}, report);
}
} // namespace metal4
