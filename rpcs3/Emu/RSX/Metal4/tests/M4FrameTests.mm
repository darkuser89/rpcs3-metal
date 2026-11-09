#include "M4Frame.h"
#include "M4FrameCaptureDelivery.h"
#include "M4HeapBudget.h"

#import <AppKit/NSApplication.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numbers>
#include <thread>

namespace metal4
{
class frame_capture_test_access
{
  public:
	static auto encode(native_frame& frame, const std::optional<present_image>& image,
		id<MTLTexture> reference, const frame_policy& policy)
	{
		const auto prepared = frame.prepare_image(image, reference, policy);
		frame.encode_prepared(prepared, reference);
		return frame.encode_capture(prepared, {static_cast<std::uint32_t>(reference.width),
			static_cast<std::uint32_t>(reference.height)}, reference.pixelFormat);
	}
	static auto black(native_frame& frame, frame_extent size, MTLPixelFormat format)
	{
		return frame.encode_capture({}, size, format);
	}
	static auto compose(native_frame& frame, const std::optional<present_image>& image, id<MTLTexture> target,
		const frame_policy& policy, const overlay_frame& overlays)
	{
		const auto prepared = frame.prepare_image(image, target, policy);
		const auto composition = frame.prepare_overlays(overlays, prepared, target);
		frame.encode_prepared(prepared, target);
		frame.encode_overlays(composition, prepared, target);
		const frame_extent extent{static_cast<std::uint32_t>(target.width), static_cast<std::uint32_t>(target.height)};
		return std::array{frame.encode_capture(prepared, extent, target.pixelFormat, &composition),
			frame.encode_capture(prepared, extent, target.pixelFormat)};
	}
};
}

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void rejects(F&& fn)
{
	try
	{
		fn();
	}
	catch (const std::exception&)
	{
		return;
	}
	throw std::runtime_error("Invalid native frame accepted");
}
std::vector<std::byte> download(metal4::context& ctx, id<MTLTexture> image)
{
	using namespace metal4;
	const auto bytes = image.width * image.height * 4;
	const auto buffer = make_buffer(ctx, bytes, buffer_storage::shared);
	ctx.begin();
	texture_to_buffer(ctx, image, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(image.width, image.height, 1)}, buffer, 0);
	ctx.submit().wait();
	const auto span = map_buffer(buffer, 0, bytes);
	return {span.begin(), span.end()};
}

void check_capture(metal4::context& ctx, metal4::native_frame& frame, const metal4::present_image& image)
{
	using namespace metal4;
	struct capture_profile { frame_extent extent; MTLPixelFormat format; bool fsr; bool stereo; bool black; };
	const std::array profiles{
		capture_profile{{7, 5}, MTLPixelFormatBGRA8Unorm, false, false, false},
		capture_profile{{65, 3}, MTLPixelFormatBGRA8Unorm_sRGB, false, false, false},
		capture_profile{{9, 7}, MTLPixelFormatBGRA8Unorm, true, false, false},
		capture_profile{{13, 9}, MTLPixelFormatBGRA8Unorm, true, true, false},
		capture_profile{{11, 3}, MTLPixelFormatBGRA8Unorm, false, false, true}};
	std::size_t checked_bytes = 0;
	for (const auto& profile : profiles)
	{
		const auto [width, height] = profile.extent;
		const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:profile.format
			width:width height:height mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		const auto reference = make_texture(ctx, descriptor);
		const auto stride = (std::size_t(width) * 4 + 255) & ~std::size_t(255);
		const auto independent = make_buffer(ctx, stride * height, buffer_storage::shared);
		auto capture_image = image;
		capture_image.region.origin.x = 1;
		capture_image.region.size.width = 3;
		capture_image.flip_x = true;
		frame_policy policy{{4, 3}, 4.0 / 3};
		policy.gamma = 0.8f;
		policy.limited_rgb = true;
		policy.fsr = profile.fsr;
		if (profile.stereo)
		{
			capture_image.right_eye = present_eye{image.texture, image.region};
			policy.stereo.mode = stereo_mode::side_by_side;
		}
		const auto gate = [ctx.device() newSharedEvent];
		ctx.begin();
		ctx.wait_event(gate, 1);
		const auto before = ctx.encoding_statistics();
		auto capture = frame_capture_test_access::encode(frame,
			profile.black ? std::nullopt : std::optional{capture_image}, reference, policy);
		const auto after = ctx.encoding_statistics();
		require(capture && !capture->ready() && !capture->take_if_ready(), "Unsubmitted frame readback became visible");
		require(after.checked_dispatches - before.checked_dispatches == (profile.fsr ? (profile.stereo ? 4 : 2) : 0),
			"Frame capture repeated FSR work");
		texture_to_buffer(ctx, reference, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(width, height, 1)}, independent, 0, stride);
		const auto work = ctx.submit();
		const bool premature = capture->ready() || bool(capture->take_if_ready());
		gate.signaledValue = 1;
		work.wait();
		require(!premature, "Blocked frame readback became visible");
		auto pixels = capture->take_if_ready();
		require(pixels && pixels->extent == profile.extent && pixels->bgra.size() == std::size_t(width) * height * 4 &&
			!capture->ready() && !capture->take_if_ready(), "Completed frame readback lost shape or was delivered twice");
		const auto expected = map_buffer(independent, 0, stride * height);
		for (std::size_t row = 0; row < height; ++row)
			require(std::memcmp(pixels->bgra.data() + row * width * 4, expected.data() + row * stride, width * 4) == 0,
				"Frame capture changed a display pixel or included row padding");
		if (profile.black)
			for (std::size_t i = 0; i < pixels->bgra.size(); ++i)
				require(pixels->bgra[i] == (i % 4 == 3 ? 255 : 0), "Captured fallback was not opaque black");
		checked_bytes += pixels->bgra.size();
	}

	ctx.begin();
	const auto token = ctx.recording_submission();
	const auto encoders = ctx.encoding_statistics().render_encoders;
	require(!frame_capture_test_access::black(frame, {8192, 8192}, MTLPixelFormatBGRA8Unorm) &&
		!frame_capture_test_access::black(frame, {7, 5}, MTLPixelFormatRGBA8Unorm) &&
		ctx.encoding_statistics().render_encoders == encoders && token.same_batch(ctx.recording_submission()),
		"Refused media capture changed the caller recording");
	auto cancelled = frame_capture_test_access::black(frame, {7, 5}, MTLPixelFormatBGRA8Unorm);
	require(bool(cancelled), "Cancellation fixture failed capture admission");
	ctx.cancel();
	rejects([&] { (void)cancelled->take_if_ready(); });

	context limited;
	const auto budget = std::make_shared<heap_budget>(limited, 4096);
	limited.set_buffer_budget(budget);
	native_frame limited_frame(limited);
	limited.begin();
	const auto constrained = limited.recording_submission();
	require(!frame_capture_test_access::black(limited_frame, {65, 65}, MTLPixelFormatBGRA8Unorm) &&
		constrained.same_batch(limited.recording_submission()) && budget->statistics().reserved_bytes == 0,
		"Rejected capture allocation left native charges or replaced the recording");
	limited.cancel();
	std::cout << "Native media capture: five gated GPU profiles, " << checked_bytes
		<< " exact BGRA bytes, shared FSR, single delivery, cancellation and budget guards passed\n";
}

void check_capture_delivery(metal4::context& ctx, metal4::native_frame& frame)
{
	using namespace metal4;
	enum class mode { stopped, active };
	enum class scenario { stable, restart, video_only_restart, stop_in_screenshot, stop_in_readiness,
		backpressure, late_backpressure, readiness_error, late_readiness_error, screenshot_error, video_error };
	struct consumer
	{
		mode& recording;
		bool accepts_video = true;
		bool stop_after_screenshot = false;
		std::string_view failure;
		unsigned screenshots = 0, video_frames = 0;
		scenario test = scenario::stable;
		void fail_if(std::string_view operation) const
		{
			if (failure == operation) throw std::runtime_error("Injected capture consumer error");
		}
		bool can_consume_frame() const
		{
			fail_if("video readiness");
			if (test == scenario::stop_in_readiness) recording = mode::stopped;
			return accepts_video;
		}
		void check(const std::vector<std::uint8_t>& bytes, unsigned width, unsigned height, bool bgra) const
		{
			require(width == 7 && height == 5 && bgra && bytes.size() == 140, "Capture delivery lost BGRA extent");
			for (std::size_t i = 0; i < bytes.size(); ++i)
				require(bytes[i] == (i % 4 == 3 ? 255 : 0), "Capture delivery changed opaque black");
		}
		void take_screenshot(std::vector<std::uint8_t> bytes, unsigned width, unsigned height, bool bgra)
		{
			check(bytes, width, height, bgra);
			fail_if("screenshot delivery");
			++screenshots;
			if (stop_after_screenshot) recording = mode::stopped;
			if (test == scenario::late_backpressure) accepts_video = false;
			if (test == scenario::late_readiness_error) failure = "video readiness";
		}
		void present_frame(std::vector<std::uint8_t> bytes, unsigned pitch, unsigned width, unsigned height, bool bgra)
		{
			require(pitch == 28, "Capture delivery kept padded readback rows");
			check(bytes, width, height, bgra);
			fail_if("video delivery");
			++video_frames;
		}
	};
	for (const auto test : {scenario::stable, scenario::restart, scenario::video_only_restart,
		scenario::stop_in_screenshot, scenario::stop_in_readiness, scenario::backpressure,
		scenario::late_backpressure, scenario::readiness_error, scenario::late_readiness_error,
		scenario::screenshot_error, scenario::video_error})
	{
		auto recording = mode::active;
		const std::string_view failure = test == scenario::readiness_error ? "video readiness" :
			test == scenario::screenshot_error ? "screenshot delivery" : test == scenario::video_error ? "video delivery" : "";
		consumer receiver{recording, test != scenario::backpressure, test == scenario::stop_in_screenshot, failure};
		receiver.test = test;
		const auto expected_error = test == scenario::late_readiness_error ? "video readiness" : failure;
		bool screenshot = test != scenario::video_only_restart;
		const bool wants_screenshot = screenshot;
		std::optional requested_mode{recording};
		unsigned retries = 0, errors = 0;
		const auto gate = [ctx.device() newSharedEvent];
		struct open_gate { id<MTLSharedEvent> event; ~open_gate() { event.signaledValue = 1; } } open{gate};
		ctx.begin();
		ctx.wait_event(gate, 1);
		auto capture = frame_capture_test_access::black(frame, {7, 5}, MTLPixelFormatBGRA8Unorm);
		require(bool(capture), "Capture delivery fixture failed admission");
		const auto work = ctx.submit();
		const auto collect = [&]
		{
			collect_frame_capture(capture, screenshot, requested_mode, [&] { return recording; },
				receiver, [&] { ++retries; }, [&](std::string_view operation, std::exception_ptr error)
				{
					require(operation == expected_error, "Reported the wrong capture operation");
					try { std::rethrow_exception(error); }
					catch (const std::runtime_error& e)
					{
						require(std::string_view(e.what()) == "Injected capture consumer error", "Unexpected capture failure");
						++errors;
					}
				});
		};
		if (test == scenario::restart || test == scenario::video_only_restart)
		{
			recording = mode::stopped;
			collect();
			require(capture && !requested_mode && screenshot == wants_screenshot && !work.ready(),
				"Observed recording stop released the pending capture slot or kept stale video");
			recording = mode::active;
		}
		gate.signaledValue = 1;
		work.wait();
		collect();
		collect();
		require(!capture && !requested_mode && !screenshot && retries == unsigned(test == scenario::screenshot_error) &&
			errors == unsigned(!expected_error.empty()) && receiver.screenshots == unsigned(wants_screenshot && test != scenario::screenshot_error) &&
			receiver.video_frames == unsigned(test == scenario::stable || test == scenario::screenshot_error),
			"Capture routing lost screenshot delivery or accepted video after a mode or readiness change");
	}
	mode recording = mode::active;
	consumer receiver{recording};
	std::optional requested_mode{recording};
	bool screenshot = true;
	unsigned retries = 0, errors = 0;
	ctx.begin();
	auto cancelled = frame_capture_test_access::black(frame, {7, 5}, MTLPixelFormatBGRA8Unorm);
	require(bool(cancelled), "Cancelled delivery fixture failed admission");
	ctx.cancel();
	rejects([&]
	{
		collect_frame_capture(cancelled, screenshot, requested_mode, [&] { return recording; }, receiver,
			[&] { ++retries; }, [&](const char*, std::exception_ptr) { ++errors; });
	});
	require(!cancelled && !screenshot && !requested_mode && retries == 1 && !errors &&
		!receiver.screenshots && !receiver.video_frames, "Cancelled GPU capture was reported as a consumer failure");
	std::cout << "Native capture delivery: eleven gated GPU profiles including mode changes and late backpressure, independent consumer errors, exact BGRA, one screenshot retry and propagated GPU cancellation passed\n";

	for (const auto dropped : {scenario::video_only_restart, scenario::backpressure, scenario::readiness_error})
	{
		@autoreleasepool
		{
			recording = mode::active;
			requested_mode = recording;
			const bool blocked = dropped == scenario::backpressure;
			const bool throws = dropped == scenario::readiness_error;
			consumer sink{recording, !blocked, false, throws ? "video readiness" : ""};
			const auto gate = [ctx.device() newSharedEvent];
			struct open_gate { id<MTLSharedEvent> event; ~open_gate() { event.signaledValue = 1; } } open{gate};
			ctx.begin();
			ctx.wait_event(gate, 1);
			auto capture = frame_capture_test_access::black(frame, {7, 5}, MTLPixelFormatBGRA8Unorm);
			require(bool(capture), "Discarded capture fixture failed admission");
			const auto status = make_buffer(ctx, sizeof(std::uint32_t), buffer_storage::shared);
			std::memset(status.contents, 0, status.length);
			fill_buffer(ctx, status, 0, status.length, 1);
			ctx.require_gpu_success(status, 0, @"Discarded capture GPU marker");
			const auto work = ctx.submit();
			const auto collect = [&]
			{
				collect_frame_capture(capture, screenshot, requested_mode, [&] { return recording; }, sink,
					[&] { ++retries; }, [&](std::string_view operation, std::exception_ptr)
					{
						require(operation == "video readiness", "GPU failure was reported as a consumer failure");
						++errors;
					});
			};
			if (dropped == scenario::video_only_restart) recording = mode::stopped;
			collect();
			require(capture && !work.ready(), "Discard released a capture before GPU completion");
			gate.signaledValue = 1;
			rejects([&] { work.wait(); });
			const unsigned previous_errors = errors;
			bool gpu_failure = false;
			try { collect(); }
			catch (const std::runtime_error& e)
			{
				gpu_failure = std::string_view(e.what()).find("Discarded capture GPU marker") != std::string_view::npos;
			}
			collect();
			require(gpu_failure && !capture && !requested_mode && !screenshot && !sink.screenshots && !sink.video_frames &&
				retries == 1 && errors == previous_errors + unsigned(throws), "Discarded capture hid the GPU failure or delivered stale data");
			ctx.recover_completed_errors();
		}
	}
	std::cout << "Discarded capture retirement: three gated native GPU validation failures propagated after recording stop, backpressure and readiness exception\n";
}

void check_overlays(metal4::context& ctx, metal4::native_frame& frame, const metal4::present_image& image, id<MTLTexture> target)
{
	using namespace metal4;
	const frame_policy policy{{4, 3}, 4.0 / 3};
	bool captured_detached = false;
	require(!frame.present(image, policy, nullptr, [&](frame_extent) { captured_detached = true; return overlay_frame{}; }) &&
		!captured_detached, "Detached layer attempted to build an overlay snapshot");
	const auto quad = [](float x, float y, float w, float h, std::array<float, 4> color)
	{
		overlay_draw draw;
		draw.virtual_extent = {8, 8};
		draw.window_space = true;
		draw.color = color;
		draw.vertices = {{x, y, 0, 0}, {x+w, y, 1, 0}, {x, y+h, 0, 1}, {x+w, y+h, 1, 1}};
		return draw;
	};
	overlay_frame overlays;
	overlays.images.push_back({1, 1, 1, false, {100, 50, 200, 255}});
	overlays.images.push_back({2, 2, 2, true, {0, 0, 0, 0, 255, 255, 255, 255}});
	overlays.draws.push_back(quad(0, 0, 8, 1, {1, 0, 0, 1}));
	auto blended = quad(0, 0, 8, 6, {0, 1, 0, .5f});
	blended.virtual_extent = {8, 6};
	blended.window_space = false;
	blended.clip = std::array{1.f, 1.f, 4.f, 4.f};
	overlays.draws.push_back(std::move(blended));
	auto raw = quad(4, 0, 4, 2, {1, 1, 1, 1});
	raw.image = 0;
	raw.primitive = overlay_primitive::triangle_strip;
	overlays.draws.push_back(std::move(raw));
	for (unsigned page = 0; page < 2; ++page)
	{
		auto glyph = quad(float(page * 2), 5, 2, 2, {1, 1, 0, 1});
		glyph.image = 1;
		for (auto& vertex : glyph.vertices) vertex[3] = float(page) + .25f;
		overlays.draws.push_back(std::move(glyph));
	}
	auto fan = quad(6, 5, 2, 3, {0, 1, 1, 1});
	fan.primitive = overlay_primitive::triangle_fan;
	std::swap(fan.vertices[2], fan.vertices[3]);
	overlays.draws.push_back(std::move(fan));
	overlay_draw lines;
	lines.primitive = overlay_primitive::line_strip;
	lines.virtual_extent = {8, 8};
	lines.window_space = true;
	lines.snap = false;
	lines.color = {1, 1, 1, 1};
	lines.vertices = {{4.5f, 6.5f, 0, 0}, {4.5f, 7.5f, 0, 0}, {5.5f, 7.5f, 0, 0}};
	overlays.draws.push_back(lines);
	lines.primitive = overlay_primitive::lines;
	lines.vertices = {{5.5f, 5.5f, 0, 0}, {5.5f, 6.5f, 0, 0}};
	overlays.draws.push_back(std::move(lines));

	const auto gate = [ctx.device() newSharedEvent];
	ctx.begin();
	ctx.wait_event(gate, 1);
	auto captures = frame_capture_test_access::compose(frame, image, target, policy, overlays);
	overlays = {};
	const auto work = ctx.submit();
	const bool early = !captures[0] || !captures[1] || captures[0]->ready() || captures[1]->ready();
	gate.signaledValue = 1;
	work.wait();
	require(!early, "Overlay readback became ready before the gate");
	const auto composited = captures[0]->take_if_ready();
	const auto clean = captures[1]->take_if_ready();
	const auto expected_display = download(ctx, target);
	require(composited && clean && composited->bgra.size() == expected_display.size() &&
		std::memcmp(composited->bgra.data(), expected_display.data(), expected_display.size()) == 0,
		"Capture overlay did not match the display pixels");
	const auto pixel = [](const captured_frame& frame, unsigned x, unsigned y)
	{
		const auto* p = frame.bgra.data() + (y * 8 + x) * 4;
		return std::array<unsigned, 4>{p[0], p[1], p[2], p[3]};
	};
	require(pixel(*clean, 0, 0) == std::array{0u, 0u, 0u, 255u} &&
		pixel(*clean, 2, 2) == std::array{128u, 64u, 32u, 255u}, "Clean capture acquired overlay colors");
	require(pixel(*composited, 0, 0) == std::array{0u, 0u, 255u, 255u} &&
		pixel(*composited, 4, 0) == std::array{200u, 50u, 100u, 255u}, "Native overlay order or raw RGBA binding differs");
	require(pixel(*composited, 0, 2) == pixel(*clean, 0, 2) && pixel(*composited, 4, 2) == pixel(*clean, 4, 2),
		"Overlay clipping changed pixels outside the rectangle");
	const auto blend = pixel(*composited, 2, 2);
	require(blend[0] == 64 && (blend[1] == 159 || blend[1] == 160) && blend[2] == 16 && blend[3] == 255,
		"Native overlay blend or content viewport differs");
	require(pixel(*composited, 0, 5) == pixel(*clean, 0, 5) &&
		pixel(*composited, 2, 5) == std::array{0u, 255u, 255u, 255u} &&
		pixel(*composited, 7, 7) == std::array{255u, 255u, 0u, 255u}, "Glyph pages or native triangle fan differ");
	unsigned line_pixels = 0;
	for (unsigned y = 5; y < 8; ++y)
		for (unsigned x = 4; x < 6; ++x)
			line_pixels += pixel(*composited, x, y) == std::array{255u, 255u, 255u, 255u};
	require(line_pixels >= 2, "Native line primitives wrote no expected pixels");
	std::cout << "Native overlays: eight commands, two owned images, 256 exact capture bytes, window/content clip, blend, glyph pages and clean capture passed\n";

	for (const auto format : {MTLPixelFormatBGRA8Unorm_sRGB, MTLPixelFormatRGBA16Float})
	{
		const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:2 height:2 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		const auto image = make_texture(ctx, descriptor);
		const auto bytes = format == MTLPixelFormatRGBA16Float ? 8u : 4u;
		const auto readback = make_buffer(ctx, 4 * bytes, buffer_storage::shared);
		overlay_frame solid;
		solid.draws.push_back(quad(0, 0, 8, 8, {1, 0, 0, 1}));
		ctx.begin();
		frame.encode({}, image, policy, solid);
		texture_to_buffer(ctx, image, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(2, 2, 1)}, readback, 0);
		ctx.submit().wait();
		const auto expected = format == MTLPixelFormatRGBA16Float ?
			std::array<std::uint8_t, 8>{0, 0x3c, 0, 0, 0, 0, 0, 0x3c} :
			std::array<std::uint8_t, 8>{0, 0, 255, 255};
		const auto observed = map_buffer(readback, 0, 4 * bytes);
		for (unsigned pixel = 0; pixel < 4; ++pixel)
			require(std::memcmp(observed.data() + pixel * bytes, expected.data(), bytes) == 0,
				"Native overlay over a black frame used the wrong attachment format");
	}
	std::cout << "Native overlays: exact sRGB BGRA8 and RGBA16Float black-frame composition passed\n";
}

void check_overlay_effects(metal4::context& ctx, metal4::native_frame& frame, id<MTLTexture> target)
{
	using namespace metal4;
	const frame_policy policy{{8, 8}, 1};
	const auto quad = [](std::array<std::uint32_t, 2> extent, float x, float width)
	{
		overlay_draw draw;
		draw.virtual_extent = extent;
		draw.window_space = true;
		draw.vertices = {{x, 0, 0, 0}, {x + width, 0, 1, 0}, {x, float(extent[1]), 0, 1},
			{x + width, float(extent[1]), 1, 1}};
		return draw;
	};
	const auto pixel = [](auto& bytes, unsigned width, unsigned x, unsigned y)
	{
		std::array<std::uint8_t, 4> value;
		std::memcpy(value.data(), bytes.data() + (y * width + x) * 4, 4);
		return value;
	};
	const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:24 height:20 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	const auto shape_target = make_texture(ctx, descriptor);
	const auto render = [&](const overlay_frame& overlays, id<MTLTexture> image)
	{
		ctx.begin();
		frame.encode({}, image, policy, overlays);
		ctx.submit().wait();
		return download(ctx, image);
	};
	for (const auto shape : {overlay_shape::ellipse, overlay_shape::box, overlay_shape::rounded_box})
	{
		overlay_frame shapes;
		auto draw = quad({12, 20}, 0, 12);
		draw.color = {1, 0, 0, 1};
		draw.distance_field = overlay_distance_field{shape, {6, 10}, {5, 8}, 4, 2, {0, 1, 0, 1}};
		shapes.draws.push_back(draw);
		const auto result = render(shapes, shape_target);
		require(pixel(result, 24, 12, 10) == std::array<std::uint8_t, 4>{0, 0, 255, 255} &&
			pixel(result, 24, 0, 0) == std::array<std::uint8_t, 4>{0, 0, 0, 255}, "Overlay SDF fill or outside classification differs");
		const auto edge = pixel(result, 24, 22, 10);
		require(edge[0] == 0 && edge[1] >= 62 && edge[1] <= 66 && edge[2] == 0 && edge[3] == 255,
			"Overlay SDF outer-edge antialiasing differs");
		const auto corner = pixel(result, 24, 3, 3)[1];
		require((shape == overlay_shape::box ? corner > 180 : corner < 50), "Overlay SDF corner or radius scaling differs");
		shapes.draws.front().color[3] = 0;
		const auto transparent = render(shapes, shape_target);
		require(pixel(transparent, 24, 12, 10) == std::array<std::uint8_t, 4>{0, 0, 0, 255} &&
			pixel(transparent, 24, 22, 10) == edge, "Overlay SDF lost a border on a transparent fill");
	}
	{
		overlay_frame pulses;
		const float half_pi = std::numbers::pi_v<float> / 2;
		for (unsigned i = 0; i < 3; ++i)
		{
			auto draw = quad({8, 8}, float(i * 2), 2);
			draw.pulse_phase = (float(i) - 1.f) * half_pi;
			pulses.draws.push_back(draw);
		}
		pulses.images.push_back({1, 1, 1, true, {128}});
		auto glyph = quad({8, 8}, 6, 2);
		glyph.image = 0;
		pulses.draws.push_back(glyph);
		const auto result = render(pulses, target);
		for (unsigned x = 0; x < 8; ++x)
		{
			const auto gray = std::array<std::uint8_t, 4>{0, 128, 255, 64}[x / 2];
			require(pixel(result, 8, x, 3) == std::array<std::uint8_t, 4>{gray, gray, gray, 255},
				"Overlay pulse phase or partial glyph modulation differs");
		}
	}
	{
		overlay_frame images;
		images.images.push_back({8, 8, 1, false, {}});
		auto& input = images.images.front().bytes;
		for (unsigned y = 0; y < 8; ++y)
			for (unsigned x = 0; x < 8; ++x)
				input.insert(input.end(), {std::uint8_t(((x ^ y) & 1) * 255), std::uint8_t(x * 31), std::uint8_t(y * 31), 255});
		auto draw = quad({8, 8}, 0, 8);
		draw.image = 0;
		images.draws.push_back(draw);
		const auto read_texel = [&](int x, int y, unsigned channel)
		{
			return input[(std::clamp(y, 0, 7) * 8 + std::clamp(x, 0, 7)) * 4 + (2 - channel)];
		};
		for (float amount : {0.f, .5f, 1.f})
		{
			images.draws.front().blur = amount;
			const auto result = render(images, target);
			for (int y = 0; y < 8; ++y)
				for (int x = 0; x < 8; ++x)
				{
					const auto observed = pixel(result, 8, x, y);
					for (unsigned c = 0; c < 3; ++c)
					{
						int sum = 0;
						for (const auto offset : {std::array{-1, 0}, std::array{1, 0}, std::array{0, 1}})
							for (int dy = -1; dy <= 1; ++dy)
								for (int dx = -1; dx <= 1; ++dx)
									sum += read_texel(x + offset[0] + dx, y + offset[1] + dy, c) * (2 - std::abs(dx)) * (2 - std::abs(dy));
						const auto expected = std::lerp(float(read_texel(x, y, c)), float(sum) / 48.f, amount);
						require(std::abs(float(observed[c]) - expected) <= 1, "Overlay blur differs from the CPU texel oracle");
					}
					require(observed[3] == 255, "Overlay blur changed opaque alpha");
				}
		}
		const auto gate = [ctx.device() newSharedEvent];
		ctx.begin();
		ctx.wait_event(gate, 1);
		auto capture = frame_capture_test_access::compose(frame, {}, target, policy, images);
		const auto work = ctx.submit();
		const bool early = !capture[0] || capture[0]->ready();
		gate.signaledValue = 1;
		work.wait();
		require(!early, "Overlay effect capture ignored GPU completion");
		const auto pixels = capture[0]->take_if_ready();
		const auto displayed = download(ctx, target);
		require(pixels && pixels->bgra.size() == displayed.size() &&
			std::memcmp(pixels->bgra.data(), displayed.data(), displayed.size()) == 0,
			"Blurred overlay capture differs from displayed pixels");
	}
	std::cout << "Native overlay effects: three SDF shapes, transparent borders, three pulse phases, partial glyph, 576 blur channels and 256 capture bytes passed\n";
}

void check_overlay_image_cache()
{
	using namespace metal4;
	context ctx(3);
	overlay_renderer renderer(ctx);
	const auto image = [](std::uint8_t red)
	{
		overlay_image result{2, 2, 1, false, {}};
		for (unsigned i = 0; i < 4; ++i) result.bytes.insert(result.bytes.end(), {red, 12, 32, 255});
		return result;
	};
	const auto fixture = [](overlay_image pixels)
	{
		overlay_frame frame;
		frame.images.push_back(std::move(pixels));
		overlay_draw draw;
		draw.virtual_extent = {8, 8};
		draw.image = 0;
		draw.vertices = {{0, 0, 0, 0}, {8, 0, 1, 0}, {0, 8, 0, 1}, {8, 8, 1, 1}};
		frame.draws.push_back(std::move(draw));
		return frame;
	};
	const auto bytes_are = [&](id<MTLTexture> texture, const overlay_image& expected)
	{
		const auto pixels = download(ctx, texture);
		return pixels.size() == expected.bytes.size() && std::memcmp(pixels.data(), expected.bytes.data(), pixels.size()) == 0;
	};
	auto original = fixture(image(17));
	ctx.begin();
	const auto first = renderer.prepare(ctx, original).images.front();
	require(renderer.prepare(ctx, original).images.front() == first && renderer.image_statistics().uploads == 1,
		"Identical overlay image duplicated the upload in one recording");
	auto different_shape = original;
	different_shape.images.front().width = 1;
	different_shape.images.front().height = 4;
	const auto reshaped = renderer.prepare(ctx, different_shape).images.front();
	auto glyph = original;
	glyph.images.front().glyphs = true;
	glyph.images.front().layers = 4;
	const auto atlas = renderer.prepare(ctx, glyph).images.front();
	require(reshaped != first && atlas != first && atlas.textureType == MTLTextureType2DArray,
		"Exact payload aliases ignored image shape or glyph atlas type");
	ctx.submit().wait();
	require(bytes_are(first, original.images.front()), "Cold overlay upload differs");
	ctx.begin();
	const auto before = renderer.image_statistics();
	require(renderer.prepare(ctx, original).images.front() == first &&
		renderer.prepare(ctx, glyph).images.front() == atlas && renderer.image_statistics().uploads == before.uploads &&
		renderer.image_statistics().hits == before.hits + 2, "Completed image/atlas hit recorded a new upload");
	original.images.front() = image(211);
	const auto changed = renderer.prepare(ctx, original).images.front();
	ctx.submit().wait();
	require(changed != first && bytes_are(changed, original.images.front()) && bytes_are(first, image(17)),
		"Dirty overlay bytes replaced a retained texture or reused stale storage");

	const auto cancelled_frame = fixture(image(88));
	ctx.begin();
	const auto abandoned = renderer.prepare(ctx, cancelled_frame).images.front();
	ctx.cancel();
	ctx.begin();
	const auto fresh = renderer.prepare(ctx, cancelled_frame).images.front();
	ctx.submit().wait();
	require(fresh != abandoned && renderer.image_statistics().rejected_uploads == 1 && bytes_are(fresh, cancelled_frame.images.front()),
		"Cancelled overlay upload was reused or its retry lost pixels");

	overlay_renderer pending_renderer(ctx);
	const auto pending_frame = fixture(image(97));
	const auto gate = [ctx.device() newSharedEvent];
	ctx.begin();
	ctx.wait_event(gate, 1);
	const auto pending = pending_renderer.prepare(ctx, pending_frame).images.front();
	const auto blocked = ctx.submit();
	bool independent_upload = false;
	try
	{
		ctx.begin();
		const auto duplicate = pending_renderer.prepare(ctx, pending_frame).images.front();
		independent_upload = duplicate != pending && !blocked.ready() && !pending_renderer.reclaim_one_image();
		const auto later = ctx.submit();
		gate.signaledValue = 1;
		blocked.wait();
		later.wait();
		independent_upload &= bytes_are(duplicate, pending_frame.images.front());
	}
	catch (...)
	{
		gate.signaledValue = 1;
		throw;
	}
	require(independent_upload && bytes_are(pending, pending_frame.images.front()), "Pending overlay upload leaked into an independent recording");

	overlay_renderer lru(ctx);
	ctx.begin();
	const auto oldest = lru.prepare(ctx, fixture(image(0))).images.front();
	const auto next = lru.prepare(ctx, fixture(image(1))).images.front();
	for (unsigned i = 2; i < 32; ++i) (void)lru.prepare(ctx, fixture(image(static_cast<std::uint8_t>(i))));
	require(lru.prepare(ctx, fixture(image(0))).images.front() == oldest, "Overlay cache did not promote a hit");
	(void)lru.prepare(ctx, fixture(image(32)));
	const auto eviction = lru.image_statistics();
	require(eviction.entries == 32 && eviction.evicted == 1 && eviction.host_bytes + eviction.native_bytes <= (32u << 20) &&
		lru.prepare(ctx, fixture(image(0))).images.front() == oldest && lru.prepare(ctx, fixture(image(1))).images.front() != next,
		"Overlay cache did not enforce its limits in LRU order");
	ctx.submit().wait();
	require(bytes_are(next, image(1)), "Evicted overlay texture lost its recorded contents");
	const auto large = fixture({2048, 2049, 1, false, std::vector<std::uint8_t>(2048u * 2049u * 4)});
	ctx.begin();
	const auto before_large = lru.image_statistics();
	const auto large_texture = lru.prepare(ctx, large).images.front();
	ctx.submit().wait();
	const auto after_large = lru.image_statistics();
	require(large_texture && after_large.uploads == before_large.uploads + 1 && after_large.entries == before_large.entries &&
		after_large.host_bytes == before_large.host_bytes && after_large.native_bytes == before_large.native_bytes,
		"Oversized overlay image displaced the bounded reusable cache");
	std::cout << "Native overlay cache: image/atlas warm hits, byte/descriptor changes, cancellation, independent gated uploads, 32-entry LRU and 32 MiB cap passed\n";
}

void check_overlay_special_sources(metal4::context& ctx, metal4::native_frame& frame, id<MTLTexture> target)
{
	using namespace metal4;
	const std::array<std::uint8_t, 79> png{137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
		0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0, 114, 182, 13, 36, 0, 0, 0, 22, 73, 68, 65, 84, 120, 156,
		99, 248, 207, 192, 240, 31, 8, 27, 24, 128, 52, 144, 205, 192, 0, 0, 58, 220, 5, 124, 127, 215, 35, 28,
		0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130};
	const std::array<std::uint8_t, 83> wide_png{137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
		0, 0, 4, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 175, 204, 183, 11, 0, 0, 0, 26, 73, 68, 65, 84, 120, 156,
		237, 193, 1, 1, 0, 0, 0, 130, 32, 255, 175, 110, 72, 64, 1, 0, 0, 0, 23, 6, 16, 5, 0, 1, 222, 4, 43, 145,
		0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130};
	const auto icon = decode_overlay_game_icon(png);
	require(icon && *icon == overlay_image{2, 2, 1, false, {255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 255, 0, 0, 0, 0}},
		"Game icon decode changed RGBA order, row order or straight alpha");
	auto not_png = png;
	not_png[0] = 0;
	require(!decode_overlay_game_icon(not_png), "Game icon decoder admitted a wrong signature");
	require(!decode_overlay_game_icon(std::span(png).first(44)), "Game icon decoder admitted a truncated file");
	require(!decode_overlay_game_icon(wide_png), "Game icon decoder admitted an oversized dimension");
	require(!decode_overlay_game_icon(std::vector<std::uint8_t>(max_overlay_icon_file_bytes + 1)), "Game icon decoder admitted an oversized file");
	const auto quad = [](std::uint32_t width, float x, float right, float u_max)
	{
		overlay_draw draw;
		draw.virtual_extent = {width, width};
		draw.window_space = true;
		draw.vertices = {{x, 0, 0, 0}, {right, 0, u_max, 0}, {x, float(width), 0, 1}, {right, float(width), u_max, 1}};
		return draw;
	};
	{
		const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:2 height:2 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		const auto icon_target = make_texture(ctx, descriptor);
		overlay_frame overlays;
		overlays.images.push_back(*icon);
		auto draw = quad(2, 0, 2, 1);
		draw.image = 0;
		overlays.draws.push_back(draw);
		ctx.begin();
		frame.encode({}, icon_target, frame_policy{{2, 2}, 1}, overlays);
		ctx.submit().wait();
		const auto output = download(ctx, icon_target);
		const std::array<std::uint8_t, 16> expected{0, 0, 255, 255, 0, 128, 0, 255, 255, 0, 0, 255, 0, 0, 0, 255};
		require(std::memcmp(output.data(), expected.data(), expected.size()) == 0, "Game icon overlay alpha or placement differs");
	}
	{
		const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead;
		const auto source = make_texture(ctx, descriptor);
		std::array<std::array<std::uint8_t, 4>, 64> pixels;
		for (unsigned i = 0; i < pixels.size(); ++i)
			pixels[i] = i % 8 < 4 ? std::array<std::uint8_t, 4>{255, 0, 0, 255} : std::array<std::uint8_t, 4>{0, 0, 255, 255};
		const auto input = make_upload(ctx, std::as_bytes(std::span(pixels)));
		const texture_region region{0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(8, 8, 1)};
		ctx.begin();
		buffer_to_texture(ctx, input, 0, source, region);
		ctx.submit().wait();
		overlay_frame overlays;
		auto solid = quad(8, 0, 8, 1);
		solid.color = {0, 1, 0, 1};
		overlays.draws.push_back(solid);
		auto backbuffer = quad(8, 4, 8, .5f);
		backbuffer.backbuffer = true;
		overlays.draws.push_back(backbuffer);
		const auto gate = [ctx.device() newSharedEvent];
		ctx.begin();
		ctx.wait_event(gate, 1);
		auto jobs = frame_capture_test_access::compose(frame, present_image{source, region}, target, frame_policy{{8, 8}, 1}, overlays);
		const auto work = ctx.submit();
		const bool early = !jobs[0] || !jobs[1] || jobs[0]->ready() || jobs[1]->ready();
		gate.signaledValue = 1;
		work.wait();
		require(!early, "Backbuffer capture was published before GPU completion");
		const auto composed = jobs[0]->take_if_ready();
		const auto clean = jobs[1]->take_if_ready();
		const auto display = download(ctx, target);
		require(composed && clean && display.size() == composed->bgra.size() &&
			std::memcmp(display.data(), composed->bgra.data(), display.size()) == 0, "Backbuffer capture did not match the display");
		for (unsigned i = 0; i < pixels.size(); ++i)
		{
			const auto expected = i % 8 < 4 ? std::array<std::uint8_t, 4>{0, 255, 0, 255} : std::array<std::uint8_t, 4>{0, 0, 255, 255};
			const std::array<std::uint8_t, 4> base{pixels[i][2], pixels[i][1], pixels[i][0], pixels[i][3]};
			require(std::memcmp(composed->bgra.data() + i * 4, expected.data(), 4) == 0 &&
				std::memcmp(clean->bgra.data() + i * 4, base.data(), 4) == 0,
				"Backbuffer did not sample the clean frame or changed clean capture");
		}
	}
	{
		context limited;
		const auto budget = std::make_shared<heap_budget>(limited, 2u << 20);
		limited.set_buffer_budget(budget);
		native_frame output(limited);
		const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:1024 height:1024 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		const auto oversized = make_texture(limited, descriptor);
		overlay_frame overlays;
		auto green = quad(8, 0, 8, 1);
		green.color = {0, 1, 0, 1};
		overlays.draws.push_back(green);
		auto backbuffer = quad(8, 0, 8, 1);
		backbuffer.backbuffer = true;
		overlays.draws.push_back(backbuffer);
		const auto pixel = make_buffer(limited, 4, buffer_storage::shared);
		limited.begin();
		output.encode({}, oversized, frame_policy{{8, 8}, 1}, overlays);
		texture_to_buffer(limited, oversized, {0, 0, MTLOriginMake(32, 32, 0), MTLSizeMake(1, 1, 1)}, pixel, 0);
		limited.submit().wait();
		const std::array<std::uint8_t, 4> expected{0, 255, 0, 255};
		require(std::memcmp(pixel.contents, expected.data(), 4) == 0 && budget->statistics().live_heaps == 0,
			"Backbuffer budget refusal discarded unrelated overlay work");
	}
	std::cout << "Native overlay sources: exact PNG RGBA and alpha, four PNG guards, clean backbuffer snapshot, 256 matching capture bytes and budget fallback passed\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			using namespace metal4;
			require(NSApp == nil, "Test began with an application instance");
			context ctx;
			native_frame frame(ctx);
			frame.update_geometry(640.25, 480.5, 1.5, true);
			require(frame.extent() == frame_extent{961, 721} && frame.layer().drawableSize.width == 961 && frame.layer().contentsScale == 1.5, "Layer Retina geometry differs");
			require(frame.layer().maximumDrawableCount == 3 && frame.layer().allowsNextDrawableTimeout && !frame.layer().presentsWithTransaction && frame.layer().colorspace, "Native frame layer policy differs");
			rejects([&]
					{ frame.update_geometry(-1, 1, 1, true); });
			require(frame.extent() == frame_extent{961, 721}, "Rejected resize published geometry");
			bool thread_rejected = false;
			std::thread worker([&]
							   { try { frame.update_geometry(1, 1, 1, true); } catch (const std::logic_error&) { thread_rejected = true; } });
			worker.join();
			require(thread_rejected && frame.extent() == frame_extent{961, 721}, "Wrong-thread resize was accepted");
			rejects([&]
					{ frame.attach(nil); });
			frame_policy policy{{4, 3}, 4.0 / 3};
			require(!frame.present({}, policy), "Unattached frame attempted drawable acquisition");
			policy.pacing = frame_pacing::adaptive;
			require(!frame.present({}, policy), "Unattached adaptive frame attempted drawable acquisition");
			std::cout << "Detached layer Retina metadata,invalid/wrong-thread resize and no-acquire unattached frame passed\n";
			auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:3 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead;
			const auto source = make_texture(ctx, descriptor);
			std::array<std::byte, 48> source_bytes;
			for (std::size_t i = 0; i < source_bytes.size(); ++i)
				source_bytes[i] = std::byte(std::array{32, 64, 128, 255}[i % 4]);
			const auto input = make_upload(ctx, source_bytes);
			ctx.begin();
			buffer_to_texture(ctx, input, 0, source, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 3, 1)});
			ctx.submit().wait();
			descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:8 height:8 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageRenderTarget;
			const auto target = make_texture(ctx, descriptor);
			const present_image image{source, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 3, 1)}};
			std::size_t cases = 0;
			for (const float gamma : {1.f, .8f, 1.2f, 2.f})
				for (const bool limited : {false, true})
					for (const bool stretch : {false, true})
					{
						policy.gamma = gamma;
						policy.limited_rgb = limited;
						policy.stretch = stretch;
						ctx.begin();
						frame.encode(image, target, policy);
						ctx.submit().wait();
						const auto result = download(ctx, target);
						for (std::size_t y = 0; y < 8; ++y)
							for (std::size_t x = 0; x < 8; ++x)
								for (std::size_t channel = 0; channel < 4; ++channel)
								{
									const bool border = !stretch && (y == 0 || y == 7);
									double expected = channel == 3 ? 255 : border ? 0
																				  : std::pow(std::array{128, 64, 32}[channel] / 255.0, gamma) * 255;
									if (!border && channel != 3 && limited)
										expected = (expected * 220 / 255) + 16;
									const auto actual = std::to_integer<unsigned>(result[(y * 8 + x) * 4 + channel]);
									require(std::abs(double(actual) - expected) <= 1, "Native frame aspect/gamma/range pixel differs");
								}
						++cases;
					}
			ctx.begin();
			frame.encode({}, target, policy);
			ctx.submit().wait();
			const auto black = download(ctx, target);
			for (std::size_t i = 0; i < black.size(); ++i)
				require(black[i] == std::byte(i % 4 == 3 ? 255 : 0), "Missing display didn't clear opaque black");
			ctx.begin();
			auto invalid = image;
			invalid.region.level = 99;
			rejects([&]
					{ frame.encode(invalid, target, policy); });
			ctx.cancel();
			require(download(ctx, target) == black, "Rejected frame encoding mutated target");
			check_capture(ctx, frame, image);
			check_capture_delivery(ctx, frame);
			check_overlays(ctx, frame, image, target);
			check_overlay_effects(ctx, frame, target);
			check_overlay_image_cache();
			check_overlay_special_sources(ctx, frame, target);
			frame.drain();
			frame.detach();
			require(NSApp == nil, "Frame test created an application instance");
			std::cout << cases << " native frame aspect/stretch/gamma/limited-range pixel oracles,black fallback and invalid-source preservation passed;no NSView/window/NSApplication created\n";
			return 0;
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
