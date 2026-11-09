#include "stdafx.h"
#include "M4FrameCaptureDelivery.h"
#include "util/media_utils.h"
#include "util/video_provider.h"

#import <AppKit/NSApplication.h>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/sha.h>
#include <libswscale/swscale.h>
}

#include <filesystem>
#include <fstream>
#include <iostream>

extern atomic_t<bool> g_headless;
extern atomic_t<recording_mode> g_recording_mode;

namespace metal4
{
class frame_capture_test_access
{
public:
	static auto encode(native_frame& frame, const present_image& image, id<MTLTexture> reference)
	{
		const frame_extent extent{static_cast<u32>(reference.width), static_cast<u32>(reference.height)};
		const auto prepared = frame.prepare_image(image, reference, frame_policy{extent, double(extent.width) / extent.height});
		return frame.encode_capture(prepared, extent, reference.pixelFormat);
	}
};
}

namespace
{
struct headless_diagnostics
{
	headless_diagnostics() { g_headless = true; }
} diagnostics;

constexpr u32 width = 80, height = 48, frame_count = 3;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
template <typename Predicate> void wait_for(Predicate predicate, const char* message)
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!predicate())
	{
		require(std::chrono::steady_clock::now() < deadline, message);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}
struct temporary_directory
{
	std::filesystem::path path;
	temporary_directory()
	{
		char pattern[] = "/private/tmp/rpcs3-m4-recording-test.XXXXXX";
		const auto value = ::mkdtemp(pattern);
		require(value != nullptr, "Cannot create external capture scratch directory");
		path = value;
	}
	~temporary_directory()
	{
		std::error_code error;
		std::filesystem::remove_all(path, error);
	}
};
std::vector<u8> pattern(u32 index)
{
	constexpr std::array<std::array<u8, 4>, 4> colors{{{16, 48, 224, 255}, {200, 40, 20, 255},
		{40, 200, 64, 255}, {180, 180, 180, 255}}};
	std::vector<u8> result(width * height * 4);
	for (u32 y = 0; y < height; ++y)
		for (u32 x = 0; x < width; ++x)
		{
			const auto& color = colors[((y >= height / 2) * 2 + (x >= width / 2) + index) % colors.size()];
			std::copy(color.begin(), color.end(), result.begin() + (y * width + x) * 4);
		}
	return result;
}
struct recording_consumer
{
	utils::video_provider& provider;
	bool swap_channels = false;
	u32 frames = 0;
	bool can_consume_frame() { return provider.can_consume_frame(); }
	[[noreturn]] void take_screenshot(std::vector<u8>, u32, u32, bool)
	{
		throw std::runtime_error("Video-only test requested a screenshot");
	}
	void present_frame(std::vector<u8> data, u32 pitch, u32 w, u32 h, bool bgra)
	{
		require(w == width && h == height && pitch == width * 4 && bgra && data == pattern(frames),
			"Native capture changed packed BGRA data before the real consumer");
		provider.present_frame(std::move(data), pitch, w, h, swap_channels ? false : bgra);
		require(data.empty(), "Real recording consumer did not accept a ready frame");
		++frames;
	}
};
std::string sha256(const std::filesystem::path& file)
{
	std::unique_ptr<AVSHA, decltype(&av_free)> sha(av_sha_alloc(), &av_free);
	require(bool(sha) && av_sha_init(sha.get(), 256) == 0, "Cannot initialize file hash");
	std::ifstream input(file, std::ios::binary);
	std::array<char, 4096> bytes;
	while (input.read(bytes.data(), bytes.size()) || input.gcount())
		av_sha_update(sha.get(), reinterpret_cast<const u8*>(bytes.data()), input.gcount());
	require(input.eof(), "Cannot read encoded file for hash");
	std::array<u8, 32> digest;
	av_sha_final(sha.get(), digest.data());
	std::string hex;
	for (const auto byte : digest)
	{
		hex += "0123456789abcdef"[byte >> 4];
		hex += "0123456789abcdef"[byte & 15];
	}
	return hex;
}
void inspect_sample_times(const std::filesystem::path& file)
{
	std::ifstream input(file, std::ios::binary);
	const std::vector<u8> bytes{std::istreambuf_iterator<char>(input), {}};
	const auto word = [&](std::size_t offset)
	{
		require(offset <= bytes.size() && bytes.size() - offset >= 4, "Truncated MP4 sample-time table");
		return (u32(bytes[offset]) << 24) | (u32(bytes[offset + 1]) << 16) |
			(u32(bytes[offset + 2]) << 8) | bytes[offset + 3];
	};
	const auto boxes = [&](auto&& self, std::size_t begin, std::size_t end) -> void
	{
		for (auto at = begin; at < end;)
		{
			require(end - at >= 8, "Truncated MP4 box");
			const auto size = word(at);
			require(size >= 8 && size <= end - at, "Unexpected MP4 box size in small test file");
			const std::string_view type(reinterpret_cast<const char*>(bytes.data() + at + 4), 4);
			if (type == "moov" || type == "trak" || type == "mdia" || type == "minf" || type == "stbl")
				self(self, at + 8, at + size);
			else if (type == "stts")
			{
				require(size >= 16, "MP4 sample-time table has no entry count");
				const auto entries = word(at + 12);
				require(entries <= (size - 16) / 8, "MP4 sample-time table exceeds box");
				std::cout << "MP4 stts:";
				for (u32 i = 0; i < entries; ++i)
					std::cout << " samples=" << word(at + 16 + i * 8) << ",duration=" << word(at + 20 + i * 8);
				std::cout << '\n';
			}
			at += size;
		}
	};
	boxes(boxes, 0, bytes.size());
}
void decode_and_check(const std::filesystem::path& file, bool inspect_all_packets)
{
	inspect_sample_times(file);
	AVFormatContext* raw = nullptr;
	AVDictionary* options = nullptr;
	// Diagnostic comparison only. Normal playback must pass without disabling
	// the MP4 edit list; the default strict mode deliberately keeps that check.
	if (inspect_all_packets) av_dict_set(&options, "ignore_editlist", "1", 0);
	const int opened = avformat_open_input(&raw, file.c_str(), nullptr, &options);
	av_dict_free(&options);
	require(opened == 0, "Cannot reopen real consumer file");
	const auto close = [](AVFormatContext* value) { avformat_close_input(&value); };
	std::unique_ptr<AVFormatContext, decltype(close)> input(raw, close);
	require(avformat_find_stream_info(raw, nullptr) >= 0, "Cannot probe real consumer file");
	const int video = av_find_best_stream(raw, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
	require(video >= 0, "Real consumer file contains no video stream");
	const auto stream = raw->streams[video];
	require(stream->codecpar->width == width && stream->codecpar->height == height &&
		stream->codecpar->codec_id == AV_CODEC_ID_MPEG4, "Real consumer file changed extent or codec");
	const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
	std::unique_ptr<AVCodecContext, void(*)(AVCodecContext*)> codec(avcodec_alloc_context3(decoder),
		[](AVCodecContext* value) { avcodec_free_context(&value); });
	require(codec && avcodec_parameters_to_context(codec.get(), stream->codecpar) == 0 &&
		avcodec_open2(codec.get(), decoder, nullptr) == 0, "Cannot initialize real file decoder");
	std::unique_ptr<AVFrame, void(*)(AVFrame*)> frame(av_frame_alloc(), [](AVFrame* v) { av_frame_free(&v); });
	std::unique_ptr<AVPacket, void(*)(AVPacket*)> packet(av_packet_alloc(), [](AVPacket* v) { av_packet_free(&v); });
	require(frame && packet, "Cannot allocate decode buffers");
	u32 decoded = 0, max_error = 0, checked_channels = 0;
	s64 previous_pts = AV_NOPTS_VALUE;
	const auto receive = [&]
	{
		int result;
		while ((result = avcodec_receive_frame(codec.get(), frame.get())) == 0)
		{
			require(decoded < frame_count && frame->width == width && frame->height == height &&
				frame->best_effort_timestamp != AV_NOPTS_VALUE &&
				(previous_pts == AV_NOPTS_VALUE || frame->best_effort_timestamp > previous_pts),
				"Decoded video lost frame count, extent or increasing timestamps");
			previous_pts = frame->best_effort_timestamp;
			std::unique_ptr<SwsContext, decltype(&sws_freeContext)> scale(sws_getContext(width, height,
				static_cast<AVPixelFormat>(frame->format), width, height, AV_PIX_FMT_BGRA, SWS_BICUBIC, nullptr, nullptr, nullptr), &sws_freeContext);
			require(bool(scale), "Cannot convert decoded video to BGRA");
			std::vector<u8> bgra(width * height * 4);
			u8* output[]{bgra.data()};
			int pitch[]{width * 4};
			require(sws_scale(scale.get(), frame->data, frame->linesize, 0, height, output, pitch) == height,
				"Decoded color conversion lost rows");
			const auto expected = pattern(decoded);
			// MPEG4/YUV420 is lossy. Check flat interiors independently of
			// chroma subsampling/ringing at the four rectangle boundaries.
			for (u32 y = 8; y < height - 8; ++y)
				for (u32 x = 8; x < width - 8; ++x)
					if (std::abs(int(x) - int(width / 2)) >= 8 && std::abs(int(y) - int(height / 2)) >= 8)
						for (u32 c = 0; c < 3; ++c)
						{
							const auto offset = (y * width + x) * 4 + c;
							const auto error = u32(std::abs(int(bgra[offset]) - expected[offset]));
							max_error = std::max(max_error, error);
							++checked_channels;
							require(error <= 12, "Decoded BGRA colors differ from native capture beyond lossy tolerance");
						}
			++decoded;
			av_frame_unref(frame.get());
		}
		require(result == AVERROR(EAGAIN) || result == AVERROR_EOF, "Real file decoder failed");
	};
	int read;
	u32 video_packets = 0;
	while ((read = av_read_frame(raw, packet.get())) == 0)
	{
		if (packet->stream_index == video)
		{
			++video_packets;
			std::cout << "Video packet: index=" << video_packets << " pts=" << packet->pts << " dts=" << packet->dts
				<< " duration=" << packet->duration << " discard=" << bool(packet->flags & AV_PKT_FLAG_DISCARD) << '\n';
			require(avcodec_send_packet(codec.get(), packet.get()) == 0, "Real file video packet refused");
			receive();
		}
		av_packet_unref(packet.get());
	}
	require(read == AVERROR_EOF && avcodec_send_packet(codec.get(), nullptr) == 0, "Real file was truncated");
	receive();
	std::cout << "Real recording probe: format=" << raw->iformat->name << " decoded=" << decoded
		<< " expected=" << frame_count << " packets=" << video_packets << " stream_frames=" << stream->nb_frames
		<< " duration=" << stream->duration << " time_base=" << stream->time_base.num << '/' << stream->time_base.den
		<< " ignore_editlist=" << inspect_all_packets << '\n';
	std::cout << "Real recording file: format=" << raw->iformat->name << " codec=mpeg4 dimensions=" << width << 'x' << height
		<< " frames=" << decoded << " increasing_pts=1 checked_rgb_channels=" << checked_channels
		<< " maximum_lossy_error=" << max_error << " tolerance=12 bytes=" << std::filesystem::file_size(file)
		<< " sha256=" << sha256(file) << '\n';
	require(decoded == frame_count, "Real file lost captured video frames");
}
void test_recording(bool swap_channels, bool inspect_all_packets)
{
	using namespace metal4;
	temporary_directory scratch;
	const auto file = scratch.path / "capture.mp4";
	auto encoder = std::make_shared<utils::video_encoder>();
	encoder->use_internal_video = true;
	encoder->set_path(file.string());
	encoder->set_framerate(25);
	encoder->set_video_bitrate(4000000);
	encoder->set_output_format({AV_PIX_FMT_YUV420P, width, height, width});
	encoder->set_video_codec(AV_CODEC_ID_MPEG4, "mpeg4");
	encoder->set_audio_codec(AV_CODEC_ID_AAC, "aac");
	encoder->set_max_b_frames(0);
	encoder->set_gop_size(1);
	utils::video_provider provider;
	require(provider.set_video_sink(encoder, recording_mode::rpcs3), "Cannot bind real recording sink");
	g_recording_mode = recording_mode::rpcs3;
	encoder->encode();
	wait_for([&]
	{
		require(!encoder->has_error, "Real recording encoder failed before opening output");
		return std::filesystem::exists(file);
	}, "Real recording encoder did not open its file");
	context ctx;
	native_frame native(ctx);
	recording_consumer consumer{provider, swap_channels};
	const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
		width:width height:height mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead;
	const auto source = make_texture(ctx, descriptor);
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	const auto target = make_texture(ctx, descriptor);
	const auto read_mode = [] { return g_recording_mode.load(); };
	const auto report = [](const char*, std::exception_ptr error) { std::rethrow_exception(error); };
	for (u32 i = 0; i < frame_count; ++i)
	{
		wait_for([&] { return provider.can_consume_frame(); }, "Real recording consumer remained backpressured");
		const auto bytes = pattern(i);
		const auto upload = make_upload(ctx, std::as_bytes(std::span(bytes)));
		present_image image;
		image.texture = source;
		image.region = {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(width, height, 1)};
		image.destination = {0, 0, width, height};
		const auto gate = [ctx.device() newSharedEvent];
		ctx.begin();
		ctx.wait_event(gate, 1);
		buffer_to_texture(ctx, upload, 0, source, image.region);
		auto capture = frame_capture_test_access::encode(native, image, target);
		require(bool(capture), "Native capture admission failed");
		const auto work = ctx.submit();
		bool screenshot = false;
		std::optional mode{recording_mode::rpcs3};
		collect_frame_capture(capture, screenshot, mode, read_mode, consumer, [] {}, report);
		const bool blocked = consumer.frames == i && capture.has_value();
		gate.signaledValue = 1;
		work.wait();
		require(blocked, "Real recording consumer observed a blocked GPU capture");
		collect_frame_capture(capture, screenshot, mode, read_mode, consumer, [] {}, report);
		require(!capture && !mode && consumer.frames == i + 1, "Native capture was not delivered once");
	}
	require(!encoder->has_error, "Real recording encoder rejected native frames");
	encoder->stop(true);
	// last_video_pts is read only after joining the worker, never polled.
	require(encoder->last_video_pts() >= frame_count - 1, "Real recording encoder lost timestamps");
	g_recording_mode = recording_mode::stopped;
	require(provider.set_video_sink({}, recording_mode::rpcs3), "Cannot detach recording sink");
	decode_and_check(file, inspect_all_packets);
	std::cout << "Native capture to real video_provider/video_encoder: three gated GPU frames, "
		<< width * height * 4 * frame_count << " exact pre-encoder BGRA bytes; no screenshot/UI/session claim\n";
}
}

int main(int argc, char** argv)
{
	try
	{
		@autoreleasepool
		{
			require(NSApp == nil, "Capture test inherited a running application");
			bool swap_channels = false, inspect_all_packets = false;
			for (int i = 1; i < argc; ++i)
			{
				const std::string_view argument(argv[i]);
				if (argument == "--swap-video-channels") swap_channels = true;
				else if (argument == "--inspect-all-packets") inspect_all_packets = true;
				else throw std::runtime_error("Unknown capture test argument");
			}
			test_recording(swap_channels, inspect_all_packets);
			require(NSApp == nil, "Capture test created an application");
		}
		std::cout << "Selected real video file checks passed without app, GSRender, window, drawable or guest execution; "
			<< "packet-inspection mode does not approve normal MP4 playback\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
