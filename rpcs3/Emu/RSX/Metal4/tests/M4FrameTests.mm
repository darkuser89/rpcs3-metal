#include "M4Frame.h"

#import <AppKit/NSApplication.h>

#include <algorithm>
#include <iostream>
#include <thread>

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
	const auto buffer = make_buffer(ctx, 256, buffer_storage::shared);
	ctx.begin();
	texture_to_buffer(ctx, image, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(8, 8, 1)}, buffer, 0);
	ctx.submit().wait();
	const auto span = map_buffer(buffer, 0, 256);
	return {span.begin(), span.end()};
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
