#include "M4Present.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

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
	throw std::runtime_error("Invalid presentation input accepted");
}
id<MTLTexture> texture(metal4::context& ctx, MTLPixelFormat format, std::size_t width, std::size_t height, bool mip = false)
{
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:mip];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	return metal4::make_texture(ctx, descriptor);
}
std::vector<std::byte> download(metal4::context& ctx, id<MTLTexture> image, std::size_t bytes)
{
	const auto output = metal4::make_buffer(ctx, image.width * image.height * bytes, metal4::buffer_storage::shared);
	ctx.begin();
	metal4::texture_to_buffer(ctx, image, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(image.width, image.height, 1)}, output, 0);
	ctx.submit().wait();
	const auto span = metal4::map_buffer(output, 0, output.length);
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
			context ctx;
			presenter engine(ctx);
			// No NSApplication, view, window, nextDrawable or actual presentation.
			CAMetalLayer* layer = [CAMetalLayer layer];
			engine.configure_layer(layer);
			require(layer.device == ctx.device() && layer.framebufferOnly && layer.pixelFormat == MTLPixelFormatBGRA8Unorm, "Layer metadata differs");
			rejects([&]
					{ engine.configure_layer(layer, MTLPixelFormatR8Unorm); });
			require(layer.pixelFormat == MTLPixelFormatBGRA8Unorm, "Rejected layer format mutated layer");
			rejects([&] { engine.configure_layer(layer, MTLPixelFormatRGBA16Float); });
			const auto source = texture(ctx, MTLPixelFormatRGBA8Unorm, 4, 4, true);
			std::array<std::uint8_t, 64> pixels{};
			for (std::size_t y = 0; y < 4; ++y)
				for (std::size_t x = 0; x < 4; ++x)
				{
					const auto i = (y * 4 + x) * 4;
					pixels[i] = 20 + x * 50;
					pixels[i + 1] = 10 + y * 60;
					pixels[i + 2] = 7;
					pixels[i + 3] = 255;
				}
			const auto input = make_upload(ctx, std::as_bytes(std::span{pixels}));
			ctx.begin();
			buffer_to_texture(ctx, input, 0, source, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)});
			ctx.submit().wait();
			const auto destination = texture(ctx, MTLPixelFormatBGRA8Unorm, 6, 6);
			present_image image{source, {0, 0, MTLOriginMake(1, 1, 0), MTLSizeMake(2, 2, 1)}, {1, 1, 4, 4}};
			for (unsigned flips = 0; flips < 4; ++flips)
			{
				image.flip_x = flips & 1;
				image.flip_y = flips & 2;
				ctx.begin();
				engine.encode(ctx, image, destination);
				ctx.submit().wait();
				const auto actual = download(ctx, destination, 4);
				for (std::size_t y = 0; y < 6; ++y)
					for (std::size_t x = 0; x < 6; ++x)
					{
						const auto i = (y * 6 + x) * 4;
						if (x == 0 || x == 5 || y == 0 || y == 5)
							require(actual[i] == std::byte{0} && actual[i + 1] == std::byte{0} && actual[i + 2] == std::byte{0} && actual[i + 3] == std::byte{255}, "Presentation letterbox differs");
						else
						{
							const auto sx = 1 + (image.flip_x ? 1 - (x - 1) / 2 : (x - 1) / 2);
							const auto sy = 1 + (image.flip_y ? 1 - (y - 1) / 2 : (y - 1) / 2);
							const auto s = (sy * 4 + sx) * 4;
							require(actual[i] == std::byte{7} && actual[i + 1] == std::byte{pixels[s + 1]} && actual[i + 2] == std::byte{pixels[s]} && actual[i + 3] == std::byte{255}, "Nearest crop/flip/BGRA differs");
						}
					}
			}
			std::cout << "Presentation nearest crop,all four flips,BGRA and opaque letterbox passed\n";
			image.flip_x = image.flip_y = false;
			image.filter = present_filter::linear;
			ctx.begin();
			engine.encode(ctx, image, destination);
			ctx.submit().wait();
			const auto linear = download(ctx, destination, 4);
			const auto close = [](std::byte value, double expected)
			{ return std::abs(double(std::to_integer<unsigned>(value)) - expected) <= 1; };
			for (std::size_t y = 1; y <= 4; ++y)
				for (std::size_t x = 1; x <= 4; ++x)
				{
					const auto sx = std::clamp((double(x) - 0.5) / 2, 0.5, 1.5);
					const auto sy = std::clamp((double(y) - 0.5) / 2, 0.5, 1.5);
					const auto i = (y * 6 + x) * 4;
					require(close(linear[i + 2], 70 + (sx - 0.5) * 50) && close(linear[i + 1], 70 + (sy - 0.5) * 60), "Linear crop edge bleed/interpolation differs");
				}
			// Explicit mip selection; level zero contains an unrelated pattern.
			const std::array<std::uint8_t, 16> mip{128, 64, 32, 255, 128, 64, 32, 255, 128, 64, 32, 255, 128, 64, 32, 255};
			const auto mip_input = make_upload(ctx, std::as_bytes(std::span{mip}));
			ctx.begin();
			buffer_to_texture(ctx, mip_input, 0, source, {1, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(2, 2, 1)});
			ctx.submit().wait();
			image.region = {1, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(2, 2, 1)};
			image.destination = {0, 0, 6, 6};
			const auto srgb = texture(ctx, MTLPixelFormatBGRA8Unorm_sRGB, 6, 6);
			ctx.begin();
			engine.encode(ctx, image, srgb);
			ctx.submit().wait();
			const auto encoded = download(ctx, srgb, 4);
			const auto srgb_value = [](double value)
			{ return 255 * (value <= 0.0031308 ? 12.92 * value : 1.055 * std::pow(value, 1 / 2.4) - 0.055); };
			for (std::size_t i = 0; i < encoded.size(); i += 4)
				require(close(encoded[i], srgb_value(32 / 255.0)) && close(encoded[i + 1], srgb_value(64 / 255.0)) && close(encoded[i + 2], srgb_value(128 / 255.0)), "Mip/sRGB encoding differs");
			const auto floating = texture(ctx, MTLPixelFormatRGBA16Float, 6, 6);
			ctx.begin();
			engine.encode(ctx, image, floating);
			ctx.submit().wait();
			const auto halves = download(ctx, floating, 8);
			for (std::size_t i = 0; i < halves.size(); i += 8)
			{
				std::uint16_t values[4];
				std::memcpy(values, halves.data() + i, 8);
				require(values[0] == 0x3804 && values[1] == 0x3404 && values[2] == 0x3004 && values[3] == 0x3c00, "Float display conversion differs");
			}
			std::cout << "Presentation linear crop clamps,explicit mip,sRGB and float display passed\n";
			const auto srgb_source = make_texture_view(source, MTLPixelFormatRGBA8Unorm_sRGB, MTLTextureType2D,
													   NSMakeRange(0, 3), NSMakeRange(0, 1), MTLTextureSwizzleChannelsDefault);
			image.texture = srgb_source;
			ctx.begin();
			engine.encode(ctx, image, srgb);
			ctx.submit().wait();
			const auto roundtrip = download(ctx, srgb, 4);
			for (std::size_t i = 0; i < roundtrip.size(); i += 4)
				require(close(roundtrip[i], 32) && close(roundtrip[i + 1], 64) && close(roundtrip[i + 2], 128), "sRGB decode/encode roundtrip differs");
			image.texture = source;
			const auto before = download(ctx, destination, 4);
			ctx.begin();
			for (unsigned bad = 0; bad < 7; ++bad)
			{
				auto invalid = image;
				if (bad == 0)
					invalid.texture = nil;
				if (bad == 1)
					invalid.region.origin.x = 2;
				if (bad == 2)
					invalid.destination.width = 0;
				if (bad == 3)
					invalid.destination.x = 6;
				if (bad == 4)
					invalid.filter = static_cast<present_filter>(99);
				if (bad == 5)
					invalid.region.slice = 1;
				if (bad == 6)
					invalid.texture = destination;
				rejects([&]
						{ engine.encode(ctx, invalid, destination); });
			}
			ctx.cancel();
			require(download(ctx, destination, 4) == before, "Rejected presentation modified output");
			std::cout << "Presentation invalid inputs rejected before output mutation;no window/app created\n";
			const auto gate = [ctx.device() newSharedEvent];
			require(bool(gate), "Presentation shared event missing");
			const auto pending = [&]
			{
				auto temporary = std::make_unique<presenter>(ctx);
				ctx.begin();
				ctx.wait_event(gate, 1);
				temporary->encode(ctx, image, destination);
				auto token = ctx.submit();
				temporary.reset();
				return token;
			}();
			bool timeout = false;
			try
			{
				pending.wait(std::chrono::milliseconds(2));
			}
			catch (const std::runtime_error&)
			{
				timeout = true;
			}
			require(timeout && !pending.ready(), "Presentation did not remain pending behind event");
			gate.signaledValue = 1;
			pending.wait();
			const auto retained = download(ctx, destination, 4);
			for (std::size_t i = 0; i < retained.size(); i += 4)
				require(retained[i] == std::byte{32} && retained[i + 1] == std::byte{64} && retained[i + 2] == std::byte{128}, "Presenter destruction lost pending GPU resources");
			std::cout << "Presentation sRGB decode/encode and delayed submission after presenter destruction passed\n";
			return 0;
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
