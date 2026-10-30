#include "M4Display.h"
#include "M4Frame.h"
#include "M4HostAccessFixture.h"

#import <AppKit/NSApplication.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

namespace
{
using namespace metal4;
using color = std::array<double, 4>;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void rejects(F&& fn)
{
	bool failed = false; try { fn(); } catch (const std::exception&) { failed = true; }
	require(failed, "Invalid stereo operation accepted");
}
enum class config_mode
{
	disabled = 91, side_by_side = 14, over_under = 33, interlaced = 7,
	anaglyph_red_green = 22, anaglyph_red_blue, anaglyph_red_cyan,
	anaglyph_magenta_cyan, anaglyph_trioscopic, anaglyph_amber_blue, anaglyph_custom
};
constexpr std::array modes{config_mode::disabled,config_mode::side_by_side,config_mode::over_under,config_mode::interlaced,
	config_mode::anaglyph_red_green,config_mode::anaglyph_red_blue,config_mode::anaglyph_red_cyan,
	config_mode::anaglyph_magenta_cyan,config_mode::anaglyph_trioscopic,config_mode::anaglyph_amber_blue,config_mode::anaglyph_custom};
struct configuration
{
	struct column { float rgb[3]{}; };
	struct matrix { std::array<column, 3> left, right; } values{};
	config_mode wanted = config_mode::disabled, actual = config_mode::disabled;
	void update_from_config(bool enabled) { actual = enabled ? wanted : config_mode::disabled; }
	config_mode stereo_mode() const { return actual; }
	const matrix& matrices() const { return values; }
};
stereo_output output(unsigned variant)
{
	configuration config; config.wanted = modes.at(variant);
	if (variant < 4) for (unsigned c = 0; c < 3; ++c) config.values.left[c].rgb[c] = 1;
	if (variant >= 4 && variant <= 6)
	{
		config.values.left[0].rgb[0] = 1;
		if (variant != 5) config.values.right[1].rgb[1] = 1;
		if (variant != 4) config.values.right[2].rgb[2] = 1;
	}
	if (variant == 7)
	{
		config.values.left[0].rgb[0] = config.values.right[1].rgb[1] = 1;
		config.values.left[2].rgb[2] = config.values.right[2].rgb[2] = .5;
	}
	if (variant == 8)
	{
		config.values.left[1].rgb[1] = 1;
		config.values.right[0].rgb[0] = config.values.right[2].rgb[2] = 1;
	}
	if (variant == 9)
	{
		config.values.left[0].rgb[0] = config.values.left[1].rgb[1] = 1;
		for (auto& col : config.values.right) col.rgb[2] = 1.f / 3;
	}
	if (variant == 10)
		for (unsigned col = 0; col < 3; ++col) for (unsigned row = 0; row < 3; ++row)
		{
			config.values.left[col].rgb[row] = float(int(col * 3 + row) - 3) * .3f;
			config.values.right[col].rgb[row] = float(5 - int(col * 2 + row)) * .2f;
		}
	const auto result = capture_stereo_configuration(config, true);
	require(result.mode == static_cast<stereo_mode>(variant < 4 ? variant : 4), "Stereo configuration mode was cast instead of mapped");
	config.values.left[0].rgb[0] = -99;
	require(result.left[0][0] != -99 && capture_stereo_configuration(config, false).mode == stereo_mode::disabled, "Stereo capture borrowed matrix storage or ignored AV disable");
	return result;
}
id<MTLTexture> texture(context& ctx, MTLPixelFormat format, unsigned w, unsigned h, bool mip = false)
{
	auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:mip];
	d.storageMode = MTLStorageModePrivate;
	d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	return make_texture(ctx, d);
}
std::vector<std::byte> download(context& ctx, id<MTLTexture> source)
{
	const auto bpp = source.pixelFormat == MTLPixelFormatRGBA16Float ? 8u : 4u;
	const auto out = make_buffer(ctx, source.width * source.height * bpp, buffer_storage::shared);
	ctx.begin(); texture_to_buffer(ctx, source, {0,0,MTLOriginMake(0,0,0),MTLSizeMake(source.width,source.height,1)},out,0); ctx.submit().wait();
	const auto bytes = map_buffer(out,0,out.length); return {bytes.begin(),bytes.end()};
}
color source_pixel(bool right, unsigned x, unsigned y)
{
	if (right) return {.75 - x / 8., .625 - y / 8., .875 - (x + y) / 16., .25};
	return {.125 + x / 8., .25 + y / 8., .125 + (x + y) / 16., .5};
}
color sample(bool right, double u, double v, bool linear)
{
	// Left selects mip0 crop(1,1,2,2);right selects mip1(0,0,2,2).
	const unsigned origin = right ? 0 : 1;
	const auto x = std::clamp(origin + u * 2, origin + .5, origin + 1.5) - .5;
	const auto y = std::clamp(origin + v * 2, origin + .5, origin + 1.5) - .5;
	if (!linear) return source_pixel(right, static_cast<unsigned>(std::floor(x + .5)), static_cast<unsigned>(std::floor(y + .5)));
	const auto x0 = static_cast<unsigned>(std::floor(x)), y0 = static_cast<unsigned>(std::floor(y));
	const auto fx = x - x0, fy = y - y0;
	const auto a = source_pixel(right,x0,y0), b = source_pixel(right,std::min(x0 + 1,origin + 1),y0);
	const auto c = source_pixel(right,x0,std::min(y0 + 1,origin + 1)), d = source_pixel(right,std::min(x0 + 1,origin + 1),std::min(y0 + 1,origin + 1));
	color result{};
	for (unsigned lane = 0; lane < 4; ++lane) result[lane] = (a[lane] * (1 - fx) + b[lane] * fx) * (1 - fy) + (c[lane] * (1 - fx) + d[lane] * fx) * fy;
	return result;
}
color expected_pixel(const present_image& image, unsigned x, unsigned y)
{
	const auto& rect = image.destination;
	if (x < rect.x || y < rect.y || x >= rect.x + rect.width || y >= rect.y + rect.height) return {0,0,0,1};
	double u = (x + .5 - rect.x) / rect.width, v = (y + .5 - rect.y) / rect.height;
	bool right = false;
	switch (image.stereo.mode)
	{
	case stereo_mode::side_by_side: right = u >= .5; u = u * 2 - right; break;
	case stereo_mode::over_under: right = v >= .5; v = v * 2 - right; break;
	case stereo_mode::interlaced: right = !(y & 1); break;
	default: break;
	}
	if (image.flip_x) u = 1 - u;
	if (image.flip_y) v = 1 - v;
	const auto linear = image.filter == present_filter::linear;
	auto result = sample(right,u,v,linear);
	if (image.stereo.mode == stereo_mode::anaglyph)
	{
		const auto other = sample(true,u,v,linear), left = result;
		for (unsigned row = 0; row < 3; ++row)
		{
			double sum = 0;
			for (unsigned col = 0; col < 3; ++col) sum += image.stereo.left[col][row] * left[col] + image.stereo.right[col][row] * other[col];
			result[row] = std::clamp(sum,0.,1.);
		}
	}
	if (image.stereo.mode != stereo_mode::disabled) result[3] = 1;
	for (unsigned c = 0; c < 3; ++c)
	{
		result[c] = std::pow(result[c],image.gamma);
		if (image.limited_rgb) result[c] = (result[c] * 220 + 16) / 255;
	}
	return result;
}
void verify(context& ctx, id<MTLTexture> target, const present_image& image)
{
	const auto data = download(ctx,target);
	for (unsigned y = 0; y < target.height; ++y) for (unsigned x = 0; x < target.width; ++x)
	{
		const auto expected = expected_pixel(image,x,y);
		for (unsigned c = 0; c < 4; ++c)
		{
			const auto pixel = y * target.width + x;
			if (target.pixelFormat == MTLPixelFormatRGBA16Float)
			{
				_Float16 actual; std::memcpy(&actual,data.data() + pixel * 8 + c * 2,2);
				require(std::abs(double(actual) - expected[c]) <= .0015, "Native stereo float16 pixel differs from independent oracle");
			}
			else
			{
				double value = expected[c];
				if (target.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB && c < 3)
					value = value <= .0031308 ? value * 12.92 : 1.055 * std::pow(value,1 / 2.4) - .055;
				const auto lane = c == 0 ? 2 : c == 2 ? 0 : c;
				const auto actual = std::to_integer<unsigned>(data[pixel * 4 + lane]);
				if (std::abs(int(actual) - int(std::lround(value * 255))) > 2)
					throw std::runtime_error("Stereo pixel:format=" + std::to_string(target.pixelFormat) + ",mode=" + std::to_string(static_cast<unsigned>(image.stereo.mode)) +
						",linear=" + std::to_string(image.filter == present_filter::linear) + ",flip=" + std::to_string(image.flip_x) + std::to_string(image.flip_y) +
						",gamma=" + std::to_string(image.gamma) + ",limited=" + std::to_string(image.limited_rgb) + ",x=" + std::to_string(x) + ",y=" + std::to_string(y) +
						",c=" + std::to_string(c) + ",actual=" + std::to_string(actual) + ",expected=" + std::to_string(std::lround(value * 255)));
			}
		}
	}
}
void pixel_matrix()
{
	context ctx; presenter present(ctx); native_frame frame(ctx);
	const auto left = texture(ctx,MTLPixelFormatRGBA32Float,4,4,true), right = texture(ctx,MTLPixelFormatRGBA32Float,4,4,true);
	std::array<float,64> l{}; std::array<float,16> r{};
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x)
		for (unsigned c = 0; c < 4; ++c) l[(y * 4 + x) * 4 + c] = source_pixel(false,x,y)[c];
	for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x)
		for (unsigned c = 0; c < 4; ++c) r[(y * 2 + x) * 4 + c] = source_pixel(true,x,y)[c];
	const auto lu = make_upload(ctx,std::as_bytes(std::span(l))), ru = make_upload(ctx,std::as_bytes(std::span(r)));
	ctx.begin(); buffer_to_texture(ctx,lu,0,left,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)});
	buffer_to_texture(ctx,ru,0,right,{1,0,MTLOriginMake(0,0,0),MTLSizeMake(2,2,1)}); ctx.submit().wait();
	present_image image{left,{0,0,MTLOriginMake(1,1,0),MTLSizeMake(2,2,1)},{1,1,8,8}};
	image.right_eye = present_eye{right,{1,0,MTLOriginMake(0,0,0),MTLSizeMake(2,2,1)}};
	unsigned cases = 0, frames = 0;
	for (const auto format : {MTLPixelFormatBGRA8Unorm,MTLPixelFormatBGRA8Unorm_sRGB,MTLPixelFormatRGBA16Float})
	{
		const auto target = texture(ctx,format,10,10);
		for (unsigned variant = 0; variant < modes.size(); ++variant)
			for (const bool linear : {false,true}) for (unsigned flips = 0; flips < 4; ++flips)
				for (const float gamma : {.8f,1.f,2.f}) for (const bool limited : {false,true})
				{
					image.stereo = output(variant); image.filter = linear ? present_filter::linear : present_filter::nearest;
					image.flip_x = flips & 1; image.flip_y = flips & 2; image.gamma = gamma; image.limited_rgb = limited;
					ctx.begin(); present.encode(ctx,image,target); ctx.submit().wait(); verify(ctx,target,image); ++cases;
				}
		for (unsigned variant = 0; variant < modes.size(); ++variant)
		{
			frame_policy policy{{4,3},4. / 3}; policy.stereo = output(variant); policy.gamma = .8f; policy.limited_rgb = true;
			image.stereo = {}; image.gamma = 1; image.limited_rgb = false;
			ctx.begin(); frame.encode(image,target,policy); ctx.submit().wait();
			auto expected = image; expected.stereo = policy.stereo; expected.gamma = policy.gamma; expected.limited_rgb = policy.limited_rgb; expected.filter = present_filter::linear;
			const auto rect = plan_frame_rectangle({2,2},{10,10},policy); expected.destination = {rect.x,rect.y,rect.width,rect.height};
			verify(ctx,target,expected); ++frames;
		}
		const auto before = download(ctx,target);
		for (unsigned bad = 0; bad < 7; ++bad)
		{
			auto invalid = image; invalid.stereo = output(1);
			if (bad == 0) invalid.right_eye.reset();
			if (bad == 1) invalid.right_eye->texture = nil;
			if (bad == 2) invalid.right_eye->region.origin.x = 99;
			if (bad == 3) invalid.right_eye->texture = target;
			if (bad == 4) invalid.stereo.left[0][1] = std::numeric_limits<float>::quiet_NaN();
			if (bad == 5) invalid.stereo.right[0][3] = 1;
			if (bad == 6) invalid.stereo.mode = static_cast<stereo_mode>(99);
			ctx.begin(); const auto caller = ctx.recording_submission(); const auto encoders = ctx.encoding_statistics().render_encoders;
			rejects([&] { present.encode(ctx,invalid,target); });
			require(caller.same_batch(ctx.recording_submission()) && ctx.encoding_statistics().render_encoders == encoders,"Rejected stereo preflight changed caller recording");
			ctx.cancel(); require(download(ctx,target) == before,"Rejected right eye/matrix changed target pixels");
		}
	}
	std::cout << "Native stereo:" << cases << " independent 10x10 pixel profiles," << frames << " native-frame AV/aspect/gamma/range profiles;all11config modes;nearest/linear;4flips;distinct eye crops/mips;absolute interlace parity;column matrices/clamp;BGRA/sRGB/RGBA16F;21preflight guards passed\n";
}
}
int main()
{
	@autoreleasepool
	{
		try
		{
			require(NSApp == nil,"Stereo test began with NSApplication");
			pixel_matrix();
			require(NSApp == nil,"Stereo test created NSApplication");
			std::cout << "Native stereo checks passed;no NSView/window/NSApplication/renderer/guest VM started\n";
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
