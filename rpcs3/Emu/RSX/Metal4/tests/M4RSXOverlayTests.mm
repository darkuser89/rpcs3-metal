#include "stdafx.h"
#include "Emu/emu_callbacks.h"
#include "M4RSXOverlay.h"
#include "M4Frame.h"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
using namespace metal4;
namespace fsys = std::filesystem;
using pixel = std::array<std::uint8_t, 4>;
constexpr pixel red{0, 0, 255, 255}, blue{255, 0, 0, 255}, black{0, 0, 0, 255};
constexpr std::array<std::uint8_t, 70> red_png{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
	0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0,
	0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99, 0x3d, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,
	0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
constexpr auto blue_png = []
{
	auto result = red_png;
	constexpr std::array<std::uint8_t, 13> data{
		0x60, 0x60, 0xf8, 0xff, 0x1f, 0x00, 0x03, 0x02, 0x01, 0xff, 0xe6, 0x77, 0x0b};
	std::copy(data.begin(), data.end(), result.begin() + 44);
	result[57] = 0xae;
	return result;
}();

void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}

struct temporary_title
{
	fsys::path root;
	temporary_title()
	{
		char name[] = "/private/tmp/rpcs3-metal4-overlay-icon.XXXXXX";
		const auto* result = ::mkdtemp(name);
		require(result != nullptr, "Could not create Metal4 overlay title fixture");
		root = result;
	}
	~temporary_title() { std::error_code error; fsys::remove_all(root, error); }
};

struct icon_overlay final : rsx::overlays::overlay
{
	u16 get_virtual_width() const override { return 1; }
	u16 get_virtual_height() const override { return 1; }
	rsx::overlays::compiled_resource get_compiled() override
	{
		rsx::overlays::compiled_resource result;
		auto& command = result.draw_commands.emplace_back();
		command.config.texture_ref = rsx::overlays::image_resource_id::game_icon;
		command.verts = {{0, 0, 0, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}, {1, 1, 1, 1}};
		return result;
	}
};

struct glyph_overlay final : rsx::overlays::overlay
{
	rsx::overlays::font* font = nullptr;
	std::array<float, 2> uv{};
	areaf clip{3, 3, 5, 5};
	u16 get_virtual_width() const override { return 8; }
	u16 get_virtual_height() const override { return 8; }
	rsx::overlays::compiled_resource get_compiled() override
	{
		rsx::overlays::compiled_resource result;
		auto& command = result.draw_commands.emplace_back();
		command.config.set_font(font);
		command.config.clip_region = true;
		command.config.clip_rect = clip;
		command.verts = {vertex{0.f, 0.f, uv[0], uv[1]}, vertex{8.f, 0.f, uv[0], uv[1]},
			vertex{0.f, 8.f, uv[0], uv[1]}, vertex{8.f, 8.f, uv[0], uv[1]}};
		return result;
	}
};

void test_glyph_clip()
{
	constexpr auto font_path = "/System/Library/Fonts/Supplemental/Arial.ttf";
	require(fsys::is_regular_file(font_path), "Production glyph fixture requires macOS Arial.ttf");
	struct font_directories
	{
		decltype(g_emu_callbacks.get_font_dirs) previous = std::exchange(g_emu_callbacks.get_font_dirs,
			[] { return std::vector<std::string>{"/System/Library/Fonts/Supplemental/"}; });
		~font_directories() { g_emu_callbacks.get_font_dirs = std::move(previous); }
	} directories;
	rsx::overlays::font font(font_path, 18);
	rsx::overlays::display_manager overlays(0);
	auto view = overlays.create<glyph_overlay>();
	view->visible = true;
	view->use_window_space = true;
	view->font = &font;
	std::optional<rsx::overlays::resource_config> resources;
	context device(1);
	native_frame frame(device);
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget;
	const auto target = make_texture(device, descriptor);
	const auto readback = make_buffer(device, 4 * 256, buffer_storage::shared);

	unsigned page = 0;
	for (const char32_t character : {U'A', U'Ω'})
	{
		float x_advance = 0, y_advance = 0;
		const auto quad = font.get_char(character, x_advance, y_advance);
		const auto dimensions = font.get_glyph_data_dimensions();
		const auto& atlas = font.get_glyph_data();
		require(dimensions.depth == page + 1 && std::floor(quad.t0) == float(page) &&
			std::floor(quad.t1) == float(page), "Production glyph did not create a separate expected atlas page");
		const auto low_x = unsigned(quad.s0 * dimensions.width), high_x = unsigned(quad.s1 * dimensions.width);
		const auto low_y = unsigned((quad.t0 - float(page)) * dimensions.height);
		const auto high_y = unsigned((quad.t1 - float(page)) * dimensions.height);
		std::uint8_t coverage = 0;
		for (auto y = low_y; y < high_y; ++y)
			for (auto x = low_x; x < high_x; ++x)
			{
				const auto sample = atlas[(std::size_t(page) * dimensions.height + y) * dimensions.width + x];
				if (sample > coverage && sample < 240)
				{
					coverage = sample;
					view->uv = {(float(x) + .5f) / dimensions.width, float(page) + (float(y) + .5f) / dimensions.height};
				}
			}
		require(coverage >= 128, "Production glyph has no usable antialiased texel");
		const auto gray = static_cast<std::uint8_t>(std::lround(float(coverage) * coverage / 255.f));
		for (const auto reverse : {false, true})
		{
			view->clip = reverse ? areaf{5, 5, 3, 3} : areaf{3, 3, 5, 5};
			const auto commands = capture_rsx_overlays(overlays, resources, []() -> const overlay_image* { return nullptr; }, {4, 4}, 0);
			require(commands.draws.size() == 1 && commands.images.size() == 1 && commands.images[0].glyphs &&
				commands.images[0].layers == dimensions.depth && commands.images[0].bytes == atlas,
				"Native overlay adapter did not retain the production font atlas");
			device.begin();
			frame.encode({}, target, frame_policy{{4, 4}, 1}, commands);
			texture_to_buffer(device, target, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, readback, 0, 256);
			device.submit().wait();
			const auto* pixels = static_cast<const std::uint8_t*>(readback.contents);
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 4; ++x)
				{
					const pixel expected = x >= 1 && x <= 2 && y >= 1 && y <= 2 ? pixel{gray, gray, gray, 255} : black;
					require(std::memcmp(pixels + y * 256 + x * 4, expected.data(), expected.size()) == 0,
						"Production glyph coverage or inclusive clip edge differs");
				}
		}
		++page;
	}
	std::cout << "Metal4 production glyphs: 2 Arial atlas pages, normal and reversed inclusive clip, 256 exact BGRA bytes\n";
}

void write(const fsys::path& path, std::span<const std::uint8_t> bytes)
{
	std::ofstream stream(path, std::ios::binary | std::ios::trunc);
	stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	stream.close();
	require(bool(stream), "Could not write Metal4 overlay PNG fixture");
}
} // namespace

void test_rsx_overlay_icon()
{
	@autoreleasepool
	{
		temporary_title files;
		const auto primary = files.root / "primary", fallback = files.root / "fallback";
		fsys::create_directory(primary);
		fsys::create_directory(fallback);
		const auto icon = primary / "ICON0.PNG", other = fallback / "ICON0.PNG";
		rsx::overlays::display_manager overlays(0);
		auto view = overlays.create<icon_overlay>();
		view->visible = true;
		view->use_window_space = true;
		rsx_game_icon_source icons;
		std::optional<rsx::overlays::resource_config> resources;
		context device(1);
		native_frame frame(device);
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:1 height:1 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		const auto target = make_texture(device, descriptor);
		const auto readback = make_buffer(device, 4, buffer_storage::shared);
		const auto render = [&](pixel expected, const char* message)
		{
			const auto commands = capture_rsx_overlays(overlays, resources, [&]
			{
				return icons.get(primary.string(), fallback.string());
			}, {1, 1}, 0);
			const std::size_t count = expected == black ? 0 : 1;
			require(commands.images.size() == count && commands.draws.size() == count && !resources, message);
			device.begin();
			frame.encode({}, target, frame_policy{{1, 1}, 1}, commands);
			texture_to_buffer(device, target, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(1, 1, 1)}, readback, 0);
			device.submit().wait();
			require(std::memcmp(readback.contents, expected.data(), expected.size()) == 0, message);
		};
		render(black, "Missing title files produced an icon");
		write(other, blue_png);
		render(blue, "Late fallback icon stayed absent");
		write(icon, red_png);
		render(red, "Late preferred icon did not replace the fallback");
		render(red, "Warm preferred icon changed");
		const auto replacement = primary / "replacement";
		write(replacement, blue_png);
		fsys::last_write_time(replacement, fsys::last_write_time(icon));
		fsys::rename(replacement, icon);
		render(blue, "Same-size and same-mtime file replacement was missed");
		write(icon, red_png);
		render(red, "In-place icon update was missed");
		constexpr std::array<std::uint8_t, 3> malformed{1, 2, 3};
		write(icon, malformed);
		render(blue, "Malformed preferred icon did not select the fallback");
		fsys::remove(icon);
		require(::mkfifo(icon.c_str(), 0600) == 0, "Could not create nonregular title fixture");
		render(blue, "Nonregular preferred icon did not select the fallback");
		fsys::remove(icon);
		fsys::remove(other);
		render(black, "Removed title files left stale icon pixels");
		write(icon, red_png);
		render(red, "Recreated preferred icon stayed absent");
		std::cout << "Metal4 title icon files: 10 overlay GPU profiles and 40 exact BGRA bytes; late, replaced, malformed, nonregular and removed sources handled\n";
		test_glyph_clip();
	}
}
