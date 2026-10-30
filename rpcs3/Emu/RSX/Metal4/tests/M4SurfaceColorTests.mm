#include "M4SurfaceColor.h"
#include "M4SurfaceColorKernels.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
void require(bool value)
{
	if (!value) throw std::runtime_error("Native surface color invariant failed");
}
template <typename Function> bool accepts(Function function)
{
	try { function(); return true; }
	catch (const std::invalid_argument&) { return false; }
	catch (const std::out_of_range&) { return false; }
	catch (const std::overflow_error&) { return false; }
}

struct fixture
{
	std::uint32_t guest, bytes, element;
	MTLPixelFormat native;
	metal4::color_export_remap remap;
};
constexpr std::array<fixture, 14> formats{{
	{1, 2, 2, MTLPixelFormatBGR5A1Unorm, metal4::color_export_remap::rgb_zero},
	{2, 2, 2, MTLPixelFormatBGR5A1Unorm, metal4::color_export_remap::rgb_one},
	{3, 2, 2, MTLPixelFormatB5G6R5Unorm, metal4::color_export_remap::rgba},
	{4, 4, 4, MTLPixelFormatBGRA8Unorm, metal4::color_export_remap::rgb_zero},
	{5, 4, 4, MTLPixelFormatBGRA8Unorm, metal4::color_export_remap::rgb_one},
	{8, 4, 4, MTLPixelFormatBGRA8Unorm, metal4::color_export_remap::rgba},
	{9, 1, 1, MTLPixelFormatR8Unorm, metal4::color_export_remap::blue},
	{10, 2, 2, MTLPixelFormatRG8Unorm, metal4::color_export_remap::blue_green},
	{11, 8, 2, MTLPixelFormatRGBA16Float, metal4::color_export_remap::rgba},
	{12, 16, 4, MTLPixelFormatRGBA32Float, metal4::color_export_remap::rgba},
	{13, 4, 4, MTLPixelFormatR32Float, metal4::color_export_remap::rgba},
	{14, 4, 4, MTLPixelFormatRGBA8Unorm, metal4::color_export_remap::rgb_zero},
	{15, 4, 4, MTLPixelFormatRGBA8Unorm, metal4::color_export_remap::rgb_one},
	{16, 4, 4, MTLPixelFormatRGBA8Unorm, metal4::color_export_remap::rgba}}};

void descriptors()
{
	using namespace metal4;
	for (const auto& f : formats)
	{
		const auto color = static_cast<surface_color>(f.guest);
		const auto d = describe_surface_color(color);
		require(d.format == f.native && d.layout.pixel_bytes == f.bytes &&
			d.layout.endian_element_bytes == f.element && d.layout.export_remap == f.remap);
		if (f.guest == 1 || f.guest == 4 || f.guest == 14) require(d.sampling_swizzle.alpha == MTLTextureSwizzleZero);
		if (f.guest == 2 || f.guest == 5 || f.guest == 15) require(d.sampling_swizzle.alpha == MTLTextureSwizzleOne);
		for (unsigned mask = 0; mask < 16; ++mask)
		{
			const auto expected = f.guest == 3 ? mask % 8 : f.guest == 9 ? (mask / 4 % 2) :
				f.guest == 10 ? (mask / 4 % 2) + (mask / 2 % 2) * 2 : f.guest == 13 ? mask % 2 : mask;
			require(color_write_mask(color, mask) == expected);
			for (std::size_t slot = 0; slot < 4; ++slot)
			{
				render_pipeline_configuration pipeline;
				fragment_program_options fragment;
				configure_surface_color(pipeline, fragment, slot, color, mask);
				require(pipeline.colors[slot] == f.native && pipeline.state.colors[slot].write_mask == expected && fragment.color_remap[slot] == f.remap);
				require(fragment.color_count >= slot + 1);
				if (f.remap != color_export_remap::rgba)
				{
					require(pipeline.shader_blend_mask == (1u << slot));
					pipeline.state.colors[slot].enabled = true;
					require(accepts([&] { configure_surface_color(pipeline, fragment, slot, color, mask); }));
					MTL4LibraryFunctionDescriptor* dummy = [MTL4LibraryFunctionDescriptor new]; dummy.name = @"test";
					require(!accepts([&] { (void)make_render_descriptor(pipeline, dummy, dummy); }));
					configure_shader_blending(pipeline, fragment);
					require(accepts([&] { (void)make_render_descriptor(pipeline, dummy, dummy); }));
					require(fragment.blend_rop.colors[slot].enabled && fragment.blend_rop.remap[slot] == f.remap);
				}
			}
		}
	}
	for (unsigned raw = 0; raw < 256; ++raw)
	{
		const bool supported = (raw >= 1 && raw <= 5) || (raw >= 8 && raw <= 16);
		require(accepts([&] { (void)describe_surface_color(static_cast<surface_color>(raw)); }) == supported);
	}
	require(!accepts([] { (void)color_write_mask(surface_color::argb8, 16); }));
	render_pipeline_configuration pipeline;
	fragment_program_options fragment;
	require(!accepts([&] { configure_surface_color(pipeline, fragment, 4, surface_color::argb8, 15); }));
}

void layouts()
{
	using namespace metal4;
	std::size_t models = 0;
	for (const auto& f : formats)
		for (std::size_t width = 1; width <= 9; ++width)
			for (std::size_t height = 1; height <= 7; ++height)
				for (std::size_t pitch = 0; pitch <= 160; ++pitch)
				{
					const auto actual_pitch = pitch ? pitch : width * f.bytes;
					const bool valid = actual_pitch >= width * f.bytes && actual_pitch % f.bytes == 0;
					texture_buffer_layout layout{};
					require(accepts([&] { layout = surface_byte_layout(static_cast<surface_color>(f.guest), width, height, {13, pitch}); }) == valid);
					if (valid)
					{
						std::size_t size = 0;
						// Independent byte-address enumeration includes complete row padding.
						for (std::size_t y = 0; y < height; ++y)
							for (std::size_t byte = 0; byte < actual_pitch; ++byte) size = y * actual_pitch + byte + 1;
						require(layout.allocation_size == size && layout.bytes_per_row == actual_pitch && layout.bytes_per_image == 0);
					}
					++models;
				}
	constexpr auto maximum = std::numeric_limits<std::size_t>::max();
	require(!accepts([] { (void)surface_byte_layout(surface_color::argb8, 0, 1, {}); }));
	require(!accepts([] { (void)surface_byte_layout(surface_color::argb8, 1, 0, {}); }));
	require(!accepts([] { (void)surface_byte_layout(surface_color::rgba32f, maximum, 1, {}); }));
	require(!accepts([] { (void)surface_byte_layout(surface_color::b8, 1, maximum, {1, 1}); }));
	require(surface_byte_layout(surface_color::b8, 1, 1, {maximum - 1, 1}).allocation_size == 1);
	std::cout << models << " independent surface pitch/address models passed\n";
}
}

int main(int argc, char** argv)
{
	@autoreleasepool
	{
		try
		{
			descriptors(); layouts();
			if (argc == 2)
			{
				std::ofstream out(argv[1]); out << metal4::surface_color_kernels;
				if (!out) throw std::runtime_error("Cannot export native surface kernel");
			}
			std::cout << "All fourteen native surface descriptors/channel masks/boundaries passed without GPU\n";
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
