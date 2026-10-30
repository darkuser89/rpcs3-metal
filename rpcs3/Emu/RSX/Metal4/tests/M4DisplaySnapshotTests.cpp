#include "M4DisplaySnapshot.h"
#include <array>
#include <iostream>

namespace
{
struct buffer
{
	std::uint32_t offset, width, height, pitch;
};
struct flip
{
	unsigned buffer;
	bool skip_frame;
};
struct av
{
	unsigned resolution_x = 1280, resolution_y = 720, format = 0;
	bool state = false, stereo_enabled = false;
};
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Display snapshot assertion failed");
}
template <typename F>
void rejects(F&& f)
{
	try
	{
		f();
	}
	catch (const std::exception&)
	{
		return;
	}
	throw std::runtime_error("Invalid display snapshot accepted");
}
struct checked_buffers
{
	mutable unsigned reads = 0;
	std::array<buffer, 8> values{};
	std::size_t size() const { return values.size(); }
	const buffer& operator[](std::size_t index) const
	{
		++reads;
		return values.at(index);
	}
};
} // namespace
int main()
{
	try
	{
		using namespace metal4;
		checked_buffers buffers;
		av config;
		require(!capture_display_snapshot(buffers, 0, flip{0, false}, config));
		require(!capture_display_snapshot(buffers, 8, flip{UINT32_MAX, false}, config));
		require(!capture_display_snapshot(buffers, 9, flip{UINT32_MAX, true}, config));
		rejects([&]
				{ (void)capture_display_snapshot(buffers, 9, flip{0, false}, config); });
		require(buffers.reads == 0);
		buffers.values[2] = {0x100, 0, 0, 0};
		auto fallback = capture_display_snapshot(buffers, 3, flip{2, false}, config);
		require(fallback && fallback->width == 1280 && fallback->height == 720 && fallback->pitch == 5120);
		buffers.values[2] = {0x200, 1920, 1080, 0};
		config.state = true;
		for (unsigned format = 0; format < 3; ++format)
		{
			config.format = format;
			const auto result = capture_display_snapshot(buffers, 3, flip{2, false}, config);
			require(result && result->width == 1280 && result->height == 720 && result->pitch == 1920 * (format == 2 ? 8 : 4));
			buffers.values[2].width = 1;
			require(result->width == 1280); // Owned result survives source mutation.
			buffers.values[2].width = 1920;
		}
		config.stereo_enabled = true;
		config.resolution_y = 1470;
		const auto stereo = capture_display_snapshot(buffers, 3, flip{2, false}, config);
		require(stereo && stereo->width == 1280 && stereo->height == 720 && stereo->pitch == 1920 * 8 &&
			stereo->right_offset == 0x200u + (720u + 30) * 1920u * 8);
		const auto right = display_eye(*stereo, 1);
		require(right.offset == *stereo->right_offset && !right.right_offset && right.width == 1280 && right.height == 720);
		require(display_eye(*stereo, 0).offset == 0x200 && !display_eye(*stereo, 0).right_offset);
		rejects([&] { (void)display_eye(right, 1); });
		rejects([&] { (void)display_eye(*stereo, 2); });
		for (unsigned format = 0; format < 3; ++format)
		{
			config.format = format;
			const auto value = capture_display_snapshot(buffers, 3, flip{2, false}, config);
			require(value->height == 720 && value->pitch == 1920 * (format == 2 ? 8 : 4) &&
				*value->right_offset == value->offset + 750 * value->pitch);
		}
		config.resolution_y = 30;
		rejects([&] { (void)capture_display_snapshot(buffers, 3, flip{2, false}, config); });
		config.resolution_y = 1470;
		buffers.values[2].offset = 0x7ffffff0;
		rejects([&] { (void)capture_display_snapshot(buffers, 3, flip{2, false}, config); });
		buffers.values[2].offset = 0x200;
		config.stereo_enabled = false;
		config.resolution_y = 720;
		config.format = 99;
		rejects([&]
				{ (void)capture_display_snapshot(buffers, 3, flip{2, false}, config); });
		config.format = 0;
		config.resolution_x = 0;
		require(!capture_display_snapshot(buffers, 3, flip{2, false}, config));
		config.resolution_x = 1280;
		buffers.values[2].pitch = 1;
		rejects([&]
				{ (void)capture_display_snapshot(buffers, 3, flip{2, false}, config); });
		buffers.values[2] = {0, UINT32_MAX, 1, 0};
		rejects([&]
				{ (void)capture_display_snapshot(buffers, 3, flip{2, false}, config); });
		rejects([]
				{ (void)display_coverage({0x80000000, 1, 1, 4}, 0); });
		rejects([]
				{ (void)display_coverage({0, 2, 2, 8}, UINT32_MAX - 7); });
		std::cout << "Display index/count/skip checked before access;owned AV fallback,crop,pitch,three formats and invalid inputs passed\n";
		std::cout << "Stereo owned eye extents/AV crop,uncropped pitch,30-line padding,all3formats,explicit eye selection,right-eye domain check andlocal-domain overflow passed\n";
		unsigned stereo_cases = 0;
		for (const unsigned eye_height : {1u,2u,7u,720u}) for (const unsigned width : {2u,7u})
			for (const unsigned storage_height : {eye_height,eye_height + 3,eye_height * 2 + 30})
				for (unsigned format = 0; format < 3; ++format) for (const bool packed : {false,true})
					for (const unsigned offset : {0u,65536u})
					{
						const unsigned bpp = format == 2 ? 8 : 4;
						av video{width - 1,eye_height * 2 + 30,format,true,true};
						const std::array<buffer,1> inputs{{{offset,width,storage_height,packed ? 0u : width * bpp + 3}}};
						const auto snapshot = capture_display_snapshot(inputs,1,flip{0,false},video);
						const auto pitch = packed ? width * bpp : width * bpp + 3;
						require(snapshot && snapshot->width == width - 1 && snapshot->height == eye_height && snapshot->pitch == pitch &&
							*snapshot->right_offset == offset + (eye_height + 30) * pitch && display_eye(*snapshot,1).format == static_cast<display_format>(format));
						++stereo_cases;
					}
		std::cout << stereo_cases << " independent stereo AV crop/pitch/height/format/offset models passed\n";
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
