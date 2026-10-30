#include "M4FramePlan.h"

#include <array>
#include <iostream>
#include <limits>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native frame plan differs");
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
	throw std::runtime_error("Invalid frame plan accepted");
}
} // namespace
int main()
{
	try
	{
		using namespace metal4;
		require(plan_drawable_extent(640, 480, 2, 16384) == frame_extent{1280, 960});
		require(plan_drawable_extent(640.25, 480.5, 1.5, 16384) == frame_extent{961, 721});
		require(plan_drawable_extent(0, 480, 2, 16384) == frame_extent{0, 960});
		for (const double bad : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
		{
			rejects([&]
					{ (void)plan_drawable_extent(bad, 1, 1, 16384); });
			rejects([&]
					{ (void)plan_drawable_extent(1, 1, bad, 16384); });
		}
		rejects([]
				{ (void)plan_drawable_extent(1, 1, 0, 16384); });
		rejects([]
				{ (void)plan_drawable_extent(16384.1, 1, 1, 16384); });
		frame_policy policy{{1280, 720}};
		require(plan_frame_rectangle({1280, 720}, {1920, 1080}, policy) == frame_rectangle{0, 0, 1920, 1080});
		require(plan_frame_rectangle({960, 720}, {1920, 1080}, policy) == frame_rectangle{240, 0, 1440, 1080});
		require(plan_frame_rectangle({1280, 720}, {1000, 1000}, policy) == frame_rectangle{0, 219, 1000, 562});
		policy.display_aspect = 4.0 / 3;
		require(plan_frame_rectangle({1280, 720}, {1920, 1080}, policy) == frame_rectangle{240, 0, 1440, 1080});
		require(plan_frame_rectangle({640, 480}, {1920, 1080}, policy) == frame_rectangle{420, 0, 1080, 1080});
		policy.stretch = true;
		require(plan_frame_rectangle({1, 3}, {1920, 1080}, policy) == frame_rectangle{0, 0, 1920, 1080});
		require(plan_frame_rectangle({1, 0}, {1920, 1080}, policy) == frame_rectangle{});
		policy.stretch = false;
		std::size_t cases = 0;
		for (const unsigned width : {1u, 2u, 7u, 1000u, 1920u})
			for (const unsigned height : {1u, 3u, 11u, 720u, 1080u})
				for (const double aspect : {4.0 / 3, 16.0 / 9, 1.0})
				{
					policy.display_aspect = aspect;
					const auto rect = plan_frame_rectangle({640, 480}, {width, height}, policy);
					require(rect.x <= width && rect.width <= width - rect.x && rect.y <= height && rect.height <= height - rect.y);
					++cases;
				}
		policy.gamma = 0;
		rejects([&]
				{ (void)plan_frame_rectangle({1, 1}, {1, 1}, policy); });
		std::cout << "Retina fractional geometry,device bounds,two-stage AV fit,stretch,zero-size and " << cases << " bounded aspect cases passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
