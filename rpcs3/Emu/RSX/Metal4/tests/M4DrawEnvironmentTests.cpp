#include "M4DrawEnvironment.h"
#include "M4DrawEnvironmentFixtures.h"

#include <cmath>
#include <iostream>
#include <limits>

using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
int main()
{
	try
	{
		unsigned profiles = 0;
		test::environment_state state;
		state.width = 8;
		state.scale = {2, 1, .5f};
		state.offset = {3, 1, .25f};
		state.point = 3;
		state.near = -.25f;
		state.far = 2;
		state.branch = 0x80012345;
		state.reference = .75f;
		state.fog0 = .375f;
		state.fog1 = .625f;
		for (unsigned i = 0; i < 32; ++i)
			state.pattern[i] = 1u << i;
		for (unsigned configuration = 0; configuration < 729; ++configuration)
		{
			unsigned bits = configuration, mask = 0;
			std::array<float, 6> factors{};
			state.clip = 0xf0000000;
			for (unsigned plane = 0; plane < 6; ++plane)
			{
				const auto op = bits % 3;
				bits /= 3;
				state.clip |= op << (4 * plane);
				if (op)
				{
					mask |= 1u << plane;
					factors[plane] = op == 1 ? -1.f : 1.f;
				}
			}
			for (unsigned mode = 0; mode < 6; ++mode)
				for (unsigned window = 0; window < 4; ++window)
					for (unsigned alpha = 0; alpha < 8; ++alpha)
					{
						state.fog = mode ? 0x7ff + mode : 0x2601;
						state.origin = window & 1;
						state.center = window >> 1;
						state.alpha = 0x200 + alpha;
						const auto snapshot = draw_environment_snapshot::capture(state);
						vertex_context vertex;
						fragment_context fragment;
						vertex.constant_count = 512;
						vertex.output_mask = 0xffff;
						fragment.alpha_test = 1;
						fragment.texcoord_2d_mask = 19;
						snapshot.apply(vertex, fragment);
						require(vertex.clip_enable == mask && vertex.clip_factors == factors, "User clip plane op/enable mapping differs");
						require(vertex.branch_bits == state.branch && vertex.point_size == 3 && vertex.z_near == -.25f && vertex.z_far == 2,
								"Vertex runtime parameters changed");
						require(fragment.fog_mode == mode + 1 && fragment.fog_param0 == .375f && fragment.fog_param1 == .625f &&
									fragment.alpha_function == alpha && fragment.alpha_reference == .75f,
								"Fog/alpha function or parameters differ");
						require(fragment.wpos_scale == (state.origin ? -1 : 1) && fragment.wpos_bias_x == (state.center ? -.5f : 0) &&
									fragment.wpos_bias_y == (state.origin ? 4 : 0) - (state.center ? .5f : 0),
								"Window origin/pixel center mapping differs");
						require(vertex.constant_count == 512 && vertex.output_mask == 0xffff && fragment.alpha_test == 1 &&
									fragment.texcoord_2d_mask == 19 && fragment.stipple_pattern == state.pattern,
								"Environment overwrote program/data state or stipple words");
						++profiles;
					}
		}
		// Independent viewport equation: guest pixel = scale * coordinate +
		// offset. Compare to row-vector shader matrix then native viewport.
		auto snapshot = draw_environment_snapshot::capture(state);
		vertex_context vertex;
		fragment_context fragment;
		snapshot.apply(vertex, fragment);
		for (std::size_t width = 1; width <= snapshot.width(); ++width)
			for (std::size_t height = 1; height <= snapshot.height(); ++height)
				snapshot.validate_framebuffer_extent(width, height, framebuffer_extent_policy::clip_to_framebuffer);
		unsigned extent_refusals = 0;
		const auto invalid_extent = [&](std::size_t width, std::size_t height, framebuffer_extent_policy policy)
		{
			try
			{
				snapshot.validate_framebuffer_extent(width, height, policy);
			}
			catch (const std::invalid_argument&)
			{
				++extent_refusals;
			}
		};
		invalid_extent(0, 4, framebuffer_extent_policy::clip_to_framebuffer);
		invalid_extent(4, 0, framebuffer_extent_policy::clip_to_framebuffer);
		invalid_extent(9, 4, framebuffer_extent_policy::clip_to_framebuffer);
		invalid_extent(8, 5, framebuffer_extent_policy::clip_to_framebuffer);
		invalid_extent(7, 4, framebuffer_extent_policy::exact);
		invalid_extent(8, 3, framebuffer_extent_policy::exact);
		invalid_extent(8, 4, static_cast<framebuffer_extent_policy>(2));
		require(extent_refusals == 7, "Framebuffer crop/exact extent validation differs");
		for (float x : {-2.f, -.5f, 0.f, 2.f})
			for (float y : {-3.f, -.25f, 0.f, 4.f})
				for (float w : {.5f, 1.f, 2.f})
				{
					const auto& m = vertex.scale_offset;
					const float metal_x = ((x * m[0] + w * m[3]) / w + 1) * state.width / 2.f;
					const float metal_y = (1 - (y * m[5] + w * m[7]) / w) * state.height / 2.f;
					require(std::abs(metal_x - (2 * x / w + 3)) < 1e-5f && std::abs(metal_y - (y / w + 1)) < 1e-5f,
							"Native viewport does not reconstruct guest pixels");
				}
		state.pattern.fill(0);
		state.branch = 0;
		snapshot.apply(vertex, fragment);
		require(fragment.stipple_pattern[31] == 0x80000000 && vertex.branch_bits == 0x80012345, "Environment snapshot borrowed live registers");
		state.multisample = true;
		state.samples = 0x5a5a;
		draw_environment_snapshot::capture(state).apply(vertex, fragment);
		require(fragment.sample_mask == 0x5a5a, "Enabled guest sample mask omitted");
		state.multisample = false;
		draw_environment_snapshot::capture(state).apply(vertex, fragment);
		require(fragment.sample_mask == 0x5a5a, "Disabled MSAA hint discarded guest sample mask");
		unsigned sample_profiles = 0;
		for (unsigned flags = 0; flags < 8; ++flags)
			for (const unsigned mask : {0u, 1u, 5u, 0x5a5au, 0xffffu, 0xffffffffu})
			{
				auto source = state;
				source.multisample = flags & 1;
				source.alpha_coverage = flags & 2;
				source.alpha_one = flags & 4;
				source.samples = mask;
				const auto owned = draw_environment_snapshot::capture(source);
				source.multisample = !source.multisample;
				source.alpha_coverage = !source.alpha_coverage;
				source.alpha_one = !source.alpha_one;
				source.samples = ~mask;
				owned.apply(vertex, fragment);
				require(owned.msaa_enabled() == bool(flags & 1) && owned.alpha_to_coverage() == bool(flags & 2) &&
					owned.alpha_to_one() == bool(flags & 4) && fragment.sample_mask == mask,
					"Owned sample mask/A2C/A2One borrowed registers or gated the mask on MSAA hint");
				++sample_profiles;
			}
		std::cout << "Native owned multisample environment:" << sample_profiles << " mask/MSAA-hint/A2C/A2One profiles survive source mutation passed\n";
		unsigned bounds_profiles = 0;
		for (bool enabled : {false, true})
			for (const auto range : {std::array{0.f, 1.f}, std::array{.2f, .3f}, std::array{.25f, .25f}, std::array{.8f, 1.f}})
			{
				auto source = state;
				source.bounds = enabled;
				source.bounds_min = range[0];
				source.bounds_max = range[1];
				const auto owned = draw_environment_snapshot::capture(source);
				source.bounds = !enabled;
				source.bounds_min = std::numeric_limits<float>::quiet_NaN();
				source.bounds_max = std::numeric_limits<float>::infinity();
				require(owned.depth_bounds_enabled() == enabled && owned.depth_min() == (enabled ? range[0] : 0.f) &&
							owned.depth_max() == (enabled ? range[1] : 1.f), "Depth bounds snapshot borrowed or changed guest limits");
				fragment.depth_bounds_enabled = 19;
				owned.apply(vertex, fragment);
				require(fragment.depth_bounds_enabled == 19, "Environment enabled an unprepared shader depth mirror");
				++bounds_profiles;
			}
		auto disabled_bounds = state;
		disabled_bounds.bounds_min = std::numeric_limits<float>::quiet_NaN();
		disabled_bounds.bounds_max = std::numeric_limits<float>::infinity();
		const auto disabled = draw_environment_snapshot::capture(disabled_bounds);
		require(!disabled.depth_bounds_enabled() && disabled.depth_min() == 0 && disabled.depth_max() == 1,
				"Disabled depth bounds consumed poisoned limits");
		unsigned rejected = 0;
		unsigned raster_profiles = 0;
		for (bool clamp : {false, true}) for (bool clip : {false, true})
			for (bool fill : {false, true}) for (float bias : {-32.f, 0.f, 64.f})
				for (float slope : {-2.f, 0.f, 3.f})
				{
					auto source = state;
					source.depth_clamp = clamp; source.depth_clip = clip; source.offset_fill = fill;
					source.polygon_bias = bias; source.polygon_slope = slope;
					const auto captured = draw_environment_snapshot::capture(source);
					source.depth_clamp = !clamp; source.depth_clip = !clip; source.offset_fill = !fill;
					source.polygon_bias = source.polygon_slope = std::numeric_limits<float>::quiet_NaN();
					require(captured.depth_clamp_enabled() == (clamp || !clip) &&
						captured.polygon_bias() == (fill ? bias : 0) && captured.polygon_slope() == (fill ? slope : 0),
						"Owned polygon offset/clip/clamp changed after capture");
					++raster_profiles;
				}
		auto disabled_offset = state;
		disabled_offset.polygon_bias = std::numeric_limits<float>::quiet_NaN();
		disabled_offset.polygon_slope = std::numeric_limits<float>::infinity();
		require(draw_environment_snapshot::capture(disabled_offset).polygon_bias() == 0 &&
			draw_environment_snapshot::capture(disabled_offset).polygon_slope() == 0, "Disabled polygon offset read poisoned values");
		std::cout << "Native owned raster environment:" << raster_profiles << " fill/bias/slope/clip/clamp profiles and disabled poison isolation passed\n";
		const auto refusal = [&](auto mutate)
		{
			auto bad = state;
			mutate(bad);
			try
			{
				(void)draw_environment_snapshot::capture(bad);
			}
			catch (const std::exception&)
			{
				++rejected;
			}
		};
		refusal([](auto& s)
				{ s.width = 0; });
		refusal([](auto& s)
				{ s.height = 0; });
		refusal([](auto& s)
				{ s.clip = 3; });
		refusal([](auto& s)
				{ s.clip = 15u << 20; });
		refusal([](auto& s)
				{ s.alpha = 0x208; });
		refusal([](auto& s)
				{ s.alpha = 0x1ff; });
		refusal([](auto& s)
				{ s.fog = 0x805; });
		refusal([](auto& s)
				{ s.origin = 2; });
		refusal([](auto& s)
				{ s.center = 2; });
		refusal([](auto& s)
				{ s.null_pattern = true; });
		refusal([](auto& s) { s.bounds = true; s.bounds_min = -.1f; });
		refusal([](auto& s) { s.bounds = true; s.bounds_max = 1.1f; });
		refusal([](auto& s) { s.bounds = true; s.bounds_min = .8f; s.bounds_max = .2f; });
		refusal([](auto& s) { s.bounds = true; s.bounds_min = std::numeric_limits<float>::quiet_NaN(); });
		refusal([](auto& s) { s.bounds = true; s.bounds_max = std::numeric_limits<float>::infinity(); });
		refusal([](auto& s) { s.offset_fill = true; s.polygon_bias = std::numeric_limits<float>::quiet_NaN(); });
		refusal([](auto& s) { s.offset_fill = true; s.polygon_slope = std::numeric_limits<float>::infinity(); });
		require(rejected == 17, "Environment refusal missing");
		std::cout << "Native owned depth-bounds environment:" << bounds_profiles << " enabled/disabled ranges,exact ownership,disabled poison isolation and five malformed bounds refusals passed\n";

		std::cout << "Native draw environment: " << profiles << " clip/fog/window/alpha profiles, guest pixel oracle,owned stipple/branch and seventeen refusals passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
