#include "M4PipelineSnapshot.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
using namespace metal4;
void require(bool v)
{
	if (!v)
		throw std::runtime_error("RSX pipeline snapshot oracle failed");
}
std::vector<std::uint32_t> registers()
{
	std::vector<std::uint32_t> m(0x1840 / 4);
	m[0x314 / 4] = 0x00010001;
	m[0x318 / 4] = 0;
	m[0x320 / 4] = 0x80068006;
	m[0x324 / 4] = 0x01010101;
	m[0x370 / 4] = 0xfff0;
	m[0xa6c / 4] = 0x201;
	m[0x330 / 4] = m[0x350 / 4] = 0x207;
	for (unsigned a : {0x33cu, 0x340u, 0x344u, 0x35cu, 0x360u, 0x364u})
		m[a / 4] = 0x1e00;
	m[0x32c / 4] = m[0x338 / 4] = m[0x34c / 4] = m[0x358 / 4] = 255;
	m[0x378 / 4] = 0x1503;
	m[0x1834 / 4] = 0x900;
	m[0x1830 / 4] = 0x405;
	return m;
}
} // namespace
int main()
{
	try
	{
		std::size_t count = 0;
		auto m = registers();
		for (unsigned masks = 0; masks < 65536; ++masks)
		{
			const unsigned first = masks & 15;
			m[0x324 / 4] = ((first & 1) ? 0x10000 : 0) | ((first & 2) ? 0x100 : 0) | ((first & 4) ? 1 : 0) | ((first & 8) ? 0x1000000 : 0);
			m[0x370 / 4] = 0;
			for (unsigned i = 1; i < 4; ++i)
			{
				const auto logical = (masks >> (i * 4)) & 15;
				const auto raw = ((logical & 1) ? 2u : 0u) | ((logical & 2) ? 4u : 0u) | ((logical & 4) ? 8u : 0u) | ((logical & 8) ? 1u : 0u);
				m[0x370 / 4] |= raw << (i * 4);
			}
			const auto state = pipeline_register_snapshot::capture(m).render_state(1);
			for (unsigned i = 0; i < 4; ++i)
				require(state.colors[i].write_mask == ((masks >> (i * 4)) & 15));
			++count;
		}
		m = registers();
		for (unsigned main : {0u, 1u, 2u, 0x80000000u})
			for (unsigned mrt = 0; mrt < 256; ++mrt)
				for (unsigned logic : {0u, 1u, 2u})
					for (bool forced : {false, true})
					{
						m[0x310 / 4] = main;
						m[0x36c / 4] = mrt;
						m[0x374 / 4] = logic;
						const auto state = pipeline_register_snapshot::capture(m).render_state(4, forced);
						for (unsigned i = 0; i < 4; ++i)
							require(state.colors[i].enabled == (!logic && !forced && (((bool(main) ? 1u : 0u) | mrt) & (1u << i))));
						++count;
					}
		m = registers();
		for (unsigned bits = 0; bits < 32; ++bits)
			for (unsigned ref = 0; ref < 65536; ++ref)
			{
				m[0xa74 / 4] = bits & 1;
				m[0xa70 / 4] = bits & 2;
				m[0x328 / 4] = bits & 4;
				m[0x348 / 4] = bits & 8;
				m[0x374 / 4] = bits & 16;
				m[0x334 / 4] = ref;
				m[0x354 / 4] = ~ref;
				m[0x338 / 4] = ref * 17;
				m[0x358 / 4] = ref * 31;
				m[0x32c / 4] = ~ref;
				m[0x34c / 4] = ref * 7;
				const auto frozen = pipeline_register_snapshot::capture(m);
				const auto decoded = decode_render_state(frozen.render_state(2));
				require(decoded.depth_write == bool((bits & 1) && (bits & 2)) && decoded.stencil_test == bool(bits & 4));
				if (bits & 4)
				{
					require(decoded.front.reference == (ref & 255) && decoded.front.read_mask == ((ref * 17) & 255) && decoded.front.write_mask == ((~ref) & 255));
					require(decoded.back.reference == ((bits & 8) ? ((~ref) & 255) : (ref & 255)));
					require(decoded.back.read_mask == ((bits & 8) ? ((ref * 31) & 255) : ((ref * 17) & 255)));
				}
				++count;
			}
		for (std::size_t size = 0; size < 0x183c / 4 + 1; ++size)
		{
			bool refused = false;
			try
			{
				(void)pipeline_register_snapshot::capture(std::span(m).first(size));
			}
			catch (const std::out_of_range&)
			{
				refused = true;
			}
			require(refused);
		}
		m = registers();
		const auto captured = pipeline_register_snapshot::capture(m);
		std::fill(m.begin(), m.end(), 0xffffffffu);
		require(captured.render_state(1).colors[0].write_mask == 15 && captured.clockwise() && captured.cull() == pipeline_cull::none);
		m = registers();
		for (unsigned v = 0; v < 65536; ++v)
		{
			m[0x31c / 4] = v | (v << 16);
			m[0x37c / 4] = v | (v << 16);
			const auto snapshot = pipeline_register_snapshot::capture(m);
			const auto native = snapshot.blend_color(true);
			const auto sign = v >> 15, exponent = (v >> 10) & 31, mantissa = v & 1023;
			const float oracle = exponent == 31 ? (mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) : exponent ? std::ldexp(float(1024 + mantissa), int(exponent) - 25)
																																						   : std::ldexp(float(mantissa), -24);
			for (float lane : native)
				if (std::isnan(oracle))
					require(std::isnan(lane));
				else
					require(std::bit_cast<std::uint32_t>(lane) == std::bit_cast<std::uint32_t>(sign ? -oracle : oracle));
			const auto unorm = snapshot.blend_color(false);
			require(unorm == std::array<float, 4>{(v & 255) / 255.f, (v >> 8) / 255.f, (v & 255) / 255.f, (v >> 8) / 255.f});
		}
		std::cout << count << " raw pipeline mask/blend/logic/depth/stencil fixtures,short spans and frozen ownership passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
