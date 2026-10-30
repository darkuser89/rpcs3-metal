#include "M4VertexTextureState.h"
#include <iostream>

using namespace metal4;
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native vertex texture register contract differs");
}
int main()
{
	try
	{
		std::vector<std::uint32_t> methods(0x2000 / 4);
		for (unsigned unit = 0; unit < 4; ++unit)
			for (unsigned word = 0; word < 8; ++word)
				methods[vertex_texture_method_base + unit * 8 + word] = (unit + 1) * 100 + word;
		for (unsigned mask = 0; mask < 16; ++mask)
		{
			const auto snapshot = vertex_texture_snapshot::capture(methods, mask);
			require(snapshot.referenced_mask() == mask);
			for (unsigned unit = 0; unit < 4; ++unit)
				if (mask & (1u << unit))
				{
					const auto& words = snapshot.at(unit);
					const auto base = (unit + 1) * 100;
					require(words.offset == base && words.format == base + 1 && words.address == base + 2 &&
							words.control0 == base + 3 && words.control3 == base + 4 && words.filter == base + 5 &&
							words.image_rect == base + 6 && words.border_color == base + 7);
				}
		}
		const auto frozen = vertex_texture_snapshot::capture(methods, 8);
		methods.assign(methods.size(), 0);
		require(frozen.at(3).control3 == 404);
		unsigned profiles = 0;
		vertex_texture_registers words;
		words.image_rect = (8u << 16) | 8;
		words.control3 = (8u << 20) | 128;
		words.border_color = 0xff000000;
		words.filter = 0xffffffffu;
		texture_cache_state cache;
		cache.actual_mips = 4;
		for (unsigned dimension : {1u, 2u, 3u, 6u})
			for (unsigned s = 1; s <= 8; ++s)
				for (unsigned t = 1; t <= 8; ++t)
					for (unsigned lod = 0; lod < 256; ++lod)
					{
						words.format = 1 | ((dimension == 6 ? 2u : dimension) << 4) | (dimension == 6 ? 4u : 0u) | (0xbbu << 8) | (4u << 16);
						words.address = s | (t << 8) | 0xfff00000;
						words.control0 = 0x80000074 | (lod << 19) | (4095u << 7);
						const auto state = decode_vertex_texture_state(words, cache);
						const auto expected = dimension == 1 ? vertex_texture_dimension::one : dimension == 3 ? vertex_texture_dimension::three
																						   : dimension == 6	  ? vertex_texture_dimension::cube
																											  : vertex_texture_dimension::two;
						require(state.policy.dimension == expected && state.policy.requested_mips == 4 && state.policy.actual_mips == 4);
						require(state.sampler.min_lod == float(lod) / 256 && state.sampler.max_lod == float(4095) / 256);
						require(state.policy.stage == sampler_stage::vertex && state.sampler.min_filter == sampler_min_filter::nearest &&
								state.sampler.mag_filter == sampler_mag_filter::nearest && state.sampler.anisotropy == 1 &&
								state.sampler.bias == 0 && !state.sampler.depth_compare && state.sampler.wrap[2] == sampler_wrap::repeat);
						require(!state.texels.control && !state.texels.sign_mask && !state.texels.gamma_mask &&
								state.view_channels == std::array<std::uint32_t, 4>{0, 1, 2, 3});
						const auto upload = words.upload_words();
						require((upload.format & 8) && upload.control3 == words.control3 && upload.control1 == 0xaae4 &&
								!(upload.control0 & 4) && upload.filter == 0x01010000);
						++profiles;
					}
		words.address = 3 | (3 << 8);
		words.control0 = 0x80000000;
		words.format = 1 | (2 << 4) | (0xfbu << 8) | (4u << 16);
		cache.coordinates = normalize_guest_pixels(8, 8);
		require(!decode_vertex_texture_state(words, cache).sampler.normalized);
		words.format = 1 | (2 << 4) | (0xbbu << 8) | (4u << 16);
		cache.coordinates = {};
		unsigned rejected = 0;
		const auto refusal = [&](auto action)
		{
			try
			{
				action();
			}
			catch (const std::exception&)
			{
				++rejected;
			}
		};
		refusal([&]
				{ (void)vertex_texture_snapshot::capture(methods, 16); });
		refusal([&]
				{ (void)vertex_texture_snapshot::capture(std::span(methods).first(vertex_texture_method_base + 7), 1); });
		refusal([&]
				{ (void)frozen.at(0); });
		refusal([&]
				{ (void)frozen.at(4); });
		const auto bad = [&](auto mutate)
		{
			auto w = words;
			auto c = cache;
			mutate(w, c);
			refusal([&]
					{ (void)decode_vertex_texture_state(w, c); });
		};
		bad([](auto& w, auto&)
			{ w.format &= ~3u; });
		bad([](auto& w, auto&)
			{ w.format &= ~0xf0u; });
		bad([](auto& w, auto&)
			{ w.image_rect = 0; });
		bad([](auto& w, auto&)
			{ w.format &= 0xffff; });
		bad([](auto&, auto& c)
			{ c.actual_mips = 3; });
		bad([](auto&, auto& c)
			{ c.depth = true; });
		bad([](auto&, auto& c)
			{ c.samples = 4; });
		bad([](auto& w, auto&)
			{ w.control0 |= 1u << 19; });
		bad([](auto& w, auto&)
			{ w.address &= ~15u; });
		bad([](auto&, auto& c)
			{ c.native_argb_channels[0] = 6; });
		bad([](auto&, auto& c)
			{ c.coordinates.bias[0] = .01f; });
		bad([](auto&, auto& c)
			{ c.coordinates.scale[0] = .5f; });
		require(rejected == 16);
		std::cout << "Native vertex texture state: " << profiles << " fixed-policy/LOD/dimension profiles,all sixteen masks,CONTROL3/ownership and16 refusals passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
