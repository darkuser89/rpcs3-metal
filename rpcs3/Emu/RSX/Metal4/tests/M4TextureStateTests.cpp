#include "M4TextureState.h"
#include "M4TextureSnapshot.h"
#include "M4GuestTextureLayout.h"

#include <cmath>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void rejects(F&& action)
{
	bool rejected = false;
	try
	{
		action();
	}
	catch (const std::invalid_argument&)
	{
		rejected = true;
	}
	require(rejected, "Native register adapter accepted invalid state");
}
fragment_texture_registers registers(unsigned format = 0x85)
{
	fragment_texture_registers regs;
	regs.format = (5u << 16) | (format << 8) | (2u << 4) | 1;
	regs.address = (7u << 28) | 0x030303;
	regs.control0 = (1u << 31) | (1024u << 7);
	regs.control1 = 0xaae4;
	regs.filter = (1u << 24) | (3u << 16);
	regs.image_rect = (16u << 16) | 16;
	regs.border_color = 0xff804020;
	return regs;
}
unsigned rgba_mask(const std::array<bool, 4>& argb)
{
	return unsigned(argb[1]) | (unsigned(argb[2]) << 1) | (unsigned(argb[3]) << 2) | (unsigned(argb[0]) << 3);
}
} // namespace
int main()
{
	try
	{
		texture_cache_state cache;
		unsigned layout_cases=0;
		const std::array<unsigned,27> formats={0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x8b,0x8d,0x8e,0x8f,0x90,0x91,0x92,0x93,0x94,0x95,0x97,0x98,0x99,0x9a,0x9b,0x9c,0x9d,0x9e,0x9f};
		const std::array<unsigned,27> block_bytes={1,2,2,2,4,8,16,16,2,4,4,2,4,4,2,2,2,4,2,2,2,8,16,4,2,4,4};
		for(unsigned f=0;f<formats.size();++f) for(unsigned type=0;type<4;++type) for(bool linear:{false,true}) for(bool border:{false,true})
		{
			auto r=registers(formats[f]|(linear?0x20:0));r.offset=0x80000200;
			const unsigned dimension=type==0?1:type==2?3:2;
			r.format=(r.format&~0xffu)|(dimension<<4)|2|(type==3?4:0)|(border?0:8);
			r.control3=(4u<<20)|512;
			const auto p=plan_guest_texture(r);
			const bool bc=formats[f]>=0x86&&formats[f]<=0x88,pair=formats[f]==0x8d||formats[f]==0x8e;
			const unsigned mips=bc?(type==0?1:3):5,layers=type==3?6:1,bw=bc?4:pair?2:1,bh=bc?4:1;
			require(p.offset==0x200&&p.location==1&&p.mips==mips&&p.layers==layers&&p.subresources.size()==mips*layers,"Native guest texture top-level plan differs from register/layout oracle");
			std::size_t position=0;
			for(unsigned face=0;face<layers;++face)
			{
				unsigned w=16,h=type==0?1:16,d=type==2?4:1;
				for(unsigned level=0;level<mips;++level)
				{
					const auto& s=p.subresources[face*mips+level];
					const unsigned cols=(w+bw-1)/bw,rows=(h+bh-1)/bh,edge=border&&!bc?(linear?1:4):0;
					const unsigned columns=linear?512/block_bytes[f]:edge?std::bit_ceil(cols+8):cols;
					const unsigned padded_rows=linear?rows+2*edge:edge?std::bit_ceil(rows+8):rows;
					const std::size_t size=std::size_t(columns)*block_bytes[f]*padded_rows*d;
					require(s.offset==position&&s.storage_bytes==size&&s.row_pitch==columns*block_bytes[f]&&s.width==w&&s.height==h&&s.depth==d&&s.border==edge,"Native mip/face pitch/border offsets differ from independent arithmetic");
					position+=size;w=std::max(1u,w/2);h=std::max(1u,h/2);d=std::max(1u,d/2);
				}
				if(!linear&&face+1<layers) position=((position+127)/128)*128;
			}
			require(p.storage_bytes==position,"Guest texture range includes missing face alignment or unused terminal padding");++layout_cases;
		}
		std::cout<<layout_cases<<" independent27-format mip/face/pitch/border/address layout profiles passed\n";
		struct rsx_register_fixture
		{
			std::array<std::uint32_t,0x10000/4> registers{};
		} method_state;
		for (unsigned i = 0; i < method_state.registers.size(); ++i)
			method_state.registers[i] = i ^ 0x9e3779b9u;
		unsigned snapshot_units = 0;
		for (unsigned mask = 0; mask < 65536; ++mask)
		{
			const auto snapshot = capture_fragment_texture_snapshot(method_state,mask);
			require(snapshot.referenced_mask() == mask,"Frozen texture mask changed");
			for (unsigned unit = 0; unit < 16; ++unit)
			{
				if (!(mask & (1u << unit))) continue;
				const auto& words = snapshot.at(unit);
				const unsigned base = (0x1a00+32*unit)/4;
				const std::array<unsigned,9> expected = {base,base+1,base+2,base+3,base+4,base+5,base+6,base+7,(0x1840+4*unit)/4};
				const std::array<unsigned,9> actual = {words.offset,words.format,words.address,words.control0,words.control1,words.filter,words.image_rect,words.border_color,words.control3};
				for (unsigned c = 0; c < 9; ++c)
					require(actual[c] == (expected[c] ^ 0x9e3779b9u),"Frozen method word/CONTROL3 stride differs from address oracle");
				++snapshot_units;
			}
		}
		for (unsigned unit = 0; unit < 16; ++unit)
			for (unsigned length = 0; length <= 0x700; ++length)
			{
				bool rejected = false;
				try
				{
					(void)fragment_texture_snapshot::capture(std::span(method_state.registers).first(length),1u << unit);
				}
				catch (const std::out_of_range&) { rejected = true; }
				require(rejected == (length < (0x1a20+32*unit)/4),"Referenced method-span bounds changed");
			}
		const auto frozen = capture_fragment_texture_snapshot(method_state,0x8001);
		method_state.registers.fill(0);
		require(frozen.at(15).offset == (((0x1a00+32*15)/4) ^ 0x9e3779b9u),"Register mutation changed frozen texture words");
		for (unsigned index : {1u,16u})
		{
			bool rejected = false;
			try { (void)frozen.at(index); } catch (const std::out_of_range&) { rejected = true; }
			require(rejected,"Unreferenced/out-of-range snapshot unit was exposed");
		}
		rejects([&] { (void)capture_fragment_texture_snapshot(method_state,0x10000); });
		require(fragment_texture_snapshot::capture({},0).referenced_mask() == 0,"Empty snapshot read absent method words");
		std::cout << "65536 frozen method masks," << snapshot_units << " independent referenced-unit address checks,28688 span boundaries,mutation/empty/invalid snapshot cases passed\n";
		const auto baseline = decode_fragment_texture_state(registers(), cache);
		require(baseline.enabled && baseline.format == 0x85 && baseline.policy.requested_mips == 5 && baseline.policy.actual_mips == 1, "Default register/mip decode changed");
		require(baseline.texels.sign_mask == 0 && baseline.texels.gamma_mask == 0 && baseline.texels.expand_mask == 0 && baseline.texels.bx2_mask == 15, "Default identity conversion masks changed");
		unsigned remap_cases = 0;
		for (unsigned remap = 0; remap < 65536; ++remap)
			for (unsigned sign = 0; sign < 16; ++sign)
			{
				auto regs = registers();
				regs.control1 = remap;
				regs.filter |= sign << 28;
				regs.address |= (15u << 20) | (1u << 12);
				const auto state = decode_fragment_texture_state(regs, cache);
				std::array<bool, 4> signs{}, reads{};
				for (unsigned c = 0; c < 4; ++c)
				{
					const unsigned source = (remap >> (c * 2)) % 4, action = (remap >> (8 + c * 2)) % 4;
					reads[c] = action == 2;
					signs[c] = reads[c] && bool(sign & (1u << source));
					const unsigned expected_view = action == 0 ? 4 : action == 1 ? 5
																 : source == 0	 ? 3
																				 : source - 1;
					require(state.view_channels[(c + 3) % 4] == expected_view, "Native RGBA view channels differ from independent ARGB selectors");
				}
				const unsigned expected_sign = rgba_mask(signs), expected_reads = rgba_mask(reads);
				require(state.texels.sign_mask == expected_sign && state.texels.gamma_mask == (15u ^ expected_sign) && state.texels.expand_mask == 0 && state.texels.bx2_mask == expected_reads, "ARGB sign remap,RGBA gamma or constant precedence differs from independent channel tags");
				regs.address &= ~(15u << 20);
				const auto biased = decode_fragment_texture_state(regs, cache);
				require(biased.texels.expand_mask == (expected_reads & ~expected_sign), "Unsigned biased remap touched a constant or signed output");
				++remap_cases;
			}
		for (unsigned raw = 0; raw < 8192; ++raw)
		{
			auto regs = registers();
			regs.filter |= raw;
			const auto state = decode_fragment_texture_state(regs, cache);
			const float expected = raw < 4096 ? float(raw) / 256 : -float(8192 - raw) / 256;
			require(state.sampler.bias == expected, "Signed Q4.8 bias decode changed");
		}
		for (unsigned raw = 0; raw < 4096; ++raw)
		{
			auto regs = registers();
			regs.control0 = (1u << 31) | (raw << 19) | (raw << 7);
			const auto state = decode_fragment_texture_state(regs, cache);
			require(state.sampler.min_lod == float(raw) / 256 && state.sampler.max_lod == float(raw) / 256, "Unsigned Q4.8 LOD decode changed");
		}
		for (unsigned min = 1; min <= 7; ++min)
			for (unsigned mag : {1u, 2u, 4u})
				for (unsigned aniso = 0; aniso < 8; ++aniso)
				{
					auto regs = registers();
					regs.filter = (min << 16) | (mag << 24);
					regs.control0 |= aniso << 4;
					const auto state = decode_fragment_texture_state(regs, cache);
					require(state.sampler.min_filter == static_cast<sampler_min_filter>(min - 1), "Register minification enum changed");
					require(state.sampler.anisotropy == std::array{1u, 2u, 4u, 6u, 8u, 10u, 12u, 16u}[aniso], "Anisotropy register table changed");
				}
		for (unsigned wrap = 1; wrap <= 8; ++wrap)
			for (unsigned axis = 0; axis < 3; ++axis)
			{
				auto regs = registers();
				regs.address = (regs.address & ~(15u << (axis * 8))) | (wrap << (axis * 8));
				require(decode_fragment_texture_state(regs, cache).sampler.wrap[axis] == static_cast<sampler_wrap>(wrap - 1), "Texture wrap register decode changed");
			}
		for (unsigned format : {0x81u, 0x82u, 0x83u, 0x84u, 0x85u, 0x86u, 0x87u, 0x88u, 0x8bu, 0x8du, 0x8eu, 0x8fu, 0x90u, 0x91u, 0x92u, 0x93u, 0x94u, 0x95u, 0x97u, 0x98u, 0x99u, 0x9au, 0x9bu, 0x9cu, 0x9du, 0x9eu, 0x9fu})
		{
			auto regs = registers(format);
			regs.address |= (15u << 20) | (1u << 12);
			regs.filter |= 15u << 28;
			auto policy = cache;
			policy.depth = format >= 0x90 && format <= 0x93;
			const auto state = decode_fragment_texture_state(regs, policy);
			const bool featureless = format == 0x99 || format == 0x9a || format == 0x9b || format == 0x9c || format == 0x9f;
			require(bool(state.texels.control & texel_biased_format) != featureless, "Format feature gating changed");
			require(bool(state.texels.control & texel_channels16) == (format == 0x94 || format == 0x95), "16-bit format feature gating changed");
			if (featureless)
				require(state.texels.sign_mask == 0 && state.texels.gamma_mask == 0 && state.texels.expand_mask == 0, "Float/HILO signed format gained unsupported conversions");
		}
		{
			auto regs = registers();
			regs.control1 = 0xa954; // ARGB:one,R,R,R.
			regs.filter |= 2u << 28;
			regs.address |= (15u << 20) | (1u << 12);
			auto state = decode_fragment_texture_state(regs, cache);
			require(state.texels.sign_mask == 7 && state.texels.gamma_mask == 8 && state.texels.bx2_mask == 7, "Repeated-R known fixture changed");
			require(!state.policy.linear_filtering && state.sampler.border_color == std::array<float, 4>{0, 0, 0, 1}, "Sign filtering/border compensation fixture changed");
			auto host = cache;
			host.host_sign_mask = 7;
			state = decode_fragment_texture_state(regs, host);
			require(state.policy.linear_filtering && state.texels.sign_mask == 0, "Native SNORM optimization did not restore VK linear policy");
			require(std::abs(state.sampler.border_color[0]) < 2e-6f, "Native SNORM border compensation changed");
		}
		for (unsigned compare = 0; compare < 8; ++compare)
			for (bool alpha : {false, true})
			{
				auto regs = registers(0x92);
				regs.address = (regs.address & ~(15u << 28)) | (compare << 28);
				regs.control0 |= alpha ? 4u : 0u;
				auto depth = cache;
				depth.depth = true;
				const auto state = decode_fragment_texture_state(regs, depth);
				require(state.shadow == (!alpha && compare > 0 && compare < 7), "Depth compare/alpha-kill shadow routing changed");
			}
		{
			auto regs = registers(0xc5); // A8R8G8B8 with UN.
			auto transform = cache;
			transform.coordinates = normalize_guest_pixels(16, 16);
			const auto state = decode_fragment_texture_state(regs, transform);
			require(state.coordinates.bias == std::array<float, 3>{.01f / 16, .01f / 16, .01f}, "UN nearest subpixel bias changed");
			regs.filter = (2u << 16) | (2u << 24);
			require(decode_fragment_texture_state(regs, transform).coordinates.bias == std::array<float, 3>{}, "UN linear filter unexpectedly gained subpixel bias");
			texture_state_overrides override;
			override.lod_bias = 100;
			override.anisotropy = 6;
			const auto configured = decode_fragment_texture_state(regs, cache, override);
			require(configured.sampler.bias == 15.99609375f && configured.sampler.anisotropy == 6, "Config override/clamp changed");
		}
		{
			auto regs = registers(0x86);
			regs.image_rect = (1u << 16) | 1;
			require(decode_fragment_texture_state(regs, cache).policy.requested_mips == 1, "Sub-block DXT floor_log2(0) contract changed");
		}
		for (unsigned invalid = 0; invalid < 10; ++invalid)
			rejects([&]
					{
					auto regs = registers(); auto policy = cache; texture_state_overrides override;
					if (invalid == 0) regs.format &= ~(15u << 4);
					if (invalid == 1) regs.format = (regs.format & ~(255u << 8)) | (0x80u << 8);
					if (invalid == 2) regs.image_rect = 0;
					if (invalid == 3) regs.filter &= ~(7u << 16);
					if (invalid == 4) regs.address &= ~15u;
					if (invalid == 5) policy.actual_mips = 6;
					if (invalid == 6) policy.host_sign_mask = 16;
					if (invalid == 7) override.anisotropy = 3;
					if (invalid == 8) regs.address |= 15u << 28;
					if (invalid == 9) policy.samples = 3;
					(void)decode_fragment_texture_state(regs, policy, override); });
		fragment_texture_registers disabled;
		{
			auto regs = registers();
			regs.address = (7u << 28) | 0x040404;
			regs.control1 = 0xa900;
			regs.border_color = 0;
			const auto state = decode_fragment_texture_state(regs, cache);
			require(!state.border_emulation && state.view_border_color == std::array<float, 4>{0, 0, 0, 1} && state.sampler.border_color == std::array<float, 4>{0, 0, 0, 0}, "Native border preimage did not avoid double view remapping");
			regs.control1 = 0xaae4;
			regs.border_color = 0xff804020;
			require(decode_fragment_texture_state(regs, cache).border_emulation, "Custom border without a native canonical preimage was silently approximated");
		}
		require(!decode_fragment_texture_state(disabled, {}).enabled, "Disabled texture decoded active register state");
		std::cout << remap_cases << " independent register/remap sign/gamma/BX2 fixtures,8192 signed bias words,4096 LOD words,27 format feature profiles,filter/wrap/aniso/depth/UN/override/invalid cases passed\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
