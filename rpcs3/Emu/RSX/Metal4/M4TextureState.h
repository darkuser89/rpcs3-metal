#pragma once

#include "M4DepthSampling.h"
#include "M4SamplerState.h"
#include "M4TexelProcessing.h"
#include "M4TextureCoordinates.h"

#include <algorithm>
#include <bit>

namespace metal4
{
// Host-endian snapshots of the NV4097 texture methods, not byte-swapped guest
// memory and not Vulkan state. Cache layout/format decisions are kept separate.
struct fragment_texture_registers
{
	std::uint32_t offset = 0;
	std::uint32_t format = 0, address = 0, control0 = 0, control1 = 0;
	std::uint32_t filter = 0, image_rect = 0, control3 = 0, border_color = 0;
	bool operator==(const fragment_texture_registers&) const = default;
};
struct texture_cache_state
{
	texture_coordinates coordinates;
	// Native storage channels in ARGB order:RGBA0..3,zero4,one5. The actual
	// cache/format codec determines this layout before applying register remaps.
	std::array<std::uint32_t, 4> native_argb_channels{3, 0, 1, 2};
	std::uint32_t actual_mips = 1, samples = 1;
	bool depth = false, depth_float = false, linear_filtering = true;
	// Source depth codec:0=color,1=Z16U,2=Z24S8U,3=Z16F,4=Z24S8F.
	// A native D32F image alone cannot recover the guest precision/class.
	std::uint32_t source_depth_format = 0;
	bool hardware_renormalization = false;
	// Post-view RGBA conversions already performed by a native SNORM/sRGB view.
	std::uint32_t host_sign_mask = 0, host_gamma_mask = 0;
};
struct texture_state_overrides
{
	float lod_bias = 0;
	// Zero keeps the register value. Rendering configuration resolves strict
	// mode before passing its override here.
	std::uint32_t anisotropy = 0;
};
struct decoded_texture_state
{
	bool enabled = false, depth = false, shadow = false, redirected = false;
	bool border_emulation = false;
	std::array<float, 4> view_border_color{};
	std::uint32_t format = 0, samples = 1;
	guest_sampler_state sampler;
	sampler_texture_policy policy;
	texture_coordinates coordinates;
	texel_processing texels;
	depth_sample_remap depth_remap{};
	// Decoded ARGB remap; used by the native texture-view/format adapter.
	std::array<std::uint32_t, 4> remap_channels{}, remap_controls{};
	std::array<std::uint32_t, 4> view_channels{0, 1, 2, 3};
};
namespace texture_state_detail
{
inline std::uint32_t features(std::uint32_t format)
{
	switch (format)
	{
	case 0x81:
	case 0x82:
	case 0x83:
	case 0x84:
	case 0x85:
	case 0x86:
	case 0x87:
	case 0x88:
	case 0x8b:
	case 0x8d:
	case 0x8e:
	case 0x8f:
	case 0x97:
	case 0x9d:
	case 0x9e:
		return 7; // Signed, biased normalization and gamma.
	case 0x90:
	case 0x91:
	case 0x92:
	case 0x93:
	case 0x98:
		return 2;
	case 0x94:
		return 10; // Biased normalization,16-bit channels.
	case 0x95:
		return 11;
	case 0x99:
	case 0x9a:
	case 0x9b:
	case 0x9c:
	case 0x9f:
		return 0;
	default:
		throw std::invalid_argument("Unknown RSX texture format in native register adapter");
	}
}
inline std::uint32_t remap_encoding(std::uint32_t format, std::uint32_t value)
{
	const bool high = value >> 16;
	switch (format)
	{
	case 0x9a:
	case 0x9b:
	case 0x9c:
		return (value & ~255u) | 0xe4;
	case 0x9f:
		return (value & ~255u) | (high ? 0x56 : 0x66);
	case 0x94:
	case 0x95:
	case 0x98:
	case 0x99:
	{
		std::uint32_t low = value & 255;
		switch (low)
		{
		case 0xe4:
			low = high ? 0x56 : 0x66;
			break;
		case 0x4e:
			low = high ? 0xa9 : 0x99;
			break;
		case 0xee:
			low = 0xaa;
			break;
		case 0x44:
			low = 0x55;
			break;
		}
		return (value & 0xff00) | low;
	}
	default:
		return value;
	}
}
inline std::uint32_t argb_to_rgba(std::uint32_t mask)
{
	return (mask >> 1) | ((mask & 1) << 3);
}
inline std::uint32_t shuffle_mask(std::uint32_t mask, const decoded_texture_state& state)
{
	std::uint32_t result = 0;
	for (unsigned i = 0; i < 4; ++i)
		if (state.remap_controls[i] == 2 && (mask & (1u << state.remap_channels[i])))
			result |= 1u << i;
	return result;
}
} // namespace texture_state_detail

[[nodiscard]] inline decoded_texture_state decode_fragment_texture_state(const fragment_texture_registers& regs,
																		 const texture_cache_state& cache, const texture_state_overrides& overrides = {})
{
	using namespace texture_state_detail;
	decoded_texture_state state;
	state.enabled = regs.control0 >> 31;
	if (!state.enabled)
		return state;
	if (!cache.actual_mips || (cache.samples != 1 && cache.samples != 2 && cache.samples != 4 && cache.samples != 8) ||
		((cache.host_sign_mask | cache.host_gamma_mask) & ~15u) || (cache.host_sign_mask & cache.host_gamma_mask) ||
		!std::isfinite(overrides.lod_bias))
		throw std::invalid_argument("Invalid native texture cache/override metadata");
	validate_texture_coordinates(cache.coordinates);
	for (auto channel : cache.native_argb_channels)
		if (channel > 5)
			throw std::invalid_argument("Invalid native texture storage channel mapping");
	state.coordinates = cache.coordinates;
	state.samples = cache.samples;
	const unsigned dimension = (regs.format >> 4) & 15;
	const bool cube = regs.format & 4;
	if (dimension < 1 || dimension > 3 || (cube && dimension != 2))
		throw std::invalid_argument("Invalid RSX texture dimension/cube register");
	state.policy.dimension = dimension == 1 ? vertex_texture_dimension::one : dimension == 3 ? vertex_texture_dimension::three
																		  : cube			 ? vertex_texture_dimension::cube
																							 : vertex_texture_dimension::two;
	const unsigned width = regs.image_rect >> 16, height = dimension == 1 ? 1 : regs.image_rect & 65535;
	if (!width || !height || (dimension == 3 && !(regs.control3 >> 20)))
		throw std::invalid_argument("RSX texture has zero logical dimensions");
	const unsigned format_bits = (regs.format >> 8) & 255;
	state.format = format_bits & ~0x60u;
	const auto format_features = features(state.format);
	unsigned mip_extent = std::max(width, height);
	if (state.format >= 0x86 && state.format <= 0x88)
	{
		// RSX floor_log2(0) is zero, so even a sub-block image has one level.
		mip_extent = std::max(1u, std::min(width / 4, height / 4));
	}
	state.policy.requested_mips = std::min(regs.format >> 16, static_cast<unsigned>(std::bit_width(mip_extent)));
	state.policy.actual_mips = cache.actual_mips;
	if (!state.policy.requested_mips || cache.actual_mips > state.policy.requested_mips)
		throw std::invalid_argument("Native cache mip count exceeds RSX exact mip count");
	const auto remap = remap_encoding(state.format, regs.control1);
	for (unsigned i = 0; i < 4; ++i)
	{
		state.remap_channels[i] = (remap >> (i * 2)) & 3;
		state.remap_controls[i] = (remap >> (8 + i * 2)) & 3;
	}
	const unsigned raw_sign = (regs.filter >> 28) & 15;
	const unsigned sign = (format_features & 1) ? shuffle_mask(raw_sign, state) : 0;
	// Address gamma bits are already RGBA; unlike sign/BX2, they are not remapped.
	state.texels.sign_mask = argb_to_rgba(sign);
	state.texels.gamma_mask = (format_features & 4) ? ((regs.address >> 20) & 15) & ~state.texels.sign_mask : 0;
	state.texels.bx2_mask = argb_to_rgba(shuffle_mask(15, state));
	if ((format_features & 2) && ((regs.address >> 12) & 15) == 1)
		state.texels.expand_mask = state.texels.bx2_mask & ~(state.texels.sign_mask | state.texels.gamma_mask);
	if (format_features & 2)
		state.texels.control |= texel_biased_format;
	if (format_features & 8)
		state.texels.control |= texel_channels16;
	if (regs.control0 & 4)
		state.texels.control |= texel_alpha_kill;
	state.depth = cache.depth;
	const unsigned compare = (regs.address >> 28) & 15;
	if (compare > 7)
		throw std::invalid_argument("Invalid RSX texture comparison register");
	const bool depth_format = state.format >= 0x90 && state.format <= 0x93;
	state.shadow = state.depth && depth_format && !(regs.control0 & 4) && compare > 0 && compare < 7;
	state.redirected = state.depth && (state.format == 0x85 || state.format == 0x9e);
	if (state.depth && !depth_format && !state.redirected && state.format != 0x94)
		throw std::invalid_argument("RSX depth cache bound through an unsupported texture format");
	if (depth_format && !state.depth)
		throw std::invalid_argument("RSX depth format requires explicit native depth cache metadata");
	if (!state.depth && !cache.hardware_renormalization &&
		(state.format == 0x82 || state.format == 0x83 || state.format == 0x84 || state.format == 0x8f || state.format == 0x97 || state.format == 0x9d))
		state.texels.control |= texel_renormalize;
	// VK clears conversions implemented by a native view before resolving the
	// filter policy. Shader sign-extension forces nearest; native SNORM does not.
	state.texels.sign_mask &= ~cache.host_sign_mask;
	state.texels.gamma_mask &= ~cache.host_gamma_mask;
	state.policy.linear_filtering = cache.linear_filtering && !state.texels.sign_mask && !state.redirected;
	state.sampler.depth_compare = state.shadow;
	state.sampler.depth_float = state.shadow && cache.depth_float;
	state.sampler.compare_function = 0x200 | compare;
	state.sampler.min_lod = float((regs.control0 >> 19) & 4095) / 256;
	state.sampler.max_lod = float((regs.control0 >> 7) & 4095) / 256;
	const unsigned bias_bits = regs.filter & 8191;
	const int bias = bias_bits & 4096 ? int(bias_bits) - 8192 : int(bias_bits);
	state.sampler.bias = std::clamp(float(bias) / 256 + overrides.lod_bias, -16.f, 16.f - 1.f / 256);
	constexpr std::array<unsigned, 8> anisotropy = {1, 2, 4, 6, 8, 10, 12, 16};
	state.sampler.anisotropy = overrides.anisotropy ? overrides.anisotropy : anisotropy[(regs.control0 >> 4) & 7];
	if (std::find(anisotropy.begin(), anisotropy.end(), state.sampler.anisotropy) == anisotropy.end())
		throw std::invalid_argument("Invalid native anisotropy configuration override");
	const unsigned min = (regs.filter >> 16) & 7, mag = (regs.filter >> 24) & 7;
	if (!min || (mag != 1 && mag != 2 && mag != 4))
		throw std::invalid_argument("Invalid RSX min/mag filter register");
	state.sampler.min_filter = static_cast<sampler_min_filter>(min - 1);
	state.sampler.mag_filter = mag == 1 ? sampler_mag_filter::nearest : mag == 2 ? sampler_mag_filter::linear
																				 : sampler_mag_filter::convolution;
	for (unsigned i = 0; i < 3; ++i)
	{
		const unsigned wrap = (regs.address >> (i * 8)) & 15;
		if (wrap < 1 || wrap > 8)
			throw std::invalid_argument("Invalid RSX texture wrap register");
		state.sampler.wrap[i] = static_cast<sampler_wrap>(wrap - 1);
	}
	// Fragment samplers stay normalized. Cache coordinates already include the
	// logical guest/native layout scale; UN adds VK's point-filter subpixel bias.
	if ((format_bits & 0x40) && (min == 1 || mag == 1))
		for (unsigned i = 0; i < 3; ++i)
			state.coordinates.bias[i] += .01f * state.coordinates.scale[i];
	// Border compensation precedes shader sign-extension, as in RSXTexture.
	std::array<float, 4> border_argb;
	for (unsigned i = 0; i < 4; ++i)
	{
		unsigned byte = (regs.border_color >> (24 - i * 8)) & 255;
		if (state.texels.sign_mask && (raw_sign & (1u << i)))
			byte = (byte - 128) & 255;
		border_argb[i] = float(byte) / 255;
	}
	for (unsigned rgba = 0; rgba < 4; ++rgba)
	{
		const unsigned argb = (rgba + 1) % 4, control = state.remap_controls[argb];
		state.view_channels[rgba] = control == 0 ? 4 : control == 1 ? 5
																	: cache.native_argb_channels[state.remap_channels[argb]];
		state.sampler.border_color[rgba] = control == 0 ? 0.f : control == 1 ? 1.f
																			 : border_argb[state.remap_channels[argb]];
		if (cache.host_sign_mask && (raw_sign & (1u << argb)))
			state.sampler.border_color[rgba] = (state.sampler.border_color[rgba] - 128.f / 255) * (255.f / 127);
		state.depth_remap[rgba] = control == 0 ? depth_sample_channel::zero : control == 1 ? depth_sample_channel::one
																						   : depth_sample_channel::depth;
	}
	state.view_border_color = state.sampler.border_color;
	if (std::find(state.sampler.wrap.begin(), state.sampler.wrap.end(), sampler_wrap::border) != state.sampler.wrap.end())
	{
		// Metal applies the texture-view swizzle to native border values too.
		// Find a canonical descriptor colour that yields the requested final
		// view colour; pre-remapping the descriptor itself applies remaps twice.
		state.border_emulation = true;
		for (const auto candidate : {std::array<float, 4>{0, 0, 0, 0}, std::array<float, 4>{0, 0, 0, 1}, std::array<float, 4>{1, 1, 1, 1}})
		{
			bool matches = true;
			for (unsigned c = 0; c < (state.shadow ? 1u : 4u); ++c)
			{
				const unsigned channel = state.view_channels[c];
				const auto depth_channel = state.depth_remap[c];
				const float actual = state.shadow ? candidate[0] : state.depth ? (depth_channel == depth_sample_channel::depth ? candidate[0] : depth_channel == depth_sample_channel::zero ? 0.f
																																															: 1.f)
															   : channel < 4   ? candidate[channel]
															   : channel == 4  ? 0.f
																			   : 1.f;
				matches &= actual == state.view_border_color[c];
			}
			if (matches)
			{
				state.sampler.border_color = candidate;
				state.border_emulation = false;
				break;
			}
		}
	}
	validate_texture_coordinates(state.coordinates);
	validate_texel_processing(state.texels);
	return state;
}
} // namespace metal4
