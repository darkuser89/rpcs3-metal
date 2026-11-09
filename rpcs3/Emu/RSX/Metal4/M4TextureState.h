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
	// Interior extent of a native image with a stored one-texel ring.
	std::array<std::uint32_t, 2> stored_border_extent{};
	std::uint32_t stored_border_depth = 0;
};
struct texture_state_overrides
{
	float lod_bias = 0;
	// Zero keeps the register value. Rendering configuration resolves strict
	// mode before passing its override here.
	std::uint32_t anisotropy = 0;
};
[[nodiscard]] inline bool supports_direct_msaa_sampling(const fragment_texture_registers& regs, bool packed_depth)
{
	// Packed depth/stencil uses point addressing for all valid wrap modes.
	if (packed_depth)
		return true;
	const auto min = (regs.filter >> 16) & 7u;
	const auto mag = (regs.filter >> 24) & 7u;
	return min && (mag == 1 || mag == 2 || mag == 4);
}
struct decoded_texture_state
{
	bool enabled = false, depth = false, shadow = false, redirected = false;
	bool redirected_float = false;
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
inline bool stored_border_wrap(unsigned wrap)
{
	return wrap >= static_cast<unsigned>(sampler_wrap::repeat) + 1 &&
		wrap <= static_cast<unsigned>(sampler_wrap::mirror_once_clamp) + 1;
}

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
	state.redirected_float = state.redirected && cache.depth_float;
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
	if (!cache.linear_filtering || state.redirected)
		state.sampler.anisotropy = 1;
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
	const bool one_dimensional = state.policy.dimension == vertex_texture_dimension::one;
	const bool volume = state.policy.dimension == vertex_texture_dimension::three;
	const bool cube_atlas = state.policy.dimension == vertex_texture_dimension::cube;
	const bool stored_border = cache.stored_border_extent != std::array<std::uint32_t, 2>{};
	if (bool(cache.stored_border_depth) != (stored_border && volume))
		throw std::invalid_argument("Stored-border depth disagrees with the guest texture dimension");
	if (stored_border && (cache.actual_mips > 1 || one_dimensional || volume || cube_atlas || state.sampler.anisotropy > 1))
	{
		const auto [logical_width, logical_height] = cache.stored_border_extent;
		if ((cube_atlas && logical_width != logical_height) || state.redirected || cache.samples != 1 ||
			(volume && (state.shadow || cache.stored_border_depth > (regs.control3 >> 20))) ||
			(regs.format & 8) || !logical_width || !logical_height ||
			logical_width > width || logical_height > height)
			throw std::invalid_argument("Unsupported dimension or sampling mode for the stored texture border atlas");
		const bool pixels = !cube_atlas && (format_bits & 0x40);
		auto modes = static_cast<unsigned>(state.sampler.wrap[0]) | (static_cast<unsigned>(state.sampler.wrap[1]) << 3);
		if (one_dimensional) modes |= border_atlas_one_dimensional;
		if (cube_atlas) modes |= border_atlas_cube;
		if (state.sampler.anisotropy > 1)
			modes |= state.sampler.anisotropy << border_atlas_anisotropy_shift;
		if (volume) modes |= static_cast<unsigned>(state.sampler.wrap[2]) << border_atlas_wrap_z_shift;
		if (pixels)
		{
			modes |= border_atlas_pixels;
			if (min == 1 || mag == 1) modes |= border_atlas_point_bias;
		}
		state.coordinates.flags = texture_coordinate_border_atlas;
		state.coordinates.border_atlas = {logical_width, logical_height, cache.actual_mips, modes};
		state.coordinates.border_atlas_depth = cache.stored_border_depth;
		state.sampler.wrap.fill(sampler_wrap::edge);
	}
	else if (stored_border)
	{
		const bool min_linear = min == 2 || min == 4 || min == 6 || min == 7;
		if (state.policy.dimension != vertex_texture_dimension::two || state.redirected || cache.samples != 1 ||
			(regs.format & 8))
			throw std::invalid_argument("Stored texture borders require a supported 2D image");
		const bool mixed = state.policy.linear_filtering && min_linear != (mag != 1);
		const bool linear = state.policy.linear_filtering && (min_linear || mixed);
		if (mixed)
		{
			state.coordinates.flags |= texture_coordinate_mixed_filter;
			if (min_linear) state.coordinates.flags |= texture_coordinate_mixed_min_linear;
			if (format_bits & 0x40) state.coordinates.flags |= texture_coordinate_mixed_pixels;
			state.sampler.min_filter = sampler_min_filter::linear;
			state.sampler.mag_filter = sampler_mag_filter::linear;
		}
		const std::array declared{width, height};
		for (unsigned i = 0; i < declared.size(); ++i)
		{
			const auto logical = cache.stored_border_extent[i];
			if (!logical || logical > declared[i])
				throw std::invalid_argument("Stored texture border extent exceeds the guest texture");
			const float extent = float(logical);
			const float padded = extent + 2.f;
			const auto wrap = state.sampler.wrap[i];
			if (mixed && (wrap == sampler_wrap::repeat || wrap == sampler_wrap::clamp || wrap == sampler_wrap::mirror_once_clamp))
				state.coordinates.flags |= texture_coordinate_mixed_point_edge_x << i;
			if (wrap == sampler_wrap::repeat || wrap == sampler_wrap::mirror)
			{
				const auto axis_flag = wrap == sampler_wrap::repeat ? texture_coordinate_repeat_x : texture_coordinate_mirror_repeat_x;
				state.coordinates.flags |= axis_flag << i;
				if (wrap == sampler_wrap::repeat && linear)
					state.coordinates.flags |= texture_coordinate_repeat_linear;
				if (!mixed && (format_bits & 0x40) && (min == 1 || mag == 1))
					state.coordinates.flags |= texture_coordinate_periodic_pixel_bias;
				state.coordinates.minimum[i] = (format_bits & 0x40) ? extent : 1.f;
				state.coordinates.maximum[i] = .5f / padded;
				continue;
			}
			const bool mirror = wrap == sampler_wrap::mirror_once_edge ||
				wrap == sampler_wrap::mirror_once_border || wrap == sampler_wrap::mirror_once_clamp;
			if (mirror)
				state.coordinates.flags |= i ? texture_coordinate_mirror_y : texture_coordinate_mirror_x;
			// Values are in padded texel units. Clamp to a texel center for
			// nearest, or to the edge/ring midpoint for linear GL-style clamp.
			const float first = wrap == sampler_wrap::border ? .5f : wrap == sampler_wrap::clamp && linear ? 1.f : 1.5f;
			const bool border = wrap == sampler_wrap::border || wrap == sampler_wrap::mirror_once_border;
			const bool clamp = wrap == sampler_wrap::clamp || wrap == sampler_wrap::mirror_once_clamp;
			const float last_inset = border ? .5f : clamp && linear ? 1.f : 1.5f;
			state.coordinates.scale[i] = ((format_bits & 0x40) ? 1.f : extent) / padded;
			state.coordinates.bias[i] = 1.f / padded;
			state.coordinates.minimum[i] = first / padded;
			state.coordinates.maximum[i] = (padded - last_inset) / padded;
		}
		state.coordinates.flags |= texture_coordinate_clamp;
		// The ring is part of the sampled image. Clamp the native sampler to
		// the image edge after selecting the guest edge or border coordinates.
		state.sampler.wrap.fill(sampler_wrap::edge);
	}
	// Fragment samplers stay normalized. Cache coordinates already include the
	// logical guest/native layout scale; UN adds VK's point-filter subpixel bias.
	if (!(state.coordinates.flags & (texture_coordinate_mixed_filter | texture_coordinate_border_atlas)) && (format_bits & 0x40) && (min == 1 || mag == 1))
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
	if (cache.stored_border_depth && state.depth)
		state.view_border_color.fill(border_argb[1]); // 3D depth is stored in native R.
	if (!state.redirected && state.samples == 1 && !(state.coordinates.flags & texture_coordinate_sample_raster) &&
		std::find(state.sampler.wrap.begin(), state.sampler.wrap.end(), sampler_wrap::border) != state.sampler.wrap.end())
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
