#include "M4Sampler.h"
#include "M4TextureCoordinates.h"
#include "M4TexelProcessing.h"

#include <cmath>
#include <iostream>
#include <limits>

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
	require(rejected, "Invalid/unimplemented sampler state was silently accepted");
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			unsigned cases = 0;
			const std::array filters{sampler_min_filter::nearest, sampler_min_filter::linear, sampler_min_filter::nearest_nearest, sampler_min_filter::linear_nearest, sampler_min_filter::nearest_linear, sampler_min_filter::linear_linear, sampler_min_filter::convolution};
			for (unsigned f = 0; f < filters.size(); ++f)
				for (const auto mag : {sampler_mag_filter::nearest, sampler_mag_filter::linear, sampler_mag_filter::convolution})
					for (const bool linear : {false, true})
						for (const auto mips : {std::array{1u, 1u}, std::array{5u, 1u}, std::array{5u, 3u}, std::array{5u, 5u}})
							for (const unsigned anisotropy : {1u, 2u, 4u, 6u, 8u, 10u, 12u, 16u})
							{
								guest_sampler_state state;
								state.min_filter = filters[f];
								state.mag_filter = mag;
								state.anisotropy = anisotropy;
								state.min_lod = 1.25;
								state.max_lod = 9;
								state.bias = 0.26;
								sampler_texture_policy policy;
								policy.linear_filtering = linear;
								policy.requested_mips = mips[0];
								policy.actual_mips = mips[1];
								const auto key = resolve_sampler(state, policy);
								const bool mipmapped = f >= 2 && f <= 5 && mips[1] > 1;
								const bool trilinear = mipmapped && linear && f >= 4;
								require(key.min_filter == (linear && (f == 1 || f == 3 || f == 5 || f == 6) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest), "Minification/format policy differs from VK");
								require(key.mag_filter == (linear && mag != sampler_mag_filter::nearest ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest), "Magnification/convolution policy differs from VK");
								require(key.mip_filter == (!mipmapped ? MTLSamplerMipFilterNotMipmapped : trilinear ? MTLSamplerMipFilterLinear
																													: MTLSamplerMipFilterNearest),
										"Mipmap/actual-upload policy differs");
								require(key.min_lod == (mipmapped ? std::min(1.25f, float(mips[1] - 1)) : 0) && key.max_lod == (mipmapped ? float(mips[1] - 1) : 0) && key.bias == (mipmapped ? trilinear ? state.bias : 0.5f : 0), "LOD clamp or half-step bias policy differs");
								auto descriptor = make_sampler_descriptor(key);
								require(descriptor.supportArgumentBuffers && descriptor.maxAnisotropy == anisotropy && descriptor.lodBias == 0 && descriptor.lodMinClamp == key.min_lod && descriptor.lodMaxClamp == key.max_lod, "Native sampler descriptor lost resolved state");
								++cases;
							}
			const std::array wraps{sampler_wrap::repeat, sampler_wrap::mirror, sampler_wrap::edge, sampler_wrap::border, sampler_wrap::clamp, sampler_wrap::mirror_once_edge, sampler_wrap::mirror_once_border, sampler_wrap::mirror_once_clamp};
			const std::array native{MTLSamplerAddressModeRepeat, MTLSamplerAddressModeMirrorRepeat, MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeClampToBorderColor, MTLSamplerAddressModeClampToEdge, MTLSamplerAddressModeMirrorClampToEdge, MTLSamplerAddressModeMirrorClampToEdge, MTLSamplerAddressModeMirrorClampToEdge};
			for (unsigned w = 0; w < wraps.size(); ++w)
				for (unsigned axis = 0; axis < 3; ++axis)
				{
					guest_sampler_state state;
					state.wrap[axis] = wraps[w];
					require(resolve_sampler(state).wrap[axis] == native[w], "Wrap/mirror-once policy differs from VK");
					++cases;
				}
			const std::array comparisons{MTLCompareFunctionNever, MTLCompareFunctionGreater, MTLCompareFunctionEqual, MTLCompareFunctionGreaterEqual, MTLCompareFunctionLess, MTLCompareFunctionNotEqual, MTLCompareFunctionLessEqual, MTLCompareFunctionAlways};
			for (unsigned i = 0; i < 8; ++i)
			{
				guest_sampler_state state;
				state.depth_compare = true;
				state.compare_function = 0x200 + i;
				require(resolve_sampler(state).compare == comparisons[i], "Depth sampler comparison direction not reversed");
				++cases;
			}
			for (const float bias : {-0.76f, -0.75f, -0.74f, -0.26f, -0.25f, -0.24f, 0.24f, 0.25f, 0.26f})
			{
				guest_sampler_state state;
				state.min_filter = sampler_min_filter::nearest_nearest;
				state.bias = bias;
				state.max_lod = 4;
				sampler_texture_policy policy;
				policy.requested_mips = policy.actual_mips = 5;
				require(resolve_sampler(state, policy).bias == std::floor(bias * 2 + 0.5f) * 0.5f, "Signed nearest-mip bias ties differ");
				++cases;
			}
			guest_sampler_state vertex;
			vertex.normalized = false;
			vertex.wrap = {sampler_wrap::edge, sampler_wrap::clamp, sampler_wrap::border};
			vertex.border_color = {0.2, 0.3, 0.4, 0.5};
			vertex.depth_compare = true;
			vertex.compare_function = 0x201;
			vertex.min_filter = sampler_min_filter::linear_linear;
			vertex.mag_filter = sampler_mag_filter::linear;
			vertex.anisotropy = 16;
			vertex.bias = 4;
			sampler_texture_policy vp;
			vp.stage = sampler_stage::vertex;
			const auto vkey = resolve_sampler(vertex, vp);
			require(!vkey.normalized && vkey.wrap[2] == MTLSamplerAddressModeRepeat && vkey.min_filter == MTLSamplerMinMagFilterNearest && vkey.mag_filter == MTLSamplerMinMagFilterNearest && vkey.anisotropy == 1 && vkey.bias == 0 && vkey.compare == MTLCompareFunctionNever, "Vertex sampler overrides or ignored R border differ");
			(void)make_sampler_descriptor(vkey);
			rejects([&]
					{
						vp.dimension = vertex_texture_dimension::three;
						(void)resolve_sampler(vertex, vp);
					});
			vp.dimension = vertex_texture_dimension::two;
			rejects([&]
					{
						vertex.wrap[0] = sampler_wrap::repeat;
						(void)resolve_sampler(vertex, vp);
					});
			rejects([]
					{
						guest_sampler_state state;
						state.wrap[0] = sampler_wrap::border;
						state.border_color = {0.2, 0.3, 0.4, 0.5};
						(void)resolve_sampler(state);
					});
			for (const float invalid : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
				rejects([&]
						{
							guest_sampler_state state;
							state.min_lod = invalid;
							(void)resolve_sampler(state);
						});
			rejects([]
					{
						guest_sampler_state state;
						state.anisotropy = 3;
						(void)resolve_sampler(state);
					});
			rejects([]
					{
						guest_sampler_state state;
						state.min_filter = static_cast<sampler_min_filter>(99);
						(void)resolve_sampler(state);
					});
			rejects([]
					{
						guest_sampler_state state;
						sampler_texture_policy policy;
						policy.actual_mips = 0;
						(void)resolve_sampler(state, policy);
					});
			rejects([]
					{
						auto key = resolve_sampler({});
						key.anisotropy = 0;
						(void)make_sampler_descriptor(key);
					});
			guest_sampler_state shadow;
			shadow.depth_compare = true;
			shadow.depth_float = true;
			const auto shadow_key = resolve_sampler(shadow);
			require(shadow_key.depth_compare && shadow_key.depth_float, "Shadow sampler lost integer/float reference policy");
			sampler_texture_policy vertex_policy;
			vertex_policy.stage = sampler_stage::vertex;
			const auto ordinary_key = resolve_sampler(shadow, vertex_policy);
			require(!ordinary_key.depth_compare && !ordinary_key.depth_float, "Vertex sampler leaked fragment shadow metadata");
			sampler_parameters parameters;
			parameters.flags = sampler_depth_compare | sampler_depth_float;
			validate_sampler_parameters(parameters);
			rejects([]
					{
						sampler_parameters parameters;
						parameters.flags = sampler_depth_float;
						validate_sampler_parameters(parameters);
					});
			rejects([]
					{
						sampler_parameters parameters;
						parameters.flags = 16;
						validate_sampler_parameters(parameters);
					});
			validate_texture_coordinates({});
			const auto coordinate = normalize_guest_pixels(4, 8, 2);
			require(coordinate.scale == std::array<float, 3>{.25f, .125f, .5f}, "Logical guest pixel normalization differs");
			rejects([]
					{
						(void)normalize_guest_pixels(0);
					});
			rejects([]
					{
						texture_coordinates value;
						value.scale[0] = std::numeric_limits<float>::quiet_NaN();
						validate_texture_coordinates(value);
					});
			rejects([]
					{
						texture_coordinates value;
						value.flags = 2;
						validate_texture_coordinates(value);
					});
			rejects([]
					{
						texture_coordinates value;
						value.minimum[0] = 1;
						value.maximum[0] = 0;
						validate_texture_coordinates(value);
					});
			validate_texel_processing({});
			rejects([]
					{
						texel_processing value;
						value.sign_mask = 16;
						validate_texel_processing(value);
					});
			rejects([]
					{
						texel_processing value;
						value.control = 16;
						validate_texel_processing(value);
					});
			rejects([]
					{
						texel_processing value;
						value.reserved[1] = 1;
						validate_texel_processing(value);
					});
			std::cout << cases << " RSX sampler policy/descriptor cases, reversed comparisons, vertex overrides and explicit unsupported/invalid-state rejection passed\n";
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
