#include "M4Sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace metal4
{
namespace
{
MTLSamplerAddressMode wrap(sampler_wrap value)
{
	switch (value)
	{
	case sampler_wrap::repeat:
		return MTLSamplerAddressModeRepeat;
	case sampler_wrap::mirror:
		return MTLSamplerAddressModeMirrorRepeat;
	case sampler_wrap::edge:
	case sampler_wrap::clamp:
		return MTLSamplerAddressModeClampToEdge;
	case sampler_wrap::border:
		return MTLSamplerAddressModeClampToBorderColor;
	case sampler_wrap::mirror_once_edge:
	case sampler_wrap::mirror_once_border:
	case sampler_wrap::mirror_once_clamp:
		return MTLSamplerAddressModeMirrorClampToEdge;
	}
	throw std::invalid_argument("Unknown RSX sampler wrap state");
}
MTLSamplerBorderColor border(const std::array<float, 4>& value)
{
	if (value == std::array<float, 4>{0, 0, 0, 0})
		return MTLSamplerBorderColorTransparentBlack;
	if (value == std::array<float, 4>{0, 0, 0, 1})
		return MTLSamplerBorderColorOpaqueBlack;
	if (value == std::array<float, 4>{1, 1, 1, 1})
		return MTLSamplerBorderColorOpaqueWhite;
	throw std::invalid_argument("Custom RSX border color requires a native shader sampling emulation path");
}
MTLCompareFunction comparison(std::uint32_t value)
{
	switch (decode_compare(value, true))
	{
	case compare_operation::never:
		return MTLCompareFunctionNever;
	case compare_operation::less:
		return MTLCompareFunctionLess;
	case compare_operation::equal:
		return MTLCompareFunctionEqual;
	case compare_operation::less_equal:
		return MTLCompareFunctionLessEqual;
	case compare_operation::greater:
		return MTLCompareFunctionGreater;
	case compare_operation::not_equal:
		return MTLCompareFunctionNotEqual;
	case compare_operation::greater_equal:
		return MTLCompareFunctionGreaterEqual;
	case compare_operation::always:
		return MTLCompareFunctionAlways;
	}
	throw std::invalid_argument("Unknown RSX sampler comparison state");
}
} // namespace

native_sampler_key resolve_sampler(const guest_sampler_state& state, const sampler_texture_policy& policy)
{
	if ((policy.stage != sampler_stage::fragment && policy.stage != sampler_stage::vertex) ||
		(policy.dimension != vertex_texture_dimension::one && policy.dimension != vertex_texture_dimension::two && policy.dimension != vertex_texture_dimension::three && policy.dimension != vertex_texture_dimension::cube) ||
		!policy.requested_mips || !policy.actual_mips || policy.actual_mips > policy.requested_mips ||
		!std::isfinite(state.min_lod) || !std::isfinite(state.max_lod) || !std::isfinite(state.bias) || state.min_lod < 0 || state.max_lod < state.min_lod)
		throw std::invalid_argument("RSX sampler stage/dimension/mip/LOD state is invalid");
	native_sampler_key key{};
	for (std::size_t i = 0; i < 3; ++i)
		key.wrap[i] = policy.stage == sampler_stage::vertex && i == 2 ? MTLSamplerAddressModeRepeat : wrap(state.wrap[i]);
	key.border = std::find(key.wrap.begin(), key.wrap.end(), MTLSamplerAddressModeClampToBorderColor) != key.wrap.end() ? border(state.border_color) : MTLSamplerBorderColorOpaqueBlack;
	key.normalized = state.normalized;
	key.compare = policy.stage == sampler_stage::fragment && state.depth_compare ? comparison(state.compare_function) : MTLCompareFunctionNever;
	key.depth_compare = policy.stage == sampler_stage::fragment && state.depth_compare;
	key.depth_float = key.depth_compare && state.depth_float;
	key.anisotropy = policy.stage == sampler_stage::vertex ? 1 : state.anisotropy;
	if (key.anisotropy != 1 && key.anisotropy != 2 && key.anisotropy != 4 && key.anisotropy != 6 && key.anisotropy != 8 && key.anisotropy != 10 && key.anisotropy != 12 && key.anisotropy != 16)
		throw std::invalid_argument("Invalid RSX anisotropy factor");
	key.min_filter = MTLSamplerMinMagFilterNearest;
	key.mag_filter = MTLSamplerMinMagFilterNearest;
	key.mip_filter = MTLSamplerMipFilterNotMipmapped;
	if (policy.stage == sampler_stage::vertex)
	{
		key.wrap[2] = MTLSamplerAddressModeRepeat;
		key.anisotropy = 1;
		key.compare = MTLCompareFunctionNever;
		key.min_lod = state.min_lod;
		key.max_lod = state.max_lod;
		key.mip_filter = MTLSamplerMipFilterNearest;
	}
	else
	{
		// Fragment coordinates are normalized in the VK shader path as well.
		key.normalized = true;
		bool mipmaps = false;
		switch (state.min_filter)
		{
		case sampler_min_filter::nearest:
			break;
		case sampler_min_filter::linear:
		case sampler_min_filter::convolution:
			key.min_filter = MTLSamplerMinMagFilterLinear;
			break;
		case sampler_min_filter::nearest_nearest:
			mipmaps = true;
			break;
		case sampler_min_filter::linear_nearest:
			key.min_filter = MTLSamplerMinMagFilterLinear;
			mipmaps = true;
			break;
		case sampler_min_filter::nearest_linear:
			key.mip_filter = MTLSamplerMipFilterLinear;
			mipmaps = true;
			break;
		case sampler_min_filter::linear_linear:
			key.min_filter = MTLSamplerMinMagFilterLinear;
			key.mip_filter = MTLSamplerMipFilterLinear;
			mipmaps = true;
			break;
		default:
			throw std::invalid_argument("Unknown RSX minification filter");
		}
		switch (state.mag_filter)
		{
		case sampler_mag_filter::nearest:
			break;
		case sampler_mag_filter::linear:
		case sampler_mag_filter::convolution:
			key.mag_filter = MTLSamplerMinMagFilterLinear;
			break;
		default:
			throw std::invalid_argument("Unknown RSX magnification filter");
		}
		if (!policy.linear_filtering)
		{
			key.min_filter = key.mag_filter = MTLSamplerMinMagFilterNearest;
			key.mip_filter = MTLSamplerMipFilterNearest;
		}
		if (mipmaps && policy.requested_mips > 1 && policy.actual_mips > 1)
		{
			if (key.mip_filter != MTLSamplerMipFilterLinear)
				key.mip_filter = MTLSamplerMipFilterNearest;
			key.min_lod = std::min(state.min_lod, float(policy.actual_mips - 1));
			key.max_lod = std::min(state.max_lod, float(policy.actual_mips - 1));
			key.bias = key.mip_filter == MTLSamplerMipFilterNearest ? std::floor(state.bias * 2.f + 0.5f) * 0.5f : state.bias;
		}
		else
			key.mip_filter = MTLSamplerMipFilterNotMipmapped;
	}
	if (key.bias < -16.f || key.bias > 15.999f)
		throw std::invalid_argument("Effective RSX LOD bias exceeds native Metal26 S4.6 range; shader emulation required");
	if (!key.normalized)
	{
		if ((policy.dimension != vertex_texture_dimension::one && policy.dimension != vertex_texture_dimension::two) ||
			key.wrap[0] != MTLSamplerAddressModeClampToEdge || key.wrap[1] != MTLSamplerAddressModeClampToEdge || key.min_lod != 0 || key.max_lod != 0)
			throw std::invalid_argument("Unnormalized RSX vertex sampling requires a native coordinate/LOD emulation path");
		key.mip_filter = MTLSamplerMipFilterNotMipmapped;
	}
	return key;
}
MTLSamplerDescriptor* make_sampler_descriptor(const native_sampler_key& key)
{
	if (std::any_of(key.wrap.begin(), key.wrap.end(), [](auto mode)
					{
						return mode > MTLSamplerAddressModeClampToBorderColor;
					}) ||
		key.min_filter > MTLSamplerMinMagFilterLinear || key.mag_filter > MTLSamplerMinMagFilterLinear || key.mip_filter > MTLSamplerMipFilterLinear || key.border > MTLSamplerBorderColorOpaqueWhite || key.compare > MTLCompareFunctionAlways || !key.anisotropy || key.anisotropy > 16 || !std::isfinite(key.min_lod) || !std::isfinite(key.max_lod) || !std::isfinite(key.bias) || key.min_lod < 0 || key.max_lod < key.min_lod || key.bias < -16 || key.bias > 15.999f || (!key.normalized && (key.wrap[0] != MTLSamplerAddressModeClampToEdge || key.wrap[1] != MTLSamplerAddressModeClampToEdge || key.mip_filter != MTLSamplerMipFilterNotMipmapped || key.min_lod != 0 || key.max_lod != 0)))
		throw std::invalid_argument("Invalid resolved native sampler descriptor");
	auto descriptor = [MTLSamplerDescriptor new];
	descriptor.sAddressMode = key.wrap[0];
	descriptor.tAddressMode = key.wrap[1];
	descriptor.rAddressMode = key.wrap[2];
	descriptor.minFilter = key.min_filter;
	descriptor.magFilter = key.mag_filter;
	descriptor.mipFilter = key.mip_filter;
	descriptor.borderColor = key.border;
	descriptor.compareFunction = key.compare;
	descriptor.maxAnisotropy = key.anisotropy;
	descriptor.normalizedCoordinates = key.normalized;
	descriptor.lodMinClamp = key.min_lod;
	descriptor.lodMaxClamp = key.max_lod;
	descriptor.lodBias = 0;
	descriptor.supportArgumentBuffers = YES;
	return descriptor;
}
sampler_cache::sampler_cache(context& ctx, std::size_t capacity, std::shared_ptr<heap_budget> budget)
	: m_context(ctx), m_views(ctx, std::min<std::size_t>(capacity, 4096)), m_budget(std::move(budget))
{
	if (!capacity)
		throw std::invalid_argument("Native sampler cache needs nonzero capacity");
	m_statistics.capacity = capacity;
}
id<MTLTexture> sampler_cache::border_coverage(id<MTLTexture> source)
{
	if (!m_coverage)
	{
		if (!m_budget) m_budget = std::make_shared<heap_budget>(m_context, 128u << 20);
		m_coverage = std::make_unique<border_coverage_cache>(m_context, m_budget, std::min<std::size_t>(m_statistics.capacity, 4096));
	}
	return m_coverage->get(source);
}
id<MTLTexture> sampler_cache::border_color_view(id<MTLTexture> source, std::array<std::uint32_t,4> channels)
{
	(void)border_coverage(source);
	return m_coverage->color_view(source,channels);
}
id<MTLTexture> sampler_cache::border_shadow_view(id<MTLTexture> source, float value)
{
	if (!m_coverage)
	{
		if (!m_budget)
			m_budget = std::make_shared<heap_budget>(m_context, 128u << 20);
		m_coverage = std::make_unique<border_coverage_cache>(m_context, m_budget, std::min<std::size_t>(m_statistics.capacity, 4096));
	}
	return m_coverage->shadow_view(source, value);
}
sampler_binding sampler_cache::acquire(const guest_sampler_state& state, const sampler_texture_policy& policy)
{
	const auto key = resolve_sampler(state, policy);
	std::uint32_t flags = key.border == MTLSamplerBorderColorOpaqueBlack && std::find(key.wrap.begin(), key.wrap.end(), MTLSamplerAddressModeClampToBorderColor) != key.wrap.end() ? sampler_opaque_black_border : 0;
	if (key.depth_compare)
		flags |= sampler_depth_compare;
	if (key.depth_float)
		flags |= sampler_depth_float;
	// These values belong to each shader binding, not the immutable native
	// descriptor. In particular, compareFunction (kept below) is the complete
	// native comparison state; Metal has no separate compare-enable field.
	auto descriptor_key = key;
	descriptor_key.bias = 0;
	descriptor_key.depth_compare = descriptor_key.depth_float = false;
	for (auto& entry : m_entries)
		if (entry.key == descriptor_key)
		{
			entry.last_use = ++m_clock;
			++m_statistics.reused;
			return {entry.sampler, {key.min_lod, key.max_lod, key.bias, flags}};
		}
	const auto sampler = make_sampler(m_context, make_sampler_descriptor(key));
	if (m_entries.size() == m_statistics.capacity)
	{
		auto oldest = std::min_element(m_entries.begin(), m_entries.end(), [](const auto& a, const auto& b)
									   {
										   return a.last_use < b.last_use;
									   });
		*oldest = {descriptor_key, sampler, ++m_clock};
		++m_statistics.evicted;
	}
	else
		m_entries.push_back({descriptor_key, sampler, ++m_clock});
	++m_statistics.created;
	return {sampler, {key.min_lod, key.max_lod, key.bias, flags}};
}
sampler_cache_statistics sampler_cache::statistics() const noexcept
{
	auto result = m_statistics;
	result.entries = m_entries.size();
	return result;
}
} // namespace metal4
