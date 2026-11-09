#pragma once

#include "M4Resources.h"
#include "M4RenderState.h"
#include "M4SamplerState.h"
#include "M4SamplerParameters.h"
#include "M4BorderCoverage.h"
#include "M4Texture.h"

#include <vector>

namespace metal4
{
struct native_sampler_key
{
	std::array<MTLSamplerAddressMode, 3> wrap;
	MTLSamplerMinMagFilter min_filter, mag_filter;
	MTLSamplerMipFilter mip_filter;
	MTLSamplerBorderColor border;
	MTLCompareFunction compare;
	std::uint32_t anisotropy;
	float min_lod, max_lod, bias;
	bool normalized;
	bool depth_compare = false, depth_float = false;
	bool operator==(const native_sampler_key&) const = default;
};

// Uses VK's effective filter/mip/vertex and reversed depth-compare policies.
// Custom borders and unsupported unnormalized sampling need native MSL
// emulation, and are rejected rather than replaced with a nearby color/mode.
[[nodiscard]] native_sampler_key resolve_sampler(const guest_sampler_state& state, const sampler_texture_policy& policy = {});
[[nodiscard]] MTLSamplerDescriptor* make_sampler_descriptor(const native_sampler_key& key);

struct sampler_cache_statistics
{
	std::size_t entries = 0, capacity = 0;
	std::uint64_t created = 0, reused = 0, evicted = 0;
};
struct sampler_binding
{
	id<MTLSamplerState> sampler;
	sampler_parameters parameters;
};
// Bounded render-thread LRU. Eviction drops only the cache's reference;
// argument snapshots keep samplers alive until their GPU job completes.
class sampler_cache final
{
	struct entry
	{
		// Descriptor fields only: shader bias/depth flags are canonical zero.
		native_sampler_key key;
		id<MTLSamplerState> sampler;
		std::uint64_t last_use;
	};
	context& m_context;
	texture_view_cache m_views;
	std::vector<entry> m_entries;
	sampler_cache_statistics m_statistics;
	std::uint64_t m_clock = 0;
	std::shared_ptr<heap_budget> m_budget;
	std::unique_ptr<border_coverage_cache> m_coverage;

  public:
	explicit sampler_cache(context& ctx, std::size_t capacity = 256, std::shared_ptr<heap_budget> budget = {});
	[[nodiscard]] texture_view_cache& texture_views() noexcept { return m_views; }
	[[nodiscard]] id<MTLTexture> border_coverage(id<MTLTexture> source, border_texture_preparation* preparation = nullptr);
	[[nodiscard]] id<MTLTexture> border_shadow_view(id<MTLTexture> source, float value, border_texture_preparation* preparation = nullptr);
	[[nodiscard]] id<MTLTexture> border_color_view(id<MTLTexture> source, std::array<std::uint32_t,4> channels, border_texture_preparation* preparation = nullptr);
	// Always carry parameters into the shader input. Native descriptor bias is
	// zero: tested Apple GPU ignored lodBias, so MSL performs bias explicitly.
	[[nodiscard]] sampler_binding acquire(const guest_sampler_state& state, const sampler_texture_policy& policy = {});
	[[nodiscard]] sampler_cache_statistics statistics() const noexcept;
};
} // namespace metal4
