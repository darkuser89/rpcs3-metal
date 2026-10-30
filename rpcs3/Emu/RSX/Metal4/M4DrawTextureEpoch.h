#pragma once

#include "M4FragmentTextureBatch.h"
#include "M4ProgramSamplerPlan.h"
#include "M4VertexTextureState.h"

#include <functional>

namespace metal4
{
[[nodiscard]] inline bool texture_words_match(const vertex_texture_snapshot& a, const fragment_texture_snapshot& b,
	const vertex_texture_snapshot& c, const fragment_texture_snapshot& d)
{
	if (a.referenced_mask() != c.referenced_mask() || b.referenced_mask() != d.referenced_mask()) return false;
	for (unsigned unit = 0; unit < 4; ++unit)
		if ((a.referenced_mask() & (1u << unit)) && a.at(unit) != c.at(unit)) return false;
	for (unsigned unit = 0; unit < 16; ++unit)
		if ((b.referenced_mask() & (1u << unit)) && b.at(unit) != d.at(unit)) return false;
	return true;
}
[[nodiscard]] inline bool needs_null_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment)
{
	for (unsigned unit = 0; unit < 4; ++unit)
		if ((vertex.referenced_mask() & (1u << unit)) && !(vertex.at(unit).control0 >> 31))
			return true;
	for (unsigned unit = 0; unit < 16; ++unit)
		if ((fragment.referenced_mask() & (1u << unit)) && !(fragment.at(unit).control0 >> 31))
			return true;
	return false;
}
// Immutable native images and metadata resolved for exactly these register
// snapshots. Copying retains Metal images under ARC. Callers freeze inputs
// under producer exclusion and validate VM receipts separately;this is no lock.
class draw_texture_epoch final
{
	id<MTLDevice> m_device = nil;
	vertex_texture_snapshot m_vertex;
	fragment_texture_snapshot m_fragment;
	std::array<resolved_fragment_texture, 4> m_vertex_images{};
	std::array<resolved_fragment_texture, 16> m_fragment_images{};
	null_texture_family m_null_images{};

  public:
	template <typename Resolve>
	[[nodiscard]] static draw_texture_epoch capture(context& ctx, const vertex_texture_snapshot& vertex,
													const fragment_texture_snapshot& fragment, Resolve&& resolve, const null_texture_family* nulls = nullptr)
	{
		return capture(ctx.device(), vertex, fragment, std::forward<Resolve>(resolve), nulls);
	}
	template <typename Resolve>
	[[nodiscard]] static draw_texture_epoch capture(id<MTLDevice> device, const vertex_texture_snapshot& vertex,
		const fragment_texture_snapshot& fragment, Resolve&& resolve, const null_texture_family* nulls = nullptr)
	{
		draw_texture_epoch epoch;
		epoch.m_device = device;
		epoch.m_vertex = vertex;
		epoch.m_fragment = fragment;
		if (needs_null_textures(vertex, fragment))
		{
			for (unsigned dimension = 0; dimension < 4; ++dimension)
			{
				const auto image = null_texture_image(nulls, static_cast<vertex_texture_dimension>(dimension));
				if (image.device != epoch.m_device)
					throw std::invalid_argument("Null texture epoch belongs to another device");
			}
			epoch.m_null_images = *nulls;
		}
		const auto checked = [&](bool is_vertex, unsigned unit, const fragment_texture_registers& words)
		{
			if (!(words.control0 >> 31))
				return epoch.m_null_images[static_cast<unsigned>(vertex_texture_dimension::two)];
			auto image = std::invoke(resolve, is_vertex, unit, words);
			if (!image.texture || image.texture.device != epoch.m_device)
				throw std::invalid_argument("Texture epoch image is absent or belongs to another device");
			(void)program_sampler_plan::capture(words, image.metadata);
			return image;
		};
		for (unsigned unit = 0; unit < 4; ++unit)
			if (vertex.referenced_mask() & (1u << unit))
				epoch.m_vertex_images[unit] = checked(true, unit, vertex.at(unit).upload_words());
		for (unsigned unit = 0; unit < 16; ++unit)
			if (fragment.referenced_mask() & (1u << unit))
				epoch.m_fragment_images[unit] = checked(false, unit, fragment.at(unit));
		return epoch;
	}
	[[nodiscard]] const vertex_texture_snapshot& vertex_words() const noexcept { return m_vertex; }
	[[nodiscard]] const fragment_texture_snapshot& fragment_words() const noexcept { return m_fragment; }
	[[nodiscard]] const null_texture_family& null_images() const noexcept { return m_null_images; }
	[[nodiscard]] const resolved_fragment_texture& vertex(unsigned unit) const
	{
		(void)m_vertex.at(unit);
		return m_vertex_images[unit];
	}
	[[nodiscard]] const resolved_fragment_texture& fragment(unsigned unit) const
	{
		(void)m_fragment.at(unit);
		return m_fragment_images[unit];
	}
	void validate(context& ctx, const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment) const
	{
		if (ctx.device() != m_device)
			throw std::invalid_argument("Native texture epoch belongs to another device");
		validate_words(vertex, fragment);
	}
	void validate_words(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment) const
	{
		if (!matches_words(vertex, fragment))
			throw std::invalid_argument("Native texture epoch masks/registers differ from the draw");
	}
	[[nodiscard]] bool matches_words(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment) const
	{
		return texture_words_match(vertex, fragment, m_vertex, m_fragment);
	}
};
} // namespace metal4
