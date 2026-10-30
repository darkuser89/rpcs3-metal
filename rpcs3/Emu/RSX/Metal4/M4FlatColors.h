#pragma once

#include "M4FlatColorLayout.h"
#include "M4Geometry.h"
#include "M4Resources.h"

namespace metal4
{
class vertex_shader;
// One draw's output allocation. Results never enter a cross-draw cache.
// Admission precedes surface encoding; the single-use guard prevents stale
// constants/textures or overlapping recordings from reusing these results.
class prepared_flat_colors final
{
	friend class vertex_shader;
	struct state;
	std::unique_ptr<state> m_state;
	prepared_flat_colors(context& ctx, const converted_geometry& geometry, std::size_t byte_limit);
	void validate(context& ctx, const converted_geometry& geometry) const;
	void consume();
  public:
	~prepared_flat_colors();
	prepared_flat_colors(prepared_flat_colors&&) noexcept;
	prepared_flat_colors& operator=(prepared_flat_colors&&) noexcept;
	prepared_flat_colors(const prepared_flat_colors&) = delete;
	prepared_flat_colors& operator=(const prepared_flat_colors&) = delete;
	[[nodiscard]] static prepared_flat_colors admit(context& ctx, const converted_geometry& geometry,
		std::size_t byte_limit = 0x20000000u);
	// byte_limit bounds logical result+ID data. Upload padding and actual
	// native backing are independently charged by the context heap ledger.
	[[nodiscard]] std::span<const std::uint32_t> source_ids() const;
	// Read-only handles. Only the owned native compute pass writes results.
	[[nodiscard]] id<MTLBuffer> sources() const;
	[[nodiscard]] id<MTLBuffer> results() const;
};
} // namespace metal4
