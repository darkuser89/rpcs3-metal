#pragma once

#include "M4Geometry.h"
#include "M4Resources.h"

namespace metal4
{
// Immutable native reference/index packet admitted before surface encoding.
// CPU metadata is owned; command residency retains native backing through
// completion independently of packet/cache/context destruction.
// buffer() is a read-only binding handle; never modify its contents.
class prepared_geometry final
{
	converted_geometry m_references;
	id<MTLBuffer> m_buffer = nil;
	std::uint32_t m_count, m_index_width;
	prepared_geometry(context& ctx, const geometry_index_plan& plan);
  public:
	[[nodiscard]] static prepared_geometry prepare(context& ctx, const geometry_index_plan& plan) { return prepared_geometry(ctx,plan); }
	[[nodiscard]] const converted_geometry& references() const noexcept { return m_references; }
	[[nodiscard]] id<MTLBuffer> buffer() const noexcept { return m_buffer; }
	[[nodiscard]] std::uint32_t count() const noexcept { return m_count; }
	[[nodiscard]] std::uint32_t index_width() const noexcept { return m_index_width; }
	[[nodiscard]] std::size_t index_offset() const noexcept { return m_references.vertices.size() * sizeof(vertex_reference); }
	[[nodiscard]] MTLPrimitiveType primitive() const noexcept
	{
		switch (m_references.topology)
		{
		case list_topology::points: return MTLPrimitiveTypePoint;
		case list_topology::lines: return MTLPrimitiveTypeLine;
		default: return MTLPrimitiveTypeTriangle;
		}
	}
};
} // namespace metal4
