#include "M4GeometryBinding.h"
#include "M4ConstantUpload.h"

namespace metal4
{
prepared_geometry::prepared_geometry(context& ctx, const geometry_index_plan& plan)
	: m_references(plan.references()),m_buffer(ctx.geometry_uploads().packed_bytes(plan.bytes())),m_count(plan.count()),m_index_width(plan.index_width())
{
}
} // namespace metal4
