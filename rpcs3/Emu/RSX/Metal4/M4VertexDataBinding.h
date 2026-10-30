#pragma once

#include "M4DrawExecutor.h"
#include "M4VertexDataSnapshot.h"
#include "M4ConstantUpload.h"

namespace metal4
{
// Upload immutable CPU captures before executing the draw. All allocations
// succeed before the request changes; existing geometry and samplers survive.
inline void apply_vertex_data_snapshot(context& ctx, const vertex_data_snapshot& snapshot, draw_request& request)
{
	const auto persistent = ctx.vertex_uploads().packed_bytes(snapshot.persistent());
	const auto transient = ctx.vertex_uploads().packed_bytes(snapshot.transient());
	const auto constants = ctx.constant_uploads().words(std::as_bytes(snapshot.constants()));
	request.fetch = snapshot.fetch();
	request.source_vertices = snapshot.source_vertices();
	request.persistent = persistent;
	request.transient = transient;
	request.vertex_resources.constants = constants;
	request.vertex_resources.constant_offset = 0;
	request.vertex_uniforms.constant_count = static_cast<std::uint32_t>(snapshot.constants().size());
}
} // namespace metal4
