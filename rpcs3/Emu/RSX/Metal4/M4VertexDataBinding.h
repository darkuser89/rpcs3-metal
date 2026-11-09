#pragma once

#include "M4DrawExecutor.h"
#include "M4VertexDataSnapshot.h"
#include "M4ConstantUpload.h"

namespace metal4
{
struct vertex_data_upload
{
	packed_vertex_inputs packed;
	id<MTLBuffer> constants = nil;
};

[[nodiscard]] inline vertex_data_upload upload_vertex_data(context& ctx, const vertex_data_snapshot& snapshot)
{
	return {{snapshot.fetch(), ctx.vertex_uploads().packed_bytes(snapshot.persistent()),
		ctx.vertex_uploads().packed_bytes(snapshot.transient()), snapshot.source_vertices()},
		ctx.constant_uploads().words(std::as_bytes(snapshot.constants()))};
}

inline void apply_vertex_data_snapshot(context& ctx, const vertex_data_snapshot& snapshot, draw_request& request)
{
	const auto upload = upload_vertex_data(ctx, snapshot);
	request.fetch = upload.packed.request;
	request.source_vertices = upload.packed.source_vertices;
	request.persistent = upload.packed.persistent;
	request.transient = upload.packed.transient;
	request.vertex_resources.constants = upload.constants;
	request.vertex_resources.constant_offset = 0;
	request.vertex_uniforms.constant_count = static_cast<std::uint32_t>(snapshot.constants().size());
}
} // namespace metal4
