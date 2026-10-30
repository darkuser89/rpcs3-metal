#pragma once

#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4VertexFetchLayout.h"
#include <vector>

namespace metal4
{

// CPU-only bounds/device validation shared by direct vertex reads and the
// reference compute decoder. Does not allocate, encode or wait.
void validate_vertex_fetch(context& ctx, const vertex_fetch_request& request, id<MTLBuffer> persistent, id<MTLBuffer> transient);

// Converts the two packed guest streams directly on the GPU into the float4[16]
// records used by vertex_shader::prepare(). Call after begin(), before opening
// the render pass. Results and source resources survive until GPU completion.
// Frequencies operate on vertex IDs, as in VK. They are not instance divisors;
// per-instance constants and guest-memory coherence remain renderer contracts.
class vertex_fetch_engine final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;

public:
	explicit vertex_fetch_engine(context& ctx);
	// Immutable admission result. Owns both source streams and the fresh output;
	// preparing it allocates/validates only and never begins or encodes GPU work.
	class prepared_fetch final
	{
		friend class vertex_fetch_engine;
		vertex_fetch_request request;
		id<MTLBuffer> persistent, transient, output;
		id<MTLDevice> device;
		prepared_fetch(const vertex_fetch_request& request, id<MTLBuffer> persistent,
			id<MTLBuffer> transient, id<MTLBuffer> output, id<MTLDevice> device)
			: request(request), persistent(persistent), transient(transient), output(output), device(device) {}
	  public:
		[[nodiscard]] id<MTLBuffer> buffer() const noexcept { return output; }
	};
	[[nodiscard]] prepared_fetch prepare(context& ctx, const vertex_fetch_request& request,
		id<MTLBuffer> persistent, id<MTLBuffer> transient) const;
	[[nodiscard]] id<MTLBuffer> decode(context& ctx, const prepared_fetch& prepared) const;
	[[nodiscard]] std::vector<upload_requirement> uploads(const prepared_fetch& prepared) const;
	// Nonempty streams require buffers on this device. Nil is allowed for an
	// unused stream. All offsets are descriptor offsets relative to each buffer.
	// A zero vertex count returns nil and doesn't encode a dispatch.
	[[nodiscard]] id<MTLBuffer> decode(context& ctx, const vertex_fetch_request& request,
		id<MTLBuffer> persistent, id<MTLBuffer> transient) const;
};

}
