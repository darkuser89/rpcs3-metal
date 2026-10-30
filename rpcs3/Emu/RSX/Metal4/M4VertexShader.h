#pragma once

#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4Geometry.h"
#include "M4VertexProgram.h"
#include "M4Sampler.h"
#include "M4VertexFetchLayout.h"
#include "M4FlatColors.h"
#include <optional>
#include <vector>

namespace metal4
{

struct packed_vertex_inputs
{
	vertex_fetch_request request;
	id<MTLBuffer> persistent = nil, transient = nil;
	std::shared_ptr<const std::vector<std::uint32_t>> source_vertices;
};

struct vertex_inputs
{
	id<MTLBuffer> attributes = nil;
	std::size_t attribute_offset = 0;
	id<MTLBuffer> constants = nil;
	std::size_t constant_offset = 0;
	std::optional<packed_vertex_inputs> packed;
	std::array<id<MTLTexture>, 4> textures{};
	std::array<id<MTLTexture>, 4> border_coverage{};
	// Samplers must be created with argumentBufferSupport, via make_sampler().
	std::array<id<MTLSamplerState>, 4> samplers{};
	std::array<sampler_parameters, 4> sampling{};
	void set_sampler(std::size_t index, const sampler_binding& binding)
	{
		samplers.at(index) = binding.sampler;
		sampling.at(index) = binding.parameters;
	}
};

// Native MTL4Compiler library and stage bindings for a translated RSX vertex
// program. Supply function() to an MTL4RenderPipelineDescriptor. Inputs use the
// compiled packed_inputs ABI or decoded vertex_fetch_engine records.
// shader_cache owns translated program reuse;
// the RSX renderer adapter supplies fresh per-draw resources and uniforms.
class vertex_shader final
{
	friend class shader_cache;
	vertex_shader(const compiler& compiler, const generated_vertex_program& program, const vertex_program_options& options,
		id<MTLLibrary> prepared = nil);
	id<MTLLibrary> m_library;
	std::array<vertex_texture_dimension, 4> m_dimensions;
	std::uint32_t m_texture_mask;
	std::uint32_t m_border_mask;
	bool m_packed_inputs;
	bool m_flat_result_inputs;
	id<MTLComputePipelineState> m_flat_pipeline = nil;

  public:
	vertex_shader(const compiler& compiler, std::span<const std::array<std::uint32_t, 4>> instructions,
				  const vertex_program_options& options = {});
	[[nodiscard]] MTL4LibraryFunctionDescriptor* function() const;
	[[nodiscard]] std::vector<upload_requirement> uploads(const vertex_inputs& inputs, const prepared_flat_colors* flat = nullptr) const;
	// Must run after context.begin(). The returned table omits geometry buffer
	// 30; render_commands::draw_geometry uploads and binds that array itself.
	// State and source references are checked before any GPU command is encoded.
	// flat_result_inputs additionally requires a fresh admitted flat preparation,
	// outside a render facade. Encodes its one native prepass before returning.
	[[nodiscard]] bindings prepare(context& ctx, vertex_context state,
								   const converted_geometry& geometry, const vertex_inputs& inputs, prepared_flat_colors* flat = nullptr) const;
};

} // namespace metal4
