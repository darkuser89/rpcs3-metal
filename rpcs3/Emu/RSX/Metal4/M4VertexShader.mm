#include "M4VertexShader.h"
#include "M4VertexData.h"
#include "M4Sampling.h"
#include "M4VertexFetch.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <limits>
#include <algorithm>
#include <stdexcept>

namespace metal4
{
vertex_shader::vertex_shader(const compiler& compiler,
							 std::span<const std::array<std::uint32_t, 4>> instructions, const vertex_program_options& options)
	: vertex_shader(compiler, generate_vertex_program(instructions, options), options)
{
}

vertex_shader::vertex_shader(const compiler& compiler, const generated_vertex_program& program, const vertex_program_options& options, id<MTLLibrary> prepared)
	: m_dimensions(options.textures), m_border_mask(options.border_textures), m_packed_inputs(options.packed_inputs),m_flat_result_inputs(options.flat_result_inputs)
{
	m_texture_mask = program.texture_mask;
	m_library = prepared ? prepared : compiler.library(program.source, @"RPCS3 native RSX vertex program");
	if (m_library.device != compiler.device() || ![m_library.functionNames containsObject:@"rpcs3_rsx_vertex"])
		throw std::invalid_argument("Native prepared vertex library has a different device or entry");
	if (m_flat_result_inputs)
	{
		auto descriptor=[MTL4ComputePipelineDescriptor new];
		descriptor.computeFunctionDescriptor=library_function(m_library,@"rpcs3_rsx_flat_colors");
		m_flat_pipeline=compiler.compute(descriptor);
	}
}

MTL4LibraryFunctionDescriptor* vertex_shader::function() const
{
	return library_function(m_library, @"rpcs3_rsx_vertex");
}

bindings vertex_shader::prepare(context& ctx, vertex_context state,
								const converted_geometry& geometry, const vertex_inputs& inputs, prepared_flat_colors* flat) const
{
	if (m_flat_result_inputs != bool(flat)) throw std::invalid_argument("Native flat result binding contract differs from shader ABI");
	if (flat)
	{
		flat->validate(ctx,geometry);
		if (ctx.current_render_encoder()) throw std::logic_error("Native flat color prepass must precede a render facade");
	}
	if (m_library.device != ctx.device() ||
		(inputs.attributes && inputs.attributes.device != ctx.device()) ||
		(inputs.constants && inputs.constants.device != ctx.device()) ||
		geometry.vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
		m_packed_inputs != inputs.packed.has_value() ||
		!(m_packed_inputs ? valid_vertex_context(state,inputs.constants.length,inputs.constant_offset) :
			valid_vertex_data(state, inputs.attributes.length, inputs.attribute_offset, inputs.constants.length, inputs.constant_offset)))
	{
		throw std::invalid_argument("Metal 4 RSX vertex buffers, offsets or instruction budget are invalid");
	}
	if (inputs.packed)
	{
		const auto& packed = *inputs.packed;
		if (inputs.attributes || inputs.attribute_offset || state.vertex_base != packed.request.first_vertex || state.vertex_count != packed.request.vertex_count)
			throw std::invalid_argument("Packed vertex input range or compiled binding contract differs");
		validate_vertex_fetch(ctx,packed.request,packed.persistent,packed.transient);
	}
	// Don't trust cached source_range: public converted_geometry values may be
	// assembled by callers other than convert_geometry(). Check every fetch.
	for (const auto& reference : geometry.vertices)
	{
		if (!valid_vertex_reference(state, reference.vertex) || !valid_vertex_reference(state, reference.flat_vertex))
		{
			throw std::out_of_range("Metal 4 RSX vertex reference is outside the decoded attribute buffer");
		}
		if (inputs.packed && inputs.packed->source_vertices)
		{
			const auto& ids = *inputs.packed->source_vertices;
			if (!std::binary_search(ids.begin(),ids.end(),reference.vertex) || !std::binary_search(ids.begin(),ids.end(),reference.flat_vertex))
				throw std::out_of_range("Native packed vertex reference was not captured");
		}
	}
	bindings result;
	for (std::size_t i = 0; i < m_dimensions.size(); ++i)
	{
		if (!(m_texture_mask & (1u << i)))
			continue;
		const auto texture = inputs.textures[i];
		const auto sampler = inputs.samplers[i];
		validate_sampled_texture(ctx, texture, sampler, m_dimensions[i], inputs.sampling[i]);
		if (bool(m_border_mask & (1u << i)) != bool(inputs.sampling[i].flags & sampler_custom_border))
			throw std::invalid_argument("Compiled vertex border coverage contract differs from bindings");
		if (m_border_mask & (1u << i))
		{
			validate_border_coverage(ctx, texture, inputs.border_coverage[i], inputs.sampling[i]);
			result.set_texture(4 + i, inputs.border_coverage[i]);
		}
		result.set_texture(i, texture);
		result.set_sampler(i, sampler);
	}
	// Even unused pointer arguments receive a valid allocation. A zero register
	// count still prevents shader reads; the dummy doesn't extend its bounds.
	const std::array<std::byte, 16> zero{};
	const bool missing = (inputs.packed ? (!inputs.packed->persistent || !inputs.packed->transient || !inputs.constants) : (!inputs.attributes || !inputs.constants)) || (flat && flat->source_ids().empty());
	const auto dummy = missing ? ctx.upload(zero, 16) : upload_slice{};
	if (inputs.packed)
	{
		const auto& p = *inputs.packed;
		result.set_buffer(0,p.persistent ? p.persistent : dummy.buffer,p.persistent ? 0 : dummy.offset);
		result.set_buffer(4,p.transient ? p.transient : dummy.buffer,p.transient ? 0 : dummy.offset);
		const auto& r = p.request;
		const vertex_fetch_parameters parameters{r.attributes,p.persistent.length,p.transient.length,r.first_vertex,r.vertex_count,r.vertex_base_index,r.vertex_index_offset,0,0,r.attribute_mask,0,r.element_counts};
		const auto uniforms = ctx.upload(std::as_bytes(std::span{&parameters,1}),256);
		result.set_buffer(5,uniforms.buffer,uniforms.offset);
	}
	else
		result.set_buffer(0, inputs.attributes ? inputs.attributes : dummy.buffer, inputs.attributes ? inputs.attribute_offset : dummy.offset);
	result.set_buffer(1, inputs.constants ? inputs.constants : dummy.buffer,
					  inputs.constants ? inputs.constant_offset : dummy.offset);
	state.geometry_count = static_cast<std::uint32_t>(geometry.vertices.size());
	state.flat_count=flat ? static_cast<std::uint32_t>(flat->source_ids().size()) : 0;
	if (flat)
	{
		result.set_buffer(6,flat->sources() ? flat->sources() : dummy.buffer,flat->sources() ? 0 : dummy.offset);
		result.set_buffer(7,flat->results() ? flat->results() : dummy.buffer,flat->results() ? 0 : dummy.offset);
	}
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&state, 1}), 256);
	result.set_buffer(2, uniforms.buffer, uniforms.offset);
	const auto sampling = ctx.upload(std::as_bytes(std::span(inputs.sampling)), 16);
	result.set_buffer(3, sampling.buffer, sampling.offset);
	if (flat)
	{
		if (state.flat_count && state.predicate)
		{
			const auto table=result.snapshot(ctx);
			flat->consume();
			ctx.end_encoding();
			const auto encoder=ctx.compute(); ctx.keep_alive(m_flat_pipeline);
			[encoder setComputePipelineState:m_flat_pipeline]; [encoder setArgumentTable:table];
			[encoder dispatchThreads:MTLSizeMake(state.flat_count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(state.flat_count,m_flat_pipeline.maxTotalThreadsPerThreadgroup),1,1)];
			ctx.end_encoding();
		}
		else flat->consume();
	}
	return result;
}

std::vector<upload_requirement> vertex_shader::uploads(const vertex_inputs& inputs,const prepared_flat_colors* flat) const
{
	std::vector<upload_requirement> result;
	if ((inputs.packed ? (!inputs.packed->persistent || !inputs.packed->transient || !inputs.constants) : (!inputs.attributes || !inputs.constants)) || (m_flat_result_inputs && (!flat || flat->source_ids().empty()))) result.push_back({16, 16});
	if (inputs.packed) result.push_back({sizeof(vertex_fetch_parameters),256});
	result.push_back({sizeof(vertex_context), 256});
	result.push_back({sizeof(inputs.sampling), 16});
	return result;
}

} // namespace metal4
