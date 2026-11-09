#include "M4VertexFetch.h"
#include "M4VertexFetchKernels.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <stdexcept>

namespace metal4
{
namespace
{
using fetch_parameters = vertex_fetch_parameters;
constexpr std::uint32_t max_chunk_vertices = 1u << 26;
constexpr auto decoded_vertex_bytes = *vertex_fetch_bytes(vertex_fetch_request{.vertex_count = 1});
}

vertex_fetch_engine::vertex_fetch_engine(context& ctx) : m_device(ctx.device())
{
	helper_catalog compiler(ctx);
	const auto library = compiler.library(vertex_fetch_kernels, @"RPCS3 native guest vertex fetch");

	m_pipeline = compiler.compute(library, @"rpcs3_fetch_vertices");
}

void validate_vertex_fetch(context& ctx, const vertex_fetch_request& request,
	id<MTLBuffer> persistent, id<MTLBuffer> transient)
{
	const auto bytes = vertex_fetch_bytes(request);
	if ((persistent && persistent.device != ctx.device()) ||
		(transient && transient.device != ctx.device()) || !bytes)
		throw std::invalid_argument("Metal 4 vertex fetch device, count or attribute mask is invalid");
	for (std::size_t i = 0; i < request.attributes.size(); ++i)
	{
		if (!(request.attribute_mask & (1u << i))) continue;
		const auto& desc = request.attributes[i];
		const auto extent = vertex_fetch_extent(desc, request,i);
		const auto length = desc.transient() ? transient.length : persistent.length;
		if (!extent || *extent > length)
			throw std::out_of_range("Metal 4 guest vertex format, index or stream byte range is invalid");
	}
}

vertex_fetch_engine::prepared_fetch vertex_fetch_engine::prepare(context& ctx, const vertex_fetch_request& request,
	id<MTLBuffer> persistent, id<MTLBuffer> transient) const
{
	if (ctx.device() != m_device) throw std::invalid_argument("Metal 4 vertex fetch belongs to another device");
	validate_vertex_fetch(ctx,request,persistent,transient);
	const auto bytes = *vertex_fetch_bytes(request);
	const auto output = bytes ? make_buffer(ctx, bytes, buffer_storage::private_gpu) : nil;
	return {request, persistent, transient, output, m_device};
}

id<MTLBuffer> vertex_fetch_engine::decode(context& ctx, const vertex_fetch_request& request,
	id<MTLBuffer> persistent, id<MTLBuffer> transient) const
{
	return decode(ctx, prepare(ctx, request, persistent, transient));
}

std::vector<upload_requirement> vertex_fetch_engine::uploads(const prepared_fetch& prepared) const
{
	if (prepared.device != m_device) throw std::invalid_argument("Foreign prepared vertex fetch uploads");
	std::vector<upload_requirement> result;
	if (!prepared.output) return result;
	if (!prepared.persistent || !prepared.transient) result.push_back({16, 16});
	for (std::uint64_t first = 0; first < prepared.request.vertex_count; first += max_chunk_vertices)
		result.push_back({sizeof(fetch_parameters), 256});
	return result;
}

id<MTLBuffer> vertex_fetch_engine::decode(context& ctx, const prepared_fetch& prepared) const
{
	if (ctx.device() != m_device || prepared.device != m_device)
		throw std::invalid_argument("Metal 4 prepared vertex fetch belongs to another device");
	if (!prepared.output) return nil;
	const auto& request = prepared.request;
	const auto persistent = prepared.persistent, transient = prepared.transient, output = prepared.output;
	const std::array<std::byte, 16> zero{};
	const auto dummy = (!persistent || !transient) ? ctx.upload(zero, 16) : upload_slice{};
	bindings resources;
	resources.set_buffer(0, persistent ? persistent : dummy.buffer, persistent ? 0 : dummy.offset);
	resources.set_buffer(1, transient ? transient : dummy.buffer, transient ? 0 : dummy.offset);
	resources.set_buffer(2, output);
	for (std::uint32_t first = 0; first < request.vertex_count;)
	{
		const auto count = std::min(request.vertex_count - first, max_chunk_vertices);
		const fetch_parameters params{request.attributes, persistent.length, transient.length,
			request.first_vertex, request.vertex_count, request.vertex_base_index,
			request.vertex_index_offset, first, count, request.attribute_mask,0,request.element_counts};
		const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
		resources.set_buffer(3, uniforms.buffer, uniforms.offset);
		std::array<context::accessed_resource, 4> accesses{{
			{output, first * decoded_vertex_bytes, count * decoded_vertex_bytes, true},
			{uniforms.buffer, uniforms.offset, uniforms.size}}};
		std::size_t used = 2;
		if (params.persistent_length)
			accesses[used++] = {persistent, 0, params.persistent_length};
		if (params.transient_length)
			accesses[used++] = {transient, 0, params.transient_length};
		ctx.checked_dispatch(m_pipeline, resources, std::span{accesses}.first(used), static_cast<NSUInteger>(count) * 16);
		first += count;
	}
	return output;
}
}
