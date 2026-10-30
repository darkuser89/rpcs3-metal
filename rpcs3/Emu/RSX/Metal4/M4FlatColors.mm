#include "M4FlatColors.h"
#include "M4ConstantUpload.h"
#include <algorithm>
#include <stdexcept>

namespace metal4
{
struct prepared_flat_colors::state
{
	id<MTLDevice> device;
	std::vector<std::uint32_t> ids;
	id<MTLBuffer> sources = nil, results = nil;
	bool consumed = false;
};
prepared_flat_colors::prepared_flat_colors(context& ctx,const converted_geometry& geometry,std::size_t byte_limit)
	: m_state(std::make_unique<state>())
{
	m_state->device=ctx.device(); m_state->ids=plan_flat_color_sources(geometry.vertices);
	const auto bytes=flat_color_bytes(m_state->ids.size(),std::min<std::size_t>(byte_limit,0x20000000u));
	if (!bytes) throw std::length_error("Native flat color result/source bytes exceed admission budget");
	if (*bytes)
	{
		m_state->sources=ctx.geometry_uploads().packed_bytes(std::as_bytes(std::span(m_state->ids)));
		m_state->results=make_buffer(ctx,*bytes,buffer_storage::private_gpu);
	}
}
prepared_flat_colors::~prepared_flat_colors()=default;
prepared_flat_colors::prepared_flat_colors(prepared_flat_colors&&) noexcept=default;
prepared_flat_colors& prepared_flat_colors::operator=(prepared_flat_colors&&) noexcept=default;
prepared_flat_colors prepared_flat_colors::admit(context& ctx,const converted_geometry& geometry,std::size_t byte_limit)
{
	return prepared_flat_colors(ctx,geometry,byte_limit);
}
void prepared_flat_colors::validate(context& ctx,const converted_geometry& geometry) const
{
	if (!m_state || m_state->device!=ctx.device() || m_state->consumed)
		throw std::invalid_argument("Native flat color preparation is foreign, moved or already consumed");
	if (m_state->ids!=plan_flat_color_sources(geometry.vertices))
		throw std::invalid_argument("Native flat color preparation source set differs from draw geometry");
}
void prepared_flat_colors::consume() { m_state->consumed=true; }
std::span<const std::uint32_t> prepared_flat_colors::source_ids() const { return m_state ? std::span<const std::uint32_t>(m_state->ids) : std::span<const std::uint32_t>{}; }
id<MTLBuffer> prepared_flat_colors::sources() const { return m_state ? m_state->sources : nil; }
id<MTLBuffer> prepared_flat_colors::results() const { return m_state ? m_state->results : nil; }
} // namespace metal4
