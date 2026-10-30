#pragma once

#include "M4VertexProgram.h"

#include <limits>

namespace metal4
{

// Fixed ABI: sixteen float4 attributes for every source vertex, followed by
// a separate uncompacted float4 constant register file. Division checks avoid
// overflow even when the caller's buffer length approaches SIZE_MAX.
[[nodiscard]] inline bool valid_vertex_context(const vertex_context& state, std::size_t constant_bytes, std::size_t constant_offset)
{
	constexpr auto max_vertex = std::numeric_limits<std::uint32_t>::max();
	return (state.vertex_count == 0 || state.vertex_count - 1 <= max_vertex - state.vertex_base) &&
		state.instruction_budget > 0 && state.instruction_budget <= 65536 &&
		constant_offset % 16 == 0 && constant_offset <= constant_bytes &&
		state.constant_count <= (constant_bytes - constant_offset) / 16;
}

[[nodiscard]] inline bool valid_vertex_data(const vertex_context& state,
	std::size_t attribute_bytes, std::size_t attribute_offset,
	std::size_t constant_bytes, std::size_t constant_offset)
{
	return valid_vertex_context(state,constant_bytes,constant_offset) &&
		attribute_offset % 16 == 0 && attribute_offset <= attribute_bytes &&
		state.vertex_count <= (attribute_bytes - attribute_offset) / 256;
}

[[nodiscard]] inline bool valid_vertex_reference(const vertex_context& state, std::uint32_t vertex)
{
	return vertex >= state.vertex_base && vertex - state.vertex_base < state.vertex_count;
}

}
