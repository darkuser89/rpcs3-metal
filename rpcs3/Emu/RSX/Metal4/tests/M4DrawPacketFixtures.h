#pragma once

#include "M4DrawEnvironmentFixtures.h"
#include "M4RSXDrawPacket.h"
#include "M4VertexDataFixtures.h"

namespace metal4::test
{
struct pass_clause
{
	unsigned primitive = 5, command = 3, first = 0, count = 3;
	bool is_immediate_draw = false;
	bool is_trivial_instanced_draw = true, rendering = false;
	unsigned pass_count = 1, pass_index = 0;
	std::vector<std::uint32_t> inline_vertex_array;
	std::vector<draw_subrange> ranges;
	bool empty() const { return command == 2 ? inline_vertex_array.empty() : !count; }
	bool is_single_draw() const { return ranges.empty(); }
	const auto& get_subranges() const { return ranges; }
	draw_subrange get_range() const
	{
		if (command == 2 || empty()) throw std::logic_error("Inline fixture has no draw range");
		return {first, count};
	}
	unsigned min_index() const { return first; }
	unsigned get_elements_count() const { return count; }
	void begin()
	{
		pass_index = 0;
		rendering = true;
	}
	void end() { pass_index = pass_count - 1; }
	bool next()
	{
		if (++pass_index >= pass_count)
		{
			pass_index = 0;
			rendering = false;
			return false;
		}
		return true;
	}
	template <typename Context>
	unsigned execute_pipeline_dependencies(Context* context) const
	{
		context->dependencies.push_back(pass_index);
		return pass_index * 8;
	}
	template <typename Context>
	void post_execute_cleanup(Context* context) const { ++context->cleanups; }
};
struct pass_state : environment_state, vertex_data_state
{
	std::vector<std::uint32_t> registers = std::vector<std::uint32_t>(0x2000 / 4);
	pass_clause current_draw_clause;
	unsigned index_width = 0, base_index = 4, restart = 0;
	bool restart_enabled = false;
	auto index_type() const { return index_width; }
	auto vertex_data_base_index() const { return base_index; }
	auto restart_index() const { return restart; }
	auto restart_index_enabled() const { return restart_enabled; }
};
struct pass_metadata : vertex_data_metadata
{
	std::uint32_t referenced_textures_mask = 0;
};
struct pass_block
{
	unsigned attribute_stride = 16;
};
struct pass_layout
{
	std::vector<pass_block*> interleaved_blocks;
};
struct pass_array
{
};
struct pass_inline
{
};
struct pass_indexed
{
	std::span<const std::byte> raw_index_buffer;
};
struct pass_processor : vertex_data_processor
{
	std::vector<std::byte> index_bytes;
	mutable unsigned command_calls = 0;
	unsigned analysis_calls = 0;
	void analyse_inputs_interleaved(pass_layout&, const pass_metadata&) { ++analysis_calls; }
	std::variant<pass_array, pass_inline, pass_indexed> get_draw_command(const pass_state& state) const
	{
		++command_calls;
		if (state.current_draw_clause.command == 3)
			return pass_indexed{index_bytes};
		if (state.current_draw_clause.command == 2)
			return pass_inline{};
		return pass_array{};
	}
	void fill_vertex_layout_state(const pass_layout&, const pass_metadata& metadata, std::uint32_t first, std::uint32_t count,
								  std::int32_t* words, std::uint32_t p, std::uint32_t t) const
	{
		vertex_data_processor::fill_vertex_layout_state(0, metadata, first, count, words, p, t);
	}
	void write_vertex_data_to_memory(const pass_layout&, std::uint32_t first, std::uint32_t count, void* p, void* t) const
	{
		vertex_data_processor::write_vertex_data_to_memory(0, first, count, p, t);
	}
};
} // namespace metal4::test
