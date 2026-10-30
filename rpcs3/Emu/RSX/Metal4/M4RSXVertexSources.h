#pragma once

#include "M4VertexSparseCapture.h"
#include <bit>
#include <type_traits>

namespace metal4
{
// Form a protected member pointer legally inside a derived type, then apply
// it to the actual processor. No cast, downcast or replacement processor.
// The existing processor exposes no public raw vertex push-buffer accessor.
template <typename Processor>
struct vertex_push_source_access : private Processor
{
	static std::span<const std::byte> bytes(const Processor& processor, unsigned unit)
	{
		const auto member = &vertex_push_source_access::m_vertex_push_buffers;
		const auto& data = (processor.*member)[unit].data;
		return std::as_bytes(std::span{data.data(),data.size()});
	}
};

// Analyse only owned layout metadata: cached {0,1} ranges stop the shared
// helper from re-scanning raw index memory or calculating a high-ID prefix.
// All source spans/callbacks expire when synchronous capture returns.
template <typename Processor, typename Layout, typename Metadata, typename State>
[[nodiscard]] vertex_sparse_inputs rsx_vertex_sources(const Processor& processor, const Layout& layout,
	const Metadata& metadata, const State& state, const vertex_stream_range& range)
{
	vertex_sparse_inputs result;
	result.attribute_mask = metadata.referenced_inputs_mask;
	if (!range.source_vertices || range.source_vertices->empty()) return result;
	Layout descriptors;
	descriptors.attribute_mask = layout.attribute_mask;
	descriptors.attribute_placement = layout.attribute_placement;
	for (const auto reg : layout.referenced_registers) descriptors.referenced_registers.push_back(reg);
	for (const auto block : layout.volatile_blocks) descriptors.volatile_blocks.push_back(block);
	for (const auto original : layout.interleaved_blocks)
	{
		if (!original || descriptors.interleaved_blocks.size() >= 16) throw std::invalid_argument("Native source layout block is invalid");
		const auto owned = descriptors.alloc_interleaved_block();
		owned->interleaved = original->interleaved;
		owned->single_vertex = original->single_vertex;
		owned->base_offset = original->base_offset;
		owned->real_offset_address = original->real_offset_address;
		owned->memory_location = original->memory_location;
		owned->attribute_stride = original->attribute_stride;
		owned->vertex_range = {0,1};
		for (const auto location : original->locations) owned->locations.push_back(location);
		descriptors.interleaved_blocks.push_back(owned);
	}
	std::array<std::int32_t,32> words{};
	processor.fill_vertex_layout_state(descriptors,metadata,range.source_vertices->back(),1,words.data(),0,0);
	for (unsigned unit = 0; unit < 16; ++unit)
	{
		auto& source = result.attributes[unit];
		source.descriptor = {std::bit_cast<std::uint32_t>(words[unit * 2]),std::bit_cast<std::uint32_t>(words[unit * 2 + 1])};
		const auto d = source.descriptor;
		if (!(result.attribute_mask & (1u << unit)) || !d.components()) continue;
		if (!d.transient())
		{
			bool found = false;
			for (const auto block : layout.interleaved_blocks) for (const auto location : block->locations)
				if (location.index == unit)
				{
					const auto offset = state.vertex_arrays_info[unit].offset() & 0x7fffffffu;
					if (found || offset < block->base_offset) throw std::invalid_argument("Native persistent vertex source is ambiguous");
					found = true; source.memory = true;
					source.base = std::uint64_t(block->real_offset_address) + offset - block->base_offset;
				}
			if (!found) throw std::invalid_argument("Native persistent vertex source is absent");
		}
		else if (!d.frequency())
		{
			const auto& data = state.register_vertex_info[unit].data;
			source.bytes = std::as_bytes(std::span{data.data(),data.size()});
		}
		else if (static_cast<unsigned>(state.current_draw_clause.command) == 2)
		{
			const auto& data = state.current_draw_clause.inline_vertex_array;
			source.bytes = std::as_bytes(std::span{data.data(),data.size()});
			const auto prefix = std::uint64_t(layout.referenced_registers.size()) * 16;
			if (d.offset() < prefix) throw std::invalid_argument("Native inline vertex offset overlaps register prefix");
			source.base = d.offset() - prefix;
		}
		else
		{
			if constexpr (requires { processor.vertex_source_bytes(unit); }) source.bytes = processor.vertex_source_bytes(unit);
			else source.bytes = vertex_push_source_access<Processor>::bytes(processor,unit);
		}
	}
	result.read_memory = [&processor](std::uint32_t address, std::span<std::byte> output)
	{
		if (output.empty() || output.size() > 4080 || std::uint64_t(address) + output.size() > (1ull << 32))
			throw std::out_of_range("Native vertex DMA read is invalid");
		Layout reads;
		for (std::size_t offset = 0; offset < output.size();)
		{
			const auto block = reads.alloc_interleaved_block();
			block->real_offset_address = address + static_cast<std::uint32_t>(offset);
			block->attribute_stride = static_cast<std::uint8_t>(std::min<std::size_t>(255,output.size() - offset));
			block->vertex_range = {0,1}; reads.interleaved_blocks.push_back(block);
			offset += block->attribute_stride;
		}
		processor.write_vertex_data_to_memory(reads,0,1,output.data(),nullptr);
	};
	return result;
}
} // namespace metal4
