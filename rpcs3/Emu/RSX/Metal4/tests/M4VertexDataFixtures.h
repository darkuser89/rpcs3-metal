#pragma once

#include "M4VertexDataSnapshot.h"
#include <algorithm>

namespace metal4::test
{
struct vertex_data_state
{
	std::array<std::uint32_t[4], 512> transform_constants{};
};
struct vertex_data_metadata
{
	std::uint32_t referenced_inputs_mask = 1;
};
struct vertex_data_processor
{
	std::array<vertex_attribute_descriptor, 16> descriptors{};
	std::vector<std::byte> persistent, transient;
	mutable unsigned layout_calls = 0, write_calls = 0;
	mutable std::uint32_t captured_first = 0, captured_count = 0;
	void fill_vertex_layout_state(int, const vertex_data_metadata&, std::uint32_t first, std::uint32_t count,
								  std::int32_t* words, std::uint32_t persistent_base, std::uint32_t transient_base) const
	{
		if (persistent_base || transient_base)
			throw std::logic_error("Fixture expects independent zero-based uploads");
		++layout_calls;
		captured_first = first;
		captured_count = count;
		for (unsigned i = 0; i < 16; ++i)
		{
			words[2 * i] = std::bit_cast<std::int32_t>(descriptors[i].format);
			words[2 * i + 1] = std::bit_cast<std::int32_t>(descriptors[i].storage);
		}
	}
	void write_vertex_data_to_memory(int, std::uint32_t first, std::uint32_t count, void* p, void* t) const
	{
		if (first != captured_first || count != captured_count)
			throw std::logic_error("Capture epoch range changed");
		++write_calls;
		if (!persistent.empty())
			std::memcpy(p, persistent.data(), persistent.size());
		if (!transient.empty())
			std::memcpy(t, transient.data(), transient.size());
	}
};
} // namespace metal4::test
