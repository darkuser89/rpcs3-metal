#pragma once

#include "M4VertexFetchLayout.h"
#include "M4VertexSparseCapture.h"

#include <bit>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <memory>

namespace metal4
{
using vertex_constant_registers = std::array<std::array<std::uint32_t, 4>, 512>;
struct vertex_constant_statistics
{
	std::uint64_t versions = 0, reused = 0, copied_bytes = 0;
};
// One current immutable CPU version per renderer. Compare all register bits
// under the existing capture epoch; no unverified dirty flag or pointer hint.
// Older packets retain their own versions across clause changes and eviction.
class vertex_constant_cache final
{
	std::shared_ptr<const vertex_constant_registers> m_current;
	vertex_constant_statistics m_statistics;
  public:
	[[nodiscard]] std::shared_ptr<const vertex_constant_registers> capture(std::span<const std::byte> bytes)
	{
		if (bytes.size() != sizeof(vertex_constant_registers)) throw std::invalid_argument("Native vertex constants require all 512 registers");
		if (m_current && std::memcmp(m_current->data(), bytes.data(), bytes.size()) == 0)
		{
			++m_statistics.reused;
			return m_current;
		}
		auto next = std::make_shared<vertex_constant_registers>();
		std::memcpy(next->data(), bytes.data(), bytes.size());
		m_current = std::move(next);
		++m_statistics.versions;
		m_statistics.copied_bytes += bytes.size();
		return m_current;
	}
	[[nodiscard]] vertex_constant_statistics statistics() const noexcept { return m_statistics; }
	void clear() { m_current.reset(); }
};

class vertex_data_snapshot final
{
	vertex_fetch_request m_fetch;
	std::vector<std::byte> m_persistent, m_transient;
	std::shared_ptr<const vertex_constant_registers> m_constants;
	std::shared_ptr<const std::vector<std::uint32_t>> m_source_vertices;
	vertex_sparse_statistics m_sparse_statistics;

  public:
	template <typename State>
	[[nodiscard]] static vertex_data_snapshot capture_sparse(const vertex_sparse_inputs& inputs, const State& state,
		vertex_stream_range range, std::size_t byte_limit, vertex_constant_cache* constants = nullptr)
	{
		static_assert(sizeof(state.transform_constants) == 512 * 16);
		auto packed = capture_sparse_vertex_bytes(inputs,range,byte_limit);
		vertex_data_snapshot result;
		result.m_fetch = packed.fetch;
		result.m_persistent = std::move(packed.bytes);
		result.m_source_vertices = std::move(range.source_vertices);
		result.m_sparse_statistics = packed.statistics;
		vertex_constant_cache local;
		result.m_constants = (constants ? *constants : local).capture(std::as_bytes(std::span(state.transform_constants)));
		return result;
	}
	// Caller holds one analysed layout/register/VM epoch and excludes writes
	// until capture returns. Size calculation and processor must belong to that
	// same epoch. This owns bytes; it does not provide VM locking or DMA barriers.
	template <typename Processor, typename Layout, typename Metadata, typename State>
	[[nodiscard]] static vertex_data_snapshot capture(const Processor& processor, const Layout& layout,
													  const Metadata& metadata, const State& state, vertex_stream_range range,
													  std::pair<std::uint32_t, std::uint32_t> sizes, std::size_t byte_limit, vertex_constant_cache* constants = nullptr)
	{
		static_assert(sizeof(state.transform_constants) == 512 * 16);
		vertex_data_snapshot result;
		result.m_fetch.first_vertex = range.first_vertex;
		result.m_fetch.vertex_count = range.vertex_count;
		result.m_fetch.vertex_base_index = range.vertex_base_index;
		result.m_fetch.vertex_index_offset = range.vertex_index_offset;
		result.m_fetch.attribute_mask = metadata.referenced_inputs_mask;
		if (!vertex_fetch_bytes(result.m_fetch) || (range.vertex_count &&
													range.vertex_count - 1 > std::numeric_limits<std::uint32_t>::max() - range.stream_first))
			throw std::invalid_argument("Native vertex snapshot range or input mask is invalid");
		if (sizes.first > 0x20000000u || sizes.second > 0x20000000u ||
			sizes.first > byte_limit || sizes.second > byte_limit - sizes.first)
			throw std::length_error("Native vertex snapshot exceeds packed offset or capture budget");
		std::array<std::int32_t, 32> descriptors{};
		if (range.vertex_count)
			processor.fill_vertex_layout_state(layout, metadata, range.stream_first, range.vertex_count,
											   descriptors.data(), 0, 0);
		for (unsigned i = 0; i < 16; ++i)
		{
			auto& descriptor = result.m_fetch.attributes[i];
			descriptor = {std::bit_cast<std::uint32_t>(descriptors[2 * i]),
						  std::bit_cast<std::uint32_t>(descriptors[2 * i + 1])};
			if (!(result.m_fetch.attribute_mask & (1u << i)))
				continue;
			const auto extent = vertex_fetch_extent(descriptor, result.m_fetch);
			if (!extent || *extent > (descriptor.transient() ? sizes.second : sizes.first))
				throw std::out_of_range("Native vertex snapshot descriptor exceeds its stream");
		}
		result.m_persistent.resize(sizes.first);
		// The legacy inline writer copies the entire list regardless of count.
		// Reject a cropped allocation before calling it. Productive sparse
		// capture reads inline fields directly and never enters this writer.
		if constexpr (requires { state.current_draw_clause.command; state.current_draw_clause.inline_vertex_array; layout.referenced_registers; })
		{
			if (range.vertex_count && static_cast<unsigned>(state.current_draw_clause.command) == 2 && sizes.second &&
				std::uint64_t(state.current_draw_clause.inline_vertex_array.size()) * 4 + layout.referenced_registers.size() * 16 > sizes.second)
				throw std::length_error("Native legacy inline capture allocation is smaller than the writer");
		}
		result.m_transient.resize(sizes.second);
		if (range.vertex_count)
			processor.write_vertex_data_to_memory(layout, range.stream_first, range.vertex_count,
												  result.m_persistent.empty() ? nullptr : result.m_persistent.data(),
												  result.m_transient.empty() ? nullptr : result.m_transient.data());
		// Native shader constants are uncompressed host-order register words.
		// Copy all 512 registers, including 468..511 omitted by VK's empty-reloc
		// helper. Preserve bits (NaNs and signed zero included), without float math.
		vertex_constant_cache local;
		result.m_constants = (constants ? *constants : local).capture(std::as_bytes(std::span(state.transform_constants)));
		return result;
	}
	[[nodiscard]] const vertex_fetch_request& fetch() const { return m_fetch; }
	[[nodiscard]] std::span<const std::byte> persistent() const { return m_persistent; }
	[[nodiscard]] std::span<const std::byte> transient() const { return m_transient; }
	[[nodiscard]] std::span<const std::array<std::uint32_t, 4>> constants() const
	{
		static const vertex_constant_registers zero{};
		return m_constants ? std::span(*m_constants) : std::span(zero);
	}
	[[nodiscard]] const std::shared_ptr<const vertex_constant_registers>& constant_version() const noexcept { return m_constants; }
	[[nodiscard]] const auto& source_vertices() const noexcept { return m_source_vertices; }
	[[nodiscard]] vertex_sparse_statistics sparse_statistics() const noexcept { return m_sparse_statistics; }
};
} // namespace metal4
