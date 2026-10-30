#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <memory>
#include <vector>

namespace metal4
{

// IDs retain their original RSX frequency/modulo phase. The interval is only
// a shader bounds envelope; source_vertices is the exact capture set.
struct vertex_stream_range
{
	std::uint32_t stream_first = 0, first_vertex = 0, vertex_count = 0;
	std::uint32_t vertex_base_index = 0;
	std::int32_t vertex_index_offset = 0;
	std::shared_ptr<const std::vector<std::uint32_t>> source_vertices;
};

// The two packed words produced by fill_vertex_layout_state(), also consumed
// by VK's vertex input fetch. Stream bytes stay in their original byte order.
struct vertex_attribute_descriptor
{
	std::uint32_t format = 0;
	std::uint32_t storage = 0;

	[[nodiscard]] constexpr std::uint32_t stride() const { return format & 255u; }
	[[nodiscard]] constexpr std::uint32_t frequency() const { return (format >> 8) & 65535u; }
	[[nodiscard]] constexpr std::uint32_t type() const { return (format >> 24) & 7u; }
	[[nodiscard]] constexpr std::uint32_t components() const { return (format >> 27) & 7u; }
	[[nodiscard]] constexpr std::uint32_t offset() const { return storage & 0x1fffffffu; }
	[[nodiscard]] constexpr bool big_endian() const { return (storage & 0x20000000u) != 0; }
	[[nodiscard]] constexpr bool transient() const { return (storage & 0x40000000u) != 0; }
	[[nodiscard]] constexpr bool modulo() const { return (storage & 0x80000000u) != 0; }
	// 0: ordinary/sized dense records;1: window base followed by records;
	// 2: sorted original-element keys followed by each raw attribute payload.
	[[nodiscard]] constexpr std::uint32_t storage_mode() const { return format >> 30; }
};
static_assert(sizeof(vertex_attribute_descriptor) == 8);

struct vertex_fetch_request
{
	std::array<vertex_attribute_descriptor, 16> attributes{};
	std::uint32_t first_vertex = 0;
	std::uint32_t vertex_count = 0;
	std::uint32_t vertex_base_index = 0;
	std::int32_t vertex_index_offset = 0;
	std::uint32_t attribute_mask = 65535;
	std::array<std::uint32_t,16> element_counts{};
};

// Shared CPU/MSL layout for packed vertex reads and the reference decoder.
struct vertex_fetch_parameters
{
	std::array<vertex_attribute_descriptor, 16> attributes;
	std::uint64_t persistent_length, transient_length;
	std::uint32_t first_vertex, vertex_count, vertex_base_index;
	std::int32_t vertex_index_offset;
	std::uint32_t chunk_base = 0, chunk_count = 0, attribute_mask = 0, reserved = 0;
	std::array<std::uint32_t,16> element_counts{};
};
static_assert(sizeof(vertex_fetch_parameters) == 240 && offsetof(vertex_fetch_parameters, persistent_length) == 128 &&
	offsetof(vertex_fetch_parameters, first_vertex) == 144 && offsetof(vertex_fetch_parameters, chunk_base) == 160);

[[nodiscard]] constexpr std::optional<std::size_t> vertex_fetch_bytes(const vertex_fetch_request& request)
{
	if ((request.attribute_mask & ~65535u) ||
		(request.vertex_count && request.vertex_count - 1 >
			std::numeric_limits<std::uint32_t>::max() - request.first_vertex) ||
		static_cast<std::uint64_t>(request.vertex_count) > std::numeric_limits<std::size_t>::max() / 256)
	{
		return std::nullopt;
	}
	return static_cast<std::size_t>(request.vertex_count) * 256;
}

// Highest byte read, exclusive. Modulo ranges can wrap even when their first
// and last indices are small. Arithmetic deliberately doesn't wrap at 32 bits.
// A null result rejects unsupported formats or invalid source index arithmetic.
[[nodiscard]] constexpr std::optional<std::uint64_t> vertex_fetch_extent(
	const vertex_attribute_descriptor& desc, const vertex_fetch_request& request, std::size_t location = 16)
{
	if (!vertex_fetch_bytes(request)) return std::nullopt;
	if (!request.vertex_count || !desc.components()) return 0;
	constexpr std::array<std::uint32_t, 8> widths{0, 2, 4, 2, 1, 2, 4, 1};
	if (!desc.type() || desc.components() > 4 || (desc.type() == 6 && desc.components() != 1))
		return std::nullopt;
	const auto records = location < 16 ? request.element_counts[location] : 0;
	if (records || desc.storage_mode())
	{
		const auto payload = desc.components() * widths[desc.type()];
		if (!records || desc.storage_mode() > 2 || desc.stride() < payload + (desc.storage_mode() == 2 ? 4u : 0u)) return std::nullopt;
		return desc.offset() + (desc.storage_mode() == 1 ? 4ull : 0ull) + std::uint64_t(records) * desc.stride();
	}
	std::uint64_t largest = 0;
	if (desc.frequency() && desc.stride())
	{
		const auto first = static_cast<std::int64_t>(request.first_vertex);
		const auto last = first + request.vertex_count - 1;
		if (desc.modulo())
		{
			const auto start = first + request.vertex_index_offset;
			const auto finish = last + request.vertex_index_offset;
			if (start < 0 || finish > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;
			const auto remainder = static_cast<std::uint64_t>(start) % desc.frequency();
			largest = remainder + request.vertex_count - 1;
			if (largest >= desc.frequency()) largest = desc.frequency() - 1;
		}
		else
		{
			if (first < request.vertex_base_index) return std::nullopt;
			largest = static_cast<std::uint64_t>(last - request.vertex_base_index) / desc.frequency();
		}
	}
	return desc.offset() + largest * desc.stride() + desc.components() * widths[desc.type()];
}

}
