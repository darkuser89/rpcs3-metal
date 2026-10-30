#pragma once

#include "M4VertexFetchLayout.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>
#include <span>
#include <stdexcept>

namespace metal4
{
struct vertex_sparse_source
{
	vertex_attribute_descriptor descriptor;
	bool memory = false;
	// Memory: absolute guest address. CPU: offset within the supplied span.
	std::uint64_t base = 0;
	std::span<const std::byte> bytes;
};
struct vertex_sparse_inputs
{
	std::array<vertex_sparse_source,16> attributes{};
	std::uint32_t attribute_mask = 0;
	// Must perform the original RSX DMA read, synchronously in the capture
	// epoch. Never retained by the result or by a submitted GPU command.
	std::function<void(std::uint32_t,std::span<std::byte>)> read_memory;
};
struct vertex_sparse_statistics
{
	std::uint64_t source_vertices = 0, source_records = 0, memory_bytes = 0, memory_calls = 0, cpu_bytes = 0;
	std::uint32_t dense_fields = 0, window_fields = 0, sparse_fields = 0, shared_fields = 0;
};
struct vertex_sparse_bytes
{
	vertex_fetch_request fetch;
	std::vector<std::byte> bytes;
	vertex_sparse_statistics statistics;
};

// Capture raw fields, not decoded float4 records. Per-field quotient/modulo
// remains in the native vertex stage; only their required element keys are
// planned here. All addresses and allocations are checked before any read.
[[nodiscard]] inline vertex_sparse_bytes capture_sparse_vertex_bytes(const vertex_sparse_inputs& inputs,
	const vertex_stream_range& range, std::size_t byte_limit)
{
	vertex_sparse_bytes result;
	result.fetch.first_vertex = range.first_vertex;
	result.fetch.vertex_count = range.vertex_count;
	result.fetch.vertex_base_index = range.vertex_base_index;
	result.fetch.vertex_index_offset = range.vertex_index_offset;
	result.fetch.attribute_mask = inputs.attribute_mask;
	if (!vertex_fetch_bytes(result.fetch) || !range.source_vertices ||
		!std::is_sorted(range.source_vertices->begin(),range.source_vertices->end()) ||
		std::adjacent_find(range.source_vertices->begin(),range.source_vertices->end()) != range.source_vertices->end())
		throw std::invalid_argument("Native sparse vertex source set is invalid");
	for (const auto id : *range.source_vertices)
		if (id < range.first_vertex || std::uint64_t(id) - range.first_vertex >= range.vertex_count)
			throw std::out_of_range("Native sparse source is outside its shader envelope");
	result.statistics.source_vertices = range.source_vertices->size();
	struct field
	{
		vertex_sparse_source source;
		std::vector<std::uint32_t> keys;
		std::size_t output = 0;
		std::uint32_t payload = 0, mode = 0, stride = 0, records = 0;
		std::size_t destination(std::size_t i) const
		{
			return output + (mode == 1 ? 4 : 0) + (mode == 2 ? i : keys[i] - (mode == 1 ? keys.front() : 0)) * stride + (mode == 2 ? 4 : 0);
		}
		std::uint64_t address(std::size_t i) const { return source.base + std::uint64_t(keys[i]) * source.descriptor.stride(); }
	};
	std::vector<field> fields;
	std::size_t output_size = 0;
	constexpr std::array<unsigned,8> widths{0,2,4,2,1,2,4,1};
	for (unsigned unit = 0; unit < 16; ++unit)
	{
		const auto& source = inputs.attributes[unit];
		const auto d = source.descriptor;
		if (!(inputs.attribute_mask & (1u << unit)) || !d.components() || range.source_vertices->empty()) continue;
		if (!d.type() || d.components() > 4 || (d.type() == 6 && d.components() != 1) || d.storage_mode())
			throw std::invalid_argument("Native sparse vertex format is invalid");
		field next; next.source = source; next.payload = widths[d.type()] * d.components();
		next.keys.reserve(range.source_vertices->size());
		for (const auto id : *range.source_vertices)
		{
			std::int64_t element = 0;
			if (d.frequency() && d.stride())
			{
				if (d.modulo())
				{
					const auto adjusted = std::int64_t(id) + range.vertex_index_offset;
					if (adjusted < 0 || adjusted > 0xffffffffll) throw std::out_of_range("Native sparse modulo source wraps");
					element = adjusted % d.frequency();
				}
				else
				{
					if (id < range.vertex_base_index) throw std::out_of_range("Native sparse quotient source is negative");
					element = (id - range.vertex_base_index) / d.frequency();
				}
			}
			next.keys.push_back(static_cast<std::uint32_t>(element));
		}
		std::sort(next.keys.begin(),next.keys.end());
		next.keys.erase(std::unique(next.keys.begin(),next.keys.end()),next.keys.end());
		const auto address = next.address(next.keys.size() - 1);
		const std::uint64_t limit = source.memory ? (1ull << 32) : source.bytes.size();
		if (source.base > limit || address > limit || next.payload > limit - address || (source.memory && !inputs.read_memory))
			throw std::out_of_range("Native sparse vertex source exceeds backing storage");
		auto existing = std::find_if(fields.begin(),fields.end(),[&](const auto& f)
		{
			return f.source.memory == source.memory && f.source.base == source.base &&
				(source.memory || (f.source.bytes.data() == source.bytes.data() && f.source.bytes.size() == source.bytes.size())) &&
				f.source.descriptor.stride() == d.stride() && f.payload == next.payload && f.keys == next.keys;
		});
		if (existing == fields.end())
		{
			const std::uint64_t dense_count = std::uint64_t(next.keys.back()) - next.keys.front() + 1;
			const std::uint64_t dense_bytes = dense_count * next.payload + (next.keys.front() ? 4 : 0);
			const std::uint64_t sparse_bytes = next.keys.size() * (next.payload + 4ull);
			next.mode = dense_bytes <= sparse_bytes ? (next.keys.front() ? 1 : 0) : 2;
			next.records = static_cast<std::uint32_t>(next.mode == 2 ? next.keys.size() : dense_count);
			next.stride = next.payload + (next.mode == 2 ? 4 : 0);
			const auto size = next.mode == 2 ? sparse_bytes : dense_bytes;
			const auto aligned = (std::uint64_t(output_size) + 3) & ~3ull;
			if (aligned > byte_limit || size > byte_limit - aligned || aligned + size > 0x20000000ull)
				throw std::length_error("Native sparse vertex capture exceeds budget");
			next.output = static_cast<std::size_t>(aligned);
			output_size = static_cast<std::size_t>(aligned + size);
			result.statistics.source_records += next.keys.size();
			if (next.mode == 0) ++result.statistics.dense_fields;
			else if (next.mode == 1) ++result.statistics.window_fields;
			else ++result.statistics.sparse_fields;
			fields.push_back(std::move(next)); existing = fields.end() - 1;
		}
		else ++result.statistics.shared_fields;
		const auto& f = *existing;
		// Preserve type/components, endian and original element calculation.
		result.fetch.attributes[unit] = {(d.format & 0x3fffff00u) | f.stride | (f.mode << 30),
			(d.storage & 0xa0000000u) | static_cast<std::uint32_t>(f.output)};
		if (!d.stride()) result.fetch.attributes[unit].format &= ~0xffff00u;
		result.fetch.element_counts[unit] = f.records;
	}
	result.bytes.resize(output_size); // holes are zero, never fetched from VM
	const auto word = [&](std::size_t offset, std::uint32_t value)
	{
		for (unsigned b = 0; b < 4; ++b) result.bytes[offset + b] = std::byte(value >> (b * 8));
	};
	struct cursor { std::uint64_t address; std::size_t field, index; };
	const auto greater = [](const cursor& a, const cursor& b) { return a.address > b.address; };
	std::priority_queue<cursor,std::vector<cursor>,decltype(greater)> pending(greater);
	for (std::size_t i = 0; i < fields.size(); ++i)
	{
		const auto& f = fields[i];
		if (f.mode == 1) word(f.output,f.keys.front());
		if (f.mode == 2) for (std::size_t k = 0; k < f.keys.size(); ++k) word(f.output + k * f.stride,f.keys[k]);
		if (f.source.memory) pending.push({f.address(0),i,0});
		else for (std::size_t k = 0; k < f.keys.size(); ++k)
		{
			std::memcpy(result.bytes.data() + f.destination(k),f.source.bytes.data() + f.address(k),f.payload);
			result.statistics.cpu_bytes += f.payload;
		}
	}
	// Merge adjacent/overlapping needed byte intervals. The bounded scratch
	// avoids a VM read for every attribute of every interleaved vertex.
	std::array<std::byte,4080> scratch;
	std::vector<cursor> consumers;
	while (!pending.empty())
	{
		const auto start = pending.top().address;
		auto end = start;
		consumers.clear();
		while (!pending.empty())
		{
			const auto c = pending.top();
			const auto& f = fields[c.field];
			if (c.address > end || c.address + f.payload - start > scratch.size()) break;
			pending.pop(); consumers.push_back(c); end = std::max(end,c.address + f.payload);
			if (c.index + 1 < f.keys.size()) pending.push({f.address(c.index + 1),c.field,c.index + 1});
		}
		inputs.read_memory(static_cast<std::uint32_t>(start),std::span(scratch).first(static_cast<std::size_t>(end - start)));
		++result.statistics.memory_calls; result.statistics.memory_bytes += end - start;
		for (const auto c : consumers)
		{
			const auto& f = fields[c.field];
			std::memcpy(result.bytes.data() + f.destination(c.index),scratch.data() + c.address - start,f.payload);
		}
	}
	return result;
}
} // namespace metal4
