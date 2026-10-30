#include "M4ConstantUpload.h"
#include "M4Resources.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace metal4
{
struct constant_upload_cache::implementation
{
	using word = std::array<std::uint32_t, 4>;
	struct entry
	{
		std::uint64_t fingerprint;
		std::size_t logical_bytes;
		std::vector<word> words;
		id<MTLBuffer> buffer;
	};
	context& ctx;
	std::list<entry> entries;
	std::unordered_multimap<std::uint64_t, std::list<entry>::iterator> index;
	constant_upload_statistics stats;
	explicit implementation(context& value) : ctx(value) {}
	void remove_last()
	{
		const auto last_entry = std::prev(entries.end());
		const auto [first, last] = index.equal_range(last_entry->fingerprint);
		for (auto it = first; it != last; ++it)
			if (it->second == last_entry) { index.erase(it); break; }
		stats.payload_bytes -= last_entry->words.size() * sizeof(word);
		entries.erase(last_entry);
		++stats.evicted;
	}
	template <typename Read>
	id<MTLBuffer> get(std::size_t count, Read read, std::size_t maximum_bytes = 1u << 20, std::size_t exact_bytes = 0)
	{
		if (count > maximum_bytes / sizeof(word))
			throw std::length_error("Native immutable upload exceeds its input bound");
		// Empty literal arrays still bind a valid zero register, with count=0
		// in the independent shader context. Sharing zero bytes is safe.
		const auto size = std::max(count, std::size_t{1});
		const auto logical_bytes = exact_bytes ? exact_bytes : size * sizeof(word);
		const auto get_word = [&](std::size_t i) { return count ? read(i) : word{}; };
		std::uint64_t hash = 14695981039346656037ull ^ logical_bytes;
		for (std::size_t i = 0; i < size; ++i)
			for (auto value : get_word(i)) hash = (hash ^ value) * 1099511628211ull;
		const auto [first, last] = index.equal_range(hash);
		for (auto candidate = first; candidate != last; ++candidate)
		{
			const auto it = candidate->second;
			if (it->words.size() != size || it->logical_bytes != logical_bytes) continue;
			bool equal = true;
			for (std::size_t i = 0; i < size; ++i)
				if (it->words[i] != get_word(i)) { equal = false; break; }
			if (!equal) continue;
			const auto buffer = it->buffer;
			entries.splice(entries.begin(), entries, it);
			++stats.reused;
			return buffer;
		}
		std::vector<word> values(size);
		for (std::size_t i = 0; i < size; ++i) values[i] = get_word(i);
		const auto payload = size * sizeof(word);
		// Allocate before publishing the new version. Heap recovery may trim this
		// cache; pending bindings own earlier buffers independently.
		const auto buffer = make_upload(ctx, std::as_bytes(std::span(values)));
		++stats.created;
		stats.uploaded_bytes += payload;
		if (!stats.capacity || payload > stats.payload_budget)
		{
			++stats.bypassed;
			return buffer;
		}
		entries.push_front({hash, logical_bytes, std::move(values), buffer});
		try { index.emplace(hash, entries.begin()); }
		catch (...) { entries.pop_front(); throw; }
		while (entries.size() > stats.capacity || stats.payload_bytes > stats.payload_budget - payload) remove_last();
		stats.payload_bytes += payload;
		stats.entries = entries.size();
		return buffer;
	}
};
constant_upload_cache::constant_upload_cache(context& ctx, std::size_t capacity, std::size_t payload_budget)
{
	if (capacity > 4096 || (capacity && !payload_budget))
		throw std::invalid_argument("Native constant cache capacity/budget is invalid");
	m_impl = std::make_unique<implementation>(ctx);
	m_impl->stats.capacity = capacity;
	m_impl->stats.payload_budget = payload_budget;
}
constant_upload_cache::~constant_upload_cache() = default;
id<MTLBuffer> constant_upload_cache::words(std::span<const std::byte> bytes)
{
	if (bytes.size() % 16) throw std::invalid_argument("Native constant words require complete registers");
	return m_impl->get(bytes.size() / 16, [&](std::size_t i)
	{
		implementation::word result;
		std::memcpy(result.data(), bytes.data() + i * 16, 16);
		return result;
	});
}
id<MTLBuffer> constant_upload_cache::packed_bytes(std::span<const std::byte> bytes)
{
	// Preserve the existing per-stream packed-offset domain. The snapshot's
	// caller-selected capture budget (normally64MiB total) is checked earlier.
	if (bytes.size() > 0x20000000u) throw std::length_error("Native packed stream exceeds512MiB");
	if (bytes.empty()) return nil;
	if (!m_impl->stats.capacity || bytes.size() > m_impl->stats.payload_budget)
	{
		const auto buffer = make_upload(m_impl->ctx,bytes);
		++m_impl->stats.created; ++m_impl->stats.bypassed; m_impl->stats.uploaded_bytes += bytes.size();
		return buffer;
	}
	return m_impl->get((bytes.size() + 15) / 16,[&](std::size_t i)
	{
		implementation::word result{};
		const auto offset = i * 16;
		std::memcpy(result.data(),bytes.data() + offset,std::min<std::size_t>(16,bytes.size() - offset));
		return result;
	},0x20000000u,bytes.size());
}
id<MTLBuffer> constant_upload_cache::fragment(const decoded_fragment_program& compiled, std::span<const std::byte> bytes)
{
	const auto count = validate_fragment_constants(compiled, bytes);
	m_impl->stats.instruction_checks += compiled.instructions.size();
	// Store only offsets on the stack. No decoded-program/literal vector is
	// allocated on a hit; exact instruction/layout checks precede cache access.
	std::array<std::size_t, 4096> offsets;
	if (count > offsets.size()) throw std::length_error("Native fragment literal count exceeds decoder limit");
	std::size_t index = 0;
	for (const auto& instruction : compiled.instructions)
		if (instruction.literal) offsets[index++] = instruction.byte_offset + 16;
	return m_impl->get(count, [&](std::size_t i) { return decode_fragment_block(bytes.subspan(offsets[i], 16).first<16>()); });
}
constant_upload_statistics constant_upload_cache::statistics() const noexcept { return m_impl->stats; }
void constant_upload_cache::clear()
{
	m_impl->stats.evicted += m_impl->entries.size();
	m_impl->index.clear();
	m_impl->entries.clear();
	m_impl->stats.entries = m_impl->stats.payload_bytes = 0;
}
} // namespace metal4
