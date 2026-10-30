#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>

namespace metal4
{
// Allocation-free native-page selection and protection union. The VM owner
// supplies current logical-page permissions/watch intersections under writer
// exclusion. No source bytes or allocation handles are kept here.
class page_protection_plan final
{
	struct state { bool owned = false, dirty = false; };
	const std::uint64_t m_page_size;
	std::vector<state> m_states;
	std::vector<std::uint32_t> m_dirty;
	std::size_t m_count = 0;
  public:
	struct access { int baseline = PROT_READ | PROT_WRITE, overlay = PROT_READ | PROT_WRITE; bool selected = false; };
	struct statistics { std::uint64_t native_pages = 0, logical_pages = 0, protection_runs = 0; };
  private:
	statistics m_statistics{};
	[[nodiscard]] static std::uint64_t checked_size(std::uint64_t size)
	{
		if (size < 4096 || size > 65536 || !std::has_single_bit(size)) throw std::invalid_argument("Invalid native protection page size");
		return size;
	}
  public:
	explicit page_protection_plan(std::uint64_t page_size)
		: m_page_size(checked_size(page_size)), m_states(0x1'0000'0000ull / m_page_size), m_dirty(m_states.size()) {}
	page_protection_plan(const page_protection_plan&) = delete;
	page_protection_plan& operator=(const page_protection_plan&) = delete;
	[[nodiscard]] bool owned(std::uint32_t address) const noexcept { return m_states[address / m_page_size].owned; }
	[[nodiscard]] statistics counters() const noexcept { return m_statistics; }
	[[nodiscard]] std::size_t pending_pages() const noexcept { return m_count; }
	[[nodiscard]] std::size_t metadata_bytes() const noexcept { return m_states.size() * sizeof(state) + m_dirty.size() * sizeof(std::uint32_t); }
	void mark(std::uint32_t address) noexcept
	{
		const auto id = static_cast<std::uint32_t>(address / m_page_size);
		if (!m_states[id].dirty) { m_states[id].dirty = true; m_dirty[m_count++] = id; }
	}
	// Apply must terminate/fail closed if protection cannot be installed.
	// Widening a failed union and continuing producer execution is forbidden.
	template <typename Lookup, typename Apply>
	void synchronize(std::uint64_t first, std::uint64_t end, Lookup&& lookup, Apply&& apply)
	{
		if (first > end || end > 0x1'0000'0000ull) throw std::invalid_argument("Invalid native protection interval");
		if (first == end) return;
		first &= ~(m_page_size - 1);
		end = (end + m_page_size - 1) & ~(m_page_size - 1);
		std::uint64_t pending_first = 0, pending_end = 0;
		int pending_permissions = 0;
		const auto flush = [&]
		{
			if (pending_first != pending_end)
			{
				++m_statistics.protection_runs;
				apply(pending_first, pending_end - pending_first, pending_permissions);
			}
			pending_first = pending_end;
		};
		for (auto address = first; address < end; address += m_page_size)
		{
			++m_statistics.native_pages;
			auto& state = m_states[address / m_page_size];
			int baseline = PROT_READ | PROT_WRITE, overlay = baseline;
			bool selected = false;
			for (auto logical = address; logical < address + m_page_size; logical += 4096)
			{
				++m_statistics.logical_pages;
				const auto part = lookup(static_cast<std::uint32_t>(logical));
				baseline &= part.baseline; overlay &= part.overlay; selected |= part.selected;
			}
			if (!selected && !state.owned) { flush(); continue; }
			if (pending_end != address || pending_permissions != (baseline & overlay)) flush();
			if (pending_first == pending_end) { pending_first = address; pending_permissions = baseline & overlay; }
			pending_end = address + m_page_size; state.owned = selected;
		}
		flush();
	}
	template <typename Lookup, typename Apply>
	void synchronize_dirty(Lookup&& lookup, Apply&& apply)
	{
		std::sort(m_dirty.begin(), m_dirty.begin() + m_count);
		for (std::size_t i = 0; i < m_count;)
		{
			const auto first = m_dirty[i]; auto last = first;
			m_states[last].dirty = false;
			while (++i < m_count && m_dirty[i] == last + 1) { last = m_dirty[i]; m_states[last].dirty = false; }
			synchronize(std::uint64_t(first) * m_page_size, (std::uint64_t(last) + 1) * m_page_size, lookup, apply);
		}
		m_count = 0;
	}
};
} // namespace metal4
