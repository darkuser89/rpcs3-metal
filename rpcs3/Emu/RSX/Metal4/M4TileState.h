#pragma once

#include "M4CPUCapture.h"
#include "M4GuestSurface.h"
#include <algorithm>
#include <array>
#include <limits>

namespace metal4
{
struct tile_descriptor
{
	std::uint32_t location = 0, offset = 0, size = 0, pitch = 0, comp = 0, base = 0, bank = 0;
	bool bound = false;
	bool operator==(const tile_descriptor&) const = default;
};
// All native descriptor readers and HLE/LLE mutations share this mutex.
// The CPU provider is acquired BEFORE this mutex and retains it through byte
// capture or CAS publication. Producers must never enter VM/GPU work here.
// Recursive acquisition is only for short queries within that outer epoch.
class tile_state final
{
	mutable std::recursive_mutex m_mutex;
	std::array<tile_descriptor, 15> m_tiles;
	std::array<std::array<std::uint64_t, 65536>, 2> m_entries{};
	std::uint64_t m_generation;
	static constexpr auto poisoned = std::numeric_limits<std::uint64_t>::max();
	void mark(const tile_descriptor& tile) noexcept
	{
		if (!tile.bound)
			return;
		if (tile.location >= m_entries.size() || !tile.size || ((tile.offset | tile.size) & 0xffff) || tile.size > 0x1'0000'0000ull - tile.offset)
		{
			m_generation = poisoned;
			return;
		}
		const auto end = (std::uint64_t(tile.offset) + tile.size) >> 16;
		std::fill(m_entries[tile.location].begin() + (tile.offset >> 16), m_entries[tile.location].begin() + end, m_generation);
	}

  public:
	static constexpr std::size_t table_bytes = sizeof(m_entries);
	explicit tile_state(std::array<tile_descriptor, 15> initial = {}, std::uint64_t generation = 1)
		: m_tiles(initial), m_generation(generation)
	{
		if (!generation || generation == poisoned)
			throw std::invalid_argument("Native tile epoch requires a nonzero unexhausted generation");
		for (const auto& tile : m_tiles) mark(tile);
	}
	template <typename F>
	decltype(auto) execute(F&& operation) const
	{
		std::lock_guard lock(m_mutex);
		return std::invoke(std::forward<F>(operation));
	}
	[[nodiscard]] auto snapshot() const
	{
		return execute([&] { return m_tiles; });
	}
	[[nodiscard]] tile_descriptor get(std::size_t index) const
	{
		return execute([&] { return m_tiles.at(index); });
	}
	void update(std::size_t index, const tile_descriptor& value)
	{
		execute([&]
		{
			auto& tile = m_tiles.at(index);
			if (tile == value) return;
			if (m_generation != poisoned) ++m_generation;
			mark(tile);
			mark(value);
			tile = value;
		});
	}
	[[nodiscard]] std::uint64_t current() const
	{
		return execute([&]
		{
			if (m_generation == poisoned) throw std::overflow_error("Native tile epoch is exhausted or invalid");
			return m_generation;
		});
	}
	[[nodiscard]] std::uint64_t current(unsigned location, guest_surface_range range) const
	{
		validate_guest_surface_range(range);
		return execute([&]
		{
			(void)current();
			if (location >= m_entries.size()) throw std::out_of_range("Native tile location is invalid");
			std::uint64_t result = 0;
			if (range.size)
			{
				const auto end = (std::uint64_t(range.address) + range.size + 0xffff) >> 16;
				for (auto entry = range.address >> 16; entry < end; ++entry)
					result = std::max(result, m_entries[location][entry]);
			}
			return result;
		});
	}
};
// Capture the shared owner by value:retained/foreign provider callbacks still
// go through checked_cpu_capture;no stack reference outlives this dispatch.
inline cpu_capture_dispatch tile_capture_dispatch(cpu_capture_dispatch provider, std::shared_ptr<tile_state> tiles)
{
	if (!provider || !tiles) throw std::invalid_argument("Native tile capture needs a provider and descriptor owner");
	return [provider = std::move(provider), tiles = std::move(tiles)](const std::function<void()>& operation)
	{
		provider([tiles, operation] { tiles->execute(operation); });
	};
}
} // namespace metal4
