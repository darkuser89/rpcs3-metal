#pragma once

#include "M4GuestSurface.h"

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <utility>

namespace metal4
{
template <typename Lookup>
[[nodiscard]] bool equivalent_guest_memory_ranges(guest_surface_range a, guest_surface_range b,
	Lookup&& lookup, std::uint64_t block_size = 65536)
{
	validate_guest_surface_range(a);
	validate_guest_surface_range(b);
	if (!block_size || !std::has_single_bit(block_size) || block_size > 0x100000000ull)
		throw std::invalid_argument("Guest memory mapping block size is invalid");
	if (a.size != b.size)
		return false;
	const auto mask = block_size - 1;
	std::uint64_t offset = 0;
	while (offset < a.size)
	{
		const auto from = std::uint64_t(a.address) + offset, to = std::uint64_t(b.address) + offset;
		const auto af = std::invoke(lookup, static_cast<std::uint32_t>(from & ~mask));
		const auto bf = std::invoke(lookup, static_cast<std::uint32_t>(to & ~mask));
		if ((af | bf) & mask)
			throw std::invalid_argument("Guest memory identity is not block aligned");
		if (af > std::numeric_limits<std::uint64_t>::max() - (from & mask) ||
			bf > std::numeric_limits<std::uint64_t>::max() - (to & mask))
			throw std::overflow_error("Guest memory identity wraps64bits");
		if (af + (from & mask) != bf + (to & mask))
			return false;
		offset += std::min({std::uint64_t(a.size) - offset, block_size - (from & mask), block_size - (to & mask)});
	}
	return true;
}
// Rebase a requested virtual range into one unique source-storage interval.
// Every byte is proven by equal identities at each linear mapping-block
// boundary,not merely by overlapping endpoints. Reordered/missing mappings
// and ambiguous duplicate source mirrors cannot yield a valid translation.
template <typename Lookup>
[[nodiscard]] std::optional<guest_surface_range> rebase_guest_memory_range(
	guest_surface_range source, guest_surface_range request, Lookup&& lookup, std::uint64_t block_size = 65536)
{
	validate_guest_surface_range(source);
	validate_guest_surface_range(request);
	if (!block_size || !std::has_single_bit(block_size) || block_size > 0x100000000ull)
		throw std::invalid_argument("Guest memory mapping block size is invalid");
	if (!source.size || !request.size || request.size > source.size)
		return {};
	const auto mask = block_size - 1;
	const auto identity = [&](std::uint64_t address)
	{
		const auto base = std::invoke(lookup, static_cast<std::uint32_t>(address & ~mask));
		if (base & mask)
			throw std::invalid_argument("Guest memory rebase identity is not block aligned");
		if (base > std::numeric_limits<std::uint64_t>::max() - (address & mask))
			throw std::overflow_error("Guest memory rebase identity wraps64bits");
		return base + (address & mask);
	};
	const auto source_end = std::uint64_t(source.address) + source.size;
	const auto first_identity = identity(request.address);
	std::optional<guest_surface_range> result;
	for (auto block = std::uint64_t(source.address) & ~mask; block < source_end; block += block_size)
	{
		const auto candidate = block + (request.address & mask);
		if (candidate < source.address || candidate >= source_end || request.size > source_end - candidate ||
			identity(candidate) != first_identity)
			continue;
		const guest_surface_range translated{static_cast<std::uint32_t>(candidate), request.size};
		if (!equivalent_guest_memory_ranges(translated, request, lookup, block_size))
			continue;
		if (result)
			return {}; // More than one source rectangle would be possible.
		result = translated;
	}
	return result;
}
// VM canonical alias identities are aligned64KiB bases. Preserve all high
// identity/lock bits;never compare only guest virtual addresses or low32bits.
template <typename Lookup>
[[nodiscard]] bool guest_memory_aliases_overlap(guest_surface_range a, guest_surface_range b, Lookup&& lookup)
{
	validate_guest_surface_range(a);
	validate_guest_surface_range(b);
	const auto next = [&](std::uint64_t& address, std::uint64_t end)
	{
		const auto base = std::invoke(lookup, static_cast<std::uint32_t>(address) & ~65535u);
		if (base & 65535u)
			throw std::invalid_argument("Guest memory alias identity is not64KiB aligned");
		const auto first = base | (address & 65535u);
		auto size = std::min(end - address, 65536 - (address & 65535u));
		address += size;
		while (address < end)
		{
			const auto following = std::invoke(lookup, static_cast<std::uint32_t>(address));
			if ((following & 65535u) || first > std::numeric_limits<std::uint64_t>::max() - size || following != first + size)
				break;
			const auto count = std::min(end - address, std::uint64_t(65536));
			size += count;
			address += count;
		}
		if (first > std::numeric_limits<std::uint64_t>::max() - size)
			throw std::overflow_error("Guest memory alias identity wraps64bits");
		return std::pair{first, size};
	};
	const auto a_end = std::uint64_t(a.address) + a.size, b_end = std::uint64_t(b.address) + b.size;
	for (auto ap = std::uint64_t(a.address); ap < a_end;)
	{
		const auto ar = next(ap, a_end);
		for (auto bp = std::uint64_t(b.address); bp < b_end;)
		{
			const auto br = next(bp, b_end);
			if (ar.first <= br.first ? br.first - ar.first < ar.second : ar.first - br.first < br.second)
				return true;
		}
	}
	return false;
}
} // namespace metal4
