#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>

namespace metal4
{
inline constexpr std::size_t backing_alignment = 0x10000;
inline std::size_t independent_backing_charge(std::size_t bytes)
{
	if (!bytes || bytes > std::numeric_limits<std::size_t>::max() - backing_alignment)
		throw std::length_error("Independent backing reservation size is invalid");
	return bytes + backing_alignment;
}
// The mapper replaces exactly bytes at the supplied aligned address. Reserve
// the whole range first;never invoke a fixed-address mapper after mmap fails.
// Retain both alignment padding and the replacement in one owned reservation.
template <typename Mapper>
std::shared_ptr<void> map_independent_backing(std::size_t bytes, Mapper&& mapper)
{
	const auto charge = independent_backing_charge(bytes);
	auto* base = ::mmap(nullptr, charge, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
	if (base == MAP_FAILED) throw std::runtime_error("Independent backing reservation failed");
	auto reservation = std::shared_ptr<void>(base, [charge](void* value) noexcept { (void)::munmap(value, charge); });
	const auto address = (reinterpret_cast<std::uintptr_t>(base) + backing_alignment - 1) & ~(std::uintptr_t(backing_alignment) - 1);
	auto* aligned = reinterpret_cast<void*>(address);
	const auto mapped = mapper(aligned);
	if (mapped != aligned) throw std::runtime_error("Independent backing replacement failed");
	return std::shared_ptr<void>(std::move(reservation), aligned);
}
} // namespace metal4
