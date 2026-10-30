#pragma once

#include "M4PhysicalBacking.h"

namespace metal4
{
class guarded_guest_memory;
struct physical_surface_state
{
	physical_identity identity;
	std::uint64_t revision;
};

// A one-publication ticket for the exact active bytes captured before native
// output. Virtual remaps do not invalidate it;writes to those physical bytes
// do. Only guarded_guest_memory can create or validate a ticket in its domain.
class physical_surface final
{
	friend class guarded_guest_memory;
	guest_surface_layout m_coverage;
	std::vector<std::shared_ptr<const physical_backing>> m_parts;
	std::shared_ptr<physical_surface_state> m_state;
	std::uint64_t m_expected;
	physical_surface(guest_surface_layout coverage, std::vector<std::shared_ptr<const physical_backing>> parts,
		std::shared_ptr<physical_surface_state> state)
		: m_coverage(std::move(coverage)), m_parts(std::move(parts)), m_state(std::move(state)), m_expected(m_state->revision) {}
  public:
	[[nodiscard]] const guest_surface_layout& coverage() const noexcept { return m_coverage; }
	[[nodiscard]] std::span<const std::shared_ptr<const physical_backing>> parts() const noexcept { return m_parts; }
	[[nodiscard]] bool overlaps(const physical_surface& other) const noexcept
	{
		for (const auto& backing : other.m_parts)
			if (m_state->identity.overlaps(*backing)) return true;
		return false;
	}
};
struct physical_surface_write
{
	std::shared_ptr<const physical_surface> surface;
	std::span<const std::byte> bytes;
};
} // namespace metal4
