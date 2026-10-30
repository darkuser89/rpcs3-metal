#pragma once

#include "M4GuestSurface.h"
#include <limits>
#include <span>
#include <vector>

namespace metal4
{
// Immutable exact guest-to-physical slices captured under VM exclusion. The
// owner is an independent mapping,not the guest VA or the VM allocation handle.
// Keeping it alive preserves bytes after unmap;it does not exclude writers or
// prove where a later retried instruction stores.
class physical_backing final
{
  public:
	struct segment
	{
		guest_surface_range guest;
		std::shared_ptr<void> owner;
		std::size_t offset, allocation_bytes;
		[[nodiscard]] std::byte* data() const noexcept { return static_cast<std::byte*>(owner.get()) + offset; }
	};
  private:
	guest_surface_range m_range;
	std::vector<segment> m_segments;
  public:
	physical_backing(guest_surface_range range, std::vector<segment> segments)
		: m_range(range), m_segments(std::move(segments))
	{
		validate_guest_surface_range(range);
		if (!range.size || m_segments.empty()) throw std::invalid_argument("Physical backing requires nonempty exact coverage");
		std::uint64_t cursor = range.address;
		for (const auto& part : m_segments)
		{
			validate_guest_surface_range(part.guest);
			if (!part.owner || !part.guest.size || part.guest.address != cursor ||
				part.guest.size > std::uint64_t(range.address) + range.size - cursor ||
				part.offset > part.allocation_bytes || part.guest.size > part.allocation_bytes - part.offset)
				throw std::invalid_argument("Physical backing slices have a gap,overlap or invalid allocation extent");
			cursor += part.guest.size;
		}
		if (cursor != std::uint64_t(range.address) + range.size)
			throw std::invalid_argument("Physical backing slices do not cover the requested bytes");
	}
	[[nodiscard]] guest_surface_range range() const noexcept { return m_range; }
	[[nodiscard]] std::span<const segment> segments() const noexcept { return m_segments; }
	[[nodiscard]] static bool same_allocation(const segment& a, const segment& b) noexcept
	{
		return !a.owner.owner_before(b.owner) && !b.owner.owner_before(a.owner);
	}
	// Guest addresses may have been replaced or may name different aliases.
	// Only the independent allocation identity and exact byte offsets decide
	// whether an already completed store changed the current physical slice.
	[[nodiscard]] bool overlaps(const physical_backing& current) const noexcept
	{
		for (const auto& written : m_segments)
			for (const auto& mapped : current.m_segments)
				if (same_allocation(written, mapped) && written.offset < mapped.offset + mapped.guest.size &&
					mapped.offset < written.offset + written.guest.size) return true;
		return false;
	}
};

// Revision bookkeeping keeps allocation control-block identities without
// retaining mappings. The actual ticket owns the independently mapped bytes.
// Comparing weak owners does not lock them or release a mapping under the VM.
class physical_identity final
{
  public:
	struct segment { std::weak_ptr<void> owner; std::size_t offset, size; };
  private:
	std::vector<segment> m_segments;
  public:
	explicit physical_identity(std::vector<segment> segments) : m_segments(std::move(segments))
	{
		const std::weak_ptr<void> empty;
		for (const auto& part : m_segments)
			if ((!part.owner.owner_before(empty) && !empty.owner_before(part.owner)) || !part.size ||
				part.offset > std::numeric_limits<std::size_t>::max() - part.size)
				throw std::invalid_argument("Physical identity requires bounded nonempty slices");
	}
	explicit physical_identity(std::span<const std::shared_ptr<const physical_backing>> backings)
	{
		for (const auto& backing : backings)
		{
			if (!backing) throw std::invalid_argument("Physical identity requires retained backing slices");
			for (const auto& part : backing->segments()) m_segments.push_back({part.owner, part.offset, part.guest.size});
		}
	}
	[[nodiscard]] std::span<const segment> segments() const noexcept { return m_segments; }
	[[nodiscard]] bool overlaps(const physical_identity& other) const noexcept
	{
		for (const auto& kept : m_segments)
			for (const auto& part : other.m_segments)
				if (!kept.owner.owner_before(part.owner) && !part.owner.owner_before(kept.owner) &&
					kept.offset < part.offset + part.size && part.offset < kept.offset + kept.size) return true;
		return false;
	}
	[[nodiscard]] bool overlaps(const physical_backing& written) const noexcept
	{
		for (const auto& kept : m_segments)
			for (const auto& part : written.segments())
				if (!kept.owner.owner_before(part.owner) && !part.owner.owner_before(kept.owner) &&
					kept.offset < part.offset + part.guest.size && part.offset < kept.offset + kept.size) return true;
		return false;
	}
};
} // namespace metal4
