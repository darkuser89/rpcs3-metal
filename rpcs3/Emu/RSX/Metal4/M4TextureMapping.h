#pragma once

#include "M4IOMappingEpoch.h"
#include "M4GuestSegments.h"
#include <functional>
#include <memory>

namespace metal4
{
// Probe main IO one logical 1MiB entry at a time under common producer
// exclusion. Adjacent physical spans coalesce;disjoint/duplicate spans retain
// logical order. No fictitious first-EA-plus-allocation envelope is consulted.
[[nodiscard]] inline std::vector<guest_storage_segment> texture_storage_segments(
	std::uint32_t offset, unsigned location, std::size_t size,
	const std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)>& translate)
{
	validate_guest_surface_range({offset, size});
	if (!size || location > 1 || !translate)
		throw std::invalid_argument("Texture storage requires local/main mapping and nonempty allocation");
	std::vector<guest_storage_segment> result;
	for (std::size_t done = 0; done < size;)
	{
		const auto logical = static_cast<std::uint32_t>(std::uint64_t(offset) + done);
		const auto bytes = location == 1 ? std::min(size - done, std::size_t(0x100000 - (logical & 0xfffff))) : size;
		const auto address = translate(logical, location, bytes);
		validate_guest_surface_range({address, bytes});
		if (!result.empty() && std::uint64_t(result.back().address) + result.back().size == address)
			result.back().size += bytes;
		else
			result.push_back({address, done, bytes});
		done += bytes;
	}
	return result;
}
// Normalized RSX texture locations are local=0,main=1. Only main textures
// depend on the logical IO allocation. Keep this counter separate from byte
// revisions,which may already be larger and cannot represent IO ABA.
// Construct/query inside the same producer exclusion as translation/VM reads.
class texture_io_revision final
{
	std::shared_ptr<io_mapping_epoch> m_epochs;
	guest_surface_range m_range;
	std::uint64_t m_generation = 0;

  public:
	texture_io_revision(std::shared_ptr<io_mapping_epoch> epochs = {}, unsigned location = 0, guest_surface_range range = {})
		: m_epochs(location == 1 ? std::move(epochs) : nullptr), m_range(range)
	{
		if (location > 1) throw std::invalid_argument("Native texture location is not local/main");
		validate_guest_surface_range(range);
		if (m_epochs)
		{
			if (!range.size) throw std::invalid_argument("Native main texture IO range is empty");
			m_generation = m_epochs->current(range);
		}
	}
	[[nodiscard]] bool current() const
	{
		return !m_epochs || m_epochs->current(m_range) == m_generation;
	}
};
} // namespace metal4
