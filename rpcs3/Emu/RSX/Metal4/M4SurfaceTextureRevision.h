#pragma once

#include "M4SurfaceMapping.h"
#include <algorithm>
#include <stdexcept>

namespace metal4
{
class texture_revision_conflict final : public std::runtime_error
{
  public:
	texture_revision_conflict() : std::runtime_error("Guest texture changed during native preparation") {}
};

// Immutable CPU-only snapshot dependency. Retains watched actual storage
// and its independent versions,not the cache or mutable native allocation.
// In particular bank permutation may address bytes outside a shader's
// nominal logical rectangle,so the source watches must survive too.
class surface_texture_revision final
{
	std::shared_ptr<guest_surface_memory> m_memory;
	std::vector<guest_surface_layout> m_coverage;
	std::vector<std::uint64_t> m_versions;
	std::shared_ptr<const surface_mapping_revision> m_mapping;
	std::vector<std::shared_ptr<const surface_texture_revision>> m_children;
  public:
	explicit surface_texture_revision(std::vector<std::shared_ptr<const surface_texture_revision>> children):m_children(std::move(children))
	{
		if(m_children.empty()||m_children.size()>64||std::any_of(m_children.begin(),m_children.end(),[](const auto& p){return !p;}))throw std::invalid_argument("Native texture assembly lacks bounded source revisions");
	}
	surface_texture_revision(std::shared_ptr<guest_surface_memory> memory, std::vector<guest_surface_layout> coverage,
		std::vector<std::uint64_t> versions, std::shared_ptr<const surface_mapping_revision> mapping)
		: m_memory(std::move(memory)), m_coverage(std::move(coverage)), m_versions(std::move(versions)), m_mapping(std::move(mapping))
	{
		if (!m_memory || m_coverage.empty() || m_coverage.size() != m_versions.size())
			throw std::invalid_argument("Native texture snapshot lacks owned storage revisions");
	}
	// Call under the same producer/VM exclusion used to capture these watches.
	[[nodiscard]] bool current() const
	{
		for(const auto& child:m_children)if(!child->current())return false;
		if (m_mapping && !m_mapping->current()) return false;
		for (std::size_t i = 0; i < m_coverage.size(); ++i)
			if (m_memory->surface_revision(m_coverage[i]) != m_versions[i]) return false;
		if(m_mapping&&!m_mapping->current())return false;
		return std::all_of(m_children.begin(),m_children.end(),[](const auto& child){return child->current();});
	}
};
} // namespace metal4
