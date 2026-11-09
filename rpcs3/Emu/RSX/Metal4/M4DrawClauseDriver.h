#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>

namespace metal4
{
enum class clause_cleanup_owner
{
	driver,
	rsx_thread_end
};
// Inspect only the range cursor after drive_draw_clause has consumed it.
// The caller identifies owned inline input: its vertices were copied before
// submission and it has no trailing range dependency to consume at cleanup.
template <typename Clause>
[[nodiscard]] bool owned_clause_requires_draw_completion(Clause& clause, bool inline_vertices)
{
	if (inline_vertices) return false;
	if (clause.empty()) return false;
	clause.end();
	const bool trailing_dependency = clause.get_range().count == 0;
	(void)clause.next();
	return trailing_dependency;
}
// Drive each current range separately;never collapse a trivial instanced
// clause because each pass may own different constants or input addresses.
// First-pass state already exists at end(),as in VK::emit_geometry(0).
// Callback owns preparation/VM writer exclusion and native submission.
// The later-pass hook runs before guest dependencies and can retire GPU work.
template <typename Clause, typename Context, typename Emit, typename BeforeLaterDependencies = decltype([] {})>
std::uint32_t drive_draw_clause(Clause& clause, Context* context, Emit&& emit, std::uint32_t first_flags,
								clause_cleanup_owner cleanup = clause_cleanup_owner::driver, bool apply_first_dependencies = false,
								const BeforeLaterDependencies& before_later_dependencies = {})
{
	if (cleanup != clause_cleanup_owner::driver && cleanup != clause_cleanup_owner::rsx_thread_end)
		throw std::invalid_argument("Native clause cleanup owner is invalid");
	if (clause.empty())
		return 0;
	std::uint32_t pass = 0;
	clause.begin();
	try
	{
		do
		{
			auto flags = pass ? 0u : first_flags;
			if (pass) std::invoke(before_later_dependencies);
			if (pass || apply_first_dependencies) flags |= clause.execute_pipeline_dependencies(context);
			std::invoke(emit, pass, flags);
			++pass;
		} while (clause.next());
		if (cleanup == clause_cleanup_owner::driver)
			clause.post_execute_cleanup(context);
		return pass;
	}
	catch (...)
	{
		// Do not execute trailing dependencies after a failed preparation or
		// submission. Normalize iterator/rendering state before propagating.
		clause.end();
		(void)clause.next();
		throw;
	}
}
} // namespace metal4
