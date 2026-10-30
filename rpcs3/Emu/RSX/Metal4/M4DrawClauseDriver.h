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
// Drive each current range separately;never collapse a trivial instanced
// clause because each pass may own different constants or input addresses.
// First-pass state already exists at end(),as in VK::emit_geometry(0).
// Callback owns preparation/VM writer exclusion and native submission.
template <typename Clause, typename Context, typename Emit>
std::uint32_t drive_draw_clause(Clause& clause, Context* context, Emit&& emit, std::uint32_t first_flags,
								clause_cleanup_owner cleanup = clause_cleanup_owner::driver)
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
			const auto flags = pass ? clause.execute_pipeline_dependencies(context) : first_flags;
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
