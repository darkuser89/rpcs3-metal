#include "M4DrawClauseDriver.h"
#include <iostream>
#include <stdexcept>
#include <vector>

struct context
{
	unsigned cleanup = 0;
	std::vector<unsigned> dependencies;
};
struct clause
{
	unsigned count = 3, index = 0, begins = 0;
	bool rendering = false, is_trivial_instanced_draw = true;
	bool empty() const { return !count; }
	void begin()
	{
		index = 0;
		rendering = true;
		++begins;
	}
	void end() { index = count - 1; }
	bool next()
	{
		if (++index >= count)
		{
			index = 0;
			rendering = false;
			return false;
		}
		return true;
	}
	unsigned execute_pipeline_dependencies(context* state)
	{
		state->dependencies.push_back(index);
		return index * 8;
	}
	void post_execute_cleanup(context* state)
	{
		if (index || rendering)
			throw std::logic_error("Unfinished clause reached cleanup");
		++state->cleanup;
	}
};
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native draw clause driver contract differs");
}
int main()
{
	try
	{
		context state;
		clause draw;
		std::vector<unsigned> passes, flags;
		const auto count = metal4::drive_draw_clause(draw, &state, [&](unsigned pass, unsigned flag)
													 {
			require(draw.rendering && draw.index == pass);
			passes.push_back(pass); flags.push_back(flag); }, 4);
		require(count == 3 && passes == std::vector<unsigned>{0, 1, 2} && flags == std::vector<unsigned>{4, 8, 16} &&
				state.dependencies == std::vector<unsigned>{1, 2} && state.cleanup == 1 && !draw.rendering && draw.index == 0);
		context deferred;
		clause thread_owned;
		require(metal4::drive_draw_clause(thread_owned, &deferred, [](auto, auto) {}, 4,
			metal4::clause_cleanup_owner::rsx_thread_end) == 3 && !deferred.cleanup &&
			!thread_owned.rendering && thread_owned.index == 0);
		thread_owned.post_execute_cleanup(&deferred);
		require(deferred.cleanup == 1);
		context rejected;
		clause invalid_owner;
		bool invalid = false;
		try
		{
			(void)metal4::drive_draw_clause(invalid_owner, &rejected, [](auto, auto) {}, 4,
				static_cast<metal4::clause_cleanup_owner>(99));
		}
		catch (const std::invalid_argument&) { invalid = true; }
		require(invalid && !invalid_owner.begins && rejected.dependencies.empty() && !rejected.cleanup);
		// Failure at every range must normalize the iterator,keep the original
		// exception and avoid applying dangling post-execution dependencies.
		for (unsigned failure = 0; failure < 3; ++failure)
		{
			context failed;
			clause interrupted;
			unsigned visited = 0;
			bool caught = false;
			try
			{
				(void)metal4::drive_draw_clause(interrupted, &failed, [&](unsigned pass, unsigned)
												{
					++visited;
					if (pass == failure) throw std::range_error("expected submit failure"); }, 4);
			}
			catch (const std::range_error&)
			{
				caught = true;
			}
			require(caught && visited == failure + 1 && !interrupted.rendering && interrupted.index == 0 && !failed.cleanup);
		}
		clause empty;
		empty.count = 0;
		unsigned emits = 0;
		require(metal4::drive_draw_clause(empty, &state, [&](auto, auto)
										  { ++emits; }, 4) == 0 &&
				!emits && !empty.begins);
		std::cout << "Native clause driver: three separate passes including trivial instancing,ordered dependencies,cleanup,empty andall submit-failure normalization passed\n";
		std::cout << "Native real-renderer cleanup ownership: driver defers cleanup,RSX end performs exactly once,invalid owner rejected before state mutation\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
