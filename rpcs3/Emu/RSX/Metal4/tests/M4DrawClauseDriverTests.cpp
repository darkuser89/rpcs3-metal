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
	struct range { unsigned count = 1; } last;
	unsigned range_reads = 0;
	const range& get_range()
	{
		if (!count || index >= count) throw std::logic_error("Inline clause has no range");
		++range_reads;
		return last;
	}
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
		clause owned_inline;
		owned_inline.count = 0;
		if (metal4::owned_clause_requires_draw_completion(owned_inline, true))
			throw std::runtime_error("Owned inline clause retained a completion wait");
		require(!owned_inline.range_reads && !owned_inline.begins);
		for (unsigned ranges : {0u, 1u, 3u}) for (bool trailing : {false, true})
		{
			clause completion;
			completion.count = ranges;
			completion.last.count = trailing ? 0u : 3u;
			require(metal4::owned_clause_requires_draw_completion(completion, false) == bool(ranges && trailing));
			require(completion.index == 0 && !completion.rendering && completion.range_reads == unsigned(bool(ranges)));
		}
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
		context conditional;
		clause predicated;
		unsigned conditional_emits = 0;
		unsigned completion_boundaries = 0;
		require(metal4::drive_draw_clause(predicated, &conditional, [&](unsigned pass, unsigned flags)
			{
				require(conditional.dependencies.size() == pass + 1 && flags == (pass ? pass * 8 : 4) &&
					completion_boundaries == pass);
				++conditional_emits;
			}, 4, metal4::clause_cleanup_owner::rsx_thread_end, true, [&]
			{
				require(conditional_emits == conditional.dependencies.size() && conditional_emits == completion_boundaries + 1);
				++completion_boundaries;
			}) == 3 && conditional_emits == 3 && completion_boundaries == 2 &&
			conditional.dependencies == std::vector<unsigned>{0, 1, 2} && !conditional.cleanup);
		predicated.post_execute_cleanup(&conditional);
		require(conditional.cleanup == 1);
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
		context blocked;
		clause incomplete;
		bool wait_failed = false;
		try
		{
			(void)metal4::drive_draw_clause(incomplete, &blocked, [](auto, auto) {}, 4,
				metal4::clause_cleanup_owner::rsx_thread_end, true,
				[] { throw std::range_error("expected completion wait failure"); });
		}
		catch (const std::range_error&) { wait_failed = true; }
		require(wait_failed && blocked.dependencies == std::vector<unsigned>{0} && !blocked.cleanup &&
			!incomplete.rendering && incomplete.index == 0);
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
		std::cout << "Native predicated draw clauses apply first and later dependencies before submission and retain one RSX cleanup\n";
		std::cout << "Owned clause completion: inline inputs and empty clauses do not wait; six ordinary/trailing-range profiles retain dependency boundary and normalized cursor\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
