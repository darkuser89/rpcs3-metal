#pragma once

#include "M4CPUCapture.h"
#include <utility>

namespace metal4
{
// The GPU has already completed. Keep its owned output if a CPU dispatcher
// refuses entry before invoking its checked action. Once an action starts,
// its exception is terminal: a writer may already have changed guest bytes.
// Publication may validate in one capture and store in a later capture; it
// must not dispatch another guarded action after a successful store.
template<typename Pending,typename Publish>
void publish_completed_transfer(std::unique_ptr<Pending>& pending,
	const cpu_capture_dispatch& dispatch,Publish&& publish)
{
	if(!pending)throw std::logic_error("Completed transfer publication lacks an owner");
	bool refused=false;
	// Allocate the wrapper before moving ownership, so allocation failure also
	// leaves the completed GPU output available to its original owner.
	const cpu_capture_dispatch capture=[&](const auto& operation)
	{
		bool entered=false;refused=false;
		try{checked_cpu_capture(dispatch,[&]{entered=true;operation();});}
		catch(...){refused=!entered;throw;}
	};
	auto completed=std::move(pending);
	try{std::invoke(std::forward<Publish>(publish),*completed,capture);}
	catch(...)
	{
		if(refused)pending=std::move(completed);
		throw;
	}
}
} // namespace metal4
