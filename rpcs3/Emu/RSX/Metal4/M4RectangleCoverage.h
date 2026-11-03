#pragma once

#include "M4ClearPlan.h"
#include <span>
#include <vector>
#include <stdexcept>

namespace metal4
{
inline constexpr std::size_t rectangle_coverage_limit=65536;
// Exact axis-aligned union proof,without pixel/area arithmetic. Rectangle
// edges are compressed;the sweep tracks covered X length for every Y slab.
// Oversized metadata returns conservative false,not a guessed full overwrite.
[[nodiscard]] inline bool rectangles_cover(std::span<const clear_rectangle> rectangles,std::size_t width,std::size_t height)
{
	if(!width||!height)throw std::invalid_argument("Rectangle coverage requires positive dimensions");
	if(rectangles.size()>rectangle_coverage_limit)return false;
	struct event {std::size_t y,left,right;int delta;};
	std::vector<event> events;std::vector<std::size_t> xs{0,width};
	events.reserve(rectangles.size()*2);xs.reserve(rectangles.size()*2+2);
	for(const auto& r:rectangles)
	{
		if(r.x>width||r.y>height||r.width>width-r.x||r.height>height-r.y)
			throw std::invalid_argument("Rectangle coverage exceeds its canvas");
		if(!r.width||!r.height)continue;
		xs.push_back(r.x);xs.push_back(r.x+r.width);
		events.push_back({r.y,r.x,r.x+r.width,1});events.push_back({r.y+r.height,r.x,r.x+r.width,-1});
	}
	if(events.empty())return false;
	std::sort(xs.begin(),xs.end());xs.erase(std::unique(xs.begin(),xs.end()),xs.end());
	std::sort(events.begin(),events.end(),[](const auto& a,const auto& b){return a.y<b.y;});
	std::vector<int> counts(xs.size()*4);std::vector<std::size_t> lengths(xs.size()*4);
	const auto update=[&](auto&& self,std::size_t node,std::size_t left,std::size_t right,std::size_t begin,std::size_t end,int delta)->void
	{
		if(begin<=left&&right<=end)counts[node]+=delta;
		else
		{
			const auto middle=left+(right-left)/2;
			if(begin<middle)self(self,node*2,left,middle,begin,end,delta);
			if(end>middle)self(self,node*2+1,middle,right,begin,end,delta);
		}
		lengths[node]=counts[node]?xs[right]-xs[left]:(right-left==1?0:lengths[node*2]+lengths[node*2+1]);
	};
	std::size_t previous=0;
	for(std::size_t i=0;i<events.size();)
	{
		const auto y=events[i].y;if(y>previous&&lengths[1]!=width)return false;
		do
		{
			const auto& e=events[i++];
			const auto begin=std::lower_bound(xs.begin(),xs.end(),e.left)-xs.begin(),end=std::lower_bound(xs.begin(),xs.end(),e.right)-xs.begin();
			update(update,1,0,xs.size()-1,begin,end,e.delta);
		}while(i<events.size()&&events[i].y==y);
		previous=y;
	}
	return previous==height;
}
} // namespace metal4
