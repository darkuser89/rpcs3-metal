#pragma once

#include "M4GuestSegments.h"
#include "M4BankedTile.h"
#include <algorithm>
#include <functional>
#include <map>
#include <limits>
#include <set>

namespace metal4
{
struct display_storage_source
{
	guest_surface_layout geometry; // Relative to the complete logical storage.
	std::vector<guest_storage_segment> segments;
	bool dirty;
	std::uint64_t epoch;
	std::optional<banked_tile_layout> tile = {};
	// A logical view packs the native raster without bank permutation,but
	// proves active bytes through the original complete allocation segments.
	std::optional<std::size_t> logical_storage_offset = {};
};
struct display_storage_copy
{
	std::size_t target, source, size, input;
	bool operator==(const display_storage_copy&) const = default;
};
struct display_storage_assembly
{
	std::vector<display_storage_copy> copies;
	std::vector<guest_surface_range> vm;
};
struct display_storage_epoch_run
{
	std::size_t size;
	std::optional<std::uint64_t> epoch;
	bool dirty;
};
// Optional per-source subdivision of an exact matched physical run.
using display_storage_epochs = std::function<display_storage_epoch_run(std::size_t input, std::size_t source, std::size_t size)>;
// Same main-memory bank permutation as the native bank transfer. Address
// bits0..4 are preserved, so active byte runs are projected in32-byte groups
// rather than visiting every texel. Local VRAM never enters this mapping.
[[nodiscard]] inline std::uint32_t banked_display_storage_offset(const banked_tile_layout& tile,std::size_t logical)
{
	if(logical>=tile.size)throw std::out_of_range("Display bank address exceeds allocation");
	const auto factor=banked_tile_factor(tile.pitch),columns=tile.pitch/256;
	const auto address=tile.address+static_cast<std::uint32_t>(logical),texel=static_cast<std::uint32_t>(logical)/256;
	const auto tx=texel%columns,ty=(texel/columns)/64,selector=(ty*columns+tx+(tile.address>>14))&0x3ffff;
	constexpr std::array<unsigned,16> banks{0,1,2,3,2,3,0,1,1,2,3,0,3,0,1,2};
	auto bank=selector&3;
	if(factor==2)bank=banks[((selector+((ty&1)<<1))&3)*4+(ty&3)];
	else if(factor>=4)bank=banks[(selector&3)*4+(ty&3)];
	bank=(bank+tile.bank)&3;
	const auto line=(texel/columns)%64;
	const auto column=(((line>>3)&7)<<7)|(((address>>5)&7)<<4)|((line&3)<<2);
	const auto partition=(((line>>2)&1)+((address>>6)&1))&1;
	auto physical=(((selector>>2)&65535)<<16)|(bank<<14)|(((column>>4)&63)<<8)|(partition<<7)|(((column>>2)&3)<<5)|(address&31);
	physical^=(((physical>>12)^((bank^selector)&1)^(physical>>14))&1)<<9;
	physical^=((physical>>11)&1)<<10;
	return physical-tile.address;
}
[[nodiscard]] inline std::vector<guest_surface_range> banked_display_storage_ranges(const guest_surface_layout& geometry,
	const banked_tile_layout& tile,std::size_t limit=1u<<20)
{
	if(geometry.range.address)throw std::invalid_argument("Banked display geometry must be relative");
	auto image=geometry;image.range={tile.address,tile.size};validate_banked_tile_geometry(image,tile);
	if(geometry.morton&&(geometry.morton->storage[2]!=1||geometry.morton->origin[2]))throw std::invalid_argument("Banked display requires a2D source slice");
	guest_surface_byte_cursor cursor(geometry);std::vector<guest_surface_range> runs;
	while(const auto active=cursor.next())for(std::size_t position=active->address,end=position+active->size;position<end;)
	{
		const auto logical=tile.offset+position,count=std::min(end-position,32-(logical&31));
		const auto physical=banked_display_storage_offset(tile,logical);
		// The shared RSX writer and native bank kernel leave cells outside the
		// owned allocation untouched. Such native texels own no physical bytes.
		if(physical>=tile.size||count>tile.size-physical){position+=count;continue;}
		if(runs.size()==limit)throw std::length_error("Banked display projection exceeds run budget");
		runs.push_back({physical,count});position+=count;
	}
	std::sort(runs.begin(),runs.end(),[](auto a,auto b){return a.address<b.address;});std::size_t kept=0;
	for(const auto run:runs)
	{
		if(kept&&std::uint64_t(runs[kept-1].address)+runs[kept-1].size>run.address)throw std::invalid_argument("Banked native bytes have ambiguous physical storage");
		if(kept&&std::uint64_t(runs[kept-1].address)+runs[kept-1].size==run.address)runs[kept-1].size+=run.size;
		else runs[kept++]=run;
	}
	runs.resize(kept);return runs;
}
[[nodiscard]] inline std::vector<guest_storage_part> project_display_storage_source(const display_storage_source& source,std::size_t limit=1u<<20)
{
	if(source.geometry.range.address||source.segments.empty())throw std::invalid_argument("Display source lacks ordered storage");
	if(source.logical_storage_offset)
	{
		if(source.tile)throw std::invalid_argument("Logical native storage cannot also request physical bank packing");
		const auto& tail=source.segments.back();if(tail.size>std::numeric_limits<std::size_t>::max()-tail.offset)throw std::overflow_error("Logical source allocation overflow");const auto bytes=tail.offset+tail.size;
		auto allocation=guest_byte_coverage({0,bytes});allocation.range.address=source.segments.front().address;validate_guest_segments(allocation,source.segments);
		std::vector<guest_storage_part> result;guest_surface_byte_cursor cursor(source.geometry);
		while(const auto run=cursor.next())for(std::size_t at=run->address,remaining=run->size;remaining;)
		{
			if(at>std::numeric_limits<std::size_t>::max()-*source.logical_storage_offset)throw std::overflow_error("Logical source offset overflow");const auto offset=*source.logical_storage_offset+at;const auto end=std::upper_bound(source.segments.begin(),source.segments.end(),offset,[](auto value,const auto& segment){return value<segment.offset;});
			if(end==source.segments.begin())throw std::invalid_argument("Logical native bytes precede their owned segments");const auto& segment=*std::prev(end);
			if(offset-segment.offset>=segment.size)throw std::invalid_argument("Logical native bytes exceed their owned allocation");const auto count=std::min(remaining,segment.size-(offset-segment.offset));
			if(result.size()==limit)throw std::length_error("Logical native storage exceeds projection budget");
			result.push_back({guest_byte_coverage({static_cast<std::uint32_t>(std::uint64_t(segment.address)+offset-segment.offset),count}),at});at+=count;remaining-=count;
		}
		return result;
	}
	auto storage=source.geometry;storage.range.address=source.segments.front().address;
	if(!source.tile)return project_guest_segments(storage,source.segments,false,limit);
	if(source.tile->address!=source.segments.front().address)throw std::invalid_argument("Display bank base differs from first physical segment");
	storage.range.size=source.tile->size;validate_guest_segments(storage,source.segments);
	const auto active=banked_display_storage_ranges(source.geometry,*source.tile,limit);std::vector<guest_storage_part> result;std::size_t first=0;
	for(const auto run:active)
	{
		while(first<source.segments.size()&&source.segments[first].offset+source.segments[first].size<=run.address)++first;
		for(auto i=first;i<source.segments.size()&&source.segments[i].offset<std::uint64_t(run.address)+run.size;++i)
		{
			const auto& segment=source.segments[i];const auto begin=std::max<std::size_t>(run.address,segment.offset),end=std::min<std::size_t>(std::size_t(run.address)+run.size,segment.offset+segment.size);
			if(result.size()==limit)throw std::length_error("Banked display segments exceed run budget");
			result.push_back({guest_byte_coverage({static_cast<std::uint32_t>(std::uint64_t(segment.address)+begin-segment.offset),end-begin}),begin});
		}
	}
	return result;
}
// Exact physical intersections, followed by a byte-authority sweep. No pixel
// data is mapped or copied here. Weak allocation identities prevent recycled
// addresses from masquerading as old storage. Caller owns writer exclusion.
[[nodiscard]] inline display_storage_assembly plan_display_storage_assembly(guest_surface_memory& memory,
	const guest_surface_layout& display, std::span<const display_storage_source> inputs,
	std::span<const guest_storage_segment> requested_segments = {}, const display_storage_epochs& epochs = {})
{
	if (display.morton || display.column_stride || inputs.empty() || inputs.size() > 64)
		throw std::invalid_argument("Display assembly requires linear AV bytes and one to64 native sources");
	(void)guest_surface_byte_cursor(display);
	constexpr std::size_t limit = 1u << 20;
	struct slice { std::size_t offset, logical, size, input, maximum = 0; };
	using owner = std::weak_ptr<void>;
	std::map<owner,std::vector<slice>,std::owner_less<owner>> domains;
	const auto direct_domain = std::make_shared<int>(0);
	bool direct = false; memory.test_storage([&](bool value,const auto&) { direct = value; return false; });
	std::size_t slices = 0;
	const auto identities = [&](guest_surface_range range)
	{
		auto result = memory.storage_identity(range);
		if (!result && direct) result = std::vector<storage_identity_span>{{direct_domain,range.address,range}};
		if (!result) throw std::invalid_argument("Display assembly requires exact physical storage metadata");
		std::uint64_t cursor = range.address;
		for (const auto& part : *result)
		{
			if (++slices > limit) throw std::length_error("Display assembly exceeds identity budget");
			if (part.allocation.expired() || !part.guest.size || part.guest.address != cursor ||
				part.guest.size > std::uint64_t(range.address) + range.size - cursor ||
				part.offset > std::numeric_limits<std::size_t>::max() - part.guest.size)
				throw std::invalid_argument("Display identity slices are incomplete or invalid");
			cursor += part.guest.size;
		}
		if (cursor != std::uint64_t(range.address) + range.size) throw std::invalid_argument("Display identity has a gap");
		return *result;
	};
	for (std::size_t input = 0; input < inputs.size(); ++input)
	{
		const auto& source = inputs[input];
		const auto parts = project_display_storage_source(source,limit);
		for (const auto& part : parts)
		{
			guest_surface_byte_cursor cursor(part.coverage);
			while (const auto active = cursor.next())
				for (const auto& identity : identities(*active))
					domains[identity.allocation].push_back({identity.offset,part.offset + identity.guest.address - part.coverage.range.address,identity.guest.size,input});
		}
	}
	for (auto& [allocation,parts] : domains)
	{
		std::sort(parts.begin(),parts.end(),[](const auto& a,const auto& b) { return a.offset < b.offset; });
		const auto build = [&](auto&& self,std::size_t first,std::size_t last) -> std::size_t
		{
			if (first == last) return 0;
			const auto middle = first + (last-first)/2;
			return parts[middle].maximum = std::max({parts[middle].offset+parts[middle].size,self(self,first,middle),self(self,middle+1,last)});
		};
		build(build,0,parts.size());
	}
	// RTT planes provide a dense canonical payload whose physical addresses
	// can interleave Morton Z cells or cross unrelated IO/alias allocations.
	// Build source domains once,then project all destination runs through them.
	std::vector<guest_storage_part> requested_parts;
	if(requested_segments.empty())requested_parts.push_back({display,0});
	else
	{
		if(requested_segments.size()>limit||display.width*display.pixel_bytes!=display.row_pitch||display.range.size!=display.row_pitch*display.height)
			throw std::invalid_argument("Segmented assembly target must be dense and bounded");
		std::size_t position=0;
		for(const auto& segment:requested_segments)
		{
			validate_guest_surface_range({segment.address,segment.size});
			if(!segment.size||segment.offset!=position||segment.size>display.range.size-position)throw std::invalid_argument("Segmented assembly target has a gap oroverlap");
			requested_parts.push_back({guest_byte_coverage({segment.address,segment.size}),segment.offset});position+=segment.size;
		}
		if(position!=display.range.size)throw std::invalid_argument("Segmented assembly target is incomplete");
	}
	struct match : display_storage_copy { std::uint64_t epoch; bool dirty; };
	std::vector<match> matches;
	for(const auto& requested_part:requested_parts)
	{
	guest_surface_byte_cursor requested(requested_part.coverage);
	while (const auto active = requested.next())
		for (const auto& target : identities(*active))
		{
			const auto found = domains.find(target.allocation); if (found == domains.end()) continue;
			const auto& parts = found->second;
			const auto query = [&](auto&& self,std::size_t first,std::size_t last) -> void
			{
				if (first == last) return;
				const auto middle = first + (last-first)/2; const auto& source = parts[middle];
				const auto end = target.offset + target.guest.size;
				if (source.maximum <= target.offset || parts[first].offset >= end) return;
				self(self,first,middle);
				const auto begin = std::max(source.offset,target.offset), finish = std::min(source.offset+source.size,end);
			for (auto at = begin; at < finish;)
				{
				const auto offset = source.logical + at - source.offset;
				const auto remaining = finish - at;
				const auto& input = inputs[source.input];
				const auto run = epochs ? epochs(source.input, offset, remaining) : display_storage_epoch_run{remaining, input.epoch, input.dirty};
				if (!run.size || run.size > remaining)
					throw std::logic_error("Display assembly epoch subdivision exceeds its match");
				if (run.epoch)
				{
					if (matches.size() == limit) throw std::length_error("Display assembly exceeds intersection budget");
					matches.push_back({{requested_part.offset+std::size_t(target.guest.address-requested_part.coverage.range.address)+at-target.offset,
						offset,run.size,source.input},*run.epoch,run.dirty});
				}
				at += run.size;
				}
				if (source.offset < end) self(self,middle+1,last);
			};
			query(query,0,parts.size());
		}
	}
	struct event { std::size_t position,index; bool start; };
	std::vector<event> events; events.reserve(matches.size()*2);
	for (std::size_t i=0;i<matches.size();++i) { events.push_back({matches[i].target,i,true}); events.push_back({matches[i].target+matches[i].size,i,false}); }
	std::sort(events.begin(),events.end(),[](const auto& a,const auto& b) { return a.position < b.position; });
	const auto order = [&](std::size_t ai,std::size_t bi)
	{
		const auto& a = matches[ai]; const auto& b = matches[bi];
		if (a.dirty != b.dirty) return a.dirty > b.dirty;
		if (matches[ai].epoch != matches[bi].epoch) return matches[ai].epoch > matches[bi].epoch;
		return ai < bi;
	};
	std::set<std::size_t,decltype(order)> live(order);
	display_storage_assembly result;
	for (std::size_t i=0;i<events.size();)
	{
		const auto position = events[i].position;
		while (i<events.size() && events[i].position==position)
		{
			if (events[i].start) live.insert(events[i].index); else live.erase(events[i].index); ++i;
		}
		if (live.empty() || i==events.size()) continue;
		const auto& best = matches[*live.begin()]; const auto source = best.source+position-best.target;
		for (auto it=std::next(live.begin());it!=live.end();++it)
		{
			const auto& next = matches[*it];
			if (best.dirty!=next.dirty || best.epoch!=next.epoch) break;
			if (best.input!=next.input || source!=next.source+position-next.target)
				throw std::invalid_argument("Display assembly lacks a unique native byte authority");
		}
		const auto count = events[i].position-position;
		if (!result.copies.empty() && result.copies.back().input==best.input &&
			result.copies.back().target+result.copies.back().size==position && result.copies.back().source+result.copies.back().size==source)
			result.copies.back().size+=count;
		else result.copies.push_back({position,source,count,best.input});
	}
	guest_surface_byte_cursor cursor(display);
	std::size_t next = 0;
	while (const auto active = cursor.next())
	{
		std::size_t position=active->address-display.range.address,end=position+active->size;
		while (position<end)
		{
			while (next<result.copies.size() && result.copies[next].target+result.copies[next].size<=position) ++next;
			if (next<result.copies.size() && result.copies[next].target<=position) position=std::min(end,result.copies[next].target+result.copies[next].size);
			else
			{
				const auto finish=next<result.copies.size()?std::min(end,result.copies[next].target):end;
				if (result.vm.size()==limit) throw std::length_error("Display assembly exceeds VM interval budget");
				result.vm.push_back({static_cast<std::uint32_t>(position),finish-position});position=finish;
			}
		}
	}
	return result;
}
} // namespace metal4
