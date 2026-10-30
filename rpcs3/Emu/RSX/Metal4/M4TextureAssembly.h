#pragma once

#include "M4DisplayAssembly.h"
#include "M4GuestTextureLayout.h"
#include "M4SurfaceTextureRegion.h"

namespace metal4
{
// One canonical2D block raster,with exact real EAs for its original mip,
// cube face or volume slice. Gaps between rows/faces/Z cells are not pixels.
struct texture_assembly_plane
{
	guest_texture_layout view;
	std::vector<guest_storage_segment> spans;
	unsigned level,face;
	std::size_t z;
	std::uint32_t logical;
	std::vector<guest_storage_segment> logical_spans = {};
};
// Inverse metadata for banked native cells. This maps addresses only; no
// guest/native pixel data is read. Partial texels must prove every byte.
struct texture_storage_run {std::size_t physical,logical,size;};
[[nodiscard]] inline std::vector<texture_storage_run> texture_storage_runs(const display_storage_source& source)
{
	if(source.geometry.range.address)throw std::invalid_argument("Native texture inverse geometry must be relative");
	std::vector<texture_storage_run> result;guest_surface_byte_cursor cursor(source.geometry);
	while(const auto run=cursor.next())for(std::size_t at=run->address,end=at+run->size;at<end;)
	{
		const auto logical=at+(source.tile?source.tile->offset:0),count=source.tile?std::min(end-at,32-(logical&31)):end-at;
		const auto physical=source.tile?banked_display_storage_offset(*source.tile,logical):at;
		if(!source.tile||(physical<source.tile->size&&count<=source.tile->size-physical))
		{
			if(result.size()==(1u<<20))throw std::length_error("Native texture inverse metadata exceeds run budget");
			result.push_back({physical,at,count});
		}
		at+=count;
	}
	std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.physical<b.physical;});
	for(std::size_t i=1;i<result.size();++i)if(result[i-1].physical+result[i-1].size>result[i].physical)throw std::invalid_argument("Native texture inverse storage is ambiguous");
	return result;
}
[[nodiscard]] inline std::optional<std::size_t> texture_storage_logical(std::span<const texture_storage_run> runs,std::size_t physical)
{
	const auto end=std::upper_bound(runs.begin(),runs.end(),physical,[](auto value,const auto& run){return value<run.physical;});
	if(end==runs.begin())return {};const auto& run=*std::prev(end);return physical-run.physical<run.size?std::optional{run.logical+physical-run.physical}:std::nullopt;
}
[[nodiscard]] inline std::optional<std::array<std::size_t,2>> texture_storage_pixel(const guest_surface_layout& source,std::size_t logical)
{
	const auto bpp=source.pixel_bytes;if(source.range.address||source.column_stride||!bpp||!source.row_pitch||logical>=source.range.size)return {};std::array<std::size_t,3> at{};
	if(!source.morton){at={logical%source.row_pitch/bpp,logical/source.row_pitch,0};if((logical%source.row_pitch)%bpp)return {};}
	else
	{
		if(logical%bpp)return {};const auto shape=plan_swizzle(*source.morton,bpp).log2;unsigned output=0;
		for(unsigned bit=0;bit<std::max({shape[0],shape[1],shape[2]});++bit)for(unsigned axis=0;axis<3;++axis)if(bit<shape[axis])at[axis]|=(((logical/bpp)>>output++)&1)<<bit;
		for(unsigned axis=0;axis<3;++axis){if(at[axis]<source.morton->origin[axis])return {};at[axis]-=source.morton->origin[axis];}
	}
	if(at[0]>=source.width||at[1]>=source.height||at[2])return {};return std::array{at[0],at[1]};
}
[[nodiscard]] inline const guest_storage_segment* texture_assembly_segment(std::span<const guest_storage_segment> parts,std::size_t offset)
{
	const auto end=std::upper_bound(parts.begin(),parts.end(),offset,[](auto value,const auto& part){return value<part.offset;});
	if(end==parts.begin())return nullptr;const auto& part=*std::prev(end);return offset-part.offset<part.size?&part:nullptr;
}
[[nodiscard]] inline std::vector<texture_assembly_plane> plan_texture_assembly_planes(const guest_texture_layout& plan,
	std::span<const guest_storage_segment> segments,std::size_t run_limit=1u<<20)
{
	if(plan.stored_border||plan.adjusted_pitch||segments.empty()||plan.subresources.empty())throw std::invalid_argument("RTT assembly lacks supported owned texture storage");
	auto storage=guest_byte_coverage({0,plan.storage_bytes});storage.range.address=segments.front().address;validate_guest_segments(storage,segments);
	std::vector<texture_assembly_plane> result;std::size_t runs=0;
	for(const auto& s:plan.subresources)
	{
		const bool morton=plan.swizzled&&plan.codec.block.width==1;
		const auto shape=morton?plan_swizzle({{s.columns,s.rows,s.depth},{},{s.columns,s.rows,s.depth}},plan.codec.block.bytes).log2:std::array<std::uint32_t,3>{};
		const auto row_bytes=checked_texture_product(s.columns,plan.codec.block.bytes),image_bytes=checked_texture_product(row_bytes,s.rows);
		const auto address=[&](std::size_t x,std::size_t y,std::size_t z)
		{
			if(guest_texture_vtc(plan))return guest_texture_block_offset(plan,s,x,y,z);
			return checked_texture_sum(s.offset,morton?checked_texture_product(morton_index({x,y,z},shape),plan.codec.block.bytes):
				checked_texture_sum(checked_texture_product(checked_texture_sum(checked_texture_product(z,s.storage_rows),y),s.row_pitch),checked_texture_product(x,plan.codec.block.bytes)));
		};
		for(std::size_t z=0;z<s.depth;++z)
		{
			if(result.size()==4096)throw std::length_error("RTT assembly exceeds its slice bound");
			auto view=plan;view.width=s.width;view.height=s.height;view.depth=view.layers=view.mips=1;view.dimension=vertex_texture_dimension::two;view.swizzled=false;
			view.storage_bytes=image_bytes;view.offset=static_cast<std::uint32_t>(std::uint64_t(plan.offset)+address(0,0,z));
			view.subresources={{0,image_bytes,row_bytes,s.width,s.height,1,s.columns,s.rows,s.rows,0,0,0}};
			texture_assembly_plane plane{std::move(view),{},s.level,s.face,z,static_cast<std::uint32_t>(std::uint64_t(plan.offset)+address(0,0,z))};
			for(std::size_t y=0;y<s.rows;++y)for(std::size_t x=0;x<s.columns;)
			{
				const auto columns=guest_texture_vtc(plan)?1:morton&&((s.rows>1)||(s.depth>1))?std::min<std::size_t>(s.columns-x,s.columns>1?2:1):s.columns-x;
				auto original=address(x,y,z),payload=y*row_bytes+x*plan.codec.block.bytes,remaining=columns*plan.codec.block.bytes;
				while(remaining)
				{
					if(++runs>run_limit)throw std::length_error("RTT assembly exceeds its physical run budget");
					const auto* segment=texture_assembly_segment(segments,original);if(!segment)throw std::invalid_argument("RTT slice addresses missing storage");
					const auto count=std::min(remaining,segment->size-(original-segment->offset));const auto ea=std::uint64_t(segment->address)+original-segment->offset;
					const auto logical=std::uint64_t(plan.offset)+original;
					if(logical>0xffffffffu||count>0x100000000ull-logical)throw std::out_of_range("RTT logical span wraps guest offsets");
					if(!plane.logical_spans.empty()&&plane.logical_spans.back().offset+plane.logical_spans.back().size==payload&&std::uint64_t(plane.logical_spans.back().address)+plane.logical_spans.back().size==logical)plane.logical_spans.back().size+=count;
					else plane.logical_spans.push_back({static_cast<std::uint32_t>(logical),payload,count});
					if(!plane.spans.empty()&&plane.spans.back().offset+plane.spans.back().size==payload&&std::uint64_t(plane.spans.back().address)+plane.spans.back().size==ea)plane.spans.back().size+=count;
					else plane.spans.push_back({static_cast<std::uint32_t>(ea),payload,count});
					original+=count;payload+=count;remaining-=count;
				}
				x+=columns;
			}
			result.push_back(std::move(plane));
		}
	}
	return result;
}
// Prove a complete native rectangle against the plane's actual active-byte
// order. Unlike a2D storage-layout guess,this also handles interleaved3D Z.
// A banked logical RTT alias requires the original caller-supplied anchor;
// its native snapshot retains the original complete bank allocation receipt.
[[nodiscard]] inline std::optional<surface_texture_region> plan_native_texture_plane(guest_surface_memory& memory,
	const display_storage_source& source,const texture_assembly_plane& plane,std::optional<std::size_t> logical_anchor={})
{
	if(source.geometry.range.address||source.segments.empty()||plane.spans.empty()||(source.tile&&!logical_anchor))return {};
	const auto& g=source.geometry;const auto bpp=g.pixel_bytes,row=plane.view.subresources.at(0).row_pitch,rows=plane.view.subresources[0].rows;
	if(!bpp||!row||!rows)return {};
	(void)guest_surface_byte_cursor(g);auto storage=guest_byte_coverage({0,source.tile?source.tile->size:g.range.size});storage.range.address=source.segments.front().address;validate_guest_segments(storage,source.segments);
	const auto shape=g.morton?plan_swizzle(*g.morton,bpp).log2:std::array<std::uint32_t,3>{};
	std::optional<surface_texture_region> result;bool ambiguous=false;std::size_t comparisons=0;
	const auto prove=[&](const guest_surface_memory::storage_rebase& rebase,const guest_surface_memory::storage_equivalent& equal)
	{
		const auto candidate=[&](std::size_t anchor)
		{
			if(anchor>=g.range.size)return;const auto prefix=(g.morton?anchor:anchor%g.row_pitch)%bpp;std::size_t sx=0,sy=0;
			if(!g.morton){sx=(anchor%g.row_pitch)/bpp;sy=anchor/g.row_pitch;}
			else
			{
				std::array<std::size_t,3> at{};std::size_t output=0;for(unsigned bit=0;bit<std::max({shape[0],shape[1],shape[2]});++bit)for(unsigned axis=0;axis<3;++axis)if(bit<shape[axis])at[axis]|=(((anchor/bpp)>>output++)&1)<<bit;
				if(at[0]<g.morton->origin[0]||at[1]<g.morton->origin[1]||at[2]!=g.morton->origin[2])return;sx=at[0]-g.morton->origin[0];sy=at[1]-g.morton->origin[1];
			}
			const auto width=(prefix+row)/bpp+bool((prefix+row)%bpp);if(sx>g.width||width>g.width-sx||sy>g.height||rows>g.height-sy)return;
			for(const auto& span:plane.spans)for(std::size_t position=span.offset,remaining=span.size;remaining;)
			{
				if(++comparisons>(1u<<20))throw std::length_error("Native RTT rectangle proof exceeds run budget");
				const auto y=position/row,byte=prefix+position%row,x=sx+byte/bpp,lane=byte%bpp;
				const auto raw=g.morton?morton_index({x+g.morton->origin[0],sy+y+g.morton->origin[1],g.morton->origin[2]},shape)*bpp+lane:(sy+y)*g.row_pitch+x*bpp+lane;
				const auto offset=checked_texture_sum(raw,source.tile?source.tile->offset:0);const auto* segment=texture_assembly_segment(source.segments,offset);if(!segment)return;
				const auto available=g.morton?bpp-lane:row-position%row;
				const auto count=std::min({remaining,row-position%row,available,segment->size-(offset-segment->offset)});
				if(!equal({static_cast<std::uint32_t>(std::uint64_t(segment->address)+offset-segment->offset),count},{static_cast<std::uint32_t>(std::uint64_t(span.address)+position-span.offset),count}))return;
				position+=count;remaining-=count;
			}
			const surface_texture_region value{{sx,sy,width,rows},prefix};
			if(result&&(result->pixels.x!=sx||result->pixels.y!=sy||result->pixels.width!=width||result->pixels.height!=rows||result->guest_row_offset!=prefix))ambiguous=true;result=value;
		};
		if(logical_anchor)candidate(*logical_anchor);
		else for(const auto& segment:source.segments)if(const auto rebased=rebase({segment.address,segment.size},{plane.spans.front().address,1}))candidate(segment.offset+rebased->address-segment.address);
		return bool(result)&&!ambiguous;
	};
	(void)memory.test_rebase(prove);return ambiguous?std::nullopt:result;
}
// A target may consume both logical pixels of a bound banked tile and raw
// physical background bytes. Keep those views distinct while sharing each
// immutable native source. Logical claims must prove the original IO byte
// correspondence;they override only the same source's raw interpretation.
struct texture_storage_assembly
{
	display_storage_assembly bytes;
	std::vector<display_storage_source> layouts;
	std::vector<std::size_t> sources;
};
[[nodiscard]] inline texture_storage_assembly plan_texture_storage_assembly(guest_surface_memory& memory,
	const texture_assembly_plane& plane,std::span<const display_storage_source> inputs,
	std::span<const std::optional<std::uint32_t>> logical_origins)
{
	if(inputs.empty()||inputs.size()>64||logical_origins.size()!=inputs.size()||plane.logical_spans.empty())throw std::invalid_argument("RTT logical assembly lacks bounded source/address metadata");
	constexpr std::size_t limit=1u<<20;
	if(plane.logical_spans.size()>limit)throw std::length_error("Logical RTT target exceeds span budget");
	std::size_t payload=0;for(const auto& span:plane.logical_spans){validate_guest_surface_range({span.address,span.size});if(!span.size||span.offset!=payload||span.size>plane.view.storage_bytes-payload)throw std::invalid_argument("Logical RTT target has gaps oroverlaps");payload+=span.size;}
	if(payload!=plane.view.storage_bytes)throw std::invalid_argument("Logical RTT target is incomplete");
	const auto& sub=plane.view.subresources.at(0);const auto target=plan_guest_surface(0,sub.columns,sub.rows,plane.view.codec.block.bytes,sub.row_pitch);
	texture_storage_assembly result;result.bytes=plan_display_storage_assembly(memory,target,inputs,plane.spans);result.layouts.assign(inputs.begin(),inputs.end());
	for(std::size_t i=0;i<inputs.size();++i)result.sources.push_back(i);
	if(std::none_of(logical_origins.begin(),logical_origins.end(),[](const auto& p){return p.has_value();}))return result;
	struct claim {display_storage_copy copy;std::size_t owner;bool logical;};
	std::vector<claim> claims;for(const auto& copy:result.bytes.copies)claims.push_back({copy,copy.input,false});
	std::size_t proof_runs=0;
	for(std::size_t input=0;input<inputs.size();++input)if(logical_origins[input])
	{
		const auto& source=inputs[input];if(!source.tile||source.logical_storage_offset)throw std::invalid_argument("Logical RTT interpretation requires the original banked source");
		auto logical=source;logical.logical_storage_offset=source.tile->offset;logical.tile.reset();
		// Validate actual ownership of every active native byte,including a
		// last row whose unused padding lies outside the tile allocation.
		(void)project_display_storage_source(logical);
		std::vector<guest_surface_range> active;guest_surface_byte_cursor cursor(source.geometry);while(const auto run=cursor.next()){if(active.size()==limit)throw std::length_error("Logical RTT active metadata exceeds budget");active.push_back(*run);}
		std::optional<std::size_t> view;
		for(const auto& span:plane.logical_spans)
		{
			const auto origin=std::uint64_t(*logical_origins[input]),end=std::uint64_t(span.address)+span.size;if(end<=origin)continue;
			const auto first=std::max<std::uint64_t>(span.address,origin)-origin,last=end-origin;
			auto run=std::lower_bound(active.begin(),active.end(),first,[](auto range,auto value){return std::uint64_t(range.address)+range.size<=value;});
			for(;run!=active.end()&&run->address<last;++run)
			{
				const auto begin=std::max<std::size_t>(first,run->address),finish=std::min<std::size_t>(last,std::uint64_t(run->address)+run->size);if(begin==finish)continue;
				if(!view){view=result.layouts.size();result.layouts.push_back(logical);result.sources.push_back(input);}
				const display_storage_copy copy{span.offset+origin+begin-span.address,begin,finish-begin,*view};
				const auto prove=[&](const guest_surface_memory::storage_rebase&,const guest_surface_memory::storage_equivalent& equal)
				{
					for(std::size_t position=0;position<copy.size;)
					{
						if(++proof_runs>limit)throw std::length_error("Logical RTT IO proof exceeds run budget");
						const auto src=checked_texture_sum(*logical.logical_storage_offset,copy.source+position),dst=copy.target+position;
						const auto* a=texture_assembly_segment(source.segments,src);const auto* b=texture_assembly_segment(plane.spans,dst);if(!a||!b)return false;
						const auto count=std::min({copy.size-position,a->size-(src-a->offset),b->size-(dst-b->offset)});
						if(!equal({static_cast<std::uint32_t>(std::uint64_t(a->address)+src-a->offset),count},{static_cast<std::uint32_t>(std::uint64_t(b->address)+dst-b->offset),count}))return false;
						position+=count;
					}
					return true;
				};
				if(!memory.test_rebase(prove))throw std::invalid_argument("Logical RTT bytes differ from the original mapped storage");
				if(claims.size()==limit)throw std::length_error("Logical RTT combined claim budget exceeded");claims.push_back({copy,input,true});
			}
		}
	}
	struct event {std::size_t position,index;bool start;};std::vector<event> events;events.reserve(claims.size()*2);
	for(std::size_t i=0;i<claims.size();++i){events.push_back({claims[i].copy.target,i,true});events.push_back({claims[i].copy.target+claims[i].copy.size,i,false});}
	std::sort(events.begin(),events.end(),[](auto a,auto b){return a.position<b.position;});
	const auto order=[&](std::size_t ai,std::size_t bi)
	{
		const auto& a=claims[ai];const auto& b=claims[bi];const auto& x=inputs[a.owner];const auto& y=inputs[b.owner];
		if(x.dirty!=y.dirty)return x.dirty>y.dirty;if(x.epoch!=y.epoch)return x.epoch>y.epoch;if(a.owner!=b.owner)return a.owner<b.owner;if(a.logical!=b.logical)return a.logical>b.logical;return ai<bi;
	};
	std::set<std::size_t,decltype(order)> live(order);result.bytes={};
	for(std::size_t i=0;i<events.size();)
	{
		const auto position=events[i].position;while(i<events.size()&&events[i].position==position){if(events[i].start)live.insert(events[i].index);else live.erase(events[i].index);++i;}
		if(live.empty()||i==events.size())continue;
		const auto& best=claims[*live.begin()];const auto from=best.copy.source+position-best.copy.target;
		for(auto it=std::next(live.begin());it!=live.end();++it)
		{
			const auto& next=claims[*it];const auto& a=inputs[best.owner];const auto& b=inputs[next.owner];if(a.dirty!=b.dirty||a.epoch!=b.epoch)break;
			if(best.owner!=next.owner||(best.logical==next.logical&&from!=next.copy.source+position-next.copy.target))throw std::invalid_argument("Logical RTT lacks a unique native byte authority");
		}
		const auto count=events[i].position-position;if(!count)continue;
		if(!result.bytes.copies.empty()&&result.bytes.copies.back().input==best.copy.input&&result.bytes.copies.back().target+result.bytes.copies.back().size==position&&result.bytes.copies.back().source+result.bytes.copies.back().size==from)result.bytes.copies.back().size+=count;
		else result.bytes.copies.push_back({position,from,count,best.copy.input});
	}
	std::size_t position=0;for(const auto& copy:result.bytes.copies){if(copy.target>position)result.bytes.vm.push_back({static_cast<std::uint32_t>(position),copy.target-position});position=copy.target+copy.size;}
	if(position<target.range.size)result.bytes.vm.push_back({static_cast<std::uint32_t>(position),target.range.size-position});
	return result;
}
} // namespace metal4
