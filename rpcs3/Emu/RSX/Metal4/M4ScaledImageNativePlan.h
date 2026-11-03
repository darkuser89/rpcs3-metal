#pragma once

#include "M4ScaledImagePlan.h"
#include "M4TextureAssembly.h"

namespace metal4
{
// Address metadata only. Geometry is relative to its complete allocation;
// segments retain the real EAs,including split pixels and physical mirrors.
[[nodiscard]] inline display_storage_source scaled_image_storage(const guest_surface_layout& image,
	const std::optional<banked_tile_layout>& tile,std::span<const guest_storage_segment> segments)
{
	auto geometry=plan_guest_surface(0,image.width,image.height,image.pixel_bytes,image.morton?0:image.row_pitch,image.morton);
	std::vector<guest_storage_segment> storage(segments.begin(),segments.end());
	if(storage.empty())storage.push_back({image.range.address,0,image.range.size});
	return {std::move(geometry),std::move(storage),false,0,tile};
}
[[nodiscard]] inline std::optional<std::size_t> scaled_image_storage_offset(const display_storage_source& source,
	std::size_t x,std::size_t y,std::size_t lane=0)
{
	const auto& g=source.geometry;
	if(x>=g.width||y>=g.height||lane>=g.pixel_bytes)return {};
	const auto raw=g.morton?checked_texture_sum(checked_texture_product(morton_index(
		{x+g.morton->origin[0],y+g.morton->origin[1],g.morton->origin[2]},plan_swizzle(*g.morton,g.pixel_bytes).log2),g.pixel_bytes),lane):
		checked_texture_sum(checked_texture_sum(checked_texture_product(y,g.row_pitch),checked_texture_product(x,g.pixel_bytes)),lane);
	if(!source.tile)return raw;
	const auto logical=checked_texture_sum(source.tile->offset,raw);
	if(logical>=source.tile->size)return {};
	const auto physical=banked_display_storage_offset(*source.tile,logical);
	return physical<source.tile->size?std::optional<std::size_t>{physical}:std::nullopt;
}
// Prove the complete requested raster against one native rectangle. The
// first matching address is only a candidate:every byte is checked through
// one stable physical-identity callback. No guest or native pixels are read.
[[nodiscard]] inline std::optional<clear_rectangle> plan_native_scaled_image_region(guest_surface_memory& memory,
	const display_storage_source& source,const display_storage_source& requested)
{
	const auto& a=source.geometry;const auto& b=requested.geometry;
	if(a.range.address||b.range.address||source.segments.empty()||requested.segments.empty()||
		a.pixel_bytes!=b.pixel_bytes||b.width>a.width||b.height>a.height)return {};
	const auto inverse=texture_storage_runs(source);
	const auto first=scaled_image_storage_offset(requested,0,0);
	if(!first)return {};
	const auto* first_segment=texture_assembly_segment(requested.segments,*first);if(!first_segment)return {};
	const guest_surface_range initial{static_cast<std::uint32_t>(std::uint64_t(first_segment->address)+*first-first_segment->offset),1};
	std::optional<clear_rectangle> result;bool ambiguous=false;std::size_t checks=0;
	const auto prove=[&](const guest_surface_memory::storage_rebase& rebase,const guest_surface_memory::storage_equivalent& equal)
	{
		for(const auto& segment:source.segments)if(const auto anchor=rebase({segment.address,segment.size},initial))
		{
			const auto logical=texture_storage_logical(inverse,segment.offset+anchor->address-segment.address);
			const auto pixel=logical?texture_storage_pixel(a,*logical):std::nullopt;
			if(!pixel||b.width>a.width-(*pixel)[0]||b.height>a.height-(*pixel)[1])continue;
			bool valid=true;
			for(std::size_t y=0;y<b.height&&valid;++y)for(std::size_t byte=0;byte<b.width*b.pixel_bytes;)
			{
				if(++checks>(1u<<20))throw std::length_error("Native scaled-image identity proof exceeds run budget");
				const auto x=byte/b.pixel_bytes,lane=byte%b.pixel_bytes;
				const auto from=scaled_image_storage_offset(source,(*pixel)[0]+x,(*pixel)[1]+y,lane),to=scaled_image_storage_offset(requested,x,y,lane);
				if(!from||!to){valid=false;break;}
				const auto* s=texture_assembly_segment(source.segments,*from);const auto* d=texture_assembly_segment(requested.segments,*to);
				if(!s||!d){valid=false;break;}
				auto count=b.width*b.pixel_bytes-byte;
				if(a.morton||b.morton)count=std::min(count,b.pixel_bytes-lane);
				if(source.tile)count=std::min(count,32-(*from&31));
				if(requested.tile)count=std::min(count,32-(*to&31));
				count=std::min({count,s->size-(*from-s->offset),d->size-(*to-d->offset)});
				if(!equal({static_cast<std::uint32_t>(std::uint64_t(s->address)+*from-s->offset),count},
					{static_cast<std::uint32_t>(std::uint64_t(d->address)+*to-d->offset),count})){valid=false;break;}
				byte+=count;
			}
			if(!valid)continue;
			const clear_rectangle value{(*pixel)[0],(*pixel)[1],b.width,b.height};
			if(result&&(result->x!=value.x||result->y!=value.y))ambiguous=true;
			result=value;
		}
		return result.has_value()&&!ambiguous;
	};
	(void)memory.test_rebase(prove);return ambiguous?std::nullopt:result;
}
} // namespace metal4
