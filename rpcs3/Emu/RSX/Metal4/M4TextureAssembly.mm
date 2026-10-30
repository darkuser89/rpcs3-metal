#include "M4GuestTexture.h"
#include "M4SurfaceCache.h"
#include "M4Draw.h"
#include "M4Pipeline.h"

namespace metal4
{
namespace
{
constexpr std::string_view normalize_source=R"metal(
#include <metal_stdlib>
using namespace metal;
struct params { uint4 map;uint2 origin;uint2 padding; };
vertex float4 rtt_vertex(uint i [[vertex_id]]) {const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1);}
fragment float4 rtt_channels(float4 p [[position]],texture2d<float,access::read> input [[texture(0)]],constant params& state [[buffer(0)]],device const uchar4* pixel_map [[buffer(1)]]) {
 float4 v=input.read(uint2(p.xy)+state.origin);float4 result;
 if(state.padding.x){const uchar4 map=pixel_map[uint(p.y)*state.padding.y+uint(p.x)];float4 mapped;for(uint c=0;c<4;++c)mapped[c]=map[c]<4?v[map[c]]:map[c]==4?0.f:1.f;v=mapped;}
 for(uint c=0;c<4;++c)result[c]=state.map[c]<4?v[state.map[c]]:state.map[c]==4?0.f:1.f;
 return result;
}
)metal";
bool same_input(const auto& a,const auto& b)
{
	return a.segments==b.segments&&a.plan.location==b.plan.location&&a.plan.offset==b.plan.offset&&a.plan.format==b.plan.format&&a.plan.width==b.plan.width&&a.plan.height==b.plan.height&&
		a.plan.depth==b.plan.depth&&a.plan.mips==b.plan.mips&&a.plan.layers==b.plan.layers&&a.plan.dimension==b.plan.dimension&&
		a.plan.swizzled==b.plan.swizzled&&a.plan.subresources[0].row_pitch==b.plan.subresources[0].row_pitch;
}
display_storage_source source_layout(const cached_surface_description& description)
{
	const auto& a=description.allocation;const auto size=a.samples==1?std::array{a.width,a.height}:sample_expanded_size(a.width,a.height,a.samples);
	const auto bpp=surface_guest_bytes(a.format);auto geometry=plan_guest_surface(0,size[0],size[1],bpp,description.row_pitch,description.morton);
	auto segments=description.segments;if(segments.empty())segments.push_back({description.tile?description.tile->address:description.address,0,description.tile?description.tile->size:geometry.range.size});
	return {std::move(geometry),std::move(segments),false,0,description.tile};
}
std::optional<std::size_t> source_anchor(const cached_surface_description& description,const texture_assembly_plane& plane)
{
	if(!description.mapping)return {};const auto& mapping=description.mapping->state();
	if(mapping.location!=plane.view.location||plane.logical<mapping.logical.address)return {};
	const auto raw=std::size_t(plane.logical-mapping.logical.address);const auto origin=description.tile?description.tile->offset:0;
	return raw>=origin&&raw-origin<mapping.logical.size?std::optional{raw-origin}:std::nullopt;
}
} // namespace
void guest_texture_resolver::prepare_texture_assembly(texture_assembly_input& input)
{
	if(input.native.empty()||input.native.size()>maximum_upload_images||input.slices.empty())throw std::invalid_argument("Native RTT assembly lacks bounded sources");
	if(guest_texture_depth_volume(input.input.plan)){input.metadata.native_argb_channels={0,0,0,0};return;}
	const auto target=guest_texture_vtc(input.input.plan)?MTLPixelFormatRGBA32Float:native_format(input.input.plan.codec.storage);auto channels=input.slices.front().channels;bool canonical=false;
	for(const auto& slice:input.slices)
	{
		const auto format=slice.cast||slice.vm||slice.bytes?native_format(slice.plane.view.codec.storage):slice.native->texture().pixelFormat;
		canonical|=format!=target||slice.channels!=channels;
	}
	if(canonical)
	{
		channels=input.input.plan.codec.base_argb;
		// These guest formats normally discard alpha,but native compatible
		// sources may have real or source-specific X alpha. Materialize it.
		if(input.input.plan.format==0x9d||input.input.plan.format==0x9e)channels[0]=3;
	}
	input.metadata.native_argb_channels=channels;
	for(auto& slice:input.slices)
	{
		const auto format=slice.cast||slice.vm||slice.bytes?native_format(slice.plane.view.codec.storage):slice.native->texture().pixelFormat;
		if(format==target&&slice.channels==channels&&slice.pixel_map.empty())continue;
		if(input.input.plan.codec.depth)throw std::invalid_argument("Depth RTT assembly cannot normalize color channels");
		std::array<bool,4> assigned{};slice.output_map={4,4,4,5};
		for(unsigned argb=0;argb<4;++argb)
		{
			const auto out=channels[argb],in=slice.channels[argb];
			if(out>=4){if(out!=in)throw std::invalid_argument("RTT assembly has incompatible constant channels");continue;}
			if(assigned[out]&&slice.output_map[out]!=in)throw std::invalid_argument("RTT assembly cannot represent source channel aliases");
			assigned[out]=true;slice.output_map[out]=in;
		}
		helper_catalog compile(m_context);const auto library=compile.library(normalize_source,@"Native RTT assembly channels");
		helper_render_configuration config;config.colors[0]=target;slice.normalizer=compile.render(library,@"rtt_vertex",@"rtt_channels",config);
	}
}
void guest_texture_resolver::encode_texture_assembly(const texture_assembly_input& input,id<MTLTexture> texture)
{
	for(const auto& slice:input.slices)
	{
		id<MTLTexture> image=nil;texture_region source;
		if(slice.bytes)
		{
			image=allocate(slice.plane.view);encode_display_assembly(*slice.bytes,image);source.size=MTLSizeMake(image.width,image.height,1);
		}
		else if(slice.vm)
		{
			image=allocate(slice.vm->m_plan);const auto bytes=make_upload(m_context,slice.vm->m_bytes);encode_upload(slice.vm->m_plan,bytes,image);source.size=MTLSizeMake(image.width,image.height,1);
		}
		else if(slice.cast)
		{
			image=allocate(slice.cast->plan);encode_cast(*slice.cast,image);source.size=MTLSizeMake(image.width,image.height,1);
		}
		else
		{
			image=slice.native->texture();source.origin=MTLOriginMake(slice.region.pixels.x,slice.region.pixels.y,0);source.size=MTLSizeMake(slice.region.pixels.width,slice.region.pixels.height,1);
		}
		// Native float depth can carry more precision than its guest byte
		// representation. Restore every proven complete compatible pixel with
		// a raw native copy,including its independent stencil plane.
		for(const auto& patch:slice.precision)
		{
			texture_region from,to;from.origin=MTLOriginMake(patch.source.x,patch.source.y,0);to.origin=MTLOriginMake(patch.target.x,patch.target.y,0);from.size=to.size=MTLSizeMake(patch.source.width,patch.source.height,1);copy_texture(m_context,patch.texture,from,image,to);
		}
		if(guest_texture_depth_volume(input.input.plan)){encode_depth_volume_plane(image,source,texture,slice.plane.level,slice.plane.z);continue;}
		if(slice.normalizer)
		{
			auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:texture.pixelFormat width:slice.plane.view.width height:slice.plane.view.height mipmapped:NO];
			descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;const auto normalized=make_texture(m_context,descriptor);
			struct parameters{std::array<std::uint32_t,4> map;std::array<std::uint32_t,2> origin,padding;};
			const parameters params{slice.output_map,{static_cast<std::uint32_t>(source.origin.x),static_cast<std::uint32_t>(source.origin.y)},{!slice.pixel_map.empty(),static_cast<std::uint32_t>(slice.plane.view.width)}};static_assert(sizeof(params)==32);
			const auto upload=m_context.upload(std::as_bytes(std::span{&params,1}),256);bindings vertex,fragment;fragment.set_texture(0,image);fragment.set_buffer(0,upload.buffer,upload.offset);
			if(slice.pixel_map.empty())fragment.set_buffer(1,upload.buffer,upload.offset);
			else {const auto map=m_context.upload(std::as_bytes(std::span{slice.pixel_map}),256);fragment.set_buffer(1,map.buffer,map.offset);}
			auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=normalized;pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
			raster_state raster;raster.viewport={0,0,double(normalized.width),double(normalized.height),0,1};raster.scissor={0,0,normalized.width,normalized.height};render_commands commands(m_context,pass);commands.configure(slice.normalizer,nil,raster,vertex,fragment);commands.draw(MTLPrimitiveTypeTriangle,0,3);commands.end();
			image=normalized;source.origin=MTLOriginMake(0,0,0);source.size=MTLSizeMake(image.width,image.height,1);
		}
		texture_region destination;destination.level=slice.plane.level;destination.slice=slice.plane.face;destination.origin=MTLOriginMake(0,0,slice.plane.z);destination.size=source.size;
		copy_texture(m_context,image,source,texture,destination);
	}
}
std::optional<resolved_fragment_texture> guest_texture_cache::resolve_assembly(const fragment_texture_registers& words,
	const guest_texture_resolver::source& input,surface_cache& surfaces,std::shared_ptr<const surface_texture_revision>* receipt,
	pending_textures* pending,std::optional<pending_image>* pending_index)
{
	const auto epoch=surfaces.texture_source_epoch();
	const auto current=[&](const auto& assembly)
	{
		bool valid=assembly.native_epoch==epoch&&std::all_of(assembly.native.begin(),assembly.native.end(),[](const auto& p){return p->source_current();});
		if(valid)m_resolver.cpu_capture([&]{valid=m_resolver.current(assembly.input,assembly.revisions)&&assembly.receipt->current();});return valid;
	};
	const auto metadata=[&](texture_cache_state value){value.coordinates=((words.format>>8)&0x40)?normalize_guest_pixels(input.plan.width,input.plan.height,input.plan.depth):texture_coordinates{};return value;};
	for(auto it=m_entries.begin();it!=m_entries.end();++it)if(it->assembly&&same_input(it->input,input))
	{
		if(current(*it->assembly)){auto result=it->resource;result.metadata=metadata(result.metadata);if(receipt)*receipt=it->assembly->receipt;m_entries.splice(m_entries.begin(),m_entries,it);++m_stats.hits;++m_stats.assemblies_reused;return result;}
		m_entries.erase(it);++m_stats.invalidations;break;
	}
	if(pending)for(std::size_t i=0;i<pending->casts.size();++i)if(const auto& assembly=pending->casts[i].assembly;assembly&&same_input(assembly->input,input)&&current(*assembly))
	{
		if(receipt)*receipt=assembly->receipt;*pending_index=pending_image{i,true};++m_stats.assemblies_reused;return resolved_fragment_texture{nil,metadata(assembly->metadata)};
	}
	if(surfaces.texture_sources(input.coverage).empty())return {};
	auto planes=plan_texture_assembly_planes(input.plan,input.segments);std::vector<guest_surface_layout> all_coverage;for(const auto& plane:planes)for(auto span:plane.spans)all_coverage.push_back(guest_byte_coverage({span.address,span.size}));
	// Discover before VM capture. An unresolved partial native owner must never
	// be silently replaced by old RAM or flushed by the legacy fallback route.
	const auto initial_sources=surfaces.texture_sources(all_coverage);if(initial_sources.empty())return {};
	if(m_resolver.m_context.buffer_ledger()!=m_resolver.m_budget)throw std::invalid_argument("Native RTT assembly requires one shared buffer/texture ledger for guest EA "+std::to_string(input.segments.front().address));
	auto assembly=std::make_shared<guest_texture_resolver::texture_assembly_input>();assembly->input=input;
	m_resolver.cpu_capture([&]{m_resolver.watch(assembly->input);for(const auto& coverage:assembly->input.coverage)assembly->revisions.push_back(m_resolver.m_memory->surface_revision(coverage));});
	const auto retain_native=[&](std::shared_ptr<prepared_surface_snapshot> value)
	{
		for(const auto& previous:assembly->native)if(previous->same_content(*value))return previous;
		assembly->native.push_back(value);return value;
	};
	// Settle already-known deferred authority before planning any VM holes.
	// Later sources must not change the owner set of an earlier target plane.
	for(const auto& ref:initial_sources)if(surfaces.status(ref).deferred_copies)
	{
		std::vector<std::shared_ptr<prepared_surface_snapshot>> previous=assembly->native;if(pending)for(const auto& source:pending->native)if(std::find(previous.begin(),previous.end(),source)==previous.end())previous.push_back(source);
		const auto full=[](const cached_surface_description& description)->std::optional<surface_texture_region>{const auto layout=source_layout(description);return surface_texture_region{{0,0,layout.geometry.width,layout.geometry.height},0};};
		const auto snapshot=surfaces.prepare_texture_mapping(all_coverage,full,m_resolver.m_budget,false,previous,previous.size()<guest_texture_resolver::maximum_upload_images,false,ref);
		if(!snapshot)throw texture_revision_conflict();(void)retain_native(snapshot);
	}
	const auto planned_epoch=surfaces.texture_source_epoch();
	for(auto& plane:planes)
	{
		std::vector<guest_surface_layout> coverage;for(auto span:plane.spans)coverage.push_back(guest_byte_coverage({span.address,span.size}));
		const auto mapping=[&](const cached_surface_description& description)->std::optional<surface_texture_region>
		{
			const auto source=source_layout(description);const auto region=plan_native_texture_plane(*m_resolver.m_memory,source,plane,source_anchor(description,plane));
			return region?std::optional<surface_texture_region>{{{0,0,source.geometry.width,source.geometry.height},0}}:std::nullopt;
		};
		std::vector<std::shared_ptr<prepared_surface_snapshot>> previous;for(const auto& native:assembly->native)if(native->needs_copy())previous.push_back(native);
		if(pending)for(const auto& native:pending->native)if(std::find(previous.begin(),previous.end(),native)==previous.end())previous.push_back(native);
		auto native=surfaces.prepare_texture_mapping(coverage,mapping,m_resolver.m_budget,false,previous,previous.size()<guest_texture_resolver::maximum_upload_images,true);
		if(native)native=retain_native(std::move(native));
		guest_texture_resolver::texture_assembly_slice slice; slice.plane=std::move(plane);slice.native=native;
		if(native)
		{
			const auto& description=native->source_description();std::optional<surface_texture_region> region;
			m_resolver.cpu_capture([&]{region=plan_native_texture_plane(*m_resolver.m_memory,source_layout(description),slice.plane,source_anchor(description,slice.plane));});
			if(!region)throw texture_revision_conflict();slice.region=*region;
			const auto channels=surface_texture_channels(native->format(),slice.plane.view);const bool direct=channels&&!region->guest_row_offset&&region->pixels.width*surface_guest_bytes(native->format())==slice.plane.view.subresources[0].row_pitch;
			slice.channels=direct?*channels:slice.plane.view.codec.base_argb;
			if(!direct)slice.cast=m_resolver.prepare_cast(slice.plane.view,native->format(),native->texture(),{},region->guest_row_offset,region->pixels);
		}
		else
		{
			slice.channels=slice.plane.view.codec.base_argb;
			auto refs=surfaces.texture_sources(coverage);const auto initial=refs;
			if(!refs.empty())
			{
				bool deferred=false;std::vector<display_storage_source> layouts;std::vector<std::optional<std::uint32_t>> origins;std::vector<std::shared_ptr<prepared_surface_snapshot>> snapshots;
				for(const auto& ref:refs)
				{
					const auto status=surfaces.status(ref);const auto description=surfaces.description(ref);auto layout=source_layout(description);std::optional<std::uint32_t> origin;
					if(description.tile&&description.mapping)
					{
						const auto& state=description.mapping->state();
						if(state.location==input.plan.location&&overlaps(state.logical,guest_surface_range{input.plan.offset,input.plan.storage_bytes}))
						{
							if(state.tile!=description.tile||state.logical.size!=description.tile->size||state.segments!=layout.segments||state.pixel_address!=description.address)throw surface_mapping_conflict();
							const auto at=std::uint64_t(state.logical.address)+description.tile->offset;if(at>0xffffffffu)throw surface_mapping_conflict();origin=static_cast<std::uint32_t>(at);
						}
					}
					layout.dirty=status.dirty||status.deferred_copies;layout.epoch=status.native_write_epoch;deferred|=bool(status.deferred_copies);layouts.push_back(std::move(layout));origins.push_back(origin);
				}
				texture_storage_assembly planned;
				const auto build=[&]{m_resolver.cpu_capture([&]{planned=plan_texture_storage_assembly(*m_resolver.m_memory,slice.plane,layouts,origins);});};
				const auto trim=[&]
				{
					std::vector<bool> used(refs.size());for(const auto& copy:planned.bytes.copies)used.at(planned.sources.at(copy.input))=true;
					std::vector<cached_surface_ref> kept;std::vector<display_storage_source> storage;std::vector<std::optional<std::uint32_t>> logical;std::vector<std::shared_ptr<prepared_surface_snapshot>> native;
					for(std::size_t i=0;i<refs.size();++i)if(used[i]){kept.push_back(refs[i]);storage.push_back(layouts[i]);logical.push_back(origins[i]);if(!snapshots.empty())native.push_back(snapshots[i]);}
					const bool changed=kept.size()!=refs.size();refs=std::move(kept);layouts=std::move(storage);origins=std::move(logical);snapshots=std::move(native);return changed;
				};
				if(!deferred){build();trim();}
				for(std::size_t i=0;i<refs.size();++i)
				{
					const auto full=[](const cached_surface_description& description)->std::optional<surface_texture_region>{const auto layout=source_layout(description);return surface_texture_region{{0,0,layout.geometry.width,layout.geometry.height},0};};
					auto snapshot=surfaces.prepare_texture_mapping(coverage,full,m_resolver.m_budget,false,previous,previous.size()<guest_texture_resolver::maximum_upload_images,false,refs[i]);
					if(!snapshot)throw texture_revision_conflict();snapshot=retain_native(std::move(snapshot));snapshots.push_back(snapshot);
					if(snapshot->needs_copy()&&std::find(previous.begin(),previous.end(),snapshot)==previous.end())previous.push_back(snapshot);
					const auto status=surfaces.status(refs[i]);layouts[i].dirty=status.dirty||status.deferred_copies;layouts[i].epoch=status.native_write_epoch;
				}
				for(std::size_t i=0;i<refs.size();++i)if(!snapshots[i]->matches_source(refs[i],false))throw texture_revision_conflict();
				if(!refs.empty())
				{
					build();if(trim()&&!refs.empty())build();
					if(!refs.empty())
					{
						std::vector<bool> used(planned.layouts.size());for(const auto& copy:planned.bytes.copies)used.at(copy.input)=true;std::vector<std::size_t> indices(used.size());
						for(std::size_t view=0;view<used.size();++view)if(used[view]){indices[view]=slice.partial_layouts.size();slice.partial_layouts.push_back(std::move(planned.layouts[view]));slice.partial_native.push_back(snapshots.at(planned.sources[view]));}
						for(auto& copy:planned.bytes.copies)copy.input=indices[copy.input];slice.partial_plan=std::move(planned.bytes);
					}
				}
				for(const auto& ref:surfaces.texture_sources(coverage))if(std::find(initial.begin(),initial.end(),ref)==initial.end()){const auto status=surfaces.status(ref);if(status.dirty||status.deferred_copies)throw texture_revision_conflict();}
				// X-channel semantics belong to a complete compatible source pixel.
				// Byte mixtures/casts and VM pixels retain the target upload policy.
				const bool color_map=input.plan.format==0x82||input.plan.format==0x9d||input.plan.format==0x85||input.plan.format==0x9e||input.plan.format==0x8b||input.plan.format==0x98;
				const bool precision=input.plan.codec.depth&&(*input.plan.codec.depth==surface_depth::z16_float||*input.plan.codec.depth==surface_depth::z24s8_float);
				if(!slice.partial_native.empty()&&(color_map||precision))
				{
					const auto bpp=input.plan.codec.block.bytes;const auto count=slice.plane.view.storage_bytes/bpp;if(color_map){const bool constant=input.plan.format==0x9d||input.plan.format==0x9e;slice.pixel_map.assign(count,std::array<std::uint8_t,4>{0,1,2,static_cast<std::uint8_t>(constant?5:3)});if(constant)slice.channels[0]=3;}std::size_t first=0;
					std::vector<std::vector<texture_storage_run>> inverse;for(const auto& layout:slice.partial_layouts)inverse.push_back(texture_storage_runs(layout));
					for(std::size_t pixel=0;pixel<count;++pixel)
					{
						const auto begin=pixel*bpp,end=begin+bpp;while(first<slice.partial_plan.copies.size()&&slice.partial_plan.copies[first].target+slice.partial_plan.copies[first].size<=begin)++first;
						if(first==slice.partial_plan.copies.size()||slice.partial_plan.copies[first].target>begin)continue;
						const auto& head=slice.partial_plan.copies[first];const auto offset=texture_storage_logical(inverse[head.input],head.source+begin-head.target);if(!offset)continue;std::size_t position=begin,index=first;
						while(position<end&&index<slice.partial_plan.copies.size())
						{
							const auto& copy=slice.partial_plan.copies[index];if(copy.target>position||copy.input!=head.input)break;
							const auto finish=std::min(end,copy.target+copy.size);bool valid=true;
							for(;position<finish;++position)if(texture_storage_logical(inverse[head.input],copy.source+position-copy.target)!=*offset+position-begin){valid=false;break;}
							if(!valid)break;++index;
						}
						const auto& layout=slice.partial_layouts[head.input];const auto coordinates=texture_storage_pixel(layout.geometry,*offset);
						if(position!=end||layout.geometry.pixel_bytes!=bpp||!coordinates)continue;
						const auto channels=surface_texture_channels(slice.partial_native[head.input]->format(),slice.plane.view);if(!channels)continue;
						if(color_map)
						{
							auto inputs=*channels;const auto source_format=slice.partial_native[head.input]->texture().pixelFormat,target_format=guest_texture_resolver::native_format(slice.plane.view.codec.storage);
							if(source_format!=target_format){if(!((source_format==MTLPixelFormatRGBA8Unorm&&target_format==MTLPixelFormatBGRA8Unorm)||(source_format==MTLPixelFormatBGRA8Unorm&&target_format==MTLPixelFormatRGBA8Unorm)))throw std::invalid_argument("Native RTT pixel map lacks a storage-compatible channel cast");for(auto& channel:inputs)if(channel==0||channel==2)channel^=2;}
							auto& map=slice.pixel_map[pixel];std::array<bool,4> assigned{};
							for(unsigned c=0;c<4;++c){const auto out=slice.channels[c],in=inputs[c];if(out>=4){if(out!=in)throw std::invalid_argument("Native RTT pixel has incompatible constant channels");continue;}if(assigned[out]&&map[out]!=in)throw std::invalid_argument("Native RTT pixel cannot represent channel aliases");map[out]=static_cast<std::uint8_t>(in);assigned[out]=true;}
						}
						if(precision)
						{
							const auto texture=slice.partial_native[head.input]->texture();const auto x=pixel%slice.plane.view.width,y=pixel/slice.plane.view.width;
							if(!slice.precision.empty()&&slice.precision.back().texture==texture&&slice.precision.back().source.y==(*coordinates)[1]&&slice.precision.back().target.y==y&&slice.precision.back().source.x+slice.precision.back().source.width==(*coordinates)[0]&&slice.precision.back().target.x+slice.precision.back().target.width==x){++slice.precision.back().source.width;++slice.precision.back().target.width;}
							else slice.precision.push_back({texture,{(*coordinates)[0],(*coordinates)[1],1,1},{x,y,1,1}});
						}
					}
				}
			}
		}
		assembly->slices.push_back(std::move(slice));
		if(surfaces.texture_source_epoch()!=planned_epoch)throw texture_revision_conflict();
	}
	if(assembly->native.empty())return {};
	if(assembly->native.size()>guest_texture_resolver::maximum_upload_images)throw std::length_error("RTT assembly exceeds native source count");
	std::vector<std::shared_ptr<const surface_texture_revision>> revisions;for(const auto& native:assembly->native){if(!native->source_current(false))throw texture_revision_conflict();revisions.push_back(native->revision());}
	assembly->receipt=std::make_shared<surface_texture_revision>(std::move(revisions));
	m_resolver.cpu_capture([&]
	{
		if(!assembly->receipt->current()||!m_resolver.current(assembly->input,assembly->revisions))throw texture_revision_conflict();
		for(auto& slice:assembly->slices)if(!slice.native)
		{
			if(!slice.partial_native.empty())
			{
				slice.bytes=std::make_shared<guest_texture_resolver::display_assembly_input>();
				for(auto hole:slice.partial_plan.vm)
					for(std::size_t offset=hole.address,remaining=hole.size;remaining;)
					{
						const auto* segment=texture_assembly_segment(slice.plane.spans,offset);if(!segment)throw std::logic_error("RTT VM hole lacks its actual address");const auto count=std::min(remaining,segment->size-(offset-segment->offset));
						const auto captured=m_resolver.m_memory->read_surface(guest_byte_coverage({static_cast<std::uint32_t>(std::uint64_t(segment->address)+offset-segment->offset),count}));if(captured.bytes.size()!=count)throw std::runtime_error("RTT VM fragment capture incomplete");slice.bytes->vm_bytes.insert(slice.bytes->vm_bytes.end(),captured.bytes.begin(),captured.bytes.end());offset+=count;remaining-=count;
					}
				continue;
			}
			std::vector<std::byte> bytes(slice.plane.view.storage_bytes);
			for(auto span:slice.plane.spans)
			{
				const auto captured=m_resolver.m_memory->read_surface(guest_byte_coverage({span.address,span.size}));if(captured.bytes.size()!=span.size)throw std::runtime_error("RTT VM plane capture incomplete");std::copy(captured.bytes.begin(),captured.bytes.end(),bytes.begin()+span.offset);
			}
			slice.vm=captured_guest_texture(slice.plane.view,{},std::move(bytes));
		}
		if(!assembly->receipt->current()||!m_resolver.current(assembly->input,assembly->revisions))throw texture_revision_conflict();
	});
	for(auto& slice:assembly->slices)if(slice.bytes)
		*slice.bytes=m_resolver.prepare_display_assembly(slice.plane.view,slice.partial_native,slice.partial_layouts,slice.partial_plan,std::move(slice.bytes->vm_bytes));
	assembly->metadata.actual_mips=input.plan.mips;assembly->metadata.depth=bool(input.plan.codec.depth);assembly->metadata.source_depth_format=input.plan.codec.depth?static_cast<unsigned>(*input.plan.codec.depth):0;
	assembly->metadata.depth_float=input.plan.codec.depth&&(*input.plan.codec.depth==surface_depth::z16_float||*input.plan.codec.depth==surface_depth::z24s8_float);assembly->metadata=metadata(assembly->metadata);
	for(const auto& slice:assembly->slices){assembly->source_bytes=checked_texture_sum(assembly->source_bytes,checked_texture_product(slice.pixel_map.size(),4));if(slice.vm)assembly->source_bytes=checked_texture_sum(assembly->source_bytes,slice.vm->storage_bytes());if(slice.cast)assembly->source_bytes=checked_texture_sum(assembly->source_bytes,slice.cast->packed.storage_bytes);if(slice.bytes)assembly->source_bytes=checked_texture_sum(assembly->source_bytes,checked_texture_sum(checked_texture_sum(slice.bytes->plan.storage_bytes,slice.bytes->raw_bytes),checked_texture_sum(slice.bytes->linear_bytes,slice.bytes->vm_bytes.size())));}
	if(assembly->source_bytes>m_resolver.m_limit)throw std::length_error("RTT assembly source bytes exceed budget");
	m_resolver.prepare_texture_assembly(*assembly);assembly->native_epoch=planned_epoch;
	guest_texture_resolver::native_cast_input cast{};cast.plan=input.plan;cast.packed=input.plan;cast.packed.storage_bytes=assembly->source_bytes;cast.format=assembly->native.front()->format();cast.source=assembly->native.front()->texture();cast.metadata=assembly->metadata;cast.assembly=assembly;
	if(receipt)*receipt=assembly->receipt;
	++m_stats.assemblies_created;for(const auto& slice:assembly->slices){m_stats.assembly_native_planes+=bool(slice.native||slice.bytes);m_stats.assembly_vm_planes+=bool(slice.vm);if(slice.vm)m_stats.assembly_vm_bytes+=slice.vm->storage_bytes();if(slice.bytes)m_stats.assembly_vm_bytes+=slice.bytes->vm_bytes.size();}
	if(pending)
	{
		check_pending_bytes(*pending,cast.packed.storage_bytes);for(const auto& native:assembly->native)if(native->needs_copy()&&std::find(pending->native.begin(),pending->native.end(),native)==pending->native.end())pending->native.push_back(native);
		*pending_index=pending_image{pending->casts.size(),true};pending->casts.push_back(std::move(cast));return resolved_fragment_texture{nil,assembly->metadata};
	}
	std::vector<std::shared_ptr<prepared_surface_snapshot>> copies;for(const auto& native:assembly->native)if(native->needs_copy())copies.push_back(native);
	const auto work=m_resolver.enqueue_mixed({},std::span{&cast,1},{},copies);auto image=work.finish().at(0);for(const auto& native:copies)native->completed();
	m_resolver.cpu_capture([&]{if(!assembly->receipt->current()||!m_resolver.current(assembly->input,assembly->revisions))throw texture_revision_conflict();});for(const auto& native:copies)native->publish();publish_assembly(assembly,image);return image;
}
void guest_texture_cache::publish_assembly(const std::shared_ptr<guest_texture_resolver::texture_assembly_input>& assembly,const resolved_fragment_texture& image)
{
	if(assembly->native.empty()||!assembly->native.front()->cache_native_epoch(assembly->native_epoch)||!std::all_of(assembly->native.begin(),assembly->native.end(),[](const auto& p){return p->source_current();}))return;
	auto compact=std::make_shared<guest_texture_resolver::texture_assembly_input>();compact->input=assembly->input;compact->revisions=assembly->revisions;compact->receipt=assembly->receipt;compact->metadata=assembly->metadata;compact->native_epoch=assembly->native_epoch;
	std::size_t bytes=image.texture.heap.size;for(const auto& native:assembly->native){compact->native.push_back(native->needs_copy()?native->settled():native);bytes=checked_texture_sum(bytes,native->texture().heap.size);}if(bytes>m_resolver.m_limit)return;
	std::vector<std::shared_ptr<const physical_surface>> physical;physical.reserve(assembly->input.coverage.size());bool complete=true;
	const auto publication=publication_guard();const auto memory=m_resolver.m_memory;const auto translate=m_resolver.m_translate;const auto capture=m_resolver.m_capture;
	const auto validate=[&]
	{
		const auto current=[&]{if(!assembly->receipt->current()||texture_storage_segments(assembly->input.plan.offset,assembly->input.plan.location,assembly->input.plan.storage_bytes,translate)!=assembly->input.segments)return false;for(std::size_t i=0;i<assembly->revisions.size();++i)if(memory->surface_revision(assembly->input.coverage[i])!=assembly->revisions[i])return false;return true;};
		if(!current())throw texture_revision_conflict();
		for(std::size_t i=0;i<assembly->input.coverage.size();++i){try{auto ticket=memory->capture_physical_surface(assembly->input.coverage[i],assembly->revisions[i]);if(!ticket){complete=false;break;}physical.push_back(std::move(ticket));}catch(const std::length_error&){complete=false;break;}}
		if(!current())throw texture_revision_conflict();
	};if(capture)checked_cpu_capture(capture,validate);else validate();if(!complete)physical.clear();
	if(!publication())return;
	std::erase_if(m_entries,[&](const auto& entry){return entry.assembly&&same_input(entry.input,assembly->input);});
	while(m_entries.size()>=m_capacity){m_entries.pop_back();++m_stats.evictions;}
	m_entries.push_front({assembly->input,assembly->revisions,image,std::move(physical),std::move(compact)});
}
} // namespace metal4
