#include "M4PhysicalTextureRegion.h"
#include "M4DisplayStorage.h"
#include "M4DisplayAssembly.h"
#include "M4TextureAssembly.h"

#include <iostream>
#include <tuple>
#include <source_location>

namespace
{
using namespace metal4;
void require(bool value, const std::source_location location = std::source_location::current())
{
	if (!value)
		throw std::runtime_error("Guest/native surface-region address oracle disagreed at line " + std::to_string(location.line()));
}
std::size_t morton(std::size_t x, std::size_t y, std::size_t z, std::array<std::size_t, 3> dimensions)
{
	std::size_t result = 0, weight = 1;
	for (std::size_t bit = 1; bit < std::max({dimensions[0], dimensions[1], dimensions[2]}); bit *= 2)
		for (unsigned axis = 0; axis < 3; ++axis)
			if (bit < dimensions[axis])
			{
				if (std::array{x, y, z}[axis] & bit)
					result += weight;
				weight *= 2;
			}
	return result;
}
guest_texture_layout texture(unsigned format, bool packed, unsigned width, unsigned height, unsigned pitch)
{
	fragment_texture_registers r;
	r.format = (1u << 16) | ((format | (packed ? 0 : 0x20)) << 8) | (2u << 4) | 8 | 1;
	r.control0 = 1u << 31;
	r.image_rect = (width << 16) | height;
	r.control3 = pitch;
	return plan_guest_texture(r);
}
class identity_memory : public guest_surface_memory
{
  public:
	guest_surface_snapshot read_surface(const guest_surface_layout&) override { throw std::logic_error("Region proof read VM pixels"); }
	std::uint64_t surface_revision(const guest_surface_layout&) override { return 1; }
	std::optional<std::uint64_t> compare_write(const guest_surface_layout&, std::uint64_t, std::span<const std::byte>) override { throw std::logic_error("Region proof wrote VM"); }
	void write(guest_surface_range, std::span<const std::byte>) override { throw std::logic_error("Region proof wrote VM"); }
};
void morton_display_cases()
{
	std::size_t profiles = 0, accepted = 0, mixed = 0, vm_only = 0, missing = 0;
	for (const auto dimensions : {std::array<std::size_t,3>{8,8,1},{16,4,2},{4,16,4}})
	for (bool cropped : {false,true}) for (unsigned bytes : {1u,2u,4u,8u,16u})
	{
		const std::array<std::size_t,3> origin{cropped ? 1u : 0u,cropped ? 1u : 0u,dimensions[2]-1};
		const std::array<std::size_t,3> size{dimensions[0]-(cropped ? 2 : 0),dimensions[1]-(cropped ? 2 : 0),1};
		const auto source = plan_guest_surface(0,size[0],size[1],bytes,0,swizzle_region{dimensions,origin,size});
		std::vector<bool> active(source.range.size);
		for (std::size_t y=0;y<size[1];++y)for(std::size_t x=0;x<size[0];++x)
			for(unsigned lane=0;lane<bytes;++lane)active[morton(x+origin[0],y+origin[1],origin[2],dimensions)*bytes+lane]=true;
		for (unsigned av : {4u,8u}) for(unsigned width : {1u,3u})for(unsigned height : {1u,3u})for(unsigned odd : {0u,1u})
		{
			const auto display=plan_guest_surface(0,width,height,av,width*av+odd);
			for(std::size_t offset=0;offset<=source.range.size+1;offset+=3)
			{
				bool expected=true;
				std::vector<std::size_t> absent;
				for(unsigned y=0;y<height;++y)for(unsigned byte=0;byte<width*av;++byte)
				{
					const auto index=offset+y*display.row_pitch+byte;
					expected &= index<active.size()&&active[index];
					if(index<active.size()&&!active[index])absent.push_back(index);
				}
				require(morton_display_bytes_active(source,display,offset)==expected);++profiles;accepted+=expected;
				const bool bounded=offset<=source.range.size&&(height-1)*display.row_pitch+width*av<=source.range.size-offset;
				const auto holes=morton_display_holes(source,display,offset);require(bool(holes)==bounded);
				if(holes)
				{
					std::vector<std::size_t> actual;
					for(auto run:*holes)for(std::size_t byte=0;byte<run.size;++byte)actual.push_back(run.address+byte);
					require(actual==absent);mixed+=!actual.empty()&&actual.size()<width*height*av;vm_only+=actual.size()==width*height*av;missing+=actual.size();
				}
			}
		}
	}
	identity_memory memory;
	const auto source=plan_guest_surface(0,8,8,4,0,swizzle_region{{8,8,1},{},{8,8,1}});
	const auto display=plan_guest_surface(0,3,2,4,13);
	const std::array storage{guest_storage_segment{0xffff0000,0,7},guest_storage_segment{0x200000,7,source.range.size-7}};
	std::array requested{guest_storage_segment{0xffff0001,0,6},guest_storage_segment{0x200000,6,display.range.size-6}};
	require(plan_morton_display_storage(memory,source,storage,display,requested)==1);
	require(plan_morton_display_storage(memory,source,storage,display,requested,1)==1);
	requested[1].address+=1;require(!plan_morton_display_storage(memory,source,storage,display,requested,1));
	// Padding can lie outside native storage, but the last consumed byte cannot.
	const auto edge=plan_guest_surface(0,1,1,4,5);
	const std::array whole{guest_storage_segment{0x1000,0,source.range.size}};
	const std::array tail{guest_storage_segment{0x1000+252,0,5}};
	require(plan_morton_display_storage(memory,source,whole,edge,tail)==252);
	require(!morton_display_bytes_active(source,edge,253));
	const std::array duplicate{guest_storage_segment{0x1000,0,128},guest_storage_segment{0x1000,128,128}};
	const std::array repeated_request{guest_storage_segment{0x1001,0,display.range.size}};
	require(!plan_morton_display_storage(memory,source,duplicate,display,repeated_request));
	require(plan_morton_display_storage(memory,source,duplicate,display,repeated_request,129)==129);
	const std::array padding_only{guest_storage_segment{0x1001,0,12},guest_storage_segment{0x9000,12,1},
		guest_storage_segment{0x100e,13,12},guest_storage_segment{0x9001,25,1}};
	require(plan_morton_display_storage(memory,source,whole,display,padding_only)==1);
	const auto cropped=plan_guest_surface(0,6,6,4,0,swizzle_region{{8,8,1},{1,1,0},{6,6,1}});
	const std::array head{guest_storage_segment{0x1000,0,display.range.size}};
	require(!plan_morton_display_storage(memory,cropped,whole,display,head,0));
	require(plan_morton_display_storage(memory,cropped,whole,display,head,0,true)==0);
	require(profiles>100000&&accepted>1000);
	std::cout<<"Morton display active-byte oracle:"<<profiles<<" rectangular2D/3D-slice/cropped/1-2-4-8-16byte/AV4-8/pitch-offset profiles,"<<accepted<<" accepted;segmented midtexel/row physical identity,wrong later EA,trailing-padding boundary,duplicate physical ambiguity/explicit logical anchor andpadding-only divergence passed;zero VM pixel access\n";
	std::cout<<"Morton hole oracle:"<<mixed<<" native+VM/"<<vm_only<<" VM-only bounded profiles/"<<missing<<" independently enumerated missing bytes;exact intervals exclude allnative bytes andAVpadding,bounds checked\n";
}
void display_assembly_cases()
{
	identity_memory memory;
	const auto a=plan_guest_surface(0,6,5,4,27);
	const auto b=plan_guest_surface(0,3,3,8,0,swizzle_region{{4,4,1},{},{3,3,1}});
	std::array<display_storage_source,2> sources{{{a,{{0x1000,0,7},{0x1010,7,a.range.size-7}},false,3},
		{b,{{0x1003,0,b.range.size}},false,1}}};
	std::map<unsigned,std::vector<std::pair<int,std::size_t>>> bytes;
	for(unsigned y=0;y<5;++y)for(unsigned x=0;x<6;++x)for(unsigned lane=0;lane<4;++lane)
	{
		const auto logical=y*27+x*4+lane;const auto address=logical<7?0x1000+logical:0x1010+logical-7;
		bytes[address].push_back({0,logical});
	}
	for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)for(unsigned lane=0;lane<8;++lane)
	{
		const auto logical=morton(x,y,0,{4,4,1})*8+lane;bytes[0x1003+logical].push_back({1,logical});
	}
	std::size_t profiles=0,refused=0,checked=0;
	for(bool ad:{false,true})for(bool bd:{false,true})for(unsigned epoch:{1u,3u,7u})
	for(unsigned av:{4u,8u})for(unsigned width:{1u,3u,7u})for(unsigned offset:{0u,1u,3u,7u,31u})for(unsigned height:{1u,3u,5u})
	{
		sources[0].dirty=ad;sources[1].dirty=bd;sources[1].epoch=epoch;
		const auto display=plan_guest_surface(0x1000+offset,width,height,av,width*av+3);
		std::vector<std::pair<int,std::size_t>> expected(display.range.size,{-2,0}),actual(display.range.size,{-2,0});bool ambiguous=false;
		for(unsigned y=0;y<height;++y)for(unsigned lane=0;lane<width*av;++lane)
		{
			const auto at=y*display.row_pitch+lane;auto& wanted=expected[at];wanted={-1,0};
			const auto found=bytes.find(display.range.address+at);if(found==bytes.end())continue;
			for(const auto candidate:found->second)
			{
				if(wanted.first<0){wanted=candidate;continue;}
				const auto& old=sources[wanted.first];const auto& next=sources[candidate.first];
				if(old.dirty==next.dirty&&old.epoch==next.epoch)ambiguous=true;
				if(std::pair{next.dirty,next.epoch}>std::pair{old.dirty,old.epoch})wanted=candidate;
			}
		}
		bool rejected=false;display_storage_assembly plan;
		try{plan=plan_display_storage_assembly(memory,display,sources);}catch(const std::invalid_argument&){rejected=true;}
		require(rejected==ambiguous);++profiles;if(rejected){++refused;continue;}
		for(auto copy:plan.copies)for(std::size_t byte=0;byte<copy.size;++byte)
		{
			require(copy.target+byte<actual.size()&&actual[copy.target+byte].first==-2);
			actual[copy.target+byte]={int(copy.input),copy.source+byte};++checked;
		}
		for(auto run:plan.vm)for(std::size_t byte=0;byte<run.size;++byte)
		{
			require(run.address+byte<actual.size()&&actual[run.address+byte].first==-2);actual[run.address+byte]={-1,0};++checked;
		}
		require(actual==expected);
	}
	require(profiles==1080&&refused>0&&checked>10000);
	std::cout<<"Display assembly direct oracle:"<<profiles<<" two-source linear/Morton/segmented/odd-pitch/AV4-8/offset/authority profiles,"<<refused<<" ambiguous refusals,"<<checked<<" active byte decisions;outside-source VM,dirty/epoch precedence,padding exclusion,zero VM pixel access\n";
}
void display_physical_assembly_cases()
{
	class physical_memory final : public identity_memory
	{
	  public:
		// Same raw pointer, independent control blocks: pointer equality is not identity.
		int token=0;std::array<std::shared_ptr<void>,2> owners{std::shared_ptr<void>(&token,[](void*){}),std::shared_ptr<void>(&token,[](void*){})};
		std::map<unsigned,std::pair<unsigned,std::size_t>> physical;
		unsigned invalid=0;
		bool test_storage(const storage_test& test) override {return test(false,[](auto,auto){return true;});}
		std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
		{
			if(invalid==1)return {};if(invalid==2)return std::vector<storage_identity_span>{};
			std::vector<storage_identity_span> result;
			for(unsigned byte=0;byte<range.size;++byte)
			{
				const auto [owner,offset]=physical.at(range.address+byte);result.push_back({owners[owner],offset,{range.address+byte,1}});
			}
			if(invalid==3)result.front().allocation.reset();
			if(invalid==4)++result.front().guest.address;
			if(invalid==5)result.front().offset=std::numeric_limits<std::size_t>::max();
			if(invalid==6)result.front().guest.size=range.size+1;
			return result;
		}
	} memory;
	const auto a=plan_guest_surface(0,5,3,4,23),b=plan_guest_surface(0,3,3,8,0,swizzle_region{{4,4,1},{},{3,3,1}});
	const std::array<display_storage_source,2> sources{{{a,{{0x1000,0,a.range.size}},true,11},{b,{{0x2000,0,b.range.size}},true,17}}};
	for(unsigned i=0;i<a.range.size;++i)memory.physical[0x1000+i]={0,91+i};
	for(unsigned i=0;i<b.range.size;++i)memory.physical[0x2000+i]={1,91+i};
	std::map<std::pair<unsigned,std::size_t>,std::pair<int,std::size_t>> native;
	for(unsigned y=0;y<3;++y)for(unsigned x=0;x<5;++x)for(unsigned byte=0;byte<4;++byte)native[{0,91+y*23+x*4+byte}]={0,y*23+x*4+byte};
	for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)for(unsigned byte=0;byte<8;++byte){const auto logical=morton(x,y,0,{4,4,1})*8+byte;native[{1,91+logical}]={1,logical};}
	std::size_t profiles=0,checked=0;
	for(unsigned shift:{0u,1u,7u,31u})for(unsigned split:{1u,3u,7u,19u})for(unsigned av:{4u,8u})for(unsigned width:{1u,3u})
	{
		const auto display=plan_guest_surface(0x9000,width,4,av,width*av+1);
		for(unsigned i=0;i<display.range.size;++i)memory.physical[0x9000+i]={(i/split)%2,91+(i+shift)%140};
		const auto plan=plan_display_storage_assembly(memory,display,sources);
		std::vector<std::pair<int,std::size_t>> expected(display.range.size,{-2,0}),actual=expected;
		for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<width*av;++byte)
		{
			const auto at=y*display.row_pitch+byte;const auto found=native.find(memory.physical.at(0x9000+at));expected[at]=found==native.end()?std::pair<int,std::size_t>{-1,0}:found->second;++checked;
		}
		for(auto copy:plan.copies)for(std::size_t i=0;i<copy.size;++i){require(actual[copy.target+i].first==-2);actual[copy.target+i]={int(copy.input),copy.source+i};}
		for(auto hole:plan.vm)for(unsigned i=0;i<hole.size;++i){require(actual[hole.address+i].first==-2);actual[hole.address+i]={-1,0};}
		require(actual==expected);++profiles;
		// The same physical identities in a dense RTT plane,with reordered
		// actual EAs and a top32-bit split. Source domains must be reused while
		// target offsets remain in canonical pixel order rather than EA order.
		const auto dense=plan_guest_surface(0,width,4,av);const auto bytes=dense.range.size;const std::array spans{guest_storage_segment{0xfffffff0,0,std::min<std::size_t>(16,bytes)},guest_storage_segment{0x6000,std::min<std::size_t>(16,bytes),bytes-std::min<std::size_t>(16,bytes)}};
		std::vector<guest_storage_segment> segments;for(const auto span:spans)if(span.size)segments.push_back(span);
		std::vector<std::pair<int,std::size_t>> canonical(bytes,{-2,0}),wanted(bytes,{-2,0});
		for(const auto& segment:segments)for(unsigned i=0;i<segment.size;++i){const auto offset=segment.offset+i,original=(offset/(width*av))*display.row_pitch+offset%(width*av);memory.physical[segment.address+i]=memory.physical.at(0x9000+original);wanted[offset]=expected[original];}
		const auto projected=plan_display_storage_assembly(memory,dense,sources,segments);
		for(const auto copy:projected.copies)for(unsigned i=0;i<copy.size;++i){require(canonical[copy.target+i].first==-2);canonical[copy.target+i]={int(copy.input),copy.source+i};}
		for(const auto hole:projected.vm)for(unsigned i=0;i<hole.size;++i){require(canonical[hole.address+i].first==-2);canonical[hole.address+i]={-1,0};}
		require(canonical==wanted);
	}
	for(unsigned invalid=1;invalid<=6;++invalid)
	{
		memory.invalid=invalid;bool rejected=false;
		try{(void)plan_display_storage_assembly(memory,plan_guest_surface(0x9000,1,1,4,5),sources);}catch(const std::invalid_argument&){rejected=true;}
		require(rejected);
	}
	std::cout<<"Display assembly physical oracle:"<<profiles<<" split-allocation aliases/"<<checked<<" active byte decisions,also projected todense RTT planes withreordered/top32-bit splitEAs;equal raw pointers withdistinct controlblocks staydistinct,midrow owner changes,source padding andAV padding excluded,6 absent/malformed/expired/overflow metadata refusals;zero VM pixel access\n";
}
void texture_inverse_cases()
{
	std::size_t profiles=0,bytes=0;
	for(unsigned bpp:{1u,2u,4u,8u,16u})for(unsigned bank=0;bank<4;++bank)for(unsigned pitch:{256u,512u,1024u})for(bool swizzled:{false,true})for(unsigned offset:{0u,3u})
	{
		const auto region=swizzled?std::optional<swizzle_region>{{{32,8,1},{2,1,0},{7,5,1}}}:std::nullopt;
		const auto geometry=plan_guest_surface(0,7,5,bpp,swizzled?0:pitch,region);const banked_tile_layout tile{0x40000,0x40000,pitch,bank,offset*bpp};const display_storage_source source{geometry,{{tile.address,0,tile.size}},true,1,tile};const auto inverse=texture_storage_runs(source);
		for(unsigned y=0;y<5;++y)for(unsigned x=0;x<7;++x)for(unsigned lane=0;lane<bpp;++lane)
		{
			const auto logical=swizzled?morton(x+2,y+1,0,{32,8,1})*bpp+lane:y*pitch+x*bpp+lane;const auto physical=banked_display_storage_offset(tile,tile.offset+logical);const auto mapped=texture_storage_logical(inverse,physical);require(mapped==logical);
			const auto pixel=texture_storage_pixel(geometry,logical-lane);require(pixel&&(*pixel)[0]==x&&(*pixel)[1]==y);if(lane)require(!texture_storage_pixel(geometry,logical));++bytes;
		}
		require(!texture_storage_logical(inverse,tile.size));++profiles;
	}
	for(unsigned z=0;z<4;++z)
	{
		const auto geometry=plan_guest_surface(0,7,5,4,0,swizzle_region{{16,8,4},{3,2,z},{7,5,1}});const display_storage_source source{geometry,{{0x1000,0,geometry.range.size}},true,1};const auto inverse=texture_storage_runs(source);
		for(unsigned y=0;y<5;++y)for(unsigned x=0;x<7;++x){const auto logical=morton(x+3,y+2,z,{16,8,4})*4;require(texture_storage_logical(inverse,logical)==logical);const auto pixel=texture_storage_pixel(geometry,logical);require(pixel&&(*pixel)[0]==x&&(*pixel)[1]==y);}
	}
	std::cout<<"RTT inverse metadata:"<<profiles<<" bank/pitch/offset/linear-Morton profiles,"<<bytes<<" active byte inverses andfour interleaved3D slices recover original native coordinates/lanes;gaps andmisalignment refused,zero VMpixel access\n";
}
void texture_assembly_cases()
{
	std::size_t profiles=0,planes=0,bytes=0;
	for(unsigned format:{0x81u,0x82u,0x83u,0x84u,0x85u,0x86u,0x87u,0x88u,0x8bu,0x8du,0x8eu,0x8fu,0x90u,0x91u,0x92u,0x93u,0x94u,0x95u,0x97u,0x98u,0x99u,0x9au,0x9bu,0x9cu,0x9du,0x9eu,0x9fu})
	for(unsigned dimension:{1u,2u,3u,4u})for(bool linear:{false,true})for(unsigned odd:{0u,1u})
	{
		const auto codec=describe_guest_texture_format(format);fragment_texture_registers words;words.format=(4u<<16)|((format|(linear?0x20:0))<<8)|((dimension==4?2:dimension)<<4)|8|1|(dimension==4?4:0);words.control0=1u<<31;words.image_rect=(8u<<16)|8;words.control3=((8+codec.block.width-1)/codec.block.width)*codec.block.bytes+odd;if(dimension==3)words.control3|=4u<<20;
		const auto plan=plan_guest_texture(words);const auto first=std::min<std::size_t>(7,plan.storage_bytes/3),second=std::min<std::size_t>(24,plan.storage_bytes*2/3);
		const std::array segments{guest_storage_segment{0xffff0000,0,first},guest_storage_segment{0x90000,first,second-first},guest_storage_segment{0x40000,second,plan.storage_bytes-second}};
		const auto assembly=plan_texture_assembly_planes(plan,segments);std::size_t index=0;std::vector<bool> active(plan.storage_bytes);
		for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)
		{
			const auto& plane=assembly.at(index++);require(plane.level==s.level&&plane.face==s.face&&plane.z==z&&plane.view.width==s.width&&plane.view.height==s.height&&plane.view.depth==1&&plane.view.layers==1&&plane.view.mips==1&&!plane.view.swizzled);
			std::vector<unsigned> actual(plane.view.storage_bytes);std::size_t cursor=0;
			for(auto span:plane.spans){require(span.offset==cursor);for(unsigned i=0;i<span.size;++i)actual[cursor+i]=span.address+i;cursor+=span.size;}require(cursor==actual.size());
			for(unsigned y=0;y<s.rows;++y)for(unsigned x=0;x<s.columns;++x)for(unsigned lane=0;lane<codec.block.bytes;++lane)
			{
				const auto raw=s.offset+(dimension==3&&codec.block.width==4&&z<(s.depth&~std::size_t(3))?(z/4)*4*s.row_pitch*s.storage_rows+4*(y*s.row_pitch+x*codec.block.bytes)+(z%4)*codec.block.bytes:!linear&&codec.block.height==1?morton(x,y,z,{s.columns,s.rows,s.depth})*codec.block.bytes:z*s.row_pitch*s.storage_rows+y*s.row_pitch+x*codec.block.bytes)+lane;
				const auto part=std::find_if(segments.begin(),segments.end(),[&](auto p){return raw>=p.offset&&raw-p.offset<p.size;});require(part!=segments.end());const auto ea=part->address+raw-part->offset;
				require(actual[(y*s.columns+x)*codec.block.bytes+lane]==ea&&!active[raw]);active[raw]=true;++bytes;
				const auto payload=(y*s.columns+x)*codec.block.bytes+lane;const auto* logical=texture_assembly_segment(plane.logical_spans,payload);require(logical&&std::uint64_t(logical->address)+payload-logical->offset==std::uint64_t(plan.offset)+raw);
			}
			++planes;
		}
		require(index==assembly.size());++profiles;
	}
	identity_memory memory;
	for(unsigned z=0;z<4;++z)
	{
		fragment_texture_registers words;words.format=(1u<<16)|(0x85u<<8)|(3u<<4)|8|1;words.control0=1u<<31;words.image_rect=(8u<<16)|8;words.control3=4u<<20;const auto plan=plan_guest_texture(words);
		const std::array segments{guest_storage_segment{0x1000,0,7},guest_storage_segment{0x9000,7,plan.storage_bytes-7}};auto slices=plan_texture_assembly_planes(plan,segments);
		const auto geometry=plan_guest_surface(0,8,8,4,0,swizzle_region{{8,8,4},{0,0,z},{8,8,1}});const display_storage_source source{geometry,{segments.begin(),segments.end()},true,1};
		const auto region=plan_native_texture_plane(memory,source,slices[z]);require(region&&region->pixels.x==0&&region->pixels.y==0&&region->pixels.width==8&&region->pixels.height==8&&!region->guest_row_offset);
		slices[z].spans.back().address+=1;require(!plan_native_texture_plane(memory,source,slices[z]));
	}
	fragment_texture_registers words;words.format=(1u<<16)|((0x81u|0x20u)<<8)|(2u<<4)|8|1;words.control0=1u<<31;words.image_rect=(7u<<16)|3;words.control3=29;const auto plan=plan_guest_texture(words);const std::array requested{guest_storage_segment{0x1007,0,plan.storage_bytes}};
	const auto geometry=plan_guest_surface(0,6,5,4,29);const display_storage_source source{geometry,{{0x1000,0,geometry.range.size}},true,1};const auto slices=plan_texture_assembly_planes(plan,requested);const auto mapped=plan_native_texture_plane(memory,source,slices[0]);
	require(mapped&&mapped->pixels.x==1&&mapped->pixels.y==0&&mapped->pixels.width==3&&mapped->pixels.height==3&&mapped->guest_row_offset==3);
	std::cout<<"RTT assembly plane oracle:"<<profiles<<" layouts/"<<planes<<" planes/"<<bytes<<" bytes,27formats/1D-2D-3D-cube/linear-Morton/mips/oddpitch/top32-bit splitsegments;allactive bytes inoriginal guestorder,no padding/facemip/Zgap captures,interleaved3D native rectangles andwrong laterEA/odd-source-pitch-prefix proofs passed;zero VMpixel access\n";
}
void stored_border_projection_cases()
{
	std::size_t profiles = 0, adjusted = 0, checked = 0;
	for (unsigned format : {0x81u, 0x85u, 0x91u, 0x92u, 0x9au, 0x9bu})
		for (unsigned base_width : {1u, 4u, 7u})
			for (unsigned height : {1u, 5u})
				for (unsigned padding : {0u, 1u, 5u})
				for (unsigned mode : {0u, 1u, 2u})
				{
					if (mode == 2 && (height != 1 || padding))
						continue;
					fragment_texture_registers r;
					r.offset = 0x200;
					r.format = (1u << 16) | ((format | 0x20) << 8) | (2u << 4) | 2;
					r.control0 = 1u << 31;
					const unsigned declared_width = mode == 1 ? base_width + 6 : base_width;
					const unsigned declared_height = mode == 2 ? 5 : height;
					r.image_rect = (declared_width << 16) | declared_height;
					const unsigned bpp = describe_guest_texture_format(format).block.bytes;
					const unsigned pitch = (base_width + 2) * bpp + padding;
					const unsigned width = mode == 1 ? base_width + padding / bpp : base_width;
					const unsigned row = (width + 2) * bpp;
					r.control3 = mode == 2 ? 0 : pitch;
					const auto plan = plan_guest_texture(r);
					require(plan.adjusted_pitch == (mode != 0) && plan.width == width && plan.height == height &&
						plan.subresources.front().row_pitch == pitch && plan.storage_bytes == pitch * (height + 2));
					const unsigned first = row - 1;
					const unsigned second = row + (pitch > row ? 1 : 0);
					const std::array segments{
						guest_storage_segment{0xffffffffu - first + 1, 0, first},
						guest_storage_segment{0x10000, first, second - first},
						guest_storage_segment{0x20000, second, plan.storage_bytes - second}};
					const auto planes = plan_texture_assembly_planes(plan, segments);
					require(planes.size() == 1);
					const auto& p = planes.front();
					require(!p.view.stored_border && p.view.width == width + 2 && p.view.height == height + 2);
					std::vector<std::uint32_t> physical;
					for (const auto span : p.spans)
					{
						require(span.offset == physical.size());
						for (std::size_t i = 0; i < span.size; ++i) physical.push_back(span.address + i);
					}
					std::vector<std::uint32_t> watch;
					for (const auto& coverage : stored_border_texture_coverage(plan, segments))
					{
						guest_surface_byte_cursor cursor(coverage);
						while (const auto run = cursor.next())
							for (std::size_t i = 0; i < run->size; ++i) watch.push_back(run->address + i);
					}
					require(physical == watch && physical.size() == row * (height + 2));
					for (unsigned y = 0; y < height + 2; ++y)
						for (unsigned byte = 0; byte < row; ++byte)
						{
							const auto raw = y * pitch + byte;
							const auto ea = raw < first ? 0xffffffffu - first + 1 + raw :
								raw < second ? 0x10000u + raw - first : 0x20000u + raw - second;
							const auto packed = y * row + byte;
							require(physical.at(packed) == ea);
							const auto* logical = texture_assembly_segment(p.logical_spans, packed);
							require(logical && logical->address + packed - logical->offset == r.offset + raw);
							++checked;
						}
					++profiles;
					adjusted += mode != 0;
				}
	std::cout << "Linear stored border address oracle: " << profiles << " segmented layouts including " << adjusted << " adjusted pitches, "
		<< checked << " active byte addresses and watches, no padding or VM reads\n";
}
void morton_border_projection_cases()
{
	std::size_t profiles = 0, checked = 0;
	for (const auto& [width, height] : {std::pair{1u, 1u}, {15u, 3u}, {5u, 14u}, {16u, 17u}})
		for (const unsigned format : {0x81u, 0x85u, 0x91u, 0x9bu})
		{
			const auto bpp = describe_guest_texture_format(format).block.bytes;
			fragment_texture_registers r{};
			r.offset = 0x200;
			r.format = (1u << 16) | (format << 8) | (2u << 4) | 2;
			r.control0 = 1u << 31;
			r.image_rect = (width << 16) | height;
			const auto plan = plan_guest_texture(r);
			const auto image = guest_texture_image_subresource(plan, plan.subresources.front());
			const auto sx = std::bit_ceil(width + 8), sy = std::bit_ceil(height + 8);
			require(plan.storage_bytes == sx * sy * bpp);
			const auto mid = 15 * bpp + 1; // Morton (3,3), then one byte into the first ring texel.
			const auto cut = mid + 13;
			const std::array segments{guest_storage_segment{0x1000, 0, mid},
				guest_storage_segment{0x20000, mid, cut - mid},
				guest_storage_segment{0x40000, cut, plan.storage_bytes - cut}};
			std::vector<std::uint32_t> expected((width + 2) * (height + 2) * bpp);
			std::vector<bool> active(plan.storage_bytes);
			const auto address = [&](std::size_t raw)
			{
				if (raw < mid) return 0x1000u + raw;
				if (raw < cut) return 0x20000u + raw - mid;
				return 0x40000u + raw - cut;
			};
			for (unsigned ordinal = 0; ordinal < sx * sy; ++ordinal)
			{
				unsigned x = 0, y = 0, code = ordinal;
				for (unsigned bit = 1; bit < std::max(sx, sy); bit <<= 1)
				{
					if (bit < sx) { x |= (code & 1) * bit; code >>= 1; }
					if (bit < sy) { y |= (code & 1) * bit; code >>= 1; }
				}
				if (x < 3 || x >= width + 5 || y < 3 || y >= height + 5) continue;
				if (guest_texture_block_offset(plan, image, x - 3, y - 3, 0) != ordinal * bpp)
					throw std::runtime_error("Morton stored border block offset omitted its storage origin");
				for (unsigned byte = 0; byte < bpp; ++byte)
				{
					const auto raw = ordinal * bpp + byte;
					active[raw] = true;
					expected[((y - 3) * (width + 2) + x - 3) * bpp + byte] = static_cast<std::uint32_t>(address(raw));
				}
			}
			const auto planes = plan_texture_assembly_planes(plan, segments);
			require(planes.size() == 1 && planes[0].view.width == width + 2 && planes[0].view.height == height + 2);
			std::size_t cursor = 0;
			for (const auto& span : planes[0].spans)
			{
				require(span.offset == cursor);
				for (unsigned i = 0; i < span.size; ++i) require(expected.at(cursor++) == span.address + i);
			}
			require(cursor == expected.size());
			std::vector<bool> captured(active.size());
			for (const auto& part : project_stored_border_texture(plan, segments))
				for (std::size_t i = 0; i < part.coverage.range.size; ++i)
				{
					const auto offset = part.offset + i;
					require(!captured.at(offset) && address(offset) == part.coverage.range.address + i);
					captured[offset] = true;
				}
			require(captured == active);
			checked += cursor;
			++profiles;
		}
	std::cout << "Morton stored border address oracle: " << profiles << " segmented layouts, "
		<< checked << " active byte addresses for capture and native assembly\n";
}
void stored_border_mip_projection_cases(bool one_dimensional = false)
{
	std::size_t profiles = 0, levels = 0, bytes = 0, guards = 0;
	for (const auto [base_width, base_height] : {std::pair{5u, 3u}, {17u, 9u}, {3u, 19u}, {32u, 1u}})
		for (const unsigned format : {0x81u, 0x85u, 0x91u, 0x9bu})
			for (const unsigned mode : {0u, 1u, 2u})
			{
				const bool swizzled = mode == 2;
				const auto bpp = describe_guest_texture_format(format).block.bytes;
				fragment_texture_registers r{};
				r.offset = 0x200;
				r.format = (15u << 16) | ((format | (swizzled ? 0u : 0x20u)) << 8) | ((one_dimensional ? 1u : 2u) << 4) | 2u;
				r.control0 = 1u << 31;
				r.image_rect = ((base_width + (mode == 1 ? 4u : 0u)) << 16) | base_height;
				const auto pitch = (base_width + 2) * bpp + (bpp > 1 ? 1u : 0u);
				r.control3 = static_cast<std::uint32_t>(pitch);
				const auto plan = plan_guest_texture(r);
				const auto logical_height = one_dimensional ? 1u : base_height;
				require(plan.width == base_width && plan.height == logical_height && plan.adjusted_pitch == (mode == 1));
				require(plan.mips == std::bit_width(std::max(base_width, logical_height)) && uses_stored_border_mip_atlas(plan));
				const auto next = plan.subresources.at(1).offset;
				std::vector cuts{std::size_t{0}, std::size_t{1}, next - 1, next + 1, plan.storage_bytes - 1, plan.storage_bytes};
				if (one_dimensional)
				{
					const auto middle = swizzled ? morton(3, 4, 0, {std::bit_ceil(std::size_t(base_width) + 8), 16, 1}) * bpp : pitch;
					cuts.insert(cuts.begin() + 2, middle + 1);
				}
				std::vector<guest_storage_segment> segments;
				for (std::size_t i = 0; i + 1 < cuts.size(); ++i)
					segments.push_back({static_cast<std::uint32_t>(0x10000 + 0x20000 * i), cuts[i], cuts[i + 1] - cuts[i]});
				const auto address = [&](std::size_t offset)
				{
					const auto index = std::upper_bound(cuts.begin(), cuts.end(), offset) - cuts.begin() - 1;
					return static_cast<std::uint32_t>(0x10000 + 0x20000 * index + offset - cuts[index]);
				};
				const auto planes = plan_texture_assembly_planes(plan, segments);
				require(planes.size() == plan.mips);
				std::vector<bool> expected_active(plan.storage_bytes), observed_active(plan.storage_bytes);
				std::size_t origin = 0;
				for (std::size_t mip = 0; mip < plan.mips; ++mip)
				{
					const auto width = std::max<std::size_t>(1, base_width >> mip);
					const auto height = std::max<std::size_t>(1, logical_height >> mip);
					const auto sx = std::bit_ceil(width + 8), sy = std::bit_ceil(height + 8);
					const auto& plane = planes[mip];
					const auto rows = one_dimensional ? 1 : height + 2;
					require(plane.level == mip && plane.view.width == width + 2 && plane.view.height == rows);
					for (std::size_t y = 0; y < rows; ++y)
						for (std::size_t x = 0; x < width + 2; ++x)
						{
							const auto source_row = y + std::size_t(one_dimensional);
							const auto texel = swizzled ? morton(x + 3, source_row + 3, 0, {sx, sy, 1}) * bpp :
								source_row * pitch + x * bpp;
							for (std::size_t byte = 0; byte < bpp; ++byte)
							{
								const auto raw = origin + texel + byte, packed = (y * (width + 2) + x) * bpp + byte;
								const auto* physical = texture_assembly_segment(plane.spans, packed);
								const auto* logical = texture_assembly_segment(plane.logical_spans, packed);
								require(physical && physical->address + packed - physical->offset == address(raw));
								require(logical && logical->address + packed - logical->offset == r.offset + raw);
								require(!expected_active.at(raw));
								expected_active[raw] = true;
								++bytes;
							}
						}
					origin += swizzled ? sx * sy * bpp : pitch * (height + 2);
					++levels;
				}
				require(origin == plan.storage_bytes);
				const auto parts = project_stored_border_texture(plan, segments);
				for (const auto& part : parts)
				{
					if (one_dimensional) require(guest_surface_active_bytes(part.coverage) == part.coverage.range.size);
					guest_surface_byte_cursor cursor(part.coverage);
					while (const auto run = cursor.next())
						for (std::size_t i = 0; i < run->size; ++i)
						{
							const auto raw = part.offset + run->address - part.coverage.range.address + i;
							require(!observed_active.at(raw) && run->address + i == address(raw));
							observed_active[raw] = true;
						}
				}
				require(observed_active == expected_active);
				try { (void)project_stored_border_texture(plan, segments, parts.size() - 1); }
				catch (const std::length_error&) { ++guards; }
				++profiles;
			}
	require(guards == profiles);
	std::cout << (one_dimensional ? "Stored 1D border projection: " : "Stored-border mip projection: ") << profiles << " linear and Morton layouts, " << levels << " levels, " << bytes
		<< " exact active byte addresses for capture and RTT, " << guards << " global budget guards passed\n";
}

void cube_border_projection_cases()
{
	std::size_t profiles = 0, planes_checked = 0, bytes_checked = 0;
	for (const unsigned format : {0x81u, 0x85u, 0x92u, 0x9bu})
		for (const bool linear : {false, true})
			for (const unsigned side : {1u, 5u, 11u})
			{
				fragment_texture_registers words;
				words.offset = 0x200;
				words.format = (8u << 16) | ((format | (linear ? 0x20u : 0u)) << 8) | (2u << 4) | 6u;
				words.control0 = 1u << 31;
				words.image_rect = (side << 16) | side;
				const auto bpp = describe_guest_texture_format(format).block.bytes;
				const auto pitch = (side + 2) * bpp + 1;
				words.control3 = pitch;
				const auto plan = plan_guest_texture(words);
				const auto second_face = plan.subresources.at(plan.mips).offset;
				const auto start = linear ? 0 : morton(3, 3, 0, {std::bit_ceil(side + 8u), std::bit_ceil(side + 8u), 1}) * bpp;
				const std::array cuts{std::size_t{0}, start + 1, second_face + start + 1, plan.storage_bytes};
				const std::array bases{0x600000u, 0x800000u, 0x400000u};
				std::vector<guest_storage_segment> segments;
				for (std::size_t i = 0; i < bases.size(); ++i)
					segments.push_back({bases[i], cuts[i], cuts[i + 1] - cuts[i]});
				const auto physical = [&](std::size_t raw)
				{
					const auto i = std::upper_bound(cuts.begin(), cuts.end(), raw) - cuts.begin() - 1;
					return bases.at(i) + raw - cuts.at(i);
				};
				const auto planes = plan_texture_assembly_planes(plan, segments);
				std::vector<bool> expected(plan.storage_bytes), observed(plan.storage_bytes);
				std::size_t base = 0, index = 0;
				for (unsigned face = 0; face < 6; ++face)
				{
					for (unsigned mip = 0; mip < plan.mips; ++mip)
					{
						const auto w = std::max(1u, side >> mip), padded = std::bit_ceil(w + 8);
						const auto& plane = planes.at(index++);
						require(plane.face == face && plane.level == mip && plane.view.width == w + 2 && plane.view.height == w + 2);
						for (unsigned y = 0; y < w + 2; ++y)
							for (unsigned x = 0; x < w + 2; ++x)
								for (unsigned byte = 0; byte < bpp; ++byte)
								{
									const auto raw = base + byte + (linear ? y * pitch + x * bpp : morton(x + 3, y + 3, 0, {padded, padded, 1}) * bpp);
									const auto dense = ((y * (w + 2) + x) * bpp) + byte;
									const auto* span = texture_assembly_segment(plane.spans, dense);
									require(span && span->address + dense - span->offset == physical(raw));
									require(!expected.at(raw));
									expected[raw] = true;
									++bytes_checked;
								}
						base += linear ? pitch * (w + 2) : padded * padded * bpp;
						++planes_checked;
					}
					if (!linear && face < 5) base = (base + 127) & ~std::size_t(127);
				}
				require(base == plan.storage_bytes && index == planes.size());
				for (const auto& part : project_stored_border_texture(plan, segments))
					for (std::size_t i = 0; i < part.coverage.range.size; ++i)
					{
						const auto raw = part.offset + i;
						require(!observed.at(raw) && physical(raw) == part.coverage.range.address + i);
						observed[raw] = true;
					}
				require(observed == expected);
				++profiles;
			}
	std::cout << "Stored cube border projection: " << profiles << " segmented layouts, " << planes_checked
		<< " face-mip planes, " << bytes_checked << " exact guest byte addresses passed\n";
}

void volume_border_projection_cases()
{
	std::size_t profiles = 0, planes_checked = 0, bytes_checked = 0;
	for (const unsigned format : {0x81u, 0x85u, 0x92u, 0x9bu})
		for (const bool linear : {false, true})
			for (const auto [width, height, depth] : {std::array{5u, 4u, 4u}, {3u, 7u, 2u}, {7u, 2u, 1u}})
			{
				fragment_texture_registers words;
				words.offset = 0x200;
				words.format = (8u << 16) | ((format | (linear ? 0x20u : 0u)) << 8) | (3u << 4) | 2u;
				words.control0 = 1u << 31;
				words.image_rect = (width << 16) | height;
				const auto bpp = describe_guest_texture_format(format).block.bytes;
				const auto pitch = (width + 2) * bpp + 1;
				words.control3 = (depth << 20) | pitch;
				const auto plan = plan_guest_texture(words);
				const auto middle = linear ? pitch : morton(3, 4, 0, {std::bit_ceil(width + 8u), std::bit_ceil(height + 8u), depth}) * bpp;
				const std::array cuts{std::size_t{0}, middle + 1, plan.storage_bytes / 2, plan.storage_bytes};
				const std::array bases{0x600000u, 0x800000u, 0x400000u};
				std::vector<guest_storage_segment> segments;
				for (std::size_t i = 0; i < bases.size(); ++i)
					segments.push_back({bases[i], cuts[i], cuts[i + 1] - cuts[i]});
				const auto physical = [&](std::size_t raw)
				{
					const auto i = std::upper_bound(cuts.begin(), cuts.end(), raw) - cuts.begin() - 1;
					return bases.at(i) + raw - cuts.at(i);
				};
				const auto planes = plan_texture_assembly_planes(plan, segments);
				std::vector<bool> expected(plan.storage_bytes), observed(plan.storage_bytes);
				std::size_t base = 0, index = 0;
				for (unsigned mip = 0; mip < plan.mips; ++mip)
				{
					const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip), d = std::max(1u, depth >> mip);
					const std::array<std::size_t, 3> shape{std::bit_ceil(w + 8), std::bit_ceil(h + 8), d};
					for (unsigned z = 0; z < d; ++z)
					{
						const auto& plane = planes.at(index++);
						require(plane.z == z && plane.level == mip && plane.view.width == w + 2 && plane.view.height == h + 2);
						for (unsigned y = 0; y < h + 2; ++y)
							for (unsigned x = 0; x < w + 2; ++x)
								for (unsigned byte = 0; byte < bpp; ++byte)
								{
									const auto raw = base + byte + (linear ? (z * (h + 2) + y) * pitch + x * bpp :
										morton(x + 3, y + 3, z, shape) * bpp);
									const auto dense = ((y * (w + 2) + x) * bpp) + byte;
									const auto* span = texture_assembly_segment(plane.spans, dense);
									require(span && span->address + dense - span->offset == physical(raw));
									expected.at(raw) = true;
									++bytes_checked;
								}
						++planes_checked;
					}
					base += d * (linear ? pitch * (h + 2) : shape[0] * shape[1] * bpp);
				}
				require(base == plan.storage_bytes && index == planes.size());
				for (const auto& part : project_stored_border_texture(plan, segments))
					for (std::size_t i = 0; i < part.coverage.range.size; ++i)
					{
						const auto raw = part.offset + i;
						require(!observed.at(raw) && physical(raw) == part.coverage.range.address + i);
						observed[raw] = true;
					}
				require(observed == expected);
				++profiles;
			}
	std::cout << "Stored volume border projection: " << profiles << " segmented layouts, " << planes_checked
		<< " XY mip planes, " << bytes_checked << " exact guest byte addresses passed\n";
}

void paired_border_projection_cases()
{
	std::size_t profiles = 0, rows = 0, bytes = 0;
	for (const auto format : {0x8du, 0x8eu})
		for (const bool one : {false, true})
			for (const auto width : {1u, 4u, 5u, 15u})
				for (const auto pad : {0u, 1u})
				{
					fragment_texture_registers words;
					words.offset = 0x200;
					words.format = (8u << 16) | ((format | 0x20u) << 8) | ((one ? 1u : 2u) << 4) | 2u;
					words.control0 = 1u << 31;
					words.image_rect = (width << 16) | 3u;
					const std::size_t pitch = ((width + 1) / 2 + 2) * 4 + pad;
					words.control3 = pitch;
					const auto plan = plan_guest_texture(words);
					const auto cut = (one ? pitch : 0) + 1;
					const std::array segments{guest_storage_segment{0x10000, 0, cut},
						guest_storage_segment{0x20000, cut, plan.storage_bytes - cut}};
					const auto physical = [&](std::size_t offset) { return offset < cut ? 0x10000 + offset : 0x20000 + offset - cut; };
					const auto planes = plan_texture_assembly_planes(plan, segments);
					require(planes.size() == plan.mips);
					std::vector<bool> expected(plan.storage_bytes), observed(plan.storage_bytes);
					std::size_t base = 0;
					for (unsigned mip = 0; mip < plan.mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip), h = one ? 1u : std::max(1u, 3u >> mip);
						const auto row_bytes = ((w + 1) / 2 + 2) * 4;
						const auto& plane = planes[mip];
						require(plane.view.width == w + 2 && plane.view.height == (one ? 1 : h + 2));
						require(!plane.view.stored_border && plane.view.subresources[0].border == 1);
						for (unsigned y = 0; y < plane.view.height; ++y)
							for (unsigned byte = 0; byte < row_bytes; ++byte)
							{
								const auto packed = y * row_bytes + byte;
								const auto raw = base + (y + unsigned(one)) * pitch + byte;
								const auto* span = texture_assembly_segment(plane.spans, packed);
								require(span && span->address + packed - span->offset == physical(raw));
								expected.at(raw) = true;
								++bytes;
							}
						rows += plane.view.height;
						base += (h + 2) * pitch;
					}
					require(base == plan.storage_bytes);
					for (const auto& part : project_stored_border_texture(plan, segments))
					{
						guest_surface_byte_cursor cursor(part.coverage);
						while (const auto run = cursor.next())
							for (std::size_t i = 0; i < run->size; ++i)
							{
								const auto raw = part.offset + run->address - part.coverage.range.address + i;
								require(!observed.at(raw) && physical(raw) == run->address + i);
								observed[raw] = true;
							}
					}
					require(observed == expected);
					++profiles;
				}
	std::cout << "Paired border projection: " << profiles << " segmented layouts, " << rows
		<< " mip rows, " << bytes << " exact active block bytes for capture and RTT passed\n";
}

void morton_paired_projection_cases()
{
	std::size_t profiles = 0, planes_checked = 0, bytes_checked = 0, guards = 0;
	for (const auto format : {0x8du, 0x8eu})
		for (const auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::three, vertex_texture_dimension::cube})
			for (const bool border : {false, true})
				for (unsigned shape = 0; shape < 5; ++shape)
				{
					if (!border && dimension == vertex_texture_dimension::cube && shape == 4) continue;
					const unsigned width = border ? std::array{1u, 4u, 5u, 15u, 3u}[shape] : std::array{1u, 4u, 8u, 16u, 3u}[shape];
					const unsigned height = dimension == vertex_texture_dimension::one ? 1 : dimension == vertex_texture_dimension::cube ? width : 4;
					const unsigned depth = dimension == vertex_texture_dimension::three ? 4 : 1;
					fragment_texture_registers words;
					words.offset = 0x200;
					words.format = (4u << 16) | (format << 8) | ((dimension == vertex_texture_dimension::one ? 1u : dimension == vertex_texture_dimension::three ? 3u : 2u) << 4) |
						(dimension == vertex_texture_dimension::cube ? 4u : 0u) | (border ? 0u : 8u) | 2u;
					words.control0 = 1u << 31; words.image_rect = width << 16 | height; words.control3 = depth << 20;
					const auto plan = plan_guest_texture(words);
					std::vector<guest_storage_segment> segments;
					const auto cut = std::min<std::size_t>(13, plan.storage_bytes);
					segments.push_back({0xfffffff3, 0, cut});
					if (cut < plan.storage_bytes) segments.push_back({0x20000, cut, plan.storage_bytes - cut});
					const auto physical = [&](std::size_t at) { return at < cut ? 0xfffffff3u + at : 0x20000u + at - cut; };
					const auto planes = plan_texture_assembly_planes(plan, segments);
					std::vector<bool> expected(plan.storage_bytes), observed(plan.storage_bytes);
					std::size_t base = 0, index = 0;
					for (unsigned face = 0; face < plan.layers; ++face)
					{
						for (unsigned mip = 0; mip < plan.mips; ++mip)
						{
							const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip), d = std::max(1u, depth >> mip);
							const auto columns = (w + 1) / 2 + (border ? 2 : 0), rows = dimension == vertex_texture_dimension::one ? 1 : h + (border ? 2 : 0);
							const std::array<std::size_t, 3> storage{border ? std::bit_ceil((w + 1) / 2 + 8) : (w + 1) / 2,
								border ? std::bit_ceil(h + 8) : h, d};
							const auto& subresource = plan.subresources.at(face * plan.mips + mip);
							require(subresource.offset == base && subresource.storage_bytes == storage[0] * storage[1] * storage[2] * 4);
							for (unsigned z = 0; z < d; ++z)
							{
								const auto& plane = planes.at(index++);
								require(plane.level == mip && plane.face == face && plane.z == z && plane.view.width == w + (border ? 2 : 0) && plane.view.height == rows);
								for (unsigned y = 0; y < rows; ++y)
									for (unsigned x = 0; x < columns; ++x)
									{
										const auto raw = base + 4 * morton(x + (border ? 3 : 0), y + (border ? dimension == vertex_texture_dimension::one ? 4 : 3 : 0), z, storage);
										const auto image = guest_texture_image_subresource(plan, subresource);
										if (guest_texture_block_offset(plan, image, x, y, z) != raw)
											throw std::runtime_error("Paired Morton block offset does not match independent address");
										for (unsigned c = 0; c < 4; ++c)
										{
											const auto packed = (y * columns + x) * 4 + c;
											const auto* span = texture_assembly_segment(plane.spans, packed);
											if (!span || std::uint64_t(span->address) + packed - span->offset != physical(raw + c))
												throw std::runtime_error("Paired Morton span mismatch: profile " + std::to_string(profiles) + ", x " + std::to_string(x) + ", y " + std::to_string(y) + ", z " + std::to_string(z) + ", raw " + std::to_string(raw + c) + ", wanted EA " + std::to_string(physical(raw + c)) + ", actual EA " + std::to_string(span ? std::uint64_t(span->address) + packed - span->offset : 0));
											const auto* logical = texture_assembly_segment(plane.logical_spans, packed);
											require(logical && std::uint64_t(logical->address) + packed - logical->offset == plan.offset + raw + c);
											require(!expected.at(raw + c)); expected[raw + c] = true; ++bytes_checked;
										}
									}
								++planes_checked;
							}
							base += storage[0] * storage[1] * storage[2] * 4;
						}
						if (face + 1 < plan.layers) base = (base + 127) & ~std::size_t(127);
					}
					if (border)
					{
						for (const auto& part : project_stored_border_texture(plan, segments))
						{
							guest_surface_byte_cursor cursor(part.coverage);
							while (const auto run = cursor.next())
								for (std::size_t c = 0; c < run->size; ++c)
								{
									const auto at = part.offset + run->address - part.coverage.range.address + c;
									require(!observed.at(at) && physical(at) == run->address + c); observed[at] = true;
								}
						}
						require(observed == expected);
					}
					require(index == planes.size() && base == plan.storage_bytes);
					bool refused = false; try { (void)plan_texture_assembly_planes(plan, segments, 0); } catch (const std::length_error&) { refused = true; }
					require(refused); ++guards; ++profiles;
				}
	std::cout << "Paired Morton projection: " << profiles << " ordinary/stored-border layouts, " << planes_checked << " mip/face/Z planes, " << bytes_checked
		<< " independent active-byte addresses, " << guards << " run-budget guards; 1D/2D/3D/cube, odd widths, split top-EA blocks and untouched padding passed\n";
}

void vtc_projection_cases()
{
	std::size_t profiles=0,planes=0,blocks=0,bytes=0;unsigned guards=0;
	for(unsigned format:{0x86u,0x87u,0x88u})for(unsigned width:{1u,7u,16u,31u})for(unsigned height:{1u,5u,16u,25u})for(unsigned depth:{1u,2u,3u,4u,5u,7u,11u})for(unsigned odd:{0u,1u})
	{
		fragment_texture_registers r;r.offset=0x200;r.format=(4u<<16)|((format|0x20)<<8)|(3u<<4)|8|1;r.control0=1u<<31;r.image_rect=(width<<16)|height;const unsigned bb=describe_guest_texture_format(format).block.bytes;r.control3=(depth<<20)|(((width+3)/4)*bb+odd);const auto plan=plan_guest_texture(r);std::vector<std::vector<std::size_t>> expected;std::vector<bool> active(plan.storage_bytes);
		for(const auto& s:plan.subresources)
		{
			std::vector<std::vector<std::size_t>> slices(s.depth,std::vector<std::size_t>(s.columns*s.rows));std::size_t cursor=s.offset;const auto put=[&](unsigned x,unsigned y,unsigned z){slices[z][y*s.columns+x]=cursor;require(guest_texture_block_offset(plan,s,x,y,z)==cursor);for(unsigned c=0;c<bb;++c){require(!active.at(cursor+c));active[cursor+c]=true;}cursor+=bb;++blocks;};unsigned z=0;
			for(;z+4<=s.depth;z+=4)for(unsigned y=0;y<s.rows;++y){for(unsigned x=0;x<s.columns;++x)for(unsigned lane=0;lane<4;++lane)put(x,y,z+lane);cursor+=4*(s.row_pitch-s.columns*bb);}
			for(;z<s.depth;++z)for(unsigned y=0;y<s.rows;++y){for(unsigned x=0;x<s.columns;++x)put(x,y,z);cursor+=s.row_pitch-s.columns*bb;}require(cursor==s.offset+s.storage_bytes);for(auto& slice:slices)expected.push_back(std::move(slice));
			for(auto xyz:std::array<std::array<std::size_t,3>,3>{{{s.columns,0,0},{0,s.rows,0},{0,0,s.depth}}}){bool refused=false;try{(void)guest_texture_block_offset(plan,s,xyz[0],xyz[1],xyz[2]);}catch(const std::out_of_range&){refused=true;}require(refused);++guards;}
		}
		const std::size_t split=std::min<std::size_t>(plan.storage_bytes,13);const std::vector<guest_storage_segment> segments=split==plan.storage_bytes?std::vector<guest_storage_segment>{{0xfffffff3,0,split}}:std::vector<guest_storage_segment>{{0xfffffff3,0,split},{0x90000,split,plan.storage_bytes-split}};const auto projection=plan_texture_assembly_planes(plan,segments);require(projection.size()==expected.size());
		for(unsigned p=0;p<projection.size();++p)
		{
			const auto& plane=projection[p];std::vector<bool> seen(plane.view.storage_bytes);
			for(const auto& span:plane.spans)for(unsigned i=0;i<span.size;++i){const auto payload=span.offset+i,raw=expected[p][payload/bb]+payload%bb;const auto ea=raw<split?std::uint64_t(0xfffffff3)+raw:0x90000+raw-split;require(ea==std::uint64_t(span.address)+i&&!seen.at(payload));seen[payload]=true;const auto* logical=texture_assembly_segment(plane.logical_spans,payload);require(logical&&std::uint64_t(logical->address)+payload-logical->offset==r.offset+raw);++bytes;}
			require(std::all_of(seen.begin(),seen.end(),[](auto x){return x;}));++planes;
		}
		++profiles;
	}
	std::cout<<"VTC address oracle:"<<profiles<<" BC1/2/3 oddpitch/partialedge/depth1-2-3-4-5-7-11 layouts,"<<planes<<" planes,"<<blocks<<" independently enumerated bricks/tailblocks,"<<bytes<<" exact physical/logical bytes,"<<guards<<" XYZ bounds refusals;midblock top32bit split,no padding projection orVMpixel access passed\n";
}
void logical_texture_assembly_cases()
{
	identity_memory memory;std::size_t profiles=0,planes=0,checked=0,refused=0,mixed=0;bool bad_mapping=false;
	const auto ea=[](std::uint32_t logical){return logical<0x100000?0xfff00000u+logical:0x90000u+logical-0x100000;};
	const auto segments=[&](std::uint32_t logical,std::size_t bytes){std::vector<guest_storage_segment> result;std::size_t at=0;while(at<bytes){const auto count=std::min<std::size_t>(bytes-at,0x100000-((logical+at)&0xfffff));result.push_back({ea(logical+at),at,count});at+=count;}return result;};
	for(unsigned bpp:{1u,4u,16u})for(bool source_morton:{false,true})for(unsigned pitch:{256u,768u})for(unsigned bank=0;bank<4;++bank)for(bool crossing:{false,true})for(unsigned format:{0x81u,0x85u,0x9au})for(unsigned dimension:{2u,3u,4u})for(bool target_linear:{false,true})for(unsigned priority=0;priority<3;++priority)
	{
		const banked_tile_layout tile{0xffff0000,0x40000,pitch,bank,crossing?0x10000-4*bpp:0};const unsigned origin=0xf0000+tile.offset;
		const auto swizzle=source_morton?std::optional<swizzle_region>{{{8,8,1},{1,2,0},{3,3,1}}}:std::nullopt;const auto geometry=plan_guest_surface(0,3,3,bpp,source_morton?0:pitch,swizzle);
		const auto other=plan_guest_surface(0,7,2,1,11);const std::array<display_storage_source,2> sources{{{geometry,segments(0xf0000,tile.size),true,3,tile},{other,segments(origin+2,other.range.size),priority!=0,priority==2?3u:9u}}};
		const std::array<std::optional<std::uint32_t>,2> origins{origin,{}};
		std::map<unsigned,std::size_t> logical_a,physical_a,physical_b;
		for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)for(unsigned lane=0;lane<bpp;++lane)
		{
			const auto raw=source_morton?morton(x+1,y+2,0,{8,8,1})*bpp+lane:y*pitch+x*bpp+lane;logical_a[origin+raw]=raw;const auto physical=banked_display_storage_offset(tile,tile.offset+raw);if(physical<tile.size)physical_a[ea(0xf0000+physical)]=physical;
		}
		for(unsigned y=0;y<2;++y)for(unsigned x=0;x<7;++x)physical_b[ea(origin+2+y*11+x)]=y*11+x;
		fragment_texture_registers words;words.offset=origin-1;words.format=(2u<<16)|((format|(target_linear?0x20:0))<<8)|((dimension==4?2:dimension)<<4)|8|2|(dimension==4?4:0);words.control0=1u<<31;words.image_rect=(8u<<16)|8;words.control3=(dimension==3?4u<<20:0)|(target_linear?8*describe_guest_texture_format(format).block.bytes+1:0);const auto plan=plan_guest_texture(words);
		for(auto plane:plan_texture_assembly_planes(plan,segments(plan.offset,plan.storage_bytes)))
		{
			using decision=std::tuple<int,bool,std::size_t>;std::vector<decision> wanted(plane.view.storage_bytes,{-1,false,0}),actual=wanted;bool ambiguous=false;
			for(std::size_t byte=0;byte<wanted.size();++byte)
			{
				const auto* real=texture_assembly_segment(plane.spans,byte);const auto* logical=texture_assembly_segment(plane.logical_spans,byte);require(real&&logical);const unsigned address=real->address+byte-real->offset,offset=logical->address+byte-logical->offset;
				std::vector<decision> candidates;if(const auto it=physical_a.find(address);it!=physical_a.end())candidates.push_back({0,false,it->second});if(const auto it=physical_b.find(address);it!=physical_b.end())candidates.push_back({1,false,it->second});if(const auto it=logical_a.find(offset);it!=logical_a.end())candidates.push_back({0,true,it->second});
				for(const auto candidate:candidates)
				{
					auto& best=wanted[byte];if(std::get<0>(best)<0){best=candidate;continue;}const auto& a=sources[std::get<0>(best)];const auto& b=sources[std::get<0>(candidate)];const auto pa=std::pair{a.dirty,a.epoch},pb=std::pair{b.dirty,b.epoch};if(pa==pb&&std::get<0>(best)!=std::get<0>(candidate))ambiguous=true;if(pb>pa||(pb==pa&&std::get<0>(best)==std::get<0>(candidate)&&std::get<1>(candidate)))best=candidate;
				}
			}
			bool rejected=false;texture_storage_assembly result;try{result=plan_texture_storage_assembly(memory,plane,sources,origins);}catch(const std::invalid_argument&){rejected=true;}require(rejected==ambiguous);++planes;if(rejected){++refused;continue;}
			std::vector<bool> covered(wanted.size());bool raw=false,logical=false;for(const auto& copy:result.bytes.copies)for(unsigned i=0;i<copy.size;++i){require(!covered.at(copy.target+i));covered[copy.target+i]=true;const auto owner=result.sources.at(copy.input);const auto mode=result.layouts[copy.input].logical_storage_offset.has_value();actual[copy.target+i]={int(owner),mode,copy.source+i};raw|=!mode&&owner==0;logical|=mode&&owner==0;}for(const auto hole:result.bytes.vm)for(unsigned i=0;i<hole.size;++i){require(!covered.at(hole.address+i));covered[hole.address+i]=true;}require(actual==wanted&&std::all_of(covered.begin(),covered.end(),[](bool v){return v;}));checked+=wanted.size();mixed+=raw&&logical;
			if(logical&&!bad_mapping){auto wrong=plane;const auto copy=std::find_if(result.bytes.copies.begin(),result.bytes.copies.end(),[&](const auto& p){return result.layouts[p.input].logical_storage_offset.has_value();});auto* span=const_cast<guest_storage_segment*>(texture_assembly_segment(wrong.spans,copy->target));span->address=0x70000;bool failed=false;try{(void)plan_texture_storage_assembly(memory,wrong,sources,origins);}catch(const std::invalid_argument&){failed=true;}require(failed);bad_mapping=true;}
		}
		++profiles;
	}
	// Only active bytes must fit the original tile;last-row padding is not
	// owned storage and must never be fabricated into an IO projection.
	const banked_tile_layout tail{0x10000,0x10000,256,0,65012};const auto geometry=plan_guest_surface(0,3,3,4,256);const display_storage_source source{geometry,{{tail.address,0,tail.size}},true,1,tail};auto words=fragment_texture_registers{};words.offset=0x1000+tail.offset;words.format=(1u<<16)|(0xa5u<<8)|(2u<<4)|8|2;words.control0=1u<<31;words.image_rect=(3u<<16)|3;words.control3=256;const auto plan=plan_guest_texture(words);const std::array request{guest_storage_segment{tail.address+tail.offset,0,plan.storage_bytes}};const auto plane=plan_texture_assembly_planes(plan,request).front();const std::array<std::optional<std::uint32_t>,1> origins{words.offset};const auto result=plan_texture_storage_assembly(memory,plane,std::span{&source,1},origins);require(result.bytes.vm.empty());for(const auto& copy:result.bytes.copies)require(result.layouts[copy.input].logical_storage_offset.has_value());
	require(bad_mapping&&refused&&mixed);std::cout<<"Logical banked RTT oracle:"<<profiles<<" source/layout/IO/priority profiles,"<<planes<<" planes/"<<checked<<" byte decisions,"<<refused<<" ambiguous refusals,"<<mixed<<" mixed logical+physical views;independent logical/Morton address andowner decisions,wrong actualIO refusal,last-row padding outside allocation,zero VMpixel access passed\n";
}
void channel_alias_cases()
{
	struct profile { unsigned texture;surface_format source;std::array<std::uint32_t, 4> channels; };
	const std::vector<profile> profiles{
		{0x81,surface_color::b8,{5,0,0,0}}, {0x84,surface_color::rgb565,{3,0,1,2}},
		{0x85,surface_color::argb8,{3,0,1,2}}, {0x8b,surface_color::gb8,{1,0,1,0}},
		{0x9a,surface_color::rgba16f,{3,0,1,2}}, {0x9b,surface_color::rgba32f,{3,0,1,2}}, {0x9c,surface_color::r32f,{0,0,0,0}},
		{0x90,surface_depth::z24s8_uint,{0,0,0,0}}, {0x91,surface_depth::z24s8_float,{0,0,0,0}},
		{0x92,surface_depth::z16_uint,{0,0,0,0}}, {0x93,surface_depth::z16_float,{0,0,0,0}},
		{0x82,surface_color::x1rgb5_zero,{4,0,1,2}}, {0x82,surface_color::x1rgb5_one,{5,0,1,2}},
		{0x9d,surface_color::x1rgb5_zero,{4,0,1,2}}, {0x9d,surface_color::x1rgb5_one,{5,0,1,2}},
		{0x85,surface_color::xrgb8_zero,{4,0,1,2}}, {0x85,surface_color::xrgb8_one,{5,0,1,2}},
		{0x85,surface_color::abgr8,{3,2,1,0}}, {0x85,surface_color::xbgr8_zero,{3,2,1,0}}, {0x85,surface_color::xbgr8_one,{3,2,1,0}},
		{0x9e,surface_color::argb8,{3,0,1,2}}, {0x9e,surface_color::xrgb8_zero,{4,0,1,2}}, {0x9e,surface_color::xrgb8_one,{5,0,1,2}},
		{0x9e,surface_color::abgr8,{5,2,1,0}}, {0x9e,surface_color::xbgr8_zero,{5,2,1,0}}, {0x9e,surface_color::xbgr8_one,{5,2,1,0}},
		{0x98,surface_color::gb8,{1,0,1,0}}};
	std::vector<surface_format> surfaces;
	for (unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) surfaces.push_back(static_cast<surface_color>(c));
	for (unsigned d = 1; d <= 4; ++d) surfaces.push_back(static_cast<surface_depth>(d));
	unsigned count = 0, accepted = 0;
	for (unsigned target : {0x81u,0x82u,0x83u,0x84u,0x85u,0x86u,0x87u,0x88u,0x8bu,0x8du,0x8eu,0x8fu,0x90u,0x91u,0x92u,0x93u,0x94u,0x95u,0x97u,0x98u,0x99u,0x9au,0x9bu,0x9cu,0x9du,0x9eu,0x9fu})
		for (const auto& source : surfaces)
			for (bool packed : {false,true})
			{
				const auto plan = texture(target,packed,4,4,64);
				const auto expected = std::find_if(profiles.begin(),profiles.end(),[&](const auto& p) { return p.texture == target && p.source == source; });
				const auto actual = surface_texture_channels(source,plan);
				require(bool(actual) == (expected != profiles.end()));
				const auto bytes = std::holds_alternative<surface_color>(source) ? color_layout(std::get<surface_color>(source)).pixel_bytes :
					std::get<surface_depth>(source) == surface_depth::z16_uint || std::get<surface_depth>(source) == surface_depth::z16_float ? 2u : 4u;
				const auto geometry = plan_guest_surface(0x1000,4,4,bytes,packed ? 0 : 64,
					packed ? std::optional<swizzle_region>{{{4,4,1},{},{4,4,1}}} : std::nullopt);
				const auto region = plan_surface_texture_region(geometry,source,plan,{0x1000,plan.storage_bytes});
				require(bool(region) == (expected != profiles.end()));
				if (actual) { require(*actual == expected->channels);++accepted; }
				++count;
			}
	require(count == 972 && accepted == 54);
	std::cout << "Native storage-compatible RTT channels:972 profiles,27guest formats x18surface formats xlinear/Morton;27accepted pairs,independent ARGB tables,X constants,RGBA/BGRA casts,GB/HILO remaps and incompatible typed/depth rejection passed\n";
}
void typeless_cases()
{
	identity_memory memory;
	std::vector<surface_format> formats;
	for (unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(c));
	for (unsigned d = 1; d <= 4; ++d) formats.push_back(static_cast<surface_depth>(d));
	unsigned accepted = 0, refused = 0;
	for (const auto& format : formats)
	for (unsigned target : {0x81u,0x82u,0x83u,0x84u,0x85u,0x86u,0x87u,0x88u,0x8bu,0x8du,0x8eu,0x8fu,0x90u,0x91u,0x92u,0x93u,0x94u,0x95u,0x97u,0x98u,0x99u,0x9au,0x9bu,0x9cu,0x9du,0x9eu,0x9fu})
	for (bool packed : {false, true})
	{
		const auto bytes = std::holds_alternative<surface_color>(format) ? color_layout(std::get<surface_color>(format)).pixel_bytes :
			std::get<surface_depth>(format) == surface_depth::z16_uint || std::get<surface_depth>(format) == surface_depth::z16_float ? 2u : 4u;
		const auto codec = describe_guest_texture_format(target);
		const unsigned pitch = 32 * bytes + 16;
		const bool supported = !packed || (codec.block.width == 1 && bytes == codec.block.bytes);
		const auto plan = texture(target, packed, (packed ? 32 : 16) * bytes / codec.block.bytes * codec.block.width, (packed ? 8 : 2) * codec.block.height, pitch);
		const auto source = plan_guest_surface(0, 32, 8, bytes, packed ? 0 : pitch,
			packed ? std::optional<swizzle_region>{{{32,8,1},{},{32,8,1}}} : std::nullopt);
		const unsigned x = packed ? 0 : 8, y = packed ? 0 : 4;
		const auto delta = packed ? morton(x,y,0,{32,8,1}) * bytes : y * pitch + x * bytes;
		const auto region = plan_surface_texture_region(source, format, plan, {static_cast<unsigned>(delta), plan.storage_bytes}, true);
		require(bool(region) == supported);
		if (!supported) { ++refused;continue; }
		require(region->x == x && region->y == y && region->width == (packed ? 32u : 16u) && region->height == (packed ? 8u : 2u));
		// Split inside a source pixel and inside the requested first row.
		const auto split = delta + 3;
		const std::array storage{guest_storage_segment{0x100000,0,split},guest_storage_segment{0x200000,split,source.range.size-split}};
		std::array requested{guest_storage_segment{static_cast<unsigned>(0x100000+delta),0,3},guest_storage_segment{0x200000,3,plan.storage_bytes-3}};
		const auto segmented = plan_segmented_texture_region(memory,source,format,storage,0,plan,requested,delta,true);
		require(segmented && segmented->x == x && segmented->y == y && segmented->width == (packed ? 32u : 16u));
		requested[1].address += 1;
		require(!plan_segmented_texture_region(memory,source,format,storage,0,plan,requested,delta,true));
		++accepted;
	}
	const auto source = plan_guest_surface(0,8,4,4,48);
	require(!plan_surface_texture_region(source,surface_color::argb8,texture(0x81,false,3,2,48),{0,96},true));
	require(!plan_surface_texture_region(source,surface_color::argb8,texture(0x81,false,4,2,48),{1,96},true));
	require(!plan_surface_texture_region(source,surface_color::argb8,texture(0x81,false,4,2,52),{0,104},true));
	require(!plan_surface_texture_region(source,surface_color::argb8,texture(0x81,false,16,2,48),{24,96},true));
	std::cout << "Native typeless RTT address proofs:" << accepted << " accepted and " << refused << " refused profiles,18source/27target formats,linear byte-width scaling,equal-BPP Morton,midpixel segmented identity,wrong active mapping/nonintegral/alignment/pitch/padding rejection passed\n";
}
void partial_byte_cases()
{
	identity_memory memory;
	std::vector<surface_format> formats;
	for (unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(c));
	for (unsigned d=1;d<=4;++d) formats.push_back(static_cast<surface_depth>(d));
	unsigned profiles=0,bytes_checked=0;
	for (const auto& format : formats)
	for (unsigned target : {0x81u,0x82u,0x83u,0x84u,0x85u,0x8bu,0x8fu,0x90u,0x91u,0x92u,0x93u,0x94u,0x95u,0x97u,0x98u,0x99u,0x9au,0x9bu,0x9cu,0x9du,0x9eu,0x9fu})
	{
		const auto source_bytes=std::holds_alternative<surface_color>(format)?color_layout(std::get<surface_color>(format)).pixel_bytes:
			std::get<surface_depth>(format)==surface_depth::z16_uint||std::get<surface_depth>(format)==surface_depth::z16_float?2u:4u;
		for(unsigned prefix=0;prefix<source_bytes;++prefix)
		for(unsigned width : {1u,3u,7u})
		for(unsigned pitch : {288u,289u,291u})
		{
			const auto plan=texture(target,false,width,3,pitch);
			const auto row=width*plan.codec.block.bytes;
			const auto source=plan_guest_surface(0,256/source_bytes,8,source_bytes,pitch);
			const auto byte_x=source_bytes+prefix,delta=2*pitch+byte_x;
			const auto mapped=plan_surface_texture_mapping(source,format,plan,{delta,plan.storage_bytes},true,true);
			require(mapped && mapped->pixels.x==1 && mapped->pixels.y==2 && mapped->pixels.height==3 && mapped->guest_row_offset==prefix);
			const auto expected_width=(prefix+row+source_bytes-1)/source_bytes;
			require(mapped->pixels.width==expected_width);
			if(!prefix && row%source_bytes==0) require(bool(plan_surface_texture_region(source,format,plan,{delta,plan.storage_bytes},true)));
			for(unsigned y=0;y<3;++y)
				for(unsigned b=0;b<row;++b)
				{
					require(delta+y*pitch+b==(mapped->pixels.y+y)*pitch+mapped->pixels.x*source_bytes+mapped->guest_row_offset+b);
					require(mapped->guest_row_offset+b<mapped->pixels.width*source_bytes);++bytes_checked;
				}
			// A disjoint physical mapping splits inside the first requested pixel.
			const auto split=delta+1;
			const std::array storage{guest_storage_segment{0xffffffffu-static_cast<unsigned>(split)+1,0,split},guest_storage_segment{0x90000,split,source.range.size-split}};
			std::array requested{guest_storage_segment{0xffffffffu,0,1},guest_storage_segment{0x90000,1,plan.storage_bytes-1}};
			const auto segmented=plan_segmented_texture_mapping(memory,source,format,storage,0,plan,requested,delta,true,true);
			require(segmented && segmented->guest_row_offset==prefix && segmented->pixels.width==expected_width);
			requested[1].address+=1;
			require(!plan_segmented_texture_mapping(memory,source,format,storage,0,plan,requested,delta,true,true));
			++profiles;
		}
	}
	const auto source=plan_guest_surface(0,4,3,4,21);
	const auto t=texture(0x81,false,3,2,21);
	require(!plan_surface_texture_mapping(source,surface_color::argb8,t,{15,t.storage_bytes},true,true));
	require(!plan_surface_texture_mapping(source,surface_color::argb8,t,{16,t.storage_bytes},true,true));
	require(!plan_surface_texture_mapping(source,surface_color::argb8,t,{42,t.storage_bytes},true,true));
	require(!plan_surface_texture_mapping(source,surface_color::argb8,texture(0x81,false,3,2,22),{1,44},true,true));
	const auto morton_source=plan_guest_surface(0,4,4,4,0,swizzle_region{{4,4,1},{},{4,4,1}});
	const auto morton_t=texture(0x85,true,2,2,0);
	require(!plan_surface_texture_mapping(morton_source,surface_color::argb8,morton_t,{1,morton_t.storage_bytes},true,true));
	const auto one_pixel=plan_guest_surface(0,1,1,4,4);
	const std::array duplicate{guest_storage_segment{0x1000,0,1},guest_storage_segment{0x1000,1,3}};
	const std::array request{guest_storage_segment{0x1000,0,4}};
	const auto byte=texture(0x81,false,1,1,4);
	require(!plan_segmented_texture_mapping(memory,one_pixel,surface_color::argb8,duplicate,0,byte,request,{},true,true));
	for(unsigned anchor : {0u,1u})
	{
		const auto mapped=plan_segmented_texture_mapping(memory,one_pixel,surface_color::argb8,duplicate,0,byte,request,anchor,true,true);
		require(mapped && mapped->pixels.x==0 && mapped->guest_row_offset==anchor);
	}
	std::cout << "Native partial-byte RTT:" << profiles << " profiles," << bytes_checked << " independent byte addresses,18source/22target formats,all prefixes,1/3/7texels,even/odd pitches,topEA disjoint identity,wrong mapping/padding/height/pitch/Morton rejection and same-rectangle different-prefix ambiguity passed\n";
}
void block_target_cases()
{
	identity_memory memory;
	std::vector<surface_format> formats;
	for(unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(c));
	for(unsigned d=1;d<=4;++d) formats.push_back(static_cast<surface_depth>(d));
	unsigned profiles=0,addresses=0;
	for(const auto& format : formats)
	for(unsigned target : {0x86u,0x87u,0x88u,0x8du,0x8eu})
	{
		const unsigned bpp=std::holds_alternative<surface_color>(format)?color_layout(std::get<surface_color>(format)).pixel_bytes:
			std::get<surface_depth>(format)==surface_depth::z16_uint||std::get<surface_depth>(format)==surface_depth::z16_float?2:4;
		// Independent RSX block geometry,including complete partial edge blocks.
		const unsigned bw=target==0x8d||target==0x8e?2:4,bh=bw==2?1:4,bb=bw==2?4:target==0x86?8:16;
		for(unsigned prefix=0;prefix<bpp;++prefix)
		for(unsigned width : {1u,3u,4u,5u,7u,8u})
		for(unsigned height : {1u,3u,4u,5u,7u,8u})
		for(unsigned pitch : {288u,289u})
		{
			const auto plan=texture(target,false,width,height,pitch);
			const unsigned columns=(width+bw-1)/bw,rows=(height+bh-1)/bh,row=columns*bb;
			const auto source=plan_guest_surface(0,256/bpp,12,bpp,pitch);
			const unsigned start=2*pitch+bpp+prefix;
			const auto mapping=plan_surface_texture_mapping(source,format,plan,{start,plan.storage_bytes},true,true);
			require(mapping && mapping->pixels.x==1 && mapping->pixels.y==2 && mapping->pixels.height==rows && mapping->guest_row_offset==prefix);
			require(mapping->pixels.width==(prefix+row+bpp-1)/bpp);
			for(unsigned y=0;y<rows;++y)
				for(unsigned b=0;b<row;++b)
				{
					require(start+y*pitch+b==(mapping->pixels.y+y)*pitch+mapping->pixels.x*bpp+mapping->guest_row_offset+b);++addresses;
				}
			const unsigned split=start+1;
			const std::array storage{guest_storage_segment{0xffffffffu-split+1,0,split},guest_storage_segment{0x90000,split,source.range.size-split}};
			std::array request{guest_storage_segment{0xffffffffu,0,1},guest_storage_segment{0x90000,1,plan.storage_bytes-1}};
			const auto segmented=plan_segmented_texture_mapping(memory,source,format,storage,0,plan,request,start,true,true);
			require(segmented && segmented->pixels.height==rows && segmented->guest_row_offset==prefix);
			request[1].address++;
			require(!plan_segmented_texture_mapping(memory,source,format,storage,0,plan,request,start,true,true));
			// The final compressed/paired block is active even beyond the logical last pixel.
			require(!plan_surface_texture_mapping(source,format,plan,{static_cast<unsigned>(pitch*2+256-row+1),plan.storage_bytes},true,true));
			++profiles;
		}
		const auto plan=texture(target,true,8,8,0);
		const auto morton_source=plan_guest_surface(0,16,16,bpp,0,swizzle_region{{16,16,1},{},{16,16,1}});
		require(!plan_surface_texture_mapping(morton_source,format,plan,{0,plan.storage_bytes},true,true));
	}
	std::cout << "Native block RTT:" << profiles << " profiles," << addresses << " independent byte addresses,18sources/BC1/BC2/BC3/BR_GR/RB_RG,all prefixes,partial edge blocks,odd pitches,topEA split identity,wrong mapping/padding/Morton rejection passed\n";
}
void segmented_cases()
{
	identity_memory memory;
	unsigned count = 0;
	for (const auto& pair : {std::pair{0x81u, surface_format{surface_color::b8}}, {0x84u, surface_color::rgb565}, {0x85u, surface_color::argb8}, {0x8bu, surface_color::gb8}, {0x9au, surface_color::rgba16f}, {0x9bu, surface_color::rgba32f}, {0x9cu, surface_color::r32f}, {0x90u, surface_depth::z24s8_uint}, {0x91u, surface_depth::z24s8_float}, {0x92u, surface_depth::z16_uint}, {0x93u, surface_depth::z16_float}})
		for (bool packed : {false, true})
			for (unsigned split : {1u, 13u, 31u, 63u})
			{
				const auto bytes = describe_guest_texture_format(pair.first).block.bytes;
				const unsigned pitch = 12 * bytes;
				const auto source = plan_guest_surface(0, 8, 8, bytes, packed ? 0 : pitch,
					packed ? std::optional<swizzle_region>{{{8, 8, 1}, {}, {8, 8, 1}}} : std::nullopt);
				const auto divide = std::min<std::size_t>(split, source.range.size - 1);
				const std::array storage{guest_storage_segment{static_cast<std::uint32_t>(0x1'0000'0000ull - divide), 0, divide},
					guest_storage_segment{0x100000, divide, source.range.size - divide}};
				const std::array regions = packed ? std::array<clear_rectangle, 3>{{{0, 0, 2, 2}, {2, 2, 2, 2}, {0, 4, 4, 4}}} :
					std::array<clear_rectangle, 3>{{{1, 0, 3, 2}, {1, 3, 3, 2}, {7, 7, 1, 1}}};
				for (const auto& rectangle : regions)
				{
					const auto plan = texture(pair.first, packed, rectangle.width, rectangle.height, pitch);
					const auto delta = packed ? morton(rectangle.x, rectangle.y, 0, {8, 8, 1}) * bytes : rectangle.y * pitch + rectangle.x * bytes;
					std::vector<guest_storage_segment> requested;
					for (std::size_t position = 0; position < plan.storage_bytes;)
					{
						const auto logical = delta + position;
						const auto in_first = logical < divide;
						const auto size = std::min(plan.storage_bytes - position, in_first ? divide - logical : plan.storage_bytes - position);
						const auto address = in_first ? storage[0].address + logical : storage[1].address + logical - divide;
						requested.push_back({static_cast<std::uint32_t>(address), position, size}); position += size;
					}
					const auto result = plan_segmented_texture_region(memory, source, pair.second, storage, 0, plan, requested);
					require(result && result->x == rectangle.x && result->y == rectangle.y && result->width == rectangle.width && result->height == rectangle.height);
					const auto anchored = plan_segmented_texture_region(memory, source, pair.second, storage, 0, plan, requested, delta);
					require(anchored && anchored->x == rectangle.x && anchored->y == rectangle.y);
					requested[0].address ^= 0x100;
					require(!plan_segmented_texture_region(memory, source, pair.second, storage, 0, plan, requested, delta));
					++count;
				}
			}
	const auto repeated = plan_guest_surface(0, 2, 2, 4, 8);
	const std::array duplicate{guest_storage_segment{0x1000, 0, 8}, guest_storage_segment{0x1000, 8, 8}};
	const std::array request{guest_storage_segment{0x1000, 0, 8}};
	const auto row = texture(0x85, false, 2, 1, 8);
	require(!plan_segmented_texture_region(memory, repeated, surface_color::argb8, duplicate, 0, row, request));
	const auto anchored = plan_segmented_texture_region(memory, repeated, surface_color::argb8, duplicate, 0, row, request, 8);
	require(anchored && anchored->y == 1);
	const auto padded = plan_guest_surface(0, 1, 1, 4, 16);
	const std::array full{guest_storage_segment{0x1000, 0, 16}};
	const std::array padding{guest_storage_segment{0x1000, 0, 4}, guest_storage_segment{0x2000, 4, 12}};
	require(bool(plan_segmented_texture_region(memory, padded, surface_color::argb8, full, 0, texture(0x85, false, 1, 1, 16), padding)));
	std::cout << "Native segmented texture regions:" << count << " eleven-format linear/Morton topEA/midpixel/midrow crop proofs,logical anchors,wrong active mapping rejection,duplicate-physical ambiguity and padding-only divergence passed without VM pixel access\n";
}
} // namespace
int main(int argc, char** argv)
{
	if (argc == 2 && std::string_view(argv[1]) == "packed-morton")
	{
		try { morton_paired_projection_cases(); return 0; }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
	try
	{
		morton_display_cases(); display_assembly_cases(); display_physical_assembly_cases(); texture_assembly_cases(); texture_inverse_cases(); vtc_projection_cases();logical_texture_assembly_cases(); channel_alias_cases();
		stored_border_projection_cases();
		morton_border_projection_cases();
		stored_border_mip_projection_cases();
	stored_border_mip_projection_cases(true);
	paired_border_projection_cases();
	morton_paired_projection_cases();
	volume_border_projection_cases();
	cube_border_projection_cases();
		segmented_cases();
		typeless_cases();
		partial_byte_cases();
		block_target_cases();
		std::size_t count = 0;
		for (const auto& pair : {std::pair{0x81u, surface_format{surface_color::b8}}, {0x84u, surface_color::rgb565}, {0x85u, surface_color::argb8}, {0x8bu, surface_color::gb8}, {0x9au, surface_color::rgba16f}, {0x9bu, surface_color::rgba32f}, {0x9cu, surface_color::r32f}, {0x90u, surface_depth::z24s8_uint}, {0x91u, surface_depth::z24s8_float}, {0x92u, surface_depth::z16_uint}, {0x93u, surface_depth::z16_float}})
		{
			const auto bytes = describe_guest_texture_format(pair.first).block.bytes;
			for (unsigned width : {1u, 2u, 4u, 7u, 8u})
				for (unsigned height : {1u, 3u, 8u})
					for (unsigned padding : {0u, 1u, 4u})
					{
						const unsigned pitch = (width + padding) * bytes;
						const auto surface = plan_guest_surface(0x1000, width, height, bytes, pitch);
						for (unsigned y = 0; y <= height; ++y)
							for (unsigned x = 0; x <= width + padding; ++x)
								for (unsigned w = 1; w <= width + 1; ++w)
									for (unsigned h : {1u, 2u, height})
									{
										const auto plan = texture(pair.first, false, w, h, pitch);
										const auto address = 0x1000 + y * pitch + x * bytes;
										const auto result = plan_surface_texture_region(surface, pair.second, plan, {static_cast<std::uint32_t>(address), plan.storage_bytes});
										const unsigned actual_y = y + x / (width + padding), actual_x = x % (width + padding);
										const auto effective_width = std::min<unsigned>(w, pitch / bytes);
										const bool valid = actual_x + effective_width <= width && actual_y + h <= height;
										if (bool(result) != valid)
											throw std::runtime_error("Linear format=" + std::to_string(pair.first) + " source=" + std::to_string(width) + "x" + std::to_string(height) + " pitch=" + std::to_string(pitch) + " start=" + std::to_string(x) + "," + std::to_string(y) + " region=" + std::to_string(w) + "x" + std::to_string(h) + " result=" + std::to_string(bool(result)) + " expected=" + std::to_string(valid));
										if (result)
											require(result->x == actual_x && result->y == actual_y && result->width == effective_width && result->height == h);
										++count;
									}
						const auto plan = texture(pair.first, false, 1, 1, pitch);
						if (bytes > 1)
							require(!plan_surface_texture_region(surface, pair.second, plan, {0x1001, plan.storage_bytes}));
						require(!plan_surface_texture_region(surface, pair.second, plan, {0xfff, plan.storage_bytes}));
					}
			for (const auto storage : {std::array<std::size_t, 3>{2, 2, 1}, {4, 4, 1}, {8, 4, 1}, {4, 8, 1}, {8, 8, 1}, {4, 4, 2}})
				for (bool inner : {false, true})
				{
					const std::array<std::size_t, 3> origin = inner ? std::array<std::size_t, 3>{1, 1, storage[2] - 1} : std::array<std::size_t, 3>{0, 0, storage[2] - 1};
					const auto width = storage[0] - origin[0], height = storage[1] - origin[1];
					const auto surface = plan_guest_surface(0x1000, width, height, bytes, 0, swizzle_region{storage, origin, {width, height, 1}});
					for (std::size_t sy = 0; sy < storage[1]; ++sy)
						for (std::size_t sx = 0; sx < storage[0]; ++sx)
							for (unsigned w : {1u, 2u, 4u, 8u})
								for (unsigned h : {1u, 2u, 4u, 8u})
								{
									const auto plan = texture(pair.first, true, w, h, 0);
									const auto base = morton(sx, sy, origin[2], storage);
									bool valid = sx >= origin[0] && sy >= origin[1] && sx + w <= storage[0] && sy + h <= storage[1];
									// Compare every pixel's physical byte address; no shared
									// prefix/alignment planner is used by this oracle.
									for (unsigned y = 0; valid && y < h; ++y)
										for (unsigned x = 0; valid && x < w; ++x)
											valid = morton(sx + x, sy + y, origin[2], storage) == base + morton(x, y, 0, {w, h, 1});
									const auto result = plan_surface_texture_region(surface, pair.second, plan, {static_cast<std::uint32_t>(0x1000 + base * bytes), plan.storage_bytes});
									if (bool(result) != valid)
										throw std::runtime_error("Morton format=" + std::to_string(pair.first) + " storage=" + std::to_string(storage[0]) + "x" + std::to_string(storage[1]) + "x" + std::to_string(storage[2]) + " inner=" + std::to_string(inner) + " start=" + std::to_string(sx) + "," + std::to_string(sy) + " region=" + std::to_string(w) + "x" + std::to_string(h) + " result=" + std::to_string(bool(result)) + " expected=" + std::to_string(valid));
									if (result)
										require(result->x == sx - origin[0] && result->y == sy - origin[1] && result->width == w && result->height == h);
									++count;
								}
				}
		}
		std::cout << count << " linear/Morton native texture-region mappings matched independent per-pixel guest-address oracle across eleven formats\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
