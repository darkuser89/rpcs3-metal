#include "M4RSXScaledImage.h"
#include "M4RectangleCoverage.h"
#include "M4ScaledImageNativePlan.h"
#include <iostream>
#include <limits>

namespace
{
void require(bool v)
{
	if (!v)
		throw std::runtime_error("Native scaled-image decoded plan differs");
}
template <typename F>
void refused(F&& f)
{
	bool fail = false;
	try
	{
		f();
	}
	catch (const std::exception&)
	{
		fail = true;
	}
	require(fail);
}
void rectangle_coverage_cases()
{
    using metal4::clear_rectangle;using metal4::rectangles_cover;
    require(rectangles_cover(std::array{clear_rectangle{0,0,2,4},clear_rectangle{2,0,2,4}},4,4));
    require(!rectangles_cover(std::array{clear_rectangle{0,0,2,4},clear_rectangle{0,0,2,4}},4,4));
    require(!rectangles_cover(std::array{clear_rectangle{0,0,4,1},clear_rectangle{0,2,4,2}},4,4));
    const auto maximum=std::numeric_limits<std::size_t>::max();
    require(rectangles_cover(std::array{clear_rectangle{0,0,maximum/2,maximum},clear_rectangle{maximum/2,0,maximum-maximum/2,maximum}},maximum,maximum));
    refused([&]{(void)rectangles_cover(std::array{clear_rectangle{maximum,0,1,1}},maximum,1);});
    refused([&]{(void)rectangles_cover({},0,1);});
    require(!rectangles_cover(std::vector<clear_rectangle>(metal4::rectangle_coverage_limit+1),1,1));
    std::uint32_t state=0x3a71c4d2;const auto next=[&]{state=state*1664525u+1013904223u;return state;};
    for(unsigned profile=0;profile<20000;++profile)
    {
        const unsigned width=1+next()%8,height=1+next()%8,count=next()%25;
        std::vector<clear_rectangle> rectangles;std::array<bool,64> pixels{};
        for(unsigned i=0;i<count;++i){const auto x=next()%(width+1),y=next()%(height+1);rectangles.push_back({x,y,next()%(width-x+1),next()%(height-y+1)});}
        if(profile%5==0){rectangles.push_back({0,0,width/2,height});rectangles.push_back({width/2,0,width-width/2,height});}
        for(const auto& r:rectangles)for(unsigned y=r.y;y<r.y+r.height;++y)for(unsigned x=r.x;x<r.x+r.width;++x)pixels[y*width+x]=true;
        const bool expected=std::all_of(pixels.begin(),pixels.begin()+width*height,[](bool value){return value;});
        require(rectangles_cover(rectangles,width,height)==expected);std::reverse(rectangles.begin(),rectangles.end());if(!rectangles.empty())rectangles.push_back(rectangles.front());require(rectangles_cover(rectangles,width,height)==expected);
    }
    std::cout<<"Exact rectangle union:20000 independent grid-oracle profiles,overlaps/duplicates/order/holes,area-sum false positives,size_t maximum edges/overflow refusal andbounded metadata fallback passed\n";
}
std::size_t region_morton(std::size_t x,std::size_t y)
{
    std::size_t result=0,shift=0;
    for(std::size_t bit=1;bit<16;bit*=2){if(x&bit)result|=std::size_t(1)<<shift;++shift;if(bit<8){if(y&bit)result|=std::size_t(1)<<shift;++shift;}}
    return result;
}
void bounded_input_region_cases()
{
    using namespace metal4;std::size_t profiles=0,checked=0;
    for(unsigned bpp:{1u,2u,4u,8u,16u})for(unsigned layout=0;layout<4;++layout)
    for(unsigned words:{1u,3u,7u,19u,128u})for(unsigned seed=0;seed<32;++seed)
    {
        const bool swizzled=layout&1,banked=layout&2;
        const auto morton=swizzled?std::optional<swizzle_region>{{{16,8,1},{3,2,0},{7,5,1}}}:std::nullopt;
        const auto geometry=plan_guest_surface(0,7,5,bpp,swizzled?0:banked?256:7*bpp+3,morton);
        const auto tile=banked?std::optional<banked_tile_layout>{{0x40000,0x40000,256,seed%4,0x1200+bpp}}:std::nullopt;
        const display_storage_source source{geometry,{},true,1,tile};
        std::vector<display_storage_copy> copies;std::vector<std::array<std::size_t,3>> expected;
        std::array<bool,35> selected{},packed{};
        for(unsigned y=0;y<5;++y)for(unsigned x=0;x<7;++x)for(unsigned lane=0;lane<bpp;++lane)
        {
            if((x*7+y*3+lane+seed)%5>seed%4)continue;
            const auto logical=swizzled?region_morton(x+3,y+2)*bpp+lane:y*geometry.row_pitch+x*bpp+lane;
            const auto physical=tile?banked_display_storage_offset(*tile,tile->offset+logical):logical;
            copies.push_back({expected.size(),physical,1,3});expected.push_back({x,y,lane});selected[y*7+x]=true;
            if(seed%3==0){copies.push_back({expected.size(),physical,1,3});expected.push_back({x,y,lane});}
        }
        copies.push_back({0,std::numeric_limits<std::size_t>::max(),1,9});std::reverse(copies.begin(),copies.end());
        const auto result=plan_scaled_image_input_regions(source,copies,3,words*bpp);
        std::vector<bool> observed(expected.size());
        for(const auto& part:result)
        {
            const auto& r=part.pixels;require(r.width&&r.height&&r.x+r.width<=7&&r.y+r.height<=5);
            require(part.bytes==r.width*r.height*bpp&&part.bytes<=words*bpp);
            for(std::size_t y=r.y;y<r.y+r.height;++y)for(std::size_t x=r.x;x<r.x+r.width;++x)
            {require(selected[y*7+x]&&!packed[y*7+x]);packed[y*7+x]=true;}
            for(const auto& copy:part.copies)
            {
                require(copy.input==0&&copy.size&&copy.source+copy.size<=part.bytes&&copy.target+copy.size<=expected.size());
                for(std::size_t b=0;b<copy.size;++b)
                {
                    const auto at=copy.source+b,pixel=at/bpp,index=copy.target+b;
                    require(!observed[index]&&expected[index]==std::array<std::size_t,3>{r.x+pixel%r.width,r.y+pixel/r.width,at%bpp});
                    observed[index]=true;++checked;
                }
            }
        }
        require(packed==selected&&std::all_of(observed.begin(),observed.end(),[](bool v){return v;}));++profiles;
    }
    const display_storage_source linear{plan_guest_surface(0,2,2,4,13),{},false,0};
    const std::array valid{display_storage_copy{0,0,4,0}};
    refused([&]{(void)plan_scaled_image_input_regions(linear,valid,0,3);});
    refused([&]{(void)plan_scaled_image_input_regions(linear,valid,1,16);});
    for(const auto bad:std::array{display_storage_copy{0,8,1,0},display_storage_copy{0,26,1,0},display_storage_copy{0,0,0,0},display_storage_copy{0,std::numeric_limits<std::size_t>::max(),1,0},display_storage_copy{std::numeric_limits<std::size_t>::max(),0,1,0}})
        refused([&]{(void)plan_scaled_image_input_regions(linear,std::span{&bad,1},0,16);});
    std::cout<<"Bounded native input regions:"<<profiles<<" sparse/duplicate/partial-lane profiles,"<<checked<<" independent native-coordinate byte decisions;1-16byte words,odd pitches,Morton/banked layouts,one-word quotas,exact required texels/no padding andinvalid/overflow refusal passed\n";
}
} // namespace
class output_metadata_memory final : public metal4::guest_surface_memory
{
    std::shared_ptr<int> allocation=std::make_shared<int>(0);
  public:
    unsigned pixel_calls=0;bool identities=true;
    std::optional<std::vector<metal4::storage_identity_span>> storage_identity(metal4::guest_surface_range r) override
    {if(!identities)return {};return std::vector<metal4::storage_identity_span>{{allocation,r.address%0x1000000,r}};}
    bool test_storage(const storage_test& test) override
    {return test(false,[](auto a,auto b){return metal4::overlaps(metal4::guest_surface_range{a.address%0x1000000,a.size},metal4::guest_surface_range{b.address%0x1000000,b.size});});}
    metal4::guest_surface_snapshot read_surface(const metal4::guest_surface_layout&) override {++pixel_calls;throw std::runtime_error("Output metadata planner read pixels");}
    std::uint64_t surface_revision(const metal4::guest_surface_layout&) override {++pixel_calls;throw std::runtime_error("Output metadata planner queried pixel revisions");}
    std::optional<std::uint64_t> compare_write(const metal4::guest_surface_layout&,std::uint64_t,std::span<const std::byte>) override {++pixel_calls;throw std::runtime_error("Output metadata planner wrote pixels");}
    void write(metal4::guest_surface_range,std::span<const std::byte>) override {++pixel_calls;throw std::runtime_error("Output metadata planner wrote pixels");}
};
// Keep the historical five-writer matrices as explicit smaller-budget cases.
static auto plan_five_output_regions(metal4::guest_surface_memory& memory,
    std::span<const metal4::display_storage_source> targets, const metal4::display_storage_source& requested,
    std::size_t runs=1u<<20, std::size_t windows=4096,
    std::vector<metal4::display_storage_source>* cold=nullptr, std::size_t dimension=0)
{
    return metal4::plan_scaled_image_output_regions(memory, targets, requested, runs, windows, cold, dimension, 5);
}
void partial_output_region_cases(bool scattered=false,bool holes=false)
{
    using namespace metal4;output_metadata_memory memory;unsigned profiles=0,guards=0,reorders=0;std::size_t selected_bytes=0;
    const auto same=[](const clear_rectangle& a,const clear_rectangle& b){return a.x==b.x&&a.y==b.y&&a.width==b.width&&a.height==b.height;};
    const auto address=[](const guest_surface_layout& g,unsigned x,unsigned y,unsigned lane)
    {
        if(!g.morton)return y*g.row_pitch+x*g.pixel_bytes+lane;
        x+=g.morton->origin[0];y+=g.morton->origin[1];std::size_t word=0,shift=0;
        for(unsigned bit=1;bit<32;bit*=2)for(unsigned axis=0;axis<2;++axis)
            if(bit<g.morton->storage[axis]){word|=std::size_t(bool((axis?y:x)&bit))<<shift;++shift;}
        return word*g.pixel_bytes+lane;
    };
    for(unsigned bpp:{2u,4u})for(unsigned count:{1u,2u,3u,4u,5u})for(unsigned grid=0;grid<3;++grid)
    for(unsigned layout=0;layout<4;++layout)for(unsigned seed=0;seed<4;++seed)
    {
        if((!scattered&&count==1)||(holes&&count==5))continue;
        const bool morton=layout&1,banked=layout&2;unsigned width=grid==1?4:count*2,height=grid==1?count*2:4;
        std::vector<clear_rectangle> windows;
        if(grid==2)
        {
            width=8;height=8;windows.push_back(count==1?clear_rectangle{0,0,8,8}:clear_rectangle{0,0,8,4});
            for(unsigned i=0;i<count-1;++i){const auto left=i*8/(count-1),right=(i+1)*8/(count-1);windows.push_back({left,4,right-left,4});}
        }
        else for(unsigned i=0;i<count;++i)windows.push_back(grid?clear_rectangle{0,i*2,4,2}:clear_rectangle{i*2,0,2,4});
        const auto sw=morton?std::optional<swizzle_region>{{{32,32,1},{1,2,0},{width,height,1}}}:std::nullopt;
        const auto tile=banked?std::optional<banked_tile_layout>{{0x80000,0x40000,256,seed,0x1400+bpp}}:std::nullopt;
        const auto geometry=plan_guest_surface(0,width,height,bpp,morton?0:banked?256:width*bpp+3,sw);
        const display_storage_source requested{geometry,{{tile?tile->address:0x6000u,0,tile?tile->size:geometry.range.size}},false,0,tile};
        std::vector<display_storage_source> targets;std::vector<clear_rectangle> native;
        std::size_t hole_words=0;
        for(unsigned i=0;i<count;++i)
        {
            const auto& r=windows[i];const unsigned nx=1+(seed&1),ny=1+(seed>>1),nw=r.width+(scattered?4:3),nh=r.height+(scattered?4:3);
            const auto ns=morton?std::optional<swizzle_region>{{{32,32,1},{2,1,0},{nw,nh,1}}}:std::nullopt;
            const auto g=plan_guest_surface(0,nw,nh,bpp,morton?0:nw*bpp+1,ns);
            std::vector<unsigned> locations(g.range.size);for(unsigned at=0;at<locations.size();++at)locations[at]=0x800000+i*0x10000+at;
            std::vector<std::array<unsigned,2>> cells;
            for(unsigned y=0;y<nh;++y)for(unsigned x=0;x<nw;++x)if((x+y)%2==seed%2)cells.push_back({x,y});
            if(seed&2)std::reverse(cells.begin(),cells.end());
            require(!scattered||cells.size()>=r.width*r.height);
            for(unsigned y=0;y<r.height;++y)for(unsigned x=0;x<r.width;++x)for(unsigned lane=0;lane<bpp;++lane)
            {
                if(holes&&(x+y+i+seed)%3==0){if(!lane)++hole_words;continue;}
                auto at=address(geometry,r.x+x,r.y+y,lane);if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                const auto cell=scattered?cells[y*r.width+x]:std::array<unsigned,2>{nx+x,ny+y};
                locations[address(g,cell[0],cell[1],lane)]=(tile?tile->address:0x6000u)+at+(i%2?0x1000000:0);
            }
            std::vector<guest_storage_segment> spans;for(unsigned at=0;at<locations.size();++at)spans.push_back({locations[at],at,1});
            targets.push_back({g,std::move(spans),false,0,{}});native.push_back({nx,ny,r.width,r.height});
        }
        const auto prove=[&](const auto& sources,const auto& regions)
        {
            std::vector<bool> packed_pixels(width*height);std::vector<std::vector<bool>> native_pixels;
            for(const auto& source:sources)native_pixels.emplace_back(source.geometry.width*source.geometry.height);
            for(const auto& region:regions)
            {
                require(region.output<sources.size());const auto& source=sources[region.output];const auto& a=region.packed;const auto& n=region.native;
                require(a.width==n.width&&a.height==n.height&&a.x+a.width<=width&&a.y+a.height<=height&&n.x+n.width<=source.geometry.width&&n.y+n.height<=source.geometry.height);
                for(unsigned y=0;y<a.height;++y)for(unsigned x=0;x<a.width;++x)
                {
                    const auto packed_pixel=(a.y+y)*width+a.x+x,native_pixel=(n.y+y)*source.geometry.width+n.x+x;
                    require(!packed_pixels[packed_pixel]&&!native_pixels[region.output][native_pixel]);packed_pixels[packed_pixel]=true;native_pixels[region.output][native_pixel]=true;
                    for(unsigned lane=0;lane<bpp;++lane)
                    {
                        auto at=address(geometry,a.x+x,a.y+y,lane);if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                        const auto location=address(source.geometry,n.x+x,n.y+y,lane);
                        const auto span=std::find_if(source.segments.begin(),source.segments.end(),[&](const auto& p){return location>=p.offset&&location-p.offset<p.size;});require(span!=source.segments.end());
                        const auto actual=(span->address+location-span->offset)%0x1000000;
                        require(actual==(tile?tile->address:0x6000u)+at);
                    }
                }
            }
            require(std::all_of(packed_pixels.begin(),packed_pixels.end(),[](bool value){return value;}));
            if(holes)for(unsigned i=count;i<native_pixels.size();++i)require(std::all_of(native_pixels[i].begin(),native_pixels[i].end(),[](bool value){return value;}));
        };
        std::vector<display_storage_source> cold;
        const auto native_limit=holes?(hole_words+(5-count)-1)/(5-count):width*height;
        const auto planned=plan_five_output_regions(memory,targets,requested,1u<<20,4096,holes?&cold:nullptr,native_limit);require(planned.has_value());
        auto all=targets;all.insert(all.end(),cold.begin(),cold.end());prove(all,*planned);
        if(holes)
        {
            require(!cold.empty()&&cold.size()<=5-count);std::size_t words=0;
            for(const auto& part:cold){require(part.geometry.height==1&&part.geometry.width<=native_limit);words+=part.geometry.width;}require(words==hole_words);
            require(!plan_five_output_regions(memory,targets,requested));++guards;
            std::vector<display_storage_source> refused;
            require(!plan_five_output_regions(memory,targets,requested,1u<<20,4096,&refused,0)&&refused.empty());++guards;
            auto partial=targets;
            const auto origin=tile?tile->address:0x6000u;const auto extent=tile?tile->size:geometry.range.size;
            const auto byte=std::find_if(partial[0].segments.begin(),partial[0].segments.end(),[&](const auto& p){const auto a=p.address%0x1000000;return a>=origin&&a-origin<extent;});require(byte!=partial[0].segments.end());
            byte->address=0xf00000;refused=cold;
            require(!plan_five_output_regions(memory,partial,requested,1u<<20,4096,&refused,native_limit)&&refused.empty());++guards;
            require(!plan_five_output_regions(memory,targets,requested,1u<<20,0,&refused,native_limit)&&refused.empty());++guards;
            if(hole_words>5-count){require(!plan_five_output_regions(memory,targets,requested,1u<<20,4096,&refused,1)&&refused.empty());++guards;}
        }
        if(!scattered&&!holes)
        {
            require(planned->size()==count);
            for(unsigned i=0;i<count;++i)require((*planned)[i].output==i&&same((*planned)[i].packed,windows[i])&&same((*planned)[i].native,native[i]));
        }
        for(const auto& r:windows)selected_bytes+=r.width*r.height*bpp;
        require(!plan_five_output_regions(memory,targets,requested,1u<<20,0));++guards;
        require(!plan_five_output_regions(memory,targets,requested,1u<<20,planned->size()-1));++guards;
        require(!plan_five_output_regions(memory,targets,requested,0));require(!plan_five_output_regions(memory,targets,requested,1));guards+=2;
        if(layout==0&&seed==0)
        {
            auto duplicate=requested;duplicate.segments.clear();
            for(unsigned at=0;at<geometry.range.size;++at)duplicate.segments.push_back({0x6000u+(at>=bpp&&at<2*bpp?at-bpp:at),at,1});
            require(!plan_five_output_regions(memory,targets,duplicate));++guards;
        }
        // Corrupt each byte-level proof independently: a hole, a lane swap,
        // and two words exchanged while coverage remains complete.
        if(seed==0&&!scattered&&!holes)
        {
            memory.identities=false;require(!plan_five_output_regions(memory,targets,requested));memory.identities=true;++guards;
            auto bad=targets;const auto first=address(targets[0].geometry,native[0].x,native[0].y,0),second=address(targets[0].geometry,native[0].x+1,native[0].y,0);
            bad[0].segments[first].address=0xf00000;require(!plan_five_output_regions(memory,bad,requested));++guards;
            bad=targets;std::swap(bad[0].segments[first].address,bad[0].segments[first+1].address);require(!plan_five_output_regions(memory,bad,requested));++guards;
            bad=targets;for(unsigned lane=0;lane<bpp;++lane)std::swap(bad[0].segments[first+lane].address,bad[0].segments[second+lane].address);
            const auto reordered=plan_five_output_regions(memory,bad,requested);require(reordered.has_value());prove(bad,*reordered);++reorders;
        }
        ++profiles;
    }
    require(memory.pixel_calls==0);
    std::cout<<(holes?"Cold native output holes: ":scattered?"Disconnected native output regions: ":"Partial native output regions: ")<<profiles<<" independent linear/Morton/banked and split-mirror layouts, "<<selected_bytes<<" exact selected byte coordinates, "<<guards<<" identity/hole/lane/duplicate/run/window-budget refusals, "<<reorders<<" exact native word permutations; vertical/horizontal/T partitions, nonzero native XY and untouched exterior proved without pixel access\n";
}

void cold_only_output_region_cases()
{
    using namespace metal4;output_metadata_memory memory;unsigned profiles=0,refusals=0;std::size_t checked=0;
    for(unsigned bpp:{2u,4u})for(unsigned layout=0;layout<4;++layout)for(unsigned split:{0u,1u})
    for(const auto shape:std::array{std::array{7u,5u},std::array{9u,4u},std::array{2u,19u}})
    {
        const auto width=shape[0],height=shape[1];const bool morton=layout&1,banked=layout&2;
        const auto sw=morton?std::optional<swizzle_region>{{{32,32,1},{1,2,0},{width,height,1}}}:std::nullopt;
        const auto g=plan_guest_surface(0,width,height,bpp,morton?0:banked?256:width*bpp+3,sw);
        const auto tile=banked?std::optional<banked_tile_layout>{{0x80000,0x40000,256,split,0x1400+bpp}}:std::nullopt;
        const auto size=tile?tile->size:g.range.size;const unsigned base=tile?tile->address:0x6000;
        std::vector<guest_storage_segment> spans;
        // Odd cuts deliberately split a word between mirrored addresses.
        if(split){spans={{base,0,3},{base+3+0x1000000,3,size-8},{static_cast<unsigned>(base+size-5),size-5,5}};}
        else spans={{base,0,size}};
        const display_storage_source requested{g,spans,false,0,tile};
        std::vector<unsigned> expected;
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned lane=0;lane<bpp;++lane)
        {
            std::size_t raw=y*g.row_pitch+x*bpp+lane;
            if(morton)
            {
                std::size_t word=0,shift=0;
                for(unsigned bit=1;bit<32;bit*=2)for(unsigned axis=0;axis<2;++axis)
                {if((axis?y+2:x+1)&bit)word|=std::size_t(1)<<shift;++shift;}
                raw=word*bpp+lane;
            }
            if(tile)raw=banked_display_storage_offset(*tile,tile->offset+raw);
            expected.push_back(base+raw);
        }
        for(unsigned slots=1;slots<=5;++slots)
        {
            const auto native_limit=(width*height+slots-1)/slots;
            std::vector<display_storage_source> cold;
            const auto plan=plan_five_output_regions(memory,{},requested,1u<<20,4096,&cold,native_limit);
            require(plan.has_value()&&!cold.empty()&&cold.size()<=5);
            std::vector<bool> packed(width*height);std::vector<std::vector<bool>> used;
            for(const auto& row:cold){require(row.geometry.height<=native_limit&&row.geometry.width<=native_limit);used.emplace_back(row.geometry.width*row.geometry.height);}
            for(const auto& part:*plan)
            {
                require(part.output<cold.size());const auto& row=cold[part.output];const auto& a=part.packed;const auto& n=part.native;
                require(a.width==n.width&&a.height==n.height&&a.x+a.width<=width&&a.y+a.height<=height&&n.y+n.height<=row.geometry.height&&n.x+n.width<=row.geometry.width);
                for(unsigned y=0;y<a.height;++y)for(unsigned x=0;x<a.width;++x)
                {
                    const auto word=(a.y+y)*width+a.x+x,native=(n.y+y)*row.geometry.width+n.x+x;require(!packed[word]&&!used[part.output][native]);packed[word]=true;used[part.output][native]=true;
                    for(unsigned lane=0;lane<bpp;++lane)
                    {
                        std::size_t at=(n.y+y)*row.geometry.row_pitch+(n.x+x)*bpp+lane;
                        if(row.geometry.morton)
                        {
                            const auto& sw=*row.geometry.morton;std::size_t index=0,shift=0;
                            for(unsigned bit=1;bit<32;bit*=2)for(unsigned axis=0;axis<2;++axis){if(((axis?n.y+y:n.x+x)+sw.origin[axis])&bit)index|=std::size_t(1)<<shift;++shift;}
                            at=index*bpp+lane;
                        }
                        const auto* span=texture_assembly_segment(row.segments,at);require(span);
                        require((span->address+at-span->offset)%0x1000000==expected[word*bpp+lane]);++checked;
                    }
                }
            }
            require(std::all_of(packed.begin(),packed.end(),[](bool v){return v;}));
            for(const auto& row:used)require(std::all_of(row.begin(),row.end(),[](bool v){return v;}));
            ++profiles;
        }
        std::vector<display_storage_source> cold{requested};
        const auto reject=[&](std::size_t runs,std::size_t windows,std::size_t limit)
        {cold={requested};require(!plan_five_output_regions(memory,{},requested,runs,windows,&cold,limit)&&cold.empty());++refusals;};
        reject(1u<<20,4096,0);reject(0,4096,100);reject(1u<<20,0,100);reject(1u<<20,4096,1);
        require(!plan_five_output_regions(memory,{},requested));++refusals;
        memory.identities=false;reject(1u<<20,4096,100);memory.identities=true;
    }
    for(unsigned bpp:{2u,4u})
    {
        const display_storage_source duplicate{plan_guest_surface(0,2,1,bpp),{{0x6000,0,bpp},{0x6000,bpp,bpp}},false,0};
        std::vector<display_storage_source> cold;
        bool rejected=false;try{rejected=!plan_five_output_regions(memory,{},duplicate,1u<<20,4096,&cold,2);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected&&cold.empty());++refusals;
    }
    require(memory.pixel_calls==0);
    std::cout<<"Entirely cold output metadata: "<<profiles<<" linear/Morton/banked/split-mirror plans, "<<checked<<" exact byte coordinates, "<<refusals<<" width/slot/run/window/identity/duplicate refusals; one to five fully written native targets, no previous target or pixel access passed\n";
}

void atlas_output_region_cases()
{
    using namespace metal4;output_metadata_memory memory;unsigned profiles=0,refusals=0;std::size_t checked=0;
    for(unsigned bpp:{2u,4u})for(unsigned limit:{7u,13u})for(unsigned layout=0;layout<4;++layout)
    for(unsigned split:{0u,1u})for(unsigned owners=0;owners<=4;++owners)
    {
        const auto width=limit+1,height=limit-1;const bool morton=layout&1,banked=layout&2;
        const auto sw=morton?std::optional<swizzle_region>{{{32,32,1},{1,2,0},{width,height,1}}}:std::nullopt;
        const auto g=plan_guest_surface(0,width,height,bpp,morton?0:banked?256:width*bpp+3,sw);
        const auto tile=banked?std::optional<banked_tile_layout>{{0x80000,0x40000,256,split,0x1400+bpp}}:std::nullopt;
        const unsigned base=tile?tile->address:0x6000;const auto size=tile?tile->size:g.range.size;
        const std::vector<guest_storage_segment> spans=split?std::vector<guest_storage_segment>{{base,0,3},{base+3+0x1000000,3,size-8},{static_cast<unsigned>(base+size-5),size-5,5}}:std::vector<guest_storage_segment>{{base,0,size}};
        const display_storage_source requested{g,spans,false,0,tile};std::vector<unsigned> expected;
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned lane=0;lane<bpp;++lane)
        {
            std::size_t raw=y*g.row_pitch+x*bpp+lane;
            if(morton){std::size_t word=0,shift=0;for(unsigned bit=1;bit<32;bit*=2)for(unsigned axis=0;axis<2;++axis){if((axis?y+2:x+1)&bit)word|=std::size_t(1)<<shift;++shift;}raw=word*bpp+lane;}
            if(tile)raw=banked_display_storage_offset(*tile,tile->offset+raw);expected.push_back(base+raw);
        }
        std::vector<display_storage_source> targets;
        for(unsigned i=0;i<owners;++i)
        {std::vector<guest_storage_segment> parts;for(unsigned lane=0;lane<bpp;++lane)parts.push_back({expected[((i+1)*width+i)*bpp+lane]+(i%2?0x1000000:0),lane,1});targets.push_back({plan_guest_surface(0,1,1,bpp),parts,false,0});}
        std::vector<display_storage_source> cold{requested};
        const auto plan=plan_five_output_regions(memory,targets,requested,1u<<20,4096,&cold,limit);
        if(owners==4){require(!plan&&cold.empty());++refusals;continue;}
        require(plan.has_value()&&!cold.empty()&&targets.size()+cold.size()<=5);
        auto all=targets;all.insert(all.end(),cold.begin(),cold.end());std::vector<bool> packed(width*height);std::vector<std::vector<bool>> used;
        for(const auto& t:all){require(t.geometry.width<=limit&&t.geometry.height<=limit);used.emplace_back(t.geometry.width*t.geometry.height);}
        bool multirow=false;for(const auto& t:cold)multirow|=t.geometry.height>1;require(multirow);
        for(const auto& part:*plan)
        {
            require(part.output<all.size());const auto& t=all[part.output];const auto& a=part.packed;const auto& n=part.native;
            require(a.width==n.width&&a.height==n.height&&a.x+a.width<=width&&a.y+a.height<=height&&n.x+n.width<=t.geometry.width&&n.y+n.height<=t.geometry.height);
            for(unsigned y=0;y<a.height;++y)for(unsigned x=0;x<a.width;++x)
            {
                const auto word=(a.y+y)*width+a.x+x,native=(n.y+y)*t.geometry.width+n.x+x;require(!packed[word]&&!used[part.output][native]);packed[word]=true;used[part.output][native]=true;
                for(unsigned lane=0;lane<bpp;++lane)
                {
                    std::size_t at=(n.y+y)*t.geometry.row_pitch+(n.x+x)*bpp+lane;
                    if(t.geometry.morton){const auto& sw=*t.geometry.morton;std::size_t index=0,shift=0;for(unsigned bit=1;bit<32;bit*=2)for(unsigned axis=0;axis<2;++axis){if(((axis?n.y+y:n.x+x)+sw.origin[axis])&bit)index|=std::size_t(1)<<shift;++shift;}at=index*bpp+lane;}
                    const auto* span=texture_assembly_segment(t.segments,at);require(span);require((span->address+at-span->offset)%0x1000000==expected[word*bpp+lane]);++checked;
                }
            }
        }
        require(std::all_of(packed.begin(),packed.end(),[](bool v){return v;}));for(const auto& row:used)require(std::all_of(row.begin(),row.end(),[](bool v){return v;}));
        const auto reject=[&](std::size_t runs,std::size_t windows,std::size_t dim)
        {cold={requested};require(!plan_five_output_regions(memory,targets,requested,runs,windows,&cold,dim)&&cold.empty());++refusals;};
        reject(0,4096,limit);reject(1u<<20,0,limit);reject(1u<<20,plan->size()-1,limit);reject(1u<<20,4096,1);
        memory.identities=false;reject(1u<<20,4096,limit);memory.identities=true;++profiles;
    }
    // Exact-square, multiple-full-tile and short-tail boundaries exercise
    // maximum native height and writer capacity without padding cells.
    for(unsigned words:{49u,50u,98u,99u,147u,148u,196u,197u,245u,246u})
    {
        const display_storage_source requested{plan_guest_surface(0,words,1,4),{{0x6000,0,words*4}},false,0};std::vector<display_storage_source> cold;
        const auto plan=plan_five_output_regions(memory,{},requested,1u<<20,4096,&cold,7);
        const unsigned count=words/49+unsigned(words%49>=7)+unsigned(words%7!=0);
        require(bool(plan)==(count<=5));if(!plan){require(cold.empty());++refusals;continue;}
        std::size_t written=0;std::vector<std::vector<bool>> native;std::vector<bool> packed(words);
        for(const auto& t:cold){require(t.geometry.width<=7&&t.geometry.height<=7);written+=t.geometry.width*t.geometry.height;native.emplace_back(t.geometry.width*t.geometry.height);}
        require(written==words&&cold.size()==count);
        for(const auto& r:*plan)
        {
            require(r.output<cold.size()&&r.packed.y==0&&r.packed.height==1&&r.packed.width==r.native.width&&r.native.height==1&&r.packed.x+r.packed.width<=words);
            const auto& t=cold[r.output];require(r.native.x+r.native.width<=t.geometry.width&&r.native.y<t.geometry.height);
            for(unsigned x=0;x<r.packed.width;++x)
            {
                const auto canonical=r.packed.x+x,cell=r.native.y*t.geometry.width+r.native.x+x;require(!packed[canonical]&&!native[r.output][cell]);packed[canonical]=true;native[r.output][cell]=true;
                for(unsigned lane=0;lane<4;++lane){const auto at=cell*4+lane;const auto* span=texture_assembly_segment(t.segments,at);require(span&&span->address+at-span->offset==0x6000+canonical*4+lane);++checked;}
            }
        }
        require(std::all_of(packed.begin(),packed.end(),[](bool v){return v;}));for(const auto& target:native)require(std::all_of(target.begin(),target.end(),[](bool v){return v;}));++profiles;
    }
    require(memory.pixel_calls==0);
    std::cout<<"Compact 2D output metadata: "<<profiles<<" zero/mixed-owner linear/Morton/banked/split-mirror and exact-square/full-tile/short-tail plans, "<<checked<<" independently checked byte coordinates, "<<refusals<<" capacity/run/window/identity refusals; every native cell written once, both dimensions and five-writer bound preserved, no VM pixels passed\n";
}

void wide_output_region_cases()
{
    using namespace metal4;
    output_metadata_memory memory;
    std::size_t cases=0, checked=0, guards=0;
    for(unsigned bpp:{2u,4u})for(unsigned count:{1u,5u,6u,16u,32u,63u,64u})for(bool holes:{false,true})
    {
        const unsigned width=count*2, height=4, pitch=width*bpp+3;
        const display_storage_source requested{plan_guest_surface(0,width,height,bpp,pitch),{{0x6000,0,(height-1)*pitch+width*bpp}},false,0};
        std::vector<display_storage_source> targets;
        for(unsigned i=0;i<count-unsigned(holes);++i)
        {
            const auto geometry=plan_guest_surface(0,2,height,bpp,2*bpp+1);
            std::vector<guest_storage_segment> segments;
            for(unsigned at=0;at<geometry.range.size;++at)
            {
                const auto y=at/geometry.row_pitch, byte=at%geometry.row_pitch;
                const auto address=byte<2*bpp?0x6000+y*pitch+i*2*bpp+byte+(i%2?0x1000000:0):0xe00000+i*64+at;
                if(!segments.empty()&&std::uint64_t(segments.back().address)+segments.back().size==address)++segments.back().size;
                else segments.push_back({static_cast<unsigned>(address),at,1});
            }
            targets.push_back({geometry,std::move(segments),false,0});
        }
        std::vector<display_storage_source> cold;
        const auto plan=plan_scaled_image_output_regions(memory,targets,requested,1u<<20,4096,holes?&cold:nullptr,8);
        require(plan.has_value()&&cold.size()==unsigned(holes));
        targets.insert(targets.end(),cold.begin(),cold.end());
        require(targets.size()==count);
        std::vector<bool> packed(width*height);
        std::vector<std::vector<bool>> used;
        for(const auto& target:targets)used.emplace_back(target.geometry.width*target.geometry.height);
        for(const auto& region:*plan)
        {
            require(region.output<targets.size()&&region.packed.width==region.native.width&&region.packed.height==region.native.height);
            const auto& target=targets[region.output];
            require(region.native.x+region.native.width<=target.geometry.width&&region.native.y+region.native.height<=target.geometry.height);
            for(unsigned y=0;y<region.packed.height;++y)for(unsigned x=0;x<region.packed.width;++x)
            {
                const auto px=region.packed.x+x, py=region.packed.y+y;
                const auto nx=region.native.x+x, ny=region.native.y+y;
                require(px<width&&py<height);
                const auto from=py*width+px, to=ny*target.geometry.width+nx;
                require(!packed[from]&&!used[region.output][to]);packed[from]=true;used[region.output][to]=true;
                for(unsigned lane=0;lane<bpp;++lane)
                {
                    const auto at=ny*target.geometry.row_pitch+nx*bpp+lane;
                    const auto* span=texture_assembly_segment(target.segments,at);
                    require(span&&(span->address+at-span->offset)%0x1000000==0x6000+py*pitch+px*bpp+lane);
                    ++checked;
                }
            }
        }
        require(std::all_of(packed.begin(),packed.end(),[](bool value){return value;}));
        for(const auto& target:used)require(std::all_of(target.begin(),target.end(),[](bool value){return value;}));
        if(count>5)
        {
            bool refused=false;
            try{refused=!plan_scaled_image_output_regions(memory,std::span(targets).first(count-unsigned(holes)),requested,1u<<20,4096,&cold,8,5);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused&&cold.empty());++guards;
        }
        ++cases;
    }
    const display_storage_source requested{plan_guest_surface(0,65,1,4),{{0x6000,0,260}},false,0};
    std::vector<display_storage_source> targets;
    for(unsigned i=0;i<65;++i)targets.push_back({plan_guest_surface(0,1,1,4),{{0x6000+i*4,0,4}},false,0});
    for(const auto budget:{std::size_t(0),std::size_t(64),std::size_t(65)})
    {
        bool refused=false;std::vector<display_storage_source> cold{requested};
        try{(void)plan_scaled_image_output_regions(memory,targets,requested,1u<<20,4096,&cold,8,budget);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused&&cold.empty());++guards;
    }
    for(unsigned words:{64u,65u})
    {
        const display_storage_source raster{plan_guest_surface(0,words,1,4),{{0x6000,0,words*4}},false,0};
        std::vector<display_storage_source> cold;
        const auto plan=plan_scaled_image_output_regions(memory,{},raster,1u<<20,4096,&cold,1);
        require(bool(plan)==(words==64));
        if(!plan){require(cold.empty());++guards;continue;}
        require(cold.size()==64&&plan->size()==64);
        for(unsigned i=0;i<64;++i)
        {
            const auto& target=cold[i];const auto& region=(*plan)[i];
            require(target.geometry.width==1&&target.geometry.height==1&&target.segments.size()==1&&
                target.segments[0].address==0x6000+i*4&&target.segments[0].size==4&&
                region.output==i&&region.packed.x==i&&region.packed.width==1&&region.native.x==0);
            checked+=4;
        }
        ++cases;
    }
    require(memory.pixel_calls==0);
    std::cout<<"Wide output metadata: "<<cases<<" plans with 1/5/6/16/32/63/64 writers, "<<checked
        <<" independently mapped byte coordinates, "<<guards<<" smaller-budget/65-writer/invalid-budget refusals; packed and native cells written once, cold remainder and split mirrors, no VM pixels passed\n";
}

void byte_output_region_cases()
{
    using namespace metal4;
    output_metadata_memory memory;
    std::size_t cases=0, checked=0, guards=0, windows=0;
    const auto storage=[](const guest_surface_layout& g,unsigned x,unsigned y,unsigned lane)
    {
        if(!g.morton)return y*g.row_pitch+x*g.pixel_bytes+lane;
        x+=g.morton->origin[0];y+=g.morton->origin[1];std::size_t word=0,shift=0;
        for(unsigned bit=1;bit<1024;bit*=2)for(unsigned axis=0;axis<2;++axis)
            if(bit<g.morton->storage[axis]){word|=std::size_t(bool((axis?y:x)&bit))<<shift;++shift;}
        return word*g.pixel_bytes+lane;
    };
    for(unsigned bpp:{2u,4u})for(unsigned count:{1u,2u,5u,16u,64u})for(unsigned layout=0;layout<4;++layout)for(unsigned seed=0;seed<3;++seed)
    {
        const unsigned width=count*3,height=2;
        const auto requested_sw=layout&2?std::optional<swizzle_region>{{{std::bit_ceil(width+1),8,1},{1,2,0},{width,height,1}}}:std::nullopt;
        const auto geometry=plan_guest_surface(0,width,height,bpp,requested_sw?0:width*bpp+5,requested_sw);
        display_storage_source requested{geometry,{{0x6000,0,geometry.range.size}},false,0};
        const auto native_sw=layout&1?std::optional<swizzle_region>{{{8,8,1},{2,1,0},{5,4,1}}}:std::nullopt;
        const auto native=plan_guest_surface(0,5,4,bpp,native_sw?0:5*bpp+3,native_sw);
        std::vector<std::vector<guest_storage_segment>> segments(count);
        std::vector<std::vector<std::byte>> expected(count),actual(count);
        std::vector<std::byte> payload(width*height*bpp);
        for(unsigned i=0;i<payload.size();++i)payload[i]=std::byte((i*43+seed*71+29)&255);
        for(unsigned output=0;output<count;++output)
        {
            for(unsigned at=0;at<native.range.size;++at)segments[output].push_back({0xe00000+output*0x1000+at,at,1});
            expected[output].resize(20*bpp);
            for(unsigned i=0;i<expected[output].size();++i)expected[output][i]=std::byte((i*17+output*31+91)&255);
        }
        actual=expected;
        unsigned first_offset=0;
        for(unsigned canonical=0;canonical<payload.size();++canonical)
        {
            const unsigned output=canonical%count, local=(canonical/count*7+seed*3)%(20*bpp);
            const auto nx=local/bpp%5,ny=local/bpp/5,lane=local%bpp;
            const auto at=storage(native,nx,ny,lane);
            const auto pixel=canonical/bpp;
            const auto address=0x6000+storage(geometry,pixel%width,pixel/width,canonical%bpp);
            segments[output][at].address=address+(output%2?0x1000000:0);
            expected[output][local]=payload[canonical];
            if(!canonical)first_offset=at;
        }
        std::vector<display_storage_source> targets;
        for(auto& parts:segments)targets.push_back({native,std::move(parts),false,0});
        const unsigned byte_limit=(seed==0?1u:seed==1?5u:31u)*bpp;
        const auto plan=plan_scaled_image_output_byte_regions(memory,targets,requested,byte_limit);
        require(plan.has_value());
        std::vector<bool> canonical_written(payload.size());
        std::vector<std::vector<bool>> native_written(count,std::vector<bool>(20*bpp));
        for(const auto& part:*plan)
        {
            require(part.output<count&&part.bytes==part.native.width*part.native.height*bpp&&part.bytes<=byte_limit&&
                part.native.x+part.native.width<=5&&part.native.y+part.native.height<=4);
            for(const auto& copy:part.copies)
            {
                require(!copy.input&&copy.size&&copy.target+copy.size<=part.bytes&&copy.source+copy.size<=payload.size());
                for(unsigned i=0;i<copy.size;++i)
                {
                    const auto local=copy.target+i,canonical=copy.source+i;
                    const auto nx=part.native.x+local/bpp%part.native.width,ny=part.native.y+local/bpp/part.native.width;
                    const auto target=(ny*5+nx)*bpp+local%bpp;
                    require(!canonical_written[canonical]&&!native_written[part.output][target]);
                    canonical_written[canonical]=true;native_written[part.output][target]=true;
                    actual[part.output][target]=payload[canonical];++checked;
                }
            }
            ++windows;
        }
        require(actual==expected&&std::all_of(canonical_written.begin(),canonical_written.end(),[](bool v){return v;}));
        require(!plan_scaled_image_output_byte_regions(memory,targets,requested,bpp-1));++guards;
        require(!plan_scaled_image_output_byte_regions(memory,targets,requested,byte_limit,payload.size()-1));++guards;
        require(!plan_scaled_image_output_byte_regions(memory,targets,requested,byte_limit,1u<<20,0));++guards;
        require(!plan_scaled_image_output_byte_regions(memory,targets,requested,byte_limit,1u<<20,plan->size()-1));++guards;
        auto missing=targets;missing[0].segments[first_offset].address=0xf00000;
        require(!plan_scaled_image_output_byte_regions(memory,missing,requested,byte_limit));++guards;
        auto duplicate=requested;duplicate.segments.clear();
        for(unsigned at=0;at<geometry.range.size;++at)duplicate.segments.push_back({0x6000+at,at,1});
        const auto first=storage(geometry,0,0,0),second=storage(geometry,0,0,1);
        duplicate.segments[second].address=duplicate.segments[first].address;
        require(!plan_scaled_image_output_byte_regions(memory,targets,duplicate,byte_limit));++guards;
        memory.identities=false;require(!plan_scaled_image_output_byte_regions(memory,targets,requested,byte_limit));memory.identities=true;++guards;
        ++cases;
    }
    require(memory.pixel_calls==0);
    std::cout<<"Native output byte metadata: "<<cases<<" independent lane/owner permutations, "<<windows<<" bounded windows, "<<checked
        <<" exact mapped bytes, "<<guards<<" missing/duplicate/identity/scratch/claim/window refusals; 1/2/5/16/64 owners, RGB565/ARGB8 words, pitched/Morton views, split mirrors and all unselected native lanes preserved without VM pixels passed\n";
}

void cold_byte_output_region_cases()
{
    using namespace metal4;
    output_metadata_memory memory;std::size_t cases=0,checked=0,cold_bytes=0,guards=0,windows=0;
    const auto offset=[](const guest_surface_layout& g,unsigned x,unsigned y,unsigned lane)
    {
        if(!g.morton)return y*g.row_pitch+x*g.pixel_bytes+lane;
        x+=g.morton->origin[0];y+=g.morton->origin[1];std::size_t word=0,shift=0;
        for(unsigned bit=1;bit<1024;bit*=2)for(unsigned axis=0;axis<2;++axis)
            if(bit<g.morton->storage[axis]){word|=std::size_t(bool((axis?y:x)&bit))<<shift;++shift;}
        return word*g.pixel_bytes+lane;
    };
    for(unsigned bpp:{2u,4u})for(unsigned count:{1u,2u,5u,16u,63u})for(unsigned layout=0;layout<4;++layout)for(unsigned seed=0;seed<3;++seed)
    {
        const unsigned width=count*3+1,height=2;
        const auto sw=layout&2?std::optional<swizzle_region>{{{std::bit_ceil(width+1),8,1},{1,2,0},{width,height,1}}}:std::nullopt;
        const auto geometry=plan_guest_surface(0,width,height,bpp,sw?0:width*bpp+3,sw);
        const display_storage_source requested{geometry,{{0x6000,0,geometry.range.size}},false,0};
        std::vector<std::byte> payload(width*height*bpp);std::vector<unsigned> addresses(payload.size());
        std::map<unsigned,unsigned> canonical;
        for(unsigned j=0;j<payload.size();++j){payload[j]=std::byte((j*43+seed*17+59)&255);addresses[j]=0x6000+offset(geometry,j/bpp%width,j/bpp/width,j%bpp);canonical[addresses[j]]=j;}
        std::vector<display_storage_source> targets;
        std::vector<std::vector<std::byte>> expected,actual;std::vector<unsigned> ordinals(count);
        for(unsigned output=0;output<count;++output)
        {
            const unsigned native_bpp=std::array{1u,2u,4u,8u,16u}[output%5];
            const auto native_sw=layout&1?std::optional<swizzle_region>{{{8,16,1},{2,1,0},{5,8,1}}}:std::nullopt;
            const auto native=plan_guest_surface(0,5,8,native_bpp,native_sw?0:5*native_bpp+3,native_sw);
            std::vector<guest_storage_segment> parts;for(unsigned at=0;at<native.range.size;++at)parts.push_back({0xe00000+output*0x1000+at,at,1});
            targets.push_back({native,std::move(parts),false,0});expected.emplace_back(40*native_bpp);
            for(unsigned at=0;at<expected.back().size();++at)expected.back()[at]=std::byte((at*31+output*19+81)&255);
        }
        actual=expected;
        for(unsigned j=0;j<payload.size();++j)if(j%3)
        {
            const unsigned output=j/3%count,native_bpp=targets[output].geometry.pixel_bytes;
            const unsigned local=(ordinals[output]++*7+seed*3)%(40*native_bpp);
            const auto at=offset(targets[output].geometry,local/native_bpp%5,local/native_bpp/5,local%native_bpp);
            targets[output].segments[at].address=addresses[j]+(output%2?0x1000000:0);
            expected[output][local]=payload[j];
        }
        std::vector<display_storage_source> cold;
        const auto plan=plan_scaled_image_output_byte_regions(memory,targets,requested,64,1u<<20,4096,&cold,1024);
        require(plan.has_value()&&!cold.empty()&&targets.size()+cold.size()<=64);
        const auto previous=count;
        for(const auto& target:cold)
        {
            require(target.geometry.pixel_bytes==1);expected.emplace_back(target.geometry.width*target.geometry.height);
            actual.emplace_back(expected.back().size(),std::byte{0xcd});
            for(unsigned y=0;y<target.geometry.height;++y)for(unsigned x=0;x<target.geometry.width;++x)
            {
                const auto at=y*target.geometry.row_pitch+x;const auto* span=texture_assembly_segment(target.segments,at);require(span);
                const unsigned address=(span->address+at-span->offset)%0x1000000;const auto found=canonical.find(address);
                require(found!=canonical.end()&&found->second%3==0);expected.back()[y*target.geometry.width+x]=payload[found->second];++cold_bytes;
            }
        }
        targets.insert(targets.end(),cold.begin(),cold.end());std::vector<bool> used(payload.size());
        std::vector<std::vector<bool>> native_used;for(const auto& target:targets)native_used.emplace_back(target.geometry.width*target.geometry.height*target.geometry.pixel_bytes);
        for(const auto& part:*plan)
        {
            require(part.output<targets.size()&&part.preserve_background==(part.output<previous));const auto& g=targets[part.output].geometry;
            require(part.bytes==part.native.width*part.native.height*g.pixel_bytes&&part.bytes<=64);
            std::size_t written=0;
            for(const auto& copy:part.copies)for(unsigned j=0;j<copy.size;++j)
            {
                require(copy.source+j<payload.size()&&copy.target+j<part.bytes&&!used[copy.source+j]);used[copy.source+j]=true;
                const auto local=copy.target+j,x=part.native.x+local/g.pixel_bytes%part.native.width,y=part.native.y+local/g.pixel_bytes/part.native.width;
                const auto at=(y*g.width+x)*g.pixel_bytes+local%g.pixel_bytes;
                require(!native_used[part.output][at]);native_used[part.output][at]=true;actual[part.output][at]=payload[copy.source+j];++written;++checked;
            }
            require(part.preserve_background||written==part.bytes);++windows;
        }
        require(actual==expected&&std::all_of(used.begin(),used.end(),[](bool v){return v;}));
        cold={requested};require(!plan_scaled_image_output_byte_regions(memory,std::span(targets).first(previous),requested,64,1u<<20,4096,&cold,0)&&cold.empty());++guards;
        cold={requested};require(!plan_scaled_image_output_byte_regions(memory,std::span(targets).first(previous),requested,64,1u<<20,4096,&cold,1024,previous)&&cold.empty());++guards;
        cold={requested};require(!plan_scaled_image_output_byte_regions(memory,std::span(targets).first(previous),requested,64,payload.size()-1,4096,&cold,1024)&&cold.empty());++guards;
        ++cases;
    }
    require(memory.pixel_calls==0);
    std::cout<<"Cold native output byte metadata: "<<cases<<" mixed 1/2/4/8/16-byte-owner plans, "<<windows<<" windows, "<<checked
        <<" independent byte coordinates, "<<cold_bytes<<" fully written cold B8 cells, "<<guards
        <<" dimension/slot/claim refusals; odd missing byte counts, 1..63 old owners, 64 total targets, pitched/Morton/split mirrors, no invented background or VM pixels passed\n";
}

void factored_output_region_cases()
{
    using namespace metal4;output_metadata_memory memory;std::size_t profiles=0,accepted=0,refusals=0,checked=0;
    for(unsigned limit:{2u,3u,4u,5u,7u,8u,13u,16u,31u})
    {
        std::vector<bool> possible(limit*limit+2);for(unsigned w=1;w<=limit;++w)for(unsigned h=1;h<=limit;++h)possible[w*h]=true;
        for(unsigned bpp:{2u,4u})for(unsigned split:{0u,1u})for(unsigned words=1;words<=limit*limit+1;++words)
        {
            const auto total=words+4;const auto g=plan_guest_surface(0,total,1,bpp);
            const std::vector<guest_storage_segment> spans=split?std::vector<guest_storage_segment>{{0x6000,0,3},{0x1006003,3,g.range.size-3}}:std::vector<guest_storage_segment>{{0x6000,0,g.range.size}};
            const display_storage_source requested{g,spans,false,0};std::vector<display_storage_source> old;
            for(unsigned i=0;i<4;++i){const auto word=i*total/4;old.push_back({plan_guest_surface(0,1,1,bpp),{{0x6000+word*bpp+(i%2?0x1000000:0),0,bpp}},false,0});}
            std::vector<display_storage_source> cold{requested};const auto plan=plan_five_output_regions(memory,old,requested,1u<<20,4096,&cold,limit);++profiles;
            require(bool(plan)==possible[words]);if(!plan){require(cold.empty());++refusals;continue;}
            require(cold.size()==1&&cold[0].geometry.width<=limit&&cold[0].geometry.height<=limit&&cold[0].geometry.width*cold[0].geometry.height==words);
            auto all=old;all.push_back(cold[0]);std::vector<bool> packed(total);std::vector<std::vector<bool>> used;
            for(const auto& t:all)used.emplace_back(t.geometry.width*t.geometry.height);
            for(const auto& r:*plan)
            {
                require(r.output<5&&r.packed.y==0&&r.packed.height==1&&r.packed.width==r.native.width&&r.native.height==1&&r.packed.x+r.packed.width<=total);
                const auto& t=all[r.output];require(r.native.x+r.native.width<=t.geometry.width&&r.native.y<t.geometry.height);
                for(unsigned x=0;x<r.packed.width;++x)
                {
                    const auto canonical=r.packed.x+x,cell=r.native.y*t.geometry.width+r.native.x+x;
                    require(!packed[canonical]&&!used[r.output][cell]);packed[canonical]=true;used[r.output][cell]=true;
                    for(unsigned lane=0;lane<bpp;++lane){const auto at=cell*bpp+lane;const auto* span=texture_assembly_segment(t.segments,at);require(span&&(span->address+at-span->offset)%0x1000000==0x6000+canonical*bpp+lane);++checked;}
                }
            }
            require(std::all_of(packed.begin(),packed.end(),[](bool v){return v;}));for(const auto& target:used)require(std::all_of(target.begin(),target.end(),[](bool v){return v;}));++accepted;
        }
    }
    for(unsigned prefix=1;prefix<=4;++prefix)for(unsigned bpp:{2u,4u})for(unsigned split:{0u,1u})
    {
        const unsigned owners=4-prefix,words=prefix*49+36,total=words+owners;
        const auto g=plan_guest_surface(0,total,1,bpp);const std::vector<guest_storage_segment> spans=split?std::vector<guest_storage_segment>{{0x6000,0,3},{0x1006003,3,g.range.size-3}}:std::vector<guest_storage_segment>{{0x6000,0,g.range.size}};
        const display_storage_source requested{g,spans,false,0};std::vector<display_storage_source> old;
        for(unsigned i=0;i<owners;++i)old.push_back({plan_guest_surface(0,1,1,bpp),{{0x6000+i*bpp,0,bpp}},false,0});
        std::vector<display_storage_source> cold;const auto plan=plan_five_output_regions(memory,old,requested,1u<<20,4096,&cold,7);require(plan.has_value()&&cold.size()==prefix+1);
        for(unsigned i=0;i<prefix;++i)require(cold[i].geometry.width==7&&cold[i].geometry.height==7);require(cold.back().geometry.width==6&&cold.back().geometry.height==6);
        auto all=old;all.insert(all.end(),cold.begin(),cold.end());std::vector<bool> packed(total);std::vector<std::vector<bool>> used;for(const auto& t:all)used.emplace_back(t.geometry.width*t.geometry.height);
        for(const auto& r:*plan)
        {
            require(r.output<5&&r.packed.y==0&&r.packed.height==1&&r.packed.width==r.native.width&&r.native.height==1&&r.packed.x+r.packed.width<=total);const auto& t=all[r.output];require(r.native.x+r.native.width<=t.geometry.width&&r.native.y<t.geometry.height);
            for(unsigned x=0;x<r.packed.width;++x)
            {
                const auto canonical=r.packed.x+x,cell=r.native.y*t.geometry.width+r.native.x+x;require(!packed[canonical]&&!used[r.output][cell]);packed[canonical]=true;used[r.output][cell]=true;
                for(unsigned lane=0;lane<bpp;++lane){const auto at=cell*bpp+lane;const auto* span=texture_assembly_segment(t.segments,at);require(span&&(span->address+at-span->offset)%0x1000000==0x6000+canonical*bpp+lane);++checked;}
            }
        }
        require(std::all_of(packed.begin(),packed.end(),[](bool v){return v;}));for(const auto& target:used)require(std::all_of(target.begin(),target.end(),[](bool v){return v;}));++profiles;++accepted;
    }
    require(memory.pixel_calls==0);
    std::cout<<"Factored fifth output metadata: "<<profiles<<" exhaustive one-free-writer and square-prefix plans, "<<accepted<<" exact rectangles, "<<refusals<<" impossible/capacity refusals, "<<checked<<" independent byte coordinates; four distributed old owners, split mirrors, dimension limits 2..31, brute-force rectangle oracle, one to four full-prefix targets and no VM pixels passed\n";
}

int main()
{
	try
	{
        partial_output_region_cases();
        partial_output_region_cases(true);
        partial_output_region_cases(true,true);
        cold_only_output_region_cases();
        atlas_output_region_cases();
        factored_output_region_cases();
        wide_output_region_cases();
        byte_output_region_cases();
        cold_byte_output_region_cases();
        rectangle_coverage_cases();
        bounded_input_region_cases();
		rsx::blit_src_info source{};
		source.format = rsx::blit_engine::transfer_source_format::a8r8g8b8;
		source.origin = rsx::blit_engine::transfer_origin::center;
		source.width = 8;
		source.height = 4;
		source.pitch = 48;
		source.bpp = 4;
		source.offset_x = 2;
		source.offset_y = 1;
		source.rsx_address = 0x1000;
		source.pixels = reinterpret_cast<u8*>(1);
		rsx::blit_dst_info target{};
		target.format = rsx::blit_engine::transfer_destination_format::r5g6b5;
		target.width = 16;
		target.height = 8;
		target.pitch = 48;
		target.bpp = 2;
		target.rsx_address = 0x4000;
		target.offset_x = 3;
		target.offset_y = 2;
		target.clip_x = 2;
		target.clip_y = 1;
		target.clip_width = 4;
		target.clip_height = 3;
		target.scale_x = target.scale_y = 2;
		target.pixels = reinterpret_cast<u8*>(2);
		auto plan = metal4::capture_scaled_image(source, target, 16, 8, true, 4096);
		require(plan.source.range.address == 0x1038 && plan.source.width == 6 && plan.source.height == 3 && plan.source.row_pitch == 48);
		require(plan.destination.range.address == 0x4066 && plan.destination.width == 4 && plan.destination.height == 3 && plan.destination.row_pitch == 48);
		require(plan.input_width == 8 && plan.scaled_width == 16 && plan.clip_x == 2 && plan.clip_y == 1 && plan.interpolate);
		source.offset_x = 7;
		source.offset_y = 3;
		target.scale_x = target.scale_y = -2;
		plan = metal4::capture_scaled_image(source, target, 16, 8, false, 4096);
		require(plan.source.range.address == 0x1000 && plan.source.width == 8 && plan.source.height == 4 && plan.flip_x && plan.flip_y);
		target.swizzled = true;
		plan = metal4::capture_scaled_image(source, target, 13, 7, false, 4096);
		require(plan.destination.morton && plan.destination.morton->storage == std::array<std::size_t, 3>{16, 8, 1} &&
				plan.destination.morton->origin == std::array<std::size_t, 3>{3, 2, 0} && plan.destination.range.address == 0x4000);
		const auto valid = plan;
		source.offset_x = source.width;
		source.offset_y = source.height;
		const auto edge = metal4::capture_scaled_image(source, target, 16, 8, false, 4096);
		require(edge.source.range.address == source.rsx_address);
		const auto saved_width = target.width;
		target.width = 15;
		refused([&]
				{ (void)metal4::capture_scaled_image(source, target, 16, 8, false, 4096); });
		target.width = saved_width;
		for (unsigned failure = 0; failure < 10; ++failure)
		{
			plan = valid;
			switch (failure)
			{
			case 0:
				plan.clip_x = std::numeric_limits<unsigned>::max();
				break;
			case 1:
				plan.scaled_width = 0;
				break;
			case 2:
				plan.source_format = static_cast<metal4::scaled_image_format>(9);
				break;
			case 3:
				plan.destination.pixel_bytes = 4;
				break;
			case 4:
				plan.input_width = 65537;
				break;
			case 5:
				plan.destination.width = 100;
				break;
			case 6:
				plan.source.row_pitch = 1;
				break;
			case 7:
				plan.source.range.address = 0xffff'ffff;
				break;
			case 8:
				plan.source.morton = metal4::swizzle_region{{8, 4, 2}, {0, 0, 0}, {8, 4, 1}};
				plan.source.range.size = 256;
				break;
			case 9:
				plan.destination.range.size = 0;
				break;
			}
			refused([&]
					{ metal4::validate_scaled_image(plan, 4096); });
		}
		refused([&]
				{ metal4::validate_scaled_image(valid, valid.source.range.size + valid.destination.range.size - 1); });
		for (float value : {0.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
		{
			target.scale_x = value;
			refused([&]
					{ (void)metal4::capture_scaled_image(source, target, 16, 8, false, 4096); });
		}
		const metal4::banked_tile_layout tile{0x10000, 0x40000, 768, 2, 772};
		const auto logical = metal4::plan_guest_surface(tile.address + tile.offset, 8, 4, 4, 48);
		const auto banked = metal4::attach_banked_tile(logical, tile);
		plan = valid;
		plan.source = banked;
		plan.source_tile = tile;
		metal4::validate_scaled_image(plan, 1 << 20);
		const auto coverage = metal4::scaled_image_coverage(banked, plan.source_tile);
		require(coverage.width == tile.size && coverage.height == 1 && coverage.pixel_bytes == 1 && coverage.range.address == tile.address);
		for (std::size_t bytes : {1u, 2u, 4u, 8u, 16u})
		{
			auto wide = tile; wide.offset = static_cast<std::uint32_t>(bytes);
			const auto image = metal4::attach_banked_tile(metal4::plan_guest_surface(wide.address + wide.offset, 8, 4, bytes, wide.pitch), wide);
			metal4::validate_banked_tile(image, wide);
		}
		for (std::size_t bytes : {0u, 3u, 32u})
		{
			auto image = banked; image.pixel_bytes = bytes;
			refused([&] { metal4::validate_banked_tile(image, tile); });
		}
		for (unsigned pitch : {256u, 512u, 768u, 1280u, 1536u, 1792u, 2816u, 3328u, 4096u, 6144u})
			require(metal4::banked_tile_factor(pitch) > 0);
		for (unsigned pitch : {0u, 1u, 255u, 257u, 2304u, 4352u})
			refused([&]
					{ (void)metal4::banked_tile_factor(pitch); });
		for (unsigned failure = 0; failure < 9; ++failure)
		{
			auto bad = tile;
			auto image = banked;
			switch (failure)
			{
			case 0:
				bad.address += 4;
				break;
			case 1:
				bad.size = 0;
				break;
			case 2:
				bad.size -= 1;
				break;
			case 3:
				bad.bank = 4;
				break;
			case 4:
				bad.offset = bad.size;
				break;
			case 5:
				bad.offset = bad.size - 4;
				break;
			case 6:
				image.width = bad.pitch / 4 + 1;
				break;
			case 7:
				image.height = 65537;
				break;
			case 8:
				image.range.address = 0xffff0000;
				break;
			}
			refused([&]
					{ metal4::validate_banked_tile(image, bad); });
		}
		refused([&]
				{ metal4::validate_scaled_image(plan, tile.size + plan.destination.range.size - 1); });
		std::cout << "Native banked plan:1/2/4/8/16-byte geometry with invalid widths refused;10 pitch factorizations,whole-allocation coverage,invalid factor/bank/alignment/range/row/height/budget refused\n";
		std::cout << "Native decoded scaled-image:owned metadata/poisoned guest pointers,offsets/reverse scan/Morton origin/clip/aggregate budget/invalid layout and scale passed;no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
