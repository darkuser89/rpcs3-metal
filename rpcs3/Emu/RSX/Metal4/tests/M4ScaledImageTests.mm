#include "M4GuardedGuestMemory.h"
#include "M4ScaledImage.h"
#include "M4ScaledImageMapping.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>

namespace
{
using namespace metal4;
void require(bool v, const char* message)
{
	if (!v)
		throw std::runtime_error(message);
}
class fixture_access final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<fixture_access>
{
	std::mutex mutex;
	std::array<std::byte, 65536> bytes{};

  public:
	unsigned notifications = 0;
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& op) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			op(*current);
			return;
		}
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		op(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		const auto offset = range.address % bytes.size();
		if (range.address >= 2 * bytes.size() || range.size > bytes.size() - offset)
			throw std::out_of_range("Scaled-image fixture mapping invalid");
		return bytes.data() + offset;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		return overlaps(guest_surface_range{a.address % 65536, a.size}, guest_surface_range{b.address % 65536, b.size});
	}
	void written(guest_surface_range) noexcept override { ++notifications; }
};
std::size_t offset(const guest_surface_layout& layout, unsigned x, unsigned y)
{
	if (!layout.morton)
		return y * layout.row_pitch + x * layout.pixel_bytes;
	x += layout.morton->origin[0];
	y += layout.morton->origin[1];
	std::size_t word = 0;
	unsigned shift = 0;
	for (unsigned bit = 1; bit < layout.morton->storage[0] || bit < layout.morton->storage[1]; bit *= 2)
	{
		if (bit < layout.morton->storage[0])
		{
			word |= std::size_t(bool(x & bit)) << shift;
			++shift;
		}
		if (bit < layout.morton->storage[1])
		{
			word |= std::size_t(bool(y & bit)) << shift;
			++shift;
		}
	}
	return word * layout.pixel_bytes;
}
// Independent scalar rational oracle:normalize each channel and sum the
// four weighted texels as fractions. Neither float nor the GPU's packed
// weight/quantization arithmetic decides expected half-channel results.
struct fraction
{
    using integer=__int128_t;
    integer numerator=0,denominator=1;
    fraction(integer n=0,integer d=1):numerator(n),denominator(d)
    {
        require(d>0,"Reference denominator must be positive");
        integer a=n<0?-n:n,b=d;while(b){const auto r=a%b;a=b;b=r;}
        numerator/=a;denominator/=a;
    }
    friend fraction operator+(fraction a,fraction b){return {a.numerator*b.denominator+b.numerator*a.denominator,a.denominator*b.denominator};}
    friend fraction operator-(fraction a,fraction b){return {a.numerator*b.denominator-b.numerator*a.denominator,a.denominator*b.denominator};}
    friend fraction operator*(fraction a,fraction b){return {a.numerator*b.numerator,a.denominator*b.denominator};}
    fraction& operator+=(fraction b){return *this=*this+b;}
    int floor()const{return int(numerator/denominator-(numerator<0&&numerator%denominator!=0));}
    unsigned rounded()const{require(numerator>=0,"Negative reference channel");return unsigned((numerator*2+denominator)/(denominator*2));}
};
using rgba = std::array<fraction, 4>;
rgba read(const std::vector<std::byte>& bytes, const guest_surface_layout& layout, int x, int y)
{
	x = std::clamp(x, 0, int(layout.width) - 1);
	y = std::clamp(y, 0, int(layout.height) - 1);
	const auto at = offset(layout, x, y);
	const auto u = [&](unsigned i)
	{ return std::to_integer<unsigned>(bytes[at + i]); };
	if (layout.pixel_bytes == 4)
		return {fraction(u(1),255),fraction(u(2),255),fraction(u(3),255),fraction(u(0),255)};
	const auto word = u(0) * 256 + u(1);
	return {fraction((word>>11)&31,31),fraction((word>>5)&63,63),fraction(word&31,31),1};
}
std::vector<std::byte> reference(const scaled_image_plan& p, const std::vector<std::byte>& source, std::vector<std::byte> target)
{
	for (unsigned y = 0; y < p.destination.height; ++y)
		for (unsigned x = 0; x < p.destination.width; ++x)
		{
			const auto to = offset(p.destination, x, y);
			rgba color;
			if (!p.interpolate)
			{
				const auto coord = [](unsigned i, unsigned src, unsigned dst, bool flip)
				{const auto denominator=std::uint64_t(dst)*2;auto numerator=(std::uint64_t(i)*2+1)*src;if(flip)numerator=std::uint64_t(src)*denominator-numerator;return unsigned(numerator/denominator); };
				const auto a = std::min<unsigned>(coord(x + p.clip_x, p.input_width, p.scaled_width, p.flip_x), p.source.width - 1);
				const auto b = std::min<unsigned>(coord(y + p.clip_y, p.input_height, p.scaled_height, p.flip_y), p.source.height - 1);
				if (p.source_format == p.destination_format)
				{
					std::copy_n(source.begin() + offset(p.source, a, b), p.source.pixel_bytes, target.begin() + to);
					continue;
				}
				color = read(source, p.source, a, b);
			}
			else
			{
				fraction u((2ull*(x+p.clip_x)+1)*p.input_width,2ull*p.scaled_width),v((2ull*(y+p.clip_y)+1)*p.input_height,2ull*p.scaled_height);
				if (p.flip_x)
					u = p.input_width - u;
				if (p.flip_y)
					v = p.input_height - v;
				u = u-fraction(1,2);
				v = v-fraction(1,2);
				const int a=u.floor(),b=v.floor();
				const auto dx=u-a,dy=v-b;
				color = {};
				for (unsigned yy = 0; yy < 2; ++yy)
					for (unsigned xx = 0; xx < 2; ++xx)
					{
						const auto texel = read(source, p.source, a + xx, b + yy);
						const auto weight = (xx ? dx : 1 - dx) * (yy ? dy : 1 - dy);
						for (unsigned c = 0; c < 4; ++c)
							color[c] += texel[c] * weight;
					}
			}
			if (p.destination.pixel_bytes == 4)
			{
				target[to] = std::byte((color[3]*255).rounded());
				for (unsigned c = 0; c < 3; ++c)
					target[to + 1 + c] = std::byte((color[c]*255).rounded());
			}
			else
			{
				const auto word = (unsigned((color[0]*31).rounded()) << 11) | (unsigned((color[1]*63).rounded()) << 5) | unsigned((color[2]*31).rounded());
				target[to] = std::byte(word >> 8);
				target[to + 1] = std::byte(word);
			}
		}
	return target;
}
class native_access final : public exclusive_guest_memory,public exclusive_guest_memory::view,public std::enable_shared_from_this<native_access>
{
	std::mutex mutex;
	std::vector<std::byte> bytes=std::vector<std::byte>(1u<<20,std::byte{0xcd});
  public:
	unsigned notifications=0;
	void execute(const std::function<void(view&)>& work) override
	{
		if(auto* active=guest_memory_epoch::current(*this)){work(*active);return;}
		std::lock_guard lock(mutex);guest_memory_epoch epoch(*this,*this);work(*this);
	}
	std::byte* map(guest_surface_range range,bool) override
	{
		const auto offset=range.address%bytes.size();if(range.address>=2*bytes.size()||range.size>bytes.size()-offset)throw std::out_of_range("Native scale fixture range");return bytes.data()+offset;
	}
	bool same_storage(guest_surface_range a,guest_surface_range b) override {return overlaps(guest_surface_range{a.address%unsigned(bytes.size()),a.size},guest_surface_range{b.address%unsigned(bytes.size()),b.size});}
	std::optional<guest_surface_range> rebase_storage(guest_surface_range source,guest_surface_range request) override
	{
		const auto a=source.address%unsigned(bytes.size()),b=request.address%unsigned(bytes.size());if(b<a||request.size>source.size||b-a>source.size-request.size)return {};return guest_surface_range{source.address+b-a,request.size};
	}
	bool equivalent_storage(guest_surface_range a,guest_surface_range b) override {return a.address%bytes.size()==b.address%bytes.size()&&a.size==b.size;}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override {return shared_from_this();}
	void written(guest_surface_range) noexcept override {++notifications;}
	std::vector<std::byte> copy() {std::lock_guard lock(mutex);return bytes;}
};
class native_memory final : public guest_surface_memory
{
  public:
	std::shared_ptr<native_access> access=std::make_shared<native_access>();
	std::shared_ptr<guarded_guest_memory> raw=std::make_shared<guarded_guest_memory>(access);
	bool forbid_pixels=false;unsigned reads=0,writes=0;
    std::vector<guest_surface_range> allowed_input_reads;
    void check_read(guest_surface_range range)const
    {
        if(!forbid_pixels)return;
        constexpr std::uint64_t mirror=1u<<20;const auto address=range.address%mirror;
        require(std::any_of(allowed_input_reads.begin(),allowed_input_reads.end(),[&](const auto& p){const auto begin=p.address%mirror;return address>=begin&&address-begin<=p.size&&range.size<=p.size-(address-begin);}),"Native scale captured VM outside its RAM input");
    }
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override {return raw->watch_surface(coverage);}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override {check_read(coverage.range);++reads;return raw->read_surface(coverage);}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override {return raw->surface_revision(coverage);}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range r) override {return raw->storage_identity(r);}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override {return raw->retain_backing(ranges);}
	bool test_storage(const storage_test& test) override {return raw->test_storage(test);}
	bool test_rebase(const rebase_test& test) override {return raw->test_rebase(test);}
	std::shared_ptr<const physical_surface> capture_physical_surface(const guest_surface_layout& c,std::uint64_t v) override {return raw->capture_physical_surface(c,v);}
	bool physical_surface_current(const physical_surface& p) override {return raw->physical_surface_current(p);}
	std::optional<bool> physical_surface_overlaps_current(const physical_surface& p,guest_surface_range r) override {return raw->physical_surface_overlaps_current(p,r);}
	std::optional<bool> physical_identity_overlaps_current(const physical_identity& p,const guest_surface_layout& r) override {return raw->physical_identity_overlaps_current(p,r);}
	std::optional<bool> physical_surface_virtual_current(const physical_surface& p,std::uint64_t v) override {return raw->physical_surface_virtual_current(p,v);}
	std::optional<guest_surface_snapshot> read_physical_surface(const physical_surface& p) override {check_read(p.coverage().range);++reads;return raw->read_physical_surface(p);}
	std::optional<physical_surface_publication> compare_write_physical(std::span<const physical_surface_write> w,std::span<const std::uint64_t> v) override {require(!forbid_pixels,"Native scale wrote physical VMpixels");++writes;return raw->compare_write_physical(w,v);}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& c,std::uint64_t v,std::span<const std::byte> b) override {require(!forbid_pixels,"Native scale wrote VMpixels");++writes;return raw->compare_write(c,v,b);}
	std::optional<std::uint64_t> compare_write_many(std::span<const guest_surface_write> w) override {require(!forbid_pixels,"Native scale scattered VMpixels");++writes;return raw->compare_write_many(w);}
	void write(guest_surface_range r,std::span<const std::byte> b) override {require(!forbid_pixels,"Native scale stored VMpixels");++writes;raw->write(r,b);}
};
void native_scaled_image_cases(bool cold=false,bool ram=false,unsigned deferred=0)
{
    const bool ram_background=ram||deferred==5;
	context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);sample_transfer samples(ctx);unsigned profiles=0;std::size_t bytes_checked=0,source_bytes_checked=0;
	for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned layout=0;layout<9;++layout)for(unsigned count:{1u,2u,4u})for(unsigned flags=0;flags<8;++flags)
	{
        if(ram&&cold&&count!=1)continue;
		@autoreleasepool
		{
			auto memory=std::make_shared<native_memory>();const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned address,unsigned,std::size_t){return address;},1u<<20,ledger,64,{capture,io});scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
			const bool banked=layout>=4&&layout<=6,sm=layout==1||layout==3||layout==6||layout==8,dm=layout==2||layout==3||layout==6||layout==8;const unsigned pitch=layout==5?768:256;
			const auto ssw=sm?std::optional<swizzle_region>{{{16,16,1},{2,3,0},{8,8,1}}}:std::nullopt,dsw=dm?std::optional<swizzle_region>{{{16,16,1},{2,3,0},{8,8,1}}}:std::nullopt;
			const auto st=banked?std::optional<banked_tile_layout>{{0x10000,0x40000,pitch,1,0x1200+in}}:std::nullopt,dt=banked?std::optional<banked_tile_layout>{{0x80000,0x40000,pitch,3,0x1400+out}}:std::nullopt;
			const unsigned sa=banked?st->address+st->offset:0x1000,da=banked?dt->address+dt->offset:0x6000;const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
			const auto sf=in==2?surface_color::rgb565:surface_color::argb8,df=out==2?surface_color::rgb565:surface_color::argb8;
            cached_surface_description sd{sa,{8/factors.x,8/factors.y,count,sf,false},sm?0:banked?pitch:8*in+1,ssw,st},dd{da,{8/factors.x,8/factors.y,count,df,false},dm?0:banked?pitch:8*out+1,dsw,dt};
            const auto split=[](guest_surface_layout geometry)
            {
                std::vector<guest_storage_segment> result;std::size_t begin=0;
                for(const auto end:std::array<std::size_t,6>{1,5,37,71,111,geometry.range.size})
                {result.push_back({geometry.range.address+unsigned(result.size())*0x10000,begin,end-begin});begin=end;}
                return result;
            };
            if(layout>=7){sd.segments=split(plan_guest_surface(sa,8,8,in,sd.row_pitch,ssw));dd.segments=split(plan_guest_surface(da,8,8,out,dd.row_pitch,dsw));}
            auto source=ram?cached_surface_ref{}:surfaces.acquire(sd);auto destination=cold?cached_surface_ref{}:surfaces.acquire(dd);
			std::vector<std::byte> input(64*in),background(64*out,cold?std::byte{0xcd}:std::byte{0xa5});for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+i/5+73)&255);
			const auto seed=[&](const cached_surface_ref& surface,surface_color format,const std::vector<std::byte>& data)
			{
				surfaces.prepare(surface,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images)
				{
					id<MTLTexture> target=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;target=make_texture(commands,d);}texture_region r;r.size=MTLSizeMake(8,8,1);colors.upload(commands,format,make_upload(commands,data),{},target,r);if(count>1)samples.import(commands,target,images[0]);
				});
			};
            const auto original_input=input;
            if(ram_background)
            {
                const auto geometry=plan_guest_surface(sa,8,8,in,sm?0:sd.row_pitch,ssw);
                for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x)for(unsigned lane=0;lane<in;++lane)
                {
                    auto at=offset(geometry,x,y)+lane;if(st)at=banked_display_storage_offset(*st,st->offset+at);
                    std::uint64_t address=std::uint64_t(st?st->address:sa)+at;
                    if(!sd.segments.empty()){const auto part=std::find_if(sd.segments.begin(),sd.segments.end(),[&](const auto& p){return at>=p.offset&&at-p.offset<p.size;});require(part!=sd.segments.end(),"RAM input seed missed IO segment");address=std::uint64_t(part->address)+at-part->offset;}
                    memory->raw->write({static_cast<unsigned>(address),1},std::span{&input[(y*8+x)*in+lane],1});
                }
                if(!sd.segments.empty())for(const auto& part:sd.segments)memory->allowed_input_reads.push_back({part.address,part.size});
                else memory->allowed_input_reads.push_back(st?guest_surface_range{st->address,st->size}:geometry.range);
            }
            else if(!deferred||deferred==3||deferred==4)seed(source,sf,input);
            if(deferred)
            {
                auto primary=surfaces.acquire({0xd0000,{8/factors.x,8/factors.y,count,sf,false},8*in,{}}),other=surfaces.acquire({0xd4000,{8/factors.x,8/factors.y,count,sf,false},8*in,{}});
                auto alternate=original_input;for(unsigned i=0;i<alternate.size();++i)alternate[i]=std::byte((i*53+i/7+119)&255);seed(primary,sf,original_input);seed(other,sf,alternate);
                if(deferred==4)surfaces.spill(source);
                const auto width=8/factors.x,height=8/factors.y;
                if(deferred==1)surfaces.defer_copy(primary,{0,0,width,height},source,0,0);
                else if(deferred==2)
                {
                    surfaces.defer_copy(primary,{0,0,width/2,height},source,0,0);surfaces.defer_copy(other,{width/2,0,width-width/2,height},source,width/2,0);
                    for(unsigned y=0;y<8;++y)for(unsigned x=4;x<8;++x)std::copy_n(alternate.begin()+(y*8+x)*in,in,input.begin()+(y*8+x)*in);
                }
                else
                {
                    surfaces.defer_copy(other,{1,1,width-2,height-2},source,1,1);
                    for(unsigned y=factors.y;y<8-factors.y;++y)for(unsigned x=factors.x;x<8-factors.x;++x)std::copy_n(alternate.begin()+(y*8+x)*in,in,input.begin()+(y*8+x)*in);
                }
                if(deferred==2&&(flags&1))
                {
                    surfaces.spill(primary);surfaces.clear(primary,0xf0,0xff128904,0,255,{0,0,width,height});
                    require(surfaces.status(source).deferred_copies==2,"Deferred source epoch was materialized before the transfer");
                }
                require(surfaces.status(source).deferred_copies==(deferred==2?2u:1u),"Native deferred fixture lacks pending sections");
            }
            if(!cold)seed(destination,df,background);
			auto sg=plan_guest_surface(sa,3,2,in,sm?0:banked?pitch:8*in+1,sm?std::optional<swizzle_region>{{{16,16,1},{3,5,0},{3,2,1}}}:std::nullopt);
			auto dg=plan_guest_surface(da,4,3,out,dm?0:banked?pitch:8*out+1,dm?std::optional<swizzle_region>{{{16,16,1},{4,4,0},{4,3,1}}}:std::nullopt);
			auto sti=st,dti=dt;if(banked){if(!sm)sti->offset+=2*pitch+in;if(!dm)dti->offset+=pitch+2*out;sg.range.address=sti->address+sti->offset;dg.range.address=dti->address+dti->offset;sg=attach_banked_tile(sg,*sti);dg=attach_banked_tile(dg,*dti);}else{if(!sm)sg.range.address+=2*sg.row_pitch+in;if(!dm)dg.range.address+=dg.row_pitch+2*out;}
			scaled_image_plan plan{sg,dg,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,5,4,7,5,1,1,bool(flags&1),bool(flags&2),bool(flags&4),sti,dti};
            if(layout>=7)
            {
                const auto crop=[](std::span<const guest_storage_segment> parts,std::size_t begin,std::size_t size)
                {
                    std::vector<guest_storage_segment> result;
                    for(const auto& p:parts){const auto a=std::max(begin,p.offset),b=std::min(begin+size,p.offset+p.size);if(a<b)result.push_back({p.address+unsigned(a-p.offset)+0x100000,a-begin,b-a});}
                    return result;
                };
                plan.source_segments=crop(sd.segments,sm?0:2*sd.row_pitch+in,sg.range.size);
                plan.destination_segments=crop(dd.segments,dm?0:dd.row_pitch+2*out,dg.range.size);
                plan.source.range.address=plan.source_segments.front().address;
                plan.destination.range.address=plan.destination_segments.front().address;
            }
			const auto packed_source=plan_guest_surface(0,3,2,in),packed_destination=plan_guest_surface(0,4,3,out);std::vector<std::byte> cropped(6*in),old(12*out);for(unsigned y=0;y<2;++y)for(unsigned x=0;x<3;++x)std::copy_n(input.begin()+((y+2)*8+x+1)*in,in,cropped.begin()+(y*3+x)*in);for(unsigned y=0;y<3;++y)for(unsigned x=0;x<4;++x)std::copy_n(background.begin()+((y+1)*8+x+2)*out,out,old.begin()+(y*4+x)*out);auto canonical=plan;canonical.source=packed_source;canonical.destination=packed_destination;canonical.source_tile.reset();canonical.destination_tile.reset();const auto patch=reference(canonical,cropped,old);auto expected=background;for(unsigned y=0;y<3;++y)for(unsigned x=0;x<4;++x)std::copy_n(patch.begin()+(y*4+x)*out,out,expected.begin()+((y+1)*8+x+2)*out);
			const auto vm=memory->access->copy();const auto reads=memory->reads,writes=memory->writes,notifications=memory->access->notifications;const auto before=surfaces.transaction_statistics();memory->forbid_pixels=true;transfer.execute(plan);memory->forbid_pixels=false;require(transfer.native_completed()==1&&transfer.vm_completed()==0&&!transfer.has_pending(),"Suitable scaled image bypassed native route");require((ram_background?memory->reads>reads:memory->reads==reads)&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Native scale read target/wrote guest allocation");
            const auto created=[&](bool output)
            {
                const auto& image=output?plan.destination:plan.source;const auto& tile=output?dti:sti;const auto& segments=output?plan.destination_segments:plan.source_segments;
                auto address=image.range.address;if(tile){if(segments.empty())address=tile->address+tile->offset;else{const auto part=std::find_if(segments.begin(),segments.end(),[&](const auto& p){return tile->offset>=p.offset&&tile->offset-p.offset<p.size;});require(part!=segments.end(),"Native endpoint oracle missed IO pixel segment");address=part->address+unsigned(tile->offset-part->offset);}}
                return surfaces.acquire({address,{image.width,image.height,1,output?df:sf,false},image.morton?0:image.row_pitch,image.morton,tile,segments});
            };
            if(ram){source=created(false);require(surfaces.status(source).initialized&&!surfaces.status(source).dirty&&memory->reads-reads==surfaces.status(source).guest_revisions.size(),"RAM input did not use exactly one coherent snapshot per source watch");}
            if(cold)
            {
                destination=created(true);
                require(surfaces.status(destination).initialized,"Cold native output did not publish the requested destination authority");
            }
            const unsigned prefix=ram?1u:deferred?(deferred>=4?2u:1u):0u;
            require(surfaces.status(source).dirty==!ram&&surfaces.status(destination).dirty&&surfaces.transaction_statistics().submitted==before.submitted+1+prefix,"Native scale lost authority oraddedsource submissions");
            if(deferred)
            {
                require(surfaces.status(source).initialized&&!surfaces.status(source).deferred_copies,"Deferred source didnotmaterialize natively");
                if(ram_background)require(memory->reads-reads==surfaces.status(source).guest_revisions.size(),"Deferred RAM background was captured morethanonce perwatch");
                const auto data=make_buffer(ctx,input.size(),buffer_storage::shared);
                surfaces.draw(std::array{cached_surface_use{source,false}},[&](context& commands,auto images)
                {
                    id<MTLTexture> image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;image=make_texture(commands,d);samples.expand(commands,images[0],image);}texture_region r;r.size=MTLSizeMake(8,8,1);colors.download(commands,sf,image,r,data,{});
                });
                require(!std::memcmp(data.contents,input.data(),input.size()),"Deferred native source lost complete union/background/captured donor epoch");source_bytes_checked+=input.size();
            }
			const auto& native_expected=cold?patch:expected;const auto raw=make_buffer(ctx,native_expected.size(),buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{destination,false}},[&](context& commands,auto images)
			{
				id<MTLTexture> image=images[0];if(image.sampleCount>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;image=make_texture(commands,d);samples.expand(commands,images[0],image);}texture_region r;r.size=cold?MTLSizeMake(4,3,1):MTLSizeMake(8,8,1);colors.download(commands,df,image,r,raw,{});
			 });if(std::memcmp(raw.contents,native_expected.data(),native_expected.size()))throw std::runtime_error("Native scale bytes differ profile="+std::to_string(profiles)+" input="+std::to_string(in)+" output="+std::to_string(out)+" layout="+std::to_string(layout)+" samples="+std::to_string(count)+" flags="+std::to_string(flags));bytes_checked+=native_expected.size();
            if(cold)
            {
                auto expected_vm=vm;
                for(unsigned y=0;y<3;++y)for(unsigned x=0;x<4;++x)for(unsigned lane=0;lane<out;++lane)
                {
                    auto at=offset(plan.destination,x,y)+lane;
                    if(dti)at=banked_display_storage_offset(*dti,dti->offset+at);
                    std::uint64_t address=std::uint64_t(plan.destination.range.address)+at;
                    if(!plan.destination_segments.empty())
                    {
                        const auto part=std::find_if(plan.destination_segments.begin(),plan.destination_segments.end(),[&](const auto& p){return at>=p.offset&&at-p.offset<p.size;});
                        require(part!=plan.destination_segments.end(),"Cold native target oracle missed a real IO segment");address=std::uint64_t(part->address)+at-part->offset;
                    }
                    expected_vm[address%expected_vm.size()]=patch[(y*4+x)*out+lane];
                }
                if(plan.destination_segments.empty())surfaces.prepare_cpu_read(plan.destination.range);
                else for(const auto& part:plan.destination_segments)surfaces.prepare_cpu_read({part.address,part.size});
                require(memory->access->copy()==expected_vm&&surfaces.status(source).dirty==!ram&&!surfaces.status(destination).dirty,"Cold native output CPU publication changed source,padding/Morton/tile exterior or IO gaps");
            }
            ++profiles;
		}
	}
	std::cout<<(deferred?(cold?"Deferred to cold native scaled images:":"Deferred to existing native scaled images:"):(ram?(cold?"RAM to cold native scaled images:":"RAM to existing native scaled images:"):(cold?"Cold native scaled images:":"Native scaled images:")))<<profiles<<" deferredMode="<<deferred<<" RGB565/ARGB8 format/layout/sample/flip/filter profiles,"<<bytes_checked<<" exactguest-word/native-background bytes;cropped oddpitch/Morton/banked256-768/disjoint split-pixel IO physical-mirror storage,"<<(ram_background?"one coherent background/source snapshot per watch,zero target capture/VMwrite/notifications":"1/2/4 individual input samples,zero VMpixel reads/writes/notifications")<<",one native write submission andunchanged wholeVM passed\n";
    if(deferred)std::cout<<"Deferred complete native source:"<<source_bytes_checked<<" exact captured-source/background bytes,mode="<<deferred<<" passed\n";
    if(cold)std::cout<<"Cold native output CPU observation:all target active bytes correct;wholeVM outside output,source pixels,pitch padding,Morton/tile exterior andIO gaps unchanged across"<<profiles<<"profiles passed\n";
}
void native_scaled_image_transactions(bool cold=false,bool ram=false)
{
    using namespace std::chrono_literals;
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);unsigned cases=0;
    for(unsigned mode=0;mode<18;++mode)
    {
        if(((cold||ram)&&mode<2)||(!cold&&mode==17))continue;
        @autoreleasepool
        {
            auto memory=std::make_shared<native_memory>();auto refuse=std::make_shared<bool>(false);
            const cpu_capture_dispatch capture=[access=memory->access,refuse](const auto& work)
            {if(*refuse)throw std::range_error("Native scale CPU dispatcher refused completion");access->execute([&](auto&){work();});};
            auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned address,unsigned,std::size_t){return address;},1u<<20,ledger,64,{capture,io});
            auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto source=mode==17||ram?surfaces.acquire({0x1000+2*33+4,{3,2,1,surface_color::argb8,false},33,{}}):surfaces.acquire({0x1000,{8,8,1,surface_color::argb8,false},33,{}});
            const auto destination=mode<2?source:cold?surfaces.acquire({(mode==17?0x1000u:0x6000u)+33+8,{4,3,1,surface_color::argb8,false},33,{}}):surfaces.acquire({0x6000,{8,8,1,surface_color::argb8,false},33,{}});
            std::vector<std::byte> input(256),background(256,cold?std::byte{0xcd}:std::byte{0xa5});for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+i/5+73)&255);if(mode<2||mode==17)background=input;
            const auto seed=[&](const cached_surface_ref& surface,const std::vector<std::byte>& bytes)
            {surfaces.prepare(surface,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images){texture_region region;region.size=MTLSizeMake(8,8,1);colors.upload(commands,surface_color::argb8,make_upload(commands,bytes),{},images[0],region);});};
            if(ram||mode==16||mode==17){std::vector<std::byte> vm_source(264,std::byte{0xcd});for(unsigned y=0;y<8;++y)std::copy_n(input.begin()+y*32,32,vm_source.begin()+y*33);memory->raw->write({0x1000,264},vm_source);if(!ram)surfaces.prepare(source);}else seed(source,input);if(!cold&&source!=destination)seed(destination,background);
            const auto readback=[&](const cached_surface_ref& surface)
            {const bool partial=cold&&surface==destination;const auto size=partial?48u:256u;const auto raw=make_buffer(ctx,size,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{surface,false}},[&](context& commands,auto images){texture_region region;region.size=partial?MTLSizeMake(4,3,1):MTLSizeMake(8,8,1);colors.download(commands,surface_color::argb8,images[0],region,raw,{});});const auto bytes=map_buffer(raw,0,size);return std::vector<std::byte>(bytes.begin(),bytes.end());};
            const unsigned destination_address=(mode<2||mode==17?0x1000:0x6000)+(mode==1?0x100000:0);
            scaled_image_plan plan{plan_guest_surface(0x1000+2*33+4,3,2,4,33),plan_guest_surface(destination_address+33+8,4,3,4,33),scaled_image_format::argb8,scaled_image_format::argb8,5,4,7,5,1,1,true,false,true};
            auto canonical=plan;canonical.source=plan_guest_surface(0,3,2,4);canonical.destination=plan_guest_surface(0,4,3,4);
            std::vector<std::byte> cropped(24),old(48);for(unsigned y=0;y<2;++y)std::copy_n(input.begin()+((y+2)*8+1)*4,12,cropped.begin()+y*12);for(unsigned y=0;y<3;++y)std::copy_n(background.begin()+((y+1)*8+2)*4,16,old.begin()+y*16);
            const auto patch=reference(canonical,cropped,old);auto expected=background;for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*16,16,expected.begin()+((y+1)*8+2)*4);
            const auto& native_expected=cold?patch:expected;const auto old_native=cold?std::vector<std::byte>(48,std::byte{0xcd}):background;
            const unsigned padding_address=cold?plan.destination.range.address+16:0x6000+32;
            fragment_texture_registers words{};std::optional<versioned_guest_texture> old_image;
            if(mode==11){words.offset=cold?plan.destination.range.address:0x6000;words.format=(1u<<16)|(0xa5u<<8)|(2u<<4)|2u|8u;words.control0=1u<<31;words.control1=0xaae4;words.image_rect=cold?((4u<<16)|3u):((8u<<16)|8u);words.control3=(1u<<20)|33u;old_image=textures.resolve_versioned(words,surfaces);}
            const auto before=surfaces.status(destination);const auto vm=memory->access->copy();
            id<MTLBuffer> blocker=nil;unsigned pressure_callbacks=0;std::shared_ptr<void> pressure;
            if(mode==16)pressure=ledger->register_reclaimer([&](memory_pressure_request){++pressure_callbacks;require(!surfaces.reclaim_clean()&&surfaces.status(source).initialized,"Native transfer pressure reclaimed its prepared clean input");});
            if(mode==12){(void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();const auto state=ledger->statistics();blocker=ledger->shared_buffer(state.budget-state.reserved_bytes);}
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
            std::optional<submission> token;auto validations=std::make_shared<unsigned>(0);
            auto mappings=std::make_shared<scaled_image_mapping_epoch::mappings>(scaled_image_mapping_epoch::mappings{{{0x1000,264,{}},{0x6000,264,{}}}});
            auto versions=std::make_shared<scaled_image_mapping_versions>(1000,1,1);std::optional<scaled_image_mapping_epoch> epoch;
            capture([&]{epoch.emplace([mappings,access=memory->access]{require(guest_memory_epoch::current(*access)!=nullptr,"Native scale mapping query escaped CPU epoch");return *mappings;},[versions]{return *versions;});});
            const auto validate=[owned=*epoch,validations,access=memory->access]{require(guest_memory_epoch::current(*access)!=nullptr,"Native scale metadata check escaped CPU epoch");++*validations;owned.validate();};
            bool failed=false;
            if(ram)memory->allowed_input_reads={plan.source.range};
            memory->forbid_pixels=true;
            const auto execute=[&]
            {
            try
            {
                transfer->execute(plan,mode<2||mode==12||mode==16||mode==17?30s:1ms,[&](context& commands)
                {
                    token=commands.recording_submission();
                    if(mode==4)throw std::range_error("Native scale encode rejected");
                    if(mode==16){const auto state=ledger->statistics();blocker=ledger->shared_buffer(state.budget-state.reserved_bytes);}
                    if(mode>=2&&mode!=4&&mode!=12&&mode!=16&&mode!=17)commands.wait_event(gate,1);
                },validate);
            }
            catch(const std::runtime_error&){failed=true;}
            };
            execute();
            if(ram&&failed&&!token&&!transfer->has_pending()&&surfaces.status(source).pending)
            {
                // A source-upload timeout precedes the actual scaled writer.
                // Finish that known prefix before explicitly retrying execute.
                require(surfaces.status(destination).content_generation==before.content_generation&&memory->access->copy()==vm,"Pending input prefix published native output");
                surfaces.synchronize(30s);require(surfaces.status(source).initialized&&!surfaces.status(source).dirty,"RAM input prefix did not finish cleanly");failed=false;execute();
            }
            if(mode<2||mode==17)
            {
                require(!failed&&transfer->native_completed()==1&&!transfer->has_pending(),"Native self/new-target alias transfer did not complete");
                require(readback(destination)==native_expected&&memory->access->copy()==vm,"Native self/physical/new-target alias lost frozen source or changed VM");
                if(mode==17)
                {
                    require(!surfaces.status(source).initialized&&!surfaces.status(source).dirty&&surfaces.status(destination).dirty,"New larger target did not replace the smaller clean alias authority");
                    auto expected_vm=vm;for(unsigned y=0;y<3;++y)for(unsigned x=0;x<16;++x)expected_vm[plan.destination.range.address+y*33+x]=patch[y*16+x];
                    memory->forbid_pixels=false;surfaces.prepare_cpu_read({0x1000,264});
                    require(memory->access->copy()==expected_vm,"New overlapping target changed uncopied source/background bytes at CPU observation");
                }
            }
            else if(mode==12)
            {
                require(failed&&!transfer->has_pending()&&!surfaces.status(destination).pending&&surfaces.status(destination).content_generation==before.content_generation&&surfaces.status(source).dirty==!ram&&surfaces.status(destination).dirty==!cold,"Full shared ledger lost prior native authority or left pending work");
                blocker=nil;if(cold)require(!surfaces.status(destination).initialized&&memory->access->copy()==vm,"Budget refusal initialized a cold target");else require(readback(destination)==background,"Budget refusal changed native target");transfer->execute(plan,30s,{},validate);
                require(transfer->native_completed()==1&&readback(destination)==native_expected&&memory->access->copy()==vm,"Native scaled-image retry after budget refusal failed");
            }
            else if(mode==16)
            {
                require(failed&&pressure_callbacks==1&&!transfer->has_pending()&&!surfaces.status(destination).pending&&surfaces.status(destination).content_generation==before.content_generation&&surfaces.status(source).initialized&&!surfaces.status(source).dirty&&memory->access->copy()==vm,"Scratch budget refusal changed prepared input or published output");
                if(cold)require(!surfaces.status(destination).initialized&&!surfaces.status(destination).dirty,"Scratch allocation failure initialized a cold output");
                blocker=nil;transfer->execute(plan,30s,{},validate);
                require(transfer->native_completed()==1&&readback(destination)==native_expected&&memory->access->copy()==vm,"Native scratch quota refusal prevented a valid retry");
            }
            else if(mode==4)
            {
                require(failed&&!transfer->has_pending()&&!surfaces.status(destination).pending,"Native encode rejection left pending output");
                require((cold?(!surfaces.status(destination).initialized&&!surfaces.status(destination).dirty&&memory->access->copy()==vm):readback(destination)==background)&&surfaces.status(destination).content_generation==before.content_generation,"Native encode rejection changed native target");
            }
            else
            {
                require(failed&&transfer->has_pending()&&surfaces.status(destination).pending&&token&&!token->ready(),"Native timeout did not retain a real pending write");
                require(surfaces.status(destination).content_generation==before.content_generation&&memory->access->copy()==vm,"Pending native transfer published early");
                if(cold)require(!surfaces.status(destination).initialized&&!surfaces.status(destination).dirty,"Pending cold output published initialization early");
                if(mode==3)(*mappings)[1].address+=32;
                if(mode==13)++versions->io;
                if(mode==14)++versions->tile;
                if(mode==15)++versions->vm;
                if(mode==5)require(token->cancel_pending(),"Native queued scaled-image work could not be cancelled");
                if(mode==6)transfer.reset();
                if(mode==7||mode==8||mode==9)
                {
                    const unsigned address=mode==7?(ram?plan.source.range.address:0x1000):mode==8?(cold?plan.destination.range.address:0x6000):padding_address;
                    memory->raw->write({address,1},std::array{std::byte{0x5c}});
                }
                if(mode==10)*refuse=true;
                gate.signaledValue=1;failed=false;
                try {if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                if(mode==10)
                {
                    require(failed&&transfer->has_pending()&&surfaces.status(destination).pending,"CPU dispatcher refusal discarded native pending transaction");
                    *refuse=false;transfer->synchronize();failed=false;
                }
                if(mode==5)ctx.recover_completed_errors();
                const bool rejected=mode==3||mode==5||mode==6||mode==7||mode==8||(mode>=13&&mode<=15);
                require(failed==rejected&&(!transfer||!transfer->has_pending())&&!surfaces.status(destination).pending,"Native transaction completion/rejection state differs");
                if(rejected)
                {
                    require(surfaces.status(destination).content_generation==before.content_generation&&!surfaces.status(destination).faulted,"Rejected native output advanced generation or faulted valid old image");
                    if(cold)require(!surfaces.status(destination).initialized&&!surfaces.status(destination).dirty,"Rejected cold native transfer left an output authority");
                    else if(mode!=7&&mode!=8)require(readback(destination)==background,"Rejected native transfer lost target background");
                    if(mode==7||mode==8)require(memory->raw->read({mode==7?(ram?plan.source.range.address:0x1000u):(cold?plan.destination.range.address:0x6000u),1}).bytes[0]==std::byte{0x5c},"Rejected native transfer overwrote CPU writer");
                    else require(memory->access->copy()==vm,"Rejected native transfer changed guest VM");
                }
                else
                {
                    require(transfer->native_completed()==1&&readback(destination)==native_expected&&*validations>=3,"Successful native pending write lost pixels or metadata checks");
                    if(mode==9)require(memory->raw->read({padding_address,1}).bytes[0]==std::byte{0x5c},"Native transfer overwrote CPU padding store");
                    else require(memory->access->copy()==vm,"Native pending transfer wrote VM pixels");
                    if(mode==11)
                    {
                        bool old_current=true;capture([&]{old_current=old_image->revision.current();});require(old_current,"Native write invalidated a frozen texture VM/IO receipt");
                        const auto fresh=textures.resolve_versioned(words,surfaces);capture([&]{require(fresh.revision.current(),"Native write lost fresh texture receipt");});
                        const auto snapshot=[&](id<MTLTexture> texture){const auto size=cold?48u:256u;const auto raw=make_buffer(ctx,size,buffer_storage::shared);ctx.begin();texture_region area;area.size=cold?MTLSizeMake(4,3,1):MTLSizeMake(8,8,1);colors.download(ctx,surface_color::argb8,texture,area,raw,{});ctx.submit().wait();const auto bytes=map_buffer(raw,0,size);return std::vector<std::byte>(bytes.begin(),bytes.end());};
                        require(snapshot(old_image->image.texture)==old_native&&snapshot(fresh.image.texture)==native_expected&&fresh.image.texture!=old_image->image.texture,"Native scaled image changed retained texture or reused stale cache pixels");
                    }
                    if(cold&&mode==9){const auto at=padding_address-0x6000;expected[(at/33)*32+at%33]=std::byte{0x5c};}
                    memory->forbid_pixels=false;const unsigned observed_base=mode==17?0x1000:0x6000;surfaces.prepare_cpu_read({observed_base,264});const auto visible=memory->raw->read({observed_base,264}).bytes;
                    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<32;++x)require(visible[y*33+x]==expected[y*32+x],"CPU observation missed native scaled-image pixels/background");
                }
            }
            memory->forbid_pixels=false;++cases;
        }
    }
    std::cout<<(ram?(cold?"RAM to cold native transactions:":"RAM to existing native transactions:"):(cold?"Cold native scaled-image transactions:":"Native scaled-image transactions:"))<<cases<<((cold||ram)?" real ":" self/physical-alias frozen-source,real ")<<"timeout/success,metadata rejection,encode cancellation,queued cancellation,abandoned owner,source/target CPU-writer conflicts,padding survival,CPU dispatcher retry,retained old/fresh cache pixels,full shared-ledger refusal/retry,independent IO/tile/VM ABA rejection,protected clean-source scratch-allocation pressure/refusal/retry,"<<(cold?"new overlapping destination preserves a smaller clean source snapshot,":"")<<"and exact subsequent CPU visibility cases passed\n";
}

void deferred_scaled_image_transactions()
{
    using namespace std::chrono_literals;
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);unsigned cases=0;
    for(unsigned role=1;role<=3;++role)for(bool spill:{false,true})for(unsigned failure=0;failure<9;++failure)
    {
        @autoreleasepool
        {
            try
            {
            auto memory=std::make_shared<native_memory>();auto refuse=std::make_shared<bool>(false);
            const cpu_capture_dispatch capture=[access=memory->access,refuse](const auto& work){if(*refuse)throw std::range_error("Deferred publication dispatcher refused");access->execute([&](auto&){work();});};
            auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned address,unsigned,std::size_t){return address;},1u<<20,ledger,64,{capture,io});auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto source=surfaces.acquire({0x1000,{8,8,1,surface_color::argb8,false},33,{}}),target=surfaces.acquire({0x6000,{8,8,1,surface_color::argb8,false},33,{}});
            const auto first=surfaces.acquire({0xd0000,{8,8,1,surface_color::argb8,false},32,{}}),second=surfaces.acquire({0xd4000,{8,8,1,surface_color::argb8,false},32,{}});
            std::vector<std::byte> input(256),alternate(256),background(256,std::byte{0xa5});for(unsigned i=0;i<256;++i){input[i]=std::byte((i*37+i/5+73)&255);alternate[i]=std::byte((i*53+i/7+119)&255);}
            const auto seed=[&](const cached_surface_ref& surface,const std::vector<std::byte>& bytes){surfaces.prepare(surface,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images){texture_region r;r.size=MTLSizeMake(8,8,1);colors.upload(commands,surface_color::argb8,make_upload(commands,bytes),{},images[0],r);});};
            seed(first,input);seed(second,alternate);
            if(role&1){surfaces.defer_copy(first,{0,0,4,8},source,0,0);surfaces.defer_copy(second,{4,0,4,8},source,4,0);for(unsigned y=0;y<8;++y)std::copy_n(alternate.begin()+y*32+16,16,input.begin()+y*32+16);}else seed(source,input);
            seed(target,background);
            if(role&2){if(spill)surfaces.spill(target);surfaces.defer_copy(second,{0,0,2,2},target,0,0);for(unsigned y=0;y<2;++y)std::copy_n(alternate.begin()+y*32,8,background.begin()+y*32);}
            if(spill&&(role&1)){surfaces.spill(first);surfaces.clear(first,0xf0,0xff204060,0,255,{0,0,8,8});require(surfaces.status(source).deferred_copies==2,"Deferred transaction lost its captured donor epoch");}
            const auto before=surfaces.status(target);const auto vm=memory->access->copy();const auto expected_prefix_generation=before.content_generation+unsigned(bool(role&2));
            scaled_image_plan p{plan_guest_surface(0x1000+2*33+4,3,2,4,33),plan_guest_surface(0x106000+33+8,4,3,4,33),scaled_image_format::argb8,scaled_image_format::argb8,5,4,7,5,1,1,true,true,true};
            auto canonical=p;canonical.source=plan_guest_surface(0,3,2,4);canonical.destination=plan_guest_surface(0,4,3,4);std::vector<std::byte> cropped(24),old(48);for(unsigned y=0;y<2;++y)std::copy_n(input.begin()+((y+2)*8+1)*4,12,cropped.begin()+y*12);for(unsigned y=0;y<3;++y)std::copy_n(background.begin()+((y+1)*8+2)*4,16,old.begin()+y*16);const auto patch=reference(canonical,cropped,old);auto expected=background;for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*16,16,expected.begin()+((y+1)*8+2)*4);
            const auto readback=[&]{const auto data=make_buffer(ctx,256,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{target,false}},[&](context& commands,auto images){texture_region r;r.size=MTLSizeMake(8,8,1);colors.download(commands,surface_color::argb8,images[0],r,data,{});});const auto bytes=map_buffer(data,0,256);return std::vector<std::byte>(bytes.begin(),bytes.end());};
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};std::optional<submission> token;auto valid=std::make_shared<bool>(true);const auto validate=[valid,access=memory->access]{require(guest_memory_epoch::current(*access)!=nullptr,"Deferred metadata escaped CPU exclusion");if(!*valid)throw scaled_image_mapping_conflict();};
            std::vector<id<MTLBuffer>> blockers;memory->forbid_pixels=true;
            if(failure==7)
            {
                (void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();
                // Actual allocatedSize can round a last odd spill remainder up.
                // Fill both buffer storage classes,downshifting refusals,until
                // no private byte allocation fits the real remaining ledger.
                for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate})
                {
                    auto stats=ledger->statistics();auto attempt=stats.budget-stats.reserved_bytes;
                    while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));stats=ledger->statistics();attempt=stats.budget-stats.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}
                }
            }
            bool failed=false;
            const auto execute=[&]{try{transfer->execute(p,failure==7?30s:1ms,[&](context& commands){token=commands.recording_submission();if(failure==6)throw std::range_error("Deferred output encode refused");commands.wait_event(gate,1);},validate);}catch(const std::runtime_error&){failed=true;}};
            execute();
            for(unsigned retry=0;!token&&surfaces.status(source).pending&&retry<4;++retry){require(memory->access->copy()==vm,"Deferred prefix wrote guest bytes");surfaces.synchronize(30s);failed=false;execute();}
            if(failure==7)
            {
                require(failed&&(!token||(token->ready()&&token->cancelled()))&&!transfer->has_pending()&&!surfaces.status(target).pending&&memory->access->copy()==vm,"Deferred quota refusal published output orlost live ownership");
                blockers.clear();transfer->execute(p,30s,{},validate);require(transfer->native_completed()==1&&readback()==expected&&memory->access->copy()==vm,"Deferred quota refusal prevented native retry");
            }
            else
            {
                require(failed&&token&&!surfaces.status(source).deferred_copies&&!surfaces.status(target).deferred_copies&&surfaces.status(target).content_generation==expected_prefix_generation&&memory->access->copy()==vm,"Deferred native prefix/output ownership or generation differs");
                if(failure==6){require(!transfer->has_pending()&&!surfaces.status(target).pending&&readback()==background,"Deferred encode failure lost prior materialized background");}
                else
                {
                    require(transfer->has_pending()&&surfaces.status(target).pending&&!token->ready(),"Deferred output lacks a real gated submission");
                    if(failure==1||failure==2)memory->raw->write({failure==1?0x1000u:0x6000u,1},std::array{std::byte{0x5c}});
                    if(failure==3)*valid=false;if(failure==4)require(token->cancel_pending(),"Deferred queued output could not be cancelled");if(failure==5)transfer.reset();if(failure==8)*refuse=true;
                    gate.signaledValue=1;failed=false;try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                    if(failure==8){require(failed&&transfer->has_pending()&&surfaces.status(target).pending,"Deferred CPU dispatcher refusal lost pending output");*refuse=false;transfer->synchronize();failed=false;}
                    if(failure==4)ctx.recover_completed_errors();const bool rejected=failure>=1&&failure<=5;
                    require(failed==rejected&&(!transfer||!transfer->has_pending())&&!surfaces.status(target).pending&&!surfaces.status(target).faulted,"Deferred output completion/rejection metadata differs");
                    require(surfaces.status(target).content_generation==expected_prefix_generation+unsigned(!rejected),"Deferred rejection lost prior generation orpublished new output");
                    if(failure!=2)require(readback()==(rejected?background:expected),"Deferred output changed prior background orlost exact scaled pixels");
                    if(failure==1||failure==2)require(memory->raw->read({failure==1?0x1000u:0x6000u,1}).bytes[0]==std::byte{0x5c},"Deferred output overwrote CPU writer");else require(memory->access->copy()==vm,"Deferred output wrote VM before observation");
                }
            }
            memory->forbid_pixels=false;++cases;
            }
            catch(const std::exception& error){throw std::runtime_error("Deferred transaction role="+std::to_string(role)+" spill="+std::to_string(spill)+" failure="+std::to_string(failure)+": "+error.what());}
        }
    }
    std::cout<<"Deferred scaled-image transactions:"<<cases<<" source/target/both prefixes,full union/partial native-spilled target backgrounds,physical-mirror targets,real timeouts,CPU/metadata conflicts,encode/queued cancellation,abandoned output owner,quota refusal/retry andCPU dispatcher retry preserve prior prefix generations/backgrounds;zero VMpixel accesses andexact native outputs passed\n";
}

void ram_source_prefix_lifetime(bool deferred=false)
{
    using namespace std::chrono_literals;
    unsigned cases=0;
    for(bool cold:{false,true})for(unsigned outcome=0;outcome<3;++outcome)
    {
        std::shared_ptr<heap_budget> ledger;std::weak_ptr<native_access> provider;
        @autoreleasepool
        {
            auto ctx=std::make_unique<context>(2);ledger=std::make_shared<heap_budget>(*ctx,16u<<20);ctx->set_buffer_budget(ledger);
            auto memory=std::make_shared<native_memory>();provider=memory->access;
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};
            auto surfaces=std::make_unique<surface_cache>(*ctx,memory,8u<<20,ledger,capture);auto io=std::make_shared<io_mapping_epoch>();
            auto textures=std::make_unique<guest_texture_cache>(*ctx,memory,[](unsigned address,unsigned,std::size_t){return address;},1u<<20,ledger,64,texture_capture_bindings{capture,io});
            auto transfer=std::make_unique<scaled_image_transfer>(*ctx,*surfaces,*textures,memory,capture,1u<<20);
            const auto input=plan_guest_surface(0x1000,3,2,4,16),output=plan_guest_surface(0x6000,4,3,4,20);
            std::vector<std::byte> bytes(input.range.size,std::byte{0xcd});for(unsigned y=0;y<2;++y)for(unsigned x=0;x<12;++x)bytes[y*16+x]=std::byte((x*43+y*79+17)&255);if(!deferred)memory->raw->write(input.range,bytes);
            auto source=surfaces->acquire({0x1000,{3,2,1,surface_color::argb8,false},16,{}}),destination=surfaces->acquire({0x6000,{4,3,1,surface_color::argb8,false},20,{}});
            if(deferred)
            {
                const auto donor=surfaces->acquire({0xd0000,{3,2,1,surface_color::argb8,false},16,{}});surface_color_transfer colors(*ctx);
                surfaces->prepare(donor,surface_initialization::background);surfaces->draw(std::array{cached_surface_use{donor,true}},[&](context& commands,auto images){texture_region r;r.size=MTLSizeMake(3,2,1);colors.upload(commands,surface_color::argb8,make_upload(commands,bytes),{0,16},images[0],r);});
                surfaces->defer_copy(donor,{0,0,1,2},source,0,0);surfaces->defer_copy(donor,{1,0,2,2},source,1,0);
            }
            if(!cold)surfaces->clear(destination,0xf0,0xff556677,0,255,{0,0,4,3});
            scaled_image_plan plan{input,output,scaled_image_format::argb8,scaled_image_format::argb8,3,2,6,5,1,1,true,false,true};
            const auto expected=reference(plan,bytes,std::vector<std::byte>(output.range.size,std::byte{0xcd}));const auto before=surfaces->status(destination);const auto vm=memory->access->copy();
            const auto gate=[ctx->device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
            ctx->begin();ctx->wait_event(gate,1);const auto blocker=ctx->submit();bool dependency_called=false,timed_out=false;
            if(!deferred)memory->allowed_input_reads={input.range};memory->forbid_pixels=true;
            try{transfer->execute(plan,1ms,[&](context&){dependency_called=true;});}catch(const std::runtime_error&){timed_out=true;}
            require(timed_out&&!dependency_called&&!transfer->has_pending()&&surfaces->status(source).pending&&!surfaces->status(source).initialized&&!blocker.ready(),"RAM prefix fixture did not stop during real queued source upload");
            require(surfaces->status(destination).content_generation==before.content_generation&&surfaces->status(destination).initialized==before.initialized&&surfaces->status(destination).dirty==before.dirty&&memory->access->copy()==vm&&memory->writes==0,"RAM upload prefix published scaled destination/VM output early");
            if(outcome==2)
            {
                transfer.reset();textures.reset();surfaces.reset();source.reset();destination.reset();ctx->shutdown();ctx.reset();bool cancelled=false;try{blocker.wait();}catch(const std::runtime_error&){cancelled=true;}
                require(cancelled&&blocker.cancelled()&&memory->access->copy()==vm&&memory->writes==0&&!dependency_called,"RAM prefix shutdown lost cancellation or published destination");
            }
            else
            {
                if(outcome==1)transfer.reset();gate.signaledValue=1;blocker.wait();surfaces->synchronize(30s);
                require(surfaces->status(source).initialized&&surfaces->status(source).dirty==deferred&&!surfaces->status(source).pending&&memory->access->copy()==vm&&memory->reads==unsigned(!deferred)&&memory->writes==0&&!dependency_called,"RAM source prefix completion did not retain one clean snapshot");
                require(surfaces->status(destination).content_generation==before.content_generation&&surfaces->status(destination).initialized==before.initialized&&surfaces->status(destination).dirty==before.dirty,"RAM input prefix completion orowner abandonment published scaled destination");
                if(outcome==0)
                {
                    transfer->execute(plan);require(transfer->native_completed()==1&&transfer->vm_completed()==0&&memory->reads==unsigned(!deferred)&&memory->writes==0&&memory->access->copy()==vm,"Explicit native retry recaptured source or wrote destination VM");
                    memory->forbid_pixels=false;surfaces->prepare_cpu_read(output.range);
                    require(memory->raw->read_surface(output).bytes==expected,"RAM prefix retry CPU observation differs");
                }
            }
            ++cases;
        }
        if(!provider.expired()||ledger->statistics().reserved_bytes!=0)throw std::runtime_error("RAM prefix teardown retained provider/native charges cold="+std::to_string(cold)+" outcome="+std::to_string(outcome)+" providerExpired="+std::to_string(provider.expired())+" bytes="+std::to_string(ledger->statistics().reserved_bytes));
    }
    std::cout<<(deferred?"Deferred source materialization prefix:":"RAM source upload prefix:")<<cases<<" real queued input timeouts with existing/cold targets;one captured input epoch,explicit finished-prefix retry,transfer-owner abandonment andblocked cache/context shutdown never publish early output;provider retirement/zero ledger charges passed\n";
}

void rational_filter_boundaries()
{
    context ctx(2);scaled_image_compute compute(ctx,1u<<20);unsigned cases=0;
    // Explicit golden half-way channel:23.5 green must become 24,not 23.
    const auto two=plan_guest_surface(0,1,2,2),one=plan_guest_surface(0,1,1,2);
    const std::array<std::byte,4> half{std::byte{0x02},std::byte{0xe0},std::byte{0x03},std::byte{0x00}};
    const auto golden=make_buffer(ctx,2,buffer_storage::shared);ctx.begin();compute.encode(ctx,{two,one,scaled_image_format::rgb565,scaled_image_format::rgb565,1,2,1,1,0,0,false,false,true},make_upload(ctx,half),golden);ctx.submit().wait();
    require(static_cast<const unsigned char*>(golden.contents)[0]==0x03&&static_cast<const unsigned char*>(golden.contents)[1]==0,"565 half-channel golden word was rounded down");
    for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned iw:{1u,7u,65536u})for(unsigned ih:{1u,65536u})for(unsigned ow:{1u,9u,65536u})for(unsigned oh:{1u,65536u})for(unsigned flags=0;flags<8;++flags)for(unsigned edge=0;edge<2;++edge)
    {
        @autoreleasepool
        {
            const auto source=plan_guest_surface(0,std::min(iw,3u),std::min(ih,2u),in),destination=plan_guest_surface(0,std::min(ow,4u),std::min(oh,3u),out);
            if(edge&&ow==destination.width&&oh==destination.height)continue;
            scaled_image_plan p{source,destination,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,iw,ih,ow,oh,edge?unsigned(ow-destination.width):0,edge?unsigned(oh-destination.height):0,bool(flags&1),bool(flags&2),bool(flags&4)};
            std::vector<std::byte> bytes(source.range.size);for(unsigned i=0;i<bytes.size();++i)bytes[i]=std::byte((i*61+unsigned(i%3==0)*255)&255);
            const auto expected=reference(p,bytes,std::vector<std::byte>(destination.range.size));const auto input=make_upload(ctx,bytes);const auto output=make_buffer(ctx,expected.size(),buffer_storage::shared);
            ctx.begin();compute.encode(ctx,p,input,output);ctx.submit().wait();
            require(std::memcmp(output.contents,expected.data(),expected.size())==0,"Rational filter boundary differs at clipped/reversed 1..65536 domain");++cases;
        }
    }
    std::cout<<"Exact rational filter:"<<cases<<" cross-format nearest/linear/reversed/clipped edge profiles with 1..65536 domains,negative half-pixel floor,2D maximum denominator and explicit 565 green23.5->24 golden word passed\n";
}

} // namespace
int main(int argc,char** argv)
{
	@autoreleasepool
	{
		try
		{
            if(argc>1){if(argc==2&&std::string_view(argv[1])=="deferred-transactions"){deferred_scaled_image_transactions();return 0;}throw std::invalid_argument("Unknown scaled-image test selection");}
			rational_filter_boundaries();
            native_scaled_image_cases();
            native_scaled_image_cases(true);
            native_scaled_image_cases(false,true);
            native_scaled_image_cases(true,true);
            for(unsigned mode=1;mode<=5;++mode){native_scaled_image_cases(false,false,mode);native_scaled_image_cases(true,false,mode);}
            native_scaled_image_transactions();
            native_scaled_image_transactions(true);
            native_scaled_image_transactions(false,true);
            native_scaled_image_transactions(true,true);
            deferred_scaled_image_transactions();
            ram_source_prefix_lifetime();
            ram_source_prefix_lifetime(true);
			context ctx{1};
			auto provider = std::make_shared<fixture_access>();
			auto memory = std::make_shared<guarded_guest_memory>(provider);
			auto budget = std::make_shared<heap_budget>(ctx, 64 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, memory, 64 << 20, budget);
			guest_texture_cache textures(ctx, memory, [](unsigned address, unsigned, std::size_t)
										 { return address; }, 1 << 20, budget);
			unsigned captures = 0;
			const cpu_capture_dispatch capture = [&](const auto& op)
			{++captures;provider->execute([&](auto&){op();}); };
			scaled_image_transfer transfer(ctx, surfaces, textures, memory, capture, 1 << 20);
			unsigned cases = 0;
			for (unsigned in : {2u, 4u})
				for (unsigned out : {2u, 4u})
					for (unsigned flags = 0; flags < 8; ++flags)
						for (unsigned morton = 0; morton < 4; ++morton)
						{
							const auto src = plan_guest_surface(0x1000, 3, 2, in, (morton & 1) ? 0 : in * 5,
																(morton & 1) ? std::optional<swizzle_region>{{{8, 4, 1}, {1, 1, 0}, {3, 2, 1}}} : std::nullopt);
							const auto dst = plan_guest_surface(0x4000, 4, 3, out, (morton & 2) ? 0 : out * 7,
																(morton & 2) ? std::optional<swizzle_region>{{{8, 8, 1}, {2, 3, 0}, {4, 3, 1}}} : std::nullopt);
							const scaled_image_plan plan{src, dst, in == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8,
														 out == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8, 3, 2, 7, 5, 1, 1, bool(flags & 1), bool(flags & 2), bool(flags & 4)};
							std::vector<std::byte> input(src.range.size), output(dst.range.size, std::byte{0xa5});
							for (unsigned i = 0; i < input.size(); ++i)
								input[i] = std::byte((i * 37 + 73) % 256);
							memory->write(src.range, input);
							memory->write(dst.range, output);
							const auto expected = reference(plan, input, output);
							const auto before = captures;
							transfer.execute(plan);
							const auto actual = memory->read_surface(dst).bytes;
							if (actual != expected)
								throw std::runtime_error("Native scaled-image pixel/padding differs at case " + std::to_string(cases));
							require(captures == before + 2 && !transfer.has_pending(), "Transfer lost separate CPU capture/commit epochs");
							++cases;
						}
			scaled_image_compute compute(ctx, 1 << 20);
			for (unsigned width = 1; width <= 17; ++width)
				for (unsigned out = 1; out <= 17; ++out)
					for (bool flip : {false, true})
					{
						const auto src = plan_guest_surface(0x1000, width, 1, 4), dst = plan_guest_surface(0x4000, out, 1, 4);
						const scaled_image_plan boundary{src, dst, scaled_image_format::argb8, scaled_image_format::argb8, width, 1, out, 1, 0, 0, flip};
						std::vector<std::byte> bytes(src.range.size);
						for (unsigned i = 0; i < bytes.size(); ++i)
							bytes[i] = std::byte(i);
						const auto input_buffer = make_upload(ctx, bytes), output_buffer = make_buffer(ctx, dst.range.size, buffer_storage::shared);
						ctx.begin();
						compute.encode(ctx, boundary, input_buffer, output_buffer);
						ctx.submit().wait();
						const auto actual = map_buffer(output_buffer, 0, dst.range.size);
						const auto expected = reference(boundary, bytes, std::vector<std::byte>(dst.range.size));
						require(std::equal(actual.begin(), actual.end(), expected.begin()), "Scaled nearest rational boundary differs");
					}
			const auto packed = plan_guest_surface(0x1000, 65536, 1, 2), expanded = plan_guest_surface(0x40000, 65536, 1, 4);
			std::vector<std::byte> words565(packed.range.size);
			for (unsigned i = 0; i < 65536; ++i)
			{
				words565[i * 2] = std::byte(i >> 8);
				words565[i * 2 + 1] = std::byte(i);
			}
			const auto raw = make_upload(ctx, words565), rgba_buffer = make_buffer(ctx, expanded.range.size, buffer_storage::shared), roundtrip = make_buffer(ctx, packed.range.size, buffer_storage::shared);
			ctx.begin();
			compute.encode(ctx, {packed, expanded, scaled_image_format::rgb565, scaled_image_format::argb8, 65536, 1, 65536, 1}, raw, rgba_buffer);
			ctx.submit().wait();
			const auto colors = map_buffer(rgba_buffer, 0, expanded.range.size);
			for (unsigned i = 0; i < 65536; ++i)
			{
				const std::array expected{255u, unsigned(std::lround(((i >> 11) & 31) * 255. / 31)), unsigned(std::lround(((i >> 5) & 63) * 255. / 63)), unsigned(std::lround((i & 31) * 255. / 31))};
				for (unsigned c = 0; c < 4; ++c)
					require(colors[i * 4 + c] == std::byte(expected[c]), "RGB565 full-domain expansion differs");
			}
			ctx.begin();
			compute.encode(ctx, {expanded, packed, scaled_image_format::argb8, scaled_image_format::rgb565, 65536, 1, 65536, 1}, rgba_buffer, roundtrip);
			ctx.submit().wait();
			const auto restored = map_buffer(roundtrip, 0, packed.range.size);
			require(std::equal(restored.begin(), restored.end(), words565.begin()), "RGB565 full-domain roundtrip differs");
			const auto source = plan_guest_surface(0x1000, 4, 2, 4, 16), target = plan_guest_surface(0x4000, 2, 2, 4, 16);
			const scaled_image_plan plan{source, target, scaled_image_format::argb8, scaled_image_format::argb8, 4, 2, 2, 2};
			std::vector<std::byte> input(source.range.size), output(target.range.size, std::byte{0xa5});
			for (unsigned i = 0; i < input.size(); ++i)
				input[i] = std::byte(i + 1);
			memory->write(source.range, input);
			memory->write(target.range, output);
			const auto expected = reference(plan, input, output);
			const auto gate = [ctx.device() newSharedEvent];
			struct release_gate
			{
				id<MTLSharedEvent> event;
				std::uint64_t value = 10;
				~release_gate() { event.signaledValue = value; }
			} cleanup{gate};
			const auto start_pending = [&](std::uint64_t value)
			{
				bool timed_out = false;
				try
				{
					transfer.execute(plan, std::chrono::milliseconds(1), [&](context& commands)
									 { commands.wait_event(gate, value); });
				}
				catch (const std::runtime_error&)
				{
					timed_out = true;
				}
				require(timed_out && transfer.has_pending(), "Transfer timeout did not retain GPU transaction");
				require(memory->read_surface(target).bytes == output, "Pending transfer published destination early");
			};
			start_pending(1);
			memory->write({target.range.address + 8, 1}, std::array{std::byte{0x5c}});
			memory->write({source.range.address, 1}, std::array{std::byte{0xee}});
			gate.signaledValue = 1;
			transfer.synchronize();
			auto preserved = expected;
			preserved[8] = std::byte{0x5c};
			require(memory->read_surface(target).bytes == preserved, "Snapshot source or concurrent target padding was overwritten");
			memory->write(source.range, input);
			memory->write(target.range, output);
			start_pending(2);
			memory->write({target.range.address, 1}, std::array{std::byte{0x55}});
			gate.signaledValue = 2;
			bool conflict = false;
			try
			{
				transfer.synchronize();
			}
			catch (const texture_revision_conflict&)
			{
				conflict = true;
			}
			auto changed = output;
			changed[0] = std::byte{0x55};
			require(conflict && !transfer.has_pending() && memory->read_surface(target).bytes == changed, "Conflicting target store was overwritten");
			transfer.execute(plan);
			require(memory->read_surface(target).bytes == expected, "Transfer retry failed after target conflict");
			bool encode_failed = false;
			memory->write(target.range, output);
			try
			{
				transfer.execute(plan, std::chrono::seconds(30), [](auto&)
								 { throw std::range_error("dependency refusal"); });
			}
			catch (const std::range_error&)
			{
				encode_failed = true;
			}
			require(encode_failed && !transfer.has_pending() && memory->read_surface(target).bytes == output, "Encoding failure published VM data");
			transfer.execute(plan);
			const auto before_invalid_timeout = captures;
			bool invalid_timeout = false;
			try
			{
				transfer.execute(plan, std::chrono::milliseconds(-1));
			}
			catch (const std::invalid_argument&)
			{
				invalid_timeout = true;
			}
			require(invalid_timeout && captures == before_invalid_timeout, "Invalid wait timeout entered CPU capture");
			// Cached immutable reads of the target must be invalidated while
			// independently retained old images preserve their original pixels.
			const auto cached_target = plan_guest_surface(0x6000, 2, 2, 4, 16);
			memory->write(cached_target.range, std::vector<std::byte>(cached_target.range.size));
			fragment_texture_registers words{};
			words.offset = 0x6000;
			words.format = (1u << 16) | (0xa5u << 8) | (2u << 4) | 2u | 8u;
			words.control0 = 1u << 31;
			words.control1 = 0xaae4;
			words.image_rect = (2u << 16) | 2u;
			words.control3 = (1u << 20) | 16u;
			const auto old = textures.resolve(words);
			const auto invalidations = textures.statistics().invalidations;
			const auto native_target = surfaces.acquire({0x6000, {2, 2, 1, surface_color::argb8, false}, 16, {}});
			surfaces.prepare(native_target);
			transfer.execute({source, cached_target, scaled_image_format::argb8, scaled_image_format::argb8, 4, 2, 2, 2});
			require(textures.statistics().invalidations > invalidations && !surfaces.status(native_target).initialized,
					"Native scaled-image failed to invalidate texture/surface target mirrors");
			const auto retained = make_buffer(ctx, 16, buffer_storage::shared);
			ctx.begin();
			texture_to_buffer(ctx, old.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(2, 2, 1)}, retained, 0);
			ctx.submit().wait();
			for (const auto byte : map_buffer(retained, 0, 16))
				require(byte == std::byte{0}, "Scaled-image mutated a retained old native texture");
			// Source/destination physical mirrors overlap:must use the captured
			// source version,as for memmove,even though GPU buffers are disjoint.
			const auto overlap_src = plan_guest_surface(0x2000, 4, 1, 4, 16), overlap_dst = plan_guest_surface(0x12004, 4, 1, 4, 16);
			std::vector<std::byte> overlap(20);
			for (unsigned i = 0; i < 20; ++i)
				overlap[i] = std::byte(i + 17);
			memory->write({0x2000, 20}, overlap);
			transfer.execute({overlap_src, overlap_dst, scaled_image_format::argb8, scaled_image_format::argb8, 4, 1, 4, 1});
			auto moved = overlap;
			std::copy_n(overlap.begin(), 16, moved.begin() + 4);
			require(memory->read({0x2000, 20}).bytes == moved, "Physical overlap did not preserve captured source");
			// GPU-dirty guest storage must flush before the common CPU epoch.
			const auto dirty = surfaces.acquire({0x3000, {4, 1, 1, surface_color::argb8, false}, 16, {}});
			surfaces.clear(dirty, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 1});
			transfer.execute({plan_guest_surface(0x3000, 4, 1, 4), plan_guest_surface(0x5000, 4, 1, 4), scaled_image_format::argb8, scaled_image_format::argb8, 4, 1, 4, 1});
			const auto dirty_pixels = memory->read({0x5000, 16}).bytes;
			for (unsigned i = 0; i < 16; ++i)
				require(dirty_pixels[i] == std::byte(std::array{255u, 18u, 52u, 86u}[i % 4]), "Scaled transfer missed dirty native source");
			// Destroy a pending transfer:GPU resources remain completion-owned,
			// but no callback may publish destination after its owner disappears.
			memory->write(target.range, output);
			{
				scaled_image_transfer abandoned(ctx, surfaces, textures, memory, capture, 1 << 20);
				bool timed_out = false;
				try
				{
					abandoned.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
									  { commands.wait_event(gate, 3); });
				}
				catch (const std::runtime_error&)
				{
					timed_out = true;
				}
				require(timed_out && abandoned.has_pending(), "Abandon fixture lacks pending work");
			}
			gate.signaledValue = 3;
			ctx.wait_idle();
			require(memory->read_surface(target).bytes == output, "Destroyed transfer published late VM data");
			// Metadata has an independent lifetime from VM byte revisions. Real
			// pending work must reject a changed IO/tile assignment at publication.
			const scaled_image_mapping_epoch::mappings original_mappings{{{source.range.address, source.range.size, scaled_image_tile_mapping{0x10000, 65536, 768, 2, 1, 0x20000, 0, 0}},
																		  {target.range.address, target.range.size, scaled_image_tile_mapping{0x30000, 65536, 768, 1, 1, 0x40000, 0, 0}}}};
			const auto metadata_gate = [ctx.device() newSharedEvent];
			release_gate metadata_cleanup{metadata_gate, 1000};
			unsigned metadata_checks = 0;
			const auto validator = [&](const auto& current)
			{
				return scaled_image_mapping_epoch([current, &metadata_checks, provider]
												  {
					require(guest_memory_epoch::current(*provider) != nullptr, "Metadata validated outside CPU exclusion");
					++metadata_checks;
					return *current; });
			};
			// Capturing a stale mapping fails before any upload/publication.
			{
				const auto dirty_target = surfaces.acquire({target.range.address, {2, 2, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(dirty_target, 0xf0, 0xff765432, 0, 255, {0, 0, 2, 2});
				auto current = std::make_shared<scaled_image_mapping_epoch::mappings>(original_mappings);
				std::optional<scaled_image_mapping_epoch> epoch;
				capture([&]
						{ epoch.emplace(validator(current)); });
				(*current)[1].tile.reset();
				const auto notifications = provider->notifications;
				bool refused = false;
				try
				{
					transfer.execute(plan, std::chrono::seconds(30), {}, [owned = std::move(*epoch)]
									 { owned.validate(); });
				}
				catch (const scaled_image_mapping_conflict&)
				{
					refused = true;
				}
				require(refused && !transfer.has_pending() && provider->notifications == notifications && surfaces.status(dirty_target).dirty && memory->read_surface(target).bytes == output,
						"Stale mapping entered publication");
				// Deliberate cleanup with the original mapping restored;the refused
				// transfer itself must not flush this dirty native image to VM.
				surfaces.prepare_cpu_read(target.range);
			}
			unsigned mapping_cases = 0;
			for (unsigned side = 0; side < 2; ++side)
				for (unsigned field = 0; field < 11; ++field)
				{
					memory->write(target.range, output);
					auto current = std::make_shared<scaled_image_mapping_epoch::mappings>(original_mappings);
					const std::weak_ptr<scaled_image_mapping_epoch::mappings> lifetime = current;
					const auto notifications = provider->notifications;
					const auto signal = ++mapping_cases;
					{
						std::optional<scaled_image_mapping_epoch> epoch;
						capture([&]
								{ epoch.emplace(validator(current)); });
						bool timed_out = false;
						try
						{
							transfer.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
											 { commands.wait_event(metadata_gate, signal); }, [owned = std::move(*epoch)]
											 { owned.validate(); });
						}
						catch (const std::runtime_error&)
						{
							timed_out = true;
						}
						require(timed_out && transfer.has_pending(), "Metadata fixture did not retain pending GPU work");
					}
					auto& mapping = (*current)[side];
					switch (field)
					{
					case 0:
						mapping.address += 65536;
						break;
					case 1:
						++mapping.size;
						break;
					case 2:
						mapping.tile->address += 65536;
						break;
					case 3:
						mapping.tile->size *= 2;
						break;
					case 4:
						mapping.tile->pitch *= 2;
						break;
					case 5:
						mapping.tile->bank ^= 1;
						break;
					case 6:
						mapping.tile->location ^= 1;
						break;
					case 7:
						mapping.tile->io_offset += 65536;
						break;
					case 8:
						++mapping.tile->compression;
						break;
					case 9:
						++mapping.tile->compression_base;
						break;
					case 10:
						mapping.tile.reset();
						break;
					}
					current.reset();
					require(!lifetime.expired(), "Pending transfer lost owned metadata query");
					metadata_gate.signaledValue = signal;
					bool refused = false;
					try
					{
						transfer.synchronize();
					}
					catch (const scaled_image_mapping_conflict&)
					{
						refused = true;
					}
					require(refused && !transfer.has_pending() && lifetime.expired() && provider->notifications == notifications &&
								memory->read_surface(target).bytes == output,
							"Changed mapping published pixels or leaked its query owner");
				}
			memory->write(source.range, input);
			memory->write(target.range, output);
			{
				auto current = std::make_shared<scaled_image_mapping_epoch::mappings>(original_mappings);
				const std::weak_ptr<scaled_image_mapping_epoch::mappings> lifetime = current;
				{
					std::optional<scaled_image_mapping_epoch> epoch;
					capture([&]
							{ epoch.emplace(validator(current)); });
					bool timed_out = false;
					try
					{
						transfer.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
										 { commands.wait_event(metadata_gate, 23); }, [owned = std::move(*epoch)]
										 { owned.validate(); });
					}
					catch (const std::runtime_error&)
					{
						timed_out = true;
					}
					require(timed_out && transfer.has_pending(), "Stable metadata fixture lacks pending GPU work");
				}
				current.reset();
				require(!lifetime.expired(), "Stable pending query lost ownership");
				metadata_gate.signaledValue = 23;
				transfer.synchronize();
				require(lifetime.expired() && memory->read_surface(target).bytes == expected, "Stable metadata failed fresh publication");
			}
			require(metadata_checks == 95, "Metadata checks did not cover CPU epochs and cache preflights");
			// A failing final query also discards publication and releases its
			// owner;pending teardown must never invoke the validator later.
			for (bool abandon : {false, true})
			{
				memory->write(target.range, output);
				auto fail = std::make_shared<bool>(false);
				const std::weak_ptr<bool> lifetime = fail;
				unsigned checks = 0;
				const auto notifications = provider->notifications;
				const unsigned signal = abandon ? 25 : 24;
				{
					scaled_image_transfer isolated(ctx, surfaces, textures, memory, capture, 1 << 20);
					bool timed_out = false;
					try
					{
						isolated.execute(plan, std::chrono::milliseconds(1), [&](auto& commands)
										 { commands.wait_event(metadata_gate, signal); }, [fail, &checks, provider]
										 {
								require(guest_memory_epoch::current(*provider) != nullptr, "Query failure escaped CPU exclusion");
								++checks;
								if (*fail) throw std::range_error("mapping query failed"); });
					}
					catch (const std::runtime_error&)
					{
						timed_out = true;
					}
					require(timed_out && isolated.has_pending() && checks == 2, "Failing query fixture lacks pending work");
					*fail = true;
					fail.reset();
					require(!lifetime.expired(), "Pending failing query owner released early");
					if (!abandon)
					{
						metadata_gate.signaledValue = signal;
						bool refused = false;
						try
						{
							isolated.synchronize();
						}
						catch (const std::range_error&)
						{
							refused = true;
						}
						require(refused && !isolated.has_pending() && checks == 3, "Final query error did not discard publication");
					}
				}
				require(lifetime.expired(), "Retired query retained its owner");
				metadata_gate.signaledValue = signal;
				ctx.wait_idle();
				require(checks == (abandon ? 2 : 3) && provider->notifications == notifications && memory->read_surface(target).bytes == output,
						"Failed or destroyed metadata transaction published late VM data");
			}
			// Callbacks cannot replace the caller's plan after its bytes have been
			// captured. Both the encoded operation and published range stay owned.
			memory->write(source.range, input);
			memory->write(target.range, output);
			auto mutable_plan = plan;
			transfer.execute(mutable_plan, std::chrono::seconds(30), [&](auto&)
							 {
				mutable_plan.source.width = 1;
				mutable_plan.destination.range.address = 0x7000; });
			require(memory->read_surface(target).bytes == expected, "Callback changed captured transfer plan");
			std::cout << "Native image mapping publication:" << mapping_cases << " source/target metadata changes during real pending GPU work refused without VM publication;owned query survives timeout and retires on refusal/success;stable retry and95CPU-epoch checks,stale capture rejected,caller plan mutation cannot change encoding\n";
			std::cout << "Native scaled-image:" << cases << " GPU RGB565/ARGB8 nearest/linear/flip/clip/linear-Morton/padding cases;578rational boundaries/all65536RGB565 roundtrips;two CPU epochs,dirty source,target cache invalidation/retained old pixels,physical memmove,pending timeout/frozen source/padding survival,CAS refusal/retry,encode cancellation and no late teardown publication passed\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
