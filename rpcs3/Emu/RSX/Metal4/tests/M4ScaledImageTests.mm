#include "M4GuardedGuestMemory.h"
#include "M4ScaledImage.h"
#include "M4ScaledImageMapping.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <thread>

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
	bool typed_identities=false;
    bool retained_slices=false;
    std::shared_ptr<const physical_backing> retain_segments(guest_surface_range range) override
    {
        if(!retained_slices)return {};
        const auto offset=range.address%bytes.size();
        if(range.address>=2*bytes.size()||range.size>bytes.size()-offset)throw std::out_of_range("Native scale retained slice range");
        auto owner=std::shared_ptr<void>(shared_from_this(),bytes.data());
        return std::make_shared<physical_backing>(range,std::vector<physical_backing::segment>{{range,std::move(owner),offset,bytes.size()}});
    }
    std::vector<guest_surface_range> physical_write_aliases(std::span<const std::shared_ptr<const physical_backing>> written,std::size_t limit) override
    {
        std::vector<guest_surface_range> result;const auto owner=shared_from_this();
        for(const auto& backing:written)for(const auto& part:backing->segments())
        {
            require(!part.owner.owner_before(owner)&&!owner.owner_before(part.owner),"Physical scale fixture received another backing");
            if(result.size()>limit||limit-result.size()<2)throw std::length_error("Physical scale fixture alias budget");
            result.push_back({static_cast<std::uint32_t>(part.offset),part.guest.size});
            result.push_back({static_cast<std::uint32_t>(part.offset+bytes.size()),part.guest.size});
        }
        std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.address<b.address;});
        std::vector<guest_surface_range> merged;
        for(const auto& range:result)
        {
            if(!merged.empty()&&range.address<=std::uint64_t(merged.back().address)+merged.back().size)
                merged.back().size=std::max<std::uint64_t>(std::uint64_t(merged.back().address)+merged.back().size,std::uint64_t(range.address)+range.size)-merged.back().address;
            else merged.push_back(range);
        }
        return merged;
    }
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		if(!typed_identities)return {};
		const auto offset=range.address%bytes.size();
		if(range.address>=2*bytes.size()||range.size>bytes.size()-offset)throw std::out_of_range("Native scale identity range");
		return std::vector<storage_identity_span>{{shared_from_this(),offset,range}};
	}
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
	bool forbid_pixels=false;unsigned reads=0,writes=0;std::size_t read_bytes=0;
    std::vector<guest_surface_range> allowed_input_reads;
    void check_read(guest_surface_range range)const
    {
        if(!forbid_pixels)return;
        constexpr std::uint64_t mirror=1u<<20;const auto address=range.address%mirror;
        if(!std::any_of(allowed_input_reads.begin(),allowed_input_reads.end(),[&](const auto& p){const auto begin=p.address%mirror;return address>=begin&&address-begin<=p.size&&range.size<=p.size-(address-begin);}))
            throw std::runtime_error("Native scale captured VM outside its RAM input: address="+std::to_string(range.address)+" bytes="+std::to_string(range.size));
    }
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override {return raw->watch_surface(coverage);}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override {check_read(coverage.range);++reads;read_bytes+=coverage.range.size;return raw->read_surface(coverage);}
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
	std::optional<guest_surface_snapshot> read_physical_surface(const physical_surface& p) override {check_read(p.coverage().range);++reads;read_bytes+=p.coverage().range.size;return raw->read_physical_surface(p);}
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
			const auto vm=memory->access->copy();const auto reads=memory->reads,writes=memory->writes,notifications=memory->access->notifications;const auto before=surfaces.transaction_statistics();const auto source_before=source?surfaces.status(source):cached_surface_status{};memory->forbid_pixels=true;transfer.execute(plan);memory->forbid_pixels=false;require(transfer.native_completed()==1&&transfer.vm_completed()==0&&!transfer.has_pending(),"Suitable scaled image bypassed native route");require((ram_background?memory->reads>reads:memory->reads==reads)&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Native scale read target/wrote guest allocation");
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
            const unsigned prefix=ram?1u:deferred==5?2u:0u;
            const bool source_dirty=deferred&&deferred!=5?source_before.dirty:!ram;
            require(surfaces.status(source).dirty==source_dirty&&surfaces.status(destination).dirty&&surfaces.transaction_statistics().submitted==before.submitted+1+prefix,"Native scale lost authority or added source submissions");
            if(deferred)
            {
                if(deferred==5)require(surfaces.status(source).initialized&&!surfaces.status(source).deferred_copies,"Missing-background source did not materialize natively");
                else{const auto now=surfaces.status(source);require(now.initialized==source_before.initialized&&now.dirty==source_before.dirty&&now.spilled==source_before.spilled&&now.deferred_copies==source_before.deferred_copies&&now.content_generation==source_before.content_generation,"Native deferred input was materialized or consumed during scale");}
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
void native_scaled_image_transactions(bool cold=false,bool ram=false,bool queued=false,bool previous_policy=false)
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
            fragment_texture_registers words{};std::optional<versioned_guest_texture> old_image,observed_before_poll;
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
            bool failed=false,returned_without_wait=false;std::string execution_error;
            const bool gated=mode>=2&&mode!=4&&mode!=12&&mode!=16&&mode!=17;
            if(ram)memory->allowed_input_reads={plan.source.range};
            memory->forbid_pixels=true;
            const auto execute=[&]
            {
            try
            {
                const auto submit=queued&&!previous_policy?&scaled_image_transfer::enqueue:&scaled_image_transfer::execute;
                (transfer.get()->*submit)(plan,mode<2||mode==12||mode==16||mode==17?30s:1ms,[&](context& commands)
                {
                    token=commands.recording_submission();
                    if(mode==4)throw std::range_error("Native scale encode rejected");
                    if(mode==16){const auto state=ledger->statistics();blocker=ledger->shared_buffer(state.budget-state.reserved_bytes);}
                    if(mode>=2&&mode!=4&&mode!=12&&mode!=16&&mode!=17)commands.wait_event(gate,1);
                },validate);
                if(queued&&gated)
                {
                    returned_without_wait=true;
                    require(transfer->has_pending()&&token&&!token->ready()&&!transfer->poll()&&!transfer->native_completed(),
                        "Queued native scale did not retain an uncompleted private transaction");
                    const auto submitted=surfaces.transaction_statistics().submitted;
                    bool busy=false;try{transfer->enqueue(plan,1ms);}catch(const submission_wait_timeout&){busy=true;}
                    require(busy&&surfaces.transaction_statistics().submitted==submitted,
                        "A second queued scale bypassed its one-job ownership bound");
                }
                if(queued)transfer->synchronize(gated?1ms:30s);
            }
            catch(const std::runtime_error& error){failed=true;execution_error=error.what();}
            };
            execute();
            if(ram&&failed&&!token&&!transfer->has_pending()&&surfaces.status(source).pending)
            {
                // A source-upload timeout precedes the actual scaled writer.
                // Finish that known prefix before explicitly retrying execute.
                require(surfaces.status(destination).content_generation==before.content_generation&&memory->access->copy()==vm,"Pending input prefix published native output");
                surfaces.synchronize(30s);require(surfaces.status(source).initialized&&!surfaces.status(source).dirty,"RAM input prefix did not finish cleanly");failed=false;execute();
            }
            if(queued&&gated)require(returned_without_wait,"Native scaled-image enqueue waited for GPU completion");
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
                if(!(failed&&transfer->has_pending()&&surfaces.status(destination).pending&&token&&!token->ready()))throw std::runtime_error("Native timeout did not retain a real pending write:mode="+std::to_string(mode)+" cold="+std::to_string(cold)+" ram="+std::to_string(ram)+" failed="+std::to_string(failed)+" transfer="+std::to_string(transfer->has_pending())+" source="+std::to_string(surfaces.status(source).pending)+" target="+std::to_string(surfaces.status(destination).pending)+" token="+std::to_string(bool(token))+" error="+execution_error);
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
                bool external_rejection=false,owner_rejection=false;
                try
                {
                    if(transfer&&queued)
                    {
                        const auto deadline=std::chrono::steady_clock::now()+30s;
                        while(!token->ready()&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
                        require(token->ready(),"Queued native scale failed to reach terminal GPU state");
                        if(mode==3)
                        {
                            try{surfaces.synchronize();}catch(const std::runtime_error&){external_rejection=true;}
                            try{(void)transfer->poll();}catch(const std::runtime_error&){owner_rejection=true;throw;}
                        }
                        if(mode==2)
                        {
                            ctx.begin();const auto recording=ctx.recording_submission();
                            require(transfer->poll()&&!recording.ready(),
                                "Native scale polling flushed an unrelated recording");
                            require(recording.same_batch(ctx.recording_submission()),"Native scale polling replaced an unrelated recording");
                            ctx.cancel();
                        }
                        else
                        {
                            if(mode==11)
                            {
                                // A normal texture observer may retire the
                                // surface before the transfer owner is polled.
                                surfaces.synchronize();
                                observed_before_poll=textures.resolve_versioned(words,surfaces);
                                require(transfer->has_pending()&&!transfer->native_completed()&&
                                    observed_before_poll->image.texture!=old_image->image.texture,
                                    "Texture observer reused stale scale pixels before owner polling");
                            }
                            require(transfer->poll(),"Ready native scale was not polled");
                        }
                    }
                    else if(transfer)transfer->synchronize();else surfaces.synchronize();
                }
                catch(const std::runtime_error&){failed=true;}
                if(queued&&mode==3)require(external_rejection&&owner_rejection,
                    "Externally rejected native scale was reported successful by its owner");
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
                        if(queued)require(observed_before_poll&&snapshot(observed_before_poll->image.texture)==native_expected,
                            "Texture observation before scale-owner polling lost native output pixels");
                    }
                    if(cold&&mode==9){const auto at=padding_address-0x6000;expected[(at/33)*32+at%33]=std::byte{0x5c};}
                    memory->forbid_pixels=false;const unsigned observed_base=mode==17?0x1000:0x6000;surfaces.prepare_cpu_read({observed_base,264});const auto visible=memory->raw->read({observed_base,264}).bytes;
                    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<32;++x)require(visible[y*33+x]==expected[y*32+x],"CPU observation missed native scaled-image pixels/background");
                }
            }
            memory->forbid_pixels=false;++cases;
        }
    }
    if(queued)std::cout<<"Queued native scale: early return behind real GPU gate, nonblocking poll, unrelated recording retained and one-job bound passed; ";
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
                require(failed&&token&&surfaces.status(source).deferred_copies==((role&1)?2u:0u)&&!surfaces.status(target).deferred_copies&&surfaces.status(target).content_generation==expected_prefix_generation&&memory->access->copy()==vm,"Deferred native prefix/output ownership or generation differs");
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
    std::cout<<"Deferred scaled-image transactions:"<<cases<<" preserved native source queues and target-only prefixes,full union/partial native-spilled target backgrounds,physical-mirror targets,real timeouts,CPU/metadata conflicts,encode/queued cancellation,abandoned output owner,quota refusal/retry andCPU dispatcher retry preserve prior prefix generations/backgrounds;zero VMpixel accesses andexact native outputs passed\n";
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
            if(deferred)
            {
                require(timed_out&&dependency_called&&transfer->has_pending()&&!surfaces->status(source).initialized&&surfaces->status(source).deferred_copies==2&&surfaces->status(destination).pending&&!blocker.ready(),"Native queued source did not share the real gated output");
                require(surfaces->status(destination).content_generation==before.content_generation&&memory->access->copy()==vm&&memory->reads==0&&memory->writes==0,"Native queued source published output or used VM pixels before completion");
                if(outcome==2)
                {
                    transfer.reset();textures.reset();surfaces.reset();source.reset();destination.reset();ctx->shutdown();ctx.reset();bool cancelled=false;try{blocker.wait();}catch(const std::runtime_error&){cancelled=true;}
                    require(cancelled&&blocker.cancelled()&&memory->access->copy()==vm,"Native queued source shutdown published output or lost cancellation");
                }
                else
                {
                    if(outcome==1)transfer.reset();gate.signaledValue=1;blocker.wait();bool rejected=false;try{surfaces->synchronize(30s);}catch(const std::runtime_error&){rejected=true;}
                    require(rejected==bool(outcome)&&!surfaces->status(source).initialized&&surfaces->status(source).deferred_copies==2&&!surfaces->status(destination).pending&&surfaces->status(destination).content_generation==before.content_generation+unsigned(!outcome)&&memory->access->copy()==vm&&memory->reads==0&&memory->writes==0,"Native queued source completion consumed input or lost output rollback");
                    if(outcome==0){transfer->synchronize(30s);require(transfer->native_completed()==1&&transfer->vm_completed()==0,"Native queued source did not complete its original output");memory->forbid_pixels=false;surfaces->prepare_cpu_read(output.range);require(memory->raw->read_surface(output).bytes==expected,"Native queued source CPU visibility differs");}
                }
            }
            else
            {
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
            }
            ++cases;
        }
        if(!provider.expired()||ledger->statistics().reserved_bytes!=0)throw std::runtime_error("RAM prefix teardown retained provider/native charges cold="+std::to_string(cold)+" outcome="+std::to_string(outcome)+" providerExpired="+std::to_string(provider.expired())+" bytes="+std::to_string(ledger->statistics().reserved_bytes));
    }
    std::cout<<(deferred?"Deferred native source/output transaction:":"RAM source upload prefix:")<<cases<<" real queued input timeouts with existing/cold targets; native queued inputs share output, actual VM uploads retain explicit finished-prefix retry; abandoned owner and blocked cache/context shutdown never publish early output; provider retirement/zero ledger charges passed\n";
}

void bounded_scaled_image_cases(bool overlap=false)
{
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);
    std::size_t profiles=0,exact=0,source_bytes=0,cropped_profiles=0,cropped_texels=0,target_imports=0,target_texels=0,overlap_bytes=0;
    for(unsigned type=0;type<8;++type)for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned layout=0;layout<4;++layout)
    for(unsigned count:{1u,2u,4u})for(bool clean:{false,true})for(bool cold:{false,true})for(unsigned flags=0;flags<4;++flags)
    @autoreleasepool
    {
        if(overlap&&!clean)continue;
        try
        {
        const surface_format format=type<4?surface_format{std::array{surface_color::b8,surface_color::gb8,surface_color::argb8,surface_color::rgba32f}[type]}:surface_format{static_cast<surface_depth>(type-3)};
        const auto bpp=surface_guest_bytes(format);const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
        auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;
        const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
        const auto cache_limit=type==3&&layout==1?128u<<10:16u<<20;
        surface_cache surfaces(ctx,memory,cache_limit,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
        scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,256);
        const bool swizzled=layout&1,banked=layout&2;
        const auto morton=swizzled?std::optional<swizzle_region>{{{128,128,1},{3,2,0},{16,16,1}}}:std::nullopt;
        const auto tile=banked?std::optional<banked_tile_layout>{{0x10000,0x80000,512,1,static_cast<std::uint32_t>(0x1200+bpp)}}:std::nullopt;
        const auto geometry=plan_guest_surface(tile?tile->address+tile->offset:0x10000,16,16,bpp,swizzled?0:banked?512:16*bpp+3,morton);
        require((tile?tile->size:geometry.range.size)>256,"Bounded source does not exceed former whole-storage staging limit");
        require(cache_limit!=(128u<<10)||geometry.range.size>cache_limit,"Bounded Morton source does not exceed former cache snapshot admission limit");
        const auto source=surfaces.acquire({geometry.range.address,{16/factors.x,16/factors.y,count,format,false},swizzled?0:geometry.row_pitch,morton,tile});
        std::vector<std::byte> words(256*bpp);
        for(unsigned i=0;i<256;++i)
        {
            if(type==3)for(unsigned component=0;component<4;++component){const auto bits=0x3e000000u+i*147+component*937;for(unsigned lane=0;lane<4;++lane)words[i*bpp+component*4+lane]=std::byte(bits>>(24-lane*8));}
            else if(type<4)for(unsigned lane=0;lane<bpp;++lane)words[i*bpp+lane]=std::byte((i*79+lane*43+17)&255);
            else
            {
                const unsigned stencil=(i*53+37)&255;
                const auto word=type==4?(i*23327u+73)&65535:type==5?((i&1?0xffffffu:0)<<8)|stencil:type==6?0x1000u+(i*0x1b3u)%0x5fff:((0x3d800000u+i*0x11700u)>>7<<8)|stencil;
                for(unsigned lane=0;lane<bpp;++lane)words[i*bpp+lane]=std::byte(word>>(8*(bpp-1-lane)));
            }
        }
        const auto linear=[&](context& commands,id<MTLTexture> image,unsigned size,unsigned height=0)
        {if(count==1)return image;auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:size height:height?height:size mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;return make_texture(commands,d);};
        surfaces.prepare(source,surface_initialization::background);
        surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){const auto image=linear(commands,images[0],16);texture_region all;all.size=MTLSizeMake(16,16,1);const auto bytes=make_upload(commands,words);if(type<4)colors.upload(commands,std::get<surface_color>(format),bytes,{},image,all);else depths.upload(commands,std::get<surface_depth>(format),bytes,{},image,all);if(count>1)samples.import(commands,image,images[0]);});
        if(clean)surfaces.flush(source);if(flags&1)surfaces.spill(source);
        std::vector<std::byte> input(16*in);std::vector<guest_storage_segment> spans;std::array<bool,256> selected{};
        for(unsigned i=0;i<input.size();++i)
        {
            const auto x=(i*7+i/5)%16,y=(i*5+i/7)%16;const auto lane=(i*11+3)%bpp;
            const auto raw=offset(geometry,x,y)+lane;const auto physical=tile?banked_display_storage_offset(*tile,tile->offset+raw):raw;
            require(!tile||physical<tile->size,"Bounded fixture selects inactive bank cell");
            auto address=static_cast<unsigned>((tile?tile->address:geometry.range.address)+physical);
            if(overlap&&i>=input.size()/2){address=0xa0000+i;input[i]=std::byte((i*71+103)&255);memory->raw->write({address,1},std::span{&input[i],1});}
            else {selected[y*16+x]=true;input[i]=words[(y*16+x)*bpp+lane];if(overlap&&(flags&1))address+=0x100000;}
            if(!spans.empty()&&std::uint64_t(spans.back().address)+spans.back().size==address)++spans.back().size;else spans.push_back({address,i,1});
        }
        cached_surface_ref overlapping;
        if(overlap)
        {
            overlapping=surfaces.acquire({spans.front().address,{4*in/2/factors.x,4/factors.y,count,surface_color::gb8,false},4*in,{},{},spans});
            surfaces.prepare(overlapping);
            require(!surfaces.status(source).dirty&&!surfaces.status(overlapping).dirty&&surfaces.status(source).native_write_epoch>surfaces.status(overlapping).native_write_epoch,"Overlap fixture lacks ordered clean native authorities");
            if(flags&1)surfaces.spill(overlapping);
        }
        const auto df=out==2?surface_color::rgb565:surface_color::argb8;const auto destination=cold?cached_surface_ref{}:surfaces.acquire({0xe0000,{8/factors.x,8/factors.y,count,df,false},8*out+1,{}});
        std::vector<std::byte> background(64*out,std::byte{0xa5});
        if(destination){surfaces.prepare(destination,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{destination,true}},[&](context& commands,auto images){const auto image=linear(commands,images[0],8);texture_region all;all.size=MTLSizeMake(8,8,1);colors.upload(commands,df,make_upload(commands,background),{},image,all);if(count>1)samples.import(commands,image,images[0]);});}
        const auto source_image=plan_guest_surface(spans.front().address,4,4,in),target=plan_guest_surface(0xe0000+(8*out+1)+out,5,3,out,8*out+1);
        scaled_image_plan plan{source_image,target,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,4,4,7,5,1,1,bool(flags&1),bool(flags&1),bool(flags&2)};plan.source_segments=spans;
        auto packed=plan;packed.source=plan_guest_surface(0,4,4,in);packed.destination=plan_guest_surface(0,5,3,out);packed.source_segments.clear();const auto patch=reference(packed,input,std::vector<std::byte>(15*out));
        const auto vm=memory->access->copy();const auto reads=memory->read_bytes;const auto writes=memory->writes,notifications=memory->access->notifications;memory->forbid_pixels=true;transfer.execute(plan);
        require(transfer.native_completed()==1&&transfer.vm_completed()==0&&memory->read_bytes==reads&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Bounded native input captured source/target VMpixels orfailed native routing");
        const auto sample_stats=transfer.native_sample_statistics();const auto selected_texels=std::count(selected.begin(),selected.end(),true)+(overlap?input.size()/4:0);const bool private_spill=(flags&1)&&!overlap;
        require(sample_stats.full_expansions==0&&sample_stats.full_texels==0&&sample_stats.imports==unsigned(count>1&&!cold),"Scaled transfer retained a whole-source/target MSAA expansion");
        require(sample_stats.cropped_imports==sample_stats.imports&&sample_stats.imported_texels==(count>1&&!cold?15u:0u)&&sample_stats.cropped_imported_texels==sample_stats.imported_texels,"Scaled output imported unrelated target samples");
        target_imports+=sample_stats.cropped_imports;target_texels+=sample_stats.cropped_imported_texels;
        require(sample_stats.cropped_texels==(count>1&&!private_spill?selected_texels:0)&&bool(sample_stats.cropped_expansions)==(count>1&&!private_spill),"MSAA input expansion did not select exactly required native sample texels");
        if(count>1){++cropped_profiles;cropped_texels+=sample_stats.cropped_texels;}
        require(surfaces.status(source).dirty==!clean,"Bounded input changed source clean/dirty authority");if(private_spill)require(surfaces.status(source).spilled,"Bounded private spill was restored persistently");
        if(overlap)
        {
            require(!surfaces.status(overlapping).dirty&&!surfaces.status(source).spilled&&!surfaces.status(overlapping).spilled,"Clean overlapping readers were dirtied,flushed ornot both restored");
            const auto data=make_buffer(ctx,input.size(),buffer_storage::shared);
            surfaces.draw(std::array{cached_surface_use{overlapping,false}},[&](context& commands,auto images){const auto image=linear(commands,images[0],4*in/2,4);if(count>1)samples.expand(commands,images[0],image);texture_region all;all.size=MTLSizeMake(4*in/2,4,1);colors.download(commands,surface_color::gb8,image,all,data,{});});
            require(std::memcmp(data.contents,input.data(),input.size())==0,"Overlapping clean GB8 source bytes changed");overlap_bytes+=input.size();
        }
        const auto original=make_buffer(ctx,words.size(),buffer_storage::shared);
        surfaces.draw(std::array{cached_surface_use{source,false}},[&](context& commands,auto images){const auto image=linear(commands,images[0],16);if(count>1)samples.expand(commands,images[0],image);texture_region all;all.size=MTLSizeMake(16,16,1);if(type<4)colors.download(commands,std::get<surface_color>(format),image,all,original,{});else depths.download(commands,std::get<surface_depth>(format),image,all,original,{});});
        require(std::memcmp(original.contents,words.data(),words.size())==0,"Bounded packing changed full native source guest words");source_bytes+=words.size();memory->forbid_pixels=false;
        if(destination){for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,background.begin()+((y+1)*8+1)*out);const auto raw=make_buffer(ctx,background.size(),buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{destination,false}},[&](context& commands,auto images){const auto image=linear(commands,images[0],8);if(count>1)samples.expand(commands,images[0],image);texture_region all;all.size=MTLSizeMake(8,8,1);colors.download(commands,df,image,all,raw,{});});require(std::memcmp(raw.contents,background.data(),background.size())==0,"Bounded output/background differs from independent filter oracle");exact+=background.size();}
        else {surfaces.prepare_cpu_read(target.range);auto expected=vm;for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,expected.begin()+target.range.address+y*target.row_pitch);require(memory->access->copy()==expected,"Bounded cold observation changed unrelated VM/source/padding");exact+=patch.size();}
        ++profiles;
        }
        catch(const std::exception& error){throw std::runtime_error("Bounded scale type="+std::to_string(type)+" in="+std::to_string(in)+" out="+std::to_string(out)+" layout="+std::to_string(layout)+" samples="+std::to_string(count)+" clean="+std::to_string(clean)+" cold="+std::to_string(cold)+" flags="+std::to_string(flags)+": "+error.what());}
    }
    std::cout<<(overlap?"Overlapping clean scaled inputs:":"Bounded native scaled inputs:")<<profiles<<" sparse/partial-lane color/depth profiles with256byte transfer quota andlarger oddpitch/Morton/banked source allocations,1-2-4samples,clean-dirty/resident-spill,existing-cold,flip/filter;"<<exact<<" exact output-background bytes,"<<source_bytes<<" full source guest-word bytes unchanged;"<<cropped_profiles<<" MSAA profiles/"<<cropped_texels<<" live-source expanded input sample texels plus private raw-spill windows,"<<target_imports<<" cropped target imports/"<<target_texels<<" selected output sample texels,no whole source-target expansion/import,native target backgrounds retained;zero VMpixel capture/write andwholeVM padding exact passed\n";
    if(overlap)std::cout<<"Overlapping clean authority sources:"<<overlap_bytes<<" GB8 source bytes,ordered native/VM-decoded epochs,physical mirrors andboth actual spill restores passed\n";
    else std::cout<<"Bounded source admission:192Morton cases beyond128KiB cache snapshot admission passed\n";
}

void frozen_deferred_scaled_images(unsigned mixed=0, bool interleaved=false, unsigned partial=0, unsigned discarded=0)
{
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);
    unsigned profiles=0;std::size_t checked=0, captured=0;
    for(unsigned type=0;type<8;++type)for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned layout=0;layout<4;++layout)
    for(unsigned count:{1u,2u,4u})for(bool cold:{false,true})for(bool second_latest:{false,true})for(unsigned flags=0;flags<4;++flags)
    @autoreleasepool
    {
        if((interleaved||partial||discarded)&&!second_latest)continue;
        if((mixed||partial>1)&&(!second_latest||(type!=2&&type!=7)||layout>1||in==2))continue;
        try
        {
        auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;
        const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
        surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,256);
        const surface_format format=type<4?surface_format{std::array{surface_color::b8,surface_color::gb8,surface_color::argb8,surface_color::rgba32f}[type]}:surface_format{static_cast<surface_depth>(type-3)};
        const auto bpp=surface_guest_bytes(format);const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
        const bool swizzled=layout&1,banked=layout&2;const auto morton=swizzled?std::optional<swizzle_region>{{{128,128,1},{3,2,0},{16,16,1}}}:std::nullopt;
        const auto tile=banked?std::optional<banked_tile_layout>{{0x10000,0x80000,512,1,static_cast<std::uint32_t>(0x1200+bpp)}}:std::nullopt;
        const auto geometry=plan_guest_surface(tile?tile->address+tile->offset:0x10000,16,16,bpp,swizzled?0:banked?512:16*bpp+3,morton);
        const auto first=surfaces.acquire({geometry.range.address,{16/factors.x,16/factors.y,count,format,false},swizzled?0:geometry.row_pitch,morton,tile});
        const auto donor=mixed?first:surfaces.acquire({0xb0000,{16/factors.x,16/factors.y,count,format,false},16*bpp,{}});
        const auto other_width=discarded==2?8u:6u;
        const auto other=surfaces.acquire({0xc0000,{other_width/factors.x,4/factors.y,count,surface_color::gb8,false},other_width*2,{}});
        const auto linear=[&](context& commands,id<MTLTexture> image,unsigned w,unsigned h)
        {if(count==1)return image;auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:w height:h mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;return make_texture(commands,d);};
        std::vector<std::byte> words(256*bpp),other_words(other_width*8),input(64);std::vector<guest_storage_segment> spans;
        for(unsigned i=0;i<256;++i)
        {
            if(type==3)for(unsigned c=0;c<4;++c){const auto bits=0x3e000000u+i*147+c*937;for(unsigned l=0;l<4;++l)words[i*bpp+c*4+l]=std::byte(bits>>(24-l*8));}
            else if(type<4)for(unsigned l=0;l<bpp;++l)words[i*bpp+l]=std::byte((i*79+l*43+17)&255);
            else {const auto stencil=(i*53+37)&255;const auto word=type==4?(i*23327u+73)&65535:type==5?((i&1?0xffffffu:0)<<8)|stencil:type==6?0x1000u+(i*0x1b3u)%0x5fff:((0x3d800000u+i*0x11700u)>>7<<8)|stencil;for(unsigned l=0;l<bpp;++l)words[i*bpp+l]=std::byte(word>>(8*(bpp-1-l)));}
        }
        for(unsigned i=0;i<other_words.size();++i)other_words[i]=std::byte((i*97+61)&255);
        const auto seed=[&](const auto& target,surface_format f,std::span<const std::byte> bytes,unsigned w,unsigned h)
        {surfaces.prepare(target,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images){const auto image=linear(commands,images[0],w,h);texture_region area;area.size=MTLSizeMake(w,h,1);if(const auto* color=std::get_if<surface_color>(&f))colors.upload(commands,*color,make_upload(commands,bytes),{},image,area);else depths.upload(commands,std::get<surface_depth>(f),make_upload(commands,bytes),{},image,area);if(count>1)samples.import(commands,image,images[0]);});};
        seed(donor,format,words,16,16);if(mixed==2)surfaces.flush(donor);seed(other,surface_color::gb8,other_words,other_width,4);
        for(unsigned i=0;i<64;++i)
        {
            const auto x=discarded==1?8+(i*7+i/5)%8:(i*7+i/5)%16;
            const auto y=(i*5+i/7)%16;const auto lane=(i*11+3)%bpp;const auto raw=offset(geometry,x,y)+lane;
            auto address=i<32?static_cast<unsigned>((tile?tile->address:geometry.range.address)+(tile?banked_display_storage_offset(*tile,tile->offset+raw):raw)):0xa0000u+i;
            if(flags&1)address+=0x100000;
            if(partial||discarded==1)
            {
                input[i]=std::byte((i*41+13)&255);
                memory->raw->write({address,1},std::span<const std::byte>{&input[i],1});
            }
            else if(discarded==2)input[i]=other_words[i];
            else input[i]=i<32?words[(y*16+x)*bpp+lane]:other_words[i-16];
            spans.push_back({address,i,1});
        }
        const auto second_start=discarded==2?0u:16u;
        auto secondary_spans=std::vector<guest_storage_segment>(spans.begin()+second_start,spans.end());for(auto& span:secondary_spans)span.offset-=second_start;
        if(discarded==1)secondary_spans.push_back({0xd0000,48,48});
        const auto second=surfaces.acquire({secondary_spans.front().address,{other_width/factors.x,(discarded==1?8u:4u)/factors.y,count,surface_color::gb8,false},other_width*2,{},{},secondary_spans});
        const auto queue_first=[&]{if(flags&2){const auto half=8/factors.x;surfaces.defer_copy(donor,{0,0,half,16/factors.y},first,0,0);surfaces.defer_copy(donor,{half,0,half,16/factors.y},first,half,0);}else surfaces.defer_copy(donor,{0,0,16/factors.x,16/factors.y},first,0,0);};
        const auto queue_second=[&]{surfaces.defer_copy(other,{0,0,6/factors.x,4/factors.y},second,0,0);};
        std::size_t holes=0;
        if(discarded)
        {
            const auto half=8/factors.x;
            surfaces.defer_copy(donor,{0,0,half,16/factors.y},first,0,0);
            surfaces.defer_copy(other,{0,0,other_width/factors.x,4/factors.y},second,0,discarded==1?4/factors.y:0);
            if(discarded==1)
            {
                for(const auto& span:spans)
                {
                    auto& allowed=memory->allowed_input_reads;
                    if(!allowed.empty()&&std::uint64_t(allowed.back().address)+allowed.back().size==span.address)
                        allowed.back().size+=span.size;
                    else allowed.push_back({span.address,span.size});
                    holes+=span.size;
                }
            }
        }
        else if(partial)
        {
            if(partial>1)
            {
                seed(first,format,words,16,16);
                if(partial>=3)surfaces.flush(first);
                if(partial==4)
                {
                    // Leave the old clean allocation cached but invalidate
                    // its VM revision before taking the new deferred watch.
                    input[2]=std::byte{0xac};
                    memory->raw->write({spans[2].address,1},std::span<const std::byte>{&input[2],1});
                }
            }
            const auto half=8/factors.x;
            const bool split_aspects=type==5||type==7;
            const auto plane=!split_aspects?texture_plane::all:(flags&2)?texture_plane::stencil:texture_plane::depth;
            surfaces.defer_copy(other,{0,0,4/factors.x,4/factors.y},second,0,0);
            surfaces.defer_copy(donor,{half,0,half,16/factors.y},first,0,0,plane);
            if((flags&1)&&partial<4)
                surfaces.defer_copy(other,{4/factors.x,0,2/factors.x,4/factors.y},second,0,0);
            for(unsigned i=0;i<64;++i)
            {
                const auto x=(i*7+i/5)%16,y=(i*5+i/7)%16;
                const auto lane=(i*11+3)%bpp;
                const bool first_patch=i<32&&x<8&&(!split_aspects||(lane==3)==(plane==texture_plane::stencil));
                const bool second_patch=i>=16&&(i-16)%12<8;
                const bool second_last=second_patch&&(flags&1)&&partial<4&&(i-16)%12<4;
                if(second_last)input[i]=other_words[i-16+8];
                else if(first_patch)input[i]=words[(y*16+x+8)*bpp+lane];
                else if(second_patch)input[i]=other_words[i-16];
                else if(i<32&&(partial==2||partial==3))input[i]=words[(y*16+x)*bpp+lane];
                else
                {
                    const guest_surface_range byte{spans[i].address,1};
                    if(partial==4)input[i]=memory->raw->read_surface(guest_byte_coverage(byte)).bytes[0];
                    auto& allowed=memory->allowed_input_reads;
                    if(!allowed.empty()&&std::uint64_t(allowed.back().address)+allowed.back().size==byte.address)
                        ++allowed.back().size;
                    else allowed.push_back(byte);
                    ++holes;
                }
            }
        }
        else if(second_latest){if(!mixed)queue_first();queue_second();for(unsigned i=16;i<32;++i)input[i]=other_words[i-16];}else{queue_second();queue_first();}
        if(interleaved)
        {
            const auto half = 8 / factors.x;
            const bool split_aspects = type == 5 || type == 7;
            const auto plane = !split_aspects ? texture_plane::all :
                (flags & 2) ? texture_plane::stencil : texture_plane::depth;
            surfaces.defer_copy(donor,{half,0,half,16/factors.y},first,0,0,plane);
            for(unsigned i=0;i<32;++i)
            {
                const auto x=(i*7+i/5)%16, y=(i*5+i/7)%16;
                const auto lane=(i*11+3)%bpp;
                if(x<8 && (!split_aspects || (lane==3)==(plane==texture_plane::stencil)))
                    input[i]=words[(y*16+x+8)*bpp+lane];
            }
            if(flags&1)
            {
                surfaces.defer_copy(other,{4/factors.x,0,2/factors.x,4/factors.y},second,0,0);
                for(unsigned i=0;i<48;++i)
                    if(i%12<4)input[16+i]=other_words[i+8];
            }
        }
        if(flags&1){surfaces.spill(donor);surfaces.spill(other);if(partial==2||partial==3)surfaces.spill(first);}
        const auto first_before=surfaces.status(first),second_before=surfaces.status(second);
        require(first_before.deferred_copies==((partial||discarded)?1u:mixed?0u:(flags&2?2u:1u)+unsigned(interleaved))&&second_before.deferred_copies==1+unsigned((interleaved||(partial&&partial<4))&&(flags&1))&&first_before.initialized==bool(mixed||partial>1)&&first_before.dirty==(mixed==1||partial==2)&&!second_before.initialized,"Frozen input fixture has wrong queued or native owners");
        const auto df=out==2?surface_color::rgb565:surface_color::argb8;const auto destination=cold?cached_surface_ref{}:surfaces.acquire({0xe0000,{8/factors.x,8/factors.y,count,df,false},8*out+1,{}});std::vector<std::byte> background(64*out,std::byte{0xa5});if(destination)seed(destination,df,background,8,8);
        const auto source=plan_guest_surface(spans.front().address,16/in,4,in),target=plan_guest_surface(0xe0000+(8*out+1)+out,5,3,out,8*out+1);
        scaled_image_plan plan{source,target,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,16/in,4,11,5,1,1,bool(flags&1),bool(flags&1),bool(flags&2)};plan.source_segments=spans;
        auto canonical=plan;canonical.source=plan_guest_surface(0,16/in,4,in);canonical.destination=plan_guest_surface(0,5,3,out);canonical.source_segments.clear();const auto patch=reference(canonical,input,std::vector<std::byte>(15*out));
        const auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes),notifications=std::size_t(memory->access->notifications);
        const auto submissions=surfaces.transaction_statistics().submitted;
        memory->forbid_pixels=true;transfer.execute(plan);
        require(transfer.native_completed()==1&&transfer.vm_completed()==0&&memory->read_bytes-reads==holes&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Frozen overlapping inputs captured bytes outside holes, wrote VM or missed native route");
        if(discarded)require(surfaces.transaction_statistics().submitted==submissions+1,"Discarded native owner caused a GPU preparation submission");
        captured+=holes;
        for(const auto& pair:std::array{std::pair{first,first_before},std::pair{second,second_before}}){const auto now=surfaces.status(pair.first);require(now.deferred_copies==pair.second.deferred_copies&&now.content_generation==pair.second.content_generation&&now.initialized==pair.second.initialized&&now.dirty==pair.second.dirty&&now.spilled==pair.second.spilled,"Frozen input consumed orinvalidated original owner");}
        if(destination){for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,background.begin()+((y+1)*8+1)*out);const auto data=make_buffer(ctx,background.size(),buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{destination,false}},[&](context& commands,auto images){const auto image=linear(commands,images[0],8,8);if(count>1)samples.expand(commands,images[0],image);texture_region all;all.size=MTLSizeMake(8,8,1);colors.download(commands,df,image,all,data,{});});require(!std::memcmp(data.contents,background.data(),background.size()),"Frozen native output/background differs from independent oracle");checked+=background.size();}
        else {memory->forbid_pixels=false;surfaces.prepare_cpu_read(target.range);auto expected=vm;for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,expected.begin()+target.range.address+y*target.row_pitch);require(memory->access->copy()==expected,"Frozen cold output changed source/VMpadding");checked+=patch.size();}
        ++profiles;
        }
        catch(const std::exception& e){throw std::runtime_error("Frozen input type="+std::to_string(type)+" in="+std::to_string(in)+" out="+std::to_string(out)+" layout="+std::to_string(layout)+" samples="+std::to_string(count)+" cold="+std::to_string(cold)+" latest="+std::to_string(second_latest)+" flags="+std::to_string(flags)+": "+e.what());}
    }
    require(profiles==((mixed||partial>1)?192u:(interleaved||partial||discarded)?3072u:6144u),"Frozen input profile coverage incomplete");
    if(discarded)std::cout<<(discarded==1?"RAM-only queued inputs:":"Shadowed queued inputs:")<<profiles<<" profiles,"<<checked<<" exact output-background bytes,"<<captured<<" VM input bytes,one native transaction and original queued owners unchanged passed\n";
    else if(partial)std::cout<<"Partial deferred authority mode "<<partial<<":"<<profiles<<" profiles,"<<checked<<" exact output-background bytes,"<<captured<<" VM hole bytes,original owners unchanged and zero VM writes passed\n";
    else if(interleaved)std::cout<<"Interleaved deferred byte authority:"<<profiles<<" profiles,including depth-only/stencil-only sections and 1-2-4 samples;"<<checked<<" exact output-background bytes,unchanged queued owners and no VM capture/write passed\n";
    else if(mixed)std::cout<<"Frozen mixed "<<(mixed==1?"dirty":"clean")<<" and deferred inputs:"<<profiles<<" color/depth oddpitch/Morton profiles,1-2-4samples,resident or private spill read-version,original spill preserved;"<<checked<<" exact output-background bytes,no VM capture/write passed\n";
    else std::cout<<"Frozen deferred overlapping inputs:"<<profiles<<" color/depth oddpitch/Morton/banked/physical-mirror profiles,1-2-4samples,RGB565/ARGB8 input/output,both queue orders,full/split-union plans,captured spilled donors,existing/cold targets,flip/filter;"<<checked<<" exact output-background bytes;original deferred owners/generations unchanged,zeroVMpixel capture/write andwholeVM padding exact passed\n";
}

void overlapping_clean_transactions(bool frozen=false, bool interleaved=false, bool partial=false, unsigned discarded=0, bool donor_target=false)
{
    using namespace std::chrono_literals;context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);sample_transfer samples(ctx);unsigned cases=0,ambiguous=0,old_donors=0,pool_refusals=0;
    const unsigned case_limit=donor_target&&discarded==1?12u:frozen&&!partial&&!discarded?10u:11u;
    for(unsigned count:{1u,2u,4u})for(bool cold:{false,true})for(bool spill:{false,true})for(unsigned failure=0;failure<case_limit;++failure)
    @autoreleasepool
    {
        if((count==2&&!donor_target)||(cold&&donor_target))continue;
        try
        {
        const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;
        auto refuse=std::make_shared<bool>(false),metadata=std::make_shared<bool>(true);
        const cpu_capture_dispatch capture=[access=memory->access,refuse](const auto& work){if(*refuse)throw std::range_error("Overlap CPU dispatcher refused");access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
        const surface_allocation_description target_layout{8/factors.x,8/factors.y,count,surface_color::argb8,false};
        const auto second_width=partial?10u:12u;
        const surface_allocation_description other_layout{second_width/factors.x,4/factors.y,count,surface_color::gb8,false};
        std::size_t budget=16u<<20;
        if(failure==11)
        {
            surface_pool probe(ctx,budget);
            const auto a=probe.acquire(target_layout), b=probe.acquire(other_layout), c=probe.acquire(target_layout);
            budget=probe.statistics().reserved_bytes;
        }
        surface_cache surfaces(ctx,memory,budget,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
        const auto input=plan_guest_surface(0x1000,6,4,4,28),output=plan_guest_surface(0x6000+33+4,5,3,4,33);
        std::vector<std::byte> vm_input(input.range.size),first_words(64),packed_input(96),background(256,std::byte{0xa5});
        for(unsigned i=0;i<vm_input.size();++i)vm_input[i]=std::byte((i*17+107)&255);for(unsigned i=0;i<64;++i)first_words[i]=std::byte((i*73+29)&255);memory->raw->write(input.range,vm_input);
        const auto source_height=discarded==1?8u:4u;
        const auto first=surfaces.acquire({0x1004,{4/factors.x,source_height/factors.y,count,surface_color::argb8,false},28,{}});
        const auto linear=[&](context& commands,id<MTLTexture> image,unsigned width,unsigned height)
        {if(image.sampleCount==1)return image;auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:width height:height mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;return make_texture(commands,d);};
        const auto seed=[&](const auto& target,surface_color f,const auto& words,unsigned w){surfaces.prepare(target,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images){const auto image=linear(commands,images[0],w,4);texture_region all;all.size=MTLSizeMake(w,4,1);colors.upload(commands,f,make_upload(commands,words),{},image,all);if(count>1)samples.import(commands,image,images[0]);});};
        const auto target=cold?surfaces.acquire({output.range.address,{5,3,1,surface_color::argb8,false},33,{}}):surfaces.acquire({0x6000,target_layout,33,{}});
        const auto seed_target=[&]
        {
            surfaces.prepare(target,surface_initialization::background);
            surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images)
            {
                const auto image=linear(commands,images[0],8,8);texture_region all;all.size=MTLSizeMake(8,8,1);
                colors.upload(commands,surface_color::argb8,make_upload(commands,background),{},image,all);
                if(count>1)samples.import(commands,image,images[0]);
            });
        };
        if(donor_target)
        {
            for(unsigned i=0;i<background.size();++i)background[i]=std::byte((i*53+71)&255);
            seed_target();
            for(unsigned y=0;y<4;++y)std::copy_n(background.begin()+y*32,16,first_words.begin()+y*16);
        }
        const auto second=surfaces.acquire({0x1000,{second_width/factors.x,source_height/factors.y,count,surface_color::gb8,false},28,{}});
        if(frozen)
        {
            const auto donor=donor_target?target:surfaces.acquire({0x2000,{4/factors.x,4/factors.y,count,surface_color::argb8,false},16,{}});
            const auto other=surfaces.acquire({0x3000,other_layout,2*second_width,{}});
            std::vector<std::byte> other_words(8*second_width);for(unsigned y=0;y<4;++y)std::copy_n(vm_input.begin()+y*28,2*second_width,other_words.begin()+y*2*second_width);
            if(partial||discarded==2)for(auto& byte:other_words)byte^=std::byte{0x69};
            if(!donor_target)seed(donor,surface_color::argb8,first_words,4);
            seed(other,surface_color::gb8,other_words,second_width);
            const clear_rectangle full{0,0,4/factors.x,4/factors.y};
            if(discarded)
            {
                const auto row=discarded==1?4/factors.y:0;
                surfaces.defer_copy(donor,full,first,0,row);
                surfaces.defer_copy(other,{0,0,12/factors.x,4/factors.y},second,0,row);
            }
            else
            {
                if(interleaved)surfaces.defer_copy(donor,full,first,0,0);
                surfaces.defer_copy(other,{0,0,(partial?8u:12u)/factors.x,4/factors.y},second,0,0);
                surfaces.defer_copy(donor,(interleaved||partial)?clear_rectangle{0,0,2/factors.x,4/factors.y}:full,first,0,0);
            }
            if(spill){if(!donor_target)surfaces.spill(donor);surfaces.spill(other);}
            if(partial)for(unsigned y=0;y<4;++y)
            {
                std::copy_n(other_words.begin()+y*2*second_width,16,vm_input.begin()+y*28);
                memory->allowed_input_reads.push_back({0x1010u+y*28,8});
            }
            if(discarded)for(unsigned y=0;y<4;++y)
            {
                if(discarded==1)memory->allowed_input_reads.push_back({0x1000u+y*28,24});
                else std::copy_n(other_words.begin()+y*24,24,vm_input.begin()+y*28);
            }
        }
        else
        {
            if(failure==10)surfaces.prepare(first);
            else {seed(first,surface_color::argb8,first_words,4);surfaces.flush(first);}
            surfaces.prepare(second);if(spill){surfaces.spill(first);surfaces.spill(second);}
        }
        for(unsigned y=0;y<4;++y){if(!discarded)std::copy_n(first_words.begin()+y*16,(interleaved||partial)?8:16,vm_input.begin()+y*28+4);std::copy_n(vm_input.begin()+y*28,24,packed_input.begin()+y*24);}
        if(!cold&&!donor_target)seed_target();
        cached_surface_ref filler;
        if(failure==11)
        {
            filler=surfaces.acquire({0x9000,target_layout,33,{}});
            surfaces.prepare(filler);
        }
        scaled_image_plan plan{input,output,scaled_image_format::argb8,scaled_image_format::argb8,6,4,9,5,2,1,true,false,true};auto canonical=plan;canonical.source=plan_guest_surface(0,6,4,4);canonical.destination=plan_guest_surface(0,5,3,4);
        const auto patch=reference(canonical,packed_input,std::vector<std::byte>(60));auto expected=cold?patch:background;if(!cold)for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*20,20,expected.begin()+((y+1)*8+1)*4);
        const auto read_target=[&]{const auto data=make_buffer(ctx,expected.size(),buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{target,false}},[&](context& commands,auto images){const auto image=linear(commands,images[0],cold?5:8,cold?3:8);if(image!=images[0])samples.expand(commands,images[0],image);texture_region all;all.size=MTLSizeMake(cold?5:8,cold?3:8,1);colors.download(commands,surface_color::argb8,image,all,data,{});});return std::vector<std::byte>(static_cast<const std::byte*>(data.contents),static_cast<const std::byte*>(data.contents)+expected.size());};
        const auto before=surfaces.status(target);const auto vm=memory->access->copy();const auto bytes=memory->read_bytes,writes=std::size_t(memory->writes),notifications=std::size_t(memory->access->notifications);memory->forbid_pixels=true;
        const auto raw_backups=surfaces.transaction_statistics().raw_backup_submissions;
        const auto validate=[metadata]{if(!*metadata)throw std::range_error("Overlap metadata changed");};
        if(!frozen&&failure==10)
        {
            bool failed=false;try{transfer->execute(plan,30s,{},validate);}catch(const std::invalid_argument& error){failed=std::string_view(error.what()).find("unique native byte authority")!=std::string_view::npos;}
            require(failed&&!transfer->has_pending()&&!surfaces.status(target).pending&&surfaces.status(target).content_generation==before.content_generation&&memory->access->copy()==vm&&memory->read_bytes==bytes&&memory->writes==writes&&memory->access->notifications==notifications,"Ambiguous clean epochs reached output encoding orVM capture");++ambiguous;continue;
        }
        const bool cpu_writer=(failure>=1&&failure<=3)||(partial&&failure==10);
        const bool topology_change=discarded&&failure==10;
        const auto writer_address=failure==1?0x1004u:failure==2?0x1000u:failure==10?0x1014u:output.range.address;
        if(failure==11)
        {
            bool refused=false, encoded=false;
            try{transfer->execute(plan,30s,[&](context&){encoded=true;},validate);}
            catch(const surface_budget_error&){refused=true;}
            require(refused&&!encoded&&!transfer->has_pending()&&!surfaces.status(target).pending&&
                surfaces.status(target).content_generation==before.content_generation&&
                surfaces.status(first).deferred_copies==1&&surfaces.status(second).deferred_copies==1&&
                read_target()==background&&memory->access->copy()==vm,"Donor quota refusal changed the old allocation or reached encoding");
            ++pool_refusals;
            surfaces.erase(filler);
            transfer->execute(plan,30s,{},validate);
        }
        else if(failure==9)
        {
            (void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();std::vector<id<MTLBuffer>> blockers;
            for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate}){auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}bool failed=false;
            try{transfer->execute(plan,30s,{},validate);}catch(const std::runtime_error&){failed=true;}
            require(failed&&!transfer->has_pending()&&!surfaces.status(target).pending&&surfaces.status(target).content_generation==before.content_generation&&memory->access->copy()==vm,"Overlap quota refusal published output orlost rollback");blockers.clear();transfer->execute(plan,30s,{},validate);
        }
        else
        {
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};std::optional<submission> token;bool failed=false;
            for(unsigned attempt=0;attempt<4;++attempt)
            {
                try{transfer->execute(plan,1ms,[&](context& commands){token=commands.recording_submission();if(failure==7)throw std::range_error("Overlap encode rejected");if(topology_change)surfaces.cancel_deferred_copies(first);commands.wait_event(gate,1);},validate);}catch(const std::runtime_error&){failed=true;}
                if(failed&&!token&&!transfer->has_pending()&&surfaces.status(first).pending)
                {require(surfaces.status(target).content_generation==before.content_generation&&memory->access->copy()==vm,"Overlap spill prefix published target early");surfaces.synchronize(30s);failed=false;continue;}break;
            }
            if(failure==7)require(failed&&token&&token->ready()&&!transfer->has_pending()&&!surfaces.status(target).pending,"Overlap encode failure left pending output");
            else
            {
                require(failed&&token&&!token->ready()&&transfer->has_pending()&&surfaces.status(target).pending&&surfaces.status(target).content_generation==before.content_generation&&memory->access->copy()==vm,"Overlap output lacks actual gated pending work orpublished early");
                if(cpu_writer)memory->raw->write({writer_address,1},std::array{std::byte{0x5c}});
                if(failure==4)*metadata=false;if(failure==5)require(token->cancel_pending(),"Overlap queued output cancellation failed");if(failure==6)transfer.reset();if(failure==8)*refuse=true;
                gate.signaledValue=1;failed=false;try{if(transfer)transfer->synchronize(30s);else surfaces.synchronize(30s);}catch(const std::runtime_error& error){failed=true;if(topology_change)require(std::string_view(error.what()).find("queued source ownership changed")!=std::string_view::npos,"Ownership change did not reach publication validation");}
                if(failure==8){require(failed&&transfer->has_pending()&&surfaces.status(target).pending&&surfaces.status(target).content_generation==before.content_generation,"Overlap dispatcher refusal lost pending output");*refuse=false;transfer->synchronize(30s);failed=false;}
                require(failed==((failure>=1&&failure<=6)||(partial&&failure==10)||topology_change),"Overlap completion accepted stale authority orrejected valid retry");if(failure==5)ctx.recover_completed_errors();
            }
        }
        const bool rejected=(failure>=1&&failure<=7)||(partial&&failure==10)||topology_change;
        require(!surfaces.status(target).pending&&(!transfer||!transfer->has_pending())&&!surfaces.status(first).dirty&&!surfaces.status(second).dirty,"Overlap completion lost source cleanliness orpending retirement");
        if(frozen)require(surfaces.status(first).deferred_copies==(topology_change?0u:1+unsigned(interleaved))&&surfaces.status(second).deferred_copies==1&&!surfaces.status(first).initialized&&!surfaces.status(second).initialized,"Frozen transactions consumed the original queued plans");
        const auto hole_bytes=discarded==1?96u:partial?32u:0u;
        require(memory->read_bytes==bytes+hole_bytes*((failure==9||failure==11)?2:1)&&memory->writes==writes&&memory->access->notifications==notifications+unsigned(cpu_writer),"Overlap transfer captured bytes outside VM holes or performed unintended stores");
        if(donor_target)require(surfaces.transaction_statistics().raw_backup_submissions==raw_backups,"Donor version used an in-place raw backup");
        if(rejected)
        {
            require(surfaces.status(target).content_generation==before.content_generation&&surfaces.status(target).initialized==before.initialized&&surfaces.status(target).dirty==before.dirty&&!surfaces.status(target).faulted,"Rejected overlap output changed prior authority/generation");if(!cold&&failure!=3)require(read_target()==background,"Rejected overlap output lost native target background");
            auto wanted=vm;if(cpu_writer)wanted[writer_address]=std::byte{0x5c};require(memory->access->copy()==wanted,"Rejected overlap output overwrote writer orVM padding");
        }
        else
        {
            require(transfer&&transfer->native_completed()==1&&transfer->vm_completed()==0&&surfaces.status(target).content_generation==before.content_generation+1&&read_target()==expected&&memory->access->copy()==vm,"Overlap native result/background/retry differs");memory->forbid_pixels=false;surfaces.prepare_cpu_read(output.range);auto wanted=vm;
            for(unsigned y=0;y<(cold?3u:8u);++y)std::copy_n(expected.begin()+y*(cold?20:32),cold?20:32,wanted.begin()+(cold?output.range.address:0x6000)+y*33);require(memory->access->copy()==wanted,"Overlap CPU observation changed unrelated source/padding/VM");
            if(donor_target)
            {
                surfaces.cancel_deferred_copies(second);
                const auto data=make_buffer(ctx,source_height*16,buffer_storage::shared);
                surfaces.draw(std::array{cached_surface_use{first,false}},[&](context& commands,auto images)
                {
                    const auto image=linear(commands,images[0],4,source_height);
                    if(image!=images[0])samples.expand(commands,images[0],image);
                    texture_region all;all.size=MTLSizeMake(4,source_height,1);
                    colors.download(commands,surface_color::argb8,image,all,data,{});
                });
                const auto* old=static_cast<const std::byte*>(data.contents)+(source_height-4)*16;
                require(std::memcmp(old,first_words.data(),first_words.size())==0,"Queued consumer read the new target instead of its original donor");
                ++old_donors;
            }
        }
        ++cases;
        }
        catch(const std::exception& error){throw std::runtime_error("Overlap transaction samples="+std::to_string(count)+" cold="+std::to_string(cold)+" spill="+std::to_string(spill)+" failure="+std::to_string(failure)+": "+error.what());}
    }
    if(donor_target)
    {
        require(old_donors==(discarded==1?24u:18u)&&pool_refusals==(discarded==1?6u:0u),"Donor pixel or local pool coverage incomplete");
        std::cout<<"Versioned scaled donor mode "<<discarded<<":"<<cases<<" transactions,"<<old_donors<<" exact old donor readbacks,"<<pool_refusals<<" local pool refusals and retries passed\n";
    }
    else if(discarded)std::cout<<(discarded==1?"RAM-only queued transactions:":"Shadowed queued transactions:")<<cases<<" real cases, including eight queued ownership changes, exact input capture, rollback and retry passed\n";
    else if(partial)std::cout<<"Partial deferred transactions:"<<cases<<" real cases, including eight CPU writers outside both native source watches, exact RAM hole capture, rollback and retry passed\n";
    else std::cout<<(interleaved?"Interleaved deferred transactions:":frozen?"Frozen deferred transactions:":"Overlapping clean transactions:")<<cases<<" real gated one-four-sample/resident-spill/existing-cold cases;both overlapping-source CPUwriters,target writer,metadata,queued cancellation,abandoned owner,encode refusal,CPUdispatcher andactual Shared-Private quota refusal/retry;exact rollback/generation/background/CPU visibility,no nativeVM capture passed\n";
    require(cases==(donor_target?(discarded==1?72u:66u):(partial||discarded)?88u:80u)&&ambiguous==(frozen?0u:8u),"Overlap transaction/control coverage incomplete");
    if(!frozen)std::cout<<"Overlapping clean ambiguous authority:"<<ambiguous<<" equal-epoch source pairs refused before output encoding/VMpixel capture/generation changes passed\n";
}

void depth_scaled_image_cases(bool stencil_family)
{
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);sample_transfer samples(ctx);
    unsigned profiles=0;std::size_t exact_bytes=0,raw_bytes=0,hole_bytes=0;
    for(unsigned type=1;type<=4;++type)for(unsigned out:{2u,4u})for(unsigned layout=0;layout<4;++layout)for(unsigned kind=0;kind<3;++kind)
    for(unsigned count:{1u,2u,4u})for(bool clean:{false,true})for(bool cold:{false,true})for(unsigned flags=0;flags<8;++flags)
    {
        if(bool(type%2==0)!=stencil_family||(layout==3&&kind))continue;
        @autoreleasepool
        {
            try
            {
            const auto format=static_cast<surface_depth>(type);const auto metadata=describe_surface_depth(format);const auto in=metadata.guest_bytes;
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
            const auto morton=layout==1?std::optional<swizzle_region>{{{8,8,1},{1,2,0},{4,4,1}}}:std::nullopt;
            const auto tile=layout>=2?std::optional<banked_tile_layout>{{0x10000,0x40000,256,1,0x1200+in}}:std::nullopt;
            auto source=plan_guest_surface(tile?tile->address+tile->offset:0x1000,4,4,in,morton?0:tile?256:4*in+1,morton);if(tile)source=attach_banked_tile(source,*tile);
            std::vector<std::byte> input(16*in);std::array<std::uint32_t,16> native_depth{};std::array<unsigned char,16> native_stencil{};
            for(unsigned i=0;i<16;++i)
            {
                std::uint32_t guest=0,bits=0;const auto s=static_cast<unsigned char>(37+i*53);native_stencil[i]=s;
                if(type==1){bits=i==0?0:i==15?65535:(i*7919+73)&65535;guest=bits;}
                else if(type==2){const float value=i==0?0.f:i==15?1.f:.0625f+float(i)/32;bits=std::bit_cast<std::uint32_t>(value);guest=(static_cast<std::uint32_t>(value*16777215.f)<<8)|s;}
                else if(type==3){const auto word=i==15?0x7000u:(i*0x5b1u+0x123u)&0x6fffu;const float value=std::ldexp(1.f+float(word%4096)/4096,int(word/4096)-7);bits=std::bit_cast<std::uint32_t>(value);guest=word;}
                else {bits=i==0?0:i==15?std::bit_cast<std::uint32_t>(1.f):0x3e800000u+i*0x19017u;guest=((bits>>7)<<8)|s;}
                native_depth[i]=bits;for(unsigned lane=0;lane<in;++lane)input[i*in+lane]=std::byte(guest>>(8*(in-1-lane)));
            }
            if(kind==2)for(unsigned y=0;y<4;++y)for(unsigned byte=2*in;byte<4*in;++byte)input[y*4*in+byte]=std::byte((y*43+byte*67+19)&255);
            struct depth_source {cached_surface_ref ref;unsigned width;std::vector<std::byte> expected;};std::vector<depth_source> sources;std::vector<cached_surface_ref> pieces;
            const auto expanded=[&](context& commands,id<MTLTexture> image,unsigned width)
            {if(count==1)return image;auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:width height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;return make_texture(commands,d);};
            for(unsigned side=0;side<(kind?2u:1u);++side)
            {
                const unsigned width=kind?2:4;const bool color=kind==2&&side;const auto color_format=in==2?surface_color::rgb565:surface_color::argb8;
                cached_surface_description description;
                if(layout==3)description={tile->address+tile->offset,{4/factors.x,4/factors.y,count,format,false},256,{},tile};
                else
                {
                    const auto swizzle=layout==1?std::optional<swizzle_region>{{{width,4,1},{},{width,4,1}}}:std::nullopt;
                    const auto geometry=plan_guest_surface(0,width,4,in,swizzle?0:width*in+1,swizzle);std::vector<unsigned> addresses(geometry.range.size);
                    for(unsigned i=0;i<addresses.size();++i)addresses[i]=0xd0000+side*0x1000+i;
                    for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<width*in;++byte)
                    {
                        const auto x=side*2+byte/in,lane=byte%in,index=(y*4+x)*in+lane;auto at=offset(source,x,y)+lane;if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                        const auto actual=(tile?tile->address:source.range.address)+static_cast<unsigned>(at);
                        if(kind==1&&(y+byte+side)%3==0){input[index]^=std::byte{0x5b};memory->raw->write({actual,1},std::span{&input[index],1});memory->allowed_input_reads.push_back({actual,1});}
                        else addresses[offset(geometry,byte/in,y)+lane]=actual+(side?0x100000:0);
                    }
                    std::vector<guest_storage_segment> spans;for(unsigned i=0;i<addresses.size();++i){if(!spans.empty()&&std::uint64_t(spans.back().address)+spans.back().size==addresses[i])++spans.back().size;else spans.push_back({addresses[i],i,1});}
                    description={spans.front().address,{width/factors.x,4/factors.y,count,color?surface_format{color_format}:surface_format{format},false},swizzle?0:geometry.row_pitch,swizzle,{},spans};
                }
                const auto piece=surfaces.acquire(description);surfaces.prepare(piece,surface_initialization::background);
                std::vector<std::byte> depth_plane(256*4,std::byte{0xa5}),stencil_plane(256*4,std::byte{0xa5}),expected;
                for(unsigned y=0;y<4;++y)for(unsigned x=0;x<width;++x){const auto i=y*4+side*2+x;for(unsigned lane=0;lane<metadata.depth_bytes;++lane){const auto b=std::byte(native_depth[i]>>(lane*8));depth_plane[y*256+x*metadata.depth_bytes+lane]=b;expected.push_back(b);}stencil_plane[y*256+x]=std::byte(native_stencil[i]);}
                if(metadata.stencil)for(unsigned y=0;y<4;++y)for(unsigned x=0;x<width;++x)expected.push_back(std::byte(native_stencil[y*4+side*2+x]));
                surfaces.draw(std::array{cached_surface_use{piece,true}},[&](context& commands,auto images)
                {
                    const auto image=expanded(commands,images[0],width);texture_region all;all.size=MTLSizeMake(width,4,1);
                    if(color){std::vector<std::byte> words(width*4*in);for(unsigned y=0;y<4;++y)std::copy_n(input.begin()+(y*4+side*2)*in,width*in,words.begin()+y*width*in);colors.upload(commands,color_format,make_upload(commands,words),{},image,all);}
                    else {buffer_to_texture(commands,make_upload(commands,depth_plane),0,image,all,256,0,texture_plane::depth);if(metadata.stencil)buffer_to_texture(commands,make_upload(commands,stencil_plane),0,image,all,256,0,texture_plane::stencil);}
                    if(count>1)samples.import(commands,image,images[0]);
                });
                if(clean)surfaces.flush(piece);if(flags&1)surfaces.spill(piece);pieces.push_back(piece);if(!color)sources.push_back({piece,width,std::move(expected)});
            }
            const auto read_source=[&](const depth_source& item)
            {
                const auto depth_buffer=make_buffer(ctx,256*4,buffer_storage::shared),stencil_buffer=metadata.stencil?make_buffer(ctx,256*4,buffer_storage::shared):nil;
                surfaces.draw(std::array{cached_surface_use{item.ref,false}},[&](context& commands,auto images){const auto image=expanded(commands,images[0],item.width);if(count>1)samples.expand(commands,images[0],image);texture_region all;all.size=MTLSizeMake(item.width,4,1);texture_to_buffer(commands,image,all,depth_buffer,0,256,0,texture_plane::depth);if(metadata.stencil)texture_to_buffer(commands,image,all,stencil_buffer,0,256,0,texture_plane::stencil);});
                std::vector<std::byte> result;const auto* d=static_cast<const std::byte*>(depth_buffer.contents);for(unsigned y=0;y<4;++y)result.insert(result.end(),d+y*256,d+y*256+item.width*metadata.depth_bytes);
                if(metadata.stencil){const auto* s=static_cast<const std::byte*>(stencil_buffer.contents);for(unsigned y=0;y<4;++y)result.insert(result.end(),s+y*256,s+y*256+item.width);}return result;
            };
            for(const auto& item:sources){require(read_source(item)==item.expected,"Depth source sample placement/native bits differ before scaled input");if(flags&1)surfaces.spill(item.ref);}
            std::sort(memory->allowed_input_reads.begin(),memory->allowed_input_reads.end(),[](auto a,auto b){return a.address<b.address;});std::vector<guest_surface_range> allowed;
            for(const auto& r:memory->allowed_input_reads){if(!allowed.empty()&&std::uint64_t(allowed.back().address)+allowed.back().size==r.address)allowed.back().size+=r.size;else allowed.push_back(r);}memory->allowed_input_reads=std::move(allowed);std::size_t holes=0;for(const auto& r:memory->allowed_input_reads)holes+=r.size;
            const auto df=out==2?surface_color::rgb565:surface_color::argb8;const auto destination=cold?cached_surface_ref{}:surfaces.acquire({0x6000,{8/factors.x,8/factors.y,count,df,false},8*out+1,{}});std::vector<std::byte> background(64*out,std::byte{0xa5});
            if(destination){surfaces.prepare(destination,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{destination,true}},[&](context& commands,auto images){auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);}texture_region all;all.size=MTLSizeMake(8,8,1);colors.upload(commands,df,make_upload(commands,background),{},image,all);if(count>1)samples.import(commands,image,images[0]);});}
            const auto target=plan_guest_surface(0x6000+(8*out+1)+out,5,3,out,8*out+1);scaled_image_plan plan{source,target,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,4,4,7,5,1,1,bool(flags&1),bool(flags&2),bool(flags&4),tile};
            auto packed=plan;packed.source=plan_guest_surface(0,4,4,in);packed.destination=plan_guest_surface(0,5,3,out);packed.source_tile.reset();const auto patch=reference(packed,input,std::vector<std::byte>(15*out));
            const auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes),notifications=std::size_t(memory->access->notifications);memory->forbid_pixels=true;transfer.execute(plan);
            require(transfer.native_completed()==1&&transfer.vm_completed()==0&&memory->read_bytes-reads==holes&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Depth scaled input bypassed native byte authority orcaptured native/target VMpixels");
            for(const auto& item:sources){require(read_source(item)==item.expected,"Depth/stencil native source bits changed during scaled packing");raw_bytes+=2*item.expected.size();}
            for(const auto& piece:pieces)require(surfaces.status(piece).dirty==!clean,"Depth/color source clean-dirty authority changed");memory->forbid_pixels=false;
            if(destination){for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,background.begin()+((y+1)*8+1)*out);const auto raw=make_buffer(ctx,background.size(),buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{destination,false}},[&](context& commands,auto images){auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);samples.expand(commands,images[0],image);}texture_region all;all.size=MTLSizeMake(8,8,1);colors.download(commands,df,image,all,raw,{});});require(std::memcmp(raw.contents,background.data(),background.size())==0,"Depth packed native output/background differs from independent word/filter oracle");exact_bytes+=background.size();}
            else {surfaces.prepare_cpu_read(target.range);auto expected=vm;for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,expected.begin()+target.range.address+y*target.row_pitch);require(memory->access->copy()==expected,"Depth packed cold-output observation changed source/padding/VM gaps");exact_bytes+=patch.size();}
            hole_bytes+=holes;++profiles;
            }
            catch(const std::exception& error){throw std::runtime_error("Depth scale type="+std::to_string(type)+" out="+std::to_string(out)+" layout="+std::to_string(layout)+" kind="+std::to_string(kind)+" samples="+std::to_string(count)+" clean="+std::to_string(clean)+" cold="+std::to_string(cold)+" flags="+std::to_string(flags)+": "+error.what());}
        }
    }
    std::cout<<"Native depth scaled images "<<(stencil_family?"Z24":"Z16")<<":"<<profiles<<" raw depth/stencil seed/native-plane oracles;single/two-source/RAM-XORholes/mixed-color,oddpitch/Morton/banked,1-2-4samples,dirty-clean/resident-spill,existing-cold,flips/filter/clip;"<<exact_bytes<<" exact output-background bytes,"<<raw_bytes<<" exact raw source depth-stencil bytes,"<<hole_bytes<<" only missing RAMbytes captured;no native/target VMcapture/write/notifications andwholeVM padding exact passed\n";
}

void depth_scaled_image_transactions(bool stencil_family)
{
    using namespace std::chrono_literals;context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_depth_transfer depths(ctx);surface_color_transfer colors(ctx);sample_transfer samples(ctx);unsigned cases=0;
    for(unsigned type=1;type<=4;++type)for(unsigned count:{1u,4u})for(bool clean:{false,true})for(bool cold:{false,true})for(unsigned failure=0;failure<9;++failure)
    {
        if(bool(type%2==0)!=stencil_family)continue;
        @autoreleasepool
        {
            try
            {
            const auto format=static_cast<surface_depth>(type);const auto metadata=describe_surface_depth(format);const auto in=metadata.guest_bytes;
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;auto refuse=std::make_shared<bool>(false);const cpu_capture_dispatch capture=[access=memory->access,refuse](const auto& work){if(*refuse)throw std::range_error("Depth output dispatcher refused");access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);const auto source=surfaces.acquire({0x1000,{4/factors.x,2/factors.y,count,format,false},4*in,{}}),target=surfaces.acquire({0x6000,{cold?3u:4/factors.x,cold?2u:4/factors.y,cold?1u:count,surface_color::argb8,false},17,{}});
            std::vector<std::byte> input(8*in),background(64,std::byte{0xa5});for(unsigned i=0;i<8;++i){const auto s=(17+i*53)&255;const std::uint32_t word=type==1?(i*7919+13)&65535:type==2?((i&1?0xffffffu:0)<<8)|s:type==3?0x1000u+i*0x500:((std::bit_cast<std::uint32_t>(float(i+1)/16)>>7)<<8)|s;for(unsigned lane=0;lane<in;++lane)input[i*in+lane]=std::byte(word>>(8*(in-1-lane)));}
            surfaces.prepare(source,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:2 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;image=make_texture(commands,d);}texture_region all;all.size=MTLSizeMake(4,2,1);depths.upload(commands,format,make_upload(commands,input),{},image,all);if(count>1)samples.import(commands,image,images[0]);});
            if(clean)surfaces.flush(source);if(failure&1)surfaces.spill(source);
            if(!cold){surfaces.prepare(target,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images){auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);}texture_region all;all.size=MTLSizeMake(4,4,1);colors.upload(commands,surface_color::argb8,make_upload(commands,background),{},image,all);if(count>1)samples.import(commands,image,images[0]);});}
            const auto before=surfaces.status(target);const auto vm=memory->access->copy();const auto reads=memory->read_bytes;
            scaled_image_plan plan{plan_guest_surface(0x1000,4,2,in),plan_guest_surface(0x6000,3,2,4,17),in==2?scaled_image_format::rgb565:scaled_image_format::argb8,scaled_image_format::argb8,4,2,5,3,1,1,true,true,true};auto packed=plan;packed.source=plan_guest_surface(0,4,2,in);packed.destination=plan_guest_surface(0,3,2,4);const auto patch=reference(packed,input,std::vector<std::byte>(24));auto expected=background;for(unsigned y=0;y<2;++y)std::copy_n(patch.begin()+y*12,12,expected.begin()+y*16);
            const auto readback=[&]{const auto raw=make_buffer(ctx,64,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{target,false}},[&](context& commands,auto images){auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);samples.expand(commands,images[0],image);}texture_region all;all.size=MTLSizeMake(4,4,1);colors.download(commands,surface_color::argb8,image,all,raw,{});});const auto bytes=map_buffer(raw,0,64);return std::vector<std::byte>(bytes.begin(),bytes.end());};
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};std::optional<submission> token;auto valid=std::make_shared<bool>(true);const auto validate=[valid,access=memory->access]{require(guest_memory_epoch::current(*access)!=nullptr,"Depth metadata escaped producer exclusion");if(!*valid)throw scaled_image_mapping_conflict();};
            std::vector<id<MTLBuffer>> blockers;if(failure==8){(void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate}){auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}}
            memory->forbid_pixels=true;bool failed=false;const auto execute=[&]{try{transfer->execute(plan,failure==8?30s:1ms,[&](context& commands){token=commands.recording_submission();if(failure==6)throw std::range_error("Depth encode refused");commands.wait_event(gate,1);},validate);}catch(const std::runtime_error&){failed=true;}};execute();
            for(unsigned retry=0;!token&&surfaces.status(source).pending&&retry<4;++retry){surfaces.synchronize();failed=false;execute();}
            require(failed&&surfaces.status(target).content_generation==before.content_generation&&memory->access->copy()==vm,"Depth scaled output published before actual GPU completion");
            if(failure==8){require(!transfer->has_pending()&&(!token||(token->ready()&&token->cancelled())),"Depth quota refusal lost native owner");blockers.clear();transfer->execute(plan,30s,{},validate);failed=false;}
            else if(failure==6)require(token&&token->cancelled()&&!transfer->has_pending()&&!surfaces.status(target).pending,"Depth encode cancellation retained output");
            else
            {
                require(token&&transfer->has_pending()&&surfaces.status(target).pending&&!token->ready(),"Depth transaction lacks real gated native output");
                if(failure==1||failure==2)memory->raw->write({failure==1?0x1000u:0x6000u,1},std::array{std::byte{0x5c}});if(failure==3)*valid=false;if(failure==4)require(token->cancel_pending(),"Depth queued cancellation refused");if(failure==5)transfer.reset();if(failure==7)*refuse=true;
                gate.signaledValue=1;failed=false;try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                if(failure==7){require(failed&&transfer->has_pending()&&surfaces.status(target).pending,"Depth dispatcher refusal dropped pending output");*refuse=false;transfer->synchronize();failed=false;}if(failure==4)ctx.recover_completed_errors();
            }
            const bool rejected=failure>=1&&failure<=6;require(failed==rejected&&!surfaces.status(target).pending&&!surfaces.status(target).faulted&&(!transfer||!transfer->has_pending())&&surfaces.status(target).content_generation==before.content_generation+unsigned(!rejected),"Depth native completion/rollback metadata differs");
            if(cold){if(rejected)require(!surfaces.status(target).initialized&&!surfaces.status(target).dirty,"Rejected depth transfer initialized cold output");}else if(failure!=2)require(readback()==(rejected?background:expected),"Depth native rollback/output lost exact background bytes");
            if(failure==1||failure==2)require(memory->raw->read({failure==1?0x1000u:0x6000u,1}).bytes[0]==std::byte{0x5c},"Depth native output overwrote CPU writer");else require(memory->access->copy()==vm,"Depth native execute wrote VMpixels");
            require(memory->read_bytes==reads&&surfaces.status(source).dirty==!clean,"Depth native preparation recaptured VMpixels orchanged source authority");memory->forbid_pixels=false;
            if(cold&&!rejected){surfaces.prepare_cpu_read(plan.destination.range);auto expected_vm=vm;for(unsigned y=0;y<2;++y)std::copy_n(patch.begin()+y*12,12,expected_vm.begin()+0x6000+y*17);require(memory->access->copy()==expected_vm,"Depth cold observation changed source/padding");}++cases;
            }
            catch(const std::exception& error){throw std::runtime_error("Depth transaction type="+std::to_string(type)+" samples="+std::to_string(count)+" clean="+std::to_string(clean)+" cold="+std::to_string(cold)+" failure="+std::to_string(failure)+": "+error.what());}
        }
    }
    std::cout<<"Native depth transactions "<<(stencil_family?"Z24":"Z16")<<":"<<cases<<" actual gated dirty-clean/resident-spill,1-4sample inputs/existing-cold outputs;CPUwriter/metadata conflicts,queued-encode cancellation,abandoned helper owner,CPUdispatcher andShared-Private quota refusal/retry;exact rollback/generation/background/CPU visibility,zero VMpixel reads/writes duringexecute passed\n";
}

void assembled_scaled_image_cases(bool mixed=false,bool clean=false)
{
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);sample_transfer samples(ctx);
    unsigned profiles=0;std::size_t exact_bytes=0,hole_bytes=0;
    for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned kind=0;kind<4;++kind)for(unsigned layout=0;layout<3;++layout)
    for(bool source_morton:{false,true})for(unsigned count:{1u,2u,4u})for(bool cold:{false,true})for(unsigned flags=0;flags<8;++flags)
    {
        if(kind==3&&source_morton)continue;
        if(mixed&&kind==3)continue;
        @autoreleasepool
        {
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};
            auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto morton=layout==1?std::optional<swizzle_region>{{{8,8,1},{1,2,0},{4,4,1}}}:std::nullopt;
            const auto tile=layout==2?std::optional<banked_tile_layout>{{0x10000,0x40000,256,1,0x1200+in}}:std::nullopt;
            auto source=plan_guest_surface(tile?tile->address+tile->offset:0x1000,4,4,in,morton?0:tile?256:4*in+1,morton);
            if(tile)source=attach_banked_tile(source,*tile);
            std::vector<std::byte> input(16*in);for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+i/3+109)&255);
            const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
            const auto format=kind==0?(in==2?surface_color::rgb565:surface_color::argb8):kind==1?surface_color::b8:kind==2?surface_color::gb8:in==2?surface_color::gb8:surface_color::abgr8;
            const auto bpp=surface_guest_bytes(surface_format{format}),width=(kind==3?4:2)*in/bpp;
            std::vector<cached_surface_ref> pieces;
            const auto seed=[&](const cached_surface_ref& ref,surface_color sf,std::span<const std::byte> bytes,unsigned w,unsigned h)
            {
                surfaces.prepare(ref,surface_initialization::background);
                surfaces.draw(std::array{cached_surface_use{ref,true}},[&](context& commands,auto images)
                {
                    auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:w height:h mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);}
                    texture_region all;all.size=MTLSizeMake(w,h,1);colors.upload(commands,sf,make_upload(commands,bytes),{},image,all);if(count>1)samples.import(commands,image,images[0]);
                });
            };
            for(unsigned side=0;side<(kind==3?1u:2u);++side)
            {
                if(kind==3)
                {
                    auto piece=surfaces.acquire({tile?tile->address+tile->offset:source.range.address,{4/factors.x,4/factors.y,count,format,false},source.morton?0:source.row_pitch,source.morton,tile});
                    seed(piece,format,input,4,4);if(clean)surfaces.flush(piece);if(flags&1)surfaces.spill(piece);pieces.push_back(piece);continue;
                }
                const auto swizzle=source_morton?std::optional<swizzle_region>{{{width,4,1},{},{width,4,1}}}:std::nullopt;
                const auto geometry=plan_guest_surface(0,width,4,bpp,swizzle?0:width*bpp+1,swizzle);
                std::vector<unsigned> addresses(geometry.range.size);for(unsigned i=0;i<addresses.size();++i)addresses[i]=0xd0000+side*0x1000+i;
                std::vector<std::byte> bytes(8*in);
                for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<2*in;++byte)
                {
                    const auto x=side*2+byte/in,lane=byte%in;
                    auto at=offset(source,x,y)+lane;if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                    const unsigned actual=(tile?tile->address:source.range.address)+unsigned(at);
                    if(mixed&&(y+byte+side)%3==0)
                    {
                        memory->raw->write({actual,1},std::span{&input[(y*4+side*2)*in+byte],1});memory->allowed_input_reads.push_back({actual,1});
                    }
                    else addresses[offset(geometry,byte/bpp,y)+byte%bpp]=actual+(side?0x100000:0);
                    bytes[y*2*in+byte]=input[(y*4+side*2)*in+byte];
                }
                std::vector<guest_storage_segment> spans;
                for(unsigned i=0;i<addresses.size();++i){if(!spans.empty()&&std::uint64_t(spans.back().address)+spans.back().size==addresses[i])++spans.back().size;else spans.push_back({addresses[i],i,1});}
                auto piece=surfaces.acquire({spans.front().address,{width/factors.x,4/factors.y,count,format,false},swizzle?0:geometry.row_pitch,swizzle,{},spans});seed(piece,format,bytes,width,4);pieces.push_back(piece);
                if(clean)surfaces.flush(piece);
                if(flags&1)surfaces.spill(piece);
            }
            std::sort(memory->allowed_input_reads.begin(),memory->allowed_input_reads.end(),[](auto a,auto b){return a.address<b.address;});
            std::vector<guest_surface_range> allowed;for(const auto& r:memory->allowed_input_reads){if(!allowed.empty()&&std::uint64_t(allowed.back().address)+allowed.back().size==r.address)allowed.back().size+=r.size;else allowed.push_back(r);}memory->allowed_input_reads=std::move(allowed);
            std::size_t requested_holes=0;for(const auto& r:memory->allowed_input_reads)requested_holes+=r.size;
            const auto df=out==2?surface_color::rgb565:surface_color::argb8;
            auto destination=cold?cached_surface_ref{}:surfaces.acquire({0x6000,{8/factors.x,8/factors.y,count,df,false},8*out+1,{}});
            std::vector<std::byte> background(64*out,std::byte{0xa5});if(destination)seed(destination,df,background,8,8);
            const auto target=plan_guest_surface(0x6000+(8*out+1)+out,5,3,out,8*out+1);
            scaled_image_plan plan{source,target,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,4,4,7,5,1,1,bool(flags&1),bool(flags&2),bool(flags&4),tile};
            auto packed=plan;packed.source=plan_guest_surface(0,4,4,in);packed.destination=plan_guest_surface(0,5,3,out);packed.source_tile.reset();
            const auto patch=reference(packed,input,std::vector<std::byte>(15*out));
            const auto vm=memory->access->copy();const auto reads=memory->reads,writes=memory->writes,notifications=memory->access->notifications;const auto bytes_read=memory->read_bytes;const auto entries=surfaces.metadata_statistics().entries;
            memory->forbid_pixels=true;transfer.execute(plan);memory->forbid_pixels=false;
            require(transfer.native_completed()==1&&transfer.vm_completed()==0,"Native split input used VM route");
            require((mixed||memory->reads==reads)&&memory->read_bytes-bytes_read==requested_holes&&memory->writes==writes&&memory->access->notifications==notifications&&memory->access->copy()==vm,"Native split input accessed VMpixels outside exact RAM holes");
            hole_bytes+=requested_holes;
            for(const auto& piece:pieces)require(surfaces.status(piece).dirty==!clean,"Native split input changed clean/dirty authority");
            require(surfaces.metadata_statistics().entries==entries+unsigned(cold),"Assembly admitted an unused guest-input raster");
            if(destination)
            {
                for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,background.begin()+((y+1)*8+1)*out);
                const auto raw=make_buffer(ctx,background.size(),buffer_storage::shared);
                surfaces.draw(std::array{cached_surface_use{destination,false}},[&](context& commands,auto images)
                {
                    auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:8 height:8 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);samples.expand(commands,images[0],image);}
                    texture_region all;all.size=MTLSizeMake(8,8,1);colors.download(commands,df,image,all,raw,{});
                });
                require(std::memcmp(raw.contents,background.data(),background.size())==0,"Native assembled output/background differs from independent rational oracle");exact_bytes+=background.size();
            }
            else
            {
                surfaces.prepare_cpu_read(target.range);auto expected=vm;
                for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*5*out,5*out,expected.begin()+target.range.address+y*target.row_pitch);
                require(memory->access->copy()==expected,"Assembled cold-output observation changed source/padding/VM gaps");exact_bytes+=patch.size();
            }
            ++profiles;
        }
    }
    std::cout<<(clean?(mixed?"Clean mixed native/RAM scaled images:":"Clean assembled scaled images:"):(mixed?"Mixed native/RAM scaled images:":"Native assembled scaled images:"))<<profiles<<(mixed?" two disjoint partial sources withbyte holes inside packed words":" two disjoint sources/split words andsingle GB8-ABGR8 reinterpretation")<<",RGB565-ARGB8/B8/GB8 byte authorities,odd-pitch/Morton/banked requests,linear-Morton native sources,1-2-4samples,raw-spilled inputs,physical mirrors,flips/filter/clip andexisting-cold targets;"<<exact_bytes<<" exact output/background bytes;"<<hole_bytes<<" exact missing RAMbytes captured,zero target capture/VMwrite/notifications duringexecute,source clean-dirty epochs retained,no unused input raster passed\n";
}

void assembled_scaled_image_transactions(bool mixed=false,bool clean=false)
{
    using namespace std::chrono_literals;
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);sample_transfer samples(ctx);unsigned cases=0;
    for(bool cold:{false,true})for(unsigned count:{1u,4u})for(unsigned failure=0;failure<(mixed?13u:10u);++failure)
    {
        @autoreleasepool
        {
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;auto refuse=std::make_shared<bool>(false);
            const cpu_capture_dispatch capture=[access=memory->access,refuse](const auto& work){if(*refuse)throw std::range_error("Assembly completion dispatcher refused");access->execute([&](auto&){work();});};
            auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
            const auto first=surfaces.acquire({0x1000,{2/factors.x,2/factors.y,count,surface_color::rgb565,false},8,{}});
            cached_surface_description second_description{0x1004,{2/factors.x,2/factors.y,count,surface_color::rgb565,false},8,{}};
            if(mixed)for(unsigned i=0;i<plan_guest_surface(0,2,2,2,8).range.size;++i)second_description.segments.push_back({i==1||i==9?0xd4000u+i:0x1004u+i,i,1});
            const auto second=surfaces.acquire(second_description);
            const auto target=surfaces.acquire({0x6000,{cold?3u:4/factors.x,cold?2u:4/factors.y,cold?1u:count,surface_color::rgb565,false},10,{}});
            std::vector<std::byte> input(16),background(32,std::byte{0xa5});for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+109)&255);
            if(mixed){for(const auto i:{5u,13u})memory->raw->write({0x1000+i,1},std::span{&input[i],1});memory->allowed_input_reads={{0x1005,1},{0x100d,1}};}
            const auto seed=[&](const cached_surface_ref& ref,std::span<const std::byte> bytes,unsigned width,unsigned height)
            {
                surfaces.prepare(ref,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{ref,true}},[&](context& commands,auto images)
                {auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:width height:height mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);}texture_region all;all.size=MTLSizeMake(width,height,1);colors.upload(commands,surface_color::rgb565,make_upload(commands,bytes),{},image,all);if(count>1)samples.import(commands,image,images[0]);});
            };
            for(unsigned side=0;side<2;++side){std::vector<std::byte> part(8);for(unsigned y=0;y<2;++y)std::copy_n(input.begin()+y*8+side*4,4,part.begin()+y*4);seed(side?second:first,part,2,2);}
            if(clean){surfaces.flush(first);surfaces.flush(second);}
            if(!cold)seed(target,background,4,4);
            const auto before=surfaces.status(target);const auto vm=memory->access->copy();const auto captured_bytes=memory->read_bytes;
            scaled_image_plan plan{plan_guest_surface(0x1000,4,2,2,8),plan_guest_surface(0x6000,3,2,2,10),scaled_image_format::rgb565,scaled_image_format::rgb565,4,2,5,3,1,1,true,true,true};
            auto canonical=plan;canonical.source=plan_guest_surface(0,4,2,2);canonical.destination=plan_guest_surface(0,3,2,2);
            const auto patch=reference(canonical,input,std::vector<std::byte>(12));auto expected=background;for(unsigned y=0;y<2;++y)std::copy_n(patch.begin()+y*6,6,expected.begin()+y*8);
            const auto readback=[&]
            {
                const auto raw=make_buffer(ctx,32,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{target,false}},[&](context& commands,auto images)
                {auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;image=make_texture(commands,d);samples.expand(commands,images[0],image);}texture_region all;all.size=MTLSizeMake(4,4,1);colors.download(commands,surface_color::rgb565,image,all,raw,{});});const auto bytes=map_buffer(raw,0,32);return std::vector<std::byte>(bytes.begin(),bytes.end());
            };
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} cleanup{gate};std::optional<submission> token;auto valid=std::make_shared<bool>(true);
            const auto validate=[valid,access=memory->access]{require(guest_memory_epoch::current(*access)!=nullptr,"Assembly metadata escaped producer exclusion");if(!*valid)throw scaled_image_mapping_conflict();};
            std::vector<id<MTLBuffer>> blockers;if(failure==9){(void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate}){auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}}
            memory->forbid_pixels=true;bool failed=false;
            try{transfer->execute(plan,failure==9?30s:1ms,[&](context& commands){token=commands.recording_submission();if(failure==7)throw std::range_error("Assembly encode refused");commands.wait_event(gate,1);},validate);}catch(const std::runtime_error&){failed=true;}
            require(failed&&surfaces.status(target).content_generation==before.content_generation&&memory->access->copy()==vm,"Assembly published before real output completion");
            if(failure==9){require(!transfer->has_pending()&&(!token||(token->ready()&&token->cancelled())),"Assembly quota refusal lost pending ownership");blockers.clear();transfer->execute(plan,30s,{},validate);failed=false;}
            else if(failure==7){require(token&&token->cancelled()&&!transfer->has_pending()&&!surfaces.status(target).pending,"Assembly encode cancellation retained current output");}
            else
            {
                require(token&&transfer->has_pending()&&surfaces.status(target).pending&&!token->ready(),"Assembly lacks actual gated native output");
                if(failure>=1&&failure<=3)memory->raw->write({failure==1?0x1000u:failure==2?0x1004u:0x6000u,1},std::array{std::byte{0x5c}});
                if(failure==10||failure==11){memory->raw->write({0x1005,1},std::array{std::byte{0x5c}});if(failure==11)memory->raw->write({0x1005,1},std::span{&input[5],1});}
                if(failure==12)memory->raw->write({0x1010,1},std::array{std::byte{0x7d}});
                if(failure==4)*valid=false;if(failure==5)require(token->cancel_pending(),"Assembly queued cancellation refused");if(failure==6)transfer.reset();if(failure==8)*refuse=true;
                gate.signaledValue=1;failed=false;try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                if(failure==8){require(failed&&transfer->has_pending()&&surfaces.status(target).pending,"Assembly dispatcher rejection dropped current transaction");*refuse=false;transfer->synchronize();failed=false;}
                if(failure==5)ctx.recover_completed_errors();
            }
            const bool rejected=(failure>=1&&failure<=7)||failure==10||failure==11;
            require(failed==rejected&&!surfaces.status(target).pending&&!surfaces.status(target).faulted&&(!transfer||!transfer->has_pending()),"Assembly completion/rejection metadata differs");
            require(surfaces.status(target).content_generation==before.content_generation+unsigned(!rejected),"Assembly changed generation after rejected native output");
            if(cold){if(rejected)require(!surfaces.status(target).initialized&&!surfaces.status(target).dirty,"Assembly rejection initialized cold output");}
            else if(failure!=3)require(readback()==(rejected?background:expected),"Assembly native rollback lost exact output/background");
            auto expected_vm=vm;if(failure==10)expected_vm[0x1005]=std::byte{0x5c};if(failure==12)expected_vm[0x1010]=std::byte{0x7d};
            if(failure>=1&&failure<=3)require(memory->raw->read({failure==1?0x1000u:failure==2?0x1004u:0x6000u,1}).bytes[0]==std::byte{0x5c},"Assembly overwrote competing CPU writer");else require(memory->access->copy()==expected_vm,"Assembly wrote VMpixels orlost unrelated CPUbytes before observation");
            if(clean){require(!surfaces.status(first).dirty&&!surfaces.status(second).dirty,"Assembly dirtied clean input authorities");if(!mixed)require(memory->read_bytes==captured_bytes,"Clean input quota/retry recaptured VMpixels");}
            memory->forbid_pixels=false;
            if(cold&&!rejected){surfaces.prepare_cpu_read(plan.destination.range);for(unsigned y=0;y<2;++y)std::copy_n(patch.begin()+y*6,6,expected_vm.begin()+0x6000+y*10);require(memory->access->copy()==expected_vm,"Assembly cold observation changed source/padding");}
            ++cases;
        }
    }
    std::cout<<(clean?(mixed?"Clean mixed native/RAM transactions:":"Clean assembled transactions:"):(mixed?"Mixed native/RAM transactions:":"Native assembled transactions:"))<<cases<<" real gated single-four-sample source/existing-cold target cases;both native source andtarget CPUwriters,owned metadata rejection,queued/encode cancellation,owner abandonment,CPUdispatcher refusal/retry andactual Shared-Private quota refusal/retry"<<(mixed?",RAM hole writer/identical-byte ABA refused,unrelated writer preserved":"")<<";exact rollback/background andno early output passed\n";
}

void mixed_source_lifetimes()
{
    using namespace std::chrono_literals;unsigned cases=0;
    for(bool cold:{false,true})for(unsigned outcome=0;outcome<3;++outcome)
    {
        std::weak_ptr<native_access> provider;std::shared_ptr<heap_budget> ledger;
        @autoreleasepool
        {
            auto ctx=std::make_unique<context>(2);ledger=std::make_shared<heap_budget>(*ctx,64u<<20);ctx->set_buffer_budget(ledger);
            {
                auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;provider=memory->access;
                const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
                auto surfaces=std::make_unique<surface_cache>(*ctx,memory,16u<<20,ledger,capture);auto textures=std::make_unique<guest_texture_cache>(*ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,texture_capture_bindings{capture,io});auto transfer=std::make_unique<scaled_image_transfer>(*ctx,*surfaces,*textures,memory,capture,1u<<20);surface_color_transfer colors(*ctx);
                const auto piece=surfaces->acquire({0x1000,{3,2,1,surface_color::b8,false},8,{}}),target=surfaces->acquire({0x6000,{2,2,1,surface_color::argb8,false},9,{}});
                std::vector<std::byte> input(16),native(6),background(16,std::byte{0xa5});for(unsigned i=0;i<16;++i)input[i]=std::byte((i*37+109)&255);const auto expected=input;
                for(unsigned y=0;y<2;++y){std::copy_n(input.begin()+y*8,3,native.begin()+y*3);memory->raw->write({0x1003+y*8,5},std::span{input}.subspan(y*8+3,5));memory->allowed_input_reads.push_back({0x1003+y*8,5});}
                const auto seed=[&](const cached_surface_ref& ref,surface_color format,std::span<const std::byte> bytes,unsigned width)
                {surfaces->prepare(ref,surface_initialization::background);surfaces->draw(std::array{cached_surface_use{ref,true}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(width,2,1);colors.upload(commands,format,make_upload(commands,bytes),{},images[0],all);});};
                seed(piece,surface_color::b8,native,3);if(!cold)seed(target,surface_color::argb8,background,2);
                const auto before=surfaces->status(target);const auto vm=memory->access->copy();const auto bytes=memory->read_bytes;
                scaled_image_plan plan{plan_guest_surface(0x1000,2,2,4),plan_guest_surface(0x6000,2,2,4,9),scaled_image_format::argb8,scaled_image_format::argb8,2,2,2,2};
                const auto gate=[ctx->device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};std::optional<submission> token;memory->forbid_pixels=true;bool failed=false;
                try{transfer->execute(plan,1ms,[&](context& commands){token=commands.recording_submission();commands.wait_event(gate,1);});}catch(const std::runtime_error&){failed=true;}
                require(failed&&token&&!token->ready()&&transfer->has_pending()&&surfaces->status(target).pending&&memory->read_bytes-bytes==10&&memory->access->copy()==vm,"Mixed input lifetime lacks real pending output orcaptured extra VMbytes");
                input.assign(input.size(),std::byte{});native.clear();
                if(outcome==1)transfer.reset();if(outcome==2)ctx->shutdown();else gate.signaledValue=1;
                failed=false;try{if(transfer)transfer->synchronize();else surfaces->synchronize();}catch(const std::runtime_error&){failed=true;}
                require(failed==bool(outcome)&&!surfaces->status(target).pending&&!surfaces->status(target).faulted&&surfaces->status(target).content_generation==before.content_generation+unsigned(!outcome)&&memory->access->copy()==vm,"Mixed input abandonment/shutdown published early orlost target metadata");
                if(outcome==2)require(token->ready()&&token->cancelled(),"Explicit mixed input shutdown left queued native work live");
                if(outcome&&cold)require(!surfaces->status(target).initialized&&!surfaces->status(target).dirty,"Mixed input teardown initialized cold target");
                if(outcome!=2)
                {
                    memory->forbid_pixels=false;surfaces->prepare_cpu_read(plan.destination.range);auto expected_vm=vm;
                    if(!outcome||!cold){const auto& pixels=outcome?background:expected;for(unsigned y=0;y<2;++y)std::copy_n(pixels.begin()+y*8,8,expected_vm.begin()+0x6000+y*9);}
                    require(memory->access->copy()==expected_vm,"Mixed input owned bytes/background did not survive pending owner lifetime");
                }
            }
            ctx.reset();
        }
        require(provider.expired()&&ledger->statistics().reserved_bytes==0,"Mixed input watches/provider/native ledger survived complete cache/context teardown");++cases;
    }
    std::cout<<"Mixed source lifetime:"<<cases<<" actual pending outputs with existing-cold targets,owned RAM bytes after caller mutation,transfer abandonment andexplicit blocked context.shutdown;no early output,exact bytes/background,provider retirement/zero native ledger passed\n";
}

void assembled_source_bounds()
{
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);
    unsigned cases=0;
    for(unsigned state=0;state<4;++state)for(unsigned mode=0;mode<7;++mode)
    {
        @autoreleasepool
        {
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=mode!=2;
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
            const bool clean=state>=2,spill=state&1;
            const unsigned width=mode==1?5:mode==6?13:mode>=4?8:6,height=mode==1||mode==6?5:mode>=4?8:4;
            const unsigned count=width*height-unsigned(mode==3||mode==5);
            std::vector<std::byte> input(width*height*2);for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+119)&255);
            if(mode==3||mode==5)memory->raw->write({0x1000+count*2,2},std::span{input}.last(2));
            cached_surface_ref first;std::vector<cached_surface_ref> pieces;
            for(unsigned i=0;i<count;++i)
            {
                const auto piece=surfaces.acquire({0x1000+i*2,{2,1,1,surface_color::b8,false},2,{}});surfaces.prepare(piece,surface_initialization::background);
                surfaces.draw(std::array{cached_surface_use{piece,true}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(2,1,1);colors.upload(commands,surface_color::b8,make_upload(commands,std::span{input}.subspan(i*2,2)),{},images[0],all);});
                if(!first)first=piece;if(clean)surfaces.flush(piece);if(spill)surfaces.spill(piece);pieces.push_back(piece);
            }
            // Public RSX draw slots keep their original bound even though
            // the private transfer batch accepts more sequential readers.
            const auto submitted=surfaces.transaction_statistics().submitted;bool encoded=false,refused=false;
            try{surfaces.draw(std::vector<cached_surface_use>(26,{first,false}),[&](context&,auto){encoded=true;});}catch(const std::invalid_argument&){refused=true;}
            require(refused&&!encoded&&surfaces.transaction_statistics().submitted==submitted,"Wide transfer changed the public RSX draw slot bound");
            scaled_image_plan plan{plan_guest_surface(0x1000,width,height,2),plan_guest_surface(0x6000,4,3,2,9),scaled_image_format::rgb565,scaled_image_format::rgb565,width,height,5,4,1,1,true,false,true};
            auto packed=plan;packed.source=plan_guest_surface(0,width,height,2);packed.destination=plan_guest_surface(0,4,3,2);const auto patch=reference(packed,input,std::vector<std::byte>(24));
            const bool direct=mode!=2&&mode!=6,native=direct||(mode==6&&clean);if(mode==3||mode==5)memory->allowed_input_reads={{0x1000+count*2,2}};
            const auto vm=memory->access->copy();const auto bytes_read=memory->read_bytes;memory->forbid_pixels=direct;transfer.execute(plan);memory->forbid_pixels=false;
            if(transfer.native_completed()!=unsigned(native)||transfer.vm_completed()!=unsigned(!native))throw std::runtime_error("Native assembly routing differs state="+std::to_string(state)+" mode="+std::to_string(mode)+" sources="+std::to_string(count)+" native="+std::to_string(transfer.native_completed())+" vm="+std::to_string(transfer.vm_completed()));
            if(native)for(const auto& piece:pieces)require(surfaces.status(piece).dirty==!clean,"Wide source preparation changed its clean/dirty authority");
            if(mode==6&&clean)require(memory->read_bytes-bytes_read==input.size(),"65 clean sources did not use one exact coherent RAM input");
            auto expected=vm;if(!native)std::copy(input.begin(),input.end(),expected.begin()+0x1000);
            if(native){require(memory->access->copy()==vm,"Native assembly wrote source VM");surfaces.prepare_cpu_read(plan.destination.range);}
            for(unsigned y=0;y<3;++y)std::copy_n(patch.begin()+y*8,8,expected.begin()+0x6000+y*9);
            require(memory->access->copy()==expected,"Assembly source bound/fallback lost exact bytes orpadding");
            ++cases;
        }
    }
    std::cout<<"Wide native assembly:"<<cases<<" dirty-clean/resident-spilled profiles;24-25-64native sources and23-63native-plus-one-RAMhole feed native output without native VMcapture;65dirty sources/missing exact identity use coherent VM fallback,65clean sources use one exact RAMinput/native output;public26-use RSX draws refused beforeencoding,wholeVM/padding exact passed\n";
}

void multi_output_fallbacks()
{
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);unsigned cases=0;
    for(unsigned mode=0;mode<4;++mode)
    @autoreleasepool
    {
        auto memory=std::make_shared<native_memory>();memory->access->typed_identities=mode!=3;memory->access->retained_slices=mode!=3;
        const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
        surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
        scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
        const unsigned width=mode==0?130:mode==1?8:mode==2?6:4,parts=mode==0?65:mode==1?3:2,pitch=(width+2)*4+1;
        const auto source=surfaces.acquire({0x1000,{4,4,1,surface_color::argb8,false},16,{}});
        std::vector<std::byte> input(64);for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*37+59)&255);
        const auto seed=[&](const auto& surface,const auto& bytes,unsigned w)
        {
            surfaces.prepare(surface,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images)
            {texture_region all;all.size=MTLSizeMake(w,4,1);colors.upload(commands,surface_color::argb8,make_upload(commands,bytes),{},images[0],all);});
        };
        seed(source,input,4);std::vector<std::pair<unsigned,std::vector<std::byte>>> backgrounds;
        for(unsigned i=0;i<parts;++i)
        {
            const auto x=mode==1&&i==2?6:mode==2&&i==1?5:i*2;
            const auto target=surfaces.acquire({0x6000+x*4,{2,4,1,surface_color::argb8,false},pitch,{}});
            std::vector<std::byte> bytes(32,std::byte(31+i*53));seed(target,bytes,2);backgrounds.push_back({x,std::move(bytes)});
        }
        const auto before=memory->access->copy();auto expected=before;
        if(mode!=1&&mode!=2)std::copy(input.begin(),input.end(),expected.begin()+0x1000);
        for(const auto& [x,bytes]:backgrounds)for(unsigned y=0;y<4;++y)std::copy_n(bytes.begin()+y*8,8,expected.begin()+0x6000+y*pitch+x*4);
        scaled_image_plan plan{plan_guest_surface(0x1000,4,4,4),plan_guest_surface(0x6000,width,4,4,pitch),scaled_image_format::argb8,scaled_image_format::argb8,4,4,width+1,5,1,1,true,false,true};
        auto packed=plan;packed.destination=plan_guest_surface(0,width,4,4);
        const auto patch=reference(packed,input,std::vector<std::byte>(width*16));
        for(unsigned y=0;y<4;++y)std::copy_n(patch.begin()+y*width*4,width*4,expected.begin()+0x6000+y*pitch);
        const bool native=mode==1||mode==2;memory->forbid_pixels=native;transfer.execute(plan);memory->forbid_pixels=false;
        require(transfer.vm_completed()==unsigned(!native)&&transfer.native_completed()==unsigned(native),"Mixed native output routing differs");
        if(native){require(memory->access->copy()==before,"Native gap output wrote VM pixels");surfaces.prepare_cpu_read(plan.destination.range);}
        require(memory->access->copy()==expected,"Multi-output fallback changed source/padding or lost an exterior native pixel");++cases;
    }
    std::cout<<"Mixed output routing: "<<cases<<" cases; two native gap outputs and coherent fallback for 65 writers/untyped dirty owners; exact source/output/exterior/padding bytes passed\n";
}

void multi_output_scaled_images(bool transactions=false,bool partial=false,bool feedback=false,bool scattered=false,bool holes=false,bool wide=false,bool byte_scatter=false,unsigned preparation=0,bool queued=false)
{
    using namespace std::chrono_literals;
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);sample_transfer samples(ctx);
    unsigned profiles=0,timeouts=0,rejections=0,old_readers=0,quota_refusals=0,cache_rollbacks=0,one_submission=0,frozen_sources=0,preload_checks=0;std::size_t exact_bytes=0;
    for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned count:{1u,2u,4u})
    for(unsigned pieces:byte_scatter&&!wide?std::vector{1u,2u,5u}:wide?(holes?std::vector{6u,63u}:std::vector{6u,64u}):std::vector{1u,2u,3u,4u,5u})for(unsigned layout=0;layout<(transactions||wide?1u:3u);++layout)
    for(unsigned mode=0;mode<(transactions?(partial?10u:9u):wide||preparation?1u:8u);++mode)
    @autoreleasepool
    {
        if((!scattered&&(pieces==1||pieces==4))||(holes&&pieces==5&&!byte_scatter))continue;
        try
        {
        auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
        if((wide&&partial)||(byte_scatter&&pieces>2))memory->raw=std::make_shared<guarded_guest_memory>(memory->access,wide&&byte_scatter?16384:8192);
        auto refuse=std::make_shared<bool>(false),valid=std::make_shared<bool>(true);
        const cpu_capture_dispatch capture=[access=memory->access,refuse](const auto& work){if(*refuse)throw std::range_error("Multi-output CPU dispatcher refused");access->execute([&](auto&){work();});};
        surface_metadata_limits metadata; if((wide&&partial)||(byte_scatter&&pieces>2))metadata.watch_parts=wide&&byte_scatter?8192:4096;
        auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,memory,16u<<20,ledger,capture,metadata);
        guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
        auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
        const auto sf=in==2?surface_color::rgb565:surface_color::argb8,df=out==2?surface_color::rgb565:surface_color::argb8;
        const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
        const unsigned width=pieces*2,tw=partial?4:2,th=partial?8:4,ox=partial?1:0,oy=partial?3:0;
        const auto swizzle=layout==1?std::optional<swizzle_region>{{{16,8,1},{1,2,0},{width,4,1}}}:std::nullopt;
        const auto tile=layout==2?std::optional<banked_tile_layout>{{0x80000,0x40000,256,3,0x1400+out}}:std::nullopt;
        auto output=plan_guest_surface(tile?tile->address:0x6000,width,4,out,swizzle?0:tile?256:width*out+3,swizzle);
        if(tile)output.range={tile->address,tile->size};
        const auto source=feedback||wide?cached_surface_ref{}:surfaces.acquire({0x1000,{8,4,1,sf,false},8*in,{}});
        std::vector<std::byte> input(32*in);for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*73+i/7+31)&255);
        const auto linear=[&](id<MTLTexture> image,unsigned w,unsigned h)
        {
            if(image.sampleCount==1)return image;
            auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:w height:h mipmapped:NO];
            d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;return make_texture(ctx,d);
        };
        const auto seed=[&](const auto& surface,surface_color format,const auto& bytes,unsigned w,unsigned h=4)
        {
            surfaces.prepare(surface,surface_initialization::background);
            surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images)
            {
                const auto image=linear(images[0],w,h);texture_region all;all.size=MTLSizeMake(w,h,1);
                colors.upload(commands,format,make_upload(commands,bytes),{},image,all);
                if(image!=images[0])samples.import(commands,image,images[0]);
            });
        };
        const auto read_image=[&](id<MTLTexture> image,surface_color format,unsigned w,unsigned h=4)
        {
            const auto data=make_buffer(ctx,w*h*color_layout(format).pixel_bytes,buffer_storage::shared);
            const auto expanded=linear(image,w,h);ctx.begin();if(expanded!=image)samples.expand(ctx,image,expanded);
            texture_region all;all.size=MTLSizeMake(w,h,1);colors.download(ctx,format,expanded,all,data,{});ctx.submit().wait();
            const auto bytes=map_buffer(data,0,data.length);return std::vector<std::byte>(bytes.begin(),bytes.end());
        };
        const auto current_image=[&](const auto& target)
        {
            id<MTLTexture> image=nil;surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){image=images[0];});return image;
        };
        if(source)seed(source,sf,input,8);
        // Sixty-four independent byte authorities exercise the full reader
        // vector together with the new writer capacity (128 uses at the cap).
        if(wide)for(unsigned i=0;i<64;++i)
        {
            const auto bytes=in/2;
            const auto piece=surfaces.acquire({0x1000+i*bytes,{bytes,1,1,surface_color::b8,false},bytes,{}});
            seed(piece,surface_color::b8,std::vector<std::byte>(input.begin()+i*bytes,input.begin()+(i+1)*bytes),bytes,1);
        }
        std::vector<cached_surface_ref> targets,readers;
        std::vector<std::vector<std::byte>> backgrounds,prior_backgrounds;
        std::vector<std::vector<unsigned>> target_addresses,selected_offsets;
        std::vector<cached_surface_status> before;
        std::vector<id<MTLTexture>> old;
        for(unsigned i=0;i<pieces;++i)
        {
            const auto morton=layout==1?std::optional<swizzle_region>{{{8,16,1},{2,1,0},{tw,th,1}}}:std::nullopt;
            const auto geometry=plan_guest_surface(0,tw,th,out,morton?0:tw*out+1,morton);
            std::vector<unsigned> addresses(geometry.range.size);
            for(unsigned at=0;at<addresses.size();++at)addresses[at]=0xd0000+i*0x400+at;
            std::vector<std::array<unsigned,2>> slots;
            for(unsigned y=0;y<th;++y)for(unsigned x=0;x<tw;++x)if((x+y)%2==0&&(x||y))slots.push_back({x,y});
            if(mode&1)std::reverse(slots.begin(),slots.end());
            require(!scattered||slots.size()>=8,"Disconnected output fixture lacks native cells");
            std::vector<unsigned> selected;
            for(unsigned y=0;y<4;++y)for(unsigned x=0;x<2;++x)for(unsigned lane=0;lane<out;++lane)
            {
                auto at=offset(output,i*2+x,y)+lane;if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                const auto ordinal=(y*2+x)*out+lane;
                const auto native_byte=byte_scatter?(ordinal*3+(i*7+mode)%out)%(tw*th*out):0;
                const auto cell=byte_scatter?std::array<unsigned,2>{native_byte/out%tw,native_byte/out/tw}:
                    scattered?slots[y*2+x]:std::array<unsigned,2>{ox+x,oy+y};
                const auto native_lane=byte_scatter?native_byte%out:lane;
                if(holes&&(x+y+i+mode+(byte_scatter?lane:0))%3==0)selected.push_back(std::numeric_limits<unsigned>::max());
                else
                {
                    addresses[offset(geometry,cell[0],cell[1])+native_lane]=(tile?tile->address:output.range.address)+at+(i%2?0x100000:0);
                    selected.push_back((cell[1]*tw+cell[0])*out+native_lane);
                }
            }
            selected_offsets.push_back(std::move(selected));
            // Single-byte IO spans deliberately split packed words and rows.
            std::vector<guest_storage_segment> spans;
            for(unsigned at=0;at<addresses.size();++at)
            {
                // Keep selected words split at every byte. Unselected native
                // exterior uses ordinary contiguous IO metadata, so the test
                // retains the production watch and physical-ticket budgets.
                const bool exterior=addresses[at]>=0xd0000&&addresses[at]<0xd2000;
                if((wide||partial&&exterior)&&!spans.empty()&&std::uint64_t(spans.back().address)+spans.back().size==addresses[at])++spans.back().size;
                else spans.push_back({addresses[at],at,1});
            }
            const auto target=surfaces.acquire({addresses.front(),{tw/factors.x,th/factors.y,count,df,false},morton?0:geometry.row_pitch,morton,{},spans});
            std::vector<std::byte> background(tw*th*out);for(unsigned at=0;at<background.size();++at)background[at]=std::byte((at*43+i*37+109)&255);
            seed(target,df,background,tw,th);
            if(!preparation&&!transactions&&(mode==4||mode==5))
            {
                const auto donor=surfaces.acquire({0x30000+i*0x1000,{tw/factors.x,th/factors.y,count,df,false},tw*out,{}});
                seed(donor,df,background,tw,th);surfaces.defer_copy(donor,{0,0,tw/factors.x,th/factors.y},target,0,0);
            }
            if(!preparation&&!transactions&&(mode&1))surfaces.flush(target);
            if(!preparation&&!transactions&&(mode==2||mode==6))surfaces.spill(target);
            if(preparation||transactions||mode==3||mode==7)
            {
                const auto reader=surfaces.acquire({0x40000+i*0x1000,{tw/factors.x,th/factors.y,count,df,false},tw*out,{}});
                surfaces.defer_copy(target,{0,0,tw/factors.x,th/factors.y},reader,0,0);readers.push_back(reader);
                old.push_back(current_image(target));
            }
            prior_backgrounds.push_back(background);
            if(preparation)
            {
                if(preparation==1||preparation==4)surfaces.spill(target);
                if(preparation>=2)
                {
                    auto data=background;for(auto& byte:data)byte^=std::byte{0x5b};
                    const auto donor=surfaces.acquire({0x30000+i*0x1000,{tw/factors.x,th/factors.y,count,df,false},tw*out,{}});
                    seed(donor,df,data,tw,th);
                    const unsigned native_width=preparation==2?tw/factors.x:tw/factors.x/2;
                    surfaces.defer_copy(donor,{0,0,native_width,th/factors.y},target,0,0);
                    for(unsigned y=0;y<th;++y)for(unsigned x=0;x<native_width*factors.x;++x)for(unsigned lane=0;lane<out;++lane)
                        background[(y*tw+x)*out+lane]=data[(y*tw+x)*out+lane];
                }
                require(surfaces.status(readers.back()).deferred_copies==1,"Prepared output fixture consumed its old reader");
            }
            std::vector<unsigned> active;
            for(unsigned y=0;y<th;++y)for(unsigned x=0;x<tw;++x)for(unsigned lane=0;lane<out;++lane)
                active.push_back(addresses[offset(geometry,x,y)+lane]%0x100000);
            target_addresses.push_back(std::move(active));
            targets.push_back(target);backgrounds.push_back(background);before.push_back(surfaces.status(target));
        }
        const unsigned source_index=(mode&1)?0:pieces-1;
        auto input_geometry=plan_guest_surface(0x1000,8,4,in);
        std::vector<guest_storage_segment> input_segments;
        id<MTLTexture> feedback_old=nil;
        const bool ready_feedback=feedback&&std::all_of(targets.begin(),targets.end(),[&](const auto& target)
            {const auto status=surfaces.status(target);return status.initialized&&!status.spilled&&!status.deferred_copies;});
        if(feedback)
        {
            input=backgrounds[source_index];
            input_geometry=plan_guest_surface(target_addresses[source_index][0],tw*out/in,th,in);
            for(unsigned at=0;at<target_addresses[source_index].size();++at)
                input_segments.push_back({target_addresses[source_index][at]+((mode&1)?0x100000u:0u),at,1});
            input_geometry.range.address=input_segments.front().address;
            if(ready_feedback)feedback_old=current_image(targets[source_index]);
        }
        scaled_image_plan plan{input_geometry,output,in==2?scaled_image_format::rgb565:scaled_image_format::argb8,
            out==2?scaled_image_format::rgb565:scaled_image_format::argb8,static_cast<unsigned>(input_geometry.width),static_cast<unsigned>(input_geometry.height),width+3,7,1,1,bool(mode&1),bool(mode&2),bool(mode&4),{},tile,input_segments};
        auto packed=plan;packed.source=plan_guest_surface(0,input_geometry.width,input_geometry.height,in);packed.source_segments.clear();packed.destination=plan_guest_surface(0,width,4,out);packed.destination_tile.reset();
        const auto expected=reference(packed,input,std::vector<std::byte>(width*4*out));
        cached_surface_ref cold_target;std::vector<std::byte> cold_expected;unsigned cold_words=0;
        if(holes)
        {
            std::vector<guest_storage_segment> segments;
            for(unsigned y=0;y<4;++y)for(unsigned x=0;x<width;++x)
                for(unsigned lane=0;lane<out;++lane)if((x%2+y+x/2+mode+(byte_scatter?lane:0))%3==0)
                {
                    auto at=offset(output,x,y)+lane;if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                    const auto address=(tile?tile->address:output.range.address)+at;
                    const auto packed=cold_expected.size();cold_expected.push_back(expected[(y*width+x)*out+lane]);
                    if(!segments.empty()&&std::uint64_t(segments.back().address)+segments.back().size==address)++segments.back().size;
                    else segments.push_back({static_cast<unsigned>(address),packed,1});
                }
            cold_words=byte_scatter?cold_expected.size():cold_expected.size()/out;
            require(cold_words>0,"Cold output fixture has no missing words");
            const auto cold_format=byte_scatter?surface_color::b8:df;
            cold_target=surfaces.acquire({segments.front().address,{cold_words,1,1,cold_format,false},cold_expected.size(),{},{},segments});
            require(!surfaces.status(cold_target).initialized&&!surfaces.status(cold_target).dirty,"Cold output fixture already owns pixels");
        }
        const auto vm=memory->access->copy();const auto reads=memory->read_bytes;const auto writes=memory->writes;
        const auto submissions=surfaces.transaction_statistics().submitted;
        auto encoding_before=ctx.encoding_statistics();
        const auto before_loads=[&](context& commands)
        {
            if(!preparation)return;
            static_assert(std::has_unique_object_representations_v<command_encoding_statistics>);
            const auto now=commands.encoding_statistics();
            require(std::memcmp(&now,&encoding_before,sizeof(now))==0,"Prepared output encoded native loads before its prerequisites");++preload_checks;
        };
        if(wide)
        {
            std::vector<cached_surface_use> ordinary;
            for(unsigned i=0;i<6;++i)ordinary.push_back({targets[i],true});
            bool encoded=false,refused=false;
            try{surfaces.draw(ordinary,[&](context&,auto){encoded=true;});}
            catch(const std::invalid_argument&){refused=true;}
            require(refused&&!encoded&&surfaces.transaction_statistics().submitted==submissions,
                "Wide transfer changed the ordinary five-writer draw limit");
        }
        memory->forbid_pixels=true;
        if(transactions)
        {
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
            std::vector<id<MTLBuffer>> blockers;
            if(mode==8)
            {
                (void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();
                for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate})
                {
                    auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;
                    while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}
                }
            }
            std::optional<submission> token;bool failed=false,returned=false;
            try
            {
                const auto submit=queued?&scaled_image_transfer::enqueue:&scaled_image_transfer::execute;
                (transfer.get()->*submit)(plan,mode==8?30s:1ms,[&](context& commands){token=commands.recording_submission();before_loads(commands);if(mode==5)throw std::range_error("Multi-output encoder refused");commands.wait_event(gate,1);},[valid]{if(!*valid)throw scaled_image_mapping_conflict();});
                if(queued)
                {
                    returned=true;
                    require(token&&!token->ready()&&transfer->has_pending()&&!transfer->poll(),
                        "Queued multi-output scale did not return with private GPU work pending");
                    transfer->synchronize(1ms);
                }
            }
            catch(const std::runtime_error&){failed=true;}
            if(queued&&mode!=5&&mode!=8)require(returned,"Queued multi-output scale waited for GPU completion");
            require(failed,"Multi-output gate or encoder did not stop completion");
            for(unsigned i=0;i<pieces;++i)require(surfaces.status(targets[i]).content_generation==before[i].content_generation,"Multi-output published a subset before completion");
            if(mode==8)
            {
                require(!token&&!transfer->has_pending(),"Multi-output quota refusal reached encoding or retained work");
                for(unsigned i=0;i<pieces;++i)require(!surfaces.status(targets[i]).faulted,"Multi-output quota refusal faulted an old target");
                blockers.clear();
                for(unsigned i=0;i<pieces;++i)require(read_image(old[i],df,tw,th)==prior_backgrounds[i],"Multi-output quota refusal changed old native words");
                encoding_before=ctx.encoding_statistics();transfer->execute(plan,30s,before_loads);++quota_refusals;
            }
            else if(mode==5)require(token&&token->cancelled()&&!transfer->has_pending(),"Multi-output encode failure retained a transaction");
            else
            {
                require(token&&!token->ready()&&transfer->has_pending(),"Multi-output transaction is not actually gated");++timeouts;
                if(mode==1)*valid=false;
                if(mode==2)require(token->cancel_pending(),"Multi-output queued cancel refused");
                if(mode==3)memory->raw->write({0x6000+(width*out+3)*3+(width-1)*out,1},std::array{std::byte{0x5c}});
                if(mode==9)memory->raw->write({target_addresses[0][0],1},std::array{std::byte{0x5c}});
                if(mode==4)memory->raw->write({feedback?target_addresses[source_index][0]:0x1000u,1},std::array{std::byte{0x5c}});
                if(mode==6)transfer.reset();
                if(mode==7)*refuse=true;
                gate.signaledValue=1;failed=false;
                try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                if(mode==7){require(failed&&transfer->has_pending(),"Multi-output dispatcher rejection lost pending work");*refuse=false;transfer->synchronize();failed=false;}
                if(mode==2)ctx.recover_completed_errors();
                require(failed==(mode>=1&&mode<=4||mode==6||mode==9),"Multi-output completion/rollback status differs");
            }
        }
        else if(preparation)transfer->execute(plan,30s,before_loads);
        else transfer->execute(plan);
        const bool rejected=transactions&&(mode>=1&&mode<=6||mode==9);
        require(!transfer||!transfer->has_pending(),"Multi-output left an unexpected transaction");
        require(!transfer||transfer->native_completed()==unsigned(!rejected),"Multi-output did not use the native route");
        require(!transfer||transfer->vm_completed()==0,"Multi-output used VM output publication");
        require(memory->read_bytes==reads&&memory->writes==writes,"Multi-output captured or wrote VM pixels");
        if(preparation)
        {
            const auto expected_submissions=transactions&&mode==5?0u:1u;
            require(surfaces.transaction_statistics().submitted-submissions==expected_submissions,
                "Prepared output used a separate native initialization submission");++one_submission;
        }
        if(ready_feedback)
        {
            const auto expected_submissions=transactions&&mode==5?0u:1u;
            require(surfaces.transaction_statistics().submitted-submissions==expected_submissions,"Feedback used an extra source snapshot submission");
            ++one_submission;
            require(read_image(feedback_old,df,tw,th)==backgrounds[source_index],"Feedback modified the immutable old native source");++frozen_sources;
        }
        if(cold_target)
        {
            const auto state=surfaces.status(cold_target);
            require(!state.pending&&!state.faulted&&state.initialized==!rejected&&state.dirty==!rejected&&state.content_generation==unsigned(!rejected),"Cold hole output published early or lost rollback metadata");
            if(!rejected){require(read_image(current_image(cold_target),byte_scatter?surface_color::b8:df,cold_words,1)==cold_expected,"Cold native hole words differ from independent filter oracle");exact_bytes+=cold_expected.size();}
        }
        for(unsigned i=0;i<pieces;++i)
        {
            const auto status=surfaces.status(targets[i]);
            // Successful deferred materialization can add its own generation.
            require(!status.pending&&!status.faulted&&(rejected?status.content_generation==before[i].content_generation:status.content_generation>before[i].content_generation),"Multi-output target generation or rollback differs");
            if(preparation)
            {
                require(surfaces.status(readers[i]).deferred_copies==1,"Prepared transfer materialized a retained old reader");
                require(read_image(old[i],df,tw,th)==prior_backgrounds[i],"Prepared transfer changed an old reader lease");++old_readers;
            }
            if(rejected)
            {
                require(read_image(old[i],df,tw,th)==prior_backgrounds[i],"Multi-output rollback changed old native words");
                // The target receiving the CPU write has a deliberately stale
                // revision. Every other target must expose its actual rollback
                // lease through the cache, not merely retain an old image.
                // The CPU writer changes lane zero of the last requested
                // pixel. Byte holes can give its other lanes different owners.
                if((mode!=3||i+1!=pieces||(holes&&selected_offsets.back()[7*out]==std::numeric_limits<unsigned>::max()))&&(mode!=9||i!=0)&&(!feedback||mode!=4||i!=source_index))
                {
                    ++cache_rollbacks;require(read_image(current_image(targets[i]),df,tw,th)==backgrounds[i],"Multi-output cache did not restore its old native target");
                }
                ++rejections;
            }
            else
            {
                auto words=backgrounds[i];for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<2*out;++byte)if(selected_offsets[i][y*2*out+byte]!=std::numeric_limits<unsigned>::max())words[selected_offsets[i][y*2*out+byte]]=expected[(y*width+i*2)*out+byte];
                require(read_image(current_image(targets[i]),df,tw,th)==words,"Multi-output native words differ from rational filter oracle");exact_bytes+=words.size();
            }
            if(!preparation&&!readers.empty())
            {
                require(surfaces.status(readers[i]).deferred_copies==1,"Multi-output materialized a retained old reader");
                require(read_image(old[i],df,tw,th)==prior_backgrounds[i],"Multi-output modified retained old reader words");++old_readers;
            }
        }
        if(!transactions||mode!=3&&mode!=4&&mode!=9)require(memory->access->copy()==vm,"Multi-output changed VM bytes/padding");
        else require(memory->raw->read({mode==4?(feedback?target_addresses[source_index][0]:0x1000u):mode==9?target_addresses[0][0]:0x6000+(width*out+3)*3+(width-1)*out,1}).bytes[0]==std::byte{0x5c},"Multi-output overwrote a CPU writer");
        memory->forbid_pixels=false;
        if(!rejected)
        {
            surfaces.prepare_cpu_read(output.range);auto visible=vm;
            for(unsigned i=0;i<pieces;++i)
            {
                auto words=backgrounds[i];for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<2*out;++byte)if(selected_offsets[i][y*2*out+byte]!=std::numeric_limits<unsigned>::max())words[selected_offsets[i][y*2*out+byte]]=expected[(y*width+i*2)*out+byte];
                for(unsigned at=0;at<words.size();++at)visible[target_addresses[i][at]]=words[at];
            }
            if(holes)for(unsigned y=0;y<4;++y)for(unsigned x=0;x<width;++x)for(unsigned lane=0;lane<out;++lane)
            {
                auto at=offset(output,x,y)+lane;if(tile)at=banked_display_storage_offset(*tile,tile->offset+at);
                visible[(tile?tile->address:output.range.address)+at]=expected[(y*width+x)*out+lane];
            }
            require(memory->access->copy()==visible,"Multi-output CPU visibility changed padding or missed native words");
        }
        ++profiles;
        }
        catch(const std::exception& error){throw std::runtime_error(std::string(holes?"Cold hole output in=":scattered?"Disconnected output in=":feedback?"Feedback output in=":partial?"Partial output in=":"Multi-output in=")+std::to_string(in)+" out="+std::to_string(out)+" samples="+std::to_string(count)+" pieces="+std::to_string(pieces)+" layout="+std::to_string(layout)+" mode="+std::to_string(mode)+": "+error.what());}
    }
    std::cout<<"preparation="<<preparation<<" "<<(byte_scatter?(feedback?"Byte-scatter feedback output images ":"Byte-scatter native output images "):holes?(feedback?"Cold-hole feedback output images ":"Cold-hole native output images "):scattered?(feedback?"Disconnected feedback output images ":"Disconnected native output images "):feedback?(partial?"Partial feedback output images ":"Feedback output images "):partial?"Partial native output images ":"Multi-output scaled images ")<<(transactions?"transactions":"profiles")<<": "<<profiles<<" cases, "<<exact_bytes<<" exact output bytes, "<<timeouts<<" gated timeouts, "<<rejections<<" rolled-back targets, "<<cache_rollbacks<<" actual cache rollback checks, "<<old_readers<<" retained old reader checks, "<<quota_refusals<<" quota refusal/retries, "<<one_submission<<" single-transaction checks, "<<frozen_sources<<" immutable source checks, "<<preload_checks<<" prerequisite-before-load checks; "<<(byte_scatter?(wide?(holes?"7/64 total (6/63 plus cold B8)":"6/64 byte-scattered"):(holes?"2/3/6 total (1/2/5 plus cold B8)":"1/2/5 byte-scattered")):wide?(holes?"6/63 plus one cold":"6/64"):"2/3/5")<<" targets, "<<(wide?"64 independent native byte readers, ":"")<<"RGB565/ARGB8 cross-format, 1/2/4 samples, split physical mirrors, exact CPU visibility and no VM pixel work during transfer passed\n";
}

std::uint32_t depth_guest(surface_depth format,std::uint32_t bits,unsigned stencil)
{
    if(format==surface_depth::z16_uint)return bits;
    if(format==surface_depth::z16_float)return ((bits-(120u<<23))>>11)&65535;
    if(format==surface_depth::z24s8_float)return ((bits>>7)<<8)|stencil;
    const auto f=std::bit_cast<float>(bits);return (static_cast<std::uint32_t>((std::isnan(f)?0.f:std::clamp(f,0.f,1.f))*16777215.f)<<8)|stencil;
}
std::uint32_t depth_expanded(surface_depth format,std::uint32_t word)
{
    if(format==surface_depth::z16_uint)return word;
    if(format==surface_depth::z16_float)return std::bit_cast<std::uint32_t>(std::ldexp(1.f+float(word%4096)/4096,int(word/4096)-7));
    if(format==surface_depth::z24s8_float)return (word>>8)*128;
    const auto integer=word>>8;float f=static_cast<float>(double(integer)/16777215.0);
    const auto got=static_cast<std::uint32_t>(f*16777215.f);
    if(got<integer)f=std::nextafter(f,1.f);else if(got>integer)f=std::nextafter(f,0.f);
    require(static_cast<std::uint32_t>(f*16777215.f)==integer,"Expected exact Z24 output is not representable");return std::bit_cast<std::uint32_t>(f);
}
std::uint32_t initial_depth(surface_depth format,unsigned i,unsigned samples)
{
    if(format==surface_depth::z16_uint)return (i*7919+71)&65535;
    if(samples==1&&format!=surface_depth::z24s8_uint)
    {
        constexpr std::array<std::uint32_t,8> bits{0,0x80000000,1,0x3eaaaaab,0x3f123457,0x7f800000,0x7f800001,0x7fc12345};return bits[i%8];
    }
    return std::bit_cast<std::uint32_t>(.125f+float(i)/32)+i%7;
}

void mixed_color_byte_outputs(bool include_depth=false,unsigned preparation=0)
{
    std::vector<surface_format> formats{surface_color::x1rgb5_zero,surface_color::x1rgb5_one,surface_color::rgb565,
        surface_color::xrgb8_zero,surface_color::xrgb8_one,surface_color::argb8,surface_color::b8,surface_color::gb8,
        surface_color::rgba16f,surface_color::rgba32f,surface_color::r32f,surface_color::xbgr8_zero,surface_color::xbgr8_one,surface_color::abgr8};
    if(include_depth)for(unsigned type=1;type<=4;++type)formats.emplace_back(static_cast<surface_depth>(type));
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);unsigned cases=0,repeats=0,feedbacks=0;std::size_t exact=0,raw_exact=0;unsigned old_readers=0,one_submission=0,preload_checks=0;
    for(unsigned out:{2u,4u})for(unsigned count:{1u,2u,4u})for(bool holes:{false,true})for(unsigned source_index=0;source_index<=formats.size();++source_index)
    @autoreleasepool
    {
        if(preparation&&source_index<formats.size()&&source_index!=8&&source_index!=9&&source_index!=10)continue;
        try
        {
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
            memory->raw=std::make_shared<guarded_guest_memory>(memory->access,8192);
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_metadata_limits metadata;metadata.watch_parts=4096;
            surface_cache surfaces(ctx,memory,16u<<20,ledger,capture,metadata);
            guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
            scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
            unsigned owned=0;for(const auto format:formats)owned+=2*surface_guest_bytes(format);
            const unsigned width=((holes?2*owned:owned)+2*out-1)/(2*out),height=2,pitch=width*out+3;
            const auto output=plan_guest_surface(0x6000,width,height,out,pitch);
            std::vector<cached_surface_ref> targets;std::vector<std::vector<std::byte>> expected,backgrounds;
            std::vector<std::vector<std::uint32_t>> native_bits,original_native_bits;
            std::vector<cached_surface_ref> prepared_readers;std::vector<id<MTLTexture>> old_images;
            std::vector<std::vector<unsigned>> addresses;std::vector<std::pair<unsigned,unsigned>> owners(width*height*out,{~0u,~0u});
            unsigned canonical=0;
            for(unsigned f=0;f<formats.size();++f)
            {
                const auto bpp=surface_guest_bytes(formats[f]);const auto geometry=plan_guest_surface(0,4,4,bpp,4*bpp+3);
                std::vector<unsigned> map(geometry.range.size);for(unsigned at=0;at<map.size();++at)map[at]=0xb0000+f*0x1000+at;
                const unsigned claims=2*bpp+(f==5&&!holes?width*height*out-owned:0);
                for(unsigned j=0;j<claims;++j)
                {
                    if(holes)while(canonical%2==0)++canonical;
                    require(canonical<owners.size(),"Mixed color fixture exceeded its packed output");
                    auto byte=(j*7+f*3+count)%(16*bpp);
                    // MSAA float depth stays in its existing finite [0,1]
                    // domain. Single-sample cases exercise every guest lane.
                    if(count>1&&formats[f]==surface_format{surface_depth::z16_float})byte=((j*7+f*3+count)%16)*2+1;
                    if(count>1&&formats[f]==surface_format{surface_depth::z24s8_float})byte=((j*7+f*3+count)%16)*4+2+j%2;
                    const auto at=(byte/bpp/4)*geometry.row_pitch+(byte/bpp%4)*bpp+byte%bpp;
                    map[at]=0x6000+(canonical/out/width)*pitch+canonical%(width*out)+(f%2?0x100000:0);
                    owners[canonical++]={f,byte};
                }
                std::vector<guest_storage_segment> segments;for(unsigned at=0;at<map.size();++at)segments.push_back({map[at],at,1});
                const auto target=surfaces.acquire({map.front(),{4/factors.x,4/factors.y,count,formats[f],false},geometry.row_pitch,{},{},segments});
                std::vector<std::byte> data(16*bpp);for(unsigned at=0;at<data.size();++at)data[at]=std::byte((at*43+f*31+count*7+97)&255);
                if(formats[f]==surface_format{surface_color::rgba16f})
                {
                    const std::array<unsigned,8> words{0,0x8000,1,0x7c00,0xfc00,0x7c01,0x7e55,0xffff};
                    for(unsigned at=0;at<data.size();at+=2){const auto value=words[(at/2)%words.size()];data[at]=std::byte(value>>8);data[at+1]=std::byte(value&255);}
                }
                if(formats[f]==surface_format{surface_color::rgba32f}||formats[f]==surface_format{surface_color::r32f})
                {
                    const std::array<std::uint32_t,8> words{0,0x80000000,1,0x7f800000,0xff800000,0x7f800001,0x7fc12345,0xffffffff};
                    for(unsigned at=0;at<data.size();at+=4)for(unsigned lane=0;lane<4;++lane)data[at+lane]=std::byte(words[(at/4)%words.size()]>>(24-8*lane));
                }
                std::vector<std::uint32_t> bits(32);
                if(const auto* depth=std::get_if<surface_depth>(&formats[f]))
                {
                    for(unsigned i=0;i<16;++i)
                    {
                        const auto native=initial_depth(*depth,i,count);
                        bits[i*2]=native;bits[i*2+1]=describe_surface_depth(*depth).stencil?(i*53+f*19+7)&255:0;
                        const auto word=depth_guest(*depth,native,bits[i*2+1]);
                        for(unsigned lane=0;lane<bpp;++lane)data[i*bpp+lane]=std::byte(word>>(8*(bpp-1-lane)));
                    }
                }
                const auto seed=[&](const cached_surface_ref& value,const std::vector<std::byte>& value_data,const std::vector<std::uint32_t>& value_bits)
                {
                surfaces.prepare(value,surface_initialization::background);
                surfaces.draw(std::array{cached_surface_use{value,true}},[&](context& commands,auto images)
                {
                    auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;image=make_texture(commands,d);}
                    texture_region all;all.size=MTLSizeMake(4,4,1);if(const auto* color=std::get_if<surface_color>(&formats[f]))colors.upload(commands,*color,make_upload(commands,value_data),{},image,all);
                    else
                    {
                        const auto md=describe_surface_depth(std::get<surface_depth>(formats[f]));std::vector<std::byte> d(4*256),st(4*256);
                        for(unsigned i=0;i<16;++i){std::memcpy(d.data()+i/4*256+i%4*md.depth_bytes,&value_bits[i*2],md.depth_bytes);st[i/4*256+i%4]=std::byte(value_bits[i*2+1]);}
                        buffer_to_texture(commands,make_upload(commands,d),0,image,all,256,0,texture_plane::depth);
                        if(md.stencil)buffer_to_texture(commands,make_upload(commands,st),0,image,all,256,0,texture_plane::stencil);
                    }
                    if(count>1)samples.import(commands,image,images[0]);
                });

                };
                seed(target,data,bits);surfaces.flush(target);
                const auto original_data=data;const auto original_bits=bits;
                if(preparation)
                {
                    id<MTLTexture> old=nil;surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){old=images[0];});old_images.push_back(old);
                    const auto reader=surfaces.acquire({0xe0000+f*0x400,{4/factors.x,4/factors.y,count,formats[f],false},4*bpp,{}});
                    surfaces.defer_copy(target,{0,0,4/factors.x,4/factors.y},reader,0,0);prepared_readers.push_back(reader);
                    if(preparation==1||preparation==4)surfaces.spill(target);
                    if(preparation>=2)
                    {
                        auto donor_data=data;auto donor_bits=bits;
                        for(unsigned i=0;i<16;++i)std::copy_n(data.begin()+((i+3)%16)*bpp,bpp,donor_data.begin()+i*bpp);
                        if(const auto* depth=std::get_if<surface_depth>(&formats[f]))for(unsigned i=0;i<16;++i)
                        {
                            donor_bits[2*i]=initial_depth(*depth,i+3,count);donor_bits[2*i+1]=describe_surface_depth(*depth).stencil?(i*47+f*23+19)&255:0;
                            const auto word=depth_guest(*depth,donor_bits[2*i],donor_bits[2*i+1]);for(unsigned lane=0;lane<bpp;++lane)donor_data[i*bpp+lane]=std::byte(word>>(8*(bpp-1-lane)));
                        }
                        const auto donor=surfaces.acquire({0x70000+f*0x1000,{4/factors.x,4/factors.y,count,formats[f],false},4*bpp,{}});seed(donor,donor_data,donor_bits);
                        const unsigned native_width=preparation==2?4/factors.x:2/factors.x;
                        surfaces.defer_copy(donor,{0,0,native_width,4/factors.y},target,0,0);
                        for(unsigned i=0;i<16;++i)if(i%4<native_width*factors.x)
                        {std::copy_n(donor_data.begin()+i*bpp,bpp,data.begin()+i*bpp);bits[2*i]=donor_bits[2*i];bits[2*i+1]=donor_bits[2*i+1];}
                    }
                }
                targets.push_back(target);expected.push_back(data);backgrounds.push_back(original_data);addresses.push_back(map);native_bits.push_back(bits);original_native_bits.push_back(original_bits);
            }
            const auto current=[&](const auto& target){id<MTLTexture> image=nil;surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){image=images[0];});return image;};
            const auto read_image=[&](id<MTLTexture> image,surface_format format,unsigned w,unsigned h)
            {
                const auto data=make_buffer(ctx,w*h*surface_guest_bytes(format),buffer_storage::shared);ctx.begin();
                auto expanded=image;if(image.sampleCount>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:w height:h mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;expanded=make_texture(ctx,d);samples.expand(ctx,image,expanded);}
                texture_region all;all.size=MTLSizeMake(w,h,1);if(const auto* color=std::get_if<surface_color>(&format))colors.download(ctx,*color,expanded,all,data,{});else depths.download(ctx,std::get<surface_depth>(format),expanded,all,data,{});ctx.submit().wait();const auto bytes=map_buffer(data,0,data.length);return std::vector<std::byte>(bytes.begin(),bytes.end());
            };
            const auto read_depth=[&](id<MTLTexture> image,surface_depth format)
            {
                const auto md=describe_surface_depth(format);const auto d=make_buffer(ctx,4*256,buffer_storage::shared),st=md.stencil?make_buffer(ctx,4*256,buffer_storage::shared):nil;
                ctx.begin();auto expanded=image;
                if(image.sampleCount>1){auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];desc.storageMode=MTLStorageModePrivate;desc.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;expanded=make_texture(ctx,desc);samples.expand(ctx,image,expanded);}
                texture_region all;all.size=MTLSizeMake(4,4,1);texture_to_buffer(ctx,expanded,all,d,0,256,0,texture_plane::depth);if(md.stencil)texture_to_buffer(ctx,expanded,all,st,0,256,0,texture_plane::stencil);ctx.submit().wait();
                std::vector<std::uint32_t> result(32);const auto raw=map_buffer(d,0,d.length);const auto stencil=st?map_buffer(st,0,st.length):std::span<std::byte>{};
                for(unsigned i=0;i<16;++i){std::memcpy(&result[i*2],raw.data()+i/4*256+i%4*md.depth_bytes,md.depth_bytes);result[i*2+1]=md.stencil?std::uint32_t(stencil[i/4*256+i%4]):0;}
                return result;
            };
            std::vector<std::byte> independent(24);for(unsigned j=0;j<independent.size();++j)independent[j]=std::byte((j*47+13)&255);
            const bool feedback=source_index<formats.size();
            auto source_geometry=plan_guest_surface(0x1000,3,2,4);std::vector<guest_storage_segment> source_segments;
            if(feedback)
            {
                const auto bpp=surface_guest_bytes(formats[source_index]);
                source_geometry=plan_guest_surface(addresses[source_index].front(),bpp,4,4);
                for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<4*bpp;++byte)
                    source_segments.push_back({addresses[source_index][y*(4*bpp+3)+byte],source_segments.size(),1});
            }
            else
            {
                const auto source=surfaces.acquire({0x1000,{3,2,1,surface_color::argb8,false},12,{}});surfaces.prepare(source,surface_initialization::background);
                surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(3,2,1);colors.upload(commands,surface_color::argb8,make_upload(commands,independent),{},images[0],all);});
            }
            scaled_image_plan plan{source_geometry,output,scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,
                static_cast<unsigned>(source_geometry.width),static_cast<unsigned>(source_geometry.height),width+3,3,1,1,bool(count&2),bool(out==2),holes,{},{},source_segments};
            // A borrowed texture ID alone does not retain a surface-pool
            // lease across later transfers. Keep a real deferred old reader.
            cached_surface_ref retained;
            if(feedback&&preparation)retained=prepared_readers[source_index];
            else if(feedback)
            {
                const auto bpp=surface_guest_bytes(formats[source_index]);
                retained=surfaces.acquire({0xf0000,{4/factors.x,4/factors.y,count,formats[source_index],false},4*bpp,{}});
                surfaces.defer_copy(targets[source_index],{0,0,4/factors.x,4/factors.y},retained,0,0);
            }
            const auto before=memory->access->copy();auto expected_vm=before;
            const auto reads=memory->read_bytes;const auto writes=memory->writes;memory->forbid_pixels=true;
            id<MTLTexture> old=feedback?(preparation?old_images[source_index]:current(targets[source_index])):nil;
            const auto old_raw=feedback&&std::holds_alternative<surface_depth>(formats[source_index])?read_depth(old,std::get<surface_depth>(formats[source_index])):std::vector<std::uint32_t>{};
            if(feedback)require(read_image(old,formats[source_index],4,4)==backgrounds[source_index],"Mixed color source seed changed native guest bits");
            for(unsigned repetition=0;repetition<2;++repetition)
            {
                auto packed=plan;packed.source=plan_guest_surface(0,source_geometry.width,source_geometry.height,4);packed.source_segments.clear();packed.destination=plan_guest_surface(0,width,height,out);
                const auto payload=reference(packed,feedback?expected[source_index]:independent,std::vector<std::byte>(width*height*out));
                for(unsigned j=0;j<payload.size();++j)if(owners[j].first!=~0u)expected[owners[j].first][owners[j].second]=payload[j];
                for(unsigned f=0;f<formats.size();++f)if(const auto* depth=std::get_if<surface_depth>(&formats[f]))
                {
                    const auto md=describe_surface_depth(*depth);std::array<unsigned,16> masks{};
                    for(const auto& owner:owners)if(owner.first==f)masks[owner.second/md.guest_bytes]|=1u<<(owner.second%md.guest_bytes);
                    for(unsigned i=0;i<16;++i)
                    {
                        std::uint32_t word=0;for(unsigned lane=0;lane<md.guest_bytes;++lane)word=word*256+std::uint32_t(expected[f][i*md.guest_bytes+lane]);
                        if(masks[i]&((1u<<(md.guest_bytes-unsigned(md.stencil)))-1))native_bits[f][i*2]=depth_expanded(*depth,word);
                        if(md.stencil&&(masks[i]&(1u<<(md.guest_bytes-1))))native_bits[f][i*2+1]=word&255;
                    }
                }
                const auto submitted=surfaces.transaction_statistics().submitted;
                const auto encoding_before=ctx.encoding_statistics();
                transfer.execute(plan,std::chrono::seconds(30),[&](context& commands)
                {
                    if(preparation&&repetition==0)
                    {
                        const auto now=commands.encoding_statistics();static_assert(std::has_unique_object_representations_v<command_encoding_statistics>);
                        require(std::memcmp(&now,&encoding_before,sizeof(now))==0,"Mixed prepared output encoded loads before prerequisites");++preload_checks;
                    }
                });
                if(preparation)
                {
                    require(surfaces.transaction_statistics().submitted-submitted==1,"Mixed prepared output used a separate initialization submission");++one_submission;
                    for(unsigned f=0;f<formats.size();++f)
                    {
                        require(surfaces.status(prepared_readers[f]).deferred_copies==1,"Mixed prepared output consumed an old reader");
                        require(read_image(old_images[f],formats[f],4,4)==backgrounds[f],"Mixed prepared output changed original native guest words");
                        if(const auto* depth=std::get_if<surface_depth>(&formats[f]))require(read_depth(old_images[f],*depth)==original_native_bits[f],"Mixed prepared output changed original native depth precision");++old_readers;
                    }
                }
                require(transfer.native_completed()==repetition+1&&transfer.vm_completed()==0&&memory->read_bytes==reads&&memory->writes==writes&&memory->access->copy()==before,
                    "Mixed color byte output used guest pixels or lost native cold-target reuse");
                for(unsigned f=0;f<formats.size();++f){const auto actual=read_image(current(targets[f]),formats[f],4,4);if(actual!=expected[f])throw std::runtime_error("Mixed color native guest-word bits differ format="+std::to_string(f));exact+=expected[f].size();if(const auto* depth=std::get_if<surface_depth>(&formats[f])){require(read_depth(current(targets[f]),*depth)==native_bits[f],"Mixed depth output changed selected or untouched native precision");const auto md=describe_surface_depth(*depth);raw_exact+=16*(md.depth_bytes+unsigned(md.stencil));}}
                if(feedback){require(surfaces.status(retained).deferred_copies==1,"Mixed color byte output materialized the retained old reader");if(read_image(old,formats[source_index],4,4)!=backgrounds[source_index])throw std::runtime_error("Mixed color byte output modified the retained source at repetition="+std::to_string(repetition));if(!old_raw.empty())require(read_depth(old,std::get<surface_depth>(formats[source_index]))==old_raw,"Mixed depth output modified raw immutable source precision");++feedbacks;}
                for(unsigned j=0;j<payload.size();++j)expected_vm[0x6000+j/out/width*pitch+j%(width*out)]=payload[j];
                ++repeats;
            }
            if(preparation)for(unsigned f=0;f<formats.size();++f)
            {
                const auto bpp=surface_guest_bytes(formats[f]);for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<4*bpp;++byte)expected_vm[addresses[f][y*(4*bpp+3)+byte]%0x100000]=expected[f][y*4*bpp+byte];
            }
            memory->forbid_pixels=false;surfaces.prepare_cpu_read(plan.destination.range);
            require(memory->access->copy()==expected_vm,"Mixed color byte output changed padding or unrequested guest bytes");++cases;
        }
        catch(const std::exception& error){throw std::runtime_error("Mixed color byte out="+std::to_string(out)+" samples="+std::to_string(count)+" holes="+std::to_string(holes)+" source="+std::to_string(source_index)+": "+error.what());}
    }
    std::cout<<"preparation="<<preparation<<" "<<(include_depth?"Mixed native depth/color byte output: ":"Mixed native color byte output: ")<<cases<<" profiles, "<<repeats<<" native/reused-cold transfers, "<<exact<<" exact old/new native guest bytes, "<<feedbacks
        <<" immutable feedback checks, "<<raw_exact<<" exact raw depth/stencil bytes, "<<old_readers<<" retained old readers, "<<one_submission<<" single-transaction checks, "<<preload_checks<<" prerequisite-before-load checks; "<<(include_depth?"all 18 depth/color formats":"all 14 color formats")<<" in each output, 1/2/4 samples, RGB565/ARGB8 requests, float signed zero/subnormals/Inf/NaN payloads, byte holes, "<<(preparation?"independent ARGB8 and three float feedback source formats":"every source format")<<" and exact full VM/padding passed\n";
}

void depth_output_transactions(unsigned preparation=0,bool feedback=false,bool queued=false)
{
    using namespace std::chrono_literals;
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);sample_transfer samples(ctx);
    unsigned cases=0,timeouts=0,rollbacks=0,quota=0,invalid=0,one_submission=0,old_readers=0,preload_checks=0;std::size_t exact=0;
    for(unsigned type=1;type<=4;++type)for(unsigned out:{2u,4u})for(unsigned count:{1u,2u,4u})for(unsigned mode=0;mode<8;++mode)
    {
        if(mode==7&&(feedback||count==1||(type!=3&&type!=4)))continue;
        @autoreleasepool
        {
            try
            {
                const auto format=static_cast<surface_depth>(type);const auto md=describe_surface_depth(format);const auto in=md.guest_bytes;
                auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
                const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
                surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
                auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
                const auto factor=count==1?sample_layout{1,1}:rsx_sample_layout(count);const unsigned pitch=4*out+1;
                std::vector<unsigned> addresses(4*(4*in+1));for(unsigned i=0;i<addresses.size();++i)addresses[i]=0xc0000+i;
                std::vector<std::pair<unsigned,unsigned>> owners(8*out,{~0u,~0u});
                for(unsigned j=0;j<2*in;++j)
                {
                    const auto pixel=(j*7+type)%16;unsigned lane=j%in;
                    if(count>1&&type==3)lane=mode==7&&j==0?0:1;
                    if(count>1&&type==4)lane=mode==7&&j==0?0:2+j%2;
                    const auto byte=pixel*in+lane,canonical=2*j+1;
                    addresses[pixel/4*(4*in+1)+pixel%4*in+lane]=0x6000+canonical/out/4*pitch+canonical%(4*out)+(j%2?0x100000:0);
                    owners[canonical]={0,byte};
                }
                std::vector<guest_storage_segment> segments;for(unsigned i=0;i<addresses.size();++i)segments.push_back({addresses[i],i,1});
                const auto target=surfaces.acquire({addresses.front(),{4/factor.x,4/factor.y,count,format,false},4*in+1,{},{},segments});
                const auto reader=surfaces.acquire({0xf0000,{4/factor.x,4/factor.y,count,format,false},4*in,{}});
                std::vector<std::uint32_t> original(32);std::vector<std::byte> guest(16*in),depth(4*256),stencil(4*256);
                for(unsigned i=0;i<16;++i)
                {
                    original[2*i]=initial_depth(format,i,count);original[2*i+1]=md.stencil?(i*53+type*19+7)&255:0;
                    const auto word=depth_guest(format,original[2*i],original[2*i+1]);for(unsigned lane=0;lane<in;++lane)guest[i*in+lane]=std::byte(word>>(8*(in-1-lane)));
                    std::memcpy(depth.data()+i/4*256+i%4*md.depth_bytes,&original[2*i],md.depth_bytes);stencil[i/4*256+i%4]=std::byte(original[2*i+1]);
                }
                surfaces.prepare(target,surface_initialization::background);
                surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images)
                {
                    auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;image=make_texture(commands,d);}
                    texture_region all;all.size=MTLSizeMake(4,4,1);buffer_to_texture(commands,make_upload(commands,depth),0,image,all,256,0,texture_plane::depth);if(md.stencil)buffer_to_texture(commands,make_upload(commands,stencil),0,image,all,256,0,texture_plane::stencil);if(count>1)samples.import(commands,image,images[0]);
                });
                surfaces.flush(target);surfaces.defer_copy(target,{0,0,4/factor.x,4/factor.y},reader,0,0);
                id<MTLTexture> old=nil;surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){old=images[0];});
                const auto read=[&](id<MTLTexture> image)
                {
                    const auto d=make_buffer(ctx,4*256,buffer_storage::shared),st=md.stencil?make_buffer(ctx,4*256,buffer_storage::shared):nil;ctx.begin();auto expanded=image;
                    if(count>1){auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;expanded=make_texture(ctx,descriptor);samples.expand(ctx,image,expanded);}
                    texture_region all;all.size=MTLSizeMake(4,4,1);texture_to_buffer(ctx,expanded,all,d,0,256,0,texture_plane::depth);if(md.stencil)texture_to_buffer(ctx,expanded,all,st,0,256,0,texture_plane::stencil);ctx.submit().wait();
                    std::vector<std::uint32_t> result(32);const auto bytes=map_buffer(d,0,d.length);const auto s=st?map_buffer(st,0,st.length):std::span<std::byte>{};for(unsigned i=0;i<16;++i){std::memcpy(&result[2*i],bytes.data()+i/4*256+i%4*md.depth_bytes,md.depth_bytes);result[2*i+1]=md.stencil?std::uint32_t(s[i/4*256+i%4]):0;}return result;
                };
                require(read(old)==original,"Depth transaction native seed differs");
                auto prepared=original;
                if(preparation==1||preparation==4)surfaces.spill(target);
                if(preparation>=2)
                {
                    const auto donor=surfaces.acquire({0x30000,{4/factor.x,4/factor.y,count,format,false},4*in,{}});
                    std::vector<std::uint32_t> donor_bits(32);std::vector<std::byte> d(4*256),st(4*256);
                    for(unsigned i=0;i<16;++i)
                    {
                        donor_bits[2*i]=initial_depth(format,i+3,count);donor_bits[2*i+1]=md.stencil?(i*47+type*23+19)&255:0;
                        std::memcpy(d.data()+i/4*256+i%4*md.depth_bytes,&donor_bits[2*i],md.depth_bytes);st[i/4*256+i%4]=std::byte(donor_bits[2*i+1]);
                    }
                    surfaces.prepare(donor,surface_initialization::background);
                    surfaces.draw(std::array{cached_surface_use{donor,true}},[&](context& commands,auto images)
                    {
                        auto image=images[0];if(count>1){auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;image=make_texture(commands,descriptor);}
                        texture_region all;all.size=MTLSizeMake(4,4,1);buffer_to_texture(commands,make_upload(commands,d),0,image,all,256,0,texture_plane::depth);if(md.stencil)buffer_to_texture(commands,make_upload(commands,st),0,image,all,256,0,texture_plane::stencil);if(count>1)samples.import(commands,image,images[0]);
                    });
                    const unsigned native_width=preparation==2?4/factor.x:2/factor.x;
                    surfaces.defer_copy(donor,{0,0,native_width,4/factor.y},target,0,0);
                    for(unsigned i=0;i<16;++i)if(i%4<native_width*factor.x){prepared[2*i]=donor_bits[2*i];prepared[2*i+1]=donor_bits[2*i+1];}
                }
                for(unsigned i=0;i<16;++i){const auto word=depth_guest(format,prepared[2*i],prepared[2*i+1]);for(unsigned lane=0;lane<in;++lane)guest[i*in+lane]=std::byte(word>>(8*(in-1-lane)));}
                require(surfaces.status(reader).deferred_copies==1,"Prepared depth fixture consumed its old reader");
                const auto source=surfaces.acquire({0x1000,{1,1,1,surface_color::argb8,false},4,{}});surfaces.prepare(source,surface_initialization::background);
                std::vector<std::byte> input=mode==7?std::vector<std::byte>(4,std::byte{0xff}):std::vector<std::byte>{std::byte{0xd3},std::byte{0x3b},std::byte{0x59},std::byte{0x71}};
                surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){texture_region one;one.size=MTLSizeMake(1,1,1);colors.upload(commands,surface_color::argb8,make_upload(commands,input),{},images[0],one);});
                scaled_image_plan plan{plan_guest_surface(0x1000,1,1,4),plan_guest_surface(0x6000,4,2,out,pitch),scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,1,1,4,2,0,0,false,false,true};
                if(feedback)
                {
                    input=guest;plan.source=plan_guest_surface(addresses.front(),in,4,4);plan.input_width=in;plan.input_height=4;
                    for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<4*in;++byte)plan.source_segments.push_back({addresses[y*(4*in+1)+byte],plan.source_segments.size(),1});
                }
                auto packed=plan;packed.source=plan_guest_surface(0,plan.source.width,plan.source.height,4);packed.source_segments.clear();packed.destination=plan_guest_surface(0,4,2,out);const auto payload=reference(packed,input,std::vector<std::byte>(8*out));
                auto expected=prepared;std::array<unsigned,16> masks{};
                for(unsigned j=0;j<owners.size();++j)if(owners[j].first!=~0u){const auto at=owners[j].second;guest[at]=payload[j];masks[at/in]|=1u<<(at%in);}
                for(unsigned i=0;i<16;++i){std::uint32_t word=0;for(unsigned lane=0;lane<in;++lane)word=word*256+std::uint32_t(guest[i*in+lane]);if(masks[i]&((1u<<(in-unsigned(md.stencil)))-1))expected[i*2]=depth_expanded(format,word);if(md.stencil&&(masks[i]&(1u<<(in-1))))expected[i*2+1]=word&255;}
                const auto before=memory->access->copy();auto expected_vm=before;const auto reads=memory->read_bytes,writes=std::size_t(memory->writes);const auto generation=surfaces.status(target).content_generation;const auto submissions=surfaces.transaction_statistics().submitted;
                const auto encoding_before=ctx.encoding_statistics();
                const auto before_loads=[&](context& commands)
                {
                    if(!preparation)return;
                    static_assert(std::has_unique_object_representations_v<command_encoding_statistics>);
                    const auto now=commands.encoding_statistics();
                    require(std::memcmp(&now,&encoding_before,sizeof(now))==0,"Prepared depth encoded native loads before its prerequisites");++preload_checks;
                };
                const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
                auto valid=std::make_shared<bool>(true);std::optional<submission> token;std::vector<id<MTLBuffer>> blockers;
                if(mode==6)
                {
                    (void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate}){auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}
                }
                memory->forbid_pixels=true;bool failed=false,returned=false;
                try
                {
                    const auto submit=queued?&scaled_image_transfer::enqueue:&scaled_image_transfer::execute;
                    (transfer.get()->*submit)(plan,mode==6||mode==7?30s:1ms,[&](context& commands){token=commands.recording_submission();before_loads(commands);if(mode==3)throw std::range_error("Depth output encode refused");if(mode!=7)commands.wait_event(gate,1);},[valid]{if(!*valid)throw scaled_image_mapping_conflict();});
                    if(queued)
                    {
                        returned=true;
                        if(mode!=7)require(token&&!token->ready()&&transfer->has_pending()&&!transfer->poll(),
                            "Queued native depth scale did not retain private pending work");
                        transfer->synchronize(mode==7?30s:1ms);
                    }
                }
                catch(const std::runtime_error&){failed=true;}
                if(queued&&mode!=3&&mode!=6)require(returned,"Queued native depth scale waited for GPU completion");
                require(failed,"Depth transaction completed through a gate, quota or invalid MSAA value");
                require(surfaces.status(target).content_generation==generation,"Depth output published before its successful completion");
                if(mode==6){require(!transfer->has_pending(),"Depth quota refusal retained work");blockers.clear();transfer->execute(plan,30s,before_loads);++quota;}
                else if(mode==3){require(token&&token->cancelled()&&!transfer->has_pending(),"Depth encode refusal retained work");}
                else if(mode==7){require(token&&token->ready(),"Invalid MSAA depth did not finish its failed transaction");ctx.recover_completed_errors();++invalid;}
                else
                {
                    require(token&&!token->ready()&&transfer->has_pending(),"Depth output timeout is not a live gated transaction");++timeouts;
                    if(mode==1)*valid=false;if(mode==2)require(token->cancel_pending(),"Depth queued cancellation refused");
                    if(mode==4)
                    {
                        unsigned at=0;while(at<addresses.size()&&addresses[at]<0xc0000)++at;require(at<addresses.size(),"Depth CPU writer has no unselected byte");
                        memory->raw->write({addresses[at],1},std::array{std::byte{0x5c}});expected_vm[addresses[at]%0x100000]=std::byte{0x5c};
                    }
                    if(mode==5)transfer.reset();gate.signaledValue=1;failed=false;
                    try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                    require(failed==(mode>=1&&mode<=5),"Depth transaction publication/rollback differs");if(mode==2)ctx.recover_completed_errors();
                }
                if(preparation)
                {
                    require(surfaces.transaction_statistics().submitted-submissions==unsigned(mode!=3),"Prepared depth output used a separate initialization submission");++one_submission;
                    require(surfaces.status(reader).deferred_copies==1,"Prepared depth transfer materialized its old reader");++old_readers;
                }
                const bool rejected=(mode>=1&&mode<=5)||mode==7;
                require(surfaces.status(target).content_generation==generation+unsigned(!rejected)*(1+unsigned(preparation>=2)),"Depth rollback changed target generation");
                require(read(old)==original,"Depth transaction changed retained old native precision");
                if(mode!=4)
                {
                    id<MTLTexture> image=nil;surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){image=images[0];});
                    require(read(image)==(rejected?prepared:expected),"Depth output current native planes differ after completion/rollback");if(rejected)++rollbacks;
                }
                exact+=16*(md.depth_bytes+unsigned(md.stencil));
                require(memory->read_bytes==reads&&memory->writes==writes,"Depth output transaction performed VM pixel work");
                require(memory->access->copy()==expected_vm,"Depth transaction changed VM before CPU observation");
                memory->forbid_pixels=false;
                if(!rejected){surfaces.prepare_cpu_read(plan.destination.range);for(unsigned y=0;y<4;++y)for(unsigned byte=0;byte<4*in;++byte)expected_vm[addresses[y*(4*in+1)+byte]%0x100000]=guest[y*4*in+byte];for(unsigned j=0;j<payload.size();++j)expected_vm[0x6000+j/out/4*pitch+j%(4*out)]=payload[j];require(memory->access->copy()==expected_vm,"Depth output CPU visibility changed guest words or padding");}
                ++cases;
            }
            catch(const std::exception& e){throw std::runtime_error("Depth transaction type="+std::to_string(type)+" out="+std::to_string(out)+" samples="+std::to_string(count)+" mode="+std::to_string(mode)+": "+e.what());}
        }
    }
    std::cout<<"preparation="<<preparation<<" feedback="<<feedback<<" Native depth output transactions: "<<cases<<" profiles, "<<timeouts<<" true event timeouts, "<<rollbacks<<" actual cache rollback reads, "<<quota<<" quota retries, "<<invalid<<" rejected out-of-domain MSAA outputs, "<<exact<<" exact native old-plane bytes, "<<one_submission<<" single-transaction checks, "<<old_readers<<" retained old readers, "<<preload_checks<<" prerequisite-before-load checks; all four formats, 1/2/4 samples, retained native precision, mixed cold B8, metadata/cancel/CPU-writer/teardown guards and exact guest visibility passed\n";
}

void wide_cold_output_rows(bool atlas=false)
{
    using namespace std::chrono_literals;context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);
    unsigned cases=0,rows_checked=0,refusals=0;std::size_t exact_bytes=0;
    for(unsigned bpp:{2u,4u})for(unsigned outcome=0;outcome<7;++outcome)
    @autoreleasepool
    {
        auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
        const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
        surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
        auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20,5);
        const unsigned limit=max_2d_texture_dimension(ctx.device()),width=limit/2+3,height=atlas?10:5,pitch=width*bpp+1;
        const auto format=bpp==2?surface_color::rgb565:surface_color::argb8;
        const auto source=surfaces.acquire({0x1000,{1,1,1,surface_color::argb8,false},4,{}}),target=surfaces.acquire({0x6000,{2,4,1,format,false},pitch,{}});
        const std::vector<std::byte> input{std::byte{0xee},std::byte{0x29},std::byte{0x57},std::byte{0x8d}},background(8*bpp,std::byte{0x6d});
        const auto seed=[&](const auto& surface,surface_color f,const auto& bytes,unsigned w,unsigned h)
        {surfaces.prepare(surface,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(w,h,1);colors.upload(commands,f,make_upload(commands,bytes),{},images[0],all);});};
        seed(source,surface_color::argb8,input,1,1);seed(target,format,background,2,4);
        scaled_image_plan plan{plan_guest_surface(0x1000,1,1,4),plan_guest_surface(0x6000,width,height,bpp,pitch),scaled_image_format::argb8,bpp==2?scaled_image_format::rgb565:scaled_image_format::argb8,1,1,width,height,0,0,false,false,true};
        auto one=plan;one.destination=plan_guest_surface(0,1,1,bpp);one.scaled_width=one.scaled_height=1;
        const auto word=reference(one,input,std::vector<std::byte>(bpp));
        std::vector<unsigned> addresses;
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)if(y>=4||x>=2)for(unsigned lane=0;lane<bpp;++lane)addresses.push_back(0x6000+y*pitch+x*bpp+lane);
        std::vector<cached_surface_ref> cold;std::vector<std::array<unsigned,2>> shapes;
        for(std::size_t at=0;at<addresses.size();)
        {
            const auto words=(addresses.size()-at)/bpp;const auto w=std::min<std::size_t>(limit,words),h=atlas?std::max<std::size_t>(1,words/limit):1;
            const auto bytes=w*h*bpp;std::vector<guest_storage_segment> segments;
            for(unsigned byte=0;byte<bytes;++byte)
            {const auto ea=addresses[at+byte];if(!segments.empty()&&std::uint64_t(segments.back().address)+segments.back().size==ea)++segments.back().size;else segments.push_back({ea,byte,1});}
            shapes.push_back({static_cast<unsigned>(w),static_cast<unsigned>(h)});cold.push_back(surfaces.acquire({segments.front().address,{w,h,1,format,false},w*bpp,{},{},segments}));at+=bytes;
        }
        require(cold.size()>=2&&cold.size()+1<=5,"Wide native holes did not require multiple cold rows");
        const auto old_status=surfaces.status(target);const auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes);
        const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
        auto valid=std::make_shared<bool>(true);std::optional<submission> token;std::vector<id<MTLBuffer>> blockers;
        if(outcome==6)
        {
            (void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();
            for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate})
            {auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}
        }
        memory->forbid_pixels=true;bool failed=false;
        try{transfer->execute(plan,outcome==6?30s:1ms,[&](context& commands){token=commands.recording_submission();if(outcome==3)throw std::range_error("Wide holes encode refused");commands.wait_event(gate,1);},[valid]{if(!*valid)throw scaled_image_mapping_conflict();});}catch(const std::runtime_error&){failed=true;}
        require(failed,"Wide cold rows completed through a closed gate");
        require(surfaces.status(target).content_generation==old_status.content_generation,"Wide holes published old target early");
        for(const auto& row:cold)require(!surfaces.status(row).initialized&&!surfaces.status(row).dirty,"Wide cold row published before output completion");
        if(outcome==6){require(!transfer->has_pending()&&(!token||token->cancelled()),"Wide cold-row quota refusal retained work");blockers.clear();transfer->execute(plan);++refusals;}
        else if(outcome==3)require(token&&token->cancelled()&&!transfer->has_pending(),"Wide cold-row encode cancellation retained work");
        else
        {
            require(token&&!token->ready()&&transfer->has_pending(),"Wide cold rows are not actually gated");
            if(outcome==1)*valid=false;if(outcome==2)require(token->cancel_pending(),"Wide row queued cancellation refused");if(outcome==4)memory->raw->write({addresses.back(),1},std::array{std::byte{0x5c}});if(outcome==5)transfer.reset();
            gate.signaledValue=1;failed=false;try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
            require(failed==(outcome>=1&&outcome<=5),"Wide cold-row publication or rollback differs");if(outcome==2)ctx.recover_completed_errors();
        }
        const bool rejected=outcome>=1&&outcome<=5;
        require(surfaces.status(target).content_generation==old_status.content_generation+unsigned(!rejected)&&!surfaces.status(target).faulted,"Wide row rollback changed old target generation");
        require(memory->read_bytes==reads&&memory->writes==writes,"Wide cold-row initialization captured or wrote VM pixels");
        for(unsigned i=0;i<cold.size();++i)
        {
            const auto state=surfaces.status(cold[i]);require(!state.pending&&!state.faulted&&state.initialized==!rejected&&state.dirty==!rejected&&state.content_generation==unsigned(!rejected),"Wide cold-row state escaped atomic publication");
            if(!rejected)
            {
                const auto raw=make_buffer(ctx,shapes[i][0]*shapes[i][1]*bpp,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{cold[i],false}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(shapes[i][0],shapes[i][1],1);colors.download(commands,format,images[0],all,raw,{});});
                const auto bytes=map_buffer(raw,0,raw.length);for(unsigned at=0;at<bytes.size();++at)require(bytes[at]==word[at%bpp],"Wide cold-row native pixels differ");exact_bytes+=bytes.size();
            }
            ++rows_checked;
        }
        if(outcome!=4)require(memory->access->copy()==vm,"Wide cold-row transfer changed VM bytes");else require(memory->raw->read({addresses.back(),1}).bytes[0]==std::byte{0x5c},"Wide cold-row rollback overwrote CPU writer");
        memory->forbid_pixels=false;
        if(!rejected)
        {
            surfaces.prepare_cpu_read(plan.destination.range);auto visible=vm;for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned lane=0;lane<bpp;++lane)visible[0x6000+y*pitch+x*bpp+lane]=word[lane];
            require(memory->access->copy()==visible,"Wide cold-row observation changed padding or lost native words");
        }
        ++cases;
    }
    std::cout<<(atlas?"Mixed native 2D output atlases: ":"Wide cold output rows: ")<<cases<<" gated/encode/cancel/metadata/CPU-writer/abandoned/quota-retry cases, "<<rows_checked<<(atlas?" cold target publication checks, ":" cold row publication checks, ")<<exact_bytes<<" exact native cold bytes, "<<refusals<<" quota retries; multiple hardware-width rows and old target publish or roll back together; no VM pixel access during transfer and exact padding passed\n";
}

void entirely_cold_output_rows(bool atlas=false,bool factor=false)
{
    using namespace std::chrono_literals;context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);surface_color_transfer colors(ctx);
    unsigned cases=0,rows_checked=0,timeouts=0,refusals=0;std::size_t exact_bytes=0;
    for(unsigned in:{2u,4u})for(unsigned out:{2u,4u})for(unsigned layout=0;layout<4;++layout)for(unsigned outcome=0;outcome<8;++outcome)
    @autoreleasepool
    {
        if(atlas&&layout!=0&&layout!=3)continue;
        try
        {
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
            auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20,5);
            const unsigned limit=max_2d_texture_dimension(ctx.device()),width=factor?3*limit/4-1:layout==1?2:limit+(layout==3?1:3),height=factor?8:atlas?7:layout==1?limit+1:layout==3?4:2,pitch=width*out+1;
            const auto sf=in==2?surface_color::rgb565:surface_color::argb8,df=out==2?surface_color::rgb565:surface_color::argb8;
            const auto source=surfaces.acquire({0x1000,{3,2,1,sf,false},3*in,{}});
            std::vector<std::byte> input(6*in);for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*43+i*i*7+21)&255);
            surfaces.prepare(source,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(3,2,1);colors.upload(commands,sf,make_upload(commands,input),{},images[0],all);});
            const auto sw=layout==2?std::optional<swizzle_region>{{{2*limit,4,1},{1,1,0},{width,height,1}}}:std::nullopt;
            scaled_image_plan plan{plan_guest_surface(0x1000,3,2,in),plan_guest_surface(0x6000,width,height,out,sw?0:pitch,sw),in==2?scaled_image_format::rgb565:scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,3,2,width+2,height+1,1,1,bool(layout&1),bool(layout&2),outcome!=7};
            if(layout==3){const auto size=plan.destination.range.size;plan.destination_segments={{0x6000,0,3},{0x6003+0x100000,3,size-8},{static_cast<unsigned>(0x6000+size-5),size-5,5}};}
            const auto ea=[&](std::size_t at)
            {
                if(plan.destination_segments.empty())return static_cast<unsigned>(0x6000+at);
                const auto it=std::find_if(plan.destination_segments.begin(),plan.destination_segments.end(),[&](const auto& p){return at>=p.offset&&at-p.offset<p.size;});require(it!=plan.destination_segments.end(),"Cold-only expected EA lacks a segment");return static_cast<unsigned>(it->address+at-it->offset);
            };
            std::vector<cached_surface_ref> previous;std::vector<cached_surface_status> previous_status;
            const std::vector<std::byte> old_word(out,std::byte{0x6d});
            if(factor)for(unsigned y=0;y<4;++y)
            {
                const auto target=surfaces.acquire({ea(y*pitch),{1,1,1,df,false},out,{}});surfaces.prepare(target,surface_initialization::background);
                surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(1,1,1);colors.upload(commands,df,make_upload(commands,old_word),{},images[0],all);});
                previous.push_back(target);previous_status.push_back(surfaces.status(target));
            }
            std::vector<unsigned> addresses,selected_words;for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)
            {if(factor&&x==0&&y<4)continue;selected_words.push_back(y*width+x);for(unsigned lane=0;lane<out;++lane)addresses.push_back(ea(offset(plan.destination,x,y)+lane));}
            std::vector<cached_surface_ref> cold;std::vector<std::array<unsigned,2>> native_shapes;std::vector<std::vector<unsigned>> native_words;
            if(layout==1||layout==2)
            {
                for(unsigned y=0;y<height;y+=limit)for(unsigned x=0;x<width;x+=limit)
                {
                    const auto w=std::min(limit,width-x),h=std::min(limit,height-y);auto selected=sw;
                    if(selected){selected->origin[0]+=x;selected->origin[1]+=y;selected->size={w,h,1};}
                    const auto address=static_cast<unsigned>(0x6000+(sw?0:y*pitch));
                    const auto segments=!sw&&h==1?std::vector<guest_storage_segment>{{address,0,pitch}}:std::vector<guest_storage_segment>{};
                    cold.push_back(surfaces.acquire({address,{w,h,1,df,false},sw?0:pitch,selected,{},segments}));native_shapes.push_back({w,h});
                    std::vector<unsigned> words;for(unsigned iy=0;iy<h;++iy)for(unsigned ix=0;ix<w;++ix)words.push_back((y+iy)*width+x+ix);native_words.push_back(std::move(words));
                }
            }
            else for(std::size_t at=0;at<addresses.size();)
            {
                const auto remaining_words=(addresses.size()-at)/out;const auto w=factor?limit-2:std::min<std::size_t>(limit,remaining_words),h=factor?6:atlas?std::max<std::size_t>(1,remaining_words/limit):1;
                const auto bytes=w*h*out;std::vector<guest_storage_segment> segments;
                for(unsigned byte=0;byte<bytes;++byte){const auto address=addresses[at+byte];if(!segments.empty()&&std::uint64_t(segments.back().address)+segments.back().size==address)++segments.back().size;else segments.push_back({address,byte,1});}
                native_shapes.push_back({static_cast<unsigned>(w),static_cast<unsigned>(h)});cold.push_back(surfaces.acquire({segments.front().address,{w,h,1,df,false},w*out,{},{},segments}));
                std::vector<unsigned> words;for(unsigned word=0;word<bytes/out;++word)words.push_back(selected_words[at/out+word]);native_words.push_back(std::move(words));at+=bytes;
            }
            require(cold.size()==(factor?1:atlas?2:layout==3?5:layout==0?3:2),"Cold-only output did not reach the expected native target count");
            for(const auto& row:cold)require(!surfaces.status(row).initialized&&!surfaces.status(row).dirty,"Cold-only output already had native pixels");
            const auto source_status=surfaces.status(source);const auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes);
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
            auto valid=std::make_shared<bool>(true);std::optional<submission> token;std::vector<id<MTLBuffer>> blockers;
            if(outcome==6)
            {
                (void)surfaces.reclaim_idle();(void)ctx.trim_upload_pages();
                for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate})
                {auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));state=ledger->statistics();attempt=state.budget-state.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}
            }
            memory->forbid_pixels=true;bool failed=false;
            try{transfer->execute(plan,outcome==6?30s:1ms,[&](context& commands){token=commands.recording_submission();if(outcome==3)throw std::range_error("Cold-only encode refused");commands.wait_event(gate,1);},[valid]{if(!*valid)throw scaled_image_mapping_conflict();});}catch(const std::runtime_error&){failed=true;}
            require(failed,"Entirely cold output completed through a closed gate");
            for(const auto& row:cold)require(!surfaces.status(row).initialized&&!surfaces.status(row).dirty,"Cold-only output published before its transaction");
            for(unsigned i=0;i<previous.size();++i)require(surfaces.status(previous[i]).content_generation==previous_status[i].content_generation,"Factored output published an old target early");
            if(outcome==6){require(!transfer->has_pending()&&(!token||token->cancelled()),"Cold-only quota refusal retained work");blockers.clear();transfer->execute(plan);++refusals;}
            else if(outcome==3)require(token&&token->cancelled()&&!transfer->has_pending(),"Cold-only encode refusal retained work");
            else
            {
                require(token&&!token->ready()&&transfer->has_pending(),"Cold-only output is not actually gated");++timeouts;
                if(outcome==1)*valid=false;if(outcome==2)require(token->cancel_pending(),"Cold-only queued cancellation refused");if(outcome==4)memory->raw->write({addresses.back(),1},std::array{std::byte{0x5c}});if(outcome==5)transfer.reset();
                gate.signaledValue=1;failed=false;try{if(transfer)transfer->synchronize();else surfaces.synchronize();}catch(const std::runtime_error&){failed=true;}
                require(failed==(outcome>=1&&outcome<=5),"Cold-only output publication or rollback differs");if(outcome==2)ctx.recover_completed_errors();
            }
            const bool rejected=outcome>=1&&outcome<=5;
            require(memory->read_bytes==reads&&memory->writes==writes,"Cold-only output captured or wrote VM pixels");
            require(surfaces.status(source).content_generation==source_status.content_generation&&!surfaces.status(source).faulted,"Cold-only transaction changed its source authority");
            std::vector<std::byte> expected;if(!rejected)expected=reference(plan,input,std::vector<std::byte>(plan.destination.range.size,std::byte{0xcd}));
            for(unsigned i=0;i<cold.size();++i)
            {
                const auto state=surfaces.status(cold[i]);require(!state.pending&&!state.faulted&&state.initialized==!rejected&&state.dirty==!rejected&&state.content_generation==unsigned(!rejected),"Cold-only rows escaped atomic publication");
                if(!rejected)
                {
                    const auto raw=make_buffer(ctx,native_words[i].size()*out,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{cold[i],false}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(native_shapes[i][0],native_shapes[i][1],1);colors.download(commands,df,images[0],all,raw,{});});
                    const auto bytes=map_buffer(raw,0,raw.length);for(unsigned at=0;at<bytes.size();++at){const auto word=native_words[i][at/out];require(bytes[at]==expected[offset(plan.destination,word%width,word/width)+at%out],"Cold-only native output differs from rational oracle");}exact_bytes+=bytes.size();
                }
                ++rows_checked;
            }
            for(unsigned i=0;i<previous.size();++i)
            {
                const auto state=surfaces.status(previous[i]);require(!state.pending&&!state.faulted&&state.content_generation==previous_status[i].content_generation+unsigned(!rejected),"Factored output changed old target generation on rollback");
                const auto raw=make_buffer(ctx,out,buffer_storage::shared);surfaces.draw(std::array{cached_surface_use{previous[i],false}},[&](context& commands,auto images){texture_region all;all.size=MTLSizeMake(1,1,1);colors.download(commands,df,images[0],all,raw,{});});
                const auto bytes=map_buffer(raw,0,raw.length);for(unsigned lane=0;lane<out;++lane)require(bytes[lane]==(rejected?old_word[lane]:expected[offset(plan.destination,0,i)+lane]),"Factored output old-target native rollback pixels differ");
            }
            require(memory->read_bytes==reads&&memory->writes==writes,"Factored output readback captured or wrote VM pixels");
            if(outcome!=4)require(memory->access->copy()==vm,"Cold-only transfer changed VM bytes");else require(memory->raw->read({addresses.back(),1}).bytes[0]==std::byte{0x5c},"Cold-only rollback overwrote CPU writer");
            memory->forbid_pixels=false;
            if(!rejected)
            {
                surfaces.prepare_cpu_read(plan.destination.range);auto visible=vm;
                for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned lane=0;lane<out;++lane)visible[ea(offset(plan.destination,x,y)+lane)%visible.size()]=expected[offset(plan.destination,x,y)+lane];
                require(memory->access->copy()==visible,"Cold-only CPU observation changed padding, mirror or source bytes");
            }
            ++cases;
        }
        catch(const std::exception& e){throw std::runtime_error("Cold-only in="+std::to_string(in)+" out="+std::to_string(out)+" layout="+std::to_string(layout)+" outcome="+std::to_string(outcome)+": "+e.what());}
    }
    std::cout<<(factor?"Factored fifth native output: ":atlas?"Entirely cold 2D output atlases: ":"Entirely cold native outputs: ")<<cases<<(atlas?" wide linear/split-mirror and RGB565/ARGB8 cross-format cases, ":" wide/tall/Morton/split-mirror and RGB565/ARGB8 cross-format cases, ")<<rows_checked<<(factor?" one-target publication checks with four old writers, ":atlas?" two-target publication checks, ":" two/three/five-target publication checks, ")<<timeouts<<" gated timeouts, "<<exact_bytes<<" exact rational native bytes, "<<refusals<<" quota retries; encode/metadata/cancel/CPU-writer/abandoned rollback, immutable source, exact VM padding and no transfer VM pixel access passed\n";
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
// The old full-image frozen checkout consumes a third/fourth surface heap.
// A selected 2x3 source window must fit while old deferred readers stay live.
void frozen_window_pressure_cases(unsigned composition=0)
{
    using namespace std::chrono_literals;
    context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,128u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);sample_transfer samples(ctx);
    unsigned cases=0,timeouts=0,rollbacks=0,old_reads=0,rollback_reads=0;std::size_t checked=0;
    for(unsigned count:{1u,2u,4u})for(unsigned out:{2u,4u})for(bool deferred:{false,true})for(bool clean:{false,true})for(unsigned outcome=0;outcome<6;++outcome)
    @autoreleasepool
    {
        if(composition&&!deferred)continue;
        try
        {
            const auto f=count==1?sample_layout{1,1}:rsx_sample_layout(count);
            const surface_allocation_description allocation{128/f.x,128/f.y,count,surface_color::argb8,count>1};
            auto donor_allocation=allocation;const auto donor_format=composition>=4?surface_color::gb8:surface_color::argb8;
            donor_allocation.format=donor_format;const auto donor_bpp=surface_guest_bytes(donor_format);
            std::size_t heap=0,donor_heap=0,window_bytes=0;
            @autoreleasepool
            {
                surface_pool probe(ctx,16u<<20);const auto lease=probe.acquire(allocation),donor=probe.acquire(donor_allocation);
                heap=lease->reserved_bytes();donor_heap=donor->reserved_bytes();
                if(composition>=4&&count>1)
                {
                    const auto window=probe.acquire({2,3,1,surface_color::argb8,false}),expanded=probe.acquire({8,8,1,donor_format,false});
                    window_bytes=window->reserved_bytes()+expanded->reserved_bytes();
                }
            }
            const auto budget=2*heap+(deferred?donor_heap:0)+std::max({std::size_t(16384),heap/8,window_bytes});
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_cache surfaces(ctx,memory,budget,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
            auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto source=surfaces.acquire({0x20000,allocation,512,{}}),reader=surfaces.acquire({0xc0000,allocation,512,{}});
            std::vector<std::byte> original(128*128*4),prepared;
            for(unsigned i=0;i<original.size();++i)original[i]=std::byte((i*37+i/7+19)&255);
            const auto seed=[&](const auto& target,const auto& words,surface_color format=surface_color::argb8)
            {
                @autoreleasepool
                {
                    surfaces.prepare(target,surface_initialization::background);
                    surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images)
                    {
                        auto image=images[0];if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:128 height:128 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;image=make_texture(commands,d);}
                        texture_region all;all.size=MTLSizeMake(128,128,1);colors.upload(commands,format,make_upload(commands,words),{},image,all);if(count>1)samples.import(commands,image,images[0]);
                    });
                }
            };
            seed(source,original);if(clean)surfaces.flush(source);
            id<MTLTexture> old=nil;surfaces.draw(std::array{cached_surface_use{source,false}},[&](context&,auto images){old=images[0];});
            surfaces.defer_copy(source,{0,0,128/f.x,128/f.y},reader,0,0);
            prepared=original;
            if(deferred)
            {
                if(composition==3)surfaces.spill(source);
                const auto donor=surfaces.acquire({0x60000,donor_allocation,128*donor_bpp,{}});
                std::vector<std::byte> words(128*128*donor_bpp);for(unsigned i=0;i<words.size();++i)words[i]=std::byte((i*71+i/11+107)&255);
                seed(donor,words,donor_format);
                if(composition==1)
                {
                    surfaces.defer_copy(donor,{0,0,126/f.x,128/f.y},source,0,0);
                    surfaces.defer_copy(donor,{126/f.x,0,2/f.x,128/f.y},source,126/f.x,0);prepared=words;
                }
                else if(composition>=2)
                {
                    const unsigned dw=composition==3?2:128,dh=composition==3?4:128;
                    const unsigned side=composition==5?128:8,origin=composition==5?0:4;
                    const clear_rectangle src{origin/f.x,origin/f.y,side/f.x,side/f.y},dst{composition==3?126/f.x:0,composition==3?124/f.y:0,dw/f.x,dh/f.y};
                    if(composition>=4)surfaces.defer_blit_typeless(donor,src,source,dst,{true,true,false});
                    else surfaces.defer_blit_nearest(donor,src,source,dst,texture_plane::all,{true,true,false});
                    const unsigned sw=side*donor_bpp/4;std::vector<std::byte> packed_donor(side*side*donor_bpp);
                    for(unsigned y=0;y<side;++y)std::copy_n(words.begin()+((origin+y)*128+origin)*donor_bpp,side*donor_bpp,packed_donor.begin()+y*side*donor_bpp);
                    const scaled_image_plan deferred_plan{plan_guest_surface(0,sw,side,4),plan_guest_surface(0,dw,dh,4),scaled_image_format::argb8,scaled_image_format::argb8,sw,side,dw,dh,0,0,true,true,false};
                    const auto patch=reference(deferred_plan,packed_donor,std::vector<std::byte>(dw*dh*4));
                    if(composition==3)for(unsigned y=0;y<dh;++y)std::copy_n(patch.begin()+y*dw*4,dw*4,prepared.begin()+(124+y)*512+126*4);
                    else prepared=patch;
                }
                else {prepared=words;surfaces.defer_copy(donor,{0,0,128/f.x,128/f.y},source,0,0);}
            }
            else surfaces.spill(source);
            // Empty both command slots before reclaiming genuinely idle heaps.
            for(unsigned i=0;i<2;++i){ctx.begin();ctx.submit().wait();}(void)surfaces.reclaim_idle();
            const unsigned input_x=composition?125:126;
            scaled_image_plan plan{plan_guest_surface(0x20000+125*512+input_x*4,2,3,4,512),plan_guest_surface(0x20000+10*512+8*4,2,2,out,512),scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,2,3,3,3,1,1,bool(outcome&1),bool(outcome&2),true};
            std::vector<std::byte> input(24);for(unsigned y=0;y<3;++y)std::copy_n(prepared.begin()+(125+y)*512+input_x*4,8,input.begin()+y*8);
            auto packed=plan;packed.source=plan_guest_surface(0,2,3,4);packed.destination=plan_guest_surface(0,2,2,out);
            const auto patch=reference(packed,input,std::vector<std::byte>(4*out));auto expected=prepared;
            for(unsigned y=0;y<2;++y)std::copy_n(patch.begin()+y*2*out,2*out,expected.begin()+(10+y)*512+8*4);
            const auto before=surfaces.status(source);const auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes);const auto submitted=surfaces.transaction_statistics().submitted;
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
            auto valid=std::make_shared<bool>(true);std::optional<submission> token;bool failed=false;
            memory->forbid_pixels=true;
            try
            {
                transfer->execute(plan,outcome?1ms:30s,[&](context& commands)
                {
                    token=commands.recording_submission();if(outcome==2)throw std::range_error("Frozen window encode refusal");if(outcome)commands.wait_event(gate,1);
                },[valid]{if(!*valid)throw scaled_image_mapping_conflict();});
            }
            catch(const surface_budget_error& e){throw std::runtime_error("Windowed frozen source exhausted full-image checkout budget: requested="+std::to_string(e.requested)+" reserved="+std::to_string(e.reserved)+" budget="+std::to_string(e.budget));}
            catch(const std::runtime_error&){failed=true;}
            const bool rejected=outcome>=2;
            if(outcome)
            {
                require(failed,"Frozen window completed through a closed gate");
                require(surfaces.status(source).content_generation==before.content_generation,"Frozen window published before completion");
                if(outcome==2)require(token&&token->cancelled(),"Frozen window encode refusal did not cancel");
                else
                {
                    require(token&&!token->ready()&&transfer->has_pending(),"Frozen window test did not retain real gated work");++timeouts;
                    if(outcome==3)*valid=false;if(outcome==4)require(token->cancel_pending(),"Frozen window queued cancellation refused");if(outcome==5)transfer.reset();
                    gate.signaledValue=1;
                    if(outcome!=5){bool conflict=false;try{transfer->synchronize(30s);}catch(const std::runtime_error&){conflict=true;}require(conflict==rejected,"Frozen window rejected transaction published");}
                    else{try{surfaces.synchronize(30s);}catch(const std::runtime_error&){} }
                }
            }
            if(outcome==4)ctx.recover_completed_errors();
            const auto after=surfaces.status(source);
            require(!after.pending&&after.content_generation==before.content_generation+unsigned(!rejected)*(deferred?2:1),"Frozen window generation differs");
            if(rejected){require(after.spilled==before.spilled&&after.deferred_copies==before.deferred_copies,"Frozen window rollback lost native source metadata");++rollbacks;}
            require(surfaces.transaction_statistics().submitted-submitted<=1,"Frozen window used a preparation submission");
            require(surfaces.memory_statistics().reserved_bytes<=budget,"Frozen window exceeded pool quota");
            require(memory->read_bytes==reads&&memory->writes==writes&&memory->access->copy()==vm,"Frozen window accessed VM pixels");
            require(surfaces.status(reader).deferred_copies==1,"Frozen window materialized an old reader");
            const auto read=[&](id<MTLTexture> image)
            {
                const auto raw=make_buffer(ctx,65536,buffer_storage::shared);ctx.begin();auto expanded=image;
                if(count>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:128 height:128 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;expanded=make_texture(ctx,d);samples.expand(ctx,image,expanded);}
                texture_region all;all.size=MTLSizeMake(128,128,1);colors.download(ctx,surface_color::argb8,expanded,all,raw,{});ctx.submit().wait();const auto bytes=map_buffer(raw,0,raw.length);return std::vector<std::byte>(bytes.begin(),bytes.end());
            };
            require(read(old)==original,"Frozen window changed an old deferred reader");++old_reads;
            if(!rejected)
            {
                id<MTLTexture> image=nil;surfaces.draw(std::array{cached_surface_use{source,false}},[&](context&,auto images){image=images[0];});require(read(image)==expected,"Frozen window output or native complement differs");checked+=expected.size();
                require(transfer->native_completed()==1&&transfer->vm_completed()==0,"Frozen window left native route");
                memory->forbid_pixels=false;surfaces.prepare_cpu_read({0x20000,65536});auto visible=vm;std::copy(expected.begin(),expected.end(),visible.begin()+0x20000);require(memory->access->copy()==visible,"Frozen window CPU visibility or VM padding differs");
            }
            if(rejected&&composition)
            {
                surfaces.cancel_deferred_copies(reader);old=nil;
                id<MTLTexture> restored=nil;surfaces.draw(std::array{cached_surface_use{source,false}},[&](context&,auto images){restored=images[0];});
                require(read(restored)==prepared,"Composed window actual cached rollback contents differ");
                require(memory->read_bytes==reads&&memory->writes==writes&&memory->access->copy()==vm,"Composed window rollback read used guest pixels");++rollback_reads;
            }
            ++cases;
        }
        catch(const std::exception& e){throw std::runtime_error("Frozen window composition="+std::to_string(composition)+" samples="+std::to_string(count)+" out="+std::to_string(out)+" deferred="+std::to_string(deferred)+" clean="+std::to_string(clean)+" outcome="+std::to_string(outcome)+": "+e.what());}
    }
    std::cout<<"composition="<<composition<<" Frozen source windows: "<<cases<<" tight-pool spill/deferred self-feedback cases, 1/2/4 samples, clean/dirty, RGB565/ARGB8 output, "<<timeouts<<" real gated timeouts, "<<rollbacks<<" rollbacks, "<<old_reads<<" unchanged old readers, "<<rollback_reads<<" actual cached rollback readbacks, "<<checked<<" exact full native output/background bytes; one transaction, no VM pixel access, quotas and later CPU visibility passed\n";
}

void frozen_input_recency_cases()
{
    using namespace std::chrono_literals;
    context ctx(2);
    auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);
    ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);
    unsigned cases=0,timeouts=0,rejections=0;
    for(unsigned bytes:{2u,4u})for(unsigned count:{1u,2u,4u})for(unsigned state=0;state<5;++state)
    for(bool cold:{false,true})for(unsigned outcome=0;outcome<4;++outcome)
    @autoreleasepool
    {
        try
        {
            auto memory=std::make_shared<native_memory>();
            memory->access->typed_identities=true;
            memory->access->retained_slices=true;
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};
            surface_cache surfaces(ctx,memory,8u<<20,ledger,capture);
            auto io=std::make_shared<io_mapping_epoch>();
            guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});
            scaled_image_transfer transfer(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto format=bytes==2?surface_color::rgb565:surface_color::argb8;
            const auto factors=count==1?sample_layout{1,1}:rsx_sample_layout(count);
            const surface_allocation_description allocation{8/factors.x,8/factors.y,count,format,false};
            std::vector<std::byte> original(64*bytes),other(original.size());
            for(unsigned i=0;i<original.size();++i){original[i]=std::byte((i*37+23)&255);other[i]=std::byte((i*53+71)&255);}
            memory->write({0x20000,original.size()},original);
            memory->write({0x60000,other.size()},other);
            const auto source=surfaces.acquire({0x20000,allocation,8*bytes,{}});
            surfaces.prepare(source);
            if(state==1||state==3)surfaces.spill(source);
            cached_surface_ref donor;
            auto prepared=original;
            if(state>=2)
            {
                donor=surfaces.acquire({0x60000,allocation,8*bytes,{}});
                const unsigned rows=state==2?8u:4u;
                surfaces.defer_copy(donor,{0,0,8/factors.x,rows/factors.y},source,0,0);
                std::copy_n(other.begin(),rows*8*bytes,prepared.begin());
            }
            // Make the competing clean allocation newer than the source.
            // Only the following frozen read should reverse their LRU order.
            const auto decoy=surfaces.acquire({0xa0000,allocation,8*bytes,{}});
            surfaces.prepare(decoy);
            const auto target=surfaces.acquire({0x10000,{2,2,1,surface_color::argb8,false},8,{}});
            if(!cold)surfaces.clear(target,0xf0,0x11223344,0,255,{0,0,2,2});
            const scaled_image_plan plan{plan_guest_surface(0x20000,8,8,bytes,8*bytes),plan_guest_surface(0x10000,2,2,4,8),
                bytes==2?scaled_image_format::rgb565:scaled_image_format::argb8,scaled_image_format::argb8,8,8,2,2,0,0,false,false,true};
            auto packed=plan;packed.source=plan_guest_surface(0,8,8,bytes);packed.destination=plan_guest_surface(0,2,2,4);
            const auto expected=reference(packed,prepared,std::vector<std::byte>(16));
            const auto before=surfaces.status(source);
            const auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes);
            memory->forbid_pixels=true;
            const auto gate=[ctx.device() newSharedEvent];
            struct release {id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};
            std::optional<submission> token;
            auto valid=std::make_shared<bool>(true);
            bool failed=false;
            try
            {
                transfer.execute(plan,outcome>=2?1ms:30s,[&](context& commands)
                {
                    token=commands.recording_submission();
                    if(outcome==1)throw std::range_error("Frozen recency encode refusal");
                    if(outcome>=2)commands.wait_event(gate,1);
                },[valid]{if(!*valid)throw scaled_image_mapping_conflict();});
            }
            catch(const std::runtime_error&){failed=true;}
            require(failed==bool(outcome),"Frozen recency request completion differs");
            if(outcome==1)require(token&&token->cancelled(),"Frozen recency encode refusal was not cancelled");
            if(outcome>=2)
            {
                require(token&&!token->ready()&&transfer.has_pending(),"Frozen recency lacks actual gated work");++timeouts;
                if(outcome==2)require(token->cancel_pending(),"Frozen recency queued cancellation failed");
                else *valid=false;
                gate.signaledValue=1;bool rejected=false;
                try{transfer.synchronize(30s);}catch(const std::runtime_error&){rejected=true;}
                require(rejected,"Frozen recency cancelled/invalid work was published");
                if(outcome==2)ctx.recover_completed_errors();
            }
            if(outcome)++rejections;
            const auto now=surfaces.status(source);
            require(now.initialized==before.initialized&&now.dirty==before.dirty&&now.spilled==before.spilled&&
                now.deferred_copies==before.deferred_copies&&now.content_generation==before.content_generation&&
                now.native_write_epoch==before.native_write_epoch&&now.guest_revisions==before.guest_revisions&&!now.faulted,
                "Frozen recency read changed source authority");
            require(memory->read_bytes==reads&&memory->writes==writes&&memory->access->copy()==vm,"Frozen recency read used guest pixels");
            if(!outcome)
            {
                const auto raw=make_buffer(ctx,16,buffer_storage::shared);
                surfaces.draw(std::array{cached_surface_use{target,false}},[&](context& commands,auto images)
                {texture_region all;all.size=MTLSizeMake(2,2,1);colors.download(commands,surface_color::argb8,images[0],all,raw,{});});
                const auto pixels=map_buffer(raw,0,16);
                require(std::equal(pixels.begin(),pixels.end(),expected.begin()),"Frozen recency output differs from rational oracle");
            }
            // Remove exclusions without preparing/touching the source again.
            // Cancellation drops deferred ownership, exposing its original clean
            // background to the same actual pressure policy as a resident reader.
            textures.clear();surfaces.cancel_deferred_copies(source);
            if(donor)surfaces.erase(donor);
            for(unsigned i=0;i<2;++i){ctx.begin();ctx.submit().wait();}
            require(!surfaces.status(source).dirty&&!surfaces.status(source).native_write_epoch&&surfaces.status(decoy).initialized,
                "Frozen recency pressure candidates are not reconstructible clean sources");
            require(surfaces.reclaim_clean(),"Frozen recency pressure had no eligible allocation");
            require(surfaces.status(source).initialized&&!surfaces.status(decoy).initialized,
                "Frozen reader was evicted before the older clean allocation");
            require(surfaces.reclaim_clean()&&!surfaces.status(source).initialized&&!surfaces.reclaim_clean(),
                "Frozen reader remained pinned instead of joining normal clean reclamation");
            require(memory->read_bytes==reads&&memory->writes==writes&&memory->access->copy()==vm,"Frozen recency reclamation touched guest pixels");
            memory->forbid_pixels=false;surfaces.prepare(source);
            require(surfaces.status(source).initialized&&!surfaces.status(source).dirty&&!surfaces.status(source).spilled,
                "Frozen recency source did not reconstruct after later eviction");
            ++cases;
        }
        catch(const std::exception& e){throw std::runtime_error("Frozen recency bytes="+std::to_string(bytes)+" samples="+std::to_string(count)+" state="+std::to_string(state)+" cold="+std::to_string(cold)+" outcome="+std::to_string(outcome)+": "+e.what());}
    }
    std::cout<<"Frozen native reader recency: "<<cases<<" RGB565/ARGB8 resident/spill/full-deferred/partial-spill/partial-resident profiles, 1/2/4 samples, existing/cold outputs; "<<timeouts<<" actual gated timeouts, "<<rejections<<" encode/cancellation/validation rejections; actual clean reclamation evicted the older allocation first, then the source, with no guest pixel access, no permanent pin, unchanged source authority, exact successful output and later reconstruction\n";
}

void isolated_native_input_cases()
{
    using namespace std::chrono_literals;context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,128u<<20);ctx.set_buffer_budget(ledger);
    surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);
    std::vector<surface_format> formats{surface_color::argb8,surface_color::x1rgb5_zero,surface_color::x1rgb5_one,surface_color::rgb565,surface_color::xrgb8_zero,surface_color::xrgb8_one,surface_color::b8,surface_color::gb8,surface_color::rgba16f,surface_color::rgba32f,surface_color::r32f,surface_color::xbgr8_zero,surface_color::xbgr8_one,surface_color::abgr8};
    for(unsigned i=1;i<=4;++i)formats.emplace_back(static_cast<surface_depth>(i));
    unsigned cases=0,timeouts=0,rollbacks=0,old_reads=0,quota=0,single=0;std::size_t exact=0;
    for(const auto format:formats)for(unsigned count:{1u,2u,4u})for(unsigned out:{2u,4u})for(unsigned state=0;state<5;++state)for(bool cold:{false,true})for(unsigned outcome=0;outcome<(cold?2u:9u);++outcome)
    @autoreleasepool
    {
        try
        {
            auto memory=std::make_shared<native_memory>();memory->access->typed_identities=true;memory->access->retained_slices=true;
            const cpu_capture_dispatch capture=[access=memory->access](const auto& work){access->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();
            surface_cache surfaces(ctx,memory,16u<<20,ledger,capture);guest_texture_cache textures(ctx,memory,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,ledger,64,{capture,io});auto transfer=std::make_unique<scaled_image_transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
            const auto f=count==1?sample_layout{1,1}:rsx_sample_layout(count);const unsigned bpp=surface_guest_bytes(format),pitch=64*bpp;
            const surface_allocation_description allocation{64/f.x,64/f.y,count,format,false};
            const auto source=surfaces.acquire({0x20000,allocation,pitch,{}}),reader=surfaces.acquire({0xc0000,allocation,pitch,{}});
            const auto linear=[&](context& commands,id<MTLTexture> image,unsigned w,unsigned h)
            {if(image.sampleCount==1)return image;auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:w height:h mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;return make_texture(commands,d);};
            const auto read=[&](id<MTLTexture> image,surface_format sf,unsigned w,unsigned h)
            {
                const auto raw=make_buffer(ctx,w*h*surface_guest_bytes(sf),buffer_storage::shared);ctx.begin();auto expanded=linear(ctx,image,w,h);if(image.sampleCount>1)samples.expand(ctx,image,expanded);
                texture_region all;all.size=MTLSizeMake(w,h,1);if(const auto* c=std::get_if<surface_color>(&sf))colors.download(ctx,*c,expanded,all,raw,{});else depths.download(ctx,std::get<surface_depth>(sf),expanded,all,raw,{});ctx.submit().wait();const auto bytes=map_buffer(raw,0,raw.length);return std::vector<std::byte>(bytes.begin(),bytes.end());
            };
            const auto seed=[&](const auto& target,surface_format sf,const auto& words,unsigned w,unsigned h)
            {surfaces.prepare(target,surface_initialization::background);surfaces.draw(std::array{cached_surface_use{target,true}},[&](context& commands,auto images){const auto image=linear(commands,images[0],w,h);texture_region all;all.size=MTLSizeMake(w,h,1);if(const auto* c=std::get_if<surface_color>(&sf))colors.upload(commands,*c,make_upload(commands,words),{},image,all);else depths.upload(commands,std::get<surface_depth>(sf),make_upload(commands,words),{},image,all);if(images[0].sampleCount>1)samples.import(commands,image,images[0]);});};
            std::vector<std::byte> words(4096*bpp);for(unsigned i=0;i<words.size();++i)words[i]=std::byte((i*37+i/11+23)&255);
            if(const auto* depth=std::get_if<surface_depth>(&format))for(unsigned i=0;i<4096;++i){const auto bits=initial_depth(*depth,i%16,count),word=depth_guest(*depth,bits,(i*53+19)&255);for(unsigned lane=0;lane<bpp;++lane)words[i*bpp+lane]=std::byte(word>>(8*(bpp-1-lane)));}
            seed(source,format,words,64,64);if(state==1)surfaces.flush(source);
            id<MTLTexture> old=nil;surfaces.draw(std::array{cached_surface_use{source,false}},[&](context&,auto images){old=images[0];});const auto original=read(old,format,64,64);auto prepared=original;
            surfaces.defer_copy(source,{0,0,64/f.x,64/f.y},reader,0,0);
            if(state<=1||state==3)surfaces.spill(source);
            if(state>=2)
            {
                const auto donor=surfaces.acquire({0x60000,allocation,pitch,{}});for(unsigned i=0;i<words.size();++i)words[i]^=std::byte{0x31};
                if(const auto* depth=std::get_if<surface_depth>(&format))for(unsigned i=0;i<4096;++i){const auto word=depth_guest(*depth,initial_depth(*depth,(i+3)%16,count),(i*17+73)&255);for(unsigned lane=0;lane<bpp;++lane)words[i*bpp+lane]=std::byte(word>>(8*(bpp-1-lane)));}
                seed(donor,format,words,64,64);id<MTLTexture> image=nil;surfaces.draw(std::array{cached_surface_use{donor,false}},[&](context&,auto images){image=images[0];});const auto donor_words=read(image,format,64,64);
                if(state==2){surfaces.defer_copy(donor,{0,0,64/f.x,64/f.y},source,0,0);prepared=donor_words;}
                else{surfaces.defer_copy(donor,{0,4/f.y,64/f.x,2/f.y},source,0,62/f.y);for(unsigned y=0;y<2;++y)std::copy_n(donor_words.begin()+(4+y)*pitch,pitch,prepared.begin()+(62+y)*pitch);}
            }
            const auto df=out==2?surface_color::rgb565:surface_color::argb8;const auto target=surfaces.acquire({0x10000,{2,2,1,df,false},2*out+3,{}});std::vector<std::byte> background(4*out,std::byte{0x6d});id<MTLTexture> old_target=nil;
            if(!cold){seed(target,df,background,2,2);surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){old_target=images[0];});background=read(old_target,df,2,2);const auto retained=surfaces.acquire({0xe0000,{2,2,1,df,false},2*out,{}});surfaces.defer_copy(target,{0,0,2,2},retained,0,0);}
            for(unsigned i=0;i<2;++i){ctx.begin();ctx.submit().wait();}(void)surfaces.reclaim_idle();
            const unsigned input_address=0x20000+61*pitch+pitch-8;
            scaled_image_plan plan{plan_guest_surface(input_address+((outcome&1)?0x100000:0),2,3,4,pitch),plan_guest_surface(0x10000,2,2,out,2*out+3),scaled_image_format::argb8,out==2?scaled_image_format::rgb565:scaled_image_format::argb8,2,3,3,3,1,1,bool(outcome&1),bool(outcome&2),true};
            std::vector<std::byte> input(24);for(unsigned y=0;y<3;++y)std::copy_n(prepared.begin()+(61+y)*pitch+pitch-8,8,input.begin()+y*8);
            auto packed=plan;packed.source=plan_guest_surface(0,2,3,4);packed.destination=plan_guest_surface(0,2,2,out);const auto expected=reference(packed,input,std::vector<std::byte>(4*out));
            const auto before=surfaces.status(source),target_before=surfaces.status(target);auto vm=memory->access->copy();const auto reads=memory->read_bytes,writes=std::size_t(memory->writes),submits=std::size_t(surfaces.transaction_statistics().submitted);
            const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} cleanup{gate};std::optional<submission> token;auto valid=std::make_shared<bool>(true);std::vector<id<MTLBuffer>> blockers;
            if(outcome==8){(void)ctx.trim_upload_pages();for(const auto storage:{MTLStorageModeShared,MTLStorageModePrivate}){auto s=ledger->statistics();auto attempt=s.budget-s.reserved_bytes;while(attempt){try{blockers.push_back(ledger->buffer(attempt,storage));s=ledger->statistics();attempt=s.budget-s.reserved_bytes;}catch(const heap_budget_error&){attempt/=2;}}}}
            memory->forbid_pixels=true;bool failed=false;
            try{transfer->execute(plan,outcome&&outcome!=8?1ms:30s,[&](context& commands){token=commands.recording_submission();if(outcome==2)throw std::range_error("Isolated source encode refusal");if(outcome&&outcome!=8)commands.wait_event(gate,1);},[valid]{if(!*valid)throw scaled_image_mapping_conflict();});}catch(const std::runtime_error&){failed=true;}
            if(surfaces.transaction_statistics().submitted-submits>1)throw std::runtime_error("Isolated native source used a preparation submission");
            if(outcome)
            {
                require(failed,"Isolated source completed through a closed gate or full quota");require(surfaces.status(target).content_generation==target_before.content_generation,"Isolated source published target early");
                if(outcome==8){require(!token&&!transfer->has_pending(),"Isolated source quota refusal retained GPU work");blockers.clear();transfer->execute(plan);++quota;}
                else if(outcome==2)require(token&&token->cancelled(),"Isolated source encode refusal did not cancel");
                else
                {
                    require(token&&!token->ready()&&transfer->has_pending(),"Isolated source lacks actual gated pending work");++timeouts;
                    if(outcome==3)*valid=false;if(outcome==4)require(token->cancel_pending(),"Isolated source queued cancellation failed");if(outcome==5)transfer.reset();
                    if(outcome==6||outcome==7){const unsigned address=outcome==6?input_address:0x10000;memory->raw->write({address,1},std::array{std::byte{0x5c}});vm[address]=std::byte{0x5c};}
                    gate.signaledValue=1;
                    bool conflict=false;try{if(transfer)transfer->synchronize(30s);else surfaces.synchronize(30s);}catch(const std::runtime_error&){conflict=true;}require(conflict==(outcome>=3),"Isolated source completion/rejection differs");
                }
            }
            if(outcome==4)ctx.recover_completed_errors();
            require(surfaces.transaction_statistics().submitted-submits<=1,"Isolated native source retry used a preparation submission");++single;
            const bool rejected=outcome>=2&&outcome<=7;const auto now=surfaces.status(source);
            require(now.initialized==before.initialized&&now.dirty==before.dirty&&now.spilled==before.spilled&&now.deferred_copies==before.deferred_copies&&now.content_generation==before.content_generation&&now.native_write_epoch==before.native_write_epoch&&now.guest_revisions==before.guest_revisions&&now.faulted==before.faulted,"Isolated scaled input consumed its original native source");
            require(memory->read_bytes==reads&&memory->writes==writes&&memory->access->copy()==vm,"Isolated source captured or published VM pixels");
            require(surfaces.status(reader).deferred_copies==1&&read(old,format,64,64)==original,"Isolated source changed an old reader");++old_reads;
            const auto final=surfaces.status(target);require(!final.pending&&!final.faulted&&final.content_generation==target_before.content_generation+unsigned(!rejected),"Isolated source changed target publication or rollback generation");
            if(rejected){if(!cold)require(read(old_target,df,2,2)==background,"Isolated source rollback changed old target bits");++rollbacks;}
            else
            {
                id<MTLTexture> result=nil;surfaces.draw(std::array{cached_surface_use{target,false}},[&](context&,auto images){result=images[0];});require(read(result,df,2,2)==expected,"Isolated source native output differs from rational oracle");exact+=expected.size();require(transfer->native_completed()==1&&transfer->vm_completed()==0,"Isolated source lost native routing");
                memory->forbid_pixels=false;surfaces.prepare_cpu_read(plan.destination.range);for(unsigned y=0;y<2;++y)std::copy_n(expected.begin()+y*2*out,2*out,vm.begin()+0x10000+y*(2*out+3));require(memory->access->copy()==vm,"Isolated source CPU visibility or padding differs");
            }
            ++cases;
        }
        catch(const std::exception& e){throw std::runtime_error("Isolated input format="+(std::holds_alternative<surface_color>(format)?std::string("C")+std::to_string(static_cast<unsigned>(std::get<surface_color>(format))):std::string("D")+std::to_string(static_cast<unsigned>(std::get<surface_depth>(format))))+" samples="+std::to_string(count)+" out="+std::to_string(out)+" state="+std::to_string(state)+" cold="+std::to_string(cold)+" outcome="+std::to_string(outcome)+": "+e.what());}
    }
    std::cout<<"Isolated native scaled inputs: "<<cases<<" 18-format clean/dirty raw-spill, full-deferred and partial resident/spill background profiles; 1/2/4 samples, RGB565/ARGB8, existing/cold targets and physical mirrors; "<<timeouts<<" real gated timeouts, "<<rollbacks<<" rollbacks, "<<old_reads<<" unchanged original readers, "<<quota<<" quota retries, "<<single<<" at-most-one-scale-transaction checks and "<<exact<<" exact native output bytes; source authority unchanged, zero VM pixel capture/write and later exact CPU visibility/padding passed\n";
}

int main(int argc,char** argv)
{
	@autoreleasepool
	{
		try
		{
            const bool vm_only=argc==2&&std::string_view(argv[1])=="vm-publication";
            if(argc>1&&!vm_only)
            {
                if(argc!=2)throw std::invalid_argument("Unknown scaled-image test selection");
                const std::string_view selection=argv[1];
                if(selection=="queued-depth-output")
                {
                    for(bool feedback:{false,true})depth_output_transactions(1,feedback,true);
                    std::cout<<"Queued native depth scale: all admitted transactions returned before required GPU observation; four depth formats,1/2/4 samples and joint cold-byte output retained private rollback\n";
                    return 0;
                }
                if(selection=="queued-multi-output")
                {
                    for(bool feedback:{false,true})multi_output_scaled_images(true,true,feedback,true,true,false,true,1,true);
                    std::cout<<"Queued multi-output scale: every admitted native transaction returned with the real GPU gate held; all targets remained private until joint retirement\n";
                    return 0;
                }
                if(selection=="queued-native-transactions"||selection=="previous-native-wait-policy")
                {
                    const bool previous=selection=="previous-native-wait-policy";
                    for(bool cold:{false,true})for(bool ram:{false,true})native_scaled_image_transactions(cold,ram,true,previous);
                    return 0;
                }
                if(selection=="input-readiness"){for(unsigned mode=1;mode<=5;++mode){native_scaled_image_cases(false,false,mode);native_scaled_image_cases(true,false,mode);}ram_source_prefix_lifetime();ram_source_prefix_lifetime(true);}
                else if(selection=="isolated-native-inputs")isolated_native_input_cases();
                else if(selection=="frozen-input-recency")frozen_input_recency_cases();
                else if(selection=="frozen-large-typeless")frozen_window_pressure_cases(5);
                else if(selection=="frozen-composed-windows")for(unsigned mode=1;mode<=4;++mode)frozen_window_pressure_cases(mode);
                else if(selection=="frozen-windows")frozen_window_pressure_cases();
                else if(selection=="inline-mixed-targets")for(unsigned preparation=1;preparation<=4;++preparation)mixed_color_byte_outputs(true,preparation);
                else if(selection=="inline-depth-targets")for(unsigned preparation=1;preparation<=4;++preparation)
                {
                    depth_output_transactions(preparation);
                    depth_output_transactions(preparation,true);
                }
                else if(selection=="inline-target-repro")multi_output_scaled_images(false,true,false,true,true,false,true,1);
                else if(selection=="inline-targets")for(unsigned preparation=1;preparation<=4;++preparation)
                {
                    multi_output_scaled_images(false,true,false,true,true,false,true,preparation);
                    multi_output_scaled_images(false,true,true,true,true,false,true,preparation);
                }
                else if(selection=="inline-target-transactions")for(unsigned preparation=1;preparation<=4;++preparation)
                {
                    multi_output_scaled_images(true,true,false,true,true,false,true,preparation);
                    multi_output_scaled_images(true,true,true,true,true,false,true,preparation);
                }
                else if(selection=="depth-output-transactions")depth_output_transactions();
                else if(selection=="mixed-depth-byte-outputs")mixed_color_byte_outputs(true);
                else if(selection=="mixed-color-byte-outputs")mixed_color_byte_outputs();
                else if(selection=="cold-byte-outputs"){multi_output_scaled_images(false,true,false,true,true,false,true);multi_output_scaled_images(false,true,true,true,true,false,true);}
                else if(selection=="cold-byte-transactions"){multi_output_scaled_images(true,true,false,true,true,false,true);multi_output_scaled_images(true,true,true,true,true,false,true);}
                else if(selection=="cold-byte-wide")multi_output_scaled_images(false,true,false,true,true,true,true);
                else if(selection=="byte-output-wide")multi_output_scaled_images(false,true,false,true,false,true,true);
                else if(selection=="byte-outputs"){multi_output_scaled_images(false,true,false,true,false,false,true);multi_output_scaled_images(false,true,true,true,false,false,true);}
                else if(selection=="byte-output-transactions"){multi_output_scaled_images(true,true,false,true,false,false,true);multi_output_scaled_images(true,true,true,true,false,false,true);}
                else if(selection=="wide-outputs")multi_output_scaled_images(false,false,false,false,false,true);
                else if(selection=="wide-output-holes")multi_output_scaled_images(false,true,false,true,true,true);
                else if(selection=="wide-output-transactions")multi_output_scaled_images(true,true,false,false,false,true);
                else if(selection=="factored-output")entirely_cold_output_rows(true,true);
                else if(selection=="output-atlases"){entirely_cold_output_rows(true);wide_cold_output_rows(true);}
                else if(selection=="cold-only-outputs")entirely_cold_output_rows();
                else if(selection=="wide-output-holes")wide_cold_output_rows();
                else if(selection=="output-holes"){multi_output_scaled_images(false,true,false,true,true);multi_output_scaled_images(false,true,true,true,true);}
                else if(selection=="output-hole-transactions"){multi_output_scaled_images(true,true,false,true,true);multi_output_scaled_images(true,true,true,true,true);}
                else if(selection=="disconnected-outputs"){multi_output_scaled_images(false,true,false,true);multi_output_scaled_images(false,true,true,true);}
                else if(selection=="disconnected-transactions"){multi_output_scaled_images(true,true,false,true);multi_output_scaled_images(true,true,true,true);}
                else if(selection=="feedback-outputs"){multi_output_scaled_images(false,false,true);multi_output_scaled_images(false,true,true);}
                else if(selection=="feedback-output-transactions"){multi_output_scaled_images(true,false,true);multi_output_scaled_images(true,true,true);}
                else if(selection=="partial-outputs")multi_output_scaled_images(false,true);
                else if(selection=="partial-output-transactions")multi_output_scaled_images(true,true);
                else if(selection=="multi-output"){multi_output_scaled_images();multi_output_fallbacks();}
                else if(selection=="multi-output-transactions")multi_output_scaled_images(true);
                else if(selection=="native-transactions"){native_scaled_image_transactions();native_scaled_image_transactions(true);native_scaled_image_transactions(false,true);native_scaled_image_transactions(true,true);}
                else if(selection=="deferred-transactions")deferred_scaled_image_transactions();
                else if(selection=="assembly"){assembled_scaled_image_cases();assembled_scaled_image_transactions();assembled_source_bounds();}
                else if(selection=="holes"){assembled_scaled_image_cases(true);assembled_scaled_image_transactions(true);mixed_source_lifetimes();assembled_source_bounds();}
                else if(selection=="hole-transactions"){assembled_scaled_image_transactions(true);mixed_source_lifetimes();assembled_source_bounds();}
                else if(selection=="clean"){assembled_scaled_image_cases(false,true);assembled_scaled_image_cases(true,true);assembled_scaled_image_transactions(false,true);assembled_scaled_image_transactions(true,true);}
                else if(selection=="clean-transactions"){assembled_scaled_image_transactions(false,true);assembled_scaled_image_transactions(true,true);assembled_source_bounds();}
                else if(selection=="bounds")assembled_source_bounds();
                else if(selection=="regions")bounded_scaled_image_cases();
                else if(selection=="overlap"){bounded_scaled_image_cases(true);overlapping_clean_transactions();}
                else if(selection=="overlap-transactions")overlapping_clean_transactions();
                else if(selection=="frozen-deferred")
                {
                    frozen_deferred_scaled_images();
                    frozen_deferred_scaled_images(1);
                    frozen_deferred_scaled_images(2);
                    overlapping_clean_transactions(true);
                    frozen_deferred_scaled_images(0,true);
                    overlapping_clean_transactions(true,true);
                    for(unsigned mode=1;mode<=4;++mode)frozen_deferred_scaled_images(0,false,mode);
                    overlapping_clean_transactions(true,false,true);
                    for(unsigned mode=1;mode<=2;++mode)
                    {
                        frozen_deferred_scaled_images(0,false,0,mode);
                        overlapping_clean_transactions(true,false,false,mode);
                        overlapping_clean_transactions(true,false,false,mode,true);
                    }
                }
                else if(selection=="frozen-transactions")overlapping_clean_transactions(true);
                else if(selection=="interleaved-deferred"){frozen_deferred_scaled_images(0,true);overlapping_clean_transactions(true,true);}
                else if(selection=="partial-deferred"){for(unsigned mode=1;mode<=4;++mode)frozen_deferred_scaled_images(0,false,mode);overlapping_clean_transactions(true,false,true);}
                else if(selection=="unused-deferred")
                    for(unsigned mode=1;mode<=2;++mode)
                    {
                        frozen_deferred_scaled_images(0,false,0,mode);
                        overlapping_clean_transactions(true,false,false,mode);
                    }
                else if(selection=="versioned-donor")
                    for(unsigned mode=1;mode<=2;++mode)
                        overlapping_clean_transactions(true,false,false,mode,true);
                else if(selection=="depth-z16"){depth_scaled_image_cases(false);depth_scaled_image_transactions(false);}
                else if(selection=="depth-z24"){depth_scaled_image_cases(true);depth_scaled_image_transactions(true);}
                else if(selection=="depth-transactions"){depth_scaled_image_transactions(false);depth_scaled_image_transactions(true);}
                else throw std::invalid_argument("Unknown scaled-image test selection");
                return 0;
            }
            if(!vm_only)
            {
			wide_cold_output_rows();
            entirely_cold_output_rows();
            entirely_cold_output_rows(true);
            entirely_cold_output_rows(true,true);
            wide_cold_output_rows(true);
			multi_output_scaled_images(false,true,false,true,true);
			multi_output_scaled_images(false,true,true,true,true);
			multi_output_scaled_images(true,true,false,true,true);
			multi_output_scaled_images(true,true,true,true,true);
			multi_output_scaled_images(false,true,false,true);
			multi_output_scaled_images(false,true,true,true);
			multi_output_scaled_images(true,true,false,true);
			multi_output_scaled_images(true,true,true,true);
			multi_output_scaled_images(false,false,true);
			multi_output_scaled_images(false,true,true);
			multi_output_scaled_images(true,false,true);
			multi_output_scaled_images(true,true,true);
			multi_output_scaled_images(false,true);
			multi_output_scaled_images(true,true);
			multi_output_fallbacks();
			multi_output_scaled_images();
			multi_output_scaled_images(true);
			assembled_scaled_image_cases();
			assembled_scaled_image_transactions();
			assembled_source_bounds();
			assembled_scaled_image_cases(true);
			assembled_scaled_image_transactions(true);
			mixed_source_lifetimes();
			rational_filter_boundaries();
            frozen_input_recency_cases();
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
            }
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
			const auto gathered = make_buffer(ctx, packed.range.size, buffer_storage::private_gpu);
			ctx.begin();
			const auto before = ctx.encoding_statistics();
			const std::array copies{display_storage_copy{0, 0, packed.range.size, 0}};
			compute.gather(ctx, raw, gathered, copies, 0);
			compute.encode(ctx, {packed, expanded, scaled_image_format::rgb565, scaled_image_format::argb8, 65536, 1, 65536, 1}, gathered, rgba_buffer);
			compute.encode(ctx, {expanded, packed, scaled_image_format::argb8, scaled_image_format::rgb565, 65536, 1, 65536, 1}, rgba_buffer, roundtrip);
			const auto after = ctx.encoding_statistics();
			require(after.compute_encoders == before.compute_encoders + 1 && after.checked_dispatches == before.checked_dispatches + 3 &&
				after.dispatch_barriers == before.dispatch_barriers + 2, "Gather and scaled roundtrip must preserve two dependencies in one encoder");
			ctx.submit().wait();
			const auto colors = map_buffer(rgba_buffer, 0, expanded.range.size);
			for (unsigned i = 0; i < 65536; ++i)
			{
				const std::array expected{255u, unsigned(std::lround(((i >> 11) & 31) * 255. / 31)), unsigned(std::lround(((i >> 5) & 63) * 255. / 63)), unsigned(std::lround((i & 31) * 255. / 31))};
				for (unsigned c = 0; c < 4; ++c)
					require(colors[i * 4 + c] == std::byte(expected[c]), "RGB565 full-domain expansion differs");
			}
			const auto restored = map_buffer(roundtrip, 0, packed.range.size);
			require(std::equal(restored.begin(), restored.end(), words565.begin()), "RGB565 full-domain roundtrip differs");
			std::cout << "Three checked gather/scale dispatches and two dependencies preserve all 65536 RGB565 words\n";
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
