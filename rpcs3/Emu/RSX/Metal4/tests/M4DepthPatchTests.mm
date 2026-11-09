#include "M4SurfaceDepth.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::uint32_t load(std::span<const std::byte> data, unsigned count, bool be)
{
    std::uint32_t word=0;for(unsigned i=0;i<count;++i)word|=std::uint32_t(data[i])<<(8*(be?count-1-i:i));return word;
}
void store(std::span<std::byte> data, unsigned count, std::uint32_t word, bool be)
{ for(unsigned i=0;i<count;++i)data[i]=std::byte(word>>(8*(be?count-1-i:i))); }
std::uint32_t guest_word(surface_depth format, std::uint32_t bits, unsigned stencil)
{
    if(format==surface_depth::z16_uint)return bits;
    if(format==surface_depth::z16_float)return ((bits-(120u<<23))>>11)&65535;
    if(format==surface_depth::z24s8_float)return ((bits>>7)<<8)|stencil;
    const auto f=std::bit_cast<float>(bits);const auto depth=std::isnan(f)?0.f:std::clamp(f,0.f,1.f);
    return (static_cast<std::uint32_t>(depth*16777215.f)<<8)|stencil;
}
std::uint32_t expanded(surface_depth format, std::uint32_t raw)
{
    if(format==surface_depth::z16_uint)return raw;
    if(format==surface_depth::z16_float)
    {
        // Numerical E4M12 reference, independent of the GPU shift/add codec.
        return std::bit_cast<std::uint32_t>(std::ldexp(1.f+float(raw%4096)/4096,int(raw/4096)-7));
    }
    if(format==surface_depth::z24s8_float)return (raw>>8)*128;
    const auto integer=raw>>8;
    float f=static_cast<float>(double(integer)/16777215.0);
    const auto contracted=static_cast<std::uint32_t>(f*16777215.f);
    if(contracted<integer)f=std::nextafter(f,1.f);
    else if(contracted>integer)f=std::nextafter(f,0.f);
    require(static_cast<std::uint32_t>(f*16777215.f)==integer,"CPU exact Z24 patch interval is absent");
    return std::bit_cast<std::uint32_t>(f);
}
void masks(context& ctx,surface_depth_transfer& codec)
{
    unsigned profiles=0,pixels=0,guards=0;std::size_t bytes_checked=0;
    for(unsigned type=1;type<=4;++type)for(unsigned placement=0;placement<2;++placement)
    @autoreleasepool
    {
        const auto format=static_cast<surface_depth>(type);const auto md=describe_surface_depth(format);
        auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:md.format width:placement?20:10 height:placement?18:9 mipmapped:placement];
        descriptor.textureType=placement?MTLTextureType2DArray:MTLTextureType2D;descriptor.arrayLength=placement?2:1;
        descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
        const auto image=make_texture(ctx,descriptor);
        texture_region whole;whole.level=placement;whole.slice=placement;whole.size=MTLSizeMake(10,9,1);
        texture_region selected=whole;selected.origin=MTLOriginMake(1,2,0);selected.size=MTLSizeMake(7,5,1);
        std::vector<std::byte> depth(256*9,std::byte{0xa5}),stencil(256*9,std::byte{0xb6});
        constexpr std::array<std::uint32_t,12> floats{0,0x80000000,1,0x3eaaaaab,0x3f123457,0x3f800000,0xbf000001,0x7f800000,0xff800000,0x7f800001,0x7fc12345,0xffffffff};
        for(unsigned i=0;i<90;++i)
        {
            const auto bits=type==1?(i*7919+13)&65535:floats[i%floats.size()];
            store(std::span(depth).subspan(i/10*256+i%10*md.depth_bytes),md.depth_bytes,bits,false);stencil[i/10*256+i%10]=std::byte((i*53+7)&255);
        }
        auto expected_depth=depth,expected_stencil=stencil;
        const unsigned pitch=10*md.guest_bytes,offset=3;
        std::vector<std::byte> guest(offset+5*pitch+7,std::byte{0xc7}),mask(guest.size(),std::byte{0});
        for(unsigned y=0;y<5;++y)for(unsigned x=0;x<7;++x)
        {
            const auto bits=load(std::span(depth).subspan((y+2)*256+(x+1)*md.depth_bytes),md.depth_bytes,false);
            const auto s=std::uint32_t(stencil[(y+2)*256+x+1]);
            const auto raw=guest_word(format,bits,s);const auto at=offset+y*pitch+x*md.guest_bytes;
            store(std::span(guest).subspan(at),md.guest_bytes,raw,true);
            const unsigned select=(y*7+x)%(1u<<md.guest_bytes);bool write_depth=false;
            for(unsigned lane=0;lane<md.guest_bytes;++lane)if(select&(1u<<lane))
            {
                mask[at+lane]=std::byte{1};guest[at+lane]^=std::byte((x*13+y*29+0x5b)&255);
                if(lane<md.guest_bytes-unsigned(md.stencil))write_depth=true;
            }
            const auto assembled=load(std::span(guest).subspan(at),md.guest_bytes,true);
            if(write_depth)store(std::span(expected_depth).subspan((y+2)*256+(x+1)*md.depth_bytes),md.depth_bytes,expanded(format,assembled),false);
            if(md.stencil&&(select&(1u<<(md.guest_bytes-1))))expected_stencil[(y+2)*256+x+1]=std::byte(assembled);
            ++pixels;
        }
        const auto raw=make_upload(ctx,guest),selection=make_upload(ctx,mask);
        const auto result_depth=make_buffer(ctx,depth.size(),buffer_storage::shared),result_stencil=md.stencil?make_buffer(ctx,stencil.size(),buffer_storage::shared):nil;
        ctx.begin();
        buffer_to_texture(ctx,make_upload(ctx,depth),0,image,whole,256,0,texture_plane::depth);
        if(md.stencil)buffer_to_texture(ctx,make_upload(ctx,stencil),0,image,whole,256,0,texture_plane::stencil);
        codec.patch(ctx,format,raw,selection,{offset,pitch},image,selected);
        fill_buffer(ctx,result_depth,0,result_depth.length,0xa5);
        texture_to_buffer(ctx,image,whole,result_depth,0,256,0,texture_plane::depth);
        if(md.stencil){fill_buffer(ctx,result_stencil,0,result_stencil.length,0xb6);texture_to_buffer(ctx,image,whole,result_stencil,0,256,0,texture_plane::stencil);}
        ctx.submit().wait();
        require(std::ranges::equal(map_buffer(result_depth,0,result_depth.length),expected_depth),"Masked patch changed selected or untouched raw native depth bits");bytes_checked+=depth.size();
        if(md.stencil){require(std::ranges::equal(map_buffer(result_stencil,0,result_stencil.length),expected_stencil),"Masked patch changed selected or untouched native stencil bytes");bytes_checked+=stencil.size();}
        require(std::ranges::equal(map_buffer(raw,0,raw.length),guest)&&std::ranges::equal(map_buffer(selection,0,selection.length),mask),"Depth patch changed its source guest/mask bytes");
        ctx.begin();
        for(unsigned bad=0;bad<4;++bad)
        {
            bool refused=false;try
            {
                if(bad==0)codec.patch(ctx,format,raw,nil,{offset,pitch},image,selected);
                if(bad==1)codec.patch(ctx,format,raw,make_buffer(ctx,2,buffer_storage::shared),{offset,pitch},image,selected);
                if(bad==2)codec.patch(ctx,format,raw,selection,{guest.size(),pitch},image,selected);
                if(bad==3){auto outside=selected;outside.size.width=10;codec.patch(ctx,format,raw,selection,{offset,pitch},image,outside);}
            }catch(const std::invalid_argument&){refused=true;}catch(const std::out_of_range&){refused=true;}
            require(refused,"Invalid depth patch guest/mask/texture bounds were accepted");++guards;
        }
        ctx.submit().wait();++profiles;
    }
    std::cout<<"Depth byte masks: "<<profiles<<" raw-plane profiles, "<<pixels<<" independent pixel decisions, "<<bytes_checked<<" exact native/padding bytes, "<<guards<<" range refusals; all four formats, every lane mask, signed-zero/subnormal/Inf/NaN and native low mantissa bits, pitched guest/unaligned offset, mip/array/origin passed\n";
}
void exact_z24(context& ctx,surface_depth_transfer& codec)
{
    constexpr unsigned side=4096,pixels=side*side,bytes=pixels*4;
    std::vector<std::byte> words(bytes);for(unsigned i=0;i<pixels;++i)store(std::span(words).subspan(i*4),4,(i<<8)|((i*37)&255),true);
    const auto guest=make_upload(ctx,words),mask=make_buffer(ctx,bytes,buffer_storage::private_gpu),output=make_buffer(ctx,bytes,buffer_storage::shared);
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:side height:side mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    const auto image=make_texture(ctx,d);texture_region all;all.size=MTLSizeMake(side,side,1);
    ctx.begin();codec.upload(ctx,surface_depth::z24s8_uint,guest,{},image,all);fill_buffer(ctx,mask,0,bytes,1);
    codec.patch(ctx,surface_depth::z24s8_uint,guest,mask,{},image,all);codec.download(ctx,surface_depth::z24s8_uint,image,all,output,{});ctx.submit().wait();
    require(std::ranges::equal(map_buffer(output,0,bytes),words),"Exact Z24 byte patch failed exhaustive integer gather roundtrip");
    std::cout<<"Exact Z24 patch: all "<<pixels<<" integer depth values and stencil payloads, "<<bytes<<" bit-exact guest bytes passed; ordinary upload rounding unchanged\n";
}
}
int main()
{
    std::cout.setf(std::ios::unitbuf);
    @autoreleasepool {try{metal4::context ctx(2);metal4::surface_depth_transfer codec(ctx);masks(ctx,codec);exact_z24(ctx,codec);ctx.wait_idle();}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
}
