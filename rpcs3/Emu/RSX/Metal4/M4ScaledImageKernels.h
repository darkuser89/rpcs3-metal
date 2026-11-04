#pragma once
#include <string_view>
namespace metal4
{
inline constexpr std::string_view scaled_image_kernels = R"metal(
#include <metal_stdlib>
using namespace metal;
struct layout { ulong row; uint width,height,bytes,morton; uint log_x,log_y,origin_x,origin_y; uint tiled,tile_base,tile_size,tile_pitch,tile_bank,tile_offset,tile_factor,tile_columns; };
struct parameters { layout source,destination; uint input_w,input_h,scaled_w,scaled_h,clip_x,clip_y,flip_x,flip_y,linear,padding; };
ulong rpcs3_pixel_offset(uint2 p,constant layout& l)
{
    ulong offset=ulong(p.y)*l.row+ulong(p.x)*l.bytes;
    if(l.morton) {
    p+=uint2(l.origin_x,l.origin_y);
    ulong index=0;uint output=0;
    for(uint bit=0;bit<max(l.log_x,l.log_y);++bit) {
        if(bit<l.log_x) index|=ulong((p.x>>bit)&1)<<output++;
        if(bit<l.log_y) index|=ulong((p.y>>bit)&1)<<output++;
    }
    offset=index*l.bytes;
    }
    if(!l.tiled) return offset;
    ulong logical=ulong(l.tile_offset)+offset;
    if(logical>=l.tile_size) return ~0ul;
    uint address=l.tile_base+uint(logical),texel=uint(logical)/256;
    uint tile_x=texel%l.tile_columns,tile_y=(texel/l.tile_columns)/64;
    uint selector=(tile_y*l.tile_columns+tile_x+(l.tile_base>>14))&0x3ffff;
    const uint banks[16]={0,1,2,3,2,3,0,1,1,2,3,0,3,0,1,2};
    uint bank=selector&3;
    if(l.tile_factor==2) bank=banks[((selector+((tile_y&1)<<1))&3)*4+(tile_y&3)];
    else if(l.tile_factor>=4) bank=banks[(selector&3)*4+(tile_y&3)];
    bank=(bank+l.tile_bank)&3;
    uint line=(texel/l.tile_columns)%64;
    uint column=(((line>>3)&7)<<7)|(((address>>5)&7)<<4)|((line&3)<<2);
    uint partition=(((line>>2)&1)+((address>>6)&1))&1;
    uint physical=(((selector>>2)&65535)<<16)|(bank<<14)|(((column>>4)&63)<<8)|(partition<<7)|(((column>>2)&3)<<5)|(address&31);
    physical^=(((physical>>12)^((bank^selector)&1)^(physical>>14))&1)<<9;
    physical^=((physical>>11)&1)<<10;
    uint relative=physical-l.tile_base;
    return relative<=l.tile_size-l.bytes?ulong(relative):~0ul;
}
uint4 read_pixel(device const uchar* data,int2 p,constant layout& l)
{
    p=clamp(p,int2(0),int2(l.width-1,l.height-1));
    ulong offset=rpcs3_pixel_offset(uint2(p),l);
    if(offset==~0ul) return l.bytes==2?uint4(0,0,0,1):uint4(0);
    if(l.bytes==4) return uint4(data[offset+1],data[offset+2],data[offset+3],data[offset]);
    uint word=(uint(data[offset])<<8)|uint(data[offset+1]);
    return uint4((word>>11)&31,(word>>5)&63,word&31,1);
}
kernel void rpcs3_scaled_image(device const uchar* source [[buffer(0)]],device uchar* destination [[buffer(1)]],
    constant parameters& p [[buffer(2)]],uint2 pixel [[thread_position_in_grid]])
{
    if(pixel.x>=p.destination.width||pixel.y>=p.destination.height) return;
    uint2 output=pixel+uint2(p.clip_x,p.clip_y);
    // Keep affine centers and bilinear weights rational. At half-channel
    // boundaries,float coordinates can move a 565 channel to the wrong word.
    // Dimensions <=65536 bound denominator products by 2^34 and each
    // rescaled channel numerator below 2^50,including cross-format copies.
    ulong4 channels;ulong divisor=1;
    if(!p.linear) {
        // Exact rational center mapping;flip BEFORE integer truncation.
        ulong nx=(ulong(output.x)*2+1)*p.input_w,dx=ulong(p.scaled_w)*2;
        ulong ny=(ulong(output.y)*2+1)*p.input_h,dy=ulong(p.scaled_h)*2;
        if(p.flip_x) nx=ulong(p.input_w)*dx-nx;
        if(p.flip_y) ny=ulong(p.input_h)*dy-ny;
        int2 selected=int2(nx/dx,ny/dy);
        if(p.source.bytes==p.destination.bytes) {
            selected=clamp(selected,int2(0),int2(p.source.width-1,p.source.height-1));
            ulong from=rpcs3_pixel_offset(uint2(selected),p.source),to=rpcs3_pixel_offset(pixel,p.destination);
            if(to==~0ul) return;
            for(uint i=0;i<p.source.bytes;++i) destination[to+i]=from==~0ul?0:source[from+i];
            return;
        }
        channels=ulong4(read_pixel(source,selected,p.source));
    } else {
        long dx=long(p.scaled_w)*2,dy=long(p.scaled_h)*2;
        long nx=(long(output.x)*2+1)*p.input_w,ny=(long(output.y)*2+1)*p.input_h;
        if(p.flip_x) nx=long(p.input_w)*dx-nx;
        if(p.flip_y) ny=long(p.input_h)*dy-ny;
        nx-=p.scaled_w;ny-=p.scaled_h;
        // Signed integer division truncates toward zero; sampling needs floor.
        long ax=nx/dx-(nx<0&&nx%dx!=0),ay=ny/dy-(ny<0&&ny%dy!=0);
        ulong wx=ulong(nx-ax*dx),wy=ulong(ny-ay*dy);
        int2 a=int2(ax,ay);
        channels=ulong4(read_pixel(source,a,p.source))*((ulong(dx)-wx)*(ulong(dy)-wy))+
            ulong4(read_pixel(source,a+int2(1,0),p.source))*(wx*(ulong(dy)-wy))+
            ulong4(read_pixel(source,a+int2(0,1),p.source))*((ulong(dx)-wx)*wy)+
            ulong4(read_pixel(source,a+int2(1,1),p.source))*(wx*wy);
        divisor=ulong(dx)*ulong(dy);
    }
    ulong to=rpcs3_pixel_offset(pixel,p.destination);
    if(to==~0ul) return;
    uint4 source_max=p.source.bytes==2?uint4(31,63,31,1):uint4(255);
    uint4 destination_max=p.destination.bytes==2?uint4(31,63,31,1):uint4(255);
    uint4 v;
    for(uint c=0;c<4;++c) {
        ulong denominator=divisor*source_max[c];
        v[c]=uint((channels[c]*destination_max[c]+denominator/2)/denominator);
    }
    if(p.destination.bytes==4) {
        destination[to]=uchar(v.w);destination[to+1]=uchar(v.x);destination[to+2]=uchar(v.y);destination[to+3]=uchar(v.z);
    } else {
        uint word=(v.x<<11)|(v.y<<5)|v.z;
        destination[to]=uchar(word>>8);destination[to+1]=uchar(word);
    }
}
struct gather_copy {ulong target,source,ordinal,size;};
kernel void rpcs3_scaled_gather(device const uchar* source [[buffer(0)]],device uchar* destination [[buffer(1)]],
    device const gather_copy* copies [[buffer(2)]],constant ulong2& shape [[buffer(3)]],uint tid [[thread_position_in_grid]])
{
    const ulong at=tid;if(at>=shape.x)return;
    ulong lo=0,hi=shape.y;
    while(lo+1<hi){const ulong middle=(lo+hi)/2;if(copies[middle].ordinal<=at)lo=middle;else hi=middle;}
    const auto copy=copies[lo];const ulong delta=at-copy.ordinal;
    destination[copy.target+delta]=source[copy.source+delta];
}
)metal";
} // namespace metal4
