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
float4 read_pixel(device const uchar* data,int2 p,constant layout& l)
{
    p=clamp(p,int2(0),int2(l.width-1,l.height-1));
    ulong offset=rpcs3_pixel_offset(uint2(p),l);
    if(offset==~0ul) return l.bytes==2?float4(0,0,0,1):float4(0);
    if(l.bytes==4) return float4(data[offset+1],data[offset+2],data[offset+3],data[offset])/255.0f;
    uint word=(uint(data[offset])<<8)|uint(data[offset+1]);
    return float4(float((word>>11)&31)/31.0f,float((word>>5)&63)/63.0f,float(word&31)/31.0f,1.0f);
}
kernel void rpcs3_scaled_image(device const uchar* source [[buffer(0)]],device uchar* destination [[buffer(1)]],
    constant parameters& p [[buffer(2)]],uint2 pixel [[thread_position_in_grid]])
{
    if(pixel.x>=p.destination.width||pixel.y>=p.destination.height) return;
    uint2 output=pixel+uint2(p.clip_x,p.clip_y);
    float4 color;
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
        color=read_pixel(source,selected,p.source);
    } else {
        float2 coord=(float2(output)+0.5f)*float2(p.input_w,p.input_h)/float2(p.scaled_w,p.scaled_h);
        if(p.flip_x) coord.x=float(p.input_w)-coord.x;
        if(p.flip_y) coord.y=float(p.input_h)-coord.y;
        coord-=0.5f;
        int2 a=int2(floor(coord));float2 weight=coord-float2(a);
        color=mix(mix(read_pixel(source,a,p.source),read_pixel(source,a+int2(1,0),p.source),weight.x),
                  mix(read_pixel(source,a+int2(0,1),p.source),read_pixel(source,a+int2(1,1),p.source),weight.x),weight.y);
    }
    ulong to=rpcs3_pixel_offset(pixel,p.destination);
    if(to==~0ul) return;
    color=clamp(color,0.0f,1.0f);
    if(p.destination.bytes==4) {
        uint4 v=uint4(round(color*255.0f));
        destination[to]=uchar(v.w);destination[to+1]=uchar(v.x);destination[to+2]=uchar(v.y);destination[to+3]=uchar(v.z);
    } else {
        uint3 v=uint3(round(color.xyz*float3(31,63,31)));uint word=(v.x<<11)|(v.y<<5)|v.z;
        destination[to]=uchar(word>>8);destination[to+1]=uchar(word);
    }
}
)metal";
} // namespace metal4
