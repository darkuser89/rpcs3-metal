#pragma once

#include <string_view>
namespace metal4
{
// Build-time insertion uses only the vendored AMD FP32 EASU/RCAS functions,
// adapted to Metal references and explicit texture/constant arguments.
inline constexpr std::string_view fsr_kernels = R"metal(
// AMD_FSR_LICENSE
#include <metal_stdlib>
using namespace metal;
using AF1=float; using AF2=float2; using AF3=float3; using AF4=float4;
using AU2=uint2; using AU4=uint4; using ASU2=int2; using AP1=bool;
#define AF1_(x) float(x)
#define AF2_(x) float2(x)
#define AF3_(x) float3(x)
#define AF4_(x) float4(x)
#define AF2_AU2(x) as_type<float2>(x)
#define AF1_AU1(x) as_type<float>(x)
#define FSR_RCAS_LIMIT (0.25-(1.0/16.0))
struct fsr_params {
    uint4 easu[4],rcas;
    uint2 origin,extent;
    uint level,width,height,padding;
};
float APrxLoRcpF1(float a) { return as_type<float>(0x7ef07ebbu-as_type<uint>(a)); }
float APrxLoRsqF1(float a) { return as_type<float>(0x5f347d74u-(as_type<uint>(a)>>1)); }
float APrxMedRcpF1(float a) { const float b=as_type<float>(0x7ef19fffu-as_type<uint>(a));return b*(-b*a+2.0f); }
float ARcpF1(float a) { return 1.0f/a; }
float ASatF1(float a) { return clamp(a,0.0f,1.0f); }
float AMin3F1(float a,float b,float c) { return min(a,min(b,c)); }
float AMax3F1(float a,float b,float c) { return max(a,max(b,c)); }
float3 AMin3F3(float3 a,float3 b,float3 c) { return min(a,min(b,c)); }
float3 AMax3F3(float3 a,float3 b,float3 c) { return max(a,max(b,c)); }
float4 fsr_gather(float2 p,uint channel,texture2d<float> source,constant fsr_params& params) {
    const int2 lower=int2(params.origin),upper=lower+int2(params.extent)-1;
    const int2 at=int2(floor(p*float2(source.get_width(params.level),source.get_height(params.level))-0.5f));
    // Match AMD gather ordering while clamping to the selected crop, not the
    // complete resource. Texture read keeps native format/sRGB interpretation.
    return float4(source.read(uint2(clamp(at+int2(0,1),lower,upper)),params.level)[channel],
        source.read(uint2(clamp(at+int2(1,1),lower,upper)),params.level)[channel],
        source.read(uint2(clamp(at+int2(1,0),lower,upper)),params.level)[channel],
        source.read(uint2(clamp(at,lower,upper)),params.level)[channel]);
}
float4 FsrEasuRF(float2 p,texture2d<float> source,constant fsr_params& params) { return fsr_gather(p,0,source,params); }
float4 FsrEasuGF(float2 p,texture2d<float> source,constant fsr_params& params) { return fsr_gather(p,1,source,params); }
float4 FsrEasuBF(float2 p,texture2d<float> source,constant fsr_params& params) { return fsr_gather(p,2,source,params); }
float4 FsrRcasLoadF(int2 p,texture2d<float> source,constant fsr_params& params) {
    return source.read(uint2(clamp(p,int2(0),int2(params.width,params.height)-1)));
}
void FsrRcasInputF(thread float&,thread float&,thread float&) {}
// AMD_FSR_FUNCTIONS
kernel void m4_fsr_easu(texture2d<float> source [[texture(0)]],texture2d<float,access::write> output [[texture(1)]],
    constant fsr_params& params [[buffer(0)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=params.width || pixel.y>=params.height) return;
    float3 color;
    FsrEasuF(color,pixel,params.easu[0],params.easu[1],params.easu[2],params.easu[3],source,params);
    output.write(float4(color,1.0f),pixel);
}
kernel void m4_fsr_rcas(texture2d<float> source [[texture(0)]],texture2d<float,access::write> output [[texture(1)]],
    constant fsr_params& params [[buffer(0)]],uint2 pixel [[thread_position_in_grid]]) {
    if(pixel.x>=params.width || pixel.y>=params.height) return;
    float r,g,b;
    FsrRcasF(r,g,b,pixel,params.rcas,source,params);
    output.write(float4(r,g,b,1.0f),pixel);
}
)metal";
} // namespace metal4
