#pragma once
#include "M4FSRPlan.h"
#include <algorithm>
#include <array>
#include <bit>
#include <vector>

namespace metal4::test_fsr
{
using AF1=float;
using AF2=float __attribute__((ext_vector_type(2)));
using AF3=float __attribute__((ext_vector_type(3)));
using AF4=float __attribute__((ext_vector_type(4)));
using AU2=std::uint32_t __attribute__((ext_vector_type(2)));
using AU4=std::uint32_t __attribute__((ext_vector_type(4)));
using ASU2=std::int32_t __attribute__((ext_vector_type(2)));
using AP1=bool;
inline AF2 make_f2(float x,float y) { return {x,y}; }
inline AF2 make_f2(AU2 value) { return __builtin_convertvector(value,AF2); }
inline AF3 make_f3(float x,float y,float z) { return {x,y,z}; }
inline ASU2 make_i2(int x,int y) { return {x,y}; }
inline ASU2 make_i2(AU2 value) { return __builtin_convertvector(value,ASU2); }
inline float min(float a,float b) { return std::fmin(a,b); }
inline float max(float a,float b) { return std::fmax(a,b); }
inline AF3 min(AF3 a,AF3 b) { return {min(a.x,b.x),min(a.y,b.y),min(a.z,b.z)}; }
inline AF3 max(AF3 a,AF3 b) { return {max(a.x,b.x),max(a.y,b.y),max(a.z,b.z)}; }
inline float abs(float x) { return std::abs(x); }
inline AF2 floor(AF2 x) { return {std::floor(x.x),std::floor(x.y)}; }
inline float APrxLoRcpF1(float a) { return std::bit_cast<float>(0x7ef07ebbu-std::bit_cast<std::uint32_t>(a)); }
inline float APrxLoRsqF1(float a) { return std::bit_cast<float>(0x5f347d74u-(std::bit_cast<std::uint32_t>(a)>>1)); }
inline float APrxMedRcpF1(float a) { const auto b=std::bit_cast<float>(0x7ef19fffu-std::bit_cast<std::uint32_t>(a));return b*(-b*a+2.f); }
inline float ARcpF1(float a) { return 1.f/a; }
inline float ASatF1(float a) { return min(1.f,max(0.f,a)); }
inline float AMin3F1(float a,float b,float c) { return min(a,min(b,c)); }
inline float AMax3F1(float a,float b,float c) { return max(a,max(b,c)); }
inline AF3 AMin3F3(AF3 a,AF3 b,AF3 c) { return min(a,min(b,c)); }
inline AF3 AMax3F3(AF3 a,AF3 b,AF3 c) { return max(a,max(b,c)); }
using pixel=std::array<float,4>;
struct source_view
{
	const std::vector<pixel>* pixels=nullptr;
	unsigned width=0,height=0;
	frame_rectangle crop;
};
inline thread_local source_view easu_source,rcas_source;
inline AF4 load(const source_view& source,int x,int y)
{
	x=std::clamp(x,int(source.crop.x),int(source.crop.x+source.crop.width-1));
	y=std::clamp(y,int(source.crop.y),int(source.crop.y+source.crop.height-1));
	const auto& p=source.pixels->at(y*source.width+x); return {p[0],p[1],p[2],p[3]};
}
inline AF4 gather(AF2 p,unsigned c)
{
	const int x=std::floor(p.x*easu_source.width-.5f),y=std::floor(p.y*easu_source.height-.5f);
	return {load(easu_source,x,y+1)[c],load(easu_source,x+1,y+1)[c],load(easu_source,x+1,y)[c],load(easu_source,x,y)[c]};
}
inline AF4 FsrEasuRF(AF2 p) { return gather(p,0); }
inline AF4 FsrEasuGF(AF2 p) { return gather(p,1); }
inline AF4 FsrEasuBF(AF2 p) { return gather(p,2); }
inline AF4 FsrRcasLoadF(ASU2 p) { return load(rcas_source,p.x,p.y); }
inline void FsrRcasInputF(AF1&,AF1&,AF1&) {}
#define AF2(...) make_f2(__VA_ARGS__)
#define AF3(...) make_f3(__VA_ARGS__)
#define ASU2(...) make_i2(__VA_ARGS__)
#define AF1_(x) float(x)
#define AF2_(x) AF2{x,x}
#define AF3_(x) AF3{x,x,x}
#define AF4_(x) AF4{x,x,x,x}
#define AF2_AU2(x) std::bit_cast<AF2>(x)
#define AF1_AU1(x) std::bit_cast<float>(x)
#define FSR_RCAS_LIMIT (0.25-(1.0/16.0))
#include "M4FSRReference.inc"
#undef AF2
#undef AF3
#undef ASU2
#undef AF1_
#undef AF2_
#undef AF3_
#undef AF4_
#undef AF2_AU2
#undef AF1_AU1
#undef FSR_RCAS_LIMIT
std::array<std::array<std::uint32_t,4>,4> amd_constants(frame_extent input,frame_rectangle crop,frame_extent output);
float amd_sharpness(float sharpness);
inline std::vector<pixel> reference(const std::vector<pixel>& source,frame_extent input,frame_rectangle crop,frame_extent output,float sharpness)
{
	const auto constants=amd_constants(input,crop,output);
	const auto as_u4=[](const std::array<std::uint32_t,4>& c) { return AU4{c[0],c[1],c[2],c[3]}; };
	easu_source={&source,input.width,input.height,crop};
	std::vector<pixel> intermediate(output.width*output.height),result(intermediate.size());
	for(unsigned y=0;y<output.height;++y) for(unsigned x=0;x<output.width;++x)
	{
		AF3 value;
		FsrEasuF(value,AU2{x,y},as_u4(constants[0]),as_u4(constants[1]),as_u4(constants[2]),as_u4(constants[3]));
		for(unsigned c=0;c<3;++c) intermediate[y*output.width+x][c]=float(_Float16(value[c]));
		intermediate[y*output.width+x][3]=1;
	}
	rcas_source={&intermediate,output.width,output.height,{0,0,output.width,output.height}};
	const AU4 rcas{std::bit_cast<std::uint32_t>(amd_sharpness(sharpness)),0,0,0};
	for(unsigned y=0;y<output.height;++y) for(unsigned x=0;x<output.width;++x)
	{
		float r,g,b; FsrRcasF(r,g,b,AU2{x,y},rcas);
		result[y*output.width+x]={float(_Float16(r)),float(_Float16(g)),float(_Float16(b)),1};
	}
	easu_source={};rcas_source={}; return result;
}
} // namespace metal4::test_fsr
