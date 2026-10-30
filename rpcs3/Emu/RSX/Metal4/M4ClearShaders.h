#pragma once
#include <string_view>

namespace metal4
{
inline constexpr std::string_view clear_shaders = R"msl(
#include <metal_stdlib>
using namespace metal;
struct clear_vertex { float4 position [[position]]; };
vertex clear_vertex rpcs3_clear_vertex(uint id [[vertex_id]]) {
	const float2 position[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
	return {float4(position[id],0,1)};
}
struct clear_parameters { float4 colors[4]; float depth; uint reserved[3]; };
struct clear_color_output {float4 a [[color(0)]],b [[color(1)]],c [[color(2)]],d [[color(3)]];};
struct clear_depth_output {float4 a [[color(0)]],b [[color(1)]],c [[color(2)]],d [[color(3)]];float depth [[depth(any)]];};
fragment clear_color_output rpcs3_clear_color(constant clear_parameters& p [[buffer(0)]]) {
	return {p.colors[0],p.colors[1],p.colors[2],p.colors[3]};
}
fragment clear_depth_output rpcs3_clear_depth(constant clear_parameters& p [[buffer(0)]]) {
	return {p.colors[0],p.colors[1],p.colors[2],p.colors[3],p.depth};
}
)msl";
}
