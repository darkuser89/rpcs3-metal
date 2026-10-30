#include "M4SampleShader.h"
#include <sstream>
#include <stdexcept>

namespace metal4
{
std::string generate_sample_shader(sample_aspect aspect, bool import_samples)
{
	if (aspect != sample_aspect::color && aspect != sample_aspect::depth && aspect != sample_aspect::depth_stencil && aspect != sample_aspect::stencil)
		throw std::invalid_argument("Invalid native sample transfer aspect");
	const bool depth = aspect == sample_aspect::depth || aspect == sample_aspect::depth_stencil;
	const bool stencil = aspect == sample_aspect::depth_stencil || aspect == sample_aspect::stencil;
	std::ostringstream out;
	out << R"msl(#include <metal_stdlib>
using namespace metal;
struct sample_vertex { float4 position [[position]]; };
vertex sample_vertex rpcs3_sample_vertex(uint index [[vertex_id]]) {
	const float2 points[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
	return {float4(points[index],0,1)};
}
struct sample_parameters { uint samples_x, samples_y; };
struct sample_output {
)msl";
	if (depth)
		out << "float depth [[depth(any)]];\n";
	else if (!stencil)
		out << "float4 color [[color(0)]];\n";
	if (stencil)
		out << "uint stencil [[stencil]];\n";
	out << "};\nfragment sample_output rpcs3_sample_fragment(sample_vertex in [[stage_in]],\n";
	if (import_samples)
		out << "uint sample [[sample_id]],\n";
	out << "constant sample_parameters& p [[buffer(0)]],\n";
	if (depth)
		out << "device atomic_uint* validation [[buffer(1)]],\n";
	if (depth)
		out << (import_samples ? "depth2d<float>" : "depth2d_ms<float>");
	else if (stencil)
		out << (import_samples ? "texture2d<uint,access::read>" : "texture2d_ms<uint>");
	else
		out << (import_samples ? "texture2d<float,access::read>" : "texture2d_ms<float>");
	out << " source [[texture(0)]]";
	if (stencil && depth)
		out << ",\n"
			<< (import_samples ? "texture2d<uint,access::read>" : "texture2d_ms<uint>") << " stencil_source [[texture(1)]]";
	out << ") {\nconst uint2 pixel = uint2(in.position.xy);\n";
	if (import_samples)
		out << "const uint2 coord = pixel * uint2(p.samples_x,p.samples_y) + uint2(sample % p.samples_x,sample / p.samples_x);\n";
	else
		out << "const uint2 coord = pixel / uint2(p.samples_x,p.samples_y);\n"
			<< "const uint sample = pixel.x % p.samples_x + (pixel.y % p.samples_y) * p.samples_x;\n";
	out << "sample_output result = {};\n";
	if (depth || !stencil)
		out << "result." << (depth ? "depth" : "color") << " = source.read(coord" << (import_samples ? "" : ",sample") << ");\n";
	if (depth)
		out << "if (!isfinite(result.depth) || result.depth < 0 || result.depth > 1) {\n"
			<< "atomic_fetch_or_explicit(validation,1u,memory_order_relaxed); discard_fragment(); return result;\n}\n";
	if (stencil)
		out << "result.stencil = " << (depth ? "stencil_source" : "source") << ".read(coord" << (import_samples ? "" : ",sample") << ").x;\n";
	out << "return result;\n}\n";
	return out.str();
}
} // namespace metal4
