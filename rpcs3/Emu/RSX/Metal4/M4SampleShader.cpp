#include "M4SampleShader.h"
#include "M4TransferKernels.h"
#include <sstream>
#include <stdexcept>

namespace metal4
{
std::string generate_sample_shader(sample_aspect aspect, bool import_samples, bool single_sample_source, bool half_color)
{
	if (aspect != sample_aspect::color && aspect != sample_aspect::depth && aspect != sample_aspect::depth_stencil && aspect != sample_aspect::stencil)
		throw std::invalid_argument("Invalid native sample transfer aspect");
	const bool depth = aspect == sample_aspect::depth || aspect == sample_aspect::depth_stencil;
	const bool stencil = aspect == sample_aspect::depth_stencil || aspect == sample_aspect::stencil;
	const bool source_is_ms = !import_samples && !single_sample_source;
	if (half_color && aspect != sample_aspect::color)
		throw std::invalid_argument("Half sample transfer requires a color aspect");
	const auto color_type = half_color ? "half" : "float";
	std::ostringstream out;
	out << R"msl(#include <metal_stdlib>
using namespace metal;
struct sample_vertex { float4 position [[position]]; };
vertex sample_vertex rpcs3_sample_vertex(uint index [[vertex_id]]) {
	const float2 points[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
	return {float4(points[index],0,1)};
}
)msl";
	out << nearest_axis_msl << sample_projection_msl << "struct sample_output {\n";
	if (depth)
		out << "float depth [[depth(any)]];\n";
	else if (!stencil)
		out << color_type << "4 color [[color(0)]];\n";
	if (stencil)
		out << "uint stencil [[stencil]];\n";
	out << "};\nfragment sample_output rpcs3_sample_fragment(sample_vertex in [[stage_in]],\n";
	if (import_samples)
		out << "uint sample [[sample_id]],\n";
	out << "constant sample_parameters& p [[buffer(0)]],\n";
	if (depth)
		out << "device atomic_uint* validation [[buffer(1)]],\n";
	if (depth)
		out << (source_is_ms ? "depth2d_ms<float>" : "depth2d<float>");
	else if (stencil)
		out << (source_is_ms ? "texture2d_ms<uint>" : "texture2d<uint,access::read>");
	else
		out << (source_is_ms ? "texture2d_ms<" : "texture2d<") << color_type <<
			(source_is_ms ? ">" : ",access::read>");
	out << " source [[texture(0)]]";
	if (stencil && depth)
		out << ",\n"
			<< (source_is_ms ? "texture2d_ms<uint>" : "texture2d<uint,access::read>") << " stencil_source [[texture(1)]]";
	out << ") {\nconst uint2 pixel = uint2(in.position.xy);\n";
	if (import_samples)
		out << "const uint2 position = pixel * uint2(p.samples_x,p.samples_y) + uint2(sample % p.samples_x,sample / p.samples_x);\n"
			<< "const uint2 origin = uint2(p.origin_x,p.origin_y);\n"
			<< "if (any(position < origin)) { discard_fragment(); return {}; }\n"
			<< "const uint2 coord = position - origin;\n"
			<< "if (any(coord >= uint2(p.width,p.height))) { discard_fragment(); return {}; }\n";
	else
	{
		out << "const auto input = locate_sample(pixel - uint2(p.target_x,p.target_y), p);\n"
			<< "const uint2 coord = input.pixel;\n";
		if (source_is_ms) out << "const uint sample = input.sample;\n";
	}
	out << "sample_output result = {};\n";
	if (depth || !stencil)
		out << "result." << (depth ? "depth" : "color") << " = source.read(coord" << (source_is_ms ? ",sample" : "") << ");\n";
	if (depth)
		out << "if (!isfinite(result.depth) || result.depth < 0 || result.depth > 1) {\n"
			<< "atomic_fetch_or_explicit(validation,1u,memory_order_relaxed); discard_fragment(); return result;\n}\n";
	if (stencil)
		out << "result.stencil = " << (depth ? "stencil_source" : "source") << ".read(coord" << (source_is_ms ? ",sample" : "") << ").x;\n";
	out << "return result;\n}\n";
	return out.str();
}
} // namespace metal4
