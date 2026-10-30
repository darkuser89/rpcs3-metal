#pragma once

#include <ostream>

namespace metal4
{

// Vertex and fragment stages must agree on the flat-color option and all
// user locations. Vertex-only builtins don't need fragment input members.
inline void emit_varyings(std::ostream& out, bool flat_colors, bool fragment, bool point_output = false)
{
	out << (fragment ? "struct rsx_fragment_inputs {\n" : "struct rsx_varyings {\n");
	out << "    float4 position [[position" << (fragment ? "" : ", invariant") << "]];\n";
	if (!fragment && point_output) out << "    float point_size [[point_size]];\n";
	for (unsigned i = 0; i < 4; ++i)
	{
		out << "    float4 color" << i << " [[user(locn" << i << ")" << (flat_colors ? ", flat" : "") << "]];\n";
	}
	out << "    float4 fog [[user(locn4)]];\n";
	for (unsigned i = 0; i < 10; ++i) out << "    float4 tex" << i << " [[user(locn" << i + 5 << ")]];\n";
	if (!fragment) out << "    float clip [[clip_distance]] [6];\n";
	out << "};\n";
}

}
