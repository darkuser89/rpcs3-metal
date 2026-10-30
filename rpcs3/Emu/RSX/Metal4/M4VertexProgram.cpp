#include "M4VertexProgram.h"
#include "M4VertexRuntime.h"
#include "M4VertexFetchKernels.h"
#include "M4Varyings.h"
#include "M4SamplerParameters.h"

#include <sstream>
#include <stdexcept>

namespace metal4
{

namespace
{

std::uint32_t bits(std::uint32_t word, unsigned offset, unsigned width)
{
	return (word >> offset) & ((1u << width) - 1);
}

std::string number(std::uint32_t value)
{
	return std::to_string(value);
}

std::string swizzle(const std::array<std::uint32_t, 4>& components)
{
	std::string result = ".";
	for (auto c : components)
		result += "xyzw"[c];
	return result;
}

class emitter
{
	const vertex_program_options& m_options;
	generated_vertex_program& m_program;
	const vertex_instruction& m_instruction;

  public:
	emitter(const vertex_program_options& options, generated_vertex_program& program, const vertex_instruction& instruction)
		: m_options(options), m_program(program), m_instruction(instruction) {}

	std::string source(std::size_t index)
	{
		const auto& s = m_instruction.sources.at(index);
		const auto d0 = m_instruction.words[0];
		const auto address = "r.address[" + number(bits(d0, 24, 1)) + "][" + number(bits(d0, 0, 2)) + "]";
		std::string value;
		switch (s.type)
		{
		case 1:
			value = "r.temp[" + number(s.temporary) + "]";
			break;
		case 2:
			// A runtime address register may select any of the sixteen inputs.
			// Fetch planning must not discard descriptors outside the base input.
			m_program.attribute_mask |= bits(d0, 27, 1) ? 65535u : 1u << m_instruction.input;
			value = "rsx_attribute(attributes, ctx, source_vertex, long(" + number(m_instruction.input) + ")";
			if (bits(d0, 27, 1))
				value += " + long(" + address + ")";
			value += ", r.fault)";
			break;
		case 3:
			value = "rsx_constant(constants, ctx, long(" + number(m_instruction.constant) + ")";
			if (bits(m_instruction.words[3], 1, 1))
			{
				m_program.indexed_constants = true;
				value += " + long(" + address + ")";
			}
			value += ", r.fault)";
			break;
		default:
			throw std::invalid_argument("RSX vertex instruction reads an invalid source register type");
		}
		value = "(" + value + ")" + swizzle(s.swizzle);
		if (s.absolute)
			value = "abs(" + value + ")";
		if (s.negate)
			value = "-(" + value + ")";
		return value;
	}

	std::string vector_value()
	{
		switch (m_instruction.vector_opcode)
		{
		case 1:
			return source(0);
		case 2:
			return "(" + source(0) + " * " + source(1) + ")";
		case 3:
			return "(" + source(0) + " + " + source(2) + ")";
		case 4:
			return "fma(" + source(0) + ", " + source(1) + ", " + source(2) + ")";
		case 5:
			return "float4(dot((" + source(0) + ").xyz, (" + source(1) + ").xyz))";
		case 6:
			return "float4(dot(float4((" + source(0) + ").xyz, 1), " + source(1) + "))";
		case 7:
			return "float4(dot(" + source(0) + ", " + source(1) + "))";
		case 8:
			return "float4(1, (" + source(0) + ").y * (" + source(1) + ").y, (" + source(0) + ").z, (" + source(1) + ").w)";
		case 9:
			return "min(" + source(0) + ", " + source(1) + ")";
		case 10:
			return "max(" + source(0) + ", " + source(1) + ")";
		case 11:
			return "float4(" + source(0) + " < " + source(1) + ")";
		case 12:
			return "float4(" + source(0) + " >= " + source(1) + ")";
		case 14:
			return "fract(" + source(0) + ")";
		case 15:
			return "floor(" + source(0) + ")";
		case 16:
			return "float4(" + source(0) + " == " + source(1) + ")";
		case 17:
			return "float4(0)";
		case 18:
			return "float4(" + source(0) + " > " + source(1) + ")";
		case 19:
			return "float4(" + source(0) + " <= " + source(1) + ")";
		case 20:
			return "float4(" + source(0) + " != " + source(1) + ")";
		case 21:
			return "float4(1)";
		case 22:
			return "sign(" + source(0) + ")";
		case 25:
		{
			const auto id = m_instruction.texture;
			m_program.texture_mask |= 1u << id;
			const auto dimension = m_options.textures[id];
			// Metal texture1d doesn't expose explicit LOD. RSX 1D mip chains
			// use a height-one texture2d, sampled at the row center instead.
			const auto coords = dimension == vertex_texture_dimension::one ? "float2((" + source(0) + ").x, 0.5f)" : "(" + source(0) + ")" + (dimension == vertex_texture_dimension::two ? ".xy" : ".xyz");
			const auto sample = "vtex" + number(id) + ".sample(vsampler" + number(id) + ", " + coords +
				   ", level(clamp((" + source(0) + ").w + sampler_state[" + number(id) + "].bias, sampler_state[" + number(id) + "].min_lod, sampler_state[" + number(id) + "].max_lod)))";
			if (m_options.border_textures & (1u << id))
			{
				auto coverage = sample;
				coverage.replace(0, ("vtex" + number(id)).size(), "vborder" + number(id));
				return "rsx_border_color(" + sample + ", (" + coverage + ").r, sampler_state[" + number(id) + "])";
			}
			return sample;
		}
		default:
			throw std::invalid_argument("Unsupported RSX vertex vector opcode " + number(m_instruction.vector_opcode));
		}
	}

	std::string scalar_value()
	{
		if (m_instruction.scalar_opcode == 7)
			return "rsx_lit(" + source(2) + ")";
		if (m_instruction.scalar_opcode == 1)
			return source(2);
		// Scalar ALU operands normally encode a replicate swizzle. The VK
		// decompiler and interpreter disagree for non-replicate operands; don't
		// silently select one interpretation until that case is established.
		const auto& components = m_instruction.sources[2].swizzle;
		for (auto c : components)
		{
			if (c != components[0])
				throw std::invalid_argument("RSX scalar ALU operand has a non-replicate swizzle");
		}
		const auto s = "(" + source(2) + ").x";
		std::string result;
		switch (m_instruction.scalar_opcode)
		{
		case 2:
			result = "1.0f / " + s;
			break;
		case 3:
			result = "clamp(1.0f / " + s + ", 5.42101e-20f, 1.884467e19f)";
			break;
		case 4:
			result = "1.0f / sqrt(max(" + s + ", 1e-10f))";
			break;
		case 5:
			result = "exp(" + s + ")";
			break;
		case 6:
			result = "log(" + s + ")";
			break;
		case 13:
			result = "log2(max(" + s + ", 1e-10f))";
			break;
		case 14:
			result = "exp2(" + s + ")";
			break;
		case 15:
			result = "sin(" + s + ")";
			break;
		case 16:
			result = "cos(" + s + ")";
			break;
		default:
			throw std::invalid_argument("Unsupported RSX vertex scalar opcode " + number(m_instruction.scalar_opcode));
		}
		return "float4(" + result + ")";
	}

	void write(std::ostream& out, bool scalar, std::string value)
	{
		const auto d0 = m_instruction.words[0], d3 = m_instruction.words[3];
		const auto temporary = scalar ? bits(d3, 7, 6) : bits(d0, 15, 6);
		const bool output = scalar ? !bits(d0, 30, 1) : bits(d0, 30, 1);
		if (bits(d0, 26, 1))
			value = "clamp(" + value + ", 0.0f, 1.0f)";
		out << "            { const float4 value = " << value << ";\n";
		std::array<std::string, 2> destinations;
		std::size_t count = 0;
		if (temporary != 63)
			destinations[count++] = "r.temp[" + number(temporary) + "]";
		if (output && bits(d3, 2, 5) < 16 && (!scalar || temporary == 63))
		{
			destinations[count++] = "r.dst[" + number(bits(d3, 2, 5)) + "]";
			m_program.output_register_mask |= 1u << bits(d3, 2, 5);
		}
		if (count == 0 && !output)
			destinations[count++] = "r.condition[" + number(bits(d0, 25, 1)) + "]";
		for (std::size_t i = 0; i < count; ++i)
		{
			for (unsigned c = 0; c < 4; ++c)
			{
				if (!bits(d3, (scalar ? 20 : 16) - c, 1))
					continue;
				out << "                if (write_mask[" << c << "]) " << destinations[i] << "." << "xyzw"[c]
					<< " = value." << "xyzw"[c] << ";\n";
			}
		}
		out << "                (void)value;\n            }\n";
	}
};

std::string texture_type(vertex_texture_dimension dimension)
{
	switch (dimension)
	{
	case vertex_texture_dimension::one:
		return "texture2d<float>";
	case vertex_texture_dimension::two:
		return "texture2d<float>";
	case vertex_texture_dimension::three:
		return "texture3d<float>";
	case vertex_texture_dimension::cube:
		return "texturecube<float>";
	}
	throw std::invalid_argument("Unknown Metal 4 vertex texture dimension");
}

} // namespace

vertex_instruction decode_vertex_instruction(std::array<std::uint32_t, 4> words)
{
	vertex_instruction result{};
	result.words = words;
	const std::array packed{(bits(words[1], 0, 8) << 9) | bits(words[2], 23, 9),
							bits(words[2], 6, 17), (bits(words[2], 0, 6) << 11) | bits(words[3], 21, 11)};
	for (unsigned i = 0; i < 3; ++i)
	{
		result.sources[i] = {bits(packed[i], 0, 2), bits(packed[i], 2, 6), {bits(packed[i], 14, 2), bits(packed[i], 12, 2), bits(packed[i], 10, 2), bits(packed[i], 8, 2)}, bits(packed[i], 16, 1) != 0, bits(words[0], 21 + i, 1) != 0};
	}
	result.vector_opcode = bits(words[1], 22, 5);
	result.scalar_opcode = bits(words[1], 27, 5);
	result.input = bits(words[1], 8, 4);
	result.constant = bits(words[1], 12, 10);
	result.texture = bits(words[2], 8, 2);
	result.branch_target = (bits(words[0], 23, 1) << 9) | (bits(words[2], 0, 6) << 3) | bits(words[3], 29, 3);
	return result;
}

generated_vertex_program generate_vertex_program(std::span<const std::array<std::uint32_t, 4>> instructions,
												 const vertex_program_options& options)
{
	if (options.flat_result_inputs && !options.flat_colors)
		throw std::invalid_argument("Native flat result inputs require flat color interpolation");
	if ((options.border_textures & ~15u) || instructions.empty() || instructions.size() > 1024 || options.entry >= instructions.size() ||
		(!options.live_instructions.empty() && options.live_instructions.size() != instructions.size()))
	{
		throw std::invalid_argument("RSX vertex program length, entry or analyser mask is invalid");
	}
	if (!options.live_instructions.empty() && !options.live_instructions[options.entry])
	{
		throw std::invalid_argument("RSX vertex entry is marked dead");
	}
	for (auto dimension : options.textures)
		(void)texture_type(dimension);
	generated_vertex_program program;
	std::ostringstream body;
	for (std::size_t i = 0; i < instructions.size(); ++i)
	{
		if (!options.live_instructions.empty() && !options.live_instructions[i])
			continue;
		const auto instruction = decode_vertex_instruction(instructions[i]);
		const auto d0 = instruction.words[0], d3 = instruction.words[3];
		if (bits(d0, 14, 1) || bits(d0, 29, 1))
		{
			throw std::invalid_argument("RSX vertex condition-update flags require further hardware validation");
		}
		if (bits(d3, 2, 5) >= 16 && bits(d3, 2, 5) != 31)
		{
			throw std::invalid_argument("RSX vertex output register exceeds the register file");
		}
		const auto scalar = instruction.scalar_opcode;
		if (((scalar >= 1 && scalar <= 7) || (scalar >= 13 && scalar <= 16)) &&
			!bits(d0, 30, 1) && bits(d3, 2, 5) < 16 && bits(d3, 7, 6) != 63)
		{
			throw std::invalid_argument("RSX scalar instruction cannot write both temporary and output registers");
		}
		if (scalar == 11 || scalar >= 19)
		{
			throw std::invalid_argument("Unsupported RSX vertex flow opcode " + number(scalar));
		}
		const bool immediate_branch = scalar == 9 || scalar == 10 || scalar == 17 || scalar == 18;
		if (immediate_branch && (instruction.branch_target >= instructions.size() ||
								 (!options.live_instructions.empty() && !options.live_instructions[instruction.branch_target])))
		{
			throw std::invalid_argument("RSX immediate vertex branch target is outside live instructions");
		}
		emitter emit(options, program, instruction);
		body << "        case " << i << ": {\n";
		const std::array condition_swizzle{bits(d0, 8, 2), bits(d0, 6, 2), bits(d0, 4, 2), bits(d0, 2, 2)};
		body << "            const bool4 tested = rsx_condition(r.condition[" << bits(d0, 25, 1) << "]"
			 << swizzle(condition_swizzle) << ", " << bits(d0, 10, 3) << "u);\n";
		body << "            const bool4 write_mask = " << (bits(d0, 13, 1) ? "tested" : "bool4(true)") << ";\n";
		body << "            (void)tested; (void)write_mask;\n";
		body << "            pc = " << i + 1 << ";\n";
		body << "            if (" << (bits(d0, 13, 1) && !bits(d0, 10, 3) ? "false" : "true") << ") {\n";
		if (instruction.vector_opcode == 13)
		{
			body << "            { const int4 value = rsx_address(" << emit.source(0) << ", r.fault);\n";
			for (unsigned c = 0; c < 4; ++c)
			{
				if (bits(d3, 16 - c, 1))
					body << "                if (write_mask[" << c << "]) r.address["
						 << (bits(d0, 15, 6) & 1) << "][" << c << "] = value[" << c << "];\n";
			}
			body << "                (void)value;\n            }\n";
		}
		else if (instruction.vector_opcode)
			emit.write(body, false, emit.vector_value());
		// The current VK interpreter evaluates SCA after VEC. Dual-issue source
		// timing is retained here and remains part of the hardware parity audit.
		if ((scalar >= 1 && scalar <= 7) || (scalar >= 13 && scalar <= 16))
		{
			emit.write(body, true, emit.scalar_value());
		}
		else if (scalar)
		{
			const auto boolean_condition = "(((ctx.branch_bits >> " + number(bits(d3, 23, 5)) + ") & 1u) == " + number(bits(d3, 28, 1)) + "u)";
			body << "            if (" << (scalar == 17 || scalar == 18 ? boolean_condition : "any(tested)") << ") {\n";
			if (scalar == 8)
				body << "                pc = long(r.address[" << bits(d0, 24, 1) << "].x) - " << options.base_address << "l;\n";
			else if (scalar == 9 || scalar == 17)
				body << "                pc = " << instruction.branch_target << ";\n";
			else if (scalar == 10 || scalar == 18)
			{
				body << "                if (stack_pointer == 8) { r.fault |= 8u; return r; }\n"
					 << "                callstack[stack_pointer++] = pc; pc = " << instruction.branch_target << ";\n";
			}
			else if (scalar == 12)
				body << "                if (!stack_pointer) return r; pc = callstack[--stack_pointer];\n";
			body << "            }\n";
		}
		body << "            }\n";
		if (bits(d3, 0, 1))
			body << "            return r;\n";
		else
			body << "            break;\n";
		body << "        }\n";
	}
	std::ostringstream out;
	out << vertex_runtime;
	if (options.packed_inputs)
	{
		out << vertex_fetch_runtime;
		out << R"metal(
struct rsx_packed_vertex_inputs {
    device const uchar* persistent;
    device const uchar* transient;
    constant rsx_fetch_parameters* parameters;
};
inline float4 rsx_attribute(thread const rsx_packed_vertex_inputs& attributes,
    constant rsx_vertex_context& ctx, uint source_vertex, long slot, thread uint& fault) {
    if (source_vertex < ctx.vertex_base || source_vertex - ctx.vertex_base >= ctx.vertex_count || slot < 0 || slot >= 16) {
        fault |= 1u;
        return float4(0,0,0,1);
    }
    return rsx_fetch_attribute(attributes.persistent, attributes.transient, *attributes.parameters, source_vertex, uint(slot));
}
)metal";
	}
	out << sampler_parameters_msl;
	emit_varyings(out, options.flat_colors, false, options.point_output);
	std::string texture_arguments, texture_values, texture_entry_arguments;
	for (unsigned i = 0; i < 4; ++i)
	{
		if (!(program.texture_mask & (1u << i)))
			continue;
		const auto n = std::to_string(i);
		texture_arguments += ", " + texture_type(options.textures[i]) + " vtex" + n + ", sampler vsampler" + n;
		texture_values += ", vtex" + n + ", vsampler" + n;
		texture_entry_arguments += ", " + texture_type(options.textures[i]) + " vtex" + n + " [[texture(" + n + ")]], sampler vsampler" + n + " [[sampler(" + n + ")]]";
		if (options.border_textures & (1u << i))
		{
			texture_arguments += ", " + texture_type(options.textures[i]) + " vborder" + n;
			texture_values += ", vborder" + n;
			texture_entry_arguments += ", " + texture_type(options.textures[i]) + " vborder" + n + " [[texture(" + number(4 + i) + ")]]";
		}
	}
	out << "inline rsx_vertex_registers rsx_evaluate(" << (options.packed_inputs ? "thread const rsx_packed_vertex_inputs&" : "device const float4*") << " attributes, device const float4* constants,\n"
		<< "    constant rsx_vertex_context& ctx, uint source_vertex, constant rsx_sampler_parameters* sampler_state" << texture_arguments << ")\n{\n"
		<< "    rsx_vertex_registers r = rsx_initialize();\n    long pc = " << options.entry << ";\n"
		<< "    long callstack[8]; uint stack_pointer = 0;\n"
		<< "    (void)attributes; (void)constants; (void)source_vertex; (void)callstack; (void)stack_pointer; (void)sampler_state;\n"
		<< "    for (uint step = 0; step < ctx.instruction_budget; ++step) {\n        switch (pc) {\n"
		<< body.str() << "        default: r.fault |= 4u; return r;\n        }\n    }\n    r.fault |= 16u; return r;\n}\n";
	if (options.flat_result_inputs)
	{
		out << R"metal(
struct rsx_flat_color_result { uint4 colors[4]; uint fault; uint pad0, pad1, pad2; };
kernel void rpcs3_rsx_flat_colors(uint id [[thread_position_in_grid]],
)metal";
		if (options.packed_inputs)
			out << "    device const uchar* persistent [[buffer(0)]], device const uchar* transient [[buffer(4)]],\n"
				<< "    constant rsx_fetch_parameters& fetch [[buffer(5)]], device const float4* constants [[buffer(1)]],\n";
		else out << "    device const float4* attributes [[buffer(0)]], device const float4* constants [[buffer(1)]],\n";
		out << "    constant rsx_vertex_context& ctx [[buffer(2)]], constant rsx_sampler_parameters* sampler_state [[buffer(3)]],\n"
			<< "    device const uint* flat_ids [[buffer(6)]], device rsx_flat_color_result* flat_results [[buffer(7)]]" << texture_entry_arguments << ")\n{\n"
			<< "    if (id >= ctx.flat_count || !ctx.predicate) return;\n";
		if (options.packed_inputs) out << "    const rsx_packed_vertex_inputs attributes{persistent, transient, &fetch};\n";
		out << "    const auto r = rsx_evaluate(attributes, constants, ctx, flat_ids[id], sampler_state" << texture_values << ");\n"
			<< "    rsx_flat_color_result result;\n    for (uint i=0; i<4; ++i) result.colors[i]=as_type<uint4>(r.dst[i+1]);\n"
			<< "    result.fault=r.fault; result.pad0=0; result.pad1=0; result.pad2=0; flat_results[id]=result;\n}\n";
	}
	out << "vertex rsx_varyings rpcs3_rsx_vertex(uint vertex_id [[vertex_id]],\n";
	if (options.packed_inputs)
		out << "    device const uchar* persistent [[buffer(0)]], device const uchar* transient [[buffer(4)]],\n"
			<< "    constant rsx_fetch_parameters& fetch [[buffer(5)]], device const float4* constants [[buffer(1)]],\n";
	else
		out << "    device const float4* attributes [[buffer(0)]], device const float4* constants [[buffer(1)]],\n";
	if (options.flat_result_inputs)
		out << "    device const uint* flat_ids [[buffer(6)]], device const rsx_flat_color_result* flat_results [[buffer(7)]],\n";
	out
		<< "    constant rsx_vertex_context& ctx [[buffer(2)]],\n"
		<< "    constant rsx_sampler_parameters* sampler_state [[buffer(3)]],\n"
		<< "    device const rsx_vertex_reference* references [[buffer(30)]]" << texture_entry_arguments << ")\n{\n"
		<< "    rsx_varyings out = {};\n    out.position = float4(0, 0, 0, -1);\n";
	if (options.packed_inputs) out << "    const rsx_packed_vertex_inputs attributes{persistent, transient, &fetch};\n";
	if (options.point_output)
		out << "    out.point_size = ctx.point_size;\n";
	out << "    if (!ctx.predicate || vertex_id >= ctx.geometry_count) return out;\n"
		<< "    const auto reference = references[vertex_id];\n"
		<< "    auto r = rsx_evaluate(attributes, constants, ctx, reference.source_vertex, sampler_state" << texture_values << ");\n";
	if (options.flat_colors)
	{
		out << "    if (reference.flat_vertex != reference.source_vertex) {\n";
		if (options.flat_result_inputs)
			out << "        uint low=0, high=ctx.flat_count;\n"
				<< "        while (low<high) { const uint mid=low+(high-low)/2; if (flat_ids[mid]<reference.flat_vertex) low=mid+1; else high=mid; }\n"
				<< "        if (low==ctx.flat_count || flat_ids[low]!=reference.flat_vertex) return out;\n"
				<< "        const auto flat=flat_results[low]; r.fault |= flat.fault;\n"
				<< "        for (uint i=1; i<=4; ++i) r.dst[i]=as_type<float4>(flat.colors[i-1]);\n";
		else out << "        const auto flat = rsx_evaluate(attributes, constants, ctx, reference.flat_vertex, sampler_state" << texture_values << ");\n"
			<< "        r.fault |= flat.fault;\n        for (uint i = 1; i <= 4; ++i) r.dst[i] = flat.dst[i];\n";
		out << "    }\n";
	}
	out << "    if (r.fault) return out;\n";
	const auto enabled_output = [&](unsigned reg, unsigned mask) -> std::string
	{
		return (program.output_register_mask & (1u << reg)) ? "(ctx.output_mask & " + number(mask) + "u)" : "false";
	};
	for (unsigned i = 1; i <= 4; ++i)
	{
		const auto mask = (i == 1 || i == 3) ? 5u : 10u;
		out << "    out.color" << i - 1 << " = " << enabled_output(i, mask) << " ? r.dst[" << i << "] : float4(0, 0, 0, 1);\n";
	}
	out << "    out.fog = " << enabled_output(5, 16) << " ? float4(r.dst[5].x) : float4(0, 0, 0, 1);\n";
	if (options.point_output)
		out << "    out.point_size = " << enabled_output(6, 32) << " ? r.dst[6].x : ctx.point_size;\n";
	for (unsigned i = 0; i < 10; ++i)
	{
		const unsigned mask = i == 8 ? 12 : i == 9 ? 13
												   : i + 14;
		const unsigned reg = i == 8 ? 15 : i == 9 ? 6
												  : i + 7;
		out << "    out.tex" << i << " = " << enabled_output(reg, 1u << mask) << " ? r.dst[" << reg << "] : float4(0, 0, 0, 1);\n";
	}
	for (unsigned i = 0; i < 6; ++i)
	{
		out << "    out.clip[" << i << "] = (" << enabled_output(i < 3 ? 5 : 6, 1u << (i + 6)) << " && (ctx.clip_enable & " << (1u << i)
			<< "u)) ? r.dst[" << (i < 3 ? 5 : 6) << "][" << i % 3 + 1 << "] * ctx.clip_factors[" << i << "] : 0.5f;\n";
	}
	out << "    out.position = rsx_zclip(r.dst[0] * ctx.scale_offset, ctx.z_near, ctx.z_far);\n    return out;\n}\n";
	program.source = out.str();
	return program;
}

} // namespace metal4
