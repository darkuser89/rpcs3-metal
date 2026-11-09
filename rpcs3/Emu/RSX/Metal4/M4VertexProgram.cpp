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

bool scalar_pair_has_register_hazard(const vertex_instruction& instruction)
{
	const auto d0 = instruction.words[0], d3 = instruction.words[3];
	const auto vector_temp = bits(d0, 15, 6);
	const auto vector_mask = bits(d3, 13, 4);
	const auto& source = instruction.sources[2];
	if (instruction.vector_opcode == 13 || instruction.vector_opcode == 23 || instruction.vector_opcode == 24)
	{
		const bool indexed = (source.type == 2 && bits(d0, 27, 1)) ||
			(source.type == 3 && bits(d3, 1, 1));
		return indexed && (vector_temp & 1u) == bits(d0, 24, 1) &&
			(vector_mask & (8u >> bits(d0, 0, 2)));
	}
	if (vector_temp == 63)
		return false;
	if (vector_temp == bits(d3, 7, 6) && (vector_mask & bits(d3, 17, 4)))
		return true;
	if (source.type != 1 || vector_temp != source.temporary)
		return false;
	// MOV reads the enabled result lanes. LIT reads x/y/w. The
	// remaining scalar operations take the first swizzled component.
	const auto scalar_lanes = instruction.scalar_opcode == 1 ? bits(d3, 17, 4) :
		instruction.scalar_opcode == 7 ? 13u : 8u;
	for (unsigned lane = 0; lane < source.swizzle.size(); ++lane)
		if ((scalar_lanes & (8u >> lane)) && (vector_mask & (8u >> source.swizzle[lane])))
			return true;
	return false;
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
			return "rsx_extremum(" + source(0) + ", " + source(1) + ", false)";
		case 10:
			return "rsx_extremum(" + source(0) + ", " + source(1) + ", true)";
		case 11:
			return "rsx_set_compare(" + source(0) + ", " + source(1) + ", 1u)";
		case 12:
			return "rsx_set_compare(" + source(0) + ", " + source(1) + ", 6u)";
		case 14:
			return "rsx_fraction(" + source(0) + ")";
		case 15:
			return "rsx_floor(" + source(0) + ")";
		case 16:
			return "rsx_set_compare(" + source(0) + ", " + source(1) + ", 2u)";
		case 17:
			return "float4(0)";
		case 18:
			return "rsx_set_compare(" + source(0) + ", " + source(1) + ", 4u)";
		case 19:
			return "rsx_set_compare(" + source(0) + ", " + source(1) + ", 3u)";
		case 20:
			return "rsx_set_compare(" + source(0) + ", " + source(1) + ", 5u)";
		case 21:
			return "float4(1)";
		case 22:
			return "rsx_sign(" + source(0) + ")";
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
		// Scalar and per-lane evaluation agree when every written lane
		// selects the same source as x. For a pair, also require that the
		// visible result is independent of the relative read/write timing.
		const auto& components = m_instruction.sources[2].swizzle;
		const auto mask = bits(m_instruction.words[3], 17, 4);
		bool non_replicate = false;
		for (unsigned lane = 1; lane < components.size(); ++lane)
		{
			if (components[lane] == components[0])
				continue;
			non_replicate = true;
			if (!mask || (mask & (8u >> lane)))
				throw std::invalid_argument("RSX scalar ALU operand has an ambiguous non-replicate swizzle");
		}
		if (non_replicate && m_instruction.vector_opcode && scalar_pair_has_register_hazard(m_instruction))
			throw std::invalid_argument("RSX paired scalar ALU operand has a register hazard");
		const auto s = "(" + source(2) + ").x";
		std::string result;
		switch (m_instruction.scalar_opcode)
		{
		case 2:
			result = "rsx_reciprocal(" + s + ")";
			break;
		case 3:
			result = "rsx_rcp_clamped(" + s + ")";
			break;
		case 4:
			result = "rsx_rsqrt(" + s + ")";
			break;
		case 5:
			return "rsx_exp_legacy(" + s + ")";
		case 6:
			return "rsx_log_legacy(" + s + ")";
		case 13:
			result = "rsx_log2_scalar(" + s + ")";
			break;
		case 14:
			result = "rsx_exp2_scalar(" + s + ")";
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

	void write(std::ostream& out, bool scalar, std::string value, bool update_condition)
	{
		const auto d0 = m_instruction.words[0], d3 = m_instruction.words[3];
		const auto temporary = scalar ? bits(d3, 7, 6) : bits(d0, 15, 6);
		const bool output = scalar ? !bits(d0, 30, 1) : bits(d0, 30, 1);
		if (bits(d0, 26, 1) && bits(d0, 14, 1))
		{
			// The condition is derived from the written value. Ordered compares
			// preserve unordered operands and the sign bit of zero during SAT.
			out << "            { const float4 raw_value = " << value << ";\n"
				<< "              const float4 value = rsx_saturate_ordered(raw_value);\n";
		}
		else
		{
			if (bits(d0, 26, 1))
				value = "clamp(" + value + ", 0.0f, 1.0f)";
			out << "            { const float4 value = " << value << ";\n";
		}
		std::array<std::string, 3> destinations;
		std::size_t count = 0;
		if (temporary != 63)
			destinations[count++] = "r.temp[" + number(temporary) + "]";
		if (output && bits(d3, 2, 5) < 16)
		{
			if (bits(d0, 28, 1))
			{
				out << "                rsx_store_output(r, long(" << bits(d3, 2, 5) << ") + long(r.address["
					<< bits(d0, 24, 1) << "][" << bits(d0, 0, 2) << "]), value, write_mask & bool4(";
				for (unsigned c = 0; c < 4; ++c)
					out << (c ? ", " : "") << (bits(d3, (scalar ? 20 : 16) - c, 1) ? "true" : "false");
				out << "));\n";
				m_program.output_register_mask |= 65535u;
			}
			else
			{
				destinations[count++] = "r.dst[" + number(bits(d3, 2, 5)) + "]";
				m_program.output_register_mask |= 1u << bits(d3, 2, 5);
			}
		}
		if (update_condition || (count == 0 && !output))
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
		const auto vector = instruction.vector_opcode, scalar = instruction.scalar_opcode;
		const bool scalar_alu = (scalar >= 1 && scalar <= 7) || (scalar >= 13 && scalar <= 16);
		const bool update_condition = bits(d0, 14, 1);
		const bool condition_from_vector = bits(d0, 29, 1);
		const bool address_operation = vector == 13 || vector == 23 || vector == 24;
		const bool update_from_address = address_operation && update_condition && condition_from_vector;
		if ((vector == 23 || vector == 24) && (bits(d0, 26, 1) || (bits(d0, 30, 1) && bits(d3, 2, 5) < 16)))
			throw std::invalid_argument("RSX integer address operation has unsupported saturation or output routing");
		if (update_condition || condition_from_vector)
		{
			// Curie CDST_WM and CDST_IS_VEC select one result. Paired
			// operations use the same pre-write predicate.
			const bool valid_result = condition_from_vector ? (vector && (!scalar || scalar_alu)) : scalar_alu;
			if (!update_condition || !valid_result)
				throw std::invalid_argument("RSX vertex condition-update flags require further hardware validation");
			if (address_operation && ((update_from_address && bits(d0, 26, 1)) ||
				(bits(d0, 30, 1) && bits(d3, 2, 5) < 16)))
				throw std::invalid_argument("RSX address condition update has unsupported saturation or output routing");
			if (vector && scalar)
			{
				const bool nonselected_implicit_condition = condition_from_vector ?
					(bits(d0, 30, 1) && bits(d3, 7, 6) == 63 && bits(d3, 17, 4)) :
					(!address_operation && !bits(d0, 30, 1) && bits(d0, 15, 6) == 63 && bits(d3, 13, 4));
				if (nonselected_implicit_condition || scalar_pair_has_register_hazard(instruction))
					throw std::invalid_argument("RSX paired vertex condition update has a register hazard");
			}
		}
		if (bits(d3, 2, 5) >= 16 && bits(d3, 2, 5) != 31)
		{
			throw std::invalid_argument("RSX vertex output register exceeds the register file");
		}
		if (scalar_alu &&
			!bits(d0, 30, 1) && bits(d3, 2, 5) < 16 && bits(d3, 7, 6) != 63)
		{
			// Both destinations consume the same local value and write mask.
			if (vector && scalar_pair_has_register_hazard(instruction))
				throw std::invalid_argument("RSX scalar dual destination has a paired register hazard");
		}
		if (address_operation && scalar_alu && !bits(d0, 30, 1) && bits(d0, 28, 1) && bits(d3, 2, 5) < 16 && bits(d3, 17, 4) &&
			(bits(d0, 15, 6) & 1u) == bits(d0, 24, 1) && (bits(d3, 13, 4) & (8u >> bits(d0, 0, 2))))
			throw std::invalid_argument("RSX paired address load changes the indexed output address");
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
		if (address_operation)
		{
			body << "            {\n";
			if (update_from_address)
				body << "                uint address_fault = 0;\n";
			// ARR uses nearest-even rounding; Metal round() instead rounds
			// halfway cases away from zero. Keep ARL's conversion separate.
			const auto value = vector == 24 ? "r.address[" + number(bits(d0, 24, 1)) + "]" :
				vector == 23 ? "rint(" + emit.source(0) + ")" : emit.source(0);
			body << "                const int4 value = " << (vector == 24 ? "rsx_address_add(" : "rsx_address(") << value
				<< (update_from_address ? ", address_fault);\n                r.fault |= address_fault;\n" : ", r.fault);\n");
			for (unsigned c = 0; c < 4; ++c)
			{
				if (!bits(d3, 16 - c, 1))
					continue;
				body << "                if (write_mask[" << c << "]) { r.address["
					 << (bits(d0, 15, 6) & 1) << "][" << c << "] = value[" << c << "];";
				if (update_from_address)
					body << " if (!address_fault) r.condition[" << bits(d0, 25, 1) << "][" << c << "] = float(value[" << c << "]);";
				body << " }\n";
			}
			body << "                (void)value;\n            }\n";
		}
		else if (vector)
			emit.write(body, false, emit.vector_value(), update_condition && condition_from_vector);
		// The current VK interpreter evaluates SCA after VEC. Dual-issue source
		// timing is retained here and remains part of the hardware parity audit.
		if (scalar_alu)
		{
			emit.write(body, true, emit.scalar_value(), update_condition && !condition_from_vector);
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
