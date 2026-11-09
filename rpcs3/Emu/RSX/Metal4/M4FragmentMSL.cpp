#include "M4FragmentMSL.h"
#include "M4LogicOpMSL.h"
#include "M4BlendROPMSL.h"
#include "M4OutputROPMSL.h"
#include "M4FragmentRegisters.h"
#include "M4FragmentRuntime.h"
#include "M4Varyings.h"
#include "M4SamplerParameters.h"
#include "M4TextureCoordinates.h"
#include "M4TexelProcessing.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace metal4
{
namespace
{

std::string n(std::size_t value)
{
	return std::to_string(value);
}

std::string swizzle(const std::array<std::uint32_t, 4>& channels)
{
	std::string result = ".";
	for (auto c : channels)
		result += "xyzw"[c];
	return result;
}

std::string precision(std::string value, std::uint32_t mode)
{
	switch (mode)
	{
	case 0:
	case 5:
		return value;
	case 1:
		return "float4(half4(" + value + "))";
	case 2:
		return "rsx_fp_clamp(" + value + ", -2.0f, 2.0f)";
	case 3:
		return "rsx_fp_clamp(" + value + ", -1.0f, 1.0f)";
	case 4:
		return "rsx_fp_clamp(" + value + ", 0.0f, 1.0f)";
	default:
		throw std::invalid_argument("Unsupported RSX fragment precision modifier");
	}
}

std::string texture_type(vertex_texture_dimension dimension, bool shadow = false, bool atlas = false)
{
	switch (dimension)
	{
	case vertex_texture_dimension::one:
	case vertex_texture_dimension::two:
		return shadow ? "depth2d<float>" : "texture2d<float>";
	case vertex_texture_dimension::three:
		return "texture3d<float>";
	case vertex_texture_dimension::cube:
		if (atlas) return shadow ? "depth2d_array<float>" : "texture2d_array<float>";
		return shadow ? "depthcube<float>" : "texturecube<float>";
	}
	throw std::invalid_argument("Unknown RSX fragment texture dimension");
}

class generator
{
	const decoded_fragment_program& m_decoded;
	const fragment_program_options& m_options;
	generated_fragment_program& m_output;
	std::vector<std::size_t> m_literal_slots;
	std::ostringstream m_body;

	std::size_t target(std::uint64_t byte_offset) const
	{
		if (byte_offset == m_decoded.byte_length)
			return m_decoded.instructions.size();
		const auto found = std::lower_bound(m_decoded.instructions.begin(), m_decoded.instructions.end(), byte_offset,
											[](const fragment_instruction& instruction, std::uint64_t offset)
											{
												return instruction.byte_offset < offset;
											});
		if (found == m_decoded.instructions.end() || found->byte_offset != byte_offset)
		{
			throw std::invalid_argument("RSX fragment branch points outside instructions or into literal data");
		}
		return static_cast<std::size_t>(found - m_decoded.instructions.begin());
	}

	std::string condition(const fragment_instruction& instruction) const
	{
		return "rsx_fp_condition(r.condition[" + n(instruction.condition_read) + "]" +
			   swizzle(instruction.condition_swizzle) + ", " + n(instruction.condition_mode) + "u)";
	}

	std::string source(std::size_t index, unsigned operand)
	{
		const auto& instruction = m_decoded.instructions[index];
		const auto& s = instruction.sources[operand];
		auto mode = s.precision;
		std::string value;
		switch (s.type)
		{
		case 0:
			(void)fragment_location(s.temporary, 0, s.half);
			value = "rsx_fp_read(r, " + n(s.temporary) + "u, " + (s.half ? "true" : "false") + ")";
			break;
		case 1:
		{
			std::string attribute;
			if (instruction.indexed_input)
			{
				attribute = "4 + r.address + " + n(instruction.address);
				m_output.input_mask |= 0x3ff0;
			}
			else
			{
				if (instruction.attribute >= 15)
					throw std::invalid_argument("Unknown RSX fragment input attribute");
				attribute = n(instruction.attribute);
				m_output.input_mask |= 1u << instruction.attribute;
				// Color/facing inputs have their own fixed interpretation in VK.
				if (instruction.attribute == 1 || instruction.attribute == 2 || instruction.attribute == 14)
					mode = 0;
			}
			value = "rsx_fp_input(in, ctx, point_coord, front_facing, " + attribute + ", " +
					(instruction.perspective ? "true" : "false") + ", r.fault)";
			break;
		}
		case 2:
			if (!instruction.literal)
				throw std::invalid_argument("RSX fragment literal is missing");
			value = "rsx_fp_constant(constants, ctx, " + n(m_literal_slots[index]) + "u, r.fault)";
			break;
		default:
			throw std::invalid_argument("RSX fragment source register type 3 is invalid");
		}
		value = "(" + value + ")" + swizzle(s.swizzle);
		if (mode == 4)
			value = precision(value, mode);
		if (s.absolute)
			value = "abs(" + value + ")";
		if (mode != 4)
			value = precision(value, mode);
		if (s.negate)
			value = "-(" + value + ")";
		return value;
	}

	bool half_source(const fragment_instruction& instruction, unsigned operand) const
	{
		const auto& s = instruction.sources[operand];
		if (s.type == 1 && !instruction.indexed_input &&
			(instruction.attribute == 1 || instruction.attribute == 2 || instruction.attribute == 14))
			return false;
		return s.precision == 1 || (s.type == 0 && s.half);
	}

	std::string alu(std::size_t index)
	{
		const auto& instruction = m_decoded.instructions[index];
		const auto op = instruction.opcode;
		const auto s = [&](unsigned operand)
		{
			return source(index, operand);
		};
		// Preserve native half arithmetic when the VK expression has only half
		// operands. Operations explicitly promoted by VK use float throughout.
		const auto typed = [&](unsigned operand, bool use_half)
		{
			return use_half ? "half4(" + s(operand) + ")" : s(operand);
		};
		const bool unary_half = half_source(instruction, 0);
		const bool binary_half = unary_half && half_source(instruction, 1);
		switch (op)
		{
		case 1:
			return s(0);
		case 2:
			return "float4(" + typed(0, binary_half) + " * " + typed(1, binary_half) + ")";
		case 3:
			return "float4(" + typed(0, binary_half) + " + " + typed(1, binary_half) + ")";
		case 4:
			return "fma(" + s(0) + ", " + s(1) + ", " + s(2) + ")";
		case 5:
			return "float4(dot((" + s(0) + ").xyz, (" + s(1) + ").xyz))";
		case 6:
			return "float4(dot(" + s(0) + ", " + s(1) + "))";
		case 7:
			return "float4(1, (" + s(0) + ").y * (" + s(1) + ").y, (" + s(0) + ").z, (" + s(1) + ").w)";
		case 8:
			return "min(" + s(0) + ", " + s(1) + ")";
		case 9:
			return "max(" + s(0) + ", " + s(1) + ")";
		case 10:
			return "float4(" + s(0) + " < " + s(1) + ")";
		case 11:
			return "float4(" + s(0) + " >= " + s(1) + ")";
		case 12:
			return "float4(" + s(0) + " <= " + s(1) + ")";
		case 13:
			return "float4(" + s(0) + " > " + s(1) + ")";
		case 14:
			return "float4(" + s(0) + " != " + s(1) + ")";
		case 15:
			return "float4(" + s(0) + " == " + s(1) + ")";
		case 16:
			return "float4(fract(" + typed(0, unary_half) + "))";
		case 17:
			return "float4(floor(" + typed(0, unary_half) + "))";
		case 19:
			return "float4(as_type<float>(pack_float_to_snorm4x8(" + s(0) + ")))";
		case 20:
			return "unpack_snorm4x8_to_float(as_type<uint>((" + s(0) + ").x))";
		case 21:
			return "dfdx(" + s(0) + ")";
		case 22:
			return "dfdy(" + s(0) + ") * ctx.derivative_sign_y";
		case 26:
			return "float4(" + std::string(unary_half ? "half(1)" : "1.0f") + " / (" + typed(0, unary_half) + ").x)";
		case 27:
			return "float4(" + std::string(unary_half ? "half(1)" : "1.0f") + " / sqrt(abs((" + typed(0, unary_half) + ").x)))";
		case 28:
			return "float4(exp2((" + typed(0, unary_half) + ").x))";
		case 29:
			return "float4(log2((" + typed(0, unary_half) + ").x))";
		case 30:
			return "rsx_fp_lit(" + s(0) + ")";
		case 31:
		{
			const bool half = binary_half && half_source(instruction, 2);
			return "float4(mix(" + typed(2, half) + ", " + typed(1, half) + ", " + typed(0, half) + "))";
		}
		case 32:
			return "float4(1)";
		case 33:
			return "float4(0)";
		case 34:
			return "float4(cos((" + typed(0, unary_half) + ").x))";
		case 35:
			return "float4(sin((" + typed(0, unary_half) + ").x))";
		case 36:
			return "float4(as_type<float>(as_type<uint>(half2((" + s(0) + ").xy))))";
		case 37:
			return "float2(as_type<half2>(as_type<uint>((" + s(0) + ").x))).xyxy";
		case 39:
			return "float4(as_type<float>(pack_float_to_unorm4x8(" + s(0) + ")))";
		case 40:
			return "unpack_unorm4x8_to_float(as_type<uint>((" + s(0) + ").x))";
		case 41:
			return "float4(as_type<float>(pack_float_to_unorm2x16((" + s(0) + ").xy)))";
		case 42:
			return "float4(unpack_unorm2x16_to_float(as_type<uint>((" + s(0) + ").x)), float2(0)).xyxy";
		case 44:
			return "float4(as_type<float>(pack_float_to_unorm4x8(rsx_fp_linear_srgb(" + s(0) + "))))";
		case 45:
			return "rsx_fp_srgb_linear(unpack_unorm4x8_to_float(as_type<uint>((" + s(0) + ").x)))";
		case 46:
			return "float4(dot((" + s(0) + ").xy, (" + s(1) + ").xy) + (" + s(2) + ").x)";
		case 54:
			return "reflect(" + s(0) + ", " + s(1) + ")";
		case 56:
			return "float4(dot((" + s(0) + ").xy, (" + s(1) + ").xy))";
		case 57:
			return "rsx_fp_normalize(" + s(0) + ")";
		case 58:
			return "float4(" + typed(0, binary_half) + " / (" + typed(1, binary_half) + ").x)";
		case 59:
			return "rsx_fp_divsq(" + s(0) + ", (" + s(1) + ").x)";
		case 60:
			return "rsx_fp_lif(" + s(0) + ")";
		case 23:
		case 24:
		case 25:
		case 47:
		case 49:
		{
			const auto unit = instruction.texture;
			m_output.texture_mask |= 1u << unit;
			const auto dimension = m_options.textures[unit];
			const bool shadow = m_options.shadow_textures & (1u << unit);
			const bool depth = m_options.depth_textures & (1u << unit);
			const bool stored_repeat = m_options.stored_repeat_textures & (1u << unit);
			const bool stored_mixed = m_options.stored_mixed_filter_textures & (1u << unit);
			const bool atlas = m_options.stored_border_atlas_textures & (1u << unit);

			if (shadow && (dimension == vertex_texture_dimension::three || (dimension == vertex_texture_dimension::cube && op == 24)))
				throw std::invalid_argument("3D/projected-cube RSX shadows require a separate native sampling policy");
			if ((stored_repeat || stored_mixed) && dimension != vertex_texture_dimension::two)
				throw std::invalid_argument("Unsupported stored-border shader sampling dimension");
			const auto suffix = dimension == vertex_texture_dimension::two ? ".xy" : ".xyz";
			const auto coordinate = [&](unsigned operand, bool project, bool gradient)
			{
				const auto value = "(" + s(operand) + ")";
				if (dimension == vertex_texture_dimension::one)
				{
					const auto scalar = value + ".x" + (project ? " / " + value + ".w" : "");
					return gradient ? "float2(" + scalar + ", 0.0f)" : "float2(rsx_texture_coordinate(" + scalar + ", texture_state[" + n(unit) + "]), 0.5f)";
				}
				const auto components = value + suffix + (project ? " / " + value + ".w" : "");
				return gradient ? components : "rsx_texture_coordinate(" + components + ", texture_state[" + n(unit) + "])";
			};
			const auto process_texel = [&](const std::string& sample)
			{
				return "rsx_process_texel(" + sample + ", texel_state[" + n(unit) + "], " + (instruction.expand_texture ? "true" : "false") + ", r.killed, r.fault)";
			};
			const auto depth_remap = [&]
			{
				const auto& remap = m_options.depth_remap[unit];
				validate_depth_sample_remap(remap);
				return "uint4(" + n(static_cast<unsigned>(remap[0])) + ", " + n(static_cast<unsigned>(remap[1])) +
					", " + n(static_cast<unsigned>(remap[2])) + ", " + n(static_cast<unsigned>(remap[3])) + ")";
			};
			if ((m_options.multisampled_textures | m_options.sample_raster_textures) & (1u << unit))
			{
				const auto image = "ftex" + n(unit);
				const auto arguments = ", " + coordinate(0, op == 24, false) + ", sampler_state[" + n(unit) +
					"], texture_state[" + n(unit) + "]";
				if (shadow)
				{
					const auto reference = "(" + s(0) + ").z" + (op == 24 ? " / (" + s(0) + ").w" : "");
					return "rsx_multisample_shadow(" + image + arguments + ", " + reference + ")";
				}
				if (m_options.redirected_textures & (1u << unit))
					return process_texel("rsx_multisample_redirected(" + image + ", fstencil" + n(unit) + arguments + ")");
				// The single-level reconstruction stays in screen space. Only a
				// mixed filter uses the opcode's LOD or gradients to choose Min/Mag.
				std::string filter = ", rsx_multisample_filter_control{}";
				if (op == 25)
				{
					const auto transform = "texture_state[" + n(unit) + "]";
					const auto scale = " * float2(" + transform + ".scale_x, " + transform + ".scale_y)";
					filter = ", rsx_multisample_filter_control{" + coordinate(1, false, true) + scale + ", " +
						coordinate(2, false, true) + scale + ", 0.0f, 1u}";
				}
				if (op == 47 || op == 49)
					filter = ", rsx_multisample_filter_control{float2(0), float2(0), (" + s(1) + ").x, " +
						std::string(op == 47 ? "2u}" : "0u}");
				if (depth)
					return process_texel("rsx_multisample_depth(" + image + arguments + filter + ", " + depth_remap() + ")");
				return process_texel("rsx_multisample_color(" + image + arguments + filter + ")");
			}
			if (m_options.redirected_textures & (1u << unit))
			{
				// The redirected binding has one level and point filtering.
				return process_texel("rsx_redirected_depth(ftex" + n(unit) + ", fstencil" + n(unit) + ", " +
					coordinate(0, op == 24, false) + ", sampler_state[" + n(unit) + "])");
			}
			const auto stored_sample = [&](const std::string& reference)
			{
				const auto image = "ftex" + n(unit);
				const auto parameters = "texture_state[" + n(unit) + "]";
				const auto arguments = image + ", fsampler" + n(unit) + ", " +
					coordinate(0, op == 24, stored_mixed || atlas) + ", " + parameters + ", " + reference;
				if (atlas)
				{
					const auto sampling = "sampler_state[" + n(unit) + "]";
					const auto raw = coordinate(0, op == 24, true);
					const bool gradients = !shadow && op == 25;
					const auto dx = gradients ? coordinate(1, false, true) : "dfdx(" + raw + ")";
					const auto dy = gradients ? coordinate(2, false, true) : "dfdy(" + raw + ")";
					const bool cube_atlas = dimension == vertex_texture_dimension::cube;
					if (!shadow && op == 47)
						return std::string(cube_atlas ? "rsx_sample_border_cube(" : "rsx_sample_border_atlas(") +
							arguments + ", " + sampling + ", (" + s(1) + ").x)";
					const auto bias = !shadow && op == 49 ? "(" + s(1) + ").x" : "0.0f";
					return std::string(cube_atlas ? "rsx_sample_border_cube_anisotropic(" : "rsx_sample_border_anisotropic(") +
						arguments + ", " + sampling + ", " + dx + ", " + dy + ", " + bias + ")";
				}
				if (!stored_mixed)
					return "rsx_sample_stored_repeat(" + arguments + ")";
				auto bias = "sampler_state[" + n(unit) + "].bias";
				if (!shadow && op == 47)
					return "rsx_sample_stored_mixed(" + arguments + ", ((" + s(1) + ").x + " + bias + " > 0.0f))";
				const auto raw = coordinate(0, op == 24, true);
				const bool gradients = !shadow && op == 25;
				const auto dx = gradients ? coordinate(1, false, true) : "dfdx(" + raw + ")";
				const auto dy = gradients ? coordinate(2, false, true) : "dfdy(" + raw + ")";
				if (!shadow && op == 49)
					bias += " + (" + s(1) + ").x";
				return "rsx_sample_stored_mixed(" + arguments + ", rsx_stored_minifies(" +
					image + ", " + parameters + ", " + dx + ", " + dy + ", " + bias + "))";
			};
			std::string sample = "ftex" + n(unit) + (shadow ? ".sample_compare(fsampler" : ".sample(fsampler") + n(unit) + ", " + coordinate(0, op == 24, false);
			if (shadow)
			{
				const auto reference = "(" + s(0) + ")" + (dimension == vertex_texture_dimension::cube ? ".w" : ".z") + (op == 24 ? " / (" + s(0) + ").w" : "");
				const auto adjusted_reference = "rsx_shadow_reference(" + reference + ", sampler_state[" + n(unit) + "])";
				if (stored_repeat || stored_mixed || atlas)
					return "float4(" + stored_sample(adjusted_reference) + ")";
				sample += ", " + adjusted_reference;
				// VK's shadow redirection discards the original LOD/gradient/bias
				// opcode distinction. Sampler bias still applies; projection is
				// carried by the coordinate/reference transform above.
				sample += ", bias(sampler_state[" + n(unit) + "].bias))";
				if (m_options.border_textures & (1u << unit))
				{
					const auto arguments = ", " + adjusted_reference + ", bias(sampler_state[" + n(unit) + "].bias))";
					const auto coverage = "fborder" + n(unit) + ".sample_compare(fsampler" + n(unit) + ", " + coordinate(0, op == 24, false) + arguments;
					const auto center = "fborder" + n(unit) + ".sample_compare(fsampler" + n(unit) + ", " + (dimension == vertex_texture_dimension::cube ? "float3(1,0,0)" : "float2(0.5f)") + arguments;
					// Independent fixed-precision PCF results can round in
					// opposite directions; comparison coverage stays in [0,1].
					sample = "clamp(" + sample + " + (" + center + " - " + coverage + "), 0.0f, 1.0f)";
				}
				return "float4(" + sample + ")";
			}
			if (stored_repeat || stored_mixed || atlas)
				sample = stored_sample("false");
			else
			{
				if (op == 47 || op == 49)
					sample += op == 47 ? ", level(clamp((" + s(1) + ").x + sampler_state[" + n(unit) + "].bias, sampler_state[" + n(unit) + "].min_lod, sampler_state[" + n(unit) + "].max_lod))" : ", bias((" + s(1) + ").x + sampler_state[" + n(unit) + "].bias)";
				if (op == 25)
				{
					const auto gradient = dimension == vertex_texture_dimension::three ? "gradient3d" :
						dimension == vertex_texture_dimension::cube ? "gradientcube" : "gradient2d";
					sample += ", " + std::string(gradient) + "(" + coordinate(1, false, true) + " * exp2(sampler_state[" + n(unit) + "].bias), " + coordinate(2, false, true) + " * exp2(sampler_state[" + n(unit) + "].bias))";
				}
				else if (op != 47 && op != 49)
					sample += ", bias(sampler_state[" + n(unit) + "].bias)";
				sample += ")";
			}
			auto coverage = sample;
			coverage.replace(0, ("ftex" + n(unit)).size(), "fborder" + n(unit));
			if (depth)
			{
				if(dimension==vertex_texture_dimension::three)sample="("+sample+").r";
				sample = "rsx_depth_texel(" + sample + ", " + depth_remap() + ")";
			}
			if (m_options.border_textures & (1u << unit))
				sample = "rsx_border_color(" + sample + ", (" + coverage + ").r, sampler_state[" + n(unit) + "])";
			return process_texel(sample);
		}
		default:
			throw std::invalid_argument("Unsupported RSX fragment opcode " + n(op));
		}
	}

	void write(std::size_t index, std::string value)
	{
		const auto& instruction = m_decoded.instructions[index];
		if (instruction.scale == 4)
			throw std::invalid_argument("Unknown RSX fragment output scale");
		if (instruction.scale)
		{
			const auto shift = instruction.scale <= 3 ? instruction.scale : instruction.scale - 4;
			value = "(" + value + (instruction.scale <= 3 ? " * " : " / ") + n(1u << shift) + ".0f)";
		}
		if (!instruction.no_destination)
		{
			if (instruction.half_destination)
				value = precision(value, 1);
			if (instruction.saturate)
				value = precision(value, 4);
			else if (instruction.precision > 1)
			{
				switch (instruction.opcode)
				{
				case 1:
					if (instruction.half_destination && instruction.sources[0].half && instruction.sources[0].type == 0)
						break;
					[[fallthrough]];
				default:
					value = precision(value, instruction.precision);
					break;
				case 8:
				case 9:
				case 16:
				case 29:
				case 30:
				case 34:
				case 35:
				case 54:
				case 57:
				case 60:
					break;
				}
			}
		}
		if ((instruction.opcode == 23 || instruction.opcode == 24 || instruction.opcode == 25 || instruction.opcode == 47 || instruction.opcode == 49) && instruction.condition_mode == 0)
			value = "(false ? " + value + " : float4(0))";
		m_body << "        { const bool4 mask = " << condition(instruction) << ";\n"
			   << "          const float4 value = " << value << "; (void)value; (void)mask;\n";
		for (unsigned channel = 0; channel < 4; ++channel)
		{
			if (!(instruction.write_mask & (1u << channel)))
				continue;
			const auto c = n(channel);
			const auto cc = "r.condition[" + n(instruction.condition_write) + "][" + c + "]";
			if (instruction.no_destination)
			{
				if (instruction.set_condition)
					m_body << "          if (any(mask)) " << cc << " = value[" << c << "];\n";
				continue;
			}
			const auto location = fragment_location(instruction.destination, channel, instruction.half_destination);
			const auto word = "r.words[" + n(location.word) + "]";
			m_body << "          if (mask[" << c << "]) ";
			if (location.half)
				m_body << "rsx_fp_store_half(" << word << ", " << location.shift << "u, value[" << c << "]);\n";
			else
				m_body << word << " = as_type<uint>(value[" << c << "]);\n";
			if (instruction.set_condition)
			{
				m_body << "          " << cc << " = rsx_fp_read(r, " << instruction.destination << "u, "
					   << (instruction.half_destination ? "true" : "false") << ")[" << c << "];\n";
			}
		}
		m_body << "        }\n";
	}

	void range(std::size_t begin, std::size_t end, unsigned depth, unsigned loops)
	{
		if (depth > 64 || loops > 8)
			throw std::invalid_argument("RSX fragment flow nesting exceeds native limits");
		for (std::size_t i = begin; i < end; ++i)
		{
			const auto& instruction = m_decoded.instructions[i];
			const auto op = instruction.opcode;
			m_body << "        // RSX byte offset " << instruction.byte_offset << "\n"
				   << "        if (steps == ctx.instruction_budget) { r.fault |= 2u; return r; } ++steps;\n";
			if (op == 0)
			{ /* NOP still owns its literal and physical address. */
			}
			else if (op == 18)
			{
				m_output.has_kill_instruction = true;
				m_body << "        if (any(" << condition(instruction) << ")) r.killed = true;\n";
			}
			else if (op == 0x45)
				m_body << "        if (any(" << condition(instruction) << ")) return r;\n";
			else if (op == 0x40)
			{
				if (!loops)
					throw std::invalid_argument("RSX fragment BRK is outside a loop");
				m_body << "        if (any(" << condition(instruction) << ")) break;\n";
			}
			else if (op == 0x42 || op == 0x43 || op == 0x44)
			{
				const auto finish = target(std::uint64_t(instruction.words[3]) * 4);
				if (finish < i + 1 || finish > end)
					throw std::invalid_argument("RSX fragment flow region overlaps or runs backward");
				if (op == 0x42)
				{
					const auto alternate = target(std::uint64_t(instruction.words[2] & 0x7fffffffu) * 4);
					if (alternate < i + 1 || alternate > finish)
						throw std::invalid_argument("RSX fragment ELSE is outside its IF region");
					m_body << "        if (any(" << condition(instruction) << ")) {\n";
					range(i + 1, alternate, depth + 1, loops);
					m_body << "        } else {\n";
					range(alternate, finish, depth + 1, loops);
					m_body << "        }\n";
				}
				else
				{
					const auto count = (instruction.words[2] >> 2) & 255u;
					const auto initial = (instruction.words[2] >> 10) & 255u;
					const auto encoded_step = (instruction.words[2] >> 19) & 255u;
					const auto step = encoded_step < 128 ? static_cast<int>(encoded_step) : static_cast<int>(encoded_step) - 256;
					m_body << "        if (any(" << condition(instruction) << ")) {\n";
					if (op == 0x43)
						m_body << "          const int old_address" << i << " = r.address;\n";
					m_body << "          for (uint loop" << i << " = 0; loop" << i << " < " << count << "u; ++loop" << i << ") {\n";
					if (op == 0x43)
						m_body << "            r.address = " << initial << " + int(loop" << i << ") * (" << step << ");\n";
					range(i + 1, finish, depth + 1, loops + 1);
					m_body << "          }\n";
					if (op == 0x43)
						m_body << "          r.address = old_address" << i << ";\n";
					m_body << "        }\n";
				}
				i = finish - 1;
			}
			else
				write(i, alu(i));
			if (instruction.end)
				m_body << "        return r;\n";
		}
	}

  public:
	generator(const decoded_fragment_program& decoded, const fragment_program_options& options,
			  generated_fragment_program& output) : m_decoded(decoded), m_options(options), m_output(output)
	{
		for (const auto& instruction : decoded.instructions)
		{
			m_literal_slots.push_back(output.constant_offsets.size());
			if (instruction.literal)
				output.constant_offsets.push_back(instruction.byte_offset + 16);
		}
	}

	std::string body()
	{
		range(0, m_decoded.instructions.size(), 0, 0);
		return m_body.str();
	}
};

void input_function(std::ostream& out)
{
	out << R"metal(
inline float4 rsx_fp_input(rsx_fragment_inputs in, constant rsx_fragment_context& ctx,
    float2 point_coord, bool front_facing, int attribute, bool perspective, thread uint& fault) {
    switch (attribute) {
    case 0: return in.position * float4(abs(ctx.wpos_scale), ctx.wpos_scale, 1, 1) +
        float4(ctx.wpos_bias_x, ctx.wpos_bias_y, 0, 0);
    case 1: return rsx_fp_clamp((ctx.two_sided && !front_facing) ? in.color2 : in.color0, 0, 1);
    case 2: return rsx_fp_clamp((ctx.two_sided && !front_facing) ? in.color3 : in.color1, 0, 1);
    case 3: return rsx_fp_fog(in.fog.x, ctx);
    case 14: return float4(front_facing ? 1.0f : -1.0f);
    }
    if (attribute < 4 || attribute >= 14) { fault |= 4u; return float4(0, 0, 0, 1); }
    const uint texcoord = uint(attribute - 4);
    float4 value;
    switch (texcoord) {
)metal";
	for (unsigned i = 0; i < 10; ++i)
		out << "    case " << i << ": value = in.tex" << i << "; break;\n";
	out << R"metal(
    default: value = float4(0, 0, 0, 1); break;
    }
    const bool two_d = (ctx.texcoord_2d_mask & (1u << texcoord)) != 0;
    if (ctx.point_sprite_mask & (1u << texcoord))
        return two_d ? float4(point_coord, 0, 1.0f / in.position.w) : float4(point_coord, 1, 0);
    if (perspective)
        return two_d ? float4(value.xy * in.position.w, 0, 1) : value * in.position.w;
    return two_d ? float4(value.xy, 0, 1.0f / in.position.w) : value;
}
)metal";
}

} // namespace

generated_fragment_program generate_fragment_program(std::span<const std::byte> bytes,
													 const fragment_program_options& options)
{
	if (options.color_count > 4)
		throw std::invalid_argument("RSX fragment program has more than four color exports");
	if (options.half_storage_mask & ~((1u << options.color_count) - 1u))
		throw std::invalid_argument("Half storage export addresses an absent output");
	validate_logic_rop(options.logic_rop);
	validate_blend_rop(options.blend_rop);
	validate_depth_mirror(options.depth_mirror);
	if (options.raster_sample_count != 1 && options.raster_sample_count != 2 && options.raster_sample_count != 4 && options.raster_sample_count != 8)
		throw std::invalid_argument("Invalid native fragment raster sample count");
	if ((options.logic_rop.enabled && options.logic_rop.sample_count != options.raster_sample_count) ||
		(options.blend_rop.enabled() && options.blend_rop.sample_count != options.raster_sample_count))
		throw std::invalid_argument("Fragment ROP and raster sample counts differ");
	const bool blend_fetch = options.blend_rop.enabled();
	if (blend_fetch && options.logic_rop.enabled)
		throw std::invalid_argument("Logic and programmable blend ROPs are mutually exclusive");
	for (unsigned i = 0; i < 4; ++i)
		if (options.blend_rop.colors[i].enabled && (i >= options.color_count || options.blend_rop.remap[i] != options.color_remap[i]))
			throw std::invalid_argument("Blend ROP addresses an absent or differently remapped output");
	const bool logic_fetch = options.logic_rop.enabled && std::any_of(options.logic_rop.bits.begin(), options.logic_rop.bits.end(), [](const auto& target)
																	  {
																		  return std::any_of(target.begin(), target.end(), [](auto width)
																							 {
																								 return width != 0;
																							 });
																	  });
	for (unsigned i = options.color_count; i < 4; ++i)
		if (std::any_of(options.logic_rop.bits[i].begin(), options.logic_rop.bits[i].end(), [](auto width)
						{
							return width != 0;
						}))
			throw std::invalid_argument("Shader logic ROP addresses an absent fragment color output");
	for (auto remap : options.color_remap)
		if (remap > color_export_remap::rgb_one)
			throw std::invalid_argument("Invalid RSX surface color export remap");
	const auto stored_single = options.stored_repeat_textures | options.stored_mixed_filter_textures;
	const auto stored_border = stored_single | options.stored_border_atlas_textures;
	const auto msaa_sampling = options.multisampled_textures | options.sample_raster_textures;
	if (stored_single & options.stored_border_atlas_textures)
		throw std::invalid_argument("Stored-border single-level and atlas sampling are mutually exclusive");
	if ((options.border_textures | stored_border | options.shadow_textures | options.depth_textures | options.redirected_textures | msaa_sampling) & ~0xffffu)
	{
		throw std::invalid_argument("RSX fragment sampling mask exceeds sixteen texture units");
	}
	if (options.shadow_textures & options.depth_textures)
		throw std::invalid_argument("A native texture unit cannot use both depth-read and shadow-comparison bindings");
	if (options.redirected_textures & (options.shadow_textures | options.depth_textures | options.border_textures | stored_border))
		throw std::invalid_argument("Redirected depth requires exclusive depth and stencil bindings");
	if ((msaa_sampling & (options.border_textures | stored_border)) ||
		(options.multisampled_textures & options.sample_raster_textures))
		throw std::invalid_argument("MSAA sampling requires one storage type and its own border handling");
	if (options.border_textures & stored_border)
		throw std::invalid_argument("Stored-border shader sampling cannot use a constant border sampler");
	for (unsigned unit = 0; unit < options.textures.size(); ++unit)
	{
		(void)texture_type(options.textures[unit]);
		if (((options.redirected_textures | msaa_sampling) & (1u << unit)) && options.textures[unit] != vertex_texture_dimension::two)
			throw std::invalid_argument("Redirected depth and multisample sampling require native 2D textures");
	}
	const auto decoded = decode_fragment_program(bytes);
	generated_fragment_program program;
	generator emit(decoded, options, program);
	const auto body = emit.body();
	std::ostringstream out;
	out << fragment_runtime;
	out << sampler_parameters_msl;
	if (program.texture_mask)
	{
		out << texture_coordinates_msl;
		if (program.texture_mask & stored_border)
			out << stored_border_repeat_msl;
		if (program.texture_mask & (options.stored_mixed_filter_textures | options.stored_border_atlas_textures))
			out << stored_border_mixed_filter_msl;
		const auto atlas_units = program.texture_mask & options.stored_border_atlas_textures;
		if (atlas_units || (program.texture_mask & msaa_sampling))
			out << anisotropic_footprint_msl;
		if (atlas_units)
		{
			out << stored_border_atlas_msl;
			bool volume = false, cube = false;
			for (unsigned i = 0; i < options.textures.size(); ++i)
				if (atlas_units & (1u << i))
				{
					volume |= options.textures[i] == vertex_texture_dimension::three;
					cube |= options.textures[i] == vertex_texture_dimension::cube;
				}
			if (volume) out << stored_border_volume_msl;
			if (cube) out << stored_border_cube_msl;
			out << stored_border_anisotropic_msl;
			if (cube) out << stored_border_cube_anisotropic_msl;
		}
		if (program.texture_mask & (options.depth_textures | msaa_sampling))
			out << depth_sampling_msl;
		if (program.texture_mask & (options.redirected_textures | msaa_sampling))
			out << point_sampling_msl;
		if (program.texture_mask & options.redirected_textures)
			out << redirected_depth_sampling_msl;
		if (program.texture_mask & msaa_sampling)
			out << multisample_sampling_msl;
		if (program.texture_mask & options.redirected_textures & msaa_sampling)
			out << multisample_redirected_depth_msl;
		out << texel_processing_msl;
	}
	if (logic_fetch)
		out << logic_rop_msl;
	if (blend_fetch)
		out << blend_rop_msl;
	if (options.output_rop.enabled())
		out << output_rop_msl;
	emit_varyings(out, options.flat_colors, true);
	input_function(out);
	std::string texture_arguments = program.texture_mask ? ", constant rsx_texture_coordinates* texture_state, constant rsx_texel_processing* texel_state" : "";
	std::string texture_entry = program.texture_mask ? ", constant rsx_texture_coordinates* texture_state [[buffer(4)]], constant rsx_texel_processing* texel_state [[buffer(5)]]" : "";
	std::string texture_values = program.texture_mask ? ", texture_state, texel_state" : "";
	for (unsigned i = 0; i < 16; ++i)
	{
		if (!(program.texture_mask & (1u << i)))
			continue;
		const auto bit = 1u << i;
		const bool depth = (options.shadow_textures | options.depth_textures | options.redirected_textures) & bit;
		const bool multisampled = options.multisampled_textures & bit;
		const auto type = multisampled ? (depth ? "depth2d_ms<float>" : "texture2d_ms<float>") :
			(!depth && (options.sample_raster_textures & bit)) ? "texture2d<float, access::read>" :
			texture_type(options.textures[i], depth, options.stored_border_atlas_textures & bit);
		texture_arguments += ", " + type + " ftex" + n(i) + ", sampler fsampler" + n(i);
		texture_entry += ", " + type + " ftex" + n(i) + " [[texture(" + n(i) + ")]], sampler fsampler" + n(i) + " [[sampler(" + n(i) + ")]]";
		texture_values += ", ftex" + n(i) + ", fsampler" + n(i);
		if (options.redirected_textures & bit)
		{
			const auto stencil = std::string(multisampled ? "texture2d_ms<uint> fstencil" : "texture2d<uint, access::read> fstencil") + n(i);
			texture_arguments += ", " + stencil;
			texture_entry += ", " + stencil + " [[texture(" + n(16 + i) + ")]]";
			texture_values += ", fstencil" + n(i);
		}
		if (options.border_textures & (1u << i))
		{
			const auto coverage_type = texture_type(options.textures[i], options.shadow_textures & (1u << i));
			texture_arguments += ", " + coverage_type + " fborder" + n(i);
			texture_entry += ", " + coverage_type + " fborder" + n(i) + " [[texture(" + n(16 + i) + ")]]";
			texture_values += ", fborder" + n(i);
		}
	}
	out << "inline rsx_fragment_registers rsx_fp_evaluate(rsx_fragment_inputs in, constant rsx_fragment_context& ctx,\n"
		<< "    device const float4* constants, float2 point_coord, bool front_facing, constant rsx_sampler_parameters* sampler_state" << texture_arguments << ") {\n"
		<< "    auto r = rsx_fp_initialize(); uint steps = 0;\n"
		<< "    (void)in; (void)ctx; (void)constants; (void)point_coord; (void)front_facing; (void)sampler_state;\n";
	if (program.texture_mask)
		out << "    (void)texel_state;\n";
	out << body << "    return r;\n}\n";
	out << "struct rsx_fragment_outputs {\n    uint sample_mask [[sample_mask]];\n";
	const auto fetch = [&](unsigned i)
	{
		return options.blend_rop.colors[i].enabled || (logic_fetch && std::any_of(options.logic_rop.bits[i].begin(), options.logic_rop.bits[i].end(), [](auto width)
																				  {
																					  return width != 0;
																				  }));
	};
	const bool per_sample = options.raster_sample_count > 1;
	for (unsigned i = 0; i < options.color_count; ++i)
	{
		out << "    float4 color" << i << " [[color(" << i << ")";
		if (fetch(i))
			out << ", raster_order_group(" << i << ")";
		out << "]];\n";
	}
	if (options.depth_mirror.enabled())
		out << "    float4 stored_depth [[color(4), raster_order_group(4)]];\n";
	if (options.depth_export || options.depth_mirror.enabled())
		out << "    float depth [[depth(any)]];\n";
	out << "};\nfragment rsx_fragment_outputs rpcs3_rsx_fragment(rsx_fragment_inputs in [[stage_in]],\n"
		<< "    float2 point_coord [[point_coord]], bool front_facing [[front_facing]],\n"
		<< "    device const float4* constants [[buffer(1)]], constant rsx_fragment_context& ctx [[buffer(2)]], constant rsx_sampler_parameters* sampler_state [[buffer(3)]]" << texture_entry;
	if (per_sample)
		out << ", uint rsx_rop_sample [[sample_id]]";
	for (unsigned i = 0; i < options.color_count; ++i)
		if (fetch(i))
			out << ", float4 rsx_destination" << i << " [[color(" << i << "), raster_order_group(" << i << ")]]";
	if (options.depth_mirror.enabled())
		out << ", float4 rsx_stored_depth [[color(4), raster_order_group(4)]]";
	out << ") {\n"
		<< "    rsx_fragment_outputs out = {};\n";
	if (per_sample)
		out << "    (void)rsx_rop_sample;\n";
	out << "    if (!ctx.predicate) { discard_fragment(); return out; }\n";
	out << "    if (!(ctx.sample_mask & (1u << " << (per_sample ? "rsx_rop_sample" : "0u") << "))) { discard_fragment(); return out; }\n";
	if (options.depth_mirror.enabled())
		out << "    if (ctx.depth_bounds_enabled && !(rsx_stored_depth.r >= ctx.depth_min && rsx_stored_depth.r <= ctx.depth_max)) { discard_fragment(); return out; }\n";
	if (options.output_rop.polygon_stipple)
		out << "    const uint2 stipple = uint2(in.position.xy) & uint2(31u);\n"
			<< "    if (!(ctx.stipple_pattern[stipple.y] & (1u << stipple.x))) { discard_fragment(); return out; }\n";
	out
		<< "    auto r = rsx_fp_evaluate(in, ctx, constants, point_coord, front_facing, sampler_state" << texture_values << ");\n"
		<< "    if (r.fault || r.killed) { discard_fragment(); return out; }\n"
		<< "    float4 primary = rsx_fp_read(r, 0u, " << (options.float32_exports ? "false" : "true") << ");\n";
	constexpr std::array<std::uint32_t, 4> full{0, 2, 3, 4}, half{0, 4, 6, 8};
	for (unsigned i = 0; i < options.color_count; ++i)
	{
		out << "    float4 color" << i << " = rsx_fp_read(r, " << (options.float32_exports ? full[i] : half[i]) << "u, " << (options.float32_exports ? "false" : "true") << ");\n";
		if (options.output_rop.srgb)
			out << "    color" << i << ".rgb = " << (options.float32_exports ? "" : "float3(half3(") << "rsx_rop_srgb(color" << i << ".rgb)" << (options.float32_exports ? "" : "))") << ";\n";
		if (options.output_rop.round_to_8bit)
			out << "    color" << i << " = rsx_rop_round8(color" << i << ", " << (options.float32_exports ? "false" : "true") << ");\n";
	}
	if (options.color_count)
		out << "    primary.a = color0.a;\n";
	else if (options.output_rop.round_to_8bit)
		out << "    primary.a = rsx_rop_round8(float4(primary.a), false).x;\n";
	out << "    if (ctx.alpha_test && !rsx_fp_compare(primary.a, ctx.alpha_reference, ctx.alpha_function)) { discard_fragment(); return out; }\n";
	if (options.output_rop.stochastic_alpha_coverage)
		out << "    if (!rsx_rop_alpha_coverage(primary.a, in.position.xy)) { discard_fragment(); return out; }\n";
	out << "    out.sample_mask = ctx.sample_mask;\n";
	for (unsigned i = 0; i < options.color_count; ++i)
	{
		if (options.blend_rop.colors[i].enabled)
		{
			const auto& blend = options.blend_rop.colors[i];
			out << "    color" << i << " = rsx_blend_color(color" << i << ", rsx_destination" << i << ", ctx.blend_color, uint4("
				<< static_cast<unsigned>(blend.source_rgb) << "," << static_cast<unsigned>(blend.destination_rgb) << ","
				<< static_cast<unsigned>(blend.source_alpha) << "," << static_cast<unsigned>(blend.destination_alpha) << "), uint2("
				<< static_cast<unsigned>(blend.operation_rgb) << "," << static_cast<unsigned>(blend.operation_alpha) << "), "
				<< static_cast<unsigned>(options.blend_rop.remap[i]) << "u, " << (options.blend_rop.normalized[i] ? "true" : "false") << ");\n";
			if (options.output_rop.round_to_8bit)
				out << "    color" << i << " = rsx_rop_round8(color" << i << ", false);\n";
			if (!options.float32_exports)
				out << "    color" << i << " = float4(half4(color" << i << "));\n";
		}
		out << "    out.color" << i << " = ";
		switch (options.color_remap[i])
		{
		case color_export_remap::rgba:
			out << "color" << i;
			break;
		case color_export_remap::blue:
			out << "color" << i << ".bbbb";
			break;
		case color_export_remap::blue_green:
			out << "color" << i << ".bgbg";
			break;
		case color_export_remap::rgb_zero:
			out << "float4(color" << i << ".rgb, 0)";
			break;
		case color_export_remap::rgb_one:
			out << "float4(color" << i << ".rgb, 1)";
			break;
		}
		out << ";\n";
		if (options.blend_rop.colors[i].enabled && options.blend_rop.normalized[i])
		{
			const auto& bits = options.blend_rop.bits[i];
			out << "    out.color" << i << " = rsx_blend_store(out.color" << i << ", uint4("
				<< bits[0] << "," << bits[1] << "," << bits[2] << "," << bits[3] << "));\n";
		}
		if ((options.half_storage_mask & (1u << i)) || (options.blend_rop.colors[i].enabled && options.blend_rop.half_float[i]))
			out << "    out.color" << i << " = float4(half4(out.color" << i << "));\n";
		if (logic_fetch && fetch(i))
		{
			const auto& bits = options.logic_rop.bits[i];
			out << "    out.color" << i << " = rsx_logic_color(out.color" << i << ", rsx_destination" << i << ", " << options.logic_rop.operation
				<< "u, uint4(" << bits[0] << "," << bits[1] << "," << bits[2] << "," << bits[3] << "));\n";
		}
	}
	if (options.depth_export || options.depth_mirror.enabled())
	{
		out << "    float rsx_candidate_depth = rsx_fp_clamp(float4(" << (options.depth_export ? "rsx_fp_read(r, 1u, false).z" : "in.position.z") << "), 0, 1).x;\n";
		if (options.depth_mirror.encoding == depth_mirror_encoding::unorm16)
			out << "    rsx_candidate_depth = rint(rsx_candidate_depth * 65535.0f) / 65535.0f;\n";
		out << "    out.depth = rsx_candidate_depth;\n";
		if (options.depth_mirror.enabled())
			out << "    out.stored_depth = float4(" << (options.depth_mirror.write ? "rsx_candidate_depth" : "rsx_stored_depth.r") << ");\n";
	}
	out << "    return out;\n}\n";
	program.source = out.str();
	return program;
}

} // namespace metal4
