#include "M4FragmentProgram.h"

#include <stdexcept>

namespace metal4
{
namespace
{

std::uint32_t bits(std::uint32_t value, unsigned offset, unsigned width)
{
	return (value >> offset) & ((1u << width) - 1);
}

bool flow_opcode(std::uint32_t opcode)
{
	return opcode == 0x3d || opcode == 0x3e || (opcode >= 0x40 && opcode <= 0x45);
}

}

fragment_instruction decode_fragment_instruction(std::array<std::uint32_t, 4> words)
{
	fragment_instruction result{};
	result.words = words;
	const auto d0 = words[0], s0 = words[1], s1 = words[2], s2 = words[3];
	result.opcode = bits(d0, 24, 6) | (bits(s1, 31, 1) << 6);
	result.destination = bits(d0, 1, 6);
	result.write_mask = bits(d0, 9, 4);
	result.attribute = bits(d0, 13, 4);
	result.texture = bits(d0, 17, 4);
	result.precision = bits(d0, 22, 2);
	result.scale = bits(s1, 28, 3);
	result.condition_mode = bits(s0, 18, 3);
	result.condition_swizzle = {bits(s0, 21, 2), bits(s0, 23, 2), bits(s0, 25, 2), bits(s0, 27, 2)};
	result.condition_read = bits(s0, 31, 1);
	result.condition_write = bits(s0, 30, 1);
	result.address = bits(s2, 19, 11);
	result.indexed_input = bits(s2, 30, 1) != 0;
	result.perspective = bits(s2, 31, 1) != 0;
	result.half_destination = bits(d0, 7, 1) != 0;
	result.no_destination = bits(d0, 30, 1) != 0;
	result.set_condition = bits(d0, 8, 1) != 0;
	result.saturate = bits(d0, 31, 1) != 0;
	result.expand_texture = bits(d0, 21, 1) != 0;
	result.end = bits(d0, 0, 1) != 0;
	for (unsigned i = 0; i < 3; ++i)
	{
		const auto s = words[i + 1];
		result.sources[i] = {bits(s, 0, 2), bits(s, 2, 6), bits(s, 8, 1) != 0,
			{bits(s, 9, 2), bits(s, 11, 2), bits(s, 13, 2), bits(s, 15, 2)}, bits(s, 17, 1) != 0,
			bits(s, i == 0 ? 29 : 18, 1) != 0, bits(s1, 19 + i * 3, 3)};
	}
	return result;
}

std::array<std::uint32_t, 4> decode_fragment_block(std::span<const std::byte, 16> bytes)
{
	std::array<std::uint32_t, 4> result;
	for (std::size_t i = 0; i < 4; ++i)
	{
		result[i] = std::to_integer<std::uint32_t>(bytes[i * 4 + 1]) |
			(std::to_integer<std::uint32_t>(bytes[i * 4]) << 8) |
			(std::to_integer<std::uint32_t>(bytes[i * 4 + 3]) << 16) |
			(std::to_integer<std::uint32_t>(bytes[i * 4 + 2]) << 24);
	}
	return result;
}

template <typename Visit>
static std::size_t visit_fragment_program(std::span<const std::byte> bytes, std::size_t max_instructions, Visit visit)
{
	std::size_t pc = 0;
	for (std::size_t count = 0; count < max_instructions; ++count)
	{
		if (pc > bytes.size() || bytes.size() - pc < 16)
		{
			throw std::invalid_argument("RSX fragment program has no complete terminating instruction");
		}
		auto instruction = decode_fragment_instruction(decode_fragment_block(bytes.subspan(pc, 16).first<16>()));
		instruction.byte_offset = pc;
		pc += 16;
		if (!flow_opcode(instruction.opcode) &&
			(instruction.sources[0].type == 2 || instruction.sources[1].type == 2 || instruction.sources[2].type == 2))
		{
			if (bytes.size() - pc < 16)
			{
				throw std::invalid_argument("RSX fragment instruction has a truncated literal constant");
			}
			instruction.literal = decode_fragment_block(bytes.subspan(pc, 16).first<16>());
			pc += 16;
		}
		visit(instruction);
		if (instruction.end) return pc;
	}
	throw std::invalid_argument("RSX fragment program exceeds the decoder instruction limit");
}

decoded_fragment_program decode_fragment_program(std::span<const std::byte> bytes, std::size_t max_instructions)
{
	decoded_fragment_program result;
	result.byte_length = visit_fragment_program(bytes, max_instructions, [&](const auto& instruction) { result.instructions.push_back(instruction); });
	return result;
}

std::uint64_t fragment_program_fingerprint(std::span<const std::byte> bytes)
{
	std::uint64_t hash = 14695981039346656037ull;
	visit_fragment_program(bytes, 4096, [&](const auto& instruction)
	{
		for (auto word : instruction.words) hash = (hash ^ word) * 1099511628211ull;
	});
	return hash;
}

std::size_t validate_fragment_constants(const decoded_fragment_program& compiled, std::span<const std::byte> bytes)
{
	if (compiled.instructions.empty() || bytes.size() < compiled.byte_length)
		throw std::invalid_argument("RSX fragment program is empty or truncated");
	std::size_t pc = 0, literals = 0;
	for (std::size_t i = 0; i < compiled.instructions.size(); ++i)
	{
		const auto& original = compiled.instructions[i];
		if (pc > bytes.size() || bytes.size() - pc < 16 || original.byte_offset != pc)
			throw std::invalid_argument("RSX fragment program layout changed and requires recompilation");
		const auto words = decode_fragment_block(bytes.subspan(pc, 16).first<16>());
		const auto opcode = bits(words[0], 24, 6) | (bits(words[2], 31, 1) << 6);
		const bool literal = !flow_opcode(opcode) && ((words[1] & 3) == 2 || (words[2] & 3) == 2 || (words[3] & 3) == 2);
		if (words != original.words || bool(words[0] & 1) != (i + 1 == compiled.instructions.size()) || literal != original.literal.has_value())
			throw std::invalid_argument("RSX fragment instruction changed and requires recompilation");
		pc += 16;
		if (literal)
		{
			if (bytes.size() - pc < 16) throw std::invalid_argument("RSX fragment literal is truncated");
			pc += 16;
			++literals;
		}
	}
	if (pc != compiled.byte_length) throw std::invalid_argument("RSX fragment program extent changed");
	return literals;
}

std::vector<std::array<std::uint32_t, 4>> refresh_fragment_constants(
	const decoded_fragment_program& compiled, std::span<const std::byte> bytes)
{
	std::vector<std::array<std::uint32_t, 4>> literals;
	literals.reserve(validate_fragment_constants(compiled, bytes));
	for (const auto& instruction : compiled.instructions)
		if (instruction.literal)
			literals.push_back(decode_fragment_block(bytes.subspan(instruction.byte_offset + 16, 16).first<16>()));
	return literals;
}

}
