#include "M4FragmentProgram.h"

#include <iostream>
#include <stdexcept>

namespace
{
using words = std::array<std::uint32_t, 4>;

void require(bool condition)
{
	if (!condition) throw std::runtime_error("Metal 4 fragment decoder check failed");
}

template <typename Function>
void rejects(Function function)
{
	try { function(); }
	catch (const std::invalid_argument&) { return; }
	throw std::runtime_error("Truncated or unbounded RSX fragment program was accepted");
}

void append(std::vector<std::byte>& bytes, words instruction)
{
	for (auto value : instruction)
	{
		for (unsigned shift : {8u, 0u, 24u, 16u}) bytes.push_back(static_cast<std::byte>(value >> shift));
	}
}

}

int main()
{
	using namespace metal4;
	// Physical byte fixture: one, negative zero, positive infinity and a NaN
	// payload. Literal transport must preserve every bit without float casts.
	const std::array<std::byte, 16> literal{
		std::byte{0}, std::byte{0}, std::byte{0x3f}, std::byte{0x80},
		std::byte{0}, std::byte{0}, std::byte{0x80}, std::byte{0},
		std::byte{0}, std::byte{0}, std::byte{0x7f}, std::byte{0x80},
		std::byte{0x12}, std::byte{0x34}, std::byte{0x7f}, std::byte{0xc0}};
	const words literal_words{0x3f800000, 0x80000000, 0x7f800000, 0x7fc01234};
	require(decode_fragment_block(literal) == literal_words);
	for (std::uint32_t value = 0; value < 262144; ++value)
	{
		for (unsigned source = 0; source < 3; ++source)
		{
			words instruction{};
			instruction[source + 1] = value | (1u << (source == 0 ? 29 : 18));
			const auto decoded = decode_fragment_instruction(instruction).sources[source];
			require(decoded.type == value % 4 && decoded.temporary == (value / 4) % 64);
			require(decoded.half == ((value / 256) % 2 != 0) && decoded.negate == (value >= 131072));
			require(decoded.swizzle == std::array<std::uint32_t, 4>{(value / 512) % 4,
				(value / 2048) % 4, (value / 8192) % 4, (value / 32768) % 4});
			require(decoded.absolute);
		}
	}
	for (std::uint32_t op = 0; op < 128; ++op)
	{
		const words instruction{(op % 64) << 24, 0, (op / 64) << 31, 0};
		require(decode_fragment_instruction(instruction).opcode == op);
	}
	for (std::uint32_t mask = 0; mask < 16; ++mask)
	{
		for (std::uint32_t reg = 0; reg < 64; ++reg)
		{
			const words instruction{(mask << 9) | (reg << 1) | 0xc0c00181u,
				(5u << 18) | (0xe4u << 21) | (1u << 30),
				(1u << 19) | (2u << 22) | (3u << 25) | (7u << 28),
				(2047u << 19) | (1u << 30) | (1u << 31)};
			const auto decoded = decode_fragment_instruction(instruction);
			require(decoded.destination == reg && decoded.write_mask == mask && decoded.precision == 3);
			require(decoded.half_destination && decoded.no_destination && decoded.set_condition && decoded.saturate && decoded.end);
			require(decoded.condition_mode == 5 && decoded.condition_read == 0 && decoded.condition_write == 1);
			require(decoded.condition_swizzle == std::array<std::uint32_t, 4>{0, 1, 2, 3});
			require(decoded.sources[0].precision == 1 && decoded.sources[1].precision == 2 && decoded.sources[2].precision == 3);
			require(decoded.address == 2047 && decoded.indexed_input && decoded.perspective && decoded.scale == 7);
		}
	}
	// A literal tag on an unused NOP operand still consumes data. Treating this
	// block as code would change instruction addresses and termination.
	for (unsigned source = 0; source < 3; ++source)
	{
		std::vector<std::byte> bytes;
		words nop{};
		nop[source + 1] = 2;
		append(bytes, nop);
		bytes.insert(bytes.end(), literal.begin(), literal.end());
		append(bytes, {0x01000001, 0, 0, 0}); // final MOV
		bytes.push_back(std::byte{0xff}); // trailing memory is outside program
		const auto program = decode_fragment_program(bytes);
		require(program.instructions.size() == 2 && program.byte_length == 48);
		require(program.instructions[0].literal == literal_words);
		require(program.instructions[1].byte_offset == 32 && !program.instructions[1].literal);
		const auto hash = fragment_program_fingerprint(bytes);
		auto changed = bytes;
		std::fill(changed.begin() + 16, changed.begin() + 32, std::byte{0xff});
		changed.back() = std::byte{0};
		require(fragment_program_fingerprint(changed) == hash);
		changed[33] ^= std::byte{2};
		require(fragment_program_fingerprint(changed) != hash);
		// Every prefix shorter than the full program must fail without reading
		// beyond its explicitly provided length, including the final opcode.
		for (std::size_t length = 0; length < 48; ++length)
		{
			rejects([&] { (void)decode_fragment_program(std::span(bytes).first(length)); });
			rejects([&] { (void)fragment_program_fingerprint(std::span(bytes).first(length)); });
		}
		rejects([&] { (void)decode_fragment_program(bytes, 1); });
		rejects([&] { (void)decode_fragment_program(bytes, 0); });
		// An unaligned span must decode identically.
		bytes.insert(bytes.begin(), std::byte{0x77});
		const auto unaligned = decode_fragment_program(std::span(bytes).subspan(1));
		require(unaligned.instructions[0].literal == literal_words && unaligned.byte_length == 48);
	}
	for (std::uint32_t opcode : {0x3du, 0x3eu, 0x40u, 0x41u, 0x42u, 0x43u, 0x44u, 0x45u})
	{
		std::vector<std::byte> bytes;
		append(bytes, {(opcode % 64) << 24 | 1u, 2, (opcode / 64) << 31 | 2u, 2});
		const auto program = decode_fragment_program(bytes);
		require(program.byte_length == 16 && !program.instructions[0].literal && program.instructions[0].opcode == opcode);
		(void)fragment_program_fingerprint(bytes);
	}
	// End flag doesn't cancel the literal belonging to that instruction.
	std::vector<std::byte> final_literal;
	append(final_literal, {1, 2, 0, 0});
	rejects([&] { (void)decode_fragment_program(final_literal); });
	rejects([&] { (void)fragment_program_fingerprint(final_literal); });
	final_literal.insert(final_literal.end(), literal.begin(), literal.end());
	require(decode_fragment_program(final_literal).byte_length == 32);
	std::cout << "Metal 4 fragment decoder: 786432 operand encodings, 128 opcodes, 1024 destination cases and stream bounds plus literal-independent fingerprint bounds passed\n";
}
