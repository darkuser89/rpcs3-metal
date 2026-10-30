#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace metal4
{

struct fragment_source_operand
{
	std::uint32_t type;
	std::uint32_t temporary;
	bool half;
	std::array<std::uint32_t, 4> swizzle;
	bool negate;
	bool absolute;
	std::uint32_t precision;
};

struct fragment_instruction
{
	std::array<std::uint32_t, 4> words;
	std::size_t byte_offset = 0;
	std::array<fragment_source_operand, 3> sources;
	std::uint32_t opcode;
	std::uint32_t destination;
	std::uint32_t write_mask;
	std::uint32_t attribute;
	std::uint32_t texture;
	std::uint32_t precision;
	std::uint32_t scale;
	std::uint32_t condition_mode;
	std::array<std::uint32_t, 4> condition_swizzle;
	std::uint32_t condition_read;
	std::uint32_t condition_write;
	std::uint32_t address;
	bool indexed_input;
	bool perspective;
	bool half_destination;
	bool no_destination;
	bool set_condition;
	bool saturate;
	bool expand_texture;
	bool end;
	// Bit-exact decoded literal from the following 16-byte stream block.
	// The MSL generator must use mutable constant bindings, not specialize
	// shaders on these values: games can update them without changing code.
	std::optional<std::array<std::uint32_t, 4>> literal;
};

[[nodiscard]] fragment_instruction decode_fragment_instruction(std::array<std::uint32_t, 4> words);
// Raw bytes use the RSX FP half-word byte shuffle, independently of host
// endianness/alignment. This differs from decoding ordinary big-endian u32.
[[nodiscard]] std::array<std::uint32_t, 4> decode_fragment_block(std::span<const std::byte, 16> bytes);

struct decoded_fragment_program
{
	std::vector<fragment_instruction> instructions;
	std::size_t byte_length = 0;
};

// Keeps NOPs and flow-control words, including their physical byte addresses.
// A literal tag in ANY source consumes the next block for non-flow ops,
// even when that operand isn't used (including NOP). Flow operands overlay
// those fields and must not be treated as literal tags. Unknown opcodes are
// retained for explicit rejection by the generator, never replaced with NOP.
// Stops at the end flag and ignores trailing memory. Missing end, truncated
// instructions/literals and the instruction limit fail without an OOB read.
[[nodiscard]] decoded_fragment_program decode_fragment_program(std::span<const std::byte> bytes,
	std::size_t max_instructions = 4096);

// Allocation-free candidate index; uses the same validated layout walk as the
// decoder. Excludes literals/trailing bytes. Hash equality never proves identity.
[[nodiscard]] std::uint64_t fragment_program_fingerprint(std::span<const std::byte> bytes);

// Validate exact words and physical layout against the compiled image without
// rebuilding decoded instructions. Returns the current literal register count.
[[nodiscard]] std::size_t validate_fragment_constants(const decoded_fragment_program& compiled, std::span<const std::byte> bytes);

// Refresh mutable literals while checking that instruction words and their
// physical layout still match the compiled image. Code changes require a new
// shader; replacing constants never requires specializing shader source.
[[nodiscard]] std::vector<std::array<std::uint32_t, 4>> refresh_fragment_constants(
	const decoded_fragment_program& compiled, std::span<const std::byte> bytes);

}
