#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace metal4
{

enum class vertex_texture_dimension { one, two, three, cube };

struct vertex_source_operand
{
	std::uint32_t type;
	std::uint32_t temporary;
	std::array<std::uint32_t, 4> swizzle;
	bool negate;
	bool absolute;
};

struct vertex_instruction
{
	std::array<std::uint32_t, 4> words;
	std::array<vertex_source_operand, 3> sources;
	std::uint32_t vector_opcode;
	std::uint32_t scalar_opcode;
	std::uint32_t input;
	std::uint32_t constant;
	std::uint32_t texture;
	std::uint32_t branch_target;
};

[[nodiscard]] vertex_instruction decode_vertex_instruction(std::array<std::uint32_t, 4> words);

struct vertex_program_options
{
	// Instructions use analyser-rebased immediate branch addresses, as in VK.
	// entry is a local instruction index; base_address is used by dynamic BRA.
	std::uint32_t entry = 0;
	std::uint32_t base_address = 0;
	bool flat_colors = false;
	// Metal4 rejects [[point_size]] exports for line/triangle pipeline classes.
	// Enable only for point draws and include this option in the shader key.
	bool point_output = false;
	// Read checked immutable packed streams directly in the vertex stage.
	bool packed_inputs = false;
	// A native compute pass supplies one color/fault record per needed flat ID.
	bool flat_result_inputs = false;
	std::uint32_t border_textures = 0;
	// 'one' binds a height-one MTLTextureType2D to preserve explicit mip LOD.
	std::array<vertex_texture_dimension, 4> textures{
		vertex_texture_dimension::two, vertex_texture_dimension::two,
		vertex_texture_dimension::two, vertex_texture_dimension::two};
	// Optional analyser mask. Dead instructions don't have to contain valid code.
	std::span<const std::uint8_t> live_instructions;
};

// CPU/MSL ABI, buffer(2). Default attributes in buffer(0) are predecoded
// float4[16] records. packed_inputs instead binds persistent bytes in buffer(0),
// transient bytes in buffer(4), checked fetch metadata in buffer(5).
// Constants in buffer(1) are a full register file,
// not the Vulkan compacted constant-ID array. buffer(30) is the geometry
// vertex_reference array. Runtime faults clip the vertex; this vertex function
// has no device writes and doesn't publish GPU fault counters.
struct alignas(16) vertex_context
{
	std::array<float, 16> scale_offset{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
	std::uint32_t vertex_base = 0;
	std::uint32_t vertex_count = 0;
	std::uint32_t constant_count = 0;
	std::uint32_t branch_bits = 0;
	float point_size = 1;
	float z_near = 0;
	float z_far = 1;
	std::uint32_t output_mask = 0;
	std::uint32_t two_sided = 0;
	std::uint32_t instruction_budget = 65536;
	std::uint32_t geometry_count = 0;
	std::uint32_t predicate = 1;
	std::array<float, 6> clip_factors{};
	std::uint32_t clip_enable = 0;
	std::uint32_t flat_count = 0;
};
static_assert(sizeof(vertex_context) == 144 && offsetof(vertex_context, vertex_base) == 64 &&
	offsetof(vertex_context, point_size) == 80 && offsetof(vertex_context, clip_factors) == 112 && offsetof(vertex_context,flat_count)==140);

enum vertex_fault : std::uint32_t
{
	vertex_attribute_fault = 1,
	vertex_constant_fault = 2,
	vertex_flow_fault = 4,
	vertex_stack_fault = 8,
	vertex_budget_fault = 16,
	vertex_address_fault = 32,
	vertex_geometry_fault = 64,
};

struct generated_vertex_program
{
	std::string source;
	std::uint32_t attribute_mask = 0;
	std::uint32_t texture_mask = 0;
	std::uint32_t output_register_mask = 0;
	bool indexed_constants = false;
};

// Emits native MSL 4 directly from RSX vertex words. Unsupported instructions
// and condition-update modes fail on the CPU; they are never compiled as NOPs.
// This stage doesn't decode guest vertex formats, implement instanced constant
// indirection, or establish full RSX floating-point/hardware parity.
[[nodiscard]] generated_vertex_program generate_vertex_program(
	std::span<const std::array<std::uint32_t, 4>> instructions,
	const vertex_program_options& options = {});

}
