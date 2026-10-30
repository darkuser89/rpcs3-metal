#pragma once

#include "M4ShaderProgramSnapshot.h"
#include <bitset>

namespace metal4::test
{
struct texture_state
{
	std::uint32_t texture_dimensions = 0;
	std::uint16_t shadow_textures = 0, redirected_textures = 0, multisampled_textures = 0;
};
struct vertex_program
{
	std::vector<std::uint32_t> data;
	std::bitset<rsx_vertex_instruction_limit> instruction_mask;
	std::uint32_t entry = 0, base_address = 0, ctrl = 0, output_mask = 0;
	test::texture_state texture_state;
};
struct fragment_program
{
	const void* pointer = nullptr;
	std::uint32_t ucode_length = 0, ctrl = 0, texcoord_control_mask = 0, mrt_buffers_count = 1;
	bool valid = true, two_sided_lighting = false;
	test::texture_state texture_state;
	[[nodiscard]] const void* get_data() const noexcept { return pointer; }
};
} // namespace metal4::test
