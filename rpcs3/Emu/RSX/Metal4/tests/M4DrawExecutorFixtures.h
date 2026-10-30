#pragma once

#include "M4DrawExecutor.h"
#include "M4ShaderSnapshotFixtures.h"
#include <bit>

namespace metal4::test::executor_fixture
{
struct state
{
	std::vector<std::uint32_t> registers = std::vector<std::uint32_t>(0x2000 / 4);
};
inline state registers()
{
	state s;
	auto& m = s.registers;
	m[0x314 / 4] = 0x10001;
	m[0x320 / 4] = 0x80068006;
	m[0x324 / 4] = 0x01010101;
	m[0x330 / 4] = m[0x350 / 4] = 0x207;
	for (unsigned a : {0x33cu, 0x340u, 0x344u, 0x35cu, 0x360u, 0x364u})
		m[a / 4] = 0x1e00;
	m[0x32c / 4] = m[0x338 / 4] = m[0x34c / 4] = m[0x358 / 4] = 255;
	m[0xa6c / 4] = 0x201;
	m[0x378 / 4] = 0x1503;
	m[0x1834 / 4] = 0x900;
	m[0x1830 / 4] = 0x405;
	m[0x8c0 / 4] = m[0x8c4 / 4] = 4u << 16;
	return s;
}
inline void append(std::vector<std::byte>& bytes, unsigned word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		bytes.push_back(std::byte(word >> shift));
}
inline std::vector<std::byte> program(bool texture, std::array<float, 4> color = {64.f / 255, 128.f / 255, 192.f / 255, 1})
{
	std::vector<std::byte> bytes;
	for (unsigned word : {(1u << 24) | (texture ? 2u : 0u) | (15u << 9) | (texture ? 0u : 1u), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
		append(bytes, word);
	for (float value : texture ? std::array<float, 4>{.125f, .125f, 0, 1} : color)
		append(bytes, std::bit_cast<unsigned>(value));
	if (texture)
		for (unsigned word : {(23u << 24) | (15u << 9) | 1u, (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 4u, 0xe4u << 9, 0xe4u << 9})
			append(bytes, word);
	return bytes;
}
inline test::vertex_program vertex_program(bool texture = false)
{
	constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	test::vertex_program v;
	v.data.assign(4, 0xffffffffu);
	for (unsigned word : {(63u << 15) | (7u << 10) | (1u << 30), ((texture ? 25u : 1u) << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u})
		v.data.push_back(word);
	v.base_address = 137;
	v.entry = 138;
	v.instruction_mask.set(1);
	return v;
}
inline void texture_registers(state& state, unsigned offset)
{
	auto& m = state.registers;
	const auto b = 0x1a00 / 4;
	m[b] = offset;
	m[b + 1] = (1u << 16) | (0xa5u << 8) | (2u << 4) | 8 | 2;
	m[b + 2] = (7u << 28) | 0x030303;
	m[b + 3] = (1u << 31) | (4095u << 7);
	m[b + 4] = 0xaae4;
	m[b + 5] = (1u << 24) | (3u << 16);
	m[b + 6] = (4u << 16) | 4;
	m[0x1840 / 4] = (1u << 20) | 16;
}
inline void set_vertex_texture_registers(state& state, unsigned offset)
{
	auto& words = state.registers;
	const unsigned base = 0x900 / 4;
	words[base] = offset | 0x80000000u;
	words[base + 1] = (1u << 16) | (0xbbu << 8) | (2u << 4) | 1u;
	words[base + 2] = 0xfff00303u;
	words[base + 3] = 0x80000074u;
	words[base + 4] = 64;
	words[base + 5] = 0xffffffffu;
	words[base + 6] = (4u << 16) | 1u;
	words[base + 7] = 0xff000000u;
}
inline rsx_draw_snapshot snapshot(state& state, bool texture, std::uint32_t mask, bool vertex_texture = false, unsigned vertex_mask = 0,
								  std::array<float, 4> color = {64.f / 255, 128.f / 255, 192.f / 255, 1})
{
	auto vertex = vertex_program(vertex_texture);
	auto bytes = program(texture, color);
	test::fragment_program fragment;
	fragment.pointer = bytes.data();
	fragment.ucode_length = bytes.size();
	fragment.ctrl = rsx_fragment_float32;
	const auto result = rsx_draw_snapshot::capture(state, vertex, fragment, mask, 1 << 20, vertex_mask);
	std::fill(state.registers.begin(), state.registers.end(), 0xffffffffu);
	std::fill(vertex.data.begin(), vertex.data.end(), 0xffffffffu);
	std::fill(bytes.begin(), bytes.end(), std::byte{0xff});
	return result;
}
} // namespace metal4::test::executor_fixture
