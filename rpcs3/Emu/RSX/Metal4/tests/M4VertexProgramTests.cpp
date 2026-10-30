#include "M4VertexProgram.h"
#include "M4VertexData.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
using words = std::array<std::uint32_t, 4>;

void require(bool condition)
{
	if (!condition) throw std::runtime_error("Metal 4 vertex decoder/code generation check failed");
}

template <typename Function>
void rejects(Function function)
{
	try { function(); }
	catch (const std::invalid_argument&) { return; }
	throw std::runtime_error("Unsupported RSX vertex program was accepted");
}

// Test assembler uses physical word positions. It doesn't use decoder helpers.
void set_source(words& instruction, unsigned source, std::uint32_t packed)
{
	if (source == 0)
	{
		instruction[1] = (instruction[1] & ~255u) | (packed >> 9);
		instruction[2] = (instruction[2] & 0x007fffffu) | ((packed & 511u) << 23);
	}
	else if (source == 1)
	{
		instruction[2] = (instruction[2] & 0xff80003fu) | (packed << 6);
	}
	else
	{
		instruction[2] = (instruction[2] & ~63u) | (packed >> 11);
		instruction[3] = (instruction[3] & 0x001fffffu) | ((packed & 2047u) << 21);
	}
}

words alu(std::uint32_t vector, std::uint32_t scalar)
{
	// VEC -> o[0], SCA -> r[1], full masks, condition TR, final instruction.
	words result{(63u << 15) | (7u << 10) | (1u << 30), (vector << 22) | (scalar << 27),
		0, (15u << 13) | (15u << 17) | (1u << 7) | 1u};
	set_source(result, 0, 2u | (1u << 12) | (2u << 10) | (3u << 8));
	set_source(result, 1, 3u | (1u << 12) | (2u << 10) | (3u << 8));
	set_source(result, 2, 1u | (2u << 2)); // r[2].xxxx
	return result;
}

void decoder_tests()
{
	for (unsigned source = 0; source < 3; ++source)
	{
		for (std::uint32_t packed = 0; packed < 131072; ++packed)
		{
			words instruction{};
			set_source(instruction, source, packed);
			instruction[0] |= 1u << (21 + source);
			const auto decoded = metal4::decode_vertex_instruction(instruction).sources[source];
			require(decoded.type == packed % 4 && decoded.temporary == (packed / 4) % 64);
			require(decoded.swizzle == std::array<std::uint32_t, 4>{(packed / 16384) % 4,
				(packed / 4096) % 4, (packed / 1024) % 4, (packed / 256) % 4});
			require(decoded.negate == (packed >= 65536) && decoded.absolute);
		}
	}
	for (std::uint32_t target = 0; target < 1024; ++target)
	{
		const words instruction{((target / 512) << 23), 0, ((target / 8) % 64), (target % 8) << 29};
		require(metal4::decode_vertex_instruction(instruction).branch_target == target);
	}
	for (std::uint32_t constant = 0; constant < 1024; ++constant)
	{
		for (std::uint32_t input = 0; input < 16; ++input)
		{
			const words instruction{0, (constant << 12) | (input << 8) | (25u << 22) | (18u << 27), 3u << 8, 0};
			const auto decoded = metal4::decode_vertex_instruction(instruction);
			require(decoded.constant == constant && decoded.input == input && decoded.texture == 3 &&
				decoded.vector_opcode == 25 && decoded.scalar_opcode == 18);
		}
	}
}

void data_tests()
{
	using namespace metal4;
	vertex_context state;
	for (std::size_t bytes = 0; bytes <= 1025; bytes += 17)
	{
		for (std::size_t offset = 0; offset <= 64; ++offset)
		{
			for (std::uint32_t count = 0; count < 7; ++count)
			{
				state.vertex_count = count;
				bool expected = offset <= bytes && offset % 16 == 0;
				for (std::size_t vertex = 0; vertex < count; ++vertex)
				{
					for (std::size_t attribute = 0; attribute < 16; ++attribute)
					{
						if (offset + vertex * 256 + attribute * 16 + 15 >= bytes) expected = false;
					}
				}
				require(valid_vertex_data(state, bytes, offset, 0, 0) == expected);
			}
		}
	}
	state.vertex_count = 0;
	for (std::size_t bytes = 0; bytes <= 129; ++bytes)
	{
		for (std::size_t offset = 0; offset <= 32; ++offset)
		{
			for (std::uint32_t count = 0; count < 9; ++count)
			{
				state.constant_count = count;
				bool expected = offset <= bytes && offset % 16 == 0;
				for (std::size_t reg = 0; reg < count; ++reg)
				{
					if (offset + reg * 16 + 15 >= bytes) expected = false;
				}
				require(valid_vertex_data(state, 0, 0, bytes, offset) == expected);
			}
		}
	}
	state.constant_count = 0;
	constexpr auto max_size = std::numeric_limits<std::size_t>::max();
	constexpr auto max_vertex = std::numeric_limits<std::uint32_t>::max();
	state.vertex_base = max_vertex;
	state.vertex_count = 1;
	require(valid_vertex_data(state, 256, 0, 0, 0));
	require(valid_vertex_reference(state, max_vertex) && !valid_vertex_reference(state, 0));
	state.vertex_count = 2;
	require(!valid_vertex_data(state, max_size, 0, 0, 0));
	state.vertex_count = 1;
	require(!valid_vertex_data(state, max_size, max_size - 15, 0, 0));
	state.vertex_count = 0;
	state.constant_count = 1;
	require(!valid_vertex_data(state, 0, 0, max_size, max_size - 15));
	state.constant_count = 0;
	for (auto budget : {0u, 65537u, max_vertex})
	{
		state.instruction_budget = budget;
		require(!valid_vertex_data(state, 0, 0, 0, 0));
	}
}

}

int main(int argc, char** argv)
{
	using namespace metal4;
	decoder_tests();
	data_tests();
	std::vector<generated_vertex_program> corpus;
	constexpr std::array vectors{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u,
		12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 25u};
	for (auto op : vectors)
	{
		const std::array<words, 1> program{alu(op, 0)};
		corpus.push_back(generate_vertex_program(program));
	}
	constexpr std::array scalars{1u, 2u, 3u, 4u, 5u, 6u, 7u, 13u, 14u, 15u, 16u};
	for (auto op : scalars)
	{
		const std::array<words, 1> program{alu(1, op)};
		corpus.push_back(generate_vertex_program(program));
	}
	for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two,
		vertex_texture_dimension::three, vertex_texture_dimension::cube})
	{
		vertex_program_options options;
		options.flat_colors = true;
		options.textures.fill(dimension);
		for (std::uint32_t texture = 0; texture < 4; ++texture)
		{
			auto instruction = alu(25, 0);
			instruction[2] = (instruction[2] & ~(3u << 8)) | (texture << 8);
			const std::array<words, 1> program{instruction};
			auto generated = generate_vertex_program(program, options);
			require(generated.texture_mask == (1u << texture) && generated.attribute_mask == 1);
			corpus.push_back(std::move(generated));
		}
	}
	for (auto op : {8u, 9u, 10u, 12u, 17u, 18u})
	{
		auto branch = alu(0, op);
		branch[3] &= ~1u;
		// Immediate flow operands overlay source 2; target instruction 1.
		branch[2] &= ~63u;
		branch[3] = (branch[3] & ~(7u << 29)) | (1u << 29);
		const std::array program{branch, alu(1, 0)};
		corpus.push_back(generate_vertex_program(program));
	}
	{
		auto instruction = alu(2, 1);
		instruction[0] |= (1u << 27) | (1u << 13) | (1u << 26);
		instruction[3] |= 2u;
		const std::array<words, 1> program{instruction};
		auto generated = generate_vertex_program(program);
		require(generated.indexed_constants && generated.attribute_mask == 65535);
		corpus.push_back(std::move(generated));
	}
	rejects([] { (void)generate_vertex_program({}); });
	for (auto op : {23u, 24u, 26u, 27u, 28u, 29u, 30u, 31u})
	{
		rejects([&] { const std::array<words, 1> program{alu(op, 0)}; (void)generate_vertex_program(program); });
	}
	for (auto op : {11u, 19u, 20u, 21u, 31u})
	{
		rejects([&] { const std::array<words, 1> program{alu(0, op)}; (void)generate_vertex_program(program); });
	}
	for (unsigned bit : {14u, 29u})
	{
		rejects([&] { auto w = alu(1, 0); w[0] |= 1u << bit; const std::array<words, 1> p{w}; (void)generate_vertex_program(p); });
	}
	rejects([] { auto w = alu(0, 2); set_source(w, 2, 0x1b01); const std::array<words, 1> p{w}; (void)generate_vertex_program(p); });
	rejects([] { auto w = alu(0, 1); w[0] &= ~(1u << 30); const std::array<words, 1> p{w}; (void)generate_vertex_program(p); });
	rejects([] { auto w = alu(1, 0); set_source(w, 0, 0); const std::array<words, 1> p{w}; (void)generate_vertex_program(p); });
	rejects([] { auto w = alu(0, 9); w[3] |= 1u << 29; const std::array<words, 1> p{w}; (void)generate_vertex_program(p); });
	{
		const std::array program{alu(1, 0), words{0, 0xffffffff, 0, 0}};
		const std::array<std::uint8_t, 2> live{1, 0};
		vertex_program_options options;
		options.live_instructions = live;
		corpus.push_back(generate_vertex_program(program, options));
		options.entry = 1;
		rejects([&] { (void)generate_vertex_program(program, options); });
		options.entry = 0;
		options.live_instructions = std::span(live).first(1);
		rejects([&] { (void)generate_vertex_program(program, options); });
	}
	for (const auto& program : corpus)
	{
		require(program.source.find("[[point_size]]") == std::string::npos);
		require(program.source.find("sampler_state [[buffer(3)]]") != std::string::npos);
		if (program.texture_mask)
		{
			require(program.source.find(".min_lod") != std::string::npos);
			require(program.source.find(".max_lod") != std::string::npos);
			require(program.source.find(".bias") != std::string::npos);
		}
	}
	{
		const std::array<words,1> program{alu(1,0)};vertex_program_options options;options.flat_result_inputs=true;
		rejects([&] { (void)generate_vertex_program(program,options); });options.flat_colors=true;
		for(bool packed:{false,true})
		{
			options.packed_inputs=packed;const auto native=generate_vertex_program(program,options);
			const auto vertex=native.source.substr(native.source.find("vertex rsx_varyings rpcs3_rsx_vertex"));
			require(vertex.find("rsx_evaluate(attributes, constants, ctx, reference.flat_vertex")==std::string::npos);
			require(vertex.find("flat_results[low]")!=std::string::npos && native.source.find("kernel void rpcs3_rsx_flat_colors")!=std::string::npos);
			require(native.source.find("rsx_evaluate(attributes, constants, ctx, flat_ids[id]")!=std::string::npos);
		}
	}
	{
		vertex_program_options options;
		options.point_output = true;
		const std::array<words, 1> program{alu(1, 0)};
		auto generated = generate_vertex_program(program, options);
		require(generated.source.find("[[point_size]]") != std::string::npos);
		corpus.push_back(std::move(generated));
	}
	if (argc == 2)
	{
		const std::filesystem::path output{argv[1]};
		std::filesystem::create_directories(output);
		for (std::size_t i = 0; i < corpus.size(); ++i)
		{
			std::ofstream file(output / (std::to_string(i) + ".metal"));
			file.exceptions(std::ios::failbit | std::ios::badbit);
			file << corpus[i].source;
		}
	}
	std::cout << "Metal 4 vertex: 393216 operand encodings, 1024 branches, 16384 selectors and "
		<< corpus.size() << " shader variants plus buffer bounds and rejection checks passed\n";
}
