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
	for (auto op : {26u, 27u, 28u, 29u, 30u, 31u})
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
	for (auto [vector, scalar, vector_result] : {std::array{1u, 0u, 1u}, std::array{0u, 1u, 0u},
		std::array{1u, 1u, 1u}, std::array{1u, 1u, 0u}})
		for (unsigned bank : {0u, 1u})
		{
			auto w = alu(vector, scalar);
			w[0] |= (1u << 14) | (bank << 25) | (vector_result << 29);
			const auto generated = generate_vertex_program(std::span{&w, 1});
			const auto condition_write = "r.condition[" + std::to_string(bank) + "].x = value.x;";
			const auto first_write = generated.source.find(condition_write);
			require(first_write != std::string::npos && generated.source.find(condition_write, first_write + 1) == std::string::npos);
			corpus.push_back(generated);
		}
	for (auto [vector, scalar] : {std::array{0u, 0u}, std::array{13u, 0u}, std::array{0u, 9u}, std::array{1u, 9u}})
		for (unsigned vector_result : {0u, 1u})
		{
			auto w = alu(vector, scalar);
			w[0] |= (1u << 14) | (vector_result << 29);
			rejects([&] { (void)generate_vertex_program(std::span{&w, 1}); });
		}
	{
		auto w = alu(1, 1);
		w[0] |= (1u << 14) | (1u << 29);
		auto implicit = w;
		implicit[3] |= 63u << 7;
		rejects([&] { (void)generate_vertex_program(std::span{&implicit, 1}); });
		auto overlap = w;
		overlap[0] = (overlap[0] & ~(63u << 15)) | (1u << 15);
		rejects([&] { (void)generate_vertex_program(std::span{&overlap, 1}); });
		w[0] = (w[0] & ~(63u << 15)) | (2u << 15);
		w[3] = (w[3] & ~(15u << 13)) | (4u << 13);
		set_source(w, 2, 0x1b09); // MOV reads r[2].y, which VEC writes.
		rejects([&] { (void)generate_vertex_program(std::span{&w, 1}); });
		w[3] &= ~(4u << 17); // SCA no longer consumes y.
		corpus.push_back(generate_vertex_program(std::span{&w, 1}));
	}
	for (unsigned opcode : {13u, 23u, 24u})
	for (unsigned selected : {0u, 1u, 2u})
		for (unsigned bank : {0u, 1u})
			for (unsigned address : {0u, 1u})
			{
				auto w = alu(opcode, unsigned(selected != 0));
				w[0] = (w[0] & ~(63u << 15)) | (1u << 14) | (address << 15) |
					(bank << 25) | (unsigned(selected != 2) << 29);
				w[3] = (w[3] & ~(31u << 2)) | (31u << 2);
				const auto shader = generate_vertex_program(std::span{&w, 1});
				const auto write = "if (!address_fault) r.condition[" + std::to_string(bank) + "][0] = float(value[0]);";
				require((shader.source.find(write) != std::string::npos) == (selected != 2));
				corpus.push_back(shader);
				if (selected != 2)
				{
					auto saturated = w;
					saturated[0] |= 1u << 26;
					rejects([&] { (void)generate_vertex_program(std::span{&saturated, 1}); });
				}
				if (selected)
				{
					set_source(w, 2, 0x1b03);
					w[0] |= address << 24;
					w[3] |= 1u << 1;
					rejects([&] { (void)generate_vertex_program(std::span{&w, 1}); });
				}
			}
	rejects([] { auto w = alu(0, 2); set_source(w, 2, 0x1b01); const std::array<words, 1> p{w}; (void)generate_vertex_program(p); });
	{
		unsigned accepted = 0, rejected = 0;
		for (unsigned vector : {0u, 1u})
		for (unsigned swizzle = 0; swizzle < 256; ++swizzle)
			for (unsigned mask = 1; mask < 16; ++mask)
			{
				auto w = alu(vector, 2);
				w[3] = (w[3] & ~(15u << 17)) | (mask << 17);
				set_source(w, 2, (swizzle << 8) | 1u);
				const auto source_x = swizzle >> 6;
				bool matches = true;
				for (unsigned lane = 1; lane < 4; ++lane)
					if ((mask & (8u >> lane)) && ((swizzle >> (6 - 2 * lane)) & 3) != source_x)
						matches = false;
				try { (void)generate_vertex_program(std::span{&w, 1}); require(matches); ++accepted; }
				catch (const std::invalid_argument&) { require(!matches); ++rejected; }
			}
		require(accepted == 1488 && rejected == 6192);
		auto paired = alu(1, 2);
		paired[3] = (paired[3] & ~(15u << 17)) | (8u << 17);
		set_source(paired, 2, 0x1b01);
		corpus.push_back(generate_vertex_program(std::span{&paired, 1}));
		const auto set_vector_destination = [](words w, unsigned reg, unsigned mask)
		{
			w[0] = (w[0] & ~(63u << 15)) | (reg << 15);
			w[3] = (w[3] & ~(15u << 13)) | (mask << 13);
			return w;
		};
		for (unsigned reg : {0u, 1u})
		{
			const auto hazard = set_vector_destination(paired, reg, 8);
			rejects([&] { (void)generate_vertex_program(std::span{&hazard, 1}); });
			const auto disjoint = set_vector_destination(paired, reg, 4);
			corpus.push_back(generate_vertex_program(std::span{&disjoint, 1}));
		}
		auto arl = set_vector_destination(paired, 0, 8);
		arl[1] = (arl[1] & ~(31u << 22)) | (13u << 22);
		set_source(arl, 2, 0x1b03);
		arl[3] |= 1u << 1;
		rejects([&] { (void)generate_vertex_program(std::span{&arl, 1}); });
		arl[0] |= 1u << 24;
		corpus.push_back(generate_vertex_program(std::span{&arl, 1}));
		paired[0] |= 1u << 28;
		corpus.push_back(generate_vertex_program(std::span{&paired, 1}));
		std::cout << "Scalar swizzle write masks: 7680 CPU combinations, paired hazard and disjoint-lane checks passed\n";
	}
	{
		unsigned cases = 0;
		for (unsigned opcode : scalars) for (unsigned temporary = 0; temporary < 63; ++temporary)
			for (unsigned output = 0; output < 16; ++output) for (unsigned mask = 0; mask < 16; ++mask)
			{
				auto w = alu(0, opcode); w[0] &= ~(1u << 30);
				w[3] = (w[3] & ~((63u << 7) | (31u << 2) | (15u << 17))) | (temporary << 7) | (output << 2) | (mask << 17);
				const auto generated = generate_vertex_program(std::span{&w, 1});
				require(generated.output_register_mask == (1u << output));
				++cases;
			}
		require(cases == 177408);
		for (unsigned opcode : scalars)
		{
			auto w = alu(0, opcode); w[0] &= ~(1u << 30);
			corpus.push_back(generate_vertex_program(std::span{&w, 1}));
			w[0] |= 1u << 28;
			require(generate_vertex_program(std::span{&w, 1}).output_register_mask == 65535u);
			corpus.push_back(generate_vertex_program(std::span{&w, 1}));
			w[0] &= ~(1u << 28);w[1] |= 1u << 22;
			// Paired VEC and SCA overlap in r1. An independently indexed
			// VEC source/result timing contract is still required here.
			w[0] = (w[0] & ~(63u << 15)) | (1u << 15);
			rejects([&] { (void)generate_vertex_program(std::span{&w, 1}); });
		}
		std::cout << "Scalar dual destinations: 177408 CPU opcode/temp/output/mask combinations; 63 temporary and 16 output slots; indexed outputs supported, overlapping paired destinations remain refused\n";
	}
	{
		unsigned cases=0,refused=0,admitted=0;
		for(unsigned mode=0;mode<5;++mode)for(unsigned bank=0;bank<2;++bank)for(unsigned lane=0;lane<4;++lane)
		for(unsigned output=0;output<16;++output)for(unsigned mask=0;mask<16;++mask)
		{
			auto w=alu(mode==0?1u:mode>=3?1u:0u,mode==0?0u:1u);
			w[0]=(w[0]&~((63u<<15)|(1u<<30)))|(1u<<28)|(bank<<24)|lane|((mode>=3?4u:mode==0?5u:63u)<<15)|(unsigned(mode==0)<<30);
			w[3]=(w[3]&~((31u<<2)|(63u<<7)|(15u<<13)|(15u<<17)))|(output<<2)|((mode==1?63u:5u)<<7)|(mask<<(mode==0?13:17))|(mode>=3?10u<<13:0u);
			set_source(w,2,0x1b03);require(generate_vertex_program(std::span{&w,1}).output_register_mask==65535u);++cases;
		}
		for(unsigned opcode:{13u,23u,24u})
		for(unsigned bank=0;bank<2;++bank)for(unsigned lane=0;lane<4;++lane)for(unsigned address=0;address<2;++address)
		for(unsigned mask=0;mask<16;++mask)for(bool writes:{false,true})
		{
			auto w=alu(opcode,1);w[0]=(w[0]&~((63u<<15)|(1u<<30)))|(address<<15)|(1u<<28)|(bank<<24)|lane;
			w[3]=(w[3]&~((15u<<13)|(15u<<17)))|(mask<<13)|(writes?15u<<17:0u);
			const bool hazard=writes&&bank==address&&(mask&(8u>>lane));
			if(hazard){rejects([&]{(void)generate_vertex_program(std::span{&w,1});});++refused;}
			else{(void)generate_vertex_program(std::span{&w,1});++admitted;}
		}
		require(cases==10240&&refused==192&&admitted==1344);
		std::cout<<"Indexed vertex outputs: 10240 CPU route/bank/component/base/mask combinations; 1536 paired ARL/ARR/ARA address cases, 192 observable hazards refused and 1344 disjoint/inactive cases admitted\n";
	}
	{
		unsigned cases = 0;
		for (unsigned address : {0u, 1u}) for (unsigned mask = 0; mask < 16; ++mask)
		for (unsigned swizzle = 0; swizzle < 256; ++swizzle) for (unsigned modifier = 0; modifier < 4; ++modifier)
		{
			auto w = alu(23, 0);
			w[0] = (w[0] & ~(63u << 15)) | (address << 15) | (unsigned(modifier >= 2) << 21);
			w[3] = (w[3] & ~((15u << 13) | (31u << 2))) | (mask << 13) | (31u << 2);
			set_source(w, 0, (swizzle << 8) | ((modifier & 1u) << 16) | (1u + swizzle % 3));
			const auto shader = generate_vertex_program(std::span{&w, 1});
			require(shader.source.find("const int4 value = rsx_address(rint(") != std::string::npos && !shader.output_register_mask);
			if (!cases) corpus.push_back(shader);
			++cases;
		}
		for (unsigned opcode : {23u, 24u})
		for (unsigned flags : {0u, 1u << 14 | 1u << 29})
		{
			auto w = alu(opcode, 0); w[0] |= flags;
			rejects([&] { (void)generate_vertex_program(std::span{&w, 1}); }); // Integer address operations cannot write a float output.
			w[3] |= 31u << 2; w[0] |= 1u << 26;
			rejects([&] { (void)generate_vertex_program(std::span{&w, 1}); }); // Nor use SAT.
		}
		require(cases == 32768);
		std::cout << "ARR: 32768 CPU address/mask/swizzle/modifier cases, three source types, paired condition/source/output guards and SAT/output refusals passed\n";
	}
	{
		unsigned cases = 0;
		for (unsigned source = 0; source < 2; ++source) for (unsigned destination = 0; destination < 2; ++destination)
		for (unsigned bank = 0; bank < 2; ++bank) for (unsigned mask = 0; mask < 16; ++mask)
		for (unsigned predicate = 0; predicate < 3; ++predicate) for (unsigned update = 0; update < 3; ++update)
		{
			auto w = alu(24, 1);
			w[0] = (destination << 15) | (source << 24) | (bank << 25) | (1u << 30) |
				(unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10) |
				(unsigned(update != 0) << 14) | (unsigned(update == 1) << 29);
			w[3] = (w[3] & ~((15u << 13) | (31u << 2))) | (mask << 13) | (31u << 2);
			set_source(w, 0, 0); // ARA reads ASRC, not a float source descriptor.
			const auto shader = generate_vertex_program(std::span{&w, 1});
			require(shader.source.find("const int4 value = rsx_address_add(r.address[" + std::to_string(source) + "]") != std::string::npos);
			require(!shader.attribute_mask && !shader.indexed_constants && !shader.output_register_mask);
			if (!cases) corpus.push_back(shader);
			++cases;
		}
		require(cases == 1152);
		std::cout << "ARA: 1152 CPU source/destination/condition/mask/predicate/update cases, including source/destination aliasing and zero float-source descriptors passed\n";
	}
	{
		unsigned cases = 0;
		for (unsigned opcode : {5u, 6u}) for (unsigned component = 0; component < 4; ++component)
		for (unsigned mask = 0; mask < 16; ++mask) for (unsigned bank = 0; bank < 2; ++bank)
		for (unsigned mode = 0; mode < 3; ++mode) for (unsigned predicate = 0; predicate < 3; ++predicate) for (bool sat : {false, true})
		{
			auto w = alu(mode ? 1u : 0u, opcode);
			w[0] = ((mode ? 4u : 63u) << 15) | (1u << 14) | (bank << 25) | (unsigned(mode == 2) << 29) |
				(unsigned(sat) << 26) | (unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10);
			w[3] = (w[3] & ~((15u << 17) | (63u << 7) | (31u << 2))) | (mask << 17) | (5u << 7) | (1u << 2);
			set_source(w, 0, 0x1b03); set_source(w, 2, (component * 0x55u << 8) | 3u);
			const auto shader = generate_vertex_program(std::span{&w, 1});
			require(shader.source.find(opcode == 5 ? "= rsx_exp_legacy(" : "= rsx_log_legacy(") != std::string::npos && shader.output_register_mask == 2u);
			if (!component && mask == 15 && !bank && !mode && !predicate && !sat) corpus.push_back(shader);
			++cases;
		}
		require(cases == 4608);
		std::cout << "Legacy EXP/LOG: 4608 CPU opcode/source-component/mask/condition/pair/predicate/SAT cases select the native four-component helpers\n";
	}
	{
		unsigned cases = 0;
		for (unsigned opcode : {2u,3u,4u,13u,14u}) for (unsigned component = 0; component < 4; ++component)
		for (unsigned mask = 0; mask < 16; ++mask) for (unsigned bank = 0; bank < 2; ++bank)
		for (unsigned mode = 0; mode < 3; ++mode) for (unsigned predicate = 0; predicate < 3; ++predicate) for (bool sat : {false,true})
		{
			auto w = alu(mode ? 1u : 0u,opcode);
			w[0] = ((mode ? 4u : 63u) << 15) | (1u << 14) | (bank << 25) | (unsigned(mode == 2) << 29) |
				(unsigned(sat) << 26) | (unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10);
			w[3] = (w[3] & ~((15u << 17) | (63u << 7) | (31u << 2))) | (mask << 17) | (5u << 7) | (1u << 2);
			set_source(w,0,0x1b03);set_source(w,2,(component * 0x55u << 8) | 3u);
			const auto shader = generate_vertex_program(std::span{&w,1});
			const std::string helper = opcode == 2 ? "rsx_reciprocal" : opcode == 3 ? "rsx_rcp_clamped" : opcode == 4 ? "rsx_rsqrt" : opcode == 14 ? "rsx_exp2_scalar" : "rsx_log2_scalar";
			require(shader.source.find("= float4(" + helper + "(") != std::string::npos && shader.output_register_mask == 2u);
			if (!component && mask == 15 && !bank && !mode && !predicate && !sat) corpus.push_back(shader);
			++cases;
		}
		require(cases == 11520);
		std::cout << "Scalar RCP/RCC/RSQ/LG2/EX2: 11520 CPU source/mask/condition/pair/predicate/SAT cases select native helpers\n";
	}
	{
		unsigned cases = 0;
		for (unsigned swizzle = 0; swizzle < 256; ++swizzle) for (unsigned mask = 0; mask < 16; ++mask)
		for (unsigned bank = 0; bank < 2; ++bank) for (unsigned mode = 0; mode < 3; ++mode)
		for (unsigned predicate = 0; predicate < 3; ++predicate) for (bool sat : {false,true})
		{
			auto w = alu(mode ? 1u : 0u,7);
			w[0] = ((mode ? 4u : 63u) << 15) | (1u << 14) | (bank << 25) | (unsigned(mode == 2) << 29) |
				(unsigned(sat) << 26) | (unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10);
			w[3] = (w[3] & ~((15u << 17) | (63u << 7) | (31u << 2))) | (mask << 17) | (5u << 7) | (1u << 2);
			set_source(w,0,0x1b03);set_source(w,2,(swizzle << 8) | 3u);
			const auto shader = generate_vertex_program(std::span{&w,1});
			require(shader.source.find("= rsx_lit(") != std::string::npos && shader.output_register_mask == 2u);
			if ((swizzle == 0x1b || swizzle == 0xe4 || swizzle == 0x17 || swizzle == 0x4b) && mask == 15 && !bank && !mode && !predicate && !sat) corpus.push_back(shader);
			++cases;
		}
		require(cases == 147456);
		std::cout << "LIT: 147456 CPU swizzle/mask/condition/pair/predicate/SAT cases select native vector lighting, including all256 swizzles\n";
	}
	{
		unsigned cases = 0;
		constexpr std::array opcodes{11u,12u,16u,18u,19u,20u}, modes{1u,6u,2u,4u,3u,5u};
		for (unsigned op = 0; op < opcodes.size(); ++op) for (unsigned swizzle = 0; swizzle < 256; ++swizzle)
		for (unsigned mask = 0; mask < 16; ++mask) for (unsigned bank = 0; bank < 2; ++bank)
		for (unsigned predicate = 0; predicate < 3; ++predicate) for (bool sat : {false,true})
		{
			const unsigned mode = cases % 3;
			auto w = alu(opcodes[op],mode ? 1u : 0u);
			w[0] = (5u << 15) | (1u << 30) | (1u << 14) | (unsigned(mode != 2) << 29) | (bank << 25) |
				(unsigned(sat) << 26) | (unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10);
			w[3] = (mask << 13) | ((mode ? 15u : 0u) << 17) | (7u << 7) | (1u << 2) | 1u;
			set_source(w,0,(swizzle << 8) | 3u); set_source(w,1,((swizzle ^ 255u) << 8) | (6u << 2) | 1u);
			set_source(w,2,0x1b03);
			const auto shader = generate_vertex_program(std::span{&w,1});
			require(shader.source.find("= rsx_set_compare(") != std::string::npos && shader.output_register_mask == 2u);
			require(shader.source.find(", " + std::to_string(modes[op]) + "u);") != std::string::npos);
			if ((swizzle == 0x1b || swizzle == 0xe4 || swizzle == 0x17 || swizzle == 0x4b) && mask == 15 && !bank && !predicate && !sat) corpus.push_back(shader);
			++cases;
		}
		require(cases == 294912);
		std::cout << "Vertex comparisons: 294912 CPU opcode/swizzle/mask/bank/predicate/SAT cases; both operands cover all256 swizzles, distributed paired result selection passed\n";
	}
	{
		unsigned cases = 0;
		for (unsigned opcode : {9u,10u,14u,15u,22u}) for (unsigned swizzle = 0; swizzle < 256; ++swizzle)
		for (unsigned mask = 0; mask < 16; ++mask) for (unsigned bank = 0; bank < 2; ++bank)
		for (unsigned predicate = 0; predicate < 3; ++predicate) for (bool sat : {false,true})
		{
			const unsigned mode = cases % 3;
			auto w = alu(opcode,mode ? 1u : 0u);
			w[0] = (5u << 15) | (1u << 30) | (1u << 14) | (unsigned(mode != 2) << 29) | (bank << 25) |
				(unsigned(sat) << 26) | (unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10);
			w[3] = (mask << 13) | ((mode ? 15u : 0u) << 17) | (7u << 7) | (1u << 2) | 1u;
			set_source(w,0,(swizzle << 8) | 3u);set_source(w,1,((swizzle ^ 255u) << 8) | (6u << 2) | 1u);set_source(w,2,0x1b03);
			const auto shader = generate_vertex_program(std::span{&w,1});
			const std::string helper = opcode == 9 || opcode == 10 ? "rsx_extremum" : opcode == 14 ? "rsx_fraction" : opcode == 15 ? "rsx_floor" : "rsx_sign";
			require(shader.source.find("= " + helper + "(") != std::string::npos && shader.output_register_mask == 2u);
			if ((swizzle == 0x1b || swizzle == 0xe4 || swizzle == 0x17 || swizzle == 0x4b) && mask == 15 && !bank && !predicate && !sat) corpus.push_back(shader);
			++cases;
		}
		require(cases == 245760);
		std::cout << "Vertex MIN/MAX/FRC/FLR/SSG: 245760 CPU opcode/swizzle/mask/bank/predicate/SAT cases; all256 swizzles and distributed paired result selection passed\n";
	}
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
