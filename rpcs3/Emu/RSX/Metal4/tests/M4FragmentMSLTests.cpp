#include "M4FragmentMSL.h"
#include "M4FragmentRegisters.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
using words = std::array<std::uint32_t, 4>;

void require(bool condition)
{
	if (!condition)
		throw std::runtime_error("Metal 4 fragment MSL invariant failed");
}

template <typename Function>
void rejects(Function function)
{
	try
	{
		function();
	}
	catch (const std::invalid_argument&)
	{
		return;
	}
	catch (const std::out_of_range&)
	{
		return;
	}
	throw std::runtime_error("Invalid RSX fragment program was translated");
}

void append(std::vector<std::byte>& bytes, words instruction)
{
	for (auto value : instruction)
	{
		for (unsigned shift : {8u, 0u, 24u, 16u})
			bytes.push_back(static_cast<std::byte>(value >> shift));
	}
}

words instruction(std::uint32_t opcode, bool half = false)
{
	constexpr auto xyzw = 0xe4u << 9;
	return {(opcode % 64) << 24 | (half ? 128u : 0u) | (15u << 9) | 1u,
			xyzw | (7u << 18) | (0xe4u << 21) | (half ? 256u : 0u),
			xyzw | (2u << 2) | (opcode / 64) << 31 | (half ? 256u : 0u),
			xyzw | (3u << 2) | (half ? 256u : 0u)};
}

void register_tests()
{
	using namespace metal4;
	for (std::uint32_t value = 0; value < 65536; ++value)
	{
		require(fragment_insert_half(0xa5c39b72, static_cast<std::uint16_t>(value), 0) == (0xa5c30000u | value));
		require(fragment_insert_half(0xa5c39b72, static_cast<std::uint16_t>(value), 16) == (0x00009b72u | (value << 16)));
	}
	std::array<std::uint8_t, fragment_register_words * 4> bytes;
	std::array<std::uint32_t, fragment_register_words> bank;
	for (std::size_t i = 0; i < bytes.size(); ++i)
		bytes[i] = static_cast<std::uint8_t>(i * 37 + 11);
	for (std::size_t i = 0; i < bank.size(); ++i)
	{
		bank[i] = std::uint32_t(bytes[i * 4]) | (std::uint32_t(bytes[i * 4 + 1]) << 8) |
				  (std::uint32_t(bytes[i * 4 + 2]) << 16) | (std::uint32_t(bytes[i * 4 + 3]) << 24);
	}
	for (std::uint32_t reg = 0; reg < 64; ++reg)
	{
		for (std::uint32_t channel = 0; channel < 4; ++channel)
		{
			const auto location = fragment_location(reg, channel, true);
			const auto address = std::size_t(reg) * 8 + std::size_t(channel) * 2;
			const auto expected = std::uint32_t(bytes[address]) | (std::uint32_t(bytes[address + 1]) << 8);
			require(((bank[location.word] >> location.shift) & 65535u) == expected);
			for (std::uint16_t value : {std::uint16_t(0), std::uint16_t(1), std::uint16_t(0x8000), std::uint16_t(0xffff), std::uint16_t(0x7e37)})
			{
				bank[location.word] = fragment_insert_half(bank[location.word], value, location.shift);
				bytes[address] = static_cast<std::uint8_t>(value);
				bytes[address + 1] = static_cast<std::uint8_t>(value >> 8);
				for (std::uint32_t full = 0; full < 48; ++full)
				{
					for (std::uint32_t c = 0; c < 4; ++c)
					{
						const auto a = std::size_t(full) * 16 + std::size_t(c) * 4;
						const auto reference = std::uint32_t(bytes[a]) | (std::uint32_t(bytes[a + 1]) << 8) |
											   (std::uint32_t(bytes[a + 2]) << 16) | (std::uint32_t(bytes[a + 3]) << 24);
						require(bank[fragment_location(full, c, false).word] == reference);
					}
				}
			}
		}
	}
	rejects([]
			{
				(void)fragment_location(48, 0, false);
			});
	rejects([]
			{
				(void)fragment_location(64, 0, true);
			});
	rejects([]
			{
				(void)fragment_location(0, 4, true);
			});
	rejects([]
			{
				(void)fragment_insert_half(0, 0, 1);
			});
}

} // namespace

int main(int argc, char** argv)
{
	using namespace metal4;
	register_tests();
	std::vector<generated_fragment_program> corpus;
	constexpr std::array alu{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u,
							 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 26u, 27u, 28u, 29u, 30u, 31u, 32u,
							 33u, 34u, 35u, 36u, 37u, 39u, 40u, 41u, 42u, 44u, 45u, 46u, 54u, 56u, 57u, 58u, 59u, 60u};
	for (auto op : alu)
	{
		for (bool half : {false, true})
		{
			std::vector<std::byte> bytes;
			append(bytes, instruction(op, half));
			corpus.push_back(generate_fragment_program(bytes));
		}
	}
	for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two,
						   vertex_texture_dimension::three, vertex_texture_dimension::cube})
	{
		for (auto op : {23u, 24u, 25u, 47u, 49u})
		{
			fragment_program_options options;
			options.textures.fill(dimension);
			options.flat_colors = true;
			options.color_count = 4;
			options.depth_export = true;
			auto sample = instruction(op);
			sample[0] |= (15u << 17) | (4u << 13);
			sample[1] |= 1;
			std::vector<std::byte> bytes;
			append(bytes, sample);
			auto generated = generate_fragment_program(bytes, options);
			require(generated.texture_mask == 32768 && generated.input_mask == 16);
			require(generated.source.find("texture_state [[buffer(4)]]") != std::string::npos);
			require(generated.source.find("texture_state[15]") != std::string::npos);
			require(generated.source.find("sampler_state [[buffer(3)]]") != std::string::npos);
			require(generated.source.find("sampler_state[15].bias") != std::string::npos);
			if (op == 47)
			{
				require(generated.source.find("sampler_state[15].min_lod") != std::string::npos);
				require(generated.source.find("sampler_state[15].max_lod") != std::string::npos);
			}
			if (op == 25)
				require(generated.source.find("exp2(sampler_state[15].bias)") != std::string::npos);
			corpus.push_back(std::move(generated));
		}
	}
	// Compacted constants are identified by stream offset, not their mutable
	// values. Updating either literal must leave the shader source unchanged.
	{
		std::vector<std::byte> bytes;
		auto mov = instruction(1);
		mov[0] &= ~1u;
		mov[1] |= 2u;
		append(bytes, mov);
		append(bytes, {0x3f800000, 0x80000000, 0x7f800000, 0x7fc01234});
		mov[0] |= 1u;
		append(bytes, mov);
		append(bytes, {0x3f000000, 0x3f000000, 0, 0});
		auto generated = generate_fragment_program(bytes);
		const auto compiled = decode_fragment_program(bytes);
		const auto initial = refresh_fragment_constants(compiled, bytes);
		require(initial.size() == 2 && initial[0] == words{0x3f800000, 0x80000000, 0x7f800000, 0x7fc01234});
		require(generated.constant_offsets == std::vector<std::size_t>{16, 48});
		bytes[16] = std::byte{0xff};
		bytes[53] = std::byte{0xa5};
		require(generate_fragment_program(bytes).source == generated.source);
		const auto updated = refresh_fragment_constants(compiled, bytes);
		require(updated[0][0] == 0x3f80ff00 && updated[1][1] == 0x3f0000a5);
		bytes[4] ^= std::byte{1};
		rejects([&]
				{
					(void)refresh_fragment_constants(compiled, bytes);
				});
		bytes[4] ^= std::byte{1};
		for (std::size_t size = 0; size < 64; ++size)
		{
			rejects([&]
					{
						(void)refresh_fragment_constants(compiled, std::span(bytes).first(size));
					});
		}
		corpus.push_back(std::move(generated));
	}
	// IF then/else with a following instruction. Targets are u32-word
	// addresses; there are no explicit ELSE/ENDIF stream instructions.
	{
		std::vector<std::byte> bytes;
		auto branch = instruction(0x42);
		branch[0] &= ~1u;
		branch[2] = (1u << 31) | 8u;
		branch[3] = 12;
		append(bytes, branch);
		auto mov = instruction(1);
		mov[0] &= ~1u;
		append(bytes, mov);
		append(bytes, mov);
		append(bytes, instruction(1));
		corpus.push_back(generate_fragment_program(bytes));
	}
	for (auto op : {0x43u, 0x44u})
	{
		std::vector<std::byte> bytes;
		auto loop = instruction(op);
		loop[0] &= ~1u;
		loop[2] = (1u << 31) | (3u << 2) | (7u << 10) | (255u << 19);
		loop[3] = 12;
		append(bytes, loop);
		auto mov = instruction(1);
		mov[0] &= ~1u;
		mov[1] |= 1u;
		mov[3] = (1u << 30) | (1u << 19);
		append(bytes, mov);
		auto brk = instruction(0x40);
		brk[0] &= ~1u;
		append(bytes, brk);
		append(bytes, instruction(1));
		corpus.push_back(generate_fragment_program(bytes));
	}
	// LOOP -> IF -> REP -> RET, with BRK after the IF, exercises distinct
	// enclosing loops and restoration of the outer indexed-input address.
	{
		std::vector<std::byte> bytes;
		auto loop = instruction(0x43);
		loop[0] &= ~1u;
		loop[2] = (1u << 31) | (2u << 2) | (2u << 10) | (1u << 19);
		loop[3] = 24;
		append(bytes, loop);
		auto branch = instruction(0x42);
		branch[0] &= ~1u;
		branch[2] = (1u << 31) | 16u;
		branch[3] = 20;
		append(bytes, branch);
		auto rep = instruction(0x44);
		rep[0] &= ~1u;
		rep[2] = (1u << 31) | (3u << 2);
		rep[3] = 16;
		append(bytes, rep);
		auto ret = instruction(0x45);
		ret[0] &= ~1u;
		append(bytes, ret);
		auto mov = instruction(1);
		mov[0] &= ~1u;
		append(bytes, mov);
		auto brk = instruction(0x40);
		brk[0] &= ~1u;
		append(bytes, brk);
		append(bytes, instruction(1));
		corpus.push_back(generate_fragment_program(bytes));
		// Make the REP cross the containing IF's ENDIF.
		auto broken = bytes;
		broken[45] = std::byte{24}; // decoded src2 low byte
		rejects([&]
				{
					(void)generate_fragment_program(broken);
				});
	}
	for (std::uint32_t mode = 0; mode < 6; ++mode)
	{
		std::vector<std::byte> bytes;
		auto mov = instruction(1);
		mov[0] |= (1u << 8) | (1u << 31); // CC update and NaN-zero saturate
		mov[1] |= 1u << 29;
		mov[2] |= mode << 19;
		append(bytes, mov);
		corpus.push_back(generate_fragment_program(bytes));
	}
	{
		std::vector<std::byte> bytes;
		append(bytes, instruction(0x45));
		fragment_program_options options;
		options.color_count = 0;
		options.float32_exports = false;
		corpus.push_back(generate_fragment_program(bytes, options));
	}
	for (auto op : {38u, 43u, 48u, 50u, 51u, 52u, 53u, 55u, 61u, 62u, 63u, 65u, 70u, 127u})
	{
		rejects([&]
				{
					std::vector<std::byte> bytes;
					append(bytes, instruction(op));
					(void)generate_fragment_program(bytes);
				});
	}
	rejects([]
			{
				std::vector<std::byte> b;
				append(b, instruction(0x40));
				(void)generate_fragment_program(b);
			});
	for (auto invalid : {0u, 1u, 2u, 3u})
	{
		rejects([&]
				{
					auto w = instruction(1);
					if (invalid == 0)
						w[1] |= 3;
					if (invalid == 1)
						w[1] = (w[1] & ~252u) | (48u << 2);
					if (invalid == 2)
						w[0] |= 48u << 1;
					if (invalid == 3)
						w[2] |= 4u << 28;
					std::vector<std::byte> b;
					append(b, w);
					(void)generate_fragment_program(b);
				});
	}
	for (auto flags : {2u, 4u})
	{
		rejects([&]
				{
					auto w = instruction(23);
					fragment_program_options options;
					if (flags == 2)
						options.redirected_textures = 1;
					if (flags == 4)
						options.multisampled_textures = 1;
					std::vector<std::byte> b;
					append(b, w);
					(void)generate_fragment_program(b, options);
				});
	}
	{
		auto w = instruction(23);
		w[0] |= 1u << 21;
		std::vector<std::byte> code;
		append(code, w);
		auto generated = generate_fragment_program(code);
		require(generated.source.find("texel_state [[buffer(5)]]") != std::string::npos);
		require(generated.source.find("rsx_process_texel(ftex0.sample(") != std::string::npos);
		require(generated.source.find(", true, r.killed, r.fault)") != std::string::npos);
		require(generated.source.find("any(mask) ? rsx_process_texel") == std::string::npos);
		corpus.push_back(std::move(generated));
	}
	for (unsigned condition : {0u, 4u})
	{
		auto w = instruction(23);
		w[1] = (w[1] & ~(7u << 18)) | (condition << 18);
		std::vector<std::byte> code;
		append(code, w);
		const auto generated = generate_fragment_program(code);
		require((generated.source.find("false ? rsx_process_texel") != std::string::npos) == (condition == 0));
	}
	for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::three, vertex_texture_dimension::cube})
		for (unsigned op : {23u, 24u, 25u, 47u, 49u})
		{
			fragment_program_options options;
			options.depth_textures = 1;
			options.textures[0] = dimension;
			std::vector<std::byte> code;
			append(code, instruction(op));
			auto generated = generate_fragment_program(code, options);
			require(generated.source.find(dimension == vertex_texture_dimension::cube ? "depthcube<float>" : dimension == vertex_texture_dimension::three ? "texture3d<float>" : "depth2d<float>") != std::string::npos);
			require(generated.source.find(dimension==vertex_texture_dimension::three?"rsx_process_texel(rsx_depth_texel((ftex0.sample(":"rsx_process_texel(rsx_depth_texel(ftex0.sample(") != std::string::npos);
			if(dimension==vertex_texture_dimension::three)require(generated.source.find(").r, uint4(")!=std::string::npos);
			require(generated.source.find(".sample_compare(") == std::string::npos);
			corpus.push_back(std::move(generated));
		}
	{
		fragment_program_options options;
		options.depth_textures = 1;
		options.depth_remap[0] = {depth_sample_channel::one, depth_sample_channel::zero, depth_sample_channel::depth, depth_sample_channel::one};
		std::vector<std::byte> code;
		append(code, instruction(23));
		auto generated = generate_fragment_program(code, options);
		require(generated.source.find("uint4(2, 1, 0, 2)") != std::string::npos);
		corpus.push_back(std::move(generated));
		for (unsigned invalid = 0; invalid < 4; ++invalid)
			rejects([&]
					{
						auto bad = options;
						if (invalid == 0)
							bad.depth_textures = 1u << 16;
						if (invalid == 1)
							bad.shadow_textures = 1;
						if (invalid == 2)
							bad.textures[0] = static_cast<vertex_texture_dimension>(4);
						if (invalid == 3)
							bad.depth_remap[0][2] = static_cast<depth_sample_channel>(3);
						(void)generate_fragment_program(code, bad);
					});
	}
	for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::cube})
		for (unsigned op : {23u, 24u, 25u, 47u, 49u})
		{
			fragment_program_options options;
			options.shadow_textures = 1;
			options.textures[0] = dimension;
			std::vector<std::byte> code;
			append(code, instruction(op));
			if (dimension == vertex_texture_dimension::cube && op == 24)
			{
				rejects([&]
						{
							(void)generate_fragment_program(code, options);
						});
				continue;
			}
			auto generated = generate_fragment_program(code, options);
			require(generated.source.find(dimension == vertex_texture_dimension::cube ? "depthcube<float>" : "depth2d<float>") != std::string::npos);
			require(generated.source.find(".sample_compare(") != std::string::npos);
			require(generated.source.find("rsx_shadow_reference(") != std::string::npos);
			require(generated.source.find("rsx_process_texel(ftex") == std::string::npos);
			require(generated.source.find("bias(sampler_state[0].bias)") != std::string::npos);
			require(generated.source.find("level(clamp(") == std::string::npos);
			require(generated.source.find("gradient2d(") == std::string::npos);
			require(generated.source.find("gradientcube(") == std::string::npos);
			corpus.push_back(std::move(generated));
		}
	rejects([]
			{
				fragment_program_options options;
				options.shadow_textures = 1;
				options.textures[0] = vertex_texture_dimension::three;
				std::vector<std::byte> code;
				append(code, instruction(23));
				(void)generate_fragment_program(code, options);
			});
	{
		std::vector<std::byte> bytes;
		auto branch = instruction(0x42);
		branch[0] &= ~1u;
		branch[2] = (1u << 31) | 4u;
		branch[3] = 8u; // ENDIF inside literal
		append(bytes, branch);
		auto mov = instruction(1);
		mov[1] |= 2;
		append(bytes, mov);
		append(bytes, {0, 0, 0, 0});
		rejects([&]
				{
					(void)generate_fragment_program(bytes);
				});
	}
	for (const auto& program : corpus)
	{
		// Bounds testing must use stored attachment depth in the native encoder,
		// never discard according to incoming or shader-exported fragment depth.
		require(program.source.find("rsx_stored_depth.r") == std::string::npos);
	}
	for (const auto remap : {color_export_remap::rgba, color_export_remap::blue, color_export_remap::blue_green,
							 color_export_remap::rgb_zero, color_export_remap::rgb_one})
	{
		fragment_program_options options;
		options.color_count = 4;
		options.color_remap.fill(remap);
		std::vector<std::byte> b;
		append(b, instruction(1));
		corpus.push_back(generate_fragment_program(b, options));
	}
	rejects([]
			{
				fragment_program_options options;
				options.color_remap[0] = static_cast<color_export_remap>(999);
				std::vector<std::byte> b;
				append(b, instruction(1));
				(void)generate_fragment_program(b, options);
			});
	for (unsigned operation = 0x1500; operation <= 0x150f; ++operation)
		for (unsigned samples : {1u, 4u})
		{
			fragment_program_options options;
			options.color_count = 4;
			options.logic_rop = {true, operation, samples, {{{8, 8, 8, 8}, {}, {8, 8, 0, 0}, {16, 16, 16, 16}}}};
			options.raster_sample_count = samples;
			std::vector<std::byte> b;
			append(b, instruction(1));
			auto program = generate_fragment_program(b, options);
			require(program.source.find("rsx_destination1") == std::string::npos);
			require(program.source.find("rsx_destination3 [[color(3), raster_order_group(3)]]") != std::string::npos);
			require((program.source.find("[[sample_id]]") != std::string::npos) == (samples > 1));
			require(program.source.find("if (ctx.alpha_test") < program.source.find("out.color0 = rsx_logic_color"));
			corpus.push_back(std::move(program));
		}
	for (unsigned malformed : {0u, 1u, 2u})
		rejects([&]
				{
					fragment_program_options options;
					options.logic_rop.enabled = true;
					if (malformed == 0)
						options.logic_rop.operation = 0x1510;
					if (malformed == 1)
						options.logic_rop.bits[0][0] = 17;
					if (malformed == 2)
						options.logic_rop.bits[1][0] = 8;
					std::vector<std::byte> b;
					append(b, instruction(1));
					(void)generate_fragment_program(b, options);
				});
	{
		fragment_program_options options;
		options.logic_rop = {true, 0x1506, 4, {}};
		options.raster_sample_count = 4;
		std::vector<std::byte> b;
		append(b, instruction(1));
		const auto program = generate_fragment_program(b, options);
		require(program.source.find("rsx_logic_word") == std::string::npos);
		require(program.source.find("[[sample_id]]") != std::string::npos);
	}
	for (unsigned op = 0; op < 8; ++op)
		for (unsigned samples : {1u, 4u})
			for (bool half_storage : {false, true})
			{
				fragment_program_options options;
				options.color_count = 4;
				options.color_remap = {color_export_remap::blue, color_export_remap::rgb_zero, color_export_remap::rgba, color_export_remap::blue_green};
				auto& blend = options.blend_rop;
				blend.sample_count = samples;
				blend.remap = options.color_remap;
				options.raster_sample_count = samples;
				blend.normalized = {true, true, false, true};
				blend.bits = {{{8, 0, 0, 0}, {8, 8, 8, 8}, {}, {8, 8, 0, 0}}};
				blend.half_float[2] = half_storage;
				for (unsigned i = 0; i < 4; ++i)
				{
					blend.colors[i].enabled = true;
					blend.colors[i].source_rgb = static_cast<blend_factor>((op + i) % 15);
					blend.colors[i].destination_rgb = blend_factor::constant_color;
					blend.colors[i].source_alpha = blend_factor::source_alpha_saturated;
					blend.colors[i].destination_alpha = blend_factor::one_minus_constant_alpha;
					blend.colors[i].operation_rgb = static_cast<blend_operation>(op);
					blend.colors[i].operation_alpha = static_cast<blend_operation>((op + i) % 8);
				}
				std::vector<std::byte> b;
				append(b, instruction(1));
				auto program = generate_fragment_program(b, options);
				require(program.source.find("color0 = rsx_blend_color") < program.source.find("out.color0 = color0.bbbb"));
				require(program.source.find("rsx_destination2 [[color(2), raster_order_group(2)]]") != std::string::npos);
				require((program.source.find("[[sample_id]]") != std::string::npos) == (samples > 1));
				corpus.push_back(std::move(program));
			}
	for (unsigned malformed : {0u, 1u, 2u, 3u})
		rejects([&]
				{
					fragment_program_options options;
					options.blend_rop.colors[0].enabled = true;
					if (malformed == 0)
						options.logic_rop.enabled = true;
					if (malformed == 1)
						options.blend_rop.colors[0].source_rgb = static_cast<blend_factor>(99);
					if (malformed == 2)
						options.blend_rop.normalized[0] = true;
					if (malformed == 3)
						options.blend_rop.remap[0] = color_export_remap::blue;
					std::vector<std::byte> b;
					append(b, instruction(1));
					(void)generate_fragment_program(b, options);
				});
	for (unsigned flags = 0; flags < 16; ++flags)
		for (unsigned count : {0u, 4u})
			for (bool full_exports : {true, false})
			{
				fragment_program_options options;
				options.color_count = count;
				options.float32_exports = full_exports;
				options.depth_export = count == 0;
				options.output_rop = {bool(flags & 1), bool(flags & 2), bool(flags & 4), bool(flags & 8)};
				std::vector<std::byte> b;
				append(b, instruction(1));
				auto program = generate_fragment_program(b, options);
				if (flags & 1)
				{
					const auto quantization = program.source.find(count ? "color0 = rsx_rop_round8" : "primary.a = rsx_rop_round8");
					require(quantization < program.source.find("if (ctx.alpha_test"));
				}
				if ((flags & 2) && count)
					require(program.source.find("color0.rgb = ") < program.source.find("if (ctx.alpha_test"));
				if (flags & 4)
					require(program.source.find("ctx.stipple_pattern[stipple.y]") < program.source.find("auto r = rsx_fp_evaluate"));
				if (flags & 8)
				{
					const auto coverage = program.source.find("if (!rsx_rop_alpha_coverage");
					require(program.source.find("if (ctx.alpha_test") < coverage);
					require(coverage < program.source.find("out.sample_mask = ctx.sample_mask"));
				}
				corpus.push_back(std::move(program));
			}
	for (unsigned malformed : {3u, 9u})
		rejects([&]
				{
					fragment_program_options options;
					options.raster_sample_count = malformed;
					std::vector<std::byte> b;
					append(b, instruction(1));
					(void)generate_fragment_program(b, options);
				});
	for (auto encoding : {depth_mirror_encoding::unorm16, depth_mirror_encoding::float32})
		for (bool write : {false, true})
			for (bool export_depth : {false, true})
				for (unsigned count : {0u, 4u})
					for (unsigned samples : {1u, 2u, 4u})
					{
						fragment_program_options options;
						options.depth_mirror = {encoding, write};
						options.depth_export = export_depth;
						options.color_count = count;
						options.raster_sample_count = samples;
						std::vector<std::byte> b;
						append(b, instruction(1));
						auto program = generate_fragment_program(b, options);
						require(program.source.find("rsx_stored_depth [[color(4), raster_order_group(4)]]") != std::string::npos);
						require(program.source.find("rsx_stored_depth.r >= ctx.depth_min") < program.source.find("auto r = rsx_fp_evaluate"));
						require(program.source.find("ctx.sample_mask & (1u <<") < program.source.find("rsx_stored_depth.r >= ctx.depth_min"));
						require(program.source.find("in.position.z >= ctx.depth_min") == std::string::npos);
						corpus.push_back(std::move(program));
					}
	rejects([]
			{
				fragment_program_options options;
				options.depth_mirror = {depth_mirror_encoding::none, true};
				std::vector<std::byte> b;
				append(b, instruction(1));
				(void)generate_fragment_program(b, options);
			});
	if (argc == 2)
	{
		const std::filesystem::path directory{argv[1]};
		std::filesystem::create_directories(directory);
		for (std::size_t i = 0; i < corpus.size(); ++i)
		{
			std::ofstream file(directory / (std::to_string(i) + ".metal"));
			file.exceptions(std::ios::failbit | std::ios::badbit);
			file << corpus[i].source;
		}
	}
	std::cout << "Metal 4 fragment MSL: all half-word values, physical alias byte model, "
			  << corpus.size() << " shader variants and malformed program rejection passed\n";
}
