#include "M4ShaderSnapshotFixtures.h"

#include <iostream>

namespace
{
using namespace metal4;
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Analysed RSX shader snapshot oracle failed");
}
template <typename F>
void rejects(F&& action)
{
	bool refused = false;
	try
	{
		action();
	}
	catch (const std::exception&)
	{
		refused = true;
	}
	require(refused);
}
} // namespace
int main()
{
	try
	{
		std::size_t cases = 0;
		for (unsigned base = 0; base < rsx_vertex_instruction_limit; ++base)
			for (unsigned count = 1; count <= rsx_vertex_instruction_limit - base; ++count)
				for (unsigned entry : {0u, count - 1})
				{
					test::vertex_program source;
					source.base_address = base;
					source.entry = base + entry;
					source.data.resize(count * 4);
					for (unsigned i = 0; i < count * 4; ++i)
						source.data[i] = 0xa5000000u + i;
					for (unsigned i = 0; i < count; ++i)
						source.instruction_mask.set(i, i % 2 == 0 || i == entry);
					source.ctrl = (base & 1) ? rsx_program_flat : 0;
					source.output_mask = base ^ count;
					const auto frozen = vertex_program_snapshot::capture(source);
					const auto options = frozen.options(false);
					require(options.entry == entry && options.base_address == base && options.live_instructions.size() == count && options.flat_colors == bool(base & 1));
					vertex_context ctx;
					frozen.apply(ctx);
					require(ctx.output_mask == (base ^ count));
					std::fill(source.data.begin(), source.data.end(), 0);
					source.instruction_mask.reset();
					for (unsigned i = 0; i < count; ++i)
						require(frozen.code()[i][3] == 0xa5000000u + i * 4 + 3 && frozen.live()[i] == (i % 2 == 0 || i == entry));
					++cases;
				}
		std::array<std::byte, 32> bytes{};
		for (unsigned flags = 0; flags < 256; ++flags)
			for (unsigned dimension = 0; dimension < 4; ++dimension)
			{
				test::fragment_program source;
				source.pointer = bytes.data();
				source.ucode_length = bytes.size();
				for (unsigned i = 0; i < 16; ++i)
					source.texture_state.texture_dimensions |= dimension << (i * 2);
				source.ctrl = ((flags & 1) ? 0x40 : 0) | ((flags & 2) ? 0xe : 0) | ((flags & 4) ? 0x800 : 0) | ((flags & 8) ? 0x80000 : 0) | ((flags & 16) ? 0x100000 : 0) | ((flags & 32) ? 0x800000 : 0) | ((flags & 64) ? 0x1000000 : 0) | ((flags & 128) ? 0x400000 : 0);
				source.texcoord_control_mask = 0xa55a1234;
				source.two_sided_lighting = flags & 1;
				source.texture_state.shadow_textures = 0x1234;
				source.texture_state.redirected_textures = 0x5678;
				source.texture_state.multisampled_textures = 0x9abc;
				const auto frozen = fragment_program_snapshot::capture(source, 32);
				const auto options = frozen.options(true);
				require(options.float32_exports == bool(flags & 1) && options.depth_export == bool(flags & 2) && options.flat_colors == bool(flags & 4));
				require(options.output_rop.round_to_8bit == bool(flags & 8) && !frozen.options(false).output_rop.round_to_8bit && options.output_rop.srgb == bool(flags & 16) && options.output_rop.polygon_stipple == bool(flags & 32) && options.output_rop.stochastic_alpha_coverage == bool(flags & 64));
				require(options.shadow_textures == 0x1234 && options.redirected_textures == 0x5678 && options.multisampled_textures == 0x9abc);
				for (const auto d : options.textures)
					require(d == std::array{vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::cube, vertex_texture_dimension::three}[dimension]);
				vertex_context vc;
				fragment_context fc;
				frozen.apply(vc, fc);
				require(vc.two_sided == bool(flags & 1) && fc.two_sided == bool(flags & 1) && fc.texcoord_2d_mask == 0x1234 && fc.point_sprite_mask == 0xa55a && fc.alpha_test == bool(flags & 128));
				++cases;
			}
		test::vertex_program v;
		v.data.resize(4);
		v.instruction_mask.set(0);
		v.ctrl = rsx_program_instanced_constants;
		rejects([&]
				{ (void)vertex_program_snapshot::capture(v); });
		const auto independent = vertex_program_snapshot::capture(v, true);
		require(independent.code()[0] == std::array<std::uint32_t, 4>{});
		require(v.ctrl == rsx_program_instanced_constants);
		v.ctrl = 0;
		v.entry = 1;
		rejects([&]
				{ (void)vertex_program_snapshot::capture(v); });
		v.entry = 0;
		v.instruction_mask.reset();
		rejects([&]
				{ (void)vertex_program_snapshot::capture(v); });
		v.instruction_mask.set(0);
		v.data.resize(5);
		rejects([&]
				{ (void)vertex_program_snapshot::capture(v); });
		test::fragment_program f;
		f.pointer = bytes.data();
		f.ucode_length = 32;
		rejects([&]
				{ (void)fragment_program_snapshot::capture(f, 31); });
		f.ucode_length = 17;
		rejects([&]
				{ (void)fragment_program_snapshot::capture(f, 32); });
		f.ucode_length = 32;
		f.valid = false;
		rejects([&]
				{ (void)fragment_program_snapshot::capture(f, 32); });
		f.valid = true;
		f.pointer = nullptr;
		rejects([&]
				{ (void)fragment_program_snapshot::capture(f, 32); });
		f.pointer = bytes.data();
		f.mrt_buffers_count = 5;
		rejects([&]
				{ (void)fragment_program_snapshot::capture(f, 32); });
		f.mrt_buffers_count = 1;
		bytes[7] = std::byte{0x5a};
		auto frozen = fragment_program_snapshot::capture(f, 32);
		bytes[7] = std::byte{0xff};
		auto copied = frozen;
		auto moved = std::move(copied);
		require(moved.code()[7] == std::byte{0x5a} && frozen.code()[7] == std::byte{0x5a});
		std::cout << cases << " analysed RSX shader entry/base/mask/control/dimension snapshots,owned code and invalid-input rejection profiles passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
