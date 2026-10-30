#include "M4DrawPacketFixtures.h"
#include "M4ShaderSnapshotFixtures.h"

#include <iostream>
using namespace metal4;
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Owned current-pass capture contract differs");
}
int main()
{
	try
	{
		test::pass_state state;
		test::pass_processor processor;
		test::pass_layout layout;
		test::pass_metadata metadata;
		processor.descriptors[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27), 0xa0000000u};
		processor.persistent.resize(48, std::byte{0x91});
		for (unsigned index : {0xffffcu, 0u, 4u})
			for (unsigned shift : {24u, 16u, 8u, 0u})
				processor.index_bytes.push_back(std::byte(index >> shift));
		state.transform_constants[511][3] = 0x7fc12345;
		state.branch = 0x80001234;
		test::vertex_program vertex;
		vertex.data.resize(4);
		vertex.instruction_mask.set(0);
		std::array<std::byte, 16> fragment_bytes{};
		test::fragment_program fragment;
		fragment.pointer = fragment_bytes.data();
		fragment.ucode_length = 16;
		unsigned requirements_calls = 0;
		const auto requirements = [&](const auto&, unsigned first, unsigned count)
		{
			require(first == 0 && count == 9);
			++requirements_calls;
			return std::pair<std::uint32_t, std::uint32_t>{48, 0};
		};
		const auto capture = [&](draw_capture_limits limits = {})
		{
			return rsx_draw_packet::capture(state, processor, layout, metadata, vertex, fragment, 0,
											provoking_vertex::last, requirements, limits);
		};
		const auto packet = capture();
		require(requirements_calls == 1 && processor.write_calls == 1 && processor.command_calls == 1);
		require(packet.geometry().vertices == std::vector<vertex_reference>{{0, 8}, {4, 8}, {8, 8}});
		require(packet.vertices().fetch().vertex_count == 9 && packet.vertices().persistent().size() == 48);
		vertex_context vc;
		fragment_context fc;
		packet.environment().apply(vc, fc);
		require(vc.branch_bits == 0x80001234 && vc.scale_offset[5] == -1 && packet.environment().width() == 4);
		state.branch = 0;
		state.registers.assign(state.registers.size(), 0xffffffffu);
		state.transform_constants = {};
		vertex.data.assign(4, 0xffffffffu);
		fragment_bytes.fill(std::byte{0xff});
		processor.persistent.assign(48, std::byte{0});
		processor.index_bytes.assign(12, std::byte{0});
		require(packet.vertices().persistent()[47] == std::byte{0x91} && packet.vertices().constants()[511][3] == 0x7fc12345 &&
				packet.state().vertex().code()[0][0] == 0 && packet.state().fragment().code()[0] == std::byte{0});
		unsigned rejected = 0;
		const auto refusal = [&](auto action)
		{
			const auto writes = processor.write_calls;
			try
			{
				action();
			}
			catch (const std::exception&)
			{
				++rejected;
			}
			require(writes == processor.write_calls);
		};
		// Restore a valid owned capture epoch for individual preflight failures.
		vertex.data.assign(4, 0);
		fragment_bytes.fill(std::byte{0});
		state.registers.assign(state.registers.size(), 0);
		refusal([&]
				{ auto limits = draw_capture_limits{}; limits.fragment_bytes = 15; (void)capture(limits); });
		metadata.referenced_textures_mask = 16;
		refusal([&]
				{ (void)capture(); });
		metadata.referenced_textures_mask = 0;
		state.width = 0;
		refusal([&]
				{ (void)capture(); });
		state.width = 4;
		// Array origin/size reconstructs the same prefix without index memory.
		state.current_draw_clause.command = 1;
		state.current_draw_clause.count = 9;
		refusal([&]
				{ auto limits = draw_capture_limits{}; limits.vertex_stream_bytes = 47; (void)capture(limits); });
		require(rejected == 4);
		state.current_draw_clause.count = 0;
		const auto previous_requirements = requirements_calls, previous_writes = processor.write_calls;
		const auto empty = capture();
		require(empty.geometry().vertices.empty() && !empty.vertices().fetch().vertex_count &&
				requirements_calls == previous_requirements && processor.write_calls == previous_writes);
		std::cout << "Owned RSX pass packet: program/register/runtime/geometry/stream/512constants ownership,preflight refusal and empty capture passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
