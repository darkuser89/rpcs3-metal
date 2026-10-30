#include "M4VertexDataFixtures.h"
#include <iostream>

using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
int main()
{
	try
	{
		test::vertex_data_state state;
		for (unsigned i = 0; i < 512; ++i)
			for (unsigned lane = 0; lane < 4; ++lane)
				state.transform_constants[i][lane] = 0x80000000u ^ (i * 4 + lane);
		state.transform_constants[511][3] = 0x7fc12345;
		test::vertex_data_processor processor;
		processor.descriptors[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
		processor.descriptors[1] = {4u | (3u << 8) | (4u << 24) | (4u << 27), 0xc0000000u};
		processor.persistent.resize(48, std::byte{0x91});
		processor.transient.resize(12, std::byte{0xe7});
		test::vertex_data_metadata metadata{3};
		const auto capture = [&](vertex_stream_range range = {102, 12, 3, 12, 90},
								 std::pair<std::uint32_t, std::uint32_t> sizes = {48, 12}, std::size_t budget = 60)
		{
			return vertex_data_snapshot::capture(processor, 0, metadata, state, range, sizes, budget);
		};
		vertex_constant_cache versions;
		const auto version_capture = [&] { return vertex_data_snapshot::capture(processor, 0, metadata, state, {102, 12, 3, 12, 90}, {48, 12}, 60, &versions); };
		const auto owned_version = version_capture();
		for (unsigned i = 0; i < 100; ++i)
			require(version_capture().constant_version() == owned_version.constant_version(), "Unchanged CPU constants copied again");
		require(versions.statistics().versions == 1 && versions.statistics().reused == 100 && versions.statistics().copied_bytes == 8192,
			"Immutable constant copy counters differ");
		for (unsigned reg = 468; reg < 512; ++reg)
		{
			const auto before = version_capture();
			state.transform_constants[reg][0] ^= 0x80000000u;
			const auto changed = version_capture();
			require(changed.constant_version() != before.constant_version() && changed.constants()[reg][0] == state.transform_constants[reg][0] &&
				before.constants()[reg][0] != changed.constants()[reg][0], "High constant register mutation reused stale version");
		}
		state.transform_constants[511][3] = 0x7fc54321;
		const auto nan = version_capture();
		require(nan.constants()[511][3] == 0x7fc54321 && owned_version.constants()[511][3] == 0x7fc12345, "NaN payload versioning changed bits");
		versions.clear();
		require(version_capture().constant_version() != nan.constant_version() && nan.constants()[511][3] == 0x7fc54321,
			"Clearing current version invalidated an owned packet");
		state.transform_constants[511][3] = 0x7fc12345;
		std::cout << "CPU constant versions:100 unchanged captures copy8192 bytes once;all44 high registers,sign bits,NaN payloads and retained old packets passed\n";
		const auto snapshot = capture();
		require(processor.captured_first == 102 && processor.captured_count == 3, "Stream origin confused with shader vertex ID");
		require(snapshot.fetch().first_vertex == 12 && snapshot.fetch().vertex_base_index == 12 &&
					snapshot.fetch().vertex_index_offset == 90,
				"Index rebasing/frequency contract lost");
		for (unsigned i = 0; i < 512; ++i)
			for (unsigned lane = 0; lane < 4; ++lane)
				require(snapshot.constants()[i][lane] == state.transform_constants[i][lane], "Full constant register bits lost");
		processor.persistent.assign(48, std::byte{0});
		processor.transient.assign(12, std::byte{0});
		state.transform_constants = {};
		require(snapshot.persistent()[47] == std::byte{0x91} && snapshot.transient()[11] == std::byte{0xe7} &&
					snapshot.constants()[511][3] == 0x7fc12345,
				"Snapshot borrowed mutable source data");
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
			require(processor.write_calls == writes, "Invalid snapshot read VM bytes");
		};
		refusal([&]
				{ (void)capture({}, {48, 12}, 59); });
		refusal([&]
				{ (void)capture({102, 12, 3, 12, 90}, {47, 12}); });
		refusal([&]
				{ (void)capture({102, 12, 3, 12, 90}, {48, 11}); });
		refusal([&]
				{ (void)capture({0, 0xffffffffu, 2, 0, 0}); });
		refusal([&]
				{ (void)capture({0xffffffffu, 0, 2, 0, 0}); });
		refusal([&]
				{ (void)capture({102, 12, 3, 13, 90}); });
		refusal([&]
				{ (void)capture({102, 0, 3, 0, -1}); });
		metadata.referenced_inputs_mask = 0x10000;
		refusal([&]
				{ (void)capture(); });
		metadata.referenced_inputs_mask = 3;
		processor.descriptors[0].format |= 7u << 27;
		refusal([&]
				{ (void)capture(); });
		require(rejected == 9, "Expected refusal did not occur");
		const auto layouts = processor.layout_calls, writes = processor.write_calls;
		const auto empty = capture({}, {0, 0});
		require(empty.fetch().vertex_count == 0 && processor.layout_calls == layouts && processor.write_calls == writes,
				"Empty capture touched streams");
		std::cout << "Owned packed vertex streams, distinct origins, all 512 constant registers and nine pre-read refusals passed\n";
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
