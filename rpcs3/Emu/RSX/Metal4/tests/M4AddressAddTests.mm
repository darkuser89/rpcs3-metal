#include "M4ShaderCache.h"
#include "M4VertexShader.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>

namespace
{
using namespace metal4;
using words = std::array<std::uint32_t, 4>;

void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}

// Physical word positions, independent of the production decoder.
void operand(words& w, unsigned slot, unsigned packed)
{
	if (!slot) { w[1] |= packed >> 9; w[2] |= (packed & 511u) << 23; }
	else { w[2] |= packed >> 11; w[3] |= (packed & 2047u) << 21; }
}

words address_load(unsigned bank, unsigned constant)
{
	words w{bank << 15, (13u << 22) | (constant << 12), 0, (15u << 13) | (63u << 7) | (31u << 2)};
	operand(w, 0, 0x1b03); return w;
}

void run(bool float_add_control)
{
	const std::array samples{
		std::array{1, 2, 3, 4}, std::array{-1, -2, -3, -4}, std::array{-3, 7, 3, -7},
		std::array{16777216, -16777216, 1, -1}, std::array{2147483520, -2147483647 - 1, 127, 0},
		std::array{2147483520, 0, 128, 0}, std::array{-2147483647 - 1, 0, -1, 0},
		std::array{0, 2147483520, 0, 128}, std::array{0, -2147483647 - 1, 0, -1},
		std::array{0, 0, 0, 0}, std::array{3, -4, 1, 2}, std::array{11, 12, -11, -12}};
	const std::array<std::int32_t, 4> initial{11, 12, 13, 14};
	const std::array<float, 4> seed{-1, 0, 1, std::bit_cast<float>(0x7fc12345u)};
	context ctx; compiler compile(ctx); shader_cache cache(compile);
	vertex_context state; state.constant_count = 16; state.vertex_count = 1;
	const auto uniforms = make_upload(ctx, std::as_bytes(std::span{&state, 1}));
	const std::array<sampler_parameters, 4> sampling{};
	const auto samplers = make_upload(ctx, std::as_bytes(std::span(sampling)));
	const auto output = make_buffer(ctx, 24 * sizeof(words), buffer_storage::shared);
	std::array<std::shared_ptr<const vertex_shader>, 12> keys{};
	unsigned variants = 0, profiles = 0, overflows = 0, invalid_consumers = 0;
	for (unsigned source = 0; source < 2; ++source) for (unsigned destination = 0; destination < 2; ++destination)
	for (unsigned bank = 0; bank < 2; ++bank) for (unsigned mask : {0u, 5u, 10u, 15u})
	for (unsigned predicate = 0; predicate < 3; ++predicate) for (unsigned update = 0; update < 3; ++update)
	@autoreleasepool
	{
		const unsigned component = (variants / 3) % 4;
		words condition{(4u << 15) | (1u << 30) | (1u << 14) | (1u << 29) | (bank << 25),
			(1u << 22) | (2u << 12), 0, (15u << 13) | (63u << 7) | (31u << 2)};
		operand(condition, 0, 0x1b03);
		words add{(destination << 15) | (source << 24) | (bank << 25) | (1u << 30) | (0x1bu << 2) |
			(unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10) |
			(unsigned(update != 0) << 14) | (unsigned(update == 1) << 29),
			(24u << 22) | (1u << 27), 0, (mask << 13) | (15u << 17) | (5u << 7) | (31u << 2)};
		operand(add, 2, 0x1b01 | (4u << 2));
		std::vector<words> code{address_load(0, source == 0 ? 0 : 1), address_load(1, source == 1 ? 0 : 1), condition, add};
		for (unsigned consumer = 0; consumer < 3; ++consumer)
		{
			words w{(1u << 30) | (destination << 24) | component |
				((consumer == 2 ? 63u : 7u + consumer) << 15), 1u << 22, 0,
				(15u << 13) | (63u << 7) | ((consumer == 2 ? 4u : 31u) << 2)};
			if (!consumer) { w[1] |= 4u << 12; w[3] |= 1u << 1; operand(w, 0, 0x1b03); }
			else if (consumer == 1) { w[0] |= 1u << 27; w[1] |= 4u << 8; operand(w, 0, 0x1b02); }
			else { w[0] |= 1u << 28; w[3] |= 1u; operand(w, 0, 0x1b01 | (4u << 2)); }
			code.push_back(w);
		}
		auto generated = generate_vertex_program(code);
		require(generated.attribute_mask == 65535 && generated.indexed_constants && generated.output_register_mask == 65535,
			"ARA consumers lost conservative register planning");
		if (float_add_control)
		{
			const std::string expected = "long2(value.xy) + long2(value.zw)";
			const auto site = generated.source.find(expected);
			require(site != std::string::npos, "ARA numeric control missed the integer helper");
			generated.source.replace(site, expected.size(), "long2(float2(value.xy) + float2(value.zw))");
		}
		if (variants < keys.size())
		{
			const auto key = cache.vertex(code, {});
			require(cache.vertex(code, {}) == key && std::find(keys.begin(), keys.end(), key) == keys.end(), "ARA shader keys collided");
			keys[variants] = key;
		}
		generated.source += R"metal(
kernel void inspect_ara(device const float4* constants [[buffer(0)]], constant rsx_vertex_context& ctx [[buffer(1)]],
    constant rsx_sampler_parameters* samplers [[buffer(2)]], device uint4* out [[buffer(3)]]) {
    const auto r = rsx_evaluate(constants, constants, ctx, 0, samplers);
    out[0]=as_type<uint4>(r.address[0]); out[1]=as_type<uint4>(r.address[1]);
    out[2]=as_type<uint4>(r.condition[0]); out[3]=as_type<uint4>(r.condition[1]);
    out[4]=as_type<uint4>(r.temp[5]); out[5]=as_type<uint4>(r.temp[7]); out[6]=as_type<uint4>(r.temp[8]);
    for(uint i=0;i<16;++i) out[7+i]=as_type<uint4>(r.dst[i]); out[23]=uint4(r.fault);
}
)metal";
		const auto library = compile.library(generated.source, @"ARA integer registers and indexed consumers");
		const auto descriptor = [MTL4ComputePipelineDescriptor new]; descriptor.computeFunctionDescriptor = library_function(library, @"inspect_ara");
		const auto pipeline = compile.compute(descriptor);
		for (const auto& sample : samples)
		{
			std::array<std::array<float, 4>, 16> constants{};
			for (unsigned i = 0; i < 16; ++i) constants[i] = {float(i), float(i + 16), float(i + 32), float(i + 48)};
			for (unsigned lane = 0; lane < 4; ++lane) { constants[0][lane] = float(sample[lane]); constants[1][lane] = float(initial[lane]); }
			constants[2] = seed;
			// Signed 64-bit host oracle, independent of MSL vector math/conversion.
			const std::int64_t x = std::int64_t(sample[0]) + sample[2], y = std::int64_t(sample[1]) + sample[3];
			const bool valid = x >= INT32_MIN && x <= INT32_MAX && y >= INT32_MIN && y <= INT32_MAX;
			const std::array<std::int32_t, 4> sum = valid ? std::array{std::int32_t(x), std::int32_t(y), std::int32_t(x), std::int32_t(y)} : std::array<std::int32_t, 4>{};
			std::array<std::array<std::int32_t, 4>, 2> addresses{initial, initial}; addresses[source] = sample;
			std::array<words, 24> expected{}; expected[2 + bank] = std::bit_cast<words>(seed);
			for (unsigned i = 0; i < 16; ++i) expected[7 + i] = {0, 0, 0, 0x3f800000};
			for (unsigned lane = 0; lane < 4; ++lane)
			{
				const bool enabled = !predicate || (predicate == 1 ? seed[lane] > 0 : seed[lane] < 0);
				if (!enabled) continue;
				expected[4][lane] = std::bit_cast<unsigned>(seed[lane]);
				if (mask & (8u >> lane))
				{
					addresses[destination][lane] = sum[lane];
					if (update == 1 && valid) expected[2 + bank][lane] = std::bit_cast<unsigned>(float(sum[lane]));
				}
			}
			expected[0] = std::bit_cast<words>(addresses[0]); expected[1] = std::bit_cast<words>(addresses[1]);
			const auto index = std::int64_t(4) + addresses[destination][component]; const bool in_range = index >= 0 && index < 16;
			if (in_range) { expected[5] = expected[6] = std::bit_cast<words>(constants[index]); expected[7 + index] = std::bit_cast<words>(seed); }
			else expected[6] = {0, 0, 0, 0x3f800000};
			expected[23].fill((valid ? 0u : 32u) | (in_range ? 0u : 1u | 2u | 128u));
			const auto input = make_upload(ctx, std::as_bytes(std::span(constants)));
			ctx.begin(); bindings resources; resources.set_buffer(0, input); resources.set_buffer(1, uniforms); resources.set_buffer(2, samplers); resources.set_buffer(3, output);
			const auto table = resources.snapshot(ctx); const auto encoder = ctx.compute(); ctx.keep_alive(pipeline);
			[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
			[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; ctx.submit().wait();
			require(!std::memcmp(output.contents, expected.data(), sizeof(expected)), "ARA registers or consumers differ from independent integer oracle");
			++profiles; overflows += !valid; invalid_consumers += !in_range;
		}
		++variants;
	}
	require(variants == 288 && profiles == 3456 && overflows == 1152, "ARA test matrix is incomplete");
	std::cout << "Native ARA: " << variants << " MSL4 variants, " << profiles << " GPU profiles, " << profiles * 96
		<< " exact independent integer-oracle words; both source/destination/condition banks, aliasing, masks/predicates/condition selection, INT_MAX and beyond-float precision; "
		<< overflows << " guarded overflows, " << invalid_consumers << " guarded consumer faults, all three indexed consumers and 12 cache keys passed\n";
}
}

int main(int argc, char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--float-add-control")) throw std::invalid_argument("Unknown ARA test selection");
			run(argc == 2); return 0;
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
