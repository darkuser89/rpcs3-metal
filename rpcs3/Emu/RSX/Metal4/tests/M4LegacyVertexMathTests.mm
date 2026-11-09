#include "M4ShaderCache.h"
#include "M4VertexShader.h"
#include "M4VertexRuntime.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace
{
using namespace metal4;
using words = std::array<std::uint32_t, 4>;
using vector = std::array<float, 4>;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void source(words& w, unsigned index, unsigned packed)
{
	if (!index) { w[1] |= packed >> 9; w[2] |= (packed & 511u) << 23; }
	else { w[2] |= packed >> 11; w[3] |= (packed & 2047u) << 21; }
}
words move(unsigned constant, unsigned temporary, unsigned output, unsigned flags = 0)
{
	words w{flags | (temporary << 15) | (1u << 30), (1u << 22) | (constant << 12), 0,
		(15u << 13) | (63u << 7) | (output << 2)};
	source(w, 0, 0x1b03); return w;
}
vector reference(unsigned opcode, float value)
{
	const float inf = std::numeric_limits<float>::infinity();
	if (opcode == 2 || opcode == 3 || opcode == 4 || opcode == 13 || opcode == 14)
	{
		double result;
		if (opcode == 14) result = std::exp2(double(value));
		else if (opcode == 4) result = 1.0 / std::sqrt(std::abs(double(value)));
		else if (opcode == 13) result = std::log2(double(value));
		else
		{
			result = 1.0 / double(value);
			if (opcode == 3 && !std::isnan(result)) result = std::copysign(std::clamp(std::abs(result),0x1p-64,0x1p64),result);
		}
		return {float(result),float(result),float(result),float(result)};
	}
	if (std::isnan(value)) return {value, value, value, 1};
	if (opcode == 5)
	{
		if (std::isinf(value)) return value > 0 ? vector{inf, 0, inf, 1} : vector{0, 0, 0, 1};
		const double integral = std::floor(double(value));
		return {float(std::exp2(integral)), float(double(value) - integral), float(std::exp2(double(value))), 1};
	}
	const double magnitude = std::abs(double(value));
	if (!magnitude) return {-inf, 1, -inf, 1};
	if (std::isinf(magnitude)) return {inf, 1, inf, 1};
	int exponent;
	const double mantissa = std::frexp(magnitude, &exponent);
	return {float(exponent - 1), float(mantissa * 2), float(std::log2(magnitude)), 1};
}
vector light_reference(const vector& value)
{
	double x = value[0], y = value[1], exponent = value[3];
	if (x <= 0) x = 0; if (y <= 0) y = 0;
	exponent = std::clamp(exponent,-127.99609375,127.99609375);
	return {1,float(x),x > 0 ? float(std::pow(std::abs(y),exponent)) : 0.f,1};
}

void register_classification()
{
	std::vector<words> inputs{{0xff800000,0xbf800000,0x80800000,0x80000001},
		{0x807fffff,0x80000000,0,1},{0x7fffff,0x800000,0x3f800000,0x7f800000},
		{0x7fc12345,0xffc12345,0x7f812345,0xff812345},{0xff7fffff,0x7f7fffff,0x40600000,0xc0600000}};
	unsigned random = 0x6e76216d;
	for (unsigned i = 0; i < 512; ++i)
	{
		words value; for (auto& word : value) { random = random * 1664525u + 1013904223u; word = random; } inputs.push_back(value);
	}
	context ctx; compiler compile(ctx);
	const auto data = make_upload(ctx, std::as_bytes(std::span(inputs)));
	const auto output = make_buffer(ctx, inputs.size() * 9 * sizeof(words), buffer_storage::shared);
	const auto library = compile.library(std::string(vertex_runtime) + R"metal(
kernel void inspect_registers(device const uint4* values [[buffer(0)]], device uint4* out [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    const float4 value = as_type<float4>(values[id]);
    for(uint mode=0;mode<8;++mode) out[id*9+mode] = uint4(rsx_condition(value,mode));
    out[id*9+8] = as_type<uint4>(rsx_saturate_ordered(value));
}
)metal", @"Native predicate and ordered SAT raw-register matrix");
	const auto descriptor = [MTL4ComputePipelineDescriptor new]; descriptor.computeFunctionDescriptor = library_function(library, @"inspect_registers");
	const auto pipeline = compile.compute(descriptor);
	ctx.begin(); bindings resources; resources.set_buffer(0, data); resources.set_buffer(1, output);
	const auto table = resources.snapshot(ctx); const auto encoder = ctx.compute(); ctx.keep_alive(pipeline);
	[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(inputs.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; ctx.submit().wait();
	const auto* actual = static_cast<const unsigned*>(output.contents);
	for (unsigned i = 0; i < inputs.size(); ++i) for (unsigned lane = 0; lane < 4; ++lane)
	{
		// Double comparisons are independent of the shader's integer classifiers.
		const double value = std::bit_cast<float>(inputs[i][lane]);
		const std::array expected{false,value<0,value==0,value<=0,value>0,value!=0,value>=0,true};
		for (unsigned mode = 0; mode < 8; ++mode)
			require(actual[(i*9+mode)*4+lane] == unsigned(expected[mode]), "Native condition classifier differs from independent double comparisons");
		const auto saturated = value < 0 ? 0u : value > 1 ? 0x3f800000u : inputs[i][lane];
		require(actual[(i*9+8)*4+lane] == saturated, "Native ordered SAT changed a raw register value");
	}
	std::cout << "Raw vertex registers: 517 vectors, 16544 independent predicate booleans across all 8 modes and 2068 exact ordered SAT words, signed zero/subnormals/infinities/quiet-signaling NaNs and 512 deterministic random vectors passed\n";
}

void run(unsigned control)
{
	const bool lighting = control >= 19, ex2 = control == 17 || control == 18;
	const bool scalar = control >= 7 && !lighting;
	const float nan = std::bit_cast<float>(0x7fc12345u), inf = std::numeric_limits<float>::infinity();
	const std::array inputs{
		vector{-.75f, 1.5f, 2.5f, -3.25f}, vector{0, -0.f, 1, 2}, vector{4, 8, .25f, .5f},
		vector{.1f, .2f, .3f, .4f}, vector{std::nextafter(1.f, 0.f), std::nextafter(1.f, 2.f), 65536, 0x1p-64f},
		vector{-.5f, 64, -64, 128},
		vector{std::bit_cast<float>(1u), std::bit_cast<float>(0x7fffffu), 0x1p-126f, std::numeric_limits<float>::max()},
		vector{std::bit_cast<float>(0x80000001u), -0.f, -0x1p-126f, -1},
		vector{inf, -inf, nan, std::bit_cast<float>(0xffc12345u)},
		vector{-std::numeric_limits<float>::max(), -7.75f, 16, -16},
		vector{-126.5f,-127.25f,-148.5f,-150}, vector{-149,-149.5f,-150.5f,-125.5f},
		vector{std::nextafter(-126.f,-127.f),std::nextafter(-126.f,-125.f),std::nextafter(-150.f,-149.f),std::nextafter(-150.f,-151.f)},
		vector{0x1p-64f,std::nextafter(0x1p-64f,0.f),std::nextafter(0x1p-64f,1.f),0x1p64f},
		vector{std::nextafter(0x1p64f,0.f),std::nextafter(0x1p64f,inf),-0x1p-64f,-0x1p64f},
		vector{0x1p-128f,0x1.8p-128f,0x1p126f,0x1p127f},vector{0x1p-149f,0x1p-148f,0x1p-147f,0x1p-146f},
		vector{.3f,3,1.333f,-3}};
	const float tiny = std::bit_cast<float>(1u), maximum = std::numeric_limits<float>::max();
	const std::array light_inputs{
		vector{1,0,19,1},vector{1,0,-7,-1},vector{1,0,nan,0},vector{1,-2,inf,1},
		vector{1,.5f,nan,256},vector{1,2,nan,-256},vector{1,2,0,256},vector{1,.5f,0,-256},
		vector{tiny,2,nan,2},vector{-tiny,2,0,2},vector{nan,.5f,0,2},vector{1,nan,0,0},
		vector{1,nan,0,1},vector{1,1,nan,nan},vector{1,inf,0,2},vector{1,inf,0,-2},
		vector{1,tiny,0,1},vector{1,tiny,0,.5f},vector{1,tiny,0,-.5f},vector{1,maximum,0,1},
		vector{1,maximum,0,-1},vector{1,-tiny,0,-1},vector{-0.f,.5f,inf,128},vector{1,.75f,nan,127.99609375f},
		vector{1,2,0,inf},vector{1,2,0,-inf},vector{1,0,0,tiny},vector{1,0,0,-tiny},vector{inf,.5f,0,2},vector{1,-inf,0,1}};
	const std::span<const vector> selected_inputs = lighting ? std::span<const vector>(light_inputs) : std::span<const vector>(inputs).first(scalar ? inputs.size() : 13);
	constexpr std::array swizzles{0x1bu,0xe4u,0x17u,0x4bu};
	constexpr std::array permutations{std::array{0u,1u,2u,3u},std::array{3u,2u,1u,0u},std::array{0u,1u,1u,3u},std::array{1u,0u,2u,3u}};
	const vector seed{-1, 0, 1, nan}, marker{5, 6, 7, 8};
	context ctx; compiler compile(ctx); shader_cache cache(compile);
	vertex_context state; state.constant_count = 3;
	const auto uniforms = make_upload(ctx, std::as_bytes(std::span{&state, 1}));
	const std::array<sampler_parameters, 4> sampling{};
	const auto samplers = make_upload(ctx, std::as_bytes(std::span(sampling)));
	const auto output = make_buffer(ctx, 7 * sizeof(words), buffer_storage::shared);
	std::array<std::shared_ptr<const vertex_shader>, 12> keys{};
	unsigned variants = 0, profiles = 0, branches = 0, exact = 0, approximate = 0, unordered = 0;
	float maximum_error = 0;
	for (unsigned opcode : {2u,3u,4u,5u,6u,7u,13u,14u}) for (unsigned component = 0; component < 4; ++component)
	for (unsigned mask : {0u, 5u, 10u, 15u}) for (unsigned bank = 0; bank < 2; ++bank)
	for (unsigned predicate = 0; predicate < 3; ++predicate) for (bool sat : {false, true})
	@autoreleasepool
	{
		if (lighting ? opcode != 7 : ex2 ? opcode != 14 : scalar ? (opcode != 2 && opcode != 3 && opcode != 4 && opcode != 13) : (opcode != 5 && opcode != 6)) continue;
		if (((control == 1 || control == 3) && opcode != 6) || (control == 2 && opcode != 5)) continue;
		if ((control == 8 || control == 12) && opcode != 3) continue;
		if ((control == 9 && opcode != 4) || (control == 10 && opcode != 13) || (control == 11 && opcode != 2)) continue;
		if ((control == 13 && opcode != 2) || (control == 14 && opcode != 3) || (control == 15 && opcode != 4) || (control == 16 && opcode != 13)) continue;
		const unsigned mode = variants % 3, modifier = (variants / 3) % 4;
		const unsigned condition = (1u << 14) | (bank << 25);
		words math{((mode ? 4u : 63u) << 15) | condition | (unsigned(mode == 2) << 29) |
			(unsigned(sat) << 26) | (unsigned(predicate != 0) << 13) | (0x1bu << 2) |
			((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10) | (unsigned(modifier >= 2) << 23),
			(opcode << 27) | (unsigned(mode != 0) << 22), 0,
			(mask << 17) | (15u << 13) | (5u << 7) | (1u << 2)};
		source(math, 2, ((lighting ? swizzles[component] : component * 0x55u) << 8) | ((modifier & 1u) << 16) | 3u);
		if (mode) source(math, 0, 0x1b03);
		const std::array code{move(1, 4, 31, condition | (1u << 29)), move(2, 5, 31), math,
			words{(bank << 25) | (2u << 10), 9u << 27, 0, (5u << 29) | (31u << 2)},
			move(2, 63, 2), words{0, 0, 0, 1u | (31u << 2)}};
		if (variants < keys.size())
		{
			const auto key = cache.vertex(code, {});
			require(cache.vertex(code, {}) == key && std::find(keys.begin(), keys.end(), key) == keys.end(), "Legacy math shader keys collided"); keys[variants] = key;
		}
		auto generated = generate_vertex_program(code);
		if (control == 2 || control == 3)
		{
			// Reintroduce exactly one former natural-function result while
			// retaining current decoding, SAT, routing and the independent oracle.
			const bool exp = control == 2;
			const std::string helper = exp ? "rsx_exp_legacy" : "rsx_log_legacy";
			const auto start = generated.source.find("inline float4 " + helper + "(");
			const auto end = generated.source.find(exp ? "inline float4 rsx_saturate_ordered(" : "inline float rsx_reciprocal(", start);
			require(start != std::string::npos && end != std::string::npos, "Legacy numeric control missed its helper");
			generated.source.replace(start, end - start, "inline float4 " + helper + "(float value) { return float4(" + (exp ? "exp" : "log") + "(value)); }\n");
		}
		else if (control == 4 || control == 5)
		{
			const bool sat_control = control == 4;
			const auto start = generated.source.find(sat_control ? "inline float4 rsx_saturate_ordered(" : "inline bool4 rsx_condition(");
			const auto end = generated.source.find(sat_control ? "inline float rsx_normalized_mantissa(" : "inline void rsx_store_output(", start);
			require(start != std::string::npos && end != std::string::npos, "Legacy subnormal control missed its helper");
			const std::string previous = sat_control ?
				"inline float4 rsx_saturate_ordered(float4 value) { return select(select(value,float4(0),value<0),float4(1),value>1); }\n" :
				"inline bool4 rsx_condition(float4 value,uint mode) { switch(mode) { case 1:return value<0;case 2:return value==0;case 3:return value<=0;case 4:return value>0;case 5:return value!=0;case 6:return value>=0;case 7:return bool4(true);default:return bool4(false); } }\n";
			generated.source.replace(start, end - start, previous);
		}
		else if (control == 6)
		{
			const auto start = generated.source.find("inline float rsx_power2_integral(");
			const auto end = generated.source.find("inline float rsx_scale_positive(",start);
			require(start != std::string::npos && end != std::string::npos, "EXP underflow control missed its helper");
			generated.source.replace(start,end-start,"inline float rsx_power2_integral(float exponent) { return exp2(exponent); }\n");
		}
		else if (control >= 8 && control <= 12)
		{
			const std::string helper = control == 9 ? "rsx_rsqrt" : control == 10 ? "rsx_log2_scalar" : control == 11 ? "rsx_reciprocal" : "rsx_rcp_clamped";
			const std::string next = control == 9 ? "inline float rsx_log2_scalar(" : control == 10 ? "inline float rsx_exp2_scalar(" : control == 11 ? "inline float rsx_rcp_clamped(" : "inline float rsx_rsqrt(";
			const auto start = generated.source.find("inline float " + helper + "("), end = generated.source.find(next,start);
			require(start != std::string::npos && end != std::string::npos,"Scalar control missed its helper");
			const std::string expression = control == 9 ? "1.0f/sqrt(max(value,1e-10f))" : control == 10 ? "log2(max(value,1e-10f))" : control == 11 ? "1.0f/value" : control == 12 ? "copysign(clamp(abs(1.0f/value),5.42101e-20f,1.884467e19f),value)" : "clamp(1.0f/value,5.42101e-20f,1.884467e19f)";
			generated.source.replace(start,end-start,"inline float " + helper + "(float value) { return " + expression + "; }\n");
		}
		else if (control == 18 || control == 20)
		{
			const auto start = generated.source.find(control == 18 ? "inline float rsx_exp2_scalar(" : "inline float4 rsx_lit(");
			const auto end = generated.source.find(control == 18 ? "inline float4 rsx_lit(" : "inline float4 rsx_zclip(",start);
			require(start != std::string::npos && end != std::string::npos,"EX2/LIT control missed helper");
			generated.source.replace(start,end-start,control == 18 ? "inline float rsx_exp2_scalar(float value) { return exp2(value); }\n" : "inline float4 rsx_lit(float4 value) { const float x=max(value.x,0.0f),y=max(value.y,1e-10f);return float4(1,x,x>0?exp2(value.w*log2(y)):0,1); }\n");
		}
		else if (control >= 21)
		{
			const std::string needle = control == 21 ? "const float limit = 127.99609375f;" : control == 22 ? "const float base = rsx_condition(float4(value.y), 3u).x ? 0.0f : value.y;" : "if (!rsx_condition(float4(diffuse), 4u).x)";
			const std::string replacement = control == 21 ? "const float limit = INFINITY;" : control == 22 ? "const float base = max(value.y,1e-10f);" : "if (!(diffuse > 0.0f))";
			const auto at = generated.source.find(needle);require(at != std::string::npos,"LIT isolated control missed helper line");generated.source.replace(at,needle.size(),replacement);
		}
		generated.source += R"metal(
kernel void inspect_legacy_math(device const float4* constants [[buffer(0)]], constant rsx_vertex_context& ctx [[buffer(1)]],
    constant rsx_sampler_parameters* samplers [[buffer(2)]], device uint4* out [[buffer(3)]]) {
    const auto r = rsx_evaluate(constants, constants, ctx, 0, samplers);
    out[0]=as_type<uint4>(r.dst[1]); out[1]=as_type<uint4>(r.temp[5]);
    out[2]=as_type<uint4>(r.condition[0]); out[3]=as_type<uint4>(r.condition[1]);
    out[4]=as_type<uint4>(r.temp[4]); out[5]=as_type<uint4>(r.dst[2]); out[6]=uint4(r.fault);
}
)metal";
		const auto library = compile.library(generated.source, @"Legacy vertex EXP/LOG components");
		const auto descriptor = [MTL4ComputePipelineDescriptor new]; descriptor.computeFunctionDescriptor = library_function(library, @"inspect_legacy_math");
		const auto pipeline = compile.compute(descriptor);
		for (const auto& input : selected_inputs)
		{
			vector decoded;
			for (unsigned lane=0;lane<4;++lane) { decoded[lane]=input[lighting ? permutations[component][lane] : lane];if(modifier>=2)decoded[lane]=std::abs(decoded[lane]);if(modifier&1u)decoded[lane]=-decoded[lane]; }
			auto result = lighting ? light_reference(decoded) : reference(opcode,decoded[component]);
			const auto saturate = [sat](float v) { if (sat && v < 0) return 0.f; if (sat && v > 1) return 1.f; return v; };
			for (auto& v : result) v = saturate(v);
			const auto approximate_lane = [&](unsigned lane)
			{
				if (lighting) return lane == 2 && result[lane] != 1.f;
				if (!scalar) return lane == 2;
				const float magnitude = std::abs(result[lane]);
				return magnitude != 1.f && !(opcode == 3 && (magnitude == 0x1p-64f || magnitude == 0x1p64f));
			};
			std::array<vector, 7> expected{}; expected[0] = expected[5] = {0, 0, 0, 1}; expected[1] = marker;
			expected[2 + bank] = seed; expected[4] = seed;
			std::array<std::array<bool, 4>, 7> tolerance{};
			for (unsigned lane = 0; lane < 4; ++lane)
			{
				const bool enabled = !predicate || (predicate == 1 ? seed[lane] > 0 : seed[lane] < 0);
				if (!enabled) continue;
				if (mode) { expected[4][lane] = saturate(input[lane]); if (mode == 2) expected[2 + bank][lane] = expected[4][lane]; }
				if (!(mask & (8u >> lane))) continue;
				expected[0][lane] = expected[1][lane] = result[lane]; tolerance[0][lane] = tolerance[1][lane] = approximate_lane(lane);
				if (mode != 2) { expected[2 + bank][lane] = result[lane]; tolerance[2 + bank][lane] = approximate_lane(lane); }
			}
			const bool taken = expected[2 + bank][0] == 0; if (!taken) expected[5] = marker;
			const std::array constants{input, seed, marker}; const auto data = make_upload(ctx, std::as_bytes(std::span(constants)));
			ctx.begin(); bindings resources; resources.set_buffer(0, data); resources.set_buffer(1, uniforms); resources.set_buffer(2, samplers); resources.set_buffer(3, output);
			const auto table = resources.snapshot(ctx); const auto encoder = ctx.compute(); ctx.keep_alive(pipeline);
			[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
			[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; ctx.submit().wait();
			const auto* raw = static_cast<const unsigned*>(output.contents);
			for (unsigned reg = 0; reg < 7; ++reg) for (unsigned lane = 0; lane < 4; ++lane)
			{
				const float wanted = expected[reg][lane], actual = std::bit_cast<float>(raw[reg * 4 + lane]);
				if (std::isnan(wanted)) { require(std::isnan(actual), "Legacy EXP/LOG unordered result differs from independent oracle"); ++unordered; }
				else if (tolerance[reg][lane] && std::isfinite(wanted) && wanted != 0)
				{
					const float error = std::abs(actual - wanted), bound = std::max((lighting ? 0.00005f : 0.000002f) * std::abs(wanted), 2.f * std::numeric_limits<float>::denorm_min()); require(std::isfinite(actual) && error <= bound, "Legacy EXP/LOG approximation differs from independent oracle"); maximum_error = std::max(maximum_error, error); ++approximate;
				}
				else
				{
					if (raw[reg * 4 + lane] != std::bit_cast<unsigned>(wanted))
					{
						std::cerr << "opcode=" << opcode << " component=" << component << " mask=" << mask << " bank=" << bank << " predicate=" << predicate << " sat=" << sat << " mode=" << mode << " modifier=" << modifier << " input=" << input[component] << " register=" << reg << " lane=" << lane << " actual=" << std::hex << raw[reg * 4 + lane] << " wanted=" << std::bit_cast<unsigned>(wanted) << std::dec << '\n';
						throw std::runtime_error("Legacy EXP/LOG component or masked register differs from independent oracle");
					}
					++exact;
				}
			}
			++profiles; branches += taken;
		}
		++variants;
	}
	const unsigned expected_variants = lighting ? 192 : scalar ? (control == 7 ? 768 : 192) : (control && control < 4 ? 192 : 384);
	require(variants == expected_variants && profiles == variants * selected_inputs.size() && exact + approximate + unordered == profiles * 28, "Legacy math matrix is incomplete");
	std::cout << (lighting ? "Native LIT: " : ex2 ? "Native EX2: " : scalar ? "Native scalar RCP/RCC/RSQ/LG2: " : "Native legacy EXP/LOG: ") << variants << " MSL4 variants, " << profiles << " GPU profiles; " << exact << " exact words, " << approximate << " bounded approximation words, " << unordered << " NaN classifications; max absolute approximation error=" << maximum_error
		<< "; scalar source components/modifiers, four masks, predicates/SAT, paired VEC/SCA condition selection and " << branches << " taken branches; 12 cache keys passed\n";
}
}

int main(int argc, char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			unsigned control = 0;
			if (argc == 2)
			{
				const std::string_view selection(argv[1]);
				if (selection == "--log-only") control = 1;
				else if (selection == "--previous-exp-control") control = 2;
				else if (selection == "--previous-log-control") control = 3;
				else if (selection == "--previous-sat-control") control = 4;
				else if (selection == "--previous-condition-control") control = 5;
				else if (selection == "--previous-exp-underflow-control") control = 6;
				else if (selection == "--scalar-fixes") control = 7;
				else if (selection == "--previous-rcc-control") control = 8;
				else if (selection == "--previous-rsq-control") control = 9;
				else if (selection == "--previous-lg2-control") control = 10;
				else if (selection == "--previous-rcp-control") control = 11;
				else if (selection == "--previous-rcc-bounds-control") control = 12;
				else if (selection == "--rcp-only") control = 13;
				else if (selection == "--rcc-only") control = 14;
				else if (selection == "--rsq-only") control = 15;
				else if (selection == "--lg2-only") control = 16;
				else if (selection == "--ex2") control = 17;
				else if (selection == "--previous-ex2-control") control = 18;
				else if (selection == "--lighting") control = 19;
				else if (selection == "--previous-lit-control") control = 20;
				else if (selection == "--previous-lit-exponent-control") control = 21;
				else if (selection == "--previous-lit-floor-control") control = 22;
				else if (selection == "--previous-lit-gate-control") control = 23;
				else throw std::invalid_argument("Unknown legacy math test selection");
			}
			else if (argc != 1) throw std::invalid_argument("Unknown legacy math arguments");
			if (!control) register_classification();
			run(control); return 0;
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
