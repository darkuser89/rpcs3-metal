#include "M4ShaderCache.h"
#include "M4VertexShader.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
using namespace metal4;
using words = std::array<std::uint32_t, 4>;
using vector = std::array<float, 4>;
using operands = std::array<vector, 2>;
constexpr std::array opcodes{11u, 12u, 16u, 18u, 19u, 20u};
constexpr std::array modes{1u, 6u, 2u, 4u, 3u, 5u};
constexpr std::array selection_opcodes{9u, 10u, 14u, 15u, 22u};
constexpr std::array permutations{std::array{0u, 1u, 2u, 3u}, std::array{3u, 2u, 1u, 0u},
	std::array{0u, 1u, 1u, 3u}, std::array{1u, 0u, 2u, 3u}};
constexpr std::array swizzles{0x1bu, 0xe4u, 0x17u, 0x4bu};
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void source(words& w, unsigned index, unsigned packed)
{
	if (!index) { w[1] |= packed >> 9; w[2] |= (packed & 511u) << 23; }
	else if (index == 1) w[2] |= packed << 6;
	else { w[2] |= packed >> 11; w[3] |= (packed & 2047u) << 21; }
}
words move(unsigned constant, unsigned temporary, unsigned output, unsigned flags = 0)
{
	words w{flags | (temporary << 15) | (1u << 30) | (7u << 10), (1u << 22) | (constant << 12), 0,
		(15u << 13) | (63u << 7) | (output << 2)};
	source(w, 0, 0x1b03); return w;
}
float reference(unsigned opcode, float left, float right)
{
	// NV vertex-program comparison semantics propagate NaNs. Double
	// comparisons provide an independent oracle for every finite bit pattern.
	if (std::isnan(left)) return left;
	if (opcode != 14 && opcode != 15 && opcode != 22 && std::isnan(right)) return right;
	const double a = left, b = right;
	switch (opcode)
	{
	case 9:
	case 10:
		if (a == 0 && b == 0)
		{
			// Commutative zero choice: MIN picks -0 if either input is
			// negative; MAX picks -0 only when both inputs are negative.
			const bool negative = opcode == 9 ? std::signbit(left) || std::signbit(right) : std::signbit(left) && std::signbit(right);
			return std::copysign(0.f,negative ? -1.f : 1.f);
		}
		return float(opcode == 9 ? std::min(a,b) : std::max(a,b));
	case 11: return a < b;
	case 12: return a >= b;
	case 14:
		if (std::isinf(a)) return std::numeric_limits<float>::quiet_NaN();
		// FRC's documented range excludes1, including a subtraction that
		// rounds to1 for a tiny negative input. This is the renderer policy.
		return std::clamp(float(a-std::floor(a)),0.f,std::nextafter(1.f,0.f));
	case 15: return float(std::floor(a));
	case 16: return a == b;
	case 18: return a > b;
	case 19: return a <= b;
	case 20: return a != b;
	case 22: return a < 0 ? -1.f : a > 0 ? 1.f : 0.f;
	default: throw std::runtime_error("Invalid comparison oracle opcode");
	}
}
bool same(float actual, float expected)
{
	return std::isnan(expected) ? std::isnan(actual) : std::bit_cast<unsigned>(actual) == std::bit_cast<unsigned>(expected);
}
id<MTLComputePipelineState> pipeline(compiler& compile, std::string text)
{
	const auto library = compile.library(text, @"Native vertex comparisons");
	const auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(library, @"inspect_comparisons");
	return compile.compute(descriptor);
}
void program_comparisons(unsigned previous, bool selection)
{
	const float tiny = std::bit_cast<float>(1u), next = std::bit_cast<float>(2u);
	const float largest_subnormal = std::bit_cast<float>(0x7fffffu), normal = 0x1p-126f;
	const float inf = std::numeric_limits<float>::infinity(), nan = std::bit_cast<float>(0x7fc12345u);
	const std::array inputs{
		operands{vector{tiny,-tiny,tiny,-tiny},vector{0,0,next,-next}},
		operands{vector{0,-0.f,-0.f,0},vector{-0.f,0,-0.f,0}},
		operands{vector{nan,1,inf,-inf},vector{1,nan,inf,-inf}},
		operands{vector{largest_subnormal,-largest_subnormal,normal,-normal},vector{normal,-normal,largest_subnormal,-largest_subnormal}},
		operands{vector{1,-1,.5f,-.5f},vector{1,1,1,-1}},
		operands{vector{inf,-inf,0,-0.f},vector{-inf,inf,inf,-inf}},
		operands{vector{nan,-nan,nan,-nan},vector{nan,nan,-nan,-nan}},
		operands{vector{std::bit_cast<float>(0x7f812345u),std::bit_cast<float>(0xff812345u),0,1},vector{0,1,nan,-nan}},
		operands{vector{tiny,next,-tiny,-next},vector{next,tiny,-next,-tiny}},
		operands{vector{largest_subnormal,-largest_subnormal,normal,-normal},vector{0,-0.f,0,-0.f}},
		operands{vector{std::nextafter(1.f,0.f),std::nextafter(1.f,2.f),-1,1},vector{1,1,std::nextafter(-1.f,0.f),std::nextafter(1.f,2.f)}},
		operands{vector{std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),inf,-inf},vector{inf,-inf,std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()}},
		operands{vector{0,-0.f,tiny,-tiny},vector{tiny,-tiny,0,-0.f}},
		operands{vector{normal,-normal,largest_subnormal,-largest_subnormal},vector{normal,-normal,largest_subnormal,-largest_subnormal}},
		operands{vector{.1f,.2f,.3f,.4f},vector{.4f,.3f,.2f,.1f}},
		operands{vector{-7.75f,64,-64,128},vector{-7.5f,128,-128,64}},
		operands{vector{nan,inf,-inf,0},vector{inf,nan,nan,nan}},
		operands{vector{0x1p-64f,-0x1p-64f,0x1p64f,-0x1p64f},vector{0x1p64f,-0x1p64f,0x1p-64f,-0x1p-64f}}};
	const vector seed{-1,0,1,nan}, marker{5,6,7,8};
	std::vector<vector> constants;
	for (const auto& input : inputs) for (const auto& v : {input[0],input[1],seed,marker}) constants.push_back(v);
	context ctx; compiler compile(ctx); shader_cache cache(compile);
	vertex_context state; state.constant_count = 4;
	const auto uniforms = make_upload(ctx, std::as_bytes(std::span{&state,1}));
	const auto data = make_upload(ctx, std::as_bytes(std::span(constants)));
	const std::array<sampler_parameters,4> sampling{};
	const auto samplers = make_upload(ctx, std::as_bytes(std::span(sampling)));
	const auto output = make_buffer(ctx, inputs.size()*7*sizeof(words), buffer_storage::shared);
	std::array<std::shared_ptr<const vertex_shader>,12> keys{};
	unsigned variants=0,profiles=0,exact=0,unordered=0,branches=0;
	const std::span<const unsigned> selected = selection ? std::span<const unsigned>(selection_opcodes) : std::span<const unsigned>(opcodes);
	for (unsigned opcode : selected) for (unsigned permutation=0;permutation<4;++permutation)
	for (unsigned mask : {0u,5u,10u,15u}) for (unsigned bank=0;bank<2;++bank)
	for (unsigned predicate=0;predicate<3;++predicate) for (bool sat : {false,true})
	@autoreleasepool
	{
		if (previous && previous != opcode) continue;
		const unsigned mode=variants%3, modifier=(variants/3)%4;
		words math{(5u<<15)|(1u<<30)|(1u<<14)|(unsigned(mode!=2)<<29)|(bank<<25)|
			(unsigned(sat)<<26)|(unsigned(predicate!=0)<<13)|(0x1bu<<2)|
			((predicate==1?4u:predicate==2?1u:7u)<<10)|(unsigned(modifier>=2)<<21),
			(opcode<<22)|(unsigned(mode!=0)<<27),0,
			(mask<<13)|((mode?15u:0u)<<17)|(7u<<7)|(1u<<2)};
		source(math,0,(swizzles[permutation]<<8)|((modifier&1u)<<16)|3u);
		source(math,1,(swizzles[3-permutation]<<8)|(6u<<2)|1u);
		if (mode) source(math,2,0x1b03);
		const std::array code{move(1,6,31),move(2,4,31,(1u<<14)|(1u<<29)|(bank<<25)),move(3,5,31),math,
			words{(bank<<25)|(2u<<10),9u<<27,0,(6u<<29)|(31u<<2)},move(3,63,2),words{0,0,0,1u|(31u<<2)}};
		if (variants < keys.size())
		{
			const auto key=cache.vertex(code,{});require(cache.vertex(code,{})==key && std::find(keys.begin(),keys.end(),key)==keys.end(),"Comparison cache key collision");keys[variants]=key;
		}
		auto generated=generate_vertex_program(code);
		if (previous && selection)
		{
			const std::string helper=opcode==9||opcode==10?"rsx_extremum":opcode==14?"rsx_fraction":opcode==15?"rsx_floor":"rsx_sign";
			const std::string next=opcode==9||opcode==10?"inline float4 rsx_floor(":opcode==14?"inline float4 rsx_sign(":opcode==15?"inline float4 rsx_fraction(":"inline int4 rsx_address(";
			const auto begin=generated.source.find("inline float4 "+helper+"("),end=generated.source.find(next,begin);
			require(begin!=std::string::npos&&end!=std::string::npos,"Selection counterproof missed helper");
			const std::string former=opcode==9||opcode==10?"inline float4 rsx_extremum(float4 a,float4 b,bool maximum) { return maximum?max(a,b):min(a,b); }\n":
				"inline float4 "+helper+"(float4 value) { return "+(opcode==14?"fract":opcode==15?"floor":"sign")+"(value); }\n";
			generated.source.replace(begin,end-begin,former);
		}
		else if (previous)
		{
			const auto begin=generated.source.find("inline float4 rsx_set_compare("),end=generated.source.find("inline bool4 rsx_condition(",begin);
			require(begin!=std::string::npos && end!=std::string::npos,"Comparison counterproof missed helper");
			generated.source.replace(begin,end-begin,"inline float4 rsx_set_compare(float4 a,float4 b,uint mode) { switch(mode) {case 1:return float4(a<b);case 2:return float4(a==b);case 3:return float4(a<=b);case 4:return float4(a>b);case 5:return float4(a!=b);case 6:return float4(a>=b);default:return float4(0);} }\n");
		}
		generated.source+=R"metal(
kernel void inspect_comparisons(device const float4* constants [[buffer(0)]],constant rsx_vertex_context& ctx [[buffer(1)]],
    constant rsx_sampler_parameters* samplers [[buffer(2)]],device uint4* out [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    const auto r=rsx_evaluate(constants+id*4,constants+id*4,ctx,0,samplers);
    out[id*7]=as_type<uint4>(r.dst[1]);out[id*7+1]=as_type<uint4>(r.temp[5]);
    out[id*7+2]=as_type<uint4>(r.condition[0]);out[id*7+3]=as_type<uint4>(r.condition[1]);
    out[id*7+4]=as_type<uint4>(r.temp[7]);out[id*7+5]=as_type<uint4>(r.dst[2]);out[id*7+6]=uint4(r.fault);
}
)metal";
		const auto compiled=pipeline(compile,generated.source);
		ctx.begin();bindings resources;resources.set_buffer(0,data);resources.set_buffer(1,uniforms);resources.set_buffer(2,samplers);resources.set_buffer(3,output);
		const auto table=resources.snapshot(ctx);const auto encoder=ctx.compute();ctx.keep_alive(compiled);
		[encoder setComputePipelineState:compiled];[encoder setArgumentTable:table];
		[encoder dispatchThreads:MTLSizeMake(inputs.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];ctx.submit().wait();
		const auto* raw=static_cast<const unsigned*>(output.contents);
		for (unsigned i=0;i<inputs.size();++i)
		{
			const auto& input=inputs[i];std::array<vector,7> expected{};expected[0]=expected[5]={0,0,0,1};expected[1]=marker;expected[2+bank]=seed;
			const auto saturate=[sat](float v){return sat && v<0 ? 0.f : sat && v>1 ? 1.f : v;};
			for (unsigned lane=0;lane<4;++lane)
			{
				if (predicate && !(predicate==1 ? seed[lane]>0 : seed[lane]<0)) continue;
				if (mode) { expected[4][lane]=saturate(input[0][lane]);if (mode==2) expected[2+bank][lane]=expected[4][lane]; }
				if (!(mask&(8u>>lane))) continue;
				float a=input[0][permutations[permutation][lane]],b=input[1][permutations[3-permutation][lane]];
				if (modifier>=2) a=std::abs(a);if (modifier&1u) a=-a;
				const float result=saturate(reference(opcode,a,b));expected[0][lane]=expected[1][lane]=result;
				if (mode!=2) expected[2+bank][lane]=result;
			}
			const bool taken=expected[2+bank][0]==0; if (!taken) expected[5]=marker;branches+=taken;
			for (unsigned reg=0;reg<7;++reg) for (unsigned lane=0;lane<4;++lane)
			{
				const float actual=std::bit_cast<float>(raw[(i*7+reg)*4+lane]),wanted=expected[reg][lane];
				if (!same(actual,wanted))
				{
					std::cerr<<"opcode="<<opcode<<" permutation="<<permutation<<" mask="<<mask<<" bank="<<bank<<" predicate="<<predicate<<" sat="<<sat<<" mode="<<mode<<" modifier="<<modifier<<" input="<<i<<" register="<<reg<<" lane="<<lane<<" actual="<<std::hex<<std::bit_cast<unsigned>(actual)<<" wanted="<<std::bit_cast<unsigned>(wanted)<<std::dec<<'\n';
					throw std::runtime_error("Vertex comparison differs from independent oracle");
				}
				if (std::isnan(wanted)) ++unordered;else ++exact;
			}
			++profiles;
		}
		++variants;
	}
	require(variants==(previous?192u:selection?960u:1152u) && profiles==variants*inputs.size() && exact+unordered==profiles*28,"Comparison matrix incomplete");
	std::cout<<(selection?"Native vertex selection/rounding: ":"Native vertex comparisons: ")<<variants<<" MSL4 variants, "<<profiles<<" GPU profiles, "<<exact<<" exact words, "<<unordered<<" NaN classifications, "<<branches<<" taken branches; both condition banks, masks, predicates/SAT, paired result selection, source swizzles/modifiers and12 cache keys passed\n";
}
void raw_comparisons(bool selection)
{
	constexpr std::array<unsigned,20> values{0u,0x80000000u,1u,2u,0x7fffffu,0x800000u,0x80000001u,0x80000002u,0x807fffffu,0x80800000u,
		0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u,0x7f812345u,0xff812345u};
	std::vector<std::array<words,2>> inputs;
	for (unsigned a=0;a<values.size();++a) for (unsigned b=0;b<values.size();++b)
	{
		std::array<words,2> pair;for (unsigned lane=0;lane<4;++lane) { pair[0][lane]=values[(a+lane)%values.size()];pair[1][lane]=values[(b+lane*3)%values.size()]; }inputs.push_back(pair);
	}
	unsigned random=0x63757269;
	for (unsigned i=0;i<512;++i) { std::array<words,2> pair;for (auto& v:pair) for(auto& word:v) { random=random*1664525u+1013904223u;word=random; }inputs.push_back(pair); }
	context ctx;compiler compile(ctx);
	const std::array<words,1> code{move(0,63,0)};auto generated=generate_vertex_program(code);
	const std::span<const unsigned> selected = selection ? std::span<const unsigned>(selection_opcodes) : std::span<const unsigned>(opcodes);
	if (selection) generated.source+=R"metal(
kernel void inspect_comparisons(device const float4* values [[buffer(0)]],device uint4* out [[buffer(1)]],uint id [[thread_position_in_grid]]) {
    const float4 a=values[id*2],b=values[id*2+1];
    out[id*5]=as_type<uint4>(rsx_extremum(a,b,false));out[id*5+1]=as_type<uint4>(rsx_extremum(a,b,true));
    out[id*5+2]=as_type<uint4>(rsx_fraction(a));out[id*5+3]=as_type<uint4>(rsx_floor(a));out[id*5+4]=as_type<uint4>(rsx_sign(a));
}
)metal";
	else generated.source+=R"metal(
kernel void inspect_comparisons(device const float4* values [[buffer(0)]],device uint4* out [[buffer(1)]],uint id [[thread_position_in_grid]]) {
    const uint modes[6]={1u,6u,2u,4u,3u,5u};
    for(uint i=0;i<6;++i) out[id*6+i]=as_type<uint4>(rsx_set_compare(values[id*2],values[id*2+1],modes[i]));
}
)metal";
	const auto compiled=pipeline(compile,generated.source);const auto data=make_upload(ctx,std::as_bytes(std::span(inputs)));
	const auto output=make_buffer(ctx,inputs.size()*selected.size()*sizeof(words),buffer_storage::shared);
	ctx.begin();bindings resources;resources.set_buffer(0,data);resources.set_buffer(1,output);const auto table=resources.snapshot(ctx);const auto encoder=ctx.compute();ctx.keep_alive(compiled);
	[encoder setComputePipelineState:compiled];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(inputs.size(),1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];ctx.submit().wait();
	const auto* raw=static_cast<const unsigned*>(output.contents);unsigned exact=0,unordered=0;
	for (unsigned i=0;i<inputs.size();++i) for (unsigned op=0;op<selected.size();++op) for (unsigned lane=0;lane<4;++lane)
	{
		const float expected=reference(selected[op],std::bit_cast<float>(inputs[i][0][lane]),std::bit_cast<float>(inputs[i][1][lane]));
		require(same(std::bit_cast<float>(raw[(i*selected.size()+op)*4+lane]),expected),"Raw vertex comparison differs from independent oracle");
		if(std::isnan(expected))++unordered;else ++exact;
	}
	require(inputs.size()==912 && exact+unordered==inputs.size()*selected.size()*4,"Raw comparison matrix incomplete");
	std::cout<<(selection?"Raw vertex selection/rounding: ":"Raw vertex comparisons: ")<<"912 vector pairs, "<<exact<<" exact words, "<<unordered<<" NaN classifications; complete20x20edge-class pairs per lane and512 deterministic random pairs passed\n";
}
}
int main(int argc,char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			unsigned previous=0;bool selection=false;
			if(argc==3 && std::string_view(argv[1])=="--previous-comparison-control") { previous=std::stoul(argv[2]);require(std::find(opcodes.begin(),opcodes.end(),previous)!=opcodes.end(),"Unknown comparison control opcode"); }
			else if(argc==3 && std::string_view(argv[1])=="--previous-selection-control") { selection=true;previous=std::stoul(argv[2]);require(std::find(selection_opcodes.begin(),selection_opcodes.end(),previous)!=selection_opcodes.end(),"Unknown selection control opcode"); }
			else if(argc==2 && std::string_view(argv[1])=="--selection-rounding") selection=true;
			else require(argc==1,"Unknown comparison arguments");
			program_comparisons(previous,selection);if(!previous)raw_comparisons(selection);return 0;
		}
		catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
	}
}
