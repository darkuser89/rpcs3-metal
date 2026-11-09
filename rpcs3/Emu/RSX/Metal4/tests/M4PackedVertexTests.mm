#include "M4VertexShader.h"
#include "M4VertexFetch.h"
#include "M4ConstantUpload.h"
#include "M4HeapBudget.h"
#include "M4ShaderCache.h"
#include "M4RenderPipeline.h"
#include "M4VertexSparseCapture.h"
#include "M4Transfer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <limits>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
void address_rounding()
{
	using words = std::array<std::uint32_t, 4>;
	constexpr unsigned identity = 0x1b00;
	const auto source = [](words& w, unsigned slot, unsigned packed)
	{
		if (!slot) { w[1] |= packed >> 9; w[2] |= (packed & 511u) << 23; }
		else { w[2] |= packed >> 11; w[3] |= (packed & 2047u) << 21; }
	};
	const auto move = [&](unsigned constant, unsigned temporary, unsigned control)
	{
		words w{(temporary << 15) | control | (1u << 30) | (7u << 10),
			(1u << 22) | (constant << 12), 0, (15u << 13) | (63u << 7) | (31u << 2)};
		source(w, 0, identity | 3u); return w;
	};
	const float nan = std::bit_cast<float>(0x7fc12345u), inf = std::numeric_limits<float>::infinity();
	const std::array samples{
		std::array{.5f, 1.5f, 2.5f, 3.5f}, std::array{-.5f, -1.5f, -2.5f, -3.5f},
		std::array{std::nextafter(.5f, 0.f), std::nextafter(.5f, 1.f), std::nextafter(-.5f, 0.f), std::nextafter(-.5f, -1.f)},
		std::array{1.25f, 1.75f, -1.25f, -1.75f}, std::array{0.f, -0.f, 4.5f, 5.5f},
		std::array{-2147483648.f, 2147483520.f, -8388608.5f, 8388607.5f},
		std::array{2147483648.f, 1.f, 2.f, 3.f}, std::array{1.f, -2147483904.f, 2.f, 3.f},
		std::array{inf, 1.f, 2.f, 3.f}, std::array{1.f, -inf, 2.f, 3.f},
		std::array{1.f, 2.f, nan, 3.f}, std::array{1.f, 2.f, 3.f, nan},
		std::array{std::bit_cast<float>(1u), std::bit_cast<float>(0x80000001u), 6.5f, 7.5f},
		std::array{8.5f, 9.5f, 10.5f, 11.5f}, std::array{-4.5f, -5.5f, -6.5f, -7.5f},
		std::array{12.5f, 13.5f, 14.5f, 15.5f}};
	const std::array<float, 4> seed{-1, 0, 1, nan}, initial{11, 12, 13, 14};
	context ctx; compiler compile(ctx); shader_cache cache(compile);
	std::array<std::shared_ptr<const vertex_shader>, 12> keys{};
	vertex_context state; state.constant_count = 16; state.vertex_count = 1;
	const auto uniforms = make_upload(ctx, std::as_bytes(std::span{&state, 1}));
	const std::array<sampler_parameters, 4> sampling{};
	const auto samplers = make_upload(ctx, std::as_bytes(std::span(sampling)));
	const auto output = make_buffer(ctx, 24 * sizeof(words), buffer_storage::shared);
	unsigned variants = 0, profiles = 0, conversion_faults = 0, consumer_faults = 0, kinds = 0;
	for (unsigned address : {0u, 1u}) for (unsigned bank : {0u, 1u})
	for (unsigned mask : {0u, 5u, 10u, 15u}) for (unsigned predicate = 0; predicate < 3; ++predicate)
	for (unsigned update = 0; update < 3; ++update)
	@autoreleasepool
	{
		const unsigned kind = variants % 3, modifier = (variants / 3) % 4;
		const bool reverse = (variants / 12) % 2;
		const unsigned component = (variants / 6) % 4, swizzle = reverse ? 0xe400 : identity;
		const unsigned condition = (1u << 14) | (bank << 25), select_vector = 1u << 29;
		words load{(address << 15) | (1u << 30) | (bank << 25) | (0x1bu << 2) |
			(unsigned(predicate != 0) << 13) | ((predicate == 1 ? 4u : predicate == 2 ? 1u : 7u) << 10) |
			(unsigned(update != 0) << 14) | (unsigned(update == 1) << 29) | (unsigned(modifier >= 2) << 21),
			(23u << 22) | (1u << 27), 0, (mask << 13) | (15u << 17) | (5u << 7) | (31u << 2)};
		source(load, 0, swizzle | (kind == 0 ? 3u : kind == 1 ? 1u | (6u << 2) : 2u) | ((modifier & 1u) << 16));
		source(load, 2, identity | 1u | (4u << 2));
		auto init_address = move(2, address, 0); init_address[1] = (init_address[1] & ~(31u << 22)) | (13u << 22);
		std::vector<words> code{move(0, 6, 0), move(1, 4, condition | select_vector), init_address, load};
		for (unsigned consumer = 0; consumer < 3; ++consumer)
		{
			words w{(1u << 30) | (7u << 10) | (address << 24) | component |
				((consumer == 2 ? 63u : 7u + consumer) << 15), (1u << 22), 0,
				(15u << 13) | (63u << 7) | ((consumer == 2 ? 4u : 31u) << 2)};
			if (!consumer) { w[1] |= 4u << 12; w[3] |= 1u << 1; source(w, 0, identity | 3u); }
			else if (consumer == 1) { w[0] |= 1u << 27; w[1] |= 4u << 8; source(w, 0, identity | 2u); }
			else { w[0] |= 1u << 28; w[3] |= 1u; source(w, 0, identity | 1u | (4u << 2)); }
			code.push_back(w);
		}
		auto generated = generate_vertex_program(code);
		require(generated.attribute_mask == 65535 && generated.indexed_constants && generated.output_register_mask == 65535,
			"ARR consumers lost conservative source/output planning");
		if (variants < keys.size())
		{
			const auto key = cache.vertex(code, {});
			require(cache.vertex(code, {}) == key && std::find(keys.begin(), keys.end(), key) == keys.end(), "ARR shader keys collided");
			keys[variants] = key;
		}
		generated.source += R"metal(
kernel void inspect_arr(device const float4* constants [[buffer(0)]], constant rsx_vertex_context& ctx [[buffer(1)]],
    constant rsx_sampler_parameters* samplers [[buffer(2)]], device uint4* out [[buffer(3)]]) {
    const auto r = rsx_evaluate(constants, constants, ctx, 0, samplers);
    out[0]=as_type<uint4>(r.address[0]); out[1]=as_type<uint4>(r.address[1]);
    out[2]=as_type<uint4>(r.condition[0]); out[3]=as_type<uint4>(r.condition[1]);
    out[4]=as_type<uint4>(r.temp[5]); out[5]=as_type<uint4>(r.temp[7]); out[6]=as_type<uint4>(r.temp[8]);
    for(uint i=0;i<16;++i) out[7+i]=as_type<uint4>(r.dst[i]); out[23]=uint4(r.fault);
}
)metal";
		const auto library = compile.library(generated.source, @"ARR address rounding and consumers");
		const auto descriptor = [MTL4ComputePipelineDescriptor new]; descriptor.computeFunctionDescriptor = library_function(library, @"inspect_arr");
		const auto pipeline = compile.compute(descriptor);
		for (const auto& sample : samples)
		{
			std::array<std::array<float, 4>, 16> constants{};
			for (unsigned i = 0; i < constants.size(); ++i) constants[i] = {float(i), float(i + 16), float(i + 32), float(i + 48)};
			constants[0] = sample; constants[1] = seed; constants[2] = initial;
			std::array<std::int32_t, 4> converted{}, final_address{11, 12, 13, 14}; bool valid = true;
			for (unsigned lane = 0; lane < 4; ++lane)
			{
				// Independent double-precision floor/parity oracle: no rint/nearbyint.
				double v = sample[reverse ? 3 - lane : lane]; if (modifier >= 2) v = std::abs(v); if (modifier & 1u) v = -v;
				const double low = std::floor(v), fraction = v - low;
				const double rounded = low + (fraction > .5 || (fraction == .5 && std::fmod(std::abs(low), 2.0) == 1.0));
				if (!std::isfinite(rounded) || rounded < -2147483648.0 || rounded >= 2147483648.0) valid = false;
				else converted[lane] = static_cast<std::int32_t>(rounded);
			}
			if (!valid) converted.fill(0);
			std::array<words, 24> expected{}; expected[2 + bank] = std::bit_cast<words>(seed);
			for (unsigned i = 0; i < 16; ++i) expected[7 + i] = {0, 0, 0, 0x3f800000};
			for (unsigned lane = 0; lane < 4; ++lane)
			{
				const bool enabled = !predicate || (predicate == 1 ? seed[lane] > 0 : seed[lane] < 0);
				if (!enabled) continue;
				expected[4][lane] = std::bit_cast<unsigned>(seed[lane]);
				if (mask & (8u >> lane))
				{
					final_address[lane] = converted[lane];
					if (update == 1 && valid) expected[2 + bank][lane] = std::bit_cast<unsigned>(float(converted[lane]));
				}
				if (update == 2) expected[2 + bank][lane] = std::bit_cast<unsigned>(seed[lane]);
			}
			expected[address] = std::bit_cast<words>(final_address);
			const auto index = std::int64_t(4) + final_address[component]; const bool in_range = index >= 0 && index < 16;
			if (in_range) { expected[5] = expected[6] = std::bit_cast<words>(constants[index]); expected[7 + index] = std::bit_cast<words>(seed); }
			else expected[6] = {0, 0, 0, 0x3f800000};
			expected[23].fill((valid ? 0u : 32u) | (in_range ? 0u : 1u | 2u | 128u));
			const auto input = make_upload(ctx, std::as_bytes(std::span(constants)));
			ctx.begin(); bindings resources; resources.set_buffer(0, input); resources.set_buffer(1, uniforms); resources.set_buffer(2, samplers); resources.set_buffer(3, output);
			const auto table = resources.snapshot(ctx); const auto encoder = ctx.compute(); ctx.keep_alive(pipeline);
			[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
			[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; ctx.submit().wait();
			require(std::memcmp(output.contents, expected.data(), sizeof(expected)) == 0, "ARR registers or indexed consumers differ from independent ties-to-even oracle");
			++profiles; conversion_faults += !valid; consumer_faults += !in_range;
		}
		kinds |= 1u << (kind * 4 + modifier); ++variants;
	}
	require(variants == 144 && profiles == 2304 && kinds == 4095, "ARR test matrix is incomplete");
	std::cout << "Native ARR: " << variants << " MSL4 variants, " << profiles << " GPU profiles, " << profiles * 96
		<< " exact independent CPU-oracle words; both address/condition banks, four masks/three predicates/update selections, three source types/four modifiers, swizzles and indexed constant/attribute/output consumers; "
		<< conversion_faults << " conversion faults, " << consumer_faults << " guarded consumer faults and 12 cache keys passed\n";
}

void indexed_outputs(bool dual_only=false)
{
	using words=std::array<std::uint32_t,4>;
	constexpr unsigned identity=0x1b03;
	const auto src=[](words& w,unsigned n,unsigned packed)
	{if(n==0){w[1]|=packed>>9;w[2]|=(packed&511u)<<23;}else{w[2]|=packed>>11;w[3]|=(packed&2047u)<<21;}};
	context ctx;compiler compile(ctx);shader_cache cache(compile);
	std::array<std::shared_ptr<const vertex_shader>,12> keys{};
	const auto output=make_buffer(ctx,23*sizeof(words),buffer_storage::shared);
	vertex_context state;state.constant_count=3;const auto uniforms=make_upload(ctx,std::as_bytes(std::span{&state,1}));
	const std::array<sampler_parameters,4> samplers{};const auto sampling=make_upload(ctx,std::as_bytes(std::span(samplers)));
	const words data{0x40000000,0x80000000,0x3e800000,0x7fc54321},seed{0x3f800000,0xbf800000,0,0x7fc12345};
	const std::array addresses{-1.f,-7.f,0.f,8.f,-8.f,9.f,-2147483648.f,2147483520.f,std::bit_cast<float>(0x7fc12345u),-6.f,-5.f,-4.f,-3.f,-2.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};
	unsigned variants=0,profiles=0,invalid=0,masked=0;std::size_t checked=0;
	for(unsigned mode=0;mode<5;++mode)for(unsigned bank=0;bank<2;++bank)for(unsigned component=0;component<4;++component)
	for(unsigned base:{7u,31u})for(unsigned mask:{15u,0u,1u,8u,10u})for(unsigned predicate=0;predicate<3;++predicate)for(bool sat:{false,true})
	@autoreleasepool
	{
		if(dual_only&&mode<2)continue;
		if(base==31&&(mask!=15||predicate||sat))continue;
		try
		{
			const unsigned cc_bank=bank^1u,test=predicate==1?4u:predicate==2?2u:7u;
			words address{bank<<15,(13u<<22),0,((8u>>component)<<13)|(31u<<2)|(63u<<7)};src(address,0,identity);
			words condition{(63u<<15)|(1u<<30)|(1u<<14)|(1u<<29)|(cc_bank<<25)|(7u<<10),(1u<<22)|(2u<<12),0,(15u<<13)|(31u<<2)|(63u<<7)};src(condition,0,identity);
			words write{(1u<<28)|(bank<<24)|component|(cc_bank<<25)|(1u<<14)|(unsigned(sat)<<26)|(test<<10)|(unsigned(predicate!=0)<<13)|(0x1bu<<2),1u<<12,0,base<<2};
			if(mode==0)
			{
				write[0]|=(5u<<15)|(1u<<30)|(1u<<29);write[1]|=1u<<22;write[3]|=(mask<<13)|(63u<<7);src(write,0,identity);
			}
			else
			{
				write[0]|=(63u<<15)|(unsigned(mode==4)<<29);write[1]|=1u<<27;write[3]|=(mask<<17)|((mode==1?63u:5u)<<7);src(write,2,identity);
				if(mode>=3){write[0]=(write[0]&~(63u<<15))|(4u<<15);write[1]|=1u<<22;write[3]|=10u<<13;src(write,0,identity);}
			}
			write[3]|=1u;const std::array code{address,condition,write};auto generated=generate_vertex_program(code);
			const auto output_mask=generated.output_register_mask;
			if(variants<keys.size()){const auto shader=cache.vertex(code,{});require(cache.vertex(code,{})==shader&&std::find(keys.begin(),keys.end(),shader)==keys.end(),"Indexed output shader keys collided");keys[variants]=shader;}
			generated.source+=R"metal(
kernel void inspect_indexed_output(device const float4* constants [[buffer(0)]], constant rsx_vertex_context& ctx [[buffer(1)]],
    constant rsx_sampler_parameters* samplers [[buffer(2)]], device uint4* out [[buffer(3)]]) {
    const auto r=rsx_evaluate(constants,constants,ctx,0,samplers);
    for(uint i=0;i<16;++i)out[i]=as_type<uint4>(r.dst[i]);
    out[16]=as_type<uint4>(r.temp[5]);out[17]=as_type<uint4>(r.temp[4]);
    out[18]=as_type<uint4>(r.condition[0]);out[19]=as_type<uint4>(r.condition[1]);
    out[20]=as_type<uint4>(r.address[0]);out[21]=as_type<uint4>(r.address[1]);out[22]=uint4(r.fault);
}
)metal";
			const auto library=compile.library(generated.source,@"Indexed vertex output probe");auto descriptor=[MTL4ComputePipelineDescriptor new];descriptor.computeFunctionDescriptor=library_function(library,@"inspect_indexed_output");const auto pipeline=compile.compute(descriptor);
			for(float value:addresses)
			{
				words address_data{};address_data[component]=std::bit_cast<unsigned>(value);const std::array constants{address_data,data,seed};const auto input=make_upload(ctx,std::as_bytes(std::span(constants)));
				ctx.begin();bindings bindings;bindings.set_buffer(0,input);bindings.set_buffer(1,uniforms);bindings.set_buffer(2,sampling);bindings.set_buffer(3,output);const auto table=bindings.snapshot(ctx);const auto encoder=ctx.compute();ctx.keep_alive(pipeline);[encoder setComputePipelineState:pipeline];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];ctx.submit().wait();
				std::array<words,23> actual,expected{};std::memcpy(actual.data(),output.contents,sizeof(actual));
				for(unsigned i=0;i<16;++i)expected[i]={0,0,0,0x3f800000};expected[18+cc_bank]=seed;
				const bool converted=std::isfinite(value)&&value>=-2147483648.f&&value<2147483648.f;
				const std::int32_t integer=converted?static_cast<std::int32_t>(value):0;expected[20+bank][component]=std::bit_cast<unsigned>(integer);
				const auto index=std::int64_t(base)+integer;unsigned fault=converted?0u:32u;bool writes=false;
				for(unsigned lane=0;lane<4;++lane)
				{
					const auto prior=std::bit_cast<float>(seed[lane]);const bool enabled=!predicate||(predicate==1?prior>0:prior==0);
					auto raw=data[lane];const auto scalar=std::bit_cast<float>(raw);if(sat&&scalar<0)raw=0;if(sat&&scalar>1)raw=0x3f800000;
					if(enabled&&(mask&(8u>>lane)))
					{
						if(mode!=1)expected[16][lane]=raw;
						if(base!=31){writes=true;if(index>=0&&index<16)expected[index][lane]=raw;}
						if(mode!=4)expected[18+cc_bank][lane]=raw;
					}
					if(mode>=3&&enabled&&(10u&(8u>>lane))){expected[17][lane]=raw;if(mode==4)expected[18+cc_bank][lane]=raw;}
				}
				if(writes&&(index<0||index>=16)){fault|=128u;++invalid;}
				if(!writes)++masked;expected[22].fill(fault);
				require(actual==expected,"Indexed output register routing differs from CPU oracle");
				require(output_mask==(base==31?0u:65535u),"Indexed output metadata is not conservative");
				checked+=23*4;++profiles;
			}
			++variants;
		}
		catch(const std::exception& e){throw std::runtime_error("Indexed output mode="+std::to_string(mode)+" bank="+std::to_string(bank)+" component="+std::to_string(component)+" base="+std::to_string(base)+" mask="+std::to_string(mask)+" pred="+std::to_string(predicate)+" sat="+std::to_string(sat)+": "+e.what());}
	}
	if(!dual_only)require(variants==1240&&profiles==26040,"Indexed output matrix is incomplete");
	std::cout<<"Indexed vertex outputs: "<<variants<<" native MSL4 variants, "<<profiles<<" GPU profiles, "<<checked<<" exact independent CPU-oracle register words; two address banks/four components, VEC/SCA/dual/paired routing, masks/predicates/saturation, "<<invalid<<" guarded output faults and "<<masked<<" inactive/disabled outputs, source conversion faults, signed bounds and twelve shader keys passed\n";
}

void scalar_dual_destinations()
{
	using words=std::array<std::uint32_t,4>;
	constexpr unsigned identity_constant=0x1b03,replicated_constant=3;
	const auto source=[](words& w,unsigned index,unsigned packed)
	{
		if(index==0){w[1]|=packed>>9;w[2]|=(packed&511u)<<23;}
		else {w[2]|=packed>>11;w[3]|=(packed&2047u)<<21;}
	};
	const auto move=[&](unsigned constant,unsigned temporary,unsigned output,unsigned control=0)
	{
		words w{control|(temporary<<15)|(1u<<30)|(7u<<10), (1u<<22)|(constant<<12),0,(15u<<13)|(63u<<7)|(output<<2)};
		source(w,0,identity_constant);return w;
	};
	context ctx;compiler compile(ctx);shader_cache cache(compile);
	const auto output=make_buffer(ctx,9*sizeof(words),buffer_storage::shared);
	const std::array<sampler_parameters,4> sampling{};const auto samplers=make_upload(ctx,std::as_bytes(std::span(sampling)));
	vertex_context state;state.constant_count=3;const auto uniforms=make_upload(ctx,std::as_bytes(std::span{&state,1}));
	const words marker{0x40a00000,0x40c00000,0x40e00000,0x41000000};
	const words initial_condition{0x3f800000,0xbf800000,0,0x7fc12345};
	const std::array inputs{words{0x40000000,0x3f000000,0x3e800000,0x40800000},
		words{0xc0000000,0x80000000,0x40800000,0x7fc54321},words{0x7f800000,0xff800000,0x7fc12345,0x80000000}};
	unsigned variants=0,profiles=0;std::size_t checked=0;
	std::array<std::shared_ptr<const vertex_shader>,8> keys{};
	for(unsigned opcode:{1u,2u,3u,4u,5u,6u,7u,13u,14u,15u,16u})for(unsigned mask:{1u,8u,10u,15u})
	for(unsigned mode=0;mode<4;++mode)for(bool saturate:{false,true})for(bool from_temporary:{false,true})for(bool paired:{false,true})
	@autoreleasepool
	{
		if(mode==3&&!paired)continue;
		try
		{
			const unsigned bank=mode>=2;const bool update=mode!=0,vector_condition=mode==3;
			const unsigned test=mode==1?4u:mode==2?5u:mode==3?2u:7u;
			const auto scalar=[&](unsigned temporary,unsigned destination,bool route_output)
			{
				const unsigned control=(63u<<15)|(bank<<25)|(unsigned(saturate)<<26)|(unsigned(!route_output)<<30)|
					(unsigned(update)<<14)|(unsigned(vector_condition)<<29)|(test<<10)|(unsigned(mode!=0)<<13)|(0x1bu<<2);
				words w{control,opcode<<27,0,(temporary<<7)|(destination<<2)|(mask<<17)};
				const unsigned packed=from_temporary?(5u<<2)|1u:3u;
				source(w,2,(opcode==1||opcode==7)?packed|0x1b00u:packed);
				if(paired){w[0]=(w[0]&~(63u<<15))|(4u<<15);w[1]|=1u<<22;w[3]|=10u<<13;source(w,0,identity_constant);}
				return w;
			};
			const auto seed_condition=move(2,63,31,(1u<<14)|(1u<<29)|(bank<<25));
			std::vector<words> code{move(0,5,31),move(0,6,31),move(1,63,1),move(1,63,2),
				seed_condition,scalar(6,31,false),seed_condition,scalar(63,2,true),seed_condition,scalar(5,1,true),move(0,63,3)};
			// Copy the newly written temporary in the next instruction.
			code.back()[1]&=~255u;code.back()[2]&=0x007fffffu;source(code.back(),0,(5u<<2)|1u|0x1b00u);code.back()[3]|=1u;
			auto generated=generate_vertex_program(code);
			require((generated.output_register_mask&14u)==14u,"Scalar dual output disappeared from shader metadata");
			if(variants<keys.size())
			{
				const auto shader=cache.vertex(code,{});require(cache.vertex(code,{})==shader&&std::find(keys.begin(),keys.end(),shader)==keys.end(),"Scalar dual-destination shader key collided");keys[variants]=shader;
			}
			generated.source+=R"metal(
kernel void inspect_dual(device const float4* constants [[buffer(0)]], constant rsx_vertex_context& ctx [[buffer(1)]],
    constant rsx_sampler_parameters* sampling [[buffer(2)]], device uint4* out [[buffer(3)]]) {
    const auto r=rsx_evaluate(constants,constants,ctx,0,sampling);
    out[0]=as_type<uint4>(r.dst[1]);out[1]=as_type<uint4>(r.dst[2]);out[2]=as_type<uint4>(r.dst[3]);
    out[3]=as_type<uint4>(r.temp[5]);out[4]=as_type<uint4>(r.temp[6]);out[5]=as_type<uint4>(r.temp[4]);
    out[6]=as_type<uint4>(r.condition[0]);out[7]=as_type<uint4>(r.condition[1]);out[8]=uint4(r.fault);
}
)metal";
			const auto library=compile.library(generated.source,@"Scalar dual-destination probe");auto d=[MTL4ComputePipelineDescriptor new];d.computeFunctionDescriptor=library_function(library,@"inspect_dual");const auto pipeline=compile.compute(d);
			for(const auto& input:inputs)
			{
				const std::array constants{input,marker,initial_condition};const auto values=make_upload(ctx,std::as_bytes(std::span(constants)));
				ctx.begin();bindings bindings;bindings.set_buffer(0,values);bindings.set_buffer(1,uniforms);bindings.set_buffer(2,samplers);bindings.set_buffer(3,output);
				const auto table=bindings.snapshot(ctx);const auto encoder=ctx.compute();ctx.keep_alive(pipeline);[encoder setComputePipelineState:pipeline];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];ctx.submit().wait();
				std::array<words,9> actual;std::memcpy(actual.data(),output.contents,sizeof(actual));
				require(actual[0]==actual[1]&&actual[2]==actual[4]&&actual[3]==actual[4],"Scalar dual destinations differ from separate single-output/temp controls or next-instruction use");
				words expected_cc=initial_condition;
				for(unsigned lane=0;lane<4;++lane)
				{
					const auto cc=std::bit_cast<float>(initial_condition[lane]);
					const bool enabled=!mode||(mode==1?cc>0:mode==2?cc!=0:cc==0);
					if(!(mask&(8u>>lane))||!enabled)
						require(actual[0][lane]==marker[lane]&&actual[3][lane]==input[lane],"Scalar dual write changed an unselected destination lane");
					if(update&&enabled)
					{
						if(vector_condition){if(10u&(8u>>lane))expected_cc[lane]=actual[5][lane];}
						else if(mask&(8u>>lane))expected_cc[lane]=actual[4][lane];
					}
				}
				require(actual[6+bank]==expected_cc&&actual[7-bank]==words{}&&actual[8]==words{},"Scalar dual-destination condition selection or fault status differs");
				checked+=6*4;++profiles;
			}
			++variants;
		}
		catch(const std::exception& e){throw std::runtime_error("Scalar dual opcode="+std::to_string(opcode)+" mask="+std::to_string(mask)+" mode="+std::to_string(mode)+" sat="+std::to_string(saturate)+" temp="+std::to_string(from_temporary)+" pair="+std::to_string(paired)+": "+e.what());}
	}
	require(variants==1232&&profiles==3696&&cache.vertex_statistics().compiled==8,"Scalar dual-destination matrix or cache coverage differs");
	std::cout<<"Scalar dual destinations: "<<variants<<" MSL4 variants, "<<profiles<<" GPU profiles, "<<checked<<" exact destination/condition/fault words; all 11 scalar ALU opcodes, four masks, saturation, source self-overlap, both condition banks, pre-write predicates, paired VEC/SCA result selection, following temporary use, special float words and eight distinct retained shader keys passed\n";
}

void condition_updates()
{
	using word_vector = std::array<std::uint32_t, 4>;
	constexpr unsigned constant_xyzw = 3u | (1u << 12) | (2u << 10) | (3u << 8);
	const auto move = [](bool scalar, unsigned constant, unsigned destination, unsigned mask, unsigned control)
	{
		word_vector instruction{control | (63u << 15), constant << 12, 0, (destination << 2) | (63u << 7)};
		if (scalar)
		{
			instruction[1] |= 1u << 27;
			instruction[2] |= constant_xyzw >> 11;
			instruction[3] |= ((constant_xyzw & 2047u) << 21) | (mask << 17);
		}
		else
		{
			instruction[0] |= 1u << 30;
			instruction[1] |= (1u << 22) | (constant_xyzw >> 9);
			instruction[2] |= (constant_xyzw & 511u) << 23;
			instruction[3] |= mask << 13;
		}
		return instruction;
	};
	const std::array source_profiles{
		word_vector{0xc0000000, 0x80000000, 0x40800000, 0x7fc12345},
		word_vector{0x7f800000, 0xff800000, 0xc0000000, 0},
		word_vector{0x80000000, 0, 0x7fc54321, 0x41000000}};
	const word_vector replacement{0x80000000, 0x40a00000, 0x7f800000, 0xc0800000};
	const word_vector marker{0x3e800000, 0x3f000000, 0x3f400000, 0x3f800000};
	const word_vector default_output{0, 0, 0, 0x3f800000};
	context ctx;
	compiler compile(ctx);
	shader_cache cache(compile);
	std::array<std::shared_ptr<const vertex_shader>, 16> shaders;
	unsigned variants = 0, profiles = 0, taken = 0, fallen_through = 0;
	const auto output = make_buffer(ctx, 7 * sizeof(word_vector), buffer_storage::shared);
	const std::array<sampler_parameters, 4> sampling{};
	const auto sampler_buffer = make_upload(ctx, std::as_bytes(std::span(sampling)));
	vertex_context state;
	state.constant_count = 3;
	const auto state_buffer = make_upload(ctx, std::as_bytes(std::span{&state, 1}));
	for (bool paired : {false, true})
	for (bool scalar : {false, true})
		for (unsigned bank : {0u, 1u})
			for (bool saturate : {false, true})
			{
				const unsigned update = (1u << 14) | (bank << 25) | (unsigned(saturate) << 26) | (unsigned(!scalar) << 29);
				const unsigned test_zxzz = (2u << 8) | (2u << 4) | (2u << 2) | (4u << 10) | (1u << 13);
				std::array code{
					move(scalar, 0, 1, 15, update | (7u << 10)),
					move(scalar, 1, 2, 13, update | test_zxzz),
					word_vector{(bank << 25) | (4u << 10) | (0xaau << 2), 9u << 27, 0, (4u << 29) | (31u << 2)},
					move(false, 2, 3, 15, 7u << 10),
					word_vector{0, 0, 0, (31u << 2) | 1u}};
				if (paired)
				{
					constexpr unsigned reversed = 0xe403;
					auto& pair = code[1];
					if (scalar)
					{
						pair[0] = (pair[0] & ~(63u << 15)) | (4u << 15);
						pair[1] |= (1u << 22) | (reversed >> 9);
						pair[2] |= (reversed & 511u) << 23;
						pair[3] |= 15u << 13;
					}
					else
					{
						pair[1] |= 1u << 27;
						pair[2] |= reversed >> 11;
						pair[3] = (pair[3] & ~(63u << 7)) | ((reversed & 2047u) << 21) | (15u << 17) | (4u << 7);
					}
				}
				const auto shader = cache.vertex(code, {});
				require(cache.vertex(code, {}) == shader && std::find(shaders.begin(), shaders.begin() + variants, shader) == shaders.begin() + variants,
					"Native vertex cache aliased or recompiled a condition variant");
				shaders[variants++] = shader;
				auto generated = generate_vertex_program(code);
				generated.source += R"metal(
kernel void inspect_condition(device const float4* constants [[buffer(0)]],
    constant rsx_vertex_context& ctx [[buffer(1)]], constant rsx_sampler_parameters* sampling [[buffer(2)]],
    device uint4* output [[buffer(3)]]) {
    const auto r = rsx_evaluate(constants, constants, ctx, 0, sampling);
    output[0] = as_type<uint4>(r.dst[1]);
    output[1] = as_type<uint4>(r.dst[2]);
    output[2] = as_type<uint4>(r.dst[3]);
    output[3] = as_type<uint4>(r.condition[0]);
    output[4] = as_type<uint4>(r.condition[1]);
    output[5] = uint4(r.fault);
    output[6] = as_type<uint4>(r.temp[4]);
}
)metal";
				const auto library = compile.library(generated.source, @"RSX vertex condition register probe");
				const auto descriptor = [MTL4ComputePipelineDescriptor new];
				descriptor.computeFunctionDescriptor = library_function(library, @"inspect_condition");
				const auto pipeline = compile.compute(descriptor);
				const auto sat = [saturate](std::uint32_t raw)
				{
					const float value = std::bit_cast<float>(raw);
					if (!saturate) return raw;
					if (value < 0) return 0u;
					if (value > 1) return 0x3f800000u;
					return raw;
				};
				for (const auto& source : source_profiles)
				{
					const std::array constants{source, replacement, marker};
					const auto input = make_upload(ctx, std::as_bytes(std::span(constants)));
					word_vector first;
					std::transform(source.begin(), source.end(), first.begin(), sat);
					auto condition = first;
					auto second = default_output;
					word_vector other{};
					constexpr std::array selectors{2u, 0u, 2u, 2u};
					for (unsigned lane = 0; lane < 4; ++lane)
						if (std::bit_cast<float>(first[selectors[lane]]) > 0)
						{
							if (lane != 2) condition[lane] = second[lane] = sat(replacement[lane]);
							if (paired) other[lane] = sat(replacement[3 - lane]);
						}
					const bool branch = std::bit_cast<float>(condition[2]) > 0;
					const std::array expected{first, second, branch ? default_output : marker,
						bank ? word_vector{} : condition, bank ? condition : word_vector{}, word_vector{}, other};
					ctx.begin();
					bindings resources;
					resources.set_buffer(0, input);
					resources.set_buffer(1, state_buffer);
					resources.set_buffer(2, sampler_buffer);
					resources.set_buffer(3, output);
					const auto table = resources.snapshot(ctx);
					const auto encoder = ctx.compute();
					ctx.keep_alive(pipeline);
					[encoder setComputePipelineState:pipeline];
					[encoder setArgumentTable:table];
					[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
					ctx.submit().wait();
					require(std::memcmp(output.contents, expected.data(), sizeof(expected)) == 0,
						"Native vertex condition registers, masked writes or branch differ from the raw-bit oracle");
					++profiles;
					taken += branch;
					fallen_through += !branch;
				}
			}
	require(variants == 16 && profiles == 48 && taken == 16 && fallen_through == 32 && cache.vertex_statistics().compiled == 16,
		"Native condition test did not cover both registers, saturation, cache keys and branch outcomes");
	std::cout << "Native vertex condition updates: 48 GPU profiles (24 paired), 1344 exact raw words, both registers, 16 cache keys and branch 16/32 passed\n";
}

void address_condition_updates()
{
	using words = std::array<std::uint32_t, 4>;
	constexpr unsigned identity_constant = 0x1b03;
	const auto arl = [](unsigned address, unsigned constant, unsigned mask, unsigned control)
	{
		return words{(address << 15) | control, (13u << 22) | (constant << 12) | (identity_constant >> 9),
			(identity_constant & 511u) << 23, (mask << 13) | (63u << 7) | (31u << 2)};
	};
	const std::array<float, 4> initial{-2, 1, -5, -1}, marker{.25f, .5f, .75f, 1};
	const std::array samples{
		std::array{-.75f, 4.875f, -3.25f, -1.25f},
		std::array{-2147483648.0f, 2147483520.0f, 0.f, -0.f},
		std::array{std::bit_cast<float>(0x7fc12345u), 4.f, 2.f, 1.f}};
	context ctx;
	compiler compile(ctx);
	const std::array<sampler_parameters, 4> sampling{};
	vertex_context state;
	state.constant_count = 3;
	const auto sampler_buffer = make_upload(ctx, std::as_bytes(std::span(sampling)));
	const auto uniforms = make_upload(ctx, std::as_bytes(std::span{&state, 1}));
	const auto output = make_buffer(ctx, 7 * sizeof(words), buffer_storage::shared);
	unsigned profiles = 0, branches_taken = 0;
	for (unsigned mode : {0u, 1u, 2u})
		for (unsigned address : {0u, 1u})
			for (unsigned bank : {0u, 1u})
			{
				const auto condition = (1u << 14) | (bank << 25);
				const auto select_vector = 1u << 29;
				const auto lt_xxyx = (1u << 13) | (1u << 10) | (1u << 4);
				auto load = arl(address, 1, 13, condition | lt_xxyx | (mode == 2 ? 0 : select_vector));
				if (mode)
				{
					load[0] |= 1u << 30; // SCA writes temporary r4.
					load[1] |= 1u << 27;
					load[2] |= identity_constant >> 11;
					load[3] = (load[3] & ~(63u << 7)) | ((identity_constant & 2047u) << 21) | (11u << 17) | (4u << 7);
				}
				const std::array code{
					arl(address, 0, 15, condition | select_vector),
					load,
					words{(bank << 25) | (2u << 10), 9u << 27, 0, (4u << 29) | (31u << 2)},
					words{(63u << 15) | (1u << 30), (1u << 22) | (2u << 12) | (identity_constant >> 9),
						(identity_constant & 511u) << 23, (15u << 13) | (1u << 2)},
					words{0, 0, 0, 1u | (31u << 2)}};
				auto generated = generate_vertex_program(code);
				generated.source += R"metal(
kernel void inspect_arl(device const float4* constants [[buffer(0)]],
    constant rsx_vertex_context& ctx [[buffer(1)]], constant rsx_sampler_parameters* sampling [[buffer(2)]],
    device uint4* output [[buffer(3)]]) {
    const auto r = rsx_evaluate(constants, constants, ctx, 0, sampling);
    output[0] = as_type<uint4>(r.address[0]);
    output[1] = as_type<uint4>(r.address[1]);
    output[2] = as_type<uint4>(r.condition[0]);
    output[3] = as_type<uint4>(r.condition[1]);
    output[4] = as_type<uint4>(r.temp[4]);
    output[5] = as_type<uint4>(r.dst[1]);
    output[6] = uint4(r.fault);
}
)metal";
				const auto library = compile.library(generated.source, @"RSX ARL condition probe");
				const auto descriptor = [MTL4ComputePipelineDescriptor new];
				descriptor.computeFunctionDescriptor = library_function(library, @"inspect_arl");
				const auto pipeline = compile.compute(descriptor);
				for (const auto& source : samples)
				{
					const std::array constants{initial, source, marker};
					const auto input = make_upload(ctx, std::as_bytes(std::span(constants)));
					const bool valid = std::isfinite(source[0]);
					std::array<std::int32_t, 4> address_value{-2, 1, -5, -1};
					auto cc = initial;
					std::array<float, 4> scalar{};
					for (unsigned lane : {0u, 1u, 3u})
					{
						address_value[lane] = valid ? static_cast<std::int32_t>(source[lane]) : 0;
						if (mode != 2 && valid) cc[lane] = float(address_value[lane]);
						if (mode && lane != 1)
						{
							scalar[lane] = source[lane];
							if (mode == 2) cc[lane] = source[lane];
						}
					}
					const bool taken = cc[0] == 0;
					const auto integer_words = std::bit_cast<words>(address_value);
					const auto condition_words = std::bit_cast<words>(cc);
					const auto result = taken ? words{0, 0, 0, 0x3f800000} : std::bit_cast<words>(marker);
					const std::array expected{address ? words{} : integer_words, address ? integer_words : words{},
						bank ? words{} : condition_words, bank ? condition_words : words{}, std::bit_cast<words>(scalar), result,
						valid ? words{} : words{32, 32, 32, 32}};
					ctx.begin();
					bindings resources;
					resources.set_buffer(0, input);
					resources.set_buffer(1, uniforms);
					resources.set_buffer(2, sampler_buffer);
					resources.set_buffer(3, output);
					const auto table = resources.snapshot(ctx);
					const auto encoder = ctx.compute();
					ctx.keep_alive(pipeline);
					[encoder setComputePipelineState:pipeline];
					[encoder setArgumentTable:table];
					[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
					ctx.submit().wait();
					require(std::memcmp(output.contents, expected.data(), sizeof(expected)) == 0,
						"Native ARL condition update differs from the converted integer or overwrote a masked lane");
					++profiles;
					branches_taken += taken;
				}
			}
	require(profiles == 36 && branches_taken == 8, "ARL condition test missed an address bank or branch outcome");
	std::cout << "Native ARL condition updates: 36 GPU profiles, 1008 exact words, both address and condition banks, masks, paired selection and faults passed\n";
}

void masked_scalar_swizzles()
{
	using words = std::array<std::uint32_t, 4>;
	context ctx;
	compiler compile(ctx);
	shader_cache cache(compile);
	const std::array<float, 4> constants{2.f, 4.f, 8.f, 16.f};
	const std::array<sampler_parameters, 4> sampling{};
	vertex_context state;
	state.constant_count = 1;
	const auto input = make_upload(ctx, std::as_bytes(std::span(constants)));
	const auto uniforms = make_upload(ctx, std::as_bytes(std::span{&state, 1}));
	const auto samplers = make_upload(ctx, std::as_bytes(std::span(sampling)));
	const auto output = make_buffer(ctx, 4 * sizeof(words), buffer_storage::shared);
	for (bool paired : {false, true})
	for (unsigned source = 0; source < 4; ++source)
		for (unsigned mask : {8u, 10u})
		{
			const unsigned bank = mask == 10;
			const unsigned other = (source + 1) & 3;
			const unsigned packed = (source << 14) | (other << 12) | ((mask == 10 ? source : other) << 10) | (other << 8) | 3u;
			std::array<words, 1> code{words{(7u << 10) | (1u << 14) | (63u << 15) | (bank << 25),
				2u << 27, packed >> 11, ((packed & 2047u) << 21) | (mask << 17) | (63u << 7) | (1u << 2) | 1u}};
			if (paired)
			{
				// VEC writes y/w to r1 and o1. SCA writes x (and
				// optionally z) to r1. The read is from a constant.
				constexpr unsigned constant_xyzw = 0x1b03;
				code[0][0] = (7u << 10) | (1u << 15) | (1u << 30);
				code[0][1] |= (1u << 22) | (constant_xyzw >> 9);
				code[0][2] |= (constant_xyzw & 511u) << 23;
				code[0][3] = (code[0][3] & ~(63u << 7)) | (1u << 7) | (5u << 13);
			}
			const auto shader = cache.vertex(code, {});
			require(cache.vertex(code, {}) == shader, "Masked scalar variant did not retain its native shader");
			auto generated = generate_vertex_program(code);
			generated.source += R"metal(
kernel void inspect_scalar(device const float4* constants [[buffer(0)]],
    constant rsx_vertex_context& ctx [[buffer(1)]], constant rsx_sampler_parameters* sampling [[buffer(2)]],
    device uint4* output [[buffer(3)]]) {
    const auto r = rsx_evaluate(constants, constants, ctx, 0, sampling);
    output[0] = as_type<uint4>(r.dst[1]);
    output[1] = as_type<uint4>(r.condition[0]);
    output[2] = as_type<uint4>(r.condition[1]);
    output[3] = as_type<uint4>(r.temp[1]);
}
)metal";
			const auto library = compile.library(generated.source, @"RSX masked scalar probe");
			const auto descriptor = [MTL4ComputePipelineDescriptor new];
			descriptor.computeFunctionDescriptor = library_function(library, @"inspect_scalar");
			const auto pipeline = compile.compute(descriptor);
			const auto value = std::bit_cast<std::uint32_t>(1.f / constants[source]);
			const words condition{value, 0, mask == 10 ? value : 0, 0};
			const auto y = std::bit_cast<std::uint32_t>(constants[1]);
			const auto w = std::bit_cast<std::uint32_t>(constants[3]);
			const auto expected = paired ?
				std::array{words{0, y, 0, w}, words{}, words{}, words{value, y, condition[2], w}} :
				std::array{words{value, 0, condition[2], 0x3f800000},
					bank ? words{} : condition, bank ? condition : words{}, words{}};
			ctx.begin();
			bindings resources;
			resources.set_buffer(0, input);
			resources.set_buffer(1, uniforms);
			resources.set_buffer(2, samplers);
			resources.set_buffer(3, output);
			const auto table = resources.snapshot(ctx);
			const auto encoder = ctx.compute();
			ctx.keep_alive(pipeline);
			[encoder setComputePipelineState:pipeline];
			[encoder setArgumentTable:table];
			[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
			ctx.submit().wait();
			require(std::memcmp(output.contents, expected.data(), sizeof(expected)) == 0,
				"Masked scalar swizzle changed a destination or condition lane");
		}
	require(cache.vertex_statistics().compiled == 16, "Masked scalar shader keys collided");
	std::cout << "Masked scalar swizzles: 16 native RCP profiles (eight paired), 256 exact output, temporary and condition words passed\n";
}
id<MTLComputePipelineState> probe(compiler& compile, unsigned slot, bool indirect = false, bool selected = false)
{
	constexpr unsigned input = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	std::vector<std::array<unsigned,4>> code;
	if (indirect) code.push_back({7u << 10,(13u << 22) | (input >> 9),(input & 511) << 23,15u << 13});
	code.push_back({(63u << 15) | (7u << 10) | (1u << 30) | (indirect ? 1u << 27 : 0),
		(1u << 22) | ((indirect ? 5u : slot) << 8) | (input >> 9),(input & 511) << 23,
		(15u << 13) | (15u << 17) | (1u << 7) | 1u});
	vertex_program_options options; options.packed_inputs = true;
	auto generated = generate_vertex_program(code,options);
	require(generated.attribute_mask == (indirect ? 65535u : 1u << slot),"Packed probe lost static/dynamic input mask");
	generated.source += R"metal(
kernel void inspect_packed(device const uchar* persistent [[buffer(0)]],
 device const uchar* transient [[buffer(1)]],constant rsx_fetch_parameters& fetch [[buffer(2)]],
 device float4* output [[buffer(3)]],constant rsx_vertex_context& ctx [[buffer(4)]],
 constant rsx_sampler_parameters* sampling [[buffer(5)]],device uint* faults [[buffer(6)]],
 uint ordinal [[thread_position_in_grid]]) {
 const rsx_packed_vertex_inputs inputs{persistent,transient,&fetch};
 const auto r = rsx_evaluate(inputs,output,ctx,ctx.vertex_base+ordinal,sampling);
 output[ordinal] = r.dst[0]; faults[ordinal] = r.fault;
}
kernel void inspect_sparse(device const uchar* persistent [[buffer(0)]],
 device const uchar* transient [[buffer(1)]],constant rsx_fetch_parameters& fetch [[buffer(2)]],
 device float4* output [[buffer(3)]],constant rsx_vertex_context& ctx [[buffer(4)]],
 constant rsx_sampler_parameters* sampling [[buffer(5)]],device uint* faults [[buffer(6)]],
 device const uint* ids [[buffer(7)]],uint ordinal [[thread_position_in_grid]]) {
 const rsx_packed_vertex_inputs inputs{persistent,transient,&fetch};
 const auto r = rsx_evaluate(inputs,output,ctx,ids[ordinal],sampling);
 output[ordinal] = r.dst[0]; faults[ordinal] = r.fault;
}
)metal";
	const auto library = compile.library(generated.source,@"Packed RSX vertex probe");
	auto descriptor = [MTL4ComputePipelineDescriptor new];
	descriptor.computeFunctionDescriptor = library_function(library,selected ? @"inspect_sparse" : @"inspect_packed");
	return compile.compute(descriptor);
}
struct address_case { unsigned first,count,base; int offset; unsigned frequency; bool modulo; };
constexpr std::array address_cases{
	address_case{0,19,0,0,0,false},address_case{65535,19,65532,0,3,false},
	address_case{65534,19,0,0,7,true},address_case{1048560,16,0,-1048557,5,true},
	address_case{17,19,1,0,2,false},address_case{12,19,11,-10,5,true}};
unsigned source_element(const address_case& c, unsigned ordinal)
{
	if (!c.frequency) return 0;
	const long id = long(c.first) + ordinal;
	return static_cast<unsigned>(c.modulo ? (id + c.offset) % c.frequency : (id - c.base) / c.frequency);
}
unsigned pattern(unsigned type, unsigned element, unsigned component)
{
	const auto i = element * 3 + component;
	if (type == 2) return std::array{0u,0x80000000u,0x3f800000u,0xbf800000u,0x3eaaaaabu,0x7fc12345u,0x7f800000u,0xff800000u,0x00000001u,0x007fffffu,0x7f7fffffu}[i % 11];
	if (type == 3) return std::array{0u,0x8000u,1u,0x3ffu,0x400u,0x3c00u,0xbc00u,0x7bffu,0x7c00u,0xfc00u,0x7e11u}[i % 11];
	if (type == 6) return std::array{0u,0xffffffffu,0x7fe003ffu,0x80200400u,0x12345678u}[i % 5];
	return std::array{0u,1u,0x7fffu,0x8000u,0xffffu,0x1234u}[i % 6];
}
// Independent mathematical storage oracle. It does not call the native
// descriptor helpers, shared MSL decoder or CPU adapter of that decoder.
std::array<float,4> expected(unsigned type, unsigned count, unsigned element)
{
	std::array<float,4> result{};
	if (type == 6)
	{
		const unsigned word = pattern(type,element,0);
		for (unsigned c = 0; c < 3; ++c)
		{
			const unsigned width = c == 2 ? 10 : 11, mask = (1u << width) - 1;
			const unsigned raw = (word >> (c * 11)) & mask;
			const int value = raw < (1u << (width - 1)) ? int(raw) : int(raw) - int(1u << width);
			result[c] = float(value * (c == 2 ? 64 : 32)) / 32767.f;
		}
		result[3] = 1; return result;
	}
	for (unsigned c = 0; c < 4; ++c)
	{
		const auto raw = c < count ? pattern(type,element,c) : 0;
		const int signed16 = raw < 32768 ? int(raw) : int(raw) - 65536;
		if (type == 1) result[c] = (float(signed16) + .5f) / 32767.5f;
		else if (type == 2) result[c] = std::bit_cast<float>(raw);
		else if (type == 3) result[c] = float(std::bit_cast<_Float16>(static_cast<std::uint16_t>(raw)));
		else if (type == 4) result[c] = float(raw & 255) / 255.f;
		else if (type == 5) result[c] = float(signed16);
		else result[c] = float(raw & 255);
	}
	if (count < 4) result[3] = 1;
	return result;
}
void compare(float actual, float reference, unsigned type)
{
	if (std::isnan(reference)) { require(std::isnan(actual),"Packed native NaN classification differs"); return; }
	if (type == 1 || type == 4 || type == 6)
	{
		const float ulp = std::abs(std::nextafter(reference,INFINITY) - reference);
		require(std::abs(actual - reference) <= 2 * ulp,"Packed native normalized result differs from mathematical oracle");
	}
	else require(std::bit_cast<unsigned>(actual) == std::bit_cast<unsigned>(reference),"Packed native float/half/integer bits differ from independent storage oracle");
}
void formats()
{
	context ctx; compiler compile(ctx);
	unsigned cases = 0, values = 0;
	for (unsigned mode = 0; mode < 5; ++mode)
	{
		const unsigned slot = std::array{0u,7u,15u,7u,15u}[mode];
		const auto pipeline = probe(compile,slot,mode >= 3);
		for (unsigned type = 1; type < 8; ++type) for (unsigned count = 1; count <= (type == 6 ? 1u : 4u); ++count)
			for (bool be : {false,true}) for (bool transient : {false,true}) for (const auto& address : address_cases)
			{
				@autoreleasepool
				{
					const unsigned width = type == 2 || type == 6 ? 4 : type == 4 || type == 7 ? 1 : 2;
					unsigned largest = 0;
					for (unsigned i = 0; i < address.count; ++i) largest = std::max(largest,source_element(address,i));
					const unsigned stride = width * count;
					std::vector<std::byte> bytes(16 + (largest + 1) * stride,std::byte{0xa5});
					const float dynamic = float(slot - 5); std::memcpy(bytes.data(),&dynamic,4);
					for (unsigned element = 0; element <= largest; ++element) for (unsigned c = 0; c < count; ++c)
						for (unsigned b = 0; b < width; ++b)
							bytes[16 + element * stride + c * width + b] = std::byte(pattern(type,element,c) >> ((be ? width - 1 - b : b) * 8));
					const auto stream = make_upload(ctx,bytes), prefix = make_upload(ctx,std::span(bytes).first(16));
					const auto persistent = transient ? prefix : stream, volatile_stream = transient ? stream : nil;
					vertex_fetch_request request; request.first_vertex = address.first; request.vertex_count = address.count;
					request.vertex_base_index = address.base; request.vertex_index_offset = address.offset;
					request.attribute_mask = mode >= 3 ? 65535u : 1u << slot;
					request.attributes[slot] = {stride | (address.frequency << 8) | (type << 24) | (count << 27),16u | (be ? 1u << 29 : 0) | (transient ? 1u << 30 : 0) | (address.modulo ? 1u << 31 : 0)};
					if (mode >= 3) request.attributes[0] = {(2u << 24) | (1u << 27),0};
					validate_vertex_fetch(ctx,request,persistent,volatile_stream);
					const auto output = make_buffer(ctx,address.count * 16,buffer_storage::shared), faults = make_buffer(ctx,address.count * 4,buffer_storage::shared);
					ctx.begin();
					const std::array<std::byte,16> zero{}; const auto dummy = ctx.upload(zero,16);
					const vertex_fetch_parameters parameters{request.attributes,persistent.length,volatile_stream.length,request.first_vertex,request.vertex_count,request.vertex_base_index,request.vertex_index_offset,0,0,request.attribute_mask};
					vertex_context state; state.vertex_base = address.first; state.vertex_count = address.count;
					const std::array<sampler_parameters,4> sampling{};
					const auto p = ctx.upload(std::as_bytes(std::span{&parameters,1}),256), v = ctx.upload(std::as_bytes(std::span{&state,1}),256), s = ctx.upload(std::as_bytes(std::span(sampling)),16);
					bindings resources; resources.set_buffer(0,persistent); resources.set_buffer(1,volatile_stream ? volatile_stream : dummy.buffer,volatile_stream ? 0 : dummy.offset);
					resources.set_buffer(2,p.buffer,p.offset); resources.set_buffer(3,output); resources.set_buffer(4,v.buffer,v.offset); resources.set_buffer(5,s.buffer,s.offset); resources.set_buffer(6,faults);
					const auto table = resources.snapshot(ctx); const auto encoder = ctx.compute(); ctx.keep_alive(pipeline);
					[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
					[encoder dispatchThreads:MTLSizeMake(address.count,1,1) threadsPerThreadgroup:MTLSizeMake(address.count,1,1)];
					ctx.submit().wait();
					for (unsigned i = 0; i < address.count; ++i)
					{
						require(static_cast<const unsigned*>(faults.contents)[i] == 0,"Packed translated RSX evaluator faulted");
						const auto oracle = expected(type,count,source_element(address,i));
						for (unsigned c = 0; c < 4; ++c)
						{
							const float actual = static_cast<const float*>(output.contents)[i * 4 + c];
							try { compare(actual,oracle[c],type); }
							catch (const std::exception& error)
							{
								throw std::runtime_error(std::string(error.what()) + ":type=" + std::to_string(type) + " components=" + std::to_string(count) + " mode=" + std::to_string(mode) + " first=" + std::to_string(address.first) + " ordinal=" + std::to_string(i) + " channel=" + std::to_string(c) + " actualbits=" + std::to_string(std::bit_cast<unsigned>(actual)) + " expectedbits=" + std::to_string(std::bit_cast<unsigned>(oracle[c])));
							}
							++values;
						}
					}
					++cases;
				}
			}
	}
	require(cases == 3000,"Packed native format/address matrix incomplete");
	std::cout << "Packed native RSX evaluator:" << cases << " GPU profiles," << values << " independently checked components;all7formats/components,endian,persistent/transient,high IDs,division/modulo phases,constant inputs,dynamic ARL slots7/15,float/half NaN/Inf/signed-zero/subnormal fixtures passed\n";
}
void flat_source_and_binding(unsigned flat_vertex, bool prewarm = false)
{
	std::shared_ptr<heap_budget> budget;
	@autoreleasepool
	{
	auto owner = std::make_unique<context>(); auto& ctx = *owner;
	budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
	compiler compile(ctx); shader_cache shaders(compile);
	constexpr unsigned input = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	const std::array<std::array<unsigned,4>,2> code{{
		{(63u << 15) | (7u << 10) | (1u << 30),(1u << 22) | (input >> 9),(input & 511) << 23,(15u << 13) | (15u << 17) | (1u << 7)},
		{(63u << 15) | (7u << 10) | (1u << 30),(1u << 22) | (1u << 8) | (input >> 9),(input & 511) << 23,(15u << 13) | (15u << 17) | (1u << 7) | (1u << 2) | 1u}}};
	vertex_program_options options; options.flat_colors = true;
	const auto decoded = shaders.vertex(code,options); options.packed_inputs = true;
	const auto legacy_packed=shaders.vertex(code,options);options.flat_result_inputs=true;
	const auto jobs_before_flat_shader = compile.tasks().statistics().admitted;
	std::optional<shader_cache::prepared_vertex> prepared;
	if (prewarm)
	{
		prepared = shaders.prewarm_vertex(shaders.prepare_vertex(code,options),options);
		require(compile.tasks().statistics().admitted == jobs_before_flat_shader + 2,
			"Flat Color pipeline was not admitted with its library");
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
		while (compile.tasks().statistics().completed != compile.tasks().statistics().admitted)
		{
			require(std::chrono::steady_clock::now() < deadline,"Chained Flat Color compile did not finish without shader materialization");
			std::this_thread::yield();
		}
	}
	const auto direct = prepared ? shaders.vertex(*prepared,options) : shaders.vertex(code,options);
	require(decoded != direct && legacy_packed!=direct && shaders.vertex(code,options) == direct && shaders.vertex_statistics().entries == 3,
		"Shader cache conflated packed and decoded vertex input ABI");
	require(compile.tasks().statistics().admitted == jobs_before_flat_shader + (prewarm ? 2 : 1),
		"Cached Flat Color vertex materialization waited for its compute pipeline");
	const auto fragment = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
struct color_input { float4 color [[user(locn0),flat]]; };
fragment float4 packed_color(color_input in [[stage_in]]) { return in.color; }
)metal");
	render_pipeline_configuration config; config.colors[0] = MTLPixelFormatRGBA32Float;
	const auto pipeline = make_render_pipeline(compile,config,direct->function(),library_function(fragment,@"packed_color"));
	const auto jobs_before_bindings = compile.tasks().statistics().admitted;
	require(jobs_before_bindings == jobs_before_flat_shader + (prewarm ? 4 : 3),
		"Render pipeline setup compiled the Flat Color compute pipeline");
	std::vector<std::array<float,4>> words(14);
	words[0] = {-1,-1,.5,1}; words[1] = {3,-1,.5,1}; words[2] = {-1,3,.5,1};
	const std::array<float,4> color{flat_vertex==6 ? 1.f/7 : std::bit_cast<float>(1u),-0.f,
		flat_vertex==6 ? 2.f : std::bit_cast<float>(0x7f800000u),std::bit_cast<float>(0x7fc12345u)}; words[13] = color;
	vertex_sparse_inputs sources; sources.attribute_mask = 3;
	sources.attributes[0] = {{16u | (1u << 8) | (2u << 24) | (4u << 27),0},false,0,std::as_bytes(std::span(words))};
	sources.attributes[1] = sources.attributes[0]; sources.attributes[1].base = 7 * 16;
	if(flat_vertex!=6)
	{
		for(unsigned unit=0;unit<2;++unit){sources.attributes[unit].memory=true;sources.attributes[unit].base=unit*0x2000000ull;sources.attributes[unit].bytes={};}
		sources.read_memory=[&](unsigned address,std::span<std::byte> output)
		{
			const unsigned unit=address>=0x2000000u;
			for(std::size_t b=0;b<output.size();++b)
			{
				const auto offset=address+b-unit*0x2000000ull;const auto element=offset/16;
				require(element<=2 || element==flat_vertex,"High-ID flat fixture read an unneeded source");
				const auto local=(element==flat_vertex ? 6 : element)+unit*7;
				output[b]=std::as_bytes(std::span(words))[local*16+offset%16];
			}
		};
	}
	vertex_stream_range range; range.vertex_count = flat_vertex+1; range.source_vertices = std::make_shared<const std::vector<unsigned>>(std::vector<unsigned>{0,1,2,flat_vertex});
	const auto packed = capture_sparse_vertex_bytes(sources,range,160);
	require(packed.bytes.size() == 160 && packed.statistics.sparse_fields == 2,"Flat-only fixture did not use sparse packed records");
	id<MTLBuffer> stream = ctx.vertex_uploads().packed_bytes(packed.bytes);
	const auto request = packed.fetch;
	vertex_inputs inputs; inputs.packed = packed_vertex_inputs{request,stream,nil,range.source_vertices};
	vertex_context state; state.vertex_count = flat_vertex+1; state.output_mask = 1;
	converted_geometry geometry{list_topology::triangles,{{0,flat_vertex},{1,flat_vertex},{2,flat_vertex}},std::pair{0u,flat_vertex}};
	auto flat=std::make_optional(prepared_flat_colors::admit(ctx,geometry,84));
	require(flat->source_ids().size()==1 && flat->source_ids()[0]==flat_vertex && flat->results().length==80,"Flat-only source admission did not remain compact");
	transfer_engine transfer(ctx);const auto flat_output=make_buffer(ctx,80,buffer_storage::shared);
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageRenderTarget;
	const auto target = make_texture(ctx,descriptor);
	const auto output = make_buffer(ctx,1024,buffer_storage::shared);
	const auto gate = [ctx.device() newSharedEvent];
	ctx.begin(); ctx.wait_event(gate,1); const auto recording = ctx.recording_submission();
	unsigned rejected = 0;
	for (unsigned kind = 0; kind < 6; ++kind)
	{
		auto invalid_inputs = inputs; auto invalid_state = state; auto invalid_geometry = geometry;
		if (kind == 0) invalid_inputs.packed.reset();
		if (kind == 1) invalid_state.vertex_base = 1;
		if (kind == 2) invalid_inputs.packed->request.attributes[1].storage = unsigned(stream.length);
		if (kind == 3) invalid_geometry.vertices[0].flat_vertex = 7;
		if (kind == 5) invalid_geometry.vertices[0].flat_vertex = 3;
		try { (void)(kind == 4 ? decoded : direct)->prepare(ctx,invalid_state,invalid_geometry,invalid_inputs,kind==4 ? nullptr : &*flat); }
		catch (const std::invalid_argument&) { ++rejected; }
		catch (const std::out_of_range&) { ++rejected; }
	}
	require(rejected == 6 && recording.same_batch(ctx.recording_submission()) &&
		compile.tasks().statistics().admitted == jobs_before_bindings,
		"Invalid packed ABI/range/reference accepted, changed caller recording or started Flat Color compilation");
	const auto before = ctx.encoding_statistics();
	auto vertex = direct->prepare(ctx,state,geometry,inputs,&*flat);
	require(compile.tasks().statistics().admitted == jobs_before_bindings + (prewarm ? 0 : 1),
		"First valid Flat Color binding did not resolve exactly one compute pipeline");
	auto pass = [MTL4RenderPassDescriptor new]; pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	raster_state raster; raster.viewport = {0,0,4,4,0,1}; raster.scissor = {0,0,4,4};
	{
		render_commands commands(ctx,pass); commands.draw_geometry(geometry,pipeline.pipeline,pipeline.depth_stencil,raster,vertex,{}); commands.end();
	}
	require(ctx.encoding_statistics().compute_encoders == before.compute_encoders+1 && ctx.encoding_statistics().render_encoders == before.render_encoders + 1,
		"Separated flat result stage did not encode one prepass and one render");
	texture_region region; region.size = MTLSizeMake(4,4,1); texture_to_buffer(ctx,target,region,output,0,256);transfer.copy(ctx,flat->results(),0,flat_output,0,80); const auto work = ctx.submit();
	ctx.vertex_uploads().clear(); (void)ctx.trim_upload_pages(); flat.reset();inputs.packed.reset(); vertex = {}; stream = nil; words.assign(words.size(),{});
	bool timeout = false;
	try { work.wait(std::chrono::milliseconds(1)); } catch (const std::runtime_error&) { timeout = !work.ready(); }
	require(timeout && budget->statistics().reserved_bytes > 0,"Pending packed vertex stage lost native source/parameter backing");
	owner.reset(); gate.signaledValue = 1; work.wait();
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) for (unsigned c = 0; c < 4; ++c)
		compare(static_cast<const float*>(output.contents)[y * 64 + x * 4 + c],color[c],2);
	const auto* flat_record=static_cast<const flat_color_result*>(flat_output.contents);
	require(flat_record->fault==0 && flat_record->padding==std::array<unsigned,3>{},"Native flat color result ABI fault/padding differs");
	std::cout << (prewarm ? "Chained Flat Color pipeline: library and compute admitted together, native compute completed before shader materialization, no duplicate job on first valid use\n" :
		"Cached Flat Color pipeline: render PSO usable before compute admission, invalid bindings start no job, one compute job on first valid use\n");
	for(unsigned c=0;c<4;++c)compare(std::bit_cast<float>(flat_record->colors[0][c]),color[c],2);
	}
	require(budget->statistics().reserved_bytes == 0,"Packed native vertex stage orphan retained backing after final ownership release");
	std::cout << "Separated flat/sparse packed vertex stage:source"<<flat_vertex<<" outside triangle0/1/2,160byte sparse atlas,one80byte flat result/4byteID under8MiB;16RGBA32Fpixel andrecord oracles with finite/signed-zero bits/NaN classification;one flatcompute/one render,no attribute decode;three shader ABI variants,six guards;gated draw survives result/ID/stream packet/cache/CPU source/context destruction andtimeout,all native charges retire passed\n";
}

void sparse_formats()
{
	context ctx; compiler compile(ctx); const auto pipeline = probe(compile,7,false,true);
	unsigned cases = 0, values = 0; std::array<bool,3> modes{};
	for (unsigned type = 1; type <= 7; ++type) for (unsigned count = 1; count <= (type == 6 ? 1u : 4u); ++count)
		for (bool be : {false,true}) for (unsigned phase = 0; phase < 6; ++phase)
			for (const auto& ids : {std::vector<unsigned>{0,1,2,6},std::vector<unsigned>{65535,65536,65537},std::vector<unsigned>{0,1,1048575}})
			{
				@autoreleasepool
				{
					const unsigned frequency = std::array{0u,1u,3u,7u,5u,65535u}[phase];
					const bool modulo = phase >= 3;
					const int index_offset = phase == 4 ? 3 : 0;
					const unsigned base = phase == 2 && ids.front() ? ids.front() - 2 : 0;
					const unsigned width = type == 2 || type == 6 ? 4 : type == 4 || type == 7 ? 1 : 2, payload = width * count, stride = payload + 3;
					vertex_stream_range range; range.first_vertex = ids.front(); range.vertex_count = ids.back() - ids.front() + 1;
					range.vertex_base_index = base; range.vertex_index_offset = index_offset;
					range.source_vertices = std::make_shared<const std::vector<unsigned>>(ids);
					vertex_sparse_inputs sources; sources.attribute_mask = 1u << 7;
					sources.attributes[7] = {{stride | (frequency << 8) | (type << 24) | (count << 27),(be ? 1u << 29 : 0) | (modulo ? 1u << 31 : 0)},true,4096,{}};
					sources.read_memory = [&](unsigned address,std::span<std::byte> bytes)
					{
						for (std::size_t i = 0; i < bytes.size(); ++i)
						{
							const auto offset = address + i - 4096, element = offset / stride, component = (offset % stride) / width, byte = (offset % stride) % width;
							require(offset % stride < payload,"Sparse native probe fetched an unneeded source gap");
							bytes[i] = std::byte(pattern(type,element,component) >> (8 * (be ? width - 1 - byte : byte)));
						}
					};
					const auto packed = capture_sparse_vertex_bytes(sources,range,256);
					modes[packed.fetch.attributes[7].storage_mode()] = true;
					const auto stream = make_upload(ctx,packed.bytes), source_ids = make_upload(ctx,std::as_bytes(std::span(ids)));
					validate_vertex_fetch(ctx,packed.fetch,stream,nil);
					const auto output = make_buffer(ctx,ids.size() * 16,buffer_storage::shared), faults = make_buffer(ctx,ids.size() * 4,buffer_storage::shared);
					ctx.begin(); const std::array<std::byte,16> zero{}; const auto dummy = ctx.upload(zero,16);
					const auto& r = packed.fetch;
					const vertex_fetch_parameters parameters{r.attributes,stream.length,0,r.first_vertex,r.vertex_count,r.vertex_base_index,r.vertex_index_offset,0,0,r.attribute_mask,0,r.element_counts};
					vertex_context state; state.vertex_base = r.first_vertex; state.vertex_count = r.vertex_count;
					const std::array<sampler_parameters,4> sampling{};
					const auto p = ctx.upload(std::as_bytes(std::span{&parameters,1}),256), v = ctx.upload(std::as_bytes(std::span{&state,1}),256), s = ctx.upload(std::as_bytes(std::span(sampling)),16);
					bindings resources; resources.set_buffer(0,stream); resources.set_buffer(1,dummy.buffer,dummy.offset); resources.set_buffer(2,p.buffer,p.offset);
					resources.set_buffer(3,output); resources.set_buffer(4,v.buffer,v.offset); resources.set_buffer(5,s.buffer,s.offset); resources.set_buffer(6,faults); resources.set_buffer(7,source_ids);
					const auto table = resources.snapshot(ctx); const auto encoder = ctx.compute(); ctx.keep_alive(pipeline);
					[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
					[encoder dispatchThreads:MTLSizeMake(ids.size(),1,1) threadsPerThreadgroup:MTLSizeMake(ids.size(),1,1)]; ctx.submit().wait();
					for (std::size_t i = 0; i < ids.size(); ++i)
					{
						require(static_cast<const unsigned*>(faults.contents)[i] == 0,"Sparse translated evaluator faulted");
						const unsigned element = !frequency ? 0 : modulo ? (ids[i] + index_offset) % frequency : (ids[i] - base) / frequency;
						const auto reference = expected(type,count,element);
						for (unsigned c = 0; c < 4; ++c) { compare(static_cast<const float*>(output.contents)[i * 4 + c],reference[c],type); ++values; }
					}
					++cases;
				}
			}
	require(cases == 900 && std::all_of(modes.begin(),modes.end(),[](bool value) { return value; }),"Sparse GPU matrix omitted a storage mode/profile");
	std::cout << "Sparse native RSX evaluator:" << cases << " GPU profiles," << values << " independent component oracles;all7formats/components/endian,dense/window/sorted sparse records,original high/wrapped IDs,base quotient/modulo offset/constant phases passed\n";
}

void flat_faults_and_guards()
{
	context ctx(1);auto budget=std::make_shared<heap_budget>(ctx,8u<<20);ctx.set_buffer_budget(budget);compiler compile(ctx);transfer_engine transfer(ctx);
	const converted_geometry geometry{list_topology::triangles,{{0,6},{1,6},{2,6}},std::nullopt};
	const auto before_budget=budget->statistics().created;bool budget_refused=false;
	try{(void)prepared_flat_colors::admit(ctx,geometry,83);}catch(const std::length_error&){budget_refused=true;}
	require(budget_refused && budget->statistics().created==before_budget,"Flat admission budget failure allocated native storage");
	const unsigned operand=2u|(1u<<12)|(2u<<10)|(3u<<8);
	const auto mov=[&](unsigned unit,unsigned output,bool end=false)
	{
		return std::array<unsigned,4>{(63u<<15)|(7u<<10)|(1u<<30),(1u<<22)|(unit<<8)|(operand>>9),(operand&511)<<23,(15u<<13)|(15u<<17)|(1u<<7)|(output<<2)|unsigned(end)};
	};
	const auto fragment=compile.library("#include <metal_stdlib>\nusing namespace metal;\nfragment float4 fault_color(){return float4(1);}\n");
	auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget;
	const auto target=make_texture(ctx,descriptor);const auto output=make_buffer(ctx,1104,buffer_storage::shared);
	unsigned faults=0,guards=0;
	for(unsigned expected:{0u,1u,2u,4u,8u,16u,32u})
	{
		std::vector<std::array<unsigned,4>> code{mov(0,0),mov(1,1),mov(1,2),mov(1,3),mov(1,4,true)};
		vertex_program_options options;options.flat_colors=true;options.packed_inputs=true;options.flat_result_inputs=true;
		std::array<std::array<std::array<float,4>,8>,3> values{};
		values[0][0]={-1,-1,.5,1};values[0][1]={3,-1,.5,1};values[0][2]={-1,3,.5,1};
		for(auto& value:values[1])value={.25,.5,.75,1};
		if(expected==1 || expected==32)
		{
			auto arl=mov(2,0);arl[0]=(7u<<10);arl[1]=(13u<<22)|(2u<<8)|(operand>>9);
			auto color=mov(expected==1 ? 0 : 1,1,true);if(expected==1)color[0]|=1u<<27;
			code={mov(0,0),arl,color};values[2][6][0]=expected==1 ? 17.f : std::bit_cast<float>(0x7fc12345u);
		}
		if(expected==2)
		{
			auto constant=mov(0,1,true);const auto source=(operand&~3u)|3u;
			constant[1]=(1u<<22)|(source>>9);constant[2]=(source&511)<<23;code={mov(0,0),constant};
		}
		if(expected==4){code={std::array<unsigned,4>{7u<<10,8u<<27,0,0}};options.base_address=1;}
		if(expected==8)code={std::array<unsigned,4>{7u<<10,10u<<27,0,0}};
		if(expected==16)code={std::array<unsigned,4>{7u<<10,9u<<27,0,0}};
		vertex_shader shader(compile,code,options);render_pipeline_configuration config;config.colors[0]=MTLPixelFormatRGBA32Float;
		const auto pipeline=make_render_pipeline(compile,config,shader.function(),library_function(fragment,@"fault_color"));
		vertex_fetch_request request;request.vertex_count=7;request.attribute_mask=7;
		for(unsigned unit=0;unit<3;++unit)request.attributes[unit]={16u|(1u<<8)|(2u<<24)|(4u<<27),unit*128};
		vertex_inputs data;data.packed=packed_vertex_inputs{request,make_upload(ctx,std::as_bytes(std::span(values))),nil};
		vertex_context state;state.vertex_count=7;state.output_mask=1;state.instruction_budget=32;
		auto flat=prepared_flat_colors::admit(ctx,geometry,84);
		ctx.begin();const auto token=ctx.recording_submission();
		if(!expected)
		{
			const auto refusal=[&](auto action)
			{
				bool refused=false;try{action();}catch(const std::exception&){refused=true;}
				require(refused && token.same_batch(ctx.recording_submission()),"Invalid flat preparation changed caller recording");++guards;
			};
			refusal([&]{(void)shader.prepare(ctx,state,geometry,data);});
			auto changed=geometry;changed.vertices[0].flat_vertex=5;
			refusal([&]{(void)shader.prepare(ctx,state,changed,data,&flat);});
			auto moved=std::move(flat);refusal([&]{(void)shader.prepare(ctx,state,geometry,data,&flat);});flat=std::move(moved);
			auto invalid_state=state;invalid_state.vertex_count=6;refusal([&]{(void)shader.prepare(ctx,invalid_state,geometry,data,&flat);});
			auto invalid_data=data;invalid_data.packed->request.attributes[0].storage=unsigned(data.packed->persistent.length);
			refusal([&]{(void)shader.prepare(ctx,state,geometry,invalid_data,&flat);});
			auto no_flat=options;no_flat.flat_result_inputs=false;vertex_shader legacy(compile,code,no_flat);
			refusal([&]{(void)legacy.prepare(ctx,state,geometry,data,&flat);});
			auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
			{render_commands draw(ctx,pass);refusal([&]{(void)shader.prepare(ctx,state,geometry,data,&flat);});draw.end();}
		}
		const auto before=ctx.encoding_statistics();const auto vb=shader.prepare(ctx,state,geometry,data,&flat);
		require(ctx.encoding_statistics().compute_encoders==before.compute_encoders+1,"Flat preparation did not execute exactly one prepass");
		bool consumed=false;try{(void)shader.prepare(ctx,state,geometry,data,&flat);}catch(const std::invalid_argument&){consumed=true;}
		require(consumed && token.same_batch(ctx.recording_submission()),"Consumed flat result accepted a second evaluation");
		transfer.copy(ctx,flat.results(),0,output,1024,80);
		auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=target;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(0,0,0,0);
		raster_state raster;raster.viewport={0,0,4,4,0,1};raster.scissor={0,0,4,4};
		{render_commands draw(ctx,pass);draw.draw_geometry(geometry,pipeline.pipeline,pipeline.depth_stencil,raster,vb,{});draw.end();}
		texture_to_buffer(ctx,target,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)},output,0,256);ctx.submit().wait();
		const auto* result=reinterpret_cast<const flat_color_result*>(static_cast<const std::byte*>(output.contents)+1024);
		require(result->fault==expected && result->padding==std::array<unsigned,3>{},"Flat result lost attribute/constant/flow/stack/budget/address status");
		if(!expected)for(const auto& color:result->colors)for(unsigned c=0;c<4;++c)
			require(color[c]==std::bit_cast<unsigned>(std::array<float,4>{.25,.5,.75,1}[c]),"Native flat result did not retain all four color registers");
		for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)for(unsigned c=0;c<4;++c)
			require(static_cast<const float*>(output.contents)[y*64+x*4+c]==(expected ? 0.f : 1.f),"Flat evaluator fault did not preserve vertex clipping");
		if(expected)++faults;
		if(!expected)
		{
			for(bool empty:{false,true})
			{
				auto g=geometry;auto predicate=state;
				if(empty)for(auto& ref:g.vertices)ref.flat_vertex=ref.vertex;else predicate.predicate=0;
				auto preparation=prepared_flat_colors::admit(ctx,g,empty ? 0 : 84);
				const auto prior=ctx.encoding_statistics();ctx.begin();(void)shader.prepare(ctx,predicate,g,data,&preparation);
				require(ctx.encoding_statistics().compute_encoders==prior.compute_encoders,"Empty/disabled flat stage encoded compute work");
				const auto cancelled=ctx.recording_submission();ctx.cancel();require(cancelled.cancelled(),"Cancelled flat preparation retained an active batch");
				ctx.begin();bool refused=false;try{(void)shader.prepare(ctx,predicate,g,data,&preparation);}catch(const std::invalid_argument&){refused=true;}
				require(refused,"Cancelled flat results were reused with stale state");ctx.cancel();
			}
			auto decoded_options=options;decoded_options.packed_inputs=false;vertex_shader decoded(compile,code,decoded_options);
			std::vector<std::array<float,4>> decoded_words(1+7*16);
			for(unsigned id=0;id<7;++id)for(unsigned unit=0;unit<3;++unit)decoded_words[1+id*16+unit]=values[unit][id];
			vertex_inputs decoded_inputs;decoded_inputs.attributes=make_upload(ctx,std::as_bytes(std::span(decoded_words)));decoded_inputs.attribute_offset=16;
			auto decoded_flat=prepared_flat_colors::admit(ctx,geometry,84);
			ctx.begin();(void)decoded.prepare(ctx,state,geometry,decoded_inputs,&decoded_flat);transfer.copy(ctx,decoded_flat.results(),0,output,1024,80);ctx.submit().wait();
			const auto* decoded_result=reinterpret_cast<const flat_color_result*>(static_cast<const std::byte*>(output.contents)+1024);
			require(decoded_result->fault==0,"Decoded flat ABI produced a source fault");
			for(const auto& color:decoded_result->colors)for(unsigned c=0;c<4;++c)
				require(color[c]==std::bit_cast<unsigned>(std::array<float,4>{.25,.5,.75,1}[c]),"Decoded flat ABI lost its input offset/color registers");
		}
	}
	require(faults==6 && guards==7,"Native flat fault/guard matrix incomplete");
	std::cout<<"Native flat status:all6attribute/constant/flow/stack/budget/addressfaults exact andall16pixels clipped,including flat-only faults;zero-fault visible;7preflight guards preserve recording,83bytebudget rejects before nativeallocation,consumed/moved/cancelled ownership guards,empty/predicate-disabled no-compute;all4color registers inpacked/decoded ABIs,decoded16byteinputoffset passed\n";
}
}
int main(int argc,char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try { if(argc==2&&std::string_view(argv[1])=="--flat-compiler-chain"){flat_source_and_binding(6,true);flat_source_and_binding(1048575,true);return 0;}if(argc==2&&std::string_view(argv[1])=="--address-rounding"){address_rounding();return 0;}if(argc==2&&std::string_view(argv[1])=="--indexed-outputs"){indexed_outputs();return 0;}if(argc==2&&std::string_view(argv[1])=="--indexed-output-dual"){indexed_outputs(true);return 0;}if(argc==2&&std::string_view(argv[1])=="--scalar-dual-destinations"){scalar_dual_destinations();return 0;} if(argc!=1)throw std::invalid_argument("Unknown packed vertex selection");address_rounding();indexed_outputs();scalar_dual_destinations();condition_updates(); address_condition_updates(); masked_scalar_swizzles(); formats(); sparse_formats(); flat_source_and_binding(6);flat_source_and_binding(1048575);flat_source_and_binding(6,true);flat_source_and_binding(1048575,true); flat_faults_and_guards(); return 0; }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
