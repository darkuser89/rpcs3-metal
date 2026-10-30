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

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
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
void flat_source_and_binding(unsigned flat_vertex)
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
	const auto direct = shaders.vertex(code,options);
	require(decoded != direct && legacy_packed!=direct && shaders.vertex(code,options) == direct && shaders.vertex_statistics().entries == 3,
		"Shader cache conflated packed and decoded vertex input ABI");
	const auto fragment = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
struct color_input { float4 color [[user(locn0),flat]]; };
fragment float4 packed_color(color_input in [[stage_in]]) { return in.color; }
)metal");
	render_pipeline_configuration config; config.colors[0] = MTLPixelFormatRGBA32Float;
	const auto pipeline = make_render_pipeline(compile,config,direct->function(),library_function(fragment,@"packed_color"));
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
	require(rejected == 6 && recording.same_batch(ctx.recording_submission()),"Invalid packed ABI/range/reference accepted or caller recording changed");
	const auto before = ctx.encoding_statistics();
	auto vertex = direct->prepare(ctx,state,geometry,inputs,&*flat);
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
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try { formats(); sparse_formats(); flat_source_and_binding(6);flat_source_and_binding(1048575); flat_faults_and_guards(); return 0; }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
