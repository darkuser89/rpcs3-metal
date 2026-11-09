#include "M4CompilerGateFixture.h"
#include "M4RenderPipeline.h"
#include "M4ShaderCache.h"
#include "M4Texture.h"
#include <bit>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refuses(F&& operation, const char* message)
{
	try { operation(); } catch (const std::exception&) { return; }
	throw std::runtime_error(message);
}
struct release_gate
{
	M4GatedCompiler* gate;
	~release_gate() { [gate releaseCallbacks]; }
};
std::vector<std::byte> fragment_code(std::array<float,4> color)
{
	std::vector<std::byte> bytes;
	const auto append = [&](unsigned word) { for (unsigned shift : {8u,0u,24u,16u}) bytes.push_back(std::byte(word >> shift)); };
	for (unsigned word : {(1u<<24)|(15u<<9)|1u,(0xe4u<<9)|(7u<<18)|(0xe4u<<21)|2u,0xe4u<<9,0xe4u<<9}) append(word);
	for (float value : color) append(std::bit_cast<unsigned>(value));
	return bytes;
}
void profile(bool fragment_first, bool clear, bool previous_policy, bool capacity = false)
{
	context ctx(1);
	NSError* error = nil;
	const auto native = [ctx.device() newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
	require(native != nil, "Failed to create native Metal4 compiler");
	const auto gate = [[M4GatedCompiler alloc] initWithNative:native];
	compiler compile(ctx,256,16u<<20,(id<MTL4Compiler>)gate);
	shader_cache shaders(compile), foreign(compile);
	render_pipeline_cache pipelines(compile,2), other(compile);
	release_gate release{gate};
	constexpr unsigned xyzw = 2u|(1u<<12)|(2u<<10)|(3u<<8);
	std::array<std::array<unsigned,4>,1> code{{{(63u<<15)|(7u<<10)|(1u<<30),(1u<<22)|(xyzw>>9),(xyzw&511)<<23,(15u<<13)|(15u<<17)|(1u<<7)|1u}}};
	auto bytes = fragment_code({0,1,0,1});
	vertex_program_options vo;
	fragment_program_options fo;
	auto vp = shaders.prepare_vertex(code,vo);
	auto fp = shaders.prepare_fragment(bytes,fo);
	require(!shaders.pipeline_stages(vp,fp) && compile.tasks().statistics().admitted == 0,
		"CPU-only preparation admitted native tasks");
	vp = shaders.prewarm_vertex(vp,vo);
	fp = shaders.prewarm_fragment(fp,fo);
	[gate waitForCallbacks:2];
	const auto stages = shaders.pipeline_stages(vp,fp);
	require(stages && !shaders.ready(vp) && !shaders.ready(fp), "Gate failed to retain both actual library results");
	render_pipeline_configuration config;
	config.colors[0] = MTLPixelFormatRGBA8Unorm;
	config.state.stencil_test = true;
	config.state.two_sided_stencil = true;
	config.state.front.reference = 17; config.state.back.reference = 23;
	if (capacity)
	{
		std::vector<render_pipeline_cache::prepared_pipeline> pending;
		for (unsigned i = 0; i < 62; ++i)
		{
			auto variant = config;
			variant.state.depth_test = true;
			variant.state.depth_compare = 0x200 + i % 8;
			variant.state.colors[0].write_mask = i / 8;
			const auto ticket = pipelines.prewarm(variant,*stages);
			require(ticket.has_value(), "Dependent pipeline capacity refused work before the native bound");
			pending.push_back(*ticket);
		}
		auto overflow = config; overflow.state.depth_test = true;
		overflow.state.depth_compare = 0x206; overflow.state.colors[0].write_mask = 7;
		require(!pipelines.prewarm(overflow,*stages) && compile.tasks().statistics().admitted == 64 &&
			compile.tasks().statistics().queued == 62 && ![gate renderLaunches], "Dependent pipeline admission exceeded the 64-job native bound");
		[gate releaseCallbacks];
		for (const auto& ticket : pending) require(pipelines.get(ticket).pipeline != nil, "Admitted dependent pipeline failed after capacity recovery");
		require([gate renderLaunches] == 62 && compile.tasks().statistics().completed == 64 &&
			pipelines.statistics().entries == 2 && pipelines.statistics().evicted == 60 &&
			!compile.tasks().statistics().active && !compile.tasks().statistics().queued,
			"Dependent pipeline completion lost native jobs or two-entry cache bounds");
		std::cout << "Dependent render capacity:two held library callbacks,62 queued PSOs,65th job refused,62 native completions and two-entry LRU passed\n";
		return;
	}
	std::optional<render_pipeline_cache::prepared_pipeline> ticket;
	// Isolated former admission policy, for a deterministic negative control.
	// Both policies execute against the same real libraries and current core.
	if (!previous_policy || (shaders.ready(vp) && shaders.ready(fp))) ticket = pipelines.prewarm(config,*stages);
	require(ticket.has_value(), "Pending-stage render pipeline was not admitted");
	auto dynamic = config; dynamic.state.front.reference = 93; dynamic.state.back.reference = 41;
	const auto duplicate = pipelines.prewarm(dynamic,*stages);
	require(duplicate && compile.tasks().statistics().admitted == 3 && pipelines.statistics().prewarmed == 1 &&
		compile.tasks().statistics().active == 2 && compile.tasks().statistics().queued == 1 && ![gate renderLaunches],
		"Dependent PSO did not deduplicate while retaining two distinct library dependencies");
	for (unsigned contract = 0; contract < 3; ++contract)
	{
		auto invalid = config;
		if (contract == 0) invalid.shader_half_storage_mask = 1;
		if (contract == 1) invalid.state.sample_count = 2;
		if (contract == 2) invalid.shader_logic.enabled = true;
		refuses([&] { (void)pipelines.prewarm(invalid,*stages); }, "Pending-stage admission ignored fragment contract");
	}
	refuses([&] { (void)foreign.pipeline_stages(vp,fp); }, "Foreign shader-cache tickets were accepted");
	refuses([&] { (void)other.get(*ticket,std::chrono::milliseconds(0)); }, "Foreign pipeline-cache ticket was accepted");
	require(compile.tasks().statistics().admitted == 3, "Rejected contract admitted native work");
	code[0].fill(0xffffffffu); std::fill(bytes.begin(),bytes.end(),std::byte{0xff});
	[gate releaseLibraryCallback:fragment_first ? @"RPCS3 native RSX fragment program" : @"RPCS3 native RSX vertex program"];
	const auto half = compile.tasks().statistics();
	require(half.completed == 1 && half.active == 1 && half.queued == 1 && ![gate renderLaunches] && !ticket->ready(),
		"Native render PSO launched before both library results");
	if (clear) { pipelines.clear(); shaders.clear(); compile.clear_libraries(); }
	[gate releaseLibraryCallback:fragment_first ? @"RPCS3 native RSX vertex program" : @"RPCS3 native RSX fragment program"];
	[gate waitForCallbacks:1];
	require([gate renderLaunches] == 1 && !ticket->ready() && !shaders.vertex_statistics().compiled && !shaders.fragment_statistics().compiled,
		"Native render PSO did not launch directly from library completion before stage materialization");
	if (!clear)
	{
		const auto promoted = pipelines.prewarm(config,*stages);
		require(promoted && !promoted->ready() && compile.tasks().statistics().admitted == 3,
			"Ready-stage lookup duplicated the dependent native PSO");
	}
	refuses([&] { (void)pipelines.get(*ticket,std::chrono::milliseconds(0)); }, "Held real PSO callback did not preserve timeout");
	[gate releaseCallbacks];
	const auto bundle = pipelines.get(*ticket);
	const auto other_refs = pipelines.get(*duplicate);
	require(bundle.pipeline == other_refs.pipeline && bundle.state.front.reference == 17 && bundle.state.back.reference == 23 &&
		other_refs.state.front.reference == 93 && other_refs.state.back.reference == 41,
		"Dependent PSO lost per-ticket stencil references or identity");
	const auto vs = shaders.vertex(vp,vo);
	const auto fs = shaders.fragment(fp,fo);
	if (!clear)
	{
		const auto lookups = pipelines.statistics().descriptor_lookups;
		for (unsigned i = 0; i < 100; ++i) require(pipelines.get(dynamic,vs->function(),*fs).pipeline == bundle.pipeline,
			"Ready lookup lost adopted dependent pipeline");
		require(pipelines.statistics().descriptor_lookups == lookups && pipelines.statistics().entries == 1,
			"Adopted dependent PSO missed the ready fast path");
	}
	else require(!pipelines.statistics().entries && !shaders.vertex_statistics().entries && !shaders.fragment_statistics().entries,
		"Old-generation owners repopulated cleared caches");
	require(compile.tasks().statistics().admitted == 3 && ![gate mainThreadLaunch], "Stage materialization duplicated tasks or launched compiler on owner");
	std::array<std::array<float,4>,48> vertices{};
	vertices[0] = {-1,-1,.5f,1}; vertices[16] = {3,-1,.5f,1}; vertices[32] = {-1,3,.5f,1};
	const auto upload = make_upload(ctx,std::as_bytes(std::span{vertices}));
	const auto readback = make_buffer(ctx,1024,buffer_storage::shared);
	const auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageRenderTarget;
	const auto target = make_texture(ctx,td);
	const auto sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatStencil8 width:4 height:4 mipmapped:NO];
	sd.storageMode = MTLStorageModePrivate; sd.usage = MTLTextureUsageRenderTarget;
	const auto stencil = make_texture(ctx,sd);
	const auto geometry = convert_array_geometry(guest_topology::triangles,0,3,provoking_vertex::last);
	ctx.begin();
	vertex_context state; state.vertex_count = 3;
	const auto vb = vs->prepare(ctx,state,geometry,vertex_inputs{.attributes=upload});
	const auto current = fragment_code({1,0,0,1});
	const auto fb = fs->prepare(ctx,{},current,{});
	const auto pass = [MTL4RenderPassDescriptor new];
	pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	pass.stencilAttachment.texture = stencil; pass.stencilAttachment.loadAction = MTLLoadActionClear;
	pass.stencilAttachment.storeAction = MTLStoreActionDontCare;
	raster_state raster; raster.viewport = {0,0,4,4,0,1}; raster.scissor = {0,0,4,4};
	raster.front_stencil_reference = bundle.state.front.reference; raster.back_stencil_reference = bundle.state.back.reference;
	render_commands commands(ctx,pass);
	commands.draw_geometry(geometry,bundle.pipeline,bundle.depth_stencil,raster,vb,fb); commands.end();
	texture_region region; region.size = MTLSizeMake(4,4,1);
	texture_to_buffer(ctx,target,region,readback,0,256,1024); ctx.submit().wait();
	const auto result = map_buffer(readback,0,1024);
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) for (unsigned c = 0; c < 4; ++c)
		require(result[y*256+x*4+c] == std::byte(c == 0 || c == 3 ? 255 : 0), "Deferred native PSO changed exact pixels or dynamic fragment literals");
	std::cout << "Deferred render PSO:fragment_first=" << fragment_first << ",clear=" << clear
		<< ";both native library dependencies,one PSO before stage materialization,contract rejection,deduplication,stencil references,timeout,owner lifetime and 16 exact dynamic-literal pixels passed\n";
}
}
int main(int argc, char** argv)
{
	@autoreleasepool
	{
		try
		{
			const bool previous = argc == 2 && std::string_view(argv[1]) == "--previous-stage-gate";
			for (bool first : {false,true}) for (bool clear : {false,true}) profile(first,clear,previous);
			if (!previous) profile(false,false,false,true);
			return 0;
		}
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
