#include "M4GeometryBinding.h"
#include "M4VertexShader.h"
#include "M4RenderPipeline.h"
#include "M4Varyings.h"
#include "M4ConstantUpload.h"
#include "M4HeapBudget.h"
#include "M4Query.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <sstream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::array<unsigned,4> instruction(unsigned opcode,unsigned input,unsigned output,bool constant = false,bool end = false)
{
	const unsigned source = (constant ? 3u : 2u) | (1u << 12) | (2u << 10) | (3u << 8);
	return {(63u << 15) | (7u << 10) | (1u << 30),(opcode << 22) | (input << 8) | (source >> 9),
		(source & 511) << 23,(15u << 13) | (15u << 17) | (1u << 7) | (output << 2) | unsigned(end)};
}
id<MTLTexture> target(context& ctx)
{
	auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:16 height:16 mipmapped:NO];
	d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageRenderTarget;
	return make_texture(ctx,d);
}
MTL4RenderPassDescriptor* pass(id<MTLTexture> texture)
{
	auto d = [MTL4RenderPassDescriptor new]; d.colorAttachments[0].texture = texture;
	d.colorAttachments[0].clearColor = MTLClearColorMake(0,0,0,0);
	d.colorAttachments[0].loadAction = MTLLoadActionClear; d.colorAttachments[0].storeAction = MTLStoreActionStore; return d;
}
raster_state raster()
{
	raster_state r; r.viewport = {0,0,16,16,0,1}; r.scissor = {0,0,16,16}; return r;
}
std::vector<std::byte> attributes()
{
	std::array<std::array<std::array<float,4>,8>,4> values{};
	values[0] = {{{-1,-1,.5,1},{1,-1,.5,1},{1,1,.5,1},{-1,1,.5,1},{0,0,.25,1},{-.5,.5,.75,1},{.5,-.5,.1,1},{.5,.5,.9,1}}};
	for (unsigned i = 0; i < 8; ++i)
	{
		values[1][i] = {float(i)/8,float(7-i)/8,.25,1};
		values[2][i] = {.25f + (i % 2) * .5f,.25f + ((i / 2) % 2) * .5f,0,0};
		values[3][i] = {.5f,(i == 2 || i == 5) ? -1.f : 1.f,1,1};
	}
	const auto bytes = std::as_bytes(std::span(values)); return {bytes.begin(),bytes.end()};
}
vertex_inputs inputs(context& ctx, unsigned base)
{
	vertex_inputs result;
	vertex_fetch_request request; request.first_vertex = base; request.vertex_count = 8; request.vertex_base_index = base; request.attribute_mask = 15;
	for (unsigned unit = 0; unit < 4; ++unit) request.attributes[unit] = {16u | (1u << 8) | (2u << 24) | (4u << 27),unit * 128};
	result.packed = packed_vertex_inputs{request,make_upload(ctx,attributes()),nil};
	const std::array constants{std::array<float,4>{.125,.25,.375,.5}}; result.constants = make_upload(ctx,std::as_bytes(std::span(constants)));
	auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:2 height:2 mipmapped:NO];
	d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead;
	result.textures[0] = make_texture(ctx,d);
	const std::array pixels{std::array<float,4>{1,0,0,1},std::array<float,4>{0,1,0,1},std::array<float,4>{0,0,1,1},std::array<float,4>{1,1,0,1}};
	std::array<std::byte,512> staging{};
	std::memcpy(staging.data(),pixels.data(),32); std::memcpy(staging.data()+256,pixels.data()+2,32);
	const auto upload = make_upload(ctx,staging);
	ctx.begin(); buffer_to_texture(ctx,upload,0,result.textures[0],{0,0,MTLOriginMake(0,0,0),MTLSizeMake(2,2,1)},256); ctx.submit().wait();
	auto sampler = [MTLSamplerDescriptor new]; sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterNearest;
	result.samplers[0] = make_sampler(ctx,sampler); return result;
}
void compare_native_draws()
{
	context ctx; auto ledger = std::make_shared<heap_budget>(ctx,16u << 20); ctx.set_buffer_budget(ledger); compiler compile(ctx);
	const auto a = target(ctx), b = target(ctx);
	const auto readback = make_buffer(ctx,8192,buffer_storage::shared);
	const std::array code{instruction(1,0,0),instruction(1,1,1),instruction(25,2,7),instruction(1,0,8,true),instruction(1,3,5,false,true)};
	unsigned profiles = 0, components = 0, indexed = 0, array = 0, flat_passes=0;
	for (bool flat : {false,true}) for (unsigned kind = 0; kind < 3; ++kind)
	{
		vertex_program_options options; options.flat_colors = flat; options.packed_inputs = true; options.point_output = kind == 0;
		vertex_shader vertex(compile,code,options);
		auto separated_options=options;separated_options.flat_result_inputs=flat;
		vertex_shader separated(compile,code,separated_options);
		std::ostringstream source; source << "#include <metal_stdlib>\nusing namespace metal;\n"; emit_varyings(source,flat,true);
		source << "fragment float4 indexed_color(rsx_fragment_inputs in [[stage_in]]) { return in.color0*.25f + in.tex0*.5f + in.tex1*.25f; }\n";
		const auto library = compile.library(source.str()); render_pipeline_configuration configuration;
		configuration.colors[0] = MTLPixelFormatRGBA32Float;
		configuration.topology = kind == 0 ? MTLPrimitiveTopologyClassPoint : kind == 1 ? MTLPrimitiveTopologyClassLine : MTLPrimitiveTopologyClassTriangle;
		const auto pipeline = make_render_pipeline(compile,configuration,vertex.function(),library_function(library,@"indexed_color"));
		const auto separated_pipeline=make_render_pipeline(compile,configuration,separated.function(),library_function(library,@"indexed_color"));
		for (unsigned topology = 0; topology < 10; ++topology)
		{
			const auto native_kind = topology == 0 ? 0 : topology <= 3 ? 1 : 2; if (kind != native_kind) continue;
			for (auto provoking : {provoking_vertex::first,provoking_vertex::last}) for (unsigned profile = 0; profile < 3; ++profile) for (unsigned base : {0u,65535u})
			{
				@autoreleasepool
				{
					std::vector<unsigned> ids = profile == 0 ? std::vector<unsigned>{0,1,2,3,4,5,6,7} : profile == 1 ? std::vector<unsigned>{99,0,1,2,3,3,99,99,4,5,6,7,99} : std::vector<unsigned>{0,1,1,2,2,3,3,0};
					for (auto& id : ids) if (id != 99) id += base;
					auto geometry = convert_geometry(static_cast<guest_topology>(topology),ids,provoking,profile == 1 ? std::optional{99u} : std::nullopt);
					const auto plan = plan_indexed_references(geometry,flat); const auto native = prepared_geometry::prepare(ctx,plan);
					std::optional<prepared_flat_colors> flat_colors;if(flat)flat_colors.emplace(prepared_flat_colors::admit(ctx,native.references()));
					if (plan.index_width()) ++indexed; else ++array;
					const auto data = inputs(ctx,base); vertex_context state; state.vertex_base = base; state.vertex_count = 8; state.constant_count = 1; state.point_size = 2;
					state.output_mask = 1 | 16 | (1u << 14) | (1u << 15) | 64;
					state.clip_enable = profile == 2 ? 1 : 0; state.clip_factors.fill(1);
					auto r = raster(); if (profile == 1) { r.cull = MTLCullModeBack; r.front_face = provoking == provoking_vertex::first ? MTLWindingClockwise : MTLWindingCounterClockwise; }
					if (profile == 2) r.scissor = {2,2,12,12};
					ctx.begin(); const auto vb = vertex.prepare(ctx,state,geometry,data);
					{ render_commands draw(ctx,pass(a)); draw.draw_geometry(geometry,pipeline.pipeline,pipeline.depth_stencil,r,vb,{}); draw.end(); }
					ctx.submit().wait();
					const auto before=ctx.encoding_statistics();
					ctx.begin(); const auto optimized = separated.prepare(ctx,state,native.references(),data,flat_colors ? &*flat_colors : nullptr);
					const bool has_flat=flat_colors && !flat_colors->source_ids().empty();
					require(ctx.encoding_statistics().compute_encoders==before.compute_encoders+has_flat,"Flat source set did not select exactly one native prepass");flat_passes+=has_flat;
					{ render_commands draw(ctx,pass(b)); draw.draw_geometry(native,separated_pipeline.pipeline,separated_pipeline.depth_stencil,r,optimized,{}); draw.end(); }
					texture_to_buffer(ctx,a,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1)},readback,0,256);
					texture_to_buffer(ctx,b,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1)},readback,4096,256); ctx.submit().wait();
					require(std::memcmp(readback.contents,static_cast<const std::byte*>(readback.contents)+4096,4096)==0,"Native indexed pixels differ from expanded RSX reference:topology/flat/restart/degenerate/winding/clip/texture/constants");
					if (!profile) require(std::any_of(static_cast<const float*>(readback.contents),static_cast<const float*>(readback.contents)+1024,[](float v) { return v!=0; }),"Reference fixture produced no visible geometry");
					components += 1024; ++profiles;
				}
			}
		}
	}
	require(profiles == 240 && indexed && array,"Native indexed geometry matrix is incomplete");
	std::cout << "Native indexed/separated-flat RSX geometry:" << profiles << " profiles," << components << " exactRGBA32F components vs expanded double-evaluation path;all10topologies,both provoking rules,smooth/flat,highIDs,restart,degenerates,culling/winding,clip/scissor,vertextexture/constants;indexed=" << indexed << ",smallarrays=" << array << ",flatpasses="<<flat_passes<<" passed\n";
}

void flat_only_and_fresh_sources(unsigned color_source,vertex_texture_dimension dimension=vertex_texture_dimension::two)
{
	context ctx; compiler compile(ctx); constexpr unsigned base=65535;
	vertex_program_options options; options.flat_colors=true; options.packed_inputs=true;options.flat_result_inputs=true;
	options.textures[0]=dimension;
	const std::array code{instruction(1,0,0),instruction(color_source==1 ? 25 : 1,color_source==0 ? 1 : color_source==1 ? 2 : 0,1,color_source==2),instruction(25,2,7),instruction(1,0,8,true),instruction(1,3,5,false,true)};
	vertex_shader vertex(compile,code,options);
	std::ostringstream source;source<<"#include <metal_stdlib>\nusing namespace metal;\n";emit_varyings(source,true,true);
	source<<"fragment float4 indexed_color(rsx_fragment_inputs in [[stage_in]]) { return in.color0*.25f + in.tex0*.5f + in.tex1*.25f; }\n";
	const auto library=compile.library(source.str());render_pipeline_configuration configuration;configuration.colors[0]=MTLPixelFormatRGBA32Float;
	const auto pipeline=make_render_pipeline(compile,configuration,vertex.function(),library_function(library,@"indexed_color"));
	const converted_geometry original{list_topology::triangles,{{base,base+7},{base+1,base+7},{base+2,base+7},{base+2,base+7},{base+3,base+7},{base,base+7}},std::nullopt};
	const auto plan=plan_indexed_references(original);const auto native=prepared_geometry::prepare(ctx,plan);
	require(native.references().vertices.size()==4 && native.count()==6 && native.index_width()==2 && plan.bytes().size()==44,"Flat-only quad did not reuse four paired references");
	const auto image=target(ctx);const auto output=make_buffer(ctx,4096,buffer_storage::shared);
	for(unsigned round=0;round<2;++round)
	{
		auto data=inputs(ctx,base);auto raw=attributes();
		for(unsigned i=0;i<8;++i) { const std::array<float,4> coordinate{.25,.25,0,0};std::memcpy(raw.data()+256+i*16,coordinate.data(),16); }
		if(round)
		{
			const std::array<float,4> color{.125,.5,.75,1};std::memcpy(raw.data()+128+7*16,color.data(),16);
			const std::array constants{std::array<float,4>{.5,.75,1,1.25}};data.constants=make_upload(ctx,std::as_bytes(std::span(constants)));
			std::array<std::byte,512> bytes{};const std::array<float,4> green{0,1,0,1};
			for(unsigned y=0;y<2;++y) for(unsigned x=0;x<2;++x)std::memcpy(bytes.data()+y*256+x*16,green.data(),16);
			const auto upload=make_upload(ctx,bytes);ctx.begin();buffer_to_texture(ctx,upload,0,data.textures[0],{0,0,MTLOriginMake(0,0,0),MTLSizeMake(2,2,1)},256);ctx.submit().wait();
		}
		if(dimension!=vertex_texture_dimension::two)
		{
			auto descriptor=[MTLTextureDescriptor new];descriptor.pixelFormat=MTLPixelFormatRGBA32Float;descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageShaderRead;
			descriptor.width=2;descriptor.height=dimension==vertex_texture_dimension::one ? 1 : 2;descriptor.depth=dimension==vertex_texture_dimension::three ? 2 : 1;
			descriptor.textureType=dimension==vertex_texture_dimension::cube ? MTLTextureTypeCube : dimension==vertex_texture_dimension::three ? MTLTextureType3D : MTLTextureType2D;
			data.textures[0]=make_texture(ctx,descriptor);
			std::array<std::byte,1024> bytes{};const std::array<float,4> color=round ? std::array<float,4>{0,1,0,1} : std::array<float,4>{1,0,0,1};
			for(unsigned z=0;z<descriptor.depth;++z)for(unsigned y=0;y<descriptor.height;++y)for(unsigned x=0;x<2;++x)
				std::memcpy(bytes.data()+z*512+y*256+x*16,color.data(),16);
			const auto upload=make_upload(ctx,bytes);ctx.begin();
			for(unsigned face=0;face<(dimension==vertex_texture_dimension::cube ? 6u : 1u);++face)
				buffer_to_texture(ctx,upload,0,data.textures[0],{0,face,MTLOriginMake(0,0,0),MTLSizeMake(2,descriptor.height,descriptor.depth)},256,512);
			ctx.submit().wait();
		}
		const auto input = make_upload(ctx, raw);
		data.packed->persistent = make_buffer(ctx, raw.size(), buffer_storage::private_gpu);
		vertex_context state;state.vertex_base=base;state.vertex_count=8;state.constant_count=1;state.output_mask=1|(1u<<14)|(1u<<15);
		const auto before=ctx.geometry_uploads().statistics();const auto reused=prepared_geometry::prepare(ctx,plan);
		require(reused.buffer()==native.buffer() && ctx.geometry_uploads().statistics().created==before.created,"Fresh source inputs rebuilt immutable geometry");
		auto flat=prepared_flat_colors::admit(ctx,reused.references(),84);
		require(flat.source_ids().size()==1 && flat.source_ids()[0]==base+7 && flat.results().length==80,"Flat-only quad did not admit one compact color result");
		ctx.begin();
		copy_buffer(ctx, input, 0, data.packed->persistent, 0, raw.size());
		const auto before_flat = ctx.encoding_statistics();
		const auto vb = vertex.prepare(ctx, state, reused.references(), data, &flat);
		const auto after_flat = ctx.encoding_statistics();
		require(after_flat.checked_dispatches == before_flat.checked_dispatches + 1 &&
			after_flat.blit_to_dispatch_barriers == before_flat.blit_to_dispatch_barriers + 1 &&
			after_flat.compute_encoders == before_flat.compute_encoders,
			"Flat result prepass must wait for its packed stream blit in the same encoder");
		{render_commands draw(ctx,pass(image));draw.draw_geometry(reused,pipeline.pipeline,pipeline.depth_stencil,raster(),vb,{});draw.end();}
		texture_to_buffer(ctx,image,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1)},output,0,256);ctx.submit().wait();
		const auto expected=color_source==0 ? (round ? std::array<float,4>{.15625,.8125,.4375,1.0625} : std::array<float,4>{.75,.0625,.15625,.875}) :
			color_source==1 ? (round ? std::array<float,4>{.125,.9375,.25,1.0625} : std::array<float,4>{.78125,.0625,.09375,.875}) :
			(round ? std::array<float,4>{.25,.875,.5,1.125} : std::array<float,4>{.5625,.125,.1875,.75});
		for(unsigned pixel=0;pixel<256;++pixel)for(unsigned c=0;c<4;++c)
			require(static_cast<const float*>(output.contents)[pixel*4+c]==expected[c],"Native paired geometry changed diagonal/flat-only color or reused old attributes/texture/constants");
	}
	std::cout<<"Native indexed/separated flat-only quad:color-source="<<color_source<<"(attribute/texture/constant),dimension="<<unsigned(dimension)<<",fourpairedrefs/sixindices/44bytes,source7outsidecorners0..3,one80byteflatresult;512RGBA32Fpixels independently exact;two checked Blit-to-Dispatch dependencies;unchanged geometry buffer with fresh attribute/vertextexture/constant versions,including above-one color passed\n";
}

std::string point_shader = R"metal(
#include <metal_stdlib>
using namespace metal;
struct reference { uint source,flat; };
struct output { float4 position [[position]]; float point_size [[point_size]]; };
vertex output indexed_point(uint id [[vertex_id]],device const reference* references [[buffer(30)]],constant uint& selected [[buffer(0)]]) {
 return {float4(references[id].source == selected ? 0.f : 4.f,0,.5f,1),1};
}
vertex output conditional_point(uint id [[vertex_id]], uint instance [[instance_id]],
    device const reference* references [[buffer(30)]], constant uint& selected [[buffer(0)]])
{
    const bool visible = references[id].source == selected && (instance == 7 || instance == 8);
    return {float4(visible ? (instance == 7 ? -.25f : .25f) : 4.f, 0, .5f, 1), 1};
}
fragment float4 point_color(output in [[stage_in]]) { return float4(1,2,3,4); }
)metal";

void conditional_native_geometry()
{
	context ctx;
	compiler compile(ctx);
	const auto library = compile.library(point_shader);
	render_pipeline_configuration configuration;
	configuration.colors[0] = MTLPixelFormatRGBA32Float;
	configuration.topology = MTLPrimitiveTopologyClassPoint;
	const auto pipeline = make_render_pipeline(compile, configuration, library_function(library, @"conditional_point"),
		library_function(library, @"point_color"));
	const auto image = target(ctx);
	constexpr std::size_t image_bytes = 4096;
	const auto output = make_buffer(ctx, 3 * image_bytes, buffer_storage::shared);
	const auto read = [&](std::size_t slot)
	{
		texture_to_buffer(ctx, image, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(16, 16, 1)}, output, slot * image_bytes, 256);
	};
	query_pool sources(ctx, 2, 2);
	const std::uint32_t visible[]{0}, hidden[]{1};
	for (const unsigned width : {0u, 2u, 4u})
	{
		const unsigned count = width == 4 ? 65537u : 3u;
		converted_geometry original{list_topology::points, {}, std::nullopt};
		for (unsigned id = 0; id < count; ++id)
			original.vertices.push_back({id, id});
		if (width) original.vertices.push_back({0, 0});
		auto geometry = std::make_optional(prepared_geometry::prepare(ctx, plan_indexed_references(original)));
		require(geometry->index_width() == width, "Conditional draw fixture did not select the requested Metal index width");
		const auto selected = count - 1;
		const auto selection = make_upload(ctx, std::as_bytes(std::span{&selected, 1}));
		bindings vertex;
		vertex.set_buffer(0, selection);
		ctx.begin();
		sources.begin(0, true);
		auto descriptor = pass(image);
		const auto query = sources.attach(ctx, descriptor);
		require(query.has_value(), "Direct geometry did not attach its query");
		{
			render_commands commands(ctx, descriptor);
			commands.visibility(query->mode, query->offset);
			commands.draw_geometry(*geometry, pipeline.pipeline, pipeline.depth_stencil, raster(), vertex, {}, 2, 7);
			commands.end();
		}
		sources.end(0);
		sources.begin(1, true);
		sources.end(1);
		read(0);
		std::optional<conditional_draw> on, off;
		if (width == 4)
		{
			ctx.submit().wait();
			ctx.begin();
			const report_query_source positive[]{{0}}, negative[]{{1}};
			const auto on_value = sources.capture_report_predicate({positive, 0, true});
			const auto off_value = sources.capture_report_predicate({negative, 0, true});
			require(on_value && off_value, "UInt32 draw report capture was unavailable");
			(void)ctx.submit();
			ctx.begin();
			on = sources.encode_predicated_draw(*on_value, *geometry, 2, 7);
			off = sources.encode_predicated_draw(*off_value, *geometry, 2, 7);
		}
		else
		{
			on = sources.encode_conditional_draw(visible, *geometry, 2, 7);
			off = sources.encode_conditional_draw(hidden, *geometry, 2, 7);
		}
		require(on && off, "Prepared geometry failed to produce native indirect arguments");
		geometry.reset();
		ctx.geometry_uploads().clear();
		{
			render_commands commands(ctx, pass(image));
			commands.draw_geometry(*on, pipeline.pipeline, pipeline.depth_stencil, raster(), vertex, {});
			commands.end();
		}
		read(1);
		{
			render_commands commands(ctx, pass(image));
			commands.draw_geometry(*off, pipeline.pipeline, pipeline.depth_stencil, raster(), vertex, {});
			commands.end();
		}
		read(2);
		on.reset();
		off.reset();
		ctx.submit().wait();
		const auto bytes = map_buffer(output, 0, 3 * image_bytes);
		const auto expected = bytes.first(image_bytes);
		const auto drawn = bytes.subspan(image_bytes, image_bytes);
		const auto cleared = bytes.last(image_bytes);
		require(sources.consume(0) == 2 && sources.consume(1) == 0, "Conditional geometry sources did not produce exact visibility");
		require(std::ranges::equal(expected, drawn), "Indirect geometry changed pixels, index selection or base instance");
		require(std::ranges::all_of(cleared, [](std::byte byte) { return byte == std::byte{}; }),
			"False conditional geometry changed cleared pixels");
	}
	std::cout << "Native conditional prepared geometry: array/UInt16 and retained UInt32, two instances starting at 7, 24576 exact compared bytes, packet/cache/command release passed\n";
}

void native_index_width_and_lifetime()
{
	std::shared_ptr<heap_budget> ledger;
	@autoreleasepool
	{
	auto owner = std::make_unique<context>(); auto& ctx = *owner;
	ledger = std::make_shared<heap_budget>(ctx,16u << 20); ctx.set_buffer_budget(ledger); compiler compile(ctx);
	const auto library = compile.library(point_shader); render_pipeline_configuration configuration;
	configuration.colors[0] = MTLPixelFormatRGBA32Float; configuration.topology = MTLPrimitiveTopologyClassPoint;
	const auto pipeline = make_render_pipeline(compile,configuration,library_function(library,@"indexed_point"),library_function(library,@"point_color"));
	const auto image = target(ctx);
	const auto output = make_buffer(ctx,4096,buffer_storage::shared);
	for (unsigned count : {65535u,65536u,65537u})
	{
		converted_geometry original{list_topology::points,{},std::nullopt};
		for (unsigned id = 0; id < count; ++id) original.vertices.push_back({id,id}); original.vertices.push_back({0,0});
		const auto plan = plan_indexed_references(original); const auto native = prepared_geometry::prepare(ctx,plan);
		require(native.index_width() == (count <= 65535 ? 2 : 4),"Native GPU index width boundary is wrong");
		const auto selected = count - 1; const auto selection = make_upload(ctx,std::as_bytes(std::span{&selected,1})); bindings data; data.set_buffer(0,selection);
		ctx.begin(); { render_commands draw(ctx,pass(image)); draw.draw_geometry(native,pipeline.pipeline,pipeline.depth_stencil,raster(),data,{}); draw.end(); }
		texture_to_buffer(ctx,image,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1)},output,0,256); ctx.submit().wait();
		unsigned found = 0; for (unsigned pixel = 0; pixel < 256; ++pixel)
		{
			const auto p = static_cast<const float*>(output.contents)+pixel*4;
			if (p[3]) { require(p[0]==1 && p[1]==2 && p[2]==3 && p[3]==4,"Native wide-index point value differs"); ++found; }
		}
		require(found == 1,"Native wide-index last reference was dropped/truncated");
	}
	const converted_geometry original{list_topology::points,{{9,9},{9,9},{9,9}},std::nullopt}; const auto plan = plan_indexed_references(original);
	auto native = std::make_optional(prepared_geometry::prepare(ctx,plan));
	const auto before = ctx.geometry_uploads().statistics();
	for (unsigned i = 0; i < 100; ++i) require(prepared_geometry::prepare(ctx,plan).buffer()==native->buffer(),"Identical native geometry was uploaded again");
	require(ctx.geometry_uploads().statistics().created==before.created && ctx.geometry_uploads().statistics().reused==before.reused+100,"Native geometry cache repeated allocation/upload");
	for (unsigned i = 0; i < 65; ++i) (void)prepared_geometry::prepare(ctx,plan_indexed_references({list_topology::points,{{1000000000u+i,0}},std::nullopt}));
	require(ctx.geometry_uploads().statistics().entries == 64 && ctx.geometry_uploads().statistics().evicted>0 && ctx.geometry_uploads().statistics().payload_bytes <= (8u<<20),"Native geometry cache LRU is unbounded");
	const unsigned selected = 9; id<MTLBuffer> selection = make_upload(ctx,std::as_bytes(std::span{&selected,1})); bindings data; data.set_buffer(0,selection);
	const auto gate = [ctx.device() newSharedEvent];
	ctx.begin(); ctx.wait_event(gate,1);
	{ render_commands draw(ctx,pass(image)); draw.draw_geometry(*native,pipeline.pipeline,pipeline.depth_stencil,raster(),data,{}); draw.end(); }
	texture_to_buffer(ctx,image,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,16,1)},output,0,256); const auto work = ctx.submit();
	ctx.geometry_uploads().clear(); (void)ctx.trim_upload_pages(); native.reset(); data={}; selection=nil;
	bool timeout = false; try { work.wait(std::chrono::milliseconds(1)); } catch (const std::runtime_error&) { timeout=!work.ready(); }
	require(timeout && ledger->statistics().reserved_bytes>0,"Pending geometry lost backing at cache/packet removal");
	owner.reset(); gate.signaledValue=1; work.wait();
	unsigned pixels=0; for (unsigned i=0;i<256;++i) if(static_cast<const float*>(output.contents)[i*4+3]) ++pixels;
	require(pixels==1,"Native geometry/index backing did not survive context destruction");
	}
	require(ledger->statistics().reserved_bytes==0,"Orphan indexed geometry retained native charges");
	std::cout << "Native indexed geometry:actual UInt16/UInt32 boundary65535/65536/65537,last source pixel exact;100immutable hits,64entry/8MiB LRU;event-gated indexed draw survives cache/packet/bindings/context destruction andtimeout,all native charges retire passed\n";
}
}
int main(int argc, char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			if (argc == 2 && std::string_view(argv[1]) == "--conditional-only")
			{
				conditional_native_geometry();
				return 0;
			}
			require(argc == 1, "Unknown native indexed geometry test selector");
			conditional_native_geometry();
			compare_native_draws();for(unsigned source=0;source<3;++source)flat_only_and_fresh_sources(source);
			for(auto dimension:{vertex_texture_dimension::one,vertex_texture_dimension::three,vertex_texture_dimension::cube})flat_only_and_fresh_sources(1,dimension);
			native_index_width_and_lifetime();return 0;
		}
		catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
