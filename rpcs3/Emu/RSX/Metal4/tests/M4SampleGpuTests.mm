#include "M4SampleTransfer.h"
#include "M4ClearPlan.h"
#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void store(std::span<std::byte> bytes, std::uint32_t value, unsigned count)
{
	for (unsigned byte = 0; byte < count; ++byte)
		bytes[byte] = static_cast<std::byte>(value >> ((count - 1 - byte) * 8));
}
void populate(std::span<std::byte> bytes, unsigned raw, bool depth, std::size_t pixel)
{
	const auto index = static_cast<std::uint32_t>(pixel);
	if (depth)
	{
		const std::array<std::uint32_t, 6> e4m12{0, 0x5000, 0x6000, 0x7000, 0x2345, 0x3456};
		const auto value = raw == 1 ? index % 65536 : raw == 3 ? e4m12[index % e4m12.size()]
												  : raw == 2   ? ((index * 7919 % 0x1000000) << 8) | (index % 256)
															   : (std::bit_cast<std::uint32_t>(float(index % 1025) / 1024.0f) >> 7 << 8) | (index % 256);
		store(bytes, value, (raw == 1 || raw == 3) ? 2 : 4);
	}
	else if (raw <= 3)
		store(bytes, index % 65536, 2);
	else if (raw == 11 || raw == 12 || raw == 13)
	{
		const unsigned width = raw == 11 ? 2 : 4, count = raw == 13 ? 1 : 4;
		constexpr std::array<float, 8> values{-8, -0.5f, 0, 0.125f, 0.25f, 0.5f, 1, 2};
		for (unsigned c = 0; c < count; ++c)
		{
			const auto f = values[(pixel + c) % values.size()];
			const auto value = raw == 11 ? std::uint32_t(std::bit_cast<std::uint16_t>(static_cast<_Float16>(f))) : std::bit_cast<std::uint32_t>(f);
			store(bytes.subspan(c * width), value, width);
		}
	}
	else
		for (std::size_t byte = 0; byte < bytes.size(); ++byte)
			bytes[byte] = static_cast<std::byte>(index * 13 + byte * 41);
}

void run(metal4::context& ctx)
{
	using namespace metal4;
	sample_transfer transfer(ctx);
	surface_color_transfer colors(ctx);
	surface_depth_transfer depths(ctx);
	std::size_t cases = 0,crops=0,crop_bytes=0,imports=0,import_bytes=0;
	for (const unsigned samples : {2u, 4u})
		for (const bool depth : {false, true})
			for (const auto raw : depth ? std::vector<unsigned>{1, 2, 3, 4} : std::vector<unsigned>{1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16})
			{
				const auto color = depth ? surface_color_descriptor{} : describe_surface_color(static_cast<surface_color>(raw));
				const auto ds = depth ? describe_surface_depth(static_cast<surface_depth>(raw)) : surface_depth_descriptor{};
				const auto format = depth ? ds.format : color.format;
				const auto bytes = depth ? ds.guest_bytes : color.layout.pixel_bytes;
				const std::size_t width = !depth && raw <= 3 ? 256 : 13, height = !depth && raw <= 3 ? 128 : 7;
				const auto size = sample_expanded_size(width, height, samples);
				const auto pitch = (size[0] + 3) * bytes;
				std::vector<std::byte> expected(3 + pitch * size[1] + 11, std::byte{0xa7});
				for (std::size_t y = 0; y < size[1]; ++y)
					for (std::size_t x = 0; x < size[0]; ++x)
						populate(std::span(expected).subspan(3 + y * pitch + x * bytes, bytes), raw, depth, y * size[0] + x);
				const auto source = make_upload(ctx, expected), out = make_buffer(ctx, expected.size(), buffer_storage::shared);
				MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
				md.textureType = MTLTextureType2DMultisample;
				md.sampleCount = samples;
				md.storageMode = MTLStorageModePrivate;
				md.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				const auto multisampled = make_texture(ctx, md);
				MTLTextureDescriptor* ed = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:size[0] height:size[1] mipmapped:NO];
				ed.storageMode = MTLStorageModePrivate;
				ed.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				const auto expanded = make_texture(ctx, ed), result = make_texture(ctx, ed);
				texture_region region;
				region.size = MTLSizeMake(size[0], size[1], 1);
				ctx.begin();
				fill_buffer(ctx, out, 0, out.length, 0xa7);
				if (depth)
					depths.upload(ctx, static_cast<surface_depth>(raw), source, {3, pitch}, expanded, region);
				else
					colors.upload(ctx, static_cast<surface_color>(raw), source, {3, pitch}, expanded, region);
				transfer.import(ctx, expanded, multisampled);
				transfer.expand(ctx, multisampled, result);
				if (depth)
					depths.download(ctx, static_cast<surface_depth>(raw), result, region, out, {3, pitch});
				else
					colors.download(ctx, static_cast<surface_color>(raw), result, region, out, {3, pitch});
				ctx.submit().wait();
				const auto actual = map_buffer(out, 0, out.length);
				if (!std::equal(actual.begin(), actual.end(), expected.begin()))
				{
					const auto diff = std::mismatch(actual.begin(), actual.end(), expected.begin());
					std::cerr << "Samples " << samples << " depth " << depth << " format " << raw << " byte " << diff.first - actual.begin()
							  << ": " << unsigned(*diff.first) << " expected " << unsigned(*diff.second) << '\n';
					throw std::runtime_error("Native individual sample import/expansion bytes differ");
				}
				++cases;
				for(const auto origin:std::array<std::array<std::size_t,2>,4>{{{0,0},{1,1},{2,3},{size[0]-3,size[1]-3}}})
				for(const auto shape:std::array<std::array<std::size_t,2>,3>{{{1,1},{2,1},{3,3}}})
				{
					auto cd=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:shape[0] height:shape[1] mipmapped:NO];cd.storageMode=MTLStorageModePrivate;cd.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
					const auto crop=make_texture(ctx,cd);const auto row=(shape[0]+2)*bytes;
					std::vector<std::byte> wanted(3+shape[1]*row+7,std::byte{0xa7});
					for(std::size_t y=0;y<shape[1];++y)std::copy_n(expected.begin()+3+(origin[1]+y)*pitch+origin[0]*bytes,shape[0]*bytes,wanted.begin()+3+y*row);
					const auto output=make_buffer(ctx,wanted.size(),buffer_storage::shared);texture_region small;small.size=MTLSizeMake(shape[0],shape[1],1);
					ctx.begin();transfer.import(ctx,expanded,multisampled);fill_buffer(ctx,output,0,output.length,0xa7);transfer.expand_region(ctx,multisampled,crop,origin);
					if(depth)depths.download(ctx,static_cast<surface_depth>(raw),crop,small,output,{3,row});else colors.download(ctx,static_cast<surface_color>(raw),crop,small,output,{3,row});
					ctx.submit().wait();require(std::memcmp(output.contents,wanted.data(),wanted.size())==0,"Native MSAA crop changed sample coordinates/guest words orbyte guards");
					crop_bytes+=shape[0]*shape[1]*bytes;++crops;
					std::vector<std::byte> patch(shape[0]*shape[1]*bytes);
					for(std::size_t i=0;i<shape[0]*shape[1];++i)
					{
						if(depth&&raw==2)store(std::span(patch).subspan(i*bytes,bytes),((i&1?0xffffffu:0)<<8)|((i*31+97)&255),4);
						else populate(std::span(patch).subspan(i*bytes,bytes),raw,depth,901+i+imports*17);
					}
					auto target_expected=expected;
					for(std::size_t y=0;y<shape[1];++y)std::copy_n(patch.begin()+y*shape[0]*bytes,shape[0]*bytes,target_expected.begin()+3+(origin[1]+y)*pitch+origin[0]*bytes);
					ctx.begin();
					if(depth)depths.upload(ctx,static_cast<surface_depth>(raw),make_upload(ctx,patch),{},crop,small);else colors.upload(ctx,static_cast<surface_color>(raw),make_upload(ctx,patch),{},crop,small);
					transfer.import_region(ctx,crop,multisampled,origin);transfer.expand(ctx,multisampled,result);fill_buffer(ctx,out,0,out.length,0xa7);
					if(depth)depths.download(ctx,static_cast<surface_depth>(raw),result,region,out,{3,pitch});else colors.download(ctx,static_cast<surface_color>(raw),result,region,out,{3,pitch});
					ctx.submit().wait();require(std::memcmp(out.contents,target_expected.data(),target_expected.size())==0,"Native MSAA partial import changed exterior samples,source lanes orbyte guards");
					++imports;import_bytes+=size[0]*size[1]*bytes;
				}
				std::cout << samples << "x " << (depth ? "depth " : "color ") << raw << " native sample import/expansion passed\n";
			}
	std::cout << cases << " native MSAA surface cases preserve every sample and byte guard\n";
	require(crops==432&&transfer.statistics().cropped_expansions==crops,"Native MSAA guest crop coverage/counters differ");
	std::cout<<"Native MSAA guest crops:"<<crops<<" format/sample/origin/extent profiles,"<<crop_bytes<<" exact active guest bytes;odd sample origins,1x1/2x1/3x3 extents,edge crops,prefix/pitch/tail guards passed\n";
	require(imports==432&&transfer.statistics().cropped_imports==imports,"Native MSAA partial import coverage/counters differ");
	std::cout<<"Native MSAA cropped imports:"<<imports<<" format/sample/origin/extent profiles,"<<import_bytes<<" whole-target guest bytes;different patches,odd boundary samples,exterior/prefix/pitch/tail guards exact passed\n";
	ctx.wait_idle();
}

void independent_samples(metal4::context& ctx)
{
	using namespace metal4;
	compiler compiler(ctx);
	sample_transfer transfer(ctx);
	surface_depth_transfer depth_transfer(ctx);
	std::string source = generate_sample_shader(sample_aspect::color, false);
	source += R"msl(
struct independently_written_sample {
	float4 color [[color(0)]];
	float depth [[depth(any)]];
	uint stencil [[stencil]];
};
fragment independently_written_sample independent_fragment(sample_vertex in [[stage_in]],uint sample [[sample_id]]) {
	const uint2 pixel=uint2(in.position.xy);
	return {float4(float(sample+1)/4.0f,float(pixel.x%4)/4.0f,float(pixel.y%4)/4.0f,1),
		float(sample+1)/8.0f,(17+sample*53+pixel.x+pixel.y*7)%256};
}
)msl";
	const auto library = compiler.library(source, @"Independent MSAA sample writer");
	MTLDepthStencilDescriptor* depth_descriptor = [MTLDepthStencilDescriptor new];
	depth_descriptor.depthCompareFunction = MTLCompareFunctionAlways;
	depth_descriptor.depthWriteEnabled = YES;
	MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
	stencil.stencilCompareFunction = MTLCompareFunctionAlways;
	stencil.depthStencilPassOperation = MTLStencilOperationReplace;
	depth_descriptor.frontFaceStencil = depth_descriptor.backFaceStencil = stencil;
	const auto state = [ctx.device() newDepthStencilStateWithDescriptor:depth_descriptor];
	for (const unsigned samples : {2u, 4u})
	{
		constexpr std::size_t width = 13, height = 7;
		const auto size = sample_expanded_size(width, height, samples);
		const auto make = [&](MTLPixelFormat format, bool msaa)
		{
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:msaa ? width : size[0] height:msaa ? height : size[1] mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
			if (msaa)
			{
				descriptor.textureType = MTLTextureType2DMultisample;
				descriptor.sampleCount = samples;
			}
			return make_texture(ctx, descriptor);
		};
		const auto color = make(MTLPixelFormatRGBA32Float, true), depth = make(MTLPixelFormatDepth32Float_Stencil8, true),
				   out_color = make(MTLPixelFormatRGBA32Float, false), out_depth = make(MTLPixelFormatDepth32Float_Stencil8, false);
		MTL4RenderPipelineDescriptor* pd = [MTL4RenderPipelineDescriptor new];
		pd.vertexFunctionDescriptor = library_function(library, @"rpcs3_sample_vertex");
		pd.fragmentFunctionDescriptor = library_function(library, @"independent_fragment");
		pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		pd.rasterSampleCount = samples;
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
		const auto pipeline = compiler.render(pd);
		const auto color_buffer = make_buffer(ctx, size[0] * size[1] * 16, buffer_storage::shared),
				   depth_buffer = make_buffer(ctx, size[0] * size[1] * 4, buffer_storage::shared);
		ctx.begin();
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = color;
		pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.depthAttachment.texture = depth;
		pass.depthAttachment.loadAction = MTLLoadActionDontCare;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.stencilAttachment.texture = depth;
		pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
		pass.stencilAttachment.storeAction = MTLStoreActionStore;
		render_commands commands(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, double(width), double(height), 0, 1};
		raster.scissor = {0, 0, width, height};
		commands.configure(pipeline, state, raster, {}, {});
		commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
		commands.end();
		transfer.expand(ctx, color, out_color);
		transfer.expand(ctx, depth, out_depth);
		texture_region region;
		region.size = MTLSizeMake(size[0], size[1], 1);
		texture_to_buffer(ctx, out_color, region, color_buffer, 0, size[0] * 16);
		depth_transfer.download(ctx, surface_depth::z24s8_float, out_depth, region, depth_buffer, {});
		ctx.submit().wait();
		const auto colors = map_buffer(color_buffer, 0, color_buffer.length), depths = map_buffer(depth_buffer, 0, depth_buffer.length);
		for (std::size_t y = 0; y < size[1]; ++y)
			for (std::size_t x = 0; x < size[0]; ++x)
			{
				const auto px = x / 2, py = samples == 2 ? y : y / 2;
				const unsigned sample = static_cast<unsigned>(x % 2 + (samples == 2 ? 0 : (y % 2) * 2));
				std::array<float, 4> actual;
				std::memcpy(actual.data(), colors.data() + (y * size[0] + x) * 16, 16);
				const std::array<float, 4> expected{float(sample + 1) / 4.0f, float(px % 4) / 4.0f, float(py % 4) / 4.0f, 1};
				require(actual == expected, "Independent MSAA color sample coordinate differs");
				const auto bits = std::bit_cast<std::uint32_t>(float(sample + 1) / 8.0f);
				const auto guest = (bits >> 7 << 8) | static_cast<std::uint32_t>((17 + sample * 53 + px + py * 7) % 256);
				for (unsigned byte = 0; byte < 4; ++byte)
					require(depths[(y * size[0] + x) * 4 + byte] == static_cast<std::byte>(guest >> ((3 - byte) * 8)), "Independent MSAA depth/stencil sample coordinate differs");
			}
		std::size_t crop_cases=0;
		for(const auto origin:std::array<std::array<std::size_t,2>,4>{{{0,0},{1,1},{2,3},{size[0]-3,size[1]-3}}})
		for(unsigned selected=0;selected<4;++selected)
		{
			const bool is_color=selected==0;const auto plane=selected<2?texture_plane::all:selected==2?texture_plane::depth:texture_plane::stencil;
			auto cd=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:is_color?MTLPixelFormatRGBA32Float:MTLPixelFormatDepth32Float_Stencil8 width:3 height:3 mipmapped:NO];cd.storageMode=MTLStorageModePrivate;cd.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
			const auto crop=make_texture(ctx,cd);const auto out=make_buffer(ctx,9*(is_color?16:4),buffer_storage::shared);texture_region small;small.size=MTLSizeMake(3,3,1);
			std::vector<std::byte> seed(36);for(unsigned i=0;i<9;++i)store(std::span(seed).subspan(i*4,4),(std::bit_cast<std::uint32_t>(.75f)>>7<<8)|201,4);
			ctx.begin();if(!is_color)depth_transfer.upload(ctx,surface_depth::z24s8_float,make_upload(ctx,seed),{},crop,small);
			transfer.expand_region(ctx,is_color?color:depth,crop,origin,plane);
			if(is_color)texture_to_buffer(ctx,crop,small,out,0,48);else depth_transfer.download(ctx,surface_depth::z24s8_float,crop,small,out,{});
			ctx.submit().wait();
			for(unsigned y=0;y<3;++y)for(unsigned x=0;x<3;++x)
			{
				const auto sx=origin[0]+x,sy=origin[1]+y,px=sx/2,py=samples==2?sy:sy/2;const unsigned sample=sx%2+(samples==2?0:(sy%2)*2);
				if(is_color){std::array<float,4> got;std::memcpy(got.data(),static_cast<const std::byte*>(out.contents)+(y*3+x)*16,16);require(got==std::array<float,4>{float(sample+1)/4,float(px%4)/4,float(py%4)/4,1},"Independent cropped MSAA color source coordinate differs");}
				else{const auto bits=std::bit_cast<std::uint32_t>(plane==texture_plane::stencil?.75f:float(sample+1)/8);const auto s=plane==texture_plane::depth?201u:(17+sample*53+px+py*7)%256;const auto word=(bits>>7<<8)|s;std::array<std::byte,4> wanted;store(wanted,word,4);require(std::memcmp(static_cast<const std::byte*>(out.contents)+(y*3+x)*4,wanted.data(),4)==0,"Independent cropped MSAA depth/stencil source coordinate oruntouched plane differs");}
			}
			++crop_cases;
		}
		require(crop_cases==16,"Independent native crop cases incomplete");
		std::cout<<samples<<"x independently-written native MSAA crops:"<<crop_cases<<" exact color/depth/stencil origins andselected-plane preservation passed\n";
		std::size_t import_cases=0;
		for(const auto origin:std::array<std::array<std::size_t,2>,4>{{{0,0},{1,1},{2,3},{size[0]-3,size[1]-3}}})
		for(unsigned selected=0;selected<4;++selected)
		{
			const bool is_color=selected==0;const auto plane=selected<2?texture_plane::all:selected==2?texture_plane::depth:texture_plane::stencil;
			auto cd=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:is_color?MTLPixelFormatRGBA32Float:MTLPixelFormatDepth32Float_Stencil8 width:3 height:3 mipmapped:NO];cd.storageMode=MTLStorageModePrivate;cd.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
			const auto crop=make_texture(ctx,cd);texture_region small;small.size=MTLSizeMake(3,3,1);
			std::vector<std::byte> seed(36);for(unsigned i=0;i<9;++i)store(std::span(seed).subspan(i*4,4),(std::bit_cast<std::uint32_t>(.75f)>>7<<8)|201,4);
			std::vector<float> color_seed;for(unsigned i=0;i<9;++i)for(float f:{.125f,.75f,.5f,1.f})color_seed.push_back(f);
			ctx.begin();render_commands reset(ctx,pass);reset.configure(pipeline,state,raster,{},{});reset.draw(MTLPrimitiveTypeTriangle,0,3);reset.end();
			if(is_color)buffer_to_texture(ctx,make_upload(ctx,std::as_bytes(std::span(color_seed))),0,crop,small,48);else depth_transfer.upload(ctx,surface_depth::z24s8_float,make_upload(ctx,seed),{},crop,small);
			transfer.import_region(ctx,crop,is_color?color:depth,origin,plane);transfer.expand(ctx,color,out_color);transfer.expand(ctx,depth,out_depth);
			texture_to_buffer(ctx,out_color,region,color_buffer,0,size[0]*16);depth_transfer.download(ctx,surface_depth::z24s8_float,out_depth,region,depth_buffer,{});ctx.submit().wait();
			for(std::size_t y=0;y<size[1];++y)for(std::size_t x=0;x<size[0];++x)
			{
				const auto px=x/2,py=samples==2?y:y/2;const unsigned sample=x%2+(samples==2?0:(y%2)*2);const bool inside=x>=origin[0]&&x-origin[0]<3&&y>=origin[1]&&y-origin[1]<3;
				const std::array<float,4> wanted=is_color&&inside?std::array<float,4>{.125f,.75f,.5f,1}:std::array<float,4>{float(sample+1)/4,float(px%4)/4,float(py%4)/4,1};
				std::array<float,4> got;std::memcpy(got.data(),static_cast<const std::byte*>(color_buffer.contents)+(y*size[0]+x)*16,16);require(got==wanted,"Independent MSAA partial import changed color background orselected sample");
				const auto bits=std::bit_cast<std::uint32_t>(!is_color&&inside&&plane!=texture_plane::stencil?.75f:float(sample+1)/8);
				const auto s=!is_color&&inside&&plane!=texture_plane::depth?201u:(17+sample*53+px+py*7)%256;const auto word=(bits>>7<<8)|s;std::array<std::byte,4> expected;store(expected,word,4);
				require(std::memcmp(static_cast<const std::byte*>(depth_buffer.contents)+(y*size[0]+x)*4,expected.data(),4)==0,"Independent MSAA partial import changed exterior depth/stencil oruntouched plane");
			}
			++import_cases;
		}
		require(import_cases==16,"Independent native partial import cases incomplete");
		std::cout<<samples<<"x independently-written native MSAA cropped imports:"<<import_cases<<" whole-target color-depth-stencil/background checks passed\n";
		ctx.begin();
		const auto before_crop_stats=transfer.statistics();
		for(const auto origin:std::array<std::array<std::size_t,2>,5>{{{size[0],0},{0,size[1]},{size[0]-2,size[1]-2},{std::numeric_limits<std::size_t>::max(),0},{0,std::numeric_limits<std::size_t>::max()}}})
		{
			auto cd=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:3 height:3 mipmapped:NO];cd.storageMode=MTLStorageModePrivate;cd.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
			bool failed=false;try{transfer.expand_region(ctx,color,make_texture(ctx,cd),origin);}catch(const std::invalid_argument&){failed=true;}require(failed&&transfer.statistics().cropped_expansions==before_crop_stats.cropped_expansions,"Invalid/overflow MSAA crop encoded orchanged counters");
			failed=false;try{transfer.import_region(ctx,make_texture(ctx,cd),color,origin);}catch(const std::invalid_argument&){failed=true;}require(failed&&transfer.statistics().cropped_imports==before_crop_stats.cropped_imports,"Invalid/overflow MSAA partial import encoded orchanged counters");
		}
		bool rejected = false;
		try
		{
			transfer.expand(ctx, color, out_depth);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Incompatible native MSAA formats were accepted");
		rejected = false;
		try
		{
			transfer.expand(ctx, out_color, out_color);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Invalid native MSAA sample/type/alias contract was accepted");
		const auto reversed = make_texture_view(out_color, MTLPixelFormatRGBA32Float, MTLTextureType2D,
			NSMakeRange(0,1), NSMakeRange(0,1), MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue,MTLTextureSwizzleGreen,MTLTextureSwizzleRed,MTLTextureSwizzleAlpha));
		rejected = false;
		try { transfer.import(ctx,reversed,color); } catch (const std::invalid_argument&) { rejected = true; }
		require(rejected,"Channel-remapped view was accepted as raw native sample input");
		ctx.cancel();
	}
	std::cout << "Independent per-sample writer confirms color/depth/stencil sample placement for 2x and 4x\n";
}

void pending_crop_lifetime()
{
    using namespace metal4;using namespace std::chrono_literals;
    for(unsigned count:{2u,4u})
    {
        context ctx(2);const auto gate=[ctx.device() newSharedEvent];
        struct release {id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} release_gate{gate};
        const auto factors=rsx_sample_layout(count);const auto width=8u,height=3*factors.y;
        std::vector<std::byte> input(width*height*4);for(unsigned i=0;i<input.size();++i)input[i]=std::byte((i*43+17)&255);
        std::vector<std::byte> expected(24);for(unsigned y=0;y<2;++y)std::copy_n(input.begin()+((y+1)*width+1)*4,12,expected.begin()+y*12);
        const auto output=make_buffer(ctx,24,buffer_storage::shared);
        const auto work=[&]
        {
            @autoreleasepool
            {
                sample_transfer transfer(ctx);surface_color_transfer colors(ctx);
                auto single=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:width height:height mipmapped:NO];single.storageMode=MTLStorageModePrivate;single.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
                const auto expanded=make_texture(ctx,single);
                auto multi=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:3 mipmapped:NO];multi.storageMode=MTLStorageModePrivate;multi.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;multi.textureType=MTLTextureType2DMultisample;multi.sampleCount=count;
                const auto source=make_texture(ctx,multi);
                auto small=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:3 height:2 mipmapped:NO];small.storageMode=MTLStorageModePrivate;small.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
                const auto crop=make_texture(ctx,small);texture_region all;all.size=MTLSizeMake(width,height,1);texture_region region;region.size=MTLSizeMake(3,2,1);
                ctx.begin();ctx.wait_event(gate,1);colors.upload(ctx,surface_color::argb8,make_upload(ctx,input),{},expanded,all);transfer.import(ctx,expanded,source);transfer.expand_region(ctx,source,crop,{1,1});transfer.import_region(ctx,crop,source,{3,0});transfer.expand_region(ctx,source,crop,{3,0});colors.download(ctx,surface_color::argb8,crop,region,output,{});return ctx.submit();
            }
        }();
        bool timed_out=false;try{work.wait(1ms);}catch(const std::runtime_error&){timed_out=true;}
        require(timed_out&&!work.ready(),"Native cropped MSAA owner lifetime lacks real pending GPU work");
        gate.signaledValue=1;work.wait();require(std::memcmp(output.contents,expected.data(),expected.size())==0,"Native cropped MSAA work lost source/crop/helper/parameter owners aftertimeout");
    }
    std::cout<<"Native cropped MSAA lifetime:2 real gated input-output crop timeouts retain source/crop/parameter/shader owners after helper destruction andautorelease drain;exact resumed bytes passed\n";
}
void float_depth_bands(metal4::context& ctx)
{
	using namespace metal4;
	constexpr std::size_t width = 512, height = 130;
	const auto make = [&](std::size_t w, std::size_t h, unsigned samples)
	{
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:w height:h mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
		if (samples > 1)
		{
			descriptor.textureType = MTLTextureType2DMultisample;
			descriptor.sampleCount = samples;
		}
		return make_texture(ctx, descriptor);
	};
	const auto multisampled = make(width, height, 4);
	const auto expanded = make(width * 2 - 1, height * 2 - 1, 1);
	helper_catalog compiler(ctx);
	const auto library = compiler.library(generate_sample_shader(sample_aspect::color, false) + R"msl(
struct depth_word { float value [[depth(any)]]; };
fragment depth_word write_depth(sample_vertex in [[stage_in]], uint sample [[sample_id]]) {
	const uint2 pixel = uint2(in.position.xy);
	if (pixel.x == 1) {
		const uint small_depths[] = {0u, 1u, 0x7fffffu, 0x00800001u};
		return {as_type<float>(small_depths[sample])};
	}
	return {as_type<float>(0x3d000000u | ((pixel.x * 331u + pixel.y * 1021u + sample * 257u) & 0x7fffffu))};
}
)msl", @"Independent float-depth sample words");
	helper_render_configuration config;
	config.samples = 4;
	const auto pipeline = compiler.render(library, @"rpcs3_sample_vertex", @"write_depth", config);
	auto descriptor = [MTLDepthStencilDescriptor new];
	descriptor.depthCompareFunction = MTLCompareFunctionAlways;
	descriptor.depthWriteEnabled = YES;
	const auto state = [ctx.device() newDepthStencilStateWithDescriptor:descriptor];
	const auto pitch = width * 2 * sizeof(std::uint32_t);
	const auto output = make_buffer(ctx, pitch * expanded.height, buffer_storage::shared);
	sample_transfer transfer(ctx);
	ctx.begin();
	auto pass = [MTL4RenderPassDescriptor new];
	pass.depthAttachment.texture = multisampled;
	pass.depthAttachment.loadAction = MTLLoadActionDontCare;
	pass.depthAttachment.storeAction = MTLStoreActionStore;
	raster_state raster;
	raster.viewport = {0, 0, double(width), double(height), 0, 1};
	raster.scissor = {0, 0, width, height};
	render_commands commands(ctx, pass);
	commands.configure(pipeline, state, raster, {}, {});
	commands.draw(MTLPrimitiveTypeTriangle, 0, 3);
	commands.end();
	transfer.expand_region(ctx, multisampled, expanded, {1, 1}, texture_plane::depth);
	texture_region region;
	region.size = MTLSizeMake(expanded.width, expanded.height, 1);
	texture_to_buffer(ctx, expanded, region, output, 0, pitch, 0, texture_plane::depth);
	ctx.submit().wait();
	const auto* words = static_cast<const std::uint32_t*>(output.contents);
	constexpr std::array small_depths{0u, 1u, 0x7fffffu, 0x00800001u};
	for (std::size_t y = 0; y < expanded.height; ++y)
		for (std::size_t x = 0; x < expanded.width; ++x)
		{
			const auto sample = (x + 1) % 2 + 2 * ((y + 1) % 2);
			const auto expected = (x + 1) / 2 == 1 ? small_depths[sample] :
				0x3d000000u | ((((x + 1) / 2) * 331 + ((y + 1) / 2) * 1021 + sample * 257) & 0x7fffffu);
			require(words[y * width * 2 + x] == expected, "Float-depth expansion changed a sample word or reordered a band");
		}
	std::cout << "Float-depth expansion: " << expanded.width * expanded.height << " exact independent sample words across staging bands, including subnormals\n";
}

void raw_float_color_imports(metal4::context& ctx)
{
	using namespace metal4;
	helper_catalog compiler(ctx);
	const auto library = compiler.library(R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void read_native_float(texture2d_ms<float> source [[texture(0)]], device uint4* words [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
 const uint sy = source.get_num_samples() / 2;
 words[p.y * (2 * source.get_width()) + p.x] = as_type<uint4>(source.read(p / uint2(2,sy), p.x % 2 + 2 * (p.y % sy)));
}
kernel void read_native_half(texture2d_ms<half> source [[texture(0)]], device ushort4* words [[buffer(0)]], uint2 p [[thread_position_in_grid]]) {
 const uint sy = source.get_num_samples() / 2;
 words[p.y * (2 * source.get_width()) + p.x] = as_type<ushort4>(source.read(p / uint2(2,sy), p.x % 2 + 2 * (p.y % sy)));
}
)msl", @"Independent raw native MSAA color reader");
	const auto half_reader = compiler.compute(library, @"read_native_half");
	const auto float_reader = compiler.compute(library, @"read_native_float");
	sample_transfer transfer(ctx);
	unsigned profiles = 0, crops = 0;
	std::size_t words_checked = 0;
	for (const auto format : {MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA32Float, MTLPixelFormatR32Float})
		for (const unsigned samples : {2u, 4u})
		{
			const bool half = format == MTLPixelFormatRGBA16Float;
			const unsigned element = half ? 2 : 4, channels = format == MTLPixelFormatR32Float ? 1 : 4;
			constexpr unsigned width = 64, height = 256;
			const auto pitch = width * channels * element;
			const auto make = [&](unsigned w, unsigned h, unsigned count)
			{
				auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
				if (count > 1) { descriptor.textureType = MTLTextureType2DMultisample; descriptor.sampleCount = count; }
				return make_texture(ctx, descriptor);
			};
			const auto raster = make(width, height, 1), native = make(width / 2, height / (samples / 2), samples);
			std::vector<std::byte> expected(pitch * height);
			const auto bits = [&](unsigned i)
			{
				if (half) return i & 65535u;
				constexpr std::array values{0u, 0x80000000u, 1u, 0x807fffffu, 0x00800000u, 0x3f800000u, 0xc0000000u,
					0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc04321u, 0x7f812345u, 0xff804321u, 0x7fffffffu, 0xffffffffu, 0x7f7fffffu};
				return i % 3 == 0 ? i * 1664525u + 1013904223u : values[i % values.size()];
			};
			for (unsigned i = 0; i < width * height * channels; ++i)
			{
				const auto value = bits(i);
				std::memcpy(expected.data() + i * element, &value, element);
			}
			const auto output = make_buffer(ctx, width * height * 4 * element, buffer_storage::shared);
			const auto check = [&]
			{
				ctx.begin();
				bindings resources; resources.set_texture(0, native); resources.set_buffer(0, output);
				const auto pipeline = half ? half_reader : float_reader;
				const auto table = resources.snapshot(ctx);
				const auto encoder = ctx.compute();
				[encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
				[encoder dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth, 1, 1)];
				ctx.submit().wait();
				for (unsigned i = 0; i < width * height; ++i)
					for (unsigned c = 0; c < channels; ++c)
					{
						const auto* actual = static_cast<const std::byte*>(output.contents) + (i * 4 + c) * element;
						const auto* wanted = expected.data() + (i * channels + c) * element;
						if (std::memcmp(actual, wanted, element))
						{
							unsigned a = 0, w = 0; std::memcpy(&a, actual, element); std::memcpy(&w, wanted, element);
							throw std::runtime_error("Raw float MSAA import changed a word: format " + std::to_string(format) +
								", samples " + std::to_string(samples) + ", element " + std::to_string(i * channels + c) +
								", wanted " + std::to_string(w) + ", actual " + std::to_string(a));
						}
						++words_checked;
					}
			};
			texture_region all; all.size = MTLSizeMake(width, height, 1);
			const auto gate = [ctx.device() newSharedEvent];
			struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } release_gate{gate};
			const auto work = [&]
			{
				@autoreleasepool
				{
					sample_transfer importer(ctx);
					ctx.begin(); ctx.wait_event(gate, 1);
					buffer_to_texture(ctx, make_upload(ctx, expected), 0, raster, all, pitch);
					importer.import(ctx, raster, native); return ctx.submit();
				}
			}();
			bool timed_out = false;
			try { work.wait(std::chrono::milliseconds(1)); } catch (const submission_wait_timeout&) { timed_out = true; }
			require(timed_out && !work.ready(), "Raw float-color import lost its pending dependency or retained no real work");
			gate.signaledValue = 1; work.wait(); check();
			for (const auto crop : std::array{clear_rectangle{1, 1, 3, 3}, clear_rectangle{width - 3, height - 1, 3, 1}, clear_rectangle{3, 4, 1, 1}})
			{
				std::vector<std::byte> patch(crop.width * crop.height * channels * element);
				for (unsigned i = 0; i < crop.width * crop.height * channels; ++i)
				{
					const auto value = bits(i + 0x7c01 + crops * 97);
					std::memcpy(patch.data() + i * element, &value, element);
				}
				const auto source = make(crop.width, crop.height, 1);
				texture_region small; small.size = MTLSizeMake(crop.width, crop.height, 1);
				ctx.begin(); buffer_to_texture(ctx, make_upload(ctx, patch), 0, source, small, crop.width * channels * element);
				transfer.import_region(ctx, source, native, {crop.x, crop.y}); ctx.submit().wait();
				for (unsigned y = 0; y < crop.height; ++y)
					std::memcpy(expected.data() + (crop.y + y) * pitch + crop.x * channels * element,
						patch.data() + y * crop.width * channels * element, crop.width * channels * element);
				check(); ++crops;
			}
			++profiles;
		}
	std::cout << "Raw float-color MSAA import: " << profiles << " profiles, " << crops << " odd/exterior-preserving crops, " << words_checked
		<< " exact words against independent native multisample reads; all 65536 half patterns, float signed zero/subnormals/Inf/quiet and signaling NaNs, six event timeouts after importer destruction passed\n";
}

void invalid_depth_values()
{
	using namespace metal4;
	for (const float invalid : {-0.25f, 2.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
	{
		const auto operation = [&]
		{
			// The feedback callback must own its checks and buffers even after
			// this context and sample-transfer object have been destroyed.
			context ctx(1);
			sample_transfer transfer(ctx);
			const std::array<float, 2> values{invalid, 0.5f};
			const auto input = make_upload(ctx, std::as_bytes(std::span(values)));
			MTLTextureDescriptor* single = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:2 height:1 mipmapped:NO];
			single.storageMode = MTLStorageModePrivate;
			single.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto source = make_texture(ctx, single);
			MTLTextureDescriptor* multi = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:1 height:1 mipmapped:NO];
			multi.textureType = MTLTextureType2DMultisample;
			multi.sampleCount = 2;
			multi.storageMode = MTLStorageModePrivate;
			multi.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto target = make_texture(ctx, multi);
			ctx.begin();
			texture_region region;
			region.size = MTLSizeMake(2, 1, 1);
			buffer_to_texture(ctx, input, 0, source, region);
			transfer.import(ctx, source, target);
			return ctx.submit();
		}();
		bool rejected = false;
		try
		{
			operation.wait();
		}
		catch (const std::runtime_error& error)
		{
			rejected = std::string(error.what()).find("outside finite [0,1]") != std::string::npos;
		}
		require(rejected && operation.ready(), "Invalid MSAA depth silently clamped or completion checks lost after destruction");
	}
	std::cout << "Negative/>1/Inf/NaN depth reports GPU submission validation failure after context destruction\n";
}
} // namespace
int main(int argc, char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			metal4::context ctx(1);
			if (argc == 2 && std::string_view(argv[1]) == "raw-color-import")
			{ raw_float_color_imports(ctx); return 0; }
			run(ctx);
			independent_samples(ctx);
			float_depth_bands(ctx);
			raw_float_color_imports(ctx);
			ctx.wait_idle();
			pending_crop_lifetime();
			invalid_depth_values();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
