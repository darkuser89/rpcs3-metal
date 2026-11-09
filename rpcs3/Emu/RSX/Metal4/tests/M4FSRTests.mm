#include "M4FSR.h"
#include "M4FSRReference.h"
#include "M4Frame.h"
#include "M4HostAccessFixture.h"
#import <AppKit/NSApplication.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <limits>
#include <new>

namespace
{
using namespace metal4;
using pixel=test_fsr::pixel;
unsigned guards=0,profiles=0,frames=0; std::size_t checked=0; float maximum_error=0;
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
template<typename F> void rejects(F&& fn)
{
	bool failed=false; try { fn(); } catch(const std::exception&) { failed=true; }
	require(failed,"Invalid FSR operation accepted"); ++guards;
}
id<MTLTexture> texture(context& ctx,MTLPixelFormat format,unsigned w,unsigned h,bool mip=false)
{
	auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:mip];
	d.storageMode=MTLStorageModePrivate; d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
	return make_texture(ctx,d);
}
float linear(float x) { return x<=.04045f?x/12.92f:std::pow((x+.055f)/1.055f,2.4f); }
unsigned bpp(MTLPixelFormat f) { return f==MTLPixelFormatRGBA32Float?16:f==MTLPixelFormatRGBA16Float?8:4; }
void upload(context& ctx,id<MTLTexture> source,unsigned level,std::vector<pixel>& pixels)
{
	const unsigned size=bpp(source.pixelFormat); std::vector<std::byte> bytes(pixels.size()*size);
	for(unsigned p=0;p<pixels.size();++p) for(unsigned c=0;c<4;++c)
	{
		float& v=pixels[p][c];
		if(size==16) std::memcpy(bytes.data()+p*size+c*4,&v,4);
		else if(size==8) { const _Float16 h=v;v=float(h);std::memcpy(bytes.data()+p*size+c*2,&h,2); }
		else
		{
			const auto byte=std::uint8_t(std::lround(std::clamp(v,0.f,1.f)*255));
			const bool bgra=source.pixelFormat==MTLPixelFormatBGRA8Unorm||source.pixelFormat==MTLPixelFormatBGRA8Unorm_sRGB;
			bytes[p*4+(bgra&&c<3?2-c:c)]=std::byte(byte);v=byte/255.f;
			if(c<3&&(source.pixelFormat==MTLPixelFormatRGBA8Unorm_sRGB||source.pixelFormat==MTLPixelFormatBGRA8Unorm_sRGB)) v=linear(v);
		}
	}
	const auto data=make_upload(ctx,bytes); ctx.begin();
	buffer_to_texture(ctx,data,0,source,{level,0,MTLOriginMake(0,0,0),MTLSizeMake(std::max(NSUInteger(1),source.width>>level),std::max(NSUInteger(1),source.height>>level),1)});
	ctx.submit().wait();
}
std::vector<pixel> download(context& ctx,id<MTLTexture> image)
{
	require(image.pixelFormat==MTLPixelFormatRGBA16Float,"FSR download needs half output");
	const auto buffer=make_buffer(ctx,image.width*image.height*8,buffer_storage::shared);
	ctx.begin();texture_to_buffer(ctx,image,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(image.width,image.height,1)},buffer,0);ctx.submit().wait();
	const auto bytes=map_buffer(buffer,0,buffer.length);std::vector<pixel> result(image.width*image.height);
	for(unsigned p=0;p<result.size();++p) for(unsigned c=0;c<4;++c) { _Float16 h;std::memcpy(&h,bytes.data()+p*8+c*2,2);result[p][c]=float(h); }
	return result;
}
void compare(const std::vector<pixel>& got,const std::vector<pixel>& expected,float tolerance)
{
	require(got.size()==expected.size(),"FSR output count mismatch");
	for(unsigned p=0;p<got.size();++p) for(unsigned c=0;c<4;++c)
	{
		const float error=std::abs(got[p][c]-expected[p][c]);maximum_error=std::max(error,maximum_error);++checked;
		if(!std::isfinite(got[p][c])||!std::isfinite(expected[p][c])||error>tolerance)
		{
			std::cerr<<"profile="<<profiles<<" frame="<<frames<<" pixel="<<p<<" channel="<<c<<" actual="<<got[p][c]<<" expected="<<expected[p][c]<<" error="<<error<<'\n';
			throw std::runtime_error("Native FSR differs from AMD FP32/half reference");
		}
	}
}
std::vector<pixel> pattern(unsigned w,unsigned h,unsigned variant)
{
	std::vector<pixel> result(w*h);
	for(unsigned y=0;y<h;++y) for(unsigned x=0;x<w;++x)
	{
		const float value=variant==0?0:variant==1?1:variant==2?float((x+y)&1):variant==3?float(x>=w/2):variant==4?float(x>=y):variant==5?float((x*17+y*41+13)%101)/100.f:float(x+y)/std::max(1u,w+h-2);
		result[y*w+x]={value,variant<2?value:1-value,variant<2?value:.125f+.75f*value,.25f};
	}
	return result;
}
void constants()
{
	unsigned count=0;
	for(unsigned w:{1u,3u,17u,1920u}) for(unsigned h:{1u,5u,11u,1080u}) for(unsigned x:{0u,w/2}) for(unsigned y:{0u,h/2})
	for(float sharpness:{0.f,.5f,1.f,1.5f,2.f})
	{
		const frame_extent in{w,h},out{w*2+1,h*3+1};const frame_rectangle crop{x,y,w-x,h-y};
		const auto p=plan_fsr(in,crop,out,2,sharpness);
		require(p.easu==test_fsr::amd_constants(in,crop,out)&&p.rcas[0]==std::bit_cast<std::uint32_t>(test_fsr::amd_sharpness(sharpness)),"FSR constants differ from AMD original");++count;
	}
	for(float invalid:{-1.f,2.01f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) rejects([&]{(void)plan_fsr({4,4},{0,0,4,4},{8,8},0,invalid);});
	rejects([]{(void)plan_fsr({4,4},{3,0,2,4},{8,8},0,1);});
	rejects([]{(void)plan_fsr({0,4},{0,0,4,4},{8,8},0,1);});
	rejects([]{(void)fsr_working_bytes({UINT32_MAX,UINT32_MAX},SIZE_MAX);});
	require(!needs_fsr({4,4},{4,8})&&!needs_fsr({4,4},{8,3})&&needs_fsr({4,4},{5,5}),"FSR two-axis condition differs");
	std::cout<<count<<" AMD constant/sharpness bit comparisons passed\n";
}
void gpu(context& ctx)
{
	fsr_upscaler engine(ctx);
	for(auto format:{MTLPixelFormatRGBA32Float,MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA8Unorm,MTLPixelFormatBGRA8Unorm,MTLPixelFormatRGBA8Unorm_sRGB,MTLPixelFormatBGRA8Unorm_sRGB})
	for(unsigned geometry=0;geometry<4;++geometry)
	{
		const unsigned level=geometry==2?1:0,w=geometry==3?3:7,h=geometry==3?3:5;
		const frame_rectangle crop=geometry==0?frame_rectangle{0,0,w,h}:geometry==3?frame_rectangle{1,1,1,1}:frame_rectangle{1,1,w-2,h-2};
		const frame_extent out{crop.width*2+1,crop.height*2+3};
		const auto source=texture(ctx,format,w<<level,h<<level,level);
		for(unsigned variant=0;variant<7;++variant)
		{
			auto pixels=pattern(w,h,variant);upload(ctx,source,level,pixels);
			for(float sharpness:{0.f,1.f,2.f})
			{
				ctx.begin();
				const auto before = ctx.encoding_statistics();
				const auto result=engine.encode(ctx,{source,{level,0,MTLOriginMake(crop.x,crop.y,0),MTLSizeMake(crop.width,crop.height,1)}},out,sharpness);
				const auto after = ctx.encoding_statistics();
				require(after.compute_encoders == before.compute_encoders + 1 &&
					after.checked_dispatches == before.checked_dispatches + 2 &&
					after.dispatch_queue_barriers == before.dispatch_queue_barriers + 1 &&
					after.dispatch_barriers == before.dispatch_barriers + 1,
					"Each FSR pair needs one checked compute encoder and one EASU-to-RCAS barrier");
				ctx.submit().wait();
				compare(download(ctx,result.texture),test_fsr::reference(pixels,{w,h},crop,out,sharpness),.004f);++profiles;
			}
		}
	}
	const auto stats=engine.statistics();require(stats.reused>stats.created&&stats.easu_dispatches==profiles&&stats.rcas_dispatches==profiles,"FSR warm cache failed");
	std::cout << profiles << " checked FSR pairs share an encoder and preserve the EASU-to-RCAS dependency\n";
	std::cout<<profiles<<" GPU profiles: six formats, seven patterns, full/crop/mip/single-pixel, three sharpness stops; warm resources recompute changed source\n";
}
pixel sample(const std::vector<pixel>& pixels,frame_extent size,double u,double v)
{
	const double x=std::clamp(u*size.width-.5,0.,double(size.width-1)),y=std::clamp(v*size.height-.5,0.,double(size.height-1));
	const unsigned x0=std::floor(x),y0=std::floor(y),x1=std::min(x0+1,size.width-1),y1=std::min(y0+1,size.height-1);const float fx=x-x0,fy=y-y0;pixel result;
	for(unsigned c=0;c<4;++c) result[c]=(pixels[y0*size.width+x0][c]*(1-fx)+pixels[y0*size.width+x1][c]*fx)*(1-fy)+(pixels[y1*size.width+x0][c]*(1-fx)+pixels[y1*size.width+x1][c]*fx)*fy;
	return result;
}
void frame(context& ctx)
{
	native_frame output(ctx);const auto left=texture(ctx,MTLPixelFormatRGBA32Float,5,4),right=texture(ctx,MTLPixelFormatRGBA32Float,10,8,true),target=texture(ctx,MTLPixelFormatRGBA16Float,23,19);
	auto l=pattern(5,4,5),r=pattern(5,4,6);upload(ctx,left,0,l);upload(ctx,right,1,r);
	present_image image{left,{0,0,MTLOriginMake(1,1,0),MTLSizeMake(3,2,1)}};image.right_eye=present_eye{right,{1,0,MTLOriginMake(1,1,0),MTLSizeMake(3,2,1)}};
	for(unsigned mode=0;mode<5;++mode) for(unsigned flip=0;flip<4;++flip) for(float gamma:{.8f,1.f,1.2f}) for(bool limited:{false,true})
	{
		frame_policy policy{{3,2},16./9};policy.fsr=true;policy.fsr_sharpness=float(flip%3);policy.gamma=gamma;policy.limited_rgb=limited;policy.stretch=flip&1;policy.stereo.mode=static_cast<stereo_mode>(mode);
		policy.stereo.left={};policy.stereo.right={};policy.stereo.left[0][0]=1;policy.stereo.left[1][1]=1;policy.stereo.right[2][2]=1;
		image.flip_x=flip&1;image.flip_y=flip&2;
		const auto rect=plan_frame_rectangle({3,2},{23,19},policy);const auto size=fsr_eye_extent({rect.width,rect.height},policy.stereo.mode);
		const auto a=test_fsr::reference(l,{5,4},{1,1,3,2},size,policy.fsr_sharpness),b=test_fsr::reference(r,{5,4},{1,1,3,2},size,policy.fsr_sharpness);
		std::vector<pixel> expected(23*19,pixel{0,0,0,1});
		for(unsigned y=rect.y;y<rect.y+rect.height;++y) for(unsigned x=rect.x;x<rect.x+rect.width;++x)
		{
			double u=(x+.5-rect.x)/rect.width,v=(y+.5-rect.y)/rect.height;bool eye=false;
			if(mode==1){eye=u>=.5;u=u*2-eye;} if(mode==2){eye=v>=.5;v=v*2-eye;} if(mode==3) eye=!(y&1);
			if(image.flip_x)u=1-u;if(image.flip_y)v=1-v;auto value=sample(eye?b:a,size,u,v);
			if(mode==4){const auto other=sample(b,size,u,v);value[0]=std::clamp(value[0],0.f,1.f);value[1]=std::clamp(value[1],0.f,1.f);value[2]=std::clamp(other[2],0.f,1.f);}
			for(unsigned c=0;c<3;++c){value[c]=std::pow(value[c],gamma);if(limited)value[c]=(value[c]*220+16)/255;}
			value[3]=1;expected[y*23+x]=value;
		}
		ctx.begin();output.encode(image,target,policy);ctx.submit().wait();compare(download(ctx,target),expected,.006f);++frames;
	}
	ctx.begin();auto invalid=image;invalid.right_eye->texture=target;frame_policy p{{3,2},1};p.fsr=true;p.stereo.mode=stereo_mode::side_by_side;
	const auto token=ctx.recording_submission();rejects([&]{output.encode(invalid,target,p);});require(ctx.recording_submission().same_batch(token),"Invalid FSR preflight changed recording");ctx.cancel();
	(void)output.reclaim_fsr();require(!output.attached()&&NSApp==nil,"FSR frame test created application/view");
	std::cout<<frames<<" native_frame GPU cases: five stereo modes, independent mip/crop eyes, four flips, three gamma, two ranges, AV bars/odd output; original alias preflight\n";
}
void ownership(context& ctx)
{
	const auto source=texture(ctx,MTLPixelFormatRGBA32Float,3,2);auto pixels=pattern(3,2,5);upload(ctx,source,0,pixels);
	const present_eye input{source,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(3,2,1)}};
	fsr_upscaler engine(ctx,2*9*7*16,std::make_shared<heap_budget>(ctx,8u<<20));
	rejects([&]{(void)engine.encode(ctx,input,{9,7},1);});
	ctx.begin();const auto first=engine.encode(ctx,input,{9,7},1);(void)engine.encode(ctx,input,{9,7},1,1);
	require(engine.statistics().working_bytes==2*9*7*16&&engine.statistics().created==2,"FSR two-eye admission differs");
	rejects([&]{(void)engine.encode(ctx,input,{10,8},1,1);});
	require(engine.statistics().working_bytes==9*7*16,"FSR refused resize retained obsolete eye pair");
	(void)engine.encode(ctx,input,{9,7},1,1);require(engine.clear_eye(1),"FSR second-eye eviction failed");
	const auto before=engine.statistics();const auto bypass=engine.encode(ctx,input,{3,9},1);
	require(bypass.texture==source&&engine.statistics().easu_dispatches==before.easu_dispatches&&engine.statistics().bypassed==1,"FSR bypass encoded work");
	rejects([&]{(void)engine.encode(ctx,input,{9,7},1,2);});
	context foreign;foreign.begin();rejects([&]{(void)engine.encode(foreign,input,{9,7},1);});foreign.cancel();
	bool rejected=false;std::thread worker([&]{try{(void)engine.statistics();}catch(const std::logic_error&){rejected=true;}});worker.join();require(rejected,"FSR foreign render thread accepted");++guards;
	auto provider=std::make_shared<test::host_access_fixture>();provider->execute([&](exclusive_guest_memory::view&){rejects([&]{(void)engine.encode(ctx,input,{9,7},1);});});
	ctx.cancel();engine.clear();
	ctx.begin();
	const auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=source;
	pass.colorAttachments[0].loadAction=MTLLoadActionLoad;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
	render_commands facade(ctx,pass);const auto active=ctx.current_render_encoder();
	rejects([&]{(void)engine.encode(ctx,input,{9,7},1);});require(ctx.current_render_encoder()==active,"Rejected FSR ended caller render facade");facade.end();
	for(unsigned bad=0;bad<5;++bad)
	{
		auto invalid=input;frame_extent size{9,7};float sharpness=1;
		if(bad==0)invalid.texture=nil;if(bad==1)invalid.region.level=99;if(bad==2)invalid.region.origin.x=3;
		if(bad==3)size.width=0;if(bad==4)sharpness=std::numeric_limits<float>::quiet_NaN();
		const auto token=ctx.recording_submission();rejects([&]{(void)engine.encode(ctx,invalid,size,sharpness);});
		require(ctx.recording_submission().same_batch(token),"Rejected FSR replaced caller recording");
	}
	ctx.cancel();
	// The queue control block, rather than a recycled context address, identifies ownership.
	alignas(context) std::byte storage[sizeof(context)];auto* original=std::construct_at(reinterpret_cast<context*>(storage));auto orphan=std::make_unique<fsr_upscaler>(*original);std::destroy_at(original);
	auto* replacement=std::construct_at(reinterpret_cast<context*>(storage));replacement->begin();rejects([&]{(void)orphan->encode(*replacement,input,{9,7},1);});replacement->cancel();orphan.reset();std::destroy_at(replacement);
	// Submitted resources survive cache/upscaler/context destruction and a real event gate.
	std::shared_ptr<heap_budget> ledger;
	@autoreleasepool
	{
	id<MTLSharedEvent> gate;std::optional<submission> pending;id<MTLBuffer> readback;
	{
		auto owner=std::make_unique<context>();ledger=std::make_shared<heap_budget>(*owner,8u<<20);owner->set_buffer_budget(ledger);
		auto src=texture(*owner,MTLPixelFormatRGBA32Float,3,2);auto values=pixels;upload(*owner,src,0,values);
		auto scaler=std::make_unique<fsr_upscaler>(*owner,1u<<20,ledger);gate=[owner->device() newSharedEvent];
		struct release { id<MTLSharedEvent> event;~release(){event.signaledValue=1;} } guard{gate};
		readback=make_buffer(*owner,9*7*8,buffer_storage::shared);owner->begin();owner->wait_event(gate,1);
		const auto result=scaler->encode(*owner,{src,input.region},{9,7},1);texture_to_buffer(*owner,result.texture,result.region,readback,0);pending=owner->submit();
		scaler->clear();scaler.reset();require(!pending->ready(),"FSR lifetime gate did not hold work");
		rejects([&]{pending->wait(std::chrono::milliseconds(1));});owner.reset();gate.signaledValue=1;pending->wait();
	}
	const auto bytes=map_buffer(readback,0,readback.length);std::vector<pixel> result(9*7);
	for(unsigned p=0;p<result.size();++p)for(unsigned c=0;c<4;++c){_Float16 h;std::memcpy(&h,bytes.data()+p*8+c*2,2);result[p][c]=float(h);}
	compare(result,test_fsr::reference(pixels,{3,2},{0,0,3,2},{9,7},1),.004f);
	pending.reset();readback=nil;
	}
	require(ledger->statistics().reserved_bytes==0,"FSR native allocation charges leaked after completion");
	std::cout<<guards<<" guards: shape/budget/idle/foreign-context/recycled-address/thread/VM/alias; event-gated timeout and cache/context destruction preserve output and retire all ledger charges\n";
}
}
int main()
{
	@autoreleasepool {try {require(NSApp==nil,"Test began with App");constants();context ctx;gpu(ctx);frame(ctx);ownership(ctx);require(NSApp==nil,"Test created App");std::cout<<checked<<" channel comparisons; maximum absolute error="<<maximum_error<<"; native Metal4 only; no App/view/window/drawable/guest VM\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
}
