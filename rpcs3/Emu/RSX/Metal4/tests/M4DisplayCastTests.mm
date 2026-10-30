#include "M4Display.h"
#include "M4SampleTransfer.h"
#include "M4MemoryPressure.h"
#include "M4FramebufferSnapshot.h"
#import <AppKit/NSApplication.h>
#include <bit>
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
unsigned profiles=0,guards=0;std::size_t channels=0;std::uint64_t casts=0,hits=0;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<typename F> void rejects(F&& fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}require(failed,"Invalid native display cast accepted");++guards;}
class memory final : public guest_surface_memory
{
  public:
	std::array<std::byte,4096> bytes{};
	unsigned reads=0,writes=0;
	std::vector<guest_surface_range> forbidden,read_ranges;
	std::vector<std::pair<guest_surface_range,std::uint64_t>> changes;
	std::uint64_t revision=1;
	std::shared_ptr<int> identity=std::make_shared<int>(0);
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		validate_guest_surface_range(range);return std::vector<storage_identity_span>{{identity,canonical(range).address,range}};
	}
	static guest_surface_range canonical(guest_surface_range range){if(range.address>=0x10000)range.address-=0x10000;return range;}
	bool test_storage(const storage_test& test) override
	{
		return test(false,[](auto a,auto b){return overlaps(canonical(a),canonical(b));});
	}
	bool test_rebase(const rebase_test& test) override
	{
		return test([](auto source,auto request)->std::optional<guest_surface_range>
		{
			const auto a=canonical(source),b=canonical(request);
			if(b.address<a.address||b.size>a.size||b.address-a.address>a.size-b.size)return {};
			return guest_surface_range{source.address+b.address-a.address,b.size};
		},[](auto a,auto b){a=canonical(a);b=canonical(b);return a.address==b.address&&a.size==b.size;});
	}
	guest_surface_snapshot read_surface(const guest_surface_layout& image) override
	{
		++reads;const auto range=canonical(image.range);require(range.address<=bytes.size()&&range.size<=bytes.size()-range.address,"Display fixture read outside VM");
		for(auto blocked:forbidden)require(!overlaps(blocked,range),"Mixed display tried to read GPU-authoritative native pixels");read_ranges.push_back(range);
		return {surface_revision(image),{bytes.begin()+range.address,bytes.begin()+range.address+range.size}};
	}
	std::uint64_t surface_revision(const guest_surface_layout& image) override
	{
		auto logical=image;logical.range=canonical(image.range);std::uint64_t value=1;
		for(auto [range,version]:changes)if(overlaps(range,logical))value=version;return value;
	}
	void notify(guest_surface_range range){changes.push_back({canonical(range),++revision});}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout&,std::uint64_t,std::span<const std::byte>) override {++writes;throw std::runtime_error("Display cast wrote guest VM");}
	void write(guest_surface_range,std::span<const std::byte>) override {++writes;throw std::runtime_error("Display cast wrote guest VM");}
};
constexpr std::array formats{surface_color::x1rgb5_zero,surface_color::x1rgb5_one,surface_color::rgb565,surface_color::xrgb8_zero,surface_color::xrgb8_one,
	surface_color::argb8,surface_color::b8,surface_color::gb8,surface_color::rgba16f,surface_color::rgba32f,surface_color::r32f,
	surface_color::xbgr8_zero,surface_color::xbgr8_one,surface_color::abgr8};
constexpr std::string_view read_source=R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void display_read(texture2d<float,access::read> input [[texture(0)]],device uint4* output [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {
 if(p.x<input.get_width()&&p.y<input.get_height()) output[p.y*input.get_width()+p.x]=as_type<uint4>(input.read(p));
}
)metal";
std::vector<std::array<std::uint32_t,4>> download(context& ctx,id<MTLTexture> image)
{
	helper_catalog compile(ctx);const auto library=compile.library(read_source,@"Display cast test reader");const auto pipeline=compile.compute(library,@"display_read");
	const auto buffer=make_buffer(ctx,image.width*image.height*16,buffer_storage::shared);
	ctx.begin();bindings resources;resources.set_texture(0,image);resources.set_buffer(0,buffer,0);const auto table=resources.snapshot(ctx);ctx.keep_alive(pipeline);
	const auto encoder=ctx.compute();[encoder setComputePipelineState:pipeline];[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(image.width,image.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];ctx.submit().wait();
	const auto data=map_buffer(buffer,0,buffer.length);std::vector<std::array<std::uint32_t,4>> result(image.width*image.height);std::memcpy(result.data(),data.data(),data.size());return result;
}
struct fixture
{
	std::unique_ptr<context> ctx=std::make_unique<context>(3);
	std::shared_ptr<heap_budget> budget=std::make_shared<heap_budget>(*ctx,32u<<20);
	std::shared_ptr<memory> ram=std::make_shared<memory>();
	std::unique_ptr<surface_cache> surfaces;
	std::unique_ptr<display_resolver> display;
	std::unique_ptr<surface_color_transfer> codec;
	std::unique_ptr<sample_transfer> samples;
	cached_surface_ref source;
	std::vector<std::byte> pixels=std::vector<std::byte>(64*6);
	unsigned pitch;surface_color format;unsigned count;
	fixture(surface_color format,unsigned count,unsigned odd,unsigned slots=3,std::shared_ptr<tile_state> tiles={},std::optional<swizzle_region> morton={}):ctx(std::make_unique<context>(slots)),pitch(64+odd),format(format),count(count)
	{
		ctx->set_buffer_budget(budget);ram->bytes.fill(std::byte{0xa7});
		cpu_capture_dispatch capture;
		if(tiles)capture=[tiles](const auto& work){tiles->execute(work);};
		surfaces=std::make_unique<surface_cache>(*ctx,ram,32u<<20,budget,capture);
		display=std::make_unique<display_resolver>(*ctx,ram,[](unsigned offset,unsigned,std::size_t){return offset;},budget,4096,capture,tiles);
		codec=std::make_unique<surface_color_transfer>(*ctx);samples=std::make_unique<sample_transfer>(*ctx);
		for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte((i*37+i/7+19)%256);
		if(format==surface_color::rgba16f)
			for(unsigned i=0;i<pixels.size();i+=2){const std::uint16_t v=0x3000+((i/2)%12)*0x100;pixels[i]=std::byte(v>>8);pixels[i+1]=std::byte(v);}
		if(format==surface_color::rgba32f||format==surface_color::r32f)
			for(unsigned i=0;i<pixels.size();i+=4){const std::uint32_t v=0x3e000000+((i/4+(i/64)*3)%8)*0x00100000;for(unsigned c=0;c<4;++c)pixels[i+c]=std::byte(v>>(24-c*8));}
		const auto bytes=color_layout(format).pixel_bytes;const auto scale=count==1?sample_layout{1,1}:rsx_sample_layout(count);const unsigned w=64/bytes,h=6;
		cached_surface_description description{0x100,{w/scale.x,h/scale.y,count,format,false},morton ? 0u : pitch,morton};
		if(tiles)
		{
			framebuffer_surface surface;surface.address=0x100;surface.guest=plan_guest_surface(0x100,w,h,bytes,morton ? 0u : pitch,morton);
			map_framebuffer_surface(surface,0x100,0,[](unsigned offset,unsigned,std::size_t){return offset;},{},tiles,true,4096);
			description.mapping=surface.mapping;description.segments=surface.mapping->state().segments;
		}
		source=surfaces->acquire(description);surfaces->prepare(source,surface_initialization::background);
		if(morton)
		{
			guest_surface_byte_cursor active(plan_guest_surface(0x100,w,h,bytes,0,morton));
			while(auto run=active.next())ram->forbidden.push_back(*run);
		}
		write();
	}
	void write()
	{
		const unsigned w=64/color_layout(format).pixel_bytes,h=6;
		const std::array uses{cached_surface_use{source,true}};
		surfaces->draw(uses,[&](context& commands,auto images)
		{
			const auto input=make_upload(commands,pixels);texture_region whole{0,0,MTLOriginMake(0,0,0),MTLSizeMake(w,h,1)};
			if(count==1)codec->upload(commands,format,input,{},images[0],whole);
			else
			{
				auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:describe_surface_color(format).format width:w height:h mipmapped:NO];
				d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
				const auto expanded=make_texture(commands,d);codec->upload(commands,format,input,{},expanded,whole);samples->import(commands,expanded,images[0]);
			}
		});
	}
};
void verify(context& ctx,const present_image& image,const std::vector<std::byte>& bytes,unsigned source_x,unsigned width,display_format format,unsigned height=3,unsigned source_y=1)
{
	const auto actual=download(ctx,image.texture);require(actual.size()==width*height,"Display cast image dimensions differ");
	const unsigned bpp=display_pixel_bytes(format);
	for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned c=0;c<4;++c)
	{
		float expected=1;
		const auto at=(y+source_y)*64+source_x+x*bpp;
		if(c<3)
		{
			if(format==display_format::rgbx16f){const auto h=std::uint16_t((unsigned(bytes[at+c*2])<<8)|unsigned(bytes[at+c*2+1]));expected=float(std::bit_cast<_Float16>(h));}
			else expected=unsigned(bytes[at+(format==display_format::xrgb8?c+1:3-c)])/255.f;
		}
		const float got=std::bit_cast<float>(actual[y*width+x][c]);++channels;
		const bool equal=std::isnan(expected)?std::isnan(got):format==display_format::rgbx16f||c==3?actual[y*width+x][c]==std::bit_cast<std::uint32_t>(expected):std::abs(got-expected)<=1e-7f;
		if(!equal){std::cerr<<"profile="<<profiles<<" channel="<<c<<" pixel="<<x<<','<<y<<" actual="<<got<<" expected="<<expected<<'\n';throw std::runtime_error("Native display bitcast/sample mapping differs from original guest byte order");}
	}
}
void matrix()
{
	for(const auto format:formats)for(unsigned count:{1u,2u,4u})for(unsigned odd:{0u,1u})
	{
		@autoreleasepool
		{
			fixture f(format,count,odd);
			for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})for(unsigned x:{0u,1u})
			{
				const display_snapshot image{0x100+f.pitch+x,3,3,f.pitch,av};
				const auto resolved=f.display->prepare(image,*f.surfaces).finish();verify(*f.ctx,resolved,f.pixels,x,3,av);++profiles;
				const auto allocations=f.budget->statistics().created;const auto encoders=f.ctx->encoding_statistics();
				const auto casts_before=f.display->statistics().native_casts;
				for(unsigned repeat=0;repeat<3;++repeat)require(f.display->prepare(image,*f.surfaces).finish().texture==resolved.texture,"Warm native display replaced output view");
				require(f.budget->statistics().created==allocations&&f.ctx->encoding_statistics().compute_encoders==encoders.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoders.render_encoders&&f.display->statistics().native_casts==casts_before,"Warm native display allocated or re-encoded samples/casts");
				require(!f.ram->reads&&!f.ram->writes&&f.surfaces->status(f.source).dirty,"Native display cast touched VM pixels or lost dirty authority");
				for(auto byte:f.ram->bytes)require(byte==std::byte{0xa7},"Native display changed original guest bytes/padding");
			}
			casts+=f.display->statistics().native_casts;hits+=f.display->statistics().native_cast_hits;
		}
	}
	std::cout<<profiles<<" GPU display profiles:14 color formats/1-2-4 samples/3 AV formats/odd pitch/partial source texels;"<<channels<<" channel checks against original BE guest bytes,opaque alpha;finite half bits exact,NaN classification only;"<<casts<<" native casts/"<<hits<<" cast hits;warm views without allocations/encoders;zero VM pixel reads/writes\n";
}
std::size_t morton_oracle(std::array<std::size_t,3> at,std::array<std::size_t,3> size)
{
	std::size_t index=0,weight=1;
	for(std::size_t bit=1;bit<std::max({size[0],size[1],size[2]});bit*=2)for(unsigned axis=0;axis<3;++axis)
		if(bit<size[axis]){if(at[axis]&bit)index+=weight;weight*=2;}
	return index;
}
struct morton_reference
{
	std::vector<std::byte> bytes;
	std::vector<bool> active;
	morton_reference(const fixture& f,const swizzle_region& layout)
	{
		const auto bpp=color_layout(f.format).pixel_bytes;
		bytes.resize(layout.storage[0]*layout.storage[1]*layout.storage[2]*bpp,std::byte{0xa7});active.resize(bytes.size());
		for(std::size_t i=0;i<bytes.size()&&i+0x100<f.ram->bytes.size();++i)bytes[i]=f.ram->bytes[0x100+i];
		for(unsigned y=0;y<6;++y)for(unsigned x=0;x<64/bpp;++x)for(unsigned byte=0;byte<bpp;++byte)
		{
			const auto index=morton_oracle({x+layout.origin[0],y+layout.origin[1],layout.origin[2]},layout.storage)*bpp+byte;
			bytes[index]=f.pixels[y*64+x*bpp+byte];active[index]=true;
		}
	}
	bool contains(unsigned offset,unsigned width,unsigned height,unsigned pitch,display_format format) const
	{
		for(unsigned y=0;y<height;++y)for(unsigned byte=0;byte<width*display_pixel_bytes(format);++byte)
		{
			const auto at=offset+y*pitch+byte;if(at>=active.size()||!active[at])return false;
		}
		return true;
	}
	void verify_image(context& ctx,const present_image& image,unsigned offset,unsigned width,unsigned height,unsigned pitch,display_format format,bool mixed=false) const
	{
		require(mixed ? offset+(height-1)*pitch+width*display_pixel_bytes(format)<=bytes.size() : contains(offset,width,height,pitch,format),"Morton test expected unowned/out-of-range storage pixels");
		std::vector<std::byte> rows(height*64);
		for(unsigned y=0;y<height;++y)std::copy_n(bytes.begin()+offset+y*pitch,width*display_pixel_bytes(format),rows.begin()+y*64);
		verify(ctx,image,rows,0,width,format,height,0);
	}
};
void morton_matrix()
{
	unsigned cases=0,mixed_profiles=0;std::size_t begin=channels;
	for(auto format:formats)for(unsigned count:{1u,2u,4u})for(unsigned shape:{0u,1u,2u})
	{
		@autoreleasepool
		{
			const auto bytes=color_layout(format).pixel_bytes;const std::size_t width=64/bytes;
			const swizzle_region layout=shape==0?swizzle_region{{width,8,1},{},{width,6,1}}:
				shape==1?swizzle_region{{width*2,16,1},{1,1,0},{width,6,1}}:swizzle_region{{width*2,16,2},{1,1,1},{width,6,1}};
			auto tiles=std::make_shared<tile_state>();tiles->update(0,{0,0,0x10000,1024,8,0,3,true});
			fixture f(format,count,1,3,tiles,layout);morton_reference expected(f,layout);
			for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})for(unsigned prefix:{0u,1u})
			{
				unsigned tested=0;
				for(auto extent:{std::array{3u,3u},std::array{2u,2u},std::array{1u,1u}})
				{
					const unsigned pitch=extent[0]*display_pixel_bytes(av)+1;std::optional<unsigned> offset;
					for(unsigned at=prefix%bytes;at<expected.bytes.size();at+=bytes)
						if(expected.contains(at,extent[0],extent[1],pitch,av)){offset=at;break;}
					if(!offset)continue;
					const display_snapshot descriptor{0x100+*offset,extent[0],extent[1],pitch,av};
					const auto reads=f.ram->reads;
					const auto image=f.display->prepare(descriptor,*f.surfaces).finish();
					expected.verify_image(*f.ctx,image,*offset,extent[0],extent[1],pitch,av);++cases;++profiles;++tested;
					const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;const auto encoders=f.ctx->encoding_statistics();
					auto alias=descriptor;alias.offset+=0x10000;
					for(unsigned repeat=0;repeat<3;++repeat)require(f.display->prepare(repeat==2?alias:descriptor,*f.surfaces).finish().texture==image.texture,"Morton ready/physical alias did not reuse unchanged output");
					require(f.ram->reads==reads&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded&&
						f.ctx->encoding_statistics().compute_encoders==encoders.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoders.render_encoders,"Morton warm output allocated/re-encoded");
				}
				if(!tested)
				{
					// Interleaved Z cells can split one AV word. Only its missing
					// bytes may be captured from VM; native reads are forbidden.
					unsigned at=prefix%bytes;while(at<expected.bytes.size()&&!expected.active[at])at+=bytes;
					require(at<expected.bytes.size()&&!expected.contains(at,1,1,display_pixel_bytes(av)+1,av),"Morton refusal fixture lacks active overlap/hole");
					const display_snapshot descriptor{0x100+at,1,1,static_cast<unsigned>(display_pixel_bytes(av)+1),av};
					const auto image=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,image,at,1,1,descriptor.pitch,av,true);++mixed_profiles;
					const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;const auto reads=f.ram->reads;
					require(f.display->prepare(descriptor,*f.surfaces).finish().texture==image.texture&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded&&f.ram->reads==reads,"Mixed Morton word warm hit recaptured orre-encoded");
				}
			}
			require(!f.ram->writes&&f.surfaces->status(f.source).dirty,"Morton display lost native dirty authority orwrote VM pixels");
			for(auto value:f.ram->bytes)require(value==std::byte{0xa7},"Morton display changed original VM/padding");
		}
	}
	std::cout<<"Morton display:"<<cases<<" fullynative GPU windows,"<<mixed_profiles<<" mixed Z-cell words/"<<channels-begin<<" channels;14colors x1-2-4samples xrectangular/cropped/3Dslice x3AV,odd pitch/partial source texels,independent raw-byte oracle,local compressed tag,opaque views;native warm/physical reuse withoutallocation/recast/VM pixel access;composites capture onlymissing bytes,warm withoutrecapture\n";
}
void fill_vm_holes(fixture& f,const swizzle_region& layout)
{
	const morton_reference geometry(f,layout);
	for(unsigned i=0;i<geometry.bytes.size()&&0x100+i<f.ram->bytes.size();++i)
		if(!geometry.active[i])f.ram->bytes[0x100+i]=std::byte((i*17+i/13+9)%256);
}
void morton_mixed_matrix()
{
	unsigned cases=0;std::size_t begin=channels,total_bytes=0,total_ranges=0;
	for(auto format:formats)for(unsigned count:{1u,2u,4u})
	{
		@autoreleasepool
		{
			const auto bpp=color_layout(format).pixel_bytes;const std::size_t width=64/bpp;
			const swizzle_region layout{{width,8,1},{},{width,6,1}};fixture f(format,count,1,3,{},layout);fill_vm_holes(f,layout);
			const auto original=f.ram->bytes;const morton_reference expected(f,layout);
			unsigned hole=0;while(hole<expected.active.size()&&expected.active[hole])++hole;require(hole>0&&hole<expected.active.size(),"Mixed matrix lacks a native/VM boundary");
			const unsigned offset=hole-(bpp>1?bpp-1:1);
			for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
			{
				f.display->clear();f.ram->read_ranges.clear();const auto before=f.display->statistics();
				const unsigned pitch=3*display_pixel_bytes(av)+1;const display_snapshot descriptor{0x100+offset,3,3,pitch,av};
				const auto image=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,image,offset,3,3,pitch,av,true);
				std::size_t missing=0,actual=0;
				for(unsigned y=0;y<3;++y)for(unsigned byte=0;byte<3*display_pixel_bytes(av);++byte)missing+=!expected.active[offset+y*pitch+byte];
				for(auto run:f.ram->read_ranges)for(unsigned byte=0;byte<run.size;++byte)
				{
					const auto at=run.address+byte-0x100;require(at>=offset&&!expected.active[at],"Mixed matrix read a native byte");
					const auto relative=at-offset;require(relative/pitch<3&&relative%pitch<3*display_pixel_bytes(av),"Mixed matrix captured AV row padding");++actual;
				}
				const auto after=f.display->statistics();require(missing>0&&actual==missing&&after.captured_vm_bytes-before.captured_vm_bytes==missing&&
					after.captured_vm_ranges-before.captured_vm_ranges==f.ram->read_ranges.size()&&after.mixed_casts==before.mixed_casts+1,"Mixed matrix patch byte/range accounting differs");
				total_bytes+=missing;total_ranges+=f.ram->read_ranges.size();
				const auto allocations=f.budget->statistics().created;const auto reads=f.ram->reads;const auto encoders=f.ctx->encoding_statistics();
				require(f.display->prepare(descriptor,*f.surfaces).finish().texture==image.texture&&f.budget->statistics().created==allocations&&f.ram->reads==reads&&
					f.ctx->encoding_statistics().compute_encoders==encoders.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoders.render_encoders,"Mixed matrix warm hit allocated/recaptured/re-encoded");
				auto alias=descriptor;alias.offset+=0x10000;const auto physical=f.display->prepare(alias,*f.surfaces).finish();expected.verify_image(*f.ctx,physical,offset,3,3,pitch,av,true);
				const auto physical_stats=f.display->statistics();require(physical_stats.captured_vm_bytes-after.captured_vm_bytes==missing&&
					physical_stats.captured_vm_ranges-after.captured_vm_ranges==after.captured_vm_ranges-before.captured_vm_ranges,"Physical composite reused incomparable VM-watch versions");
				total_bytes+=physical_stats.captured_vm_bytes-after.captured_vm_bytes;total_ranges+=physical_stats.captured_vm_ranges-after.captured_vm_ranges;
				++cases;
			}
			require(!f.ram->writes&&f.ram->bytes==original&&f.surfaces->status(f.source).dirty,"Mixed matrix changed VM/padding orlost dirty native authority");
		}
	}
	std::cout<<"Mixed Morton matrix:"<<cases<<" GPU profiles/"<<channels-begin<<" channel checks,14colors x1-2-4samples x3AV,unaligned/odd pitch,multirange scatter;"<<total_bytes<<" exact missing bytes/"<<total_ranges<<" capture ranges;no native/padding capture,physical aliases correct,warm withoutrecapture/allocation/encoding,original VM/padding unchanged\n";
}
void morton_mixed_lifetime()
{
	const swizzle_region layout{{16,8,1},{},{16,6,1}};
	const display_snapshot descriptor{0x100+157,3,3,13};
	{
		auto tiles=std::make_shared<tile_state>();fixture f(surface_color::r32f,4,1,3,tiles,layout);fill_vm_holes(f,layout);morton_reference expected(f,layout);
		const auto gate=[f.ctx->device() newSharedEvent],other_gate=[f.ctx->device() newSharedEvent];
		struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate},unblock_other{other_gate};
		auto first=f.display->prepare(descriptor,*f.surfaces,std::array{display_upload_dependency{gate,1}});require(!first.ready(),"Mixed gate did not retain actual GPU work");rejects([&]{(void)first.finish(std::chrono::milliseconds(1));});
		const auto reads=f.ram->reads;const auto created=f.budget->statistics().created;auto duplicate=f.display->prepare(descriptor,*f.surfaces);
		require(!duplicate.ready()&&f.ram->reads==reads&&f.budget->statistics().created==created&&f.display->statistics().mixed_casts==1,"Identical pending mixed display recaptured orcopied");
		auto other=descriptor;other.offset+=1;auto independent=f.display->prepare(other,*f.surfaces,std::array{display_upload_dependency{other_gate,1}});
		require(!independent.ready()&&f.display->statistics().mixed_casts==2,"Different mixed byte window borrowed first output");
		require(first.cancel_pending(),"Mixed copy/scatter could not cancel before commit");other_gate.signaledValue=1;rejects([&]{(void)duplicate.finish();});
		const auto image=independent.finish();f.ctx->recover_completed_errors();expected.verify_image(*f.ctx,image,158,3,3,13,other.format,true);
		gate.signaledValue=1;const auto original=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,original,157,3,3,13,descriptor.format,true);
		auto ready=f.display->prepare(descriptor,*f.surfaces);(void)ready.finish();const auto mixed=f.display->statistics().mixed_casts;
		f.ram->bytes[0x100+160]=std::byte{0x44};f.ram->notify({0x100+160,1});expected=morton_reference(f,layout);
		rejects([&]{(void)ready.finish();});const auto latest=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,latest,157,3,3,13,descriptor.format,true);
		require(f.display->statistics().mixed_casts==mixed+1&&latest.texture!=original.texture&&f.surfaces->status(f.source).dirty,"Hole-only write reused stale payload orlost native authority");
		for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
		{
			const unsigned pitch=3*display_pixel_bytes(av)+1;const display_snapshot pair{0x100+157,3,3,pitch,av,0x100+413};
			auto job=f.display->prepare_frame(pair,*f.surfaces);const auto paired=job.finish();require(paired.right_eye.has_value(),"Mixed stereo dropped right eye");
			expected.verify_image(*f.ctx,paired,157,3,3,pitch,av,true);expected.verify_image(*f.ctx,{paired.right_eye->texture,paired.right_eye->region},413,3,3,pitch,av,true);
			const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().mixed_casts;const auto reads=f.ram->reads;
			const auto repeat=f.display->prepare_frame(pair,*f.surfaces).finish();require(repeat.texture==paired.texture&&repeat.right_eye->texture==paired.right_eye->texture&&f.budget->statistics().created==allocations&&f.display->statistics().mixed_casts==encoded&&f.ram->reads==reads,"Mixed stereo warm hit allocated/recaptured/encoded");
			auto left=f.display->prepare(display_eye(pair,0),*f.surfaces);require(!expected.active[416],"Mixed right-eye mutation is not a VM hole");
			f.ram->bytes[0x100+416]=std::byte(32+unsigned(av)*9);f.ram->notify({0x100+416,1});expected=morton_reference(f,layout);
			require(left.current(),"Right-only VM hole write invalidated disjoint left eye");rejects([&]{(void)job.finish();});
			const auto fresh=f.display->prepare_frame(pair,*f.surfaces).finish();require(fresh.texture==paired.texture&&fresh.right_eye->texture!=paired.right_eye->texture&&f.display->statistics().mixed_casts==encoded+1,"Right-only mixed update replaced left orreused stale right payload");
			expected.verify_image(*f.ctx,fresh,157,3,3,pitch,av,true);expected.verify_image(*f.ctx,{fresh.right_eye->texture,fresh.right_eye->region},413,3,3,pitch,av,true);
		}
	}
	@autoreleasepool
	{
		fixture f(surface_color::r32f,4,1,1,{},layout);fill_vm_holes(f,layout);morton_reference expected(f,layout);
		guest_texture_cache textures(*f.ctx,f.ram,[](unsigned o,unsigned,std::size_t){return o;},4096,f.budget);memory_pressure_owner pressure(*f.ctx,*f.surfaces,textures,f.budget,f.display.get());
		for(unsigned offset:{157u,158u,159u}){ @autoreleasepool { (void)f.display->prepare({0x100+offset,3,3,13},*f.surfaces).finish(); } }
		(void)f.surfaces->reclaim_idle();const auto before=f.budget->statistics();const auto leave=std::min<std::size_t>(1024,f.display->statistics().bytes/8);
		id<MTLBuffer> blocker=f.budget->shared_buffer(before.budget-before.reserved_bytes-leave);
		const auto image=f.display->prepare({0x100+156,3,3,13},*f.surfaces).finish();
		require(f.budget->statistics().recovery_successes>before.recovery_successes&&f.display->statistics().evictions>0&&f.surfaces->status(f.source).dirty,"Mixed payload/descriptor/scratch admission did not recover ledger");
		blocker=nil;expected.verify_image(*f.ctx,image,156,3,3,13,descriptor.format,true);
	}
	std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> provider;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan;std::optional<morton_reference> expected;id<MTLSharedEvent> gate;
		{
			auto f=std::make_unique<fixture>(surface_color::r32f,4,1,3,std::make_shared<tile_state>(),layout);fill_vm_holes(*f,layout);expected.emplace(*f,layout);provider=f->ram;ledger=f->budget;
			gate=[f->ctx->device() newSharedEvent];orphan.emplace(f->display->prepare(descriptor,*f->surfaces,std::array{display_upload_dependency{gate,1}}));
			require(!orphan->ready(),"Mixed orphan GPU gate did not hold");f.reset();require(!provider.expired(),"Mixed orphan lost native/VM receipt owners");
		}
		gate.signaledValue=1;const auto image=orphan->finish();context reader;expected->verify_image(reader,image,157,3,3,13,descriptor.format,true);orphan.reset();require(provider.expired(),"Mixed orphan kept provider after last job");
	}
	require(ledger->statistics().reserved_bytes==0,"Mixed orphan leaked native/VM upload/descriptor allocations");
	{
		fixture f(surface_color::r32f,4,1,3,{},layout);morton_reference expected(f,layout);const auto second=f.surfaces->acquire({0x100+160,{1,1,1,surface_color::argb8,false},4,{}});
		f.surfaces->clear(second,0xf0,0x00445566,0,255,{0,0,1,1});const auto reads=f.ram->reads;
		f.ram->forbidden.push_back({0x100+160,4});const std::array word{std::byte{0},std::byte{0x44},std::byte{0x55},std::byte{0x66}};
		std::copy(word.begin(),word.end(),expected.bytes.begin()+160);
		const auto image=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,image,157,3,3,13,descriptor.format,true);
		require(f.display->statistics().assembly_casts==1&&f.surfaces->status(f.source).dirty&&f.surfaces->status(second).dirty,"Second native owner did not remainGPU-authoritative");
		const auto created=f.budget->statistics().created,encoded=f.display->statistics().assembly_casts;const auto current_reads=f.ram->reads;
		require(f.display->prepare(descriptor,*f.surfaces).finish().texture==image.texture&&f.budget->statistics().created==created&&f.display->statistics().assembly_casts==encoded&&f.ram->reads==current_reads,"Two-source assembly warm hit recaptured/re-encoded");(void)reads;
	}
	std::cout<<"Mixed owners:real gated sharing/timeout/independent-window cancellation,ready VM-hole replacement,three stereo AV pairs/warm views/right-only VM writes,full-ledger payload/descriptor/scratch recovery andgated resolver/cache/context orphan with0oldcharges passed;second dirty native owner nowassembled natively withstablewarm output\n";
}
void native_assembly_matrix()
{
	unsigned cases=0;const auto begin=channels;std::size_t vm_bytes=0;
	for(std::size_t index=0;index<formats.size();++index)for(unsigned count:{1u,2u,4u})
	{
		@autoreleasepool
		{
			const auto format=formats[index],other=formats[(index+5)%formats.size()];const std::size_t width=64/color_layout(format).pixel_bytes;
			const swizzle_region layout{{width,8,1},{},{width,6,1}};fixture f(format,count,1,3,{},layout);fill_vm_holes(f,layout);
			const auto scale=count==1?sample_layout{1,1}:rsx_sample_layout(count);
			const unsigned bw=32/color_layout(other).pixel_bytes;
			const auto second=f.surfaces->acquire({0x100+520,{bw/scale.x,2/scale.y,count,other,false},33,{}});
			f.surfaces->prepare(second,surface_initialization::background);std::vector<std::byte> pixels(64);
			for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte((i*71+index*13+23)%256);
			if(other==surface_color::rgba16f)for(unsigned i=0;i<pixels.size();i+=2){const unsigned v=0x3100+(i/2%11)*0x100;pixels[i]=std::byte(v>>8);pixels[i+1]=std::byte(v);}
			if(other==surface_color::rgba32f||other==surface_color::r32f)for(unsigned i=0;i<pixels.size();i+=4){const unsigned v=0x3f100000+(i/4%7)*0x00100000;for(unsigned c=0;c<4;++c)pixels[i+c]=std::byte(v>>(24-c*8));}
			const std::array uses{cached_surface_use{second,true}};
			f.surfaces->draw(uses,[&](context& commands,auto images)
			{
				const auto input=make_upload(commands,pixels);texture_region region{0,0,MTLOriginMake(0,0,0),MTLSizeMake(bw,2,1)};
				if(count==1)f.codec->upload(commands,other,input,{},images[0],region);
				else
				{
					auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:describe_surface_color(other).format width:bw height:2 mipmapped:NO];
					d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;
					const auto expanded=make_texture(commands,d);f.codec->upload(commands,other,input,{},expanded,region);f.samples->import(commands,expanded,images[0]);
				}
			});
			for(unsigned y=0;y<2;++y)f.ram->forbidden.push_back({0x100+520+y*33,32});
			const auto original=f.ram->bytes;morton_reference expected(f,layout);expected.bytes.resize(1024,std::byte{0xa7});expected.active.resize(1024);
			for(unsigned y=0;y<2;++y)for(unsigned byte=0;byte<32;++byte){expected.bytes[520+y*33+byte]=pixels[y*32+byte];expected.active[520+y*33+byte]=true;}
			for(unsigned phase=0;phase<2;++phase)
			{
				for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
				{
					const unsigned pitch=3*display_pixel_bytes(av)+1,height=av==display_format::rgbx16f?(phase?7:9):(phase?13:17),offset=phase?345:333;
					const display_snapshot descriptor{0x100+offset,3,height,pitch,av};f.display->clear();f.ram->read_ranges.clear();const auto before=f.display->statistics();
					const auto image=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,image,offset,3,height,pitch,av,true);
					std::size_t missing=0,captured=0;for(unsigned y=0;y<height;++y)for(unsigned byte=0;byte<3*display_pixel_bytes(av);++byte)missing+=!expected.active[offset+y*pitch+byte];
					for(auto run:f.ram->read_ranges)for(unsigned byte=0;byte<run.size;++byte)
					{
						const auto at=run.address+byte-0x100,relative=at-offset;require(at>=offset&&!expected.active[at]&&relative/pitch<height&&relative%pitch<3*display_pixel_bytes(av),"Assembly captured native bytes orAV padding");++captured;
					}
					const auto after=f.display->statistics();require(captured==missing&&after.captured_vm_bytes-before.captured_vm_bytes==missing&&after.assembly_casts==before.assembly_casts+1,"Assembly exact complement/encoding accounting differs");vm_bytes+=missing;
					const auto reads=f.ram->reads;const auto created=f.budget->statistics().created;const auto encoded=f.ctx->encoding_statistics();
					require(f.display->prepare(descriptor,*f.surfaces).finish().texture==image.texture&&f.ram->reads==reads&&f.budget->statistics().created==created&&f.ctx->encoding_statistics().compute_encoders==encoded.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoded.render_encoders,"Assembly warm reuse allocated/captured/encoded");
					auto alias=descriptor;alias.offset+=0x10000;const auto physical=f.display->prepare(alias,*f.surfaces).finish();expected.verify_image(*f.ctx,physical,offset,3,height,pitch,av,true);++cases;
				}
			}
			require(!f.ram->writes&&f.ram->bytes==original&&f.surfaces->status(f.source).dirty,"Assembly changed VM orlost native authority");
		}
	}
	std::cout<<"Native assembly matrix:"<<cases<<" primary+physical-alias GPU profiles/"<<channels-begin<<" channels,14 rotated color pairs/1-2-4samples/3AV;Morton+odd33byte linear sources acrossallocation gaps,one-source windows withoutside-allocation VM,"<<vm_bytes<<" primary missing bytes;native/padding excluded,stablewarm withoutallocation/capture/encoding,VM unchanged\n";
}
void native_assembly_lifetime()
{
	const swizzle_region layout{{16,8,1},{},{16,6,1}};const display_snapshot descriptor{0x100+157,3,3,13};
	const auto add=[](fixture& f,morton_reference& expected,unsigned offset,unsigned value)
	{
		for(unsigned byte=0;byte<4;++byte)require(!expected.active[offset+byte],"Assembly lifecycle source overlaps existing native pixels");
		const auto source=f.surfaces->acquire({0x100+offset,{1,1,1,surface_color::argb8,false},4,{}});
		f.surfaces->clear(source,0xf0,value,0,255,{0,0,1,1});f.ram->forbidden.push_back({0x100+offset,4});
		for(unsigned byte=0;byte<4;++byte){expected.bytes[offset+byte]=std::byte(value>>(24-byte*8));expected.active[offset+byte]=true;}return source;
	};
	{
		fixture f(surface_color::r32f,4,1,3,{},layout);fill_vm_holes(f,layout);morton_reference expected(f,layout);(void)add(f,expected,160,0x00445566);
		const auto gate=[f.ctx->device() newSharedEvent],other=[f.ctx->device() newSharedEvent];
		struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate},unblock_other{other};
		auto first=f.display->prepare(descriptor,*f.surfaces,std::array{display_upload_dependency{gate,1}});require(!first.ready(),"Assembly gate did not retain GPU work");rejects([&]{(void)first.finish(std::chrono::milliseconds(1));});
		const auto reads=f.ram->reads;const auto created=f.budget->statistics().created;auto duplicate=f.display->prepare(descriptor,*f.surfaces);
		require(!duplicate.ready()&&f.ram->reads==reads&&f.budget->statistics().created==created&&f.display->statistics().assembly_casts==1,"Pending assembly did not share wholework");
		auto shifted=descriptor;shifted.offset+=1;auto independent=f.display->prepare(shifted,*f.surfaces,std::array{display_upload_dependency{other,1}});
		require(!independent.ready()&&f.display->statistics().assembly_casts==2,"Independent assembly borrowed pending sourcecopies");
		require(first.cancel_pending(),"Assembly could not cancel pending copy");other.signaledValue=1;rejects([&]{(void)duplicate.finish();});const auto image=independent.finish();f.ctx->recover_completed_errors();
		expected.verify_image(*f.ctx,image,158,3,3,13,descriptor.format,true);gate.signaledValue=1;
		const auto fresh=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,fresh,157,3,3,13,descriptor.format,true);
	}
	{
		fixture f(surface_color::r32f,4,1,3,{},layout);fill_vm_holes(f,layout);morton_reference expected(f,layout);
		for(unsigned offset:{160u,164u})
		{
			auto old=f.display->prepare(descriptor,*f.surfaces);f.ctx->wait_idle();const auto original=expected;
			(void)add(f,expected,offset,offset==160?0x00123456:0x00789abc);
			const auto latest=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,latest,157,3,3,13,descriptor.format,true);
			original.verify_image(*f.ctx,old.finish(),157,3,3,13,descriptor.format,true);
			const auto created=f.budget->statistics().created,encoded=f.display->statistics().assembly_casts;const auto reads=f.ram->reads;
			require(f.display->prepare(descriptor,*f.surfaces).finish().texture==latest.texture&&f.budget->statistics().created==created&&f.display->statistics().assembly_casts==encoded&&f.ram->reads==reads,"Late completion replaced ready image after native owner appeared inVM hole");
		}
		auto ready=f.display->prepare(descriptor,*f.surfaces);const auto old=ready.finish();require(!expected.active[168],"Assembly mutation overlaps native source");f.ram->bytes[0x100+168]=std::byte{0x35};f.ram->notify({0x100+168,1});expected=morton_reference(f,layout);
		// Preserve the two independent native words in the reference after a VM-only mutation.
		for(auto [offset,value]:{std::pair{160u,0x00123456u},std::pair{164u,0x00789abcu}})for(unsigned byte=0;byte<4;++byte)expected.bytes[offset+byte]=std::byte(value>>(24-byte*8));
		rejects([&]{(void)ready.finish();});const auto fresh=f.display->prepare(descriptor,*f.surfaces).finish();require(fresh.texture!=old.texture,"Assembly reused oldVM payload");expected.verify_image(*f.ctx,fresh,157,3,3,13,descriptor.format,true);
	}
	{
		fixture f(surface_color::r32f,4,1,3,{},layout);fill_vm_holes(f,layout);morton_reference expected(f,layout);(void)add(f,expected,160,0x00445566);
		for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
		{
			const unsigned pitch=3*display_pixel_bytes(av)+1;const display_snapshot pair{0x100+157,3,3,pitch,av,0x100+159};
			auto job=f.display->prepare_frame(pair,*f.surfaces);const auto image=job.finish();require(image.right_eye.has_value(),"Assembly stereo lostright eye");
			expected.verify_image(*f.ctx,image,157,3,3,pitch,av,true);expected.verify_image(*f.ctx,{image.right_eye->texture,image.right_eye->region},159,3,3,pitch,av,true);
			const auto created=f.budget->statistics().created,encoded=f.display->statistics().assembly_casts;const auto reads=f.ram->reads;
			const auto warm=f.display->prepare_frame(pair,*f.surfaces).finish();require(warm.texture==image.texture&&warm.right_eye->texture==image.right_eye->texture&&f.budget->statistics().created==created&&f.display->statistics().assembly_casts==encoded&&f.ram->reads==reads,"Assembly stereo didnotreuse independent snapshots");
			auto left=f.display->prepare(display_eye(pair,0),*f.surfaces);const unsigned hole=157+3*display_pixel_bytes(av);require(!expected.active[hole],"Assembly stereo mutation isnotVM padding");
			f.ram->bytes[0x100+hole]=std::byte(55+unsigned(av));f.ram->notify({0x100+hole,1});expected.bytes[hole]=f.ram->bytes[0x100+hole];
			require(left.current(),"Assembly right-only holewrite invalidated left");rejects([&]{(void)job.finish();});
			const auto fresh=f.display->prepare_frame(pair,*f.surfaces).finish();require(fresh.texture==image.texture&&fresh.right_eye->texture!=image.right_eye->texture&&f.display->statistics().assembly_casts==encoded+1,"Assembly right-only update replacedleft orreusedoldright");
			expected.verify_image(*f.ctx,fresh,157,3,3,pitch,av,true);expected.verify_image(*f.ctx,{fresh.right_eye->texture,fresh.right_eye->region},159,3,3,pitch,av,true);
		}
	}
	@autoreleasepool
	{
		fixture f(surface_color::r32f,4,1,1,{},layout);fill_vm_holes(f,layout);morton_reference expected(f,layout);(void)add(f,expected,160,0x00445566);
		guest_texture_cache textures(*f.ctx,f.ram,[](unsigned o,unsigned,std::size_t){return o;},4096,f.budget);memory_pressure_owner pressure(*f.ctx,*f.surfaces,textures,f.budget,f.display.get());
		for(unsigned offset:{157u,158u,159u}){ @autoreleasepool{(void)f.display->prepare({0x100+offset,3,3,13},*f.surfaces).finish();} }
		(void)f.surfaces->reclaim_idle();const auto before=f.budget->statistics();const auto leave=std::min<std::size_t>(1024,f.display->statistics().bytes/8);
		id<MTLBuffer> blocker=f.budget->shared_buffer(before.budget-before.reserved_bytes-leave);const auto image=f.display->prepare({0x100+156,3,3,13},*f.surfaces).finish();
		require(f.budget->statistics().recovery_successes>before.recovery_successes&&f.display->statistics().evictions>0&&f.surfaces->status(f.source).dirty,"Assembly did not recover fullshared ledger");blocker=nil;expected.verify_image(*f.ctx,image,156,3,3,13,descriptor.format,true);
	}
	std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> provider;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan;std::optional<morton_reference> expected;id<MTLSharedEvent> gate;
		{
			auto f=std::make_unique<fixture>(surface_color::r32f,4,1,3,std::make_shared<tile_state>(),layout);fill_vm_holes(*f,layout);expected.emplace(*f,layout);(void)add(*f,*expected,160,0x00445566);provider=f->ram;ledger=f->budget;
			gate=[f->ctx->device() newSharedEvent];orphan.emplace(f->display->prepare(descriptor,*f->surfaces,std::array{display_upload_dependency{gate,1}}));require(!orphan->ready(),"Assembly orphan gate absent");
			f.reset();require(!provider.expired(),"Assembly lost receipt owners beforecompletion");
		}
		gate.signaledValue=1;const auto image=orphan->finish();context reader;expected->verify_image(reader,image,157,3,3,13,descriptor.format,true);orphan.reset();require(provider.expired(),"Assembly provider didnotretire");
	}
	require(ledger->statistics().reserved_bytes==0,"Assembly leaked native/VM descriptors/scratch/outputs");
	std::cout<<"Native assembly owners:pending wholework sharing/timeout,independent windows survive cancellation,new second/third native owners cannot be evicted bylate single/assembly completion,VM-hole writes refuse oldjobs,three warm stereo pairs/right-only updates,full-ledger recovery,gated cache/resolver/context destruction andexactpixels/provider retirement/zero oldcharges passed\n";
}
void morton_pending()
{
	const swizzle_region layout{{16,8,1},{},{16,6,1}};fixture f(surface_color::r32f,4,1,3,{},layout);morton_reference expected(f,layout);
	const display_snapshot first{0x101,3,3,13,display_format::xrgb8},other{0x111,3,3,13,display_format::xrgb8};
	const auto gate=[f.ctx->device() newSharedEvent],other_gate=[f.ctx->device() newSharedEvent];
	struct release{ id<MTLSharedEvent> event;~release(){event.signaledValue=1;} } unblock{gate},unblock_other{other_gate};
	auto pending=f.display->prepare(first,*f.surfaces,std::array{display_upload_dependency{gate,1}});
	require(!pending.ready(),"Morton prepare waited for gated GPU work");rejects([&]{(void)pending.finish(std::chrono::milliseconds(1));});
	auto alias=first;alias.offset+=0x10000;alias.format=display_format::xbgr8;
	const auto allocations=f.budget->statistics().created,encodes=f.display->statistics().native_casts;
	auto shared=f.display->prepare(alias,*f.surfaces);
	require(!shared.ready()&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encodes,"Compatible pending Morton alias did not share whole work");
	auto independent=f.display->prepare(other,*f.surfaces,std::array{display_upload_dependency{other_gate,1}});
	require(!independent.ready()&&f.display->statistics().native_casts==encodes+1,"Different pending Morton byte window borrowed incompatible output");
	require(pending.cancel_pending(),"Morton pending cast could not cancel");other_gate.signaledValue=1;
	rejects([&]{(void)shared.finish();});const auto independent_image=independent.finish();
	// The context deliberately retains completed errors until its caller
	// acknowledges recovery. Old immutable tokens must still report failure.
	f.ctx->recover_completed_errors();rejects([&]{(void)pending.finish();});expected.verify_image(*f.ctx,independent_image,17,3,3,13,other.format);
	gate.signaledValue=1;const auto ready=f.display->prepare(first,*f.surfaces).finish();expected.verify_image(*f.ctx,ready,1,3,3,13,first.format);
	const auto second=f.display->prepare(other,*f.surfaces).finish();require(second.texture!=ready.texture,"Different ready Morton windows aliased one AV output");expected.verify_image(*f.ctx,second,17,3,3,13,other.format);
	auto pitched=first;pitched.pitch=14;const auto changed_pitch=f.display->prepare(pitched,*f.surfaces).finish();require(changed_pitch.texture!=ready.texture,"Different ready Morton row pitch reused prior AV output");expected.verify_image(*f.ctx,changed_pitch,1,3,3,14,first.format);
	unsigned hole=0;while(hole<expected.active.size()&&expected.active[hole])++hole;require(hole>=4&&hole<expected.active.size(),"Morton fixture contains no inactive gap");
	const auto mixed=f.display->prepare({0x100+hole-4,2,1,8},*f.surfaces).finish();expected.verify_image(*f.ctx,mixed,hole-4,2,1,8,display_format::xrgb8,true);
	require(f.ram->reads>0&&!f.ram->writes,"Mixed Morton hole was not captured orwrote VM pixels");
	std::cout<<"Morton real gated aliases share whole copy/cast;different byte windows areindependent andsurvive first-copy cancellation;ready offset/pitch keys keep distinct pixels;crossing inactive Morton hole composes onlyowned VM fragments\n";
}
void morton_owners()
{
	const swizzle_region layout{{16,8,1},{},{16,6,1}};
	@autoreleasepool
	{
		fixture f(surface_color::r32f,4,1,1,{},layout);morton_reference expected(f,layout);
		guest_texture_cache textures(*f.ctx,f.ram,[](unsigned offset,unsigned,std::size_t){return offset;},4096,f.budget);
		memory_pressure_owner pressure(*f.ctx,*f.surfaces,textures,f.budget,f.display.get());
		for(unsigned offset:{1u,17u,33u}){ @autoreleasepool { (void)f.display->prepare({0x100+offset,3,3,13},*f.surfaces).finish(); } }
		require(f.display->statistics().entries==3,"Morton pressure fixture retained no converted outputs");
		(void)f.surfaces->reclaim_idle();const auto before=f.budget->statistics();const auto leave=std::min<std::size_t>(1024,f.display->statistics().bytes/8);
		id<MTLBuffer> blocker=f.budget->shared_buffer(before.budget-before.reserved_bytes-leave);
		const auto image=f.display->prepare({0x100+49,3,3,13},*f.surfaces).finish();
		require(f.budget->statistics().recovery_successes>before.recovery_successes&&f.display->statistics().evictions>0&&
			f.surfaces->status(f.source).dirty&&!f.ram->reads&&!f.ram->writes,"Morton scratch/output admission failed to reclaim ready images orpreserve native authority");
		blocker=nil;expected.verify_image(*f.ctx,image,49,3,3,13,display_format::xrgb8);
	}
	{
		auto tiles=std::make_shared<tile_state>();fixture f(surface_color::r32f,4,1,3,tiles,layout);morton_reference expected(f,layout);
		for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
		{
			f.display->clear();f.surfaces->clear_snapshots();
			const display_snapshot pair{0x101,1,1,9,av,0x111};
			auto job=f.display->prepare_frame(pair,*f.surfaces);const auto image=job.finish();require(image.right_eye.has_value(),"Morton stereo dropped right eye");
			expected.verify_image(*f.ctx,image,1,1,1,9,av);expected.verify_image(*f.ctx,{image.right_eye->texture,image.right_eye->region},17,1,1,9,av);
			const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;const auto encoders=f.ctx->encoding_statistics();
			const auto repeat=f.display->prepare_frame(pair,*f.surfaces).finish();
			require(repeat.texture==image.texture&&repeat.right_eye->texture==image.right_eye->texture&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded&&
				f.ctx->encoding_statistics().compute_encoders==encoders.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoders.render_encoders,"Equivalent Morton raw copies defeated paired warm output reuse");
			tiles->update(0,{0,0,0x10000,1024,8,static_cast<unsigned>(av),3,true});rejects([&]{(void)job.finish();});
			const auto fresh=f.display->prepare_frame(pair,*f.surfaces).finish();require(fresh.texture==image.texture&&fresh.right_eye->texture==image.right_eye->texture,"Morton tile receipt change replaced unchanged paired pixels");
		}
		for(bool writer:{false,true})
		{
			const display_snapshot descriptor{0x101,3,3,13,display_format::xrgb8};auto old=f.display->prepare(descriptor,*f.surfaces);f.ctx->wait_idle();const auto original=expected;
			if(writer)
			{
				for(unsigned i=0;i<f.pixels.size();i+=4){const std::uint32_t word=0x3f000000+((i/4+(i/64)*3)%8)*0x00100000;for(unsigned c=0;c<4;++c)f.pixels[i+c]=std::byte(word>>(24-c*8));}f.write();expected=morton_reference(f,layout);
			}
			else f.surfaces->clear_snapshots();
			const auto latest=f.display->prepare(descriptor,*f.surfaces).finish();expected.verify_image(*f.ctx,latest,1,3,3,13,descriptor.format);
			original.verify_image(*f.ctx,old.finish(),1,3,3,13,descriptor.format);
			const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;
			require(f.display->prepare(descriptor,*f.surfaces).finish().texture==latest.texture&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded,"Late Morton completion replaced newer native/cache generation");
		}
		require(!f.ram->reads&&!f.ram->writes,"Morton paired/generation paths read/wrote VM pixels");
	}
	std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> provider;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan;std::optional<morton_reference> expected;id<MTLSharedEvent> gate;
		{
			auto tiles=std::make_shared<tile_state>();auto f=std::make_unique<fixture>(surface_color::r32f,4,1,3,tiles,layout);ledger=f->budget;provider=f->ram;expected.emplace(*f,layout);
			gate=[f->ctx->device() newSharedEvent];orphan.emplace(f->display->prepare({0x101,3,3,13},*f->surfaces,std::array{display_upload_dependency{gate,1}}));
			require(!orphan->ready(),"Morton orphan gate did not retain live GPU work");rejects([&]{(void)orphan->finish(std::chrono::milliseconds(1));});
			f.reset();require(!provider.expired(),"Morton orphan lost owned provider before completion");
		}
		gate.signaledValue=1;const auto image=orphan->finish();context reader;expected->verify_image(reader,image,1,3,3,13,display_format::xrgb8);
		orphan.reset();require(provider.expired(),"Morton orphan retained provider after last job release");
	}
	require(ledger->statistics().reserved_bytes==0,"Morton orphan retained old GPU allocations after final release");
	std::cout<<"Morton owners:full-ledger output/scratch recovery,three stereo AV pairs/warm views/strict tile receipts,late frozen jobs versusnew writer/cleared cache,real gated pack-swizzle-decode acrossresolver/cache/context destruction,exact pixels/provider retirement/zero old ledger charges\n";
}
void local_tiles()
{
	auto tiles=std::make_shared<tile_state>();fixture f(surface_color::r32f,4,1,3,tiles);
	std::array<id<MTLTexture>,3> ready{};unsigned cases=0;
	for(unsigned comp:{0u,7u,8u,9u,10u,11u,12u})for(unsigned bank=0;bank<4;++bank)for(unsigned pitch:{256u,1024u})
	{
		std::array<display_upload_job,3> old{
			f.display->prepare({0x100+f.pitch+1,3,3,f.pitch,display_format::xrgb8},*f.surfaces),
			f.display->prepare({0x100+f.pitch+1,3,3,f.pitch,display_format::xbgr8},*f.surfaces),
			f.display->prepare({0x100+f.pitch+1,3,3,f.pitch,display_format::rgbx16f},*f.surfaces)};
		for(auto& job:old)(void)job.finish();
		tiles->update(0,{0,0,0x10000,pitch,comp,cases,bank,true});
		unsigned index=0;
		for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
		{
			require(!old[index].current(),"Local tile change did not invalidate old AV receipt");rejects([&]{(void)old[index].finish();});
			const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;
			const auto image=f.display->prepare({0x100+f.pitch+1,3,3,f.pitch,av},*f.surfaces).finish();
			require(f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded,"Local metadata change forced native allocation/recast");
			verify(*f.ctx,image,f.pixels,1,3,av);if(cases)require(image.texture==ready[index],"Transparent local tags replaced unchanged native data");ready[index++]=image.texture;
		}
		++cases;
	}
	const display_snapshot pair{0x100+f.pitch+1,3,1,f.pitch,display_format::xrgb8,0x100+f.pitch*4+1};
	auto old_pair=f.display->prepare_frame(pair,*f.surfaces);(void)old_pair.finish();tiles->update(0,{});
	rejects([&]{(void)old_pair.finish();});const auto fresh_pair=f.display->prepare_frame(pair,*f.surfaces).finish();
	verify(*f.ctx,fresh_pair,f.pixels,1,3,pair.format,1,1);verify(*f.ctx,{fresh_pair.right_eye->texture,fresh_pair.right_eye->region},f.pixels,1,3,pair.format,1,4);
	auto right_alias=pair;right_alias.right_offset=*pair.right_offset+0x10000;
	auto right_old=f.display->prepare_frame(right_alias,*f.surfaces);(void)right_old.finish();
	auto left_only=f.display->prepare(display_eye(right_alias,0),*f.surfaces);(void)left_only.finish();
	tiles->update(2,{0,0x10000,0x10000,1024,12,0,2,true});
	require(left_only.current(),"Right-only local tile change invalidated unrelated left receipt");rejects([&]{(void)right_old.finish();});
	const auto right_fresh=f.display->prepare_frame(right_alias,*f.surfaces).finish();
	verify(*f.ctx,right_fresh,f.pixels,1,3,pair.format,1,1);verify(*f.ctx,{right_fresh.right_eye->texture,right_fresh.right_eye->region},f.pixels,1,3,pair.format,1,4);
	// Pending copies retain their actual GPU work but expose only fresh AV receipts.
	f.display->clear();f.surfaces->clear_snapshots();const auto gate=[f.ctx->device() newSharedEvent];
	struct release{ id<MTLSharedEvent> event;~release(){event.signaledValue=1;} } unblock{gate};
	const display_snapshot descriptor{0x100+f.pitch+1,3,3,f.pitch,display_format::xrgb8};
	auto pending=f.display->prepare(descriptor,*f.surfaces,std::array{display_upload_dependency{gate,1}});require(!pending.ready(),"Local tile test did not gate GPU copy");
	const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;
	tiles->update(0,{0,0,0x10000,256,8,0,3,true});tiles->update(0,{});
	auto fresh=f.display->prepare(descriptor,*f.surfaces);require(!fresh.ready()&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded,"Fresh AV receipt did not share unchanged pending native work");
	gate.signaledValue=1;rejects([&]{(void)pending.finish();});verify(*f.ctx,fresh.finish(),f.pixels,1,3,descriptor.format);
	tiles->update(1,{1,0,0x10000,256,0,0,0,true});require(fresh.current(),"Main tile change invalidated local AV receipt");
	require(!f.ram->reads&&!f.ram->writes&&f.surfaces->status(f.source).dirty,"Local transparent tiles touched VM pixels orlost dirty authority");
	for(auto byte:f.ram->bytes)require(byte==std::byte{0xa7},"Local transparent tiles changed guest pixels/padding");
	std::cout<<"Local tiles:56 compression/bank/pitch descriptors x3 AV formats fromactual mapped native4sample framebuffer;old AV receipts reject,new views reuse withoutallocation/recast,stereo/unbind/right-only alias,pending gated bind-unbind ABA/main isolation;zero VM pixel reads/writes\n";
}
void local_vm_tiles()
{
	auto tiles=std::make_shared<tile_state>();fixture f(surface_color::r32f,1,1,3,tiles);
	const cpu_capture_dispatch capture=[tiles](const auto& work){tiles->execute(work);};
	surface_cache empty(*f.ctx,f.ram,32u<<20,f.budget,capture);
	display_resolver display(*f.ctx,f.ram,[](unsigned offset,unsigned,std::size_t){return offset;},f.budget,4096,capture,tiles);
	for(unsigned y=0;y<6;++y)std::copy_n(f.pixels.begin()+y*64,64,f.ram->bytes.begin()+0x100+y*f.pitch);
	const auto original=f.ram->bytes;unsigned cases=0;
	for(unsigned comp:{0u,7u,8u,9u,10u,11u,12u})for(unsigned bank=0;bank<4;++bank)for(unsigned pitch:{256u,1024u})
	{
		tiles->update(0,{0,0,0x10000,pitch,comp,cases,bank,true});
		for(auto av:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
		{
			const display_snapshot descriptor{0x100+f.pitch+1,3,3,f.pitch,av};
			const auto image=display.prepare(descriptor,empty).finish();verify(*f.ctx,image,f.pixels,1,3,av);
			const auto reads=f.ram->reads;const auto allocations=f.budget->statistics().created;const auto encoders=f.ctx->encoding_statistics();
			require(display.prepare(descriptor,empty).finish().texture==image.texture&&f.ram->reads==reads&&f.budget->statistics().created==allocations&&
				f.ctx->encoding_statistics().compute_encoders==encoders.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoders.render_encoders,"Local VM display warm hit recaptured/allocated/encoded");
		}
		++cases;
	}
	require(f.ram->reads>0&&!f.ram->writes&&f.ram->bytes==original,"Local VM upload wrote bytes/padding or bypassed owned capture");
	std::cout<<"Local VM tiles:56 compression/bank/pitch descriptors x3 AV formats/odd pitch/partial source byte;owned VM capture withnativeGPUdecode,exact channels,stable warm hits,no writeback andunchanged original pixels/padding\n";
}
void late_publication()
{
	for(bool writer:{false,true})
	{
		fixture f(surface_color::r32f,4,1);
		const display_snapshot image{0x100+f.pitch+1,3,3,f.pitch,display_format::xrgb8};
		auto old=f.display->prepare(image,*f.surfaces);f.ctx->wait_idle();const auto original=f.pixels;
		if(writer)
		{
			for(unsigned i=0;i<f.pixels.size();i+=4){const std::uint32_t value=0x3f000000+((i/4+(i/64)*3)%8)*0x00100000;for(unsigned c=0;c<4;++c)f.pixels[i+c]=std::byte(value>>(24-c*8));}
			f.write();
		}
		else f.surfaces->clear_snapshots();
		const auto latest=f.display->prepare(image,*f.surfaces).finish();verify(*f.ctx,latest,f.pixels,1,3,image.format);
		verify(*f.ctx,old.finish(),original,1,3,image.format);
		const auto allocations=f.budget->statistics().created,encoded=f.display->statistics().native_casts;
		require(f.display->prepare(image,*f.surfaces).finish().texture==latest.texture&&f.budget->statistics().created==allocations&&f.display->statistics().native_casts==encoded,"Late old cast evicted or masked the newer confirmed content key");
		require(!f.ram->reads&&!f.ram->writes,"Native generation test touched guest pixels");
	}
	std::cout<<"Native cast generation:late old completion keeps frozen pixels butcannot replace newer writer/cleared-snapshot ready key;fresh warm output incurs no allocation/recast;VM unchanged\n";
}
void paired_casts()
{
	fixture f(surface_color::r32f,4,1);
	for(auto format:{display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
	{
		const display_snapshot snapshot{0x100+f.pitch+1,3,1,f.pitch,format,0x100+f.pitch*4+1};
		const auto image=f.display->prepare_frame(snapshot,*f.surfaces).finish();require(image.right_eye.has_value(),"Paired cast dropped the right eye");
		verify(*f.ctx,image,f.pixels,1,3,format,1,1);
		const present_image right{image.right_eye->texture,image.right_eye->region};verify(*f.ctx,right,f.pixels,1,3,format,1,4);
		const auto allocations=f.budget->statistics().created;const auto encoders=f.ctx->encoding_statistics();
		const auto repeat=f.display->prepare_frame(snapshot,*f.surfaces).finish();
		require(repeat.texture==image.texture&&repeat.right_eye->texture==image.right_eye->texture&&f.budget->statistics().created==allocations&&
			f.ctx->encoding_statistics().compute_encoders==encoders.compute_encoders&&f.ctx->encoding_statistics().render_encoders==encoders.render_encoders,"Paired cast warm hit changed eye storage or encoded work");
	}
	const display_snapshot descriptor{0x100+f.pitch+1,3,3,f.pitch};auto settled=f.display->prepare(descriptor,*f.surfaces);(void)settled.finish();
	bool refused=false;
	rejects([&]{(void)f.surfaces->snapshot_texture_region(display_coverage(descriptor,descriptor.offset),[&](const auto&)->std::optional<clear_rectangle>
	{
		try{(void)settled.finish();}catch(const std::logic_error&){refused=true;}return {};
	},f.budget);});
	require(refused&&!f.ram->reads&&!f.ram->writes,"Compact completed owner lost legacy CPU-epoch guard or paired casts touched VM pixels");
	std::cout<<"Three paired AV casts keep distinct row/sample content and opaque left/right views with no warm allocations/encoding;compact ready owner retains legacy CPU-capture guard\n";
}
void pressure_and_orphan()
{
	{
		fixture f(surface_color::r32f,4,1,1);
		guest_texture_cache textures(*f.ctx,f.ram,[](unsigned offset,unsigned,std::size_t){return offset;},4096,f.budget);
		memory_pressure_owner pressure(*f.ctx,*f.surfaces,textures,f.budget,f.display.get());
		for(unsigned x:{1u,5u,9u}){ @autoreleasepool { (void)f.display->prepare({0x100+f.pitch+x,3,3,f.pitch},*f.surfaces).finish(); } }
		require(f.display->statistics().entries==3,"Pressure fixture retained no converted ready images");
		(void)f.surfaces->reclaim_idle();const auto before=f.budget->statistics();
		const auto leave=std::min<std::size_t>(1024,f.display->statistics().bytes/8);
		id<MTLBuffer> blocker=f.budget->shared_buffer(before.budget-before.reserved_bytes-leave);
		const auto image=f.display->prepare({0x100+f.pitch+13,3,3,f.pitch},*f.surfaces).finish();
		require(f.budget->statistics().recovery_successes>before.recovery_successes&&f.display->statistics().evictions>0&&
			f.surfaces->status(f.source).dirty&&!f.ram->reads&&!f.ram->writes,"Full ledger failed to reclaim converted ready owners while preserving native dirty source");
		blocker=nil;verify(*f.ctx,image,f.pixels,13,3,display_format::xrgb8);
	}
	std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> memory_owner;
	@autoreleasepool
	{
		std::optional<display_upload_job> orphan;std::vector<std::byte> original;
		id<MTLSharedEvent> gate;
		{
			auto f=std::make_unique<fixture>(surface_color::r32f,4,1);ledger=f->budget;memory_owner=f->ram;original=f->pixels;
			gate=[f->ctx->device() newSharedEvent];
			orphan.emplace(f->display->prepare({0x100+f->pitch+1,3,3,f->pitch},*f->surfaces,std::array{display_upload_dependency{gate,1}}));
			require(!orphan->ready(),"Cast orphan gate did not hold actual GPU work");rejects([&]{(void)orphan->finish(std::chrono::milliseconds(1));});
			f.reset();require(!memory_owner.expired(),"Pending cast lost its receipt provider after cache/context destruction");
		}
		gate.signaledValue=1;const auto image=orphan->finish();context reader;verify(reader,image,original,1,3,display_format::xrgb8);
		orphan.reset();require(memory_owner.expired(),"Completed cast retained receipt provider after final job release");
	}
	require(ledger->statistics().reserved_bytes==0,"Cast completion/owner retirement leaked native ledger charges");
	std::cout<<"Converted display full-ledger recovery releases weak lookup owners while retaining dirty authority;real gated copy+cast survives resolver/surface/context destruction,exact pixels/provider retirement/all old native charges zero passed\n";
}
void pending_schemas()
{
	fixture f(surface_color::r32f,4,1);
	const auto gate=[f.ctx->device() newSharedEvent];struct release{ id<MTLSharedEvent> event;~release(){event.signaledValue=1;} } guard{gate};
	const display_snapshot image{0x100+f.pitch+1,4,3,f.pitch,display_format::xrgb8};
	auto first=f.display->prepare(image,*f.surfaces,std::array{display_upload_dependency{gate,1}});
	require(!first.ready(),"Native cast prepare waited for its actual event gate");rejects([&]{(void)first.finish(std::chrono::milliseconds(1));});
	const auto allocations=f.budget->statistics().created;auto reverse=image;reverse.format=display_format::xbgr8;
	auto alias=image;alias.offset+=0x10000;
	auto duplicate=f.display->prepare(reverse,*f.surfaces);auto physical=f.display->prepare(alias,*f.surfaces);
	require(!duplicate.ready()&&!physical.ready()&&f.display->statistics().native_casts==1&&f.budget->statistics().created==allocations,"Pending compatible cast/physical aliases did not share whole output work");
	gate.signaledValue=1;const auto frozen=first.finish();verify(*f.ctx,frozen,f.pixels,1,4,image.format);
	verify(*f.ctx,duplicate.finish(),f.pixels,1,4,reverse.format);require(physical.finish().texture==frozen.texture,"Physical alias did not reuse opaque native cast view");
	f.display->clear();f.surfaces->clear_snapshots();
	const auto blocked=[f.ctx->device() newSharedEvent],second_gate=[f.ctx->device() newSharedEvent];release unblock{blocked},unblock_second{second_gate};
	auto cancelled=f.display->prepare(image,*f.surfaces,std::array{display_upload_dependency{blocked,1}});
	auto different=image;different.width=2;different.format=display_format::rgbx16f;
	auto independent=f.display->prepare(different,*f.surfaces,std::array{display_upload_dependency{second_gate,1}});
	require(!cancelled.ready()&&!independent.ready()&&f.display->statistics().native_casts==3,"Different pending output schema borrowed incompatible work");
	require(cancelled.cancel_pending(),"Gated native cast could not cancel before commit");second_gate.signaledValue=1;rejects([&]{(void)cancelled.finish();});
	verify(*f.ctx,independent.finish(),f.pixels,1,2,different.format);
	require(!f.ram->reads&&!f.ram->writes,"Gated native cast fell back to guest pixels");
	std::cout<<"Real gated native cast:prepare returns,timeout retains owners,AV/physical duplicates share whole work;different schema takes independent snapshot and survives first-copy cancellation\n";
}
}
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool{try{require(NSApp==nil,"Test began with App");matrix();morton_matrix();morton_mixed_matrix();morton_mixed_lifetime();native_assembly_matrix();native_assembly_lifetime();morton_pending();morton_owners();local_tiles();local_vm_tiles();pending_schemas();late_publication();paired_casts();pressure_and_orphan();require(NSApp==nil,"Display cast test created App");std::cout<<channels<<" total channel checks;"<<guards<<" guards passed;no App/window/drawable/GSRender/guest VM\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
}
