#include "M4GuestTexture.h"
#include "M4ProgramSamplerPlan.h"
#include "M4RenderPipeline.h"
#include "M4SurfaceCache.h"
#include "M4SurfacePool.h"
#include "M4SurfaceTextureRegion.h"
#include "M4ExclusiveGuestMemory.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>

namespace
{
using namespace metal4;
using color = std::array<float, 4>;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
class memory final : public guest_surface_memory
{
  public:
	std::vector<std::byte> bytes;
	std::uint64_t version = 1;
	bool conflict = false;
	bool allow_writeback = false;
	bool forbid_vm_access = false;
	bool short_read = false;
	unsigned reads = 0;
	std::vector<guest_surface_range> forbidden;
	std::vector<guest_storage_segment> aliases;
	std::shared_ptr<int> allocation=std::make_shared<int>(0);
	std::size_t physical_offset(std::uint32_t address) const
	{
		if(address>=0x1000&&address-0x1000<bytes.size())return address-0x1000;
		for(const auto& alias:aliases)if(address>=alias.address&&std::uint64_t(address)-alias.address<alias.size)return alias.offset+address-alias.address;
		throw std::out_of_range("Test physical identity lacks its actual storage");
	}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		if(aliases.empty())return {};
		std::vector<storage_identity_span> result;
		for(std::size_t i=0;i<range.size;++i)
		{
			const auto ea=static_cast<std::uint32_t>(std::uint64_t(range.address)+i);const auto offset=physical_offset(ea);require(offset<bytes.size(),"Test alias exceeds its owned allocation");
			if(!result.empty()&&result.back().offset+result.back().guest.size==offset)++result.back().guest.size;else result.push_back({allocation,offset,{ea,1}});
		}
		return result;
	}
	bool test_storage(const storage_test& test) override
	{
		if(aliases.empty())return guest_surface_memory::test_storage(test);
		return test(false,[&](auto a,auto b){const auto left_parts=storage_identity(a),right_parts=storage_identity(b);for(const auto& left:*left_parts)for(const auto& right:*right_parts)if(left.offset<right.offset+right.guest.size&&right.offset<left.offset+left.guest.size)return true;return false;});
	}
	bool test_rebase(const rebase_test& test) override
	{
		if(aliases.empty())return guest_surface_memory::test_rebase(test);
		const storage_rebase rebase=[&](auto source,auto request)->std::optional<guest_surface_range>{for(std::size_t i=0;i<source.size;++i)if(physical_offset(source.address+i)==physical_offset(request.address)){if(request.size>source.size-i)return {};bool same=true;for(std::size_t j=0;j<request.size;++j)same&=physical_offset(source.address+i+j)==physical_offset(request.address+j);if(same)return guest_surface_range{static_cast<std::uint32_t>(source.address+i),request.size};}return {};};
		const storage_equivalent equal=[&](auto a,auto b){if(a.size!=b.size)return false;for(std::size_t i=0;i<a.size;++i)if(physical_offset(a.address+i)!=physical_offset(b.address+i))return false;return true;};return test(rebase,equal);
	}
	guest_surface_layout watch_surface(const guest_surface_layout& coverage) override
	{
		require(!forbid_vm_access, "Owned texture realization watched VM");
		return coverage;
	}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		require(!forbid_vm_access, "Owned texture realization read VM");
		++reads;
		if(!aliases.empty())
		{
			std::vector<std::byte> output;for(std::size_t i=0;i<coverage.range.size;++i){const auto offset=physical_offset(static_cast<std::uint32_t>(std::uint64_t(coverage.range.address)+i));for(const auto forbidden_range:forbidden){const auto identity=storage_identity(forbidden_range);for(const auto& span:*identity)require(offset<span.offset||offset>=span.offset+span.guest.size,"RTT alias captured native physical pixels");}output.push_back(bytes.at(offset));}return {version,std::move(output)};
		}
		require(coverage.range.address>=0x1000&&coverage.range.address-0x1000<=bytes.size()&&coverage.range.size<=bytes.size()-(coverage.range.address-0x1000), "Texture VM read has wrong translated range");
		for(auto range:forbidden)if(overlaps(range,coverage.range))throw std::runtime_error("RTT assembly captured GPU-authoritative pixels at"+std::to_string(coverage.range.address)+":"+std::to_string(coverage.range.size));
		const auto offset=coverage.range.address-0x1000;return {version, {bytes.begin()+offset, bytes.begin()+offset+coverage.range.size-unsigned(short_read)}};
	}
	std::uint64_t surface_revision(const guest_surface_layout&) override
	{
		require(!forbid_vm_access, "Owned texture realization checked VM revision");
		if (conflict)
			++version;
		return version;
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage, std::uint64_t expected, std::span<const std::byte> input) override
	{
		if (!allow_writeback)
			throw std::runtime_error("Read-only texture resolver wrote VM");
		require(coverage.range.address == 0x1000 && coverage.range.size <= bytes.size() && input.size() == coverage.range.size,
				"Surface writeback has wrong translated range");
		if (version != expected)
			return {};
		guest_surface_byte_cursor cursor(coverage);
		while (const auto range = cursor.next())
		{
			const auto offset = range->address - coverage.range.address;
			std::copy_n(input.begin() + offset, range->size, bytes.begin() + offset);
		}
		return ++version;
	}
	void write(guest_surface_range, std::span<const std::byte>) override { throw std::runtime_error("Read-only texture resolver wrote VM"); }
};
void append(std::vector<std::byte>& bytes, unsigned word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		bytes.push_back(std::byte(word >> shift));
}
std::vector<std::byte> code(color coordinate, unsigned opcode = 23, float lod = 0)
{
	std::vector<std::byte> result;
	unsigned reg = 1;
	for (const auto literal : {coordinate, color{lod, 0, 0, 0}})
	{
		for (unsigned word : {(1u << 24) | (reg << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(result, word);
		for (float v : literal)
			append(result, std::bit_cast<unsigned>(v));
		++reg;
	}
	for (unsigned word : {(opcode << 24) | (15u << 9) | 1u, (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 4u, (0xe4u << 9) | 8u, 0xe4u << 9})
		append(result, word);
	return result;
}
struct fixture
{
	context ctx{1};
	compiler compile{ctx};
	sampler_cache samplers{ctx};
	std::shared_ptr<memory> ram = std::make_shared<memory>();
	unsigned translated = 0, cases = 0;
	std::shared_ptr<heap_budget> upload_budget = std::make_shared<heap_budget>(ctx, 64 << 20);
	guest_texture_resolver resolver{ctx, ram, [&](unsigned offset, unsigned location, std::size_t size)
									{ require(!ram->forbid_vm_access,"Owned texture realization translated VM");require(offset==0x200 && location==1 && size<=ram->bytes.size(),"Frozen address translation lost offset/location/size");++translated;return 0x1000; }, 1 << 20, upload_budget};
	id<MTLLibrary> vertex;
	id<MTLTexture> target;
	id<MTLBuffer> output;
	converted_geometry geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	fixture()
	{
		vertex = compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
vertex float4 v(uint i [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1); }
)metal",
								 @"Guest texture upload regression vertex");
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:2 height:2 mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageRenderTarget;
		target = make_texture(ctx, td);
		output = make_buffer(ctx, 512, buffer_storage::shared);
	}
	void sample(resolved_fragment_texture resource, const fragment_texture_registers& words, color coordinate, color expected, unsigned opcode = 23, float lod = 0)
	{
		const auto program_metadata = program_sampler_plan::capture(words, resource.metadata);
		const auto storage_plan = plan_guest_texture(words);
		const unsigned source_depth = storage_plan.codec.depth ? static_cast<unsigned>(*storage_plan.codec.depth) : 0;
		constexpr unsigned expected_class[]{1, 2, 8, 4, 16};
		require(program_metadata.format_class == expected_class[source_depth] &&
			program_metadata.samples == 1 && program_metadata.coordinates.scale == resource.metadata.coordinates.scale,
			"Native VM/alias snapshot lost source depth class or base coordinate transform");
		const auto decoded = decode_fragment_texture_state(words, resource.metadata);
		fragment_program_options options;
		configure_fragment_texture_state(options, 0, decoded);
		const auto program = code(coordinate, opcode, lod);
		fragment_shader shader(compile, program, options);
		render_pipeline_configuration config;
		config.colors[0] = MTLPixelFormatRGBA32Float;
		const auto pipeline = make_render_pipeline(compile, config, library_function(vertex, @"v"), shader);
		fragment_inputs inputs;
		inputs.set_texture_state(0, resource.texture, samplers, decoded);
		ctx.begin();
		const auto bindings = shader.prepare(ctx, {}, program, inputs);
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = target;
		pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		render_commands draw(ctx, pass);
		raster_state raster;
		raster.viewport = {0, 0, 2, 2, 0, 1};
		raster.scissor = {0, 0, 2, 2};
		draw.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, {}, bindings);
		draw.end();
		texture_region region;
		region.size = MTLSizeMake(2, 2, 1);
		texture_to_buffer(ctx, target, region, output, 0, 256);
		ctx.submit().wait();
		for (unsigned y = 0; y < 2; ++y)
			for (unsigned x = 0; x < 2; ++x)
				for (unsigned c = 0; c < 4; ++c)
				{
					const float actual = static_cast<const float*>(output.contents)[y * 64 + x * 4 + c];
					if (!std::isfinite(actual) || std::abs(actual - expected[c]) > 2e-6f)
						throw std::runtime_error("Guest texture format=" + std::to_string((words.format >> 8) & 255) + " lane=" + std::to_string(c) + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected[c]));
				}
		++cases;
	}
	static fragment_texture_registers words(unsigned format, vertex_texture_dimension dimension = vertex_texture_dimension::two, bool linear = true, unsigned width = 4, unsigned height = 4, unsigned depth = 1, unsigned mips = 1)
	{
		fragment_texture_registers r;
		r.offset = 0x80000200;
		const unsigned dim = dimension == vertex_texture_dimension::one ? 1 : dimension == vertex_texture_dimension::three ? 3
																														   : 2;
		r.format = (mips << 16) | ((format | (linear ? 0x20 : 0)) << 8) | (dim << 4) | 8 | 2 | (dimension == vertex_texture_dimension::cube ? 4 : 0);
		r.address = (7u << 28) | 0x030303;
		r.control0 = (1u << 31) | (4095u << 7);
		r.control1 = 0xaae4;
		r.filter = (1u << 24) | (3u << 16);
		r.image_rect = (width << 16) | height;
		r.control3 = (depth << 20);
		const auto codec = describe_guest_texture_format(format);
		if (linear)
			r.control3 |= ((width + codec.block.width - 1) / codec.block.width) * codec.block.bytes + codec.block.bytes * 2;
		return r;
	}
	void fill(const guest_texture_layout& plan, const std::vector<unsigned>& block)
	{
		ram->bytes.assign(plan.storage_bytes, std::byte{0xcd});
		++ram->version;
		for (const auto& s : plan.subresources)
			for (std::size_t z = 0; z < s.depth; ++z)
				for (std::size_t y = 0; y < s.rows; ++y)
					for (std::size_t x = 0; x < s.columns; ++x)
					{
						std::size_t offset = s.offset + (z * s.rows + y) * s.row_pitch + x * plan.codec.block.bytes;
						if (plan.swizzled && plan.codec.block.width == 1)
							offset = s.offset + morton_index({x, y, z}, {unsigned(std::bit_width(s.columns) - 1), unsigned(std::bit_width(s.rows) - 1), unsigned(std::bit_width(s.depth) - 1)}) * plan.codec.block.bytes;
						for (unsigned b = 0; b < block.size(); ++b)
							ram->bytes[offset + b] = std::byte(block[b]);
					}
	}
	captured_guest_texture capture_without_gpu(const fragment_texture_registers& words)
	{
		const auto bytes = upload_budget->statistics().reserved_bytes;
		const auto reads = ram->reads;
		// An active transaction makes any hidden begin/submit fail. Capture
		// must leave that transaction and native heap usage untouched.
		ctx.begin();
		auto captured = resolver.capture(words);
		ctx.cancel();
		require(upload_budget->statistics().reserved_bytes == bytes && ram->reads == reads + 1,
			"CPU texture capture encoded/allocated GPU storage or reread VM");
		require(captured.storage_bytes() == plan_guest_texture(words).storage_bytes, "Owned capture lost storage bytes");
		return captured;
	}
	resolved_fragment_texture realize_without_vm(const captured_guest_texture& captured)
	{
		const auto reads = ram->reads;
		const auto addresses = translated;
		ram->bytes.clear();
		++ram->version;
		ram->forbid_vm_access = true;
		const auto result = resolver.realize(captured);
		ram->forbid_vm_access = false;
		require(ram->reads == reads && translated == addresses, "Owned texture realization used VM");
		return result;
	}
	void depth_volume_cases()
	{
		unsigned profiles=0,planes=0,draws=0,remaps=0,borders=0,filters=0;std::size_t components=0;
		for(unsigned format:{0x90u,0x91u,0x92u,0x93u})for(const auto shape:std::array<std::array<unsigned,3>,3>{{{8,4,4},{4,4,8},{7,5,3}}})for(unsigned layout=0;layout<3;++layout)
		{
			if(!layout&&shape[0]==7)continue;
			@autoreleasepool
			{
				auto r=words(format,vertex_texture_dimension::three,layout!=0,shape[0],shape[1],shape[2],4);const unsigned bb=describe_guest_texture_format(format).block.bytes;if(layout==2)r.control3=(r.control3&0xfff00000)|(shape[0]*bb+1);const auto plan=plan_guest_texture(r);auto source=std::make_shared<memory>();source->bytes.assign(plan.storage_bytes,std::byte{0xcd});
				const auto raw=[&](unsigned level,unsigned x,unsigned y,unsigned z){const unsigned i=x+y*13+z*71+level*397;if(format==0x92)return (0x1234u+i*97)&65535;if(format==0x93)return 0x6000u+i%0x1800;const unsigned d=format==0x90?0x123456u+i*397:std::bit_cast<unsigned>(.125f+level/32.f+z/64.f+x/512.f+y/256.f)>>7;return (d<<8)|((i*17+39)&255);};
				const auto depth=[&](unsigned level,unsigned x,unsigned y,unsigned z){const auto word=raw(level,x,y,z);return format==0x92?word/65535.f:format==0x90?float(word>>8)/16777215.f:std::bit_cast<float>(format==0x91?(word>>8)<<7:(word<<11)+(120u<<23));};
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x){const auto at=s.offset+(layout?(z*s.storage_rows+y)*s.row_pitch+x*bb:morton_index({x,y,z},plan_swizzle({{s.columns,s.rows,s.depth},{},{s.columns,s.rows,s.depth}},bb).log2)*bb);const auto v=raw(s.level,x,y,z);for(unsigned c=0;c<bb;++c)source->bytes[at+c]=std::byte(v>>((bb-1-c)*8));}
				const auto original=source->bytes;context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);guest_texture_resolver uploads(gpu,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);const auto result=uploads.resolve(r);require(result.texture.textureType==MTLTextureType3D&&result.texture.pixelFormat==MTLPixelFormatRG32Float&&result.metadata.depth&&result.metadata.source_depth_format==unsigned(*plan.codec.depth),"Depth3D lost typed volume/depth metadata");const auto statistics=uploads.upload_statistics();require(statistics.staging_buffers==1&&statistics.depth_volume_subresources==plan.mips&&statistics.conversion_dispatches==plan.mips&&source->reads==1&&source->bytes==original,"Depth3D upload recaptured/wroteVM orused per-slice scratch");source->forbid_vm_access=true;source->bytes.clear();++source->version;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)
				{
					const auto output=make_buffer(ctx,s.width*s.height*8,buffer_storage::shared);texture_region area;area.level=s.level;area.origin=MTLOriginMake(0,0,z);area.size=MTLSizeMake(s.width,s.height,1);ctx.begin();texture_to_buffer(ctx,result.texture,area,output,0,s.width*8);ctx.submit().wait();const auto* bits=static_cast<const unsigned*>(output.contents);for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x){const unsigned at=(y*s.width+x)*2;require(bits[at]==std::bit_cast<unsigned>(depth(s.level,x,y,z)),"Depth3D layout/codec/rawfloat precision differs");require(bits[at+1]==std::bit_cast<unsigned>(bb==4?float(raw(s.level,x,y,z)&255):0.f),"Depth3D independent stencil changed");components+=2;}++planes;
					const unsigned x=s.width/2,y=s.height/2;const auto d=depth(s.level,x,y,z);sample(result,r,{(x+.5f)/s.width,(y+.5f)/s.height,(z+.5f)/s.depth,1},{d,d,d,d},47,s.level);++draws;
				}
				const auto& base=plan.subresources[0];const auto d=depth(0,base.width/2,base.height/2,base.depth/2);auto remap=r;remap.control1=0x52e4;sample(result,remap,{(base.width/2+.5f)/base.width,(base.height/2+.5f)/base.height,(base.depth/2+.5f)/base.depth,1},{0,1,1,d},47,0);++remaps;
				auto border=r;border.address=(7u<<28)|0x040404;border.border_color=0x205080c0;sample(result,border,{-.25f,.5f,.5f,1},{80.f/255,128.f/255,192.f/255,32.f/255},47,0);++borders;
				const auto interpolate=[&](unsigned level){const auto& s=plan.subresources[level];const std::array<std::size_t,3> size{s.width,s.height,s.depth};const std::array<float,3> coordinate{.375f,.625f,.5f};std::array<int,3> low;std::array<float,3> fraction;for(unsigned axis=0;axis<3;++axis){const float at=coordinate[axis]*size[axis]-.5f;low[axis]=int(std::floor(at));fraction[axis]=at-low[axis];}float value=0;for(unsigned tap=0;tap<8;++tap){std::array<unsigned,3> at;float weight=1;for(unsigned axis=0;axis<3;++axis){const unsigned high=(tap>>axis)&1;at[axis]=std::clamp(low[axis]+int(high),0,int(size[axis])-1);weight*=high?fraction[axis]:1-fraction[axis];}value+=weight*depth(level,at[0],at[1],at[2]);}return value;};auto filtered=r;filtered.filter=(2u<<24)|(6u<<16);const float value=plan.mips>1?(interpolate(0)+interpolate(1))*.5f:interpolate(0);sample(result,filtered,{.375f,.625f,.5f,1},{value,value,value,value},47,plan.mips>1?.5f:0);++filters;++profiles;
			}
		}
		std::cout<<"Native Depth3D:"<<profiles<<" four-format linear/even-odd-pitch/Morton profiles,"<<planes<<" Z/mipplanes,"<<components<<" exact depth/stencil Float32 words,"<<draws<<" RSX explicitLOD draws,"<<remaps<<" remap/"<<borders<<" customborder/"<<filters<<" CPU-oracle XYZlinear/miplinear draws;oneprivate scratch,VMunchanged/removable afterupload passed;3Dshadow remainsrefused\n";
	}
	void depth_volume_specials()
	{
		unsigned profiles=0;std::size_t words_checked=0;
		for(unsigned format:{0x91u,0x93u})for(bool linear:{false,true})
		{
			const auto r=words(format,vertex_texture_dimension::three,linear,8,4,4,4);const auto plan=plan_guest_texture(r);auto source=std::make_shared<memory>();source->bytes.assign(plan.storage_bytes,std::byte{0xcd});std::vector<std::vector<unsigned>> expected;
			constexpr std::array<unsigned,10> f24{0u,1u,0x7e0000u,0x7f0000u,0x7f8000u,0x7f8123u,0x7fffffu,0xffffffu,0x800000u,0xfe0000u};
			for(const auto& s:plan.subresources){std::vector<unsigned> bits;for(unsigned z=0;z<s.depth;++z)for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x){const unsigned i=(z*s.height+y)*s.width+x;const unsigned raw=format==0x91?(f24[i%f24.size()]<<8)|(i&255):(i*7919)&65535;const auto at=guest_texture_block_offset(plan,s,x,y,z);for(unsigned c=0;c<plan.codec.block.bytes;++c)source->bytes[at+c]=std::byte(raw>>((plan.codec.block.bytes-1-c)*8));bits.push_back(format==0x91?(raw>>8)<<7:(raw<<11)+(120u<<23));bits.push_back(std::bit_cast<unsigned>(format==0x91?float(i&255):0.f));}expected.push_back(std::move(bits));}
			context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);guest_texture_resolver uploads(gpu,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);const auto image=uploads.resolve(r);
			for(unsigned level=0;level<plan.subresources.size();++level){const auto& s=plan.subresources[level];const auto output=make_buffer(gpu,expected[level].size()*4,buffer_storage::shared);texture_region area;area.level=level;area.size=MTLSizeMake(s.width,s.height,s.depth);gpu.begin();texture_to_buffer(gpu,image.texture,area,output,0,s.width*8,s.width*s.height*8);gpu.submit().wait();require(!std::memcmp(output.contents,expected[level].data(),output.length),"Depth3D float24 specialbits/stencil orE4M12 expansion changed");words_checked+=expected[level].size();}++profiles;
		}
		std::cout<<"Depth3D rawspecials:"<<profiles<<" linear/Morton F24/F16 volumes,"<<words_checked<<" bit-exact RG32 words,zero/subnormal/negative/Inf/NaN upper24 payloads andstencil preserved byintegerpack/nativeblit passed (rawreadback only,no NaNsampling claim)\n";
	}
	void depth_volume_owners()
	{
		using namespace std::chrono_literals;std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> weak;unsigned planes=0;std::size_t checked=0;
		@autoreleasepool
		{
			std::optional<guest_texture_upload_job> orphan;id<MTLSharedEvent> gate;struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{nil};
			{
				context gpu{1};ledger=std::make_shared<heap_budget>(gpu,16u<<20);gpu.set_buffer_budget(ledger);auto source=std::make_shared<memory>();weak=source;guest_texture_resolver uploads(gpu,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);std::vector<captured_guest_texture> snapshots;
				for(unsigned format:{0x90u,0x91u,0x92u,0x93u})for(bool linear:{false,true})
				{
					const auto r=words(format,vertex_texture_dimension::three,linear,8,8,4,4);const auto plan=plan_guest_texture(r);const unsigned raw=format==0x90?0x8000002au:format==0x91?0x7e00002au:format==0x92?0x8000u:0x6000u;source->bytes.assign(plan.storage_bytes,std::byte{0xcd});for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)for(unsigned y=0;y<s.rows;++y)for(unsigned x=0;x<s.columns;++x){const auto at=guest_texture_block_offset(plan,s,x,y,z);for(unsigned c=0;c<plan.codec.block.bytes;++c)source->bytes[at+c]=std::byte(raw>>((plan.codec.block.bytes-1-c)*8));}++source->version;snapshots.push_back(uploads.capture(r));
				}
				const auto before=ledger->statistics();id<MTLBuffer> blocker=ledger->shared_buffer(before.budget-before.reserved_bytes);const auto reads=source->reads;bool refused=false;try{(void)uploads.enqueue_uploads(snapshots);}catch(const heap_budget_error&){refused=true;}require(refused&&source->reads==reads&&ledger->statistics().reserved_bytes==before.budget,"Depth3D full-ledger refusal recaptured VM orleaked resources");blocker=nil;
				const auto canceled_gate=[gpu.device() newSharedEvent];release cancel_release{canceled_gate};{auto canceled=uploads.enqueue_uploads(snapshots,std::array{guest_texture_upload_dependency{canceled_gate,1}});require(!canceled.ready()&&canceled.cancel_pending(),"Depth3D pending eventjob cancellation failed");bool failed=false;try{(void)canceled.finish();}catch(const std::runtime_error&){failed=true;}require(failed&&canceled.ready(),"Cancelled Depth3D job published outputs");}gpu.recover_completed_errors();
				gate=[gpu.device() newSharedEvent];unblock.event=gate;orphan.emplace(uploads.enqueue_uploads(snapshots,std::array{guest_texture_upload_dependency{gate,1}}));require(!orphan->ready(),"Depth3D orphan gate absent");const auto charged=ledger->statistics().reserved_bytes;bool timed=false;try{(void)orphan->finish(1ms);}catch(const std::runtime_error&){timed=true;}require(timed&&!orphan->ready()&&ledger->statistics().reserved_bytes==charged,"Depth3D timeout lost native owners");source->forbid_vm_access=true;source->bytes.clear();snapshots.clear();
			}
			require(weak.expired()&&ledger->statistics().reserved_bytes>0,"Depth3D orphan retained VM orlost GPU owners");gate.signaledValue=1;auto images=orphan->finish();require(images.size()==8,"Depth3D orphan lost outputs");
			for(unsigned i=0;i<images.size();++i)for(unsigned level=0;level<4;++level)for(unsigned z=0;z<std::max(1u,4u>>level);++z)
			{
				const unsigned side=8>>level,format=std::array{0x90u,0x91u,0x92u,0x93u}[i/2];const auto output=make_buffer(ctx,side*side*8,buffer_storage::shared);texture_region area;area.level=level;area.origin=MTLOriginMake(0,0,z);area.size=MTLSizeMake(side,side,1);ctx.begin();texture_to_buffer(ctx,images[i].texture,area,output,0,side*8);ctx.submit().wait();const auto* bits=static_cast<const unsigned*>(output.contents);const unsigned depth=std::bit_cast<unsigned>(format==0x90?float(0x800000)/16777215.f:format==0x92?float(0x8000)/65535.f:.5f),stencil=std::bit_cast<unsigned>(format==0x90||format==0x91?42.f:0.f);for(unsigned p=0;p<side*side;++p){require(bits[p*2]==depth&&bits[p*2+1]==stencil,"Detached Depth3D job changed depth/stencil rawbits");checked+=2;}++planes;
			}images.clear();orphan.reset();
		}
		require(ledger->statistics().reserved_bytes==0,"Depth3D orphan retained ledger charges after final owners");
		const auto r=words(0x92,vertex_texture_dimension::three,false,8,8,4,4);const auto plan=plan_guest_texture(r);std::size_t quota;
		@autoreleasepool{auto budget=std::make_shared<heap_budget>(ctx,16u<<20);auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG32Float width:8 height:8 mipmapped:NO];d.textureType=MTLTextureType3D;d.depth=4;d.mipmapLevelCount=4;d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;const auto target=budget->texture(d);const auto source=budget->shared_buffer(plan.storage_bytes);quota=budget->statistics().reserved_bytes;(void)target;(void)source;}
		context tight{1};auto budget=std::make_shared<heap_budget>(tight,quota);tight.set_buffer_budget(budget);auto source=std::make_shared<memory>();source->bytes.resize(plan.storage_bytes);guest_texture_resolver uploads(tight,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,budget);std::array snapshots{uploads.capture(r)};
		@autoreleasepool{bool refused=false;try{(void)uploads.enqueue_uploads(snapshots);}catch(const heap_budget_error&){refused=true;}require(refused&&budget->statistics().created==2&&budget->statistics().reserved_bytes==0,"Depth3D scratch-allocation failure retained partial output/source owners");}
		std::cout<<"Depth3D upload owners:8volumes/"<<planes<<" Z-mipplanes/"<<checked<<" exact depth/stencil words,full-ledger refusal/recovery,partial scratch allocation rollback,event cancel/timeout,VM/uploader/context destruction beforecompletion,zero finalcharges passed\n";
	}
	void depth_volume_assembly_owners()
	{
		unsigned profiles=0,planes=0;std::size_t words_checked=0;
		for(unsigned format:{0x90u,0x91u,0x92u,0x93u})for(bool linear:{false,true})for(bool pieces:{false,true})
		{
			std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> provider;
			@autoreleasepool
			{
				std::optional<versioned_guest_texture_job> orphan;id<MTLSharedEvent> gate;
				struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{nil};
				const auto r=words(format,vertex_texture_dimension::three,linear,4,4,2,3);const auto plan=plan_guest_texture(r);
				const unsigned raw=format==0x90?0x4000002au:format==0x91?0x7e00002au:format==0x92?0x4000u:0x6000u;
				// Integer source formats use an exactly representable UNORM endpoint;
				// the float formats separately prove preservation of low native bits.
				const float native=format==0x90||format==0x92?1.f:std::bit_cast<float>(0x3dfcd6eau);
				const float vm=format==0x90?float(raw>>8)/16777215.f:format==0x92?float(raw)/65535.f:.5f;
				{
					context gpu{1};ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);auto source=std::make_shared<memory>();provider=source;source->bytes.assign(plan.storage_bytes,std::byte{0xcd});
					for(const auto& sub:plan.subresources)for(unsigned z=0;z<sub.depth;++z)for(unsigned y=0;y<sub.height;++y)for(unsigned x=0;x<sub.width;++x)
					{
						const auto at=guest_texture_block_offset(plan,sub,x,y,z);for(unsigned c=0;c<plan.codec.block.bytes;++c)source->bytes[at+c]=std::byte(raw>>((plan.codec.block.bytes-1-c)*8));
					}
					const auto original=source->bytes;const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();
					surface_cache surfaces(gpu,source,8u<<20,ledger,capture);guest_texture_cache textures(gpu,source,[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;},1u<<20,ledger,64,{capture,io});
					const unsigned side=pieces?2:4;const auto swizzle=linear?std::nullopt:std::optional<swizzle_region>{{{4,4,2},{},{side,side,1}}};
					const auto surface=surfaces.acquire({0x1000,{side,side,1,*plan.codec.depth,false},linear?plan.subresources[0].row_pitch:0,swizzle});surfaces.prepare(surface,surface_initialization::background);
					surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images)
					{
						auto pass=[MTL4RenderPassDescriptor new];pass.depthAttachment.texture=images[0];pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=native;
						if(format==0x90||format==0x91){pass.stencilAttachment.texture=images[0];pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=53;}(void)commands.render(pass);
					});
					guest_surface_byte_cursor cursor(plan_guest_surface(0x1000,side,side,plan.codec.block.bytes,linear?plan.subresources[0].row_pitch:0,swizzle));while(const auto run=cursor.next())source->forbidden.push_back(*run);
					gate=[gpu.device() newSharedEvent];unblock.event=gate;orphan.emplace(textures.prepare_versioned_batch(std::array{r},surfaces,std::array{guest_texture_upload_dependency{gate,1}}));
					const auto charged=ledger->statistics().reserved_bytes;bool timed=false;try{(void)orphan->finish(std::chrono::milliseconds(1));}catch(const std::runtime_error&){timed=true;}
					require(timed&&!orphan->ready()&&ledger->statistics().reserved_bytes==charged&&source->bytes==original,"Depth3D RTT pending work lost owners orwrote VM");
				}
				require(!provider.expired()&&ledger->statistics().reserved_bytes>0,"Depth3D RTT teardown released pending owners");gate.signaledValue=1;auto images=orphan->finish();require(images.size()==1,"Depth3D RTT orphan lost image");
				for(const auto& sub:plan.subresources)for(unsigned z=0;z<sub.depth;++z)
				{
					const auto output=make_buffer(ctx,sub.width*sub.height*8,buffer_storage::shared);texture_region area;area.level=sub.level;area.origin=MTLOriginMake(0,0,z);area.size=MTLSizeMake(sub.width,sub.height,1);ctx.begin();texture_to_buffer(ctx,images[0].image.texture,area,output,0,sub.width*8);ctx.submit().wait();
					const auto* bits=static_cast<const unsigned*>(output.contents);for(unsigned y=0;y<sub.height;++y)for(unsigned x=0;x<sub.width;++x)
					{
						const bool owned=sub.level==0&&z==0&&x<(pieces?2u:4u)&&y<(pieces?2u:4u);const unsigned at=(y*sub.width+x)*2;
						require(bits[at]==std::bit_cast<unsigned>(owned?native:vm)&&bits[at+1]==std::bit_cast<unsigned>((format==0x90||format==0x91)?(owned?53.f:42.f):0.f),"Detached Depth3D RTT changed native precision/stencil orVM hole");words_checked+=2;
					}++planes;
				}
				images.clear();orphan.reset();require(provider.expired(),"Depth3D RTT retained VM after final ticket");
			}
			require(ledger->statistics().reserved_bytes==0,"Depth3D RTT retained native/staging/output charges");++profiles;
		}
		std::cout<<"Depth3D RTT owners:"<<profiles<<" four-format linear/Morton full/partial-source profiles/"<<planes<<" Z-mipplanes/"<<words_checked<<" exact RG32 words;event timeout andcache/context destruction beforeGPUcompletion preserve nativeprecision/separate stencil/VMholes,provider retirement andzero finalledger charges passed\n";
	}
	void vtc_cases()
	{
		helper_catalog helpers(ctx);const auto library=helpers.library(R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void vtc_reference(texture2d<float> image [[texture(0)]],device uint4* output [[buffer(0)]],uint2 p [[thread_position_in_grid]]) {if(p.x<image.get_width()&&p.y<image.get_height())output[p.y*image.get_width()+p.x]=as_type<uint4>(image.read(p));}
)metal",@"Independent native BC2D reference reader");const auto reader=helpers.compute(library,@"vtc_reference");
		unsigned profiles=0,planes=0,draws=0,filtered=0;std::size_t components=0,blocks=0;
		for(unsigned format:{0x86u,0x87u,0x88u})for(const auto shape:std::array<std::array<unsigned,3>,10>{{{1,1,1},{3,5,2},{8,8,3},{8,8,4},{16,8,5},{16,16,7},{16,16,8},{8,4,11},{32,16,16},{32,32,9}}})for(unsigned layout=0;layout<3;++layout)
		{
			@autoreleasepool
			{
				auto r=words(format,vertex_texture_dimension::three,layout!=0,shape[0],shape[1],shape[2],4);const unsigned bb=describe_guest_texture_format(format).block.bytes;if(layout==2)r.control3=(r.control3&0xfff00000)|(((shape[0]+3)/4)*bb+1);const auto plan=plan_guest_texture(r);
				auto source=std::make_shared<memory>();source->bytes.assign(plan.storage_bytes,std::byte{0xcd});std::vector<std::vector<std::vector<std::byte>>> expected;
				const auto block=[&](unsigned x,unsigned y,unsigned z,unsigned level)
				{
					std::vector<std::byte> out(bb);const unsigned ordinal=x+y*7+z*13+level*29,color_offset=bb==8?0:8;const std::array<unsigned,6> colors{0xf800,0x07e0,0x001f,0xffff,0,0x741b};const auto a=colors[ordinal%6],b=colors[(ordinal*3+1)%6];out[color_offset]=std::byte(a);out[color_offset+1]=std::byte(a>>8);out[color_offset+2]=std::byte(b);out[color_offset+3]=std::byte(b>>8);unsigned indices=0;for(unsigned p=0;p<16;++p)indices|=((p+ordinal+(p/4))&3)<<(2*p);for(unsigned c=0;c<4;++c)out[color_offset+4+c]=std::byte(indices>>(8*c));
					if(format==0x87)for(unsigned p=0;p<16;++p)out[p/2]|=std::byte(((p+ordinal)&15)<<((p%2)*4));
					if(format==0x88){out[0]=std::byte(ordinal%2?17:231);out[1]=std::byte(ordinal%2?219:23);std::uint64_t indices=0;for(unsigned p=0;p<16;++p)indices|=std::uint64_t((p+ordinal)&7)<<(p*3);for(unsigned c=0;c<6;++c)out[c+2]=std::byte(indices>>(8*c));}return out;
				};
				for(const auto& s:plan.subresources)
				{
					std::vector<std::vector<std::byte>> slices(s.depth,std::vector<std::byte>(s.columns*s.rows*bb));std::size_t position=s.offset;const auto put=[&](unsigned x,unsigned y,unsigned z){const auto bytes=block(x,y,z,s.level);std::copy(bytes.begin(),bytes.end(),source->bytes.begin()+position);std::copy(bytes.begin(),bytes.end(),slices[z].begin()+(y*s.columns+x)*bb);position+=bb;++blocks;};
					// Enumerate bricks andpadding independently ofthe production
					// block-offset helper and GPU gather expression.
					unsigned z=0;for(;z+4<=s.depth;z+=4)for(unsigned y=0;y<s.rows;++y){for(unsigned x=0;x<s.columns;++x)for(unsigned lane=0;lane<4;++lane)put(x,y,z+lane);position+=4*(s.row_pitch-s.columns*bb);}
					for(;z<s.depth;++z)for(unsigned y=0;y<s.rows;++y){for(unsigned x=0;x<s.columns;++x)put(x,y,z);position+=s.row_pitch-s.columns*bb;}
					require(position==s.offset+s.storage_bytes,"VTC reference enumeration didnotcover source allocation");expected.push_back(std::move(slices));
				}
				const auto original=source->bytes;context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,16u<<20);gpu.set_buffer_budget(ledger);guest_texture_resolver uploader(gpu,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);const auto result=uploader.resolve(r);require(result.texture.textureType==MTLTextureType3D&&result.texture.pixelFormat==MTLPixelFormatRGBA32Float&&result.texture.mipmapLevelCount==plan.mips,"VTC target lacks native floatvolume/mips");const auto statistics=uploader.upload_statistics();require(statistics.staging_buffers==1&&statistics.conversion_dispatches==plan.mips&&statistics.vtc_array_images==1&&statistics.vtc_decode_dispatches==plan.mips&&statistics.subresources==plan.mips&&source->reads==1&&source->bytes==original,"VTC upload recaptured/wroteVM orused per-slice gather staging");source->forbid_vm_access=true;source->bytes.clear();++source->version;
				std::vector<std::vector<std::vector<color>>> decoded(plan.subresources.size());
				for(unsigned level=0;level<plan.subresources.size();++level)
				{
					const auto& s=plan.subresources[level];decoded[level].resize(s.depth);for(unsigned z=0;z<s.depth;++z)
					{
						auto reference_words=words(format,vertex_texture_dimension::two,true,s.width,s.height);reference_words.control3=s.columns*bb;ram->bytes=expected[level][z];++ram->version;const auto oracle=resolver.resolve(reference_words);const auto wanted=make_buffer(ctx,s.width*s.height*16,buffer_storage::shared),actual=make_buffer(ctx,s.width*s.height*16,buffer_storage::shared);ctx.begin();bindings resources;resources.set_texture(0,oracle.texture);resources.set_buffer(0,wanted);ctx.keep_alive(reader);const auto table=resources.snapshot(ctx);const auto encoder=ctx.compute();[encoder setComputePipelineState:reader];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(s.width,s.height,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];texture_region area;area.level=s.level;area.origin=MTLOriginMake(0,0,z);area.size=MTLSizeMake(s.width,s.height,1);texture_to_buffer(ctx,result.texture,area,actual,0,s.width*16);ctx.submit().wait();const auto expected_bits=map_buffer(wanted,0,wanted.length),actual_bits=map_buffer(actual,0,actual.length);require(std::equal(expected_bits.begin(),expected_bits.end(),actual_bits.begin()),"VTC native BC2D oracle differs in RGBAfloat bits orXYZ/mip order");components+=s.width*s.height*4;++planes;decoded[level][z].resize(s.width*s.height);std::memcpy(decoded[level][z].data(),wanted.contents,s.width*s.height*16);
						const unsigned x=s.width/2,y=s.height/2;const auto* values=static_cast<const float*>(wanted.contents)+(y*s.width+x)*4;sample(result,r,{(x+.5f)/s.width,(y+.5f)/s.height,(z+.5f)/s.depth,1},{values[0],values[1],values[2],values[3]},47,s.level);++draws;
					}
				}
				const auto interpolate=[&](unsigned level,color coordinate){const auto& s=plan.subresources[level];const std::array<std::size_t,3> size{s.width,s.height,s.depth};std::array<int,3> low;std::array<float,3> fraction;for(unsigned axis=0;axis<3;++axis){const float at=coordinate[axis]*size[axis]-.5f;low[axis]=int(std::floor(at));fraction[axis]=at-low[axis];}color value{};for(unsigned tap=0;tap<8;++tap){std::array<unsigned,3> at;float weight=1;for(unsigned axis=0;axis<3;++axis){const unsigned high=(tap>>axis)&1;at[axis]=std::clamp(low[axis]+int(high),0,int(size[axis])-1);weight*=high?fraction[axis]:1-fraction[axis];}const auto pixel=decoded[level][at[2]][at[1]*s.width+at[0]];for(unsigned c=0;c<4;++c)value[c]+=weight*pixel[c];}return value;};
				for(color coordinate:std::array<color,2>{{{.375f,.625f,.5f,1},{.5f,.5f,.5f,1}}}){auto filtered_words=r;filtered_words.filter=(2u<<24)|(6u<<16);const auto first=interpolate(0,coordinate),second=interpolate(plan.mips>1?1:0,coordinate);color value;for(unsigned c=0;c<4;++c)value[c]=plan.mips>1?(first[c]+second[c])*.5f:first[c];sample(result,filtered_words,coordinate,value,47,plan.mips>1?.5f:0);++filtered;}

				++profiles;
			}
		}
		std::cout<<"Native VTC3D:"<<profiles<<" BC1/2/3 linear/even-odd-pitch andpacked profiles,"<<planes<<" Z/mipplanes,"<<blocks<<" independently enumerated VTCblocks,"<<components<<" bit-exact native BC2D-reference floatcomponents,"<<draws<<" independent RSX explicit-LOD draws,"<<filtered<<" CPU-oracle XYZlinear/miplinear draws;depth1/2/3/4/5/7/8/9/11/16,partialblocks/tailmips,BC1transparent/BC2explicit/BC3bothalphamodes,no addedUNORMquantization,onegather scratch,VMunchanged/removed afterupload passed\n";
	}
	void vtc_piece_cases()
	{
		unsigned profiles=0,planes=0;std::size_t words_checked=0,hole_bytes=0;
		for(unsigned format:{0x86u,0x87u,0x88u})for(bool linear:{false,true})for(bool morton:{false,true})for(unsigned samples:{1u,2u,4u})
		{
			@autoreleasepool
			{
				auto r=words(format,vertex_texture_dimension::three,linear,16,16,7,3);const auto plan=plan_guest_texture(r);auto source=std::make_shared<memory>();source->bytes.resize(plan.storage_bytes);for(unsigned i=0;i<source->bytes.size();++i)source->bytes[i]=std::byte((i*37+i/11+71)&255);const auto original=source->bytes;auto expected=original;std::vector<bool> mask(plan.storage_bytes);
				context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,16u<<20);gpu.set_buffer_budget(ledger);const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;};surface_cache surfaces(gpu,source,4u<<20,ledger,capture);guest_texture_cache textures(gpu,source,translate,1u<<20,ledger,64,{capture,io});surface_color_transfer upload(gpu);sample_transfer sample_copy(gpu);std::vector<cached_surface_ref> owners;
				for(unsigned part=0;part<2;++part)
				{
					const unsigned width=part?4:8,height=4,bpp=part?4:1,base=part?0x200:5,pitch=part?41:19;const auto sf=part?surface_color::argb8:surface_color::b8;const auto swizzle=morton?std::optional<swizzle_region>{{{part?8u:16u,8,1},{1,1,0},{width,height,1}}}:std::nullopt;const auto geometry=plan_guest_surface(0,width,height,bpp,morton?0:pitch,swizzle);const auto factors=samples==1?sample_layout{1,1}:rsx_sample_layout(samples);const auto native=surfaces.acquire({0x1000+base,{width/factors.x,height/factors.y,samples,sf,false},morton?0u:pitch,swizzle});owners.push_back(native);surfaces.prepare(native,surface_initialization::background);std::vector<std::byte> pixels(width*height*bpp);for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte((i*17+part*71+29)&255);
					surfaces.draw(std::array{cached_surface_use{native,true}},[&](context& commands,auto images){id<MTLTexture> expanded=images[0];if(samples>1){auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat width:width height:height mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;expanded=make_texture(commands,d);}texture_region area;area.size=MTLSizeMake(width,height,1);upload.upload(commands,sf,make_upload(commands,pixels),{},expanded,area);if(samples>1)sample_copy.import(commands,expanded,images[0]);});
					for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned c=0;c<bpp;++c){const auto offset=base+(morton?morton_index({x+1,y+1,0},plan_swizzle(*swizzle,bpp).log2)*bpp:y*pitch+x*bpp)+c;expected.at(offset)=pixels[(y*width+x)*bpp+c];mask.at(offset)=true;}
					guest_surface_byte_cursor cursor(plan_guest_surface(0x1000+base,width,height,bpp,morton?0:pitch,swizzle));while(const auto run=cursor.next())source->forbidden.push_back(*run);
				}
				const auto image=textures.resolve_versioned(r,surfaces);auto reference_source=std::make_shared<memory>();reference_source->bytes=expected;guest_texture_resolver reference(gpu,reference_source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);const auto oracle=reference.resolve(r);std::size_t missing=0;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)
				{
					const auto actual=make_buffer(gpu,s.width*s.height*16,buffer_storage::shared),wanted=make_buffer(gpu,s.width*s.height*16,buffer_storage::shared);texture_region region;region.level=s.level;region.origin=MTLOriginMake(0,0,z);region.size=MTLSizeMake(s.width,s.height,1);gpu.begin();texture_to_buffer(gpu,image.image.texture,region,actual,0,s.width*16);texture_to_buffer(gpu,oracle.texture,region,wanted,0,s.width*16);gpu.submit().wait();require(!std::memcmp(actual.contents,wanted.contents,actual.length),"VTC RTT native pieces/VMholes differ fromindependent fullvolume upload");words_checked+=s.width*s.height*4;++planes;
					for(unsigned y=0;y<s.rows;++y)for(unsigned x=0;x<s.columns;++x)for(unsigned c=0;c<plan.codec.block.bytes;++c){const auto raw=s.offset+(z<(s.depth&~std::size_t(3))?(z/4)*4*s.row_pitch*s.storage_rows+4*(y*s.row_pitch+x*plan.codec.block.bytes)+(z%4)*plan.codec.block.bytes:z*s.row_pitch*s.storage_rows+y*s.row_pitch+x*plan.codec.block.bytes)+c;missing+=!mask.at(raw);}
				}
				require(textures.statistics().assembly_vm_bytes==missing&&source->bytes==original,"VTC RTT captured nonhole/padding bytes orwroteVM");hole_bytes+=missing;const auto created=ledger->statistics().created;const auto encoding=gpu.encoding_statistics();const auto reads=source->reads;require(textures.resolve_versioned(r,surfaces).image.texture==image.image.texture&&source->reads==reads&&ledger->statistics().created==created&&gpu.encoding_statistics().compute_encoders==encoding.compute_encoders&&gpu.encoding_statistics().render_encoders==encoding.render_encoders,"Warm VTC RTT recaptured/allocated/encoded");if(profiles%3==0){io->notify({0,0x100000});io->notify({0,0x100000});require(!image.revision.current(),"VTC RTT ticket accepted IO ABA");}else if(profiles%3==1){++source->version;require(!image.revision.current(),"VTC RTT ticket accepted changed VM input");}else{surfaces.draw(std::array{cached_surface_use{owners[0],true}},[&](context& commands,auto images){auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(.5,.25,.75,1);(void)commands.render(pass);});const auto fresh=textures.resolve_versioned(r,surfaces);require(fresh.image.texture!=image.image.texture&&textures.resolve_versioned(r,surfaces).image.texture==fresh.image.texture,"VTC RTT reused old nativewriter cacheimage");const auto old=make_buffer(gpu,16*16*16,buffer_storage::shared),wanted=make_buffer(gpu,16*16*16,buffer_storage::shared);texture_region area;area.size=MTLSizeMake(16,16,1);gpu.begin();texture_to_buffer(gpu,image.image.texture,area,old,0,16*16);texture_to_buffer(gpu,oracle.texture,area,wanted,0,16*16);gpu.submit().wait();require(!std::memcmp(old.contents,wanted.contents,old.length)&&image.revision.current(),"VTC nativewriter changed frozen pixels/VMreceipt");}source->forbidden.clear();++profiles;
			}
		}
		std::cout<<"Native VTC3D pieces:"<<profiles<<" BC1/2/3 packed/linear/Morton-native 1-2-4sample profiles,"<<planes<<" Z/mipplanes,"<<words_checked<<" exactRGBAfloat words,"<<hole_bytes<<" exactVMholebytes;two native sources/partialcompressed-block bytes,hardwareBC decode tofloat32volume,no nativeVMcapture/writeback,warm zeroallocation/capture/encoding,12IO-ABA+12VM-ticket refusals/12newwriter cache refreshes withfrozen oldpixels passed\n";
	}
	void vtc_owners()
	{
		using namespace std::chrono_literals;std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> weak;unsigned planes=0;std::size_t words_checked=0;
		@autoreleasepool
		{
			std::optional<guest_texture_upload_job> orphan;id<MTLSharedEvent> gate;struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{nil};
			{
				context gpu{1};ledger=std::make_shared<heap_budget>(gpu,16u<<20);gpu.set_buffer_budget(ledger);auto source=std::make_shared<memory>();weak=source;guest_texture_resolver uploads(gpu,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);std::vector<captured_guest_texture> snapshots;
				for(unsigned format:{0x86u,0x87u,0x88u})for(bool linear:{false,true})
				{
					const auto r=words(format,vertex_texture_dimension::three,linear,8,8,5,2);const auto plan=plan_guest_texture(r);const unsigned bb=plan.codec.block.bytes,co=bb==8?0:8;std::vector<std::byte> block(bb);block[co+1]=std::byte{0xf8};block[co+2]=std::byte{0x1f};if(format==0x87)std::fill_n(block.begin(),8,std::byte{0xff});if(format==0x88)block[0]=std::byte{0xff};source->bytes.assign(plan.storage_bytes,std::byte{0xcd});for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)for(unsigned y=0;y<s.rows;++y)for(unsigned x=0;x<s.columns;++x){const auto at=guest_texture_block_offset(plan,s,x,y,z);std::copy(block.begin(),block.end(),source->bytes.begin()+at);}++source->version;snapshots.push_back(uploads.capture(r));
				}
				const auto before=ledger->statistics();id<MTLBuffer> blocker=ledger->shared_buffer(before.budget-before.reserved_bytes);const auto reads=source->reads;bool refused=false;try{(void)uploads.enqueue_uploads(snapshots);}catch(const heap_budget_error&){refused=true;}require(refused&&source->reads==reads&&ledger->statistics().reserved_bytes==before.budget,"VTC full-ledger refusal recaptured VM orleaked resources");blocker=nil;
				const auto canceled_gate=[gpu.device() newSharedEvent];release cancel_release{canceled_gate};{
					auto canceled=uploads.enqueue_uploads(snapshots,std::array{guest_texture_upload_dependency{canceled_gate,1}});require(!canceled.ready()&&canceled.cancel_pending(),"VTC pending eventjob cancellation failed");bool failed=false;try{(void)canceled.finish();}catch(const std::runtime_error&){failed=true;}require(failed&&canceled.ready(),"Cancelled VTC job published outputs");
				}gpu.recover_completed_errors();
				gate=[gpu.device() newSharedEvent];unblock.event=gate;orphan.emplace(uploads.enqueue_uploads(snapshots,std::array{guest_texture_upload_dependency{gate,1}}));require(!orphan->ready(),"VTC orphan gate absent");const auto charged=ledger->statistics().reserved_bytes;bool timed=false;try{(void)orphan->finish(1ms);}catch(const std::runtime_error&){timed=true;}require(timed&&!orphan->ready()&&ledger->statistics().reserved_bytes==charged,"VTC timeout lost native resource owners");source->forbid_vm_access=true;source->bytes.clear();snapshots.clear();
			}
			require(weak.expired()&&ledger->statistics().reserved_bytes>0,"VTC orphan retained VM orlost GPU owners");gate.signaledValue=1;auto images=orphan->finish();require(images.size()==6,"VTC orphan lost outputs");
			for(const auto& image:images)for(unsigned level=0;level<2;++level)for(unsigned z=0;z<(5u>>level);++z)
			{
				const unsigned side=8>>level;const auto output=make_buffer(ctx,side*side*16,buffer_storage::shared);texture_region region;region.level=level;region.origin=MTLOriginMake(0,0,z);region.size=MTLSizeMake(side,side,1);ctx.begin();texture_to_buffer(ctx,image.texture,region,output,0,side*16);ctx.submit().wait();const auto* bits=static_cast<const unsigned*>(output.contents);for(unsigned i=0;i<side*side*4;++i){require(bits[i]==(i%4==0||i%4==3?0x3f800000u:0u),"Detached VTC upload changed exact RGBAfloat bits");++words_checked;}++planes;
			}images.clear();orphan.reset();
		}
		require(ledger->statistics().reserved_bytes==0,"VTC orphan retained ledger charges after final owners");
		// Force failure after both the expanded output and immutable compressed
		// source exist,but before the intermediate native BC array can be made.
		const auto r=words(0x86,vertex_texture_dimension::three,false,8,8,5,2);const auto plan=plan_guest_texture(r);std::size_t quota;
		@autoreleasepool
		{
			auto budget=std::make_shared<heap_budget>(ctx,16u<<20);auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:8 height:8 mipmapped:NO];d.textureType=MTLTextureType3D;d.depth=5;d.mipmapLevelCount=2;d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsagePixelFormatView;const auto target=budget->texture(d);const auto source=budget->shared_buffer(plan.storage_bytes);quota=budget->statistics().reserved_bytes;(void)target;(void)source;
		}
		context tight{1};auto budget=std::make_shared<heap_budget>(tight,quota);tight.set_buffer_budget(budget);auto source=std::make_shared<memory>();source->bytes.resize(plan.storage_bytes);guest_texture_resolver uploads(tight,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,budget);std::array captures{uploads.capture(r)};
		@autoreleasepool{bool refused=false;try{(void)uploads.enqueue_uploads(captures);}catch(const heap_budget_error&){refused=true;}require(refused&&budget->statistics().created==2&&budget->statistics().reserved_bytes==0,"VTC intermediate-allocation failure kept partial output/source owners");}
		std::cout<<"VTC upload owners:6volumes/"<<planes<<" Z-mipplanes/"<<words_checked<<" exactRGBAfloat words,full-ledger refusal/recovery,partialnative-BC-array allocation rollback,event cancel/timeout,VM/uploader/context destruction beforecompletion,zero finalcharges passed\n";
	}
	void depth_cube_cases()
	{
		unsigned profiles=0,planes=0,draws=0,shadow_draws=0;std::size_t words_checked=0;
		for(unsigned format:{0x90u,0x91u,0x92u,0x93u})for(unsigned layout=0;layout<3;++layout)
		{
			@autoreleasepool
			{
				auto r=words(format,vertex_texture_dimension::cube,layout!=2,8,8,1,4);const unsigned bpp=describe_guest_texture_format(format).block.bytes;
				if(layout==1)r.control3=(r.control3&0xfff00000)|(8*bpp+1);
				const auto plan=plan_guest_texture(r);ram->bytes.assign(plan.storage_bytes,std::byte{0xa7});++ram->version;
				const auto raw=[&](unsigned face,unsigned level,unsigned x,unsigned y)
				{
					if(format==0x92)return (0x1234u+face*7777+level*997+y*67+x*11)&65535;
					if(format==0x93)return 0x6800u+face*0x500+level*0x100+x*11+y*23;
					const auto value=format==0x90?0x123456u+face*0x10203+level*0x40000+y*61+x*23:std::bit_cast<unsigned>((32+face*8+level*2+x+y)/128.f)>>7;
					return (value<<8)|((face*37+level*17+x*3+y*5)&255);
				};
				for(const auto& s:plan.subresources)for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)
				{
					const auto at=s.offset+(layout==2?morton_index({x,y,0},{unsigned(std::bit_width(s.columns)-1),unsigned(std::bit_width(s.rows)-1),0})*bpp:y*s.row_pitch+x*bpp);const unsigned word=raw(s.face,s.level,x,y);for(unsigned c=0;c<bpp;++c)ram->bytes[at+c]=std::byte(word>>((bpp-1-c)*8));
				}
				const auto original=ram->bytes;const auto image=resolver.resolve(r);require(image.texture.textureType==MTLTextureTypeCube&&image.texture.mipmapLevelCount==4,"Depthcube upload lost native cube/mips");
				for(const auto& s:plan.subresources)
				{
					const unsigned unit=format==0x92?2:4;const auto output=make_buffer(ctx,512*s.height,buffer_storage::shared);texture_region region;region.level=s.level;region.slice=s.face;region.size=MTLSizeMake(s.width,s.height,1);ctx.begin();texture_to_buffer(ctx,image.texture,region,output,0,256,0,texture_plane::depth);if(bpp==4)texture_to_buffer(ctx,image.texture,region,output,256*s.height,256,0,texture_plane::stencil);ctx.submit().wait();const auto data=map_buffer(output,0,output.length);
					for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)
					{
						const auto value=raw(s.face,s.level,x,y);const unsigned expected=format==0x92?value:format==0x90?std::bit_cast<unsigned>(float(value>>8)/16777215.f):format==0x91?(value>>8)<<7:(value<<11)+(120u<<23);unsigned actual=0;std::memcpy(&actual,data.data()+y*256+x*unit,unit);require(actual==expected,"Depthcube face/mip/odd-pitch/Morton rawdepth differs");if(bpp==4)require(data[256*s.height+y*256+x]==std::byte(value),"Depthcube separate stencil differs");++words_checked;
					}
					const unsigned x=s.width/2,y=s.height/2;const auto value=raw(s.face,s.level,x,y);const float expected=format==0x92?value/65535.f:format==0x90?float(value>>8)/16777215.f:std::bit_cast<float>(format==0x91?(value>>8)<<7:(value<<11)+(120u<<23));const float u=(x+.5f)/s.width,v=(y+.5f)/s.height,sc=2*u-1,tc=2*v-1;const std::array<color,6> directions{{{1,-tc,-sc,1},{-1,-tc,sc,1},{sc,1,tc,1},{sc,-1,-tc,1},{sc,-tc,1,1},{-sc,-tc,-1,1}}};
					sample(image,r,directions[s.face],{expected,expected,expected,expected},47,s.level);++draws;
					for(float reference:{expected-1.f/2048,expected+1.f/2048}){auto shadow=r;shadow.address=(shadow.address&0x0fffffff)|(1u<<28);shadow.control0=(1u<<31)|((s.level*256u)<<19)|((s.level*256u)<<7);shadow.filter=(1u<<24)|(3u<<16);auto coordinate=directions[s.face];coordinate[3]=reference;const float passes=expected<reference;sample(image,shadow,coordinate,{passes,passes,passes,passes});++shadow_draws;}++planes;
				}
				require(ram->bytes==original,"Depthcube upload modified VM/padding");++profiles;
			}
		}
		std::cout<<"Native depthcube uploads:"<<profiles<<" four-depth-format linear/even-odd-pitch/Morton profiles,"<<planes<<" faces/mips,"<<words_checked<<" exact native depthwords andseparate stencil,"<<draws<<" independent RSX explicit-LOD draws,"<<shadow_draws<<" shadow compare draws withmipclamps;all6faces/4mips including1x1,VM/padding unchanged passed\n";
	}
	void depth_cube_owners()
	{
		using namespace std::chrono_literals;std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> provider;unsigned planes=0;std::size_t bits=0;
		@autoreleasepool
		{
			std::optional<guest_texture_upload_job> orphan;id<MTLSharedEvent> gate;struct release{id<MTLSharedEvent> gate;~release(){gate.signaledValue=1;}} unblock{nil};
			{
				context gpu{1};ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);auto source=std::make_shared<memory>();provider=source;guest_texture_resolver uploads(gpu,source,[](unsigned,unsigned,std::size_t){return 0x1000u;},1u<<20,ledger);std::vector<captured_guest_texture> snapshots;
				for(unsigned format:{0x90u,0x91u,0x92u,0x93u})for(bool linear:{false,true})
				{
					const auto r=words(format,vertex_texture_dimension::cube,linear,8,8,1,4);const auto plan=plan_guest_texture(r);const unsigned value=format==0x90?0x8000002au:format==0x91?0x7e00002au:format==0x92?0x8000u:0x6000u,bpp=plan.codec.block.bytes;std::vector<unsigned> word(bpp);for(unsigned c=0;c<bpp;++c)word[c]=(value>>((bpp-1-c)*8))&255;fill(plan,word);source->bytes=ram->bytes;++source->version;snapshots.push_back(uploads.capture(r));
				}
				// Actual exhausted native ledger must refuse without source reads
				// or retaining partial target/staging allocations.
				const auto before=ledger->statistics();id<MTLBuffer> blocker=ledger->shared_buffer(before.budget-before.reserved_bytes);const auto reads=source->reads;bool refused=false;try{(void)uploads.enqueue_uploads(snapshots);}catch(const heap_budget_error&){refused=true;}require(refused&&source->reads==reads&&ledger->statistics().reserved_bytes==before.budget,"Depthcube full-ledger admission leaked orrecaptured VM");blocker=nil;
				gate=[gpu.device() newSharedEvent];unblock.gate=gate;orphan.emplace(uploads.enqueue_uploads(snapshots,std::array{guest_texture_upload_dependency{gate,1}}));require(!orphan->ready(),"Depthcube detached job gate absent");const auto charged=ledger->statistics().reserved_bytes;bool timed=false;try{(void)orphan->finish(1ms);}catch(const std::runtime_error&){timed=true;}require(timed&&!orphan->ready()&&ledger->statistics().reserved_bytes==charged,"Depthcube timeout released pending native owners");source->forbid_vm_access=true;source->bytes.clear();snapshots.clear();
			}
			require(provider.expired()&&ledger->statistics().reserved_bytes>0,"Depthcube orphan retained VM orlost GPU owners");gate.signaledValue=1;auto images=orphan->finish();require(images.size()==8,"Depthcube orphan lost outputs");
			for(unsigned i=0;i<images.size();++i)
			{
				const unsigned format=std::array{0x90u,0x91u,0x92u,0x93u}[i/2],unit=format==0x92?2:4,wanted=format==0x92?0x8000:format==0x90?std::bit_cast<unsigned>(float(0x800000)/16777215.f):0x3f000000;
				for(unsigned face=0;face<6;++face)for(unsigned level=0;level<4;++level)
				{
					const unsigned side=8>>level;const auto output=make_buffer(ctx,512*side,buffer_storage::shared);texture_region region;region.slice=face;region.level=level;region.size=MTLSizeMake(side,side,1);ctx.begin();texture_to_buffer(ctx,images[i].texture,region,output,0,256,0,texture_plane::depth);if(format==0x90||format==0x91)texture_to_buffer(ctx,images[i].texture,region,output,256*side,256,0,texture_plane::stencil);ctx.submit().wait();const auto data=map_buffer(output,0,output.length);for(unsigned y=0;y<side;++y)for(unsigned x=0;x<side;++x){unsigned actual=0;std::memcpy(&actual,data.data()+y*256+x*unit,unit);require(actual==wanted,"Detached depthcube job changed depthbits");if(format==0x90||format==0x91)require(data[side*256+y*256+x]==std::byte{42},"Detached depthcube job changed stencil");++bits;}++planes;
				}
			}
			images.clear();orphan.reset();
		}
		require(ledger->statistics().reserved_bytes==0,"Depthcube orphan ledger didnotretire after final owners");
		std::cout<<"Depthcube upload owners:8native cubes/"<<planes<<" face-mips/"<<bits<<" exact depthwords andseparate stencil,actualfull-ledger refusal/recovery,event timeout,VM/uploader/context destruction before completion,zero finalledger charges passed\n";
	}
	void assembly_cases()
	{
		using D=vertex_texture_dimension;unsigned profiles=0,native_planes=0,vm_planes=0;const auto begin=cases;context assembly_ctx{1};auto ledger=std::make_shared<heap_budget>(assembly_ctx,8u<<20);assembly_ctx.set_buffer_budget(ledger);
		const std::array<color,6> directions{{{1,0,0,1},{-1,0,0,1},{0,1,0,1},{0,-1,0,1},{0,0,1,1},{0,0,-1,1}}};
		for(auto dimension:{D::two,D::cube,D::three})for(bool linear:{false,true})for(bool mixed_vm:{false,true})for(unsigned samples:{1u,2u,4u})for(unsigned target_format:{0x85u,0x9eu})
		{
			@autoreleasepool
			{
				const auto r=words(target_format,dimension,linear,8,8,dimension==D::three?4:1,4);const auto plan=plan_guest_texture(r);fill(plan,{96,20,64,128});
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)for(unsigned y=0;y<s.rows;++y)for(unsigned x=0;x<s.columns;++x)
				{
					const auto at=s.offset+(linear?(z*s.storage_rows+y)*s.row_pitch+x*4:morton_index({x,y,z},{unsigned(std::bit_width(s.columns)-1),unsigned(std::bit_width(s.rows)-1),unsigned(std::bit_width(s.depth)-1)})*4);ram->bytes[at+1]=std::byte(20+s.level*16+s.face+z*4);
				}
				const auto original=ram->bytes;auto io=std::make_shared<io_mapping_epoch>();const cpu_capture_dispatch capture=[](const auto& work){work();};
				const auto translate=[&](unsigned offset,unsigned location,std::size_t size){require(location==1&&offset>=0x200&&offset-0x200<=ram->bytes.size()&&size<=ram->bytes.size()-(offset-0x200),"Assembly translation outside owned texture");return 0x1000+offset-0x200;};
				surface_cache surfaces(assembly_ctx,ram,8u<<20,ledger,capture);guest_texture_cache textures(assembly_ctx,ram,translate,1u<<20,ledger,64,{capture,io});std::vector<cached_surface_ref> sources;std::vector<color> expected;
				unsigned index=0;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z,++index)
				{
					const bool native=!mixed_vm||index%3!=1;const auto format=std::array{surface_color::argb8,surface_color::abgr8,surface_color::xrgb8_zero,surface_color::xrgb8_one}[index%4];
					const unsigned red=(native?32:20)+s.level*16+s.face+z*4,alpha=96+s.face+z;
					const float a=native?(format==surface_color::xrgb8_zero?0.f:format==surface_color::xrgb8_one||format==surface_color::abgr8&&target_format==0x9e?1.f:alpha/255.f):target_format==0x9e?1.f:96.f/255;
					expected.push_back({red/255.f,64.f/255,128.f/255,a});if(!native){++vm_planes;continue;}
					const auto scale=samples==1?sample_layout{1,1}:rsx_sample_layout(samples);const unsigned count=s.width>=scale.x&&s.height>=scale.y?samples:1;const auto factors=count==1?sample_layout{1,1}:scale;
					const auto swizzle=!linear?std::optional<swizzle_region>{{{s.columns,s.rows,s.depth},{0,0,z},{s.columns,s.rows,1}}}:std::nullopt;
					const unsigned address=0x1000+s.offset+(linear?z*s.row_pitch*s.storage_rows:0);
					const auto source=surfaces.acquire({address,{s.width/factors.x,s.height/factors.y,count,format,false},linear?s.row_pitch:0,swizzle});surfaces.clear(source,0xf0,(alpha<<24)|(red<<16)|(64<<8)|128,0,255,{0,0,s.width,s.height});sources.push_back(source);
					guest_surface_byte_cursor cursor(plan_guest_surface(address,s.width,s.height,4,linear?s.row_pitch:0,swizzle));while(const auto run=cursor.next())ram->forbidden.push_back(*run);++native_planes;
				}
				auto remapped=r;remapped.control1=0xaa6c;const auto before_reads=ram->reads;const auto version=ram->version;
				auto job=textures.prepare_versioned_batch(std::array{r,remapped},surfaces);const auto result=job.finish();require(result.size()==2&&result[0].image.texture==result[1].image.texture&&result[0].image.metadata.actual_mips==4&&textures.statistics().assemblies_created==1,"RTT assembly didnotcoalesce one fullnative mip/face/Z resource");
				index=0;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z,++index)
				{
					const color coordinate=dimension==D::cube?directions[s.face]:color{.25f,.25f,dimension==D::three?(z+.5f)/s.depth:0,1};sample(result[0].image,r,coordinate,expected[index],47,s.level);
					const auto v=expected[index];sample(result[1].image,remapped,coordinate,{v[2],v[1],v[0],v[3]},47,s.level);
				}
				const auto reads=ram->reads;const auto created=ledger->statistics().created;const auto encoded=assembly_ctx.encoding_statistics();const auto warm=textures.resolve_versioned(r,surfaces);
				require(warm.image.texture==result[0].image.texture&&ram->reads==reads&&ledger->statistics().created==created&&assembly_ctx.encoding_statistics().compute_encoders==encoded.compute_encoders&&assembly_ctx.encoding_statistics().render_encoders==encoded.render_encoders,"Warm RTT assembly recaptured/allocated/encoded");
				require(ram->bytes==original&&ram->version==version&&std::all_of(sources.begin(),sources.end(),[&](const auto& s){return surfaces.status(s).dirty;}),"RTT assembly wroteback VM orlostnative authority");
				if(!mixed_vm)require(ram->reads==before_reads,"Fully native RTT assembly captured VM pixels");ram->forbidden.clear();++profiles;
			}
		}
		std::cout<<"Native RTT planes:"<<profiles<<" 2D-mip/cube/3D linear-Morton profiles,"<<native_planes<<" native/"<<vm_planes<<" VMplanes,"<<cases-begin<<" independent RSX mip/face/Z sampler draws;ARGB/ABGR/X0/X1 metadata andtargetconstant-alpha normalization,two remaps,1-2-4samples,VM poison/nativecaptureguard,wholepending coalescing,stablewarm zeroallocation/capture/encoding,unchangedVM/dirtyauthority passed\n";
	}
	void assembly_precision_cases()
	{
		unsigned profiles=0;std::size_t checks=0;context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);
		for(unsigned format:{0x9au,0x9bu,0x9cu,0x91u,0x93u})for(auto dimension:{vertex_texture_dimension::two,vertex_texture_dimension::cube,vertex_texture_dimension::three})for(bool linear:{false,true})
		{
			@autoreleasepool
			{
				const auto r=words(format,dimension,linear,4,4,dimension==vertex_texture_dimension::three?2:1,3);const auto plan=plan_guest_texture(r);const bool volume=guest_texture_depth_volume(plan);fill(plan,std::vector<unsigned>(plan.codec.block.bytes,0xa7));const auto original=ram->bytes;
				const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;};
				surface_cache surfaces(gpu,ram,8u<<20,ledger,capture);guest_texture_cache textures(gpu,ram,translate,1u<<20,ledger,64,{capture,io});std::vector<std::array<float,4>> expected;
				unsigned index=0;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z,++index)
				{
					const surface_format native=format==0x91?surface_format{surface_depth::z24s8_float}:format==0x93?surface_format{surface_depth::z16_float}:format==0x9a?surface_format{surface_color::rgba16f}:format==0x9b?surface_format{surface_color::rgba32f}:surface_format{surface_color::r32f};
					const unsigned address=0x1000+s.offset+(linear?z*s.row_pitch*s.storage_rows:0);const auto swizzle=!linear?std::optional<swizzle_region>{{{s.columns,s.rows,s.depth},{0,0,z},{s.columns,s.rows,1}}}:std::nullopt;
					const auto source=surfaces.acquire({address,{s.width,s.height,1,native,false},linear?s.row_pitch:0,swizzle});surfaces.prepare(source,surface_initialization::background);
					const float v=std::bit_cast<float>(0x3dfcd6eau+index*257u);std::array<float,4> value{v,v*2,v*3,.75f};if(format==0x9a)for(auto& f:value)f=float(_Float16(f));expected.push_back(value);
					const std::array uses{cached_surface_use{source,true}};surfaces.draw(uses,[&](context& commands,auto images)
					{
						auto pass=[MTL4RenderPassDescriptor new];if(plan.codec.depth){pass.depthAttachment.texture=images[0];pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=v;if(format==0x91){pass.stencilAttachment.texture=images[0];pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=42+index;}}
						else{pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(value[0],value[1],value[2],value[3]);}(void)commands.render(pass);
					});
					guest_surface_byte_cursor cursor(plan_guest_surface(address,s.width,s.height,plan.codec.block.bytes,linear?s.row_pitch:0,swizzle));while(const auto run=cursor.next())ram->forbidden.push_back(*run);
				}
				const auto reads=ram->reads;const auto image=textures.resolve_versioned(r,surfaces);index=0;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z,++index)
				{
					texture_region region;region.level=s.level;region.slice=s.face;region.origin=MTLOriginMake(0,0,z);region.size=MTLSizeMake(s.width,s.height,1);const auto raw=make_buffer(gpu,512*s.height,buffer_storage::shared);
					gpu.begin();texture_to_buffer(gpu,image.image.texture,region,raw,0,256,0,plan.codec.depth&&!volume?texture_plane::depth:texture_plane::all);if(format==0x91&&!volume)texture_to_buffer(gpu,image.image.texture,region,raw,256*s.height,256,0,texture_plane::stencil);gpu.submit().wait();
					const auto bytes=map_buffer(raw,0,raw.length);const unsigned components=format==0x91||format==0x93||format==0x9c?1:4,unit=format==0x9a?2:4;
					for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)for(unsigned c=0;c<components;++c)
					{
						const auto offset=y*256+(x*(volume?2:components)+c)*unit;unsigned actual=0;std::memcpy(&actual,bytes.data()+offset,unit);const auto wanted=unit==2?unsigned(std::bit_cast<std::uint16_t>(_Float16(expected[index][c]))):std::bit_cast<unsigned>(expected[index][c]);require(actual==wanted,"RTT plane composition quantized native float/depth bits");++checks;
						if(format==0x91){if(volume){unsigned stencil;std::memcpy(&stencil,bytes.data()+y*256+x*8+4,4);require(stencil==std::bit_cast<unsigned>(float(42+index)),"Depth3D RTT lost independent stencil");}else require(bytes[256*s.height+y*256+x]==std::byte(42+index),"RTT composition lost independent stencil plane");}
					}
				}
				require(ram->reads==reads&&ram->bytes==original&&textures.statistics().assemblies_created==1,"Native float RTT composition captured/wrote VM");ram->forbidden.clear();++profiles;
			}
		}
		std::cout<<"Native RTT precision:"<<profiles<<" linear/Morton RGBA16F/RGBA32F/R32F mip/cube/volume andZ16F/Z24S8F mip profiles/"<<checks<<" exactnative component bits,independent stencil;float/depth precision preserved bynative plane copies,zero VMpixelcapture/writeback passed\n";
	}
	void assembly_lifetime_cases()
	{
		struct owned
		{
			std::unique_ptr<context> gpu;
			std::shared_ptr<heap_budget> ledger;
			std::shared_ptr<memory> ram;
			std::shared_ptr<io_mapping_epoch> io;
			std::unique_ptr<surface_cache> surfaces;
			std::unique_ptr<guest_texture_cache> textures;
			cached_surface_ref source;
			fragment_texture_registers registers;
			__weak id<MTLTexture> source_texture=nil;
			std::shared_ptr<std::function<void()>> after_capture=std::make_shared<std::function<void()>>();
		};
		const auto make=[&](unsigned slots=3,bool pieces=false)
		{
			auto f=std::make_unique<owned>();f->gpu=std::make_unique<context>(slots);f->ledger=std::make_shared<heap_budget>(*f->gpu,8u<<20);f->gpu->set_buffer_budget(f->ledger);f->ram=std::make_shared<memory>();f->io=std::make_shared<io_mapping_epoch>();
			f->registers=words(0x85,vertex_texture_dimension::cube,true,4,4,1,3);const auto plan=plan_guest_texture(f->registers);f->ram->bytes.resize(plan.storage_bytes);for(unsigned i=0;i<plan.storage_bytes;++i)f->ram->bytes[i]=std::byte(std::array{255u,20u,64u,128u}[i%4]);
			const cpu_capture_dispatch capture=[after=f->after_capture](const auto& work){work();if(*after)(*after)();};const auto translate=[size=plan.storage_bytes](unsigned offset,unsigned location,std::size_t count){require(location==1&&offset>=0x200&&offset-0x200<=size&&count<=size-(offset-0x200),"Owned assembly translated outside captured storage");return 0x1000+offset-0x200;};
			f->surfaces=std::make_unique<surface_cache>(*f->gpu,f->ram,8u<<20,f->ledger,capture);f->textures=std::make_unique<guest_texture_cache>(*f->gpu,f->ram,translate,1u<<20,f->ledger,64,texture_capture_bindings{capture,f->io});
			const unsigned side=pieces?2:4;f->source=f->surfaces->acquire({0x1000,{side,side,1,surface_color::argb8,false},plan.subresources[0].row_pitch,{}});f->surfaces->clear(f->source,0xf0,0xff334455,0,255,{0,0,side,side});
			const std::array uses{cached_surface_use{f->source,true}};f->surfaces->draw(uses,[&](context& commands,auto images){f->source_texture=images[0];auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(51./255,68./255,85./255,1);(void)commands.render(pass);});
			guest_surface_byte_cursor cursor(plan_guest_surface(0x1000,side,side,4,plan.subresources[0].row_pitch));while(const auto run=cursor.next())f->ram->forbidden.push_back(*run);return f;
		};
		const auto verify=[](const resolved_fragment_texture& image,unsigned latest=0,bool pieces=false,bool deferred=false)
		{
			context reader;const auto raw=make_buffer(reader,2048,buffer_storage::shared);reader.begin();for(unsigned face=0;face<2;++face){texture_region region;region.slice=face;region.size=MTLSizeMake(4,4,1);texture_to_buffer(reader,image.texture,region,raw,face*1024,256);}reader.submit().wait();const auto data=map_buffer(raw,0,2048);
			for(unsigned face=0;face<2;++face)for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)for(unsigned c=0;c<4;++c){const auto word=((face||(pieces&&(x>=2||y>=2)))&&!(deferred&&face==1&&x<2&&y<2))?std::array{128u,64u,20u,255u}:latest?std::array{136u,119u,102u,255u}:std::array{85u,68u,51u,255u};require(data[face*1024+y*256+x*4+c]==std::byte(word[c]),"Owned RTT assembly lost native/VM face pixels");}
		};
		unsigned guards=0;const auto reject=[&](const auto& work){bool failed=false;try{work();}catch(const std::exception&){failed=true;}require(failed,"Invalid RTT assembly job published");++guards;};
		for(bool pieces:{false,true})for(unsigned mode=0;mode<9;++mode)
		{
			@autoreleasepool
			{
				auto f=make(3,pieces);const auto gate=[f->gpu->device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};auto job=f->textures->prepare_versioned_batch(std::array{f->registers,f->registers},*f->surfaces,std::array{guest_texture_upload_dependency{gate,1}});
				require(!job.ready()&&job.metadata(0).actual_mips==3&&f->textures->statistics().assemblies_created==1,"RTT pending group didnotshare assembly");reject([&]{(void)job.finish(std::chrono::milliseconds(1));});
				if(mode==1)
				{
					const auto other=[f->gpu->device() newSharedEvent];release unblock_other{other};auto r=f->registers;r.format=(r.format&0xffff)|(2u<<16);auto independent=f->textures->prepare_versioned_batch(std::array{r},*f->surfaces,std::array{guest_texture_upload_dependency{other,1}});require(!independent.ready()&&job.cancel_pending(),"RTT independent group borrowed cancelled sources");other.signaledValue=1;reject([&]{(void)job.finish();});const auto images=independent.finish();f->gpu->recover_completed_errors();verify(images[0].image,0,pieces);continue;
				}
				if(mode==2)++f->ram->version;
				if(mode==3){f->io->notify({0,0x100000});f->io->notify({0,0x100000});}
				gate.signaledValue=1;if(mode==2||mode==3){reject([&]{(void)job.finish();});require(f->textures->statistics().entries==0,"Stale RTT assembly published cache entry");continue;}
				if(mode>=7)
				{
					f->gpu->wait_idle();const auto frozen=job.finish();verify(frozen[0].image,0,pieces);const auto plan=plan_guest_texture(f->registers);const auto face=std::find_if(plan.subresources.begin(),plan.subresources.end(),[](const auto& sub){return sub.face==1&&sub.level==0;});require(face!=plan.subresources.end(),"Deferred RTT test lacks second cubeface");const unsigned address=0x1000+face->offset;const auto destination=f->surfaces->acquire({address,{2,2,1,surface_color::argb8,false},face->row_pitch,{}});f->surfaces->defer_copy(f->source,{0,0,2,2},destination,0,0);require(f->surfaces->status(destination).deferred_copies==1,"New deferred RTT owner didnotretain its copy");
					if(mode==8)f->surfaces->cancel_deferred_copies(destination);
					else{guest_surface_byte_cursor cursor(plan_guest_surface(address,2,2,4,face->row_pitch));while(const auto run=cursor.next())f->ram->forbidden.push_back(*run);}
					const auto fresh=f->textures->resolve_versioned(f->registers,*f->surfaces);verify(fresh.image,0,pieces,mode==7);require(fresh.image.texture!=frozen[0].image.texture&&f->surfaces->status(destination).deferred_copies==0,"Deferred RTT owner reused stale ready assembly orfailed materialization");verify(job.finish()[0].image,0,pieces);require(f->textures->resolve_versioned(f->registers,*f->surfaces).image.texture==fresh.image.texture,"Old RTT job displaced deferred/cancelled-owner ready result");continue;
				}
				if(mode==6)
				{
					unsigned captures=0;*f->after_capture=[&]{if(++captures==2)f->textures.reset();};const auto images=job.finish();require(!f->textures&&captures>=2,"RTT cachedeletion callback didnotrun during final publication");*f->after_capture={};verify(images[0].image,0,pieces);
				}
				else if(mode>=4)
				{
					f->gpu->wait_idle();if(mode==4)f->surfaces->clear_snapshots();else f->surfaces->clear(f->source,0xf0,0xff667788,0,255,{0,0,pieces?2u:4u,pieces?2u:4u});const auto fresh=f->textures->resolve_versioned(f->registers,*f->surfaces);verify(fresh.image,mode==5,pieces);
					const auto old=job.finish();verify(old[0].image,0,pieces);require(f->textures->resolve_versioned(f->registers,*f->surfaces).image.texture==fresh.image.texture,"Late RTT assembly replaced new cache/writer epoch");
				}
				else{const auto images=job.finish();require(images[0].image.texture==images[1].image.texture,"RTT coalesced views splitstorage");verify(images[0].image,0,pieces);}
				require(f->surfaces->status(f->source).dirty,"RTT job lost dirty source authority");
			}
		}
		for(bool pieces:{false,true})@autoreleasepool
		{
			auto f=make(3,pieces);const auto original=f->ram->bytes;const auto generation=f->surfaces->status(f->source).content_generation;const auto before=f->ledger->statistics();id<MTLBuffer> blocker=f->ledger->shared_buffer(before.budget-before.reserved_bytes);require(f->ledger->statistics().reserved_bytes==before.budget,"RTT quota test didnotfill its real native ledger");bool refused=false;
			try{(void)f->textures->prepare_versioned_batch(std::array{f->registers},*f->surfaces);}catch(const heap_budget_error&){refused=true;}
			require(refused&&f->textures->statistics().entries==0&&f->ram->bytes==original&&f->surfaces->status(f->source).dirty&&f->surfaces->status(f->source).content_generation==generation,"Full-ledger RTT admission published orchanged native authority");blocker=nil;const auto image=f->textures->resolve_versioned(f->registers,*f->surfaces);verify(image.image,0,pieces);require(f->ram->bytes==original,"Recovered RTT admission wrote VM");
		}
		for(bool pieces:{false,true})
		{
		std::shared_ptr<heap_budget> ledger;std::weak_ptr<memory> provider;
		@autoreleasepool
		{
			std::optional<versioned_guest_texture_job> orphan;id<MTLSharedEvent> gate;__weak id<MTLTexture> old_source=nil;
			@autoreleasepool
			{
				auto f=make(3,pieces);ledger=f->ledger;provider=f->ram;old_source=f->source_texture;gate=[f->gpu->device() newSharedEvent];orphan.emplace(f->textures->prepare_versioned_batch(std::array{f->registers},*f->surfaces,std::array{guest_texture_upload_dependency{gate,1}}));require(!orphan->ready(),"RTT orphan gate absent");f.reset();require(!provider.expired()&&old_source!=nil,"RTT orphan lost captured source/VM owners");
			}
			std::vector<versioned_guest_texture> images;@autoreleasepool{gate.signaledValue=1;images=orphan->finish();verify(images[0].image,0,pieces);}require(old_source==nil,"Completed RTT job stillpinned mutable source allocation");images.clear();orphan.reset();require(provider.expired(),"RTT orphan kept provider after last ticket");
		}
		require(ledger->statistics().reserved_bytes==0,"RTT orphan retained native snapshot/VM/cast/staging/output charges");
		}
		std::cout<<"RTT assembly owners:18 real gated groups(complete/partial native planes)/"<<guards<<" timeout-cancel-VMrevision-IOABA guards,wholework sharing,independent mipgroups survive cancellation,late cacheclear/nativewriter andcachedeletion callback publication preserve frozenpixels/newready;new deferred owners inold VMholes andcancellation invalidate ready sourceepochs before materialization;exact full-ledger refusal thenrecovery forcomplete/partial sources,gated cache/context destruction,completedjob releases mutable source,exact native+VM faces,provider retirement/zero oldledger charges passed\n";
	}
	void assembly_atlas_case()
	{
		context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);auto r=words(0x9b,vertex_texture_dimension::cube,true,8,8,1,4);const auto plan=plan_guest_texture(r);fill(plan,std::vector<unsigned>(16,0xa7));const auto original=ram->bytes;
		const auto pitch=plan.subresources[0].row_pitch,height=plan.storage_bytes/pitch;const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;};
		surface_cache surfaces(gpu,ram,8u<<20,ledger,capture);guest_texture_cache textures(gpu,ram,translate,1u<<20,ledger,64,{capture,io});const auto source=surfaces.acquire({0x1000,{8,height,1,surface_color::rgba32f,false},pitch,{}});surfaces.prepare(source,surface_initialization::background);
		const auto writer=compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
fragment float4 atlas_color(float4 p [[position]]) {const float f=as_type<float>(0x3e000000u+uint(p.y)*257u+uint(p.x)*7u);return float4(f,f*2,f*4,f*8);}
)metal",@"RTT atlas native bit writer");auto pd=[MTL4RenderPipelineDescriptor new];pd.vertexFunctionDescriptor=library_function(vertex,@"v");pd.fragmentFunctionDescriptor=library_function(writer,@"atlas_color");pd.inputPrimitiveTopology=MTLPrimitiveTopologyClassTriangle;pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA32Float;const auto pipeline=compile.render(pd);
		const std::array uses{cached_surface_use{source,true}};surfaces.draw(uses,[&](context& commands,auto images)
		{
			auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;pass.colorAttachments[0].storeAction=MTLStoreActionStore;raster_state raster;raster.viewport={0,0,8,double(height),0,1};raster.scissor={0,0,8,height};render_commands draw(commands,pass);draw.configure(pipeline,nil,raster,{},{});draw.draw(MTLPrimitiveTypeTriangle,0,3);draw.end();
		});guest_surface_byte_cursor cursor(plan_guest_surface(0x1000,8,height,16,pitch));while(const auto run=cursor.next())ram->forbidden.push_back(*run);
		auto gray=words(0x9c,vertex_texture_dimension::cube,true,8,8,1,4);gray.control3=(gray.control3&0xfff00000)|pitch;const auto reads=ram->reads;auto job=textures.prepare_versioned_batch(std::array{r,gray},surfaces);const auto images=job.finish();require(images.size()==2&&surfaces.snapshot_statistics().created==1,"One native atlas didnotshare a single fullsnapshot across two cube mipchains");std::size_t bits=0;
		const std::array<color,6> directions{{{1,0,0,1},{-1,0,0,1},{0,1,0,1},{0,-1,0,1},{0,0,1,1},{0,0,-1,1}}};
		for(unsigned kind=0;kind<2;++kind)for(const auto& s:plan.subresources)
		{
			const unsigned components=kind?1:4;texture_region region;region.level=s.level;region.slice=s.face;region.size=MTLSizeMake(s.width,s.height,1);const auto raw=make_buffer(gpu,256*s.height,buffer_storage::shared);gpu.begin();texture_to_buffer(gpu,images[kind].image.texture,region,raw,0,256);gpu.submit().wait();const auto bytes=map_buffer(raw,0,raw.length);
			for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)for(unsigned c=0;c<components;++c)
			{
				const auto source_x=kind?x/4:x,channel=kind?x%4:c;const unsigned wanted=0x3e000000+(s.offset/pitch+y)*257+source_x*7+channel*0x00800000;unsigned actual;std::memcpy(&actual,bytes.data()+y*256+(x*components+c)*4,4);require(actual==wanted,"Native atlas crop/cast/mip/face changed exactfloat bits");++bits;
			}
			const unsigned x=s.width/2,y=s.height/2;const unsigned base=0x3e000000+(s.offset/pitch+y)*257+(kind?x/4:x)*7;const auto value=[&](unsigned c){return std::bit_cast<float>(base+c*0x00800000);};const color expected=kind?color{value(x%4),value(x%4),value(x%4),value(x%4)}:color{value(0),value(1),value(2),value(3)};sample(images[kind].image,kind?gray:r,directions[s.face],expected,47,s.level);
		}
		const auto created=ledger->statistics().created;const auto encoded=gpu.encoding_statistics();const auto warm=textures.resolve_versioned_batch(std::array{r,gray},surfaces);require(warm[0].image.texture==images[0].image.texture&&warm[1].image.texture==images[1].image.texture&&ledger->statistics().created==created&&gpu.encoding_statistics().compute_encoders==encoded.compute_encoders&&gpu.encoding_statistics().render_encoders==encoded.render_encoders&&ram->reads==reads&&ram->bytes==original&&surfaces.status(source).dirty,"Native atlas warm composition recaptured/allocated/encoded orlostauthority");ram->forbidden.clear();std::cout<<"Native RTT atlas:one immutable RGBA32F8x90snapshot feeds24cropped faces/mips eachoftwo cube outputs (RGBA32Fdirect/R32Fpartial-row cast);"<<bits<<" exactbits/48independentRSX draws,nonzero sourceY/crops,sharedfullsnapshot,no VMpixelcapture/writeback,warm zeroallocation/encoding passed\n";
	}
	void assembly_deep_atlas_case()
	{
		context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);auto r=words(0x9b,vertex_texture_dimension::three,true,8,8,128,2);const auto plan=plan_guest_texture(r);fill(plan,std::vector<unsigned>(16,0xa7));const auto original=ram->bytes;
		const auto pitch=plan.subresources[0].row_pitch,height=plan.storage_bytes/pitch;const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;};surface_cache surfaces(gpu,ram,8u<<20,ledger,capture);guest_texture_cache textures(gpu,ram,translate,1u<<20,ledger,64,{capture,io});
		const auto source=surfaces.acquire({0x1000,{8,height,1,surface_color::rgba32f,false},pitch,{}});surfaces.prepare(source,surface_initialization::background);
		const auto writer=compile.library(R"metal(
#include <metal_stdlib>
using namespace metal;
fragment float4 deep_atlas_color(float4 p [[position]]) {const float f=as_type<float>(0x3e000000u+uint(p.y)*257u+uint(p.x)*7u);return float4(f,f*2,f*4,f*8);}
)metal",@"RTT deep atlas bit writer");auto pd=[MTL4RenderPipelineDescriptor new];pd.vertexFunctionDescriptor=library_function(vertex,@"v");pd.fragmentFunctionDescriptor=library_function(writer,@"deep_atlas_color");pd.inputPrimitiveTopology=MTLPrimitiveTopologyClassTriangle;pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA32Float;const auto pipeline=compile.render(pd);
		surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionDontCare;pass.colorAttachments[0].storeAction=MTLStoreActionStore;raster_state raster;raster.viewport={0,0,8,double(height),0,1};raster.scissor={0,0,8,height};render_commands draw(commands,pass);draw.configure(pipeline,nil,raster,{},{});draw.draw(MTLPrimitiveTypeTriangle,0,3);draw.end();});
		guest_surface_byte_cursor cursor(plan_guest_surface(0x1000,8,height,16,pitch));while(const auto run=cursor.next())ram->forbidden.push_back(*run);
		// Prewarm the complete immutable source,so each plane encounters a
		// ready snapshot wrapper rather than a still-pending copy owner.
		const auto atlas_words=words(0x9b,vertex_texture_dimension::two,true,8,height);const auto atlas=textures.resolve_versioned(atlas_words,surfaces);const auto reads=ram->reads;const auto snapshots=surfaces.snapshot_statistics().created;const auto image=textures.resolve_versioned(r,surfaces);require(surfaces.snapshot_statistics().created==snapshots&&snapshots==1,"Deep atlas copied a ready source again");unsigned planes=0;std::size_t bits=0;
		for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z)
		{
			texture_region area;area.level=s.level;area.origin=MTLOriginMake(0,0,z);area.size=MTLSizeMake(s.width,s.height,1);const auto output=make_buffer(gpu,256*s.height,buffer_storage::shared);gpu.begin();texture_to_buffer(gpu,image.image.texture,area,output,0,256);gpu.submit().wait();const auto data=map_buffer(output,0,output.length);
			for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)for(unsigned c=0;c<4;++c){unsigned actual;std::memcpy(&actual,data.data()+y*256+(x*4+c)*4,4);const unsigned wanted=0x3e000000+(s.offset/pitch+z*s.storage_rows+y)*257+x*7+c*0x800000;require(actual==wanted,"Deep atlas changed Z/mip/crop/floatbits");++bits;}++planes;
		}
		const auto allocations=ledger->statistics().created;const auto encoders=gpu.encoding_statistics();require(textures.resolve_versioned(r,surfaces).image.texture==image.image.texture&&ram->reads==reads&&ram->bytes==original&&ledger->statistics().created==allocations&&gpu.encoding_statistics().compute_encoders==encoders.compute_encoders&&gpu.encoding_statistics().render_encoders==encoders.render_encoders,"Deep atlas warm lookup recaptured/allocated/encoded");require(planes==192,"Deep atlas didnotexercise moreplanes than the64source budget");ram->forbidden.clear();(void)atlas;
		std::cout<<"Deep RTT atlas:one prewarmed immutable RGBA32F source feeds192Z/mipplanes with"<<bits<<" exactfloat words;ready wrappers deduplicated before64source budget,no source recopy/VMpixels/writeback,warm zeroallocation/capture/encoding passed\n";
	}
	void assembly_alias_pieces()
	{
		unsigned profiles=0;std::size_t checked=0;
		for(auto dimension:{vertex_texture_dimension::two,vertex_texture_dimension::cube})
		{
			@autoreleasepool
			{
				context gpu{3};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);auto source_memory=std::make_shared<memory>();auto r=words(0x85,dimension,true,8,8,1,2);r.offset=0x800ffff9;const auto plan=plan_guest_texture(r);source_memory->bytes.resize(plan.storage_bytes+64);for(std::size_t i=0;i<source_memory->bytes.size();++i)source_memory->bytes[i]=std::byte((i*7+19)&255);const auto original=source_memory->bytes;source_memory->aliases={{0xfffffff9,0,7},{0x90000,20,plan.storage_bytes-7}};
				const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[size=plan.storage_bytes](unsigned offset,unsigned location,std::size_t count){require(location==1&&offset>=0xffff9&&offset-0xffff9<=size&&count<=size-(offset-0xffff9),"Physical RTT translated outside its IO windows");return offset<0x100000?0xfffffff9u+offset-0xffff9:0x90000u+offset-0x100000;};surface_cache surfaces(gpu,source_memory,8u<<20,ledger,capture);guest_texture_cache textures(gpu,source_memory,translate,1u<<20,ledger,64,{capture,io});
				const auto a=surfaces.acquire({0x1000,{4,4,1,surface_color::argb8,false},32,{}});surfaces.prepare(a,surface_initialization::background);surfaces.clear(a,0xf0,0x66112233,0,255,{0,0,4,4});
				const auto b=surfaces.acquire({0x1010,{3,3,1,surface_color::abgr8,false},32,{}});surfaces.prepare(b,surface_initialization::background);const std::array uses{cached_surface_use{b,true}};surfaces.draw(uses,[&](context& commands,auto images){auto pass=[MTL4RenderPassDescriptor new];pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(68./255,85./255,102./255,136./255);(void)commands.render(pass);});
				for(const auto& description:std::array{surfaces.description(a),surfaces.description(b)}){guest_surface_byte_cursor cursor(plan_guest_surface(description.address,description.allocation.width,description.allocation.height,4,description.row_pitch));while(const auto run=cursor.next())source_memory->forbidden.push_back(*run);}
				const auto paint=[](auto& bytes,unsigned origin,unsigned width,unsigned height,const std::array<unsigned,4>& word){for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned c=0;c<4;++c)bytes[origin+y*32+x*4+c]=std::byte(word[c]);};auto old_bytes=original;paint(old_bytes,0,4,4,{102,17,34,51});paint(old_bytes,16,3,3,{136,102,85,68});
				const auto verify=[&](const resolved_fragment_texture& image,const auto& bytes)
				{
					context reader;for(const auto& s:plan.subresources){const auto raw=make_buffer(reader,256*s.height,buffer_storage::shared);texture_region region;region.level=s.level;region.slice=s.face;region.size=MTLSizeMake(s.width,s.height,1);reader.begin();texture_to_buffer(reader,image.texture,region,raw,0,256);reader.submit().wait();const auto data=map_buffer(raw,0,raw.length);for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)for(unsigned c=0;c<4;++c){const auto logical=s.offset+y*s.row_pitch+x*4+3-c,physical=logical<7?logical:20+logical-7;require(data[y*256+x*4+c]==bytes[physical],"Physical RTT partialtexel/ARGB-ABGR/newer-writer byte authority differed");++checked;}}
				};
				const auto gate=[gpu.device() newSharedEvent];struct unblock{id<MTLSharedEvent> gate;~unblock(){gate.signaledValue=1;}} release{gate};auto old=textures.prepare_versioned_batch(std::array{r},surfaces,std::array{guest_texture_upload_dependency{gate,1}});require(!old.ready(),"Physical RTT source snapshots were not gated");gate.signaledValue=1;gpu.wait_idle();surfaces.clear(a,0xf0,0xaa778899,0,255,{0,0,4,4});auto latest_bytes=old_bytes;paint(latest_bytes,0,4,4,{170,119,136,153});const auto fresh=textures.resolve_versioned(r,surfaces);verify(fresh.image,latest_bytes);const auto frozen=old.finish();verify(frozen[0].image,old_bytes);require(textures.resolve_versioned(r,surfaces).image.texture==fresh.image.texture&&source_memory->bytes==original,"Late physical RTT assembly replaced fresh cache entry orwroteVM");
				{auto other=std::make_shared<heap_budget>(gpu,8u<<20);guest_texture_cache mismatched(gpu,source_memory,translate,1u<<20,other,64,{capture,io});const auto reads=source_memory->reads;const auto created=ledger->statistics().created;bool refused=false;try{(void)mismatched.resolve(r,surfaces);}catch(const std::invalid_argument&){refused=true;}require(refused&&source_memory->reads==reads&&ledger->statistics().created==created&&other->statistics().created==0,"Mismatched RTT ledger captured VM orallocated native storage before refusal");}
				textures.clear();source_memory->allocation.reset();const auto reads=source_memory->reads;bool refused=false;try{(void)textures.resolve_versioned(r,surfaces);}catch(const std::invalid_argument&){refused=true;}require(refused&&source_memory->reads==reads,"Expired physical RTT identity captured pixels orpublished");source_memory->allocation=std::make_shared<int>(0);++profiles;
			}
		}
		std::cout<<"RTT physical pieces:"<<profiles<<" 2D/cube mipchains,"<<checked<<" exactGPUbytes;owned allocation metadata aliases withtop32-bit/midtexel IOsplit,disjointARGB/ABGR sources andnewer-writer authority,late frozenold result/newready retention,expired identity refusal andzero nativeVMcapture/writeback passed (controlled alias provider,no OS VM mapping claim)\n";
	}
	void assembly_piece_cases()
	{
		context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);unsigned profiles=0,draws=0;std::size_t bits=0,holes=0;
		for(unsigned format:{0x85u,0x9eu,0x82u,0x9du,0x8bu,0x98u,0x9au,0x9bu,0x9cu,0x91u,0x93u})for(auto dimension:{vertex_texture_dimension::two,vertex_texture_dimension::cube,vertex_texture_dimension::three})for(bool linear:{false,true})for(unsigned samples:{1u,2u,4u})
		{
			@autoreleasepool
			{
				const bool depth=format==0x91||format==0x93,packed=format==0x82||format==0x9d,gb=format==0x8b||format==0x98,floating=format==0x9a||format==0x9b||format==0x9c;
				const auto r=words(format,dimension,linear,8,8,dimension==vertex_texture_dimension::three?4:1,2);const auto plan=plan_guest_texture(r);const bool volume=guest_texture_depth_volume(plan);const auto bpp=plan.codec.block.bytes;
				std::vector<unsigned> vm;
				if(packed)vm={0x7b,0xde};else if(gb)vm={64,128};else if(format==0x91)vm={0x7e,0,0,42};else if(format==0x93)vm={0x60,0};else if(format==0x9a)vm={0x30,0,0x34,0,0x36,0,0x38,0};else if(format==0x9b)vm={0x3e,0,0,0,0x3e,0x80,0,0,0x3e,0xc0,0,0,0x3f,0,0,0};else if(format==0x9c)vm={0x3e,0,0,0};else vm={17,32,64,128};fill(plan,vm);const auto original=ram->bytes;
				const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;};surface_cache surfaces(gpu,ram,8u<<20,ledger,capture);guest_texture_cache textures(gpu,ram,translate,1u<<20,ledger,64,{capture,io});unsigned ordinal=0;std::size_t vm_bytes=0;
				const auto native_value=[](unsigned plane,unsigned part){return .50123456f+plane/256.f+part/32.f;};
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z,++ordinal)
				{
					vm_bytes+=(s.width*s.height-8)*bpp;
					for(unsigned part=0;part<2;++part)
					{
						const unsigned x=part?s.width-2:0,y=part?s.height-2:0;const auto factors=samples==1?sample_layout{1,1}:rsx_sample_layout(samples);
						const unsigned address=0x1000+s.offset+(linear?((z*s.storage_rows+y)*s.row_pitch+x*bpp):0);const auto swizzle=!linear?std::optional<swizzle_region>{{{s.columns,s.rows,s.depth},{x,y,z},{2,2,1}}}:std::nullopt;
						const surface_format sf=depth?surface_format{format==0x91?surface_depth::z24s8_float:surface_depth::z16_float}:surface_format{packed?(part?surface_color::x1rgb5_one:surface_color::x1rgb5_zero):gb?surface_color::gb8:format==0x9a?surface_color::rgba16f:format==0x9b?surface_color::rgba32f:format==0x9c?surface_color::r32f:part?surface_color::xrgb8_one:surface_color::xrgb8_zero};
						const auto source=surfaces.acquire({address,{2/factors.x,2/factors.y,samples,sf,false},linear?s.row_pitch:0,swizzle});surfaces.prepare(source,surface_initialization::background);
						if(depth||floating)
						{
							const auto v=native_value(ordinal,part);const std::array uses{cached_surface_use{source,true}};surfaces.draw(uses,[&](context& commands,auto images){auto pass=[MTL4RenderPassDescriptor new];if(depth){pass.depthAttachment.texture=images[0];pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=v;if(format==0x91){pass.stencilAttachment.texture=images[0];pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=50+part;}}else{pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(v,.25,.75,1);}(void)commands.render(pass);});
						}
						else surfaces.clear(source,0xf0,packed?0x4635:gb?0xff003344:0xff334455,0,255,{0,0,2/factors.x,2/factors.y});
						guest_surface_byte_cursor cursor(plan_guest_surface(address,2,2,bpp,linear?s.row_pitch:0,swizzle));while(const auto run=cursor.next())ram->forbidden.push_back(*run);
					}
				}
				const auto image=textures.resolve_versioned(r,surfaces);ordinal=0;
				for(const auto& s:plan.subresources)for(unsigned z=0;z<s.depth;++z,++ordinal)
				{
					const auto owner=[&](unsigned x,unsigned y){return x<2&&y<2?0:x>=s.width-2&&y>=s.height-2?1:2;};
					if(depth||floating)
					{
						const unsigned components=depth||format==0x9c?1:4,unit=format==0x9a?2:4;const auto raw=make_buffer(gpu,512*s.height,buffer_storage::shared);texture_region region;region.level=s.level;region.slice=s.face;region.origin=MTLOriginMake(0,0,z);region.size=MTLSizeMake(s.width,s.height,1);gpu.begin();texture_to_buffer(gpu,image.image.texture,region,raw,0,256,0,depth&&!volume?texture_plane::depth:texture_plane::all);if(format==0x91&&!volume)texture_to_buffer(gpu,image.image.texture,region,raw,256*s.height,256,0,texture_plane::stencil);gpu.submit().wait();const auto data=map_buffer(raw,0,raw.length);
						for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x)for(unsigned c=0;c<components;++c)
						{
							const auto part=owner(x,y);const float value=depth?(part<2?native_value(ordinal,part):.5f):part<2?color{native_value(ordinal,part),.25f,.75f,1}[c]:color{.125f,.25f,.375f,.5f}[c];unsigned actual=0;std::memcpy(&actual,data.data()+y*256+(x*(volume?2:components)+c)*unit,unit);const unsigned wanted=unit==2?std::bit_cast<std::uint16_t>(_Float16(value)):std::bit_cast<unsigned>(value);require(actual==wanted,"Partial RTT native/VM floatdepth components changed precision orpixel order");++bits;if(format==0x91){if(volume){unsigned stencil;std::memcpy(&stencil,data.data()+y*256+x*8+4,4);require(stencil==std::bit_cast<unsigned>(float(part<2?50+part:42)),"Partial Depth3D RTT independent stencil changed");}else require(data[256*s.height+y*256+x]==std::byte(part<2?50+part:42),"Partial RTT independent stencil changed");}
						}
					}
					else for(const auto point:std::array<std::array<std::size_t,2>,3>{{{0,0},{s.width-1,s.height-1},{s.width/2,0}}})
					{
						const auto part=owner(point[0],point[1]);color expected;
						if(packed){const auto cv=[](unsigned v){return std::floor(v/31.f*255)/255;};expected=part<2?color{cv((0x4635>>10)&31),cv((0x4635>>5)&31),cv(0x4635&31),float(part)}:color{cv(30),cv(30),cv(30),format==0x9d?1.f:0.f};}
						else if(gb)expected=part<2?color{68.f/255,51.f/255,68.f/255,51.f/255}:format==0x98?color{64.f/255,128.f/255,64.f/255,128.f/255}:color{128.f/255,64.f/255,128.f/255,64.f/255};else expected=part<2?color{51.f/255,68.f/255,85.f/255,float(part)}:color{32.f/255,64.f/255,128.f/255,format==0x9e?1.f:17.f/255};
						const float u=(point[0]+.5f)/s.width,v=(point[1]+.5f)/s.height,sc=2*u-1,tc=2*v-1;const std::array<color,6> cube{{{1,-tc,-sc,1},{-1,-tc,sc,1},{sc,1,tc,1},{sc,-1,-tc,1},{sc,-tc,1,1},{-sc,-tc,-1,1}}};sample(image.image,r,dimension==vertex_texture_dimension::cube?cube[s.face]:color{u,v,(z+.5f)/s.depth,1},expected,47,s.level);++draws;
					}
				}
				const auto created=ledger->statistics().created;const auto encoded=gpu.encoding_statistics();const auto reads=ram->reads;const auto warm=textures.resolve_versioned(r,surfaces);require(warm.image.texture==image.image.texture&&ram->reads==reads&&ram->bytes==original&&ledger->statistics().created==created&&gpu.encoding_statistics().compute_encoders==encoded.compute_encoders&&gpu.encoding_statistics().render_encoders==encoded.render_encoders,"Partial RTT warm result recaptured/allocated/encoded orchangedVM");require(textures.statistics().assembly_vm_bytes==vm_bytes,"Partial RTT didnotcapture exactly its active VM holes");holes+=vm_bytes;ram->forbidden.clear();++profiles;
			}
		}
		std::cout<<"RTT native pieces:"<<profiles<<" linear/Morton mip/cube/volume profiles,2native corners+VMholes perplane,1/2/4samples;"<<draws<<" independent RSX draws,"<<bits<<" exact float/depth components andindependent stencil,"<<holes<<" exactVMholebytes;X0/X1/targetconstantalpha/packed/GB maps,no nativeVMcapture/writeback,warm zeroallocation/capture/encoding passed\n";
	}
	void assembly_channel_cases()
	{
		context gpu{1};auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);gpu.set_buffer_budget(ledger);const auto begin=cases;unsigned profiles=0,packed_words=0;
		const std::array<color,6> directions{{{1,0,0,1},{-1,0,0,1},{0,1,0,1},{0,-1,0,1},{0,0,1,1},{0,0,-1,1}}};
		for(unsigned format:{0x82u,0x9du,0x8bu,0x98u})for(auto dimension:{vertex_texture_dimension::two,vertex_texture_dimension::cube})for(bool linear:{false,true})for(bool mixed:{false,true})
		{
			@autoreleasepool
			{
				const bool gb=format==0x8b||format==0x98;const auto r=words(format,dimension,linear,4,4,1,3);const auto plan=plan_guest_texture(r);fill(plan,gb?std::vector<unsigned>{64,128}:std::vector<unsigned>{0x7b,0xde});const auto original=ram->bytes;
				const cpu_capture_dispatch capture=[](const auto& work){work();};auto io=std::make_shared<io_mapping_epoch>();const auto translate=[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;};surface_cache surfaces(gpu,ram,8u<<20,ledger,capture);guest_texture_cache textures(gpu,ram,translate,1u<<20,ledger,64,{capture,io});std::vector<color> expected;unsigned index=0;
				for(const auto& s:plan.subresources)
				{
					const bool native=!mixed||index%3!=1;const auto source_format=gb?surface_color::gb8:index%2?surface_color::x1rgb5_one:surface_color::x1rgb5_zero;
					if(gb)expected.push_back(native?color{68.f/255,51.f/255,68.f/255,51.f/255}:format==0x98?color{64.f/255,128.f/255,64.f/255,128.f/255}:color{128.f/255,64.f/255,128.f/255,64.f/255});
					// Apple packed-unorm sampling expands to eight bits, as in the
					// existing RGB565 oracle. Also verify the exact storage word below.
					else {const auto component=[](unsigned value){return std::floor(value/31.f*255)/255;};expected.push_back(native?color{component((0x4635>>10)&31),component((0x4635>>5)&31),component(0x4635&31),index%2?1.f:0.f}:color{component(30),component(30),component(30),format==0x9d?1.f:0.f});}
					if(native)
					{
						const auto swizzle=!linear?std::optional<swizzle_region>{{{s.columns,s.rows,1},{},{s.columns,s.rows,1}}}:std::nullopt;const unsigned address=0x1000+s.offset;const auto source=surfaces.acquire({address,{s.width,s.height,1,source_format,false},linear?s.row_pitch:0,swizzle});surfaces.clear(source,0xf0,gb?0xff003344:0x4635,0,255,{0,0,s.width,s.height});guest_surface_byte_cursor cursor(plan_guest_surface(address,s.width,s.height,2,linear?s.row_pitch:0,swizzle));while(const auto run=cursor.next())ram->forbidden.push_back(*run);
					}
					++index;
				}
				const auto image=textures.resolve_versioned(r,surfaces);index=0;for(const auto& s:plan.subresources)
				{
					if(!gb)
					{
						const bool native=!mixed||index%3!=1;const unsigned wanted=native?(0x4635|(index%2?0x8000:0)):(0x7bde|(format==0x9d?0x8000:0));texture_region region;region.level=s.level;region.slice=s.face;region.size=MTLSizeMake(s.width,s.height,1);const auto raw=make_buffer(gpu,256*s.height,buffer_storage::shared);gpu.begin();texture_to_buffer(gpu,image.image.texture,region,raw,0,256);gpu.submit().wait();const auto bytes=map_buffer(raw,0,raw.length);
						for(unsigned y=0;y<s.height;++y)for(unsigned x=0;x<s.width;++x){std::uint16_t actual;std::memcpy(&actual,bytes.data()+y*256+x*2,2);require(actual==wanted,"Packed RTT normalization changed native five-bit components oralpha");++packed_words;}
					}
					sample(image.image,r,dimension==vertex_texture_dimension::cube?directions[s.face]:color{.25f,.25f,0,1},expected[index++],47,s.level);
				}
				require(ram->bytes==original&&textures.statistics().assemblies_created==1,"Packed/GB RTT composition wrote VM ormissednative assembly");ram->forbidden.clear();++profiles;
			}
		}
		std::cout<<"RTT assembly channel normalization:"<<profiles<<" packedX0-X1/GB8 profiles,"<<packed_words<<" exactpackedwords,"<<cases-begin<<" independent mip/cubeface RSX draws;2/4channel native andVM maps,targetconstant-alpha andduplicatedchannels preserved withoutVMnativecapture/writeback passed\n";
	}
	void all()
	{
		struct test
		{
			unsigned format;
			std::vector<unsigned> bytes;
			color expected;
		};
		const float u16 = 32768.f / 65535, lo = 16384.f / 65535, hi = 49152.f / 65535;
		const std::vector<test> tests = {
			{0x81, {128}, {128.f / 255, 128.f / 255, 128.f / 255, 1}}, {0x82, {0xfc, 0}, {1, 0, 0, 1}}, {0x83, {0xf1, 0x23}, {17.f / 255, 34.f / 255, 51.f / 255, 1}}, {0x84, {0xf8, 0}, {1, 0, 0, 1}}, {0x85, {255, 32, 64, 128}, {32.f / 255, 64.f / 255, 128.f / 255, 1}}, {0x86, {0, 0xf8, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 1}}, {0x87, {255, 255, 255, 255, 255, 255, 255, 255, 0, 0xf8, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 1}}, {0x88, {255, 255, 0, 0, 0, 0, 0, 0, 0, 0xf8, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 1}}, {0x8b, {32, 128}, {128.f / 255, 32.f / 255, 128.f / 255, 32.f / 255}}, {0x8d, {16, 32, 64, 128}, {32.f / 255, 64.f / 255, 16.f / 255, 1}}, {0x8e, {32, 16, 128, 64}, {32.f / 255, 64.f / 255, 16.f / 255, 1}}, {0x8f, {7, 0xe0}, {0, 250.f / 255, 0, 1}}, {0x90, {128, 0, 0, 17}, {.5f, .5f, .5f, .5f}}, {0x91, {0x7e, 0, 0, 17}, {.5f, .5f, .5f, .5f}}, {0x92, {128, 0}, {u16, u16, u16, u16}}, {0x93, {0x60, 0}, {.5f, .5f, .5f, .5f}}, {0x94, {128, 0}, {1, u16, 1, u16}}, {0x95, {0x40, 0, 0xc0, 0}, {lo, hi, lo, hi}}, {0x97, {0xf8, 1}, {1, 0, 0, 1}}, {0x98, {128, 64}, {128.f / 255, 64.f / 255, 128.f / 255, 64.f / 255}}, {0x99, {128, 64}, {-1, 64.f / 127, -1, 64.f / 127}}, {0x9a, {0x3c, 0, 0x38, 0, 0x34, 0, 0x3c, 0}, {1, .5f, .25f, 1}}, {0x9b, {0x3f, 128, 0, 0, 0x3f, 0, 0, 0, 0x3e, 128, 0, 0, 0x3f, 128, 0, 0}, {1, .5f, .25f, 1}}, {0x9c, {0x3f, 0, 0, 0}, {.5f, .5f, .5f, .5f}}, {0x9d, {0x7c, 0}, {1, 0, 0, 1}}, {0x9e, {0, 32, 64, 128}, {32.f / 255, 64.f / 255, 128.f / 255, 1}}, {0x9f, {0x38, 0, 0x34, 0}, {.25f, .5f, .25f, .5f}}};
		for (const auto& t : tests)
			for (bool linear : {true, false})
			{
				const auto r = words(t.format, vertex_texture_dimension::two, linear);
				const auto plan = plan_guest_texture(r);
				fill(plan, t.bytes);
				const auto before = ram->bytes;
				const auto resource = resolver.resolve(r);
				if (t.format == 0x90 || t.format == 0x91)
				{
					const auto stencil = make_buffer(ctx, 1024, buffer_storage::shared);
					ctx.begin();
					texture_region region;
					region.size = MTLSizeMake(4, 4, 1);
					texture_to_buffer(ctx, resource.texture, region, stencil, 0, 256, 0, texture_plane::stencil);
					ctx.submit().wait();
					for (unsigned y = 0; y < 4; ++y)
						for (unsigned x = 0; x < 4; ++x)
							require(static_cast<const std::uint8_t*>(stencil.contents)[y * 256 + x] == 17, "Guest packed depth upload lost stencil plane");
				}
				sample(resource, r, {.125f, .125f, 0, 1}, t.expected);
				require(before == ram->bytes, "Native texture upload changed VM bytes");
				const auto captured = capture_without_gpu(r);
				sample(realize_without_vm(captured), r, {.125f, .125f, 0, 1}, t.expected);
			}
		for(const auto& t : tests)
		{
			auto r=words(t.format,vertex_texture_dimension::two,true,8,8);++r.control3;
			const auto plan=plan_guest_texture(r);
			require(plan.subresources[0].row_pitch==(r.control3&0xfffff),"Linear guest texture pitch was truncated");
			fill(plan,t.bytes);const auto before=ram->bytes;
			auto right=t.expected;if(t.format==0x8d || t.format==0x8e) right[0]=128.f/255;
			sample(resolver.resolve(r),r,{.9375f,.9375f,0,1},right);
			require(ram->bytes==before,"Odd-pitch texture upload changed VM");
			const auto captured=capture_without_gpu(r);
			sample(realize_without_vm(captured),r,{.0625f,.9375f,0,1},t.expected);
		}
		std::cout << "54 independent odd-byte-pitch guest texture draws:all27formats,8x8 lower corners,BC/packed/depth/endian,owned capture/realize and VM preservation passed\n";
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::three, vertex_texture_dimension::cube})
			for (bool linear : {true, false})
			{
				const auto r = words(0x85, dimension, linear, 8, dimension == vertex_texture_dimension::one ? 1 : 8, dimension == vertex_texture_dimension::three ? 4 : 1, 4);
				const auto plan = plan_guest_texture(r);
				fill(plan, {255, 32, 64, 128});
				for (const auto& s : plan.subresources)
					for (std::size_t z = 0; z < s.depth; ++z)
						for (std::size_t y = 0; y < s.rows; ++y)
							for (std::size_t x = 0; x < s.columns; ++x)
							{
								std::size_t offset = s.offset + (z * s.rows + y) * s.row_pitch + x * 4;
								if (plan.swizzled)
									offset = s.offset + morton_index({x, y, z}, {unsigned(std::bit_width(s.columns) - 1), unsigned(std::bit_width(s.rows) - 1), unsigned(std::bit_width(s.depth) - 1)}) * 4;
								ram->bytes[offset + 1] = std::byte(32 + s.level * 32 + s.face + z * 4);
							}
				const auto captured = capture_without_gpu(r);
				const auto resource = realize_without_vm(captured);
				const std::array<color, 6> directions = {{{1, 0, 0, 1}, {-1, 0, 0, 1}, {0, 1, 0, 1}, {0, -1, 0, 1}, {0, 0, 1, 1}, {0, 0, -1, 1}}};
				for (unsigned level = 0; level < 4; ++level)
					for (unsigned selection = 0; selection < (dimension == vertex_texture_dimension::cube ? 6u : dimension == vertex_texture_dimension::three ? 2u
																																							  : 1u);
						 ++selection)
					{
						const unsigned d = std::max(1u, 4u >> level), slice = dimension == vertex_texture_dimension::three && selection ? d - 1 : 0, face = dimension == vertex_texture_dimension::cube ? selection : 0;
						sample(resource, r, dimension == vertex_texture_dimension::cube ? directions[selection] : color{.125f, .125f, selection ? .875f : .125f, 1}, {float(32 + level * 32 + face + slice * 4) / 255, 64.f / 255, 128.f / 255, 1}, 47, float(level));
					}
			}
		for (bool linear : {true, false})
		{
			auto r = words(0x85, vertex_texture_dimension::two, linear);
			r.format |= 0x40u << 8;
			const auto plan = plan_guest_texture(r);
			fill(plan, {255, 32, 64, 128});
			const auto& s = plan.subresources[0];
			for (std::size_t y = 0; y < 4; ++y)
				for (std::size_t x = 0; x < 4; ++x)
				{
					const auto at = linear ? y * s.row_pitch + x * 4 : morton_index({x, y, 0}, {2, 2, 0}) * 4;
					ram->bytes[at + 1] = std::byte(32 + x * 16);
				}
			const auto resource = resolver.resolve(r);
			sample(resource, r, {2.5f, 2.5f, 0, 1}, {64.f / 255, 64.f / 255, 128.f / 255, 1});
			sample(resource, r, {5, 5, 0, 2}, {64.f / 255, 64.f / 255, 128.f / 255, 1}, 24);
		}
		for (unsigned invalid = 0; invalid < 7; ++invalid)
		{
			auto r = words(0x85);
			if (invalid == 0)
				r.format &= ~8u;
			if (invalid == 1)
				r.control3 &= ~0xfffffu;
			if (invalid == 2)
			{
				r = words(0x85, vertex_texture_dimension::two, false, 3, 4);
			}
			if (invalid == 3)
				r = words(0x93, vertex_texture_dimension::three, true, 4, 4, 0);
			if (invalid == 4)
				r.format = (r.format & ~(255u << 8)) | (0xa0u << 8);
			if (invalid == 5)
				r = words(0x85, vertex_texture_dimension::two, true, 1024, 1024);
			if (invalid == 6)
				r.format &= ~3u;
			const auto reads = ram->reads;
			bool refused = false;
			try
			{
				(void)resolver.resolve(r);
			}
			catch (const std::exception&)
			{
				refused = true;
			}
			require(refused && reads == ram->reads, "Unsupported/invalid texture read VM before preflight");
		}
		const auto valid = words(0x85);
		fill(plan_guest_texture(valid), {255, 32, 64, 128});
		ram->conflict = true;
		bool rejected = false;
		try
		{
			(void)resolver.resolve(valid);
		}
		catch (const std::runtime_error&)
		{
			rejected = true;
		}
		require(rejected, "Concurrent VM revision change published stale texture");
		ram->conflict = false;
		std::cout << cases << " native guest-address/layout/VM->texture->RSX draws:27formats,linear/Morton,1D/2D/3D/cube,mips,endian/depth/packed/BC,VM-preservation and revision conflict passed\n";
	}
	void upload_work_cases()
	{
		struct profile { unsigned format; vertex_texture_dimension dimension; unsigned width, height, depth, mips; bool linear, odd, staged; unsigned direct; };
		using D = vertex_texture_dimension;
		const std::array profiles{
			profile{0x81,D::one,8,1,1,4,true,false,false,4}, profile{0x81,D::two,8,8,1,4,true,false,false,4},
			profile{0x86,D::cube,16,16,1,3,true,false,false,18}, profile{0x87,D::cube,16,16,1,3,true,false,false,18},
			profile{0x88,D::cube,16,16,1,3,true,false,false,18}, profile{0x86,D::cube,8,8,1,2,true,true,true,0},
			profile{0x85,D::cube,8,8,1,4,true,false,true,0}, profile{0x9a,D::three,8,8,4,4,false,false,true,0},
			profile{0x9b,D::three,7,5,3,3,true,true,true,0}, profile{0x8f,D::two,7,5,1,3,true,true,true,0},
			profile{0x8d,D::two,7,5,1,3,true,true,true,0}, profile{0x8e,D::two,7,5,1,3,true,true,true,0},
			profile{0x90,D::two,8,8,1,4,true,false,false,0}, profile{0x93,D::two,8,8,1,4,false,false,true,0},
			profile{0x81,D::two,7,5,1,3,true,true,false,3}, profile{0x85,D::two,8,8,1,4,false,false,true,0}};
		std::size_t comparisons = 0, direct_regions = 0, converted_regions = 0;
		for (const auto& t : profiles)
		{
			auto r = words(t.format,t.dimension,t.linear,t.width,t.height,t.depth,t.mips);
			if (t.odd) ++r.control3;
			const auto plan = plan_guest_texture(r);
			ram->bytes.resize(plan.storage_bytes); ++ram->version;
			for (std::size_t i = 0; i < ram->bytes.size(); ++i) ram->bytes[i] = std::byte((i * 37 + i / 3 + 0x71) & 255);
			const auto original = ram->bytes;
			const auto captured = resolver.capture(r);
			const auto before = resolver.upload_statistics(); const auto encoders = ctx.encoding_statistics();
			const auto image = realize_without_vm(captured);
			const auto after = resolver.upload_statistics(); const auto encoded = ctx.encoding_statistics();
			require(after.subresources - before.subresources == plan.subresources.size() && after.direct_blits - before.direct_blits == t.direct &&
				after.staging_buffers - before.staging_buffers == unsigned(t.staged) &&
				after.conversion_dispatches - before.conversion_dispatches == (t.staged ? plan.subresources.size() : 0) &&
				after.depth_subresources - before.depth_subresources == (plan.codec.depth ? plan.subresources.size() : 0) &&
				encoded.compute_encoders == encoders.compute_encoders + 1 && encoded.render_encoders == encoders.render_encoders,
				"Native upload did not use the expected direct/fused work andone scratch range/compute encoder across all subresources");
			std::size_t active = 0, blocks = 0;
			for (const auto& sub : plan.subresources) { blocks += sub.columns * sub.rows * sub.depth; active += sub.columns * sub.rows * sub.depth * plan.codec.block.bytes; }
			require(after.direct_bytes - before.direct_bytes == (t.direct ? active : 0) &&
				after.conversion_blocks - before.conversion_blocks == (t.staged ? blocks : 0), "Native upload work counters include padding bytes or per-byte dispatches");
			direct_regions += t.direct; converted_regions += t.staged ? plan.subresources.size() : 0;
			if (plan.codec.depth) continue; // Independent depth/stencil codec matrices cover numeric conversion.
			const auto native = storage_block(image.texture.pixelFormat);
			for (const auto& sub : plan.subresources)
			{
				const auto columns = (sub.width + native.width - 1) / native.width, rows = (sub.height + native.height - 1) / native.height;
				const auto output = make_buffer(ctx, 256 * rows * sub.depth, buffer_storage::shared);
				ctx.begin(); texture_to_buffer(ctx, image.texture, {sub.level, sub.face, MTLOriginMake(0,0,0), MTLSizeMake(sub.width,sub.height,sub.depth)}, output, 0, 256, 256 * rows); ctx.submit().wait();
				for (std::size_t z = 0; z < sub.depth; ++z) for (std::size_t y = 0; y < rows; ++y) for (std::size_t x = 0; x < columns; ++x)
				{
					const auto bx = plan.codec.conversion >= 2 ? x / 2 : x;
					std::size_t address = sub.offset + (z * sub.storage_rows + y) * sub.row_pitch + bx * plan.codec.block.bytes;
					if (plan.swizzled && plan.codec.block.width == 1)
						address = sub.offset + morton_index({bx,y,z},{unsigned(std::bit_width(sub.columns)-1),unsigned(std::bit_width(sub.rows)-1),unsigned(std::bit_width(sub.depth)-1)}) * plan.codec.block.bytes;
					std::array<std::byte,16> expected{};
					const auto byte = [&](unsigned lane) { return std::to_integer<unsigned>(original[address + lane]); };
					if (plan.codec.conversion == 1)
					{
						const auto raw = (byte(0) << 8) | byte(1); const auto value = (raw & 0xf81f) | ((raw & 0x3e0) << 1);
						expected[0] = std::byte(value); expected[1] = std::byte(value >> 8);
					}
					else if (plan.codec.conversion >= 2)
					{
						const bool br = plan.codec.conversion == 2;
						expected = {std::byte(byte(br ? 0 : 1)),std::byte(byte(br ? 2 : 3)),std::byte(byte((br ? 1 : 0) + (x % 2) * 2)),std::byte{255}};
					}
					else for (unsigned lane = 0; lane < native.bytes; ++lane) expected[lane] = original[address + (lane ^ (plan.codec.endian_width - 1))];
					require(std::memcmp(static_cast<const std::byte*>(output.contents) + (z * rows + y) * 256 + x * native.bytes, expected.data(), native.bytes) == 0,
						"Native direct/fused upload differs from independent guest-address/byte/packed-color oracle");
					comparisons += native.bytes;
				}
			}
		}
		std::cout << "Native upload work:16 profiles," << direct_regions << " direct subresources," << converted_regions << " fused subresources," << comparisons << " exact native bytes;R8/BC1/2/3 cube mips,no scratch;RGBA cube/3D Morton/odd pitches/packed odd widths use one scratch range/one compute encoder;owned source survives VM removal passed\n";
	}
	void upload_job_cases()
	{
		using namespace std::chrono_literals;
		using D = vertex_texture_dimension;
		std::shared_ptr<heap_budget> ledger;
		std::size_t compared = 0, subresources = 0;
		@autoreleasepool
		{
			std::optional<guest_texture_upload_job> job;
			id<MTLSharedEvent> gate;
			std::vector<guest_texture_layout> plans;
			std::vector<std::vector<std::byte>> expected;
			std::weak_ptr<memory> weak;
			{
				context upload_ctx(1);
				ledger = std::make_shared<heap_budget>(upload_ctx, 64u << 20);
				upload_ctx.set_buffer_budget(ledger);
				gate = [upload_ctx.device() newSharedEvent];
				auto source = std::make_shared<memory>(); weak = source;
				guest_texture_resolver uploader(upload_ctx, source, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, ledger);
				std::vector<captured_guest_texture> captured;
				for (unsigned i = 0; i < guest_texture_resolver::maximum_upload_images; ++i)
				{
					const auto r = i % 4 == 0 ? words(0x81,D::one,true,8,1,1,4) :
						i % 4 == 1 ? words(0x86,D::cube,true,8,8,1,2) :
						i % 4 == 2 ? words(0x9b,D::three,false,4,4,4,3) : words(0x85,D::two,true,7,5,1,3);
					plans.push_back(plan_guest_texture(r));
					source->bytes.resize(plans.back().storage_bytes); ++source->version;
					for (std::size_t b = 0; b < source->bytes.size(); ++b) source->bytes[b] = std::byte((b * 37 + i * 17 + b / 3) & 255);
					expected.push_back(source->bytes); captured.push_back(uploader.capture(r));
				}
				const auto before = upload_ctx.encoding_statistics();
				const std::array waits{guest_texture_upload_dependency{gate, 1}};
				job.emplace(uploader.enqueue_uploads(captured, waits));
				require(upload_ctx.encoding_statistics().compute_encoders == before.compute_encoders + 1 && !job->ready(),
					"Upload batch waited per image or used separate compute encoders");
				source->bytes.clear(); ++source->version; source->forbid_vm_access = true;
				captured.clear();
			}
			require(weak.expired() && ledger->statistics().live_buffers > 0, "Queued upload retained VM or lost its native source owners");
			const auto pending_bytes = ledger->statistics().reserved_bytes;
			bool timeout = false;
			try { (void)job->finish(1ms); } catch (const std::runtime_error&) { timeout = true; }
			require(timeout && !job->ready() && ledger->statistics().reserved_bytes == pending_bytes,
				"Upload timeout published images or retired queued sources/targets");
			gate.signaledValue = 1;
			auto images = job->finish();
			require(images.size() == plans.size() && job->ready() && !job->cancel_pending() && ledger->statistics().live_buffers == 0,
				"Successful completion lost images or retained orphan upload buffers");
			struct readback { std::size_t image, subresource; id<MTLBuffer> buffer; };
			std::vector<readback> outputs;
			ctx.begin();
			for (std::size_t i = 0; i < images.size(); ++i)
			{
				require(images[i].metadata.actual_mips == plans[i].mips && images[i].texture.device == ctx.device(), "Batch metadata/device changed");
				for (std::size_t s = 0; s < plans[i].subresources.size(); ++s)
				{
					const auto& sub = plans[i].subresources[s];
					const auto block = storage_block(images[i].texture.pixelFormat);
					const auto rows = (sub.height + block.height - 1) / block.height;
					const auto output = make_buffer(ctx, 256 * rows * sub.depth, buffer_storage::shared);
					texture_to_buffer(ctx, images[i].texture, {sub.level,sub.face,MTLOriginMake(0,0,0),MTLSizeMake(sub.width,sub.height,sub.depth)}, output, 0, 256, 256 * rows);
					outputs.push_back({i,s,output});
				}
			}
			ctx.submit().wait();
			for (const auto& output : outputs)
			{
				const auto& plan = plans[output.image]; const auto& sub = plan.subresources[output.subresource];
				for (std::size_t z = 0; z < sub.depth; ++z) for (std::size_t y = 0; y < sub.rows; ++y) for (std::size_t x = 0; x < sub.columns; ++x)
				{
					auto address = sub.offset + (z * sub.storage_rows + y) * sub.row_pitch + x * plan.codec.block.bytes;
					if (plan.swizzled && plan.codec.block.width == 1)
						address = sub.offset + morton_index({x,y,z},{unsigned(std::bit_width(sub.columns)-1),unsigned(std::bit_width(sub.rows)-1),unsigned(std::bit_width(sub.depth)-1)}) * plan.codec.block.bytes;
					for (unsigned b = 0; b < plan.codec.block.bytes; ++b)
					{
						require(static_cast<const std::byte*>(output.buffer.contents)[(z * sub.rows + y) * 256 + x * plan.codec.block.bytes + b] ==
							expected[output.image][address + (b ^ (plan.codec.endian_width - 1))], "Detached upload batch changed captured source bytes");
						++compared;
					}
				}
				++subresources;
			}
		}
		require(ledger->statistics().reserved_bytes == 0, "Released completed upload batch retained native charges");
		std::cout << "Native upload job:64 images," << subresources << " subresources," << compared << " exact bytes;one encoder/submission,unsignalled dependency returns,pending CPU timeout preserves charges,resolver/context/VM/capture teardown before completion passed\n";

		// Preflight cannot allocate or cancel a caller's existing recording.
		context local(1, 2, 100ms);
		auto budget = std::make_shared<heap_budget>(local, 8u << 20); local.set_buffer_budget(budget);
		auto source = std::make_shared<memory>(); source->bytes.assign(64, std::byte{0x73});
		guest_texture_resolver uploader(local, source, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, budget);
		auto r = words(0x81,D::two,true,2,2); r.control3 = 1u << 20 | 2;
		std::vector<captured_guest_texture> captures;
		for (unsigned i = 0; i <= guest_texture_resolver::maximum_upload_images; ++i) captures.push_back(uploader.capture(r));
		const auto first = std::span<const captured_guest_texture>(captures).first(1);
		unsigned refused = 0;
		const auto rejects = [&](const auto& call)
		{
			const auto prior = budget->statistics().created;
			bool failed = false;
			try { call(); } catch (const std::exception&) { failed = true; }
			require(failed && budget->statistics().created == prior, "Invalid upload batch allocated or escaped preflight"); ++refused;
		};
		rejects([&] { (void)uploader.enqueue_uploads({}); });
		rejects([&] { (void)uploader.enqueue_uploads(captures); });
		const std::array invalid_wait{guest_texture_upload_dependency{nil,1}};
		rejects([&] { (void)uploader.enqueue_uploads(first, invalid_wait); });
		std::array<guest_texture_upload_dependency,65> too_many{};
		rejects([&] { (void)uploader.enqueue_uploads(first, too_many); });
		guest_texture_resolver limited(local, source, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1, budget);
		rejects([&] { (void)limited.enqueue_uploads(first); });
		context no_ledger;
		guest_texture_resolver missing(no_ledger, source, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, budget);
		rejects([&] { (void)missing.enqueue_uploads(first); });
		auto foreign_budget = std::make_shared<heap_budget>(local, 8u << 20);
		guest_texture_resolver foreign(local, source, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, foreign_budget);
		rejects([&] { (void)foreign.enqueue_uploads(first); });
		require(foreign_budget->statistics().created == 0, "Foreign upload ledger allocated resources");
		struct epoch_provider final : exclusive_guest_memory, exclusive_guest_memory::view
		{
			void execute(const std::function<void(view&)>& operation) override { operation(*this); }
			std::byte* map(guest_surface_range, bool) override { throw std::logic_error("Unexpected VM mapping in upload job test"); }
			bool same_storage(guest_surface_range, guest_surface_range) override { return false; }
			void written(guest_surface_range) noexcept override {}
		} provider;
		{
			guest_memory_epoch epoch(provider, provider);
			rejects([&] { (void)uploader.enqueue_uploads(first); });
		}
		local.begin(); const auto caller = local.recording_submission();
		rejects([&] { (void)uploader.enqueue_uploads(first); });
		require(caller.same_batch(local.recording_submission()) && !caller.ready(), "Nested batch cancelled caller recording"); local.cancel();
		auto moved_capture = std::move(captures.back());
		rejects([&] { (void)uploader.enqueue_uploads(std::span<const captured_guest_texture>(captures).last(1)); });
		const auto run_cancelled = [&](bool shutdown, bool deadline)
		{
			std::optional<guest_texture_upload_job> pending;
			@autoreleasepool
			{
				const auto gate = [local.device() newSharedEvent];
				const std::array waits{guest_texture_upload_dependency{gate,1}};
				pending.emplace(uploader.enqueue_uploads(first, waits));
				{
					guest_memory_epoch epoch(provider, provider);
					bool refused = false;
					try { (void)pending->finish(1ms); } catch (const std::logic_error&) { refused = true; }
					require(refused && !pending->ready(), "Upload completion waited/published inside VM exclusion");
				}
				if (shutdown) local.shutdown();
				else if (!deadline) require(pending->cancel_pending(), "Queued native upload cancellation refused");
				bool failed = false;
				try { (void)pending->finish(5s); } catch (const std::runtime_error& error)
				{
					const std::string message = error.what();
					failed = message.find(deadline ? "deadline expired" : "cancelled before Metal4 commit") != std::string::npos;
				}
				require(failed && pending->ready() && !pending->cancel_pending(), "Failed native upload published images or lost its completion error");
				auto moved = std::move(*pending);
				bool rejected = false;
				try { (void)pending->ready(); } catch (const std::logic_error&) { rejected = true; }
				require(rejected && moved.ready(), "Moved upload job lost its completion or exposed a moved state");
				pending.reset();
			}
			require(budget->statistics().reserved_bytes == 0, "Cancelled upload retained orphan native charges");
			if (!shutdown) local.recover_completed_errors();
		};
		run_cancelled(false,false); run_cancelled(false,true); run_cancelled(true,false);
		rejects([&] { (void)uploader.enqueue_uploads(first); });
		std::cout << "Native upload jobs:bounded/empty/moved/invalid-dependency/missing-or-foreign-ledger/VM-epoch/nested/stopped preflight;explicit cancel,event deadline,shutdown and moved-job ownership passed (" << refused << " preflight refusals)\n";

		std::size_t single_charge;
		@autoreleasepool
		{
			auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:2 height:2 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto buffer = make_buffer(ctx, 4, buffer_storage::shared);
			auto probe_budget = std::make_shared<heap_budget>(ctx, 8u << 20);
			const auto placed = probe_budget->texture(descriptor);
			single_charge = placed.heap.size + std::max(buffer.length, buffer.allocatedSize);
		}
		context tight_ctx(1);
		auto tight_budget = std::make_shared<heap_budget>(tight_ctx, single_charge); tight_ctx.set_buffer_budget(tight_budget);
		guest_texture_resolver tight(tight_ctx, source, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, tight_budget);
		@autoreleasepool
		{
			bool failed = false;
			try { (void)tight.enqueue_uploads(std::span<const captured_guest_texture>(captures).first(2)); }
			catch (const heap_budget_error&) { failed = true; }
			require(failed && tight_budget->statistics().created == 2 && tight_budget->statistics().reserved_bytes == 0,
				"Upload quota did not fail after the first image or kept cancelled resources until the caller pool drains");
		}
		require(tight_budget->statistics().reserved_bytes == 0, "Partially encoded batch failed to roll back native resources");
		id<MTLSharedEvent> dropped_gate = [tight_ctx.device() newSharedEvent];
		@autoreleasepool
		{
			const std::array waits{guest_texture_upload_dependency{dropped_gate,1}};
			const auto abandoned = tight.enqueue_uploads(first, waits);
			require(!abandoned.ready(), "Gated replacement upload unexpectedly completed");
		}
		require(tight_budget->statistics().reserved_bytes == single_charge, "Dropping the upload job released queued native resources");
		dropped_gate.signaledValue = 1; tight_ctx.wait_idle();
		require(tight_budget->statistics().reserved_bytes == 0, "Dropped upload job retained resources after actual completion");
		@autoreleasepool
		{
			const auto completed = tight.enqueue_uploads(first).finish();
			require(completed.size() == 1 && tight_budget->statistics().live_buffers == 0, "Recovered context failed a later valid upload");
			const auto output = make_buffer(ctx, 512, buffer_storage::shared);
			ctx.begin(); texture_to_buffer(ctx, completed[0].texture, {0,0,MTLOriginMake(0,0,0),MTLSizeMake(2,2,1)}, output, 0, 256); ctx.submit().wait();
			for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x)
				require(static_cast<const std::byte*>(output.contents)[y * 256 + x] == std::byte{0x73}, "Later upload after quota rollback changed its captured pixels");
		}
		require(tight_budget->statistics().reserved_bytes == 0, "Released recovered upload retained native charges");
		std::cout << "Native upload job:exact full-ledger failure after first image rolls back without publication;discarded gated job retains source/target until completion;later slot reuse uploads exact pixels with zero orphan charges passed\n";
	}
	void captured_lifetime_cases()
	{
		auto r = words(0x85);
		r.format |= 0x40u << 8;
		std::weak_ptr<memory> weak;
		auto captured = [&]
		{
			auto local = std::make_shared<memory>();
			weak = local;
			local->bytes.assign(plan_guest_texture(r).storage_bytes, std::byte{255});
			guest_texture_resolver temporary(ctx, local, [](unsigned, unsigned, std::size_t)
											 { return 0x1000; }, 1 << 20, upload_budget);
			return temporary.capture(r);
		}();
		require(weak.expired(), "CPU capture retained its VM provider or resolver");
		require(captured.metadata().coordinates.scale[0] == .25f && captured.metadata().actual_mips == 1,
				"CPU capture lost frozen UN/program metadata");
		const auto resource = resolver.realize(captured);
		sample(resource, r, {.5f, .5f, 0, 1}, {1, 1, 1, 1});
		auto moved = std::move(captured);
		const auto reserved = upload_budget->statistics().reserved_bytes;
		bool refused = false;
		try
		{
			(void)resolver.realize(captured);
		}
		catch (const std::invalid_argument&)
		{
			refused = true;
		}
		require(refused && upload_budget->statistics().reserved_bytes == reserved, "Moved-from capture allocated native storage");
		guest_texture_resolver small(ctx, ram, [](unsigned, unsigned, std::size_t) -> unsigned
									 { throw std::runtime_error("Realize translated VM"); }, 1, upload_budget);
		refused = false;
		try
		{
			(void)small.realize(moved);
		}
		catch (const std::length_error&)
		{
			refused = true;
		}
		require(refused && upload_budget->statistics().reserved_bytes == reserved, "Destination capture limit allocated native storage");
		sample(resolver.realize(moved), r, {.5f, .5f, 0, 1}, {1, 1, 1, 1});
		fill(plan_guest_texture(r), {255, 32, 64, 128});
		ram->short_read = true;
		ctx.begin();
		refused = false;
		try
		{
			(void)resolver.capture(r);
		}
		catch (const std::runtime_error&)
		{
			refused = true;
		}
		ctx.cancel();
		ram->short_read = false;
		require(refused, "Short VM read published a CPU capture");
		std::cout << "CPU-only texture capture/all27formats/Morton/mips/faces survive VM removal; realization has zero VM access; provider teardown/move/limits/short reads passed\n";
	}
	void versioned_cases()
	{
		const auto r = words(0x85);
		fill(plan_guest_texture(r), {255, 32, 64, 128});
		unsigned address = 0x1000;
		auto budget = std::make_shared<heap_budget>(ctx, 64 << 20);
		guest_texture_cache textures(ctx, ram, [&](unsigned, unsigned, std::size_t) { return address; }, 1 << 20, budget);
		surface_cache surfaces(ctx, ram, 64 << 20, budget);
		const auto original = textures.resolve_versioned(r, surfaces);
		ctx.begin();
		require(original.revision.current(), "Fresh native VM ticket is stale");
		ctx.cancel();
		address = 0x2000;
		require(!original.revision.current(), "IO translation change retained native texture ticket");
		address = 0x1000;
		require(original.revision.current(), "Restored unchanged IO mapping invalidated original bytes");
		fill(plan_guest_texture(r), {255, 96, 64, 128});
		require(!original.revision.current(), "VM mutation retained old prepared texture ticket");
		sample(original.image, r, {.125f, .125f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		const auto replacement = textures.resolve_versioned(r, surfaces);
		require(replacement.revision.current(), "Replacement native texture ticket is stale");
		sample(replacement.image, r, {.125f, .125f, 0, 1}, {96.f / 255, 64.f / 255, 128.f / 255, 1});
		const auto source = surfaces.acquire({0x1000, {4, 4, 1, surface_color::argb8, false}, 24, {}});
		surfaces.clear(source, 0xf0, 0xff224466, 0, 255, {0, 0, 4, 4});
		const auto reads = ram->reads;
		const auto native = textures.resolve_versioned(r, surfaces);
		require(native.revision.current() && ram->reads == reads && textures.statistics().surface_snapshots == 1,
			"Versioned dirty native source fell back to VM or lost ticket");
		++ram->version;
		require(!native.revision.current(), "CPU change retained dirty native snapshot ticket");
		sample(native.image, r, {.125f, .125f, 0, 1}, {34.f / 255, 68.f / 255, 102.f / 255, 1});
		ram->conflict = true;
		surface_cache idle(ctx, ram, 64 << 20, budget);
		bool refused = false;
		try { (void)textures.resolve_versioned(r, idle); }
		catch (const texture_revision_conflict&) { refused = true; }
		ram->conflict = false;
		require(refused, "Changing VM version published a prepared texture ticket");
		std::cout << "Versioned VM/native-dirty texture snapshots:CPU-only validation,IO mapping/VM changes,old/new pixels and preparation conflict refusal passed\n";
	}
	void verify_native_depth(id<MTLTexture> texture, float expected)
	{
		if (texture.pixelFormat != MTLPixelFormatDepth32Float && texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
			return;
		const auto raw = make_buffer(ctx, 1024, buffer_storage::shared);
		texture_region whole;
		whole.size = MTLSizeMake(4, 4, 1);
		ctx.begin();
		texture_to_buffer(ctx, texture, whole, raw, 0, 256, 0, texture_plane::depth);
		ctx.submit().wait();
		// Exact float bits expose even sub-tolerance guest codec loss.
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				require(static_cast<const std::uint32_t*>(raw.contents)[y * 64 + x] == std::bit_cast<std::uint32_t>(expected), "Native surface depth snapshot quantized source float bits");
		if (texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
		{
			ctx.begin();
			texture_to_buffer(ctx, texture, whole, raw, 0, 256, 0, texture_plane::stencil);
			ctx.submit().wait();
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 4; ++x)
					require(static_cast<const unsigned char*>(raw.contents)[y * 256 + x] == 17, "Native depth alias lost source stencil");
		}
	}
	void partial_surface_cases()
	{
		std::vector<surface_format> formats;
		for(unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(c));
		for(unsigned d=1;d<=4;++d) formats.push_back(static_cast<surface_depth>(d));
		surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);
		unsigned profiles=0,comparisons=0,draws=0,roundtrips=0;
		for(const auto& format : formats)
		for(unsigned target_format : {0x81u,0x82u,0x83u,0x84u,0x85u,0x8bu,0x8fu,0x90u,0x91u,0x92u,0x93u,0x94u,0x95u,0x97u,0x98u,0x99u,0x9au,0x9bu,0x9cu,0x9du,0x9eu,0x9fu})
		{
			const auto bytes=std::holds_alternative<surface_color>(format)?color_layout(std::get<surface_color>(format)).pixel_bytes:describe_surface_depth(std::get<surface_depth>(format)).guest_bytes;
			const auto codec=describe_guest_texture_format(target_format);
			for(unsigned prefix=0;prefix<bytes;++prefix)
			for(unsigned width : {1u,3u})
			for(unsigned sample_count : {1u,2u,4u})
			{
				const unsigned row=width*codec.block.bytes;
				if((!prefix && row%bytes==0) || (sample_count>1 && (prefix!=bytes-1 || width!=3 || (target_format!=0x85 && target_format!=0x92)))) continue;
				const unsigned source_width=128/bytes,height=8,crop_y=2,byte_x=bytes+prefix,pitch=prefix%2?161:160;
				auto r=words(target_format,vertex_texture_dimension::two,true,width,3);r.control3=pitch;
				const auto plan=plan_guest_texture(r);
				std::vector<std::byte> source_bytes(128*height);
				for(unsigned y=0;y<height;++y)
					for(unsigned x=0;x<128;x+=2) { source_bytes[y*128+x]=std::byte(0x34+((x/2+y)%3)*2);source_bytes[y*128+x+1]=std::byte((x/2+y)&7); }
				auto reference_words=r;reference_words.control3=row;
				ram->bytes.assign(row*3,std::byte{});++ram->version;
				for(unsigned y=0;y<3;++y) std::copy_n(source_bytes.begin()+(crop_y+y)*128+byte_x,row,ram->bytes.begin()+y*row);
				const auto reference=resolver.resolve(reference_words);
				ram->bytes.assign(pitch*height,std::byte{0xe7});++ram->version;
				auto budget=std::make_shared<heap_budget>(ctx,8<<20);
				resolved_fragment_texture image;
				{
					surface_cache surfaces(ctx,ram,8<<20,budget);
					const auto factors=sample_count==1?sample_layout{1,1}:rsx_sample_layout(sample_count);
					cached_surface_description description{0x1000,{source_width/factors.x,height/factors.y,sample_count,format,false},pitch,{}};
					auto surface=surfaces.acquire(description);surfaces.prepare(surface,surface_initialization::background);
					const auto input=make_upload(ctx,source_bytes);
					surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto native)
					{
						id<MTLTexture> expanded=native[0];
						if(sample_count>1)
						{
							auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native[0].pixelFormat width:source_width height:height mipmapped:NO];
							d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;expanded=make_texture(commands,d);
						}
						texture_region region;region.size=MTLSizeMake(source_width,height,1);
						if(const auto color=std::get_if<surface_color>(&format)) colors.upload(commands,*color,input,{0,128},expanded,region);
						else depths.upload(commands,std::get<surface_depth>(format),input,{0,128},expanded,region);
						if(sample_count>1) samples.import(commands,expanded,native[0]);
					});
					const bool roundtrip=(target_format==0x81 && width==1 && prefix==bytes-1) || (target_format==0x92 && width==3 && prefix==bytes-1 && sample_count>1);
					if(roundtrip)
					{
						ram->allow_writeback=true;surfaces.flush(surface);ram->allow_writeback=false;
						for(unsigned y=0;y<height;++y)
						{
							require(std::equal(source_bytes.begin()+y*128,source_bytes.begin()+(y+1)*128,ram->bytes.begin()+y*pitch),"Odd-pitch surface scatter changed guest active bytes");
							for(unsigned x=128;x<pitch;++x) require(ram->bytes[y*pitch+x]==std::byte{0xe7},"Odd-pitch surface scatter overwrote guest padding");
						}
						surfaces.erase(surface);surface=surfaces.acquire(description);surfaces.prepare(surface);
						require(!surfaces.status(surface).dirty,"Odd-pitch VM reload retained dirty source state");++roundtrips;
					}

					r.offset+=crop_y*pitch+byte_x;
					guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return 0x1000u+offset-0x200;},1<<20,budget);
					const auto reads=ram->reads;const auto version=ram->version;
					auto prepared=textures.resolve_versioned(r,surfaces);image=prepared.image;
					require(prepared.revision.current() && ram->reads==reads && ram->version==version && surfaces.status(surface).dirty==!roundtrip && textures.statistics().uploads==0,
						"Partial RTT read/wrote VM pixels or lost its source ticket");
					require(image.texture.width==width && image.texture.height==3 && image.metadata.native_argb_channels==codec.base_argb,
						"Partial RTT retained source dimensions or native X/channel semantics");
					// Selected-prefix publication and reset are independent of decoder output.
					std::optional<surface_format> selected_format;std::optional<std::size_t> selected_prefix;std::shared_ptr<const surface_texture_revision> revision;
					const auto coverage=std::array{guest_byte_coverage({static_cast<unsigned>(0x1000+crop_y*pitch+byte_x),plan.storage_bytes})};
					const auto selected=surfaces.snapshot_texture_mapping(coverage,[&](const auto&)->std::optional<surface_texture_region>{return surface_texture_region{{1,crop_y,(prefix+row+bytes-1)/bytes,3},prefix};},budget,&revision,false,&selected_format,&selected_prefix);
					require(selected && selected_prefix==prefix && selected_format==format && revision && revision->current(),"Partial snapshot published wrong selected prefix/format/revision");
					selected_prefix=99;selected_format=surface_color::rgba32f;
					const auto absent=surfaces.snapshot_texture_mapping(std::array{guest_byte_coverage({0x70000,4})},[](const auto&)->std::optional<surface_texture_region>{return {};},budget,&revision,false,&selected_format,&selected_prefix);
					require(!absent && !selected_prefix && !selected_format && !revision,"Nil partial snapshot retained metadata");
					selected_prefix=99;selected_format=surface_color::rgba32f;bool refused=false;
					try{(void)surfaces.snapshot_texture_mapping(coverage,[&](const auto&)->std::optional<surface_texture_region>{return surface_texture_region{{0,0,1,1},bytes};},budget,&revision,false,&selected_format,&selected_prefix);}
					catch(const std::out_of_range&){refused=true;}
					require(refused && !selected_prefix && !selected_format && !revision && ram->reads==reads,"Invalid partial snapshot published metadata or read VM");
					surfaces.spill(surface);++ram->version;require(!prepared.revision.current(),"Partial RTT ticket survived source mutation");
				}
				for(const auto plane : image.texture.pixelFormat==MTLPixelFormatDepth32Float_Stencil8?std::vector{texture_plane::depth,texture_plane::stencil}:std::vector{codec.depth?texture_plane::depth:texture_plane::all})
				{
					const auto actual=make_buffer(ctx,1536,buffer_storage::shared),expected=make_buffer(ctx,1536,buffer_storage::shared);
					texture_region region;region.size=MTLSizeMake(width,3,1);
					ctx.begin();texture_to_buffer(ctx,image.texture,region,actual,0,512,0,plane);texture_to_buffer(ctx,reference.texture,region,expected,0,512,0,plane);ctx.submit().wait();
					for(unsigned y=0;y<3;++y)
						if(std::memcmp(static_cast<const std::byte*>(actual.contents)+y*512,static_cast<const std::byte*>(expected.contents)+y*512,width*storage_block(image.texture.pixelFormat,plane).bytes))
							throw std::runtime_error("Partial RTT mismatch target="+std::to_string(target_format)+" prefix="+std::to_string(prefix)+" samples="+std::to_string(sample_count));
					++comparisons;
				}
				if(target_format==0x81 || target_format==0x85 || target_format==0x92)
					for(unsigned corner : {0u,1u})
					{
						const auto position=(crop_y+corner*2)*128+byte_x+corner*(row-codec.block.bytes);
						const auto byte=[&](unsigned i){return std::to_integer<unsigned>(source_bytes[position+i]);};color expected;
						if(target_format==0x81) expected={byte(0)/255.f,byte(0)/255.f,byte(0)/255.f,1};
						else if(target_format==0x85) expected={byte(1)/255.f,byte(2)/255.f,byte(3)/255.f,byte(0)/255.f};
						else expected.fill(((byte(0)<<8)|byte(1))/65535.f);
						sample(image,r,corner?color{1-.5f/width,1-.5f/3,0,1}:color{.5f/width,.5f/3,0,1},expected);++draws;
					}
				++profiles;
			}
		}
		std::cout << profiles << " GPU partial-byte RTT profiles," << comparisons << " bit-exact native planes," << draws << " independent RSX draws," << roundtrips << " explicit VM roundtrips:17multi-byte sources/22target formats,all prefixes,partial endings,even/odd pitches,representative MSAA,VM poison,no RTT pixel read/writeback,padding-preserving VM gather/scatter,selected-prefix publication/reset,spill/mutation/cache teardown passed\n";
	}
	void block_surface_cases()
	{
		std::vector<surface_format> formats;
		for(unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(c));
		for(unsigned d=1;d<=4;++d) formats.push_back(static_cast<surface_depth>(d));
		surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);
		unsigned profiles=0,planes=0,draws=0;
		for(const auto& format : formats)
		for(unsigned target_format : {0x86u,0x87u,0x88u,0x8du,0x8eu})
		{
			const unsigned bpp=std::holds_alternative<surface_color>(format)?color_layout(std::get<surface_color>(format)).pixel_bytes:describe_surface_depth(std::get<surface_depth>(format)).guest_bytes;
			const auto codec=describe_guest_texture_format(target_format);
			for(unsigned prefix=0;prefix<bpp;++prefix)
			for(const auto shape : {std::pair{1u,1u},{3u,5u},{4u,4u},{5u,3u},{7u,9u},{8u,8u},{9u,7u}})
			for(unsigned sample_count : {1u,2u,4u})
			{
				if(sample_count>1 && (prefix!=bpp-1 || shape!=std::pair{7u,9u})) continue;
				const auto [width,height]=shape;
				const unsigned columns=(width+codec.block.width-1)/codec.block.width,rows=(height+codec.block.height-1)/codec.block.height,row=columns*codec.block.bytes;
				const unsigned pitch=prefix%2?161:160,source_width=128/bpp,source_height=12,byte_x=bpp+prefix,crop_y=2;
				auto r=words(target_format,vertex_texture_dimension::two,true,width,height);r.control3=pitch;
				std::vector<std::byte> source_bytes(128*source_height);
				for(unsigned y=0;y<source_height;++y)
					for(unsigned x=0;x<128;x+=2){source_bytes[y*128+x]=std::byte(0x34+((x/2+y)%3)*2);source_bytes[y*128+x+1]=std::byte((x/2+y)&7);}
				auto reference_words=r;reference_words.control3=row;
				ram->bytes.assign(row*rows,std::byte{});++ram->version;
				for(unsigned y=0;y<rows;++y) std::copy_n(source_bytes.begin()+(crop_y+y)*128+byte_x,row,ram->bytes.begin()+y*row);
				const auto reference=resolver.resolve(reference_words);
				ram->bytes.assign(pitch*source_height,std::byte{0xe7});++ram->version;
				auto budget=std::make_shared<heap_budget>(ctx,8<<20);resolved_fragment_texture image;
				{
					surface_cache surfaces(ctx,ram,8<<20,budget);
					const auto factors=sample_count==1?sample_layout{1,1}:rsx_sample_layout(sample_count);
					const auto surface=surfaces.acquire({0x1000,{source_width/factors.x,source_height/factors.y,sample_count,format,false},pitch,{}});
					surfaces.prepare(surface,surface_initialization::background);
					const auto input=make_upload(ctx,source_bytes);
					surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto native)
					{
						id<MTLTexture> expanded=native[0];
						if(sample_count>1)
						{
							auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native[0].pixelFormat width:source_width height:source_height mipmapped:NO];
							d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;expanded=make_texture(commands,d);
						}
						texture_region region;region.size=MTLSizeMake(source_width,source_height,1);
						if(const auto color=std::get_if<surface_color>(&format)) colors.upload(commands,*color,input,{0,128},expanded,region);
						else depths.upload(commands,std::get<surface_depth>(format),input,{0,128},expanded,region);
						if(sample_count>1) samples.import(commands,expanded,native[0]);
					});
					r.offset+=crop_y*pitch+byte_x;
					guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return 0x1000u+offset-0x200;},1<<20,budget);
					const auto reads=ram->reads;const auto version=ram->version;
					auto prepared=textures.resolve_versioned(r,surfaces);image=prepared.image;
					require(prepared.revision.current() && ram->reads==reads && ram->version==version && surfaces.status(surface).dirty && textures.statistics().uploads==0,
						"Block RTT fell back to VM pixels/writeback or lost source revision");
					require(image.texture.width==width && image.texture.height==height && image.metadata.native_argb_channels==codec.base_argb && image.texture.pixelFormat==reference.texture.pixelFormat,
						"Block RTT lost target dimensions/format/channels");
					surfaces.spill(surface);++ram->version;require(!prepared.revision.current(),"Block RTT ticket survived source mutation");
				}
				const auto actual=make_buffer(ctx,rows*512,buffer_storage::shared),expected=make_buffer(ctx,rows*512,buffer_storage::shared);
				texture_region region;region.size=MTLSizeMake(width,height,1);
				ctx.begin();texture_to_buffer(ctx,image.texture,region,actual,0,512);texture_to_buffer(ctx,reference.texture,region,expected,0,512);ctx.submit().wait();
				const unsigned native_row=codec.block.width==2?width*4:row,native_rows=codec.block.width==2?height:rows;
				for(unsigned y=0;y<native_rows;++y)
					require(!std::memcmp(static_cast<const std::byte*>(actual.contents)+y*512,static_cast<const std::byte*>(expected.contents)+y*512,native_row),"Block RTT changed native compressed bytes/packed output");
				++planes;
				if(codec.block.width==2)
					for(unsigned corner : {0u,1u})
					{
						const unsigned x=corner*(width-1),y=corner*(height-1),p=(crop_y+y)*128+byte_x+(x/2)*4;
						const auto byte=[&](unsigned i){return std::to_integer<unsigned>(source_bytes[p+i])/255.f;};
						const color expected_color=target_format==0x8d?color{byte(1+(x%2)*2),byte(2),byte(0),1}:color{byte((x%2)*2),byte(3),byte(1),1};
						sample(image,r,{(x+.5f)/width,(y+.5f)/height,0,1},expected_color);++draws;
					}
				++profiles;
			}
		}
		unsigned known_profiles=0,known_draws=0;
		for(unsigned target_format : {0x86u,0x87u,0x88u})
		for(surface_color source_format : {surface_color::b8,surface_color::argb8})
		for(unsigned prefix : {0u,1u,3u})
		for(const auto shape : {std::pair{1u,1u},{3u,5u},{4u,4u},{5u,3u},{7u,9u},{8u,8u},{9u,7u}})
		for(unsigned sample_count : {1u,2u,4u})
		{
			const auto [width,height]=shape;
			const unsigned columns=(width+3)/4,rows=(height+3)/4,bb=target_format==0x86?8:16;
			const unsigned bpp=source_format==surface_color::b8?1:4,pitch=prefix%2?161:160,byte_x=4+prefix;
			std::vector<std::byte> bytes(128*12,std::byte{0x2a});
			for(unsigned by=0;by<rows;++by)
				for(unsigned bx=0;bx<columns;++bx)
				{
					const unsigned base=(2+by)*128+byte_x+bx*bb,color_base=base+(target_format==0x86?0:8);
					const bool swap=(bx+by)%2;
					const unsigned a=swap?0x001f:0xf800,b=swap?0xf800:0x001f;
					bytes[color_base]=std::byte(a);bytes[color_base+1]=std::byte(a>>8);bytes[color_base+2]=std::byte(b);bytes[color_base+3]=std::byte(b>>8);
					std::uint32_t indices=0;std::uint64_t alpha=0;
					for(unsigned y=0;y<4;++y)
						for(unsigned x=0;x<4;++x)
						{
							const unsigned i=y*4+x,selector=(x+y)%2;indices|=selector<<(i*2);
							if(target_format==0x87) alpha|=std::uint64_t(selector?0:15)<<(i*4);
							if(target_format==0x88) alpha|=std::uint64_t(selector)<<(i*3);
						}
					for(unsigned i=0;i<4;++i) bytes[color_base+4+i]=std::byte(indices>>(i*8));
					if(target_format==0x87) for(unsigned i=0;i<8;++i) bytes[base+i]=std::byte(alpha>>(i*8));
					if(target_format==0x88)
					{
						bytes[base]=std::byte{255};bytes[base+1]=std::byte{0};
						for(unsigned i=0;i<6;++i) bytes[base+2+i]=std::byte(alpha>>(i*8));
					}
				}
			ram->bytes.assign(pitch*12,std::byte{0xe7});++ram->version;
			auto budget=std::make_shared<heap_budget>(ctx,8<<20);resolved_fragment_texture image;
			{
				surface_cache surfaces(ctx,ram,8<<20,budget);const auto factors=sample_count==1?sample_layout{1,1}:rsx_sample_layout(sample_count);
				const auto surface=surfaces.acquire({0x1000,{128/bpp/factors.x,12/factors.y,sample_count,source_format,false},pitch,{}});
				surfaces.prepare(surface,surface_initialization::background);const auto input=make_upload(ctx,bytes);
				surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto native)
				{
					id<MTLTexture> expanded=native[0];
					if(sample_count>1)
					{
						auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native[0].pixelFormat width:128/bpp height:12 mipmapped:NO];
						d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget;expanded=make_texture(commands,d);
					}
					texture_region region;region.size=MTLSizeMake(128/bpp,12,1);colors.upload(commands,source_format,input,{0,128},expanded,region);
					if(sample_count>1) samples.import(commands,expanded,native[0]);
				});
				auto r=words(target_format,vertex_texture_dimension::two,true,width,height);r.control3=pitch;r.offset+=2*pitch+byte_x;
				guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return 0x1000u+offset-0x200;},1<<20,budget);
				const auto reads=ram->reads;const auto version=ram->version;
				auto prepared=textures.resolve_versioned(r,surfaces);image=prepared.image;
				require(prepared.revision.current() && ram->reads==reads && ram->version==version && textures.statistics().uploads==0,"Known BC RTT read/wrote VM or lost ticket");
				surfaces.spill(surface);++ram->version;require(!prepared.revision.current(),"Known BC RTT ticket survived mutation");
			}
			auto r=words(target_format,vertex_texture_dimension::two,true,width,height);r.control3=pitch;
			for(const auto point : {std::pair{0u,0u},{width-1,0u},{0u,height-1},{width-1,height-1},{std::min(4u,width-1),std::min(4u,height-1)}})
			{
				const auto [x,y]=point;const bool blue=(x/4+y/4+(x+y)%2)%2;
				const float alpha=target_format==0x86?1:float((x+y)%2==0);
				sample(image,r,{(x+.5f)/width,(y+.5f)/height,0,1},blue?color{0,0,1,alpha}:color{1,0,0,alpha});++known_draws;
			}
			++known_profiles;
		}
		std::cout << known_profiles << " known BC RTT profiles," << known_draws << " independent endpoint/alpha RSX draws:BC1/BC2/BC3,checker indices,alternating block colors,partial edges,odd pitches,byte offsets,B8/ARGB8 sources,1/2/4samples and destroyed source caches passed\n";
		std::cout << profiles << " GPU block RTT profiles," << planes << " exact native compressed/packed planes," << draws << " independent paired-pixel RSX draws:18sources/BC1/BC2/BC3/BR_GR/RB_RG,all prefixes,seven edge shapes,even/odd pitches,1/2/4samples,VM poison,no VM pixel reads/writeback,spill/mutation/cache teardown passed\n";
	}
	void typeless_surface_cases()
	{
		context batch_ctx(1);
		auto batch_budget = std::make_shared<heap_budget>(batch_ctx,8u << 20); batch_ctx.set_buffer_budget(batch_budget);
		std::vector<surface_format> formats;
		for (unsigned c : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(c));
		for (unsigned d = 1; d <= 4; ++d) formats.push_back(static_cast<surface_depth>(d));
		surface_color_transfer colors(ctx);surface_depth_transfer depths(ctx);sample_transfer samples(ctx);
		unsigned profiles = 0, comparisons = 0, draws = 0;
		for (const auto& format : formats)
		for (unsigned target_format : {0x81u,0x82u,0x83u,0x84u,0x85u,0x8bu,0x8fu,0x90u,0x91u,0x92u,0x93u,0x94u,0x95u,0x97u,0x98u,0x99u,0x9au,0x9bu,0x9cu,0x9du,0x9eu,0x9fu})
		for (bool linear : {true, false})
		for (unsigned sample_count : {1u,2u,4u})
		{
			@autoreleasepool
			{
			const auto bytes = std::holds_alternative<surface_color>(format) ? color_layout(std::get<surface_color>(format)).pixel_bytes :
				describe_surface_depth(std::get<surface_depth>(format)).guest_bytes;
			const auto codec = describe_guest_texture_format(target_format);
			if ((!linear && bytes != codec.block.bytes) || (sample_count > 1 && target_format != 0x85 && target_format != 0x92)) continue;
			const unsigned width = 64 / bytes, height = 8, crop_x = linear ? 16 / bytes : 0, crop_y = linear ? 2 : 0;
			const unsigned row_bytes = linear ? 32 : 64, crop_height = linear ? 2 : 8;
			auto r = words(target_format,vertex_texture_dimension::two,linear,row_bytes/codec.block.bytes,crop_height);
			r.control3 = linear ? 96 : 0;
			const auto plan = plan_guest_texture(r);
			if (surface_texture_channels(format,plan)) continue; // Existing native-preserving fixtures cover these.
			std::vector<std::byte> source_bytes(64*height);
			for (unsigned y = 0; y < height; ++y)
				for (unsigned x = 0; x < 64; x += 2)
				{ source_bytes[y*64+x]=std::byte(0x34+((x/2+y)%3)*2);source_bytes[y*64+x+1]=std::byte((x/2+y)&7); }
			// Independently gather guest bytes for the cropped target. The VM
			// reference uses tight linear rows,regardless of source addressing.
			auto reference_words = r;reference_words.format |= 0x20u<<8;reference_words.control3 = row_bytes;
			const auto reference_plan = plan_guest_texture(reference_words);
			ram->bytes.assign(reference_plan.storage_bytes,std::byte{});++ram->version;
			for (unsigned y = 0; y < crop_height; ++y)
				std::copy_n(source_bytes.begin()+(crop_y+y)*64+crop_x*bytes,row_bytes,ram->bytes.begin()+y*row_bytes);
			const auto reference = resolver.resolve(reference_words);
			ram->bytes.assign(96*height,std::byte{0xe7});++ram->version;
			auto budget = batch_budget;
			resolved_fragment_texture image;
			{
				surface_cache surfaces(batch_ctx,ram,8<<20,budget);
				const auto factors = sample_count == 1 ? sample_layout{1,1} : rsx_sample_layout(sample_count);
				cached_surface_description description{0x1000,{width/factors.x,height/factors.y,sample_count,format,false},linear?96u:0u,{}};
				if (!linear) description.morton = swizzle_region{{width,height,1},{},{width,height,1}};
				const auto surface = surfaces.acquire(description);
				surfaces.prepare(surface,surface_initialization::background);
				const auto input = make_upload(batch_ctx,source_bytes);
				surfaces.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto native)
				{
					id<MTLTexture> expanded = native[0];
					if (sample_count > 1)
					{
						auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native[0].pixelFormat width:width height:height mipmapped:NO];
						d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;
						expanded=make_texture(commands,d);
					}
					texture_region region;region.size=MTLSizeMake(width,height,1);
					if (const auto color = std::get_if<surface_color>(&format)) colors.upload(commands,*color,input,{0,64},expanded,region);
					else depths.upload(commands,std::get<surface_depth>(format),input,{0,64},expanded,region);
					if (sample_count > 1) samples.import(commands,expanded,native[0]);
				});
				guest_texture_cache textures(batch_ctx,ram,[](unsigned offset,unsigned,std::size_t) { return 0x1000u+offset-0x200; },1<<20,budget);
				const auto morton = [](unsigned x,unsigned y,unsigned w,unsigned h)
				{ unsigned v=0,weight=1;for(unsigned b=1;b<std::max(w,h);b*=2) { if(b<w){if(x&b)v+=weight;weight*=2;}if(b<h){if(y&b)v+=weight;weight*=2;} }return v; };
				r.offset += linear ? crop_y*96+crop_x*bytes : morton(crop_x,crop_y,width,height)*bytes;
				const auto reads=ram->reads;const auto version=ram->version;
				std::array requests{r,r,r,r}; requests[1].format |= 0x40u << 8;
				const auto source_submits = surfaces.transaction_statistics().submitted;
				const auto gate = [batch_ctx.device() newSharedEvent];
				struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } gate_guard{gate};
				const std::array waits{guest_texture_upload_dependency{gate,1}};
				auto pending_batch = textures.prepare_versioned_batch(requests,surfaces,waits);
				require(!pending_batch.ready() && pending_batch.current() && surfaces.snapshot_statistics().created == 0 &&
					surfaces.snapshot_statistics().entries == 0 && surfaces.transaction_statistics().submitted == source_submits && textures.statistics().cast_entries == 0 &&
					pending_batch.metadata(1).coordinates.scale[0] == 1.f / plan.width,
					"Cold typeless source copy/MSAA expansion waited or published before its common cast batch completed");
				gate.signaledValue = 1;
				auto prepared_batch = pending_batch.finish();
				auto& prepared = prepared_batch[0]; image=prepared.image;
				require(textures.statistics().upload_batches == 1 && textures.statistics().casts_created == 1 && textures.statistics().coalesced_casts == 3 &&
					prepared_batch[1].image.texture == image.texture && prepared_batch[2].image.texture == image.texture && prepared_batch[3].image.texture == image.texture,
					"Batched typeless casts did not share one submitted conversion");
				for (unsigned i = 0; i < prepared_batch.size(); ++i)
					require(prepared_batch[i].revision.current() && prepared_batch[i].image.metadata.coordinates.scale[0] == (i == 1 ? 1.f / plan.width : 1.f),
						"Batched typeless cast lost a request's revision or UN metadata");
				require(prepared.revision.current() && ram->reads==reads && ram->version==version && surfaces.status(surface).dirty &&
					textures.statistics().uploads==0 && textures.statistics().surface_snapshots==4 && surfaces.snapshot_statistics().created==1,
					"Typeless RTT touched VM pixels/writeback,copied a repeated source or lost its source ticket");
				require(image.metadata.native_argb_channels==codec.base_argb && image.texture.width==plan.width && image.texture.height==crop_height &&
					image.texture.pixelFormat==reference.texture.pixelFormat,"Typeless RTT retained source channels/extent/type");
				const auto copies = surfaces.snapshot_statistics().created;
				const auto jobs = surfaces.transaction_statistics().submitted;
				const auto cast_jobs = textures.statistics().casts_created;
				const auto repeated = textures.resolve_versioned(r, surfaces);
				require(repeated.image.texture == image.texture && repeated.revision.current() &&
					surfaces.snapshot_statistics().created == copies && surfaces.transaction_statistics().submitted == jobs &&
					textures.statistics().casts_created == cast_jobs && textures.statistics().casts_reused == 1,
					"Repeated typeless RTT recopied/converted its immutable native source or lost revision ownership");
				surfaces.spill(surface);
				++ram->version;require(!prepared.revision.current(),"Typeless RTT source mutation retained a current ticket");
			}
			for (const auto plane : image.texture.pixelFormat==MTLPixelFormatDepth32Float_Stencil8 ? std::vector{texture_plane::depth,texture_plane::stencil} :
				std::vector{codec.depth?texture_plane::depth:texture_plane::all})
			{
				const auto actual=make_buffer(ctx,4096,buffer_storage::shared),expected=make_buffer(ctx,4096,buffer_storage::shared);
				texture_region region;region.size=MTLSizeMake(plan.width,crop_height,1);
				ctx.begin();texture_to_buffer(ctx,image.texture,region,actual,0,512,0,plane);texture_to_buffer(ctx,reference.texture,region,expected,0,512,0,plane);ctx.submit().wait();
				const auto count=plan.width*storage_block(image.texture.pixelFormat,plane).bytes;
				for(unsigned y=0;y<crop_height;++y)
					if(std::memcmp(static_cast<const std::byte*>(actual.contents)+y*512,static_cast<const std::byte*>(expected.contents)+y*512,count))
						throw std::runtime_error("Typeless RTT byte mismatch source="+std::to_string(std::holds_alternative<surface_color>(format)?unsigned(std::get<surface_color>(format)):100+unsigned(std::get<surface_depth>(format)))+" target="+std::to_string(target_format)+" linear="+std::to_string(linear)+" samples="+std::to_string(sample_count));
				++comparisons;
			}
			if (linear && (target_format==0x81 || target_format==0x85 || target_format==0x92))
				for (unsigned corner : {0u,1u})
				{
					const auto position=(crop_y+corner*(crop_height-1))*64+crop_x*bytes+corner*(row_bytes-codec.block.bytes);
					const auto byte=[&](unsigned i) { return std::to_integer<unsigned>(source_bytes[position+i]); };
					color expected;
					if(target_format==0x81) expected={byte(0)/255.f,byte(0)/255.f,byte(0)/255.f,1};
					else if(target_format==0x85) expected={byte(1)/255.f,byte(2)/255.f,byte(3)/255.f,byte(0)/255.f};
					else expected.fill(((byte(0)<<8)|byte(1))/65535.f);
					sample(image,r,corner?color{1-.5f/plan.width,1-.5f/plan.height,0,1}:color{.5f/plan.width,.5f/plan.height,0,1},expected);++draws;
				}
			++profiles;
			}
		}
		(void)batch_ctx.trim_upload_pages();
		std::cout << profiles << " batched GPU-only typeless RTT profiles and " << comparisons << " bit-exact native plane comparisons and " << draws << " independent RSX sampler draws:18source/22target formats,linear different-BPP/Morton equal-BPP,crops,1/2/4samples,four requests share one cast/submit,UN metadata,target maps,VM poison,no pixel reads/writeback,revision mutation,spill and destroyed-cache ownership passed\n";
	}
	void surface_alias_cases()
	{
		const auto translate = [](unsigned, unsigned, std::size_t)
		{ return 0x1000u; };
		struct alias_case
		{
			unsigned format;
			surface_format surface;
			color clear;
			color expected;
		};
		const float depth = .50123456f;
		const float d16 = std::round(depth * 65535) / 65535;
		const std::vector<alias_case> cases = {
			{0x81, surface_color::b8, {128.f / 255, 0, 0, 1}, {128.f / 255, 128.f / 255, 128.f / 255, 1}},
			{0x84, surface_color::rgb565, {1, 0, 1, 1}, {1, 0, 1, 1}},
			{0x85, surface_color::argb8, {32.f / 255, 64.f / 255, 128.f / 255, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1}},
			{0x8b, surface_color::gb8, {64.f / 255, 128.f / 255, 0, 1}, {64.f / 255, 128.f / 255, 64.f / 255, 128.f / 255}},
			{0x9a, surface_color::rgba16f, {1, .5f, .25f, 1}, {1, .5f, .25f, 1}},
			{0x9b, surface_color::rgba32f, {1, .5f, .25f, 1}, {1, .5f, .25f, 1}},
			{0x9c, surface_color::r32f, {.25f, 0, 0, 1}, {.25f, .25f, .25f, .25f}},
			{0x90, surface_depth::z24s8_uint, {}, {depth, depth, depth, depth}},
			{0x91, surface_depth::z24s8_float, {}, {depth, depth, depth, depth}},
			{0x92, surface_depth::z16_uint, {}, {d16, d16, d16, d16}},
			{0x93, surface_depth::z16_float, {}, {depth, depth, depth, depth}},
			{0x82, surface_color::x1rgb5_zero, {1, 0, 1, 1}, {1, 0, 1, 0}},
			{0x82, surface_color::x1rgb5_one, {1, 0, 1, 0}, {1, 0, 1, 1}},
			{0x9d, surface_color::x1rgb5_zero, {1, 0, 1, 0}, {1, 0, 1, 0}},
			{0x9d, surface_color::x1rgb5_one, {1, 0, 1, 0}, {1, 0, 1, 1}},
			{0x85, surface_color::xrgb8_zero, {32.f/255, 64.f/255, 128.f/255, .6f}, {32.f/255, 64.f/255, 128.f/255, 0}},
			{0x85, surface_color::xrgb8_one, {32.f/255, 64.f/255, 128.f/255, .6f}, {32.f/255, 64.f/255, 128.f/255, 1}},
			{0x85, surface_color::abgr8, {32.f/255, 64.f/255, 128.f/255, .6f}, {128.f/255, 64.f/255, 32.f/255, .6f}},
			{0x85, surface_color::xbgr8_zero, {32.f/255, 64.f/255, 128.f/255, .6f}, {128.f/255, 64.f/255, 32.f/255, .6f}},
			{0x85, surface_color::xbgr8_one, {32.f/255, 64.f/255, 128.f/255, .6f}, {128.f/255, 64.f/255, 32.f/255, .6f}},
			{0x9e, surface_color::argb8, {32.f/255, 64.f/255, 128.f/255, .6f}, {32.f/255, 64.f/255, 128.f/255, .6f}},
			{0x9e, surface_color::xrgb8_zero, {32.f/255, 64.f/255, 128.f/255, .6f}, {32.f/255, 64.f/255, 128.f/255, 0}},
			{0x9e, surface_color::xrgb8_one, {32.f/255, 64.f/255, 128.f/255, .6f}, {32.f/255, 64.f/255, 128.f/255, 1}},
			{0x9e, surface_color::abgr8, {32.f/255, 64.f/255, 128.f/255, .6f}, {128.f/255, 64.f/255, 32.f/255, 1}},
			{0x9e, surface_color::xbgr8_zero, {32.f/255, 64.f/255, 128.f/255, .6f}, {128.f/255, 64.f/255, 32.f/255, 1}},
			{0x9e, surface_color::xbgr8_one, {32.f/255, 64.f/255, 128.f/255, .6f}, {128.f/255, 64.f/255, 32.f/255, 1}},
			{0x98, surface_color::gb8, {64.f/255, 128.f/255, 0, 1}, {64.f/255, 128.f/255, 64.f/255, 128.f/255}}};
		unsigned draws = 0;
		for (const unsigned samples : {1u, 2u, 4u})
		for (const auto& c : cases)
			for (bool linear : {true, false})
			{
				// Native assembly jobs share one actual buffer/texture ledger.
				context ctx{1};
				auto r = words(c.format, vertex_texture_dimension::two, linear);
				const auto plan = plan_guest_texture(r);
				fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
				auto budget = std::make_shared<heap_budget>(ctx, 4 << 20);ctx.set_buffer_budget(budget);
				resolved_fragment_texture first, second, deferred, clean;
				{
					surface_cache surfaces(ctx, ram, 4 << 20, budget);
					guest_texture_cache textures(ctx, ram, translate, 1 << 20, budget);
					const auto stale = textures.resolve(r);
					require(textures.resolve(r, surfaces).texture == stale.texture, "Surface-aware VM fallback missed a coherent cache hit");
					cached_surface_description description{0x1000, {4, 4, 1, c.surface, false}, linear ? plan.subresources[0].row_pitch : 0, {}};
					if (!linear)
						description.morton = swizzle_region{{4, 4, 1}, {}, {4, 4, 1}};
					const auto factors = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
					description.allocation.width /= factors.x;
					description.allocation.height /= factors.y;
					description.allocation.samples = samples;
					const auto target_surface = surfaces.acquire(description);
					const std::array uses{cached_surface_use{target_surface, true}};
					const auto clear_native = [&](color value, float z)
					{
						surfaces.draw(uses, [&](context& commands, auto native)
									  {
							auto pass = [MTL4RenderPassDescriptor new];
							if (std::holds_alternative<surface_color>(c.surface))
							{
								pass.colorAttachments[0].texture = native[0];
								pass.colorAttachments[0].loadAction = MTLLoadActionClear;
								pass.colorAttachments[0].storeAction = MTLStoreActionStore;
								pass.colorAttachments[0].clearColor = MTLClearColorMake(value[0],value[1],value[2],value[3]);
							}
							else
							{
								pass.depthAttachment.texture = native[0];
								pass.depthAttachment.loadAction = MTLLoadActionClear;
								pass.depthAttachment.storeAction = MTLStoreActionStore;
								pass.depthAttachment.clearDepth = z;
								if (native[0].pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
								{
									pass.stencilAttachment.texture = native[0];
									pass.stencilAttachment.loadAction = MTLLoadActionClear;
									pass.stencilAttachment.storeAction = MTLStoreActionStore;
									pass.stencilAttachment.clearStencil = 17;
								}
							}
							(void)commands.render(pass);
							commands.end_encoding(); });
					};
					clear_native(c.clear, depth);
					auto reads = ram->reads;
					const auto revision = ram->version;
					first = textures.resolve(r, surfaces);
					require(first.texture != stale.texture && ram->reads == reads && ram->version == revision && surfaces.status(target_surface).dirty,
							"Dirty GPU alias used stale VM texture or forced quantized VM writeback");
					verify_native_depth(first.texture, depth);
					std::optional<surface_format> selected_format = surface_color::rgba32f;
					const auto selected = surfaces.snapshot_texture_regions(
						std::array{guest_byte_coverage({0x1000, plan.storage_bytes})},
						[](const auto&) { return std::optional<clear_rectangle>{{0, 0, 4, 4}}; }, budget, nullptr, false, &selected_format);
					require(selected && selected_format == c.surface, "Snapshot published the wrong selected source format");
					selected_format = surface_color::rgba32f;
					const auto absent = surfaces.snapshot_texture_regions(
						std::array{guest_byte_coverage({0x70000, 4})},
						[](const auto&) -> std::optional<clear_rectangle> { return {}; }, budget, nullptr, false, &selected_format);
					require(!absent && !selected_format, "Missing snapshot retained stale source format metadata");
					selected_format = surface_color::rgba32f;
					bool invalid_region = false;
					try { (void)surfaces.snapshot_texture_regions(std::array{guest_byte_coverage({0x1000, plan.storage_bytes})},
						[](const auto&) { return std::optional<clear_rectangle>{{4, 0, 1, 1}}; }, budget, nullptr, false, &selected_format); }
					catch (const std::out_of_range&) { invalid_region = true; }
					require(invalid_region && !selected_format, "Failed snapshot published source format metadata");
					ram->allow_writeback = true;
					surfaces.flush(target_surface);
					ram->allow_writeback = false;
					require(!surfaces.status(target_surface).dirty && ram->version != revision, "Native surface did not complete VM writeback");
					reads = ram->reads;
					surfaces.spill(target_surface);
					const auto spill_submits = surfaces.transaction_statistics().submitted;
					const auto spill_allocations = budget->statistics().created;
					clean = textures.resolve(r, surfaces);
					require(ram->reads == reads && !surfaces.status(target_surface).dirty && surfaces.status(target_surface).spilled &&
								clean.texture == first.texture && surfaces.transaction_statistics().submitted == spill_submits && budget->statistics().created == spill_allocations &&
								textures.statistics().uploads == 1 && textures.statistics().surface_snapshots == 2,
							"Clean native source lost precision/backing after VM flush and spill or reuploaded guest RAM");
					verify_native_depth(clean.texture, depth);
					if (c.format == 0x93 || c.format == 0x91)
					{
						const auto be = [&](unsigned n)
						{ std::uint32_t v = 0; for (unsigned i = 0; i < n; ++i) v = (v << 8) | std::to_integer<unsigned>(ram->bytes[i]); return v; };
						const auto guest_bits = c.format == 0x93 ? (be(2) << 11) + (120u << 23) : (be(4) >> 8) << 7;
						require(guest_bits != std::bit_cast<std::uint32_t>(depth), "Float-depth fixture did not actually exercise native precision beyond guest writeback");
					}
					auto deferred_description = description;
					deferred_description.address = 0x2000;
					const auto destination = surfaces.acquire(deferred_description);
					surfaces.defer_copy(target_surface, {0, 0, description.allocation.width, description.allocation.height}, destination, 0, 0);
					require(surfaces.status(destination).deferred_copies == 1, "Surface texture deferred copy was not queued");
					guest_texture_cache deferred_textures(ctx, ram, [](unsigned, unsigned, std::size_t)
														  { return 0x2000u; }, 1 << 20, budget);
					deferred = deferred_textures.resolve(r, surfaces);
					require(surfaces.status(destination).deferred_copies == 0 && surfaces.status(destination).dirty && ram->reads == reads,
							"Deferred surface texture source read stale VM or failed to materialize its GPU epoch");
					clear_native({1, 1, 1, 1}, .75f);
					surfaces.spill(target_surface);
					second = textures.resolve(r, surfaces);
					require(first.texture != second.texture && textures.statistics().surface_snapshots == 3 && textures.statistics().uploads == 1 && textures.statistics().hits == 1 && ram->reads == reads,
							"GPU epoch overwrite reused mutable snapshot or VM cache hit");
					// The old single-rectangle mapper rejects an oversized request.
					// A valid8x2 Morton raster reinterprets the old4x4 byte storage;
					// non-power-of-two Morton stays invalid. Linear extension consumes
					// exactly one formerly padded VM texel per row.
					auto partial = r;
					partial.image_rect = ((linear?5u:8u) << 16) | 2;
					if(!linear){auto invalid=r;invalid.image_rect=(5u<<16)|2;bool rejected=false;try{(void)textures.resolve(invalid,surfaces);}catch(const std::invalid_argument&){rejected=true;}require(rejected&&ram->reads==reads,"Invalid NPOT Morton alias captured pixels orwas admitted");}
					bool refused = false;
					resolved_fragment_texture oversized;
					std::string refusal;
					try
					{
						oversized=textures.resolve(partial, surfaces);
					}
					catch (const std::invalid_argument& error)
					{
						refused = true;refusal=error.what();
					}
					const auto partial_reads=ram->reads;
					{
						if(refused||!(linear?ram->reads>reads:ram->reads==reads)||!textures.statistics().assemblies_created||textures.statistics().assembly_vm_bytes!=(linear?2*plan.codec.block.bytes:0))throw std::runtime_error("Oversized alias format="+std::to_string(c.format)+" linear="+std::to_string(linear)+" samples="+std::to_string(samples)+" refused="+std::to_string(refused)+" reads="+std::to_string(ram->reads-reads)+" holebytes="+std::to_string(textures.statistics().assembly_vm_bytes)+" diagnostic="+refusal);
						color value=std::holds_alternative<surface_depth>(c.surface)?color{.75f,.75f,.75f,.75f}:color{1,1,1,1};if(((c.format==0x85||c.format==0x9e)&&c.surface==surface_format{surface_color::xrgb8_zero})||((c.format==0x82||c.format==0x9d)&&c.surface==surface_format{surface_color::x1rgb5_zero}))value[3]=0;if(c.format==0x92)value.fill(std::round(.75f*65535)/65535);sample(oversized,partial,{.1f,.25f,0,1},value);++draws;
					}
					if (c.format == 0x85 && linear)
					{
						auto mismatch = r;
						mismatch.format ^= (0x85u ^ 0x86u) << 8;
						const auto compressed = textures.resolve_versioned(mismatch, surfaces);
						require(compressed.revision.current() && ram->reads == partial_reads && compressed.image.texture.pixelFormat == MTLPixelFormatBC1_RGBA,
							"Compressed GPU alias read stale VM memory or lost source revision");
						const auto raw = make_buffer(ctx, 256, buffer_storage::shared);
						texture_region block;block.size = MTLSizeMake(4, 4, 1);
						ctx.begin();texture_to_buffer(ctx, compressed.image.texture, block, raw, 0, 256);ctx.submit().wait();
						for(unsigned i=0;i<8;++i) require(static_cast<const unsigned char*>(raw.contents)[i]==255,
							"Compressed alias did not preserve the current white source guest bytes");
						auto other_ram = std::make_shared<memory>();
						surface_cache unrelated(ctx, other_ram, 1 << 20);
						refused = false;
						try
						{
							(void)textures.resolve(r, unrelated);
						}
						catch (const std::invalid_argument&)
						{
							refused = true;
						}
						require(refused, "Surface texture resolver accepted unrelated VM adapters");
					}
				}
				// Independent heaps must survive cache/pool destruction and source
				// overwrites. Depthfloat keeps native precision beyond guest E4M12.
				sample(first, r, {.125f, .125f, 0, 1}, c.expected);
				sample(clean, r, {.125f, .125f, 0, 1}, c.expected);
				sample(deferred, r, {.125f, .125f, 0, 1}, c.expected);
				const bool fixed_float_remap = c.format == 0x9a || c.format == 0x9b || c.format == 0x9c;
				auto remapped = r;remapped.control1 = 0xaa6c; // A,B,G,R in guest ARGB order.
				sample(first, remapped, {.125f, .125f, 0, 1}, fixed_float_remap ? c.expected : color{c.expected[2], c.expected[1], c.expected[0], c.expected[3]});
				remapped.control1 = 0xaa39; // R,G,B,A in guest ARGB order.
				sample(clean, remapped, {.125f, .125f, 0, 1}, fixed_float_remap ? c.expected : color{c.expected[1], c.expected[2], c.expected[3], c.expected[0]});
				color updated = std::holds_alternative<surface_depth>(c.surface) ? color{.75f, .75f, .75f, .75f} : color{1, 1, 1, 1};
				if (((c.format == 0x85 || c.format == 0x9e) && c.surface == surface_format{surface_color::xrgb8_zero}) ||
					((c.format == 0x82 || c.format == 0x9d) && c.surface == surface_format{surface_color::x1rgb5_zero})) updated[3] = 0;
				if (c.format == 0x92)
					updated.fill(std::round(.75f * 65535) / 65535);
				sample(second, r, {.125f, .125f, 0, 1}, updated);
				draws += 6;
			}
		std::cout << draws << " native GPU surface alias RSX draws:27 storage-compatible pairs,1/2/4 samples, linear/Morton, source X/GB maps and cross-format raw casts before two register remaps,stale VM bypass,immutable epochs,destroyed-cache lifetime,oversized linear/Morton byte assembly andexact formerly padded VMtexels passed\n";
		// Two actual heaps admit the source and one immutable snapshot. A held
		// old epoch must prevent allocating another, without mutating either.
		auto r = words(0x85);
		fill(plan_guest_texture(r), {255, 0, 0, 0});
		const cached_surface_description description{0x1000, {4, 4, 1, surface_color::argb8, false}, 24, {}};
		std::size_t source_bytes;
		{
			surface_pool probe(ctx, 1 << 20);
			source_bytes = probe.acquire(description.allocation)->reserved_bytes();
		}
		auto budget = std::make_shared<heap_budget>(ctx, source_bytes * 2);
		surface_cache surfaces(ctx, ram, 1 << 20, budget);
		// Explicit uncached control: every request must allocate a fresh owner.
		// Default snapshot reuse is exercised independently below.
		surfaces.set_snapshot_limits({0, 0});
		guest_texture_cache textures(ctx, ram, translate, 1 << 20, budget);
		const auto target_surface = surfaces.acquire(description);
		surfaces.clear(target_surface, 0xf0, 0xff204080, 0, 255, {0, 0, 4, 4});
		auto first = textures.resolve(r, surfaces);
		const auto reads = ram->reads;
		const auto revision = ram->version;
		bool refused = false;
		try
		{
			(void)textures.resolve(r, surfaces);
		}
		catch (const heap_budget_error&)
		{
			refused = true;
		}
		require(refused && ram->reads == reads && textures.statistics().surface_snapshots == 1,
				"Retained native surface snapshot escaped quota or published a failed epoch");
		first.texture = nil;
		@autoreleasepool
		{
			resolved_fragment_texture retried;
			try { retried = textures.resolve(r, surfaces); }
			catch (const heap_budget_error&) { throw std::runtime_error("Uncached snapshot retry retained the released first heap"); }
			sample(retried, r, {.125f, .125f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		}
		ram->conflict = true;
		refused = false;
		try
		{
			(void)textures.resolve(r, surfaces);
		}
		catch (const surface_conflict_error&)
		{
			refused = true;
		}
		catch (const heap_budget_error&) { throw std::runtime_error("Uncached snapshot VM-conflict preflight allocated before detecting the conflict"); }
		ram->conflict = false;
		ram->version = revision;
		require(refused && ram->reads == reads && textures.statistics().surface_snapshots == 2,
				"Concurrent VM conflict published a dirty GPU snapshot or read guest bytes");
		// A queued writer has not published its epoch while the actual GPU
		// submission is blocked. Resolution must finish that transaction first.
		const auto gate = [ctx.device() newSharedEvent];
		require(bool(gate), "Surface alias gate creation failed");
		surfaces.set_wait_timeout(std::chrono::milliseconds(1));
		const std::array uses{cached_surface_use{target_surface, true}};
		bool timeout = false;
		try
		{
			surfaces.draw(uses, [&](context& commands, auto native)
						  {
				commands.wait_event(gate,1);
				auto pass = [MTL4RenderPassDescriptor new];
				pass.colorAttachments[0].texture = native[0];
				pass.colorAttachments[0].loadAction = MTLLoadActionClear;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				pass.colorAttachments[0].clearColor = MTLClearColorMake(1,0,0,1);
				(void)commands.render(pass);
				commands.end_encoding(); });
		}
		catch (const std::runtime_error&)
		{
			timeout = true;
		}
		require(timeout && surfaces.status(target_surface).pending, "Surface alias writer was not actually pending");
		timeout = false;
		try
		{
			(void)textures.resolve(r, surfaces);
		}
		catch (const std::runtime_error&)
		{
			timeout = true;
		}
		require(timeout && surfaces.status(target_surface).pending && ram->reads == reads && textures.statistics().surface_snapshots == 2,
				"Timed-out surface texture resolve published a stale epoch or fell back to guest RAM");
		gate.signaledValue = 1;
		surfaces.set_wait_timeout(std::chrono::seconds(30));
		resolved_fragment_texture completed;
		try { completed = textures.resolve(r, surfaces); }
		catch (const heap_budget_error&) { throw std::runtime_error("Uncached completed writer retained an obsolete snapshot heap"); }
		sample(completed, r, {.125f, .125f, 0, 1}, {1, 0, 0, 1});
		require(!surfaces.status(target_surface).pending && ram->reads == reads && textures.statistics().surface_snapshots == 3,
				"Completed surface writer did not publish its new immutable texture epoch");
		std::cout << "Native GPU surface alias shared-quota/retry/VM-conflict and actual pending-writer timeout/publication cases passed\n";
	}
	void native_snapshot_cache_cases()
	{
		auto r = words(0x85);
		fill(plan_guest_texture(r), {255, 0, 0, 0});
		auto budget = std::make_shared<heap_budget>(ctx, 8u << 20);
		surface_cache surfaces(ctx, ram, 8u << 20, budget);
		guest_texture_cache textures(ctx, ram, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, budget, 2);
		const auto source = surfaces.acquire({0x1000, {4, 4, 1, surface_color::argb8, false}, 24, {}});
		surfaces.clear(source, 0xf0, 0xff204080, 0, 255, {0, 0, 4, 4});
		const auto first = textures.resolve_versioned(r, surfaces);
		const auto cost = first.image.texture.heap.size;
		const auto created = budget->statistics().created;
		const auto submits = surfaces.transaction_statistics().submitted;
		const auto reads = ram->reads;
		for (unsigned repeat = 0; repeat < 100; ++repeat)
		{
			auto request = r;
			if (repeat & 1) request.format |= 0x40u << 8;
			request.control1 ^= repeat & 3;
			const auto image = textures.resolve_versioned(request, surfaces);
			require(image.image.texture == first.image.texture && image.revision.current(), "Unchanged native source/remap/UN missed its immutable snapshot");
			require(image.image.metadata.coordinates.scale[0] == (repeat & 1 ? .25f : 1.f), "Reused native snapshot retained another request's coordinate scale");
		}
		require(surfaces.snapshot_statistics().created == 1 && surfaces.snapshot_statistics().reused == 100 &&
			surfaces.transaction_statistics().submitted == submits && budget->statistics().created == created && ram->reads == reads,
			"Unchanged native aliases allocated,submitted or read VM on a snapshot hit");
		sample(first.image, r, {.125f, .125f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		surfaces.spill(source);
		const auto spill_submits = surfaces.transaction_statistics().submitted;
		const auto spill_allocations = budget->statistics().created;
		const auto restored = textures.resolve_versioned(r, surfaces);
		require(restored.image.texture == first.image.texture && restored.revision.current() &&
			surfaces.snapshot_statistics().created == 1 && ram->reads == reads && surfaces.status(source).spilled && surfaces.memory_statistics().leased_allocations == 0 &&
			surfaces.transaction_statistics().submitted == spill_submits && budget->statistics().created == spill_allocations,
			"Ready snapshot unnecessarily restored a spilled source or lost its exact immutable pixels");
		ram->allow_writeback = true; surfaces.flush(source); ram->allow_writeback = false;
		const auto flush_reads = ram->reads;
		const auto flushed = textures.resolve_versioned(r, surfaces);
		require(flushed.image.texture == first.image.texture && flushed.revision.current() && !first.revision.current() &&
			surfaces.snapshot_statistics().created == 1 && ram->reads == flush_reads,
			"Own VM writeback failed to refresh receipts without recopying unchanged native pixels");
		// A native writer forces a new generation even if the clear value is identical.
		surfaces.clear(source, 0xf0, 0xff204080, 0, 255, {0, 0, 4, 4});
		const auto rewritten = textures.resolve_versioned(r, surfaces);
		require(rewritten.image.texture != first.image.texture && surfaces.snapshot_statistics().created == 2,
			"New native write epoch reused an old snapshot based only on matching pixels");
		surfaces.clear(source, 0xf0, 0xff804020, 0, 255, {0, 0, 4, 4});
		const auto latest = textures.resolve_versioned(r, surfaces);
		require(latest.image.texture != rewritten.image.texture && surfaces.snapshot_statistics().created == 3, "Changed native writer did not replace its snapshot");
		sample(first.image, r, {.125f, .125f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		sample(latest.image, r, {.125f, .125f, 0, 1}, {128.f / 255, 64.f / 255, 32.f / 255, 1});
		const auto coverage = guest_byte_coverage({0x1000, 96});
		const auto region = [&](unsigned x)
		{
			return surfaces.snapshot_texture_region(coverage, [=](const auto&) -> std::optional<clear_rectangle>
				{ return clear_rectangle{x, 0, 1, 1}; }, budget);
		};
		surfaces.clear_snapshots();
		surfaces.set_snapshot_limits({2, cost * 2});
		const auto a = region(0), b = region(1);
		require(a == region(0), "Snapshot LRU hit replaced its native identity");
		const auto c = region(2);
		require(surfaces.snapshot_statistics().entries == 2 && surfaces.snapshot_statistics().bytes <= cost * 2 && a == region(0),
			"Snapshot entry/byte quota or recently used owner was lost");
		require(region(1) != b, "Snapshot LRU did not evict the oldest region");
		surfaces.set_snapshot_limits({64, cost});
		(void)region(2);
		require(surfaces.snapshot_statistics().entries == 1 && surfaces.snapshot_statistics().bytes <= cost, "Snapshot byte quota relied only on its entry count");
		surfaces.set_snapshot_limits({64, 1});
		const auto bypass = surfaces.snapshot_statistics().bypassed;
		require(region(0) != region(0) && surfaces.snapshot_statistics().entries == 0 && surfaces.snapshot_statistics().bypassed == bypass + 2,
			"Oversized snapshots were retained or refused instead of safely bypassing the cache");
		surfaces.set_snapshot_limits({0, 0});
		require(region(1) != region(1) && surfaces.snapshot_statistics().entries == 0, "Disabled snapshot cache retained an owner");
		bool invalid = false;
		try { surfaces.set_snapshot_limits({4097, cost}); } catch (const std::invalid_argument&) { invalid = true; }
		require(invalid && surfaces.snapshot_statistics().limits.entries == 0, "Invalid snapshot limits mutated the previous policy");
		surfaces.set_snapshot_limits({64, 64u << 20});
		guest_texture_cache casts(ctx, ram, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, budget, 2);
		const auto cast_words = [&](unsigned format, unsigned width)
		{
			auto value = words(format, vertex_texture_dimension::two, true, width, 4);
			value.control3 = (1u << 20) | 24; return value;
		};
		const auto wa = cast_words(0x81, 16), wb = cast_words(0x8b, 8), wc = cast_words(0x92, 8);
		const auto ca = casts.resolve(wa, surfaces), cb = casts.resolve(wb, surfaces);
		const auto copies = surfaces.snapshot_statistics().created;
		require(casts.resolve(wa, surfaces).texture == ca.texture, "Native cast cache missed the complete source/codec key");
		const auto cc = casts.resolve(wc, surfaces);
		require(casts.statistics().cast_entries == 2 && casts.statistics().cast_bytes <= (1u << 20) && casts.resolve(wb, surfaces).texture != cb.texture &&
			surfaces.snapshot_statistics().created == copies && casts.statistics().casts_created == 4,
			"Native cast LRU did not share the source snapshot or evict the oldest codec result");
		casts.clear(); require(casts.statistics().cast_entries == 0 && casts.statistics().cast_bytes == 0, "Native cast clear retained cache accounting");
		guest_texture_cache byte_limited(ctx, ram, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 96, budget, 2);
		const auto uncached_a = byte_limited.resolve(wa, surfaces), uncached_b = byte_limited.resolve(wa, surfaces);
		require(uncached_a.texture != uncached_b.texture && byte_limited.statistics().cast_entries == 0 && byte_limited.statistics().cast_bytes == 0 &&
			byte_limited.statistics().casts_created == 2 && surfaces.snapshot_statistics().created == copies,
			"Native cast cache ignored actual heap bytes or recopied its still-valid source under byte-limit bypass");
		std::cout << "Native snapshot cache:100 exact source/remap/UN hits without allocation/copy/submission;spill reuse,new identical/changed writer generations,old/new GPU pixels;LRU/actual heap-byte quotas,oversized/disabled bypass andinvalid limits passed\n";
		std::cout << "Native cast cache:source shared across three target codecs,two-entry LRU,exact-hit reuse,clear accounting andactual heap-byte bypass passed\n";
		context pending_ctx{1};
		auto pending_budget = std::make_shared<heap_budget>(pending_ctx, 8u << 20);
		const auto gate = [pending_ctx.device() newSharedEvent];
		const auto output = make_buffer(pending_ctx, 64, buffer_storage::shared);
		std::size_t retained = 0;
		const auto work = [&]
		{
			@autoreleasepool
			{
				surface_cache owner(pending_ctx, ram, 8u << 20, pending_budget);
				guest_texture_cache texture_owner(pending_ctx, ram, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1u << 20, pending_budget);
				const auto target = owner.acquire({0x1000, {4, 4, 1, surface_color::argb8, false}, 24, {}});
				owner.clear(target, 0xf0, 0xff204080, 0, 255, {0, 0, 4, 4});
				const auto image = texture_owner.resolve(r, owner);
				retained = image.texture.heap.size;
				pending_ctx.begin(); pending_ctx.wait_event(gate, 1);
				texture_to_buffer(pending_ctx, image.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, output, 0);
				auto result = pending_ctx.submit();
				owner.clear_snapshots(); texture_owner.clear();
				return result;
			}
		}();
		bool timed_out = false;
		try { work.wait(std::chrono::milliseconds(1)); } catch (const std::runtime_error&) { timed_out = true; }
		require(timed_out && !work.ready() && pending_budget->statistics().reserved_bytes >= retained,
			"Cleared/destroyed native snapshot owners released a truly pending GPU image or its quota");
		gate.signaledValue = 1; work.wait();
		const std::array<std::byte, 4> pixel{std::byte{128}, std::byte{64}, std::byte{32}, std::byte{255}};
		for (unsigned i = 0; i < 64; ++i) require(static_cast<const std::byte*>(output.contents)[i] == pixel[i % 4], "Pending native snapshot changed after owner destruction");
		require(pending_budget->statistics().reserved_bytes == 0, "Completed native snapshot retained orphan heap quota");
		std::cout << "Native snapshot owner destruction:actual gated GPU timeout retains heap charge andexact64 BGRA bytes through cache clear/destruction;completion releases all charges passed\n";
	}
	void clean_surface_cpu_changes()
	{
		unsigned draws = 0;
		for (const unsigned format : {0x85u, 0x93u, 0x91u})
			for (bool linear : {true, false})
			{
				const auto r = words(format, vertex_texture_dimension::two, linear);
				const auto plan = plan_guest_texture(r);
				fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
				auto budget = std::make_shared<heap_budget>(ctx, 4 << 20);
				surface_cache surfaces(ctx, ram, 4 << 20, budget);
				guest_texture_cache textures(ctx, ram, [](unsigned, unsigned, std::size_t)
											 { return 0x1000u; }, 1 << 20, budget);
				const surface_format native_format = format == 0x85 ? surface_format{surface_color::argb8} : surface_format{*plan.codec.depth};
				cached_surface_description description{0x1000, {4, 4, 1, native_format, false}, linear ? plan.subresources[0].row_pitch : 0, {}};
				if (!linear)
					description.morton = swizzle_region{{4, 4, 1}, {}, {4, 4, 1}};
				const auto source = surfaces.acquire(description);
				// Background initializes native contents without a guest upload.
				surfaces.prepare(source, surface_initialization::background);
				ram->allow_writeback = true;
				surfaces.flush(source);
				ram->allow_writeback = false;
				const auto original = textures.resolve(r, surfaces);
				const color original_value = format == 0x85 ? color{0, 0, 0, 1} : color{1, 1, 1, 1};
				const auto old_revision = surfaces.status(source).guest_revision;
				// Simulate the atomic VM adapter publishing a CPU write after the
				// clean native epoch. Old snapshots remain immutable read owners.
				fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
				const auto reads = ram->reads;
				const auto replacement = textures.resolve(r, surfaces);
				require(replacement.texture != original.texture && ram->reads == reads + 1 && textures.statistics().surface_snapshots == 1 && textures.statistics().uploads == 1,
						"CPU revision change reused stale clean native data or missed immutable VM upload");
				require(textures.resolve(r, surfaces).texture == replacement.texture && ram->reads == reads + 1,
						"Stale clean native source prevented a coherent VM cache hit");
				surfaces.prepare(source);
				require(surfaces.status(source).guest_revision != old_revision && !surfaces.status(source).dirty,
						"CPU revision change failed to refresh native clean surface backing");
				const auto native_refreshed = textures.resolve(r, surfaces);
				require(native_refreshed.texture != replacement.texture && textures.statistics().surface_snapshots == 2 && textures.statistics().uploads == 1 && ram->reads == reads + 2,
						"Refreshed native surface did not resume independent snapshot resolution");
				sample(original, r, {.125f, .125f, 0, 1}, original_value);
				// RSX E4M12 zero bits expand to exponent120 (1/128), not IEEE zero.
				const color replacement_value = format == 0x93 ? color{.0078125f, .0078125f, .0078125f, .0078125f} : color{0, 0, 0, 0};
				sample(replacement, r, {.125f, .125f, 0, 1}, replacement_value);
				sample(native_refreshed, r, {.125f, .125f, 0, 1}, replacement_value);
				draws += 3;
			}
		std::cout << draws << " native clean surface CPU-revision/fallback/hit/refresh/immutable-old-epoch draws passed\n";
	}
	void region_alias_cases()
	{
		unsigned draws = 0;
		for (const unsigned samples : {1u, 2u, 4u})
		for (unsigned format : {0x81u, 0x84u, 0x85u, 0x8bu, 0x9au, 0x9bu, 0x9cu, 0x90u, 0x91u, 0x92u, 0x93u})
			for (unsigned storage_case : {0u, 1u, 2u, 3u})
			{
				const bool linear = storage_case == 0;
				const unsigned source_width = storage_case == 2 ? 4 : 8;
				const std::array<std::size_t, 3> storage{source_width, 8, storage_case == 3 ? 2u : 1u};
				const auto source_words = words(format, vertex_texture_dimension::two, linear, source_width, 8);
				const auto plan = plan_guest_texture(source_words);
				fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
				ram->bytes.resize((linear ? plan.storage_bytes : storage[0] * storage[1] * storage[2] * plan.codec.block.bytes), std::byte{0});
				auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
				surface_cache surfaces(ctx, ram, 8 << 20, budget);
				guest_texture_cache textures(ctx, ram, [](unsigned offset, unsigned, std::size_t)
											 { return 0x1000u + offset - 0x200; }, 1 << 20, budget);
				const auto color_format = [&]() -> surface_color
				{
					switch (format)
					{
					case 0x81:
						return surface_color::b8;
					case 0x84:
						return surface_color::rgb565;
					case 0x85:
						return surface_color::argb8;
					case 0x8b:
						return surface_color::gb8;
					case 0x9a:
						return surface_color::rgba16f;
					case 0x9b:
						return surface_color::rgba32f;
					default:
						return surface_color::r32f;
					}
				};
				const surface_format native_format = plan.codec.depth ? surface_format{*plan.codec.depth} : surface_format{color_format()};
				cached_surface_description description{0x1000, {source_width, 8, 1, native_format, false}, linear ? plan.subresources[0].row_pitch : 0, {}};
				if (!linear)
					description.morton = swizzle_region{storage, {0, 0, storage[2] - 1}, {source_width, 8, 1}};
				const auto factors = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
				description.allocation.width /= factors.x;
				description.allocation.height /= factors.y;
				description.allocation.samples = samples;
				const auto source = surfaces.acquire(description);
				const auto expected = [&](unsigned x, unsigned y) -> color
				{
					const float depth = .50123456f + x / 256.f + y / 16.f;
					switch (format)
					{
					case 0x81:
						return {(32 + x * 13 + y * 7) / 255.f, (32 + x * 13 + y * 7) / 255.f, (32 + x * 13 + y * 7) / 255.f, 1};
					case 0x84:
						return {std::floor((x + 1) / 31.f * 255) / 255, std::floor((y + 1) / 63.f * 255) / 255, std::floor((x + y + 1) / 31.f * 255) / 255, 1};
					case 0x85:
						return {(32 + x) / 255.f, (64 + y) / 255.f, (128 + x + y) / 255.f, 1};
					case 0x8b:
						return {(32 + x) / 255.f, (64 + y) / 255.f, (32 + x) / 255.f, (64 + y) / 255.f};
					case 0x9a:
					case 0x9b:
						return {x / 8.f, y / 8.f, (x + y) / 16.f, 1};
					case 0x9c:
						return {(x + 8 * y + 1) / 64.f, (x + 8 * y + 1) / 64.f, (x + 8 * y + 1) / 64.f, (x + 8 * y + 1) / 64.f};
					case 0x92:
					{
						const float z = std::round(depth * 65535) / 65535;
						return {z, z, z, z};
					}
					default:
						return {depth, depth, depth, depth};
					}
				};
				const auto raw = make_buffer(ctx, 2048, buffer_storage::shared), stencil = make_buffer(ctx, 2048, buffer_storage::shared);
				const auto put = [&](unsigned x, unsigned y, unsigned byte, unsigned value, unsigned count)
				{
					for (unsigned i = 0; i < count; ++i)
						static_cast<unsigned char*>(raw.contents)[y * 256 + byte * x + i] = static_cast<unsigned char>(value >> (i * 8));
				};
				const auto half = [](float value) -> unsigned
				{ const auto b=std::bit_cast<unsigned>(value);return b ? ((((b>>23)&255)-112)<<10)|((b>>13)&1023) : 0; };
				for (unsigned y = 0; y < 8; ++y)
					for (unsigned x = 0; x < 8; ++x)
					{
						const auto p = expected(x, y);
						switch (format)
						{
						case 0x81:
							put(x, y, 1, 32 + x * 13 + y * 7, 1);
							break;
						case 0x84:
							put(x, y, 2, ((x + 1) << 11) | ((y + 1) << 5) | (x + y + 1), 2);
							break;
						case 0x85:
							put(x, y, 4, 0xff000000u | ((32 + x) << 16) | ((64 + y) << 8) | (128 + x + y), 4);
							break;
						case 0x8b:
							put(x, y, 2, (32 + x) | ((64 + y) << 8), 2);
							break;
						case 0x9a:
						case 0x9b:
							for (unsigned c = 0; c < 4; ++c)
							{
								const unsigned bytes = format == 0x9a ? 2 : 4;
								const unsigned value = format == 0x9a ? half(p[c]) : std::bit_cast<unsigned>(p[c]);
								for (unsigned b = 0; b < bytes; ++b)
									static_cast<unsigned char*>(raw.contents)[y * 256 + x * bytes * 4 + c * bytes + b] = static_cast<unsigned char>(value >> (8 * b));
							}
							break;
						case 0x92:
							put(x, y, 2, static_cast<unsigned>(std::round(p[0] * 65535)), 2);
							break;
						default:
							put(x, y, 4, std::bit_cast<unsigned>(p[0]), 4);
							break;
						}
						static_cast<unsigned char*>(stencil.contents)[y * 256 + x] = 42 + x + y;
					}
				const std::array uses{cached_surface_use{source, true}};
				surfaces.draw(uses, [&](context& commands, auto native)
							  {
					texture_region whole;whole.size=MTLSizeMake(source_width,8,1);
					id<MTLTexture> storage = native[0];
					if (samples > 1)
					{
						auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native[0].pixelFormat width:source_width height:8 mipmapped:NO];
						descriptor.storageMode = MTLStorageModePrivate;
						descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
						storage = make_texture(commands, descriptor);
					}
					buffer_to_texture(commands,raw,0,storage,whole,256,0,plan.codec.depth?texture_plane::depth:texture_plane::all);
					if (native[0].pixelFormat==MTLPixelFormatDepth32Float_Stencil8)
						buffer_to_texture(commands,stencil,0,storage,whole,256,0,texture_plane::stencil);
					if (samples > 1)
					{
						sample_transfer transfer(commands);
						transfer.import(commands, storage, native[0]);
					} });
				const auto morton = [&](unsigned x, unsigned y)
				{ unsigned result=0,weight=1;for (unsigned bit=1;bit<std::max({storage[0],storage[1],storage[2]});bit*=2) for (unsigned axis=0;axis<3;++axis) if (bit<storage[axis]) { if (std::array<std::size_t,3>{x,y,storage[2]-1}[axis]&bit) result+=weight;weight*=2; }return result; };
				const auto input_words = [&](clear_rectangle rectangle)
				{
					auto r = words(format, vertex_texture_dimension::two, linear, static_cast<unsigned>(rectangle.width), static_cast<unsigned>(rectangle.height));
					r.control3 = linear ? static_cast<unsigned>(plan.subresources[0].row_pitch) : 0;
					const unsigned offset = linear ? rectangle.y * plan.subresources[0].row_pitch + rectangle.x * plan.codec.block.bytes : morton(rectangle.x, rectangle.y) * plan.codec.block.bytes;
					r.offset = 0x80000200u + offset;
					return r;
				};
				if (linear)
				{
					// A dirty adjacent pixel lies inside the requested last-row
					// padding span, not inside the sampled pixel coverage.
					const auto neighbor = surfaces.acquire({static_cast<std::uint32_t>(0x1000 + plan.storage_bytes), {1, 1, 1, native_format, false}, 0, {}});
					surfaces.prepare(neighbor, surface_initialization::background);
				}
				const std::array<clear_rectangle, 4> regions = linear ? std::array<clear_rectangle, 4>{{{1, 2, 3, 2}, {3, 6, 2, 2}, {0, 0, 8, 1}, {7, 7, 1, 1}}} : storage_case == 2 ? std::array<clear_rectangle, 4>{{{0, 2, 4, 4}, {1, 3, 1, 1}, {0, 4, 4, 2}, {0, 0, 4, 8}}}
																																							   : storage_case == 3	 ? std::array<clear_rectangle, 4>{{{2, 2, 2, 2}, {6, 6, 2, 2}, {3, 5, 1, 1}, {7, 7, 1, 1}}}
																																													 : std::array<clear_rectangle, 4>{{{2, 2, 2, 2}, {4, 6, 4, 2}, {3, 5, 1, 1}, {0, 0, 8, 8}}};
				const auto reads = ram->reads;
				for (const auto rectangle : regions)
				{
					const auto r = input_words(rectangle);
					const auto image = textures.resolve(r, surfaces);
					require(image.texture.width == rectangle.width && image.texture.height == rectangle.height && ram->reads == reads,
							"GPU surface subregion read VM or used incorrect native snapshot dimensions");
					sample(image, r, {.5f / rectangle.width, .5f / rectangle.height, 0, 1}, expected(rectangle.x, rectangle.y));
					sample(image, r, {1 - .5f / rectangle.width, 1 - .5f / rectangle.height, 0, 1}, expected(rectangle.x + rectangle.width - 1, rectangle.y + rectangle.height - 1));
					draws += 2;
					if (format == 0x85 && linear && rectangle.x == 1)
					{
						auto un = r;
						un.format |= 0x40u << 8;
						const auto pixels = textures.resolve(un, surfaces);
						require(pixels.metadata.coordinates.scale == std::array<float, 3>{1.f / 3, 1.f / 2, 1}, "Cropped UN metadata used the containing surface dimensions");
						sample(pixels, un, {.5f, .5f, 0, 1}, expected(rectangle.x, rectangle.y));
						++draws;
					}
					// Compare every native storage word, not only the two sampled endpoints.
					// This detects guest float-codec quantization and any sample-coordinate mixup.
					const auto plane = plan.codec.depth ? texture_plane::depth : texture_plane::all;
					const auto native_bytes = storage_block(image.texture.pixelFormat, plane).bytes;
					const auto copied = make_buffer(ctx, 256 * rectangle.height, buffer_storage::shared);
					texture_region copied_region;copied_region.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
					ctx.begin();
					texture_to_buffer(ctx, image.texture, copied_region, copied, 0, 256, 0, plane);
					ctx.submit().wait();
					for (unsigned y = 0; y < rectangle.height; ++y)
						for (unsigned x = 0; x < rectangle.width; ++x)
							require(std::memcmp(static_cast<const std::byte*>(copied.contents) + y * 256 + x * native_bytes,
								static_cast<const std::byte*>(raw.contents) + (rectangle.y + y) * 256 + (rectangle.x + x) * native_bytes,
								native_bytes) == 0, "Native MSAA/region snapshot changed a native sample word");
					if (image.texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
					{
						const auto out = make_buffer(ctx, 256 * rectangle.height, buffer_storage::shared);
						texture_region whole;
						whole.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
						ctx.begin();
						texture_to_buffer(ctx, image.texture, whole, out, 0, 256, 0, texture_plane::stencil);
						ctx.submit().wait();
						for (unsigned y = 0; y < rectangle.height; ++y)
							for (unsigned x = 0; x < rectangle.width; ++x)
								require(static_cast<const unsigned char*>(out.contents)[y * 256 + x] == 42 + rectangle.x + x + rectangle.y + y, "Native partial depth copy lost or rebased source stencil incorrectly");
					}
				}
				bool refused = false;
				auto invalid = input_words(linear ? clear_rectangle{0, 0, 2, 2} : clear_rectangle{0, 0, 2, 4});
				if (linear)
					invalid.control3 += plan.codec.block.bytes;
				try
				{
					(void)textures.resolve(invalid, surfaces);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && ram->reads == reads, "Incompatible pitch/Morton bit sequence fell back to stale VM data");
			}
		std::cout << draws << " patterned native partial surface RSX draws: eleven formats, 1/2/4 samples, linear address/pitch and Morton axis/carry/volume-slice mapping, cropped dimensions and depth stencil regions passed\n";
	}
	void multisample_snapshot_cases()
	{
		const auto writer = compile.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 sample_vertex(uint i [[vertex_id]]) {
 const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[i],0,1);
}
fragment float4 sample_color(float4 p [[position]],uint s [[sample_id]]) {
 return float4(float(s+1)/8,float(uint(p.x)%4)/8,float(uint(p.y)%4)/8,1);
}
struct depth_sample {float depth [[depth(any)]];uint stencil [[stencil]];};
fragment depth_sample sample_depth(float4 p [[position]],uint s [[sample_id]]) {
 const uint x=uint(p.x),y=uint(p.y);
 return {.50123456f+float(s)/16+float(x)/256+float(y)/64,17+s*53+x+y*7};
}
)msl", @"Independent native MSAA RTT writer");
		MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
		ds.depthCompareFunction = MTLCompareFunctionAlways;ds.depthWriteEnabled = YES;
		MTLStencilDescriptor* st = [MTLStencilDescriptor new];
		st.stencilCompareFunction = MTLCompareFunctionAlways;st.depthStencilPassOperation = MTLStencilOperationReplace;
		ds.frontFaceStencil = ds.backFaceStencil = st;
		const auto depth_state = [ctx.device() newDepthStencilStateWithDescriptor:ds];
		unsigned checks = 0;
		for (const unsigned samples : {2u, 4u})
			for (const bool depth : {false, true})
				for (const bool linear : {true, false})
				{
					const unsigned width = 8, height = samples == 2 ? 4 : 8;
					const auto r = words(depth ? 0x91 : 0x9b, vertex_texture_dimension::two, linear, width, height);
					const auto plan = plan_guest_texture(r);
					fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
					const auto original_ram = ram->bytes;
					auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
					resolved_fragment_texture retained;
					id<MTLTexture> whole = nil;
					{
						surface_cache surfaces(ctx, ram, 8 << 20, budget);
						guest_texture_cache textures(ctx, ram, [](unsigned, unsigned, std::size_t) { return 0x1000u; }, 1 << 20, budget);
						const surface_format format = depth ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::rgba32f};
						cached_surface_description description{0x1000, {4, 4, samples, format, true}, linear ? plan.subresources[0].row_pitch : 0, {}};
						if (!linear) description.morton = swizzle_region{{width, height, 1}, {}, {width, height, 1}};
						const auto surface = surfaces.acquire(description);
						MTL4RenderPipelineDescriptor* pd = [MTL4RenderPipelineDescriptor new];
						pd.vertexFunctionDescriptor = library_function(writer, @"sample_vertex");
						pd.fragmentFunctionDescriptor = library_function(writer, depth ? @"sample_depth" : @"sample_color");
						pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;pd.rasterSampleCount = samples;
						if (!depth) pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
						const auto pipeline = compile.render(pd);
						const std::array uses{cached_surface_use{surface, true}};
						surfaces.draw(uses, [&](context& commands, auto native)
						{
							auto pass = [MTL4RenderPassDescriptor new];
							if (depth)
							{
								pass.depthAttachment.texture = pass.stencilAttachment.texture = native[0];
								pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
								pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
							}
							else
							{
								pass.colorAttachments[0].texture = native[0];pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
								pass.colorAttachments[0].storeAction = MTLStoreActionStore;
							}
							render_commands draw(commands, pass);
							raster_state raster;raster.viewport = {0, 0, 4, 4, 0, 1};raster.scissor = {0, 0, 4, 4};
							draw.configure(pipeline, depth ? depth_state : nil, raster, {}, {});draw.draw(MTLPrimitiveTypeTriangle, 0, 3);draw.end();
						});
						const auto reads = ram->reads;
						const auto revision = ram->version;
						const auto versioned = textures.resolve_versioned(r, surfaces);
						require(versioned.revision.current(), "Fresh native MSAA source ticket is stale");
						retained = versioned.image;
						const auto copies = surfaces.snapshot_statistics().created;
						const auto submits = surfaces.transaction_statistics().submitted;
						for (unsigned repeat = 0; repeat < 16; ++repeat)
							require(textures.resolve_versioned(r, surfaces).image.texture == retained.texture,
								"Unchanged MSAA native epoch did not reuse its exact expanded image");
						require(surfaces.snapshot_statistics().created == copies && surfaces.transaction_statistics().submitted == submits,
							"Unchanged MSAA epoch re-expanded/re-copied native sample storage");
						whole = surfaces.snapshot_texture({0x1000, plan.storage_bytes}, [](const auto&) { return true; }, budget);
						require(whole == retained.texture && surfaces.snapshot_statistics().created == copies,
							"Whole/region snapshot APIs duplicated the same immutable MSAA source epoch");
						require(whole.width == width && whole.height == height && retained.texture.width == width && retained.texture.height == height &&
							ram->reads == reads && ram->version == revision && ram->bytes == original_ram && surfaces.status(surface).dirty,
							"Native MSAA snapshot used VM,changed source storage or lost expanded dimensions");
						// Bounds are interpreted in sample coordinates, not rendered pixels.
						bool refused = false;
						try { (void)surfaces.snapshot_texture_region(guest_byte_coverage({0x1000, plan.storage_bytes}),
							[&](const auto&) -> std::optional<clear_rectangle> { return clear_rectangle{width, 0, 1, 1}; }, budget); }
						catch (const std::out_of_range&) { refused = true; }
						require(refused && ram->reads == reads, "Out-of-raster MSAA snapshot touched VM or was accepted");
						// Change every original sample,then release its native allocation.
						surfaces.draw(uses, [&](context& commands, auto native)
						{
							auto pass = [MTL4RenderPassDescriptor new];
							if (depth)
							{
								pass.depthAttachment.texture = pass.stencilAttachment.texture = native[0];
								pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = MTLLoadActionClear;
								pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
								pass.depthAttachment.clearDepth = 0;pass.stencilAttachment.clearStencil = 0;
							}
							else
							{
								pass.colorAttachments[0].texture = native[0];pass.colorAttachments[0].loadAction = MTLLoadActionClear;
								pass.colorAttachments[0].storeAction = MTLStoreActionStore;pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
							}
							(void)commands.render(pass);commands.end_encoding();
						});
						surfaces.spill(surface);
						++ram->version;
						require(!versioned.revision.current(), "Native MSAA source ticket ignored a CPU revision change");
					}
					// Both APIs retain the same immutable heap after source/cache destruction.
					const auto raw = make_buffer(ctx, 256 * height, buffer_storage::shared);
					const auto stencil = make_buffer(ctx, 256 * height, buffer_storage::shared);
					for (const auto& image : {whole, retained.texture})
					{
						texture_region region;region.size = MTLSizeMake(width, height, 1);
						ctx.begin();texture_to_buffer(ctx, image, region, raw, 0, 256, 0, depth ? texture_plane::depth : texture_plane::all);
						if (depth) texture_to_buffer(ctx, image, region, stencil, 0, 256, 0, texture_plane::stencil);
						ctx.submit().wait();
						for (unsigned y = 0; y < height; ++y)
							for (unsigned x = 0; x < width; ++x)
							{
								const unsigned px = x / 2, py = samples == 2 ? y : y / 2, sample_id = x % 2 + (samples == 2 ? 0 : (y % 2) * 2);
								if (depth)
								{
									const float expected = .50123456f + float(sample_id) / 16 + float(px) / 256 + float(py) / 64;
									require(static_cast<const unsigned*>(raw.contents)[y * 64 + x] == std::bit_cast<unsigned>(expected), "Independent MSAA depth snapshot lost sample position/native float bits");
									require(static_cast<const unsigned char*>(stencil.contents)[y * 256 + x] == 17 + sample_id * 53 + px + py * 7, "Independent MSAA snapshot lost sample stencil");
								}
								else
								{
									const color expected{float(sample_id + 1) / 8, float(px) / 8, float(py) / 8, 1};
									require(std::memcmp(static_cast<const std::byte*>(raw.contents) + y * 256 + x * 16, expected.data(), 16) == 0, "Independent MSAA color snapshot averaged or reordered samples");
								}
							}
						++checks;
					}
				}
		std::cout << checks << " native MSAA API readbacks across8sample_id-written color/depth/stencil profiles:16 repeated hits per profile,whole/region share one immutable image,exact float bits,VM preservation,source overwrite/spill/cache destruction and revision conflicts passed\n";
	}
	void cache_cases()
	{
		const auto r = words(0x85);
		fill(plan_guest_texture(r), {255, 32, 64, 128});
		const auto translate = [&](unsigned offset, unsigned location, std::size_t size)
		{
			require((offset == 0x200 || offset == 0x400) && location <= 1 && size <= ram->bytes.size(), "Cache address mapping lost frozen physical range");
			return 0x1000;
		};
		auto budget = std::make_shared<heap_budget>(ctx, 1 << 20);
		guest_texture_cache textures(ctx, ram, translate, 1 << 20, budget, 2);
		const auto before_reads = ram->reads;
		auto old = textures.resolve(r);
		const auto heap_bytes = old.texture.heap.size;
		require(heap_bytes > 0 && budget->statistics().reserved_bytes == heap_bytes, "Texture cache did not charge actual native heap backing");
		const auto hit = textures.resolve(r);
		require(hit.texture == old.texture && ram->reads == before_reads + 1, "Unchanged VM texture was reuploaded");
		auto remapped = r;
		remapped.control1 = 0xaaaa;
		const auto same_storage = textures.resolve(remapped);
		require(same_storage.texture == old.texture, "Sampler/register remap unnecessarily changed image storage");
		sample(same_storage, remapped, {.125f, .125f, 0, 1}, {64.f / 255, 64.f / 255, 64.f / 255, 64.f / 255});
		auto un = r;
		un.format |= 0x40u << 8;
		const auto normalized = textures.resolve(un);
		require(normalized.texture == old.texture && normalized.metadata.coordinates.scale == std::array<float, 3>{.25f, .25f, 1}, "UN cache hit reused stale coordinate metadata or image");
		auto alias = r;
		alias.offset = 0x400;
		alias.format = (alias.format & ~3u) | 1;
		require(textures.resolve(alias).texture == old.texture, "Translated local/main aliases did not reuse identical storage/range");
		fill(plan_guest_texture(r), {255, 96, 64, 128});
		auto replacement = textures.resolve(r);
		require(replacement.texture != old.texture && textures.statistics().invalidations == 1, "VM revision change overwrote old GPU image or reused stale pixels");
		sample(old, r, {.125f, .125f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		sample(replacement, r, {.125f, .125f, 0, 1}, {96.f / 255, 64.f / 255, 128.f / 255, 1});
		require(budget->statistics().reserved_bytes >= old.texture.heap.size + replacement.texture.heap.size, "Externally retained old revision left heap quota prematurely");
		textures.invalidate({0x1000, 1});
		require(textures.statistics().entries == 0, "Overlapping CPU invalidation left cache entry current");
		(void)textures.resolve(r);
		require(textures.statistics().uploads == 3, "Explicit invalidation did not trigger fresh immutable upload");
		auto other = r;
		other.format = (other.format & ~(255u << 8)) | ((0x9eu | 0x20u) << 8);
		(void)textures.resolve(other);
		auto third = r;
		third.format = (third.format & ~(255u << 8)) | ((0x82u | 0x20u) << 8);
		(void)textures.resolve(third);
		require(textures.statistics().entries == 2 && textures.statistics().evictions == 1, "Distinct format storage did not enforce cache LRU capacity");
		// Surface storage and texture versions consume the very same ledger.
		{
			surface_pool surfaces(ctx, 1 << 20, budget);
			const auto prior = budget->statistics().reserved_bytes;
			auto surface = surfaces.acquire({4, 4, 1, surface_color::argb8, false});
			const auto bytes = surface->reserved_bytes();
			require(budget->statistics().reserved_bytes >= prior + bytes, "Shared surface/texture heap accounting diverged");
			ctx.begin();
			id<MTLTexture> retained = surface->render_texture(ctx);
			require(bool(retained), "Shared surface texture retention failed");
			ctx.cancel();
			surface.reset();
			(void)surfaces.trim(0);
			require(budget->statistics().reserved_bytes >= prior + bytes, "Retired externally held surface escaped shared quota");
			retained = nil;
		}
		// A single-image quota cannot discard a still-held texture version to
		// admit another heap,whether requested by a texture or a surface pool.
		{
			auto tight = std::make_shared<heap_budget>(ctx, heap_bytes);
			guest_texture_cache limited(ctx, ram, translate, 1 << 20, tight, 1);
			auto owned = limited.resolve(r);
			auto view = make_register_texture_view(owned.texture, decode_fragment_texture_state(remapped, owned.metadata));
			require(bool(view) && view.parentTexture == owned.texture, "Retired view did not reference original budgeted parent");
			owned.texture = nil;
			fill(plan_guest_texture(r), {255, 128, 64, 128});
			const auto reads = ram->reads;
			bool refused = false;
			try
			{
				(void)limited.resolve(r);
			}
			catch (const heap_budget_error&)
			{
				refused = true;
			}
			require(refused && ram->reads == reads && tight->statistics().reserved_bytes == heap_bytes, "Texture quota ignored retained view or performed denied VM upload");
			surface_pool surfaces(ctx, 1 << 20, tight);
			refused = false;
			try
			{
				(void)surfaces.acquire({4, 4, 1, surface_color::argb8, false});
			}
			catch (const surface_budget_error&)
			{
				refused = true;
			}
			require(refused, "Surface pool exceeded shared quota occupied by retired texture view");
			{
				surface_cache shared(ctx, ram, 1 << 20, tight);
				const auto surface = shared.acquire({0x1000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				refused = false;
				try
				{
					shared.prepare(surface);
				}
				catch (const surface_budget_error&)
				{
					refused = true;
				}
				require(refused && !shared.status(surface).initialized, "Shared VM surface cache bypassed texture heap quota or published failed initialization");
			}
			view = nil;
			require(tight->statistics().reserved_bytes == 0, "Dropping retired texture view did not release actual native heap;inspect autorelease ownership");
			@autoreleasepool
			{
				const auto retried = limited.resolve(r);
				sample(retried, r, {.125f, .125f, 0, 1}, {128.f / 255, 64.f / 255, 128.f / 255, 1});
			}
			limited.clear();
			require(tight->statistics().reserved_bytes == 0, "Released immutable cache image did not return actual heap quota");
		}
		// Drop the shared view-cache owner below as well as this local
		// cache/view/shader scope, leaving only queued native owners.
		fill(plan_guest_texture(r), {255, 32, 64, 128});
		auto pending_budget = std::make_shared<heap_budget>(ctx, heap_bytes);
		const auto gate = [ctx.device() newSharedEvent];
		require(bool(gate), "Cache lifetime shared event creation failed");
		const auto pending_output = make_buffer(ctx, 512, buffer_storage::shared);
		auto pending = [&]
		{
			@autoreleasepool
			{
				guest_texture_cache local(ctx, ram, translate, 1 << 20, pending_budget, 1);
				const auto image = local.resolve(r);
				auto input_words = r;
				input_words.control1 = 0xa954;
				const auto decoded = decode_fragment_texture_state(input_words, image.metadata);
				fragment_inputs inputs;
				inputs.set_texture_state(0, image.texture, samplers, decoded);
				fragment_program_options options;
				configure_fragment_texture_state(options, 0, decoded);
				const auto program = code({.125f, .125f, 0, 1});
				fragment_shader shader(compile, program, options);
				render_pipeline_configuration config;
				config.colors[0] = MTLPixelFormatRGBA32Float;
				const auto pipeline = make_render_pipeline(compile, config, library_function(vertex, @"v"), shader);
				ctx.begin();
				ctx.wait_event(gate, 1);
				const auto binding = shader.prepare(ctx, {}, program, inputs);
				auto pass = [MTL4RenderPassDescriptor new];
				pass.colorAttachments[0].texture = target;
				pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				render_commands draw(ctx, pass);
				raster_state raster;
				raster.viewport = {0, 0, 2, 2, 0, 1};
				raster.scissor = {0, 0, 2, 2};
				draw.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, {}, binding);
				draw.end();
				texture_region region;
				region.size = MTLSizeMake(2, 2, 1);
				texture_to_buffer(ctx, target, region, pending_output, 0, 256);
				local.clear();
				fill(plan_guest_texture(r), {255, 128, 64, 128});
				return ctx.submit();
			}
		}();
		bool timed_out = false;
		try
		{
			pending.wait(std::chrono::milliseconds(1));
		}
		catch (const std::runtime_error&)
		{
			timed_out = true;
		}
		require(timed_out && !pending.ready() && pending_budget->statistics().reserved_bytes == heap_bytes, "Queued cache image was not truly blocked/heap-accounted after cache destruction");
		samplers.texture_views().clear();
		require(pending_budget->statistics().reserved_bytes == heap_bytes, "Clearing cached views released a still-queued image's heap quota");
		{
			guest_texture_cache next(ctx, ram, translate, 1 << 20, pending_budget, 1);
			const auto reads = ram->reads;
			bool denied = false;
			try
			{
				(void)next.resolve(r);
			}
			catch (const heap_budget_error&)
			{
				denied = true;
			}
			require(denied && ram->reads == reads, "Pending retired image quota admitted conflicting new texture upload");
		}
		gate.signaledValue = 1;
		pending.wait();
		for (unsigned y = 0; y < 2; ++y)
			for (unsigned x = 0; x < 2; ++x)
				for (unsigned c = 0; c < 4; ++c)
					require(std::abs(static_cast<const float*>(pending_output.contents)[y * 64 + x * 4 + c] - (c == 3 ? 1.f : 32.f / 255)) < 2e-6f, "Pending immutable cached view changed after cache destruction and VM update");
		require(pending_budget->statistics().reserved_bytes == 0, "Completed queued cache image did not release retired shared quota");
		std::cout << "Native immutable texture cache hit/remap/UN/translated-alias/revision/invalidation and shared surface/texture/retired-view heap quota cases passed\n";
		std::cout << "Native immutable cache destruction/VM-change/GPU-timeout kept old sampled version and actual heap quota until completion\n";
	}
};
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			fixture f;
			f.depth_volume_cases();
			f.depth_volume_specials();
			f.depth_volume_owners();
			f.depth_volume_assembly_owners();
			f.vtc_cases();
			f.vtc_piece_cases();
			f.vtc_owners();
			f.depth_cube_cases();
			f.depth_cube_owners();
			f.assembly_cases();
			f.assembly_precision_cases();
			f.assembly_lifetime_cases();
			f.assembly_atlas_case();
			f.assembly_deep_atlas_case();
			f.assembly_channel_cases();
			f.assembly_piece_cases();
			f.assembly_alias_pieces();
			f.all();
			f.upload_work_cases();
			f.upload_job_cases();
			f.captured_lifetime_cases();
			f.versioned_cases();
			f.cache_cases();
			f.surface_alias_cases();
			f.native_snapshot_cache_cases();
			f.typeless_surface_cases();
			f.partial_surface_cases();
			f.block_surface_cases();
			f.clean_surface_cpu_changes();
			f.region_alias_cases();
			f.multisample_snapshot_cases();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
