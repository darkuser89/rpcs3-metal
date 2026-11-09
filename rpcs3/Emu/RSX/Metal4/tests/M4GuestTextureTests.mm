#include "M4GuestTexture.h"
#include "Emu/RSX/Common/tiled_dma_copy.hpp"
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
std::vector<std::byte> code(color coordinate, unsigned opcode = 23, float lod = 0, bool varying = false, unsigned lod_axis = 0)
{
	std::vector<std::byte> result;
	unsigned reg = 1;
	color lod_word{};
	lod_word.at(lod_axis) = lod;
	for (const auto literal : {coordinate, lod_word})
	{
		for (unsigned word : {(1u << 24) | (reg << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(result, word);
		for (float v : literal)
			append(result, std::bit_cast<unsigned>(v));
		++reg;
	}
	if (varying)
		for (unsigned word : {(1u << 24) | (1u << 1) | (3u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 1u, 0xe4u << 9, 0xe4u << 9})
			append(result, word);
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
	void sample(resolved_fragment_texture resource, const fragment_texture_registers& words, color coordinate, color expected,
		unsigned opcode = 23, float lod = 0, const fragment_context* varying = nullptr, unsigned lod_axis = 0,
		const texture_state_overrides& overrides = {}, const std::array<color, 4>* pixels = nullptr)
	{
		const auto program_metadata = program_sampler_plan::capture(words, resource.metadata);
		const auto storage_plan = plan_guest_texture(words);
		const unsigned source_depth = storage_plan.codec.depth ? static_cast<unsigned>(*storage_plan.codec.depth) :
			resource.metadata.depth ? resource.metadata.source_depth_format : 0;
		constexpr unsigned expected_class[]{1, 2, 8, 4, 16};
		require(program_metadata.format_class == expected_class[source_depth] &&
			program_metadata.samples == resource.texture.sampleCount && program_metadata.coordinates.scale == resource.metadata.coordinates.scale,
			"Native VM/alias snapshot lost source depth class or base coordinate transform");
		std::array<std::uint32_t, fragment_texture_method_base + 8> methods{};
		const std::array registers{words.offset, words.format, words.address, words.control0,
			words.control1, words.filter, words.image_rect, words.border_color};
		std::ranges::copy(registers, methods.begin() + fragment_texture_method_base);
		methods[fragment_texture_control3_base] = words.control3;
		const auto snapshot = fragment_texture_snapshot::capture(methods, 1);
		fragment_program_options options;
		fragment_inputs inputs;
		apply_fragment_texture_snapshot(snapshot, [&](unsigned, const auto& frozen)
		{
			require(frozen == words, "Fragment binding changed the frozen RSX words");
			return resource;
		}, ctx, samplers, options, inputs, overrides);
		const auto program = code(coordinate, opcode, lod, varying != nullptr, lod_axis);
		fragment_shader shader(compile, program, options);
		render_pipeline_configuration config;
		config.colors[0] = MTLPixelFormatRGBA32Float;
		const auto pipeline = make_render_pipeline(compile, config, library_function(vertex, @"v"), shader);
		ctx.begin();
		const auto bindings = shader.prepare(ctx, varying ? *varying : fragment_context{}, program, inputs);
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
					const auto want = pixels ? (*pixels)[y * 2 + x][c] : expected[c];
					if (!std::isfinite(actual) || std::abs(actual - want) > 2e-6f)
						throw std::runtime_error("Guest texture format=" + std::to_string((words.format >> 8) & 255) + " lane=" + std::to_string(c) + " actual=" + std::to_string(actual) + " expected=" + std::to_string(want));
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
						if (plan.swizzled && plan.codec.block.height == 1)
							offset = s.offset + morton_index({x, y, z}, {unsigned(std::bit_width(s.columns) - 1), unsigned(std::bit_width(s.rows) - 1), unsigned(std::bit_width(s.depth) - 1)}) * plan.codec.block.bytes;
						for (unsigned b = 0; b < block.size(); ++b)
							ram->bytes[offset + b] = std::byte(block[b]);
					}
	}
	captured_guest_texture capture_without_gpu(const fragment_texture_registers& words, std::size_t expected_reads = 1)
	{
		const auto bytes = upload_budget->statistics().reserved_bytes;
		const auto reads = ram->reads;
		// An active transaction makes any hidden begin/submit fail. Capture
		// must leave that transaction and native heap usage untouched.
		ctx.begin();
		auto captured = resolver.capture(words);
		ctx.cancel();
		require(upload_budget->statistics().reserved_bytes == bytes && ram->reads == reads + expected_reads,
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
					const auto fresh=f->textures->resolve_versioned(f->registers,*f->surfaces);verify(fresh.image,0,pieces,mode==7);require(fresh.image.texture!=frozen[0].image.texture&&f->surfaces->status(destination).deferred_copies==unsigned(mode==7),"Deferred RTT owner reused stale ready assembly or changed its queued authority");verify(job.finish()[0].image,0,pieces);require(f->textures->resolve_versioned(f->registers,*f->surfaces).image.texture==fresh.image.texture,"Old RTT job displaced deferred/cancelled-owner ready result");continue;
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
	void adjusted_pitch_cases()
	{
		const auto first=cases;
		unsigned profiles=0, refusals=0;
		const std::array<color,6> directions{{{1,0,0,1},{-1,0,0,1},{0,1,0,1},{0,-1,0,1},{0,0,1,1},{0,0,-1,1}}};
		for(auto dimension:{vertex_texture_dimension::one,vertex_texture_dimension::two,vertex_texture_dimension::three,vertex_texture_dimension::cube})
			for(unsigned mode:{0u,1u,2u})
				for(bool pixel_coordinates:{false,true})
				{
					const bool cube=dimension==vertex_texture_dimension::cube;
					if(cube&&pixel_coordinates)continue;
					const unsigned declared_width=cube&&mode==0?1:17;
					const unsigned declared_height=dimension==vertex_texture_dimension::one?1:mode==2?5:3;
					const unsigned declared_depth=dimension==vertex_texture_dimension::three?5:1;
					const unsigned width=mode==0?declared_width:declared_height==1?3:declared_height;
					const unsigned height=mode==0?1:declared_height, depth=mode==0?1:declared_depth;
					const unsigned pitch=4*width+(mode==2);
					auto r=words(0x85,dimension,true,declared_width,declared_height,declared_depth,6);
					r.control3=(declared_depth<<20)|(mode?pitch:0);
					if(pixel_coordinates)r.format|=0x40u<<8;
					const auto levels=std::bit_width(std::max({width,height,depth}));
					const unsigned faces=cube?6:1;
					const auto plan=plan_guest_texture(r);
					require(plan.adjusted_pitch&&plan.width==width&&plan.height==height&&plan.depth==depth&&
						plan.mips==levels&&plan.layers==faces,"Adjusted pitch did not produce legal native geometry");
					const auto red=[cube](unsigned l,unsigned f,unsigned x,unsigned y,unsigned z)
					{return 17+l*21+f*7+(cube?0:x*3+y*5+z*11);};
					ram->bytes.assign(65536,std::byte{0xcd});
					std::size_t cursor=0;
					for(unsigned f=0;f<faces;++f)for(unsigned l=0;l<levels;++l)
					{
						const auto w=std::max(1u,width>>l),h=std::max(1u,height>>l),d=std::max(1u,depth>>l);
						const auto& s=plan.subresources[f*levels+l];
						require(s.offset==cursor&&s.row_pitch==pitch&&s.storage_bytes==pitch*h*d,"Adjusted mip stride or face offset changed");
						for(unsigned z=0;z<d;++z)for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
						{
							const auto at=cursor+(z*h+y)*pitch+x*4;
							ram->bytes[at]=std::byte{255};ram->bytes[at+1]=std::byte(red(l,f,x,y,z));
							ram->bytes[at+2]=std::byte{53};ram->bytes[at+3]=std::byte{97};
						}
						cursor+=pitch*h*d;
					}
					require(plan.storage_bytes==cursor,"Adjusted snapshot extends past the last planned mip");
					const auto before=ram->bytes;
					const auto captured=capture_without_gpu(r);
					require(ram->bytes==before,"Adjusted capture changed VM padding or pixels");
					const auto image=realize_without_vm(captured);
					require(image.texture.width==width&&image.texture.height==height&&image.texture.depth==depth&&
						image.texture.mipmapLevelCount==levels&&image.metadata.actual_mips==levels,"Adjusted upload lost native dimensions");
					for(unsigned f=0;f<faces;++f)for(unsigned l=0;l<levels;++l)
					{
						const auto w=std::max(1u,width>>l),h=std::max(1u,height>>l),d=std::max(1u,depth>>l);
						color coordinate{(w-.5f)/w,(h-.5f)/h,(d-.5f)/d,1};
						if(pixel_coordinates){coordinate[0]*=width;coordinate[1]*=height;coordinate[2]*=depth;}
						sample(image,r,cube?directions[f]:coordinate,{red(l,f,w-1,h-1,d-1)/255.f,53.f/255,97.f/255,1},47,float(l));
					}
					require(ram->bytes.empty(),"Adjusted realization recreated detached VM storage");
					++profiles;
				}
		for(auto r:{words(0x85,vertex_texture_dimension::two,true,9,4),words(0x86,vertex_texture_dimension::two,true,16,8),
			words(0x85,vertex_texture_dimension::cube,true,8,8)})
		{
			r.control3=1u<<20 | (((r.format>>8)&0x9f)==0x86?7:3);
			if(r.format&4)r.control3=(1u<<20)|16u;
			const auto reads=ram->reads,translations=translated;
			bool rejected=false;
			try{(void)resolver.resolve(r);}catch(const std::invalid_argument&){rejected=true;}
			require(rejected&&ram->reads==reads&&translated==translations,"Invalid adjusted pitch reached VM");
			++refusals;
		}
		std::cout<<"Adjusted pitch native geometry:"<<profiles<<" profiles,"<<cases-first<<" exact sampled mip/face results,"<<refusals<<" pre-VM block/cube refusals passed\n";
	}
	void adjusted_pitch_aliases()
	{
		unsigned profiles=0;
		const auto first=cases;
		for(unsigned mode:{0u,1u,2u})
		{
			const auto r=[&]
			{
				auto words=fixture::words(0x85,vertex_texture_dimension::two,true,9,4,1,mode==2?3:1);
				words.control3=(1u<<20)|(mode?13:0);
				return words;
			}();
			const auto plan=plan_guest_texture(r);
			const cpu_capture_dispatch capture=[](const auto& work){work();};
			auto io=std::make_shared<io_mapping_epoch>();
			context gpu{1};
			auto ledger=std::make_shared<heap_budget>(gpu,8u<<20);
			gpu.set_buffer_budget(ledger);
			surface_cache surfaces(gpu,ram,8u<<20,ledger,capture);
			guest_texture_cache textures(gpu,ram,[](unsigned offset,unsigned,std::size_t){return 0x1000+offset-0x200;},1u<<20,ledger,64,{capture,io});
			fill(plan,{255,17,33,65});
			const auto old=textures.resolve(r);
			const auto reads=ram->reads;
			require(textures.resolve(r).texture==old.texture&&ram->reads==reads,"Adjusted VM texture lost its cache hit");
			ram->bytes[1]=std::byte{73};
			++ram->version;
			const auto updated=textures.resolve(r);
			require(updated.texture!=old.texture,"Adjusted texture missed a CPU revision change");
			sample(updated,r,{.5f/plan.width,.5f/plan.height,0,1},{73.f/255,33.f/255,65.f/255,1},47,0);
			const auto original=ram->bytes;
			for(const auto& s:plan.subresources)
			{
				const auto ea=static_cast<std::uint32_t>(0x1000+s.offset);
				const auto image=surfaces.acquire({ea,{s.width,s.height,1,surface_color::argb8,false},s.row_pitch,{}});
				surfaces.clear(image,0xf0,0xff104080u+(s.level<<16),0,255,{0,0,s.width,s.height});
				guest_surface_byte_cursor active(plan_guest_surface(ea,s.width,s.height,4,s.row_pitch));
				while(const auto range=active.next())ram->forbidden.push_back(*range);
			}
			const auto before=ram->reads;
			const auto native=textures.resolve_versioned(r,surfaces);
			require(native.image.texture!=updated.texture&&native.image.metadata.actual_mips==plan.mips,
				"Adjusted native aliases missed their corrected texture");
			for(const auto& s:plan.subresources)
				sample(native.image,r,{.75f,.75f,0,1},{(16.f+s.level)/255,64.f/255,128.f/255,1},47,s.level);
			require(ram->reads==before&&ram->bytes==original,"Adjusted native alias accessed or changed guest pixels");
			require(textures.resolve_versioned(r,surfaces).image.texture==native.image.texture,
				"Adjusted native alias lost its warm cache entry");
			ram->forbidden.clear();
			++profiles;
		}
		std::cout<<"Adjusted pitch cache and RTT:"<<profiles<<" profiles,"<<cases-first<<" exact samples with CPU invalidation, native single/mip alias reuse and no native VM capture passed\n";
	}
	void stored_border_cases()
	{
		const auto first = cases;
		constexpr unsigned width = 4, height = 3, pitch = 27;
		auto r = words(0x85, vertex_texture_dimension::two, true, width, height);
		r.format &= ~8u;
		r.address = (7u << 28) | 0x040404;
		r.control3 = (1u << 20) | pitch;
		r.border_color = 0xffeeddcc;
		const auto texel = [](float x, float y) -> color
		{
			return {(16 + 12 * x + 4 * y) / 255, (24 + 4 * x + 20 * y) / 255,
				(40 + 8 * x + 8 * y) / 255, 1};
		};
		ram->bytes.assign(pitch * (height + 2), std::byte{0xcd});
		for (unsigned y = 0; y < height + 2; ++y)
			for (unsigned x = 0; x < width + 2; ++x)
			{
				const auto value = texel(x, y);
				const std::size_t at = y * pitch + x * 4;
				ram->bytes[at] = std::byte{255};
				for (unsigned c = 0; c < 3; ++c)
					ram->bytes[at + c + 1] = std::byte(std::lround(value[c] * 255));
			}
		++ram->version;
		const auto original = ram->bytes;
		const auto image = resolver.resolve(r);
		require(image.metadata.stored_border_extent == std::array{width, height} && image.texture.width == width + 2 && image.texture.height == height + 2,
			"Linear guest texel ring was not allocated");
		for (unsigned y = 0; y < height + 2; ++y)
			for (unsigned x = 0; x < width + 2; ++x)
				sample(image, r, {(float(x) - .5f) / width, (float(y) - .5f) / height, 0, 1}, texel(x, y));
		sample(image, r, {-16, -32, 0, 1}, texel(0, 0));
		sample(image, r, {16, 32, 0, 1}, texel(width + 1, height + 1));
		auto edge = r;
		edge.address = (edge.address & ~15u) | 3u;
		sample(image, edge, {-16, -32, 0, 1}, texel(1, 0));
		sample(image, edge, {16, 32, 0, 1}, texel(width, height + 1));
		auto linear = r;
		linear.filter = 0x02020000;
		sample(image, linear, {0, 0, 0, 1}, texel(.5f, .5f));
		sample(image, linear, {1, 1, 0, 1}, texel(width + .5f, height + .5f));
		sample(image, linear, {0, .5f, 0, 1}, texel(.5f, 2));
		linear.address = edge.address;
		sample(image, linear, {0, 0, 0, 1}, texel(1, .5f));
		auto pixels = r;
		pixels.format |= 0x40u << 8;
		const auto pixel_image = resolver.resolve(pixels);
		sample(pixel_image, pixels, {-1, 1, 0, 1}, texel(0, 2));
		sample(pixel_image, pixels, {3, 1, 0, 2}, texel(2, 1), 24);
		require(ram->bytes == original, "Stored-border upload changed guest bytes or padding");
		const auto captured = capture_without_gpu(r);
		sample(realize_without_vm(captured), r, {2, 2, 0, 1}, texel(width + 1, height + 1), 47);
		ram->bytes = original;
		++ram->version;

		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 8u << 20);
		gpu.set_buffer_budget(ledger);
		surface_cache surfaces(gpu, ram, 8u << 20, ledger, capture);
		guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000 + offset - 0x200; },
			1u << 20, ledger, 64, {capture, std::make_shared<io_mapping_epoch>()});
		const std::array requests{r, pixels};
		const auto batch = textures.resolve_versioned_batch(requests, surfaces);
		require(batch.size() == 2 && batch[0].image.texture == batch[1].image.texture,
			"Normalized and pixel-coordinate requests did not coalesce their stored ring");
		sample(batch[1].image, pixels, {-2, -2, 0, 1}, texel(0, 0));
		const auto reads = ram->reads;
		require(textures.resolve_versioned(r, surfaces).image.texture == batch[0].image.texture && ram->reads == reads,
			"Stored ring lost its versioned cache hit");
		ram->bytes[1] = std::byte{144};
		++ram->version;
		require(!batch[0].revision.current(), "Border-only write left a stale texture receipt current");
		const auto updated = textures.resolve_versioned(r, surfaces);
		require(updated.image.texture != batch[0].image.texture, "Border-only write reused the old native texture");
		sample(updated.image, r, {-2, -2, 0, 1}, {144.f / 255, 24.f / 255, 40.f / 255, 1});
		sample(batch[0].image, r, {-2, -2, 0, 1}, texel(0, 0));

		const auto native = surfaces.acquire({0x1000 + (height + 1) * pitch + (width + 1) * 4,
			{1, 1, 1, surface_color::argb8, false}, 4, {}});
		surfaces.clear(native, 0xf0, 0xff104080, 0, 255, {0, 0, 1, 1});
		ram->forbidden.push_back({0x1000 + (height + 1) * pitch + (width + 1) * 4, 4});
		const auto vm_before_assembly = ram->bytes;
		const auto composed = textures.resolve_versioned(r, surfaces);
		require(composed.image.texture != updated.image.texture && composed.image.metadata.stored_border_extent == std::array{width, height} &&
			textures.statistics().assembly_vm_bytes == ((width + 2) * (height + 2) - 1) * 4,
			"Border-only render target did not supply exactly its owned texel");
		sample(composed.image, r, {8, 8, 0, 1}, {16.f / 255, 64.f / 255, 128.f / 255, 1});
		sample(composed.image, r, {-8, -8, 0, 1}, {144.f / 255, 24.f / 255, 40.f / 255, 1});
		sample(updated.image, r, {8, 8, 0, 1}, texel(width + 1, height + 1));
		require(ram->bytes == vm_before_assembly, "Stored border assembly wrote guest bytes");
		ram->forbidden.clear();
		const auto before = ram->reads;

		const auto addresses = translated;
		auto invalids = std::array{r, r, r, r};
		invalids[0].address = r.address & ~15u;
		invalids[1].control3 = (1u << 20) | 11u; // Two border texels and an incomplete interior texel.
		invalids[2].format |= 4u;
		invalids[3].address = r.address & ~(15u << 8);
		for (const auto& invalid : invalids)
		{
			bool refused = false;
			try { (void)resolver.resolve(invalid); }
			catch (const std::invalid_argument&) { refused = true; }
			require(refused && translated == addresses && ram->reads == before,
				"Unsupported stored-border layout reached guest translation or capture");
		}
		std::cout << "Linear stored texel ring: " << cases - first
			<< " exact native samples, four pre-VM refusals and one border-only RTT composition passed\n";
	}
	static unsigned morton_border_cell(unsigned x, unsigned y, unsigned x_bits, unsigned y_bits)
	{
		// Independent reference: spread each axis into even/odd bit lanes,
		// then append the unmatched bits of the longer axis.
		const unsigned shared = std::min(x_bits, y_bits);
		unsigned result = 0;
		for (unsigned bit = 0; bit < shared; ++bit)
			result |= ((x >> bit) & 1) << (2 * bit) | ((y >> bit) & 1) << (2 * bit + 1);
		return result | ((x_bits > y_bits ? x : y) >> shared) << (2 * shared);
	}
	void stored_border_mip_atlas_cases()
	{
		using namespace std::chrono_literals;
		context gpu(1);
		auto budget = std::make_shared<heap_budget>(gpu, 16u << 20);
		gpu.set_buffer_budget(budget);
		guest_texture_resolver upload(gpu, ram, [&](unsigned offset, unsigned location, std::size_t size)
		{
			require(!ram->forbid_vm_access && offset == 0x200 && location == 1 && size <= ram->bytes.size(), "Mip atlas changed VM translation");
			return 0x1000;
		}, 1u << 20, budget);
		std::size_t profiles = 0, levels = 0, words_checked = 0, stencil_checked = 0, gated = 0;
		const auto first_sample = cases;
		for (const auto [width, height, mips] : {std::array{9u, 5u, 4u}, {4u, 11u, 3u}, {2u, 1u, 2u}})
			for (const auto format : {0x85u, 0x90u, 0x91u, 0x92u, 0x93u})
				for (const bool linear : {false, true})
				{
					auto r = words(format, vertex_texture_dimension::two, linear, width, height, 1, mips);
					r.format &= ~8u;
					const auto bpp = describe_guest_texture_format(format).block.bytes;
					const auto pitch = (width + 2) * bpp + 1;
					if (linear) r.control3 = (1u << 20) | pitch;
					const auto plan = plan_guest_texture(r);
					const auto raw = [&](unsigned mip, unsigned x, unsigned y)
					{
						const unsigned n = mip * 23 + x * 3 + y * 11;
						if (format == 0x85) return 0xff000000u | (n << 16) | ((n ^ 71u) << 8) | (n ^ 23u);
						if (format == 0x92) return 0x2000u + n * 67;
						if (format == 0x93) return 0x5000u + n;
						const unsigned depth = format == 0x90 ? 0x400000u + n * 139 : std::bit_cast<unsigned>(.25f + n / 1024.f) >> 7;
						return (depth << 8) | (n ^ 17u);
					};
					ram->bytes.assign(plan.storage_bytes, std::byte{0xda});
					std::size_t source_offset = 0;
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
						const auto sw = std::bit_ceil(w + 8), sh = std::bit_ceil(h + 8);
						for (unsigned y = 0; y < h + 2; ++y)
							for (unsigned x = 0; x < w + 2; ++x)
							{
								const auto position = linear ? y * pitch + x * bpp :
									morton_border_cell(x + 3, y + 3, std::bit_width(sw) - 1, std::bit_width(sh) - 1) * bpp;
								for (unsigned byte = 0; byte < bpp; ++byte)
									ram->bytes.at(source_offset + position + byte) = std::byte(raw(mip, x, y) >> ((bpp - byte - 1) * 8));
							}
						source_offset += linear ? pitch * (h + 2) : sw * sh * bpp;
					}
					require(source_offset == ram->bytes.size(), "Independent atlas source size differs");
					++ram->version;
					auto capture = upload.capture(r);
					const auto reads = ram->reads;
					ram->forbid_vm_access = true;
					const auto image = [&]() -> resolved_fragment_texture
					{
						if (linear) return upload.realize(capture);
						const auto gate = [gpu.device() newSharedEvent];
						struct release_gate { id<MTLSharedEvent> event; ~release_gate() { event.signaledValue = 1; } } release{gate};
						const auto job = [&]
						{
							const std::array captures{std::move(capture)};
							return upload.enqueue_uploads(captures, std::array{guest_texture_upload_dependency{gate, 1}});
						}();
						bool timed_out = false;
						try { (void)job.finish(1ms); } catch (const submission_wait_timeout&) { timed_out = true; }
						require(timed_out && !job.ready(), "Mip atlas escaped its GPU gate");
						gate.signaledValue = 1;
						++gated;
						return job.finish().front();
					}();
					ram->forbid_vm_access = false;
					require(image.texture.mipmapLevelCount == 1 && image.metadata.actual_mips == mips && ram->reads == reads,
						"Atlas lost logical mip metadata or recaptured VM");
					const bool wide = width >= height, depth = format != 0x85, stencil = format == 0x90 || format == 0x91;
					const unsigned native_bytes = format == 0x92 ? 2u : 4u;
					unsigned cursor = 0;
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto columns = std::max(1u, width >> mip) + 2, rows = std::max(1u, height >> mip) + 2;
						texture_region region;
						region.size = MTLSizeMake(columns, rows, 1);
						if (mip)
						{
							region.origin = wide ? MTLOriginMake(cursor, height + 2, 0) : MTLOriginMake(width + 2, cursor, 0);
							cursor += wide ? columns : rows;
						}
						const auto result = make_buffer(gpu, 512 * rows, buffer_storage::shared);
						gpu.begin();
						texture_to_buffer(gpu, image.texture, region, result, 0, 256, 0, depth ? texture_plane::depth : texture_plane::all);
						if (stencil) texture_to_buffer(gpu, image.texture, region, result, 256 * rows, 256, 0, texture_plane::stencil);
						gpu.submit().wait();
						const auto bytes = map_buffer(result, 0, result.length);
						for (unsigned y = 0; y < rows; ++y)
							for (unsigned x = 0; x < columns; ++x)
							{
								const auto word = raw(mip, x, y);
								unsigned actual = 0;
								std::memcpy(&actual, bytes.data() + y * 256 + x * native_bytes, native_bytes);
								const auto expected = format == 0x90 ? std::bit_cast<unsigned>(float(word >> 8) / 16777215.f) :
									format == 0x91 ? (word >> 8) << 7 : format == 0x93 ? (word << 11) + (120u << 23) : word;
								require(actual == expected, "Native atlas changed a color or depth texel");
								if (stencil)
								{
									require(bytes[(rows + y) * 256 + x] == std::byte(word & 255), "Native atlas changed a stencil texel");
									++stencil_checked;
								}
								++words_checked;
							}
						++levels;
					}
					const unsigned mip = mips - 1;
					r.address = (7u << 28) | 0x040404;
					const auto word = raw(mip, 0, 0);
					if (format == 0x85)
						sample(image, r, {-8, -8, 0, 1}, {float((word >> 16) & 255) / 255, float((word >> 8) & 255) / 255,
							float(word & 255) / 255, 1}, 47, float(mip));
					else
					{
						const float depth = format == 0x90 ? float(word >> 8) / 16777215.f : format == 0x91 ?
							std::bit_cast<float>((word >> 8) << 7) : format == 0x92 ? float(word) / 65535 :
							std::bit_cast<float>((word << 11) + (120u << 23));
						sample(image, r, {-8, -8, 0, 1}, {depth, depth, depth, depth}, 47, float(mip));
						r.address = (1u << 28) | 0x040404;
						r.control0 = (1u << 31) | (mip * 256u << 19) | (mip * 256u << 7);
						sample(image, r, {-8, -8, depth + 1.f / 1024, 1}, {1, 1, 1, 1});
						sample(image, r, {-8, -8, depth - 1.f / 1024, 1}, {0, 0, 0, 0});
					}
					++profiles;
				}
		std::cout << "Stored-border mip atlas: " << profiles << " linear/Morton profiles, " << levels << " rings, "
			<< words_checked << " exact color/depth words, " << stencil_checked << " exact stencil bytes, "
			<< gated << " gated immutable uploads passed\n";
		std::cout << "Stored-border atlas formats: " << cases - first_sample
			<< " point color, depth and shadow RSX samples passed\n";
	}
	void stored_border_1d_cases()
	{
		const auto first = cases;
		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 16u << 20);
		gpu.set_buffer_budget(ledger);
		std::size_t profiles = 0, rings = 0, active_bytes = 0, stencil_bytes = 0, assembled = 0;
		for (const auto [width, mips] : {std::array{1u, 1u}, {5u, 3u}, {9u, 4u}})
			for (const unsigned format : {0x85u, 0x90u, 0x91u, 0x92u, 0x93u})
				for (const bool linear : {false, true})
				{
					auto r = words(format, vertex_texture_dimension::one, linear, width, 1, 1, mips);
					r.format &= ~8u;
					r.address = (7u << 28) | 0x040404;
					const auto bpp = static_cast<unsigned>(describe_guest_texture_format(format).block.bytes);
					const auto pitch = (width + 2) * bpp + 1;
					if (linear) r.control3 = (1u << 20) | pitch;
					const auto plan = plan_guest_texture(r);
					ram->bytes.assign(plan.storage_bytes, std::byte{0xcd});
					std::vector<bool> active(plan.storage_bytes);
					std::vector<unsigned> starts;
					const auto raw = [format](unsigned mip, unsigned x)
					{
						const unsigned n = mip * 24 + x * 8;
						if (format == 0x85) return 0xff000000u | ((16 + n) << 16);
						if (format == 0x92) return 0x2000u + n * 128;
						if (format == 0x93) return 0x5000u + n * 8;
						const unsigned depth = format == 0x90 ? 0x400000u + n * 4096 :
							std::bit_cast<unsigned>(.25f + n / 512.f) >> 7;
						return (depth << 8) | n;
					};
					const auto expected = [&](unsigned mip, unsigned x) -> color
					{
						const auto value = raw(mip, x);
						if (format == 0x85) return {float((value >> 16) & 255) / 255, 0, 0, 1};
						const float depth = format == 0x90 ? float(value >> 8) / 16777215.f : format == 0x91 ?
							std::bit_cast<float>((value >> 8) << 7) : format == 0x92 ? float(value) / 65535.f :
							std::bit_cast<float>((value << 11) + (120u << 23));
						return {depth, depth, depth, depth};
					};
					unsigned offset = 0, touched = 0;
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip);
						const auto sw = std::bit_ceil(w + 8);
						const auto byte_at = [&](unsigned x)
						{
							return offset + (linear ? pitch + x * bpp : morton_border_cell(x + 3, 4, std::bit_width(sw) - 1, 4) * bpp);
						};
						starts.push_back(byte_at(0));
						for (unsigned x = 0; x < w + 2; ++x)
							for (unsigned byte = 0; byte < bpp; ++byte)
							{
								const auto at = byte_at(x) + byte;
								active.at(at) = true;
								ram->bytes.at(at) = std::byte(raw(mip, x) >> ((bpp - 1 - byte) * 8));
								++touched;
							}
						offset += linear ? pitch * 3 : sw * 16 * bpp;
					}
					require(offset == ram->bytes.size(), "1D guest storage disagrees with independent strides");
					for (unsigned at = 0; at < offset;)
					{
						const auto start = at++;
						while (at < offset && active[at] == active[start]) ++at;
						if (!active[start]) ram->forbidden.push_back({0x1000 + start, at - start});
					}
					++ram->version;
					const auto original = ram->bytes;
					surface_cache surfaces(gpu, ram, 16u << 20, ledger, capture);
					guest_texture_cache textures(gpu, ram, [](unsigned ea, unsigned, std::size_t) { return 0x1000 + ea - 0x200; },
						1u << 20, ledger, 16, {capture, std::make_shared<io_mapping_epoch>()});
					auto pixels = r;
					pixels.format |= 0x40u << 8;
					const auto batch = textures.resolve_versioned_batch(std::array{r, pixels}, surfaces);
					const auto image = batch[0].image;
					require(image.texture == batch[1].image.texture && image.texture.width == width + 2 &&
						image.texture.height == mips && image.texture.mipmapLevelCount == 1 && image.metadata.actual_mips == mips,
						"Versioned 1D storage lost or duplicated a mip row");
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip);
						sample(image, r, {-4, 8, 0, 1}, expected(mip, 0), 47, float(mip));
						sample(image, r, {4, -8, 0, 1}, expected(mip, w + 1), 47, float(mip));
						sample(image, pixels, {width * .375f, -8, 0, 1}, expected(mip, 1 + unsigned(w * .375f)), 47, float(mip));
						if (format != 0x85)
						{
							auto shadow = r;
							shadow.address = (1u << 28) | 0x040404;
							shadow.control0 = (1u << 31) | (mip * 256u << 19) | (mip * 256u << 7);
							const auto depth = expected(mip, 0)[0];
							sample(image, shadow, {-4, -8, depth - 1.f / 1024, 1}, {0, 0, 0, 0});
							sample(image, shadow, {-4, 8, depth + 1.f / 1024, 1}, {1, 1, 1, 1});
						}
						if (format == 0x90 || format == 0x91)
						{
							const auto stencil = make_buffer(gpu, 256, buffer_storage::shared);
							texture_region area;
							area.origin = MTLOriginMake(0, mip, 0);
							area.size = MTLSizeMake(w + 2, 1, 1);
							gpu.begin();
							texture_to_buffer(gpu, image.texture, area, stencil, 0, 256, 0, texture_plane::stencil);
							gpu.submit().wait();
							const auto values = map_buffer(stencil, 0, w + 2);
							for (unsigned x = 0; x < w + 2; ++x)
								require(values[x] == std::byte(raw(mip, x) & 255), "1D mip row lost its stencil byte");
							stencil_bytes += w + 2;
						}
						++rings;
					}
					if (format == 0x85)
					{
						auto filter = r;
						filter.filter = (2u << 24) | (4u << 16);
						const auto left = expected(mips - 1, 0), inner = expected(mips - 1, 1);
						sample(image, filter, {0, 8, 0, 1}, {.5f * (left[0] + inner[0]), 0, 0, 1}, 47, float(mips - 1));
						auto repeat = r;
						repeat.address = (r.address & ~15u) | 1u;
						sample(image, repeat, {-0.0625f, -8, 0, 1}, expected(0, width));
						const auto address = 0x1000 + starts.back();
						const auto native = surfaces.acquire({address, {1, 1, 1, surface_color::argb8, false}, 4, {}});
						surfaces.clear(native, 0xf0, 0xffcc0000, 0, 255, {0, 0, 1, 1});
						ram->forbidden.push_back({address, 4});
						const auto mixed = textures.resolve_versioned(r, surfaces);
						require(mixed.image.texture != image.texture && textures.statistics().assemblies_created == 1 &&
							textures.statistics().assembly_vm_bytes == touched - 4, "1D RTT did not capture exactly its active holes");
						sample(mixed.image, r, {-4, 8, 0, 1}, {.8f, 0, 0, 1}, 47, float(mips - 1));
						sample(image, r, {-4, 8, 0, 1}, expected(mips - 1, 0), 47, float(mips - 1));
						const auto reads = ram->reads;
						require(textures.resolve_versioned(r, surfaces).image.texture == mixed.image.texture &&
							ram->reads == reads, "Warm 1D RTT recaptured the mip rows");
						++assembled;
					}
					require(ram->bytes == original, "1D border processing changed VM bytes");
					ram->forbidden.clear();
					active_bytes += touched;
					++profiles;
				}
		std::cout << "Stored 1D borders: " << profiles << " linear/Morton profiles, " << rings
			<< " mip rows, " << active_bytes << " active source bytes, " << cases - first
			<< " RSX samples, " << stencil_bytes << " stencil bytes and " << assembled << " mixed native/VM assemblies passed\n";
	}
	void stored_border_anisotropic_cases()
	{
		const auto first = cases;
		using D = vertex_texture_dimension;
		const auto rgba = [](float value) -> color { return {value, value, value, value}; };
		const auto fill = [&](const guest_texture_layout& plan, auto value)
		{
			ram->bytes.assign(plan.storage_bytes, std::byte{0xcd});
			for (const auto& mip : plan.subresources)
			{
				const auto image = guest_texture_image_subresource(plan, mip);
				for (unsigned z = 0; z < image.depth; ++z)
					for (unsigned y = 0; y < image.height; ++y)
						for (unsigned x = 0; x < image.width; ++x)
						{
							const auto word = value(mip.level, x, y, z);
							const auto at = guest_texture_block_offset(plan, image, x, y, z);
							for (unsigned byte = 0; byte < 4; ++byte)
								ram->bytes.at(at + byte) = std::byte(word >> ((3 - byte) * 8));
						}
			}
			++ram->version;
		};
		unsigned profiles = 0, shadow_samples = 0;
		for (const auto dimension : {D::one, D::two, D::three, D::cube})
			for (const bool linear : {false, true})
				for (const unsigned mips : {1u, 3u})
					for (const unsigned format : {0x9cu, 0x91u})
					{
						const auto height = dimension == D::cube ? 8u : dimension == D::one ? 1u : 4u;
						const auto depth = dimension == D::three ? 4u : 1u;
						auto r = words(format, dimension, linear, 8, height, depth, mips);
						r.format &= ~8u;
						r.address = (7u << 28) | 0x030404;
						const auto plan = plan_guest_texture(r);
						fill(plan, [&](unsigned mip, unsigned x, unsigned, unsigned z)
						{
							const float value = mip == 2 ? .875f : mip == 1 ? .625f :
								x == 3 || (dimension == D::three && z == 1) ? .75f : .25f;
							const auto bits = std::bit_cast<unsigned>(value);
							return format == 0x91 ? (bits >> 7) << 8 : bits;
						});
						const auto image = resolver.resolve(r);
						const color coord = dimension == D::cube ? color{0, 0, 1, 1} : color{.5f, .5f, .625f, 1};
						const float gradient = dimension == D::cube ? 1.f : .5f;
						for (const auto [setting, expected] : {std::pair{0u, mips > 1 ? .875f : .25f},
							{1u, mips > 1 ? .625f : .25f}, {2u, .375f}, {7u, .375f}})
						{
							auto state = r;
							state.control0 |= setting << 4;
							sample(image, state, coord, rgba(expected), 25, gradient);
							sample(image, state, coord, rgba(.25f), 47, 0);
						}
						auto anisotropic = r;
						anisotropic.control0 |= 2u << 4;
						if (dimension != D::cube)
						{
							anisotropic.format |= 0x40u << 8;
							sample(image, anisotropic, {4, height * .5f, depth * .625f, 1}, rgba(.375f), 25, 4);
						}
						if (dimension == D::three)
							sample(image, anisotropic, {4, height * .5f, depth * .5f, 1}, rgba(.375f), 25, 4, nullptr, 2);
						++profiles;
					}

		for (const bool linear : {false, true})
		{
			auto r = words(0x91, D::two, linear, 8, 1);
			r.format &= ~8u;
			r.address = (1u << 28) | 0x030301; // Depth less than reference, repeat X.
			r.filter = (1u << 24) | (1u << 16);
			r.control0 |= 2u << 4;
			const auto plan = plan_guest_texture(r);
			fill(plan, [](unsigned, unsigned x, unsigned, unsigned)
			{
				return (std::bit_cast<unsigned>((x == 1 || x == 5) ? .75f : .25f) >> 7) << 8;
			});
			const auto image = resolver.resolve(r);
			fragment_context varying;
			varying.wpos_scale = .5f;
			sample(image, r, {0, 0, .5f, 1}, rgba(.75f), 23, 0, &varying);
			++shadow_samples;
		}
		// A one-level texture with a 16-texel footprint makes each hardware
		// anisotropy setting produce an independently predictable sample.
		auto caps = words(0x9c, D::two, true, 32, 1);
		caps.format &= ~8u;
		caps.address = (7u << 28) | 0x030303;
		caps.filter = (1u << 24) | (1u << 16);
		const auto cap_plan = plan_guest_texture(caps);
		fill(cap_plan, [](unsigned, unsigned x, unsigned, unsigned)
		{
			return std::bit_cast<unsigned>(float(x * x) / 1024.f);
		});
		const auto cap_image = resolver.resolve(caps);
		constexpr std::array factors{1u, 2u, 4u, 6u, 8u, 10u, 12u, 16u};
		const auto oracle = [&](unsigned count)
		{
			float total = 0;
			for (unsigned i = 1; i <= count; ++i)
			{
				const auto x = 1u + static_cast<unsigned>(std::floor(16.125 + 16.0 * (double(i) / (count + 1) - .5)));
				total += float(x * x) / 1024.f;
			}
			return rgba(total / count);
		};
		const color coord{16.125f / 32, .5f, 0, 1};
		for (unsigned i = 0; i < factors.size(); ++i)
		{
			caps.control0 = (caps.control0 & ~112u) | (i << 4);
			sample(cap_image, caps, coord, oracle(factors[i]), 25, .5f);
		}
		sample(cap_image, caps, coord, oracle(2), 25, .5f, nullptr, 0, {0, 2});
		std::cout << "Stored-border anisotropy: " << profiles << " linear/Morton dimension/mip/format profiles, "
			<< cases - first << " exact shader samples including " << shadow_samples
			<< " anisotropic shadow samples, all eight factors and a configuration override passed\n";
	}
	void stored_border_cube_cases()
	{
		const auto first = cases;
		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 32u << 20);
		gpu.set_buffer_budget(ledger);
		const auto translate = [](unsigned ea, unsigned, std::size_t) { return 0x1000 + ea - 0x200; };
		guest_texture_resolver upload(gpu, ram, translate, 1u << 20, ledger, capture);
		unsigned profiles = 0, planes = 0, words_checked = 0, stencil_checked = 0;
		for (const auto [side, mips] : {std::array{1u, 1u}, {5u, 3u}})
			for (const unsigned format : {0x85u, 0x90u, 0x91u, 0x92u, 0x93u})
				for (const bool linear : {false, true})
				{
					auto r = words(format, vertex_texture_dimension::cube, linear, side, side, 1, mips);
					r.format &= ~8u;
					r.address = (7u << 28) | 0x040404;
					const auto codec = describe_guest_texture_format(format);
					const unsigned bpp = codec.block.bytes, pitch = (side + 2) * bpp + 1;
					if (linear) r.control3 = (1u << 20) | pitch;
					const auto plan = plan_guest_texture(r);
					ram->bytes.assign(plan.storage_bytes, std::byte{0xdc});
					std::vector<bool> active(plan.storage_bytes);
					std::vector<unsigned> bases;
					const auto raw = [format](unsigned face, unsigned mip, unsigned x, unsigned y)
					{
						const unsigned n = face * 19 + mip * 23 + x * 3 + y * 7;
						if (format == 0x85) return 0xff000000u | (n << 16) | ((n ^ 17u) << 8) | (n ^ 31u);
						if (format == 0x92) return 0x4000u + n * 64;
						if (format == 0x93) return 0x5000u + n * 8;
						const unsigned depth = format == 0x90 ? 0x400000u + n * 4096 :
							std::bit_cast<unsigned>(.25f + n / 512.f) >> 7;
						return (depth << 8) | (n ^ 7u);
					};
					const auto address = [&](unsigned face, unsigned mip, unsigned x, unsigned y)
					{
						const auto w = std::max(1u, side >> mip);
						const auto log = std::bit_width(std::bit_ceil(w + 8)) - 1;
						const auto texel = linear ? y * pitch + x * bpp : morton_border_cell(x + 3, y + 3, log, log) * bpp;
						return bases.at(face * mips + mip) + texel;
					};
					unsigned offset = 0, active_bytes = 0;
					for (unsigned face = 0; face < 6; ++face)
					{
						for (unsigned mip = 0; mip < mips; ++mip)
						{
							bases.push_back(offset);
							const auto w = std::max(1u, side >> mip), padded = std::bit_ceil(w + 8);
							for (unsigned y = 0; y < w + 2; ++y)
								for (unsigned x = 0; x < w + 2; ++x)
									for (unsigned byte = 0; byte < bpp; ++byte)
									{
										const auto at = address(face, mip, x, y) + byte;
										require(!active.at(at), "Cube border oracle overlaps active bytes");
										active[at] = true;
										ram->bytes[at] = std::byte(raw(face, mip, x, y) >> ((bpp - 1 - byte) * 8));
										++active_bytes;
									}
							offset += linear ? pitch * (w + 2) : padded * padded * bpp;
						}
						if (!linear && face < 5) offset = (offset + 127u) & ~127u;
					}
					require(offset == ram->bytes.size(), "Stored cube allocation disagrees with independent strides");
					for (unsigned at = 0; at < offset;)
					{
						const auto start = at++;
						while (at < offset && active[at] == active[start]) ++at;
						if (!active[start]) ram->forbidden.push_back({0x1000 + start, at - start});
					}
					++ram->version;
					const auto original = ram->bytes;
					const auto captured = upload.capture(r);
					const auto reads_before = ram->reads;
					ram->forbid_vm_access = true;
					const auto uploaded = upload.realize(captured);
					ram->forbid_vm_access = false;
					require(ram->reads == reads_before && uploaded.texture.textureType == MTLTextureType2DArray &&
						uploaded.texture.arrayLength == 6 && uploaded.texture.mipmapLevelCount == 1 &&
						uploaded.metadata.actual_mips == mips, "Detached cube storage recaptured VM or lost a face");
					const unsigned native_face = 4, native_mip = mips - 1;
					const auto atlas = plan_stored_border_atlas(side, side, mips, false);
					const auto verify = [&](id<MTLTexture> image, bool with_owner)
					{
						for (unsigned face = 0; face < 6; ++face)
							for (unsigned mip = 0; mip < mips; ++mip)
							{
								const auto w = std::max(1u, side >> mip) + 2;
								const auto out = make_buffer(gpu, w * 512, buffer_storage::shared);
								texture_region area;
								area.origin = MTLOriginMake(atlas.origins[mip][0], atlas.origins[mip][1], 0);
								area.size = MTLSizeMake(w, w, 1);
								area.slice = face;
								gpu.begin();
								texture_to_buffer(gpu, image, area, out, 0, 256, 0, codec.depth ? texture_plane::depth : texture_plane::all);
								const bool stencil = format == 0x90 || format == 0x91;
								if (stencil) texture_to_buffer(gpu, image, area, out, w * 256, 256, 0, texture_plane::stencil);
								gpu.submit().wait();
								const auto data = map_buffer(out, 0, w * 512);
								const unsigned stride = format == 0x92 ? 2 : 4;
								for (unsigned y = 0; y < w; ++y)
									for (unsigned x = 0; x < w; ++x)
									{
										const bool native = with_owner && face == native_face && mip == native_mip && x == 1 && y == 1;
										const auto word = raw(face, mip, x, y);
										unsigned value = 0;
										std::memcpy(&value, data.data() + y * 256 + x * stride, stride);
										const auto expected = format == 0x90 ? std::bit_cast<unsigned>(float(word >> 8) / 16777215.f) :
											format == 0x91 ? (word >> 8) << 7 : format == 0x93 ? (word << 11) + (120u << 23) : word;
										const auto written = format == 0x85 ? 0xff204060u : format == 0x92 ? 65535u : std::bit_cast<unsigned>(1.f);
										require(value == (native ? written : expected), "Native cube lost a color/depth ring word");
										if (stencil)
										{
											require(data[(w + y) * 256 + x] == std::byte(native ? 53u : word & 255u), "Native cube lost a stencil byte");
											++stencil_checked;
										}
										++words_checked;
									}
								++planes;
							}
					};
					verify(uploaded.texture, false);
					const auto expected = [&](unsigned face, unsigned mip, unsigned x, unsigned y) -> color
					{
						const auto word = raw(face, mip, x, y);
						if (!codec.depth) return {float((word >> 16) & 255) / 255, float((word >> 8) & 255) / 255, float(word & 255) / 255, 1};
						const float value = format == 0x90 ? float(word >> 8) / 16777215.f : format == 0x91 ?
							std::bit_cast<float>((word >> 8) << 7) : format == 0x92 ? float(word) / 65535.f :
							std::bit_cast<float>((word << 11) + (120u << 23));
						return {value, value, value, value};
					};
					const auto direction = [](unsigned face, float s, float t)
					{
						const float u = 2 * s - 1, v = 2 * t - 1;
						const std::array<color, 6> mapped{{{1,-v,-u,1}, {-1,-v,u,1}, {u,1,v,1},
							{u,-1,-v,1}, {u,-v,1,1}, {-u,-v,-1,1}}};
						return mapped.at(face);
					};
					for (unsigned face = 0; face < 6; ++face)
						for (unsigned mip = 0; mip < mips; ++mip)
						{
							const auto w = std::max(1u, side >> mip);
							const auto xy = direction(face, .225f, .775f);
							const auto pixel = expected(face, mip, 1 + unsigned(w * .225f), 1 + unsigned(w * .775f));
							sample(uploaded, r, xy, pixel, 47, float(mip));
							if (codec.depth)
							{
								auto shadow = r;
								shadow.address = (1u << 28) | 0x040404;
								shadow.control0 = (1u << 31) | (mip * 256u << 19) | (mip * 256u << 7);
								auto ref = xy;
								ref[3] = pixel[0] - 1.f / 1024;
								sample(uploaded, shadow, ref, {0, 0, 0, 0});
								ref[3] = pixel[0] + 1.f / 1024;
								sample(uploaded, shadow, ref, {1, 1, 1, 1});
							}
						}
					const auto center = (side / 2 + .5f) / side;
					const auto corner = direction(0, 1, center);
					sample(uploaded, r, corner, expected(0, 0, side + 1, side / 2 + 1));
					if (!codec.depth)
					{
						// Either side of the shared +X/-Z edge must select its
						// own oriented face before addressing the atlas.
						sample(uploaded, r, {1, 0, -.9375f, 1}, expected(0, 0, side, side / 2 + 1));
						sample(uploaded, r, {.9375f, 0, -1, 1}, expected(5, 0, 1, side / 2 + 1));
						auto clamped = r;
						clamped.address = (r.address & ~15u) | 3u;
						sample(uploaded, clamped, corner, expected(0, 0, side, side / 2 + 1));
						auto repeated = r;
						repeated.address = (r.address & ~15u) | 1u;
						sample(uploaded, repeated, corner, expected(0, 0, 1, side / 2 + 1));
						auto projected = direction(3, .225f, .775f);
						for (auto& component : projected) component *= 2;
						sample(uploaded, r, projected, expected(3, 0, 1 + unsigned(side * .225f), 1 + unsigned(side * .775f)), 24);
						if (mips > 1)
						{
							sample(uploaded, r, {1, 0, 0, 1}, expected(0, 0, 1 + side / 2, 1 + side / 2), 25, .8f, nullptr, 0);
							sample(uploaded, r, {1, 0, 0, 1}, expected(0, 1, 2, 2), 25, .8f, nullptr, 2);
						}
					}
					if (format == 0x91 || format == 0x93)
					{
						auto filter = r;
						filter.filter = (2u << 24) | (4u << 16);
						const auto edge = expected(0, 0, side, side / 2 + 1);
						const auto border = expected(0, 0, side + 1, side / 2 + 1);
						const float value = (edge[0] + border[0]) * .5f;
						sample(uploaded, filter, corner, {value, value, value, value});
						filter.address = (1u << 28) | 0x040404;
						auto ref = corner;
						ref[3] = value;
						sample(uploaded, filter, ref, {.5f, .5f, .5f, .5f});
						if (mips > 1)
						{
							filter = r;
							filter.filter = (1u << 24) | (5u << 16);
							const auto coarse = expected(0, 1, 3, 2);
							const float lod_value = .5f * (border[0] + coarse[0]);
							sample(uploaded, filter, corner, {lod_value, lod_value, lod_value, lod_value}, 47, .5f);
						}
					}
					surface_cache surfaces(gpu, ram, 32u << 20, ledger, capture);
					guest_texture_cache textures(gpu, ram, translate, 1u << 20, ledger, 16, {capture, std::make_shared<io_mapping_epoch>()});
					const auto ea = 0x1000 + address(native_face, native_mip, 1, 1);
					const surface_format fmt = codec.depth ? surface_format{*codec.depth} : surface_format{surface_color::argb8};
					const auto owner = surfaces.acquire({ea, {1, 1, 1, fmt, false}, bpp, {}});
					if (!codec.depth)
						surfaces.clear(owner, 0xf0, 0xff204060, 0, 255, {0, 0, 1, 1});
					else
						surfaces.draw(std::array{cached_surface_use{owner, true}}, [&](context& commands, auto images)
						{
							auto pass = [MTL4RenderPassDescriptor new];
							pass.depthAttachment.texture = images[0];
							pass.depthAttachment.loadAction = MTLLoadActionClear;
							pass.depthAttachment.storeAction = MTLStoreActionStore;
							pass.depthAttachment.clearDepth = 1;
							if (format <= 0x91)
							{
								pass.stencilAttachment.texture = images[0];
								pass.stencilAttachment.loadAction = MTLLoadActionClear;
								pass.stencilAttachment.storeAction = MTLStoreActionStore;
								pass.stencilAttachment.clearStencil = 53;
							}
							(void)commands.render(pass);
						});
					ram->forbidden.push_back({ea, bpp});
					const auto mixed = textures.resolve_versioned_batch(std::array{r}, surfaces).front();
					require(textures.statistics().assembly_vm_bytes == active_bytes - bpp && mixed.image.texture != uploaded.texture,
						"Cube ring assembly did not read exactly the VM holes");
					verify(mixed.image.texture, true);
					sample(mixed.image, r, direction(native_face, .5f, .5f),
						codec.depth ? color{1, 1, 1, 1} : color{32.f / 255, 64.f / 255, 96.f / 255, 1}, 47, float(native_mip));
					const auto reads = ram->reads;
					require(textures.resolve_versioned(r, surfaces).image.texture == mixed.image.texture &&
						ram->reads == reads && ram->bytes == original, "Warm cube assembly recaptured or changed guest bytes");
					ram->forbidden.clear();
					++profiles;
				}
		std::cout << "Stored cube borders: " << profiles << " linear/Morton profiles, " << planes << " face-mip planes, "
			<< words_checked << " exact color/depth words, " << stencil_checked << " stencil bytes and "
			<< profiles << " mixed native/VM assemblies passed\n";
		std::cout << "Stored cube sampling: " << cases - first
			<< " RSX color/depth/shadow samples through the regular Metal 4 binding passed\n";
	}
	void stored_border_volume_cases()
	{
		const auto first = cases;
		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 32u << 20);
		gpu.set_buffer_budget(ledger);
		const auto translate = [](unsigned ea, unsigned, std::size_t) { return 0x1000 + ea - 0x200; };
		guest_texture_resolver upload(gpu, ram, translate, 1u << 20, ledger, capture);
		unsigned profiles = 0, planes = 0, words_checked = 0, composed = 0;
		for (const auto [width, height, depth, mips] : {std::array{1u, 1u, 2u, 1u}, {5u, 4u, 4u, 3u}})
			for (const unsigned format : {0x85u, 0x90u, 0x91u, 0x92u, 0x93u})
				for (const bool linear : {false, true})
				{
					auto r = words(format, vertex_texture_dimension::three, linear, width, height, depth, mips);
					r.format &= ~8u;
					r.border_color = 0xff6080a0;
					const auto codec = describe_guest_texture_format(format);
					const unsigned bpp = codec.block.bytes;
					const auto pitch = (width + 2) * bpp + 1;
					if (linear) r.control3 = (depth << 20) | pitch;
					const auto plan = plan_guest_texture(r);
					ram->bytes.assign(plan.storage_bytes, std::byte{0xdc});
					std::vector<bool> active(plan.storage_bytes);
					std::vector<unsigned> bases;
					const auto raw = [format](unsigned mip, unsigned x, unsigned y, unsigned z)
					{
						const auto n = 20 * mip + 2 * x + 3 * y + 11 * z;
						if (format == 0x85) return 0xff000000u | (n << 16) | ((n ^ 17u) << 8) | (n ^ 31u);
						if (format == 0x92) return 0x4000u + n * 64;
						if (format == 0x93) return 0x5000u + n * 8;
						const unsigned depth_bits = format == 0x90 ? 0x400000u + n * 4096 :
							std::bit_cast<unsigned>(.25f + n / 512.f) >> 7;
						return (depth_bits << 8) | (n ^ 7u);
					};
					const auto address = [&](unsigned mip, unsigned x, unsigned y, unsigned z)
					{
						const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
						if (linear) return bases.at(mip) + (z * (h + 2) + y) * pitch + x * bpp;
						const std::array size{std::bit_ceil(w + 8), std::bit_ceil(h + 8), std::max(1u, depth >> mip)};
						const std::array coord{x + 3, y + 3, z};
						unsigned ordinal = 0, weight = 1;
						for (unsigned bit = 1; bit < std::max({size[0], size[1], size[2]}); bit <<= 1)
							for (unsigned axis = 0; axis < 3; ++axis)
								if (bit < size[axis])
								{
									if (coord[axis] & bit) ordinal += weight;
									weight <<= 1;
								}
						return bases.at(mip) + ordinal * bpp;
					};
					unsigned offset = 0, active_bytes = 0;
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						bases.push_back(offset);
						const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip), d = std::max(1u, depth >> mip);
						for (unsigned z = 0; z < d; ++z)
							for (unsigned y = 0; y < h + 2; ++y)
								for (unsigned x = 0; x < w + 2; ++x)
									for (unsigned byte = 0; byte < bpp; ++byte)
									{
										const auto at = address(mip, x, y, z) + byte;
										require(!active.at(at), "Volume border oracle maps two active bytes together");
										active[at] = true;
										ram->bytes[at] = std::byte(raw(mip, x, y, z) >> ((bpp - 1 - byte) * 8));
										++active_bytes;
									}
						offset += d * (linear ? pitch * (h + 2) : std::bit_ceil(w + 8) * std::bit_ceil(h + 8) * bpp);
					}
					require(offset == ram->bytes.size(), "Stored volume allocation disagrees with independent strides");
					for (unsigned at = 0; at < offset;)
					{
						const auto start = at++;
						while (at < offset && active[at] == active[start]) ++at;
						if (!active[start]) ram->forbidden.push_back({0x1000 + start, at - start});
					}
					++ram->version;
					const auto original = ram->bytes;
					const auto owned = upload.capture(r);
					const auto reads_before = ram->reads;
					ram->forbid_vm_access = true;
					const auto uploaded = upload.realize(owned);
					ram->forbid_vm_access = false;
					require(ram->reads == reads_before && uploaded.texture.textureType == MTLTextureType3D &&
						uploaded.texture.depth == depth && uploaded.texture.mipmapLevelCount == 1 && uploaded.metadata.actual_mips == mips,
						"Detached volume storage recaptured VM or has the wrong shape");

					const auto atlas = mips > 1 ? plan_stored_border_atlas(width, height, mips, false) : stored_border_mip_atlas{};
					const unsigned source_mip = mips > 1 ? 1 : 0, source_z = 1;
					const auto verify = [&](id<MTLTexture> image, bool with_owner)
					{
						for (unsigned mip = 0; mip < mips; ++mip)
						{
							const auto cols = std::max(1u, width >> mip) + 2, rows = std::max(1u, height >> mip) + 2;
							for (unsigned z = 0; z < std::max(1u, depth >> mip); ++z)
							{
								const auto out = make_buffer(gpu, rows * 256, buffer_storage::shared);
								texture_region area;
								area.origin = MTLOriginMake(atlas.origins[mip][0], atlas.origins[mip][1], z);
								area.size = MTLSizeMake(cols, rows, 1);
								gpu.begin();
								texture_to_buffer(gpu, image, area, out, 0, 256);
								gpu.submit().wait();
								const auto data = map_buffer(out, 0, rows * 256);
								for (unsigned y = 0; y < rows; ++y)
									for (unsigned x = 0; x < cols; ++x)
									{
										const bool native = with_owner && mip == source_mip && z == source_z && !x && !y;
										const auto word = raw(mip, x, y, z);
										const auto at = y * 256 + x * (codec.depth ? 8 : 4);
										unsigned value;
										std::memcpy(&value, data.data() + at, 4);
										if (!codec.depth)
											require(value == (native ? 0xff204060u : word), "Native volume lost an XY color ring texel");
										else
										{
											const auto expected = format == 0x90 ? std::bit_cast<unsigned>(float(word >> 8) / 16777215.f) :
												format == 0x91 ? (word >> 8) << 7 : format == 0x92 ? std::bit_cast<unsigned>(float(word) / 65535.f) :
												(word << 11) + (120u << 23);
											require(value == (native ? std::bit_cast<unsigned>(1.f) : expected), "Native volume lost an XY depth ring word");
											std::memcpy(&value, data.data() + at + 4, 4);
											const float stencil = format <= 0x91 ? (native ? 53.f : float(word & 255)) : 0.f;
											require(value == std::bit_cast<unsigned>(stencil), "Native volume lost an XY stencil word");
											++words_checked;
										}
										++words_checked;
									}
								++planes;
							}
						}
					};
					verify(uploaded.texture, false);
					const auto expected = [&](unsigned mip, unsigned x, unsigned y, unsigned z) -> color
					{
						const auto word = raw(mip, x, y, z);
						if (!codec.depth) return {float((word >> 16) & 255) / 255, float((word >> 8) & 255) / 255, float(word & 255) / 255, 1};
						const float value = format == 0x90 ? float(word >> 8) / 16777215.f : format == 0x91 ?
							std::bit_cast<float>((word >> 8) << 7) : format == 0x92 ? float(word) / 65535.f :
							std::bit_cast<float>((word << 11) + (120u << 23));
						return {value, value, value, value};
					};
					const auto average = [](color left, color right)
					{
						for (unsigned c = 0; c < left.size(); ++c) left[c] = (left[c] + right[c]) * .5f;
						return left;
					};
					auto border = r;
					border.address = (7u << 28) | 0x030404;
					auto pixels = border;
					pixels.format |= 0x40u << 8;
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip), d = std::max(1u, depth >> mip);
						sample(uploaded, border, {-2, -2, .75f, 1}, expected(mip, 0, 0, d * 3 / 4), 47, float(mip));
						sample(uploaded, border, {2, 2, .125f, 1}, expected(mip, w + 1, h + 1, 0), 47, float(mip));
						sample(uploaded, pixels, {-2, -2, depth * .75f, 1}, expected(mip, 0, 0, d * 3 / 4), 47, float(mip));
					}
					sample(uploaded, border, {-4, -4, .25f, 2}, expected(0, 0, 0, 0), 24);
					auto z_border = border;
					z_border.address = (7u << 28) | 0x040404;
					const color z_color = codec.depth ? color{96.f / 255, 96.f / 255, 96.f / 255, 96.f / 255} :
						color{96.f / 255, 128.f / 255, 160.f / 255, 1};
					sample(uploaded, z_border, {-2, -2, 2, 1}, z_color);
					z_border.filter = (2u << 24) | (4u << 16);
					sample(uploaded, z_border, {-2, -2, 0, 1}, average(z_color, expected(0, 0, 0, 0)));
					auto repeat = z_border;
					repeat.address = (7u << 28) | 0x010404;
					sample(uploaded, repeat, {-2, -2, 0, 1}, average(expected(0, 0, 0, 0), expected(0, 0, 0, depth - 1)));
					if (mips > 1)
					{
						sample(uploaded, border, {-2, -2, .75f, 1}, expected(1, 0, 0, 1), 25, .5f, nullptr, 2);
						auto trilinear = border;
						trilinear.filter = (1u << 24) | (5u << 16);
						sample(uploaded, trilinear, {-2, -2, .75f, 1}, average(expected(1, 0, 0, 1), expected(2, 0, 0, 0)), 47, 1.5f);
						auto z_linear = border;
						z_linear.filter = (2u << 24) | (4u << 16);
						sample(uploaded, z_linear, {-2, -2, .5f, 1}, average(expected(1, 0, 0, 0), expected(1, 0, 0, 1)), 47, 1);
					}
					if (!codec.depth)
						for (unsigned wrap = 1; wrap <= 8; ++wrap)
						{
							auto wrapped = border;
							wrapped.address = (7u << 28) | (wrap << 16) | 0x0404;
							const auto z = wrap == 1 ? depth - 1 : 0;
							sample(uploaded, wrapped, {-2, -2, -.125f, 1}, wrap == 4 ? z_color : expected(0, 0, 0, z));
						}
					surface_cache surfaces(gpu, ram, 32u << 20, ledger, capture);
					guest_texture_cache textures(gpu, ram, translate, 1u << 20, ledger, 16, {capture, std::make_shared<io_mapping_epoch>()});
					const auto ea = 0x1000 + address(source_mip, 0, 0, source_z);
					const surface_format fmt = codec.depth ? surface_format{*codec.depth} : surface_format{surface_color::argb8};
					const auto owner = surfaces.acquire({ea, {1, 1, 1, fmt, false}, bpp, {}});
					if (!codec.depth)
						surfaces.clear(owner, 0xf0, 0xff204060, 0, 255, {0, 0, 1, 1});
					else
						surfaces.draw(std::array{cached_surface_use{owner, true}}, [&](context& commands, auto images)
						{
							auto pass = [MTL4RenderPassDescriptor new];
							pass.depthAttachment.texture = images[0];
							pass.depthAttachment.loadAction = MTLLoadActionClear;
							pass.depthAttachment.storeAction = MTLStoreActionStore;
							pass.depthAttachment.clearDepth = 1;
							if (format <= 0x91)
							{
								pass.stencilAttachment.texture = images[0];
								pass.stencilAttachment.loadAction = MTLLoadActionClear;
								pass.stencilAttachment.storeAction = MTLStoreActionStore;
								pass.stencilAttachment.clearStencil = 53;
							}
							(void)commands.render(pass);
						});
					ram->forbidden.push_back({ea, bpp});
					const auto mixed = textures.resolve_versioned(r, surfaces);
					require(textures.statistics().assembly_vm_bytes == active_bytes - bpp && mixed.image.texture != uploaded.texture,
						"Volume ring assembly did not read exactly the VM holes");
					verify(mixed.image.texture, true);
					const auto position = 1.5f / std::max(1u, depth >> source_mip);
					sample(mixed.image, border, {-2, -2, position, 1}, codec.depth ? color{1, 1, 1, 1} :
						color{32.f / 255, 64.f / 255, 96.f / 255, 1}, 47, float(source_mip));
					const auto reads = ram->reads;
					require(textures.resolve_versioned(r, surfaces).image.texture == mixed.image.texture &&
						ram->reads == reads && ram->bytes == original, "Warm volume assembly recaptured or changed guest bytes");
					ram->forbidden.clear();
					++profiles;
					++composed;
				}
		std::cout << "Stored volume borders: " << profiles << " linear/Morton profiles, " << planes << " XY mip planes, "
			<< words_checked << " exact native color/depth/stencil words and " << composed << " mixed native/VM assemblies passed\n";
		std::cout << "Stored volume sampling: " << cases - first << " RSX color/depth samples through the regular Metal 4 binding passed\n";
	}
	void morton_paired_cases()
	{
		const auto before = cases;
		context gpu{1}; auto ledger = std::make_shared<heap_budget>(gpu, 16u << 20); gpu.set_buffer_budget(ledger);
		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		unsigned profiles = 0, planes = 0, texels = 0, mixed = 0, full = 0;
		std::size_t holes = 0;
		const auto address = [](unsigned x, unsigned y, unsigned z, const std::array<std::size_t, 3>& storage)
		{
			std::size_t result = 0, weight = 1;
			for (unsigned bit = 1; bit < std::max({storage[0], storage[1], storage[2]}); bit *= 2)
				for (unsigned axis = 0; axis < 3; ++axis) if (bit < storage[axis]) { if (std::array{x,y,z}[axis] & bit) result += weight; weight *= 2; }
			return result;
		};
		for (const unsigned format : {0x8du, 0x8eu})
			for (const auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::three, vertex_texture_dimension::cube})
				for (const bool border : {false, true})
					for (unsigned shape = 0; shape < 4; ++shape)
						@autoreleasepool
						{
							if (!border && dimension == vertex_texture_dimension::cube && shape == 3) continue;
							const unsigned width = border ? std::array{1u,4u,5u,15u}[shape] : std::array{1u,4u,8u,3u}[shape];
							const unsigned height = dimension == vertex_texture_dimension::one ? 1 : dimension == vertex_texture_dimension::cube ? width : 4;
							const unsigned depth = dimension == vertex_texture_dimension::three ? 4 : 1;
							auto r = words(format, dimension, false, width, height, depth, shape ? 3 : 1);
							if (border) r.format &= ~8u;
							r.address = (7u << 28) | 0x040404;
							const auto plan = plan_guest_texture(r);
							ram->bytes.assign(plan.storage_bytes, std::byte{0xee}); ram->forbidden.clear(); ++ram->version;
							const auto channels = [](unsigned face, unsigned mip, unsigned z, unsigned x, unsigned y)
							{ const unsigned n = (face * 31 + mip * 17 + z * 13 + x * 7 + y * 3) & 127; return std::array{11u+n,41u+n,71u+n,101u+n}; };
							const auto unpack = [format](const auto& b, unsigned lane) -> color
							{ return format == 0x8d ? color{b[1+lane*2]/255.f,b[2]/255.f,b[0]/255.f,1} : color{b[lane*2]/255.f,b[3]/255.f,b[1]/255.f,1}; };
							std::vector<bool> active(plan.storage_bytes);
							std::vector<std::size_t> cells;
							for (const auto& s : plan.subresources)
							{
								const auto image = guest_texture_image_subresource(plan, s);
								const std::array<std::size_t,3> storage{s.row_pitch/4,s.storage_rows,s.depth};
								for (unsigned z = 0; z < s.depth; ++z) for (unsigned y = 0; y < image.rows; ++y) for (unsigned x = 0; x < image.columns; ++x)
								{
									const auto at = s.offset + 4 * address(x+(border?3:0),y+(border?dimension==vertex_texture_dimension::one?4:3:0),z,storage);
									const auto b = channels(s.face,s.level,z,x,y); cells.push_back(at);
									for (unsigned c = 0; c < 4; ++c) { ram->bytes.at(at+c)=std::byte(b[c]); active.at(at+c)=true; }
								}
							}
							const auto original = ram->bytes;
							if (border) for (std::size_t at = 0; at < active.size();)
							{ if (active[at]) { ++at; continue; } const auto start=at; while (at<active.size()&&!active[at]) ++at; ram->forbidden.push_back({static_cast<unsigned>(0x1000+start),at-start}); }
							surface_cache surfaces(gpu,ram,16u<<20,ledger,capture);
							const auto translate = [](unsigned ea,unsigned,std::size_t) { return 0x1000u+ea-0x200; };
							guest_texture_cache textures(gpu,ram,translate,1u<<20,ledger,64,{capture,std::make_shared<io_mapping_epoch>()});
							resolved_fragment_texture uploaded;
							if (profiles == 0)
							{
								const auto gate=[gpu.device() newSharedEvent]; struct release { id<MTLSharedEvent> e; ~release(){e.signaledValue=1;} } unblock{gate};
								auto job=textures.prepare_versioned_batch(std::array{r},surfaces,std::array{guest_texture_upload_dependency{gate,1}});
								bool timeout=false; try{(void)job.finish(std::chrono::milliseconds(1));}catch(const submission_wait_timeout&){timeout=true;}
								require(timeout&&!job.ready(),"Packed Morton upload lost its pending event"); gate.signaledValue=1; uploaded=job.finish().front().image;
							}
							else uploaded=textures.resolve_versioned(r,surfaces).image;
							const auto atlas=uses_stored_border_mip_atlas(plan)?std::optional{plan_stored_border_mip_atlas(plan)}:std::nullopt;
							const auto oracle = [&](const guest_texture_subresource& s,unsigned z,unsigned x,unsigned y,bool native) -> color
							{
								const unsigned block=border?(x==0?0:x==s.width+1?(s.width+1)/2+1:(x-1)/2+1):x/2;
								const unsigned lane=border?(x==0?1:x==s.width+1?0:(x-1)%2):x%2;
								const auto b=native&&!s.face&&!s.level&&!z&&!block&&!y?std::array{255u,35u,69u,103u}:channels(s.face,s.level,z,block,y);
								return unpack(b,lane);
							};
							const auto check = [&](const resolved_fragment_texture& image,bool native,bool sampling)
							{
								for (const auto& s : plan.subresources) for (unsigned z=0;z<s.depth;++z)
								{
									const unsigned w=s.width+(border?2:0),h=dimension==vertex_texture_dimension::one?1:s.height+(border?2:0);
									const auto out=make_buffer(gpu,h*256,buffer_storage::shared); texture_region area;
									area.level=atlas?0:s.level; area.slice=s.face; area.origin=MTLOriginMake(atlas?atlas->origins[s.level][0]:0,atlas?atlas->origins[s.level][1]:0,z); area.size=MTLSizeMake(w,h,1);
									gpu.begin();texture_to_buffer(gpu,image.texture,area,out,0,256);gpu.submit().wait();
									for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)
									{
										const auto wanted=oracle(s,z,x,y,native);const auto* actual=static_cast<const unsigned char*>(out.contents)+y*256+x*4;
										for(unsigned c=0;c<4;++c)require(actual[c]==std::lround(wanted[c==0||c==2?c^2:c]*255),"Packed Morton native pixel differs from its independent block/lane oracle"); ++texels;
									}
									++planes;
									if (!sampling) continue;
									const unsigned x=s.width/2+(border?1:0),y=dimension==vertex_texture_dimension::one?0:s.height/2+(border?1:0);
									const std::array<color,6> directions{color{1,0,0,1},color{-1,0,0,1},color{0,1,0,1},color{0,-1,0,1},color{0,0,1,1},color{0,0,-1,1}};
									const color coordinate=dimension==vertex_texture_dimension::cube?directions[s.face]:color{.5f,.5f,(z+.5f)/s.depth,1};
									sample(image,r,coordinate,oracle(s,z,x,y,native),47,s.level);
									if(!s.level){auto projected=coordinate;for(auto& c:projected)c*=2;sample(image,r,projected,oracle(s,z,x,y,native),24);}
									auto linear=r;linear.filter=(4u<<16)|(2u<<24);color filtered{};
									const float px=s.width*.5f+(border?.5f:-.5f),py=dimension==vertex_texture_dimension::one?0:s.height*.5f+(border?.5f:-.5f);
									const int ix=std::floor(px),iy=std::floor(py);const float fx=px-ix,fy=py-iy;
									for(unsigned yy=0;yy<2;++yy)for(unsigned xx=0;xx<2;++xx){const auto value=oracle(s,z,std::clamp(ix+int(xx),0,int(w)-1),std::clamp(iy+int(yy),0,int(h)-1),native);const auto weight=(xx?fx:1-fx)*(yy?fy:1-fy);for(unsigned c=0;c<4;++c)filtered[c]+=value[c]*weight;}
									sample(image,linear,coordinate,filtered,47,s.level);
								}
							};
							check(uploaded,false,true);
							const auto reads=ram->reads;const auto created=ledger->statistics().created;const auto encoders=gpu.encoding_statistics();
							require(textures.resolve_versioned(r,surfaces).image.texture==uploaded.texture&&ram->reads==reads&&ledger->statistics().created==created&&gpu.encoding_statistics().compute_encoders==encoders.compute_encoders,"Packed Morton ready image was recaptured or re-encoded");
							const auto anchor=cells.front();const auto donor=surfaces.acquire({static_cast<unsigned>(0x1000+anchor),{1,1,1,surface_color::argb8,false},4,{}});
							surfaces.clear(donor,0xf0,0xff234567,0,255,{0,0,1,1});ram->forbidden.push_back({static_cast<unsigned>(0x1000+anchor),4});
							const auto vm_before=textures.statistics().assembly_vm_bytes;const auto assembled=textures.resolve_versioned(r,surfaces).image;
							require(assembled.texture!=uploaded.texture&&textures.statistics().assembly_vm_bytes-vm_before==cells.size()*4-4&&ram->bytes==original,"Packed Morton mixed RTT captured native pixels/padding or changed VM");
							holes+=cells.size()*4-4;check(assembled,true,true);check(uploaded,false,false);++mixed;
							if (border&&dimension==vertex_texture_dimension::two&&shape==1)
							{
								auto memory=std::make_shared<class memory>();memory->bytes=original;
								surface_cache complete(gpu,memory,16u<<20,ledger,capture);guest_texture_cache all(gpu,memory,translate,1u<<20,ledger,64,{capture,std::make_shared<io_mapping_epoch>()});
								surface_color_transfer colors(gpu);
								for(const auto& s:plan.subresources){const auto image=guest_texture_image_subresource(plan,s);const std::array<std::size_t,3> storage{s.row_pitch/4,s.storage_rows,s.depth};const auto source=complete.acquire({static_cast<unsigned>(0x1000+s.offset),{image.columns,image.rows,1,surface_color::argb8,false},0,swizzle_region{storage,{3,3,0},{image.columns,image.rows,1}}});complete.prepare(source,surface_initialization::background);std::vector<std::byte> pixels(image.columns*image.rows*4);for(unsigned y=0;y<image.rows;++y)for(unsigned x=0;x<image.columns;++x){const auto b=channels(0,s.level,0,x,y);for(unsigned c=0;c<4;++c)pixels[(y*image.columns+x)*4+c]=std::byte(b[c]);}complete.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images){texture_region area;area.size=MTLSizeMake(image.columns,image.rows,1);colors.upload(commands,surface_color::argb8,make_upload(commands,pixels),{},images[0],area);});}
								memory->forbidden.push_back({0x1000,original.size()});const auto image=all.resolve_versioned(r,complete).image;require(all.statistics().assembly_vm_bytes==0&&memory->reads==0,"Complete packed Morton RTT captured guest bytes");check(image,false,true);++full;
							}
							ram->forbidden.clear();++profiles;
						}
		unsigned refused = 0;
		for (unsigned format : {0x8du, 0x8eu})
		{
			const auto invalid = words(format, vertex_texture_dimension::two, false, 5, 4);
			const auto reads = ram->reads, translations = translated;
			bool rejected = false; try { (void)resolver.resolve(invalid); } catch (const std::invalid_argument&) { rejected = true; }
			require(rejected && ram->reads == reads && translated == translations, "Invalid packed Morton storage reached VM"); ++refused;
		}
		std::cout<<"Paired Morton textures: "<<profiles<<" ordinary/stored-border 1D/2D/3D/cube profiles, "<<planes<<" mip/face/Z checks, "<<texels<<" exact BGRA texels, "<<cases-before<<" RSX point/projected/linear samples, "<<mixed<<" mixed and "<<full<<" complete native RTT assemblies, "<<holes<<" exact VM-hole bytes, "<<refused<<" pre-VM storage refusals, pending timeout and warm reuse passed\n";
	}
	void paired_border_cases()
	{
		const auto first = cases;
		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 16u << 20);
		gpu.set_buffer_budget(ledger);
		unsigned profiles = 0, rows_checked = 0, native_texels = 0, mixed = 0, full = 0;
		for (const unsigned format : {0x8du, 0x8eu})
			for (const bool one : {false, true})
				for (const auto [width, mips] : {std::array{1u, 1u}, {4u, 3u}, {5u, 3u}})
				{
					const auto height = one ? 1u : 3u;
					auto r = words(format, one ? vertex_texture_dimension::one : vertex_texture_dimension::two, true, width, height, 1, mips);
					r.format &= ~8u;
					r.address = (7u << 28) | 0x040404;
					const unsigned pitch = ((width + 1) / 2 + 2) * 4 + (width != 1 ? 1 : 0);
					r.control3 = (1u << 20) | pitch;
					const auto plan = plan_guest_texture(r);
					ram->bytes.assign(plan.storage_bytes, std::byte{0xee});
					const auto channels = [](unsigned mip, unsigned block, unsigned row)
					{
						const unsigned n = mip * 24 + block * 8 + row * 4;
						return std::array{12u + n, 52u + n, 92u + n, 132u + n};
					};
					const auto unpack = [format](const std::array<unsigned, 4>& b, unsigned lane) -> color
					{
						return format == 0x8d ? color{b[1 + lane * 2] / 255.f, b[2] / 255.f, b[0] / 255.f, 1} :
							color{b[lane * 2] / 255.f, b[3] / 255.f, b[1] / 255.f, 1};
					};
					std::vector<unsigned> offsets;
					unsigned offset = 0, active_bytes = 0;
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
						const auto blocks = (w + 1) / 2 + 2;
						offsets.push_back(offset);
						for (unsigned y = 0; y < h + 2; ++y)
							for (unsigned block = 0; block < blocks; ++block)
							{
								const auto bytes = channels(mip, block, y);
								for (unsigned b = 0; b < bytes.size(); ++b)
									ram->bytes.at(offset + y * pitch + block * 4 + b) = std::byte(bytes[b]);
							}
						active_bytes += blocks * 4 * (one ? 1 : h + 2);
						offset += (h + 2) * pitch;
					}
					require(offset == ram->bytes.size(), "Paired guest ring has wrong block strides");
					++ram->version;
					const auto original = ram->bytes;
					surface_cache surfaces(gpu, ram, 16u << 20, ledger, capture);
					guest_texture_cache textures(gpu, ram, [](unsigned ea, unsigned, std::size_t) { return 0x1000 + ea - 0x200; },
						1u << 20, ledger, 16, {capture, std::make_shared<io_mapping_epoch>()});
					const auto uploaded = textures.resolve_versioned(r, surfaces).image;
					const auto atlas = one || mips > 1 ? plan_stored_border_atlas(width, height, mips, one) : stored_border_mip_atlas{};
					for (unsigned mip = 0; mip < mips; ++mip)
					{
						const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
						const auto cols = w + 2, rows = one ? 1u : h + 2;
						const auto oracle = [&](unsigned x, unsigned y)
						{
							const auto block = x == 0 ? 0 : x == w + 1 ? (w + 1) / 2 + 1 : (x - 1) / 2 + 1;
							return unpack(channels(mip, block, y + unsigned(one)), x == 0 ? 1 : x == w + 1 ? 0 : (x - 1) % 2);
						};
						const auto out = make_buffer(gpu, rows * 256, buffer_storage::shared);
						texture_region area;
						if (one || mips > 1)
							area.origin = MTLOriginMake(atlas.origins[mip][0], atlas.origins[mip][1], 0);
						area.size = MTLSizeMake(cols, rows, 1);
						gpu.begin();
						texture_to_buffer(gpu, uploaded.texture, area, out, 0, 256);
						gpu.submit().wait();
						const auto bytes = map_buffer(out, 0, rows * 256);
						for (unsigned y = 0; y < rows; ++y)
							for (unsigned x = 0; x < cols; ++x)
							{
								const auto value = oracle(x, y);
								for (unsigned c = 0; c < 4; ++c)
								{
									const unsigned rgba = c == 0 || c == 2 ? c ^ 2 : c;
									require(bytes[y * 256 + x * 4 + c] == std::byte(std::lround(value[rgba] * 255)),
										"Paired border decoded the wrong lane or shared channel");
								}
								++native_texels;
							}
						sample(uploaded, r, {-8, -8, 0, 1}, oracle(0, 0), 47, float(mip));
						sample(uploaded, r, {8, 8, 0, 1}, oracle(cols - 1, rows - 1), 47, float(mip));
						auto filter = r;
						filter.filter = (2u << 24) | (4u << 16);
						const auto left = oracle(0, 0), interior = oracle(1, 0);
						color average{};
						for (unsigned c = 0; c < average.size(); ++c)
							average[c] = (left[c] + interior[c]) * .5f;
						sample(uploaded, filter, {0, -8, 0, 1}, average, 47, float(mip));
						rows_checked += rows;
					}
					const auto mip = mips - 1;
					const auto address = 0x1000 + offsets.back() + (one ? pitch : 0);
					const auto native = surfaces.acquire({address, {1, 1, 1, surface_color::argb8, false}, 4, {}});
					surfaces.clear(native, 0xf0, 0xaabbccdd, 0, 255, {0, 0, 1, 1});
					ram->forbidden.push_back({address, 4});
					const auto composed = textures.resolve_versioned(r, surfaces);
					require(composed.image.texture != uploaded.texture && textures.statistics().assembly_vm_bytes == active_bytes - 4,
						"Paired border RTT did not use exactly its VM holes");
					sample(composed.image, r, {-8, -8, 0, 1}, unpack({0xaa, 0xbb, 0xcc, 0xdd}, 1), 47, float(mip));
					sample(uploaded, r, {-8, -8, 0, 1}, unpack(channels(mip, 0, unsigned(one)), 1), 47, float(mip));
					require(ram->bytes == original, "Paired border path changed guest bytes");
					ram->forbidden.clear();
					++mixed;
					++profiles;

					if (width == 1 && !one)
					{
						surface_cache full_surfaces(gpu, ram, 16u << 20, ledger, capture);
						guest_texture_cache full_textures(gpu, ram, [](unsigned ea, unsigned, std::size_t) { return 0x1000 + ea - 0x200; },
							1u << 20, ledger, 16, {capture, std::make_shared<io_mapping_epoch>()});
						const auto owner = full_surfaces.acquire({0x1000, {3, height + 2, 1, surface_color::argb8, false}, pitch, {}});
						full_surfaces.clear(owner, 0xf0, 0xff234567, 0, 255, {0, 0, 3, height + 2});
						ram->forbidden.push_back({0x1000, ram->bytes.size()});
						const auto cast = full_textures.resolve_versioned(r, full_surfaces);
						require(full_textures.statistics().assembly_vm_bytes == 0, "Complete paired RTT cast captured VM");
						sample(cast.image, r, {-8, -8, 0, 1}, unpack({0xff, 0x23, 0x45, 0x67}, 1));
						sample(cast.image, r, {8, 8, 0, 1}, unpack({0xff, 0x23, 0x45, 0x67}, 0));
						ram->forbidden.clear();
						++full;
					}
				}
		std::cout << "Linear paired borders: " << profiles << " profiles, " << rows_checked << " mip rows, "
			<< native_texels << " exact native texels, " << cases - first << " RSX samples, "
			<< mixed << " mixed and " << full << " complete native RTT assemblies passed\n";
	}
	void sampled_border_atlas_cases()
	{
		const auto first = cases;
		constexpr unsigned width = 8, height = 4, mips = 4, pitch = (width + 2) * 4 + 1;
		auto r = words(0x85, vertex_texture_dimension::two, true, width, height, 1, mips);
		r.format &= ~8u;
		r.control3 = (1u << 20) | pitch;
		r.address = (7u << 28) | 0x040404;
		const auto plan = plan_guest_texture(r);
		ram->bytes.assign(plan.storage_bytes, std::byte{0xcd});
		const auto red = [](unsigned mip, unsigned x, unsigned y) { return 8 + mip * 40 + x * 8 + y * 4; };
		const auto shade = [](float value) -> color { return {value / 255.f, 0, 0, 1}; };
		for (unsigned mip = 0; mip < mips; ++mip)
		{
			const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
			for (unsigned y = 0; y < h + 2; ++y)
				for (unsigned x = 0; x < w + 2; ++x)
				{
					const auto at = plan.subresources[mip].offset + y * pitch + x * 4;
					ram->bytes[at] = std::byte{255};
					ram->bytes[at + 1] = std::byte(red(mip, x, y));
					ram->bytes[at + 2] = ram->bytes[at + 3] = std::byte{};
				}
		}
		++ram->version;
		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 16u << 20);
		gpu.set_buffer_budget(ledger);
		surface_cache surfaces(gpu, ram, 16u << 20, ledger, capture);
		guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000 + offset - 0x200; },
			1u << 20, ledger, 64, {capture, std::make_shared<io_mapping_epoch>()});
		const auto pixels = [&]
		{
			auto value = r;
			value.format |= 0x40u << 8;
			return value;
		}();
		const auto uploaded = textures.resolve_versioned_batch(std::array{r, pixels}, surfaces);
		const auto image = uploaded[0].image;
		require(image.texture == uploaded[1].image.texture && image.texture.mipmapLevelCount == 1 &&
			image.metadata.actual_mips == mips, "Versioned draw cache did not coalesce the atlas upload");
		for (unsigned mip = 0; mip < mips; ++mip)
		{
			const auto w = std::max(1u, width >> mip), h = std::max(1u, height >> mip);
			sample(image, r, {-2, -2, 0, 1}, shade(red(mip, 0, 0)), 47, float(mip));
			sample(image, r, {2, 2, 0, 1}, shade(red(mip, w + 1, h + 1)), 47, float(mip));
			sample(image, r, {.375f, .625f, 0, 1}, shade(red(mip, 1 + unsigned(.375f * w), 1 + unsigned(.625f * h))), 47, float(mip));
			sample(image, pixels, {3, 2.5f, 0, 1}, shade(red(mip, 1 + unsigned(.375f * w), 1 + unsigned(.625f * h))), 47, float(mip));
		}
		for (unsigned wrap = 1; wrap <= 8; ++wrap)
		{
			auto state = r;
			state.address = (7u << 28) | 0x040400 | wrap;
			const unsigned point[]{4, 1, 1, 0, 1, 1, 1, 1};
			sample(image, state, {-.1875f, -.25f, 0, 1}, shade(red(1, point[wrap - 1], 0)), 47, 1);
		}
		auto mip_blend = r;
		mip_blend.filter = (1u << 24) | (5u << 16);
		sample(image, mip_blend, {-2, -2, 0, 1}, shade(.75f * red(1, 0, 0) + .25f * red(2, 0, 0)), 47, 1.25f);
		auto level_blend = r;
		level_blend.filter = (1u << 24) | (4u << 16);
		sample(image, level_blend, {.25f, -.25f, 0, 1}, shade(.5f * (red(1, 1, 0) + red(1, 2, 0))), 47, 1);
		level_blend.address = (7u << 28) | 0x040401;
		sample(image, level_blend, {0, -.25f, 0, 1}, shade(.5f * (red(1, 1, 0) + red(1, 4, 0))), 47, 1);
		sample(image, r, {-.25f, -.25f, 0, 1}, shade(red(1, 0, 0)), 25, .25f);
		sample(image, r, {-.25f, -.25f, 0, 1}, shade(red(0, 0, 0)), 49, 2.f);
		sample(image, r, {-1, -1, 0, 2}, shade(red(0, 0, 0)), 24);
		auto clamped = r;
		clamped.control0 = (1u << 31) | (256u << 19) | (512u << 7);
		sample(image, clamped, {-2, -2, 0, 1}, shade(red(1, 0, 0)), 23);
		sample(image, clamped, {-2, -2, 0, 1}, shade(red(2, 0, 0)), 47, 9);
		clamped.filter |= 128;
		sample(image, clamped, {-2, -2, 0, 1}, shade(red(2, 0, 0)), 47, 1.2f);

		// Only a border texel in mip one is native. Remaining active bytes
		// come from VM, and the old cached atlas must stay immutable.
		const auto at = static_cast<unsigned>(0x1000 + plan.subresources[1].offset);
		const auto native = surfaces.acquire({at, {1, 1, 1, surface_color::argb8, false}, 4, {}});
		surfaces.clear(native, 0xf0, 0xffcc0000, 0, 255, {0, 0, 1, 1});
		ram->forbidden.push_back({at, 4});
		const auto before = ram->bytes;
		const auto assembled = textures.resolve_versioned(r, surfaces);
		require(assembled.image.texture != image.texture && assembled.image.texture.mipmapLevelCount == 1 &&
			textures.statistics().assemblies_created == 1, "Native owner did not rebuild the stored-border atlas");
		sample(assembled.image, r, {-2, -2, 0, 1}, shade(204), 47, 1);
		sample(assembled.image, r, {-2, -2, 0, 1}, shade(red(2, 0, 0)), 47, 2);
		sample(image, r, {-2, -2, 0, 1}, shade(red(1, 0, 0)), 47, 1);
		const auto reads = ram->reads;
		require(textures.resolve_versioned(r, surfaces).image.texture == assembled.image.texture &&
			ram->reads == reads && ram->bytes == before, "Warm RTT atlas recaptured or changed guest bytes");
		ram->forbidden.clear();

		// Float texels give an exact independent oracle for simultaneous
		// within-level and between-level filtering without UNORM rounding.
		auto floats = words(0x9b, vertex_texture_dimension::two, true, width, height, 1, mips);
		floats.format &= ~8u;
		floats.address = r.address;
		floats.filter = (2u << 24) | (6u << 16);
		const auto float_plan = plan_guest_texture(floats);
		ram->bytes.assign(float_plan.storage_bytes, std::byte{0xcd});
		for (const auto& level : float_plan.subresources)
			for (unsigned y = 0; y < level.height + 2; ++y)
				for (unsigned x = 0; x < level.width + 2; ++x)
				{
					const color value{float(level.level) * .25f + float(x) / 64, 0, 0, 1};
					auto at = level.offset + y * level.row_pitch + x * 16;
					for (const auto channel : value)
					{
						const auto bits = std::bit_cast<unsigned>(channel);
						for (unsigned byte = 0; byte < 4; ++byte)
							ram->bytes[at++] = std::byte(bits >> ((3 - byte) * 8));
					}
				}
		++ram->version;
		const auto float_image = resolver.resolve(floats);
		sample(float_image, floats, {0, -.25f, 0, 1}, {.375f + 1.f / 128, 0, 0, 1}, 47, 1.5f);
		std::cout << "Stored-border atlas shader: " << cases - first
			<< " exact RSX samples, versioned cache reuse and one mixed native/VM mip assembly passed\n";
	}
	static std::size_t count_active_runs(std::span<const bool> active)
	{
		std::size_t runs = 0;
		bool previous = false;
		for (bool present : active)
		{
			if (present && !previous) ++runs;
			previous = present;
		}
		return runs;
	}
	void stored_border_clamp_cases()
	{
		const auto first = cases;
		unsigned periodic_samples = 0;
		unsigned mixed_samples = 0;
		constexpr unsigned width = 4, height = 3, pitch = 27;
		const auto texel = [](float x, float y) -> color
		{
			return {(24 + 16 * x + 8 * y) / 255, (16 + 8 * x + 24 * y) / 255, (32 + 12 * x + 12 * y) / 255, 1};
		};
		for (bool morton : {false, true})
		{
			auto r = words(0x85, vertex_texture_dimension::two, !morton, width, height);
			r.format &= ~8u;
			r.control3 = (1u << 20) | pitch;
			ram->bytes.assign(morton ? 16 * 16 * 4 : pitch * (height + 2), std::byte{0xcd});
			for (unsigned y = 0; y < height + 2; ++y)
				for (unsigned x = 0; x < width + 2; ++x)
				{
					const auto at = morton ? morton_border_cell(x + 3, y + 3, 4, 4) * 4 : y * pitch + x * 4;
					const auto value = texel(x, y);
					ram->bytes[at] = std::byte{255};
					for (unsigned c = 0; c < 3; ++c)
						ram->bytes[at + c + 1] = std::byte(std::lround(value[c] * 255));
				}
			++ram->version;
			const auto before = ram->bytes;
			for (unsigned mode : {5u, 6u, 7u, 8u})
				for (unsigned axis : {0u, 1u})
				{
					r.address = (7u << 28) | (0x040404u & ~(15u << (8 * axis))) | (mode << (8 * axis));
					r.filter = 0x01010000;
					const auto image = resolver.resolve(r);
					const auto size = axis ? height : width;
					const auto coordinate = [&](float p, bool pixels, float w)
					{
						color value{.375f, .5f, 0, w};
						if (pixels) { value[0] *= width; value[1] *= height; }
						value[axis] = pixels ? p : p / size;
						value[0] *= w;
						value[1] *= w;
						return value;
					};
					const auto expected = [&](float p)
					{
						return axis ? texel(2, p) : texel(p, 2);
					};
					sample(image, r, coordinate(-1.75f, false, 1), expected(mode == 5 ? 1 : 2));
					sample(image, r, coordinate(-8.f, false, -2), expected(mode == 5 ? 1 : mode == 7 ? size + 1 : size), 24);
					sample(image, r, coordinate(8.f, false, 1), expected(mode == 7 ? size + 1 : size), 47);
					auto linear = r;
					linear.filter = 0x02020000;
					sample(image, linear, coordinate(-.125f, false, 1), expected(mode == 5 ? .5f : 1.f));
					sample(image, linear, coordinate(-8.f, false, 1), expected(mode == 5 ? .5f : mode == 6 ? float(size) : mode == 7 ? size + 1.f : size + .5f));
					sample(image, linear, coordinate(8.f, false, 1), expected(mode == 6 ? float(size) : mode == 7 ? size + 1.f : size + .5f), 49);
					auto pixels = linear;
					pixels.format |= 0x40u << 8;
					const auto pixel_image = resolver.resolve(pixels);
					sample(pixel_image, pixels, coordinate(-1.25f, true, -2), expected(mode == 5 ? .5f : 1.75f), 24);
				}
			const auto index = [](float coordinate, int extent, unsigned wrap, bool linear)
			{
				const auto texel_index = [&](int pixel)
				{
					if (wrap == 1) return (pixel % extent + extent) % extent + 1;
					if (wrap == 2)
					{
						const auto period = 2 * extent;
						const auto cell = (pixel % period + period) % period;
						return cell < extent ? cell + 1 : period - cell;
					}
					return std::clamp(pixel + 1, 0, extent + 1);
				};
				const auto cell = coordinate - (linear ? .5f : 0.f);
				const auto low = int(std::floor(cell));
				if (!linear) return float(texel_index(low));
				return std::lerp(float(texel_index(low)), float(texel_index(low + 1)), cell - low);
			};
			for (const auto& [s, t] : {std::pair{1u, 1u}, {1u, 4u}, {4u, 1u}, {2u, 2u}, {1u, 2u}, {2u, 1u}})
				for (bool pixels : {false, true})
					for (bool linear : {false, true})
					{
						r.address = (7u << 28) | 0x040000u | (t << 8) | s;
						r.filter = linear ? 0x02020000 : 0x01010000;
						r.format = (r.format & ~(0x40u << 8)) | (pixels ? 0x40u << 8 : 0);
						const auto image = resolver.resolve(r);
						const float offset = pixels && !linear ? .01f : 0.f;
						unsigned lookup = 0;
						for (const auto& point : {std::array{0.f, 0.f}, {-.25f, .25f}, {4.25f, 2.75f},
							{-9.75f, 8.25f}, {6.25f, -4.25f}})
						{
							const unsigned opcode = std::array{23u, 24u, 25u, 47u, 49u}[lookup++];
							const float q = opcode == 24 ? -2.f : 1.f;
							const color coordinate{q * point[0] / (pixels ? 1 : width), q * point[1] / (pixels ? 1 : height), 0, q};
							sample(image, r, coordinate, texel(index(point[0] + offset, width, s, linear),
								index(point[1] + offset, height, t, linear)), opcode, 1.75f);
							++periodic_samples;
						}
					}
			const auto mixed_begin = cases;
			for (bool pixels : {false, true})
				for (bool min_linear : {false, true})
				{
					r.address = 0x70040401; // Repeat S, stored border T.
					r.filter = min_linear ? 0x01020000 : 0x02010000;
					r.format = (r.format & ~(0x40u << 8)) | (pixels ? 0x40u << 8 : 0);
					const auto image = resolver.resolve(r);
					const auto expected = [&](bool minifies) { return texel(minifies == min_linear ? 2.5f : 1.f, 2.f); };
					const color seam{0, pixels ? 1.5f : .5f, 0, 1};
					const color projected{0, -2 * seam[1], 0, -2};
					sample(image, r, seam, expected(false));
					sample(image, r, projected, expected(false), 24);
					sample(image, r, seam, expected(false), 25, .125f);
					sample(image, r, seam, expected(true), 25, 2.f);
					sample(image, r, seam, expected(false), 47, -1.f);
					sample(image, r, seam, expected(true), 47, 1.f);

					// Even though every sample clamps to the same guest corner,
					// the original fragment-coordinate footprint crosses one texel.
					r.address = 0x70040505;
					fragment_context fragment;
					fragment.wpos_bias_x = fragment.wpos_bias_y = -16;
					const auto corner = [&](bool minifies) { return minifies == min_linear ? texel(.5f, .5f) : texel(1.f, 1.f); };
					fragment.wpos_scale = 2.f;
					sample(image, r, {}, corner(true), 23, 0, &fragment);
					fragment.wpos_scale = .125f;
					sample(image, r, {}, corner(false), 23, 0, &fragment);
					sample(image, r, {}, corner(true), 49, 4.f, &fragment);
				}
			mixed_samples += cases - mixed_begin;
			require(ram->bytes == before, "Stored border clamp sampling modified guest storage");
		}
		std::cout << "Stored border clamp modes: " << cases - first - periodic_samples - mixed_samples
			<< " linear/Morton color samples across both axes, point/linear and projected/pixel coordinates\n";
		std::cout << "Stored border periodic modes: " << periodic_samples
			<< " linear/Morton color samples, mixed axes and simultaneous seams across five TEX opcodes\n";
		std::cout << "Stored border mixed filters: " << mixed_samples
			<< " linear/Morton color samples across implicit, projected, gradient, explicit-level and biased lookup\n";
	}
	void stored_morton_border_cases()
	{
		const auto first = cases;
		constexpr unsigned width = 15, height = 3, columns = width + 2, rows = height + 2, bytes = 32 * 16 * 4;
		auto r = words(0x85, vertex_texture_dimension::two, false, width, height);
		r.format &= ~8u;
		r.address = (7u << 28) | 0x040404;
		const auto offset = [](unsigned x, unsigned y) { return morton_border_cell(x + 3, y + 3, 5, 4) * 4; };
		const auto texel = [](float x, float y) -> color
		{
			return {(16 + 8 * x + 4 * y) / 255, (24 + 4 * x + 20 * y) / 255, (40 + 8 * x + 8 * y) / 255, 1};
		};
		ram->bytes.assign(bytes, std::byte{0xcd});
		std::array<bool, bytes> active{};
		for (unsigned y = 0; y < rows; ++y)
			for (unsigned x = 0; x < columns; ++x)
			{
				const auto value = texel(x, y);
				const auto at = offset(x, y);
				ram->bytes[at] = std::byte{255};
				for (unsigned channel = 0; channel < 3; ++channel)
					ram->bytes[at + 1 + channel] = std::byte(std::lround(value[channel] * 255));
				std::fill_n(active.begin() + at, 4, true);
			}
		for (unsigned i = 0; i < bytes;)
		{
			if (active[i]) { ++i; continue; }
			const unsigned start = i;
			while (i < bytes && !active[i]) ++i;
			ram->forbidden.push_back({0x1000 + start, i - start});
		}
		++ram->version;
		const auto original = ram->bytes;
		const auto image = resolver.resolve(r);
		require(image.texture.width == columns && image.texture.height == rows && image.metadata.stored_border_extent == std::array{width, height},
			"Morton texture did not produce its inner ring");
		for (unsigned y = 0; y < rows; ++y)
			for (unsigned x = 0; x < columns; ++x)
				sample(image, r, {(float(x) - .5f) / width, (float(y) - .5f) / height, 0, 1}, texel(x, y));
		auto linear = r;
		linear.filter = 0x02020000;
		sample(image, linear, {0, 0, 0, 1}, texel(.5f, .5f));
		auto pixels = r;
		pixels.format |= 0x40u << 8;
		const auto detached = realize_without_vm(capture_without_gpu(pixels, count_active_runs(active)));
		sample(detached, pixels, {-2, 4, 0, 2}, texel(0, 3), 24);
		ram->bytes = original;
		++ram->version;

		const cpu_capture_dispatch capture = [](const auto& work) { work(); };
		context gpu{2};
		auto ledger = std::make_shared<heap_budget>(gpu, 8u << 20);
		gpu.set_buffer_budget(ledger);
		surface_cache surfaces(gpu, ram, 8u << 20, ledger, capture);
		guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000 + offset - 0x200; },
			1u << 20, ledger, 64, {capture, std::make_shared<io_mapping_epoch>()});
		const std::array requests{r, pixels};
		const auto batch = textures.resolve_versioned_batch(requests, surfaces);
		require(batch[0].image.texture == batch[1].image.texture, "Morton ring requests did not coalesce");
		const auto reads = ram->reads;
		require(textures.resolve_versioned(r, surfaces).image.texture == batch[0].image.texture && ram->reads == reads,
			"Morton ring lost its warm VM cache hit");
		const auto corner = 0x1000 + offset(width + 1, height + 1);
		const auto native = surfaces.acquire({corner, {1, 1, 1, surface_color::argb8, false}, 4, {}});
		surfaces.clear(native, 0xf0, 0xff104080, 0, 255, {0, 0, 1, 1});
		ram->forbidden.push_back({corner, 4});
		const auto composed = textures.resolve_versioned_batch(std::span{requests}.first(1), surfaces);
		require(composed[0].image.texture != batch[0].image.texture &&
			textures.statistics().assembly_vm_bytes == (columns * rows - 1) * 4,
			"Morton border RTT did not restrict guest reads to active holes");
		sample(composed[0].image, r, {8, 8, 0, 1}, {16.f / 255, 64.f / 255, 128.f / 255, 1});
		sample(composed[0].image, r, {-8, -8, 0, 1}, texel(0, 0));
		sample(batch[0].image, r, {8, 8, 0, 1}, texel(width + 1, height + 1));
		require(ram->bytes == original, "Morton border upload or assembly changed guest bytes");
		ram->forbidden.clear();
		std::cout << "Morton stored color ring: " << cases - first << " native samples, coalesced VM upload and border-only RTT with exact active holes\n";
	}
	void stored_depth_border_cases()
	{
		const auto first = cases;
		constexpr unsigned width = 4, height = 3, columns = width + 2, rows = height + 2;
		std::size_t depth_words = 0, stencil_bytes = 0;
		for (unsigned format : {0x90u, 0x91u, 0x92u, 0x93u})
			for (unsigned layout = 0; layout < 3; ++layout)
			{
				const bool morton = layout == 2;
				auto r = words(format, vertex_texture_dimension::two, !morton, width, height);
				r.format &= ~8u;
				r.address = (7u << 28) | 0x040404;
				r.filter = 0x01010000;
				r.border_color = 0xffffffff;
				const unsigned bpp = describe_guest_texture_format(format).block.bytes;
				const unsigned pitch = columns * bpp + (layout == 1 ? 1 : bpp);
				r.control3 = (1u << 20) | pitch;
				const auto raw = [&](unsigned x, unsigned y)
				{
					const bool high = (x + y) % 2;
					const unsigned n = y * columns + x;
					if (format == 0x92)
						// Keep the bilinear oracle on a representable UNORM16 value.
						return (high ? 0x9000u : 0x2000u) + n * 52;
					if (format == 0x93)
						return (high ? 0x6800u : 0x5000u) + n * 5;
					const unsigned depth = format == 0x90 ? (high ? 0xa00000u : 0x400000u) + n * 139
						: std::bit_cast<unsigned>((high ? .75f : .25f) + n / 1024.f) >> 7;
					return (depth << 8) | (17 + n * 7);
				};
				const auto depth_bits = [&](unsigned value)
				{
					if (format == 0x92)
						return value;
					if (format == 0x90)
						return std::bit_cast<unsigned>(float(value >> 8) / 16777215.f);
					return format == 0x91 ? (value >> 8) << 7 : (value << 11) + (120u << 23);
				};
				const auto depth_at = [&](unsigned x, unsigned y)
				{
					const auto bits = depth_bits(raw(x, y));
					return format == 0x92 ? bits / 65535.f : std::bit_cast<float>(bits);
				};
				ram->bytes.assign(morton ? 16 * 16 * bpp : pitch * rows, std::byte{0xe3});
				std::array<bool, 16 * 16> active{};
				for (unsigned y = 0; y < rows; ++y)
					for (unsigned x = 0; x < columns; ++x)
					{
						auto at = y * pitch + x * bpp;
						if (morton)
						{
							const auto cell = morton_border_cell(x + 3, y + 3, 4, 4);
							active[cell] = true;
							at = cell * bpp;
						}
						for (unsigned byte = 0; byte < bpp; ++byte)
							ram->bytes[at + byte] = std::byte(raw(x, y) >> ((bpp - 1 - byte) * 8));
					}
				++ram->version;
				const auto original = ram->bytes;
				const auto image = resolver.resolve(r);
				require(image.metadata.stored_border_extent == std::array{width, height} && image.metadata.depth &&
					image.texture.width == columns && image.texture.height == rows, "Native depth ring lost its format or geometry");

				const bool packed_stencil = bpp == 4;
				const unsigned native_depth_bytes = format == 0x92 ? 2 : 4;
				const auto output = make_buffer(ctx, 512 * rows, buffer_storage::shared);
				texture_region region;
				region.size = MTLSizeMake(columns, rows, 1);
				ctx.begin();
				texture_to_buffer(ctx, image.texture, region, output, 0, 256, 0, texture_plane::depth);
				if (packed_stencil)
					texture_to_buffer(ctx, image.texture, region, output, 256 * rows, 256, 0, texture_plane::stencil);
				ctx.submit().wait();
				const auto data = map_buffer(output, 0, output.length);
				for (unsigned y = 0; y < rows; ++y)
					for (unsigned x = 0; x < columns; ++x)
					{
						unsigned actual = 0;
						std::memcpy(&actual, data.data() + y * 256 + x * native_depth_bytes, native_depth_bytes);
						require(actual == depth_bits(raw(x, y)), "Stored depth ring differs from the guest depth word");
						++depth_words;
						if (packed_stencil)
						{
							require(data[256 * rows + y * 256 + x] == std::byte(raw(x, y)), "Stored depth ring lost its stencil byte");
							++stencil_bytes;
						}
					}

				const auto rgba = [](float depth) -> color { return {depth, depth, depth, depth}; };
				sample(image, r, {-8, -8, 0, 1}, rgba(depth_at(0, 0)));
				sample(image, r, {8, 8, 0, 1}, rgba(depth_at(width + 1, height + 1)));
				sample(image, r, {.375f, .5f, 0, 1}, rgba(depth_at(2, 2)), 47);
				auto mixed = r;
				mixed.address = (mixed.address & ~15u) | 3u;
				sample(image, mixed, {-8, -8, 0, 1}, rgba(depth_at(1, 0)));
				auto pixels = r;
				pixels.format |= 0x40u << 8;
				sample(image, pixels, {-2, 3, 0, 2}, rgba(depth_at(0, 2)), 24);
				auto linear = r;
				linear.filter = 0x02020000;
				sample(image, linear, {0, 0, 0, 1},
					rgba((depth_at(0, 0) + depth_at(1, 0) + depth_at(0, 1) + depth_at(1, 1)) / 4));

				auto shadow = r;
				shadow.address = (shadow.address & 0x0fffffffu) | (1u << 28);
				sample(image, shadow, {-8, -8, .5f, 1}, rgba(1));
				sample(image, shadow, {8, 8, .5f, 1}, rgba(0));
				sample(image, shadow, {-4, 1, 1, 2}, rgba(1), 24);
				shadow.filter = linear.filter;
				sample(image, shadow, {0, 0, .5f, 1}, rgba(.5f));
				shadow.address = (shadow.address & ~15u) | 3u;
				sample(image, shadow, {-8, -8, .5f, 1}, rgba(0));
				sample(image, shadow, {0, 0, .5f, 1}, rgba(.5f));
				for (const auto& [wrap, result] : {std::pair{6u, 1.f}, {7u, 0.f}, {8u, .5f}})
				{
					shadow.address = (shadow.address & ~15u) | wrap;
					sample(image, shadow, {-8, -8, .5f, 1}, rgba(result));
				}
				shadow.address = (shadow.address & ~0x0f0fu) | 0x0101u;
				sample(image, shadow, {0, 0, .5f, 1}, rgba(.5f));
				linear.address = (linear.address & ~0x0f0fu) | 0x0101u;
				sample(image, linear, {0, .5f, 0, 1}, rgba((depth_at(1, 2) + depth_at(4, 2)) / 2));
				for (bool min_linear : {false, true})
				{
					linear.filter = shadow.filter = min_linear ? 0x01020000 : 0x02010000;
					sample(image, linear, {0, .5f, 0, 1}, rgba(min_linear ? depth_at(1, 2) : (depth_at(1, 2) + depth_at(4, 2)) / 2));
					sample(image, shadow, {0, 0, .5f, 1}, rgba(min_linear ? 1.f : .5f));
					fragment_context minified;
					minified.wpos_scale = 2.f;
					minified.wpos_bias_x = minified.wpos_bias_y = -1.f;
					sample(image, shadow, {0, 0, .5f, 1}, rgba(min_linear ? .5f : 1.f), 23, 0, &minified);
				}
				require(ram->bytes == original, "Stored depth upload changed guest bytes or row padding");
				const auto captured = capture_without_gpu(r, morton ? count_active_runs(active) : 1);
				sample(realize_without_vm(captured), r, {-8, -8, 0, 1}, rgba(depth_at(0, 0)));
			}
		std::cout << "Stored depth rings: eight linear and four Morton profiles, " << depth_words << " exact depth words, "
			<< stencil_bytes << " exact stencil bytes, " << cases - first
			<< " depth/shadow/PCF samples and unchanged VM passed\n";
	}
	void stored_border_assembly_cases(bool morton = false)
	{
		const auto first = cases;
		constexpr unsigned extent = 4, side = extent + 2;
		unsigned profiles = 0;
		std::size_t depth_words = 0, stencil_bytes = 0, hole_bytes = 0;
		for (unsigned format : {0x85u, 0x90u, 0x91u, 0x92u, 0x93u})
			for (bool partial : {false, true})
				for (unsigned samples : {1u, 2u, 4u})
				{
					@autoreleasepool
					{
						auto r = words(format, vertex_texture_dimension::two, !morton, extent, extent);
						r.format &= ~8u;
						r.address = (7u << 28) | 0x040404;
						const auto codec = describe_guest_texture_format(format);
						const bool depth = codec.depth.has_value(), stencil = format == 0x90 || format == 0x91;
						const unsigned bpp = codec.block.bytes, pitch = side * bpp + 3;
						r.control3 = (1u << 20) | pitch;
						const unsigned vm_word = format == 0x85 ? 0xff204060u : format == 0x90 ? 0x8000002au :
							format == 0x91 ? 0x7e00002au : format == 0x92 ? 0x8000u : 0x6000u;
						ram->bytes.assign(morton ? 16 * 16 * bpp : pitch * side, std::byte{0xb7});
						for (unsigned y = 0; y < side; ++y)
							for (unsigned x = 0; x < side; ++x)
							{
								const auto offset = morton ? morton_border_cell(x + 3, y + 3, 4, 4) * bpp : y * pitch + x * bpp;
								for (unsigned i = 0; i < bpp; ++i)
									ram->bytes[offset + i] = std::byte(vm_word >> (8 * (bpp - 1 - i)));
							}
						++ram->version;
						const auto original = ram->bytes;

						const cpu_capture_dispatch capture = [](const auto& work) { work(); };
						context gpu{1};
						auto ledger = std::make_shared<heap_budget>(gpu, 8u << 20);
						gpu.set_buffer_budget(ledger);
						surface_cache surfaces(gpu, ram, 8u << 20, ledger, capture);
						guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000 + offset - 0x200; },
							1u << 20, ledger, 64, {capture, std::make_shared<io_mapping_epoch>()});
						const surface_format sf = depth ? surface_format{*codec.depth} : surface_format{surface_color::argb8};
						const auto factors = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
						const auto native_depth = [&](unsigned part) -> float
						{
							return format == 0x90 || format == 0x92 ? float(part) : part ? .734567f : .267891f;
						};
						const unsigned region_size = partial ? 2 : side;
						for (unsigned part = 0; part < (partial ? 2u : 1u); ++part)
						{
							const unsigned origin = part ? side - region_size : 0;
							const unsigned address = morton ? 0x1000 : 0x1000 + origin * (pitch + bpp);
							const auto swizzle = morton ? std::optional<swizzle_region>{{{16, 16, 1},
								{origin + 3, origin + 3, 0}, {region_size, region_size, 1}}} : std::nullopt;
							const auto row_pitch = morton ? 0 : pitch;
							const auto source = surfaces.acquire({address,
								{region_size / factors.x, region_size / factors.y, samples, sf, false}, row_pitch, swizzle});
							surfaces.prepare(source, surface_initialization::background);
							if (depth)
							{
								const std::array uses{cached_surface_use{source, true}};
								surfaces.draw(uses, [&](context& commands, auto images)
								{
									auto pass = [MTL4RenderPassDescriptor new];
									pass.depthAttachment.texture = images[0];
									pass.depthAttachment.loadAction = MTLLoadActionClear;
									pass.depthAttachment.storeAction = MTLStoreActionStore;
									pass.depthAttachment.clearDepth = native_depth(part);
									if (stencil)
									{
										pass.stencilAttachment.texture = images[0];
										pass.stencilAttachment.loadAction = MTLLoadActionClear;
										pass.stencilAttachment.storeAction = MTLStoreActionStore;
										pass.stencilAttachment.clearStencil = 63 + part;
									}
									(void)commands.render(pass);
								});
							}
							else
								surfaces.clear(source, 0xf0, part ? 0xff8090a0 : 0xff103050, 0, 255,
									{0, 0, region_size / factors.x, region_size / factors.y});
							guest_surface_byte_cursor cursor(plan_guest_surface(address, region_size, region_size, bpp, row_pitch, swizzle));
							while (const auto active = cursor.next())
								ram->forbidden.push_back(*active);
						}
						auto pixels = r;
						pixels.format |= 0x40u << 8;
						const auto before = ram->reads;
						const auto batch = textures.resolve_versioned_batch(std::array{r, pixels}, surfaces);
						const auto& image = batch[0].image;
						require(image.metadata.stored_border_extent == std::array{extent, extent} && image.texture.width == side && image.texture.height == side &&
							image.texture == batch[1].image.texture && textures.statistics().assemblies_created == 1,
							"Stored RTT ring did not preserve padded geometry or batch ownership");
						const std::size_t holes = partial ? (side * side - 8) * bpp : 0;
						require(textures.statistics().assembly_vm_bytes == holes && (holes || ram->reads == before),
							"Stored RTT ring captured more than the active VM holes");
						hole_bytes += holes;
						const auto owner = [&](unsigned x, unsigned y)
						{
							if (!partial || (x < 2 && y < 2))
								return 0u;
							return x >= side - 2 && y >= side - 2 ? 1u : 2u;
						};
						if (depth)
						{
							texture_region area;
							area.size = MTLSizeMake(side, side, 1);
							const auto data = make_buffer(gpu, 512 * side, buffer_storage::shared);
							gpu.begin();
							texture_to_buffer(gpu, image.texture, area, data, 0, 256, 0, texture_plane::depth);
							if (stencil)
								texture_to_buffer(gpu, image.texture, area, data, 256 * side, 256, 0, texture_plane::stencil);
							gpu.submit().wait();
							const auto bytes = map_buffer(data, 0, data.length);
							const unsigned unit = format == 0x92 ? 2 : 4;
							for (unsigned y = 0; y < side; ++y)
								for (unsigned x = 0; x < side; ++x)
								{
									const unsigned part = owner(x, y);
									const unsigned expected = format == 0x92 ? (part == 2 ? vm_word : part * 65535) :
										std::bit_cast<unsigned>(part < 2 ? native_depth(part) :
											format == 0x90 ? float(vm_word >> 8) / 16777215.f : .5f);
									unsigned actual = 0;
									std::memcpy(&actual, bytes.data() + y * 256 + x * unit, unit);
									require(actual == expected, "Stored RTT ring changed the winning depth word");
									++depth_words;
									if (stencil)
									{
										require(bytes[256 * side + y * 256 + x] == std::byte(part < 2 ? 63 + part : 42),
											"Stored RTT ring changed the winning stencil byte");
										++stencil_bytes;
									}
								}
						}
						for (const auto point : {std::array{0u, 0u}, std::array{side - 1, side - 1}, std::array{side / 2, 0u}})
						{
							const unsigned part = owner(point[0], point[1]);
							const color coord{(float(point[0]) - .5f) / extent, (float(point[1]) - .5f) / extent, 0, 1};
							if (depth)
							{
								const float value = part < 2 ? native_depth(part) : format == 0x92 ? 32768.f / 65535 :
									format == 0x90 ? float(vm_word >> 8) / 16777215.f : .5f;
								sample(image, r, coord, {value, value, value, value});
							}
							else
							{
								const std::array<color, 3> colors{{{16.f / 255, 48.f / 255, 80.f / 255, 1},
									{128.f / 255, 144.f / 255, 160.f / 255, 1}, {32.f / 255, 64.f / 255, 96.f / 255, 1}}};
								sample(image, r, coord, colors[part]);
							}
						}
						const auto reads = ram->reads;
						const auto allocations = ledger->statistics().created;
						const auto warm = textures.resolve_versioned(pixels, surfaces);
						require(warm.image.texture == image.texture && warm.revision.current() && ram->reads == reads &&
							ledger->statistics().created == allocations && ram->bytes == original,
							"Warm stored RTT ring recaptured pixels, allocated or changed VM");
						ram->forbidden.clear();
						++profiles;
					}
				}
		std::cout << (morton ? "Morton" : "Linear") << " stored border RTT: " << profiles << " full/partial native profiles, "
			<< cases - first << " samples, " << depth_words << " exact depth words, " << stencil_bytes
			<< " exact stencil bytes, " << hole_bytes << " active VM hole bytes, immutable warm reuse passed\n";
	}
	void stored_border_adjusted_pitch_cases()
	{
		const auto first = cases;
		unsigned profiles = 0, refusals = 0;
		std::size_t vm_holes = 0;
		for (unsigned format : {0x85u, 0x90u, 0x91u, 0x92u, 0x93u})
			for (unsigned mode : {0u, 1u, 2u})
			{
				@autoreleasepool
				{
					const unsigned width = mode == 0 ? 3 : mode == 1 ? 5 : 1;
					const unsigned height = mode == 0 ? 4 : 1;
					const unsigned columns = width + 2, rows = height + 2;
					const auto codec = describe_guest_texture_format(format);
					const unsigned bpp = codec.block.bytes, pitch = columns * bpp + (mode != 1);
					auto r = words(format, vertex_texture_dimension::two, true,
						mode == 1 ? width : 9, mode == 1 ? 7 : height, 1, mode == 2 ? 4 : 1);
					r.format &= ~8u;
					r.address = (7u << 28) | 0x040404;
					r.control3 = (1u << 20) | (mode == 1 ? 0 : pitch);
					const auto plan = plan_guest_texture(r);
					const auto& level = plan.subresources.front();
					require(plan.adjusted_pitch && plan.width == width && plan.height == height &&
						plan.mips == 1 && level.row_pitch == pitch && level.storage_rows == rows &&
						plan.storage_bytes == pitch * rows, "Adjusted stored ring has inconsistent geometry or byte bounds");
					const auto raw = [&](unsigned x, unsigned y)
					{
						const bool high = (x + y) % 2;
						switch (format)
						{
						case 0x90: return high ? 0xc0000034u : 0x40000017u;
						case 0x91: return high ? 0x7e800034u : 0x7d000017u;
						case 0x92: return high ? 0xc000u : 0x4000u;
						case 0x93: return high ? 0x6800u : 0x5000u;
						default: return 0xff000000u | ((16 + 16 * x + 4 * y) << 16) | 0x4050u;
						}
					};
					const auto expected = [&](unsigned x, unsigned y) -> color
					{
						const unsigned word = raw(x, y);
						if (!codec.depth)
							return {float((word >> 16) & 255) / 255, 64.f / 255, 80.f / 255, 1};
						const float depth = format == 0x90 ? float(word >> 8) / 16777215.f :
							format == 0x92 ? float(word) / 65535.f : (x + y) % 2 ? .75f : .25f;
						return {depth, depth, depth, depth};
					};
					ram->bytes.assign(pitch * rows, std::byte{0xbd});
					for (unsigned y = 0; y < rows; ++y)
						for (unsigned x = 0; x < columns; ++x)
							for (unsigned byte = 0; byte < bpp; ++byte)
								ram->bytes[y * pitch + x * bpp + byte] = std::byte(raw(x, y) >> (8 * (bpp - 1 - byte)));
					++ram->version;
					const auto original = ram->bytes;
					const auto image = resolver.resolve(r);
					require(image.metadata.stored_border_extent == std::array{width, height} &&
						image.texture.width == columns && image.texture.height == rows && image.texture.mipmapLevelCount == 1,
						"Adjusted stored ring lost its effective extent");
					const float right = (width - .5f) / width, bottom = (height - .5f) / height;
					sample(image, r, {right, bottom, 0, 1}, expected(width, height), 47, 0);
					sample(image, r, {8, 8, 0, 1}, expected(width + 1, height + 1));
					auto pixels = r;
					pixels.format |= 0x40u << 8;
					sample(resolver.resolve(pixels), pixels, {2 * width - 1.f, 2 * height - 1.f, 0, 2}, expected(width, height), 24);
					auto edge = r;
					edge.address = (edge.address & ~0xffffu) | 0x0303;
					sample(image, edge, {8, 8, 0, 1}, expected(width, height));
					require(ram->bytes == original, "Adjusted stored ring upload changed guest storage");

					const cpu_capture_dispatch capture = [](const auto& work) { work(); };
					context gpu{1};
					auto ledger = std::make_shared<heap_budget>(gpu, 8u << 20);
					gpu.set_buffer_budget(ledger);
					surface_cache surfaces(gpu, ram, 8u << 20, ledger, capture);
					guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000 + offset - 0x200; },
						1u << 20, ledger, 64, {capture, std::make_shared<io_mapping_epoch>()});
					const auto old = textures.resolve_versioned(r, surfaces);
					const unsigned corner = 0x1000 + (rows - 1) * pitch + (columns - 1) * bpp;
					const surface_format sf = codec.depth ? surface_format{*codec.depth} : surface_format{surface_color::argb8};
					const auto source = surfaces.acquire({corner, {1, 1, 1, sf, false}, bpp, {}});
					if (codec.depth)
					{
						surfaces.prepare(source, surface_initialization::background);
						const std::array uses{cached_surface_use{source, true}};
						surfaces.draw(uses, [&](context& commands, auto images)
						{
							auto pass = [MTL4RenderPassDescriptor new];
							pass.depthAttachment.texture = images[0];
							pass.depthAttachment.loadAction = MTLLoadActionClear;
							pass.depthAttachment.storeAction = MTLStoreActionStore;
							pass.depthAttachment.clearDepth = 1;
							(void)commands.render(pass);
						});
					}
					else
						surfaces.clear(source, 0xf0, 0xff1070b0, 0, 255, {0, 0, 1, 1});
					ram->forbidden.push_back({corner, bpp});
					const auto batch = textures.resolve_versioned_batch(std::array{r, pixels}, surfaces);
					const auto& fresh = batch[0].image;
					const auto holes = (columns * rows - 1) * bpp;
					require(fresh.texture != old.image.texture && fresh.texture == batch[1].image.texture &&
						fresh.metadata.stored_border_extent == std::array{width, height} && textures.statistics().assembly_vm_bytes == holes,
						"Adjusted stored ring did not coalesce native corner and VM pixels");
					const color new_corner = codec.depth ? color{1, 1, 1, 1} : color{16.f / 255, 112.f / 255, 176.f / 255, 1};
					sample(fresh, r, {8, 8, 0, 1}, new_corner);
					sample(batch[1].image, pixels, {width + 4.f, height + 4.f, 0, 1}, new_corner);
					sample(fresh, r, {right, bottom, 0, 1}, expected(width, height));
					sample(old.image, r, {8, 8, 0, 1}, expected(width + 1, height + 1));
					const auto reads = ram->reads;
					require(textures.resolve_versioned(r, surfaces).image.texture == fresh.texture &&
						ram->reads == reads && ram->bytes == original, "Adjusted stored ring lost warm reuse or changed VM");
					ram->forbidden.clear();
					vm_holes += holes;
					++profiles;

					auto invalid = r;
					invalid.control3 = (1u << 20) | (3 * bpp - 1);
					const auto translations = translated;
					bool refused = false;
					try { (void)resolver.resolve(invalid); }
					catch (const std::invalid_argument&) { refused = true; }
					require(refused && translated == translations && ram->reads == reads,
						"Incomplete adjusted ring accessed VM");
					++refusals;
				}
			}
		std::cout << "Adjusted stored border: " << profiles << " color/depth profiles, " << cases - first
			<< " native samples, " << vm_holes << " active RTT hole bytes, " << refusals
			<< " early refusals, warm and old versions preserved\n";
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
		const auto adjusted_begin=cases;
		for(const auto& t:tests)for(bool zero:{false,true})
		{
			auto r=words(t.format,vertex_texture_dimension::two,true,12,4);
			const auto codec=describe_guest_texture_format(t.format);
			const unsigned columns=4/codec.block.width;
			const unsigned pitch=columns*codec.block.bytes;
			r.control3=1u<<20 | (zero?0u:pitch+(codec.block.bytes>1));
			const auto plan=plan_guest_texture(r);
			require(plan.adjusted_pitch&&plan.width==(zero?12u:4u)&&plan.height==(zero?1u:4u),"Adjusted codec geometry differs");
			fill(plan,t.bytes);
			const auto original=ram->bytes;
			sample(resolver.resolve(r),r,{.5f/plan.width,.875f,0,1},t.expected);
			require(ram->bytes==original,"Adjusted codec changed VM or padding");
		}
		std::cout<<"Adjusted pitch codec draws:"<<cases-adjusted_begin<<" across all 27 codecs with zero and short pitch passed\n";
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
			{
				r.format &= ~8u;
				r.address = (r.address & ~15u) | 1u;
			}
			if (invalid == 1)
				r.control3 = (r.control3 & ~0xfffffu) | 3u; // Less than one ARGB8 block.
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
	void conversion_dependencies()
	{
		const auto first = cases;
		const color argb{23.f / 255, 61.f / 255, 97.f / 255, 1};
		auto mipmapped = words(0x85, vertex_texture_dimension::two, true, 4, 4, 1, 3);
		fill(plan_guest_texture(mipmapped), {255, 23, 61, 97});
		const auto before = ctx.encoding_statistics();
		const auto image = resolver.resolve(mipmapped);
		const auto after = ctx.encoding_statistics();
		require(after.checked_dispatches - before.checked_dispatches == 3 &&
			after.dispatch_barriers - before.dispatch_barriers == 2 &&
			after.dispatch_to_blit_barriers - before.dispatch_to_blit_barriers == 3 &&
			after.blit_to_dispatch_barriers - before.blit_to_dispatch_barriers == 2 &&
			after.checked_blit_raw_transitions == before.checked_blit_raw_transitions,
			"Mip scratch reuse did not track both directions of the blit/dispatch dependency");
		for (unsigned mip = 0; mip < 3; ++mip)
			sample(image, mipmapped, {.5f, .5f, 0, 1}, argb, 47, mip);

		auto vtc = words(0x86, vertex_texture_dimension::three, true, 8, 8, 4);
		vtc.control3 = (4u << 20) | 16u;
		fill(plan_guest_texture(vtc), {0, 0xf8, 0, 0, 0, 0, 0, 0});
		const auto vtc_before = ctx.encoding_statistics();
		const auto decoded = resolver.resolve(vtc);
		const auto vtc_after = ctx.encoding_statistics();
		require(vtc_after.checked_dispatches - vtc_before.checked_dispatches == 2 &&
			vtc_after.blit_to_dispatch_barriers - vtc_before.blit_to_dispatch_barriers == 1 &&
			vtc_after.dispatch_to_blit_barriers - vtc_before.dispatch_to_blit_barriers == 1 &&
			vtc_after.checked_blit_raw_transitions == vtc_before.checked_blit_raw_transitions,
			"VTC gather and hardware decode did not share checked stage dependencies");
		sample(decoded, vtc, {.5f, .5f, .625f, 1}, {1, 0, 0, 1});

		context gpu{1};
		auto ledger = std::make_shared<heap_budget>(gpu, 8u << 20);
		gpu.set_buffer_budget(ledger);
		guest_texture_resolver uploads(gpu, ram, [](unsigned, unsigned, std::size_t) { return 0x1000; }, 1u << 20, ledger);
		const auto single = words(0x85);
		std::vector<captured_guest_texture> snapshots;
		for (unsigned red : {37u, 123u})
		{
			fill(plan_guest_texture(single), {255, red, 61, 97});
			snapshots.push_back(uploads.capture(single));
		}
		const auto images = uploads.enqueue_uploads(snapshots).finish();
		const auto batch = gpu.encoding_statistics();
		require(batch.compute_encoders == 1 && batch.checked_dispatches == 2 && batch.dispatch_barriers == 0 &&
			batch.dispatch_barriers_elided == 2 && batch.dispatch_to_blit_barriers == 2 &&
			batch.blit_to_dispatch_barriers == 0 && batch.dispatch_queue_barriers == 1 && batch.texture_blit_queue_barriers == 1,
			"Independent conversions introduced an unnecessary dispatch dependency");
		sample(images[0], single, {.5f, .5f, 0, 1}, {37.f / 255, 61.f / 255, 97.f / 255, 1});
		sample(images[1], single, {.5f, .5f, 0, 1}, {123.f / 255, 61.f / 255, 97.f / 255, 1});
		std::cout << "Checked native texture conversions: " << cases - first
			<< " exact samples, three mip dispatches, VTC gather/decode, two independent batch dispatches without intra-pass waits passed\n";
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
					if (plan.swizzled && plan.codec.block.height == 1)
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
					if (plan.swizzled && plan.codec.block.height == 1)
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
	void packed_depth_color_filtering()
	{
		const auto translate = [](unsigned, unsigned, std::size_t) { return 0x1000u; };
		constexpr std::array<std::uint32_t, 2> clears{0x23456738, 0x9abcde72};
		const auto byte = [](std::uint32_t word, unsigned shift) { return float((word >> shift) & 255) / 255; };
		for (const auto depth : {surface_depth::z24s8_uint, surface_depth::z24s8_float})
		for (const unsigned samples : {1u, 2u, 4u})
		{
			context gpu{1};
			auto budget = std::make_shared<heap_budget>(gpu, 4 << 20);
			gpu.set_buffer_budget(budget);
			surface_cache surfaces(gpu, ram, 4 << 20, budget);
			guest_texture_cache textures(gpu, ram, translate, 1 << 20, budget);
			guest_texture_cache batched(gpu, ram, translate, 1 << 20, budget);
			auto r = words(0x85);
			const auto plan = plan_guest_texture(r);
			fill(plan, {0, 0, 0, 0});
			for (const unsigned format : {0x85u, 0x9eu})
				require(textures.resolve(words(format)).metadata.linear_filtering, "Ordinary VM color lost linear filtering");
			const auto factors = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
			const auto width = 4u / factors.x, height = 4u / factors.y;
			const auto source = surfaces.acquire({0x1000, {width, height, samples, depth, false}, plan.subresources[0].row_pitch, {}});
			surfaces.clear(source, 3, 0, clears[0], 255, {0, 0, width, height});
			surfaces.clear(source, 3, 0, clears[1], 255, {width / 2, 0, width / 2, height});
			const auto reads = ram->reads;
			const auto revision = ram->version;
			for (const unsigned format : {0x85u, 0x9eu})
			{
				r = words(format);
				r.filter = (2u << 24) | (2u << 16);
				const auto first = textures.resolve(r, surfaces);
				const auto warm = textures.resolve(r, surfaces);
				const auto deferred = batched.resolve_versioned_batch(std::array{r}, surfaces).front().image;
				require(first.texture == warm.texture && textures.statistics().casts_reused != 0,
					"Packed depth color cache did not reuse the native cast");
				for (const auto& image : {first, warm, deferred})
				{
					const texture_state_overrides overrides{.anisotropy = 8};
					for (unsigned i = 0; i < clears.size(); ++i)
					{
						auto packed = clears[i];
						if (depth == surface_depth::z24s8_float)
							packed = ((std::bit_cast<std::uint32_t>(float(packed >> 8) / 16777215.f) >> 7) << 8) | (packed & 255);
						sample(image, r, {i ? .51f : .49f, .375f, 0, 1},
							{byte(packed, 16), byte(packed, 8), byte(packed, 0), format == 0x85 ? byte(packed, 24) : 1.f},
							23, 0, nullptr, 0, overrides);
					}
					const auto state = decode_fragment_texture_state(r, image.metadata, {.anisotropy = 8});
					const auto sampler = resolve_sampler(state.sampler, state.policy);
					require(sampler.min_filter == MTLSamplerMinMagFilterNearest &&
						sampler.mag_filter == MTLSamplerMinMagFilterNearest && sampler.anisotropy == 1,
						"Packed depth color retained interpolation");
				}
			}
			require(ram->reads == reads && ram->version == revision && surfaces.status(source).dirty,
				"Packed depth color read or wrote guest pixels");
		}
		std::cout << "Packed D24S8 color point filtering: 72 native draws, integer/float depth, 1/2/4 source samples, ARGB8/XRGB8, cold/warm/batched, no surface VM transfer\n";
	}
	void redirected_depth_shader()
	{
		const auto translate = [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; };
		constexpr std::array<std::uint32_t, 2> clears{0x23456738, 0x9abcde72};
		const auto expected = [](std::uint32_t value, surface_depth depth, unsigned format)
		{
			if (depth == surface_depth::z24s8_float)
				value = ((std::bit_cast<std::uint32_t>(float(value >> 8) / 16777215.f) >> 7) << 8) | (value & 255);
			const auto byte = [&](unsigned shift) { return float((value >> shift) & 255) / 255; };
			return color{byte(16), byte(8), byte(0), format == 0x85 ? byte(24) : 1.f};
		};
		const auto before = cases;
		for (const auto depth : {surface_depth::z24s8_uint, surface_depth::z24s8_float})
		for (const unsigned samples : {1u, 2u, 4u})
		{
			context gpu{1};
			auto budget = std::make_shared<heap_budget>(gpu, 4 << 20);
			gpu.set_buffer_budget(budget);
			surface_cache surfaces(gpu, ram, 4 << 20, budget);
			guest_texture_cache textures(gpu, ram, translate, 1 << 20, budget);
			auto r = words(0x85);
			const auto plan = plan_guest_texture(r);
			const auto factors = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
			const auto width = 4u / factors.x, height = 4u / factors.y;
			const auto source = surfaces.acquire({0x1000, {width, height, samples, depth, false}, plan.subresources[0].row_pitch, {}});
			surfaces.clear(source, 3, 0, clears[0], 255, {0, 0, width, height});
			surfaces.clear(source, 3, 0, clears[1], 255, {width / 2, 0, width / 2, height});
			const auto reads = ram->reads;
			const auto revision = ram->version;
			for (const unsigned format : {0x85u, 0x9eu})
			{
				r = words(format);
				r.filter = (2u << 24) | (2u << 16);
				const auto resolve = [&](auto regs)
				{
					return textures.prepare_versioned_batch(std::array{regs}, surfaces, {}, {}, texture_upload_path::separate, 1).finish().front().image;
				};
				const auto image = resolve(r);
				const auto warm = resolve(r);
				require(image.metadata.depth && image.metadata.depth_float == (depth == surface_depth::z24s8_float) &&
					image.texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && image.texture == warm.texture &&
					textures.statistics().casts_created == 0,
					"Draw-only redirected depth did not retain and reuse its original native snapshot");
				for (unsigned opcode : {23u, 24u, 25u, 47u, 49u})
					for (unsigned i = 0; i < clears.size(); ++i)
					{
						const auto x = i ? .51f : .49f;
						const color coords = opcode == 24 ? color{2 * x, .75f, 0, 2} : color{x, .375f, 0, 1};
						sample(image, r, coords, expected(clears[i], depth, format), opcode, 2, nullptr, 0, {.anisotropy = 8});
					}
				const auto first = expected(clears[0], depth, format);
				const auto second = expected(clears[1], depth, format);
				const color border{32.f / 255, 64.f / 255, 128.f / 255, 16.f / 255};
				r.border_color = 0x10204080;
				for (const auto& [wrap, x, value] : {std::tuple{1u, -0.25f, second}, std::tuple{2u, -0.25f, first},
					std::tuple{3u, -0.75f, first}, std::tuple{4u, -0.25f, border}, std::tuple{5u, -0.75f, first},
					std::tuple{6u, -0.75f, second}, std::tuple{7u, -1.25f, second}, std::tuple{8u, -1.25f, second}})
				{
					r.address = (r.address & ~15u) | wrap;
					sample(image, r, {x, .375f, 0, 1}, value);
				}
				r = words(format);
				r.control1 = 0x921b; // RGBA output selects zero, one, A, B.
				sample(image, r, {.25f, .375f, 0, 1}, {0, 1, first[3], first[2]});
				r = words(format);
				r.format |= 0x40u << 8;
				sample(resolve(r), r, {2, 1, 0, 1}, second);
				r = words(format);
				r.offset += 4;
				r.image_rect = (2u << 16) | 4u;
				const auto crop = resolve(r);
				require(crop.metadata.depth && crop.texture.width == 2 && textures.statistics().casts_created == 0,
					"Aligned redirected depth crop did not retain both source planes");
				sample(crop, r, {.25f, .375f, 0, 1}, first);
				sample(crop, r, {.75f, .375f, 0, 1}, second);
			}
			require(ram->reads == reads && ram->version == revision && surfaces.status(source).dirty,
				"Redirected depth shader read or wrote guest pixels");
		}
		std::cout << "Redirected D24S8 shader: " << cases - before
			<< " native draws, paired plane reads, TEX/TXP/TXD/TXL/TXB, remap, eight wrap modes, pixel coordinates and aligned crops, no cast or VM transfer\n";
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
					require(surfaces.status(destination).deferred_copies == 1 && !surfaces.status(destination).initialized && ram->reads == reads,
							"Deferred surface texture source read stale VM or materialized its destination");
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
	enum class region_source { resident, spilled, native_spilled, deferred };
	void region_alias_cases(region_source input = region_source::resident)
	{
		const bool native_spilled = input == region_source::native_spilled;
		const bool spilled = input == region_source::spilled || native_spilled;
		const bool deferred = input == region_source::deferred;
		std::optional<context> upload_context;
		std::shared_ptr<heap_budget> upload_budget;
		if (spilled || deferred)
		{
			upload_context.emplace(1);
			upload_budget = std::make_shared<heap_budget>(*upload_context, 8u << 20);
			upload_context->set_buffer_budget(upload_budget);
		}
		auto& gpu = upload_context ? *upload_context : ctx;
		unsigned draws = 0, async_regions = 0, lifetime_cases = 0, native_regions = 0;
		for (const unsigned samples : {1u, 2u, 4u})
		for (unsigned format : {0x81u, 0x82u, 0x84u, 0x85u, 0x8bu, 0x9au, 0x9bu, 0x9cu, 0x90u, 0x91u, 0x92u, 0x93u})
			for (unsigned storage_case : {0u, 1u, 2u, 3u, 4u})
			{
				if ((native_spilled && samples == 1) || (format == 0x82 && !native_spilled)) continue;
				if (storage_case == 4 && (!native_spilled || format != 0x85)) continue;
				const bool direct_spill = native_spilled &&
					(format == 0x81 || format == 0x82 || format == 0x84 || format == 0x85 || format == 0x8b ||
					 format == 0x9a || format == 0x9b || format == 0x9c || format == 0x92);
				const bool linear = storage_case == 0 || storage_case == 4;
				const unsigned source_width = storage_case == 2 ? 4 : 8;
				const std::array<std::size_t, 3> storage{source_width, 8, storage_case == 3 ? 2u : 1u};
				const auto source_words = words(format, vertex_texture_dimension::two, linear, source_width, 8);
				const auto plan = plan_guest_texture(source_words);
				fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
				ram->bytes.resize((linear ? plan.storage_bytes : storage[0] * storage[1] * storage[2] * plan.codec.block.bytes), std::byte{0});
				auto budget = upload_context ? upload_budget : std::make_shared<heap_budget>(gpu, 8 << 20);
				surface_cache surfaces(gpu, ram, 8 << 20, budget);
				guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t)
											 { return 0x1000u + offset - 0x200; }, 1 << 20, budget);
				const auto color_format = [&]() -> surface_color
				{
					switch (format)
					{
					case 0x81:
						return surface_color::b8;
					case 0x82:
						return surface_color::x1rgb5_one;
					case 0x84:
						return surface_color::rgb565;
					case 0x85:
						return storage_case == 4 ? surface_color::abgr8 : surface_color::argb8;
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
				auto writer = source;
				if (deferred)
				{
					ram->bytes.resize(0x1000 + ram->bytes.size());
					auto donor = description;
					donor.address = 0x2000;
					writer = surfaces.acquire(donor);
				}
				const auto expected = [&](unsigned x, unsigned y) -> color
				{
					const float depth = .50123456f + x / 256.f + y / 16.f;
					switch (format)
					{
					case 0x81:
						return {(32 + x * 13 + y * 7) / 255.f, (32 + x * 13 + y * 7) / 255.f, (32 + x * 13 + y * 7) / 255.f, 1};
					case 0x82:
						return {std::floor((x + 1) / 31.f * 255) / 255, std::floor((y + 1) / 31.f * 255) / 255, std::floor((x + y + 1) / 31.f * 255) / 255, 1};
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
				const auto raw = make_buffer(gpu, 2048, buffer_storage::shared), stencil = make_buffer(gpu, 2048, buffer_storage::shared);
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
						case 0x82:
							put(x, y, 2, 0x8000u | ((x + 1) << 10) | ((y + 1) << 5) | (x + y + 1), 2);
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
				const std::array uses{cached_surface_use{writer, true}};
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
				const auto revision = ram->version;
				if (spilled) surfaces.spill(source);
				if (deferred)
					surfaces.defer_copy(writer, {0, 0, description.allocation.width, description.allocation.height}, source, 0, 0);
				const auto stays_pending = [&]
				{
					const auto state = surfaces.status(source);
					return spilled ? state.spilled : state.deferred_copies == 1 && !state.initialized;
				};
				for (const auto rectangle : regions)
				{
					const auto r = input_words(rectangle);
					resolved_fragment_texture image;
					if (upload_context)
					{
						const auto before = surfaces.transaction_statistics().submitted;
						const auto leases = surfaces.memory_statistics().leased_allocations;
						const auto gate = [gpu.device() newSharedEvent];
						struct unblock { id<MTLSharedEvent> event; ~unblock() { event.signaledValue = 1; } } release{gate};
						auto pixels = r;
						pixels.format |= 0x40u << 8;
						const auto check_lifetime = format == 0x85 && linear && rectangle.x == 1;
						if (check_lifetime)
						{
							auto canceled = textures.prepare_versioned_batch(std::array{r}, surfaces,
								std::array{guest_texture_upload_dependency{gate, 1}}, {}, texture_upload_path::separate, native_spilled ? 1 : 0);
							require(canceled.cancel_pending(), "Cold snapshot job could not cancel before its dependency");
							bool rejected = false;
							try { (void)canceled.finish(); }
							catch (const std::runtime_error&) { rejected = true; }
							require(rejected && stays_pending() &&
								surfaces.transaction_statistics().submitted == before,
								deferred ? "Cold native texture materialized its deferred source before the upload dependency" :
									"Canceled spill job published or restored the source");
							gpu.recover_completed_errors();
						}
						const bool cached = direct_spill && surfaces.snapshot_statistics().created == 1;
						auto job = textures.prepare_versioned_batch(std::array{r, pixels}, surfaces,
							std::array{guest_texture_upload_dependency{gate, 1}}, {}, texture_upload_path::separate, native_spilled ? 3 : 0);
						require(job.ready() == cached && stays_pending() &&
							surfaces.memory_statistics().leased_allocations == leases && surfaces.transaction_statistics().submitted == before,
							deferred ? "Cold native texture materialized its deferred source before the upload dependency" :
								"Cold native texture restored its spilled source before the upload dependency");
						if (check_lifetime)
						{
							const auto reserved = budget->statistics().reserved_bytes;
							bool timed_out = false;
							try { (void)job.finish(std::chrono::milliseconds(1)); }
							catch (const submission_wait_timeout&) { timed_out = true; }
							require(timed_out && !job.ready() && budget->statistics().reserved_bytes == reserved &&
								stays_pending(), "Timed-out snapshot job retired live storage");
							++lifetime_cases;
						}
						gate.signaledValue = 1;
						const auto ready = job.finish();
						image = ready[0].image;
						const auto warm = native_spilled ? textures.prepare_versioned_batch(std::array{r}, surfaces, {}, {},
							texture_upload_path::separate, 1).finish().front() : textures.resolve_versioned(r, surfaces);
						require(ready[0].revision.current() && ready[1].revision.current() &&
							image.texture == ready[1].image.texture && warm.image.texture == image.texture &&
							stays_pending() && surfaces.transaction_statistics().submitted == before && ram->version == revision,
							"Pending native snapshot changed authority or missed immutable reuse");
						++async_regions;
					}
					else
						image = textures.resolve(r, surfaces);
					require(image.metadata.samples == (direct_spill ? samples : 1) &&
						image.texture.width == (direct_spill ? description.allocation.width : rectangle.width) &&
						image.texture.height == (direct_spill ? description.allocation.height : rectangle.height) && ram->reads == reads,
							"GPU surface subregion read VM or used incorrect native snapshot dimensions");
					if (direct_spill) ++native_regions;
					sample(image, r, {.5f / rectangle.width, .5f / rectangle.height, 0, 1}, expected(rectangle.x, rectangle.y));
					sample(image, r, {1 - .5f / rectangle.width, 1 - .5f / rectangle.height, 0, 1}, expected(rectangle.x + rectangle.width - 1, rectangle.y + rectangle.height - 1));
					draws += 2;
					if (format == 0x85 && linear && rectangle.x == 1)
					{
						auto un = r;
						un.format |= 0x40u << 8;
						const auto pixels = native_spilled ? textures.prepare_versioned_batch(std::array{un}, surfaces, {}, {},
							texture_upload_path::separate, 1).finish().front().image : textures.resolve(un, surfaces);
						require(pixels.metadata.coordinates.scale == std::array<float, 3>{1.f / 3, 1.f / 2, 1}, "Cropped UN metadata used the containing surface dimensions");
						sample(pixels, un, {.5f, .5f, 0, 1}, expected(rectangle.x, rectangle.y));
						++draws;
					}
					// Compare every native storage word, not only the two sampled endpoints.
					// This detects guest float-codec quantization and any sample-coordinate mixup.
					const auto plane = plan.codec.depth ? texture_plane::depth : texture_plane::all;
					const auto native_bytes = storage_block(image.texture.pixelFormat, plane).bytes;
					const auto copied = make_buffer(gpu, 256 * rectangle.height, buffer_storage::shared);
					texture_region copied_region;copied_region.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
					gpu.begin();
					auto copy_source = image.texture;
					if (direct_spill)
					{
						auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.texture.pixelFormat
							width:rectangle.width height:rectangle.height mipmapped:NO];
						descriptor.storageMode = MTLStorageModePrivate;
						descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
						copy_source = make_texture(gpu, descriptor);
						sample_transfer transfer(gpu);
						transfer.expand_region(gpu, image.texture, copy_source, {rectangle.x, rectangle.y});
					}
					texture_to_buffer(gpu, copy_source, copied_region, copied, 0, 256, 0, plane);
					gpu.submit().wait();
					for (unsigned y = 0; y < rectangle.height; ++y)
						for (unsigned x = 0; x < rectangle.width; ++x)
							require(std::memcmp(static_cast<const std::byte*>(copied.contents) + y * 256 + x * native_bytes,
								static_cast<const std::byte*>(raw.contents) + (rectangle.y + y) * 256 + (rectangle.x + x) * native_bytes,
								native_bytes) == 0, "Native MSAA/region snapshot changed a native sample word");
					if (image.texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
					{
						const auto out = make_buffer(gpu, 256 * rectangle.height, buffer_storage::shared);
						texture_region whole;
						whole.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
						gpu.begin();
						texture_to_buffer(gpu, image.texture, whole, out, 0, 256, 0, texture_plane::stencil);
						gpu.submit().wait();
						for (unsigned y = 0; y < rectangle.height; ++y)
							for (unsigned x = 0; x < rectangle.width; ++x)
								require(static_cast<const unsigned char*>(out.contents)[y * 256 + x] == 42 + rectangle.x + x + rectangle.y + y, "Native partial depth copy lost or rebased source stencil incorrectly");
					}
				}
				if (upload_context) continue;
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
		std::cout << draws << " patterned native partial surface RSX draws: "
			<< (native_spilled ? "twelve formats, 2/4 samples" : "eleven formats, 1/2/4 samples")
			<< ", linear address/pitch and Morton axis/carry/volume-slice mapping, cropped dimensions and depth stencil regions passed\n";
		if (native_spilled)
			std::cout << "Native MSAA spill snapshots: " << native_regions << " native windows and " << async_regions - native_regions
				<< " raw float windows, exact sample words and warm reuse, " << lifetime_cases << " cancellation/timeout/retry groups\n";
		else if (upload_context)
			std::cout << (spilled ? "Cold raw spill snapshots: " : "Cold deferred snapshots: ") << async_regions
				<< " event-gated crops, exact native planes, no surface preparation submission, VM pixel read or revision change, "
				<< lifetime_cases << " cancellation/timeout/retry groups\n";
	}
	void deferred_snapshot_epochs()
	{
		context gpu{2};
		auto budget = std::make_shared<heap_budget>(gpu, 8u << 20);
		gpu.set_buffer_budget(budget);
		surface_cache surfaces(gpu, ram, 8u << 20, budget);
		guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200; },
			1u << 20, budget);
		ram->bytes.assign(0x3000, std::byte{0});
		++ram->version;
		const surface_allocation_description format{4, 4, 1, surface_color::argb8, false};
		const auto a = surfaces.acquire({0x2000, format, 16, {}});
		const auto b = surfaces.acquire({0x3000, format, 16, {}});
		const auto target = surfaces.acquire({0x1000, {6, 6, 1, surface_color::argb8, false}, 24, {}});
		surfaces.clear(a, 0xf0, 0xff101010, 0, 255, {0, 0, 4, 4});
		surfaces.clear(a, 0xf0, 0xff204080, 0, 255, {2, 1, 2, 2});
		surfaces.clear(b, 0xf0, 0xff806020, 0, 255, {0, 0, 4, 4});
		surfaces.defer_copy(a, {2, 1, 2, 2}, target, 1, 2);
		auto crop = words(0x85, vertex_texture_dimension::two, true, 2, 2);
		crop.offset += 2 * 24 + 4;
		crop.control3 = (1u << 20) | 24;
		const auto original = ram->bytes;
		const auto reads = ram->reads;
		const auto gate = [gpu.device() newSharedEvent];
		struct unblock { id<MTLSharedEvent> event; ~unblock() { event.signaledValue = 1; } } release{gate};
		auto old = textures.prepare_versioned_batch(std::array{crop}, surfaces, std::array{guest_texture_upload_dependency{gate, 1}});
		require(!old.ready() && surfaces.status(target).deferred_copies == 1, "Shifted deferred crop materialized before completion");
		gate.signaledValue = 1;
		gpu.wait_idle();
		surfaces.defer_copy(b, {0, 0, 2, 2}, target, 4, 4);
		const auto disjoint = textures.resolve_versioned(crop, surfaces);
		require(surfaces.status(target).deferred_copies == 2 && ram->reads == reads,
			"A later disjoint copy forced materialization of the earlier covered crop");
		sample(disjoint.image, crop, {.25f, .25f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		surfaces.defer_copy(b, {0, 0, 2, 2}, target, 1, 2);
		const auto latest = textures.resolve_versioned(crop, surfaces);
		const auto frozen = old.finish()[0];
		require(latest.image.texture != frozen.image.texture &&
			textures.resolve_versioned(crop, surfaces).image.texture == latest.image.texture &&
			surfaces.status(target).deferred_copies == 3 && ram->reads == reads,
			"Older completed deferred crop displaced the newer content epoch");
		sample(frozen.image, crop, {.25f, .25f, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		sample(latest.image, crop, {.75f, .75f, 0, 1}, {128.f / 255, 96.f / 255, 32.f / 255, 1});
		surfaces.cancel_deferred_copies(target);
		const auto canceled = textures.resolve_versioned(crop, surfaces);
		require(canceled.image.texture != latest.image.texture && !surfaces.status(target).initialized,
			"Cancelled deferred crop retained native authority");
		sample(canceled.image, crop, {.25f, .25f, 0, 1}, {0, 0, 0, 0});

		surfaces.defer_copy(a, {2, 1, 2, 2}, target, 1, 2);
		auto whole = words(0x85, vertex_texture_dimension::two, true, 6, 6);
		whole.control3 = (1u << 20) | 24;
		const auto uncovered = textures.resolve_versioned(whole, surfaces);
		require(surfaces.status(target).initialized && surfaces.status(target).dirty &&
			surfaces.status(target).deferred_copies == 0 && ram->bytes == original,
			"Uncovered deferred request lost normal materialization or wrote guest bytes");
		sample(uncovered.image, whole, {.25f, 2.5f / 6, 0, 1}, {32.f / 255, 64.f / 255, 128.f / 255, 1});
		sample(uncovered.image, whole, {.5f / 6, .5f / 6, 0, 1}, {0, 0, 0, 0});
		std::cout << "Deferred snapshot epochs: shifted donor, disjoint and replacing copies, late old completion, cancellation and uncovered fallback passed\n";
	}
	void multisample_snapshot_cases(bool spilled_raster = false)
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
		unsigned filtered_draws = 0;
		unsigned wrap_draws = 0;
		unsigned mixed_draws = 0;
		unsigned anisotropic_draws = 0;
		unsigned window_draws = 0;
		const auto before = cases;
		for (const unsigned samples : {2u, 4u})
			for (const unsigned format_code : {0x9bu, 0x90u, 0x91u})
				for (const bool linear : {true, false})
				{
					const bool depth = format_code != 0x9b;
					const unsigned width = 8, height = samples == 2 ? 4 : 8;
					const auto r = words(format_code, vertex_texture_dimension::two, linear, width, height);
					const auto plan = plan_guest_texture(r);
					fill(plan, std::vector<unsigned>(plan.codec.block.bytes, 0));
					const auto original_ram = ram->bytes;
					context gpu{1};
					auto budget = std::make_shared<heap_budget>(gpu, 8 << 20);
					gpu.set_buffer_budget(budget);
					resolved_fragment_texture retained;
					id<MTLTexture> whole = nil;
					std::vector point_words{r};
					if (depth)
					{
						auto shadow = r;
						shadow.address = (shadow.address & 0x0fffffffu) | (1u << 28);
						point_words.push_back(shadow);
						auto packed = r;
						packed.format = (packed.format & ~(0x9fu << 8)) | (0x85u << 8);
						point_words.push_back(packed);
					}
					auto filter_words = point_words;
					for (auto& regs : filter_words)
					{
						regs.filter = (2u << 16) | ((linear ? 2u : 4u) << 24);
						if (!linear)
							regs.address = (regs.address & ~0x0f0fu) | 0x0505u;
					}
					std::vector<versioned_guest_texture> point;
					struct crop_profile { clear_rectangle rectangle; bool pixels; };
					const std::array crops = linear ?
						std::array{crop_profile{{1, 1, 3, 2}, false}, crop_profile{{2, 0, 4, 3}, true}} :
						std::array{crop_profile{{2, 2, 2, 2}, false}, crop_profile{{4, 0, 2, 2}, true}};
					std::vector<fragment_texture_registers> crop_words;
					std::size_t crop_start = 0;
					{
						surface_cache surfaces(gpu, ram, 8 << 20, budget);
						guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; }, 1 << 20, budget);
						const surface_format format = depth ?
							surface_format{format_code == 0x90 ? surface_depth::z24s8_uint : surface_depth::z24s8_float} :
							surface_format{surface_color::rgba32f};
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
						if (spilled_raster) surfaces.spill(surface);
						const auto reads = ram->reads;
						const auto revision = ram->version;
						const auto spill_submissions = surfaces.transaction_statistics().submitted;
						auto batch_words = point_words;
						batch_words.insert(batch_words.end(), filter_words.begin(), filter_words.end());
						auto wrapped = filter_words[0];
						wrapped.address = (wrapped.address & ~0x0f0fu) | 0x0401u;
						batch_words.push_back(wrapped);
						auto mixed = filter_words[0];
						mixed.filter = (2u << 16) | (1u << 24);
						batch_words.push_back(mixed);
						mixed.filter = (3u << 16) | (2u << 24);
						batch_words.push_back(mixed);
						auto anisotropic = filter_words[0];
						anisotropic.control0 |= 7u << 4;
						batch_words.push_back(anisotropic);
						crop_start = batch_words.size();
						for (const auto& crop : crops)
							for (auto regs : point_words)
							{
								const auto& area = crop.rectangle;
								regs.offset += guest_texture_block_offset(plan, plan.subresources[0], area.x, area.y, 0);
								regs.image_rect = (area.width << 16) | area.height;
								if (crop.pixels) regs.format |= 0x40u << 8;
								crop_words.push_back(regs);
							}
						batch_words.insert(batch_words.end(), crop_words.begin(), crop_words.end());
						// A raw, non-fragment request still needs its expanded raster.
						const auto mask = (std::uint64_t{1} << batch_words.size()) - 1;
						batch_words.push_back(r);
						point = textures.prepare_versioned_batch(batch_words, surfaces, {}, {}, texture_upload_path::separate, mask).finish();
						const auto expanded = point.back().image;
						point.pop_back();
						require(expanded.texture.sampleCount == 1 && expanded.texture.width == width && expanded.texture.height == height,
							"Raw texture request did not keep the expanded MSAA raster");
						const auto copies_after_first = surfaces.snapshot_statistics().created;
						auto warm = textures.prepare_versioned_batch(batch_words, surfaces, {}, {}, texture_upload_path::separate, mask);
						require(warm.ready() && surfaces.snapshot_statistics().created == copies_after_first &&
							warm.finish().back().image.texture == expanded.texture, "MSAA storage variants displaced one another");
						if (spilled_raster && depth)
						{
							for (std::size_t i = 0; i < point.size(); ++i)
							{
								const auto& image = point[i].image;
								const auto area = i < crop_start ? clear_rectangle{0, 0, width, height} :
									crops[(i - crop_start) / point_words.size()].rectangle;
								require(image.texture.sampleCount == 1 && image.metadata.samples == 1 &&
									image.texture.width == area.width && image.texture.height == area.height &&
									image.metadata.coordinates.flags == texture_coordinate_sample_raster,
									"Float spill did not bind an exact direct sample raster");
							}
							require(expanded.texture == point.front().image.texture && !expanded.metadata.coordinates.flags &&
								surfaces.status(surface).spilled && surfaces.transaction_statistics().submitted == spill_submissions,
								"Sample raster replaced the raw binding, restored its source or submitted separate work");
						}
						else
						{
							for (const auto& item : point)
								require(item.image.texture == point.front().image.texture && item.image.texture.sampleCount == samples &&
									item.image.metadata.samples == samples && item.image.texture.width == 4 && item.image.texture.height == 4,
									"Point and filtered MSAA requests did not share one native sample snapshot");
						}
						if (!spilled_raster || !depth)
							for (std::size_t i = 0; i < crop_words.size(); ++i)
							{
								const auto& area = crops[i / point_words.size()].rectangle;
								const auto& coords = point[crop_start + i].image.metadata.coordinates;
								require(coords.flags == texture_coordinate_msaa_window &&
									coords.minimum == std::array{float(area.x), float(area.y)} &&
									coords.maximum == std::array{float(area.x + area.width), float(area.y + area.height)},
									"Coalesced native MSAA crop lost its own sample window");
							}
						const auto versioned = textures.resolve_versioned(r, surfaces);
						require(versioned.image.texture == expanded.texture && copies_after_first == (spilled_raster && depth ? 3 : 2),
							"Mixed point/filtered requests created the wrong snapshot variants");
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
						const auto spilled = textures.prepare_versioned_batch(std::array{r}, surfaces, {}, {}, texture_upload_path::separate, 1).finish();
						require(spilled[0].image.metadata.samples == (depth ? 1 : samples) && spilled[0].image.texture.sampleCount == (depth ? 1 : samples) &&
							ram->reads == reads && ram->version == revision, "Spilled MSAA did not retain its native color or expanded depth snapshot");
						++ram->version;
						require(!versioned.revision.current() && !point.front().revision.current(), "Native MSAA source ticket ignored a CPU revision change");
					}
					// Each physical sample has a unique oracle, checked after the source and cache die.
					for (const unsigned p : {0u, 3u})
						for (unsigned s = 0; s < samples; ++s)
							for (const unsigned opcode : {23u, 24u, 25u, 47u, 49u})
							{
								const float x = (2 * p + s % 2 + .5f) / width;
								const float y = ((samples / 2) * p + s / 2 + .5f) / height;
								const color coords = opcode == 24 ? color{2 * x, 2 * y, 1.2f, 2} : color{x, y, .6f, 1};
								const float z = .50123456f + float(s) / 16 + float(p) / 256 + float(p) / 64;
								color result{float(s + 1) / 8, float(p) / 8, float(p) / 8, 1};
								if (depth) result.fill(z);
								sample(point[0].image, point_words[0], coords, result, opcode, 2);
								if (depth)
								{
									result.fill(z < .6f ? 1.f : 0.f);
									sample(point[1].image, point_words[1], coords, result, opcode, 2);
									const auto bits = format_code == 0x90 ? static_cast<std::uint32_t>(z * 16777215.f) :
										std::bit_cast<std::uint32_t>(z) >> 7;
									sample(point[2].image, point_words[2], coords,
										{float((bits >> 8) & 255) / 255, float(bits & 255) / 255,
											float(17 + s * 53 + p + p * 7) / 255, float((bits >> 16) & 255) / 255}, opcode, 2);
								}
							}
					const auto axis_weights = [](float coordinate, float step, unsigned extent)
					{
						const float uv = 1.f / extent;
						const float base = std::floor(coordinate / uv);
						if (std::abs(step - uv) < .000001f)
							return std::array{1.f, 0.f, 0.f};
						if (step < uv)
						{
							const float fraction = coordinate / uv - base;
							return std::array{1.f - fraction, fraction, 0.f};
						}
						const float end = coordinate + step;
						const float one = (base + 1.f) * uv;
						const float two = (base + 2.f) * uv;
						const float middle = std::min(two, end) - one;
						const float last = std::max(two, end) - two;
						return std::array{1.f - middle - last, middle, last};
					};
					const auto address = [](int tap, int extent, unsigned mode)
					{
						if (mode == 4)
							return tap;
						if (mode == 1 || mode == 2)
						{
							const int period = mode == 1 ? extent : extent * 2;
							const int phase = (tap % period + period) % period;
							return phase >= extent ? period - 1 - phase : phase;
						}
						return std::clamp(tap, 0, extent - 1);
					};
					for (const auto [scale, bias] : std::array{std::pair{.0625f, .15625f}, std::pair{.125f, .0625f},
						std::pair{.375f, .0625f}, std::pair{-.125f, .5625f}, std::pair{.1875f, .8125f}})
						for (const unsigned opcode : {23u, 24u, 25u, 47u, 49u})
						{
							fragment_context vary;
							vary.wpos_scale = scale;
							vary.wpos_bias_x = vary.wpos_bias_y = bias;
							const float w = opcode == 24 ? 2.f : 1.f;
							const color coord{0, 0, .6f * w, w};
							std::array<color, 4> pixels, shadow_pixels, packed_pixels, point_pixels, point_shadows;
							for (unsigned y = 0; y < 2; ++y)
								for (unsigned x = 0; x < 2; ++x)
								{
									const float u = ((x + .5f) * std::abs(scale) + bias) / w;
									const float v = ((y + .5f) * scale + bias) / w;
									const auto ix = static_cast<unsigned>(std::clamp(u, 0.f, 1.f) * width);
									const auto iy = static_cast<unsigned>(std::clamp(v, 0.f, 1.f) * height);
									const auto wx = axis_weights(u, std::abs(scale) / w, width);
									const auto wy = axis_weights(v, scale / w, height);
									const auto read = [&](unsigned xx, unsigned yy)
									{
										const auto px = xx / 2, py = yy / (samples / 2);
										const auto s = xx % 2 + 2 * (yy % (samples / 2));
										const float z = .50123456f + float(s) / 16 + float(px) / 256 + float(py) / 64;
										return depth ? color{z, z, z, z} : color{float(s + 1) / 8, float(px) / 8, float(py) / 8, 1};
									};
									auto& result = pixels[y * 2 + x];
									result.fill(0);
									for (unsigned dy = 0; dy < 2; ++dy)
										for (unsigned dx = 0; dx < 3; ++dx)
										{
											const auto texel = read(std::min(ix + dx, width - 1), std::min(iy + dy, height - 1));
											for (unsigned lane = 0; lane < 4; ++lane)
												result[lane] += texel[lane] * wx[dx] * (dy ? wy[1] + wy[2] : wy[0]);
										}
									shadow_pixels[y * 2 + x].fill(result[0] < .6f ? 1.f : 0.f);
									const auto px = std::min(ix, width - 1), py = std::min(iy, height - 1);
									const auto s = px % 2 + 2 * (py % (samples / 2));
									const auto value = read(px, py);
									const auto z = value[0];
									point_pixels[y * 2 + x] = value;
									point_shadows[y * 2 + x].fill(z < .6f ? 1.f : 0.f);
									const auto bits = format_code == 0x90 ? static_cast<std::uint32_t>(z * 16777215.f) :
										std::bit_cast<std::uint32_t>(z) >> 7;
									packed_pixels[y * 2 + x] = {float((bits >> 8) & 255) / 255, float(bits & 255) / 255,
										float(17 + s * 53 + px / 2 + py / (samples / 2) * 7) / 255, float((bits >> 16) & 255) / 255};
								}
							const auto offset = point_words.size();
							sample(point[offset].image, filter_words[0], coord, {}, opcode, 2, &vary, 0, {}, &pixels);
							if (depth)
							{
								sample(point[offset + 1].image, filter_words[1], coord, {}, opcode, 2, &vary, 0, {}, &shadow_pixels);
								sample(point[offset + 2].image, filter_words[2], coord, {}, opcode, 2, &vary, 0, {}, &packed_pixels);
							}
							filtered_draws += depth ? 3 : 1;
							const float slope = std::abs(scale) / w * std::max(width, height);
							const bool implicit_min = slope > 1.f;
							const auto gradient_axis = scale < 0 ? 1u : 0u;
							const float gradient = scale == .0625f ? .25f : scale == .375f ? .03125f : .125f;
							const float level_or_bias = scale == .0625f ? 2.f : scale == .375f ? -2.f : 0.f;
							const float operand = opcode == 25 ? gradient : level_or_bias;
							bool minifies = implicit_min;
							if (opcode == 25)
								minifies = gradient * (gradient_axis ? height : width) > 1.f;
							if (opcode == 47)
								minifies = operand > 0.f;
							if (opcode == 49)
								minifies = slope * std::exp2(operand) > 1.f;
							for (const bool min_linear : {false, true})
							{
								const unsigned min = min_linear ? (linear ? 2u : 4u) : (linear ? 1u : 5u);
								const unsigned mag = min_linear ? 1u : linear ? 2u : 4u;
								auto regs = filter_words[0];
								regs.filter = (min << 16) | (mag << 24);
								const bool selected_linear = min_linear == minifies;
								sample(point[offset].image, regs, coord, {}, opcode, operand, &vary, gradient_axis,
									{}, selected_linear ? &pixels : &point_pixels);
								++mixed_draws;
								if (depth)
								{
									auto shadow_regs = filter_words[1];
									shadow_regs.filter = regs.filter;
									sample(point[offset + 1].image, shadow_regs, coord, {}, opcode, operand, &vary, gradient_axis,
										{}, min_linear == implicit_min ? &shadow_pixels : &point_shadows);
									++mixed_draws;
								}
							}
						}
					struct wrap_profile { float scale, x, y; unsigned s, t; };
					constexpr std::array wrap_profiles{
						wrap_profile{.0625f, .90625f, -.09375f, 1, 1},
						wrap_profile{.25f, .8125f, -1.09375f, 1, 2},
						wrap_profile{.0625f, .90625f, -.09375f, 2, 1},
						wrap_profile{.0625f, -.09375f, .90625f, 4, 4},
						wrap_profile{.0625f, 1.125f, -1.125f, 4, 4},
						wrap_profile{.1875f, -1.125f, -.1875f, 6, 7},
						wrap_profile{-.1875f, -.125f, -.75f, 7, 8},
						wrap_profile{.0625f, -.96875f, .90625f, 8, 3}};
					for (const auto& profile : wrap_profiles)
						for (const unsigned opcode : {23u, 24u})
						{
							const float w = opcode == 24 ? 2.f : 1.f;
							fragment_context vary;
							vary.wpos_scale = profile.scale * w;
							vary.wpos_bias_x = profile.x * w;
							vary.wpos_bias_y = profile.y * w;
							std::array<color, 4> pixels, shadow_pixels;
							const color border{39.f / 255, 1.f, 225.f / 255, 163.f / 255};
							for (unsigned y = 0; y < 2; ++y)
								for (unsigned x = 0; x < 2; ++x)
								{
									float dx = std::abs(profile.scale), dy = profile.scale;
									float u = (x + .5f) * dx + profile.x, v = (y + .5f) * dy + profile.y;
									if (profile.s >= 6 && u < 0) { u = -u; dx = -dx; }
									if (profile.t >= 6 && v < 0) { v = -v; dy = -dy; }
									const auto ix = static_cast<int>(std::floor(u * width));
									const auto iy = static_cast<int>(std::floor(v * height));
									const auto wx = axis_weights(u, dx, width), wy = axis_weights(v, dy, height);
									auto& result = pixels[y * 2 + x];
									result.fill(0);
									for (int row = 0; row < 2; ++row)
										for (int col = 0; col < 3; ++col)
										{
											const int xx = address(ix + col, width, profile.s);
											const int yy = address(iy + row, height, profile.t);
											auto tap = border;
											if (xx >= 0 && yy >= 0 && xx < int(width) && yy < int(height))
											{
												const unsigned px = unsigned(xx) / 2, py = unsigned(yy) / (samples / 2);
												const unsigned s = unsigned(xx) % 2 + 2 * (unsigned(yy) % (samples / 2));
												const float z = .50123456f + float(s) / 16 + float(px) / 256 + float(py) / 64;
												tap = depth ? color{z, 1, z, z} : color{float(s + 1) / 8, 1, float(py) / 8, 1};
											}
											const float weight = wx[col] * (row ? wy[1] + wy[2] : wy[0]);
											for (unsigned lane = 0; lane < 4; ++lane)
												result[lane] += tap[lane] * weight;
										}
									shadow_pixels[y * 2 + x].fill(result[0] < .6f ? 1.f : 0.f);
								}
							for (unsigned view = 0; view < (depth ? 2u : 1u); ++view)
							{
								auto regs = filter_words[view];
								regs.address = (regs.address & ~0x0f0fu) | profile.s | (profile.t << 8);
								regs.border_color = 0xa32779e1;
								regs.control1 = 0x9ae4; // Force green to one in the texture and border.
								sample(point[point_words.size() + view].image, regs, {0, 0, .6f * w, w}, {},
									opcode, 2, &vary, 0, {}, view ? &shadow_pixels : &pixels);
								++wrap_draws;
							}
						}
					struct anisotropic_profile
					{
						unsigned opcode, gradient_axis, wrap_s, wrap_t;
						float scale, bias_x, bias_y, operand;
						bool min_linear;
					};
					constexpr std::array profiles{
						anisotropic_profile{23, 0, 1, 4, .5f, .15625f, -.09375f, 0, true},
						anisotropic_profile{24, 0, 2, 3, .5f, -.15625f, .09375f, 0, false},
						anisotropic_profile{25, 0, 1, 4, .0625f, .90625f, -.0625f, 2.25f, true},
						anisotropic_profile{25, 1, 4, 2, .125f, -.09375f, .90625f, 2.5f, false},
						anisotropic_profile{49, 0, 3, 1, .5f, -.03125f, .40625f, -1.f, false},
						anisotropic_profile{47, 0, 1, 3, .125f, .90625f, .09375f, 2.f, true}};
					constexpr std::array limits{1u, 2u, 4u, 6u, 8u, 10u, 12u, 16u};
					for (unsigned factor = 0; factor < limits.size(); ++factor)
						for (const auto& profile : profiles)
							for (unsigned view = 0; view < (depth ? 2u : 1u); ++view)
							{
								const float w = profile.opcode == 24 ? 2.f : 1.f;
								fragment_context vary;
								vary.wpos_scale = profile.scale * w;
								vary.wpos_bias_x = profile.bias_x * w;
								vary.wpos_bias_y = profile.bias_y * w;
								const float slope = std::abs(profile.scale);
								std::array<float, 2> dx{slope, 0.f}, dy{0.f, profile.scale};
								if (!view && profile.opcode == 25)
								{
									dx = {};
									dy = {};
									dx[profile.gradient_axis] = profile.operand;
								}
								const float x_length = std::hypot(dx[0] * width, dx[1] * height);
								const float y_length = std::hypot(dy[0] * width, dy[1] * height);
								const bool major_x = x_length >= y_length;
								const float major_length = std::max(x_length, y_length);
								const bool explicit_level = !view && profile.opcode == 47;
								const unsigned taps = explicit_level ? 1u :
									std::min(limits[factor], unsigned(std::ceil(major_length / std::max(1.f, std::min(x_length, y_length)))));
								const float bias = !view && profile.opcode == 49 ? profile.operand : 0.f;
								const bool minifies = taps > 1 || (explicit_level ? profile.operand > 0 :
									major_length * std::exp2(bias) > 1.f);
								const bool filter_linear = minifies == profile.min_linear;
								const auto major = major_x ? dx : dy;
								const float x_step = taps > 1 ? dx[0] / (major_x ? taps : 1u) : slope;
								const float y_step = taps > 1 ? dy[1] / (major_x ? 1u : taps) : profile.scale;
								const color border{39.f / 255, 121.f / 255, 225.f / 255, 163.f / 255};
								const auto read = [&](int x, int y)
								{
									x = address(x, width, profile.wrap_s);
									y = address(y, height, profile.wrap_t);
									if (x < 0 || y < 0 || x >= int(width) || y >= int(height))
										return border;
									const unsigned px = unsigned(x) / 2, py = unsigned(y) / (samples / 2);
									const unsigned s = unsigned(x) % 2 + 2 * (unsigned(y) % (samples / 2));
									const float z = .50123456f + float(s) / 16 + float(px) / 256 + float(py) / 64;
									return depth ? color{z, z, z, z} : color{float(s + 1) / 8, float(px) / 8, float(py) / 8, 1};
								};
								std::array<color, 4> pixels{};
								for (unsigned y = 0; y < 2; ++y)
									for (unsigned x = 0; x < 2; ++x)
									{
										auto& result = pixels[y * 2 + x];
										for (unsigned tap = 1; tap <= taps; ++tap)
										{
											const float offset = float(tap) / float(taps + 1) - .5f;
											const float u = (x + .5f) * slope + profile.bias_x + major[0] * offset;
											const float v = (y + .5f) * profile.scale + profile.bias_y + major[1] * offset;
											const int ix = int(std::floor(u * width)), iy = int(std::floor(v * height));
											const auto wx = filter_linear ? axis_weights(u, x_step, width) : std::array{1.f, 0.f, 0.f};
											const auto wy = filter_linear ? axis_weights(v, y_step, height) : std::array{1.f, 0.f, 0.f};
											for (int row = 0; row < 2; ++row)
												for (int col = 0; col < 3; ++col)
												{
													const auto value = read(ix + col, iy + row);
													const float weight = wx[col] * (row ? wy[1] + wy[2] : wy[0]) / taps;
													for (unsigned lane = 0; lane < 4; ++lane)
														result[lane] += value[lane] * weight;
												}
										}
										if (view) result.fill(result[0] < .6f ? 1.f : 0.f);
									}
								auto regs = filter_words[view];
								regs.address = (regs.address & ~0x0f0fu) | profile.wrap_s | (profile.wrap_t << 8);
								regs.border_color = 0xa32779e1;
								regs.filter = profile.min_linear ? (2u << 16) | (1u << 24) : (1u << 16) | (2u << 24);
								const texture_state_overrides overrides{.anisotropy = linear ? 0u : limits[factor]};
								regs.control0 |= linear ? factor << 4 : 0u;
								sample(point[point_words.size() + view].image, regs, {0, 0, .6f * w, w}, {},
									profile.opcode, profile.operand, &vary, profile.gradient_axis, overrides, &pixels);
								++anisotropic_draws;
							}
					for (unsigned index = 0; index < crops.size(); ++index)
					{
						const auto& [area, pixels] = crops[index];
						const auto resource = crop_start + index * point_words.size();
						const auto read = [&](unsigned x, unsigned y, unsigned view)
						{
							x += area.x;
							y += area.y;
							const unsigned px = x / 2, py = y / (samples / 2);
							const unsigned s = x % 2 + 2 * (y % (samples / 2));
							if (!depth)
								return color{float(s + 1) / 8, float(px) / 8, float(py) / 8, 1};
							const float z = .50123456f + float(s) / 16 + float(px) / 256 + float(py) / 64;
							if (!view)
								return color{z, z, z, z};
							if (view == 1)
							{
								const float lit = z < .6f ? 1.f : 0.f;
								return color{lit, lit, lit, lit};
							}
							const auto bits = format_code == 0x90 ? static_cast<std::uint32_t>(z * 16777215.f) :
								std::bit_cast<std::uint32_t>(z) >> 7;
							return color{float((bits >> 8) & 255) / 255, float(bits & 255) / 255,
								float(17 + s * 53 + px + py * 7) / 255, float((bits >> 16) & 255) / 255};
						};
						const float scale_x = pixels ? 1.f : 1.f / area.width;
						const float scale_y = pixels ? 1.f : 1.f / area.height;
						for (unsigned view = 0; view < point_words.size(); ++view)
							for (unsigned y = 0; y < area.height; ++y)
								for (unsigned x = 0; x < area.width; ++x)
								{
									sample(point[resource + view].image, crop_words[index * point_words.size() + view],
										{(2 * x + 1) * scale_x, (2 * y + 1) * scale_y, 1.2f, 2},
										read(x, y, view), 24);
									++window_draws;
								}
						auto regs = crop_words[index * point_words.size()];
						regs.address = (regs.address & ~0x0f0fu) | 0x0101u;
						regs.filter = (2u << 16) | (2u << 24);
						color seam{};
						for (unsigned y = 0; y < 2; ++y)
							for (unsigned x = 0; x < 2; ++x)
							{
								const auto tap = read(x ? 0 : area.width - 1, y ? 0 : area.height - 1, 0);
								const float weight = (x ? .75f : .25f) * (y ? .75f : .25f);
								for (unsigned lane = 0; lane < 4; ++lane) seam[lane] += tap[lane] * weight;
							}
						sample(point[resource].image, regs, {(area.width - .25f) * scale_x, (area.height - .25f) * scale_y, 0, 1}, seam);
						++window_draws;
						regs.filter = (1u << 16) | (1u << 24);
						regs.control0 |= 7u << 4;
						constexpr float gradient = .625f;
						const unsigned taps = unsigned(std::ceil(gradient * area.width));
						color average{};
						for (unsigned tap = 1; tap <= taps; ++tap)
						{
							const float displacement = gradient * area.width * (float(tap) / float(taps + 1) - .5f);
							const unsigned x = unsigned(address(int(std::floor(.25f + displacement)), area.width, 1));
							const auto value = read(x, 0, 0);
							for (unsigned lane = 0; lane < 4; ++lane) average[lane] += value[lane] / taps;
						}
						sample(point[resource].image, regs, {.25f * scale_x, .25f * scale_y, 0, 1},
							average, 25, pixels ? gradient * area.width : gradient);
						++window_draws;
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
		if (spilled_raster)
			std::cout << cases - before << " spilled MSAA draws: native float-color and direct depth sample-raster profiles, " << checks
				<< " exact readbacks, point/linear/mixed/anisotropic footprints, wrap, crop, projection, depth, shadow and stencil\n";
		else
		{
			std::cout << checks << " native MSAA API readbacks across 12 sample_id-written color/depth/stencil profiles:16 repeated hits per profile,whole/region share one immutable image,exact float bits,VM preservation,source overwrite/spill/cache destruction and revision conflicts passed\n";
			std::cout << cases - before - filtered_draws - wrap_draws - mixed_draws - anisotropic_draws - window_draws << " direct MSAA point draws: 2x/4x, color/depth/shadow/redirected stencil, five TEX opcodes, linear/Morton sources and immutable sample order\n";
			std::cout << filtered_draws << " direct MSAA draws with linear filters requested: derivative footprints, shadow after depth filtering and exact packed stencil\n";
			std::cout << wrap_draws << " direct MSAA wrapped-filter draws: seam taps, mirror folding, custom border weights, constant channels and projected coordinates\n";
			std::cout << mixed_draws << " direct MSAA mixed-filter draws: both Min/Mag orders, implicit and explicit LOD, gradients, bias and shadow selection\n";
			std::cout << anisotropic_draws << " direct MSAA anisotropic draws: eight limits, major-axis taps, opcode gradients, projected coordinates, custom border and shadow\n";
			std::cout << window_draws << " direct MSAA window draws: coalesced linear/Morton crops, sample-aligned and odd origins, projected and pixel coordinates, crop-local seams and anisotropy\n";
		}
	}
	void float_multisample_spill_words()
	{
		const auto library = compile.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 special_vertex(uint i [[vertex_id]]) {
    const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    return float4(p[i], 0, 1);
}
fragment float4 write_float_words(float4 p [[position]], uint sample [[sample_id]]) {
    const uint words[] = {0u, 0x80000000u, 1u, 0x807fffffu, 0x00800001u, 0xbf234567u,
        0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc04321u, 0x7f812345u, 0xff854321u};
    uint4 lanes = uint(p.x) + 3 * uint(p.y) + 5 * sample + uint4(0, 1, 4, 7);
    return as_type<float4>(uint4(words[lanes.x % 12], words[lanes.y % 12], words[lanes.z % 12], words[lanes.w % 12]));
}
fragment half4 write_half_words(float4 p [[position]], uint sample [[sample_id]]) {
    const ushort words[] = {0u, 0x8000u, 1u, 0x83ffu, 0x0401u, 0xb923u,
        0x7c00u, 0xfc00u, 0x7e55u, 0xfe21u, 0x7c55u, 0xfc21u};
    uint4 lanes = uint(p.x) + 3 * uint(p.y) + 5 * sample + uint4(0, 1, 4, 7);
    return as_type<half4>(ushort4(words[lanes.x % 12], words[lanes.y % 12], words[lanes.z % 12], words[lanes.w % 12]));
}
kernel void read_float_words(texture2d_ms<float> source [[texture(0)]], device uint4* out [[buffer(0)]],
    uint2 p [[thread_position_in_grid]]) {
    const uint sy = source.get_num_samples() / 2;
    out[p.y * (2 * source.get_width()) + p.x] =
        as_type<uint4>(source.read(p / uint2(2, sy), p.x % 2 + 2 * (p.y % sy)));
}
kernel void read_half_words(texture2d_ms<half> source [[texture(0)]], device ushort4* out [[buffer(0)]],
    uint2 p [[thread_position_in_grid]]) {
    const uint sy = source.get_num_samples() / 2;
    out[p.y * (2 * source.get_width()) + p.x] =
        as_type<ushort4>(source.read(p / uint2(2, sy), p.x % 2 + 2 * (p.y % sy)));
}
)msl", @"Independent float-color MSAA storage words");
		unsigned checked_profiles = 0;
		std::size_t checked_words = 0;
		for (const auto samples : {2u, 4u})
			for (const auto [code, color_format] : std::array{
				std::pair{0x9au, surface_color::rgba16f}, std::pair{0x9bu, surface_color::rgba32f},
				std::pair{0x9cu, surface_color::r32f}})
			{
				const bool half = code == 0x9a;
				const unsigned word_bytes = half ? 2 : 4, channels = code == 0x9c ? 1 : 4;
				const unsigned width = code == 0x9b && samples == 4 ? 257u : 17u;
				const unsigned height = code == 0x9b && samples == 4 ? 129u : 9u;
				const auto size = sample_expanded_size(width, height, samples);
				const auto pitch = size[0] * word_bytes * channels;
				context gpu{1};
				auto budget = std::make_shared<heap_budget>(gpu, 64u << 20);
				gpu.set_buffer_budget(budget);
				surface_cache surfaces(gpu, ram, 64u << 20, budget);
				guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; },
					8u << 20, budget);
				ram->bytes.assign(pitch * size[1], std::byte{});
				++ram->version;
				const auto source = surfaces.acquire({0x1000, {width, height, samples, color_format, false}, pitch, {}});
				const auto reference = make_buffer(gpu, size[0] * size[1] * 4 * word_bytes, buffer_storage::shared);
				auto compute_descriptor = [MTL4ComputePipelineDescriptor new];
				compute_descriptor.computeFunctionDescriptor = library_function(library, half ? @"read_half_words" : @"read_float_words");
				const auto reader = compile.compute(compute_descriptor);
				auto descriptor = [MTL4RenderPipelineDescriptor new];
				descriptor.vertexFunctionDescriptor = library_function(library, @"special_vertex");
				descriptor.fragmentFunctionDescriptor = library_function(library, half ? @"write_half_words" : @"write_float_words");
				descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
				descriptor.rasterSampleCount = samples;
				descriptor.colorAttachments[0].pixelFormat = describe_surface_color(color_format).format;
				const auto pipeline = compile.render(descriptor);
				surfaces.draw(std::array{cached_surface_use{source, true}}, [&](context& commands, auto attachments)
				{
					auto pass = [MTL4RenderPassDescriptor new];
					pass.colorAttachments[0].texture = attachments[0];
					pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
					pass.colorAttachments[0].storeAction = MTLStoreActionStore;
					raster_state raster;
					raster.viewport = {0, 0, double(width), double(height), 0, 1};
					raster.scissor = {0, 0, width, height};
					render_commands draw(commands, pass);
					draw.configure(pipeline, nil, raster, {}, {});
					draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
					draw.end();
					bindings reads;
					reads.set_texture(0, attachments[0]);
					reads.set_buffer(0, reference);
					const auto table = reads.snapshot(commands);
					const auto encoder = commands.compute();
					[encoder setComputePipelineState:reader];
					[encoder setArgumentTable:table];
					[encoder dispatchThreads:MTLSizeMake(size[0], size[1], 1)
						threadsPerThreadgroup:MTLSizeMake(reader.threadExecutionWidth, 1, 1)];
				});
				const auto contains_word = [&](std::uint32_t wanted)
				{
					const auto* bytes = static_cast<const std::byte*>(reference.contents);
					for (std::size_t i = 0; i < size[0] * size[1]; ++i)
						for (unsigned lane = 0; lane < channels; ++lane)
						{
							std::uint32_t actual = 0;
							std::memcpy(&actual, bytes + (i * 4 + lane) * word_bytes, word_bytes);
							if (actual == wanted) return true;
						}
					return false;
				};
				const auto specials = half ? std::array{0x8000u, 1u, 0x83ffu, 0x7c00u, 0xfc00u, 0x7e55u, 0xfe21u} :
					std::array{0x80000000u, 1u, 0x807fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc04321u};
				require(std::ranges::all_of(specials, contains_word), "Native MSAA writer did not store the special words required by the spill oracle");
				const auto reads = ram->reads;
				const auto version = ram->version;
				surfaces.spill(source);
				for (const auto crop : std::array{clear_rectangle{0, 0, size[0], size[1]}, clear_rectangle{1, 1, 3, 3},
					clear_rectangle{size[0] - 3, size[1] - 1, 3, 1}})
				{
					auto regs = words(code, vertex_texture_dimension::two, true, crop.width, crop.height);
					regs.offset += crop.y * pitch + crop.x * word_bytes * channels;
					regs.control3 = (1u << 20) | pitch;
					const auto image = textures.prepare_versioned_batch(std::array{regs}, surfaces, {}, {},
						texture_upload_path::separate, 1).finish().front().image;
					require(image.texture.sampleCount == samples && image.texture.width == width && image.texture.height == height &&
						image.metadata.samples == samples && surfaces.status(source).spilled && ram->reads == reads && ram->version == version,
						"Float special-value snapshot did not import its immutable native MSAA storage");
					const auto output = make_buffer(gpu, reference.length, buffer_storage::shared);
					gpu.begin();
					bindings reads; reads.set_texture(0, image.texture); reads.set_buffer(0, output);
					const auto table = reads.snapshot(gpu);
					const auto encoder = gpu.compute();
					[encoder setComputePipelineState:reader]; [encoder setArgumentTable:table];
					[encoder dispatchThreads:MTLSizeMake(size[0], size[1], 1) threadsPerThreadgroup:MTLSizeMake(reader.threadExecutionWidth, 1, 1)];
					gpu.submit().wait();
					for (std::size_t y = 0; y < crop.height; ++y)
						for (std::size_t x = 0; x < crop.width; ++x)
						{
							const auto* expected = static_cast<const std::byte*>(reference.contents) +
								((crop.y + y) * size[0] + crop.x + x) * 4 * word_bytes;
							const auto* actual = static_cast<const std::byte*>(output.contents) + ((crop.y + y) * size[0] + crop.x + x) * 4 * word_bytes;
							if (std::memcmp(actual, expected, channels * word_bytes))
								throw std::runtime_error("Float MSAA spill changed a native sample word, format " + std::to_string(code) +
									", samples " + std::to_string(samples) + ", x " + std::to_string(x) + ", y " + std::to_string(y));
							checked_words += channels;
						}
				}
				++checked_profiles;
			}
		std::cout << "Float MSAA spill/import words: " << checked_profiles << " profiles, " << checked_words
			<< " exact words against independent source and imported multisample reads, signed zero, subnormals, infinities and distinct NaN payloads, full and odd crops\n";
		for (const auto color_format : {surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f})
			multisample_window_budget_case(true, color_format);
	}
	void deferred_multisample_snapshot_cases()
	{
		const auto before = cases;
		unsigned crossing_draws = 0, split_plane_draws = 0, guest_background_draws = 0;
		for (const unsigned samples : {2u, 4u})
			for (const unsigned format : {0x85u, 0x90u, 0x91u})
			{
				context gpu{2};
				auto budget = std::make_shared<heap_budget>(gpu, 8u << 20);
				gpu.set_buffer_budget(budget);
				surface_cache surfaces(gpu, ram, 8u << 20, budget);
				guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; },
					1u << 20, budget);
				ram->bytes.assign(0x6000, std::byte{});
				++ram->version;
				const bool depth = format != 0x85;
				const surface_format native = depth ?
					surface_format{format == 0x90 ? surface_depth::z24s8_uint : surface_depth::z24s8_float} : surface_format{surface_color::argb8};
				const auto a = surfaces.acquire({0x3000, {4, 4, samples, native, false}, 32, {}});
				const auto b = surfaces.acquire({0x5000, {4, 4, samples, native, false}, 32, {}});
				const auto target = surfaces.acquire({0x1000, {6, 5, samples, native, false}, 64, {}});
				constexpr std::array values{0x20408031u, 0x90402072u, 0x6080a045u};
				const auto paint = [&](const cached_surface_ref& surface, unsigned i, clear_rectangle area)
				{
					const auto v = values[i];
					surfaces.clear(surface, depth ? 3u : 0xf0u, 0xff000000u | (v >> 8), v, 255, area);
				};
				paint(a, 0, {0, 0, 4, 4});
				paint(a, 1, {2, 0, 2, 4});
				paint(b, 2, {0, 0, 4, 4});
				surfaces.defer_copy(a, {1, 0, 2, 2}, target, 0, 0);
				surfaces.defer_copy(b, {0, 1, 2, 2}, target, 3, 2);
				const unsigned sy = samples / 2;
				const auto crop = [&](unsigned x, unsigned y, bool pixels, unsigned codec)
				{
					auto regs = words(codec, vertex_texture_dimension::two, true, 2, 1);
					regs.offset += y * 64 + x * 4;
					regs.control3 = (1u << 20) | 64u;
					if (pixels) regs.format |= 0x40u << 8;
					return regs;
				};
				std::vector requests{crop(1, 0, false, format), crop(2, 1, true, format), crop(7, 2 * sy + 1, false, format)};
				if (depth) requests.push_back(crop(1, 0, false, 0x85));
				const auto mask = (std::uint64_t{1} << requests.size()) - 1;
				const auto reads = ram->reads;
				const auto version = ram->version;
				const auto gate = [gpu.device() newSharedEvent];
				struct unblock { id<MTLSharedEvent> event; ~unblock() { event.signaledValue = 1; } } release{gate};
				auto old = textures.prepare_versioned_batch(requests, surfaces,
					std::array{guest_texture_upload_dependency{gate, 1}}, {}, texture_upload_path::separate, mask);
				require(!old.ready() && surfaces.snapshot_statistics().entries == 0 &&
					!surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 2,
					"Deferred native MSAA job published or materialized before its dependency");
				gate.signaledValue = 1;
				gpu.wait_idle();
				surfaces.defer_copy(b, {0, 0, 2, 2}, target, 0, 0);
				const auto latest = textures.prepare_versioned_batch(std::span(requests).first(1), surfaces, {}, {},
					texture_upload_path::separate, 1).finish().front();
				const auto frozen = old.finish();
				require(frozen[0].image.texture == frozen[1].image.texture &&
					frozen[0].image.texture != frozen[2].image.texture &&
					frozen[0].image.texture != latest.image.texture &&
					(!depth || frozen[0].image.texture == frozen[3].image.texture),
					"Deferred native MSAA windows did not isolate donors or coalesce matching windows");
				require(surfaces.snapshot_statistics().created == 3 && surfaces.snapshot_statistics().entries == 1,
					"Late deferred MSAA completion displaced the current donor snapshot");
				require(
					!surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 3 &&
					ram->reads == reads && ram->version == version,
					"Covered deferred MSAA windows materialized their destination or read guest pixels");
				auto warm = textures.prepare_versioned_batch(std::span(requests).first(1), surfaces, {}, {}, texture_upload_path::separate, 1);
				require(warm.ready() && warm.finish().front().image.texture == latest.image.texture,
					"Current native MSAA donor snapshot did not remain reusable");
				for (const auto& item : frozen)
					require(item.image.metadata.samples == samples && item.image.texture.width == 4 &&
						item.image.metadata.coordinates.flags == texture_coordinate_msaa_window,
						"Deferred window did not retain the donor's native sample texture");
				const auto expected = [&](unsigned i, bool packed = false, std::optional<unsigned> stencil_owner = {})
				{
					const auto raw = values[i];
					const auto byte = [](unsigned value, unsigned shift) { return float((value >> shift) & 255) / 255; };
					if (!depth)
						return color{byte(raw, 24), byte(raw, 16), byte(raw, 8), 1};
					const float z = float(raw >> 8) / 16777215.f;
					if (!packed) return color{z, z, z, z};
					const unsigned bits = format == 0x90 ? static_cast<unsigned>(z * 16777215.f) : std::bit_cast<unsigned>(z) >> 7;
					return color{byte(bits, 8), byte(bits, 0), byte(values[stencil_owner.value_or(i)], 0), byte(bits, 16)};
				};
				sample(frozen[0].image, requests[0], {.25f, .5f, 0, 1}, expected(0));
				sample(frozen[0].image, requests[0], {.75f, .5f, 0, 1}, expected(1));
				sample(frozen[1].image, requests[1], {.5f, .5f, 0, 1}, expected(1));
				sample(frozen[2].image, requests[2], {.25f, .5f, 0, 1}, expected(2));
				sample(latest.image, requests[0], {.25f, .5f, 0, 1}, expected(2));
				if (depth) sample(frozen[3].image, requests[3], {.25f, .5f, 0, 1}, expected(0, true));
				surfaces.cancel_deferred_copies(target);
				paint(a, 2, {0, 0, 4, 4});
				sample(frozen[0].image, requests[0], {.25f, .5f, 0, 1}, expected(0));
				const auto cancelled = textures.prepare_versioned_batch(std::span(requests).first(1), surfaces, {}, {},
					texture_upload_path::separate, 1).finish().front().image;
				require(cancelled.metadata.samples == 1 && !surfaces.status(target).initialized &&
					surfaces.status(target).deferred_copies == 0, "Cancelled copy retained native MSAA donor authority");
				sample(cancelled, requests[0], {.25f, .5f, 0, 1}, {0, 0, 0, 0});
				paint(a, 0, {0, 0, 4, 4});
				paint(b, 1, {0, 0, 4, 4});
				surfaces.defer_blit_typeless(b, {0, 0, 1, 1}, target, {0, 0, 1, 1});
				surfaces.defer_copy(a, {0, 0, 1, 1}, target, 0, 0);
				surfaces.defer_copy(b, {0, 0, 1, 1}, target, 1, 0);
				const auto crossing_reads = ram->reads;
				const auto crossing_submissions = surfaces.transaction_statistics().submitted;
				const auto crossing_leases = surfaces.memory_statistics().leased_allocations;
				const auto cross_gate = [gpu.device() newSharedEvent];
				unblock cross_release{cross_gate};
				auto crossing_job = textures.prepare_versioned_batch(std::span(requests).first(1), surfaces,
					std::array{guest_texture_upload_dependency{cross_gate, 1}}, {}, texture_upload_path::separate, 1);
				require(!crossing_job.ready() && !surfaces.status(target).initialized &&
					surfaces.status(target).deferred_copies == 3 && ram->reads == crossing_reads &&
					surfaces.transaction_statistics().submitted == crossing_submissions &&
					surfaces.memory_statistics().leased_allocations == crossing_leases,
					"Cross-donor MSAA snapshot materialized the cold destination before its dependency");
				bool timed_out = false;
				try { (void)crossing_job.finish(std::chrono::milliseconds(1)); }
				catch (const submission_wait_timeout&) { timed_out = true; }
				require(timed_out && !crossing_job.ready(), "Cross-donor snapshot did not retain its gated work");
				cross_gate.signaledValue = 1;
				gpu.wait_idle();
				surfaces.defer_copy(b, {1, 0, 1, 1}, target, 0, 0);
				const auto replacement = textures.prepare_versioned_batch(std::span(requests).first(1), surfaces, {}, {},
					texture_upload_path::separate, 1).finish().front().image;
				const auto crossing = crossing_job.finish().front().image;
				require(crossing.texture.sampleCount == 1 && crossing.texture.width == 2 &&
					crossing.metadata.coordinates.flags == texture_coordinate_sample_raster &&
					replacement.texture != crossing.texture && !surfaces.status(target).initialized &&
					surfaces.status(target).deferred_copies == 4 && ram->reads == crossing_reads &&
					surfaces.transaction_statistics().submitted == crossing_submissions &&
					surfaces.memory_statistics().leased_allocations == crossing_leases,
					"Cross-donor MSAA snapshot changed target state or lost its frozen epoch");
				auto crossing_warm = textures.prepare_versioned_batch(std::span(requests).first(1), surfaces, {}, {},
					texture_upload_path::separate, 1);
				require(crossing_warm.ready() && crossing_warm.finish().front().image.texture == replacement.texture &&
					surfaces.snapshot_statistics().entries == 1, "Late composed snapshot displaced or republished over the current epoch");
				sample(crossing, requests[0], {.25f, .5f, 0, 1}, expected(0));
				sample(crossing, requests[0], {.75f, .5f, 0, 1}, expected(1));
				sample(replacement, requests[0], {.25f, .5f, 0, 1}, expected(1));
				crossing_draws += 3;

				surfaces.cancel_deferred_copies(target);
				if (depth)
				{
					// Each plane is supplied by a different donor and overwrite
					// sequence. There is no previous target allocation.
					surfaces.defer_blit_nearest(a, {0, 0, 2, 2}, target, {0, 0, 2, 1}, texture_plane::depth, {true, true, false});
					surfaces.defer_copy(b, {0, 0, 2, 1}, target, 0, 0, texture_plane::stencil);
					surfaces.defer_copy(b, {0, 0, 1, 1}, target, 1, 0, texture_plane::depth);
					surfaces.defer_copy(a, {0, 0, 1, 1}, target, 1, 0, texture_plane::stencil);
					const auto reads_before = ram->reads;
					const auto submits_before = surfaces.transaction_statistics().submitted;
					const auto leases_before = surfaces.memory_statistics().leased_allocations;
					const std::array selected_requests{requests[0], requests[3]};
					const auto split_gate = [gpu.device() newSharedEvent];
					unblock split_release{split_gate};
					auto job = textures.prepare_versioned_batch(selected_requests, surfaces,
						std::array{guest_texture_upload_dependency{split_gate, 1}}, {}, texture_upload_path::separate, 3);
					require(!job.ready() && !surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 4,
						"Independent MSAA planes materialized the cold destination before their dependency");
					split_gate.signaledValue = 1;
					const auto split = job.finish();
					require(split.front().image.texture == split.back().image.texture &&
						split.front().image.metadata.coordinates.flags == texture_coordinate_sample_raster &&
						!surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 4 &&
						ram->reads == reads_before && surfaces.transaction_statistics().submitted == submits_before &&
						surfaces.memory_statistics().leased_allocations == leases_before,
						"Independent MSAA planes lost their shared snapshot or restored the cold target");
					auto warm_split = textures.prepare_versioned_batch(selected_requests, surfaces, {}, {}, texture_upload_path::separate, 3);
					require(warm_split.ready() && warm_split.finish().front().image.texture == split.front().image.texture,
						"Independent MSAA planes missed the current ready snapshot");
					sample(split.front().image, selected_requests[0], {.25f, .5f, 0, 1}, expected(0));
					sample(split.front().image, selected_requests[0], {.75f, .5f, 0, 1}, expected(1));
					sample(split.back().image, selected_requests[1], {.25f, .5f, 0, 1}, expected(0, true, 1));
					sample(split.back().image, selected_requests[1], {.75f, .5f, 0, 1}, expected(1, true, 0));
					split_plane_draws += 4;
					surfaces.cancel_deferred_copies(target);
					surfaces.defer_copy(a, {0, 0, 1, 1}, target, 0, 0, texture_plane::depth);
					surfaces.defer_copy(b, {0, 0, 1, 1}, target, 1, 0, texture_plane::depth);
					const auto missing_stencil_reads = ram->reads;
					const auto guest_gate = [gpu.device() newSharedEvent];
					unblock guest_release{guest_gate};
					auto guest_job = textures.prepare_versioned_batch(selected_requests, surfaces,
						std::array{guest_texture_upload_dependency{guest_gate, 1}}, {}, texture_upload_path::separate, 3);
					bool guest_timeout = false;
					try { (void)guest_job.finish(std::chrono::milliseconds(1)); }
					catch (const submission_wait_timeout&) { guest_timeout = true; }
					require(guest_timeout && !guest_job.ready() && !surfaces.status(target).initialized &&
						surfaces.status(target).deferred_copies == 2 && ram->reads == missing_stencil_reads + 1,
						"Guest MSAA background was not retained across the event timeout");
					guest_gate.signaledValue = 1;
					gpu.wait_idle();
					surfaces.defer_copy(b, {0, 0, 2, 1}, target, 0, 0, texture_plane::stencil);
					const auto latest_guest = textures.prepare_versioned_batch(selected_requests, surfaces, {}, {},
						texture_upload_path::separate, 3).finish();
					const auto fallback = guest_job.finish();
					require(!surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 3 &&
						ram->reads == missing_stencil_reads + 1 && fallback.front().image.texture == fallback.back().image.texture,
						"Missing MSAA stencil background materialized the target or lost its shared snapshot");
					auto warm_guest = textures.prepare_versioned_batch(selected_requests, surfaces, {}, {}, texture_upload_path::separate, 3);
					require(warm_guest.ready() && warm_guest.finish().front().image.texture == latest_guest.front().image.texture &&
						latest_guest.front().image.texture != fallback.front().image.texture,
						"Late guest-background snapshot displaced the newer stencil epoch");
					sample(fallback.front().image, selected_requests[0], {.25f, .5f, 0, 1}, expected(0));
					sample(fallback.front().image, selected_requests[0], {.75f, .5f, 0, 1}, expected(1));
					auto left = expected(0, true), right = expected(1, true);
					left[2] = right[2] = 0; // Stencil came from the untouched guest background.
					sample(fallback.back().image, selected_requests[1], {.25f, .5f, 0, 1}, left);
					sample(fallback.back().image, selected_requests[1], {.75f, .5f, 0, 1}, right);
					sample(latest_guest.back().image, selected_requests[1], {.25f, .5f, 0, 1}, expected(0, true, 1));
					sample(latest_guest.back().image, selected_requests[1], {.75f, .5f, 0, 1}, expected(1, true, 1));
					guest_background_draws += 6;
				}
				paint(target, 2, {0, 0, 6, 5});
				for (const bool spill : {false, true})
				{
					if (spill)
					{
						surfaces.cancel_deferred_copies(target);
						surfaces.spill(target);
					}
					const unsigned first_row = spill ? 3 : 0, y_origin = spill ? first_row * sy - 1 : 0;
					surfaces.defer_copy(a, {0, 0, 2, 2}, target, 0, first_row);
					surfaces.defer_copy(b, {1, 1, 2, 1}, target, 1, first_row + 1);
					if (depth)
					{
						surfaces.defer_copy(b, {2, 2, 1, 1}, target, 0, first_row, texture_plane::stencil);
						surfaces.defer_copy(a, {1, 1, 1, 1}, target, 1, first_row + 1, texture_plane::depth);
					}
					auto overlap = crop(1, y_origin, false, format);
					const unsigned height = sy * 2 + 1;
					overlap.image_rect = (6u << 16) | height;
					auto packed = overlap;
					packed.format = (packed.format & ~(0x9fu << 8)) | (0x85u << 8);
					const auto pending_background = surfaces.status(target);
					const auto background_reads = ram->reads;
					const auto background_submissions = surfaces.transaction_statistics().submitted;
					const std::array background_requests{overlap, packed};
					const auto background = textures.prepare_versioned_batch(std::span(background_requests).first(depth ? 2 : 1),
						surfaces, {}, {}, texture_upload_path::separate, depth ? 3 : 1).finish();
					const auto after = surfaces.status(target);
					require(background.front().image.texture.width == 6 && background.front().image.texture.height == height &&
						background.front().image.metadata.coordinates.flags == texture_coordinate_sample_raster &&
						(!depth || background.front().image.texture == background.back().image.texture) &&
						after.spilled == spill && after.deferred_copies == (depth ? 4 : 2) && after.content_generation == pending_background.content_generation &&
						ram->reads == background_reads && surfaces.transaction_statistics().submitted == background_submissions,
						"Overlapping MSAA sections materialized the destination or lost its native background");
					for (unsigned y = 0; y < height; ++y)
						for (unsigned x = 0; x < 6; ++x)
						{
							const unsigned cell_x = (x + 1) / 2, cell_y = (y + y_origin) / sy;
							const unsigned i = cell_y == first_row + 1 && cell_x >= 1 && cell_x < 3 ? 1 :
								cell_y >= first_row && cell_y < first_row + 2 && cell_x < 2 ? 0 : 2;
							const unsigned z = depth && cell_x == 1 && cell_y == first_row + 1 ? 0 : i;
							const unsigned s = depth && cell_x == 0 && cell_y == first_row ? 1 : i;
							const color uv{(x + .5f) / 6, (y + .5f) / height, 0, 1};
							sample(background.front().image, overlap, uv, expected(z));
							if (depth) sample(background.back().image, packed, uv, expected(z, true, s));
							crossing_draws += depth ? 2 : 1;
						}
				}
			}
		std::cout << cases - before - crossing_draws - split_plane_draws - guest_background_draws << " deferred native MSAA draws: separate donors in one epoch, shared windows, gate, replacement, old completion and cancellation\n";
		std::cout << crossing_draws << " composed MSAA raster draws: cold/overlapping windows, resident and spilled backgrounds, hidden incompatible copy, timeout and late replacement without materialization or guest reads\n";
		std::cout << split_plane_draws << " independent depth/stencil MSAA draws: cold targets with no allocation, mixed projection and plane order, packed and depth bindings share one snapshot\n";
		std::cout << guest_background_draws << " missing-plane MSAA draws: guest stencil captured once into the shared snapshot, event timeout, late completion and newer stencil epoch without target materialization\n";
	}
	void deferred_guest_multisample_cases()
	{
		const std::array formats{
			surface_format{surface_color::x1rgb5_zero}, surface_format{surface_color::x1rgb5_one},
			surface_format{surface_color::rgb565}, surface_format{surface_color::xrgb8_zero}, surface_format{surface_color::xrgb8_one},
			surface_format{surface_color::argb8}, surface_format{surface_color::b8}, surface_format{surface_color::gb8},
			surface_format{surface_color::rgba16f}, surface_format{surface_color::rgba32f}, surface_format{surface_color::r32f},
			surface_format{surface_color::xbgr8_zero}, surface_format{surface_color::xbgr8_one}, surface_format{surface_color::abgr8},
			surface_format{surface_depth::z16_uint}, surface_format{surface_depth::z16_float},
			surface_format{surface_depth::z24s8_uint}, surface_format{surface_depth::z24s8_float}};
		unsigned windows = 0, profiles = 0;
		std::size_t checked_bytes = 0;
		for (const auto& format : formats)
			for (const unsigned samples : {2u, 4u})
				for (unsigned storage = 0; storage < 5; ++storage)
				{
					context gpu{1};
					auto budget = std::make_shared<heap_budget>(gpu, 8u << 20);
					gpu.set_buffer_budget(budget);
					surface_cache surfaces(gpu, ram, 8u << 20, budget);
					const auto* color_format = std::get_if<surface_color>(&format);
					const bool depth = !color_format;
					const auto metadata = depth ? describe_surface_depth(std::get<surface_depth>(format)) : surface_depth_descriptor{};
					const auto color_metadata = depth ? surface_color_layout{} : color_layout(*color_format);
					const auto bytes = depth ? metadata.guest_bytes : color_metadata.pixel_bytes;
					const auto native_bytes = depth ? metadata.depth_bytes : bytes;
					const auto size = sample_expanded_size(6, 5, samples);
					const unsigned width = size[0], height = size[1];
					const bool banked = storage >= 3, morton = storage == 2 || storage == 4;
					cached_surface_description descriptor{banked ? 0x10400u + 2 * bytes : 0x1000u,
						{6, 5, samples, format, false}, morton ? 0u : banked ? 512u : width * bytes + 3 * bytes + unsigned(storage == 1), {}};
					if (morton) descriptor.morton = swizzle_region{{32, 16, banked ? 1u : 2u}, {3, 2, banked ? 0u : 1u}, {width, height, 1}};
					if (banked) descriptor.tile = banked_tile_layout{0x10000, 0x20000, 512, 3, 1024 + 2 * bytes};
					ram->bytes.assign(0x30000, std::byte{0xda});
					++ram->version;
					std::vector<std::byte> guest(width * height * bytes);
					for (unsigned y = 0; y < height; ++y)
						for (unsigned x = 0; x < width; ++x)
							for (unsigned c = 0; c < bytes; ++c)
								guest[(y * width + x) * bytes + c] = std::byte(0x30 + ((x * 3 + y * 7 + c * 5) & 15));
					// Preserve special float words in guest holes without passing
					// them through a render shader or a normalized MSAA import.
					if (color_format && (*color_format == surface_color::rgba16f || *color_format == surface_color::rgba32f || *color_format == surface_color::r32f))
					{
						const unsigned element = color_metadata.endian_element_bytes;
						const auto special = element == 2 ? std::array{0x8000u, 1u, 0x83ffu, 0x7c00u, 0xfc00u, 0x7e55u, 0xfe21u} :
							std::array{0x80000000u, 1u, 0x807fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u, 0xffc04321u};
						for (unsigned i = 0; i < guest.size() / element; ++i)
							for (unsigned c = 0; c < element; ++c)
								guest[i * element + c] = std::byte(special[i % special.size()] >> (8 * (element - c - 1)));
					}
					auto layout = plan_guest_surface(descriptor.address, width, height, bytes, descriptor.row_pitch, descriptor.morton);
					if (banked)
					{
						layout = attach_banked_tile(layout, *descriptor.tile);
						std::vector<std::byte> tile(descriptor.tile->size, std::byte{0xda});
						for (unsigned y = 0; y < height; ++y)
							for (unsigned x = 0; x < width; ++x)
								for (unsigned c = 0; c < bytes; ++c)
								{
									const auto& bank = *descriptor.tile;
									const auto logical = bank.offset + (morton ? morton_index({x + 3, y + 2, 0}, plan_swizzle(*descriptor.morton, bytes).log2) * bytes : y * bank.pitch + x * bytes) + c;
									std::array<char, 4> word{};
									const rsx::detiler_config config{0, banked_tile_factor(bank.pitch), bank.pitch / 256, bank.address, bank.size,
										static_cast<std::uint32_t>(logical & ~std::size_t(3)), 0, bank.pitch, bank.bank, 1, 1, 4, 4};
									rsx::tiled_dma_copy(0, 0, config, reinterpret_cast<char*>(tile.data()), word.data(), 1);
									word[logical & 3] = std::to_integer<char>(guest[(y * width + x) * bytes + c]);
									rsx::tiled_dma_copy(0, 0, config, reinterpret_cast<char*>(tile.data()), word.data(), 0);
								}
						std::copy(tile.begin(), tile.end(), ram->bytes.begin() + descriptor.tile->address - 0x1000);
					}
					else
						for (unsigned y = 0; y < height; ++y)
							for (unsigned x = 0; x < width; ++x)
								for (unsigned c = 0; c < bytes; ++c)
								{
									const auto offset = morton ? morton_index({x + 3, y + 2, 1}, plan_swizzle(*descriptor.morton, bytes).log2) * bytes : y * descriptor.row_pitch + x * bytes;
									ram->bytes[offset + c] = guest[(y * width + x) * bytes + c];
								}
					const auto original = ram->bytes;
					const auto target = surfaces.acquire(descriptor);
					std::array<cached_surface_ref, 2> donors;
					std::array<id<MTLBuffer>, 2> donor_words, donor_stencil;
					for (unsigned i = 0; i < donors.size(); ++i)
					{
						donors[i] = surfaces.acquire({0x5000u + i * 0x1000, {3, 3, 1, format, false}, 3 * bytes, {}});
						surfaces.prepare(donors[i], surface_initialization::background);
						surfaces.clear(donors[i], depth ? 3u : 0xf0u, i ? 0xffa04020u : 0xff204080u,
							i ? 0x40800052u : 0x30800031u, 255, {0, 0, 3, 3});
						donor_words[i] = make_buffer(gpu, native_bytes, buffer_storage::shared);
						if (metadata.stencil) donor_stencil[i] = make_buffer(gpu, 1, buffer_storage::shared);
						surfaces.draw(std::array{cached_surface_use{donors[i], false}}, [&](context& commands, auto images)
						{
							texture_region area; area.size = MTLSizeMake(1, 1, 1);
							texture_to_buffer(commands, images[0], area, donor_words[i], 0, native_bytes, 0, depth ? texture_plane::depth : texture_plane::all);
							if (metadata.stencil) texture_to_buffer(commands, images[0], area, donor_stencil[i], 0, 1, 0, texture_plane::stencil);
						});
					}
					for (unsigned mode = 0; mode < (metadata.stencil ? 4u : 1u); ++mode)
					{
						const auto first_plane = mode == 0 ? texture_plane::all : mode == 2 ? texture_plane::stencil : texture_plane::depth;
						const auto second_plane = mode == 3 ? texture_plane::stencil : first_plane;
						surfaces.defer_copy(donors[0], {0, 0, 2, 3}, target, 2, 2, first_plane);
						surfaces.defer_copy(donors[1], {0, 0, 2, 1}, target, 1, 3, second_plane);
						for (const auto crop : std::array{clear_rectangle{0, 0, width, height}, clear_rectangle{1, 1, width - 2, height - 2}, clear_rectangle{1, height - 1, width - 1, 1}})
						{
							auto reads = ram->reads;
							const auto submitted = surfaces.transaction_statistics().submitted;
							const auto leases = surfaces.memory_statistics().leased_allocations;
							const auto prepare = [&]
							{
								return surfaces.snapshot_texture_mapping(std::array{banked ? guest_byte_coverage(layout.range) : layout}, [&](const auto&) -> std::optional<surface_texture_region>
									{ return surface_texture_region{crop, 0, true}; }, budget);
							};
							if (windows == 0)
							{
								@autoreleasepool
								{
									const auto before = budget->statistics();
									const auto blocker = budget->shared_buffer(((before.budget - before.reserved_bytes) / 16384) * 16384);
									std::vector<id<MTLBuffer>> remaining;
									while (const auto tail = budget->try_shared_buffer(1)) remaining.push_back(tail);
									bool refused = false;
									try { (void)prepare(); } catch (const heap_budget_error&) { refused = true; }
									require(blocker && refused && !surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 2 &&
										surfaces.snapshot_statistics().entries == 0 && surfaces.memory_statistics().leased_allocations == leases,
										"Guest-hole MSAA pressure refusal changed the target authority or published partial work");
								}
								ram->short_read = true;
								bool short_refused = false;
								try { (void)prepare(); }
								catch (const std::runtime_error& error) { short_refused = std::string_view(error.what()) == "VM surface snapshot has the wrong byte count"; }
								ram->short_read = false;
								require(short_refused && !surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 2 &&
									surfaces.snapshot_statistics().entries == 0 && surfaces.memory_statistics().leased_allocations == leases && ram->bytes == original,
									"Short guest-hole capture changed authority or published incomplete pixels");
								reads = ram->reads;
							}
							const auto snapshot = prepare();
							require(snapshot && snapshot.sampleCount == 1 && snapshot.width == crop.width && snapshot.height == crop.height && ram->reads == reads + 1,
								"Deferred guest-hole snapshot did not capture exactly one guest background");
							require(!surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 2 &&
								surfaces.memory_statistics().leased_allocations == leases && surfaces.transaction_statistics().submitted == submitted + 1 && ram->bytes == original,
								"Deferred guest-hole snapshot mutated the target authority or guest bytes");
							const auto warm = prepare();
							require(warm == snapshot && ram->reads == reads + 1 && surfaces.transaction_statistics().submitted == submitted + 1,
								"Deferred guest-hole snapshot missed the ready cache or recaptured VM");
							const auto row = crop.width * native_bytes;
							const auto packed = make_buffer(gpu, row * crop.height, buffer_storage::shared);
							const auto stencil = metadata.stencil ? make_buffer(gpu, crop.width * crop.height, buffer_storage::shared) : nil;
							texture_region area; area.size = MTLSizeMake(crop.width, crop.height, 1);
							gpu.begin();
							texture_to_buffer(gpu, snapshot, area, packed, 0, row, 0, depth ? texture_plane::depth : texture_plane::all);
							if (metadata.stencil) texture_to_buffer(gpu, snapshot, area, stencil, 0, crop.width, 0, texture_plane::stencil);
							gpu.submit().wait();
							for (unsigned y = 0; y < crop.height; ++y)
								for (unsigned x = 0; x < crop.width; ++x)
								{
									const unsigned gx = x + crop.x, gy = y + crop.y, nx = gx / 2, ny = gy / (samples / 2);
									const auto owner = [&](texture_plane selected) -> std::optional<unsigned>
									{
										if (nx >= 1 && nx < 3 && ny == 3 && (second_plane == texture_plane::all || second_plane == selected)) return 1;
										if (nx >= 2 && nx < 4 && ny >= 2 && (first_plane == texture_plane::all || first_plane == selected)) return 0;
										return {};
									};
									std::array<std::byte, 16> expected{};
									if (const auto input = owner(depth ? texture_plane::depth : texture_plane::all))
										std::memcpy(expected.data(), donor_words[*input].contents, native_bytes);
									else if (depth)
									{
										unsigned raw = 0;
										for (unsigned c = 0; c < bytes; ++c) raw = raw * 256 + std::to_integer<unsigned>(guest[(gy * width + gx) * bytes + c]);
										const auto type = std::get<surface_depth>(format);
										const unsigned bits = type == surface_depth::z16_uint ? raw : type == surface_depth::z16_float ?
											(raw << 11) + (120u << 23) : type == surface_depth::z24s8_float ? (raw >> 8) << 7 : std::bit_cast<unsigned>(float(raw >> 8) / 16777215.f);
										std::memcpy(expected.data(), &bits, native_bytes);
									}
									else
										for (unsigned c = 0; c < bytes; ++c)
											expected[c] = guest[(gy * width + gx) * bytes + (c ^ (color_metadata.endian_element_bytes - 1))];
									if (std::memcmp(expected.data(), static_cast<const std::byte*>(packed.contents) + y * row + x * native_bytes, native_bytes))
										throw std::runtime_error("Deferred guest-hole raw word mismatch: format " + std::to_string(profiles) + ", storage " + std::to_string(storage) + ", mode " + std::to_string(mode) + ", x " + std::to_string(gx) + ", y " + std::to_string(gy));
									if (metadata.stencil)
									{
										const auto input = owner(texture_plane::stencil);
										const auto expected_stencil = input ? *static_cast<const std::byte*>(donor_stencil[*input].contents) : guest[(gy * width + gx) * bytes + 3];
										require(static_cast<const std::byte*>(stencil.contents)[y * crop.width + x] == expected_stencil,
											"Deferred guest-hole stencil overwrite order differs from its independent oracle");
									}
									checked_bytes += native_bytes + unsigned(metadata.stencil);
								}
							++windows;
						}
						surfaces.cancel_deferred_copies(target);
						++profiles;
					}
				}
		std::cout << "Deferred guest-hole MSAA: " << profiles << " format/storage/aspect profiles, " << windows << " windows, " << checked_bytes
			<< " exact native bytes; all 18 formats, 2/4 samples, aligned/odd pitches, Morton volume slice and banked linear/Morton, overlaps, final-row crops, float special words, independent depth/stencil, ready-cache, pressure refusal/retry, short-read refusal and unchanged target/VM passed\n";
	}
	void projected_deferred_multisample_cases()
	{
		const auto writer = compile.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 project_vertex(uint i [[vertex_id]]) {
    const float2 p[] = {float2(-1,-1), float2(3,-1), float2(-1,3)};
    return float4(p[i], 0, 1);
}
float4 project_color(float4 p, uint s) {
    return float4(uint(p.x) + 1, uint(p.y) + 2, s + 3, 64) / 64.0f;
}
struct depth_word { float z [[depth(any)]]; uint stencil [[stencil]]; };
depth_word project_depth(float4 p, uint s) {
    return {0.25f + float(13 * uint(p.x) + 7 * uint(p.y) + s) / 512.0f,
        20 + 3 * uint(p.x) + 11 * uint(p.y) + 37 * s};
}
fragment float4 write_color(float4 p [[position]], uint s [[sample_id]]) { return project_color(p, s); }
fragment float4 write_color_single(float4 p [[position]]) { return project_color(p, 0); }
fragment depth_word write_depth(float4 p [[position]], uint s [[sample_id]]) { return project_depth(p, s); }
fragment depth_word write_depth_single(float4 p [[position]]) { return project_depth(p, 0); }
)msl", @"Independent source for projected deferred MSAA");
		auto ds = [MTLDepthStencilDescriptor new];
		ds.depthCompareFunction = MTLCompareFunctionAlways;
		ds.depthWriteEnabled = YES;
		auto stencil = [MTLStencilDescriptor new];
		stencil.stencilCompareFunction = MTLCompareFunctionAlways;
		stencil.depthStencilPassOperation = MTLStencilOperationReplace;
		ds.frontFaceStencil = ds.backFaceStencil = stencil;
		const auto depth_state = [ctx.device() newDepthStencilStateWithDescriptor:ds];
		struct profile { clear_rectangle source, destination; std::array<bool, 3> flip; bool overlap = false; };
		const std::array profiles{profile{{1, 1, 3, 2}, {1, 1, 4, 3}, {true, false, false}},
			profile{{0, 0, 5, 4}, {1, 0, 3, 2}, {false, true, false}},
			profile{{1, 1, 3, 2}, {0, 2, 3, 2}, {true, true, false}, true}};
		const auto axis = [](unsigned i, unsigned source, unsigned destination, bool flip)
		{
			const auto numerator = (2 * i + 1) * source, divisor = 2 * destination;
			return flip ? (2 * source * destination - numerator) / divisor : numerator / divisor;
		};
		std::size_t pixels = 0;
		unsigned windows = 0;
		for (const unsigned output_samples : {2u, 4u})
			for (const unsigned input_samples : {1u, 2u, 4u})
				for (const auto [code, format] : std::array{
					std::pair{0x85u, surface_format{surface_color::argb8}}, std::pair{0x9au, surface_format{surface_color::rgba16f}},
					std::pair{0x9bu, surface_format{surface_color::rgba32f}}, std::pair{0x9cu, surface_format{surface_color::r32f}},
					std::pair{0x90u, surface_format{surface_depth::z24s8_uint}},
					std::pair{0x91u, surface_format{surface_depth::z24s8_float}}})
				{
					const bool depth = code == 0x90 || code == 0x91, single = input_samples == 1;
					const auto in = single ? sample_layout{1, 1} : rsx_sample_layout(input_samples);
					const auto out = rsx_sample_layout(output_samples);
					const auto bytes = describe_guest_texture_format(code).block.bytes;
					const auto native = depth ? MTLPixelFormatDepth32Float_Stencil8 : describe_surface_color(std::get<surface_color>(format)).format;
					context gpu{1};
					auto budget = std::make_shared<heap_budget>(gpu, 8u << 20);
					gpu.set_buffer_budget(budget);
					surface_cache surfaces(gpu, ram, 8u << 20, budget);
					guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; },
						1u << 20, budget);
					ram->bytes.assign(0x7000, std::byte{});
					++ram->version;
					const auto source = surfaces.acquire({0x4000, {5, 4, input_samples, format, false}, 256, {}});
					const auto target = surfaces.acquire({0x1000, {6, 5, output_samples, format, false}, 256, {}});
					auto descriptor = [MTL4RenderPipelineDescriptor new];
					descriptor.vertexFunctionDescriptor = library_function(writer, @"project_vertex");
					descriptor.fragmentFunctionDescriptor = library_function(writer, depth ?
						(single ? @"write_depth_single" : @"write_depth") : (single ? @"write_color_single" : @"write_color"));
					descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
					descriptor.rasterSampleCount = input_samples;
					if (!depth) descriptor.colorAttachments[0].pixelFormat = native;
					const auto pipeline = compile.render(descriptor);
					surfaces.draw(std::array{cached_surface_use{source, true}}, [&](context& commands, auto attachments)
					{
						auto pass = [MTL4RenderPassDescriptor new];
						if (depth)
						{
							pass.depthAttachment.texture = pass.stencilAttachment.texture = attachments[0];
							pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = MTLLoadActionDontCare;
							pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
						}
						else
						{
							pass.colorAttachments[0].texture = attachments[0];
							pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
							pass.colorAttachments[0].storeAction = MTLStoreActionStore;
						}
						raster_state raster;
						raster.viewport = {0, 0, 5, 4, 0, 1};
						raster.scissor = {0, 0, 5, 4};
						render_commands draw(commands, pass);
						draw.configure(pipeline, depth ? depth_state : nil, raster, {}, {});
						draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
						draw.end();
					});
					for (const auto& [sr, dr, flip, overlap] : profiles)
					{
						surfaces.defer_blit_nearest(source, sr, target, dr, texture_plane::all, flip);
						if (overlap)
						{
							surfaces.defer_copy(source, {4, 3, 1, 1}, target, dr.x + 1, dr.y + 1,
								depth ? texture_plane::depth : texture_plane::all);
							if (depth) surfaces.defer_copy(source, {0, 0, 1, 1}, target, dr.x + 1, dr.y + 1, texture_plane::stencil);
						}
						const unsigned width = dr.width * out.x - 1, height = dr.height * out.y - 1;
						auto request = words(code, vertex_texture_dimension::two, true, width, height);
						request.offset += (dr.x * out.x + 1) * bytes + (dr.y * out.y + 1) * 256;
						request.control3 = (1u << 20) | 256u;
						const auto reads = ram->reads;
						const auto submitted = surfaces.transaction_statistics().submitted;
						const auto lease_count = surfaces.memory_statistics().leased_allocations;
						const auto image = textures.prepare_versioned_batch(std::array{request}, surfaces, {}, {},
							texture_upload_path::separate, 1).finish().front().image;
						require(image.metadata.coordinates.flags == texture_coordinate_sample_raster &&
							image.texture.width == width && image.texture.height == height &&
							!surfaces.status(target).initialized &&
							surfaces.status(target).deferred_copies == 1 + unsigned(overlap) * (depth ? 2 : 1) &&
							ram->reads == reads && surfaces.transaction_statistics().submitted == submitted &&
							surfaces.memory_statistics().leased_allocations == lease_count,
							"Projected MSAA snapshot materialized its destination or read guest pixels");
						auto warm = textures.prepare_versioned_batch(std::array{request}, surfaces, {}, {}, texture_upload_path::separate, 1);
						require(warm.ready() && warm.finish().front().image.texture == image.texture, "Projected MSAA window missed its ready snapshot");
						const auto row = width * bytes;
						const auto packed = make_buffer(gpu, row * height, buffer_storage::shared);
						const auto stencil_words = depth ? make_buffer(gpu, width * height, buffer_storage::shared) : nil;
						texture_region area;
						area.size = MTLSizeMake(width, height, 1);
						gpu.begin();
						texture_to_buffer(gpu, image.texture, area, packed, 0, row, 0, depth ? texture_plane::depth : texture_plane::all);
						if (depth) texture_to_buffer(gpu, image.texture, area, stencil_words, 0, width, 0, texture_plane::stencil);
						gpu.submit().wait();
						for (unsigned y = 0; y < height; ++y)
							for (unsigned x = 0; x < width; ++x)
							{
								const bool patch = overlap && (x + 1) / out.x == 1 && (y + 1) / out.y == 1;
								const auto sx = patch ? 4 * in.x + axis((x + 1) % out.x, in.x, out.x, false) :
									sr.x * in.x + axis(x + 1, sr.width * in.x, dr.width * out.x, flip[0]);
								const auto sy = patch ? 3 * in.y + axis((y + 1) % out.y, in.y, out.y, false) :
									sr.y * in.y + axis(y + 1, sr.height * in.y, dr.height * out.y, flip[1]);
								const auto px = sx / in.x, py = sy / in.y, sample = sx % in.x + in.x * (sy % in.y);
								const auto* actual = static_cast<const std::byte*>(packed.contents) + y * row + x * bytes;
								if (depth)
								{
									const float z = .25f + float(13 * px + 7 * py + sample) / 512;
									const unsigned s = patch ? 20 + 37 * sample : 20 + 3 * px + 11 * py + 37 * sample;
									require(std::memcmp(actual, &z, bytes) == 0 &&
										static_cast<const unsigned char*>(stencil_words.contents)[y * width + x] == s,
										"Projected depth/stencil used the wrong source texel or sample");
								}
								else
								{
									const color rgba{float(px + 1) / 64, float(py + 2) / 64, float(sample + 3) / 64, 1};
									const std::array bgra{static_cast<unsigned char>(std::lround(rgba[2] * 255)),
										static_cast<unsigned char>(std::lround(rgba[1] * 255)), static_cast<unsigned char>(std::lround(rgba[0] * 255)),
										static_cast<unsigned char>(255)};
									const std::array half{_Float16(rgba[0]), _Float16(rgba[1]), _Float16(rgba[2]), _Float16(rgba[3])};
									const void* expected = code == 0x85 ? static_cast<const void*>(bgra.data()) :
										code == 0x9a ? static_cast<const void*>(half.data()) : static_cast<const void*>(rgba.data());
									require(std::memcmp(actual, expected, bytes) == 0, "Projected color used the wrong source texel or sample");
								}
								++pixels;
							}
						surfaces.cancel_deferred_copies(target);
						++windows;
					}
				}
		std::cout << "Projected deferred MSAA: " << windows << " windows, " << pixels
			<< " exact color/depth/stencil texels; 1/2/4 source samples, 2/4 destination samples, odd crops, scaling, X/Y flips and newer overlapping copy, no materialization or guest reads\n";
	}
	void deferred_typeless_multisample_cases()
	{
		context gpu{1};
		auto budget = std::make_shared<heap_budget>(gpu, 32u << 20);
		gpu.set_buffer_budget(budget);
		surface_color_transfer colors(gpu);
		surface_depth_transfer depths(gpu);
		sample_transfer sample_copy(gpu);
		const std::array<surface_format, 6> formats{surface_color::rgb565, surface_color::argb8,
			surface_color::rgba16f, surface_color::rgba32f, surface_depth::z24s8_uint, surface_depth::z24s8_float};
		const std::array destinations{
			std::pair{surface_format{surface_color::argb8}, 0x85u}, std::pair{surface_format{surface_color::gb8}, 0x8bu},
			std::pair{surface_format{surface_color::rgba16f}, 0x9au}, std::pair{surface_format{surface_color::rgba32f}, 0x9bu},
			std::pair{surface_format{surface_depth::z24s8_uint}, 0x90u}, std::pair{surface_format{surface_depth::z24s8_float}, 0x91u}};
		const auto axis = [](unsigned i, unsigned from, unsigned to, bool flip)
		{
			const auto numerator = (2 * i + 1) * from, denominator = 2 * to;
			return flip ? (from * denominator - numerator) / denominator : numerator / denominator;
		};
		unsigned windows = 0;
		std::size_t bytes_checked = 0;
		for (const auto& sf : formats)
			for (const auto& [df, code] : destinations)
				for (const unsigned input_samples : {1u, 2u, 4u})
					for (const unsigned output_samples : {2u, 4u})
						for (const auto flip : {std::array{true, false, false}, std::array{false, true, false}})
						{
							@autoreleasepool
							{
								const auto in = input_samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(input_samples);
								const auto out = rsx_sample_layout(output_samples);
								const auto sb = surface_guest_bytes(sf), db = surface_guest_bytes(df);
								const unsigned expanded_width = 64 / sb, expanded_height = 8;
								const clear_rectangle sr{1, 1, 32 / (in.x * sb), 2}, dr{1, 1, 4, 3};
								surface_cache surfaces(gpu, ram, 16u << 20, budget);
								guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; },
									4u << 20, budget);
								ram->bytes.assign(0x9000, std::byte{0xe7});
								++ram->version;
								const auto source = surfaces.acquire({0x4000, {expanded_width / in.x, expanded_height / in.y, input_samples, sf, false}, 128, {}});
								const auto target = surfaces.acquire({0x1000, {6, 5, output_samples, df, false}, 256, {}});
								surfaces.prepare(source, surface_initialization::background);
								std::vector<std::byte> original(64 * expanded_height);
								for (unsigned y = 0; y < expanded_height; ++y)
									for (unsigned x = 0; x < 64; x += 2)
									{
										original[y * 64 + x] = std::byte(0x34 + ((x / 2 + y) % 3) * 2);
										original[y * 64 + x + 1] = std::byte((x / 2 + y) & 7);
									}
								const auto input = make_upload(gpu, original);
								surfaces.draw(std::array{cached_surface_use{source, true}}, [&](context& commands, auto images)
								{
									id<MTLTexture> expanded = images[0];
									if (input_samples > 1)
									{
										auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[0].pixelFormat
											width:expanded_width height:expanded_height mipmapped:NO];
										descriptor.storageMode = MTLStorageModePrivate;
										descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
										expanded = make_texture(commands, descriptor);
									}
									texture_region area;
									area.size = MTLSizeMake(expanded_width, expanded_height, 1);
									if (const auto color = std::get_if<surface_color>(&sf)) colors.upload(commands, *color, input, {0, 64}, expanded, area);
									else depths.upload(commands, std::get<surface_depth>(sf), input, {0, 64}, expanded, area);
									if (input_samples > 1) sample_copy.import(commands, expanded, images[0]);
								});
								surfaces.defer_blit_typeless(source, sr, target, dr, flip);
								const bool split = std::holds_alternative<surface_depth>(df);
								if (split)
								{
									const auto stencil = surfaces.acquire({0x6000, {6, 5, output_samples, df, false}, 256, {}});
									surfaces.clear(stencil, 3, 0, 0xffffff99, 255, {0, 0, 6, 5});
									surfaces.defer_copy(stencil, {0, 0, 1, 1}, target, dr.x + 1, dr.y + 1, texture_plane::stencil);
								}
								const unsigned width = dr.width * out.x - 1, height = dr.height * out.y - 1;
								auto regs = words(code, vertex_texture_dimension::two, true, width, height);
								regs.offset += (dr.x * out.x + 1) * db + (dr.y * out.y + 1) * 256;
								regs.control3 = (1u << 20) | 256u;
								const auto reads = ram->reads;
								const auto submitted = surfaces.transaction_statistics().submitted;
								const auto leases = surfaces.memory_statistics().leased_allocations;
								if (windows == 0)
								{
									@autoreleasepool
									{
										const auto before = budget->statistics();
										const auto block_bytes = ((before.budget - before.reserved_bytes) / 16384) * 16384;
										const auto blocker = budget->shared_buffer(block_bytes);
										std::vector<id<MTLBuffer>> remaining_space;
										while (const auto remainder = budget->try_shared_buffer(1))
											remaining_space.push_back(remainder);
										bool refused = false;
										try { (void)textures.prepare_versioned_batch(std::array{regs}, surfaces, {}, {}, texture_upload_path::separate, 1).finish(); }
										catch (const heap_budget_error&) { refused = true; }
										require(blocker && refused && surfaces.snapshot_statistics().entries == 0 &&
											!surfaces.status(target).initialized && surfaces.status(target).deferred_copies == 1 && ram->reads == reads,
											"Typeless MSAA ledger-pressure refusal changed authority or published a partial result");
									}
								}
								const auto gate = windows == 0 ? [gpu.device() newSharedEvent] : nil;
								struct release { id<MTLSharedEvent> event; ~release() { if (event) event.signaledValue = 1; } } unblock{gate};
								const std::array dependency{guest_texture_upload_dependency{gate, 1}};
								auto job = textures.prepare_versioned_batch(std::array{regs}, surfaces,
									gate ? std::span<const guest_texture_upload_dependency>{dependency} : std::span<const guest_texture_upload_dependency>{},
									{}, texture_upload_path::separate, 1);
								if (gate)
								{
									bool timed_out = false;
									try { (void)job.finish(std::chrono::milliseconds(1)); }
									catch (const submission_wait_timeout&) { timed_out = true; }
									require(timed_out && !job.ready() && surfaces.snapshot_statistics().entries == 0 &&
										!surfaces.status(target).initialized, "Typeless MSAA timeout lost its sources or published before completion");
									gate.signaledValue = 1;
								}
								const auto image = job.finish().front().image;
								require(image.metadata.coordinates.flags == texture_coordinate_sample_raster &&
									image.texture.width == width && image.texture.height == height && !surfaces.status(target).initialized &&
									surfaces.status(target).deferred_copies == (split ? 2 : 1) && ram->reads == reads &&
									surfaces.transaction_statistics().submitted == submitted && surfaces.memory_statistics().leased_allocations == leases,
									"Deferred typeless MSAA window materialized its destination or read guest pixels");
								auto warm = textures.prepare_versioned_batch(std::array{regs}, surfaces, {}, {}, texture_upload_path::separate, 1);
								require(warm.ready() && warm.finish().front().image.texture == image.texture,
									"Deferred typeless MSAA window missed its current snapshot");
								const auto row = width * db;
								const auto output = make_buffer(gpu, row * height, buffer_storage::shared);
								texture_region area;
								area.size = MTLSizeMake(width, height, 1);
								gpu.begin();
								if (const auto color = std::get_if<surface_color>(&df)) colors.download(gpu, *color, image.texture, area, output, {0, row});
								else depths.download(gpu, std::get<surface_depth>(df), image.texture, area, output, {0, row});
								gpu.submit().wait();
								for (unsigned y = 0; y < height; ++y)
									for (unsigned x = 0; x < width; ++x)
									{
										const auto sx = axis(x + 1, sr.width * in.x * sb / db, dr.width * out.x, flip[0]);
										const auto sy = axis(y + 1, sr.height * in.y, dr.height * out.y, flip[1]);
										const auto start = (sr.y * in.y + sy) * 64 + sr.x * in.x * sb + sx * db;
										for (unsigned byte = 0; byte < db; ++byte)
										{
											const bool newer_stencil = split && byte == 3 && (x + 1) / out.x == 1 && (y + 1) / out.y == 1;
											const auto expected = newer_stencil ? std::byte{0x99} : original[start + byte];
											require(static_cast<const std::byte*>(output.contents)[y * row + x * db + byte] == expected,
												"Deferred typeless MSAA projection changed a guest byte or another plane");
											++bytes_checked;
										}
									}
								++windows;
							}
						}
		std::cout << "Deferred typeless MSAA: " << windows << " windows, " << bytes_checked
			<< " exact BE guest bytes; six source/target formats, 1/2/4 to 2/4 samples, scaling, flips, odd crops and newer stencil, ledger-pressure refusal/retry and event timeout, no target materialization or guest reads\n";
	}
	void multisample_window_budget_case(bool spill = false, surface_color color_format = surface_color::argb8)
	{
		context gpu{1};
		const bool float_color = color_format == surface_color::rgba16f || color_format == surface_color::rgba32f || color_format == surface_color::r32f;
		const auto budget_bytes = (float_color ? 128u : 32u) << 20;
		auto budget = std::make_shared<heap_budget>(gpu, budget_bytes);
		gpu.set_buffer_budget(budget);
		surface_cache surfaces(gpu, ram, budget_bytes, budget);
		guest_texture_cache textures(gpu, ram, [](unsigned offset, unsigned, std::size_t) { return 0x1000u + offset - 0x200u; },
			1u << 20, budget);
		constexpr unsigned size = 512;
		const auto pitch = size * 2 * color_layout(color_format).pixel_bytes;
		ram->bytes.resize(std::max<std::size_t>(5u << 20, pitch * size * 2));
		++ram->version;
		const auto source = surfaces.acquire({0x1000, {size, size, 4, color_format, false}, pitch, {}});
		if (float_color)
		{
			// The conventional RSX clear is a NOP for float surfaces. Seed
			// this fixture with an actual native attachment clear instead.
			surfaces.prepare(source, surface_initialization::background);
			surfaces.draw(std::array{cached_surface_use{source, true}}, [&](context& commands, auto images)
			{
				auto pass = [MTL4RenderPassDescriptor new];
				pass.colorAttachments[0].texture = images[0];
				pass.colorAttachments[0].loadAction = MTLLoadActionClear;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				pass.colorAttachments[0].clearColor = MTLClearColorMake(32. / 255, 64. / 255, 128. / 255, 1);
				(void)commands.render(pass);
			});
		}
		else surfaces.clear(source, 0xf0, 0xff204080, 0, 255, {0, 0, size, size});
		if (spill) surfaces.spill(source);
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:describe_surface_color(color_format).format
			width:size height:size mipmapped:NO];
		descriptor.textureType = MTLTextureType2DMultisample;
		descriptor.sampleCount = 4;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
		descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		if (spill) descriptor.usage |= MTLTextureUsageRenderTarget;
		const auto full_charge = [gpu.device() heapTextureSizeAndAlignWithDescriptor:descriptor].size;
		const auto headroom = spill ? full_charge + full_charge / 2 : full_charge / 2;
		const auto before = budget->statistics();
		require(before.budget - before.reserved_bytes > headroom, "MSAA pressure fixture cannot hold its source and blocker");
		const auto blocker = budget->shared_buffer(before.budget - before.reserved_bytes - headroom);
		const auto limited = budget->statistics();
		require(blocker && limited.budget - limited.reserved_bytes == headroom,
			"MSAA pressure fixture did not reserve the expected headroom");
		if (spill)
		{
			auto expanded = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:descriptor.pixelFormat
				width:size * 2 height:size * 2 mipmapped:NO];
			expanded.storageMode = descriptor.storageMode;
			expanded.hazardTrackingMode = descriptor.hazardTrackingMode;
			expanded.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto scratch_charge = [gpu.device() heapTextureSizeAndAlignWithDescriptor:expanded].size;
			require(full_charge <= headroom && headroom - full_charge < scratch_charge,
				"MSAA spill pressure fixture did not isolate scratch allocation failure");
		}
		const auto code = color_format == surface_color::rgba16f ? 0x9au : color_format == surface_color::rgba32f ? 0x9bu : color_format == surface_color::r32f ? 0x9cu : 0x85u;
		auto regs = words(code, vertex_texture_dimension::two, true, 3, 2);
		regs.offset += pitch + color_layout(color_format).pixel_bytes;
		regs.control3 = (1u << 20) | pitch;
		const auto reads = ram->reads;
		const auto revision = ram->version;
		const auto result = textures.prepare_versioned_batch(std::array{regs}, surfaces, {}, {}, texture_upload_path::separate, 1).finish().front().image;
		const auto after = budget->statistics();
		if (!(result.texture.sampleCount == 1 && result.texture.width == 3 && result.texture.height == 2 &&
			result.metadata.coordinates.flags == texture_coordinate_sample_raster &&
			surfaces.snapshot_statistics().created == 1 && ram->reads == reads && ram->version == revision &&
			(!spill || (surfaces.status(source).spilled && after.created == limited.created + 2 &&
				after.reserved_bytes < limited.reserved_bytes + full_charge))))
			throw std::runtime_error("MSAA window pressure did not fall back to its small exact native crop: format " + std::to_string(code) +
			", samples " + std::to_string(result.texture.sampleCount) + ", width " + std::to_string(result.texture.width) +
			", flags " + std::to_string(result.metadata.coordinates.flags) + ", created " + std::to_string(after.created - limited.created) +
			", reserved delta " + std::to_string(after.reserved_bytes - limited.reserved_bytes) + ", full charge " + std::to_string(full_charge));
		color expected{32.f / 255, 64.f / 255, 128.f / 255, 1};
		if (color_format == surface_color::rgba16f) for (auto& lane : expected) lane = float(_Float16(lane));
		if (color_format == surface_color::r32f) expected.fill(expected[0]);
		sample(result, regs, {.5f, .5f, 0, 1}, expected);
		std::cout << (spill ? "MSAA spill budget: import scratch refused, native output released" : "MSAA window budget: full native copy refused")
			<< ", one exact 3x2 expanded snapshot published without guest reads\n";
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
int main(int argc,char** argv)
{
	@autoreleasepool
	{
		try
		{
			fixture f;
			if (argc == 2 && std::string_view(argv[1]) == "direct-msaa")
			{
				f.multisample_snapshot_cases();
				f.multisample_window_budget_case();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "deferred-msaa")
			{
				f.deferred_multisample_snapshot_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "guest-hole-msaa")
			{
				f.deferred_guest_multisample_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "projected-msaa")
			{
				f.projected_deferred_multisample_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "typeless-msaa")
			{
				f.deferred_typeless_multisample_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "float-msaa-spill")
			{
				f.multisample_snapshot_cases(true);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "float-msaa-spill-words")
			{
				f.float_multisample_spill_words();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "redirected-depth")
			{
				f.redirected_depth_shader();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "packed-depth-color")
			{
				f.packed_depth_color_filtering();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "cold-spill")
			{
				f.region_alias_cases(fixture::region_source::spilled);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "native-msaa-spill")
			{
				f.region_alias_cases(fixture::region_source::native_spilled);
				f.multisample_window_budget_case(true);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "cold-deferred")
			{
				f.region_alias_cases(fixture::region_source::deferred);
				f.deferred_snapshot_epochs();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "conversion-dependencies")
			{
				f.conversion_dependencies();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "stored-border")
			{
				f.stored_border_cases();
				f.stored_morton_border_cases();
				f.stored_border_clamp_cases();
				f.stored_depth_border_cases();
				f.stored_border_assembly_cases();
				f.stored_border_assembly_cases(true);
				f.stored_border_adjusted_pitch_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "stored-border-anisotropy")
			{
				f.stored_border_anisotropic_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "stored-border-cube")
			{
				f.stored_border_cube_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "stored-border-volume")
			{
				f.stored_border_volume_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "paired-morton")
			{ f.morton_paired_cases(); return 0; }
			if (argc == 2 && std::string_view(argv[1]) == "paired-border")
			{
				f.paired_border_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "stored-border-1d")
			{
				f.stored_border_1d_cases();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "stored-border-mip-atlas")
			{
				f.stored_border_mip_atlas_cases();
				f.sampled_border_atlas_cases();
				return 0;
			}
			f.adjusted_pitch_cases();
			f.adjusted_pitch_aliases();
			if(argc==2&&std::string_view(argv[1])=="adjusted-pitch")return 0;
			f.stored_border_cases();
			f.stored_border_anisotropic_cases();
			f.stored_border_cube_cases();
			f.stored_border_volume_cases();
			f.paired_border_cases();
			f.morton_paired_cases();
			f.stored_border_1d_cases();
			f.stored_border_mip_atlas_cases();
			f.sampled_border_atlas_cases();
			f.stored_morton_border_cases();
			f.stored_border_clamp_cases();
			f.stored_depth_border_cases();
			f.stored_border_assembly_cases();
			f.stored_border_assembly_cases(true);
			f.stored_border_adjusted_pitch_cases();
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
			f.conversion_dependencies();
			f.upload_job_cases();
			f.captured_lifetime_cases();
			f.versioned_cases();
			f.cache_cases();
			f.packed_depth_color_filtering();
			f.redirected_depth_shader();
			f.surface_alias_cases();
			f.native_snapshot_cache_cases();
			f.typeless_surface_cases();
			f.partial_surface_cases();
			f.block_surface_cases();
			f.clean_surface_cpu_changes();
			f.region_alias_cases();
			f.region_alias_cases(fixture::region_source::spilled);
			f.region_alias_cases(fixture::region_source::native_spilled);
			f.multisample_window_budget_case(true);
			f.region_alias_cases(fixture::region_source::deferred);
			f.deferred_snapshot_epochs();
			f.multisample_snapshot_cases();
			f.multisample_snapshot_cases(true);
			f.float_multisample_spill_words();
			f.multisample_window_budget_case();
			f.deferred_multisample_snapshot_cases();
			f.deferred_guest_multisample_cases();
			f.projected_deferred_multisample_cases();
			f.deferred_typeless_multisample_cases();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
