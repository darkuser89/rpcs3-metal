#include "M4GuestTexture.h"
#include "M4DisplayStorage.h"
#include "M4GuestMemoryOverlap.h"
#include "M4ExclusiveGuestMemory.h"
#include "M4CachedTextureRegion.h"
#include "M4SurfaceCache.h"
#include "M4SurfaceTextureRegion.h"

#include <algorithm>
#include <atomic>

#if !__has_feature(objc_arc)
#error "Native immutable texture ownership requires ARC."
#endif

namespace metal4
{
MTLPixelFormat guest_texture_resolver::native_format(guest_texture_storage value)
{
	using S = guest_texture_storage;
	switch (value)
	{
	case S::r8:
		return MTLPixelFormatR8Unorm;
	case S::bgr5a1:
		return MTLPixelFormatBGR5A1Unorm;
	case S::abgr4:
		return MTLPixelFormatABGR4Unorm;
	case S::bgr565:
		return MTLPixelFormatB5G6R5Unorm;
	case S::bgra8:
		return MTLPixelFormatBGRA8Unorm;
	case S::bc1:
		return MTLPixelFormatBC1_RGBA;
	case S::bc2:
		return MTLPixelFormatBC2_RGBA;
	case S::bc3:
		return MTLPixelFormatBC3_RGBA;
	case S::rg8:
		return MTLPixelFormatRG8Unorm;
	case S::d16:
		return MTLPixelFormatDepth16Unorm;
	case S::d32s8:
		return MTLPixelFormatDepth32Float_Stencil8;
	case S::d32:
		return MTLPixelFormatDepth32Float;
	case S::r16:
		return MTLPixelFormatR16Unorm;
	case S::rg16:
		return MTLPixelFormatRG16Unorm;
	case S::a1bgr5:
		return MTLPixelFormatA1BGR5Unorm;
	case S::rg8snorm:
		return MTLPixelFormatRG8Snorm;
	case S::rgba16f:
		return MTLPixelFormatRGBA16Float;
	case S::rgba32f:
		return MTLPixelFormatRGBA32Float;
	case S::r32f:
		return MTLPixelFormatR32Float;
	case S::rg16f:
		return MTLPixelFormatRG16Float;
	}
	throw std::invalid_argument("Unknown native guest texture storage");
}
namespace
{
struct conversion_parameters
{
	std::uint64_t source_offset, source_row, source_image, first, count;
	std::uint32_t columns, rows, depth, width;
	std::uint32_t block_bytes, swap_width, mode, morton;
	std::array<std::uint32_t, 3> log2;
	std::uint32_t reserved = 0;
};
static_assert(sizeof(conversion_parameters) == 88 && offsetof(conversion_parameters, columns) == 40 &&
	offsetof(conversion_parameters, log2) == 72);
constexpr std::string_view conversion_source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct params {
    ulong source_offset,source_row,source_image,first,count;
    uint columns,rows,depth,width,block_bytes,swap_width,mode,morton;
    uint log2[3],reserved;
};
kernel void convert_texture(device const uchar* src [[buffer(0)]],device uchar* dst [[buffer(1)]],
    constant params& p [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
    if (ulong(tid)>=p.count) return;
    const ulong ordinal=p.first+tid;
    const ulong x=ordinal%p.columns,y=(ordinal/p.columns)%p.rows,z=ordinal/(ulong(p.columns)*p.rows);
    ulong address=p.source_offset+z*p.source_image+y*p.source_row+x*p.block_bytes;
    if(p.morton==2u) {
        if(z<(p.depth&~3u)) address=p.source_offset+(z&~3ul)*p.source_image+4*(y*p.source_row+x*p.block_bytes)+(z&3ul)*p.block_bytes;
    } else if(p.morton) {
        const ulong3 position(x,y,z);ulong index=0;uint output=0;
        for(uint bit=0;bit<max(max(p.log2[0],p.log2[1]),p.log2[2]);++bit)
            for(uint axis=0;axis<3;++axis)
                if(bit<p.log2[axis]) index|=((position[axis]>>bit)&1ul)<<output++;
        address=p.source_offset+index*p.block_bytes;
    }
    if(!p.mode) {
        for(uint byte=0;byte<p.block_bytes;++byte)
            dst[ordinal*p.block_bytes+byte]=src[address+(byte^(p.swap_width-1))];
    } else if(p.mode==1) {
        uint v=(uint(src[address])<<8)|src[address+1];v=(v&0xf81fu)|((v&0x3e0u)<<1);
        dst[ordinal*2]=uchar(v);dst[ordinal*2+1]=uchar(v>>8);
    } else {
        const uint blue=p.mode==2?src[address]:src[address+1];
        const uint green=p.mode==2?src[address+2]:src[address+3];
        for(uint lane=0;lane<2 && x*2+lane<p.width;++lane) {
            const uint red=p.mode==2?src[address+1+lane*2]:src[address+lane*2];
            const ulong output=((z*p.rows+y)*p.width+x*2+lane)*4;
            dst[output]=uchar(blue);dst[output+1]=uchar(green);dst[output+2]=uchar(red);dst[output+3]=255;
        }
    }
}
// Integer stores plus native blits keep float-depth raw bits out of a
// floating-point arithmetic/store path. G stores the stencil byte exactly.
kernel void decode_depth_volume(device const uchar* src [[buffer(0)]],device uint2* dst [[buffer(1)]],
    constant params& p [[buffer(2)]],uint tid [[thread_position_in_grid]]) {
    if(ulong(tid)>=p.count)return;const ulong ordinal=p.first+tid;
    const ulong x=ordinal%p.columns,y=(ordinal/p.columns)%p.rows,z=ordinal/(ulong(p.columns)*p.rows);
    ulong address=p.source_offset+z*p.source_image+y*p.source_row+x*p.block_bytes;
    if(p.morton){const ulong3 position(x,y,z);ulong index=0;uint output=0;
        for(uint bit=0;bit<max(max(p.log2[0],p.log2[1]),p.log2[2]);++bit)for(uint axis=0;axis<3;++axis)if(bit<p.log2[axis])index|=((position[axis]>>bit)&1ul)<<output++;
        address=p.source_offset+index*p.block_bytes;}
    uint raw=0;for(uint i=0;i<p.block_bytes;++i)raw=(raw<<8)|src[address+i];uint depth;
    if(p.mode==1u)depth=as_type<uint>(float(raw)/65535.f);
    else if(p.mode==2u)depth=as_type<uint>(float(raw>>8)/16777215.f);
    else if(p.mode==3u)depth=(raw<<11)+(120u<<23);
    else depth=(raw>>8)<<7;
    dst[ordinal]=uint2(depth,(p.mode==2u||p.mode==4u)?as_type<uint>(float(raw&255u)):0u);
}
kernel void interleave_depth_volume(device const uchar* src [[buffer(0)]],device const uchar* stencil [[buffer(1)]],
    device uint2* dst [[buffer(2)]],constant ulong2& shape [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    if(ulong(tid)>=shape.x)return;const uint bytes=uint(shape.y&255ul);uint raw=0;
    for(uint i=0;i<bytes;++i)raw|=uint(src[ulong(tid)*bytes+i])<<(i*8);
    if(bytes==2u)raw=as_type<uint>(float(raw)/65535.f);
    dst[tid]=uint2(raw,(shape.y&256ul)?as_type<uint>(float(stencil[tid])):0u);
}
// Hardware S3TC decoding feeds a float volume without an extra UNORM
// quantization. Source slices were gathered from the VTC block ordering.
kernel void decode_vtc(texture2d_array<float> src [[texture(0)]],texture3d<float,access::write> dst [[texture(1)]],
    constant uint4& shape [[buffer(0)]],uint3 p [[thread_position_in_grid]]) {
    if(any(p>=shape.xyz))return;
    dst.write(src.read(p.xy,p.z,shape.w),p,shape.w);
}
struct display_patch { ulong target,payload,size; };
kernel void scatter_display_holes(device const uchar* src [[buffer(0)]],device uchar* dst [[buffer(1)]],
    device const display_patch* patches [[buffer(2)]],constant ulong2& shape [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    if(ulong(tid)>=shape.x) return;
    ulong first=0,last=shape.y;
    while(first<last) {
        const ulong middle=first+(last-first)/2;
        if(patches[middle].payload<=ulong(tid)) first=middle+1;else last=middle;
    }
    if(!first) return;
    const auto patch=patches[first-1];const ulong lane=ulong(tid)-patch.payload;
    if(lane<patch.size) dst[patch.target+lane]=src[tid];
}
struct display_copy { ulong target,source,ordinal,size; };
kernel void copy_display_storage(device const uchar* src [[buffer(0)]],device uchar* dst [[buffer(1)]],
    device const display_copy* copies [[buffer(2)]],constant ulong2& shape [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    if(ulong(tid)>=shape.x) return;
    ulong first=0,last=shape.y;
    while(first<last) {
        const ulong middle=first+(last-first)/2;
        if(copies[middle].ordinal<=ulong(tid)) first=middle+1;else last=middle;
    }
    if(!first) return;
    const auto copy=copies[first-1];const ulong lane=ulong(tid)-copy.ordinal;
    if(lane<copy.size) dst[copy.target+lane]=src[copy.source+lane];
}
)metal";
} // namespace
guest_texture_resolver::guest_texture_resolver(context& ctx, std::shared_ptr<guest_surface_memory> memory,
											   std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate, std::size_t maximum_snapshot_bytes, std::shared_ptr<heap_budget> budget, cpu_capture_dispatch capture)
	: m_context(ctx), m_memory(std::move(memory)), m_translate(std::move(translate)), m_limit(maximum_snapshot_bytes), m_capture(std::move(capture)), m_budget(std::move(budget)), m_depth(ctx), m_color(ctx)
{
	if (!m_memory || !m_translate || !m_limit)
		throw std::invalid_argument("Guest texture resolver requires VM/address mapping and a snapshot limit");
	if (m_budget && m_budget->device() != ctx.device())
		throw std::invalid_argument("Guest texture/shared budget devices differ");
	helper_catalog compile(ctx);
	const auto library = compile.library(conversion_source, @"Native RSX texture input conversion");

	m_conversion = compile.compute(library, @"convert_texture");
}
guest_texture_resolver::source guest_texture_resolver::prepare(const fragment_texture_registers& words) const
{
	const auto plan = plan_guest_texture(words);
	if (plan.storage_bytes > m_limit)
		throw std::length_error("Guest texture snapshot exceeds its explicit limit");
	if (plan.stored_border || plan.adjusted_pitch)
		throw std::invalid_argument("Stored-border/degenerate-pitch texture needs full native coordinate/cache policy");
	const bool dxt = plan.codec.block.width == 4;

	if (dxt && !m_context.device().supportsBCTextureCompression)
		throw std::invalid_argument("Native GPU lacks BC texture sampling");
	for (const auto& s : plan.subresources)
		if (plan.swizzled && plan.codec.block.width == 1)
			(void)plan_swizzle({{s.columns, s.rows, s.depth}, {}, {s.columns, s.rows, s.depth}}, plan.codec.block.bytes);
	auto segments = texture_storage_segments(plan.offset, plan.location, plan.storage_bytes, m_translate);
	std::vector<guest_surface_layout> coverage;
	for (const auto& part : segments) coverage.push_back(guest_byte_coverage({part.address, part.size}));
	return {plan, std::move(segments), std::move(coverage)};
}

void guest_texture_resolver::cpu_capture(const std::function<void()>& operation) const
{
	if (m_capture) checked_cpu_capture(m_capture, operation);
	else operation(); // Legacy standalone callers own producer exclusion.
}
void guest_texture_resolver::watch(source& input) const
{
	for (auto& coverage : input.coverage) coverage = m_memory->watch_surface(coverage);
}
bool guest_texture_resolver::current(const source& input, const std::vector<std::uint64_t>& revisions) const
{
	if (texture_storage_segments(input.plan.offset, input.plan.location, input.plan.storage_bytes, m_translate) != input.segments ||
		input.coverage.size() != revisions.size()) return false;
	for (std::size_t i = 0; i < revisions.size(); ++i)
		if (m_memory->surface_revision(input.coverage[i]) != revisions[i]) return false;
	return true;
}
resolved_fragment_texture guest_texture_resolver::resolve(const fragment_texture_registers& words)
{
	source input;
	cpu_capture([&] { input = prepare(words); watch(input); });
	std::vector<std::uint64_t> revisions;
	return upload(words, input, revisions);
}
resolved_fragment_texture guest_texture_resolver::upload(const fragment_texture_registers& words, const source& input, std::vector<std::uint64_t>& revisions)
{
	// Preserve quota preflight before a potentially expensive VM snapshot.
	const auto texture = allocate(input.plan);
	const auto snapshot = capture(words, input, revisions);
	const auto result = upload(snapshot, texture);
	cpu_capture([&] { if (!current(input, revisions)) throw texture_revision_conflict(); });
	return result;
}
captured_guest_texture guest_texture_resolver::capture(const fragment_texture_registers& words)
{
	source input;
	cpu_capture([&] { input = prepare(words); watch(input); });
	std::vector<std::uint64_t> revisions;
	return capture(words, input, revisions);
}
captured_guest_texture guest_texture_resolver::capture(const fragment_texture_registers& words, const source& input, std::vector<std::uint64_t>& revisions)
{
	std::vector<std::byte> bytes(input.plan.storage_bytes);
	cpu_capture([&]
	{
		if (texture_storage_segments(input.plan.offset, input.plan.location, input.plan.storage_bytes, m_translate) != input.segments)
			throw texture_revision_conflict();
		revisions.clear();
		for (std::size_t i = 0; i < input.segments.size(); ++i)
		{
			const auto& part = input.segments[i];
			auto snapshot = m_memory->read_surface(input.coverage[i]);
			if (snapshot.bytes.size() != part.size)
				throw std::runtime_error("VM texture snapshot has incomplete storage");
			std::copy(snapshot.bytes.begin(), snapshot.bytes.end(), bytes.begin() + part.offset);
			revisions.push_back(snapshot.revision);
		}
		if (!current(input, revisions)) throw texture_revision_conflict();
	});
	const auto& plan = input.plan;
	texture_cache_state metadata;
	metadata.actual_mips = static_cast<std::uint32_t>(plan.mips);
	metadata.depth = bool(plan.codec.depth);
	metadata.source_depth_format = plan.codec.depth ? static_cast<std::uint32_t>(*plan.codec.depth) : 0;
	metadata.depth_float = plan.codec.depth && (*plan.codec.depth == surface_depth::z16_float || *plan.codec.depth == surface_depth::z24s8_float);
	metadata.native_argb_channels = plan.codec.base_argb;
	if ((words.format >> 8) & 0x40)
		metadata.coordinates = normalize_guest_pixels(plan.width, plan.height, plan.depth);
	return {plan, metadata, std::move(bytes)};
}
struct guest_texture_upload_job::state
{
	submission work;
	std::vector<resolved_fragment_texture> images;
};
guest_texture_upload_job::guest_texture_upload_job(submission work, std::vector<resolved_fragment_texture> images)
	: m_state(std::make_unique<state>(state{std::move(work), std::move(images)})) {}
guest_texture_upload_job::~guest_texture_upload_job() = default;
guest_texture_upload_job::guest_texture_upload_job(guest_texture_upload_job&&) noexcept = default;
guest_texture_upload_job& guest_texture_upload_job::operator=(guest_texture_upload_job&&) noexcept = default;
bool guest_texture_upload_job::ready() const
{
	if (!m_state) throw std::logic_error("Moved-from native texture upload job");
	return m_state->work.ready();
}
bool guest_texture_upload_job::cancel_pending() const
{
	if (!m_state) throw std::logic_error("Moved-from native texture upload job");
	return m_state->work.cancel_pending();
}
std::vector<resolved_fragment_texture> guest_texture_upload_job::finish(std::chrono::milliseconds timeout) const
{
	if (!m_state) throw std::logic_error("Moved-from native texture upload job");
	if (guest_memory_epoch::active()) throw std::logic_error("Native texture upload completion requires leaving VM exclusion");
	m_state->work.wait(timeout);
	return m_state->images;
}
guest_texture_upload_job guest_texture_resolver::enqueue_uploads(std::span<const captured_guest_texture> snapshots,
	std::span<const guest_texture_upload_dependency> dependencies)
{
	return enqueue_mixed(snapshots, {}, dependencies);
}
guest_texture_upload_job guest_texture_resolver::enqueue_mixed(std::span<const captured_guest_texture> snapshots,
	std::span<const native_cast_input> casts, std::span<const guest_texture_upload_dependency> dependencies,
	std::span<const std::shared_ptr<prepared_surface_snapshot>> native)
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native texture upload submission requires leaving VM exclusion");
	if (!m_budget || m_context.buffer_ledger() != m_budget)
		throw std::invalid_argument("Native texture upload batch requires one shared buffer/texture ledger");
	if ((snapshots.empty() && casts.empty() && native.empty()) || snapshots.size() > maximum_upload_images || casts.size() > maximum_upload_images - snapshots.size() || dependencies.size() > 64 || native.size() > maximum_upload_images)
		throw std::length_error("Native texture upload batch exceeds its image/dependency bounds");
	for (const auto& copy : native)
		if (!copy || !copy->needs_copy() || copy->texture().device != m_context.device())
			throw std::invalid_argument("Native texture batch has an invalid prepared snapshot");
	std::size_t total = 0;
	for (const auto& snapshot : snapshots)
	{
		if (snapshot.m_bytes.empty() || snapshot.m_bytes.size() != snapshot.m_plan.storage_bytes)
			throw std::invalid_argument("Moved-from or incomplete captured guest texture");
		if (snapshot.m_bytes.size() > m_limit || snapshot.m_bytes.size() > maximum_upload_source_bytes - total)
			throw std::length_error("Native texture upload batch exceeds its source byte limit");
		if (snapshot.m_plan.codec.block.width == 4 && !m_context.device().supportsBCTextureCompression)
			throw std::invalid_argument("Native GPU lacks BC texture sampling");
		total += snapshot.m_bytes.size();
	}
	for (const auto& cast : casts)
	{
		if (!cast.source || cast.source.device != m_context.device()) throw std::invalid_argument("Native texture cast source belongs to another device");
		if (cast.packed.storage_bytes > maximum_upload_source_bytes - total) throw std::length_error("Native texture batch exceeds its combined source byte limit");
		total += cast.packed.storage_bytes;
	}
	for (const auto& dependency : dependencies)
		if (!dependency.event || (dependency.event.device && dependency.event.device != m_context.device()))
			throw std::invalid_argument("Native texture upload dependency belongs to another device");
	std::vector<resolved_fragment_texture> images;
	images.reserve(snapshots.size() + casts.size());
	std::optional<submission> work;
	std::exception_ptr failure;
	@autoreleasepool
	{
		bool recording = false;
		try
		{
			m_context.begin(); recording = true;
			for (const auto& dependency : dependencies) m_context.wait_event(dependency.event, dependency.value);
			for (const auto& copy : native) copy->encode(m_context);
			for (const auto& snapshot : snapshots)
			{
				const auto image = allocate(snapshot.m_plan);
				const auto source = make_upload(m_context, snapshot.m_bytes);
				encode_upload(snapshot.m_plan, source, image);
				images.push_back({image, snapshot.m_metadata});
			}
			for (const auto& cast : casts)
			{
				const auto image = allocate(cast.plan);
				encode_cast(cast, image);
				images.push_back({image, cast.metadata});
			}
			work.emplace(m_context.submit()); recording = false;
		}
		catch (...)
		{
			if (recording) { try { m_context.cancel(); } catch (...) {} }
			images.clear();
			failure = std::current_exception();
		}
	}
	// Leave the native pool normally even on C++ failure. Rethrow only after
	// cancelled resources/transient autoreleased owners have actually retired.
	if (failure) std::rethrow_exception(failure);
	return guest_texture_upload_job(std::move(*work), std::move(images));
}
resolved_fragment_texture guest_texture_resolver::realize(const captured_guest_texture& snapshot)
{
	if (snapshot.m_bytes.size() != snapshot.m_plan.storage_bytes)
		throw std::invalid_argument("Moved-from or incomplete captured guest texture");
	if (snapshot.m_bytes.size() > m_limit)
		throw std::length_error("Captured guest texture exceeds destination snapshot limit");
	if (snapshot.m_plan.codec.block.width == 4 && !m_context.device().supportsBCTextureCompression)
		throw std::invalid_argument("Native GPU lacks BC texture sampling");
	return upload(snapshot, allocate(snapshot.m_plan));
}
id<MTLTexture> guest_texture_resolver::allocate(const guest_texture_layout& plan)
{
	const auto format = guest_texture_vtc(plan) ? MTLPixelFormatRGBA32Float : guest_texture_depth_volume(plan) ? MTLPixelFormatRG32Float : native_format(plan.codec.storage);
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:plan.width height:plan.height mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	if(guest_texture_vtc(plan))descriptor.usage |= MTLTextureUsageShaderWrite;
	descriptor.mipmapLevelCount = plan.mips;
	if (plan.codec.depth && !guest_texture_depth_volume(plan))
		descriptor.usage |= MTLTextureUsageRenderTarget;
	if (plan.dimension == vertex_texture_dimension::cube)
		descriptor.textureType = MTLTextureTypeCube;
	if (plan.dimension == vertex_texture_dimension::three)
	{
		descriptor.textureType = MTLTextureType3D;
		descriptor.depth = plan.depth;
	}
	return m_budget ? m_budget->texture(descriptor) : make_texture(m_context, descriptor);
}
void guest_texture_resolver::encode_depth_volume(const guest_texture_layout& plan,id<MTLBuffer> guest,id<MTLTexture> texture)
{
	if(!guest||guest.device!=m_context.device()||guest.length<plan.storage_bytes||!texture||texture.device!=m_context.device()||texture.textureType!=MTLTextureType3D||texture.pixelFormat!=MTLPixelFormatRG32Float||texture.width!=plan.width||texture.height!=plan.height||texture.depth!=plan.depth||texture.mipmapLevelCount!=plan.mips)
		throw std::invalid_argument("Depth volume upload lacks complete native storage");
	std::size_t maximum=0;for(const auto& s:plan.subresources)maximum=std::max(maximum,checked_texture_product(checked_texture_product(s.width,s.height),checked_texture_product(s.depth,8)));
	if(!m_depth_volume_decode){helper_catalog compile(m_context);const auto library=compile.library(conversion_source,@"Native RSX texture input conversion");m_depth_volume_decode=compile.compute(library,@"decode_depth_volume");}
	const auto staging=make_buffer(m_context,maximum,buffer_storage::private_gpu);++m_upload_statistics.staging_buffers;m_upload_statistics.staging_bytes+=std::max(staging.length,staging.allocatedSize);
	for(const auto& s:plan.subresources)
	{
		const auto texels=checked_texture_product(checked_texture_product(s.width,s.height),s.depth);const auto log2=plan.swizzled?plan_swizzle({{s.columns,s.rows,s.depth},{},{s.columns,s.rows,s.depth}},plan.codec.block.bytes).log2:std::array<std::uint32_t,3>{};
		for(std::size_t first=0;first<texels;)
		{
			const auto count=std::min<std::size_t>(texels-first,1u<<30);const conversion_parameters params{s.offset,s.row_pitch,checked_texture_product(s.row_pitch,s.storage_rows),first,count,static_cast<std::uint32_t>(s.columns),static_cast<std::uint32_t>(s.rows),static_cast<std::uint32_t>(s.depth),static_cast<std::uint32_t>(s.width),static_cast<std::uint32_t>(plan.codec.block.bytes),1,static_cast<std::uint32_t>(*plan.codec.depth),plan.swizzled,log2,0};
			const auto upload=m_context.upload(std::as_bytes(std::span{&params,1}),256);bindings resources;resources.set_buffer(0,guest);resources.set_buffer(1,staging);resources.set_buffer(2,upload.buffer,upload.offset);m_context.keep_alive(m_depth_volume_decode);const auto table=resources.snapshot(m_context);const auto encoder=m_context.compute();m_context.barrier(MTLStageBlit|MTLStageDispatch,MTLStageDispatch);[encoder setComputePipelineState:m_depth_volume_decode];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,m_depth_volume_decode.maxTotalThreadsPerThreadgroup),1,1)];++m_upload_statistics.conversion_dispatches;first+=count;
		}
		texture_region region;region.level=s.level;region.size=MTLSizeMake(s.width,s.height,s.depth);buffer_to_texture(m_context,staging,0,texture,region,s.width*8,s.width*s.height*8);++m_upload_statistics.subresources;++m_upload_statistics.depth_volume_subresources;m_upload_statistics.depth_volume_texels+=texels;
	}
}
void guest_texture_resolver::encode_depth_volume_plane(id<MTLTexture> source,const texture_region& region,id<MTLTexture> target,unsigned level,std::size_t z)
{
	const auto format=source.pixelFormat;const bool short_depth=format==MTLPixelFormatDepth16Unorm,stencil=format==MTLPixelFormatDepth32Float_Stencil8;
	if(!target||target.pixelFormat!=MTLPixelFormatRG32Float||target.textureType!=MTLTextureType3D||(!short_depth&&format!=MTLPixelFormatDepth32Float&&!stencil)||region.size.depth!=1)
		throw std::invalid_argument("Depth volume plane requires matching native depth/stencil source");
	const auto texels=checked_texture_product(region.size.width,region.size.height);const unsigned unit=short_depth?2u:4u;if(texels>(1u<<30))throw std::length_error("Depth volume plane exceeds dispatch budget");
	if(!m_depth_volume_interleave){helper_catalog compile(m_context);const auto library=compile.library(conversion_source,@"Native RSX texture input conversion");m_depth_volume_interleave=compile.compute(library,@"interleave_depth_volume");}
	const auto depth=make_buffer(m_context,checked_texture_product(texels,unit),buffer_storage::private_gpu),output=make_buffer(m_context,checked_texture_product(texels,8),buffer_storage::private_gpu);const auto stencil_data=stencil?make_buffer(m_context,texels,buffer_storage::private_gpu):depth;
	texture_to_buffer(m_context,source,region,depth,0,region.size.width*unit,0,texture_plane::depth);if(stencil)texture_to_buffer(m_context,source,region,stencil_data,0,region.size.width,0,texture_plane::stencil);
	const std::array<std::uint64_t,2> shape{texels,unit|(stencil?256u:0u)};const auto upload=m_context.upload(std::as_bytes(std::span{&shape,1}),256);bindings resources;resources.set_buffer(0,depth);resources.set_buffer(1,stencil_data);resources.set_buffer(2,output);resources.set_buffer(3,upload.buffer,upload.offset);m_context.keep_alive(m_depth_volume_interleave);const auto table=resources.snapshot(m_context);const auto encoder=m_context.compute();m_context.barrier(MTLStageBlit|MTLStageDispatch,MTLStageDispatch);[encoder setComputePipelineState:m_depth_volume_interleave];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(texels,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,m_depth_volume_interleave.maxTotalThreadsPerThreadgroup),1,1)];
	texture_region destination;destination.level=level;destination.origin=MTLOriginMake(0,0,z);destination.size=region.size;buffer_to_texture(m_context,output,0,target,destination,region.size.width*8,texels*8);++m_upload_statistics.depth_volume_plane_packs;
}
void guest_texture_resolver::encode_vtc(const guest_texture_layout& plan,id<MTLBuffer> guest,id<MTLTexture> texture)
{
	if(!guest||guest.device!=m_context.device()||guest.length<plan.storage_bytes||!texture||texture.device!=m_context.device()||
		texture.textureType!=MTLTextureType3D||texture.pixelFormat!=MTLPixelFormatRGBA32Float||texture.width!=plan.width||texture.height!=plan.height||texture.depth!=plan.depth||texture.mipmapLevelCount!=plan.mips)
		throw std::invalid_argument("VTC upload lacks complete native source/destination storage");
	std::size_t maximum=0;
	for(const auto& s:plan.subresources)
	{
		(void)guest_texture_block_offset(plan,s,s.columns-1,s.rows-1,s.depth-1);
		maximum=std::max(maximum,checked_texture_product(checked_texture_product(s.columns,s.rows),checked_texture_product(s.depth,plan.codec.block.bytes)));
	}
	if(!m_vtc_decode){helper_catalog compile(m_context);const auto library=compile.library(conversion_source,@"Native RSX texture input conversion");m_vtc_decode=compile.compute(library,@"decode_vtc");}
	auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native_format(plan.codec.storage) width:plan.width height:plan.height mipmapped:NO];
	descriptor.textureType=MTLTextureType2DArray;descriptor.arrayLength=plan.depth;descriptor.mipmapLevelCount=plan.mips;descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageShaderRead;
	const auto compressed=m_budget?m_budget->texture(descriptor):make_texture(m_context,descriptor);++m_upload_statistics.vtc_array_images;
	const auto staging=make_buffer(m_context,maximum,buffer_storage::private_gpu);++m_upload_statistics.staging_buffers;m_upload_statistics.staging_bytes+=std::max(staging.length,staging.allocatedSize);
	for(const auto& s:plan.subresources)
	{
		const auto blocks=checked_texture_product(checked_texture_product(s.columns,s.rows),s.depth);
		for(std::size_t first=0;first<blocks;)
		{
			const auto count=std::min<std::size_t>(blocks-first,1u<<30);
			const conversion_parameters params{s.offset,s.row_pitch,checked_texture_product(s.row_pitch,s.storage_rows),first,count,
				static_cast<std::uint32_t>(s.columns),static_cast<std::uint32_t>(s.rows),static_cast<std::uint32_t>(s.depth),static_cast<std::uint32_t>(s.width),static_cast<std::uint32_t>(plan.codec.block.bytes),1,0,2,{},0};
			const auto upload=m_context.upload(std::as_bytes(std::span{&params,1}),256);bindings resources;resources.set_buffer(0,guest);resources.set_buffer(1,staging);resources.set_buffer(2,upload.buffer,upload.offset);m_context.keep_alive(m_conversion);const auto table=resources.snapshot(m_context);const auto encoder=m_context.compute();m_context.barrier(MTLStageBlit|MTLStageDispatch,MTLStageDispatch);[encoder setComputePipelineState:m_conversion];[encoder setArgumentTable:table];[encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,m_conversion.maxTotalThreadsPerThreadgroup),1,1)];
			++m_upload_statistics.conversion_dispatches;m_upload_statistics.conversion_blocks+=count;first+=count;
		}
		const auto row=checked_texture_product(s.columns,plan.codec.block.bytes),image=checked_texture_product(row,s.rows);
		for(std::size_t z=0;z<s.depth;++z){texture_region region;region.level=s.level;region.slice=z;region.size=MTLSizeMake(s.width,s.height,1);buffer_to_texture(m_context,staging,z*image,compressed,region,row,image);}
		const std::array<std::uint32_t,4> shape{static_cast<std::uint32_t>(s.width),static_cast<std::uint32_t>(s.height),static_cast<std::uint32_t>(s.depth),s.level};const auto upload=m_context.upload(std::as_bytes(std::span{&shape,1}),256);bindings resources;resources.set_texture(0,compressed);resources.set_texture(1,texture);resources.set_buffer(0,upload.buffer,upload.offset);m_context.keep_alive(m_vtc_decode);const auto table=resources.snapshot(m_context);const auto encoder=m_context.compute();m_context.barrier(MTLStageBlit|MTLStageDispatch,MTLStageDispatch);[encoder setComputePipelineState:m_vtc_decode];[encoder setArgumentTable:table];const NSUInteger x=std::min<NSUInteger>(8,m_vtc_decode.maxTotalThreadsPerThreadgroup);[encoder dispatchThreads:MTLSizeMake(s.width,s.height,s.depth) threadsPerThreadgroup:MTLSizeMake(x,std::min<NSUInteger>(8,m_vtc_decode.maxTotalThreadsPerThreadgroup/x),1)];
		++m_upload_statistics.subresources;++m_upload_statistics.vtc_decode_dispatches;m_upload_statistics.vtc_texels+=s.width*s.height*s.depth;m_upload_statistics.converted_bytes+=image*s.depth;
	}
}
void guest_texture_resolver::encode_upload(const guest_texture_layout& plan, id<MTLBuffer> guest, id<MTLTexture> texture)
{
	if(guest_texture_depth_volume(plan)){encode_depth_volume(plan,guest,texture);return;}
	if(guest_texture_vtc(plan)){encode_vtc(plan,guest,texture);return;}
	struct operation
	{
		const guest_texture_subresource* subresource;
		texture_region region;
		bool morton, direct;
		std::size_t staging_bytes;
	};
	std::vector<operation> operations;
	operations.reserve(plan.subresources.size());
	std::size_t maximum_staging = 0;
	if (!guest || guest.device != m_context.device() || guest.length < plan.storage_bytes)
		throw std::invalid_argument("Native texture upload source does not own its complete layout");
	for (const auto& s : plan.subresources)
	{
		const bool morton = plan.swizzled && plan.codec.block.width == 1;
		texture_region region;
		region.level = s.level; region.slice = s.face;
		region.size = MTLSizeMake(s.width, s.height, s.depth);
		const auto native_block = storage_block(texture.pixelFormat, plan.codec.depth ? texture_plane::depth : texture_plane::all);
		const auto pitched_bytes = checked_texture_product(checked_texture_product(s.row_pitch, s.storage_rows), s.depth);
		const bool direct = !morton && s.offset <= guest.length && pitched_bytes <= guest.length - s.offset &&
			(plan.codec.depth ? s.row_pitch % plan.codec.block.bytes == 0 :
			!plan.codec.conversion && plan.codec.endian_width == 1 && s.offset % native_block.bytes == 0 && s.row_pitch % native_block.bytes == 0);
		const auto bytes = direct ? 0 : plan.codec.depth ?
			transfer_layout({1, 1, plan.codec.block.bytes}, s.columns, s.rows, s.depth).allocation_size :
			transfer_layout(native_block, s.width, s.height, s.depth).allocation_size;
		maximum_staging = std::max(maximum_staging, bytes);
		operations.push_back({&s, region, morton, direct, bytes});
	}
	// One bounded private scratch range for this upload, reused only after the
	// prior native blit/codec reader. Dispatch|Blit -> Dispatch barriers enforce
	// that ordering; argument bindings/completion retain the one allocation.
	const auto staging = maximum_staging ? make_buffer(m_context, maximum_staging, buffer_storage::private_gpu) : nil;
	if (staging)
	{
		++m_upload_statistics.staging_buffers;
		m_upload_statistics.staging_bytes += std::max(staging.length, staging.allocatedSize);
	}
	for (const auto& op : operations)
	{
		const auto& s = *op.subresource;
		if (op.direct)
		{
			if (plan.codec.depth)
			{
				m_depth.upload(m_context, *plan.codec.depth, guest, {s.offset, s.row_pitch}, texture, op.region);
				++m_upload_statistics.depth_subresources;
			}
			else
			{
				buffer_to_texture(m_context, guest, s.offset, texture, op.region, s.row_pitch, checked_texture_product(s.row_pitch, s.storage_rows));
				++m_upload_statistics.direct_blits;
				m_upload_statistics.direct_bytes += checked_texture_product(checked_texture_product(s.columns, s.rows), checked_texture_product(s.depth, plan.codec.block.bytes));
			}
		}
		else
		{
			std::array<std::uint32_t, 3> log2{};
			if (op.morton) log2 = plan_swizzle({{s.columns, s.rows, s.depth}, {}, {s.columns, s.rows, s.depth}}, plan.codec.block.bytes).log2;
			const auto blocks = checked_texture_product(checked_texture_product(s.columns, s.rows), s.depth);
			constexpr std::size_t max_count = 1u << 30;
			for (std::size_t first = 0; first < blocks;)
			{
				const auto count = std::min(blocks - first, max_count);
				const conversion_parameters params{s.offset, s.row_pitch, checked_texture_product(s.row_pitch, s.storage_rows), first, count,
					static_cast<std::uint32_t>(s.columns), static_cast<std::uint32_t>(s.rows), static_cast<std::uint32_t>(s.depth), static_cast<std::uint32_t>(s.width),
					static_cast<std::uint32_t>(plan.codec.block.bytes), plan.codec.depth ? 1u : plan.codec.endian_width,
					plan.codec.depth ? 0u : plan.codec.conversion, op.morton ? 1u : 0u, log2};
				const auto uniforms = m_context.upload(std::as_bytes(std::span{&params, 1}), 256);
				bindings resources;
				resources.set_buffer(0, guest); resources.set_buffer(1, staging); resources.set_buffer(2, uniforms.buffer, uniforms.offset);
				m_context.keep_alive(m_conversion);
				const auto arguments = resources.snapshot(m_context);
				const auto encoder = m_context.compute();
				m_context.barrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
				[encoder setComputePipelineState:m_conversion]; [encoder setArgumentTable:arguments];
				[encoder dispatchThreads:MTLSizeMake(count, 1, 1)
					threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, m_conversion.maxTotalThreadsPerThreadgroup), 1, 1)];
				++m_upload_statistics.conversion_dispatches; m_upload_statistics.conversion_blocks += count;
				first += count;
			}
			m_upload_statistics.converted_bytes += op.staging_bytes;
			if (plan.codec.depth)
			{
				m_depth.upload(m_context, *plan.codec.depth, staging, {0, s.columns * plan.codec.block.bytes}, texture, op.region);
				++m_upload_statistics.depth_subresources;
			}
			else
			{
				const auto native = transfer_layout(storage_block(texture.pixelFormat), s.width, s.height, s.depth);
				buffer_to_texture(m_context, staging, 0, texture, op.region, native.bytes_per_row, native.bytes_per_image);
			}
		}
		++m_upload_statistics.subresources;
	}
}
guest_texture_resolver::native_cast_input guest_texture_resolver::prepare_cast(const guest_texture_layout& plan, surface_format format,
	id<MTLTexture> source, texture_cache_state metadata, std::size_t row_offset,clear_rectangle rectangle) const
{
	const auto bytes = std::holds_alternative<surface_color>(format) ? color_layout(std::get<surface_color>(format)).pixel_bytes :
		describe_surface_depth(std::get<surface_depth>(format)).guest_bytes;
	const auto& level = plan.subresources.at(0);
	const auto row = checked_texture_product(level.columns, plan.codec.block.bytes);
	const auto rows = level.rows;
	if(source&&!rectangle.width&&!rectangle.height)rectangle={0,0,source.width,source.height};
	const auto packed_row = source ? checked_texture_product(rectangle.width, bytes) : 0;
	if (!source || source.device != m_context.device() || source.sampleCount != 1 || source.textureType != MTLTextureType2D ||
		plan.dimension != vertex_texture_dimension::two || plan.mips != 1 || plan.depth != 1 || plan.layers != 1 ||
		rectangle.x>source.width||rectangle.width>source.width-rectangle.x||rectangle.y>source.height||rectangle.height>source.height-rectangle.y||
		rectangle.height != rows || row_offset >= bytes || row_offset > packed_row || row > packed_row - row_offset)
		throw std::logic_error("Native texture cast guest bytes exceed its source row");
	const auto packed_size = checked_texture_product(packed_row, rows);
	if (packed_size > m_limit) throw std::length_error("Native texture cast staging exceeds snapshot limit");
	// Snapshot already proved the original linear/Morton/segmented addresses.
	// Pack that native rectangle into tight guest-order rows on the GPU,then
	// decode using the target codec. No VM publication or CPU pixel readback.
	auto tight = plan;
	tight.swizzled = false;
	tight.storage_bytes = packed_size;
	tight.subresources = {{row_offset, packed_size - row_offset, packed_row, plan.width, plan.height, 1, level.columns, rows, rows, 0, 0, 0}};
	auto result=native_cast_input{plan, std::move(tight), format, source, metadata, row_offset};result.source_rectangle=rectangle;return result;
}
void guest_texture_resolver::prepare_display_helpers(bool morton,bool holes,bool assembly,bool banked)
{
	if (morton && !m_display_swizzle) m_display_swizzle=std::make_unique<swizzle_engine>(m_context);
	if(banked&&!m_display_banked)m_display_banked=std::make_unique<surface_banked_transfer>(m_context);
	if ((holes && !m_display_hole_scatter) || (assembly && !m_display_storage_copy))
	{
		helper_catalog compile(m_context);const auto library=compile.library(conversion_source,@"Native RSX texture input conversion");
		if (holes && !m_display_hole_scatter) m_display_hole_scatter=compile.compute(library,@"scatter_display_holes");
		if (assembly && !m_display_storage_copy) m_display_storage_copy=compile.compute(library,@"copy_display_storage");
	}
}
guest_texture_resolver::native_cast_input guest_texture_resolver::prepare_morton_display_cast(const guest_texture_layout& plan,
	surface_format format, id<MTLTexture> source, const swizzle_region& layout, std::size_t offset,
	std::vector<display_byte_patch> patches, std::vector<std::byte> patch_bytes)
{
	const auto* color = std::get_if<surface_color>(&format);
	if (!color || !source || source.device != m_context.device() || source.sampleCount != 1 || source.textureType != MTLTextureType2D ||
		plan.dimension != vertex_texture_dimension::two || plan.mips != 1 || plan.depth != 1 || plan.layers != 1 || plan.swizzled ||
		plan.subresources.size() != 1 || plan.codec.block.width != 1 || plan.codec.block.height != 1 ||
		source.width != layout.size[0] || source.height != layout.size[1] || layout.size[2] != 1)
		throw std::invalid_argument("Native Morton display cast requires a complete owned color raster");
	const auto geometry = plan_guest_surface(0, source.width, source.height, color_layout(*color).pixel_bytes, 0, layout);
	const auto display = plan_guest_surface(0, plan.width, plan.height, plan.codec.block.bytes, plan.subresources[0].row_pitch);
	const auto holes = morton_display_holes(geometry, display, offset);
	if (geometry.range.size > m_limit || !holes || holes->size() != patches.size())
		throw std::out_of_range("Native Morton display cast lacks its exact VM byte patches or exceeds budget");
	std::size_t payload = 0;
	for (std::size_t i = 0; i < patches.size(); ++i)
	{
		if (patches[i].target != (*holes)[i].address || patches[i].size != (*holes)[i].size || patches[i].payload != payload)
			throw std::invalid_argument("Mixed Morton display patch differs from missing active bytes");
		payload = checked_texture_sum(payload, patches[i].size);
	}
	if (payload != patch_bytes.size() || payload > m_limit) throw std::invalid_argument("Mixed Morton display payload is incomplete");
	auto packed = plan; packed.storage_bytes = geometry.range.size;
	packed.subresources[0].offset = offset; packed.subresources[0].storage_bytes = geometry.range.size - offset;
	// Compile outside command recording/VM exclusion. The swizzle helper is
	// lazy so linear uploads incur no additional helper/compiler work.
	prepare_display_helpers(true,!patches.empty(),false);
	return {plan, std::move(packed), format, source, {}, 0, layout, std::move(patches), std::move(patch_bytes)};
}
void guest_texture_resolver::encode_cast(const native_cast_input& input, id<MTLTexture> texture)
{
	if(input.assembly){encode_texture_assembly(*input.assembly,texture);return;}
	const auto packed = make_buffer(m_context, input.packed.storage_bytes, buffer_storage::private_gpu);
	texture_region region;
	region.size = MTLSizeMake(input.source.width, input.source.height, 1);
	if(input.source_rectangle.width){region.origin=MTLOriginMake(input.source_rectangle.x,input.source_rectangle.y,0);region.size=MTLSizeMake(input.source_rectangle.width,input.source_rectangle.height,1);}
	const auto packed_row = input.display_morton ? checked_texture_product(input.source.width, color_layout(std::get<surface_color>(input.format)).pixel_bytes) :
		input.packed.subresources[0].row_pitch;
	const auto linear = input.display_morton ? make_buffer(m_context, checked_texture_product(packed_row, input.source.height), buffer_storage::private_gpu) : packed;
	if (const auto color = std::get_if<surface_color>(&input.format))
		m_color.download(m_context, *color, input.source, region, linear, {0, packed_row});
	else
		m_depth.download(m_context, std::get<surface_depth>(input.format), input.source, region, packed, {0, packed_row});
	if (input.display_morton)
		m_display_swizzle->convert(m_context, packed, 0, linear, 0, *input.display_morton,
			color_layout(std::get<surface_color>(input.format)).pixel_bytes, 0, 0, true);
	encode_display_holes(packed,input.display_vm_patches,input.display_vm_bytes);
	encode_upload(input.packed, packed, texture);
}
void guest_texture_resolver::encode_display_holes(id<MTLBuffer> target,std::span<const display_byte_patch> holes,std::span<const std::byte> bytes)
{
	if (!holes.empty())
	{
		static_assert(sizeof(display_byte_patch) == 24);
		const auto payload = m_context.upload(bytes, 256);
		const auto patches = m_context.upload(std::as_bytes(holes), 256);
		const std::array<std::uint64_t, 2> shape{bytes.size(),holes.size()};
		const auto sizes = m_context.upload(std::as_bytes(std::span{shape}), 256);
		bindings resources; resources.set_buffer(0,payload.buffer,payload.offset); resources.set_buffer(1,target);
		resources.set_buffer(2,patches.buffer,patches.offset); resources.set_buffer(3,sizes.buffer,sizes.offset);
		m_context.keep_alive(m_display_hole_scatter); const auto table = resources.snapshot(m_context);
		const auto encoder = m_context.compute(); m_context.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
		[encoder setComputePipelineState:m_display_hole_scatter]; [encoder setArgumentTable:table];
		[encoder dispatchThreads:MTLSizeMake(shape[0],1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,m_display_hole_scatter.maxTotalThreadsPerThreadgroup),1,1)];
	}
}
guest_texture_resolver::display_assembly_input guest_texture_resolver::prepare_display_assembly(const guest_texture_layout& plan,
	std::span<const std::shared_ptr<prepared_surface_snapshot>> native,std::span<const display_storage_source> sources,
	const display_storage_assembly& assembly,std::vector<std::byte> vm_bytes)
{
	if (native.empty() || native.size()!=sources.size() || native.size()>128 || plan.dimension!=vertex_texture_dimension::two ||
		plan.mips!=1 || plan.layers!=1 || plan.depth!=1 || plan.swizzled || plan.subresources.size()!=1 || plan.storage_bytes>m_limit)
		throw std::invalid_argument("Native display assembly has invalid inputs or budget");
	// A banked RTT source can have a logical raster view and a physical byte
	// view. Both retain the same immutable image;the actual image bound stays64.
	std::vector<id<MTLTexture>> images;
	for(const auto& source:native){if(!source||!source->texture())throw std::invalid_argument("Native assembly has a missing source image");if(std::find(images.begin(),images.end(),source->texture())==images.end())images.push_back(source->texture());if(images.size()>64)throw std::length_error("Native assembly exceeds64 distinct images");}
	display_assembly_input result;result.plan=plan;result.vm_bytes=std::move(vm_bytes);
	std::vector<guest_surface_range> target_ranges;
	bool morton=false,banked=false;
	for (std::size_t i=0;i<native.size();++i)
	{
		const auto variant=native[i]->format();const auto image=native[i]->texture();
		const auto& geometry=sources[i].geometry;
		if (!image || image.device!=m_context.device() || image.sampleCount!=1 || image.textureType!=MTLTextureType2D ||
			image.width!=geometry.width || image.height!=geometry.height || geometry.range.address || geometry.pixel_bytes!=surface_guest_bytes(variant))
			throw std::invalid_argument("Display assembly snapshot geometry differs from its byte plan");
		const auto pitch=checked_texture_product(geometry.width,geometry.pixel_bytes),linear=checked_texture_product(pitch,geometry.height);
		const auto tile=sources[i].tile;
		const auto raw=tile?tile->size:geometry.morton?geometry.range.size:linear;
		if (raw>m_limit || linear>m_limit) throw std::length_error("Display assembly source exceeds staging budget");
		result.raw_bytes=std::max(result.raw_bytes,raw);if (geometry.morton&&!tile)result.linear_bytes=std::max(result.linear_bytes,linear);morton|=bool(geometry.morton)&&!tile;banked|=bool(tile);
		display_assembly_native input{variant,image,geometry,{},pitch,tile};
		std::vector<display_storage_copy> selected;
		for (const auto copy:assembly.copies)if(copy.input==i)selected.push_back(copy);
		std::sort(selected.begin(),selected.end(),[](const auto& a,const auto& b){return a.source<b.source;});
		guest_surface_byte_cursor cursor(geometry);const auto bank_runs=tile?banked_display_storage_ranges(geometry,*tile):std::vector<guest_surface_range>{};std::size_t next_bank=0;
		auto active=tile?(bank_runs.empty()?std::optional<guest_surface_range>{}:bank_runs[next_bank++]):cursor.next();
		for (const auto copy:selected)
		{
			while(active && std::uint64_t(active->address)+active->size<=copy.source)active=tile?(next_bank==bank_runs.size()?std::optional<guest_surface_range>{}:bank_runs[next_bank++]):cursor.next();
			if (!copy.size || !active || copy.source<active->address || copy.source-active->address>active->size || copy.size>active->size-(copy.source-active->address))
				throw std::invalid_argument("Display assembly reads inactive source bytes");
		}
		std::size_t ordinal=0;
		for (const auto copy:assembly.copies)if(copy.input==i)
		{
			if(copy.target>plan.storage_bytes || copy.size>plan.storage_bytes-copy.target)throw std::out_of_range("Display assembly target extent differs");
			const auto source=tile||geometry.morton?copy.source:(copy.source/geometry.row_pitch)*pitch+copy.source%geometry.row_pitch;
			if(source>raw || copy.size>raw-source)throw std::out_of_range("Display source packed extent differs");
			input.copies.push_back({copy.target,source,ordinal,copy.size});ordinal=checked_texture_sum(ordinal,copy.size);
			target_ranges.push_back({static_cast<std::uint32_t>(copy.target),copy.size});
		}
		if(input.copies.empty())throw std::invalid_argument("Display assembly retained an unused native source");
		result.native.push_back(std::move(input));
	}
	std::size_t payload=0;
	for (auto hole:assembly.vm){result.vm_patches.push_back({hole.address,payload,hole.size});payload=checked_texture_sum(payload,hole.size);target_ranges.push_back(hole);}
	if(payload!=result.vm_bytes.size())throw std::invalid_argument("Display assembly VM payload differs from its intervals");
	std::sort(target_ranges.begin(),target_ranges.end(),[](auto a,auto b){return a.address<b.address;});
	const auto target=plan_guest_surface(0,plan.subresources[0].columns,plan.subresources[0].rows,plan.codec.block.bytes,plan.subresources[0].row_pitch);
	guest_surface_byte_cursor cursor(target);std::size_t next=0;
	while(auto active=cursor.next())
	{
		std::size_t position=active->address,end=position+active->size;
		while(position<end)
		{
			if(next==target_ranges.size() || !target_ranges[next].size || target_ranges[next].address!=position || target_ranges[next].size>end-position)
				throw std::invalid_argument("Display assembly target has gaps,overlaps orpadding writes");
			position+=target_ranges[next++].size;
		}
	}
	if(next!=target_ranges.size())throw std::invalid_argument("Display assembly writes outside active AV bytes");
	prepare_display_helpers(morton,!assembly.vm.empty(),true,banked);return result;
}
void guest_texture_resolver::encode_display_assembly(const display_assembly_input& input,id<MTLTexture> output)
{
	const auto target=make_buffer(m_context,input.plan.storage_bytes,buffer_storage::private_gpu);
	const auto raw=make_buffer(m_context,input.raw_bytes,buffer_storage::private_gpu);
	const auto linear=input.linear_bytes?make_buffer(m_context,input.linear_bytes,buffer_storage::private_gpu):nil;
	for(const auto& source:input.native)
	{
		texture_region region;region.size=MTLSizeMake(source.geometry.width,source.geometry.height,1);
		if(source.tile)
		{
			auto storage=source.geometry;storage.range={source.tile->address,source.tile->size};
			if(const auto* color=std::get_if<surface_color>(&source.format))m_display_banked->download_color(m_context,*color,source.texture,region,raw,storage,*source.tile);
			else m_display_banked->download_depth(m_context,std::get<surface_depth>(source.format),source.texture,region,raw,storage,*source.tile);
		}
		else
		{
			if(const auto* color=std::get_if<surface_color>(&source.format))m_color.download(m_context,*color,source.texture,region,source.geometry.morton?linear:raw,{0,source.tight_pitch});
			else m_depth.download(m_context,std::get<surface_depth>(source.format),source.texture,region,source.geometry.morton?linear:raw,{0,source.tight_pitch});
			if(source.geometry.morton)m_display_swizzle->convert(m_context,raw,0,linear,0,*source.geometry.morton,source.geometry.pixel_bytes,0,0,true);
		}
		static_assert(sizeof(display_buffer_copy)==32);
		const auto copies=m_context.upload(std::as_bytes(std::span{source.copies}),256);
		const std::array<std::uint64_t,2> shape{source.copies.back().ordinal+source.copies.back().size,source.copies.size()};
		const auto sizes=m_context.upload(std::as_bytes(std::span{shape}),256);
		bindings resources;resources.set_buffer(0,raw);resources.set_buffer(1,target);resources.set_buffer(2,copies.buffer,copies.offset);resources.set_buffer(3,sizes.buffer,sizes.offset);
		m_context.keep_alive(m_display_storage_copy);const auto table=resources.snapshot(m_context);const auto encoder=m_context.compute();
		m_context.barrier(MTLStageDispatch|MTLStageBlit,MTLStageDispatch);[encoder setComputePipelineState:m_display_storage_copy];[encoder setArgumentTable:table];
		[encoder dispatchThreads:MTLSizeMake(shape[0],1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,m_display_storage_copy.maxTotalThreadsPerThreadgroup),1,1)];
	}
	encode_display_holes(target,input.vm_patches,input.vm_bytes);encode_upload(input.plan,target,output);
}
resolved_fragment_texture guest_texture_resolver::cast_surface(const guest_texture_layout& plan, surface_format format,
	id<MTLTexture> source, texture_cache_state metadata, std::size_t row_offset)
{
	const auto input = prepare_cast(plan, format, source, metadata, row_offset);
	const auto texture = allocate(plan);
	m_context.begin();
	try
	{
		encode_cast(input, texture);
		m_context.submit().wait();
	}
	catch (...)
	{
		try { m_context.cancel(); } catch (...) {}
		throw;
	}
	return {texture, metadata};
}

resolved_fragment_texture guest_texture_resolver::upload(const captured_guest_texture& snapshot, id<MTLTexture> texture)
{
	const auto& plan = snapshot.m_plan;
	const auto work = [&]
	{
		m_context.begin();
		try
		{
			const auto guest = make_upload(m_context, snapshot.m_bytes);
			encode_upload(plan, guest, texture);
			return m_context.submit();
		}
		catch (...)
		{
			try
			{
				m_context.cancel();
			}
			catch (...)
			{
			}
			throw;
		}
	}();
	work.wait();
	return {texture, snapshot.m_metadata};
}

struct guest_texture_batch_owner
{
	explicit guest_texture_batch_owner(guest_texture_cache* cache) : cache(cache) {}
	guest_texture_cache* cache;
	std::thread::id thread = std::this_thread::get_id();
	std::uint64_t generation = 1;
	std::atomic_size_t unconsumed = 0;
	bool preparing = false;
};
std::function<bool()> guest_texture_cache::publication_guard() const
{
	const auto owner=m_batch_owner;const auto generation=owner->generation;return [owner,generation,cache=this]{return owner->cache==cache&&owner->generation==generation;};
}
struct versioned_guest_texture_job::state
{
	std::shared_ptr<guest_texture_batch_owner> owner;
	std::uint64_t generation;
	cpu_capture_dispatch capture;
	std::shared_ptr<guest_surface_memory> memory;
	std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate;
	std::vector<versioned_guest_texture> result;
	guest_texture_cache::pending_textures pending;
	std::vector<std::optional<guest_texture_cache::pending_image>> indices;
	std::optional<guest_texture_upload_job> work;
	bool admitted = false, realized = false, published = false, finishing = false;
	~state() { if (admitted) --owner->unconsumed; }
	void check_thread() const
	{
		if (owner->thread != std::this_thread::get_id()) throw std::logic_error("Native versioned texture job belongs to another thread");
	}
	void outside() const
	{
		check_thread();
		if (guest_memory_epoch::active()) throw std::logic_error("Native versioned texture completion requires leaving VM exclusion");
	}
	bool current() const
	{
		for (const auto& image : result) if (!image.revision.current()) return false;
		for (const auto& vm : pending.vm)
		{
			if (texture_storage_segments(vm.input.plan.offset, vm.input.plan.location, vm.input.plan.storage_bytes, translate) != vm.input.segments ||
				vm.input.coverage.size() != vm.revisions.size()) return false;
			for (std::size_t i = 0; i < vm.revisions.size(); ++i)
				if (memory->surface_revision(vm.input.coverage[i]) != vm.revisions[i]) return false;
		}
		return std::all_of(result.begin(), result.end(), [](const auto& image) { return image.revision.current(); });
	}
};
versioned_guest_texture_job::versioned_guest_texture_job(std::shared_ptr<state> state) : m_state(std::move(state)) {}
versioned_guest_texture_job::~versioned_guest_texture_job() = default;
versioned_guest_texture_job::versioned_guest_texture_job(versioned_guest_texture_job&&) noexcept = default;
versioned_guest_texture_job& versioned_guest_texture_job::operator=(versioned_guest_texture_job&&) noexcept = default;
bool versioned_guest_texture_job::ready() const
{
	if (!m_state) throw std::logic_error("Native versioned texture job was moved");
	m_state->check_thread();
	return !m_state->work || m_state->work->ready();
}
bool versioned_guest_texture_job::cancel_pending() const
{
	if (!m_state) throw std::logic_error("Native versioned texture job was moved");
	m_state->outside();
	return m_state->work && m_state->work->cancel_pending();
}
const texture_cache_state& versioned_guest_texture_job::metadata(std::size_t index) const
{
	if (!m_state) throw std::logic_error("Native versioned texture job was moved");
	m_state->check_thread();
	return m_state->result.at(index).image.metadata;
}
bool versioned_guest_texture_job::current() const
{
	if (!m_state) throw std::logic_error("Native versioned texture job was moved");
	m_state->check_thread();
	return m_state->current();
}
std::vector<versioned_guest_texture> versioned_guest_texture_job::finish(std::chrono::milliseconds timeout)
{
	if (!m_state) throw std::logic_error("Native versioned texture job was moved");
	const auto retained = m_state;
	auto& state = *retained;
	state.outside();
	for (const auto& native : state.pending.native) native->validate_outside();
	if (timeout <= std::chrono::milliseconds::zero()) throw std::invalid_argument("Native versioned texture timeout must be positive");
	if (state.finishing) throw std::logic_error("Native versioned texture completion cannot reenter itself");
	state.finishing = true;
	struct finish_scope { bool& active; ~finish_scope() { active = false; } } finish_guard{state.finishing};
	std::exception_ptr failure;
	std::vector<versioned_guest_texture> output;
	@autoreleasepool
	{
		try
		{
			if (!state.realized)
			{
				const auto images = state.work ? state.work->finish(timeout) : std::vector<resolved_fragment_texture>{};
				for (const auto& native : state.pending.native) native->completed();
				for (std::size_t i = 0; i < state.result.size(); ++i)
					if (state.indices[i]) state.result[i].image.texture = images.at(state.indices[i]->index + (state.indices[i]->cast ? state.pending.vm.size() : 0)).texture;
				state.realized = true;
			}
			auto* cache = !state.published && state.owner->generation == state.generation ? state.owner->cache : nullptr;
			std::list<guest_texture_cache::entry> candidates;
			std::list<guest_texture_cache::cast_entry> cast_candidates;
			if (cache)
			{
				for (std::size_t i = 0; i < state.pending.vm.size(); ++i)
				{
					const auto image = std::find_if(state.indices.begin(), state.indices.end(), [&](const auto& index) { return index && !index->cast && index->index == i; });
					const auto& vm = state.pending.vm[i];
					candidates.push_front({vm.input, vm.revisions, state.result.at(std::distance(state.indices.begin(), image)).image, {}});
					candidates.front().physical.reserve(vm.input.coverage.size());
				}
				for (std::size_t i = 0; i < state.pending.casts.size(); ++i)
				{
					const auto& input = state.pending.casts[i];
					if(input.assembly)continue;
					const auto image = std::find_if(state.indices.begin(), state.indices.end(), [&](const auto& index) { return index && index->cast && index->index == i; });
					const auto texture = state.result.at(std::distance(state.indices.begin(), image)).image.texture;
					cast_candidates.push_back({input.source,input.format,input.plan.format,input.plan.width,input.plan.height,input.row_offset,texture.heap.size,texture});
				}
			}
			std::vector<bool> complete(candidates.size(), true);
			const auto validate = [&]
			{
				if (!state.current()) throw texture_revision_conflict();
				std::size_t index = 0;
				for (auto& candidate : candidates)
				{
					for (std::size_t i = 0; i < candidate.input.coverage.size(); ++i)
					{
						try
						{
							auto ticket = state.memory->capture_physical_surface(candidate.input.coverage[i], candidate.revisions[i]);
							if (!ticket) { complete[index] = false; break; }
							candidate.physical.push_back(std::move(ticket));
						}
						catch (const std::length_error&) { complete[index] = false; break; }
					}
					++index;
				}
				if (!state.current()) throw texture_revision_conflict();
			};
			// Empty/disabled groups own only native null images. They have no
			// guest receipts or publication candidates requiring CPU exclusion.
			if (state.capture && !state.result.empty()) checked_cpu_capture(state.capture, validate); else validate();
			std::size_t index = 0;
			for (auto& candidate : candidates) if (!complete[index++]) candidate.physical.clear();
			output = state.result; // Complete caller allocation before mutating cache ownership.
			if (state.owner->cache != cache || state.owner->generation != state.generation) cache = nullptr;
			for (const auto& native : state.pending.native) native->publish();
			if (cache)
			{
				cache->m_stats.uploads += candidates.size();
				cache->m_stats.coalesced_uploads += std::count_if(state.indices.begin(), state.indices.end(), [](const auto& i) { return i && !i->cast; }) - candidates.size();
				cache->m_stats.coalesced_casts += std::count_if(state.indices.begin(), state.indices.end(), [&](const auto& i) { return i && i->cast && !state.pending.casts[i->index].assembly; }) - cast_candidates.size();
				cache->m_entries.splice(cache->m_entries.begin(), candidates);
				while (cache->m_entries.size() > cache->m_capacity) { cache->m_entries.pop_back(); ++cache->m_stats.evictions; }
				cache->m_stats.casts_created += cast_candidates.size();
				while (!cast_candidates.empty())
				{
					const auto first = cast_candidates.begin();
					if (first->bytes > cache->m_resolver.m_limit) { cast_candidates.erase(first); continue; }
					while (cache->m_casts.size() >= cache->m_capacity || cache->m_stats.cast_bytes > cache->m_resolver.m_limit - first->bytes)
						cache->evict_cast(std::prev(cache->m_casts.end()));
					cache->m_stats.cast_bytes += first->bytes;
					cache->m_casts.splice(cache->m_casts.begin(),cast_candidates,first);
				}
				cache->m_stats.cast_entries = cache->m_casts.size();
				for(std::size_t i=0;i<state.pending.casts.size()&&state.owner->cache==cache&&state.owner->generation==state.generation;++i)if(const auto& assembly=state.pending.casts[i].assembly)
				{
					const auto image=std::find_if(state.indices.begin(),state.indices.end(),[&](const auto& p){return p&&p->cast&&p->index==i;});
					cache->publish_assembly(assembly,state.result.at(std::distance(state.indices.begin(),image)).image);
				}
			}
			if(std::any_of(state.pending.casts.begin(),state.pending.casts.end(),[](const auto& cast){return bool(cast.assembly);}))
			{
				// Successful assembled images own their final storage and receipts.
				// A caller retaining the completed job need not pin mutable source
				// allocations,VM payloads or the old command completion owners.
				for(auto& cast:state.pending.casts)if(cast.assembly){cast.assembly->slices.clear();cast.assembly->native.clear();cast.source=nil;}
				state.pending.native.clear();state.work.reset();
			}
			state.published = true;
			if (state.admitted) { --state.owner->unconsumed; state.admitted = false; }
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return output;
}

guest_texture_cache::guest_texture_cache(context& ctx, std::shared_ptr<guest_surface_memory> memory,
										 std::function<std::uint32_t(std::uint32_t, unsigned, std::size_t)> translate, std::size_t maximum_snapshot_bytes,
										 std::shared_ptr<heap_budget> budget, std::size_t capacity, texture_capture_bindings access)
	: m_resolver(ctx, std::move(memory), std::move(translate), maximum_snapshot_bytes, budget, access.capture), m_capacity(capacity), m_access(std::move(access)), m_batch_owner(std::make_shared<guest_texture_batch_owner>(this))
{
	if (bool(m_access.capture) != bool(m_access.io))
		throw std::invalid_argument("Native texture capture and IO epochs must be installed together");
	if (!budget || !capacity)
		throw std::invalid_argument("Guest texture cache needs a shared heap budget and nonzero capacity");
}
guest_texture_cache::~guest_texture_cache()
{
	m_batch_owner->cache = nullptr;
}
bool guest_texture_cache::reclaim_one()
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native texture reclaim cannot run inside VM exclusion");
	if (!m_casts.empty()) { evict_cast(std::prev(m_casts.end())); return true; }
	if (m_entries.empty()) return false;
	m_entries.pop_back();
	++m_stats.evictions;
	return true;
}
void guest_texture_cache::evict_cast(std::list<cast_entry>::iterator it)
{
	m_stats.cast_bytes -= it->bytes;
	m_casts.erase(it); ++m_stats.evictions;
	m_stats.cast_entries = m_casts.size();
}
resolved_fragment_texture guest_texture_cache::resolve(const fragment_texture_registers& words)
{
	return m_access.capture ? resolve_versioned_impl(words, nullptr).image : resolve_impl(words, nullptr);
}
resolved_fragment_texture guest_texture_cache::resolve(const fragment_texture_registers& words, surface_cache& surfaces)
{
	surfaces.validate_texture_adapter(m_resolver.m_context, *m_resolver.m_memory);
	return m_access.capture ? resolve_versioned_impl(words, &surfaces).image : resolve_impl(words, &surfaces);
}
versioned_guest_texture guest_texture_cache::resolve_versioned(const fragment_texture_registers& words, surface_cache& surfaces)
{
	surfaces.validate_texture_adapter(m_resolver.m_context, *m_resolver.m_memory);
	return resolve_versioned_impl(words, &surfaces);
}
versioned_guest_texture_job guest_texture_cache::prepare_versioned_batch(std::span<const fragment_texture_registers> words,
	surface_cache& surfaces, std::span<const guest_texture_upload_dependency> dependencies)
{
	if (m_batch_owner->thread != std::this_thread::get_id()) throw std::logic_error("Native texture batch belongs to another thread");
	if (guest_memory_epoch::active()) throw std::logic_error("Native texture batch preparation requires leaving VM exclusion");
	surfaces.validate_texture_adapter(m_resolver.m_context, *m_resolver.m_memory);
	if (words.size() > guest_texture_resolver::maximum_upload_images || dependencies.size() > 64 || m_batch_owner->unconsumed >= 64)
		throw std::length_error("Native texture cache batch exceeds its input/job bounds");
	for (const auto& dependency : dependencies)
		if (!dependency.event || (dependency.event.device && dependency.event.device != m_resolver.m_context.device()))
			throw std::invalid_argument("Native texture cache batch dependency belongs to another device");
	if (m_resolver.m_context.buffer_ledger() != m_resolver.m_budget)
		throw std::invalid_argument("Native texture cache batch requires one buffer/texture ledger");
	if (m_batch_owner->preparing) throw std::logic_error("Native texture batch preparation cannot reenter itself");
	surfaces.prepare_texture_boundary(!words.empty());
	m_batch_owner->preparing = true;
	struct preparation_scope { bool& active; ~preparation_scope() { active = false; } } preparation_guard{m_batch_owner->preparing};
	auto state = std::make_shared<versioned_guest_texture_job::state>();
	state->owner = m_batch_owner; state->generation = m_batch_owner->generation;
	state->capture = m_access.capture; state->memory = m_resolver.m_memory; state->translate = m_resolver.m_translate;
	state->indices.resize(words.size()); state->result.reserve(words.size());
	state->pending.vm.reserve(words.size()); state->pending.casts.reserve(words.size()); state->pending.native.reserve(words.size());
	std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			for (std::size_t i = 0; i < words.size(); ++i)
				state->result.push_back(resolve_versioned_impl(words[i], &surfaces, &state->pending, &state->indices[i]));
			if (!state->pending.vm.empty() || !state->pending.casts.empty() || !state->pending.native.empty())
			{
				std::vector<captured_guest_texture> captured;
				captured.reserve(state->pending.vm.size());
				for (auto& upload : state->pending.vm) captured.push_back(std::move(upload.snapshot));
				// Leave each failed native pool normally before reclaiming ready
				// owners. No submitted job is retried or discarded on timeout.
				for (;;)
				{
					std::exception_ptr admission_failure;
					@autoreleasepool
					{
						try { state->work.emplace(m_resolver.enqueue_mixed(captured, state->pending.casts, dependencies, state->pending.native)); }
						catch (...) { admission_failure = std::current_exception(); }
					}
					if (!admission_failure) break;
					try { std::rethrow_exception(admission_failure); }
					catch (const heap_budget_error&) { if (!reclaim_one()) throw; }
				}
				++m_stats.upload_batches;
			}
			++m_batch_owner->unconsumed; state->admitted = true;
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return versioned_guest_texture_job(std::move(state));
}
std::vector<versioned_guest_texture> guest_texture_cache::resolve_versioned_batch(std::span<const fragment_texture_registers> words,
	surface_cache& surfaces, std::span<const guest_texture_upload_dependency> dependencies, std::chrono::milliseconds timeout)
{
	if (timeout <= std::chrono::milliseconds::zero()) throw std::invalid_argument("Native texture cache batch timeout must be positive");
	auto job = prepare_versioned_batch(words, surfaces, dependencies);
	return job.finish(timeout);
}
void guest_texture_cache::check_pending_bytes(const pending_textures& pending, std::size_t extra) const
{
	std::size_t total = 0;
	for (const auto& upload : pending.vm) total += upload.snapshot.storage_bytes();
	for (const auto& cast : pending.casts) total += cast.packed.storage_bytes;
	if (extra > guest_texture_resolver::maximum_upload_source_bytes - total)
		throw std::length_error("Native texture cache batch exceeds its combined source byte limit");
}
versioned_guest_texture guest_texture_cache::resolve_versioned_impl(const fragment_texture_registers& words, surface_cache* surfaces,
	pending_textures* pending, std::optional<pending_image>* pending_index)
{
	std::optional<guest_texture_revision> ticket;
	const auto capture = [&](const std::function<void()>& operation)
	{
		if (m_access.capture) checked_cpu_capture(m_access.capture, operation);
		else operation(); // Legacy standalone callers keep their exclusion contract.
	};
	capture([&]
	{
		// Before translation only the logical location is known. A global main
		// stamp brackets initial probing;later checks cover this allocation only.
		const bool main = (words.format & 3) == 2;
		const auto before = m_access.io && main ? m_access.io->current() : 0;
		auto input = m_resolver.prepare(words);
		const auto& plan = input.plan;
		if (input.segments.size() == 1 && plan.dimension == vertex_texture_dimension::two && plan.mips == 1 && plan.codec.block.width == 1)
		{
			const auto morton = plan.swizzled ? std::optional<swizzle_region>{{{plan.width, plan.height, 1}, {}, {plan.width, plan.height, 1}}} : std::nullopt;
			input.coverage[0] = plan_guest_surface(input.segments[0].address, plan.width, plan.height, plan.codec.block.bytes, plan.swizzled ? 0 : plan.subresources[0].row_pitch, morton);
		}
		m_resolver.watch(input);
		std::vector<std::uint64_t> revisions;
		for (const auto& coverage : input.coverage) revisions.push_back(m_resolver.m_memory->surface_revision(coverage));
		texture_io_revision io(m_access.io, plan.location, {plan.offset, plan.storage_bytes});
		ticket = guest_texture_revision(m_resolver.m_memory, input.coverage, input.segments, m_resolver.m_translate,
			plan.offset, plan.storage_bytes, plan.location, std::move(revisions), std::move(io));
		if (m_access.io && main && m_access.io->current() != before) throw texture_revision_conflict();
	});
	// Native aliases keep their precision and original IO/VM tickets. Only a
	// guest-format fallback may write back surfaces and refresh our own tags.
	const auto refresh = [&]
	{
		capture([&]
		{
			if (!ticket->m_io.current() || texture_storage_segments(ticket->m_offset, ticket->m_location, ticket->m_storage_bytes, ticket->m_translate) != ticket->m_segments)
				throw texture_revision_conflict();
			for (std::size_t i = 0; i < ticket->m_coverage.size(); ++i)
				ticket->m_revisions[i] = m_resolver.m_memory->surface_revision(ticket->m_coverage[i]);
			if (!ticket->m_io.current()) throw texture_revision_conflict();
		});
	};
	// Refresh only our pre-gather writeback,never tags after GPU upload.
	// Otherwise a later CPU writer could bless an obsolete uploaded image.
	auto image = resolve_impl(words, surfaces, &ticket->m_surface_revision, refresh, pending, pending_index);
	capture([&] { if (!ticket->current()) throw texture_revision_conflict(); });
	return {std::move(image), std::move(*ticket)};
}
void guest_texture_cache::validate_surface_adapter(context& context, surface_cache& surfaces) const
{
	if (&context != &m_resolver.m_context)
		throw std::invalid_argument("Native draw/texture cache contexts differ");
	surfaces.validate_texture_adapter(context, *m_resolver.m_memory);
}
null_texture_family guest_texture_cache::null_images()
{
	if (m_null_images[0].texture)
		return m_null_images;
	auto& ctx = m_resolver.m_context;
	null_texture_family candidate;
	constexpr std::array types{MTLTextureType2D, MTLTextureType2D, MTLTextureType3D, MTLTextureTypeCube};
	for (unsigned dimension = 0; dimension < 4; ++dimension)
	{
		auto descriptor = [MTLTextureDescriptor new];
		descriptor.textureType = types[dimension];
		descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
		descriptor.width = descriptor.height = descriptor.depth = 1;
		descriptor.mipmapLevelCount = descriptor.arrayLength = descriptor.sampleCount = 1;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead;
		candidate[dimension].texture = m_resolver.m_budget->texture(descriptor);
	}
	const std::array<std::byte, 4> black{};
	ctx.begin();
	try
	{
		const auto source = ctx.upload(black, 4);
		for (unsigned dimension = 0; dimension < 4; ++dimension)
			for (unsigned slice = 0; slice < (dimension == 3 ? 6u : 1u); ++slice)
			{
				texture_region region;
				region.slice = slice;
				buffer_to_texture(ctx, source.buffer, source.offset, candidate[dimension].texture, region);
			}
		ctx.submit().wait();
	}
	catch (...)
	{
		try
		{
			ctx.cancel();
		}
		catch (...)
		{
		}
		throw;
	}
	m_null_images = std::move(candidate);
	++m_stats.null_initializations;
	return m_null_images;
}
resolved_fragment_texture guest_texture_cache::resolve_impl(const fragment_texture_registers& words, surface_cache* surfaces,
	std::shared_ptr<const surface_texture_revision>* source_revision, const std::function<void()>& after_flush,
	pending_textures* pending, std::optional<pending_image>* pending_index)
{
	if (bool(pending) != bool(pending_index)) throw std::logic_error("Native deferred texture resolution requires its pending index");
	if (pending_index) pending_index->reset();
	guest_texture_resolver::source input;
	m_resolver.cpu_capture([&] { input = m_resolver.prepare(words); });
	if (source_revision) source_revision->reset();
	if (surfaces)
	{
		auto coverage = input.coverage;
		const auto& plan = input.plan;
		if (plan.dimension == vertex_texture_dimension::two && plan.mips == 1 && plan.layers == 1 && plan.depth == 1 && (plan.codec.block.width == 1 || !plan.swizzled))
		{
			const auto morton = plan.swizzled ? std::optional<swizzle_region>{{{plan.width, plan.height, 1}, {}, {plan.width, plan.height, 1}}} : std::nullopt;
			auto geometry = plan_guest_surface(0, plan.subresources[0].columns, plan.subresources[0].rows, plan.codec.block.bytes, plan.swizzled ? 0 : plan.subresources[0].row_pitch, morton);
			geometry.range.address = input.segments.front().address;
			coverage.clear();
			for (const auto& part : project_guest_segments(geometry, input.segments, false)) coverage.push_back(part.coverage);
		}
		const auto mapping = [&](const cached_surface_description& description)
		{
			return plan_cached_texture_mapping(*m_resolver.m_memory, description, input.plan, input.segments, {}, true, true);
		};
		id<MTLTexture> texture = nil;
		std::optional<surface_format> source_format;
		std::optional<std::size_t> row_offset;
		for (;;)
		{
			try
			{
				if (pending)
				{
					const auto native = surfaces->prepare_texture_mapping(coverage,mapping,m_resolver.m_budget,false,pending->native,true,true);
					if (native)
					{
						texture = native->texture(); source_format = native->format(); row_offset = native->row_offset();
						if (source_revision) *source_revision = native->revision();
						if (native->needs_copy() && std::find(pending->native.begin(),pending->native.end(),native) == pending->native.end()) pending->native.push_back(native);
					}
				}
				else texture = surfaces->snapshot_texture_mapping(coverage, mapping, m_resolver.m_budget, source_revision, false, &source_format, &row_offset,true);
				break;
			}
			catch (const heap_budget_error&)
			{
				if (m_entries.empty())
					throw;
				m_entries.pop_back();
				++m_stats.evictions;
			}
		}
		if (texture)
		{
			++m_stats.surface_snapshots;
			texture_cache_state metadata;
			metadata.actual_mips = 1;
			metadata.depth = bool(input.plan.codec.depth);
			metadata.source_depth_format = input.plan.codec.depth ? static_cast<std::uint32_t>(*input.plan.codec.depth) : 0;
			metadata.depth_float = input.plan.codec.depth && (*input.plan.codec.depth == surface_depth::z16_float || *input.plan.codec.depth == surface_depth::z24s8_float);
			if (!source_format || !row_offset) throw std::logic_error("Native texture snapshot omitted its selected source metadata");
			const auto channels = surface_texture_channels(*source_format, input.plan);
			const bool direct = channels && !*row_offset && checked_texture_product(texture.width, surface_guest_bytes(*source_format)) == checked_texture_product(input.plan.subresources[0].columns, input.plan.codec.block.bytes);
			metadata.native_argb_channels = direct ? *channels : input.plan.codec.base_argb;
			if ((words.format >> 8) & 0x40)
				metadata.coordinates = normalize_guest_pixels(input.plan.width, input.plan.height, input.plan.depth);
			if (direct) return {texture, metadata};
			for (auto it = m_casts.begin(); it != m_casts.end();)
			{
				if (!it->source) { auto stale = it++; evict_cast(stale); continue; }
				if (it->source == texture && it->format == *source_format && it->target_format == input.plan.format &&
					it->width == input.plan.width && it->height == input.plan.height && it->row_offset == *row_offset)
				{
					const auto result = it->texture;
					m_casts.splice(m_casts.begin(), m_casts, it); ++m_stats.casts_reused;
					return {result, metadata};
				}
				++it;
			}
			if (pending)
			{
				for (std::size_t i = 0; i < pending->casts.size(); ++i)
				{
					const auto& input_cast = pending->casts[i];
					if (input_cast.source == texture && input_cast.format == *source_format && input_cast.plan.format == input.plan.format &&
						input_cast.plan.width == input.plan.width && input_cast.plan.height == input.plan.height && input_cast.row_offset == *row_offset)
					{
						*pending_index = pending_image{i,true};
						return {nil,metadata};
					}
				}
				auto cast = m_resolver.prepare_cast(input.plan,*source_format,texture,metadata,*row_offset);
				check_pending_bytes(*pending,cast.packed.storage_bytes);
				*pending_index = pending_image{pending->casts.size(),true};
				pending->casts.push_back(std::move(cast));
				return {nil,metadata};
			}
			resolved_fragment_texture cast;
			for (;;)
			{
				try { cast = m_resolver.cast_surface(input.plan, *source_format, texture, metadata, *row_offset); break; }
				catch (const heap_budget_error&)
				{
					if (!reclaim_one()) throw;
				}
			}
			++m_stats.casts_created;
			const auto bytes = cast.texture.heap.size;
			if (bytes <= m_resolver.m_limit)
			{
				while (m_casts.size() >= m_capacity || m_stats.cast_bytes > m_resolver.m_limit - bytes)
					evict_cast(std::prev(m_casts.end()));
				m_casts.push_front({texture, *source_format, input.plan.format, input.plan.width, input.plan.height, *row_offset, bytes, cast.texture});
				m_stats.cast_bytes += bytes; m_stats.cast_entries = m_casts.size();
			}
			return cast;
		}
	}
	if(surfaces)if(const auto assembled=resolve_assembly(words,input,*surfaces,source_revision,pending,pending_index))return *assembled;
	if (surfaces && input.segments.size() > 1)
	{
		for (const auto& part : input.segments) surfaces->prepare_cpu_read({part.address, part.size});
		if (after_flush) after_flush();
	}
	m_resolver.cpu_capture([&] { m_resolver.watch(input); });
	const auto same = [&](const guest_texture_resolver::source& candidate)
	{
		const auto& a = candidate.plan;
		const auto& b = input.plan;
		return candidate.segments == input.segments && a.format == b.format && a.width == b.width && a.height == b.height && a.depth == b.depth && a.mips == b.mips && a.layers == b.layers && a.dimension == b.dimension && a.swizzled == b.swizzled && a.subresources[0].row_pitch == b.subresources[0].row_pitch;
	};
	for (auto it = m_entries.begin(); it != m_entries.end(); ++it)
		if (same(it->input))
		{
			if(it->assembly){m_entries.erase(it);++m_stats.invalidations;break;}
			// The immutable resource is keyed by actual storage. Validate the
			// requesting logical mapping,not a previous alias that may be unmapped.
			auto candidate = it->input;
			candidate.plan = input.plan;
			bool valid = false;
			m_resolver.cpu_capture([&] { valid = m_resolver.current(candidate, it->revisions); });
			if (valid)
			{
				m_entries.splice(m_entries.begin(), m_entries, it);
				++m_stats.hits;
				auto result = m_entries.front().resource;
				result.metadata.coordinates = ((words.format >> 8) & 0x40) ? normalize_guest_pixels(input.plan.width, input.plan.height, input.plan.depth) : texture_coordinates{};
				return result;
			}
			m_entries.erase(it);
			++m_stats.invalidations;
			break;
		}
	if (pending)
	{
		for (std::size_t i = 0; i < pending->vm.size(); ++i)
		{
			const auto& candidate = pending->vm[i];
			if (!same(candidate.input)) continue;
			bool valid = false;
			m_resolver.cpu_capture([&] { valid = m_resolver.current(input, candidate.revisions); });
			if (!valid) continue; // The first receipt will reject the whole stale group.
			*pending_index = pending_image{i,false};
			auto metadata = candidate.snapshot.metadata();
			metadata.coordinates = ((words.format >> 8) & 0x40) ? normalize_guest_pixels(input.plan.width, input.plan.height, input.plan.depth) : texture_coordinates{};
			return {nil, metadata};
		}
		check_pending_bytes(*pending,input.plan.storage_bytes);
		std::vector<std::uint64_t> revisions;
		auto snapshot = m_resolver.capture(words, input, revisions);
		const auto metadata = snapshot.metadata();
		*pending_index = pending_image{pending->vm.size(),false};
		pending->vm.push_back({std::move(input), std::move(revisions), std::move(snapshot)});
		return {nil, metadata};
	}
	while (m_entries.size() >= m_capacity)
	{
		m_entries.pop_back();
		++m_stats.evictions;
	}
	std::vector<std::uint64_t> revisions;
	resolved_fragment_texture resource;
	for (;;)
	{
		try
		{
			resource = m_resolver.upload(words, input, revisions);
			break;
		}
		catch (const heap_budget_error&)
		{
			if (m_entries.empty())
				throw;
			m_entries.pop_back();
			++m_stats.evictions;
		}
	}
	std::vector<std::shared_ptr<const physical_surface>> physical;
	// Reserve outside exclusion so appending a captured ticket cannot allocate
	// or release its last backing owner on vector growth failure under VM locks.
	physical.reserve(input.coverage.size());
	bool complete = true;
	m_resolver.cpu_capture([&]
	{
		if (!m_resolver.current(input, revisions)) throw texture_revision_conflict();
		for (std::size_t i = 0; i < input.coverage.size(); ++i)
		{
			try
			{
				auto ticket = m_resolver.m_memory->capture_physical_surface(input.coverage[i], revisions[i]);
				if (!ticket) { complete = false; break; }
				physical.push_back(std::move(ticket));
			}
			catch (const std::length_error&) { complete = false; break; }
		}
		if (!m_resolver.current(input, revisions)) throw texture_revision_conflict();
	});
	// Quota fallback never guesses non-overlap. Release partial backing owners
	// outside the provider epoch; the immutable upload remains revision-valid.
	if (!complete) physical.clear();
	m_entries.push_front({input, std::move(revisions), resource, std::move(physical)});
	++m_stats.uploads;
	return resource;
}
guest_texture_cache_statistics guest_texture_cache::statistics() const noexcept
{
	auto result = m_stats;
	result.entries = m_entries.size();
	return result;
}
void guest_texture_cache::clear()
{
	if (m_batch_owner->thread != std::this_thread::get_id() || guest_memory_epoch::active() || m_batch_owner->preparing)
		throw std::logic_error("Native texture clear requires its owner outside VM exclusion");
	if (m_batch_owner->generation == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Native texture cache generation exhausted");
	++m_batch_owner->generation;
	m_stats.evictions += m_entries.size();
	m_entries.clear();
	while (!m_casts.empty()) evict_cast(std::prev(m_casts.end()));
}
void guest_texture_cache::clear_null_images()
{
	m_null_images = {};
}
void guest_texture_cache::invalidate(guest_surface_range range)
{
	validate_guest_surface_range(range);
	invalidate_active(guest_byte_coverage(range));
}
void guest_texture_cache::invalidate_active(const guest_surface_layout& coverage)
{
	(void)guest_surface_byte_cursor(coverage);
	for (auto it = m_entries.begin(); it != m_entries.end();)
		if (std::any_of(it->input.segments.begin(), it->input.segments.end(), [&](const auto& part)
			{ return memory_storage_overlap(*m_resolver.m_memory, coverage, guest_byte_coverage({part.address, part.size})); }))
		{
			it = m_entries.erase(it);
			++m_stats.invalidations;
		}
		else
			++it;
}
std::size_t guest_texture_cache::invalidate_mapping(guest_surface_range range)
{
	validate_guest_surface_range(range);
	if (guest_memory_epoch::active()) throw std::logic_error("Native texture mapping retirement requires leaving VM exclusion");
	if (!range.size) return 0;
	std::vector<std::list<entry>::iterator> selected;
	m_resolver.cpu_capture([&]
	{
		for (auto it = m_entries.begin(); it != m_entries.end(); ++it)
		{
			bool matches = it->physical.size() != it->input.coverage.size();
			for (const auto& part : it->input.segments)
				if (overlaps(range, guest_surface_range{part.address, part.size})) { matches = true; break; }
			if (!matches)
				for (const auto& ticket : it->physical)
				{
					const auto alias = m_resolver.m_memory->physical_surface_overlaps_current(*ticket, range);
					if (!alias || *alias) { matches = true; break; }
				}
			if (matches) selected.push_back(it);
		}
	});
	// Complete selection first, then release Metal/physical owners outside VM
	// exclusion. A failed metadata query leaves the whole cache retryable.
	std::list<entry> retired;
	for (const auto it : selected) retired.splice(retired.end(), m_entries, it);
	m_stats.invalidations += selected.size();
	return selected.size();
}
std::size_t guest_texture_cache::invalidate_physical_fault(const physical_identity& identity,
	std::span<const std::shared_ptr<void>> registrations)
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native texture fault retirement requires leaving VM exclusion");
	std::vector<std::list<entry>::iterator> selected;
	m_resolver.cpu_capture([&]
	{
		for (auto it = m_entries.begin(); it != m_entries.end(); ++it)
			if (std::any_of(it->input.coverage.begin(), it->input.coverage.end(), [&](const auto& coverage)
				{ return memory_physical_fault_overlap(*m_resolver.m_memory, identity, registrations, coverage); })) selected.push_back(it);
	});
	// The render owner keeps this list stable across the short CPU capture.
	// No driver/resource destructor or last registration runs under VM locks.
	std::list<entry> retired;
	for (const auto it : selected) retired.splice(retired.end(), m_entries, it);
	m_stats.invalidations += selected.size();
	return selected.size();
}
} // namespace metal4
