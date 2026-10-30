#include "M4SurfaceBanked.h"
#include "M4SurfaceValidation.h"

#include <algorithm>

#if !__has_feature(objc_arc)
#error "Native banked surface ownership requires ARC."
#endif

namespace metal4
{
namespace
{
struct banked_parameters
{
	std::uint32_t width, height, bytes, base, size, pitch, bank, offset, factor, columns;
	std::uint32_t morton, log_x, log_y, origin_x, origin_y, encode, linear_pitch, element_bytes;
};
static_assert(sizeof(banked_parameters) == 72);
constexpr std::string_view banked_source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct params { uint width,height,bytes,base,size,pitch,bank,offset,factor,columns,morton,log_x,log_y,origin_x,origin_y,encode,linear_pitch,element_bytes; };
kernel void banked_surface(device uchar* tile [[buffer(0)]],device uchar* linear [[buffer(1)]],constant params& p [[buffer(2)]],uint i [[thread_position_in_grid]])
{
    if(ulong(i)>=ulong(p.width)*p.height) return;
    uint x=i%p.width,y=i/p.width;
    const ulong linear_offset=ulong(y)*p.linear_pitch+ulong(x)*p.bytes;
    ulong logical=ulong(y)*p.pitch+ulong(x)*p.bytes;
    if(p.morton) {
        x+=p.origin_x;y+=p.origin_y;ulong index=0;uint output=0;
        for(uint bit=0;bit<max(p.log_x,p.log_y);++bit) {
            if(bit<p.log_x) index|=ulong((x>>bit)&1)<<output++;
            if(bit<p.log_y) index|=ulong((y>>bit)&1)<<output++;
        }
        logical=index*p.bytes;
    }
    logical+=p.offset;
    uint relative=~0u;
    if(logical<p.size) {
        uint address=p.base+uint(logical),texel=uint(logical)/256;
        uint tx=texel%p.columns,ty=(texel/p.columns)/64;
        uint selector=(ty*p.columns+tx+(p.base>>14))&0x3ffff;
        const uint banks[16]={0,1,2,3,2,3,0,1,1,2,3,0,3,0,1,2};
        uint bank=selector&3;
        if(p.factor==2) bank=banks[((selector+((ty&1)<<1))&3)*4+(ty&3)];
        else if(p.factor>=4) bank=banks[(selector&3)*4+(ty&3)];
        bank=(bank+p.bank)&3;
        uint line=(texel/p.columns)%64;
        uint column=(((line>>3)&7)<<7)|(((address>>5)&7)<<4)|((line&3)<<2);
        uint partition=(((line>>2)&1)+((address>>6)&1))&1;
        uint physical=(((selector>>2)&65535)<<16)|(bank<<14)|(((column>>4)&63)<<8)|(partition<<7)|(((column>>2)&3)<<5)|(address&31);
        physical^=(((physical>>12)^((bank^selector)&1)^(physical>>14))&1)<<9;
        physical^=((physical>>11)&1)<<10;
        relative=physical-p.base;
    }
    // The permutation preserves address bits0..4. Aligned1/2/4/8/16-byte
    // texels never cross that32-byte group:one bank/Morton address per texel.
    const bool valid=relative<p.size && p.bytes<=p.size-relative;
    for(uint lane=0;lane<p.bytes;++lane) {
        if(p.encode) { if(valid) tile[relative+lane]=linear[linear_offset+(lane^(p.element_bytes-1))]; }
        else linear[linear_offset+lane]=valid?tile[relative+(lane^(p.element_bytes-1))]:uchar(0);
    }
}
)metal";
} // namespace
surface_banked_transfer::surface_banked_transfer(context& ctx)
	: m_device(ctx.device()), m_depth(ctx)
{
	helper_catalog compile(ctx);
	const auto library = compile.library(banked_source, @"Native banked surface texel transfer");

	m_pipeline = compile.compute(library, @"banked_surface");
}
id<MTLBuffer> surface_banked_transfer::staging(context& ctx, id<MTLBuffer> guest,
	const guest_surface_layout& image, const banked_tile_layout& tile, const texture_region& region, std::size_t pitch) const
{
	validate_banked_tile_geometry(image, tile);
	if (ctx.device() != m_device || !guest || guest.device != m_device || guest.length < tile.size ||
		image.width != region.size.width || image.height != region.size.height || region.size.depth != 1 ||
		(image.morton && (image.morton->storage[2] != 1 || image.morton->origin[2])))
		throw std::invalid_argument("Native banked surface device,storage or view is invalid");
	const auto bytes = checked_texture_product(pitch ? pitch : checked_texture_product(image.width, image.pixel_bytes), image.height);
	if (bytes > std::numeric_limits<std::uint32_t>::max())
		throw std::length_error("Native banked surface dispatch exceeds its byte index");
	const auto buffer = make_buffer(ctx, bytes, buffer_storage::private_gpu);
	++m_statistics.staging_buffers; m_statistics.staging_bytes += std::max(buffer.length, buffer.allocatedSize);
	return buffer;
}
void surface_banked_transfer::convert(context& ctx, const guest_surface_layout& image, const banked_tile_layout& tile,
	id<MTLBuffer> guest, id<MTLBuffer> linear, bool encode, std::size_t pitch, std::uint32_t element_bytes) const
{
	banked_parameters params{static_cast<std::uint32_t>(image.width), static_cast<std::uint32_t>(image.height), static_cast<std::uint32_t>(image.pixel_bytes),
		tile.address, tile.size, tile.pitch, tile.bank, tile.offset, banked_tile_factor(tile.pitch), tile.pitch / 256, bool(image.morton), 0, 0, 0, 0, encode,
		static_cast<std::uint32_t>(pitch ? pitch : checked_texture_product(image.width, image.pixel_bytes)), element_bytes};
	if (image.morton)
	{
		const auto plan = plan_swizzle(*image.morton, image.pixel_bytes);
		params.log_x = plan.log2[0]; params.log_y = plan.log2[1];
		params.origin_x = image.morton->origin[0]; params.origin_y = image.morton->origin[1];
	}
	bindings resources;
	resources.set_buffer(0, guest); resources.set_buffer(1, linear);
	const auto uniforms = ctx.upload(std::as_bytes(std::span(&params, 1)), 256);
	resources.set_buffer(2, uniforms.buffer, uniforms.offset);
	ctx.keep_alive(m_pipeline);
	const auto arguments = resources.snapshot(ctx);
	const auto encoder = ctx.compute();
	ctx.barrier(MTLStageBlit | MTLStageDispatch, MTLStageDispatch);
	[encoder setComputePipelineState:m_pipeline];
	[encoder setArgumentTable:arguments];
	const auto count = checked_texture_product(image.width, image.height);
	[encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, m_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
	++m_statistics.dispatches; m_statistics.texels += count;
	m_statistics.active_bytes += checked_texture_product(count, image.pixel_bytes);
}
void surface_banked_transfer::upload_color(context& ctx, surface_color format, id<MTLBuffer> guest, const guest_surface_layout& image,
	const banked_tile_layout& tile, id<MTLTexture> texture, const texture_region& region) const
{
	const auto metadata = describe_surface_color(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	if (image.pixel_bytes != metadata.layout.pixel_bytes) throw std::invalid_argument("Banked color byte width differs from its format");
	const auto pitch = (checked_texture_product(image.width, image.pixel_bytes) + 255) & ~std::size_t(255);
	const auto linear = staging(ctx, guest, image, tile, region, pitch);
	convert(ctx, image, tile, guest, linear, false, pitch, metadata.layout.endian_element_bytes);
	buffer_to_texture(ctx, linear, 0, texture, region, pitch);
	++m_statistics.fused_color_uploads;
}
void surface_banked_transfer::download_color(context& ctx, surface_color format, id<MTLTexture> texture, const texture_region& region,
	id<MTLBuffer> guest, const guest_surface_layout& image, const banked_tile_layout& tile) const
{
	const auto metadata = describe_surface_color(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	if (image.pixel_bytes != metadata.layout.pixel_bytes) throw std::invalid_argument("Banked color byte width differs from its format");
	const auto pitch = (checked_texture_product(image.width, image.pixel_bytes) + 255) & ~std::size_t(255);
	const auto linear = staging(ctx, guest, image, tile, region, pitch);
	texture_to_buffer(ctx, texture, region, linear, 0, pitch);
	convert(ctx, image, tile, guest, linear, true, pitch, metadata.layout.endian_element_bytes);
	++m_statistics.fused_color_downloads;
}
void surface_banked_transfer::upload_depth(context& ctx, surface_depth format, id<MTLBuffer> guest, const guest_surface_layout& image,
	const banked_tile_layout& tile, id<MTLTexture> texture, const texture_region& region) const
{
	const auto metadata = describe_surface_depth(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	if (image.pixel_bytes != metadata.guest_bytes) throw std::invalid_argument("Banked depth byte width differs from its format");
	const auto linear = staging(ctx, guest, image, tile, region);
	convert(ctx, image, tile, guest, linear, false);
	m_depth.upload(ctx, format, linear, {}, texture, region);
}
void surface_banked_transfer::download_depth(context& ctx, surface_depth format, id<MTLTexture> texture, const texture_region& region,
	id<MTLBuffer> guest, const guest_surface_layout& image, const banked_tile_layout& tile) const
{
	const auto metadata = describe_surface_depth(format);
	validate_surface_texture(ctx, metadata.format, texture, region);
	if (image.pixel_bytes != metadata.guest_bytes) throw std::invalid_argument("Banked depth byte width differs from its format");
	const auto linear = staging(ctx, guest, image, tile, region);
	m_depth.download(ctx, format, texture, region, linear, {});
	convert(ctx, image, tile, guest, linear, true);
}
} // namespace metal4
