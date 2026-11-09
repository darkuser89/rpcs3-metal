#include "M4Texture.h"
#include "M4Resources.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <bit>
#include <list>
#include <stdexcept>
#include <vector>

namespace metal4
{

std::size_t max_2d_texture_dimension(id<MTLDevice> device)
{
	// Apple10 raises the documented 2D limit from 16K to 32K.
	return [device supportsFamily:MTLGPUFamilyApple10] ? 32768 : 16384;
}

texel_block storage_block(MTLPixelFormat format, texture_plane plane)
{
	if (format == MTLPixelFormatDepth32Float_Stencil8)
	{
		if (plane == texture_plane::depth)
			return {1, 1, 4};
		if (plane == texture_plane::stencil)
			return {1, 1, 1};
		throw std::invalid_argument("Metal 4 combined depth/stencil transfer needs an explicit plane");
	}
	if (plane != texture_plane::all &&
		!(plane == texture_plane::depth && (format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float)) &&
		!(plane == texture_plane::stencil && format == MTLPixelFormatStencil8))
	{
		throw std::invalid_argument("Metal 4 texture does not contain the requested plane");
	}
	switch (format)
	{
	case MTLPixelFormatR8Unorm:
	case MTLPixelFormatR8Unorm_sRGB:
	case MTLPixelFormatR8Snorm:
	case MTLPixelFormatR8Uint:
	case MTLPixelFormatR8Sint:
	case MTLPixelFormatA8Unorm:
	case MTLPixelFormatStencil8:
		return {1, 1, 1};
	case MTLPixelFormatRG8Unorm:
	case MTLPixelFormatRG8Unorm_sRGB:
	case MTLPixelFormatRG8Snorm:
	case MTLPixelFormatRG8Uint:
	case MTLPixelFormatRG8Sint:
	case MTLPixelFormatR16Unorm:
	case MTLPixelFormatR16Snorm:
	case MTLPixelFormatR16Uint:
	case MTLPixelFormatR16Sint:
	case MTLPixelFormatR16Float:
	case MTLPixelFormatDepth16Unorm:
	case MTLPixelFormatB5G6R5Unorm:
	case MTLPixelFormatA1BGR5Unorm:
	case MTLPixelFormatABGR4Unorm:
	case MTLPixelFormatBGR5A1Unorm:
		return {1, 1, 2};
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatRGBA8Unorm_sRGB:
	case MTLPixelFormatRGBA8Snorm:
	case MTLPixelFormatRGBA8Uint:
	case MTLPixelFormatRGBA8Sint:
	case MTLPixelFormatBGRA8Unorm:
	case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatRG16Unorm:
	case MTLPixelFormatRG16Snorm:
	case MTLPixelFormatRG16Uint:
	case MTLPixelFormatRG16Sint:
	case MTLPixelFormatRG16Float:
	case MTLPixelFormatR32Uint:
	case MTLPixelFormatR32Sint:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRGB10A2Unorm:
	case MTLPixelFormatRGB10A2Uint:
	case MTLPixelFormatBGR10A2Unorm:
	case MTLPixelFormatRG11B10Float:
	case MTLPixelFormatRGB9E5Float:
	case MTLPixelFormatDepth32Float:
		return {1, 1, 4};
	case MTLPixelFormatRGBA16Unorm:
	case MTLPixelFormatRGBA16Snorm:
	case MTLPixelFormatRGBA16Uint:
	case MTLPixelFormatRGBA16Sint:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatRG32Uint:
	case MTLPixelFormatRG32Sint:
	case MTLPixelFormatRG32Float:
		return {1, 1, 8};
	case MTLPixelFormatRGBA32Uint:
	case MTLPixelFormatRGBA32Sint:
	case MTLPixelFormatRGBA32Float:
		return {1, 1, 16};
	case MTLPixelFormatBC1_RGBA:
	case MTLPixelFormatBC1_RGBA_sRGB:
	case MTLPixelFormatBC4_RUnorm:
	case MTLPixelFormatBC4_RSnorm:
		return {4, 4, 8};
	case MTLPixelFormatBC2_RGBA:
	case MTLPixelFormatBC2_RGBA_sRGB:
	case MTLPixelFormatBC3_RGBA:
	case MTLPixelFormatBC3_RGBA_sRGB:
	case MTLPixelFormatBC5_RGUnorm:
	case MTLPixelFormatBC5_RGSnorm:
	case MTLPixelFormatBC6H_RGBFloat:
	case MTLPixelFormatBC6H_RGBUfloat:
	case MTLPixelFormatBC7_RGBAUnorm:
	case MTLPixelFormatBC7_RGBAUnorm_sRGB:
		return {4, 4, 16};
	default:
		throw std::invalid_argument("Metal 4 texture transfer format is not implemented");
	}
}

namespace
{

std::size_t slice_count(id<MTLTexture> texture)
{
	switch (texture.textureType)
	{
	case MTLTextureTypeCube:
	case MTLTextureTypeCubeArray:
		return checked_texture_product(texture.arrayLength, 6);
	case MTLTextureType3D:
		return 1;
	default:
		return texture.arrayLength;
	}
}

MTLBlitOption plane_option(id<MTLTexture> texture, texture_plane plane)
{
	if (texture.pixelFormat != MTLPixelFormatDepth32Float_Stencil8)
		return MTLBlitOptionNone;
	return plane == texture_plane::depth ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionStencilFromDepthStencil;
}

texture_buffer_layout validate_transfer(context& ctx, id<MTLTexture> texture,
										const texture_region& region, id<MTLBuffer> buffer, std::size_t offset,
										std::size_t row_pitch, std::size_t image_pitch, texture_plane plane)
{
	if (!texture || !buffer || texture.device != ctx.device() || buffer.device != ctx.device() ||
		texture.framebufferOnly || texture.storageMode == MTLStorageModeMemoryless ||
		texture.sampleCount != 1 || texture.textureType == MTLTextureTypeTextureBuffer ||
		region.level >= texture.mipmapLevelCount || region.slice >= slice_count(texture))
	{
		throw std::invalid_argument("Metal 4 texture transfer resource or subresource is invalid");
	}
	id<MTLTexture> owner = texture;
	while (owner.parentTexture)
		owner = owner.parentTexture;
	if (owner.buffer == buffer)
	{
		throw std::invalid_argument("Metal 4 texture transfer cannot alias its buffer backing");
	}
	const auto mip_dimension = [level = region.level](std::size_t dimension)
	{
		return level >= std::numeric_limits<std::size_t>::digits ? 1 : std::max<std::size_t>(1, dimension >> level);
	};
	const auto width = mip_dimension(texture.width);
	const auto height = mip_dimension(texture.height);
	const auto depth = texture.textureType == MTLTextureType3D ? mip_dimension(texture.depth) : 1;
	if (region.origin.x > width || region.size.width > width - region.origin.x ||
		region.origin.y > height || region.size.height > height - region.origin.y ||
		region.origin.z > depth || region.size.depth > depth - region.origin.z)
	{
		throw std::out_of_range("Metal 4 texture transfer exceeds mip dimensions");
	}
	const auto block = storage_block(texture.pixelFormat, plane);
	if (region.origin.x % block.width || region.origin.y % block.height ||
		(region.size.width % block.width && region.size.width != width - region.origin.x) ||
		(region.size.height % block.height && region.size.height != height - region.origin.y))
	{
		throw std::invalid_argument("Metal 4 compressed transfer must cover complete blocks or the mip edge");
	}
	const auto layout = transfer_layout(block, region.size.width, region.size.height, region.size.depth, row_pitch, image_pitch);
	if (offset % block.bytes || offset > buffer.length || layout.allocation_size > buffer.length - offset)
	{
		throw std::out_of_range("Metal 4 texture transfer buffer range or alignment is invalid");
	}
	return layout;
}

} // namespace

id<MTLTexture> make_texture(context& ctx, MTLTextureDescriptor* descriptor)
{
	if (!descriptor || !descriptor.width || !descriptor.height || !descriptor.depth ||
		!descriptor.arrayLength || !descriptor.mipmapLevelCount || descriptor.storageMode != MTLStorageModePrivate ||
		![ctx.device() supportsTextureSampleCount:descriptor.sampleCount])
	{
		throw std::invalid_argument("Metal 4 persistent textures require a valid private-storage descriptor");
	}
	const auto max_mips = static_cast<NSUInteger>(std::bit_width(std::max({descriptor.width, descriptor.height, descriptor.depth})));
	if (descriptor.mipmapLevelCount > max_mips)
	{
		throw std::invalid_argument("Metal 4 texture has too many mip levels");
	}
	MTLTextureDescriptor* copy = [descriptor copy];
	copy.hazardTrackingMode = MTLHazardTrackingModeUntracked;
	auto texture = [ctx.device() newTextureWithDescriptor:copy];
	if (!texture)
		throw std::runtime_error("Metal 4 texture allocation failed");
	return texture;
}

id<MTLTexture> make_texture_view(id<MTLTexture> texture, MTLPixelFormat format,
								 MTLTextureType type, NSRange levels, NSRange slices, MTLTextureSwizzleChannels swizzle)
{
	if (!texture || !levels.length || levels.location > texture.mipmapLevelCount ||
		levels.length > texture.mipmapLevelCount - levels.location || !slices.length ||
		slices.location > slice_count(texture) || slices.length > slice_count(texture) - slices.location)
	{
		throw std::out_of_range("Metal 4 texture view range is invalid");
	}
	auto view = [texture newTextureViewWithPixelFormat:format textureType:type levels:levels slices:slices swizzle:swizzle];
	if (!view)
		throw std::runtime_error("Metal 4 texture view creation failed");
	return view;
}

struct texture_view_cache::implementation
{
	struct entry
	{
		id<MTLTexture> source, view;
		MTLPixelFormat format;
		MTLTextureType type;
		NSRange levels, slices;
		MTLTextureSwizzleChannels swizzle;
		bool matches(id<MTLTexture> s, MTLPixelFormat f, MTLTextureType t,
			NSRange l, NSRange a, MTLTextureSwizzleChannels c) const noexcept
		{
			return source == s && format == f && type == t && NSEqualRanges(levels, l) && NSEqualRanges(slices, a) &&
				swizzle.red == c.red && swizzle.green == c.green && swizzle.blue == c.blue && swizzle.alpha == c.alpha;
		}
	};
	id<MTLDevice> device;
	std::list<entry> entries;
	texture_view_cache_statistics stats;
};

texture_view_cache::texture_view_cache(context& ctx, std::size_t capacity) : m_impl(std::make_unique<implementation>())
{
	if (capacity > 4096) throw std::invalid_argument("Metal4 texture-view cache capacity exceeds 4096");
	m_impl->device = ctx.device();
	m_impl->stats.capacity = capacity;
}
texture_view_cache::~texture_view_cache() = default;

id<MTLTexture> texture_view_cache::get(id<MTLTexture> source, MTLPixelFormat format, MTLTextureType type,
	NSRange levels, NSRange slices, MTLTextureSwizzleChannels swizzle)
{
	if (!source || source.device != m_impl->device)
		throw std::invalid_argument("Metal4 cached texture view requires a source on its device");
	for (auto it = m_impl->entries.begin(); it != m_impl->entries.end(); ++it)
		if (it->matches(source, format, type, levels, slices, swizzle))
		{
			const auto result = it->view;
			m_impl->entries.splice(m_impl->entries.begin(), m_impl->entries, it);
			++m_impl->stats.reused;
			return result;
		}
	const auto view = make_texture_view(source, format, type, levels, slices, swizzle);
	++m_impl->stats.created;
	if (!m_impl->stats.capacity)
	{
		++m_impl->stats.bypassed;
		return view;
	}
	// Publish only a successfully created view. Invalid requests never evict a
	// working binding, and allocation failure leaves previous entries intact.
	m_impl->entries.push_front({source, view, format, type, levels, slices, swizzle});
	if (m_impl->entries.size() > m_impl->stats.capacity)
	{
		m_impl->entries.pop_back();
		++m_impl->stats.evicted;
	}
	return view;
}
texture_view_cache_statistics texture_view_cache::statistics() const noexcept
{
	auto result = m_impl->stats;
	result.entries = m_impl->entries.size();
	return result;
}
void texture_view_cache::clear()
{
	m_impl->stats.evicted += m_impl->entries.size();
	m_impl->entries.clear();
}

void buffer_to_texture(context& ctx, id<MTLBuffer> source, std::size_t offset,
					   id<MTLTexture> destination, const texture_region& region,
					   std::size_t row_pitch, std::size_t image_pitch, texture_plane plane)
{
	const auto layout = validate_transfer(ctx, destination, region, source, offset, row_pitch, image_pitch, plane);
	auto encoder = ctx.checked_blit({source, offset, layout.allocation_size},
		{destination, 0, 0, true, region.level, region.slice, {region.origin, region.size}});
	[encoder copyFromBuffer:source
			   sourceOffset:offset
		  sourceBytesPerRow:layout.bytes_per_row
		sourceBytesPerImage:layout.bytes_per_image
				 sourceSize:region.size
				  toTexture:destination
		   destinationSlice:region.slice
		   destinationLevel:region.level
		  destinationOrigin:region.origin
					options:plane_option(destination, plane)];
}

void texture_to_buffer(context& ctx, id<MTLTexture> source, const texture_region& region,
					   id<MTLBuffer> destination, std::size_t offset,
					   std::size_t row_pitch, std::size_t image_pitch, texture_plane plane)
{
	const auto layout = validate_transfer(ctx, source, region, destination, offset, row_pitch, image_pitch, plane);
	auto encoder = ctx.checked_blit({source, 0, 0, false, region.level, region.slice, {region.origin, region.size}},
		{destination, offset, layout.allocation_size});
	[encoder copyFromTexture:source
					 sourceSlice:region.slice
					 sourceLevel:region.level
					sourceOrigin:region.origin
					  sourceSize:region.size
						toBuffer:destination
			   destinationOffset:offset
		  destinationBytesPerRow:layout.bytes_per_row
		destinationBytesPerImage:layout.bytes_per_image
						 options:plane_option(source, plane)];
}

std::array<std::size_t, 3> validate_texture_region(context& ctx, id<MTLTexture> texture, const texture_region& region, texture_plane plane)
{
	if (!texture || texture.device != ctx.device() || texture.framebufferOnly || texture.storageMode == MTLStorageModeMemoryless ||
		texture.textureType == MTLTextureTypeTextureBuffer || region.level >= texture.mipmapLevelCount || region.slice >= slice_count(texture) ||
		!region.size.width || !region.size.height || !region.size.depth)
		throw std::invalid_argument("Native texture copy resource/subresource is invalid");
	const auto block = storage_block(texture.pixelFormat, texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && plane == texture_plane::all ? texture_plane::depth : plane);
	const auto mip = [level = region.level](std::size_t size)
	{
		return level >= std::numeric_limits<std::size_t>::digits ? 1 : std::max<std::size_t>(1, size >> level);
	};
	const std::array dimensions{mip(texture.width), mip(texture.height), texture.textureType == MTLTextureType3D ? mip(texture.depth) : std::size_t{1}};
	const std::array origin{region.origin.x, region.origin.y, region.origin.z}, size{region.size.width, region.size.height, region.size.depth};
	for (std::size_t axis = 0; axis < 3; ++axis)
		if (origin[axis] > dimensions[axis] || size[axis] > dimensions[axis] - origin[axis])
			throw std::out_of_range("Native texture copy exceeds mip dimensions");
	if (region.origin.x % block.width || region.origin.y % block.height ||
		(region.size.width % block.width && region.size.width != dimensions[0] - region.origin.x) ||
		(region.size.height % block.height && region.size.height != dimensions[1] - region.origin.y))
		throw std::invalid_argument("Native compressed texture copy must cover whole blocks or mip edges");
	return dimensions;
}
namespace
{
id<MTLTexture> allocation_owner(id<MTLTexture> texture)
{
	while (texture.parentTexture)
		texture = texture.parentTexture;
	return texture;
}
bool texture_allocations_overlap(id<MTLTexture> a, id<MTLTexture> b)
{
	a = allocation_owner(a);
	b = allocation_owner(b);
	// Buffer-backed and IOSurface-backed textures can expose the same storage
	// through distinct native wrappers (including a texture versus a buffer in
	// the same heap). Without an exclusive backing identity, snapshot them.
	if (a == b || a.buffer || b.buffer || a.iosurface || b.iosurface)
		return true;
	if (!a.heap || a.heap != b.heap)
		return false;
	if (!a.allocatedSize || !b.allocatedSize) return true;
	return a.heapOffset <= b.heapOffset ? b.heapOffset - a.heapOffset < a.allocatedSize : a.heapOffset - b.heapOffset < b.allocatedSize;
}
} // namespace

void copy_texture(context& ctx, id<MTLTexture> source, const texture_region& sr, id<MTLTexture> destination, const texture_region& dr, texture_plane plane)
{
	if (!source || !destination || source.pixelFormat != destination.pixelFormat || source.sampleCount != destination.sampleCount ||
		sr.size.width != dr.size.width || sr.size.height != dr.size.height || sr.size.depth != dr.size.depth)
		throw std::invalid_argument("Native texture copy requires matching format/sample count/region sizes");
	const bool combined = source.pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
	const auto sd = validate_texture_region(ctx, source, sr, plane), dd = validate_texture_region(ctx, destination, dr, plane);
	const bool aliased = texture_allocations_overlap(source, destination);
	if (source.sampleCount > 1)
	{
		if (plane != texture_plane::all || source.textureType != MTLTextureType2DMultisample || destination.textureType != MTLTextureType2DMultisample || sd != dd)
			throw std::invalid_argument("Native MSAA bit copy requires all aspects and equal full 2D dimensions");
		id<MTLTexture> readable = source;
		if (aliased)
		{
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat width:source.width height:source.height mipmapped:NO];
			descriptor.textureType = MTLTextureType2DMultisample;
			descriptor.sampleCount = source.sampleCount;
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
			readable = make_texture(ctx, descriptor);
			const auto encoder = ctx.checked_blit({source}, {readable});
			[encoder copyFromTexture:source toTexture:readable];
		}
		const auto encoder = ctx.checked_blit({readable}, {destination});
		[encoder copyFromTexture:readable
					 sourceSlice:sr.slice
					 sourceLevel:sr.level
					sourceOrigin:sr.origin
					  sourceSize:sr.size
					   toTexture:destination
				destinationSlice:dr.slice
				destinationLevel:dr.level
			   destinationOrigin:dr.origin];
		return;
	}
	// The native region blit requires equal mip dimensions. It copies every
	// aspect; selected planes of combined D/S still need their staging path.
	// Parent views and heap placements must be proven disjoint first.
	if (sd == dd && (!combined || plane == texture_plane::all) && !aliased)
	{
		const auto encoder = ctx.checked_blit({source, 0, 0, false, sr.level, sr.slice, {sr.origin, sr.size}},
			{destination, 0, 0, true, dr.level, dr.slice, {dr.origin, dr.size}});
		[encoder copyFromTexture:source
					 sourceSlice:sr.slice
					 sourceLevel:sr.level
					sourceOrigin:sr.origin
					  sourceSize:sr.size
					   toTexture:destination
				destinationSlice:dr.slice
				destinationLevel:dr.level
			   destinationOrigin:dr.origin];
		return;
	}
	struct transfer
	{
		texture_plane plane;
		id<MTLBuffer> buffer;
		texture_buffer_layout layout;
	};
	std::vector<transfer> transfers;
	const auto prepare = [&](texture_plane aspect)
	{
		const auto pixel = storage_block(source.pixelFormat, aspect);
		const auto minimal = transfer_layout(pixel, sr.size.width, sr.size.height, sr.size.depth);
		if (minimal.bytes_per_row > std::numeric_limits<std::size_t>::max() - 255)
			throw std::overflow_error("Native texture copy pitch overflow");
		const auto layout = transfer_layout(pixel, sr.size.width, sr.size.height, sr.size.depth, (minimal.bytes_per_row + 255) & ~std::size_t(255));
		const auto staging = make_buffer(ctx, layout.allocation_size, buffer_storage::private_gpu);
		(void)validate_transfer(ctx, source, sr, staging, 0, layout.bytes_per_row, layout.bytes_per_image, aspect);
		(void)validate_transfer(ctx, destination, dr, staging, 0, layout.bytes_per_row, layout.bytes_per_image, aspect);
		transfers.push_back({aspect, staging, layout});
	};
	if (combined && plane == texture_plane::all)
	{
		prepare(texture_plane::depth);
		prepare(texture_plane::stencil);
	}
	else
		prepare(plane);
	for (const auto& transfer : transfers)
		texture_to_buffer(ctx, source, sr, transfer.buffer, 0, transfer.layout.bytes_per_row, transfer.layout.bytes_per_image, transfer.plane);
	for (const auto& transfer : transfers)
		buffer_to_texture(ctx, transfer.buffer, 0, destination, dr, transfer.layout.bytes_per_row, transfer.layout.bytes_per_image, transfer.plane);
}

} // namespace metal4
