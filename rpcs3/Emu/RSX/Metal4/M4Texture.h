#pragma once

#include "M4Device.h"
#include "M4TextureLayout.h"

namespace metal4
{

[[nodiscard]] std::size_t max_2d_texture_dimension(id<MTLDevice> device);

enum class texture_plane
{
	all,
	depth,
	stencil,
};

struct texture_region
{
	std::size_t level = 0;
	std::size_t slice = 0;
	MTLOrigin origin = MTLOriginMake(0, 0, 0);
	MTLSize size = MTLSizeMake(1, 1, 1);
};

// Combined depth/stencil transfers require an explicit plane. This describes
// native storage, not guest formats (for example RSX Z16F is not Metal D32F).
[[nodiscard]] texel_block storage_block(MTLPixelFormat format, texture_plane plane = texture_plane::all);
// Validate a copy/transfer region and return the native mip's dimensions.
[[nodiscard]] std::array<std::size_t, 3> validate_texture_region(context& ctx, id<MTLTexture> texture,
																 const texture_region& region, texture_plane plane = texture_plane::all);
[[nodiscard]] id<MTLTexture> make_texture(context& ctx, MTLTextureDescriptor* descriptor);
[[nodiscard]] id<MTLTexture> make_texture_view(id<MTLTexture> texture, MTLPixelFormat format,
											   MTLTextureType type, NSRange levels, NSRange slices, MTLTextureSwizzleChannels swizzle) __attribute__((ns_returns_retained));

struct texture_view_cache_statistics
{
	std::size_t entries = 0, capacity = 0;
	std::uint64_t created = 0, reused = 0, evicted = 0, bypassed = 0;
};

// Render-thread cache of immutable native view descriptors. Keys retain the
// exact source object, including views of views; no raw address can be recycled
// into a false hit. Eviction releases only cache references: argument bindings
// retain views and their backing allocations through command completion.
class texture_view_cache final
{
	struct implementation;
	std::unique_ptr<implementation> m_impl;

  public:
	explicit texture_view_cache(context& ctx, std::size_t capacity = 256);
	~texture_view_cache();
	texture_view_cache(const texture_view_cache&) = delete;
	texture_view_cache& operator=(const texture_view_cache&) = delete;
	[[nodiscard]] id<MTLTexture> get(id<MTLTexture> source, MTLPixelFormat format, MTLTextureType type,
		NSRange levels, NSRange slices, MTLTextureSwizzleChannels swizzle) __attribute__((ns_returns_retained));
	[[nodiscard]] texture_view_cache_statistics statistics() const noexcept;
	void clear();
};

void buffer_to_texture(context& ctx, id<MTLBuffer> source, std::size_t offset,
					   id<MTLTexture> destination, const texture_region& region,
					   std::size_t row_pitch = 0, std::size_t image_pitch = 0, texture_plane plane = texture_plane::all);
// Wait for the submission before reading the resulting shared buffer on the CPU.
void texture_to_buffer(context& ctx, id<MTLTexture> source, const texture_region& region,
					   id<MTLBuffer> destination, std::size_t offset,
					   std::size_t row_pitch = 0, std::size_t image_pitch = 0, texture_plane plane = texture_plane::all);

// Bit-exact native-format region copy; no sampling/filtering/guest conversion.
// Disjoint single-sample allocations with matching mip dimensions use native
// Metal4 texture blits. Aliases/uncertain backing, unequal mip dimensions and
// selected combined D/S planes snapshot before writing, preserving memmove
// semantics for self/view/heap/buffer aliases. Combined depth/stencil supports
// all, depth-only or stencil-only. MSAA all-aspect copies use native Metal4
// texture blits (same full mip dimensions/sample count), with an alias snapshot.
// Selected MSAA aspects require a separate plane path and are not accepted here.
void copy_texture(context& ctx, id<MTLTexture> source, const texture_region& source_region,
				  id<MTLTexture> destination, const texture_region& destination_region, texture_plane plane = texture_plane::all);

} // namespace metal4
