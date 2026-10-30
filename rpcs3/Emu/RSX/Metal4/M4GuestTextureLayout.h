#pragma once

#include "M4GuestSurface.h"
#include "M4TextureState.h"

namespace metal4
{
enum class guest_texture_storage
{
	r8,
	bgr5a1,
	abgr4,
	bgr565,
	bgra8,
	bc1,
	bc2,
	bc3,
	rg8,
	d16,
	d32s8,
	d32,
	r16,
	rg16,
	a1bgr5,
	rg8snorm,
	rgba16f,
	rgba32f,
	r32f,
	rg16f
};
struct guest_texture_format
{
	texel_block block;
	unsigned endian_width;
	guest_texture_storage storage;
	unsigned conversion = 0; // RGB655=1,BR_GR=2,RB_RG=3.
	std::optional<surface_depth> depth{};
	std::array<std::uint32_t, 4> base_argb{3, 0, 1, 2};
};
[[nodiscard]] inline guest_texture_format describe_guest_texture_format(unsigned format)
{
	using S = guest_texture_storage;
	using D = surface_depth;
	switch (format)
	{
	case 0x81:
		return {{1, 1, 1}, 1, S::r8, 0, {}, {5, 0, 0, 0}};
	case 0x82:
		return {{1, 1, 2}, 2, S::bgr5a1};
	case 0x83:
		return {{1, 1, 2}, 2, S::abgr4, 0, {}, {0, 1, 2, 3}};
	case 0x84:
		return {{1, 1, 2}, 2, S::bgr565};
	case 0x85:
		return {{1, 1, 4}, 4, S::bgra8};
	case 0x86:
		return {{4, 4, 8}, 1, S::bc1};
	case 0x87:
		return {{4, 4, 16}, 1, S::bc2};
	case 0x88:
		return {{4, 4, 16}, 1, S::bc3};
	case 0x8b:
		return {{1, 1, 2}, 2, S::rg8, 0, {}, {1, 0, 1, 0}};
	case 0x8d:
		return {{2, 1, 4}, 1, S::bgra8, 2};
	case 0x8e:
		return {{2, 1, 4}, 1, S::bgra8, 3};
	case 0x8f:
		return {{1, 1, 2}, 2, S::bgr565, 1};
	case 0x90:
		return {{1, 1, 4}, 4, S::d32s8, 0, D::z24s8_uint, {0, 0, 0, 0}};
	case 0x91:
		return {{1, 1, 4}, 4, S::d32s8, 0, D::z24s8_float, {0, 0, 0, 0}};
	case 0x92:
		return {{1, 1, 2}, 2, S::d16, 0, D::z16_uint, {0, 0, 0, 0}};
	case 0x93:
		return {{1, 1, 2}, 2, S::d32, 0, D::z16_float, {0, 0, 0, 0}};
	case 0x94:
		return {{1, 1, 2}, 2, S::r16, 0, {}, {0, 5, 0, 5}};
	case 0x95:
		return {{1, 1, 4}, 2, S::rg16, 0, {}, {1, 0, 1, 0}};
	case 0x97:
		return {{1, 1, 2}, 2, S::a1bgr5};
	case 0x98:
		return {{1, 1, 2}, 2, S::rg8, 0, {}, {0, 1, 0, 1}};
	case 0x99:
		return {{1, 1, 2}, 2, S::rg8snorm, 0, {}, {0, 1, 0, 1}};
	case 0x9a:
		return {{1, 1, 8}, 2, S::rgba16f};
	case 0x9b:
		return {{1, 1, 16}, 4, S::rgba32f};
	case 0x9c:
		return {{1, 1, 4}, 4, S::r32f, 0, {}, {0, 0, 0, 0}};
	case 0x9d:
		return {{1, 1, 2}, 2, S::bgr5a1, 0, {}, {5, 0, 1, 2}};
	case 0x9e:
		return {{1, 1, 4}, 4, S::bgra8, 0, {}, {5, 0, 1, 2}};
	case 0x9f:
		return {{1, 1, 4}, 2, S::rg16f, 0, {}, {0, 1, 0, 1}};
	default:
		throw std::invalid_argument("Unknown native guest texture storage format");
	}
}
struct guest_texture_subresource
{
	std::size_t offset, storage_bytes, row_pitch;
	std::size_t width, height, depth, columns, rows, storage_rows;
	unsigned level, face, border;
};
struct guest_texture_layout
{
	unsigned format, location;
	std::uint32_t offset;
	std::size_t width, height, depth, mips, layers, storage_bytes;
	vertex_texture_dimension dimension;
	bool swizzled, stored_border, adjusted_pitch;
	guest_texture_format codec;
	std::vector<guest_texture_subresource> subresources;
};
// RG32Float volume: R keeps native depth precision;G keeps the stencil
// byte as an exact float integer. Ordinary depth sampling consumes R only.
[[nodiscard]] inline bool guest_texture_depth_volume(const guest_texture_layout& plan)
{
	return plan.dimension == vertex_texture_dimension::three && plan.codec.depth.has_value();
}
[[nodiscard]] inline bool guest_texture_vtc(const guest_texture_layout& plan)
{
	return plan.dimension == vertex_texture_dimension::three && plan.codec.block.width == 4;
}
inline std::size_t checked_texture_sum(std::size_t a, std::size_t b)
{
	if (b > std::numeric_limits<std::size_t>::max() - a)
		throw std::overflow_error("Guest texture layout sum overflow");
	return a + b;
}
// NV_texture_compression_vtc Appendix: four consecutive S3TC blocks per
// (Xblock,Yblock,Z/4),then planar one/two/three-slice tail storage. Pitch is
// retained in the row/slice strides;only active blocks are projected.
[[nodiscard]] inline std::size_t guest_texture_block_offset(const guest_texture_layout& plan,
	const guest_texture_subresource& s,std::size_t x,std::size_t y,std::size_t z)
{
	if(x>=s.columns||y>=s.rows||z>=s.depth)throw std::out_of_range("Texture block exceeds subresource");
	const auto pixel=checked_texture_product(x,plan.codec.block.bytes),row=checked_texture_product(y,s.row_pitch),image=checked_texture_product(s.row_pitch,s.storage_rows);
	std::size_t offset;
	if(guest_texture_vtc(plan)&&z<(s.depth&~std::size_t(3)))
		offset=checked_texture_sum(checked_texture_product(z&~std::size_t(3),image),checked_texture_sum(checked_texture_product(4,checked_texture_sum(row,pixel)),checked_texture_product(z&3,plan.codec.block.bytes)));
	else if(plan.swizzled&&plan.codec.block.width==1)
		offset=checked_texture_product(morton_index({x,y,z},plan_swizzle({{s.columns,s.rows,s.depth},{},{s.columns,s.rows,s.depth}},plan.codec.block.bytes).log2),plan.codec.block.bytes);
	else offset=checked_texture_sum(checked_texture_product(z,image),checked_texture_sum(row,pixel));
	if(offset>s.storage_bytes||plan.codec.block.bytes>s.storage_bytes-offset)throw std::out_of_range("Texture block exceeds owned storage");
	return checked_texture_sum(s.offset,offset);
}
[[nodiscard]] inline guest_texture_layout plan_guest_texture(const fragment_texture_registers& words)
{
	if (!(words.control0 >> 31))
		throw std::invalid_argument("Disabled texture has no guest upload layout");
	const unsigned location = words.format & 3, raw_format = (words.format >> 8) & 255, dimension = (words.format >> 4) & 15;
	if (location < 1 || location > 2 || dimension < 1 || dimension > 3 || ((words.format & 4) && dimension != 2))
		throw std::invalid_argument("Invalid RSX texture address location/dimension");
	guest_texture_layout plan{};
	plan.format = raw_format & ~0x60u;
	plan.location = location - 1;
	plan.offset = words.offset & 0x7fffffffu;
	plan.codec = describe_guest_texture_format(plan.format);
	plan.dimension = dimension == 1 ? vertex_texture_dimension::one : dimension == 3 ? vertex_texture_dimension::three
																  : words.format & 4 ? vertex_texture_dimension::cube
																					 : vertex_texture_dimension::two;
	plan.width = words.image_rect >> 16;
	plan.height = dimension == 1 ? 1 : words.image_rect & 65535;
	plan.depth = dimension == 3 ? words.control3 >> 20 : 1;
	plan.layers = plan.dimension == vertex_texture_dimension::cube ? 6 : 1;
	if (!plan.width || !plan.height || !plan.depth)
		throw std::invalid_argument("Zero guest texture layout dimensions");
	const bool dxt = plan.codec.block.width == 4;
	const auto mip_extent = dxt ? std::max<std::size_t>(1, std::min(plan.width / 4, plan.height / 4)) : std::max(plan.width, plan.height);
	plan.mips = std::min<std::size_t>(words.format >> 16, std::bit_width(mip_extent));
	if (!plan.mips)
		throw std::invalid_argument("Guest texture mip count is zero");
	plan.swizzled = !(raw_format & 0x20);
	plan.stored_border = !(words.format & 8) && !dxt;
	std::size_t pitch = words.control3 & 0xfffff;
	const auto packed_pitch = checked_texture_product((plan.width + plan.codec.block.width - 1) / plan.codec.block.width + (!(words.format & 8) ? 2 : 0), plan.codec.block.bytes);
	if (!plan.swizzled && pitch < packed_pitch)
	{
		plan.adjusted_pitch = true;
		if (pitch)
			plan.width = std::max<std::size_t>(1, (pitch / plan.codec.block.bytes) * plan.codec.block.width);
		else
		{
			plan.height = plan.depth = 1;
			pitch = packed_pitch;
		}
	}
	std::size_t position = 0;
	for (unsigned face = 0; face < plan.layers; ++face)
	{
		std::size_t w = plan.width, h = plan.height, d = plan.depth;
		for (unsigned level = 0; level < plan.mips; ++level)
		{
			const auto columns = (w + plan.codec.block.width - 1) / plan.codec.block.width, rows = (h + plan.codec.block.height - 1) / plan.codec.block.height;
			const unsigned border = plan.stored_border ? (plan.swizzled ? 4 : 1) : 0;
			const auto storage_columns = plan.swizzled ? (border ? std::bit_ceil(columns + 8) : columns) : pitch / plan.codec.block.bytes;
			const auto storage_rows = plan.swizzled ? (border ? std::bit_ceil(rows + 8) : rows) : rows + 2 * border;
			const auto row_pitch = plan.swizzled ? checked_texture_product(storage_columns, plan.codec.block.bytes) : pitch;
			const auto bytes = checked_texture_product(checked_texture_product(row_pitch, storage_rows), d);
			plan.subresources.push_back({position, bytes, row_pitch, w, h, d, columns, rows, storage_rows, level, face, border});
			position = checked_texture_sum(position, bytes);
			w = std::max<std::size_t>(1, w / 2);
			h = std::max<std::size_t>(1, h / 2);
			d = std::max<std::size_t>(1, d / 2);
		}
		if (plan.swizzled && face + 1 < plan.layers)
			position = checked_texture_sum(position, 127) & ~std::size_t(127);
	}
	plan.storage_bytes = position;
	if (position > 0x100000000ull - plan.offset)
		throw std::overflow_error("Guest texture offset range wraps RSX address space");
	return plan;
}
} // namespace metal4
