#include "M4SurfaceCache.h"
#include "M4GuestMemoryOverlap.h"
#include "M4PhysicalSurface.h"
#include "M4ExclusiveGuestMemory.h"
#include "M4ScaledImageNativePlan.h"
#include "M4RectangleCoverage.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <unordered_map>
#include <unordered_set>

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

namespace metal4
{
namespace
{
// Every part owns a real VM watch. Image geometry is deliberately separate;
// bank permutation consumes the gathered flat allocation, never a fake EA.
struct surface_storage
{
	std::vector<guest_storage_part> parts;
	std::size_t size = 0;
};
using physical_tickets = std::vector<std::shared_ptr<const physical_surface>>;
struct surface_storage_snapshot
{
	std::vector<std::uint64_t> revisions;
	std::vector<std::byte> bytes;
};
bool active_storage_overlap(bool direct, const guest_surface_memory::storage_lookup& lookup,
	const surface_storage& a, const surface_storage& b)
{
	for (const auto& ap : a.parts)
		for (const auto& bp : b.parts)
			if (metal4::active_storage_overlap(direct, lookup, ap.coverage, bp.coverage)) return true;
	return false;
}
bool active_storage_overlap(bool direct, const guest_surface_memory::storage_lookup& lookup,
	const guest_surface_layout& a, const surface_storage& b)
{
	for (const auto& bp : b.parts)
		if (metal4::active_storage_overlap(direct, lookup, a, bp.coverage)) return true;
	return false;
}
template <typename A>
bool memory_storage_overlap(guest_surface_memory& memory, const A& a, const surface_storage& b)
{
	return memory.test_storage([&](bool direct, const auto& lookup)
		{ return active_storage_overlap(direct, lookup, a, b); });
}
bool equivalent_storage(guest_surface_memory& memory, const surface_storage& a, const surface_storage& b)
{
	if (a.parts.size() != b.parts.size()) return false;
	return memory.test_rebase([&](const auto&, const guest_surface_memory::storage_equivalent& equal)
	{
		for (std::size_t i = 0; i < a.parts.size(); ++i)
			if (a.parts[i].offset != b.parts[i].offset || !equal(a.parts[i].coverage.range, b.parts[i].coverage.range)) return false;
		return true;
	});
}
struct raw_plane
{
	texture_plane aspect;
	std::size_t offset, pitch;
};
struct raw_surface_snapshot
{
	id<MTLBuffer> buffer = nil;
	std::vector<raw_plane> planes;
};
void copy_spill_rectangle(context& ctx, const raw_surface_snapshot& spill, std::size_t spill_height,
	clear_rectangle input, id<MTLTexture> storage, texture_region target, texture_plane selected = texture_plane::all)
{
	for (const auto& plane : spill.planes)
	{
		if (selected != texture_plane::all && selected != plane.aspect) continue;
		const auto bytes = storage_block(storage.pixelFormat, plane.aspect).bytes;
		const auto offset = plane.offset + input.y * plane.pitch + input.x * bytes;
		auto rows = target;
		if (input.x && input.y + input.height == spill_height)
		{
			// The regular transfer validates a full stride for every row.
			// A tight last row keeps its validated range inside this plane.
			rows.size.height -= 1;
			if (rows.size.height)
				buffer_to_texture(ctx, spill.buffer, offset, storage, rows, plane.pitch, 0, plane.aspect);
			rows.origin.y += rows.size.height;
			const auto last = offset + rows.size.height * plane.pitch;
			rows.size.height = 1;
			buffer_to_texture(ctx, spill.buffer, last, storage, rows, 0, 0, plane.aspect);
		}
		else
			buffer_to_texture(ctx, spill.buffer, offset, storage, rows, plane.pitch, 0, plane.aspect);
	}
}
struct deferred_copy_section
{
	surface_lease source;
	std::uint64_t generation;
	std::uint64_t write_epoch;
	clear_rectangle rectangle;
	clear_rectangle destination;
	texture_plane plane;
	std::array<bool, 3> flip;
	bool typeless;
};
class surface_list_guard
{
	std::vector<cached_surface_ref>& list;
	const std::size_t size;

  public:
	surface_list_guard(std::vector<cached_surface_ref>& list, std::span<const cached_surface_ref> surfaces)
		: list(list), size(list.size()) { list.insert(list.end(), surfaces.begin(), surfaces.end()); }
	surface_list_guard(std::vector<cached_surface_ref>& list, const cached_surface_ref& surface)
		: list(list), size(list.size())
	{
		list.push_back(surface);
	}
	~surface_list_guard()
	{
		list.resize(size);
	}
};
bool same(const swizzle_region& a, const swizzle_region& b)
{
	return a.storage == b.storage && a.origin == b.origin && a.size == b.size;
}
bool same(const cached_surface_description& a, const cached_surface_description& b)
{
	return a.address == b.address && a.allocation == b.allocation && a.row_pitch == b.row_pitch &&
		   bool(a.morton) == bool(b.morton) && (!a.morton || same(*a.morton, *b.morton)) && a.tile == b.tile && a.segments == b.segments &&
		bool(a.mapping) == bool(b.mapping) && (!a.mapping || a.mapping->same_epoch(*b.mapping));
}
std::size_t descriptor_hash(const cached_surface_description& value)
{
	std::size_t result = 0;
	const auto add = [&](std::size_t word) { result ^= word + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2); };
	add(value.address); add(value.row_pitch);
	const auto& allocation = value.allocation;
	add(allocation.width); add(allocation.height); add(allocation.samples); add(allocation.expanded);
	add(allocation.format.index()); std::visit([&](auto format) { add(static_cast<std::size_t>(format)); }, allocation.format);
	add(bool(value.morton));
	if (value.morton)
		for (const auto& axis : {value.morton->storage, value.morton->origin, value.morton->size})
			for (const auto word : axis) add(word);
	add(bool(value.tile));
	if (value.tile)
	{
		add(value.tile->address); add(value.tile->size); add(value.tile->pitch);
		add(value.tile->bank); add(value.tile->offset);
	}
	for (const auto& part : value.segments) { add(part.address); add(part.offset); add(part.size); }
	add(bool(value.mapping)); if (value.mapping) add(value.mapping->epoch_hash());
	return result;
}
std::size_t pixel_bytes(const surface_allocation_description& a)
{
	if (const auto color = std::get_if<surface_color>(&a.format))
		return color_layout(*color).pixel_bytes;
	return describe_surface_depth(std::get<surface_depth>(a.format)).guest_bytes;
}
MTLPixelFormat native_format(const surface_allocation_description& a)
{
	if (const auto color = std::get_if<surface_color>(&a.format))
		return describe_surface_color(*color).format;
	return describe_surface_depth(std::get<surface_depth>(a.format)).format;
}
bool supports_native_spill_import(MTLPixelFormat format)
{
	// Color imports retain their storage type through the fragment export.
	// Float depth still requires an alternative to the validated raster path.
	switch (format)
	{
	case MTLPixelFormatR8Unorm:
	case MTLPixelFormatRG8Unorm:
	case MTLPixelFormatBGRA8Unorm:
	case MTLPixelFormatRGBA8Unorm:
	case MTLPixelFormatBGR5A1Unorm:
	case MTLPixelFormatB5G6R5Unorm:
	case MTLPixelFormatR32Float:
	case MTLPixelFormatRGBA16Float:
	case MTLPixelFormatRGBA32Float:
	case MTLPixelFormatDepth16Unorm:
		return true;
	default:
		return false;
	}
}
void validate_surface_blit(const surface_allocation_description& a, clear_rectangle rectangle,
						   const surface_allocation_description& b, clear_rectangle destination, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	if (!typeless && a.format != b.format)
		throw std::invalid_argument("Cached nearest copy requires equal guest format");
	if (rectangle.x > a.width || rectangle.width > a.width - rectangle.x || rectangle.y > a.height || rectangle.height > a.height - rectangle.y ||
		destination.x > b.width || destination.width > b.width - destination.x || destination.y > b.height || destination.height > b.height - destination.y)
		throw std::out_of_range("Cached native copy exceeds surface dimensions");
	if (flip[2])
		throw std::invalid_argument("Cached 2D blit cannot flip a volume axis");
	if (typeless)
	{
		if (plane != texture_plane::all)
			throw std::invalid_argument("Cached typeless transfers require all guest aspects");
		const auto source_size = a.samples == 1 ? std::array{rectangle.width, rectangle.height} : sample_expanded_size(rectangle.width, rectangle.height, a.samples);
		const auto target_size = b.samples == 1 ? std::array{destination.width, destination.height} : sample_expanded_size(destination.width, destination.height, b.samples);
		(void)plan_typeless(a.format, b.format, source_size[0], source_size[1], target_size[0], target_size[1]);
		return;
	}
	(void)storage_block(native_format(a), native_format(a) == MTLPixelFormatDepth32Float_Stencil8 && plane == texture_plane::all ? texture_plane::depth : plane);
}
} // namespace

class cached_surface final
{
  public:
	cached_surface_description description;
	guest_surface_layout layout;
	surface_storage storage;
	surface_lease lease;
	// Independent physical VM mapping,not a writer/VM lock. Keep dirty and
	// deferred output storage alive until writeback,discard or cache teardown.
	std::shared_ptr<void> backing;
	std::shared_ptr<physical_tickets> physical;
	raw_surface_snapshot spill;
	std::vector<deferred_copy_section> deferred;
	std::vector<std::uint64_t> deferred_revision;
	std::uint64_t generation = 0;
	bool initialized = false, dirty = false, faulted = false;
	std::vector<std::uint64_t> revision;
	std::uint64_t last_use = 0;
	// Zero identifies VM-decoded data. Successful native writes publish a
	// cache-wide epoch;flush and bit-exact spill/restore retain that provenance.
	std::uint64_t native_write_epoch = 0;
	void reset_backing() noexcept { backing.reset(); physical.reset(); }
};

namespace
{
std::uint64_t snapshot_epoch(const cached_surface& surface)
{
	return surface.deferred.empty() ? surface.native_write_epoch : surface.deferred.back().write_epoch;
}

const std::vector<std::uint64_t>& snapshot_versions(const cached_surface& surface)
{
	return surface.deferred.empty() ? surface.revision : surface.deferred_revision;
}

struct deferred_snapshot_input
{
	surface_lease allocation;
	clear_rectangle rectangle;
};

struct deferred_snapshot_piece
{
	surface_lease allocation;
	sample_raster_projection projection;
	clear_rectangle destination;
	texture_plane plane;
	bool typeless = false;
	bool guest = false;
};

// Copies retain the compiled codecs so queued snapshots remain independent
// of the surface cache. Only visible guest holes are decoded, directly into
// the single-sample snapshot; no destination MSAA allocation is needed.
struct snapshot_guest_transfer
{
	surface_color_transfer color;
	surface_depth_transfer depth;
	surface_swizzle_transfer swizzle;
	surface_banked_transfer banked;
	transfer_engine rows;
	void upload(context& ctx, const cached_surface_description& description, guest_surface_layout layout,
		id<MTLBuffer> buffer, clear_rectangle input, id<MTLTexture> texture, const texture_region& output,
		texture_plane plane) const
	{
		layout.width = input.width; layout.height = input.height;
		if (layout.morton)
		{
			layout.morton->origin[0] += input.x;
			layout.morton->origin[1] += input.y;
			layout.morton->size = {input.width, input.height, 1};
		}
		auto tile = description.tile;
		if (tile && !layout.morton)
			tile->offset += input.y * tile->pitch + input.x * layout.pixel_bytes;
		surface_byte_region bytes{input.y * layout.row_pitch + input.x * layout.pixel_bytes, layout.row_pitch};
		if (!tile && !layout.morton && (bytes.row_pitch % layout.pixel_bytes ||
			checked_texture_product(bytes.row_pitch, input.height) > buffer.length - bytes.offset))
		{
			const auto row = checked_texture_product(input.width, layout.pixel_bytes);
			const auto packed = make_buffer(ctx, checked_texture_product(row, input.height), buffer_storage::private_gpu);
			for (std::size_t y = 0; y < input.height; ++y)
				rows.copy(ctx, buffer, bytes.offset + y * bytes.row_pitch, packed, y * row, row);
			buffer = packed; bytes = {0, row};
		}
		if (const auto format = std::get_if<surface_color>(&description.allocation.format))
		{
			if (tile) banked.upload_color(ctx, *format, buffer, layout, *tile, texture, output);
			else if (layout.morton) swizzle.upload_color(ctx, *format, buffer, 0, *layout.morton, texture, output);
			else color.upload(ctx, *format, buffer, bytes, texture, output);
		}
		else
		{
			const auto depth_format = std::get<surface_depth>(description.allocation.format);
			if (tile) banked.upload_depth(ctx, depth_format, buffer, layout, *tile, texture, output, plane);
			else if (layout.morton) swizzle.upload_depth(ctx, depth_format, buffer, 0, *layout.morton, texture, output, plane);
			else depth.upload(ctx, depth_format, buffer, bytes, texture, output, plane);
		}
	}
};

bool projectable_deferred_copy(const deferred_copy_section& copy, const surface_allocation_description& target)
{
	return !copy.flip[2] && (copy.typeless || copy.source->description().format == target.format);
}

bool exact_deferred_copy(const deferred_copy_section& copy, const surface_allocation_description& target)
{
	const auto& donor = copy.source->description();
	return projectable_deferred_copy(copy, target) && copy.plane == texture_plane::all &&
		donor.format == target.format &&
		!copy.flip[0] && !copy.flip[1] && donor.samples == target.samples &&
		copy.rectangle.width == copy.destination.width && copy.rectangle.height == copy.destination.height;
}

bool same_snapshot_donor(const std::weak_ptr<surface_allocation>& a, const std::weak_ptr<surface_allocation>& b)
{
	return !a.owner_before(b) && !b.owner_before(a);
}

// When the last intersecting copy supplies every requested sample, older
// sections and the destination background cannot contribute to this image.
std::optional<deferred_snapshot_input> project_deferred_snapshot(const cached_surface& surface, clear_rectangle requested)
{
	if (surface.deferred.empty() || surface.faulted) return {};
	const auto& target = surface.description.allocation;
	const auto factors = target.samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(target.samples);
	const auto requested_right = requested.x + requested.width, requested_bottom = requested.y + requested.height;
	for (auto it = surface.deferred.rbegin(); it != surface.deferred.rend(); ++it)
	{
		const auto& copy = *it;
		const auto x = copy.destination.x * factors.x, y = copy.destination.y * factors.y;
		const auto right = x + copy.destination.width * factors.x, bottom = y + copy.destination.height * factors.y;
		if (requested_right <= x || right <= requested.x || requested_bottom <= y || bottom <= requested.y)
			continue;
		if (requested.x < x || requested.y < y || requested_right > right || requested_bottom > bottom ||
			!exact_deferred_copy(copy, target))
			return {};
		requested.x = copy.rectangle.x * factors.x + requested.x - x;
		requested.y = copy.rectangle.y * factors.y + requested.y - y;
		return deferred_snapshot_input{copy.source, requested};
	}
	return {};
}

constexpr std::size_t deferred_piece_limit = 64;

bool compose_deferred_plane(const cached_surface& surface, clear_rectangle requested, texture_plane plane,
	std::vector<deferred_snapshot_piece>& pieces, bool& used_copy)
{
	const auto& target = surface.description.allocation;
	const auto factors = target.samples == 1 ? sample_layout{1,1} : rsx_sample_layout(target.samples);
	std::vector<clear_rectangle> uncovered{requested}, remaining;
	const auto add_piece = [&](const surface_lease& input, sample_raster_projection projection, clear_rectangle visible,
		bool typeless = false, bool guest = false)
	{
		visible.x -= requested.x;
		visible.y -= requested.y;
		pieces.push_back({input, projection, visible, plane, typeless, guest});
	};
	for (auto it = surface.deferred.rbegin(); it != surface.deferred.rend() && !uncovered.empty(); ++it)
	{
		const auto& copy = *it;
		if (plane != texture_plane::all && copy.plane != texture_plane::all && copy.plane != plane) continue;
		const auto x = copy.destination.x * factors.x, y = copy.destination.y * factors.y;
		const auto right = x + copy.destination.width * factors.x, bottom = y + copy.destination.height * factors.y;
		remaining.clear();
		for (const auto& hole : uncovered)
		{
			const auto hole_right = hole.x + hole.width, hole_bottom = hole.y + hole.height;
			const auto left = std::max(hole.x, x), top = std::max(hole.y, y);
			const auto end_x = std::min(hole_right, right), end_y = std::min(hole_bottom, bottom);
			if (left >= end_x || top >= end_y)
				remaining.push_back(hole);
			else
			{
				// A newer section already owns all samples outside the holes.
				if (!projectable_deferred_copy(copy, target)) return false;
				used_copy = true;
				const auto samples = copy.source->description().samples;
				const auto input = samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(samples);
				add_piece(copy.source, {{copy.rectangle.x * input.x, copy.rectangle.y * input.y},
					{copy.rectangle.width * input.x, copy.rectangle.height * input.y}, {left - x, top - y},
					{right - x, bottom - y}, {copy.flip[0], copy.flip[1]}},
					{left, top, end_x - left, end_y - top}, copy.typeless && copy.source->description().format != target.format);
				if (top > hole.y) remaining.push_back({hole.x, hole.y, hole.width, top - hole.y});
				if (end_y < hole_bottom) remaining.push_back({hole.x, end_y, hole.width, hole_bottom - end_y});
				if (left > hole.x) remaining.push_back({hole.x, top, left - hole.x, end_y - top});
				if (end_x < hole_right) remaining.push_back({end_x, top, hole_right - end_x, end_y - top});
			}
			if (pieces.size() + remaining.size() > deferred_piece_limit) return false;
		}
		uncovered.swap(remaining);
	}
	const bool guest = !surface.initialized || (!surface.lease && !surface.spill.buffer) ||
		surface.revision != surface.deferred_revision;
	if (!uncovered.empty() && guest && surface.dirty) return false;
	if (pieces.size() + uncovered.size() > deferred_piece_limit) return false;
	const auto background = guest || surface.spill.buffer ? surface_lease{} : surface.lease;
	for (const auto& hole : uncovered)
		add_piece(background, {{hole.x, hole.y}, {hole.width, hole.height}, {}, {hole.width, hole.height}}, hole, false, guest);
	return true;
}

std::vector<deferred_snapshot_piece> compose_deferred_snapshot(const cached_surface& surface, clear_rectangle requested, bool require_copy = true)
{
	if (surface.faulted || surface.deferred.empty() || surface.deferred.size() > deferred_piece_limit) return {};
	std::vector<deferred_snapshot_piece> pieces;
	bool used_copy = false;
	const bool split = native_format(surface.description.allocation) == MTLPixelFormatDepth32Float_Stencil8 &&
		std::ranges::any_of(surface.deferred, [](const auto& copy) { return copy.plane != texture_plane::all; });
	if (split)
	{
		if (!compose_deferred_plane(surface, requested, texture_plane::depth, pieces, used_copy) ||
			!compose_deferred_plane(surface, requested, texture_plane::stencil, pieces, used_copy)) return {};
	}
	else if (!compose_deferred_plane(surface, requested, texture_plane::all, pieces, used_copy)) return {};
	if (require_copy && !used_copy) return {};
	return pieces;
}

bool complete_deferred_input(const cached_surface_ref& surface)
{
	if (surface->deferred.empty() || surface->deferred.size() > rectangle_coverage_limit)
		return false;
	std::vector<clear_rectangle> rectangles;
	for (const auto& section : surface->deferred)
		if (section.plane == texture_plane::all)
			rectangles.push_back(section.destination);
	const auto& allocation = surface->description.allocation;
	return rectangles_cover(rectangles, allocation.width, allocation.height);
}

std::pair<std::size_t, bool> deferred_copy_start(const cached_surface_ref& surface)
{
	const auto& allocation = surface->description.allocation;
	for (auto i = surface->deferred.size(); i != 0; --i)
	{
		const auto& section = surface->deferred[i - 1];
		if (section.plane == texture_plane::all && section.destination.x == 0 && section.destination.y == 0 &&
			section.destination.width == allocation.width && section.destination.height == allocation.height)
			return {i - 1, true};
	}
	return {0, complete_deferred_input(surface)};
}

bool current_deferred_background(const cached_surface& surface)
{
	return surface.initialized && (surface.lease || surface.spill.buffer) &&
		surface.revision == surface.deferred_revision;
}

void encode_surface_background(context& ctx, const surface_format& format, id<MTLTexture> texture)
{
	MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
	if (std::holds_alternative<surface_color>(format))
	{
		pass.colorAttachments[0].texture = texture;
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
	}
	else
	{
		pass.depthAttachment.texture = texture;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.depthAttachment.clearDepth = 1;
		if (describe_surface_depth(std::get<surface_depth>(format)).stencil)
		{
			pass.stencilAttachment.texture = texture;
			pass.stencilAttachment.loadAction = MTLLoadActionClear;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
			pass.stencilAttachment.clearStencil = 255;
		}
	}
	(void)ctx.render(pass);
	ctx.end_encoding();
}

display_storage_epoch_run deferred_input_epoch(const cached_surface& surface, const guest_surface_layout& geometry,
	std::span<const texture_storage_run> inverse, std::size_t offset, std::size_t size, std::size_t& checks)
{
	const auto end = std::upper_bound(inverse.begin(), inverse.end(), offset,
		[](auto byte, const auto& run) { return byte < run.physical; });
	if (end == inverse.begin() || offset - std::prev(end)->physical >= std::prev(end)->size)
		throw std::logic_error("Deferred authority selects inactive native storage");
	const auto& run = *std::prev(end);
	const auto logical = run.logical + offset - run.physical;
	const auto lane = (geometry.morton ? logical : logical % geometry.row_pitch) % geometry.pixel_bytes;
	const auto pixel = texture_storage_pixel(geometry, logical - lane);
	if (!pixel) throw std::logic_error("Deferred authority byte has no native texel");

	const auto& allocation = surface.description.allocation;
	const auto samples = allocation.samples == 1 ? sample_layout{1, 1} : rsx_sample_layout(allocation.samples);
	const auto x = (*pixel)[0] / samples.x, y = (*pixel)[1] / samples.y;
	const auto* depth = std::get_if<surface_depth>(&allocation.format);
	const bool has_stencil = depth && (*depth == surface_depth::z24s8_uint || *depth == surface_depth::z24s8_float);
	const auto plane = has_stencil && lane == 3 ? texture_plane::stencil : texture_plane::depth;
	const auto end_lane = has_stencil && lane < 3 ? 3 : geometry.pixel_bytes;
	auto next_edge = allocation.width;
	const auto run_bytes = [&](std::size_t right)
	{
		auto bytes = end_lane - lane;
		if (!geometry.morton && !has_stencil)
			bytes = (right * samples.x - (*pixel)[0]) * geometry.pixel_bytes - lane;
		return std::min({size, run.size - (offset - run.physical), bytes});
	};
	for (auto it = surface.deferred.rbegin(); it != surface.deferred.rend(); ++it)
	{
		if (++checks > (1u << 22)) throw std::length_error("Deferred authority exceeds rectangle check budget");
		const auto& rect = it->destination;
		if (it->plane != texture_plane::all && (!depth || it->plane != plane)) continue;
		if (y < rect.y || y - rect.y >= rect.height) continue;
		if (x < rect.x)
		{
			next_edge = std::min(next_edge, rect.x);
			continue;
		}
		if (x - rect.x >= rect.width) continue;

		// Later sections can interrupt this epoch at their left edge.
		return {run_bytes(std::min(next_edge, rect.x + rect.width)), it->write_epoch, true};
	}
	// The queued plan owns only the bytes its sections write. Uncovered
	// bytes retain the old native authority, if its revision still matches.
	const auto epoch = current_deferred_background(surface) ? std::optional{surface.native_write_epoch} : std::nullopt;
	if (surface.dirty && !epoch)
		throw std::logic_error("Deferred input lost its dirty native background");
	return {run_bytes(next_edge), epoch, surface.dirty};
}
}

surface_conflict_error::surface_conflict_error(guest_surface_range range)
	: std::runtime_error("Guest memory changed while native surface data was dirty; serialize VM access before writeback"), range(range)
{
}

struct surface_validation
{
	cached_surface_ref surface;
	std::vector<std::uint64_t> revisions;
	bool physical = false;
};

struct draw_target_load
{
	cached_surface_ref surface;
	id<MTLBuffer> guest = nil;
	raw_surface_snapshot spill;
	surface_lease background;
	std::vector<deferred_copy_section> sections;

	[[nodiscard]] bool has_input() const { return guest || spill.buffer || background || !sections.empty(); }
	[[nodiscard]] unsigned generations() const { return unsigned(bool(guest)) + unsigned(!sections.empty()); }
};
using staged_target_load = std::pair<draw_target_load, surface_validation>;
struct prefetched_target_data
{
	surface_storage_snapshot snapshot;
	id<MTLBuffer> upload = nil;
	std::vector<deferred_copy_section> sections;
};
constexpr std::size_t target_prefetch_limit = 16u << 20;
struct target_reload
{
	surface_cache& owner;
	std::optional<staged_target_load> staged;
};

struct surface_backing
{
	cached_surface_ref surface;
	std::shared_ptr<void> owner;
	std::shared_ptr<physical_tickets> physical;
};

// Keep replaced GPU/storage owners until CPU publication has left the VM
// transaction. Last lease release may enter the pool/driver and must not run
// while guest writers and mappings are excluded.
struct surface_retirement
{
	cached_surface_ref surface;
	surface_lease allocation;
	id<MTLBuffer> spill;
	std::shared_ptr<void> backing;
	std::shared_ptr<physical_tickets> physical;
	std::vector<surface_lease> deferred;
};

struct surface_cache::impl
{
	context& ctx;
	const std::thread::id owner_thread = std::this_thread::get_id();
	std::shared_ptr<guest_surface_memory> memory;
	const std::size_t snapshot_limit;
	cpu_capture_dispatch capture;
	std::shared_ptr<heap_budget> shared_ledger;
	surface_pool pool;
	surface_color_transfer color;
	surface_depth_transfer depth;
	surface_swizzle_transfer swizzle;
	surface_banked_transfer banked;
	std::shared_ptr<sample_transfer> samples;
	std::shared_ptr<surface_typeless_transfer> typeless;
	std::shared_ptr<texture_blitter> frozen_blits;
	std::shared_ptr<snapshot_guest_transfer> snapshot_guest;
	clear_engine clears;
	surface_sample_blitter sample_blits;
	transfer_engine row_transfers;
	std::unique_ptr<depth_bounds_engine> bounds;
	std::vector<cached_surface_ref> surfaces;
	// These indexes do not own surfaces, watches or native allocations.
	std::unordered_map<const cached_surface*, std::size_t> positions;
	std::unordered_multimap<std::size_t, cached_surface*> descriptors;
	struct range_node
	{
		std::uint64_t begin, end, maximum;
		cached_surface* surface;
	};
	std::vector<range_node> ranges;
	bool ranges_dirty = true;
	surface_metadata_statistics metadata;
	surface_transaction_statistics transactions;
	struct snapshot_entry
	{
		std::weak_ptr<cached_surface> source;
		std::uint64_t generation, native_epoch;
		std::vector<std::uint64_t> versions;
		std::shared_ptr<heap_budget> budget;
		clear_rectangle rectangle;
		std::size_t row_offset, bytes;
		id<MTLTexture> texture;
		std::shared_ptr<const surface_texture_revision> revision;
		std::weak_ptr<surface_allocation> native_donor;
		struct view { MTLPixelFormat format = MTLPixelFormatInvalid; id<MTLTexture> texture = nil; };
		std::array<view, 4> views{};
	};
	std::list<snapshot_entry> snapshots;
	surface_snapshot_statistics snapshot_stats;
	std::shared_ptr<void> snapshot_lifetime = std::make_shared<int>(0);
	std::uint64_t snapshot_generation = 1;
	void evict_snapshot(std::list<snapshot_entry>::iterator it)
	{
		snapshot_stats.bytes -= it->bytes;
		snapshots.erase(it); ++snapshot_stats.evicted;
		snapshot_stats.entries = snapshots.size();
	}
	void prune_snapshots()
	{
		for (auto it = snapshots.begin(); it != snapshots.end();)
		{
			const auto source = it->source.lock();
			if (!source || source->faulted || source->generation != it->generation ||
				snapshot_epoch(*source) != it->native_epoch)
			{ auto stale = it++; evict_snapshot(stale); }
			else ++it;
		}
	}
	std::vector<cached_surface_ref> pinned;
	std::vector<cached_surface_ref> resolving;
	std::uint64_t clock = 0;
	std::uint64_t native_clock = 0;
	// Deferred/cancelled ownership can change a former VM hole without leaving
	// a materialized native source for a later ready-image lookup to discover.
	std::uint64_t texture_topology_epoch = 0;
	[[nodiscard]] std::uint64_t reserve_native_epoch()
	{
		if (native_clock == std::numeric_limits<std::uint64_t>::max())
			throw std::overflow_error("Native surface write epoch exhausted");
		return ++native_clock;
	}
	std::chrono::milliseconds timeout = std::chrono::seconds(30);
	struct rollback_surface
	{
		cached_surface_ref surface;
		cached_surface previous;
		raw_surface_snapshot snapshot;
		bool copy = false;
		surface_lease output;
	};
	struct native_rollback
	{
		std::vector<rollback_surface> surfaces;
		std::optional<submission> backup_work;
		bool preserve_contents = true;
		// A multi-draw commit can invalidate aliases before a later callback
		// fails. Retain their exact pre-batch validity until publication ends.
		std::vector<std::pair<cached_surface_ref, cached_surface>> aliases;
	};
	struct recorded_draw_batch;
	struct transaction
	{
		submission work;
		std::function<void()> commit;
		std::vector<cached_surface_ref> mutated;
		std::vector<surface_validation> validated;
		std::vector<surface_backing> backings;
		std::vector<surface_retirement> retirements;
		bool trim_pool = false;
		bool native_output = false;
		std::shared_ptr<native_rollback> rollback;
		draw_publication publish_draw;
		std::shared_ptr<const recorded_draw_batch> draw_batch{};
	};
	std::optional<transaction> pending;
	struct recorded_draw_batch
	{
		std::optional<submission> recording;
		std::shared_ptr<native_rollback> rollback = std::make_shared<native_rollback>();
		std::vector<std::function<void()>> commits;
		std::vector<cached_surface_ref> mutated;
		std::vector<surface_validation> validated;
		std::vector<surface_backing> backings;
		std::vector<surface_retirement> retirements;
	};
	std::size_t batch_capacity = 0;
	predicated_draw_completion predicated_completion = predicated_draw_completion::synchronous;
	std::shared_ptr<void> batch_lifetime = std::make_shared<int>(0);
	std::shared_ptr<recorded_draw_batch> recording_batch;
	impl(context& ctx, std::shared_ptr<guest_surface_memory> memory, std::size_t budget, std::shared_ptr<heap_budget> shared_budget, cpu_capture_dispatch capture, surface_metadata_limits limits)
		: ctx(ctx), memory(std::move(memory)), snapshot_limit(budget), capture(std::move(capture)), shared_ledger(shared_budget), pool(ctx, budget, std::move(shared_budget)), color(ctx), depth(ctx), swizzle(ctx), banked(ctx), samples(std::make_shared<sample_transfer>(ctx)), clears(ctx), sample_blits(ctx), row_transfers(ctx)
	{
		if (!this->memory)
			throw std::invalid_argument("Native surface cache requires a VM memory adapter");
		if (!limits.entries || !limits.watch_parts || limits.entries > 65536 || limits.watch_parts > (1u << 20))
			throw std::invalid_argument("Native surface metadata limits are invalid");
		metadata.limits = limits;
	}
	inline static thread_local const impl* captured = nullptr;
	struct capture_scope
	{
		const impl* previous;
		explicit capture_scope(const impl* owner) : previous(std::exchange(captured, owner)) {}
		~capture_scope() { captured = previous; }
	};
	void cpu(const std::function<void()>& operation, const cpu_capture_dispatch& writer_capture = {}) const
	{
		if (captured == this) { operation(); return; }
		const auto invoke = [&] { capture_scope scope(this); operation(); };
		const auto& dispatch = writer_capture ? writer_capture : capture;
		if (dispatch) checked_cpu_capture(dispatch, invoke);
		else invoke(); // Legacy single-watch callers provide writer exclusion.
	}
	void outside_epoch() const
	{
		if (guest_memory_epoch::active() || captured)
			throw std::logic_error("Native surface retirement cannot run inside CPU/VM exclusion");
	}
	void rebuild_ranges()
	{
		if (!ranges_dirty) return;
		ranges.clear();
		for (const auto& surface : surfaces)
			for (const auto& part : surface->storage.parts)
				ranges.push_back({part.coverage.range.address, std::uint64_t(part.coverage.range.address) + part.coverage.range.size, 0, surface.get()});
		std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.begin < b.begin; });
		// A static balanced interval tree stored in sorted order. Every middle
		// element owns the maximum end of its complete implicit subtree.
		const auto build = [&](auto&& self, std::size_t first, std::size_t last) -> std::uint64_t
		{
			if (first == last) return 0;
			const auto middle = first + (last - first) / 2;
			return ranges[middle].maximum = std::max({ranges[middle].end, self(self, first, middle), self(self, middle + 1, last)});
		};
		build(build, 0, ranges.size());
		ranges_dirty = false;
	}
	std::vector<cached_surface_ref> range_candidates(std::span<const guest_surface_range> requested)
	{
		rebuild_ranges();
		std::vector<bool> selected(surfaces.size());
		const auto query = [&](auto&& self, std::size_t first, std::size_t last, guest_surface_range range) -> void
		{
			if (first == last) return;
			const auto middle = first + (last - first) / 2;
			const auto& node = ranges[middle];
			++metadata.range_nodes;
			const auto end = std::uint64_t(range.address) + range.size;
			if (node.maximum <= range.address || ranges[first].begin >= end) return;
			self(self, first, middle, range);
			if (node.begin < end && range.address < node.end) selected[positions.at(node.surface)] = true;
			if (node.begin < end) self(self, middle + 1, last, range);
		};
		for (const auto range : requested) { ++metadata.range_queries; if (range.size) query(query, 0, ranges.size(), range); }
		std::vector<cached_surface_ref> result;
		for (std::size_t index = 0; index < selected.size(); ++index)
			if (selected[index]) result.push_back(surfaces[index]);
		return result;
	}
	std::vector<cached_surface_ref> candidates(bool direct, std::span<const guest_surface_range> requested)
	{
		if (direct) return range_candidates(requested);
		// Physical aliases need not share any virtual range. Never discard them
		// using an EA-only index; this fallback is bounded by the metadata cap.
		metadata.physical_candidates += surfaces.size();
		return surfaces;
	}
	void remove(const cached_surface_ref& surface)
	{
		outside_epoch();
		const auto index = positions.at(surface.get());
		const auto bucket = descriptors.equal_range(descriptor_hash(surface->description));
		for (auto i = bucket.first; i != bucket.second; ++i)
			if (i->second == surface.get()) { descriptors.erase(i); break; }
		positions.erase(surface.get());
		metadata.watch_parts -= surface->storage.parts.size();
		// Invalidate raw pointers before the final owner can be released.
		ranges.clear(); ranges_dirty = true;
		surfaces.erase(surfaces.begin() + index);
		for (auto i = index; i < surfaces.size(); ++i) positions.at(surfaces[i].get()) = i;
		metadata.entries = surfaces.size();
		++metadata.retired;
	}
	bool retirable(const cached_surface_ref& surface)
	{
		++metadata.retirement_visits;
		if (surface.use_count() != 1 || surface->dirty || !surface->deferred.empty() || surface->faulted ||
			(surface->lease && surface->lease.use_count() != 1)) return false;
		// Pending transactions/pins/resolving lists own shared handles too.
		if (!surface->initialized || !surface->native_write_epoch) return true;
		// Clean native precision can serve RTT after writeback. Keep it until
		// its mapping or active-byte revision is positively known to be stale.
		bool stale = false;
		cpu([&]
		{
			if (surface->description.mapping && !surface->description.mapping->current()) { stale = true; return; }
			for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
				if (memory->surface_revision(surface->storage.parts[i].coverage) != surface->revision[i]) { stale = true; break; }
		});
		return stale;
	}
	std::size_t retire_to(std::size_t target_entries, std::size_t target_parts)
	{
		outside_epoch();
		if (surfaces.size() <= target_entries && metadata.watch_parts <= target_parts) return 0;
		std::vector<cached_surface*> ordered;
		for (const auto& surface : surfaces) ordered.push_back(surface.get());
		std::stable_sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) { return a->last_use < b->last_use; });
		const auto before = surfaces.size();
		for (const auto* candidate : ordered)
		{
			if (surfaces.size() <= target_entries && metadata.watch_parts <= target_parts) break;
			const auto& surface = surfaces[positions.at(candidate)];
			if (retirable(surface)) remove(surface);
		}
		return before - surfaces.size();
	}
	bool reclaim_clean(const cached_surface_ref& protected_surface = {})
	{
		outside_epoch();
		if (pending || recording_batch) return false;
		cached_surface_ref oldest;
		for (const auto& surface : surfaces)
		{
			if (surface == protected_surface || (!surface->lease && !surface->spill.buffer) || surface->dirty || surface->faulted || !surface->deferred.empty() ||
				(surface->initialized && surface->native_write_epoch) || (surface->lease && surface->lease.use_count() != 1) ||
				std::find(pinned.begin(), pinned.end(), surface) != pinned.end() ||
				std::find(resolving.begin(), resolving.end(), surface) != resolving.end()) continue;
			if (!oldest || surface->last_use < oldest->last_use) oldest = surface;
		}
		if (!oldest) return false;
		oldest->lease.reset();
		oldest->spill = {};
		oldest->reset_backing();
		oldest->initialized = false;
		oldest->native_write_epoch = 0;
		(void)pool.trim();
		return true;
	}
	void reserve_metadata(std::size_t parts)
	{
		if (parts > metadata.limits.watch_parts) throw surface_metadata_budget_error();
		retire_to(metadata.limits.entries - 1, metadata.limits.watch_parts - parts);
		if (surfaces.size() >= metadata.limits.entries || metadata.watch_parts > metadata.limits.watch_parts - parts)
			throw surface_metadata_budget_error();
	}
	static void check_mapping(const cached_surface_ref& surface)
	{
		if (surface->description.mapping && !surface->description.mapping->current()) throw surface_mapping_conflict();
	}
	void validate_in_epoch(std::span<const surface_validation> values) const
	{
		for (const auto& value : values)
		{
			const auto& surface = value.surface;
			if (value.physical)
			{
				if (!surface->physical || surface->physical->size() != surface->storage.parts.size())
					throw std::logic_error("Native surface lacks complete physical tickets");
				for (const auto& ticket : *surface->physical)
					if (!memory->physical_surface_current(*ticket)) throw surface_conflict_error(ticket->coverage().range);
				continue;
			}
			check_mapping(surface);
			if (value.revisions.size() != surface->storage.parts.size()) throw std::logic_error("Surface validation lacks watch versions");
			for (std::size_t i = 0; i < value.revisions.size(); ++i)
				if (memory->surface_revision(surface->storage.parts[i].coverage) != value.revisions[i])
					throw surface_conflict_error(surface->storage.parts[i].coverage.range);
			check_mapping(surface);
		}
	}
	void validate_mappings(std::span<const surface_validation> values) const
	{
		if (!values.empty()) cpu([&] { validate_in_epoch(values); });
	}
	void validate_completion(const transaction& completed) const
	{
		const auto retained_inputs_current = [&]
		{
			// A successful native write belongs to the physical inputs captured
			// before submission. A later VA/IO/tile change cannot turn that valid
			// output into a GPU failure. CPU changes to those physical bytes still
			// refuse;uploads and unknown/failed GPU status never take this path.
			if (!completed.native_output || completed.backings.empty()) return false;
			for (const auto& input : completed.backings)
				if (!input.physical || input.physical->size() != input.surface->storage.parts.size()) return false;
			for (const auto& input : completed.backings)
				for (const auto& ticket : *input.physical)
					if (!memory->physical_surface_current(*ticket)) throw surface_conflict_error(ticket->coverage().range);
			return true;
		};
		// Virtual access and IO/tile policy were checked immediately before
		// submission. Requiring those VAs again after success would lose valid
		// old-allocation output on unmap/protection as well as on remap.
		if (!retained_inputs_current()) validate_in_epoch(completed.validated);
	}
	std::vector<surface_backing> retain_backings(std::span<const surface_validation> values) const
	{
		std::vector<surface_backing> result;
		if (values.empty()) return result;
		result.reserve(values.size());
		cpu([&]
		{
			validate_in_epoch(values);
			for (const auto& value : values)
			{
				if (std::any_of(result.begin(), result.end(), [&](const auto& retained) { return retained.surface == value.surface; })) continue;
				if (value.physical)
				{
					result.push_back({value.surface, value.surface->backing, value.surface->physical});
					continue;
				}
				auto tickets = std::make_shared<physical_tickets>();
				tickets->reserve(value.surface->storage.parts.size());
				std::vector<guest_surface_range> ranges;
				for (const auto& part : value.surface->storage.parts)
				{
					guest_surface_byte_cursor cursor(part.coverage);
					while (const auto range = cursor.next())
					{
						if (ranges.size() == (1u << 20)) throw std::length_error("Surface backing capture exceeds active-range budget");
						ranges.push_back(*range);
					}
				}
				for (std::size_t i = 0; i < value.surface->storage.parts.size(); ++i)
				{
					const auto ticket = memory->capture_physical_surface(value.surface->storage.parts[i].coverage, value.revisions[i]);
					if (ticket) tickets->push_back(ticket);
				}
				if (!tickets->empty() && tickets->size() != value.surface->storage.parts.size())
					throw std::runtime_error("Native surface has only partial physical publication support");
				result.push_back({value.surface, memory->retain_backing(ranges), tickets->empty() ? nullptr : std::move(tickets)});
			}
			validate_in_epoch(values);
		});
		return result;
	}
	static void retain_mutated_backings(const transaction& value) noexcept
	{
		for (const auto& surface : value.mutated)
			for (const auto& retained : value.backings)
				if (retained.surface == surface) { surface->backing = retained.owner; surface->physical = retained.physical; break; }
	}
	static std::vector<surface_retirement> retain_metadata_storage(std::span<const surface_validation> validated,
		std::span<const cached_surface_ref> mutated, std::span<const cached_surface_ref> aliases)
	{
		std::vector<surface_retirement> result;
		const auto retain = [&](const cached_surface_ref& surface)
		{
			if (std::any_of(result.begin(), result.end(), [&](const auto& item) { return item.surface == surface; })) return;
			surface_retirement owner{surface, surface->lease, surface->spill.buffer, surface->backing, surface->physical, {}};
			owner.deferred.reserve(surface->deferred.size());
			for (const auto& section : surface->deferred) owner.deferred.push_back(section.source);
			result.push_back(std::move(owner));
		};
		for (const auto& value : validated) retain(value.surface);
		for (const auto& surface : mutated) retain(surface);
		for (const auto& surface : aliases) retain(surface);
		return result;
	}
	std::vector<std::uint64_t> current(const cached_surface_ref& surface) const
	{
		std::vector<std::uint64_t> result(surface->storage.parts.size());
		cpu([&]
		{
			check_mapping(surface);
			for (std::size_t i = 0; i < result.size(); ++i)
				result[i] = memory->surface_revision(surface->storage.parts[i].coverage);
			check_mapping(surface);
		});
		return result;
	}
	surface_storage_snapshot read(const cached_surface_ref& surface) const
	{
		surface_storage_snapshot result{std::vector<std::uint64_t>(surface->storage.parts.size()), std::vector<std::byte>(surface->storage.size)};
		cpu([&]
		{
			check_mapping(surface);
			for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
			{
				const auto& part = surface->storage.parts[i];
				const auto snapshot = memory->read_surface(part.coverage);
				if (snapshot.bytes.size() != part.coverage.range.size)
					throw std::runtime_error("VM surface snapshot has the wrong byte count");
				std::copy(snapshot.bytes.begin(), snapshot.bytes.end(), result.bytes.begin() + part.offset);
				result.revisions[i] = snapshot.revision;
			}
			check_mapping(surface);
		});
		return result;
	}
	std::vector<std::uint64_t> publish(const cached_surface_ref& surface, std::span<const std::byte> bytes) const
	{
		if (bytes.size() != surface->storage.size || surface->revision.size() != surface->storage.parts.size())
			throw std::logic_error("Native surface publication differs from captured storage");
		std::vector<guest_surface_write> writes;
		writes.reserve(surface->storage.parts.size());
		std::vector<std::uint64_t> result(surface->storage.parts.size());
		for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
		{
			const auto& part = surface->storage.parts[i];
			writes.push_back({part.coverage, surface->revision[i], bytes.subspan(part.offset, part.coverage.range.size)});
		}
		if (!capture)
		{
			// Preserve the original single-watch CAS contract. A separately
			// unlocked version query could accept an intervening CPU write.
			if (writes.size() != 1) throw std::logic_error("Disjoint publication lacks CPU exclusion");
			const auto revision = memory->compare_write(writes[0].coverage, writes[0].expected, writes[0].bytes);
			if (!revision) throw surface_conflict_error(writes[0].coverage.range);
			result[0] = *revision;
			return result;
		}
		cpu([&]
		{
			check_mapping(surface);
			if (!memory->compare_write_many(writes)) throw surface_conflict_error(surface->storage.parts.front().coverage.range);
			// The batch result is provider-defined. Query each watch under the
			// same writer exclusion instead of assuming a common version domain.
			for (std::size_t i = 0; i < result.size(); ++i)
				result[i] = memory->surface_revision(surface->storage.parts[i].coverage);
		});
		return result;
	}
	void check(const cached_surface_ref& surface) const
	{
		if (!surface || !positions.contains(surface.get()))
			throw std::invalid_argument("Native cached surface belongs to another cache");
	}
	// Versioned writes retain untouched old native storage. In-place fallback
	// requires a successful independent read-only backup. Keep replaced leases
	// in rollback/retirements until publication leaves CPU exclusion.
	void reject(transaction& completed, bool backup_trusted) noexcept
	{
		if (completed.rollback)
		{
			for (auto& [surface, previous] : completed.rollback->aliases)
			{
				surface->initialized = previous.initialized;
				surface->spill = std::move(previous.spill);
				surface->native_write_epoch = previous.native_write_epoch;
			}
			retain_mutated_backings(completed);
			for (auto& saved : completed.rollback->surfaces)
			{
				auto& target = *saved.surface;
				auto& previous = saved.previous;
				if (!saved.output && !backup_trusted)
				{
					target.faulted = true;
					target.initialized = false;
					continue;
				}
				// A versioned writer never modified the old native allocation;
				// its contents are independent of the failed command's status.
				target.lease = saved.output ? previous.lease : surface_lease{};
				target.spill = saved.output ? std::move(previous.spill) : std::move(saved.snapshot);
				target.backing = std::move(previous.backing);
				target.physical = std::move(previous.physical);
				target.deferred = std::move(previous.deferred);
				target.deferred_revision = std::move(previous.deferred_revision);
				target.revision = std::move(previous.revision);
				target.generation = previous.generation;
				target.native_write_epoch = previous.native_write_epoch;
				target.initialized = saved.output && previous.initialized;
				target.dirty = previous.dirty;
				target.faulted = previous.faulted;
			}
			return;
		}
		retain_mutated_backings(completed);
		for (const auto& surface : completed.mutated) { surface->faulted = true; surface->initialized = false; }
	}
	void restore_recording(const std::shared_ptr<recorded_draw_batch>& batch) noexcept
	{
		for (const auto& saved : batch->rollback->surfaces)
			if (saved.output) saved.surface->lease = saved.previous.lease;
	}
	void discard_recording() noexcept
	{
		if (!recording_batch) return;
		auto batch = std::move(recording_batch);
		try { ctx.cancel(); } catch (...) {}
		restore_recording(batch);
		++transactions.cancelled_draw_batches;
	}
	void cancel_draw_batch() noexcept
	{
		batch_capacity = 0;
		predicated_completion = predicated_draw_completion::synchronous;
		discard_recording();
	}
	void flush_draw_batch()
	{
		if (!recording_batch) return;
		auto batch = recording_batch;
		try
		{
			ctx.end_encoding();
			validate_mappings(batch->validated);
			// Construct every callback/container before the irreversible submit.
			transaction value{ctx.recording_submission(), [batch]
				{ for (const auto& commit : batch->commits) commit(); },
				batch->mutated, batch->validated, batch->backings, batch->retirements, false, true, batch->rollback, {}, batch};
			value.work = ctx.submit();
			pending.emplace(std::move(value));
			recording_batch.reset();
			++transactions.submitted;
			++transactions.draw_batches;
		}
		catch (...)
		{
			ctx.cancel();
			restore_recording(batch);
			recording_batch.reset();
			++transactions.cancelled_draw_batches;
			throw;
		}
	}
	void synchronize(std::chrono::milliseconds wait, bool flush_recording = true)
	{
		if (flush_recording) flush_draw_batch();
		if (!pending) return;
		std::exception_ptr gpu_error;
		try { pending->work.wait(wait); }
		catch(const submission_wait_timeout&)
		{
			// Completion may turn ready between wait's deadline and this
			// handler. A wait timeout must never become a failed GPU result.
			throw;
		}
		catch (...)
		{
			if (!pending->work.ready()) throw; // A timeout never retires live work.
			gpu_error = std::current_exception();
		}
		// Never wait for either GPU submission under guest writer exclusion.
		bool backup_trusted = bool(pending->rollback);
		if (backup_trusted && pending->rollback->backup_work)
		{
			try { pending->rollback->backup_work->wait(wait); }
			catch (...) { backup_trusted = false; }
		}
		auto completed = std::move(*pending);
		pending.reset();
		bool publish = true;
		if (!gpu_error && completed.publish_draw)
		{
			try { publish = completed.publish_draw(std::chrono::milliseconds::zero()); }
			catch (...) { gpu_error = std::current_exception(); }
		}
		bool finalized = false;
		try
		{
			cpu([&]
			{
				try
				{
					if (gpu_error) std::rethrow_exception(gpu_error);
					if (!publish)
					{
						reject(completed, backup_trusted);
						finalized = true;
						return;
					}
					validate_completion(completed);
					retain_mutated_backings(completed);
					completed.commit();
					for (const auto& surface : completed.mutated)
						if (!surface->dirty && surface->deferred.empty() && !surface->faulted) surface->reset_backing();
					finalized = true;
				}
				catch (...) { reject(completed, backup_trusted); finalized = true; throw; }
			});
		}
		catch (...)
		{
			// A dispatcher which refused to enter leaves completion available for
			// retry, rather than losing the output or its only rollback owner.
			if (!finalized) pending.emplace(std::move(completed));
			throw;
		}
		completed.commit = {};
		completed.retirements.clear();
		if (completed.trim_pool) (void)pool.trim();
	}
	bool batch_alias(const cached_surface_ref& surface)
	{
		if (!recording_batch) return false;
		for (const auto& other : recording_batch->mutated)
			if (other != surface && memory_storage_overlap(*memory, surface->storage, other->storage)) return true;
		return false;
	}
	const surface_validation* recorded_draw_target(const cached_surface_ref& surface) const
	{
		if (!recording_batch || std::find(recording_batch->mutated.begin(), recording_batch->mutated.end(), surface) == recording_batch->mutated.end())
			return nullptr;
		const auto& validated = recording_batch->validated;
		const auto found = std::ranges::find(validated, surface, &surface_validation::surface);
		return found == validated.end() ? nullptr : &*found;
	}
	void batch_surface_boundary(const cached_surface_ref& surface, bool allow_provisional_target = false)
	{
		if (!recording_batch) return;
		const auto* target = recorded_draw_target(surface);
		if (batch_alias(surface) || (target && ((!allow_provisional_target && (!surface->initialized || surface->spill.buffer ||
			!surface->deferred.empty() || target->revisions != surface->revision)) ||
			current(surface) != target->revisions)))
			synchronize(timeout);
	}
	bool append_draw_batch(const std::function<void()>& encode, std::function<void()>& commit,
		const std::vector<cached_surface_ref>& mutated, std::vector<surface_validation>& validated,
		const std::vector<cached_surface_ref>& aliases, bool preserve_contents, const submission* predecessor,
		std::span<const draw_target_load> target_loads)
	{
		if (!batch_capacity || mutated.empty()) return false;
		if (recording_batch && recording_batch->commits.size() == batch_capacity) synchronize(timeout);
		for (const auto& surface : mutated) batch_surface_boundary(surface, true);
		for (const auto& surface : mutated)
			if (((!surface->initialized || surface->spill.buffer || !surface->deferred.empty()) && !recorded_draw_target(surface) &&
				std::ranges::find(target_loads, surface, &draw_target_load::surface) == target_loads.end()) || surface->faulted)
			{
				synchronize(timeout);
				return false;
			}
		if (validated.empty()) for (const auto& surface : mutated) validated.push_back({surface, surface->revision});
		auto backings = retain_backings(validated);
		auto retirements = retain_metadata_storage(validated, mutated, aliases);
		surface_list_guard pins(pinned, mutated);
		std::vector<rollback_surface> additions;
		for (const auto& surface : mutated)
		{
			if (recording_batch && std::any_of(recording_batch->rollback->surfaces.begin(), recording_batch->rollback->surfaces.end(),
				[&](const auto& saved) { return saved.surface == surface; })) continue;
			if (std::any_of(additions.begin(), additions.end(), [&](const auto& saved) { return saved.surface == surface; })) continue;
			rollback_surface saved{surface, *surface, surface->spill, false};
			try { saved.output = pool.acquire(surface->description.allocation); }
			catch (const surface_budget_error&)
			{
				additions.clear();
				synchronize(timeout);
				return false;
			}
			catch (const heap_budget_error&)
			{
				additions.clear();
				synchronize(timeout);
				return false;
			}
			catch (const native_allocation_error&)
			{
				additions.clear();
				synchronize(timeout);
				return false;
			}
			additions.push_back(std::move(saved));
		}
		auto batch = recording_batch ? recording_batch : std::make_shared<recorded_draw_batch>();
		try
		{
			// All owners/callbacks are attached before encoding starts. Any later
			// exception cancels this complete unsubmitted batch and restores leases.
			batch->commits.push_back(std::move(commit));
			batch->validated.insert(batch->validated.end(), validated.begin(), validated.end());
			batch->backings.insert(batch->backings.end(), backings.begin(), backings.end());
			batch->retirements.insert(batch->retirements.end(), retirements.begin(), retirements.end());
			for (const auto& surface : mutated)
				if (std::find(batch->mutated.begin(), batch->mutated.end(), surface) == batch->mutated.end()) batch->mutated.push_back(surface);
			for (const auto& surface : aliases)
				if (std::none_of(batch->rollback->aliases.begin(), batch->rollback->aliases.end(), [&](const auto& item) { return item.first == surface; }))
					batch->rollback->aliases.emplace_back(surface, *surface);
			const auto first_addition = batch->rollback->surfaces.size();
			batch->rollback->surfaces.insert(batch->rollback->surfaces.end(), std::make_move_iterator(additions.begin()), std::make_move_iterator(additions.end()));
			if (!recording_batch)
			{
				ctx.begin(predecessor);
				batch->recording.emplace(ctx.recording_submission());
				recording_batch = batch;
				ctx.keep_alive(batch);
			}
			else if (predecessor) ctx.depend_on_predecessor(*predecessor);
			for (const auto& input : backings)
			{
				if (input.owner) ctx.keep_alive(input.owner);
				if (input.physical) ctx.keep_alive(input.physical);
			}
			validate_mappings(validated);
			for (std::size_t i = first_addition; i < batch->rollback->surfaces.size(); ++i)
			{
				const auto& saved = batch->rollback->surfaces[i];
				if (preserve_contents && saved.previous.initialized && !saved.previous.spill.buffer &&
					std::ranges::find(target_loads, saved.surface, &draw_target_load::surface) == target_loads.end())
				{
					texture_region whole;
					whole.size = MTLSizeMake(saved.surface->description.allocation.width, saved.surface->description.allocation.height, 1);
					copy_texture(ctx, saved.previous.lease->render_texture(ctx), whole, saved.output->render_texture(ctx), whole);
					++transactions.native_version_blits;
				}
				saved.surface->lease = saved.output;
				++transactions.native_versions;
			}
			encode();
			++transactions.batched_draws;
			return true;
		}
		catch (...)
		{
			if (recording_batch) { ctx.cancel(); recording_batch.reset(); ++transactions.cancelled_draw_batches; }
			restore_recording(batch);
			throw;
		}
	}
	bool record_independent_targets(const std::function<void()>& encode_loads, const std::function<void()>& encode_draw,
		std::function<void()>& commit, const std::vector<cached_surface_ref>& written,
		std::vector<surface_validation>& validated, const submission& previous, std::span<const draw_target_load> loads,
		bool preserve_contents)
	{
		struct stale_recording {};
		bool invoking_draw = false;
		try
		{
			if (append_draw_batch([&]
			{
				encode_loads();
				// Retire the predecessor before exposing the draw callback. Its
				// upload plan, query and texture helpers may be single-use.
				synchronize(timeout, false);
				for (const auto& value : validated)
					if (current(value.surface) != value.revisions || !plan_aliases(value.surface).empty())
						throw stale_recording{};
				invoking_draw = true;
				if (encode_draw) encode_draw();
			}, commit, written, validated, {}, preserve_contents, &previous, loads)) return true;
		}
		catch (const stale_recording&) {}
		catch (const surface_conflict_error&)
		{
			if (invoking_draw || !pending) throw;
		}
		catch (...)
		{
			// begin() can observe cancellation before the new slot opens.
			if (pending && pending->work.ready()) synchronize(timeout, false);
			throw;
		}
		synchronize(timeout, false);
		return false;
	}
	void execute(const std::function<void()>& encode, std::function<void()> commit,
				 std::vector<cached_surface_ref> mutated = {}, std::vector<surface_validation> validated = {},
				 std::vector<cached_surface_ref> aliases = {}, bool trim_pool = false, bool native_output = false, bool preserve_contents = true, bool batchable = false,
				 bool require_native_version = false, draw_publication publish_draw = {}, const submission* predecessor = nullptr,
				 std::span<const draw_target_load> target_loads = {}, bool keep_deferred_readers = false, bool defer_native_completion = false)
	{
		if (!require_native_version && batchable && native_output && append_draw_batch(encode, commit, mutated, validated, aliases, preserve_contents, predecessor, target_loads)) return;
		synchronize(timeout);
		if (validated.empty())
			for (const auto& surface : mutated) validated.push_back({surface, surface->revision});
		auto backings = retain_backings(validated);
		auto retirements = retain_metadata_storage(validated, mutated, aliases);
		auto rollback = native_output ? backup_native(mutated, preserve_contents, require_native_version, target_loads, keep_deferred_readers) : nullptr;
		const bool defer_completion = defer_native_completion || (publish_draw && predicated_completion == predicated_draw_completion::deferred);
		// Allocate the transaction's callbacks and owners before submission.
		@autoreleasepool
		{
			ctx.begin(predecessor);
			try
			{
				if (rollback)
				{
					ctx.keep_alive(rollback);
					for (auto& saved : rollback->surfaces)
					{
						if (!saved.output) continue;
						texture_region whole;
						const auto& allocation = saved.surface->description.allocation;
						whole.size = MTLSizeMake(allocation.width, allocation.height, 1);
						if (rollback->preserve_contents && saved.previous.initialized && !saved.previous.spill.buffer &&
							std::ranges::find(target_loads, saved.surface, &draw_target_load::surface) == target_loads.end())
						{
							copy_texture(ctx, saved.previous.lease->render_texture(ctx), whole, saved.output->render_texture(ctx), whole);
							++transactions.native_version_blits;
						}
						saved.surface->lease = saved.output;
					}
				}
				for (const auto& retained : backings)
				{
					if (retained.owner) ctx.keep_alive(retained.owner);
					if (retained.physical) ctx.keep_alive(retained.physical);
				}
				encode();
			}
			catch (...)
			{
				ctx.cancel();
				if (rollback) for (auto& saved : rollback->surfaces) if (saved.output) saved.surface->lease = saved.previous.lease;
				throw;
			}
			try
			{
				ctx.end_encoding();
				validate_mappings(validated);
				const auto work = ctx.submit();
				++transactions.submitted;
				pending.emplace(transaction{work, std::move(commit), std::move(mutated), std::move(validated), std::move(backings), std::move(retirements), trim_pool, native_output, std::move(rollback), std::move(publish_draw)});
			}
			catch (...)
			{
				ctx.cancel();
				if (rollback) for (auto& saved : rollback->surfaces) if (saved.output) saved.surface->lease = saved.previous.lease;
				throw;
			}
		}
		if (!defer_completion) synchronize(timeout);
	}
	texture_region region(const cached_surface_ref& surface) const
	{
		texture_region result;
		result.size = MTLSizeMake(surface->layout.width, surface->layout.height, 1);
		return result;
	}
	id<MTLTexture> transfer_texture(const cached_surface_ref& surface, const surface_lease& target = {}) __attribute__((ns_returns_retained))
	{
		const auto& allocation = target ? target : surface->lease;
		return surface->description.allocation.samples > 1 ? allocation->expanded_texture(ctx) : allocation->render_texture(ctx);
	}
	void restore_spill(const cached_surface_ref& surface, const raw_surface_snapshot& snapshot, const surface_lease& target)
	{
		const auto texture = transfer_texture(surface, target);
		for (const auto& plane : snapshot.planes)
			buffer_to_texture(ctx, snapshot.buffer, plane.offset, texture, region(surface), plane.pitch, 0, plane.aspect);
		if (surface->description.allocation.samples > 1)
			samples->import(ctx, texture, target->render_texture(ctx));
	}
	void upload_guest(const cached_surface_ref& surface, id<MTLBuffer> buffer, const surface_lease& target = {})
	{
		const auto& allocation = target ? target : surface->lease;
		const auto texture = transfer_texture(surface, allocation);
		const auto area = region(surface);
		auto pitch = surface->layout.row_pitch;
		if (!surface->description.tile && !surface->layout.morton && pitch % surface->layout.pixel_bytes)
		{
			const auto row = checked_texture_product(surface->layout.width, surface->layout.pixel_bytes);
			const auto packed = make_buffer(ctx, checked_texture_product(row, surface->layout.height), buffer_storage::private_gpu);
			for (std::size_t y = 0; y < surface->layout.height; ++y)
				row_transfers.copy(ctx, buffer, y * pitch, packed, y * row, row);
			buffer = packed;pitch = row;
		}
		if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
		{
			if (surface->description.tile)
				banked.upload_color(ctx, *format, buffer, surface->layout, *surface->description.tile, texture, area);
			else if (surface->layout.morton)
				swizzle.upload_color(ctx, *format, buffer, 0, *surface->layout.morton, texture, area);
			else
				color.upload(ctx, *format, buffer, {0, pitch}, texture, area);
		}
		else
		{
			const auto depth_format = std::get<surface_depth>(surface->description.allocation.format);
			if (surface->description.tile)
				banked.upload_depth(ctx, depth_format, buffer, surface->layout, *surface->description.tile, texture, area);
			else if (surface->layout.morton)
				swizzle.upload_depth(ctx, depth_format, buffer, 0, *surface->layout.morton, texture, area);
			else
				depth.upload(ctx, depth_format, buffer, {0, pitch}, texture, area);
		}
		if (surface->description.allocation.samples > 1)
			samples->import(ctx, texture, allocation->render_texture(ctx));
	}
	void download_guest(const cached_surface_ref& surface, id<MTLBuffer> buffer)
	{
		const auto texture = transfer_texture(surface);
		const auto area = region(surface);
		const auto output = buffer;
		auto pitch = surface->layout.row_pitch;
		const bool repack = !surface->description.tile && !surface->layout.morton && pitch % surface->layout.pixel_bytes;
		if (repack)
		{
			pitch = checked_texture_product(surface->layout.width, surface->layout.pixel_bytes);
			buffer = make_buffer(ctx, checked_texture_product(pitch, surface->layout.height), buffer_storage::private_gpu);
		}
		if (surface->description.allocation.samples > 1)
			samples->expand(ctx, surface->lease->render_texture(ctx), texture);
		if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
		{
			if (surface->description.tile)
				banked.download_color(ctx, *format, texture, area, buffer, surface->layout, *surface->description.tile);
			else if (surface->layout.morton)
				swizzle.download_color(ctx, *format, texture, area, buffer, 0, *surface->layout.morton);
			else
				color.download(ctx, *format, texture, area, buffer, {0, pitch});
		}
		else
		{
			const auto depth_format = std::get<surface_depth>(surface->description.allocation.format);
			if (surface->description.tile)
				banked.download_depth(ctx, depth_format, texture, area, buffer, surface->layout, *surface->description.tile);
			else if (surface->layout.morton)
				swizzle.download_depth(ctx, depth_format, texture, area, buffer, 0, *surface->layout.morton);
			else
				depth.download(ctx, depth_format, texture, area, buffer, {0, pitch});
		}
		if (repack)
			for (std::size_t y = 0; y < surface->layout.height; ++y)
				row_transfers.copy(ctx, buffer, y * pitch, output, y * surface->layout.row_pitch, pitch);
	}
	[[nodiscard]] std::vector<cached_surface_ref> plan_aliases(const cached_surface_ref& surface)
	{
		std::vector<cached_surface_ref> result;
		std::vector<guest_surface_range> requested;
		for (const auto& part : surface->storage.parts) requested.push_back(part.coverage.range);
		memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
							 {
			for (const auto& other : candidates(direct, requested))
				if (other != surface && active_storage_overlap(direct, lookup, surface->storage, other->storage))
					result.push_back(other);
			return false; });
		return result;
	}
	static void invalidate_aliases(const std::vector<cached_surface_ref>& aliases) noexcept
	{
		for (const auto& other : aliases)
		{
			other->initialized = false;
			other->spill = {};
			other->native_write_epoch = 0;
		}
	}
	[[nodiscard]] std::vector<cached_surface_ref> cpu_aliases(const guest_surface_layout& coverage)
	{
		return cpu_aliases(std::span{&coverage, 1});
	}
	[[nodiscard]] std::vector<cached_surface_ref> cpu_aliases(std::span<const guest_surface_layout> coverage, bool native_only = false)
	{
		std::vector<cached_surface_ref> result;
		std::vector<guest_surface_range> requested;
		for (const auto& part : coverage) requested.push_back(part.range);
		memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
							 {
			for (const auto& surface : candidates(direct, requested))
				if ((!native_only || surface->dirty || !surface->deferred.empty() ||
					(surface->initialized && !surface->faulted && (surface->lease || surface->spill.buffer))) &&
					std::any_of(coverage.begin(), coverage.end(), [&](const auto& part)
					{ return active_storage_overlap(direct, lookup, part, surface->storage); }))
					result.push_back(surface);
			return false; });
		return result;
	}
	template <typename Visitor>
	bool visit_write_aliases(const cached_surface_ref& surface, bool include_dirty, Visitor&& visit)
	{
		const auto eligible = [&](const cached_surface_ref& other)
		{
			return other != surface && (!other->deferred.empty() || (include_dirty && other->dirty)) &&
				std::ranges::find(resolving, other) == resolving.end();
		};
		std::size_t next = 0;
		for (;;)
		{
			while (next < surfaces.size() && !eligible(surfaces[next])) ++next;
			if (next == surfaces.size()) return true;
			const bool found = memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
			{
				for (; next < surfaces.size(); ++next)
				{
					const auto& other = surfaces[next];
					if (eligible(other) && active_storage_overlap(direct, lookup, surface->storage, other->storage))
						return true;
				}
				return false;
			});
			if (!found) return true;
			// Resolving one alias can write guest memory and change the
			// remaining authorities. Resume with a fresh mapping afterward.
			const auto other = surfaces[next++];
			if (!visit(other)) return false;
		}
	}
	void flush(const cached_surface_ref& surface)
	{
		check(surface);
		synchronize(timeout);
		if (surface->physical && (surface->dirty || !surface->deferred.empty()))
		{
			flush_physical(surface, true);
			return;
		}
		materialize(surface);
		// Canonical writeback changes the revision of overlapping copy plans.
		// Consume those plans first. Nested writeback while a plan initializes its
		// background skips the resolving target and completes the old owner once.
		if (surface->dirty)
			visit_write_aliases(surface, false, [&](const auto& other)
			{
				materialize(other);
				return true;
			});
		if (!surface->dirty)
			return;
		prepare(surface, surface_initialization::guest);
		const auto snapshot = read(surface);
		if (snapshot.revisions != surface->revision)
			throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		if (snapshot.bytes.size() != surface->layout.range.size)
			throw std::runtime_error("VM surface snapshot has the wrong byte count");
		const auto output = make_upload(ctx, snapshot.bytes);
		struct writeback
		{
			id<MTLBuffer> buffer;
			std::span<std::byte> bytes;
		};
		// Capture the shared CPU pointer before entering publication;do not
		// query Metal properties while the VM writer lock is held.
		const writeback destination{output, map_buffer(output, 0, output.length)};
		const auto aliases = plan_aliases(surface);
		execute([&]
				{ download_guest(surface, output); },
				[this, surface, destination, aliases]
				{
					surface->revision = publish(surface, destination.bytes);
					surface->dirty = false;
					surface->reset_backing();
					invalidate_aliases(aliases);
				}, {}, {{surface, surface->revision}}, aliases);
	}
	std::vector<cached_surface_ref> physical_aliases(const cached_surface_ref& surface) const
	{
		std::vector<cached_surface_ref> aliases;
		std::vector<std::shared_ptr<const physical_backing>> parts;
		for (const auto& ticket : *surface->physical)
			parts.insert(parts.end(), ticket->parts().begin(), ticket->parts().end());
		const physical_identity identity(parts);
		for (const auto& other : surfaces)
		{
			if (other == surface) continue;
			if (other->dirty || !other->deferred.empty())
			{
				if (!other->physical) throw std::runtime_error("Physical recovery cannot order untyped dirty aliases");
				for (const auto& a : *surface->physical)
					for (const auto& b : *other->physical)
						if (a->overlaps(*b)) throw surface_conflict_error(a->coverage().range);
			}
			else
			{
				// Typed providers can preserve disjoint clean native precision.
				// Legacy metadata without this query stays conservative.
				for (const auto& part : other->storage.parts)
				{
					const auto overlap = memory->physical_identity_overlaps_current(identity, part.coverage);
					if (!overlap || *overlap) { aliases.push_back(other); break; }
				}
			}
		}
		return aliases;
	}
	std::vector<std::byte> read_physical(const cached_surface_ref& surface) const
	{
		const surface_validation validation{surface, {}, true};
		std::vector<std::byte> old(surface->storage.size);
		cpu([&]
		{
			validate_in_epoch({&validation, 1});
			for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
			{
				const auto snapshot = memory->read_physical_surface(*(*surface->physical)[i]);
				const auto& part = surface->storage.parts[i];
				if (!snapshot || snapshot->bytes.size() != part.coverage.range.size)
					throw std::runtime_error("Physical recovery lacks an exact old-owner snapshot");
				std::copy(snapshot->bytes.begin(), snapshot->bytes.end(), old.begin() + part.offset);
			}
		});
		return old;
	}
	void materialize_physical(const cached_surface_ref& surface)
	{
		if (surface->deferred.empty()) return;
		if (surface->faulted) throw std::runtime_error("Cannot materialize faulted native physical output");
		const surface_validation validation{surface, {}, true};
		validate_mappings({&validation, 1});
		const auto aliases = physical_aliases(surface);
		surface_list_guard pin(pinned, surface);
		std::size_t first = 0;
		bool full = false;
		for (std::size_t i = 0; i < surface->deferred.size(); ++i)
		{
			const auto& section = surface->deferred[i];
			if (section.plane == texture_plane::all && section.destination.x == 0 && section.destination.y == 0 &&
				section.destination.width == surface->description.allocation.width && section.destination.height == surface->description.allocation.height)
			{
				first = i;
				full = true;
			}
		}
		if(!full&&surface->deferred.size()<=rectangle_coverage_limit)
		{
			std::vector<clear_rectangle> rectangles;
			for(const auto& section:surface->deferred)if(section.plane==texture_plane::all)rectangles.push_back(section.destination);
			full=rectangles_cover(rectangles,surface->description.allocation.width,surface->description.allocation.height);
		}
		const bool native_background = surface->dirty || surface->revision == surface->deferred_revision;
		const auto spill = !full && native_background ? surface->spill : raw_surface_snapshot{};
		const auto background = !full && native_background && surface->initialized && !spill.buffer ? surface->lease : surface_lease{};
		if (!full && surface->dirty && !spill.buffer && !background)
			throw std::runtime_error("Deferred physical output lost its prior native dirty contents");
		const auto input = !full && !spill.buffer && !background ? make_upload(ctx, read_physical(surface)) : nil;
		// Write into a distinct allocation. Captured sources (including self)
		// and the prior target stay intact on timeout,error or failed CPU commit.
		const auto destination = allocate_transient(surface->description.allocation, surface);
		const auto native_epoch = reserve_native_epoch();
		auto revision = surface->deferred_revision;
		execute([&]
		{
			ctx.keep_alive(destination);
			const auto texture = destination->render_texture(ctx);
			if (input) upload_guest(surface, input, destination);
			else if (spill.buffer)
			{
				const auto transfer = transfer_texture(surface, destination);
				for (const auto& plane : spill.planes)
					buffer_to_texture(ctx, spill.buffer, plane.offset, transfer, region(surface), plane.pitch, 0, plane.aspect);
				if (surface->description.allocation.samples > 1) samples->import(ctx, transfer, texture);
			}
			else if (background)
			{
				texture_region area;
				area.size = MTLSizeMake(surface->description.allocation.width, surface->description.allocation.height, 1);
				copy_texture(ctx, background->render_texture(ctx), area, texture, area);
			}
			for (std::size_t i = first; i < surface->deferred.size(); ++i)
			{
				const auto& section = surface->deferred[i];
				texture_region sr, dr;
				sr.origin = MTLOriginMake(section.rectangle.x, section.rectangle.y, 0);
				dr.origin = MTLOriginMake(section.destination.x, section.destination.y, 0);
				sr.size = MTLSizeMake(section.rectangle.width, section.rectangle.height, 1);
				dr.size = MTLSizeMake(section.destination.width, section.destination.height, 1);
				sample_blits.nearest(ctx, section.source->description().format, section.source->render_texture(ctx), sr,
					surface->description.allocation.format, texture, dr, section.plane, section.flip, section.typeless);
			}
		}, [surface, destination, native_epoch, aliases, revision = std::move(revision)]() mutable
		{
			surface->lease = destination;
			surface->revision = std::move(revision);
			surface->deferred.clear();
			surface->spill = {};
			surface->initialized = true;
			surface->dirty = true;
			surface->native_write_epoch = native_epoch;
			++surface->generation;
			invalidate_aliases(aliases);
		}, {}, {validation}, aliases);
	}
	void flush_physical(const cached_surface_ref& surface, bool preserve_native = false)
	{
		check(surface);
		synchronize(timeout);
		materialize_physical(surface);
		if (!surface->dirty) return;
		if (surface->faulted) throw std::runtime_error("Cannot publish faulted native physical output");
		const surface_validation validation{surface, {}, true};
		validate_mappings({&validation, 1});
		const auto aliases = physical_aliases(surface);
		if (!surface->lease)
		{
			if (!surface->spill.buffer) throw std::runtime_error("Physical dirty surface lost native output and spill");
			allocate(surface);
		}
		if (surface->spill.buffer)
		{
			const auto snapshot = surface->spill;
			execute([&]
			{
				const auto texture = transfer_texture(surface);
				for (const auto& plane : snapshot.planes)
					buffer_to_texture(ctx, snapshot.buffer, plane.offset, texture, region(surface), plane.pitch, 0, plane.aspect);
				if (surface->description.allocation.samples > 1) samples->import(ctx, texture, surface->lease->render_texture(ctx));
			}, [surface] { surface->initialized = true; surface->spill = {}; }, {}, {validation});
		}
		if (!surface->initialized) throw std::runtime_error("Physical dirty surface has no completed native contents");
		const auto output = make_upload(ctx, read_physical(surface));
		const auto bytes = map_buffer(output, 0, output.length);
		std::vector<physical_surface_write> writes;
		writes.reserve(surface->storage.parts.size());
		for (std::size_t i = 0; i < surface->storage.parts.size(); ++i)
		{
			const auto& part = surface->storage.parts[i];
			writes.push_back({(*surface->physical)[i], bytes.subspan(part.offset, part.coverage.range.size)});
		}
		execute([&] { download_guest(surface, output); },
			[this, surface, output, writes = std::move(writes), aliases, preserve_native, virtual_revisions = surface->revision]
			{
				(void)output; // Retain the shared readback buffer through pending CPU publication.
				bool retain = preserve_native;
				if (retain)
				{
					try { check_mapping(surface); }
					catch (const surface_mapping_conflict&) { retain = false; }
				}
				auto publication = memory->compare_write_physical(writes, retain ? std::span<const std::uint64_t>(virtual_revisions) : std::span<const std::uint64_t>{});
				if (!publication) throw surface_conflict_error(surface->layout.range);
				// Receipt storage was prepared before guest mutation. Moving it and
				// publishing metadata cannot allocate or query mappings after the store.
				const bool current = !publication->virtual_revisions.empty();
				if (current) surface->revision = std::move(publication->virtual_revisions);
				surface->dirty = false;
				surface->initialized = current;
				surface->spill = {};
				surface->reset_backing();
				invalidate_aliases(aliases);
			}, {}, {validation}, aliases);
	}
	bool flush_aliases(const cached_surface_ref& surface, const std::function<bool()>& permit = {})
	{
		return visit_write_aliases(surface, true, [&](const auto& other)
		{
			if (permit && !permit()) return false;
			flush(other);
			return true;
		});
	}
	void preserve_readers(const surface_lease& allocation, const cached_surface_ref& writer)
	{
		if (!allocation)
			return;
		// Keep the allocation identity even if materializing a reader spills the
		// writer or invalidates its logical guest alias. Never retain an MTL handle.
		const auto captured = allocation;
		for (const auto& other : surfaces)
			if (other != writer && std::find(resolving.begin(), resolving.end(), other) == resolving.end() &&
				std::ranges::any_of(other->deferred, [&](const auto& section) { return section.source == captured; }))
				materialize(other);
	}
	void encode_deferred_copies(const cached_surface_ref& surface, id<MTLTexture> destination,
		std::span<const deferred_copy_section> sections, id<MTLTexture> self_copy = nil)
	{
		for (const auto& section : sections)
		{
			const auto source = self_copy && section.source == surface->lease ? self_copy : section.source->render_texture(ctx);
			texture_region sr, dr;
			sr.origin = MTLOriginMake(section.rectangle.x, section.rectangle.y, 0);
			dr.origin = MTLOriginMake(section.destination.x, section.destination.y, 0);
			sr.size = MTLSizeMake(section.rectangle.width, section.rectangle.height, 1);
			dr.size = MTLSizeMake(section.destination.width, section.destination.height, 1);
			sample_blits.nearest(ctx, section.source->description().format, source, sr,
				surface->description.allocation.format, destination, dr, section.plane, section.flip, section.typeless);
		}
	}
	static bool needs_target_load(const cached_surface_ref& target)
	{
		return !target->faulted && (!target->deferred.empty() || target->spill.buffer ||
			(!target->initialized && !target->dirty));
	}
	bool independent_target(const cached_surface_ref& target, surface_cache& owner)
	{
		if (!pending || !pending->draw_batch || !plan_aliases(target).empty()) return false;
		std::vector<guest_surface_layout> coverage;
		for (const auto& part : target->storage.parts) coverage.push_back(part.coverage);
		return !owner.prepare_texture_read_boundary(coverage, true);
	}
	bool independent_cold_target(const cached_surface_ref& target, surface_cache& owner)
	{
		return pending && !pending->work.ready() && !target->initialized && !target->dirty && !target->faulted &&
			!target->lease && !target->spill.buffer && target->deferred.empty() && independent_target(target, owner);
	}
	std::optional<prefetched_target_data> prefetch_guest_target(const cached_surface_ref& target, surface_cache& owner,
		std::size_t limit)
	{
		if (target->storage.size > limit || target->faulted || target->dirty) return {};
		const bool deferred = !target->deferred.empty();
		if (deferred)
		{
			if (!pending || pending->work.ready() || current_deferred_background(*target) ||
				deferred_copy_start(target).second || !independent_target(target, owner)) return {};
		}
		else if (!independent_cold_target(target, owner)) return {};
		@autoreleasepool
		{
			try
			{
				prefetched_target_data result{read(target)};
				if (deferred && result.snapshot.revisions != target->deferred_revision) return {};
				// This optional staging must not reclaim other surfaces or wait.
				if (const auto ledger = ctx.buffer_ledger())
					result.upload = ledger->try_shared_buffer(result.snapshot.bytes.size());
				if (result.upload)
				{
					std::memcpy(result.upload.contents, result.snapshot.bytes.data(), result.snapshot.bytes.size());
					std::vector<std::byte>{}.swap(result.snapshot.bytes);
					if (deferred) result.sections = target->deferred;
				}
				return result;
			}
			catch (const std::exception&)
			{
				// Normal preparation reports errors against the settled authority.
				return {};
			}
		}
	}
	staged_target_load stage_guest_target(const cached_surface_ref& target, surface_cache& owner,
		std::span<const std::uint64_t> expected = {}, prefetched_target_data* prefetched = nullptr)
	{
		const bool reuse = prefetched && current(target) == prefetched->snapshot.revisions;
		id<MTLBuffer> input = reuse ? prefetched->upload : nil;
		if (prefetched) prefetched->upload = nil;
		auto snapshot = reuse ? std::move(prefetched->snapshot) : read(target);
		if ((input ? input.length : snapshot.bytes.size()) != target->layout.range.size)
			throw std::runtime_error("VM surface snapshot has the wrong byte count");
		if (!expected.empty() && !std::ranges::equal(snapshot.revisions, expected))
			throw surface_conflict_error(target->layout.range);
		if (!input) owner.prepare_draw_resources([&] { input = make_upload(ctx, snapshot.bytes); });
		return {{target, input}, {target, std::move(snapshot.revisions)}};
	}
	static bool plan_deferred_target_load(draw_target_load& load)
	{
		const auto& target = load.surface;
		const auto [first, full] = deferred_copy_start(target);
		load.sections.assign(target->deferred.begin() + first, target->deferred.end());
		if (full) return true;
		if (!current_deferred_background(*target)) return false;
		load.spill = target->spill;
		if (!load.spill.buffer) load.background = target->lease;
		return true;
	}
	std::optional<staged_target_load> plan_independent_write(const cached_surface_ref& target, surface_cache& owner,
		prefetched_target_data* captured)
	{
		if (captured && captured->upload && current(target) == captured->snapshot.revisions)
		{
			draw_target_load load{target, captured->upload};
			load.sections = std::move(captured->sections);
			return staged_target_load{std::move(load), {target, captured->snapshot.revisions}};
		}
		if (target->faulted || (!target->initialized && target->deferred.empty()) || !independent_target(target, owner)) return {};
		const auto& revision = target->deferred.empty() ? target->revision : target->deferred_revision;
		if (current(target) != revision) return {};
		draw_target_load load{target};
		if (!target->deferred.empty())
		{
			// Deferred sources are frozen leases. The predecessor writes its
			// own versions even when it rewrites a donor.
			if (!plan_deferred_target_load(load)) return {};
		}
		else if (target->spill.buffer) load.spill = target->spill;
		else if (!target->lease) return {};
		// A plain resident version is copied by append_draw_batch.
		return staged_target_load{std::move(load), {target, revision}};
	}
	std::optional<staged_target_load> stage_target_load(
		const cached_surface_ref& target, surface_cache& owner, const std::function<bool()>& permit_aliases = {},
		prefetched_target_data* prefetched = nullptr)
	{
		if (!needs_target_load(target)) return {};
		if (!target->deferred.empty() && current(target) != target->deferred_revision)
			throw surface_conflict_error(target->layout.range);
		if (!flush_aliases(target, permit_aliases)) return {};
		if (!needs_target_load(target)) return {};

		draw_target_load load{target};
		surface_validation validation{target};
		bool guest = true;
		if (!target->deferred.empty())
		{
			validation.revisions = target->deferred_revision;
			if (current(target) != validation.revisions) throw surface_conflict_error(target->layout.range);
			guest = !plan_deferred_target_load(load);
		}
		else if (target->spill.buffer && current(target) == target->revision)
		{
			load.spill = target->spill;
			validation.revisions = target->revision;
			guest = false;
		}
		if (guest)
		{
			if (target->dirty) throw surface_conflict_error(target->layout.range);
			auto staged = stage_guest_target(target, owner, validation.revisions, prefetched);
			load.guest = staged.first.guest;
			validation.revisions = std::move(staged.second.revisions);
		}
		return std::pair{std::move(load), std::move(validation)};
	}
	void encode_target_load(const draw_target_load& load)
	{
		if (load.spill.buffer) restore_spill(load.surface, load.spill, load.surface->lease);
		else if (load.guest) upload_guest(load.surface, load.guest);
		else if (load.background)
		{
			texture_region whole;
			const auto& allocation = load.surface->description.allocation;
			whole.size = MTLSizeMake(allocation.width, allocation.height, 1);
			copy_texture(ctx, load.background->render_texture(ctx), whole, load.surface->lease->render_texture(ctx), whole);
		}
		if (!load.sections.empty())
			encode_deferred_copies(load.surface, load.surface->lease->render_texture(ctx), load.sections);
	}
	void materialize(const cached_surface_ref& surface)
	{
		if (surface->deferred.empty() || std::find(resolving.begin(), resolving.end(), surface) != resolving.end())
			return;
		if (surface->faulted)
			throw std::runtime_error("Deferred native surface is faulted; overwrite or recover before use");
		if (current(surface) != surface->deferred_revision)
			throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		surface_list_guard resolution(resolving, surface);
		surface_list_guard pin(pinned, surface);
		preserve_readers(surface->lease, surface);
		flush_aliases(surface);
		const auto [first, full] = deferred_copy_start(surface);
		std::vector<std::uint64_t> revision;
		if (full)
		{
			revision = current(surface);
			allocate(surface);
		}
		else
		{
			prepare(surface, surface_initialization::guest);
			revision = surface->revision;
		}
		const auto aliases = plan_aliases(surface);
		const auto native_epoch = reserve_native_epoch();
		execute([&]
				{
					const auto destination = surface->lease->render_texture(ctx);
					id<MTLTexture> snapshot = nil;
					// Several queued reads of this allocation must all see the captured
					// epoch, including self-copies followed by other destination writes.
					for (std::size_t i = first; i < surface->deferred.size(); ++i)
						if (surface->deferred[i].source == surface->lease)
						{
							auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:destination.pixelFormat width:destination.width height:destination.height mipmapped:NO];
							descriptor.sampleCount = destination.sampleCount;
							descriptor.textureType = destination.textureType;
							descriptor.storageMode = MTLStorageModePrivate;
							descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
							snapshot = make_texture(ctx, descriptor);
							texture_region area;
							area.size = MTLSizeMake(destination.width, destination.height, 1);
							copy_texture(ctx, destination, area, snapshot, area);
							break;
						}
					encode_deferred_copies(surface, destination, std::span{surface->deferred}.subspan(first), snapshot); },
				[surface, revision, aliases, native_epoch]
				{
					surface->deferred.clear();
					surface->revision = revision;
					surface->initialized = true;
					surface->dirty = true;
					surface->native_write_epoch = native_epoch;
					surface->spill = {};
					++surface->generation;
					invalidate_aliases(aliases);
				},
				{surface}, {{surface, revision}}, aliases, false, true);
	}
	void allocate(const cached_surface_ref& surface)
	{
		if (surface->lease)
			return;
		surface->lease = allocate_transient(surface->description.allocation, surface);
	}
	surface_lease allocate_transient(const surface_allocation_description& description, const cached_surface_ref& protected_surface = {})
	{
		for (;;)
		{
			try
			{
				return pool.acquire(description);
			}
			catch (const surface_budget_error& error)
			{
				if (error.requested > error.budget)
					throw;
				if (!snapshots.empty()) { evict_snapshot(std::prev(snapshots.end())); continue; }
				if (recording_batch)
				{
					synchronize(timeout);
					++transactions.draw_resource_retries;
					continue;
				}
				// A local pool quota can fail before the shared ledger is entered.
				// Reconstructible data needs no additional raw spill allocation.
				if (reclaim_clean(protected_surface)) continue;
				cached_surface_ref oldest;
				for (const auto& other : surfaces)
					if (other != protected_surface && other->lease && !(other->faulted && other->dirty) &&
						std::find(pinned.begin(), pinned.end(), other) == pinned.end() && (!oldest || other->last_use < oldest->last_use))
						oldest = other;
				if (!oldest)
					throw;
				spill_surface(oldest);
			}
			catch (const heap_budget_error&)
			{
				if (!recording_batch) throw;
				synchronize(timeout);
				(void)ctx.trim_upload_pages();
				++transactions.draw_resource_retries;
			}
			catch (const native_allocation_error&)
			{
				if (!recording_batch) throw;
				synchronize(timeout);
				(void)ctx.trim_upload_pages();
				++transactions.draw_resource_retries;
			}
		}
	}
	raw_surface_snapshot raw_snapshot(const cached_surface_ref& surface, std::size_t limit = std::numeric_limits<std::size_t>::max())
	{
		raw_surface_snapshot result;
		const auto format = native_format(surface->description.allocation);
		const auto aspects = format == MTLPixelFormatDepth32Float_Stencil8 ? std::vector{texture_plane::depth, texture_plane::stencil} : std::vector{std::holds_alternative<surface_depth>(surface->description.allocation.format) ? texture_plane::depth : texture_plane::all};
		std::size_t size = 0;
		for (const auto aspect : aspects)
		{
			const auto block = storage_block(format, aspect);
			const auto row = checked_texture_product(surface->layout.width, block.bytes);
			if (row > std::numeric_limits<std::size_t>::max() - 255)
				throw std::overflow_error("Native surface spill row overflow");
			const auto pitch = (row + 255) & ~std::size_t(255);
			const auto bytes = checked_texture_product(pitch, surface->layout.height);
			if (size > std::numeric_limits<std::size_t>::max() - bytes)
				throw std::overflow_error("Native surface spill size overflow");
			result.planes.push_back({aspect, size, pitch});
			size += bytes;
		}
		if (size > limit) throw std::runtime_error("Native rollback exceeds bounded raw snapshot budget");
		result.buffer = make_buffer(ctx, size, buffer_storage::shared);
		return result;
	}
	std::shared_ptr<native_rollback> backup_native(std::span<const cached_surface_ref> mutated, bool preserve_contents, bool require_native_version,
		std::span<const draw_target_load> target_loads = {}, bool keep_deferred_readers = false)
	{
		surface_list_guard pins(pinned, mutated);
		auto result = std::make_shared<native_rollback>();
		result->surfaces.reserve(mutated.size());
		result->preserve_contents = preserve_contents;
		std::size_t remaining = snapshot_limit;
		bool copy = false;
		for (const auto& surface : mutated)
		{
			if (std::any_of(result->surfaces.begin(), result->surfaces.end(), [&](const auto& saved) { return saved.surface == surface; })) continue;
			rollback_surface saved{surface, *surface, surface->spill, false};
			if (std::ranges::find(target_loads, surface, &draw_target_load::surface) != target_loads.end())
			{
				saved.output = require_native_version ? pool.acquire(surface->description.allocation) :
					allocate_transient(surface->description.allocation, surface);
				++transactions.native_versions;
			}
			else if (surface->initialized && !surface->faulted && !saved.snapshot.buffer)
			{
				// Reserve a private native destination. The old lease is the
				// rollback version; no sample expansion/readback is necessary.
				try { saved.output = pool.acquire(surface->description.allocation); }
				catch (const surface_budget_error&)
				{
					if (require_native_version) throw;
					// Bounded legacy raw backup under pressure.
				}
				if (saved.output) ++transactions.native_versions;
				else
				{
					// The raw fallback writes in place. Only this path has to
					// finish readers before the old allocation can be modified.
					if (keep_deferred_readers)
					{
						preserve_readers(saved.previous.lease, surface);
						if (surface->lease != saved.previous.lease || !surface->initialized ||
							surface->faulted || surface->spill.buffer || surface->revision != saved.previous.revision)
							throw surface_conflict_error(surface->layout.range);
					}
					saved.snapshot = raw_snapshot(surface, remaining);
					remaining -= saved.snapshot.buffer.length;
					transactions.raw_backup_bytes += saved.snapshot.buffer.length;
					saved.copy = copy = true;
				}
			}
			if (require_native_version && !saved.output)
				throw std::logic_error("Scaled donor write lacks an independent native target");
			result->surfaces.push_back(std::move(saved));
		}
		if (!copy) return result;
		// Same ordered Metal4 queue as the write. This read-only submission
		// has its own status and retains all old resources through completion.
		ctx.begin();
		try
		{
			ctx.keep_alive(result);
			for (const auto& saved : result->surfaces)
			{
				if (!saved.copy) continue;
				const auto texture = transfer_texture(saved.surface);
				if (saved.surface->description.allocation.samples > 1)
					samples->expand(ctx, saved.previous.lease->render_texture(ctx), texture);
				for (const auto& plane : saved.snapshot.planes)
					texture_to_buffer(ctx, texture, region(saved.surface), saved.snapshot.buffer, plane.offset, plane.pitch, 0, plane.aspect);
			}
			result->backup_work = ctx.submit();
			++transactions.raw_backup_submissions;
		}
		catch (...) { ctx.cancel(); throw; }
		return result;
	}
	void spill_surface(const cached_surface_ref& surface)
	{
		check(surface);
		synchronize(timeout);
		surface_list_guard pin(pinned, surface);
		if (!surface->lease)
			return;
		if (surface->faulted && surface->dirty)
			throw std::runtime_error("Cannot spill faulted dirty native contents");
		if (surface->faulted || !surface->initialized || (!surface->dirty && current(surface) != surface->revision))
		{
			surface->lease.reset();
			surface->initialized = false;
			surface->spill = {};
			(void)pool.trim();
			return;
		}
		const auto snapshot = raw_snapshot(surface);
		execute([&]
				{
					const auto texture = transfer_texture(surface);
					if (surface->description.allocation.samples > 1)
						samples->expand(ctx, surface->lease->render_texture(ctx), texture);
					for (const auto& plane : snapshot.planes)
						texture_to_buffer(ctx, texture, region(surface), snapshot.buffer, plane.offset, plane.pitch, 0, plane.aspect); },
				[surface, snapshot]
				{
					surface->spill = snapshot;
					surface->lease.reset();
				}, {}, {{surface, surface->revision}}, {}, true);
	}
	bool prepare(const cached_surface_ref& surface, surface_initialization initialization, const std::function<bool()>& permit = {},
		target_reload* reload = nullptr)
	{
		check(surface);
		batch_surface_boundary(surface);
		synchronize(timeout, false);
		surface_list_guard pin(pinned, surface);
		if (permit && (surface->faulted || !surface->deferred.empty()) && !permit()) return false;
		if (surface->faulted)
			throw std::runtime_error("Native cached surface is faulted; restore valid backing before use");
		materialize(surface);
		if (!flush_aliases(surface, permit)) return false;
		const auto revision = current(surface);
		if (surface->initialized && surface->lease && surface->revision == revision)
		{
			surface->last_use = ++clock;
			return true;
		}
		// A cold target may reload an old allocation or materialize its
		// deferred readers. Resolve the report before altering either.
		if (permit && !permit()) return false;
		if (surface->dirty && surface->revision != revision)
			throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		surface->last_use = ++clock;
		if (reload && !surface->dirty && !surface->spill.buffer && surface->deferred.empty())
		{
			reload->staged = stage_guest_target(surface, reload->owner);
			return true;
		}
		if (surface->spill.buffer && surface->revision != revision)
		{
			surface->spill = {};
			surface->initialized = false;
		}
		allocate(surface);
		preserve_readers(surface->lease, surface);
		if (surface->spill.buffer)
		{
			const auto snapshot = surface->spill;
			execute([&] { restore_spill(surface, snapshot, surface->lease); },
					[surface]
					{
						surface->spill = {};
						surface->initialized = true;
					},
					{surface});
			return true;
		}
		if (initialization == surface_initialization::background)
		{
			const auto aliases = plan_aliases(surface);
			const auto native_epoch = reserve_native_epoch();
			execute([&]
					{
						encode_surface_background(ctx, surface->description.allocation.format, surface->lease->render_texture(ctx)); },
					[surface, revision, aliases, native_epoch]
					{
						surface->revision = revision;
						surface->initialized = true;
						surface->dirty = true;
						surface->native_write_epoch = native_epoch;
						++surface->generation;
						invalidate_aliases(aliases);
					},
					{surface}, {{surface, revision}}, aliases, false, true);
			return true;
		}
		const auto snapshot = read(surface);
		if (snapshot.bytes.size() != surface->layout.range.size)
			throw std::runtime_error("VM surface snapshot has the wrong byte count");
		const auto input = make_upload(ctx, snapshot.bytes);
		execute([&]
				{ upload_guest(surface, input); },
				[surface, revision = snapshot.revisions]
				{
					surface->revision = revision;
					surface->initialized = true;
					surface->dirty = false;
					surface->reset_backing();
					surface->native_write_epoch = 0;
					++surface->generation;
				},
				{surface}, {{surface, snapshot.revisions}});
		return true;
	}
	std::vector<cached_surface_ref> current_texture_sources(std::span<const guest_surface_layout> coverage)
	{
		std::vector<cached_surface_ref> result;
		cpu([&]
		{
			result = cpu_aliases(coverage,true);
			// Retained dirty physical output must never be mistaken for VM bytes
			// just because its original source VA was replaced or unmapped.
			for (const auto& surface : surfaces)
				if (surface->physical && (surface->dirty || !surface->deferred.empty()) &&
					std::find(result.begin(),result.end(),surface)==result.end())
					for (const auto& ticket : *surface->physical)
					{
						bool found = false;
						for (const auto& part : coverage)
						{
							guest_surface_byte_cursor cursor(part);
							while (const auto run = cursor.next())
							{
								const auto match = memory->physical_surface_overlaps_current(*ticket,*run);
								if (!match) throw std::runtime_error("Native display lacks physical source identity");
								if (*match) { found=true;break; }
							}
							if (found) break;
						}
						if (found) { result.push_back(surface);break; }
					}
			std::erase_if(result,[&](const auto& source)
			{
				if (source->dirty || !source->deferred.empty())
				{
					check_mapping(source);
					if (current(source)!=(source->deferred.empty()?source->revision:source->deferred_revision))
						throw surface_conflict_error(source->storage.parts.front().coverage.range);
					return false;
				}
				if (!source->native_write_epoch || source->faulted) return true;
				if (source->description.mapping && !source->description.mapping->current()) return true;
				return current(source) != source->revision;
			});
		});
		if (result.size()>64) throw std::length_error("Native display source bound exceeded");
		return result;
	}
};

surface_cache::surface_cache(context& ctx, std::shared_ptr<guest_surface_memory> memory, std::size_t heap_budget, std::shared_ptr<metal4::heap_budget> shared_budget, cpu_capture_dispatch capture, surface_metadata_limits metadata)
	: m_impl(std::make_unique<impl>(ctx, std::move(memory), heap_budget, std::move(shared_budget), std::move(capture), metadata))
{
}
surface_cache::~surface_cache() { m_impl->snapshot_lifetime.reset(); m_impl->cancel_draw_batch(); }
surface_cache::draw_batch::draw_batch(surface_cache& owner, std::size_t capacity, predicated_draw_completion completion)
	: m_owner(&owner), m_lifetime(owner.m_impl->batch_lifetime)
{
	owner.m_impl->outside_epoch();
	if (!capacity || capacity > 256) throw std::invalid_argument("Native draw batch capacity must be in[1,256]");
	if (owner.m_impl->batch_capacity) throw std::logic_error("Native draw batch scopes cannot nest");
	owner.m_impl->batch_capacity = capacity;
	owner.m_impl->predicated_completion = completion;
}
surface_cache::draw_batch::~draw_batch() { if (m_owner && !m_lifetime.expired()) m_owner->m_impl->cancel_draw_batch(); }
void surface_cache::draw_batch::submit()
{
	if (!m_owner || m_lifetime.expired()) throw std::logic_error("Native draw batch finished or owner destroyed");
	m_owner->m_impl->outside_epoch();
	auto* owner = m_owner;
	m_owner = nullptr;
	owner->m_impl->batch_capacity = 0;
	owner->m_impl->predicated_completion = predicated_draw_completion::synchronous;
	owner->m_impl->flush_draw_batch();
}
void surface_cache::draw_batch::finish()
{
	auto* owner = m_owner;
	submit();
	owner->m_impl->synchronize(owner->m_impl->timeout, false);
}
cached_surface_ref surface_cache::acquire(const cached_surface_description& description)
{
	m_impl->outside_epoch();
	auto normalized = description;
	if (normalized.allocation.samples != 1 && normalized.allocation.samples != 2 && normalized.allocation.samples != 4)
		throw std::invalid_argument("Native cached surface samples are invalid");
	normalized.allocation.expanded = normalized.allocation.samples > 1;
	const auto size = normalized.allocation.samples == 1 ? std::array{normalized.allocation.width, normalized.allocation.height} : sample_expanded_size(normalized.allocation.width, normalized.allocation.height, normalized.allocation.samples);
	const auto limit = max_2d_texture_dimension(m_impl->ctx.device());
	if (size[0] > limit || size[1] > limit)
		throw std::invalid_argument("Cached expanded dimensions exceed Apple GPU family limit");
	const bool segmented = !normalized.segments.empty();
	if ((segmented || normalized.mapping) && !m_impl->capture)
		throw std::invalid_argument("Mapped/disjoint surfaces require a common CPU capture provider");
	if (normalized.tile && !normalized.morton)
	{
		if (normalized.row_pitch && normalized.row_pitch != normalized.tile->pitch)
			throw std::invalid_argument("Banked surface row pitch differs from tile pitch");
		normalized.row_pitch = normalized.tile->pitch;
	}
	auto layout = plan_guest_surface(segmented ? 0 : normalized.address, size[0], size[1], pixel_bytes(normalized.allocation), normalized.row_pitch, normalized.morton);
	if (!layout.morton)
		normalized.row_pitch = layout.row_pitch;
	if (normalized.tile)
	{
		if (normalized.tile->size > m_impl->snapshot_limit) throw std::length_error("Banked surface snapshot exceeds cache byte limit");
		if (segmented)
		{
			layout.range = {normalized.tile->address, normalized.tile->size};
			validate_banked_tile_geometry(layout, *normalized.tile);
			validate_guest_segments(layout, normalized.segments);
			const auto origin = normalized.tile->offset;
			const auto part = std::find_if(normalized.segments.begin(), normalized.segments.end(), [&](const auto& segment)
				{ return origin >= segment.offset && origin - segment.offset < segment.size; });
			if (part == normalized.segments.end() || std::uint64_t(part->address) + origin - part->offset != normalized.address)
				throw std::invalid_argument("Banked surface pixel origin differs from actual storage segment");
		}
		else layout = attach_banked_tile(std::move(layout), *normalized.tile);
	}
	else if (segmented)
	{
		layout.range.address = normalized.address;
		validate_guest_segments(layout, normalized.segments);
		if (layout.range.size > m_impl->snapshot_limit) throw std::length_error("Disjoint surface snapshot exceeds cache byte limit");
	}
	if (normalized.mapping)
	{
		const auto& mapped = normalized.mapping->state();
		if (mapped.pixel_address != normalized.address || mapped.tile != normalized.tile || mapped.segments != normalized.segments || mapped.logical.size != layout.range.size)
			throw std::invalid_argument("Surface descriptor differs from owned framebuffer mapping");
		m_impl->cpu([&] { if (!normalized.mapping->current()) throw surface_mapping_conflict(); });
	}
	const auto hash = descriptor_hash(normalized);
	++m_impl->metadata.descriptor_lookups;
	const auto bucket = m_impl->descriptors.equal_range(hash);
	for (auto i = bucket.first; i != bucket.second; ++i)
	{
		++m_impl->metadata.descriptor_comparisons;
		if (same(i->second->description, normalized))
		{
			i->second->last_use = ++m_impl->clock;
			return m_impl->surfaces[m_impl->positions.at(i->second)];
		}
	}
	m_impl->synchronize(m_impl->timeout);
	auto surface = std::make_shared<cached_surface>();
	surface->description = normalized;
	surface->layout = layout;
	surface->storage.size = layout.range.size;
	if (segmented)
		surface->storage.parts = project_guest_segments(layout, normalized.segments, bool(normalized.tile), m_impl->metadata.limits.watch_parts);
	else surface->storage.parts.push_back({normalized.tile ? guest_byte_coverage(layout.range) : layout, 0});
	surface->revision.resize(surface->storage.parts.size());
	m_impl->reserve_metadata(surface->storage.parts.size());
	m_impl->cpu([&]
	{
		impl::check_mapping(surface);
		for (auto& part : surface->storage.parts) part.coverage = m_impl->memory->watch_surface(part.coverage);
		impl::check_mapping(surface);
	});
	m_impl->surfaces.push_back(surface);
	try
	{
		m_impl->positions.emplace(surface.get(), m_impl->surfaces.size() - 1);
		m_impl->descriptors.emplace(hash, surface.get());
	}
	catch (...)
	{
		m_impl->positions.erase(surface.get());
		m_impl->surfaces.pop_back();
		throw;
	}
	surface->last_use = ++m_impl->clock;
	m_impl->ranges_dirty = true;
	m_impl->metadata.entries = m_impl->surfaces.size();
	m_impl->metadata.watch_parts += surface->storage.parts.size();
	return surface;
}
cached_surface_status surface_cache::status(const cached_surface_ref& surface) const
{
	m_impl->check(surface);
	return {surface->initialized, surface->dirty, bool(surface->spill.buffer), surface->faulted, bool(m_impl->pending || m_impl->recording_batch), surface->revision.size() == 1 ? surface->revision[0] : 0, surface->generation, surface->deferred.size(), surface->native_write_epoch, surface->revision};
}
void surface_cache::prepare(const cached_surface_ref& surface, surface_initialization initialization)
{
	m_impl->prepare(surface, initialization);
}
cached_surface_description surface_cache::description(const cached_surface_ref& surface) const
{
	m_impl->check(surface);
	return surface->description;
}
void surface_cache::clear(const cached_surface_ref& surface, std::uint32_t mask, std::uint32_t color, std::uint32_t zstencil, std::uint32_t stencil_mask, clear_rectangle rectangle)
{
	m_impl->check(surface);
	const auto clipped = clip_clear_rectangle(rectangle, surface->description.allocation.width, surface->description.allocation.height);
	if (!clipped.width || !clipped.height)
		return;
	bool changed = false, full = false;
	if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
	{
		const auto plan = decode_color_clear(*format, color, mask);
		changed = plan.write_mask != 0;
		full = changed && plan.write_mask == decode_color_clear(*format, color, 0xf0).write_mask;
	}
	else
	{
		const auto depth = describe_surface_depth(std::get<surface_depth>(surface->description.allocation.format));
		changed = (mask & 1) || ((mask & 2) && depth.stencil && (stencil_mask & 255));
		full = (mask & 1) && (!depth.stencil || ((mask & 2) && (stencil_mask & 255) == 255));
	}
	if (!changed)
		return;
	const bool full_rectangle = clipped.x == 0 && clipped.y == 0 && clipped.width == surface->description.allocation.width &&
		clipped.height == surface->description.allocation.height;
	full &= full_rectangle;
	const auto clear_upload = clear_engine::upload_for(surface->description.allocation.format, mask, stencil_mask, full_rectangle);
	auto prefetched = !full && m_impl->batch_capacity ?
		m_impl->prefetch_guest_target(surface, *this, std::min(m_impl->snapshot_limit, target_prefetch_limit)) : std::nullopt;
	std::optional<submission> early_clear;
	std::optional<staged_target_load> early_load;
	std::optional<upload_plan> uploads;
	if (m_impl->batch_capacity && m_impl->pending && m_impl->ctx.can_begin_after(m_impl->pending->work))
	{
		if (full && !clear_upload && m_impl->independent_cold_target(surface, *this))
			early_clear = m_impl->pending->work;
		else if (!full && m_impl->pending)
		{
			early_load = m_impl->plan_independent_write(surface, *this, prefetched ? &*prefetched : nullptr);
			if (early_load && m_impl->pending)
			{
				if (clear_upload) uploads = m_impl->ctx.plan_uploads(std::span{&*clear_upload, 1}, false);
				if (!clear_upload || uploads) early_clear = m_impl->pending->work;
			}
		}
	}
	if (!early_clear)
	{
		early_load.reset();
		m_impl->synchronize(m_impl->timeout, false);
	}
	m_impl->batch_surface_boundary(surface, true);
	if (clear_upload && !uploads)
		prepare_draw_resources([&] { uploads = m_impl->ctx.prepare_uploads(std::span{&*clear_upload, 1}); });
	if (!early_clear) m_impl->preserve_readers(surface->lease, surface);
	surface->last_use = ++m_impl->clock;
	std::vector<std::uint64_t> revision;
	std::optional<draw_target_load> target_load;
	if (full)
	{
		if (!early_clear) m_impl->flush_aliases(surface);
		revision = m_impl->current(surface);
		if (!m_impl->recorded_draw_target(surface))
		{
			if (m_impl->batch_capacity && !surface->lease && m_impl->needs_target_load(surface))
				// A full overwrite only needs a private target, with no old contents.
				target_load.emplace(draw_target_load{surface});
			else
				m_impl->allocate(surface);
		}
	}
	else if (early_clear)
	{
		revision = std::move(early_load->second.revisions);
		if (early_load->first.has_input()) target_load.emplace(std::move(early_load->first));
		early_load.reset();
		prefetched.reset();
	}
	else if (const auto* previous = m_impl->recorded_draw_target(surface))
		revision = previous->revisions;
	else
	{
		surface_list_guard pin(m_impl->pinned, surface);
		if (auto staged = m_impl->stage_target_load(surface, *this, {}, prefetched ? &*prefetched : nullptr))
		{
			target_load.emplace(std::move(staged->first));
			revision = std::move(staged->second.revisions);
		}
		else
		{
			target_reload reload{*this};
			m_impl->prepare(surface, surface_initialization::guest, {}, &reload);
			if (reload.staged)
			{
				auto& [load, validation] = *reload.staged;
				target_load.emplace(std::move(load));
				revision = std::move(validation.revisions);
			}
			else revision = surface->revision;
		}
	}
	const auto generations = 1u + (target_load ? target_load->generations() : 0u);
	const auto loads = target_load ? std::span<const draw_target_load>{&*target_load, 1} : std::span<const draw_target_load>{};
	const auto aliases = m_impl->plan_aliases(surface);
	const auto native_epoch = m_impl->reserve_native_epoch();
	const auto encode = [&]
	{
		if (!full && target_load) m_impl->encode_target_load(*target_load);
		clear_request request;
		request.mask = mask;
		request.color_value = color;
		request.zstencil_value = zstencil;
		request.stencil_write_mask = stencil_mask;
		request.rectangle = clipped;
		if (const auto format = std::get_if<surface_color>(&surface->description.allocation.format))
			request.colors[0] = {{surface->lease->render_texture(m_impl->ctx), 0, 0}, *format};
		else
		{
			request.depth = {surface->lease->render_texture(m_impl->ctx), 0, 0};
			request.depth_format = std::get<surface_depth>(surface->description.allocation.format);
		}
		if (uploads) m_impl->ctx.with_uploads(*uploads, [&] { (void)m_impl->clears.run(m_impl->ctx, request); });
		else (void)m_impl->clears.run(m_impl->ctx, request);
	};
	std::function<void()> commit = [this, surface, revision, aliases, native_epoch, generations]
	{
		surface->revision = revision;
		surface->initialized = true;
		surface->dirty = true;
		surface->native_write_epoch = native_epoch;
		surface->faulted = false;
		surface->spill = {};
		surface->deferred.clear();
		surface->generation += generations;
		m_impl->invalidate_aliases(aliases);
	};
	if (early_clear)
	{
		std::vector<surface_validation> validated{{surface, revision}};
		if (aliases.empty() && m_impl->record_independent_targets(encode, {}, commit, {surface}, validated, *early_clear, loads, !full)) return;
		m_impl->synchronize(m_impl->timeout, false);
		uploads.reset();
		target_load.reset();
		return clear(surface, mask, color, zstencil, stencil_mask, rectangle);
	}
	m_impl->execute(encode, std::move(commit), {surface}, {{surface, revision}}, aliases,
		false, true, !full, true, false, {}, nullptr, loads);
}
void surface_cache::draw(std::span<const cached_surface_use> uses, const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode,
	draw_publication publish_draw, const submission* predecessor)
{
	draw_impl(uses, encode, false, nullptr, nullptr, {}, {}, false, false, false, std::move(publish_draw), predecessor);
}
void surface_cache::draw_feedback(std::span<const cached_surface_use> uses, const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode)
{
	draw_impl(uses, encode, true);
}
void surface_cache::draw_depth_bounds(std::span<const cached_surface_use> uses, std::size_t depth_index,
									  const std::function<MTL4RenderPassDescriptor*(std::span<const id<MTLTexture>>)>& descriptor,
									  const std::function<void(context&, std::span<const id<MTLTexture>>, depth_bounds_pass&)>& encode, bool feedback,
									  draw_publication publish_draw, const submission* predecessor)
{
	if (depth_index >= uses.size() || !uses[depth_index].write || !descriptor || !encode)
		throw std::invalid_argument("Cached depth bounds require a depth write target and both callbacks");
	m_impl->check(uses[depth_index].surface);
	const auto& allocation = uses[depth_index].surface->description.allocation;
	if (!std::holds_alternative<surface_depth>(allocation.format))
		throw std::invalid_argument("Cached depth bounds target must have a depth format");
	if (!m_impl->bounds)
		m_impl->bounds = std::make_unique<depth_bounds_engine>(m_impl->ctx);
	const surface_allocation_description mirror_description{allocation.width, allocation.height, allocation.samples, surface_color::r32f, false};
	surface_lease mirror;
	draw_impl(uses, [&](context& ctx, auto textures)
			  {
				  MTL4RenderPassDescriptor* pass = descriptor(textures);
				  if (!pass || pass.depthAttachment.texture != textures[depth_index] || pass.depthAttachment.level || pass.depthAttachment.slice)
					  throw std::invalid_argument("Cached depth bounds descriptor must select the supplied depth write texture");
				  auto prepared = m_impl->bounds->prepare(ctx, pass, mirror);
				  encode(ctx, textures, prepared);
				  prepared.end(); }, feedback, &mirror_description, &mirror, {}, {}, false, false, false, std::move(publish_draw), predecessor);
}
std::optional<surface_cache::scaled_image_endpoints> surface_cache::prepare_scaled_image(const scaled_image_plan& plan,
	std::chrono::milliseconds timeout, std::size_t writer_limit, std::size_t byte_limit)
{
	m_impl->outside_epoch();
	if (!writer_limit || writer_limit > scaled_image_max_writers)
		throw std::invalid_argument("Scaled-image native writer budget is invalid");
	// The closed native route publishes metadata under the cache's producer
	// exclusion. Legacy callers without that exclusion retain the VM route.
	if(!m_impl->capture)return {};
	prepare_texture_boundary(true);m_impl->synchronize(timeout);
	scaled_image_endpoints result;
	std::array<std::optional<cached_surface_description>,2> create;
	std::vector<cached_surface_description> create_outputs;
	std::vector<std::pair<guest_surface_range,std::size_t>> input_holes;
	std::vector<cached_surface_ref> protected_owners;
	const auto native_limit=max_2d_texture_dimension(m_impl->ctx.device());
	bool complete=true, single_target_proved=false;
	std::optional<clear_rectangle> single_target_region;
	m_impl->cpu([&]
	{
		for(unsigned side=0;side<2&&complete;++side)
		{
			const auto& image=side?plan.destination:plan.source;
			const auto requested=scaled_image_storage(image,side?plan.destination_tile:plan.source_tile,
				side?plan.destination_segments:plan.source_segments);
			const auto projected=project_display_storage_source(requested);
			std::vector<guest_surface_layout> coverage;for(const auto& part:projected)coverage.push_back(part.coverage);
			const auto candidates=m_impl->cpu_aliases(coverage,true);
			const auto protected_owner = [&](const auto& surface)
			{
				return std::find(protected_owners.begin(), protected_owners.end(), surface) != protected_owners.end();
			};
			if (side && std::any_of(candidates.begin(), candidates.end(), protected_owner))
			{
				complete = false;
				break;
			}
			const auto dirty=std::count_if(candidates.begin(),candidates.end(),[](const auto& s){return s->dirty||!s->deferred.empty();});
			if (side && (!candidates.empty() || image.width > native_limit || image.height > native_limit))
			{
				// A scale operation may span several disjoint native targets.
				// Prove every active output byte against their native windows
				// before preparation, so a refusal cannot flush dirty owners.
				const auto format = plan.destination_format == scaled_image_format::rgb565 ? surface_color::rgb565 : surface_color::argb8;
				std::vector<cached_surface_ref> outputs;
				std::vector<scaled_image_output_region> rectangles;
				std::vector<scaled_image_output_byte_region> byte_regions;
				std::vector<surface_format> output_formats;
				std::vector<display_storage_source> output_storage;
				std::size_t pixels = 0, mapped_pixels = 0;
				bool partial = false, byte_formats = false;
				bool supported = candidates.size() <= writer_limit;
				bool versioned = true, native_loads = true, readers = false, self_feedback = false;
				for (const auto& target : candidates)
				{
					if (!supported) break;
					m_impl->check_mapping(target);
					const auto& d = target->description;
					const auto& a = d.allocation;
					if (target->faulted ||
						(target->deferred.empty() && (!target->initialized || (!target->lease && !target->spill.buffer))))
					{ supported = false; break; }
					const auto& revision = target->deferred.empty() ? target->revision : target->deferred_revision;
					if (m_impl->current(target) != revision)
					{
						if (target->dirty || !target->deferred.empty()) throw surface_conflict_error(target->layout.range);
						supported = false; break;
					}
					const auto size = a.samples == 1 ? std::array{a.width, a.height} : sample_expanded_size(a.width, a.height, a.samples);
					auto geometry = plan_guest_surface(d.tile ? d.tile->address : d.address, size[0], size[1], surface_guest_bytes(a.format), d.morton ? 0 : d.row_pitch, d.morton);
					if (d.tile) geometry.range = {d.tile->address, d.tile->size};
					const auto storage = scaled_image_storage(geometry, d.tile, d.segments);
					if (candidates.size() == 1 && a.format == surface_format{format})
					{
						single_target_proved = true;
						single_target_region = plan_native_scaled_image_region(*m_impl->memory, storage, requested);
						if (single_target_region) { supported = false; break; }
					}
					const auto region = plan_native_scaled_image_region(*m_impl->memory, requested, storage);
					byte_formats |= a.format != surface_format{format};
					partial |= !region.has_value();
					if (region) mapped_pixels += region->width * region->height;
					for (const auto& previous : outputs)
						if (memory_storage_overlap(*m_impl->memory, target->storage, previous->storage)) supported = false;
					// A source which is this exact authority keeps its immutable
					// old lease while all output targets receive native versions.
					// Distinct alias owners still need a broader authority plan.
					const auto read_target = [&](const auto& input)
					{
						if (input == target) self_feedback = true;
						else if (input && memory_storage_overlap(*m_impl->memory, target->storage, input->storage)) supported = false;
					};
					read_target(result.surfaces[0]);
					for (const auto& input : result.inputs) read_target(input);
					for (const auto& alias : m_impl->plan_aliases(target))
						if (alias->dirty || !alias->deferred.empty()) supported = false;
					// A dirty retained backing can outlive its original VA.
					// Current logical disjointness alone cannot order that owner.
					for (const auto& other : m_impl->surfaces)
					{
						if (other == target || (!other->dirty && other->deferred.empty())) continue;
						if (!other->physical) { supported = false; break; }
						if (target->physical)
						{
							for (const auto& a : *target->physical) for (const auto& b : *other->physical)
								if (a->overlaps(*b)) supported = false;
						}
						else for (const auto& ticket : *other->physical) for (const auto& part : target->storage.parts)
						{
							guest_surface_byte_cursor cursor(part.coverage);
							while (const auto run = cursor.next())
							{
								const auto overlap = m_impl->memory->physical_surface_overlaps_current(*ticket, *run);
								if (!overlap || *overlap) supported = false;
							}
						}
					}
					for (const auto& owner : m_impl->surfaces)
						if (target->lease && owner != target && std::any_of(owner->deferred.begin(), owner->deferred.end(),
							[&](const auto& section) { return section.source == target->lease; })) readers = true;
					versioned &= target->initialized && target->lease && !target->spill.buffer && target->deferred.empty();
					if (!target->deferred.empty())
						native_loads &= deferred_copy_start(target).second || current_deferred_background(*target);
					outputs.push_back(target); output_storage.push_back(storage);
					output_formats.push_back(a.format);
					rectangles.push_back({region.value_or(clear_rectangle{}), {0, 0, size[0], size[1]}, outputs.size()-1});
				}
				if (supported && (byte_formats || partial || mapped_pixels != image.width * image.height))
				{
					std::vector<display_storage_source> cold;
					const auto projected = byte_formats ? std::optional<std::vector<scaled_image_output_region>>{} :
						plan_scaled_image_output_regions(*m_impl->memory, output_storage, requested,
							1u << 20, 4096, &cold, native_limit, writer_limit);
					if (projected)
					{
						rectangles = *projected;
					}
					else if (const auto bytes = plan_scaled_image_output_byte_regions(*m_impl->memory, output_storage,
						requested, std::min(byte_limit, m_impl->snapshot_limit), 1u << 20, 4096, &cold, native_limit, writer_limit))
					{
						byte_regions = *bytes;
						rectangles.clear();
					}
					else supported = false;
					if (supported)
					{
						// Retained dirty owners may have lost their original VA.
						// They must not be mistaken for a missing native word.
						for (const auto& storage : cold) for (const auto& part : project_display_storage_source(storage))
							for (const auto& other : m_impl->surfaces)
							{
								if (!other->dirty && other->deferred.empty()) continue;
								if (!other->physical) { supported = false; break; }
								for (const auto& ticket : *other->physical)
								{
									guest_surface_byte_cursor cursor(part.coverage);
									while (const auto run = cursor.next())
									{
										const auto overlap = m_impl->memory->physical_surface_overlaps_current(*ticket, *run);
										if (!overlap || *overlap) supported = false;
									}
								}
							}
						if (supported)
						{
							for (const auto& storage : cold)
							{
								const auto& g = storage.geometry;
								auto segments = storage.segments;
								if ((g.height > 1 || g.morton) && segments.size() == 1 && segments[0].offset == 0 && segments[0].size == g.range.size) segments.clear();
								const auto cold_format = g.pixel_bytes == 1 ? surface_color::b8 : format;
								create_outputs.push_back({storage.segments.front().address,
										{g.width, g.height, 1, cold_format, false}, g.morton ? 0 : g.row_pitch, g.morton, {}, std::move(segments)});
								output_formats.push_back(cold_format);
							}
						}
					}
				}
				if (supported && !byte_regions.empty())
					pixels = checked_texture_product(image.width, image.height);
				else if (supported)
				{
					std::vector<clear_rectangle> coverage;
					for (const auto& region : rectangles)
					{
						pixels = checked_texture_sum(pixels, checked_texture_product(region.packed.width, region.packed.height));
						coverage.push_back(region.packed);
					}
					supported = rectangles_cover(coverage, image.width, image.height);
				}
				if (supported && pixels == image.width * image.height && (!readers || versioned || native_loads))
				{
					result.outputs = std::move(outputs);
					result.output_regions = std::move(rectangles);
					result.output_byte_regions = std::move(byte_regions);
					result.output_formats = std::move(output_formats);
					if (!result.outputs.empty()) result.surfaces[1] = result.outputs.front();
					result.regions[1] = {0, 0, image.width, image.height};
					// Known native backgrounds and frozen deferred leases can be
					// loaded into private output versions in the scale transaction.
					// No old reader needs materialization or an earlier submission.
					result.version_target = readers || self_feedback || (native_loads && !versioned);
					result.prepare_version_targets = self_feedback && !versioned && !native_loads;
					result.frozen_inputs |= self_feedback;
					continue;
				}
				create_outputs.clear();
			}
			if(dirty>1&&side){complete=false;break;}
			if(side&&!result.inputs.empty()&&std::any_of(result.inputs.begin(),result.inputs.end(),[&](const auto& input)
				{return std::find(candidates.begin(),candidates.end(),input)!=candidates.end();})){complete=false;break;}
			const auto expected=(side?plan.destination_format:plan.source_format)==scaled_image_format::rgb565?surface_color::rgb565:surface_color::argb8;
			for(const auto& surface:candidates)
			{
				if(dirty>1)break;
				if(dirty&&!surface->dirty&&surface->deferred.empty())continue;
				m_impl->check_mapping(surface);
				const bool deferred=!surface->deferred.empty();
				if(surface->faulted||(!deferred&&(!surface->initialized||(!surface->lease&&!surface->spill.buffer))))continue;
				if(deferred)
				{
					// Resolving a queued owner is GPU-only except for a genuine
					// missing background. Other dirty aliases require separate
					// authority assembly;do not flush them as a side effect.
					const auto aliases=m_impl->plan_aliases(surface);
					bool competing=std::any_of(aliases.begin(),aliases.end(),[](const auto& s){return s->dirty||!s->deferred.empty();});
					// Retained physical owners need not still share a current VA.
					if(!competing&&surface->physical)for(const auto& other:m_impl->surfaces)
					{
						if(other==surface||(!other->dirty&&other->deferred.empty()))continue;
						if(!other->physical){competing=true;break;}
						for(const auto& a:*surface->physical)for(const auto& b:*other->physical)if(a->overlaps(*b))competing=true;
						if(competing)break;
					}
					if(competing)continue;
				}
				if(m_impl->current(surface)!=(deferred?surface->deferred_revision:surface->revision))
				{
					if(surface->dirty||deferred)throw surface_conflict_error(surface->storage.parts.front().coverage.range);
					continue;
				}
				if(surface->description.allocation.format!=surface_format{expected})continue;
				const auto& description=surface->description;
				const auto size=description.allocation.samples==1?std::array{description.allocation.width,description.allocation.height}:
					sample_expanded_size(description.allocation.width,description.allocation.height,description.allocation.samples);
				auto geometry=plan_guest_surface(description.tile?description.tile->address:description.address,size[0],size[1],image.pixel_bytes,
					description.morton?0:description.row_pitch,description.morton);
				if(description.tile)geometry.range={description.tile->address,description.tile->size};
				const auto storage=scaled_image_storage(geometry,description.tile,description.segments);
				const auto region=side&&single_target_proved?single_target_region:plan_native_scaled_image_region(*m_impl->memory,storage,requested);
				if(!region)continue;
				if(result.surfaces[side]&&result.surfaces[side]->native_write_epoch>=surface->native_write_epoch)continue;
				result.surfaces[side]=surface;result.regions[side]=*region;
			}
			if(!side&&!result.surfaces[0]&&!candidates.empty())
			{
				// Resolve byte authority before preparing any source. Preparing
				// overlapping dirty owners would otherwise flush one to VM.
				std::vector<display_storage_source> layouts;
				std::vector<cached_surface_ref> sources;
				bool supported=true;
				for(const auto& surface:candidates)
				{
					m_impl->check_mapping(surface);
					if(surface->faulted)
					{if(surface->dirty||!surface->deferred.empty())supported=false;continue;}
					if(m_impl->current(surface)!=(surface->deferred.empty()?surface->revision:surface->deferred_revision))
					{if(surface->dirty||!surface->deferred.empty())throw surface_conflict_error(surface->storage.parts.front().coverage.range);continue;}
					if(!surface->initialized&&surface->deferred.empty())continue;
					const auto& d=surface->description;const auto& a=d.allocation;
					const auto size=a.samples==1?std::array{a.width,a.height}:sample_expanded_size(a.width,a.height,a.samples);
					auto storage=plan_guest_surface(d.tile?d.tile->address:d.address,size[0],size[1],surface_guest_bytes(a.format),d.row_pitch,d.morton);
					if(d.tile)storage.range={d.tile->address,d.tile->size};
					// Native storage extent is address metadata,not a pixel snapshot.
					// Scaled-image packing reserves bounded selected texel regions.
					auto layout=scaled_image_storage(storage,d.tile,d.segments);layout.dirty=surface->dirty||!surface->deferred.empty();
					layout.epoch=surface->deferred.empty()?surface->native_write_epoch:surface->deferred.back().write_epoch;
					layouts.push_back(std::move(layout));sources.push_back(surface);
				}
				if(supported&&!sources.empty()&&sources.size()<=64)if(const auto spans=scaled_image_dense_segments(requested))
				{
					bool identities=true;for(const auto& span:*spans)if(!m_impl->memory->storage_identity({span.address,span.size})){identities=false;break;}
					if(!identities){complete=false;break;}
					// Aliased deferred readers must stay private. For a partial
					// plan, the uncovered bytes have separate authority.
					std::vector<bool> full_coverage(sources.size(), true);
					bool overlapping_deferred = false;
					for (std::size_t i = 0; i < sources.size(); ++i)
					{
						if (sources[i]->deferred.empty()) continue;
						full_coverage[i] = complete_deferred_input(sources[i]);
						if (!overlapping_deferred)
							for (std::size_t j = 0; j < sources.size(); ++j)
								if (i != j && memory_storage_overlap(*m_impl->memory, sources[i]->storage, sources[j]->storage))
								{
									overlapping_deferred = true;
									break;
								}
					}
					std::vector<std::vector<texture_storage_run>> epoch_storage(sources.size());
					std::size_t epoch_checks = 0;
					display_storage_epochs epochs;
					for (std::size_t i = 0; i < sources.size(); ++i)
					{
						const auto& sections = sources[i]->deferred;
						if (sections.empty() || !overlapping_deferred) continue;
						if (!full_coverage[i])
						{
							epoch_storage[i] = texture_storage_runs(layouts[i]);
							continue;
						}
						if (sections.size() < 2) continue;
						for (std::size_t j = 0; j < sources.size(); ++j)
						{
							if (i == j || !layouts[j].dirty) continue;
							const auto first = sources[j]->deferred.empty() ? layouts[j].epoch : sources[j]->deferred.front().write_epoch;
							if (full_coverage[j] && (sections.front().write_epoch > layouts[j].epoch || sections.back().write_epoch < first)) continue;
							if (!memory_storage_overlap(*m_impl->memory, sources[i]->storage, sources[j]->storage)) continue;
							epoch_storage[i] = texture_storage_runs(layouts[i]);
							break;
						}
					}
					if (std::any_of(epoch_storage.begin(), epoch_storage.end(), [](const auto& runs) { return !runs.empty(); }))
						epochs = [&](auto input, auto offset, auto size)
						{
							if (epoch_storage[input].empty()) return display_storage_epoch_run{size, layouts[input].epoch, layouts[input].dirty};
							return deferred_input_epoch(*sources[input], layouts[input].geometry, epoch_storage[input], offset, size, epoch_checks);
						};
					auto assembly=plan_display_storage_assembly(*m_impl->memory,plan_guest_surface(0,image.width,image.height,image.pixel_bytes),layouts,*spans,epochs);
					if(!assembly.copies.empty() || overlapping_deferred)
					{
						std::vector<std::size_t> selected;
						for(const auto& copy:assembly.copies)if(std::find(selected.begin(),selected.end(),copy.input)==selected.end())selected.push_back(copy.input);
						const bool frozen = overlapping_deferred;
						const bool ram_only = assembly.copies.empty();
						// An unselected owner can stay queued only if its retained
						// physical backing still agrees with the virtual plan.
						std::vector<cached_surface_ref> current_owners;
						if (frozen)
							for (const auto& source : sources)
							{
								if (!source->dirty && source->deferred.empty()) continue;
								const auto& revisions = source->deferred.empty() ? source->revision : source->deferred_revision;
								bool current = !source->physical || source->physical->size() == revisions.size();
								for (std::size_t k = 0; current && source->physical && k < revisions.size(); ++k)
									current = m_impl->memory->physical_surface_virtual_current(*(*source->physical)[k], revisions[k]).value_or(false);
								if (current) current_owners.push_back(source);
							}
						const auto known_owner = [&](const auto& surface)
						{
							return std::find(current_owners.begin(), current_owners.end(), surface) != current_owners.end();
						};
						const bool readable = !frozen || std::all_of(selected.begin(), selected.end(), [&](auto i)
						{
							const auto& source = sources[i];
							if (!source->deferred.empty())
								return full_coverage[i] || !source->dirty || current_deferred_background(*source);
							return source->initialized && (source->lease || source->spill.buffer);
						});
						bool isolated=readable && selected.size()<=64 && (!ram_only ||
							std::all_of(sources.begin(), sources.end(), [&](const auto& source)
							{ return (!source->dirty && source->deferred.empty()) || known_owner(source); }));
						for(const auto i:selected)
						{
							const auto admitted=[&](const auto& other){return frozen&&(known_owner(other)||std::any_of(selected.begin(),selected.end(),[&](auto j){return sources[j]==other;}));};
							for(const auto& other:m_impl->plan_aliases(sources[i]))if((other->dirty||!other->deferred.empty())&&!admitted(other))isolated=false;
							// Known current owners can remain queued even if
							// their bytes lost the authority sweep.
							// A retained old output may overlap physical storage after
							// losing its original VA. Do not mistake it for clean RAM.
							if(sources[i]->physical)for(const auto& other:m_impl->surfaces)
							{
								if(other==sources[i]||admitted(other)||(!other->dirty&&other->deferred.empty()))continue;
								if(!other->physical){isolated=false;break;}
								for(const auto& a:*sources[i]->physical)for(const auto& b:*other->physical)if(a->overlaps(*b))isolated=false;
							}
							if(!sources[i]->physical)for(const auto& other:m_impl->surfaces)
							{
								if(other==sources[i]||admitted(other)||!other->physical||(!other->dirty&&other->deferred.empty()))continue;
								for(const auto& ticket:*other->physical)for(const auto& part:sources[i]->storage.parts)
								{
									guest_surface_byte_cursor cursor(part.coverage);while(const auto run=cursor.next())
									{const auto match=m_impl->memory->physical_surface_overlaps_current(*ticket,*run);if(!match||*match)isolated=false;}
								}
							}
						}
						if(isolated)
						{
							result.frozen_inputs=frozen && !ram_only;
							result.ram_only=ram_only;
							if (ram_only || std::any_of(current_owners.begin(), current_owners.end(), [&](const auto& owner)
								{ return std::none_of(selected.begin(), selected.end(), [&](auto i) { return sources[i] == owner; }); }))
								result.ownership_epoch = m_impl->texture_topology_epoch;
							protected_owners=std::move(current_owners);
							// The complement is expressed in packed pixel order.
							// Rebase only those missing bytes to actual EAs;never
							// capture source pixels supplied by a native authority.
							for(const auto& hole:assembly.vm)for(std::size_t at=hole.address,end=at+hole.size;at<end;)
							{
								const auto* part=texture_assembly_segment(*spans,at);if(!part)throw std::logic_error("Scaled-image RAM hole lacks storage metadata");
								const auto count=std::min(end-at,part->size-(at-part->offset));
								const guest_surface_range range{static_cast<std::uint32_t>(std::uint64_t(part->address)+at-part->offset),count};
								// A retained dirty owner may have lost its original VA.
								// Its physical bytes must never become an apparent RAM hole.
								for(const auto& other:m_impl->surfaces)if(other->physical&&(other->dirty||!other->deferred.empty())&&
									(std::find(sources.begin(),sources.end(),other)==sources.end() ||
									 (frozen && !protected_owner(other))))for(const auto& ticket:*other->physical)
								{const auto match=m_impl->memory->physical_surface_overlaps_current(*ticket,range);if(!match||*match)throw surface_conflict_error(range);}
								if(input_holes.size()==(1u<<20))throw std::length_error("Scaled-image RAM hole metadata exceeds budget");
								input_holes.push_back({range,at});at+=count;
							}
							for(auto& copy:assembly.copies)copy.input=std::find(selected.begin(),selected.end(),copy.input)-selected.begin();
							for(const auto i:selected){result.inputs.push_back(sources[i]);result.descriptions.push_back(sources[i]->description);}
							result.assembly=std::move(assembly);result.regions[0]={0,0,image.width,image.height};
						}
					}
				}
			}
			const auto& tile=side?plan.destination_tile:plan.source_tile;
			const auto& segments=side?plan.destination_segments:plan.source_segments;
			if (side && result.surfaces[side])
			{
				const auto& target = result.surfaces[side];
				const auto aliases = m_impl->plan_aliases(target);
				const auto reads_target = [&](const auto& owner)
				{
					return target->lease && owner != target &&
						std::any_of(owner->deferred.begin(), owner->deferred.end(),
							[&](const auto& section) { return section.source == target->lease; });
				};
				const bool has_readers = std::any_of(m_impl->surfaces.begin(), m_impl->surfaces.end(), reads_target);
				const bool resident = target->deferred.empty() && target->initialized && target->lease && !target->spill.buffer;
				const bool self_feedback = result.surfaces[0] == target;
				const bool native_load = target->deferred.empty() ? target->initialized && (target->lease || target->spill.buffer) :
					deferred_copy_start(target).second || current_deferred_background(*target);
				result.version_target = ((has_readers && resident) || (self_feedback && native_load)) &&
					m_impl->current(target) == (target->deferred.empty() ? target->revision : target->deferred_revision) &&
					std::none_of(aliases.begin(), aliases.end(), [](const auto& alias) { return alias->dirty || !alias->deferred.empty(); });
				result.frozen_inputs |= self_feedback && result.version_target;
				if (std::any_of(aliases.begin(), aliases.end(), protected_owner) ||
					(!result.version_target && std::any_of(protected_owners.begin(), protected_owners.end(), reads_target)))
				{
					complete = false;
					break;
				}
			}
			if(!result.surfaces[side]&&(side||(!result.ram_only&&result.inputs.empty()))&&!dirty&&image.width<=native_limit&&image.height<=native_limit&&
				((!tile&&segments.empty())||image.range.size<=m_impl->snapshot_limit))
			{
				// Missing input captures only its requested raster through the
				// ordinary coherent guest upload. Missing output needs no VM
				// pixels:its entire active raster will be overwritten on GPU.
				// Pitch/Morton/tile exterior stays in guest storage.
				auto address=image.range.address;
				if(tile)
				{
					if(segments.empty())address=tile->address+tile->offset;
					else
					{
						const auto* part=texture_assembly_segment(segments,tile->offset);
						if(!part)throw std::logic_error("Native scaled-image storage lacks its pixel segment");
						address=static_cast<std::uint32_t>(std::uint64_t(part->address)+tile->offset-part->offset);
					}
				}
				create[side]=cached_surface_description{address,{image.width,image.height,1,expected,false},
					image.morton?0:image.row_pitch,image.morton,tile,segments};
			}
			complete=bool(result.surfaces[side])||create[side].has_value()||(!side&&(result.ram_only||!result.inputs.empty()));
		}
		if(complete&&!input_holes.empty())
		{
			std::vector<guest_surface_layout> watched;std::vector<std::uint64_t> revisions;
			for(const auto& [range,target]:input_holes)
			{
				if(range.size>m_impl->snapshot_limit-result.input_bytes.size())throw std::length_error("Scaled-image RAM holes exceed snapshot budget");
				auto coverage=m_impl->memory->watch_surface(guest_byte_coverage(range));
				const auto captured=m_impl->memory->read_surface(coverage);
				if(captured.bytes.size()!=range.size)throw std::runtime_error("Scaled-image RAM hole snapshot is incomplete");
				result.input_patches.push_back({target,result.input_bytes.size(),range.size,0});
				result.input_bytes.insert(result.input_bytes.end(),captured.bytes.begin(),captured.bytes.end());
				watched.push_back(std::move(coverage));revisions.push_back(captured.revision);
			}
			result.input_revision=std::make_shared<surface_texture_revision>(m_impl->memory,std::move(watched),std::move(revisions),nullptr);
		}
	});
	if(!complete)return {};
	// Metadata admission must not retire either selected endpoint while its
	// missing peer is registered. Handles remain cold until draw preparation.
	surface_list_guard pins(m_impl->pinned,result.surfaces);
	surface_list_guard input_pins(m_impl->pinned,result.inputs);
	surface_list_guard output_pins(m_impl->pinned,result.outputs);
	for(unsigned side=0;side<2;++side)if(create[side])
	{
		result.surfaces[side]=acquire(*create[side]);
		if(result.surfaces[side]->dirty||!result.surfaces[side]->deferred.empty())
			throw std::logic_error("Native scaled-image cold endpoint already owns output");
		m_impl->pinned.push_back(result.surfaces[side]);
		const auto& image=side?plan.destination:plan.source;
		result.regions[side]={0,0,image.width,image.height};
		if(side==1)result.cold_outputs.push_back(result.surfaces[1]);
	}
	for (const auto& description : create_outputs)
	{
		const auto target = acquire(description);
		if (target->dirty || !target->deferred.empty()) throw std::logic_error("Cold output hole already owns native data");
		result.outputs.push_back(target); result.cold_outputs.push_back(target); m_impl->pinned.push_back(target);
	}
	if (!result.surfaces[1] && !result.outputs.empty()) result.surfaces[1] = result.outputs.front();
	// Only a feedback target whose missing background cannot be loaded from
	// native storage still needs a separate preparation prefix. Known native
	// spill/deferred backgrounds are encoded into private output versions below.
	if (result.prepare_version_targets)
		for (const auto& target : result.outputs)
			if (std::find(result.cold_outputs.begin(), result.cold_outputs.end(), target) == result.cold_outputs.end())
				m_impl->prepare(target, surface_initialization::guest);
	return result;
}
void surface_cache::draw_scaled_image(const scaled_image_endpoints& endpoints,
	std::span<const std::vector<clear_rectangle>> input_windows,
	const std::function<void(context&,std::span<const id<MTLTexture>>,std::span<const std::vector<scaled_image_input_view>>)>& encode,
	std::function<void()> publication_check,std::chrono::milliseconds timeout,
	const std::function<void(context&)>& before_loads, bool defer_completion, std::function<void()> after_commit)
{
	m_impl->outside_epoch();
	const auto previous=m_impl->timeout;
	struct restore_timeout{impl& state;std::chrono::milliseconds value;~restore_timeout(){state.timeout=value;}} restore{*m_impl,previous};
	m_impl->timeout=timeout;
	surface_list_guard pins(m_impl->pinned,endpoints.surfaces);
	surface_list_guard input_pins(m_impl->pinned,endpoints.inputs);
	surface_list_guard output_pins(m_impl->pinned,endpoints.outputs);
	std::vector<cached_surface_use> uses;
	if (endpoints.ram_only && !endpoints.input_revision)
		throw std::logic_error("RAM-only scaled transfer lacks a source revision");
	if(endpoints.surfaces[0])uses.push_back({endpoints.surfaces[0],false});
	else for(const auto& input:endpoints.inputs)uses.push_back({input,false});
	if (endpoints.outputs.empty()) uses.push_back({endpoints.surfaces[1],true});
	else for (const auto& output : endpoints.outputs) uses.push_back({output,true});
	if (endpoints.ownership_epoch)
		publication_check = [this, check=std::move(publication_check), epoch=*endpoints.ownership_epoch]
		{
			if (check) check();
			if (m_impl->texture_topology_epoch != epoch)
				throw std::runtime_error("Scaled-image queued source ownership changed before native output publication");
		};
	if(endpoints.input_revision)publication_check=[check=std::move(publication_check),receipt=endpoints.input_revision]
	{
		if(check)check();
		if(!receipt->current())throw std::runtime_error("Scaled-image RAM input changed before native output publication");
	};
	bool frozen_inputs=endpoints.frozen_inputs;
	if (!frozen_inputs && !input_windows.empty())
		m_impl->cpu([&]
		{
			// An isolated native reader needs neither VM initialization nor
			// a persistent full-image restoration. Keep its current lease/spill
			// and queued copy epochs private; selected windows are encoded with
			// the output transaction below. Other alias/preparation paths retain
			// their existing coherence boundary.
			for (const auto& use:uses) if (!use.write)
			{
				const auto& source=use.surface;
				if (source->faulted || m_impl->current(source)!=(source->deferred.empty()?source->revision:source->deferred_revision) ||
					!m_impl->plan_aliases(source).empty()) return;
				if (source->deferred.empty())
				{
					if (!source->initialized || (!source->lease && !source->spill.buffer)) return;
				}
				else if (!complete_deferred_input(source) && !current_deferred_background(*source)) return;
				for (const auto& output:uses) if (output.write &&
					!(source==output.surface && endpoints.version_target) &&
					memory_storage_overlap(*m_impl->memory,source->storage,output.surface->storage)) return;
			}
			frozen_inputs=true;
		});
	std::vector<std::vector<scaled_image_input_view>> input_views;
	draw_impl(uses,[&](context& ctx,auto images){encode(ctx,images,input_views);},true,nullptr,nullptr,
		std::move(publication_check),endpoints.cold_outputs,true,frozen_inputs,endpoints.version_target,{},nullptr,before_loads,input_windows,&input_views,defer_completion,std::move(after_commit));
}
void surface_cache::draw_impl(std::span<const cached_surface_use> uses, const std::function<void(context&, std::span<const id<MTLTexture>>)>& encode, bool feedback,
							  const surface_allocation_description* transient, surface_lease* reservation,std::function<void()> publication_check,std::span<const cached_surface_ref> full_overwrites,bool scaled_inputs,bool frozen_scaled_inputs,bool version_scaled_donor,
							  draw_publication publish_draw, const submission* predecessor,
							  const std::function<void(context&)>& before_loads,
							  std::span<const std::vector<clear_rectangle>> input_windows,
							  std::vector<std::vector<scaled_image_input_view>>* input_views, bool defer_scaled_completion, std::function<void()> after_scaled_commit)
{
	const std::vector<cached_surface_ref> overwrites(full_overwrites.begin(), full_overwrites.end());
	const auto is_overwrite = [&](const auto& surface) { return std::find(overwrites.begin(), overwrites.end(), surface) != overwrites.end(); };
	if(publication_check)m_impl->cpu(publication_check);
	// Ordinary RSX draws retain their sixteen fragment/four vertex/five target
	// bound. A scaled transfer has separate bounded reader/writer vectors;
	// its disjoint writers are encoded sequentially on the GPU.
	if (uses.empty() || uses.size() > (scaled_inputs ? scaled_image_max_readers + scaled_image_max_writers : 25u) || !encode)
		throw std::invalid_argument("Native cache draw exceeds its use/write limits or lacks an encoder");
	const auto write_count = static_cast<std::size_t>(std::ranges::count(uses, true, &cached_surface_use::write));
	if (write_count > (scaled_inputs ? scaled_image_max_writers : 5u))
		throw std::invalid_argument("Native cache draw exceeds its use/write limits or lacks an encoder");
	if (input_views)
	{
		if (!scaled_inputs || input_windows.size() != uses.size() - write_count)
			throw std::invalid_argument("Scaled input windows lost their reader correspondence");
		input_views->assign(input_windows.size(), {});
	}
	if (scaled_inputs && (write_count == 0 || uses.size() - write_count > scaled_image_max_readers ||
		std::any_of(uses.begin(), uses.end() - write_count, [](const auto& use) { return use.write; }) ||
		std::any_of(uses.end() - write_count, uses.end(), [](const auto& use) { return !use.write; })))
		throw std::invalid_argument("Scaled input batches require read-only sources followed by disjoint outputs");
	for (const auto& use : uses) m_impl->check(use.surface);
	const bool allow_read_alias = feedback && !frozen_scaled_inputs;
	if (uses.size() > 1 && write_count > (allow_read_alias ? 1 : 0))
	{
		// All pairs must see one stable mapping. The provider can suspend guest
		// writers here, so keep GPU work and resource preparation outside.
		const bool overlap = m_impl->memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
		{
			for (std::size_t i = 1; i < uses.size(); ++i)
				for (std::size_t j = 0; j < i; ++j)
					if ((allow_read_alias ? uses[i].write && uses[j].write : uses[i].write || uses[j].write) &&
						!(frozen_scaled_inputs && version_scaled_donor && uses[i].surface == uses[j].surface && uses[i].write != uses[j].write) &&
						active_storage_overlap(direct, lookup, uses[i].surface->storage, uses[j].surface->storage))
						return true;
			return false;
		});
		if (overlap) throw std::invalid_argument("Native cache draw write targets overlap guest storage");
	}
	const bool conditional = bool(publish_draw);
	const bool require_native_version = version_scaled_donor || conditional || (scaled_inputs && defer_scaled_completion);
	const bool batchable = !publication_check && !conditional;
	// A raw fallback may materialize a reader transitively. Settle draws
	// that also consume deferred inputs before capturing any target loads.
	const bool keep_deferred_readers = !feedback && !scaled_inputs &&
		(uses.size() == 1 || std::ranges::none_of(uses, [](const auto& use) { return !use.surface->deferred.empty(); }));
	const bool initialize_in_draw = (conditional || (batchable && m_impl->batch_capacity)) && !feedback && !scaled_inputs;
	const bool prefetch_targets = initialize_in_draw && !conditional && std::size_t(write_count) == uses.size();
	std::vector<std::pair<cached_surface_ref, prefetched_target_data>> prefetched_targets;
	if (prefetch_targets && m_impl->pending && m_impl->pending->draw_batch)
	{
		// Keep the early snapshots bounded across all targets of the draw.
		std::size_t remaining = std::min(m_impl->snapshot_limit, target_prefetch_limit);
		for (const auto& use : uses)
		{
			const auto& target = use.surface;
			if (auto prefetched = m_impl->prefetch_guest_target(target, *this, remaining))
			{
				prefetched_targets.emplace_back(target, std::move(*prefetched));
				remaining -= target->storage.size;
			}
		}
	}
	std::optional<submission> early_draw;
	std::vector<draw_target_load> target_loads;
	std::vector<surface_validation> load_revisions;
	if (prefetch_targets && !transient && !predecessor && m_impl->pending && m_impl->ctx.can_begin_after(m_impl->pending->work))
	{
		for (const auto& use : uses)
		{
			const auto& target = use.surface;
			const auto captured = std::ranges::find_if(prefetched_targets, [&](const auto& item) { return item.first == target; });
			auto planned = m_impl->plan_independent_write(target, *this, captured == prefetched_targets.end() ? nullptr : &captured->second);
			if (!planned) break;
			if (planned->first.has_input()) target_loads.push_back(std::move(planned->first));
			load_revisions.push_back(std::move(planned->second));
		}
		if (m_impl->pending && load_revisions.size() == uses.size()) early_draw = m_impl->pending->work;
		else { target_loads.clear(); load_revisions.clear(); }
	}
	if (!early_draw) m_impl->synchronize(m_impl->timeout, false);
	for (const auto& surface : overwrites)
		if (!scaled_inputs || !surface || surface->dirty || !surface->deferred.empty() ||
			std::none_of(uses.begin(), uses.end(), [&](const auto& use) { return use.write && use.surface == surface; }))
			throw std::logic_error("Native full-overwrite requires clean selected output authorities");
	std::function<bool()> permit_preparation;
	if (conditional)
		permit_preparation = [&] { return publish_draw(m_impl->timeout); };
	if (std::any_of(uses.begin(), uses.end(), [](const auto& use) { return !use.write; }))
		m_impl->synchronize(m_impl->timeout);
	// Versioned draws leave the old allocation intact. Deferred readers
	// already own that lease and only need materialization before an in-place write.
	for (const auto& use : uses)
		if (use.write && !early_draw && !require_native_version && !keep_deferred_readers)
			m_impl->preserve_readers(use.surface->lease, use.surface);
	std::vector<cached_surface_ref> pins;
	for (const auto& use : uses)
		pins.push_back(use.surface);
	auto previous_pins = m_impl->pinned;
	m_impl->pinned.insert(m_impl->pinned.end(), pins.begin(), pins.end());
	std::vector<surface_lease> snapshots(uses.size());
	struct frozen_reader
	{
		struct window
		{
			surface_lease allocation;
			clear_rectangle pixels;
			std::vector<deferred_snapshot_piece> pieces = {};
			std::vector<surface_lease> typeless_inputs = {};
			bool spill = false;
			std::vector<typeless_source_window> typeless_windows = {};
		};
		std::vector<window> windows;
		std::optional<clear_rectangle> spill_box;
		std::shared_ptr<sample_transfer> samples;
		std::shared_ptr<surface_typeless_transfer> typeless;
		std::shared_ptr<texture_blitter> blits;
		surface_lease allocation;
		surface_lease background;
		raw_surface_snapshot spill;
		std::vector<deferred_copy_section> sections;
		bool clear_background = false;
	};
	auto frozen=frozen_scaled_inputs?std::make_shared<std::vector<frozen_reader>>(uses.size()):std::shared_ptr<std::vector<frozen_reader>>{};
	try
	{
		// Capture readers before writer preparation can invalidate/reinterpret
		// their guest aliases. All required allocations stay pinned until copied.
		for (std::size_t i=0;i<uses.size();++i)
			if (!uses[i].write)
			{
				const auto& surface=uses[i].surface;
				if (!frozen_scaled_inputs)
				{
					if (!m_impl->prepare(surface, surface_initialization::guest, permit_preparation))
					{
						m_impl->pinned = std::move(previous_pins);
						return;
					}
				}
				else
				{
					const auto& revision=surface->deferred.empty()?surface->revision:surface->deferred_revision;
					if(surface->faulted||m_impl->current(surface)!=revision)throw surface_conflict_error(surface->layout.range);
					// Frozen reads bypass prepare(), but still count as cache use.
					surface->last_use = ++m_impl->clock;
					auto& reader=(*frozen)[i];
					// Selected spill rectangles already contain expanded sample bits.
					// Check out only those single-sample images, not the full source
					// or an MSAA import backing. Exact deferred projections can retain
					// their donor leases directly without any destination image.
					if (input_views && !input_windows[i].empty())
					{
						const auto& a = surface->description.allocation;
						const auto size = a.samples == 1 ? std::array{a.width,a.height} : sample_expanded_size(a.width,a.height,a.samples);
						for (const auto r : input_windows[i])
							if (!r.width || !r.height || r.x > size[0] || r.y > size[1] || r.width > size[0]-r.x || r.height > size[1]-r.y)
								throw std::out_of_range("Frozen scaled input window exceeds native sample raster");
						if (surface->deferred.empty() && surface->initialized && surface->spill.buffer)
						{
							reader.spill = surface->spill;
							const auto enclosing_box=[&]
							{
								if (input_windows[i].size()<2) throw;
								reader.windows.clear();
								auto left=size[0],top=size[1];std::size_t right=0,bottom=0;
								for (const auto r:input_windows[i]){left=std::min(left,r.x);top=std::min(top,r.y);right=std::max(right,r.x+r.width);bottom=std::max(bottom,r.y+r.height);}
								reader.spill_box=clear_rectangle{left,top,right-left,bottom-top};
								reader.allocation=m_impl->pool.acquire({right-left,bottom-top,1,a.format,false});
								for (const auto r:input_windows[i])reader.windows.push_back({reader.allocation,{r.x-left,r.y-top,r.width,r.height}});
							};
							try
							{
								for (const auto r : input_windows[i])
									reader.windows.push_back({m_impl->pool.acquire({r.width,r.height,1,a.format,false}),r,{},{},true});
							}
							catch (const surface_budget_error&) { enclosing_box(); }
							catch (const heap_budget_error&) { enclosing_box(); }
						}
						else if (!surface->deferred.empty())
						{
							for (const auto r : input_windows[i])
							{
								const auto donor = project_deferred_snapshot(*surface,r);
								if (donor) { reader.windows.push_back({donor->allocation,donor->rectangle}); continue; }
								auto pieces = compose_deferred_snapshot(*surface,r,false);
								if (pieces.empty()) { reader.windows.clear(); break; }
								frozen_reader::window window{m_impl->pool.acquire({r.width,r.height,1,a.format,false}),r,std::move(pieces)};
								window.typeless_inputs.resize(window.pieces.size());
								window.typeless_windows.resize(window.pieces.size());
								for (std::size_t j=0;j<window.pieces.size();++j)
								{
									const auto& piece=window.pieces[j];
									if (piece.typeless && piece.allocation->description().samples>1)
									{
										const auto& p=piece.projection;
										const nearest_projection projection{{static_cast<std::uint32_t>(p.copy_extent[0]),static_cast<std::uint32_t>(p.copy_extent[1]),1},
											{static_cast<std::uint32_t>(p.copy_offset[0]),static_cast<std::uint32_t>(p.copy_offset[1]),0}};
										const auto selected=plan_typeless_source_window(piece.allocation->description().format,a.format,p.source_extent[0],p.source_extent[1],piece.destination.width,piece.destination.height,{p.flip[0],p.flip[1],false},projection);
										window.typeless_windows[j]=selected;
										window.typeless_inputs[j]=m_impl->pool.acquire({selected.extent[0],selected.extent[1],1,piece.allocation->description().format,false});
									}
								}
								reader.spill=surface->spill;reader.samples=m_impl->samples;
								if (!m_impl->frozen_blits) m_impl->frozen_blits=std::make_shared<texture_blitter>(m_impl->ctx);
								reader.blits=m_impl->frozen_blits;
								if (std::ranges::any_of(window.pieces,[](const auto& p){return p.typeless;}))
								{
									if (!m_impl->typeless) m_impl->typeless=std::make_shared<surface_typeless_transfer>(m_impl->ctx);
									reader.typeless=m_impl->typeless;
								}
								reader.windows.push_back(std::move(window));
							}
						}
						if (!reader.windows.empty()) continue;
					}
					if(!surface->deferred.empty())
					{
						if (!complete_deferred_input(surface))
						{
							if (current_deferred_background(*surface))
							{
								reader.spill = surface->spill;
								if (!reader.spill.buffer) reader.background = surface->lease;
							}
							else
							{
								if (surface->dirty) throw std::logic_error("Frozen scaled reader lost its dirty background");
								reader.clear_background = true;
							}
						}
						reader.sections=surface->deferred;
						reader.allocation=m_impl->allocate_transient(surface->description.allocation,surface);
					}
					else
					{
						if(!surface->initialized||(!surface->lease&&!surface->spill.buffer))throw std::logic_error("Frozen scaled reader lost native contents");
						reader.spill=surface->spill;
						reader.allocation=reader.spill.buffer?m_impl->allocate_transient(surface->description.allocation,surface):surface->lease;
					}
				}
			}
		std::vector<std::size_t> copies;
		if (feedback&&!frozen_scaled_inputs)
			for (std::size_t i = 0; i < uses.size(); ++i)
			{
				if (uses[i].write || !std::any_of(uses.begin(), uses.end(), [&](const auto& other)
												  { return other.write && memory_storage_overlap(*m_impl->memory, uses[i].surface->storage, other.surface->storage); }))
					continue;
				for (std::size_t j = 0; j < i; ++j)
					if (!uses[j].write && uses[j].surface == uses[i].surface)
					{
						snapshots[i] = snapshots[j];
						break;
					}
				if (snapshots[i])
					continue;
				auto description = uses[i].surface->description.allocation;
				description.expanded = false; // Capture native samples, without a second guest transfer backing.
				snapshots[i] = m_impl->allocate_transient(description);
				copies.push_back(i);
			}
		if (!copies.empty())
		{
			std::vector<surface_validation> captured;
			for (const auto i : copies) captured.push_back({uses[i].surface, uses[i].surface->revision});
			m_impl->execute([&]
							{
								for (const auto i : copies)
								{
									texture_region whole;
									const auto& description = uses[i].surface->description.allocation;
									whole.size = MTLSizeMake(description.width, description.height, 1);
									copy_texture(m_impl->ctx, uses[i].surface->lease->render_texture(m_impl->ctx), whole, snapshots[i]->render_texture(m_impl->ctx), whole);
								} },
							[] {}, {}, std::move(captured));
		}
		for(const auto& use:uses)if(use.write)
		{
			const auto& target = use.surface;
			if (early_draw)
			{
				target->last_use = ++m_impl->clock;
				continue;
			}
			if (initialize_in_draw && !conditional && m_impl->recorded_draw_target(target))
			{
				m_impl->batch_surface_boundary(target, true);
				if (const auto* previous = m_impl->recorded_draw_target(target))
				{
					target->last_use = ++m_impl->clock;
					load_revisions.push_back(*previous);
					continue;
				}
			}
			if (initialize_in_draw && m_impl->needs_target_load(target))
			{
				if (conditional)
				{
					try
					{
						if (!publish_draw(std::chrono::milliseconds::zero()))
						{
							m_impl->pinned = std::move(previous_pins);
							return;
						}
					}
					catch (const submission_wait_timeout&) {} // The GPU draw can consume the report later.
					m_impl->synchronize(m_impl->timeout);
				}
				else
				{
					m_impl->batch_surface_boundary(target, true);
					m_impl->synchronize(m_impl->timeout, false);
				}
				if (m_impl->needs_target_load(target))
				{
					target->last_use = ++m_impl->clock;
					std::optional<staged_target_load> staged;
					const auto early = std::ranges::find_if(prefetched_targets, [&](const auto& item) { return item.first == target; });
					auto* snapshot = early == prefetched_targets.end() ? nullptr : &early->second;
					try { staged = m_impl->stage_target_load(target, *this, permit_preparation, snapshot); }
					catch (const submission_wait_timeout&) { throw; }
					catch (...)
					{
						// An unresolved predicate must not turn an unused target's
						// invalid backing into a draw failure.
						if (!conditional || permit_preparation()) throw;
						m_impl->pinned = std::move(previous_pins);
						return;
					}
					if (staged)
					{
						target_loads.push_back(std::move(staged->first));
						load_revisions.push_back(std::move(staged->second));
						continue;
					}
				}
			}
			if (is_overwrite(target))
			{
				m_impl->flush_aliases(target);
				target_loads.push_back(draw_target_load{target});
				load_revisions.push_back({target, m_impl->current(target)});
				continue;
			}
			if (version_scaled_donor)
			{
				// The old native version and its deferred readers stay private.
				// Stage only native loads here; never capture a guest background
				// or materialize a source before the scale transaction is admitted.
				const auto& revision = target->deferred.empty() ? target->revision : target->deferred_revision;
				if (target->faulted || m_impl->current(target) != revision)
					throw surface_conflict_error(use.surface->layout.range);
				const auto aliases = m_impl->plan_aliases(use.surface);
				if (std::any_of(aliases.begin(), aliases.end(), [](const auto& alias) { return alias->dirty || !alias->deferred.empty(); }))
					throw surface_conflict_error(use.surface->layout.range);
				if (!target->deferred.empty() || target->spill.buffer)
				{
					draw_target_load load{target};
					if (!target->deferred.empty())
					{
						if (!impl::plan_deferred_target_load(load))
							throw surface_conflict_error(target->layout.range);
					}
					else load.spill = target->spill;
					target_loads.push_back(std::move(load));
					load_revisions.push_back({target, revision});
				}
				else if (!target->initialized || !target->lease)
					throw surface_conflict_error(target->layout.range);
				use.surface->last_use = ++m_impl->clock;
				continue;
			}
			else
			{
				target_reload reload{*this};
				if (!m_impl->prepare(target, surface_initialization::guest, permit_preparation, initialize_in_draw ? &reload : nullptr))
				{
					m_impl->pinned = std::move(previous_pins);
					return;
				}
				if (reload.staged)
				{
					target_loads.push_back(std::move(reload.staged->first));
					load_revisions.push_back(std::move(reload.staged->second));
				}
			}
		}
		if (transient)
			*reservation = m_impl->allocate_transient(*transient);
	}
	catch (...)
	{
		m_impl->pinned = std::move(previous_pins);
		throw;
	}
	m_impl->pinned = std::move(previous_pins);
	std::vector<cached_surface_ref> written;
	std::vector<cached_surface_ref> aliases;
	for (const auto& use : uses)
		if (use.write)
		{
			written.push_back(use.surface);
			const auto affected = m_impl->plan_aliases(use.surface);
			aliases.insert(aliases.end(), affected.begin(), affected.end());
		}
	const auto native_epoch = written.empty() ? 0 : m_impl->reserve_native_epoch();
	const auto encode_loads = [&]
	{
		if (before_loads) before_loads(m_impl->ctx);
		for (const auto& load : target_loads) m_impl->encode_target_load(load);
	};
	const auto encode_draw = [&]
	{
		if (frozen_scaled_inputs) m_impl->ctx.keep_alive(frozen);
		std::vector<id<MTLTexture>> textures;
		for (std::size_t i = 0; i < uses.size(); ++i)
		{
			if (frozen_scaled_inputs && !uses[i].write)
			{
				const auto& reader = (*frozen)[i];
				if (!reader.windows.empty())
				{
					if (reader.spill_box)
					{
						const auto& a=uses[i].surface->description.allocation;const auto box=*reader.spill_box;
						const auto height=a.samples==1?a.height:sample_expanded_size(a.width,a.height,a.samples)[1];
						texture_region target;target.size=MTLSizeMake(box.width,box.height,1);
						copy_spill_rectangle(m_impl->ctx,reader.spill,height,box,reader.allocation->render_texture(m_impl->ctx),target);
					}
					for (const auto& window : reader.windows)
					{
						const auto image = window.allocation->render_texture(m_impl->ctx);
						auto pixels = window.pixels;
						if (!window.pieces.empty())
						{
							const auto& a=uses[i].surface->description.allocation;
							encode_surface_background(m_impl->ctx,a.format,image);
							for (std::size_t j=0;j<window.pieces.size();++j)
							{
								const auto& piece=window.pieces[j];
								// Missing bytes already have separate RAM authority in
								// the packed input. Match the old dummy-background path
								// without capturing any guest pixels in this encoder.
								if (piece.guest) continue;
								texture_region target;target.origin=MTLOriginMake(piece.destination.x,piece.destination.y,0);
								target.size=MTLSizeMake(piece.destination.width,piece.destination.height,1);
								const auto& p=piece.projection;
								if (!piece.allocation)
								{
									if (!reader.spill.buffer) throw std::logic_error("Composed frozen background lost its native spill");
									const auto height=a.samples==1?a.height:sample_expanded_size(a.width,a.height,a.samples)[1];
									copy_spill_rectangle(m_impl->ctx,reader.spill,height,{p.source_origin[0],p.source_origin[1],piece.destination.width,piece.destination.height},image,target,piece.plane);
									continue;
								}
								auto source=piece.allocation->render_texture(m_impl->ctx);
								texture_region input;input.origin=MTLOriginMake(p.source_origin[0],p.source_origin[1],0);
								input.size=MTLSizeMake(p.source_extent[0],p.source_extent[1],1);
								nearest_projection projection{{static_cast<std::uint32_t>(p.copy_extent[0]),static_cast<std::uint32_t>(p.copy_extent[1]),1},
									{static_cast<std::uint32_t>(p.copy_offset[0]),static_cast<std::uint32_t>(p.copy_offset[1]),0}};
								if (piece.typeless)
								{
									if (source.sampleCount>1)
									{
										const auto expanded=window.typeless_inputs[j]->render_texture(m_impl->ctx);
										const auto& selected=window.typeless_windows[j];
										reader.samples->expand_region(m_impl->ctx,source,expanded,{p.source_origin[0]+selected.origin[0],p.source_origin[1]+selected.origin[1]});
										source=expanded;input.origin=MTLOriginMake(0,0,0);input.size=MTLSizeMake(selected.extent[0],selected.extent[1],1);projection=selected.projection;
									}
									reader.typeless->nearest(m_impl->ctx,piece.allocation->description().format,source,input,a.format,image,target,{p.flip[0],p.flip[1],false},projection,piece.plane);
								}
								else if (source.sampleCount>1)
									reader.samples->project_into(m_impl->ctx,source,image,p,target,true,piece.plane);
								else
									reader.blits->nearest(m_impl->ctx,source,input,image,target,piece.plane,{p.flip[0],p.flip[1],false},projection);
							}
							pixels.x=pixels.y=0;
						}
						else if (window.spill)
						{
							const auto& a = uses[i].surface->description.allocation;
							const auto height = a.samples == 1 ? a.height : sample_expanded_size(a.width,a.height,a.samples)[1];
							texture_region target; target.size = MTLSizeMake(pixels.width,pixels.height,1);
							copy_spill_rectangle(m_impl->ctx,reader.spill,height,pixels,image,target);
							pixels.x = pixels.y = 0;
						}
						(*input_views)[i].push_back({image,pixels});
					}
					textures.push_back((*input_views)[i].front().image);
					continue;
				}
				const auto image = reader.allocation->render_texture(m_impl->ctx);
				if (reader.spill.buffer)
				{
					m_impl->restore_spill(uses[i].surface, reader.spill, reader.allocation);
				}
				else if (reader.background)
				{
					const auto& allocation = uses[i].surface->description.allocation;
					texture_region whole;
					whole.size = MTLSizeMake(allocation.width, allocation.height, 1);
					copy_texture(m_impl->ctx, reader.background->render_texture(m_impl->ctx), whole, image, whole);
				}
				else if (reader.clear_background)
					encode_surface_background(m_impl->ctx, uses[i].surface->description.allocation.format, image);
				m_impl->encode_deferred_copies(uses[i].surface, image, reader.sections);
				textures.push_back(image);
			}
			else if (snapshots[i])
			{
				textures.push_back(snapshots[i]->render_texture(m_impl->ctx));
			}
			else
				textures.push_back(uses[i].surface->lease->render_texture(m_impl->ctx));
		}
		encode(m_impl->ctx, textures);
	};
	std::function<void()> commit = [this, written, aliases, native_epoch, check = std::move(publication_check),
		overwrites, target_loads, load_revisions, after_scaled_commit]
	{
		if (check) check();
		for (const auto& surface : written)
		{
			if (std::find(overwrites.begin(), overwrites.end(), surface) != overwrites.end())
			{
				const auto init = std::ranges::find(load_revisions, surface, &surface_validation::surface);
				surface->revision = init->revisions; surface->initialized = true;
				surface->faulted = false; surface->spill = {};
			}
			const auto load = std::ranges::find(target_loads, surface, &draw_target_load::surface);
			if (load != target_loads.end())
			{
				if (load->guest || !load->sections.empty())
				{
					const auto init = std::ranges::find(load_revisions, surface, &surface_validation::surface);
					surface->revision = init->revisions;
					surface->initialized = true;
					surface->generation += load->generations();
				}
				surface->spill = {};
				if (!load->sections.empty()) surface->deferred.clear();
			}
			surface->dirty = true;
			surface->native_write_epoch = native_epoch;
			++surface->generation;
		}
		m_impl->invalidate_aliases(aliases);
		if (after_scaled_commit) after_scaled_commit();
	};
	auto validated = [&]
	{
		std::vector<surface_validation> values;
		for (const auto& use : uses)
		{
			const auto load = std::ranges::find(load_revisions, use.surface, &surface_validation::surface);
			if (load != load_revisions.end()) values.push_back(*load);
			else values.push_back({use.surface,
				(frozen_scaled_inputs&&!use.write&&!use.surface->deferred.empty()?use.surface->deferred_revision:use.surface->revision)});
		}
		return values;
	}();
	if (early_draw)
	{
		if (aliases.empty() && m_impl->record_independent_targets(encode_loads, encode_draw, commit, written, validated, *early_draw, target_loads, true)) return;
		m_impl->synchronize(m_impl->timeout, false);
		commit = {};
		target_loads.clear();
		prefetched_targets.clear();
		return draw_impl(uses, encode, feedback, transient, reservation, {}, overwrites, scaled_inputs, frozen_scaled_inputs,
			version_scaled_donor, std::move(publish_draw), predecessor, before_loads, input_windows, input_views, defer_scaled_completion,std::move(after_scaled_commit));
	}
	// Cold overwrite admission can reclaim pool storage. Keep every selected
	// input and output pinned until its private target allocations are admitted.
	surface_list_guard overwrite_pins(m_impl->pinned, overwrites.empty() ? std::span<const cached_surface_ref>{} : std::span<const cached_surface_ref>{pins});
	m_impl->execute([&] { encode_loads(); encode_draw(); }, std::move(commit), written, std::move(validated), aliases,
		false, !written.empty(), overwrites.size() != written.size(), batchable, require_native_version, std::move(publish_draw), predecessor,
		target_loads, keep_deferred_readers, scaled_inputs && defer_scaled_completion);
}
void surface_cache::copy(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
						 std::size_t x, std::size_t y, texture_plane plane)
{
	blit_nearest(source, rectangle, destination, {x, y, rectangle.width, rectangle.height}, plane);
}
void surface_cache::blit_nearest(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
								 clear_rectangle target, texture_plane plane, std::array<bool, 3> flip)
{
	transfer(source, rectangle, destination, target, plane, flip, false);
}
void surface_cache::blit_typeless(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
								  clear_rectangle target, std::array<bool, 3> flip)
{
	transfer(source, rectangle, destination, target, texture_plane::all, flip, true);
}
void surface_cache::transfer(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
							 clear_rectangle target, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	m_impl->check(source);
	m_impl->check(destination);
	m_impl->synchronize(m_impl->timeout);
	const auto& a = source->description.allocation;
	const auto& b = destination->description.allocation;
	if (!rectangle.width || !rectangle.height || !target.width || !target.height)
		return;
	validate_surface_blit(a, rectangle, b, target, plane, flip, typeless);
	m_impl->preserve_readers(destination->lease, destination);
	std::vector<cached_surface_ref> pins{source, destination};
	auto previous_pins = m_impl->pinned;
	m_impl->pinned.insert(m_impl->pinned.end(), pins.begin(), pins.end());
	const bool full = plane == texture_plane::all && target.x == 0 && target.y == 0 && target.width == b.width && target.height == b.height && source != destination;
	std::vector<std::uint64_t> revision;
	try
	{
		m_impl->prepare(source, surface_initialization::guest);
		if (full)
		{
			m_impl->flush_aliases(destination);
			revision = m_impl->current(destination);
			m_impl->allocate(destination);
		}
		else
		{
			m_impl->prepare(destination, surface_initialization::guest);
			revision = destination->revision;
		}
	}
	catch (...)
	{
		m_impl->pinned = std::move(previous_pins);
		throw;
	}
	m_impl->pinned = std::move(previous_pins);
	source->last_use = destination->last_use = ++m_impl->clock;
	const auto aliases = m_impl->plan_aliases(destination);
	const auto native_epoch = m_impl->reserve_native_epoch();
	m_impl->execute([&]
					{
						texture_region sr, dr;
						sr.origin = MTLOriginMake(rectangle.x, rectangle.y, 0);
						dr.origin = MTLOriginMake(target.x, target.y, 0);
						sr.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
						dr.size = MTLSizeMake(target.width, target.height, 1);
						m_impl->sample_blits.nearest(m_impl->ctx, a.format, source->lease->render_texture(m_impl->ctx), sr,
													 b.format, destination->lease->render_texture(m_impl->ctx), dr, plane, flip, typeless); },
					[this, destination, revision, aliases, native_epoch]
					{
						destination->revision = revision;
						destination->initialized = true;
						destination->dirty = true;
						destination->native_write_epoch = native_epoch;
						destination->faulted = false;
						destination->spill = {};
						destination->deferred.clear();
						++destination->generation;
						m_impl->invalidate_aliases(aliases);
					},
					{destination}, {{source, source->revision}, {destination, revision}}, aliases, false, true);
}
void surface_cache::defer_copy(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
							   std::size_t x, std::size_t y, texture_plane plane)
{
	defer_blit_nearest(source, rectangle, destination, {x, y, rectangle.width, rectangle.height}, plane);
}
void surface_cache::defer_blit_nearest(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
									   clear_rectangle target, texture_plane plane, std::array<bool, 3> flip)
{
	defer_transfer(source, rectangle, destination, target, plane, flip, false);
}
void surface_cache::defer_blit_typeless(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
										clear_rectangle target, std::array<bool, 3> flip)
{
	defer_transfer(source, rectangle, destination, target, texture_plane::all, flip, true);
}
void surface_cache::defer_transfer(const cached_surface_ref& source, clear_rectangle rectangle, const cached_surface_ref& destination,
								   clear_rectangle target, texture_plane plane, std::array<bool, 3> flip, bool typeless)
{
	m_impl->check(source);
	m_impl->check(destination);
	m_impl->synchronize(m_impl->timeout);
	if (!rectangle.width || !rectangle.height || !target.width || !target.height)
		return;
	validate_surface_blit(source->description.allocation, rectangle, destination->description.allocation, target, plane, flip, typeless);
	m_impl->prepare(source, surface_initialization::guest);
	auto revision = m_impl->current(destination);
	if ((destination->dirty && revision != destination->revision) ||
		(!destination->deferred.empty() && revision != destination->deferred_revision))
		throw surface_conflict_error(destination->storage.parts.front().coverage.range);
	const surface_validation validation{destination, revision};
	const auto retained = m_impl->retain_backings({&validation, 1});
	auto plans = destination->deferred;
	const auto write_epoch=m_impl->reserve_native_epoch();
	plans.push_back({source->lease, source->generation, write_epoch, rectangle, target, plane, flip, typeless});
	// A new deferred owner can cover a former VM hole in an assembled image.
	// Invalidate its source-set identity now,not only when materialized later.
	m_impl->texture_topology_epoch = write_epoch;
	destination->deferred = std::move(plans);
	destination->deferred_revision = std::move(revision);
	destination->backing = retained.front().owner;
	destination->physical = retained.front().physical;
}
void surface_cache::cancel_deferred_copies(const cached_surface_ref& surface)
{
	m_impl->check(surface);
	m_impl->synchronize(m_impl->timeout);
	if(!surface->deferred.empty())m_impl->texture_topology_epoch = m_impl->reserve_native_epoch();
	surface->deferred.clear();
	if (!surface->dirty) surface->reset_backing();
	(void)m_impl->pool.trim();
}
void surface_cache::flush(const cached_surface_ref& surface)
{
	m_impl->flush(surface);
}
void surface_cache::flush_physical(const cached_surface_ref& surface)
{
	m_impl->flush_physical(surface);
}
std::size_t surface_cache::prepare_physical_fault(const physical_identity& identity,
	std::span<const std::shared_ptr<void>> registrations, bool writing)
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native cache fault preparation cannot wait inside VM exclusion");
	m_impl->synchronize(m_impl->timeout);
	std::vector<cached_surface_ref> selected, outputs;
	m_impl->cpu([&]
	{
		for (const auto& surface : m_impl->surfaces)
		{
			bool matches = false;
			if (surface->dirty || !surface->deferred.empty())
			{
				if (!surface->physical || surface->physical->size() != surface->storage.parts.size())
					throw std::runtime_error("Native fault discovery cannot omit untyped dirty output");
				for (const auto& ticket : *surface->physical)
					for (const auto& part : ticket->parts())
						if (identity.overlaps(*part)) matches = true;
			}
			else
				for (const auto& part : surface->storage.parts)
					if (memory_physical_fault_overlap(*m_impl->memory, identity, registrations, part.coverage)) { matches = true; break; }
			if (!matches) continue;
			selected.push_back(surface);
			if (surface->dirty || !surface->deferred.empty()) outputs.push_back(surface);
		}
		// Preflight every selected authority before the first GPU writeback.
		// Later host writes can still make a CAS fail;already valid publications
		// remain committed and all remaining owners/protection survive for retry.
		for (const auto& surface : outputs)
		{
			if (surface->faulted) throw std::runtime_error("Native fault discovery found faulted dirty output");
			const surface_validation validation{surface, {}, true};
			m_impl->validate_in_epoch({&validation, 1});
			(void)m_impl->physical_aliases(surface);
		}
	});
	for (const auto& surface : outputs) m_impl->flush_physical(surface, !writing);
	if (writing)
		for (const auto& surface : selected)
		{
			surface->initialized = false;
			surface->spill = {};
			surface->native_write_epoch = 0;
			surface->reset_backing();
		}
	return outputs.size();
}
std::size_t surface_cache::invalidate_mapping(guest_surface_range range, surface_mapping_event cause)
{
	validate_guest_surface_range(range);
	if (cause != surface_mapping_event::read && cause != surface_mapping_event::unmap)
		throw std::invalid_argument("Invalid native surface mapping event");
	if (guest_memory_epoch::active()) throw std::logic_error("Native mapping recovery cannot run inside VM exclusion");
	if (!range.size) return 0;
	m_impl->synchronize(m_impl->timeout);
	const auto coverage = guest_byte_coverage(range);
	std::vector<cached_surface_ref> selected;
	m_impl->cpu([&]
	{
		for (const auto& surface : m_impl->surfaces)
		{
			if (!surface->dirty && surface->deferred.empty()) continue;
			bool matches = std::any_of(surface->storage.parts.begin(), surface->storage.parts.end(),
				[&](const auto& part) { return overlaps(coverage, part.coverage); });
			if (!matches)
			{
				if (!surface->physical) throw std::runtime_error("Native mapping selection lacks retained physical tickets");
				for (const auto& ticket : *surface->physical)
				{
					const auto alias = m_impl->memory->physical_surface_overlaps_current(*ticket, range);
					if (!alias) throw std::runtime_error("Native mapping selection lacks current allocation metadata");
					if (*alias) { matches = true; break; }
				}
			}
			if (matches) selected.push_back(surface);
		}
	});
	for (const auto& surface : selected)
		m_impl->flush_physical(surface, cause == surface_mapping_event::read);
	if (cause == surface_mapping_event::unmap)
	{
		// Clean mirrors need no old-VA lookup. Conservatively retire all of them
		// only after every selected output was reconciled successfully. Resource
		// release stays outside provider exclusion;dirty/deferred owners survive.
		for (const auto& surface : m_impl->surfaces)
			if (!surface->dirty && surface->deferred.empty())
			{
				surface->initialized = false;
				surface->spill = {};
				surface->native_write_epoch = 0;
				surface->reset_backing();
			}
	}
	const auto count = selected.size();
	selected.clear();
	if (cause == surface_mapping_event::unmap) m_impl->retire_to(0, 0);
	return count;
}
void surface_cache::spill(const cached_surface_ref& surface)
{
	m_impl->spill_surface(surface);
}
void surface_cache::recover(const cached_surface_ref& surface)
{
	m_impl->check(surface);
	m_impl->synchronize(m_impl->timeout);
	if (surface->faulted && surface->dirty && !surface->spill.buffer)
		throw std::runtime_error("Faulted dirty surface has no recoverable backing");
	m_impl->ctx.recover_completed_errors();
	if (surface->faulted)
	{
		surface->faulted = false;
		surface->initialized = false;
	}
}
void surface_cache::erase(const cached_surface_ref& surface)
{
	m_impl->outside_epoch();
	m_impl->flush(surface);
	surface->lease.reset();
	surface->spill = {};
	surface->reset_backing();
	m_impl->remove(surface);
	(void)m_impl->pool.trim();
}
void surface_cache::prepare_texture_boundary(bool gpu_work)
{
	if (m_impl->owner_thread != std::this_thread::get_id()) throw std::logic_error("Native texture boundary belongs to another render thread");
	m_impl->outside_epoch();
	if (m_impl->recording_batch)
	{
		const auto& recording = m_impl->recording_batch->recording;
		if (!recording || !recording->same_batch(m_impl->ctx.recording_submission()))
			throw std::logic_error("Native surface batch no longer owns the current recording");
		if (!gpu_work) return;
		m_impl->flush_draw_batch();
	}
	m_impl->ctx.validate_begin();
}
void surface_cache::validate_texture_adapter(context& ctx, const guest_surface_memory& memory) const
{
	if (&ctx != &m_impl->ctx || &memory != m_impl->memory.get())
		throw std::invalid_argument("Native texture/surface caches require the same context and VM adapter");
}
bool surface_cache::prepare_texture_read_boundary(std::span<const guest_surface_layout> coverage, bool revision_tracked)
{
	prepare_texture_boundary(false);
	if (coverage.empty() || coverage.size() > (1u << 20))
		throw std::invalid_argument("Native texture read coverage count is invalid");
	for (const auto& part : coverage) { validate_guest_surface_range(part.range); (void)guest_surface_byte_cursor(part); }
	const auto& pending = m_impl->pending;
	if (pending && (!revision_tracked || !pending->draw_batch))
	{
		m_impl->synchronize(m_impl->timeout, false);
		return true;
	}
	const auto batch = pending ? pending->draw_batch : m_impl->recording_batch;
	if (!batch) return false;
	bool hazard = false;
	m_impl->cpu([&]
	{
		bool direct_addressing = false;
		hazard = m_impl->memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
		{
			direct_addressing = direct;
			// Publication can invalidate an entire alias, including bytes outside the writer.
			return std::ranges::any_of(batch->retirements, [&](const auto& retained)
			{
				return std::ranges::any_of(coverage, [&](const auto& part)
					{ return active_storage_overlap(direct, lookup, part, retained.surface->storage); });
			});
		});
		if (hazard) return;
		const auto overlaps_retained = [&](const std::shared_ptr<physical_tickets>& tickets)
		{
			if (!tickets) return false;
			for (const auto& ticket : *tickets)
				for (const auto& part : coverage)
				{
					guest_surface_byte_cursor cursor(part);
					while (const auto run = cursor.next())
					{
						const auto overlap = m_impl->memory->physical_surface_overlaps_current(*ticket, *run);
						if (!overlap || *overlap) return true;
					}
				}
			return false;
		};
		// Remapping can hide the allocation captured at submission. Missing
		// physical identity on a remappable provider cannot prove independence.
		hazard = std::ranges::any_of(batch->backings, [&](const auto& backing)
		{
			return (pending && !direct_addressing && !backing.physical) || overlaps_retained(backing.physical);
		}) || std::ranges::any_of(batch->retirements, [&](const auto& retained)
		{
			return overlaps_retained(retained.physical);
		});
	});
	if (hazard) m_impl->synchronize(m_impl->timeout, !pending);
	return hazard;
}
struct prepared_surface_snapshot::state
{
	surface_cache::impl* cache = nullptr;
	std::weak_ptr<void> lifetime;
	std::uint64_t cache_generation = 0;
	std::thread::id thread = std::this_thread::get_id();
	context* context_owner = nullptr;
	cached_surface_ref source;
	std::weak_ptr<cached_surface> authority;
	std::uint64_t authority_generation = 0, authority_epoch = 0;
	surface_lease allocation;
	std::weak_ptr<surface_allocation> native_donor;
	std::vector<deferred_snapshot_piece> deferred_pieces;
	raw_surface_snapshot spill;
	id<MTLTexture> spill_import;
	std::size_t spill_height = 0;
	std::vector<surface_backing> backings;
	std::shared_ptr<sample_transfer> samples;
	std::shared_ptr<surface_typeless_transfer> typeless;
	std::shared_ptr<snapshot_guest_transfer> guest_transfer;
	id<MTLBuffer> guest_input;
	guest_surface_layout guest_layout{};
	std::list<surface_cache::impl::snapshot_entry> candidate;
	id<MTLTexture> image;
	cached_surface_description description;
	clear_rectangle rectangle{}, input_rectangle{};
	surface_format format;
	std::size_t prefix = 0;
	std::shared_ptr<const surface_texture_revision> receipt;
	std::optional<submission> encoded;
	bool copy = false, complete = false, published = false, from_deferred = false;
	std::uint64_t views_created = 0, views_reused = 0;
	void outside() const
	{
		if (thread != std::this_thread::get_id() || guest_memory_epoch::active() || surface_cache::impl::captured)
			throw std::logic_error("Native snapshot encoding/publication requires its owner outside CPU exclusion");
	}
};
prepared_surface_snapshot::prepared_surface_snapshot(std::shared_ptr<state> state) : m_state(std::move(state)) {}
id<MTLTexture> prepared_surface_snapshot::texture() const { return m_state->image; }
surface_format prepared_surface_snapshot::format() const { return m_state->format; }
std::size_t prepared_surface_snapshot::row_offset() const { return m_state->prefix; }
const cached_surface_description& prepared_surface_snapshot::source_description() const { return m_state->description; }
bool prepared_surface_snapshot::same_content(const prepared_surface_snapshot& other) const
{
	const auto& a = *m_state; const auto& b = *other.m_state;
	const auto source = a.authority.lock();
	return source && source == b.authority.lock() && a.cache == b.cache && a.cache_generation == b.cache_generation &&
		a.authority_generation == b.authority_generation && a.authority_epoch == b.authority_epoch &&
		a.format == b.format && a.image.sampleCount == b.image.sampleCount && a.prefix == b.prefix &&
		same_snapshot_donor(a.native_donor, b.native_donor) &&
		a.rectangle.x == b.rectangle.x && a.rectangle.y == b.rectangle.y &&
		a.rectangle.width == b.rectangle.width && a.rectangle.height == b.rectangle.height;
}
bool prepared_surface_snapshot::matches_source(const cached_surface_ref& source,bool require_cache_epoch) const
{
	const auto& state=*m_state;state.outside();const auto alive=state.lifetime.lock();
	return alive && source && state.authority.lock()==source && state.cache->positions.contains(source.get()) &&
		(!require_cache_epoch || state.cache->snapshot_generation==state.cache_generation) &&
		!source->faulted && source->generation==state.authority_generation && snapshot_epoch(*source)==state.authority_epoch &&
		(source->initialized || !source->deferred.empty());
}
bool prepared_surface_snapshot::cache_native_epoch(std::uint64_t epoch) const
{
	m_state->outside();const auto alive=m_state->lifetime.lock();return alive && m_state->cache->native_clock==epoch;
}
bool prepared_surface_snapshot::cache_texture_sources(std::uint64_t epoch, std::uint64_t topology,
	std::span<const guest_surface_layout> coverage, std::span<const std::shared_ptr<prepared_surface_snapshot>> retained) const
{
	const auto& state = *m_state;
	state.outside();
	const auto alive = state.lifetime.lock();
	if (!alive || state.cache->texture_topology_epoch != topology) return false;
	if (state.cache->native_clock == epoch) return true;
	// A completed inline draw may have advanced the clock for unrelated
	// targets. Never force another recording or pending writer to complete.
	if (state.cache->recording_batch || state.cache->pending) return false;
	try
	{
		const auto current = state.cache->current_texture_sources(coverage);
		return std::ranges::all_of(current, [&](const auto& surface)
		{
			return std::ranges::any_of(retained, [&](const auto& source) { return source->matches_source(surface); });
		});
	}
	catch (const std::runtime_error&) { return false; }
	catch (const std::length_error&) { return false; }
}
bool prepared_surface_snapshot::source_current(bool require_cache_epoch) const {return matches_source(m_state->authority.lock(),require_cache_epoch);}
std::shared_ptr<const surface_texture_revision> prepared_surface_snapshot::revision() const { return m_state->receipt; }
bool prepared_surface_snapshot::needs_copy() const { return m_state->copy; }
void prepared_surface_snapshot::validate_outside() const { m_state->outside(); }
id<MTLTexture> prepared_surface_snapshot::opaque_view(MTLPixelFormat format)
{
	auto& state = *m_state; state.outside();
	if (!state.image || state.image.textureType != MTLTextureType2D || state.image.sampleCount != 1)
		throw std::invalid_argument("Prepared opaque view requires a single-sample 2D native snapshot");
	const auto make = [&]
	{
		return make_texture_view(state.image,format,MTLTextureType2D,NSMakeRange(0,1),NSMakeRange(0,1),
			MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed,MTLTextureSwizzleGreen,MTLTextureSwizzleBlue,MTLTextureSwizzleOne));
	};
	const auto get = [&](auto& views, std::uint64_t& created, std::uint64_t& reused)
	{
		for (const auto& view : views) if (view.texture && view.format == format) { ++reused; return view.texture; }
		const auto image = make();
		for (auto& view : views) if (!view.texture) { view = {format,image}; ++created; break; }
		return image;
	};
	if (!state.candidate.empty()) return get(state.candidate.front().views,state.views_created,state.views_reused);
	if (const auto alive = state.lifetime.lock())
		for (auto& entry : state.cache->snapshots)
			if (entry.texture == state.image) return get(entry.views,state.cache->snapshot_stats.views_created,state.cache->snapshot_stats.views_reused);
	return make();
}
void prepared_surface_snapshot::encode(context& ctx)
{
	auto& state = *m_state; state.outside();
	if (&ctx != state.context_owner) throw std::invalid_argument("Prepared native snapshot belongs to another context");
	if (!state.copy) return;
	if (state.from_deferred && !matches_source(state.source))
		throw texture_revision_conflict();
	const auto recording = ctx.recording_submission();
	if (state.encoded)
	{
		if (state.encoded->same_batch(recording)) return;
		if (!state.encoded->ready() || !state.encoded->cancelled()) throw std::logic_error("Prepared native snapshot already belongs to a submission");
	}
	ctx.keep_alive(m_state);
	for (const auto& input : state.backings)
	{
		if (input.owner) ctx.keep_alive(input.owner);
		if (input.physical) ctx.keep_alive(input.physical);
	}
	const auto& rectangle = state.candidate.front().rectangle;
	texture_region target; target.size = MTLSizeMake(rectangle.width, rectangle.height, 1);
	if (!state.deferred_pieces.empty())
	{
		std::optional<texture_plane> previous_plane;
		for (const auto& piece : state.deferred_pieces)
		{
			texture_region area;
			const auto& output = piece.destination;
			area.origin = MTLOriginMake(output.x, output.y, 0);
			area.size = MTLSizeMake(output.width, output.height, 1);
			if (piece.guest)
			{
				const auto& origin = piece.projection.source_origin;
				state.guest_transfer->upload(ctx, state.description, state.guest_layout, state.guest_input,
					{origin[0], origin[1], output.width, output.height}, state.image, area, piece.plane);
			}
			else if (piece.typeless)
			{
				const auto& projection = piece.projection;
				const auto input = piece.allocation->render_texture(ctx);
				id<MTLTexture> readable = input;
				texture_region source_region;
				source_region.origin = MTLOriginMake(projection.source_origin[0], projection.source_origin[1], 0);
				source_region.size = MTLSizeMake(projection.source_extent[0], projection.source_extent[1], 1);
				nearest_projection window{
					{static_cast<std::uint32_t>(projection.copy_extent[0]), static_cast<std::uint32_t>(projection.copy_extent[1]), 1},
					{static_cast<std::uint32_t>(projection.copy_offset[0]), static_cast<std::uint32_t>(projection.copy_offset[1]), 0}};
				if (input.sampleCount > 1)
				{
					const auto selected=plan_typeless_source_window(piece.allocation->description().format,state.format,
						source_region.size.width,source_region.size.height,area.size.width,area.size.height,{projection.flip[0],projection.flip[1],false},window);
					auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:input.pixelFormat
						width:selected.extent[0] height:selected.extent[1] mipmapped:NO];
					descriptor.storageMode = MTLStorageModePrivate;
					descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
					readable = make_texture(ctx, descriptor);
					state.samples->expand_region(ctx, input, readable,{projection.source_origin[0]+selected.origin[0],projection.source_origin[1]+selected.origin[1]});
					source_region.origin = MTLOriginMake(0, 0, 0);
					source_region.size=MTLSizeMake(selected.extent[0],selected.extent[1],1);window=selected.projection;
				}
				state.typeless->nearest(ctx, piece.allocation->description().format, readable, source_region, state.format, state.image, area,
					{projection.flip[0], projection.flip[1], false}, window, piece.plane);
			}
			else if (piece.allocation)
				state.samples->project_into(ctx, piece.allocation->render_texture(ctx), state.image, piece.projection, area,
					previous_plane == piece.plane, piece.plane);
			else
			{
				const auto& origin = piece.projection.source_origin;
				copy_spill_rectangle(ctx, state.spill, state.spill_height,
					{origin[0], origin[1], output.width, output.height}, state.image, area, piece.plane);
			}
			previous_plane = piece.plane;
		}
	}
	else if (state.spill.buffer)
	{
		const auto storage = state.spill_import ? state.spill_import : state.image;
		// The spill already contains the expanded sample raster, with native
		// depth and stencil stored independently. Retain it until this blit ends.
		copy_spill_rectangle(ctx, state.spill, state.spill_height, rectangle, storage, target);
		if (state.spill_import)
			state.samples->import(ctx, storage, state.image);
	}
	else
	{
		auto input = target; input.origin = MTLOriginMake(state.input_rectangle.x, state.input_rectangle.y, 0);
		const auto native = state.samples ? state.allocation->expanded_texture(ctx) : state.allocation->render_texture(ctx);
		if (state.samples) state.samples->expand(ctx, state.allocation->render_texture(ctx), native);
		if (state.image.sampleCount > 1)
			input.size = target.size = MTLSizeMake(state.image.width, state.image.height, 1);
		copy_texture(ctx, native, input, state.image, target);
	}
	state.encoded = recording;
}
void prepared_surface_snapshot::completed()
{
	auto& state = *m_state; state.outside();
	if (state.copy)
	{
		if (!state.encoded || !state.encoded->ready()) throw std::logic_error("Native snapshot completion has not succeeded");
		state.encoded->wait(std::chrono::milliseconds(1));
		state.deferred_pieces.clear();
		state.spill_import = nil;
		state.guest_input = nil;
		state.guest_transfer.reset();
	}
	state.complete = true;
}
void prepared_surface_snapshot::publish()
{
	auto& state = *m_state; state.outside();
	if (!state.copy || state.published) return;
	if (!state.complete) throw std::logic_error("Native snapshot cannot publish before successful completion");
	state.published = true;
	const auto alive = state.lifetime.lock();
	if (!alive || state.cache->snapshot_generation != state.cache_generation) return;
	auto& cache = *state.cache;
	++cache.snapshot_stats.created;
	const auto& first = state.candidate.front();
	// A later native writer may keep an older job's immutable image valid,
	// but that job must not reinsert an obsolete ready-cache content key.
	if (state.source->faulted || state.source->generation != first.generation || snapshot_epoch(*state.source) != first.native_epoch) return;
	const auto limits = cache.snapshot_stats.limits;
	if (!limits.entries || first.bytes > limits.bytes) { ++cache.snapshot_stats.bypassed; return; }
	for (const auto& ready : cache.snapshots)
		if (ready.source.lock() == state.source && ready.budget == first.budget && ready.generation == first.generation && ready.native_epoch == first.native_epoch &&
			ready.texture.sampleCount == first.texture.sampleCount &&
			same_snapshot_donor(ready.native_donor, first.native_donor) &&
			ready.row_offset == first.row_offset && ready.rectangle.x == first.rectangle.x && ready.rectangle.y == first.rectangle.y &&
			ready.rectangle.width == first.rectangle.width && ready.rectangle.height == first.rectangle.height)
			return; // Another completed owner already published this content key.
	while (cache.snapshots.size() >= limits.entries || cache.snapshot_stats.bytes > limits.bytes - first.bytes)
		cache.evict_snapshot(std::prev(cache.snapshots.end()));
	cache.snapshot_stats.bytes += first.bytes;
	cache.snapshot_stats.views_created += state.views_created; cache.snapshot_stats.views_reused += state.views_reused;
	cache.snapshots.splice(cache.snapshots.begin(),state.candidate);
	cache.snapshot_stats.entries = cache.snapshots.size();
}
std::shared_ptr<prepared_surface_snapshot> prepared_surface_snapshot::settled() const
{
	const auto& original = *m_state;
	original.outside();
	if (original.copy && (!original.complete || !original.published))
		throw std::logic_error("Native snapshot execution owners cannot retire before confirmed publication");
	auto compact = std::make_shared<state>();
	compact->image = original.image;
	compact->format = original.format;
	compact->prefix = original.prefix;
	compact->description = original.description;
	compact->rectangle = original.rectangle;
	compact->native_donor = original.native_donor;
	compact->receipt = original.receipt;
	compact->context_owner = original.context_owner;
	compact->cache = original.cache;
	compact->lifetime = original.lifetime;
	compact->cache_generation = original.cache_generation;
	compact->authority = original.authority;
	compact->authority_generation = original.authority_generation;
	compact->authority_epoch = original.authority_epoch;
	compact->complete = true;
	return std::shared_ptr<prepared_surface_snapshot>(new prepared_surface_snapshot(std::move(compact)));
}
bool prepared_surface_snapshot::cache_current() const
{
	const auto& state = *m_state;
	state.outside();
	const auto alive = state.lifetime.lock();
	const auto source = state.authority.lock();
	return alive && source && state.cache->snapshot_generation == state.cache_generation && state.cache->positions.contains(source.get()) &&
		!source->faulted && source->generation == state.authority_generation && snapshot_epoch(*source) == state.authority_epoch;
}

id<MTLTexture> surface_cache::snapshot_texture(guest_surface_range range,
											   const std::function<bool(const cached_surface_description&)>& accepts, const std::shared_ptr<heap_budget>& budget)
{
	if (!accepts)
		throw std::invalid_argument("Native surface texture snapshot requires a validator");
	const auto mapping = [&](const cached_surface_description& description) -> std::optional<clear_rectangle>
	{
		if (!accepts(description))
			return {};
		const auto& allocation = description.allocation;
		const auto size = allocation.samples == 1 ? std::array{allocation.width, allocation.height} :
			sample_expanded_size(allocation.width, allocation.height, allocation.samples);
		return clear_rectangle{0, 0, size[0], size[1]};
	};
	return snapshot_texture_region(guest_byte_coverage(range), mapping, budget);
}
id<MTLTexture> surface_cache::snapshot_texture_region(const guest_surface_layout& coverage,
													  const std::function<std::optional<clear_rectangle>(const cached_surface_description&)>& region, const std::shared_ptr<heap_budget>& budget)
{
	return snapshot_texture_regions({&coverage, 1}, region, budget);
}
id<MTLTexture> surface_cache::snapshot_texture_regions(std::span<const guest_surface_layout> coverage,
	const std::function<std::optional<clear_rectangle>(const cached_surface_description&)>& region,
	const std::shared_ptr<heap_budget>& budget, std::shared_ptr<const surface_texture_revision>* source_revision, bool allow_guest_fallback, std::optional<surface_format>* source_format)
{
	if (source_revision) source_revision->reset();
	if (source_format) source_format->reset();
	if (!region) throw std::invalid_argument("Native surface texture snapshot needs a validator");
	const auto mapping = [&](const cached_surface_description& source) -> std::optional<surface_texture_region>
	{
		const auto pixels = region(source);
		return pixels ? std::optional{surface_texture_region{*pixels, 0}} : std::nullopt;
	};
	return snapshot_texture_mapping(coverage, mapping, budget, source_revision, allow_guest_fallback, source_format);
}
id<MTLTexture> surface_cache::snapshot_texture_mapping(std::span<const guest_surface_layout> coverage,
	const std::function<std::optional<surface_texture_region>(const cached_surface_description&)>& region,
	const std::shared_ptr<heap_budget>& budget, std::shared_ptr<const surface_texture_revision>* source_revision, bool allow_guest_fallback,
	std::optional<surface_format>* source_format, std::optional<std::size_t>* source_row_offset,bool defer_unsupported, const draw_publication& permit)
{
	if (source_revision) source_revision->reset();
	if (source_format) source_format->reset();
	if (source_row_offset) source_row_offset->reset();
	const auto prepared = prepare_texture_mapping(coverage,region,budget,allow_guest_fallback,{},true,defer_unsupported,{},permit);
	if (!prepared) return nil;
	if (prepared->needs_copy())
	{
		const auto& state = *prepared->m_state;
		m_impl->execute([&] { prepared->encode(m_impl->ctx); },[] {},{},{{state.source,snapshot_versions(*state.source)}});
		prepared->completed();
		m_impl->cpu([&] { if (!prepared->revision()->current()) throw surface_conflict_error(state.source->storage.parts.front().coverage.range); });
		prepared->publish();
	}
	if (source_revision) *source_revision = prepared->revision();
	if (source_format) *source_format = prepared->format();
	if (source_row_offset) *source_row_offset = prepared->row_offset();
	return prepared->texture();
}
std::uint64_t surface_cache::texture_source_epoch() const
{
	m_impl->outside_epoch();return m_impl->native_clock;
}
std::uint64_t surface_cache::texture_source_topology_epoch() const
{
	m_impl->outside_epoch();return m_impl->texture_topology_epoch;
}
std::vector<cached_surface_ref> surface_cache::texture_sources(std::span<const guest_surface_layout> coverage, bool revision_tracked)
{
	prepare_texture_read_boundary(coverage, revision_tracked);
	return m_impl->current_texture_sources(coverage);
}
std::shared_ptr<prepared_surface_snapshot> surface_cache::prepare_texture_mapping(std::span<const guest_surface_layout> coverage,
	const std::function<std::optional<surface_texture_region>(const cached_surface_description&)>& region,
	const std::shared_ptr<heap_budget>& budget, bool allow_guest_fallback, std::span<const std::shared_ptr<prepared_surface_snapshot>> pending, bool allow_new,
	bool defer_unsupported, const cached_surface_ref& source_only, const draw_publication& permit, bool revision_tracked,
	clear_rectangle* multisample_window, bool allow_full_multisample)
{
	m_impl->outside_epoch();
	if (multisample_window) *multisample_window = {};
	if (pending.size() > 64) throw std::length_error("Native snapshot preparation exceeds its batch bound");
	if (coverage.empty() || coverage.size() > (1u << 20)) throw std::invalid_argument("Native texture coverage count is invalid");
	for (const auto& part : coverage) { validate_guest_surface_range(part.range); (void)guest_surface_byte_cursor(part); }
	if (!region || !budget || budget->device() != m_impl->ctx.device())
		throw std::invalid_argument("Native surface texture snapshot needs a validator and matching heap budget");
	prepare_texture_read_boundary(coverage, revision_tracked);
	if (source_only) m_impl->check(source_only);
	const auto candidates = source_only ? std::vector<cached_surface_ref>{source_only} : m_impl->cpu_aliases(coverage, true);
	const auto map_region = [&](const cached_surface_ref& surface)
	{
		std::optional<surface_texture_region> value;
		m_impl->cpu([&]
		{
			m_impl->check_mapping(surface); value = region(surface->description); m_impl->check_mapping(surface);
		});
		return value;
	};
	cached_surface_ref source;
	clear_rectangle selected{};
	std::size_t row_offset = 0;
	bool prefer_native_multisample = false;
	bool ambiguous_clean_epoch = false;
	for (const auto& surface : candidates)
		if (surface->dirty || !surface->deferred.empty())
		{
			const auto mapping = map_region(surface);
			if (source || !mapping)
			{
				if (allow_guest_fallback || defer_unsupported) return {};
				throw std::invalid_argument("GPU surface texture overlap requires an unsupported alias conversion");
			}
			source = surface;
			selected = mapping->pixels;
			row_offset = mapping->guest_row_offset;
			prefer_native_multisample = mapping->native_multisample;
		}
	// VM writeback synchronizes guest storage but does not replace native
	// depthfloat precision. Prefer a compatible initialized backing while its
	// active-byte revision remains valid. Dirty/deferred aliases above always
	// take priority, including unsupported overlaps that must never fall back.
	if (!source)
		for (const auto& surface : candidates)
			if (surface->initialized && !surface->faulted && (surface->lease || surface->spill.buffer) &&
				m_impl->current(surface) == surface->revision)
			{
				const auto mapping = map_region(surface);
				if (!mapping)
					continue;
				if (source)
				{
					if (surface->native_write_epoch < source->native_write_epoch)
						continue;
					if (surface->native_write_epoch == source->native_write_epoch)
					{
						const auto& a = source->description;
						const auto& b = surface->description;
						const bool same_shape = a.allocation == b.allocation && a.row_pitch == b.row_pitch && a.tile == b.tile &&
												(!a.tile || a.address == b.address) &&
												bool(a.morton) == bool(b.morton) && (!a.morton || same(*a.morton, *b.morton)) &&
												selected.x == mapping->pixels.x && selected.y == mapping->pixels.y && selected.width == mapping->pixels.width && selected.height == mapping->pixels.height &&
												row_offset == mapping->guest_row_offset;
						const bool equivalent_vm = !surface->native_write_epoch && same_shape &&
							equivalent_storage(*m_impl->memory, source->storage, surface->storage);
						ambiguous_clean_epoch |= !equivalent_vm;
						continue;
					}
				}
				ambiguous_clean_epoch = false;
				source = surface;
				selected = mapping->pixels;
				row_offset = mapping->guest_row_offset;
				prefer_native_multisample = mapping->native_multisample;
			}
	if (ambiguous_clean_epoch)
		throw std::invalid_argument("Native clean surface texture aliases lack a unique write epoch");
	if (!source)
		return {};
	if (revision_tracked && m_impl->pending)
	{
		// An assembly may snapshot more than the bytes requested by the texture.
		std::vector<guest_surface_layout> source_coverage;
		source_coverage.reserve(source->storage.parts.size());
		for (const auto& part : source->storage.parts) source_coverage.push_back(part.coverage);
		if (prepare_texture_read_boundary(source_coverage, true)) throw texture_revision_conflict();
	}
	const auto& allocation = source->description.allocation;
	const auto size = allocation.samples == 1 ? std::array{allocation.width, allocation.height} :
		sample_expanded_size(allocation.width, allocation.height, allocation.samples);
	if (row_offset >= surface_guest_bytes(allocation.format))
		throw std::out_of_range("Native texture snapshot row prefix exceeds source guest texel");
	if (!selected.width || !selected.height || selected.x > size[0] || selected.width > size[0] - selected.x ||
		selected.y > size[1] || selected.height > size[1] - selected.y)
		throw std::out_of_range("Native texture snapshot rectangle exceeds source pixels");
	surface_list_guard pin(m_impl->pinned, source);
	const bool sample_raster = multisample_window && prefer_native_multisample && allocation.samples > 1 && !row_offset;
	auto deferred_input = project_deferred_snapshot(*source, selected);
	auto deferred_pieces = !deferred_input && allocation.samples > 1 ? compose_deferred_snapshot(*source, selected) :
		std::vector<deferred_snapshot_piece>{};
	if (deferred_input || !deferred_pieces.empty())
	{
		if (m_impl->recording_batch)
		{
			m_impl->batch_surface_boundary(source);
			deferred_input = project_deferred_snapshot(*source, selected);
			deferred_pieces = !deferred_input && allocation.samples > 1 ? compose_deferred_snapshot(*source, selected) :
				std::vector<deferred_snapshot_piece>{};
		}
		const auto aliases = m_impl->plan_aliases(source);
		if (std::any_of(aliases.begin(), aliases.end(), [](const auto& alias)
			{ return alias->dirty || !alias->deferred.empty(); }))
		{
			deferred_input.reset();
			deferred_pieces.clear();
		}
	}
	const bool from_deferred = deferred_input || !deferred_pieces.empty();
	const bool native_multisample_eligible = sample_raster && allow_full_multisample &&
		(source->deferred.empty() || deferred_input.has_value());
	const auto pixel_format = native_format(allocation);
	const bool native_multisample = native_multisample_eligible &&
		(!source->spill.buffer || deferred_input || supports_native_spill_import(pixel_format));
	const auto snapshot_samples = native_multisample ? allocation.samples : 1u;
	const auto& physical = deferred_input ? deferred_input->allocation->description() : allocation;
	const auto physical_size = allocation.samples > 1 ? sample_expanded_size(physical.width, physical.height, allocation.samples) : size;
	const clear_rectangle whole{0, 0, physical_size[0], physical_size[1]};
	const auto& shader_window = deferred_input ? deferred_input->rectangle : selected;
	const auto bind_sample_window = [&](unsigned samples)
	{
		if (sample_raster)
			*multisample_window = samples > 1 ? shader_window : clear_rectangle{0, 0, selected.width, selected.height};
	};
	const std::weak_ptr<surface_allocation> native_donor = deferred_input ? deferred_input->allocation : surface_lease{};
	const auto make_revision = [&]
	{
		std::vector<guest_surface_layout> watched;
		for (const auto& part : source->storage.parts) watched.push_back(part.coverage);
		return std::make_shared<surface_texture_revision>(m_impl->memory, std::move(watched), snapshot_versions(*source), source->description.mapping);
	};
	const auto existing = [&](unsigned samples) -> std::shared_ptr<prepared_surface_snapshot>
	{
		const auto& key = samples > 1 ? whole : selected;
		const auto donor = samples > 1 ? native_donor : std::weak_ptr<surface_allocation>{};
		// A covered deferred input has its own content epoch even before the
		// destination acquires a native allocation.
		if (source->faulted || (source->deferred.empty() ? !source->initialized : !from_deferred)) return {};
		const auto epoch = snapshot_epoch(*source);
		const auto& versions = snapshot_versions(*source);
		for (auto it = m_impl->snapshots.begin(); it != m_impl->snapshots.end(); ++it)
			if (it->source.lock() == source && it->budget == budget && it->generation == source->generation &&
				it->native_epoch == epoch && it->row_offset == row_offset && it->texture.sampleCount == samples &&
				same_snapshot_donor(it->native_donor, donor) &&
				it->rectangle.x == key.x && it->rectangle.y == key.y &&
				it->rectangle.width == key.width && it->rectangle.height == key.height)
			{
				// Our own writeback changes VM receipts, not the native content
				// version. Reloads/writers already change generation/epoch in prepare.
				std::shared_ptr<const surface_texture_revision> revision = it->versions == versions ? it->revision : make_revision();
				bool valid = false;
				m_impl->cpu([&] { valid = revision->current(); });
				if (!valid) return {};
				it->versions = versions; it->revision = std::move(revision);
				const auto texture = it->texture;
				m_impl->snapshots.splice(m_impl->snapshots.begin(), m_impl->snapshots, it);
				++m_impl->snapshot_stats.reused;
				source->last_use = ++m_impl->clock;
				auto state = std::make_shared<prepared_surface_snapshot::state>();
				state->image = texture; state->format = allocation.format; state->prefix = row_offset; state->receipt = m_impl->snapshots.front().revision;
				state->description = source->description;
				state->rectangle = key;
				state->native_donor = donor;
				state->authority = source; state->authority_generation = source->generation; state->authority_epoch = epoch;
				state->complete = true; state->context_owner = &m_impl->ctx;
				state->cache = m_impl.get(); state->lifetime = m_impl->snapshot_lifetime; state->cache_generation = m_impl->snapshot_generation;
				bind_sample_window(samples);
				return std::shared_ptr<prepared_surface_snapshot>(new prepared_surface_snapshot(std::move(state)));
			}
		for (const auto& previous : pending)
		{
			if (!previous || !previous->needs_copy()) continue;
			const auto& state = *previous->m_state;
			if (state.candidate.empty()) continue;
			const auto& item = state.candidate.front();
			if (state.source == source && state.cache == m_impl.get() && state.cache_generation == m_impl->snapshot_generation && state.context_owner == &m_impl->ctx && item.budget == budget && item.generation == source->generation &&
				item.native_epoch == epoch && item.versions == versions && item.row_offset == row_offset && item.texture.sampleCount == samples &&
				same_snapshot_donor(item.native_donor, donor) &&
				item.rectangle.x == key.x && item.rectangle.y == key.y && item.rectangle.width == key.width && item.rectangle.height == key.height)
			{
				bool valid = false;
				m_impl->cpu([&] { valid = state.receipt->current(); });
				if (!valid) return {};
				source->last_use = ++m_impl->clock;
				bind_sample_window(samples);
				return previous;
			}
		}
		return {};
	};
	m_impl->prune_snapshots();
	// A current snapshot stays valid when the source moves to spill storage.
	if (native_multisample_eligible && !native_multisample)
		if (const auto hit = existing(allocation.samples)) return hit;
	if (const auto hit = existing(snapshot_samples)) return hit;
	std::function<bool()> permit_preparation;
	if (permit) permit_preparation = [&] { return permit(m_impl->timeout); };
	if (from_deferred)
	{
		if (permit_preparation && !permit_preparation()) throw predicated_texture_suppressed{};
		if (m_impl->current(source) != source->deferred_revision)
			throw surface_conflict_error(source->storage.parts.front().coverage.range);
	}
	const auto valid_spill = [&]
	{
		return source->spill.buffer && source->initialized && !source->faulted && source->deferred.empty() &&
			m_impl->current(source) == source->revision;
	};
	bool from_spill = false;
	if (valid_spill())
	{
		m_impl->batch_surface_boundary(source);
		if (!m_impl->flush_aliases(source, permit_preparation)) throw predicated_texture_suppressed{};
		from_spill = valid_spill();
	}
	if (!from_deferred && !from_spill)
	{
		// An immutable resident source can be pinned for a later copy. Leave
		// materialization, alias writeback and reload to normal preparation.
		const bool resident = revision_tracked && m_impl->pending && source->initialized && source->lease &&
			!source->faulted && !source->spill.buffer && source->deferred.empty() &&
			m_impl->current(source) == source->revision &&
			std::ranges::none_of(m_impl->plan_aliases(source), [](const auto& alias)
				{ return alias->dirty || !alias->deferred.empty(); });
		if (resident) source->last_use = ++m_impl->clock;
		else if (!m_impl->prepare(source, surface_initialization::guest, permit_preparation))
			throw predicated_texture_suppressed{};
	}
	m_impl->prune_snapshots();
	if (const auto hit = existing(snapshot_samples)) return hit;
	if (!allow_new) throw std::length_error("Native snapshot pending-owner bound exceeded before allocation");
	auto state = std::make_shared<prepared_surface_snapshot::state>();
	state->cache = m_impl.get(); state->lifetime = m_impl->snapshot_lifetime; state->cache_generation = m_impl->snapshot_generation;
	state->source = source; state->format = allocation.format; state->prefix = row_offset;
	state->from_deferred = from_deferred;
	if (deferred_pieces.empty())
		state->allocation = deferred_input ? deferred_input->allocation : source->lease;
	state->deferred_pieces = std::move(deferred_pieces);
	if (std::ranges::any_of(state->deferred_pieces, [](const auto& piece) { return piece.typeless; }))
	{
		if (!m_impl->typeless) m_impl->typeless = std::make_shared<surface_typeless_transfer>(m_impl->ctx);
		state->typeless = m_impl->typeless;
	}
	if (std::ranges::any_of(state->deferred_pieces, [](const auto& piece) { return piece.guest; }))
	{
		auto guest = m_impl->read(source);
		if (guest.revisions != source->deferred_revision)
			throw surface_conflict_error(source->storage.parts.front().coverage.range);
		state->guest_input = make_upload(m_impl->ctx, guest.bytes);
		state->guest_layout = source->layout;
		if (!m_impl->snapshot_guest)
			m_impl->snapshot_guest = std::make_shared<snapshot_guest_transfer>(snapshot_guest_transfer{
				m_impl->color, m_impl->depth, m_impl->swizzle, m_impl->banked, m_impl->row_transfers});
		state->guest_transfer = m_impl->snapshot_guest;
	}
	if (from_spill || std::ranges::any_of(state->deferred_pieces, [](const auto& piece) { return !piece.allocation && !piece.guest; }))
	{
		state->spill = source->spill;
		state->spill_height = source->layout.height;
	}
	if (from_spill || from_deferred) source->last_use = ++m_impl->clock;
	state->description = source->description;
	state->rectangle = native_multisample ? whole : selected;
	state->input_rectangle = deferred_input && !native_multisample ? deferred_input->rectangle : state->rectangle;
	if (native_multisample) state->native_donor = native_donor;
	state->authority = source; state->authority_generation = source->generation; state->authority_epoch = snapshot_epoch(*source);
	state->context_owner = &m_impl->ctx; state->receipt = make_revision(); state->copy = true;
	state->backings = m_impl->retain_backings(std::array{surface_validation{source,snapshot_versions(*source)}});
	if (!from_spill && allocation.samples > 1 && !native_multisample) state->samples = m_impl->samples;
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
		width:native_multisample ? physical.width : selected.width
		height:native_multisample ? physical.height : selected.height mipmapped:NO];
	if (native_multisample)
	{
		bind_sample_window(snapshot_samples);
		descriptor.textureType = MTLTextureType2DMultisample;
		descriptor.sampleCount = snapshot_samples;
	}
	descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	if (!state->deferred_pieces.empty())
		descriptor.usage |= MTLTextureUsageRenderTarget;
	MTLTextureDescriptor* spill_import = nil;
	if (native_multisample && from_spill)
	{
		state->samples = m_impl->samples;
		descriptor.usage |= MTLTextureUsageRenderTarget;
		spill_import = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixel_format
			width:whole.width height:whole.height mipmapped:NO];
		spill_import.storageMode = MTLStorageModePrivate;
		spill_import.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	}
	for (;;)
	{
		std::exception_ptr failure;
		@autoreleasepool
		{
			try
			{
				const auto image = budget->texture(descriptor);
				if (spill_import) state->spill_import = budget->texture(spill_import);
				state->image = image;
			}
			catch (...) { failure = std::current_exception(); }
		}
		if (!failure) break;
		try { std::rethrow_exception(failure); }
		catch (const heap_budget_error&)
		{
			if (m_impl->pool.trim()) continue;
			if (m_impl->snapshots.empty()) throw;
			m_impl->evict_snapshot(std::prev(m_impl->snapshots.end()));
		}
	}
	state->candidate.push_front({source,source->generation,state->authority_epoch,snapshot_versions(*source),budget,state->rectangle,row_offset,state->image.heap.size,state->image,state->receipt,state->native_donor});
	m_impl->cpu([&] { if (!state->receipt->current()) throw surface_conflict_error(source->storage.parts.front().coverage.range); });
	bind_sample_window(snapshot_samples);
	return std::shared_ptr<prepared_surface_snapshot>(new prepared_surface_snapshot(std::move(state)));
}
guest_surface_snapshot surface_cache::cpu_read(guest_surface_range range)
{
	prepare_cpu_read(range);
	guest_surface_snapshot result{};
	m_impl->cpu([&] { result = m_impl->memory->read(range); });
	return result;
}
void surface_cache::prepare_cpu_read(guest_surface_range range)
{
	prepare_cpu_read_active(guest_byte_coverage(range));
}
void surface_cache::prepare_cpu_read_active(const guest_surface_layout& coverage)
{
	prepare_cpu_read_active(coverage, {});
}
void surface_cache::prepare_cpu_read_active(const guest_surface_layout& coverage, const draw_publication& permit)
{
	(void)guest_surface_byte_cursor(coverage);
	m_impl->synchronize(m_impl->timeout);
	const auto aliases = m_impl->cpu_aliases(coverage);
	for (const auto& surface : aliases)
		if (surface->dirty || !surface->deferred.empty())
		{
			if (permit && !permit(m_impl->timeout)) throw predicated_texture_suppressed{};
			m_impl->flush(surface);
		}
}
void surface_cache::cpu_write(guest_surface_range range, std::span<const std::byte> bytes)
{
	validate_guest_surface_range(range);
	if (bytes.size() != range.size)
		throw std::invalid_argument("CPU surface write byte count differs from range");
	const std::vector<std::byte> payload(bytes.begin(), bytes.end());
	cpu_write(range, [&]
			  { m_impl->memory->write(range, payload); });
}
void surface_cache::cpu_write(guest_surface_range range, const std::function<void()>& store)
{
	cpu_write(std::span<const guest_surface_range>{&range, 1}, store);
}
void surface_cache::cpu_write(std::span<const guest_surface_range> ranges, const std::function<void()>& store,
	const cpu_capture_dispatch& capture)
{
	std::vector<guest_surface_layout> coverage;
	coverage.reserve(ranges.size());
	for (const auto range : ranges) coverage.push_back(guest_byte_coverage(range));
	cpu_write_active(coverage, store, capture);
}
void surface_cache::cpu_write_active(std::span<const guest_surface_layout> coverage, const std::function<void()>& store,
	const cpu_capture_dispatch& capture)
{
	for (const auto& part : coverage) (void)guest_surface_byte_cursor(part);
	if (!store)
		throw std::invalid_argument("External surface writer is absent");
	m_impl->synchronize(m_impl->timeout);
	std::vector<cached_surface_ref> aliases;
	std::unordered_set<const cached_surface*> selected;
	const auto append_aliases = [&]
	{
		if (coverage.empty()) return;
		m_impl->memory->test_storage([&](bool direct, const guest_surface_memory::storage_lookup& lookup)
		{
			// Keep coverage order for writeback. Physical candidates are the
			// same for every part while the storage mapping is held.
			const auto physical = direct ? std::vector<cached_surface_ref>{} : m_impl->candidates(false, {});
			for (const auto& part : coverage)
			{
				const auto indexed = direct ? m_impl->range_candidates(std::span{&part.range, 1}) : std::vector<cached_surface_ref>{};
				for (const auto& surface : direct ? indexed : physical)
					if (active_storage_overlap(direct, lookup, part, surface->storage) && selected.insert(surface.get()).second)
						aliases.push_back(surface);
			}
			return false;
		});
	};
	append_aliases();
	for (const auto& surface : aliases)
		if (surface->dirty || !surface->deferred.empty())
			m_impl->flush(surface);
	std::vector<surface_retirement> retirements;
	m_impl->cpu([&]
	{
		// Mapping transitions can introduce another physical alias after the
		// preceding flush. Never wait for that alias's GPU work under exclusion.
		append_aliases();
		for (const auto& surface : aliases)
			if (surface->dirty || !surface->deferred.empty())
				throw surface_conflict_error(surface->storage.parts.front().coverage.range);
		retirements = impl::retain_metadata_storage({}, {}, aliases);
		std::vector<surface_validation> before_store;
		for (const auto& surface : aliases) before_store.push_back({surface, m_impl->current(surface)});
		const auto invalidate = [&]() noexcept
		{
			for (const auto& surface : aliases)
			{
				surface->initialized = false;
				surface->spill = {};
				surface->dirty = false;
				surface->reset_backing();
				surface->faulted = false;
				surface->native_write_epoch = 0;
			}
		};
		// A throwing producer may already have written a prefix. Preserve a
		// flushed native mirror only if its mapping and all watch versions
		// still match;otherwise guest bytes are authoritative before unlock.
		try { store(); }
		catch (...)
		{
			bool unchanged = false;
			try { m_impl->validate_in_epoch(before_store); unchanged = true; }
			catch (...) {}
			if (!unchanged) invalidate();
			throw;
		}
		invalidate();
	}, capture);
}
void surface_cache::synchronize(std::chrono::milliseconds timeout)
{
	m_impl->synchronize(timeout);
}
void surface_cache::prepare_draw_resources(const std::function<void()>& allocate)
{
	m_impl->outside_epoch();
	if (!allocate) throw std::invalid_argument("Native draw resource admission requires an allocator");
	const auto attempt = [&] { @autoreleasepool { allocate(); } };
	try { attempt(); return; }
	catch (const heap_budget_error& error)
	{
		if (error.requested > error.budget || (!m_impl->recording_batch && !m_impl->pending)) throw;
	}
	catch (const native_allocation_error&)
	{
		if (!m_impl->recording_batch && !m_impl->pending) throw;
	}
	// Never discard a successful prefix just to make room for the next draw.
	// GPU errors, VM conflicts and timeouts propagate without an allocation retry.
	m_impl->synchronize(m_impl->timeout);
	(void)m_impl->ctx.trim_upload_pages();
	++m_impl->transactions.draw_resource_retries;
	attempt();
}
void surface_cache::submit_draw_batch()
{
	m_impl->outside_epoch();
	m_impl->flush_draw_batch();
}
bool surface_cache::poll_submitted()
{
	if (m_impl->owner_thread != std::this_thread::get_id())
		throw std::logic_error("Native surface completion belongs to another render thread");
	m_impl->outside_epoch();
	const auto& pending = m_impl->pending;
	if (!pending) return true;
	if (!pending->work.ready()) return false;
	if (pending->rollback && pending->rollback->backup_work && !pending->rollback->backup_work->ready())
		return false;
	m_impl->synchronize(std::chrono::milliseconds::zero(), false);
	return true;
}
void surface_cache::set_wait_timeout(std::chrono::milliseconds timeout)
{
	if (timeout.count() < 0)
		throw std::invalid_argument("Native cache wait timeout must be nonnegative");
	m_impl->timeout = timeout;
}
surface_transaction_statistics surface_cache::transaction_statistics() const noexcept
{
	return m_impl->transactions;
}
surface_snapshot_statistics surface_cache::snapshot_statistics() const noexcept { return m_impl->snapshot_stats; }
void surface_cache::set_snapshot_limits(surface_snapshot_limits limits)
{
	m_impl->outside_epoch();
	if (limits.entries > 4096 || (limits.entries && !limits.bytes))
		throw std::invalid_argument("Native snapshot cache limits are invalid");
	m_impl->snapshot_stats.limits = limits;
	while (m_impl->snapshots.size() > limits.entries || m_impl->snapshot_stats.bytes > limits.bytes)
		m_impl->evict_snapshot(std::prev(m_impl->snapshots.end()));
}
void surface_cache::clear_snapshots()
{
	m_impl->outside_epoch();
	if (m_impl->snapshot_generation == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Native snapshot cache generation exhausted");
	++m_impl->snapshot_generation;
	while (!m_impl->snapshots.empty()) m_impl->evict_snapshot(std::prev(m_impl->snapshots.end()));
}
id<MTLTexture> surface_cache::opaque_snapshot_view(id<MTLTexture> texture, MTLPixelFormat format)
{
	m_impl->outside_epoch();
	if (!texture || texture.device != m_impl->ctx.device() || texture.textureType != MTLTextureType2D || texture.sampleCount != 1)
		throw std::invalid_argument("Opaque snapshot view requires a native 2D image on this device");
	const auto make = [&]
	{
		return make_texture_view(texture, format, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1),
			MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleOne));
	};
	for (auto& entry : m_impl->snapshots)
		if (entry.texture == texture)
		{
			for (const auto& view : entry.views)
				if (view.texture && view.format == format) { ++m_impl->snapshot_stats.views_reused; return view.texture; }
			const auto result = make();
			for (auto& view : entry.views)
				if (!view.texture) { view = {format, result}; break; }
			++m_impl->snapshot_stats.views_created;
			return result;
		}
	return make();
}
surface_pool_statistics surface_cache::memory_statistics() const
{
	return m_impl->pool.statistics();
}
std::shared_ptr<heap_budget> surface_cache::heap_ledger() const
{
	return m_impl->shared_ledger;
}
std::size_t surface_cache::reclaim_idle()
{
	m_impl->outside_epoch();
	clear_snapshots();
	return m_impl->pool.trim();
}
bool surface_cache::reclaim_clean()
{
	return m_impl->reclaim_clean();
}

surface_metadata_statistics surface_cache::metadata_statistics() const
{
	return m_impl->metadata;
}
std::size_t surface_cache::trim_metadata(std::size_t target_entries)
{
	m_impl->outside_epoch();
	m_impl->synchronize(m_impl->timeout);
	return m_impl->retire_to(target_entries, m_impl->metadata.limits.watch_parts);
}
} // namespace metal4
