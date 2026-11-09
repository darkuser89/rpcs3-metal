#pragma once

#include "M4ScaledImagePlan.h"
#include "M4TextureAssembly.h"
#include <tuple>

namespace metal4
{
// Transfer writers are encoded sequentially; they do not share a render
// pass's four color/one depth attachment limit. Keep metadata/retirement bounded.
inline constexpr std::size_t scaled_image_max_writers = 64;
inline constexpr std::size_t scaled_image_max_readers = 64;
struct scaled_image_input_region
{
	clear_rectangle pixels;
	std::size_t bytes;
	std::vector<display_storage_copy> copies;
};
// Project only selected guest-byte claims back to native texels. Packing a
// tile/Morton allocation is unnecessary:the ordinary linear surface codec can
// download these bounded rectangles,then gather their exact requested lanes.
// Metadata only;duplicate consumers share a packed texel,holes stay untouched.
[[nodiscard]] inline std::vector<scaled_image_input_region> plan_scaled_image_input_regions(
	const display_storage_source& source,std::span<const display_storage_copy> copies,std::size_t input,std::size_t byte_limit)
{
	constexpr std::size_t limit=1u<<20;
	const auto& g=source.geometry;const auto bpp=g.pixel_bytes;
	if(!bpp||byte_limit<bpp||copies.size()>limit)throw std::invalid_argument("Scaled input region budget cannot hold one native guest word");
	const auto inverse=texture_storage_runs(source);
	struct claim {std::size_t x,y,lane,target;};std::vector<claim> claims;
	for(const auto& copy:copies)if(copy.input==input)
	{
		if(!copy.size||copy.size>limit-claims.size()||copy.size>std::numeric_limits<std::size_t>::max()-copy.source||
			copy.size>std::numeric_limits<std::size_t>::max()-copy.target)throw std::length_error("Scaled input byte claims exceed bounded metadata");
		for(std::size_t at=0;at<copy.size;++at)
		{
			const auto logical=texture_storage_logical(inverse,copy.source+at);
			if(!logical)throw std::invalid_argument("Scaled input copy selects inactive native storage");
			const auto lane=g.morton?*logical%bpp:(*logical%g.row_pitch)%bpp;
			const auto pixel=texture_storage_pixel(g,*logical-lane);
			if(!pixel)throw std::invalid_argument("Scaled input byte has no native texel");
			claims.push_back({(*pixel)[0],(*pixel)[1],lane,copy.target+at});
		}
	}
	if(claims.empty())throw std::invalid_argument("Scaled input region has no selected bytes");
	std::sort(claims.begin(),claims.end(),[](const auto& a,const auto& b){return std::tie(a.y,a.x,a.lane,a.target)<std::tie(b.y,b.x,b.lane,b.target);});
	struct row_piece {std::size_t y,x,width,region;};std::vector<row_piece> rows;
	std::vector<scaled_image_input_region> result;
	std::map<std::pair<std::size_t,std::size_t>,std::size_t> previous;
	for(std::size_t begin=0;begin<claims.size();)
	{
		const auto x=claims[begin].x,y=claims[begin].y;auto last=x;std::size_t end=begin;
		while(end<claims.size()&&claims[end].y==y&&claims[end].x<=last+1&&claims[end].x-x<byte_limit/bpp)
		{last=claims[end].x;++end;}
		const auto width=last-x+1,bytes=width*bpp;const auto key=std::pair{x,width};
		auto found=previous.find(key);std::size_t region=result.size();
		if(found!=previous.end())
		{
			auto& part=result[found->second];
			if(part.pixels.y+part.pixels.height==y&&bytes<=byte_limit-part.bytes)
			{region=found->second;++part.pixels.height;part.bytes+=bytes;}
		}
		if(region==result.size()){result.push_back({{x,y,width,1},bytes,{}});previous[key]=region;}
		rows.push_back({y,x,width,region});begin=end;
	}
	std::size_t row=0;
	for(const auto& byte:claims)
	{
		while(row<rows.size()&&(rows[row].y<byte.y||(rows[row].y==byte.y&&byte.x>=rows[row].x+rows[row].width)))++row;
		if(row==rows.size()||rows[row].y!=byte.y||byte.x<rows[row].x)throw std::logic_error("Scaled input texel lost its bounded rectangle");
		auto& part=result[rows[row].region];
		const auto from=((byte.y-part.pixels.y)*part.pixels.width+byte.x-part.pixels.x)*bpp+byte.lane;
		if(!part.copies.empty()&&part.copies.back().source+part.copies.back().size==from&&part.copies.back().target+part.copies.back().size==byte.target)++part.copies.back().size;
		else part.copies.push_back({byte.target,from,1,0});
	}
	return result;
}
// Address metadata only. Geometry is relative to its complete allocation;
// segments retain the real EAs,including split pixels and physical mirrors.
[[nodiscard]] inline display_storage_source scaled_image_storage(const guest_surface_layout& image,
	const std::optional<banked_tile_layout>& tile,std::span<const guest_storage_segment> segments)
{
	auto geometry=plan_guest_surface(0,image.width,image.height,image.pixel_bytes,image.morton?0:image.row_pitch,image.morton);
	std::vector<guest_storage_segment> storage(segments.begin(),segments.end());
	if(storage.empty())storage.push_back({image.range.address,0,image.range.size});
	return {std::move(geometry),std::move(storage),false,0,tile};
}
[[nodiscard]] inline std::optional<std::size_t> scaled_image_storage_offset(const display_storage_source& source,
	std::size_t x,std::size_t y,std::size_t lane=0)
{
	const auto& g=source.geometry;
	if(x>=g.width||y>=g.height||lane>=g.pixel_bytes)return {};
	const auto raw=g.morton?checked_texture_sum(checked_texture_product(morton_index(
		{x+g.morton->origin[0],y+g.morton->origin[1],g.morton->origin[2]},plan_swizzle(*g.morton,g.pixel_bytes).log2),g.pixel_bytes),lane):
		checked_texture_sum(checked_texture_sum(checked_texture_product(y,g.row_pitch),checked_texture_product(x,g.pixel_bytes)),lane);
	if(!source.tile)return raw;
	const auto logical=checked_texture_sum(source.tile->offset,raw);
	if(logical>=source.tile->size)return {};
	const auto physical=banked_display_storage_offset(*source.tile,logical);
	return physical<source.tile->size?std::optional<std::size_t>{physical}:std::nullopt;
}
// Canonical packed pixel order -> actual guest byte spans. These are address
// receipts,never pixel data. Inactive bank cells cannot supply an input word.
[[nodiscard]] inline std::optional<std::vector<guest_storage_segment>> scaled_image_dense_segments(const display_storage_source& source)
{
	std::vector<guest_storage_segment> result;const auto& g=source.geometry;
	for(std::size_t y=0;y<g.height;++y)for(std::size_t byte=0;byte<g.width*g.pixel_bytes;)
	{
		const auto at=scaled_image_storage_offset(source,byte/g.pixel_bytes,y,byte%g.pixel_bytes);if(!at)return {};
		const auto* part=texture_assembly_segment(source.segments,*at);if(!part)return {};
		auto count=g.width*g.pixel_bytes-byte;
		if(g.morton)count=std::min(count,g.pixel_bytes-byte%g.pixel_bytes);
		if(source.tile)count=std::min(count,32-(*at&31));
		count=std::min(count,part->size-(*at-part->offset));
		const auto address=static_cast<std::uint32_t>(std::uint64_t(part->address)+*at-part->offset);
		const auto offset=y*g.width*g.pixel_bytes+byte;
		if(!result.empty()&&std::uint64_t(result.back().address)+result.back().size==address&&result.back().offset+result.back().size==offset)result.back().size+=count;
		else {if(result.size()==(1u<<20))throw std::length_error("Scaled-image packed source exceeds span budget");result.push_back({address,offset,count});}
		byte+=count;
	}
	return result;
}
struct scaled_image_output_region
{
	clear_rectangle packed, native;
	std::size_t output = 0;
};
struct scaled_image_output_byte_region
{
	clear_rectangle native;
	std::size_t output = 0, bytes = 0;
	// Canonical output bytes -> packed native window bytes. Unselected lanes
	// come from a GPU download of the current native window before this gather.
	std::vector<display_storage_copy> copies;
	bool preserve_background = true;
};

// Resolve output byte identities to bounded translation-only rectangles per
// native target. No pixels are read: every byte lane must preserve its word and every
// requested word must have exactly one complete native destination. Targets
// can extend past the requested image; their exterior stays in the old version.
[[nodiscard]] inline std::optional<std::vector<scaled_image_output_region>> plan_scaled_image_output_regions(
	guest_surface_memory& memory, std::span<const display_storage_source> targets,
	const display_storage_source& requested, std::size_t run_limit = 1u << 20, std::size_t window_limit = 4096,
	std::vector<display_storage_source>* cold = nullptr, std::size_t native_width_limit = 0,
	std::size_t writer_limit = scaled_image_max_writers)
{
	if (cold) cold->clear();
	if (!writer_limit || writer_limit > scaled_image_max_writers || targets.size() > writer_limit || requested.geometry.range.address)
		throw std::invalid_argument("Scaled output projection exceeds its bounded relative native targets");
	const auto& g = requested.geometry;
	const auto bpp = g.pixel_bytes;
	for (const auto& target : targets)
		if (target.geometry.pixel_bytes != bpp) return {};
	const auto spans = scaled_image_dense_segments(requested);
	if (!spans || !run_limit || !window_limit) return {};
	bool direct = false;
	memory.test_storage([&](bool value, const auto&) { direct = value; return false; });
	if (!direct)
	{
		// Unknown identity support must keep the coherent route available.
		// Probe only active bytes, never padding or inactive bank cells.
		std::size_t identities = 0;
		const auto known = [&](guest_surface_range range)
		{ return ++identities <= (1u << 20) && memory.storage_identity(range).has_value(); };
		for (const auto& span : *spans) if (!known({span.address, span.size})) return {};
		for (const auto& target : targets)
			for (const auto& part : project_display_storage_source(target))
			{
				guest_surface_byte_cursor cursor(part.coverage);
				while (const auto run = cursor.next()) if (!known(*run)) return {};
			}
	}
	const auto packed = plan_guest_surface(0, g.width, g.height, bpp);
	if (targets.empty() && cold && native_width_limit && !requested.tile &&
		(g.height > native_width_limit || g.morton))
	{
		// Preserve pitched vertical groups and Morton subviews instead of
		// flattening thousands of separated rows/words into IO segments.
		const auto columns = (g.width - 1) / native_width_limit + 1;
		const auto rows = (g.height - 1) / native_width_limit + 1;
		if (columns > writer_limit || rows > writer_limit / columns || (!g.morton && columns != 1)) return {};
		std::vector<display_storage_source> created;
		for (std::size_t y = 0; y < g.height; y += native_width_limit)
			for (std::size_t x = 0; x < g.width; x += native_width_limit)
			{
				const auto width = std::min(native_width_limit, g.width - x), height = std::min(native_width_limit, g.height - y);
				auto sw = g.morton;
				if (sw) { sw->origin[0] += x; sw->origin[1] += y; sw->size = {width, height, 1}; }
				display_storage_source storage{plan_guest_surface(0, width, height, bpp, sw ? 0 : g.row_pitch, sw), {}, false, 0};
				const auto begin = sw ? 0 : y * g.row_pitch;
				for (const auto& span : requested.segments)
				{
					const auto first = std::max(begin, span.offset), last = std::min(begin + storage.geometry.range.size, span.offset + span.size);
					if (first < last) storage.segments.push_back({static_cast<std::uint32_t>(std::uint64_t(span.address) + first - span.offset), first - begin, last - first});
				}
				created.push_back(std::move(storage));
			}
		const auto windows = plan_scaled_image_output_regions(memory, created, requested, run_limit, window_limit, nullptr, 0, writer_limit);
		if (!windows) return {};
		*cold = std::move(created); return windows;
	}
	// With no previous native owner, every active word is a cold output.
	// Reuse the same bounded row construction and recursive identity proof;
	// neither hardware-sized dimensions nor guest padding require VM pixels.
	display_storage_assembly assembly;
	if (targets.empty()) assembly.vm.push_back(packed.range);
	else assembly = plan_display_storage_assembly(memory, packed, targets, *spans);
	if (!assembly.vm.empty())
	{
		if (!cold || !native_width_limit || targets.size() == writer_limit) return {};
		std::size_t missing = 0;
		for (const auto& gap : assembly.vm)
		{
			if (gap.address % bpp || gap.size % bpp) return {};
			missing = checked_texture_sum(missing, gap.size / bpp);
		}
		const auto slots = writer_limit - targets.size();
		const bool atlas = (missing - 1) / native_width_limit + 1 > slots;
		std::optional<std::array<std::size_t, 2>> factored;
		if (atlas)
		{
			// A writer is a 2D texture, not one hardware-width row. Pack
			// complete rows together when flat rows exhaust writer slots;
			// keep the final short row separate so no cell is uninitialized.
			const auto rows = missing / native_width_limit;
			const auto count = rows / native_width_limit + (rows % native_width_limit != 0) + (missing % native_width_limit != 0);
			if (count > slots)
			{
				const auto squares = rows / native_width_limit;
				if (squares >= slots) return {};
				const auto remainder = missing - (squares * native_width_limit) * native_width_limit;
				// The remaining writer may fit an exact rectangle with a
				// narrower width. Search integer factors only; no spare cells,
				// float rounding or overflow-prone dimension squares.
				for (auto height = (remainder - 1) / native_width_limit + 1; height <= remainder / height; ++height)
					if (remainder % height == 0 && remainder / height <= native_width_limit)
					{ factored = {{remainder / height, height}}; break; }
				if (!factored) return {};
			}
		}
		std::vector<display_storage_source> combined(targets.begin(), targets.end()), created;
		std::size_t gap_index = 0, gap_at = 0;
		while (missing)
		{
			const auto width = factored && missing / native_width_limit < native_width_limit ? (*factored)[0] : std::min(missing, native_width_limit);
			const auto height = factored && missing / native_width_limit < native_width_limit ? (*factored)[1] :
				atlas ? std::max<std::size_t>(1, std::min(native_width_limit, missing / native_width_limit)) : 1;
			const auto words = width * height;
			display_storage_source storage{plan_guest_surface(0, width, height, bpp), {}, false, 0};
			for (std::size_t at = 0; at < storage.geometry.range.size;)
			{
				const auto& gap = assembly.vm[gap_index];
				const auto position = std::size_t(gap.address) + gap_at;
				const auto* span = texture_assembly_segment(*spans, position);
				if (!span) throw std::logic_error("Scaled output hole lacks guest storage");
				const auto bytes = std::min({storage.geometry.range.size - at, gap.size - gap_at,
					span->size - (position - span->offset)});
				const auto address = static_cast<std::uint32_t>(std::uint64_t(span->address) + position - span->offset);
				if (!storage.segments.empty() && std::uint64_t(storage.segments.back().address) + storage.segments.back().size == address)
					storage.segments.back().size += bytes;
				else
				{
					if (storage.segments.size() == run_limit) return {};
					storage.segments.push_back({address, at, bytes});
				}
				at += bytes; gap_at += bytes;
				if (gap_at == gap.size) { ++gap_index; gap_at = 0; }
			}
			created.push_back(storage); combined.push_back(std::move(storage)); missing -= words;
		}
		const auto windows = plan_scaled_image_output_regions(memory, combined, requested, run_limit, window_limit, nullptr, 0, writer_limit);
		if (!windows) return {};
		*cold = std::move(created); return windows;
	}
	std::vector<std::vector<texture_storage_run>> inverse;
	for (const auto& target : targets) inverse.push_back(texture_storage_runs(target));
	struct row
	{
		std::size_t output, x, y, nx, ny, width;
	};
	struct word
	{
		std::size_t output, nx, ny;
		std::uint32_t lanes = 0;
	};
	std::vector<row> rows;
	std::map<std::size_t, word> fragments;
	if (!bpp || bpp > 16) return {};
	const auto append = [&](row value)
	{
		if (!rows.empty())
		{
			auto& last = rows.back();
			if (last.output == value.output && last.y == value.y && last.ny == value.ny &&
				last.x + last.width == value.x && last.nx + last.width == value.nx)
			{ last.width += value.width; return true; }
		}
		if (rows.size() == run_limit) return false;
		rows.push_back(value); return true;
	};
	const auto fragment = [&](std::size_t canonical, std::size_t output, std::size_t nx, std::size_t ny,
		std::size_t lane, std::size_t bytes)
	{
		const auto key = canonical / bpp;
		const auto found = fragments.find(key);
		if (found == fragments.end() && fragments.size() == run_limit) return false;
		auto& value = fragments.try_emplace(key, word{output, nx, ny}).first->second;
		const auto mask = ((1u << bytes) - 1) << lane;
		if (value.output != output || value.nx != nx || value.ny != ny || (value.lanes & mask)) return false;
		value.lanes |= mask; return true;
	};
	std::size_t runs = 0;
	for (const auto& copy : assembly.copies)
	{
		if (copy.input >= targets.size()) throw std::logic_error("Scaled output lost its native target");
		const auto& native = targets[copy.input].geometry;
		const auto& mapping = inverse[copy.input];
		for (std::size_t at = 0; at < copy.size;)
		{
			if (++runs > run_limit) return {};
			const auto physical = copy.source + at, canonical = copy.target + at;
			const auto end = std::upper_bound(mapping.begin(), mapping.end(), physical,
				[](auto value, const auto& run) { return value < run.physical; });
			if (end == mapping.begin()) return {};
			const auto& run = *std::prev(end);
			if (physical - run.physical >= run.size) return {};
			const auto logical = run.logical + physical - run.physical;
			const auto lane = (native.morton ? logical : logical % native.row_pitch) % bpp;
			const auto pixel = texture_storage_pixel(native, logical - lane);
			if (!pixel || lane != canonical % bpp) return {};
			const auto y = canonical / packed.row_pitch;
			const auto native_row = native.morton ? bpp - lane : (native.width - (*pixel)[0]) * bpp - lane;
			const auto count = std::min({copy.size - at, run.size - (physical - run.physical),
				native_row, packed.row_pitch - canonical % packed.row_pitch});
			if (!count || canonical >= packed.range.size || count > packed.range.size - canonical) return {};
			std::size_t used = 0;
			if (lane)
			{
				used = std::min(count, bpp - lane);
				if (!fragment(canonical, copy.input, (*pixel)[0], (*pixel)[1], lane, used)) return {};
			}
			const auto words = (count - used) / bpp;
			if (words && !append({copy.input, (canonical + used) % packed.row_pitch / bpp, y,
				(*pixel)[0] + (lane + used) / bpp, (*pixel)[1], words})) return {};
			used += words * bpp;
			if (used < count && !fragment(canonical + used, copy.input, (*pixel)[0] + (lane + used) / bpp,
				(*pixel)[1], 0, count - used)) return {};
			at += count;
		}
	}
	for (const auto& [key, value] : fragments)
	{
		if (value.lanes != (1u << bpp) - 1 ||
			!append({value.output, key % g.width, key / g.width, value.nx, value.ny, 1})) return {};
	}
	// Every native word has one writer, even when guest IO reorders words or
	// maps several requested rows into separated parts of the same texture.
	auto native_order = rows;
	std::sort(native_order.begin(), native_order.end(), [](const auto& a, const auto& b)
	{ return std::tie(a.output, a.ny, a.nx) < std::tie(b.output, b.ny, b.nx); });
	for (std::size_t i = 1; i < native_order.size(); ++i)
	{
		const auto& a = native_order[i - 1]; const auto& b = native_order[i];
		if (a.output == b.output && a.ny == b.ny && a.nx + a.width > b.nx) return {};
	}
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b)
	{ return std::tie(a.output, a.y, a.x) < std::tie(b.output, b.y, b.x); });
	std::vector<row> merged;
	std::vector<bool> selected(targets.size());
	std::size_t words = 0;
	for (const auto& part : rows)
	{
		selected[part.output] = true; words = checked_texture_sum(words, part.width);
		if (!merged.empty())
		{
			auto& last = merged.back();
			if (last.output == part.output && last.y == part.y && last.ny == part.ny &&
				last.x + last.width == part.x && last.nx + last.width == part.nx)
			{ last.width += part.width; continue; }
		}
		merged.push_back(part);
	}
	if (words != checked_texture_product(g.width, g.height) ||
		std::any_of(selected.begin(), selected.end(), [](bool value) { return !value; })) return {};
	std::vector<scaled_image_output_region> result;
	std::map<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>, std::size_t> previous;
	for (const auto& part : merged)
	{
		const auto key = std::tuple{part.output, part.x, part.nx, part.width};
		const auto found = previous.find(key);
		if (found != previous.end())
		{
			auto& last = result[found->second];
			if (last.packed.y + last.packed.height == part.y && last.native.y + last.native.height == part.ny)
			{ ++last.packed.height; ++last.native.height; continue; }
		}
		if (result.size() == window_limit) return {};
		previous[key] = result.size();
		result.push_back({{part.x, part.y, part.width, 1}, {part.nx, part.ny, part.width, 1}, part.output});
	}
	return result;
}

// Byte-exact fallback for existing native authorities when requested words do
// not preserve the target's lane alignment. No guest/native pixels are read.
// Missing native byte owners can become cold B8 authorities. Every cold
// cell is written; no unselected guest byte is invented as background.
[[nodiscard]] inline std::optional<std::vector<scaled_image_output_byte_region>> plan_scaled_image_output_byte_regions(
	guest_surface_memory& memory, std::span<const display_storage_source> targets,
	const display_storage_source& requested, std::size_t byte_limit,
	std::size_t claim_limit = 1u << 20, std::size_t window_limit = 4096,
	std::vector<display_storage_source>* cold = nullptr, std::size_t native_width_limit = 0,
	std::size_t writer_limit = scaled_image_max_writers)
{
	if (cold) cold->clear();
	if ((!cold && targets.empty()) || !writer_limit || writer_limit > scaled_image_max_writers ||
		targets.size() > writer_limit || requested.geometry.range.address)
		throw std::invalid_argument("Scaled byte output needs bounded existing relative native targets");
	const auto& g = requested.geometry;
	if (!g.pixel_bytes || byte_limit < g.pixel_bytes || !claim_limit || !window_limit) return {};
	for (const auto& target : targets)
		if (!target.geometry.pixel_bytes || target.geometry.pixel_bytes > 16 || byte_limit < target.geometry.pixel_bytes) return {};
	const auto packed = plan_guest_surface(0, g.width, g.height, g.pixel_bytes);
	if (packed.range.size > claim_limit) return {};
	const auto spans = scaled_image_dense_segments(requested);
	if (!spans) return {};
	bool direct = false;
	memory.test_storage([&](bool value, const auto&) { direct = value; return false; });
	if (!direct)
	{
		std::size_t identities = 0;
		const auto known = [&](guest_surface_range range)
		{ return ++identities <= claim_limit && memory.storage_identity(range).has_value(); };
		for (const auto& span : *spans) if (!known({span.address, span.size})) return {};
		for (const auto& target : targets) for (const auto& part : project_display_storage_source(target))
		{
			guest_surface_byte_cursor cursor(part.coverage);
			while (const auto run = cursor.next()) if (!known(*run)) return {};
		}
	}
	display_storage_assembly assembly;
	if (targets.empty()) assembly.vm.push_back(packed.range);
	else assembly = plan_display_storage_assembly(memory, packed, targets, *spans);
	if (!assembly.vm.empty())
	{
		if (!cold || !native_width_limit || targets.size() == writer_limit) return {};
		std::vector<guest_storage_segment> missing;
		std::size_t bytes = 0;
		for (const auto& gap : assembly.vm)
			for (std::size_t at = 0; at < gap.size;)
			{
				const auto position = std::size_t(gap.address) + at;
				const auto* span = texture_assembly_segment(*spans, position);
				if (!span) throw std::logic_error("Cold byte output lacks requested storage");
				const auto count = std::min(gap.size - at, span->size - (position - span->offset));
				if (!count || count > claim_limit - bytes || missing.size() == claim_limit) return {};
				const auto address = static_cast<std::uint32_t>(std::uint64_t(span->address) + position - span->offset);
				if (!missing.empty() && std::uint64_t(missing.back().address) + missing.back().size == address)
					missing.back().size += count;
				else missing.push_back({address, bytes, count});
				at += count; bytes += count;
			}
		const display_storage_source requested_bytes{plan_guest_surface(0, bytes, 1, 1), std::move(missing), false, 0};
		std::vector<display_storage_source> created;
		if (!plan_scaled_image_output_regions(memory, {}, requested_bytes, claim_limit, window_limit,
			&created, native_width_limit, writer_limit - targets.size())) return {};
		std::vector<display_storage_source> combined(targets.begin(), targets.end());
		combined.insert(combined.end(), created.begin(), created.end());
		auto windows = plan_scaled_image_output_byte_regions(memory, combined, requested, byte_limit,
			claim_limit, window_limit, nullptr, 0, writer_limit);
		if (!windows) return {};
		for (auto& part : *windows) if (part.output >= targets.size())
		{
			std::size_t copied = 0;
			for (const auto& copy : part.copies) copied += copy.size;
			if (copied != part.bytes) return {};
			part.preserve_background = false;
		}
		*cold = std::move(created);
		return windows;
	}
	std::vector<std::vector<texture_storage_run>> inverse;
	for (const auto& target : targets) inverse.push_back(texture_storage_runs(target));
	std::set<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> written;
	std::vector<bool> selected(targets.size());
	std::size_t bytes = 0;
	for (const auto& copy : assembly.copies)
	{
		if (copy.input >= targets.size() || !copy.size || copy.size > claim_limit - bytes) return {};
		selected[copy.input] = true;
		const auto& target = targets[copy.input].geometry;
		for (std::size_t at = 0; at < copy.size; ++at)
		{
			const auto logical = texture_storage_logical(inverse[copy.input], copy.source + at);
			if (!logical) return {};
			const auto lane = (target.morton ? *logical : *logical % target.row_pitch) % target.pixel_bytes;
			const auto pixel = texture_storage_pixel(target, *logical - lane);
			if (!pixel || !written.emplace(copy.input, (*pixel)[1], (*pixel)[0], lane).second) return {};
		}
		bytes += copy.size;
	}
	if (bytes != packed.range.size || std::ranges::any_of(selected, [](bool value) { return !value; })) return {};
	std::vector<scaled_image_output_byte_region> result;
	for (std::size_t output = 0; output < targets.size(); ++output)
		for (auto& part : plan_scaled_image_input_regions(targets[output], assembly.copies, output, byte_limit))
		{
			if (result.size() == window_limit) return {};
			for (auto& copy : part.copies) std::swap(copy.source, copy.target);
			result.push_back({part.pixels, output, part.bytes, std::move(part.copies)});
		}
	return result;
}

// Prove the complete requested raster against one native rectangle. The
// first matching address is only a candidate:every byte is checked through
// one stable physical-identity callback. No guest or native pixels are read.
[[nodiscard]] inline std::optional<clear_rectangle> plan_native_scaled_image_region(guest_surface_memory& memory,
	const display_storage_source& source,const display_storage_source& requested)
{
	const auto& a=source.geometry;const auto& b=requested.geometry;
	if(a.range.address||b.range.address||source.segments.empty()||requested.segments.empty()||
		a.pixel_bytes!=b.pixel_bytes||b.width>a.width||b.height>a.height)return {};
	const auto inverse=texture_storage_runs(source);
	const auto first=scaled_image_storage_offset(requested,0,0);
	if(!first)return {};
	const auto* first_segment=texture_assembly_segment(requested.segments,*first);if(!first_segment)return {};
	const guest_surface_range initial{static_cast<std::uint32_t>(std::uint64_t(first_segment->address)+*first-first_segment->offset),1};
	std::optional<clear_rectangle> result;bool ambiguous=false;std::size_t checks=0;
	const auto prove=[&](const guest_surface_memory::storage_rebase& rebase,const guest_surface_memory::storage_equivalent& equal)
	{
		for(const auto& segment:source.segments)if(const auto anchor=rebase({segment.address,segment.size},initial))
		{
			const auto logical=texture_storage_logical(inverse,segment.offset+anchor->address-segment.address);
			const auto pixel=logical?texture_storage_pixel(a,*logical):std::nullopt;
			if(!pixel||b.width>a.width-(*pixel)[0]||b.height>a.height-(*pixel)[1])continue;
			bool valid=true;
			for(std::size_t y=0;y<b.height&&valid;++y)for(std::size_t byte=0;byte<b.width*b.pixel_bytes;)
			{
				if(++checks>(1u<<20))throw std::length_error("Native scaled-image identity proof exceeds run budget");
				const auto x=byte/b.pixel_bytes,lane=byte%b.pixel_bytes;
				const auto from=scaled_image_storage_offset(source,(*pixel)[0]+x,(*pixel)[1]+y,lane),to=scaled_image_storage_offset(requested,x,y,lane);
				if(!from||!to){valid=false;break;}
				const auto* s=texture_assembly_segment(source.segments,*from);const auto* d=texture_assembly_segment(requested.segments,*to);
				if(!s||!d){valid=false;break;}
				auto count=b.width*b.pixel_bytes-byte;
				if(a.morton||b.morton)count=std::min(count,b.pixel_bytes-lane);
				if(source.tile)count=std::min(count,32-(*from&31));
				if(requested.tile)count=std::min(count,32-(*to&31));
				count=std::min({count,s->size-(*from-s->offset),d->size-(*to-d->offset)});
				if(!equal({static_cast<std::uint32_t>(std::uint64_t(s->address)+*from-s->offset),count},
					{static_cast<std::uint32_t>(std::uint64_t(d->address)+*to-d->offset),count})){valid=false;break;}
				byte+=count;
			}
			if(!valid)continue;
			const clear_rectangle value{(*pixel)[0],(*pixel)[1],b.width,b.height};
			if(result&&(result->x!=value.x||result->y!=value.y))ambiguous=true;
			result=value;
		}
		return result.has_value()&&!ambiguous;
	};
	(void)memory.test_rebase(prove);return ambiguous?std::nullopt:result;
}
} // namespace metal4
