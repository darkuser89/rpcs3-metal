#pragma once

#include "Emu/RSX/Overlays/overlay_manager.h"
#include "Emu/RSX/Overlays/overlays.h"
#include "M4Overlay.h"
#include "M4FramePlan.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <mutex>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace metal4
{
class rsx_game_icon_source final
{
	struct file_version
	{
		dev_t device;
		ino_t inode;
		std::size_t size;
		std::array<long, 4> times;
		bool operator==(const file_version&) const = default;
	};

	struct source
	{
		std::string directory, path;
		std::optional<file_version> version;
		std::optional<overlay_image> icon;

		static std::optional<file_version> identify(const struct stat& info)
		{
			if (!S_ISREG(info.st_mode) || info.st_size <= 0 || static_cast<std::uint64_t>(info.st_size) > max_overlay_icon_file_bytes)
				return {};
			return file_version{info.st_dev, info.st_ino, static_cast<std::size_t>(info.st_size),
				{info.st_mtimespec.tv_sec, info.st_mtimespec.tv_nsec, info.st_ctimespec.tv_sec, info.st_ctimespec.tv_nsec}};
		}

		const overlay_image* get(std::string name)
		{
			if (name != directory)
			{
				directory = std::move(name);
				path = directory.empty() ? std::string{} : directory + "/ICON0.PNG";
				version.reset();
			}
			struct stat info{};
			const auto current = !path.empty() && ::stat(path.c_str(), &info) == 0 ? identify(info) : std::nullopt;
			if (!current)
			{
				version.reset();
				icon.reset();
				return nullptr;
			}
			if (current == version) return icon ? &*icon : nullptr;

			version.reset();
			icon.reset();
			// A path can be replaced between stat and open. Validate the actual
			// opened file and never block if the replacement is a FIFO.
			struct opened_file
			{
				int fd;
				~opened_file() { if (fd >= 0) ::close(fd); }
			} file{::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC)};
			if (file.fd < 0 || ::fstat(file.fd, &info) != 0 || identify(info) != current) return nullptr;
			std::vector<std::uint8_t> png(current->size);
			for (std::size_t offset = 0; offset < png.size();)
			{
				const auto count = ::read(file.fd, png.data() + offset, png.size() - offset);
				if (count < 0 && errno == EINTR) continue;
				if (count <= 0) return nullptr;
				offset += static_cast<std::size_t>(count);
			}
			if (::fstat(file.fd, &info) != 0 || identify(info) != current) return nullptr;
			icon = decode_overlay_game_icon(png);
			version = current;
			return icon ? &*icon : nullptr;
		}
	};
	source m_preferred, m_fallback;

  public:
	const overlay_image* get(std::string preferred, std::string fallback)
	{
		const bool distinct = preferred != fallback;
		if (const auto* icon = m_preferred.get(std::move(preferred))) return icon;
		return distinct ? m_fallback.get(std::move(fallback)) : nullptr;
	}
};

inline overlay_frame capture_rsx_overlays(rsx::overlays::display_manager& manager,
	std::optional<rsx::overlays::resource_config>& standard_images, const auto& game_icon, frame_extent extent, u64 timestamp)
{
	using namespace rsx::overlays;
	constexpr std::size_t max_views = 256;
	std::vector<std::shared_ptr<overlay>> views;
	std::vector<u32> dirty;
	{
		std::lock_guard lock(manager);
		for (const auto& view : manager.get_views())
		{
			if (views.size() == max_views) break;
			if (view && view->visible) views.push_back(view);
		}
		for (const auto& view : manager.get_dirty())
		{
			if (dirty.size() == max_views) break;
			if (view) dirty.push_back(view->uid);
		}
	}
	if (!dirty.empty()) manager.dispose(dirty);
	struct image_key
	{
		const void* identity;
		std::uint32_t layers;
		bool glyphs;
		bool operator==(const image_key&) const = default;
	};
	std::vector<image_key> keys;
	overlay_frame result;
	std::size_t image_bytes = 0, vertices = 0;
	std::optional<const overlay_image*> title_icon;
	for (const auto& view : views)
	{
		view->set_render_viewport(static_cast<u16>(extent.width), static_cast<u16>(extent.height));
		view->update(timestamp);
		if (!view->visible) continue;
		const auto compiled = view->get_compiled();
		for (const auto& source : compiled.draw_commands)
		{
			if (result.draws.size() == overlay_frame::max_commands) return result;
			const auto& config = source.config;
			if (source.verts.empty()) continue;
			overlay_draw draw;
			draw.virtual_extent = {view->get_virtual_width(), view->get_virtual_height()};
			if (!draw.virtual_extent[0] || !draw.virtual_extent[1]) continue;
			draw.window_space = view->use_window_space;
			draw.snap = !config.disable_vertex_snap;
			draw.color = {config.color.r, config.color.g, config.color.b, config.color.a};
			draw.blur = std::min(float(config.blur_strength) / 100.f, 1.f);
			if (config.pulse_glow)
			{
				draw.pulse_phase = float(timestamp / 1000) * config.pulse_speed_modifier - config.pulse_sinus_offset;
				if (!std::isfinite(*draw.pulse_phase)) continue;
			}
			const auto& sdf = config.sdf_config;
			if (sdf.func != sdf_function::none)
			{
				static_assert(static_cast<unsigned>(sdf_function::ellipse) == static_cast<unsigned>(overlay_shape::ellipse) &&
					static_cast<unsigned>(sdf_function::box) == static_cast<unsigned>(overlay_shape::box) &&
					static_cast<unsigned>(sdf_function::rounded_box) == static_cast<unsigned>(overlay_shape::rounded_box));
				draw.distance_field = overlay_distance_field{static_cast<overlay_shape>(sdf.func), {sdf.cx, sdf.cy},
					{sdf.hx, sdf.hy}, sdf.br, sdf.bw, {sdf.border_color.r, sdf.border_color.g, sdf.border_color.b, sdf.border_color.a}};
				if (!draw.distance_field->valid()) continue;
			}
			if (draw.color[3] <= 0 && (!draw.distance_field || draw.distance_field->border_color[3] <= 0)) continue;
			if (config.clip_region)
				draw.clip = std::array{config.clip_rect.x1, config.clip_rect.y1, config.clip_rect.x2, config.clip_rect.y2};
			const auto finite = [](float value) { return std::isfinite(value); };
			if (!std::ranges::all_of(draw.color, finite) || (draw.clip && !std::ranges::all_of(*draw.clip, finite))) continue;
			const auto count = source.verts.size();
			if (count > overlay_frame::max_vertices - vertices) continue;
			std::size_t expanded = count;
			switch (config.primitives)
			{
			case primitive_type::quad_list:
				if (count % 4) continue;
				draw.primitive = overlay_primitive::quads;
				expanded = count / 4 * 6;
				break;
			case primitive_type::triangle_fan:
				if (count < 3) continue;
				draw.primitive = overlay_primitive::triangle_fan;
				expanded = (count - 2) * 3;
				break;
			case primitive_type::triangle_strip:
				if (count < 3) continue;
				draw.primitive = overlay_primitive::triangle_strip;
				break;
			case primitive_type::line_list:
				if (count % 2) continue;
				draw.primitive = overlay_primitive::lines;
				break;
			case primitive_type::line_strip:
				if (count < 2) continue;
				draw.primitive = overlay_primitive::line_strip;
				break;
			default: continue;
			}
			if (expanded > overlay_frame::max_vertices - vertices) continue;
			if (!std::ranges::all_of(source.verts, [&](const auto& vertex) { return std::ranges::all_of(vertex.values, finite); })) continue;
			if (config.texture_ref == image_resource_id::backbuffer)
				draw.backbuffer = true;
			else if (config.texture_ref != image_resource_id::none)
			{
				const bool glyphs = config.texture_ref == image_resource_id::font_file;
				const overlay_image* icon = nullptr;
				if (config.texture_ref == image_resource_id::game_icon)
				{
					if (!title_icon) title_icon = game_icon();
					icon = *title_icon;
				}
				const auto* raw = config.texture_ref == image_resource_id::raw_image ?
					static_cast<const image_info_base*>(config.external_data_ref) : nullptr;
				if (config.texture_ref < image_resource_id::raw_image)
				{
					if (!standard_images)
					{
						resource_config resources;
						resources.load_files();
						standard_images.emplace(std::move(resources));
					}
					const auto index = std::size_t(config.texture_ref - 1);
					if (index < standard_images->texture_raw_data.size()) raw = standard_images->texture_raw_data[index].get();
				}
				if ((!glyphs || !config.font_ref) && !icon && (!raw || raw->channels != 4 || raw->w <= 0 || raw->h <= 0)) continue;
				const auto size = glyphs ? config.font_ref->get_glyph_data_dimensions() :
					icon ? size3u{icon->width, icon->height, 1} : size3u{static_cast<u32>(raw->w), static_cast<u32>(raw->h), 1};
				if (!size.depth || size.depth > 32 || size.width > 16384 || size.height > 16384) continue;
				const void* identity = glyphs ? static_cast<const void*>(config.font_ref) : icon ? static_cast<const void*>(icon) : raw;
				const image_key key{identity, size.depth, glyphs};
				const auto known = std::find(keys.begin(), keys.end(), key);
				draw.image = static_cast<std::size_t>(known - keys.begin());
				if (known == keys.end())
				{
					const auto bytes = std::size_t(size.width) * size.height * size.depth * (glyphs ? 1 : 4);
					if (!bytes || bytes > overlay_frame::max_image_bytes - image_bytes) continue;
					std::span<const u8> pixels;
					if (glyphs) pixels = config.font_ref->get_glyph_data();
					else if (icon) pixels = icon->bytes;
					else if (const auto* data = raw->get_data()) pixels = {data, raw->size_bytes()};
					if (pixels.size() != bytes) continue;
					result.images.push_back({size.width, size.height, size.depth, glyphs, {pixels.begin(), pixels.end()}});
					keys.push_back(key);
					image_bytes += bytes;
				}
			}
			draw.vertices.reserve(count);
			for (const auto& vertex : source.verts)
				draw.vertices.push_back({vertex.values[0], vertex.values[1], vertex.values[2], vertex.values[3]});
			result.draws.push_back(std::move(draw));
			vertices += expanded;
		}
	}
	return result;
}
}
