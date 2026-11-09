#pragma once

#include "M4Present.h"

#include <vector>

namespace metal4
{
enum class overlay_primitive { quads, triangle_strip, triangle_fan, lines, line_strip };
enum class overlay_shape : std::uint32_t { ellipse = 1, box, rounded_box };

struct overlay_distance_field
{
	overlay_shape shape;
	std::array<float, 2> center, half_size;
	float radius, border_width;
	std::array<float, 4> border_color;
	[[nodiscard]] bool valid() const;
};

struct overlay_image
{
	std::uint32_t width, height, layers;
	bool glyphs;
	std::vector<std::uint8_t> bytes;
	bool operator==(const overlay_image&) const = default;
};

inline constexpr std::size_t max_overlay_icon_file_bytes = 4u << 20;
[[nodiscard]] std::optional<overlay_image> decode_overlay_game_icon(std::span<const std::uint8_t> png);

struct overlay_draw
{
	overlay_primitive primitive = overlay_primitive::quads;
	std::array<std::uint32_t, 2> virtual_extent{};
	bool window_space = false;
	bool snap = true;
	std::array<float, 4> color{1, 1, 1, 1};
	std::optional<std::array<float, 4>> clip;
	std::optional<std::size_t> image;
	bool backbuffer = false;
	std::optional<overlay_distance_field> distance_field;
	std::optional<float> pulse_phase;
	float blur = 0;
	std::vector<std::array<float, 4>> vertices;
};

struct overlay_frame
{
	static constexpr std::size_t max_commands = 4096;
	static constexpr std::size_t max_vertices = 262144;
	static constexpr std::size_t max_image_bytes = 32u << 20;
	std::vector<overlay_draw> draws;
	std::vector<overlay_image> images;
};

struct overlay_image_cache_statistics
{
	std::size_t entries = 0, host_bytes = 0, native_bytes = 0;
	std::uint64_t uploads = 0, hits = 0, evicted = 0, rejected_uploads = 0;
};

class overlay_renderer final
{
	struct image_cache;
	std::unique_ptr<image_cache> m_images;
	compiler m_compiler;
	id<MTLLibrary> m_library = nil;
	id<MTLSamplerState> m_sampler = nil;
	std::array<id<MTLRenderPipelineState>, 18> m_pipelines{};
	[[nodiscard]] id<MTLRenderPipelineState> pipeline(MTLPixelFormat format, bool lines, unsigned mode);

  public:
	struct prepared_draw
	{
		MTLPrimitiveType primitive;
		std::size_t first, count;
		const overlay_draw* command;
	};
	struct prepared_frame
	{
		upload_slice vertices{};
		std::vector<prepared_draw> draws;
		std::vector<id<MTLTexture>> images;
		bool needs_backbuffer = false;
		id<MTLTexture> backbuffer = nil;
	};
	explicit overlay_renderer(context& ctx);
	~overlay_renderer();
	[[nodiscard]] overlay_image_cache_statistics image_statistics() const;
	bool reclaim_one_image();
	// The caller keeps the immutable snapshot alive until both destinations are encoded.
	[[nodiscard]] prepared_frame prepare(context& ctx, const overlay_frame& frame);
	void encode(context& ctx, const prepared_frame& frame, id<MTLTexture> target, MTLScissorRect content);
};
}
