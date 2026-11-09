#pragma once

#include "M4Draw.h"
#include "M4Pipeline.h"
#include "M4Texture.h"
#include "M4StereoPlan.h"

#include <optional>

namespace metal4
{
enum class present_filter
{
	nearest,
	linear
};
struct present_eye
{
	id<MTLTexture> texture = nil;
	texture_region region;
};
struct present_image
{
	id<MTLTexture> texture = nil;
	texture_region region;
	MTLScissorRect destination{0, 0, 1, 1};
	present_filter filter = present_filter::nearest;
	bool flip_x = false, flip_y = false;
	float gamma = 1;
	bool limited_rgb = false;
	std::optional<present_eye> right_eye;
	stereo_output stereo;
	std::array<std::optional<submission>, 2> uploads;

	void depend_on_uploads(context& ctx) const;
};

// An explicit-predecessor context can record into a free slot while the
// latest upload runs. Unrelated prior work follows normal queue admission.
void begin_present_commands(context& ctx, const present_image* image);

// Native final-image pass. The caller resolves the display-buffer epoch first;
// this object never reads guest VM or selects an RSX display buffer implicitly.
class presenter final
{
	friend class native_frame;
	id<MTLDevice> m_device;
	id<MTLLibrary> m_library;
	helper_catalog m_compiler;
	id<MTLSamplerState> m_nearest, m_linear;
	std::array<id<MTLRenderPipelineState>, 3> m_pipelines{};
	void validate(context& ctx, const present_image& image, id<MTLTexture> destination) const;
	[[nodiscard]] id<MTLRenderPipelineState> pipeline(MTLPixelFormat format);

  public:
	explicit presenter(context& ctx);
	// Metadata only; the platform owns view/layer attachment and drawable size.
	void configure_layer(CAMetalLayer* layer, MTLPixelFormat format = MTLPixelFormatBGRA8Unorm) const;
	// Encode into an already recording context, with a full opaque-black clear.
	// Tested offscreen; framebuffer-only drawable attachments are supported.
	void encode(context& ctx, const present_image& image, id<MTLTexture> destination);
	void encode_black(context& ctx, id<MTLTexture> destination) const;
	// Acquires no window. A nil drawable (occluded/unavailable layer) skips work.
	// Requires an idle encoding context. Queue ownership/lifetime is in context.
	[[nodiscard]] std::optional<submission> present(context& ctx, const present_image& image, CAMetalLayer* layer);
};
} // namespace metal4
