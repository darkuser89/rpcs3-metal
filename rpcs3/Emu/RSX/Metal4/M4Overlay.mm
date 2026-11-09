#include "M4Overlay.h"
#include "M4HeapBudget.h"

#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <cmath>
#include <list>

namespace metal4
{
namespace
{
constexpr auto overlay_source = R"metal(
#include <metal_stdlib>
using namespace metal;
struct overlay_parameters {
    float4 color, clip, border_color, distance_field;
    float2 virtual_size, pixel_size, center;
    uint snap, shape;
    float blur;
};
struct overlay_varyings {
    float4 position [[position]];
    float2 uv;
};
vertex overlay_varyings overlay_vertex(device const float4* vertices [[buffer(0)]],
    constant overlay_parameters& p [[buffer(1)]], uint id [[vertex_id]]) {
    const float4 v = vertices[id];
    float2 unit = v.xy / p.virtual_size;
    if (p.snap) unit = round(unit * p.pixel_size) / p.pixel_size;
    return {float4(unit * float2(2, -2) + float2(-1, 1), 0, 1), v.zw};
}
inline void overlay_clip(overlay_varyings in, constant overlay_parameters& p) {
    if (any(in.position.xy < p.clip.xy) || any(in.position.xy > p.clip.zw)) discard_fragment();
}
inline float4 overlay_tint(overlay_varyings in, constant overlay_parameters& p) {
    if (p.shape == 0) return p.color;
    const float2 point = floor(in.position.xy) - p.center;
    const float2 half_size = p.distance_field.xy;
    const float radius = p.shape == 3 ? p.distance_field.z : 0.0f;
    const float2 q = abs(point) - half_size + radius;
    const float distance = p.shape == 1 ? (length(point / half_size) - 1.0f) * length(half_size) :
        length(max(q, 0.0f)) + min(max(q.x, q.y), 0.0f) - radius;
    const float width = max(fwidth(distance), 1e-6f);
    const float inner = -p.distance_field.w;
    const float fill = 1.0f - smoothstep(inner - width, inner + width, distance);
    const float border = 1.0f - smoothstep(-width, width, distance);
    return mix(p.border_color * border, p.color, fill);
}
inline float4 overlay_blur(texture2d<float> image, sampler filter, float2 uv, float2 spacing) {
    const float weights[3] = {1.0f, 2.0f, 1.0f};
    float4 result = 0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            result += image.sample(filter, uv + float2(x, y) * spacing, level(0)) * weights[x+1] * weights[y+1];
    return result * (1.0f / 16.0f);
}
inline float4 overlay_image_color(texture2d<float> image, sampler filter, float2 uv, float strength) {
    const float4 original = image.sample(filter, uv, level(0));
    if (strength == 0) return original;
    const float2 texel = 1.0f / float2(image.get_width(), image.get_height());
    const float2 spacing = max(texel, 1.0f / float2(640, 360));
    const float4 blurred = (overlay_blur(image, filter, uv - float2(texel.x, 0), spacing) +
        overlay_blur(image, filter, uv + float2(texel.x, 0), spacing) +
        overlay_blur(image, filter, uv + float2(0, texel.y), spacing)) / 3.0f;
    return mix(original, blurred, strength);
}
fragment float4 overlay_solid(overlay_varyings in [[stage_in]], constant overlay_parameters& p [[buffer(1)]]) {
    const float4 tint = overlay_tint(in, p);
    overlay_clip(in, p);
    return tint;
}
fragment float4 overlay_rgba(overlay_varyings in [[stage_in]], constant overlay_parameters& p [[buffer(1)]],
    texture2d<float> image [[texture(0)]], sampler filter [[sampler(0)]]) {
    const float4 tint = overlay_tint(in, p);
    overlay_clip(in, p);
    return overlay_image_color(image, filter, in.uv, p.blur) * tint;
}
fragment float4 overlay_glyph(overlay_varyings in [[stage_in]], constant overlay_parameters& p [[buffer(1)]],
    texture2d_array<float> image [[texture(0)]], sampler filter [[sampler(0)]]) {
    const float4 tint = overlay_tint(in, p);
    overlay_clip(in, p);
    const uint page = uint(clamp(floor(in.uv.y), 0.0f, float(image.get_array_size() - 1)));
    const float coverage = image.sample(filter, float2(in.uv.x, fract(in.uv.y)), page).r;
    return tint * coverage;
}
)metal";

struct alignas(16) overlay_parameters
{
	std::array<float, 4> color, clip, border_color, distance_field;
	std::array<float, 2> virtual_size, pixel_size, center;
	std::uint32_t snap, shape;
	float blur;
};
static_assert(sizeof(overlay_parameters) == 112 && offsetof(overlay_parameters, blur) == 96);

unsigned format_index(MTLPixelFormat format)
{
	switch (format)
	{
	case MTLPixelFormatBGRA8Unorm: return 0;
	case MTLPixelFormatBGRA8Unorm_sRGB: return 1;
	case MTLPixelFormatRGBA16Float: return 2;
	default: throw std::invalid_argument("Invalid native overlay target format");
	}
}

template <std::size_t N>
bool finite(const std::array<float, N>& values)
{
	return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}
}

bool overlay_distance_field::valid() const
{
	return (shape == overlay_shape::ellipse || shape == overlay_shape::box || shape == overlay_shape::rounded_box) &&
		finite(center) && finite(half_size) && half_size[0] > 0 && half_size[1] > 0 &&
		std::isfinite(radius) && radius >= 0 && std::isfinite(border_width) && border_width >= 0 && finite(border_color);
}

std::optional<overlay_image> decode_overlay_game_icon(std::span<const std::uint8_t> png)
{
	constexpr std::array<std::uint8_t, 8> signature{137, 80, 78, 71, 13, 10, 26, 10};
	constexpr std::array<std::uint8_t, 12> end_chunk{0, 0, 0, 0, 'I', 'E', 'N', 'D', 174, 66, 96, 130};
	constexpr NSUInteger dimension_limit = 1024;
	if (png.size() < signature.size() + end_chunk.size() || png.size() > max_overlay_icon_file_bytes ||
		!std::equal(signature.begin(), signature.end(), png.begin()) ||
		!std::equal(end_chunk.rbegin(), end_chunk.rend(), png.rbegin())) return {};
	// ImageIO can return a partial image from truncated data. Walk the chunk
	// boundaries first, so a terminator embedded inside a chunk is rejected.
	for (std::size_t cursor = signature.size(); cursor != png.size() - end_chunk.size();)
	{
		const auto body_end = png.size() - end_chunk.size();
		if (cursor > body_end || body_end - cursor < 12) return {};
		const auto length = (std::uint32_t(png[cursor]) << 24) | (std::uint32_t(png[cursor + 1]) << 16) |
			(std::uint32_t(png[cursor + 2]) << 8) | png[cursor + 3];
		if (length > body_end - cursor - 12) return {};
		cursor += 12 + length;
	}
	@autoreleasepool
	{
		NSData* bytes = [NSData dataWithBytes:png.data() length:png.size()];
		const std::unique_ptr<std::remove_pointer_t<CGImageSourceRef>, decltype(&CFRelease)> source(
			CGImageSourceCreateWithData((__bridge CFDataRef)bytes, nullptr), CFRelease);
		if (!source || CGImageSourceGetCount(source.get()) != 1 || CGImageSourceGetStatus(source.get()) != kCGImageStatusComplete) return {};
		NSDictionary* properties = CFBridgingRelease(CGImageSourceCopyPropertiesAtIndex(source.get(), 0, nullptr));
		const auto width = [properties[(__bridge NSString*)kCGImagePropertyPixelWidth] unsignedIntegerValue];
		const auto height = [properties[(__bridge NSString*)kCGImagePropertyPixelHeight] unsignedIntegerValue];
		if (!width || !height || width > dimension_limit || height > dimension_limit) return {};
		const std::unique_ptr<std::remove_pointer_t<CGImageRef>, decltype(&CGImageRelease)> image(
			CGImageSourceCreateImageAtIndex(source.get(), 0, nullptr), CGImageRelease);
		if (!image || CGImageGetWidth(image.get()) != width || CGImageGetHeight(image.get()) != height) return {};
		const std::unique_ptr<std::remove_pointer_t<CGColorSpaceRef>, decltype(&CGColorSpaceRelease)> color(
			CGColorSpaceCreateWithName(kCGColorSpaceSRGB), CGColorSpaceRelease);
		if (!color) return {};
		overlay_image output{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1, false,
			std::vector<std::uint8_t>(width * height * 4)};
		const std::unique_ptr<std::remove_pointer_t<CGContextRef>, decltype(&CGContextRelease)> bitmap(
			CGBitmapContextCreate(output.bytes.data(), width, height, 8, width * 4, color.get(),
				CGBitmapInfo(kCGBitmapByteOrder32Big) | CGBitmapInfo(kCGImageAlphaPremultipliedLast)), CGContextRelease);
		if (!bitmap) return {};
		CGContextSetBlendMode(bitmap.get(), kCGBlendModeCopy);
		CGContextDrawImage(bitmap.get(), CGRectMake(0, 0, width, height), image.get());
		CGContextFlush(bitmap.get());
		// The overlay pipeline uses straight source alpha. Restore it after Core Graphics.
		for (std::size_t i = 0; i < output.bytes.size(); i += 4)
		{
			const unsigned alpha = output.bytes[i + 3];
			if (alpha == 0 || alpha == 255) continue;
			for (unsigned channel = 0; channel < 3; ++channel)
				output.bytes[i + channel] = static_cast<std::uint8_t>(std::min(255u,
					(unsigned(output.bytes[i + channel]) * 255 + alpha / 2) / alpha));
		}
		return output;
	}
}

struct overlay_renderer::image_cache
{
	static constexpr std::size_t capacity = 32;
	static constexpr std::size_t byte_limit = 32u << 20;
	struct entry
	{
		std::size_t fingerprint;
		overlay_image image;
		id<MTLTexture> texture;
		std::size_t native_bytes;
		std::optional<submission> upload;
	};
	std::list<entry> entries;
	overlay_image_cache_statistics stats;

	void remove(std::list<entry>::iterator item)
	{
		stats.host_bytes -= item->image.bytes.size();
		stats.native_bytes -= item->native_bytes;
		entries.erase(item);
		--stats.entries;
		++stats.evicted;
	}

	id<MTLTexture> get(context& ctx, const overlay_image& image)
	{
		const auto recording = ctx.recording_submission();
		const auto host_bytes = image.bytes.size();
		const auto bytes = std::string_view(reinterpret_cast<const char*>(image.bytes.data()), host_bytes);
		const auto fingerprint = std::hash<std::string_view>{}(bytes);
		for (auto item = entries.begin(); item != entries.end();)
		{
			const auto candidate = item++;
			if (candidate->fingerprint != fingerprint || candidate->image != image) continue;
			if (auto& upload = candidate->upload; upload && !upload->same_batch(recording))
			{
				if (!upload->ready()) continue;
				try { upload->wait(); }
				catch (const std::runtime_error&)
				{
					++stats.rejected_uploads;
					remove(candidate);
					continue;
				}
				upload.reset();
			}
			entries.splice(entries.begin(), entries, candidate);
			++stats.hits;
			return candidate->texture;
		}

		const auto format = image.glyphs ? MTLPixelFormatR8Unorm : MTLPixelFormatRGBA8Unorm;
		const auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
			width:image.width height:image.height mipmapped:NO];
		if (image.glyphs) descriptor.textureType = MTLTextureType2DArray;
		descriptor.arrayLength = image.layers;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead;
		const auto ledger = ctx.buffer_ledger();
		// Native allocation can reclaim older cache entries through the shared ledger.
		const auto texture = ledger ? ledger->texture(descriptor) : make_texture(ctx, descriptor);
		const auto staging = make_upload(ctx, std::as_bytes(std::span(image.bytes)));
		const auto layer_bytes = host_bytes / image.layers;
		for (unsigned slice = 0; slice < image.layers; ++slice)
			buffer_to_texture(ctx, staging, slice * layer_bytes, texture,
				{0, slice, MTLOriginMake(0, 0, 0), MTLSizeMake(image.width, image.height, 1)});
		++stats.uploads;
		const auto native_bytes = std::max<std::size_t>({host_bytes, texture.allocatedSize, texture.heap.size});
		if (native_bytes > byte_limit - host_bytes) return texture;
		while (entries.size() >= capacity || stats.host_bytes + stats.native_bytes > byte_limit - host_bytes - native_bytes)
			remove(std::prev(entries.end()));
		entries.push_front({fingerprint, image, texture, native_bytes, recording});
		++stats.entries;
		stats.host_bytes += host_bytes;
		stats.native_bytes += native_bytes;
		return texture;
	}
};

overlay_renderer::overlay_renderer(context& ctx) : m_images(std::make_unique<image_cache>()), m_compiler(ctx) {}
overlay_renderer::~overlay_renderer() = default;

overlay_image_cache_statistics overlay_renderer::image_statistics() const
{
	return m_images->stats;
}

bool overlay_renderer::reclaim_one_image()
{
	const auto candidate = std::find_if(m_images->entries.rbegin(), m_images->entries.rend(),
		[](const auto& item) { return !item.upload || item.upload->ready(); });
	if (candidate == m_images->entries.rend()) return false;
	m_images->remove(std::prev(candidate.base()));
	return true;
}

id<MTLRenderPipelineState> overlay_renderer::pipeline(MTLPixelFormat format, bool lines, unsigned mode)
{
	auto& cached = m_pipelines[format_index(format) * 6 + unsigned(lines) * 3 + mode];
	if (!cached)
	{
		if (!m_library) m_library = m_compiler.library(overlay_source, @"RPCS3 native overlays");
		NSString* names[] = {@"overlay_solid", @"overlay_rgba", @"overlay_glyph"};
		const auto descriptor = [MTL4RenderPipelineDescriptor new];
		descriptor.vertexFunctionDescriptor = library_function(m_library, @"overlay_vertex");
		descriptor.fragmentFunctionDescriptor = library_function(m_library, names[mode]);
		descriptor.inputPrimitiveTopology = lines ? MTLPrimitiveTopologyClassLine : MTLPrimitiveTopologyClassTriangle;
		descriptor.rasterizationEnabled = YES;
		descriptor.rasterSampleCount = 1;
		descriptor.colorAttachmentMappingState = MTL4LogicalToPhysicalColorAttachmentMappingStateIdentity;
		const auto color = descriptor.colorAttachments[0];
		color.pixelFormat = format;
		color.blendingState = MTL4BlendStateEnabled;
		color.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
		color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
		color.sourceAlphaBlendFactor = MTLBlendFactorOne;
		color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
		color.rgbBlendOperation = color.alphaBlendOperation = MTLBlendOperationAdd;
		color.writeMask = MTLColorWriteMaskAll;
		cached = m_compiler.render(descriptor);
	}
	return cached;
}

overlay_renderer::prepared_frame overlay_renderer::prepare(context& ctx, const overlay_frame& frame)
{
	prepared_frame prepared;
	if (frame.draws.empty()) return prepared;
	if (frame.draws.size() > overlay_frame::max_commands || frame.images.size() > frame.draws.size())
		throw std::length_error("Native overlay command budget exceeded");
	const auto limit = max_2d_texture_dimension(ctx.device());
	std::size_t image_bytes = 0;
	for (const auto& image : frame.images)
	{
		if (!image.width || !image.height || !image.layers || image.width > limit || image.height > limit ||
			image.layers > 32 || (!image.glyphs && image.layers != 1))
			throw std::invalid_argument("Invalid native overlay image dimensions");
		const auto bytes = std::size_t(image.width) * image.height * image.layers * (image.glyphs ? 1 : 4);
		if (bytes != image.bytes.size() || bytes > overlay_frame::max_image_bytes - image_bytes)
			throw std::length_error("Native overlay image budget or payload is invalid");
		image_bytes += bytes;
	}
	std::vector<std::array<float, 4>> vertices;
	for (const auto& draw : frame.draws)
	{
		const auto count = draw.vertices.size();
		if (!count) continue;
		if (!draw.virtual_extent[0] || !draw.virtual_extent[1] || !finite(draw.color) ||
			(draw.clip && !finite(*draw.clip)) || (draw.image && (*draw.image >= frame.images.size() || draw.backbuffer)) ||
			(draw.distance_field && !draw.distance_field->valid()) || (draw.pulse_phase && !std::isfinite(*draw.pulse_phase)) ||
			!std::isfinite(draw.blur) || draw.blur < 0 || draw.blur > 1 ||
			!std::all_of(draw.vertices.begin(), draw.vertices.end(), finite<4>))
			throw std::invalid_argument("Invalid native overlay draw state");
		const auto first = vertices.size();
		if (count > overlay_frame::max_vertices - first)
			throw std::length_error("Native overlay vertex budget exceeded");
		MTLPrimitiveType primitive;
		switch (draw.primitive)
		{
		case overlay_primitive::quads:
		case overlay_primitive::triangle_fan:
		{
			const bool quads = draw.primitive == overlay_primitive::quads;
			if ((quads && count % 4) || (!quads && count < 3)) throw std::invalid_argument("Incomplete native overlay primitive");
			const auto expanded = quads ? count / 4 * 6 : (count - 2) * 3;
			if (expanded > overlay_frame::max_vertices - first) throw std::length_error("Expanded overlay vertex budget exceeded");
			if (quads)
				for (std::size_t i = 0; i < count; i += 4)
					for (unsigned lane : {0u, 1u, 2u, 2u, 1u, 3u}) vertices.push_back(draw.vertices[i + lane]);
			else
				for (std::size_t i = 1; i + 1 < count; ++i)
					vertices.insert(vertices.end(), {draw.vertices[0], draw.vertices[i], draw.vertices[i + 1]});
			primitive = MTLPrimitiveTypeTriangle;
			break;
		}
		case overlay_primitive::triangle_strip: primitive = MTLPrimitiveTypeTriangleStrip; break;
		case overlay_primitive::lines: primitive = MTLPrimitiveTypeLine; break;
		case overlay_primitive::line_strip: primitive = MTLPrimitiveTypeLineStrip; break;
		default: throw std::invalid_argument("Invalid native overlay primitive");
		}
		if (vertices.size() == first) vertices.insert(vertices.end(), draw.vertices.begin(), draw.vertices.end());
		prepared.draws.push_back({primitive, first, vertices.size() - first, &draw});
		prepared.needs_backbuffer |= draw.backbuffer;
	}
	if (vertices.empty()) return prepared;
	prepared.vertices = ctx.upload(std::as_bytes(std::span(vertices)), 16);
	prepared.images.reserve(frame.images.size());
	for (const auto& image : frame.images)
		prepared.images.push_back(m_images->get(ctx, image));
	return prepared;
}

void overlay_renderer::encode(context& ctx, const prepared_frame& frame, id<MTLTexture> target, MTLScissorRect content)
{
	if (frame.draws.empty()) return;
	(void)format_index(target.pixelFormat);
	const MTLScissorRect window{0, 0, target.width, target.height};
	if (!m_sampler && (!frame.images.empty() || frame.backbuffer))
	{
		const auto descriptor = [MTLSamplerDescriptor new];
		descriptor.minFilter = descriptor.magFilter = MTLSamplerMinMagFilterLinear;
		descriptor.sAddressMode = descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
		m_sampler = make_sampler(ctx, descriptor);
	}
	const auto pass = [MTL4RenderPassDescriptor new];
	pass.colorAttachments[0].texture = target;
	pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
	pass.colorAttachments[0].storeAction = MTLStoreActionStore;
	render_commands commands(ctx, pass);
	for (const auto& draw : frame.draws)
	{
		const auto& state = *draw.command;
		if (state.backbuffer && !frame.backbuffer) continue;
		const auto& rect = state.window_space ? window : content;
		if (!rect.width || !rect.height) continue;
		const auto image = state.backbuffer ? frame.backbuffer : state.image ? frame.images[*state.image] : nil;
		const unsigned mode = !image ? 0 : image.textureType == MTLTextureType2DArray ? 2 : 1;
		const bool lines = draw.primitive == MTLPrimitiveTypeLine || draw.primitive == MTLPrimitiveTypeLineStrip;
		const auto pso = pipeline(target.pixelFormat, lines, mode);
		const std::array virtual_size{float(state.virtual_extent[0]), float(state.virtual_extent[1])};
		overlay_parameters p{};
		p.color = state.color;
		p.virtual_size = virtual_size;
		p.pixel_size = {float(rect.width), float(rect.height)};
		const float scale_x = p.pixel_size[0] / virtual_size[0], scale_y = p.pixel_size[1] / virtual_size[1];
		const auto clip = state.clip.value_or(std::array{0.f, 0.f, virtual_size[0], virtual_size[1]});
		const auto [left, right] = std::minmax(clip[0], clip[2]);
		const auto [top, bottom] = std::minmax(clip[1], clip[3]);
		p.clip = {std::fma(left, scale_x, float(rect.x)), std::fma(top, scale_y, float(rect.y)),
			std::fma(right, scale_x, float(rect.x)), std::fma(bottom, scale_y, float(rect.y))};
		p.snap = state.snap;
		p.blur = state.blur;
		if (state.pulse_phase)
			p.color[3] *= (std::sin(*state.pulse_phase) + 1.f) * .5f;
		if (const auto& sdf = state.distance_field)
		{
			const float average = (scale_x + scale_y) * .5f;
			const float hx = sdf->half_size[0] * scale_x, hy = sdf->half_size[1] * scale_y;
			p.shape = static_cast<std::uint32_t>(sdf->shape);
			p.center = {float(rect.x) + sdf->center[0] * scale_x, float(rect.y) + sdf->center[1] * scale_y};
			p.border_color = sdf->border_color;
			p.distance_field = {hx, hy, std::min({sdf->radius * average, hx, hy}), sdf->border_width * average};
		}
		const auto upload = ctx.upload(std::as_bytes(std::span{&p, 1}));
		bindings vertex, fragment;
		vertex.set_buffer(0, frame.vertices.buffer, frame.vertices.offset);
		vertex.set_buffer(1, upload.buffer, upload.offset);
		fragment.set_buffer(1, upload.buffer, upload.offset);
		if (image)
		{
			fragment.set_texture(0, image);
			fragment.set_sampler(0, m_sampler);
		}
		raster_state raster;
		raster.viewport = {double(rect.x), double(rect.y), double(rect.width), double(rect.height), 0, 1};
		raster.scissor = rect;
		commands.configure(pso, nil, raster, vertex, fragment);
		commands.draw(draw.primitive, draw.first, draw.count);
	}
	commands.end();
}
}
