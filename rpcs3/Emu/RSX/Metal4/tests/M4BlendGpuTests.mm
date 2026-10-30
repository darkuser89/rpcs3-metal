#include "M4FragmentShader.h"
#include "M4VertexShader.h"
#include "M4VertexFetch.h"
#include "M4SurfaceColor.h"
#include "M4SampleTransfer.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>

namespace
{
using namespace metal4;
using color = std::array<float, 4>;
constexpr color source{0.25f, 0.5f, 0.75f, 0.375f};
constexpr std::array<unsigned, 15> factors{0, 1, 0x300, 0x301, 0x302, 0x303, 0x306, 0x307, 0x304, 0x305, 0x308, 0x8001, 0x8002, 0x8003, 0x8004};
constexpr std::array<unsigned, 8> equations{0x8006, 0x800a, 0x800b, 0x8007, 0x8008, 0xf005, 0xf006, 0xf007};
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void append(std::vector<std::byte>& bytes, unsigned word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		bytes.push_back(static_cast<std::byte>(word >> shift));
}
std::vector<std::byte> program(const color& values = source, bool full_exports = true, bool depth = false)
{
	std::vector<std::byte> bytes;
	for (unsigned index = 0; index < 4; ++index)
	{
		if (depth && index == 3)
		{
			for (unsigned word : {(1u << 24) | (1u << 1) | (15u << 9), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
				append(bytes, word);
			for (float value : {0.f, 0.f, 0.5f, 1.f})
				append(bytes, std::bit_cast<unsigned>(value));
		}
		constexpr std::array<unsigned, 4> full{0, 2, 3, 4}, half{0, 4, 6, 8};
		unsigned reg = full_exports ? full[index] : half[index];
		for (unsigned word : {(1u << 24) | (reg << 1) | (15u << 9) | (index == 3 ? 1u : 0u) | (full_exports ? 0u : 128u), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(bytes, word);
		for (float value : values)
			append(bytes, std::bit_cast<unsigned>(value));
	}
	return bytes;
}
unsigned read(std::span<const std::byte> bytes, unsigned width)
{
	unsigned result = 0;
	for (unsigned i = 0; i < width; ++i)
		result = (result << 8) | unsigned(bytes[i]);
	return result;
}
void store(std::span<std::byte> bytes, unsigned value, unsigned width)
{
	for (unsigned i = 0; i < width; ++i)
		bytes[i] = static_cast<std::byte>(value >> ((width - 1 - i) * 8));
}
bool floating(surface_color format)
{
	return format == surface_color::rgba16f || format == surface_color::rgba32f || format == surface_color::r32f;
}
struct fields
{
	std::array<unsigned, 4> shifts, widths;
};
fields layout(surface_color format)
{
	switch (format)
	{
	case surface_color::x1rgb5_zero:
	case surface_color::x1rgb5_one:
		return {{10, 5, 0, 15}, {5, 5, 5, 1}};
	case surface_color::rgb565:
		return {{11, 5, 0, 0}, {5, 6, 5, 0}};
	case surface_color::b8:
		return {{0, 0, 0, 0}, {0, 0, 8, 0}};
	case surface_color::gb8:
		return {{0, 8, 0, 0}, {0, 8, 8, 0}};
	case surface_color::xbgr8_zero:
	case surface_color::xbgr8_one:
	case surface_color::abgr8:
		return {{0, 8, 16, 24}, {8, 8, 8, 8}};
	default:
		return {{16, 8, 0, 24}, {8, 8, 8, 8}};
	}
}
// Independent guest word model: format channel layout, not native MSL remaps.
color destination(std::span<const std::byte> bytes, surface_color format)
{
	color result{0, 0, 0, 1};
	if (floating(format))
	{
		unsigned width = format == surface_color::rgba16f ? 2 : 4, channels = format == surface_color::r32f ? 1 : 4;
		for (unsigned c = 0; c < channels; ++c)
		{
			unsigned value = read(bytes.subspan(c * width), width);
			result[c] = width == 2 ? float(std::bit_cast<_Float16>(static_cast<std::uint16_t>(value))) : std::bit_cast<float>(value);
		}
		return result;
	}
	const auto fields = layout(format);
	const auto word = read(bytes, color_layout(format).pixel_bytes);
	for (unsigned c = 0; c < 4; ++c)
		if (fields.widths[c])
		{
			unsigned maximum = (1u << fields.widths[c]) - 1;
			result[c] = float((word >> fields.shifts[c]) & maximum) / float(maximum);
		}
	if (format == surface_color::b8)
		result.fill(result[2]);
	if (format == surface_color::gb8)
		result = {result[2], result[1], result[2], result[1]};
	if (format == surface_color::x1rgb5_zero || format == surface_color::x1rgb5_one || format == surface_color::xrgb8_zero || format == surface_color::xrgb8_one || format == surface_color::xbgr8_zero || format == surface_color::xbgr8_one)
		result[3] = 1;
	return result;
}
void encode(std::span<std::byte> bytes, surface_color format, color value, unsigned mask)
{
	if (floating(format))
	{
		unsigned width = format == surface_color::rgba16f ? 2 : 4, channels = format == surface_color::r32f ? 1 : 4;
		for (unsigned c = 0; c < channels; ++c)
			if (mask & (1u << c))
				store(bytes.subspan(c * width), width == 2 ? std::bit_cast<std::uint16_t>(static_cast<_Float16>(value[c])) : std::bit_cast<unsigned>(value[c]), width);
		return;
	}
	if (format == surface_color::x1rgb5_zero || format == surface_color::xrgb8_zero || format == surface_color::xbgr8_zero)
		value[3] = 0;
	if (format == surface_color::x1rgb5_one || format == surface_color::xrgb8_one || format == surface_color::xbgr8_one)
		value[3] = 1;
	const auto fields = layout(format);
	const unsigned width = color_layout(format).pixel_bytes;
	unsigned word = read(bytes, width);
	for (unsigned c = 0; c < 4; ++c)
		if ((mask & (1u << c)) && fields.widths[c])
		{
			unsigned maximum = (1u << fields.widths[c]) - 1;
			unsigned component = static_cast<unsigned>(std::nearbyint(std::clamp(value[c], 0.f, 1.f) * float(maximum)));
			word = (word & ~(maximum << fields.shifts[c])) | (component << fields.shifts[c]);
		}
	store(bytes, word, width);
}
float factor(unsigned operation, const color& s, const color& d, const color& k, unsigned channel)
{
	switch (operation)
	{
	case 0:
		return 0;
	case 1:
		return 1;
	case 0x300:
		return s[channel];
	case 0x301:
		return 1 - s[channel];
	case 0x302:
		return s[3];
	case 0x303:
		return 1 - s[3];
	case 0x306:
		return d[channel];
	case 0x307:
		return 1 - d[channel];
	case 0x304:
		return d[3];
	case 0x305:
		return 1 - d[3];
	case 0x308:
		return channel == 3 ? 1 : std::min(s[3], 1 - d[3]);
	case 0x8001:
		return k[channel];
	case 0x8002:
		return 1 - k[channel];
	case 0x8003:
		return k[3];
	case 0x8004:
		return 1 - k[3];
	default:
		throw std::runtime_error("Invalid independent blend fixture");
	}
}
color blend(const guest_blend_state& state, const color& s, const color& d, const color& k, bool normalized)
{
	color result;
	const auto saturation = [&](float v)
	{
		return normalized ? std::clamp(v, 0.f, 1.f) : v;
	};
	const auto signed_source = [&](float v)
	{
		if (!normalized)
			return v;
		unsigned byte = unsigned(std::floor(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f));
		return float(byte < 128 ? int(byte) : int(byte) - 256) / 255.f;
	};
	for (unsigned c = 0; c < 4; ++c)
	{
		unsigned op = c == 3 ? state.operation_alpha : state.operation_rgb;
		float a = s[c], b = d[c];
		if (op == 0x8007)
			result[c] = std::min(saturation(a), b);
		else if (op == 0x8008)
			result[c] = std::max(saturation(a), b);
		else if (op == 0xf005)
			result[c] = b - signed_source(a);
		else if (op == 0xf006)
			result[c] = signed_source(a) + b;
		else if (op == 0xf007)
			result[c] = saturation(a) + signed_source(b);
		else
		{
			a = saturation(a * factor(c == 3 ? state.source_alpha : state.source_rgb, s, d, k, c));
			b *= factor(c == 3 ? state.destination_alpha : state.destination_rgb, s, d, k, c);
			result[c] = op == 0x8006 ? a + b : op == 0x800a ? a - b
															: b - a;
		}
	}
	return result;
}

float round8(float value, bool half_precision)
{
	float truncated = std::bit_cast<float>(std::bit_cast<unsigned>(value) & 0xfffff000u);
	unsigned raw = unsigned(std::max(0.f, std::floor(std::fma(truncated, 255.f, 0.5f))));
	return half_precision ? float(static_cast<_Float16>(static_cast<_Float16>(raw) / static_cast<_Float16>(255))) : float(raw) / 255.f;
}
color preprocess(color value, output_rop_configuration flags, bool full_exports)
{
	if (!full_exports)
		for (auto& component : value)
			component = float(static_cast<_Float16>(component));
	if (flags.srgb)
		for (unsigned c = 0; c < 3; ++c)
		{
			float v = value[c];
			value[c] = std::clamp(v < 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f, 0.f, 1.f);
			if (!full_exports)
				value[c] = float(static_cast<_Float16>(value[c]));
		}
	if (flags.round_to_8bit)
		for (auto& component : value)
			component = round8(component, !full_exports);
	return value;
}
class fixture
{
	context m_ctx{1};
	compiler m_compiler{m_ctx};
	vertex_fetch_engine m_fetch{m_ctx};
	surface_color_transfer m_colors{m_ctx};
	sample_transfer m_samples{m_ctx};
	std::unique_ptr<vertex_shader> m_vertex;
	id<MTLBuffer> m_vertices;
	converted_geometry m_geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);

  public:
	std::size_t cases = 0;
	unsigned srgb_max_ulp = 0;
	unsigned native_targets = 0, shader_targets = 0;
	unsigned native_half_ulp = 0, native_float_ulp = 0;
	fixture()
	{
		constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
		const std::array<std::array<unsigned, 4>, 1> code{{{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
		m_vertex = std::make_unique<vertex_shader>(m_compiler, code);
		std::vector<std::byte> bytes;
		for (float value : {-1.f, -1.f, 0.5f, 1.f, 3.f, -1.f, 0.5f, 1.f, -1.f, 3.f, 0.5f, 1.f})
			for (unsigned shift : {24u, 16u, 8u, 0u})
				bytes.push_back(static_cast<std::byte>(std::bit_cast<unsigned>(value) >> shift));
		m_vertices = make_upload(m_ctx, bytes);
	}
	void run(std::span<const surface_color> formats, std::span<const guest_blend_state> states, unsigned count, unsigned first_slot, const color& source_values = source, output_rop_configuration output_flags = {}, bool full_exports = true, bool alpha_test = false, std::optional<color> blend_constants = {})
	{
		render_pipeline_configuration config;
		config.state.sample_count = count;
		config.state.output_rop = output_flags;
		fragment_program_options options;
		options.float32_exports = full_exports;
		for (unsigned i = 0; i < formats.size(); ++i)
		{
			unsigned slot = first_slot + i;
			config.state.colors[slot] = states[i];
			configure_surface_color(config, options, slot, formats[i], 15);
		}
#ifdef METAL4_TEST_NATIVE_BLENDS
		configure_blending(config, options);
		for (unsigned i = 0; i < formats.size(); ++i)
		{
			const unsigned slot = first_slot + i;
			native_targets += bool(config.native_blend_mask & (1u << slot));
			shader_targets += config.shader_blend.colors[slot].enabled;
		}
#else
		configure_shader_blending(config, options);
#endif
		configure_output_rops(config, options);
		fragment_shader fragment(m_compiler, program(source, full_exports), options);
		const auto current_code = program(source_values, full_exports);
		bool rejected = false;
		try
		{
			(void)make_render_pipeline(m_compiler, config, m_vertex->function(), fragment.function());
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected == (config.native_blend_mask || config.shader_blend.enabled() || output_flags.enabled() || config.shader_half_storage_mask || count > 1), "Raw descriptor verified fragment contract mismatch");
		auto mismatch = config;
		mismatch.shader_blend.sample_count = config.shader_blend.sample_count == 1 ? 2 : 1;
		rejected = false;
		try
		{
			(void)make_render_pipeline(m_compiler, mismatch, m_vertex->function(), fragment);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Mismatched compiled blend sample contract was accepted");
		for (unsigned mask : {5u, 10u, 15u})
		{
			struct attachment
			{
				id<MTLTexture> expanded, target;
				id<MTLBuffer> input, output;
				std::vector<std::byte> original;
			};
			std::vector<attachment> targets;
			const unsigned sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1;
			const unsigned native_width = output_flags.polygon_stipple ? 48 : 4, native_height = output_flags.polygon_stipple ? 40 : 4;
			const unsigned width = native_width * sx, height = native_height * sy;
			for (unsigned i = 0; i < formats.size(); ++i)
			{
				unsigned slot = first_slot + i;
				configure_surface_color(config, options, slot, formats[i], mask);
				const auto info = describe_surface_color(formats[i]);
				const unsigned bytes = info.layout.pixel_bytes;
				attachment a;
				a.original.resize(width * height * bytes);
				for (unsigned p = 0; p < width * height; ++p)
					encode(std::span(a.original).subspan(p * bytes), formats[i], {0.375f, 0.625f, 0.25f + float(p % 3) * 0.125f, 0.25f}, 15);
				a.input = make_upload(m_ctx, a.original);
				a.output = make_buffer(m_ctx, a.original.size(), buffer_storage::shared);
				auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:info.format width:width height:height mipmapped:NO];
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				a.expanded = make_texture(m_ctx, descriptor);
				descriptor.width = native_width;
				descriptor.height = native_height;
				descriptor.sampleCount = count;
				if (count > 1)
					descriptor.textureType = MTLTextureType2DMultisample;
				a.target = count == 1 ? a.expanded : make_texture(m_ctx, descriptor);
				targets.push_back(std::move(a));
			}
			configure_output_rops(config, options);
			const auto pipeline = make_render_pipeline(m_compiler, config, m_vertex->function(), fragment);
			fragment_context fc;
			fc.blend_color = blend_constants.value_or(mask == 10 ? color{0.125f, 0.75f, 0.375f, 0.625f} : color{0.625f, 0.25f, 0.5f, 0.125f});
			for (unsigned row = 0; row < 32; ++row)
				fc.stipple_pattern[row] = std::rotl(mask == 10 ? 0x9a572461u : 0x2d496b93u, row);
			const auto effective_source = preprocess(source_values, output_flags, full_exports);
			fc.alpha_test = alpha_test;
			fc.alpha_function = 2;
			fc.alpha_reference = full_exports ? 128.f / 255.f : float(static_cast<_Float16>(128.f / 255.f));
			m_ctx.begin();
			texture_region whole;
			whole.size = MTLSizeMake(width, height, 1);
			auto pass = [MTL4RenderPassDescriptor new];
			for (unsigned i = 0; i < targets.size(); ++i)
			{
				auto& a = targets[i];
				m_colors.upload(m_ctx, formats[i], a.input, {}, a.expanded, whole);
				if (count > 1)
					m_samples.import(m_ctx, a.expanded, a.target);
				pass.colorAttachments[first_slot + i].texture = a.target;
				pass.colorAttachments[first_slot + i].loadAction = MTLLoadActionLoad;
				pass.colorAttachments[first_slot + i].storeAction = MTLStoreActionStore;
			}
			vertex_fetch_request request;
			request.vertex_count = 3;
			request.attribute_mask = 1;
			request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
			vertex_inputs v;
			v.attributes = m_fetch.decode(m_ctx, request, m_vertices, nil);
			vertex_context vc;
			vc.vertex_count = 3;
			const auto vb = m_vertex->prepare(m_ctx, vc, m_geometry, v), fb = fragment.prepare(m_ctx, fc, current_code, {});
			render_commands draw(m_ctx, pass);
			raster_state raster;
			raster.blend_color = fc.blend_color;
			raster.viewport = {0, 0, 4, 4, 0, 1};
			raster.scissor = {1, 1, 2, 2};
			if (output_flags.polygon_stipple)
			{
				raster.viewport = {3, 2, 45, 38, 0, 1};
				raster.scissor = {5, 3, 38, 34};
			}
			draw.draw_geometry(m_geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb, 2);
			draw.end();
			for (unsigned i = 0; i < targets.size(); ++i)
			{
				auto& a = targets[i];
				if (count > 1)
					m_samples.expand(m_ctx, a.target, a.expanded);
				m_colors.download(m_ctx, formats[i], a.expanded, whole, a.output, {});
			}
			m_ctx.submit().wait();
			for (unsigned i = 0; i < targets.size(); ++i)
			{
				auto& a = targets[i];
				auto expected = a.original;
				unsigned bytes = color_layout(formats[i]).pixel_bytes;
				for (unsigned y = raster.scissor.y * sy; y < (raster.scissor.y + raster.scissor.height) * sy; ++y)
					for (unsigned x = raster.scissor.x * sx; x < (raster.scissor.x + raster.scissor.width) * sx; ++x)
					{
						if (output_flags.polygon_stipple && !(fc.stipple_pattern[(y / sy) % 32] & (1u << ((x / sx) % 32))))
							continue;
						if (alpha_test && effective_source[3] != fc.alpha_reference)
							continue;
						auto pixel = std::span(expected).subspan((y * width + x) * bytes, bytes);
						for (unsigned instance = 0; instance < 2; ++instance)
						{
							auto value = effective_source;
							if (states[i].enabled)
							{
								value = blend(states[i], value, destination(pixel, formats[i]), fc.blend_color, !floating(formats[i]));
								if (output_flags.round_to_8bit)
									for (auto& component : value)
										component = round8(component, false);
								if (!full_exports)
									for (auto& component : value)
										component = float(static_cast<_Float16>(component));
							}
							encode(pixel, formats[i], value, mask);
						}
					}
				auto actual = map_buffer(a.output, 0, a.output.length);
				bool matches = std::equal(actual.begin(), actual.end(), expected.begin());
#ifdef METAL4_TEST_NATIVE_BLENDS
				// Native fixed-function arithmetic may fuse products before the
				// attachment conversion. Only written floating lanes permit a
				// bounded arithmetic difference; masks/padding and nonfinite bits
				// remain exact. The unchanged programmable suite is byte-exact.
				if (!matches && floating(formats[i]) && (config.native_blend_mask & (1u << (first_slot + i))))
				{
					matches = true;
					const unsigned word_bytes = formats[i] == surface_color::rgba16f ? 2 : 4;
					for (unsigned offset = 0; offset < actual.size(); offset += word_bytes)
					{
						const auto value = read(actual.subspan(offset), word_bytes), reference = read(std::span(expected).subspan(offset), word_bytes);
						if (value == reference) continue;
						const auto channel = (offset % bytes) / word_bytes, pixel = offset / bytes;
						const auto x = (pixel % width) / sx, y = (pixel / width) / sy;
						const auto distance = value > reference ? value - reference : reference - value;
						const auto exponent_mask = word_bytes == 2 ? 0x7c00u : 0x7f800000u;
						const bool finite = (value & exponent_mask) != exponent_mask && (reference & exponent_mask) != exponent_mask;
						const bool covered = x >= raster.scissor.x && x < raster.scissor.x + raster.scissor.width && y >= raster.scissor.y && y < raster.scissor.y + raster.scissor.height;
						if (!covered || !(mask & (1u << channel)) || !finite || (value >> (word_bytes * 8 - 1)) != (reference >> (word_bytes * 8 - 1)) || distance > (word_bytes == 2 ? 0u : 4u))
							matches = false;
						else if (word_bytes == 2) native_half_ulp = std::max(native_half_ulp, distance);
						else native_float_ulp = std::max(native_float_ulp, distance);
					}
				}
#endif
				// pow is a transcendental operation, not an IEEE basic operation.
				// Only changed, written Float32 RGB may vary by <=4 ULP for sRGB.
				// Quantized/half output, alpha, masks and untouched storage stay exact.
				if (!matches && output_flags.srgb && !output_flags.round_to_8bit && (formats[i] == surface_color::rgba32f || formats[i] == surface_color::r32f))
				{
					matches = true;
					for (unsigned offset = 0; offset < actual.size(); offset += 4)
					{
						unsigned value = read(actual.subspan(offset), 4), reference = read(std::span(expected).subspan(offset), 4);
						if (value == reference)
							continue;
						unsigned channel = (offset % bytes) / 4, original = read(std::span(a.original).subspan(offset), 4);
						unsigned distance = value > reference ? value - reference : reference - value;
						if (channel >= 3 || !(mask & (1u << channel)) || reference == original || (value >> 31) != (reference >> 31) || distance > 4)
							matches = false;
						else
							srgb_max_ulp = std::max(srgb_max_ulp, distance);
					}
				}
				if (!matches)
				{
					if (floating(formats[i]))
					{
						const unsigned stride = formats[i] == surface_color::rgba16f ? 2 : 4;
						unsigned maximum_distance = 0;
						for (unsigned offset = 0; offset < actual.size(); offset += stride)
						{
							const auto a = read(actual.subspan(offset), stride), b = read(std::span(expected).subspan(offset), stride);
							maximum_distance = std::max(maximum_distance, a > b ? a - b : b - a);
						}
						std::cerr << "Floating difference max word distance=" << maximum_distance << " source=" << source_values[0] << ',' << source_values[1] << ',' << source_values[2] << ',' << source_values[3] << " constant=" << fc.blend_color[0] << ',' << fc.blend_color[1] << ',' << fc.blend_color[2] << ',' << fc.blend_color[3] << '\n';
					}
					auto diff = std::mismatch(actual.begin(), actual.end(), expected.begin());
					std::cerr << "Blend format=" << unsigned(formats[i]) << " samples=" << count << " slot=" << first_slot + i << " full_exports=" << full_exports << " rounding=" << output_flags.round_to_8bit << " srgb=" << output_flags.srgb << " stipple=" << output_flags.polygon_stipple << " op=" << std::hex << states[i].operation_rgb << " source=" << states[i].source_rgb << " destination=" << states[i].destination_rgb << std::dec << " mask=" << mask << " byte=" << diff.first - actual.begin() << " actual=" << unsigned(*diff.first) << " expected=" << unsigned(*diff.second) << '\n';
					throw std::runtime_error("Native programmable blend differs from independent guest reference");
				}
			}
			++cases;
		}
	}
	void selection_contracts()
	{
		render_pipeline_configuration config; fragment_program_options options;
		const std::array formats{surface_color::argb8, surface_color::rgba32f, surface_color::rgba16f, surface_color::xrgb8_zero};
		for (unsigned i = 0; i < 4; ++i)
		{
			auto& color = config.state.colors[i]; color.enabled = true;
			color.source_rgb = color.source_alpha = color.destination_rgb = color.destination_alpha = 1;
			configure_surface_color(config, options, i, formats[i], 15);
		}
		config.state.colors[1].operation_rgb = 0xf006;
		config.state.colors[2].operation_rgb = config.state.colors[2].operation_alpha = 0x8007;
		configure_blending(config, options);
		require(config.native_blend_mask == 5 && !config.shader_blend.colors[0].enabled && config.shader_blend.colors[1].enabled &&
			!config.shader_blend.colors[2].enabled && config.shader_blend.colors[3].enabled && config.shader_half_storage_mask == 4,
			"Mixed MRT selection lost signed/remap/half-rounding semantics");
		const auto source = generate_fragment_program(program(), options).source;
		for (unsigned i = 0; i < 4; ++i)
			require((source.find("rsx_destination" + std::to_string(i)) != std::string::npos) == bool(i & 1), "Selected native target retained framebuffer fetch or shader target lost it");
		fragment_shader fragment(m_compiler, program(), options);
		const auto descriptor = make_render_descriptor(config, m_vertex->function(), fragment.function());
		for (unsigned i = 0; i < 4; ++i)
			require(descriptor.colorAttachments[i].blendingState == ((i & 1) ? MTL4BlendStateDisabled : MTL4BlendStateEnabled), "Native MRT descriptor selection differs");
		unsigned refusals = 0;
		for (unsigned failure = 0; failure < 6; ++failure)
		{
			auto bad = config;
			if (failure == 0) bad.native_blend_mask |= 2;
			if (failure == 1) bad.native_blend_mask |= 8;
			if (failure == 2) bad.native_blend_mask |= 16;
			if (failure == 3) bad.state.output_rop.round_to_8bit = true;
			if (failure == 4) bad.colors[0] = MTLPixelFormatB5G6R5Unorm;
			if (failure == 5) bad.state.logic_enabled = true;
			try { (void)make_render_descriptor(bad, m_vertex->function(), fragment.function()); }
			catch (const std::invalid_argument&) { ++refusals; }
		}
		auto half_options = options; half_options.float32_exports = false;
		fragment_shader half(m_compiler, program(::source, false), half_options);
		try { (void)make_render_pipeline(m_compiler, config, m_vertex->function(), half); }
		catch (const std::invalid_argument&) { ++refusals; }
		require(refusals == 7, "Invalid native selection or half-export mismatch accepted");
		auto half_config = config; configure_blending(half_config, half_options);
		require(half_config.native_blend_mask == 4 && half_config.shader_blend.colors[0].enabled && !half_config.shader_blend.colors[2].enabled, "Half-register arithmetic rounding or exact min/max selection lost");
		auto rounded = config; rounded.state.output_rop.round_to_8bit = true; configure_blending(rounded, options);
		require(!rounded.native_blend_mask, "Post-blend RSX rounding entered native blending");
		config.state.logic_enabled = true; configure_logic_operations(config, options);
		require(!config.native_blend_mask && !config.shader_blend.enabled() && config.shader_logic.enabled && config.shader_half_storage_mask == 4,
			"Logic transition retained old native blend or lost storage rounding");
		std::cout << "Native blend selection:independent MRT descriptors/generated fetch declarations,signed/remap/half/rounding/logic gates and seven forged-contract refusals passed\n";
	}
	void depth_only(bool full_exports, bool round_output, bool stipple)
	{
		render_pipeline_configuration config;
		config.state.depth_test = true;
		config.state.depth_write = true;
		config.state.depth_compare = 0x207;
		config.state.output_rop = {round_output, true, stipple};
		fragment_program_options options;
		options.color_count = 0;
		options.depth_export = true;
		options.float32_exports = full_exports;
		configure_output_rops(config, options);
		const auto code = program({0.25f, 0.5f, 0.75f, 0.5f}, full_exports, true);
		fragment_shader fragment(m_compiler, code, options);
		const auto pipeline = make_render_pipeline(m_compiler, config, m_vertex->function(), fragment);
		fragment_context fc;
		fc.alpha_test = 1;
		fc.alpha_function = 2;
		fc.alpha_reference = 128.f / 255.f;
		for (unsigned row = 0; row < 32; ++row)
			fc.stipple_pattern[row] = std::rotl(0x24935a6du, row);
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:48 height:40 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		const auto target = make_texture(m_ctx, descriptor);
		const auto output = make_buffer(m_ctx, 40 * 256, buffer_storage::shared);
		m_ctx.begin();
		fill_buffer(m_ctx, output, 0, output.length, 0xa7);
		vertex_fetch_request request;
		request.vertex_count = 3;
		request.attribute_mask = 1;
		request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
		vertex_inputs inputs;
		inputs.attributes = m_fetch.decode(m_ctx, request, m_vertices, nil);
		vertex_context vc;
		vc.vertex_count = 3;
		const auto vb = m_vertex->prepare(m_ctx, vc, m_geometry, inputs), fb = fragment.prepare(m_ctx, fc, code, {});
		auto pass = [MTL4RenderPassDescriptor new];
		pass.depthAttachment.texture = target;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.clearDepth = 0.125;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		render_commands draw(m_ctx, pass);
		raster_state raster;
		raster.viewport = {3, 2, 45, 38, 0, 1};
		raster.scissor = {5, 3, 38, 34};
		draw.draw_geometry(m_geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb);
		draw.end();
		texture_region region;
		region.size = MTLSizeMake(48, 40, 1);
		texture_to_buffer(m_ctx, target, region, output, 0, 256);
		m_ctx.submit().wait();
		const auto actual = map_buffer(output, 0, output.length);
		for (unsigned y = 0; y < 40; ++y)
			for (unsigned x = 0; x < 64; ++x)
			{
				unsigned expected = 0xa7a7a7a7;
				if (x < 48)
				{
					bool covered = x >= 5 && x < 43 && y >= 3 && y < 37;
					bool patterned = !stipple || (fc.stipple_pattern[y % 32] & (1u << (x % 32)));
					expected = std::bit_cast<unsigned>(covered && patterned && round_output ? 0.5f : 0.125f);
				}
				unsigned word;
				std::memcpy(&word, actual.data() + y * 256 + x * 4, 4);
				require(word == expected, "Colorless ROP alpha/stipple changed unexpected depth or row padding");
			}
		++cases;
	}
};
} // namespace
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			fixture test;
#ifdef METAL4_TEST_NATIVE_BLENDS
			test.selection_contracts();
			for (auto format : {surface_color::argb8, surface_color::abgr8, surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f})
			{
				for (unsigned op = 0; op < 5; ++op) for (unsigned f = 0; f < 15; ++f)
				{
					guest_blend_state state; state.enabled = true;
					state.source_rgb = factors[f]; state.destination_rgb = factors[(f + 7) % 15];
					state.source_alpha = factors[(f * 7 + 3) % 15]; state.destination_alpha = factors[(f * 11 + 5) % 15];
					state.operation_rgb = equations[op]; state.operation_alpha = equations[(op + 3) % 5];
					const std::array formats{format}; const std::array states{state};
					for (const auto values : {source, color{-.25f, 1.25f, .5f, 1.5f}, color{1.5f, -.5f, 2.f, -.25f}})
						test.run(formats, states, 1, op % 4, values);
				}
				std::cout << "Automatic blend format " << unsigned(format) << " passed\n";
			}
			for (auto format : {surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f})
				for (unsigned f = 11; f < 15; ++f)
				{
					guest_blend_state state; state.enabled = true;
					state.source_rgb = state.source_alpha = factors[f]; state.destination_rgb = state.destination_alpha = 1;
					const std::array formats{format}; const std::array states{state};
					test.run(formats, states, 1, 0, source, {}, true, false, color{-.25f, 1.25f, 2.f, -.5f});
					test.run(formats, states, 1, 0, {.1234567f, .9876543f, 1.234567f, .333333f});
				}
			for (unsigned count : {1u, 2u, 4u})
			{
				const std::array formats{surface_color::argb8, surface_color::rgba32f, surface_color::rgba16f, surface_color::xrgb8_zero};
				std::array<guest_blend_state, 4> states;
				for (auto& state : states) { state.enabled = true; state.source_rgb = state.source_alpha = 1; state.destination_rgb = state.destination_alpha = 1; }
				states[3].operation_rgb = 0xf006;
				for (auto values : {source, color{-.25f, 1.25f, .5f, 1.5f}})
				{
					test.run(formats, states, count, 0, values);
					test.run(formats, states, count, 0, values, {}, false);
					test.run(formats, states, count, 0, values, {true, false, false});
				}
				test.run(formats, states, count, 0, source, {false, true, true});
				for (auto op : {0x8007u, 0x8008u})
				{
					for (auto& state : states) state.operation_rgb = state.operation_alpha = op;
					test.run(formats, states, count, 0, {-.25f, 1.25f, .5f, 1.5f});
					test.run(formats, states, count, 0, {-.0f, .0f, .37512f, .2501f});
					test.run(formats, states, count, 0, {-.0f, .0f, .37512f, .2501f}, {}, false);
				}
			}
			require(test.native_targets && test.shader_targets && !test.native_half_ulp, "Native/shader selection matrix omitted a path or changed half selection bits");
			{
				const std::array formats{surface_color::rgba16f};
				guest_blend_state state; state.enabled = true;
				state.source_rgb = state.source_alpha = state.destination_rgb = state.destination_alpha = 1;
				state.operation_rgb = state.operation_alpha = 0x800a;
				const std::array states{state};
				test.run(formats, states, 1, 0, {.37512f, .6252f, .25006f, .2501f});
			}
			{
				const std::array formats{surface_color::rgb565};
				guest_blend_state state; state.enabled = true;
				state.source_rgb = 0; state.destination_rgb = 0x307;
				state.source_alpha = 0x301; state.destination_alpha = 0x303; state.operation_alpha = 0x8007;
				const std::array states{state};
				for (unsigned count : {1u, 2u, 4u}) test.run(formats, states, count, 0);
			}
			std::cout << test.cases << " automatic native/shader blend draws passed;selected native targets=" << test.native_targets << " shader targets=" << test.shader_targets << " max half/float ULP=" << test.native_half_ulp << '/' << test.native_float_ulp << '\n';
#elif defined(METAL4_TEST_OUTPUT_ROPS)
			for (unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
				for (unsigned count : {1u, 2u, 4u})
					for (unsigned flags = 0; flags < 8; ++flags)
						for (bool full_exports : {true, false})
							for (unsigned mode = 0; mode < 2; ++mode)
							{
								output_rop_configuration output{bool(flags & 1), bool(flags & 2), bool(flags & 4)};
								guest_blend_state state;
								state.enabled = mode;
								state.source_rgb = factors[4];
								state.destination_rgb = factors[9];
								state.source_alpha = factors[11];
								state.destination_alpha = factors[5];
								state.operation_rgb = equations[6];
								state.operation_alpha = equations[2];
								const std::array formats{static_cast<surface_color>(raw)};
								const std::array states{state};
								test.run(formats, states, count, flags % 4, {0.0031308f, 0.49985f, 1.25f, 0.5f}, output, full_exports);
							}
			for (bool full_exports : {true, false})
			{
				const std::array formats{surface_color::rgba32f};
				const std::array states{guest_blend_state{}};
				for (float alpha : {0.4994f, 0.5f, 0.5006f})
					test.run(formats, states, 4, 0, {-0.25f, 0.0031307f, 0.75f, alpha}, {true, true, true}, full_exports, true);
				for (unsigned bits = 0x3effe000u; bits < 0x3f001000u; bits += 257)
					test.run(formats, states, 1, 0, {std::bit_cast<float>(bits), 0.5f, 1.003f, 0.75f}, {true, false, false}, full_exports);
				for (bool rounding : {false, true})
					for (bool stipple : {false, true})
						test.depth_only(full_exports, rounding, stipple);
			}
			std::cout << test.cases << " native output ROP draws verify 14 guest formats, sRGB/8-bit/stipple combinations, full/half exports, alpha ordering, signed blends, colorless depth and 1/2/4 samples\n";
			std::cout << "Float32 sRGB maximum CPU-reference distance: " << test.srgb_max_ulp << " ULP; other comparisons remain byte exact\n";
#else
			for (unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			{
				for (unsigned count : {1u, 2u, 4u})
					for (unsigned op = 0; op < 8; ++op)
						for (unsigned f = 0; f < (raw == 4 || raw == 9 || raw == 10 || raw == 12 ? 15u : 1u); ++f)
						{
							const std::array formats{static_cast<surface_color>(raw)};
							guest_blend_state state;
							state.enabled = true;
							state.source_rgb = factors[f];
							state.destination_rgb = factors[(f + 7) % 15];
							state.source_alpha = factors[(f * 7 + 3) % 15];
							state.destination_alpha = factors[(f * 11 + 5) % 15];
							state.operation_rgb = equations[op];
							state.operation_alpha = equations[(op + 3) % 8];
							const std::array states{state};
							test.run(formats, states, count, op % 4);
						}
				std::cout << "Guest blend format " << raw << " passed\n";
			}
			for (unsigned count : {1u, 2u, 4u})
			{
				const std::array formats{surface_color::b8, surface_color::rgba32f, surface_color::xrgb8_zero, surface_color::gb8};
				std::array<guest_blend_state, 4> states;
				for (unsigned i = 0; i < 4; ++i)
				{
					states[i].enabled = true;
					states[i].source_rgb = 0x302;
					states[i].destination_rgb = 0x305;
					states[i].source_alpha = 0x8003;
					states[i].destination_alpha = 0x303;
					states[i].operation_rgb = equations[4 + i];
					states[i].operation_alpha = equations[i];
				}
				test.run(formats, states, count, 0);
				test.run(formats, states, count, 0, {-0.25f, 1.25f, 0.5f, 1.f});
				test.run(formats, states, count, 0, {127.f / 255.f, 128.f / 255.f, 1.f / 255.f, 0.5f});
			}
			std::cout << test.cases << " native programmable blend draws verify 14 guest formats, 15 factors, 8 equations, separate RGB/alpha, guest masks, dynamic constants, primitive ordering and mixed four-MRT 1/2/4-sample passes\n";
#endif
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
