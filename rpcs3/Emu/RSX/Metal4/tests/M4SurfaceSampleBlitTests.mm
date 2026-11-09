#include "M4SurfaceSampleBlit.h"
#include "M4HeapBudget.h"

#include <algorithm>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
const std::array<surface_format, 18> formats{surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565,
											 surface_color::xrgb8_zero, surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8,
											 surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero,
											 surface_color::xbgr8_one, surface_color::abgr8, surface_depth::z16_uint, surface_depth::z24s8_uint,
											 surface_depth::z16_float, surface_depth::z24s8_float};
sample_layout grid(unsigned samples)
{
	return {samples == 1 ? 1u : 2u, samples == 4 ? 2u : 1u};
}
std::size_t nearest(std::size_t i, std::size_t source, std::size_t destination, bool flip)
{
	const auto denominator = 2 * destination, value = (2 * i + 1) * source;
	return (flip ? source * denominator - value : value) / denominator;
}
id<MTLTexture> texture(context& ctx, surface_format format, std::size_t width, std::size_t height, unsigned samples)
{
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:surface_native_format(format) width:width height:height mipmapped:NO];
	descriptor.sampleCount = samples;
	if (samples > 1)
		descriptor.textureType = MTLTextureType2DMultisample;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
	return make_texture(ctx, descriptor);
}
class fixture
{
  public:
	context& ctx;
	surface_sample_blitter blitter;
	sample_transfer samples;
	surface_color_transfer colors;
	surface_depth_transfer depths;
	std::size_t bounded_bytes = 0;
	explicit fixture(context& ctx) : ctx(ctx), blitter(ctx), samples(ctx), colors(ctx), depths(ctx) {}
	void upload(surface_format format, id<MTLBuffer> buffer, id<MTLTexture> target, texture_region area)
	{
		if (const auto color = std::get_if<surface_color>(&format))
			colors.upload(ctx, *color, buffer, {}, target, area);
		else
			depths.upload(ctx, std::get<surface_depth>(format), buffer, {}, target, area);
	}
	void download(surface_format format, id<MTLTexture> source, texture_region area, id<MTLBuffer> buffer)
	{
		if (const auto color = std::get_if<surface_color>(&format))
			colors.download(ctx, *color, source, area, buffer, {});
		else
			depths.download(ctx, std::get<surface_depth>(format), source, area, buffer, {});
	}
	void check(surface_format sf, surface_format df, unsigned sc, unsigned dc, unsigned flip, bool typeless,
			   texture_plane plane = texture_plane::all, bool partial = false, bool bounded = false, bool overlap = false, bool gated = false)
	{
		const auto sb = surface_guest_bytes(sf), db = surface_guest_bytes(df);
		const std::size_t sw = bounded ? 512 : partial ? 8 : 16 / sb, sh = bounded ? 256 : partial ? 6 : 3;
		const std::size_t dw = overlap ? sw : bounded ? 514 : partial ? 9 : 16 / db + 2, dh = overlap ? sh : bounded ? 258 : partial ? 7 : 4;
		const auto a = grid(sc), b = grid(dc);
		const auto source = texture(ctx, sf, sw, sh, sc), target = overlap ? source : texture(ctx, df, dw, dh, dc);
		const auto input_image = sc == 1 ? source : texture(ctx, sf, sw * a.x, sh * a.y, 1);
		const auto output_image = overlap ? input_image : dc == 1 ? target : texture(ctx, df, dw * b.x, dh * b.y, 1);
		std::vector<std::byte> original(sw * a.x * sh * a.y * sb), initial(dw * b.x * dh * b.y * db, std::byte{0x44});
		for (std::size_t i = 0; i < original.size(); ++i)
			original[i] = static_cast<std::byte>(0x10 + (i * 7 + i / 11) % 0x50);
		if (bounded && !typeless)
			if (const auto color = std::get_if<surface_color>(&sf); color &&
				(*color == surface_color::rgba16f || *color == surface_color::rgba32f || *color == surface_color::r32f))
			{
				const unsigned element = *color == surface_color::rgba16f ? 2 : 4;
				constexpr std::array words{0u, 0x80000000u, 1u, 0x807fffffu, 0x00800000u, 0x7f800000u, 0xff800000u,
					0x7fc12345u, 0xffc04321u, 0x7f812345u, 0xff804321u, 0x7f7fffffu};
				const auto fill = [&](auto& bytes, unsigned seed)
				{
					for (unsigned i = 0; i < bytes.size() / element; ++i)
					{
						const auto word = element == 2 ? (i + seed) & 65535u : words[(i + seed) % words.size()];
						for (unsigned c = 0; c < element; ++c) bytes[i * element + c] = std::byte(word >> (8 * (element - c - 1)));
					}
				};
				fill(original, 0); fill(initial, 31745);
			}
		if (overlap) initial = original;
		const auto input = make_upload(ctx, original), seed = make_upload(ctx, initial), output = make_buffer(ctx, initial.size(), buffer_storage::shared);
		texture_region sa, da;
		sa.size = MTLSizeMake(sw * a.x, sh * a.y, 1);
		da.size = MTLSizeMake(dw * b.x, dh * b.y, 1);
		const auto gate = gated ? [ctx.device() newSharedEvent] : nil;
		struct release { id<MTLSharedEvent> event; ~release() { if (event) event.signaledValue = 1; } } release_gate{gate};
		ctx.begin();
		if (gate) ctx.wait_event(gate, 1);
		upload(sf, input, input_image, sa);
		upload(df, seed, output_image, da);
		if (sc > 1)
			samples.import(ctx, input_image, source);
		if (dc > 1)
			samples.import(ctx, output_image, target);
		texture_region sr, dr;
		sr.size = MTLSizeMake(sw, sh, 1);
		dr.size = MTLSizeMake(dw, dh, 1);
		if (partial)
		{
			sr.origin = MTLOriginMake(1, 1, 0);
			sr.size = MTLSizeMake(4, 3, 1);
			dr.origin = MTLOriginMake(2, 2, 0);
			dr.size = MTLSizeMake(5, 4, 1);
		}
		const auto encode = [&]
		{
			const auto run = [&](auto& helper) { helper.nearest(ctx, sf, source, sr, df, target, dr, plane, {bool(flip & 1), bool(flip & 2), false}, typeless); };
			if (gated) { surface_sample_blitter temporary(ctx); run(temporary); }
			else run(blitter);
		};
		if (bounded)
		{
			ctx.submit().wait();
			@autoreleasepool
			{
				const auto ledger = ctx.buffer_ledger();
				const auto before = ledger->statistics();
				constexpr std::size_t headroom = 128u << 10;
				require(before.budget - before.reserved_bytes > headroom, "Window blit fixture has no pressure headroom");
				const auto blocker = ledger->shared_buffer(((before.budget - before.reserved_bytes - headroom) / 16384) * 16384);
				const auto limited = ledger->statistics();
				require(blocker && limited.budget - limited.reserved_bytes == headroom,
					"Window blit fixture did not bound its actual native ledger");
				ctx.begin(); encode(); ctx.submit().wait();
			}
			ctx.begin();
		}
		else encode();
		if (dc > 1)
			samples.expand(ctx, target, output_image);
		download(df, output_image, da, output);
		const auto work = ctx.submit();
		if (gate)
		{
			bool timed_out = false;
			try { work.wait(std::chrono::milliseconds(1)); } catch (const submission_wait_timeout&) { timed_out = true; }
			require(timed_out && !work.ready(), "Sample-window lifetime case has no real pending dependency");
			gate.signaledValue = 1;
		}
		work.wait();
		auto expected = initial;
		const auto width = sr.size.width * a.x * sb / db, height = sr.size.height * a.y;
		for (std::size_t y = 0; y < dr.size.height * b.y; ++y)
			for (std::size_t x = 0; x < dr.size.width * b.x; ++x)
			{
				const auto from = (sr.origin.y * a.y + nearest(y, height, dr.size.height * b.y, flip & 2)) * sw * a.x * sb +
								  sr.origin.x * a.x * sb + nearest(x, width, dr.size.width * b.x, flip & 1) * db;
				const auto to = ((y + dr.origin.y * b.y) * dw * b.x + x + dr.origin.x * b.x) * db;
				for (std::size_t byte = 0; byte < db; ++byte)
					if (plane == texture_plane::all || (plane == texture_plane::depth && byte < 3) || (plane == texture_plane::stencil && byte == 3))
						expected[to + byte] = original[from + byte];
			}
		const auto actual = map_buffer(output, 0, output.length);
		if (bounded) bounded_bytes += actual.size();
		if (!std::equal(actual.begin(), actual.end(), expected.begin()))
		{
			const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
			std::cerr << "Sample blit sf=" << sf.index() << ":" << std::visit([](auto f)
																			  {
																				  return unsigned(f);
																			  },
																			  sf)
					  << " df=" << df.index() << ":" << std::visit([](auto f)
																   {
																	   return unsigned(f);
																   },
																   df)
					  << " sc=" << sc << " dc=" << dc << " flip=" << flip << " plane=" << unsigned(plane) << " byte=" << mismatch.first - actual.begin() << '\n';
			throw std::runtime_error("Spatial sample nearest/typeless scale averaged samples or changed expected guest bytes/aspects");
		}
	}
};
void bounded_windows()
{
	context ctx(1);
	auto ledger = std::make_shared<heap_budget>(ctx, 128u << 20);
	ctx.set_buffer_budget(ledger);
	fixture f(ctx);
	const std::array counts{std::array{1u, 4u}, std::array{2u, 4u}, std::array{4u, 2u}, std::array{4u, 1u}};
	unsigned regular = 0, casts = 0, planes = 0, overlaps = 0;
	for (const auto format : formats)
		for (unsigned c = 0; c < counts.size(); ++c)
			@autoreleasepool
			{
				f.check(format, format, counts[c][0], counts[c][1], c, false, texture_plane::all, true, true); ++regular;
			}
	const std::array pairs{
		std::pair{surface_format{surface_color::rgba32f}, surface_format{surface_color::argb8}},
		std::pair{surface_format{surface_color::rgb565}, surface_format{surface_color::rgba16f}},
		std::pair{surface_format{surface_depth::z24s8_uint}, surface_format{surface_depth::z24s8_float}},
		std::pair{surface_format{surface_depth::z24s8_float}, surface_format{surface_color::argb8}},
		std::pair{surface_format{surface_color::argb8}, surface_format{surface_depth::z24s8_float}},
		std::pair{surface_format{surface_color::rgba16f}, surface_format{surface_color::gb8}}};
	for (const auto& [source, target] : pairs)
		for (const auto& count : counts)
			for (unsigned flip : {0u, 1u, 2u, 3u})
				@autoreleasepool
				{
					f.check(source, target, count[0], count[1], flip, true, texture_plane::all, true, true); ++casts;
				}
	for (const auto format : {surface_depth::z24s8_uint, surface_depth::z24s8_float})
		for (const auto& count : counts)
			for (const auto plane : {texture_plane::depth, texture_plane::stencil})
				@autoreleasepool
				{
					f.check(format, format, count[0], count[1], 3, false, plane, true, true); ++planes;
				}
	for (const auto format : formats)
		for (unsigned count : {2u, 4u})
			for (unsigned flip : {0u, 3u})
				@autoreleasepool
				{
					f.check(format, format, count, count, flip, false, texture_plane::all, true, true, true); ++overlaps;
				}
	unsigned pending = 0;
	for (const auto format : formats)
		for (unsigned count : {2u, 4u})
			@autoreleasepool
			{
				f.check(format, format, count, count, 3, false, texture_plane::all, true, false, false, true); ++pending;
			}
	for (const auto& [source, target] : pairs)
		for (const auto& count : counts)
			@autoreleasepool
			{
				f.check(source, target, count[0], count[1], 3, true, texture_plane::all, true, false, false, true); ++pending;
			}
	ctx.wait_idle();
	std::cout << "Pending MSAA windows: " << pending << " real event timeouts after blitter destruction preserve private windows and shader/transfer owners, exact resumed output passed\n";
	std::cout << "Bounded MSAA windows: " << regular << " regular, " << casts << " typeless, " << planes << " selected-aspect, " << overlaps
		<< " self-overlapping copies; " << f.bounded_bytes << " exact guest bytes across whole targets with 128 KiB headroom, all 18 formats, 1/2/4 samples, scaling/flips, all half patterns and float special words, preserved exterior passed\n";
}
void stencil_without_depth(context& ctx)
{
	sample_transfer samples(ctx);
	const auto multi = texture(ctx, surface_depth::z24s8_float, 4, 3, 4), expanded = texture(ctx, surface_depth::z24s8_float, 8, 6, 1);
	texture_region whole;
	whole.size = MTLSizeMake(8, 6, 1);
	std::vector<std::uint32_t> invalid(48, 0x7fc12345u);
	std::vector<std::byte> stencil(48);
	for (std::size_t i = 0; i < stencil.size(); ++i)
		stencil[i] = static_cast<std::byte>(i * 7);
	const auto depth = make_upload(ctx, std::as_bytes(std::span(invalid))), si = make_upload(ctx, stencil);
	const auto output = make_buffer(ctx, invalid.size() * 4, buffer_storage::shared), so = make_buffer(ctx, stencil.size(), buffer_storage::shared);
	ctx.begin();
	buffer_to_texture(ctx, depth, 0, expanded, whole, 32, 0, texture_plane::depth);
	buffer_to_texture(ctx, si, 0, expanded, whole, 8, 0, texture_plane::stencil);
	samples.import(ctx, expanded, multi, texture_plane::stencil);
	// Replacing the expanded stencil only must not touch its NaN depth words,
	// and must never ask the GPU depth-domain validator to inspect them.
	samples.expand(ctx, multi, expanded, texture_plane::stencil);
	texture_to_buffer(ctx, expanded, whole, output, 0, 32, 0, texture_plane::depth);
	texture_to_buffer(ctx, expanded, whole, so, 0, 8, 0, texture_plane::stencil);
	ctx.submit().wait();
	const auto actual = map_buffer(output, 0, output.length), sa = map_buffer(so, 0, so.length);
	require(std::equal(actual.begin(), actual.end(), std::as_bytes(std::span(invalid)).begin()), "Stencil-only sample transfer modified/validated unrelated NaN depth");
	require(std::equal(sa.begin(), sa.end(), stencil.begin()), "Stencil-only native sample transfer lost sample identity");
	std::cout << "Stencil-only import/expansion preserves unrelated NaN depth payloads without depth validation\n";
}
} // namespace
int main(int argc, char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			if (argc == 2 && std::string_view(argv[1]) == "bounded-windows") { bounded_windows(); return 0; }
			context ctx(1);
			fixture f(ctx);
			std::size_t regular = 0, typeless = 0, planes = 0;
			for (const auto format : formats)
				for (const unsigned source : {1u, 2u, 4u})
					for (const unsigned target : {1u, 2u, 4u})
						for (const unsigned flip : {0u, 1u, 2u, 3u})
						{
							@autoreleasepool
							{
								f.check(format, format, source, target, flip, false);
							}
							++regular;
						}
			std::cout << regular << " spatial 1/2/4-sample color/depth nearest scales and flips passed without averaging\n";
			for (const auto sf : formats)
				for (const auto df : formats)
					for (const auto counts : {std::array{2u, 4u}, std::array{4u, 2u}, std::array{1u, 4u}, std::array{4u, 1u}})
						for (const unsigned flip : {0u, 3u})
						{
							@autoreleasepool
							{
								f.check(sf, df, counts[0], counts[1], flip, true);
							}
							++typeless;
						}
			std::cout << typeless << " spatial multisample typeless 18x18 format-pair scales preserve independent guest sample words\n";
			for (const unsigned source : {1u, 2u, 4u})
				for (const unsigned target : {1u, 2u, 4u})
					for (const auto plane : {texture_plane::all, texture_plane::depth, texture_plane::stencil})
					{
						@autoreleasepool
						{
							f.check(surface_depth::z24s8_float, surface_depth::z24s8_float, source, target, 3, false, plane, true);
						}
						++planes;
					}
			std::cout << planes << " partial spatial sample blits preserve exterior pixels and unselected depth/stencil aspects\n";
			stencil_without_depth(ctx);
			bounded_windows();
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
