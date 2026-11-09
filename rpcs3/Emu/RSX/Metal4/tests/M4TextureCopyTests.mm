#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"
#include "M4SampleTransfer.h"
#include "M4HeapBudget.h"
#include "M4Pipeline.h"
#include "M4TextureBlit.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <vector>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void store(std::span<std::byte> data, std::uint32_t word, unsigned bytes)
{
	for (unsigned byte = 0; byte < bytes; ++byte)
		data[byte] = static_cast<std::byte>(word >> ((bytes - byte - 1) * 8));
}
void texture_blit_dependencies()
{
	context ctx{2};
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:4 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsagePixelFormatView;
	const auto a = make_texture(ctx, descriptor), b = make_texture(ctx, descriptor), c = make_texture(ctx, descriptor);
	std::array<std::uint32_t, 16> words;
	for (unsigned i = 0; i < words.size(); ++i) words[i] = 0x1f103400u + i;
	const auto input = make_upload(ctx, std::as_bytes(std::span(words)));
	const auto output = make_buffer(ctx, sizeof(words) * 2, buffer_storage::shared);
	texture_region whole;
	whole.size = MTLSizeMake(4, 4, 1);
	const auto view = make_texture_view(a, MTLPixelFormatR32Uint, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1),
		MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha));
	ctx.begin();
	buffer_to_texture(ctx, input, 0, a, whole, 16);
	buffer_to_texture(ctx, input, 0, b, whole, 16);
	const auto independent = ctx.encoding_statistics();
	require(independent.texture_blit_queue_barriers == 1 && independent.texture_blit_barriers_elided == 2 &&
		independent.texture_blit_barriers == 0, "Independent texture uploads synchronized inside the pass");
	copy_texture(ctx, view, whole, c, whole);
	const auto parent_hazard = ctx.encoding_statistics();
	require(parent_hazard.texture_blit_alias_barriers == 1, "Texture view lost the parent upload dependency");
	texture_to_buffer(ctx, b, whole, output, 0, 16);
	texture_to_buffer(ctx, c, whole, output, sizeof(words), 16);
	ctx.submit().wait();
	for (std::size_t i = 0; i < 2; ++i)
		require(!std::memcmp(map_buffer(output, i * sizeof(words), sizeof(words)).data(), words.data(), sizeof(words)),
			"Texture blits returned a stale upload or copy");
	const auto complete = ctx.encoding_statistics();
	require(complete.texture_blit_barriers == 2 && complete.texture_blit_barriers_elided == 3,
		"Texture and buffer dependencies did not produce the expected barrier count");

	helper_catalog helpers(ctx);
	const auto library = helpers.library(R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void blit_texture_dependency(texture2d<uint, access::read> input [[texture(0)]],
    texture2d<uint, access::write> output [[texture(1)]], uint2 at [[thread_position_in_grid]]) {
    output.write(input.read(at) ^ uint4(0xffffffffu), at);
}
)metal", @"Native texture blit stage dependency");
	const auto pipeline = helpers.compute(library, @"blit_texture_dependency");
	ctx.begin();
	buffer_to_texture(ctx, input, 0, a, whole, 16);
	bindings io;
	io.set_texture(0, a);
	io.set_texture(1, c);
	const auto table = io.snapshot(ctx);
	const auto encoder = ctx.compute();
	ctx.keep_alive(pipeline);
	[encoder setComputePipelineState:pipeline];
	[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(4, 4, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
	texture_to_buffer(ctx, c, whole, output, 0, 16);
	ctx.submit().wait();
	for (std::size_t i = 0; i < words.size(); ++i)
		require(static_cast<const std::uint32_t*>(output.contents)[i] == ~words[i], "Compute transition lost a texture write");
	const auto mixed = ctx.encoding_statistics();
	require(mixed.checked_blit_raw_transitions == complete.checked_blit_raw_transitions + 1 &&
		mixed.texture_blit_alias_barriers == complete.texture_blit_alias_barriers + 1 &&
		mixed.compute_encoders == complete.compute_encoders + 1, "Raw compute handover changed the texture encoder contract");
	std::cout << "Native checked texture blits: three elided barriers, two dependent barriers, "
		<< "192 exact bytes across view, readback and raw compute transitions passed\n";
}

void texture_subresource_dependencies()
{
	context ctx{1};
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:8 height:8 mipmapped:YES];
	descriptor.textureType = MTLTextureType2DArray;
	descriptor.arrayLength = 3;
	descriptor.mipmapLevelCount = 3;
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsagePixelFormatView;
	const auto texture = make_texture(ctx, descriptor), copy = make_texture(ctx, descriptor);
	const auto identity = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
	const auto parent = make_texture_view(texture, texture.pixelFormat, MTLTextureType2DArray, NSMakeRange(1, 2), NSMakeRange(1, 2), identity);
	const auto nested = make_texture_view(parent, texture.pixelFormat, MTLTextureType2D, NSMakeRange(1, 1), NSMakeRange(1, 1), identity);
	const auto size = MTLSizeMake(2, 2, 1);
	const auto origin = MTLOriginMake(0, 0, 0);
	const std::array regions{texture_region{0, 0, origin, size}, texture_region{1, 0, origin, size},
		texture_region{1, 1, origin, size}, texture_region{2, 1, origin, size}, texture_region{2, 2, origin, size}};
	std::array<std::array<std::uint32_t, 4>, regions.size()> words;
	for (std::size_t image = 0; image < words.size(); ++image)
		for (std::size_t pixel = 0; pixel < words[image].size(); ++pixel)
			words[image][pixel] = 0x18a20000u + 16 * image + pixel;
	const auto input = make_upload(ctx, std::as_bytes(std::span(words)));
	const auto output = make_buffer(ctx, sizeof(words) + sizeof(words.back()), buffer_storage::shared);
	ctx.begin();
	for (std::size_t i = 0; i < regions.size() - 1; ++i)
		buffer_to_texture(ctx, input, i * sizeof(words[i]), texture, regions[i], 8);
	buffer_to_texture(ctx, input, sizeof(words) - sizeof(words.back()), nested, regions.front(), 8);
	const auto independent = ctx.encoding_statistics();
	require(independent.texture_blit_barriers == 0 && independent.texture_blit_barriers_elided == 5,
		"Disjoint texture mips and slices must use checked blits without intra-pass waits");
	copy_texture(ctx, texture, regions.back(), copy, regions[3]);
	const auto alias = ctx.encoding_statistics();
	require(alias.texture_blit_barriers == 1 && alias.texture_blit_alias_barriers == 1,
		"The nested view write must be visible through the corresponding root mip and slice");
	texture_to_buffer(ctx, copy, regions[3], output, sizeof(words), 8);
	require(ctx.encoding_statistics().texture_blit_barriers == 2, "The destination mip must wait for its direct texture copy");
	for (std::size_t i = 0; i < regions.size(); ++i)
		texture_to_buffer(ctx, texture, regions[i], output, i * sizeof(words[i]), 8);
	ctx.submit().wait();
	const auto actual = map_buffer(output, 0, output.length);
	require(!std::memcmp(actual.data(), words.data(), sizeof(words)) &&
		!std::memcmp(actual.data() + sizeof(words), words.back().data(), sizeof(words.back())),
		"Texture mip and slice tracking changed source or copied bytes");
	std::cout << "Native texture subresources: five independent uploads, two dependent blits, 96 exact bytes across mips, slices and nested views passed\n";
}

void texture_box_dependencies()
{
	std::size_t exact_bytes = 0;
	unsigned independent_pairs = 0, conservative_pairs = 0;
	enum class shape { array_view, volume, compressed, format_view };
	for (const auto type : {shape::array_view, shape::volume, shape::compressed, shape::format_view})
	{
		context ctx{1};
		const bool volume = type == shape::volume;
		const bool compressed = type == shape::compressed;
		const bool format_view = type == shape::format_view;
		const auto format = compressed ? MTLPixelFormatBC1_RGBA : MTLPixelFormatR32Uint;
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:8 height:4 mipmapped:NO];
		descriptor.textureType = volume ? MTLTextureType3D : MTLTextureType2DArray;
		descriptor.arrayLength = volume ? 1 : 2;
		descriptor.depth = volume ? 2 : 1;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsagePixelFormatView;
		const auto texture = make_texture(ctx, descriptor), copy = make_texture(ctx, descriptor);
		const auto identity = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
		const auto view = volume ? texture : make_texture_view(texture, format_view ? MTLPixelFormatR32Float : format,
			MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(1, 1), identity);
		const texture_region a{0, volume ? 0u : 1u, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)};
		auto b = a;
		b.origin = volume ? MTLOriginMake(0, 0, 1) : MTLOriginMake(4, 0, 0);
		auto view_b = b;
		view_b.slice = 0;
		auto patch = a;
		if (!compressed)
		{
			patch.origin.x = 1;
			patch.size.width = 2;
		}
		const auto block = storage_block(format);
		const auto box_bytes = transfer_layout(block, a.size.width, a.size.height, 1).allocation_size;
		const auto patch_bytes = transfer_layout(block, patch.size.width, patch.size.height, 1).allocation_size;
		std::vector<std::byte> original(2 * box_bytes + patch_bytes);
		for (std::size_t i = 0; i < original.size(); ++i)
			original[i] = static_cast<std::byte>((i * 37 + 29) & 255);
		const auto input = make_upload(ctx, original);
		const auto output = make_buffer(ctx, 3 * box_bytes, buffer_storage::shared);
		ctx.begin();
		buffer_to_texture(ctx, input, 0, texture, a);
		buffer_to_texture(ctx, input, box_bytes, view, view_b);
		const auto separate = ctx.encoding_statistics();
		require(separate.texture_blit_barriers == static_cast<unsigned>(format_view) &&
			separate.texture_blit_barriers_elided == 2u - static_cast<unsigned>(format_view),
			"Disjoint boxes must elide a barrier only when view coordinates are proven");
		independent_pairs += !format_view;
		conservative_pairs += format_view;
		buffer_to_texture(ctx, input, 2 * box_bytes, texture, patch);
		require(ctx.encoding_statistics().texture_blit_barriers == separate.texture_blit_barriers + 1,
			"An overlapping texture patch must wait for the earlier write");
		copy_texture(ctx, texture, a, copy, b);
		require(ctx.encoding_statistics().texture_blit_barriers == separate.texture_blit_barriers + 2,
			"A direct texture copy must wait for the overlapping patch");
		texture_to_buffer(ctx, copy, b, output, 2 * box_bytes);
		require(ctx.encoding_statistics().texture_blit_barriers == separate.texture_blit_barriers + 3,
			"The direct copy must be visible before readback");
		texture_to_buffer(ctx, texture, a, output, 0);
		texture_to_buffer(ctx, texture, b, output, box_bytes);
		ctx.submit().wait();
		auto expected = original;
		expected.resize(3 * box_bytes);
		if (compressed)
			std::copy_n(original.begin() + 2 * box_bytes, patch_bytes, expected.begin());
		else
			for (std::size_t y = 0; y < 4; ++y)
				std::copy_n(original.begin() + 2 * box_bytes + y * 8, 8, expected.begin() + y * 16 + 4);
		std::copy_n(expected.begin(), box_bytes, expected.begin() + 2 * box_bytes);
		const auto actual = map_buffer(output, 0, output.length);
		require(std::equal(actual.begin(), actual.end(), expected.begin()), "Native texture box tracking changed pixels or compressed blocks");
		exact_bytes += actual.size();
	}
	std::cout << "Native texture boxes: " << independent_pairs << " disjoint upload pairs, " << conservative_pairs
		<< " conservative format-view pair, 12 dependent barriers and " << exact_bytes << " exact bytes passed\n";
}

void formats(context& ctx, heap_budget& budget)
{
	surface_color_transfer colors(ctx);
	surface_depth_transfer depths(ctx);
	sample_transfer samples(ctx);
	std::size_t cases = 0;
	std::size_t msaa_barriers = 0;
	for (const bool is_depth : {false, true})
		for (const unsigned count : {1u, 2u, 4u})
			for (const bool alias : {false, true})
				for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
				{
					if (is_depth && raw > 4)
						continue;
					@autoreleasepool
					{
						const auto native = is_depth ? describe_surface_depth(static_cast<surface_depth>(raw)).format : describe_surface_color(static_cast<surface_color>(raw)).format;
						const auto bytes = is_depth ? describe_surface_depth(static_cast<surface_depth>(raw)).guest_bytes : color_layout(static_cast<surface_color>(raw)).pixel_bytes;
						const auto size = count == 1 ? std::array<std::size_t, 2>{8, 6} : sample_expanded_size(8, 6, count);
						const auto pitch = size[0] * bytes;
						std::vector<std::byte> original(pitch * size[1]), initial(original.size());
						for (std::size_t pixel = 0; pixel < size[0] * size[1]; ++pixel)
							for (unsigned target = 0; target < 2; ++target)
							{
								auto& data = target ? initial : original;
								const auto offset = pixel * bytes;
								if (!is_depth && (raw == 11 || raw == 12 || raw == 13))
								{
									const auto element = raw == 11 ? 2u : 4u;
									for (unsigned c = 0; c < bytes / element; ++c)
										store(std::span(data).subspan(offset + c * element), static_cast<std::uint32_t>((raw == 11 ? 0x3400u : 0x3e800000u) + pixel + c + target * 128), element);
								}
								else
								{
									const auto word = !is_depth ? 0x16325070u + pixel + target * 256 : raw == 1 ? 0x4000u + pixel + target * 256
																								   : raw == 3	? 0x5000u + pixel + target * 128
																												: (((raw == 2 ? 0x400000u : 0x7d0000u) + pixel + target * 256) << 8) | ((pixel * 7 + target * 64) % 256);
									store(std::span(data).subspan(offset), static_cast<std::uint32_t>(word), bytes);
								}
							}
						const auto input = make_upload(ctx, original), target_input = make_upload(ctx, initial), output = make_buffer(ctx, original.size(), buffer_storage::shared);
						MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native width:size[0] height:size[1] mipmapped:NO];
						descriptor.storageMode = MTLStorageModePrivate;
						descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
						const auto expanded_source = make_texture(ctx, descriptor), expanded_destination = alias ? expanded_source : make_texture(ctx, descriptor);
						descriptor.width = 8;
						descriptor.height = 6;
						if (count > 1)
						{
							descriptor.textureType = MTLTextureType2DMultisample;
							descriptor.sampleCount = count;
						}
						const auto source = count == 1 ? expanded_source : make_texture(ctx, descriptor), destination = alias ? source : count == 1 ? expanded_destination
																																					: make_texture(ctx, descriptor);
						texture_region whole;
						whole.size = MTLSizeMake(size[0], size[1], 1);
						ctx.begin();
						const auto before_copy = ctx.encoding_statistics();
						const auto upload = [&](id<MTLBuffer> data, id<MTLTexture> texture)
						{
							if (is_depth)
								depths.upload(ctx, static_cast<surface_depth>(raw), data, {}, texture, whole);
							else
								colors.upload(ctx, static_cast<surface_color>(raw), data, {}, texture, whole);
						};
						upload(input, expanded_source);
						if (!alias)
							upload(target_input, expanded_destination);
						if (count > 1)
						{
							samples.import(ctx, expanded_source, source);
							if (!alias)
								samples.import(ctx, expanded_destination, destination);
						}
						texture_region sr, dr;
						sr.size = dr.size = MTLSizeMake(4, 4, 1);
						dr.origin = MTLOriginMake(2, 1, 0);
						const auto before = budget.statistics().created;
						copy_texture(ctx, source, sr, destination, dr);
						if (!alias)
							require(budget.statistics().created == before, "Disjoint native texture blit allocated a staging buffer");
						else if (count == 1)
							require(budget.statistics().created > before, "Single-sample alias copy bypassed its snapshot");
						if (count > 1)
						{
							samples.expand(ctx, destination, expanded_destination);
							const auto after_copy = ctx.encoding_statistics();
							// Raw float expansion now uses compute plus a texture blit.
							// Imports and normalized/stencil expansion still rasterize.
							const bool compute_expansion = native == MTLPixelFormatR32Float || native == MTLPixelFormatRGBA16Float ||
								native == MTLPixelFormatRGBA32Float || native == MTLPixelFormatDepth32Float;
							const auto expected_barriers = (alias ? 1u : 2u) + unsigned(!compute_expansion);
							require(after_copy.full_render_barriers == before_copy.full_render_barriers + expected_barriers &&
								after_copy.fragment_render_barriers == before_copy.fragment_render_barriers,
								"MSAA sample transfers around native texture copies must keep full render barriers");
							msaa_barriers += expected_barriers;
						}
						if (is_depth)
							depths.download(ctx, static_cast<surface_depth>(raw), expanded_destination, whole, output, {});
						else
							colors.download(ctx, static_cast<surface_color>(raw), expanded_destination, whole, output, {});
						ctx.submit().wait();
						auto expected = alias ? original : initial;
						const std::size_t sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1;
						for (std::size_t y = sy; y < 5 * sy; ++y)
							for (std::size_t x = 2 * sx; x < 6 * sx; ++x)
								std::copy_n(original.begin() + static_cast<std::ptrdiff_t>((y - sy) * pitch + (x - 2 * sx) * bytes), bytes, expected.begin() + static_cast<std::ptrdiff_t>(y * pitch + x * bytes));
						const auto actual = map_buffer(output, 0, output.length);
						if (!std::equal(actual.begin(), actual.end(), expected.begin()))
						{
							const auto diff = std::mismatch(actual.begin(), actual.end(), expected.begin());
							std::cerr << "Copy depth=" << is_depth << " fmt=" << raw << " samples=" << count << " alias=" << alias << " byte=" << diff.first - actual.begin() << " actual=" << unsigned(*diff.first) << " expected=" << unsigned(*diff.second) << '\n';
							throw std::runtime_error("Native texture region copy differs");
						}
						++cases;
					}
				}
	std::cout << cases << " native color/depth/stencil region copies, overlap snapshots and 1/2/4 samples passed\n";
	std::cout << msaa_barriers << " full MSAA render barriers around native texture copies passed\n";
}

void planes(context& ctx, heap_budget& budget)
{
	MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:8 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	const auto texture = make_texture(ctx, descriptor);
	const auto separate = make_texture(ctx, descriptor);
	std::vector<std::uint32_t> original(32);
	std::vector<std::byte> stencil(32);
	for (std::size_t i = 0; i < 32; ++i)
	{
		original[i] = i % 4 == 0 ? 0x7fc12345u : i % 4 == 1 ? 0x80000000u
															: 0x3f000007u + static_cast<unsigned>(i);
		stencil[i] = static_cast<std::byte>(i * 7);
	}
	const auto input = make_upload(ctx, std::as_bytes(std::span(original))), si = make_upload(ctx, stencil), output = make_buffer(ctx, 128, buffer_storage::shared), so = make_buffer(ctx, 32, buffer_storage::shared);
	texture_region whole;
	whole.size = MTLSizeMake(8, 4, 1);
	for (const bool alias : {false, true})
	for (const auto plane : {texture_plane::all, texture_plane::depth, texture_plane::stencil})
	{
		const auto destination = alias ? texture : separate;
		ctx.begin();
		buffer_to_texture(ctx, input, 0, texture, whole, 32, 0, texture_plane::depth);
		buffer_to_texture(ctx, si, 0, texture, whole, 8, 0, texture_plane::stencil);
		if (!alias)
		{
			buffer_to_texture(ctx, input, 0, destination, whole, 32, 0, texture_plane::depth);
			buffer_to_texture(ctx, si, 0, destination, whole, 8, 0, texture_plane::stencil);
		}
		texture_region sr, dr;
		sr.size = dr.size = MTLSizeMake(4, 3, 1);
		dr.origin = MTLOriginMake(2, 1, 0);
		const auto before = budget.statistics().created;
		copy_texture(ctx, texture, sr, destination, dr, plane);
		require((budget.statistics().created == before) == (!alias && plane == texture_plane::all),
			"Combined D/S copy did not choose the expected direct or plane-snapshot path");
		texture_to_buffer(ctx, destination, whole, output, 0, 32, 0, texture_plane::depth);
		texture_to_buffer(ctx, destination, whole, so, 0, 8, 0, texture_plane::stencil);
		ctx.submit().wait();
		auto expected_depth = original;
		auto expected_stencil = stencil;
		for (std::size_t y = 1; y < 4; ++y)
			for (std::size_t x = 2; x < 6; ++x)
			{
				if (plane != texture_plane::stencil)
					expected_depth[y * 8 + x] = original[(y - 1) * 8 + x - 2];
				if (plane != texture_plane::depth)
					expected_stencil[y * 8 + x] = stencil[(y - 1) * 8 + x - 2];
			}
		const auto actual = map_buffer(output, 0, 128), sa = map_buffer(so, 0, 32);
		require(std::equal(actual.begin(), actual.end(), std::as_bytes(std::span(expected_depth)).begin()), "Raw depth copy changed bits or untouched depth aspect");
		require(std::equal(sa.begin(), sa.end(), expected_stencil.begin()), "Stencil copy changed untouched stencil aspect");
	}
	std::cout << "Raw f32 depth -0/NaN payloads: direct combined D/S, selected planes and overlap snapshots passed\n";
}
void geometry(context& ctx, heap_budget& budget)
{
	for (const bool volume : {false, true})
		for (const bool compressed : {false, true})
		for (const bool alias : {false, true})
		{
			if (volume && compressed)
				continue;
			const auto format = compressed ? MTLPixelFormatBC1_RGBA : volume ? MTLPixelFormatR8Unorm
																			 : MTLPixelFormatRGBA32Float;
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:16 height:16 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			if (volume)
			{
				descriptor.textureType = MTLTextureType3D;
				descriptor.depth = 4;
				descriptor.width = descriptor.height = 8;
			}
			else
			{
				descriptor.textureType = MTLTextureType2DArray;
				descriptor.arrayLength = 2;
				descriptor.mipmapLevelCount = 2;
			}
			const auto texture = make_texture(ctx, descriptor);
			const auto destination = alias ? texture : make_texture(ctx, descriptor);
			const auto block = storage_block(format);
			texture_region whole;
			whole.level = volume ? 0 : 1;
			whole.size = MTLSizeMake(8, 8, volume ? 4 : 1);
			const auto layout = transfer_layout(block, 8, 8, whole.size.depth);
			std::vector<std::byte> original(layout.allocation_size), initial(original.size(), std::byte{0xa5});
			for (std::size_t i = 0; i < original.size(); ++i)
				original[i] = static_cast<std::byte>((i * 13 + 7) % 256);
			if (!volume && !compressed)
				for (std::size_t i = 0; i < original.size(); i += 4)
				{
					const std::uint32_t word = i % 16 == 0 ? 0x7fc12345u : i % 16 == 4 ? 0x80000000u : i % 16 == 8 ? 0x7f800000u : 0x3f000007u + static_cast<unsigned>(i);
					std::copy_n(reinterpret_cast<const std::byte*>(&word), 4, original.begin() + static_cast<std::ptrdiff_t>(i));
				}
			const auto input = make_upload(ctx, original), seed = make_upload(ctx, initial), output = make_buffer(ctx, original.size(), buffer_storage::shared);
			ctx.begin();
			buffer_to_texture(ctx, input, 0, texture, whole, layout.bytes_per_row, layout.bytes_per_image);
			texture_region destination_whole = whole;
			if (!volume)
			{
				destination_whole.slice = 1;
			}
			if (!volume || !alias)
				buffer_to_texture(ctx, seed, 0, destination, destination_whole, layout.bytes_per_row, layout.bytes_per_image);
			texture_region sr = whole, dr = destination_whole;
			sr.size = dr.size = MTLSizeMake(4, 4, volume ? 2 : 1);
			dr.origin = MTLOriginMake(volume ? 2 : 4, volume ? 2 : 4, volume ? 1 : 0);
			const auto before = budget.statistics().created;
			if (!volume && !compressed)
			{
				const auto identity = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha);
				const auto view = make_texture_view(texture, format, MTLTextureType2D, NSMakeRange(1, 1), NSMakeRange(0, 1), identity);
				sr.level = 0;
				const auto nested = make_texture_view(view, format, MTLTextureType2D, NSMakeRange(0, 1), NSMakeRange(0, 1), identity);
				copy_texture(ctx, nested, sr, destination, dr);
			}
			else
				copy_texture(ctx, texture, sr, destination, dr);
			require((budget.statistics().created == before) == !alias, "Array/mip/view/3D copy used the wrong allocation path");
			texture_to_buffer(ctx, destination, destination_whole, output, 0, layout.bytes_per_row, layout.bytes_per_image);
			ctx.submit().wait();
			auto expected = volume && alias ? original : initial;
			const auto row_blocks = 4 / block.height, columns = 4 / block.width;
			for (std::size_t z = 0; z < sr.size.depth; ++z)
				for (std::size_t y = 0; y < row_blocks; ++y)
					for (std::size_t x = 0; x < columns; ++x)
					{
						const auto source = z * (layout.bytes_per_image ? layout.bytes_per_image : layout.allocation_size) + y * layout.bytes_per_row + x * block.bytes;
						const auto target = (z + dr.origin.z) * (layout.bytes_per_image ? layout.bytes_per_image : layout.allocation_size) + (y + dr.origin.y / block.height) * layout.bytes_per_row + (x + dr.origin.x / block.width) * block.bytes;
						std::copy_n(original.begin() + static_cast<std::ptrdiff_t>(source), block.bytes, expected.begin() + static_cast<std::ptrdiff_t>(target));
					}
			const auto actual = map_buffer(output, 0, output.length);
			require(std::equal(actual.begin(), actual.end(), expected.begin()), "Native array/mip/view/3D/compressed region copy differs");
		}
	std::cout << "Direct and alias array/mip/nested-view/3D/BC1 copies preserve exact bytes; direct paths allocate no staging buffers\n";
}

void backing_aliases(context& ctx, heap_budget& budget)
{
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:8 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate;
	descriptor.hazardTrackingMode = MTLHazardTrackingModeUntracked;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	const auto size = [ctx.device() heapTextureSizeAndAlignWithDescriptor:descriptor];
	const auto stride = (size.size + size.align - 1) / size.align * size.align;
	auto hd = [MTLHeapDescriptor new];
	hd.type = MTLHeapTypePlacement;
	hd.storageMode = MTLStorageModePrivate;
	hd.hazardTrackingMode = MTLHazardTrackingModeUntracked;
	hd.size = stride * 2;
	const auto heap = budget.allocate(hd);
	const auto source = [heap newTextureWithDescriptor:descriptor offset:0];
	const auto same = [heap newTextureWithDescriptor:descriptor offset:0];
	const auto disjoint = [heap newTextureWithDescriptor:descriptor offset:stride];
	require(source && same && disjoint && source != same, "Placement-heap alias fixture could not be created");
	std::array<std::uint32_t, 32> words;
	for (unsigned i = 0; i < words.size(); ++i) words[i] = 0x10203000u + i;
	const auto input = make_upload(ctx, std::as_bytes(std::span(words)));
	const auto output = make_buffer(ctx, sizeof(words), buffer_storage::shared);
	texture_region whole;
	whole.size = MTLSizeMake(8, 4, 1);
	for (const bool overlap : {false, true})
	{
		const auto destination = overlap ? same : disjoint;
		ctx.begin();
		const auto initial_barriers = ctx.encoding_statistics();
		buffer_to_texture(ctx, input, 0, source, whole, 32);
		if (!overlap) buffer_to_texture(ctx, input, 0, destination, whole, 32);
		if (!overlap)
		{
			const auto independent = ctx.encoding_statistics();
			require(independent.texture_blit_barriers_elided == initial_barriers.texture_blit_barriers_elided + 2 &&
				independent.texture_blit_barriers == initial_barriers.texture_blit_barriers,
				"Disjoint placement heap textures required a false alias barrier");
		}
		texture_region sr, dr;
		sr.size = dr.size = MTLSizeMake(6, 3, 1);
		dr.origin = MTLOriginMake(2, 1, 0);
		const auto before = budget.statistics().created;
		copy_texture(ctx, source, sr, destination, dr);
		require((budget.statistics().created > before) == overlap,
			"Distinct heap placements were incorrectly classified as disjoint or overlapping");
		texture_to_buffer(ctx, destination, whole, output, 0, 32);
		ctx.submit().wait();
		auto expected = words;
		for (unsigned y = 1; y < 4; ++y)
			for (unsigned x = 2; x < 8; ++x) expected[y * 8 + x] = words[(y - 1) * 8 + x - 2];
		require(std::equal(expected.begin(), expected.end(), static_cast<const std::uint32_t*>(output.contents)),
			"Placement-heap alias/disjoint region copy lost snapshot bits");
	}
	// Distinct buffer textures overlap by two rows. The original buffer must
	// remain unchanged in its first row and padding, like memmove.
	descriptor.storageMode = MTLStorageModeShared;
	descriptor.height = 3;
	const auto alignment = [ctx.device() minimumLinearTextureAlignmentForPixelFormat:descriptor.pixelFormat];
	const auto pitch = (32 + alignment - 1) / alignment * alignment;
	const auto buffer = make_buffer(ctx, pitch * 4, buffer_storage::shared);
	descriptor.hazardTrackingMode = buffer.hazardTrackingMode;
	descriptor.cpuCacheMode = buffer.cpuCacheMode;
	std::fill_n(static_cast<std::byte*>(buffer.contents), buffer.length, std::byte{0xa5});
	for (unsigned y = 0; y < 4; ++y)
		std::memcpy(static_cast<std::byte*>(buffer.contents) + y * pitch, words.data() + y * 8, 32);
	const auto first = [buffer newTextureWithDescriptor:descriptor offset:0 bytesPerRow:pitch];
	const auto second = [buffer newTextureWithDescriptor:descriptor offset:pitch bytesPerRow:pitch];
	require(first && second && first != second, "Buffer alias fixture could not be created");
	whole.size.height = 3;
	ctx.begin();
	const auto before = budget.statistics().created;
	copy_texture(ctx, first, whole, second, whole);
	require(budget.statistics().created > before, "Buffer-backed texture aliases bypassed their snapshot");
	ctx.submit().wait();
	for (unsigned y = 0; y < 4; ++y)
	{
		const auto row = static_cast<const std::byte*>(buffer.contents) + y * pitch;
		require(std::memcmp(row, words.data() + (y ? y - 1 : 0) * 8, 32) == 0, "Buffer-backed alias copy differs from memmove");
		for (std::size_t x = 32; x < pitch; ++x) require(row[x] == std::byte{0xa5}, "Buffer alias copy changed row padding");
	}
	const auto before_mixed = ctx.encoding_statistics();
	ctx.begin();
	fill_buffer(ctx, buffer, 0, 32, 0x52);
	texture_to_buffer(ctx, first, whole, output, 0, 32);
	ctx.submit().wait();
	for (unsigned byte = 0; byte < 32; ++byte)
		require(map_buffer(output, 0, 32)[byte] == std::byte{0x52}, "Buffer-backed view did not see the earlier buffer write");
	require(ctx.encoding_statistics().texture_blit_alias_barriers == before_mixed.texture_blit_alias_barriers + 1,
		"Buffer-backed view lost its buffer alias barrier");
	std::cout << "Distinct heap wrappers: overlapping placements snapshot, disjoint placements blit directly; overlapping buffer textures preserve rows and padding\n";
	std::cout << "Native texture backing dependencies: disjoint placement uploads elide barriers and a buffer-backed view sees 32 exact bytes\n";
}

void cube_edges(context& ctx, heap_budget& budget)
{
	for (const auto format : {MTLPixelFormatBC1_RGBA, MTLPixelFormatBC2_RGBA, MTLPixelFormatBC3_RGBA})
	{
		auto descriptor = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format size:12 mipmapped:YES];
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		const auto source = make_texture(ctx, descriptor), destination = make_texture(ctx, descriptor);
		texture_region sr;
		sr.level = 2;
		sr.slice = 5;
		sr.size = MTLSizeMake(3, 3, 1); // A partial compressed block at the mip edge.
		auto dr = sr;
		dr.slice = 2;
		const auto bytes = storage_block(format).bytes;
		std::vector<std::byte> raw(bytes);
		for (std::size_t i = 0; i < bytes; ++i) raw[i] = std::byte(i * 17 + 5);
		const auto input = make_upload(ctx, raw), output = make_buffer(ctx, bytes, buffer_storage::shared);
		ctx.begin();
		buffer_to_texture(ctx, input, 0, source, sr, bytes);
		const auto before = budget.statistics().created;
		copy_texture(ctx, source, sr, destination, dr);
		require(budget.statistics().created == before, "Compatible BC cube-edge copy allocated staging");
		texture_to_buffer(ctx, destination, dr, output, 0, bytes);
		ctx.submit().wait();
		require(std::equal(raw.begin(), raw.end(), static_cast<const std::byte*>(output.contents)), "BC cube/mip-edge copy changed raw blocks");
	}
	std::cout << "BC1/2/3 cube faces at partial-block mip edges copy exact compressed bits without staging\n";
}

// Source bits are the oracle, including signaling NaNs and unbounded depth.
// Projected coordinates use an independent integer center-nearest calculation.
void projected_raw_windows(context& ctx)
{
    texture_blitter blits(ctx);unsigned cases=0;std::size_t checked=0;std::set<std::uint32_t> selected_depth_words;
    constexpr std::array words{0x00000000u,0x80000000u,0x7f800000u,0xff800000u,0x7f812345u,0x7fc54321u,0x00000001u,0x007fffffu,0x3f800001u,0xc0000000u};
    constexpr std::array<std::uint16_t,10> halves{0,0x8000,0x7c00,0xfc00,0x7d23,0x7e45,1,0x3ff,0x3c01,0xc000};
    for(const auto format:{MTLPixelFormatR8Unorm,MTLPixelFormatRG8Unorm,MTLPixelFormatRGBA8Unorm,MTLPixelFormatR32Float,MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA32Float,MTLPixelFormatDepth16Unorm,MTLPixelFormatDepth32Float,MTLPixelFormatDepth32Float_Stencil8})
    for(unsigned flip=0;flip<4;++flip)for(bool alias:{false,true})for(bool equal:{false,true})
    for(const auto selected:format==MTLPixelFormatDepth32Float_Stencil8?std::vector{texture_plane::all,texture_plane::depth,texture_plane::stencil}:std::vector{format==MTLPixelFormatDepth16Unorm||format==MTLPixelFormatDepth32Float?texture_plane::depth:texture_plane::all})
    @autoreleasepool
    {
        auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:9 height:7 mipmapped:NO];
        descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
        const auto source=make_texture(ctx,descriptor),destination=alias?source:make_texture(ctx,descriptor);
        texture_region whole;whole.size=MTLSizeMake(9,7,1);
        texture_region input;input.origin=MTLOriginMake(1,1,0);input.size=equal?MTLSizeMake(3,2,1):MTLSizeMake(7,5,1);
        texture_region target;target.origin=MTLOriginMake(2,2,0);target.size=MTLSizeMake(3,2,1);
        struct plane_data{texture_plane plane;std::size_t bpp;std::vector<std::byte> original,expected;id<MTLBuffer> output;};
        std::vector<plane_data> data;
        const auto aspects=format==MTLPixelFormatDepth32Float_Stencil8?std::vector{texture_plane::depth,texture_plane::stencil}:std::vector{selected};
        ctx.begin();
        for(const auto plane:aspects)
        {
            const auto bpp=storage_block(format,plane).bytes;std::vector<std::byte> original(7*256,std::byte{0xa5}),background(7*256,std::byte{0x6d});
            for(unsigned y=0;y<7;++y)for(unsigned x=0;x<9;++x)for(unsigned lane=0;lane<bpp;++lane)
            {
                const auto pixel=y*9+x;
                const auto bits=format==MTLPixelFormatRGBA16Float?std::uint32_t(halves[(pixel+lane/2+flip*3+unsigned(equal)*2+unsigned(alias)*4)%halves.size()]):words[(pixel+lane/4+flip*3+unsigned(equal)*2+unsigned(alias)*4)%words.size()];
                original[y*256+x*bpp+lane]=std::byte(bits>>(8*(lane%(format==MTLPixelFormatRGBA16Float?2:4))));
                background[y*256+x*bpp+lane]=original[y*256+x*bpp+lane]^std::byte{0x55};
            }
            buffer_to_texture(ctx,make_upload(ctx,original),0,source,whole,256,0,plane);
            if(!alias)buffer_to_texture(ctx,make_upload(ctx,background),0,destination,whole,256,0,plane);
            auto expected=alias?original:background;
            if(selected==texture_plane::all||selected==plane)
            {
                const auto coordinate=[](unsigned at,unsigned offset,unsigned extent,unsigned copy,bool flip)
                {
                    const auto denominator=std::uint64_t(copy)*2;auto numerator=(std::uint64_t(at+offset)*2+1)*extent;
                    if(flip)numerator=std::uint64_t(extent)*denominator-numerator;
                    return unsigned(numerator/denominator);
                };
                for(unsigned y=0;y<2;++y)for(unsigned x=0;x<3;++x)
                {
                    const auto sx=1+coordinate(x,4,input.size.width,11,flip&1),sy=1+coordinate(y,5,input.size.height,9,flip&2);
                    std::copy_n(original.begin()+sy*256+sx*bpp,bpp,expected.begin()+(2+y)*256+(2+x)*bpp);
                    if(format==MTLPixelFormatDepth32Float){std::uint32_t bits=0;std::memcpy(&bits,original.data()+sy*256+sx*bpp,4);selected_depth_words.insert(bits);}
                }
            }
            data.push_back({plane,bpp,std::move(original),std::move(expected),make_buffer(ctx,7*256,buffer_storage::shared)});
        }
        blits.nearest(ctx,source,input,destination,target,selected,{bool(flip&1),bool(flip&2),false},{{11,9,1},{4,5,0}});
        for(const auto& plane:data)texture_to_buffer(ctx,destination,whole,plane.output,0,256,0,plane.plane);
        ctx.submit().wait();
        for(const auto& plane:data)
        {
            const auto bytes=map_buffer(plane.output,0,plane.output.length);
            for(unsigned y=0;y<7;++y)require(std::equal(bytes.begin()+y*256,bytes.begin()+y*256+9*plane.bpp,plane.expected.begin()+y*256),"Projected native window changed selected raw bits, exterior or unselected plane");
            checked+=7*9*plane.bpp;
        }
        ++cases;
    }
    for(const auto word:words)require(selected_depth_words.contains(word),"Raw projection did not actually select every special f32 depth word");
    std::cout<<"Projected raw native windows: "<<cases<<" nine-format center-nearest profiles, equal and unequal boxes, all X/Y flips, self aliases, independent depth/stencil, f16/f32 signed zero/subnormal/Inf/signaling and quiet NaN words, "<<checked<<" exact native bytes and untouched exterior passed\n";
}

void bounded_nearest_staging()
{
    unsigned cases=0;std::size_t exact=0;
    for(const auto format:{MTLPixelFormatR32Float,MTLPixelFormatRGBA16Float,MTLPixelFormatRGBA32Float,MTLPixelFormatDepth32Float,MTLPixelFormatDepth32Float_Stencil8,MTLPixelFormatR8Unorm,MTLPixelFormatRG8Unorm,MTLPixelFormatRGBA8Unorm})
    for(unsigned flips=0;flips<4;++flips)
    @autoreleasepool
    {
        context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,64u<<20);ctx.set_buffer_budget(ledger);texture_blitter blits(ctx);
        auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:256 height:256 mipmapped:NO];d.storageMode=MTLStorageModePrivate;d.usage=MTLTextureUsageShaderRead|MTLTextureUsageRenderTarget|MTLTextureUsagePixelFormatView;
        const auto source=make_texture(ctx,d);d.width=2;d.height=3;const auto target=make_texture(ctx,d);
        const auto planes=format==MTLPixelFormatDepth32Float_Stencil8?std::vector{texture_plane::depth,texture_plane::stencil}:std::vector{format==MTLPixelFormatDepth32Float?texture_plane::depth:texture_plane::all};
        struct data{id<MTLBuffer> output;std::vector<std::byte> expected;texture_plane plane;std::size_t bpp;};std::vector<data> checks;
        texture_region whole;whole.size=MTLSizeMake(256,256,1);texture_region window;window.size=MTLSizeMake(2,3,1);
        @autoreleasepool
        {
            ctx.begin();
            for(const auto plane:planes)
            {
                const auto bpp=storage_block(format,plane).bytes;std::vector<std::byte> words(256*256*bpp),expected(6*bpp);
                for(unsigned i=0;i<words.size();++i)words[i]=std::byte((i*73+i/11+29)&255);
                for(unsigned y=0;y<3;++y)for(unsigned x=0;x<2;++x)
                {
                    const unsigned sx=flips&1?255-(125+x):125+x,sy=flips&2?255-(126+y):126+y;
                    std::copy_n(words.begin()+(sy*256+sx)*bpp,bpp,expected.begin()+(y*2+x)*bpp);
                }
                buffer_to_texture(ctx,make_upload(ctx,words),0,source,whole,256*bpp,0,plane);
                checks.push_back({make_buffer(ctx,3*256,buffer_storage::shared),std::move(expected),plane,bpp});
            }
            ctx.submit().wait();
        }
        for(unsigned i=0;i<2;++i){ctx.begin();ctx.submit().wait();}(void)ctx.trim_upload_pages();
        // Admit the context's reusable uniform page before measuring spare
        // staging capacity; both old and new kernels require that page.
        ctx.begin();const std::uint32_t zero=0;(void)ctx.upload(std::as_bytes(std::span{&zero,1}),256);
        std::vector<id<MTLBuffer>> blockers;constexpr std::size_t reserve=192u<<10;
        auto state=ledger->statistics();auto attempt=state.budget-state.reserved_bytes-reserve;
        while(attempt)
        {
            try{blockers.push_back(ledger->buffer(attempt,MTLStorageModePrivate));state=ledger->statistics();attempt=state.budget-state.reserved_bytes>reserve?state.budget-state.reserved_bytes-reserve:0;}
            catch(const heap_budget_error&){attempt/=2;}
        }
        require(ledger->statistics().budget-ledger->statistics().reserved_bytes<=reserve,"Raw window fixture did not impose the intended spare budget");
        try{blits.nearest(ctx,source,whole,target,window,texture_plane::all,{bool(flips&1),bool(flips&2),false},{{256,256,1},{125,126,0}});}
        catch(const heap_budget_error& e){ctx.cancel();throw std::runtime_error("Nearest donor window allocated the full source staging image: requested="+std::to_string(e.requested));}
        for(const auto& check:checks)texture_to_buffer(ctx,target,window,check.output,0,256,0,check.plane);
        ctx.submit().wait();
        for(const auto& check:checks){const auto bytes=map_buffer(check.output,0,check.output.length);for(unsigned y=0;y<3;++y)require(std::equal(bytes.begin()+y*256,bytes.begin()+y*256+2*check.bpp,check.expected.begin()+y*2*check.bpp),"Bounded donor staging changed raw selected bits");exact+=check.expected.size();}
        require(ledger->statistics().reserved_bytes<=ledger->statistics().budget,"Bounded donor staging exceeded shared quota");++cases;
    }
    std::cout<<"Bounded nearest staging: "<<cases<<" eight-format 256x256 donors, 2x3 projected windows, all X/Y flips, at most 192KiB spare shared quota and "<<exact<<" exact native color/depth/stencil bytes passed\n";
}

void pending_copy(context& ctx, heap_budget& budget)
{
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 1; } } guard{gate};
	std::optional<submission> pending;
	const std::array<std::uint32_t, 4> raw{0x80000000, 0x7fc12345, 0xff800000, 0x3f800000};
	const auto output = make_buffer(ctx, sizeof(raw), buffer_storage::shared);
	__weak id<MTLTexture> weak_source, weak_destination;
	@autoreleasepool
	{
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:1 height:1 mipmapped:NO];
		descriptor.storageMode = MTLStorageModePrivate;
		const auto source = make_texture(ctx, descriptor), destination = make_texture(ctx, descriptor);
		weak_source = source;
		weak_destination = destination;
		ctx.begin();
		buffer_to_texture(ctx, make_upload(ctx, std::as_bytes(std::span(raw))), 0, source, {}, sizeof(raw));
		ctx.submit().wait();
		ctx.begin();
		ctx.wait_event(gate, 1);
		const auto before = budget.statistics().created;
		copy_texture(ctx, source, {}, destination, {});
		require(budget.statistics().created == before, "Pending direct copy allocated staging");
		texture_to_buffer(ctx, destination, {}, output, 0, sizeof(raw));
		pending = ctx.submit();
	}
	require(!pending->ready() && weak_source && weak_destination, "Pending direct copy did not retain both native texture owners");
	gate.signaledValue = 1;
	pending->wait();
	require(std::equal(raw.begin(), raw.end(), static_cast<const std::uint32_t*>(output.contents)),
		"Direct copy lost owner lifetime or float payloads after external references were released");
	std::cout << "Pending native direct copy retains released texture owners and exact -0/NaN/infinity bits\n";
}
} // namespace
int main(int argc,char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			if(argc>1){require(argc==2&&std::string_view(argv[1])=="bounded-nearest","Unknown native copy test selection");bounded_nearest_staging();return 0;}
			texture_blit_dependencies();
			texture_subresource_dependencies();
			texture_box_dependencies();
			context ctx(1);
			auto budget = std::make_shared<heap_budget>(ctx, 64u << 20);
			ctx.set_buffer_budget(budget);
			formats(ctx, *budget);
			planes(ctx, *budget);
			geometry(ctx, *budget);
			backing_aliases(ctx, *budget);
			cube_edges(ctx, *budget);
			projected_raw_windows(ctx);
			pending_copy(ctx, *budget);
			ctx.wait_idle();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
