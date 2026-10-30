#include "M4SurfaceColor.h"
#include "M4SurfaceDepth.h"
#include "M4SampleTransfer.h"
#include "M4HeapBudget.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>
#include <optional>
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
void formats(context& ctx, heap_budget& budget)
{
	surface_color_transfer colors(ctx);
	surface_depth_transfer depths(ctx);
	sample_transfer samples(ctx);
	std::size_t cases = 0;
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
							samples.expand(ctx, destination, expanded_destination);
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
		buffer_to_texture(ctx, input, 0, source, whole, 32);
		if (!overlap) buffer_to_texture(ctx, input, 0, destination, whole, 32);
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
	std::cout << "Distinct heap wrappers: overlapping placements snapshot, disjoint placements blit directly; overlapping buffer textures preserve rows and padding\n";
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
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			context ctx(1);
			auto budget = std::make_shared<heap_budget>(ctx, 64u << 20);
			ctx.set_buffer_budget(budget);
			formats(ctx, *budget);
			planes(ctx, *budget);
			geometry(ctx, *budget);
			backing_aliases(ctx, *budget);
			cube_edges(ctx, *budget);
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
