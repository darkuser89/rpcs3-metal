#include "M4SurfaceCache.h"
#include "M4GuardedGuestMemory.h"
#include "M4VertexFetch.h"
#include "M4VertexShader.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
class memory_fixture final : public guest_surface_memory
{
	std::mutex mutex;
	std::vector<std::byte> data;
	std::vector<std::uint64_t> versions;
	std::uint64_t counter = 0;
	void check(guest_surface_range range) const
	{
		validate_guest_surface_range(range);
		if (range.address > data.size() || range.size > data.size() - range.address)
			throw std::out_of_range("Fixture VM range is unmapped");
	}
	std::uint64_t version(guest_surface_range range) const
	{
		std::uint64_t result = 0;
		for (std::size_t byte = 0; byte < range.size; ++byte)
			result = std::max(result, versions[range.address + byte]);
		return result;
	}
	std::uint64_t version(const guest_surface_layout& coverage) const
	{
		std::uint64_t result = 0;
		guest_surface_byte_cursor cursor(coverage);
		while (const auto range = cursor.next())
			result = std::max(result, version(*range));
		return result;
	}
	std::uint64_t put(guest_surface_range range, std::span<const std::byte> bytes)
	{
		require(bytes.size() == range.size, "Fixture write size differs");
		++counter;
		std::copy(bytes.begin(), bytes.end(), data.begin() + range.address);
		std::fill_n(versions.begin() + range.address, range.size, counter);
		return counter;
	}

  public:
	std::function<void()> before_compare;
	std::function<void(guest_surface_range)> after_read;
	std::function<std::shared_ptr<void>(std::span<const guest_surface_range>)> backing;
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range> ranges) override
	{
		return backing ? backing(ranges) : std::shared_ptr<void>{};
	}
	explicit memory_fixture(std::size_t bytes = 1 << 20) : data(bytes), versions(bytes) {}
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		guest_surface_snapshot snapshot;
		{
			std::lock_guard lock(mutex);
			check(coverage.range);
			snapshot = {version(coverage), {data.begin() + coverage.range.address, data.begin() + coverage.range.address + coverage.range.size}};
		}
		if (after_read) after_read(coverage.range);
		return snapshot;
	}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override
	{
		std::lock_guard lock(mutex);
		check(coverage.range);
		return version(coverage);
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage, std::uint64_t expected, std::span<const std::byte> bytes) override
	{
		if (before_compare)
		{
			auto hook = std::move(before_compare);
			before_compare = {};
			hook();
		}
		std::lock_guard lock(mutex);
		check(coverage.range);
		require(bytes.size() == coverage.range.size, "Fixture surface write size differs");
		if (version(coverage) != expected)
			return {};
		++counter;
		guest_surface_byte_cursor cursor(coverage);
		while (const auto range = cursor.next())
		{
			const auto offset = range->address - coverage.range.address;
			std::copy_n(bytes.begin() + offset, range->size, data.begin() + range->address);
			std::fill_n(versions.begin() + range->address, range->size, counter);
		}
		return counter;
	}
	std::optional<std::uint64_t> compare_write_many(std::span<const guest_surface_write> writes) override
	{
		if (writes.empty()) return 0;
		if (writes.size() == 1) return compare_write(writes[0].coverage, writes[0].expected, writes[0].bytes);
		std::lock_guard lock(mutex);
		for (const auto& write : writes)
		{
			check(write.coverage.range);
			require(write.bytes.size() == write.coverage.range.size, "Fixture batch payload size differs");
			if (version(write.coverage) != write.expected) return {};
		}
		++counter;
		for (const auto& write : writes)
		{
			guest_surface_byte_cursor cursor(write.coverage);
			while (const auto range = cursor.next())
			{
				std::copy_n(write.bytes.begin() + range->address - write.coverage.range.address, range->size, data.begin() + range->address);
				std::fill_n(versions.begin() + range->address, range->size, counter);
			}
		}
		return counter;
	}
	void write(guest_surface_range range, std::span<const std::byte> bytes) override
	{
		std::lock_guard lock(mutex);
		check(range);
		(void)put(range, bytes);
	}
};
void store(std::span<std::byte> data, std::uint32_t word, unsigned bytes)
{
	for (unsigned byte = 0; byte < bytes; ++byte)
		data[byte] = static_cast<std::byte>(word >> ((bytes - byte - 1) * 8));
}
std::size_t morton(std::size_t x, std::size_t y, std::size_t z, const std::array<std::size_t, 3>& size)
{
	std::size_t output = 0, stride = 1;
	for (std::size_t bit = 1; bit < size[0] || bit < size[1] || bit < size[2]; bit *= 2)
		for (unsigned axis = 0; axis < 3; ++axis)
			if (bit < size[axis])
			{
				if (std::array{x, y, z}[axis] & bit)
					output += stride;
				stride *= 2;
			}
	return output;
}
unsigned bpp(const surface_allocation_description& allocation)
{
	if (const auto color = std::get_if<surface_color>(&allocation.format))
		return color_layout(*color).pixel_bytes;
	return describe_surface_depth(std::get<surface_depth>(allocation.format)).guest_bytes;
}
std::uint32_t seed_word(bool depth, unsigned format, std::size_t pixel)
{
	if (!depth)
		return static_cast<std::uint32_t>(0xa5c37e12u + pixel);
	if (format == 1)
		return static_cast<std::uint32_t>(0x4000 + pixel);
	if (format == 3)
		return static_cast<std::uint32_t>(0x5000 + pixel);
	return static_cast<std::uint32_t>(((format == 2 ? 0x400000u : 0x7d0000u) + pixel) << 8) | static_cast<std::uint32_t>((0xa5u + pixel * 3) % 256);
}
std::uint32_t color_clear_word(unsigned format, std::uint32_t seed, std::uint32_t word)
{
	if (format == 1)
		return (word & 0x7fff);
	if (format == 2)
		return word | 0x8000;
	if (format == 3)
		return word & 0xffff;
	if (format == 9)
		return word & 255;
	if (format == 10)
		return word & 65535;
	(void)seed;
	return word;
}
void cases(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 8 << 20);
	std::size_t count = 0;
	for (const bool depth : {false, true})
		for (const bool swizzled : {false, true})
			for (const unsigned samples : {1u, 2u, 4u})
				for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
				{
					if (depth && raw > 4)
						continue;
					cached_surface_description description;
					description.address = 0x1000;
					description.allocation.width = 8;
					description.allocation.height = 6;
					description.allocation.samples = samples;
					if (depth)
						description.allocation.format = static_cast<surface_depth>(raw);
					else
						description.allocation.format = static_cast<surface_color>(raw);
					const auto size = samples == 1 ? std::array<std::size_t, 2>{8, 6} : sample_expanded_size(8, 6, samples);
					const auto bytes = bpp(description.allocation);
					if (swizzled)
						description.morton = swizzle_region{{32, 16, 2}, {1, 2, 1}, {size[0], size[1], 1}};
					else
						description.row_pitch = (size[0] + 3) * bytes;
					const auto layout = plan_guest_surface(description.address, size[0], size[1], bytes, description.row_pitch, description.morton);
					std::vector<std::byte> original(layout.range.size, std::byte{0xa5});
					const auto offset = [&](std::size_t x, std::size_t y)
					{
						return swizzled ? morton(x + 1, y + 2, 1, {32, 16, 2}) * bytes : y * layout.row_pitch + x * bytes;
					};
					for (std::size_t y = 0; y < size[1]; ++y)
						for (std::size_t x = 0; x < size[0]; ++x)
						{
							const auto pixel = y * size[0] + x;
							if (!depth && (raw == 11 || raw == 12 || raw == 13))
							{
								const auto element = raw == 11 ? 2u : 4u;
								for (unsigned c = 0; c < bytes / element; ++c)
									store(std::span(original).subspan(offset(x, y) + c * element), raw == 11 ? 0x3800 : 0x3f000000, element);
							}
							else
								store(std::span(original).subspan(offset(x, y)), seed_word(depth, raw, pixel), bytes);
						}
					memory->write(layout.range, original);
					const auto surface = cache.acquire(description);
					cache.prepare(surface);
					require(cache.status(surface).initialized && !cache.status(surface).dirty, "Guest upload cache state differs");
					const auto value = depth ? (raw == 1 || raw == 3 ? 0x8000u : 0x8000003cu) : 0x7b3dc5e9u;
					cache.clear(surface, depth ? 3 : 0xf0, 0x7b3dc5e9, value, 0x59, {2, 1, 3, 3});
					const bool float_nop = !depth && (raw == 11 || raw == 12 || raw == 13);
					require(cache.status(surface).dirty != float_nop, "Clear dirty cache state differs");
					require(memory->read(layout.range).bytes == original, "Clear published guest bytes before explicit writeback");
					cache.spill(surface);
					require(cache.status(surface).spilled, "Surface didn't spill native backing");
					cache.prepare(surface);
					require(!cache.status(surface).spilled, "Native spill wasn't restored");
					cache.flush(surface);
					auto expected = original;
					for (std::size_t y = 0; y < size[1]; ++y)
						for (std::size_t x = 0; x < size[0]; ++x)
						{
							const auto px = samples == 1 ? x : x / 2, py = samples == 4 ? y / 2 : y;
							if (px < 2 || px >= 5 || py < 1 || py >= 4 || float_nop)
								continue;
							const auto pixel = y * size[0] + x;
							std::uint32_t word;
							if (depth)
							{
								const auto z = raw == 1 || raw == 3 ? 0x8000u : 0x800000u;
								const float normalized = float(z) / (raw == 1 || raw == 3 ? 65535.0f : 16777215.0f);
								const auto fb = std::bit_cast<std::uint32_t>(normalized);
								const auto depth_word = raw == 1 ? z : raw == 3 ? ((fb - 0x3c000000u) / 2048) % 65536
																   : raw == 2	? static_cast<std::uint32_t>(normalized * 16777215.0f)
																				: fb / 128;
								const auto stencil = (seed_word(true, raw, pixel) & (~0x59u & 255)) | (0x3cu & 0x59);
								word = raw == 1 || raw == 3 ? depth_word : (depth_word << 8) | stencil;
							}
							else
								word = color_clear_word(raw, seed_word(false, raw, pixel), 0x7b3dc5e9);
							store(std::span(expected).subspan(offset(x, y)), word, bytes);
						}
					require(memory->read(layout.range).bytes == expected, "Cached clear/spill/reload/writeback pixels or guest padding differ");
					require(!cache.status(surface).dirty, "Successful writeback stayed dirty");
					cache.erase(surface);
					++count;
				}
	std::cout << count << " cached 18-format linear/Morton/MSAA load-clear-spill-reload-writeback cases passed\n";
}

void aliases_and_vm(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 1 << 20);
	cached_surface_description a;
	a.address = 0x4000;
	a.allocation = {4, 2, 1, surface_color::argb8, false};
	std::vector<std::byte> seed(32, std::byte{0xa5});
	memory->write({a.address, 32}, seed);
	const auto first = cache.acquire(a);
	cache.clear(first, 0xf0, 0x11223344, 0, 255, {0, 0, 4, 2});
	cached_surface_description b = a;
	b.allocation = {8, 2, 1, surface_color::gb8, false};
	const auto second = cache.acquire(b);
	cache.clear(second, 0x60, 0x173b7dc1, 0, 255, {0, 0, 8, 2});
	auto bytes = memory->read({a.address, 32}).bytes;
	for (std::size_t i = 0; i < 32; i += 4)
		require(bytes[i] == std::byte{0x11} && bytes[i + 1] == std::byte{0x22} && bytes[i + 2] == std::byte{0x33} && bytes[i + 3] == std::byte{0x44}, "New alias didn't preserve old dirty guest content first");
	cache.prepare(first);
	bytes = cache.cpu_read({a.address, 32}).bytes;
	for (std::size_t i = 0; i < 32; i += 2)
		require(bytes[i] == std::byte{0x7d} && bytes[i + 1] == std::byte{0xc1}, "Typed guest alias didn't flush/reload latest bytes");
	require(cache.acquire(cached_surface_description{a.address, a.allocation, 16, {}}) == first, "Equivalent pitched descriptors didn't normalize");
	cache.clear(first, 0xf0, 0x22334455, 0, 255, {0, 0, 4, 2});
	const std::array patch{std::byte{0xff}};
	cache.cpu_write({a.address + 1, 1}, patch);
	bytes = memory->read({a.address, 32}).bytes;
	require(bytes[0] == std::byte{0x22} && bytes[1] == std::byte{0xff} && bytes[2] == std::byte{0x44}, "CPU write didn't flush native dirty data first");
	require(!cache.status(first).initialized, "CPU VM write didn't invalidate surface");
	cache.prepare(first);
	cache.clear(first, 0xf0, 0x55667788, 0, 255, {0, 0, 4, 2});
	memory->before_compare = [&]
	{
		const std::array changed{std::byte{0xee}};
		memory->write({a.address + 2, 1}, changed);
	};
	bool conflict = false;
	try
	{
		cache.flush(first);
	}
	catch (const surface_conflict_error&)
	{
		conflict = true;
	}
	require(conflict && cache.status(first).dirty && memory->read({a.address, 32}).bytes[2] == std::byte{0xee}, "Concurrent VM write was silently overwritten");
	cache.clear(first, 0xf0, 0x0a1b2c3d, 0, 255, {0, 0, 4, 2});
	cache.flush(first);
	bytes = memory->read({a.address, 32}).bytes;
	require(bytes[0] == std::byte{0x0a} && bytes[1] == std::byte{0x1b} && bytes[2] == std::byte{0x2c}, "Ordered full overwrite didn't recover from VM conflict");
	std::cout << "Typed guest aliases, descriptor normalization, CPU flush/invalidation and atomic VM conflict preservation passed\n";
}

void callbacks_and_timeout(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 1 << 20);
	cached_surface_description description;
	description.address = 0x8000;
	description.allocation = {8, 8, 1, surface_color::argb8, false};
	std::vector<std::byte> seed(256, std::byte{0xa5});
	memory->write({description.address, 256}, seed);
	const auto surface = cache.acquire(description);
	cache.prepare(surface);
	compiler compile(ctx);
	const auto library = compile.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 cache_vertex(uint id [[vertex_id]]) {const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);}
fragment float4 cache_fragment(float4 p [[position]]) {return float4(float(uint(p.x)+1)/255.0f,float(uint(p.y)+2)/255.0f,17.0f/255.0f,1.0f);}
)msl",
										 @"Independent cache draw");
	MTL4RenderPipelineDescriptor* pd = [MTL4RenderPipelineDescriptor new];
	pd.vertexFunctionDescriptor = library_function(library, @"cache_vertex");
	pd.fragmentFunctionDescriptor = library_function(library, @"cache_fragment");
	pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
	pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	const auto pipeline = compile.render(pd);
	const std::array uses{cached_surface_use{surface, true}};
	cache.draw(uses, [&](context& commands, std::span<const id<MTLTexture>> textures)
			   {
				   MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
				   pass.colorAttachments[0].texture = textures[0];
				   pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
				   pass.colorAttachments[0].storeAction = MTLStoreActionStore;
				   render_commands draw(commands, pass);
				   raster_state raster;
				   raster.viewport = {0, 0, 8, 8, 0, 1};
				   raster.scissor = {0, 0, 8, 8};
				   draw.configure(pipeline, nil, raster, {}, {});
				   draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
				   draw.end(); });
	cache.flush(surface);
	const auto bytes = memory->read({description.address, 256}).bytes;
	for (std::size_t y = 0; y < 8; ++y)
		for (std::size_t x = 0; x < 8; ++x)
			require(bytes[(y * 8 + x) * 4] == std::byte{255} && unsigned(bytes[(y * 8 + x) * 4 + 1]) == x + 1 && unsigned(bytes[(y * 8 + x) * 4 + 2]) == y + 2 && bytes[(y * 8 + x) * 4 + 3] == std::byte{17}, "Independent cached native draw output differs");
	bool cancelled = false;
	try
	{
		cache.draw(uses, [](context&, auto)
				   { throw std::runtime_error("Cancel this native cache encoding"); });
	}
	catch (const std::runtime_error&)
	{
		cancelled = true;
	}
	require(cancelled && !cache.status(surface).dirty && !cache.status(surface).faulted, "Cancelled encoding changed cache metadata");
	const auto event = [ctx.device() newSharedEvent];
	struct release_event
	{
		id<MTLSharedEvent> event;
		~release_event()
		{
			event.signaledValue = 1;
		}
	} guard{event};
	cache.set_wait_timeout(std::chrono::milliseconds(1));
	bool timeout = false;
	try
	{
		cache.draw(uses, [&](context& commands, auto)
				   { commands.wait_event(event, 1); });
	}
	catch (const std::runtime_error&)
	{
		timeout = true;
	}
	require(timeout && cache.status(surface).pending && !cache.status(surface).dirty, "Timeout committed pending native metadata");
	event.signaledValue = 1;
	cache.synchronize();
	cache.set_wait_timeout(std::chrono::seconds(30));
	require(!cache.status(surface).pending && cache.status(surface).dirty, "Completed native transaction didn't publish dirty state");
	cache.flush(surface);
	const auto status = make_buffer(ctx, 4, buffer_storage::shared);
	*static_cast<std::uint32_t*>(status.contents) = 1;
	bool failed = false;
	try
	{
		cache.draw(uses, [&](context& commands, auto)
				   { commands.require_gpu_success(status, 0, @"Intentional cached draw failure"); });
	}
	catch (const std::runtime_error&)
	{
		failed = true;
	}
	require(failed && !cache.status(surface).faulted && !cache.status(surface).spilled && cache.status(surface).initialized, "Failed GPU write lost its exact prior native backup");
	cache.recover(surface);
	cache.prepare(surface);
	require(cache.status(surface).initialized && !cache.status(surface).faulted, "Failed upload/draw didn't recover through guest backing");
	std::cout << "Independent native draw, encode cancellation, pending timeout transaction, GPU failure and completed-slot recovery passed\n";
}

void background(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 2 << 20);
	std::size_t count = 0;
	for (const bool depth : {false, true})
		for (const unsigned samples : {1u, 2u, 4u})
			for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			{
				if (depth && raw > 4)
					continue;
				cached_surface_description description;
				description.address = 0x10000;
				description.allocation.width = 4;
				description.allocation.height = 3;
				description.allocation.samples = samples;
				if (depth)
					description.allocation.format = static_cast<surface_depth>(raw);
				else
					description.allocation.format = static_cast<surface_color>(raw);
				const auto size = samples == 1 ? std::array<std::size_t, 2>{4, 3} : sample_expanded_size(4, 3, samples);
				const auto bytes = bpp(description.allocation);
				description.row_pitch = (size[0] + 2) * bytes;
				std::vector<std::byte> expected(description.row_pitch * size[1], std::byte{0xa5});
				memory->write({description.address, expected.size()}, expected);
				const auto surface = cache.acquire(description);
				cache.prepare(surface, surface_initialization::background);
				require(cache.status(surface).dirty, "Background initialization didn't mark new native content dirty");
				cache.flush(surface);
				for (std::size_t y = 0; y < size[1]; ++y)
					for (std::size_t x = 0; x < size[0]; ++x)
					{
						const auto offset = y * description.row_pitch + x * bytes;
						if (depth)
							store(std::span(expected).subspan(offset), raw == 1 ? 0xffffu : raw == 3 ? 0x7000u
																						: raw == 2	 ? 0xffffffffu
																									 : 0x7f0000ffu,
								  bytes);
						else if (raw == 11 || raw == 12 || raw == 13)
						{
							const unsigned element = raw == 11 ? 2 : 4;
							for (unsigned channel = 0; channel < bytes / element; ++channel)
								store(std::span(expected).subspan(offset + channel * element), channel == 3 ? (raw == 11 ? 0x3c00u : 0x3f800000u) : 0, element);
						}
						else
							store(std::span(expected).subspan(offset), raw <= 2 ? 0x8000u : raw == 3 || raw == 9 || raw == 10 ? 0u
																															  : 0xff000000u,
								  bytes);
					}
				require(memory->read({description.address, expected.size()}).bytes == expected, "Native background values or padding differ from VK initialization");
				cache.erase(surface);
				++count;
			}
	std::cout << count << " native background color/depth/stencil initializations across 18 formats and 1/2/4 samples passed\n";
}

void native_write_versions(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	clear_engine damage(ctx);
	const std::array<unsigned, 1> error_word{1};
	const auto status = make_upload(ctx, std::as_bytes(std::span(error_word)));
	std::size_t cases = 0;
	for (const bool depth_target : {false, true})
		for (const unsigned samples : {1u, 2u, 4u})
			for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			{
				if (depth_target && raw > 4) continue;
				surface_cache cache(ctx, memory, 8 << 20);
				cached_surface_description description;
				description.address = 0x10000;
				description.allocation = {4, 4, samples, depth_target ? surface_format(static_cast<surface_depth>(raw)) : surface_format(static_cast<surface_color>(raw)), samples > 1};
				const auto extent = samples == 1 ? std::array<std::size_t, 2>{4, 4} : sample_expanded_size(4, 4, samples);
				const auto bytes = extent[0] * extent[1] * bpp(description.allocation);
				memory->write({description.address, bytes}, std::vector<std::byte>(bytes));
				const auto target = cache.acquire(description);
				cache.prepare(target);
				const std::array writes{cached_surface_use{target, true}};
				cache.draw(writes, [&](context& commands, auto textures)
				{
					auto pass = [MTL4RenderPassDescriptor new];
					if (depth_target)
					{
						pass.depthAttachment.texture = textures[0];
						pass.depthAttachment.loadAction = MTLLoadActionClear;
						pass.depthAttachment.storeAction = MTLStoreActionStore;
						pass.depthAttachment.clearDepth = .5;
						if (describe_surface_depth(static_cast<surface_depth>(raw)).stencil)
						{
							pass.stencilAttachment.texture = textures[0];
							pass.stencilAttachment.loadAction = MTLLoadActionClear;
							pass.stencilAttachment.storeAction = MTLStoreActionStore;
							pass.stencilAttachment.clearStencil = 0x5a;
						}
					}
					else
					{
						pass.colorAttachments[0].texture = textures[0];
						pass.colorAttachments[0].loadAction = MTLLoadActionClear;
						pass.colorAttachments[0].storeAction = MTLStoreActionStore;
						pass.colorAttachments[0].clearColor = MTLClearColorMake(.25, .5, .75, 1);
					}
					(void)commands.render(pass);
				});
				cache.flush(target);
				const auto expected = memory->read({description.address, bytes}).bytes;
				const auto generation = cache.status(target).content_generation;
				const auto before = cache.transaction_statistics();
				bool rejected = false;
				std::string rejection;
				try
				{
					cache.draw(writes, [&](context& commands, auto textures)
					{
						clear_request request;
						request.mask = 0xf3; request.color_value = 0; request.zstencil_value = 0;
						request.stencil_write_mask = 255; request.rectangle = {1, 1, 2, 2};
						if (depth_target) { request.depth = {textures[0], 0, 0}; request.depth_format = static_cast<surface_depth>(raw); }
						else request.colors[0] = {{textures[0], 0, 0}, static_cast<surface_color>(raw)};
						if (!depth_target && !decode_color_clear(static_cast<surface_color>(raw), 0, 0xf0).write_mask)
						{
							// Guest clears intentionally do nothing on float targets.
							// Damage those native surfaces with a real load clear.
							auto pass = [MTL4RenderPassDescriptor new];
							pass.colorAttachments[0].texture = textures[0];
							pass.colorAttachments[0].loadAction = MTLLoadActionClear;
							pass.colorAttachments[0].storeAction = MTLStoreActionStore;
							pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
							(void)commands.render(pass);
						}
						else (void)damage.run(commands, request);
						commands.require_gpu_success(status, 0, @"Reject actual partial versioned output");
					});
				}
				catch (const std::runtime_error& error) { rejected = true; rejection = error.what(); }
				const auto after = cache.transaction_statistics();
				const auto state = cache.status(target);
				if (!(rejected && state.initialized && !state.faulted && state.content_generation == generation &&
					after.submitted == before.submitted + 1 && after.native_versions == before.native_versions + 1 &&
					after.native_version_blits == before.native_version_blits + 1 && after.raw_backup_submissions == before.raw_backup_submissions &&
					after.raw_backup_bytes == before.raw_backup_bytes))
					std::cerr << "Version case depth=" << depth_target << " raw=" << raw << " samples=" << samples << " rejection=" << rejection
						<< " state=" << state.initialized << ',' << state.faulted << ',' << state.content_generation << '/' << generation
						<< " deltas submitted/versions/blits/raw=" << after.submitted - before.submitted << ',' << after.native_versions - before.native_versions
						<< ',' << after.native_version_blits - before.native_version_blits << ',' << after.raw_backup_submissions - before.raw_backup_submissions << '\n';
				require(rejected && cache.status(target).initialized && !cache.status(target).faulted &&
					cache.status(target).content_generation == generation && after.submitted == before.submitted + 1 &&
					after.native_versions == before.native_versions + 1 && after.native_version_blits == before.native_version_blits + 1 &&
					after.raw_backup_submissions == before.raw_backup_submissions && after.raw_backup_bytes == before.raw_backup_bytes,
					"Versioned native write used extra raw backup or lost its prior state");
				cache.recover(target);
				cache.flush(target);
				require(memory->read({description.address, bytes}).bytes == expected, "Rejected native write changed guest bytes");
				// Force a native readback of retained contents, even though the
				// original clean VM bytes already match the expected image.
				cache.draw(writes, [](context&, auto) {});
				cache.flush(target);
				require(memory->read({description.address, bytes}).bytes == expected, "Native rollback changed color/depth/stencil/sample bits");
				const auto full_before = cache.transaction_statistics();
				cache.clear(target, 0xf3, 0x12345678, 0, 255, {0, 0, 4, 4});
				const auto full_after = cache.transaction_statistics();
				const auto full_versions = depth_target || decode_color_clear(static_cast<surface_color>(raw), 0x12345678, 0xf0).write_mask ? 1u : 0u;
				require(full_after.native_versions == full_before.native_versions + full_versions && full_after.native_version_blits == full_before.native_version_blits &&
					full_after.raw_backup_submissions == full_before.raw_backup_submissions, "Full overwrite copied old target contents");
				++cases;
			}
	require(cases == 54, "Versioned rollback omitted a format/sample case");
	std::cout << "Native target versions:54 color/depth/stencil cases at1/2/4 samples;actual partial/native-float GPU-failure rollback,one submission,no raw backup;full overwrite omits preservation blit passed\n";
}

void pressure_and_native_bits(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	const surface_allocation_description allocation{16, 16, 1, surface_color::argb8, false};
	std::size_t cost = 0;
	{
		surface_pool probe(ctx, 1 << 20);
		auto lease = probe.acquire(allocation);
		cost = lease->reserved_bytes();
	}
	surface_cache cache(ctx, memory, cost);
	cached_surface_description a{0x20000, allocation, 0, {}};
	auto b = a;
	b.address = 0x30000;
	const std::vector<std::byte> seed(1024, std::byte{0xa5});
	memory->write({a.address, 1024}, seed);
	memory->write({b.address, 1024}, seed);
	const auto first = cache.acquire(a), second = cache.acquire(b);
	cache.clear(first, 0xf0, 0x11223344, 0, 255, {0, 0, 16, 16});
	const std::array too_many{cached_surface_use{first, true}, cached_surface_use{second, true}};
	bool refused = false;
	try
	{
		cache.draw(too_many, [](context&, auto) {});
	}
	catch (const surface_budget_error&)
	{
		refused = true;
	}
	require(refused && cache.status(first).initialized && cache.status(first).dirty, "Draw preparation spilled a pinned target under insufficient budget");
	cache.prepare(second);
	require(cache.status(first).spilled && cache.status(first).dirty && cache.memory_statistics().reserved_bytes <= cost, "Budget pressure didn't spill dirty native content without guest publication");
	require(memory->read({a.address, 1024}).bytes == seed, "Automatic native spill wrote guest memory");
	cache.prepare(first);
	require(!cache.status(second).initialized && !cache.status(second).spilled && cache.memory_statistics().reserved_bytes <= cost,
		"Single-heap recovery failed to discard reconstructible clean backing");
	cache.flush(first);
	const auto bytes = memory->read({a.address, 1024}).bytes;
	for (std::size_t pixel = 0; pixel < 256; ++pixel)
		require(bytes[pixel * 4] == std::byte{0x11} && bytes[pixel * 4 + 1] == std::byte{0x22} && bytes[pixel * 4 + 2] == std::byte{0x33} && bytes[pixel * 4 + 3] == std::byte{0x44}, "Pressure spill/reload lost dirty pixels");
	cache.erase(first);
	cache.prepare(second);
	const std::array untouched{cached_surface_use{second, true}};
	const auto constrained_before = cache.transaction_statistics();
	cache.draw(untouched, [](context&, auto) {});
	require(cache.transaction_statistics().raw_backup_submissions == constrained_before.raw_backup_submissions + 1 &&
		cache.transaction_statistics().raw_backup_bytes > constrained_before.raw_backup_bytes,
		"One-heap writer failed to retain bounded raw rollback fallback");
	cache.flush(second);
	require(memory->read({b.address, 1024}).bytes == seed, "Reconstructed clean surface lost guest pixels");
	cache.erase(second);
	std::cout << "Pinned draw budget refusal,bounded raw rollback fallback,dirty spill/reload andclean discard/reconstruction under one native color-heap budget passed\n";

	// Keep noncanonical f32 depth bits in the native spill, rather than
	// contracting them through the guest E4M12 / float24 codecs.
	surface_cache raw_cache(ctx, memory, 1 << 20);
	for (const auto format : {surface_depth::z16_float, surface_depth::z24s8_float})
	{
		const std::array words{0x3eaaaaabu, 0x40000000u, 0x7fc12345u, 0x3f000007u};
		const auto input = make_upload(ctx, std::as_bytes(std::span(words)));
		const auto output = make_buffer(ctx, 16, buffer_storage::shared);
		cached_surface_description description;
		description.address = 0x40000;
		description.allocation = {4, 1, 1, format, false};
		const auto width_bytes = format == surface_depth::z16_float ? 8u : 16u;
		std::vector<std::byte> zeros(width_bytes);
		memory->write({description.address, width_bytes}, zeros);
		const auto surface = raw_cache.acquire(description);
		raw_cache.prepare(surface);
		const std::array write{cached_surface_use{surface, true}}, read{cached_surface_use{surface, false}};
		raw_cache.draw(write, [&](context& commands, auto textures)
					   {
						   texture_region region;
						   region.size = MTLSizeMake(4, 1, 1);
						   buffer_to_texture(commands, input, 0, textures[0], region, 16, 0, texture_plane::depth); });
		const auto before_failure = raw_cache.transaction_statistics();
		const std::array<unsigned, 1> rejected_status{1};
		const auto bad_status = make_upload(ctx, std::as_bytes(std::span(rejected_status)));
		clear_engine damage(ctx);
		bool rejected_write = false;
		try
		{
			raw_cache.draw(write, [&](context& commands, auto textures)
			{
				clear_request request;
				request.depth = {textures[0], 0, 0};
				request.depth_format = format;
				request.mask = 1; request.zstencil_value = 0; request.rectangle = {0, 0, 2, 1};
				(void)damage.run(commands, request);
				commands.require_gpu_success(bad_status, 0, @"Reject partially overwritten native depth");
			});
		}
		catch (const std::runtime_error&) { rejected_write = true; }
		require(rejected_write && raw_cache.status(surface).initialized && raw_cache.status(surface).dirty && !raw_cache.status(surface).faulted &&
			raw_cache.transaction_statistics().raw_backup_submissions == before_failure.raw_backup_submissions,
			"Native version rollback did not preserve unquantized depth");
		raw_cache.recover(surface);
		raw_cache.spill(surface);
		raw_cache.prepare(surface);
		raw_cache.draw(read, [&](context& commands, auto textures)
					   {
						   texture_region region;
						   region.size = MTLSizeMake(4, 1, 1);
						   texture_to_buffer(commands, textures[0], region, output, 0, 16, 0, texture_plane::depth); });
		const auto actual = map_buffer(output, 0, 16);
		require(std::equal(actual.begin(), actual.end(), std::as_bytes(std::span(words)).begin()), "Raw native float-depth spill contracted or clamped bits");
		raw_cache.erase(surface);
	}
	bool rejected = false;
	try
	{
		auto invalid = a;
		invalid.allocation.width = max_2d_texture_dimension(ctx.device()) + 1;
		(void)cache.acquire(invalid);
	}
	catch (const std::invalid_argument&)
	{
		rejected = true;
	}
	require(rejected, "Cached surface dimensions bypassed native Apple family limit");
	std::cout << "Automatic dirty spill/reload under one-heap budget,exact native f32 depth/NaN rollback after actual partial GPU failure andGPU dimension preflight passed\n";
}
void active_alias_mrts(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 2 << 20);
	for (const bool swizzled : {false, true})
	{
		cached_surface_description a, b;
		a.address = swizzled ? 0x60000 : 0x50000;
		a.allocation = {swizzled ? 4u : 2u, swizzled ? 4u : 3u, 1, surface_color::argb8, false};
		if (swizzled)
			a.morton = swizzle_region{{8, 8, 4}, {1, 1, 0}, {4, 4, 1}};
		else
			a.row_pitch = 16;
		b = a;
		if (swizzled)
			b.morton->origin[2] = 1;
		else
			b.address += 8;
		const guest_surface_range storage{a.address, swizzled ? 1024u : 56u};
		std::vector<std::byte> expected(storage.size, std::byte{0xa5});
		memory->write(storage, expected);
		const auto first = cache.acquire(a), second = cache.acquire(b);
		const std::array uses{cached_surface_use{first, true}, cached_surface_use{second, true}};
		cache.draw(uses, [](context& commands, auto textures)
				   {
					   MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
					   for (NSUInteger slot = 0; slot < 2; ++slot)
					   {
						   pass.colorAttachments[slot].texture = textures[slot];
						   pass.colorAttachments[slot].loadAction = MTLLoadActionClear;
						   pass.colorAttachments[slot].storeAction = MTLStoreActionStore;
					   }
					   pass.colorAttachments[0].clearColor = MTLClearColorMake(0x22 / 255.0, 0x33 / 255.0, 0x44 / 255.0, 0x11 / 255.0);
					   pass.colorAttachments[1].clearColor = MTLClearColorMake(0x66 / 255.0, 0x77 / 255.0, 0x88 / 255.0, 0x55 / 255.0);
					   (void)commands.render(pass);
					   commands.end_encoding(); });
		require(cache.status(first).initialized && cache.status(first).dirty && cache.status(second).initialized && cache.status(second).dirty, "Disjoint active MRTs invalidated each other");
		const auto offset = [&](unsigned surface, std::size_t x, std::size_t y)
		{
			return swizzled ? morton(x + 1, y + 1, surface, {8, 8, 4}) * 4 : surface * 8 + y * 16 + x * 4;
		};
		if (swizzled)
		{
			const auto unused = morton(2, 2, 2, {8, 8, 4}) * 4;
			memory->before_compare = [&, unused]
			{
				const std::array patch{std::byte{0xee}};
				memory->write({storage.address + static_cast<std::uint32_t>(unused), 1}, patch);
			};
			expected[unused] = std::byte{0xee};
		}
		cache.flush(first);
		require(cache.status(second).initialized && cache.status(second).dirty, "First MRT writeback changed disjoint second surface revision");
		cache.flush(second);
		for (unsigned surface = 0; surface < 2; ++surface)
			for (std::size_t y = 0; y < a.allocation.height; ++y)
				for (std::size_t x = 0; x < a.allocation.width; ++x)
					store(std::span(expected).subspan(offset(surface, x, y)), surface ? 0x55667788 : 0x11223344, 4);
		require(memory->read(storage).bytes == expected, "Disjoint active MRT writeback overwrote unrelated cells or lost either surface");
		if (!swizzled)
		{
			cache.clear(first, 0xf0, 0x11223344, 0, 255, {0, 0, 2, 3});
			cache.clear(second, 0xf0, 0x55667788, 0, 255, {0, 0, 2, 3});
			const std::array patch{std::byte{0xcc}};
			cache.cpu_write({b.address, 1}, patch);
			require(cache.status(first).initialized && cache.status(first).dirty && !cache.status(second).initialized, "CPU write into one MRT flushed/invalidated disjoint active bytes");
			cache.flush(first);
			expected[8] = std::byte{0xcc};
			require(memory->read(storage).bytes == expected, "Padding-adjacent CPU patch was overwritten by other MRT flush");
		}
		cache.erase(first);
		cache.erase(second);
	}
	std::cout << "Native MRTs with interleaved pitched active rows and separate Morton slices preserve revisions, inactive concurrent writes and CPU patches\n";
}
void native_cache_copies(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 4 << 20);
	std::size_t cases = 0;
	for (const unsigned samples : {1u, 2u, 4u})
	{
		cached_surface_description a;
		a.address = 0x70000;
		a.allocation = {8, 6, samples, surface_color::argb8, false};
		auto b = a;
		b.address = 0x80000;
		const auto size = samples == 1 ? std::array<std::size_t, 2>{8, 6} : sample_expanded_size(8, 6, samples);
		a.row_pitch = b.row_pitch = (size[0] + 3) * 4;
		const auto range = guest_surface_range{a.address, a.row_pitch * size[1]}, br = guest_surface_range{b.address, b.row_pitch * size[1]};
		std::vector<std::byte> original(range.size, std::byte{0xa5}), target(range.size, std::byte{0xcc});
		for (std::size_t y = 0; y < size[1]; ++y)
			for (std::size_t x = 0; x < size[0]; ++x)
				store(std::span(original).subspan(y * a.row_pitch + x * 4), static_cast<std::uint32_t>(0x16325070 + x + y * size[0]), 4);
		memory->write(range, original);
		memory->write(br, target);
		const auto source = cache.acquire(a), destination = cache.acquire(b);
		cache.copy(source, {0, 0, 4, 4}, destination, 2, 1);
		require(cache.status(destination).dirty && memory->read(br).bytes == target, "Cache region copy published guest data early");
		cache.flush(destination);
		auto expected = target;
		const std::size_t sx = samples == 1 ? 1 : 2, sy = samples == 4 ? 2 : 1;
		for (std::size_t y = sy; y < 5 * sy; ++y)
			for (std::size_t x = 2 * sx; x < 6 * sx; ++x)
				std::copy_n(original.begin() + static_cast<std::ptrdiff_t>((y - sy) * a.row_pitch + (x - 2 * sx) * 4), 4, expected.begin() + static_cast<std::ptrdiff_t>(y * b.row_pitch + x * 4));
		require(memory->read(br).bytes == expected, "Cached native region copy or outside pixels/padding differ");
		cache.copy(source, {0, 0, 4, 4}, source, 2, 1);
		cache.flush(source);
		expected = original;
		for (std::size_t y = sy; y < 5 * sy; ++y)
			for (std::size_t x = 2 * sx; x < 6 * sx; ++x)
				std::copy_n(original.begin() + static_cast<std::ptrdiff_t>((y - sy) * a.row_pitch + (x - 2 * sx) * 4), 4, expected.begin() + static_cast<std::ptrdiff_t>(y * a.row_pitch + x * 4));
		require(memory->read(range).bytes == expected, "Cached native overlapping self-copy differs");
		cache.erase(source);
		cache.erase(destination);
		cases += 2;
	}
	// Full native copy must not import invalid old float-depth samples that it
	// replaces completely. Only the source is uploaded in this path.
	cached_surface_description a;
	a.address = 0x70000;
	a.allocation = {4, 4, 4, surface_depth::z16_float, false};
	auto b = a;
	b.address = 0x80000;
	std::vector<std::byte> source_bytes(8 * 8 * 2), invalid(source_bytes.size());
	for (std::size_t i = 0; i < source_bytes.size(); i += 2)
	{
		store(std::span(source_bytes).subspan(i), 0x6000, 2);
		store(std::span(invalid).subspan(i), 0x8000, 2);
	}
	memory->write({a.address, source_bytes.size()}, source_bytes);
	memory->write({b.address, invalid.size()}, invalid);
	const auto source = cache.acquire(a), destination = cache.acquire(b);
	cache.copy(source, {0, 0, 4, 4}, destination, 0, 0);
	cache.flush(destination);
	require(memory->read({b.address, source_bytes.size()}).bytes == source_bytes, "Full MSAA depth copy loaded/clamped invalid old destination pixels");
	std::cout << cases << " cached native region/self-copy cases and full MSAA depth overwrite without old guest import passed\n";
}
void deferred_formats(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 4 << 20);
	std::size_t count = 0;
	for (const bool depth : {false, true})
		for (const unsigned samples : {1u, 2u, 4u})
			for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
			{
				if (depth && raw > 4)
					continue;
				cached_surface_description a;
				a.address = 0x70000;
				a.allocation = {8, 6, samples, surface_color::argb8, false};
				if (depth)
					a.allocation.format = static_cast<surface_depth>(raw);
				else
					a.allocation.format = static_cast<surface_color>(raw);
				const auto size = samples == 1 ? std::array<std::size_t, 2>{8, 6} : sample_expanded_size(8, 6, samples);
				const auto bytes = bpp(a.allocation);
				a.row_pitch = (size[0] + 3) * bytes;
				auto b = a, d = a;
				b.address = 0x80000;
				d.address = 0x90000;
				const auto layout = plan_guest_surface(a.address, size[0], size[1], bytes, a.row_pitch);
				const auto range = layout.range;
				const guest_surface_range br{b.address, range.size}, dr{d.address, range.size};
				std::vector<std::byte> first(range.size, std::byte{0xa5}), second(range.size, std::byte{0x5a}), target(range.size, std::byte{0xcc});
				for (std::size_t y = 0; y < size[1]; ++y)
					for (std::size_t x = 0; x < size[0]; ++x)
					{
						const auto pixel = y * size[0] + x, offset = y * a.row_pitch + x * bytes;
						if (!depth && (raw == 11 || raw == 12 || raw == 13))
						{
							const auto element = raw == 11 ? 2u : 4u;
							for (unsigned c = 0; c < bytes / element; ++c)
							{
								store(std::span(first).subspan(offset + c * element), raw == 11 ? 0x3800 : 0x3f000000, element);
								store(std::span(second).subspan(offset + c * element), raw == 11 ? 0x3a00 : 0x3f400000, element);
							}
						}
						else
						{
							store(std::span(first).subspan(offset), seed_word(depth, raw, pixel), bytes);
							store(std::span(second).subspan(offset), seed_word(depth, raw, pixel + 800), bytes);
						}
					}
				memory->write(range, first);
				memory->write(br, second);
				memory->write(dr, target);
				const auto source = cache.acquire(a), other = cache.acquire(b), destination = cache.acquire(d);
				cache.defer_copy(source, {0, 0, 3, 3}, destination, 0, 0);
				cache.defer_copy(other, {0, 0, 8, 6}, destination, 0, 0);
				cache.defer_copy(source, {0, 0, 4, 4}, destination, 2, 1);
				require(cache.status(destination).deferred_copies == 3 && !cache.status(destination).initialized && !cache.status(destination).dirty,
						"Deferred copy initialized or dirtied destination before execution");
				require(memory->read(dr).bytes == target, "Deferred scheduling published guest bytes");
				cache.flush(destination);
				auto expected = target;
				const std::size_t sx = samples == 1 ? 1 : 2, sy = samples == 4 ? 2 : 1;
				for (std::size_t y = 0; y < size[1]; ++y)
					for (std::size_t x = 0; x < size[0]; ++x)
					{
						const bool partial = x >= 2 * sx && x < 6 * sx && y >= sy && y < 5 * sy;
						const auto from = partial ? (y - sy) * a.row_pitch + (x - 2 * sx) * bytes : y * a.row_pitch + x * bytes;
						const auto& input = partial ? first : second;
						std::copy_n(input.begin() + from, bytes, expected.begin() + y * d.row_pitch + x * bytes);
					}
				require(memory->read(dr).bytes == expected && cache.status(destination).deferred_copies == 0,
						"Ordered deferred full/partial copy, outside pixels or guest padding differs");
				cache.erase(source);
				cache.erase(other);
				cache.erase(destination);
				++count;
			}
	std::cout << count << " ordered deferred native copy batches across 18 formats and 1/2/4 samples passed\n";
}
void deferred_lifetime(context& ctx)
{
	for (const unsigned samples : {1u, 2u, 4u})
		for (const unsigned action : {0u, 1u, 2u, 3u, 4u, 5u})
		{
			auto memory = std::make_shared<memory_fixture>();
			surface_cache cache(ctx, memory, 4 << 20);
			cached_surface_description a;
			a.address = 0x70000;
			a.allocation = {8, 6, samples, surface_color::argb8, false};
			auto b = a;
			b.address = 0x80000;
			const auto size = samples == 1 ? std::array<std::size_t, 2>{8, 6} : sample_expanded_size(8, 6, samples);
			a.row_pitch = b.row_pitch = (size[0] + 3) * 4;
			const guest_surface_range ar{a.address, a.row_pitch * size[1]}, br{b.address, b.row_pitch * size[1]};
			std::vector<std::byte> original(ar.size, std::byte{0xa5}), target(ar.size, std::byte{0xcc});
			for (std::size_t y = 0; y < size[1]; ++y)
				for (std::size_t x = 0; x < size[0]; ++x)
					store(std::span(original).subspan(y * a.row_pitch + x * 4), static_cast<std::uint32_t>(0x16325070 + x + y * size[0]), 4);
			memory->write(ar, original);
			memory->write(br, target);
			auto source = cache.acquire(a);
			const auto destination = cache.acquire(b);
			cache.defer_copy(source, {0, 0, 4, 4}, destination, 2, 1);
			const auto generation = cache.status(source).content_generation;
			if (action == 1)
			{
				cache.erase(source);
				const auto replacement = cache.acquire(a);
				require(replacement != source, "Erased deferred source identity was reused");
				source = replacement;
			}
			if (action == 2)
			{
				cache.spill(source);
				require(cache.status(source).spilled && cache.memory_statistics().leased_allocations == 1,
						"Deferred source lease disappeared after native spill");
				cache.prepare(source);
				require(cache.status(source).content_generation == generation, "Spill/reload changed logical content generation");
			}
			if (action == 3)
			{
				// Host invalidation leaves the captured native epoch intact. A later
				// upload must materialize its readers before reusing that allocation.
				auto changed = original;
				store(changed, 0x778899aa, 4);
				cache.cpu_write(ar, changed);
				cache.prepare(source);
				require(cache.status(destination).deferred_copies == 0 && cache.status(destination).dirty,
						"Source guest reupload changed an unexecuted captured epoch");
			}
			if (action == 4)
			{
				auto c = a;
				c.address = 0x90000;
				memory->write({c.address, target.size()}, target);
				cache.copy(cache.acquire(c), {0, 0, 8, 6}, source, 0, 0);
				require(cache.status(destination).deferred_copies == 0, "Native copy modified an unconsumed captured source");
			}
			if (action == 5)
			{
				const std::array uses{cached_surface_use{source, true}};
				cache.draw(uses, [](context& commands, auto textures)
						   {
							   auto pass = [MTL4RenderPassDescriptor new];
							   pass.colorAttachments[0].texture = textures[0];
							   pass.colorAttachments[0].loadAction = MTLLoadActionClear;
							   pass.colorAttachments[0].storeAction = MTLStoreActionStore;
							   pass.colorAttachments[0].clearColor = MTLClearColorMake(1, 0, 0, 1);
							   (void)commands.render(pass);
							   commands.end_encoding(); });
				require(cache.status(destination).deferred_copies == 1 && !cache.status(destination).dirty,
					"Private native draw published an unrelated deferred reader");
			}
			cache.clear(source, 0xf0, 0x11223344, 0, 255, {0, 0, 8, 6});
			if (action == 0)
				require(cache.status(destination).deferred_copies == 0 && cache.status(source).content_generation > generation,
						"In-place native clear did not materialize frozen source readers");
			if (action == 1 || action == 2 || action == 5)
				require(cache.status(destination).deferred_copies == 1,
						"Replacement allocation altered a frozen retired native source epoch");
			cache.flush(destination);
			auto expected = target;
			const std::size_t sx = samples == 1 ? 1 : 2, sy = samples == 4 ? 2 : 1;
			for (std::size_t y = sy; y < 5 * sy; ++y)
				for (std::size_t x = 2 * sx; x < 6 * sx; ++x)
					std::copy_n(original.begin() + (y - sy) * a.row_pitch + (x - 2 * sx) * 4, 4, expected.begin() + y * b.row_pitch + x * 4);
			require(memory->read(br).bytes == expected, "Deferred source overwrite/erase/spill/reupload copied the wrong content epoch");
		}
	std::cout << "18 frozen native source clear/copy/draw/erase/spill/reupload lifetimes passed\n";
}
void deferred_conflicts_and_overwrite(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 4 << 20);
	cached_surface_description a;
	a.address = 0x70000;
	a.allocation = {8, 6, 1, surface_color::argb8, false};
	auto b = a;
	b.address = 0x80000;
	const guest_surface_range ar{a.address, 8 * 6 * 4}, br{b.address, 8 * 6 * 4};
	std::vector<std::byte> old(ar.size, std::byte{0xa5}), original(ar.size, std::byte{0xcc});
	memory->write(ar, old);
	memory->write(br, original);
	const auto source = cache.acquire(a), destination = cache.acquire(b);
	cache.defer_copy(source, {0, 0, 8, 6}, destination, 0, 0);
	auto changed = original;
	changed[0] = std::byte{0x19};
	memory->write(br, changed);
	bool conflict = false;
	try
	{
		cache.prepare(destination);
	}
	catch (const surface_conflict_error&)
	{
		conflict = true;
	}
	require(conflict && memory->read(br).bytes == changed && cache.status(destination).deferred_copies == 1,
			"External VM write was overwritten or lost the deferred plan on conflict");
	cache.cancel_deferred_copies(destination);
	cache.defer_copy(source, {0, 0, 8, 6}, destination, 0, 0);
	cache.clear(destination, 0xf0, 0x11223344, 0, 255, {0, 0, 8, 6});
	require(cache.status(destination).deferred_copies == 0 && cache.status(destination).content_generation == 1,
			"Full clear evaluated superseded destination copy plans");
	cache.flush(destination);
	auto expected = original;
	for (std::size_t i = 0; i < expected.size(); i += 4)
		store(std::span(expected).subspan(i), 0x11223344, 4);
	require(memory->read(br).bytes == expected, "Full overwrite of deferred destination differs");
	cache.defer_copy(source, {0, 0, 8, 6}, destination, 0, 0);
	cache.clear(destination, 0x10, 0x001a0000, 0, 255, {2, 1, 3, 3});
	cache.flush(destination);
	expected = old;
	for (std::size_t y = 1; y < 4; ++y)
		for (std::size_t x = 2; x < 5; ++x)
			expected[(y * 8 + x) * 4 + 1] = std::byte{0x1a};
	require(memory->read(br).bytes == expected, "Partial clear did not consume deferred contents or retain their guest revision");
	// Use a nonuniform source to prove overlapping self-copy snapshot semantics.
	for (std::size_t i = 0; i < old.size(); i += 4)
		store(std::span(old).subspan(i), static_cast<std::uint32_t>(0x12345000 + i / 4), 4);
	cache.cpu_write(ar, old);
	cache.defer_copy(source, {0, 0, 4, 4}, source, 2, 1);
	auto self_expected = old;
	for (std::size_t y = 1; y < 5; ++y)
		for (std::size_t x = 2; x < 6; ++x)
			std::copy_n(old.begin() + ((y - 1) * 8 + x - 2) * 4, 4, self_expected.begin() + (y * 8 + x) * 4);
	cache.flush(source);
	require(memory->read(ar).bytes == self_expected, "Deferred overlapping self-copy differs");
	// A partial MSAA copy needs the old destination pixels. Invalid raw float
	// depth must fault the upload without discarding the captured native source.
	cached_surface_description z;
	z.address = 0x90000;
	z.allocation = {4, 4, 4, surface_depth::z16_float, false};
	auto zd = z;
	zd.address = 0xa0000;
	std::vector<std::byte> valid(8 * 8 * 2), invalid(valid.size());
	for (std::size_t i = 0; i < valid.size(); i += 2)
	{
		store(std::span(valid).subspan(i), 0x6000, 2);
		store(std::span(invalid).subspan(i), 0x8000, 2);
	}
	memory->write({z.address, valid.size()}, valid);
	memory->write({zd.address, invalid.size()}, invalid);
	const auto depth_source = cache.acquire(z), depth_target = cache.acquire(zd);
	cache.defer_copy(depth_source, {0, 0, 2, 2}, depth_target, 1, 1);
	bool failed = false;
	try
	{
		cache.prepare(depth_target);
	}
	catch (const std::runtime_error&)
	{
		failed = true;
	}
	require(failed && cache.status(depth_target).faulted && cache.status(depth_target).deferred_copies == 1 &&
				memory->read({zd.address, invalid.size()}).bytes == invalid,
			"Failed deferred destination upload discarded captured copies or published invalid guest data");
	cache.recover(depth_target);
	cache.copy(depth_source, {0, 0, 4, 4}, depth_target, 0, 0);
	cache.flush(depth_target);
	require(memory->read({zd.address, valid.size()}).bytes == valid && cache.status(depth_target).deferred_copies == 0 && !cache.status(depth_target).faulted,
			"Full native overwrite did not recover failed deferred destination without old float-depth import");
	std::cout << "Deferred VM conflict/cancellation, full overwrite, partial clear revision and self-copy passed\n";
	std::cout << "Failed deferred MSAA guest import preserves captured plan; full native overwrite recovers it\n";
}
void deferred_pending_and_budget()
{
	@autoreleasepool
	{
		context ctx(2);
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 4 << 20);
		cached_surface_description a;
		a.address = 0x70000;
		a.allocation = {8, 6, 1, surface_color::argb8, false};
		auto b = a;
		b.address = 0x80000;
		std::vector<std::byte> old(8 * 6 * 4, std::byte{0xa5}), target(old.size(), std::byte{0xcc});
		memory->write({a.address, old.size()}, old);
		memory->write({b.address, old.size()}, target);
		const auto source = cache.acquire(a), destination = cache.acquire(b);
		cache.defer_copy(source, {0, 0, 8, 6}, destination, 0, 0);
		id<MTLSharedEvent> event = [ctx.device() newSharedEvent];
		struct event_guard
		{
			id<MTLSharedEvent> event;
			~event_guard()
			{
				event.signaledValue = 1;
			}
		} release{event};
		ctx.begin();
		ctx.wait_event(event, 1);
		const auto blocked = ctx.submit();
		cache.set_wait_timeout(std::chrono::milliseconds(1));
		bool timed_out = false;
		try
		{
			cache.prepare(destination);
		}
		catch (const std::runtime_error&)
		{
			timed_out = true;
		}
		require(timed_out && cache.status(destination).pending && cache.status(destination).deferred_copies == 1 &&
					!cache.status(destination).initialized && cache.status(destination).content_generation == 0,
				"Deferred timeout published or discarded uncompleted copy metadata");
		event.signaledValue = 1;
		blocked.wait();
		cache.synchronize();
		cache.set_wait_timeout(std::chrono::seconds(30));
		require(cache.status(destination).dirty && cache.status(destination).deferred_copies == 0 &&
					memory->read({b.address, old.size()}).bytes == target,
				"Deferred completion did not atomically publish native copy metadata");
		cache.flush(destination);
		require(memory->read({b.address, old.size()}).bytes == old, "Completed pending deferred copy differs");
		surface_pool probe(ctx, 4 << 20);
		auto allocation = probe.acquire(a.allocation);
		const auto budget = allocation->reserved_bytes();
		allocation.reset();
		(void)probe.trim();
		surface_cache constrained(ctx, memory, budget);
		const auto first = constrained.acquire(a), second = constrained.acquire(b);
		constrained.defer_copy(first, {0, 0, 8, 6}, second, 0, 0);
		constrained.erase(first);
		bool refused = false;
		try
		{
			constrained.prepare(second);
		}
		catch (const surface_budget_error&)
		{
			refused = true;
		}
		require(refused && constrained.status(second).deferred_copies == 1 && !constrained.status(second).faulted &&
					constrained.memory_statistics().reserved_bytes <= budget,
				"Frozen source was reused or lost under insufficient copy destination budget");
		constrained.cancel_deferred_copies(second);
		constrained.prepare(second);
		require(constrained.status(second).initialized && constrained.memory_statistics().reserved_bytes <= budget,
				"Cancelled deferred source did not release its bounded native allocation");
		ctx.wait_idle();
	}
	std::cout << "Deferred actual pending GPU transaction, strict frozen-lease budget refusal and cancellation release passed\n";
}
void deferred_guest_aliases(context& ctx)
{
	for (const bool retiring : {false, true})
	{
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 4 << 20);
		cached_surface_description a;
		a.address = 0x70000;
		a.allocation = {8, 6, 1, surface_color::argb8, false};
		a.row_pitch = 48;
		auto b = a;
		b.address += 4;
		const auto layout = plan_guest_surface(a.address, 8, 6, 4, a.row_pitch);
		std::vector<std::byte> original(layout.range.size + 4, std::byte{0xa5});
		memory->write({a.address, original.size()}, original);
		const auto source = cache.acquire(a), destination = cache.acquire(b);
		cache.clear(source, 0xf0, 0x11223344, 0, 255, {0, 0, 8, 6});
		cache.defer_copy(source, {0, 0, 8, 6}, destination, 0, 0);
		if (retiring)
			cache.erase(source);
		else
			(void)cache.cpu_read(layout.range);
		cache.flush(destination);
		auto expected = original;
		for (std::size_t y = 0; y < 6; ++y)
		{
			// The old owner's first pixel is outside the new active range and
			// still needs canonical writeback; the destination covers pixels 1..8.
			for (std::size_t x = 0; x < 9; ++x)
				store(std::span(expected).subspan(y * 48 + x * 4), 0x11223344, 4);
		}
		require(memory->read({a.address, expected.size()}).bytes == expected && cache.status(destination).deferred_copies == 0,
				"Deferred alias source writeback/erase falsely conflicted or lost old exterior dirty bytes");
	}
	std::cout << "Deferred overlapping guest aliases preserve old exterior dirty pixels on CPU read and source erase\n";
}
void cached_nearest_blits(context& ctx)
{
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 4 << 20);
	const auto nearest = [](std::size_t i, std::size_t source, std::size_t destination, bool flip)
	{
		const auto denominator = 2 * destination;
		const auto value = (2 * i + 1) * source;
		return (flip ? source * denominator - value : value) / denominator;
	};
	std::size_t count = 0;
	for (const bool depth : {false, true})
		for (const unsigned raw : {1u, 2u, 3u, 4u, 5u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u})
		{
			if (depth && raw > 4)
				continue;
			for (const bool deferred : {false, true})
				for (const unsigned flip : {0u, 1u, 2u, 3u})
				{
					cached_surface_description a;
					a.address = 0x70000;
					a.allocation = {8, 6, 1, surface_color::argb8, false};
					if (depth)
						a.allocation.format = static_cast<surface_depth>(raw);
					else
						a.allocation.format = static_cast<surface_color>(raw);
					const auto bytes = bpp(a.allocation);
					a.row_pitch = 11 * bytes;
					auto b = a;
					b.address = 0x80000;
					b.allocation.width = 9;
					b.allocation.height = 7;
					b.row_pitch = 13 * bytes;
					const auto ar = plan_guest_surface(a.address, 8, 6, bytes, a.row_pitch).range;
					const auto br = plan_guest_surface(b.address, 9, 7, bytes, b.row_pitch).range;
					std::vector<std::byte> original(ar.size, std::byte{0xa5}), target(br.size, std::byte{0xcc});
					for (std::size_t y = 0; y < 6; ++y)
						for (std::size_t x = 0; x < 8; ++x)
						{
							const auto offset = y * a.row_pitch + x * bytes;
							if (!depth && (raw == 11 || raw == 12 || raw == 13))
							{
								const auto element = raw == 11 ? 2u : 4u;
								for (unsigned c = 0; c < bytes / element; ++c)
									store(std::span(original).subspan(offset + c * element), static_cast<std::uint32_t>((raw == 11 ? 0x3800u : 0x3f000000u) + x + y * 8 + c), element);
							}
							else
								store(std::span(original).subspan(offset), seed_word(depth, raw, y * 8 + x), bytes);
						}
					memory->write(ar, original);
					memory->write(br, target);
					const auto source = cache.acquire(a), destination = cache.acquire(b);
					const std::array axes{bool(flip & 1), bool(flip & 2), false};
					if (deferred)
					{
						cache.defer_blit_nearest(source, {1, 1, 3, 3}, destination, {2, 1, 5, 4});
						cache.defer_blit_nearest(source, {0, 0, 8, 6}, destination, {0, 0, 9, 7}, texture_plane::all, axes);
						cache.erase(source);
						require(cache.status(destination).deferred_copies == 2 && !cache.status(destination).initialized,
								"Deferred scaled blit executed before use or lost its retired source");
						cache.prepare(destination);
					}
					else
						cache.blit_nearest(source, {0, 0, 8, 6}, destination, {0, 0, 9, 7}, texture_plane::all, axes);
					require(cache.status(destination).content_generation == 1 && memory->read(br).bytes == target,
							"Full scaled overwrite imported old destination or published guest bytes early");
					cache.flush(destination);
					auto expected = target;
					for (std::size_t y = 0; y < 7; ++y)
						for (std::size_t x = 0; x < 9; ++x)
						{
							const auto from = nearest(y, 6, 7, flip & 2) * a.row_pitch + nearest(x, 8, 9, flip & 1) * bytes;
							std::copy_n(original.begin() + from, bytes, expected.begin() + y * b.row_pitch + x * bytes);
						}
					require(memory->read(br).bytes == expected, "Cached scaled/flip/deferred native blit or guest padding differs");
					if (!deferred)
						cache.erase(source);
					cache.erase(destination);
					++count;
				}
		}
	// Partial self-blits must use the original source snapshot while retaining
	// all target pixels outside their smaller destination rectangle.
	for (const bool deferred : {false, true})
	{
		cached_surface_description a;
		a.address = 0x70000;
		a.allocation = {8, 6, 1, surface_color::argb8, false};
		a.row_pitch = 48;
		const auto ar = plan_guest_surface(a.address, 8, 6, 4, a.row_pitch).range;
		std::vector<std::byte> original(ar.size, std::byte{0xa5});
		for (std::size_t y = 0; y < 6; ++y)
			for (std::size_t x = 0; x < 8; ++x)
				store(std::span(original).subspan(y * a.row_pitch + x * 4), static_cast<std::uint32_t>(0x12345000 + y * 8 + x), 4);
		memory->write(ar, original);
		const auto surface = cache.acquire(a);
		if (deferred)
			cache.defer_blit_nearest(surface, {1, 0, 4, 5}, surface, {2, 1, 6, 4}, texture_plane::all, {true, true, false});
		else
			cache.blit_nearest(surface, {1, 0, 4, 5}, surface, {2, 1, 6, 4}, texture_plane::all, {true, true, false});
		cache.flush(surface);
		auto expected = original;
		for (std::size_t y = 0; y < 4; ++y)
			for (std::size_t x = 0; x < 6; ++x)
			{
				const auto from = nearest(y, 5, 4, true) * a.row_pitch + (nearest(x, 4, 6, true) + 1) * 4;
				std::copy_n(original.begin() + from, 4, expected.begin() + (y + 1) * a.row_pitch + (x + 2) * 4);
			}
		require(memory->read(ar).bytes == expected, "Cached scaled self-copy used modified source pixels or changed outside guest bytes");
		cache.erase(surface);
	}
	std::cout << count << " cached direct/deferred 18-format nearest full blits, retired sources and two partial overlapping self-scales preserve exact guest bytes/padding\n";
}
void cached_typeless_blits(context& ctx)
{
	const std::array<surface_format, 18> formats{surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565,
												 surface_color::xrgb8_zero, surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8,
												 surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero,
												 surface_color::xbgr8_one, surface_color::abgr8, surface_depth::z16_uint, surface_depth::z24s8_uint,
												 surface_depth::z16_float, surface_depth::z24s8_float};
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 4 << 20);
	const auto coordinate = [](std::size_t i, std::size_t source, std::size_t destination, bool flip)
	{
		const auto divisor = 2 * destination, value = (2 * i + 1) * source;
		return (flip ? source * divisor - value : value) / divisor;
	};
	std::size_t count = 0;
	for (const auto sf : formats)
		for (const auto df : formats)
			for (const bool deferred : {false, true})
				for (const unsigned flip : {0u, 1u, 2u, 3u})
				{
					const auto sb = surface_guest_bytes(sf), db = surface_guest_bytes(df);
					const auto sw = 32 / sb, reinterpreted = 32 / db, dw = reinterpreted + 3;
					cached_surface_description a;
					a.address = 0x70000;
					a.allocation = {sw, 3, 1, sf, false};
					a.row_pitch = (sw + 2) * sb;
					auto b = a;
					b.address = 0x80000;
					b.allocation = {dw, 5, 1, df, false};
					b.row_pitch = (dw + 3) * db;
					const auto ar = plan_guest_surface(a.address, sw, 3, sb, a.row_pitch).range;
					const auto br = plan_guest_surface(b.address, dw, 5, db, b.row_pitch).range;
					std::vector<std::byte> original(ar.size), target(br.size, std::byte{0xcc});
					for (std::size_t i = 0; i < original.size(); ++i)
						original[i] = static_cast<std::byte>((i * 37 + i / 7 + 19) % 256);
					memory->write(ar, original);
					memory->write(br, target);
					const auto source = cache.acquire(a), destination = cache.acquire(b);
					const std::array axes{bool(flip & 1), bool(flip & 2), false};
					if (deferred)
					{
						cache.defer_blit_typeless(source, {0, 0, sw, 3}, destination, {0, 0, dw, 5}, axes);
						cache.erase(source);
						require(cache.status(destination).deferred_copies == 1 && !cache.status(destination).initialized,
								"Captured typeless source format/allocation was lost on erase");
						cache.prepare(destination);
					}
					else
						cache.blit_typeless(source, {0, 0, sw, 3}, destination, {0, 0, dw, 5}, axes);
					require(cache.status(destination).content_generation == 1 && memory->read(br).bytes == target,
							"Typeless full overwrite imported old target guest bytes or published early");
					cache.flush(destination);
					auto expected = target;
					for (std::size_t y = 0; y < 5; ++y)
						for (std::size_t x = 0; x < dw; ++x)
						{
							const auto from = coordinate(y, 3, 5, flip & 2) * a.row_pitch + coordinate(x, reinterpreted, dw, flip & 1) * db;
							std::copy_n(original.begin() + from, db, expected.begin() + y * b.row_pitch + x * db);
						}
					require(memory->read(br).bytes == expected && cache.status(destination).deferred_copies == 0,
							"Cached typeless source words/target reinterpretation/scaling or padding differ");
					if (!deferred)
						cache.erase(source);
					cache.erase(destination);
					++count;
				}
	// Overlapping guest descriptors with differing bytes-per-pixel retain the
	// captured source representation even when its logical owner is retired.
	cached_surface_description a;
	a.address = 0x70000;
	a.allocation = {8, 3, 1, surface_color::argb8, false};
	auto b = a;
	b.allocation = {16, 3, 1, surface_depth::z16_float, false};
	std::vector<std::byte> original(32 * 3);
	for (std::size_t i = 0; i < original.size(); ++i)
		original[i] = static_cast<std::byte>((i * 7 + 3) % 256);
	memory->write({a.address, original.size()}, original);
	const auto source = cache.acquire(a), destination = cache.acquire(b);
	cache.defer_blit_typeless(source, {0, 0, 8, 3}, destination, {0, 0, 16, 3});
	cache.erase(source);
	cache.flush(destination);
	require(memory->read({a.address, original.size()}).bytes == original,
			"Equal-storage ARGB8->E4M12 typeless alias lost BE byte identity on source erase");
	cache.erase(destination);
	// Reject incomplete target words before touching either surface/allocation.
	a.address = 0x70000;
	a.allocation = {3, 3, 1, surface_color::b8, false};
	b.address = 0x80000;
	b.allocation = {1, 3, 1, surface_color::rgba32f, false};
	const auto incomplete = cache.acquire(a), wide = cache.acquire(b);
	bool refused = false;
	try
	{
		cache.defer_blit_typeless(incomplete, {0, 0, 3, 3}, wide, {0, 0, 1, 3});
	}
	catch (const std::invalid_argument&)
	{
		refused = true;
	}
	require(refused && !cache.status(incomplete).initialized && cache.status(wide).deferred_copies == 0,
			"Typeless incomplete-row preflight allocated/initialized source or queued invalid target words");
	std::cout << count << " cached direct/deferred typeless 18x18 format pairs with flips preserve BE guest words/padding; retired aliases and incomplete-row preflight passed\n";
}
void cached_sample_blits(context& ctx)
{
	const std::array<surface_format, 18> formats{surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565,
												 surface_color::xrgb8_zero, surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8,
												 surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero,
												 surface_color::xbgr8_one, surface_color::abgr8, surface_depth::z16_uint, surface_depth::z24s8_uint,
												 surface_depth::z16_float, surface_depth::z24s8_float};
	const auto grid = [](unsigned samples)
	{
		return std::array<std::size_t, 2>{samples == 1 ? 1u : 2u, samples == 4 ? 2u : 1u};
	};
	const auto coordinate = [](std::size_t i, std::size_t source, std::size_t destination)
	{
		return (source * 2 * destination - (2 * i + 1) * source) / (2 * destination);
	};
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, 8 << 20);
	std::size_t cases = 0, aspects = 0;
	for (const auto sf : formats)
		for (const auto df : formats)
			for (const auto counts : {std::array{1u, 4u}, std::array{4u, 2u}, std::array{2u, 1u}})
				for (const bool deferred : {false, true})
				{
					const auto a = grid(counts[0]), b = grid(counts[1]);
					const auto sb = surface_guest_bytes(sf), db = surface_guest_bytes(df), sw = 16 / sb, dw = 16 / db + 2;
					cached_surface_description source_description;
					source_description.address = 0x70000;
					source_description.allocation = {sw, 3, counts[0], sf, false};
					source_description.row_pitch = (sw * a[0] + 2) * sb;
					cached_surface_description target_description;
					target_description.address = 0x80000;
					target_description.allocation = {dw, 4, counts[1], df, false};
					target_description.row_pitch = (dw * b[0] + 3) * db;
					const auto ar = plan_guest_surface(source_description.address, sw * a[0], 3 * a[1], sb, source_description.row_pitch).range;
					const auto br = plan_guest_surface(target_description.address, dw * b[0], 4 * b[1], db, target_description.row_pitch).range;
					std::vector<std::byte> original(ar.size), initial(br.size, std::byte{0xcc});
					for (std::size_t i = 0; i < original.size(); ++i)
						original[i] = static_cast<std::byte>(0x10 + (i * 7 + i / 11) % 0x50);
					memory->write(ar, original);
					memory->write(br, initial);
					const auto source = cache.acquire(source_description), target = cache.acquire(target_description);
					if (deferred)
					{
						cache.defer_blit_typeless(source, {0, 0, sw, 3}, target, {0, 0, dw, 4}, {true, true, false});
						cache.erase(source);
						cache.prepare(target);
					}
					else
						cache.blit_typeless(source, {0, 0, sw, 3}, target, {0, 0, dw, 4}, {true, true, false});
					require(cache.status(target).content_generation == 1 && memory->read(br).bytes == initial,
							"Spatial sample full overwrite loaded invalid old guest data or published before completion");
					cache.flush(target);
					auto expected = initial;
					for (std::size_t y = 0; y < 4 * b[1]; ++y)
						for (std::size_t x = 0; x < dw * b[0]; ++x)
						{
							const auto from = coordinate(y, 3 * a[1], 4 * b[1]) * source_description.row_pitch + coordinate(x, sw * a[0] * sb / db, dw * b[0]) * db;
							std::copy_n(original.begin() + from, db, expected.begin() + y * target_description.row_pitch + x * db);
						}
					require(memory->read(br).bytes == expected && cache.status(target).deferred_copies == 0,
							"Cached spatial MSAA typeless mapping changed sample words or guest padding");
					if (!deferred)
						cache.erase(source);
					cache.erase(target);
					++cases;
				}
	for (const auto counts : {std::array{2u, 4u}, std::array{4u, 2u}, std::array{1u, 4u}})
		for (const auto plane : {texture_plane::depth, texture_plane::stencil})
			for (const bool deferred : {false, true})
			{
				const auto a = grid(counts[0]), b = grid(counts[1]);
				cached_surface_description s, d;
				s.address = 0x70000;
				d.address = 0x80000;
				s.allocation = {8, 6, counts[0], surface_depth::z24s8_float, false};
				d.allocation = {9, 7, counts[1], surface_depth::z24s8_float, false};
				s.row_pitch = (8 * a[0] + 2) * 4;
				d.row_pitch = (9 * b[0] + 3) * 4;
				const auto ar = plan_guest_surface(s.address, 8 * a[0], 6 * a[1], 4, s.row_pitch).range;
				const auto br = plan_guest_surface(d.address, 9 * b[0], 7 * b[1], 4, d.row_pitch).range;
				std::vector<std::byte> original(ar.size), initial(br.size, std::byte{0xcc});
				for (std::size_t i = 0; i < original.size(); ++i)
					original[i] = static_cast<std::byte>(0x10 + (i * 7 + i / 11) % 0x50);
				for (std::size_t y = 0; y < 7 * b[1]; ++y)
					for (std::size_t x = 0; x < 9 * b[0]; ++x)
						store(std::span(initial).subspan(y * d.row_pitch + x * 4), 0x44444444, 4);
				memory->write(ar, original);
				memory->write(br, initial);
				const auto source = cache.acquire(s), target = cache.acquire(d);
				if (deferred)
				{
					cache.defer_blit_nearest(source, {1, 1, 4, 3}, target, {2, 2, 5, 4}, plane, {true, true, false});
					cache.erase(source);
				}
				else
					cache.blit_nearest(source, {1, 1, 4, 3}, target, {2, 2, 5, 4}, plane, {true, true, false});
				cache.flush(target);
				auto expected = initial;
				for (std::size_t y = 0; y < 4 * b[1]; ++y)
					for (std::size_t x = 0; x < 5 * b[0]; ++x)
					{
						const auto from = (a[1] + coordinate(y, 3 * a[1], 4 * b[1])) * s.row_pitch + (a[0] + coordinate(x, 4 * a[0], 5 * b[0])) * 4;
						const auto to = (2 * b[1] + y) * d.row_pitch + (2 * b[0] + x) * 4;
						for (std::size_t byte = 0; byte < 4; ++byte)
							if ((plane == texture_plane::depth && byte < 3) || (plane == texture_plane::stencil && byte == 3))
								expected[to + byte] = original[from + byte];
					}
				require(memory->read(br).bytes == expected, "Cached partial sample transfer changed outside pixels/unselected aspect/padding");
				if (!deferred)
					cache.erase(source);
				cache.erase(target);
				++aspects;
			}
	// Converting ARGB8 guest words into 2x float-depth can generate depth >1.
	// The native sample import must report failure, preserve VM bytes/plan and
	// permit explicit completed-error recovery followed by a full valid clear.
	cached_surface_description s, d;
	s.address = 0x70000;
	s.allocation = {4, 2, 1, surface_color::argb8, false};
	d.address = 0x80000;
	d.allocation = {2, 2, 2, surface_depth::z24s8_float, false};
	std::vector<std::byte> invalid(4 * 2 * 4), initial(invalid.size(), std::byte{0x44});
	for (std::size_t i = 0; i < invalid.size(); i += 4)
		store(std::span(invalid).subspan(i), 0x80000055, 4);
	memory->write({s.address, invalid.size()}, invalid);
	memory->write({d.address, initial.size()}, initial);
	const auto source = cache.acquire(s), target = cache.acquire(d);
	cache.defer_blit_typeless(source, {0, 0, 4, 2}, target, {0, 0, 2, 2});
	bool failed = false;
	try
	{
		cache.prepare(target);
	}
	catch (const std::runtime_error&)
	{
		failed = true;
	}
	require(failed && !cache.status(target).faulted && cache.status(target).deferred_copies == 1 && memory->read({d.address, initial.size()}).bytes == initial,
			"Invalid MSAA typeless depth silently clamped/published or discarded its capture plan");
	cache.recover(target);
	cache.clear(target, 3, 0, 0x80000033, 255, {0, 0, 2, 2});
	cache.flush(target);
	auto expected = initial;
	for (std::size_t i = 0; i < expected.size(); i += 4)
		store(std::span(expected).subspan(i), 0x7e000033, 4);
	require(memory->read({d.address, expected.size()}).bytes == expected && cache.status(target).deferred_copies == 0,
			"Full clear after invalid MSAA typeless depth did not recover precise guest contents");
	std::cout << cases << " direct/deferred spatial MSAA typeless cache cases, " << aspects << " selected-aspect partial copies and rejected out-of-range depth recovery passed\n";
}
void cached_feedback(context& ctx)
{
	for (const unsigned count : {1u, 2u, 4u})
	{
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 4 << 20);
		cached_surface_description description;
		description.address = 0x70000;
		description.allocation = {8, 6, count, surface_color::argb8, false};
		const std::size_t sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1;
		description.row_pitch = (8 * sx + 3) * 4;
		const auto range = plan_guest_surface(description.address, 8 * sx, 6 * sy, 4, description.row_pitch).range;
		std::vector<std::byte> original(range.size, std::byte{0xa5});
		for (std::size_t y = 0; y < 6 * sy; ++y)
			for (std::size_t x = 0; x < 8 * sx; ++x)
			{
				const auto sample = (x % sx) + (y % sy) * sx;
				const auto word = std::uint32_t((0x80 + sample * 7) << 24) | std::uint32_t((x / sx + sample * 31 + 1) << 16) | std::uint32_t((y / sy + sample * 17 + 2) << 8) | std::uint32_t(11 + sample * 13);
				store(std::span(original).subspan(y * description.row_pitch + x * 4), word, 4);
			}
		memory->write(range, original);
		const auto surface = cache.acquire(description);
		compiler compiler(ctx);
		std::string source = R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 feedback_vertex(uint id [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1); }
fragment float4 feedback_fragment(float4 p [[position]],
)msl";
		if (count > 1)
			source += "uint sample [[sample_id]],texture2d_ms<float> input [[texture(0)]]) {\n";
		else
			source += "texture2d<float,access::read> input [[texture(0)]]) {\n";
		source += "const uint2 coord=uint2((uint(p.x)+1)%8,uint(p.y));\nconst float4 value=input.read(coord";
		if (count > 1)
			source += ",sample";
		source += ");return float4(value.b,value.r,value.g,value.a);}\n";
		const auto library = compiler.library(source, @"Native barrier-boundary feedback");
		auto descriptor = [MTL4RenderPipelineDescriptor new];
		descriptor.vertexFunctionDescriptor = library_function(library, @"feedback_vertex");
		descriptor.fragmentFunctionDescriptor = library_function(library, @"feedback_fragment");
		descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
		descriptor.rasterSampleCount = count;
		descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		const auto pipeline = compiler.render(descriptor);
		const std::array uses{cached_surface_use{surface, false}, cached_surface_use{surface, false}, cached_surface_use{surface, true}};
		bool refused = false;
		try
		{
			cache.draw(uses, [](context&, auto) {});
		}
		catch (const std::invalid_argument&)
		{
			refused = true;
		}
		require(refused && !cache.status(surface).initialized, "Ordinary draw silently accepted feedback or prepared before rejecting it");
		auto expected = original;
		for (unsigned iteration = 0; iteration < 3; ++iteration)
		{
			cache.draw_feedback(uses, [&](context& commands, auto textures)
								{
									require(textures[0] == textures[1] && textures[0] != textures[2], "Duplicate feedback inputs did not share an independent snapshot");
									auto pass = [MTL4RenderPassDescriptor new];
									pass.colorAttachments[0].texture = textures[2];
									pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
									pass.colorAttachments[0].storeAction = MTLStoreActionStore;
									render_commands draw(commands, pass);
									bindings fragment;
									fragment.set_texture(0, textures[0]);
									raster_state raster;
									raster.viewport = {0, 0, 8, 6, 0, 1};
									raster.scissor = {1, 1, 6, 4};
									draw.configure(pipeline, nil, raster, {}, fragment);
									// Both native draws in this callback see one explicit captured epoch.
									draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
									draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
									draw.end(); });
			auto before = expected;
			for (std::size_t y = sy; y < 5 * sy; ++y)
				for (std::size_t x = sx; x < 7 * sx; ++x)
				{
					const auto from = y * description.row_pitch + (((x / sx + 1) % 8) * sx + x % sx) * 4;
					const auto to = y * description.row_pitch + x * 4;
					expected[to] = before[from];
					expected[to + 1] = before[from + 3];
					expected[to + 2] = before[from + 1];
					expected[to + 3] = before[from + 2];
				}
			require(memory->read(range).bytes == original && cache.status(surface).dirty, "Feedback draw published guest data before explicit writeback");
		}
		cache.flush(surface);
		require(memory->read(range).bytes == expected, "Consecutive feedback epochs changed sample identity, exterior pixels, channel cycle or guest padding");
		const auto generation = cache.status(surface).content_generation;
		bool cancelled = false;
		try
		{
			cache.draw_feedback(uses, [](context&, auto)
								{ throw std::runtime_error("Cancel feedback encoding"); });
		}
		catch (const std::runtime_error&)
		{
			cancelled = true;
		}
		require(cancelled && !cache.status(surface).dirty && !cache.status(surface).faulted && cache.status(surface).content_generation == generation,
				"Cancelled feedback draw changed content generation/dirty/fault state");
		if (count == 1)
		{
			const auto gate = [ctx.device() newSharedEvent];
			struct event_guard
			{
				id<MTLSharedEvent> event;
				~event_guard()
				{
					event.signaledValue = 1;
				}
			} release{gate};
			bool timeout = false;
			try
			{
				cache.draw_feedback(uses, [&](context& commands, auto textures)
									{
										cache.set_wait_timeout(std::chrono::milliseconds(1));
										commands.wait_event(gate, 1);
										auto pass = [MTL4RenderPassDescriptor new];
										pass.colorAttachments[0].texture = textures[2];
										pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
										pass.colorAttachments[0].storeAction = MTLStoreActionStore;
										render_commands draw(commands, pass);
										bindings fragment;
										fragment.set_texture(0, textures[0]);
										raster_state raster;
										raster.viewport = {0, 0, 8, 6, 0, 1};
										raster.scissor = {1, 1, 6, 4};
										draw.configure(pipeline, nil, raster, {}, fragment);
										draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
										draw.end(); });
			}
			catch (const std::runtime_error&)
			{
				timeout = true;
			}
			require(timeout && cache.status(surface).pending && !cache.status(surface).dirty && cache.status(surface).content_generation == generation,
					"Feedback timeout published uncompleted draw metadata");
			gate.signaledValue = 1;
			cache.synchronize();
			cache.set_wait_timeout(std::chrono::seconds(30));
			auto before = expected;
			for (std::size_t y = 1; y < 5; ++y)
				for (std::size_t x = 1; x < 7; ++x)
				{
					const auto from = y * description.row_pitch + ((x + 1) % 8) * 4, to = y * description.row_pitch + x * 4;
					expected[to] = before[from];
					expected[to + 1] = before[from + 3];
					expected[to + 2] = before[from + 1];
					expected[to + 3] = before[from + 2];
				}
			cache.flush(surface);
			require(memory->read(range).bytes == expected, "Feedback snapshot/argument bindings died while actual GPU draw was counter-blocked");
		}
	}
	// Eight uses (four readers and four MRTs) must retain four independent input
	// epochs; the former five-use cap conflated texture inputs with attachments.
	{
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 4 << 20);
		std::array<cached_surface_ref, 4> surfaces;
		std::vector<cached_surface_use> uses;
		for (std::size_t slot = 0; slot < 4; ++slot)
		{
			cached_surface_description d;
			d.address = 0x40000 + std::uint32_t(slot) * 0x1000;
			d.allocation = {4, 3, 1, surface_color::argb8, false};
			surfaces[slot] = cache.acquire(d);
			cache.clear(surfaces[slot], 0xf0, 0x80112233 + std::uint32_t(slot) * 0x00030405, 0, 255, {0, 0, 4, 3});
			uses.push_back({surfaces[slot], false});
		}
		for (const auto& surface : surfaces)
			uses.push_back({surface, true});
		compiler compiler(ctx);
		const auto library = compiler.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 mrt_vertex(uint id [[vertex_id]]) {const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);}
struct mrt_output {float4 c0 [[color(0)]],c1 [[color(1)]],c2 [[color(2)]],c3 [[color(3)]];};
fragment mrt_output mrt_fragment(float4 p [[position]],texture2d<float,access::read> a [[texture(0)]],texture2d<float,access::read> b [[texture(1)]],texture2d<float,access::read> c [[texture(2)]],texture2d<float,access::read> d [[texture(3)]]) {const uint2 xy=uint2(p.xy);return {b.read(xy),c.read(xy),d.read(xy),a.read(xy)};}
)msl",
											  @"Native four-MRT feedback");
		auto pd = [MTL4RenderPipelineDescriptor new];
		pd.vertexFunctionDescriptor = library_function(library, @"mrt_vertex");
		pd.fragmentFunctionDescriptor = library_function(library, @"mrt_fragment");
		pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		for (std::size_t i = 0; i < 4; ++i)
			pd.colorAttachments[i].pixelFormat = MTLPixelFormatBGRA8Unorm;
		const auto pipeline = compiler.render(pd);
		cache.draw_feedback(uses, [&](context& commands, auto textures)
							{
								auto pass = [MTL4RenderPassDescriptor new];
								bindings fragment;
								for (std::size_t i = 0; i < 4; ++i)
								{
									require(textures[i] != textures[i + 4], "MRT feedback source aliases writable target");
									fragment.set_texture(i, textures[i]);
									pass.colorAttachments[i].texture = textures[i + 4];
									pass.colorAttachments[i].loadAction = MTLLoadActionLoad;
									pass.colorAttachments[i].storeAction = MTLStoreActionStore;
								}
								render_commands draw(commands, pass);
								raster_state raster;
								raster.viewport = {0, 0, 4, 3, 0, 1};
								raster.scissor = {0, 0, 4, 3};
								draw.configure(pipeline, nil, raster, {}, fragment);
								draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
								draw.end(); });
		for (std::size_t slot = 0; slot < 4; ++slot)
		{
			cache.flush(surfaces[slot]);
			const auto actual = memory->read({0x40000 + std::uint32_t(slot) * 0x1000, 48}).bytes;
			std::vector<std::byte> expected(48);
			for (std::size_t i = 0; i < 48; i += 4)
				store(std::span(expected).subspan(i), 0x80112233 + std::uint32_t((slot + 1) % 4) * 0x00030405, 4);
			require(actual == expected, "Four-MRT feedback did not retain independent prepared source epochs");
		}
		const std::array overlapping{cached_surface_use{surfaces[0], true}, cached_surface_use{surfaces[0], true}};
		bool rejected = false;
		try
		{
			cache.draw_feedback(overlapping, [](context&, auto) {});
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		require(rejected, "Feedback mode accepted overlapping simultaneous write targets");
	}
	// Typed guest aliases are snapshotted before writer preparation invalidates
	// the logical reader. Sample ARGB native RGBA values into ABGR storage.
	{
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 4 << 20);
		cached_surface_description a;
		a.address = 0x70000;
		a.allocation = {4, 3, 1, surface_color::argb8, false};
		auto b = a;
		b.allocation.format = surface_color::abgr8;
		std::vector<std::byte> original(4 * 3 * 4);
		for (std::size_t i = 0; i < original.size(); i += 4)
			store(std::span(original).subspan(i), std::uint32_t(0x80112233 + i / 4), 4);
		memory->write({a.address, original.size()}, original);
		const auto reader = cache.acquire(a), writer = cache.acquire(b);
		compiler compiler(ctx);
		const auto library = compiler.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 alias_vertex(uint id [[vertex_id]]) {const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};return float4(p[id],0,1);}
fragment float4 alias_fragment(float4 p [[position]],texture2d<float,access::read> source [[texture(0)]]) {return source.read(uint2(p.xy));}
)msl",
											  @"Typed feedback alias");
		auto pd = [MTL4RenderPipelineDescriptor new];
		pd.vertexFunctionDescriptor = library_function(library, @"alias_vertex");
		pd.fragmentFunctionDescriptor = library_function(library, @"alias_fragment");
		pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
		pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
		const auto pipeline = compiler.render(pd);
		const std::array uses{cached_surface_use{reader, false}, cached_surface_use{writer, true}};
		cache.draw_feedback(uses, [&](context& commands, auto textures)
							{
								require(textures[0] != textures[1] && textures[0].pixelFormat == MTLPixelFormatBGRA8Unorm, "Typed feedback reader lost its captured native format");
								auto pass = [MTL4RenderPassDescriptor new];
								pass.colorAttachments[0].texture = textures[1];
								pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
								pass.colorAttachments[0].storeAction = MTLStoreActionStore;
								render_commands draw(commands, pass);
								bindings fragment;
								fragment.set_texture(0, textures[0]);
								raster_state raster;
								raster.viewport = {0, 0, 4, 3, 0, 1};
								raster.scissor = {0, 0, 4, 3};
								draw.configure(pipeline, nil, raster, {}, fragment);
								draw.draw(MTLPrimitiveTypeTriangle, 0, 3);
								draw.end(); });
		cache.flush(writer);
		auto expected = original;
		for (std::size_t i = 0; i < expected.size(); i += 4)
			std::swap(expected[i + 1], expected[i + 3]);
		require(memory->read({a.address, expected.size()}).bytes == expected, "Typed feedback alias sampling used destination format or lost source channels");
	}
	// A feedback capture itself must not normalize float-depth bits. Mutate the
	// live target, then observe the immutable source through native plane copies.
	{
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 4 << 20);
		cached_surface_description a;
		a.address = 0x70000;
		a.allocation = {8, 2, 1, surface_depth::z24s8_float, false};
		std::vector<std::byte> initial(64, std::byte{0});
		memory->write({a.address, initial.size()}, initial);
		const auto value = cache.acquire(a);
		std::vector<std::uint32_t> raw(16);
		std::vector<std::byte> stencil(16);
		for (std::size_t i = 0; i < 16; ++i)
		{
			raw[i] = i % 3 == 0 ? 0x7fc12345u : i % 3 == 1 ? 0x40000000u
														   : 0x3f000007u + std::uint32_t(i);
			stencil[i] = static_cast<std::byte>(i * 7);
		}
		const auto depth = make_upload(ctx, std::as_bytes(std::span(raw))), si = make_upload(ctx, stencil), output = make_buffer(ctx, 64, buffer_storage::shared), so = make_buffer(ctx, 16, buffer_storage::shared);
		texture_region whole;
		whole.size = MTLSizeMake(8, 2, 1);
		const std::array write{cached_surface_use{value, true}};
		cache.draw(write, [&](context& commands, auto textures)
				   {
					   buffer_to_texture(commands, depth, 0, textures[0], whole, 32, 0, texture_plane::depth);
					   buffer_to_texture(commands, si, 0, textures[0], whole, 8, 0, texture_plane::stencil); });
		const std::array uses{cached_surface_use{value, false}, cached_surface_use{value, true}};
		cache.draw_feedback(uses, [&](context& commands, auto textures)
							{
								auto pass = [MTL4RenderPassDescriptor new];
								pass.depthAttachment.texture = textures[1];
								pass.depthAttachment.loadAction = MTLLoadActionClear;
								pass.depthAttachment.storeAction = MTLStoreActionStore;
								pass.depthAttachment.clearDepth = 1;
								pass.stencilAttachment.texture = textures[1];
								pass.stencilAttachment.loadAction = MTLLoadActionClear;
								pass.stencilAttachment.storeAction = MTLStoreActionStore;
								pass.stencilAttachment.clearStencil = 255;
								(void)commands.render(pass);
								commands.end_encoding();
								texture_to_buffer(commands, textures[0], whole, output, 0, 32, 0, texture_plane::depth);
								texture_to_buffer(commands, textures[0], whole, so, 0, 8, 0, texture_plane::stencil); });
		const auto actual = map_buffer(output, 0, 64), sa = map_buffer(so, 0, 16);
		require(std::equal(actual.begin(), actual.end(), std::as_bytes(std::span(raw)).begin()) && std::equal(sa.begin(), sa.end(), stencil.begin()), "Feedback snapshot changed NaN/>1/native depth bits or stencil after target mutation");
	}
	std::cout << "Native feedback snapshots preserve consecutive 1/2/4-sample epochs, duplicate bindings, four-MRT input epochs, callback-batch visibility, actual GPU timeout lifetime, scissor/padding, typed aliases, cancelled metadata and raw NaN depth/stencil bits\n";
}
void cached_depth_bounds(context& ctx)
{
	compiler compile(ctx);
	vertex_fetch_engine fetch(ctx);
	sample_transfer samples(ctx);
	constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
	const std::array<std::array<unsigned, 4>, 1> vertex_code{{{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
	vertex_shader vertex(compile, vertex_code);
	std::vector<std::byte> vertices, code;
	for (float value : {-1.f, -1.f, 0.75f, 1.f, 3.f, -1.f, 0.75f, 1.f, -1.f, 3.f, 0.75f, 1.f})
		for (unsigned shift : {24u, 16u, 8u, 0u})
			vertices.push_back(static_cast<std::byte>(std::bit_cast<unsigned>(value) >> shift));
	for (unsigned word : {(1u << 24) | (15u << 9) | 1u, (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9, 0u, 0u, 0u, 0u})
		for (unsigned shift : {8u, 0u, 24u, 16u})
			code.push_back(static_cast<std::byte>(word >> shift));
	const auto source = make_upload(ctx, vertices);
	const auto geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	unsigned cases = 0;
	for (const auto format : {surface_depth::z16_uint, surface_depth::z24s8_uint, surface_depth::z16_float, surface_depth::z24s8_float})
		for (const unsigned count : {1u, 2u, 4u})
		{
			@autoreleasepool
			{
				const surface_allocation_description allocation{8, 8, count, format, count > 1};
				const surface_allocation_description mirror_description{8, 8, count, surface_color::r32f, false};
				std::size_t depth_cost, mirror_cost;
				{
					surface_pool probe(ctx, 1 << 20);
					depth_cost = probe.acquire(allocation)->reserved_bytes();
					mirror_cost = probe.acquire(mirror_description)->reserved_bytes();
				}
				auto memory = std::make_shared<memory_fixture>();
				surface_cache cache(ctx, memory, depth_cost + mirror_cost);
				const auto target = cache.acquire({0x60000, allocation, 0, {}});
				const std::array uses{cached_surface_use{target, true}};
				const auto native = describe_surface_depth(format);
				auto descriptor = [&](auto textures)
				{
					auto pass = [MTL4RenderPassDescriptor new];
					pass.depthAttachment.texture = textures[0];
					pass.depthAttachment.loadAction = MTLLoadActionLoad;
					pass.depthAttachment.storeAction = MTLStoreActionStore;
					if (native.stencil)
					{
						pass.stencilAttachment.texture = textures[0];
						pass.stencilAttachment.loadAction = MTLLoadActionLoad;
						pass.stencilAttachment.storeAction = MTLStoreActionStore;
					}
					return pass;
				};
				auto reset = [&]
				{
					cache.draw(uses, [&](context& commands, auto textures)
							   {
								   auto pass = descriptor(textures);
								   pass.depthAttachment.loadAction = MTLLoadActionClear;
								   pass.depthAttachment.clearDepth = 0.25;
								   if (native.stencil)
								   {
									   pass.stencilAttachment.loadAction = MTLLoadActionClear;
									   pass.stencilAttachment.clearStencil = 17;
								   }
								   (void)commands.render(pass);
								   commands.end_encoding(); });
					cache.flush(target);
				};
				reset();
				const unsigned sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1;
				auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:native.format width:8 * sx height:8 * sy mipmapped:NO];
				td.storageMode = MTLStorageModePrivate;
				td.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
				const auto expanded = make_texture(ctx, td);
				const auto output = make_buffer(ctx, 256 * 8 * sy, buffer_storage::shared);
				const auto before = make_buffer(ctx, output.length, buffer_storage::shared);
				const auto stencil = make_buffer(ctx, output.length, buffer_storage::shared);
				const std::array read{cached_surface_use{target, false}};
				auto capture = [&](id<MTLBuffer> depth_output)
				{
					cache.draw(read, [&](context& commands, auto textures)
							   {
								   id<MTLTexture> texture = textures[0];
								   if (count > 1)
								   {
									   samples.expand(commands, texture, expanded);
									   texture = expanded;
								   }
								   texture_region region;
								   region.size = MTLSizeMake(8 * sx, 8 * sy, 1);
								   texture_to_buffer(commands, texture, region, depth_output, 0, 256, 0, texture_plane::depth);
								   if (native.stencil)
									   texture_to_buffer(commands, texture, region, stencil, 0, 256, 0, texture_plane::stencil); });
				};
				capture(before);
				std::unique_ptr<fragment_shader> fragment;
				std::optional<render_pipeline_bundle> pipeline;
				auto encode = [&](context& commands, auto, depth_bounds_pass& pass)
				{
					if (!fragment)
					{
						render_pipeline_configuration config;
						config.state.sample_count = count;
						config.state.depth_test = true;
						config.state.depth_write = true;
						config.state.depth_compare = 0x207;
						fragment_program_options options;
						options.color_count = 0;
						pass.configure_pipeline(config, options);
						fragment = std::make_unique<fragment_shader>(compile, code, options);
						pipeline = make_render_pipeline(compile, config, vertex.function(), *fragment);
					}
					vertex_fetch_request request;
					request.vertex_count = 3;
					request.attribute_mask = 1;
					request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
					vertex_inputs inputs;
					inputs.attributes = fetch.decode(commands, request, source, nil);
					vertex_context state;
					state.vertex_count = 3;
					const auto bindings = vertex.prepare(commands, state, geometry, inputs);
					raster_state raster;
					raster.viewport = {0, 0, 8, 8, 0, 1};
					raster.scissor = {1, 1, 6, 6};
					raster.depth_bounds_enabled = true;
					raster.depth_min = 0.2;
					raster.depth_max = 0.3;
					pass.draw_geometry(geometry, *pipeline, *fragment, bindings, {}, code, {}, raster, 2);
				};
				// One depth allocation alone is insufficient: the pinned target must
				// survive mirror refusal and encoding must never start.
				{
					surface_cache tight(ctx, memory, depth_cost);
					const auto other = tight.acquire({0x70000, allocation, 0, {}});
					tight.prepare(other);
					const auto generation = tight.status(other).content_generation;
					const std::array write{cached_surface_use{other, true}};
					bool refused = false, called = false;
					try
					{
						tight.draw_depth_bounds(write, 0, descriptor, [&](auto&, auto, auto&)
												{ called = true; });
					}
					catch (const surface_budget_error&)
					{
						refused = true;
					}
					require(refused && !called && tight.status(other).initialized && !tight.status(other).dirty &&
								tight.status(other).content_generation == generation && tight.memory_statistics().reserved_bytes <= depth_cost,
							"Insufficient mirror budget spilled/mutated its pinned depth target or invoked encoding");
					const std::array feedback{cached_surface_use{other, false}, cached_surface_use{other, false}, cached_surface_use{other, true}};
					refused = false;
					try
					{
						tight.draw_feedback(feedback, [&](auto&, auto)
											{ called = true; });
					}
					catch (const surface_budget_error&)
					{
						refused = true;
					}
					require(refused && !called && tight.status(other).initialized && !tight.status(other).dirty &&
								tight.status(other).content_generation == generation && tight.memory_statistics().reserved_bytes <= depth_cost,
							"Feedback snapshot escaped the pinned target's heap budget or changed its contents");
				}
				cached_surface_ref donor;
				if (count == 1)
				{
					donor = cache.acquire({0x80000, mirror_description, 0, {}});
					cache.prepare(donor, surface_initialization::background);
					require(cache.status(donor).dirty && cache.memory_statistics().leased_allocations == 2, "Mirror pressure fixture did not fill both budgeted allocations");
				}
				const auto generation = cache.status(target).content_generation;
				struct cancelled_encoding
				{
				};
				bool cancelled = false;
				try
				{
					cache.draw_depth_bounds(uses, 0, descriptor, [&](auto& commands, auto textures, auto& pass)
											{
												encode(commands, textures, pass);
												throw cancelled_encoding{}; });
				}
				catch (const cancelled_encoding&)
				{
					cancelled = true;
				}
				require(cancelled && !cache.status(target).dirty && !cache.status(target).faulted &&
							cache.status(target).content_generation == generation && cache.memory_statistics().leased_allocations == 1,
						"Cancelled bounds encoding published metadata or kept its mirror lease active");
				if (donor)
					require(cache.status(donor).spilled && cache.status(donor).dirty && cache.status(target).initialized,
							"Mirror allocation failed to spill an unpinned dirty surface while preserving its depth target");
				id<MTLSharedEvent> event = [ctx.device() newSharedEvent];
				struct release_event
				{
					id<MTLSharedEvent> event;
					~release_event()
					{
						event.signaledValue = 1;
					}
				} release{event};
				bool timeout = false;
				try
				{
					cache.draw_depth_bounds(uses, 0, descriptor, [&](auto& commands, auto textures, auto& pass)
											{
												cache.set_wait_timeout(std::chrono::milliseconds(1));
												commands.wait_event(event, 1);
												encode(commands, textures, pass); });
				}
				catch (const std::runtime_error&)
				{
					timeout = true;
				}
				require(timeout && cache.status(target).pending && !cache.status(target).dirty &&
							cache.status(target).content_generation == generation && cache.memory_statistics().leased_allocations == 2 &&
							cache.memory_statistics().reserved_bytes <= depth_cost + mirror_cost,
						"Bounds timeout lost its budgeted mirror or published uncompleted native metadata");
				event.signaledValue = 1;
				cache.synchronize();
				cache.set_wait_timeout(std::chrono::seconds(30));
				ctx.wait_idle();
				require(cache.status(target).dirty && cache.status(target).content_generation == generation + 1 &&
							cache.memory_statistics().leased_allocations == 1 && cache.memory_statistics().allocations_reused >= 1,
						"Completed bounds work failed to commit metadata/release/reuse its pooled mirror");
				capture(output);
				for (unsigned y = 0; y < 8 * sy; ++y)
					for (unsigned x = 0; x < 8 * sx; ++x)
					{
						const bool written = x / sx >= 1 && x / sx < 7 && y / sy >= 1 && y / sy < 7;
						if (native.depth_bytes == 2)
						{
							const auto actual = static_cast<const std::uint16_t*>(output.contents)[y * 128 + x];
							const auto expected = written ? 49151 : static_cast<const std::uint16_t*>(before.contents)[y * 128 + x];
							if (actual != expected)
								throw std::runtime_error("Cached bounds D16 mismatch sample count=" + std::to_string(count) + " x=" + std::to_string(x) + " y=" + std::to_string(y) + " actual=" + std::to_string(actual));
						}
						else
							require(static_cast<const float*>(output.contents)[y * 64 + x] == (written ? 0.75f : static_cast<const float*>(before.contents)[y * 64 + x]), "Cached bounds changed float depth/scissor/sample identity");
						if (native.stencil)
							require(static_cast<const std::uint8_t*>(stencil.contents)[y * 256 + x] == 17, "Cached bounds changed preserved stencil");
					}
				if (format == surface_depth::z16_uint && count == 1)
				{
					const auto lifetime_output = make_buffer(ctx, 256 * 8, buffer_storage::shared);
					auto transient_cache = std::make_unique<surface_cache>(ctx, memory, depth_cost + mirror_cost);
					const std::array temporary{cached_surface_use{transient_cache->acquire({0x60000, allocation, 0, {}}), true}};
					id<MTLSharedEvent> gate = [ctx.device() newSharedEvent];
					release_event release_lifetime{gate};
					bool delayed = false;
					try
					{
						transient_cache->draw_depth_bounds(temporary, 0, descriptor, [&](context& commands, auto, depth_bounds_pass& pass)
														   {
															   transient_cache->set_wait_timeout(std::chrono::milliseconds(1));
															   commands.wait_event(gate, 1);
															   texture_region region;
															   region.size = MTLSizeMake(8, 8, 1);
															   texture_to_buffer(commands, pass.mirror_texture(), region, lifetime_output, 0, 256); });
					}
					catch (const std::runtime_error&)
					{
						delayed = true;
					}
					require(delayed && transient_cache->status(temporary[0].surface).pending && transient_cache->memory_statistics().leased_allocations == 2,
							"Cache destruction fixture did not retain a pending budgeted mirror");
					transient_cache.reset();
					gate.signaledValue = 1;
					ctx.wait_idle();
					const float expected = float(static_cast<const std::uint16_t*>(before.contents)[0]) / 65535.f;
					for (unsigned y = 0; y < 8; ++y)
						for (unsigned x = 0; x < 8; ++x)
							require(static_cast<const float*>(lifetime_output.contents)[y * 64 + x] == expected, "Pending mirror lost contents when its cache/pool was destroyed");
				}
				// Snapshot capture precedes the real depth clear and stored-depth
				// mirror initialization. Two readers share one immutable lease.
				auto snapshot_description = allocation;
				snapshot_description.expanded = false;
				std::size_t snapshot_cost;
				{
					surface_pool probe(ctx, 1 << 20);
					snapshot_cost = probe.acquire(snapshot_description)->reserved_bytes();
				}
				surface_cache combined(ctx, memory, depth_cost + snapshot_cost + mirror_cost);
				const auto feedback_target = combined.acquire({0x90000, allocation, 0, {}});
				const std::array writer{cached_surface_use{feedback_target, true}};
				combined.draw(writer, [&](context& commands, auto textures)
							  {
								  auto pass = descriptor(textures);
								  pass.depthAttachment.loadAction = MTLLoadActionClear;
								  pass.depthAttachment.clearDepth = 0.75;
								  if (native.stencil)
								  {
									  pass.stencilAttachment.loadAction = MTLLoadActionClear;
									  pass.stencilAttachment.clearStencil = 17;
								  }
								  (void)commands.render(pass);
								  commands.end_encoding(); });
				combined.flush(feedback_target);
				const std::array feedback_uses{cached_surface_use{feedback_target, false}, cached_surface_use{feedback_target, false}, cached_surface_use{feedback_target, true}};
				const auto old_depth = make_buffer(ctx, output.length, buffer_storage::shared);
				const auto captured = make_buffer(ctx, output.length, buffer_storage::shared);
				const auto captured_stencil = make_buffer(ctx, output.length, buffer_storage::shared);
				auto record = [&](context& commands, id<MTLTexture> texture, id<MTLBuffer> data, id<MTLBuffer> stencil_data)
				{
					if (count > 1)
					{
						samples.expand(commands, texture, expanded);
						texture = expanded;
					}
					texture_region region;
					region.size = MTLSizeMake(8 * sx, 8 * sy, 1);
					texture_to_buffer(commands, texture, region, data, 0, 256, 0, texture_plane::depth);
					if (native.stencil)
						texture_to_buffer(commands, texture, region, stencil_data, 0, 256, 0, texture_plane::stencil);
				};
				const std::array feedback_read{cached_surface_use{feedback_target, false}};
				combined.draw(feedback_read, [&](context& commands, auto textures)
							  { record(commands, textures[0], old_depth, stencil); });
				const auto feedback_generation = combined.status(feedback_target).content_generation;
				id<MTLSharedEvent> feedback_gate = [ctx.device() newSharedEvent];
				release_event release_feedback{feedback_gate};
				bool feedback_timeout = false;
				try
				{
					combined.draw_depth_bounds(feedback_uses, 2, [&](auto textures)
											   {
												   auto pass = descriptor(textures.subspan(2));
												   pass.depthAttachment.loadAction = MTLLoadActionClear;
												   pass.depthAttachment.clearDepth = 0.25;
												   if (native.stencil)
												   {
													   pass.stencilAttachment.loadAction = MTLLoadActionClear;
													   pass.stencilAttachment.clearStencil = 33;
												   }
												   return pass; }, [&](context& commands, auto textures, depth_bounds_pass& pass)
											   {
												   require(textures[0] == textures[1] && textures[0] != textures[2], "Combined bounds readers did not share an independent snapshot");
												   combined.set_wait_timeout(std::chrono::milliseconds(1));
												   commands.wait_event(feedback_gate, 1);
												   record(commands, textures[0], captured, captured_stencil);
												   encode(commands, textures, pass); }, true);
				}
				catch (const std::runtime_error&)
				{
					feedback_timeout = true;
				}
				require(feedback_timeout && combined.status(feedback_target).pending && !combined.status(feedback_target).dirty &&
							combined.status(feedback_target).content_generation == feedback_generation && combined.memory_statistics().leased_allocations == 3 &&
							combined.memory_statistics().reserved_bytes <= depth_cost + snapshot_cost + mirror_cost,
						"Combined bounds timeout failed to hold exactly target/snapshot/mirror or published metadata early");
				feedback_gate.signaledValue = 1;
				combined.synchronize();
				combined.set_wait_timeout(std::chrono::seconds(30));
				ctx.wait_idle();
				require(combined.status(feedback_target).dirty && combined.status(feedback_target).content_generation == feedback_generation + 1 &&
							combined.memory_statistics().leased_allocations == 1,
						"Combined bounds completion leaked leases or changed generation twice");
				combined.draw(feedback_read, [&](context& commands, auto textures)
							  { record(commands, textures[0], output, stencil); });
				for (unsigned y = 0; y < 8 * sy; ++y)
					for (unsigned x = 0; x < 8 * sx; ++x)
					{
						const auto offset = y * 256 + x * native.depth_bytes;
						const auto captured_bytes = static_cast<const std::byte*>(captured.contents) + offset;
						const auto old_bytes = static_cast<const std::byte*>(old_depth.contents) + offset;
						require(std::equal(captured_bytes, captured_bytes + native.depth_bytes, old_bytes), "Bounds initialization changed captured feedback depth epoch/sample bits");
						const bool written = x / sx >= 1 && x / sx < 7 && y / sy >= 1 && y / sy < 7;
						if (native.depth_bytes == 2)
							require(static_cast<const std::uint16_t*>(output.contents)[y * 128 + x] == (written ? 49151 : static_cast<const std::uint16_t*>(before.contents)[0]), "Combined bounds depth output/scissor differs");
						else
							require(static_cast<const float*>(output.contents)[y * 64 + x] == (written ? 0.75f : 0.25f), "Combined bounds float output/scissor differs");
						if (native.stencil)
							require(static_cast<const std::uint8_t*>(captured_stencil.contents)[y * 256 + x] == 17 && static_cast<const std::uint8_t*>(stencil.contents)[y * 256 + x] == 33, "Combined feedback stencil epoch was not independent of the writer clear");
					}
				++cases;
			}
		}
	std::cout << cases << " native cached bounds cases: four guest depth formats, 1/2/4 samples, strict mirror budget, dirty spill, actual encoded cancellation, counter-blocked timeout, pooled reuse, destroyed-cache lifetime and completed depth/stencil/scissor output passed\n";
	std::cout << cases << " combined native bounds/feedback cases: duplicate pooled readers, old depth/stencil epoch before new real clears, counter-blocked target/snapshot/mirror lifetime and exact native depth/scissor/sample output passed\n";
}
void spilled_snapshot_ready_cases()
{
	using namespace std::chrono_literals;
	unsigned cases = 0;
	std::vector<surface_format> formats;
	for (unsigned f : {1u,2u,3u,4u,5u,8u,9u,10u,11u,12u,13u,14u,15u,16u}) formats.push_back(static_cast<surface_color>(f));
	for (unsigned f = 1; f <= 4; ++f) formats.push_back(static_cast<surface_depth>(f));
	for (const auto& format : formats) for (unsigned samples : {1u,2u,4u})
	{
		context ctx(1,64,3s); auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		{
			@autoreleasepool
			{
				auto ram = std::make_shared<memory_fixture>(); surface_cache cache(ctx,ram,8u << 20,ledger);
				cached_surface_description description{0x1000,{4,4,samples,format,false},0,{}};
				const auto bytes = bpp(description.allocation);
				const auto size = samples == 1 ? std::array<std::size_t,2>{4,4} : sample_expanded_size(4,4,samples);
				description.row_pitch = (size[0] + 2) * bytes;
				const auto source = cache.acquire(description);
				// Conventional RSX clears intentionally leave float color targets
				// untouched. Give every profile real native initialized contents.
				cache.prepare(source,surface_initialization::background);
				cache.clear(source,std::holds_alternative<surface_color>(format) ? 0xf0 : 3,0xff204080,0x6000,37,{0,0,4,4});
				const auto coverage = plan_guest_surface(description.address,size[0],size[1],bytes,description.row_pitch);
				const auto matches = [&](const cached_surface_description& value) { return value.address == description.address && value.allocation == cache.description(source).allocation; };
				const auto original = cache.snapshot_texture(coverage.range,matches,ledger);
				cache.spill(source);
				if (!cache.status(source).spilled || cache.memory_statistics().leased_allocations != 0)
					throw std::runtime_error("Ready-source spill fixture:format=" + std::to_string(std::visit([](auto value) { return unsigned(value); },format)) +
						" kind=" + std::to_string(format.index()) + " samples=" + std::to_string(samples) + " spilled=" + std::to_string(cache.status(source).spilled) +
						" leased=" + std::to_string(cache.memory_statistics().leased_allocations));
				const auto submits = cache.transaction_statistics().submitted, allocations = ledger->statistics().created;
				const auto encoders = ctx.encoding_statistics();
				const auto gate = [ctx.device() newSharedEvent];
				struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
				ctx.begin(); ctx.wait_event(gate,1); const auto pending = ctx.submit();
				for (unsigned hit = 0; hit < 32; ++hit)
				{
					const auto image = cache.snapshot_texture(coverage.range,matches,ledger);
					require(image == original && !pending.ready() && cache.status(source).spilled && cache.memory_statistics().leased_allocations == 0 &&
						cache.transaction_statistics().submitted == submits && ledger->statistics().created == allocations &&
						ctx.encoding_statistics().compute_encoders == encoders.compute_encoders && ctx.encoding_statistics().render_encoders == encoders.render_encoders,
						"Ready snapshot waited/restored/submitted/allocated behind an unrelated genuinely pending queue gate");
				}
				gate.signaledValue = 1; pending.wait();
				// A genuinely new copy reads the spill directly. A subsequent
				// writer still creates a new native generation.
				cache.clear_snapshots();
				const auto fresh = cache.snapshot_texture(coverage.range,matches,ledger);
				require(fresh != original && cache.status(source).spilled && cache.memory_statistics().leased_allocations == 0 &&
					cache.transaction_statistics().submitted == submits + 1,
					"Snapshot miss did not copy the spill in a single submission");
				cache.clear(source,std::holds_alternative<surface_color>(format) ? 0xf0 : 3,0xff804020,0x4000,53,{0,0,4,4});
				if (const auto color = std::get_if<surface_color>(&format); color && !decode_color_clear(*color,0xff804020,0xf0).write_mask)
				{
					const std::array writes{cached_surface_use{source,true}};
					cache.draw(writes,[](context& commands,auto textures)
					{
						auto pass = [MTL4RenderPassDescriptor new];
						pass.colorAttachments[0].texture = textures[0];
						pass.colorAttachments[0].loadAction = MTLLoadActionClear;
						pass.colorAttachments[0].storeAction = MTLStoreActionStore;
						pass.colorAttachments[0].clearColor = MTLClearColorMake(.5,.25,.125,1);
						(void)commands.render(pass);
						commands.end_encoding();
					});
				}
				const auto newer = cache.snapshot_texture(coverage.range,matches,ledger);
				require(newer != fresh && newer != original, "Native source generation reused an earlier immutable snapshot");
				(void)ctx.trim_upload_pages();
			}
		}
		require(ledger->statistics().reserved_bytes == 0, "Spilled ready-source cases retained native backing after owner release");
		++cases;
	}
	require(cases == 54, "Spilled ready-source matrix omitted color/depth/sample profiles");
	std::cout << "Spilled ready snapshot:54 color/depth/1-2-4sample profiles,1728 hits while the only native slot is truly event-gated;no source restore,allocation,encoder orsubmit,raw spill remains;cache miss copies in one submission,new writer changes identity,all native charges retire passed\n";
}
void clean_texture_active_coverage(context& ctx)
{
	unsigned count = 0;
	for (const surface_format format : {surface_format{surface_color::argb8}, surface_format{surface_depth::z16_float}, surface_format{surface_depth::z24s8_float}})
		for (bool swizzled : {false, true})
		{
			auto memory = std::make_shared<memory_fixture>();
			auto budget = std::make_shared<heap_budget>(ctx, 1 << 20);
			surface_cache cache(ctx, memory, 1 << 20, budget);
			cached_surface_description description{0x1000, {4, 4, 1, format, false}, 0, {}};
			const auto bytes = bpp(description.allocation);
			if (swizzled)
				description.morton = swizzle_region{{8, 8, 1}, {2, 2, 0}, {4, 4, 1}};
			else
				description.row_pitch = 4 * bytes + 4;
			const auto coverage = plan_guest_surface(description.address, 4, 4, bytes, description.row_pitch, description.morton);
			const auto surface = cache.acquire(description);
			cache.prepare(surface, surface_initialization::background);
			cache.flush(surface);
			const auto revision = cache.status(surface).guest_revision;
			const auto whole_revision = memory->revision(coverage.range);
			const auto matches = [&](const cached_surface_description& candidate)
			{ return candidate.address == description.address && candidate.allocation.format == format; };
			const auto original = cache.snapshot_texture(coverage.range, matches, budget);
			require(bool(original) && !cache.status(surface).dirty, "Clean native active-coverage texture was not resolved");
			const auto padding = std::uint32_t(0x1000 + (swizzled ? 0 : 4 * bytes));
			const std::array sentinel{std::byte{0x5a}};
			memory->write({padding, 1}, sentinel);
			require(memory->surface_revision(coverage) == revision && memory->revision(coverage.range) != whole_revision,
					"Padding fixture did not separate active-pixel revision from whole guest storage");
			cache.spill(surface);
			const auto spill_submits = cache.transaction_statistics().submitted;
			const auto spill_allocations = budget->statistics().created;
			const auto padded = cache.snapshot_texture(coverage.range, matches, budget);
			require(padded == original && cache.snapshot_statistics().created == 1 && cache.snapshot_statistics().reused == 1 &&
				cache.status(surface).spilled && cache.memory_statistics().leased_allocations == 0 && cache.status(surface).guest_revision == revision &&
				cache.transaction_statistics().submitted == spill_submits && budget->statistics().created == spill_allocations,
					"Padding-only VM write discarded revision-valid native spill/texture data");
			// Independent native handles remain usable after a real active-pixel
			// VM write invalidates the cache's clean source for future resolution.
			const auto pixel = std::uint32_t(0x1000 + (swizzled ? morton(2, 2, 0, {8, 8, 1}) * bytes : 0));
			memory->write({pixel, 1}, sentinel);
			require(memory->surface_revision(coverage) != revision && !cache.snapshot_texture(coverage.range, matches, budget),
					"Active-pixel VM write did not retire a stale clean native texture source");
			const auto output = make_buffer(ctx, 1024, buffer_storage::shared);
			texture_region whole;
			whole.size = MTLSizeMake(4, 4, 1);
			for (const auto& texture : {original, padded})
			{
				ctx.begin();
				texture_to_buffer(ctx, texture, whole, output, 0, 256, 0, std::holds_alternative<surface_depth>(format) ? texture_plane::depth : texture_plane::all);
				ctx.submit().wait();
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
						require(static_cast<const std::uint32_t*>(output.contents)[y * 64 + x] == (std::holds_alternative<surface_depth>(format) ? std::bit_cast<std::uint32_t>(1.f) : 0xff000000u),
								"Clean native snapshot changed after padding/active VM writes");
			}
			++count;
		}
	std::cout << count << " clean native active-byte coverage cases: linear padding/Morton exterior revisions, VM writeback/spill, snapshot survival and active-write retirement passed\n";
}
} // namespace
namespace
{
class publication_provider final : public exclusive_guest_memory, public exclusive_guest_memory::view,
	public std::enable_shared_from_this<publication_provider>
{
	std::mutex mutex;
	std::array<std::byte, 262144> bytes{};
  public:
	unsigned notifications = 0;
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* nested = guest_memory_epoch::current(*this)) { operation(*nested); return; }
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		operation(*this);
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		require(guest_memory_epoch::current(*this) == this, "Publication fixture escaped provider exclusion");
		validate_guest_surface_range(range);
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Publication fixture bounds");
		return bytes.data() + range.address;
	}
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(a, b); }
	void written(guest_surface_range) noexcept override { ++notifications; }
};
void atomic_cpu_writers()
{
	unsigned cases = 0;
	for (const bool disjoint : {false, true})
		for (const bool transfer_capture : {false, true})
			for (const bool throwing : {false, true})
			{
				context ctx{1};
				auto provider = std::make_shared<publication_provider>();
				auto memory = std::make_shared<guarded_guest_memory>(provider);
				std::function<void()> observe;
				bool stored = false;
				unsigned depth = 0, observers = 0;
				const cpu_capture_dispatch capture = [&](const auto& operation)
				{
					require(depth == 0, "CPU writer recursively dispatched capture");
					++depth;
					struct scope { unsigned& depth; ~scope() { --depth; } } active{depth};
					std::thread next;
					std::exception_ptr failure;
					provider->execute([&](auto&)
					{
						operation();
						if (stored && observe)
						{
							next = std::thread([&]
							{
								try { provider->execute([&](auto&) { observe(); ++observers; }); }
								catch (...) { failure = std::current_exception(); }
							});
						}
					});
					if (next.joinable()) next.join();
					if (failure) std::rethrow_exception(failure);
				};
				surface_cache cache(ctx, memory, 8 << 20, {}, transfer_capture && !disjoint ? cpu_capture_dispatch{} : capture);
				cached_surface_description description{65536, {4, 4, 1, surface_color::argb8, false}, 16, {}};
				if (disjoint) description.segments = {{65536, 0, 17}, {131072, 17, 47}};
				const auto target = cache.acquire(description);
				cache.clear(target, 0xf0, 0xff223344, 0, 255, {0, 0, 4, 4});
				if (disjoint) cache.spill(target);
				const std::vector<guest_surface_range> ranges = disjoint ? std::vector<guest_surface_range>{{65536, 17}, {131072, 47}} : std::vector<guest_surface_range>{{65536, 64}};
				observe = [&]
				{
					const auto state = cache.status(target);
					require(!state.initialized && !state.dirty && !state.spilled && !state.faulted && !state.pending,
						"Next writer entered before CPU-store mirror invalidation");
					require(memory->read({ranges[0].address, 1}).bytes[0] == std::byte{0x6d}, "CPU store prefix disappeared before next writer");
					memory->write({ranges[0].address + 1, 1}, std::array{std::byte{0xd9}});
				};
				const auto mutation = [&]
				{
					require(guest_memory_epoch::current(*provider) == provider.get(), "CPU store escaped capture exclusion");
					for (std::size_t i = 0; i < ranges.size(); ++i)
					{
						memory->write({ranges[i].address, 1}, std::array{std::byte(0x6d + i)});
						stored = true;
						if (throwing) throw std::runtime_error("producer failed after prefix");
					}
				};
				const auto store = [&]
				{
					// Also models the previous transfer writer contract,which
					// acquired its provider epoch inside the producer callback.
					if (guest_memory_epoch::current(*provider)) mutation();
					else checked_cpu_capture(capture, mutation);
				};
				bool rejected = false;
				try { cache.cpu_write(ranges, store, transfer_capture ? capture : cpu_capture_dispatch{}); }
				catch (const std::runtime_error& error)
				{
					require(std::string_view(error.what()) == "producer failed after prefix", error.what());
					rejected = true;
				}
				require(rejected == throwing && observers == 1, "CPU producer completion/next-writer count differs");
				observe = {};
				cache.prepare(target);
				cache.flush(target);
				std::vector<std::byte> bytes;
				for (const auto range : ranges)
				{
					const auto part = memory->read(range).bytes;
					bytes.insert(bytes.end(), part.begin(), part.end());
				}
				for (unsigned i = 0; i < bytes.size(); ++i)
				{
					const auto expected = i == 0 ? std::byte{0x6d} : i == 1 ? std::byte{0xd9} :
						disjoint && !throwing && i == 17 ? std::byte{0x6e} : std::array{std::byte{0xff}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}}[i % 4];
					require(bytes[i] == expected, "CPU producer/next writer bytes overwritten during native reload");
				}
				++cases;
			}
	std::cout << cases << " CPU writer publication cases: configured/transfer capture, contiguous/disjoint storage, dirty/spilled aliases and throwing prefixes invalidate mirrors before a competing provider writer; subsequent native reload preserves all guest bytes\n";
}
void atomic_publication()
{
	for (const bool disjoint : {false, true})
	{
		context ctx{1};
		auto provider = std::make_shared<publication_provider>();
		auto memory = std::make_shared<guarded_guest_memory>(provider);
		std::function<bool()> boundary;
		std::function<void()> before;
		std::function<void()> observe;
		unsigned depth = 0, observers = 0;
		const cpu_capture_dispatch capture = [&](const auto& operation)
		{
			require(depth == 0, "Publication recursively dispatched a nested CPU capture");
			++depth;
			struct scope { unsigned& depth; ~scope() { --depth; } } active{depth};
			std::thread writer;
			std::exception_ptr failure;
			provider->execute([&](auto&)
			{
				if (auto work = std::exchange(before, {})) work();
				operation();
				if (boundary && boundary())
				{
					boundary = {};
					writer = std::thread([&]
					{
						try { provider->execute([&](auto&) { observe(); ++observers; }); }
						catch (...) { failure = std::current_exception(); }
					});
				}
			});
			// Force the next legitimate writer to finish after exclusion ends,
			// before dispatch returns. A check-then-commit gap fails deterministically.
			if (writer.joinable()) writer.join();
			if (failure) std::rethrow_exception(failure);
		};
		surface_cache cache(ctx, memory, 8 << 20, {}, capture);
		cached_surface_description description{65536, {4, 4, 1, surface_color::argb8, false}, 16, {}};
		if (disjoint) description.segments = {{65536, 0, 17}, {131072, 17, 47}};
		const auto layout = plan_guest_surface(description.address, 4, 4, 4, 16);
		const auto parts = disjoint ? project_guest_segments(layout, description.segments, false) : std::vector{guest_storage_part{layout, 0}};
		const auto target = cache.acquire(description);
		cache.prepare(target);
		const auto matches_revisions = [&](const cached_surface_status& state)
		{
			require(state.guest_revisions.size() == parts.size(), "Published watch count differs from physical parts");
			for (std::size_t i = 0; i < parts.size(); ++i)
				require(state.guest_revisions[i] == memory->surface_revision(parts[i].coverage), "Writer observed unpublished guest watch revisions");
		};
		id<MTLSharedEvent> gate = [ctx.device() newSharedEvent];
		struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 2; } } cleanup{gate};
		cache.set_wait_timeout(std::chrono::milliseconds(1));
		bool timed_out = false;
		const std::array uses{cached_surface_use{target, true}};
		try { cache.draw(uses, [&](context& commands, auto) { commands.wait_event(gate, 1); }); }
		catch (const std::runtime_error&) { timed_out = true; }
		require(timed_out && cache.status(target).pending, "Atomic publication fixture did not gate real GPU work");
		observe = [&]
		{
			const auto state = cache.status(target);
			require(state.initialized && state.dirty && !state.pending, "Next writer entered before GPU metadata publication");
			matches_revisions(state);
			memory->write({description.address + 4, 1}, std::array{std::byte{0x7a}});
		};
		boundary = [] { return true; };
		gate.signaledValue = 1;
		cache.set_wait_timeout(std::chrono::seconds(30));
		cache.synchronize();
		require(observers == 1, "Completion-boundary writer was not observed");
		bool conflict = false;
		try { cache.flush(target); }
		catch (const surface_conflict_error&) { conflict = true; }
		require(conflict && memory->read({description.address + 4, 1}).bytes[0] == std::byte{0x7a}, "Post-publication guest write was overwritten");
		cache.clear(target, 0xf0, 0xff223344, 0, 255, {0, 0, 4, 4});
		const auto notifications = provider->notifications;
		boundary = [&, notifications] { return provider->notifications != notifications; };
		observe = [&]
		{
			const auto state = cache.status(target);
			require(state.initialized && !state.dirty && !state.pending, "Next writer entered between VM writeback and clean metadata publication");
			matches_revisions(state);
			memory->write({description.address + 4, 1}, std::array{std::byte{0x9c}});
		};
		cache.flush(target);
		require(observers == 2 && !boundary, "Writeback-boundary writer did not execute exactly once");
		std::vector<std::byte> result;
		if (disjoint)
			for (const auto segment : description.segments)
			{
				const auto data = memory->read({segment.address, segment.size}).bytes;
				result.insert(result.end(), data.begin(), data.end());
			}
		else result = memory->read({description.address, 64}).bytes;
		for (unsigned byte = 0; byte < 64; ++byte)
			require(result[byte] == (byte == 4 ? std::byte{0x9c} : std::array{std::byte{0xff}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}}[byte % 4]), "Atomic writeback/post-publication writer bytes differ");
		cache.prepare(target);
		cache.set_wait_timeout(std::chrono::milliseconds(1));
		timed_out = false;
		try { cache.draw(uses, [&](context& commands, auto) { commands.wait_event(gate, 2); }); }
		catch (const std::runtime_error&) { timed_out = true; }
		require(timed_out && cache.status(target).pending, "Fault publication fixture did not gate GPU work");
		before = [&] { memory->write({description.address + 4, 1}, std::array{std::byte{0xa7}}); };
		boundary = [] { return true; };
		observe = [&]
		{
			const auto state = cache.status(target);
			require(!state.faulted && !state.spilled && state.initialized && !state.pending, "Next writer entered before rejected GPU output was rolled back");
			require(memory->read({description.address + 4, 1}).bytes[0] == std::byte{0xa7}, "Rejected output replaced the pre-publication writer");
		};
		gate.signaledValue = 2;
		cache.set_wait_timeout(std::chrono::seconds(30));
		conflict = false;
		try { cache.synchronize(); }
		catch (const surface_conflict_error&) { conflict = true; }
		require(conflict && observers == 3 && !cache.status(target).faulted && cache.status(target).initialized, "Rollback publication omitted rejection or next-writer observation");
		cache.recover(target);
		cache.prepare(target);
		cache.spill(target);
		require(cache.status(target).spilled && cache.memory_statistics().leased_allocations == 0 && cache.memory_statistics().reserved_bytes == 0,
			"Publication retirement prevented post-transaction pool trim");
		cache.prepare(target);
		cache.erase(target);
	}
	std::cout << "Final revision/metadata,fault metadata and VM-CAS/clean metadata publish in one nonrecursive CPU epoch;next real provider writers observe completed contiguous/disjoint GPU state,post-publication bytes survive,retired allocations trim outside publication\n";
}
} // namespace
void backing_lifetime()
{
	auto memory = std::make_shared<memory_fixture>();
	std::vector<std::weak_ptr<void>> owners;
	std::vector<guest_surface_range> requested;
	std::function<void()> mutate_on_retain;
	memory->backing = [&](std::span<const guest_surface_range> ranges)
	{
		for (const auto range : ranges) requested.push_back(range);
		auto owner = std::make_shared<int>(1);
		owners.push_back(owner);
		if (auto mutate = std::exchange(mutate_on_retain, {})) mutate();
		return owner;
	};
	const auto retired = [&] { return std::all_of(owners.begin(), owners.end(), [](const auto& owner) { return owner.expired(); }); };
	{
		context ctx{1};
		surface_cache cache(ctx, memory, 8 << 20, {}, [](const auto& operation) { operation(); });
		const auto source = cache.acquire({4096, {4, 4, 1, surface_color::argb8, false}, 16, {}});
		cache.clear(source, 0xf0, 0xff123456, 0, 255, {0, 0, 4, 4});
		require(!owners.empty() && !owners.back().expired() && requested.back().address == 4096 && requested.back().size == 64,
			"Dirty native output did not retain its exact physical backing request");
		const auto dirty = owners.back();
		cache.spill(source);
		require(!dirty.expired(), "Native spill released dirty guest backing before writeback");
		cache.flush(source);
		require(retired(), "Completed writeback retained clean guest backing");
		cache.clear(source, 0xf0, 0xffabcdef, 0, 255, {0, 0, 4, 4});
		const auto destination = cache.acquire({8192, {4, 4, 1, surface_color::argb8, false}, 16, {}});
		cache.defer_copy(source, {0, 0, 4, 4}, destination, 0, 0);
		const auto deferred = owners.back();
		require(!deferred.expired() && requested.back().address == 8192, "Deferred native output lost destination backing");
		cache.cancel_deferred_copies(destination);
		require(deferred.expired(), "Cancelled clean deferred output retained its backing");
		cache.flush(source);
		cache.prepare(destination);
		require(retired(), "Clean VM upload retained backing after GPU completion");
		for (const unsigned profile : {0u, 1u, 2u})
		{
			cached_surface_description description{16384, {3, 2, 1, surface_color::argb8, false}, 20, {}};
			if (profile == 1) { description.row_pitch = 0; description.morton = swizzle_region{{8, 8, 1}, {1, 2, 0}, {3, 2, 1}}; }
			if (profile == 2) description.address = 65536;
			const auto layout = plan_guest_surface(description.address, 3, 2, 4, description.row_pitch, description.morton);
			if (profile == 2) description.segments = {{65536, 0, 17}, {131072, 17, layout.range.size - 17}};
			const auto active = cache.acquire(description);
			requested.clear();
			cache.clear(active, 0xf0, 0xff456789, 0, 255, {0, 0, 3, 2});
			std::vector<bool> expected(1 << 20), observed(expected.size());
			for (unsigned y = 0; y < 2; ++y)
				for (unsigned x = 0; x < 3; ++x)
					for (unsigned lane = 0; lane < 4; ++lane)
					{
						const auto offset = profile == 1 ? morton(x + 1, y + 2, 0, {8, 8, 1}) * 4 + lane : y * 20 + x * 4 + lane;
						const auto address = profile != 2 ? description.address + offset : offset < 17 ? 65536 + offset : 131072 + offset - 17;
						expected[address] = true;
					}
			for (const auto range : requested)
				for (std::size_t byte = 0; byte < range.size; ++byte) observed.at(range.address + byte) = true;
			require(expected == observed, "Backing capture included padding or omitted active Morton/disjoint bytes");
			cache.erase(active);
			require(retired(), "Erased native output retained physical backing");
		}
		bool cancelled = false;
		const std::array uses{cached_surface_use{destination, true}};
		try { cache.draw(uses, [](auto&, auto) { throw std::runtime_error("cancelled encoder"); }); }
		catch (const std::runtime_error&) { cancelled = true; }
		require(cancelled && retired() && !cache.status(destination).pending && !cache.status(destination).dirty,
			"Unsubmitted cancellation retained guest backing or published output");
		bool conflict = false;
		try
		{
			cache.draw(uses, [&](auto&, auto) { memory->write({8192, 1}, std::array{std::byte{0x4a}}); });
		}
		catch (const surface_conflict_error&) { conflict = true; }
		require(conflict && retired() && !cache.status(destination).pending && !cache.status(destination).dirty,
			"Unmapped-descriptor CPU write escaped pre-submit backing revision validation");
		bool entered = false;
		mutate_on_retain = [&] { memory->write({8192, 1}, std::array{std::byte{0x5b}}); };
		conflict = false;
		try { cache.draw(uses, [&](auto&, auto) { entered = true; }); }
		catch (const surface_conflict_error&) { conflict = true; }
		require(conflict && !entered && retired(), "Backing-acquisition write reached GPU encoding or leaked owners");
	}
	{
		auto ctx = std::make_unique<context>(1);
		auto cache = std::make_unique<surface_cache>(*ctx, memory, 8 << 20);
		auto target = cache->acquire({12288, {4, 4, 1, surface_color::argb8, false}, 16, {}});
		cache->prepare(target);
		id<MTLSharedEvent> gate = [ctx->device() newSharedEvent];
		const auto marker = make_buffer(*ctx, 4, buffer_storage::shared);
		std::optional<submission> work;
		bool timeout = false;
		cache->set_wait_timeout(std::chrono::milliseconds(1));
		const std::array uses{cached_surface_use{target, true}};
		try
		{
			cache->draw(uses, [&](context& commands, auto)
			{
				commands.wait_event(gate, 1);
				fill_buffer(commands, marker, 0, 4, 0x6c);
				work = commands.recording_submission();
			});
		}
		catch (const std::runtime_error&) { timeout = true; }
		require(timeout && cache->status(target).pending && !owners.back().expired() && work && !work->ready(),
			"Blocked GPU submission did not retain guest backing through timeout");
		const auto held = owners.back();
		cache.reset(); target.reset(); ctx.reset();
		require(!held.expired(), "Cache/context destruction released backing before GPU completion");
		gate.signaledValue = 1;
		work->wait();
		require(held.expired() && retired(), "GPU completion did not retire destroyed-cache backing owners");
		const auto output = map_buffer(marker, 0, 4);
		require(std::all_of(output.begin(), output.end(), [](auto byte) { return byte == std::byte{0x6c}; }), "Retained GPU work did not execute after cache/context destruction");
	}
	std::cout << "Guest backing owners survive dirty spill,deferred output,GPU timeout and cache/context destruction;writeback,cancel,revision conflict and actual GPU completion retire them\n";
}
void typed_wait_timeout_completion()
{
    using namespace metal4;using namespace std::chrono_literals;
    context ctx(2);auto memory=std::make_shared<memory_fixture>();surface_cache cache(ctx,memory,16u<<20);surface_color_transfer colors(ctx);
    const auto surface=cache.acquire({0x1000,{4,4,1,surface_color::argb8,false},16,{}});
    cache.clear(surface,0xf0,0xff123456,0,255,{0,0,4,4});cache.flush(surface);
    unsigned timeouts=0,ready_at_catch=0;
    for(unsigned trial=0;trial<256;++trial)
    {
        std::vector<std::byte> expected(64);for(unsigned i=0;i<64;++i)expected[i]=std::byte((trial*37+i*17+73)&255);
        const auto before=cache.status(surface).content_generation;const auto vm=memory->read({0x1000,64}).bytes;
        std::optional<submission> work;cache.set_wait_timeout(0ms);
        try
        {
            cache.draw(std::array{cached_surface_use{surface,true}},[&](context& commands,auto images)
            {work=commands.recording_submission();texture_region all;all.size=MTLSizeMake(4,4,1);colors.upload(commands,surface_color::argb8,make_upload(commands,expected),{},images[0],all);});
        }
        catch(const submission_wait_timeout&)
        {
            ++timeouts;require(work&&cache.status(surface).pending&&cache.status(surface).content_generation==before,"Typed wait timeout lost native pending transaction orpublished metadata");
            ready_at_catch+=work->ready();work->wait(30s);
            require(work->ready()&&cache.status(surface).pending&&cache.status(surface).content_generation==before&&memory->read({0x1000,64}).bytes==vm,"Completed timed-out work lost pending ownership orpublished before explicit retry");
        }
        cache.set_wait_timeout(30s);cache.synchronize(30s);
        require(work&&work->ready()&&!cache.status(surface).pending&&!cache.status(surface).faulted&&cache.status(surface).content_generation==before+1,"Explicit timeout completion retry rejected valid GPU work orchanged generation twice");
        cache.flush(surface);require(memory->read({0x1000,64}).bytes==expected,"Late native completion retry lost exact GPU pixels at CPU observation");
    }
    require(timeouts>0,"Zero-deadline native completion fixture observed no CPU timeouts");
    std::cout<<"Typed native wait deadlines:256 actual GPU writes,"<<timeouts<<" typed CPU timeouts,"<<ready_at_catch<<" ready at caller catch;after GPUcompletion,pending owners/old generations/VM retained until explicit retry,one generation andexact pixels,no false GPU failure passed\n";
}
void draw_target_loads(bool spilled)
{
	using namespace std::chrono_literals;
	for (unsigned profile = 0; profile < 5; ++profile)
	{
		context ctx(2);
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 8 << 20);
		clear_engine clears(ctx);
		const bool depth = profile == 2;
		const bool swizzled = profile == 1;
		const bool cancel = profile == 3;
		const bool supersede = profile == 4;
		const unsigned samples = spilled ? (depth ? 2u : swizzled ? 4u : 1u) : 1u;
		const auto grid = samples == 1 ? sample_layout{1,1} : rsx_sample_layout(samples);
		const std::array<std::size_t,3> storage{4 * grid.x, 4 * grid.y, 1};
		const surface_format format = depth ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		cached_surface_description description{0x1000, {4,4,samples,format,false}, swizzled ? 0u : storage[0] * 4 + 1, {}};
		if (swizzled) description.morton = swizzle_region{storage, {}, storage};
		const auto layout = plan_guest_surface(description.address, storage[0], storage[1], 4, description.row_pitch, description.morton);
		const auto offset = [&](unsigned x, unsigned y)
		{
			return swizzled ? 4 * morton(x,y,0,storage) : y * layout.row_pitch + x * 4;
		};
		const std::uint32_t seed = depth ? 0x7d00002au : 0xff112233u;
		const std::uint32_t painted = depth ? 0x7e000035u : 0xff345678u;
		std::vector<std::byte> bytes(layout.range.size, std::byte{0xa5});
		for (unsigned y = 0; y < storage[1]; ++y)
			for (unsigned x = 0; x < storage[0]; ++x)
			{
				const auto sample = x % grid.x + y % grid.y * grid.x;
				const auto value = seed + static_cast<std::uint32_t>(sample) * (depth ? 0x00000101u : 0x00010203u);
				store(std::span{bytes}.subspan(offset(x,y), 4), value, 4);
			}
		memory->write(layout.range, bytes);
		const auto target = cache.acquire(description);
		const auto native_seed = depth ? 0x7f00002au : 0xff789abcu;
		if (spilled)
		{
			cache.prepare(target);
			if (!supersede)
				cache.clear(target, depth ? 3 : 0xf0, native_seed, 0xffffff2a, 255,
					cancel ? clear_rectangle{0,0,4,4} : clear_rectangle{0,3,1,1});
			cache.spill(target);
			require(cache.status(target).spilled && memory->read_surface(layout).bytes == bytes,
				"Fixture spill published guest bytes");
		}
		const auto target_before = cache.status(target);
		const auto prefix_target = cache.acquire({0x8000, {4,4,1,surface_color::argb8,false},16,{}});
		cache.clear(prefix_target, 0xf0, 0xff112233, 0, 255, {0,0,4,4});
		const auto before = cache.transaction_statistics();
		const auto gate = [ctx.device() newSharedEvent];
		struct release_event { id<MTLSharedEvent> event; ~release_event() { event.signaledValue = 1; } } release{gate};
		auto batch = cache.batch_draws();
		cache.clear(prefix_target, 0xf0, 0xff405060, 0, 255, {0,0,4,4});
		const auto token = ctx.recording_submission();
		const std::array uses{cached_surface_use{target,true}};
		for (unsigned row = 0; row < 2; ++row)
		{
			cache.draw(uses, [&](context& commands, auto images)
			{
				if (row == 0) commands.wait_event(gate, 1);
				clear_request request;
				request.rectangle = {1,row,2,1};
				request.color_value = painted;
				request.zstencil_value = painted;
				request.stencil_write_mask = 255;
				request.mask = depth ? 3 : 0xf0;
				if (depth)
				{
					request.depth = {images[0],0,0};
					request.depth_format = surface_depth::z24s8_float;
				}
				else request.colors[0] = {{images[0],0,0},surface_color::argb8};
				(void)clears.run(commands, request);
			});
			const auto state = cache.status(target);
			require(token.same_batch(ctx.recording_submission()) && state.initialized == target_before.initialized &&
				state.spilled == target_before.spilled && state.content_generation == target_before.content_generation &&
				cache.transaction_statistics().submitted == before.submitted,
				"Draw target published or submitted while recording its load and partial writers");
		}
		batch.submit();
		require(!cache.poll_submitted() && cache.transaction_statistics().submitted == before.submitted + 1,
			"Cold target and independent prefix did not share a gated submission");
		if (cancel) require(token.cancel_pending(), "Cold-target GPU work escaped its cancellation gate");
		if (supersede)
		{
			bytes[offset(0,0)] = std::byte{0x77};
			memory->write(layout.range, bytes);
		}
		if (!cancel) gate.signaledValue = 1;
		bool failed = false;
		try { cache.synchronize(); }
		catch (const submission_wait_timeout&) { throw; }
		catch (const surface_conflict_error&) { require(supersede, "Unexpected cold-target revision conflict"); failed = true; }
		catch (const std::runtime_error&) { require(cancel, "Unexpected cold-target GPU failure"); failed = true; }
		if (cancel || supersede)
		{
			const auto state = cache.status(target);
			require(failed && state.initialized == target_before.initialized && state.dirty == target_before.dirty &&
				state.spilled == target_before.spilled && !state.faulted && !state.pending &&
				state.content_generation == target_before.content_generation && memory->read_surface(layout).bytes == bytes,
				"Rejected draw target retained provisional native state or changed guest bytes");
			if (cancel) ctx.recover_completed_errors();
			cache.prepare(target, surface_initialization::guest);
			require(cache.status(target).initialized && !cache.status(target).spilled && cache.status(target).dirty == target_before.dirty,
				"Rejected draw target could not reload its authoritative contents");
			if (spilled && cancel)
			{
				cache.flush(target);
				for (unsigned y = 0; y < storage[1]; ++y)
					for (unsigned x = 0; x < storage[0]; ++x)
						store(std::span{bytes}.subspan(offset(x,y), 4), native_seed, 4);
				require(memory->read_surface(layout).bytes == bytes, "Cancelled draw lost the dirty spill's original bytes");
			}
			continue;
		}
		const auto state = cache.status(target);
		require(!failed && state.initialized && !state.spilled && state.dirty &&
			state.content_generation == target_before.content_generation + (spilled ? 2u : 3u),
			"Draw target load and two partial writes did not publish once in order");
		cache.flush(target);
		const auto stored = depth
			? (std::bit_cast<std::uint32_t>(float(painted >> 8) / 16777215.0f) >> 7 << 8) | (painted & 255)
			: painted;
		for (unsigned y = 0; y < storage[1]; ++y)
			for (unsigned x = 0; x < storage[0]; ++x)
			{
				const auto pixel_x = x / grid.x, pixel_y = y / grid.y;
				const bool covered = pixel_y < 2 && pixel_x >= 1 && pixel_x < 3;
				if (covered || (spilled && pixel_x == 0 && pixel_y == 3))
					store(std::span{bytes}.subspan(offset(x,y), 4), covered ? stored : native_seed, 4);
			}
		if (memory->read_surface(layout).bytes != bytes)
			throw std::runtime_error("Draw target profile " + std::to_string(profile) + " changed the expected guest bytes");
	}
	std::cout << (spilled ? "Spilled" : "Cold")
		<< " draw targets: 3 exact byte profiles in one submission each, cancellation and VM revision rollback/reload passed\n";
}

void deferred_draw_targets()
{
	using namespace std::chrono_literals;
	for (unsigned profile = 0; profile < 7; ++profile)
	{
		context ctx(2);
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 8 << 20);
		clear_engine clears(ctx);
		const bool full = profile == 0, spilled = profile == 2, self = profile == 3;
		const bool cancel = profile == 4, supersede = profile == 5;
		const bool depth = profile == 6;
		const unsigned a = depth ? 0x7f000035 : 0xff123456, b = depth ? 0x7e000067 : 0xff654321;
		const unsigned seed = depth ? 0x7d000010 : 0xff102030, native = 0xff485868;
		const unsigned draw = depth ? 0x7d000053 : 0xffa1b2c3;
		const surface_format format = depth ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		const auto layout = plan_guest_surface(0x1000, 4, 4, 4, 16);
		std::vector<std::byte> bytes(layout.range.size);
		for (unsigned i = 0; i < 16; ++i) store(std::span{bytes}.subspan(i * 4, 4), seed + i, 4);
		memory->write(layout.range, bytes);
		const auto target = cache.acquire({0x1000, {4,4,1,format,false},16,{}});
		const auto first = cache.acquire({0x2000, {4,4,1,format,false},16,{}});
		const auto second = cache.acquire({0x3000, {4,4,1,format,false},16,{}});
		const auto prefix = cache.acquire({0x4000, {4,4,1,surface_color::argb8,false},16,{}});
		cache.clear(first, depth ? 3 : 0xf0, a, 0xffffff35, 255, {0,0,4,4});
		cache.clear(second, depth ? 3 : 0xf0, b, 0x80000067, 255, {0,0,4,4});
		cache.clear(prefix, 0xf0, seed, 0, 255, {0,0,4,4});
		if (spilled)
		{
			cache.clear(target, 0xf0, native, 0, 255, {0,0,4,4});
			cache.spill(target);
		}
		if (full)
		{
			cache.defer_copy(first, {0,0,2,4}, target, 0, 0);
			cache.defer_copy(second, {0,0,2,4}, target, 2, 0);
		}
		else if (self)
		{
			cache.defer_copy(target, {0,0,1,4}, target, 1, 0);
			cache.defer_copy(first, {0,0,1,4}, target, 2, 0);
			cache.defer_copy(second, {0,0,1,1}, target, 2, 3);
		}
		else
		{
			cache.defer_copy(first, {0,0,2,3}, target, 1, 0, depth ? texture_plane::depth : texture_plane::all);
			cache.defer_copy(second, {0,0,1,2}, target, 2, 1, depth ? texture_plane::stencil : texture_plane::all);
		}
		const auto target_before = cache.status(target);
		const auto submitted = cache.transaction_statistics().submitted;
		const auto gate = [ctx.device() newSharedEvent];
		struct unblock { id<MTLSharedEvent> gate; ~unblock() { gate.signaledValue = 1; } } release{gate};
		auto batch = cache.batch_draws();
		cache.clear(prefix, 0xf0, native, 0, 255, {0,0,4,4});
		const auto token = ctx.recording_submission();
		const std::array uses{cached_surface_use{target,true}};
		for (const unsigned pixel : {0u,3u})
		{
			cache.draw(uses, [&](context& commands, auto images)
			{
				if (pixel == 0) commands.wait_event(gate, 1);
				clear_request request;
				request.rectangle = {pixel,pixel,1,1};
				request.color_value = draw;
				request.mask = depth ? 3 : 0xf0;
				if (depth)
				{
					request.depth = {images[0],0,0};
					request.depth_format = surface_depth::z24s8_float;
					request.zstencil_value = 0x40000053;
					request.stencil_write_mask = 255;
				}
				else request.colors[0] = {{images[0],0,0},surface_color::argb8};
				(void)clears.run(commands, request);
			});
			const auto pending = cache.status(target);
			require(token.same_batch(ctx.recording_submission()) && pending.deferred_copies == target_before.deferred_copies &&
				pending.content_generation == target_before.content_generation && cache.transaction_statistics().submitted == submitted,
				"Deferred target submitted early, repeated preparation or exposed its new generation");
		}
		batch.submit();
		require(!cache.poll_submitted() && cache.transaction_statistics().submitted == submitted + 1,
			"Deferred target, its draws and unrelated prefix did not share a gated submission");
		if (cancel) require(token.cancel_pending(), "Deferred target escaped its cancellation gate");
		if (supersede)
		{
			bytes[0] = std::byte{0x77};
			memory->write(layout.range, bytes);
		}
		if (!cancel) gate.signaledValue = 1;
		bool failed = false;
		try { cache.synchronize(); }
		catch (const submission_wait_timeout&) { throw; }
		catch (const surface_conflict_error&) { require(supersede, "Unexpected deferred target VM conflict"); failed = true; }
		catch (const std::runtime_error&) { require(cancel, "Unexpected deferred target GPU failure"); failed = true; }
		if (cancel || supersede)
		{
			const auto state = cache.status(target);
			require(failed && state.initialized == target_before.initialized && state.dirty == target_before.dirty &&
				state.spilled == target_before.spilled && !state.faulted && !state.pending &&
				state.deferred_copies == target_before.deferred_copies && state.content_generation == target_before.content_generation &&
				memory->read_surface(layout).bytes == bytes, "Rejected deferred target lost its frozen sources or changed VM");
			if (supersede)
			{
				cache.cancel_deferred_copies(target);
				continue;
			}
			ctx.recover_completed_errors();
		}
		else
		{
			const auto state = cache.status(target);
			require(!failed && state.initialized && state.dirty && !state.spilled && state.deferred_copies == 0 &&
				state.content_generation == target_before.content_generation + (profile == 1 || depth ? 4u : 3u),
				"Deferred target did not publish copies and both writes in order");
		}
		cache.flush(target);
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
			{
				auto value = spilled ? native : seed + y * 4 + x;
				if (full) value = x < 2 ? a : b;
				else if (self)
				{
					if (x == 1) value = seed + y * 4;
					if (x == 2) value = y == 3 ? b : a;
				}
				else
				{
					if (x >= 1 && x <= 2 && y < 3) value = depth ? (a & ~255u) | (value & 255u) : a;
					if (x == 2 && y >= 1 && y < 3) value = depth ? (value & ~255u) | (b & 255u) : b;
				}
				if (!cancel && ((x == 0 && y == 0) || (x == 3 && y == 3))) value = draw;
				store(std::span{bytes}.subspan((y * 4 + x) * 4, 4), value, 4);
			}
		require(memory->read_surface(layout).bytes == bytes, "Deferred draw changed frozen background, copy order or exact draw bytes");
	}
	std::cout << "Deferred draw targets: 5 exact byte profiles in one submission each, cancellation and VM revision rollback passed\n";
}

void clear_target_loads()
{
	for (unsigned profile = 0; profile < 5; ++profile)
	{
		context ctx(2);
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 8 << 20);
		const bool depth = profile == 1, deferred = profile == 2, stale = profile == 3, cancel = profile == 4;
		const surface_format format = depth ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		const auto layout = plan_guest_surface(0x1000,4,4,4,17);
		std::vector<std::byte> bytes(layout.range.size, std::byte{0xa5});
		const auto paint = [&](unsigned x, unsigned y, std::uint32_t value)
		{
			store(std::span{bytes}.subspan(y * layout.row_pitch + x * 4, 4), value, 4);
		};
		const std::uint32_t seed = depth ? 0x7d00002a : 0xff112233;
		const std::uint32_t cleared = depth ? 0x7d000025 : 0xff6789ab;
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
				paint(x, y, seed);
		memory->write(layout.range, bytes);
		const auto target = cache.acquire({layout.range.address,{4,4,1,format,false},layout.row_pitch,{}});
		if (depth || stale)
		{
			cache.prepare(target);
			if (depth) cache.clear(target, 2, 0, 0x2f, 255, {0,3,1,1});
			cache.spill(target);
			if (stale)
			{
				paint(0, 3, 0xff776655);
				memory->write(layout.range, bytes);
			}
		}
		if (deferred)
		{
			const auto donor = cache.acquire({0x2000,{4,4,1,surface_color::argb8,false},16,{}});
			cache.clear(donor, 0xf0, 0xff776655, 0, 255, {0,0,4,4});
			cache.defer_copy(donor, {0,0,2,1}, target, 0, 3);
		}
		const auto partial_clear = [&] { cache.clear(target, depth ? 2 : 0xf0, cleared, 5, 0x0f, {1,0,2,2}); };
		if (cancel)
		{
			const auto prefix = cache.acquire({0x4000,{4,4,1,surface_color::argb8,false},16,{}});
			cache.clear(prefix, 0xf0, seed, 0, 255, {0,0,4,4});
			const auto gate = [ctx.device() newSharedEvent];
			struct release_event { id<MTLSharedEvent> gate; ~release_event() { gate.signaledValue = 1; } } release{gate};
			auto batch = cache.batch_draws();
			cache.draw(std::array{cached_surface_use{prefix,true}},
				[&](context& commands, auto) { commands.wait_event(gate, 1); });
			const auto token = ctx.recording_submission();
			const auto submitted = cache.transaction_statistics().submitted;
			partial_clear();
			cache.clear(target, 0xf0, cleared, 0, 255, {3,3,1,1});
			require(token.same_batch(ctx.recording_submission()) && !cache.status(target).initialized &&
				cache.transaction_statistics().submitted == submitted, "Partial clear split its target load or published the cold target early");
			batch.submit();
			require(token.cancel_pending(), "Partial clear escaped the cancellation gate");
			bool rejected = false;
			try { cache.synchronize(); } catch (const submission_wait_timeout&) { throw; } catch (const std::runtime_error&) { rejected = true; }
			const auto state = cache.status(target);
			require(rejected && !state.initialized && !state.dirty && !state.faulted && state.content_generation == 0 &&
				memory->read_surface(layout).bytes == bytes, "Cancelled partial clear changed target authority or guest bytes");
			ctx.recover_completed_errors();
			continue;
		}
		const auto original = cache.status(target);
		const auto submitted = cache.transaction_statistics().submitted;
		partial_clear();
		const auto state = cache.status(target);
		require(cache.transaction_statistics().submitted == submitted + 1 && state.initialized && state.dirty &&
			!state.spilled && state.deferred_copies == 0 &&
			state.content_generation == original.content_generation + (deferred ? 3u : depth ? 1u : 2u),
			"Partial clear did not commit its load and clear in one submission");
		require(memory->read_surface(layout).bytes == bytes, "Partial clear wrote back before explicit flush");
		cache.flush(target);
		for (unsigned y = 0; y < 2; ++y)
			for (unsigned x = 1; x <= 2; ++x)
				paint(x, y, cleared);
		if (depth) paint(0, 3, 0x7d00002f);
		if (deferred) { paint(0, 3, 0xff776655); paint(1, 3, 0xff776655); }
		require(memory->read_surface(layout).bytes == bytes, "Partial clear changed untouched pixels, aspect or padding");
	}
	std::cout << "Clear target loads: cold, spilled, deferred and stale VM profiles passed exact bytes in one submission each; cancelled batch kept prior authority\n";
}

void deferred_reader_pressure()
{
	context ctx(2);
	const surface_allocation_description allocation{16,16,1,surface_color::argb8,false};
	std::size_t cost;
	{
		surface_pool probe(ctx, 1 << 20);
		cost = probe.acquire(allocation)->reserved_bytes();
	}
	auto memory = std::make_shared<memory_fixture>();
	surface_cache cache(ctx, memory, cost * 2);
	const auto writer = cache.acquire({0x1000, allocation, 64, {}});
	const auto reader = cache.acquire({0x2000, allocation, 64, {}});
	cache.clear(writer, 0xf0, 0xff112233, 0, 255, {0,0,16,16});
	cache.clear(reader, 0xf0, 0xff000000, 0, 255, {0,0,16,16});
	cache.defer_copy(writer, {0,0,16,16}, reader, 0, 0);
	const auto before = cache.transaction_statistics();
	cache.draw(std::array{cached_surface_use{writer,true}}, [](context& commands, auto images)
	{
		auto pass = [MTL4RenderPassDescriptor new];
		pass.colorAttachments[0].texture = images[0];
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].storeAction = MTLStoreActionStore;
		pass.colorAttachments[0].clearColor = MTLClearColorMake(1,0,0,1);
		(void)commands.render(pass);
		commands.end_encoding();
	});
	require(cache.status(reader).deferred_copies == 0 && cache.status(reader).dirty &&
		cache.transaction_statistics().raw_backup_submissions == before.raw_backup_submissions + 2,
		"In-place pressure fallback failed to finish the reader before changing its donor");
	cache.flush(reader);
	cache.flush(writer);
	const auto exact = [&](std::uint32_t address, std::uint32_t color)
	{
		std::vector<std::byte> expected(1024);
		for (unsigned i = 0; i < 256; ++i) store(std::span{expected}.subspan(i * 4, 4), color, 4);
		return memory->read({address,1024}).bytes == expected;
	};
	require(exact(0x1000, 0xffff0000) && exact(0x2000, 0xff112233),
		"In-place fallback changed the deferred reader's captured bytes");
	std::cout << "Deferred reader pressure: raw in-place fallback finished reader before writer and retained exact bytes\n";
}

void resident_target_reload()
{
	for (unsigned profile = 0; profile < 4; ++profile)
	{
		context ctx(2);
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 8 << 20);
		clear_engine clears(ctx);
		const bool clear = profile == 1, cancel = profile == 2, dirty = profile == 3;
		const auto layout = plan_guest_surface(0x1000,4,4,4,17);
		auto filled = [&](std::uint32_t value)
		{
			std::vector<std::byte> data(layout.range.size, std::byte{0xa5});
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 4; ++x)
					store(std::span{data}.subspan(y * 17 + x * 4, 4), value, 4);
			return data;
		};
		const auto old = filled(0xff112233);
		auto fresh = filled(0xff789abc);
		memory->write(layout.range, old);
		const auto target = cache.acquire({0x1000,{4,4,1,surface_color::argb8,false},17,{}});
		const auto reader = cache.acquire({0x2000,{4,4,1,surface_color::argb8,false},17,{}});
		cache.prepare(target);
		if (dirty) cache.clear(target, 0xf0, 0xff445566, 0, 255, {0,0,4,4});
		if (!clear && !dirty) cache.defer_copy(target, {0,0,4,4}, reader, 0, 0);
		const auto before = cache.status(target);
		memory->write(layout.range, fresh);
		const auto submitted = cache.transaction_statistics().submitted;
		if (dirty)
		{
			bool conflict = false;
			auto batch = cache.batch_draws();
			try { cache.draw(std::array{cached_surface_use{target,true}}, [](context&, auto) {}); }
			catch (const surface_conflict_error&) { conflict = true; }
			batch.finish();
			require(conflict && cache.transaction_statistics().submitted == submitted &&
				cache.status(target).dirty && cache.status(target).content_generation == before.content_generation,
				"Resident reload replaced dirty native authority after a VM write");
			continue;
		}
		const auto gate = [ctx.device() newSharedEvent];
		struct unblock { id<MTLSharedEvent> gate; ~unblock() { gate.signaledValue = 1; } } release{gate};
		if (clear)
			cache.clear(target, 0xf0, 0xffabcdef, 0, 255, {1,0,2,2});
		else
		{
			auto batch = cache.batch_draws();
			std::optional<submission> token;
			for (unsigned y = 0; y < 2; ++y)
			{
				cache.draw(std::array{cached_surface_use{target,true}}, [&](context& commands, auto images)
				{
					if (y == 0) commands.wait_event(gate, 1);
					clear_request request;
					request.mask = 0xf0;
					request.color_value = 0xffabcdef;
					request.rectangle = {1,y,2,1};
					request.colors[0] = {{images[0],0,0}, surface_color::argb8};
					(void)clears.run(commands, request);
				});
				if (!token) token = ctx.recording_submission();
				require(token->same_batch(ctx.recording_submission()) && cache.transaction_statistics().submitted == submitted &&
					cache.status(target).content_generation == before.content_generation && cache.status(reader).deferred_copies == 1,
					"Resident reload split the draw or retired the frozen reader prematurely");
			}
			batch.submit();
			require(!cache.poll_submitted(), "Resident reload bypassed the pending GPU event");
			if (cancel) require(token->cancel_pending(), "Resident reload bypassed cancellation");
			else gate.signaledValue = 1;
			bool rejected = false;
			try { cache.synchronize(); } catch (const submission_wait_timeout&) { throw; } catch (const std::runtime_error&) { rejected = true; }
			require(rejected == cancel && cache.status(reader).deferred_copies == 1, "Resident reload lost the old reader");
			if (cancel) ctx.recover_completed_errors();
			cache.flush(reader);
			const auto old_read = memory->read({0x2000,layout.range.size}).bytes;
			for (unsigned y = 0; y < 4; ++y)
				require(std::equal(old.begin() + y * 17, old.begin() + y * 17 + 16, old_read.begin() + y * 17),
					"Reader saw the new VM revision or new draw instead of its original pixels");
		}
		const auto state = cache.status(target);
		require(cache.transaction_statistics().submitted >= submitted + 1 && state.dirty == !cancel &&
			state.content_generation == before.content_generation + (cancel ? 0u : clear ? 2u : 3u) &&
			memory->read_surface(layout).bytes == fresh,
			"Resident reload published stale authority or wrote back early");
		if (clear)
			require(cache.transaction_statistics().submitted == submitted + 1, "Resident clear made a separate upload submission");
		if (!cancel)
		{
			cache.flush(target);
			for (unsigned y = 0; y < 2; ++y)
				for (unsigned x = 1; x <= 2; ++x)
					store(std::span{fresh}.subspan(y * 17 + x * 4, 4), 0xffabcdef, 4);
			require(memory->read_surface(layout).bytes == fresh, "Resident reload lost the newer VM background or an earlier draw");
		}
	}
	std::cout << "Resident target reload: exact draw and clear bytes, frozen reader and rollback, dirty conflict refused\n";
}

void prefetched_draw_targets(bool partial_clear = false)
{
	using namespace std::chrono_literals;
	const unsigned profiles = partial_clear ? 5u : 4u;
	for (unsigned profile = 0; profile < profiles; ++profile)
	{
		const bool superseded = profile == 1, alias = profile == 2, cancel = profile == 3, pressure = profile == 4;
		context ctx(2);
		auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
		ctx.set_buffer_budget(budget);
		auto memory = std::make_shared<memory_fixture>();
		id<MTLBuffer> blocker = nil;
		std::vector<id<MTLBuffer>> pressure_tail;
		std::optional<submission> release_budget_after;
		const cpu_capture_dispatch capture = [&](const auto& work)
		{
			if (release_budget_after && release_budget_after->ready())
			{
				blocker = nil;
				pressure_tail.clear();
				release_budget_after.reset();
			}
			work();
		};
		surface_cache cache(ctx, memory, 8 << 20, budget, capture);
		clear_engine clears(ctx);
		const auto layout = plan_guest_surface(0x1000, 4, 4, 4, 16);
		const auto target = cache.acquire({0x1000, {4,4,1,surface_color::argb8,false}, 16, {}});
		const auto prior = cache.acquire({alias ? 0x1000u : 0x2000u, {2,4,1,surface_color::argb8,false}, 16, {}});
		std::array<std::byte, 64> initial, newer;
		for (unsigned i = 0; i < 16; ++i)
		{
			store(std::span{initial}.subspan(i * 4, 4), 0xff123456, 4);
			store(std::span{newer}.subspan(i * 4, 4), 0xff456789, 4);
		}
		memory->write(layout.range, initial);
		const auto gate = [ctx.device() newSharedEvent];
		struct open_gate { id<MTLSharedEvent> event; ~open_gate() { event.signaledValue = 1; } } release{gate};
		auto prior_batch = cache.batch_draws();
		cache.draw(std::array{cached_surface_use{prior, true}}, [&](context& commands, auto)
		{
			commands.wait_event(gate, 1);
		});
		cache.clear(prior, 0xf0, 0xff112233, 0, 255, {0,0,2,4});
		const auto predecessor = ctx.recording_submission();
		prior_batch.submit();
		require(!predecessor.ready(), "Prefetch predecessor was not gated");
		unsigned reclaims = 0;
		const auto reclaimer = budget->register_reclaimer([&](memory_pressure_request) { ++reclaims; });
		if (pressure)
		{
			const auto usage = budget->statistics();
			// Account for the large allocation's page rounding, then consume
			// enough of the tail to refuse a buffer of the target's exact size.
			const auto available = usage.budget - usage.reserved_bytes;
			blocker = budget->shared_buffer(available & ~std::size_t(16383));
			while (auto buffer = budget->try_shared_buffer(layout.range.size)) pressure_tail.push_back(buffer);
			release_budget_after = predecessor;
		}
		const auto allocations = budget->statistics().created;
		std::atomic<bool> staged_while_pending = false;
		std::jthread release_after_staging;
		if (!alias && !cancel && !pressure)
			release_after_staging = std::jthread([&](std::stop_token stop)
			{
				while (!stop.stop_requested() && budget->statistics().created == allocations)
					std::this_thread::sleep_for(100us);
				if (stop.stop_requested()) return;
				staged_while_pending = !predecessor.ready();
				gate.signaledValue = 1;
			});

		unsigned reads = 0;
		bool early = false, cancelled = false, encoded = false, cancelled_before_slot = false;
		memory->after_read = [&](guest_surface_range range)
		{
			if (range.address != layout.range.address || range.size != layout.range.size || ++reads != 1) return;
			early = !predecessor.ready();
			if (superseded) memory->write(layout.range, newer);
			if (cancel)
			{
				cancelled = predecessor.cancel_pending();
				if (partial_clear && cancelled)
				{
					// This fixture holds no VM lock. Force cancellation to finish
					// before the partial clear attempts to open its own slot.
					try { predecessor.wait(2s); }
					catch (const submission_wait_timeout&) { throw; }
					catch (const std::runtime_error&) { cancelled_before_slot = predecessor.ready(); }
				}
			}
			if (pressure) gate.signaledValue = 1;
		};
		const auto draw = [&]
		{
			auto next = cache.batch_draws();
			if (partial_clear)
			{
				cache.clear(target, 0xf0, 0xffaabbcc, 0, 255, {3,3,1,1});
				encoded = true;
			}
			else cache.draw(std::array{cached_surface_use{target, true}}, [&](context& commands, auto images)
			{
				encoded = true;
				clear_request request;
				request.colors[0] = {{images[0], 0, 0}, surface_color::argb8};
				request.mask = 0xf0;
				request.color_value = 0xffaabbcc;
				request.rectangle = {3, 3, 1, 1};
				(void)clears.run(commands, request);
			});
			next.finish();
		};
		cache.set_wait_timeout(alias ? 1ms : 2s);
		bool timed_out = false, failed = false;
		try { draw(); }
		catch (const submission_wait_timeout&) { timed_out = true; }
		catch (const std::runtime_error&) { if (!cancel) throw; failed = true; }
		if (alias || cancel)
		{
			const auto state = cache.status(target);
			require(timed_out == alias && failed == cancel && cancelled == cancel && early == cancel &&
				reads == unsigned(cancel) && !encoded && !state.initialized && !state.dirty && !state.faulted,
				"Aliased or cancelled predecessor exposed a provisional target");
			if (partial_clear && cancel) require(cancelled_before_slot, "Partial clear did not exercise cancellation before slot start");
			if (cancel) ctx.recover_completed_errors();
			gate.signaledValue = 1;
			cache.set_wait_timeout(5s);
			draw();
		}
		else if (pressure)
			require(!timed_out && early && !blocker && !reclaims && reads == 1,
				"Early allocation pressure reclaimed resources or lost the captured guest bytes");
		else
			require(!timed_out && early && staged_while_pending && reads == (superseded ? 2u : 1u),
				"Cold target did not stage a Metal upload while the predecessor was gated or refresh a changed revision");
		memory->after_read = {};
		auto expected = superseded ? newer : initial;
		if (alias)
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 2; ++x)
					store(std::span{expected}.subspan((y * 4 + x) * 4, 4), 0xff112233, 4);
		store(std::span{expected}.subspan(60, 4), 0xffaabbcc, 4);
		const auto actual = cache.cpu_read(layout.range).bytes;
		require(encoded && std::ranges::equal(actual, expected), "Prefetched target lost alias, VM revision or draw pixels");
	}
	std::cout << "Cold target staging for " << (partial_clear ? "partial clear" : "draw")
		<< ": " << profiles << " gated profiles, " << profiles * 64
		<< " exact guest bytes, early Metal upload, revision reuse/refresh, alias wait and cancellation passed\n";
	if (partial_clear) std::cout << "Cold partial clear: full early budget, CPU snapshot reused after completion without reclaiming, cancellation before slot start passed\n";
}

enum class overlap_writer { full_clear, partial_color, partial_stencil, draw, mixed_draw, resident_draw,
	spilled_draw, deferred_draw, deferred_spill_draw, deferred_guest_draw, deferred_guest_clear,
	resident_clear, spilled_clear, deferred_spill_clear };
void overlapping_target_recording(overlap_writer writer)
{
	using namespace std::chrono_literals;
	const bool partial = writer != overlap_writer::full_clear;
	const bool stencil = writer == overlap_writer::partial_stencil;
	const bool mixed = writer == overlap_writer::mixed_draw;
	const bool resident = writer == overlap_writer::resident_draw;
	const bool resident_clear = writer == overlap_writer::resident_clear;
	const bool spilled_clear = writer == overlap_writer::spilled_clear;
	const bool deferred_spill_clear = writer == overlap_writer::deferred_spill_clear;
	const bool native_clear = resident_clear || spilled_clear || deferred_spill_clear;
	const bool spilled = writer == overlap_writer::spilled_draw || writer == overlap_writer::deferred_spill_draw;
	const bool deferred_guest = writer == overlap_writer::deferred_guest_draw || writer == overlap_writer::deferred_guest_clear;
	const bool deferred_clear = writer == overlap_writer::deferred_guest_clear || deferred_spill_clear;
	const bool deferred = writer == overlap_writer::deferred_draw || writer == overlap_writer::deferred_spill_draw || deferred_guest || deferred_clear;
	const bool draw = !deferred_clear && (writer == overlap_writer::draw || mixed || resident || spilled || deferred);
	enum class scenario { success, timeout, cancelled, revised, alias, one_slot };
	for (const auto test : {scenario::success, scenario::timeout, scenario::cancelled,
		scenario::revised, scenario::alias, scenario::one_slot})
	{
		// The draw variant places its deferred target in the independent slot
		// when testing an alias on the other target.
		if (deferred_clear && test == scenario::alias) continue;
		const bool rejects_revision = deferred_clear && test == scenario::revised;
		context ctx(test == scenario::one_slot ? 1u : 2u, 64, 5s, context::recording_overlap::explicit_predecessor);
		auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
		ctx.set_buffer_budget(budget);
		auto memory = std::make_shared<memory_fixture>();
		const auto gate = [ctx.device() newSharedEvent];
		struct open_gate { id<MTLSharedEvent> event; ~open_gate() { event.signaledValue = 1; } } release{gate};
		std::optional<submission> predecessor;
		std::uint64_t first_encoders = 0, first_compute = 0;
		bool observed_recording = false, observed_publication = false, cancel_accepted = false;
		const auto layout = plan_guest_surface(0x1000, 4, 4, 4, 16);
		const auto companion_layout = plan_guest_surface(0x2000, 4, 4, 4, 16);
		const surface_format format = stencil ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		const std::uint32_t revised_pixel = stencil ? 0x7c000044 : 0xff212223;
		std::array<std::byte, 64> initial;
		for (unsigned i = 0; i < 16; ++i) store(std::span{initial}.subspan(i * 4, 4), (stencil ? 0x7d00002a : 0xff506070) + i, 4);
		memory->write(layout.range, initial);
		if (draw) memory->write(companion_layout.range, initial);
		const cpu_capture_dispatch capture = [&](const auto& work)
		{
			if (predecessor && !observed_recording && !predecessor->ready())
			{
				try { observed_recording = !ctx.recording_submission().same_batch(*predecessor); }
				catch (const std::logic_error&) {}
				if (observed_recording && (test == scenario::success || test == scenario::revised)) gate.signaledValue = 1;
				if (observed_recording && test == scenario::cancelled) cancel_accepted = predecessor->cancel_pending();
			}
			const auto stats = ctx.encoding_statistics();
			if (predecessor && !observed_publication && predecessor->ready() && observed_recording &&
				(draw ? stats.compute_encoders > first_compute : stats.render_encoders > first_encoders))
			{
				observed_publication = true;
				if (test == scenario::revised)
				{
					std::array<std::byte, 4> guest_write;
					store(guest_write, revised_pixel, 4);
					memory->write({layout.range.address, guest_write.size()}, guest_write);
				}
			}
			work();
		};
		surface_cache cache(ctx, memory, 8 << 20, budget, capture);
		clear_engine clears(ctx);
		const auto target = cache.acquire({layout.range.address, {4,4,1,format,false},16,{}});
		const auto companion = draw ? cache.acquire({companion_layout.range.address, {4,4,1,format,false},16,{}}) : cached_surface_ref{};
		if (resident)
		{
			cache.prepare(target);
			cache.prepare(companion);
		}
		if (mixed || spilled) cache.clear(companion, 0xf0, 0xff765432, 0, 255, {0,0,4,4});
		if (spilled) cache.spill(companion);
		if (native_clear) cache.prepare(target);
		if (deferred_spill_clear && !rejects_revision) cache.clear(target, 0xf0, 0xff765432, 0, 255, {0,0,4,4});
		if (spilled_clear || deferred_spill_clear) cache.spill(target);
		cached_surface_ref donor;
		if (deferred)
		{
			donor = cache.acquire({0x4000, {4,4,1,format,false},16,{}});
			cache.clear(donor, 0xf0, 0xff38495a, 0, 255, {0,0,4,4});
			const auto area = spilled || deferred_guest || deferred_spill_clear ? clear_rectangle{0,0,1,2} : clear_rectangle{0,0,4,4};
			cache.defer_copy(donor, area, deferred_clear ? target : companion, 0, 0);
		}
		const bool rewrites_donor = deferred && test != scenario::alias;
		const auto prior = rewrites_donor ? donor : cache.acquire({test == scenario::alias ? layout.range.address : 0x3000u,
			{2,4,1,format,false},16,{}});
		const auto prior_before = cache.status(prior);
		const auto target_before = cache.status(target);
		const auto companion_before = companion ? cache.status(companion) : cached_surface_status{};
		auto previous_batch = cache.batch_draws();
		cache.draw(std::array{cached_surface_use{prior, true}}, [&](context& commands, auto)
		{
			commands.wait_event(gate, 1);
		});
		cache.clear(prior, stencil ? 2 : 0xf0, 0xff112233, 0x33, 255, {0,0,rewrites_donor ? 4u : 2u,4});
		const auto old = ctx.recording_submission();
		previous_batch.submit();
		predecessor = old;
		first_encoders = ctx.encoding_statistics().render_encoders;
		first_compute = ctx.encoding_statistics().compute_encoders;
		const auto before = cache.transaction_statistics();
		const bool blocks = test == scenario::timeout || test == scenario::alias || test == scenario::one_slot;
		cache.set_wait_timeout(blocks ? 2ms : 2s);
		auto next_batch = cache.batch_draws();
		const auto area = partial ? clear_rectangle{1,1,2,2} : clear_rectangle{0,0,4,4};
		unsigned draw_calls = 0;
		const auto clear = [&]
		{
			if (!draw)
			{
				cache.clear(target, stencil ? 2 : 0xf0, 0xff345678, 8, stencil ? 0x0f : 255, area);
				return;
			}
			const auto requirements = std::array{*clear_engine::upload_for(format, 0xf0, 255, false)};
			const auto plan = ctx.prepare_uploads(requirements);
			cache.draw(std::array{cached_surface_use{target, true}, cached_surface_use{companion, true}},
				[&](context& commands, auto images)
				{
					++draw_calls;
					require(!predecessor || (predecessor->ready() && cache.status(prior).initialized),
						"Draw callback ran before predecessor publication");
					clear_request request;
					request.colors[0] = {{images[0], 0, 0}, surface_color::argb8};
					request.colors[1] = {{images[1], 0, 0}, surface_color::argb8};
					request.mask = 0xf0;
					request.color_value = 0xff345678;
					request.rectangle = area;
					commands.with_uploads(plan, [&] { (void)clears.run(commands, request); });
				});
		};
		bool timed_out = false, cancelled = false, rejected_revision = false;
		try { clear(); }
		catch (const submission_wait_timeout&) { timed_out = true; }
		catch (const surface_conflict_error&) { if (!rejects_revision) throw; rejected_revision = true; }
		catch (const std::runtime_error&) { if (test != scenario::cancelled) throw; cancelled = true; }
		const bool overlaps = test != scenario::alias && test != scenario::one_slot;
		const auto encoded = ctx.encoding_statistics().render_encoders - first_encoders;
		const bool discards = test == scenario::timeout || test == scenario::cancelled || test == scenario::revised;
		const auto staged = ctx.encoding_statistics().compute_encoders - first_compute;
		const unsigned attempts = unsigned(overlaps) + unsigned(test == scenario::revised && !rejects_revision);
		const unsigned rendered = draw ? unsigned(test == scenario::success || test == scenario::revised) : attempts;
		const auto prepared = cache.transaction_statistics();
		const auto target_after = cache.status(target);
		require(timed_out == blocks && cancelled == (test == scenario::cancelled) && observed_recording == overlaps &&
			rejected_revision == rejects_revision &&
			encoded == rendered && staged == (partial ? attempts : 0) && (!draw || draw_calls == rendered) &&
			prepared.cancelled_draw_batches == before.cancelled_draw_batches + unsigned(discards) &&
			prepared.submitted == before.submitted && target_after.initialized == target_before.initialized &&
			target_after.content_generation == target_before.content_generation && target_after.deferred_copies == target_before.deferred_copies &&
			target_after.spilled == target_before.spilled,
			"Early target preparation did not record privately, retry a changed revision, or respect the alias/slot boundary");
		if (resident_clear)
			require(prepared.native_version_blits - before.native_version_blits == unsigned(overlaps),
				"Early partial clear did not preserve the resident allocation through a native copy");
		if (draw)
		{
			const auto companion_after = cache.status(companion);
			const unsigned expected_blits = mixed ? attempts : resident ? unsigned(overlaps) * 2 + unsigned(test == scenario::revised) : 0;
			require(prepared.native_version_blits - before.native_version_blits == expected_blits &&
				companion_after.content_generation == companion_before.content_generation && companion_after.dirty == companion_before.dirty &&
				companion_after.spilled == companion_before.spilled && companion_after.deferred_copies == companion_before.deferred_copies,
				"Early target preparation did not copy the resident versions privately");
		}
		if (test == scenario::cancelled)
		{
			require(cancel_accepted && cache.status(prior).initialized == prior_before.initialized &&
				cache.status(prior).content_generation == prior_before.content_generation && !cache.status(target).faulted,
				"Predecessor cancellation changed surface authority");
			ctx.recover_completed_errors();
		}
		if (test == scenario::success || test == scenario::revised)
			require(observed_publication && cache.status(prior).initialized, "Clear preceded its predecessor's publication");
		predecessor.reset();
		if (rejects_revision) cache.cancel_deferred_copies(target);
		if (blocks || cancelled || rejects_revision)
		{
			gate.signaledValue = 1;
			cache.set_wait_timeout(2s);
			clear();
		}
		next_batch.finish();
		auto expected = initial;
		if (deferred_spill_clear && !rejects_revision)
			for (unsigned i = 0; i < 16; ++i) store(std::span{expected}.subspan(i * 4, 4), 0xff765432, 4);
		if (test == scenario::revised) store(std::span{expected}.first(4), revised_pixel, 4);
		if (deferred_clear && !rejects_revision)
			for (unsigned y = 0; y < 2; ++y)
				store(std::span{expected}.subspan(y * 16, 4), 0xff38495a, 4);
		if (test == scenario::alias)
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 2; ++x)
				{
					auto word = std::span{expected}.subspan((y * 4 + x) * 4, 4);
					if (stencil) word[3] = std::byte{0x33};
					else store(word, 0xff112233, 4);
				}
		for (std::size_t y = area.y; y < area.y + area.height; ++y)
			for (std::size_t x = area.x; x < area.x + area.width; ++x)
			{
				auto word = std::span{expected}.subspan((y * 4 + x) * 4, 4);
				if (stencil) word[3] = (word[3] & std::byte{0xf0}) | std::byte{8};
				else store(word, 0xff345678, 4);
			}
		require(std::ranges::equal(cache.cpu_read(layout.range).bytes, expected), "Early clear changed guest pixel bytes");
		if (draw)
		{
			auto companion_expected = initial;
			if (mixed || spilled)
				for (unsigned i = 0; i < 16; ++i) store(std::span{companion_expected}.subspan(i * 4, 4), 0xff765432, 4);
			if (deferred)
				for (unsigned y = 0; y < (spilled || deferred_guest ? 2u : 4u); ++y)
					for (unsigned x = 0; x < (spilled || deferred_guest ? 1u : 4u); ++x)
						store(std::span{companion_expected}.subspan((y * 4 + x) * 4, 4), 0xff38495a, 4);
			for (std::size_t y = area.y; y < area.y + area.height; ++y)
				for (std::size_t x = area.x; x < area.x + area.width; ++x)
					store(std::span{companion_expected}.subspan((y * 4 + x) * 4, 4), 0xff345678, 4);
			require(draw_calls == 1 && std::ranges::equal(cache.cpu_read(companion_layout.range).bytes, companion_expected),
				"Early draw lost its second target or consumed the upload plan twice");
		}
	}
	if (draw)
	{
		const auto label = deferred_guest ? "deferred guest draw" : deferred ? (spilled ? "deferred spill draw" : "deferred draw") :
			spilled ? "spilled draw" : mixed ? "mixed draw" : resident ? "resident draw" : "draw";
		std::cout << "Early " << label
			<< " target recording: six gated strict-context profiles, 768 exact guest bytes across two targets, single-use draw upload after predecessor publication passed\n";
		return;
	}
	if (native_clear)
	{
		const auto label = resident_clear ? "resident" : spilled_clear ? "spilled" : "deferred spill";
		std::cout << "Early native " << label << " partial clear: " << (deferred_spill_clear ? 5 : 6)
			<< " gated profiles, " << (deferred_spill_clear ? 320 : 384) << " exact guest bytes passed\n";
		return;
	}
	if (deferred_clear)
	{
		std::cout << "Early deferred guest partial clear: five gated profiles, 320 exact guest bytes, stale deferred revision rejected before retry passed\n";
		return;
	}
	std::cout << "Early " << (stencil ? "partial stencil" : partial ? "partial color" : "full")
		<< " clear recording: six gated strict-context profiles, 384 exact guest bytes, success, timeout, cancellation, revision retry, alias and slot boundaries passed\n";
}

void full_clear_batch_targets()
{
	for (unsigned profile = 0; profile < 5; ++profile)
	{
		context ctx(2);
		auto memory = std::make_shared<memory_fixture>();
		surface_cache cache(ctx, memory, 8 << 20);
		const bool depth = profile == 1, spilled = profile == 2, deferred = profile == 3, cancel = profile == 4;
		const surface_format format = depth ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		const auto layout = plan_guest_surface(0x1000,4,4,4,17);
		std::vector<std::byte> bytes(layout.range.size, std::byte{0xa5});
		memory->write(layout.range, bytes);
		const auto target = cache.acquire({0x1000,{4,4,1,format,false},17,{}});
		const auto prefix = cache.acquire({0x3000,{4,4,1,surface_color::argb8,false},16,{}});
		const auto companion = cache.acquire({0x4000,{4,4,1,surface_color::argb8,false},16,{}});
		cache.clear(prefix, 0xf0, 0xff102030, 0, 255, {0,0,4,4});
		if (spilled)
		{
			cache.clear(target, 0xf0, 0xff112233, 0, 255, {0,0,4,4});
			cache.spill(target);
		}
		if (deferred)
			cache.defer_copy(prefix, {0,0,2,2}, target, 0, 0);
		const auto before = cache.status(target);
		const auto stats = cache.transaction_statistics();
		const auto gate = [ctx.device() newSharedEvent];
		struct unblock { id<MTLSharedEvent> gate; ~unblock() { gate.signaledValue = 1; } } release{gate};
		auto batch = cache.batch_draws();
		cache.draw(std::array{cached_surface_use{prefix,true}},
			[&](context& commands, auto) { commands.wait_event(gate, 1); });
		const auto token = ctx.recording_submission();
		cache.clear(target, depth ? 3 : 0xf0, 0xff234567, 0xffffff2a, 255, {0,0,4,4});
		cache.clear(companion, 0xf0, 0xff345678, 0, 255, {0,0,4,4});
		cache.clear(target, depth ? 2 : 0xf0, 0xffaabbcc, 0x35, 255, {1,1,2,2});
		const auto recording = cache.status(target);
		require(token.same_batch(ctx.recording_submission()) && cache.transaction_statistics().submitted == stats.submitted &&
			recording.initialized == before.initialized && recording.spilled == before.spilled &&
			recording.deferred_copies == before.deferred_copies && recording.content_generation == before.content_generation,
			"Full clear split the prefix or published provisional target state");
		batch.submit();
		cache.set_wait_timeout(std::chrono::milliseconds(1));
		bool read_waited = false, draw_waited = false, clear_waited = false, miss_waited = false, encoded = false;
		try { (void)cache.cpu_read(layout.range); } catch (const submission_wait_timeout&) { read_waited = true; }
		{
			auto next = cache.batch_draws();
			require(cache.acquire(cache.description(target)) == target && cache.acquire(cache.description(companion)) == companion,
				"Descriptor hit could not prepare the next batch during GPU work");
			cache.clear(target, depth ? 3 : 0xf0, 0, 0, 255, {4,4,1,1});
			const std::array uses{cached_surface_use{target, true}};
			try { cache.draw(uses, [&](auto&, auto) { encoded = true; }); } catch (const submission_wait_timeout&) { draw_waited = true; }
			try { cache.clear(target, depth ? 3 : 0xf0, 0, 0, 255, {0,0,4,4}); } catch (const submission_wait_timeout&) { clear_waited = true; }
			try { (void)cache.acquire({0x5000,{4,4,1,surface_color::argb8,false},16,{}}); } catch (const submission_wait_timeout&) { miss_waited = true; }
			next.submit();
		}
		cache.set_wait_timeout(std::chrono::seconds(30));
		const auto pending = cache.status(target);
		require(read_waited && draw_waited && clear_waited && miss_waited && !encoded && !cache.poll_submitted() && pending.pending &&
			pending.content_generation == before.content_generation && memory->read_surface(layout).bytes == bytes,
			"CPU read or next draw bypassed a pending full clear");
		bool rejected = false;
		if (cancel)
		{
			require(token.cancel_pending(), "Full clear escaped cancellation");
			try { cache.synchronize(); } catch (const submission_wait_timeout&) { throw; } catch (const std::runtime_error&) { rejected = true; }
		}
		else
		{
			gate.signaledValue = 1;
			token.wait();
			require(cache.poll_submitted() && !cache.status(target).pending, "Completed full clear did not retire during polling");
		}
		const auto state = cache.status(target);
		require(cache.transaction_statistics().submitted == stats.submitted + 1 &&
			rejected == cancel && state.initialized == !cancel && state.dirty == !cancel && !state.faulted &&
			!state.spilled && state.deferred_copies == 0 && state.content_generation == before.content_generation + (cancel ? 0u : 2u) &&
			cache.status(companion).initialized == !cancel && memory->read_surface(layout).bytes == bytes,
			"Full-clear batch published incomplete authority");
		if (cancel) { ctx.recover_completed_errors(); continue; }
		const auto actual = cache.cpu_read(layout.range).bytes;
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
			{
				const bool partial = x >= 1 && x <= 2 && y >= 1 && y <= 2;
				const auto value = depth ? (partial ? 0x7f000035u : 0x7f00002au) : (partial ? 0xffaabbccu : 0xff234567u);
				store(std::span{bytes}.subspan(y * 17 + x * 4, 4), value, 4);
			}
		require(actual == bytes, "Full-clear batch changed exact pixels, aspect or padding");
		cache.flush(companion);
		std::array<std::byte,64> companion_bytes;
		for (unsigned i = 0; i < 16; ++i) store(std::span{companion_bytes}.subspan(i * 4, 4), 0xff345678, 4);
		const auto actual_companion = memory->read({0x4000,64}).bytes;
		require(std::ranges::equal(actual_companion, companion_bytes), "Full-clear batch changed its second attachment");
	}
	std::cout << "Clear batch preparation: descriptor hits and empty clears returned; CPU read, draw, writing clear and descriptor miss waited; four exact attachment profiles and cancellation passed\n";
}

int main(int argc,char** argv)
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			if(argc==2&&std::string_view(argv[1])=="typed-timeouts"){typed_wait_timeout_completion();return 0;}
			if(argc==2&&std::string_view(argv[1])=="cold-draw-targets"){draw_target_loads(false);return 0;}
			if(argc==2&&std::string_view(argv[1])=="prefetched-draw-targets"){prefetched_draw_targets();return 0;}
			if(argc==2&&std::string_view(argv[1])=="prefetched-clear-targets"){prefetched_draw_targets(true);return 0;}
			if(argc==2&&std::string_view(argv[1])=="spilled-draw-targets"){draw_target_loads(true);return 0;}
			if(argc==2&&std::string_view(argv[1])=="deferred-draw-targets"){deferred_draw_targets();return 0;}
			if(argc==2&&std::string_view(argv[1])=="clear-target-loads"){clear_target_loads();return 0;}
			if(argc==2&&std::string_view(argv[1])=="deferred-reader-pressure"){deferred_reader_pressure();return 0;}
			if(argc==2&&std::string_view(argv[1])=="resident-target-reload"){resident_target_reload();return 0;}
			if(argc==2&&std::string_view(argv[1])=="full-clear-batch-targets"){full_clear_batch_targets();return 0;}
			if(argc==2&&std::string_view(argv[1])=="overlapping-full-clear"){overlapping_target_recording(overlap_writer::full_clear);return 0;}
			if(argc==2&&std::string_view(argv[1])=="overlapping-partial-clear")
			{
				overlapping_target_recording(overlap_writer::partial_color);
				overlapping_target_recording(overlap_writer::partial_stencil);
				return 0;
			}
			if(argc==2&&std::string_view(argv[1])=="overlapping-draw-targets"){overlapping_target_recording(overlap_writer::draw);return 0;}
			if(argc==2&&std::string_view(argv[1])=="overlapping-resident-targets")
			{
				overlapping_target_recording(overlap_writer::mixed_draw);
				overlapping_target_recording(overlap_writer::resident_draw);
				return 0;
			}
			if(argc==2&&std::string_view(argv[1])=="overlapping-spilled-targets")
			{
				overlapping_target_recording(overlap_writer::spilled_draw);
				overlapping_target_recording(overlap_writer::deferred_draw);
				overlapping_target_recording(overlap_writer::deferred_spill_draw);
				return 0;
			}
			if(argc==2&&std::string_view(argv[1])=="overlapping-deferred-guest-targets")
			{
				overlapping_target_recording(overlap_writer::deferred_guest_draw);
				overlapping_target_recording(overlap_writer::deferred_guest_clear);
				return 0;
			}
			if(argc==2&&std::string_view(argv[1])=="overlapping-native-clear-targets")
			{
				overlapping_target_recording(overlap_writer::resident_clear);
				overlapping_target_recording(overlap_writer::spilled_clear);
				overlapping_target_recording(overlap_writer::deferred_spill_clear);
				return 0;
			}
			if(argc!=1)throw std::invalid_argument("Unknown native surface-cache test selection");
			typed_wait_timeout_completion();
			draw_target_loads(false);
			prefetched_draw_targets();
			prefetched_draw_targets(true);
			draw_target_loads(true);
			deferred_draw_targets();
			clear_target_loads();
			deferred_reader_pressure();
			resident_target_reload();
			full_clear_batch_targets();
			overlapping_target_recording(overlap_writer::full_clear);
			overlapping_target_recording(overlap_writer::partial_color);
			overlapping_target_recording(overlap_writer::partial_stencil);
			overlapping_target_recording(overlap_writer::draw);
			overlapping_target_recording(overlap_writer::mixed_draw);
			overlapping_target_recording(overlap_writer::resident_draw);
			overlapping_target_recording(overlap_writer::spilled_draw);
			overlapping_target_recording(overlap_writer::deferred_draw);
			overlapping_target_recording(overlap_writer::deferred_spill_draw);
			overlapping_target_recording(overlap_writer::deferred_guest_draw);
			overlapping_target_recording(overlap_writer::deferred_guest_clear);
			overlapping_target_recording(overlap_writer::resident_clear);
			overlapping_target_recording(overlap_writer::spilled_clear);
			overlapping_target_recording(overlap_writer::deferred_spill_clear);
			atomic_cpu_writers();
			atomic_publication();
			backing_lifetime();
			context ctx(1);
			cases(ctx);
			aliases_and_vm(ctx);
			callbacks_and_timeout(ctx);
			background(ctx);
			native_write_versions(ctx);
			pressure_and_native_bits(ctx);
			active_alias_mrts(ctx);
			native_cache_copies(ctx);
			deferred_formats(ctx);
			deferred_lifetime(ctx);
			deferred_conflicts_and_overwrite(ctx);
			deferred_pending_and_budget();
			deferred_guest_aliases(ctx);
			cached_nearest_blits(ctx);
			cached_typeless_blits(ctx);
			cached_sample_blits(ctx);
			cached_feedback(ctx);
			cached_depth_bounds(ctx);
			spilled_snapshot_ready_cases();
			clean_texture_active_coverage(ctx);
			ctx.wait_idle();
		}
		catch (const surface_budget_error& e)
		{
			std::cerr << e.what() << " requested=" << e.requested << " reserved=" << e.reserved << " budget=" << e.budget << '\n';
			return 1;
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
