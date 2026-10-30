#include "M4Display.h"
#include "M4DrawExecutorFixtures.h"
#include "M4DrawPacketFixtures.h"
#include "M4Frame.h"
#include "M4FramebufferBinding.h"
#include "M4FramebufferFixtures.h"
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
class fixture_memory final : public guest_surface_memory
{
  public:
	std::vector<std::byte> bytes = std::vector<std::byte>(2 << 20);
	unsigned reads = 0;
	guest_surface_snapshot read_surface(const guest_surface_layout& layout) override
	{
		++reads;
		const auto r = layout.range;
		if (r.address > bytes.size() || r.size > bytes.size() - r.address)
			throw std::out_of_range("Lifetime memory range");
		return {1, {bytes.begin() + r.address, bytes.begin() + r.address + r.size}};
	}
	std::uint64_t surface_revision(const guest_surface_layout&) override
	{
		return 1;
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout&, std::uint64_t, std::span<const std::byte>) override
	{
		throw std::runtime_error("Unexpected lifetime VM write");
	}
	void write(guest_surface_range, std::span<const std::byte>) override
	{
		throw std::runtime_error("Unexpected lifetime VM write");
	}
};
std::vector<std::byte> pixels(context& ctx, id<MTLTexture> image)
{
	@autoreleasepool
	{
		const auto result = make_buffer(ctx, image.width * image.height * 4, buffer_storage::shared);
		texture_region region;
		region.size = MTLSizeMake(image.width, image.height, 1);
		ctx.begin();
		texture_to_buffer(ctx, image, region, result, 0, image.width * 4);
		ctx.submit().wait();
		const auto span = map_buffer(result, 0, result.length);
		return {span.begin(), span.end()};
	}
}
void verify(const std::vector<std::byte>& data, std::array<unsigned, 4> bgra)
{
	for (unsigned i = 0; i < data.size(); ++i)
		require(data[i] == std::byte(bgra[i % 4]), "Lifetime loop changed native pixels");
}
rsx_draw_packet packet()
{
	using namespace metal4::test::executor_fixture;
	test::pass_state state;
	state.registers = registers().registers;
	state.base_index = 0;
	state.index_width = 1;
	test::pass_processor processor;
	processor.descriptors[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27), (1u << 31) | (1u << 29)};
	processor.persistent.resize(1 << 20);
	std::vector<std::byte> positions;
	for (float v : {-1.f, -1.f, .5f, 1.f, 3.f, -1.f, .5f, 1.f, -1.f, 3.f, .5f, 1.f})
		for (unsigned shift : {24u, 16u, 8u, 0u})
			positions.push_back(std::byte(std::bit_cast<unsigned>(v) >> shift));
	std::copy(positions.begin(), positions.end(), processor.persistent.begin());
	for (unsigned index : {0u, 1u, 4094u})
	{
		processor.index_bytes.push_back(std::byte(index >> 8));
		processor.index_bytes.push_back(std::byte(index));
	}
	test::pass_layout layout;
	test::pass_metadata metadata;
	auto vp = vertex_program();
	auto code = program(false);
	test::fragment_program fp;
	fp.pointer = code.data();
	fp.ucode_length = code.size();
	fp.ctrl = rsx_fragment_float32;
	return rsx_draw_packet::capture(state, processor, layout, metadata, vp, fp, 0, provoking_vertex::last,
									[](const auto&, unsigned, unsigned)
									{ return std::pair<std::uint32_t, std::uint32_t>{1 << 20, 0}; });
}
} // namespace
int main(int argc, char** argv)
{
	@autoreleasepool
	{
		try
		{
			require(argc == 2, "Specify draw,clear or display lifetime mode");
			const std::string mode = argv[1];
			context ctx{1};
			auto memory = std::make_shared<fixture_memory>();
			auto budget = std::make_shared<heap_budget>(ctx, 12 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, memory, 12 << 20, budget);
			const unsigned count = 32;
			if (mode == "draw")
			{
				guest_texture_cache textures(ctx, memory, [](unsigned offset, unsigned, std::size_t)
											 { return offset; }, 1 << 20, budget);
				draw_executor executor(ctx, surfaces, textures);
				const auto owned = packet();
				test::framebuffer_layout layout;
				layout.color_addresses = {0x1000, 0, 0, 0};
				layout.actual_color_pitch[0] = 16;
				layout.zeta_address = 0;
				const auto framebuffer = framebuffer_snapshot::capture(layout, true);
				const auto targets = acquire_framebuffer(framebuffer, surfaces);
				surfaces.clear(targets.colors[0], 0xf0, 0xff000000, 0, 255, {0, 0, 4, 4});
				const auto generation = surfaces.status(targets.colors[0]).content_generation;
				for (unsigned i = 0; i < count; ++i)
					executor.execute(owned, framebuffer);
				require(surfaces.status(targets.colors[0]).content_generation == generation + count, "Repeated owned draw lost publication");
				const auto image = surfaces.snapshot_texture({0x1000, 64}, [](const auto&)
															 { return true; }, budget);
				verify(pixels(ctx, image), {192, 128, 64, 255});
			}
			else if (mode == "clear")
			{
				const auto target = surfaces.acquire({0x1000, {512, 512, 1, surface_color::argb8, false}, 2048, {}});
				for (unsigned i = 0; i < count; ++i)
				{
					surfaces.clear(target, 0xf0, 0xff112233 + i, 0, 255, {0, 0, 512, 512});
					const auto image = surfaces.snapshot_texture({0x1000, 1 << 20}, [](const auto&)
																 { return true; }, budget);
					verify(pixels(ctx, image), {51 + i, 34, 17, 255});
				}
			}
			else if (mode == "display")
			{
				display_resolver resolver(ctx, memory, [](unsigned offset, unsigned, std::size_t)
										  { return offset; }, budget, 1 << 20);
				native_frame frame(ctx);
				auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:4 mipmapped:NO];
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = MTLTextureUsageRenderTarget;
				const auto output = budget->texture(descriptor);
				for (unsigned i = 0; i < memory->bytes.size(); i += 4)
				{
					memory->bytes[i] = std::byte{0};
					memory->bytes[i + 1] = std::byte{0x11};
					memory->bytes[i + 2] = std::byte{0x22};
					memory->bytes[i + 3] = std::byte{0x33};
				}
				for (unsigned i = 0; i < count; ++i)
				{
					const auto image = resolver.resolve({0, 512, 512, 2048, display_format::xrgb8}, surfaces);
					ctx.begin();
					frame.encode(image, output, frame_policy{{512, 512}, 1, true});
					ctx.submit().wait();
				}
				verify(pixels(ctx, output), {51, 34, 17, 255});
				require(memory->reads == 1 && resolver.statistics().uploads == 1 && resolver.statistics().hits == count - 1,
					"Unchanged RAM display loop failed to reuse its confirmed active-byte version");
			}
			else
				throw std::invalid_argument("Unknown lifetime mode");
			ctx.wait_idle();
			std::cout << "Native entry lifetime:" << mode << "32 iterations without caller autorelease drains;reserved=" << budget->statistics().reserved_bytes << "/" << budget->statistics().budget << ";pixels/publication intact,no App/UI/guest VM initialized\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
