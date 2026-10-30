#include "Emu/RSX/Common/tiled_dma_copy.hpp"
#include "M4GuardedGuestMemory.h"
#include "M4ScaledImage.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <mutex>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
class tile_memory final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<tile_memory>
{
	std::mutex mutex;
	std::vector<std::byte> data = std::vector<std::byte>(1 << 20);

  public:
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& op) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			op(*current);
			return;
		}
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		op(*this);
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		return overlaps(a, b);
	}
	void written(guest_surface_range) noexcept override
	{
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		if (range.address > data.size() || range.size > data.size() - range.address)
			throw std::out_of_range("Tile fixture range");
		return data.data() + range.address;
	}
};
rsx::detiler_config config(const banked_tile_layout& tile, unsigned width, unsigned height, unsigned bpp, unsigned pitch)
{
	return {0, banked_tile_factor(tile.pitch), tile.pitch / 256, tile.address, tile.size, tile.offset, 0, tile.pitch, tile.bank, width, height, pitch, bpp};
}
void reference(const rsx::detiler_config& c, std::vector<std::byte>& tiled, std::vector<std::byte>& linear, bool decode)
{
	// Execute the unchanged shared RSX reference,never the Metal address helper.
	for (unsigned y = 0; y < c.image_height; ++y)
		for (unsigned x = 0; x < c.image_width; ++x)
			rsx::tiled_dma_copy(y, x, c, reinterpret_cast<char*>(tiled.data()), reinterpret_cast<char*>(linear.data()), decode ? 1 : 0);
}
void check_gpu(context& ctx, scaled_image_compute& compute, const scaled_image_plan& plan, const std::vector<std::byte>& input,
			   const std::vector<std::byte>& initial, const std::vector<std::byte>& expected)
{
	@autoreleasepool
	{
		const auto source = make_upload(ctx, input), target = make_upload(ctx, initial);
		ctx.begin();
		compute.encode(ctx, plan, source, target);
		ctx.submit().wait();
		const auto result = map_buffer(target, 0, expected.size());
		require(std::equal(result.begin(), result.end(), expected.begin()), "Banked Metal bytes differ from actual shared reference");
	}
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx{1};
			auto budget = std::make_shared<heap_budget>(ctx, 64 << 20);
			ctx.set_buffer_budget(budget);
			scaled_image_compute compute(ctx, 2 << 20);
			unsigned cases = 0;
			for (unsigned bpp : {2u, 4u})
				for (unsigned pitch : {256u, 512u, 768u, 1280u, 1536u, 1792u, 2816u, 3328u, 4096u, 6144u})
					for (unsigned bank = 0; bank < 4; ++bank)
					{
						const unsigned width = 256 / bpp - 3, height = 131, linear_pitch = (width + 3) * bpp;
						const banked_tile_layout tile{0x50000, 0x100000, pitch, bank, pitch * 2 + bpp};
						const auto view = attach_banked_tile(plan_guest_surface(tile.address + tile.offset, width, height, bpp, linear_pitch), tile);
						const auto linear = plan_guest_surface(0x1000, width, height, bpp, linear_pitch);
						const auto format = bpp == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8;
						std::vector<std::byte> input(linear.range.size), initial(tile.size, std::byte{0xa5});
						for (unsigned i = 0; i < input.size(); ++i)
							input[i] = std::byte(i * 37 + 71);
						auto expected = initial, unchanged = input;
						reference(config(tile, width, height, bpp, linear_pitch), expected, unchanged, false);
						scaled_image_plan encode{linear, view, format, format, width, height, width, height};
						encode.destination_tile = tile;
						check_gpu(ctx, compute, encode, input, initial, expected);
						++cases;
						std::vector<std::byte> output(linear.range.size, std::byte{0x93});
						auto decoded = output;
						for (unsigned i = 0; i < initial.size(); ++i)
							initial[i] = std::byte(i * 13 + 29);
						auto saved = initial;
						reference(config(tile, width, height, bpp, linear_pitch), saved, decoded, true);
						scaled_image_plan decode{view, linear, format, format, width, height, width, height};
						decode.source_tile = tile;
						check_gpu(ctx, compute, decode, initial, output, decoded);
						++cases;
					}
			for (unsigned bpp : {2u, 4u})
				for (unsigned bank = 0; bank < 4; ++bank)
				{
					const unsigned width = 61, height = 67;
					const banked_tile_layout tile{0x50000, 0x40000, 768, bank, bpp * 5};
					const swizzle_region morton{{128, 128, 1}, {3, 5, 0}, {width, height, 1}};
					const auto view = attach_banked_tile(plan_guest_surface(tile.address + tile.offset, width, height, bpp, 0, morton), tile);
					const auto plain = plan_guest_surface(0x1000, width, height, bpp);
					const auto format = bpp == 2 ? scaled_image_format::rgb565 : scaled_image_format::argb8;
					std::vector<std::byte> input(plain.range.size), initial(tile.size, std::byte{0x87}), logical(128 * 128 * bpp), decoded(plain.range.size);
					for (unsigned i = 0; i < input.size(); ++i)
						input[i] = std::byte(i * 17 + 31);
					auto expected = initial;
					const auto conf = config(tile, 256 / bpp, 128, bpp, tile.pitch);
					for (unsigned y = 0; y < height; ++y)
						for (unsigned x = 0; x < width; ++x)
						{
							std::size_t cell = 0;
							for (unsigned bit = 0; bit < 7; ++bit)
							{
								cell |= std::size_t(((x + 3) >> bit) & 1) << (2 * bit);
								cell |= std::size_t(((y + 5) >> bit) & 1) << (2 * bit + 1);
							}
							const auto byte = cell * bpp;
							std::copy_n(input.begin() + (y * width + x) * bpp, bpp, logical.begin() + byte);
							rsx::tiled_dma_copy(byte / tile.pitch, (byte % tile.pitch) / bpp, conf, reinterpret_cast<char*>(expected.data()), reinterpret_cast<char*>(logical.data()), 0);
						}
					scaled_image_plan encode{plain, view, format, format, width, height, width, height};
					encode.destination_tile = tile;
					check_gpu(ctx, compute, encode, input, initial, expected);
					++cases;
					scaled_image_plan decode{view, plain, format, format, width, height, width, height};
					decode.source_tile = tile;
					check_gpu(ctx, compute, decode, expected, decoded, input);
					++cases;
				}
			// The last partial physical tile may bank-permute outside its
			// allocation even though its logical rectangle is in bounds.
			const banked_tile_layout tail{0x50000, 0x10000, 768, 3, 768 * 80 + 256};
			const auto tail_view = attach_banked_tile(plan_guest_surface(tail.address + tail.offset, 8, 3, 2, 16), tail);
			const auto tail_linear = plan_guest_surface(0x1000, 8, 3, 2, 16), tail_argb = plan_guest_surface(0x1000, 8, 3, 4, 32);
			std::vector<std::byte> tail_bytes(tail.size, std::byte{0x71}), tail_zero(tail_linear.range.size), tail_output(tail_argb.range.size);
			auto tail_reference = tail_bytes;
			reference(config(tail, 8, 3, 2, 16), tail_reference, tail_zero, true);
			require(std::all_of(tail_zero.begin(), tail_zero.end(), [](auto value)
								{ return value == std::byte{0}; }),
					"Tail fixture unexpectedly maps physical storage");
			scaled_image_plan tail_plan{tail_view, tail_linear, scaled_image_format::rgb565, scaled_image_format::rgb565, 8, 3, 8, 3};
			tail_plan.source_tile = tail;
			check_gpu(ctx, compute, tail_plan, tail_bytes, tail_zero, tail_zero);
			tail_plan.destination = tail_argb;
			tail_plan.destination_format = scaled_image_format::argb8;
			for (unsigned i = 0; i < tail_output.size(); i += 4)
				tail_output[i] = std::byte{0xff};
			check_gpu(ctx, compute, tail_plan, tail_bytes, std::vector<std::byte>(tail_output.size()), tail_output);
			tail_plan = {tail_linear, tail_view, scaled_image_format::rgb565, scaled_image_format::rgb565, 8, 3, 8, 3};
			tail_plan.destination_tile = tail;
			check_gpu(ctx, compute, tail_plan, std::vector<std::byte>(tail_linear.range.size, std::byte{0x53}), tail_bytes, tail_bytes);
			// Bank-to-bank flipped copy using two independently configured layouts.
			const unsigned w = 61, h = 67, bpp = 4;
			const banked_tile_layout a{0x10000, 0x40000, 768, 3, 768 * 3 + 4}, b{0x50000, 0x40000, 1536, 1, 1536 * 2 + 12};
			const auto source = attach_banked_tile(plan_guest_surface(a.address + a.offset, w, h, bpp, w * bpp), a);
			const auto target = attach_banked_tile(plan_guest_surface(b.address + b.offset, w, h, bpp, w * bpp), b);
			scaled_image_plan plan{source, target, scaled_image_format::argb8, scaled_image_format::argb8, w, h, w, h, 0, 0, true, true};
			plan.source_tile = a;
			plan.destination_tile = b;
			std::vector<std::byte> input(a.size), output(b.size, std::byte{0x83}), linear(w * h * bpp), flipped(linear.size());
			for (unsigned i = 0; i < input.size(); ++i)
				input[i] = std::byte(i * 71 + 31);
			auto saved = input;
			reference(config(a, w, h, bpp, w * bpp), saved, linear, true);
			for (unsigned y = 0; y < h; ++y)
				for (unsigned x = 0; x < w; ++x)
					std::copy_n(linear.begin() + ((h - 1 - y) * w + w - 1 - x) * bpp, bpp, flipped.begin() + (y * w + x) * bpp);
			auto expected = output;
			reference(config(b, w, h, bpp, w * bpp), expected, flipped, false);
			check_gpu(ctx, compute, plan, input, output, expected);
			auto provider = std::make_shared<tile_memory>();
			auto memory = std::make_shared<guarded_guest_memory>(provider);
			surface_cache surfaces(ctx, memory, 64 << 20, budget);
			guest_texture_cache textures(ctx, memory, [](unsigned address, unsigned, std::size_t)
										 { return address; }, 1 << 20, budget);
			unsigned captures = 0;
			scaled_image_transfer transfer(ctx, surfaces, textures, memory, [&](const auto& op)
										   {++captures;provider->execute([&](auto&){op();}); }, 1 << 20);
			memory->write(source.range, input);
			memory->write(target.range, output);
			transfer.execute(plan);
			require(memory->read(target.range).bytes == expected && captures == 2, "Banked owned capture/CAS/whole-allocation preservation failed");
			// A write elsewhere in the allocation must refuse a pending whole-tile
			// publication,even when it does not intersect one of the copied pixels.
			memory->write(target.range, output);
			auto gate = [ctx.device() newSharedEvent];
			bool timeout = false;
			try
			{
				transfer.execute(plan, std::chrono::milliseconds(1), [&](context& active)
								 { active.wait_event(gate, 1); });
			}
			catch (const std::exception&)
			{
				timeout = true;
			}
			require(timeout && transfer.has_pending(), "Banked pending gate failed");
			const std::array<std::byte, 1> writer{std::byte{0x45}};
			memory->write({b.address + b.size - 1, 1}, writer);
			gate.signaledValue = 1;
			bool conflict = false;
			try
			{
				transfer.synchronize();
			}
			catch (const texture_revision_conflict&)
			{
				conflict = true;
			}
			output.back() = writer[0];
			require(conflict && !transfer.has_pending() && memory->read(target.range).bytes == output, "Banked CAS overwrote an external writer");
			expected.back() = writer[0];
			transfer.execute(plan);
			require(memory->read(target.range).bytes == expected, "Banked retry failed to preserve whole allocation");
			for (unsigned i = 0; i < 160; ++i)
				transfer.execute(plan);
			ctx.wait_idle();
			require(memory->read(target.range).bytes == expected, "Repeated banked transfers changed output");
			std::cout << "Native banked transfer repeated160 times without caller autorelease drains;reserved=" << budget->statistics().reserved_bytes << "/" << budget->statistics().budget << "\n";
			require(cases == 176, "Missing banked GPU configurations");
			std::cout << "Native banked-image:176 GPU encode/decode cases against actual unchanged shared RSX tile reference;10 pitches/all4 banks/RGB565/ARGB8/64row crossings/partial rows/all allocation bytes/Morton composition;out-of-allocation zero/opaque RGB565 decode and skipped writes;bank-to-bank flip,owned two epochs,whole-tile CAS refusal/retry passed;no guest VM initialized\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
