#include "Emu/RSX/RSXOffload.h"
#include "M4ClauseRendererFixtures.h"
#include "M4CompilerGateFixture.h"
#include "M4ConstantUpload.h"
#include "M4DrawEnvironmentBinding.h"
#include "M4DrawEnvironmentFixtures.h"
#include "M4DrawExecutor.h"
#include "M4DrawExecutorFixtures.h"
#include "M4DrawGeometryPlan.h"
#include "M4DrawPacketFixtures.h"
#include "M4FaultRange.h"
#include "M4FramebufferBinding.h"
#include "M4FramebufferFixtures.h"
#include "M4GuardedGuestMemory.h"
#include "M4GuestLabel.h"
#include "M4GuestMemoryAliases.h"
#include "M4GuestReport.h"
#include "M4HostAccessFixture.h"
#include "M4RenderTasks.h"
#include "M4ReportAtomic.h"
#include "M4SampleTransfer.h"
#include "M4ShaderSnapshotFixtures.h"
#include "M4VertexDataBinding.h"
#include "M4VertexDataFixtures.h"
#include "M4VertexTextureBatch.h"

#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
#include <atomic>

namespace
{
using namespace metal4;
void require(bool v, const char* message)
{
	if (!v)
		throw std::runtime_error(message);
}
class guarded_gpu_memory final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<guarded_gpu_memory>
{
	std::mutex m_mutex;
	std::vector<std::byte> m_bytes = std::vector<std::byte>(1 << 20);

  public:
	bool reject_alias_query = false;
	bool remapped = false;
	bool provide_storage_identity = false;
	guest_surface_range physical(guest_surface_range range) const
	{
		const auto address = range.address % m_bytes.size();
		return {static_cast<unsigned>(address + (remapped && range.address < 65536 ? 0x80000 : 0)), range.size};
	}
	std::byte* map(guest_surface_range range, bool) override
	{
		validate_guest_surface_range(range);
		const auto physical = range.address % m_bytes.size();
		if (range.address >= 2 * m_bytes.size() || range.size > m_bytes.size() - physical)
			throw std::out_of_range("Guarded GPU fixture memory range is invalid");
		return m_bytes.data() + this->physical(range).address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override
	{
		if (reject_alias_query)
			throw std::runtime_error("Fixture alias query failure");
		return overlaps(physical(a), physical(b));
	}
	std::optional<guest_surface_range> rebase_storage(guest_surface_range source, guest_surface_range request) override
	{
		return rebase_guest_memory_range(source, request, [&](unsigned block)
										 { return std::uint64_t(physical({block, 1}).address); });
	}
	bool equivalent_storage(guest_surface_range a, guest_surface_range b) override
	{
		return equivalent_guest_memory_ranges(a, b, [&](unsigned block)
											  { return std::uint64_t(physical({block, 1}).address); });
	}
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		if (!provide_storage_identity || remapped || range.address >= m_bytes.size() || range.size > m_bytes.size() - range.address)
			return {};
		return std::vector<storage_identity_span>{{shared_from_this(), range.address, range}};
	}
	unsigned notifications = 0;
	void written(guest_surface_range) noexcept override { ++notifications; }
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* memory = guest_memory_epoch::current(*this))
		{
			operation(*memory);
			return;
		}
		std::lock_guard lock(m_mutex);
		guest_memory_epoch epoch(*this, *this);
		operation(*this);
	}
};
class memory final : public guest_surface_memory
{
	std::mutex mutex;
	std::vector<std::byte> data = std::vector<std::byte>(1 << 20);
	std::vector<std::uint64_t> versions = std::vector<std::uint64_t>(data.size());
	std::uint64_t counter = 0;
	std::uint64_t version(const guest_surface_layout& coverage)
	{
		if (coverage.range.address > data.size() || coverage.range.size > data.size() - coverage.range.address)
			throw std::out_of_range("Executor fixture VM range is invalid");
		std::uint64_t result = 0;
		guest_surface_byte_cursor cursor(coverage);
		while (const auto range = cursor.next())
			for (std::size_t i = 0; i < range->size; ++i)
				result = std::max(result, versions[range->address + i]);
		return result;
	}

  public:
	unsigned reads = 0;
	guest_surface_snapshot read_surface(const guest_surface_layout& coverage) override
	{
		std::lock_guard lock(mutex);
		++reads;
		const auto revision = version(coverage);
		return {revision, {data.begin() + coverage.range.address, data.begin() + coverage.range.address + coverage.range.size}};
	}
	std::uint64_t surface_revision(const guest_surface_layout& coverage) override
	{
		std::lock_guard lock(mutex);
		return version(coverage);
	}
	std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage, std::uint64_t expected, std::span<const std::byte> bytes) override
	{
		std::lock_guard lock(mutex);
		if (version(coverage) != expected)
			return {};
		require(bytes.size() == coverage.range.size, "Executor writeback length mismatch");
		++counter;
		guest_surface_byte_cursor cursor(coverage);
		while (const auto range = cursor.next())
			for (std::size_t i = 0; i < range->size; ++i)
			{
				data[range->address + i] = bytes[range->address - coverage.range.address + i];
				versions[range->address + i] = counter;
			}
		return counter;
	}
	void write(guest_surface_range range, std::span<const std::byte> bytes) override
	{
		std::lock_guard lock(mutex);
		(void)version(guest_byte_coverage(range));
		require(bytes.size() == range.size, "Executor CPU write length mismatch");
		++counter;
		std::copy(bytes.begin(), bytes.end(), data.begin() + range.address);
		std::fill_n(versions.begin() + range.address, range.size, counter);
	}
};
using namespace metal4::test::executor_fixture;
void verify(context& ctx, surface_cache& surfaces, const cached_surface_ref& target, const std::array<unsigned, 4>& rgba,
			std::array<unsigned, 4> region = {0, 0, 4, 4}, std::array<unsigned, 4> outside = {17, 34, 51, 255})
{
	const auto output = make_buffer(ctx, 1024, buffer_storage::shared);
	const std::array uses{cached_surface_use{target, false}};
	surfaces.draw(uses, [&](context& commands, auto textures)
				  {texture_region whole;whole.size=MTLSizeMake(4,4,1);texture_to_buffer(commands,textures[0],whole,output,0,256); });
	for (unsigned y = 0; y < 4; ++y)
		for (unsigned x = 0; x < 4; ++x)
			for (unsigned c = 0; c < 4; ++c)
			{
				const auto expected = x >= region[0] && y >= region[1] && x - region[0] < region[2] && y - region[1] < region[3] ? rgba[c] : outside[c];
				require(static_cast<const unsigned char*>(output.contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] == expected, "Native draw executor color/scissor output differs");
			}
}
void verify_executor_bounds(context& ctx, surface_cache& surfaces, draw_executor& executor, const draw_request& base)
{
	query_pool queries(ctx, 2, 16);
	unsigned cases = 0;
	unsigned native_draws = 0, mirror_draws = 0, disabled_draws = 0;
	unsigned independent_mirrors = 0, dependent_mirrors = 0;
	const bool native_supported = supports_native_depth_bounds(ctx.device());
	for (const auto format : {surface_depth::z16_uint, surface_depth::z16_float, surface_depth::z24s8_uint, surface_depth::z24s8_float})
	{
		const auto native = describe_surface_depth(format);
		const bool stencil = native.format == MTLPixelFormatDepth32Float_Stencil8;
		const bool unorm16 = native.format == MTLPixelFormatDepth16Unorm;
		const auto target = surfaces.acquire({0x51000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
		const auto depth = surfaces.acquire({0x50000, {4, 4, 1, format, false}, unorm16 || format == surface_depth::z16_float ? 8u : 16u, {}});
		draw_targets targets;
		targets.colors[0] = target;
		targets.depth = depth;
		const auto z = make_buffer(ctx, 1024, buffer_storage::shared), s = make_buffer(ctx, 1024, buffer_storage::shared);
		const auto read = [&]
		{
			const std::array uses{cached_surface_use{depth, false}};
			surfaces.draw(uses, [&](context& commands, auto textures)
			{
				texture_region whole;
				whole.size = MTLSizeMake(4, 4, 1);
				texture_to_buffer(commands, textures[0], whole, z, 0, 256, 0, texture_plane::depth);
				if (stencil) texture_to_buffer(commands, textures[0], whole, s, 0, 256, 0, texture_plane::stencil);
			});
		};
		const auto stored = [&](unsigned x, unsigned y)
		{
			if (!unorm16) return static_cast<const float*>(z.contents)[y * 64 + x];
			std::uint16_t value;
			std::memcpy(&value, static_cast<const std::byte*>(z.contents) + y * 256 + x * 2, 2);
			return value / 65535.f;
		};
		for (unsigned kind = 0; kind < 20; ++kind)
		{
			if (kind >= 12 && unorm16) continue;
			surfaces.clear(target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
			surfaces.clear(depth, stencil ? 3 : 1, 0, stencil ? 0x40000042 : 0x4000, 255, {0, 0, 4, 4});
			if (kind >= 12)
			{
				const std::array invalid_depth{0x7fc12345u,0x7f800000u,0xbf000000u,0x3fc00000u};
				for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x)
					std::memcpy(static_cast<std::byte*>(z.contents) + y * 256 + x * 4,&invalid_depth[(kind - 12) % 4],4);
				const std::array writes{cached_surface_use{depth,true}};
				surfaces.draw(writes,[&](context& commands,auto textures)
				{
					texture_region whole; whole.size = MTLSizeMake(4,4,1);
					buffer_to_texture(commands,z,0,textures[0],whole,256,0,texture_plane::depth);
				});
			}
			read();
			const auto initial = stored(0, 0);
			if (kind >= 12)
				require(std::bit_cast<std::uint32_t>(initial) == std::array{0x7fc12345u,0x7f800000u,0xbf000000u,0x3fc00000u}[(kind - 12) % 4],
					"Depth bounds fixture lost the independently specified raw stored float bits");
			auto words = registers();
			auto& m = words.registers;
			m[0xa74 / 4] = kind != 2 && kind < 12;
			m[0xa70 / 4] = kind != 1 && kind < 12;
			m[0xa6c / 4] = kind == 3 ? 0x201 : kind == 4 && !stencil ? 0x200 : 0x207;
			m[0x328 / 4] = stencil;
			m[0x330 / 4] = kind == 4 || kind == 6 ? 0x200 : 0x207;
			m[0x33c / 4] = m[0x340 / 4] = 0x1e02;
			m[0x344 / 4] = kind == 5 ? 0x1e01 : 0x1e02;
			m[0x334 / 4] = 0xab;
			m[0x32c / 4] = kind == 5 ? 0xf0 : 255;
			if (kind == 9) m[0x8c0 / 4] = m[0x8c4 / 4] = (2u << 16) | 1;
			auto vertex = vertex_program();
			auto bytes = program(false);
			test::fragment_program fragment;
			fragment.pointer = bytes.data();
			fragment.ucode_length = bytes.size();
			fragment.ctrl = rsx_fragment_float32 | (kind == 7 ? 0x400000u : 0u);
			const auto draw = rsx_draw_snapshot::capture(words, vertex, fragment, 0, 1 << 20);
			auto request = base;
			const auto triangle = request.geometry.vertices;
			request.geometry.vertices.insert(request.geometry.vertices.end(), triangle.begin(), triangle.end());
			request.queries = &queries;
			request.raster.depth_bounds_enabled = kind != 11;
			request.raster.depth_min = kind >= 10 && kind < 16 ? 0 : kind == 6 ? .6f : kind == 9 ? initial : .2f;
			request.raster.depth_max = kind >= 10 && kind < 16 ? 1 : kind == 6 ? .7f : kind == 9 ? initial : .3f;
			request.fragment_uniforms.alpha_function = 0;
			request.fragment_uniforms.sample_mask = kind == 8 ? 0 : 0xffffffffu;
			const bool color_pass = kind == 0 || kind == 1 || kind == 2 || kind == 5 || kind == 9 || kind == 10 || kind == 11;
			const bool writes_depth = kind == 0 || kind == 5 || kind == 9 || kind == 10 || kind == 11;
			for (unsigned repeat = 0; repeat < 2; ++repeat)
			{
				queries.begin(repeat, true);
				const auto before = ctx.encoding_statistics().render_encoders;
				executor.execute(draw, targets, request);
				const bool mirror = kind != 11 && (!native_supported || (kind >= 10 && kind < 16));
				require(ctx.encoding_statistics().render_encoders == before + (mirror ? 2 : 1),
					"Depth bounds did not select exactly one native draw or initializer-plus-mirror draw");
				if (mirror) ++mirror_draws; else if (kind != 11) ++native_draws; else ++disabled_draws;
				queries.end(repeat);
				const unsigned expected_count = kind == 1 || kind == 2 || kind == 10 || kind == 11 ? 32 : !color_pass || repeat ? 0 : kind == 9 ? 4 : 16;
				const auto count = queries.consume(repeat);
				if (count != expected_count)
					throw std::runtime_error("Executor depth-bounds query mismatch: format=" + std::to_string(static_cast<unsigned>(format)) +
						" kind=" + std::to_string(kind) + " repeat=" + std::to_string(repeat) + " actual=" + std::to_string(count));
				read();
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
					{
						const bool inside = kind != 9 || (x >= 1 && x < 3 && y >= 1 && y < 3);
						const float written = unorm16 ? 32768.f / 65535.f : .5f;
						require(std::bit_cast<std::uint32_t>(stored(x, y)) == std::bit_cast<std::uint32_t>(writes_depth && inside ? written : initial), "Executor stored-depth bounds changed native depth incorrectly");
						if (stencil)
						{
							const unsigned expected_stencil = !inside || kind == 6 || kind == 7 || kind == 8 || kind >= 12 ? 0x42 : kind == 5 ? 0xa2 :
								kind == 0 || kind == 9 ? 0x43 : 0x42 + 2 * (repeat + 1);
							require(static_cast<const unsigned char*>(s.contents)[y * 256 + x] == expected_stencil, "Executor depth-bounds late stencil operation differs");
						}
					}
			}
			verify(ctx, surfaces, target, color_pass ? std::array{64u, 128u, 192u, 255u} : std::array{17u, 34u, 51u, 255u},
				kind == 9 ? std::array{1u, 1u, 2u, 2u} : std::array{0u, 0u, 4u, 4u});
			if (kind == 10)
			{
				request.queries = nullptr;
				auto dependent = request;
				dependent.persistent = make_buffer(ctx, request.persistent.length, buffer_storage::private_gpu);
				for (const bool blit_vertex : {false, true})
				{
					{
						auto batch = surfaces.batch_draws();
						surfaces.clear(target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
						if (blit_vertex)
							copy_buffer(ctx, request.persistent, 0, dependent.persistent, 0, request.persistent.length);
						const auto before = ctx.encoding_statistics();
						executor.execute(draw, targets, blit_vertex ? dependent : request);
						const auto after = ctx.encoding_statistics();
						require(after.render_encoders == before.render_encoders + 2 &&
							after.guest_draw_full_barriers == before.guest_draw_full_barriers + blit_vertex &&
							after.guest_draw_fragment_barriers == before.guest_draw_fragment_barriers + !blit_vertex,
							"Stored-depth guest draw must wait for the mirror and any dependent vertex input");
						independent_mirrors += !blit_vertex;
						dependent_mirrors += blit_vertex;
						batch.finish();
					}
					verify(ctx, surfaces, target, {64, 128, 192, 255});
				}
			}
			++cases;
		}
	}
	require(cases == 72 && native_draws == (native_supported ? 104u : 0u) && mirror_draws == (native_supported ? 32u : 136u) && disabled_draws == 8,
		"Depth bounds matrix omitted native,mirror,disabled or full-range nonfinite storage cases");
	std::cout << "Native executor stored-depth bounds:" << cases << " guest-depth cases,two overlapping triangles,no-write/disabled depth,late depth/stencil failure,masked replacement,alpha/sample-mask rejection,equal bounds/scissor;narrow/full-range stored NaN/Inf/negative/above-one rejected with exact depth bits/stencil,queries exclude initializer;native-selector draws=" << native_draws << ",mirror draws=" << mirror_draws << ",disabled draws=" << disabled_draws << ";exact encoder selection passed\n";
	std::cout << "Checked stored-depth guest draws: " << independent_mirrors << " fragment/tile waits and "
		<< dependent_mirrors << " full barriers for vertex blits; four depth formats and exact pixels passed\n";
}
std::size_t sample_storage_offset(const framebuffer_surface& surface, unsigned x, unsigned y)
{
	if (!surface.morton) return y * surface.row_pitch + x * surface.guest.pixel_bytes;
	// Independent high-to-low coordinate oracle,including rectangular storage.
	std::size_t width = surface.morton->storage[0], height = surface.morton->storage[1], index = 0;
	while (width > 1 || height > 1)
	{
		if (width > height)
		{
			width /= 2;
			index = index * 2 + bool(x & width);
		}
		else
		{
			height /= 2;
			index = index * 2 + bool(y & height);
		}
	}
	return index * surface.guest.pixel_bytes;
}
rsx_draw_packet sample_packet(unsigned kind, unsigned mask, bool hint, const std::vector<std::byte>& positions,
	const std::vector<std::byte>& indices, const std::array<vertex_attribute_descriptor, 16>& descriptors, bool stencil_enabled = true)
{
	test::pass_state state;
	state.registers = registers().registers;
	state.registers[0xa74 / 4] = state.registers[0xa70 / 4] = 1;
	state.registers[0xa6c / 4] = 0x207;
	state.registers[0x328 / 4] = stencil_enabled;
	state.registers[0x344 / 4] = 0x1e02;
	if (kind == 10) state.registers[0x8c0 / 4] = state.registers[0x8c4 / 4] = (2u << 16) | 1;
	state.multisample = hint;
	state.samples = mask;
	state.alpha_coverage = kind == 4 || kind == 5 || kind == 6 || kind == 8 || kind == 11 || kind == 12 || kind == 13 || kind == 14;
	state.alpha_one = kind == 7 || kind == 8 || kind == 13 || kind == 15;
	state.bounds = kind == 9 || kind == 14;
	state.bounds_min = .2f;
	state.bounds_max = .3f;
	state.alpha = kind == 11 ? 0x200 : 0x207;
	test::pass_processor processor;
	processor.descriptors = descriptors;
	processor.persistent = positions;
	processor.index_bytes = indices;
	test::pass_layout layout;
	test::pass_metadata metadata;
	auto vertex = vertex_program();
	const float alpha = kind == 4 ? 0.f : kind == 5 || kind == 7 || kind == 8 || kind >= 12 ? .5f : 1.f;
	auto bytes = program(false, {64.f / 255, 128.f / 255, 192.f / 255, alpha});
	test::fragment_program fragment;
	fragment.pointer = bytes.data();
	fragment.ucode_length = bytes.size();
	fragment.ctrl = rsx_fragment_float32 | (kind == 11 ? 0x400000u : 0u);
	const auto packet = rsx_draw_packet::capture(state, processor, layout, metadata, vertex, fragment, 0, provoking_vertex::last,
		[](const auto&, unsigned first, unsigned count)
		{
			require(first == 0 && count == 9, "Sample packet changed sparse indexed input prefix");
			return std::pair<std::uint32_t, std::uint32_t>{48, 0};
		});
	state.samples = ~mask;
	state.multisample = !hint;
	state.alpha_coverage = !state.alpha_coverage;
	state.alpha_one = !state.alpha_one;
	state.bounds = !state.bounds;
	state.registers.assign(state.registers.size(), 0xffffffffu);
	processor.persistent.assign(48, std::byte{0});
	processor.index_bytes.assign(12, std::byte{0});
	vertex.data.assign(vertex.data.size(), 0xffffffffu);
	bytes.assign(bytes.size(), std::byte{0xff});
	return packet;
}
void verify_framebuffer_samples(context& ctx, surface_cache& surfaces, draw_executor& executor, memory& ram,
	const std::vector<std::byte>& positions, const std::vector<std::byte>& indices,
	const std::array<vertex_attribute_descriptor, 16>& descriptors)
{
	query_pool queries(ctx, 2, 16);
	sample_transfer transfer(ctx);
	unsigned cases = 0;
	unsigned checked_msaa = 0, checked_bounds = 0, checked_copy = 0;
	for (const auto format : {surface_depth::z16_uint, surface_depth::z16_float, surface_depth::z24s8_uint, surface_depth::z24s8_float})
	{
		const auto info = describe_surface_depth(format);
		// Independent single-sample codec baseline for the exact serialized
		// initial/written guest depth words. Sample placement/padding is separate.
		const auto reference = surfaces.acquire({0x6c000, {4, 4, 1, format, false}, info.guest_bytes * 4, {}});
		draw_targets reference_targets;
		reference_targets.depth = reference;
		const auto read_reference = [&]
		{
			surfaces.flush(reference);
			return ram.read_surface(plan_guest_surface(0x6c000, 4, 4, info.guest_bytes)).bytes;
		};
		surfaces.clear(reference, info.stencil ? 3 : 1, 0, info.stencil ? 0x40000042 : 0x4000, 255, {0, 0, 4, 4});
		const auto initial_word = read_reference();
		executor.execute(sample_packet(0, 0xffffffffu, true, positions, indices, descriptors, info.stencil), reference_targets);
		const auto written_word = read_reference();
		for (unsigned mode : {0u, 3u, 4u, 5u})
			for (unsigned raster : {1u, 2u})
			{
				const unsigned count = mode == 0 ? 1 : mode == 3 ? 2 : 4, sx = count == 1 ? 1 : 2, sy = count == 4 ? 2 : 1;
				require([ctx.device() supportsTextureSampleCount:count], "Required guest framebuffer sample count is unavailable");
				test::framebuffer_layout layout;
				layout.target = 2;
				layout.aa_mode = mode;
				layout.aa_factors = {sx, sy};
				layout.raster_type = raster;
				layout.color_addresses = {0, 0x60000, 0, 0};
				layout.actual_color_pitch[1] = 4 * sx * 4 + 16;
				layout.zeta_address = 0x64000;
				layout.depth_format = static_cast<unsigned>(format);
				layout.actual_zeta_pitch = 4 * sx * info.guest_bytes + 16;
				const auto framebuffer = framebuffer_snapshot::capture(layout, true, raster == 2 ? std::optional{std::array<std::size_t, 3>{16, 16, 1}} : std::nullopt);
				layout.aa_mode = 0;
				layout.aa_factors = {1, 1};
				layout.color_addresses.fill(0);
				layout.zeta_address = 0;
				const auto targets = acquire_framebuffer(framebuffer, surfaces);
				require(targets.colors[0] && targets.depth && !targets.colors[1] && surfaces.description(targets.depth).allocation.samples == count &&
					surfaces.description(targets.depth).allocation.expanded == (count > 1), "MSAA framebuffer lost B/depth/sample-expansion metadata");
				auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:info.format width:4 * sx height:4 * sy mipmapped:NO];
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
				const auto expanded_depth = make_texture(ctx, descriptor);
				descriptor.pixelFormat = MTLPixelFormatBGRA8Unorm;
				const auto expanded_color = make_texture(ctx, descriptor);
				const auto color = make_buffer(ctx, 2048, buffer_storage::shared), depth = make_buffer(ctx, 2048, buffer_storage::shared), stencil = make_buffer(ctx, 2048, buffer_storage::shared);
				std::vector<bool> half_coverage;
				for (unsigned kind = 0; kind < 16; ++kind)
				{
					const auto& cs = *framebuffer.colors()[0];
					const auto& ds = *framebuffer.depth();
					const std::vector<std::byte> color_padding(cs.guest.range.size, std::byte{0xa7}), depth_padding(ds.guest.range.size, std::byte{0xb9});
					ram.write(cs.guest.range, color_padding);
					ram.write(ds.guest.range, depth_padding);
					surfaces.clear(targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
					surfaces.clear(targets.depth, info.stencil ? 3 : 1, 0, info.stencil ? 0x40000042 : 0x4000, 255, {0, 0, 4, 4});
					const unsigned mask = kind == 1 || kind == 15 ? 0 : kind == 2 || kind == 3 || kind == 9 || kind == 10 || (kind >= 12 && kind <= 14) ? 5 : 0xffff;
					const unsigned effective_mask = count == 1 ? 0xffff : mask;
					const auto packet = sample_packet(kind, mask, kind != 3, positions, indices, descriptors, info.stencil);
					draw_execution_state execution;
					execution.queries = &queries;
					queries.begin(0, true);
					executor.execute(packet, framebuffer, execution);
					queries.end(0);
					const auto query = queries.consume(0);
					const std::array uses{cached_surface_use{targets.colors[0], false}, cached_surface_use{targets.depth, false}};
					surfaces.draw(uses, [&](context& commands, auto textures)
					{
						if (count > 1)
						{
							transfer.expand(commands, textures[0], expanded_color);
							transfer.expand(commands, textures[1], expanded_depth);
						}
						texture_region whole;
						whole.size = MTLSizeMake(4 * sx, 4 * sy, 1);
						texture_to_buffer(commands, count == 1 ? textures[0] : expanded_color, whole, color, 0, 256);
						texture_to_buffer(commands, count == 1 ? textures[1] : expanded_depth, whole, depth, 0, 256, 0, texture_plane::depth);
						if (info.stencil) texture_to_buffer(commands, count == 1 ? textures[1] : expanded_depth, whole, stencil, 0, 256, 0, texture_plane::stencil);
					});
					std::vector<bool> coverage;
					auto expected_color = color_padding, expected_depth = depth_padding;
					unsigned visible = 0, eligible = 0;
					for (unsigned y = 0; y < 4 * sy; ++y)
						for (unsigned x = 0; x < 4 * sx; ++x)
						{
							const unsigned sample = x % sx + sx * (y % sy), px = x / sx, py = y / sy;
							const bool inside = kind != 10 || (px >= 1 && px < 3 && py >= 1 && py < 3);
							const bool allowed = inside && (effective_mask & (1u << sample));
							float z;
							if (info.depth_bytes == 2)
							{
								std::uint16_t word;
								std::memcpy(&word, static_cast<const std::byte*>(depth.contents) + y * 256 + x * 2, 2);
								z = word / 65535.f;
							}
							else z = static_cast<const float*>(depth.contents)[y * 64 + x];
							const bool changed = z > .4f;
							const bool partial = kind == 5 || kind == 8 || (kind >= 12 && kind <= 14);
							const bool expected = allowed && kind != 4 && kind != 11;
							require(partial ? (!changed || expected) : changed == expected, "Owned framebuffer sample coverage/mask/alpha/scissor differs");
							require(changed ? z == (info.depth_bytes == 2 ? 32768.f / 65535.f : .5f) : z > .24f && z < .26f, "MSAA framebuffer corrupted stored native depth");
							coverage.push_back(changed);
							visible += changed;
							eligible += allowed;
							const unsigned alpha = kind == 7 || kind == 8 || kind == 13 || kind == 15 ? 255 : kind == 5 || kind == 12 || kind == 14 ? 128 : 255;
							const std::array<unsigned, 4> rgba = changed ? std::array{64u, 128u, 192u, alpha} : std::array{17u, 34u, 51u, 255u};
							for (unsigned c = 0; c < 4; ++c)
								require(static_cast<const unsigned char*>(color.contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] == rgba[c], "MSAA framebuffer color/A2One output differs");
							if (info.stencil) require(static_cast<const unsigned char*>(stencil.contents)[y * 256 + x] == (changed ? 0x43 : 0x42), "MSAA stencil changed outside surviving samples");
							const auto co = sample_storage_offset(cs, x, y), zo = sample_storage_offset(ds, x, y);
							for (unsigned c = 0; c < 4; ++c) expected_color[co + c] = std::byte(rgba[std::array{3u, 0u, 1u, 2u}[c]]);
							std::copy_n((changed ? written_word : initial_word).begin(), info.guest_bytes, expected_depth.begin() + zo);
						}
					if (kind == 5) half_coverage = coverage;
					if (kind == 8) require(coverage == half_coverage, "Alpha-to-one changed coverage derived from original fragment alpha");
					if (kind >= 12 && kind <= 14)
						for (std::size_t i = 0; i < coverage.size(); ++i)
						{
							const unsigned x = i % (4 * sx), y = i / (4 * sx), sample = x % sx + sx * (y % sy);
							require(coverage[i] == (half_coverage[i] && bool(effective_mask & (1u << sample))), "A2C/mask/A2One/depth-bounds intersection differs");
						}
					if (count > 1 && (kind == 5 || kind == 8)) require(visible && visible < eligible, "Native half-alpha A2C did not create partial sample coverage");
					require(query == visible, "MSAA visibility counted different samples or depth initializer");
					if (kind == 9 || kind == 14)
					{
						queries.begin(1, true);
						executor.execute(packet, framebuffer, execution);
						queries.end(1);
						require(queries.consume(1) == 0, "Fresh per-sample stored-depth mirror accepted prior written samples");
					}
					surfaces.flush(targets.colors[0]);
					surfaces.flush(targets.depth);
					const auto actual_color = ram.read_surface(cs.guest).bytes, actual_depth = ram.read_surface(ds.guest).bytes;
					for (const auto& pair : {std::pair{&actual_color, &expected_color}, std::pair{&actual_depth, &expected_depth}})
						for (std::size_t i = 0; i < pair.first->size(); ++i)
							if ((*pair.first)[i] != (*pair.second)[i])
								throw std::runtime_error("MSAA guest byte mismatch:format=" + std::to_string(static_cast<unsigned>(format)) +
									" mode=" + std::to_string(mode) + " raster=" + std::to_string(raster) + " kind=" + std::to_string(kind) +
									" depth=" + std::to_string(pair.first == &actual_depth) + " byte=" + std::to_string(i) +
									" actual=" + std::to_string(std::to_integer<unsigned>((*pair.first)[i])) + " expected=" + std::to_string(std::to_integer<unsigned>((*pair.second)[i])));
					if (count > 1 && (kind == 0 || kind == 9))
					{
						for (const bool copy_samples : {false, true})
						{
							if (copy_samples && kind != 9) continue;
							auto batch = surfaces.batch_draws();
							surfaces.clear(targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
							surfaces.clear(targets.depth, info.stencil ? 3 : 1, 0, info.stencil ? 0x40000042 : 0x4000, 255, {0, 0, 4, 4});
							if (copy_samples)
							{
								auto copy_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:4 mipmapped:NO];
								copy_desc.textureType = MTLTextureType2DMultisample;
								copy_desc.sampleCount = count;
								copy_desc.storageMode = MTLStorageModePrivate;
								copy_desc.usage = MTLTextureUsageRenderTarget;
								const auto copy = make_texture(ctx, copy_desc);
								surfaces.draw(std::array{cached_surface_use{targets.colors[0], true}}, [&](context& commands, auto images)
								{
									const texture_region whole{0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)};
									copy_texture(commands, images[0], whole, copy, whole);
									copy_texture(commands, copy, whole, images[0], whole);
								});
							}
							const auto prefix = ctx.recording_submission();
							const auto before = ctx.encoding_statistics();
							executor.execute(packet, framebuffer);
							const auto after = ctx.encoding_statistics();
							require(prefix.same_batch(ctx.recording_submission()) &&
								after.guest_draw_fragment_barriers == before.guest_draw_fragment_barriers + !copy_samples &&
								after.guest_draw_full_barriers == before.guest_draw_full_barriers + copy_samples,
								"MSAA guest draw selected the wrong barrier after attachment or native copy work");
							batch.finish();
							surfaces.flush(targets.colors[0]);
							surfaces.flush(targets.depth);
							require(ram.read_surface(cs.guest).bytes == expected_color && ram.read_surface(ds.guest).bytes == expected_depth,
								"Checked MSAA guest draw changed samples, stencil or guest padding");
							checked_msaa += kind == 0;
							checked_bounds += kind == 9 && !copy_samples;
							checked_copy += copy_samples;
						}
					}
					++cases;
				}
			}
	}
	require(cases == 512, "MSAA framebuffer policy matrix omitted guest formats/modes/storage/state");
	std::cout << "Native owned MSAA framebuffers:512 GPU policy cases,four guest depth formats,AA0/3/4/5,linear/Morton B-plus-depth,sample-mask with disabled hint,A2C endpoints/half coverage,A2One ordering,alpha rejection,scissor,stored-depth bounds and query counts;individual native color/depth/stencil samples,exploded guest words and whole-allocation padding passed\n";
	std::cout << "Checked MSAA guest draws: " << checked_msaa << " ordinary and " << checked_bounds
		<< " depth-bounds fragment/tile barriers after tracked clears, " << checked_copy
		<< " full barriers after native copies; exact color, depth, stencil and padding passed\n";
}
void verify_framebuffer_sample_colors(context& ctx, surface_cache& surfaces, draw_executor& executor, memory& ram,
	const std::vector<std::byte>& positions, const std::vector<std::byte>& indices,
	const std::array<vertex_attribute_descriptor, 16>& descriptors)
{
	query_pool queries(ctx, 1, 16);
	unsigned cases = 0;
	for (const auto format : {surface_color::x1rgb5_zero, surface_color::x1rgb5_one, surface_color::rgb565,
		surface_color::xrgb8_zero, surface_color::xrgb8_one, surface_color::argb8, surface_color::b8, surface_color::gb8,
		surface_color::rgba16f, surface_color::rgba32f, surface_color::r32f, surface_color::xbgr8_zero, surface_color::xbgr8_one, surface_color::abgr8})
	{
		const auto bytes = color_layout(format).pixel_bytes;
		const auto reference = surfaces.acquire({0x68000, {4, 4, 1, format, false}, bytes * 4, {}});
		const auto reference_depth = surfaces.acquire({0x6a000, {4, 4, 1, surface_depth::z24s8_uint, false}, 16, {}});
		draw_targets reference_targets;
		reference_targets.colors[0] = reference;
		reference_targets.depth = reference_depth;
		reference_targets.register_color_format = format;
		const auto reference_layout = plan_guest_surface(0x68000, 4, 4, bytes);
		ram.write(reference_layout.range, std::vector<std::byte>(reference_layout.range.size, std::byte{0xa7}));
		surfaces.clear(reference, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
		surfaces.clear(reference_depth, 3, 0, 0x40000042, 255, {0, 0, 4, 4});
		surfaces.flush(reference);
		const auto initial = ram.read_surface(reference_layout).bytes;
		const auto full_packet = sample_packet(0, 0, false, positions, indices, descriptors);
		executor.execute(full_packet, reference_targets);
		surfaces.flush(reference);
		const auto written = ram.read_surface(reference_layout).bytes;
		for (unsigned mode : {3u, 4u, 5u})
			for (unsigned raster : {1u, 2u})
				for (unsigned mask : {0xffffu, 5u})
				{
					const unsigned count = mode == 3 ? 2 : 4, sx = 2, sy = count / 2;
					test::framebuffer_layout layout;
					layout.target = 2;
					layout.color_format = static_cast<unsigned>(format);
					layout.color_addresses = {0, 0x70000, 0, 0};
					layout.actual_color_pitch[1] = 4 * sx * bytes + 16;
					layout.zeta_address = 0x74000;
					layout.actual_zeta_pitch = 4 * sx * 4 + 16;
					layout.aa_mode = mode;
					layout.aa_factors = {sx, sy};
					layout.raster_type = raster;
					const auto framebuffer = framebuffer_snapshot::capture(layout, true, raster == 2 ? std::optional{std::array<std::size_t, 3>{16, 16, 1}} : std::nullopt);
					const auto targets = acquire_framebuffer(framebuffer, surfaces);
					const auto& color = *framebuffer.colors()[0];
					const auto& depth = *framebuffer.depth();
					std::vector<std::byte> expected(color.guest.range.size, std::byte{0xa7});
					ram.write(color.guest.range, expected);
					ram.write(depth.guest.range, std::vector<std::byte>(depth.guest.range.size, std::byte{0xb9}));
					surfaces.clear(targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
					surfaces.clear(targets.depth, 3, 0, 0x40000042, 255, {0, 0, 4, 4});
					const auto packet = sample_packet(0, mask, false, positions, indices, descriptors);
					draw_execution_state execution;
					execution.queries = &queries;
					queries.begin(0, true);
					executor.execute(packet, framebuffer, execution);
					queries.end(0);
					require(queries.consume(0) == 16u * std::popcount(mask & ((1u << count) - 1)), "Allformat MSAA query lost selected samples");
					for (unsigned y = 0; y < 4 * sy; ++y)
						for (unsigned x = 0; x < 4 * sx; ++x)
						{
							const unsigned sample = x % sx + sx * (y % sy);
							const auto& pixel = mask & (1u << sample) ? written : initial;
							std::copy_n(pixel.begin(), bytes, expected.begin() + sample_storage_offset(color, x, y));
						}
					surfaces.flush(targets.colors[0]);
					surfaces.flush(targets.depth);
					const auto actual = ram.read_surface(color.guest).bytes;
					for (std::size_t i = 0; i < actual.size(); ++i)
						if (actual[i] != expected[i]) throw std::runtime_error("Allformat MSAA guest color/padding differs:format=" + std::to_string(static_cast<unsigned>(format)) +
							" mode=" + std::to_string(mode) + " raster=" + std::to_string(raster) + " mask=" + std::to_string(mask) + " byte=" + std::to_string(i) +
							" actual=" + std::to_string(std::to_integer<unsigned>(actual[i])) + " expected=" + std::to_string(std::to_integer<unsigned>(expected[i])));
					++cases;
				}
	}
	require(cases == 168, "Allformat MSAA framebuffer matrix omitted formats/modes/storage/masks");
	std::cout << "Native owned MSAA colors:168 GPU cases,all14guest color formats,AA3/4/5,linear/Morton B-plus-depth,full/selective masks despite disabled hint,individual exploded guest samples match single-sample native codec baseline with complete padding preservation passed\n";
}
void verify_versioned_upload_batch()
{
	context ctx(1);
	auto ledger = std::make_shared<heap_budget>(ctx, 8u << 20); ctx.set_buffer_budget(ledger);
	auto ram = std::make_shared<memory>();
	surface_cache surfaces(ctx, ram, 8u << 20, ledger);
	guest_texture_cache textures(ctx, ram, [](unsigned offset, unsigned, std::size_t) { return offset * 16; }, 1u << 20, ledger);
	draw_executor executor(ctx, surfaces, textures);
	auto state = registers(); texture_registers(state, 0x700); set_vertex_texture_registers(state, 0x800);
	for (unsigned unit = 1; unit < 4; ++unit) for (unsigned word = 0; word < 8; ++word)
		state.registers[0x900 / 4 + unit * 8 + word] = state.registers[0x900 / 4 + word];
	for (unsigned unit = 1; unit < 16; ++unit)
	{
		for (unsigned word = 0; word < 8; ++word) state.registers[0x1a00 / 4 + unit * 8 + word] = state.registers[0x1a00 / 4 + word];
		state.registers[0x1840 / 4 + unit] = state.registers[0x1840 / 4];
	}
	std::array<std::array<std::byte,64>,20> source{};
	for (unsigned i = 0; i < 20; ++i)
	{
		const auto offset = i < 4 ? 0x800u + i * 0x10 : 0x700u + (i - 4) * 0x10;
		state.registers[(i < 4 ? 0x900 / 4 + i * 8 : 0x1a00 / 4 + (i - 4) * 8)] = offset;
		for (unsigned b = 0; b < 64; ++b) source[i][b] = std::byte((b * 37 + i * 17 + b / 3) & 255);
		ram->write({offset * 16,64}, source[i]);
	}
	const auto vertex = vertex_texture_snapshot::capture(state.registers,15);
	const auto fragment = fragment_texture_snapshot::capture(state.registers,65535);
	ctx.begin(); const auto caller = ctx.recording_submission(); const auto nested_reads = ram->reads;
	bool nested = false;
	try { (void)executor.prepare_versioned_textures(vertex,fragment); } catch (const std::logic_error&) { nested = true; }
	require(nested && caller.same_batch(ctx.recording_submission()) && !caller.ready() && ram->reads == nested_reads,
		"Productive batch disturbed a caller recording before preflight"); ctx.cancel();
	const auto encoders = ctx.encoding_statistics(); const auto reads = ram->reads;
	const auto initial_gate = [ctx.device() newSharedEvent];
	struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } initial_gate_guard{initial_gate};
	const std::array initial_waits{guest_texture_upload_dependency{initial_gate,1}};
	auto initial_job = executor.enqueue_versioned_textures(vertex,fragment,initial_waits);
	require(!initial_job.ready() && initial_job.current(vertex,fragment) && textures.statistics().uploads == 0 && textures.statistics().entries == 0,
		"Twenty-source draw preparation did not return with native work pending and private");
	for (unsigned i = 0; i < 20; ++i)
	{
		const auto plan = program_sampler_plan::capture(i < 4 ? vertex.at(i).upload_words() : fragment.at(i - 4),initial_job.metadata(i < 4,i < 4 ? i : i - 4));
		require(plan.dimension == 1 && plan.format_class == 1 && plan.samples == 1, "Pending twenty-source sampler metadata changed its program contract");
	}
	initial_gate.signaledValue = 1;
	auto prepared = initial_job.finish();
	require(prepared.revisions.size() == 20 && prepared.current(vertex,fragment) && textures.statistics().uploads == 20 &&
		textures.statistics().upload_batches == 1 && textures.statistics().coalesced_uploads == 0 && ram->reads == reads + 20 &&
		ctx.encoding_statistics().compute_encoders == encoders.compute_encoders + 1, "Productive twenty-source preparation did not use one native VM upload batch");
	const auto output = make_buffer(ctx,20 * 1024,buffer_storage::shared);
	ctx.begin();
	for (unsigned i = 0; i < 20; ++i)
	{
		const auto image = i < 4 ? prepared.textures.vertex(i).texture : prepared.textures.fragment(i - 4).texture;
		texture_to_buffer(ctx,image,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(image.width,image.height,1)},output,i * 1024,256);
	}
	ctx.submit().wait();
	for (unsigned i = 0; i < 20; ++i) for (unsigned b = 0; b < 64; ++b)
	{
		const auto address = i * 1024 + (i < 4 ? b : (b / 16) * 256 + b % 16);
		require(static_cast<const std::byte*>(output.contents)[address] == source[i][b ^ 3], "Productive upload batch changed a vertex/fragment captured word");
	}
	const auto unrelated = surfaces.acquire({0xc000,{4,4,1,surface_color::argb8,false},16,{}});
	surfaces.clear(unrelated,0xf0,0xff112233,0,255,{0,0,4,4});
	const auto after_reads = ram->reads; const auto after_encoders = ctx.encoding_statistics();
	const auto before_ready = surfaces.transaction_statistics();
	{
		auto batch = surfaces.batch_draws(); std::optional<submission> recording;
		for (unsigned i = 0; i < 32; ++i)
		{
			surfaces.clear(unrelated,0xf0,0xff204080,0,255,{0,0,4,4});
			if (!recording) recording = ctx.recording_submission();
			const auto current = executor.prepare_versioned_textures(vertex,fragment);
			require(current.current(vertex,fragment) && recording->same_batch(ctx.recording_submission()) && !recording->ready(),
				"Twenty-source ready epoch closed an unrelated draw batch");
		}
		batch.finish();
	}
	require(textures.statistics().uploads == 20 && textures.statistics().upload_batches == 1 && ram->reads == after_reads &&
		ctx.encoding_statistics().compute_encoders == after_encoders.compute_encoders &&
		surfaces.transaction_statistics().submitted == before_ready.submitted + 1 && surfaces.transaction_statistics().draw_batches == before_ready.draw_batches + 1,
		"Ready twenty-source hits reread VM or submitted another upload/draw batch");
	verify(ctx,surfaces,unrelated,{32,64,128,255});
	for (unsigned i = 0; i < 20; ++i)
	{
		if (i) prepared = executor.prepare_versioned_textures(vertex,fragment);
		require(prepared.current(vertex,fragment), "Refreshed batch already has stale source tickets");
		const auto address = i < 4 ? 0x8000u + i * 256 : 0x7000u + (i - 4) * 256;
		ram->write({address,1},std::array{std::byte{0x62}});
		require(!prepared.current(vertex,fragment), "One of twenty independent batch source revisions escaped final validation");
	}
	textures.clear();
	std::array<fragment_texture_registers,5> aliases; aliases.fill(fragment.at(0)); aliases[1].format |= 0x40u << 8;
	const auto before = textures.statistics(); const auto alias_reads = ram->reads;
	const auto shared = textures.resolve_versioned_batch(aliases,surfaces);
	for (unsigned i = 0; i < shared.size(); ++i)
		require(shared[i].image.texture == shared[0].image.texture && shared[i].revision.current() &&
			shared[i].image.metadata.coordinates.scale[0] == (i == 1 ? .25f : 1.f), "Coalesced source lost its per-request UN metadata or revision");
	require(textures.statistics().upload_batches == before.upload_batches + 1 && textures.statistics().uploads == before.uploads + 1 &&
		textures.statistics().coalesced_uploads == before.coalesced_uploads + 4 && ram->reads == alias_reads + 1,
		"Duplicate pending sources were captured/uploaded separately");
	textures.clear();
	const auto native = surfaces.acquire({0x4000,{4,4,1,surface_color::argb8,false},16,{}});
	surfaces.clear(native,0xf0,0xff112233,0,255,{0,0,4,4});
	auto alias_state = registers(); texture_registers(alias_state,0x400);
	const auto direct = fragment_texture_snapshot::capture(alias_state.registers,1).at(0);
	auto cast = direct; cast.format = (cast.format & ~(0xffu << 8)) | (0xa1u << 8); cast.image_rect = (16u << 16) | 4;
	auto cast16 = cast; cast16.format = (cast16.format & ~(0xffu << 8)) | (0xb2u << 8); cast16.image_rect = (8u << 16) | 4;
	auto un_cast = cast; un_cast.format |= 0x40u << 8;
	const std::array mixed{fragment.at(0),direct,fragment.at(1),cast,cast16,un_cast};
	const auto mixed_before = textures.statistics();
	const auto mixed_encoders = ctx.encoding_statistics();
	const auto source_before = surfaces.snapshot_statistics().created;
	const auto source_submits = surfaces.transaction_statistics().submitted;
	const auto mixed_gate = [ctx.device() newSharedEvent]; gate_scope mixed_guard{mixed_gate};
	const std::array mixed_waits{guest_texture_upload_dependency{mixed_gate,1}};
	auto mixed_job = textures.prepare_versioned_batch(mixed,surfaces,mixed_waits);
	require(!mixed_job.ready() && mixed_job.current() && textures.statistics().uploads == mixed_before.uploads &&
		textures.statistics().casts_created == mixed_before.casts_created && mixed_job.metadata(5).coordinates.scale[0] == 1.f / 16 &&
		surfaces.snapshot_statistics().created == source_before && surfaces.transaction_statistics().submitted == source_submits,
		"Mixed pending VM/cast job waited,published early or lost its independent UN metadata");
	mixed_gate.signaledValue = 1;
	const auto native_mixed = mixed_job.finish();
	require(textures.statistics().upload_batches == mixed_before.upload_batches + 1 && textures.statistics().uploads == mixed_before.uploads + 2 &&
		textures.statistics().casts_created == mixed_before.casts_created + 2 && textures.statistics().coalesced_casts == mixed_before.coalesced_casts + 1 &&
		ctx.encoding_statistics().compute_encoders == mixed_encoders.compute_encoders + 1 && surfaces.snapshot_statistics().created == source_before + 1 &&
		surfaces.transaction_statistics().submitted == source_submits && native_mixed[1].revision.current() && native_mixed[3].revision.current() &&
		native_mixed[4].revision.current() && native_mixed[5].image.texture == native_mixed[3].image.texture && native_mixed[5].image.metadata.coordinates.scale[0] == 1.f / 16,
		"Mixed VM/native alias/cast inputs overlapped encoding or omitted their source receipts");
	const auto readback = make_buffer(ctx,3072,buffer_storage::shared);
	ctx.begin();
	texture_to_buffer(ctx,native_mixed[1].image.texture,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)},readback,0,256);
	texture_to_buffer(ctx,native_mixed[3].image.texture,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(16,4,1)},readback,1024,256);
	texture_to_buffer(ctx,native_mixed[4].image.texture,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(8,4,1)},readback,2048,256);
	ctx.submit().wait();
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 16; ++x)
	{
		require(static_cast<const unsigned char*>(readback.contents)[y * 256 + x] == std::array{0x33u,0x22u,0x11u,0xffu}[x % 4], "Mixed batch direct native alias changed bytes");
		require(static_cast<const unsigned char*>(readback.contents)[1024 + y * 256 + x] == std::array{0xffu,0x11u,0x22u,0x33u}[x % 4], "Mixed batch native R8 cast changed guest-order bytes");
		require(static_cast<const unsigned char*>(readback.contents)[2048 + y * 256 + x] == std::array{0x11u,0xffu,0x33u,0x22u}[x % 4], "Mixed batch native R16 cast changed target endian bytes");
	}
	const auto cast_hits_before = textures.statistics(); const auto cast_hit_encoders = ctx.encoding_statistics();
	const auto cast_hits = textures.resolve_versioned_batch(std::array{cast,cast16},surfaces);
	require(cast_hits[0].image.texture == native_mixed[3].image.texture && cast_hits[1].image.texture == native_mixed[4].image.texture &&
		textures.statistics().upload_batches == cast_hits_before.upload_batches && textures.statistics().casts_reused == cast_hits_before.casts_reused + 2 &&
		ctx.encoding_statistics().compute_encoders == cast_hit_encoders.compute_encoders, "Ready batched casts were reconverted or resubmitted");
	ram->write({0x4000,1},std::array{std::byte{0x44}});
	require(!native_mixed[1].revision.current() && !native_mixed[3].revision.current() && !native_mixed[4].revision.current() && !native_mixed[5].revision.current() &&
		native_mixed[0].revision.current() && native_mixed[2].revision.current(),
		"Mixed batch CPU write invalidated the wrong source receipts");
	std::cout << "Productive versioned DrawExecutor:20 vertex/fragment misses,one native encoder/submission,1280 exact bytes;32 twenty-source hits without upload/read/encoding;all20 final revisions independently reject writes;five normalized/UN aliases share one pending upload with separate receipts passed\n";
	std::cout << "Native cache mixed batch:one cold native source snapshot,two VM misses,two distinct R8/R16 casts andone UN cast alias share one native encoder/submission;all remain private behind the same event,192 exact bytes,separate metadata/receipts,ready casts without new encoding andexact source invalidation passed\n";
	std::array<std::byte,64> pixels;
	for (unsigned b = 0; b < pixels.size(); ++b) pixels[b] = std::array{std::byte{255},std::byte{20},std::byte{40},std::byte{60}}[b % 4];
	ram->write({0x6000,64},pixels);
	auto draw_state = registers(); texture_registers(draw_state,0x600);
	const auto draw = snapshot(draw_state,true,1);
	const auto epoch = executor.prepare_versioned_textures(draw.vertex_textures(),draw.textures());
	draw_request request;
	const std::array<std::byte,12> indices{std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{1},std::byte{0},std::byte{0},std::byte{0},std::byte{2}};
	const auto geometry = plan_indexed_geometry(guest_topology::triangles,indices,4,0,{}, {},provoking_vertex::last);
	request.geometry = geometry.geometry; request.fetch.attribute_mask = 1;
	request.fetch.attributes[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27),(1u << 31) | (1u << 29)};
	test::vertex_data_processor processor; processor.descriptors = request.fetch.attributes;
	for (float value : {-1.f,-1.f,.5f,1.f,3.f,-1.f,.5f,1.f,-1.f,3.f,.5f,1.f})
		for (unsigned shift : {24u,16u,8u,0u}) processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
	const auto vertices = vertex_data_snapshot::capture(processor,0,test::vertex_data_metadata{1},test::vertex_data_state{},geometry.streams,{48,0},48);
	apply_vertex_data_snapshot(ctx,vertices,request);
	apply_draw_environment(draw_environment_snapshot::capture(test::environment_state{}),4,4,request);
	draw_targets targets; targets.colors[0] = surfaces.acquire({0xa000,{4,4,1,surface_color::argb8,false},16,{}});
	const auto uploads = textures.statistics().uploads;
	executor.execute(draw,targets,request,epoch.textures);
	verify(ctx,surfaces,targets.colors[0],{20,40,60,255});
	require(textures.statistics().uploads == uploads && epoch.current(draw.vertex_textures(),draw.textures()), "Draw reran resolution instead of consuming the completed batch epoch");
	std::cout << "Productive grouped epoch consumed by native RSX vertex/fragment DrawExecutor:all16 RGBA pixels exact,no additional texture upload during draw passed\n";

	ctx.geometry_uploads().clear();
	draw_executor cold(ctx, surfaces, textures);
	const auto free = ledger->statistics();
	const auto available = free.budget - free.reserved_bytes;
	constexpr std::size_t block = 64 << 10;
	require(available > block, "Geometry budget test lacks free space");
	std::vector<id<MTLBuffer>> reservations{ledger->shared_buffer((available - block) / block * block)};
	for (;;)
	{
		try { reservations.push_back(ledger->shared_buffer(16)); }
		catch (const heap_budget_error&) { break; }
	}
	unsigned admissions = 0;
	const auto recover = ledger->register_reclaimer([&](memory_pressure_request pressure)
	{
		require(!pressure.native_failure && admissions == 0 && cold.vertex_statistics().prewarmed == 1 &&
			cold.fragment_statistics().prewarmed == 1 && !cold.vertex_statistics().compiled &&
			!cold.fragment_statistics().compiled && cold.pipeline_statistics().prewarmed == 1 && !cold.pipeline_statistics().created,
			"Geometry allocation reached pressure recovery after waiting for a shader");
		++admissions;
		reservations.clear();
	});
	const auto before_geometry = ctx.geometry_uploads().statistics().created;
	cold.execute(draw, targets, request, epoch.textures);
	require(admissions == 1 && ctx.geometry_uploads().statistics().created == before_geometry + 1 &&
		cold.compilation_statistics().admitted == 3 && cold.pipeline_statistics().prewarmed == 1,
		"Early geometry admission failed to recover or duplicated a shader or pipeline job");
	verify(ctx,surfaces,targets.colors[0],{20,40,60,255});
	std::cout << "Early draw geometry: budget recovery ran after both shader jobs started and before shader materialization; one PSO and 16 exact pixels passed\n";
}
void verify_owned_texture_boundary()
{
	using namespace std::chrono_literals;
	context ctx(1, 64, 5s); auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
	auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,ledger);
	guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger);
	draw_executor executor(ctx,surfaces,textures); (void)textures.null_images();
	auto state = registers();
	const auto empty_vertex = vertex_texture_snapshot::capture(state.registers,0), disabled_vertex = vertex_texture_snapshot::capture(state.registers,1);
	const auto empty_fragment = fragment_texture_snapshot::capture(state.registers,0), disabled_fragment = fragment_texture_snapshot::capture(state.registers,1);
	const auto target = surfaces.acquire({0xa000,{4,4,1,surface_color::argb8,false},16,{}});
	surfaces.clear(target,0xf0,0xff112233,0,255,{0,0,4,4});
	const auto initial = surfaces.status(target).content_generation; const auto before = surfaces.transaction_statistics();
	{
		auto batch = surfaces.batch_draws(); std::optional<submission> recording;
		for (unsigned i = 0; i < 32; ++i)
		{
			auto epoch_job = executor.enqueue_versioned_textures(i & 1 ? empty_vertex : disabled_vertex,i & 1 ? empty_fragment : disabled_fragment);
			require(epoch_job.ready(), "Empty/disabled draw epoch did not remain CPU-only in its owned recording");
			const auto epoch = epoch_job.finish();
			require(epoch.revisions.empty(), "Empty/disabled draw epoch captured VM sources");
			auto direct = textures.prepare_versioned_batch({},surfaces); require(direct.ready() && direct.finish().empty(), "Empty cache epoch disturbed an owned draw recording");
			surfaces.clear(target,0xf0,0xff204080,0,255,{0,0,4,4});
			if (!recording) recording = ctx.recording_submission();
			require(recording->same_batch(ctx.recording_submission()) && !recording->ready() && surfaces.transaction_statistics().submitted == before.submitted &&
				surfaces.status(target).content_generation == initial, "Texture preparation submitted/cancelled an owned texture-free draw prefix");
		}
		const std::array nil_wait{guest_texture_upload_dependency{nil,1}};
		bool invalid = false;
		try { (void)executor.enqueue_versioned_textures(empty_vertex,empty_fragment,nil_wait); } catch (const std::invalid_argument&) { invalid = true; }
		require(invalid && recording->same_batch(ctx.recording_submission()) && !recording->ready(), "Invalid preflight modified an owned draw recording");
		batch.finish();
	}
	require(surfaces.transaction_statistics().draw_batches == before.draw_batches + 1 && surfaces.transaction_statistics().batched_draws == before.batched_draws + 32 &&
		surfaces.status(target).content_generation == initial + 32 && ram->reads == 0, "Thirty-two prepared texture-free passes did not retain one native submission");
	verify(ctx,surfaces,target,{32,64,128,255});
	{
		textures.clear_null_images();
		auto batch = surfaces.batch_draws(); surfaces.clear(target,0xf0,0xff112233,0,255,{0,0,4,4});
		const auto prefix = ctx.recording_submission();
		const auto encoders = ctx.encoding_statistics();
		const auto submissions = surfaces.transaction_statistics().submitted;
		auto cold_nulls = executor.enqueue_versioned_textures(disabled_vertex, disabled_fragment);
		require(cold_nulls.ready() && cold_nulls.finish().revisions.empty() && textures.statistics().null_initializations == 2 &&
			!prefix.ready() && prefix.same_batch(ctx.recording_submission()) &&
			surfaces.transaction_statistics().submitted == submissions &&
			ctx.encoding_statistics().compute_encoders == encoders.compute_encoders,
			"Cold disabled samplers submitted or interrupted the owned draw prefix");
		batch.finish(); prefix.wait();
	}
	{
		auto batch = surfaces.batch_draws(); surfaces.clear(target,0xf0,0xff778899,0,255,{0,0,4,4});
		const auto prefix = ctx.recording_submission(); const auto batches = surfaces.transaction_statistics().draw_batches;
		auto textured = registers(); texture_registers(textured,0x100);
		const auto vertex = vertex_texture_snapshot::capture(textured.registers,0); const auto fragment = fragment_texture_snapshot::capture(textured.registers,1);
		const auto gate = [ctx.device() newSharedEvent];
		struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
		auto upload = executor.enqueue_versioned_textures(vertex,fragment,std::array{guest_texture_upload_dependency{gate,1}});
		require(!upload.ready() && surfaces.transaction_statistics().draw_batches == batches + 1 && textures.statistics().entries == 0,
			"Enabled sampler did not submit its owned prefix before a new event-gated texture job");
		batch.finish(); gate.signaledValue = 1; (void)upload.finish();
		verify(ctx,surfaces,target,{119,136,153,255});
	}
	{
		textures.clear_null_images();
		const auto gate = [ctx.device() newSharedEvent];
		struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } guard{gate};
		ctx.begin();
		ctx.wait_event(gate, 1);
		const auto occupied = ctx.submit();
		const auto encoders = ctx.encoding_statistics();
		const auto reads = ram->reads;
		const auto cold_nulls = executor.prepare_versioned_textures(disabled_vertex, disabled_fragment);
		require(!occupied.ready() && cold_nulls.revisions.empty() &&
			cold_nulls.textures.null_images()[3].texture && textures.statistics().null_initializations == 3 &&
			ctx.encoding_statistics().compute_encoders == encoders.compute_encoders && ram->reads == reads,
			"Cold disabled samplers waited for the only Metal4 command slot or read VM");
		gate.signaledValue = 1;
		occupied.wait();
	}
	ctx.begin(); const auto caller = ctx.recording_submission(); bool foreign = false;
	try { (void)executor.enqueue_versioned_textures(empty_vertex,empty_fragment); } catch (const std::logic_error&) { foreign = true; }
	require(foreign && caller.same_batch(ctx.recording_submission()) && !caller.ready(), "Texture boundary submitted a foreign caller recording"); ctx.cancel();
	std::cout << "Cold null textures: disabled units preserve the recording and initialize with the sole slot occupied; enabled uploads still close their owned prefix; no disabled VM reads\n";
}
void verify_ready_texture_boundary()
{
	unsigned profiles=0,hits=0;std::size_t checked_bytes=0;
	for(unsigned kind=0;kind<6;++kind)for(unsigned samples:{1u,4u})
	{
		if(!kind&&samples!=1)continue;
		@autoreleasepool
		{
			context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,16u<<20);ctx.set_buffer_budget(ledger);
			auto ram=std::make_shared<memory>();surface_cache surfaces(ctx,ram,16u<<20,ledger);
			guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return offset*16;},1u<<20,ledger);draw_executor executor(ctx,surfaces,textures);
			auto state=registers();texture_registers(state,0x100);const unsigned base=0x1a00/4;
			if(kind==2){state.registers[base+1]=(state.registers[base+1]&~(255u<<8))|(0xa1u<<8);state.registers[base+6]=(16u<<16)|4;}
			if(kind==3){state.registers[base+1]=(state.registers[base+1]&~(255u<<8))|(0xb2u<<8);state.registers[base+6]=(8u<<16)|4;}
			if(kind==4)state.registers[base+1]=(state.registers[base+1]&65535u)|(2u<<16)|4;
			if(kind==5){state.registers[base+1]=(2u<<16)|(0xb1u<<8)|(3u<<4)|8|2;state.registers[0x1840/4]=(2u<<20)|16;}
			const auto vertex=vertex_texture_snapshot::capture(state.registers,0);const auto fragment=fragment_texture_snapshot::capture(state.registers,1);const auto words=fragment.at(0);const auto plan=plan_guest_texture(words);
			std::vector<std::byte> bytes(plan.storage_bytes);for(unsigned i=0;i<bytes.size();++i)bytes[i]=std::byte(std::array{255u,20u,40u,60u}[i%4]);if(kind==5)for(unsigned i=0;i<bytes.size();++i)bytes[i]=std::byte(std::array{0x7eu,0u,0u,42u}[i%4]);ram->write({0x1000,bytes.size()},bytes);
			cached_surface_ref source;
			if(kind)
			{
				const unsigned side=samples==1?4:2;const surface_format format=kind==5?surface_format{surface_depth::z24s8_float}:surface_format{surface_color::argb8};source=surfaces.acquire({0x1000,{side,side,samples,format,false},16,{}});surfaces.prepare(source,surface_initialization::background);
				surfaces.draw(std::array{cached_surface_use{source,true}},[&](context& commands,auto images)
				{
					auto pass=[MTL4RenderPassDescriptor new];if(kind==5){pass.depthAttachment.texture=images[0];pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionStore;pass.depthAttachment.clearDepth=.5;pass.stencilAttachment.texture=images[0];pass.stencilAttachment.loadAction=MTLLoadActionClear;pass.stencilAttachment.storeAction=MTLStoreActionStore;pass.stencilAttachment.clearStencil=53;}
					else{pass.colorAttachments[0].texture=images[0];pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;pass.colorAttachments[0].clearColor=MTLClearColorMake(20./255,40./255,60./255,1);}(void)commands.render(pass);
				});
			}
			const auto initial=executor.prepare_versioned_textures(vertex,fragment);const auto image=initial.textures.fragment(0).texture;require(image!=nil,"Ready boundary lacks an initial image");
			if(kind==1||kind==2||kind==3)surfaces.spill(source);
			const auto target=surfaces.acquire({0xa000,{4,4,1,surface_color::argb8,false},16,{}});surfaces.clear(target,0xf0,0xff112233,0,255,{0,0,4,4});
			const auto before=surfaces.transaction_statistics();const auto uploads=textures.statistics();const auto reads=ram->reads;
			const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};
			{
				auto batch=surfaces.batch_draws();std::optional<submission> recording;
				for(unsigned i=0;i<32;++i)
				{
					surfaces.clear(target,0xf0,0xff204080,0,255,{0,0,4,4});if(!recording)recording=ctx.recording_submission();const auto allocated=ledger->statistics().created;const auto encoders=ctx.encoding_statistics();
					auto job=executor.enqueue_versioned_textures(vertex,fragment,std::array{guest_texture_upload_dependency{gate,1}});require(job.ready(),"Ready texture began event-gated GPU work");const auto current=job.finish();
					require(current.textures.fragment(0).texture==image&&current.current(vertex,fragment)&&recording->same_batch(ctx.recording_submission())&&!recording->ready()&&surfaces.transaction_statistics().submitted==before.submitted,"Ready texture closed unrelated recording orlost receipt/image");
					require(ledger->statistics().created==allocated&&ctx.encoding_statistics().compute_encoders==encoders.compute_encoders&&ctx.encoding_statistics().render_encoders==encoders.render_encoders&&ram->reads==reads,"Ready texture allocated,encoded orrecaptured VM");++hits;
				}
				batch.finish();
			}
			require(surfaces.transaction_statistics().submitted==before.submitted+1&&surfaces.transaction_statistics().draw_batches==before.draw_batches+1&&textures.statistics().upload_batches==uploads.upload_batches,"Ready textures split one 32-draw batch");if(kind==1||kind==2||kind==3)require(surfaces.status(source).spilled,"Ready texture restored spilled source");verify(ctx,surfaces,target,{32,64,128,255});
			const auto readable = image.sampleCount == 1 ? image : textures.resolve_versioned(words, surfaces).image.texture;
			const auto row=kind==2?16u:kind==3?16u:kind==5?32u:16u;const auto raw=make_buffer(ctx,256*4,buffer_storage::shared);texture_region area;area.size=MTLSizeMake(readable.width,readable.height,1);ctx.begin();texture_to_buffer(ctx,readable,area,raw,0,256);ctx.submit().wait();
			for(unsigned y=0;y<4;++y)for(unsigned x=0;x<row;++x)
			{
				unsigned wanted=kind==2?std::array{255u,20u,40u,60u}[x%4]:kind==3?std::array{20u,255u,60u,40u}[x%4]:std::array{60u,40u,20u,255u}[x%4];
				if(kind==5){const unsigned bits=std::bit_cast<unsigned>(x%8<4?.5f:53.f);wanted=(bits>>((x%4)*8))&255;}
				require(static_cast<const unsigned char*>(raw.contents)[y*256+x]==wanted,"Ready texture changed exact native bytes");++checked_bytes;
			}
			textures.clear();if(kind==1)surfaces.clear_snapshots();
			{
				auto batch=surfaces.batch_draws();surfaces.clear(target,0xf0,0xff335577,0,255,{0,0,4,4});const auto prefix=ctx.recording_submission();const auto count=surfaces.transaction_statistics().draw_batches;
				const auto cold=textures.resolve_versioned(words,surfaces);require(cold.image.texture!=nil&&prefix.ready()&&surfaces.transaction_statistics().draw_batches==count+1,"Synchronous VM/cast/assembly miss didnotclose its native draw prefix");batch.finish();
			}
			verify(ctx,surfaces,target,{51,85,119,255});++profiles;
		}
	}
	std::cout<<"Ready texture boundary:"<<profiles<<" VM/native1-4sample/R8-R16cast/cube-Depth3Dassembly profiles,"<<hits<<" active ready epochs preserve32-draw single batches despite unsignalled dependencies;zero new allocation/encoding/VMreads,spilled sources stay spilled,"<<checked_bytes<<" exact native bytes andtarget pixels;11synchronous cold misses closeprefix correctly passed\n";
}
void verify_predicated_texture_preparation()
{
	context ctx(2);
	auto ledger = std::make_shared<heap_budget>(ctx, 16u << 20);
	ctx.set_buffer_budget(ledger);
	auto ram = std::make_shared<memory>();
	surface_cache surfaces(ctx, ram, 16u << 20, ledger);
	guest_texture_cache textures(ctx, ram, [](unsigned offset, unsigned, std::size_t) { return offset * 16; }, 1u << 20, ledger);
	draw_executor executor(ctx, surfaces, textures);
	const surface_allocation_description color{4, 4, 1, surface_color::argb8, false};
	const auto source = surfaces.acquire({0x1000, color, 16, {}});
	const auto donor = surfaces.acquire({0x2000, color, 16, {}});
	draw_targets targets;
	targets.colors[0] = surfaces.acquire({0x3000, color, 16, {}});
	surfaces.clear(donor, 0xf0, 0xff14283c, 0, 255, {0, 0, 4, 4});
	surfaces.clear(targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
	surfaces.defer_copy(donor, {0, 0, 4, 4}, source, 0, 0);
	auto words = registers();
	texture_registers(words, 0x100);
	const auto draw = snapshot(words, true, 1);
	const std::array<std::byte, 12> indices{{{}, {}, {}, {}, {}, {}, {}, std::byte{1}, {}, {}, {}, std::byte{2}}};
	const auto geometry = plan_indexed_geometry(guest_topology::triangles, indices, 4, 0, {}, {}, provoking_vertex::last);
	draw_request request;
	request.geometry = geometry.geometry;
	request.fetch.attribute_mask = 1;
	request.fetch.attributes[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27), (1u << 31) | (1u << 29)};
	test::vertex_data_processor processor;
	processor.descriptors = request.fetch.attributes;
	for (float value : {-1.f,-1.f,.5f,1.f,3.f,-1.f,.5f,1.f,-1.f,3.f,.5f,1.f})
		for (unsigned shift : {24u,16u,8u,0u}) processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
	const auto vertices = vertex_data_snapshot::capture(processor, 0, test::vertex_data_metadata{1}, test::vertex_data_state{}, geometry.streams, {48,0}, 48);
	apply_vertex_data_snapshot(ctx, vertices, request);
	apply_draw_environment(draw_environment_snapshot::capture(test::environment_state{}), 4, 4, request);
	query_pool queries(ctx, 1, 1, ledger);
	ctx.begin();
	const auto off = queries.capture_report_predicate({{}, 0, true});
	const auto on = queries.capture_report_predicate({{}, 1, true});
	require(off && on, "Texture preparation fixture could not capture both reports");
	ctx.submit().wait();
	request.queries = &queries;
	request.predicate = off;
	const auto before = ram->read({0x1000, 64});
	const auto upload_count = textures.statistics().upload_batches;
	const auto enqueue = [&](const report_predicate& predicate)
	{
		return executor.enqueue_versioned_textures(draw.vertex_textures(), draw.textures(), {}, &predicate);
	};
	executor.execute(draw, targets, request);
	bool suppressed = false;
	try { (void)enqueue(*off); }
	catch (const predicated_texture_suppressed&) { suppressed = true; }
	require(suppressed && textures.statistics().upload_batches == upload_count,
		"False predicate started a versioned upload for its deferred texture input");
	const auto gate = [ctx.device() newSharedEvent];
	ctx.begin();
	ctx.wait_event(gate, 1);
	const auto pending = queries.capture_report_predicate({{}, 0, true});
	const auto producer = ctx.submit();
	surfaces.set_wait_timeout(std::chrono::milliseconds(1));
	bool timed_out = false;
	std::exception_ptr unexpected;
	try { (void)enqueue(*pending); }
	catch (const submission_wait_timeout&) { timed_out = true; }
	catch (...) { unexpected = std::current_exception(); }
	const bool producer_blocked = !producer.ready();
	gate.signaledValue = 1;
	producer.wait();
	surfaces.set_wait_timeout(std::chrono::seconds(30));
	if (unexpected) std::rethrow_exception(unexpected);
	require(timed_out && producer_blocked && textures.statistics().upload_batches == upload_count,
		"Unresolved predicate started a versioned upload for its deferred texture input");
	const auto after = ram->read({0x1000, 64});
	require(surfaces.status(source).deferred_copies == 1 && before.revision == after.revision && before.bytes == after.bytes,
		"False predicate materialized its deferred texture input");
	verify(ctx, surfaces, targets.colors[0], {17,34,51,255});
	request.predicate = on;
	const auto prepared = enqueue(*on).finish();
	executor.execute(draw, targets, request, prepared.textures);
	verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
	ctx.begin();
	ctx.wait_event(gate, 2);
	const auto pending_warm = queries.capture_report_predicate({{}, 0, true});
	const auto warm_producer = ctx.submit();
	surfaces.set_wait_timeout(std::chrono::milliseconds(1));
	bool warm_ready = false;
	unexpected = {};
	try
	{
		auto job = enqueue(*pending_warm);
		warm_ready = job.ready() && !warm_producer.ready();
	}
	catch (...) { unexpected = std::current_exception(); }
	gate.signaledValue = 2;
	warm_producer.wait();
	surfaces.set_wait_timeout(std::chrono::seconds(30));
	if (unexpected) std::rethrow_exception(unexpected);
	require(warm_ready, "Immutable texture hit waited on a pending predicate");
	std::cout << "Predicated texture preparation: direct and versioned false reports preserve deferred input, timeout keeps upload idle, true samples exact pixels, warm snapshot remains ready\n";
}

void verify_ready_textured_draws(bool completed_waits_only=false)
{
	unsigned completed_profiles=0,completed_draws=0,completed_timeouts=0,completed_cancels=0;
	unsigned draws=0, narrowed=0, same_batch_waits=0, pending_waits=0, independent_pending=0, opaque_pending=0;
	unsigned load_clears = 0, shader_clears = 0, opaque_passes = 0, multisample_clears = 0, multisample_copies = 0;
	for(bool native:{false,true})
	{
		context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,16u<<20);ctx.set_buffer_budget(ledger);auto ram=std::make_shared<memory>();surface_cache surfaces(ctx,ram,16u<<20,ledger);guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return offset*16;},1u<<20,ledger);draw_executor executor(ctx,surfaces,textures);
		std::array<std::byte,64> pixels;for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte(std::array{255u,20u,40u,60u}[i%4]);ram->write({0x1000,64},pixels);
		if(native){const auto source=surfaces.acquire({0x1000,{4,4,1,surface_color::argb8,false},16,{}});surfaces.clear(source,0xf0,0xff14283c,0,255,{0,0,4,4});}
		auto source=registers();texture_registers(source,0x100);
		const auto owned_registers = source.registers;
		const auto draw=snapshot(source,true,1);const auto initial=executor.prepare_versioned_textures(draw.vertex_textures(),draw.textures());
		draw_request request;const std::array<std::byte,12> indices{std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{1},std::byte{0},std::byte{0},std::byte{0},std::byte{2}};const auto geometry=plan_indexed_geometry(guest_topology::triangles,indices,4,0,{}, {},provoking_vertex::last);request.geometry=geometry.geometry;request.fetch.attribute_mask=1;request.fetch.attributes[0]={16u|(3u<<8)|(2u<<24)|(4u<<27),(1u<<31)|(1u<<29)};
		test::vertex_data_processor processor;processor.descriptors=request.fetch.attributes;for(float value:{-1.f,-1.f,.5f,1.f,3.f,-1.f,.5f,1.f,-1.f,3.f,.5f,1.f})for(unsigned shift:{24u,16u,8u,0u})processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value)>>shift));
		const auto vertices=vertex_data_snapshot::capture(processor,0,test::vertex_data_metadata{1},test::vertex_data_state{},geometry.streams,{48,0},48);apply_vertex_data_snapshot(ctx,vertices,request);apply_draw_environment(draw_environment_snapshot::capture(test::environment_state{}),4,4,request);
		draw_targets targets;targets.colors[0]=surfaces.acquire({0xa000,{4,4,1,surface_color::argb8,false},16,{}});surfaces.clear(targets.colors[0],0xf0,0xff112233,0,255,{0,0,4,4});executor.execute(draw,targets,request,initial.textures);
		ctx.wait_idle();
		const auto before=surfaces.transaction_statistics();const auto uploads=textures.statistics().upload_batches;const auto reads=ram->reads;std::optional<submission> recording;
		const auto before_barriers = ctx.encoding_statistics();
		{
			auto batch=surfaces.batch_draws();
			for(unsigned i=0;i<32;++i)
			{
				auto job=executor.enqueue_versioned_textures(draw.vertex_textures(),draw.textures());require(job.ready(),"Warm textured draw started GPU preparation");const auto ready=job.finish();require(ready.current(draw.vertex_textures(),draw.textures()),"Warm textured draw lost source receipt");executor.execute(draw,targets,request,ready.textures);if(!recording)recording=ctx.recording_submission();require(recording->same_batch(ctx.recording_submission())&&!recording->ready()&&surfaces.transaction_statistics().submitted==before.submitted,"Ready texture split productive textured draw batch");++draws;
			}batch.finish();
		}
		const auto after_barriers = ctx.encoding_statistics();
		require(after_barriers.guest_draw_fragment_barriers == before_barriers.guest_draw_fragment_barriers + 32 &&
			after_barriers.guest_draw_full_barriers == before_barriers.guest_draw_full_barriers,
			"Independent translated vertex/index inputs did not keep the vertex stage free");
		narrowed += 32;
		require(surfaces.transaction_statistics().submitted==before.submitted+1&&surfaces.transaction_statistics().draw_batches==before.draw_batches+1&&textures.statistics().upload_batches==uploads&&ram->reads==reads,"Textured draw batch uploaded orrecaptured source");verify(ctx,surfaces,targets.colors[0],{20,40,60,255});
		test::pass_state state;
		state.registers = owned_registers;
		state.current_draw_clause.command = 1;
		state.base_index = 0;
		test::pass_processor owned_processor;
		owned_processor.persistent = processor.persistent;
		owned_processor.descriptors = processor.descriptors;
		test::pass_layout layout;
		test::pass_metadata metadata;
		const auto fragment_code = program(true);
		test::fragment_program fragment;
		fragment.pointer = fragment_code.data();
		fragment.ucode_length = fragment_code.size();
		fragment.ctrl = rsx_fragment_float32;
		const auto owned = rsx_draw_packet::capture(state, owned_processor, layout, metadata, vertex_program(), fragment, 1,
			provoking_vertex::last, [](const auto&, unsigned, unsigned) { return std::pair<unsigned, unsigned>{48,0}; });
		const auto owned_encoders = ctx.encoding_statistics().render_encoders;
		const auto owned_submissions = surfaces.transaction_statistics().submitted;
		{
			auto batch = surfaces.batch_draws();
			for (unsigned i = 0; i < 32; ++i)
				executor.execute(owned, targets, initial.textures);
			batch.finish();
		}
		require(ctx.encoding_statistics().render_encoders == owned_encoders + 1 &&
			surfaces.transaction_statistics().submitted == owned_submissions + 1,
			"Independent ready fragment texture split the owned render pass");
		verify(ctx, surfaces, targets.colors[0], {20,40,60,255});


        draw_targets other;other.colors[0]=surfaces.acquire({0xc000,{4,4,1,surface_color::argb8,false},16,{}});
        for(bool precise:{false,true})for(unsigned mode=0;mode<8;++mode)
        {
            surfaces.clear(targets.colors[0],0xf0,0xff112233,0,255,{0,0,4,4});surfaces.clear(other.colors[0],0xf0,0xff112233,0,255,{0,0,4,4});
            const auto generation=surfaces.status(targets.colors[0]).content_generation;
            const auto reached=[ctx.device() newSharedEvent],pending=[ctx.device() newSharedEvent],hold=[ctx.device() newSharedEvent];
            struct release { id<MTLSharedEvent> a,b,c;~release(){a.signaledValue=1;b.signaledValue=1;c.signaledValue=1;} } unblock{reached,pending,hold};
            reached.signaledValue=mode==3||mode==5?0:1;
            query_pool visibility(ctx,1,64,ledger);draw_execution_state state;state.queries=&visibility;
            const auto encoders=ctx.encoding_statistics().render_encoders,submissions=surfaces.transaction_statistics().submitted;
            auto batch=surfaces.batch_draws();
            if(mode==7)surfaces.draw(std::array{cached_surface_use{targets.colors[0],true}},[&](context& commands,auto){commands.wait_event(hold,1);});
            visibility.begin(0,precise);
            for(unsigned i=0;i<32;++i)
            {
                state.waits.clear();
                if(mode!=0&&!(mode==3&&i==0))state.waits.push_back({reached,mode==2?0u:1u});
                if(mode==4)state.waits.push_back({pending,1});
                if((mode==3||mode==5)&&i==1)reached.signaledValue=1;
                executor.execute(owned,mode==6&&i>=8&&i<16?other:targets,initial.textures,state);++completed_draws;
            }
            visibility.end(0);
            const auto expected_encoders=mode==4?32u:mode==5?2u:mode==6?3u:1u;
            require(ctx.encoding_statistics().render_encoders==encoders+expected_encoders,
                "Completed draw waits split compatible native render encoders or lost a required pass boundary");
            require(surfaces.transaction_statistics().submitted==submissions,"Draw wait encoder test submitted before its batch boundary");
            if(mode==4||mode==7)
            {
                const auto recording=ctx.recording_submission();surfaces.submit_draw_batch();bool timed_out=false;
                try{recording.wait(std::chrono::milliseconds(1));}catch(const submission_wait_timeout&){timed_out=true;}
                require(timed_out&&!recording.ready()&&surfaces.status(targets.colors[0]).content_generation==generation,
                    "Draw waits lost their actual queue gate or published target state early");++completed_timeouts;
                if(mode==7)
                {
                    require(recording.cancel_pending(),"Held shared-encoder batch could not be cancelled");bool failed=false;
                    try{batch.finish();}catch(const std::runtime_error&){failed=true;}
                    require(failed&&!surfaces.status(targets.colors[0]).pending&&!surfaces.status(targets.colors[0]).faulted&&surfaces.status(targets.colors[0]).content_generation==generation,
                        "Cancelled shared-encoder batch published an independent prefix or draw output");
                    ctx.recover_completed_errors();visibility.discard(0);verify(ctx,surfaces,targets.colors[0],{17,34,51,255});verify(ctx,surfaces,other.colors[0],{17,34,51,255});++completed_cancels;++completed_profiles;continue;
                }
                pending.signaledValue=1;
            }
            batch.finish();
            require(surfaces.transaction_statistics().submitted==submissions+1,"Completed draw waits split a surface transaction");
            require(visibility.consume(0)==(precise?512u:1u),"Shared encoder with completed draw waits lost exact visibility samples");
            verify(ctx,surfaces,targets.colors[0],{20,40,60,255});verify(ctx,surfaces,other.colors[0],mode==6?std::array{20u,40u,60u,255u}:std::array{17u,34u,51u,255u});++completed_profiles;
        }
        if(completed_waits_only)continue;

		auto image = initial.textures.fragment(0);
		auto texture_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.texture.pixelFormat
			width:image.texture.width height:image.texture.height mipmapped:NO];
		texture_desc.storageMode = MTLStorageModePrivate;
		texture_desc.usage = MTLTextureUsageShaderRead;
		const auto scratch = make_texture(ctx, texture_desc);
		const auto source_texture = image.texture;
		image.texture = scratch;
		const auto dependent_epoch = draw_texture_epoch::capture(ctx, owned.state().vertex_textures(), owned.state().textures(),
			[&](bool, unsigned, const auto&) { return image; });
		{
			auto batch = surfaces.batch_draws();
			executor.execute(owned, targets, initial.textures);
			texture_region full;
			full.size = MTLSizeMake(scratch.width, scratch.height, 1);
			copy_texture(ctx, source_texture, full, scratch, full);
			const auto after_copy = ctx.encoding_statistics().render_encoders;
			executor.execute(owned, targets, dependent_epoch);
			executor.execute(owned, targets, dependent_epoch);
			require(ctx.encoding_statistics().render_encoders == after_copy + 2,
				"A pending fragment texture write must prevent encoder reuse");
			batch.finish();
		}
		verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
		std::cout << "Owned textured pass reuse: 32 draws in one encoder and submission, pending sampled writes split the pass, exact pixels (native source " << native << ")\n";
		auto dependent = request;
		dependent.persistent = make_buffer(ctx, request.persistent.length, buffer_storage::private_gpu);
		{
			auto batch = surfaces.batch_draws();
			executor.execute(draw, targets, request, initial.textures);
			copy_buffer(ctx, request.persistent, 0, dependent.persistent, 0, request.persistent.length);
			const auto before_copy_draw = ctx.encoding_statistics();
			executor.execute(draw, targets, dependent, initial.textures);
			require(ctx.encoding_statistics().guest_draw_full_barriers == before_copy_draw.guest_draw_full_barriers + 1,
				"A vertex input written earlier in the same recording must keep the full barrier");
			++same_batch_waits;
			batch.finish();
		}
		verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
		enum class pending_writer { vertex, unrelated, opaque };
		for (const auto writer : {pending_writer::vertex, pending_writer::unrelated, pending_writer::opaque})
		{
			const auto gate = [ctx.device() newSharedEvent];
			struct release_gate { id<MTLSharedEvent> event; ~release_gate() { event.signaledValue = 1; } } release{gate};
			ctx.begin();
			ctx.wait_event(gate, 1);
			copy_buffer(ctx, request.persistent, 0, dependent.persistent, 0, request.persistent.length);
			if (writer == pending_writer::opaque) (void)ctx.compute();
			const auto producer = ctx.submit();
			{
				auto batch = surfaces.batch_draws();
				const auto before_pending_draw = ctx.encoding_statistics();
				executor.execute(draw, targets, writer == pending_writer::vertex ? dependent : request, initial.textures);
				const auto after_pending_draw = ctx.encoding_statistics();
				const bool producer_pending = !producer.ready();
				gate.signaledValue = 1;
				const bool needs_full = writer != pending_writer::unrelated;
				require(producer_pending && after_pending_draw.guest_draw_full_barriers == before_pending_draw.guest_draw_full_barriers + needs_full &&
					after_pending_draw.guest_draw_fragment_barriers == before_pending_draw.guest_draw_fragment_barriers + !needs_full,
					"Pending checked and opaque producers must select the appropriate guest draw barrier");
				pending_waits += writer == pending_writer::vertex;
				independent_pending += writer == pending_writer::unrelated;
				opaque_pending += writer == pending_writer::opaque;
				batch.finish();
			}
			producer.wait();
			verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
		}
		const auto multisample = surfaces.acquire({0x1a000, {4, 4, 2, surface_color::argb8, false}, 32, {}});
		surfaces.clear(multisample, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
		auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:4 height:4 mipmapped:NO];
		descriptor.textureType = MTLTextureType2DMultisample;
		descriptor.sampleCount = 2;
		descriptor.storageMode = MTLStorageModePrivate;
		descriptor.usage = MTLTextureUsageRenderTarget;
		const auto copied_samples = make_texture(ctx, descriptor);
		ctx.wait_idle();
		auto scissored_registers = registers();
		texture_registers(scissored_registers, 0x100);
		scissored_registers.registers[0x8c0 / 4] = 2u << 16;
		const auto scissored = snapshot(scissored_registers, true, 1);
		enum class prior_pass { load_clear, shader_clear, opaque, multisample, sample_copy };
		for (const auto prior : {prior_pass::load_clear, prior_pass::shader_clear, prior_pass::opaque, prior_pass::multisample, prior_pass::sample_copy})
		{
			const bool scissor = prior != prior_pass::multisample && prior != prior_pass::sample_copy;
			{
				auto batch = surfaces.batch_draws();
				if (prior == prior_pass::opaque)
					surfaces.draw(std::array{cached_surface_use{targets.colors[0], true}}, [&](context& commands, auto images)
					{
						auto pass = [MTL4RenderPassDescriptor new];
						pass.colorAttachments[0].texture = images[0];
						pass.colorAttachments[0].loadAction = MTLLoadActionClear;
						pass.colorAttachments[0].storeAction = MTLStoreActionStore;
						pass.colorAttachments[0].clearColor = MTLClearColorMake(17. / 255, 34. / 255, 51. / 255, 1);
						(void)commands.render(pass);
					});
				else if (prior == prior_pass::sample_copy)
					surfaces.draw(std::array{cached_surface_use{multisample, true}}, [&](context& commands, auto images)
					{
						const texture_region whole{0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)};
						copy_texture(commands, images[0], whole, copied_samples, whole);
					});
				else
					surfaces.clear(prior == prior_pass::multisample ? multisample : targets.colors[0],
						0xf0, 0xff112233, 0, 255, prior == prior_pass::shader_clear ? clear_rectangle{2, 0, 2, 4} : clear_rectangle{0, 0, 4, 4});
				const auto prefix = ctx.recording_submission();
				const auto before_draw = ctx.encoding_statistics();
				executor.execute(scissor ? scissored : draw, targets, request, initial.textures);
				const auto after_draw = ctx.encoding_statistics();
				const bool full = prior == prior_pass::opaque || prior == prior_pass::sample_copy;
				require(prefix.same_batch(ctx.recording_submission()) &&
					after_draw.guest_draw_full_barriers == before_draw.guest_draw_full_barriers + full &&
					after_draw.guest_draw_fragment_barriers == before_draw.guest_draw_fragment_barriers + !full,
					"A preceding clear or opaque pass selected the wrong guest draw barrier");
				load_clears += prior == prior_pass::load_clear;
				shader_clears += prior == prior_pass::shader_clear;
				opaque_passes += prior == prior_pass::opaque;
				multisample_clears += prior == prior_pass::multisample;
				multisample_copies += prior == prior_pass::sample_copy;
				batch.finish();
			}
			verify(ctx, surfaces, targets.colors[0], {20,40,60,255},
				scissor ? std::array<unsigned, 4>{0, 0, 2, 4} : std::array<unsigned, 4>{0, 0, 4, 4});
		}
	}
	std::cout<<"Completed draw encoder waits: "<<completed_profiles<<" VM/native ready-texture profiles, "<<completed_draws<<" translated draws, "<<completed_timeouts<<" actual queue timeouts, "<<completed_cancels<<" cancellations; completed/zero/late signals share encoders, mixed pending waits and attachment changes split passes, precise512/boolean1 visibility and exact target/prefix rollback pixels passed\n";
	if(completed_waits_only)return;
	std::cout<<"Ready productive textures:"<<draws<<" actual translated RSX draws acrossVM/native sources,two32-draw batches/two submissions,all16RGBA targetpixels exact,no newtextureupload/VMcapture passed\n";
	std::cout << "Checked guest draw barriers: " << narrowed << " fragment/tile waits, " << same_batch_waits
		<< " same-recording vertex waits, " << pending_waits << " pending vertex waits, " << independent_pending
		<< " independent pending writes and " << opaque_pending << " opaque pending fallbacks passed\n";
	std::cout << "Render helper write tracking: " << load_clears << " load-clear and " << shader_clears
		<< " shader-clear and " << multisample_clears << " MSAA-clear prefixes kept vertex independent, "
		<< opaque_passes << " opaque and " << multisample_copies << " MSAA-copy prefixes kept full guest draw barriers; exact pixels passed\n";
}
void verify_texture_writer_boundary()
{
	unsigned guards=0;
	for(bool physical:{false,true})
	{
		context ctx(2);auto ledger=std::make_shared<heap_budget>(ctx,16u<<20);ctx.set_buffer_budget(ledger);auto provider=std::make_shared<guarded_gpu_memory>();auto ram=std::make_shared<guarded_guest_memory>(provider);const cpu_capture_dispatch capture=[provider](const auto& work){provider->execute([&](auto&){work();});};auto io=std::make_shared<io_mapping_epoch>();surface_cache surfaces(ctx,ram,16u<<20,ledger,capture);guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return offset;},1u<<20,ledger,64,{capture,io});draw_executor executor(ctx,surfaces,textures);
		auto source=registers();texture_registers(source,physical?0x101000:0x1000);const auto vertex=vertex_texture_snapshot::capture(source.registers,0);const auto fragment=fragment_texture_snapshot::capture(source.registers,1);std::array<std::byte,64> pixels;for(unsigned i=0;i<pixels.size();++i)pixels[i]=std::byte(std::array{255u,20u,40u,60u}[i%4]);ram->write({0x1000,64},pixels);
		const auto old=executor.prepare_versioned_textures(vertex,fragment);const auto target=surfaces.acquire({0x1000,{4,4,1,surface_color::argb8,false},16,{}});surfaces.clear(target,0xf0,0xff112233,0,255,{0,0,4,4});const auto before=surfaces.transaction_statistics();
		{
			auto batch=surfaces.batch_draws();surfaces.clear(target,0xf0,0xff446688,0,255,{0,0,4,4});const auto prefix=ctx.recording_submission();auto next=executor.enqueue_versioned_textures(vertex,fragment);const auto fresh=next.finish();require(prefix.ready()&&surfaces.transaction_statistics().draw_batches==before.draw_batches+1&&fresh.textures.fragment(0).texture!=old.textures.fragment(0).texture,"Texture reused a stale image acrosspending logical/physical writer");batch.finish();
			const auto raw=make_buffer(ctx,2048,buffer_storage::shared);ctx.begin();texture_region area;area.size=MTLSizeMake(4,4,1);texture_to_buffer(ctx,fresh.textures.fragment(0).texture,area,raw,0,256);texture_to_buffer(ctx,old.textures.fragment(0).texture,area,raw,1024,256);ctx.submit().wait();for(unsigned y=0;y<4;++y)for(unsigned x=0;x<16;++x)require(static_cast<const unsigned char*>(raw.contents)[y*256+x]==std::array{136u,102u,68u,255u}[x%4],"Texture missed pending logical/physical writer pixels");for(unsigned y=0;y<4;++y)for(unsigned x=0;x<16;++x)require(static_cast<const unsigned char*>(raw.contents)[1024+y*256+x]==std::array{60u,40u,20u,255u}[x%4],"Pending writer changed retained old immutable pixels");
		}++guards;
	}
	std::cout<<"Texture writer boundaries:"<<guards<<" actual recorded logical/physical-alias writers closeprefix before readylookup,newexactpixels replaceoldimmutableimage passed\n";
}
enum class upload_outcome { success, cancellation, superseded, deferred_source, texture_writer, border_helper,
	inline_color, inline_mixed, inline_cancel, inline_border, inline_superseded, inline_codec };
struct inline_codec_case { unsigned format; bool volume; bool linear; };
enum class inline_cast_source { none, color, depth, redirected_depth };
struct inline_cast_case { inline_cast_source source; bool d16; unsigned samples = 1; bool spilled = false; bool plane_copies = false; };
enum class inline_assembly_case
{
	none, cube, depth_volume, partial_color, partial_morton, partial_banked, partial_morton_depth, partial_banked_depth
};
enum class inline_dependency_case { none, no_prefix, after_prefix, completed_before, completed_after, completed_zero, first_completed, opaque_completed };
enum class draw_framebuffer_case { initialized, cold, spilled, deferred, deferred_reader, stale_resident };
void verify_pending_draw_capture(upload_outcome outcome, inline_codec_case codec = {}, inline_cast_case cast = {},
	inline_assembly_case assembly = inline_assembly_case::none, bool defer_batch_completion = false,
	inline_dependency_case dependency = inline_dependency_case::none, draw_framebuffer_case target_state = draw_framebuffer_case::initialized)
{
	using namespace std::chrono_literals;
	const bool inline_upload = outcome == upload_outcome::inline_color || outcome == upload_outcome::inline_mixed ||
		outcome == upload_outcome::inline_cancel || outcome == upload_outcome::inline_border ||
		outcome == upload_outcome::inline_superseded || outcome == upload_outcome::inline_codec;
	const bool depth_codec = codec.format >= 0x90;
	const bool vtc = codec.format && !depth_codec;
	const bool redirected_depth = cast.source == inline_cast_source::redirected_depth;
	const bool has_cast = cast.source != inline_cast_source::none && !redirected_depth;
	const bool cast_depth_source = cast.source == inline_cast_source::depth || redirected_depth;
	const bool has_assembly = assembly != inline_assembly_case::none;
	const bool cube = assembly == inline_assembly_case::cube;
	const bool depth_assembly = assembly == inline_assembly_case::depth_volume ||
		assembly == inline_assembly_case::partial_morton_depth || assembly == inline_assembly_case::partial_banked_depth;
	const bool morton = assembly == inline_assembly_case::partial_morton || assembly == inline_assembly_case::partial_morton_depth;
	const bool banked = assembly == inline_assembly_case::partial_banked || assembly == inline_assembly_case::partial_banked_depth;
	const bool partial = assembly == inline_assembly_case::partial_color || morton || banked;
	const bool custom_border = outcome == upload_outcome::border_helper || outcome == upload_outcome::inline_border;
	const bool superseded = outcome == upload_outcome::superseded || outcome == upload_outcome::inline_superseded;
	const bool native_source = outcome == upload_outcome::deferred_source || outcome == upload_outcome::inline_mixed;
	const bool upload_wait = dependency != inline_dependency_case::none;
	const bool completed_waits = dependency == inline_dependency_case::completed_before || dependency == inline_dependency_case::completed_after || dependency == inline_dependency_case::completed_zero;
	const bool opaque_wait = dependency == inline_dependency_case::opaque_completed;
	const bool split_prefix = dependency == inline_dependency_case::after_prefix || dependency == inline_dependency_case::first_completed || opaque_wait;
	const bool cold_framebuffer = target_state == draw_framebuffer_case::cold;
	const bool spilled_framebuffer = target_state == draw_framebuffer_case::spilled;
	const bool deferred_framebuffer = target_state == draw_framebuffer_case::deferred;
	const bool hold_reader = target_state == draw_framebuffer_case::deferred_reader;
	const bool reload_resident = target_state == draw_framebuffer_case::stale_resident;
	context ctx(2, 64, 5s, context::recording_overlap::explicit_predecessor);
	auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
	auto provider = std::make_shared<guarded_gpu_memory>();
	provider->provide_storage_identity = partial;
	auto ram = std::make_shared<guarded_guest_memory>(provider); auto io = std::make_shared<io_mapping_epoch>();
	const cpu_capture_dispatch capture = [provider](const auto& work) { provider->execute([&](auto&) { work(); }); };
	surface_cache surfaces(ctx,ram,8u << 20,ledger,capture);
	guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,{capture,io});
	draw_executor executor(ctx,surfaces,textures);
	std::array<std::byte,64> pixels;
	for (unsigned b = 0; b < pixels.size(); ++b) pixels[b] = std::array{std::byte{255},std::byte{20},std::byte{40},std::byte{60}}[b % 4];
	ram->write({0x1000,64},pixels);
	std::vector<std::byte> position_image;
	for (float value : {-1.f, -1.f, .5f, 1.f, 3.f, -1.f, .5f, 1.f, -1.f, 3.f, .5f, 1.f})
		for (unsigned shift : {24u, 16u, 8u, 0u})
			position_image.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
	position_image.resize(64);
	ram->write({0x2000, 64}, position_image);
	const unsigned texture_address = banked ? 0x10000 : 0x1000;
	auto words = registers(); texture_registers(words, texture_address / 16); set_vertex_texture_registers(words, 0x200);
	if (codec.format)
	{
		constexpr unsigned base = 0x1a00 / 4;
		const unsigned depth = codec.volume ? 4 : 1;
		const unsigned width = vtc ? 8 : 4;
		const auto block = describe_guest_texture_format(codec.format).block;
		words.registers[base + 1] = (2u << 16) | ((codec.format | (codec.linear ? 0x20u : 0u)) << 8) |
			((codec.volume ? 3u : 2u) << 4) | 8u | 2u;
		words.registers[base + 6] = (width << 16) | width;
		words.registers[0x1840 / 4] = (depth << 20) | (codec.linear ? ((width + block.width - 1) / block.width) * block.bytes : 0);
		const auto regs = fragment_texture_snapshot::capture(words.registers, 1).at(0);
		const auto plan = plan_guest_texture(regs);
		std::vector<std::byte> bytes(plan.storage_bytes);
		for (const auto& level : plan.subresources)
			for (unsigned z = 0; z < level.depth; ++z)
				for (unsigned y = 0; y < level.rows; ++y)
					for (unsigned x = 0; x < level.columns; ++x)
					{
						const auto offset = guest_texture_block_offset(plan, level, x, y, z);
						if (vtc)
						{
							const unsigned color_offset = block.bytes == 8 ? 0 : 8;
							if (codec.format == 0x87) std::fill_n(bytes.begin() + offset, 8, std::byte{255});
							if (codec.format == 0x88) bytes[offset] = std::byte{255};
							bytes[offset + color_offset + 1] = std::byte{0xf8};
						}
						else
						{
							const unsigned raw = codec.format == 0x90 ? 0x8000002au : codec.format == 0x91 ? 0x7e00002au :
								codec.format == 0x92 ? 0x8000u : 0x6000u;
							for (unsigned c = 0; c < block.bytes; ++c)
								bytes[offset + c] = std::byte(raw >> ((block.bytes - 1 - c) * 8));
						}
					}
		ram->write({0x1000, bytes.size()}, bytes);
	}
	if (has_assembly)
	{
		constexpr unsigned base = 0x1a00 / 4;
		words.registers[base + 1] = depth_assembly ? (2u << 16) | (0xb1u << 8) | (3u << 4) | 8u | 2u :
			(words.registers[base + 1] & 0xffffu) | (2u << 16) | (cube ? 4u : 0u);
		if (depth_assembly) words.registers[0x1840 / 4] = (2u << 20) | 16u;
		const auto plan = plan_guest_texture(fragment_texture_snapshot::capture(words.registers, 1).at(0));
		std::vector<std::byte> bytes(plan.storage_bytes);
		const auto value = depth_assembly ? std::array{0x7du,0u,0u,42u} : std::array{255u,90u,100u,110u};
		for (const auto& level : plan.subresources)
			for (unsigned z = 0; z < level.depth; ++z)
				for (unsigned y = 0; y < level.rows; ++y)
					for (unsigned x = 0; x < level.columns; ++x)
					{
						const auto offset = guest_texture_block_offset(plan, level, x, y, z);
						for (unsigned c = 0; c < 4; ++c) bytes[offset + c] = std::byte(value[c]);
					}
		ram->write({texture_address, bytes.size()}, bytes);
		const surface_format format = depth_assembly ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		cached_surface_description source{texture_address, {partial ? 2u : 4u,4,1,format,false}, morton ? 0u : banked ? 256u : 16u, {}};
		if (morton) source.morton = swizzle_region{{4,4,1}, {}, {2,4,1}};
		if (banked) source.tile = banked_tile_layout{texture_address, 0x10000, 256, 0, 0};
		const auto native = surfaces.acquire(source);
		surfaces.clear(native, depth_assembly ? 3u : 0xf0u, 0xff14283c, 0xffffff35, 255, {0,0,partial ? 2u : 4u,4});
	}
	cached_surface_ref split_target;
	if (has_cast || redirected_depth)
	{
		constexpr unsigned base = 0x1a00 / 4;
		words.registers[base + 1] = (words.registers[base + 1] & ~(0xffu << 8)) |
			((redirected_depth ? 0xa5u : cast.d16 ? 0xb2u : 0xa1u) << 8);
		words.registers[base + 6] = ((cast.d16 ? 8u : 16u) << 16) | 4u;
		const surface_format format = cast_depth_source ? surface_format{surface_depth::z24s8_float} : surface_format{surface_color::argb8};
		const auto width = cast.samples > 1 ? 2u : 4u;
		const auto height = cast.samples == 4 ? 2u : 4u;
		if (cast.samples > 1)
			words.registers[base + 5] = (1u << 24) | (1u << 16);
		const auto native = surfaces.acquire({0x1000, {width,height,cast.samples,format,false}, 16, {}});
		if (cast.plane_copies)
		{
			const auto depth_donor = surfaces.acquire({0x4000, {width,height,cast.samples,format,false}, 16, {}});
			const auto stencil_donor = surfaces.acquire({0x5000, {width,height,cast.samples,format,false}, 16, {}});
			surfaces.clear(depth_donor, 3, 0, 0xffffff07, 255, {0,0,width,height});
			surfaces.clear(stencil_donor, 3, 0, 0x0000002a, 255, {0,0,width,height});
			surfaces.defer_copy(depth_donor, {0,0,width,height}, native, 0, 0, texture_plane::depth);
			surfaces.defer_copy(stencil_donor, {0,0,width,height}, native, 0, 0, texture_plane::stencil);
			split_target = native;
		}
		else surfaces.clear(native, cast_depth_source ? 3u : 0xf0u, 0xff112233, 0xffffff2a, 255, {0,0,width,height});
		if (cast.spilled) surfaces.spill(native);
	}
	if (custom_border)
	{
		words.registers[0x1a00 / 4 + 2] = (7u << 28) | 0x040404u;
		words.registers[0x1a00 / 4 + 7] = 0xff506070u;
	}
	const auto vertex_words = vertex_texture_snapshot::capture(words.registers,1);
	const auto fragment_words = fragment_texture_snapshot::capture(words.registers,1);
	test::framebuffer_layout framebuffer_words;
	framebuffer_words.color_addresses[0] = outcome == upload_outcome::texture_writer ? 0x1000 : 0xa000;
	framebuffer_words.actual_color_pitch[0] = 16;
	framebuffer_words.zeta_address = 0;
	const auto framebuffer = framebuffer_snapshot::capture(framebuffer_words, true);
	const auto targets = acquire_framebuffer(framebuffer, surfaces);
	if (!cold_framebuffer && !deferred_framebuffer && !reload_resident)
		surfaces.clear(targets.colors[0], 0xf0,
			outcome == upload_outcome::texture_writer ? 0xff14283c : 0xff112233, 0, 255, {0, 0, 4, 4});
	if (spilled_framebuffer) surfaces.spill(targets.colors[0]);
	if (reload_resident)
	{
		surfaces.prepare(targets.colors[0]);
		ram->write({framebuffer_words.color_addresses[0], pixels.size()}, pixels);
	}
	if (deferred_framebuffer)
	{
		const auto donor = surfaces.acquire({0x9000, {4,4,1,surface_color::argb8,false}, 16, {}});
		surfaces.clear(donor, 0xf0, 0xff112233, 0, 255, {0,0,4,4});
		surfaces.defer_copy(donor, {0,0,2,4}, targets.colors[0], 0, 0);
	}
	cached_surface_ref reader;
	if (hold_reader)
	{
		reader = surfaces.acquire({0xc000, {4,4,1,surface_color::argb8,false}, 16, {}});
		surfaces.defer_copy(targets.colors[0], {0,0,4,4}, reader, 0, 0);
	}
	const auto original = surfaces.status(targets.colors[0]);
	cached_surface_ref deferred_source;
	if (native_source)
	{
		const surface_allocation_description color{4, 4, 1, surface_color::argb8, false};
		deferred_source = surfaces.acquire({0x1000, color, 16, {}});
		const auto donor = surfaces.acquire({0x3000, color, 16, {}});
		surfaces.clear(donor, 0xf0, 0xff14283c, 0, 255, {0, 0, 4, 4});
		surfaces.defer_copy(donor, {0, 0, 4, 4}, deferred_source, 0, 0);
	}
	const auto gate = [ctx.device() newSharedEvent];
	struct gate_scope { id<MTLSharedEvent> event; unsigned release_value = 1; ~gate_scope() { event.signaledValue = release_value; } } gate_guard{gate, completed_waits || opaque_wait || dependency == inline_dependency_case::first_completed ? 2u : 1u};
	const auto second_gate = upload_wait ? [ctx.device() newSharedEvent] : nil;
	gate_scope second_guard{second_gate};
	const unsigned upload_value = dependency == inline_dependency_case::completed_zero ? 0u : 1u;
	std::vector waits{guest_texture_upload_dependency{gate,upload_value}};
	if (second_gate) waits.push_back({second_gate, upload_value});
	if (dependency == inline_dependency_case::completed_before || opaque_wait) { [gate setSignaledValue:1]; [second_gate setSignaledValue:1]; }
	if (dependency == inline_dependency_case::first_completed) [gate setSignaledValue:1];
    if (opaque_wait)
    {
        const auto event=[ctx.device() newEvent];const auto proof=[ctx.device() newSharedEvent];const auto producer=[ctx.device() newMTL4CommandQueue];
        require(event&&proof&&producer&&![event conformsToProtocol:@protocol(MTLSharedEvent)],"Opaque wait fixture received an observable event");
        [producer signalEvent:event value:1];[producer signalEvent:proof value:1];
        require([proof waitUntilSignaledValue:1 timeoutMS:1000],"Opaque event was not actually signaled by its native Metal4 producer");
        waits.push_back({event,1});
    }
	cached_surface_ref unrelated;
	if (inline_upload)
	{
		unrelated = surfaces.acquire({0xb000, {4,4,1,surface_color::argb8,false}, 16, {}});
		surfaces.clear(unrelated, 0xf0, 0xff112233, 0, 255, {0,0,4,4});
	}
	auto batch = surfaces.batch_draws();
	std::optional<submission> prefix;
	if (inline_upload && dependency != inline_dependency_case::no_prefix)
	{
		surfaces.clear(unrelated, 0xf0, 0xff405060, 0, 255, {0,0,4,4});
		prefix = ctx.recording_submission();
	}
	const auto transactions = surfaces.transaction_statistics();
	const auto before = ctx.encoding_statistics();
	auto job = executor.enqueue_versioned_textures(vertex_words, fragment_words,
		!inline_upload || upload_wait ? std::span<const guest_texture_upload_dependency>{waits} : std::span<const guest_texture_upload_dependency>{},
		nullptr, inline_upload ? texture_upload_path::draw_batch : texture_upload_path::separate);
	waits.clear();
	require(!job.ready() && textures.statistics().entries == 0 && textures.statistics().uploads == 0,
		"Unfinished draw texture epoch published images");
	if (inline_upload)
	{
		require(!job.has_completion() && !job.cancel_pending() && textures.statistics().upload_batches == 0 &&
			ctx.encoding_statistics().compute_encoders == before.compute_encoders &&
			(!prefix || prefix->same_batch(ctx.recording_submission())) &&
			surfaces.transaction_statistics().submitted == transactions.submitted,
			"Inline texture admission split the prefix or encoded GPU work before the draw");
		bool unrecorded = false;
		try { (void)job.finish(1ms); } catch (const std::logic_error&) { unrecorded = true; }
		require(unrecorded && !job.ready(), "Unrecorded draw texture job escaped as a ready image");
	}
	else
	{
		require(textures.statistics().upload_batches == 1 && ctx.encoding_statistics().compute_encoders == before.compute_encoders + 1,
			"Draw enqueue waited before the unsignalled native gate");
		bool timed_out = false;
		try { (void)job.finish(1ms); } catch (const submission_wait_timeout&) { timed_out = true; }
		require(timed_out && !job.ready() && ledger->statistics().reserved_bytes > 0, "Pending draw timeout lost its completion owners");
	}
	std::optional<rsx_draw_packet> packet;
	test::pass_state source; source.registers = words.registers; source.base_index = 0;
	source.transform_constants[511][3] = 0x7fc12345;
	test::pass_processor processor;
	processor.descriptors[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27),(1u << 31) | (1u << 29)};
	for (float value : {.125f, .5f, 0.f, 0.f, .375f, .5f, 0.f, 0.f, .625f, .5f, 0.f, 0.f})
		for (unsigned shift : {24u,16u,8u,0u}) processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
	processor.index_bytes = {std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{0},std::byte{1},std::byte{0},std::byte{0},std::byte{0},std::byte{2}};
	test::pass_layout layout; test::pass_metadata metadata;
	metadata.referenced_textures_mask = 1;
	auto vertex = vertex_program(true); auto fragment_bytes = program(true); test::fragment_program fragment;
	if (custom_border)
	{
		std::vector<std::byte> outside;
		append(outside, std::bit_cast<unsigned>(-.25f));
		std::copy(outside.begin(), outside.end(), fragment_bytes.begin() + 16);
	}
	if (cube)
	{
		std::vector<std::byte> direction;
		for (const float value : {1.f,0.f,0.f,1.f}) append(direction, std::bit_cast<unsigned>(value));
		std::copy(direction.begin(), direction.end(), fragment_bytes.begin() + 16);
	}
	fragment.pointer = fragment_bytes.data(); fragment.ucode_length = fragment_bytes.size(); fragment.ctrl = rsx_fragment_float32;
	if (redirected_depth)
	{
		fragment.texture_state.redirected_textures = 1;
		fragment.texture_state.texture_dimensions = 1; // Unit zero is a 2D descriptor.
		fragment.texture_state.multisampled_textures = cast.samples > 1 ? 1 : 0;
	}
	checked_cpu_capture(capture,[&]
	{
		require(guest_memory_epoch::active() && job.current(vertex_words,fragment_words) && !job.ready(), "Pending draw metadata/receipts require GPU completion or lost the CPU epoch");
		const auto sampler = program_sampler_plan::capture(fragment_words.at(0),job.metadata(false,0));
		require(sampler.dimension == (codec.volume || depth_assembly ? 3 : cube ? 2 : 1) &&
			(redirected_depth ? sampler.format_class == 16 : codec.format || has_cast || has_assembly || sampler.format_class == 1) &&
			sampler.samples == (cast.spilled || cast.plane_copies ? 1 : cast.samples), "Pending draw program metadata differs from its final source");
		bool wait_refused = false;
		try { (void)job.finish(); } catch (const std::logic_error&) { wait_refused = true; }
		require(wait_refused, "Pending draw waited inside real VM exclusion");
		packet.emplace(rsx_draw_packet::capture(source,processor,layout,metadata,vertex,fragment,1,provoking_vertex::last,
			[](const auto&,unsigned first,unsigned count)
			{
				require(first == 0 && count == 3, "Pending packet geometry changed its stream extent");
				return std::pair<std::uint32_t,std::uint32_t>{48,0};
			}));
		bool prewarm_refused = false;
		try { (void)executor.prewarm_stages(*packet, framebuffer, job); } catch (const std::logic_error&) { prewarm_refused = true; }
		require(prewarm_refused && executor.compilation_statistics().admitted == 0, "Shader prewarm crossed VM exclusion");
	});
	require(packet && processor.write_calls == 1 && !guest_memory_epoch::active() && !job.ready() && textures.statistics().entries == 0,
		"Owned program/geometry/stream/constants capture did not complete while the real GPU upload remained pending");
	// Destroy every mutable input after the coherent capture, before upload completion.
	source.registers.assign(source.registers.size(),0xffffffffu); source.transform_constants = {}; source.scale = {100,100,100};
	processor.persistent.assign(48,std::byte{0}); processor.index_bytes.assign(12,std::byte{0});
	vertex.data.assign(vertex.data.size(),0xffffffffu); fragment_bytes.assign(fragment_bytes.size(),std::byte{0xff});
	bool foreign = false;
	std::jthread other([&] { try { (void)job.finish(); } catch (const std::logic_error&) { foreign = true; } }); other.join();
	require(foreign, "Foreign thread consumed a pending draw job");
	auto moved = std::move(job); bool moved_refused = false;
	try { (void)job.ready(); } catch (const std::logic_error&) { moved_refused = true; }
	require(moved_refused && !moved.ready(), "Moved pending draw job retained access or lost its work");
	const auto before_prewarm = ctx.encoding_statistics();
	const auto allocations = ledger->statistics().created;
	auto early_pipeline = executor.prewarm_stages(*packet, framebuffer, moved);
	require(!moved.ready() && executor.compilation_statistics().admitted == 2 + unsigned(early_pipeline.has_value()) &&
		executor.vertex_statistics().prewarmed == 1 && executor.fragment_statistics().prewarmed == 1 &&
		ctx.encoding_statistics().compute_encoders == before_prewarm.compute_encoders &&
		ctx.encoding_statistics().render_encoders == before_prewarm.render_encoders &&
		ledger->statistics().created == allocations && textures.statistics().entries == 0,
		"Early stage prewarm must admit both libraries while textures are pending without encoding or allocating GPU resources");
	if (outcome == upload_outcome::success && !early_pipeline)
	{
		const auto deadline = std::chrono::steady_clock::now() + 3s;
		while (executor.compilation_statistics().completed < 2 && !executor.compilation_statistics().failed &&
			std::chrono::steady_clock::now() < deadline)
			std::this_thread::sleep_for(1ms);
		early_pipeline = executor.prewarm_stages(*packet, framebuffer, moved);
	}
	if (outcome == upload_outcome::success)
		require(early_pipeline && !moved.ready() && executor.pipeline_statistics().prewarmed == 1 &&
			executor.compilation_statistics().admitted == 3,
			"Ready stages did not start the pipeline while the texture upload remained gated");
	draw_execution_state execution;
	if (completed_waits || opaque_wait || dependency == inline_dependency_case::first_completed) execution.waits.push_back({gate, 2});
	else if (inline_upload && !upload_wait) execution.waits.push_back({gate, 1});
	if (dependency == inline_dependency_case::completed_after) { [gate setSignaledValue:1]; [second_gate setSignaledValue:1]; }
	if (split_prefix)
		surfaces.clear(unrelated, 0xf0, 0xff405060, 0, 255, {0,0,4,4});
	executor.execute(*packet, framebuffer, moved, execution);
	if (outcome == upload_outcome::success)
		require(executor.pipeline_statistics().pending_reused == 1 && executor.compilation_statistics().admitted == 3,
			"Draw did not reuse the pipeline started before texture completion");
	const auto draw = ctx.recording_submission();
	if (reader)
		require(surfaces.status(reader).deferred_copies == 1 && !surfaces.status(reader).initialized,
			"Versioned draw materialized a disjoint reader while recording");
	if (cold_framebuffer)
		require(!surfaces.status(targets.colors[0]).initialized,
			"Cold framebuffer published before the translated draw completed");
	if (spilled_framebuffer)
		require(surfaces.status(targets.colors[0]).spilled,
			"Spilled framebuffer published before the translated draw completed");
	if (deferred_framebuffer)
		require(surfaces.status(targets.colors[0]).deferred_copies == 1,
			"Deferred framebuffer published before the translated draw completed");
	if (inline_upload)
	{
		require(moved.has_completion() && (!prefix || (split_prefix ? prefix->ready() && !prefix->same_batch(draw) : prefix->same_batch(draw))) &&
			surfaces.transaction_statistics().submitted == transactions.submitted + unsigned(split_prefix) && textures.statistics().upload_batches == 1,
			completed_waits ? "Completed upload dependency unnecessarily submitted the independent prefix" : "Cold upload and draw did not respect the preceding batch or encoded an extra upload submission");
		const auto conversions = ctx.encoding_statistics().checked_dispatches;
		executor.execute(*packet, framebuffer, moved);
		require(draw.same_batch(ctx.recording_submission()) && ctx.encoding_statistics().checked_dispatches == conversions &&
			textures.statistics().upload_batches == 1,
			"Second draw repeated an upload already encoded in the recording");
	}
	require(!moved.ready() && !draw.ready() && textures.statistics().entries == 0 &&
		(inline_upload ? ctx.encoding_statistics().render_encoders > before_prewarm.render_encoders :
			ctx.encoding_statistics().render_encoders == before_prewarm.render_encoders + 1),
		"Owned draw did not record while the texture upload remained gated");
	if (outcome == upload_outcome::texture_writer)
	{
		gate.signaledValue = completed_waits || opaque_wait || dependency == inline_dependency_case::first_completed ? 2 : 1;
		bool stale = false;
		try { executor.execute(*packet, framebuffer, moved); }
		catch (const texture_revision_conflict&) { stale = true; }
		require(stale && draw.ready() &&
			surfaces.status(targets.colors[0]).content_generation == original.content_generation + 1,
			"Texture reuse did not finish the recorded writer before rejecting its old input");
		auto fresh = executor.enqueue_versioned_textures(vertex_words, fragment_words);
		executor.execute(*packet, framebuffer, fresh);
		batch.finish();
		require(surfaces.status(targets.colors[0]).content_generation == original.content_generation + 2,
			"Fresh capture did not draw after the texture writer boundary");
		verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
		std::cout << "Pending texture draw: recorded source writer closed the batch before reuse and fresh capture sampled 16 exact pixels\n";
		return;
	}
	if (outcome == upload_outcome::border_helper)
	{
		const auto initialized = ctx.encoding_statistics().checked_dispatches;
		require(initialized == before_prewarm.checked_dispatches + 2,
			"Cold border coverage and color conversion were not encoded in the dependent draw");
		executor.execute(*packet, framebuffer, moved);
		require(draw.same_batch(ctx.recording_submission()) && !moved.ready() &&
			ctx.encoding_statistics().checked_dispatches == initialized,
			"Same recording rebuilt its cold border helpers or waited for the upload");
	}
	if (outcome == upload_outcome::success)
	{
		executor.execute(*packet, framebuffer, moved);
		require(draw.same_batch(ctx.recording_submission()) && !moved.ready() && textures.statistics().upload_batches == 1,
			"Consecutive draws did not reuse the same pending upload in one recording");
	}
	if (defer_batch_completion)
	{
		require(surfaces.poll_submitted() && draw.same_batch(ctx.recording_submission()),
			"Completion poll flushed the active recording");
		batch.submit();
		require(!surfaces.poll_submitted() && surfaces.status(targets.colors[0]).pending,
			"Deferred draw scope waited or published work behind the closed event");
		auto reader_words = words;
		texture_registers(reader_words, framebuffer_words.color_addresses[0] / 16);
		const auto reader = fragment_texture_snapshot::capture(reader_words.registers, 1);
		surfaces.set_wait_timeout(1ms);
		bool consumer_waited = false;
		try { (void)executor.enqueue_versioned_textures(vertex_words, reader); }
		catch (const submission_wait_timeout&) { consumer_waited = true; }
		surfaces.set_wait_timeout(5s);
		require(consumer_waited && !surfaces.poll_submitted(),
			"Texture reader crossed a pending draw or retired its work on timeout");
	}
	else surfaces.submit_draw_batch();
	bool draw_timed_out = false;
	try { draw.wait(1ms); } catch (const submission_wait_timeout&) { draw_timed_out = true; }
	require(draw_timed_out && !moved.ready() && surfaces.status(targets.colors[0]).content_generation == original.content_generation,
		"Queued draw executed or published before the texture upload succeeded");

	if (outcome == upload_outcome::inline_cancel)
	{
		require(!moved.cancel_pending() && draw.cancel_pending(), "Texture job could cancel another owner's draw, or draw cancellation failed");
		bool failed = false;
		if (defer_batch_completion)
		{
			try { draw.wait(5s); } catch (const submission_wait_timeout&) { throw; } catch (const std::runtime_error&) {}
			try { (void)surfaces.poll_submitted(); } catch (const std::runtime_error&) { failed = true; }
		}
		else
			try { batch.finish(); } catch (const std::runtime_error&) { failed = true; }
		bool unpublished = false;
		try { (void)moved.finish(); } catch (const std::runtime_error&) { unpublished = true; }
		require(failed && unpublished && textures.statistics().entries == 0 && textures.statistics().cast_entries == 0 &&
			surfaces.snapshot_statistics().created == 0 &&
			surfaces.status(targets.colors[0]).content_generation == original.content_generation,
			"Cancelled recording exposed inline uploads or target pixels");
		ctx.recover_completed_errors();
		verify(ctx, surfaces, targets.colors[0], {17,34,51,255});
		verify(ctx, surfaces, unrelated, split_prefix ? std::array{64u,80u,96u,255u} : std::array{17u,34u,51u,255u});
		if (reader)
		{
			require(surfaces.status(reader).deferred_copies == 1 && !surfaces.status(reader).dirty,
				"Cancelled draw published or lost its disjoint reader");
			verify(ctx, surfaces, reader, {17,34,51,255});
			std::cout << "Draw deferred readers: cancelled batch kept the pending reader and its exact old pixels\n";
		}
		if (upload_wait)
			std::cout << (completed_waits ? "Completed upload waits: cancelled unified batch rolled back independent prefix and published no images\n" : "Draw-owned upload waits: cancelled dependent draw preserved completed prefix and published no images\n");
		if (defer_batch_completion)
			std::cout << "Deferred draw retirement: cancellation rolled back both surfaces and kept textures unpublished\n";
		std::cout << (has_assembly ? "Inline assembly draw: " : has_cast ? "Inline cast draw: " : "Inline texture draw: ")
			<< "cancellation retained both target and prefix pixels and published no images\n";
		return;
	}
	if (outcome == upload_outcome::cancellation)
	{
		require(moved.cancel_pending(), "Could not cancel the still-gated texture producer");
		bool failed = false;
		try { batch.finish(); } catch (const std::runtime_error&) { failed = true; }
		require(failed && draw.cancelled() && textures.statistics().entries == 0 &&
			surfaces.status(targets.colors[0]).content_generation == original.content_generation &&
			surfaces.status(targets.colors[0]).native_write_epoch == original.native_write_epoch,
			"Failed texture producer allowed the draw to commit or changed prior target authority");
		ctx.recover_completed_errors();
		verify(ctx, surfaces, targets.colors[0], {17,34,51,255});
		std::cout << "Pending texture draw: cancelled producer blocked native commit and preserved 16 prior pixels and target authority\n";
		return;
	}
	if (superseded)
	{
		pixels.fill(std::byte{99});
		ram->write({0x1000,64}, pixels);
	}
	gate.signaledValue = completed_waits || opaque_wait || dependency == inline_dependency_case::first_completed ? 2 : 1;
	if (second_gate && !completed_waits && !opaque_wait)
	{
		bool waits_for_second = false;
		try { draw.wait(1ms); } catch (const submission_wait_timeout&) { waits_for_second = true; }
		require(waits_for_second && !moved.ready(), "Draw committed before the second upload dependency");
		second_gate.signaledValue = 1;
	}
	if (defer_batch_completion)
	{
		draw.wait(5s);
		require(surfaces.status(targets.colors[0]).content_generation == original.content_generation &&
			textures.statistics().entries == 0 && surfaces.poll_submitted() &&
			surfaces.status(targets.colors[0]).content_generation == original.content_generation + 2,
			"Submitted draw changed authority before its owner retired the completion");
		auto next = surfaces.batch_draws();
		next.submit();
	}
	else batch.finish();
	if (superseded)
	{
		bool stale = false;
		try { (void)moved.finish(); } catch (const texture_revision_conflict&) { stale = true; }
		require(stale && textures.statistics().entries == 0 && textures.statistics().uploads == 0,
			"Superseded texture was published to the live cache after the draw");
		verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
		if (defer_batch_completion)
			std::cout << "Deferred draw retirement: VM revision suppressed texture publication and frozen draw pixels survived\n";
		std::cout << (inline_upload ? "Inline texture draw: " : "Pending texture draw: ")
			<< "later VM write kept the live cache clean and the recorded draw sampled 16 frozen pixels\n";
		return;
	}
	const auto epoch = moved.finish();
	checked_cpu_capture(capture,[&] { require(epoch.current(packet->state().vertex_textures(),packet->state().textures()), "Completed draw epoch differs from the independently owned packet"); });
	const auto repeated = moved.finish();
	if (inline_upload)
	{
		const unsigned vm_images = native_source || has_cast || has_assembly || redirected_depth ? 1u : 2u;
		require(surfaces.transaction_statistics().submitted == transactions.submitted + 1 + unsigned(split_prefix) &&
			textures.statistics().uploads == vm_images && textures.statistics().entries == vm_images + unsigned(has_assembly) &&
			textures.statistics().casts_created == unsigned(has_cast) &&
			repeated.textures.fragment(0).texture == epoch.textures.fragment(0).texture &&
			(!native_source || surfaces.status(deferred_source).deferred_copies == 1),
			"Inline recording duplicated GPU work, published a wrong epoch or materialized its deferred source");
		const unsigned cast_sample = cast_depth_source ? 0u : 34u;
		const auto expected = redirected_depth ? std::array{0u,0u,42u,127u} :
			has_cast ? std::array{cast_sample,cast_sample,cast_sample,cast.d16 ? cast_sample : 255u} :
			depth_assembly ? std::array{255u,255u,255u,255u} : depth_codec ? std::array{128u,128u,128u,128u} : vtc ? std::array{255u,0u,0u,255u} :
			custom_border ? std::array{80u,96u,112u,255u} : std::array{20u,40u,60u,255u};
		verify(ctx, surfaces, targets.colors[0], expected);
		verify(ctx, surfaces, unrelated, prefix ? std::array{64u,80u,96u,255u} : std::array{17u,34u,51u,255u});
		if (redirected_depth)
		{
			const auto& image = epoch.textures.fragment(0);
			auto warm = executor.enqueue_versioned_textures(vertex_words, fragment_words, {}, nullptr, texture_upload_path::draw_batch);
			require(image.texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 && image.metadata.depth &&
				image.texture.sampleCount == (cast.spilled || cast.plane_copies ? 1 : cast.samples) &&
				bool(image.metadata.coordinates.flags & texture_coordinate_sample_raster) == (cast.spilled || cast.plane_copies) &&
				warm.ready() && warm.finish().textures.fragment(0).texture == image.texture,
				"Draw executor did not reuse the original native depth/stencil snapshot");
			if (split_target)
				require(!surfaces.status(split_target).initialized && surfaces.status(split_target).deferred_copies == 2,
					"Draw executor materialized the independent depth/stencil copy target");
			std::cout << (cast.plane_copies ? "Inline independent D24S8 planes: " :
				cast.spilled ? "Inline redirected D24S8 sample raster: " : "Inline redirected D24S8 draw: ")
				<< "one submission, 16 exact pixels, original depth/stencil snapshot from "
				<< cast.samples << " sample(s), no cast\n";
			return;
		}
		if (reader)
		{
			require(surfaces.status(reader).deferred_copies == 1 && !surfaces.status(reader).dirty,
				"Successful draw published its disjoint reader");
			verify(ctx, surfaces, reader, {17,34,51,255});
			std::cout << "Draw deferred readers: one submission for prefix, upload and two draws; reader stayed deferred with exact old pixels\n";
		}
		if (upload_wait)
		{
			const std::array warm_waits{guest_texture_upload_dependency{second_gate, 2}};
			const auto warm = executor.enqueue_versioned_textures(vertex_words, fragment_words, warm_waits, nullptr, texture_upload_path::draw_batch);
			require(warm.ready() && textures.statistics().upload_batches == 1,
				"An upload dependency gated an already resident texture");
			std::cout << (completed_waits ? "Completed upload waits: independent prefix retained in upload/draw batch; " : "Draw-owned upload waits: ") << (split_prefix ? "independent prefix completed; " : "")
				<< "separate draw gate, exact pixels and warm lookup passed\n";
		}
		if (defer_batch_completion)
			std::cout << "Deferred draw retirement: unsignalled submit and poll returned, texture reader waited, two draws and prefix published exact pixels\n";
		if (cold_framebuffer)
		{
			require(surfaces.status(targets.colors[0]).content_generation == 3,
				"Cold framebuffer and both translated draws did not publish in order");
			std::cout << "Cold framebuffer: target and sampled textures uploaded in one batch with two translated draws and exact pixels\n";
		}
		if (spilled_framebuffer)
		{
			const auto completed = surfaces.status(targets.colors[0]);
			require(!completed.spilled && completed.content_generation == original.content_generation + 2,
				"Spilled framebuffer was not retired together with both translated draws");
			std::cout << "Spilled framebuffer: target restored and sampled textures uploaded in one batch with two translated draws and exact pixels\n";
		}
		if (deferred_framebuffer)
		{
			const auto completed = surfaces.status(targets.colors[0]);
			require(completed.initialized && completed.deferred_copies == 0 && completed.content_generation == 4,
				"Deferred framebuffer was not retired together with both translated draws");
			std::cout << "Deferred framebuffer: target copies, cold texture uploads and two translated draws shared one submission with exact pixels\n";
		}
		if (reload_resident)
		{
			require(surfaces.status(targets.colors[0]).content_generation == original.content_generation + 3,
				"Resident framebuffer reload or translated draws published the wrong generation count");
			std::cout << "Resident framebuffer: new guest revision and cold sampled textures shared one submission with two translated draws\n";
		}
		if (has_assembly)
		{
			const auto image = epoch.textures.fragment(0).texture;
			const auto warm = executor.enqueue_versioned_textures(vertex_words, fragment_words, {}, nullptr, texture_upload_path::draw_batch);
			require(warm.ready() && warm.metadata(false, 0).actual_mips == 2 &&
				textures.statistics().assemblies_reused == 1u + unsigned(upload_wait),
				"Unrelated draw epoch prevented publication of the finished assembly");
			const auto raw = make_buffer(ctx, 512, buffer_storage::shared);
			texture_region first, second;
			second.slice = cube ? 1 : 0;
			second.origin.z = depth_assembly && !partial ? 1 : 0;
			second.origin.x = partial ? 3 : 0;
			second.origin.y = morton ? 1 : 0;
			const auto stride = depth_assembly ? 8u : 4u;
			ctx.begin();
			texture_to_buffer(ctx, image, first, raw, 0, stride);
			texture_to_buffer(ctx, image, second, raw, 256, stride);
			ctx.submit().wait();
			if (depth_assembly)
			{
				const std::array<float,2> native{1.f,53.f}, vm{.25f,42.f};
				require(!std::memcmp(raw.contents, native.data(), sizeof(native)) &&
					!std::memcmp(static_cast<const std::byte*>(raw.contents) + 256, vm.data(), sizeof(vm)),
					"Inline assembly lost exact native depth or VM stencil");
			}
			else
			{
				const std::array<std::uint8_t,4> native{60,40,20,255}, vm{110,100,90,255};
				require(!std::memcmp(raw.contents, native.data(), sizeof(native)) &&
					!std::memcmp(static_cast<const std::byte*>(raw.contents) + 256, vm.data(), sizeof(vm)),
					"Inline assembly did not preserve exact native and VM color bytes");
			}
			if (cube)
			{
				const auto plan = plan_guest_texture(fragment_words.at(0));
				const auto& face = *std::ranges::find_if(plan.subresources, [](const auto& level) { return level.face == 1 && level.level == 0; });
				const auto newer = surfaces.acquire({static_cast<std::uint32_t>(0x1000 + face.offset),
					{4,4,1,surface_color::argb8,false}, face.row_pitch, {}});
				surfaces.clear(newer, 0xf0, 0xff778899, 0, 255, {0,0,4,4});
				const auto refresh = executor.enqueue_versioned_textures(vertex_words, fragment_words, {}, nullptr, texture_upload_path::draw_batch);
				require(!refresh.ready() && !refresh.has_completion(),
					"Native writer in a former VM face left the stale assembly reusable");
			}
			std::cout << "Inline assembly draw: " << (morton ? "partial Morton" : banked ? "partial banked" :
				depth_assembly ? "depth volume" : cube ? "cube" : "partial color")
				<< (partial && depth_assembly ? " depth/stencil" : "")
				<< "; one submission, exact draw pixels and native/VM bytes, warm reuse\n";
			return;
		}
		if (has_cast)
		{
			const auto texture = epoch.textures.fragment(0).texture;
			const auto output = make_buffer(ctx, 1024, buffer_storage::shared);
			texture_region region;
			region.size = MTLSizeMake(texture.width, 4, 1);
			ctx.begin();
			texture_to_buffer(ctx, texture, region, output, 0, 256, 0, cast.d16 ? texture_plane::depth : texture_plane::all);
			ctx.submit().wait();
			const auto packed = cast_depth_source ? std::array{0x7fu,0u,0u,0x2au} : std::array{0xffu,0x11u,0x22u,0x33u};
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 16; ++x)
					require(static_cast<const unsigned char*>(output.contents)[y * 256 + x] == packed[(cast.d16 ? x ^ 1u : x) % 4],
						"Inline native cast changed guest-order or target-endian bytes");
			std::cout << "Inline cast draw: source " << (cast_depth_source ? "depth/stencil" : "color") << ", target "
				<< (cast.d16 ? "D16" : "R8") << "; one submission, exact draw pixels and 64 raw bytes\n";
			return;
		}
		if (codec.format)
		{
			const auto texture = epoch.textures.fragment(0).texture;
			const auto output = make_buffer(ctx, 256, buffer_storage::shared);
			const bool stencil = codec.format == 0x90 || codec.format == 0x91;
			const float expected_depth = codec.format == 0x90 ? float(0x800000) / 16777215.f :
				codec.format == 0x92 ? 32768.f / 65535.f : .5f;
			for (unsigned mip = 0; mip < 2; ++mip)
			{
				texture_region region;
				region.level = mip;
				region.origin.z = codec.volume ? (4u >> mip) - 1 : 0;
				const auto plane = codec.volume ? texture_plane::all : texture_plane::depth;
				const auto pitch = vtc ? 16u : codec.volume ? 8u : codec.format == 0x92 ? 2u : 4u;
				ctx.begin();
				texture_to_buffer(ctx, texture, region, output, 0, pitch, 0, plane);
				if (!codec.volume && stencil)
					texture_to_buffer(ctx, texture, region, output, 64, 1, 0, texture_plane::stencil);
				ctx.submit().wait();
				if (vtc)
				{
					const std::array<float, 4> red{1, 0, 0, 1};
					require(!std::memcmp(output.contents, red.data(), sizeof(red)), "Inline VTC mip readback differs");
				}
				else if (!codec.volume && codec.format == 0x92)
					require(*static_cast<const std::uint16_t*>(output.contents) == 0x8000u, "Inline D16 mip bits differ");
				else
				{
					const auto* texels = static_cast<const float*>(output.contents);
					require(texels[0] == expected_depth, "Inline depth mip bits differ");
					if (codec.volume)
						require(texels[1] == (stencil ? 42.f : 0.f), "Inline volume stencil differs");
					else if (stencil)
						require(static_cast<const std::byte*>(output.contents)[64] == std::byte{42}, "Inline stencil plane differs");
				}
			}
			std::cout << "Inline depth/VTC draw: format " << codec.format << ", volume " << codec.volume <<
				", linear " << codec.linear << "; one submission, exact draw pixels and two raw mip checks\n";
			return;
		}
		if (!upload_wait)
			std::cout << (native_source ? "Inline mixed texture draw: " : custom_border ? "Inline border texture draw: " : "Inline VM texture draw: ")
				<< "one submission for uploads, unrelated prefix and two draws; 32 exact pixels\n";
		return;
	}
	if (outcome == upload_outcome::border_helper)
	{
		verify(ctx, surfaces, targets.colors[0], {80,96,112,255});
		std::cout << "Cold border draw: two helpers and two translated draws shared one recording behind a gated texture upload, 16 exact custom-border pixels\n";
		return;
	}
	if (outcome == upload_outcome::deferred_source)
	{
		require(surfaces.status(deferred_source).deferred_copies == 1 &&
			textures.statistics().entries == 1 && repeated.textures.fragment(0).texture == epoch.textures.fragment(0).texture,
			"Draw materialized the deferred target or lost its published native snapshot");
		verify(ctx, surfaces, targets.colors[0], {20,40,60,255});
		std::cout << "Pending texture draw: gated native snapshot sampled 16 exact pixels and left its source deferred\n";
		return;
	}
	require(textures.statistics().uploads == 2 && textures.statistics().entries == 2 && repeated.textures.fragment(0).texture == epoch.textures.fragment(0).texture &&
		repeated.textures.vertex(0).texture == epoch.textures.vertex(0).texture,
		"Repeated completion published the same pending draw twice");
	verify(ctx,surfaces,targets.colors[0],{20,40,60,255});
	require(textures.statistics().uploads == 2 && executor.compilation_statistics().admitted == 3 &&
		executor.vertex_statistics().prewarm_reused >= 1 && executor.fragment_statistics().prewarm_reused >= 1,
		"The draws must reuse both early library jobs and admit only one pipeline job");
	std::cout << "Pending texture draw: two draws shared one recording before the gated vertex/fragment upload completed, 16 exact pixels, one PSO, and two cached images\n";
	std::cout << "Early pipeline: ready shader stages started one PSO behind the texture gate; draw reused the preparation and all 16 pixels were exact\n";
}
void verify_pending_texture_draws()
{
	for (const auto outcome : {upload_outcome::success, upload_outcome::cancellation, upload_outcome::superseded,
		upload_outcome::deferred_source, upload_outcome::texture_writer, upload_outcome::border_helper})
		verify_pending_draw_capture(outcome);
}
void verify_inline_texture_draws()
{
	for (const auto outcome : {upload_outcome::inline_color, upload_outcome::inline_mixed, upload_outcome::inline_cancel,
		upload_outcome::inline_border, upload_outcome::inline_superseded})
		verify_pending_draw_capture(outcome);
}
void verify_inline_depth_and_vtc()
{
	for (const bool linear : {false, true})
	{
		for (const auto format : {0x90u, 0x91u, 0x92u, 0x93u})
			for (const bool volume : {false, true})
				verify_pending_draw_capture(upload_outcome::inline_codec, {format, volume, linear});
		for (const auto format : {0x86u, 0x87u, 0x88u})
			verify_pending_draw_capture(upload_outcome::inline_codec, {format, true, linear});
	}
}
void verify_inline_native_casts()
{
	for (const auto source : {inline_cast_source::color, inline_cast_source::depth})
		for (const bool d16 : {false, true})
			verify_pending_draw_capture(upload_outcome::inline_color, {}, {source, d16});
	verify_pending_draw_capture(upload_outcome::inline_cancel, {}, {inline_cast_source::depth, true});
}
void verify_inline_assemblies()
{
	for (const auto profile : {inline_assembly_case::cube, inline_assembly_case::depth_volume, inline_assembly_case::partial_color,
		inline_assembly_case::partial_morton, inline_assembly_case::partial_banked,
		inline_assembly_case::partial_morton_depth, inline_assembly_case::partial_banked_depth})
		verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, profile);
	for (const auto profile : {inline_assembly_case::partial_color, inline_assembly_case::partial_morton,
		inline_assembly_case::partial_banked_depth})
		verify_pending_draw_capture(upload_outcome::inline_cancel, {}, {}, profile);
}
void verify_deferred_draw_retirement()
{
	for (const auto outcome : {upload_outcome::inline_color, upload_outcome::inline_superseded, upload_outcome::inline_cancel})
		verify_pending_draw_capture(outcome, {}, {}, inline_assembly_case::none, true);
}
void verify_submitted_texture_independence()
{
	using namespace std::chrono_literals;
	context ctx(2);
	auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
	ctx.set_buffer_budget(budget);
	auto provider = std::make_shared<test::host_access_fixture>();
	provider->provide_storage_identity = true;
	provider->pin_first_alias();
	auto ram = std::make_shared<guarded_guest_memory>(provider);
	const cpu_capture_dispatch capture = [provider](const auto& work) { provider->execute([&](auto&) { work(); }); };
	surface_cache surfaces(ctx, ram, 8 << 20, budget, capture);
	guest_texture_cache textures(ctx, ram, [](unsigned offset, unsigned, std::size_t) { return offset * 16; },
		1 << 20, budget, 8, {capture, std::make_shared<io_mapping_epoch>()});
	draw_executor executor(ctx, surfaces, textures);
	const auto enqueue = [&](unsigned address, texture_upload_path path = texture_upload_path::draw_batch)
	{
		auto words = registers();
		texture_registers(words, address / 16);
		return executor.enqueue_versioned_textures(vertex_texture_snapshot::capture(words.registers, 0),
			fragment_texture_snapshot::capture(words.registers, 1), {}, nullptr, path);
	};
	auto words = registers();
	texture_registers(words, 0x11000 / 16);
	const auto warm = executor.prepare_versioned_textures(vertex_texture_snapshot::capture(words.registers, 0),
		fragment_texture_snapshot::capture(words.registers, 1));
	std::array<std::byte, 64> vm_pixels;
	const std::array vm_pixel{std::byte{0xff}, std::byte{0x44}, std::byte{0x55}, std::byte{0x66}};
	for (std::size_t i = 0; i < vm_pixels.size(); ++i) vm_pixels[i] = vm_pixel[i % vm_pixel.size()];
	for (const auto address : {0x13000u, 0x14000u})
	{
		ram->write({address, vm_pixels.size()}, vm_pixels);
		const auto native = surfaces.acquire({address, {2,4,1,surface_color::argb8,false}, 16, {}});
		surfaces.clear(native, 0xf0, 0xff112233, 0, 255, {0,0,2,4});
	}
	auto first_assembly = enqueue(0x13000, texture_upload_path::separate);
	const auto warm_assembly = first_assembly.finish().textures.fragment(0).texture;
	const auto warm_stats = textures.statistics();
	require(warm_stats.assemblies_created == 1, "Partial native texture did not create a mixed assembly");
	const auto target = surfaces.acquire({0x3000, {4,4,1,surface_color::argb8,false}, 16, {}});
	const auto wide_alias = surfaces.acquire({0x3000, {4,8,1,surface_color::argb8,false}, 16, {}});
	const auto gate = [ctx.device() newSharedEvent];
	struct release_gate { id<MTLSharedEvent> gate; ~release_gate() { gate.signaledValue = 1; } } release{gate};
	auto batch = surfaces.batch_draws();
	const std::array uses{cached_surface_use{target, true}};
	surfaces.draw(uses, [&](context& commands, auto) { commands.wait_event(gate, 1); });
	const auto work = ctx.recording_submission();
	surfaces.clear(target, 0xf0, 0xff234567, 0, 255, {0,0,4,4});
	batch.submit();
	surfaces.set_wait_timeout(1ms);
	auto independent = enqueue(0x11000);
	const auto cold = enqueue(0x12000);
	require(independent.ready() && independent.finish().textures.fragment(0).texture == warm.textures.fragment(0).texture &&
		!cold.ready() && !surfaces.poll_submitted(),
		"Independent warm or cold texture preparation waited for the gated clear");
	auto ready_assembly = enqueue(0x13000);
	const auto inline_assembly = enqueue(0x14000);
	require(ready_assembly.ready() && ready_assembly.finish().textures.fragment(0).texture == warm_assembly &&
		!inline_assembly.ready() && textures.statistics().assemblies_created == warm_stats.assemblies_created + 1 &&
		!surfaces.poll_submitted(), "Independent warm or cold mixed assembly waited for the gated clear");
	auto cold_assembly = enqueue(0x14000, texture_upload_path::separate);
	require(!cold_assembly.ready() && !surfaces.poll_submitted(), "Mixed assembly published while the GPU was gated");
	const auto waits = [&](unsigned address)
	{
		try { (void)enqueue(address); } catch (const submission_wait_timeout&) { return true; }
		return false;
	};
	require(waits(0x3000) && waits(0x3040), "Pending clear exposed a target or its wider alias");
	provider->execute([&](auto&)
	{
		ram->begin_mapping_change({0, 65536});
		provider->replace_first();
		ram->end_mapping_change({0, 65536});
	});
	require(waits(0x43000) && enqueue(0x11000).ready() && !surfaces.poll_submitted(),
		"Retained physical alias escaped the wait or forced an independent texture to wait");
	surfaces.set_wait_timeout(5s);
	gate.signaledValue = 1;
	work.wait(5s);
	require(surfaces.poll_submitted() && surfaces.status(target).dirty && !surfaces.status(wide_alias).initialized,
		"Clear after independent texture preparation did not publish in order");
	const auto image = cold_assembly.finish().textures.fragment(0).texture;
	const auto assembly_bytes = make_buffer(ctx, 4 * 256, buffer_storage::shared);
	ctx.begin();
	texture_to_buffer(ctx, image, {0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)}, assembly_bytes, 0, 256);
	ctx.submit().wait(5s);
	const auto* pixels = static_cast<const unsigned char*>(assembly_bytes.contents);
	for (unsigned y = 0; y < 4; ++y)
		for (unsigned x = 0; x < 4; ++x)
		{
			const auto expected = x < 2 ? std::array{0x33u,0x22u,0x11u,0xffu} : std::array{0x66u,0x55u,0x44u,0xffu};
			for (unsigned c = 0; c < 4; ++c)
				require(pixels[y * 256 + x * 4 + c] == expected[c], "Mixed assembly changed native or guest pixels");
		}
	std::cout << "Submitted mixed assembly: warm reuse and cold native/VM preparation returned behind the gated clear; all 16 pixels exact\n";
	surfaces.flush(target);
	const auto result = ram->read({0x43000, 64}).bytes;
	const std::array pixel{std::byte{0xff}, std::byte{0x23}, std::byte{0x45}, std::byte{0x67}};
	for (std::size_t i = 0; i < result.size(); ++i)
		require(result[i] == pixel[i % pixel.size()], "Retained physical clear returned incorrect bytes");
	std::cout << "Submitted texture independence: warm and cold disjoint preparation returned; target, wider alias and remapped physical alias waited; exact clear bytes passed\n";
}
void verify_inline_upload_dependencies()
{
	verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, inline_assembly_case::none, false, inline_dependency_case::no_prefix);
	verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, inline_assembly_case::partial_morton_depth, false, inline_dependency_case::after_prefix);
	verify_pending_draw_capture(upload_outcome::inline_cancel, {}, {}, inline_assembly_case::none, false, inline_dependency_case::after_prefix);
}
void verify_completed_inline_upload_dependencies()
{
    unsigned cases=0;
    for(const auto dependency:{inline_dependency_case::completed_before,inline_dependency_case::completed_after,inline_dependency_case::completed_zero})
        for(const auto outcome:{upload_outcome::inline_color,upload_outcome::inline_cancel})
        {
            verify_pending_draw_capture(outcome,{}, {},dependency==inline_dependency_case::completed_after?inline_assembly_case::partial_morton_depth:inline_assembly_case::none,false,dependency);
            ++cases;
        }
    verify_pending_draw_capture(upload_outcome::inline_color,{}, {},inline_assembly_case::none,false,inline_dependency_case::first_completed);++cases;
    for(const auto outcome:{upload_outcome::inline_color,upload_outcome::inline_cancel})
    {verify_pending_draw_capture(outcome,{}, {},inline_assembly_case::none,false,inline_dependency_case::opaque_completed);++cases;}
    std::cout<<"Completed inline upload dependencies: "<<cases<<" cold VM/mixed Morton-depth cases; before-enqueue/after-enqueue/zero-value completion preserves one prefix/upload/draw transaction, queued cancellation rolls both surfaces back, mixed ready/pending and GPU-completed opaque events split prefix; original GPU waits and exact pixels preserved\n";
}

void verify_cold_framebuffer()
{
	verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, inline_assembly_case::none,
		false, inline_dependency_case::none, draw_framebuffer_case::cold);
}
void verify_spilled_framebuffer()
{
	verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, inline_assembly_case::none,
		false, inline_dependency_case::none, draw_framebuffer_case::spilled);
}
void verify_deferred_framebuffer()
{
	verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, inline_assembly_case::none,
		false, inline_dependency_case::none, draw_framebuffer_case::deferred);
}
void verify_draw_deferred_readers()
{
	for (const auto outcome : {upload_outcome::inline_color, upload_outcome::inline_cancel})
		verify_pending_draw_capture(outcome, {}, {}, inline_assembly_case::none,
			false, inline_dependency_case::none, draw_framebuffer_case::deferred_reader);
}
void verify_resident_framebuffer()
{
	verify_pending_draw_capture(upload_outcome::inline_color, {}, {}, inline_assembly_case::none,
		false, inline_dependency_case::none, draw_framebuffer_case::stale_resident);
}
void verify_versioned_job_boundaries()
{
	using namespace std::chrono_literals;
	for (unsigned mode = 0; mode < 7; ++mode)
	{
		context ctx(2,2,mode == 5 ? 100ms : 5s);
		auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		{
			@autoreleasepool
			{
				auto ram = std::make_shared<memory>(); auto io = std::make_shared<io_mapping_epoch>();
				const cpu_capture_dispatch capture = [](const auto& work) { work(); };
				surface_cache surfaces(ctx,ram,8u << 20,ledger,capture);
				guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,{capture,io});
				auto source = registers(); texture_registers(source,0x100);
				auto first = fragment_texture_snapshot::capture(source.registers,1).at(0);
				auto second = first; second.offset += 0x10;
				const auto gate = [ctx.device() newSharedEvent];
				struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } gate_guard{gate};
				const std::array waits{guest_texture_upload_dependency{gate,1}};
				auto job = textures.prepare_versioned_batch(std::array{first,second,first},surfaces,waits);
				require(!job.ready() && job.current() && textures.statistics().entries == 0 && ram->reads == 2,
					"Asynchronous versioned preparation waited,duplicated or prematurely published VM images");
				if (mode == 0) ram->write({0x1000,1},std::array{std::byte{0xab}});
				if (mode == 1 || mode == 2) { io->notify({mode == 1 ? 0u : 1u << 20,1u << 20}); io->notify({mode == 1 ? 0u : 1u << 20,1u << 20}); }
				if (mode == 3) textures.clear();
				if (mode == 4) require(job.cancel_pending(), "Queued versioned-job cancellation refused");
				if (mode == 6) ctx.shutdown();
				if (mode < 4) gate.signaledValue = 1;
				bool failed = false;
				try
				{
					const auto images = job.finish(5s);
					require((mode == 2 || mode == 3) && images.size() == 3 && images[0].image.texture == images[2].image.texture,
						"Changed/failed versioned job exposed images or dropped a coalesced request");
				}
				catch (const texture_revision_conflict&) { failed = mode < 2; }
				catch (const std::runtime_error& error)
				{
					const std::string message = error.what();
					failed = mode == 5 ? message.find("deadline expired") != std::string::npos :
						(mode == 4 || mode == 6) && message.find("cancelled") != std::string::npos;
				}
				require(failed == (mode != 2 && mode != 3) && textures.statistics().entries == (mode == 2 ? 2 : 0) &&
					textures.statistics().uploads == (mode == 2 ? 2 : 0), "Versioned job failure/clear published a partial or stale cache group");
				if (mode >= 4) ctx.recover_completed_errors();
				if (mode < 4)
				{
					const auto fresh = textures.resolve_versioned_batch(std::array{first,second},surfaces);
					require(fresh.size() == 2 && textures.statistics().entries >= 2, "Versioned job conflict/clear prevented a fresh grouped retry");
				}
				textures.clear(); (void)ctx.trim_upload_pages();
			}
		}
		require(ledger->statistics().reserved_bytes == 0, "Versioned job mutation/cancel/deadline/shutdown retained orphan native charges");
	}
	{
		context ctx(2); auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,ledger);
		guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger);
		std::vector<versioned_guest_texture_job> jobs; jobs.reserve(65);
		for (unsigned i = 0; i < 64; ++i) jobs.push_back(textures.prepare_versioned_batch({},surfaces));
		bool bounded = false;
		try { (void)textures.prepare_versioned_batch({},surfaces); } catch (const std::length_error&) { bounded = true; }
		require(bounded && ram->reads == 0 && ledger->statistics().created == 0, "Versioned-job admission bound ran after VM/GPU work");
		require(jobs[0].finish().empty(), "Empty versioned job produced images");
		jobs.push_back(textures.prepare_versioned_batch({},surfaces));
		jobs.erase(jobs.begin() + 1);
		for (unsigned i = 0; i < 128; ++i) require(textures.resolve_versioned_batch({},surfaces).empty(), "Consumed jobs retained their admission slots");
	}
	std::cout << "Native versioned jobs:late VM write/same-EA IO ABA/cancel/deadline/shutdown expose no images or cache entries;unrelated IO accepted,clear generation prevents repopulation,fresh retries work,64unconsumed bound before VM/allocation andall charges retire passed\n";
}
void verify_versioned_job_callback_ownership()
{
	context ctx(2); auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
	{
		@autoreleasepool
		{
			auto ram = std::make_shared<memory>(); auto io = std::make_shared<io_mapping_epoch>();
			std::unique_ptr<guest_texture_cache> textures;
			std::optional<versioned_guest_texture_job> job;
			unsigned phase = 0, refused = 0;
			surface_cache surfaces(ctx,ram,8u << 20,ledger);
			const cpu_capture_dispatch capture = [&](const auto& work)
			{
				if (phase == 1)
				{
					phase = 0;
					try { (void)textures->prepare_versioned_batch({},surfaces); } catch (const std::logic_error&) { ++refused; }
					try { textures->clear(); } catch (const std::logic_error&) { ++refused; }
				}
				if (phase == 2)
				{
					phase = 0;
					try { (void)job->finish(); } catch (const std::logic_error&) { ++refused; }
					textures->clear(); textures.reset(); job.reset();
				}
				work();
			};
			textures = std::make_unique<guest_texture_cache>(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,texture_capture_bindings{capture,io});
			auto source = registers(); texture_registers(source,0x100);
			const auto words = fragment_texture_snapshot::capture(source.registers,1).at(0);
			phase = 1; job.emplace(textures->prepare_versioned_batch(std::array{words},surfaces));
			require(refused == 2 && phase == 0, "Capture reentry modified a preparing texture cache");
			phase = 2;
			const auto images = job->finish();
			require(refused == 3 && !textures && !job && images.size() == 1 && images[0].revision.current(),
				"Native completion lost owned state or published through a destroyed/cleared cache after its CPU callback");
			(void)ctx.trim_upload_pages();
		}
	}
	require(ledger->statistics().reserved_bytes == 0, "Reentrant capture teardown retained completion charges");
	{
		auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,ledger);
		guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger);
		draw_executor executor(ctx,surfaces,textures);
		auto source = registers(); texture_registers(source,0x100); set_vertex_texture_registers(source,0x200);
		source.registers[0x900 / 4 + 3] = 0; source.registers[0x1a00 / 4 + 3] = 0;
		const auto vertex = vertex_texture_snapshot::capture(source.registers,1), no_vertex = vertex_texture_snapshot::capture(source.registers,0);
		const auto fragment = fragment_texture_snapshot::capture(source.registers,1);
		auto disabled = executor.enqueue_versioned_textures(vertex,fragment);
		require(disabled.ready() && disabled.current(vertex,fragment) && !disabled.current(no_vertex,fragment) &&
			program_sampler_plan::capture(vertex.at(0).upload_words(),disabled.metadata(true,0)).format_class == 0 &&
			program_sampler_plan::capture(fragment.at(0),disabled.metadata(false,0)).format_class == 0 && ram->reads == 0,
			"Disabled pending metadata read guest memory or invented an enabled sampler");
		bool unreferenced = false, out_of_bounds = false;
		try { (void)disabled.metadata(false,1); } catch (const std::invalid_argument&) { unreferenced = true; }
		try { (void)disabled.metadata(true,4); } catch (const std::out_of_range&) { out_of_bounds = true; }
		const auto epoch = disabled.finish();
		require(unreferenced && out_of_bounds && epoch.revisions.empty() && textures.statistics().uploads == 0 && textures.statistics().null_initializations == 1 &&
			epoch.textures.vertex(0).texture && epoch.textures.fragment(0).texture, "Disabled pending units lost null ownership or acquired VM tickets");
	}
	std::cout << "Native versioned callback ownership:preparation/finish reentry refused;completion survives job/cache deletion inside its CPU dispatcher without stale publication or charges;disabled vertex/fragment metadata uses owned null images,no VM reads/tickets,unreferenced/bounds guards passed\n";
}
void verify_versioned_job_orphan()
{
	using namespace std::chrono_literals;
	for (unsigned source_kind = 0; source_kind < 4; ++source_kind)
	{
		const bool multisampled = source_kind >= 2;
		std::optional<versioned_draw_texture_job> orphan;
		std::shared_ptr<heap_budget> ledger;
		std::weak_ptr<guest_surface_memory> weak_memory;
		id<MTLSharedEvent> gate;
		{
			@autoreleasepool
			{
				auto ctx = std::make_unique<context>(2);
				ledger = std::make_shared<heap_budget>(*ctx,8u << 20); ctx->set_buffer_budget(ledger);
				auto ram = std::make_shared<memory>(); weak_memory = ram;
				const cpu_capture_dispatch capture = [ram](const auto& work) { work(); };
				auto io = std::make_shared<io_mapping_epoch>();
				surface_cache surfaces(*ctx,ram,8u << 20,ledger,capture);
				guest_texture_cache textures(*ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,{capture,io});
				draw_executor executor(*ctx,surfaces,textures);
				std::array<std::byte,64> pixels;
				for (unsigned b = 0; b < pixels.size(); ++b) pixels[b] = std::array{std::byte{255},std::byte{20},std::byte{40},std::byte{60}}[b % 4];
				if (!source_kind) ram->write({0x1000,64},pixels);
				else
				{
					const auto surface = surfaces.acquire({0x1000,{multisampled ? 2u : 4u,4,multisampled ? 2u : 1u,surface_color::argb8,false},16,{}});
					surfaces.clear(surface,0xf0,0xff14283c,0,255,{0,0,4,4});
					if (source_kind == 3) surfaces.spill(surface);
				}
				auto source = registers(); texture_registers(source,0x100);
				gate = [ctx->device() newSharedEvent]; const std::array waits{guest_texture_upload_dependency{gate,1}};
				orphan.emplace(executor.enqueue_versioned_textures(vertex_texture_snapshot::capture(source.registers,0),fragment_texture_snapshot::capture(source.registers,1),waits));
				require(!orphan->ready(), "Orphan fixture did not retain a pending native draw upload");
			}
		}
		require(!orphan->ready() && !weak_memory.expired() && ledger->statistics().reserved_bytes > 0,
			"Destroying executor/cache/surface/context/caller pool lost pending job services or native backing");
		bool timed_out = false;
		try { (void)orphan->finish(1ms); } catch (const std::runtime_error& error) { timed_out = std::string(error.what()).find("Timed out waiting") != std::string::npos; }
		require(timed_out, "Orphan completion ignored an unsignalled native dependency");
		gate.signaledValue = 1;
		{
			@autoreleasepool
			{
				const auto epoch = orphan->finish();
				context consumer(1);
				const auto output = make_buffer(consumer,1024,buffer_storage::shared);
				auto image = epoch.textures.fragment(0).texture;
				require(image.sampleCount == (multisampled ? 2u : 1u), "Orphan job lost its native sample storage");
				consumer.begin();
				if (image.sampleCount > 1)
				{
					auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:4 height:4 mipmapped:NO];
					td.storageMode = MTLStorageModePrivate;
					td.usage = MTLTextureUsageRenderTarget;
					const auto expanded = make_texture(consumer, td);
					sample_transfer transfer(consumer);
					transfer.expand(consumer, image, expanded);
					image = expanded;
				}
				texture_to_buffer(consumer,image,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)},output,0,256);
				consumer.submit().wait();
				for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 16; ++x)
					require(static_cast<const unsigned char*>(output.contents)[y * 256 + x] == std::array{60u,40u,20u,255u}[x % 4], "Orphan versioned job changed native pixels after context teardown");
			}
		}
		orphan.reset();
		require(weak_memory.expired() && ledger->statistics().reserved_bytes == 0, "Completed orphan job retained receipt services or native heap charges");
	}
	std::cout << "Native orphan draw jobs: VM, single-sample, resident and spilled MSAA snapshots survive executor/cache/context destruction; timeout retains backing, another context reads 64 exact BGRA bytes per profile, final release retires all charges\n";
}
void verify_native_snapshot_batch()
{
	using namespace std::chrono_literals;
	for (unsigned mode = 0; mode < 3; ++mode)
	{
		context ctx(3); auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		{
			@autoreleasepool
			{
				auto ram = std::make_shared<memory>(); auto io = std::make_shared<io_mapping_epoch>();
				const cpu_capture_dispatch capture = [](const auto& work) { work(); };
				surface_cache surfaces(ctx,ram,8u << 20,ledger,capture);
				guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,{capture,io});
				std::array<cached_surface_ref,4> sources;
				std::array<fragment_texture_registers,5> words;
				std::array<unsigned,4> colors;
				for (unsigned i = 0; i < 4; ++i)
				{
					colors[i] = 0xff102030u + i * 0x00091117u;
					sources[i] = surfaces.acquire({0x1000u + i * 0x1000,{4,4,1,surface_color::argb8,false},16,{}});
					surfaces.clear(sources[i],0xf0,colors[i],0,255,{0,0,4,4});
					auto state = registers(); texture_registers(state,0x100u + i * 0x100);
					words[i] = fragment_texture_snapshot::capture(state.registers,1).at(0);
				}
				words[4] = words[1]; words[4].format |= 0x40u << 8;
				const auto source_submits = surfaces.transaction_statistics().submitted;
				const auto reads = ram->reads;
				const auto encoders = ctx.encoding_statistics();
				const auto gate = [ctx.device() newSharedEvent];
				struct gate_scope { id<MTLSharedEvent> event; ~gate_scope() { event.signaledValue = 1; } } gate_guard{gate};
				const std::array waits{guest_texture_upload_dependency{gate,1}};
				auto job = textures.prepare_versioned_batch(words,surfaces,waits);
				require(!job.ready() && job.current() && surfaces.snapshot_statistics().created == 0 && surfaces.snapshot_statistics().entries == 0 &&
					textures.statistics().uploads == 0 && textures.statistics().casts_created == 0 && textures.statistics().upload_batches == 1 &&
					ctx.encoding_statistics().compute_encoders == encoders.compute_encoders + 1 && surfaces.transaction_statistics().submitted == source_submits && ram->reads == reads,
					"Four cold native snapshots waited,submitted separately,published early or read VM");
				bool timeout = false;
				try { (void)job.finish(1ms); } catch (const std::runtime_error& error) { timeout = std::string(error.what()).find("Timed out waiting") != std::string::npos; }
				require(timeout && !job.ready() && ledger->statistics().reserved_bytes > 0, "Pure native snapshot timeout released its source leases");
				if (mode == 1) surfaces.clear_snapshots();
				gate.signaledValue = 1;
				if (mode == 2) surfaces.clear(sources[0],0xf0,0xffabcdef,0,255,{0,0,4,4});
				const auto images = job.finish();
				require(images.size() == 5 && images[1].image.texture == images[4].image.texture && images[4].image.metadata.coordinates.scale[0] == .25f &&
					surfaces.snapshot_statistics().entries == (mode == 0 ? 4 : mode == 1 ? 0 : 3),
					"Pure native snapshots lost coalescing/UN metadata or republished a cleared/overwritten cache key");
				const auto output = make_buffer(ctx,4096,buffer_storage::shared);
				ctx.begin();
				for (unsigned i = 0; i < 4; ++i) texture_to_buffer(ctx,images[i].image.texture,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(4,4,1)},output,i * 1024,256);
				ctx.submit().wait();
				for (unsigned i = 0; i < 4; ++i) for (unsigned y = 0; y < 4; ++y) for (unsigned b = 0; b < 16; ++b)
					require(static_cast<const unsigned char*>(output.contents)[i * 1024 + y * 256 + b] == ((colors[i] >> ((b % 4) * 8)) & 255),
						"Pending native copy changed captured pixels after clear/new-writer completion");
				const auto copies = surfaces.snapshot_statistics().created, batches = textures.statistics().upload_batches;
				const auto retry = textures.resolve_versioned_batch(words,surfaces);
				require(retry.size() == 5 && surfaces.snapshot_statistics().entries == 4 && ram->reads == reads,
					"Cleared/overwritten native snapshot batch did not recover without VM pixels");
				if (!mode) require(surfaces.snapshot_statistics().created == copies && textures.statistics().upload_batches == batches && retry[0].image.texture == images[0].image.texture,
					"Completed native snapshot group did not remain a ready hit");
				if (mode == 2) require(retry[0].image.texture != images[0].image.texture, "Later native writer reused its old immutable snapshot");
				(void)ctx.trim_upload_pages();
			}
		}
		require(ledger->statistics().reserved_bytes == 0, "Pure native snapshot batch retained orphan native heap charges");
	}
	std::cout << "Native-only snapshot batch:four cold sources plusUN alias share one event-gated encoder/submission,no VM pixels or premature entries;CPU timeout,clear generation,newer native writer preserve256exact old BGRA bytes;ready hits,fresh retries andzero final charges passed\n";
}
void verify_upload_batch_publication()
{
	using namespace std::chrono_literals;
	for (unsigned mode = 0; mode < 3; ++mode)
	{
		context ctx(1);
		auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		auto ram = std::make_shared<memory>(); auto io = std::make_shared<io_mapping_epoch>();
		std::mutex exclusion;
		const cpu_capture_dispatch capture = [&](const std::function<void()>& work) { std::lock_guard lock(exclusion); work(); };
		surface_cache surfaces(ctx,ram,8u << 20,ledger,capture);
		guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,{capture,io});
		std::array<fragment_texture_registers,3> words;
		for (unsigned i = 0; i < words.size(); ++i)
		{
			auto state = registers(); texture_registers(state,0x100 + i * 0x10);
			words[i] = fragment_texture_snapshot::capture(state.registers,1).at(0);
		}
		const auto gate = [ctx.device() newSharedEvent]; const std::array waits{guest_texture_upload_dependency{gate,1}};
		std::atomic<bool> changed = false;
		std::jthread producer([&]
		{
			const auto until = std::chrono::steady_clock::now() + 3s;
			while (!ledger->statistics().live_buffers && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
			if (ledger->statistics().live_buffers)
			{
				std::lock_guard lock(exclusion);
				if (mode == 0) ram->write({0x1000,1},std::array{std::byte{0xab}});
				else { io->notify({mode == 1 ? 0u : 1u << 20,1u << 20}); io->notify({mode == 1 ? 0u : 1u << 20,1u << 20}); }
				changed = true;
			}
			gate.signaledValue = 1;
		});
		bool conflict = false;
		try
		{
			const auto images = textures.resolve_versioned_batch(words,surfaces,waits,5s);
			require(mode == 2 && images.size() == 3, "Changed VM/IO source published a prepared batch");
			for (const auto& image : images) require(image.revision.current(), "Unrelated IO mutation invalidated the batch");
		}
		catch (const texture_revision_conflict&) { conflict = true; }
		producer.join();
		require(changed && conflict == (mode != 2) && textures.statistics().upload_batches == 1 &&
			textures.statistics().uploads == (mode == 2 ? 3 : 0) && textures.statistics().entries == (mode == 2 ? 3 : 0),
			"Batch post-completion publication ignored exact VM/IO receipts or cached a partial failed group");
		const auto retry = textures.resolve_versioned_batch(words,surfaces);
		require(retry.size() == 3 && textures.statistics().entries == 3, "Version conflict prevented a fresh valid grouped retry");
	}
	std::cout << "Native cache batch publication:VM write andsame-EA IO ABA after immutable capture refuse every new entry after GPU completion;unrelated IO entry accepted;fresh grouped retries succeed passed\n";
	for (bool deadline : {false,true})
	{
		context ctx(1,2,deadline ? 100ms : 5s);
		auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		{
			@autoreleasepool
			{
				auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,ledger);
				guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger);
				auto state = registers(); texture_registers(state,0x100);
				std::array<fragment_texture_registers,2> words; words.fill(fragment_texture_snapshot::capture(state.registers,1).at(0)); words[1].offset += 0x10;
				const auto gate = [ctx.device() newSharedEvent]; const std::array waits{guest_texture_upload_dependency{gate,1}};
				bool failed = false;
				try { (void)textures.resolve_versioned_batch(words,surfaces,waits,deadline ? 5s : 1ms); }
				catch (const std::runtime_error& error)
				{
					failed = std::string(error.what()).find(deadline ? "deadline expired" : "Timed out waiting") != std::string::npos;
				}
				require(failed && textures.statistics().upload_batches == 1 && textures.statistics().uploads == 0 && textures.statistics().entries == 0,
					"Failed batch completion published a cache entry");
				if (deadline) ctx.recover_completed_errors();
				else { require(ledger->statistics().reserved_bytes > 0, "Cache timeout prematurely retired native job resources"); gate.signaledValue = 1; ctx.wait_idle(); }
				(void)ctx.trim_upload_pages();
				const auto fresh = textures.resolve_versioned_batch(words,surfaces);
				require(fresh.size() == 2 && textures.statistics().uploads == 2 && textures.statistics().entries == 2, "Timed-out/failed batch prevented valid later slot reuse");
				textures.clear(); (void)ctx.trim_upload_pages();
			}
		}
		require(ledger->statistics().reserved_bytes == 0, "Retired failed/retried cache batch retained native backing");
	}
	std::cout << "Native cache batch timeout/deadline errors publish zero entries,retain pending native backing through completion,permit valid grouped retry and release all charges after cache/caller retirement passed\n";
	{
		context ctx(1);
		auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,ledger);
		guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger);
		auto state = registers(); texture_registers(state,0x100);
		std::array<fragment_texture_registers,3> words; words.fill(fragment_texture_snapshot::capture(state.registers,1).at(0)); words[2].format &= ~3u;
		bool failed = false;
		try { (void)textures.resolve_versioned_batch(words,surfaces); } catch (const std::invalid_argument&) { failed = true; }
		require(failed && textures.statistics().entries == 0 && textures.statistics().uploads == 0 && textures.statistics().upload_batches == 0 &&
			ledger->statistics().created == 0 && ctx.encoding_statistics().compute_encoders == 0 && ram->reads == 1,
			"Invalid final texture allocated/submitted/published earlier pending misses");
		words[2] = words[0];
		const auto reads = ram->reads;
		const auto invalid = [&](const auto& work)
		{
			bool refused = false; try { work(); } catch (const std::exception&) { refused = true; }
			require(refused && ram->reads == reads && ledger->statistics().created == 0, "Batch input guard ran after capture/allocation");
		};
		std::array<fragment_texture_registers,65> excessive; excessive.fill(words[0]);
		invalid([&] { (void)textures.resolve_versioned_batch(excessive,surfaces); });
		const std::array nil_wait{guest_texture_upload_dependency{nil,1}};
		invalid([&] { (void)textures.resolve_versioned_batch(words,surfaces,nil_wait); });
		invalid([&] { (void)textures.resolve_versioned_batch(words,surfaces,{},0ms); });
		auto provider = std::make_shared<guarded_gpu_memory>();
		provider->execute([&](auto&) { invalid([&] { (void)textures.resolve_versioned_batch(words,surfaces); }); });
		ctx.shutdown(); invalid([&] { (void)textures.resolve_versioned_batch(words,surfaces); });
	}
	std::cout << "Native cache batch preparation:invalid last input publishes/encodes/allocates none of the earlier misses;count/dependency/timeout/real VM-epoch/stopped-context preflight rejects before capture passed\n";
	{
		context ctx(1); std::size_t heap_bytes, buffer_bytes;
		@autoreleasepool
		{
			auto probe = std::make_shared<heap_budget>(ctx,8u << 20);
			auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm width:2 height:2 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto image = probe->texture(descriptor); const auto buffer = probe->shared_buffer(4);
			heap_bytes = image.heap.size; buffer_bytes = std::max(buffer.length,buffer.allocatedSize);
		}
		auto ledger = std::make_shared<heap_budget>(ctx,2 * (heap_bytes + buffer_bytes)); ctx.set_buffer_budget(ledger);
		@autoreleasepool
		{
			auto ram = std::make_shared<memory>(); surface_cache surfaces(ctx,ram,8u << 20,ledger);
			guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,2);
			auto state = registers(); texture_registers(state,0x100);
			std::array<fragment_texture_registers,2> words; words.fill(fragment_texture_snapshot::capture(state.registers,1).at(0));
			for (auto& word : words) { word.format = (word.format & ~(0xffu << 8)) | (0xa1u << 8); word.image_rect = (2u << 16) | 2; word.control3 = (1u << 20) | 2; }
			words[1].offset += 0x10;
			try { (void)textures.resolve_versioned_batch(words,surfaces); }
			catch (const heap_budget_error& error) { throw std::runtime_error("Initial cache batch quota:requested=" + std::to_string(error.requested) + " reserved=" + std::to_string(error.reserved) + " limit=" + std::to_string(error.budget)); }
			require(ledger->statistics().reserved_bytes == 2 * heap_bytes, "Ready two-entry cache retained initial source buffers");
			words[0].offset += 0x100; words[1].offset += 0x100;
			const auto fresh = [&]
			{
				try { return textures.resolve_versioned_batch(words,surfaces); }
				catch (const heap_budget_error& error)
				{
					const auto s = ledger->statistics();
					throw std::runtime_error("Replacement cache batch quota:requested=" + std::to_string(error.requested) + " reserved=" + std::to_string(error.reserved) + " limit=" + std::to_string(error.budget) + " now=" + std::to_string(s.reserved_bytes) + " heaps=" + std::to_string(s.live_heaps) + " buffers=" + std::to_string(s.live_buffers) + " evictions=" + std::to_string(textures.statistics().evictions));
				}
			}();
			require(fresh.size() == 2 && textures.statistics().upload_batches == 2 && textures.statistics().uploads == 4 && textures.statistics().evictions == 2 &&
				ledger->statistics().reserved_bytes == 2 * heap_bytes && ledger->statistics().created > 8,
				"Full-ledger batch admission failed to unwind partial encodings before bounded ready-cache eviction/retry");
			textures.clear();
		}
		require(ledger->statistics().reserved_bytes == 0, "Full-ledger batch recovery retained retired native allocations");
	}
	std::cout << "Native cache batch exact-ledger pressure:two old entries evicted only after failed partial encodings unwind;two new misses still commit together,onlytwo successful batches,four published images,zero orphan charges passed\n";
}
void verify_native_cast_batch_publication()
{
	using namespace std::chrono_literals;
	for (unsigned mode = 0; mode < 4; ++mode)
	{
		context ctx(1,2,mode == 3 ? 100ms : 5s);
		auto ledger = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(ledger);
		const auto gate = [ctx.device() newSharedEvent];
		@autoreleasepool
		{
			auto ram = std::make_shared<memory>(); auto io = std::make_shared<io_mapping_epoch>();
			std::mutex exclusion;
			const cpu_capture_dispatch capture = [&](const std::function<void()>& work) { std::lock_guard lock(exclusion); work(); };
			@autoreleasepool
			{
				surface_cache surfaces(ctx,ram,8u << 20,ledger,capture);
				guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset * 16; },1u << 20,ledger,8,{capture,io});
				const auto source = surfaces.acquire({0x4000,{4,4,1,surface_color::argb8,false},16,{}});
				surfaces.clear(source,0xf0,0xff112233,0,255,{0,0,4,4});
				auto state = registers(); texture_registers(state,0x400);
				const auto direct = fragment_texture_snapshot::capture(state.registers,1).at(0);
				// Keep the native snapshot cold:copy andcasts must share this job.
				(void)ctx.trim_upload_pages();
				auto r8 = direct; r8.format = (r8.format & ~(0xffu << 8)) | (0xa1u << 8); r8.image_rect = (16u << 16) | 4;
				auto r16 = direct; r16.format = (r16.format & ~(0xffu << 8)) | (0xb2u << 8); r16.image_rect = (8u << 16) | 4;
				std::vector<fragment_texture_registers> words{r8,r16};
				if (mode == 1) { auto vm = direct; vm.offset = 0x700; words.push_back(vm); }
				auto invalid = r8; invalid.format &= ~3u;
				const auto encoders = ctx.encoding_statistics();
				bool refused = false;
				try { (void)textures.resolve_versioned_batch(std::array{r8,r16,invalid},surfaces); }
				catch (const std::invalid_argument&) { refused = true; }
				require(refused && textures.statistics().upload_batches == 0 && textures.statistics().cast_entries == 0 &&
					textures.statistics().casts_created == 0 && ctx.encoding_statistics().compute_encoders == encoders.compute_encoders,
					"Invalid final input encoded or published an earlier pending native cast");
				const auto buffers = ledger->statistics().live_buffers; const auto reads = ram->reads;
				std::atomic<bool> changed = false;
				std::optional<std::jthread> producer;
				if (mode < 2) producer.emplace([&]
				{
					const auto until = std::chrono::steady_clock::now() + 3s;
					while (ledger->statistics().live_buffers <= buffers && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(1ms);
					if (ledger->statistics().live_buffers > buffers)
					{
						std::lock_guard lock(exclusion);
						if (mode == 0) { io->notify({0,1u << 20}); io->notify({0,1u << 20}); }
						else ram->write({0x4000,1},std::array{std::byte{0xab}});
						changed = true;
					}
					gate.signaledValue = 1;
				});
				bool failed = false;
				const std::array waits{guest_texture_upload_dependency{gate,1}};
				try { (void)textures.resolve_versioned_batch(words,surfaces,waits,mode == 2 ? 1ms : 5s); }
				catch (const texture_revision_conflict&) { failed = mode < 2; }
				catch (const std::runtime_error& error)
				{
					failed = mode >= 2 && std::string(error.what()).find(mode == 2 ? "Timed out waiting" : "deadline expired") != std::string::npos;
				}
				if (producer) producer->join();
				require(failed && (mode >= 2 || changed) && textures.statistics().upload_batches == 1 && textures.statistics().entries == 0 &&
					textures.statistics().uploads == 0 && textures.statistics().cast_entries == 0 && textures.statistics().casts_created == 0 &&
					surfaces.snapshot_statistics().created == 0 && surfaces.snapshot_statistics().entries == 0 && ram->reads == reads + unsigned(mode == 1),
					"Failed native cast batch published a partial entry,waited per cast or read VM pixels");
				if (mode == 3) ctx.recover_completed_errors();
			}
			// Native source/target ownership must outlive BOTH cache adapters.
			if (mode == 2)
			{
				require(ledger->statistics().reserved_bytes > 0, "Timed-out cast job lost backing after cache/surface destruction");
				gate.signaledValue = 1; ctx.wait_idle();
			}
			(void)ctx.trim_upload_pages();
		}
		require(ledger->statistics().reserved_bytes == 0, "Completed/failed native cast batch retained orphan backing");
	}
	std::cout << "Native two-cast batches:late invalid input encodes/publishes none;event-gated VM-write/IO-ABA/timeout/deadline failures publish zero VM/cast entries;pure casts no VM pixel reads,mixed VM source captured once;timed-out source/target leases survive cache/surface destruction until real completion,all charges retire passed\n";
}
void verify_owned_raster()
{
	context ctx;
	auto ram = std::make_shared<memory>();
	auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
	surface_cache surfaces(ctx, ram, 8 << 20, budget);
	guest_texture_cache textures(ctx, ram, [](unsigned offset, unsigned, std::size_t) { return offset; }, 1 << 20, budget);
	draw_executor executor(ctx, surfaces, textures);
	query_pool queries(ctx, 1, 2);
	compiler compile(ctx);
	const auto library = compile.library(R"msl(
#include <metal_stdlib>
using namespace metal;
vertex float4 raster_reference(uint i [[vertex_id]], const device float4* p [[buffer(0)]])
{ return float4(p[i].x, -p[i].y, p[i].z, p[i].w); }
fragment float4 raster_color() { return float4(1); }
)msl");
	auto pipeline_desc = [MTL4RenderPipelineDescriptor new];
	pipeline_desc.vertexFunctionDescriptor = library_function(library, @"raster_reference");
	pipeline_desc.fragmentFunctionDescriptor = library_function(library, @"raster_color");
	pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
	pipeline_desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
	pipeline_desc.rasterSampleCount = 1;
	const auto pipeline = compile.render(pipeline_desc);
	auto depth_state = [MTLDepthStencilDescriptor new];
	depth_state.depthCompareFunction = MTLCompareFunctionAlways;
	depth_state.depthWriteEnabled = YES;
	const auto native_depth = [ctx.device() newDepthStencilStateWithDescriptor:depth_state];
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageRenderTarget;
	const auto reference_color = make_texture(ctx, descriptor);
	const auto actual = make_buffer(ctx, 1024, buffer_storage::shared), expected = make_buffer(ctx, 1024, buffer_storage::shared);
	texture_region whole; whole.size = MTLSizeMake(4, 4, 1);
	const auto color = surfaces.acquire({0x1000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	struct profile { float base, dx, bias, slope; bool fill, clamp, clip; bool bounds = false; };
	std::vector<profile> profiles;
	for (float base : {.25f, .75f}) for (float dx : {0.f, -.03125f, .03125f})
		for (float bias : {-32.f, 0.f, 32.f}) for (float slope : {-2.f, 0.f, 2.f}) for (bool fill : {false, true})
			profiles.push_back({base, dx, bias, slope, fill, false, true});
	for (bool clamp : {false, true}) for (bool clip : {false, true})
		for (const auto geometry : {std::array{-0.25f, 0.f}, std::array{1.25f, 0.f}, std::array{-.5f, .25f}, std::array{.5f, .25f}})
			profiles.push_back({geometry[0], geometry[1], 0, 0, false, clamp, clip});
	for (float bias : {-32.f, 0.f, 32.f}) for (float slope : {-2.f, 0.f, 2.f})
		profiles.push_back({.25f, .03125f, bias, slope, true, false, true, true});
	for (const auto topology : {list_topology::triangles, list_topology::lines, list_topology::points})
	{
		test::environment_state state; state.offset_fill = true; state.polygon_bias = 32; state.polygon_slope = -2;
		draw_request request; request.geometry.topology = topology; request.raster.bias_clamp = 1;
		apply_draw_environment(draw_environment_snapshot::capture(state), 4, 4, request);
		require(request.raster.depth_bias == (topology == list_topology::triangles ? 32 : 0) &&
			request.raster.slope_scale == (topology == list_topology::triangles ? -2 : 0) && request.raster.bias_clamp == 0,
			"Fill offset leaked into point/line rasterization or retained stale bias clamp");
	}
	unsigned cases = 0;
	for (const auto format : {surface_depth::z16_uint, surface_depth::z24s8_uint, surface_depth::z16_float, surface_depth::z24s8_float})
	{
		const auto description = describe_surface_depth(format);
		descriptor.pixelFormat = description.format;
		const auto reference_depth = make_texture(ctx, descriptor);
		const auto depth = surfaces.acquire({0x2000 + 0x1000 * static_cast<unsigned>(format), {4, 4, 1, format, false}, 4 * description.guest_bytes, {}});
		draw_targets targets; targets.colors[0] = color; targets.depth = depth;
		for (const auto p : profiles)
		{
			test::pass_state source;
			source.registers = registers().registers; source.base_index = 0;
			source.registers[0xa74 / 4] = source.registers[0xa70 / 4] = 1;
			source.registers[0xa6c / 4] = 0x207;
			source.offset_fill = p.fill; source.polygon_bias = p.bias; source.polygon_slope = p.slope;
			source.depth_clamp = p.clamp; source.depth_clip = p.clip;
			source.bounds = p.bounds; source.bounds_min = 0; source.bounds_max = 1;
			const std::array<float, 12> vertices{-1, -1, p.base, 1, 3, -1, p.base + 8 * p.dx, 1, -1, 3, p.base, 1};
			test::pass_processor processor;
			processor.descriptors[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
			for (float value : vertices) for (unsigned shift : {24u, 16u, 8u, 0u})
				processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
			for (unsigned value : {0u, 1u, 2u}) for (unsigned shift : {24u, 16u, 8u, 0u})
				processor.index_bytes.push_back(std::byte(value >> shift));
			test::pass_layout layout; test::pass_metadata metadata;
			auto vertex = vertex_program(); auto bytes = program(false);
			test::fragment_program fragment; fragment.pointer = bytes.data(); fragment.ucode_length = bytes.size(); fragment.ctrl = rsx_fragment_float32;
			const auto packet = rsx_draw_packet::capture(source, processor, layout, metadata, vertex, fragment, 0, provoking_vertex::last,
				[](const auto&, unsigned first, unsigned count) { require(first == 0 && count == 3, "Raster packet geometry differs"); return std::pair<std::uint32_t, std::uint32_t>{48, 0}; });
			source.offset_fill = !p.fill; source.depth_clamp = !p.clamp; source.depth_clip = !p.clip;
			source.polygon_bias = source.polygon_slope = std::numeric_limits<float>::quiet_NaN();
			processor.persistent.assign(48, std::byte{0});
			// Use the same exact native clear depth for all four guest codecs.
			const std::array writable{cached_surface_use{depth, true}};
			surfaces.draw(writable, [&](context& commands, auto textures)
			{
				auto pass = [MTL4RenderPassDescriptor new]; pass.depthAttachment.texture = textures[0];
				pass.depthAttachment.loadAction = MTLLoadActionClear; pass.depthAttachment.clearDepth = 1;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
				render_commands clear(commands, pass); clear.end();
			});
			draw_execution_state execution; execution.queries = &queries;
			// Owned guest raster state must replace stale encoder overrides.
			execution.raster.depth_bias = 999; execution.raster.slope_scale = 999; execution.raster.bias_clamp = .01f;
			execution.raster.depth_clip = p.clamp || !p.clip ? MTLDepthClipModeClip : MTLDepthClipModeClamp;
			queries.begin(0, true); executor.execute(packet, targets, execution); queries.end(0);
			const auto count = queries.consume(0);
			const std::array readable{cached_surface_use{depth, false}};
			surfaces.draw(readable, [&](context& commands, auto textures) { texture_to_buffer(commands, textures[0], whole, actual, 0, 256, 0, texture_plane::depth); });
			// Independent direct MSL/native rasterizer path: no RSX snapshots,
			// vertex translator or environment binding supplies these states.
			ctx.begin();
			const auto positions = make_upload(ctx, std::as_bytes(std::span{vertices}));
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = reference_color; pass.colorAttachments[0].loadAction = MTLLoadActionDontCare; pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
			pass.depthAttachment.texture = reference_depth; pass.depthAttachment.loadAction = MTLLoadActionClear;
			pass.depthAttachment.clearDepth = 1; pass.depthAttachment.storeAction = MTLStoreActionStore;
			raster_state raster; raster.viewport = {0, 0, 4, 4, 0, 1}; raster.scissor = {0, 0, 4, 4};
			raster.depth_bias = p.fill ? p.bias : 0; raster.slope_scale = p.fill ? p.slope : 0;
			raster.depth_clip = p.clamp || !p.clip ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
			bindings bindings; bindings.set_buffer(0, positions);
			render_commands commands(ctx, pass); commands.configure(pipeline, native_depth, raster, bindings, {});
			commands.draw(MTLPrimitiveTypeTriangle, 0, 3); commands.end();
			texture_to_buffer(ctx, reference_depth, whole, expected, 0, 256, 0, texture_plane::depth); ctx.submit().wait();
			for (unsigned y = 0; y < 4; ++y)
				require(std::memcmp(static_cast<const std::byte*>(actual.contents) + y * 256, static_cast<const std::byte*>(expected.contents) + y * 256, 4 * description.depth_bytes) == 0,
					"Owned raster depth differs from independent native bias/clip/clamp pipeline");
			if (p.dx == 0 && (p.base < 0 || p.base > 1))
				require(count == (p.clamp || !p.clip ? 16 : 0), "Outside plane clip/clamp visibility differs");
			if (p.dx == 0 && p.base > 0 && p.base < 1)
			{
				float stored;
				if (description.depth_bytes == 2) stored = *static_cast<const std::uint16_t*>(actual.contents) / 65535.f;
				else stored = *static_cast<const float*>(actual.contents);
				if (p.fill && p.bias != 0) require((stored - p.base) * p.bias > 0, "Constant native bias sign disappeared");
				else if ((!p.fill || p.slope == 0) && std::abs(stored - p.base) >= 1.f / 65535) throw std::runtime_error("Disabled/zero native bias changed flat depth:format=" + std::to_string(static_cast<unsigned>(format)) +
					" base=" + std::to_string(p.base) + " stored=" + std::to_string(stored) + " fill=" + std::to_string(p.fill) + " bias=" + std::to_string(p.bias) + " slope=" + std::to_string(p.slope));
			}
			++cases;
		}
	}
	require(cases == 532, "Owned raster matrix omitted depth formats or raster profiles");
	std::cout << "Native owned raster:" << cases << " Z16/Z24/integer/float bias and signed slope/fill/clip/clamp/stored-depth-bounds cases match independent direct MSL4 depth bytes;frozen registers,stale override replacement,point/line isolation and near/far visibility passed\n";
}
void verify_separated_flat_draws(bool gate_stages = false)
{
	for(bool mirror:{false,true})
	{
		context ctx(2);auto ram=std::make_shared<memory>();auto ledger=std::make_shared<heap_budget>(ctx,16u<<20);ctx.set_buffer_budget(ledger);
		surface_cache surfaces(ctx,ram,16u<<20,ledger);guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t){return offset;},1u<<20,ledger);
		M4GatedCompiler* gate = nil;
		if (gate_stages)
		{
			NSError* error = nil;
			const auto native = [ctx.device() newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
			require(native != nil, "Unable to create real gated draw compiler");
			gate = [[M4GatedCompiler alloc] initWithNative:native];
		}
		draw_executor executor(ctx,surfaces,textures,(id<MTL4Compiler>)gate);
		struct release_gate { M4GatedCompiler* gate; ~release_gate() { [gate releaseCallbacks]; } } release{gate};
		draw_targets targets;targets.colors[0]=surfaces.acquire({0x11000,{4,4,1,surface_color::argb8,false},16,{}});
		if(mirror){targets.depth=surfaces.acquire({0x21000,{4,4,1,surface_depth::z24s8_uint,false},16,{}});surfaces.clear(targets.depth,3,0,0x40000000,255,{0,0,4,4});}
		const auto packet=[&](std::array<unsigned,4> rgba,unsigned x,unsigned width,bool flat = true, unsigned color_mask = 0x01010101)
		{
			test::pass_state state;state.registers=registers().registers;state.current_draw_clause.command=1;state.current_draw_clause.primitive=8;state.current_draw_clause.first=0;state.current_draw_clause.count=4;
			state.bounds=mirror;state.bounds_min=.2f;state.bounds_max=.3f;state.registers[0x8c0/4]=(width<<16)|x;
			state.registers[0x324 / 4] = color_mask;
			test::pass_processor processor;test::pass_layout layout;test::pass_metadata metadata;metadata.referenced_inputs_mask=3;
			for(float value:{-1.f,-1.f,.5f,1.f,1.f,-1.f,.5f,1.f,1.f,1.f,.5f,1.f,-1.f,1.f,.5f,1.f})
				for(unsigned shift:{24u,16u,8u,0u})processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value)>>shift));
			for(unsigned vertex=0;vertex<4;++vertex)for(unsigned c=0;c<4;++c)
			{
				const auto raw=std::bit_cast<unsigned>((!flat || vertex==3) ? rgba[c]/255.f : (c==3 ? 1.f : 0.f));
				for(unsigned shift:{24u,16u,8u,0u})processor.persistent.push_back(std::byte(raw>>shift));
			}
			for(unsigned unit=0;unit<2;++unit)processor.descriptors[unit]={16u|(1u<<8)|(2u<<24)|(4u<<27),(1u<<29)|unit*64};
			auto vertex=vertex_program();vertex.data[7]&=~1u;vertex.ctrl=flat ? rsx_program_flat : 0;vertex.output_mask=1;
			constexpr unsigned input=2u|(1u<<12)|(2u<<10)|(3u<<8);
			for(unsigned word:{(63u<<15)|(7u<<10)|(1u<<30),(1u<<22)|(1u<<8)|(input>>9),(input&511)<<23,(15u<<13)|(15u<<17)|(1u<<7)|(1u<<2)|1u})vertex.data.push_back(word);
			vertex.instruction_mask.set(2);
			std::vector<std::byte> bytes;for(unsigned word:{(1u<<24)|(15u<<9)|(1u<<13)|1u,(0xe4u<<9)|(7u<<18)|(0xe4u<<21)|1u,0xe4u<<9,0xe4u<<9})append(bytes,word);
			test::fragment_program fragment;fragment.pointer=bytes.data();fragment.ucode_length=bytes.size();fragment.ctrl=rsx_fragment_float32|(flat ? rsx_program_flat : 0);
			draw_capture_limits limits;limits.vertex_stream_bytes=128;
			limits.sparse_sources=[&](const vertex_stream_range& range)
			{
				require(range.source_vertices && *range.source_vertices==std::vector<unsigned>{0,1,2,3},"Flat quad capture lost source IDs");
				vertex_sparse_inputs sources;sources.attribute_mask=3;
				for(unsigned unit=0;unit<2;++unit)sources.attributes[unit]={processor.descriptors[unit],false,unit*64,processor.persistent};
				return sources;
			};
			return rsx_draw_packet::capture(state,processor,layout,metadata,vertex,fragment,0,provoking_vertex::last,
				[](const auto&,unsigned,unsigned)->std::pair<unsigned,unsigned>{throw std::runtime_error("Flat quad entered legacy prefix capture");},limits);
		};
		const auto full=packet({64,128,192,255},0,4);surfaces.clear(targets.colors[0],0xf0,0xff112233,0,255,{0,0,4,4});
		const auto under_pressure = [&](const rsx_draw_packet& draw, const auto& check_admission)
		{
			const auto free = ledger->statistics();
			const auto available = free.budget - free.reserved_bytes;
			constexpr std::size_t block = 64 << 10;
			require(available > block, "Owned vertex upload test lacks free space");
			std::vector<id<MTLBuffer>> reservations{ledger->shared_buffer((available - block) / block * block)};
			for (;;)
			{
				try { reservations.push_back(ledger->shared_buffer(16)); }
				catch (const heap_budget_error&) { break; }
			}
			unsigned admissions = 0;
			const auto recover = ledger->register_reclaimer([&](memory_pressure_request pressure)
			{
				require(!pressure.native_failure && admissions == 0, "Unexpected draw pressure recovery");
				check_admission();
				++admissions;
				reservations.clear();
			});
			executor.execute(draw, targets);
			require(admissions == 1, "Draw storage did not reach pressure recovery");
		};
		under_pressure(full, [&]
		{
			require(executor.vertex_statistics().prewarmed == 1 && executor.fragment_statistics().prewarmed == 1 &&
				!executor.vertex_statistics().prewarm_waited && !executor.fragment_statistics().prewarm_waited &&
				!executor.pipeline_statistics().created && !ctx.geometry_uploads().statistics().created,
				"Owned vertex allocation reached pressure recovery after waiting for a shader or uploading geometry");
			if (gate_stages)
			{
				[gate waitForCallbacks:2];
				require(executor.pipeline_statistics().prewarmed == 1 && executor.compilation_statistics().admitted == 4 &&
					!executor.vertex_statistics().compiled && !executor.fragment_statistics().compiled && ![gate renderLaunches],
					"Productive draw did not queue its PSO before storage admission with both stage callbacks held");
				[gate releaseLibraryCallback:@"RPCS3 native RSX vertex program"];
				require(![gate renderLaunches], "Productive draw launched PSO without fragment dependency");
				[gate releaseLibraryCallback:@"RPCS3 native RSX fragment program"];
				[gate releaseCallbacks];
			}
		});
		// Two stage libraries, one flat-color compute pipeline and one render pipeline.
		require(ctx.vertex_uploads().statistics().created && executor.compilation_statistics().admitted == 4 &&
			executor.pipeline_statistics().created == 1, "Early owned vertex upload duplicated a shader or pipeline job");
		verify(ctx,surfaces,targets.colors[0],{64,128,192,255});
		std::cout << "Early owned vertex upload: pressure recovery followed both shader submissions and preceded materialization and geometry; one PSO and 16 exact pixels passed (depth mirror " << mirror << ")\n";
		if (gate_stages)
		{
			require([gate renderLaunches] == 1 && [gate computeLaunches] == 1 && ![gate mainThreadLaunch],
				"Productive gated draw duplicated native jobs or launched compilation on owner");
			std::cout << "Pending-stage productive draw: native render dependency admitted before vertex storage, both libraries required, same PSO adopted and 16 exact pixels passed (depth mirror " << mirror << ")\n";
			continue;
		}
		const auto left=packet({255,0,0,255},0,2),right=packet({0,255,0,255},2,2);
		const auto before=surfaces.transaction_statistics();
		{auto batch=surfaces.batch_draws();for(unsigned i=0;i<8;++i)executor.execute(i&1 ? right : left,targets);batch.finish();}
		require(surfaces.transaction_statistics().draw_batches==before.draw_batches+1 && surfaces.transaction_statistics().batched_draws==before.batched_draws+8,
			"Flat result prepasses split compatible owned draws or lost their transaction");
		require(executor.compilation_statistics().admitted == 4 && executor.pipeline_statistics().created == 1,
			"Flat Color draws started new compiler jobs for an already resolved pipeline");
		require(executor.geometry_plans_created() == 1,
			"Compatible owned draws rebuilt the CPU geometry plan");
		verify(ctx,surfaces,targets.colors[0],{255,0,0,255},{0,0,2,4},{0,255,0,255});
		std::cout << "Owned geometry plan: one CPU plan for nine draws with fresh vertex data and exact scissored output (depth mirror " << mirror << ")\n";
		const auto masked = packet({128, 200, 251, 255}, 0, 4, true, 0x00010000);
		const auto before_pso = executor.pipeline_statistics();
		const auto before_vertex = executor.vertex_statistics();
		const auto before_fragment = executor.fragment_statistics();
		const auto before_jobs = executor.compilation_statistics().admitted;
		under_pressure(masked, [&]
		{
			require(executor.pipeline_statistics().prewarmed == before_pso.prewarmed + 1 &&
				executor.pipeline_statistics().created == before_pso.created &&
				executor.compilation_statistics().admitted == before_jobs + 1 &&
				executor.vertex_statistics().compiled == before_vertex.compiled &&
				executor.fragment_statistics().compiled == before_fragment.compiled,
				"Cold render PSO did not start before native vertex storage admission");
		});
		require(executor.pipeline_statistics().created == before_pso.created + 1 &&
			executor.compilation_statistics().admitted == before_jobs + 1,
			"Cold render PSO was not adopted exactly once");
		verify(ctx, surfaces, targets.colors[0], {128,0,0,255}, {0,0,2,4}, {128,255,0,255});
		std::cout << "Ready-stage cold PSO: one job before native vertex storage admission, 16 exact masked pixels (depth mirror " << mirror << ")\n";
		{
			auto batch = surfaces.batch_draws();
			executor.execute(left, targets);
			executor.execute(right, targets);
			batch.finish();
		}

		if (!mirror)
		{
			draw_targets other;
			other.colors[0] = surfaces.acquire({0x12000, {4,4,1,surface_color::argb8,false},16,{}});
			surfaces.clear(other.colors[0], 0xf0, 0xff112233, 0, 255, {0,0,4,4});
			const auto base = packet({255,0,0,255}, 0, 4, false);
			const auto overlay = packet({0,255,0,255}, 2, 2, false);
			const auto encoders = ctx.encoding_statistics().render_encoders;
			const auto submissions = surfaces.transaction_statistics().submitted;
			{
				auto batch = surfaces.batch_draws();
				executor.execute(base, targets);
				executor.execute(overlay, targets);
				require(ctx.encoding_statistics().render_encoders == encoders + 1,
					"Compatible owned draws opened more than one Metal4 render encoder");
				executor.execute(base, other);
				require(ctx.encoding_statistics().render_encoders == encoders + 2,
					"An attachment change did not start a new Metal4 render encoder");
				executor.execute(overlay, targets);
				require(ctx.encoding_statistics().render_encoders == encoders + 3,
					"Switching between already prepared attachments reused the wrong encoder");
				batch.finish();
			}
			require(surfaces.transaction_statistics().submitted == submissions + 1,
				"Render pass reuse split the native draw batch");
			verify(ctx, surfaces, targets.colors[0], {255,0,0,255}, {0,0,2,4}, {0,255,0,255});
			verify(ctx, surfaces, other.colors[0], {255,0,0,255});
			std::cout << "Owned render pass reuse: two ordered draws in one encoder, two attachment switches, one submission and 32 exact pixels\n";
			for (bool precise : {true, false})
			{
				query_pool shared_query(ctx, 3, 3, ledger);
				query_pool foreign_query(ctx, 1, 1, ledger);
				draw_execution_state queried;
				queried.queries = &shared_query;
				draw_execution_state foreign;
				foreign.queries = &foreign_query;
				const auto query_encoders = ctx.encoding_statistics().render_encoders;
				const auto query_submissions = surfaces.transaction_statistics().submitted;
				auto batch = surfaces.batch_draws();
				shared_query.begin(0, precise);
				executor.execute(base, targets, queried);
				executor.execute(overlay, targets, queried);
				shared_query.end(0);
				require(ctx.encoding_statistics().render_encoders == query_encoders + 1 &&
					shared_query.statistics().initializations == 1 && shared_query.statistics().reused_batch == 1,
					"Compatible visibility draws did not share the Metal4 encoder and one initialized counter");
				executor.execute(overlay, targets, queried);
				shared_query.begin(1, !precise);
				executor.execute(overlay, targets, queried);
				shared_query.end(1);
				shared_query.begin(2, true);
				executor.execute(base, targets, queried);
				executor.execute(overlay, targets);
				require(ctx.encoding_statistics().render_encoders == query_encoders + 1 &&
					shared_query.statistics().initializations == 3 && shared_query.statistics().reused_batch == 1,
					"Same-buffer query offsets and disabled draws split the Metal4 encoder");
				executor.execute(overlay, targets, queried);
				shared_query.end(2);
				require(ctx.encoding_statistics().render_encoders == query_encoders + 2 &&
					shared_query.statistics().reused_batch == 2, "Reactivating a used visibility offset did not start a new encoder");
				foreign_query.begin(0, true);
				executor.execute(overlay, targets, foreign);
				foreign_query.end(0);
				require(ctx.encoding_statistics().render_encoders == query_encoders + 3,
					"A different query buffer reused the previous Metal4 encoder");
				batch.finish();
				require(surfaces.transaction_statistics().submitted == query_submissions + 1 &&
					shared_query.consume(0) == (precise ? 24u : 1u) && shared_query.consume(1) == (precise ? 1u : 8u) &&
					shared_query.consume(2) == 24u && foreign_query.consume(0) == 8u,
					"Dynamic visibility mode or offset changed a query result");
				verify(ctx, surfaces, targets.colors[0], {255,0,0,255}, {0,0,2,4}, {0,255,0,255});
				std::cout << "Owned dynamic visibility: three counters and two disabled draws in one encoder, reactivated offset and foreign arena split; exact results "
					<< (precise ? "24/1/24/8" : "1/8/24/8") << " and 16 pixels\n";
			}
		}

		for (const auto& target : {targets.colors[0], targets.depth})
			if (target) { surfaces.flush(target); surfaces.prepare(target); }
		const auto color_before = surfaces.status(targets.colors[0]);
		const auto depth_before = mirror ? surfaces.status(targets.depth) : cached_surface_status{};
		query_pool queries(ctx, 1, 2, ledger);
		ctx.begin();
		const auto off = queries.capture_report_predicate({{}, 0, true});
		const auto on = queries.capture_report_predicate({{}, 1, true});
		require(off && on, "Native executor predicate capture failed");
		ctx.submit().wait();
		const auto unchanged = [](const cached_surface_status& a, const cached_surface_status& b)
		{
			return a.initialized == b.initialized && a.dirty == b.dirty && a.spilled == b.spilled &&
				!a.pending && !a.faulted && a.content_generation == b.content_generation &&
				a.native_write_epoch == b.native_write_epoch && a.guest_revisions == b.guest_revisions;
		};
		draw_execution_state conditional;
		conditional.queries = &queries;
		conditional.predicate = queries.submit_report_predicate({{}, 0, true});
		require(conditional.predicate && conditional.predicate->settled_cpu_result(ctx) == false, "Completed CPU-known false report was not recognized");
		const auto skipped_transactions = surfaces.transaction_statistics().submitted;
		const auto skipped_encoders = ctx.encoding_statistics();
		const auto skipped_allocations = ledger->statistics().created;
		const auto skipped_vertex = executor.vertex_statistics().lookups;
		const auto skipped_fragment = executor.fragment_statistics().lookups;
		queries.begin(0, true);
		executor.execute(full, targets, conditional);
		queries.end(0);
		require(queries.consume(0) == 0 && surfaces.transaction_statistics().submitted == skipped_transactions &&
			ctx.encoding_statistics().compute_encoders == skipped_encoders.compute_encoders &&
			ctx.encoding_statistics().render_encoders == skipped_encoders.render_encoders &&
			ledger->statistics().created == skipped_allocations && executor.vertex_statistics().lookups == skipped_vertex &&
			executor.fragment_statistics().lookups == skipped_fragment &&
			unchanged(surfaces.status(targets.colors[0]), color_before) &&
			(!mirror || unchanged(surfaces.status(targets.depth), depth_before)),
			"Known false report prepared or submitted a native draw");
		{
			const auto event = [ctx.device() newSharedEvent];
			struct release { id<MTLSharedEvent> event; ~release() { event.signaledValue = 1; } } release_gate{event};
			ctx.begin();
			ctx.wait_event(event, 1);
			const auto producer = ctx.submit();
			const auto blocked_false = queries.submit_report_predicate({{}, 0, true});
			const auto blocked_true = queries.submit_report_predicate({{}, 1, true});
			require(blocked_false && !blocked_false->settled_cpu_result(ctx) &&
				blocked_true && !blocked_true->settled_cpu_result(ctx) && producer.cancel_pending(),
				"CPU-known report bypassed a pending predecessor");
			event.signaledValue = 1;
			try { producer.wait(); } catch (const submission_wait_timeout&) { throw; } catch (const std::runtime_error&) {}
			bool rejected = false;
			try { (void)blocked_false->settled_cpu_result(ctx); }
			catch (const submission_wait_timeout&) { throw; }
			catch (const std::runtime_error&) { rejected = true; }
			require(producer.cancelled() && rejected, "Known false report hid a failed predecessor");
			rejected = false;
			try { (void)blocked_true->settled_cpu_result(ctx); }
			catch (const submission_wait_timeout&) { throw; }
			catch (const std::runtime_error&) { rejected = true; }
			require(rejected, "Known true report hid a failed predecessor");
			ctx.recover_completed_errors();
		}
		conditional.predicate = off;
		queries.begin(0, true);
		const auto false_dispatches = ctx.encoding_statistics().checked_dispatches;
		executor.execute(full, targets, conditional);
		queries.end(0);
		require(queries.consume(0) == 0 && ctx.encoding_statistics().checked_dispatches == false_dispatches + 1,
			"False predicate ran flat-color compute or emitted visible geometry");
		require(unchanged(surfaces.status(targets.colors[0]), color_before) &&
			(!mirror || unchanged(surfaces.status(targets.depth), depth_before)),
			"False predicate published a new surface version");
		verify(ctx,surfaces,targets.colors[0],{255,0,0,255},{0,0,2,4},{0,255,0,255});
		conditional.predicate = on;
		queries.begin(0, true);
		const auto true_dispatches = ctx.encoding_statistics().checked_dispatches;
		executor.execute(full, targets, conditional);
		queries.end(0);
		const auto color_after = surfaces.status(targets.colors[0]);
		require(queries.consume(0) == 16 && ctx.encoding_statistics().checked_dispatches == true_dispatches + 1 &&
			color_after.dirty && color_after.content_generation == color_before.content_generation + 1 &&
			color_after.native_write_epoch > color_before.native_write_epoch,
			"True predicate lost native visibility or failed to publish the surface");
		verify(ctx,surfaces,targets.colors[0],{64,128,192,255});

		constexpr std::uint32_t color_address = 0x11000, label_address = color_address + 4, label_value = 0x12345678;
		constexpr std::array old_pixel{255u, 17u, 34u, 51u}, new_pixel{255u, 64u, 128u, 192u};
		constexpr std::array label_bytes{std::byte{0x12}, std::byte{0x34}, std::byte{0x56}, std::byte{0x78}};
		const auto check_guest = [&](const std::array<unsigned, 4>& pixel, bool labelled)
		{
			const auto snapshot = ram->read({color_address, 64});
			for (std::size_t i = 0; i < snapshot.bytes.size(); ++i)
			{
				const auto expected = labelled && i >= 4 && i < 8 ? label_bytes[i - 4] : std::byte(pixel[i % 4]);
				require(snapshot.bytes[i] == expected, "Predicated draw and label produced incorrect guest bytes");
			}
		};
		const auto prepare_label_target = [&]
		{
			surfaces.clear(targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
			surfaces.flush(targets.colors[0]);
			surfaces.prepare(targets.colors[0]);
		};
		const auto queue_draw = [&](const report_predicate& predicate, id<MTLSharedEvent> gate)
		{
			draw_execution_state state;
			state.queries = &queries;
			state.predicate = predicate;
			if (gate) state.waits.push_back({gate, 1});
			surfaces.set_wait_timeout(std::chrono::milliseconds(1));
			bool timed_out = false;
			try { executor.execute(full, targets, state); }
			catch (const submission_wait_timeout&) { timed_out = true; }
			surfaces.set_wait_timeout(std::chrono::seconds(30));
			require(timed_out && surfaces.status(targets.colors[0]).pending, "Predicated label test lost its pending draw");
		};
		struct gate_guard
		{
			id<MTLSharedEvent> event;
			~gate_guard() { if (!event.signaledValue) event.signaledValue = 1; }
			void open() { event.signaledValue = 1; }
		};
		for (const bool visible : {false, true})
		{
			prepare_label_target();
			const auto prior = surfaces.status(targets.colors[0]);
			gate_guard gate{[ctx.device() newSharedEvent]};
			queue_draw(visible ? *on : *off, gate.event);
			unsigned dependencies = 0, stores = 0;
			const auto sync = [&]
			{
				const auto after = surfaces.status(targets.colors[0]);
				require(!after.pending && after.dirty == visible && after.content_generation == prior.content_generation + visible,
					"Label dependencies saw the wrong predicated surface version");
				++dependencies;
			};
			const auto store = [&](unsigned address, unsigned value)
			{
				require(dependencies == 1 && !stores && address == label_address && value == label_value &&
					!surfaces.status(targets.colors[0]).dirty, "Predicated alias label preceded writeback");
				check_guest(visible ? new_pixel : old_pixel, false);
				ram->write({address, 4}, label_bytes);
				++stores;
			};
			bool timed_out = false;
			try { release_guest_label(surfaces, textures, label_address, label_value, sync, store, std::chrono::milliseconds(1)); }
			catch (const submission_wait_timeout&) { timed_out = true; }
			require(timed_out && !stores && !dependencies, "Pending predicated draw allowed an early label");
			check_guest(old_pixel, false);
			gate.open();
			release_guest_label(surfaces, textures, label_address, label_value, sync, store);
			require(stores == 1 && dependencies == 1, "Completed predicated draw did not publish the label");
			check_guest(visible ? new_pixel : old_pixel, true);
		}

		prepare_label_target();
		const auto before_failure = surfaces.status(targets.colors[0]);
		const auto failure = make_buffer(ctx, 4, buffer_storage::shared);
		*static_cast<std::uint32_t*>(failure.contents) = 1;
		gate_guard failed_gate{[ctx.device() newSharedEvent]};
		ctx.begin();
		ctx.wait_event(failed_gate.event, 1);
		const auto failed_predicate = queries.capture_report_predicate({{}, 1, true});
		require(failed_predicate.has_value(), "Predicate failure fixture could not capture its report");
		ctx.require_gpu_success(failure, 0, @"Intentional predicated label producer failure");
		const auto failed_producer = ctx.submit();
		queue_draw(*failed_predicate, nil);
		failed_gate.open();
		unsigned failed_callbacks = 0;
		bool failed_label = false;
		try
		{
			release_guest_label(surfaces, textures, label_address, label_value,
				[&] { ++failed_callbacks; }, [&](auto, auto) { ++failed_callbacks; });
		}
		catch (const std::runtime_error& error)
		{
			failed_label = std::string_view(error.what()).find("Intentional predicated label producer failure") != std::string_view::npos;
		}
		require(failed_label && failed_producer.ready() && !failed_callbacks &&
			unchanged(surfaces.status(targets.colors[0]), before_failure), "Failed predicate leaked a label or a writable surface");
		check_guest(old_pixel, false);
		ctx.recover_completed_errors();

		for (const bool visible : {false, true})
		{
			prepare_label_target();
			gate_guard gate{[ctx.device() newSharedEvent]};
			queue_draw(visible ? *on : *off, gate.event);
			store_guest_flip_label(label_address, label_value, [&](unsigned address, unsigned)
				{ ram->write({address, 4}, label_bytes); });
			require(!gate.event.signaledValue && surfaces.status(targets.colors[0]).pending,
				"Predicated flip label blocked on GPU completion");
			check_guest(old_pixel, true);
			gate.open();
			bool conflict = false;
			try { surfaces.synchronize(); }
			catch (const surface_conflict_error&) { conflict = true; }
			require(conflict == visible && !surfaces.status(targets.colors[0]).pending &&
				!surfaces.status(targets.colors[0]).dirty, "Flip label accepted a stale conditional surface");
			surfaces.flush(targets.colors[0]);
			check_guest(old_pixel, true);
		}

		prepare_label_target();
		const auto deferred_reader = surfaces.acquire({0x31000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
		surfaces.defer_copy(targets.colors[0], {0, 0, 4, 4}, deferred_reader, 0, 0);
		for (const auto* predicate : {&*off, &*on})
		{
			conditional.predicate = *predicate;
			executor.execute(full, targets, conditional);
			const auto reader = surfaces.status(deferred_reader);
			require(reader.deferred_copies == 1 && !reader.dirty, "Versioned predicated draw materialized a disjoint reader");
		}
		verify(ctx, surfaces, targets.colors[0], {64, 128, 192, 255});
		verify(ctx, surfaces, deferred_reader, {17, 34, 51, 255});

		const auto alias = surfaces.acquire({color_address + 8, {2, 4, 1, surface_color::argb8, false}, 16, {}});
		const auto donor = surfaces.acquire({0x32000, {2, 4, 1, surface_color::argb8, false}, 8, {}});
		for (const bool deferred : {false, true})
		{
			prepare_label_target();
			if (deferred)
			{
				surfaces.clear(donor, 0xf0, 0xff00ff00, 0, 255, {0, 0, 2, 4});
				surfaces.defer_copy(donor, {0, 0, 2, 4}, alias, 0, 0);
			}
			else surfaces.clear(alias, 0xf0, 0xff00ff00, 0, 255, {0, 0, 2, 4});
			const auto prior = surfaces.status(alias);
			const auto guest_before = ram->read({color_address, 64});
			if (!deferred)
			{
				gate_guard gate{[ctx.device() newSharedEvent]};
				ctx.begin();
				ctx.wait_event(gate.event, 1);
				conditional.predicate = queries.capture_report_predicate({{}, 0, true});
				require(conditional.predicate.has_value(), "Overlap timeout fixture could not capture its report");
				const auto producer = ctx.submit();
				surfaces.set_wait_timeout(std::chrono::milliseconds(1));
				bool timed_out = false;
				try { executor.execute(left, targets, conditional); }
				catch (const submission_wait_timeout&) { timed_out = true; }
				surfaces.set_wait_timeout(std::chrono::seconds(30));
				const auto during = ram->read({color_address, 64});
				require(timed_out && !producer.ready() && during.bytes == guest_before.bytes &&
					during.revision == guest_before.revision && unchanged(surfaces.status(alias), prior),
					"Pending predicate allowed overlapping writeback");
				gate.open();
				producer.wait();
			}
			conditional.predicate = off;
			executor.execute(left, targets, conditional);
			const auto guest_after = ram->read({color_address, 64});
			const auto unchanged_alias = surfaces.status(alias);
			require(guest_after.bytes == guest_before.bytes && guest_after.revision == guest_before.revision &&
				unchanged(unchanged_alias, prior) && unchanged_alias.deferred_copies == prior.deferred_copies,
				"False predicate published overlapping surface preparation");
			conditional.predicate = on;
			executor.execute(left, targets, conditional);
			verify(ctx, surfaces, targets.colors[0], {255, 0, 0, 255}, {0, 0, 2, 4}, {0, 255, 0, 255});
		}
		auto cold_targets = targets;
		cold_targets.colors[0] = surfaces.acquire({0x33000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
		const auto cold = surfaces.status(cold_targets.colors[0]);
		const auto reads = ram->reads;
		const auto render_encoders = ctx.encoding_statistics().render_encoders;
		conditional.predicate = off;
		executor.execute(left, cold_targets, conditional);
		require(unchanged(surfaces.status(cold_targets.colors[0]), cold) && ram->reads == reads &&
			ctx.encoding_statistics().render_encoders == render_encoders, "False predicate initialized a cold surface");
		for (unsigned profile = 0; profile < 3; ++profile)
		{
			const auto address = 0x34000u + profile * 0x1000;
			std::array<std::byte,64> bytes;
			for (unsigned i = 0; i < bytes.size(); ++i)
				bytes[i] = std::byte(std::array{255u,17u,34u,51u}[i % 4]);
			ram->write({address, bytes.size()}, bytes);
			cold_targets.colors[0] = surfaces.acquire({address, {4,4,1,surface_color::argb8,false},16,{}});
			const auto target = cold_targets.colors[0];
			if (profile == 1)
			{
				surfaces.clear(target, 0xf0, 0xff00ff00, 0, 255, {0,0,4,4});
				surfaces.spill(target);
			}
			if (profile == 2)
			{
				const auto source = surfaces.acquire({0x37000, {4,4,1,surface_color::argb8,false},16,{}});
				surfaces.clear(source, 0xf0, 0xff00ff00, 0, 255, {0,0,4,4});
				surfaces.defer_copy(source, {0,0,2,4}, target, 2, 0);
			}
			const auto original = surfaces.status(target);
			const auto before = surfaces.transaction_statistics().submitted;
			const auto original_reads = ram->reads;
			conditional.predicate = off;
			executor.execute(left, cold_targets, conditional);
			require(unchanged(surfaces.status(target), original) && surfaces.status(target).deferred_copies == original.deferred_copies &&
				ram->reads == original_reads && surfaces.transaction_statistics().submitted == before,
				"Ready false report touched an unprepared target");
			conditional.predicate = on;
			executor.execute(left, cold_targets, conditional);
			require(surfaces.transaction_statistics().submitted == before + 1 && !surfaces.status(target).spilled &&
				surfaces.status(target).deferred_copies == 0 &&
				surfaces.status(target).content_generation == original.content_generation + (profile == 2 ? 3u : profile == 1 ? 1u : 2u),
				"True report split target initialization from its draw");
			verify(ctx, surfaces, target, {255,0,0,255}, {0,0,2,4},
				profile ? std::array{0u,255u,0u,255u} : std::array{17u,34u,51u,255u});
		}
		context ordered(2, 64, std::chrono::seconds(5), context::recording_overlap::explicit_predecessor);
		auto ordered_budget = std::make_shared<heap_budget>(ordered, 16u << 20);
		auto ordered_ram = std::make_shared<memory>();
		ordered.set_buffer_budget(ordered_budget);
		surface_cache ordered_surfaces(ordered, ordered_ram, 16u << 20, ordered_budget);
		guest_texture_cache ordered_textures(ordered, ordered_ram,
			[](unsigned offset, unsigned, std::size_t) { return offset; }, 1u << 20, ordered_budget);
		draw_executor ordered_executor(ordered, ordered_surfaces, ordered_textures);
		query_pool ordered_queries(ordered, 1, 2, ordered_budget);
		draw_targets ordered_targets;
		if (mirror)
		{
			ordered_targets.depth = ordered_surfaces.acquire({0x21000,{4,4,1,surface_depth::z24s8_uint,false},16,{}});
			ordered_surfaces.clear(ordered_targets.depth, 3, 0, 0x40000000, 255, {0,0,4,4});
		}
		draw_execution_state pending_draw;
		pending_draw.queries = &ordered_queries;
		for (const bool deferred : {false, true})
		for (const bool visible : {false, true})
		{
			const auto address = 0x38000u + unsigned(visible) * 0x1000u + unsigned(deferred) * 0x2000u;
			ordered_targets.colors[0] = ordered_surfaces.acquire({address, {4,4,1,surface_color::argb8,false},16,{}});
			const auto target = ordered_targets.colors[0];
			const auto original = ordered_surfaces.status(target);
			gate_guard gate{[ordered.device() newSharedEvent]};
			ordered.begin();
			ordered.wait_event(gate.event, 1);
			pending_draw.predicate = ordered_queries.capture_report_predicate({{}, unsigned(visible), true});
			require(pending_draw.predicate.has_value(), "Pending cold target could not capture a report");
			const auto producer = ordered.submit();
			const auto submitted = ordered_surfaces.transaction_statistics().submitted;
			ordered_surfaces.set_wait_timeout(std::chrono::milliseconds(1));
			bool timed_out = false;
			try
			{
				if (deferred)
				{
					auto batch = ordered_surfaces.batch_draws(64, surface_cache::predicated_draw_completion::deferred);
					ordered_executor.execute(left, ordered_targets, pending_draw);
					batch.submit();
				}
				else
					ordered_executor.execute(left, ordered_targets, pending_draw);
			}
			catch (const submission_wait_timeout&) { timed_out = true; }
			if (deferred && !timed_out)
			{
				bool read_waited = false;
				try { (void)ordered_surfaces.cpu_read({address, 64}); }
				catch (const submission_wait_timeout&) { read_waited = true; }
				require(read_waited && !ordered_surfaces.poll_submitted(),
					"CPU read or completion poll crossed the pending predicated draw");
			}
			ordered_surfaces.set_wait_timeout(std::chrono::seconds(30));
			require(timed_out != deferred && !producer.ready() && ordered_surfaces.status(target).pending &&
				!ordered_surfaces.status(target).initialized && ordered_surfaces.transaction_statistics().submitted == submitted + 1,
				"Pending report prevented recording the private target load and draw");
			gate.open();
			if (deferred)
			{
				ordered.wait_idle();
				require(ordered_surfaces.poll_submitted() && !ordered_surfaces.status(target).pending,
					"Completed predicated draw did not retire when polled");
			}
			else
				ordered_surfaces.synchronize();
			if (visible)
				verify(ordered, ordered_surfaces, target, {255,0,0,255}, {0,0,2,4}, {0,0,0,0});
			else
				require(unchanged(ordered_surfaces.status(target), original), "Pending false report published the target load");
		}

		draw_targets clause_targets = targets;
		const auto clause_target = surfaces.acquire({0x3d000, {4,4,1,surface_color::argb8,false},16,{}});
		clause_targets.colors[0] = clause_target;
		surfaces.clear(clause_target, 0xf0, 0xff112233, 0, 255, {0,0,4,4});
		const auto clause_generation = surfaces.status(clause_target).content_generation;
		const auto submissions = surfaces.transaction_statistics().submitted;
		gate_guard first_gate{[ctx.device() newSharedEvent]}, last_gate{[ctx.device() newSharedEvent]};
		test::pass_clause clause;
		clause.pass_count = 2;
		test::clause_context clause_context;
		draw_execution_state clause_draw;
		clause_draw.queries = &queries;
		clause_draw.predicate = on;
		auto clause_batch = surfaces.batch_draws(64, surface_cache::predicated_draw_completion::deferred);
		const auto passes = drive_draw_clause(clause, &clause_context, [&](unsigned pass, unsigned)
		{
			require(clause_context.dependencies.size() == pass + 1 &&
				surfaces.status(clause_target).content_generation == clause_generation + pass,
				"Predicated pass observed stale dependency or target state");
			clause_draw.waits = {{pass ? last_gate.event : first_gate.event, 1}};
			executor.execute(pass ? right : left, clause_targets, clause_draw);
		}, 4, clause_cleanup_owner::rsx_thread_end, true, [&]
		{
			bool waited = false;
			try { surfaces.synchronize(std::chrono::milliseconds(1)); }
			catch (const submission_wait_timeout&) { waited = true; }
			require(waited && !surfaces.poll_submitted() && clause_context.dependencies == std::vector<unsigned>{0},
				"Predicated dependency ran while the preceding pass was gated");
			first_gate.open();
			surfaces.synchronize();
		});
		clause_batch.submit();
		require(passes == 2 && !surfaces.poll_submitted() && !clause_context.cleanups &&
			surfaces.transaction_statistics().submitted == submissions + 2 &&
			surfaces.status(clause_target).content_generation == clause_generation + 1,
			"Predicated clause waited for its last pass or published it early");
		clause.post_execute_cleanup(&clause_context);
		last_gate.open();
		ctx.wait_idle();
		require(surfaces.poll_submitted() && surfaces.status(clause_target).content_generation == clause_generation + 2,
			"Final predicated pass did not publish through polling");
		require(clause_context.cleanups == 1, "Predicated clause cleanup ran more than once");
		verify(ctx, surfaces, clause_target, {255,0,0,255}, {0,0,2,4}, {0,255,0,255});

		surfaces.clear(clause_target, 0xf0, 0xff112233, 0, 255, {0,0,4,4});
		draw_execution_state direct;
		direct.queries = &queries;
		direct.predicate = queries.submit_report_predicate({{}, 1, true});
		require(direct.predicate && direct.predicate->settled_cpu_result(ctx) == true,
			"Completed CPU-known true report was not recognized");
		const auto direct_before = surfaces.transaction_statistics();
		const auto direct_generation = surfaces.status(clause_target).content_generation;
		auto direct_batch = surfaces.batch_draws();
		queries.begin(0, true);
		executor.execute(left, clause_targets, direct);
		executor.execute(right, clause_targets, direct);
		queries.end(0);
		require(surfaces.transaction_statistics().submitted == direct_before.submitted &&
			surfaces.status(clause_target).content_generation == direct_generation,
			"Known true report submitted the two draws separately");
		direct_batch.finish();
		const auto direct_after = surfaces.transaction_statistics();
		require(direct_after.submitted == direct_before.submitted + 1 &&
			direct_after.batched_draws == direct_before.batched_draws + 2 &&
			surfaces.status(clause_target).content_generation == direct_generation + 2 && queries.consume(0) == 16,
			"Known true report did not use the direct native batch or lost visibility");
		verify(ctx, surfaces, clause_target, {255,0,0,255}, {0,0,2,4}, {0,255,0,255});
	}
	if (gate_stages) return;
	std::cout<<"Productive separated flat draws:normal andstored-depth-mirror paths,flat-only quad source3,18owned draws includingtwo8drawnativebatches;fresh red/green flat results retain bothscissor halves,all64checked RGBA8pixels exact,onebatch each passed\n";
	std::cout<<"Native executor conditional draws: normal and depth-bounds packets, false surface rollback, true publication, exact pixels/visibility and no flat dispatch passed\n";
	std::cout<<"Native predicated labels: ten normal/depth-bounds profiles, timeout ordering, exact guest bytes, failed producer refusal and flip conflict passed\n";
	std::cout<<"Native predicated draw retains disjoint deferred readers and their original pixels across false and true writes\n";
	std::cout<<"Native predicated overlap: four dirty/deferred profiles preserve guest state on false and compose exact pixels on true, with pending-report timeouts and cold targets checked\n";
	std::cout<<"Conditional target loads: six cold/spilled/deferred profiles used one submission on true and no work on ready false; four pending reports retained or published exact target state\n";
	std::cout<<"Deferred conditional draws: four normal/depth-bounds profiles returned behind the report gate; CPU reads waited and polling published true or rolled back false\n";
	std::cout<<"Deferred conditional clauses: two GPU profiles waited before later dependencies, returned with the final pass gated and retained both exact scissor halves\n";
	std::cout<<"Known false draws: normal and depth-bounds skipped all draw preparation and GPU submission; pending and failed predecessors remained ordered\n";
	std::cout<<"Known true draws: normal and depth-bounds used one direct batch for two passes; exact pixels, visibility and predecessor errors preserved\n";
}
void verify_owned_inline_retirement(unsigned route_selection=0)
{
	using namespace std::chrono_literals;
	unsigned cases=0,timeouts=0,cancelled=0,abandoned=0,captures=0;std::size_t checked=0;
	for(unsigned routing=0;routing<29;++routing)for(bool mirror:{false,true})for(unsigned samples:{1u,2u,4u})for(bool cold:{false,true})for(unsigned outcome=0;outcome<4;++outcome)
	@autoreleasepool
	{
		if((route_selection==1&&routing!=1)||(route_selection==2&&(routing<2||routing>=4))||
			(route_selection==3&&(routing<4||routing>=6))||(route_selection==4&&(routing<6||routing>=8))||
			(route_selection==5&&(routing<8||routing>=10))||(route_selection==6&&(routing<10||routing>=14))||
			(route_selection==7&&(routing<14||routing>=18))||(route_selection==8&&(routing<18||routing>=24))||
			(route_selection==9&&routing<24))continue;
		try
		{
			context ctx(2);auto ram=std::make_shared<memory>();auto budget=std::make_shared<heap_budget>(ctx,8u<<20);ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx,ram,8u<<20,budget);guest_texture_cache textures(ctx,ram,[](unsigned a,unsigned,std::size_t){return a;},1u<<20,budget);
			draw_executor executor(ctx,surfaces,textures);draw_targets targets;
			const auto factors=samples==1?sample_layout{1,1}:rsx_sample_layout(samples);
			const auto coverage=plan_guest_surface(0x1000,4*factors.x,4*factors.y,4,16*factors.x);
			targets.colors[0]=surfaces.acquire({0x1000,{4,4,samples,surface_color::argb8,false},16*factors.x,{}});
			if(!cold)surfaces.clear(targets.colors[0],0xf0,0xff112233,0,255,{0,0,4,4});
			if(mirror){targets.depth=surfaces.acquire({0x10000,{4,4,samples,surface_depth::z24s8_uint,false},16*factors.x,{}});surfaces.clear(targets.depth,3,0,0x40000000,255,{0,0,4,4});}
			test::pass_state source;source.registers=registers().registers;source.current_draw_clause.command=2;source.base_index=0;
			source.bounds=mirror;source.bounds_min=0;source.bounds_max=1;
			std::vector<std::byte> positions;for(float v:{-1.f,-1.f,.5f,1.f,3.f,-1.f,.5f,1.f,-1.f,3.f,.5f,1.f})for(unsigned shift:{24u,16u,8u,0u})positions.push_back(std::byte(std::bit_cast<unsigned>(v)>>shift));
			source.current_draw_clause.inline_vertex_array.resize(12);std::memcpy(source.current_draw_clause.inline_vertex_array.data(),positions.data(),positions.size());
			test::pass_processor processor;processor.descriptors[0]={16u|(1u<<8)|(2u<<24)|(4u<<27),0x60000000u};
			test::pass_block block;test::pass_layout layout;layout.interleaved_blocks.push_back(&block);test::pass_metadata metadata;
			auto vertex=vertex_program();auto bytes=program(false);test::fragment_program fragment;fragment.pointer=bytes.data();fragment.ucode_length=bytes.size();fragment.ctrl=rsx_fragment_float32;
			if(routing)
			{
				constexpr unsigned input=0x1b02;
				vertex.data[4]=(63u<<15)|(7u<<10);
				vertex.data[5]=1u<<27;
				vertex.data[6]=input>>11;
				vertex.data[7]=((input&2047u)<<21)|(15u<<17)|(3u<<7)|1u;
			}
			if(routing>=2&&routing<8)
			{
				const bool rounded=routing>=4&&routing<6,added=routing>=6,invalid=routing%2;
				const std::array<unsigned,4> scalar{(63u<<15)|(7u<<10)|(1u<<28)|(1u<<24)|1u,1u<<27,0x1b02u>>11,((0x1b02u&2047u)<<21)|(15u<<17)|(3u<<7)|((rounded||added?8u:7u)<<2)|1u};
				const std::array<unsigned,4> address{(1u<<15)|(7u<<10),((rounded?23u:13u)<<22)|(0x1b03u>>9),(0x1b03u&511u)<<23,((added?5u:4u)<<13)|(31u<<2)|(63u<<7)};
				source.transform_constants[0][1]=std::bit_cast<unsigned>(added?(invalid?4.f:-4.f):rounded?(invalid?7.5f:-7.5f):(invalid?16.f:-7.f));
				if(added)source.transform_constants[0][3]=std::bit_cast<unsigned>(invalid?4.f:-4.f);
				vertex.data.resize((invalid?16:12)+(added?4:0));
				if(invalid)
				{
					const auto ordinary=vertex_program();std::copy_n(ordinary.data.begin()+4,4,vertex.data.begin()+4);vertex.data[7]&=~1u;
				}
				const unsigned first=invalid?8:4;
				std::copy(address.begin(),address.end(),vertex.data.begin()+first);
				if(added)
				{
					const std::array<unsigned,4> sum{(1u<<15)|(1u<<24)|(7u<<10),24u<<22,0,(4u<<13)|(31u<<2)|(63u<<7)};
					std::copy(sum.begin(),sum.end(),vertex.data.begin()+first+4);
				}
				std::copy(scalar.begin(),scalar.end(),vertex.data.begin()+first+4+(added?4:0));
				vertex.instruction_mask.set(2);if(invalid||added)vertex.instruction_mask.set(3);if(invalid&&added)vertex.instruction_mask.set(4);
			}
			if(routing>=8&&routing<18)
			{
				using words=std::array<unsigned,4>;
				const auto operand=[](words& w,unsigned slot,unsigned packed)
				{
					if(slot==0){w[1]|=packed>>9;w[2]|=(packed&511u)<<23;}
					else if(slot==1)w[2]|=packed<<6;
					else{w[2]|=packed>>11;w[3]|=(packed&2047u)<<21;}
				};
				const bool subnormal=routing==10||routing==14,lit=routing>=15;
				const std::array<unsigned,10> opcodes{5,6,2,3,4,13,14,7,7,7};
				const std::array<float,10> values{1,2,std::numeric_limits<float>::max(),-2,-4,0x1p-64f,-127,1,1,std::bit_cast<float>(1u)},expected{1,1,0,-.5f,.5f,-64,0,0,std::numeric_limits<float>::infinity(),4};
				words math{(63u<<15)|(1u<<30)|(7u<<10)|(subnormal?1u<<14:0u),opcodes[routing-8]<<27,0,(15u<<17)|(3u<<7)|(31u<<2)};operand(math,2,lit?0x1b03u:3u);
				words compare{(4u<<15)|(1u<<30)|(7u<<10),((routing==16?11u:16u)<<22)|(1u<<12),0,(15u<<13)|(63u<<7)|(31u<<2)};operand(compare,0,lit?0xaa0d:0xff0d);operand(compare,1,3u);
				words position{(63u<<15)|(1u<<30)|(7u<<10),2u<<22,0,(15u<<13)|(63u<<7)|1u};operand(position,0,0x1b02);operand(position,1,0x1b11);
				source.transform_constants[0][0]=std::bit_cast<unsigned>(values[routing-8]);for(auto& word:source.transform_constants[1])word=std::bit_cast<unsigned>(expected[routing-8]);
				if(lit){source.transform_constants[0][1]=std::bit_cast<unsigned>(routing==15?0.f:2.f);source.transform_constants[0][3]=std::bit_cast<unsigned>(routing==16?256.f:routing==15?1.f:2.f);source.transform_constants[0][2]=0x7fc12345u;}
				vertex.data.resize(subnormal?20:16);std::copy(math.begin(),math.end(),vertex.data.begin()+4);
				if(subnormal)
				{
					// Observe a subnormal scalar result through the bit-classified
					// condition register; a floating SEQ could flush both sides.
					const words branch{2u<<10,9u<<27,0,(4u<<29)|(31u<<2)},end{0,0,0,1u|(31u<<2)};
					const auto ordinary=vertex_program();std::copy(branch.begin(),branch.end(),vertex.data.begin()+8);std::copy_n(ordinary.data.begin()+4,4,vertex.data.begin()+12);std::copy(end.begin(),end.end(),vertex.data.begin()+16);vertex.instruction_mask.set(4);
				}
				else{std::copy(compare.begin(),compare.end(),vertex.data.begin()+8);std::copy(position.begin(),position.end(),vertex.data.begin()+12);}
				vertex.instruction_mask.set(2);vertex.instruction_mask.set(3);
			}
			if(routing>=18&&routing<24)
			{
				using words=std::array<unsigned,4>;
				constexpr std::array opcodes{11u,12u,16u,18u,19u,20u};
				const bool expected=routing==18||routing==21||routing==23;
				words math{(3u<<15)|(1u<<30)|(1u<<14)|(1u<<29)|(7u<<10),opcodes[routing-18]<<22,
					(3u<<23)|(0x5503u<<6),(15u<<13)|(63u<<7)|(31u<<2)};
				const words branch{(expected?2u:5u)<<10,9u<<27,0,(4u<<29)|(31u<<2)},end{0,0,0,1u|(31u<<2)};
				// Every old native comparison loses this magnitude against zero.
				source.transform_constants[0][0]=routing<20?0x80000001u:1u;source.transform_constants[0][1]=0;
				const auto ordinary=vertex_program();vertex.data.resize(20);
				std::copy(math.begin(),math.end(),vertex.data.begin()+4);std::copy(branch.begin(),branch.end(),vertex.data.begin()+8);
				std::copy_n(ordinary.data.begin()+4,4,vertex.data.begin()+12);std::copy(end.begin(),end.end(),vertex.data.begin()+16);
				vertex.instruction_mask.set(2);vertex.instruction_mask.set(3);vertex.instruction_mask.set(4);
			}
			if(routing>=24)
			{
				using words=std::array<unsigned,4>;
				constexpr std::array opcodes{9u,10u,14u,15u,22u};
				words math{(3u<<15)|(1u<<30)|(1u<<14)|(1u<<29)|(7u<<10),opcodes[routing-24]<<22,
					(3u<<23)|(0x5503u<<6),(15u<<13)|(63u<<7)|(31u<<2)};
				const words branch{2u<<10,9u<<27,0,(4u<<29)|(31u<<2)},end{0,0,0,1u|(31u<<2)};
				source.transform_constants[0][0]=routing==24||routing==27?0x80000001u:1u;source.transform_constants[0][1]=0;
				const auto ordinary=vertex_program();vertex.data.resize(20);
				std::copy(math.begin(),math.end(),vertex.data.begin()+4);std::copy(branch.begin(),branch.end(),vertex.data.begin()+8);
				std::copy_n(ordinary.data.begin()+4,4,vertex.data.begin()+12);std::copy(end.begin(),end.end(),vertex.data.begin()+16);
				vertex.instruction_mask.set(2);vertex.instruction_mask.set(3);vertex.instruction_mask.set(4);
			}
			draw_capture_limits limits;limits.vertex_stream_bytes=48;limits.sparse_sources=[&](const vertex_stream_range& range)
			{
				require(range.source_vertices&&*range.source_vertices==std::vector<unsigned>{0,1,2},"Owned inline capture lost its exact source vertices");
				vertex_sparse_inputs input;input.attribute_mask=1;input.attributes[0]={processor.descriptors[0],false,0,std::as_bytes(std::span{source.current_draw_clause.inline_vertex_array})};return input;
			};
			const auto gate=[ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};
			draw_execution_state execution;execution.waits={{gate,1}};
			const auto before=surfaces.status(targets.colors[0]);const auto submissions=surfaces.transaction_statistics().submitted;
			std::optional<submission> token;test::clause_context clause_context;
			surfaces.set_wait_timeout(1ms);
			{
				auto batch=surfaces.batch_draws();
				require(drive_draw_clause(source.current_draw_clause,&clause_context,[&](unsigned pass,unsigned flags)
				{
					require(!pass&&flags==4,"Owned inline draw lost first-pass state");
					const auto packet=rsx_draw_packet::capture(source,processor,layout,metadata,vertex,fragment,0,provoking_vertex::last,
						[](const auto&,unsigned,unsigned)->std::pair<unsigned,unsigned>{throw std::runtime_error("Owned inline entered legacy input writer");},limits);
					executor.execute(packet,targets,execution);
					token=ctx.recording_submission();
				},4,clause_cleanup_owner::rsx_thread_end)==1&&!clause_context.cleanups,"Owned inline pass cleaned up before submission");
				const bool completion=owned_clause_requires_draw_completion(source.current_draw_clause,true);
				require(!completion,"Owned inline still required GPU completion");
				// The packet, registers, programs, vertex words and capture views
				// all die before the GPU can execute its first instruction.
				std::fill(source.current_draw_clause.inline_vertex_array.begin(),source.current_draw_clause.inline_vertex_array.end(),0xffffffffu);
				std::fill(source.registers.begin(),source.registers.end(),0xffffffffu);std::fill(vertex.data.begin(),vertex.data.end(),0xffffffffu);
				for(auto& constant:source.transform_constants)for(auto& word:constant)word=0xffffffffu;
				std::fill(bytes.begin(),bytes.end(),std::byte{0xff});std::fill(positions.begin(),positions.end(),std::byte{0xff});
				source.current_draw_clause.inline_vertex_array.clear();source.current_draw_clause.inline_vertex_array.shrink_to_fit();
				source.registers.clear();source.registers.shrink_to_fit();vertex.data.clear();vertex.data.shrink_to_fit();
				bytes.clear();bytes.shrink_to_fit();positions.clear();positions.shrink_to_fit();processor.descriptors={};limits.sparse_sources={};
				if(outcome==3)
				{
					bool timed_out=false;try{batch.finish();}catch(const submission_wait_timeout&){timed_out=true;}
					require(timed_out,"Frame-capture boundary did not retain its synchronous wait");++captures;
				}
				else if(outcome!=2)batch.submit();
			}
			source.current_draw_clause.post_execute_cleanup(&clause_context);
			require(clause_context.cleanups==1&&!source.current_draw_clause.rendering,"Owned inline RSX cleanup was not exactly once");
			if(outcome==2)
			{
				require(token&&token->cancelled()&&!surfaces.status(targets.colors[0]).pending&&surfaces.transaction_statistics().submitted==submissions,
					"Abandoned inline recording was submitted or retained");++abandoned;
			}
			else
			{
				require(token&&!token->ready()&&!surfaces.poll_submitted()&&surfaces.status(targets.colors[0]).content_generation==before.content_generation&&
					surfaces.transaction_statistics().submitted==submissions+1,"Owned inline submitted early metadata or waited for its closed GPU gate");
				bool timed_out=false;try{surfaces.synchronize(1ms);}catch(const submission_wait_timeout&){timed_out=true;}
				require(timed_out,"Owned inline lacks actual pending GPU work");++timeouts;
				if(outcome==1)require(token->cancel_pending(),"Owned inline queued cancellation failed");
				gate.signaledValue=1;bool rejected=false;try{surfaces.synchronize(30s);}catch(const std::runtime_error&){rejected=true;}
				require(rejected==(outcome==1),"Owned inline publication/cancellation differs");
				if(outcome==1){++cancelled;ctx.recover_completed_errors();}
			}
			surfaces.set_wait_timeout(30s);
			const bool success=outcome==0||outcome==3;
			require(!surfaces.status(targets.colors[0]).pending&&!surfaces.status(targets.colors[0]).faulted&&
				surfaces.status(targets.colors[0]).content_generation==before.content_generation+(success?1+unsigned(cold):0),"Owned inline output generation/rollback differs");
			surfaces.flush(targets.colors[0]);const auto actual=ram->read_surface(coverage).bytes;
			const std::array<unsigned,4> pixel=success&&routing!=3&&routing!=5&&routing!=7?std::array{255u,64u,128u,192u}:cold?std::array{0u,0u,0u,0u}:std::array{255u,17u,34u,51u};
			for(unsigned i=0;i<actual.size();++i)require(actual[i]==std::byte(pixel[i%4]),"Owned inline output/rollback sample differs after input destruction");
			checked+=actual.size();++cases;
		}
		catch(const std::exception& e){throw std::runtime_error("Owned inline routing="+std::to_string(routing)+" mirror="+std::to_string(mirror)+" samples="+std::to_string(samples)+" cold="+std::to_string(cold)+" outcome="+std::to_string(outcome)+": "+e.what());}
	}
	std::cout<<"Owned inline retirement: "<<cases<<" normal/depth-mirror, 1/2/4-sample, existing/cold-target profiles; "<<timeouts<<" real pending GPU timeouts, "<<cancelled<<" queued cancellations, "<<abandoned<<" abandoned recordings, "<<captures<<" retained frame-capture waits, exactly one RSX cleanup and "<<checked<<" exact output/rollback bytes after packet/register/program/inline data destruction\n";
}

void verify_packed_high_vertices()
{
	context ctx;
	auto ram = std::make_shared<memory>();
	auto budget = std::make_shared<heap_budget>(ctx,8u << 20); ctx.set_buffer_budget(budget);
	surface_cache surfaces(ctx,ram,8u << 20,budget);
	guest_texture_cache textures(ctx,ram,[](unsigned offset,unsigned,std::size_t) { return offset; },1u << 20,budget);
	draw_executor executor(ctx,surfaces,textures);
	draw_targets targets; targets.colors[0] = surfaces.acquire({0x1000,{4,4,1,surface_color::argb8,false},16,{}});
	for (unsigned first : {65535u,1048572u})
	{
		test::pass_state state; state.registers = registers().registers;
		state.current_draw_clause.command = 1; state.current_draw_clause.first = first;
		test::pass_processor processor;
		for (float value : {-1.f,-1.f,.5f,1.f,3.f,-1.f,.5f,1.f,-1.f,3.f,.5f,1.f})
			for (unsigned shift : {24u,16u,8u,0u}) processor.persistent.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
		// The original IDs select positions by modulo3, including the high IDs;
		// no rebased division phase or densely decoded prefix is permitted.
		processor.descriptors[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27),0xa0000000u};
		test::pass_layout layout; test::pass_metadata metadata;
		auto vertex = vertex_program(); auto bytes = program(false);
		test::fragment_program fragment; fragment.pointer = bytes.data(); fragment.ucode_length = bytes.size(); fragment.ctrl = rsx_fragment_float32;
		draw_capture_limits limits; limits.vertex_stream_bytes = 48; limits.geometry.source_vertices = 3;
		limits.sparse_sources = [&](const vertex_stream_range& range)
		{
			require(range.first_vertex == first && range.vertex_count == 3 && range.source_vertices &&
				*range.source_vertices == std::vector<unsigned>{first,first + 1,first + 2},"High-ID packet lost exact source set");
			vertex_sparse_inputs sources; sources.attribute_mask = 1;
			sources.attributes[0] = {processor.descriptors[0],false,0,processor.persistent};
			return sources;
		};
		const auto packet = rsx_draw_packet::capture(state,processor,layout,metadata,vertex,fragment,0,provoking_vertex::last,
			[&](const auto&,unsigned,unsigned) -> std::pair<unsigned,unsigned>
			{
				throw std::runtime_error("Sparse packet entered legacy prefix requirements");
			},limits);
		require(packet.vertices().fetch().first_vertex == first && packet.vertices().fetch().vertex_count == 3 &&
			packet.vertices().persistent().size() == 48 && packet.vertices().source_vertices()->size() == 3,"Sparse owned capture retained a prefix");
		processor.persistent.assign(48,std::byte{0}); state.registers.assign(state.registers.size(),0xffffffffu);
		vertex.data.assign(vertex.data.size(),0xffffffffu); bytes.assign(bytes.size(),std::byte{0xff});
		surfaces.clear(targets.colors[0],0xf0,0xff112233,0,255,{0,0,4,4});
		executor.execute(packet,targets);
		verify(ctx,surfaces,targets.colors[0],{64,128,192,255});
		const auto uploads = ctx.vertex_uploads().statistics();
		for (unsigned repeat = 0; repeat < 32; ++repeat) executor.execute(packet,targets);
		require(ctx.vertex_uploads().statistics().created == uploads.created && ctx.vertex_uploads().statistics().uploaded_bytes == uploads.uploaded_bytes &&
			ctx.vertex_uploads().statistics().reused == uploads.reused + 32,
			"Identical owned high-ID vertex packets uploaded streams again");
		verify(ctx,surfaces,targets.colors[0],{64,128,192,255});
	}
	require(ctx.vertex_uploads().statistics().created == 1 && ctx.vertex_uploads().statistics().uploaded_bytes == 48,
		"Equal captured packed content did not share its exact immutable native upload");
	std::cout << "Direct sparse packed vertex draw:two owned triangles atIDs65535/1048572,three captured source IDs/48bytes without prefix requirements,66draws under8MiB native ledger,exact pixels andoriginal modulo phase;one48byte stream upload,65content hits;mutable inputs destroyed passed\n";
	// Non-modulo positions prove both cropped high-ID windows and sparse
	// 20-bit base wrap through the complete owned packet/native draw boundary.
	for (bool wrapped : {false,true})
	{
		test::pass_state state; state.registers = registers().registers;
		state.current_draw_clause.command = wrapped ? 3 : 1; state.current_draw_clause.first = 65535; state.base_index = 1;
		test::pass_processor processor; test::pass_layout layout; test::pass_metadata metadata;
		if (wrapped) for (unsigned index : {0xffffeu,0xfffffu,0u}) for (unsigned shift : {24u,16u,8u,0u}) processor.index_bytes.push_back(std::byte(index >> shift));
		const std::array<unsigned,3> ids = wrapped ? std::array{1048575u,0u,1u} : std::array{65535u,65536u,65537u};
		const std::array<std::array<float,4>,3> positions{{{-1,-1,.5,1},{3,-1,.5,1},{-1,3,.5,1}}};
		unsigned source_bytes = 0;
		draw_capture_limits limits; limits.vertex_stream_bytes = 60; limits.geometry.source_vertices = 3;
		limits.sparse_sources = [&](const vertex_stream_range& range)
		{
			require(range.source_vertices->size() == 3,"Window/wrapped packet captured excess sources");
			vertex_sparse_inputs sources; sources.attribute_mask = 1;
			sources.attributes[0] = {{16u | (1u << 8) | (2u << 24) | (4u << 27),1u << 29},true,0,{}};
			sources.read_memory = [&](unsigned address,std::span<std::byte> output)
			{
				for (std::size_t b = 0; b < output.size(); ++b)
				{
					const auto element = (address + b) / 16, byte = (address + b) % 16;
					const auto found = std::find(ids.begin(),ids.end(),element);
					require(found != ids.end(),"Window/wrapped capture read an unused VM source");
					const auto value = std::bit_cast<unsigned>(positions[found - ids.begin()][byte / 4]);
					output[b] = std::byte(value >> (8 * (3 - byte % 4))); ++source_bytes;
				}
			};
			return sources;
		};
		auto vertex = vertex_program(); auto bytes = program(false);
		test::fragment_program fragment; fragment.pointer = bytes.data(); fragment.ucode_length = bytes.size(); fragment.ctrl = rsx_fragment_float32;
		const auto packet = rsx_draw_packet::capture(state,processor,layout,metadata,vertex,fragment,0,provoking_vertex::last,
			[](const auto&,unsigned,unsigned) -> std::pair<unsigned,unsigned> { throw std::runtime_error("Window/wrapped capture called prefix requirements"); },limits);
		require(source_bytes == 48 && packet.vertices().persistent().size() == (wrapped ? 60 : 52) &&
			packet.vertices().fetch().attributes[0].storage_mode() == (wrapped ? 2 : 1),"Window/wrapped capture was not compact");
		processor.index_bytes.clear(); state.registers.assign(state.registers.size(),0xffffffffu); bytes.assign(bytes.size(),std::byte{0xff});
		surfaces.clear(targets.colors[0],0xf0,0xff112233,0,255,{0,0,4,4}); executor.execute(packet,targets);
		verify(ctx,surfaces,targets.colors[0],{64,128,192,255});
	}
	std::cout << "Complete owned native draws:non-modulo high-ID window52bytes and20-bit base-wrapped sparse60bytes,exactly48needed VM source bytes per draw,no prefix requirements,all16RGBA8pixels exact passed\n";
}
} // namespace
int main(int argc, char** argv)
{
	@autoreleasepool
	{
		try
		{
			if (argc == 2 && std::string_view(argv[1]) == "--deferred-pipeline-only")
			{
				verify_separated_flat_draws(true);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--selection-rounding-only")
			{
				verify_owned_inline_retirement(9);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--comparisons-only")
			{
				verify_owned_inline_retirement(8);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--ex2-lit-only")
			{
				verify_owned_inline_retirement(7);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--scalar-math-only")
			{
				verify_owned_inline_retirement(6);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--legacy-math-only")
			{
				verify_owned_inline_retirement(5);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--address-add-only")
			{
				verify_owned_inline_retirement(4);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--address-rounding-only")
			{
				verify_owned_inline_retirement(3);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--indexed-vertex-output-only")
			{
				verify_owned_inline_retirement(2);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--scalar-dual-draws-only")
			{
				verify_owned_inline_retirement(true);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--owned-inline-retirement-only")
			{
				verify_owned_inline_retirement();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--geometry-admission-only")
			{
				verify_versioned_upload_batch();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--conditional-texture-only")
			{
				verify_predicated_texture_preparation();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--cold-null-texture-only")
			{
				verify_owned_texture_boundary();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--inline-assembly-only")
			{
				verify_inline_assemblies();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--deferred-draw-retirement-only")
			{
				verify_deferred_draw_retirement();
				verify_submitted_texture_independence();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--completed-upload-dependencies-only")
			{
				verify_completed_inline_upload_dependencies();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--inline-upload-dependencies-only")
			{
				verify_inline_upload_dependencies();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--cold-framebuffer-only")
			{
				verify_cold_framebuffer();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--spilled-framebuffer-only")
			{
				verify_spilled_framebuffer();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--deferred-framebuffer-only")
			{
				verify_deferred_framebuffer();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--draw-deferred-readers-only")
			{
				verify_draw_deferred_readers();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--resident-framebuffer-only")
			{
				verify_resident_framebuffer();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--inline-native-cast-only")
			{
				verify_inline_native_casts();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--redirected-depth-only")
			{
				for (unsigned samples : {1u, 2u, 4u})
					verify_pending_draw_capture(upload_outcome::inline_color, {}, {inline_cast_source::redirected_depth, false, samples});
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--sample-raster-only")
			{
				for (unsigned samples : {2u, 4u})
					verify_pending_draw_capture(upload_outcome::inline_color, {}, {inline_cast_source::redirected_depth, false, samples, true});
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--split-planes-only")
			{
				for (unsigned samples : {2u, 4u})
					verify_pending_draw_capture(upload_outcome::inline_color, {}, {inline_cast_source::redirected_depth, false, samples, false, true});
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--inline-depth-vtc-only")
			{
				verify_inline_depth_and_vtc();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--inline-texture-draw-only")
			{
				verify_inline_texture_draws();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--cold-border-draw-only")
			{
				verify_pending_draw_capture(upload_outcome::border_helper);
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--pending-texture-draw-only")
			{
				verify_pending_texture_draws();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--completed-draw-waits-only")
			{verify_ready_textured_draws(true);return 0;}
			if (argc == 2 && std::string_view(argv[1]) == "--ready-textured-draw-only")
			{
				verify_ready_textured_draws();
				return 0;
			}
			if (argc == 2 && std::string_view(argv[1]) == "--conditional-only")
			{
				verify_separated_flat_draws();
				return 0;
			}
			require(argc == 1, "Unknown native draw executor test selector");
			verify_versioned_upload_batch();
			verify_owned_texture_boundary();
			verify_ready_texture_boundary();
			verify_ready_textured_draws();
			verify_predicated_texture_preparation();
			verify_texture_writer_boundary();
			verify_pending_texture_draws();
			verify_inline_texture_draws();
			verify_inline_depth_and_vtc();
			verify_inline_native_casts();
			for (unsigned samples : {1u, 2u, 4u})
				verify_pending_draw_capture(upload_outcome::inline_color, {}, {inline_cast_source::redirected_depth, false, samples});
			for (unsigned samples : {2u, 4u})
				verify_pending_draw_capture(upload_outcome::inline_color, {}, {inline_cast_source::redirected_depth, false, samples, true});
			for (unsigned samples : {2u, 4u})
				verify_pending_draw_capture(upload_outcome::inline_color, {}, {inline_cast_source::redirected_depth, false, samples, false, true});
			verify_inline_assemblies();
			verify_deferred_draw_retirement();
			verify_submitted_texture_independence();
			verify_inline_upload_dependencies();
			verify_completed_inline_upload_dependencies();
			verify_cold_framebuffer();
			verify_spilled_framebuffer();
			verify_deferred_framebuffer();
			verify_draw_deferred_readers();
			verify_resident_framebuffer();
			verify_versioned_job_boundaries();
			verify_versioned_job_callback_ownership();
			verify_versioned_job_orphan();
			verify_native_snapshot_batch();
			verify_upload_batch_publication();
			verify_native_cast_batch_publication();
			verify_owned_raster();
			verify_owned_inline_retirement();
			verify_packed_high_vertices();
			verify_separated_flat_draws();
			context ctx{1};
			auto ram = std::make_shared<memory>();
			auto budget = std::make_shared<heap_budget>(ctx, 8 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, ram, 8 << 20, budget);
			guest_texture_cache textures(ctx, ram, [](unsigned offset, unsigned, std::size_t)
										 { return offset * 16; }, 1 << 20, budget);
			draw_executor executor(ctx, surfaces, textures);
			{
				const auto reads = ram->reads;
				const auto uploads = textures.statistics().uploads;
				const auto budget_before = budget->statistics();
				null_texture_family nulls;
				@autoreleasepool { nulls = textures.null_images(); }
				std::size_t null_bytes = 0;
				for (const auto& image : nulls)
				{
					require(image.texture.storageMode == MTLStorageModeShared && image.texture.heap.storageMode == MTLStorageModeShared,
						"CPU-initialized null texture must have budgeted shared backing");
					null_bytes += image.texture.heap.size;
				}
				require(budget->statistics().reserved_bytes == budget_before.reserved_bytes + null_bytes &&
							budget->statistics().live_heaps == budget_before.live_heaps + 4,
						"Native null family bypassed shared heap accounting");
				{
					auto tiny = std::make_shared<heap_budget>(ctx, nulls[0].texture.heap.size);
					unsigned mapped = 0;
					guest_texture_cache constrained(ctx, ram, [&](unsigned, unsigned, std::size_t) -> unsigned
													{ ++mapped; throw std::runtime_error("Null allocation entered VM mapping"); }, 1 << 20, tiny);
					for (unsigned attempt = 0; attempt < 2; ++attempt)
					{
						bool refused = false;
						@autoreleasepool
						{
							try
							{
								(void)constrained.null_images();
							}
							catch (const heap_budget_error&)
							{
								refused = true;
							}
						}
						require(refused && !mapped && constrained.statistics().null_initializations == 0 && tiny->statistics().reserved_bytes == 0,
								"Partial native null allocation published family or leaked backing budget");
					}
					std::cout << "Native null family charges four actual heaps;partial-budget refusal rolls back without publication or VM mapping passed\n";
				}
				auto poison = registers();
				std::fill(poison.registers.begin(), poison.registers.end(), 0xffffffffu);
				poison.registers[0x90c / 4] = poison.registers[0x1a0c / 4] = 0x7fffffffu;
				const auto vertex_words = vertex_texture_snapshot::capture(poison.registers, 1);
				const auto fragment_words = fragment_texture_snapshot::capture(poison.registers, 1);
				sampler_cache samplers(ctx);
				unsigned callbacks = 0;
				const auto forbidden = [&](unsigned, const fragment_texture_registers&) -> resolved_fragment_texture
				{
					++callbacks;
					throw std::runtime_error("Disabled null binding entered guest resolver");
				};
				fragment_inputs fragment_inputs;
				for (unsigned type = 0; type < 4; ++type)
				{
					const auto dimension = static_cast<vertex_texture_dimension>(type);
					vertex_program_options vertex_options;
					vertex_options.textures.fill(dimension);
					vertex_inputs vertex_inputs;
					apply_vertex_texture_snapshot(vertex_words, forbidden, ctx, samplers, vertex_options, vertex_inputs, &nulls);
					fragment_program_options fragment_options;
					fragment_options.textures.fill(dimension);
					fragment_options.shadow_textures = fragment_options.depth_textures = fragment_options.redirected_textures = fragment_options.multisampled_textures = 1;
					apply_fragment_texture_snapshot(fragment_words, forbidden, ctx, samplers, fragment_options, fragment_inputs, {}, &nulls);
					require(vertex_inputs.textures[0] == nulls[type].texture && fragment_inputs.textures[0] == nulls[type].texture &&
								!(fragment_options.shadow_textures | fragment_options.depth_textures | fragment_options.redirected_textures | fragment_options.multisampled_textures),
							"Disabled typed null binding retained depth/compare/MSAA semantics");
				}
				const auto epoch = executor.prepare_versioned_textures(vertex_words, fragment_words);
				require(!callbacks && ram->reads == reads && textures.statistics().uploads == uploads &&
							epoch.revisions.empty() && epoch.current(vertex_words, fragment_words) &&
							epoch.textures.null_images()[3].texture == nulls[3].texture,
						"Disabled null epoch reached guest memory or failed ownership");
				compiler compile(ctx);
				constexpr std::string_view source = R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void sample_nulls(device float4* out [[buffer(0)]],texture2d<float> one [[texture(0)]],
    texture2d<float> two [[texture(1)]],texture3d<float> three [[texture(2)]],texturecube<float> cube [[texture(3)]],sampler s [[sampler(0)]]) {
    out[0]=one.sample(s,float2(.5f),level(0));out[1]=two.sample(s,float2(.5f),level(0));out[2]=three.sample(s,float3(.5f),level(0));
    float3 directions[6]={float3(1,0,0),float3(-1,0,0),float3(0,1,0),float3(0,-1,0),float3(0,0,1),float3(0,0,-1)};
    for(uint i=0;i<6;++i)out[3+i]=cube.sample(s,directions[i],level(0));
}
)metal";
				const auto library = compile.library(source);
				auto descriptor = [MTL4ComputePipelineDescriptor new];
				descriptor.computeFunctionDescriptor = library_function(library, @"sample_nulls");
				const auto pipeline = compile.compute(descriptor);
				const auto output = make_buffer(ctx, 9 * 16, buffer_storage::shared);
				std::memset(output.contents, 0xa5, output.length);
				ctx.begin();
				ctx.keep_alive(pipeline);
				bindings inputs;
				inputs.set_buffer(0, output);
				for (unsigned type = 0; type < 4; ++type)
					inputs.set_texture(type, nulls[type].texture);
				inputs.set_sampler(0, fragment_inputs.samplers[0]);
				const auto table = inputs.snapshot(ctx);
				const auto encoder = ctx.compute();
				[encoder setComputePipelineState:pipeline];
				[encoder setArgumentTable:table];
				[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
				ctx.submit().wait();
				for (const auto byte : map_buffer(output, 0, output.length))
					require(byte == std::byte{}, "Typed native null sample is not transparent black");
				textures.clear();
				require(textures.null_images()[3].texture == nulls[3].texture, "VM cache clear unnecessarily rebuilt native null images");
				textures.clear_null_images();
				require(epoch.textures.null_images()[3].texture == nulls[3].texture, "Cache clear released held null epoch");
				std::cout << "Native disabled nulls:1D/2D/3D/all six cube faces sample exact transparent black;invalid unused words,zero guest callbacks and held epoch after cache clear passed\n";
			}
			{
				auto unrelated = std::make_shared<memory>();
				surface_cache foreign(ctx, unrelated, 8 << 20, budget);
				bool refused = false;
				try
				{
					draw_executor invalid(ctx, foreign, textures);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused, "Executor accepted unrelated VM cache adapters");
			}
			const auto target = surfaces.acquire({0x1000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
			const auto depth = surfaces.acquire({0x2000, {4, 4, 1, surface_depth::z24s8_uint, false}, 16, {}});
			draw_targets targets;
			targets.colors[0] = target;
			targets.depth = depth;
			draw_request request;
			std::vector<std::byte> raw_indices;
			for (const unsigned index : {0xffffcu, 0u, 4u})
				for (const unsigned shift : {24u, 16u, 8u, 0u})
					raw_indices.push_back(std::byte(index >> shift));
			const auto geometry_plan = plan_indexed_geometry(guest_topology::triangles, raw_indices, 4, 4, {}, {}, provoking_vertex::last);
			request.geometry = geometry_plan.geometry;
			request.fetch.attribute_mask = 1;
			request.fetch.attributes[0] = {16u | (3u << 8) | (2u << 24) | (4u << 27), (1u << 31) | (1u << 29)};
			request.raster.viewport = {0, 0, 4, 4, 0, 1};
			std::vector<std::byte> positions;
			for (float v : {-1.f, -1.f, .5f, 1.f, 3.f, -1.f, .5f, 1.f, -1.f, 3.f, .5f, 1.f})
				for (unsigned shift : {24u, 16u, 8u, 0u})
					positions.push_back(std::byte(std::bit_cast<unsigned>(v) >> shift));
			test::vertex_data_processor vertex_processor;
			vertex_processor.descriptors = request.fetch.attributes;
			vertex_processor.persistent = positions;
			test::vertex_data_state vertex_state;
			vertex_state.transform_constants[511][3] = 0x7fc12345;
			const auto vertex_capture = vertex_data_snapshot::capture(vertex_processor, 0, test::vertex_data_metadata{1},
																	  vertex_state, geometry_plan.streams, {48, 0}, 48);
			vertex_processor.persistent.assign(48, std::byte{0});
			vertex_state.transform_constants = {};
			apply_vertex_data_snapshot(ctx, vertex_capture, request);
			require(request.fetch.vertex_count == 9 && request.geometry.vertices[1].vertex == 4 && request.geometry.vertices[2].vertex == 8,
					"20-bit indexed geometry did not retain sparse/modulo fetch IDs");
			require(request.vertex_uniforms.constant_count == 512 && request.vertex_resources.constants.length == 8192,
					"Full native constant register upload missing");
			const auto constant_bytes = map_buffer(request.vertex_resources.constants, 511 * 16 + 12, 4);
			std::uint32_t last_constant = 0;
			std::memcpy(&last_constant, constant_bytes.data(), 4);
			require(last_constant == 0x7fc12345, "High constant register bits changed during native upload");
			apply_draw_environment(draw_environment_snapshot::capture(test::environment_state{}), 4, 4, request);
			surfaces.clear(target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
			surfaces.clear(depth, 3, 0, 0xffffff42, 255, {0, 0, 4, 4});
			auto state = registers();
			state.registers[0xa74 / 4] = state.registers[0xa70 / 4] = 1;
			state.registers[0x328 / 4] = 1;
			state.registers[0x334 / 4] = 0xab;
			state.registers[0x32c / 4] = 0xf0;
			state.registers[0x344 / 4] = 0x1e01;
			state.registers[0x8c0 / 4] = state.registers[0x8c4 / 4] = (2u << 16) | 1;
			const auto draw = snapshot(state, false, 0);
			const auto generation = surfaces.status(target).content_generation;
			query_pool queries(ctx, 2, 32);
			request.queries = &queries;
			queries.begin(0, true);
			executor.execute(draw, targets, request);
			queries.end(0);
			require(queries.ready(0) && queries.consume(0) == 4, "Executor did not count four surviving scissor/depth/stencil samples");
			std::cout << "Integrated native draw query:four surviving scissor/depth/stencil samples passed\n";
			require(surfaces.status(target).content_generation == generation + 1 && surfaces.status(target).dirty, "Executor did not publish successful target writes");
			verify(ctx, surfaces, target, {64, 128, 192, 255}, {1, 1, 2, 2});
			const auto query_before = queries.statistics();
			queries.begin(1, true);
			for (unsigned draw_index = 0; draw_index < 32; ++draw_index)
			{
				// Restore the same depth/stencil acceptance. Helper clears must
				// neither count visibility nor split the active query interval.
				surfaces.clear(depth, 3, 0, 0xffffff42, 255, {0, 0, 4, 4});
				executor.execute(draw, targets, request);
			}
			queries.end(1);
			require(queries.consume(1) == 128 && queries.statistics().acquired - query_before.acquired == 1 &&
				queries.statistics().initializations - query_before.initializations == 1 &&
				queries.statistics().reused_completed - query_before.reused_completed == 31,
				"Productive draw executor did not accumulate one native query interval across completed draws");
			std::cout << "32 productive executor draws plus helper clears:128 samples,one counter,one zero,31 completed continuations passed\n";
			const auto z = make_buffer(ctx, 1024, buffer_storage::shared), s = make_buffer(ctx, 1024, buffer_storage::shared);
			const std::array uses{cached_surface_use{depth, false}};
			surfaces.draw(uses, [&](context& commands, auto textures)
						  {texture_region whole;whole.size=MTLSizeMake(4,4,1);texture_to_buffer(commands,textures[0],whole,z,0,256,0,texture_plane::depth);texture_to_buffer(commands,textures[0],whole,s,0,256,0,texture_plane::stencil); });
			for (unsigned y = 0; y < 4; ++y)
				for (unsigned x = 0; x < 4; ++x)
				{
					const bool inside = x >= 1 && x < 3 && y >= 1 && y < 3;
					require(static_cast<const float*>(z.contents)[y * 64 + x] == (inside ? .5f : 1.f) && static_cast<const unsigned char*>(s.contents)[y * 256 + x] == (inside ? 0xa2 : 0x42), "Executor depth/stencil transaction output differs");
				}
			verify_executor_bounds(ctx, surfaces, executor, request);
			verify_framebuffer_samples(ctx, surfaces, executor, *ram, positions, raw_indices, vertex_processor.descriptors);
			verify_framebuffer_sample_colors(ctx, surfaces, executor, *ram, positions, raw_indices, vertex_processor.descriptors);
			std::array<std::byte, 64> pixels;
			{
				const auto null_target = surfaces.acquire({0x3d00, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(null_target, 0xf0, 0xff445566, 0, 255, {0, 0, 4, 4});
				draw_targets null_targets;
				null_targets.colors[0] = null_target;
				auto null_request = request;
				null_request.queries = nullptr;
				auto words = registers();
				for (unsigned index = 0; index < 8; ++index)
					words.registers[0x1a00 / 4 + index] = 0xffffffffu;
				words.registers[0x1a0c / 4] = 0x7fffffffu;
				words.registers[0x1840 / 4] = 0xffffffffu;
				const auto disabled = snapshot(words, true, 1);
				const auto reads = ram->reads;
				const auto uploads = textures.statistics().uploads;
				const auto epoch = executor.prepare_textures(disabled.vertex_textures(), disabled.textures());
				const auto initialized = textures.statistics().null_initializations;
				executor.execute(disabled, null_targets, null_request, epoch);
				verify(ctx, surfaces, null_target, {0, 0, 0, 0});
				textures.clear();
				textures.clear_null_images();
				executor.execute(disabled, null_targets, null_request, epoch);
				verify(ctx, surfaces, null_target, {0, 0, 0, 0});
				require(ram->reads == reads && textures.statistics().uploads == uploads && textures.statistics().null_initializations == initialized,
						"Prepared disabled fragment consulted VM or recreated nulls after cache clear");
				std::cout << "Native disabled fragment draw ignores poisoned guest/sampler words,writes transparent black and retains exact null epoch after cache clear without VM reads/reinitialization passed\n";
			}
			{
				draw_executor repeated(ctx, surfaces, textures);
				const auto repeated_target = surfaces.acquire({0x3c00, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				draw_targets repeated_targets;
				repeated_targets.colors[0] = repeated_target;
				auto repeated_request = request;
				repeated_request.queries = nullptr;
				auto original_words = registers();
				const auto original_program = snapshot(original_words, false, 0);
				repeated.execute(original_program, repeated_targets, repeated_request);
				verify(ctx, surfaces, repeated_target, {64, 128, 192, 255});
				const auto initial = repeated.library_statistics();
				const auto initial_pipeline = repeated.pipeline_statistics();
				require(initial_pipeline.created == 1 && initial_pipeline.entries == 1, "First draw did not cache its native pipeline");
				require(initial.created == 2 && initial.entries == 2, "First native draw did not retain both stage libraries");
				require(repeated.vertex_statistics().prewarmed == 1 && repeated.fragment_statistics().prewarmed == 1 &&
					repeated.pipeline_statistics().prewarmed == 1 && repeated.compilation_statistics().admitted == 3,
					"Productive native draw bypassed stage/pipeline prewarm or compiled duplicate jobs");
				auto changed_words = registers();
				const auto changed_literals = snapshot(changed_words, false, 0, false, 0, {0, 0, 1, 1});
				repeated.execute(changed_literals, repeated_targets, repeated_request);
				verify(ctx, surfaces, repeated_target, {0, 0, 255, 255});
				const auto constants_before_return = ctx.constant_uploads().statistics();
				repeated.execute(original_program, repeated_targets, repeated_request);
				verify(ctx, surfaces, repeated_target, {64, 128, 192, 255});
				require(ctx.constant_uploads().statistics().uploaded_bytes == constants_before_return.uploaded_bytes &&
					ctx.constant_uploads().statistics().reused > constants_before_return.reused, "Returning fragment literals were reuploaded");
				require(repeated.library_statistics().created == initial.created && repeated.library_statistics().reused == initial.reused,
						"Repeated draw/literal update recompiled stages or kept stale uniforms");
				require(repeated.compilation_statistics().admitted == 3 && repeated.vertex_statistics().prewarm_reused == 2 &&
					repeated.fragment_statistics().prewarm_reused == 2,
					"Literal changes submitted additional native compilation jobs");
				require(repeated.vertex_statistics().translated == 1 && repeated.fragment_statistics().translated == 1 &&
							repeated.vertex_statistics().compiled == 1 && repeated.fragment_statistics().compiled == 1,
						"Repeated draw/literal changes translated/compiled shader stages again");
				require(repeated.vertex_statistics().lookups == 3 && repeated.fragment_statistics().lookups == 3 &&
					repeated.vertex_statistics().prepared_hits == 3 && repeated.fragment_statistics().prepared_hits == 3 &&
					repeated.pipeline_statistics().fast_hits == 2 && repeated.pipeline_statistics().descriptor_lookups == 1,
					"Executor repeated program lookup or render descriptor construction after preflight");
				std::cout << "Native executor:three draws with changed literal pixels,one lookup per stage/draw,three prepared hits per stage,two pipeline fast hits and one descriptor lookup passed\n";
				require(repeated.pipeline_statistics().created == 1 && repeated.pipeline_statistics().reused == 2, "Literal changes recreated native pipeline/depth state");
				std::cout << "Native executor reuses pipeline and depth state across literal changes with refreshed pixels passed\n";
				std::cout << "Native executor reuses both stage libraries across literal changes while publishing refreshed blue/original pixels passed\n";
			}
			for (unsigned i = 0; i < 16; ++i)
				for (unsigned c = 0; c < 4; ++c)
					pixels[i * 4 + c] = std::byte(std::array{255u, 20u, 40u, 60u}[c]);
			ram->write({0x3000, 64}, pixels);
			state = registers();
			texture_registers(state, 0x300);
			auto textured = snapshot(state, true, 1);
			targets.depth.reset();
			executor.execute(textured, targets, request);
			verify(ctx, surfaces, target, {20, 40, 60, 255});
			{
				const auto versioned = executor.prepare_versioned_textures(textured.vertex_textures(), textured.textures());
				require(versioned.revisions.size() == 1 && versioned.current(textured.vertex_textures(), textured.textures()), "Prepared draw ticket lost referenced VM source");
				auto all_state = registers();
				texture_registers(all_state, 0x300);
				set_vertex_texture_registers(all_state, 0x300);
				for (unsigned unit = 1; unit < 4; ++unit)
					for (unsigned word = 0; word < 8; ++word)
						all_state.registers[0x900 / 4 + unit * 8 + word] = all_state.registers[0x900 / 4 + word];
				for (unsigned unit = 1; unit < 16; ++unit)
				{
					for (unsigned word = 0; word < 8; ++word)
						all_state.registers[0x1a00 / 4 + unit * 8 + word] = all_state.registers[0x1a00 / 4 + word];
					all_state.registers[0x1840 / 4 + unit] = all_state.registers[0x1840 / 4];
				}
				for (unsigned unit = 0; unit < 4; ++unit)
					all_state.registers[0x900 / 4 + unit * 8] = 0x80000000u | (0x8000 + unit * 0x10);
				for (unsigned unit = 0; unit < 16; ++unit)
					all_state.registers[0x1a00 / 4 + unit * 8] = 0x7000 + unit * 0x10;
				const auto all_vertex = vertex_texture_snapshot::capture(all_state.registers, 15);
				const auto all_fragment = fragment_texture_snapshot::capture(all_state.registers, 65535);
				auto all_sources = executor.prepare_versioned_textures(all_vertex, all_fragment);
				require(all_sources.revisions.size() == 20 && all_sources.current(all_vertex, all_fragment), "Prepared epoch did not validate all four vertex/sixteen fragment sources");
				for (unsigned source = 0; source < 20; ++source)
				{
					if (source)
						all_sources = executor.prepare_versioned_textures(all_vertex, all_fragment);
					require(all_sources.current(all_vertex, all_fragment), "Refreshed twenty-source epoch is stale");
					const unsigned address = source < 4 ? 0x80000 + source * 256 : 0x70000 + (source - 4) * 256;
					ram->write({address, 1}, std::array{std::byte(source)});
					require(!all_sources.current(all_vertex, all_fragment), "Independent vertex/fragment source mutation escaped final validation");
				}
				auto changed = registers();
				texture_registers(changed, 0x300);
				changed.registers[0x1840 / 4] ^= 1;
				require(!versioned.current(textured.vertex_textures(), fragment_texture_snapshot::capture(changed.registers, 1)), "Prepared draw accepted changed texture words");
				auto original = std::make_unique<draw_texture_epoch>(versioned.textures);
				const auto retained = *original;
				original.reset();
				textures.clear();
				ram->write({0x3000, 64}, std::array<std::byte, 64>{});
				require(!versioned.current(textured.vertex_textures(), textured.textures()), "Prepared draw accepted changed VM revision");
				const auto reads = ram->reads;
				executor.execute(textured, targets, request, retained);
				verify(ctx, surfaces, target, {20, 40, 60, 255});
				require(ram->reads == reads && textures.statistics().entries == 0, "Prepared texture execution consulted VM/cache after capture");
				const auto generation = surfaces.status(target).content_generation;
				// Every fragment register word belongs to the image epoch,including
				// sampler-only words and the separate CONTROL3 block.
				for (const unsigned method : {0x1a00u / 4, 0x1a04u / 4, 0x1a08u / 4, 0x1a0cu / 4,
											  0x1a10u / 4, 0x1a14u / 4, 0x1a18u / 4, 0x1a1cu / 4, 0x1840u / 4})
				{
					state = registers();
					texture_registers(state, 0x300);
					state.registers[method] ^= 1;
					const auto different = snapshot(state, true, 1);
					bool refused = false;
					try
					{
						executor.execute(different, targets, request, retained);
					}
					catch (const std::invalid_argument&)
					{
						refused = true;
					}
					require(refused && ram->reads == reads && surfaces.status(target).content_generation == generation,
							"Prepared texture register mismatch reached VM or target publication");
				}
				bool refused = false;
				try
				{
					executor.execute(draw, targets, request, retained);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && ram->reads == reads && surfaces.status(target).content_generation == generation,
						"Prepared texture mask mismatch published target state");
				state = registers();
				texture_registers(state, 0x300);
				set_vertex_texture_registers(state, 0x600);
				const auto mixed_vertex = vertex_texture_snapshot::capture(state.registers, 1);
				const auto missing_fragment = fragment_texture_snapshot::capture(state.registers, 3);
				unsigned callbacks = 0;
				refused = false;
				try
				{
					(void)draw_texture_epoch::capture(ctx, mixed_vertex, missing_fragment, [&](bool, unsigned, const auto&)
													  { ++callbacks; return retained.fragment(0); });
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && !callbacks && ram->reads == reads, "Disabled fragment preflight occurred after vertex resolution");
				for (unsigned word = 0; word < 8; ++word)
					state.registers[0x1a00 / 4 + 8 + word] = state.registers[0x1a00 / 4 + word];
				state.registers[0x1840 / 4 + 1] = state.registers[0x1840 / 4];
				const auto two_fragments = fragment_texture_snapshot::capture(state.registers, 3);
				const auto no_vertex = vertex_texture_snapshot::capture(state.registers, 0);
				callbacks = 0;
				refused = false;
				try
				{
					(void)draw_texture_epoch::capture(ctx, no_vertex, two_fragments, [&](bool, unsigned unit, const auto&)
													  { ++callbacks; return unit == 0 ? retained.fragment(0) : resolved_fragment_texture{}; });
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && callbacks == 2 && surfaces.status(target).content_generation == generation,
						"Failed texture epoch capture published a partial draw");
				const auto replacement = executor.prepare_textures(textured.vertex_textures(), textured.textures());
				executor.execute(textured, targets, request, replacement);
				verify(ctx, surfaces, target, {0, 0, 0, 0});
				executor.execute(textured, targets, request, retained);
				verify(ctx, surfaces, target, {20, 40, 60, 255});
				std::cout << "Native prepared fragment epoch: copied ownership,cache eviction/VM mutation,old/new pixels,nine register words,mask,disabled preflight and partial capture refusals passed\n";
				std::cout << "Native versioned draw preparation validates20independentSources/rejects final VM/register changes;disabled units create no revision tickets;old immutable GPU draw preserved passed\n";
			}
			ram->write({0x3000, 64}, std::array<std::byte, 64>{});
			state = registers();
			texture_registers(state, 0x100);
			const auto self = snapshot(state, true, 1);
			const auto reads = ram->reads;
			executor.execute(self, targets, request);
			verify(ctx, surfaces, target, {20, 40, 60, 255});
			require(ram->reads == reads, "Self sampling used stale VM pixels instead of native surface snapshot");
			const auto prior = surfaces.status(target).content_generation;
			state = registers();
			texture_registers(state, 0x300);
			const auto badmask = snapshot(state, true, 0);
			bool refused = false;
			try
			{
				executor.execute(badmask, targets, request);
			}
			catch (const std::invalid_argument&)
			{
				refused = true;
			}
			require(refused && surfaces.status(target).content_generation == prior && ram->reads == reads, "Incorrect texture metadata mutated target or read guest data");
			auto broken = request;
			const auto canceled_epoch = surfaces.status(target).native_write_epoch;
			broken.geometry.vertices[0].vertex = 9;
			refused = false;
			try
			{
				executor.execute(draw, targets, broken);
			}
			catch (const std::out_of_range&)
			{
				refused = true;
			}
			require(refused && !surfaces.status(target).pending && !surfaces.status(target).faulted && surfaces.status(target).content_generation == prior, "Cancelled executor draw published target writes");
			require(surfaces.status(target).native_write_epoch == canceled_epoch, "Cancelled native encoding published an epoch");
			verify(ctx, surfaces, target, {20, 40, 60, 255});
			{
				auto empty = request;
				empty.geometry.vertices.clear();
				executor.execute(draw, targets, empty);
				require(surfaces.status(target).content_generation == prior, "Empty executor geometry published writes");
				auto mismatched = targets;
				mismatched.depth = surfaces.acquire({0x4000, {2, 2, 1, surface_depth::z24s8_uint, false}, 8, {}});
				refused = false;
				try
				{
					executor.execute(draw, mismatched, request);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && surfaces.status(target).content_generation == prior, "Mismatched framebuffer dimensions modified target");
				auto no_depth = targets;
				no_depth.depth.reset();
				auto bounds = request;
				bounds.raster.depth_bounds_enabled = true;
				refused = false;
				try
				{
					executor.execute(draw, no_depth, bounds);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && surfaces.status(target).content_generation == prior, "Missing depth bounds target published writes");
				auto bounded_targets = targets;
				bounded_targets.depth = depth;
				const auto initial_depth = surfaces.status(depth);
				const auto initial_reads = ram->reads;
				for (const auto range : {std::array{-.1f, 1.f}, std::array{0.f, 1.1f}, std::array{.8f, .2f},
					std::array{std::numeric_limits<float>::quiet_NaN(), 1.f}, std::array{0.f, std::numeric_limits<float>::infinity()}})
				{
					bounds.raster.depth_min = range[0];
					bounds.raster.depth_max = range[1];
					refused = false;
					try { executor.execute(self, bounded_targets, bounds); }
					catch (const std::invalid_argument&) { refused = true; }
					require(refused && ram->reads == initial_reads && surfaces.status(target).content_generation == prior &&
						surfaces.status(depth).content_generation == initial_depth.content_generation &&
						surfaces.status(depth).native_write_epoch == initial_depth.native_write_epoch,
						"Malformed bounds entered VM or published framebuffer writes");
				}
				bounds.raster.depth_min = 0;
				bounds.raster.depth_max = 1;
				bounds.geometry.vertices[0].vertex = 9;
				queries.begin(0, true);
				refused = false;
				try { executor.execute(draw, bounded_targets, bounds); }
				catch (const std::out_of_range&) { refused = true; }
				queries.end(0);
				require(refused && queries.consume(0) == 0 && !surfaces.status(target).pending && !surfaces.status(depth).pending &&
					!surfaces.status(target).faulted && !surfaces.status(depth).faulted && surfaces.status(target).content_generation == prior &&
					surfaces.status(depth).content_generation == initial_depth.content_generation &&
					surfaces.status(depth).native_write_epoch == initial_depth.native_write_epoch,
					"Bounds vertex failure after mirror initialization published writes or query samples");
				std::cout << "Native bounds preflight:five malformed ranges before VM;vertex failure after initializer cancels color/depth epochs and query samples passed\n";

			}
			{
				const auto gate = [ctx.device() newSharedEvent];
				require(bool(gate), "Executor shared-event creation failed");
				state = registers();
				auto captured = std::make_unique<rsx_draw_snapshot>(snapshot(state, false, 0));
				const auto pending_epoch = surfaces.status(target).native_write_epoch;
				auto delayed = request;
				auto delayed_targets = targets;
				delayed_targets.depth = depth;
				surfaces.clear(depth,3,0,0x40000042,255,{0,0,4,4});
				delayed.raster.depth_bounds_enabled = true;
				delayed.raster.depth_min = .2f;
				delayed.raster.depth_max = .3f;
				delayed.persistent = make_upload(ctx, positions);
				delayed.waits.push_back({gate, 1});
				auto temporary = std::make_unique<draw_executor>(ctx, surfaces, textures);
				surfaces.set_wait_timeout(std::chrono::milliseconds(1));
				bool timeout = false;
				try
				{
					temporary->execute(*captured, delayed_targets, delayed);
				}
				catch (const std::runtime_error&)
				{
					timeout = true;
				}
				require(timeout && surfaces.status(target).pending && surfaces.status(target).content_generation == prior, "Executor wait did not preserve a genuinely pending target transaction");
				require(surfaces.status(target).native_write_epoch == pending_epoch, "Pending GPU draw published its epoch before completion");
				require(temporary->pipeline_statistics().entries == 1, "Pending draw lacks cached native states");
				temporary.reset();
				captured.reset();
				delayed.persistent = nil;
				delayed.geometry.vertices.clear();
				gate.signaledValue = 1;
				surfaces.set_wait_timeout(std::chrono::seconds(30));
				surfaces.synchronize();
				require(!surfaces.status(target).pending && surfaces.status(target).content_generation == prior + 1, "Executor pending transaction did not publish after dependency completion");
				require(surfaces.status(target).native_write_epoch > pending_epoch, "Completed GPU draw did not publish its reserved epoch");
				verify(ctx, surfaces, target, {64, 128, 192, 255});
				std::cout << "Native pending narrow stored-depth bounds draw retains selected path/pipeline/depth states after executor/cache destruction passed\n";
			}
			std::cout << "Native integrated draw executor: owned real-interface epoch,program/pipeline/texture binding,GPU fetch,scissor/depth/stencil,VM texture upload,native self-feedback,metadata refusal and encoded cancellation passed\n";
			{
				ram->write({0x3000, 64}, pixels);
				state = registers();
				texture_registers(state, 0x300);
				const auto captured = snapshot(state, true, 1);
				auto owned = std::make_unique<draw_texture_epoch>(executor.prepare_textures(captured.vertex_textures(), captured.textures()));
				textures.clear();
				ram->write({0x3000, 64}, std::array<std::byte, 64>{});
				const auto reads = ram->reads;
				const auto generation = surfaces.status(target).content_generation;
				const auto gate = [ctx.device() newSharedEvent];
				auto delayed = request;
				delayed.waits.push_back({gate, 1});
				auto temporary = std::make_unique<draw_executor>(ctx, surfaces, textures);
				surfaces.set_wait_timeout(std::chrono::milliseconds(1));
				bool timeout = false;
				try
				{
					temporary->execute(captured, targets, delayed, *owned);
				}
				catch (const std::runtime_error&)
				{
					timeout = true;
				}
				require(timeout && surfaces.status(target).pending && ram->reads == reads &&
							surfaces.status(target).content_generation == generation,
						"Prepared texture epoch lost pending transaction");
				owned.reset();
				temporary.reset();
				gate.signaledValue = 1;
				surfaces.set_wait_timeout(std::chrono::seconds(30));
				surfaces.synchronize();
				require(surfaces.status(target).content_generation == generation + 1, "Prepared pending draw failed to publish");
				verify(ctx, surfaces, target, {20, 40, 60, 255});
				std::cout << "Native prepared texture epoch: pending GPU work retained old image after epoch/executor/cache destruction and VM change\n";
			}
			std::cout << "Native indexed geometry planner: 20-bit base wrap, sparse IDs0/4/8 and modulo3 GPU fetch passed\n";
			{
				const auto asymmetric = surfaces.acquire({0x5000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(asymmetric, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				draw_targets environment_targets;
				environment_targets.colors[0] = asymmetric;
				auto environment_request = request;
				test::environment_state runtime;
				runtime.scale = {1, 1, 1};
				runtime.offset = {1, 1, 0};
				const auto captured_environment = draw_environment_snapshot::capture(runtime);
				runtime.scale = {100, 100, 100};
				runtime.offset = {100, 100, 100};
				apply_draw_environment(captured_environment, 4, 4, environment_request);
				state = registers();
				const auto environment_draw = snapshot(state, false, 0);
				executor.execute(environment_draw, environment_targets, environment_request);
				const auto pixels = make_buffer(ctx, 1024, buffer_storage::shared);
				const std::array environment_uses{cached_surface_use{asymmetric, false}};
				const auto download = [&](context& commands, auto attachments)
				{
					texture_region whole;
					whole.size = MTLSizeMake(4, 4, 1);
					texture_to_buffer(commands, attachments[0], whole, pixels, 0, 256);
				};
				surfaces.draw(environment_uses, download);
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
					{
						if (x + y == 3)
							continue; // Separate diagonal top-left parity contract.
						const auto expected = x + y < 3 ? std::array{64u, 128u, 192u, 255u} : std::array{17u, 34u, 51u, 255u};
						for (unsigned c = 0; c < 4; ++c)
							require(static_cast<const unsigned char*>(pixels.contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] == expected[c],
									"Captured native viewport scale/offset/Y orientation differs from guest pixels");
					}
				const auto previous_viewport = environment_request.raster.viewport;
				bool scale_refused = false;
				try
				{
					apply_draw_environment(captured_environment, 8, 4, environment_request);
				}
				catch (const std::invalid_argument&)
				{
					scale_refused = true;
				}
				require(scale_refused && environment_request.raster.viewport.width == previous_viewport.width,
						"Uncoordinated environment scaling changed the request");
				std::cout << "Native captured draw environment: asymmetric guest pixels/Y orientation and scaling refusal passed\n";
			}
			std::cout << "Native draw executor context/VM affinity,empty/mismatched/missing-depth bounds refusal and actual pending dependency with destroyed executor resources passed\n";
			{
				test::pass_state source;
				source.registers = registers().registers;
				source.transform_constants[511][3] = 0x7fc12345;
				test::pass_processor processor;
				processor.descriptors = vertex_processor.descriptors;
				processor.persistent = positions;
				processor.index_bytes = raw_indices;
				test::pass_layout layout;
				test::pass_metadata metadata;
				auto vertex = vertex_program();
				auto fragment_bytes = program(false);
				test::fragment_program fragment;
				fragment.pointer = fragment_bytes.data();
				fragment.ucode_length = fragment_bytes.size();
				fragment.ctrl = rsx_fragment_float32;
				const auto requirements = [&](const auto&, unsigned first, unsigned count)
				{
					require(first == 0 && count == 9, "Packet input prefix disagrees with geometry");
					return std::pair<std::uint32_t, std::uint32_t>{48, 0};
				};
				const auto packet = rsx_draw_packet::capture(source, processor, layout, metadata, vertex, fragment, 0,
															 provoking_vertex::last, requirements);
				source.registers.assign(source.registers.size(), 0xffffffffu);
				source.scale = {100, 100, 100};
				source.transform_constants = {};
				vertex.data.assign(vertex.data.size(), 0xffffffffu);
				fragment_bytes.assign(fragment_bytes.size(), std::byte{0xff});
				processor.persistent.assign(48, std::byte{0});
				processor.index_bytes.assign(12, std::byte{0});
				test::framebuffer_layout framebuffer;
				framebuffer.target = 2;
				framebuffer.color_addresses = {0, 0x8000, 0, 0};
				framebuffer.actual_color_pitch[1] = 16;
				framebuffer.zeta_address = 0;
				const auto captured_framebuffer = framebuffer_snapshot::capture(framebuffer, true);
				framebuffer.color_addresses.fill(0);
				framebuffer.actual_color_pitch.fill(0);
				const auto imported = acquire_framebuffer(captured_framebuffer, surfaces);
				const auto owned_target = imported.colors[0];
				require(owned_target && !imported.colors[1] && surfaces.description(owned_target).address == 0x8000,
						"Physical framebuffer B did not map to logical shader output zero");
				surfaces.clear(owned_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				auto packet_targets = imported;
				const auto generation = surfaces.status(owned_target).content_generation;
				executor.execute(packet, packet_targets);
				require(surfaces.status(owned_target).content_generation == generation + 1,
						"Owned pass overload failed to publish successful draw");
				verify(ctx, surfaces, owned_target, {64, 128, 192, 255});
				{
					const auto before = surfaces.transaction_statistics();
					const auto initial = surfaces.status(owned_target).content_generation;
					const auto counters = queries.statistics();
					draw_execution_state execution;
					execution.queries = &queries;
					queries.begin(0, true);
					auto batch = surfaces.batch_draws();
					for (unsigned i = 0; i < 32; ++i)
					{
						executor.execute(packet, packet_targets, execution);
						require(surfaces.transaction_statistics().submitted == before.submitted &&
							surfaces.status(owned_target).content_generation == initial,
							"Owned packet draw waited/submitted or published during batch recording");
					}
					queries.end(0);
					batch.finish();
					const auto after = surfaces.transaction_statistics();
					require(after.submitted == before.submitted + 1 && after.native_versions == before.native_versions + 1 &&
						after.native_version_blits == before.native_version_blits + 1 && after.batched_draws == before.batched_draws + 32 &&
						after.raw_backup_submissions == before.raw_backup_submissions &&
						surfaces.status(owned_target).content_generation == initial + 32,
						"32 owned packets did not share one ordered native transaction andtarget version");
					require(queries.consume(0) == 512 && queries.statistics().acquired == counters.acquired + 1 &&
						queries.statistics().initializations == counters.initializations + 1 && queries.statistics().reused_batch == counters.reused_batch + 31,
						"Batched owned packets lost native visibility samples");
					verify(ctx, surfaces, owned_target, {64, 128, 192, 255});
					std::cout << "32 actual owned RSX packets:translated vertex/fragment programs,GPU fetch,admitted uploads,one submission/target version/preservation blit,32 delayed generations,one counter/512samples andexact pixels passed\n";
				}
				require(packet.vertices().constants()[511][3] == 0x7fc12345, "Owned pass lost high constant register bits");
				packet_targets.colors[0] = surfaces.acquire({0x8100, {2, 2, 1, surface_color::argb8, false}, 8, {}});
				const auto rejected_generation = surfaces.status(packet_targets.colors[0]).content_generation;
				bool refused = false;
				try
				{
					executor.execute(packet, packet_targets);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && surfaces.status(packet_targets.colors[0]).content_generation == rejected_generation,
						"Owned pass/framebuffer mismatch published target writes");
				std::cout << "Native owned pass executor: captured full input packet,all live sources mutated,correct pixels/generation and framebuffer mismatch refusal passed\n";
				{
					test::pass_state bounded_source;
					bounded_source.registers = registers().registers;
					bounded_source.registers[0xa74 / 4] = bounded_source.registers[0xa70 / 4] = 1;
					bounded_source.registers[0xa6c / 4] = 0x207;
					bounded_source.bounds = true;
					bounded_source.bounds_min = .2f;
					bounded_source.bounds_max = .3f;
					auto bounded_processor = processor;
					bounded_processor.persistent = positions;
					bounded_processor.index_bytes = raw_indices;
					auto bounded_vertex = vertex_program();
					auto bounded_bytes = program(false);
					test::fragment_program bounded_fragment;
					bounded_fragment.pointer = bounded_bytes.data();
					bounded_fragment.ucode_length = bounded_bytes.size();
					bounded_fragment.ctrl = rsx_fragment_float32;
					const auto bounded_packet = rsx_draw_packet::capture(bounded_source, bounded_processor, layout, metadata,
						bounded_vertex, bounded_fragment, 0, provoking_vertex::last, requirements);
					bounded_source.bounds = false;
					bounded_source.bounds_min = bounded_source.bounds_max = std::numeric_limits<float>::quiet_NaN();
					bounded_source.registers.assign(bounded_source.registers.size(), 0xffffffffu);
					bounded_processor.persistent.assign(48, std::byte{0});
					bounded_processor.index_bytes.assign(12, std::byte{0});
					bounded_vertex.data.assign(bounded_vertex.data.size(), 0xffffffffu);
					bounded_bytes.assign(bounded_bytes.size(), std::byte{0xff});
					test::framebuffer_layout bounded_layout;
					bounded_layout.target = 2;
					bounded_layout.color_addresses = {0, 0x8600, 0, 0};
					bounded_layout.actual_color_pitch[1] = 16;
					bounded_layout.zeta_address = 0x8700;
					bounded_layout.actual_zeta_pitch = 16;
					const auto bounded_framebuffer = framebuffer_snapshot::capture(bounded_layout, true);
					const auto bounded_targets = acquire_framebuffer(bounded_framebuffer, surfaces);
					surfaces.clear(bounded_targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
					surfaces.clear(bounded_targets.depth, 3, 0, 0x40000042, 255, {0, 0, 4, 4});
					draw_execution_state execution;
					execution.queries = &queries;
					for (unsigned repeat = 0; repeat < 2; ++repeat)
					{
						queries.begin(0, true);
						executor.execute(bounded_packet, bounded_framebuffer, execution);
						queries.end(0);
						require(queries.consume(0) == (repeat ? 0 : 16), "Owned packet did not apply captured stored-depth bounds");
					}
					verify(ctx, surfaces, bounded_targets.colors[0], {64, 128, 192, 255});
					std::cout << "Native owned packet bounds:captured registers/environment/program/vertex/index sources poisoned,physical B plus depth framebuffer,16 then zero visibility samples passed\n";
				}

				source.registers = registers().registers;
				source.scale = {1, 1, 1};
				source.offset = {1, 1, 0};
				processor.persistent = positions;
				processor.index_bytes = raw_indices;
				vertex = vertex_program();
				fragment_bytes = program(false);
				fragment.pointer = fragment_bytes.data();
				fragment.ucode_length = fragment_bytes.size();
				const auto cropped_packet = rsx_draw_packet::capture(source, processor, layout, metadata, vertex, fragment, 0,
																	 provoking_vertex::last, requirements);
				test::framebuffer_layout cropped_layout;
				cropped_layout.target = 2;
				cropped_layout.width = 2;
				cropped_layout.height = 4;
				cropped_layout.color_addresses = {0, 0x8200, 0, 0};
				cropped_layout.actual_color_pitch[1] = 32;
				cropped_layout.zeta_address = 0;
				const auto cropped_framebuffer = framebuffer_snapshot::capture(cropped_layout, true);
				const auto cropped_targets = acquire_framebuffer(cropped_framebuffer, surfaces);
				surfaces.clear(cropped_targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 2, 4});
				executor.execute(cropped_packet, cropped_framebuffer);
				const auto crop_output = make_buffer(ctx, 1024, buffer_storage::shared);
				const std::array crop_uses{cached_surface_use{cropped_targets.colors[0], false}};
				const auto download_crop = [&](context& commands, auto attachments)
				{
					texture_region region;
					region.size = MTLSizeMake(2, 4, 1);
					texture_to_buffer(commands, attachments[0], region, crop_output, 0, 256);
				};
				surfaces.draw(crop_uses, download_crop);
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 2; ++x)
					{
						if (x + y == 3)
							continue;
						const auto expected = x + y < 3 ? std::array{64u, 128u, 192u, 255u} : std::array{17u, 34u, 51u, 255u};
						for (unsigned c = 0; c < 4; ++c)
							require(static_cast<const unsigned char*>(crop_output.contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] == expected[c],
									"Validated framebuffer crop rescaled viewport or misplaced guest pixels");
					}
				cropped_layout.width = 5;
				cropped_layout.color_addresses[1] = 0x8400;
				const auto oversized_framebuffer = framebuffer_snapshot::capture(cropped_layout, true);
				refused = false;
				const auto crop_generation = surfaces.status(cropped_targets.colors[0]).content_generation;
				const auto crop_reads = ram->reads;
				try
				{
					executor.execute(cropped_packet, oversized_framebuffer);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && ram->reads == crop_reads && surfaces.status(cropped_targets.colors[0]).content_generation == crop_generation,
						"Oversized analysed framebuffer reached VM or target publication");
				cropped_layout.width = 2;
				cropped_layout.aa_mode = 3;
				cropped_layout.aa_factors = {1, 1};
				refused = false;
				try { (void)framebuffer_snapshot::capture(cropped_layout, true); }
				catch (const std::invalid_argument&) { refused = true; }
				require(refused && ram->reads == crop_reads, "Malformed framebuffer AA geometry reached VM");
				std::cout << "Native combined framebuffer/packet executor: B-only2x4 target with4x4 viewport crops without rescale,oversize/malformed-AA refusals passed\n";
				test::clause_renderer renderer;
				renderer.processor.descriptors = vertex_processor.descriptors;
				renderer.processor.persistent = positions;
				renderer.processor.index_bytes = raw_indices;
				renderer.current_vertex_program = vertex_program();
				renderer.current_vertex_program.ctrl = rsx_program_instanced_constants;
				renderer.current_fragment_program.ctrl = rsx_fragment_float32;
				test::pass_state clause_state;
				clause_state.current_draw_clause.pass_count = 3;
				test::pass_layout clause_layout;
				test::framebuffer_layout clause_framebuffer_layout;
				clause_framebuffer_layout.color_addresses = {0x8800, 0, 0, 0};
				clause_framebuffer_layout.actual_color_pitch[0] = 16;
				clause_framebuffer_layout.zeta_address = 0;
				const auto clause_framebuffer = framebuffer_snapshot::capture(clause_framebuffer_layout, true);
				const auto clause_targets = acquire_framebuffer(clause_framebuffer, surfaces);
				surfaces.clear(clause_targets.colors[0], 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				unsigned guard_count = 0, prepared = 0, submitted = 0;
				vertex_constant_cache clause_constants;
				draw_capture_limits clause_limits;
				clause_limits.constants = &clause_constants;
				std::vector<std::shared_ptr<const vertex_constant_registers>> clause_versions;
				struct guard
				{
					unsigned* active;
					explicit guard(unsigned& value) : active(&value) { ++*active; }
					~guard() { --*active; }
				};
				struct epoch
				{
					framebuffer_snapshot framebuffer;
					std::shared_ptr<guard> writer_exclusion;
					draw_texture_epoch textures;
				};
				std::vector<std::byte> current_fragment;
				const auto prepare_clause = [&](unsigned pass, unsigned flags) -> std::optional<epoch>
				{
					require(!guard_count && flags == (pass ? pass * 8 : 4), "Clause preparation order/flags differ");
					++prepared;
					if (pass == 1)
						return {};
					clause_state.registers = registers().registers;
					clause_state.transform_constants[511][3] = 0x7fc12345u + pass;
					clause_state.registers[0x8c0 / 4] = (1u << 16) | pass;
					clause_state.registers[0x8c4 / 4] = 4u << 16;
					current_fragment = program(false, {pass == 0 ? 1.f : 0.f, 0.f, pass == 2 ? 1.f : 0.f, 1});
					renderer.current_fragment_program.pointer = current_fragment.data();
					renderer.current_fragment_program.ucode_length = current_fragment.size();
					const auto vertex_words = vertex_texture_snapshot::capture(clause_state.registers, 0);
					const auto fragment_words = fragment_texture_snapshot::capture(clause_state.registers, 0);
					return epoch{clause_framebuffer, std::make_shared<guard>(guard_count), executor.prepare_textures(vertex_words, fragment_words)};
				};
				const auto submit_clause = [&](const rsx_draw_packet& captured, const framebuffer_snapshot& target, unsigned, const epoch& prepared)
				{
					require(guard_count == 1, "Clause epoch guard expired before native submit");
					++submitted;
					clause_versions.push_back(captured.vertices().constant_version());
					executor.execute(captured, target, prepared.textures);
				};
				const auto clause_before = surfaces.transaction_statistics();
				const auto clause_generation = surfaces.status(clause_targets.colors[0]).content_generation;
				auto clause_batch = surfaces.batch_draws();
				const auto passes = renderer.submit_owned_clause(clause_state, clause_layout, prepare_clause, submit_clause, provoking_vertex::last, clause_limits);
				require(surfaces.transaction_statistics().submitted == clause_before.submitted &&
					surfaces.status(clause_targets.colors[0]).content_generation == clause_generation,
					"Owned clause submitted/published its individual passes");
				clause_batch.finish();
				require(surfaces.transaction_statistics().draw_batches == clause_before.draw_batches + 1 &&
					surfaces.status(clause_targets.colors[0]).content_generation == clause_generation + 2,
					"Owned clause did not finish both native passes in one batch");
				require(clause_versions.size() == 2 && clause_versions[0] != clause_versions[1] &&
					(*clause_versions[0])[511][3] == 0x7fc12345 && (*clause_versions[1])[511][3] == 0x7fc12347 &&
					clause_constants.statistics().versions == 2, "Clause constants collapsed independent pass versions");
				require(passes == 3 && prepared == 3 && submitted == 2 && !guard_count && renderer.processor.analysis_calls == 2 &&
							renderer.context.dependencies == std::vector<unsigned>{1, 2} && renderer.context.cleanups == 1 &&
							!clause_state.current_draw_clause.rendering,
						"Clause driver skipped dependencies,collapsed passes or leaked epoch guard");
				const auto clause_output = make_buffer(ctx, 1024, buffer_storage::shared);
				const std::array clause_uses{cached_surface_use{clause_targets.colors[0], false}};
				const auto download_clause = [&](context& commands, auto attachments)
				{
					texture_region region;
					region.size = MTLSizeMake(4, 4, 1);
					texture_to_buffer(commands, attachments[0], region, clause_output, 0, 256);
				};
				surfaces.draw(clause_uses, download_clause);
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
					{
						const auto expected = x == 0 ? std::array{255u, 0u, 0u, 255u} : x == 2 ? std::array{0u, 0u, 255u, 255u}
																							   : std::array{17u, 34u, 51u, 255u};
						for (unsigned c = 0; c < 4; ++c)
							require(static_cast<const unsigned char*>(clause_output.contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] == expected[c],
									"Clause pass programs/scissors did not produce independent epochs");
					}
				std::cout << "Native clause renderer bridge: ordered dependencies,skipped middle pass,guarded capture/submit,independent register511 versions andred/blue GPU epochs passed\n";
			}
			{
				std::vector<std::byte> coordinates;
				for (const float value : {.125f, .5f, 0.f, 0.f, .375f, .5f, 0.f, 0.f, .625f, .5f, 0.f, 0.f})
					for (const unsigned shift : {24u, 16u, 8u, 0u})
						coordinates.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
				auto processor = vertex_processor;
				processor.persistent = coordinates;
				const auto coordinate_capture = vertex_data_snapshot::capture(processor, 0, test::vertex_data_metadata{1},
																			  vertex_state, geometry_plan.streams, {48, 0}, 48);
				auto vertex_request = request;
				apply_vertex_data_snapshot(ctx, coordinate_capture, vertex_request);
				processor.persistent.assign(48, std::byte{0});
				auto image = positions;
				image.resize(64);
				ram->write({0x6000, 64}, image);
				const auto vertex_target = surfaces.acquire({0x7000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(vertex_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				draw_targets vertex_targets;
				vertex_targets.colors[0] = vertex_target;
				state = registers();
				set_vertex_texture_registers(state, 0x600);
				const auto vertex_draw = snapshot(state, false, 0, true, 1);
				const auto prepared_vertex = executor.prepare_textures(vertex_draw.vertex_textures(), vertex_draw.textures());
				const auto vertex_reads = ram->reads;
				textures.clear();
				ram->write({0x6000, 64}, std::array<std::byte, 64>{});
				executor.execute(vertex_draw, vertex_targets, vertex_request, prepared_vertex);
				verify(ctx, surfaces, vertex_target, {64, 128, 192, 255});
				require(ram->reads == vertex_reads, "Prepared vertex image was re-resolved after VM change");
				const auto vertex_generation = surfaces.status(vertex_target).content_generation;
				for (unsigned word = 0; word < 8; ++word)
				{
					state = registers();
					set_vertex_texture_registers(state, 0x600);
					state.registers[0x900 / 4 + word] ^= 1;
					const auto different = snapshot(state, false, 0, true, 1);
					bool refused = false;
					try
					{
						executor.execute(different, vertex_targets, vertex_request, prepared_vertex);
					}
					catch (const std::invalid_argument&)
					{
						refused = true;
					}
					require(refused && ram->reads == vertex_reads && surfaces.status(vertex_target).content_generation == vertex_generation,
							"Prepared vertex register mismatch reached VM or target state");
				}
				ram->write({0x6000, 64}, image);
				// The deliberate VM mutation invalidated storage. Rebuild the
				// normalized cache entry before the separate UN reuse assertion.
				(void)executor.prepare_textures(vertex_draw.vertex_textures(), vertex_draw.textures());
				std::cout << "Native prepared vertex epoch: VM mutation/cache eviction retained sampled positions and all eight register mismatches refused\n";
				std::vector<std::byte> pixel_coordinates;
				for (const float value : {.5f, .5f, 0.f, 0.f, 1.5f, .5f, 0.f, 0.f, 2.5f, .5f, 0.f, 0.f})
					for (const unsigned shift : {24u, 16u, 8u, 0u})
						pixel_coordinates.push_back(std::byte(std::bit_cast<unsigned>(value) >> shift));
				auto unnormalized_request = vertex_request;
				unnormalized_request.persistent = make_upload(ctx, pixel_coordinates);
				state = registers();
				set_vertex_texture_registers(state, 0x600);
				state.registers[0x904 / 4] |= 0x40u << 8;
				const auto unnormalized = snapshot(state, false, 0, true, 1);
				const auto uploads_before = textures.statistics().uploads;
				executor.execute(unnormalized, vertex_targets, unnormalized_request);
				verify(ctx, surfaces, vertex_target, {64, 128, 192, 255});
				require(textures.statistics().uploads == uploads_before, "Vertex UN change duplicated immutable storage upload");
				const auto generation = surfaces.status(vertex_target).content_generation;
				const auto reads = ram->reads;
				state = registers();
				set_vertex_texture_registers(state, 0x600);
				const auto wrong_mask = snapshot(state, false, 0, true, 0);
				bool refused = false;
				try
				{
					executor.execute(wrong_mask, vertex_targets, vertex_request);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && ram->reads == reads && surfaces.status(vertex_target).content_generation == generation,
						"Vertex shader mask mismatch reached VM or target publication");
				state = registers();
				set_vertex_texture_registers(state, 0x600);
				state.registers[0x90c / 4] = 0;
				const auto disabled = snapshot(state, false, 0, true, 1);
				executor.execute(disabled, vertex_targets, vertex_request);
				require(ram->reads == reads && surfaces.status(vertex_target).content_generation == generation + 1,
						"Disabled vertex null binding read VM or failed native draw publication");
				verify(ctx, surfaces, vertex_target, {64, 128, 192, 255});
				state = registers();
				set_vertex_texture_registers(state, 0x600);
				const auto batch = vertex_texture_snapshot::capture(state.registers, 3);
				vertex_program_options options;
				options.textures.fill(vertex_texture_dimension::three);
				vertex_inputs inputs;
				sampler_cache vertex_samplers(ctx, 4);
				unsigned resolved = 0;
				const auto resolve = [&](unsigned, const fragment_texture_registers& words)
				{
					++resolved;
					return textures.resolve(words, surfaces);
				};
				refused = false;
				try
				{
					apply_vertex_texture_snapshot(batch, resolve, ctx, vertex_samplers, options, inputs);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && resolved == 1 && !inputs.textures[0] && !inputs.samplers[0] &&
							options.textures[0] == vertex_texture_dimension::three,
						"Failed vertex batch published partial shader/resource state");
				// The VTF source now lives only in a dirty native RGBA32F surface.
				// VM remains unchanged;sampling must snapshot that GPU version.
				const auto vertex_surface = surfaces.acquire({0x6500, {4, 1, 1, surface_color::rgba32f, false}, 64, {}});
				const std::array source_uses{cached_surface_use{vertex_surface, true}};
				const auto initialize_native = [&](context& commands, auto attachments)
				{
					auto pass = [MTL4RenderPassDescriptor new];
					pass.colorAttachments[0].texture = attachments[0];
					pass.colorAttachments[0].loadAction = MTLLoadActionClear;
					pass.colorAttachments[0].storeAction = MTLStoreActionStore;
					pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, .5, 1);
					(void)commands.render(pass);
					commands.end_encoding();
				};
				surfaces.draw(source_uses, initialize_native);
				draw_targets source_targets;
				source_targets.colors[0] = vertex_surface;
				source_targets.register_color_format = surface_color::rgba32f;
				for (unsigned x = 0; x < 3; ++x)
				{
					state = registers();
					state.registers[0x8c0 / 4] = (1u << 16) | x;
					state.registers[0x8c4 / 4] = 1u << 16;
					const auto source_draw = snapshot(state, false, 0, false, 0, {x == 1 ? 1.f : 0.f, x == 2 ? 1.f : 0.f, .5f, 1});
					executor.execute(source_draw, source_targets, request);
				}
				surfaces.clear(vertex_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				test::environment_state runtime;
				runtime.scale = {4, 4, 1};
				runtime.offset = {0, 0, 0};
				apply_draw_environment(draw_environment_snapshot::capture(runtime), 4, 4, vertex_request);
				state = registers();
				set_vertex_texture_registers(state, 0x650);
				const auto native_vertex_draw = snapshot(state, false, 0, true, 1);
				const auto native_reads = ram->reads;
				executor.execute(native_vertex_draw, vertex_targets, vertex_request);
				require(ram->reads == native_reads, "Vertex native surface sampling read stale VM bytes");
				const auto output = make_buffer(ctx, 1024, buffer_storage::shared);
				const std::array uses{cached_surface_use{vertex_target, false}};
				const auto download = [&](context& commands, auto attachments)
				{
					texture_region region;
					region.size = MTLSizeMake(4, 4, 1);
					texture_to_buffer(commands, attachments[0], region, output, 0, 256);
				};
				surfaces.draw(uses, download);
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
					{
						if (x + y == 3)
							continue;
						const auto expected = x + y < 3 ? std::array{64u, 128u, 192u, 255u} : std::array{17u, 34u, 51u, 255u};
						for (unsigned c = 0; c < 4; ++c)
							require(static_cast<const unsigned char*>(output.contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] == expected[c],
									"Vertex native surface texels did not produce expected guest triangle");
					}
				std::cout << "Native vertex texture draw: frozen RGBA32F VM upload,normalized/UN TXL position,dirty native surface without VM reads,mask refusal,disabled null and atomic batch passed\n";
			}
			{
				auto mapped = std::make_shared<guarded_gpu_memory>();
				auto small = std::make_shared<guarded_guest_memory>(mapped, 2);
				auto budget = ctx.buffer_ledger();
				surface_cache retired_surfaces(ctx, small, 8 << 20, budget);
				for (unsigned index = 0; index < 32; ++index)
				{
					auto current = retired_surfaces.acquire({0x20000 + index * 64, {4, 4, 1, surface_color::argb8, false}, 16, {}});
					retired_surfaces.clear(current, 0xf0, 0xff102030, 0, 255, {0, 0, 4, 4});
					retired_surfaces.erase(current);
				}
				guest_texture_cache retired_textures(ctx, small, [](unsigned offset, unsigned, std::size_t)
													 { return offset * 16; }, 1 << 20, budget, 1);
				std::vector<id<MTLTexture>> frozen;
				for (unsigned index = 0; index < 32; ++index)
				{
					std::array<std::byte, 64> bytes;
					for (unsigned byte = 0; byte < bytes.size(); ++byte)
						bytes[byte] = std::byte(std::array{255u, index, 20u, 30u}[byte % 4]);
					small->write({0x4000 + index * 64, 64}, bytes);
					auto words = registers();
					texture_registers(words, 0x400 + index * 4);
					const auto texture = retired_textures.resolve(capture_fragment_texture_snapshot(words, 1).at(0));
					frozen.push_back(texture.texture);
				}
				retired_textures.clear();
				small->write({0x4000, 64}, std::array<std::byte, 64>{});
				const auto output = make_buffer(ctx, 64, buffer_storage::shared);
				ctx.begin();
				texture_to_buffer(ctx, frozen.front(), {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, output, 0);
				ctx.submit().wait();
				const auto pixels = map_buffer(output, 0, 64);
				for (unsigned byte = 0; byte < pixels.size(); ++byte)
					require(pixels[byte] == std::byte(std::array{30u, 20u, 0u, 255u}[byte % 4]), "Watch retirement changed externally retained immutable image");
				std::cout << "32 native surface retirements and32 texture evictions through two VM watch slots;retained image stays immutable passed\n";
			}
			{
				auto mapped = std::make_shared<guarded_gpu_memory>();
				auto memory = std::make_shared<guarded_guest_memory>(mapped);
				auto budget = ctx.buffer_ledger();
				surface_cache mapped_surfaces(ctx, memory, 8 << 20, budget);
				guest_texture_cache mapped_textures(ctx, memory, [](unsigned offset, unsigned, std::size_t)
													{ return offset * 16; }, 1 << 20, budget);
				const auto target = mapped_surfaces.acquire({0x4000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				mapped_surfaces.clear(target, 0xf0, 0xff4080c0, 0, 255, {0, 0, 4, 4});
				mapped_surfaces.flush(target);
				auto words = registers();
				texture_registers(words, 0x400);
				const auto registers = capture_fragment_texture_snapshot(words, 1).at(0);
				const auto old_native = mapped_textures.resolve(registers, mapped_surfaces);
				(void)mapped_textures.resolve(registers); // Also populate RAM-revision cache.
				const auto mirror = memory->watch_surface(guest_byte_coverage({0x104000, 64}));
				const auto unchanged = memory->read_surface(mirror);
				std::array<std::byte, 64> replacement;
				for (unsigned byte = 0; byte < replacement.size(); ++byte)
					replacement[byte] = std::byte(std::array{255u, 9u, 19u, 29u}[byte % 4]);
				memory->write({0x84000, 64}, replacement);
				(void)memory->mutate_mapping({0, 65536}, [&](auto&) noexcept
											 { mapped->remapped = true; return true; });
				const auto fresh = mapped_textures.resolve(registers, mapped_surfaces);
				const auto check = [&](id<MTLTexture> image, std::array<unsigned, 4> expected)
				{
					const auto output = make_buffer(ctx, 64, buffer_storage::shared);
					ctx.begin();
					texture_to_buffer(ctx, image, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, output, 0);
					ctx.submit().wait();
					const auto pixels = map_buffer(output, 0, 64);
					for (unsigned byte = 0; byte < pixels.size(); ++byte)
						require(pixels[byte] == std::byte(expected[byte % 4]), "Remapped native image has stale pixels");
				};
				check(fresh.texture, {29, 19, 9, 255});
				check(old_native.texture, {192, 128, 64, 255});
				require(memory->read_surface(mirror).bytes == unchanged.bytes && memory->surface_revision(mirror) == unchanged.revision,
						"Virtual remap damaged old physical mirror");
				mapped_surfaces.clear(target, 0xf0, 0xff556677, 0, 255, {0, 0, 4, 4});
				(void)memory->mutate_mapping({0, 65536}, [&](auto&) noexcept
											 { mapped->remapped = false; return true; });
				bool refused = false;
				try
				{
					mapped_surfaces.flush(target);
				}
				catch (const surface_conflict_error&)
				{
					refused = true;
				}
				require(refused && memory->read_surface(mirror).bytes == unchanged.bytes, "Old dirty native surface wrote into replaced mapping");
				std::cout << "Native remap reloads stale surface/VM texture,retains old immutable image/mirror andrefuses dirty writeback across mapping ABA passed\n";
			}
			{
				auto mapped = std::make_shared<guarded_gpu_memory>();
				auto memory = std::make_shared<guarded_guest_memory>(mapped);
				auto ledger = ctx.buffer_ledger();
				surface_cache guarded_surfaces(ctx, memory, 8 << 20, ledger);
				guest_texture_cache guarded_textures(ctx, memory, [](unsigned offset, unsigned, std::size_t)
													 { return offset * 16; }, 1 << 20, ledger);
				draw_executor guarded_executor(ctx, guarded_surfaces, guarded_textures);
				const auto guarded_target = guarded_surfaces.acquire({0xd000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				draw_targets guarded_targets;
				guarded_targets.colors[0] = guarded_target;
				guarded_surfaces.clear(guarded_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				memory->write({0xc000, 64}, pixels);
				state = registers();
				texture_registers(state, 0xc00);
				const auto guarded_draw = snapshot(state, true, 1);
				const auto prepared = guarded_executor.prepare_versioned_textures(guarded_draw.vertex_textures(), guarded_draw.textures());
				mapped->execute([&](auto&)
								{
					require(prepared.current(guarded_draw.vertex_textures(), guarded_draw.textures()), "Final CPU epoch rejected current native sources");
					(void)memory->read({0xc000, 64}); });
				require(!guest_memory_epoch::current(*mapped), "Final CPU epoch leaked into GPU submission");
				const auto& owned = prepared.textures;
				memory->write({0xc000, 64}, std::array<std::byte, 64>{});
				guarded_textures.clear();
				guarded_executor.execute(guarded_draw, guarded_targets, request, owned);
				std::cout << "Native versioned guarded draw:final CPU callback validates nested VM views then releases exclusion before GPU submission passed\n";
				verify(ctx, guarded_surfaces, guarded_target, {20, 40, 60, 255});
				const auto flushed = guarded_surfaces.cpu_read({0xd000, 64});
				for (unsigned pixel = 0; pixel < 16; ++pixel)
					for (unsigned channel = 0; channel < 4; ++channel)
						require(flushed.bytes[pixel * 4 + channel] == pixels[pixel * 4 + channel],
								"Guarded memory CAS flush lost native target bytes");
				const auto revision = flushed.revision;
				guarded_surfaces.cpu_write({0xd000, 4}, std::array{std::byte{255}, std::byte{99}, std::byte{88}, std::byte{77}});
				require(memory->revision({0xd000, 64}) > revision, "Guarded CPU patch omitted target revision");
				state = registers();
				texture_registers(state, 0xd00);
				const auto patched_feedback = snapshot(state, true, 1);
				guarded_executor.execute(patched_feedback, guarded_targets, request);
				verify(ctx, guarded_surfaces, guarded_target, {99, 88, 77, 255});
				// A second virtual bank maps the same physical bytes. CPU reads
				// through it must flush the dirty GPU owner before returning bytes.
				const auto mirrored = guarded_surfaces.cpu_read({0x10d000, 64});
				for (unsigned pixel = 0; pixel < 16; ++pixel)
					for (unsigned channel = 0; channel < 4; ++channel)
						require(mirrored.bytes[pixel * 4 + channel] == std::byte(std::array{255u, 99u, 88u, 77u}[channel]),
								"CPU mirror read missed a dirty physical GPU alias");
				guarded_surfaces.spill(guarded_target);
				require(guarded_surfaces.status(guarded_target).spilled, "Physical mirror spill precondition failed");
				guarded_surfaces.cpu_write({0x10d000, 4}, std::array{std::byte{255}, std::byte{7}, std::byte{8}, std::byte{9}});
				require(!guarded_surfaces.status(guarded_target).spilled && !guarded_surfaces.status(guarded_target).initialized,
						"CPU mirror patch retained stale native/spill state");
				guarded_executor.execute(patched_feedback, guarded_targets, request);
				verify(ctx, guarded_surfaces, guarded_target, {7, 8, 9, 255});
				// Both GPU targets are physically identical despite disjoint guest
				// addresses. Reject concurrent writes before invoking the encoder.
				const auto mirror_target = guarded_surfaces.acquire({0x10d000, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				const std::array conflicting{cached_surface_use{guarded_target, true}, cached_surface_use{mirror_target, true}};
				const auto generation = guarded_surfaces.status(guarded_target).content_generation;
				unsigned encodes = 0;
				bool refused = false;
				try
				{
					guarded_surfaces.draw(conflicting, [&](context&, auto)
										  { ++encodes; });
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && !encodes && guarded_surfaces.status(guarded_target).content_generation == generation,
						"Physical mirror targets were allowed to write concurrently");
				// Alias-query failures must occur before GPU encoding/publication.
				mapped->reject_alias_query = true;
				refused = false;
				try
				{
					guarded_surfaces.clear(mirror_target, 0xf0, 0xff020304, 0, 255, {0, 0, 4, 4});
				}
				catch (const std::runtime_error&)
				{
					refused = true;
				}
				mapped->reject_alias_query = false;
				require(refused && !guarded_surfaces.status(mirror_target).pending && !guarded_surfaces.status(mirror_target).initialized &&
							guarded_surfaces.status(guarded_target).content_generation == generation,
						"Alias query failure published a GPU target mutation");
				guarded_surfaces.clear(mirror_target, 0xf0, 0xff0b1621, 0, 255, {0, 0, 4, 4});
				require(!guarded_surfaces.status(guarded_target).initialized,
						"GPU mirror writer retained an initialized native alias");
				const auto uploads = guarded_textures.statistics().uploads;
				const auto rebased = guarded_textures.resolve(patched_feedback.textures().at(0), guarded_surfaces);
				require(rebased.texture && guarded_textures.statistics().uploads == uploads && guarded_surfaces.status(mirror_target).dirty,
						"Dirty physical texture mirror used VM upload/writeback instead of native snapshot");
				const auto rebased_epoch = guarded_executor.prepare_textures(patched_feedback.vertex_textures(), patched_feedback.textures());
				guarded_executor.execute(patched_feedback, guarded_targets, request, rebased_epoch);
				verify(ctx, guarded_surfaces, guarded_target, {11, 22, 33, 255});
				const auto mirror_pixels = guarded_surfaces.cpu_read({0xd000, 64});
				for (unsigned pixel = 0; pixel < 16; ++pixel)
					for (unsigned channel = 0; channel < 4; ++channel)
						require(mirror_pixels.bytes[pixel * 4 + channel] == std::byte(std::array{255u, 11u, 22u, 33u}[channel]),
								"GPU mirror clear was not flushed through the original CPU address");
				(void)guarded_textures.resolve(guarded_draw.textures().at(0));
				const auto before_clear = guarded_textures.statistics().entries;
				guarded_textures.invalidate({0x1c0000, 1});
				require(guarded_textures.statistics().entries == before_clear, "Unrelated physical bytes invalidated texture storage");
				guarded_textures.invalidate({0x10c000, 64});
				require(guarded_textures.statistics().entries < before_clear, "Texture invalidation ignored a physical VM mirror");
				// R32F source remains GPU-dirty with zero VM bytes. Native sampling
				// through its mirror must preserve the exact float bits.
				const auto float_source = guarded_surfaces.acquire({0xe000, {4, 4, 1, surface_color::r32f, false}, 16, {}});
				const float precision = 1.f / 7;
				const std::array float_source_use{cached_surface_use{float_source, true}};
				guarded_surfaces.draw(float_source_use, [&](context& commands, auto attachments)
									  {
					auto pass = [MTL4RenderPassDescriptor new];
					pass.colorAttachments[0].texture = attachments[0];
					pass.colorAttachments[0].loadAction = MTLLoadActionClear;
					pass.colorAttachments[0].storeAction = MTLStoreActionStore;
					pass.colorAttachments[0].clearColor = MTLClearColorMake(precision, 0, 0, 1);
					(void)commands.render(pass);
					commands.end_encoding(); });
				const auto float_target = guarded_surfaces.acquire({0xf000, {4, 4, 1, surface_color::r32f, false}, 16, {}});
				draw_targets float_targets;
				float_targets.colors[0] = float_target;
				float_targets.register_color_format = surface_color::r32f;
				state = registers();
				texture_registers(state, 0x10e00);
				state.registers[0x1a04 / 4] = (state.registers[0x1a04 / 4] & ~0xff00u) | (0xbcu << 8);
				const auto float_draw = snapshot(state, true, 1);
				const auto float_uploads = guarded_textures.statistics().uploads;
				guarded_executor.execute(float_draw, float_targets, request);
				require(guarded_textures.statistics().uploads == float_uploads && guarded_surfaces.status(float_source).dirty,
						"Float physical mirror used VM upload/writeback");
				const auto untouched_vm = memory->read({0xe000, 64});
				require(std::all_of(untouched_vm.bytes.begin(), untouched_vm.bytes.end(), [](std::byte value)
									{ return value == std::byte{0}; }),
						"Native mirror sampling published float source to VM");
				const auto float_output = make_buffer(ctx, 1024, buffer_storage::shared);
				const std::array float_target_use{cached_surface_use{float_target, false}};
				guarded_surfaces.draw(float_target_use, [&](context& commands, auto attachments)
									  {
					texture_region whole;
					whole.size = MTLSizeMake(4, 4, 1);
					texture_to_buffer(commands, attachments[0], whole, float_output, 0, 256); });
				for (unsigned y = 0; y < 4; ++y)
					for (unsigned x = 0; x < 4; ++x)
						require(std::bit_cast<unsigned>(static_cast<const float*>(float_output.contents)[y * 64 + x]) == std::bit_cast<unsigned>(precision),
								"Native float mirror sampling changed source precision");
				std::cout << "Native guarded guest memory: frozen texture epoch,CAS target writeback,CPU patch revision and refreshed self-feedback pixels passed\n";
				std::cout << "Native physical VM aliases: CPU mirror read/write,spill invalidation,conflicting write refusal,query failure before encoding,native dirty texture rebase andtexture invalidation passed\n";
				std::cout << "Native R32F physical mirror: exact1/7 float bits in16pixels,no VM upload/writeback andunchanged zero source VM passed\n";
				const auto depth_original = guarded_surfaces.acquire({0x18000, {4, 4, 1, surface_depth::z16_float, false}, 8, {}});
				const auto depth_mirror = guarded_surfaces.acquire({0x118000, {4, 4, 1, surface_depth::z16_float, false}, 8, {}});
				const auto write_native_depth = [&](const cached_surface_ref& destination, float value)
				{
					const std::array use{cached_surface_use{destination, true}};
					guarded_surfaces.draw(use, [&](context& commands, auto attachments)
										  {
						auto pass = [MTL4RenderPassDescriptor new];
						pass.depthAttachment.texture = attachments[0];
						pass.depthAttachment.loadAction = MTLLoadActionClear;
						pass.depthAttachment.storeAction = MTLStoreActionStore;
						pass.depthAttachment.clearDepth = value;
						(void)commands.render(pass);
						commands.end_encoding(); });
				};
				write_native_depth(depth_original, precision);
				const auto original_epoch = guarded_surfaces.status(depth_original).native_write_epoch;
				guarded_surfaces.flush(depth_original);
				guarded_surfaces.prepare(depth_mirror); // E4M12 VM decode loses lower float bits.
				require(original_epoch && guarded_surfaces.status(depth_original).native_write_epoch == original_epoch &&
							!guarded_surfaces.status(depth_original).dirty && !guarded_surfaces.status(depth_mirror).native_write_epoch,
						"Flush/VM upload lost native precision provenance");
				const auto depth_readback = make_buffer(ctx, 1024, buffer_storage::shared);
				const std::array mirror_read{cached_surface_use{depth_mirror, false}};
				guarded_surfaces.draw(mirror_read, [&](context& commands, auto attachments)
									  {
					texture_region whole;
					whole.size = MTLSizeMake(4, 4, 1);
					texture_to_buffer(commands, attachments[0], whole, depth_readback, 0, 256, 0, texture_plane::depth); });
				require(std::bit_cast<unsigned>(*static_cast<const float*>(depth_readback.contents)) != std::bit_cast<unsigned>(precision),
						"Clean VM mirror did not exercise E4M12 precision loss");
				state = registers();
				texture_registers(state, 0x11800);
				state.registers[0x1a04 / 4] = (state.registers[0x1a04 / 4] & ~0xff00u) | (0xb3u << 8);
				state.registers[0x1840 / 4] = (1u << 20) | 8;
				const auto precise_depth_draw = snapshot(state, true, 1);
				const auto verify_precise_depth = [&](float expected)
				{
					const auto uploads = guarded_textures.statistics().uploads;
					guarded_executor.execute(precise_depth_draw, float_targets, request);
					require(guarded_textures.statistics().uploads == uploads, "Clean precision source fell back to texture VM upload");
					guarded_surfaces.draw(float_target_use, [&](context& commands, auto attachments)
										  {
						texture_region whole;
						whole.size = MTLSizeMake(4, 4, 1);
						texture_to_buffer(commands, attachments[0], whole, float_output, 0, 256); });
					for (unsigned y = 0; y < 4; ++y)
						for (unsigned x = 0; x < 4; ++x)
							require(std::bit_cast<unsigned>(static_cast<const float*>(float_output.contents)[y * 64 + x]) == std::bit_cast<unsigned>(expected),
									"Clean alias selection used guest-quantized depth instead of native precision");
				};
				verify_precise_depth(precision);
				const float newer_precision = 1.f / 9;
				write_native_depth(depth_mirror, newer_precision);
				const auto newer_epoch = guarded_surfaces.status(depth_mirror).native_write_epoch;
				require(newer_epoch > original_epoch, "Native epoch was not ordered across mirror owners");
				guarded_surfaces.flush(depth_mirror);
				guarded_surfaces.prepare(depth_original);
				guarded_surfaces.spill(depth_mirror);
				require(guarded_surfaces.status(depth_mirror).spilled && guarded_surfaces.status(depth_mirror).native_write_epoch == newer_epoch &&
							!guarded_surfaces.status(depth_original).native_write_epoch,
						"Spill/VM refresh corrupted precision epoch");
				verify_precise_depth(newer_precision);
				require(guarded_surfaces.status(depth_mirror).spilled && guarded_surfaces.status(depth_mirror).native_write_epoch == newer_epoch,
						"Direct spill snapshot restored its source or lost selected provenance");
				std::cout << "Native clean mirror selection: original1/7 andnewer1/9 depth float bits beat quantized E4M12 VM aliases in32pixels,raw spill and native provenance retained\n";
				guarded_surfaces.cpu_write({0x18000, 32}, std::array<std::byte, 32>{});
				guarded_surfaces.prepare(depth_original);
				guarded_surfaces.prepare(depth_mirror);
				require(!guarded_surfaces.status(depth_original).native_write_epoch && !guarded_surfaces.status(depth_mirror).native_write_epoch,
						"CPU replacement retained obsolete native provenance");
				verify_precise_depth(.0078125f);
				std::cout << "Native clean mirror CPU replacement: bothVM epochs cleared,equivalent decoded aliases selected andexact E4M12 zero bits refreshed\n";
			}
			{
				constexpr unsigned base = 0xe0000, address = base + 4, value = 0x12345678;
				ram->write({base, 64}, std::array<std::byte, 64>{});
				textures.clear();
				state = registers();
				texture_registers(state, base / 16);
				const auto words = capture_fragment_texture_snapshot(state, 1).at(0);
				const auto old = textures.resolve(words);
				const auto label_target = surfaces.acquire({base, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(label_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				const auto label_batches = surfaces.transaction_statistics().draw_batches;
				auto label_batch = surfaces.batch_draws();
				surfaces.clear(label_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				require(surfaces.status(label_target).pending && surfaces.transaction_statistics().draw_batches == label_batches,
					"Label fixture failed to retain an unsubmitted native batch");
				unsigned dependencies = 0, stores = 0;
				const auto sync = [&]
				{ ++dependencies; };
				const auto store = [&](unsigned target, unsigned data)
				{
					require(target == address && data == value && dependencies > stores && !surfaces.status(label_target).pending && !surfaces.status(label_target).dirty,
							"Label became visible before native/dependency/dirty-alias synchronization");
					if (!stores)
					{
						const auto flushed = ram->read({base, 64});
						for (unsigned byte = 0; byte < 64; ++byte)
							require(flushed.bytes[byte] == std::byte(std::array{255u, 17u, 34u, 51u}[byte % 4]), "Label store lost prior dirty GPU bytes");
					}
					ram->write({target, 4}, std::array{std::byte(data >> 24), std::byte(data >> 16), std::byte(data >> 8), std::byte(data)});
					++stores;
				};
				release_guest_label(surfaces, textures, address, value, sync, store);
				label_batch.finish();
				require(surfaces.transaction_statistics().draw_batches == label_batches + 1,
					"Label release failed to submit exactly one recorded prefix");
				require(dependencies == 1 && stores == 1 && !textures.statistics().entries && !surfaces.status(label_target).initialized,
						"Label did not invalidate cached/native aliases after atomic store");
				const auto expected = ram->read({base, 64});
				for (unsigned byte = 0; byte < 64; ++byte)
					require(expected.bytes[byte] == std::byte(byte >= 4 && byte < 8 ? std::array{0x12u, 0x34u, 0x56u, 0x78u}[byte - 4] : std::array{255u, 17u, 34u, 51u}[byte % 4]),
							"Big-endian label changed bytes outside its four-byte word");
				release_guest_label(surfaces, textures, address, value, sync, store);
				require(dependencies == 2 && stores == 2 && ram->revision({base, 64}) > expected.revision, "Same-value label skipped ordering or writer notification");
				bool refused = false;
				try
				{
					release_guest_label(surfaces, textures, address + 1, value, sync, store);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && dependencies == 2 && stores == 2, "Misaligned label reached dependency/store callbacks");
				surfaces.clear(label_target, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
				refused = false;
				try
				{
					release_guest_label(surfaces, textures, address, value, sync, [](auto, auto)
										{ throw std::range_error("store preflight failure"); });
				}
				catch (const std::range_error&)
				{
					refused = true;
				}
				require(refused && dependencies == 3 && stores == 2 && !surfaces.status(label_target).dirty && surfaces.status(label_target).initialized,
						"Failed label store lost successfully flushed native data");
				const auto gate = [ctx.device() newSharedEvent];
				surfaces.set_wait_timeout(std::chrono::milliseconds(1));
				const std::array uses{cached_surface_use{label_target, true}};
				refused = false;
				try
				{
					surfaces.draw(uses, [&](context& commands, auto)
								  { commands.wait_event(gate, 1); });
				}
				catch (const std::runtime_error&)
				{
					refused = true;
				}
				require(refused && surfaces.status(label_target).pending, "Native label test did not hold pending GPU work");
				refused = false;
				try
				{
					release_guest_label(surfaces, textures, address, value, sync, store, std::chrono::milliseconds(1));
				}
				catch (const std::runtime_error&)
				{
					refused = true;
				}
				require(refused && dependencies == 3 && stores == 2 && surfaces.status(label_target).pending, "Pending native timeout published label or dependencies early");
				gate.signaledValue = 1;
				surfaces.set_wait_timeout(std::chrono::seconds(30));
				release_guest_label(surfaces, textures, address, value, sync, store);
				require(dependencies == 4 && stores == 3 && !surfaces.status(label_target).pending, "Released native GPU work did not complete ordered label store");
				const auto old_pixels = make_buffer(ctx, 64, buffer_storage::shared);
				ctx.begin();
				texture_to_buffer(ctx, old.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, old_pixels, 0);
				ctx.submit().wait();
				for (const auto byte : map_buffer(old_pixels, 0, 64))
					require(byte == std::byte{0}, "Label mutation changed retained immutable texture");
				std::cout << "Native GCM label release:unsubmitted batch boundary,GPU/dependency/dirty-alias ordering,BE word/padding,VM/cache/native invalidation,same-value notification,store failure,pending timeout/retry and old immutable pixels passed\n";
			}
			{
				constexpr unsigned base = 0x60000, alias = base + (1 << 20), address = alias + 16, value = 0xcafebabe;
				constexpr std::uint64_t timestamp = 0x0102030405060708ull;
				const std::array encoded{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8},
										 std::byte{0xca}, std::byte{0xfe}, std::byte{0xba}, std::byte{0xbe}, std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}};
				auto provider = std::make_shared<guarded_gpu_memory>();
				auto memory = std::make_shared<guarded_guest_memory>(provider);
				auto ledger = ctx.buffer_ledger();
				surface_cache targets(ctx, memory, 8 << 20, ledger);
				guest_texture_cache snapshots(ctx, memory, [](unsigned offset, unsigned, std::size_t)
											  { return offset * 16; }, 1 << 20, ledger);
				state = registers();
				texture_registers(state, base / 16);
				const auto old = snapshots.resolve(capture_fragment_texture_snapshot(state, 1).at(0));
				texture_registers(state, alias / 16);
				(void)snapshots.resolve(capture_fragment_texture_snapshot(state, 1).at(0));
				require(snapshots.statistics().entries == 2, "Report fixture did not retain both virtual texture aliases");
				const auto target = targets.acquire({base, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				const auto direct_watch = memory->watch_surface(guest_byte_coverage({base + 16, 16}));
				const auto alias_watch = memory->watch_surface(guest_byte_coverage({address, 16}));
				const auto outside_watch = memory->watch_surface(guest_byte_coverage({base + 48, 16}));
				targets.clear(target, 0xf0, 0x88776655, 0, 255, {0, 0, 4, 4});
				const auto report_batches = targets.transaction_statistics().draw_batches;
				auto report_batch = targets.batch_draws();
				targets.clear(target, 0xf0, 0x88776655, 0, 255, {0, 0, 4, 4});
				require(targets.status(target).pending && targets.transaction_statistics().draw_batches == report_batches,
					"Report fixture failed to retain an unsubmitted native batch");
				unsigned stores = 0;
				const auto store = [&](unsigned sink, std::uint64_t time, unsigned result)
				{
					require(sink == address && time == timestamp && result == value && !targets.status(target).pending && !targets.status(target).dirty,
							"Report reached VM store before native pending/dirty alias completion");
					provider->execute([&](exclusive_guest_memory::view& view)
									  {
						const auto direct_before = memory->surface_revision(direct_watch), alias_before = memory->surface_revision(alias_watch), outside_before = memory->surface_revision(outside_watch);
						const guest_surface_range range{sink, 16};
						const auto bytes = view.map(range, true);
						memory->before_write_in_transaction(view, range);
						store_report_atomic(*reinterpret_cast<atomic_t<CellGcmReportData>*>(bytes), time, result);
						view.written(range);
						require(memory->surface_revision(direct_watch) > direct_before && memory->surface_revision(alias_watch) > alias_before && memory->surface_revision(outside_watch) == outside_before,
							"Report revision notification missed physical mirror or invalidated outside bytes"); });
					++stores;
				};
				store_guest_report(targets, snapshots, address, timestamp, value, store);
				report_batch.finish();
				require(targets.transaction_statistics().draw_batches == report_batches + 1,
					"Report store failed to submit exactly one recorded prefix");
				require(stores == 1 && provider->notifications >= 2 && !targets.status(target).initialized && !snapshots.statistics().entries,
						"Report did not retire native/VM texture aliases and notify provider");
				const auto written = memory->read({base, 64});
				for (unsigned byte = 0; byte < 64; ++byte)
					require(written.bytes[byte] == (byte >= 16 && byte < 32 ? encoded[byte - 16] : std::byte(std::array{0x88u, 0x77u, 0x66u, 0x55u}[byte % 4])),
							"Report changed bytes outside its sixteen-byte timer/value/padding record");
				const auto revision = memory->surface_revision(direct_watch);
				store_guest_report(targets, snapshots, address, timestamp, value, store);
				require(stores == 2 && memory->surface_revision(direct_watch) > revision, "Same-value report omitted revision store");
				for (unsigned invalid : {address + 4, address + 8, address + 12})
				{
					bool refused = false;
					try
					{
						store_guest_report(targets, snapshots, invalid, timestamp, value, store);
					}
					catch (const std::invalid_argument&)
					{
						refused = true;
					}
					require(refused && stores == 2, "Misaligned report reached VM callback");
				}
				bool refused = false;
				try
				{
					store_guest_report(targets, snapshots, address, timestamp, value, {});
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused && stores == 2, "Report accepted absent atomic writer");
				targets.clear(target, 0xf0, 0x88776655, 0, 255, {0, 0, 4, 4});
				refused = false;
				try
				{
					store_guest_report(targets, snapshots, address, timestamp, value, [](auto, auto, auto)
									   { throw std::range_error("report mapping refusal"); });
				}
				catch (const std::range_error&)
				{
					refused = true;
				}
				require(refused && stores == 2 && !targets.status(target).dirty && targets.status(target).initialized, "Report pre-store refusal discarded flushed native contents");
				const auto gate = [ctx.device() newSharedEvent];
				struct open_gate
				{
					id<MTLSharedEvent> event;
					~open_gate() { event.signaledValue = 1; }
				} cleanup{gate};
				targets.set_wait_timeout(std::chrono::milliseconds(1));
				const std::array uses{cached_surface_use{target, true}};
				refused = false;
				try
				{
					targets.draw(uses, [&](context& commands, auto)
								 { commands.wait_event(gate, 1); });
				}
				catch (const std::runtime_error&)
				{
					refused = true;
				}
				require(refused && targets.status(target).pending, "Report fixture lacks pending native work");
				refused = false;
				try
				{
					store_guest_report(targets, snapshots, address, timestamp, value, store, std::chrono::milliseconds(1));
				}
				catch (const std::runtime_error&)
				{
					refused = true;
				}
				require(refused && stores == 2 && targets.status(target).pending, "Report timeout published VM data");
				gate.signaledValue = 1;
				targets.set_wait_timeout(std::chrono::seconds(30));
				store_guest_report(targets, snapshots, address, timestamp, value, store);
				require(stores == 3 && !targets.status(target).pending, "Report retry did not publish after GPU completion");
				const auto pixels = make_buffer(ctx, 64, buffer_storage::shared);
				ctx.begin();
				texture_to_buffer(ctx, old.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, pixels, 0);
				ctx.submit().wait();
				for (const auto byte : map_buffer(pixels, 0, 64))
					require(byte == std::byte{0}, "Report mutated retained immutable native image");
				std::cout << "Native report storage:unsubmitted batch boundary,dirty physical alias/GPU ordering,BE16record/padding,exact revisions/provider notifications,same-value,alignment/missing/store refusal,pending timeout/retry and old immutable pixels passed\n";
			}
			{
				constexpr unsigned base = 0x70000;
				const auto target = surfaces.acquire({base, {4, 4, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(target, 0xf0, 0xff778899, 0, 255, {0, 0, 4, 4});
				const auto owner = std::this_thread::get_id();
				render_tasks tasks([] {}, 4);
				tasks.bind_owner();
				guest_surface_snapshot transferred;
				std::exception_ptr failure;
				std::thread reader([&]
								   {
					try
					{
						tasks.invoke([&]
						{
							require(std::this_thread::get_id() == owner, "Foreign fault worker reached native context");
							transferred = surfaces.cpu_read({base, 64});
						});
					}
					catch (...) { failure = std::current_exception(); } });
				const auto wait_pending = [&]
				{
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
					while (!tasks.pending())
					{
						if (std::chrono::steady_clock::now() > deadline)
							throw std::runtime_error("Native GPU handoff fixture did not post");
						std::this_thread::yield();
					}
				};
				wait_pending();
				require(tasks.drain() == 1, "Native fault handoff did not execute one owner operation");
				reader.join();
				if (failure)
					std::rethrow_exception(failure);
				require(transferred.bytes.size() == 64 && !surfaces.status(target).dirty, "Native read handoff published before dirty GPU flush");
				for (unsigned byte = 0; byte < 64; ++byte)
					require(transferred.bytes[byte] == std::byte(std::array{255u, 119u, 136u, 153u}[byte % 4]), "Native owner handoff lost guest BE pixels");
				const auto gate = [ctx.device() newSharedEvent];
				struct open_gate
				{
					id<MTLSharedEvent> event;
					~open_gate() { event.signaledValue = 1; }
				} cleanup{gate};
				surfaces.set_wait_timeout(std::chrono::milliseconds(1));
				const std::array uses{cached_surface_use{target, true}};
				bool refused = false;
				try
				{
					surfaces.draw(uses, [&](context& commands, auto)
								  { commands.wait_event(gate, 1); });
				}
				catch (const std::runtime_error&)
				{
					refused = true;
				}
				require(refused && surfaces.status(target).pending, "Native handoff fixture lacks pending GPU work");
				failure = {};
				std::thread waiting([&]
									{
					try { tasks.invoke([&] { surfaces.synchronize(std::chrono::milliseconds(1)); }); }
					catch (...) { failure = std::current_exception(); } });
				wait_pending();
				(void)tasks.drain();
				waiting.join();
				require(bool(failure) && tasks.available() && surfaces.status(target).pending, "GPU handoff timeout lost pending native ownership or failed to reach producer");
				gate.signaledValue = 1;
				surfaces.set_wait_timeout(std::chrono::seconds(30));
				surfaces.synchronize();
				failure = {};
				bool executed = false;
				std::thread cancelled([&]
									  {
					try { tasks.invoke([&] { executed = true; surfaces.clear(target, 0xf0, 0xff000000, 0, 255, {0, 0, 4, 4}); }); }
					catch (...) { failure = std::current_exception(); } });
				wait_pending();
				tasks.close();
				cancelled.join();
				require(bool(failure) && !executed && !tasks.pending(), "Closed native handoff executed late GPU work");
				const auto retained = surfaces.cpu_read({base, 64});
				require(retained.bytes == transferred.bytes, "Cancelled native handoff changed GPU/guest contents");
				std::cout << "Native fault/task owner handoff:foreign producer dirty GPU readback,BE pixels,pending GPU timeout/error transport/recovery and shutdown cancellation without late GPU work passed\n";
			}
			{
				// Two quads write twelve u16 indices:24bytes. The old index-count
				// range was12bytes and missed this dirty native target at byte16.
				constexpr unsigned base = 0x74000;
				const auto bytes = *rsx::dma_manager::index_emulation_bytes(rsx::primitive_type::quads, 8);
				require(bytes == 24, "Native DMA fixture lost actual index byte extent");
				const auto job_range = *rsx::dma_manager::fault_range_from_bytes(base, bytes);
				const guest_surface_range range{job_range.start, job_range.length()};
				const auto target = surfaces.acquire({base + 16, {4, 1, 1, surface_color::argb8, false}, 16, {}});
				surfaces.clear(target, 0xf0, 0xffa1b2c3, 0, 255, {0, 0, 4, 1});
				validate_fault_transfer_range(range, base + 23);
				bool refused = false;
				try
				{
					validate_fault_transfer_range({base, 12}, base + 23);
				}
				catch (const std::invalid_argument&)
				{
					refused = true;
				}
				require(refused, "Old element-count span accepted a tail-byte fault");
				render_tasks tasks([] {}, 1);
				tasks.bind_owner();
				guest_surface_snapshot transferred;
				std::exception_ptr failure;
				std::thread producer([&]
									 {
					try { tasks.invoke([&] { transferred = surfaces.cpu_read(range); }); }
					catch (...) { failure = std::current_exception(); } });
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
				while (!tasks.pending() && std::chrono::steady_clock::now() < deadline)
					std::this_thread::yield();
				const auto consumed = tasks.drain();
				producer.join();
				if (failure)
					std::rethrow_exception(failure);
				require(consumed == 1 && transferred.bytes.size() == 24 && !surfaces.status(target).dirty,
						"Actual DMA tail byte extent did not flush its dirty native overlap");
				for (unsigned byte = 16; byte < 24; ++byte)
					require(transferred.bytes[byte] == std::byte(std::array{255u, 161u, 178u, 195u}[byte % 4]),
							"Native DMA tail-byte recovery lost BE pixels");
				std::cout << "Native DMA index fault:24byte extent includes dirty GPU overlap outside old12element span,tail fault coverage/BE readback and owner handoff passed\n";
			}
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
