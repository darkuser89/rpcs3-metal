#include "M4PipelineStateBinding.h"
#include "M4ShaderProgramBinding.h"
#include "M4ShaderSnapshotFixtures.h"
#include "M4VertexFetch.h"
#include "M4VertexShader.h"
#include "M4ShaderCache.h"

#include <bit>
#include <cmath>
#include <iostream>
#include <thread>

namespace
{
using namespace metal4;
using color = std::array<float, 4>;
constexpr color source{.25f, .5f, .75f, .5f}, destination{.125f, .25f, .5f, .25f};
void require(bool v, const char* message)
{
	if (!v)
		throw std::runtime_error(message);
}
std::vector<std::uint32_t> registers()
{
	std::vector<std::uint32_t> m(0x1840 / 4);
	m[0x314 / 4] = 0x10001;
	m[0x320 / 4] = 0x80068006;
	m[0x324 / 4] = 0x01010101;
	m[0x370 / 4] = 0xfff0;
	m[0x330 / 4] = m[0x350 / 4] = 0x207;
	for (unsigned a : {0x33cu, 0x340u, 0x344u, 0x35cu, 0x360u, 0x364u})
		m[a / 4] = 0x1e00;
	m[0x32c / 4] = m[0x338 / 4] = m[0x34c / 4] = m[0x358 / 4] = 255;
	m[0xa6c / 4] = 0x201;
	m[0x378 / 4] = 0x1503;
	m[0x1834 / 4] = 0x900;
	m[0x1830 / 4] = 0x405;
	m[0x8c0 / 4] = m[0x8c4 / 4] = 4u << 16;
	return m;
}
void append(std::vector<std::byte>& b, unsigned word)
{
	for (unsigned shift : {8u, 0u, 24u, 16u})
		b.push_back(std::byte(word >> shift));
}
std::vector<std::byte> program(color value, unsigned outputs)
{
	std::vector<std::byte> result;
	for (unsigned slot = 0; slot < outputs; ++slot)
	{
		const unsigned reg = std::array{0u, 2u, 3u, 4u}[slot];
		for (unsigned w : {(1u << 24) | (reg << 1) | (15u << 9) | (slot + 1 == outputs ? 1u : 0u), (0xe4u << 9) | (7u << 18) | (0xe4u << 21) | 2u, 0xe4u << 9, 0xe4u << 9})
			append(result, w);
		for (float v : value)
			append(result, std::bit_cast<unsigned>(v));
	}
	return result;
}
unsigned logic(unsigned op, unsigned s, unsigned d)
{
	switch (op)
	{
	case 0:
		return 0;
	case 1:
		return s & d;
	case 2:
		return s & ~d;
	case 3:
		return s;
	case 4:
		return ~s & d;
	case 5:
		return d;
	case 6:
		return s ^ d;
	case 7:
		return s | d;
	case 8:
		return ~(s | d);
	case 9:
		return ~(s ^ d);
	case 10:
		return ~d;
	case 11:
		return s | ~d;
	case 12:
		return ~s;
	case 13:
		return ~s | d;
	case 14:
		return ~(s & d);
	default:
		return 255;
	}
}
struct fixture
{
	context ctx{1};
	compiler compile{ctx};
	render_pipeline_cache pipelines{compile};
	shader_cache shaders{compile};
	vertex_fetch_engine fetch{ctx};
	std::unique_ptr<vertex_shader> vertex;
	vertex_program_snapshot vertex_image;
	id<MTLBuffer> vertices;
	converted_geometry geometry = convert_array_geometry(guest_topology::triangles, 0, 3, provoking_vertex::last);
	unsigned cases = 0;
	fixture()
	{
		constexpr unsigned xyzw = 2u | (1u << 12) | (2u << 10) | (3u << 8);
		const std::array<std::array<unsigned, 4>, 1> code{{{(63u << 15) | (7u << 10) | (1u << 30), (1u << 22) | (xyzw >> 9), (xyzw & 511) << 23, (15u << 13) | (15u << 17) | (1u << 7) | 1u}}};
		test::vertex_program imported;
		imported.base_address = 137;
		imported.entry = 138;
		imported.data.assign(4, 0xffffffffu);
		imported.data.insert(imported.data.end(), code[0].begin(), code[0].end());
		imported.instruction_mask.set(1);
		vertex_image = vertex_program_snapshot::capture(imported);
		std::fill(imported.data.begin(), imported.data.end(), 0xffffffffu);
		imported.instruction_mask.reset();
		vertex = std::make_unique<vertex_shader>(compile, vertex_image.code(), vertex_image.options(false));
		std::vector<std::byte> bytes;
		for (float v : {-1.f, -1.f, .5f, 1.f, 3.f, -1.f, .5f, 1.f, -1.f, 3.f, .5f, 1.f})
			for (unsigned shift : {24u, 16u, 8u, 0u})
				bytes.push_back(std::byte(std::bit_cast<unsigned>(v) >> shift));
		vertices = make_upload(ctx, bytes);
	}
	void shader_cache_contracts()
	{
		shader_cache cache(compile, 2);
		auto code = std::vector(vertex_image.code().begin(), vertex_image.code().end());
		const auto original = code;
		auto vo = vertex_image.options(false);
		std::vector<std::uint8_t> live(vo.live_instructions.begin(), vo.live_instructions.end());
		const auto saved_live = live;
		vo.live_instructions = live;
		const auto libraries_before = compile.library_statistics().created;
		require(cache.vertex_mask(code, vo) == 0, "Cached vertex mask is incorrect");
		require(compile.library_statistics().created == libraries_before && cache.vertex_statistics().compiled == 0,
			"Mask preflight compiled a Metal library");
		const auto vp = cache.vertex(code, vo);
		const auto prior_vertex = cache.vertex_statistics();
		bool bad_mask = false;
		try { auto invalid = vo; invalid.live_instructions = std::span(live).first(0); invalid.entry = 1024; (void)cache.vertex(code, invalid); }
		catch (const std::invalid_argument&) { bad_mask = true; }
		require(bad_mask && cache.vertex_statistics().translated == prior_vertex.translated && cache.vertex_statistics().evicted == prior_vertex.evicted,
			"Invalid vertex entry disturbed cached programs");
		std::fill(code.begin(), code.end(), std::array<unsigned,4>{0xffffffff,0xffffffff,0xffffffff,0xffffffff});
		std::fill(live.begin(), live.end(), 0);
		vo.live_instructions = saved_live;
		require(cache.vertex(original, vo) == vp && cache.vertex_statistics().translated == 1,
			"Cache retained caller code/liveness span instead of ownership");
		auto point = vo;
		point.point_output = true;
		require(cache.vertex(original, point) != vp, "Point output option collided with triangle shader");
		auto literal = program(source, 1);
		fragment_program_options fo;
		require(cache.fragment_mask(literal, fo) == 0 && cache.fragment_statistics().compiled == 0,
			"Fragment mask preflight compiled a library");
		const auto fp = cache.fragment(literal, fo);
		const auto changed = program(color{0,0,1,1}, 1);
		require(cache.fragment(changed, fo) == fp && cache.fragment_statistics().translated == 1,
			"Mutable fragment literal specialized program key");
		auto trailing = changed;
		trailing.insert(trailing.end(), 19, std::byte{0xff});
		require(cache.fragment(trailing, fo) == fp, "Trailing bytes after END changed shader identity");
		auto flat = fo; flat.flat_colors = true;
		const auto old_flat = cache.fragment(literal, flat);
		(void)cache.fragment(literal, fo); // MRU
		auto half = fo; half.float32_exports = false;
		(void)cache.fragment(literal, half);
		require(cache.fragment_statistics().entries == 2 && cache.fragment_statistics().evicted == 1 &&
			cache.fragment(literal, fo) == fp, "Shader LRU lost most recently used program");
		require(cache.fragment(literal, flat) != old_flat, "Evicted fragment shader was not reconstructed");
		const auto before = cache.fragment_statistics();
		bool refused = false;
		try { (void)cache.fragment(std::span(literal).first(20), fo); }
		catch (const std::invalid_argument&) { refused = true; }
		require(refused && cache.fragment_statistics().translated == before.translated && cache.fragment_statistics().evicted == before.evicted,
			"Truncated literal reused shader or evicted good entries");
		const auto changed_code = program(source, 2);
		require(cache.fragment(changed_code, fo) != fp, "Changed instruction layout reused old shader");
		shader_cache quota_probe(compile, 4);
		(void)quota_probe.fragment(literal, fo);
		const auto charge = quota_probe.fragment_statistics().host_bytes;
		shader_cache quota(compile, 4, charge);
		(void)quota.fragment(literal, fo);
		(void)quota.fragment(literal, flat);
		require(quota.fragment_statistics().host_bytes <= charge && quota.fragment_statistics().entries <= 1,
			"Generated source/decoded image bypassed host quota");
		shader_cache tiny(compile, 4, 1), disabled(compile, 0);
		(void)tiny.fragment(literal, fo);
		(void)disabled.fragment(literal, fo);
		(void)disabled.fragment(literal, fo);
		require(tiny.fragment_statistics().bypassed == 1 && tiny.fragment_statistics().entries == 0 &&
			disabled.fragment_statistics().translated == 2 && disabled.fragment_statistics().compiled == 2 &&
			disabled.fragment_statistics().entries == 0, "Tiny/disabled shader quota retained entries");
		for (unsigned failure = 0; failure < 2; ++failure)
		{
			bool rejected = false;
			try { shader_cache invalid(compile, failure ? 1 : 4097, failure ? 0 : 1024); }
			catch (const std::invalid_argument&) { rejected = true; }
			require(rejected, "Shader cache accepted invalid capacity/budget");
		}
		cache.clear();
		require(cache.fragment_statistics().entries == 0 && cache.vertex_statistics().host_bytes == 0 && fp->function().library && vp->function().library,
			"Clear invalidated held shader or retained cache charges");
		std::cout << "Native shader cache:owned code/options/liveness,dynamic literals/trailing bytes,preflight without compile,LRU/host quota,invalid layout isolation and held owners passed\n";
	}
	void prepared_cache_contracts()
	{
		shader_cache cache(compile, 16), foreign(compile, 16);
		auto code = std::vector(vertex_image.code().begin(), vertex_image.code().end());
		auto vo = vertex_image.options(false);
		auto bytes = program(source, 1);
		fragment_program_options fo;
		const auto vp = cache.prepare_vertex(code, vo);
		const auto fp = cache.prepare_fragment(bytes, fo);
		require(vp.texture_mask() == 0 && fp.texture_mask() == 0 && !cache.vertex_statistics().compiled && !cache.fragment_statistics().compiled,
			"Prepared programs compiled before preflight");
		const auto vertex = cache.vertex(vp, vo);
		const auto fragment = cache.fragment(fp, fo);
		const auto vs = cache.vertex_statistics(), fs = cache.fragment_statistics();
		for (unsigned i = 0; i < 100; ++i)
		{
			require(cache.vertex(vp, vo) == vertex && cache.fragment(fp, fo) == fragment, "Prepared hit changed compiled identity");
		}
		require(cache.vertex_statistics().lookups == vs.lookups && cache.fragment_statistics().lookups == fs.lookups &&
			cache.vertex_statistics().candidates == vs.candidates && cache.fragment_statistics().candidates == fs.candidates &&
			cache.vertex_statistics().prepared_hits == vs.prepared_hits + 100 && cache.fragment_statistics().prepared_hits == fs.prepared_hits + 100,
			"Prepared hits repeated program index/word searches");
		// Different programs occupy different buckets; options deliberately share
		// code fingerprints and must still be compared in full.
		for (unsigned i = 1; i < 16; ++i)
		{
			auto other = bytes;
			other[1] = std::byte(1u | (i << 1)); // MOV destination, same literal.
			(void)cache.fragment_mask(other, fo);
		}
		const auto before = cache.fragment_statistics();
		require(cache.fragment(bytes, fo) == fragment && cache.fragment_statistics().candidates == before.candidates + 1,
			"Indexed hit scanned unrelated shader programs");
		auto flat = fo; flat.flat_colors = true;
		require(cache.fragment(fp, flat) != fragment && cache.fragment(fp, fo) == fragment, "Prepared option collision reused a different shader");
		auto point = vo; point.point_output = true;
		require(cache.vertex(vp, point) != vertex && cache.vertex(vp, vo) == vertex, "Prepared vertex ignored final options");
		for (unsigned bad = 0; bad < 4; ++bad)
		{
			bool refused = false;
			try
			{
				if (bad == 0) (void)foreign.vertex(vp, vo);
				if (bad == 1) (void)foreign.fragment(fp, fo);
				if (bad == 2) (void)cache.vertex(shader_cache::prepared_vertex{}, vo);
				if (bad == 3) (void)cache.fragment(shader_cache::prepared_fragment{}, fo);
			}
			catch (const std::invalid_argument&) { refused = true; }
			require(refused, "Foreign/empty prepared program bypassed compiler identity");
		}
		std::fill(code.begin(), code.end(), std::array<unsigned, 4>{});
		std::fill(bytes.begin(), bytes.end(), std::byte{0xff});
		cache.clear();
		require(cache.vertex(vp, vo) == vertex && cache.fragment(fp, fo) == fragment && !cache.fragment_statistics().entries,
			"Prepared owner retained caller memory or became invalid after clear");
		shader_cache uncached(compile, 0);
		const auto once = uncached.prepare_fragment(program(source, 1), fo);
		(void)uncached.fragment(once, fo);
		require(uncached.fragment_statistics().translated == 1 && uncached.fragment_statistics().compiled == 1 && !uncached.fragment_statistics().entries,
			"Disabled cache repeated preflight translation");
		std::cout << "Prepared shader hits:100 per stage without index/word searches;16 program buckets,option collisions,foreign/empty owners,mutable caller storage,clear and zero-capacity preflight passed\n";
	}
	void prewarm_contracts()
	{
		const auto wait_ready = [](const auto& ready)
		{
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
			while (!ready())
			{
				if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Native prewarm did not complete");
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		};
		compiler warm(ctx);
		shader_cache cache(warm, 4), foreign(warm, 4);
		auto code = std::vector(vertex_image.code().begin(), vertex_image.code().end());
		auto bytes = program(source, 1);
		const auto vo = vertex_image.options(false);
		fragment_program_options fo;
		const auto prepared = cache.prepare_fragment(bytes, fo);
		const color alternate{0, 0, 1, 1};
		const auto initial_constants = cache.prepare_constants(ctx, prepared, bytes);
		const auto alternate_constants = cache.prepare_constants(ctx, prepared, program(alternate, 1));
		bool foreign_constants = false, invalid_constants = false;
		try { (void)foreign.prepare_constants(ctx, prepared, bytes); } catch (const std::invalid_argument&) { foreign_constants = true; }
		try { (void)cache.prepare_constants(ctx, prepared, std::span(bytes).first(20)); } catch (const std::invalid_argument&) { invalid_constants = true; }
		require(foreign_constants && invalid_constants && warm.tasks().statistics().admitted == 0 && !cache.fragment_statistics().compiled,
			"Fragment literal preparation compiled a shader or accepted foreign/truncated input");
		vertex_fetch_request request;
		request.vertex_count = 3;
		request.attribute_mask = 1;
		request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
		ctx.begin();
		vertex_inputs input;
		input.attributes = fetch.decode(ctx, request, vertices, nil);
		ctx.submit().wait();
		auto requirements = vertex_shader::plan_uploads(input, vo.flat_result_inputs, nullptr);
		const auto fragment_uploads = fragment_shader::plan_uploads(prepared.texture_mask());
		requirements.insert(requirements.end(), fragment_uploads.begin(), fragment_uploads.end());
		const auto cold_uploads = ctx.prepare_uploads(requirements);
		require(warm.tasks().statistics().admitted == 0 && !cache.vertex_statistics().compiled && !cache.fragment_statistics().compiled,
			"Draw uniform reservation required a compiled shader");
		const auto vp = cache.prewarm_vertex(cache.prepare_vertex(code, vo), vo);
		const auto fp = cache.prewarm_fragment(prepared, fo);
		require(warm.tasks().statistics().admitted == 2 && !cache.vertex_statistics().compiled && !cache.fragment_statistics().compiled,
			"Native shader prewarm waited/materialized or failed to start both stages");
		std::fill(code.begin(), code.end(), std::array<unsigned, 4>{});
		std::fill(bytes.begin(), bytes.end(), std::byte{0xff});
		for (unsigned i = 0; i < 100; ++i)
		{
			(void)cache.prewarm_vertex(vp, vo); (void)cache.prewarm_fragment(fp, fo);
		}
		wait_ready([&] { return cache.ready(vp) && cache.ready(fp); });
		const auto vertex = cache.vertex(vp, vo);
		const auto fragment = cache.fragment(fp, fo);
		require(warm.tasks().statistics().admitted == 2 && warm.library_statistics().created == 2 &&
			cache.vertex_statistics().prewarmed == 1 && cache.fragment_statistics().prewarmed == 1 &&
			cache.vertex_statistics().prewarm_reused == 100 && cache.fragment_statistics().prewarm_reused == 100 &&
			!cache.vertex_statistics().prewarm_waited && !cache.fragment_statistics().prewarm_waited,
			"Native stage materialization recompiled/waited or lost prepared owners");
		auto flat = fo; flat.flat_colors = true;
		const auto changed = cache.prewarm_fragment(fp, flat);
		const auto lookups = cache.fragment_statistics().lookups;
		const auto changed_shader = cache.fragment(changed, flat);
		require(changed_shader != fragment && cache.fragment_statistics().lookups == lookups,
			"Final prewarm options repeated program lookup or reused another variant");
		bool refused = false;
		try { (void)foreign.prewarm_fragment(fp, fo); } catch (const std::invalid_argument&) { refused = true; }
		require(refused, "Foreign prewarm shader owner accepted");

		render_pipeline_cache pipelines(warm, 2), other(warm, 2);
		render_pipeline_configuration config; config.colors[0] = MTLPixelFormatRGBA32Float;
		const auto admitted = warm.tasks().statistics().admitted;
		const auto pa = pipelines.prewarm(config, vertex->function(), *fragment);
		require(pa.has_value(), "Native pipeline prewarm refused empty service");
		auto dynamic = config;
		dynamic.state.stencil_test = dynamic.state.two_sided_stencil = true;
		dynamic.state.front.write_mask = dynamic.state.back.write_mask = 0;
		dynamic.state.front.reference = 41; dynamic.state.back.reference = 79;
		const auto pb = pipelines.prewarm(dynamic, vertex->function(), *fragment);
		require(pb && warm.tasks().statistics().admitted == admitted + 1 && pipelines.statistics().pending_reused == 1,
			"Effective pending pipeline key did not deduplicate changed dynamic references");
		wait_ready([&] { return pa->ready(); });
		const auto pc = pipelines.prewarm(config, vertex->function(), *fragment);
		require(pc && warm.tasks().statistics().admitted == admitted + 1,
			"Completed but unconsumed native PSO was compiled again");
		const auto a = pipelines.get(*pa, std::chrono::milliseconds(0));
		const auto b = pipelines.get(*pb, std::chrono::milliseconds(0));
		require(a.pipeline == b.pipeline && a.depth_stencil == b.depth_stencil && b.state.front.reference == 41 && b.state.back.reference == 79 &&
			a.state.front.reference == 0 && pipelines.statistics().created == 1 && !pipelines.statistics().prewarm_waited,
			"Prepared native PSOs lost independent dynamic references or created duplicate results");
		refused = false;
		try { (void)other.get(*pa); } catch (const std::invalid_argument&) { refused = true; }
		require(refused, "Foreign prepared pipeline accepted");
		const auto hot = pipelines.statistics();
		for (unsigned i = 0; i < 100; ++i)
		{
			config.state.front.reference = i;
			const auto p = pipelines.prewarm(config, vertex->function(), *fragment);
			require(p && p->ready() && pipelines.get(*p).pipeline == a.pipeline, "Warm preflight changed native PSO");
		}
		require(pipelines.statistics().fast_hits == hot.fast_hits + 100 && pipelines.statistics().descriptor_lookups == hot.descriptor_lookups,
			"Warm preflight rebuilt native descriptors");
		config.state.front.reference = 0;
		const auto render_pixels = [&](color value, const fragment_shader::constant_binding& constants, const upload_plan& uploads)
		{
			auto description = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:4 height:4 mipmapped:NO];
			description.storageMode = MTLStorageModePrivate; description.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			const auto target = make_texture(ctx, description);
			const auto output = make_buffer(ctx, 1024, buffer_storage::shared);
			ctx.begin();
			vertex_context vc; vc.vertex_count = 3;
			bool foreign_variant = false;
			try { (void)changed_shader->prepare(ctx, {}, constants, {}); } catch (const std::invalid_argument&) { foreign_variant = true; }
			require(foreign_variant, "Fragment literal binding escaped to a different shader variant");
			bindings vb, fb;
			ctx.with_uploads(uploads, [&]
			{
				vb = vertex->prepare(ctx, vc, geometry, input);
				fb = fragment->prepare(ctx, {}, constants, {});
			});
			auto pass = [MTL4RenderPassDescriptor new];
			pass.colorAttachments[0].texture = target; pass.colorAttachments[0].loadAction = MTLLoadActionClear; pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			render_commands draw(ctx, pass);
			raster_state raster; raster.viewport = {0, 0, 4, 4, 0, 1}; raster.scissor = {0, 0, 4, 4};
			draw.draw_geometry(geometry, a.pipeline, a.depth_stencil, raster, vb, fb); draw.end();
			texture_region region; region.size = MTLSizeMake(4, 4, 1);
			texture_to_buffer(ctx, target, region, output, 0, 256); ctx.submit().wait();
			const auto* words = static_cast<const float*>(output.contents);
			for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 4; ++x) for (unsigned c = 0; c < 4; ++c)
				require(std::bit_cast<unsigned>(words[y * 64 + 4 * x + c]) == std::bit_cast<unsigned>(value[c]), "Prewarmed native pipeline used stale literal bits");
		};
		render_pixels(source, initial_constants, cold_uploads);
		render_pixels(alternate, alternate_constants, ctx.prepare_uploads(requirements));
		std::cout << "Fragment literal preparation: no compiler job, invalid/foreign inputs refused, exact 32 pixels after shader materialization\n";
		std::cout << "Draw uniform reservation: admitted before either library job, consumed exactly by both native stages, 32 bit-exact pixels\n";
		auto altered = config; altered.state.colors[0].write_mask = 1;
		const auto mask = pipelines.prewarm(altered, vertex->function(), *fragment);
		require(mask && pipelines.get(*mask).pipeline != a.pipeline, "Pending pipeline key ignored color mask");
		altered.state.depth_test = altered.state.depth_write = true;
		const auto depth = pipelines.prewarm(altered, vertex->function(), *fragment);
		require(depth && pipelines.get(*depth).depth_stencil != a.depth_stencil && pipelines.statistics().entries == 2,
			"Pending key ignored depth state or exceeded LRU capacity");
		altered.state.colors[0].write_mask = 2;
		const auto retained = pipelines.prewarm(altered, vertex->function(), *fragment);
		require(retained.has_value(), "Native pipeline clear fixture admission failed");
		pipelines.clear();
		require(pipelines.get(*retained).pipeline && !pipelines.statistics().entries, "Old prewarm result repopulated cleared pipeline cache");
		render_pipeline_cache uncached(warm, 0);
		const auto once = uncached.prewarm(config, vertex->function(), *fragment);
		const auto before_get = warm.tasks().statistics().admitted;
		require(once && uncached.get(config, vertex->function(), *fragment).pipeline && warm.tasks().statistics().admitted == before_get &&
			uncached.statistics().created == 1 && uncached.statistics().bypassed == 1 && !uncached.statistics().entries,
			"Ordinary disabled-cache getter recompiled its pending prewarm");

		shader_cache probe(warm, 4);
		(void)probe.prepare_fragment(program(source, 2), fo);
		const auto budget = probe.fragment_statistics().host_bytes;
		shader_cache tight(warm, 4, budget);
		const auto base = tight.prepare_fragment(program(source, 2), fo);
		const auto bounded = tight.prewarm_fragment(base, fo);
		require(!tight.fragment_statistics().entries && !tight.fragment_statistics().host_bytes && tight.fragment(bounded, fo),
			"Prewarm-owned source key bypassed stage host quota or invalidated evicted owner");
		warm.clear_libraries();
		const auto old_jobs = warm.tasks().statistics().admitted;
		const auto old = probe.prewarm_fragment(probe.prepare_fragment(program(source, 2), fo), fo);
		require(warm.tasks().statistics().admitted == old_jobs + 1, "Clear-lifetime fixture did not create an unconsumed native job");
		probe.clear(); warm.clear_libraries();
		require(probe.fragment(old, fo) && !probe.fragment_statistics().entries && !warm.library_statistics().entries,
			"Pending shader ownership failed after stage/source-cache clear");
		render_pipeline_cache bounded_pipelines(warm, 2);
		std::vector<render_pipeline_cache::prepared_pipeline> pending;
		for (unsigned i = 0; i < 64; ++i)
		{
			auto variant = config;
			variant.state.colors[0].write_mask = i & 15;
			variant.state.depth_test = true;
			variant.state.depth_compare = 0x200u + (i >> 4);
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
			for (;;)
			{
				auto p = bounded_pipelines.prewarm(variant, vertex->function(), *fragment);
				if (p) { pending.push_back(*p); break; }
				if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Native pending pipeline window did not admit64 distinct states");
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
		auto overflow = config; overflow.state.depth_test = true; overflow.state.depth_compare = 0x204u;
		require(!bounded_pipelines.prewarm(overflow, vertex->function(), *fragment), "Native pending pipeline index exceeded64 owners");
		const auto requests = warm.tasks().statistics().admitted;
		for (unsigned i = 0; i < pending.size(); ++i)
		{
			const auto result = bounded_pipelines.get(pending[i]);
			require(result.pipeline && static_cast<unsigned>(result.state.depth_compare) == (i >> 4) &&
				result.state.colors[0].write_mask == (i & 15), "Bounded native pending pipeline keys collided");
		}
		require(warm.tasks().statistics().admitted == requests && bounded_pipelines.statistics().created == 64 &&
			bounded_pipelines.statistics().entries == 2, "Pending PSOs recompiled during materialization or exceeded LRU budget");
		std::cout << "Native prewarm:both stages start before materialization,200 duplicate owners/two jobs,final options/host quota/clear lifetime;pending PSO full keys/dynamic refs/ready reuse/LRU/foreign guards;32 exact GPU pixels with changed literals;no cache-hit descriptor rebuild passed\n";
		std::cout << "Bounded native pipeline prewarm:64 distinct depth/color-mask keys,65th refusal,64 owned results with no new jobs andtwo-entry completed LRU passed\n";
	}
	void cache_contracts()
	{
		render_pipeline_configuration config;
		config.colors[0] = MTLPixelFormatRGBA32Float;
		fragment_shader fragment(compile, program(source, 1));
		render_pipeline_cache cache(compile, 2);
		const auto a = cache.get(config, vertex->function(), fragment);
		config.state.front.reference = 41;
		config.state.back.reference = 79;
		config.state.stencil_test = config.state.two_sided_stencil = true;
		config.state.front.write_mask = config.state.back.write_mask = 0;
		// Enabled all-keep/always/zero-write stencil yields the same descriptor.
		const auto dynamic = cache.get(config, vertex->function(), fragment);
		require(dynamic.pipeline == a.pipeline && dynamic.depth_stencil == a.depth_stencil &&
			dynamic.state.front.reference == 41 && dynamic.state.back.reference == 79,
			"Cache retained stale references or keyed ineffective stencil enable");
		const auto hot = cache.statistics();
		for (unsigned reference = 0; reference < 100; ++reference)
		{
			config.state.front.reference = reference + 256;
			config.state.back.reference = 255 - reference;
			const auto hit = cache.get(config, vertex->function(), fragment);
			require(hit.pipeline == a.pipeline && hit.state.front.reference == reference && hit.state.back.reference == 255 - reference,
				"Fast pipeline hit retained dynamic stencil references");
		}
		require(cache.statistics().fast_hits == hot.fast_hits + 100 && cache.statistics().descriptor_lookups == hot.descriptor_lookups &&
			cache.statistics().candidates == hot.candidates, "Unchanged pipeline rebuilt descriptor or searched index");
		config.state.depth_test = config.state.depth_write = true;
		const auto b = cache.get(config, vertex->function(), fragment);
		require(b.depth_stencil != a.depth_stencil, "Depth write/test failed to separate native states");
		config.state.depth_test = config.state.depth_write = false;
		(void)cache.get(config, vertex->function(), fragment); // A becomes MRU.
		config.state.colors[0].write_mask = 1;
		const auto c = cache.get(config, vertex->function(), fragment);
		require(c.pipeline != a.pipeline && cache.statistics().evicted == 1 && cache.statistics().entries == 2,
			"Write mask did not create bounded pipeline variant");
		config.state.colors[0].write_mask = 15;
		require(cache.get(config, vertex->function(), fragment).pipeline == a.pipeline, "LRU evicted MRU pipeline");
		config.state.depth_test = config.state.depth_write = true;
		require(cache.get(config, vertex->function(), fragment).pipeline != b.pipeline, "Evicted pipeline was not rebuilt");
		const auto before = cache.statistics();
		for (unsigned failure = 0; failure < 3; ++failure)
		{
			bool refused = false;
			try
			{
				auto invalid = config;
				if (failure == 0) invalid.shader_logic.enabled = true;
				if (failure == 1) invalid.colors[0] = MTLPixelFormatDepth32Float;
				auto fn = vertex->function();
				if (failure == 2) fn = library_function(fn.library, @"missing_vertex_entry");
				(void)cache.get(invalid, fn, fragment);
			}
			catch (const std::exception&) { refused = true; }
			require(refused, "Pipeline cache concealed invalid contract/function");
		}
		require(cache.statistics().created == before.created && cache.statistics().evicted == before.evicted &&
			cache.statistics().entries == before.entries, "Failed pipeline creation changed cached entries");
		compiler separate(ctx);
		vertex_shader alternate(separate, vertex_image.code(), vertex_image.options(false));
		const auto other = cache.get(config, alternate.function(), fragment);
		require(other.pipeline != cache.get(config, vertex->function(), fragment).pipeline,
			"Different native libraries with identical entry names collided");
		cache.clear();
		require(cache.statistics().entries == 0 && a.pipeline && a.depth_stencil, "Clear invalidated retained state");
		render_pipeline_cache uncached(compile, 0);
		(void)uncached.get(config, vertex->function(), fragment);
		(void)uncached.get(config, vertex->function(), fragment);
		require(uncached.statistics().created == 2 && uncached.statistics().bypassed == 2 && !uncached.statistics().entries,
			"Disabled pipeline cache retained entries");
		bool refused = false;
		try { render_pipeline_cache excessive(compile, 4097); }
		catch (const std::invalid_argument&) { refused = true; }
		require(refused, "Unbounded pipeline cache accepted");
		std::cout << "Native pipeline LRU:100 fast hits without render descriptor/index searches,effective descriptors,dynamic stencil references,depth/write-mask variants,eviction,failure isolation,clear and disabled capacity passed\n";
	}
	void run(const std::vector<std::uint32_t>& methods, const std::array<color, 4>& expected,
			 float expected_depth = .75f, unsigned expected_stencil = 0x5a, bool forced = false, bool normalized = false, unsigned outputs = 1,
			 std::array<unsigned, 4> region = {0, 0, 4, 4}, bool visible = true, bool half_format = false)
	{
		auto live = methods;
		const auto frozen = pipeline_register_snapshot::capture(live);
		std::fill(live.begin(), live.end(), 0xffffffffu);
		render_pipeline_configuration config;
		fragment_program_options options;
		raster_state raster;
		fragment_context constants;
		raster.viewport = {0, 0, 4, 4, 0, 1};
		std::array<std::optional<surface_color>, 4> formats{};
		for (unsigned slot = 0; slot < outputs; ++slot)
			formats[slot] = half_format ? surface_color::rgba16f : normalized ? surface_color::argb8
																			  : surface_color::rgba32f;
		const color s = normalized ? color{64.f / 255, 128.f / 255, 192.f / 255, 160.f / 255} : source;
		const color d = normalized ? color{16.f / 255, 32.f / 255, 48.f / 255, 80.f / 255} : destination;
		auto code = program(s, outputs);
		test::fragment_program imported;
		imported.pointer = code.data();
		imported.ucode_length = static_cast<std::uint32_t>(code.size());
		imported.ctrl = rsx_fragment_float32;
		imported.mrt_buffers_count = outputs;
		const auto fragment_image = fragment_program_snapshot::capture(imported, 1 << 20);
		vertex_program_options vertex_options;
		vertex_context vc;
		apply_shader_program_snapshots(vertex_image, fragment_image, false, true, config, vertex_options, options, vc, constants);
		apply_pipeline_register_snapshot(frozen, formats, half_format ? surface_color::rgba16f : normalized ? surface_color::argb8
																											: surface_color::rgba32f,
										 4, 4, config, options, raster, constants, forced);
		auto initial_values = s;
		for (auto& lane : initial_values)
			lane *= .5f;
		auto initial_code = program(initial_values, outputs);
		imported.pointer = initial_code.data();
		const auto initial_image = fragment_program_snapshot::capture(imported, 1 << 20);
		const auto fragment = shaders.fragment(initial_image.code(), options);
		std::fill(initial_code.begin(), initial_code.end(), std::byte{0xff});
		std::fill(code.begin(), code.end(), std::byte{0xff});
		const auto original_pipeline = pipelines.get(config, vertex->function(), *fragment);
		const auto pipeline = pipelines.get(config, vertex->function(), *fragment);
		require(pipeline.pipeline == original_pipeline.pipeline && pipeline.depth_stencil == original_pipeline.depth_stencil,
			"Real RSX draw failed to consume a cached native pipeline");
		std::array<id<MTLTexture>, 4> targets{};
		std::array<id<MTLBuffer>, 4> output{};
		for (unsigned slot = 0; slot < outputs; ++slot)
		{
			auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:config.colors[slot] width:4 height:4 mipmapped:NO];
			td.storageMode = MTLStorageModePrivate;
			td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
			targets[slot] = make_texture(ctx, td);
			output[slot] = make_buffer(ctx, 1024, buffer_storage::shared);
		}
		auto td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8 width:4 height:4 mipmapped:NO];
		td.storageMode = MTLStorageModePrivate;
		td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
		const auto depth = make_texture(ctx, td);
		const auto z = make_buffer(ctx, 1024, buffer_storage::shared), stencil = make_buffer(ctx, 1024, buffer_storage::shared);
		ctx.begin();
		vertex_fetch_request request;
		request.vertex_count = 3;
		request.attribute_mask = 1;
		request.attributes[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27), 1u << 29};
		vertex_inputs input;
		input.attributes = fetch.decode(ctx, request, vertices, nil);
		vc.vertex_count = 3;
		const auto vb = vertex->prepare(ctx, vc, geometry, input), fb = fragment->prepare(ctx, constants, fragment_image.code(), {});
		auto pass = [MTL4RenderPassDescriptor new];
		for (unsigned slot = 0; slot < outputs; ++slot)
		{
			auto a = pass.colorAttachments[slot];
			a.texture = targets[slot];
			a.loadAction = MTLLoadActionClear;
			a.storeAction = MTLStoreActionStore;
			a.clearColor = MTLClearColorMake(d[0], d[1], d[2], d[3]);
		}
		pass.depthAttachment.texture = depth;
		pass.depthAttachment.loadAction = MTLLoadActionClear;
		pass.depthAttachment.storeAction = MTLStoreActionStore;
		pass.depthAttachment.clearDepth = .75;
		pass.stencilAttachment.texture = depth;
		pass.stencilAttachment.loadAction = MTLLoadActionClear;
		pass.stencilAttachment.storeAction = MTLStoreActionStore;
		pass.stencilAttachment.clearStencil = 0x5a;
		render_commands commands(ctx, pass);
		commands.draw_geometry(geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb);
		commands.end();
		texture_region whole;
		whole.size = MTLSizeMake(4, 4, 1);
		for (unsigned slot = 0; slot < outputs; ++slot)
			texture_to_buffer(ctx, targets[slot], whole, output[slot], 0, 256);
		texture_to_buffer(ctx, depth, whole, z, 0, 256, 0, texture_plane::depth);
		texture_to_buffer(ctx, depth, whole, stencil, 0, 256, 0, texture_plane::stencil);
		ctx.submit().wait();
		for (unsigned y = 0; y < 4; ++y)
			for (unsigned x = 0; x < 4; ++x)
			{
				const bool inside = visible && x >= region[0] && y >= region[1] && x - region[0] < region[2] && y - region[1] < region[3];
				for (unsigned slot = 0; slot < outputs; ++slot)
					for (unsigned c = 0; c < 4; ++c)
					{
						const auto actual = normalized ? static_cast<const unsigned char*>(output[slot].contents)[y * 256 + x * 4 + std::array{2u, 1u, 0u, 3u}[c]] / 255.f : half_format ? static_cast<float>(static_cast<const _Float16*>(output[slot].contents)[y * 128 + x * 4 + c])
																																														 : static_cast<const float*>(output[slot].contents)[y * 64 + x * 4 + c];
						const auto wanted = inside ? expected[slot][c] : d[c];
						if (std::abs(actual - wanted) > 2e-6f)
							throw std::runtime_error("Frozen RSX pipeline case=" + std::to_string(cases) + " slot=" + std::to_string(slot) + " lane=" + std::to_string(c) + " actual=" + std::to_string(actual) + " expected=" + std::to_string(wanted));
					}
				require(static_cast<const float*>(z.contents)[y * 64 + x] == (inside ? expected_depth : .75f), "Captured RSX depth test/write or scissor produced wrong depth");
				require(static_cast<const unsigned char*>(stencil.contents)[y * 256 + x] == (inside ? expected_stencil : 0x5a), "Captured RSX stencil reference/mask or scissor produced wrong stencil");
			}
		++cases;
	}
	void all()
	{
		for (unsigned mask = 0; mask < 16; ++mask)
		{
			auto m = registers();
			m[0x324 / 4] = ((mask & 1) ? 0x10000 : 0) | ((mask & 2) ? 0x100 : 0) | ((mask & 4) ? 1 : 0) | ((mask & 8) ? 0x1000000 : 0);
			color expected = destination;
			for (unsigned c = 0; c < 4; ++c)
				if (mask & (1u << c))
					expected[c] = source[c];
			run(m, {expected});
		}
		for (unsigned enable : {0u, 1u, 0x80000000u})
			for (unsigned write : {0u, 1u, 2u})
			{
				auto m = registers();
				m[0xa74 / 4] = enable;
				m[0xa70 / 4] = write;
				run(m, {source}, enable && write ? .5f : .75f);
			}
		{
			auto m = registers();
			m[0xa74 / 4] = 1;
			m[0xa70 / 4] = 1;
			m[0xa6c / 4] = 0x200;
			run(m, {destination}, .75f, 0x5a, false, false, 1, {0, 0, 4, 4}, false);
		}
		for (unsigned ref : {0x31u, 0x102au, 0x3f7bu, 0xffabu})
			for (unsigned mask : {0u, 15u, 240u, 255u})
			{
				auto m = registers();
				m[0x328 / 4] = 1;
				m[0x334 / 4] = ref;
				m[0x32c / 4] = mask;
				m[0x344 / 4] = 0x1e01;
				m[0x350 / 4] = 0x200;
				m[0x354 / 4] = ~ref;
				m[0x34c / 4] = ~mask;
				run(m, {source}, .75f, (0x5a & ~mask) | (ref & mask));
			}
		for (bool forced : {false, true})
		{
			auto m = registers();
			m[0x310 / 4] = 1;
			m[0x318 / 4] = 0x10001;
			color expected = source;
			if (!forced)
				for (unsigned c = 0; c < 4; ++c)
					expected[c] += destination[c];
			run(m, {expected}, .75f, 0x5a, forced);
		}
		{
			auto m = registers();
			m[0x310 / 4] = 1;
			m[0x31c / 4] = 0x80604020;
			m[0x314 / 4] = 0x80038001;
			m[0x318 / 4] = 0x80048002;
			const color k{96.f / 255, 64.f / 255, 32.f / 255, 128.f / 255};
			color expected;
			for (unsigned c = 0; c < 4; ++c)
				expected[c] = source[c] * k[c] + destination[c] * (1 - k[c]);
			run(m, {expected});
		}
		{
			auto m = registers();
			m[0x310 / 4] = 1;
			m[0x31c / 4] = 0x38003400;
			m[0x37c / 4] = 0x38003a00;
			m[0x314 / 4] = 0x80038001;
			m[0x318 / 4] = 0x80048002;
			run(m, {color{.15625f, .375f, .6875f, .375f}}, .75f, 0x5a, false, false, 1, {0, 0, 4, 4}, true, true);
		}
		for (unsigned op = 0; op < 16; ++op)
		{
			auto m = registers();
			m[0x374 / 4] = 1;
			m[0x378 / 4] = 0x1500 + op;
			m[0x310 / 4] = 1;
			m[0x314 / 4] = m[0x318 / 4] = m[0x320 / 4] = 0xffffffffu;
			color expected;
			for (unsigned c = 0; c < 4; ++c)
				expected[c] = (logic(op, std::array{64u, 128u, 192u, 160u}[c], std::array{16u, 32u, 48u, 80u}[c]) & 255) / 255.f;
			run(m, {expected}, .75f, 0x5a, false, true);
		}
		{
			auto m = registers();
			m[0x370 / 4] = (2u << 4) | (4u << 8) | (8u << 12);
			run(m, {source, color{source[0], destination[1], destination[2], destination[3]}, color{destination[0], source[1], destination[2], destination[3]}, color{destination[0], destination[1], source[2], destination[3]}}, .75f, 0x5a, false, false, 4);
		}
		{
			auto m = registers();
			m[0x8c0 / 4] = (65535u << 16) | 1;
			m[0x8c4 / 4] = (2u << 16) | 1;
			run(m, {source}, .75f, 0x5a, false, false, 1, {1, 1, 3, 2});
			m[0x8c0 / 4] = (4u << 16) | 5;
			run(m, {destination}, .75f, 0x5a, false, false, 1, {0, 0, 0, 0}, false);
			m = registers();
			m[0x183c / 4] = 1;
			m[0x1830 / 4] = 0x408;
			run(m, {destination}, .75f, 0x5a, false, false, 1, {0, 0, 4, 4}, false);
		}
		for (unsigned failure = 0; failure < 3; ++failure)
		{
			auto m = registers();
			if (failure == 0)
			{
				m[0x310 / 4] = 1;
				m[0x314 / 4] = 0xffffffffu;
			}
			else if (failure == 1)
				m[0x1834 / 4] = 0xffffffffu;
			else
			{
				m[0x183c / 4] = 1;
				m[0x1830 / 4] = 0xffffffffu;
			}
			const auto snapshot = pipeline_register_snapshot::capture(m);
			render_pipeline_configuration config;
			config.colors[0] = MTLPixelFormatRGBA32Float;
			fragment_program_options options;
			options.color_count = 3;
			raster_state raster;
			raster.scissor = {1, 2, 3, 4};
			raster.front_stencil_reference = 77;
			fragment_context uniforms;
			uniforms.blend_color = {.25f, .5f, .75f, 1};
			bool refused = false;
			try
			{
				apply_pipeline_register_snapshot(snapshot, {surface_color::argb8, {}, {}, {}}, surface_color::argb8, 4, 4, config, options, raster, uniforms);
			}
			catch (const std::invalid_argument&)
			{
				refused = true;
			}
			require(refused && config.colors[0] == MTLPixelFormatRGBA32Float && !config.state.colors[0].enabled && options.color_count == 3 && raster.scissor.x == 1 && raster.scissor.height == 4 && raster.front_stencil_reference == 77 && uniforms.blend_color == color{.25f, .5f, .75f, 1}, "Failed RSX pipeline binding published partial pipeline/shader/raster/uniform state");
		}
		{
			auto bytes = program(source, 1);
			test::fragment_program imported;
			imported.pointer = bytes.data();
			imported.ucode_length = bytes.size();
			imported.ctrl = rsx_fragment_float32 | rsx_program_flat;
			const auto image = fragment_program_snapshot::capture(imported, 1 << 20);
			render_pipeline_configuration config;
			config.state.output_rop.srgb = true;
			vertex_program_options vo;
			vo.entry = 77;
			fragment_program_options fo;
			fo.color_count = 3;
			vertex_context vc;
			vc.output_mask = 0x77;
			fragment_context fc;
			fc.alpha_test = 1;
			bool refused = false;
			try
			{
				apply_shader_program_snapshots(vertex_image, image, false, true, config, vo, fo, vc, fc);
			}
			catch (const std::invalid_argument&)
			{
				refused = true;
			}
			require(refused && config.state.output_rop.srgb && vo.entry == 77 && fo.color_count == 3 && vc.output_mask == 0x77 && fc.alpha_test == 1, "Incompatible analysed stage metadata published partial shader/pipeline/uniform state");
		}
		std::cout << cases << " real RSX vertex/fetch/fragment draws imported analysed owned shader snapshots and bound frozen NV4097 color/MRT masks,depth,stencil references,masks,byte/FP16 blend constants,logic-over-blend,forced blend disable,scissor and all-face cull after live source mutation and captured literal refresh;failed bindings preserve all outputs\n";
	}
};
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			fixture f;
			f.shader_cache_contracts();
			f.prepared_cache_contracts();
			f.prewarm_contracts();
			f.cache_contracts();
			f.all();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
