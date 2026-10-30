#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4Transfer.h"
#include "M4SurfaceSwizzle.h"
#include "M4SurfaceBanked.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <optional>
#include <thread>

namespace
{
using namespace metal4;
constexpr auto source = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void first(device uint* out [[buffer(0)]]) { out[0] = 0x11223344u; }
kernel void second(device uint* out [[buffer(0)]]) { out[0] = 0x55667788u; }
vertex float4 vertex_main(uint i [[vertex_id]]) {
    return float4(float2(i == 1 ? 3 : -1, i == 2 ? 3 : -1), 0, 1);
}
vertex float4 vertex_other(uint i [[vertex_id]]) {
    return float4(float2(i == 1 ? 3 : -1, i == 2 ? 3 : -1), 0, 1);
}
fragment float4 color_main() { return float4(.25, .5, .75, 1); }
fragment float4 color_other() { return float4(1, .75, .5, .25); }
)msl";
void require(bool condition, const char* message)
{
	if (!condition) throw std::runtime_error(message);
}
template <typename F> void refuses(F function)
{
	bool failed = false;
	try { function(); } catch (const std::exception&) { failed = true; }
	require(failed, "Invalid helper compilation request succeeded");
}
void dispatch(context& ctx, id<MTLComputePipelineState> pipeline, id<MTLBuffer> output)
{
	bindings resources;
	resources.set_buffer(0, output);
	ctx.keep_alive(pipeline);
	const auto table = resources.snapshot(ctx);
	const auto encoder = ctx.compute();
	[encoder setComputePipelineState:pipeline];
	[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}
void production_sharing(context& ctx)
{
	helper_catalog catalog(ctx);
	const auto before = catalog.statistics();
	transfer_engine first(ctx);
	const auto one = catalog.statistics();
	require(one.compiler_instances == 1 && one.libraries.created == before.libraries.created + 1 &&
		one.compute_created == before.compute_created + 4, "Transfer helper did not use one shared compiler/library and four native kernels");
	transfer_engine second(ctx);
	context other;
	transfer_engine third(other);
	helper_catalog second_catalog(other);
	const auto three = catalog.statistics();
	require(three.compute_created == one.compute_created && three.libraries.created == one.libraries.created &&
		three.compute_reused == one.compute_reused + 8 && second_catalog.statistics().compute_created == three.compute_created,
		"Repeated helpers across contexts duplicated native compiler/library/PSO work");
	surface_swizzle_transfer swizzle(ctx);
	surface_banked_transfer banked(ctx);
	const auto nested = catalog.statistics();
	surface_swizzle_transfer swizzle_other(other);
	surface_banked_transfer banked_other(other);
	require(catalog.statistics().libraries.created == nested.libraries.created && catalog.statistics().compute_created == nested.compute_created,
		"Nested surface helpers duplicated their color/depth/swizzle/banked programs");
	const std::array<std::uint32_t, 4> words{0x12345678, 0x87654321, 0x0a0b0c0d, 0x01020304};
	const auto input = make_upload(other, std::as_bytes(std::span(words)));
	const auto output = make_buffer(other, sizeof(words), buffer_storage::shared);
	other.begin();
	third.copy(other, input, 0, output, 0, sizeof(words), 4);
	other.submit().wait();
	const std::array<std::uint32_t, 4> expected{0x78563412, 0x21436587, 0x0d0c0b0a, 0x04030201};
	require(std::equal(expected.begin(), expected.end(), static_cast<const std::uint32_t*>(output.contents)),
		"Shared native helper lost per-context bindings or exact transfer results");
	std::cout << "Three transfer owners/two contexts: one library, four native PSOs; nested surface helper creation deduplicated; exact GPU transfer passed\n";
}
void concurrent_requests(context& owner)
{
	helper_catalog shared(owner);
	const auto before = shared.statistics();
	std::array<id<MTLLibrary>, 8> libraries{};
	std::array<id<MTLComputePipelineState>, 8> pipelines{};
	std::array<std::exception_ptr, 8> errors{};
	std::array<std::thread, 8> threads;
	for (unsigned i = 0; i < threads.size(); ++i)
		threads[i] = std::thread([&, i]
		{
			@autoreleasepool
			{
				try
				{
					context ctx;
					helper_catalog catalog(ctx);
					libraries[i] = catalog.library(source, @"Concurrent native helper fixture");
					pipelines[i] = catalog.compute(libraries[i], @"first");
				}
				catch (...) { errors[i] = std::current_exception(); }
			}
		});
	for (auto& thread : threads) thread.join();
	for (unsigned i = 0; i < threads.size(); ++i)
	{
		if (errors[i]) std::rethrow_exception(errors[i]);
		require(libraries[i] == libraries[0] && pipelines[i] == pipelines[0], "Concurrent helper misses produced different native library/PSO objects");
	}
	const auto after = shared.statistics();
	require(after.libraries.created == before.libraries.created + 1 && after.compute_created == before.compute_created + 1 &&
		after.compute_reused == before.compute_reused + 7, "Concurrent helper misses were not deduplicated");
	std::cout << "Eight concurrent contexts share one successful library/compute compilation\n";
}
void keys_and_failures(context& ctx)
{
	helper_catalog catalog(ctx, 1);
	const auto library = catalog.library(source);
	NSMutableString* name = [NSMutableString stringWithString:@"first"];
	const auto first = catalog.compute(library, name);
	[name setString:@"second"];
	require(catalog.compute(library, @"first") == first, "Mutable caller function name corrupted a stored helper key");
	(void)catalog.compute(library, name);
	require(catalog.statistics().compute_created == 2 && catalog.statistics().compute_entries == 1 &&
		catalog.statistics().compute_evicted == 1, "Compute helper LRU did not enforce capacity");
	const auto before = catalog.statistics();
	refuses([&] { (void)catalog.library("This is not a Metal shader"); });
	refuses([&] { (void)catalog.compute(library, @"missing_function"); });
	require(catalog.statistics().compute_created == before.compute_created && catalog.statistics().compute_evicted == before.compute_evicted &&
		catalog.statistics().libraries.created == before.libraries.created, "Failed helper compilation published/evicted a successful entry");
	helper_render_configuration config;
	config.colors[0] = MTLPixelFormatRGBA8Unorm;
	const auto render = catalog.render(library, @"vertex_main", @"color_main", config);
	require(catalog.render(library, @"vertex_main", @"color_main", config) == render, "Identical render helper did not reuse native PSO");
	for (unsigned variant = 0; variant < 5; ++variant)
	{
		auto changed = config;
		if (variant == 0) changed.colors[0] = MTLPixelFormatBGRA8Unorm;
		if (variant == 1) changed.masks[0] = MTLColorWriteMaskRed;
		if (variant == 2) changed.samples = 2;
		(void)catalog.render(library, variant == 3 ? @"vertex_other" : @"vertex_main", variant == 4 ? @"color_other" : @"color_main", changed);
	}
	require(catalog.statistics().render_created == 6 && catalog.statistics().render_evicted == 5 &&
		catalog.statistics().render_entries == 1, "Render helper format/mask/sample/function keys or capacity differ");
	const auto valid = catalog.statistics();
	auto invalid = config;
	invalid.samples = 0;
	refuses([&] { (void)catalog.render(library, @"vertex_main", @"color_main", invalid); });
	refuses([&] { (void)catalog.render(library, @"vertex_main", @"missing_fragment", config); });
	require(catalog.statistics().render_created == valid.render_created && catalog.statistics().render_evicted == valid.render_evicted,
		"Failed render helper request evicted/published native state");
	catalog.clear();
	require(catalog.statistics().compute_entries == 0 && catalog.statistics().render_entries == 0 && catalog.statistics().libraries.entries == 0,
		"Cleared helper catalogue retained cached objects");
	helper_catalog bypass(ctx, 0);
	const auto uncached = bypass.library(source);
	for (unsigned i = 0; i < 2; ++i)
	{
		(void)bypass.compute(uncached, @"first");
		(void)bypass.render(uncached, @"vertex_main", @"color_main", config);
	}
	require(bypass.statistics().compute_created == 2 && bypass.statistics().compute_bypassed == 2 &&
		bypass.statistics().render_created == 2 && bypass.statistics().render_bypassed == 2 &&
		bypass.statistics().compute_entries == 0 && bypass.statistics().render_entries == 0,
		"Zero-capacity helper catalogue retained PSOs");
	// Existing dynamic compiler cache quotas remain independent.
	compiler isolated(ctx, 0);
	(void)isolated.library(source);
	(void)isolated.library(source);
	require(isolated.library_statistics().created == 2 && isolated.library_statistics().bypassed == 2,
		"Shared helpers changed explicit dynamic compiler cache semantics");
	std::cout << "Owned function keys, render profiles, bounded LRU/bypass, compile-error isolation and independent dynamic compiler quotas passed\n";
}
void pending_lifetime(context& ctx)
{
	const auto gate = [ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> gate; ~release() { gate.signaledValue = 1; } } guard{gate};
	const auto output = make_buffer(ctx, 4, buffer_storage::shared);
	std::optional<submission> pending;
	@autoreleasepool
	{
		helper_catalog catalog(ctx, 1);
		const auto library = catalog.library(source);
		const auto pipeline = catalog.compute(library, @"first");
		ctx.begin();
		ctx.wait_event(gate, 1);
		dispatch(ctx, pipeline, output);
		(void)catalog.compute(library, @"second");
		catalog.clear();
		pending = ctx.submit();
	}
	require(!pending->ready(), "Helper lifetime fixture did not remain pending");
	gate.signaledValue = 1;
	pending->wait();
	require(*static_cast<const std::uint32_t*>(output.contents) == 0x11223344,
		"Eviction/clear/catalogue destruction invalidated pending native helper code");
	std::cout << "Pending helper kernel survives eviction, clear, compiler/catalogue destruction and external handle release\n";
}
}
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx;
			production_sharing(ctx);
			concurrent_requests(ctx);
			keys_and_failures(ctx);
			pending_lifetime(ctx);
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
