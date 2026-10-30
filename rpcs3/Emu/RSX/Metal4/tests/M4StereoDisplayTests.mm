#include "M4Display.h"
#include "M4HostAccessFixture.h"

#import <AppKit/NSApplication.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>

namespace
{
using namespace metal4;
using namespace std::chrono_literals;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void rejects(F&& work)
{
	bool failed = false; try { work(); } catch (const std::exception&) { failed = true; }
	require(failed,"Invalid stereo display operation accepted");
}
struct capture_state
{
	std::shared_ptr<test::host_access_fixture> access;
	unsigned calls = 0;
	std::function<void(unsigned)> after;
};
struct fixture
{
	std::unique_ptr<context> ctx = std::make_unique<context>(2);
	std::shared_ptr<heap_budget> budget = std::make_shared<heap_budget>(*ctx,16u << 20);
	std::shared_ptr<test::host_access_fixture> access = std::make_shared<test::host_access_fixture>();
	std::shared_ptr<guarded_guest_memory> memory = std::make_shared<guarded_guest_memory>(access);
	std::shared_ptr<tile_state> tiles = std::make_shared<tile_state>();
	std::shared_ptr<capture_state> captures = std::make_shared<capture_state>(capture_state{access});
	cpu_capture_dispatch capture;
	std::unique_ptr<surface_cache> surfaces;
	std::unique_ptr<display_resolver> resolver;
	display_snapshot image;
	explicit fixture(display_format format = display_format::xrgb8)
	{
		ctx->set_buffer_budget(budget);
		capture = tile_capture_dispatch([state = captures](const auto& work)
		{
			state->access->execute([&](auto&) { ++state->calls; work(); });
			const auto hook = state->after; if (hook) hook(state->calls);
		},tiles);
		surfaces = std::make_unique<surface_cache>(*ctx,memory,16u << 20,budget,capture);
		resolver = std::make_unique<display_resolver>(*ctx,memory,[](unsigned offset,unsigned location,std::size_t)
		{
			if (location) throw std::invalid_argument("Stereo display source is not local");
			return offset;
		},budget,4096,capture,tiles);
		image = {0,2,2,format == display_format::rgbx16f ? 48u : 24u,format};
		image.offset = 65536 - 32 * image.pitch + 256; image.right_offset = 65536 + 256;
		std::fill_n(access->bytes,131072,std::byte{0xd3});
		for (unsigned eye = 0; eye < 2; ++eye) for (unsigned y = 0; y < 2; ++y)
		{
			std::vector<std::byte> row(2 * display_pixel_bytes(format));
			for (unsigned x = 0; x < 2; ++x)
			{
				const auto rgba = color(eye,x,y);
				if (format == display_format::rgbx16f)
				{
					for (unsigned c = 0; c < 4; ++c)
					{
						const _Float16 half = rgba[c] / 255.f;
						std::uint16_t bits; std::memcpy(&bits,&half,2);
						row[x * 8 + c * 2] = std::byte(bits >> 8); row[x * 8 + c * 2 + 1] = std::byte(bits & 255);
					}
				}
				else
				{
					row[x * 4] = std::byte{0};
					row[x * 4 + 1] = std::byte(rgba[format == display_format::xrgb8 ? 0 : 2]);
					row[x * 4 + 2] = std::byte(rgba[1]);
					row[x * 4 + 3] = std::byte(rgba[format == display_format::xrgb8 ? 2 : 0]);
				}
			}
			memory->write({display_eye(image,eye).offset + y * image.pitch,row.size()},row);
		}
	}
	static std::array<unsigned,4> color(unsigned eye,unsigned x,unsigned y)
	{
		return eye ? std::array{16u + x * 16,128u + y * 16,64u,255u} : std::array{64u + x * 32,32u + y * 32,16u,255u};
	}
	bool current(const display_frame_job& job)
	{
		bool result = false; checked_cpu_capture(capture,[&] { result = job.current(); }); return result;
	}
	void replace_left(bool unmap = false)
	{
		access->execute([&](auto&)
		{
			memory->begin_mapping_change({0,65536});
			if (unmap) access->unmap_first(); else access->replace_first();
			memory->end_mapping_change({0,65536});
		});
	}
};
std::vector<std::byte> render(context& ctx,present_image image)
{
	presenter engine(ctx);
	auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:8 height:4 mipmapped:NO];
	d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageRenderTarget;
	const auto target = make_texture(ctx,d); const auto out = make_buffer(ctx,128,buffer_storage::shared);
	image.destination = {0,0,8,4}; image.stereo.mode = stereo_mode::side_by_side;
	ctx.begin(); engine.encode(ctx,image,target);
	texture_to_buffer(ctx,target,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(8,4,1)},out,0); ctx.submit().wait();
	const auto bytes = map_buffer(out,0,128); return {bytes.begin(),bytes.end()};
}
void verify_pattern(std::span<const std::byte> bytes)
{
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 8; ++x)
	{
		const auto c = fixture::color(x / 4,(x % 4) / 2,y / 2);
		for (unsigned lane = 0; lane < 4; ++lane)
		{
			const auto expected = c[lane == 0 ? 2 : lane == 2 ? 0 : lane];
			require(std::to_integer<unsigned>(bytes[(y * 8 + x) * 4 + lane]) == expected,
				"Resolved stereo eye/format/padding pixel differs");
		}
	}
}
void eye_versions()
{
	unsigned hits = 0;
	for (const auto format : {display_format::xrgb8,display_format::xbgr8,display_format::rgbx16f})
	{
		fixture f(format);
		const auto before = std::vector<std::byte>(f.access->bytes,f.access->bytes + 131072);
		auto job = f.resolver->prepare_frame(f.image,*f.surfaces);
		auto image = job.finish(); require(image.right_eye && f.current(job),"Stereo frame omitted an eye orjoint current receipt");
		verify_pattern(render(*f.ctx,image));
		require(std::equal(before.begin(),before.end(),f.access->bytes),"Stereo VM/native upload changed source or30-line padding bytes");
		const auto uploads = f.resolver->statistics().uploads, created = f.budget->statistics().created;
		const auto encoded = f.ctx->encoding_statistics();
		for (unsigned n = 0; n < 20; ++n)
		{
			const auto same = f.resolver->prepare_frame(f.image,*f.surfaces).finish();
			require(same.texture == image.texture && same.right_eye->texture == image.right_eye->texture,"Stereo ready hits replaced owned eyes"); ++hits;
		}
		require(f.resolver->statistics().uploads == uploads && f.budget->statistics().created == created &&
			f.ctx->encoding_statistics().compute_encoders == encoded.compute_encoders,"Stereo ready hits uploaded orallocated new images");
		const std::array mark{std::byte{0x51}};
		f.memory->write({f.image.offset + static_cast<unsigned>(2 * display_pixel_bytes(format)),1},mark);
		f.memory->write({f.image.offset + 10 * f.image.pitch,1},mark);
		require(f.current(job),"Pitch or30-line eye padding invalidated active eye receipts");
		f.memory->write({*f.image.right_offset + 1,1},mark);
		require(!f.current(job),"Right-eye CPU write escaped joint receipt"); rejects([&] { (void)job.finish(); });
		const auto fresh = f.resolver->prepare_frame(f.image,*f.surfaces).finish();
		require(fresh.texture == image.texture && fresh.right_eye->texture != image.right_eye->texture,"Right-eye mutation discarded unrelated left eye orreused stale right eye");
	}
	std::cout << "Stereo display:3AVformats/96patterned BGRA pixels;distinct eyes inphysical banks;30-line/pitch padding unchanged;" << hits << " paired readyhits without upload/allocation/compute;right-only active write invalidates joint frame andpreserves left hit\n";
}
void joint_validation_and_mapping()
{
	fixture f;
	auto job = f.resolver->prepare_frame(f.image,*f.surfaces);
	const auto start = f.captures->calls;
	bool changed = false;
	f.captures->after = [&,start](unsigned call)
	{
		if (call != start + 1) return;
		f.captures->after = {}; const std::array value{std::byte{0x57}};
		f.memory->write({f.image.offset + 1,1},value); changed = true;
	};
	rejects([&] { (void)job.finish(); });
	require(changed && f.captures->calls >= start + 3,"Stereo frame missed mutation between separately validated eyes");
	auto current = f.resolver->prepare_frame(f.image,*f.surfaces); const auto original = current.finish();
	f.replace_left(); require(!f.current(current),"Stereo left remap escaped joint source validation");
	rejects([&] { (void)current.finish(); });
	const auto replacement = f.resolver->prepare_frame(f.image,*f.surfaces).finish();
	require(replacement.texture != original.texture && replacement.right_eye->texture == original.right_eye->texture,"Remap followed replaced eye storage ordiscarded unrelated right eye");
	f.replace_left(true); rejects([&] { (void)f.resolver->prepare_frame(f.image,*f.surfaces).finish(); });
	std::cout << "Stereo frame:joint final producer epoch refuses left mutation afterleft-only validation;actual OS backing replacement/unmap rejects old receipt;fresh replacement doesnot reuse stale left ordiscard right eye\n";
}
void preflight_and_native_sources()
{
	fixture f;
	const auto calls = f.captures->calls; const auto created = f.budget->statistics().created;
	auto invalid = f.image; invalid.right_offset = 0x7ffffffc;
	rejects([&] { (void)f.resolver->prepare_frame(invalid,*f.surfaces); });
	rejects([&] { (void)f.resolver->prepare(f.image,*f.surfaces); });
	display_resolver small(*f.ctx,f.memory,[](unsigned o,unsigned,std::size_t) { return o; },f.budget,1,f.capture,f.tiles);
	rejects([&] { (void)small.prepare_frame(f.image,*f.surfaces); });
	require(f.captures->calls == calls && f.budget->statistics().created == created,"Invalid stereo second eye orcombined byte budget captured/allocated first eye");
	// Match the real provider's privileged/sudo view: base CPU permissions
	// may be NO while revision metadata can still validate a readable mapping.
	f.access->pin_first_alias();
	display_resolver native(*f.ctx,f.memory,[](unsigned o,unsigned,std::size_t) { return o + 0x40000; },f.budget,4096,f.capture,f.tiles);
	for (unsigned eye = 0; eye < 2; ++eye)
	{
		const auto descriptor = display_eye(f.image,eye);
		const auto surface = f.surfaces->acquire({descriptor.offset + 0x40000,{2,2,1,surface_color::argb8,false},descriptor.pitch,{}});
		f.surfaces->clear(surface,0xf0,eye ? 0xff0000ffu : 0xffff0000u,0,255,{0,0,2,2});
	}
	f.surfaces->synchronize(); // Same completed surface boundary as GSRender::flip.
	f.access->protect_first(false);
	present_image image;
	try { image = native.prepare_frame(f.image,*f.surfaces).finish(); }
	catch (const std::exception& e)
	{
		throw std::runtime_error("Native stereo sources:snapshots=" + std::to_string(native.statistics().native_snapshots) +
			",uploads=" + std::to_string(native.statistics().uploads) + ",error=" + e.what());
	}
	const auto bytes = render(*f.ctx,image);
	for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 8; ++x)
	{
		const auto red = x < 4;
		require(bytes[(y * 8 + x) * 4] == std::byte(red ? 0 : 255) && bytes[(y * 8 + x) * 4 + 1] == std::byte{0} &&
			bytes[(y * 8 + x) * 4 + 2] == std::byte(red ? 255 : 0) && bytes[(y * 8 + x) * 4 + 3] == std::byte{255},"Native stereo surface eye colors differ");
	}
	require(native.statistics().native_snapshots == 2 && native.statistics().uploads == 0,"Stereo native sources fell back to VM upload");
	f.access->protect_first(true);
	std::cout << "Stereo preflight:invalid second local offset/single-eye misuse/combined byte budget rejected beforecapture/allocation;two native GPU surface eyes,32exact BGRA pixels,zeroVMuploads withleft CPU mapping protected\n";
}
void orphan_and_callback_lifetime()
{
	for (const bool native : {false,true})
	{
		std::shared_ptr<heap_budget> budget;
		std::optional<display_frame_job> orphan;
		id<MTLSharedEvent> gate;
		{
			fixture f; budget = f.budget;
			if (native)
			{
				for (unsigned eye = 0; eye < 2; ++eye)
				{
					const auto descriptor = display_eye(f.image,eye);
					const auto surface = f.surfaces->acquire({descriptor.offset,{2,2,1,surface_color::argb8,false},descriptor.pitch,{}});
					f.surfaces->clear(surface,0xf0,0xff000000,0,255,{0,0,2,2});
					for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x)
					{
						const auto c = fixture::color(eye,x,y);
						f.surfaces->clear(surface,0xf0,0xff000000 | (c[0] << 16) | (c[1] << 8) | c[2],0,255,{x,y,1,1});
					}
				}
				f.surfaces->synchronize();
			}
			gate = [f.ctx->device() newSharedEvent];
			const std::array dependencies{display_upload_dependency{gate,1}};
			orphan.emplace(f.resolver->prepare_frame(f.image,*f.surfaces,dependencies));
			require(!orphan->ready(),"Stereo upload didnot retain native event gate");
			if (native) require(f.resolver->statistics().native_snapshots == 2 && f.resolver->statistics().uploads == 0,"Native stereo orphan used VM upload");
			const auto bytes = budget->statistics().reserved_bytes;
			rejects([&] { (void)orphan->finish(1ms); });
			require(budget->statistics().reserved_bytes == bytes && !orphan->ready(),"Stereo timeout freed pending eye backing");
			f.resolver->clear();
		}
		@autoreleasepool
		{
			gate.signaledValue = 1;
			const auto image = orphan->finish(5s); orphan.reset();
			context reader; verify_pattern(render(reader,image));
		}
		require(budget->statistics().reserved_bytes == 0,"Orphan stereo eye/receipt/context backing retained charges afterfinal release");
	}

	{
		fixture f;
		std::optional<display_frame_job> job{f.resolver->prepare_frame(f.image,*f.surfaces)};
		const auto start = f.captures->calls;
		bool reentry = false, destroyed = false;
		f.captures->after = [&,start](unsigned call)
		{
			if (call != start + 3) return;
			try { (void)job->finish(); } catch (const std::logic_error&) { reentry = true; }
			f.captures->after = {}; job.reset(); f.resolver.reset(); f.surfaces.reset(); destroyed = true;
		};
		const auto image = job->finish();
		require(reentry && destroyed && image.right_eye,"Stereo final callback didnot exercise reentry/job/cache destruction");
		verify_pattern(render(*f.ctx,image));
	}
	std::cout << "Stereo ownership:gated VM ANDnative two-eye upload/timeout/cache clear outlives resolver/surfaces/context/provider caller;new native contexts read64exact patterned BGRA pixels;allcharges retire;final joint callback destroys job/caches andreentry refuses\n";
}
void packed_surface_and_guards()
{
	{
		fixture f; f.access->pin_first_alias();
		display_resolver resolver(*f.ctx,f.memory,[](unsigned o,unsigned,std::size_t) { return o + 0x40000; },f.budget,4096,f.capture,f.tiles);
		const auto packed = f.surfaces->acquire({f.image.offset + 0x40000,{2,34,1,surface_color::argb8,false},f.image.pitch,{}});
		f.surfaces->clear(packed,0xf0,0xff00ff00,0,255,{0,0,2,34});
		f.surfaces->clear(packed,0xf0,0xffff0000,0,255,{0,0,2,2});
		f.surfaces->clear(packed,0xf0,0xff0000ff,0,255,{0,32,2,2});
		f.surfaces->synchronize(); f.access->protect_first(false);
		const auto image = resolver.prepare_frame(f.image,*f.surfaces).finish();
		const auto bytes = render(*f.ctx,image);
		for (unsigned y = 0; y < 4; ++y) for (unsigned x = 0; x < 8; ++x)
			require(bytes[(y * 8 + x) * 4] == std::byte(x < 4 ? 0 : 255) && bytes[(y * 8 + x) * 4 + 1] == std::byte{0} &&
				bytes[(y * 8 + x) * 4 + 2] == std::byte(x < 4 ? 255 : 0),"Packed native frame sampled30-line green gap orwrong eye crop");
		require(resolver.statistics().native_snapshots == 2 && resolver.statistics().uploads == 0,"Packed native eyes used VM upload");
		f.access->protect_first(true);
	}
	{
		fixture f; auto original = f.resolver->prepare_frame(f.image,*f.surfaces); auto moved = std::move(original);
		rejects([&] { (void)original.ready(); }); rejects([&] { (void)original.finish(); });
		bool foreign = false;
		std::jthread worker([&] { try { (void)moved.ready(); } catch (const std::logic_error&) { foreign = true; } }); worker.join();
		require(foreign,"Stereo frame accepted foreign thread");
		f.access->execute([&](auto&) { rejects([&] { (void)moved.finish(); }); rejects([&] { (void)moved.cancel_pending(); }); });
		rejects([&] { (void)moved.finish(0ms); }); verify_pattern(render(*f.ctx,moved.finish()));
	}
	for (const bool shutdown : {false,true})
	{
		fixture f; const auto gate = [f.ctx->device() newSharedEvent];
		const std::array dependencies{display_upload_dependency{gate,1}};
		auto job = f.resolver->prepare_frame(f.image,*f.surfaces,dependencies);
		if (shutdown) f.ctx->shutdown(); else require(job.cancel_pending(),"Unsubmitted stereo jobs couldnot cancel");
		rejects([&] { (void)job.finish(); });
		require(f.resolver->statistics().published == 0,"Cancelled/stopped stereo jobs published source views");
		gate.signaledValue = 1;
	}
	std::cout << "Stereo packed native surface:2exact crops separated by30green rows,32BGRApixels,zeroVMuploads;move/foreign/VM-epoch/timeout guards;real queued cancel andshutdown refuse whole-frame output andcache publication\n";
}
}
int main()
{
	@autoreleasepool
	{
		try
		{
			require(NSApp == nil,"Stereo display test began with NSApplication");
			eye_versions(); joint_validation_and_mapping(); preflight_and_native_sources(); orphan_and_callback_lifetime(); packed_surface_and_guards();
			require(NSApp == nil,"Stereo display test created NSApplication");
			std::cout << "Stereo display checks passed;noApp/view/window/renderer/guestVM started\n";
		}
		catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
	}
}
