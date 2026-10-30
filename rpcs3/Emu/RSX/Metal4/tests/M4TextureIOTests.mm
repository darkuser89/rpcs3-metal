#include "M4DrawExecutor.h"
#include "M4DrawExecutorFixtures.h"
#include "M4GuardedGuestMemory.h"
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

namespace
{
using namespace metal4;
void require(bool value, const char* message)
{
	if (!value) throw std::runtime_error(message);
}
template <typename F>
void conflict(F&& operation)
{
	bool failed = false;
	try { operation(); }
	catch (const texture_revision_conflict&) { failed = true; }
	require(failed, "Native texture IO transition escaped preparation");
}
class provider final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<provider>
{
	std::mutex mutex;
	std::vector<std::byte> bytes = std::vector<std::byte>(32 << 20);
  public:
	unsigned notifications = 0;
	std::function<void(guest_surface_range, bool)> on_map;
	// Exact metadata for the fixed owned allocation;IO remaps select actual
	// vector offsets andnever change its physical owner orread pixel bytes.
	std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range range) override
	{
		validate_guest_surface_range(range);
		if(range.address>bytes.size()||range.size>bytes.size()-range.address)throw std::out_of_range("Texture IO physical identity bounds");
		return std::vector<storage_identity_span>{{shared_from_this(),range.address,range}};
	}
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& operation) override
	{
		if (auto* view = guest_memory_epoch::current(*this)) { operation(*view); return; }
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		operation(*this);
	}
	std::byte* map(guest_surface_range range, bool write) override
	{
		require(guest_memory_epoch::current(*this) == this, "Texture IO fixture map escaped producer exclusion");
		if (range.address > bytes.size() || range.size > bytes.size() - range.address) throw std::out_of_range("Texture IO fixture mapping");
		if (on_map) on_map(range, write);
		return bytes.data() + range.address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(a, b); }
	void written(guest_surface_range) noexcept override { ++notifications; }
};
struct fixture
{
	context ctx{1};
	std::shared_ptr<provider> access = std::make_shared<provider>();
	std::shared_ptr<guarded_guest_memory> memory = std::make_shared<guarded_guest_memory>(access);
	std::shared_ptr<io_mapping_epoch> io = std::make_shared<io_mapping_epoch>();
	std::array<unsigned, 32> table{};
	unsigned capture_depth = 0, captures = 0, translations = 0;
	std::function<void(unsigned)> on_translate;
	std::function<void(unsigned)> on_capture_end;
	cpu_capture_dispatch capture;
	std::shared_ptr<heap_budget> budget = std::make_shared<heap_budget>(ctx, 32 << 20);
	std::unique_ptr<surface_cache> surfaces;
	std::unique_ptr<guest_texture_cache> textures;
	std::unique_ptr<draw_executor> executor;
	fixture()
	{
		for (unsigned i = 0; i < table.size(); ++i) table[i] = i << 20;
		capture = [this](const auto& operation)
		{
			access->execute([&](auto&)
			{
				++captures; ++capture_depth;
				try { operation(); }
				catch (...) { --capture_depth; throw; }
				--capture_depth;
			});
			const auto hook = on_capture_end;
			if (hook) hook(captures);
		};
		ctx.set_buffer_budget(budget);
		surfaces = std::make_unique<surface_cache>(ctx, memory, 32 << 20, budget);
		textures = std::make_unique<guest_texture_cache>(ctx, memory, [this](unsigned offset, unsigned location, std::size_t size) { return translate(offset, location, size); },
			4 << 20, budget, 256, texture_capture_bindings{capture, io});
		executor = std::make_unique<draw_executor>(ctx, *surfaces, *textures);
	}
	unsigned translate(unsigned offset, unsigned location, std::size_t size)
	{
		unsigned result = 0;
		access->execute([&](auto&)
		{
			const auto hook = on_translate;
			if (hook) hook(++translations); else ++translations;
			if (location == 0) { result = offset; return; }
			require(location == 1 && size && size <= 0x1'0000'0000ull - offset, "Texture logical range is invalid");
			const auto first = offset >> 20;
			const auto last = (std::uint64_t(offset) + size - 1) >> 20;
			require(last < table.size(), "Texture IO fixture table exhausted");
			for (auto entry = first; entry <= last; ++entry)
				require(table[entry] == table[first] + ((entry - first) << 20), "Fixture texture IO is disjoint");
			result = table[first] + (offset & 0xfffff);
		});
		return result;
	}
	void aba(unsigned entry)
	{
		access->execute([&](auto&)
		{
			table[entry] ^= 0x100000;
			io->notify({entry << 20, 0x100000});
			table[entry] ^= 0x100000;
			io->notify({entry << 20, 0x100000});
		});
	}
	void fill(const guest_texture_layout& plan)
	{
		std::vector<std::byte> bytes(plan.storage_bytes);
		for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = std::byte(std::array{255u, 20u, 40u, 60u}[i % 4]);
		checked_cpu_capture(capture, [&]
		{
			for (const auto& part : texture_storage_segments(plan.offset, plan.location, bytes.size(),
				[this](auto offset, auto location, auto size) { return translate(offset, location, size); }))
				memory->write({part.address, part.size}, std::span(bytes).subspan(part.offset, part.size));
		});
	}
	bool current(const guest_texture_revision& ticket)
	{
		bool result = false;
		checked_cpu_capture(capture, [&] { result = ticket.current(); });
		return result;
	}
};
fragment_texture_registers words(unsigned offset, bool main = true)
{
	auto state = test::executor_fixture::registers(); test::executor_fixture::texture_registers(state, offset);
	auto result = fragment_texture_snapshot::capture(state.registers, 1).at(0);
	if (!main) result.format = (result.format & ~3u) | 1;
	return result;
}
void ticket_cases(fixture& f)
{
	const auto source = words(0x100100);
	f.fill(plan_guest_texture(source));
	// Byte revisions are deliberately larger than initial IO revisions.
	for (unsigned i = 0; i < 64; ++i) f.memory->write({source.offset, 1}, std::array{std::byte{255}});
	const auto old = f.textures->resolve_versioned(source, *f.surfaces);
	const auto revision = f.memory->revision({source.offset, 64});
	const auto notifications = f.access->notifications;
	require(f.current(old.revision) && revision > f.io->current(), "Texture fixture did not exercise a larger VM revision");
	f.aba(1);
	require(f.memory->revision({source.offset, 64}) == revision && f.access->notifications == notifications && !f.current(old.revision), "Identical IO ABA escaped texture ticket or touched VM");
	const auto fresh = f.textures->resolve_versioned(source, *f.surfaces);
	f.aba(7); require(f.current(fresh.revision), "Unrelated IO entry invalidated texture ticket");
	const auto local_source = words(0x200100, false); f.fill(plan_guest_texture(local_source));
	const auto local = f.textures->resolve_versioned(local_source, *f.surfaces);
	f.aba(2); require(f.current(local.revision), "IO mapping invalidated local-only texture");
	// Complete mip/face storage crosses an IO boundary;the first entry alone
	// cannot guard the last face or final mip bytes.
	auto cube = words(0xfff00); cube.format |= 4; cube.format = (cube.format & 0xffff) | (3u << 16);
	const auto plan = plan_guest_texture(cube);
	require(plan.layers == 6 && plan.mips == 3 && cube.offset + plan.storage_bytes > 0x100000, "Cube fixture did not cross its final IO entry");
	f.fill(plan); const auto faces = f.textures->resolve_versioned(cube, *f.surfaces);
	f.aba((cube.offset + plan.storage_bytes - 1) >> 20);
	require(!f.current(faces.revision), "Last cube/mip IO entry retained stale native ticket");
	// Initial probing and final validation must bracket their queries,not
	// sample a new generation only after a changing translation callback.
	f.textures->clear(); f.translations = 0;
	const auto uploads = f.textures->statistics().uploads;
	f.on_translate = [&](unsigned call) { if (call == 1) { require(f.capture_depth == 1, "Initial texture probe escaped CPU capture"); f.aba(1); } };
	conflict([&] { (void)f.textures->resolve_versioned(source, *f.surfaces); });
	f.on_translate = {};
	require(f.textures->statistics().uploads == uploads && !f.capture_depth, "Initial IO conflict performed native upload or leaked capture");
	(void)f.textures->resolve_versioned(source, *f.surfaces); // Warm cache:initial,prepare,hit-query,final-query.
	f.translations = 0;
	f.on_translate = [&](unsigned call) { if (call == 4) { require(f.capture_depth == 1, "Final texture validation escaped CPU capture"); f.aba(1); } };
	conflict([&] { (void)f.textures->resolve_versioned(source, *f.surfaces); });
	f.on_translate = {};
	// Native unversioned entry points use the same guarded preparation too.
	for (bool surface : {false, true})
	{
		const auto initial = f.captures + 1;
		f.on_capture_end = [&](unsigned call) { if (call == initial) { require(!f.capture_depth, "GPU preparation retained CPU capture"); f.aba(1); } };
		conflict([&] { if (surface) (void)f.textures->resolve(source, *f.surfaces); else (void)f.textures->resolve(source); });
		f.on_capture_end = {};
	}
	// The opposite counter ordering must not mask a same-value VM store.
	for (unsigned i = 0; i < 100; ++i) f.aba(1);
	const auto high_io = f.textures->resolve_versioned(source, *f.surfaces);
	require(f.io->current() > f.memory->revision({source.offset, 64}), "Texture fixture did not exercise a larger IO stamp");
	f.memory->write({source.offset, 1}, std::array{std::byte{255}});
	require(!f.current(high_io.revision), "High IO counter masked a same-value VM write");
	// Prepared old native images survive changes independently of validity.
	const auto output = make_buffer(f.ctx, 64, buffer_storage::shared);
	f.ctx.begin(); texture_to_buffer(f.ctx, old.image.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, output, 0); f.ctx.submit().wait();
	for (unsigned i = 0; i < 64; ++i) require(map_buffer(output, 0, 64)[i] == std::byte(std::array{60u, 40u, 20u, 255u}[i % 4]), "IO refusal damaged retained native image");
	const auto native_source = f.surfaces->acquire({source.offset, {4, 4, 1, surface_color::argb8, false}, 16, {}});
	f.surfaces->clear(native_source, 0xf0, 0xff5078a0, 0, 255, {0, 0, 4, 4});
	const auto native_revision = f.memory->revision({source.offset, 64});
	const auto native = f.textures->resolve_versioned(source, *f.surfaces);
	require(f.current(native.revision) && f.textures->statistics().surface_snapshots > 0, "Native dirty source did not yield versioned snapshot");
	f.aba(1);
	require(!f.current(native.revision) && f.memory->revision({source.offset, 64}) == native_revision, "Dirty native source lost IO ABA or wrote guest bytes");
	f.ctx.begin(); texture_to_buffer(f.ctx, native.image.texture, {0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(4, 4, 1)}, output, 0); f.ctx.submit().wait();
	const auto native_pixels = map_buffer(output, 0, 64);
	for (unsigned i = 0; i < 64; ++i) require(native_pixels[i] == std::byte(std::array{160u, 120u, 80u, 255u}[i % 4]), "Native IO conflict damaged dirty-source GPU snapshot");
	std::cout << "Native texture IO:identical ABA with unchanged VM revision/notifications,local/unrelated entry independence,full cube/mip last entry,initial/final query changes,two native resolve entry points,separate VM-IO counter orderings,dirty native source and retained GPU pixels passed\n";
}
std::vector<std::byte> pattern(const guest_texture_layout& plan)
{
	std::vector<std::byte> bytes(plan.storage_bytes, std::byte{0xdd});
	for (const auto& part : plan.subresources)
		for (std::size_t z = 0; z < part.depth; ++z)
			for (std::size_t y = 0; y < part.rows; ++y)
				for (std::size_t x = 0; x < part.columns; ++x)
				{
					auto offset = part.offset + (z * part.rows + y) * part.row_pitch + x * 4;
					if (plan.swizzled)
						offset = part.offset + morton_index({x, y, z}, {unsigned(std::bit_width(part.columns) - 1), unsigned(std::bit_width(part.rows) - 1), unsigned(std::bit_width(part.depth) - 1)}) * 4;
					const std::array<std::size_t, 4> pixel{255u, 32u + part.face * 8 + part.level, 40u + y, 60u + x + z * 16};
					for (unsigned c = 0; c < 4; ++c) bytes[offset + c] = std::byte(pixel[c]);
				}
	return bytes;
}
void scatter(fixture& f, const guest_texture_layout& plan, std::span<const std::byte> bytes)
{
	checked_cpu_capture(f.capture, [&]
	{
		for (const auto& part : texture_storage_segments(plan.offset, plan.location, plan.storage_bytes,
			[&](auto offset, auto location, auto size) { return f.translate(offset, location, size); }))
			f.memory->write({part.address, part.size}, bytes.subspan(part.offset, part.size));
	});
}
void pixels(fixture& f, id<MTLTexture> image, const guest_texture_layout& plan, bool dirty_tail = false, unsigned green_add = 0)
{
	require(!f.capture_depth, "Native texture GPU readback retained producer exclusion");
	for (const auto& part : plan.subresources)
	{
		const auto size = part.columns * part.rows * part.depth * 4;
		const auto output = make_buffer(f.ctx, size, buffer_storage::shared);
		f.ctx.begin(); texture_to_buffer(f.ctx, image, {part.level, part.face, MTLOriginMake(0, 0, 0), MTLSizeMake(part.columns, part.rows, part.depth)}, output, 0); f.ctx.submit().wait();
		const auto read = map_buffer(output, 0, size);
		for (std::size_t z = 0; z < part.depth; ++z)
			for (std::size_t y = 0; y < part.rows; ++y)
				for (std::size_t x = 0; x < part.columns; ++x)
				{
					const std::array expected = dirty_tail && y ? std::array<std::size_t, 4>{0xef, 0xcd, 0xab, 255} :
						std::array<std::size_t, 4>{60 + x + z * 16, 40 + y + (y ? green_add : 0), 32 + part.face * 8 + part.level, 255};
					const auto index = ((z * part.rows + y) * part.columns + x) * 4;
					for (unsigned c = 0; c < 4; ++c) require(read[index + c] == std::byte(expected[c]), "Disjoint native texture pixel/mip/face/order mismatch");
				}
	}
}
void segment_cases(fixture& f)
{
	f.textures->clear();
	checked_cpu_capture(f.capture, [&]
	{
		f.table[0] = 31 << 20; f.table[1] = 12 << 20;
		f.table[2] = 31 << 20; f.table[3] = 13 << 20;
		f.io->notify({0, 4 << 20});
	});
	const auto source = words(0xffff0); const auto plan = plan_guest_texture(source);
	require(plan.storage_bytes == 64, "Disjoint ARGB fixture storage changed");
	const auto data = pattern(plan); scatter(f, plan, data);
	const auto initial = f.textures->resolve_versioned(source, *f.surfaces); pixels(f, initial.image.texture, plan);
	require(f.current(initial.revision), "Fresh disjoint native texture is stale");
	const auto stats = f.textures->statistics();
	const auto hit = f.textures->resolve_versioned(source, *f.surfaces);
	require(hit.image.texture == initial.image.texture && f.textures->statistics().hits == stats.hits + 1, "Disjoint texture cache did not reuse immutable source");
	f.memory->write({5 << 20, 1}, std::array{std::byte{1}});
	require(f.current(hit.revision), "Fictitious EA envelope invalidated a disjoint source");
	f.aba(1); require(!f.current(hit.revision), "Last disjoint IO entry escaped native ticket");
	const auto after_io = f.textures->resolve_versioned(source, *f.surfaces);
	f.memory->write({12 << 20, 1}, std::array{data[16]});
	require(!f.current(after_io.revision), "Same-value tail-segment write escaped native ticket");
	const auto after_vm = f.textures->resolve_versioned(source, *f.surfaces); pixels(f, after_vm.image.texture, plan);
	// All source pieces and their tags must be captured in one provider epoch.
	guest_texture_resolver resolver(f.ctx, f.memory, [&](auto offset, auto location, auto size) { return f.translate(offset, location, size); }, 4 << 20, f.budget, f.capture);
	unsigned epoch = 0, reads = 0;
	f.access->on_map = [&](guest_surface_range range, bool write)
	{
		if (!write && (range.address == 0x1fffff0 || range.address == (12 << 20)))
		{
			require(f.capture_depth == 1, "Disjoint gather escaped common CPU capture");
			if (!epoch) epoch = f.captures;
			require(epoch == f.captures, "Disjoint gather split its source bytes/tags across CPU epochs"); ++reads;
		}
	};
	const auto captured = resolver.capture(source); f.access->on_map = {};
	require(reads >= 4 && captured.storage_bytes() == 64, "Disjoint owned capture omitted a segment/revision");
	scatter(f, plan, data); // Owned snapshot is independent of subsequent VM writes.
	f.on_translate = [](auto) { throw std::runtime_error("Owned disjoint realization translated VM"); };
	f.access->on_map = [](auto, auto) { throw std::runtime_error("Owned disjoint realization accessed VM"); };
	pixels(f, resolver.realize(captured).texture, plan);
	f.on_translate = {}; f.access->on_map = {};
	bool changed = false;
	f.access->on_map = [&](guest_surface_range range, bool write)
	{
		if (!write && range.address == (12 << 20) && !changed)
		{
			changed = true;
			f.memory->write({0x1fffff0, 1}, std::array{std::byte{255}});
		}
	};
	conflict([&] { (void)resolver.capture(source); }); f.access->on_map = {};
	require(changed && !f.capture_depth, "Disjoint gather failed to compare its earlier segment revision");
	// Same first EA and layout with another tail must not collide in the LRU.
	const auto other = words(0x2ffff0); const auto other_plan = plan_guest_texture(other);
	auto other_data = pattern(other_plan);
	for (unsigned i = 16; i < other_data.size(); i += 4) other_data[i + 2] = std::byte(std::to_integer<unsigned>(other_data[i + 2]) + 10);
	scatter(f, other_plan, other_data); f.textures->clear();
	const auto a = f.textures->resolve_versioned(source, *f.surfaces);
	const auto b = f.textures->resolve_versioned(other, *f.surfaces);
	require(a.image.texture != b.image.texture && f.textures->statistics().entries == 2, "Disjoint source tails collided in cache key");
	pixels(f, a.image.texture, plan); pixels(f, b.image.texture, other_plan, false, 10);
	f.textures->invalidate({5 << 20, 1}); require(f.textures->statistics().entries == 2, "Unused physical gap evicted disjoint cache source");
	f.textures->invalidate({13 << 20, 1}); require(f.textures->statistics().entries == 1, "Tail-segment explicit invalidation missed its source");
	// Physical reuse must validate the new logical alias;the old logical IO
	// allocation may have changed while its immutable EA/byte image stays valid.
	f.textures->clear(); const auto shared = f.textures->resolve_versioned(source, *f.surfaces);
	checked_cpu_capture(f.capture, [&] { f.table[1] = 13 << 20; f.table[3] = 12 << 20; f.io->notify({1 << 20, 3 << 20}); });
	const auto shared_hits = f.textures->statistics().hits;
	const auto alias = f.textures->resolve_versioned(other, *f.surfaces);
	require(alias.image.texture == shared.image.texture && f.textures->statistics().hits == shared_hits + 1 && f.current(alias.revision) && !f.current(shared.revision),
		"Disjoint physical reuse validated an obsolete logical IO alias");
	pixels(f, alias.image.texture, other_plan);
	checked_cpu_capture(f.capture, [&] { f.table[1] = 12 << 20; f.table[3] = 13 << 20; f.io->notify({1 << 20, 3 << 20}); });
	// Native tail stays GPU-authoritative while the disjoint VM first row
	// andsource snapshot are composed into an owned immutable image.
	const auto native = f.surfaces->acquire({12 << 20, {4, 3, 1, surface_color::argb8, false}, 16, {}});
	f.surfaces->clear(native, 0xf0, 0xffabcdef, 0, 255, {0, 0, 4, 3});
	const auto notices = f.access->notifications;
	const auto dirty = f.textures->resolve_versioned(source, *f.surfaces);
	require(f.current(dirty.revision) && f.access->notifications == notices && f.surfaces->status(native).dirty, "Disjoint native tail lost dirty authority orwrote back during composition");
	pixels(f, dirty.image.texture, plan, true); f.surfaces->erase(native);
	// A same-value VM writer after immutable capture must refuse publication,
	// even though native tail pixels were copied without guest writeback.
	const auto partial = f.surfaces->acquire({12 << 20, {4, 3, 1, surface_color::argb8, false}, 16, {}});
	f.surfaces->clear(partial, 0xf0, 0xffabcdef, 0, 255, {0, 0, 4, 3}); f.textures->clear();
	const auto gate=[f.ctx.device() newSharedEvent];struct release{id<MTLSharedEvent> event;~release(){event.signaledValue=1;}} unblock{gate};
	auto pending=f.textures->prepare_versioned_batch(std::array{source},*f.surfaces,std::array{guest_texture_upload_dependency{gate,1}});
	require(!pending.ready()&&f.textures->statistics().entries==0,"Disjoint partial alias didnotretain private pending work");
	f.memory->write({0x1fffff0, 1}, std::array{std::byte{255}});gate.signaledValue=1;
	conflict([&]{(void)pending.finish();});require(f.textures->statistics().entries==0,"Conflicted disjoint native composition published cache entries");
	pixels(f,dirty.image.texture,plan,true);f.surfaces->erase(partial);
	std::cout << "Disjoint native composition:dirty tail preserved without implicit writeback;post-capture same-value writer refuses gated publication/all entries,old native pixels retained passed\n";
	// Linear/Morton,all faces/mips and 3D slices keep their logical byte order.
	unsigned subresources = 0;
	for (bool linear : {true, false})
		for (auto dimension : {vertex_texture_dimension::one, vertex_texture_dimension::two, vertex_texture_dimension::three, vertex_texture_dimension::cube})
		{
			auto state = source;
			state.format = (3u << 16) | ((0x85u | (linear ? 0x20u : 0)) << 8) | ((dimension == vertex_texture_dimension::one ? 1u : dimension == vertex_texture_dimension::three ? 3u : 2u) << 4) | 8 | 2 | (dimension == vertex_texture_dimension::cube ? 4 : 0);
			state.image_rect = (4u << 16) | (dimension == vertex_texture_dimension::one ? 1u : 4u);
			state.control3 = ((dimension == vertex_texture_dimension::three ? 4u : 1u) << 20) | (linear ? 24u : 0u);
			const auto layout = plan_guest_texture(state); const auto bytes = pattern(layout); scatter(f, layout, bytes);
			const auto resource = f.textures->resolve_versioned(state, *f.surfaces);
			require(f.current(resource.revision), "Disjoint dimension/mip ticket is stale"); pixels(f, resource.image.texture, layout);
			subresources += layout.subresources.size();
		}
	checked_cpu_capture(f.capture, [&]
	{
		for (unsigned i = 0; i < 4; ++i) f.table[i] = i << 20;
		f.io->notify({0, 4 << 20});
	});
	f.textures->clear();
	std::cout << "Native disjoint texture gather:top provider EA plus independent tail,one CPU epoch for all reads/tags,owned realization,exact cache keys/invalidation,IO ABA/same-value VM writes,dirty native tail composition without implicit writeback;linear/Morton 1D/2D/3D/cube " << subresources << " mip/face GPU pixel readbacks passed\n";
}
void pending_case(fixture& f)
{
	const auto source = words(0x300100); f.fill(plan_guest_texture(source)); f.textures->clear();
	id<MTLSharedEvent> gate = [f.ctx.device() newSharedEvent];
	f.ctx.begin(); f.ctx.wait_event(gate, 1); const auto pending = f.ctx.submit();
	require(!pending.ready(), "Texture upload gate was not pending");
	std::promise<void> proceed; auto ready = proceed.get_future(); std::exception_ptr failure;
	std::jthread producer([&]
	{
		try { require(ready.wait_for(std::chrono::seconds(10)) == std::future_status::ready, "Texture producer was never released"); f.aba(3); }
		catch (...) { failure = std::current_exception(); }
		gate.signaledValue = 1;
	});
	const auto initial = f.captures + 1;
	f.on_capture_end = [&](unsigned call) { if (call == initial) { require(!f.capture_depth, "Texture preparation waited under CPU capture"); proceed.set_value(); } };
	conflict([&] { (void)f.textures->resolve_versioned(source, *f.surfaces); });
	producer.join(); f.on_capture_end = {}; if (failure) std::rethrow_exception(failure);
	pending.wait(); require(!f.capture_depth, "Pending texture IO leaked CPU exclusion");
	const auto fresh = f.textures->resolve_versioned(source, *f.surfaces); require(f.current(fresh.revision), "Fresh texture retry after pending native work failed");
	std::cout << "Native texture preparation:pending Metal4 queue work retains GPU ownership while IO producer acquires provider and releases event;IO ABA refuses prepared upload,fresh retry passes;CPU capture released before GPU wait\n";
}
void draw_cases(fixture& f)
{
	auto state = test::executor_fixture::registers(); test::executor_fixture::texture_registers(state, 1 << 20); test::executor_fixture::set_vertex_texture_registers(state, 17 << 20);
	for (unsigned unit = 0; unit < 16; ++unit)
	{
		for (unsigned word = 0; word < 8; ++word) state.registers[0x1a00 / 4 + unit * 8 + word] = state.registers[0x1a00 / 4 + word];
		state.registers[0x1840 / 4 + unit] = state.registers[0x1840 / 4];
		state.registers[0x1a00 / 4 + unit * 8] = (unit + 1) << 20;
	}
	for (unsigned unit = 0; unit < 4; ++unit)
	{
		for (unsigned word = 0; word < 8; ++word) state.registers[0x900 / 4 + unit * 8 + word] = state.registers[0x900 / 4 + word];
		state.registers[0x900 / 4 + unit * 8] = 0x80000000u | ((unit + 17) << 20);
		state.registers[0x900 / 4 + unit * 8 + 1] = (state.registers[0x900 / 4 + unit * 8 + 1] & ~3u) | 2;
	}
	const auto vertex = vertex_texture_snapshot::capture(state.registers, 15);
	const auto fragment = fragment_texture_snapshot::capture(state.registers, 65535);
	for (unsigned unit = 0; unit < 4; ++unit) f.fill(plan_guest_texture(vertex.at(unit).upload_words()));
	for (unsigned unit = 0; unit < 16; ++unit) f.fill(plan_guest_texture(fragment.at(unit)));
	unsigned refused = 0;
	for (unsigned source = 0; source < 20; ++source)
	{
		const auto prepared = f.executor->prepare_versioned_textures(vertex, fragment);
		require(prepared.revisions.size() == 20, "Native draw omitted referenced texture tickets");
		const auto notices = f.access->notifications;
		f.aba(source + 1);
		checked_cpu_capture(f.capture, [&] { if (!prepared.current(vertex, fragment)) ++refused; });
		require(f.access->notifications == notices, "Draw IO transition wrote guest bytes");
	}
	require(refused == 20, "A vertex/fragment IO ABA escaped final draw validation");
	const auto prepared = f.executor->prepare_versioned_textures(vertex, fragment);
	f.aba(29);
	checked_cpu_capture(f.capture, [&] { require(prepared.current(vertex, fragment), "Unreferenced IO entry invalidated complete native draw"); });
	for (unsigned unit = 0; unit < 4; ++unit) state.registers[0x900 / 4 + unit * 8 + 3] &= ~(1u << 31);
	for (unsigned unit = 0; unit < 16; ++unit) state.registers[0x1a00 / 4 + unit * 8 + 3] &= ~(1u << 31);
	const auto disabled_vertex = vertex_texture_snapshot::capture(state.registers, 15);
	const auto disabled_fragment = fragment_texture_snapshot::capture(state.registers, 65535);
	const auto captures = f.captures, translations = f.translations;
	f.io->notify({1, 1}); // Poison must be irrelevant to disabled/null resources.
	const auto disabled = f.executor->prepare_versioned_textures(disabled_vertex, disabled_fragment);
	require(disabled.revisions.empty() && disabled.current(disabled_vertex, disabled_fragment) && f.captures == captures && f.translations == translations,
		"Disabled textures consulted VM/IO or required CPU capture");
	const auto local = f.textures->resolve_versioned(words(0x200100, false), *f.surfaces);
	require(f.current(local.revision), "Poisoned unrelated IO prevented local native texture preparation");
	std::cout << "Native draw IO:all4 vertex/all16 fragment source ABA transitions independently refused under final CPU callback;unchanged guest bytes and unrelated IO accepted;disabled/null textures bypass poisoned IO and CPU capture,local native textures survive IO poison;native image preparation completes outside capture\n";
}
} // namespace
int main()
{
	@autoreleasepool
	{
		try { fixture f; ticket_cases(f); segment_cases(f); pending_case(f); draw_cases(f); std::cout << "Native texture/draw IO epochs passed;no App/UI/guest VM initialized\n"; }
		catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
	}
}
