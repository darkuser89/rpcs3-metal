#include "M4Display.h"
#include "M4CachedTextureRegion.h"
#include "M4ExclusiveGuestMemory.h"

#include <list>
#include <thread>
#include <limits>

namespace metal4
{
struct display_cache_state
{
	using image_state = display_upload_job::state;
	std::thread::id owner = std::this_thread::get_id();
	display_cache_limits limits;
	display_cache_statistics stats;
	std::list<std::shared_ptr<image_state>> ready;
	std::vector<std::weak_ptr<image_state>> pending;
	std::uint64_t epoch = 1;
	bool capturing = false;
	void outside() const
	{
		if (std::this_thread::get_id() != owner || capturing || guest_memory_epoch::active())
			throw std::logic_error("Display cache ownership changes require its render thread outside VM exclusion");
	}
	void evict();
	void publish(const std::shared_ptr<image_state>& image);
};
struct display_source_version
{
	std::shared_ptr<guest_surface_memory> memory;
	std::function<std::uint32_t(std::uint32_t,unsigned,std::size_t)> translate;
	cpu_capture_dispatch dispatch;
	std::weak_ptr<display_cache_state> cache;
	display_snapshot image;
	guest_surface_layout coverage;
	std::uint64_t revision = 0;
	std::shared_ptr<const surface_mapping_revision> mapping;
	std::shared_ptr<const surface_texture_revision> native_revision;
	std::vector<std::shared_ptr<const surface_texture_revision>> assembly_revisions;
	void capture(const std::function<void()>& operation) const
	{
		const auto state = cache.lock();
		if (state && state->capturing) throw std::logic_error("Display CPU capture cannot reenter its cache");
		if (state) state->capturing = true;
		try { if (dispatch) checked_cpu_capture(dispatch,operation); else operation(); }
		catch (...) { if (state) state->capturing = false; throw; }
		if (state) state->capturing = false;
	}
	bool mapped() const
	{
		return (!mapping || mapping->current()) && translate(image.offset,0,coverage.range.size) == coverage.range.address;
	}
	bool current() const
	{
		return mapped() && (!native_revision || native_revision->current()) &&
			std::all_of(assembly_revisions.begin(),assembly_revisions.end(),[](const auto& revision){return revision->current();}) &&
			memory->surface_revision(coverage) == revision && mapped();
	}
	bool same_key(const display_source_version& other) const
	{
		return image.offset == other.image.offset && image.width == other.image.width && image.height == other.image.height &&
			image.pitch == other.image.pitch && display_pixel_bytes(image.format) == display_pixel_bytes(other.image.format) &&
			coverage.range.address == other.coverage.range.address && coverage.range.size == other.coverage.range.size &&
			bool(mapping) == bool(other.mapping) && (!mapping || mapping->same_epoch(*other.mapping));
	}
};
struct display_upload_job::state
{
	std::shared_ptr<display_source_version> version;
	std::array<id<MTLTexture>,3> views{};
	id<MTLTexture> base;
	std::optional<submission> work;
	std::shared_ptr<prepared_surface_snapshot> native;
	std::thread::id owner = std::this_thread::get_id();
	std::uint64_t cache_epoch = 0;
	std::size_t bytes = 0;
	bool vm = false, converted = false, published = false, finishing = false;
	std::optional<std::size_t> morton_offset;
	bool mixed_vm = false;
	std::vector<std::shared_ptr<prepared_surface_snapshot>> assembly_sources;
	display_storage_assembly assembly;
	std::uint64_t assembly_epoch=0;
};
struct display_resolver::upload_batch
{
	struct upload
	{
		std::shared_ptr<display_upload_job::state> image;
		std::function<void()> encode;
	};
	context& ctx;
	display_cache_statistics& stats;
	std::span<const display_upload_dependency> dependencies;
	std::array<upload, 2> staged;
	std::size_t count = 0;

	void add(const std::shared_ptr<display_upload_job::state>& image, std::function<void()> encode = {})
	{
		staged.at(count) = {image, std::move(encode)};
		++count;
	}

	void submit()
	{
		if (!count) return;
		ctx.begin();
		try
		{
			for (const auto& dep : dependencies) ctx.wait_event(dep.event, dep.value);
			const auto images = std::span{staged}.first(count);
			for (const auto& item : images)
				if (item.encode) item.encode();
			const auto work = ctx.submit();
			bool native = false;
			for (const auto& item : images)
			{
				item.image->work = work;
				if (!item.encode) continue; // An alias of an image staged in this batch.
				const auto& image = *item.image;
				if (image.vm) ++stats.uploads;
				else
				{
					native = true;
					if (!image.assembly_sources.empty()) ++stats.assembly_casts;
					else if (image.converted)
					{
						++stats.native_casts;
						if (image.mixed_vm) ++stats.mixed_casts;
					}
				}
			}
			stats.native_submissions += native;
		}
		catch (...)
		{
			try { ctx.cancel(); } catch (...) {}
			throw;
		}
	}
};
struct display_frame_job::state
{
	std::array<std::optional<display_upload_job>, 2> eyes;
	cpu_capture_dispatch capture;
	std::thread::id owner = std::this_thread::get_id();
	bool finishing = false;
	void check(bool outside = false) const
	{
		if (std::this_thread::get_id() != owner || (outside && guest_memory_epoch::active()))
			throw std::logic_error("Display frame job requires its owner outside VM exclusion");
	}
};
display_frame_job::display_frame_job(std::shared_ptr<state> state) : m_state(std::move(state)) {}
display_frame_job::~display_frame_job() = default;
display_frame_job::display_frame_job(display_frame_job&&) noexcept = default;
display_frame_job& display_frame_job::operator=(display_frame_job&&) noexcept = default;
bool display_frame_job::ready() const
{
	if (!m_state) throw std::logic_error("Moved-from display frame job");
	m_state->check();
	return m_state->eyes[0]->ready() && (!m_state->eyes[1] || m_state->eyes[1]->ready());
}
bool display_frame_job::cancel_pending() const
{
	if (!m_state) throw std::logic_error("Moved-from display frame job");
	m_state->check(true);
	bool cancelled = false;
	for (const auto& eye : m_state->eyes) if (eye) cancelled |= eye->cancel_pending();
	return cancelled;
}
bool display_frame_job::current() const
{
	if (!m_state) throw std::logic_error("Moved-from display frame job");
	m_state->check();
	return m_state->eyes[0]->current() && (!m_state->eyes[1] || m_state->eyes[1]->current());
}
present_image display_frame_job::image_for_encoding() const
{
	if (!current()) throw surface_mapping_conflict();
	const auto& left = *m_state->eyes[0];
	auto result = display_upload_job::owned_image(*left.m_state, left.m_format);
	result.uploads[0] = left.m_state->work;
	if (const auto& right = m_state->eyes[1])
	{
		const auto image = display_upload_job::owned_image(*right->m_state, right->m_format);
		result.right_eye = present_eye{image.texture, image.region};
		result.uploads[1] = right->m_state->work;
	}
	return result;
}
present_image display_frame_job::finish(std::chrono::milliseconds timeout) const
{
	const auto retained = m_state;
	if (!retained) throw std::logic_error("Moved-from display frame job");
	retained->check(true);
	if (timeout <= std::chrono::milliseconds::zero()) throw std::invalid_argument("Display frame timeout must be positive");
	if (retained->finishing) throw std::logic_error("Display frame completion cannot reenter itself");
	retained->finishing = true;
	struct finish_scope { bool& flag; ~finish_scope() { flag = false; } } guard{retained->finishing};
	const auto start = std::chrono::steady_clock::now();
	std::array<std::optional<present_image>, 2> images;
	for (unsigned eye = 0; eye < 2; ++eye)
		if (retained->eyes[eye])
		{
			const auto remaining = timeout - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
			if (remaining <= std::chrono::milliseconds::zero() && !retained->eyes[eye]->ready())
				throw std::runtime_error("Display frame GPU deadline expired");
			images[eye] = retained->eyes[eye]->finish(std::max(remaining, std::chrono::milliseconds(1)));
		}
	const auto validate = [&]
	{
		for (const auto& eye : retained->eyes) if (eye && !eye->current()) throw surface_mapping_conflict();
	};
	if (retained->eyes[1])
	{
		if (retained->capture) checked_cpu_capture(retained->capture, validate); else validate();
	}
	auto result = std::move(*images[0]);
	if (images[1]) result.right_eye = present_eye{images[1]->texture, images[1]->region};
	return result;
}
void display_cache_state::evict()
{
	stats.bytes -= ready.back()->bytes;
	ready.pop_back(); ++stats.evictions; stats.entries = ready.size();
}
void display_cache_state::publish(const std::shared_ptr<image_state>& image)
{
	outside();
	if (image->cache_epoch != epoch || !limits.entries || image->bytes > limits.bytes) return;
	// Allocate the replacement list node before changing the confirmed cache.
	std::list<std::shared_ptr<image_state>> candidate{image};
	for (auto it = ready.begin(); it != ready.end();)
		if ((*it)->version->same_key(*image->version))
		{
			stats.bytes -= (*it)->bytes; it = ready.erase(it); ++stats.evictions;
		}
		else ++it;
	while (ready.size() >= limits.entries || stats.bytes > limits.bytes - image->bytes) evict();
	ready.splice(ready.begin(),candidate); stats.bytes += image->bytes;
	stats.entries = ready.size(); ++stats.published;
}
display_upload_job::display_upload_job(std::shared_ptr<state> state,display_format format) : m_state(std::move(state)),m_format(format) {}
display_upload_job::~display_upload_job() = default;
display_upload_job::display_upload_job(display_upload_job&&) noexcept = default;
display_upload_job& display_upload_job::operator=(display_upload_job&&) noexcept = default;
bool display_upload_job::ready() const
{
	if (!m_state) throw std::logic_error("Moved-from display upload job");
	return !m_state->work || m_state->work->ready();
}
bool display_upload_job::cancel_pending() const
{
	if (!m_state) throw std::logic_error("Moved-from display upload job");
	return m_state->work && m_state->work->cancel_pending();
}
bool display_upload_job::current() const
{
	if (!m_state || std::this_thread::get_id() != m_state->owner) throw std::logic_error("Display receipt requires its preparing thread");
	return m_state->version->current();
}
present_image display_upload_job::owned_image(const state& state, display_format format)
{
	const auto& image = state.version->image;
	return {state.views.at(static_cast<unsigned>(format)), {0,0,MTLOriginMake(0,0,0),MTLSizeMake(image.width,image.height,1)}, {0,0,image.width,image.height}};
}
present_image display_upload_job::finish(std::chrono::milliseconds timeout) const
{
	const auto state = m_state;
	const auto format = m_format;
	if (!state || std::this_thread::get_id() != state->owner || guest_memory_epoch::active())
		throw std::logic_error("Display completion requires its preparing thread outside VM exclusion");
	if (const auto cache = state->version->cache.lock()) cache->outside();
	if (state->native) state->native->validate_outside();
	for(const auto& source:state->assembly_sources)source->validate_outside();
	if (timeout <= std::chrono::milliseconds::zero()) throw std::invalid_argument("Display completion timeout must be positive");
	if (state->finishing) throw std::logic_error("Display completion cannot reenter itself");
	state->finishing = true;
	struct finish_scope { bool& active; ~finish_scope() { active = false; } } finish_guard{state->finishing};
	if (state->work) state->work->wait(timeout);
	if (state->native) state->native->completed();
	for(const auto& source:state->assembly_sources)source->completed();
	state->version->capture([&] { if (!state->version->current()) throw surface_conflict_error(state->version->coverage.range); });
	if (state->native) state->native->publish();
	for(const auto& source:state->assembly_sources)source->publish();
	if (state->converted && state->native && state->native->needs_copy()) state->native = state->native->settled();
	for(auto& source:state->assembly_sources)if(source->needs_copy())source=source->settled();
	if (!state->published)
	{
		const bool assembly_current=!state->assembly_sources.empty() && state->assembly_sources.front()->cache_native_epoch(state->assembly_epoch) &&
			std::all_of(state->assembly_sources.begin(),state->assembly_sources.end(),[](const auto& source){return source->cache_current();});
		const bool single_current=state->native && state->native->cache_current() &&
			(!state->mixed_vm || state->native->cache_native_epoch(state->assembly_epoch));
		if (state->vm || (state->converted && (state->native?single_current:assembly_current)))
			if (const auto cache = state->version->cache.lock()) cache->publish(state);
		state->published = true;
	}
	return owned_image(*state, format);
}
display_resolver::display_resolver(context& ctx,std::shared_ptr<guest_surface_memory> memory,
	std::function<std::uint32_t(std::uint32_t,unsigned,std::size_t)> translate,std::shared_ptr<heap_budget> budget,std::size_t snapshot_limit,
	cpu_capture_dispatch capture,std::shared_ptr<tile_state> tiles,display_cache_limits limits)
	: m_context(ctx),m_memory(std::move(memory)),m_translate(std::move(translate)),m_budget(std::move(budget)),m_limit(snapshot_limit),
	  m_capture(std::move(capture)),m_tiles(std::move(tiles)),m_cache(std::make_shared<display_cache_state>()),
	  m_upload(ctx,m_memory,m_translate,m_limit,m_budget,m_capture)
{
	if (!m_memory || !m_translate || !m_budget || m_budget->device() != ctx.device() || !m_limit || (m_tiles && !m_capture) || limits.entries > 64)
		throw std::invalid_argument("Native display resolver needs coherent VM,translation,budget andbounded cache services");
	m_cache->limits = limits; m_cache->pending.reserve(64);
}
std::shared_ptr<display_upload_job::state> display_resolver::prepare_impl(const display_snapshot& image,surface_cache& surfaces,
	upload_batch& batch,bool require_ledger)
{
	m_cache->outside(); m_context.validate_begin();
	if ((require_ledger || m_context.buffer_ledger()) && m_context.buffer_ledger() != m_budget)
		throw std::invalid_argument("Display upload job requires the installed shared buffer/texture ledger");
	if (batch.dependencies.size() > 64) throw std::length_error("Display dependency bound exceeded");
	for (const auto& dep : batch.dependencies)
		if (!dep.event || (dep.event.device && dep.event.device != m_context.device())) throw std::invalid_argument("Foreign/nil display dependency");
	surfaces.validate_texture_adapter(m_context,*m_memory);
	if (image.right_offset) throw std::invalid_argument("Single-eye display preparation requires explicit eye selection");
	auto coverage = display_coverage(image,0);
	if (coverage.range.size > m_limit || image.width > max_2d_texture_dimension(m_context.device()) || image.height > max_2d_texture_dimension(m_context.device()))
		throw std::out_of_range("Native display snapshot exceeds storage/dimension limit");
	std::erase_if(m_cache->pending,[](const auto& weak)
	{
		const auto candidate = weak.lock();
		if (!candidate || candidate->published) return true;
		if (candidate->work && candidate->work->ready())
		{
			try { candidate->work->wait(std::chrono::milliseconds::zero()); }
			catch (const std::runtime_error&) { return true; }
		}
		return false;
	});
	auto version = std::make_shared<display_source_version>();
	version->memory = m_memory; version->translate = m_translate; version->dispatch = m_capture; version->cache = m_cache; version->image = image;
	version->capture([&]
	{
		if (m_tiles)
		{
			// Local VRAM tile access is transparent in the shared RSX memory
			// controller contract. Keep a strict tile receipt for this AV frame;
			// never apply main-memory bank detiling to local display bytes.
			version->mapping = capture_surface_mapping(coverage,image.offset,0,m_translate,{},m_tiles,false,m_limit);
		}
		coverage = display_coverage(image,m_translate(image.offset,0,coverage.range.size));
		version->coverage = m_memory->watch_surface(coverage);
		version->revision = m_memory->surface_revision(version->coverage);
		if (!version->mapped()) throw surface_mapping_conflict();
	});
	guest_texture_layout plan{};
	plan.format = image.format == display_format::rgbx16f ? 0x9a : 0x85;
	plan.offset = image.offset;
	plan.width = image.width; plan.height = image.height; plan.depth = plan.mips = plan.layers = 1;
	plan.dimension = vertex_texture_dimension::two; plan.storage_bytes = coverage.range.size; plan.codec = describe_guest_texture_format(plan.format);
	plan.subresources.push_back({0,plan.storage_bytes,image.pitch,image.width,image.height,1,image.width,image.height,image.height,0,0,0});
	const auto relative_coverage = display_coverage(image,0);
	const std::array requested{guest_storage_segment{coverage.range.address,0,coverage.range.size}};
	const auto morton_mapping = [&](const cached_surface_description& source)
	{
		return plan_cached_morton_display_storage(*m_memory,source,relative_coverage,requested,image.offset,true);
	};
	const auto native_mapping = [&](const cached_surface_description& source) -> std::optional<surface_texture_region>
	{
		if (!std::holds_alternative<surface_color>(source.allocation.format)) return {};
		if(source.tile)return {}; // AV addresses physical bytes,not main-memory logical tile coordinates.
		if (source.morton)
		{
			if (!morton_mapping(source)) return {};
			const auto size = source.allocation.samples == 1 ? std::array{source.allocation.width,source.allocation.height} :
				sample_expanded_size(source.allocation.width,source.allocation.height,source.allocation.samples);
			return surface_texture_region{{0,0,size[0],size[1]},0};
		}
		// Prove ordered active guest bytes, including sample-expanded coordinates
		// and partial source texels. The immutable snapshot expands MSAA on GPU;
		// incompatible AV storage is subsequently packed/cropped/decoded on GPU.
		return plan_cached_texture_mapping(*m_memory,source,plan,requested,{},true,true);
	};
	std::vector<std::weak_ptr<display_upload_job::state>> native_candidates;
	std::vector<std::shared_ptr<display_upload_job::state>> pending_owners;
	std::vector<std::shared_ptr<prepared_surface_snapshot>> pending_native;
	for (const auto& candidate : m_cache->ready)
		if (candidate->converted && candidate->native && candidate->version->image.width == image.width && candidate->version->image.height == image.height &&
			display_pixel_bytes(candidate->version->image.format) == display_pixel_bytes(image.format)) native_candidates.push_back(candidate);
	for (const auto& weak : m_cache->pending)
		if (const auto candidate = weak.lock(); candidate && candidate->native)
		{
			// Share pending work only when its complete output schema matches.
			// A different cast gets an independent snapshot/copy; it must not
			// borrow a still-pending source copy into another command recording.
			const auto& prior = candidate->version->image;
			if (prior.width != image.width || prior.height != image.height ||
				display_pixel_bytes(prior.format) != display_pixel_bytes(image.format)) continue;
			if (candidate->morton_offset)
			{
				bool same = false;
				version->capture([&]
				{
					same = prior.pitch == image.pitch && morton_mapping(candidate->native->source_description()) == candidate->morton_offset;
					// Composite uploads also own VM bytes. Revisions from different
					// virtual watches are not interchangeable, even for pixel aliases.
					if (candidate->mixed_vm) same = same && candidate->version->same_key(*version) &&
						candidate->version->revision == version->revision && candidate->version->current();
				});
				if (!same) continue; // Different byte window needs an independent pending source copy.
			}
			native_candidates.push_back(candidate);
			pending_owners.push_back(candidate);
			if (std::find(pending_native.begin(),pending_native.end(),candidate->native) == pending_native.end()) pending_native.push_back(candidate->native);
		}
	const auto native = surfaces.prepare_texture_mapping(std::span{&coverage,1},native_mapping,m_budget,false,pending_native,m_cache->pending.size() < 64,true);
	const auto pixel_format = image.format == display_format::rgbx16f ? MTLPixelFormatRGBA16Float : image.format == display_format::xbgr8 ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatBGRA8Unorm;
	if (native)
	{
		const auto source_format = std::get<surface_color>(native->format());
		const bool rgba8 = source_format == surface_color::argb8 || source_format == surface_color::abgr8 || source_format == surface_color::xrgb8_zero ||
			source_format == surface_color::xrgb8_one || source_format == surface_color::xbgr8_zero || source_format == surface_color::xbgr8_one;
		const bool direct = !native->source_description().morton && (image.format == display_format::rgbx16f ? source_format == surface_color::rgba16f : rgba8) &&
			!native->row_offset() && native->texture().width == image.width && native->texture().height == image.height;
		version->native_revision = native->revision();
		std::shared_ptr<display_upload_job::state> shared_work, selected;
		std::optional<std::size_t> morton_offset;
		std::vector<guest_surface_range> morton_holes;
		version->capture([&]
		{
			if (!version->mapped() || !version->native_revision || !version->native_revision->current()) throw surface_conflict_error(coverage.range);
			version->revision = m_memory->surface_revision(version->coverage);
			if (native->source_description().morton)
			{
				morton_offset = morton_mapping(native->source_description());
				if (!morton_offset) throw surface_mapping_conflict();
				const auto& source = native->source_description();
				const auto size = source.allocation.samples == 1 ? std::array{source.allocation.width,source.allocation.height} :
					sample_expanded_size(source.allocation.width,source.allocation.height,source.allocation.samples);
				const auto geometry = plan_guest_surface(0,size[0],size[1],color_layout(source_format).pixel_bytes,0,source.morton);
				auto holes = morton_display_holes(geometry,relative_coverage,*morton_offset);
				if (!holes) throw surface_mapping_conflict();
				morton_holes = std::move(*holes);
			}
			for (const auto& weak : native_candidates)
				if (const auto candidate = weak.lock(); candidate && candidate->converted == !direct &&
					(candidate->native->texture() == native->texture() || (morton_offset && candidate->native->same_content(*native))) &&
					candidate->native->format() == native->format() && candidate->native->row_offset() == native->row_offset() &&
					candidate->morton_offset == morton_offset && (!morton_offset || candidate->version->image.pitch == image.pitch) &&
					candidate->mixed_vm == !morton_holes.empty() && (morton_holes.empty() ||
					(candidate->version->same_key(*version) && candidate->version->revision == version->revision && candidate->version->current())))
				{
					shared_work = candidate;
					if (candidate->version->same_key(*version) && candidate->version->revision == version->revision && candidate->version->current())
						{ selected = candidate; break; }
				}
		});
		// Weak ready lookup owners permit ledger recovery to actually retire an
		// evicted output while admitting the next snapshot/cast allocation.
		native_candidates.clear(); pending_native.clear(); pending_owners.clear();
		if (selected)
		{
			if(selected->mixed_vm)selected->assembly_epoch=surfaces.texture_source_epoch();
			if (direct) selected->views[static_cast<unsigned>(image.format)] = native->opaque_view(pixel_format);
			if (selected->published)
			{
				const auto found = std::find(m_cache->ready.begin(),m_cache->ready.end(),selected);
				if (found != m_cache->ready.end()) m_cache->ready.splice(m_cache->ready.begin(),m_cache->ready,found);
				++m_cache->stats.hits;
			}
			else ++m_cache->stats.pending_hits;
			if (selected->converted) ++m_cache->stats.native_cast_hits;
			++m_cache->stats.native_snapshots; return selected;
		}
		if ((native->needs_copy() || !direct) && m_cache->pending.size() >= 64) throw std::length_error("Display native pending-owner bound exceeded");
		auto result = std::make_shared<display_upload_job::state>(); result->version = version; result->base = native->texture(); result->native = native;
		result->morton_offset = morton_offset;
		result->mixed_vm = !morton_holes.empty();
		result->assembly_epoch=surfaces.texture_source_epoch();
		result->converted = !direct; result->cache_epoch = m_cache->epoch;
		std::optional<guest_texture_resolver::native_cast_input> cast;
		if (shared_work)
		{
			result->base = shared_work->base; result->bytes = shared_work->bytes; result->views = shared_work->views;
			result->work = shared_work->work; ++m_cache->stats.pending_hits;
			if (!result->work && !shared_work->published) batch.add(result);
			if (result->converted) ++m_cache->stats.native_cast_hits;
		}
		else if (result->converted)
		{
			std::vector<guest_texture_resolver::display_byte_patch> patches;
			std::vector<std::byte> vm_bytes;
			if (!morton_holes.empty())
			{
				std::size_t total = 0; for (const auto hole : morton_holes) total = checked_texture_sum(total,hole.size);
				if (total > m_limit) throw std::length_error("Mixed display VM payload exceeds snapshot limit");
				vm_bytes.reserve(total); patches.reserve(morton_holes.size());
				version->capture([&]
				{
					if (!version->current()) throw surface_conflict_error(coverage.range);
					for (const auto hole : morton_holes)
					{
						const auto address = std::uint64_t(coverage.range.address) + hole.address - *morton_offset;
						const auto fragment = m_memory->read_surface(guest_byte_coverage({static_cast<std::uint32_t>(address),hole.size}));
						if (fragment.bytes.size() != hole.size) throw std::runtime_error("Incomplete mixed display VM fragment");
						patches.push_back({hole.address,vm_bytes.size(),hole.size});
						vm_bytes.insert(vm_bytes.end(),fragment.bytes.begin(),fragment.bytes.end());
					}
					if (!version->current()) throw surface_conflict_error(coverage.range);
				});
				m_cache->stats.captured_vm_bytes += vm_bytes.size(); m_cache->stats.captured_vm_ranges += patches.size();
			}
			cast = morton_offset ? m_upload.prepare_morton_display_cast(plan,native->format(),native->texture(),*native->source_description().morton,*morton_offset,std::move(patches),std::move(vm_bytes)) :
				m_upload.prepare_cast(plan,native->format(),native->texture(),{},native->row_offset());
			result->base = m_upload.allocate(plan);
			// Count both owned immutable images, conservatively even when the
			// source snapshot is also retained in the surface cache.
			result->bytes = checked_texture_sum(result->base.heap.size,native->texture().heap.size);
			if (image.format == display_format::rgbx16f) result->views[2] = surfaces.opaque_snapshot_view(result->base,MTLPixelFormatRGBA16Float);
			else
			{
				result->views[0] = surfaces.opaque_snapshot_view(result->base,MTLPixelFormatBGRA8Unorm);
				result->views[1] = surfaces.opaque_snapshot_view(result->base,MTLPixelFormatRGBA8Unorm);
			}
		}
		if (direct) result->views[static_cast<unsigned>(image.format)] = native->opaque_view(pixel_format);
		if (!shared_work && (native->needs_copy() || cast))
			batch.add(result, [this, native, cast = std::move(cast), base = result->base]
			{
				native->encode(m_context);
				if (cast) m_upload.encode_cast(*cast, base);
			});
		if (native->needs_copy() || cast || (shared_work && !shared_work->published)) m_cache->pending.push_back(result);
		++m_cache->stats.native_snapshots;
		return result;
	}
	native_candidates.clear(); pending_native.clear(); pending_owners.clear();
	if(const auto assembly=prepare_assembly(version,surfaces,plan,batch))return assembly;
	std::vector<std::shared_ptr<display_upload_job::state>> candidates(m_cache->ready.begin(),m_cache->ready.end());
	for (const auto& weak : m_cache->pending) if (const auto value = weak.lock()) candidates.push_back(value);
	std::shared_ptr<display_upload_job::state> selected;
	version->capture([&]
	{
		if (!version->mapped()) throw surface_mapping_conflict();
		version->revision = m_memory->surface_revision(version->coverage);
		for (const auto& candidate : candidates)
			if (candidate->vm && candidate->version->same_key(*version) && candidate->version->revision == version->revision && candidate->version->current()) { selected = candidate; break; }
	});
	// Lookup owners must not pin evictable old images during new admission.
	candidates.clear();
	if (selected)
	{
		const auto found = std::find(m_cache->ready.begin(),m_cache->ready.end(),selected);
		if (found != m_cache->ready.end()) { ++m_cache->stats.hits; m_cache->ready.splice(m_cache->ready.begin(),m_cache->ready,found); }
		else ++m_cache->stats.pending_hits;
		return selected;
	}
	if (m_cache->pending.size() >= 64) throw std::length_error("Display job pending-owner bound exceeded");
	guest_surface_snapshot snapshot;
	version->capture([&]
	{
		if (!version->mapped()) throw surface_mapping_conflict();
		snapshot = m_memory->read_surface(version->coverage);
		if (snapshot.bytes.size() != coverage.range.size) throw std::runtime_error("Incomplete display VM snapshot");
		version->revision = snapshot.revision;
		if (!version->current()) throw surface_conflict_error(coverage.range);
	});
	auto result = std::make_shared<display_upload_job::state>(); result->version = version; result->vm = true; result->cache_epoch = m_cache->epoch;
	const auto canonical = image.format == display_format::rgbx16f ? surface_color::rgba16f : surface_color::argb8;
	const auto base_format = describe_surface_color(canonical).format;
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:base_format width:image.width height:image.height mipmapped:NO];
	descriptor.storageMode = MTLStorageModePrivate; descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	result->base = m_budget->texture(descriptor); result->bytes = result->base.heap.size;
	if (image.format == display_format::rgbx16f) result->views[2] = surfaces.opaque_snapshot_view(result->base,MTLPixelFormatRGBA16Float);
	else
	{
		result->views[0] = surfaces.opaque_snapshot_view(result->base,MTLPixelFormatBGRA8Unorm);
		result->views[1] = surfaces.opaque_snapshot_view(result->base,MTLPixelFormatRGBA8Unorm);
	}
	batch.add(result, [this, plan = std::move(plan), bytes = std::move(snapshot.bytes), base = result->base]
	{
		m_upload.encode_upload(plan, make_upload(m_context, bytes), base);
	});
	m_cache->pending.push_back(result);
	return result;
}
std::shared_ptr<display_upload_job::state> display_resolver::prepare_assembly(const std::shared_ptr<display_source_version>& version,
	surface_cache& surfaces,const guest_texture_layout& plan,upload_batch& batch)
{
	const auto& image=version->image;const auto coverage=version->coverage;
	auto refs=surfaces.texture_sources(std::span{&coverage,1});
	const auto initial_refs=refs;
	std::vector<display_storage_source> layouts;
	bool deferred=false;
	for(auto it=refs.begin();it!=refs.end();)
	{
		const auto status=surfaces.status(*it);const auto description=surfaces.description(*it);
		const bool dirty=status.dirty||status.deferred_copies;
		if(!std::holds_alternative<surface_color>(description.allocation.format))
		{
			if(dirty)throw std::invalid_argument("Native display source storage is not implemented");
			it=refs.erase(it);continue;
		}
		const auto& allocation=description.allocation;
		const auto size=allocation.samples==1?std::array{allocation.width,allocation.height}:sample_expanded_size(allocation.width,allocation.height,allocation.samples);
		auto geometry=plan_guest_surface(0,size[0],size[1],color_layout(std::get<surface_color>(allocation.format)).pixel_bytes,description.row_pitch,description.morton);
		auto segments=description.segments;if(segments.empty())segments.push_back({description.tile?description.tile->address:description.address,0,description.tile?description.tile->size:geometry.range.size});
		layouts.push_back({std::move(geometry),std::move(segments),dirty,status.native_write_epoch,description.tile});deferred|=bool(status.deferred_copies);++it;
	}
	if(refs.empty())return {};
	display_storage_assembly assembly;
	std::vector<std::shared_ptr<prepared_surface_snapshot>> native;
	const auto build=[&]
	{
		version->capture([&]
		{
			if(!version->mapped())throw surface_mapping_conflict();
			version->assembly_revisions.clear();
			for(const auto& source:native)version->assembly_revisions.push_back(source->revision());
			for(const auto& revision:version->assembly_revisions)if(!revision->current())throw surface_conflict_error(coverage.range);
			version->revision=m_memory->surface_revision(coverage);
			assembly=plan_display_storage_assembly(*m_memory,coverage,layouts);
			if(!version->current())throw surface_conflict_error(coverage.range);
		});
	};
	const auto trim=[&]
	{
		std::vector<bool> used(refs.size());for(auto copy:assembly.copies)used.at(copy.input)=true;
		std::vector<std::size_t> indices(refs.size());std::vector<cached_surface_ref> kept;std::vector<display_storage_source> storage;
		std::vector<std::shared_ptr<prepared_surface_snapshot>> snapshots;
		for(std::size_t i=0;i<refs.size();++i)if(used[i])
		{
			indices[i]=kept.size();kept.push_back(refs[i]);storage.push_back(layouts[i]);if(!native.empty())snapshots.push_back(native[i]);
		}
		for(auto& copy:assembly.copies)copy.input=indices[copy.input];refs=std::move(kept);layouts=std::move(storage);native=std::move(snapshots);
	};
	if(!deferred){build();trim();if(refs.empty())return {};}
	const auto same_plan=[&](const display_storage_assembly& other)
	{
		return assembly.copies==other.copies && assembly.vm.size()==other.vm.size() &&
			std::equal(assembly.vm.begin(),assembly.vm.end(),other.vm.begin(),[](auto a,auto b){return a.address==b.address&&a.size==b.size;});
	};
	if(!deferred)
	{
		std::vector<std::weak_ptr<display_upload_job::state>> candidates;
		for(const auto& candidate:m_cache->ready)if(!candidate->assembly_sources.empty())candidates.push_back(candidate);
		for(const auto& weak:m_cache->pending)if(auto candidate=weak.lock();candidate&&!candidate->assembly_sources.empty())candidates.push_back(candidate);
		for(const auto& weak:candidates)if(auto candidate=weak.lock();candidate&&candidate->assembly_sources.size()==refs.size()&&same_plan(candidate->assembly))
		{
			bool current=true;for(std::size_t i=0;i<refs.size();++i)current&=candidate->assembly_sources[i]->matches_source(refs[i]);
			if(!current)continue;
			version->capture([&]{current=candidate->version->same_key(*version)&&candidate->version->revision==version->revision&&candidate->version->current()&&version->mapped();});
			if(!current)continue;
			candidate->assembly_epoch=surfaces.texture_source_epoch();
			if(candidate->published)
			{
				const auto found=std::find(m_cache->ready.begin(),m_cache->ready.end(),candidate);if(found!=m_cache->ready.end())m_cache->ready.splice(m_cache->ready.begin(),m_cache->ready,found);++m_cache->stats.hits;
			}
			else ++m_cache->stats.pending_hits;
			++m_cache->stats.assembly_hits;return candidate;
		}
	}
	if(m_cache->pending.size()>=64)throw std::length_error("Display assembly pending-owner bound exceeded");
	for(std::size_t i=0;i<refs.size();++i)
	{
		const auto full=[](const cached_surface_description& source)->std::optional<surface_texture_region>
		{
			const auto& a=source.allocation;const auto size=a.samples==1?std::array{a.width,a.height}:sample_expanded_size(a.width,a.height,a.samples);
			return surface_texture_region{{0,0,size[0],size[1]},0};
		};
		const auto snapshot=surfaces.prepare_texture_mapping(std::span{&coverage,1},full,m_budget,false,{},true,false,refs[i]);
		if(!snapshot)throw surface_conflict_error(coverage.range);native.push_back(snapshot);
		const auto status=surfaces.status(refs[i]);layouts[i].dirty=status.dirty||status.deferred_copies;layouts[i].epoch=status.native_write_epoch;
	}
	// A later source allocation may reclaim snapshot lookup entries while our
	// immutable copies remain owned. Require unchanged source authority here;
	// publication still requires the original cache epoch in finish().
	for(std::size_t i=0;i<native.size();++i)if(!native[i]->matches_source(refs[i],false))throw surface_conflict_error(coverage.range);
	build();trim();if(refs.empty())return {};
	const auto fresh=surfaces.texture_sources(std::span{&coverage,1});
	for(const auto& source:fresh)if(std::find(initial_refs.begin(),initial_refs.end(),source)==initial_refs.end())
	{
		const auto status=surfaces.status(source);if(status.dirty||status.deferred_copies)throw surface_conflict_error(coverage.range);
	}
	version->assembly_revisions.clear();for(const auto& source:native)version->assembly_revisions.push_back(source->revision());
	std::vector<std::byte> vm_bytes;std::size_t total=0;for(auto hole:assembly.vm)total=checked_texture_sum(total,hole.size);
	if(total>m_limit)throw std::length_error("Display assembly VM payload exceeds limit");vm_bytes.reserve(total);
	version->capture([&]
	{
		if(!version->current())throw surface_conflict_error(coverage.range);
		for(auto hole:assembly.vm)
		{
			const auto fragment=m_memory->read_surface(guest_byte_coverage({static_cast<std::uint32_t>(std::uint64_t(coverage.range.address)+hole.address),hole.size}));
			if(fragment.bytes.size()!=hole.size)throw std::runtime_error("Incomplete display assembly VM fragment");vm_bytes.insert(vm_bytes.end(),fragment.bytes.begin(),fragment.bytes.end());
		}
		if(!version->current())throw surface_conflict_error(coverage.range);
	});
	m_cache->stats.captured_vm_bytes+=vm_bytes.size();m_cache->stats.captured_vm_ranges+=assembly.vm.size();
	auto upload=m_upload.prepare_display_assembly(plan,native,layouts,assembly,std::move(vm_bytes));
	auto result=std::make_shared<display_upload_job::state>();result->version=version;result->converted=true;result->mixed_vm=!assembly.vm.empty();
	result->cache_epoch=m_cache->epoch;result->assembly_sources=std::move(native);result->assembly=std::move(assembly);result->assembly_epoch=surfaces.texture_source_epoch();
	result->base=m_upload.allocate(plan);result->bytes=result->base.heap.size;
	for(const auto& source:result->assembly_sources)result->bytes=checked_texture_sum(result->bytes,source->texture().heap.size);
	if(image.format==display_format::rgbx16f)result->views[2]=surfaces.opaque_snapshot_view(result->base,MTLPixelFormatRGBA16Float);
	else{result->views[0]=surfaces.opaque_snapshot_view(result->base,MTLPixelFormatBGRA8Unorm);result->views[1]=surfaces.opaque_snapshot_view(result->base,MTLPixelFormatRGBA8Unorm);}
	batch.add(result, [this, result, upload = std::move(upload)]
	{
		for (const auto& source : result->assembly_sources) source->encode(m_context);
		m_upload.encode_display_assembly(upload, result->base);
	});
	m_cache->pending.push_back(result);
	return result;
}
display_upload_job display_resolver::prepare(const display_snapshot& image,surface_cache& surfaces,std::span<const display_upload_dependency> dependencies)
{
	std::shared_ptr<display_upload_job::state> result; std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			upload_batch batch{m_context, m_cache->stats, dependencies};
			result = prepare_impl(image, surfaces, batch, true);
			batch.submit();
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return display_upload_job(std::move(result),image.format);
}
present_image display_resolver::resolve(const display_snapshot& image,surface_cache& surfaces)
{
	std::shared_ptr<display_upload_job::state> result; std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			upload_batch batch{m_context, m_cache->stats, {}};
			result = prepare_impl(image, surfaces, batch, false);
			batch.submit();
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return display_upload_job(std::move(result),image.format).finish();
}
display_frame_job display_resolver::prepare_frame(const display_snapshot& image, surface_cache& surfaces,
	std::span<const display_upload_dependency> dependencies)
{
	m_cache->outside();
	std::array<std::optional<display_snapshot>, 2> eyes;
	std::size_t bytes = 0;
	for (unsigned eye = 0; eye < (image.right_offset ? 2u : 1u); ++eye)
	{
		eyes[eye] = display_eye(image, eye);
		const auto count = display_coverage(*eyes[eye], 0).range.size;
		if (count > m_limit - bytes || eyes[eye]->width > max_2d_texture_dimension(m_context.device()) ||
			eyes[eye]->height > max_2d_texture_dimension(m_context.device()))
			throw std::out_of_range("Native display eyes exceed combined snapshot/dimension limit");
		bytes += count;
	}
	auto state = std::make_shared<display_frame_job::state>();
	state->capture = m_capture;
	std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			upload_batch batch{m_context, m_cache->stats, dependencies};
			for (unsigned eye = 0; eye < eyes.size(); ++eye)
				if (eyes[eye])
					state->eyes[eye] = display_upload_job(prepare_impl(*eyes[eye], surfaces, batch, true), eyes[eye]->format);
			batch.submit();
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return display_frame_job(std::move(state));
}
display_cache_statistics display_resolver::statistics() const { m_cache->outside(); return m_cache->stats; }
void display_resolver::clear()
{
	m_cache->outside();
	if (m_cache->epoch == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("Display cache epoch exhausted");
	++m_cache->epoch;
	while (!m_cache->ready.empty()) m_cache->evict(); m_cache->pending.clear();
}
bool display_resolver::reclaim_one()
{
	m_cache->outside(); if (m_cache->ready.empty()) return false; m_cache->evict(); return true;
}
} // namespace metal4
