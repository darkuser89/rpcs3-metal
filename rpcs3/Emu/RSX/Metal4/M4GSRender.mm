#include "M4GSRender.h"
#include "Emu/RSX/Host/MM.h"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/System.h"
#include "Emu/system_config.h"
#include "M4DMA.h"
#include "M4RSXDMA.h"
#include "M4FaultRange.h"
#include "M4FramebufferBinding.h"
#include "M4GuestLabel.h"
#include "M4GuestProducer.h"
#include "M4GuestReport.h"
#include "M4HostAccessPreparation.h"
#include "M4HostStoreTransaction.h"
#include "M4MappingInvalidation.h"
#include "M4MemoryPressure.h"
#include "M4RendererRuntimeRegistration.h"
#include "M4RSXCacheFault.h"
#include "M4RSXFrameService.h"
#include "M4RSXReports.h"
#include "M4RSXScaledImage.h"
#include "M4RSXStereo.h"
#include "M4RSXVMAccess.h"
#include "M4RSXMappingObserver.h"
#include "M4RSXOverlay.h"
#include "M4RSXHostWrite.h"
#include "M4ScaledImage.h"
#include "stdafx.h"
#include "M4FrameCaptureDelivery.h"
#include "util/video_provider.h"

#include <deque>

extern atomic_t<bool> g_user_asked_for_screenshot;
extern atomic_t<recording_mode> g_recording_mode;

LOG_CHANNEL(metal4_capture_log, "Metal4Capture");

namespace metal4
{
namespace
{
void report_capture_consumer_error(const char* operation, std::exception_ptr error)
{
	try { std::rethrow_exception(error); }
	catch (const std::exception& e) { metal4_capture_log.error("%s failed: %s", operation, e.what()); }
	catch (...) { metal4_capture_log.error("%s failed with an unknown exception", operation); }
}

tile_descriptor native_tile(const GcmTileInfo& value)
{
	return {value.location, value.offset, value.size, value.pitch, value.comp, value.base, value.bank, value.bound};
}
GcmTileInfo rsx_tile(const tile_descriptor& value)
{
	return {value.location, value.offset, value.size, value.pitch, value.comp, value.base, value.bank, value.bound};
}

bool clause_requires_draw_completion(rsx::draw_clause& clause)
{
	return owned_clause_requires_draw_completion(clause, clause.command == rsx::draw_command::inlined_array);
}
} // namespace
struct gs_renderer::runtime
{
	static std::shared_ptr<heap_budget> create_budget(context& device, std::size_t bytes)
	{
		auto result = std::make_shared<heap_budget>(device, bytes);
		device.set_buffer_budget(result);
		return result;
	}
	context device{2, 64, std::chrono::seconds(30), context::recording_overlap::explicit_predecessor};
	std::shared_ptr<heap_budget> budget;
	surface_cache surfaces;
	guest_texture_cache textures;
	mapping_invalidation_service mapping_invalidations;
	scaled_image_transfer images;
	dma_transfer dma;
	draw_executor executor;
	std::vector<versioned_draw_texture_job> draw_textures;
	bool surface_batch_submitted = false;
	vertex_constant_cache constants;
	display_resolver display;
	std::deque<display_frame_job> display_uploads;
	query_pool queries;
	std::optional<report_predicate> conditional_report;
	native_frame frame;
	rsx_frame_service frame_service;
	memory_pressure_owner memory_pressure;
	std::optional<frame_capture_job> capture;
	std::optional<rsx::overlays::resource_config> overlay_images;
	rsx_game_icon_source game_icon;
	bool capture_screenshot = false;
	std::optional<recording_mode> capture_recording;
	renderer_runtime_registration registration;
	std::optional<rsx_cache_fault> instruction_faults;
	std::shared_ptr<arm64_fault_channel> fault_channel;
	bool draining_faults = false;

	void create_instruction_faults(const renderer_arm64_fault_bindings& bindings,
		const std::shared_ptr<guarded_guest_memory>& memory)
	{
		instruction_faults.emplace(device, bindings.vm, memory, surfaces, textures);
		fault_channel = std::make_shared<arm64_fault_channel>(reinterpret_cast<std::uint64_t>(vm::g_base_addr), 0xe0000000ull);
	}

	[[nodiscard]] bool instruction_fault_pending() const noexcept
	{
		return fault_channel && !draining_faults && !guest_memory_epoch::active() && fault_channel->has_pending();
	}

	void drain_instruction_faults(const rsx_cache_fault::instruction_lease& retain)
	{
		if (!instruction_fault_pending()) return;
		struct reset_drain { bool& active; ~reset_drain() { active = false; } } reset{draining_faults};
		draining_faults = true;
		(void)fault_channel->drain([&](const auto& input) -> std::optional<arm64_memory_state>
		{
			if (const auto completed = instruction_faults->complete_parked(input, retain)) return completed->reply;
			return {};
		});
	}

	void close_instruction_faults()
	{
		if (!fault_channel) return;
		fault_channel->close();
		fault_channel.reset();
		instruction_faults.reset();
	}

	~runtime() { close_instruction_faults(); }

	void publish_ready_draw_textures()
	{
		std::erase_if(draw_textures, [](auto& job)
		{
			if (!job.ready()) return false;
			try { (void)job.finish(); }
			catch (const texture_revision_conflict&) {} // A later producer owns the live cache version.
			return true;
		});
	}

	void submit_surface_batch(surface_cache::draw_batch& batch)
	{
		batch.submit();
		surface_batch_submitted = true;
	}

	void poll_surface_batch_completion()
	{
		if (!surface_batch_submitted || !surfaces.poll_submitted()) return;
		publish_ready_draw_textures();
		surface_batch_submitted = false;
	}

	void collect_capture(GSFrameBase& consumer)
	{
		collect_frame_capture(capture, capture_screenshot, capture_recording,
			[] { return g_recording_mode.load(); }, consumer, [] { g_user_asked_for_screenshot = true; },
			report_capture_consumer_error);
	}

	void collect_display_uploads(bool drain = false)
	{
		while (!display_uploads.empty() &&
			(drain || display_uploads.size() >= 3 || display_uploads.front().ready()))
		{
			auto upload = std::move(display_uploads.front());
			display_uploads.pop_front();
			try { (void)upload.finish(); }
			catch (const surface_conflict_error&) {}
			catch (const surface_mapping_conflict&) {}
		}
	}

	void present(const std::optional<present_image>& image, const frame_policy& policy, GSFrameBase& consumer,
		rsx::overlays::display_manager* manager)
	{
		const auto read_mode = [] { return g_recording_mode.load(); };
		const auto recording = !capture ? read_mode() : recording_mode::stopped;
		const bool video = recording != recording_mode::stopped &&
			can_deliver_frame_video(recording, read_mode, consumer, report_capture_consumer_error);
		const bool screenshot = !capture && g_user_asked_for_screenshot.exchange(false);
		try
		{
			std::function<overlay_frame(frame_extent)> overlays;
			if (manager)
				overlays = [this, manager](frame_extent extent)
				{
					const auto icon = [this] { return game_icon.get(Emu.GetSfoDir(true), Emu.GetSfoDir(false)); };
					return capture_rsx_overlays(*manager, overlay_images, icon, extent, get_system_time());
				};
			(void)frame.present(image, policy, screenshot || video ? &capture : nullptr, overlays,
				g_cfg.video.record_with_overlays.get());
		}
		catch (...)
		{
			if (screenshot) g_user_asked_for_screenshot = true;
			throw;
		}
		if (screenshot || video)
		{
			if (capture)
			{
				capture_screenshot = screenshot;
				capture_recording = video && recording == read_mode() ? std::optional{recording} : std::nullopt;
			}
			else if (screenshot)
				g_user_asked_for_screenshot = true;
		}
	}
	explicit runtime(const renderer_bindings& bindings, const std::shared_ptr<io_mapping_epoch>& io, const std::shared_ptr<tile_state>& tiles)
		: budget(create_budget(device, bindings.heap_bytes)),
		  surfaces(device, bindings.memory, bindings.heap_bytes, budget, bindings.capture_epoch),
		  textures(device, bindings.memory, translate_rsx_vm_range, bindings.snapshot_bytes, budget, 256, {bindings.capture_epoch, io}),
		  mapping_invalidations(device, surfaces, textures),
		  images(device, surfaces, textures, bindings.memory, bindings.capture_epoch, bindings.snapshot_bytes),
		  dma(device, surfaces, textures, bindings.memory, bindings.capture_epoch, bindings.snapshot_bytes),
		  executor(device, surfaces, textures),
		  display(device, bindings.memory, translate_rsx_vm_range, budget, bindings.snapshot_bytes, bindings.capture_epoch, tiles), queries(device, 2048, 16384, budget), frame(device), frame_service(frame),
		  memory_pressure(device, surfaces, textures, budget, &display, &frame)
	{
		// Null uploads can wait for GPU completion. Initialize before installing
		// runtime dispatch or acquiring any method/VM epoch for a draw.
		(void)textures.null_images();
	}
};
utils::serial* gs_renderer::validate_bindings(utils::serial* archive, const renderer_bindings& b)
{
	if (!b.writer_and_mapping_hooks_installed || !b.memory || !b.vm_mapping_observer || !b.vm_mapping_observer->observes(b.memory) || !b.capture_epoch || !b.store_label || !b.store_report || !b.bind_runtime || !g_emu_callbacks.call_from_main_thread ||
		!b.memory_fault || !b.local_task || !b.shutdown || !b.heap_bytes || !b.snapshot_bytes)
		throw std::invalid_argument("Native GSRender requires installed VM/protection/mapping,frame,presentation and report services");
	if (b.arm64_faults && (!b.arm64_faults->vm || !b.memory->uses_provider(*b.arm64_faults->vm) ||
		!b.arm64_faults->retain_instruction || !b.arm64_faults->install_channel))
		throw std::invalid_argument("Native ARM64 faults require the renderer VM, an instruction lease and a platform bridge");
	return archive;
}
gs_renderer::gs_renderer(utils::serial* archive, renderer_bindings bindings)
	: rsx_program_frontend<GSRender>(validate_bindings(archive, bindings)), m_bindings(std::move(bindings)),
	  m_tasks([this]
			  { m_eng_interrupt_mask |= rsx::backend_interrupt; })
{
	static_assert(std::extent_v<decltype(tiles)> == 15);
	std::array<tile_descriptor, 15> initial{};
	if (archive)
		for (std::size_t i = 0; i < initial.size(); ++i) initial[i] = native_tile(tiles[i]);
	else
		for (auto& tile : tiles) tile = rsx_tile({});
	m_tiles = std::make_shared<tile_state>(initial);
	m_bindings.capture_epoch = tile_capture_dispatch(std::move(m_bindings.capture_epoch), m_tiles);
	backend_config = {};
	backend_config.supports_normalized_barycentrics = true;
	backend_config.supports_last_provoking_vertex = true;
	backend_config.supports_programmable_blending = true;
	backend_config.supports_host_gpu_labels = true;
	backend_config.supports_hw_conditional_render = true;
	// Match VK's global AA policy. Native framebuffers retain individual
	// samples;the guest performs its own resolve. Owned draws bind A2C/A2One.
	if (g_cfg.video.antialiasing_level != msaa_level::none)
	{
		backend_config.supports_hw_msaa = true;
		backend_config.supports_hw_a2c = true;
		backend_config.supports_hw_a2c_1spp = true;
		backend_config.supports_hw_a2one = true;
	}
}
gs_renderer::~gs_renderer()
{
	@autoreleasepool
	{
		set_instruction_channel({});
		m_tasks.close();
		// Controller references native query storage:release it before runtime,
		// including failed-init/failed-exit teardown before the GSRender base dies.
		zcull_ctrl.reset();
		m_runtime.reset();
	}
}
void gs_renderer::fail_renderer() noexcept
{
	m_failed = true;
	std::shared_ptr<arm64_fault_channel> channel;
	{
		std::lock_guard lock(m_instruction_channel_mutex);
		channel = m_instruction_channel;
	}
	if (channel) channel->request_close();
}
void gs_renderer::set_instruction_channel(std::shared_ptr<arm64_fault_channel> channel)
{
	{
		std::lock_guard lock(m_instruction_channel_mutex);
		channel.swap(m_instruction_channel);
	}
	if (channel) channel->request_close();
}
u64 gs_renderer::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<gs_renderer>&>(*this));
}
gs_renderer::runtime& gs_renderer::device_runtime()
{
	if (!m_runtime || m_failed)
		throw std::logic_error("Native renderer is uninitialized or failed");
	service_mapping_notifications();
	return *m_runtime;
}
void gs_renderer::on_init_thread()
{
	@autoreleasepool
	{
		if (m_runtime)
			throw std::logic_error("Native renderer initialized twice");
		m_failed = false;
		auto candidate = std::make_unique<runtime>(m_bindings, m_io_mapping, m_tiles);
		if (backend_config.supports_hw_msaa && (![candidate->device.device() supportsTextureSampleCount:2] ||
			![candidate->device.device() supportsTextureSampleCount:4]))
			throw std::runtime_error("Native RSX MSAA requires two- and four-sample support");
		m_tasks.bind_owner();
		m_mapping_notifications.bind_owner();
		try
		{
			candidate->registration.install(m_bindings.shutdown, [&]
			{
				m_bindings.bind_runtime(candidate->device, candidate->surfaces, candidate->textures);
			});
			candidate->frame_service.initialize(m_frame);
			if (!candidate->frame.attached())
				throw std::runtime_error("Native frame service did not attach the Metal4 layer");
			auto reports = std::make_unique<rsx_report_controller>(candidate->queries, []
																   { return g_cfg.video.precise_zpass_count.get(); }, [this, owner = candidate.get()](std::uint32_t address, std::uint64_t timestamp, std::uint32_t value)
																   {
				try
				{
					if (m_failed)
						throw std::logic_error("Failed native renderer cannot publish reports");
					store_guest_report(owner->surfaces, owner->textures, address, timestamp, value, m_bindings.store_report);
				}
				catch (...)
				{
					fail_renderer();
					throw;
				} }, [this, owner = candidate.get()]
				{
					try
					{
						if (m_failed) throw std::logic_error("Failed native renderer cannot submit reports");
						owner->surfaces.submit_draw_batch();
					}
					catch (...) { fail_renderer(); throw; }
				});
			if (m_bindings.arm64_faults)
			{
				candidate->create_instruction_faults(*m_bindings.arm64_faults, m_bindings.memory);
				set_instruction_channel(candidate->fault_channel);
				if (m_failed)
					throw std::runtime_error("Native renderer failed before ARM64 fault channel publication");
				m_bindings.arm64_faults->install_channel(candidate->fault_channel);
				if (m_failed)
					throw std::runtime_error("Native renderer failed during ARM64 fault channel publication");
			}
			zcull_ctrl = std::move(reports);
		}
		catch (...)
		{
			const auto failure = std::current_exception();
			set_instruction_channel({});
			candidate->close_instruction_faults();
			m_tasks.close();
			try
			{
				candidate->frame_service.shutdown();
			}
			catch (...)
			{
			}
			try
			{
				candidate->registration.close();
			}
			catch (...)
			{
			}
			std::rethrow_exception(failure);
		}
		m_runtime = std::move(candidate);
	}
}
void gs_renderer::on_exit()
{
	@autoreleasepool
	{
		std::exception_ptr failure;
		const auto finish = [&](const auto& operation)
		{
			try { operation(); }
			catch (...)
			{
				if (!failure) failure = std::current_exception();
				fail_renderer();
			}
		};
		set_instruction_channel({});
		if (m_runtime) m_runtime->close_instruction_faults();
		if (m_runtime) finish([&] { service_mapping_notifications(mapping_notifications::maximum_ranges); });
		m_tasks.close();
		if (m_runtime)
		{
			if (!m_failed)
			{
				finish([&] { m_runtime->images.synchronize(); });
				finish([&] { m_runtime->dma.synchronize(); });
			}
			finish([&] { m_runtime->surfaces.synchronize(); });
			finish([&]
			{
				m_runtime->frame.drain();
				m_runtime->collect_capture(*m_frame);
			});
			finish([&] { m_runtime->collect_display_uploads(true); });
			finish([&] { m_runtime->frame_service.shutdown(); });
		}
		// No GSFrame legacy draw_context was created by this renderer.
		if (m_failed) zcull_ctrl.reset();
		finish([&] { rsx::thread::on_exit(); });
		zcull_ctrl.reset();
		if (m_runtime)
		{
			finish([&] { m_runtime->registration.close(); });
			m_runtime.reset();
		}
		if (m_frame && !m_continuous_mode)
			finish([&] { m_frame->hide(); });
		if (failure) std::rethrow_exception(failure);
	}
}
std::optional<framebuffer_snapshot> gs_renderer::current_framebuffer(rsx::framebuffer_creation_context mode)
{
	get_framebuffer_layout(mode, m_framebuffer_layout);
	if (!m_graphics_state.test(rsx::rtt_config_valid))
		return {};
	auto& state = rsx::method_registers;
	const std::array offsets{NV4097_SET_SURFACE_COLOR_AOFFSET, NV4097_SET_SURFACE_COLOR_BOFFSET,
		NV4097_SET_SURFACE_COLOR_COFFSET, NV4097_SET_SURFACE_COLOR_DOFFSET, NV4097_SET_SURFACE_ZETA_OFFSET};
	const std::array contexts{NV4097_SET_CONTEXT_DMA_COLOR_A, NV4097_SET_CONTEXT_DMA_COLOR_B,
		NV4097_SET_CONTEXT_DMA_COLOR_C, NV4097_SET_CONTEXT_DMA_COLOR_D, NV4097_SET_CONTEXT_DMA_ZETA};
	const auto window_x = state.window_offset_x(), window_y = state.window_offset_y();
	std::uint32_t zero_origins = (window_x || window_y) ? framebuffer_window_zero_colors(m_framebuffer_layout) : 0;
	if ((window_x || window_y) && !m_framebuffer_layout.zeta_address && m_framebuffer_layout.actual_zeta_pitch)
	{
		const auto depth = framebuffer_depth_bytes(static_cast<surface_depth>(m_framebuffer_layout.depth_format));
		const auto write_limit = std::min<std::uint32_t>(state.surface_clip_width(), std::uint32_t(state.scissor_origin_x()) + state.scissor_width());
		const auto minimum_pitch = std::size_t(depth) * m_framebuffer_layout.aa_factors[0] * write_limit;
		const auto base = translate_rsx_vm_range(state.registers[NV4097_SET_SURFACE_ZETA_OFFSET], state.registers[NV4097_SET_CONTEXT_DMA_ZETA], 1);
		zero_origins |= framebuffer_window_zero_depth(m_framebuffer_layout, base, window_x, window_y, state.surface_z_pitch(), minimum_pitch);
	}
	auto snapshot = framebuffer_snapshot::capture(m_framebuffer_layout, true, {}, m_bindings.snapshot_bytes, [&](framebuffer_surface& surface)
	{
		const auto dma = state.registers[contexts[surface.physical_slot]];
		unsigned location;
		if (dma == CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER || dma == CELL_GCM_LOCATION_MAIN) location = 1;
		else if (dma == CELL_GCM_CONTEXT_DMA_MEMORY_FRAME_BUFFER || dma == CELL_GCM_LOCATION_LOCAL) location = 0;
		else throw std::invalid_argument("Native framebuffer DMA context is not local/main");
		const auto raw = state.registers[offsets[surface.physical_slot]];
		const auto pitch = surface.physical_slot == 4 ? m_framebuffer_layout.actual_zeta_pitch : m_framebuffer_layout.actual_color_pitch[surface.physical_slot];
		map_framebuffer_window_surface(surface, raw, location, window_x, window_y, pitch,
			translate_rsx_vm_range, m_io_mapping, m_tiles, g_cfg.video.handle_tiled_memory.get(), m_bindings.snapshot_bytes);
	}, zero_origins);
	// Shared consumers must observe the corrected physical origins too.
	apply_framebuffer_origins(m_framebuffer_layout, snapshot);
	on_framebuffer_layout_updated();
	return snapshot;
}
void gs_renderer::begin()
{
	(void)device_runtime();
	rsx::thread::begin();
}
void gs_renderer::begin_conditional_rendering(const std::vector<rsx::reports::occlusion_query_info*>& sources)
{
	auto& runtime = device_runtime();
	if (sources.empty() || runtime.conditional_report)
		throw std::logic_error("Invalid native conditional report transition");
	auto& reports = static_cast<rsx_report_controller&>(*zcull_ctrl);
	runtime.surfaces.submit_draw_batch();
	std::optional<report_predicate> predicate;
	try
	{
		predicate = reports.submit_conditional_predicate(cond_render_ctrl.eval_address, cond_render_ctrl.eval_sync_tag);
	}
	catch (...)
	{
		fail_renderer();
		throw;
	}
	if (!predicate)
	{
		// Forwarded and unsupported report types preserve the RSX scheduler's
		// memory-based evaluation and retirement ordering.
		if (const auto previous = runtime.device.last_submission()) previous->wait();
		zcull_ctrl->read_barrier(this, cond_render_ctrl.eval_address, sources.front());
		if (cond_render_ctrl.eval_pending())
			throw std::logic_error("Native conditional report fallback did not resolve the predicate");
		return;
	}
	runtime.conditional_report = std::move(predicate);
	rsx::thread::begin_conditional_rendering(sources);
}
void gs_renderer::end_conditional_rendering()
{
	if (m_runtime) m_runtime->conditional_report.reset();
	rsx::thread::end_conditional_rendering();
}
void gs_renderer::end()
{
	@autoreleasepool
	{
		auto& runtime = device_runtime();
		try
		{
			if (skip_current_frame || cond_render_ctrl.disable_rendering())
			{
				if (runtime.surface_batch_submitted)
				{
					runtime.surfaces.synchronize();
					runtime.poll_surface_batch_completion();
				}
				execute_nop_draw();
				rsx::thread::end();
				return;
			}
			struct owned_pass
			{
				rsx_draw_packet packet;
				framebuffer_snapshot framebuffer;
			};
			auto& state = rsx::method_registers;
			const auto predicate = runtime.conditional_report;
			rsx::vertex_input_layout layout;
			auto& pending_textures = runtime.draw_textures;
			const auto emit = [&](unsigned, unsigned)
			{
				runtime.publish_ready_draw_textures();
				const auto settled = predicate ? predicate->settled_cpu_result(runtime.device) : std::nullopt;
				if (!capture_current_frame && settled == false)
					return;
				const auto* draw_predicate = predicate && settled != true ? &*predicate : nullptr;
				// Preparation owns independent native images. CPU producers run
				// during all surface/upload/helper/encoder waits. The final short
				// CPU epoch validates every source while copying the whole packet.
				// Draw-owned uploads are recorded after that independent copy.
				for (unsigned attempt = 0; attempt < 3; ++attempt)
				{
					const versioned_draw_texture_job* reuse = nullptr;
					try
					{
						vertex_texture_snapshot vertex;
						fragment_texture_snapshot fragment;
						bool skip = false;
						checked_cpu_capture(m_bindings.capture_epoch, [&]
											{
						if (!current_framebuffer(rsx::framebuffer_creation_context::context_draw))
						{
							skip = true;
							return;
						}
						analyse_current_rsx_pipeline();
						vertex = vertex_texture_snapshot::capture(std::span<const std::uint32_t>(state.registers), current_vp_metadata.referenced_textures_mask);
						fragment = capture_fragment_texture_snapshot(state, current_fp_metadata.referenced_textures_mask);
						for (auto it = pending_textures.rbegin(); it != pending_textures.rend(); ++it)
							if (it->current(vertex, fragment))
							{
								reuse = &*it;
								break;
							}
						});
						if (skip)
							return;
						std::optional<versioned_draw_texture_job> new_job;
						if (!reuse)
							new_job.emplace(runtime.executor.enqueue_versioned_textures(vertex, fragment, {},
								draw_predicate, texture_upload_path::draw_batch));
						const auto& texture_job = reuse ? *reuse : *new_job;
						std::optional<owned_pass> owned;
						checked_cpu_capture(m_bindings.capture_epoch, [&]
											{
						auto framebuffer = current_framebuffer(rsx::framebuffer_creation_context::context_draw);
						if (!framebuffer) return;
						auto program = prepare_owned_programs(state, [&](const auto& v, const auto& f)
						{
							if (!texture_job.current(v, f)) throw texture_revision_conflict();
							return rsx_texture_program_epoch::capture_metadata(state, texture_job);
						});
						GRAPH_frontend().analyse_inputs_interleaved(layout, current_vp_metadata);
						draw_capture_limits limits;
						limits.constants_are_current_pass = true;
						limits.constants = &runtime.constants;
						owned.emplace(owned_pass{capture_owned_pass(state, layout, provoking_vertex::last, limits), std::move(*framebuffer)}); });
						if (!owned)
							return;
						draw_execution_state execution;
						execution.queries = &runtime.queries;
						if (draw_predicate) execution.predicate = *draw_predicate;
						[[maybe_unused]] const auto pipeline_preparation = !texture_job.ready() ?
							runtime.executor.prewarm_stages(owned->packet, owned->framebuffer, texture_job, execution) : std::nullopt;
						runtime.executor.execute(owned->packet, owned->framebuffer, texture_job, execution);
						if (new_job && new_job->has_completion()) pending_textures.push_back(std::move(*new_job));
						return;
					}
					catch (const predicated_texture_suppressed&)
					{
						return;
					}
					catch (const texture_revision_conflict&)
					{
						if (reuse)
							std::erase_if(pending_textures, [&](const auto& job) { return &job == reuse; });
						if (attempt == 2)
							throw;
					}
				}
			};
			// Conditional clauses can apply RSX dependencies before their first draw.
			if (capture_current_frame || predicate)
				runtime.surfaces.synchronize();
			const bool defer_predicated = predicate && !capture_current_frame && !state.current_draw_clause.empty();
			using predicated_completion = surface_cache::predicated_draw_completion;
			auto batch = runtime.surfaces.batch_draws(64, defer_predicated ? predicated_completion::deferred : predicated_completion::synchronous);
			runtime.poll_surface_batch_completion();
			(void)drive_draw_clause(state.current_draw_clause, m_ctx, emit, 4, clause_cleanup_owner::rsx_thread_end, bool(predicate), [&]
			{
				// The next pass may execute guest dependencies before touching a surface.
				if (defer_predicated) runtime.surfaces.synchronize();
			});
			if (capture_current_frame || clause_requires_draw_completion(state.current_draw_clause))
			{
				batch.finish();
				runtime.publish_ready_draw_textures();
			}
			else
			{
				runtime.submit_surface_batch(batch);
				if (defer_predicated) m_eng_interrupt_mask |= rsx::backend_interrupt;
			}
			rsx::thread::end();
		}
		catch (...)
		{
			// No trailing clause dependencies after a failed preparation/submission.
			// Require teardown instead of resuming a partially modified RSX epoch.
			fail_renderer();
			in_begin_end = false;
			GRAPH_frontend().clear_push_buffers();
			throw;
		}
	}
}
void gs_renderer::clear_surface(u32 mask)
{
	@autoreleasepool
	{
		if (skip_current_frame || !(mask & 0xf3u))
			return;
		auto& runtime = device_runtime();
		try
		{
			struct clear_capture
			{
				framebuffer_snapshot framebuffer;
				clear_rectangle area;
				u32 mask, color, depth;
				u8 stencil_mask;
			};
			std::optional<clear_capture> captured;
			checked_cpu_capture(m_bindings.capture_epoch, [&]
								{
			auto& state = rsx::method_registers;
			if (!state.stencil_mask()) mask &= ~2u;
			u8 mode = rsx::framebuffer_creation_context::context_draw;
			if (mask & 0xf0u) mode |= rsx::framebuffer_creation_context::context_clear_color;
			if (mask & 3u) mode |= rsx::framebuffer_creation_context::context_clear_depth;
			auto framebuffer = current_framebuffer(rsx::framebuffer_creation_context{mode});
			if (!framebuffer) return;
			captured.emplace(clear_capture{std::move(*framebuffer),
				{state.scissor_origin_x(), state.scissor_origin_y(), state.scissor_width(), state.scissor_height()},
				mask, state.registers[NV4097_SET_COLOR_CLEAR_VALUE], state.registers[NV4097_SET_ZSTENCIL_CLEAR_VALUE], state.stencil_mask()}); });
			if (!captured)
				return;
			const auto targets = acquire_framebuffer(captured->framebuffer, runtime.surfaces);
			auto batch = runtime.surfaces.batch_draws();
			runtime.poll_surface_batch_completion();
			for (const auto& color : targets.colors)
				if (color && (captured->mask & 0xf0u))
					runtime.surfaces.clear(color, captured->mask & 0xf0u, captured->color, 0, 255, captured->area);
			if (targets.depth && (captured->mask & 3u))
				runtime.surfaces.clear(targets.depth, captured->mask & 3u, 0, captured->depth, captured->stencil_mask, captured->area);
			if (capture_current_frame)
				batch.finish();
			else
			{
				runtime.submit_surface_batch(batch);
				m_eng_interrupt_mask |= rsx::backend_interrupt;
			}
		}
		catch (...)
		{
			fail_renderer();
			throw;
		}
	}
}
void gs_renderer::flip(const rsx::display_flip_info_t& info)
{
	@autoreleasepool
	{
		auto& runtime = device_runtime();
		try
		{
			runtime.collect_capture(*m_frame);
			runtime.collect_display_uploads();
		}
		catch (...) { fail_renderer(); throw; }
		if (info.skip_frame)
		{
			runtime.surfaces.synchronize();
			rsx::thread::flip(info);
			return;
		}
		for (unsigned attempt = 0; attempt < 3; ++attempt)
		{
			try
			{
				std::optional<display_snapshot> display;
				frame_policy policy;
				checked_cpu_capture(m_bindings.capture_epoch, [&]
									{
				const auto& av = g_fxo->get<rsx::avconf>();
				display = capture_display_snapshot(display_buffers, display_buffers_count, info, av);
				const auto scaling = g_cfg.video.output_scaling.get();
				const auto vsync = g_cfg.video.vsync.get();
				auto pacing = frame_pacing::synchronized;
				if (vsync == vsync_mode::off) pacing = frame_pacing::unsynchronized;
				else if (vsync == vsync_mode::adaptive) pacing = frame_pacing::adaptive;
				policy = capture_frame_policy(av, g_cfg.video.stretch_to_display_area.get(), pacing,
					scaling == output_scaling_mode::bilinear, g_cfg.video.full_rgb_range_output.get());
				policy.fsr = scaling == output_scaling_mode::fsr;
				policy.fsr_sharpness = 2.f - g_cfg.video.rcas_sharpening_intensity.get() / 50.f;
				stereo_config stereo(true);
				policy.stereo = capture_rsx_stereo(stereo, av.state && av.stereo_enabled); });
				// Native synchronization stays outside the short producer epochs.
				runtime.surfaces.synchronize();
				std::optional<display_frame_job> upload;
				if (display) upload.emplace(runtime.display.prepare_frame(*display,runtime.surfaces));
				// Main-thread run-loop observation publishes owned geometry. Flips
				// consume that snapshot without posting or waiting for a UI job.
				std::optional<present_image> image;
				checked_cpu_capture(m_bindings.capture_epoch,[&]
				{
					const auto fresh = capture_display_snapshot(display_buffers,display_buffers_count,info,g_fxo->get<rsx::avconf>());
					if (fresh != display) throw surface_mapping_conflict();
					if (upload) image = upload->image_for_encoding();
				});
				if (upload) runtime.display_uploads.push_back(std::move(*upload));
				runtime.present(image, policy, *m_frame, m_overlay_manager);
				rsx::thread::flip(info);
				return;
			}
			catch (const surface_conflict_error&) { if (attempt == 2) throw; }
			catch (const surface_mapping_conflict&) { if (attempt == 2) throw; }
		}
	}
}
bool gs_renderer::release_GCM_label(u32 /*type*/, u32 address, u32 value)
{
	@autoreleasepool
	{
		auto& runtime = device_runtime();
		try
		{
			release_guest_label(runtime.surfaces, runtime.textures, address, value, [&]
								{
			// Our store makes the shared NV47 same-value early return true.
			// Perform every prerequisite here before making the value visible.
			synchronize_guest_producers([] { rsx::mm_flush(); }, [] { return g_fxo->get<rsx::dma_manager>().sync(); });
			runtime.images.synchronize();
			runtime.dma.synchronize();
			m_graphics_state |= rsx::pipeline_state::fragment_program_needs_rehash;
			rsx::thread::sync(); }, m_bindings.store_label);
			return true;
		}
		catch (...)
		{
			fail_renderer();
			throw;
		}
	}
}
bool gs_renderer::release_GCM_flip_label(u32 address, u32 value)
{
	// Flip reset must remain available after GPU failure while VM services live.
	// It neither accesses native GPU state nor clears the renderer failure flag.
	if (!m_runtime)
		throw std::logic_error("Uninitialized native renderer cannot publish flip labels");
	try
	{
		store_guest_flip_label(address, value, m_bindings.store_label);
		return true;
	}
	catch (...)
	{
		fail_renderer();
		throw;
	}
}
bool gs_renderer::buffer_transfer(u32 notify)
{
	@autoreleasepool
	{
		auto& runtime = device_runtime();
		try
		{
			synchronize_guest_producers([]
										{ rsx::mm_flush(); }, []
										{ return g_fxo->get<rsx::dma_manager>().sync(); });
			std::optional<dma_plan> plan;
			std::optional<scaled_image_mapping_epoch> mapping;
			dma_command command{};
			checked_cpu_capture(m_bindings.capture_epoch, [&]
								{
				command = capture_dma_command(rsx::method_registers, notify);
				plan = plan_dma(command);
				validate_dma(*plan, m_bindings.snapshot_bytes, true);
				mapping.emplace([command, source = plan->source, destination = plan->destination]
				{
					if (capture_dma_command(rsx::method_registers, command.notify) != command)
						throw scaled_image_mapping_conflict();
					return scaled_image_mapping_epoch::mappings{
						probe_rsx_dma_mapping(source, command.columns, command.rows, command.source_dma, translate_rsx_vm_range),
						probe_rsx_dma_mapping(destination, command.columns, command.rows, command.destination_dma, translate_rsx_vm_range)};
				}, [memory = m_bindings.memory, io = m_io_mapping]
				{
					return scaled_image_mapping_versions{memory->mapping_generation(), io->current()};
				}, [memory = m_bindings.memory, io = m_io_mapping](const auto& values)
				{
					return dma_mapping_versions(values, *memory, *io);
				});
				attach_dma_mapping(plan->source, mapping->original()[0]); attach_dma_mapping(plan->destination, mapping->original()[1]);
				validate_dma(*plan, m_bindings.snapshot_bytes, true); });
			// Materialize report producers before owned source capture. The native
			// path uses completion rather than the legacy deferred CPU-copy shortcut.
			for_each_dma_active_range(plan->source, plan->columns, plan->rows, [&](const auto range)
				{ (void)read_barrier(range.address, static_cast<u32>(range.size), true); });
			for_each_dma_active_range(plan_dma(command).destination, plan->columns, plan->rows, [&](const auto range)
				{ invalidate_fragment_program(command.destination_dma, range.address, static_cast<u32>(range.size)); });
			runtime.dma.execute(std::move(*plan), std::chrono::seconds(30), {}, [owned = std::move(*mapping)]
								{ owned.validate(); });
			m_graphics_state |= rsx::pipeline_state::fragment_program_needs_rehash;
			return true;
		}
		catch (...)
		{
			fail_renderer();
			throw;
		}
	}
}
bool gs_renderer::scaled_image_from_memory(const rsx::blit_src_info& source, const rsx::blit_dst_info& destination, bool interpolate)
{
	auto& runtime = device_runtime();
	try
	{
		synchronize_guest_producers([]
									{ rsx::mm_flush(); }, []
									{ return g_fxo->get<rsx::dma_manager>().sync(); });
		std::optional<scaled_image_plan> plan;
		std::optional<scaled_image_mapping_epoch> mapping;
		checked_cpu_capture(m_bindings.capture_epoch, [&]
							{
			auto& state = rsx::method_registers;
			plan = capture_scaled_image(source, destination, state.blit_engine_output_width(),
				state.blit_engine_output_height(), interpolate, m_bindings.snapshot_bytes);
			const auto source_offset = state.blit_engine_input_offset();
			const auto destination_offset = destination.swizzled ? state.blit_engine_nv309E_offset() : state.blit_engine_output_offset_nv3062();
			const auto source_delta = plan->source.range.address - source.rsx_address;
			const auto destination_delta = plan->destination.range.address - destination.rsx_address;
			const bool tiling = g_cfg.video.handle_tiled_memory.get();
			const auto lookup = [tiles = m_tiles, tiling](std::uint32_t io, unsigned dma, std::uint32_t, std::size_t) -> std::optional<scaled_image_tile_mapping>
			{
				if (!tiling || (dma != CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER && dma != CELL_GCM_LOCATION_MAIN)) return {};
				// Select in IO space:later disjoint EA entries may lie outside the
				// first tile EA's apparent linear range used by the legacy lookup.
				for (const auto& tile : tiles->snapshot())
					if (tile.bound && tile.location == CELL_GCM_LOCATION_MAIN && io >= tile.offset && io - tile.offset < tile.size)
						return scaled_image_tile_mapping{translate_rsx_vm_range(tile.offset, dma, 1), tile.size, tile.pitch, tile.bank, tile.location, tile.offset, tile.comp, tile.base};
				return {};
			};
			mapping.emplace([source_offset, destination_offset, source_delta, destination_delta,
				source_dma = source.dma, destination_dma = destination.dma,
				source_bytes = plan->source.range.size, destination_bytes = plan->destination.range.size, lookup, tiling]
			{
				return scaled_image_mapping_epoch::mappings{
					probe_scaled_image_mapping(lookup, translate_rsx_vm_range, source_offset, source_dma, source_delta, source_bytes, tiling),
					probe_scaled_image_mapping(lookup, translate_rsx_vm_range, destination_offset, destination_dma, destination_delta, destination_bytes, tiling)};
			}, [memory = m_bindings.memory, io = m_io_mapping, tiles = m_tiles]
				{
					return scaled_image_mapping_versions{memory->mapping_generation(), io->current(), tiles->current()};
				}, [memory = m_bindings.memory, io = m_io_mapping, tiles = m_tiles](const auto& values)
				{
					return scaled_image_mapping_versions{scaled_image_mapping_generation(values, *memory), scaled_image_io_mapping_generation(values, *io), scaled_image_tile_mapping_generation(values, *tiles)};
				});
			// The shared decoder ran before this exclusion epoch. Refuse stale
			// translated addresses instead of capturing unrelated guest storage.
			if (translate_rsx_vm_range(source_offset, source.dma, 1) != source.rsx_address || translate_rsx_vm_range(destination_offset, destination.dma, 1) != destination.rsx_address)
				throw scaled_image_mapping_conflict();
			const auto attach = [](guest_surface_layout& image, std::optional<banked_tile_layout>& owned, std::vector<guest_storage_segment>& segments,
				const scaled_image_memory_mapping& current, std::uint32_t io)
			{
				if (current.tile)
				{
					const auto& tile = *current.tile;
					owned = banked_tile_layout{tile.address, tile.size, tile.pitch, tile.bank, io - tile.io_offset};
					image.range.address = tile.address + owned->offset;
					image = attach_banked_tile(std::move(image), *owned);
				}
				else image.range.address = current.address;
				if (current.segments.size() > 1) segments = current.segments;
			};
			attach(plan->source, plan->source_tile, plan->source_segments, mapping->original()[0], source_offset + source_delta);
			attach(plan->destination, plan->destination_tile, plan->destination_segments, mapping->original()[1], destination_offset + destination_delta);
			validate_scaled_image(*plan, m_bindings.snapshot_bytes); });
		runtime.images.enqueue(std::move(*plan), std::chrono::seconds(30), {}, [owned = std::move(*mapping)]
							   { owned.validate(); });
		m_graphics_state |= rsx::pipeline_state::fragment_program_needs_rehash;
		return true;
	}
	catch (...)
	{
		fail_renderer();
		throw;
	}
}
bool gs_renderer::on_access_violation(u32 address, bool writing)
{
	@autoreleasepool
	{
		if (!m_tasks.available())
		{
			// A staged instruction must never fall through to an unguarded
			// VM fallback/retry after its render-owner queue has closed.
			if (auto interrupted = host_store_transaction::pause_current(*m_bindings.memory))
			{
				g_fxo->get<rsx::dma_manager>().release_native_source_reservations();
				fail_renderer();
				throw std::runtime_error("Native DMA fault lost its render owner before guarded retry");
			}
			return false;
		}
		memory_fault_request request{address, writing, {}};
		auto& dma = g_fxo->get<rsx::dma_manager>();
		const bool offloader = dma.is_current_thread();
		if (const auto range = dma.get_native_fault_range(writing))
			request.dma_transfer_range = guest_surface_range{range->start, range->length()};
		else if (offloader)
		{
			const auto range = dma.get_fault_range(writing);
			request.dma_transfer_range = guest_surface_range{range.start, range.length()};
		}
		if (request.dma_transfer_range) validate_fault_transfer_range(*request.dma_transfer_range, address);
		if (offloader) dma.set_mem_fault_flag();
		struct offloader_recovery
		{
			rsx::dma_manager* manager;
			~offloader_recovery()
			{
				if (manager)
					manager->clear_mem_fault_flag();
			}
		} recovery{offloader ? &dma : nullptr};
		// Yield the actual writer before MM ordering or synchronous owner
		// handoff. Resume it before clearing recovery and retrying the store.
		auto store_fault = host_store_transaction::pause_current(*m_bindings.memory);
		dma.release_native_source_reservations();
		auto writer_fault = dma.pause_native_writer_fault();
		if (request.dma_transfer_range && !writer_fault)
			throw std::logic_error("Native DMA writer could not pause for fault recovery");
		rsx::mm_flush(address);
		if (!is_current_thread())
			vm::temporary_unlock(); // Publish CPU wait/drop passive VM lock before handoff.
		bool handled = false;
		m_tasks.invoke([this, request, &handled, &writer_fault, &store_fault]
					   {
		const auto resolve = [&]
		{
			try
			{
				if (m_failed)
					throw std::logic_error("Failed native renderer cannot recover memory faults");
				handled = m_bindings.memory_fault(request);
				if (!handled && zcull_ctrl)
					handled = zcull_ctrl->on_access_violation(request.address);
			}
			catch (...)
			{
				fail_renderer();
				throw;
			}
		};
		if (writer_fault)
		{
			auto fault_capture = writer_fault->capture_context();
			if (!fault_capture) throw std::logic_error("Native DMA fault writer resumed before owner recovery");
			if (store_fault) store_fault->service(resolve);
			else resolve();
		}
		else resolve();
		});
		if (store_fault)
		{
			if (handled)
			{
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
				for (;;)
				{
					dma.resume_native_source_reservations();
					const auto result = store_fault->attempt_resume();
					if (result == host_store_transaction::resume_result::ready) break;
					dma.release_native_source_reservations();
					if (result != host_store_transaction::resume_result::busy ||
						std::chrono::steady_clock::now() >= deadline || !m_tasks.available() || m_failed)
					{
						handled = false;
						break;
					}
					std::this_thread::yield();
				}
			}
			if (!handled)
			{
				fail_renderer();
				throw std::runtime_error("Native DMA fault cannot safely retry its original physical mapping");
			}
		}
		else dma.resume_native_source_reservations();
		return handled;
	}
}
std::optional<host_write_journal::intent> gs_renderer::on_native_host_write(void* destination, const void* source, u64 bytes)
{
	const auto access = describe_rsx_host_access(destination, source, bytes);
	if (!access) return {};
	if (!m_tasks.available() || m_failed)
		throw std::logic_error("Native guest DMA preparation requires a live render owner");
	return stage_rsx_host_write(g_fxo->get<rsx::dma_manager>(), m_bindings.memory, access, is_current_thread(),
		[this](const auto& operation) { m_tasks.invoke(operation); },
		[this](const rsx_host_access& ranges)
		{
			try { return prepare_host_transfer(device_runtime().surfaces, *m_bindings.memory, ranges.destination, ranges.source); }
			catch (...) { fail_renderer(); throw; }
		});
}

void gs_renderer::update_tile(u32 index, const std::function<void(GcmTileInfo&)>& mutation)
{
	m_tiles->execute([&]
	{
		auto value = rsx_tile(m_tiles->get(index));
		mutation(value);
		m_tiles->update(index, native_tile(value));
		tiles[index] = value;
	});
}
bool gs_renderer::is_tile_bound(u32 index) const
{
	return m_tiles->get(index).bound;
}
void gs_renderer::on_io_mapping_changed(u32 offset, u32 size) noexcept
{
	m_io_mapping->notify({offset, size});
}
bool gs_renderer::defer_memory_map_notification(u32 /*address*/, u32 /*size*/) noexcept
{
	// Old unmaps remain in the bitmap;never acquire m_mtx_task from VM mapping.
	if (m_mapping_notifications.pending()) m_eng_interrupt_mask |= rsx::backend_interrupt;
	return true;
}
bool gs_renderer::defer_memory_unmap_notification(u32 address, u32 size) noexcept
{
	m_mapping_notifications.notify({address, size}, mapping_notifications::kind::unmap);
	m_eng_interrupt_mask |= rsx::backend_interrupt;
	return true;
}
void gs_renderer::on_invalidate_memory_range(const utils::address_range32& range, rsx::invalidation_cause cause)
{
	if (!range.valid()) return;
	if (cause != rsx::invalidation_cause::read && cause != rsx::invalidation_cause::unmap)
		throw std::invalid_argument("Native mapping notification has an unsupported invalidation cause");
	m_mapping_notifications.notify({range.start, range.length()}, cause == rsx::invalidation_cause::read ?
		mapping_notifications::kind::read : mapping_notifications::kind::unmap);
	m_eng_interrupt_mask |= rsx::backend_interrupt;
}
void gs_renderer::service_mapping_notifications(std::size_t range_limit)
{
	if (!m_runtime || !m_mapping_notifications.pending()) return;
	try
	{
		m_mapping_notifications.drain([&](guest_surface_range range, mapping_notifications::kind cause)
		{
			@autoreleasepool
			{
				(void)m_runtime->mapping_invalidations.service(range, cause);
				if (m_bindings.memory_invalidated) m_bindings.memory_invalidated(utils::address_range32::start_length(range.address, static_cast<u32>(range.size)),
					cause == mapping_notifications::kind::read ? rsx::invalidation_cause::read : rsx::invalidation_cause::unmap);
			}
		}, range_limit);
	}
	catch (...) { m_eng_interrupt_mask |= rsx::backend_interrupt; throw; }
	if (m_mapping_notifications.pending()) m_eng_interrupt_mask |= rsx::backend_interrupt;
}
void gs_renderer::do_local_task(rsx::FIFO::state state)
{
	@autoreleasepool
	{
		if (m_runtime && !m_failed && m_bindings.arm64_faults)
			m_runtime->drain_instruction_faults(m_bindings.arm64_faults->retain_instruction);
		if (m_runtime && !m_failed && !guest_memory_epoch::active())
		{
			try { m_runtime->poll_surface_batch_completion(); (void)m_runtime->images.poll(); }
			catch (...) { fail_renderer(); throw; }
		}
		// Process fault waiters before the common path takes m_mtx_task or waits
		// for DMA. Reentrant semaphore polling must not execute another foreign job.
		service_mapping_notifications();
		if (m_tasks.available())
			(void)m_tasks.drain();
		m_bindings.local_task(state);
		rsx::thread::do_local_task(state);
		// Common stopped/save handling queues read/unmap while holding its task
		// lock. Only service after that routine has returned and released it.
		// Stop/save must finish its queued read/unmap barrier before returning.
		service_mapping_notifications(is_stopped() ? mapping_notifications::maximum_ranges : 256);
		// A producer may enqueue while the common routine clears backend_interrupt.
		if (m_tasks.pending() || m_mapping_notifications.pending())
			m_eng_interrupt_mask |= rsx::backend_interrupt;
	}
}
void gs_renderer::on_semaphore_acquire_wait()
{
	@autoreleasepool
	{
		if ((m_runtime && !m_failed && m_runtime->instruction_fault_pending()) ||
			m_tasks.pending() || m_mapping_notifications.pending() || (m_eng_interrupt_mask & rsx::backend_interrupt) ||
			(async_flip_requested & flip_request::emu_requested))
			do_local_task(rsx::FIFO::state::lock_wait);
	}
}
} // namespace metal4
