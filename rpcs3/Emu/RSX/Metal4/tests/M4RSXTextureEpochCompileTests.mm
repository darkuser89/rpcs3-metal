#include "Emu/RSX/GSRender.h"
#include "Emu/RSX/rsx_methods.h"
#include "M4DrawExecutor.h"
#include "M4RSXTextureEpoch.h"
#include "M4GSRender.h"
#include "M4RSXFrameService.h"
#include "M4RSXReports.h"
#include "M4RSXVMAccess.h"
#include "M4RSXMappingObserver.h"
#include "stdafx.h"

static_assert(!std::is_abstract_v<metal4::gs_renderer>);
static_assert(std::is_base_of_v<GSRender, metal4::gs_renderer>);
static_assert(!std::is_abstract_v<metal4::rsx_report_controller>);

// Same real provider owns both outer suspension and nested revision reads.
// Other services and installed flag remain absent:never activates a backend.
metal4::renderer_bindings bind_real_capture_checked(std::function<std::shared_ptr<void>()> host_guard)
{
	auto access = std::make_shared<metal4::rsx_vm_access>(std::move(host_guard));
	metal4::renderer_bindings bindings;
	auto memory = std::make_shared<metal4::guarded_guest_memory>(access);
	bindings.memory = memory;
	bindings.vm_mapping_observer = std::make_shared<metal4::rsx_mapping_observer>(memory);
	bindings.capture_epoch = [access](const std::function<void()>& operation) { access->capture_cpu(operation); };
	bindings.store_label = [access, memory](std::uint32_t address, std::uint32_t value) { access->store_label(*memory, address, value); };
	bindings.store_report = [access, memory](std::uint32_t address, std::uint64_t timestamp, std::uint32_t value) { access->store_report(*memory, address, timestamp, value); };
	return bindings;
}

void compile_frame_service(metal4::native_frame& frame, GSFrameBase* platform)
{
	metal4::rsx_frame_service service(frame);
	service.initialize(platform);
	service.shutdown();
}

// Compile-only:never constructs a renderer or starts RPCS3. Concrete bodies
// instantiate real protected program preparation,vertex layout and clause
// dispatch together with native texture retention and prepared execution.
class native_texture_epoch_compile : public metal4::rsx_program_frontend<GSRender>
{
  public:
	std::optional<metal4::display_snapshot> freeze_display(const rsx::display_flip_info_t& info, const rsx::avconf& av)
	{
		return metal4::capture_display_snapshot(display_buffers, display_buffers_count, info, av);
	}
	metal4::frame_policy freeze_policy(const rsx::avconf& av)
	{
		return metal4::capture_frame_policy(av, false, metal4::frame_pacing::adaptive, true, true);
	}
	struct epoch
	{
		std::shared_ptr<void> guard;
		metal4::rsx_texture_program_epoch program;
		metal4::framebuffer_snapshot framebuffer;
	};
	std::uint32_t submit_checked(rsx::rsx_state& state, rsx::vertex_input_layout& layout,
								 const metal4::framebuffer_snapshot& framebuffer, metal4::draw_executor& executor,
								 const std::function<std::shared_ptr<void>(unsigned, unsigned)>& acquire_guard)
	{
		const auto prepare = [&](unsigned pass, unsigned flags) -> std::optional<epoch>
		{
			auto guard = acquire_guard(pass, flags);
			if (!guard)
				throw std::invalid_argument("Native program capture requires a live renderer epoch guard");
			auto program = prepare_owned_programs(state, [&](const auto& vertex, const auto& fragment)
												  { return metal4::rsx_texture_program_epoch::capture(state, executor.prepare_textures(vertex, fragment)); });
			return epoch{std::move(guard), std::move(program), framebuffer};
		};
		const auto submit = [&](const metal4::rsx_draw_packet& packet, const metal4::framebuffer_snapshot& target, unsigned, const epoch& prepared)
		{
			executor.execute(packet, target, prepared.program.textures());
		};
		return submit_owned_clause(state, layout, prepare, submit, metal4::provoking_vertex::last);
	}
};
