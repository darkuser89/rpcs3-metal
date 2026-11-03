#include "M4DrawExecutor.h"
#include "M4ExclusiveGuestMemory.h"
#include "M4DrawEnvironmentBinding.h"
#include "M4FramebufferBinding.h"
#include "M4PipelineStateBinding.h"
#include "M4ShaderProgramBinding.h"
#include "M4VertexDataBinding.h"
#include "M4VertexTextureBatch.h"

namespace metal4
{
struct versioned_draw_texture_job::state
{
	id<MTLDevice> device;
	std::thread::id thread = std::this_thread::get_id();
	vertex_texture_snapshot vertex;
	fragment_texture_snapshot fragment;
	null_texture_family nulls;
	std::array<std::optional<std::size_t>, 4> vertex_indices;
	std::array<std::optional<std::size_t>, 16> fragment_indices;
	std::optional<versioned_guest_texture_job> work;
	void check_thread() const
	{
		if (thread != std::this_thread::get_id()) throw std::logic_error("Native draw texture job belongs to another thread");
	}
};
versioned_draw_texture_job::versioned_draw_texture_job(std::shared_ptr<state> state) : m_state(std::move(state)) {}
versioned_draw_texture_job::~versioned_draw_texture_job() = default;
versioned_draw_texture_job::versioned_draw_texture_job(versioned_draw_texture_job&&) noexcept = default;
versioned_draw_texture_job& versioned_draw_texture_job::operator=(versioned_draw_texture_job&&) noexcept = default;
const vertex_texture_snapshot& versioned_draw_texture_job::vertex_words() const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	return m_state->vertex;
}
const fragment_texture_snapshot& versioned_draw_texture_job::fragment_words() const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	return m_state->fragment;
}
const texture_cache_state& versioned_draw_texture_job::metadata(bool vertex, unsigned unit) const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	const auto index = vertex ? m_state->vertex_indices.at(unit) : m_state->fragment_indices.at(unit);
	const auto referenced = vertex ? m_state->vertex.referenced_mask() : m_state->fragment.referenced_mask();
	if (!(referenced & (1u << unit))) throw std::invalid_argument("Native draw metadata unit was not referenced");
	return index ? m_state->work->metadata(*index) : m_state->nulls[static_cast<unsigned>(vertex_texture_dimension::two)].metadata;
}
bool versioned_draw_texture_job::ready() const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	return m_state->work->ready();
}
bool versioned_draw_texture_job::cancel_pending() const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	return m_state->work->cancel_pending();
}
bool versioned_draw_texture_job::current(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment) const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	return texture_words_match(vertex, fragment, m_state->vertex, m_state->fragment) && m_state->work->current();
}
versioned_draw_texture_epoch versioned_draw_texture_job::finish(std::chrono::milliseconds timeout)
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	const auto retained = m_state;
	retained->check_thread();
	versioned_draw_texture_epoch result;
	std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			auto images = retained->work->finish(timeout);
			result.revisions.reserve(images.size());
			std::size_t next = 0;
			result.textures = draw_texture_epoch::capture(retained->device, retained->vertex, retained->fragment, [&](bool, unsigned, const auto&)
			{
				auto image = std::move(images.at(next++));
				result.revisions.push_back(std::move(image.revision));
				return std::move(image.image);
			}, &retained->nulls);
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return result;
}

draw_executor::draw_executor(context& context, surface_cache& surfaces, guest_texture_cache& textures)
	: m_context(context), m_surfaces(surfaces), m_textures(textures), m_compiler(context), m_pipelines(m_compiler), m_shaders(m_compiler), m_samplers(context, 256, textures.heap_ledger()), m_native_depth_bounds_supported(supports_native_depth_bounds(context.device()))
{
	m_textures.validate_surface_adapter(context, surfaces);
}
void draw_executor::execute(const rsx_draw_packet& packet, const draw_targets& targets, const draw_execution_state& state)
{
	execute_owned(packet, targets, state, framebuffer_extent_policy::exact);
}
draw_texture_epoch draw_executor::prepare_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment)
{
	@autoreleasepool
	{
		const auto nulls = needs_null_textures(vertex, fragment) ? m_textures.null_images() : null_texture_family{};
		return draw_texture_epoch::capture(m_context, vertex, fragment, [&](bool, unsigned, const fragment_texture_registers& words)
										   { return m_textures.resolve(words, m_surfaces); }, &nulls);
	}
}
versioned_draw_texture_job draw_executor::enqueue_versioned_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment,
	std::span<const guest_texture_upload_dependency> dependencies)
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native draw texture preparation requires leaving VM exclusion");
	if (m_context.buffer_ledger() != m_textures.heap_ledger()) throw std::invalid_argument("Native draw texture jobs require one buffer/texture ledger");
	if (dependencies.size() > 64) throw std::length_error("Native draw texture job exceeds dependency bounds");
	for (const auto& dependency : dependencies)
		if (!dependency.event || (dependency.event.device && dependency.event.device != m_context.device()))
			throw std::invalid_argument("Native draw texture dependency belongs to another device");
	auto state = std::make_shared<versioned_draw_texture_job::state>();
	state->device = m_context.device(); state->vertex = vertex; state->fragment = fragment;
	std::vector<fragment_texture_registers> words;
	words.reserve(20);
	for (unsigned unit = 0; unit < 4; ++unit)
		if ((vertex.referenced_mask() & (1u << unit)) && (vertex.at(unit).control0 >> 31))
		{
			state->vertex_indices[unit] = words.size(); words.push_back(vertex.at(unit).upload_words());
		}
	for (unsigned unit = 0; unit < 16; ++unit)
		if ((fragment.referenced_mask() & (1u << unit)) && (fragment.at(unit).control0 >> 31))
		{
			state->fragment_indices[unit] = words.size(); words.push_back(fragment.at(unit));
		}
	m_surfaces.prepare_texture_boundary(needs_null_textures(vertex, fragment) && !m_textures.null_images_ready());
	std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			state->nulls = needs_null_textures(vertex, fragment) ? m_textures.null_images() : null_texture_family{};
			state->work.emplace(m_textures.prepare_versioned_batch(words, m_surfaces, dependencies));
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return versioned_draw_texture_job(std::move(state));
}
versioned_draw_texture_epoch draw_executor::prepare_versioned_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment)
{
	if (m_context.buffer_ledger())
	{
		auto job = enqueue_versioned_textures(vertex, fragment);
		return job.finish();
	}
	// Existing standalone contexts without an installed buffer ledger retain
	// their synchronous API. The real renderer always uses the native job.
	@autoreleasepool
	{
		versioned_draw_texture_epoch result;
		const auto nulls = needs_null_textures(vertex, fragment) ? m_textures.null_images() : null_texture_family{};
		result.revisions.reserve(20);
		result.textures = draw_texture_epoch::capture(m_context, vertex, fragment, [&](bool, unsigned, const fragment_texture_registers& words)
		{
			auto image = m_textures.resolve_versioned(words, m_surfaces);
			result.revisions.push_back(std::move(image.revision));
			return std::move(image.image);
		}, &nulls);
		return result;
	}
}
void draw_executor::execute(const rsx_draw_packet& packet, const draw_targets& targets, const draw_texture_epoch& textures, const draw_execution_state& state)
{
	execute_owned(packet, targets, state, framebuffer_extent_policy::exact, &textures);
}
void draw_executor::execute(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_texture_epoch& textures, const draw_execution_state& state)
{
	execute_framebuffer(packet, framebuffer, state, &textures);
}
void draw_executor::execute(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_execution_state& state)
{
	execute_framebuffer(packet, framebuffer, state, nullptr);
}
void draw_executor::execute_framebuffer(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_execution_state& state,
										const draw_texture_epoch* textures)
{
	@autoreleasepool
	{
		if (packet.geometry().vertices.empty())
			return;
		if (textures)
			textures->validate(m_context, packet.state().vertex_textures(), packet.state().textures());
		const auto validate = [&](const framebuffer_surface& surface)
		{
			packet.environment().validate_framebuffer_extent(surface.width, surface.height, framebuffer_extent_policy::clip_to_framebuffer);
		};
		for (const auto& color : framebuffer.colors())
			if (color)
				validate(*color);
		if (framebuffer.depth())
			validate(*framebuffer.depth());
		const auto targets = acquire_framebuffer(framebuffer, m_surfaces);
		execute_owned(packet, targets, state, framebuffer_extent_policy::clip_to_framebuffer, textures);
	}
}
void draw_executor::execute_owned(const rsx_draw_packet& packet, const draw_targets& targets, const draw_execution_state& state,
								  framebuffer_extent_policy extent_policy, const draw_texture_epoch* textures)
{
	@autoreleasepool
	{
		if (packet.geometry().vertices.empty())
			return;
		if (textures)
			textures->validate(m_context, packet.state().vertex_textures(), packet.state().textures());
		std::size_t width = 0, height = 0;
		std::uint32_t samples = 1;
		const auto validate_extent = [&](const cached_surface_ref& target)
		{
			if (!target)
				return;
			const auto allocation = m_surfaces.description(target).allocation;
			packet.environment().validate_framebuffer_extent(allocation.width, allocation.height, extent_policy);
			width = allocation.width;
			height = allocation.height;
			samples = allocation.samples;
		};
		for (const auto& color : targets.colors)
			validate_extent(color);
		validate_extent(targets.depth);
		if (!width)
			throw std::invalid_argument("Native owned pass has no framebuffer targets");
		draw_request request;
		request.geometry = packet.geometry();
		request.raster = state.raster;
		request.waits = state.waits;
		request.allow_output_rounding = state.allow_output_rounding;
		request.force_disable_blending = state.force_disable_blending;
		request.queries = state.queries;
		apply_draw_environment(packet.environment(), width, height, request, extent_policy, samples);
		m_surfaces.prepare_draw_resources([&] { apply_vertex_data_snapshot(m_context, packet.vertices(), request); });
		execute_draw(packet.state(), targets, request, textures);
	}
}
void draw_executor::execute(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request)
{
	execute_draw(snapshot, targets, request, nullptr);
}
void draw_executor::execute(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request, const draw_texture_epoch& textures)
{
	execute_draw(snapshot, targets, request, &textures);
}
void draw_executor::execute_draw(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request,
								 const draw_texture_epoch* textures)
{
	@autoreleasepool
	{
		if (request.geometry.vertices.empty())
			return;
		if (textures)
			textures->validate(m_context, snapshot.vertex_textures(), snapshot.textures());
		std::vector<cached_surface_use> uses;
		std::array<std::optional<surface_color>, 4> formats{};
		std::array<std::size_t, 4> indices{};
		std::size_t depth_index = 0, width = 0, height = 0;
		std::uint32_t samples = 0;
		std::optional<surface_depth> depth_format;
		const auto add = [&](const cached_surface_ref& surface)
		{
			const auto description = m_surfaces.description(surface);
			const auto& allocation = description.allocation;
			if (width && (allocation.width != width || allocation.height != height || allocation.samples != samples))
				throw std::invalid_argument("Native draw targets disagree in dimensions/sample count");
			width = allocation.width;
			height = allocation.height;
			samples = allocation.samples;
			uses.push_back({surface, true});
			return allocation;
		};
		for (unsigned slot = 0; slot < 4; ++slot)
			if (targets.colors[slot])
			{
				indices[slot] = uses.size();
				const auto allocation = add(targets.colors[slot]);
				if (!std::holds_alternative<surface_color>(allocation.format))
					throw std::invalid_argument("Native color slot contains a depth surface");
				formats[slot] = std::get<surface_color>(allocation.format);
			}
		if (targets.depth)
		{
			depth_index = uses.size();
			const auto allocation = add(targets.depth);
			if (!std::holds_alternative<surface_depth>(allocation.format))
				throw std::invalid_argument("Native depth slot contains a color surface");
			depth_format = std::get<surface_depth>(allocation.format);
		}
		if (uses.empty())
			throw std::invalid_argument("Native draw has no framebuffer targets");
		render_pipeline_configuration config;
		config.state.sample_count = samples;
		config.state.alpha_to_coverage = request.alpha_to_coverage;
		config.state.alpha_to_one = request.alpha_to_one;
		vertex_program_options vertex_options;
		fragment_program_options fragment_options;
		auto vertex_uniforms = request.vertex_uniforms;
		auto fragment_uniforms = request.fragment_uniforms;
		auto raster = request.raster;
		const bool points = request.geometry.topology == list_topology::points;
		config.topology = points ? MTLPrimitiveTopologyClassPoint : request.geometry.topology == list_topology::lines ? MTLPrimitiveTopologyClassLine
																													  : MTLPrimitiveTopologyClassTriangle;
		apply_shader_program_snapshots(snapshot.vertex(), snapshot.fragment(), points, request.allow_output_rounding, config, vertex_options, fragment_options, vertex_uniforms, fragment_uniforms);
		vertex_options.packed_inputs = true;
		vertex_options.flat_result_inputs = vertex_options.flat_colors;
		const auto geometry_plan = plan_indexed_references(request.geometry,vertex_options.flat_colors);
		apply_pipeline_register_snapshot(snapshot.pipeline(), formats, targets.register_color_format, width, height, config, fragment_options, raster, fragment_uniforms, request.force_disable_blending);
		const auto depth_bounds = plan_depth_bounds(raster.depth_bounds_enabled,raster.depth_min,raster.depth_max,m_native_depth_bounds_supported);
		if (depth_bounds != depth_bounds_path::disabled)
		{
			if (!depth_format) throw std::invalid_argument("Native stored-depth bounds require a depth target");
			if (depth_bounds == depth_bounds_path::mirror)
				configure_depth_bounds_pipeline(config, fragment_options, describe_surface_depth(*depth_format).format);
			else
				fragment_uniforms.depth_bounds_enabled = 0; // The native raster selector owns this test.
		}
		// Validate the complete compiled texture-reference contract before resolving
		// any VM snapshots or changing writable target state.
		const auto fragment_program = m_shaders.prepare_fragment(snapshot.fragment().code(), fragment_options);
		const auto mask = fragment_program.texture_mask();
		if (mask != snapshot.textures().referenced_mask())
			throw std::invalid_argument("Native draw snapshot texture mask disagrees with its shader");
		const auto vertex_program = m_shaders.prepare_vertex(snapshot.vertex().code(), vertex_options);
		const auto vertex_mask = vertex_program.texture_mask();
		if (vertex_mask != snapshot.vertex_textures().referenced_mask())
			throw std::invalid_argument("Native draw snapshot vertex texture mask disagrees with its shader");
		auto vertex_resources = request.vertex_resources;
		const auto resolve_vertex = [&](unsigned unit, const fragment_texture_registers& words)
		{
			if (textures)
				return textures->vertex(unit);
			return m_textures.resolve(words, m_surfaces);
		};
		const auto resolve_fragment = [&](unsigned unit, const fragment_texture_registers& words)
		{
			if (textures)
				return textures->fragment(unit);
			return m_textures.resolve(words, m_surfaces);
		};
		const auto nulls = textures ? textures->null_images() : needs_null_textures(snapshot.vertex_textures(), snapshot.textures()) ? m_textures.null_images()
																																	 : null_texture_family{};
		apply_vertex_texture_snapshot(snapshot.vertex_textures(), resolve_vertex, m_context, m_samplers, vertex_options, vertex_resources, &nulls);
		fragment_inputs fragment_resources;
		apply_fragment_texture_snapshot(snapshot.textures(), resolve_fragment, m_context, m_samplers, fragment_options, fragment_resources, {}, &nulls);
		// Final texture/ROP options are known. Both native stage jobs may run
		// while vertex storage is admitted, without borrowing mutable registers.
		const auto warm_vertex = m_shaders.prewarm_vertex(vertex_program, vertex_options);
		const auto warm_fragment = m_shaders.prewarm_fragment(fragment_program, fragment_options);
		validate_vertex_fetch(m_context,request.fetch,request.persistent,request.transient);
		vertex_resources.attributes = nil;
		vertex_resources.attribute_offset = 0;
		vertex_resources.packed = packed_vertex_inputs{request.fetch,request.persistent,request.transient,request.source_vertices};
		const auto vertex = m_shaders.vertex(warm_vertex, vertex_options);
		const auto fragment = m_shaders.fragment(warm_fragment, fragment_options);
		// Native PSO compilation overlaps immutable constants/arena admission.
		const auto warm_pipeline = m_pipelines.prewarm(config, vertex->function(), *fragment);
		std::optional<fragment_shader::constant_binding> constants;
		std::optional<upload_plan> uploads;
		std::optional<prepared_geometry> geometry;
		std::optional<prepared_flat_colors> flat_colors;
		m_surfaces.prepare_draw_resources([&]
		{
			auto candidate_geometry = prepared_geometry::prepare(m_context,geometry_plan);
			std::optional<prepared_flat_colors> candidate_flat;
			if (vertex_options.flat_result_inputs) candidate_flat.emplace(prepared_flat_colors::admit(m_context,candidate_geometry.references()));
			auto candidate_constants = fragment->prepare_constants(m_context, snapshot.fragment().code());
			auto requirements = vertex->uploads(vertex_resources,candidate_flat ? &*candidate_flat : nullptr);
			const auto fragment_uploads = fragment->uploads();
			requirements.insert(requirements.end(), fragment_uploads.begin(), fragment_uploads.end());
			auto candidate_uploads = m_context.prepare_uploads(requirements);
			constants = std::move(candidate_constants);
			uploads = std::move(candidate_uploads);
			geometry = std::move(candidate_geometry);
			flat_colors = std::move(candidate_flat);
		});
		const auto pipeline = warm_pipeline ? m_pipelines.get(*warm_pipeline) : m_pipelines.get(config, vertex->function(), *fragment);
		const auto descriptor = [&](std::span<const id<MTLTexture>> textures)
		{
			auto pass = [MTL4RenderPassDescriptor new];
			for (unsigned slot = 0; slot < 4; ++slot)
				if (targets.colors[slot])
				{
					pass.colorAttachments[slot].texture = textures[indices[slot]];
					pass.colorAttachments[slot].loadAction = MTLLoadActionLoad;
					pass.colorAttachments[slot].storeAction = MTLStoreActionStore;
				}
			if (targets.depth)
			{
				const auto texture = textures[depth_index];
				pass.depthAttachment.texture = texture;
				pass.depthAttachment.loadAction = MTLLoadActionLoad;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
				if (texture.pixelFormat == MTLPixelFormatDepth32Float_Stencil8)
				{
					pass.stencilAttachment.texture = texture;
					pass.stencilAttachment.loadAction = MTLLoadActionLoad;
					pass.stencilAttachment.storeAction = MTLStoreActionStore;
				}
			}
			return pass;
		};
		const auto prepare_vertex = [&](context& context)
		{
			auto input = vertex_resources;
			vertex_uniforms.vertex_base = request.fetch.first_vertex;
			vertex_uniforms.vertex_count = request.fetch.vertex_count;
			return vertex->prepare(context, vertex_uniforms, geometry->references(), input,flat_colors ? &*flat_colors : nullptr);
		};
		const auto wait = [&](context& context)
		{
			for (const auto& dependency : request.waits) context.wait_event(dependency.event, dependency.value);
		};
		if (depth_bounds == depth_bounds_path::mirror)
		{
			std::optional<visibility_binding> visibility;
			m_surfaces.draw_depth_bounds(uses, depth_index, [&](auto textures)
			{
				wait(m_context);
				auto pass = descriptor(textures);
				visibility = request.queries ? request.queries->attach(m_context, pass) : std::nullopt;
				return pass;
			}, [&](context& context, auto, depth_bounds_pass& prepared)
			{
				context.with_uploads(*uploads, [&]
				{
					const auto vb = prepare_vertex(context);
					if (visibility) prepared.visibility(visibility->mode, visibility->offset);
					prepared.draw_geometry(*geometry, pipeline, *fragment, vb, fragment_uniforms, snapshot.fragment().code(), fragment_resources, raster, 1, 0, &*constants);
				});
			});
		}
		else
		{
			m_surfaces.draw(uses, [&](context& context, auto textures)
			{
				context.with_uploads(*uploads, [&]
				{
					wait(context);
					const auto vb = prepare_vertex(context);
					const auto fb = fragment->prepare(context, fragment_uniforms, *constants, fragment_resources);
					auto pass = descriptor(textures);
					const auto visibility = request.queries ? request.queries->attach(context, pass) : std::nullopt;
					render_commands commands(context, pass);
					if (visibility) commands.visibility(visibility->mode, visibility->offset);
					commands.draw_geometry(*geometry, pipeline.pipeline, pipeline.depth_stencil, raster, vb, fb);
					commands.end();
				});
			});
		}
	}
}
} // namespace metal4
