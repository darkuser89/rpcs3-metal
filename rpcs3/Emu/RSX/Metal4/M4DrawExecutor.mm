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
	std::weak_ptr<const std::byte> queue;
	std::thread::id thread = std::this_thread::get_id();
	vertex_texture_snapshot vertex;
	fragment_texture_snapshot fragment;
	null_texture_family nulls;
	std::array<std::optional<std::size_t>, 4> vertex_indices;
	std::array<std::optional<std::size_t>, 16> fragment_indices;
	std::optional<versioned_guest_texture_job> work;
	[[nodiscard]] draw_texture_epoch bind(const std::vector<versioned_guest_texture>& images) const
	{
		std::size_t next = 0;
		return draw_texture_epoch::capture(device, vertex, fragment, [&](bool, unsigned, const auto&)
		{
			return images.at(next++).image;
		}, &nulls);
	}
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
bool versioned_draw_texture_job::has_completion() const
{
	if (!m_state) throw std::logic_error("Native draw texture job was moved");
	m_state->check_thread();
	return m_state->work->has_completion();
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
			result.textures = retained->bind(images);
			for (auto& image : images) result.revisions.push_back(std::move(image.revision));
		}
		catch (...) { failure = std::current_exception(); }
	}
	if (failure) std::rethrow_exception(failure);
	return result;
}

draw_executor::draw_executor(context& context, surface_cache& surfaces, guest_texture_cache& textures)
	: draw_executor(context,surfaces,textures,nil)
{
}
draw_executor::draw_executor(context& context, surface_cache& surfaces, guest_texture_cache& textures, id<MTL4Compiler> compiler)
	: m_context(context), m_surfaces(surfaces), m_textures(textures), m_compiler(context,256,16u<<20,compiler), m_pipelines(m_compiler), m_shaders(m_compiler), m_samplers(context, 256, textures.heap_ledger()), m_native_depth_bounds_supported(supports_native_depth_bounds(context.device()))
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
draw_publication draw_executor::publication_for(const report_predicate* predicate) const
{
	if (!predicate) return {};
	if (!predicate->belongs_to(m_context))
		throw std::invalid_argument("Native draw predicate belongs to another queue");
	return [snapshot = *predicate](std::chrono::milliseconds timeout) { return snapshot.permits_publication(timeout); };
}
versioned_draw_texture_job draw_executor::enqueue_versioned_textures(const vertex_texture_snapshot& vertex, const fragment_texture_snapshot& fragment,
	std::span<const guest_texture_upload_dependency> dependencies, const report_predicate* predicate, texture_upload_path path)
{
	if (guest_memory_epoch::active()) throw std::logic_error("Native draw texture preparation requires leaving VM exclusion");
	if (m_context.buffer_ledger() != m_textures.heap_ledger()) throw std::invalid_argument("Native draw texture jobs require one buffer/texture ledger");
	if (dependencies.size() > 64) throw std::length_error("Native draw texture job exceeds dependency bounds");
	for (const auto& dependency : dependencies)
		if (!dependency.event || (dependency.event.device && dependency.event.device != m_context.device()))
			throw std::invalid_argument("Native draw texture dependency belongs to another device");
	const auto permit = publication_for(predicate);
	auto state = std::make_shared<versioned_draw_texture_job::state>();
	state->device = m_context.device(); state->queue = m_context.queue_identity();
	state->vertex = vertex; state->fragment = fragment;
	std::vector<fragment_texture_registers> words;
	words.reserve(20);
	for (unsigned unit = 0; unit < 4; ++unit)
		if ((vertex.referenced_mask() & (1u << unit)) && (vertex.at(unit).control0 >> 31))
		{
			state->vertex_indices[unit] = words.size(); words.push_back(vertex.at(unit).upload_words());
		}
	std::uint64_t fragment_shader_reads = 0;
	for (unsigned unit = 0; unit < 16; ++unit)
		if ((fragment.referenced_mask() & (1u << unit)) && (fragment.at(unit).control0 >> 31))
		{
			fragment_shader_reads |= std::uint64_t{1} << words.size();
			state->fragment_indices[unit] = words.size(); words.push_back(fragment.at(unit));
		}
	m_surfaces.prepare_texture_boundary(false);
	std::exception_ptr failure;
	@autoreleasepool
	{
		try
		{
			state->nulls = needs_null_textures(vertex, fragment) ? m_textures.null_images() : null_texture_family{};
			state->work.emplace(m_textures.prepare_versioned_batch(words, m_surfaces, dependencies, permit, path, fragment_shader_reads));
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
namespace
{
struct draw_program_configuration
{
	render_pipeline_configuration pipeline;
	vertex_program_options vertex_options;
	fragment_program_options fragment_options;
	vertex_context vertex_uniforms;
	fragment_context fragment_uniforms;
	raster_state raster;
	depth_bounds_path depth_bounds;
};

draw_program_configuration configure_draw_programs(const rsx_draw_snapshot& snapshot, const draw_request& request,
	const std::array<std::optional<surface_color>, 4>& formats, std::optional<surface_depth> depth_format,
	surface_color register_color_format, std::size_t width, std::size_t height, std::uint32_t samples,
	bool native_depth_bounds_supported, bool predicated)
{
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
	// A suppressed draw must not dispatch the flat-color evaluator. The
	// generated vertex stage can evaluate flat sources on demand.
	vertex_options.flat_result_inputs = vertex_options.flat_colors && !predicated;
	apply_pipeline_register_snapshot(snapshot.pipeline(), formats, register_color_format, width, height, config, fragment_options, raster, fragment_uniforms, request.force_disable_blending);
	const auto depth_bounds = plan_depth_bounds(raster.depth_bounds_enabled,raster.depth_min,raster.depth_max,native_depth_bounds_supported);
	if (depth_bounds != depth_bounds_path::disabled)
	{
		if (!depth_format) throw std::invalid_argument("Native stored-depth bounds require a depth target");
		if (depth_bounds == depth_bounds_path::mirror)
			configure_depth_bounds_pipeline(config, fragment_options, describe_surface_depth(*depth_format).format);
		else
			fragment_uniforms.depth_bounds_enabled = 0; // The native raster selector owns this test.
	}
	return {std::move(config), std::move(vertex_options), std::move(fragment_options), vertex_uniforms, fragment_uniforms, raster, depth_bounds};
}
}

std::optional<render_pipeline_cache::prepared_pipeline> draw_executor::prewarm_stages(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer,
	const versioned_draw_texture_job& textures, const draw_execution_state& state)
{
	if (guest_memory_epoch::active())
		throw std::logic_error("Native shader prewarm requires leaving VM exclusion");
	if (packet.geometry().vertices.empty()) return {};
	const auto& snapshot = packet.state();
	if (!textures.m_state || textures.m_state->device != m_context.device() ||
		!texture_words_match(snapshot.vertex_textures(), snapshot.textures(), textures.vertex_words(), textures.fragment_words()))
		throw std::invalid_argument("Native shader prewarm requires the captured draw texture metadata");

	std::array<std::optional<surface_color>, 4> formats;
	std::optional<surface_depth> depth;
	const framebuffer_surface* extent = nullptr;
	for (unsigned slot = 0; slot < formats.size(); ++slot)
		if (const auto& surface = framebuffer.colors()[slot])
		{
			formats[slot] = std::get<surface_color>(surface->format);
			extent = &*surface;
		}
	if (framebuffer.depth())
	{
		extent = &*framebuffer.depth();
		depth = std::get<surface_depth>(extent->format);
	}
	if (!extent) throw std::invalid_argument("Native shader prewarm has no framebuffer targets");
	draw_request request;
	request.geometry.topology = packet.geometry().topology;
	request.raster = state.raster;
	request.allow_output_rounding = state.allow_output_rounding;
	request.force_disable_blending = state.force_disable_blending;
	request.predicate = state.predicate;
	apply_draw_environment(packet.environment(), extent->width, extent->height, request, framebuffer_extent_policy::clip_to_framebuffer, extent->samples);
	auto configuration = configure_draw_programs(snapshot, request, formats, depth, framebuffer.color_format(),
		extent->width, extent->height, extent->samples, m_native_depth_bounds_supported,
		request.predicate && request.predicate->settled_cpu_result(m_context) != true);
	auto& vertex = configuration.vertex_options;
	auto& fragment = configuration.fragment_options;
	for (unsigned unit = 0; unit < vertex.textures.size(); ++unit)
		if (snapshot.vertex_textures().referenced_mask() & (1u << unit))
		{
			const auto& words = snapshot.vertex_textures().at(unit);
			const auto texture = words.control0 >> 31 ? decode_vertex_texture_state(words, textures.metadata(true, unit)) :
				null_texture_state(vertex.textures[unit], sampler_stage::vertex);
			configure_vertex_texture_state(vertex, unit, texture);
		}
	for (unsigned unit = 0; unit < fragment.textures.size(); ++unit)
		if (snapshot.textures().referenced_mask() & (1u << unit))
		{
			const auto& words = snapshot.textures().at(unit);
			const auto texture = words.control0 >> 31 ? checked_fragment_texture_state(words, textures.metadata(false, unit)) :
				null_texture_state(fragment.textures[unit], sampler_stage::fragment);
			configure_fragment_texture_state(fragment, unit, texture);
		}
	const auto vp = m_shaders.prepare_vertex(snapshot.vertex().code(), vertex);
	const auto fp = m_shaders.prepare_fragment(snapshot.fragment().code(), fragment);
	if (vp.texture_mask() != snapshot.vertex_textures().referenced_mask() || fp.texture_mask() != snapshot.textures().referenced_mask())
		throw std::invalid_argument("Native shader prewarm texture mask disagrees with the captured draw");
	const auto warm_vertex = m_shaders.prewarm_vertex(vp, vertex);
	const auto warm_fragment = m_shaders.prewarm_fragment(fp, fragment);
	if ((!m_shaders.ready(warm_vertex) || !m_shaders.ready(warm_fragment)))
		if (const auto stages = m_shaders.pipeline_stages(warm_vertex,warm_fragment))
			if (auto pipeline = m_pipelines.prewarm(configuration.pipeline,*stages)) return pipeline;
	if (!m_shaders.ready(warm_vertex)) return {};
	const bool fragment_ready = m_shaders.ready(warm_fragment);
	if (!fragment_ready && !vertex.flat_result_inputs) return {};
	const auto vertex_shader = m_shaders.vertex(warm_vertex, vertex);
	if (!fragment_ready)
	{
		vertex_shader->prewarm_flat_pipeline();
		return {};
	}
	const auto fragment_shader = m_shaders.fragment(warm_fragment, fragment);
	auto pipeline = m_pipelines.prewarm(configuration.pipeline, vertex_shader->function(), *fragment_shader);
	vertex_shader->prewarm_flat_pipeline();
	return pipeline;
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
void draw_executor::execute(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const versioned_draw_texture_job& textures, const draw_execution_state& state)
{
	if (!textures.m_state || textures.m_state->queue.lock() != m_context.queue_identity().lock())
		throw std::invalid_argument("Native draw texture job belongs to another queue");
	if (packet.geometry().vertices.empty()) return;
	const auto [images, predecessor] = textures.m_state->work->draw_images(m_surfaces);
	const auto epoch = textures.m_state->bind(images);
	execute_framebuffer(packet, framebuffer, state, &epoch, predecessor ? &*predecessor : nullptr, &*textures.m_state->work);
}
void draw_executor::execute_framebuffer(const rsx_draw_packet& packet, const framebuffer_snapshot& framebuffer, const draw_execution_state& state,
	const draw_texture_epoch* textures, const submission* predecessor, const versioned_guest_texture_job* draw_uploads)
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
		execute_owned(packet, targets, state, framebuffer_extent_policy::clip_to_framebuffer, textures, predecessor, draw_uploads);
	}
}
void draw_executor::execute_owned(const rsx_draw_packet& packet, const draw_targets& targets, const draw_execution_state& state,
								  framebuffer_extent_policy extent_policy, const draw_texture_epoch* textures, const submission* predecessor,
	const versioned_guest_texture_job* draw_uploads)
{
	@autoreleasepool
	{
		if (packet.geometry().vertices.empty())
			return;
		if (textures)
			textures->validate(m_context, packet.state().vertex_textures(), packet.state().textures());
		if (state.predicate && state.queries && !predecessor && !draw_uploads && state.waits.empty() &&
			state.predicate->settled_cpu_result(m_context) == false)
			return;
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
		request.geometry.topology = packet.geometry().topology;
		request.raster = state.raster;
		request.waits = state.waits;
		request.allow_output_rounding = state.allow_output_rounding;
		request.force_disable_blending = state.force_disable_blending;
		request.queries = state.queries;
		request.predicate = state.predicate;
		apply_draw_environment(packet.environment(), width, height, request, extent_policy, samples);
		request.vertex_uniforms.constant_count = static_cast<std::uint32_t>(packet.vertices().constants().size());
		execute_draw(packet.state(), targets, request, packet.geometry(), textures, predecessor, draw_uploads, &packet.vertices());
	}
}
void draw_executor::execute(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request)
{
	execute_draw(snapshot, targets, request, request.geometry, nullptr);
}
void draw_executor::execute(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request, const draw_texture_epoch& textures)
{
	execute_draw(snapshot, targets, request, request.geometry, &textures);
}
void draw_executor::execute_draw(const rsx_draw_snapshot& snapshot, const draw_targets& targets, const draw_request& request,
	const converted_geometry& references, const draw_texture_epoch* textures, const submission* predecessor, const versioned_guest_texture_job* draw_uploads,
	const vertex_data_snapshot* vertices)
{
	@autoreleasepool
	{
		if (references.vertices.empty())
			return;
		if (request.predicate && !request.queries)
			throw std::invalid_argument("Native predicated draw requires its query pool");
		const auto* predicate = request.predicate && request.predicate->settled_cpu_result(m_context) != true ? &*request.predicate : nullptr;
		const auto publish = publication_for(predicate);
		if (!predecessor && predicate && predicate->m_producer)
			predecessor = &*predicate->m_producer;
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
		auto [config, vertex_options, fragment_options, vertex_uniforms, fragment_uniforms, raster, depth_bounds] =
			configure_draw_programs(snapshot, request, formats, depth_format, targets.register_color_format, width, height, samples,
				m_native_depth_bounds_supported, predicate != nullptr);
		auto geometry_plan = m_recent_geometry;
		if (!geometry_plan || !geometry_plan->matches(references, vertex_options.flat_colors))
		{
			geometry_plan = std::make_shared<geometry_index_plan>(plan_indexed_references(references, vertex_options.flat_colors));
			++m_geometry_plans_created;
			// Consecutive passes often share geometry. Keep at most one small
			// CPU plan; oversized plans live only for their current draw.
			constexpr std::size_t retained_geometry_limit = 1u << 20;
			if (geometry_plan->storage_bytes() <= retained_geometry_limit)
				m_recent_geometry = geometry_plan;
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
			return m_textures.resolve(words, m_surfaces, publish);
		};
		const auto resolve_fragment = [&](unsigned unit, const fragment_texture_registers& words)
		{
			if (textures)
				return textures->fragment(unit);
			return m_textures.resolve(words, m_surfaces, publish);
		};
		const auto nulls = textures ? textures->null_images() : needs_null_textures(snapshot.vertex_textures(), snapshot.textures()) ? m_textures.null_images()
																																	 : null_texture_family{};
		fragment_inputs fragment_resources;
		border_texture_preparation border_helpers;
		try
		{
			apply_vertex_texture_snapshot(snapshot.vertex_textures(), resolve_vertex, m_context, m_samplers, vertex_options, vertex_resources, &nulls, &border_helpers);
			apply_fragment_texture_snapshot(snapshot.textures(), resolve_fragment, m_context, m_samplers, fragment_options, fragment_resources, {}, &nulls, &border_helpers);
		}
		catch (const predicated_texture_suppressed&)
		{
			return;
		}
		// Final texture/ROP options are known. Both native stage jobs may run
		// while vertex storage is admitted, without borrowing mutable registers.
		const auto warm_vertex = m_shaders.prewarm_vertex(vertex_program, vertex_options);
		const auto warm_fragment = m_shaders.prewarm_fragment(fragment_program, fragment_options);
		vertex_resources.attributes = nil;
		vertex_resources.attribute_offset = 0;
		std::shared_ptr<const vertex_shader> vertex;
		std::shared_ptr<const fragment_shader> fragment;
		std::optional<render_pipeline_cache::prepared_pipeline> warm_pipeline;
		if (!m_shaders.ready(warm_vertex) || !m_shaders.ready(warm_fragment))
			if (const auto stages = m_shaders.pipeline_stages(warm_vertex,warm_fragment))
				warm_pipeline = m_pipelines.prewarm(config,*stages);
		std::optional<fragment_shader::constant_binding> constants;
		std::optional<upload_plan> uploads;
		std::optional<prepared_geometry> geometry;
		std::optional<prepared_flat_colors> flat_colors;
		const auto start_pipeline = [&]
		{
			if (!vertex) vertex = m_shaders.vertex(warm_vertex, vertex_options);
			if (!m_shaders.ready(warm_fragment)) vertex->prewarm_flat_pipeline();
			auto candidate_fragment = m_shaders.fragment(warm_fragment, fragment_options);
			if (!warm_pipeline) warm_pipeline = m_pipelines.prewarm(config, vertex->function(), *candidate_fragment);
			vertex->prewarm_flat_pipeline();
			fragment = std::move(candidate_fragment);
		};
		const auto start_ready_pipeline = [&]
		{
			if (fragment || (!vertex && !m_shaders.ready(warm_vertex)))
				return;
			if (m_shaders.ready(warm_fragment))
				start_pipeline();
			else if (!vertex && vertex_options.flat_result_inputs)
			{
				vertex = m_shaders.vertex(warm_vertex, vertex_options);
				vertex->prewarm_flat_pipeline();
			}
		};
		// Ready stages can start a cold PSO before allocating draw storage.
		// Pending stages are polled again after independent work.
		start_ready_pipeline();
		m_surfaces.prepare_draw_resources([&]
		{
			auto candidate_vertex_resources = vertex_resources;
			if (vertices)
			{
				auto vertex_data = upload_vertex_data(m_context, *vertices);
				candidate_vertex_resources.packed = std::move(vertex_data.packed);
				candidate_vertex_resources.constants = vertex_data.constants;
				candidate_vertex_resources.constant_offset = 0;
			}
			else
				candidate_vertex_resources.packed = packed_vertex_inputs{request.fetch, request.persistent, request.transient, request.source_vertices};
			const auto& packed = *candidate_vertex_resources.packed;
			validate_vertex_fetch(m_context, packed.request, packed.persistent, packed.transient);
			auto candidate_geometry = prepared_geometry::prepare(m_context,*geometry_plan);
			std::optional<prepared_flat_colors> candidate_flat;
			if (vertex_options.flat_result_inputs) candidate_flat.emplace(prepared_flat_colors::admit(m_context,candidate_geometry.references()));
			start_ready_pipeline();
			auto candidate_constants = m_shaders.prepare_constants(m_context, warm_fragment, snapshot.fragment().code());
			auto requirements = vertex_shader::plan_uploads(candidate_vertex_resources, vertex_options.flat_result_inputs,
				candidate_flat ? &*candidate_flat : nullptr);
			if (predicate) requirements.insert(requirements.begin(), query_pool::predicated_draw_upload());
			const auto fragment_uploads = fragment_shader::plan_uploads(warm_fragment.texture_mask());
			requirements.insert(requirements.end(), fragment_uploads.begin(), fragment_uploads.end());
			auto candidate_uploads = m_context.prepare_uploads(requirements);
			vertex_resources = std::move(candidate_vertex_resources);
			constants = std::move(candidate_constants);
			uploads = std::move(candidate_uploads);
			geometry = std::move(candidate_geometry);
			flat_colors = std::move(candidate_flat);
		});
		if (!fragment) start_pipeline();
		const auto pipeline = warm_pipeline ? m_pipelines.get(*warm_pipeline) : m_pipelines.get(config, vertex->function(), *fragment);
		if (flat_colors && !flat_colors->source_ids().empty() && vertex_uniforms.predicate)
			(void)vertex->flat_pipeline();
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
			vertex_uniforms.vertex_base = vertex_resources.packed->request.first_vertex;
			vertex_uniforms.vertex_count = vertex_resources.packed->request.vertex_count;
			return vertex->prepare(context, vertex_uniforms, geometry->references(), vertex_resources,flat_colors ? &*flat_colors : nullptr);
		};
		const auto wait = [&](context& context)
		{
			for (const auto& dependency : request.waits) context.wait_event(dependency.event, dependency.value);
		};
		const auto prepare_conditional = [&]() -> std::optional<conditional_draw>
		{
			if (!predicate) return {};
			return request.queries->encode_predicated_draw(*predicate, *geometry);
		};
		if (depth_bounds == depth_bounds_path::mirror)
		{
			std::optional<visibility_binding> visibility;
			m_surfaces.draw_depth_bounds(uses, depth_index, [&](auto textures)
			{
				m_context.end_guest_render_pass();
				wait(m_context);
				if (draw_uploads) draw_uploads->record_draw_uploads(m_context);
				border_helpers.encode(m_context);
				auto pass = descriptor(textures);
				visibility = request.queries ? request.queries->attach(m_context, pass) : std::nullopt;
				return pass;
			}, [&](context& context, auto, depth_bounds_pass& prepared)
			{
				context.with_uploads(*uploads, [&]
				{
					const auto conditional = prepare_conditional();
					const auto vb = prepare_vertex(context);
					if (visibility) prepared.visibility(visibility->mode, visibility->offset);
					prepared.draw_guest_geometry(*geometry, pipeline, *fragment, vb, fragment_uniforms, snapshot.fragment().code(), fragment_resources, raster, &*constants,
						conditional ? &*conditional : nullptr);
				});
			}, false, publish, predecessor);
		}
		else
		{
			const bool waits_reached = std::all_of(request.waits.begin(), request.waits.end(), [](const auto& dependency)
			{ return shared_event_reached(dependency.event, dependency.value); });
			const bool retain_guest_pass = vertices && (textures || !(vertex_mask | mask)) && !predicate && !flat_colors &&
				!fragment_options.logic_rop.enabled && !fragment_options.blend_rop.enabled() && waits_reached && !predecessor;
			m_surfaces.draw(uses, [&](context& context, auto textures)
			{
				if (!retain_guest_pass) context.end_guest_render_pass();
				if (draw_uploads) draw_uploads->record_draw_uploads(context);
				border_helpers.encode(context);
				context.with_uploads(*uploads, [&]
				{
					wait(context);
					const auto conditional = prepare_conditional();
					const auto vb = prepare_vertex(context);
					const auto fb = fragment->prepare(context, fragment_uniforms, *constants, fragment_resources);
					auto pass = descriptor(textures);
					const auto visibility = request.queries ? request.queries->attach(context, pass) : std::nullopt;
					render_commands::draw_prepared_guest(context, pass, *geometry, pipeline.pipeline, pipeline.depth_stencil,
						raster, vb, fb, visibility ? visibility->mode : MTLVisibilityResultModeDisabled, visibility ? visibility->offset : 0,
						conditional ? &*conditional : nullptr, retain_guest_pass);
				});
			}, publish, predecessor);
		}
	}
}
} // namespace metal4
