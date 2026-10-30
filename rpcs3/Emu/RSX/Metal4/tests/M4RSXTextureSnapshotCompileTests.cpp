#include "Emu/RSX/GSRender.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/rsx_methods.h"
#include "M4PipelineSnapshot.h"
#include "M4RSXDrawFrontend.h"
#include "M4RSXDrawSnapshot.h"
#include "M4ShaderProgramSnapshot.h"
#include "M4TextureSnapshot.h"
#include "M4RSXSamplerDescriptor.h"
#include "stdafx.h"

static_assert(metal4::fragment_texture_method_base == NV4097_SET_TEXTURE_OFFSET);
static_assert(metal4::fragment_texture_control3_base == NV4097_SET_TEXTURE_CONTROL3);
static_assert(NV4097_SET_TEXTURE_FORMAT - NV4097_SET_TEXTURE_OFFSET == 1);
static_assert(NV4097_SET_TEXTURE_ADDRESS - NV4097_SET_TEXTURE_OFFSET == 2);
static_assert(NV4097_SET_TEXTURE_CONTROL0 - NV4097_SET_TEXTURE_OFFSET == 3);
static_assert(NV4097_SET_TEXTURE_CONTROL1 - NV4097_SET_TEXTURE_OFFSET == 4);
static_assert(NV4097_SET_TEXTURE_FILTER - NV4097_SET_TEXTURE_OFFSET == 5);
static_assert(NV4097_SET_TEXTURE_IMAGE_RECT - NV4097_SET_TEXTURE_OFFSET == 6);
static_assert(NV4097_SET_TEXTURE_BORDER_COLOR - NV4097_SET_TEXTURE_OFFSET == 7);
static_assert(metal4::vertex_texture_method_base == NV4097_SET_VERTEX_TEXTURE_OFFSET);
static_assert(NV4097_SET_VERTEX_TEXTURE_FORMAT - NV4097_SET_VERTEX_TEXTURE_OFFSET == 1);
static_assert(NV4097_SET_VERTEX_TEXTURE_ADDRESS - NV4097_SET_VERTEX_TEXTURE_OFFSET == 2);
static_assert(NV4097_SET_VERTEX_TEXTURE_CONTROL0 - NV4097_SET_VERTEX_TEXTURE_OFFSET == 3);
static_assert(NV4097_SET_VERTEX_TEXTURE_CONTROL3 - NV4097_SET_VERTEX_TEXTURE_OFFSET == 4);
static_assert(NV4097_SET_VERTEX_TEXTURE_FILTER - NV4097_SET_VERTEX_TEXTURE_OFFSET == 5);
static_assert(NV4097_SET_VERTEX_TEXTURE_IMAGE_RECT - NV4097_SET_VERTEX_TEXTURE_OFFSET == 6);
static_assert(NV4097_SET_VERTEX_TEXTURE_BORDER_COLOR - NV4097_SET_VERTEX_TEXTURE_OFFSET == 7);
static_assert(static_cast<unsigned>(rsx::draw_command::array) == 1);
static_assert(static_cast<unsigned>(rsx::draw_command::inlined_array) == 2);
static_assert(static_cast<unsigned>(rsx::draw_command::indexed) == 3);
static_assert(static_cast<unsigned>(rsx::index_array_type::u32) == 0);
static_assert(static_cast<unsigned>(rsx::index_array_type::u16) == 1);
static_assert(static_cast<unsigned>(rsx::primitive_type::points) == 1);
static_assert(static_cast<unsigned>(rsx::primitive_type::lines) == 2);
static_assert(static_cast<unsigned>(rsx::primitive_type::line_loop) == 3);
static_assert(static_cast<unsigned>(rsx::primitive_type::line_strip) == 4);
static_assert(static_cast<unsigned>(rsx::primitive_type::triangles) == 5);
static_assert(static_cast<unsigned>(rsx::primitive_type::triangle_strip) == 6);
static_assert(static_cast<unsigned>(rsx::primitive_type::triangle_fan) == 7);
static_assert(static_cast<unsigned>(rsx::primitive_type::quads) == 8);
static_assert(static_cast<unsigned>(rsx::primitive_type::quad_strip) == 9);
static_assert(static_cast<unsigned>(rsx::primitive_type::polygon) == 10);
static_assert(static_cast<unsigned>(rsx::user_clip_plane_op::disable) == 0);
static_assert(static_cast<unsigned>(rsx::user_clip_plane_op::less_than) == 1);
static_assert(static_cast<unsigned>(rsx::user_clip_plane_op::greater_or_equal) == 2);
static_assert(static_cast<unsigned>(rsx::window_origin::top) == 0);
static_assert(static_cast<unsigned>(rsx::window_origin::bottom) == 1);
static_assert(static_cast<unsigned>(rsx::window_pixel_center::half) == 0);
static_assert(static_cast<unsigned>(rsx::window_pixel_center::integer) == 1);
static_assert(static_cast<unsigned>(rsx::comparison_function::never) == 0x200);
static_assert(static_cast<unsigned>(rsx::comparison_function::always) == 0x207);
static_assert(static_cast<unsigned>(rsx::fog_mode::linear) == 0x2601);
static_assert(static_cast<unsigned>(rsx::fog_mode::exponential) == 0x800);
static_assert(static_cast<unsigned>(rsx::fog_mode::exponential2) == 0x801);
static_assert(static_cast<unsigned>(rsx::fog_mode::exponential_abs) == 0x802);
static_assert(static_cast<unsigned>(rsx::fog_mode::exponential2_abs) == 0x803);
static_assert(static_cast<unsigned>(rsx::fog_mode::linear_abs) == 0x804);
static_assert(static_cast<unsigned>(rsx::surface_color_format::x1r5g5b5_z1r5g5b5) == static_cast<unsigned>(metal4::surface_color::x1rgb5_zero));
static_assert(static_cast<unsigned>(rsx::surface_color_format::x1r5g5b5_o1r5g5b5) == static_cast<unsigned>(metal4::surface_color::x1rgb5_one));
static_assert(static_cast<unsigned>(rsx::surface_color_format::r5g6b5) == static_cast<unsigned>(metal4::surface_color::rgb565));
static_assert(static_cast<unsigned>(rsx::surface_color_format::x8r8g8b8_z8r8g8b8) == static_cast<unsigned>(metal4::surface_color::xrgb8_zero));
static_assert(static_cast<unsigned>(rsx::surface_color_format::x8r8g8b8_o8r8g8b8) == static_cast<unsigned>(metal4::surface_color::xrgb8_one));
static_assert(static_cast<unsigned>(rsx::surface_color_format::a8r8g8b8) == static_cast<unsigned>(metal4::surface_color::argb8));
static_assert(static_cast<unsigned>(rsx::surface_color_format::b8) == static_cast<unsigned>(metal4::surface_color::b8));
static_assert(static_cast<unsigned>(rsx::surface_color_format::g8b8) == static_cast<unsigned>(metal4::surface_color::gb8));
static_assert(static_cast<unsigned>(rsx::surface_color_format::w16z16y16x16) == static_cast<unsigned>(metal4::surface_color::rgba16f));
static_assert(static_cast<unsigned>(rsx::surface_color_format::w32z32y32x32) == static_cast<unsigned>(metal4::surface_color::rgba32f));
static_assert(static_cast<unsigned>(rsx::surface_color_format::x32) == static_cast<unsigned>(metal4::surface_color::r32f));
static_assert(static_cast<unsigned>(rsx::surface_color_format::x8b8g8r8_z8b8g8r8) == static_cast<unsigned>(metal4::surface_color::xbgr8_zero));
static_assert(static_cast<unsigned>(rsx::surface_color_format::x8b8g8r8_o8b8g8r8) == static_cast<unsigned>(metal4::surface_color::xbgr8_one));
static_assert(static_cast<unsigned>(rsx::surface_color_format::a8b8g8r8) == static_cast<unsigned>(metal4::surface_color::abgr8));
static_assert(static_cast<unsigned>(rsx::surface_depth_format2::z16_uint) == 1);
static_assert(static_cast<unsigned>(rsx::surface_depth_format2::z24s8_uint) == 2);
static_assert(static_cast<unsigned>(rsx::surface_depth_format2::z16_float) == 3);
static_assert(static_cast<unsigned>(rsx::surface_depth_format2::z24s8_float) == 4);
static_assert(static_cast<unsigned>(rsx::surface_target::none) == 0);
static_assert(static_cast<unsigned>(rsx::surface_target::surface_a) == 1);
static_assert(static_cast<unsigned>(rsx::surface_target::surface_b) == 2);
static_assert(static_cast<unsigned>(rsx::surface_target::surfaces_a_b) == 0x13);
static_assert(static_cast<unsigned>(rsx::surface_target::surfaces_a_b_c) == 0x17);
static_assert(static_cast<unsigned>(rsx::surface_target::surfaces_a_b_c_d) == 0x1f);
static_assert(static_cast<unsigned>(rsx::surface_raster_type::linear) == 1);
static_assert(static_cast<unsigned>(rsx::surface_raster_type::swizzle) == 2);
static_assert(static_cast<unsigned>(rsx::surface_antialiasing::center_1_sample) == 0);
static_assert(static_cast<unsigned>(rsx::surface_antialiasing::diagonal_centered_2_samples) == 3);
static_assert(static_cast<unsigned>(rsx::surface_antialiasing::square_centered_4_samples) == 4);
static_assert(static_cast<unsigned>(rsx::surface_antialiasing::square_rotated_4_samples) == 5);
static_assert(rsx::vertex_arrays_changed == 4);

// Compile-only verification of the real rsx_state public register-array
// interface. Never constructs/starts RPCS3 or reads a running application.
metal4::fragment_texture_snapshot capture_real_rsx_texture_state(const rsx::rsx_state& state, std::uint32_t mask)
{
	return metal4::capture_fragment_texture_snapshot(state, mask);
}

std::uint32_t translate_real_rsx_texture_address(std::uint32_t offset, unsigned location, std::size_t size)
{
	if (size > std::numeric_limits<std::uint32_t>::max())
		throw std::overflow_error("RSX address-check size exceeds32bits");
	return rsx::get_address(offset, location, static_cast<std::uint32_t>(size));
}

metal4::pipeline_register_snapshot capture_real_rsx_pipeline_state(const rsx::rsx_state& state)
{
	return metal4::capture_pipeline_register_snapshot(state);
}

static_assert(metal4::pipeline_method_indices[0] == NV4097_SET_BLEND_ENABLE);
static_assert(metal4::pipeline_method_indices[1] == NV4097_SET_BLEND_FUNC_SFACTOR);
static_assert(metal4::pipeline_method_indices[2] == NV4097_SET_BLEND_FUNC_DFACTOR);
static_assert(metal4::pipeline_method_indices[3] == NV4097_SET_BLEND_EQUATION);
static_assert(metal4::pipeline_method_indices[4] == NV4097_SET_COLOR_MASK);
static_assert(metal4::pipeline_method_indices[5] == NV4097_SET_STENCIL_TEST_ENABLE);
static_assert(metal4::pipeline_method_indices[6] == NV4097_SET_STENCIL_MASK);
static_assert(metal4::pipeline_method_indices[7] == NV4097_SET_STENCIL_FUNC);
static_assert(metal4::pipeline_method_indices[8] == NV4097_SET_STENCIL_FUNC_REF);
static_assert(metal4::pipeline_method_indices[9] == NV4097_SET_STENCIL_FUNC_MASK);
static_assert(metal4::pipeline_method_indices[10] == NV4097_SET_STENCIL_OP_FAIL);
static_assert(metal4::pipeline_method_indices[11] == NV4097_SET_STENCIL_OP_ZFAIL);
static_assert(metal4::pipeline_method_indices[12] == NV4097_SET_STENCIL_OP_ZPASS);
static_assert(metal4::pipeline_method_indices[13] == NV4097_SET_TWO_SIDED_STENCIL_TEST_ENABLE);
static_assert(metal4::pipeline_method_indices[14] == NV4097_SET_BACK_STENCIL_MASK);
static_assert(metal4::pipeline_method_indices[15] == NV4097_SET_BACK_STENCIL_FUNC);
static_assert(metal4::pipeline_method_indices[16] == NV4097_SET_BACK_STENCIL_FUNC_REF);
static_assert(metal4::pipeline_method_indices[17] == NV4097_SET_BACK_STENCIL_FUNC_MASK);
static_assert(metal4::pipeline_method_indices[18] == NV4097_SET_BACK_STENCIL_OP_FAIL);
static_assert(metal4::pipeline_method_indices[19] == NV4097_SET_BACK_STENCIL_OP_ZFAIL);
static_assert(metal4::pipeline_method_indices[20] == NV4097_SET_BACK_STENCIL_OP_ZPASS);
static_assert(metal4::pipeline_method_indices[21] == NV4097_SET_BLEND_ENABLE_MRT);
static_assert(metal4::pipeline_method_indices[22] == NV4097_SET_COLOR_MASK_MRT);
static_assert(metal4::pipeline_method_indices[23] == NV4097_SET_LOGIC_OP_ENABLE);
static_assert(metal4::pipeline_method_indices[24] == NV4097_SET_LOGIC_OP);
static_assert(metal4::pipeline_method_indices[25] == NV4097_SET_SCISSOR_HORIZONTAL);
static_assert(metal4::pipeline_method_indices[26] == NV4097_SET_SCISSOR_VERTICAL);
static_assert(metal4::pipeline_method_indices[27] == NV4097_SET_DEPTH_FUNC);
static_assert(metal4::pipeline_method_indices[28] == NV4097_SET_DEPTH_MASK);
static_assert(metal4::pipeline_method_indices[29] == NV4097_SET_DEPTH_TEST_ENABLE);
static_assert(metal4::pipeline_method_indices[30] == NV4097_SET_CULL_FACE);
static_assert(metal4::pipeline_method_indices[31] == NV4097_SET_FRONT_FACE);
static_assert(metal4::pipeline_method_indices[32] == NV4097_SET_CULL_FACE_ENABLE);

static_assert(metal4::pipeline_method_indices[33] == NV4097_SET_BLEND_COLOR);
static_assert(metal4::pipeline_method_indices[34] == NV4097_SET_BLEND_COLOR2);

static_assert(metal4::rsx_vertex_instruction_limit == rsx::max_vertex_program_instructions);
static_assert(metal4::rsx_program_flat == RSX_SHADER_CONTROL_FLAT_SHADING);
static_assert(metal4::rsx_program_instanced_constants == RSX_SHADER_CONTROL_INSTANCED_CONSTANTS);
static_assert(metal4::rsx_fragment_float32 == CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS);
static_assert(metal4::rsx_fragment_depth_export == CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT);
metal4::vertex_program_snapshot capture_real_rsx_vertex_program(const RSXVertexProgram& program)
{
	return metal4::vertex_program_snapshot::capture(program);
}
metal4::fragment_program_snapshot capture_real_rsx_fragment_program(const RSXFragmentProgram& program, std::size_t maximum_bytes)
{
	return metal4::fragment_program_snapshot::capture(program, maximum_bytes);
}

class native_draw_frontend_compile : public metal4::rsx_program_frontend<GSRender>
{
  public:
	metal4::rsx_program_sampler_epoch prepare_programs_checked(const rsx::rsx_state& state,
		const std::function<std::unique_ptr<rsx::sampled_image_descriptor_base>(unsigned, const metal4::vertex_texture_registers&)>& vertex,
		const std::function<std::unique_ptr<rsx::sampled_image_descriptor_base>(unsigned, const metal4::fragment_texture_registers&)>& fragment)
	{
		return prepare_owned_programs(state, vertex, fragment);
	}
	struct epoch
	{
		metal4::framebuffer_snapshot framebuffer;
	};
	std::uint32_t submit_clause_checked(rsx::rsx_state& state, rsx::vertex_input_layout& layout,
										const std::function<std::optional<epoch>(std::uint32_t, std::uint32_t)>& prepare,
										const std::function<void(const metal4::rsx_draw_packet&, const metal4::framebuffer_snapshot&, std::uint32_t)>& submit)
	{
		return submit_owned_clause(state, layout, prepare, submit, metal4::provoking_vertex::last);
	}
	metal4::framebuffer_snapshot capture_framebuffer_checked(const rsx::framebuffer_layout& layout, bool validated) const
	{
		return capture_owned_framebuffer(layout, validated);
	}
	metal4::rsx_draw_packet capture_pass_checked(const rsx::rsx_state& state, const rsx::vertex_input_layout& layout)
	{
		return capture_owned_pass(state, layout, metal4::provoking_vertex::last);
	}
	metal4::draw_environment_snapshot capture_environment_checked(const rsx::rsx_state& state) const
	{
		return capture_owned_environment(state);
	}
	metal4::draw_geometry_plan capture_geometry_checked(const rsx::rsx_state& state,
														const rsx::vertex_input_layout& layout, metal4::provoking_vertex provoking)
	{
		return capture_owned_geometry(state, layout, provoking);
	}
	metal4::rsx_draw_snapshot capture_checked(const rsx::rsx_state& state, std::size_t limit) const
	{
		return capture_owned_draw(state, limit);
	}
	metal4::vertex_data_snapshot capture_vertices_checked(const rsx::rsx_state& state,
														  const rsx::vertex_input_layout& layout, metal4::vertex_stream_range range, std::size_t limit)
	{
		return capture_owned_vertex_data(state, layout, range, limit);
	}
};

std::unique_ptr<rsx::sampled_image_descriptor_base> make_fragment_program_descriptor_checked(
	const rsx::rsx_state& state, unsigned unit, const metal4::fragment_texture_registers& words, const metal4::texture_cache_state& metadata)
{
	return std::make_unique<metal4::rsx_sampler_descriptor>(words, metadata, state.fragment_textures[unit].format_ex());
}
std::unique_ptr<rsx::sampled_image_descriptor_base> make_vertex_program_descriptor_checked(
	const metal4::vertex_texture_registers& words, const metal4::texture_cache_state& metadata)
{
	return std::make_unique<metal4::rsx_sampler_descriptor>(words.upload_words(), metadata, rsx::texture_format_ex((words.format >> 8) & 255));
}
static_assert(static_cast<unsigned>(rsx::texture_dimension_extended::texture_dimension_1d) == 0);
static_assert(static_cast<unsigned>(rsx::texture_dimension_extended::texture_dimension_2d) == 1);
static_assert(static_cast<unsigned>(rsx::texture_dimension_extended::texture_dimension_cubemap) == 2);
static_assert(static_cast<unsigned>(rsx::texture_dimension_extended::texture_dimension_3d) == 3);
static_assert(rsx::RSX_FORMAT_CLASS_COLOR == 1);
static_assert(rsx::RSX_FORMAT_CLASS_DEPTH16_UNORM == 2);
static_assert(rsx::RSX_FORMAT_CLASS_DEPTH24_UNORM_X8_PACK32 == 8);
static_assert(rsx::RSX_FORMAT_CLASS_DEPTH16_FLOAT == 4);
static_assert(rsx::RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32 == 16);
