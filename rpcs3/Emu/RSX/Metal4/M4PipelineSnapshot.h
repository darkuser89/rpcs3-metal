#pragma once

#include "M4RenderState.h"

#include <bit>
#include <span>
#include <stdexcept>

namespace metal4
{
enum class pipeline_register : std::size_t
{
	blend_enable,
	source_factor,
	destination_factor,
	equation,
	color_mask,
	stencil_enable,
	stencil_write_mask,
	stencil_compare,
	stencil_reference,
	stencil_read_mask,
	stencil_fail,
	stencil_depth_fail,
	stencil_pass,
	stencil_two_sided,
	back_write_mask,
	back_compare,
	back_reference,
	back_read_mask,
	back_fail,
	back_depth_fail,
	back_pass,
	blend_mrt,
	color_mask_mrt,
	logic_enable,
	logic_operation,
	scissor_horizontal,
	scissor_vertical,
	depth_compare,
	depth_write,
	depth_enable,
	cull_mode,
	front_face,
	cull_enable,
	blend_color,
	blend_color2,
	count
};
inline constexpr std::array<std::size_t, std::size_t(pipeline_register::count)> pipeline_method_indices{
	0x310 / 4, 0x314 / 4, 0x318 / 4, 0x320 / 4, 0x324 / 4,
	0x328 / 4, 0x32c / 4, 0x330 / 4, 0x334 / 4, 0x338 / 4, 0x33c / 4, 0x340 / 4, 0x344 / 4, 0x348 / 4,
	0x34c / 4, 0x350 / 4, 0x354 / 4, 0x358 / 4, 0x35c / 4, 0x360 / 4, 0x364 / 4,
	0x36c / 4, 0x370 / 4, 0x374 / 4, 0x378 / 4,
	0x8c0 / 4, 0x8c4 / 4, 0xa6c / 4, 0xa70 / 4, 0xa74 / 4, 0x1830 / 4, 0x1834 / 4, 0x183c / 4, 0x31c / 4, 0x37c / 4};
struct pipeline_scissor
{
	std::uint32_t x, y, width, height;
};
enum class pipeline_cull
{
	none,
	front,
	back,
	all
};

// Owned fixed/dynamic pipeline words. Capture while RSX method writes are
// excluded on the render thread. Attachment/sample/viewport/ROP environment,
// shaders, fetch, VM and actual GSRender scheduling are separate contracts.
class pipeline_register_snapshot final
{
	std::array<std::uint32_t, pipeline_method_indices.size()> m_words{};

  public:
	[[nodiscard]] static pipeline_register_snapshot capture(std::span<const std::uint32_t> methods)
	{
		pipeline_register_snapshot result;
		for (const auto index : pipeline_method_indices)
			if (index >= methods.size())
				throw std::out_of_range("RSX method array is too short for a pipeline snapshot");
		for (std::size_t i = 0; i < word_count(); ++i)
			result.m_words[i] = methods[pipeline_method_indices[i]];
		return result;
	}
	[[nodiscard]] static constexpr std::size_t word_count() noexcept { return pipeline_method_indices.size(); }
	[[nodiscard]] std::uint32_t word(pipeline_register index) const
	{
		return m_words.at(std::size_t(index));
	}
	[[nodiscard]] guest_render_state render_state(std::uint32_t samples, bool force_disable_blending = false) const
	{
		using R = pipeline_register;
		guest_render_state result;
		result.sample_count = samples;
		result.depth_test = bool(word(R::depth_enable));
		result.depth_write = bool(word(R::depth_write));
		result.depth_compare = word(R::depth_compare);
		result.stencil_test = bool(word(R::stencil_enable));
		result.two_sided_stencil = bool(word(R::stencil_two_sided));
		result.front = {word(R::stencil_compare), word(R::stencil_fail), word(R::stencil_depth_fail), word(R::stencil_pass), word(R::stencil_read_mask) & 255, word(R::stencil_write_mask) & 255, word(R::stencil_reference) & 255};
		result.back = {word(R::back_compare), word(R::back_fail), word(R::back_depth_fail), word(R::back_pass), word(R::back_read_mask) & 255, word(R::back_write_mask) & 255, word(R::back_reference) & 255};
		result.logic_enabled = bool(word(R::logic_enable));
		result.logic_operation = word(R::logic_operation);
		const auto blend_mask = ((word(R::blend_enable) ? 1u : 0u) | word(R::blend_mrt)) & 15;
		for (unsigned slot = 0; slot < 4; ++slot)
		{
			auto& color = result.colors[slot];
			color.enabled = !force_disable_blending && !result.logic_enabled && (blend_mask & (1u << slot));
			color.source_rgb = word(R::source_factor) & 65535;
			color.source_alpha = word(R::source_factor) >> 16;
			color.destination_rgb = word(R::destination_factor) & 65535;
			color.destination_alpha = word(R::destination_factor) >> 16;
			color.operation_rgb = word(R::equation) & 65535;
			color.operation_alpha = word(R::equation) >> 16;
			if (!slot)
			{
				const auto mask = word(R::color_mask);
				color.write_mask = ((mask & 0xff0000) ? 1u : 0u) | ((mask & 0xff00) ? 2u : 0u) | ((mask & 255) ? 4u : 0u) | ((mask & 0xff000000) ? 8u : 0u);
			}
			else
			{
				const auto mask = (word(R::color_mask_mrt) >> (slot * 4)) & 15;
				color.write_mask = ((mask & 2) ? 1u : 0u) | ((mask & 4) ? 2u : 0u) | ((mask & 8) ? 4u : 0u) | ((mask & 1) ? 8u : 0u);
			}
		}
		return result;
	}
	[[nodiscard]] std::array<float, 4> blend_color(bool half_format) const
	{
		const auto first = word(pipeline_register::blend_color), second = word(pipeline_register::blend_color2);
		if (half_format)
		{
			const auto decode = [](std::uint32_t bits)
			{ return static_cast<float>(std::bit_cast<_Float16>(static_cast<std::uint16_t>(bits))); };
			return {decode(first), decode(first >> 16), decode(second), decode(second >> 16)};
		}
		return {((first >> 16) & 255) / 255.f, ((first >> 8) & 255) / 255.f, (first & 255) / 255.f, (first >> 24) / 255.f};
	}
	[[nodiscard]] pipeline_scissor scissor() const noexcept
	{
		const auto x = m_words[std::size_t(pipeline_register::scissor_horizontal)], y = m_words[std::size_t(pipeline_register::scissor_vertical)];
		return {x & 65535, y & 65535, x >> 16, y >> 16};
	}
	[[nodiscard]] bool clockwise() const
	{
		const auto value = word(pipeline_register::front_face);
		if (value != 0x900 && value != 0x901)
			throw std::invalid_argument("Invalid RSX front-face register");
		return value == 0x900;
	}
	[[nodiscard]] pipeline_cull cull() const
	{
		if (!word(pipeline_register::cull_enable))
			return pipeline_cull::none;
		switch (word(pipeline_register::cull_mode))
		{
		case 0x404:
			return pipeline_cull::front;
		case 0x405:
			return pipeline_cull::back;
		case 0x408:
			return pipeline_cull::all;
		default:
			throw std::invalid_argument("Invalid RSX cull-face register");
		}
	}
};
template <typename RsxState>
[[nodiscard]] pipeline_register_snapshot capture_pipeline_register_snapshot(const RsxState& state)
{
	return pipeline_register_snapshot::capture(std::span<const std::uint32_t>(state.registers));
}
} // namespace metal4
