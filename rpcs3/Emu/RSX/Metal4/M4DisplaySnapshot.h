#pragma once

#include "M4GuestSurface.h"

#include <algorithm>
#include <iterator>

namespace metal4
{
enum class display_format : std::uint32_t
{
	xrgb8 = 0,
	xbgr8 = 1,
	rgbx16f = 2
};
struct display_snapshot
{
	std::uint32_t offset = 0, width = 0, height = 0, pitch = 0;
	display_format format = display_format::xrgb8;
	std::optional<std::uint32_t> right_offset;
	bool operator==(const display_snapshot&) const = default;
};
inline std::size_t display_pixel_bytes(display_format format)
{
	switch (format)
	{
	case display_format::xrgb8:
	case display_format::xbgr8:
		return 4;
	case display_format::rgbx16f:
		return 8;
	default:
		throw std::invalid_argument("Unknown RSX video-output format");
	}
}
inline guest_surface_layout display_coverage(const display_snapshot& image, std::uint32_t address)
{
	if (image.offset & 0x80000000u)
		throw std::invalid_argument("RSX display offset exceeds local address domain");
	const auto bytes = display_pixel_bytes(image.format);
	if (!image.width || !image.height || !image.pitch || image.pitch < std::uint64_t(image.width) * bytes)
		throw std::invalid_argument("Invalid RSX display dimensions/pitch");
	const auto result = plan_guest_surface(address, image.width, image.height, bytes, image.pitch);
	if (result.range.size > 0x80000000ull - image.offset)
		throw std::out_of_range("RSX display coverage exceeds the local offset domain");
	return result;
}
inline display_snapshot display_eye(const display_snapshot& image, unsigned eye)
{
	if (eye > 1 || (eye && !image.right_offset)) throw std::invalid_argument("Absent RSX display eye");
	auto result = image;
	if (eye) result.offset = *image.right_offset;
	result.right_offset.reset();
	(void)display_coverage(result, 0);
	return result;
}

// Freeze on the RSX thread under its method/AV epoch. Bounds precede any
// buffer access. No GPU work or VM access in this capture operation.
template <typename Buffers, typename Flip, typename Av>
[[nodiscard]] std::optional<display_snapshot> capture_display_snapshot(const Buffers& buffers, std::size_t count,
																	   const Flip& flip, const Av& av)
{
	if (flip.skip_frame)
		return {};
	if (count > std::size(buffers))
		throw std::out_of_range("RSX display buffer count exceeds storage");
	if (flip.buffer >= count)
		return {};
	const auto& buffer = buffers[flip.buffer];
	display_snapshot image{buffer.offset, buffer.width, buffer.height, buffer.pitch};
	if (!image.width)
	{
		image.width = av.resolution_x;
		image.height = av.resolution_y;
	}
	if (av.state)
	{
		image.format = static_cast<display_format>(av.format);
		// Default pitch uses the uncropped buffer width, as the VK AV contract.
		const auto packed = std::uint64_t(image.width) * display_pixel_bytes(image.format);
		if (!image.pitch)
		{
			if (packed > UINT32_MAX)
				throw std::overflow_error("RSX display pitch overflow");
			image.pitch = static_cast<std::uint32_t>(packed);
		}
		image.width = std::min<std::uint32_t>(image.width, av.resolution_x);
		if (av.stereo_enabled && av.resolution_y <= 30)
			throw std::invalid_argument("RSX stereo frame has no complete eye extent");
		const auto height = av.stereo_enabled ? (av.resolution_y - 30) / 2 : av.resolution_y;
		image.height = std::min<std::uint32_t>(image.height, height);
		if (av.stereo_enabled)
		{
			const auto right = std::uint64_t(image.offset) + (std::uint64_t(image.height) + 30) * image.pitch;
			if (right >= 0x80000000ull) throw std::overflow_error("RSX right display eye exceeds the local offset domain");
			image.right_offset = static_cast<std::uint32_t>(right);
		}
	}
	else if (!image.pitch)
	{
		const auto packed = std::uint64_t(image.width) * 4;
		if (packed > UINT32_MAX)
			throw std::overflow_error("RSX display pitch overflow");
		image.pitch = static_cast<std::uint32_t>(packed);
	}
	if (!image.width || !image.height)
		return {};
	(void)display_coverage(image, 0);
	if (image.right_offset) (void)display_eye(image, 1);
	return image;
}
} // namespace metal4
