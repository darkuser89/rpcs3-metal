#pragma once

#include "M4Device.h"

#include <array>
#include <cstddef>
#include <span>

namespace metal4
{

enum class buffer_storage
{
	shared,
	private_gpu,
};

[[nodiscard]] id<MTLBuffer> make_buffer(context& ctx, std::size_t size, buffer_storage storage) __attribute__((ns_returns_retained));
[[nodiscard]] id<MTLBuffer> make_upload(context& ctx, std::span<const std::byte> bytes) __attribute__((ns_returns_retained));
// CPU access requires external synchronization with any submitted GPU uses.
[[nodiscard]] std::span<std::byte> map_buffer(id<MTLBuffer> buffer, std::size_t offset, std::size_t size);
void copy_buffer(context& ctx, id<MTLBuffer> source, std::size_t source_offset,
	id<MTLBuffer> destination, std::size_t destination_offset, std::size_t size);
void fill_buffer(context& ctx, id<MTLBuffer> buffer, std::size_t offset, std::size_t size, std::uint8_t value);
[[nodiscard]] id<MTLSamplerState> make_sampler(context& ctx, MTLSamplerDescriptor* descriptor);

// Each encoding gets an exclusive immutable table snapshot through completion.
// A completed/cancelled command slot may recycle its tables for later work;
// callers must not use returned table handles beyond their submission lifetime.
class bindings final
{
	friend class context;
	friend class vertex_shader;

	struct buffer_binding
	{
		id<MTLBuffer> buffer = nil;
		std::size_t offset = 0;
		std::size_t stride = 0;
	};
	std::array<buffer_binding, 31> m_buffers{};
	std::array<id<MTLTexture>, 128> m_textures{};
	std::array<id<MTLSamplerState>, 16> m_samplers{};

public:

	void set_buffer(std::size_t index, id<MTLBuffer> buffer, std::size_t offset = 0, std::size_t stride = 0);
	void set_texture(std::size_t index, id<MTLTexture> texture);
	void set_sampler(std::size_t index, id<MTLSamplerState> sampler);
	[[nodiscard]] id<MTL4ArgumentTable> snapshot(context& ctx) const;
};

}
