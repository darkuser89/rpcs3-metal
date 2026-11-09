#include "M4Resources.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <cstring>
#include <stdexcept>

namespace metal4
{

namespace
{

void validate_range(id<MTLBuffer> buffer, std::size_t offset, std::size_t size)
{
	// Subtraction keeps the check safe even for adversarial 64-bit offsets.
	if (!buffer || offset > buffer.length || size > buffer.length - offset)
	{
		throw std::out_of_range("Metal 4 buffer range exceeds allocation");
	}
}

} // namespace

id<MTLBuffer> make_buffer(context& ctx, std::size_t size, buffer_storage storage)
{
	if (size == 0 || size > ctx.device().maxBufferLength)
	{
		throw std::invalid_argument("Metal 4 buffer size exceeds device limits");
	}
	MTLStorageMode mode;
	switch (storage)
	{
	case buffer_storage::shared:
		mode = MTLStorageModeShared;
		break;
	case buffer_storage::private_gpu:
		mode = MTLStorageModePrivate;
		break;
	default:
		throw std::invalid_argument("Invalid Metal 4 buffer storage mode");
	}
	return ctx.allocate_buffer(size, mode);
}

id<MTLBuffer> make_upload(context& ctx, std::span<const std::byte> bytes)
{
	auto buffer = make_buffer(ctx, bytes.size(), buffer_storage::shared);
	std::memcpy(buffer.contents, bytes.data(), bytes.size());
	return buffer;
}

std::span<std::byte> map_buffer(id<MTLBuffer> buffer, std::size_t offset, std::size_t size)
{
	validate_range(buffer, offset, size);
	if (buffer.storageMode != MTLStorageModeShared || !buffer.contents)
	{
		throw std::invalid_argument("Only shared Metal 4 buffers can be mapped");
	}
	return {static_cast<std::byte*>(buffer.contents) + offset, size};
}

void copy_buffer(context& ctx, id<MTLBuffer> source, std::size_t source_offset,
				 id<MTLBuffer> destination, std::size_t destination_offset, std::size_t size)
{
	validate_range(source, source_offset, size);
	validate_range(destination, destination_offset, size);
	if (size == 0)
		return;
	if ((source_offset | destination_offset | size) & 3)
	{
		throw std::invalid_argument("Metal 4 buffer blits require four-byte alignment");
	}
	if (source == destination &&
		(source_offset < destination_offset + size && destination_offset < source_offset + size))
	{
		throw std::invalid_argument("Metal 4 buffer blits cannot overlap");
	}
	auto encoder = ctx.checked_blit({source, source_offset, size}, {destination, destination_offset, size});
	[encoder copyFromBuffer:source sourceOffset:source_offset toBuffer:destination destinationOffset:destination_offset size:size];
}

void fill_buffer(context& ctx, id<MTLBuffer> buffer, std::size_t offset, std::size_t size, std::uint8_t value)
{
	validate_range(buffer, offset, size);
	if (size == 0)
		return;
	auto encoder = ctx.checked_blit({}, {buffer, offset, size});
	[encoder fillBuffer:buffer range:NSMakeRange(offset, size) value:value];
}

id<MTLSamplerState> make_sampler(context& ctx, MTLSamplerDescriptor* descriptor)
{
	if (!descriptor)
	{
		throw std::invalid_argument("Metal 4 sampler requires a descriptor");
	}
	if (descriptor.lodBias != 0)
		throw std::invalid_argument("Native sampler descriptor bias is not verified; use explicit shader sampler parameters");
	MTLSamplerDescriptor* copy = [descriptor copy];
	copy.supportArgumentBuffers = YES;
	auto sampler = [ctx.device() newSamplerStateWithDescriptor:copy];
	if (!sampler)
	{
		throw std::runtime_error("Metal 4 sampler allocation failed");
	}
	return sampler;
}

void bindings::set_buffer(std::size_t index, id<MTLBuffer> buffer, std::size_t offset, std::size_t stride)
{
	if (buffer)
	{
		validate_range(buffer, offset, 1);
	}
	else if (offset || stride)
	{
		throw std::invalid_argument("An unbound Metal 4 buffer cannot have an offset or stride");
	}
	m_buffers.at(index) = {buffer, offset, stride};
}

void bindings::set_texture(std::size_t index, id<MTLTexture> texture)
{
	m_textures.at(index) = texture;
}

void bindings::set_sampler(std::size_t index, id<MTLSamplerState> sampler)
{
	m_samplers.at(index) = sampler;
}

id<MTL4ArgumentTable> bindings::snapshot(context& ctx) const
{
	@autoreleasepool
	{
		auto table = ctx.argument_table();
		for (std::size_t i = 0; i < m_buffers.size(); ++i)
		{
			const auto& binding = m_buffers[i];
			if (binding.buffer)
			{
				ctx.use_resource(binding.buffer);
				[table setAddress:binding.buffer.gpuAddress + binding.offset attributeStride:binding.stride atIndex:i];
			}
		}
		for (std::size_t i = 0; i < m_textures.size(); ++i)
		{
			if (m_textures[i])
			{
				ctx.use_resource(m_textures[i]);
				[table setTexture:m_textures[i].gpuResourceID atIndex:i];
			}
		}
		for (std::size_t i = 0; i < m_samplers.size(); ++i)
		{
			if (m_samplers[i])
			{
				if (m_samplers[i].device != ctx.device())
				{
					throw std::invalid_argument("Metal 4 sampler belongs to another device");
				}
				ctx.keep_alive(m_samplers[i]);
				[table setSamplerState:m_samplers[i].gpuResourceID atIndex:i];
			}
		}
		return table;
	}
}

} // namespace metal4
