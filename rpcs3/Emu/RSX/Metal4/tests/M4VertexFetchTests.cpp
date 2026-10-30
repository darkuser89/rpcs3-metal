#include "M4VertexFetchLayout.h"
#include "M4VertexFetchKernels.h"
#include "M4VertexFetchHost.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
void require(bool condition)
{
	if (!condition) throw std::runtime_error("Metal 4 vertex fetch layout check failed");
}

metal4::vertex_attribute_descriptor descriptor(std::uint32_t stride, std::uint32_t frequency,
	std::uint32_t type, std::uint32_t count, std::uint32_t offset, unsigned flags)
{
	return {stride | (frequency << 8) | (type << 24) | (count << 27), offset | (flags << 29)};
}

void packed_fields()
{
	for (std::uint32_t stride = 0; stride < 256; ++stride)
		for (std::uint32_t type = 0; type < 8; ++type)
			for (std::uint32_t count = 0; count < 8; ++count)
				for (unsigned flags = 0; flags < 8; ++flags)
				{
					const auto d = descriptor(stride, 65535, type, count, 0x1fffffff, flags);
					require(d.stride() == stride && d.frequency() == 65535 && d.type() == type &&
						d.components() == count && d.offset() == 0x1fffffff &&
						d.big_endian() == bool(flags & 1) && d.transient() == bool(flags & 2) &&
						d.modulo() == bool(flags & 4));
				}
	for (std::uint32_t frequency = 0; frequency < 65536; ++frequency)
	{
		const auto d = descriptor(255, frequency, 7, 4, 17, 7);
		require(d.frequency() == frequency && d.stride() == 255 && d.offset() == 17);
	}
}

void address_models()
{
	std::size_t models = 0;
	// Independent, per-vertex/per-byte enumeration. In particular, the expected
	// modulo high-water mark isn't inferred from the endpoints of a draw.
	for (std::uint32_t first = 0; first < 19; ++first)
		for (std::uint32_t count = 0; count < 20; ++count)
			for (std::uint32_t frequency = 0; frequency < 12; ++frequency)
				for (std::uint32_t stride : {0u, 1u, 5u, 255u})
					for (unsigned modulo = 0; modulo < 2; ++modulo)
						for (const std::int32_t adjustment : {-7, 0, 13})
						{
							metal4::vertex_fetch_request request;
							request.first_vertex = first;
							request.vertex_count = count;
							request.vertex_base_index = 4;
							request.vertex_index_offset = adjustment;
							const auto d = descriptor(stride, frequency, 2, 3, 7, modulo * 4);
							bool valid = true;
							std::uint64_t expected = 0;
							for (std::uint32_t v = first; v < first + count; ++v)
							{
								std::int64_t element = 0;
								if (frequency && stride)
								{
									const auto adjusted = static_cast<std::int64_t>(v) + (modulo ? adjustment : -4);
									if (adjusted < 0) { valid = false; break; }
									element = modulo ? adjusted % frequency : adjusted / frequency;
								}
								for (unsigned component = 0; component < 3; ++component)
									for (unsigned byte = 0; byte < 4; ++byte)
									{
										const auto address = 7 + static_cast<std::uint64_t>(element) * stride + component * 4 + byte;
										expected = std::max(expected, address + 1);
									}
							}
							const auto extent = metal4::vertex_fetch_extent(d, request);
							require(extent.has_value() == valid);
							if (valid) require(*extent == expected);
							++models;
							}
	std::cout << models << " independent stream-address models passed\n";
}

void format_and_boundary_models()
{
	// An independently stated storage contract, rather than reusing a width
	// lookup from the implementation. Compressed XYZ is exactly one dword.
	constexpr std::uint32_t payloads[8][5] = {
		{0, 0, 0, 0, 0}, {0, 2, 4, 6, 8}, {0, 4, 8, 12, 16}, {0, 2, 4, 6, 8},
		{0, 1, 2, 3, 4}, {0, 2, 4, 6, 8}, {0, 4, 0, 0, 0}, {0, 1, 2, 3, 4}};
	metal4::vertex_fetch_request request;
	request.vertex_count = 1;
	for (std::uint32_t type = 0; type < 8; ++type)
		for (std::uint32_t count = 0; count < 8; ++count)
			for (unsigned flags = 0; flags < 8; ++flags)
			{
				const auto extent = metal4::vertex_fetch_extent(descriptor(0, 0, type, count, 31, flags), request);
				const bool valid = count == 0 || (count < 5 && payloads[type][count]);
				require(extent.has_value() == valid);
				if (valid) require(*extent == (count ? 31 + payloads[type][count] : 0));
			}
	request.first_vertex = 0xffffffffu;
	request.vertex_count = 1;
	require(metal4::vertex_fetch_bytes(request) == 256);
	auto d = descriptor(255, 1, 2, 4, 0x1fffffff, 0);
	require(metal4::vertex_fetch_extent(d, request) == 0x1fffffffull + 0xffffffffull * 255 + 16);
	request.vertex_count = 2;
	require(!metal4::vertex_fetch_bytes(request));
	request.vertex_count = 1;
	d.storage |= 0x80000000u;
	request.vertex_index_offset = 1;
	require(!metal4::vertex_fetch_extent(d, request));
	request.first_vertex = 0;
	request.vertex_index_offset = -1;
	require(!metal4::vertex_fetch_extent(d, request));
	request.vertex_count = 0;
	require(metal4::vertex_fetch_extent(d, request) == 0);
	request.attribute_mask = 65536;
	require(!metal4::vertex_fetch_bytes(request));
	request.attribute_mask = 0;
	request.first_vertex = 0;
	request.vertex_count = 0xffffffffu;
	require(metal4::vertex_fetch_bytes(request) == 0xffffffffull * 256);
}

void kernel_formats()
{
	fetch_host::rsx_fetch_parameters p{};
	p.persistent_length = 16;
	p.vertex_count = 1;
	p.chunk_count = 1;
	p.attribute_mask = 1;
	std::array<unsigned char, 16> bytes{};
	std::array<fetch_host::float4, 16> output{};
	const auto set = [&](unsigned type, unsigned count, bool big_endian)
	{
		p.attributes[0] = {type << 24 | count << 27, big_endian ? 1u << 29 : 0};
	};
	const auto store = [&](std::uint32_t value, unsigned width, bool big_endian, unsigned component = 0)
	{
		for (unsigned b = 0; b < width; ++b)
			bytes[component * width + b] = static_cast<unsigned char>(value >> ((big_endian ? width - 1 - b : b) * 8));
	};
	const auto run = [&] { fetch_host::rpcs3_fetch_vertices(bytes.data(), nullptr, output.data(), p, 0); };
	const auto near = [](float actual, double expected)
	{
		const auto rounded = static_cast<float>(expected);
		const auto ulp = std::abs(std::nextafter(rounded, INFINITY) - rounded);
		require(std::abs(actual - rounded) <= ulp);
	};
	for (const bool be : {false, true})
	{
		for (unsigned raw = 0; raw < 65536; ++raw)
		{
			store(raw, 2, be);
			const auto signed_value = raw < 32768 ? static_cast<int>(raw) : static_cast<int>(raw) - 65536;
			set(1, 1, be); run();
			near(output[0].x, (static_cast<double>(signed_value) + 0.5) / 32767.5);
			near(output[0].y, 0.5 / 32767.5); // VK's missing SNORM lanes include the bias.
			require(output[0].w == 1);
			set(5, 1, be); run();
			require(output[0].x == static_cast<float>(signed_value) && output[0].y == 0 && output[0].w == 1);
			set(3, 1, be); run();
			const unsigned exponent = raw / 1024 % 32, mantissa = raw % 1024;
			const double magnitude = exponent ? std::ldexp(1 + mantissa / 1024.0, static_cast<int>(exponent) - 15)
				: std::ldexp(static_cast<double>(mantissa), -24);
			if (exponent == 31)
				require(mantissa ? std::isnan(output[0].x) : std::isinf(output[0].x));
			else
			{
				const float expected = static_cast<float>((raw & 32768) ? -magnitude : magnitude);
				require(std::bit_cast<std::uint32_t>(output[0].x) == std::bit_cast<std::uint32_t>(expected));
			}
		}
		for (unsigned raw = 0; raw < 256; ++raw)
		{
			store(raw, 1, be); set(4, 1, be); run(); near(output[0].x, raw / 255.0);
			set(7, 1, be); run(); require(output[0].x == static_cast<float>(raw));
		}
		for (const std::uint32_t raw : {0u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x3eaaaaabu,
			0x00000001u, 0x007fffffu, 0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u})
		{
			store(raw, 4, be); set(2, 1, be); run();
			if (std::isnan(std::bit_cast<float>(raw))) require(std::isnan(output[0].x));
			else require(std::bit_cast<std::uint32_t>(output[0].x) == raw);
		}
		// All signed encodings of each independent 11/11/10-bit component.
		for (unsigned axis = 0; axis < 3; ++axis)
		{
			const unsigned width = axis == 2 ? 10 : 11;
			for (unsigned raw = 0; raw < (1u << width); ++raw)
			{
				store(raw << (axis * 11), 4, be); set(6, 1, be); run();
				const int signed_value = raw < (1u << (width - 1)) ? static_cast<int>(raw) : static_cast<int>(raw) - (1 << width);
				near(output[0][axis], static_cast<double>(signed_value) * (axis == 2 ? 64 : 32) / 32767);
				require(output[0].w == 1);
			}
		}
		for (unsigned type = 1; type < 8; ++type)
			for (unsigned components = 1; components < 5; ++components)
			{
				if (type == 6) continue;
				const unsigned width = type == 2 ? 4 : (type == 4 || type == 7) ? 1 : 2;
				for (unsigned c = 0; c < components; ++c)
					store(type == 2 ? std::bit_cast<std::uint32_t>(float(c + 1)) : type == 3 ? 0x3c00u + c * 0x400u : c + 1, width, be, c);
				set(type, components, be); run();
				for (unsigned c = 0; c < components; ++c)
				{
					const double value = type == 3 ? std::ldexp(1.0, static_cast<int>(c)) : c + 1;
					near(output[0][c], type == 1 ? (value + 0.5) / 32767.5 : type == 4 ? value / 255 : value);
				}
				if (components < 4) require(output[0].w == 1);
			}
	}
	std::cout << "393216 exhaustive signed16/half cases, 10240 compressed lanes and fixed endian/float/byte fixtures passed\n";
}

void kernel_addressing()
{
	// Invoke the actual body one thread at a time. Adjacent vertices/attributes
	// are independent, including when a fetch is encoded in multiple chunks.
	fetch_host::rsx_fetch_parameters p{};
	p.persistent_length = 32;
	p.transient_length = 32;
	p.first_vertex = 12;
	p.vertex_count = 8;
	p.vertex_base_index = 11;
	p.vertex_index_offset = -10;
	p.attribute_mask = 7;
	p.attributes[0] = {1u | (3u << 8) | (7u << 24) | (1u << 27), 1};
	p.attributes[1] = {1u | (5u << 8) | (7u << 24) | (1u << 27), 2u | 0xc0000000u};
	p.attributes[2] = {255u | (7u << 24) | (1u << 27), 19};
	std::array<unsigned char, 32> persistent{}, transient{};
	for (unsigned i = 0; i < 32; ++i) { persistent[i] = static_cast<unsigned char>(i); transient[i] = static_cast<unsigned char>(i + 100); }
	std::array<fetch_host::float4, 8 * 16 + 1> output{};
	output.back() = {999, 999, 999, 999};
	for (const unsigned begin : {0u, 3u})
	{
		p.chunk_base = begin;
		p.chunk_count = begin ? 5 : 3;
		for (unsigned id = 0; id < p.chunk_count * 16; ++id)
			fetch_host::rpcs3_fetch_vertices(persistent.data(), transient.data(), output.data(), p, id);
	}
	for (unsigned v = 0; v < 8; ++v)
	{
		require(output[v * 16].x == static_cast<float>(1 + (v + 1) / 3));
		require(output[v * 16 + 1].x == static_cast<float>(102 + (v + 2) % 5));
		require(output[v * 16 + 2].x == 19);
		for (unsigned a = 3; a < 16; ++a)
			require(output[v * 16 + a].x == 0 && output[v * 16 + a].w == 1);
	}
	// Guards are a second line of defense; public host-side validation rejects
	// these cases before dispatch. CPU sanitizers catch accidental byte reads.
	p.chunk_base = 0; p.chunk_count = 1;
	p.persistent_length = 1; p.transient_length = 1;
	for (unsigned id = 0; id < 16; ++id)
		fetch_host::rpcs3_fetch_vertices(persistent.data(), transient.data(), output.data(), p, id);
	require(output[0].x == 0 && output[1].x == 0 && output[2].x == 0);
	fetch_host::rpcs3_fetch_vertices(nullptr, nullptr, output.data(), p, 16);
	p.chunk_base = p.vertex_count;
	fetch_host::rpcs3_fetch_vertices(nullptr, nullptr, output.data(), p, 0);
	require(output.back().x == 999 && output.back().w == 999);
}
}

int main(int argc, char** argv)
{
	try
	{
		packed_fields();
		address_models();
		format_and_boundary_models();
		kernel_formats();
		kernel_addressing();
		if (argc == 2)
		{
			std::ofstream output(std::filesystem::path(argv[1]), std::ios::binary);
			output << metal4::vertex_fetch_kernels;
			if (!output) throw std::runtime_error("Cannot export Metal vertex fetch kernel");
		}
		std::cout << "Metal 4 vertex fetch CPU layout checks passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
