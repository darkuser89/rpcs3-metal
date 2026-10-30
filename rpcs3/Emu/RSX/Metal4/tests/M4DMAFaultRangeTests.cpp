#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/RSXOffload.h"
#include "M4FaultRange.h"
#include "util/logs.hpp"
#include "util/sysinfo.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

// Standalone linkage only. The actual BufferUtils writer below executes on
// owned host memory;no VM,thread manager,logging service or app is initialized.
// Diagnostics fail the test;no production implementation is replaced in the
// application. AVX is absent on this arm64-only target.
logs::registerer::registerer(logs::channel&) {}
u64 utils::_get_main_tid() { return 0; }
bool utils::has_avx() { return false; }
template <>
void fmt_class_string<u8>::format(std::string&, u64) { throw std::logic_error("Unexpected shared format diagnostic"); }
namespace fmt
{
[[noreturn]] void raw_verify_error(std::source_location, const char8_t*, u64)
{
	throw std::logic_error("Unexpected shared writer verification failure");
}
[[noreturn]] void raw_throw_exception(std::source_location, const char*, const fmt_type_info*, const u64*)
{
	throw std::logic_error("Unexpected shared writer exception");
}
} // namespace fmt

namespace
{
using metal4::guest_surface_range;
using rsx::primitive_type;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
template <typename F>
void refused(F&& operation)
{
	bool failed = false;
	try
	{
		operation();
	}
	catch (const std::invalid_argument&)
	{
		failed = true;
	}
	catch (const std::overflow_error&)
	{
		failed = true;
	}
	require(failed, "Invalid transfer extent was accepted");
}
} // namespace
int main()
{
	try
	{
		unsigned cases = 0;
		for (auto primitive : {primitive_type::line_loop, primitive_type::triangle_fan, primitive_type::polygon, primitive_type::quads})
			for (unsigned count : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u, 255u, 256u, 257u, 65535u, 65536u})
			{
				const auto bytes = rsx::dma_manager::index_emulation_bytes(primitive, count);
				if ((primitive == primitive_type::triangle_fan || primitive == primitive_type::polygon) && count < 2)
				{
					require(!bytes, "Undersized fan/polygon would underflow the shared writer loop");
					continue;
				}
				require(bytes.has_value() && *bytes % sizeof(u16) == 0, "Valid expansion has no u16 byte extent");
				const auto indices = *bytes / sizeof(u16);
				// Aligned source for the real writer's eight-lane ARM NEON path.
				std::vector<u16> buffer(indices + 32, 0xdead);
				write_index_array_for_non_indexed_non_native_primitive_to_buffer(reinterpret_cast<char*>(buffer.data() + 16), primitive, count);
				require(std::all_of(buffer.begin(), buffer.begin() + 16, [](u16 v)
									{ return v == 0xdead; }) &&
							std::all_of(buffer.begin() + 16 + indices, buffer.end(), [](u16 v)
										{ return v == 0xdead; }),
						"Real writer escaped reported byte extent");
				std::size_t actual = 0;
				const auto expect = [&](u16 index)
				{
					require(actual < indices && buffer[16 + actual++] == index, "Real shared expanded index differs");
				};
				if (primitive == primitive_type::line_loop)
				{
					for (unsigned i = 0; i < count; ++i)
						expect(static_cast<u16>(i));
					expect(0);
				}
				else if (primitive == primitive_type::quads)
				{
					for (unsigned i = 0; i + 3 < count; i += 4)
						for (unsigned v : {i, i + 1, i + 2, i + 2, i + 3, i})
							expect(static_cast<u16>(v));
				}
				else
					for (unsigned i = 2; i < count; ++i)
						for (unsigned v : {0u, i - 1, i})
							expect(static_cast<u16>(v));
				require(actual == indices, "Fault extent includes unwritten indices");
				if (*bytes)
				{
					const auto start = static_cast<u32>(0x1'0000'0000ull - *bytes);
					const auto actual = rsx::dma_manager::fault_range_from_bytes(start, *bytes);
					require(actual && actual->start == start && actual->end == 0xffff'ffff && actual->length() == *bytes,
							"Shared fault-range constructor lost exact tail byte");
					require(!rsx::dma_manager::fault_range_from_bytes(start, *bytes + 1), "Shared fault range wrapped last guest byte");
					metal4::validate_fault_transfer_range({start, static_cast<std::size_t>(*bytes)}, start);
					metal4::validate_fault_transfer_range({start, static_cast<std::size_t>(*bytes)}, 0xffff'ffff);
					if (start)
						refused([&]
								{ metal4::validate_fault_transfer_range({start, static_cast<std::size_t>(*bytes)}, start - 1); });
					refused([&]
							{ metal4::validate_fault_transfer_range({start, static_cast<std::size_t>(*bytes + 1)}, start); });
					const guest_surface_range transfer{0x1000, static_cast<std::size_t>(*bytes)};
					refused([&]
							{ metal4::validate_fault_transfer_range(transfer, 0x1000 + static_cast<u32>(*bytes)); });
				}
				++cases;
			}
		for (auto primitive : {primitive_type::points, primitive_type::lines, primitive_type::line_strip, primitive_type::triangles,
							   primitive_type::triangle_strip, primitive_type::quad_strip, static_cast<primitive_type>(0), static_cast<primitive_type>(255)})
			require(!rsx::dma_manager::index_emulation_bytes(primitive, 8), "Native/invalid primitive was accepted for expansion");
		constexpr auto max = std::numeric_limits<u32>::max();
		require(rsx::dma_manager::index_emulation_bytes(primitive_type::line_loop, max) == 0x2'0000'0000ull &&
					rsx::dma_manager::index_emulation_bytes(primitive_type::triangle_fan, max) == (u64{max} - 2) * 6 &&
					rsx::dma_manager::index_emulation_bytes(primitive_type::quads, max) == u64{max / 4} * 12,
				"Index byte extent overflowed before guest range validation");
		refused([]
				{ metal4::validate_fault_transfer_range({0x1000, 0}, 0x1000); });
		for (auto bytes : {0ull, 0x1'0000'0000ull, 0x2'0000'0000ull, std::numeric_limits<unsigned long long>::max()})
			require(!rsx::dma_manager::fault_range_from_bytes(0, bytes), "Shared fault range accepted empty or wide length");
		for (auto address : {0u, 0x1234u, 0xffff'ffffu})
		{
			const auto range = rsx::dma_manager::fault_range_from_bytes(address, 1);
			require(range && range->start == address && range->end == address, "Single-byte shared fault range differs");
		}
		std::cout << "Actual ARM64 shared index writer:" << cases << " expansions,NEON/tails/u16 bytes/incomplete quads/last VM byte/canaries/wide overflow and invalid fault coverage passed;diagnostic-only standalone shims,no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
