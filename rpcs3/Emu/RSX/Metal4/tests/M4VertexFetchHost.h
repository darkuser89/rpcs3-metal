#pragma once

#include <bit>
#include <cstdint>

// Minimal CPU type adapter for the actual MSL kernel body. This exercises its
// indexing/byte decoding without creating a Metal device. It isn't an emulator
// for Metal arithmetic, address spaces, execution or GPU half denormals.
namespace fetch_host
{
using uint = std::uint32_t;
using ulong = std::uint64_t;
using uchar = unsigned char;
using ushort = unsigned short;
using half = _Float16;

struct alignas(8) uint2 { uint x, y; };
struct uint4
{
	uint x, y, z, w;
	explicit uint4(uint a) : x(a), y(a), z(a), w(a) {}
	uint& operator[](uint i)
	{
		switch (i) { case 0: return x; case 1: return y; case 2: return z; default: return w; }
	}
};
struct float4
{
	float x, y, z, w;
	float4() = default;
	float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
	explicit float4(uint4 a)
		: x(static_cast<float>(a.x)), y(static_cast<float>(a.y)), z(static_cast<float>(a.z)), w(static_cast<float>(a.w)) {}
	float& operator[](uint i)
	{
		switch (i) { case 0: return x; case 1: return y; case 2: return z; default: return w; }
	}
	float4& operator+=(float a) { x += a; y += a; z += a; w += a; return *this; }
};
inline float4 operator/(float4 a, float b) { return {a.x / b, a.y / b, a.z / b, a.w / b}; }
template <typename T, typename U> T as_type(U value) { return std::bit_cast<T>(value); }
static_assert(sizeof(uint4) == 16 && sizeof(float4) == 16 && sizeof(half) == 2);

// CMake extracts this header from vertex_fetch_kernels and removes only the
// Metal standard library include, namespace and resource attributes.
#define device
#define constant const
#define kernel
#include "M4VertexFetchTestKernel.h"
#undef kernel
#undef constant
#undef device
static_assert(sizeof(rsx_fetch_parameters) == 240);
}
