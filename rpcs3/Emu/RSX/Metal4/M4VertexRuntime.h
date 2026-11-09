#pragma once

#include <string_view>

namespace metal4
{

inline constexpr std::string_view vertex_runtime = R"metal(
#include <metal_stdlib>
using namespace metal;

struct rsx_vertex_context
{
    float4x4 scale_offset;
    uint vertex_base, vertex_count, constant_count, branch_bits;
    float point_size, z_near, z_far;
    uint output_mask;
    uint two_sided, instruction_budget, geometry_count, predicate;
    float clip_factors[6];
    uint clip_enable, flat_count;
};
struct rsx_vertex_reference { uint source_vertex; uint flat_vertex; };
struct rsx_vertex_registers
{
    float4 dst[16];
    float4 temp[64];
    int4 address[2];
    float4 condition[2];
    uint fault;
};

inline rsx_vertex_registers rsx_initialize()
{
    rsx_vertex_registers r;
    for (uint i = 0; i < 16; ++i) r.dst[i] = float4(0, 0, 0, 1);
    for (uint i = 0; i < 64; ++i) r.temp[i] = float4(0);
    for (uint i = 0; i < 2; ++i) { r.address[i] = int4(0); r.condition[i] = float4(0); }
    r.fault = 0;
    return r;
}
inline float4 rsx_attribute(device const float4* attributes, constant rsx_vertex_context& ctx,
    uint source_vertex, long slot, thread uint& fault)
{
    if (source_vertex < ctx.vertex_base || source_vertex - ctx.vertex_base >= ctx.vertex_count || slot < 0 || slot >= 16)
    {
        fault |= 1u;
        return float4(0, 0, 0, 1);
    }
    return attributes[ulong(source_vertex - ctx.vertex_base) * 16 + ulong(slot)];
}
inline float4 rsx_constant(device const float4* constants, constant rsx_vertex_context& ctx,
    long index, thread uint& fault)
{
    if (index < 0 || ulong(index) >= ctx.constant_count)
    {
        fault |= 2u;
        return float4(0);
    }
    return constants[ulong(index)];
}
inline float4 rsx_set_compare(float4 left, float4 right, uint mode)
{
    const uint4 a = as_type<uint4>(left), b = as_type<uint4>(right);
    const uint4 a_magnitude = a & 0x7fffffffu, b_magnitude = b & 0x7fffffffu;
    const bool4 a_nan = a_magnitude > 0x7f800000u, b_nan = b_magnitude > 0x7f800000u;
    const bool4 both_zero = (a_magnitude | b_magnitude) == 0u;
    // Ordered integer keys retain the magnitude of subnormal operands.
    // Treat the two zero signs as equal; NaNs propagate as in NV VP2.
    const uint4 a_key = select(a | 0x80000000u, ~a, (a >> 31u) != 0u);
    const uint4 b_key = select(b | 0x80000000u, ~b, (b >> 31u) != 0u);
    const bool4 equal = (a == b) | both_zero;
    const bool4 less = (a_key < b_key) & !both_zero;
    const bool4 greater = (a_key > b_key) & !both_zero;
    bool4 result = false;
    switch (mode)
    {
    case 1: result = less; break;
    case 2: result = equal; break;
    case 3: result = less | equal; break;
    case 4: result = greater; break;
    case 5: result = !equal; break;
    case 6: result = greater | equal; break;
    }
    return as_type<float4>(select(as_type<uint4>(float4(result)), select(b, a, a_nan), a_nan | b_nan));
}
inline bool4 rsx_condition(float4 value, uint mode)
{
    const uint4 raw = as_type<uint4>(value), magnitude = raw & 0x7fffffffu;
    const bool4 zero = magnitude == 0u, ordered = magnitude <= 0x7f800000u;
    const bool4 negative = (raw >> 31u) != 0u;
    const bool4 less = negative & !zero & ordered, greater = !negative & !zero & ordered;
    // Classify register bits so copied subnormals do not become EQ merely
    // because the floating-point comparison ALU flushes them to zero.
    switch (mode)
    {
    case 1: return less;
    case 2: return zero;
    case 3: return less | zero;
    case 4: return greater;
    case 5: return !zero;
    case 6: return greater | zero;
    case 7: return bool4(true);
    default: return bool4(false);
    }
}
inline void rsx_store_output(thread rsx_vertex_registers& r, long index, float4 value, bool4 mask)
{
    if (!any(mask)) return;
    if (index < 0 || index >= 16) { r.fault |= 128u; return; }
    r.dst[uint(index)] = select(r.dst[uint(index)], value, mask);
}
inline float4 rsx_extremum(float4 left, float4 right, bool maximum)
{
    const uint4 a = as_type<uint4>(left), b = as_type<uint4>(right);
    const uint4 compared = as_type<uint4>(rsx_set_compare(left, right, maximum ? 4u : 1u));
    uint4 result = select(b, a, compared == 0x3f800000u);
    // Commutative zero selection: MIN prefers -0, MAX prefers +0.
    const bool4 both_zero = ((a | b) & 0x7fffffffu) == 0u;
    result = select(result, (maximum ? a & b : a | b) & 0x80000000u, both_zero);
    return as_type<float4>(select(result, compared, (compared & 0x7fffffffu) > 0x7f800000u));
}
inline float4 rsx_floor(float4 value)
{
    const uint4 raw = as_type<uint4>(value), magnitude = raw & 0x7fffffffu;
    const uint4 small = select(uint4(0), uint4(0xbf800000u), (raw >> 31u) != 0u);
    uint4 result = select(as_type<uint4>(floor(value)), small, (magnitude != 0u) & (magnitude < 0x00800000u));
    // Floor preserves signed zero and nonfinite classes explicitly.
    return as_type<float4>(select(result, raw, (magnitude == 0u) | (magnitude >= 0x7f800000u)));
}
inline float4 rsx_fraction(float4 value)
{
    const uint4 raw = as_type<uint4>(value), magnitude = raw & 0x7fffffffu;
    // FRC stays below1 even when x-floor(x) rounds up for negative tiny x.
    const uint4 small = select(raw, uint4(0x3f7fffffu), (raw >> 31u) != 0u);
    uint4 result = select(as_type<uint4>(fract(value)), small, (magnitude != 0u) & (magnitude < 0x00800000u));
    result = select(result, uint4(0), magnitude == 0u);
    result = select(result, uint4(0x7fc00000u), magnitude == 0x7f800000u);
    return as_type<float4>(select(result, raw, magnitude > 0x7f800000u));
}
inline float4 rsx_sign(float4 value)
{
    const uint4 raw = as_type<uint4>(value), magnitude = raw & 0x7fffffffu;
    uint4 result = 0x3f800000u | (raw & 0x80000000u);
    result = select(result, uint4(0), magnitude == 0u);
    return as_type<float4>(select(result, raw, magnitude > 0x7f800000u));
}
inline int4 rsx_address(float4 value, thread uint& fault)
{
    if (!all(isfinite(value)) || any(value < -2147483648.0f) || any(value >= 2147483648.0f))
    {
        fault |= 32u;
        return int4(0);
    }
    // Match the current Vulkan decoder's conversion. NV ARL's documented
    // floor rule differs for negative fractions; RSX hardware parity is open.
    return int4(value);
}
inline int4 rsx_address_add(int4 value, thread uint& fault)
{
    // Snapshot both pairs before a possibly overlapping address write.
    // Widen before addition: neither signed overflow nor float rounding
    // may alter a valid integer address (including INT_MAX).
    const long2 sum = long2(value.xy) + long2(value.zw);
    if (any(sum < -2147483648l) || any(sum > 2147483647l))
    {
        fault |= 32u;
        return int4(0);
    }
    return int4(int2(sum), int2(sum));
}
inline float rsx_power2_integral(float exponent)
{
    if (exponent > 127.0f) return INFINITY;
    if (exponent < -149.0f) return 0.0f;
    const int e = int(exponent);
    return as_type<float>(e >= -126 ? uint(e + 127) << 23u : 1u << uint(e + 149));
}
inline float rsx_scale_positive(float normal, int power)
{
    // Internal positive-normal input. Adjust its representation instead of
    // multiplying by a scale that could itself flush or overflow.
    const uint bits = as_type<uint>(normal);
    const int exponent = int(bits >> 23u) + power;
    if (exponent >= 255) return INFINITY;
    if (exponent >= 1) return as_type<float>((uint(exponent) << 23u) | (bits & 0x007fffffu));
    const uint shift = uint(1 - exponent);
    if (shift > 24u) return 0.0f;
    const uint significand = (bits & 0x007fffffu) | 0x00800000u;
    const uint remainder = significand & ((1u << shift) - 1u);
    uint result = significand >> shift;
    const uint halfway = 1u << (shift - 1u);
    result += uint(remainder > halfway || (remainder == halfway && (result & 1u)));
    return as_type<float>(result);
}
inline float rsx_exp_subnormal(float integral, float fraction)
{
    if (integral < -150.0f) return 0.0f;
    return rsx_scale_positive(exp2(fraction), int(integral));
}
inline float4 rsx_exp_legacy(float value)
{
    const uint raw = as_type<uint>(value), magnitude = raw & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return float4(value, value, value, 1);
    if (magnitude == 0x7f800000u)
        return (raw >> 31u) ? float4(0, 0, 0, 1) : float4(INFINITY, 0, INFINITY, 1);
    // Preserve the fractional output even when native arithmetic flushes
    // a subnormal input. A negative subnormal has floor -1, fraction 1.
    if (magnitude && magnitude < 0x00800000u)
        return (raw >> 31u) ? float4(0.5f, 1, 1, 1) : float4(1, value, 1, 1);
    const float integral = floor(value);
    const float fraction = value - integral;
    return float4(rsx_power2_integral(integral), fraction,
        value < -126.0f ? rsx_exp_subnormal(integral, fraction) : exp2(value), 1);
}
inline float4 rsx_saturate_ordered(float4 value)
{
    const uint4 raw = as_type<uint4>(value), magnitude = raw & 0x7fffffffu;
    const bool4 negative = (raw >> 31u) != 0u;
    // Integer classification preserves NaNs and signed zero, while a
    // negative subnormal still clamps even on a flush-to-zero ALU.
    const bool4 ordered = magnitude <= 0x7f800000u;
    return select(select(value, float4(0), negative & (magnitude != 0u) & ordered),
        float4(1), !negative & (magnitude > 0x3f800000u) & ordered);
}
inline float rsx_normalized_mantissa(uint magnitude, thread int& exponent)
{
    // Internal finite, nonzero magnitude; normalize subnormal bits before
    // any float operation can flush the input.
    uint fraction = magnitude & 0x007fffffu;
    exponent = int(magnitude >> 23u) - 127;
    if (magnitude < 0x00800000u)
    {
        const uint shift = clz(fraction) - 8u;
        fraction <<= shift;
        exponent = -126 - int(shift);
    }
    return as_type<float>(0x3f800000u | (fraction & 0x007fffffu));
}
inline float4 rsx_log_legacy(float value)
{
    const uint magnitude = as_type<uint>(value) & 0x7fffffffu;
    if (!magnitude) return float4(-INFINITY, 1, -INFINITY, 1);
    if (magnitude >= 0x7f800000u)
    {
        const float v = as_type<float>(magnitude);
        return magnitude == 0x7f800000u ? float4(v, 1, v, 1) : float4(v, v, v, 1);
    }
    // Derive floor(log2(abs(value))) from the representation. Rounding
    // log2 just below a power of two must not advance the exponent.
    int exponent;
    const float mantissa = rsx_normalized_mantissa(magnitude, exponent);
    const float logarithm = magnitude >= 0x00800000u ? log2(as_type<float>(magnitude)) : float(exponent) + log2(mantissa);
    return float4(float(exponent), mantissa, logarithm, 1);
}
inline float rsx_reciprocal(float value)
{
    const uint raw = as_type<uint>(value), magnitude = raw & 0x7fffffffu, sign = raw & 0x80000000u;
    if (magnitude > 0x7f800000u) return value;
    if (!magnitude) return as_type<float>(sign | 0x7f800000u);
    if (magnitude == 0x7f800000u) return as_type<float>(sign);
    int exponent;
    const float mantissa = rsx_normalized_mantissa(magnitude, exponent);
    return as_type<float>(as_type<uint>(rsx_scale_positive(1.0f / mantissa, -exponent)) | sign);
}
inline float rsx_rcp_clamped(float value)
{
    const uint raw = as_type<uint>(value), magnitude = raw & 0x7fffffffu, sign = raw & 0x80000000u;
    if (magnitude > 0x7f800000u) return value;
    if (magnitude < 0x1f800000u) return as_type<float>(sign | 0x5f800000u);
    if (magnitude > 0x5f800000u) return as_type<float>(sign | 0x1f800000u);
    const uint reciprocal = as_type<uint>(1.0f / value) & 0x7fffffffu;
    return as_type<float>(sign | clamp(reciprocal, 0x1f800000u, 0x5f800000u));
}
inline float rsx_rsqrt(float value)
{
    const uint magnitude = as_type<uint>(value) & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return value;
    if (!magnitude) return INFINITY;
    if (magnitude == 0x7f800000u) return 0.0f;
    int exponent;
    const float mantissa = rsx_normalized_mantissa(magnitude, exponent);
    const int half_exponent = exponent >= 0 ? exponent / 2 : -((1 - exponent) / 2);
    const float adjusted = (exponent - 2 * half_exponent) ? mantissa * 2.0f : mantissa;
    return rsx_scale_positive(rsqrt(adjusted), -half_exponent);
}
inline float rsx_log2_scalar(float value)
{
    const uint raw = as_type<uint>(value), magnitude = raw & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return value;
    if (!magnitude) return -INFINITY;
    if (raw >> 31u) return as_type<float>(0x7fc00000u);
    return rsx_log_legacy(value).z;
}
inline float rsx_exp2_scalar(float value)
{
    return rsx_exp_legacy(value).z;
}
inline float4 rsx_lit(float4 value)
{
    // Clamp nonpositive inputs to positive zero, without flushing a
    // positive subnormal through a native floating-point max operation.
    const float diffuse = rsx_condition(float4(value.x), 3u).x ? 0.0f : value.x;
    const float base = rsx_condition(float4(value.y), 3u).x ? 0.0f : value.y;
    const float limit = 127.99609375f; // 128 - 1/256, NV vertex-program definition.
    float exponent = value.w;
    if (exponent < -limit) exponent = -limit;
    else if (exponent > limit) exponent = limit;
    if (!rsx_condition(float4(diffuse), 4u).x) return float4(1, diffuse, 0, 1);
    const uint base_bits = as_type<uint>(base) & 0x7fffffffu;
    const uint exponent_bits = as_type<uint>(exponent);
    float specular;
    // Preserve the power identities before any log or multiply: 0^0,
    // NaN^0 and 1^NaN all yield 1 in the stated renderer model.
    if (!(exponent_bits & 0x7fffffffu) || base_bits == 0x3f800000u) specular = 1.0f;
    else if (base_bits > 0x7f800000u) specular = base;
    else if ((exponent_bits & 0x7fffffffu) > 0x7f800000u) specular = exponent;
    else if (!base_bits) specular = (exponent_bits >> 31u) ? INFINITY : 0.0f;
    else if (base_bits == 0x7f800000u) specular = (exponent_bits >> 31u) ? 0.0f : INFINITY;
    else if (exponent == 1.0f) specular = base;
    else if (exponent == -1.0f) specular = rsx_reciprocal(base);
    else specular = rsx_exp2_scalar(exponent * rsx_log2_scalar(base));
    return float4(1, diffuse, specular, 1);
}
inline float4 rsx_zclip(float4 pos, float near_plane, float far_plane)
{
    // Float-only fallback used by the Vulkan path on Apple GPUs. Metal's
    // native vertex stage has no float64 path; exact depth parity remains open.
    float d = pos.z / pos.w;
    if (d < 0 && d >= near_plane) d = 0;
    else if (d > 1 && d <= far_plane)
        d = min(1.0f, 0.99f + 0.01f * (pos.z - near_plane) / (far_plane - near_plane));
    else return pos;
    return float4(pos.xy, d * pos.w, pos.w);
}
)metal";

}
