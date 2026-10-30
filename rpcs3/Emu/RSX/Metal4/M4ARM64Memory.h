#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace metal4
{
// Owned architectural input/output. X31 is deliberately absent: data operands
// name XZR/WZR, whereas an address or writeback operand can name SP.
struct arm64_memory_state
{
	std::array<std::uint64_t, 31> x{};
	std::uint64_t sp = 0, pc = 0;
	std::array<std::array<std::byte, 16>, 32> v{};
	std::uint32_t cpsr = 0, fpsr = 0, fpcr = 0;
	bool operator==(const arm64_memory_state&) const = default;
};

// Normal scalar/SIMD single and pair transfers, including literal loads.
// This is not an atomic/exclusive, MTE, pointer-authentication, unprivileged or
// SIMD-structure emulator. Those classes refuse before any memory side effect.
// Decode owns the exact original registers, effective address and store data;
// a fault's reported byte is never substituted for the effective address.
class arm64_memory_request final
{
	arm64_memory_state m_before;
	std::uint64_t m_address = 0, m_writeback = 0;
	unsigned m_bytes = 0, m_count = 1, m_first = 0, m_second = 0, m_base = 0;
	bool m_load = false, m_simd = false, m_signed = false, m_word = false, m_wback = false;
	[[nodiscard]] static std::int64_t signed_bits(std::uint32_t value, unsigned bits) noexcept
	{
		const auto sign = std::uint64_t{1} << (bits - 1);
		return std::bit_cast<std::int64_t>((std::uint64_t(value) ^ sign) - sign);
	}
	[[nodiscard]] static std::optional<std::uint64_t> add(std::uint64_t base, std::int64_t offset) noexcept
	{
		if (offset < 0)
		{
			const auto magnitude = std::uint64_t{0} - std::bit_cast<std::uint64_t>(offset);
			if (base < magnitude) return {};
			return base - magnitude;
		}
		if (std::uint64_t(offset) > std::numeric_limits<std::uint64_t>::max() - base) return {};
		return base + std::uint64_t(offset);
	}
	[[nodiscard]] std::uint64_t data(unsigned reg) const noexcept { return reg == 31 ? 0 : m_before.x[reg]; }
	arm64_memory_request() = default;

  public:
	[[nodiscard]] bool writing() const noexcept { return !m_load; }
	[[nodiscard]] std::uint64_t address() const noexcept { return m_address; }
	[[nodiscard]] unsigned size() const noexcept { return m_bytes * m_count; }
	[[nodiscard]] bool matches_fault(std::uint64_t address, bool writing) const noexcept
	{
		return writing == this->writing() && address >= m_address && address - m_address < size();
	}
	[[nodiscard]] static std::optional<arm64_memory_request> decode(std::uint32_t instruction, const arm64_memory_state& state)
	{
		if ((state.pc & 3) || state.pc > std::numeric_limits<std::uint64_t>::max() - 4) return {};
		arm64_memory_request r;
		r.m_before = state;
		r.m_first = instruction & 31;
		r.m_base = (instruction >> 5) & 31;
		r.m_simd = (instruction >> 26) & 1;
		const auto size = instruction >> 30;
		const auto opc = (instruction >> 22) & 3;
		std::int64_t offset = 0;
		bool literal = false, post = false;
		if ((instruction & 0x3b000000u) == 0x18000000u)
		{
			literal = true;
			r.m_load = true;
			if (size == 3) return {}; // PRFM or reserved SIMD literal.
			if (r.m_simd) r.m_bytes = 4u << size;
			else
			{
				r.m_bytes = size == 1 ? 8 : 4;
				r.m_signed = size == 2;
				r.m_word = size == 0;
			}
			offset = signed_bits((instruction >> 5) & 0x7ffff, 19) * 4;
		}
		else if ((instruction & 0x3a000000u) == 0x28000000u)
		{
			r.m_count = 2;
			r.m_second = (instruction >> 10) & 31;
			r.m_load = (instruction >> 22) & 1;
			const auto mode = (instruction >> 23) & 3;
			r.m_wback = mode == 1 || mode == 3;
			post = mode == 1;
			if (r.m_simd)
			{
				if (size == 3) return {};
				r.m_bytes = 4u << size;
			}
			else
			{
				if (size == 3 || (size == 1 && (!r.m_load || mode == 0))) return {};
				r.m_bytes = size == 2 ? 8 : 4;
				r.m_signed = size == 1; // LDPSW, never a store or LDNP.
				r.m_word = size == 0;
			}
			// Constrained-unpredictable overlapping destinations/writeback refuse.
			if (r.m_load && r.m_first == r.m_second) return {};
			offset = signed_bits((instruction >> 15) & 127, 7) * r.m_bytes;
		}
		else if ((instruction & 0x3b000000u) == 0x39000000u ||
			(instruction & 0x3b000000u) == 0x38000000u)
		{
			if (r.m_simd)
			{
				const auto scale = size + ((opc & 2) ? 4u : 0u);
				if (scale > 4) return {};
				r.m_bytes = 1u << scale;
				r.m_load = opc & 1;
			}
			else
			{
				if ((size == 3 && opc >= 2) || (size == 2 && opc == 3)) return {};
				r.m_bytes = 1u << size;
				r.m_load = opc != 0;
				r.m_signed = opc >= 2;
				r.m_word = opc == 3 || (opc == 1 && size != 3);
			}
			if (instruction & (1u << 24)) offset = ((instruction >> 10) & 4095) * r.m_bytes;
			else if (instruction & (1u << 21))
			{
				if ((instruction & 0xc00) != 0x800) return {}; // LSE atomics/other classes.
				const auto option = (instruction >> 13) & 7;
				if (!(option & 2)) return {};
				const auto rm = (instruction >> 16) & 31;
				auto value = r.data(rm);
				if (!(option & 1)) value = static_cast<std::uint32_t>(value);
				if (option == 6) value = std::bit_cast<std::uint64_t>(signed_bits(static_cast<std::uint32_t>(value), 32));
				const auto shift = (instruction & (1u << 12)) ? std::countr_zero(r.m_bytes) : 0;
				// Signed register offsets use architectural modulo shifts. Refuse
				// wrapped final effective addresses rather than guessing a host VA.
				offset = std::bit_cast<std::int64_t>(value << shift);
			}
			else
			{
				const auto mode = (instruction >> 10) & 3;
				if (mode == 2) return {}; // LDTR/STTR have a different permission regime.
				r.m_wback = mode != 0;
				post = mode == 1;
				offset = signed_bits((instruction >> 12) & 511, 9);
			}
		}
		else return {};
		if (!literal && r.m_base == 31 && (state.sp & 15)) return {};
		if (r.m_wback && !r.m_simd && r.m_base != 31 &&
			(r.m_base == r.m_first || (r.m_count == 2 && r.m_base == r.m_second))) return {};
		const auto base = literal ? state.pc : (r.m_base == 31 ? state.sp : state.x[r.m_base]);
		const auto adjusted = add(base, offset);
		if (!adjusted) return {};
		r.m_writeback = *adjusted;
		r.m_address = post ? base : *adjusted;
		if (!r.size() || r.m_address > std::numeric_limits<std::uint64_t>::max() - (r.size() - 1)) return {};
		return r;
	}
	[[nodiscard]] std::array<std::byte, 32> store_bytes() const noexcept
	{
		std::array<std::byte, 32> bytes{};
		if (m_load) return bytes;
		for (unsigned i = 0; i < m_count; ++i)
		{
			const auto reg = i ? m_second : m_first;
			if (m_simd) std::memcpy(bytes.data() + i * m_bytes, m_before.v[reg].data(), m_bytes);
			else
			{
				const auto value = data(reg);
				for (unsigned b = 0; b < m_bytes; ++b) bytes[i * m_bytes + b] = std::byte(value >> (b * 8));
			}
		}
		return bytes;
	}
	// Pure owned reply, computed before register publication to the original
	// stopped thread. Memory transactions/reservations are the caller's duty.
	[[nodiscard]] arm64_memory_state complete(std::span<const std::byte> loaded) const
	{
		if (loaded.size() != (m_load ? size() : 0)) throw std::invalid_argument("ARM memory reply has the wrong exact byte extent");
		auto result = m_before;
		if (m_load)
			for (unsigned i = 0; i < m_count; ++i)
			{
				const auto reg = i ? m_second : m_first;
				if (m_simd)
				{
					result.v[reg] = {};
					std::memcpy(result.v[reg].data(), loaded.data() + i * m_bytes, m_bytes);
				}
				else if (reg != 31)
				{
					std::uint64_t value = 0;
					for (unsigned b = 0; b < m_bytes; ++b) value |= std::uint64_t(std::to_integer<unsigned>(loaded[i * m_bytes + b])) << (b * 8);
					if (m_signed && (value & (std::uint64_t{1} << (m_bytes * 8 - 1)))) value |= ~std::uint64_t{0} << (m_bytes * 8);
					result.x[reg] = m_word ? static_cast<std::uint32_t>(value) : value;
				}
			}
		if (m_wback)
		{
			if (m_base == 31) result.sp = m_writeback;
			else result.x[m_base] = m_writeback;
		}
		result.pc += 4;
		return result;
	}
};
} // namespace metal4
