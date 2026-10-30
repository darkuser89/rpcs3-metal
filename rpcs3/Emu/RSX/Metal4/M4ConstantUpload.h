#pragma once

#include "M4Device.h"
#include "M4FragmentProgram.h"

namespace metal4
{
struct constant_upload_statistics
{
	std::size_t entries = 0, capacity = 0, payload_bytes = 0, payload_budget = 0;
	std::uint64_t created = 0, reused = 0, evicted = 0, bypassed = 0, uploaded_bytes = 0;
	std::uint64_t instruction_checks = 0;
};
// Render-thread, context-owned immutable constant versions. A hash only selects
// candidates; every word must match. Payload budget counts retained constant
// bytes; CPU keys duplicate that payload, and native allocations also use the
// context heap ledger. Returned buffers are read-only shader inputs: never
// modify their contents. Encoding retains them through context::use_resource.
class constant_upload_cache final
{
	struct implementation;
	std::unique_ptr<implementation> m_impl;
  public:
	explicit constant_upload_cache(context& ctx, std::size_t capacity = 64, std::size_t payload_budget = 1u << 20);
	~constant_upload_cache();
	constant_upload_cache(const constant_upload_cache&) = delete;
	constant_upload_cache& operator=(const constant_upload_cache&) = delete;
	[[nodiscard]] id<MTLBuffer> words(std::span<const std::byte> bytes);
	// Exact immutable byte content/length, including non-register-sized packed
	// streams. Empty inputs bind no buffer; over-budget inputs bypass retention.
	[[nodiscard]] id<MTLBuffer> packed_bytes(std::span<const std::byte> bytes);
	[[nodiscard]] id<MTLBuffer> fragment(const decoded_fragment_program& compiled, std::span<const std::byte> bytes);
	[[nodiscard]] constant_upload_statistics statistics() const noexcept;
	void clear();
};
} // namespace metal4
