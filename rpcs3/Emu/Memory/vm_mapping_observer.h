#pragma once

#include <cstdint>

namespace vm
{
struct writer_lock;
struct backing_segment;
class block_t;
class native_mapping_delivery;

// Borrowed only for a productive mapping callback under the process writer
// lock. It permits metadata lookup without recursively acquiring that lock.
// Do not retain this view or returned allocation handles after the callback.
class mapping_change_view final
{
	friend class native_mapping_delivery;
	const block_t* const m_block;
	explicit mapping_change_view(const block_t* block) : m_block(block) {}

  public:
	mapping_change_view(const mapping_change_view&) = delete;
	mapping_change_view& operator=(const mapping_change_view&) = delete;
	[[nodiscard]] backing_segment lookup(std::uint32_t address, std::uint32_t size,
		std::uint8_t required_flags) const noexcept;
};

// The VM writer lock excludes observer installation/removal for the entire
// paired notification. Callbacks must not allocate, throw, wait or recursively
// enter VM; the borrowed view's read-only metadata lookup is permitted.
// After mutation,an owned native observer may synchronously reapply its OS
// protection overlay to the CPU mapping. It must not change VM metadata or
// guest bytes;failure to restore protection must terminate before VM resumes.
struct mapping_change_observer
{
	virtual void on_mapping_change_begin(std::uint32_t address, std::uint32_t size) noexcept = 0;
	virtual void on_mapping_change_end(std::uint32_t address, std::uint32_t size) noexcept = 0;
	// Default forwarding preserves existing revision-only observers. New
	// observers may inspect the old/new allocation metadata in these callbacks;
	// no provider entry, allocation, external wait or VM/byte mutation is permitted.
	// An owned OS protection overlay may be reapplied in the end callback only.
	virtual void on_mapping_change_begin_locked(const mapping_change_view&, std::uint32_t address, std::uint32_t size) noexcept
	{
		on_mapping_change_begin(address, size);
	}
	virtual void on_mapping_change_end_locked(const mapping_change_view&, std::uint32_t address, std::uint32_t size) noexcept
	{
		on_mapping_change_end(address, size);
	}

  protected:
	~mapping_change_observer() = default;
};

// One native revision domain owns the process VM. A second installation is
// refused without replacing the live owner. The caller retains the observer
// until successful detach under the same writer exclusion.
[[nodiscard]] bool attach_mapping_observer(mapping_change_observer& observer, const writer_lock&) noexcept;
[[nodiscard]] bool detach_mapping_observer(mapping_change_observer& observer, const writer_lock&) noexcept;
} // namespace vm
