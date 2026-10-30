#pragma once

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <stdexcept>

namespace metal4
{
// Acquire under provider exclusion. Last-owner release may occur on any GPU
// completion thread and never enters the provider or the VM. Source ownership
// is weak: a retained mapping must not change the VM allocation's refcount.
class backing_pool final
{
	struct budget
	{
		std::atomic<std::size_t> bytes{0};
		const std::size_t limit;
		explicit budget(std::size_t limit) : limit(limit) {}
	};
	struct allocation
	{
		std::shared_ptr<budget> ledger;
		std::shared_ptr<void> mapping;
		std::size_t bytes = 0;
		explicit allocation(std::shared_ptr<budget> ledger) : ledger(std::move(ledger)) {}
		~allocation()
		{
			// Unmap before returning its charge to another acquiring thread.
			mapping.reset();
			ledger->bytes.fetch_sub(bytes, std::memory_order_release);
		}
	};
	struct entry
	{
		std::weak_ptr<void> source, mapping;
		std::size_t bytes;
	};
	std::shared_ptr<budget> m_budget;
	const std::size_t m_capacity;
	std::list<entry> m_entries;

  public:
	explicit backing_pool(std::size_t bytes, std::size_t capacity = 1024)
		: m_budget(std::make_shared<budget>(bytes)), m_capacity(capacity)
	{
		if (!bytes || !capacity) throw std::invalid_argument("Backing pool requires finite nonzero budgets");
	}
	[[nodiscard]] std::size_t retained_bytes() const { return m_budget->bytes.load(std::memory_order_acquire); }
	// Reverse the independent mapping identity without acquiring a strong VM
	// allocation handle. An expired original source has no live guest aliases.
	[[nodiscard]] std::weak_ptr<void> source_identity(const std::weak_ptr<void>& mapping) const noexcept
	{
		if (!mapping.expired())
			for (const auto& value : m_entries)
				if (!value.mapping.owner_before(mapping) && !mapping.owner_before(value.mapping)) return value.source;
		return {};
	}
	// Metadata lookup under the provider's exclusion;no mapping,allocation or
	// strong source ownership. A retained completion keeps its mapping alive.
	[[nodiscard]] bool matches(const std::shared_ptr<void>& source, const std::weak_ptr<void>& mapping) const noexcept
	{
		if (!source || mapping.expired()) return false;
		for (const auto& value : m_entries)
			if (!value.source.owner_before(source) && !source.owner_before(value.source) &&
				!value.mapping.owner_before(mapping) && !mapping.owner_before(value.mapping)) return true;
		return false;
	}
	[[nodiscard]] std::shared_ptr<void> acquire(const std::shared_ptr<void>& source, std::size_t bytes,
		const std::function<std::shared_ptr<void>()>& map)
	{
		if (!source || !bytes || !map) throw std::invalid_argument("Backing lease requires source,size and mapper");
		m_entries.remove_if([](const entry& value) { return value.mapping.expired(); });
		for (const auto& value : m_entries)
			if (!value.source.owner_before(source) && !source.owner_before(value.source))
			{
				if (value.bytes != bytes) throw std::logic_error("Backing allocation changed size without changing owner");
				if (auto existing = value.mapping.lock()) return existing;
			}
		if (m_entries.size() == m_capacity) throw std::length_error("Backing lease registration budget exhausted");
		auto result = std::make_shared<allocation>(m_budget);
		auto used = m_budget->bytes.load(std::memory_order_relaxed);
		for (;;)
		{
			if (bytes > m_budget->limit - used) throw std::length_error("Retained guest backing byte budget exhausted");
			if (m_budget->bytes.compare_exchange_weak(used, used + bytes, std::memory_order_acq_rel, std::memory_order_relaxed)) break;
		}
		result->bytes = bytes;
		result->mapping = map();
		if (!result->mapping) throw std::runtime_error("Backing mapper returned an absent independent mapping");
		auto owner = std::shared_ptr<void>(result, result->mapping.get());
		m_entries.push_back({source, owner, bytes});
		return owner;
	}
};
} // namespace metal4
