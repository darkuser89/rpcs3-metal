#pragma once

#include <cstddef>
#include <memory>
#include <utility>

namespace metal4
{
class host_write_journal;
class host_store_transaction;
// Fixed-size,no-allocation completion handle for Shared headers compiled with
// exceptions disabled. The journal owns the typed state and noexcept callback.
class host_write_intent final
{
	friend class host_write_journal;
	friend class host_store_transaction;
	std::shared_ptr<void> m_owner;
	std::shared_ptr<const void> m_source;
	std::size_t m_slot;
	void (*m_finish)(void*, std::size_t, bool) noexcept;
	bool (*m_start)(void*) = nullptr;
	bool m_started = false;
	host_write_intent(std::shared_ptr<void> owner, std::size_t slot, decltype(m_finish) finish, decltype(m_start) start = nullptr)
		: m_owner(std::move(owner)), m_slot(slot), m_finish(finish), m_start(start) {}
	void finish(bool completed) noexcept
	{
		auto owner = std::exchange(m_owner, {});
		if (owner) m_finish(owner.get(), m_slot, completed && (!m_start || m_started));
		m_source.reset();
	}
  public:
	host_write_intent(host_write_intent&& other) noexcept
		: m_owner(std::exchange(other.m_owner, {})), m_source(std::move(other.m_source)), m_slot(other.m_slot), m_finish(other.m_finish),
		  m_start(other.m_start), m_started(other.m_started) {}
	host_write_intent(const host_write_intent&) = delete;
	host_write_intent& operator=(const host_write_intent&) = delete;
	// An exception may follow a partial memcpy:destruction preserves ambiguity.
	~host_write_intent() { finish(false); }
	// Staged native transactions acquire mapping exclusion after the caller's
	// source reservation and BEFORE the actual instruction. Returns false on contention;may throw on invalid mapping;
	// Shared headers need no exception syntax or exception-enabled compilation.
	[[nodiscard]] bool start()
	{
		if (m_owner && m_start && !m_started) m_started = m_start(m_owner.get());
		return !m_start || m_started;
	}
	void commit() noexcept { finish(true); }
	// Source-only guest -> host copy:retain its independently mapped bytes
	// through the instruction without inventing a guest destination revision.
	[[nodiscard]] static host_write_intent keep_source(std::shared_ptr<const void> source)
	{
		host_write_intent result({}, 0, nullptr);
		result.m_source = std::move(source);
		return result;
	}
};
} // namespace metal4
