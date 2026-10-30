#include "stdafx.h"

#include "Emu/Memory/vm.h"
#include "Common/BufferUtils.h"
#include "Core/RSXReservationLock.hpp"
#include "RSXOffload.h"
#include "RSXThread.h"

#include "Utilities/lockless.h"

#include <thread>
#include <exception>
#include "util/asm.hpp"

namespace rsx
{
	// Actual raw-source reservations must not survive native fault handoff.
	// Reacquire them BEFORE the VM guard and retrying memory instruction.
	struct host_source_reservation
	{
		inline static thread_local host_source_reservation* current = nullptr;
		host_source_reservation* previous;
		std::optional<rsx::reservation_lock<true, 1>> lock;
		u32 address, bytes;
		bool enabled;
		host_source_reservation(const void* source, u64 size, bool raw)
			: previous(current), address(raw ? vm::try_get_addr(source).first : 0), bytes(static_cast<u32>(size)),
			  enabled(raw && g_cfg.video.strict_rendering_mode && address)
		{
			if (enabled) lock.emplace(address, bytes, true);
			current = this;
		}
		~host_source_reservation() { current = previous; }
	};
	template <typename Store>
	static void tracked_host_store(const dma_manager& manager, void* destination, const void* source, u64 bytes, bool raw_source, Store&& store)
	{
#if defined(__APPLE__) && defined(__aarch64__)
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
		for (;;)
		{
			// Every retry runs real owner preparation again,without retaining
			// the previous source reservation,VM guard or unstarted receipt.
			bool retry = false;
			{
				auto completion = manager.prepare_native_host_write(destination, source, bytes);
				host_source_reservation reservation(source, bytes, raw_source);
				if (completion && !completion->start()) retry = true;
				else
				{
					std::invoke(std::forward<Store>(store));
					if (completion) completion->commit();
					return;
				}
			}
			if (retry && std::chrono::steady_clock::now() >= deadline)
				fmt::throw_exception("Native DMA VM contention exceeded the pre-instruction retry deadline");
			std::this_thread::yield();
		}
#else
		(void)manager;
		(void)destination;
		host_source_reservation reservation(source, bytes, raw_source);
		std::invoke(std::forward<Store>(store));
#endif
	}

	struct dma_manager::offload_thread
	{
		lf_queue<transport_packet> m_work_queue;
		dma_manager& m_owner;
		explicit offload_thread(dma_manager& owner) : m_owner(owner) {}
#if defined(__APPLE__) && defined(__aarch64__)
		mutable std::mutex m_enqueue_mutex;
		bool m_accepting = true;
		std::exception_ptr m_failure;
		void check_failure() const
		{
			std::lock_guard lock(m_enqueue_mutex);
			if (m_failure) std::rethrow_exception(m_failure);
		}
		void retire(std::exception_ptr failure) noexcept
		{
			{
				std::lock_guard lock(m_enqueue_mutex);
				m_accepting = false;
				if (failure && !m_failure) m_failure = std::move(failure);
			}
			// Admission is sealed before removing queued owners. Retire their
			// tickets outside the submission mutex, before publishing terminal.
			{ auto cancelled = pop_work(); }
			m_current_job = nullptr;
			m_processed_count = -1;
			m_processed_count.notify_all();
		}
#endif
		lf_queue_slice<transport_packet> pop_work()
		{
#if defined(__APPLE__) && defined(__aarch64__)
			// Pair packet publication with consumer acquisition explicitly,also
			// when the platform custom wide atomics are invisible to TSan.
			std::lock_guard lock(m_enqueue_mutex);
#endif
			return m_work_queue.pop_all();
		}

		atomic_t<u64> m_enqueued_count = 0;
		atomic_t<u64> m_processed_count = 0;
		transport_packet* m_current_job = nullptr;

		atomic_t<thread_base*> current_thread_ = nullptr;

		template <typename... Args>
		void enqueue(Args&&... args)
		{
#if defined(__APPLE__) && defined(__aarch64__)
			std::lock_guard lock(m_enqueue_mutex);
			if (m_failure) std::rethrow_exception(m_failure);
			if (!m_accepting) throw std::logic_error("Native DMA offloader no longer accepts work");
#endif
			m_enqueued_count++;
			// lf_queue constructs its item before publication. Allocation failure
			// must retire both the writer ticket and this queue count.
			struct rollback
			{
				atomic_t<u64>& count;
				bool published = false;
				~rollback() { if (!published) count--; }
			} transaction{m_enqueued_count};
			const bool was_empty = m_work_queue.push<false>(std::forward<Args>(args)...);
			transaction.published = true;
			if (was_empty) m_work_queue.notify(true);
		}

		void operator ()()
		{
#if defined(__APPLE__) && defined(__aarch64__)
			std::exception_ptr failure;
			struct finalizer
			{
				offload_thread& owner;
				std::exception_ptr& failure;
				~finalizer() { owner.retire(std::move(failure)); }
			} finish{*this, failure};
			try
			{
#endif
			if (!g_cfg.video.multithreaded_rsx)
			{
				// Abort if disabled
				return;
			}

			current_thread_.release(thread_ctrl::get_current());
			ensure(current_thread_.load());

			if (g_cfg.core.thread_scheduler != thread_scheduler_mode::os)
			{
				thread_ctrl::set_thread_affinity_mask(thread_ctrl::get_affinity_mask(thread_class::rsx));
			}

			while (thread_ctrl::state() != thread_state::aborting)
			{
				for (auto&& job : pop_work())
				{
#if defined(__APPLE__) && defined(__aarch64__)
					auto writer = job.writer.activate();
					ensure(writer);
#endif
					m_current_job = &job;

					switch (job.type)
					{
					case raw_copy:
					{
						tracked_host_store(m_owner, job.dst, job.src, job.length, true, [&] { std::memcpy(job.dst, job.src, job.length); });
						break;
					}
					case vector_copy:
					{
						tracked_host_store(m_owner, job.dst, job.opt_storage.data(), job.length, false, [&] { std::memcpy(job.dst, job.opt_storage.data(), job.length); });
						break;
					}
					case index_emulate:
					{
						tracked_host_store(m_owner, job.dst, nullptr, *dma_manager::index_emulation_bytes(static_cast<rsx::primitive_type>(job.aux_param0), job.length), false, [&]
						{
							write_index_array_for_non_indexed_non_native_primitive_to_buffer(static_cast<char*>(job.dst), static_cast<rsx::primitive_type>(job.aux_param0), job.length);
						});
						break;
					}
					case callback:
					{
						rsx::get_current_renderer()->renderctl(job.aux_param0, job.src);
						break;
					}
					default: fmt::throw_exception("Unreachable");
					}

					{
#if defined(__APPLE__) && defined(__aarch64__)
						// sync rechecks failure under this same mutex after observing
						// completion; host destinations are visible before it returns.
						std::lock_guard lock(m_enqueue_mutex);
#endif
						m_processed_count.release(m_processed_count + 1);
					}
				}

				m_current_job = nullptr;

				if (m_enqueued_count.load() == m_processed_count.load())
				{
					m_processed_count.notify_all();
					std::this_thread::yield();
				}
			}

#if defined(__APPLE__) && defined(__aarch64__)
			}
			catch (...) { failure = std::current_exception(); }
#else
			m_processed_count = -1;
			m_processed_count.notify_all();
#endif
		}

		static constexpr auto thread_name = "RSX Offloader"sv;
	};

	// initialization
	void dma_manager::init()
	{
#if defined(__APPLE__) && defined(__aarch64__)
		check_native_work();
		if (m_thread) throw std::logic_error("Native DMA offloader initialized twice");
#endif
		m_thread = std::make_shared<named_thread<offload_thread>>(*this);
	}

	// General transport
	void dma_manager::copy(void *dst, std::vector<u8>& src, u32 length) const
	{
#if defined(__APPLE__) && defined(__aarch64__)
		if (length > src.size()) throw std::out_of_range("DMA vector source is shorter than its byte extent");
		auto ticket = admit_native_write({src.data(), dst, length});
#endif
		if (length <= max_immediate_transfer_size || !g_cfg.video.multithreaded_rsx)
		{
#if defined(__APPLE__) && defined(__aarch64__)
			auto writer = ticket.activate();
			ensure(writer);
#endif
			tracked_host_store(*this, dst, src.data(), length, false, [&] { std::memcpy(dst, src.data(), length); });
		}
		else
		{
			m_thread->enqueue(
#if defined(__APPLE__) && defined(__aarch64__)
				std::move(ticket),
#endif
				dst, src, length);
		}
	}

	void dma_manager::copy(void *dst, void *src, u32 length) const
	{
#if defined(__APPLE__) && defined(__aarch64__)
		auto ticket = admit_native_write({src, dst, length});
#endif
		if (length <= max_immediate_transfer_size || !g_cfg.video.multithreaded_rsx)
		{
#if defined(__APPLE__) && defined(__aarch64__)
			auto writer = ticket.activate();
			ensure(writer);
#endif
			tracked_host_store(*this, dst, src, length, true, [&] { std::memcpy(dst, src, length); });
		}
		else
		{
			m_thread->enqueue(
#if defined(__APPLE__) && defined(__aarch64__)
				std::move(ticket),
#endif
				dst, src, length);
		}
	}

	// Vertex utilities
	void dma_manager::emulate_as_indexed(void *dst, rsx::primitive_type primitive, u32 count)
	{
#if defined(__APPLE__) && defined(__aarch64__)
		check_native_work();
#endif
		const auto bytes = index_emulation_bytes(primitive, count);
		ensure(bytes);
		if (!*bytes)
			return;

#if defined(__APPLE__) && defined(__aarch64__)
		auto ticket = admit_native_write({nullptr, dst, *bytes});
#endif

		if (!g_cfg.video.multithreaded_rsx)
		{
#if defined(__APPLE__) && defined(__aarch64__)
			auto writer = ticket.activate();
			ensure(writer);
#endif
			tracked_host_store(*this, dst, nullptr, *bytes, false, [&]
			{
				write_index_array_for_non_indexed_non_native_primitive_to_buffer(static_cast<char*>(dst), primitive, count);
			});
		}
		else
		{
			m_thread->enqueue(
#if defined(__APPLE__) && defined(__aarch64__)
				std::move(ticket),
#endif
				dst, primitive, count);
		}
	}

	// Backend callback
	void dma_manager::backend_ctrl(u32 request_code, void* args)
	{
#if defined(__APPLE__) && defined(__aarch64__)
		check_native_work();
#endif
		ensure(g_cfg.video.multithreaded_rsx);

#if defined(__APPLE__) && defined(__aarch64__)
		auto ticket = admit_native_write({nullptr, nullptr, 0});
#endif

		m_thread->enqueue(
#if defined(__APPLE__) && defined(__aarch64__)
			std::move(ticket),
#endif
			request_code, args);
	}

	// Synchronization
	bool dma_manager::is_current_thread() const
	{
		if (auto cpu = thread_ctrl::get_current(); cpu && m_thread)
		{
			return m_thread->current_thread_.load() == cpu;
		}

		return false;
	}

	bool dma_manager::sync() const
	{
		if (!m_thread) return true;
		auto& _thr = *m_thread;
#if defined(__APPLE__) && defined(__aarch64__)
		_thr.check_failure();
#endif

		if (_thr.m_enqueued_count.load() <= _thr.m_processed_count.load()) [[likely]]
		{
#if defined(__APPLE__) && defined(__aarch64__)
			_thr.check_failure(); // Terminal publication may have raced the first check.
#endif
			return true;
		}

		if (auto rsxthr = get_current_renderer(); rsxthr && rsxthr->is_current_thread())
		{
			while (_thr.m_enqueued_count.load() > _thr.m_processed_count.load())
			{
				// Recovery may begin AFTER sync entered this loop. Never wait
				// for a faulting instruction which needs this caller's service.
				if (m_mem_fault_flag) return false;
				rsxthr->on_semaphore_acquire_wait();
				utils::pause();
			}
		}
		else
		{
			while (_thr.m_enqueued_count.load() > _thr.m_processed_count.load())
			{
				if (m_mem_fault_flag || is_current_thread()) return false;
				utils::pause();
			}
		}

#if defined(__APPLE__) && defined(__aarch64__)
		_thr.check_failure();
#endif
		return true;
	}

	dma_manager::~dma_manager()
	{
#if defined(__APPLE__) && defined(__aarch64__)
		// Finish/cancel workers while their manager and callback owners exist.
		try { join(); } catch (...) {}
		m_thread.reset();
#endif
	}
	void dma_manager::join()
	{
#if defined(__APPLE__) && defined(__aarch64__)
		m_native_closed.store(true, std::memory_order_release);
		std::exception_ptr failure;
		try
		{
			auto guard = capture_native_writes();
			if (!guard) throw std::logic_error("Native DMA shutdown could not drain its producers");
			(void)sync();
		}
		catch (...) { failure = std::current_exception(); }
		if (m_thread) *m_thread = thread_state::aborting;
		if (failure) std::rethrow_exception(failure);
#else
		sync();
		if (m_thread) *m_thread = thread_state::aborting;
#endif
	}

	void dma_manager::set_mem_fault_flag()
	{
		ensure(is_current_thread()); // "Access denied"
		m_mem_fault_flag.release(true);
	}

	void dma_manager::clear_mem_fault_flag()
	{
		ensure(is_current_thread()); // "Access denied"
		m_mem_fault_flag.release(false);
	}

#if defined(__APPLE__) && defined(__aarch64__)
	metal4::host_writer_gate::ticket dma_manager::admit_native_write(metal4::host_writer_gate::transfer extent) const
	{
		// Recheck closure after ticket publication. If shutdown captured before
		// admission,this late ticket cancels without executing;otherwise its
		// pending count is included in the shutdown drain. Do not hold another
		// mutex across admit(),which may wait for an owner capture to finish.
		check_native_work();
		auto ticket = m_native_writers.admit(extent);
		if (!ticket) throw std::logic_error("Native DMA cannot admit work inside an exclusive capture");
		check_native_work();
		return ticket;
	}
	void dma_manager::check_native_work() const
	{
		if (m_native_closed.load(std::memory_order_acquire)) throw std::logic_error("Native DMA manager is closed");
		if (m_thread) m_thread->check_failure();
	}
	bool dma_manager::attach_native_host_write_handler(std::shared_ptr<const native_host_write_handler> handler)
	{
		if (!handler || !*handler) throw std::invalid_argument("Native DMA requires a live host-write handler");
		check_native_work();
		auto guard = capture_native_writes();
		if (!guard) throw std::logic_error("Native DMA handler installation requires producer exclusion");
		check_native_work();
		std::lock_guard lock(m_native_handler_mutex);
		if (m_native_handler) return false;
		m_native_handler = std::move(handler);
		return true;
	}
	bool dma_manager::detach_native_host_write_handler(const std::shared_ptr<const native_host_write_handler>& handler)
	{
		auto guard = capture_native_writes();
		if (!guard) throw std::logic_error("Native DMA handler removal requires producer exclusion");
		std::shared_ptr<const native_host_write_handler> retired;
		{
			std::lock_guard lock(m_native_handler_mutex);
			if (!handler || m_native_handler != handler) return false;
			retired.swap(m_native_handler);
		}
		return true;
	}
	std::optional<metal4::host_write_intent> dma_manager::prepare_native_host_write(void* destination, const void* source, u64 bytes) const
	{
		// Already admitted work must finish while join has closed new admission.
		if (m_thread) m_thread->check_failure();
		std::shared_ptr<const native_host_write_handler> handler;
		{
			std::lock_guard lock(m_native_handler_mutex);
			handler = m_native_handler;
		}
		if (handler) return std::invoke(*handler, destination, source, bytes);
		const auto renderer = get_current_renderer();
		return renderer ? renderer->on_native_host_write(destination, source, bytes) : std::nullopt;
	}
	std::shared_ptr<void> dma_manager::capture_native_writes() const
	{
		if (is_current_thread()) return {};
		auto captured = m_native_writers.acquire_waiting_faults([]
		{
			if (auto* renderer = get_current_renderer(); renderer && renderer->is_current_thread())
				renderer->on_semaphore_acquire_wait();
			else
				utils::pause();
		});
		if (!captured) return {};
		return std::make_shared<metal4::host_writer_gate::exclusion>(std::move(*captured));
	}
	void dma_manager::release_native_source_reservations() const noexcept
	{
		for (auto* scope = host_source_reservation::current; scope; scope = scope->previous) scope->lock.reset();
	}
	void dma_manager::resume_native_source_reservations() const
	{
		std::array<host_source_reservation*, 64> scopes{};
		std::size_t count = 0;
		for (auto* scope = host_source_reservation::current; scope; scope = scope->previous)
		{
			ensure(count < scopes.size());
			scopes[count++] = scope;
		}
		while (count)
		{
			auto& scope = *scopes[--count];
			if (scope.enabled && !scope.lock) scope.lock.emplace(scope.address, scope.bytes, true);
		}
	}
	std::optional<metal4::host_writer_gate::fault_pause> dma_manager::pause_native_writer_fault() const
	{
		return m_native_writers.pause_current_fault();
	}
	std::optional<utils::address_range32> dma_manager::get_native_fault_range(bool writing) const
	{
		const auto extent = m_native_writers.current_transfer();
		if (!extent) return {};
		const auto pointer = writing ? extent->destination : extent->source;
		if (!pointer) return {};
		const auto [address, mapped] = vm::try_get_addr(pointer);
		if (!mapped) return {};
		return fault_range_from_bytes(address, extent->bytes);
	}
#endif

	// Fault recovery
	utils::address_range32 dma_manager::get_fault_range(bool writing) const
	{
		ensure(is_current_thread());
#if defined(__APPLE__) && defined(__aarch64__)
		if (const auto range = get_native_fault_range(writing)) return *range;
#endif
		const auto m_current_job = ensure(m_thread->m_current_job);

		void *address = nullptr;
		u64 range = m_current_job->length;

		switch (m_current_job->type)
		{
		case raw_copy:
			address = (writing) ? m_current_job->dst : m_current_job->src;
			break;
		case vector_copy:
			ensure(writing);
			address = m_current_job->dst;
			break;
		case index_emulate:
			ensure(writing);
			address = m_current_job->dst;
			range = *ensure(index_emulation_bytes(static_cast<rsx::primitive_type>(m_current_job->aux_param0), m_current_job->length));
			break;
		default:
			fmt::throw_exception("Unreachable");
		}

		return *ensure(fault_range_from_bytes(vm::get_addr(address), range));
	}
}
