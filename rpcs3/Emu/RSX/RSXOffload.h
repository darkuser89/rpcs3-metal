#pragma once

#include "util/types.hpp"
#include "Utilities/address_range.h"
#include "gcm_enums.h"

#include <optional>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__)
#include "Metal4/M4HostWriterGate.h"
#include "Metal4/M4HostWriteIntent.h"
#include <atomic>
#include <functional>
#include <mutex>
#endif

template <typename T>
class named_thread;

namespace rsx
{
	class dma_manager
	{
#if defined(__APPLE__) && defined(__aarch64__)
	public:
		using native_host_write_handler = std::function<std::optional<metal4::host_write_intent>(void*, const void*, u64)>;
	private:
#endif
		enum op
		{
			raw_copy = 0,
			vector_copy = 1,
			index_emulate = 2,
			callback = 3
		};

		struct transport_packet
		{
#if defined(__APPLE__) && defined(__aarch64__)
			metal4::host_writer_gate::ticket writer;
			template <typename... Args>
			transport_packet(metal4::host_writer_gate::ticket ticket, Args&&... args)
				: transport_packet(std::forward<Args>(args)...)
			{
				writer = std::move(ticket);
			}
#endif
			op type{};
			std::vector<u8> opt_storage{};
			void* src{};
			void* dst{};
			u32 length{};
			u32 aux_param0{};
			u32 aux_param1{};

			transport_packet(void *_dst, void *_src, u32 len)
				: type(op::raw_copy), src(_src), dst(_dst), length(len)
			{}

			transport_packet(void *_dst, std::vector<u8>& _src, u32 len)
				: type(op::vector_copy), opt_storage(std::move(_src)), dst(_dst), length(len)
			{}

			transport_packet(void *_dst, rsx::primitive_type prim, u32 len)
				: type(op::index_emulate), dst(_dst), length(len), aux_param0(static_cast<u8>(prim))
			{}

			transport_packet(u32 command, void* args)
				: type(op::callback), src(args), aux_param0(command)
			{}

			transport_packet(const transport_packet&) = delete;
			transport_packet& operator=(const transport_packet&) = delete;
		};

		atomic_t<bool> m_mem_fault_flag = false;

		struct offload_thread;
		std::shared_ptr<named_thread<offload_thread>> m_thread;
#if defined(__APPLE__) && defined(__aarch64__)
		mutable metal4::host_writer_gate m_native_writers;
		metal4::host_writer_gate::ticket admit_native_write(metal4::host_writer_gate::transfer extent) const;
		mutable std::mutex m_native_handler_mutex;
		std::shared_ptr<const native_host_write_handler> m_native_handler;
		std::atomic<bool> m_native_closed{false};
#endif

		// TODO: Improved benchmarks here; value determined by profiling on a Ryzen CPU, rounded to the nearest 512 bytes
		const u32 max_immediate_transfer_size = 3584;

	public:
		dma_manager() = default;
		~dma_manager();

		// initialization
		void init();

		// General tranport
		void copy(void *dst, std::vector<u8>& src, u32 length) const;
		void copy(void *dst, void *src, u32 length) const;

		// Vertex utilities
		void emulate_as_indexed(void *dst, rsx::primitive_type primitive, u32 count);

		// Renderer callback
		void backend_ctrl(u32 request_code, void* args);

		// Synchronization
		bool is_current_thread() const;
		bool sync() const;
		void join();
		void set_mem_fault_flag();
		void clear_mem_fault_flag();

#if defined(__APPLE__) && defined(__aarch64__)
		// Install/remove only after draining the same producer gate. Snapshots
		// retain callback owners through the actual instruction. An absent
		// handler uses the renderer's native hook; it never replaces legacy work.
		bool attach_native_host_write_handler(std::shared_ptr<const native_host_write_handler> handler);
		bool detach_native_host_write_handler(const std::shared_ptr<const native_host_write_handler>& handler);
		std::optional<metal4::host_write_intent> prepare_native_host_write(void* destination, const void* source, u64 bytes) const;
		void check_native_work() const;
		// Excludes this manager's queued/immediate memory instructions for a
		// complete short CPU capture. Does not cover other host/guest writers.
		std::shared_ptr<void> capture_native_writes() const;
		std::optional<metal4::host_writer_gate::fault_pause> pause_native_writer_fault() const;
		void release_native_source_reservations() const noexcept;
		void resume_native_source_reservations() const;
		std::optional<utils::address_range32> get_native_fault_range(bool writing) const;
#endif

		// Fault recovery
		// Exact u16 bytes written by the non-indexed expansion,not an index
		// count. Keep wide until validating the fault's guest address extent.
		static constexpr std::optional<u64> index_emulation_bytes(primitive_type primitive, u32 count)
		{
			switch (primitive)
			{
			case primitive_type::line_loop:
				return (u64{count} + 1) * sizeof(u16);
			case primitive_type::polygon:
			case primitive_type::triangle_fan:
				if (count < 2)
					return {};
				return (u64{count} - 2) * 3 * sizeof(u16);
			case primitive_type::quads:
				return u64{count / 4} * 6 * sizeof(u16);
			default:
				return {};
			}
		}
		static constexpr std::optional<utils::address_range32> fault_range_from_bytes(u32 address, u64 bytes)
		{
			if (!bytes || bytes > 0xffff'ffffull || bytes > 0x1'0000'0000ull - address)
				return {};
			return utils::address_range32::start_length(address, static_cast<u32>(bytes));
		}
		utils::address_range32 get_fault_range(bool writing) const;
	};
}
