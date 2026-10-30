#include "M4ReportAtomic.h"

#include <array>
#include <atomic>
#include <bit>
#include <iostream>
#include <thread>

namespace
{
void require(bool value)
{
	if (!value)
		throw std::runtime_error("Native report atomic encoding or consistency differs");
}
std::uint64_t stamp(std::uint32_t value)
{
	return (std::uint64_t(value) << 32) | std::uint32_t(~value);
}
} // namespace
int main()
{
	try
	{
		atomic_t<CellGcmReportData> report{};
		for (const auto timer : {0ull, 0x0123456789abcdefull, 0xffffffffffffffffull})
			for (const auto value : {0u, 0x89abcdefu, 0xffffffffu})
			{
				metal4::store_report_atomic(report, timer, value);
				const auto loaded = report.load();
				require(loaded.timer == timer && loaded.value == value && loaded.padding == 0);
				const auto bytes = std::bit_cast<std::array<std::byte, 16>>(loaded);
				for (unsigned byte = 0; byte < 8; ++byte)
					require(bytes[byte] == std::byte(timer >> (56 - 8 * byte)));
				for (unsigned byte = 0; byte < 4; ++byte)
				{
					require(bytes[8 + byte] == std::byte(value >> (24 - 8 * byte)));
					require(bytes[12 + byte] == std::byte{0});
				}
			}
		metal4::store_report_atomic(report, stamp(0), 0);
		std::atomic<bool> start = false, valid = true;
		std::array<std::thread, 8> threads;
		for (unsigned index = 0; index < threads.size(); ++index)
			threads[index] = std::thread([&, index]
										 {
				while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
				for (unsigned i = 0; i < 125000; ++i)
					if (index < 4)
					{
						const auto value = (index << 28) | i;
						metal4::store_report_atomic(report, stamp(value), value);
					}
					else
					{
						const auto loaded = report.load();
						const std::uint32_t value = loaded.value;
						if (loaded.timer != stamp(value) || loaded.padding != 0) valid = false;
					} });
		start.store(true, std::memory_order_release);
		for (auto& thread : threads)
			thread.join();
		require(valid.load());
		std::cout << "Real Apple-arm64 report atomic:BE timer/value/zero padding,extremes and500000concurrent stores/500000consistent loads passed;no guest VM initialized\n";
	}
	catch (const std::exception& error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
