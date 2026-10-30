#pragma once

#if !defined(__APPLE__) || !defined(__aarch64__)
#error "Native Metal4 reports require Apple Silicon."
#endif

#include "Emu/RSX/GCM.h"

namespace metal4
{
// Use the emulator's actual aligned sixteen-byte atomic ABI. Mapping,
// producer exclusion,revision and reservation notifications surround this
// primitive in the real VM provider; no guest pointer is retained here.
inline void store_report_atomic(atomic_t<CellGcmReportData>& report, std::uint64_t timestamp, std::uint32_t value)
{
	static_assert(sizeof(CellGcmReportData) == 16 && alignof(CellGcmReportData) == 16);
	report.store({timestamp, value, 0});
}
} // namespace metal4
