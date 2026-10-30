#pragma once

#include "Emu/RSX/Utils/rsx_utils.h"
#include "M4DMAMapping.h"

namespace metal4
{
template <typename Translate>
scaled_image_memory_mapping probe_rsx_dma_mapping(const dma_region& region, std::uint32_t columns, std::uint32_t rows,
	unsigned dma, const Translate& translate)
{
	std::optional<std::uint32_t> io_base;
	if (dma == CELL_GCM_CONTEXT_DMA_MEMORY_HOST_BUFFER || dma == CELL_GCM_LOCATION_MAIN)
		io_base = region.range.address;
	else if (dma == CELL_GCM_CONTEXT_DMA_REPORT_LOCATION_MAIN)
	{
		if (region.range.address >= 0x1000000 || region.range.size > 0x1000000 - region.range.address)
			throw std::out_of_range("Native main report DMA exceeds its IO aperture");
		io_base = 0x0e000000 + region.range.address;
	}
	return probe_dma_mapping(region, columns, rows, dma, translate, io_base);
}
} // namespace metal4
