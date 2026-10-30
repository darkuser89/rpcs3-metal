#pragma once

#include "M4GuardedGuestMemory.h"

namespace metal4
{
struct prepared_host_transfer
{
	std::optional<host_write_journal::preparation> destination, source;
};
} // namespace metal4
