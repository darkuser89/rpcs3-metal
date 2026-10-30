#pragma once

#include "M4SurfaceLayout.h"
#include "M4SwizzleLayout.h"

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace metal4
{
class physical_surface;
class physical_identity;
struct physical_surface_write;
// Translated PS3 VM address, not an unqualified RSX IO offset. Main/local IO
// translation and protection hooks belong to the emulator memory adapter.
struct guest_surface_range
{
	std::uint32_t address = 0;
	std::size_t size = 0;
};
[[nodiscard]] inline bool overlaps(guest_surface_range a, guest_surface_range b)
{
	return a.size && b.size && (a.address <= b.address ? std::uint64_t(b.address) - a.address < a.size : std::uint64_t(a.address) - b.address < b.size);
}
inline void validate_guest_surface_range(guest_surface_range range)
{
	if (range.size > 0x100000000ull - std::uint64_t(range.address))
		throw std::overflow_error("Guest surface range wraps PS3 VM addresses");
}
struct guest_surface_layout
{
	guest_surface_range range;
	std::size_t width, height, pixel_bytes, row_pitch;
	std::optional<swizzle_region> morton;
	// Optional provider-owned watch lifetime. Keys/active byte geometry ignore
	// this owner;retained layouts keep their revision registration alive.
	std::shared_ptr<void> registration = {};
	// CPU active-byte coverage only: a nonzero stride describes one-byte
	// columns separated by gaps. Native texture layouts leave this zero.
	std::size_t column_stride = 0;
};
// Guest rows are byte-addressed and may start inside a native texel word.
// Keep native Metal transfer alignment separate from VM storage geometry.
[[nodiscard]] inline texture_buffer_layout guest_surface_linear_layout(std::size_t width, std::size_t height,
	std::size_t pixel_bytes, std::size_t row_pitch = 0)
{
	if (!pixel_bytes) throw std::invalid_argument("Guest surface pixel size must be nonzero");
	return transfer_layout({1, 1, 1}, checked_texture_product(width, pixel_bytes), height, 1, row_pitch);
}
// Full linear pitched storage or one selected 2D slice of a Morton volume.
// Width/height here describe the spatially expanded raster for MSAA surfaces.
[[nodiscard]] inline guest_surface_layout plan_guest_surface(std::uint32_t address, std::size_t width,
															 std::size_t height, std::size_t pixel_bytes, std::size_t row_pitch = 0, const std::optional<swizzle_region>& morton = {})
{
	const auto linear = guest_surface_linear_layout(width, height, pixel_bytes, row_pitch);
	std::size_t bytes = linear.allocation_size;
	if (morton)
	{
		if (row_pitch || morton->size != std::array<std::size_t, 3>{width, height, 1})
			throw std::invalid_argument("Guest Morton surface dimensions/pitch are invalid");
		bytes = plan_swizzle(*morton, pixel_bytes).storage_bytes;
	}
	if (bytes > 0x100000000ull - std::uint64_t(address))
		throw std::overflow_error("Guest surface wraps the PS3 VM address space");
	return {{address, bytes}, width, height, pixel_bytes, linear.bytes_per_row, morton};
}
[[nodiscard]] inline guest_surface_layout plan_guest_byte_matrix(std::uint32_t address, std::size_t columns,
	std::size_t rows, std::size_t stride, std::size_t pitch = 0)
{
	if (!columns || !rows || !stride) throw std::invalid_argument("Guest byte matrix requires nonempty dimensions and stride");
	const auto extent = checked_texture_product(columns - 1, stride);
	if (extent == std::numeric_limits<std::size_t>::max()) throw std::overflow_error("Guest byte matrix row wraps");
	auto result = plan_guest_surface(address, extent + 1, rows, 1, pitch);
	result.width = columns;
	result.column_stride = stride == 1 ? 0 : stride;
	return result;
}
struct guest_surface_snapshot
{
	// Opaque monotonic revision for selected active pixel bytes. Changes to
	// padding or other Morton-volume regions must not change this revision.
	std::uint64_t revision;
	std::vector<std::byte> bytes;
};
// CPU-only current storage identity, without a guest pointer or a retained
// mapping. Intersections compare allocation control blocks and byte offsets.
// Capture/query only under the same provider exclusion as revision checks.
struct storage_identity_span
{
	std::weak_ptr<void> allocation;
	std::size_t offset;
	guest_surface_range guest;
};
struct guest_surface_write
{
	guest_surface_layout coverage;
	std::uint64_t expected;
	std::span<const std::byte> bytes;
	// Packed payloads concatenate active intervals in ascending guest address
	// order. Dense payloads retain their original storage-sized contract.
	bool packed = false;
	// An all-ordered batch publishes disjoint logical active intervals by this
	// offset, independent of translated EAs or the order of its watch parts.
	std::optional<std::size_t> logical_offset = {};
};
struct physical_surface_publication
{
	std::uint64_t revision;
	// Exact ordered virtual-watch revisions from the SAME physical commit.
	// Empty means the original virtual input changed or preservation is unsupported.
	std::vector<std::uint64_t> virtual_revisions;
};

// Atomic VM adapter contract. read returns the complete storage snapshot but
// read/revision watch only active pixel bytes. compare_write atomically checks
// that revision and patches only active pixels from the storage-sized buffer;
// a failed compare leaves memory untouched. Methods may be used concurrently
// by VM writers while the cache waits for GPU completion. Real VM accesses
// must flush through cache.cpu_read/cpu_write (or equivalent protection hooks).
class guest_surface_memory
{
  public:
	// Optional exact metadata for every byte in this nonempty range. Missing
	// support is distinct from an empty/nonoverlapping physical allocation.
	[[nodiscard]] virtual std::optional<std::vector<storage_identity_span>> storage_identity(guest_surface_range) { return {}; }
	// Optional physical publication domain. Unsupported legacy providers return
	// no ticket and refuse publication;never replay through an old guest VA.
	[[nodiscard]] virtual std::shared_ptr<const physical_surface> capture_physical_surface(const guest_surface_layout&, std::uint64_t) { return {}; }
	[[nodiscard]] virtual bool physical_surface_current(const physical_surface&) { return false; }
	// Select retained OLD active bytes against present allocation metadata.
	// Does not map/read the old VA or require the ticket's revision to be current.
	// Unsupported providers must refuse selection rather than omit aliases.
	[[nodiscard]] virtual std::optional<bool> physical_surface_overlaps_current(const physical_surface&, guest_surface_range) { return {}; }
	// CPU-only exact active-byte query,including protected current aliases.
	// The identity owns no VM handles. Unsupported metadata returns no answer;
	// callers must refuse or conservatively invalidate,never infer non-overlap.
	[[nodiscard]] virtual std::optional<bool> physical_identity_overlaps_current(const physical_identity&, const guest_surface_layout&) { return {}; }
	// Registered virtual-watch and physical-ticket check without mapping bytes.
	[[nodiscard]] virtual std::optional<bool> physical_surface_virtual_current(const physical_surface&, std::uint64_t) { return {}; }
	[[nodiscard]] virtual std::optional<guest_surface_snapshot> read_physical_surface(const physical_surface&) { return {}; }
	[[nodiscard]] virtual std::optional<std::uint64_t> compare_write_physical(std::span<const physical_surface_write> writes)
	{
		if (writes.empty()) return 0;
		throw std::logic_error("Guest memory provider lacks atomic physical publication");
	}
	// expected is empty or has one original virtual-watch revision per write.
	// A receipt is valid only if ALL original watches still matched before the
	// store,and reports their post-store revisions under the same exclusion.
	// Prepare receipt storage/validation before mutation;never query old VAs or
	// allocate/throw after publishing bytes. Legacy physical providers invalidate
	// native mirrors conservatively by returning no virtual receipt.
	[[nodiscard]] virtual std::optional<physical_surface_publication> compare_write_physical(
		std::span<const physical_surface_write> writes, std::span<const std::uint64_t> expected)
	{
		if (!expected.empty() && expected.size() != writes.size()) throw std::invalid_argument("Physical publication watch count differs from writes");
		if (const auto revision = compare_write_physical(writes)) return physical_surface_publication{*revision, {}};
		return {};
	}
	using storage_lookup = std::function<bool(guest_surface_range, guest_surface_range)>;
	using storage_test = std::function<bool(bool, const storage_lookup&)>;
	using storage_rebase = std::function<std::optional<guest_surface_range>(guest_surface_range, guest_surface_range)>;
	using storage_equivalent = std::function<bool(guest_surface_range, guest_surface_range)>;
	using rebase_test = std::function<bool(const storage_rebase&, const storage_equivalent&)>;
	// A byte-for-byte,order-preserving translation into source storage.
	// Physical providers must prove every mapping segment,not just overlap.
	virtual std::optional<guest_surface_range> rebase_storage(guest_surface_range source, guest_surface_range request)
	{
		validate_guest_surface_range(source);
		validate_guest_surface_range(request);
		if (request.address < source.address || request.size > source.size ||
			std::uint64_t(request.address) - source.address > source.size - request.size)
			return {};
		return request;
	}
	virtual bool test_rebase(const rebase_test& test)
	{
		const storage_rebase translate = [&](guest_surface_range source, guest_surface_range request)
		{ return rebase_storage(source, request); };
		const storage_equivalent equal = [](guest_surface_range a, guest_surface_range b)
		{ return a.address == b.address && a.size == b.size; };
		return test(translate, equal);
	}
	// Evaluate a CPU-only overlap query under stable mapping identities. The
	// bool identifies direct guest addressing;aliased providers override this.
	// Do not perform GPU work or recursively enter the provider in the query.
	virtual bool test_storage(const storage_test& test)
	{
		const storage_lookup direct = [](guest_surface_range a, guest_surface_range b)
		{ return overlaps(a, b); };
		return test(true, direct);
	}
	virtual ~guest_surface_memory() = default;
	// Fixed-storage providers already own their backing for their whole life.
	// Remappable VM providers must override and retain the old physical storage
	// independently of guest virtual mappings. This does not exclude writers.
	[[nodiscard]] virtual std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) { return {}; }
	[[nodiscard]] virtual guest_surface_layout watch_surface(const guest_surface_layout& coverage) { return coverage; }
	[[nodiscard]] virtual guest_surface_snapshot read_surface(const guest_surface_layout& coverage) = 0;
	// Reads only active bytes, concatenated in ascending guest address order,
	// under the same exclusion as the returned revision. No padding is mapped.
	[[nodiscard]] virtual guest_surface_snapshot read_surface_packed(const guest_surface_layout&)
	{
		throw std::logic_error("Guest memory provider lacks packed active-byte capture");
	}
	[[nodiscard]] virtual std::uint64_t surface_revision(const guest_surface_layout& coverage) = 0;
	[[nodiscard]] virtual std::optional<std::uint64_t> compare_write(const guest_surface_layout& coverage,
																	 std::uint64_t expected, std::span<const std::byte> bytes) = 0;
	// All revisions/mappings/payloads must be checked before the first store.
	// A provider without a batch transaction must refuse,never loop over CAS.
	// Frozen payloads are applied in list order,including overlapping aliases.
	[[nodiscard]] virtual std::optional<std::uint64_t> compare_write_many(std::span<const guest_surface_write> writes)
	{
		if (writes.empty())
			return 0;
		if (writes.size() != 1 || writes[0].packed || writes[0].logical_offset)
			throw std::logic_error("Guest memory provider lacks atomic batch publication");
		return compare_write(writes[0].coverage, writes[0].expected, writes[0].bytes);
	}
	virtual void write(guest_surface_range range, std::span<const std::byte> bytes) = 0;
	[[nodiscard]] guest_surface_snapshot read(guest_surface_range range)
	{
		validate_guest_surface_range(range);
		return read_surface(guest_surface_layout{range, range.size, 1, 1, range.size, {}});
	}
	[[nodiscard]] std::uint64_t revision(guest_surface_range range)
	{
		validate_guest_surface_range(range);
		return surface_revision(guest_surface_layout{range, range.size, 1, 1, range.size, {}});
	}
};
} // namespace metal4
