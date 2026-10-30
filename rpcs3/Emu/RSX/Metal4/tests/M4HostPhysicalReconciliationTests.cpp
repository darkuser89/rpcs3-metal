#include "M4HostAccessFixture.h"
#include <cstring>
#include <iostream>

namespace
{
using namespace metal4;
using test::host_access_fixture;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refused(F&& operation)
{
	bool caught = false;
	try { operation(); } catch (const std::exception&) { caught = true; }
	require(caught, "Unsafe physical reconciliation was accepted");
}
void replace_first(host_access_fixture& access, guarded_guest_memory& memory)
{
	access.execute([&](auto&)
	{
		memory.begin_mapping_change({0, 65536});
		access.replace_first();
		memory.end_mapping_change({0, 65536});
	});
}
void physical_identity_and_offsets()
{
	auto a = std::make_shared<std::array<std::byte, 64>>();
	auto b = std::make_shared<std::array<std::byte, 64>>();
	const physical_backing write({16, 8}, {{{16, 8}, a, 16, 64}});
	const physical_backing alias({0x40000, 4}, {{{0x40000, 4}, a, 20, 64}});
	const physical_backing edge({16, 8}, {{{16, 8}, a, 24, 64}});
	const physical_backing replacement({16, 8}, {{{16, 8}, b, 16, 64}});
	auto other_control = std::shared_ptr<void>(a.get(), [](void*) {});
	const physical_backing same_pointer({16, 8}, {{{16, 8}, other_control, 16, 64}});
	require(write.overlaps(alias) && alias.overlaps(write) && !write.overlaps(edge) &&
		!write.overlaps(replacement) && !write.overlaps(same_pointer),
		"Physical reconciliation confused guest address,allocation ownership,offset or adjacent bytes");
}
void single_drain_releases_scratch()
{
	auto epochs = std::make_shared<mapping_epoch>();
	host_write_journal journal(epochs, 1);
	auto owner = std::make_shared<std::array<std::byte, 16>>();
	std::weak_ptr<void> retained = owner;
	const guest_surface_range range{32, 16};
	auto completion = journal.begin({range, epochs->current(range),
		std::make_shared<physical_backing>(range, std::vector<physical_backing::segment>{{range, owner, 0, 16}})});
	owner.reset(); completion.commit();
	journal.drain([&](std::span<const host_write_journal::completion> records)
	{
		require(records.size() == 1 && records[0].backing && !retained.expired(), "Drain lost physical owner before applying revisions");
	});
	require(retained.expired() && !journal.pending(), "Single drain retained retired scratch backing until a second operation");
}
void completed_split_remap()
{
	auto access = std::make_shared<host_access_fixture>();
	access->pin_first_alias();
	guarded_guest_memory memory(access);
	const auto target = memory.watch_surface(guest_byte_coverage({65528, 16}));
	const auto old_alias = memory.watch_surface(guest_byte_coverage({0x4fff8, 8}));
	const auto second = memory.watch_surface(guest_byte_coverage({65536, 8}));
	const auto disjoint = memory.watch_surface(guest_byte_coverage({0x41000, 16}));
	const auto old = memory.surface_revision(target), alias_tag = memory.surface_revision(old_alias);
	const auto second_tag = memory.surface_revision(second), disjoint_tag = memory.surface_revision(disjoint);
	auto destination = memory.prepare_host_access(target.range, true);
	{
		std::lock_guard lock(access->mapping_mutex);
		auto completed = memory.begin_host_write(std::move(destination));
		std::fill_n(access->bytes + 65528, 16, std::byte{0x6d});
		completed.commit();
	}
	replace_first(*access, memory);
	const auto replacement = memory.read_surface(target);
	const auto retained = memory.read_surface(old_alias);
	require(replacement.revision > old && retained.revision > alias_tag && memory.surface_revision(second) > second_tag &&
		memory.surface_revision(disjoint) == disjoint_tag && !memory.pending_host_writes(),
		"Completed remap lost surviving physical aliases or revised disjoint offsets");
	require(std::all_of(retained.bytes.begin(), retained.bytes.end(), [](auto b) { return b == std::byte{0x6d}; }) &&
		std::all_of(replacement.bytes.begin(), replacement.bytes.begin() + 8, [](auto b) { return b == std::byte{0}; }) &&
		std::all_of(replacement.bytes.begin() + 8, replacement.bytes.end(), [](auto b) { return b == std::byte{0x6d}; }),
		"Physical completion replayed old bytes into replacement mapping or lost the unchanged second allocation");
	require(!memory.compare_write(old_alias, alias_tag, std::vector<std::byte>(8)),
		"Old physical alias accepted a pre-store CAS after destination remap");
}
void source_remap_after_completion()
{
	auto access = std::make_shared<host_access_fixture>();
	guarded_guest_memory memory(access);
	const auto target = memory.watch_surface(guest_byte_coverage({65568, 16}));
	const auto before = memory.surface_revision(target);
	std::fill_n(access->bytes + 32, 16, std::byte{0x48});
	auto destination = memory.prepare_host_access(target.range, true);
	auto source = memory.prepare_host_access({32, 16}, false);
	std::weak_ptr<void> old_source = source.backing->segments()[0].owner;
	{
		std::lock_guard lock(access->mapping_mutex);
		auto completed = memory.begin_host_write(std::move(destination), std::move(source));
		std::memcpy(access->bytes + 65568, access->bytes + 32, 16);
		completed.commit();
	}
	replace_first(*access, memory);
	require(!old_source.expired(), "Source lease disappeared before completed batch reconciliation");
	const auto result = memory.read_surface(target);
	require(result.revision > before && result.bytes[0] == std::byte{0x48} && !memory.pending_host_writes() && old_source.expired(),
		"A later source remap blocked completed physical destination or retained a retired source lease");
}
void padding_and_retired_owners()
{
	auto access = std::make_shared<host_access_fixture>();
	guarded_guest_memory memory(access);
	const guest_surface_layout pitched{{128, 32}, 4, 2, 1, 16, {}, {}};
	const auto watched = memory.watch_surface(pitched);
	const auto tag = memory.surface_revision(watched);
	auto destination = memory.prepare_host_access({136, 4}, true);
	std::weak_ptr<void> owner = destination.backing->segments()[0].owner;
	{
		std::lock_guard lock(access->mapping_mutex);
		auto completed = memory.begin_host_write(std::move(destination));
		std::fill_n(access->bytes + 136, 4, std::byte{0x5a});
		completed.commit();
	}
	require(memory.surface_revision(watched) == tag && !memory.pending_host_writes() && owner.expired(),
		"Physical padding-only store revised active bytes or scratch retained a retired backing");
	destination = memory.prepare_host_access({32, 16}, true);
	owner = destination.backing->segments()[0].owner;
	{
		std::lock_guard lock(access->mapping_mutex);
		auto completed = memory.begin_host_write(std::move(destination));
		std::fill_n(access->bytes + 32, 16, std::byte{0x29});
		completed.commit();
	}
	replace_first(*access, memory);
	(void)memory.surface_revision(watched);
	require(!memory.pending_host_writes() && owner.expired() && access->bytes[32] == std::byte{0},
		"Completed store without surviving aliases retained dead allocation or wrote replacement bytes");
}
void failed_batch_is_atomic()
{
	auto access = std::make_shared<host_access_fixture>();
	access->pin_first_alias();
	guarded_guest_memory memory(access);
	const auto target = memory.watch_surface(guest_byte_coverage({32, 8}));
	const auto alias = memory.watch_surface(guest_byte_coverage({0x40020, 8}));
	const auto other = memory.watch_surface(guest_byte_coverage({65568, 8}));
	const auto disjoint = memory.watch_surface(guest_byte_coverage({0x41000, 8}));
	const auto last = memory.surface_revision(disjoint);
	auto first = memory.prepare_host_access(target.range, true);
	auto second = memory.prepare_host_access(other.range, true);
	{
		std::lock_guard lock(access->mapping_mutex);
		auto a = memory.begin_host_write(std::move(first));
		std::fill_n(access->bytes + 32, 8, std::byte{0x11}); a.commit();
		auto b = memory.begin_host_write(std::move(second));
		std::fill_n(access->bytes + 65568, 8, std::byte{0x22}); b.commit();
	}
	replace_first(*access, memory);
	access->fail_physical_query = access->physical_queries + 2;
	refused([&] { (void)memory.read_surface(alias); });
	require(memory.pending_host_writes() == 2, "Failed physical planning partially retired a completion batch");
	access->fail_physical_query = 0;
	require(memory.surface_revision(target) == last + 1 && memory.surface_revision(alias) == last + 1 &&
		memory.surface_revision(other) == last + 1 && memory.surface_revision(disjoint) == last && !memory.pending_host_writes(),
		"Failed remap/physical planning published partial revisions before retry");
}
void ambiguous_stays_retained()
{
	auto access = std::make_shared<host_access_fixture>();
	guarded_guest_memory memory(access);
	auto prepared = memory.prepare_host_access({32, 8}, true);
	std::weak_ptr<void> owner = prepared.backing->segments()[0].owner;
	auto* physical = prepared.backing->segments()[0].data();
	{
		auto partial = memory.begin_host_write(std::move(prepared));
		access->bytes[32] = std::byte{0x37};
		replace_first(*access, memory);
		partial.commit(); // Commit mapping validation fails;not a completed physical store.
	}
	refused([&] { (void)memory.read({32, 8}); });
	require(memory.pending_host_writes() == 1 && !owner.expired() && physical[0] == std::byte{0x37} && access->bytes[32] == std::byte{0},
		"Ambiguous mid-store remap was incorrectly treated as a completed physical store");
}
} // namespace
int main()
{
	try
	{
		physical_identity_and_offsets(); single_drain_releases_scratch(); completed_split_remap(); source_remap_after_completion();
		padding_and_retired_owners(); failed_batch_is_atomic(); ambiguous_stays_retained();
		std::cout << "Physical completion reconciliation:post-commit destination/source remap;split allocations and surviving shifted aliases;ownership/offset/edge identity;replacement bytes untouched;stale alias CAS refused;active padding precision;atomic failed batch retry;retired backing/scratch release;ambiguous mid-store remap retained;real OS-backed bytes,no App or guest VM started\n";
	}
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
