#include "M4HostAccessFixture.h"
#include <atomic>
#include <iostream>
#include <thread>

namespace
{
using namespace metal4;
using test::host_access_fixture;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template <typename F> void refused(F&& operation)
{
	bool caught = false;
	try { operation(); } catch (const std::exception&) { caught = true; }
	require(caught, "Invalid physical surface operation was accepted");
}
auto capture(guarded_guest_memory& memory, guest_surface_layout layout)
{
	layout = memory.watch_surface(layout);
	return memory.capture_physical_surface(layout, memory.surface_revision(layout));
}
auto capture(guarded_guest_memory& memory, guest_surface_range range) { return capture(memory, guest_byte_coverage(range)); }
auto store(guarded_guest_memory& memory, std::shared_ptr<const physical_surface> surface, std::span<const std::byte> bytes)
{
	const physical_surface_write write{std::move(surface), bytes};
	return memory.compare_write_physical({&write, 1});
}
void replace(host_access_fixture& access, guarded_guest_memory& memory, bool unmap = false)
{
	access.execute([&](auto&)
	{
		memory.begin_mapping_change({0, 65536});
		if (unmap) access.unmap_first(); else access.replace_first();
		memory.end_mapping_change({0, 65536});
	});
}
void old_owner_after_remap_and_unmap()
{
	for (bool unmap : {false, true})
	{
		auto access = std::make_shared<host_access_fixture>(); access->pin_first_alias();
		guarded_guest_memory memory(access);
		auto ticket = capture(memory, guest_surface_range{65528, 16});
		const auto old_alias = memory.watch_surface(guest_byte_coverage({0x4fff8, 8}));
		const auto second = memory.watch_surface(guest_byte_coverage({65536, 8}));
		const auto before_alias = memory.surface_revision(old_alias), before_second = memory.surface_revision(second);
		replace(*access, memory, unmap);
		require(memory.physical_surface_current(*ticket), "Virtual remap invalidated unchanged physical bytes");
		const auto snapshot = memory.read_physical_surface(*ticket);
		require(snapshot && snapshot->bytes == std::vector<std::byte>(16), "Physical snapshot entered an old VA after remap/unmap");
		const std::array<std::byte, 16> payload{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6}, std::byte{7}, std::byte{8},
			std::byte{9}, std::byte{10}, std::byte{11}, std::byte{12}, std::byte{13}, std::byte{14}, std::byte{15}, std::byte{16}};
		require(store(memory, ticket, payload).has_value(), "Old physical owner refused stable writeback");
		require(std::equal(payload.begin(), payload.begin() + 8, memory.read_surface(old_alias).bytes.begin()) &&
			std::equal(payload.begin() + 8, payload.end(), access->bytes + 65536), "Physical writeback lost split old allocation bytes");
		if (!unmap) require(std::all_of(access->bytes + 65528, access->bytes + 65536, [](auto b) { return b == std::byte{}; }), "Physical writeback overwrote replacement VA");
		require(memory.surface_revision(old_alias) > before_alias && memory.surface_revision(second) > before_second, "Physical writeback failed to revise surviving aliases");
		for (unsigned i = 0; i < access->reservations; ++i)
			require(!overlaps(access->reservation_ranges[i], guest_surface_range{0, 65536}), "Physical writeback notified replaced or unmapped VA");
		require(access->reservations == 2 && !memory.physical_surface_current(*ticket) && !store(memory, ticket, payload), "Physical ticket was not consumed exactly once");
	}
}
void protected_current_alias()
{
	auto access = std::make_shared<host_access_fixture>(); guarded_guest_memory memory(access);
	auto ticket = capture(memory, guest_surface_range{32, 8});
	access->execute([&](auto&) { access->protect_first(false); });
	require(memory.read_physical_surface(*ticket)->bytes == std::vector<std::byte>(8), "Physical snapshot entered protected current VA");
	const std::array<std::byte, 8> payload{std::byte{0x41}, std::byte{0x42}};
	require(store(memory, ticket, payload).has_value(), "Physical publication entered a protected guest VA");
	refused([&] { (void)memory.read({32, 8}); });
	refused([&] { (void)memory.read_physical_surface(*ticket); });
	access->execute([&](auto&) { access->protect_first(true); });
	require(std::equal(payload.begin(), payload.end(), access->bytes + 32) && access->reservations == 2, "Protected physical write lost payload or live alias notices");
}
void cpu_writers_and_padding()
{
	for (unsigned writer = 0; writer < 4; ++writer)
	{
		auto access = std::make_shared<host_access_fixture>(); access->pin_first_alias(); guarded_guest_memory memory(access);
		const guest_surface_layout pitched{{128, 32}, 4, 2, 1, 16, {}, {}};
		auto ticket = capture(memory, pitched);
		const std::array<std::byte, 4> cpu{std::byte{0x55}, std::byte{0x66}};
		memory.write({136, 4}, cpu); // Padding must not revise active physical bytes.
		require(memory.physical_surface_current(*ticket), "Padding-only CPU store invalidated active physical bytes");
		require(memory.read_physical_surface(*ticket)->bytes == std::vector<std::byte>(32), "Physical snapshot exposed uncaptured padding bytes");
		replace(*access, memory);
		require(memory.physical_surface_current(*ticket), "Replacement mapping revised old physical ticket");
		const guest_surface_range alias{0x40080, 4};
		if (writer == 0) memory.write(alias, cpu);
		if (writer == 1)
		{
			const auto watch = memory.watch_surface(guest_byte_coverage(alias));
			require(memory.compare_write(watch, memory.surface_revision(watch), cpu).has_value(), "Virtual CAS setup failed");
		}
		if (writer == 2) access->execute([&](auto& view)
		{
			memory.before_write_in_transaction(view, alias); std::memcpy(view.map(alias, true), cpu.data(), cpu.size()); view.written(alias);
		});
		if (writer == 3)
		{
			auto intent = memory.begin_host_write(memory.prepare_host_access(alias, true));
			std::memcpy(access->map(alias, true), cpu.data(), cpu.size()); intent.commit();
		}
		refused([&] { (void)memory.read_physical_surface(*ticket); });
		const auto notices = access->reservations;
		require(!memory.physical_surface_current(*ticket) && !store(memory, ticket, std::vector<std::byte>(32, std::byte{0xaa})), "Newer CPU physical bytes accepted stale GPU publication");
		require(std::equal(cpu.begin(), cpu.end(), access->map(alias, false)) && access->reservations == notices, "Rejected physical CAS changed CPU bytes or reservations");
	}
}
void replacement_cpu_store_keeps_old_ticket()
{
	auto access = std::make_shared<host_access_fixture>(); guarded_guest_memory memory(access);
	auto ticket = capture(memory, guest_surface_range{32, 8}); replace(*access, memory);
	const std::array<std::byte, 8> replacement{std::byte{0x33}}, old{std::byte{0x77}};
	memory.write({32, 8}, replacement);
	require(memory.physical_surface_current(*ticket) && store(memory, ticket, old).has_value(), "Replacement allocation was mistaken for old physical owner");
	require(std::equal(replacement.begin(), replacement.end(), access->bytes + 32) && ticket->parts()[0]->segments()[0].data()[0] == old[0], "Old-owner writeback touched replacement CPU bytes");
}
void atomic_batch_and_alias_payload()
{
	auto access = std::make_shared<host_access_fixture>(); access->pin_first_alias(); guarded_guest_memory memory(access);
	auto a = capture(memory, guest_surface_range{32, 8}), b = capture(memory, guest_surface_range{65568, 8});
	const std::array<std::byte, 8> pa{std::byte{0x11}}, pb{std::byte{0x22}};
	const std::array writes{physical_surface_write{a, pa}, physical_surface_write{b, pb}};
	access->fail_reservation_planning = true;
	refused([&] { (void)memory.compare_write_physical(writes); });
	require(memory.physical_surface_current(*a) && memory.physical_surface_current(*b) && !access->reservations && access->bytes[32] == std::byte{} && access->bytes[65568] == std::byte{}, "Failed physical batch partially published bytes or revisions");
	access->fail_reservation_planning = false;
	require(memory.compare_write_physical(writes).has_value(), "Atomic physical batch retry failed");
	std::fill_n(access->bytes + 32, 8, std::byte{0x44}); std::fill_n(access->bytes + 65568, 8, std::byte{0x88});
	a = capture(memory, guest_surface_range{32, 8}); b = capture(memory, guest_surface_range{65568, 8});
	const std::array swap{physical_surface_write{a, {access->bytes + 65568, 8}}, physical_surface_write{b, {access->bytes + 32, 8}}};
	require(memory.compare_write_physical(swap).has_value() && access->bytes[32] == std::byte{0x88} && access->bytes[65568] == std::byte{0x44}, "Physical batch did not freeze aliased payload before stores");
}
void disjoint_tickets_and_unfinished_store()
{
	auto access = std::make_shared<host_access_fixture>(); guarded_guest_memory memory(access);
	auto a = capture(memory, guest_surface_range{32, 8}), edge = capture(memory, guest_surface_range{40, 8}), disjoint = capture(memory, guest_surface_range{64, 8});
	require(store(memory, a, std::array<std::byte, 8>{std::byte{0x22}}).has_value() && memory.physical_surface_current(*edge) && memory.physical_surface_current(*disjoint),
		"Physical publication revised adjacent or disjoint allocation offsets");
	std::optional<host_write_intent> intent(memory.begin_host_write(memory.prepare_host_access({64, 8}, true)));
	access->bytes[64] = std::byte{0x55};
	refused([&] { (void)store(memory, disjoint, std::array<std::byte, 8>{std::byte{0xaa}}); });
	require(memory.pending_host_writes() == 1 && access->bytes[64] == std::byte{0x55}, "Unfinished physical store lost its prefix or intent");
	intent.reset();
	refused([&] { (void)store(memory, disjoint, std::array<std::byte, 8>{}); });
	require(memory.pending_host_writes() == 1, "Ambiguous physical intent was silently retired");
}
void budgets_domain_and_lifetime()
{
	auto access = std::make_shared<host_access_fixture>(); guarded_guest_memory memory(access, 2, 64, 16), other(access);
	auto ticket = capture(memory, guest_surface_range{32, 8});
	refused([&] { (void)other.physical_surface_current(*ticket); });
	refused([&] { (void)other.read_physical_surface(*ticket); });
	refused([&] { (void)store(other, ticket, std::array<std::byte, 8>{}); });
	refused([&] { (void)store(memory, ticket, std::array<std::byte, 7>{}); });
	auto second = capture(memory, guest_surface_range{32, 8});
	refused([&] { (void)capture(memory, guest_surface_range{32, 8}); });
	std::weak_ptr<void> owner = ticket->parts()[0]->segments()[0].owner;
	ticket.reset(); second.reset();
	require(owner.expired(), "Weak physical revision cells retained dead backing mappings");
	ticket = capture(memory, guest_surface_range{32, 8});
	access->segments_supported = false;
	require(!capture(memory, guest_surface_range{40, 8}), "Unsupported physical provider invented a ticket");
}
void owner_writer_schedules()
{
	for (unsigned schedule = 0; schedule < 32; ++schedule)
	{
		auto access = std::make_shared<host_access_fixture>(); access->pin_first_alias(); guarded_guest_memory memory(access);
		auto ticket = capture(memory, guest_surface_range{32, 8}); replace(*access, memory);
		std::atomic<unsigned> phase{0};
		std::atomic<bool> ready{false};
		std::thread writer([&]
		{
			ready.store(true, std::memory_order_release);
			while (phase.load(std::memory_order_acquire) != 1) std::this_thread::yield();
			memory.write({0x40020, 8}, std::array<std::byte, 8>{std::byte{0x5a}}); phase.store(2, std::memory_order_release);
		});
		while (!ready.load(std::memory_order_acquire)) std::this_thread::yield();
		phase.store(1, std::memory_order_release);
		(void)store(memory, ticket, std::array<std::byte, 8>{std::byte{0xcc}}); writer.join();
		require(!memory.physical_surface_current(*ticket) && access->map({0x40020, 8}, false)[0] == std::byte{0x5a} && access->bytes[32] == std::byte{}, "Owner publication lost newer physical writer bytes");
	}
}
void atomic_virtual_receipts()
{
	for (unsigned mode = 0; mode < 5; ++mode)
	{
		auto access = std::make_shared<host_access_fixture>(); access->pin_first_alias(); guarded_guest_memory memory(access);
		const std::array layouts{memory.watch_surface(plan_guest_surface(128, 4, 2, 1, 16)), memory.watch_surface(guest_byte_coverage({65536, 8}))};
		const std::array expected{memory.surface_revision(layouts[0]), memory.surface_revision(layouts[1])};
		const std::array tickets{memory.capture_physical_surface(layouts[0], expected[0]), memory.capture_physical_surface(layouts[1], expected[1])};
		const std::vector<std::byte> first(32, std::byte{0x31}), second(8, std::byte{0x52});
		const std::array writes{physical_surface_write{tickets[0], first}, physical_surface_write{tickets[1], second}};
		if (mode == 1 || mode == 2) replace(*access, memory, mode == 2);
		if (mode == 3) access->execute([&](auto&) { access->protect_first(false); });
		if (mode == 4) memory.write({0x40080, 4}, std::array<std::byte, 4>{std::byte{0x7d}});
		const auto reservations = access->reservations;
		refused([&] { (void)memory.compare_write_physical(writes, std::span(expected).first(1)); });
		require(access->reservations == reservations, "Malformed receipt request published physical bytes");
		const auto result = memory.compare_write_physical(writes, expected);
		if (mode == 4)
		{
			require(!result && access->reservations == reservations && memory.read({0x40080, 4}).bytes[0] == std::byte{0x7d} &&
				memory.read({65536, 8}).bytes == std::vector<std::byte>(8), "Stale physical receipt changed one member of atomic batch");
			continue;
		}
		require(result && result->revision, "Stable physical receipt batch was refused");
		if (mode == 0 || mode == 3)
		{
			require(result->virtual_revisions.size() == 2, "Current/protected original watches lost native receipt");
			if (mode == 3) access->execute([&](auto&) { access->protect_first(true); });
			for (unsigned i = 0; i < 2; ++i)
				require(result->virtual_revisions[i] == memory.surface_revision(layouts[i]) && result->virtual_revisions[i] > expected[i],
					"Receipt did not report exact post-store virtual watch revision");
		}
		else
		{
			require(result->virtual_revisions.empty(), "Partially remapped batch blessed a stale native mirror");
			if (mode == 1) require(memory.read({128, 4}).bytes == std::vector<std::byte>(4), "Receipt publication overwrote replacement allocation");
		}
		require(memory.read({0x40080, 4}).bytes == std::vector<std::byte>(4, std::byte{0x31}) &&
			memory.read({0x40084, 12}).bytes == std::vector<std::byte>(12) && memory.read({65536, 8}).bytes == second,
			"Receipt publication lost active physical bytes,padding or disjoint output");
		require(!memory.compare_write_physical(writes, expected), "Receipt reused consumed physical tickets");
	}
	std::cout << "Atomic physical virtual receipts:current/protected ordered watches;partial remap/unmap no receipt;stale CPU batch refusal;bad watch count;active padding;one-shot consumption\n";
}
}
int main()
{
	try
	{
		old_owner_after_remap_and_unmap(); protected_current_alias(); cpu_writers_and_padding(); replacement_cpu_store_keeps_old_ticket();
		atomic_batch_and_alias_payload(); disjoint_tickets_and_unfinished_store(); budgets_domain_and_lifetime(); owner_writer_schedules(); atomic_virtual_receipts();
		std::cout << "Physical surface publication:old split owners after remap/unmap;protected aliases;precise reservations;four CPU writer paths;atomic batch/frozen payload;weak lifetime;32competing writer-owner schedules\n";
	}
	catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
