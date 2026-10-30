#include "M4HostAccessFixture.h"
#include <iostream>

namespace
{
using namespace metal4;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template <typename F> void refused(F&& operation)
{
	bool failed = false;
	try { operation(); } catch (const std::exception&) { failed = true; }
	require(failed, "Unsafe maintenance shortcut accepted stale/incomplete metadata");
}
void stable_epoch_and_mutations()
{
	auto provider = std::make_shared<test::host_access_fixture>();
	guarded_guest_memory memory(provider);
	std::vector<guest_surface_layout> watches;
	std::vector<std::uint64_t> revisions;
	for (unsigned i = 0; i < 512; ++i)
	{
		watches.push_back(memory.watch_surface(guest_byte_coverage({65536 + i * 128, 4})));
		revisions.push_back(memory.surface_revision(watches.back()));
	}
	std::uint64_t identity = 0;
	provider->execute([&](auto&)
	{
		identity = guest_memory_epoch::identity(*provider);
		require(identity != 0, "Outer capture has no unique identity");
		const auto before = memory.statistics();
		for (unsigned n = 0; n < 2048; ++n)
		{
			const auto i = (n * 193) % watches.size();
			require(memory.surface_revision(watches[i]) == revisions[i], "Stable indexed watch changed its revision");
		}
		const auto after = memory.statistics();
		require(after.runs - before.runs == 1 && after.watch_visits - before.watch_visits == 512 &&
			after.indexed_lookups - before.indexed_lookups == 2048 && after.reuses - before.reuses == 4095,
			"Stable capture repeated journal/retirement scans or missed indexed lookups");
		require(guest_memory_epoch::identity(*provider) == identity, "Nested reads replaced outer capture identity");
		// Ordinary guarded writes update tags immediately even though maintenance
		// remains reusable. Mapping transitions and journal mutations must break it.
		const std::array<std::byte, 4> bytes{std::byte{0x45}, std::byte{0x67}, std::byte{0x89}, std::byte{0xab}};
		memory.write(watches[0].range, bytes);
		require(memory.surface_revision(watches[0]) > revisions[0] && memory.surface_revision(watches[1]) == revisions[1], "In-epoch byte write bypassed live tags");
		const auto written = memory.surface_revision(watches[0]);
		memory.begin_mapping_change(watches[0].range);
		refused([&] { (void)memory.surface_revision(watches[0]); });
		memory.end_mapping_change(watches[0].range);
		require(memory.surface_revision(watches[0]) > written && memory.surface_revision(watches[511]) == revisions[511], "In-epoch mapping ABA was skipped or changed unrelated tag");
		const auto before_host = memory.surface_revision(watches[0]);
		auto intent = memory.begin_unleased_host_write(watches[0].range);
		refused([&] { (void)memory.surface_revision(watches[0]); });
		intent.commit();
		require(memory.surface_revision(watches[0]) > before_host && !memory.pending_host_writes(), "In-epoch journal completion was skipped");
	});
	require(guest_memory_epoch::identity(*provider) == 0, "Capture identity leaked after teardown");
	provider->execute([&](auto&) { require(guest_memory_epoch::identity(*provider) != identity, "Reused stack view reused a completed epoch identity"); });
	std::cout << "2048 indexed revisions/512 watches in one capture:one maintenance pass,512 retirement visits,4095 reuse hits;in-epoch writes,mapping ABA,active and completed journal checked\n";
}
void partial_recovery_and_retirement()
{
	auto provider = std::make_shared<test::host_access_fixture>();
	guarded_guest_memory memory(provider, 2);
	auto watch = memory.watch_surface(guest_byte_coverage({4096, 4}));
	auto retiring = memory.watch_surface(guest_byte_coverage({8192, 4}));
	const auto prepared = memory.prepare_host_access(watch.range, true);
	auto intent = memory.begin_host_write(prepared);
	provider->execute([&](auto&)
	{
		memory.with_host_write_recovery([&]
		{
			const auto first = memory.surface_revision(watch);
			const auto before = memory.statistics();
			require(memory.surface_revision(watch) == first && memory.statistics().runs == before.runs, "Unchanged partial prefix forced another drain");
			memory.mark_host_write_fault(intent);
			require(memory.surface_revision(watch) > first, "New partial prefix reused a stale journal observation");
		});
		// Same epoch, same active intent, but recovery permission has ended.
		refused([&] { (void)memory.surface_revision(watch); });
		intent.commit();
		(void)memory.surface_revision(watch);
		retiring.registration.reset();
		// Capacity pressure retires expired indexed entries within this epoch.
		auto replacement = memory.watch_surface(guest_byte_coverage({12288, 4}));
		require(bool(replacement.registration), "In-epoch expired watch blocked bounded replacement");
		const auto replacement_revision = memory.surface_revision(replacement);
		replacement.registration.reset();
		auto fresh = memory.watch_surface(guest_byte_coverage({12288, 4}));
		require(memory.surface_revision(fresh) > replacement_revision, "Expired watch index returned its old registration/revision");
	});
	std::cout << "Paused prefix refresh/recovery exit in same capture;bounded capacity and same-shape retirement/re-registration passed\n";
}
}
int main()
{
	try { stable_epoch_and_mutations(); partial_recovery_and_retirement(); }
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
