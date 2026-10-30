#include "M4GuardedGuestMemory.h"
#include "M4GuestLabel.h"
#include "M4GuestProducer.h"
#include <iostream>
#include <mutex>
namespace
{
using namespace metal4;
void require(bool v, const char* message)
{
	if (!v)
		throw std::runtime_error(message);
}
class fixture final : public exclusive_guest_memory, public exclusive_guest_memory::view, public std::enable_shared_from_this<fixture>
{
	std::mutex mutex;

  public:
	std::array<std::byte, 65536> bytes{};
	unsigned notifications = 0;
	// Fixture storage never reallocates;the provider owns every physical bank.
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& op) override
	{
		if (auto* current = guest_memory_epoch::current(*this))
		{
			op(*current);
			return;
		}
		std::lock_guard lock(mutex);
		guest_memory_epoch epoch(*this, *this);
		op(*this);
	}
	std::byte* map(guest_surface_range r, bool) override
	{
		require(guest_memory_epoch::current(*this) == this, "Label access escaped CPU exclusion");
		const auto address = r.address % bytes.size();
		if (r.address >= bytes.size() * 2 || r.size > bytes.size() - address)
			throw std::range_error("Label fixture mapping invalid");
		return bytes.data() + address;
	}
	bool same_storage(guest_surface_range a, guest_surface_range b) override { return overlaps(guest_surface_range{a.address % 65536, a.size}, guest_surface_range{b.address % 65536, b.size}); }
	void written(guest_surface_range) noexcept override { ++notifications; }
};
} // namespace
int main()
{
	@autoreleasepool
	{
		try
		{
			context ctx{1};
			auto provider = std::make_shared<fixture>();
			auto memory = std::make_shared<guarded_guest_memory>(provider);
			auto budget = std::make_shared<heap_budget>(ctx, 32 << 20);
			ctx.set_buffer_budget(budget);
			surface_cache surfaces(ctx, memory, 32 << 20, budget);
			guest_texture_cache textures(ctx, memory, [](unsigned a, unsigned, std::size_t)
										 { return a; }, 1 << 20, budget);
			constexpr unsigned base = 4096, address = base + 4, alias = address + 65536, value = 0x12345678;
			const auto watched = memory->watch_surface(guest_byte_coverage({address, 4}));
			const auto mirrored = memory->watch_surface(guest_byte_coverage({alias, 4}));
			const auto surface = surfaces.acquire({base, {4, 4, 1, surface_color::argb8, false}, 16, {}});
			surfaces.clear(surface, 0xf0, 0xff112233, 0, 255, {0, 0, 4, 4});
			std::string calls;
			unsigned stores = 0;
			const auto store = [&](unsigned a, unsigned v)
			{
				calls += 'S';
				++stores;
				memory->write({a, 4}, std::array{std::byte(v >> 24), std::byte(v >> 16), std::byte(v >> 8), std::byte(v)});
			};
			const auto blocked = [&]
			{
				synchronize_guest_producers([&]
											{ calls += 'M'; }, [&]
											{calls+='D';return false; });
				calls += 'R';
			};
			const auto initial = memory->read({base, 64});
			const auto notices = provider->notifications;
			const auto invalidations = textures.statistics().invalidations;
			bool failed = false;
			try
			{
				release_guest_label(surfaces, textures, address, value, blocked, store);
			}
			catch (const std::runtime_error&)
			{
				failed = true;
			}
			require(failed && calls == "MD" && !stores && provider->notifications == notices && memory->read({base, 64}).bytes == initial.bytes && surfaces.status(surface).dirty && textures.statistics().invalidations == invalidations, "Blocked DMA published label or flushed dirty target");
			// Same-value releases must still refuse before notifications/dirty flush.
			calls.clear();
			failed = false;
			try
			{
				release_guest_label(surfaces, textures, address, 0, blocked, store);
			}
			catch (const std::runtime_error&)
			{
				failed = true;
			}
			require(failed && calls == "MD" && !stores && provider->notifications == notices && surfaces.status(surface).dirty, "Same-value blocked DMA bypassed producer drain");
			calls.clear();
			failed = false;
			try
			{
				synchronize_guest_producers([&]
											{calls+='M';throw std::runtime_error("MM failed"); }, [&]
											{calls+='D';return true; });
			}
			catch (const std::runtime_error&)
			{
				failed = true;
			}
			require(failed && calls == "M", "DMA called after MM failure");
			calls.clear();
			failed = false;
			try
			{
				synchronize_guest_producers([&]
											{ calls += 'M'; }, [&]() -> bool
											{calls+='D';throw std::runtime_error("DMA failed"); });
			}
			catch (const std::runtime_error&)
			{
				failed = true;
			}
			require(failed && calls == "MD", "DMA exception was ignored");
			const auto completed = [&]
			{synchronize_guest_producers([&]{calls+='M';},[&]{calls+='D';return true;});calls+='R'; };
			calls.clear();
			release_guest_label(surfaces, textures, address, value, completed, store);
			require(calls == "MDRS" && stores == 1 && !surfaces.status(surface).dirty && !surfaces.status(surface).initialized, "Completed producer ordering differs");
			const auto done = memory->read({base, 64});
			for (unsigned i = 0; i < 64; ++i)
				require(done.bytes[i] == std::byte(i >= 4 && i < 8 ? std::array{0x12u, 0x34u, 0x56u, 0x78u}[i - 4] : std::array{255u, 17u, 34u, 51u}[i % 4]), "Normal label bytes/padding differ");
			// Flip release is CPU-only even while native GPU work is gated.
			surfaces.prepare(surface);
			const auto event = [ctx.device() newSharedEvent];
			struct release
			{
				id<MTLSharedEvent> event;
				~release() { event.signaledValue = 1; }
			} cleanup{event};
			surfaces.set_wait_timeout(std::chrono::milliseconds(1));
			const std::array uses{cached_surface_use{surface, true}};
			failed = false;
			try
			{
				surfaces.draw(uses, [&](auto& commands, auto)
							  { commands.wait_event(event, 1); });
			}
			catch (const std::runtime_error&)
			{
				failed = true;
			}
			require(failed && surfaces.status(surface).pending, "Flip fixture needs gated native work");
			const auto before = memory->surface_revision(watched), mirror_before = memory->surface_revision(mirrored);
			calls.clear();
			store_guest_flip_label(alias, 0xcafebabe, store);
			require(calls == "S" && event.signaledValue == 0 && surfaces.status(surface).pending && memory->surface_revision(watched) > before && memory->surface_revision(mirrored) > mirror_before, "Flip label waited for GPU or omitted physical alias revisions");
			const auto word = memory->read({address, 4}).bytes;
			require(word == std::vector<std::byte>{std::byte{0xca}, std::byte{0xfe}, std::byte{0xba}, std::byte{0xbe}}, "Flip BE word differs");
			const auto same_before = provider->notifications;
			store_guest_flip_label(alias, 0xcafebabe, store);
			require(provider->notifications > same_before, "Same-value flip omitted writer notification");
			event.signaledValue = 1;
			surfaces.set_wait_timeout(std::chrono::seconds(30));
			failed = false;
			try
			{
				surfaces.synchronize();
			}
			catch (const surface_conflict_error&)
			{
				failed = true;
			}
			require(failed && !surfaces.status(surface).pending && !surfaces.status(surface).faulted && surfaces.status(surface).spilled && !surfaces.status(surface).initialized &&
				memory->read({address, 4}).bytes == word, "GPU completion accepted a flip-invalidated surface or overwrote the label");
			surfaces.flush(surface);
			require(memory->read({address, 4}).bytes == word, "Rejected GPU output was published by a later flush");
			surfaces.prepare(surface);
			require(surfaces.status(surface).initialized && !surfaces.status(surface).dirty && !surfaces.status(surface).faulted &&
				memory->read({address, 4}).bytes == word, "Clean rollback did not reload the newer flip-label backing");
			const auto stores_before = stores;
			failed = false;
			try
			{
				store_guest_flip_label(address + 1, value, store);
			}
			catch (const std::invalid_argument&)
			{
				failed = true;
			}
			require(failed && stores == stores_before, "Misaligned flip reached store");
			failed = false;
			try
			{
				store_guest_flip_label(address, value, {});
			}
			catch (const std::invalid_argument&)
			{
				failed = true;
			}
			require(failed, "Missing flip writer accepted");
			const auto snapshot = provider->bytes;
			failed = false;
			try
			{
				store_guest_flip_label(address, value, [](auto, auto)
									   { throw std::range_error("Flip writer preflight"); });
			}
			catch (const std::range_error&)
			{
				failed = true;
			}
			require(failed && provider->bytes == snapshot, "Failed flip writer changed bytes");
			std::cout << "Native label producer:blocked DMA/same-value/MM failure/DMA exception refuse before store,dirty target flush or cache invalidation;success MM-DMA-reports-store ordering/BE padding;flip release publishes during gated GPU work with physical alias revisions/same-value notification,GPU completion rejects stale revision and faults output before later writeback/reuse;misaligned/missing/failing writers refused;no App/UI/guest VM initialized\n";
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
	}
}
