#include "M4DMA.h"
#include "M4ScaledImage.h"
#include "M4GuardedGuestMemory.h"
#include <iostream>
#include <mutex>
#include <type_traits>

namespace
{
using namespace metal4;
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
class storage final : public exclusive_guest_memory,public exclusive_guest_memory::view,public std::enable_shared_from_this<storage>
{
	std::mutex mutex;
public:
	std::vector<std::byte> bytes=std::vector<std::byte>(65536,std::byte{0xa5});
	unsigned notifications=0;
	std::shared_ptr<void> retain_backing(std::span<const guest_surface_range>) override { return shared_from_this(); }
	void execute(const std::function<void(view&)>& operation) override
	{
		if(auto* current=guest_memory_epoch::current(*this)){operation(*current);return;}
		std::lock_guard lock(mutex);guest_memory_epoch epoch(*this,*this);operation(*this);
	}
	std::byte* map(guest_surface_range range,bool) override
	{
		require(guest_memory_epoch::current(*this)==this,"Transfer fixture escaped CPU exclusion");
		if(range.address>bytes.size()||range.size>bytes.size()-range.address)throw std::range_error("Transfer fixture mapping exceeds storage");
		return bytes.data()+range.address;
	}
	bool same_storage(guest_surface_range a,guest_surface_range b) override { return overlaps(a,b); }
	void written(guest_surface_range) noexcept override { ++notifications; }
};
template<typename Transfer> void profile(unsigned phase,unsigned shape,unsigned outcome)
{
	using namespace std::chrono_literals;
	constexpr bool dma=std::is_same_v<Transfer,dma_transfer>;
	context ctx(2);auto provider=std::make_shared<storage>();auto memory=std::make_shared<guarded_guest_memory>(provider);
	auto ledger=std::make_shared<heap_budget>(ctx,8u<<20);ctx.set_buffer_budget(ledger);
	// No native surface capture adapter: exercise the native GPU compute plus
	// coherent VM publication route, independently of the native image route.
	surface_cache surfaces(ctx,memory,2u<<20,ledger);
	guest_texture_cache textures(ctx,memory,[](unsigned address,unsigned,std::size_t){return address;},1u<<20,ledger);
	unsigned calls=0,refuse_at=0,validations=0;bool valid=true;
	std::function<void()> retained;
	const cpu_capture_dispatch capture=[&](const auto& operation)
	{
		if(++calls==refuse_at){retained=operation;throw std::range_error("Transfer CPU entry refused");}
		provider->execute([&](auto&){operation();});
	};
	auto transfer=std::make_unique<Transfer>(ctx,surfaces,textures,memory,capture,1u<<20);
	std::vector<std::byte> expected=provider->bytes;
	for(unsigned i=0;i<128;++i)provider->bytes[0x1000+i]=std::byte((i*73+31)&255);
	expected=provider->bytes;
	std::optional<submission> work;
	const auto event=[ctx.device() newSharedEvent];
	struct release { id<MTLSharedEvent> event;~release(){event.signaledValue=1;} } cleanup{event};
	const auto dependencies=[&](context& commands){work=commands.recording_submission();commands.wait_event(event,1);};
	const auto validate=[&]
	{
		require(guest_memory_epoch::current(*provider)==provider.get(),"Transfer validation escaped CPU epoch");
		++validations;if(!valid)throw std::runtime_error("Transfer metadata rejected");
	};
	bool timed=false;
	if constexpr(dma)
	{
		const auto plan=plan_dma({0x1000+58,0x4000,1,2,7,3,-29,31,shape?3u:1u,shape?2u:1u,0});
		const auto frozen=provider->bytes;
		for(unsigned row=0;row<3;++row)for(unsigned column=0;column<7;++column)
			expected[0x4000+row*31+column*plan.destination.stride]=frozen[0x1000+58-row*29+column*plan.source.stride];
		try{transfer->execute(plan,1ms,dependencies,validate);}catch(const submission_wait_timeout&){timed=true;}
	}
	else
	{
		const unsigned bpp=shape?2:4;
		const auto format=shape?scaled_image_format::rgb565:scaled_image_format::argb8;
		const scaled_image_plan plan{plan_guest_surface(0x1000,3,2,bpp,3*bpp+3),plan_guest_surface(0x4000,3,2,bpp,3*bpp+5),format,format,3,2,3,2};
		for(unsigned row=0;row<2;++row)for(unsigned byte=0;byte<3*bpp;++byte)
			expected[0x4000+row*(3*bpp+5)+byte]=provider->bytes[0x1000+row*(3*bpp+3)+byte];
		try{transfer->execute(plan,1ms,dependencies,validate);}catch(const submission_wait_timeout&){timed=true;}
	}
	require(timed&&transfer->has_pending()&&work&&!work->ready(),"Transfer fixture needs genuinely held native GPU work");
	// Source bytes are immutable snapshots. A later CPU source store must not
	// replace the already encoded transfer's input during a publication retry.
	memory->write({0x1000,1},std::array{std::byte{0x6d}});expected[0x1000]=std::byte{0x6d};
	const auto before=provider->bytes;const auto notices=provider->notifications;
	event.signaledValue=1;work->wait();
	const auto dispatches=ctx.encoding_statistics().checked_dispatches;
	refuse_at=calls+phase;bool refused=false;
	try{transfer->synchronize();}catch(const std::range_error&){refused=true;}
	require(refused&&provider->bytes==before&&provider->notifications==notices,"Refused transfer entry changed VM bytes");
	require(transfer->has_pending(),dma?"DMA completed output lost after CPU entry refusal":"Scaled completed output lost after CPU entry refusal");
	require(retained!=nullptr,"Refused transfer entry did not retain its checked callback");
	retained();
	require(provider->bytes==before&&provider->notifications==notices,"Late refused callback performed a transfer write");
	refuse_at=0;
	if(outcome==1)valid=false;
	if(outcome==2)memory->write({0x4000,1},std::array{std::byte{0x79}});
	if(outcome==3)
	{
		transfer.reset();retained();require(provider->bytes==before&&provider->notifications==notices,"Destroyed transfer published a retained result");
	}
	else
	{
		const auto after=provider->bytes;const auto notifications=provider->notifications;bool rejected=false;
		try{transfer->synchronize();}catch(const std::runtime_error&){rejected=true;}
		require(rejected==(outcome!=0)&&!transfer->has_pending(),"Transfer retry did not distinguish refusal from terminal conflict");
		if(outcome)require(provider->bytes==after&&provider->notifications==notifications,"Rejected transfer retry changed a CPU writer");
		else
		{
			require(provider->bytes==expected&&provider->notifications>notices&&validations>=4,"Transfer retry lost frozen output,active bytes or validation");
			const auto published=provider->notifications;const auto captures=calls;
			transfer->synchronize();retained();
			require(provider->bytes==expected&&provider->notifications==published&&calls==captures,"Completed transfer replayed publication");
		}
	}
	require(ctx.encoding_statistics().checked_dispatches==dispatches,"Transfer publication retry resubmitted native GPU work");
}
template<typename Transfer> void matrix()
{
	for(unsigned phase:{1u,2u})for(unsigned shape:{0u,1u})for(unsigned outcome=0;outcome<4;++outcome)profile<Transfer>(phase,shape,outcome);
	std::cout<<(std::is_same_v<Transfer,dma_transfer>?"DMA":"Scaled")<<" publication retry:16 real native GPU profiles;both CPU entry phases,two layouts,frozen source/exact full VM bytes,no duplicate dispatch/store,late callback,metadata/CAS conflict and teardown passed\n";
}
}
int main(int argc,char** argv)
{
	@autoreleasepool
	{
		try
		{
			if(argc==1||std::string_view(argv[1])=="dma")matrix<dma_transfer>();
			if(argc==1||std::string_view(argv[1])=="scaled")matrix<scaled_image_transfer>();
			require(argc==1||(argc==2&&(std::string_view(argv[1])=="dma"||std::string_view(argv[1])=="scaled")),"Unknown transfer publication selector");
			return 0;
		}
		catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
	}
}
