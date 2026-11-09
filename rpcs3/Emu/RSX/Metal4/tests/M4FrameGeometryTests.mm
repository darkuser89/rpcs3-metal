#include "M4Frame.h"
#include "M4MainRunLoop.h"
#import <AppKit/NSApplication.h>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <thread>

namespace metal4
{
// Exercise the exact pre/post-acquisition checks used by present(), while never
// calling nextDrawable or supplying a fabricated Metal drawable/command queue.
class frame_geometry_test_access
{
  public:
	static auto prepare(native_frame& frame, frame_pacing pacing = frame_pacing::synchronized) { return frame.prepare_acquisition(pacing); }
	template<typename Ticket> static bool current(const native_frame& frame,const Ticket& ticket) { return frame.current_acquisition(ticket); }
};
}

// Selector-contract fixtures are NSObjects, not NSView/NSWindow subclasses.
// They exercise the production bridge without starting AppKit or acquiring a
// drawable. Real host-window/screen behavior remains a separate UI acceptance.
@interface M4GeometryWindow : NSObject
@property(nonatomic) CGFloat backingScaleFactor;
@property(nonatomic,getter=isVisible) BOOL visible;
@end
@implementation M4GeometryWindow
@end
@interface M4GeometryView : NSObject
@property(nonatomic) BOOL wantsLayer;
@property(nonatomic,strong) CALayer* layer;
@property(nonatomic) NSRect bounds;
@property(nonatomic,getter=isHidden) BOOL hidden;
@property(nonatomic) BOOL ancestorHidden;
@property(nonatomic,strong) M4GeometryWindow* window;
@property(nonatomic,copy) void (^onBounds)(void);
@property(nonatomic) unsigned reads;
@end
@implementation M4GeometryView
@synthesize wantsLayer = _wantsLayer;
@synthesize bounds = _bounds;
- (void)setWantsLayer:(BOOL)value
{
	_wantsLayer=value;
	if(value && !self.layer) self.layer=[CALayer layer];
	if(!value) self.layer=nil;
}
- (BOOL)wantsLayer { return _wantsLayer; }
- (NSRect)bounds
{
	++self.reads;
	const auto callback=self.onBounds;
	if(callback) callback();
	return _bounds;
}
- (void)setBounds:(NSRect)value { _bounds=value; }
- (BOOL)isHiddenOrHasHiddenAncestor { return self.hidden || self.ancestorHidden; }
@end

namespace
{
using namespace metal4;
unsigned guards=0;
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
template<typename F> void rejects(F&& fn)
{
	bool failed=false;try{fn();}catch(const std::exception&){failed=true;}
	require(failed,"Invalid frame geometry operation accepted");++guards;
}
class loop_source
{
	CFRunLoopSourceRef m_source;
  public:
	loop_source()
	{
		CFRunLoopSourceContext context{};context.perform=[](void*){};
		m_source=CFRunLoopSourceCreate(kCFAllocatorDefault,0,&context);
		require(m_source,"Run-loop fixture source allocation failed");
		CFRunLoopAddSource(CFRunLoopGetMain(),m_source,kCFRunLoopCommonModes);
	}
	~loop_source(){CFRunLoopSourceInvalidate(m_source);CFRelease(m_source);}
};
void tick(unsigned count=1,CFStringRef mode=kCFRunLoopDefaultMode)
{
	for(unsigned i=0;i<count;++i) (void)CFRunLoopRunInMode(mode,.0001,false);
}
bool same(frame_geometry_statistics a,frame_geometry_statistics b)
{
	return a.epoch==b.epoch&&a.layer_transactions==b.layer_transactions&&a.vsync_writes==b.vsync_writes;
}
void observer_contracts()
{
	rejects([]{main_run_loop_observer invalid({});});
	bool foreign=false;
	std::thread worker([&]{try{main_run_loop_observer invalid([]{});}catch(const std::invalid_argument&){foreign=true;}});worker.join();
	require(foreign,"Foreign-thread observer registration accepted");++guards;
	auto counter=std::make_shared<unsigned>(0);std::weak_ptr<unsigned> weak=counter;
	{
		main_run_loop_observer observer([counter]{require([NSThread isMainThread],"Geometry observed on foreign thread");++*counter;});
		require(*counter==0,"Observer posted an eager frame job");tick(20);
		require(*counter>=20&&observer.active(),"Actual CF main run-loop did not invoke geometry observer");
		const auto before=*counter;
		std::thread renderer([&]{for(unsigned i=0;i<10000;++i)require(observer.active(),"Observer died during render reads");});renderer.join();
		require(*counter==before,"Render reads executed or posted geometry work");
		CFRunLoopAddCommonMode(CFRunLoopGetMain(),CFSTR("M4GeometryTrackingFixture"));
		tick(1,CFSTR("M4GeometryTrackingFixture"));require(*counter>before,"Observer lost a common event-tracking mode");
		counter.reset();require(!weak.expired(),"Observer callback did not own its capture");
	}
	require(weak.expired(),"Removed CF observer retained callback captures");tick(3);
	unsigned errors=0;
	{
		main_run_loop_observer observer([&]{++errors;throw std::runtime_error("Owned geometry observation failure");});
		tick();require(errors==1&&!observer.active(),"CF observer did not contain and disarm a failed callback");
		rejects([&]{observer.rethrow_error();});tick(10);require(errors==1,"Failed CF observer fired again");
	}
	unsigned calls=0,nested=0;
	{
		main_run_loop_observer observer([&]{++calls;const auto before=calls;(void)CFRunLoopRunInMode(kCFRunLoopDefaultMode,0,false);if(calls!=before)++nested;});
		tick();require(calls>0&&nested==0,"Nested run loop recursively executed geometry callback");
	}
	std::cout<<"Real CF run loop:main/common modes,no render-thread dispatch,owned captures,error containment,invalidation and nested callback guard passed\n";
}
void direct_geometry(context& ctx)
{
	native_frame frame(ctx);const frame_geometry first{640.25,480.5,1.5,-7.5,9.25,true};
	frame.update_geometry(first);const auto initial=frame.geometry_statistics();
	for(unsigned i=0;i<10000;++i)frame.update_geometry(first);
	require(same(initial,frame.geometry_statistics())&&initial.layer_transactions==1,"Identical geometry wrote layer or advanced epoch");
	require(frame.extent()==frame_extent{961,721}&&CGRectEqualToRect(frame.layer().frame,CGRectMake(-7.5,9.25,640.25,480.5)),"Retina/bounds-origin geometry differs");
	auto hidden=first;hidden.visible=false;frame.update_geometry(hidden);
	require(frame.geometry_statistics().epoch==initial.epoch+1&&frame.geometry_statistics().layer_transactions==initial.layer_transactions,"Visibility change rewrote layer or missed epoch");
	for(unsigned bad=0;bad<7;++bad)
	{
		auto invalid=first;if(bad==0)invalid.x=std::numeric_limits<double>::quiet_NaN();if(bad==1)invalid.y=std::numeric_limits<double>::infinity();
		if(bad==2)invalid.width=-1;if(bad==3)invalid.height=1e30;if(bad==4)invalid.scale=0;if(bad==5)invalid.scale=std::numeric_limits<double>::quiet_NaN();if(bad==6)invalid.width=std::numeric_limits<double>::infinity();
		const auto before=frame.geometry_statistics();rejects([&]{frame.update_geometry(invalid);});require(same(before,frame.geometry_statistics()),"Invalid geometry partially published");
	}
	std::cout<<"10000 identical geometry publications:zero extra CATransactions/epochs;fractional Retina,negative bounds origin,visibility-only epoch and seven invalid preflights passed\n";
}
M4GeometryView* host()
{
	auto result=[M4GeometryView new];result.window=[M4GeometryWindow new];result.window.backingScaleFactor=1.5;result.window.visible=YES;
	result.bounds=NSMakeRect(10,20,640.25,480.5);return result;
}
void observed_frame(context& ctx)
{
	native_frame frame(ctx);auto view=host();frame.attach((NSView*)view);
	require(frame.attached()&&view.wantsLayer&&[view.layer.sublayers containsObject:frame.layer()],"Native attachment selector contract differs");
	const auto ticket=frame_geometry_test_access::prepare(frame);require(bool(ticket),"Visible host produced no acquisition snapshot");
	const auto initial=frame.geometry_statistics();tick(100);
	require(frame_geometry_test_access::current(frame,*ticket)&&same(initial,frame.geometry_statistics())&&view.reads>=100,"Idle UI observation republished identical layer state");
	const auto reads=view.reads;
	std::thread renderer([&]{for(unsigned i=0;i<10000;++i){require(frame.attached()&&frame.extent()==frame_extent{961,721},"Published geometry read differs");}});renderer.join();
	require(view.reads==reads,"Render snapshot reads called host/UI selectors");
	(void)frame_geometry_test_access::prepare(frame,frame_pacing::unsynchronized);const auto unsynced=frame.geometry_statistics();
	for(unsigned i=0;i<10000;++i)require(bool(frame_geometry_test_access::prepare(frame,frame_pacing::unsynchronized)),"Repeated visible acquisition metadata disappeared");
	require(same(unsynced,frame.geometry_statistics())&&frame_geometry_test_access::current(frame,*ticket),"Identical Vsync policy rewrote layer or invalidated geometry");
	(void)frame_geometry_test_access::prepare(frame,frame_pacing::adaptive);
	require(frame.geometry_statistics().vsync_writes==initial.vsync_writes+2,"Vsync changes did not write exactly once each");
	struct pacing_profile { frame_pacing request; bool sync; };
	const std::array profiles{
		pacing_profile{frame_pacing::unsynchronized, false},
		pacing_profile{frame_pacing::synchronized, true},
		pacing_profile{frame_pacing::adaptive, true}};
	for (const auto& from : profiles)
		for (const auto& to : profiles)
		{
			require(bool(frame_geometry_test_access::prepare(frame, from.request)), "Starting pacing request did not acquire metadata");
			require(frame.layer().displaySyncEnabled == from.sync, "Starting pacing used the wrong layer synchronization");
			const auto before = frame.geometry_statistics();
			require(bool(frame_geometry_test_access::prepare(frame, to.request)), "Pacing transition did not acquire metadata");
			const auto after = frame.geometry_statistics();
			require(frame.layer().displaySyncEnabled == to.sync && after.vsync_writes == before.vsync_writes + (from.sync != to.sync) &&
				after.epoch == before.epoch && after.layer_transactions == before.layer_transactions,
				"Pacing transition changed the wrong layer state");
		}
	const auto before_invalid_pacing = frame.geometry_statistics();
	rejects([&] { (void)frame_geometry_test_access::prepare(frame, static_cast<frame_pacing>(-1)); });
	require(same(before_invalid_pacing, frame.geometry_statistics()), "Invalid pacing changed layer state");
	const auto before_ancestor=frame.geometry_statistics();view.ancestorHidden=YES;tick();
	require(!frame_geometry_test_access::prepare(frame, frame_pacing::unsynchronized)&&!frame_geometry_test_access::current(frame,*ticket)&&
		frame.geometry_statistics().layer_transactions==before_ancestor.layer_transactions&&
		frame.geometry_statistics().vsync_writes==before_ancestor.vsync_writes,"Hidden ancestor retained visible acquisition or rewrote layer");
	std::cout << "Native pacing: nine CAMetalLayer transitions, synchronized adaptive mode, no redundant writes, hidden and invalid guards passed\n";
	view.ancestorHidden=NO;tick();require(bool(frame_geometry_test_access::prepare(frame)),"Ancestor reveal did not restore acquisition metadata");
	for(unsigned i=0;i<36;++i)
	{
		view.bounds=NSMakeRect(int(i)-15,int(i%5)-2,320.125+i,200.25+i*2);view.window.backingScaleFactor=std::array{1.,1.5,2.}[i%3];
		const auto stale=frame_geometry_test_access::prepare(frame);const auto before=frame.geometry_statistics();tick();
		require(stale&&!frame_geometry_test_access::current(frame,*stale),"Resize left an old acquisition snapshot valid");
		const double scale=view.window.backingScaleFactor;const auto bounds=view.bounds;
		require(frame.extent()==plan_drawable_extent(bounds.size.width,bounds.size.height,scale,16384)&&
			CGRectEqualToRect(frame.layer().frame,NSRectToCGRect(bounds))&&frame.layer().contentsScale==scale,"Observed resize/scale/origin differs");
		require(frame.geometry_statistics().epoch==before.epoch+1&&frame.geometry_statistics().layer_transactions==before.layer_transactions+1,"Changed geometry did not publish exactly once");
	}
	const frame_policy policy{{4,3},4./3};const auto before_hide=frame.geometry_statistics();const auto hide_ticket=frame_geometry_test_access::prepare(frame);view.hidden=YES;tick();
	require(hide_ticket&&!frame_geometry_test_access::current(frame,*hide_ticket)&&!frame_geometry_test_access::prepare(frame),"Hide did not invalidate in-flight acquisition snapshot");
	require(!frame.present({},policy)&&frame.geometry_statistics().epoch==before_hide.epoch+1&&frame.geometry_statistics().layer_transactions==before_hide.layer_transactions,"Hidden snapshot attempted acquisition or rewrote geometry");
	view.window.visible=NO;tick();const auto stable=frame.geometry_statistics();view.hidden=NO;tick();require(same(stable,frame.geometry_statistics())&&!frame.present({},policy),"Equivalent hidden/offscreen state churned epoch");
	view.window=nil;tick();require(frame.layer().contentsScale==1&&!frame.present({},policy),"Absent window used stale Retina scale/visibility");
	CALayer* original=view.layer;view.layer=[CALayer layer];tick();
	rejects([&]{(void)frame.present({},policy);});const auto failed=frame.geometry_statistics();tick(20);require(same(failed,frame.geometry_statistics()),"Repeated host error churned geometry epoch");
	view.layer=original;tick();require(!frame.present({},policy),"Recovered hidden host attempted acquisition");
	frame.detach();const auto detached_reads=view.reads;tick(10);require(view.reads==detached_reads&&!view.wantsLayer&&!frame.attached(),"Detach retained observer/backing ownership");
	view.window=[M4GeometryWindow new];view.window.backingScaleFactor=1;view.window.visible=YES;frame.attach((NSView*)view);
	const auto rebound_ticket=frame_geometry_test_access::prepare(frame);require(rebound_ticket&&!frame_geometry_test_access::current(frame,*ticket),"Rebind did not restore visibility or revived an old snapshot");
	const auto rebound=frame.geometry_statistics();tick(5);require(same(rebound,frame.geometry_statistics()),"Rebound same geometry churned publication");
	view.hidden=YES;tick();require(!frame.present({},policy),"Rebound hide attempted acquisition");
	frame.detach();const auto detached=frame.geometry_statistics();frame.detach();require(same(detached,frame.geometry_statistics()),"Identical detach churned epoch");
	frame.attach((NSView*)view);auto replacement=host();replacement.bounds=NSMakeRect(-3,4,71,53);replacement.hidden=YES;
	const auto old_binding=frame_geometry_test_access::prepare(frame);const auto before_reentry=frame.geometry_statistics();
	__weak M4GeometryView* weak_view=view;native_frame* rebound_frame=&frame;
	view.onBounds=^{weak_view.onBounds=nil;rebound_frame->detach();rebound_frame->attach((NSView*)replacement);};tick();
	require(frame.extent()==frame_extent{107,80}&&!frame_geometry_test_access::prepare(frame)&&
		frame.geometry_statistics().layer_transactions==before_reentry.layer_transactions+1,"Old callback overwrote newly rebound geometry");
	if(old_binding)require(!frame_geometry_test_access::current(frame,*old_binding),"Reentrant rebind kept old acquisition current");
	frame.detach();
	// A disappeared weak host and a failed initial sample must leave no half attach.
	M4GeometryView* expired=host();expired.hidden=YES;frame.attach((NSView*)expired);expired=nil;tick();
	require(!frame.attached(),"Frame retained its supposedly weak host");rejects([&]{(void)frame.present({},policy);});frame.detach();
	auto invalid=host();invalid.bounds=NSMakeRect(0,0,-1,10);rejects([&]{frame.attach((NSView*)invalid);});
	require(!frame.attached()&&!invalid.wantsLayer,"Invalid initial geometry left a partial attachment/backing");
	std::cout<<"Production bridge with NSObject selector fixtures:100idle run-loop cycles/10000render reads causezero layer writes;36resize/scale/origin changes publish once;hide/window loss,backing error/recovery,detach/rebind passed without drawable acquisition\n";
}
void running_orphan(context& ctx)
{
	auto frame=std::make_unique<native_frame>(ctx);auto view=host();view.hidden=YES;frame->attach((NSView*)view);CALayer* root=view.layer;
	struct gate {std::mutex mutex;std::condition_variable changed;bool entered=false,released=false;};auto owner=std::make_shared<gate>();
	view.onBounds=^{std::unique_lock lock(owner->mutex);owner->entered=true;owner->changed.notify_all();owner->changed.wait(lock,[&]{return owner->released;});};
	std::thread destroyer([&]
	{
		{std::unique_lock lock(owner->mutex);owner->changed.wait(lock,[&]{return owner->entered;});}
		frame.reset();
		{std::lock_guard lock(owner->mutex);owner->released=true;}owner->changed.notify_all();
	});
	std::exception_ptr failure;
	try { tick(); } catch (...) { failure=std::current_exception(); }
	if(failure)
	{
		{std::lock_guard lock(owner->mutex);owner->entered=owner->released=true;}owner->changed.notify_all();
	}
	destroyer.join();
	if(failure)std::rethrow_exception(failure);
	view.onBounds=nil;const auto reads=view.reads;tick(10);
	require(view.reads==reads&&root.sublayers.count==0,"Destroyed frame retained observer or orphan layer");

}
}
int main()
{
	std::cout.setf(std::ios::unitbuf);
	@autoreleasepool
	{
		try
		{
			require(NSApp==nil,"Test started with App");loop_source source;observer_contracts();context ctx;
			direct_geometry(ctx);observed_frame(ctx);for(unsigned i=0;i<16;++i)running_orphan(ctx);
			std::cout<<"16 actual running CF callbacks survive render-thread frame destruction;owned geometry finishes safely,observers stop,andqueued orphan layer removal drains;no borrowed frame/context\n";
			require(NSApp==nil,"Geometry test started App");
			std::cout<<guards<<" rejection guards passed;no NSView/NSWindow/NSApplication/GSRender/guest VM/drawable acquisition\n";
		}
		catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
	}
}
