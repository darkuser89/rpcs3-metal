#pragma once

#include "M4Device.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>

// Test-only callback gate around the real native compiler. It never fabricates
// a library/PSO: every result below comes from an actual MTL4CompilerTask.
struct M4HeldCompilerCallback { dispatch_block_t block; NSString* label; };
@interface M4GatedCompiler : NSObject
{
	id<MTL4Compiler> _native;
	std::mutex _mutex;
	std::condition_variable _changed;
	std::deque<M4HeldCompilerCallback> _callbacks;
	bool _holding;
	std::atomic<bool> _mainThreadLaunch;
	std::atomic<NSUInteger> _libraryLaunches, _computeLaunches, _renderLaunches;
}
- (instancetype)initWithNative:(id<MTL4Compiler>)native;
- (id<MTLDevice>)device;
- (void)deliver:(dispatch_block_t)callback;
- (void)deliver:(dispatch_block_t)callback label:(NSString*)label;
- (void)releaseLibraryCallback:(NSString*)label;
- (void)waitForCallbacks:(std::size_t)count;
- (void)releaseCallbacks;
- (bool)mainThreadLaunch;
- (NSUInteger)libraryLaunches;
- (NSUInteger)computeLaunches;
- (NSUInteger)renderLaunches;
@end

@implementation M4GatedCompiler
- (instancetype)initWithNative:(id<MTL4Compiler>)native
{
	if ((self = [super init])) { _native = native; _holding = true; _mainThreadLaunch = false; _libraryLaunches = _computeLaunches = _renderLaunches = 0; }
	return self;
}
- (id<MTLDevice>)device { return _native.device; }
- (void)deliver:(dispatch_block_t)callback
{
	[self deliver:callback label:nil];
}
- (void)deliver:(dispatch_block_t)callback label:(NSString*)label
{
	{
		std::lock_guard lock(_mutex);
		if (_holding) { _callbacks.push_back({[callback copy],[label copy]}); _changed.notify_all(); return; }
	}
	callback();
}
- (void)releaseLibraryCallback:(NSString*)label
{
	dispatch_block_t block;
	{
		std::lock_guard lock(_mutex);
		const auto found=std::find_if(_callbacks.begin(),_callbacks.end(),[&](const auto& callback){return [callback.label isEqualToString:label];});
		if(found==_callbacks.end())throw std::runtime_error("Missing held native library callback");
		block=found->block;_callbacks.erase(found);
	}
	block();
}
- (void)waitForCallbacks:(std::size_t)count
{
	std::unique_lock lock(_mutex);
	if (!_changed.wait_for(lock, std::chrono::seconds(10), [&] { return _callbacks.size() >= count; }))
		throw std::runtime_error("Timed out waiting for real native compiler callbacks");
}
- (void)releaseCallbacks
{
	std::deque<M4HeldCompilerCallback> callbacks;
	{ std::lock_guard lock(_mutex); _holding = false; callbacks.swap(_callbacks); }
	for (const auto& callback : callbacks) callback.block();
}
- (bool)mainThreadLaunch { return _mainThreadLaunch; }
- (NSUInteger)libraryLaunches { return _libraryLaunches; }
- (NSUInteger)computeLaunches { return _computeLaunches; }
- (NSUInteger)renderLaunches { return _renderLaunches; }
- (id<MTL4CompilerTask>)newLibraryWithDescriptor:(MTL4LibraryDescriptor*)descriptor completionHandler:(MTLNewLibraryCompletionHandler)completion
{
	if ([NSThread isMainThread]) _mainThreadLaunch = true;
	++_libraryLaunches;
	NSString* label=[descriptor.name copy];
	return [_native newLibraryWithDescriptor:descriptor completionHandler:^(id<MTLLibrary> result, NSError* error)
		{ [self deliver:^{ completion(result, error); } label:label]; }];
}
- (id<MTL4CompilerTask>)newComputePipelineStateWithDescriptor:(MTL4ComputePipelineDescriptor*)descriptor compilerTaskOptions:(MTL4CompilerTaskOptions*)options completionHandler:(MTLNewComputePipelineStateCompletionHandler)completion
{
	if ([NSThread isMainThread]) _mainThreadLaunch = true;
	++_computeLaunches;
	return [_native newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:options completionHandler:^(id<MTLComputePipelineState> result, NSError* error)
		{ [self deliver:^{ completion(result, error); }]; }];
}
- (id<MTL4CompilerTask>)newRenderPipelineStateWithDescriptor:(MTL4PipelineDescriptor*)descriptor compilerTaskOptions:(MTL4CompilerTaskOptions*)options completionHandler:(MTLNewRenderPipelineStateCompletionHandler)completion
{
	if ([NSThread isMainThread]) _mainThreadLaunch = true;
	++_renderLaunches;
	return [_native newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:options completionHandler:^(id<MTLRenderPipelineState> result, NSError* error)
		{ [self deliver:^{ completion(result, error); }]; }];
}
@end

