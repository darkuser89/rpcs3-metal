#pragma once

#include "M4Device.h"
#include <optional>
#include <string_view>

namespace metal4
{
struct compiler_job_state;
class compiler_function;
enum class compiler_job_kind { library, compute, render };
struct compiler_queue_limits
{
	std::size_t active = 2, jobs = 64, source_bytes = 16u << 20;
};
struct compiler_queue_statistics
{
	std::size_t active = 0, queued = 0, source_bytes = 0, peak_active = 0, waiting = 0;
	std::uint64_t admitted = 0, deduplicated = 0, completed = 0, failed = 0, cancelled = 0;
};
// Completion owns its result independently of compiler/context lifetime. A
// timeout never cancels a native task or releases its immutable inputs early.
class compiler_job final
{
	friend class compiler_queue;
	friend class compiler_function;
	std::shared_ptr<compiler_job_state> m_state;
	explicit compiler_job(std::shared_ptr<compiler_job_state> state) : m_state(std::move(state)) {}
	[[nodiscard]] id result(compiler_job_kind kind, std::chrono::milliseconds timeout) const;
  public:
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool cancelled() const;
	[[nodiscard]] id<MTLLibrary> library(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
	[[nodiscard]] id<MTLComputePipelineState> compute(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
	[[nodiscard]] id<MTLRenderPipelineState> render(std::chrono::milliseconds timeout = std::chrono::seconds(30)) const;
	// Immutable function dependency; no wait or cache access. The producing
	// queue must also own the consuming dependent pipeline job.
	[[nodiscard]] compiler_function function(NSString* name) const;
};

class compiler_function final
{
	friend class compiler_job;
	friend class compiler_queue;
	std::shared_ptr<compiler_job_state> m_job;
	id<MTLLibrary> m_library;
	NSString* m_name;
	compiler_function(std::shared_ptr<compiler_job_state> job, id<MTLLibrary> library, NSString* name);
	[[nodiscard]] id<MTLLibrary> library() const;
  public:
	[[nodiscard]] static compiler_function from_library(id<MTLLibrary> library, NSString* name);
	[[nodiscard]] bool ready() const;
	[[nodiscard]] bool same_function(const compiler_function& other) const;
	// Nonblocking resolution; callers must observe ready() first.
	[[nodiscard]] MTL4LibraryFunctionDescriptor* descriptor() const;
};

// Thread-safe admission; all native asynchronous MTL4Compiler entry points run
// on a launch worker. Limits cover queued/running jobs and logical MSL bytes,
// excluding opaque driver storage and caller-retained completed handles. Exact
// source bytes deduplicate in-flight library jobs under fixed safe MSL4 options.
// Native descriptor copy semantics freeze pipeline/function subdescriptors.
class compiler_queue final
{
	struct implementation;
	std::shared_ptr<implementation> m_impl;
  public:
	explicit compiler_queue(id<MTL4Compiler> compiler, compiler_queue_limits limits = {});
	~compiler_queue();
	compiler_queue(const compiler_queue&) = delete;
	compiler_queue& operator=(const compiler_queue&) = delete;
	// Zero admission timeout is nonblocking; nullopt means bounded capacity is
	// currently full. Oversized/invalid inputs and a closed queue always throw.
	[[nodiscard]] std::optional<compiler_job> library(std::string_view source, NSString* label = nil,
		std::chrono::milliseconds admission_timeout = std::chrono::milliseconds(0));
	[[nodiscard]] std::optional<compiler_job> compute(MTL4ComputePipelineDescriptor* descriptor,
		std::chrono::milliseconds admission_timeout = std::chrono::milliseconds(0));
	[[nodiscard]] std::optional<compiler_job> render(MTL4RenderPipelineDescriptor* descriptor,
		std::chrono::milliseconds admission_timeout = std::chrono::milliseconds(0));
	// Descriptor function slots must be empty. Pending library dependencies
	// occupy queued capacity, not active compiler slots. Ready independent
	// work can pass them; failure/cancellation never launches a native PSO.
	[[nodiscard]] std::optional<compiler_job> compute(MTL4ComputePipelineDescriptor* descriptor,
		const compiler_function& function, std::chrono::milliseconds admission_timeout = std::chrono::milliseconds(0));
	[[nodiscard]] std::optional<compiler_job> render(MTL4RenderPipelineDescriptor* descriptor,
		const compiler_function& vertex, const std::optional<compiler_function>& fragment,
		std::chrono::milliseconds admission_timeout = std::chrono::milliseconds(0));
	[[nodiscard]] compiler_queue_statistics statistics() const;
	// Refuse new work and cancel tasks before their worker launch commit point.
	// Metal4 offers no native task cancellation: committed launches retain their
	// inputs until callback, including across a shutdown timeout or destruction.
	// False means those tasks still run; a later call can finish the bounded join.
	[[nodiscard]] bool shutdown(std::chrono::milliseconds timeout = std::chrono::seconds(30));
};
} // namespace metal4
