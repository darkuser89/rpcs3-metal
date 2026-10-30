#include "M4Device.h"
#include "M4HeapBudget.h"
#include "M4ConstantUpload.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <bit>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <dispatch/dispatch.h>

namespace metal4
{

struct completion_state
{
	enum class phase { recording, queued, committing, finished } progress = phase::recording;
	std::mutex mutex;
	std::condition_variable changed;
	bool completed = false;
	bool gpu_completed = false;
	bool commit_finished = false;
	bool cancelled = false;
	bool cancel_requested = false;
	NSError* error = nil;
};

namespace
{

[[noreturn]] void fail(const char* operation, NSError* error = nil)
{
	std::string message = operation;
	if (error)
	{
		message += ": ";
		message += error.localizedDescription.UTF8String;
	}
	throw std::runtime_error(message);
}

struct command_slot
{
	struct upload_page
	{
		id<MTLBuffer> buffer;
		std::size_t used = 0;
	};

	id<MTL4CommandQueue> queue;
	id<MTL4CommandAllocator> allocator;
	id<MTL4CommandBuffer> commands;
	id<MTLResidencySet> residency;
	NSHashTable* retained = [NSHashTable hashTableWithOptions:
		NSPointerFunctionsStrongMemory | NSPointerFunctionsObjectPointerPersonality];
	id<CAMetalDrawable> drawable = nil;
	std::shared_ptr<completion_state> completion;
	std::shared_ptr<completion_state> observation;
	std::vector<upload_page> upload_pages;
	std::vector<id<MTL4ArgumentTable>> argument_tables;
	std::size_t argument_table_index = 0;
	std::shared_ptr<heap_budget> buffer_budget;
	bool retire_upload_pages = false; // Protected by completion->mutex after submit.
	std::size_t upload_page_index = 0;
	struct gpu_check
	{
		id<MTLBuffer> buffer;
		std::size_t offset;
		NSString* message;
	};
	std::vector<gpu_check> checks;
	std::vector<std::shared_ptr<void>> owners;
	struct event_wait
	{
		id<MTLEvent> event;
		std::uint64_t value;
	};
	std::vector<event_wait> waits;
};

}

struct upload_plan_state
{
	struct region { id<MTLBuffer> buffer; std::size_t begin, end; };
	std::shared_ptr<void> identity;
	std::optional<submission> dependency;
	std::vector<upload_requirement> requirements;
	std::vector<upload_slice> slices;
	std::vector<region> regions;
	std::size_t next = 0;
	bool consumed = false;
};

submission::submission(std::shared_ptr<completion_state> state)
	: m_state(std::move(state))
{
}

bool submission::ready() const
{
	std::lock_guard lock(m_state->mutex);
	return m_state->completed;
}
bool submission::cancelled() const
{
	std::lock_guard lock(m_state->mutex);
	return m_state->cancelled;
}

bool submission::cancel_pending() const
{
	std::lock_guard lock(m_state->mutex);
	if (m_state->progress != completion_state::phase::queued) return false;
	m_state->cancel_requested = true;
	return true;
}

void submission::wait(std::chrono::milliseconds timeout) const
{
	std::unique_lock lock(m_state->mutex);
	if (!m_state->changed.wait_for(lock, timeout, [this] { return m_state->completed; }))
	{
		fail("Timed out waiting for Metal 4 submission");
	}
	if (m_state->error)
	{
		fail("Metal 4 submission failed", m_state->error);
	}
}

struct context::impl
{

	id<MTLDevice> device;
	id<MTL4CommandQueue> queue;
	dispatch_queue_t submit_queue;
	std::vector<std::shared_ptr<command_slot>> slots;
	std::size_t next_slot = 0;
	std::shared_ptr<command_slot> recording;
	id<MTL4CommandEncoder> encoder = nil;
	id<MTL4ComputeCommandEncoder> compute_encoder = nil;
	id<MTL4RenderCommandEncoder> render_encoder = nil;
	MTLStages encoder_stages = 0;
	std::shared_ptr<heap_budget> buffer_budget;
	std::shared_ptr<void> upload_identity = std::make_shared<int>(0);
	std::shared_ptr<upload_plan_state> active_uploads;
	std::weak_ptr<upload_plan_state> admitted_uploads;
	bool allocated_buffer = false;
	std::size_t argument_tables_per_slot = 0;
	argument_table_statistics argument_stats;
	command_encoding_statistics encoding_stats;
	std::chrono::milliseconds event_wait_timeout;
	bool stopped = false;
	std::shared_ptr<std::atomic_bool> stop_submissions = std::make_shared<std::atomic_bool>(false);

	command_slot& active() const
	{
		if (!recording)
		{
			throw std::logic_error("No Metal 4 command buffer is recording");
		}
		return *recording;
	}
};

context::context(std::uint32_t max_in_flight, std::size_t argument_tables_per_slot,
	std::chrono::milliseconds event_wait_timeout)
	: m_impl(std::make_unique<impl>())
{
	@autoreleasepool
	{
		if (max_in_flight == 0 || max_in_flight > 64)
		{
			throw std::invalid_argument("Metal 4 in-flight slot count must be in [1, 64]");
		}
		if (argument_tables_per_slot > 4096)
			throw std::invalid_argument("Metal4 argument table capacity exceeds the per-slot limit");
		if (event_wait_timeout <= std::chrono::milliseconds::zero() || event_wait_timeout > std::chrono::minutes(5))
			throw std::invalid_argument("Metal4 shared-event deadline must be positive and at most five minutes");
		m_impl->event_wait_timeout = event_wait_timeout;
		m_impl->argument_tables_per_slot = argument_tables_per_slot;
		m_impl->argument_stats.capacity = max_in_flight * argument_tables_per_slot;
		if (@available(macOS 26.0, *))
		{
			m_impl->device = MTLCreateSystemDefaultDevice();
			if (!m_impl->device) fail("No Metal GPU is available to the native backend");
			if (!m_impl->device.hasUnifiedMemory) fail("Native Metal 4 requires Apple unified memory");
			if (![m_impl->device supportsFamily:MTLGPUFamilyApple7])
				fail("Native Metal 4 requires an Apple7 or newer Apple Silicon GPU");
			if (![m_impl->device supportsFamily:MTLGPUFamilyMetal4])
				fail("The Apple GPU does not advertise Metal4 GPU-family support");

			m_impl->queue = [m_impl->device newMTL4CommandQueue];
			m_impl->submit_queue = dispatch_queue_create("RPCS3 native Metal4 submission", DISPATCH_QUEUE_SERIAL);
			if (!m_impl->queue || !m_impl->submit_queue)
			{
				fail("Failed to create Metal 4 command queue");
			}

			for (std::uint32_t i = 0; i < max_in_flight; ++i)
			{
				auto slot = std::make_shared<command_slot>();
				slot->queue = m_impl->queue;
				slot->allocator = [m_impl->device newCommandAllocator];
				slot->commands = [m_impl->device newCommandBuffer];
				MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
				descriptor.label = @"RPCS3 Metal 4 command resources";
				descriptor.initialCapacity = 256;
				NSError* error = nil;
				slot->residency = [m_impl->device newResidencySetWithDescriptor:descriptor error:&error];
				if (!slot->allocator || !slot->commands || !slot->residency)
				{
					fail("Failed to allocate Metal 4 command slot", error);
				}
				m_impl->slots.push_back(std::move(slot));
			}
		}
		else
		{
			fail("Native Metal 4 requires macOS 26 or later");
		}
	}
}

// Pending feedback blocks own their slots, including allocators and resources.
// Destruction does not invalidate work that is still executing on the GPU.
context::~context()
{
	@autoreleasepool
	{
		if (m_impl->recording)
		{
			cancel();
		}
		// Native feedback blocks can outlive completed work and still own the
		// slot. Drop reusable upload pages explicitly once no GPU can use them.
		// Pending slots retain their pages and ledger until actual completion.
		for (auto& slot : m_impl->slots)
			if (slot->completion)
			{
				std::lock_guard lock(slot->completion->mutex);
				slot->retire_upload_pages = true;
				if (slot->completion->gpu_completed)
				{
					slot->upload_pages.clear();
					slot->argument_tables.clear();
				}
			}
			else
			{
				slot->upload_pages.clear();
				slot->argument_tables.clear();
			}
	}
}

id<MTLDevice> context::device() const noexcept
{
	return m_impl->device;
}

void context::set_buffer_budget(std::shared_ptr<heap_budget> budget)
{
	if (!budget || budget->device() != device())
		throw std::invalid_argument("Metal4 buffer budget requires this device");
	if (m_impl->buffer_budget || m_impl->allocated_buffer || m_impl->recording)
		throw std::logic_error("Metal4 buffer budget must be installed before allocation/recording");
	m_impl->buffer_budget = std::move(budget);
	for (auto& slot : m_impl->slots)
		slot->buffer_budget = m_impl->buffer_budget;
}

std::shared_ptr<heap_budget> context::buffer_ledger() const noexcept { return m_impl->buffer_budget; }

id<MTLBuffer> context::allocate_buffer(std::size_t bytes, MTLStorageMode storage)
{
	if (!bytes || bytes > device().maxBufferLength || (storage != MTLStorageModeShared && storage != MTLStorageModePrivate))
		throw std::invalid_argument("Metal4 buffer size or storage is invalid");
	id<MTLBuffer> result;
	if (m_impl->buffer_budget)
	{
		try
		{
			result = m_impl->buffer_budget->buffer(bytes, storage);
		}
		catch (const heap_budget_error&)
		{
			if (!trim_upload_pages()) throw;
			// One retry after reclaiming idle storage,never a GPU wait/retry loop.
			result = m_impl->buffer_budget->buffer(bytes, storage);
		}
	}
	else
	{
		const auto options = storage == MTLStorageModeShared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate;
		result = [device() newBufferWithLength:bytes options:options];
		if (!result) fail("Metal4 buffer allocation failed");
	}
	m_impl->allocated_buffer = true;
	return result;
}

constant_upload_cache& context::constant_uploads()
{
	if (!m_constants) m_constants = std::make_unique<constant_upload_cache>(*this);
	return *m_constants;
}
constant_upload_cache& context::vertex_uploads()
{
	if (!m_vertex_uploads) m_vertex_uploads = std::make_unique<constant_upload_cache>(*this,64,8u << 20);
	return *m_vertex_uploads;
}
constant_upload_cache& context::geometry_uploads()
{
	if (!m_geometry_uploads) m_geometry_uploads = std::make_unique<constant_upload_cache>(*this,64,8u << 20);
	return *m_geometry_uploads;
}

std::size_t context::trim_upload_pages()
{
	std::size_t removed = 0;
	if (m_geometry_uploads)
	{
		removed += m_geometry_uploads->statistics().entries;
		m_geometry_uploads->clear();
	}
	if (m_vertex_uploads)
	{
		removed += m_vertex_uploads->statistics().entries;
		m_vertex_uploads->clear();
	}
	if (m_constants)
	{
		removed += m_constants->statistics().entries;
		m_constants->clear();
	}
	@autoreleasepool
	{
		for (auto& slot : m_impl->slots)
		{
			if (slot == m_impl->recording)
			{
				const auto before = slot->upload_pages.size();
				std::erase_if(slot->upload_pages, [](const auto& page) { return page.used == 0; });
				removed += before - slot->upload_pages.size();
				// Rescan retained pages using their immutable used-byte boundaries.
				slot->upload_page_index = 0;
			}
			else if (slot->completion)
			{
				std::lock_guard lock(slot->completion->mutex);
				if (!slot->completion->completed) continue;
				removed += slot->upload_pages.size();
				slot->upload_pages.clear();
				slot->upload_page_index = 0;
			}
			else
			{
				removed += slot->upload_pages.size();
				slot->upload_pages.clear();
				slot->upload_page_index = 0;
			}
		}
	}
	return removed;
}

id<MTL4ArgumentTable> context::argument_table()
{
	auto& slot = m_impl->active();
	id<MTL4ArgumentTable> table;
	if (slot.argument_table_index < slot.argument_tables.size())
	{
		table = slot.argument_tables[slot.argument_table_index++];
		// Resource IDs and addresses outlive neither their bindings nor their
		// residency leases. Reset every slot, including dynamic attribute strides.
		for (NSUInteger i = 0; i < 31; ++i) [table setAddress:0 attributeStride:0 atIndex:i];
		for (NSUInteger i = 0; i < 128; ++i) [table setTexture:MTLResourceID{} atIndex:i];
		for (NSUInteger i = 0; i < 16; ++i) [table setSamplerState:MTLResourceID{} atIndex:i];
		++m_impl->argument_stats.reused;
	}
	else
	{
		auto descriptor = [MTL4ArgumentTableDescriptor new];
		descriptor.maxBufferBindCount = 31;
		descriptor.maxTextureBindCount = 128;
		descriptor.maxSamplerStateBindCount = 16;
		descriptor.initializeBindings = YES;
		descriptor.supportAttributeStrides = YES;
		NSError* error = nil;
		table = [device() newArgumentTableWithDescriptor:descriptor error:&error];
		if (!table) fail("Metal4 argument table allocation failed", error);
		if (slot.argument_tables.size() < m_impl->argument_tables_per_slot)
		{
			slot.argument_tables.push_back(table);
			++slot.argument_table_index;
		}
		else ++m_impl->argument_stats.bypassed;
		++m_impl->argument_stats.created;
	}
	keep_alive(table);
	return table;
}

argument_table_statistics context::argument_statistics() const noexcept
{
	auto result = m_impl->argument_stats;
	for (const auto& slot : m_impl->slots) result.cached += slot->argument_tables.size();
	return result;
}
command_encoding_statistics context::encoding_statistics() const noexcept { return m_impl->encoding_stats; }

void context::validate_begin() const
{
	if (m_impl->stopped) throw std::logic_error("Metal4 context has been shut down");
	if (m_impl->recording)
	{
		throw std::logic_error("Metal 4 command buffer is already recording");
	}
}

void context::begin()
{
	validate_begin();
	auto slot = m_impl->slots[m_impl->next_slot];
	if (slot->completion)
	{
		submission(slot->completion).wait();
	}
	// Only completed command storage may be reset, even when a wait times out.
	[slot->allocator reset];
	[slot->residency removeAllAllocations];
	[slot->residency commit];
	[slot->retained removeAllObjects];
	slot->drawable = nil;
	slot->completion.reset();
	slot->observation.reset();
	slot->checks.clear();
	slot->owners.clear();
	slot->waits.clear();
	for (auto& page : slot->upload_pages) page.used = 0;
	slot->upload_page_index = 0;
	slot->argument_table_index = 0;
	[slot->commands beginCommandBufferWithAllocator:slot->allocator];
	m_impl->recording = std::move(slot);
}

id<MTL4ComputeCommandEncoder> context::compute()
{
	auto& slot = m_impl->active();
	if (m_impl->compute_encoder)
	{
		return m_impl->compute_encoder;
	}
	end_encoding();
	auto encoder = [slot.commands computeCommandEncoder];
	if (!encoder)
	{
		fail("Failed to create Metal 4 compute encoder");
	}
	m_impl->encoder = encoder;
	m_impl->compute_encoder = encoder;
	++m_impl->encoding_stats.compute_encoders;
	m_impl->encoder_stages = MTLStageDispatch | MTLStageBlit | MTLStageAccelerationStructure;
	// Start conservatively. A resource hazard tracker can narrow this later.
	[encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
		visibilityOptions:MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias];
	return encoder;
}

id<MTL4RenderCommandEncoder> context::render(MTL4RenderPassDescriptor* descriptor)
{
	auto& slot = m_impl->active();
	if (!descriptor || descriptor.rasterizationRateMap)
	{
		throw std::invalid_argument("Metal 4 render pass requires a descriptor without a rasterization rate map");
	}
	for (NSUInteger i = 0; i < 8; ++i)
	{
		auto attachment = descriptor.colorAttachments[i];
		if (attachment.texture) use_resource(attachment.texture);
		if (attachment.resolveTexture) use_resource(attachment.resolveTexture);
	}
	if (descriptor.depthAttachment.texture) use_resource(descriptor.depthAttachment.texture);
	if (descriptor.depthAttachment.resolveTexture) use_resource(descriptor.depthAttachment.resolveTexture);
	if (descriptor.stencilAttachment.texture) use_resource(descriptor.stencilAttachment.texture);
	if (descriptor.stencilAttachment.resolveTexture) use_resource(descriptor.stencilAttachment.resolveTexture);
	if (descriptor.visibilityResultBuffer) use_resource(descriptor.visibilityResultBuffer);
	end_encoding();
	auto encoder = [slot.commands renderCommandEncoderWithDescriptor:descriptor];
	if (!encoder)
	{
		fail("Failed to create Metal 4 render encoder");
	}
	m_impl->encoder = encoder;
	m_impl->render_encoder = encoder;
	++m_impl->encoding_stats.render_encoders;
	m_impl->encoder_stages = MTLStageVertex | MTLStageFragment | MTLStageTile;
	[encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll
		visibilityOptions:MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias];
	return encoder;
}

id<MTL4RenderCommandEncoder> context::current_render_encoder() const noexcept
{
	return m_impl->render_encoder;
}

void context::end_encoding()
{
	[m_impl->encoder endEncoding];
	m_impl->encoder = nil;
	m_impl->compute_encoder = nil;
	m_impl->render_encoder = nil;
	m_impl->encoder_stages = 0;
}

void context::barrier(MTLStages after, MTLStages before, bool resource_alias)
{
	if (!m_impl->encoder || !after || !before ||
		(after & ~m_impl->encoder_stages) || (before & ~m_impl->encoder_stages))
	{
		throw std::invalid_argument("Metal 4 encoder barrier stages do not match the active pass");
	}
	const auto visibility = resource_alias
		? MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias
		: MTL4VisibilityOptionDevice;
	[m_impl->encoder barrierAfterEncoderStages:after beforeEncoderStages:before visibilityOptions:visibility];
}

void context::keep_alive(id object)
{
	auto& slot = m_impl->active();
	if (!object)
	{
		throw std::invalid_argument("Cannot retain a nil Metal 4 object");
	}
	[slot.retained addObject:object];
}

void context::use_resource(id<MTLResource> resource)
{
	auto& slot = m_impl->active();
	if (!resource || resource.device != m_impl->device)
	{
		throw std::invalid_argument("Metal 4 resource must belong to the context device");
	}
	keep_alive(resource);
	// Texture views and buffer-backed textures share their owner's allocation.
	if ([resource conformsToProtocol:@protocol(MTLTexture)])
	{
		id<MTLTexture> texture = static_cast<id<MTLTexture>>(resource);
		if (texture.parentTexture)
		{
			use_resource(texture.parentTexture);
			return;
		}
		if (texture.buffer)
		{
			use_resource(texture.buffer);
			return;
		}
	}
	// Heap-backed resources require the backing heap to remain resident too.
	if (resource.heap)
	{
		keep_alive(resource.heap);
		[slot.residency addAllocation:resource.heap];
	}
	else
	{
		[slot.residency addAllocation:resource];
	}
}

void context::keep_alive(std::shared_ptr<void> owner)
{
	auto& slot = m_impl->active();
	if (!owner) throw std::invalid_argument("Cannot retain an empty Metal 4 allocation lease");
	if (std::none_of(slot.owners.begin(),slot.owners.end(),[&](const auto& item){return item.get()==owner.get();}))
		slot.owners.push_back(std::move(owner));
}

void context::require_gpu_success(id<MTLBuffer> status, std::size_t offset, NSString* message)
{
	auto& slot = m_impl->active();
	if (!status || status.device != device() || status.storageMode != MTLStorageModeShared || offset % 4 ||
		offset > status.length || 4 > status.length - offset || !message.length)
		throw std::invalid_argument("Metal 4 GPU validation requires a shared uint32 status and message");
	if (slot.drawable) throw std::logic_error("GPU validation must complete before drawable presentation");
	use_resource(status);
	slot.checks.push_back({status,offset,message});
}

void context::target_drawable(id<CAMetalDrawable> drawable)
{
	auto& slot = m_impl->active();
	if (!drawable || slot.drawable)
	{
		throw std::invalid_argument("Select exactly one drawable per Metal 4 submission");
	}
	use_resource(drawable.texture);
	slot.drawable = drawable;
}

void context::wait_event(id<MTLEvent> event, std::uint64_t value)
{
	auto& slot = m_impl->active();
	// Shared events have nil device and can synchronize across devices.
	if (!event || (event.device && event.device != device()))
		throw std::invalid_argument("Metal 4 queue wait event belongs to another device");
	keep_alive(event);
	slot.waits.push_back({event, value});
}

upload_slice context::upload(std::span<const std::byte> bytes, std::size_t alignment)
{
	auto& slot = m_impl->active();
	if (const auto plan = m_impl->active_uploads)
	{
		if (plan->next >= plan->slices.size() || plan->requirements[plan->next].size != bytes.size() ||
			plan->requirements[plan->next].alignment != alignment)
			throw std::logic_error("Metal4 upload differs from its admitted draw sequence");
		const auto slice = plan->slices[plan->next];
		use_resource(slice.buffer);
		std::memcpy(static_cast<std::byte*>(slice.buffer.contents) + slice.offset, bytes.data(), bytes.size());
		++plan->next;
		return slice;
	}
	const auto limit = m_impl->device.maxBufferLength;
	if (bytes.empty() || bytes.size() > limit || !std::has_single_bit(alignment) || alignment > limit)
	{
		throw std::invalid_argument("Metal 4 upload size or alignment is invalid");
	}
	while (slot.upload_page_index < slot.upload_pages.size())
	{
		auto& page = slot.upload_pages[slot.upload_page_index];
		const auto padding = (alignment - (page.used & (alignment - 1))) & (alignment - 1);
		if (padding <= page.buffer.length - page.used)
		{
			const auto offset = page.used + padding;
			if (bytes.size() <= page.buffer.length - offset)
			{
				use_resource(page.buffer);
				std::memcpy(static_cast<std::byte*>(page.buffer.contents) + offset, bytes.data(), bytes.size());
				page.used = offset + bytes.size();
				return {page.buffer, offset, bytes.size()};
			}
		}
		++slot.upload_page_index;
	}
	const auto size = std::min<std::size_t>(limit, std::max<std::size_t>(1024 * 1024, bytes.size()));
	auto buffer = allocate_buffer(size, MTLStorageModeShared);
	use_resource(buffer);
	std::memcpy(buffer.contents, bytes.data(), bytes.size());
	slot.upload_pages.push_back({buffer, bytes.size()});
	return {buffer, 0, bytes.size()};
}

upload_plan context::prepare_uploads(std::span<const upload_requirement> requirements)
{
	if (m_impl->stopped || m_impl->active_uploads || requirements.empty() || requirements.size() > 256)
		throw std::invalid_argument("Metal4 upload admission requires an idle admission scope and1..256 uploads");
	if (const auto admitted = m_impl->admitted_uploads.lock(); admitted && !admitted->consumed)
		throw std::logic_error("Metal4 already has an unconsumed upload reservation");
	const auto limit = device().maxBufferLength;
	for (const auto& request : requirements)
		if (!request.size || request.size > limit || !std::has_single_bit(request.alignment) || request.alignment > limit)
			throw std::invalid_argument("Metal4 admitted upload size or alignment is invalid");
	auto plan = std::make_shared<upload_plan_state>();
	plan->identity = m_impl->upload_identity;
	plan->requirements.assign(requirements.begin(), requirements.end());
	plan->slices.reserve(requirements.size());
	plan->regions.reserve(requirements.size());
	std::vector<command_slot::upload_page> pages;
	std::size_t index = 0;
	if (const auto slot = m_impl->recording)
	{
		plan->dependency = recording_submission();
		pages = slot->upload_pages;
		index = slot->upload_page_index;
	}
	else
	{
		const auto next = m_impl->slots[m_impl->next_slot];
		if (!next->completion || submission(next->completion).ready())
		{
			pages = next->upload_pages;
			for (auto& page : pages) page.used = 0;
		}
	}
	// The local page copy owns native buffers while allocation may trim the
	// context's caches. No iterator/reference into a slot survives allocation.
	for (const auto& request : requirements)
	{
		while (index < pages.size())
		{
			auto& page = pages[index];
			const auto padding = (request.alignment - (page.used & (request.alignment - 1))) & (request.alignment - 1);
			if (padding <= page.buffer.length - page.used && request.size <= page.buffer.length - page.used - padding) break;
			// Selected slices already own this buffer. A skipped, unused cache
			// page must not be pinned by the planning copy during pressure trim.
			page.buffer = nil;
			++index;
		}
		if (index == pages.size())
		{
			const auto size = std::min<std::size_t>(limit, std::max<std::size_t>(1024 * 1024, request.size));
			const auto buffer = allocate_buffer(size, MTLStorageModeShared);
			pages.push_back({buffer, 0});
		}
		auto& page = pages[index];
		const auto padding = (request.alignment - (page.used & (request.alignment - 1))) & (request.alignment - 1);
		const auto offset = page.used + padding;
		plan->slices.push_back({page.buffer, offset, request.size});
		page.used = offset + request.size;
		const auto found = std::find_if(plan->regions.begin(), plan->regions.end(), [&](const auto& region) { return region.buffer == page.buffer; });
		if (found == plan->regions.end()) plan->regions.push_back({page.buffer, offset, page.used});
		else found->end = page.used;
	}
	m_impl->admitted_uploads = plan;
	return upload_plan(std::move(plan));
}

void context::with_uploads(const upload_plan& reservation, const std::function<void()>& encode)
{
	auto& slot = m_impl->active();
	const auto plan = reservation.m_state;
	if (!encode || !plan || plan->identity != m_impl->upload_identity || plan->consumed || m_impl->active_uploads)
		throw std::logic_error("Metal4 upload reservation is foreign, consumed or nested");
	if (plan->dependency)
	{
		if (plan->dependency->ready()) plan->dependency->wait();
		else if (!slot.observation || !submission(slot.observation).same_batch(*plan->dependency))
			throw std::logic_error("Metal4 upload reservation still belongs to an unfinished foreign batch");
	}
	for (const auto& region : plan->regions)
	{
		const auto found = std::find_if(slot.upload_pages.begin(), slot.upload_pages.end(), [&](const auto& page) { return page.buffer == region.buffer; });
		if (found != slot.upload_pages.end() && found->used > region.begin)
			throw std::logic_error("Metal4 upload arena changed after draw admission");
		for (const auto& other : m_impl->slots)
			if (other.get() != &slot && std::any_of(other->upload_pages.begin(), other->upload_pages.end(), [&](const auto& page) { return page.buffer == region.buffer; }) &&
				other->completion && !submission(other->completion).ready())
				throw std::logic_error("Metal4 admitted upload storage is still owned by another GPU slot");
	}
	slot.upload_pages.reserve(slot.upload_pages.size() + plan->regions.size());
	// Transfer completed arena backing rather than caching it in two slots:
	// resetting the former slot must never overwrite a new pending draw.
	for (const auto& other : m_impl->slots)
		if (other.get() != &slot)
		{
			std::erase_if(other->upload_pages, [&](const auto& page)
				{ return std::any_of(plan->regions.begin(), plan->regions.end(), [&](const auto& region) { return page.buffer == region.buffer; }); });
			other->upload_page_index = 0;
		}
	for (const auto& region : plan->regions)
	{
		const auto found = std::find_if(slot.upload_pages.begin(), slot.upload_pages.end(), [&](const auto& page) { return page.buffer == region.buffer; });
		if (found == slot.upload_pages.end()) slot.upload_pages.push_back({region.buffer, region.end});
		else found->used = region.end;
	}
	slot.upload_page_index = 0;
	plan->consumed = true;
	m_impl->active_uploads = plan;
	try
	{
		encode();
		if (plan->next != plan->slices.size()) throw std::logic_error("Metal4 draw did not consume its admitted uploads");
	}
	catch (...) { m_impl->active_uploads.reset(); throw; }
	m_impl->active_uploads.reset();
}

submission context::recording_submission()
{
	auto& slot = m_impl->active();
	if (!slot.observation) slot.observation = std::make_shared<completion_state>();
	return submission(slot.observation);
}

submission context::submit()
{
	if (m_impl->active_uploads) throw std::logic_error("Cannot submit inside an admitted Metal4 draw");
	@autoreleasepool
	{
		auto slot = m_impl->recording;
		m_impl->active();
		if (slot->drawable && !slot->checks.empty())
			throw std::logic_error("GPU validation must complete before drawable presentation");
		auto state = slot->observation ? slot->observation : std::make_shared<completion_state>();
		MTL4CommitOptions* options = [MTL4CommitOptions new];
		if (!options) fail("Metal4 commit options allocation failed");
		// Capturing the slot keeps every GPU object alive through completion,
		// even when the context is destroyed or a CPU wait times out.
		[options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback)
		{
			std::lock_guard lock(state->mutex);
			state->error = feedback.error;
			if (!state->error)
				for (const auto& check : slot->checks)
				{
					std::uint32_t status = 0;
					std::memcpy(&status,static_cast<const std::byte*>(check.buffer.contents)+check.offset,sizeof(status));
					if (status)
					{
						state->error = [NSError errorWithDomain:@"RPCS3Metal4Validation" code:status
							userInfo:@{NSLocalizedDescriptionKey:check.message}];
						break;
					}
				}
			// Feedback runs after GPU execution. Remove residency and indirect
			// object references before returning allocation leases to their pools.
			// This also releases completed slots that won't be recorded again.
			[slot->residency removeAllAllocations];
			[slot->residency commit];
			[slot->retained removeAllObjects];
			slot->checks.clear();
			slot->owners.clear();
			slot->waits.clear();
			if (slot->retire_upload_pages)
			{
				slot->upload_pages.clear();
				slot->argument_tables.clear();
			}
			state->gpu_completed = true;
			state->completed = state->commit_finished;
			if (state->completed)
			{
				state->progress = completion_state::phase::finished;
				state->changed.notify_all();
			}
			(void)slot;
		}];
		end_encoding();
		[slot->residency commit];
		[slot->commands useResidencySet:slot->residency];
		[slot->commands endCommandBuffer];
		slot->completion = state;
		slot->observation.reset();
		{
			std::lock_guard lock(state->mutex);
			state->progress = completion_state::phase::queued;
		}
		const auto deadline = std::chrono::steady_clock::now() + m_impl->event_wait_timeout;
		const auto stopping = m_impl->stop_submissions;
		// One serial submit worker preserves command order without blocking the
		// render thread. Metal4 native queue waits on shared events were observed
		// to let later work execute before its requested counter, even with one
		// distinct bridge event per wait. Gate these counters using the native
		// shared-event CPU wait before the actual Metal4 commit instead.
		dispatch_async(m_impl->submit_queue, ^
		{
			@autoreleasepool
			{
				bool expired = false;
				bool cancelled = false;
				// Do not enqueue any native queue waits before cancellation is
				// closed. Otherwise a cancelled batch could poison later commits.
				for (const auto& wait : slot->waits)
					if ([wait.event conformsToProtocol:@protocol(MTLSharedEvent)])
					{
						const auto shared = static_cast<id<MTLSharedEvent>>(wait.event);
						bool signaled = false;
						while (!signaled)
						{
							{
								std::lock_guard lock(state->mutex);
								cancelled = state->cancel_requested || stopping->load();
							}
							if (cancelled) break;
							if (shared.signaledValue >= wait.value) break;
							if (std::chrono::steady_clock::now() >= deadline) { expired = true; break; }
							@autoreleasepool { signaled = [shared waitUntilSignaledValue:wait.value timeoutMS:10]; }
						}
						if (cancelled || expired) break;
					}
				{
					std::lock_guard lock(state->mutex);
					if (expired || state->cancel_requested || stopping->load())
					{
						// Never committed: no GPU can reference this slot. Cleanup
						// completes before publishing the token or reusing storage.
						[slot->residency removeAllAllocations];
						[slot->residency commit];
						[slot->retained removeAllObjects];
						slot->checks.clear();
						slot->owners.clear();
						slot->waits.clear();
						slot->drawable = nil;
						if (slot->retire_upload_pages)
						{
							slot->upload_pages.clear();
							slot->argument_tables.clear();
						}
						state->error = [NSError errorWithDomain:@"RPCS3Metal4Cancellation" code:expired ? 2 : 1
							userInfo:@{NSLocalizedDescriptionKey:expired ? @"Shared-event deadline expired before Metal4 commit" : @"Queued batch cancelled before Metal4 commit"}];
						state->cancelled = state->gpu_completed = state->commit_finished = state->completed = true;
						state->progress = completion_state::phase::finished;
						state->changed.notify_all();
						return;
					}
					state->progress = completion_state::phase::committing;
				}
				for (const auto& wait : slot->waits)
					if (![wait.event conformsToProtocol:@protocol(MTLSharedEvent)])
						[slot->queue waitForEvent:wait.event value:wait.value];
				if (slot->drawable) [slot->queue waitForDrawable:slot->drawable];
				id<MTL4CommandBuffer> buffers[] = {slot->commands};
				[slot->queue commit:buffers count:1 options:options];
				if (slot->drawable)
				{
					[slot->queue signalDrawable:slot->drawable];
					[slot->drawable present];
				}
				// A fast GPU may deliver feedback before commit/presentation returns.
				// Publish completion only after both sides are finished, so begin()
				// cannot reset storage while the submit worker still accesses it.
				std::lock_guard lock(state->mutex);
				state->commit_finished = true;
				if (state->gpu_completed)
				{
					state->completed = true;
					state->progress = completion_state::phase::finished;
					state->changed.notify_all();
				}
			}
		});
		m_impl->recording.reset();
		m_impl->next_slot = (m_impl->next_slot + 1) % m_impl->slots.size();
		return submission(std::move(state));
	}
}

void context::cancel()
{
	if (m_impl->active_uploads) throw std::logic_error("Cannot cancel inside an admitted Metal4 draw");
	auto& slot = m_impl->active();
	end_encoding();
	[slot.commands endCommandBuffer];
	[slot.allocator reset];
	[slot.residency removeAllAllocations];
	[slot.residency commit];
	[slot.retained removeAllObjects];
	slot.checks.clear();
	slot.owners.clear();
	slot.waits.clear();
	slot.drawable = nil;
	if (const auto state = slot.observation)
	{
		std::lock_guard lock(state->mutex);
		state->cancelled = state->completed = true;
		state->progress = completion_state::phase::finished;
		state->error = [NSError errorWithDomain:@"RPCS3Metal4Cancellation" code:1 userInfo:@{NSLocalizedDescriptionKey:@"Recording was cancelled before GPU submission"}];
		state->changed.notify_all();
		slot.observation.reset();
	}
	m_impl->recording.reset();
}

void context::wait_idle()
{
	if (m_impl->recording)
	{
		throw std::logic_error("Submit or cancel before waiting for Metal 4 idle");
	}
	for (const auto& slot : m_impl->slots)
	{
		if (slot->completion)
		{
			submission(slot->completion).wait();
		}
	}
}

void context::recover_completed_errors()
{
	if(m_impl->recording)throw std::logic_error("Cannot recover Metal4 slots during recording");
	for(const auto& slot:m_impl->slots)
		if(const auto state=slot->completion)
		{
			std::lock_guard lock(state->mutex);
			if(state->completed&&state->error)slot->completion.reset();
		}
}

void context::shutdown()
{
	if (m_impl->stopped) return;
	m_impl->stopped = true;
	m_impl->stop_submissions->store(true);
	if (m_impl->recording) cancel();
	for (const auto& slot : m_impl->slots)
		if (slot->completion) (void)submission(slot->completion).cancel_pending();
	// No native GPU wait here. Each CPU gate polls cancellation at most 10ms
	// apart; jobs past the commit point remain owned by their feedback blocks.
	dispatch_sync(m_impl->submit_queue, ^{});
}

}
