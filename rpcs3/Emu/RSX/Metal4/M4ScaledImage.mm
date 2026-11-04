#include "M4ScaledImage.h"
#include "M4ScaledImageKernels.h"

namespace metal4
{
namespace
{
struct layout_parameters
{
	std::uint64_t row;
	std::uint32_t width, height, bytes, morton, log_x, log_y, origin_x, origin_y;
	std::uint32_t tiled, tile_base, tile_size, tile_pitch, tile_bank, tile_offset, tile_factor, tile_columns;
};
struct image_parameters
{
	layout_parameters source, destination;
	std::uint32_t input_w, input_h, scaled_w, scaled_h, clip_x, clip_y, flip_x, flip_y, linear, padding;
};
struct captured_image_part
{
	guest_surface_layout coverage;
	std::size_t offset;
	std::uint64_t revision;
};
struct captured_image
{
	std::vector<std::byte> bytes;
	std::vector<captured_image_part> parts;
};
captured_image capture_image(guest_surface_memory& memory, const guest_surface_layout& image,
							 const std::optional<banked_tile_layout>& tile, std::span<const guest_storage_segment> segments)
{
	captured_image result{std::vector<std::byte>(image.range.size), {}};
	const auto read = [&](guest_surface_layout coverage, std::size_t offset)
	{
		coverage = memory.watch_surface(coverage);
		const auto snapshot = memory.read_surface(coverage);
		if (snapshot.bytes.size() != coverage.range.size)
			throw std::runtime_error("Native scaled-image memory provider returned a short snapshot");
		std::copy(snapshot.bytes.begin(), snapshot.bytes.end(), result.bytes.begin() + offset);
		result.parts.push_back({std::move(coverage), offset, snapshot.revision});
	};
	if (segments.empty())
		read(scaled_image_coverage(image, tile), 0);
	else
		for (const auto& segment : segments)
			read(guest_byte_coverage({segment.address, segment.size}), segment.offset);
	return result;
}
std::vector<guest_surface_range> storage_ranges(const guest_surface_layout& image, std::span<const guest_storage_segment> segments)
{
	std::vector<guest_surface_range> result;
	if (segments.empty())
		result.push_back(image.range);
	else
		for (const auto& segment : segments)
			result.push_back({segment.address, segment.size});
	return result;
}
static_assert(sizeof(layout_parameters) == 72 && sizeof(image_parameters) == 184);
layout_parameters parameters(const guest_surface_layout& layout, const std::optional<banked_tile_layout>& tile)
{
	layout_parameters result{layout.row_pitch ? layout.row_pitch : layout.width * layout.pixel_bytes, static_cast<std::uint32_t>(layout.width), static_cast<std::uint32_t>(layout.height),
							 static_cast<std::uint32_t>(layout.pixel_bytes), bool(layout.morton), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	if (layout.morton)
	{
		const auto swizzle = plan_swizzle(*layout.morton, layout.pixel_bytes);
		if (layout.morton->storage[2] != 1 || layout.morton->origin[2])
			throw std::invalid_argument("Scaled-image Morton storage must be two-dimensional");
		result.log_x = swizzle.log2[0];
		result.log_y = swizzle.log2[1];
		result.origin_x = static_cast<std::uint32_t>(layout.morton->origin[0]);
		result.origin_y = static_cast<std::uint32_t>(layout.morton->origin[1]);
	}
	if (tile)
	{
		result.tiled = 1;
		result.tile_base = tile->address;
		result.tile_size = tile->size;
		result.tile_pitch = tile->pitch;
		result.tile_bank = tile->bank;
		result.tile_offset = tile->offset;
		result.tile_factor = banked_tile_factor(tile->pitch);
		result.tile_columns = tile->pitch / 256;
		result.row = tile->pitch;
	}
	return result;
}
} // namespace
scaled_image_compute::scaled_image_compute(context& ctx, std::size_t limit) : m_device(ctx.device()), m_limit(limit)
{
	@autoreleasepool
	{
		if (!limit)
			throw std::invalid_argument("Native scaled-image requires a snapshot budget");
		helper_catalog compile(ctx);
		const auto library = compile.library(scaled_image_kernels, @"RPCS3 native scaled-image transfer");

		m_pipeline = compile.compute(library, @"rpcs3_scaled_image");
		m_gather = compile.compute(library, @"rpcs3_scaled_gather");
	}
}
void scaled_image_compute::encode(context& ctx, const scaled_image_plan& plan, id<MTLBuffer> source, id<MTLBuffer> destination) const
{
	validate_scaled_image(plan, m_limit);
	if (ctx.device() != m_device || !source || !destination || source.device != m_device || destination.device != m_device ||
		source == destination || source.length < plan.source.range.size || destination.length < plan.destination.range.size)
		throw std::invalid_argument("Native scaled-image requires distinct adequately sized buffers on its own device");
	const image_parameters params{parameters(plan.source, plan.source_tile), parameters(plan.destination, plan.destination_tile), plan.input_width, plan.input_height,
								  plan.scaled_width, plan.scaled_height, plan.clip_x, plan.clip_y, plan.flip_x, plan.flip_y, plan.interpolate, 0};
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&params, 1}), 256);
	bindings resources;
	resources.set_buffer(0, source);
	resources.set_buffer(1, destination);
	resources.set_buffer(2, uniforms.buffer, uniforms.offset);
	const auto table = resources.snapshot(ctx);
	ctx.keep_alive(m_pipeline);
	const auto encoder = ctx.compute();
	ctx.barrier(MTLStageDispatch | MTLStageBlit, MTLStageDispatch);
	[encoder setComputePipelineState:m_pipeline];
	[encoder setArgumentTable:table];
	const NSUInteger x = std::min<NSUInteger>(16, m_pipeline.maxTotalThreadsPerThreadgroup);
	[encoder dispatchThreads:MTLSizeMake(plan.destination.width, plan.destination.height, 1)
		threadsPerThreadgroup:MTLSizeMake(x, std::min<NSUInteger>(16, m_pipeline.maxTotalThreadsPerThreadgroup / x), 1)];
}
void scaled_image_compute::gather(context& ctx,id<MTLBuffer> source,id<MTLBuffer> destination,
	std::span<const display_storage_copy> copies,std::size_t input) const
{
	struct copy {std::uint64_t target,source,ordinal,size;};static_assert(sizeof(copy)==32);
	if(ctx.device()!=m_device||!source||!destination||source.device!=m_device||destination.device!=m_device||source==destination||copies.size()>(1u<<20))
		throw std::invalid_argument("Native scaled-image gather requires bounded buffers on its own device");
	std::vector<copy> selected;std::uint64_t total=0;
	for(const auto& part:copies)if(part.input==input)
	{
		if(!part.size||part.source>source.length||part.size>source.length-part.source||part.target>destination.length||part.size>destination.length-part.target)
			throw std::out_of_range("Native scaled-image gather exceeds buffer extent");
		selected.push_back({part.target,part.source,total,part.size});total+=part.size;
	}
	if(selected.empty()||total>m_limit||total>std::numeric_limits<std::uint32_t>::max())throw std::invalid_argument("Native scaled-image gather lacks bounded disjoint bytes");
	const auto records=ctx.upload(std::as_bytes(std::span{selected}),256);
	const std::array<std::uint64_t,2> shape{total,selected.size()};const auto sizes=ctx.upload(std::as_bytes(std::span{shape}),256);
	bindings resources;resources.set_buffer(0,source);resources.set_buffer(1,destination);resources.set_buffer(2,records.buffer,records.offset);resources.set_buffer(3,sizes.buffer,sizes.offset);
	const auto table=resources.snapshot(ctx);ctx.keep_alive(m_gather);const auto encoder=ctx.compute();ctx.barrier(MTLStageDispatch|MTLStageBlit,MTLStageDispatch);
	[encoder setComputePipelineState:m_gather];[encoder setArgumentTable:table];
	[encoder dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256,m_gather.maxTotalThreadsPerThreadgroup),1,1)];
}
struct scaled_image_transfer::pending
{
	std::optional<submission> work;
	std::vector<captured_image_part> destination;
	std::vector<guest_surface_range> ranges;
	std::size_t output_bytes;
	id<MTLBuffer> output;
	std::function<void()> validate_metadata;
	std::shared_ptr<bool> native_owner={};
	cached_surface_ref native_target={};
};
scaled_image_transfer::scaled_image_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures,
											 std::shared_ptr<guest_surface_memory> memory, cpu_capture_dispatch capture, std::size_t limit)
	: m_context(ctx), m_surfaces(surfaces), m_textures(textures), m_memory(std::move(memory)), m_capture(std::move(capture)), m_compute(ctx, limit), m_colors(ctx), m_samples(ctx), m_limit(limit)
{
	if (!m_memory || !m_capture)
		throw std::invalid_argument("Native scaled-image requires coherent memory and a CPU capture epoch");
	m_surfaces.validate_texture_adapter(ctx, *m_memory);
	m_textures.validate_surface_adapter(ctx, m_surfaces);
}
scaled_image_transfer::~scaled_image_transfer()
{
	if(m_pending&&m_pending->native_owner)*m_pending->native_owner=false;
}
void scaled_image_transfer::encode_native(const scaled_image_plan& plan,std::span<const clear_rectangle> regions,
	context& ctx,std::span<const id<MTLTexture>> images,std::span<const cached_surface_description> inputs,std::span<const display_storage_copy> copies,
	std::span<const std::byte> input_bytes,std::span<const display_storage_copy> input_patches)
{
	std::vector<id<MTLTexture>> linear(images.begin(),images.end());
	for(unsigned side=0;side<images.size();++side)if(images[side].sampleCount>1)
	{
		const auto size=sample_expanded_size(images[side].width,images[side].height,static_cast<unsigned>(images[side].sampleCount));
		auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:images[side].pixelFormat width:size[0] height:size[1] mipmapped:NO];
		descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
		linear[side]=make_texture(ctx,descriptor);
		// Import writes the whole expanded sample raster. Preserve native pixels
		// outside a cropped destination too,without ever reading guest storage.
		m_samples.expand(ctx,images[side],linear[side]);
	}
	auto canonical=plan;canonical.source=plan_guest_surface(0,plan.source.width,plan.source.height,plan.source.pixel_bytes);
	canonical.destination=plan_guest_surface(0,plan.destination.width,plan.destination.height,plan.destination.pixel_bytes);
	canonical.source_tile.reset();canonical.destination_tile.reset();canonical.source_segments.clear();canonical.destination_segments.clear();
	const auto source=make_buffer(ctx,canonical.source.range.size,buffer_storage::private_gpu),target=make_buffer(ctx,canonical.destination.range.size,buffer_storage::private_gpu);
	texture_region input,output;input.origin=MTLOriginMake(regions[0].x,regions[0].y,0);input.size=MTLSizeMake(regions[0].width,regions[0].height,1);
	output.origin=MTLOriginMake(regions[1].x,regions[1].y,0);output.size=MTLSizeMake(regions[1].width,regions[1].height,1);
	const auto sf=plan.source_format==scaled_image_format::rgb565?surface_color::rgb565:surface_color::argb8;
	const auto df=plan.destination_format==scaled_image_format::rgb565?surface_color::rgb565:surface_color::argb8;
	if(inputs.empty())m_colors.download(ctx,sf,linear[0],input,source,{});
	else for(std::size_t i=0;i<inputs.size();++i)
	{
		const auto& d=inputs[i];
		const auto geometry=plan_guest_surface(d.tile?d.tile->address:0,linear[i].width,linear[i].height,surface_guest_bytes(d.allocation.format),d.row_pitch,d.morton);
		const auto tight=geometry.width*geometry.pixel_bytes;
		const auto raw=make_buffer(ctx,d.tile?d.tile->size:d.morton?geometry.range.size:tight*geometry.height,buffer_storage::private_gpu);
		texture_region whole;whole.size=MTLSizeMake(linear[i].width,linear[i].height,1);
		if(const auto* color=std::get_if<surface_color>(&d.allocation.format))
		{
			if(d.tile){auto g=geometry;g.range={d.tile->address,d.tile->size};m_banked->download_color(ctx,*color,linear[i],whole,raw,g,*d.tile);}
			else if(d.morton)m_swizzle->download_color(ctx,*color,linear[i],whole,raw,0,*d.morton);
			else m_colors.download(ctx,*color,linear[i],whole,raw,{});
		}
		else
		{
			const auto depth=std::get<surface_depth>(d.allocation.format);
			if(d.tile){auto g=geometry;g.range={d.tile->address,d.tile->size};m_banked->download_depth(ctx,depth,linear[i],whole,raw,g,*d.tile);}
			else if(d.morton)m_swizzle->download_depth(ctx,depth,linear[i],whole,raw,0,*d.morton);
			else m_depth->download(ctx,depth,linear[i],whole,raw,{});
		}
		auto packed_copies=std::vector<display_storage_copy>(copies.begin(),copies.end());
		if(!d.tile&&!d.morton)for(auto& copy:packed_copies)if(copy.input==i)copy.source=(copy.source/geometry.row_pitch)*tight+copy.source%geometry.row_pitch;
		m_compute.gather(ctx,raw,source,packed_copies,i);
	}
	if(!input_bytes.empty())m_compute.gather(ctx,make_upload(ctx,input_bytes),source,input_patches,0);
	m_compute.encode(ctx,canonical,source,target);m_colors.upload(ctx,df,target,{},linear.back(),output);
	if(images.back().sampleCount>1)m_samples.import(ctx,linear.back(),images.back());
}
void scaled_image_transfer::execute(scaled_image_plan plan, std::chrono::milliseconds timeout, const std::function<void(context&)>& dependencies,
									std::function<void()> validate_metadata)
{
	@autoreleasepool
	{
		if (timeout.count() < 0)
			throw std::invalid_argument("Native scaled-image wait timeout must be nonnegative");
		validate_scaled_image(plan, m_limit);
		synchronize(timeout);
		if (validate_metadata)
			checked_cpu_capture(m_capture, validate_metadata);
		if(const auto endpoints=m_surfaces.prepare_scaled_image(plan,timeout))
		{
			for(const auto& d:endpoints->descriptions)
			{
				if(std::holds_alternative<surface_depth>(d.allocation.format)&&!m_depth)m_depth=std::make_unique<surface_depth_transfer>(m_context);
				const auto size=d.allocation.samples==1?std::array{d.allocation.width,d.allocation.height}:sample_expanded_size(d.allocation.width,d.allocation.height,d.allocation.samples);
				const auto geometry=plan_guest_surface(0,size[0],size[1],surface_guest_bytes(d.allocation.format),d.row_pitch,d.morton);
				if((d.tile?d.tile->size:geometry.range.size)>m_limit)throw std::length_error("Native scaled-image assembly source exceeds snapshot budget");
				if(d.tile&&!m_banked)m_banked=std::make_unique<surface_banked_transfer>(m_context);
				if(!d.tile&&d.morton&&!m_swizzle)m_swizzle=std::make_unique<surface_swizzle_transfer>(m_context);
			}
			auto owner=std::make_shared<bool>(true);
			m_pending=std::make_unique<pending>(pending{std::nullopt,{},storage_ranges(plan.destination,plan.destination_segments),0,nil,validate_metadata,owner,endpoints->surfaces[1]});
			try
			{
				m_surfaces.draw_scaled_image(*endpoints,[&](context& commands,auto images)
				{
					m_pending->work=commands.recording_submission();
					if(dependencies)dependencies(commands);
					encode_native(plan,endpoints->regions,commands,images,endpoints->descriptions,endpoints->assembly.copies,endpoints->input_bytes,endpoints->input_patches);
				},[owner,validate=std::move(validate_metadata)]
				{
					if(!*owner)throw std::runtime_error("Abandoned native scaled-image output cannot publish");
					if(validate)validate();
				},timeout);
			}
			catch(...)
			{
				if(!m_pending->work||(!m_surfaces.status(m_pending->native_target).pending&&m_pending->work->ready()))m_pending.reset();
				throw;
			}
			synchronize(timeout);return;
		}
		const auto source_ranges = storage_ranges(plan.source, plan.source_segments);
		auto destination_ranges = storage_ranges(plan.destination, plan.destination_segments);
		for (const auto range : source_ranges)
			m_surfaces.prepare_cpu_read(range);
		for (const auto range : destination_ranges)
			m_surfaces.prepare_cpu_read(range);
		captured_image input, output;
		checked_cpu_capture(m_capture, [&]
							{
		if (validate_metadata) validate_metadata();
		input=capture_image(*m_memory,plan.source,plan.source_tile,plan.source_segments);
		output=capture_image(*m_memory,plan.destination,plan.destination_tile,plan.destination_segments); });
		// Allocate transaction and owners before submit;wait failures retain them.
		const auto source = make_upload(m_context, input.bytes), target = make_upload(m_context, output.bytes);
		m_context.begin();
		std::unique_ptr<pending> transaction;
		try
		{
			transaction = std::make_unique<pending>(pending{m_context.recording_submission(), std::move(output.parts), std::move(destination_ranges), output.bytes.size(), target, std::move(validate_metadata)});
			if (dependencies)
				dependencies(m_context);
			m_compute.encode(m_context, plan, source, target);
			transaction->work = m_context.submit();
		}
		catch (...)
		{
			m_context.cancel();
			throw;
		}
		m_pending = std::move(transaction);
		synchronize(timeout);
	}
}
void scaled_image_transfer::synchronize(std::chrono::milliseconds timeout)
{
	@autoreleasepool
	{
		if (timeout.count() < 0)
			throw std::invalid_argument("Native scaled-image wait timeout must be nonnegative");
		if (!m_pending)
			return;
		if(m_pending->native_owner)
		{
			try{m_surfaces.synchronize(timeout);m_pending->work->wait(timeout);}
			catch(...){if(!m_surfaces.status(m_pending->native_target).pending&&m_pending->work->ready())m_pending.reset();throw;}
			auto completed=std::move(m_pending);
			for(const auto range:completed->ranges)m_textures.invalidate(range);
			++m_native_completed;return;
		}
		try
		{
			m_pending->work->wait(timeout);
		}
		catch (...)
		{
			if (m_pending->work->ready())
				m_pending.reset();
			throw;
		}
		auto completed = std::move(m_pending);
		if (completed->validate_metadata)
			checked_cpu_capture(m_capture, completed->validate_metadata);
		const auto bytes = map_buffer(completed->output, 0, completed->output_bytes);
		std::vector<guest_surface_write> writes;
		for (const auto& part : completed->destination)
			writes.push_back({part.coverage, part.revision, bytes.subspan(part.offset, part.coverage.range.size)});
		m_surfaces.cpu_write(completed->ranges, [&]
		{
			if (completed->validate_metadata) completed->validate_metadata();
			if (!m_memory->compare_write_many(writes)) throw texture_revision_conflict();
		}, m_capture);
		for (const auto range : completed->ranges)
			m_textures.invalidate(range);
		++m_vm_completed;
	}
}
} // namespace metal4
