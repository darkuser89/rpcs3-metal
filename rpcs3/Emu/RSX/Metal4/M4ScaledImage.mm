#include "M4ScaledImage.h"
#include "M4TransferPublication.h"
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
	const std::array accesses{
		context::accessed_resource{source, 0, plan.source.range.size},
		context::accessed_resource{destination, 0, plan.destination.range.size, true},
		context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size}};
	const NSUInteger x = std::min<NSUInteger>(16, m_pipeline.maxTotalThreadsPerThreadgroup);
	ctx.checked_dispatch(m_pipeline, resources, accesses, MTLSizeMake(plan.destination.width, plan.destination.height, 1),
		MTLSizeMake(x, std::min<NSUInteger>(16, m_pipeline.maxTotalThreadsPerThreadgroup / x), 1));
}
void scaled_image_compute::gather(context& ctx,id<MTLBuffer> source,id<MTLBuffer> destination,
	std::span<const display_storage_copy> copies,std::size_t input) const
{
	struct copy {std::uint64_t target,source,ordinal,size;};static_assert(sizeof(copy)==32);
	if(ctx.device()!=m_device||!source||!destination||source.device!=m_device||destination.device!=m_device||source==destination||copies.size()>(1u<<20))
		throw std::invalid_argument("Native scaled-image gather requires bounded buffers on its own device");
	std::vector<copy> selected;std::uint64_t total=0;
	std::size_t source_begin = source.length, source_end = 0;
	std::size_t target_begin = destination.length, target_end = 0;
	for(const auto& part:copies)if(part.input==input)
	{
		if(!part.size||part.source>source.length||part.size>source.length-part.source||part.target>destination.length||part.size>destination.length-part.target)
			throw std::out_of_range("Native scaled-image gather exceeds buffer extent");
		selected.push_back({part.target,part.source,total,part.size});total+=part.size;
		source_begin = std::min(source_begin, part.source);
		source_end = std::max(source_end, part.source + part.size);
		target_begin = std::min(target_begin, part.target);
		target_end = std::max(target_end, part.target + part.size);
	}
	if(selected.empty()||total>m_limit||total>std::numeric_limits<std::uint32_t>::max())throw std::invalid_argument("Native scaled-image gather lacks bounded disjoint bytes");
	const auto records=ctx.upload(std::as_bytes(std::span{selected}),256);
	const std::array<std::uint64_t,2> shape{total,selected.size()};const auto sizes=ctx.upload(std::as_bytes(std::span{shape}),256);
	bindings resources;resources.set_buffer(0,source);resources.set_buffer(1,destination);resources.set_buffer(2,records.buffer,records.offset);resources.set_buffer(3,sizes.buffer,sizes.offset);
	const std::array accesses{
		context::accessed_resource{source, source_begin, source_end - source_begin},
		context::accessed_resource{destination, target_begin, target_end - target_begin, true},
		context::accessed_resource{records.buffer, records.offset, records.size},
		context::accessed_resource{sizes.buffer, sizes.offset, sizes.size}};
	ctx.checked_dispatch(m_gather, resources, accesses, total);
}
struct native_scale_publication
{
	bool alive = true, published = false;
};
struct scaled_image_transfer::pending
{
	std::optional<submission> work;
	std::vector<captured_image_part> destination;
	std::vector<guest_surface_range> ranges;
	std::size_t output_bytes;
	id<MTLBuffer> output;
	std::function<void()> validate_metadata;
	std::shared_ptr<native_scale_publication> native_owner={};
	cached_surface_ref native_target={};
};
scaled_image_transfer::scaled_image_transfer(context& ctx, surface_cache& surfaces, guest_texture_cache& textures,
											 std::shared_ptr<guest_surface_memory> memory, cpu_capture_dispatch capture, std::size_t limit, std::size_t writer_limit)
	: m_context(ctx), m_surfaces(surfaces), m_textures(textures), m_memory(std::move(memory)), m_capture(std::move(capture)), m_compute(ctx, limit), m_colors(ctx), m_samples(ctx), m_limit(limit), m_writer_limit(writer_limit)
{
	if (!m_memory || !m_capture || !m_writer_limit || m_writer_limit > scaled_image_max_writers)
		throw std::invalid_argument("Native scaled-image requires coherent memory and a CPU capture epoch");
	m_surfaces.validate_texture_adapter(ctx, *m_memory);
	m_textures.validate_surface_adapter(ctx, m_surfaces);
}
scaled_image_transfer::~scaled_image_transfer()
{
	if(m_pending&&m_pending->native_owner)m_pending->native_owner->alive=false;
}
void scaled_image_transfer::encode_native(const scaled_image_plan& plan,std::span<const clear_rectangle> regions,
	context& ctx,std::span<const id<MTLTexture>> images,std::span<const cached_surface_description> inputs,std::span<const std::vector<scaled_image_input_region>> input_regions,
	std::span<const std::byte> input_bytes,std::span<const display_storage_copy> input_patches,
	std::span<const scaled_image_output_region> output_regions,
	std::span<const scaled_image_output_byte_region> output_byte_regions,
	std::span<const surface_format> output_formats,std::size_t output_count,
	std::span<const std::vector<surface_cache::scaled_image_input_view>> input_views)
{
	auto canonical=plan;canonical.source=plan_guest_surface(0,plan.source.width,plan.source.height,plan.source.pixel_bytes);
	canonical.destination=plan_guest_surface(0,plan.destination.width,plan.destination.height,plan.destination.pixel_bytes);
	canonical.source_tile.reset();canonical.destination_tile.reset();canonical.source_segments.clear();canonical.destination_segments.clear();
	if (!output_count || output_count > m_writer_limit || images.size() < output_count)
		throw std::logic_error("Scaled output lost its distinct native writers");
	const bool ram_only = images.size() == output_count;
	if (ram_only)
	{
		std::size_t packed = 0;
		for (const auto& patch : input_patches)
		{
			if (patch.target != packed || patch.source != packed || patch.size > canonical.source.range.size - packed)
				throw std::logic_error("RAM-only scaled input is not densely packed");
			packed += patch.size;
		}
		if (!inputs.empty() || packed != canonical.source.range.size || packed != input_bytes.size())
			throw std::logic_error("RAM-only scaled input lacks a complete snapshot");
	}
	const auto source = ram_only ? make_upload(ctx,input_bytes) : make_buffer(ctx,canonical.source.range.size,buffer_storage::private_gpu);
	// Pitched codec windows validate complete rows, including the final
	// row padding after a horizontal subrectangle. No padding is sampled.
	const auto target_bytes = checked_texture_sum(canonical.destination.range.size,
		output_regions.empty() ? 0 : (canonical.destination.width - 1) * canonical.destination.pixel_bytes);
	const auto target = make_buffer(ctx,target_bytes,buffer_storage::private_gpu);
	texture_region input,output;input.origin=MTLOriginMake(regions[0].x,regions[0].y,0);input.size=MTLSizeMake(regions[0].width,regions[0].height,1);
	output.origin=MTLOriginMake(regions[1].x,regions[1].y,0);output.size=MTLSizeMake(regions[1].width,regions[1].height,1);
	const auto sf=plan.source_format==scaled_image_format::rgb565?surface_color::rgb565:surface_color::argb8;
	const auto df=plan.destination_format==scaled_image_format::rgb565?surface_color::rgb565:surface_color::argb8;
	const auto selected_input=[&](id<MTLTexture> image,texture_region& region)
	{
		if(image.sampleCount==1)return image;
		auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:region.size.width height:region.size.height mipmapped:NO];
		descriptor.storageMode=MTLStorageModePrivate;descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead|MTLTextureUsagePixelFormatView;
		const auto cropped=make_texture(ctx,descriptor);
		m_samples.expand_region(ctx,image,cropped,{region.origin.x,region.origin.y});
		region.origin=MTLOriginMake(0,0,0);return cropped;
	};
	if(!ram_only&&inputs.empty())
	{
		auto image=images[0];
		if (!input_views.empty() && !input_views[0].empty())
		{
			if (input_views[0].size()!=1)throw std::logic_error("Direct scaled input lost its one native window");
			const auto& view=input_views[0][0];image=view.image;
			input.origin=MTLOriginMake(view.pixels.x,view.pixels.y,0);
			input.size=MTLSizeMake(view.pixels.width,view.pixels.height,1);
		}
		image=selected_input(image,input);m_colors.download(ctx,sf,image,input,source,{});
	}
	else for(std::size_t i=0;i<inputs.size();++i)
	{
		const auto& d=inputs[i];
		const auto windowed=!input_views.empty()&&!input_views[i].empty();
		if(windowed&&input_views[i].size()!=input_regions[i].size())throw std::logic_error("Packed scaled input lost its native windows");
		for(std::size_t j=0;j<input_regions[i].size();++j)
		{
			const auto& part=input_regions[i][j];
			const auto raw=make_buffer(ctx,part.bytes,buffer_storage::private_gpu);
			const auto pixels=windowed?input_views[i][j].pixels:part.pixels;
			texture_region region;region.origin=MTLOriginMake(pixels.x,pixels.y,0);
			region.size=MTLSizeMake(pixels.width,pixels.height,1);
			const auto image=selected_input(windowed?input_views[i][j].image:images[i],region);
			if(const auto* color=std::get_if<surface_color>(&d.allocation.format))m_colors.download(ctx,*color,image,region,raw,{});
			else m_depth->download(ctx,std::get<surface_depth>(d.allocation.format),image,region,raw,{});
			m_compute.gather(ctx,raw,source,part.copies,0);
		}
	}
	if(!ram_only&&!input_bytes.empty())m_compute.gather(ctx,make_upload(ctx,input_bytes),source,input_patches,0);
	m_compute.encode(ctx,canonical,source,target);
	if (!output_byte_regions.empty())
	{
		if (output_formats.size() != output_count)
			throw std::logic_error("Scaled byte windows lost their native output formats");
		for (const auto& part : output_byte_regions)
		{
			if (part.output >= output_count || !part.bytes || part.bytes > m_limit)
				throw std::logic_error("Scaled byte window exceeds its native writer or scratch budget");
			const auto image = images[images.size() - output_count + part.output];
			const auto format = output_formats[part.output];
			const auto origin = std::array<std::size_t, 2>{part.native.x, part.native.y};
			texture_region region;
			region.origin = MTLOriginMake(origin[0], origin[1], 0);
			region.size = MTLSizeMake(part.native.width, part.native.height, 1);
			const auto selected = selected_input(image, region);
			const auto raw = make_buffer(ctx, part.bytes, buffer_storage::private_gpu);
			const auto* color = std::get_if<surface_color>(&format);
			if (part.preserve_background)
			{
				if (color) m_colors.download(ctx, *color, selected, region, raw, {});
				else m_depth->download(ctx, std::get<surface_depth>(format), selected, region, raw, {});
			}
			m_compute.gather(ctx, target, raw, part.copies, 0);
			if (color) m_colors.upload(ctx, *color, raw, {}, selected, region);
			else
			{
				const auto mask = make_buffer(ctx, part.bytes, buffer_storage::private_gpu);
				fill_buffer(ctx, mask, 0, part.bytes, 0);
				for (const auto& copy : part.copies) fill_buffer(ctx, mask, copy.target, copy.size, 1);
				m_depth->patch(ctx, std::get<surface_depth>(format), raw, mask, {}, selected, region);
			}
			if (selected != image) m_samples.import_region(ctx, selected, image, origin);
		}
		return;
	}
	for (std::size_t i = 0; i < (output_regions.empty() ? 1 : output_regions.size()); ++i)
	{
		const auto index = output_regions.empty() ? 0 : output_regions[i].output;
		if (index >= output_count) throw std::logic_error("Scaled window selects another native writer");
		const auto image = images[images.size() - output_count + index];
		auto selected = output;
		surface_byte_region bytes{};
		if (!output_regions.empty())
		{
			const auto& part = output_regions[i].packed;
			const auto& native = output_regions[i].native;
			selected.origin = MTLOriginMake(native.x, native.y, 0);
			selected.size = MTLSizeMake(native.width, native.height, 1);
			bytes = {(part.y * canonical.destination.width + part.x) * canonical.destination.pixel_bytes,
				canonical.destination.width * canonical.destination.pixel_bytes};
		}
		if (image.sampleCount > 1)
		{
			auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:image.pixelFormat width:selected.size.width height:selected.size.height mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			const auto cropped = make_texture(ctx, descriptor);
			const auto origin = std::array<std::size_t, 2>{selected.origin.x, selected.origin.y};
			selected.origin = MTLOriginMake(0, 0, 0);
			m_colors.upload(ctx, df, target, bytes, cropped, selected);
			m_samples.import_region(ctx, cropped, image, origin);
		}
		else m_colors.upload(ctx, df, target, bytes, image, selected);
	}
}
void scaled_image_transfer::execute(scaled_image_plan plan, std::chrono::milliseconds timeout, const std::function<void(context&)>& dependencies,
									std::function<void()> validate_metadata)
{
	execute_impl(std::move(plan), timeout, dependencies, std::move(validate_metadata), false);
}
void scaled_image_transfer::enqueue(scaled_image_plan plan, std::chrono::milliseconds timeout, const std::function<void(context&)>& dependencies,
	std::function<void()> validate_metadata)
{
	execute_impl(std::move(plan), timeout, dependencies, std::move(validate_metadata), true);
}
void scaled_image_transfer::execute_impl(scaled_image_plan plan, std::chrono::milliseconds timeout, const std::function<void(context&)>& dependencies,
	std::function<void()> validate_metadata, bool defer_native_completion)
{
	@autoreleasepool
	{
		if (timeout.count() < 0)
			throw std::invalid_argument("Native scaled-image wait timeout must be nonnegative");
		validate_scaled_image(plan, m_limit);
		synchronize(timeout);
		if (validate_metadata)
			checked_cpu_capture(m_capture, validate_metadata);
		if(const auto endpoints=m_surfaces.prepare_scaled_image(plan,timeout,m_writer_limit,m_limit))
		{
			std::vector<std::vector<scaled_image_input_region>> input_regions;
			if (!m_depth && std::ranges::any_of(endpoints->output_formats, [](const auto& format)
				{ return std::holds_alternative<surface_depth>(format); }))
				m_depth = std::make_unique<surface_depth_transfer>(m_context);
			for(std::size_t i=0;i<endpoints->descriptions.size();++i)
			{
				const auto& d=endpoints->descriptions[i];
				if(std::holds_alternative<surface_depth>(d.allocation.format)&&!m_depth)m_depth=std::make_unique<surface_depth_transfer>(m_context);
				const auto size=d.allocation.samples==1?std::array{d.allocation.width,d.allocation.height}:sample_expanded_size(d.allocation.width,d.allocation.height,d.allocation.samples);
				const auto geometry=plan_guest_surface(0,size[0],size[1],surface_guest_bytes(d.allocation.format),d.row_pitch,d.morton);
				input_regions.push_back(plan_scaled_image_input_regions({geometry,{},false,0,d.tile},endpoints->assembly.copies,i,m_limit));
			}
			auto owner=std::make_shared<native_scale_publication>();
			std::vector<std::vector<clear_rectangle>> input_windows;
			if(endpoints->surfaces[0])input_windows.push_back({endpoints->regions[0]});
			else for(const auto& regions:input_regions)
			{
				auto& windows=input_windows.emplace_back();
				for(const auto& part:regions)windows.push_back(part.pixels);
			}
			m_pending=std::make_unique<pending>(pending{std::nullopt,{},storage_ranges(plan.destination,plan.destination_segments),0,nil,validate_metadata,owner,endpoints->surfaces[1]});
			try
			{
				m_surfaces.draw_scaled_image(*endpoints,input_windows,[&](context& commands,auto images,auto input_views)
				{
					encode_native(plan,endpoints->regions,commands,images,endpoints->descriptions,input_regions,endpoints->input_bytes,endpoints->input_patches,endpoints->output_regions,endpoints->output_byte_regions,endpoints->output_formats,endpoints->outputs.empty()?1:endpoints->outputs.size(),input_views);
				},[owner,validate=std::move(validate_metadata)]
				{
					if(!owner->alive)throw std::runtime_error("Abandoned native scaled-image output cannot publish");
					if(validate)validate();
				},timeout,[&](context& commands)
				{
					m_pending->work=commands.recording_submission();
					if(dependencies)dependencies(commands);
				},defer_native_completion,[owner]{owner->published=true;});
			}
			catch(...)
			{
				if(!m_pending->work||(!m_surfaces.status(m_pending->native_target).pending&&m_pending->work->ready()))m_pending.reset();
				throw;
			}
			if (!defer_native_completion) synchronize(timeout);
			return;
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
void scaled_image_transfer::retire_native()
{
	auto completed=std::move(m_pending);
	// An ordinary surface/texture observer can retire this transaction before
	// its owner is polled. GPU success alone does not prove metadata publication.
	if (!completed->native_owner->published)
		throw std::runtime_error("Native scaled-image output was rejected before owner retirement");
	for(const auto range:completed->ranges)m_textures.invalidate(range);
	++m_native_completed;
}
bool scaled_image_transfer::poll()
{
	if (!m_pending) return true;
	if (!m_pending->native_owner || !m_pending->work->ready()) return false;
	try
	{
		if (!m_surfaces.poll_submitted()) return false;
		m_pending->work->wait(std::chrono::milliseconds::zero());
	}
	catch (...)
	{
		if (!m_surfaces.status(m_pending->native_target).pending && m_pending->work->ready()) m_pending.reset();
		throw;
	}
	retire_native();
	return true;
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
			retire_native();return;
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
		publish_completed_transfer(m_pending,m_capture,[&](auto& completed,const cpu_capture_dispatch& capture)
		{
		if (completed.validate_metadata)
			checked_cpu_capture(capture, completed.validate_metadata);
		const auto bytes = map_buffer(completed.output, 0, completed.output_bytes);
		std::vector<guest_surface_write> writes;
		for (const auto& part : completed.destination)
			writes.push_back({part.coverage, part.revision, bytes.subspan(part.offset, part.coverage.range.size)});
		m_surfaces.cpu_write(completed.ranges, [&]
		{
			if (completed.validate_metadata) completed.validate_metadata();
			if (!m_memory->compare_write_many(writes)) throw texture_revision_conflict();
		}, capture);
		for (const auto range : completed.ranges)
			m_textures.invalidate(range);
		++m_vm_completed;
		});
	}
}
} // namespace metal4
