#include "M4FSR.h"
#include "M4FSRGenerated.h"
#include "M4ExclusiveGuestMemory.h"

#include <thread>

namespace metal4
{
struct fsr_upscaler::impl
{
	context* owner;
	std::weak_ptr<const std::byte> identity;
	std::thread::id thread = std::this_thread::get_id();
	std::shared_ptr<heap_budget> budget;
	helper_catalog compiler;
	id<MTLLibrary> library;
	id<MTLComputePipelineState> easu, rcas;
	struct work
	{
		frame_extent size;
		id<MTLTexture> intermediate = nil, output = nil;
		std::size_t bytes = 0;
	};
	std::array<work,2> eyes{};
	fsr_statistics stats;
	impl(context& ctx, std::size_t limit, std::shared_ptr<heap_budget> supplied)
		: owner(&ctx), identity(ctx.queue_identity()), budget(supplied ? std::move(supplied) : ctx.buffer_ledger() ? ctx.buffer_ledger() : std::make_shared<heap_budget>(ctx,limit)), compiler(ctx)
	{
		if (!limit || limit > (512u << 20) || budget->device() != ctx.device()) throw std::invalid_argument("Invalid native FSR working ledger/limit");
		stats.working_limit = limit;
		library = compiler.library(fsr_kernels,@"RPCS3 native Metal4 FSR1");
		easu = compiler.compute(library,@"m4_fsr_easu"); rcas = compiler.compute(library,@"m4_fsr_rcas");
	}
	void check() const
	{
		if (thread != std::this_thread::get_id() || guest_memory_epoch::active()) throw std::logic_error("Native FSR requires its render owner outside VM exclusion");
	}
	bool clear(unsigned eye)
	{
		auto& value = eyes.at(eye);
		if (!value.bytes) return false;
		stats.working_bytes -= value.bytes; value = {}; return true;
	}
};
fsr_upscaler::fsr_upscaler(context& ctx,std::size_t limit,std::shared_ptr<heap_budget> budget)
	: m_impl(std::make_unique<impl>(ctx,limit,std::move(budget))) {}
fsr_upscaler::~fsr_upscaler() = default;
bool fsr_upscaler::clear() { m_impl->check(); const bool left = m_impl->clear(0); return m_impl->clear(1) || left; }
bool fsr_upscaler::clear_eye(unsigned eye) { m_impl->check(); return m_impl->clear(eye); }
fsr_statistics fsr_upscaler::statistics() const { m_impl->check(); return m_impl->stats; }
present_eye fsr_upscaler::encode(context& ctx,const present_eye& input,frame_extent output,float sharpness,unsigned eye)
{
	m_impl->check();
	const auto identity = ctx.queue_identity();
	if (&ctx != m_impl->owner || identity.owner_before(m_impl->identity) || m_impl->identity.owner_before(identity) || eye >= 2 || !input.texture || input.texture.device != ctx.device() || input.texture.textureType != MTLTextureType2D ||
		input.texture.sampleCount != 1 || input.texture.framebufferOnly || !(input.texture.usage & MTLTextureUsageShaderRead))
		throw std::invalid_argument("Invalid native FSR source/context/eye");
	switch (input.texture.pixelFormat)
	{
	case MTLPixelFormatRGBA8Unorm: case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB:
	case MTLPixelFormatRGBA16Float: case MTLPixelFormatRGBA32Float: break;
	default: throw std::invalid_argument("Native FSR needs a readable floating/color image");
	}
	const auto dimensions = validate_texture_region(ctx,input.texture,input.region);
	const auto maximum = max_2d_texture_dimension(ctx.device());
	if (!output.width || !output.height || output.width > maximum || output.height > maximum || ctx.current_render_encoder())
		throw std::invalid_argument("Invalid native FSR output extent oractive render facade");
	const auto parameters = plan_fsr({static_cast<std::uint32_t>(dimensions[0]),static_cast<std::uint32_t>(dimensions[1])},
		{static_cast<std::uint32_t>(input.region.origin.x),static_cast<std::uint32_t>(input.region.origin.y),
		 static_cast<std::uint32_t>(input.region.size.width),static_cast<std::uint32_t>(input.region.size.height)},output,
		static_cast<std::uint32_t>(input.region.level),sharpness);
	(void)ctx.recording_submission();
	if (!needs_fsr({parameters.extent[0],parameters.extent[1]},output)) { ++m_impl->stats.bypassed; return input; }
	const auto bytes = fsr_working_bytes(output,m_impl->stats.working_limit);
	if (auto& cached = m_impl->eyes[eye]; cached.size != output)
	{
		(void)m_impl->clear(eye); // Do not pin anobsolete size against its own replacement admission.
		if (bytes > m_impl->stats.working_limit - m_impl->stats.working_bytes) throw std::length_error("Native FSR eye pair exceeds aggregate working limit");
		impl::work candidate; candidate.size = output; candidate.bytes = bytes;
		auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:output.width height:output.height mipmapped:NO];
		d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
		candidate.intermediate = m_impl->budget->texture(d); candidate.output = m_impl->budget->texture(d);
		cached = std::move(candidate); m_impl->stats.working_bytes += bytes; ++m_impl->stats.created;
	}
	else ++m_impl->stats.reused;
	// Retain a local owner before upload/pressure callbacks can clear the cache.
	const auto work = m_impl->eyes[eye];
	const auto uniforms = ctx.upload(std::as_bytes(std::span{&parameters,1}));
	const auto dispatch = [&](id<MTLComputePipelineState> pipeline,id<MTLTexture> source,id<MTLTexture> target)
	{
		bindings resources; resources.set_buffer(0,uniforms.buffer,uniforms.offset); resources.set_texture(0,source); resources.set_texture(1,target);
		const std::array accesses{
			context::accessed_resource{source},
			context::accessed_resource{target, 0, 0, true},
			context::accessed_resource{uniforms.buffer, uniforms.offset, uniforms.size}};
		ctx.checked_dispatch(pipeline, resources, accesses, MTLSizeMake(output.width,output.height,1), MTLSizeMake(8,8,1));
	};
	dispatch(m_impl->easu,input.texture,work.intermediate); ++m_impl->stats.easu_dispatches;
	dispatch(m_impl->rcas,work.intermediate,work.output); ++m_impl->stats.rcas_dispatches;
	ctx.end_encoding();
	return {work.output,{0,0,MTLOriginMake(0,0,0),MTLSizeMake(output.width,output.height,1)}};
}
} // namespace metal4
