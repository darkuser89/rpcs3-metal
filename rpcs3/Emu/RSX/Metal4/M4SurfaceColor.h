#pragma once

#include "M4SurfaceLayout.h"
#include "M4Texture.h"
#include "M4Pipeline.h"
#include "M4Resources.h"
#include "M4RenderPipeline.h"
#include "M4FragmentMSL.h"

namespace metal4
{

struct surface_color_descriptor
{
	MTLPixelFormat format;
	MTLTextureSwizzleChannels sampling_swizzle;
	surface_color_layout layout;
};

[[nodiscard]] surface_color_descriptor describe_surface_color(surface_color format);
// Sets format, shader export remap and guest channel mask in the same logical
// slot. Call before configure_shader_blending and shader/pipeline compilation.
// Remapped surfaces require verified shader ROPs: blending precedes output
// remapping and reads the guest destination-channel/alpha representation.
void configure_surface_color(render_pipeline_configuration& pipeline, fragment_program_options& fragment,
	std::size_t slot, surface_color format, std::uint32_t guest_write_mask);

class surface_color_transfer final
{
	id<MTLDevice> m_device;
	id<MTLComputePipelineState> m_pipeline;

public:
	explicit surface_color_transfer(context& ctx);
	// Bit-exact reversible conversion between guest BE pixels and native LE
	// texture-transfer bytes. Doesn't normalize floats or replace X bits, so
	// NaN payloads/-zero and masked memory contents survive a round-trip.
	// The byte reversal is its own inverse. Padding is untouched. Caller owns
	// synchronization/VM coherence and must not provide distinct heap aliases.
	void convert(context& ctx, surface_color format, std::size_t width, std::size_t height,
		id<MTLBuffer> source, const surface_byte_region& input,
		id<MTLBuffer> destination, const surface_byte_region& output) const;
	// Region width/height and native mip/slice supplied via texture_region.
	// Texture must match the described native format and have sampleCount=1.
	void upload(context& ctx, surface_color format, id<MTLBuffer> guest, const surface_byte_region& input,
		id<MTLTexture> texture, const texture_region& region) const;
	void download(context& ctx, surface_color format, id<MTLTexture> texture, const texture_region& region,
		id<MTLBuffer> guest, const surface_byte_region& output) const;
};

}
