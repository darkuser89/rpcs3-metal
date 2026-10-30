#include "M4RenderPipeline.h"
#include "M4FragmentMSL.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace
{
void require(bool condition)
{
	if (!condition)
		throw std::runtime_error("Metal 4 render descriptor invariant failed");
}

template <typename Function>
bool accepts(Function function)
{
	try
	{
		function();
		return true;
	}
	catch (const std::invalid_argument&)
	{
		return false;
	}
}

MTL4FunctionDescriptor* function()
{
	// A descriptor is plain CPU state. No library/device/compiler is created.
	MTL4LibraryFunctionDescriptor* descriptor = [MTL4LibraryFunctionDescriptor new];
	descriptor.name = @"test_function_without_a_device";
	return descriptor;
}

void enumeration_tests()
{
	using namespace metal4;
	constexpr std::array<std::uint32_t, 15> factors{0, 1, 0x300, 0x301, 0x302, 0x303, 0x304, 0x305,
													0x306, 0x307, 0x308, 0x8001, 0x8002, 0x8003, 0x8004};
	constexpr std::array<std::uint32_t, 8> stencil{0, 0x150a, 0x1e00, 0x1e01, 0x1e02, 0x1e03, 0x8507, 0x8508};
	constexpr std::array<std::uint32_t, 8> equations{0x8006, 0x8007, 0x8008, 0x800a, 0x800b, 0xf005, 0xf006, 0xf007};
	for (std::uint32_t value = 0; value < 65536; ++value)
	{
		require(accepts([&]
						{
							(void)decode_compare(value);
						}) == (value < 8 || (value >= 0x200 && value <= 0x207)));
		require(accepts([&]
						{
							(void)decode_blend_factor(value);
						}) == (std::find(factors.begin(), factors.end(), value) != factors.end()));
		require(accepts([&]
						{
							(void)decode_stencil(value);
						}) == (std::find(stencil.begin(), stencil.end(), value) != stencil.end()));
		require(accepts([&]
						{
							(void)decode_blend_operation(value);
						}) == (std::find(equations.begin(), equations.end(), value) != equations.end()));
	}
	for (const auto invalid : {0xffffffffu, 0x10000u, 0x10201u})
	{
		require(!accepts([&]
						 {
							 (void)decode_compare(invalid);
						 }));
		require(!accepts([&]
						 {
							 (void)decode_stencil(invalid);
						 }));
		require(!accepts([&]
						 {
							 (void)decode_blend_factor(invalid);
						 }));
		require(!accepts([&]
						 {
							 (void)decode_blend_operation(invalid);
						 }));
	}
	std::cout << "262144 raw enum acceptance checks passed\n";
}

void depth_stencil_tests()
{
	using namespace metal4;
	constexpr std::array<MTLCompareFunction, 8> comparisons{MTLCompareFunctionNever, MTLCompareFunctionLess,
															MTLCompareFunctionEqual, MTLCompareFunctionLessEqual, MTLCompareFunctionGreater,
															MTLCompareFunctionNotEqual, MTLCompareFunctionGreaterEqual, MTLCompareFunctionAlways};
	constexpr std::array<compare_operation, 8> reversed{compare_operation::never, compare_operation::greater,
														compare_operation::equal, compare_operation::greater_equal, compare_operation::less,
														compare_operation::not_equal, compare_operation::less_equal, compare_operation::always};
	guest_render_state guest;
	guest.depth_write = true;
	guest.depth_compare = 0xffffffffu;
	auto descriptor = make_depth_stencil_descriptor(guest);
	require(descriptor.depthCompareFunction == MTLCompareFunctionAlways && !descriptor.depthWriteEnabled);
	require(descriptor.frontFaceStencil.writeMask == 0 && descriptor.backFaceStencil.writeMask == 0);
	guest.depth_test = true;
	guest.stencil_test = true;
	guest.two_sided_stencil = true;
	for (std::uint32_t index = 0; index < 8; ++index)
	{
		guest.depth_compare = index;
		guest.front.compare = 0x200 + index;
		guest.back.compare = 0x207 - index;
		descriptor = make_depth_stencil_descriptor(guest);
		require(descriptor.depthCompareFunction == comparisons[index] && descriptor.depthWriteEnabled);
		require(descriptor.frontFaceStencil.stencilCompareFunction == comparisons[index]);
		require(descriptor.backFaceStencil.stencilCompareFunction == comparisons[7 - index]);
		require(decode_compare(index, true) == reversed[index]);
	}
	struct fixture
	{
		std::uint32_t guest;
		MTLStencilOperation native;
	};
	const std::array<fixture, 8> operations{{{0x1e00, MTLStencilOperationKeep}, {0, MTLStencilOperationZero}, {0x1e01, MTLStencilOperationReplace}, {0x1e02, MTLStencilOperationIncrementClamp}, {0x1e03, MTLStencilOperationDecrementClamp}, {0x150a, MTLStencilOperationInvert}, {0x8507, MTLStencilOperationIncrementWrap}, {0x8508, MTLStencilOperationDecrementWrap}}};
	for (const auto& op : operations)
	{
		guest.front.fail = guest.front.depth_fail = guest.front.pass = op.guest;
		guest.back = guest.front;
		descriptor = make_depth_stencil_descriptor(guest);
		for (auto face : {descriptor.frontFaceStencil, descriptor.backFaceStencil})
			require(face.stencilFailureOperation == op.native && face.depthFailureOperation == op.native && face.depthStencilPassOperation == op.native);
	}
	for (std::uint32_t mask = 0; mask < 256; ++mask)
	{
		guest.front.read_mask = 0xff000000 | mask;
		guest.front.write_mask = 0x12340000 | (255 - mask);
		guest.front.reference = 0xabcd0000 | mask;
		guest.back.read_mask = 255 - mask;
		guest.back.write_mask = mask;
		guest.back.reference = 255 - mask;
		descriptor = make_depth_stencil_descriptor(guest);
		const auto state = decode_render_state(guest);
		require(descriptor.frontFaceStencil.readMask == mask && descriptor.frontFaceStencil.writeMask == 255 - mask);
		require(descriptor.backFaceStencil.readMask == 255 - mask && descriptor.backFaceStencil.writeMask == mask);
		require(state.front.reference == mask && state.back.reference == 255 - mask);
	}
	guest.two_sided_stencil = false;
	guest.back.compare = guest.back.fail = 0xffffffffu;
	const auto one_sided = decode_render_state(guest);
	require(one_sided.front == one_sided.back);
	guest.stencil_test = false;
	guest.front.compare = 0xffffffffu;
	descriptor = make_depth_stencil_descriptor(guest);
	require(descriptor.frontFaceStencil.stencilCompareFunction == MTLCompareFunctionAlways && descriptor.frontFaceStencil.writeMask == 0);
}

void color_tests()
{
	using namespace metal4;
	render_pipeline_configuration config;
	config.colors = {MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA16Float, MTLPixelFormatR32Float, MTLPixelFormatRGBA32Float};
	config.shader_half_storage_mask = 2;
	const auto vertex = function(), fragment = function();
	struct factor_fixture
	{
		std::uint32_t guest;
		MTLBlendFactor native;
	};
	const std::array<factor_fixture, 15> factors{{{0, MTLBlendFactorZero}, {1, MTLBlendFactorOne}, {0x300, MTLBlendFactorSourceColor}, {0x301, MTLBlendFactorOneMinusSourceColor}, {0x302, MTLBlendFactorSourceAlpha}, {0x303, MTLBlendFactorOneMinusSourceAlpha}, {0x304, MTLBlendFactorDestinationAlpha}, {0x305, MTLBlendFactorOneMinusDestinationAlpha}, {0x306, MTLBlendFactorDestinationColor}, {0x307, MTLBlendFactorOneMinusDestinationColor}, {0x308, MTLBlendFactorSourceAlphaSaturated}, {0x8001, MTLBlendFactorBlendColor}, {0x8002, MTLBlendFactorOneMinusBlendColor}, {0x8003, MTLBlendFactorBlendAlpha}, {0x8004, MTLBlendFactorOneMinusBlendAlpha}}};
	for (const auto& factor : factors)
	{
		for (auto& color : config.state.colors)
		{
			color.enabled = true;
			color.source_rgb = color.destination_rgb = color.source_alpha = color.destination_alpha = factor.guest;
		}
		const auto descriptor = make_render_descriptor(config, vertex, fragment);
		for (NSUInteger i = 0; i < 4; ++i)
		{
			const auto color = descriptor.colorAttachments[i];
			require(color.pixelFormat == config.colors[i] && color.blendingState == MTL4BlendStateEnabled);
			require(color.sourceRGBBlendFactor == factor.native && color.destinationRGBBlendFactor == factor.native &&
					color.sourceAlphaBlendFactor == factor.native && color.destinationAlphaBlendFactor == factor.native);
		}
	}
	struct operation_fixture
	{
		std::uint32_t guest;
		MTLBlendOperation native;
	};
	const std::array<operation_fixture, 5> operations{{{0x8006, MTLBlendOperationAdd}, {0x800a, MTLBlendOperationSubtract}, {0x800b, MTLBlendOperationReverseSubtract}, {0x8007, MTLBlendOperationMin}, {0x8008, MTLBlendOperationMax}}};
	for (const auto& rgb : operations)
		for (const auto& alpha : operations)
		{
			config.state.colors[0].operation_rgb = rgb.guest;
			config.state.colors[0].operation_alpha = alpha.guest;
			const auto descriptor = make_render_descriptor(config, vertex, fragment);
			require(descriptor.colorAttachments[0].rgbBlendOperation == rgb.native && descriptor.colorAttachments[0].alphaBlendOperation == alpha.native);
		}
	for (unsigned enabled = 0; enabled < 16; ++enabled)
		for (unsigned mask = 0; mask < 16; ++mask)
		{
			for (unsigned i = 0; i < 4; ++i)
			{
				config.state.colors[i].enabled = (enabled & (1u << i)) != 0;
				config.state.colors[i].write_mask = mask;
			}
			const auto descriptor = make_render_descriptor(config, vertex, fragment);
			const auto expected_mask = ((mask & 1) ? MTLColorWriteMaskRed : 0) | ((mask & 2) ? MTLColorWriteMaskGreen : 0) |
									   ((mask & 4) ? MTLColorWriteMaskBlue : 0) | ((mask & 8) ? MTLColorWriteMaskAlpha : 0);
			for (NSUInteger i = 0; i < 4; ++i)
				require(descriptor.colorAttachments[i].writeMask == expected_mask &&
						descriptor.colorAttachments[i].blendingState == ((enabled & (1u << i)) ? MTL4BlendStateEnabled : MTL4BlendStateDisabled));
		}
	// Four independently different RGB/alpha factors catch accidental slot reuse.
	config.state.colors[0] = {true, 0x302, 0x303, 1, 0, 0x8006, 0x800a, 15};
	const auto separate = make_render_descriptor(config, vertex, fragment).colorAttachments[0];
	require(separate.sourceRGBBlendFactor == MTLBlendFactorSourceAlpha && separate.destinationRGBBlendFactor == MTLBlendFactorOneMinusSourceAlpha &&
			separate.sourceAlphaBlendFactor == MTLBlendFactorOne && separate.destinationAlphaBlendFactor == MTLBlendFactorZero);
	for (const auto signed_op : {0xf005u, 0xf006u, 0xf007u})
	{
		config.state.colors[0].operation_rgb = signed_op;
		require(decode_render_state(config.state).requires_shader_rop);
		require(!accepts([&]
						 {
							 (void)make_render_descriptor(config, vertex, fragment);
						 }));
		fragment_program_options options;configure_shader_blending(config,options);
		const auto descriptor = make_render_descriptor(config,vertex,fragment);
		for(unsigned i=0;i<4;++i)require(descriptor.colorAttachments[i].blendingState == MTL4BlendStateDisabled);
		auto mismatch=config;mismatch.state.colors[0].source_rgb=0;
		require(!accepts([&] { (void)make_render_descriptor(mismatch,vertex,fragment); }));
		mismatch=config;mismatch.colors[1]=MTLPixelFormatRGBA32Float;
		require(!accepts([&] { (void)make_render_descriptor(mismatch,vertex,fragment); }));
		config.shader_blend={};
	}
	config.state.logic_enabled = true;
	config.state.colors[0].operation_rgb = config.state.colors[0].source_rgb = 0xffffffffu;
	for (std::uint32_t logic = 0x1500; logic <= 0x150f; ++logic)
	{
		config.state.logic_operation = logic;
		const auto state = decode_render_state(config.state);
		require(state.requires_shader_rop && !state.colors[0].enabled && state.logic_operation == logic);
		require(!accepts([&]
						 {
							 (void)make_render_descriptor(config, vertex, fragment);
						 }));
		fragment_program_options options;
		configure_logic_operations(config, options);
		require(options.logic_rop == config.shader_logic && options.color_count == 4);
		require(options.logic_rop.bits[0] == (std::array<std::uint32_t, 4>{8, 8, 8, 8}));
		for (unsigned i = 1; i < 4; ++i)
			require(options.logic_rop.bits[i] == (std::array<std::uint32_t, 4>{}));
		const auto descriptor = make_render_descriptor(config, vertex, fragment);
		for (unsigned i = 0; i < 4; ++i)
			require(descriptor.colorAttachments[i].blendingState == MTL4BlendStateDisabled);
		auto mismatch = config;
		mismatch.colors[0] = MTLPixelFormatRGBA16Unorm;
		require(!accepts([&]
						 {
							 (void)make_render_descriptor(mismatch, vertex, fragment);
						 }));
		mismatch = config;
		mismatch.state.sample_count = 4;
		require(!accepts([&]
						 {
							 (void)make_render_descriptor(mismatch, vertex, fragment);
						 }));
		// Restore the unverified state for the following operation.
		config.shader_logic = {};
	}
}

void pipeline_boundaries()
{
	using namespace metal4;
	render_pipeline_configuration config;
	const auto vertex = function(), fragment = function();
	for (const auto samples : {1u, 2u, 4u, 8u})
		for (unsigned alpha = 0; alpha < 4; ++alpha)
		{
			config.state.sample_count = samples;
			config.state.alpha_to_coverage = (alpha & 1) != 0;
			config.state.alpha_to_one = (alpha & 2) != 0;
			const auto descriptor = make_render_descriptor(config, vertex, fragment);
			require(descriptor.rasterSampleCount == samples && descriptor.alphaToCoverageState == ((alpha & 1) ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled) &&
					descriptor.alphaToOneState == ((alpha & 2) ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled));
			require(descriptor.colorAttachments[0].pixelFormat == MTLPixelFormatInvalid && descriptor.colorAttachments[0].writeMask == MTLColorWriteMaskNone);
		}
	for (const auto samples : {0u, 3u, 5u, 16u, 0xffffffffu})
	{
		config.state.sample_count = samples;
		require(!accepts([&]
						 {
							 (void)make_render_descriptor(config, vertex, fragment);
						 }));
	}
	config.state.sample_count = 1;
	require(!accepts([&]
					 {
						 (void)make_render_descriptor(config, nil, fragment);
					 }));
	require(!accepts([&]
					 {
						 (void)make_render_descriptor(config, vertex, nil);
					 }));
	config.rasterization = false;
	require(!make_render_descriptor(config, vertex, nil).rasterizationEnabled);
	for (const auto format : {MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float_Stencil8, MTLPixelFormatStencil8,
							  MTLPixelFormatRGBA8Uint, MTLPixelFormatR32Sint, MTLPixelFormatBC1_RGBA, static_cast<MTLPixelFormat>(0xffffffffu)})
	{
		config.colors[0] = format;
		require(!accepts([&]
						 {
							 (void)make_render_descriptor(config, vertex, fragment);
						 }));
	}
	config.colors[0] = MTLPixelFormatBGRA8Unorm;
	for (const auto topology : {MTLPrimitiveTopologyClassPoint, MTLPrimitiveTopologyClassLine, MTLPrimitiveTopologyClassTriangle, MTLPrimitiveTopologyClassUnspecified})
	{
		config.topology = topology;
		require(make_render_descriptor(config, vertex, fragment).inputPrimitiveTopology == topology);
	}
	config.topology = static_cast<MTLPrimitiveTopologyClass>(100);
	require(!accepts([&]
					 {
						 (void)make_render_descriptor(config, vertex, fragment);
					 }));
	config.state.colors[0].write_mask = 16;
	require(!accepts([&]
					 {
						 (void)decode_render_state(config.state);
					 }));
}
void output_rop_contracts()
{
	using namespace metal4;
	const auto vertex=function(),fragment=function();
	{
		render_pipeline_configuration config;config.colors[0]=MTLPixelFormatRGBA16Float;
		require(!accepts([&] { (void)make_render_descriptor(config,vertex,fragment); }));
		fragment_program_options options;configure_output_rops(config,options);
		require(options.half_storage_mask==1 && accepts([&] { (void)make_render_descriptor(config,vertex,fragment); }));
	}
	for(unsigned flags=0;flags<16;++flags)
	{
		render_pipeline_configuration config;config.colors[0]=MTLPixelFormatRGBA8Unorm;
		config.state.output_rop={bool(flags&1),bool(flags&2),bool(flags&4),bool(flags&8)};
		require(accepts([&] { (void)make_render_descriptor(config,vertex,fragment); }) == (flags==0));
		fragment_program_options options;configure_output_rops(config,options);
		require(options.output_rop == config.state.output_rop);
		require(accepts([&] { (void)make_render_descriptor(config,vertex,fragment); }));
		auto mismatch=config;mismatch.state.output_rop.round_to_8bit=!mismatch.state.output_rop.round_to_8bit;
		require(!accepts([&] { (void)make_render_descriptor(mismatch,vertex,fragment); }));
		if(flags&4)
		{
			mismatch=config;mismatch.topology=MTLPrimitiveTopologyClassLine;
			require(!accepts([&] { (void)make_render_descriptor(mismatch,vertex,fragment); }));
		}
		if(flags&2)
		{
			mismatch=config;mismatch.colors[0]=MTLPixelFormatRGBA8Unorm_sRGB;
			require(!accepts([&] { (void)make_render_descriptor(mismatch,vertex,fragment); }));
		}
	}
}
void depth_mirror_contracts()
{
	using namespace metal4;const auto vertex=function(),fragment=function();
	for(auto encoding:{depth_mirror_encoding::unorm16,depth_mirror_encoding::float32})for(bool write:{false,true})
	{
		render_pipeline_configuration config;config.depth_mirror={encoding,write};config.state.depth_test=true;config.state.depth_write=write;
		const auto descriptor=make_render_descriptor(config,vertex,fragment);
		require(descriptor.colorAttachments[4].pixelFormat==MTLPixelFormatR32Float);
		require(descriptor.colorAttachments[4].writeMask==(write?MTLColorWriteMaskRed:MTLColorWriteMaskNone));
		config.state.depth_write=!write;require(!accepts([&]{(void)make_render_descriptor(config,vertex,fragment);}));
	}
	render_pipeline_configuration config;config.depth_mirror={depth_mirror_encoding::none,true};
	require(!accepts([&]{(void)make_render_descriptor(config,vertex,fragment);}));
}
} // namespace

int main()
{
	@autoreleasepool
	{
		try
		{
			enumeration_tests();
			depth_stencil_tests();
			color_tests();
			pipeline_boundaries();
			output_rop_contracts();
			depth_mirror_contracts();
			std::cout << "Native Metal4 render/depth/stencil descriptor CPU checks passed; no device created\n";
		}
		catch (const std::exception& error)
		{
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
}
