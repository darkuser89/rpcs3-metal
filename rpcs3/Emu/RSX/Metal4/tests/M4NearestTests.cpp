#include "M4NearestLayout.h"
#include "M4TransferKernels.h"

#include <fstream>
#include <iostream>

using namespace metal4;
void require(bool value, const char* message)
{
	if (!value)
		throw std::runtime_error(message);
}
void source_window_plans()
{
    std::size_t cases=0;
    const auto oracle=[](std::uint32_t d,std::uint32_t source,std::uint32_t copy,bool flip)
    {
        const auto denominator=__uint128_t(copy)*2;auto numerator=(__uint128_t(d)*2+1)*source;
        if(flip)numerator=__uint128_t(source)*denominator-numerator;
        return std::uint32_t(numerator/denominator);
    };
    const auto check=[&](std::array<std::uint32_t,3> source,std::array<std::uint32_t,3> destination,std::array<bool,3> flip,nearest_projection p)
    {
        const auto result=plan_nearest_input_window(source,destination,flip,p);auto cropped=result.projection;
        for(unsigned axis=0;axis<3;++axis)
        {
            auto low=UINT32_MAX,high=0u;
            for(unsigned i=0;i<destination[axis];++i){const auto v=oracle(i+p.offset[axis],source[axis],p.extent[axis],flip[axis]);low=std::min(low,v);high=std::max(high,v);}
            require(result.origin[axis]==low&&result.extent[axis]==high-low+1,"Nearest source window differs from rational oracle");
            cropped.source_origin[axis]+=result.origin[axis];
        }
        const auto again=plan_nearest_input_window(result.extent,destination,flip,cropped);
        require(again.origin==std::array<std::uint32_t,3>{}&&again.extent==result.extent,"Nearest cropped source changed rational phase");++cases;
    };
    for(unsigned source=1;source<=33;++source)for(unsigned copy=1;copy<=33;++copy)for(unsigned at=0;at<copy;++at)for(unsigned n=1;n<=copy-at;++n)for(bool flip:{false,true})
        check({source,1,1},{n,1,1},{flip,false,false},{{copy,1,1},{at,0,0}});
    for(unsigned flags=0;flags<8;++flags)check({7,11,5},{3,2,2},{bool(flags&1),bool(flags&2),bool(flags&4)},{{13,17,9},{5,7,3}});
    for(unsigned source:{1u,2u,3u,0x7fffffffu,0xfffffffeu,0xffffffffu})for(unsigned at:{0u,0x7fffffffu,0xfffffffcu})for(bool flip:{false,true})
        check({source,1,1},{3,1,1},{flip,false,false},{{0xffffffffu,1,1},{at,0,0}});
    for(const auto p:{nearest_projection{{3,1,1},{2,0,0}},nearest_projection{{4,1,1},{0,0,0},{8,1,1},{4,0,0}}})
    {
        bool failed=false;try{(void)plan_nearest_input_window({4,1,1},{2,1,1},{},p);}catch(const std::invalid_argument&){failed=true;}require(failed,"Nearest planner accepted omitted or out-of-domain samples");
    }
    std::cout<<cases<<" source-window rational plans, cropped-source phase, XYZ flips and uint32 endpoint boundaries passed\n";
}
int main(int argc, char** argv)
{
	try
	{
		source_window_plans();
		std::size_t cases = 0;
		for (const unsigned bytes : {1u, 2u, 4u, 8u, 16u})
			for (std::uint32_t width = 1; width <= 7; ++width)
				for (std::uint32_t height = 1; height <= 5; ++height)
					for (std::uint32_t depth = 1; depth <= 3; ++depth)
					{
						nearest_image a{{width, height, depth}, (width + 2) * bytes, (height + 3) * (width + 2) * bytes};
						nearest_image b{{height, depth, width}, (height + 4) * bytes, (depth + 1) * (height + 4) * bytes};
						const auto plan = plan_nearest(a, b, bytes);
						std::size_t count = 0, end = 0;
						for (std::size_t z = 0; z < b.size[2]; ++z)
							for (std::size_t y = 0; y < b.size[1]; ++y)
								for (std::size_t x = 0; x < b.size[0]; ++x)
								{
									end = z * b.image_pitch + y * b.row_pitch + (x + 1) * bytes;
									require(end <= plan.destination.allocation_size, "Nearest plan omitted an active texel");
									++count;
								}
						require(count == plan.texels && plan.destination.allocation_size - end == b.row_pitch - b.size[0] * bytes,
								"Nearest plan includes unused final image padding or wrong texel count");
						++cases;
					}
		const nearest_image one{{1, 1, 1}};
		const auto rejected = [&](nearest_image a, nearest_image b, unsigned bytes)
		{
			bool failed = false;
			try
			{
				(void)plan_nearest(a, b, bytes);
			}
			catch (const std::exception&)
			{
				failed = true;
			}
			require(failed, "Nearest plan accepted zero/partial/overflowing storage");
		};
		for (const unsigned bytes : {0u, 3u, 6u, 32u})
			rejected(one, one, bytes);
		rejected({{0, 1, 1}}, one, 1);
		rejected(one, {{1, 0, 1}}, 1);
		rejected(one, {{1, 1, 0}}, 1);
		rejected({{7, 2, 1}, 6, 0}, one, 1);
		rejected(one, {{2, 2, 2}, 8, 8}, 4);
		rejected(one, {{0xffffffffu, 0xffffffffu, 0xffffffffu}}, 16);
		rejected({{1, 2, 2}, std::numeric_limits<std::size_t>::max(), 0}, one, 1);
		const auto large = plan_nearest(one, {{0xffffffffu, 1, 1}}, 16);
		require(large.texels == 0xffffffffu && large.destination.allocation_size == std::uint64_t(0xffffffffu) * 16,
				"Nearest uint32 extent was truncated to 32-bit byte arithmetic");
		if (argc == 2)
			std::ofstream(argv[1]) << transfer_shader_source();
		std::cout << cases << " nearest pitched image plans, sparse texel coverage and uint32/size_t overflow boundaries passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
