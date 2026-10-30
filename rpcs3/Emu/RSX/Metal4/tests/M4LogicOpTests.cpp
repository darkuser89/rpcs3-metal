#include "M4LogicOp.h"

#include <iostream>

using namespace metal4;
int main()
{
	try
	{
		// Independent 2-input Boolean truth tables indexed by (source,dest).
		constexpr std::array<unsigned, 16> truth{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
		std::size_t count = 0;
		for (unsigned op = 0; op < 16; ++op)
			for (unsigned source = 0; source < 256; ++source)
				for (unsigned destination = 0; destination < 256; ++destination)
				{
					unsigned expected = 0;
					for (unsigned bit = 0; bit < 8; ++bit)
						if (truth[op] & (1u << (((source >> bit) & 1) * 2 + ((destination >> bit) & 1))))
							expected |= 1u << bit;
					if ((apply_logic_operation(0x1500 + op, source, destination) & 255) != expected)
						throw std::runtime_error("RSX logic result differs from Boolean truth table");
					++count;
				}
		bool rejected = false;
		try
		{
			(void)apply_logic_operation(0x14ff, 0, 0);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		if (!rejected)
			throw std::runtime_error("Invalid logic op accepted");
		logic_rop_configuration cfg;
		cfg.bits[0][0] = 17;
		rejected = false;
		try
		{
			validate_logic_rop(cfg);
		}
		catch (const std::invalid_argument&)
		{
			rejected = true;
		}
		if (!rejected)
			throw std::runtime_error("Oversized normalized logic channel accepted");
		std::cout << count << " independently enumerated Boolean logic truth-table cases passed\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return 1;
	}
}
