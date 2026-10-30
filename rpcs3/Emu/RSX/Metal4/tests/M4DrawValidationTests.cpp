#include "M4DrawValidation.h"
#include "M4DepthBounds.h"

#include <iostream>
#include <utility>

namespace
{

void require(bool condition)
{
	if (!condition) throw std::runtime_error("Draw validation invariant failed");
}

template <typename Function>
bool accepts(Function function)
{
	try { function(); return true; }
	catch (const std::invalid_argument&) { return false; }
	catch (const std::out_of_range&) { return false; }
}

}

int main()
{
	using namespace metal4;
	constexpr std::size_t max_id = std::numeric_limits<std::uint32_t>::max();
	constexpr std::size_t max_size = std::numeric_limits<std::size_t>::max();
	validate_draw_ids(max_id, 1);
	validate_draw_ids(1, max_id);
	validate_draw_ids(0, 0);
	validate_depth_bounds(0, 1);
	validate_depth_bounds(0.25f, 0.75f);
	validate_depth_bounds(0.5f, 0.5f);
	validate_depth_bounds(-0.0f, 0.0f);
	for (bool native : {false,true})
	{
		require(plan_depth_bounds(false,0,1,native) == depth_bounds_path::disabled);
		require(plan_depth_bounds(true,0,1,native) == depth_bounds_path::mirror);
		require(plan_depth_bounds(true,-0.0f,1,native) == depth_bounds_path::mirror);
		for (const auto interval : {std::pair{0.f,.5f},std::pair{.5f,1.f},std::pair{.5f,.5f},std::pair{0.f,0.f},std::pair{1.f,1.f},std::pair{std::nextafter(0.f,1.f),1.f},std::pair{0.f,std::nextafter(1.f,0.f)}})
			require(plan_depth_bounds(true,interval.first,interval.second,native) == (native ? depth_bounds_path::native_test : depth_bounds_path::mirror));
		require(!accepts([&] { (void)plan_depth_bounds(true,.75f,.25f,native); }));
	}
	for (const float invalid : {-0.001f, 1.001f, std::numeric_limits<float>::infinity(),
		-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
	{
		require(!accepts([&] { validate_depth_bounds(invalid, 1); }));
		require(!accepts([&] { validate_depth_bounds(0, invalid); }));
		for (bool native : {false,true})
		{
			require(!accepts([&] { (void)plan_depth_bounds(true,invalid,1,native); }));
			require(!accepts([&] { (void)plan_depth_bounds(true,0,invalid,native); }));
		}
	}
	require(!accepts([] { validate_depth_bounds(0.75f, 0.25f); }));
	require(!accepts([] { validate_depth_bounds(std::nextafter(0.5f, 1.0f), 0.5f); }));
	require(!accepts([] { validate_draw_ids(max_id, 2); }));
	require(!accepts([] { validate_draw_ids(max_id + 1, 0); }));
	require(!accepts([] { validate_draw_ids(0, max_id + 1); }));
	require(!accepts([] { validate_draw_ids(1, max_size); }));
	// Compare index validation to an independently enumerated byte-address model,
	// including empty buffers, exact ends, both index sizes and odd offsets.
	std::size_t models = 0;
	for (std::size_t size = 0; size <= 64; ++size)
	{
		for (std::size_t offset = 0; offset <= 68; ++offset)
		{
			for (std::size_t count = 0; count <= 36; ++count)
			{
				for (std::size_t bytes : {std::size_t{2}, std::size_t{4}})
				{
					bool expected = offset <= size && offset % bytes == 0;
					for (std::size_t byte = 0; byte < count * bytes; ++byte)
					{
						if (offset + byte >= size) expected = false;
					}
					require(accepts([&] { checked_index_bytes(size, offset, count, bytes); }) == expected);
					++models;
				}
			}
		}
	}
	require(checked_index_bytes(max_size, 0, max_size / 4, 4) == max_size - 3);
	require(!accepts([] { checked_index_bytes(max_size, 0, max_size, 4); }));
	require(!accepts([] { checked_index_bytes(16, 0, 1, 1); }));
	validate_scissor(max_size, max_size, max_size, max_size, 0, 0);
	validate_scissor(16, 16, 8, 8, 8, 8);
	require(!accepts([] { validate_scissor(16, 16, 8, 8, 9, 8); }));
	require(!accepts([] { validate_scissor(16, 16, max_size, 0, 1, 1); }));
	require(!accepts([] { validate_scissor(0, 16, 0, 0, 0, 0); }));
	validate_visibility_offset(16, 8);
	validate_visibility_offset(max_size, max_size - 15);
	require(!accepts([] { validate_visibility_offset(max_size, max_size - 7); }));
	require(!accepts([] { validate_visibility_offset(16, 1); }));
	require(!accepts([] { validate_visibility_offset(16, 16); }));
	std::cout << "Metal 4 draw validation: " << models << " index address models and boundary checks;native/mirror capability selection,narrow/equal/full-range intervals and invalid bounds passed\n";
}
