#include "M4DrawGeometryPlan.h"
#include "M4RSXVertexSources.h"
#include <iostream>
#include <set>

using namespace metal4;
namespace
{
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
std::byte pattern(std::uint64_t address) { return std::byte((address * 37 + 11) & 255); }
unsigned word(std::span<const std::byte> bytes, std::size_t offset)
{
	unsigned result = 0;
	for (unsigned b = 0; b < 4; ++b) result |= std::to_integer<unsigned>(bytes[offset + b]) << (8 * b);
	return result;
}
// Independent reader of the documented three packed storage encodings.
std::size_t packed_address(const vertex_sparse_bytes& packed, unsigned unit, unsigned element)
{
	const auto d = packed.fetch.attributes[unit];
	const auto mode = d.format >> 30, stride = d.format & 255;
	std::size_t offset = d.storage & 0x1fffffffu;
	if (mode == 2)
	{
		for (unsigned i = 0; i < packed.fetch.element_counts[unit]; ++i)
			if (word(packed.bytes,offset + i * stride) == element) return offset + i * stride + 4;
		throw std::runtime_error("Sparse key missing from output");
	}
	const auto base = mode == 1 ? word(packed.bytes,offset) : 0;
	require(element >= base && element - base < packed.fetch.element_counts[unit],"Window omitted a needed element");
	return offset + (mode == 1 ? 4 : 0) + (element - base) * stride;
}
vertex_stream_range range(std::vector<unsigned> ids)
{
	std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
	vertex_stream_range result;
	result.first_vertex = result.stream_first = ids.front();
	result.vertex_count = ids.back() - ids.front() + 1;
	result.source_vertices = std::make_shared<const std::vector<unsigned>>(std::move(ids));
	return result;
}
void storage_profiles()
{
	unsigned profiles = 0, checked = 0;
	std::set<unsigned> modes;
	for (unsigned type = 1; type <= 7; ++type) for (unsigned count = 1; count <= (type == 6 ? 1u : 4u); ++count)
		for (bool be : {false,true}) for (unsigned frequency : {0u,1u,2u,3u,7u,65535u}) for (bool modulo : {false,true})
			for (const auto& ids : {std::vector<unsigned>{0,1,2,5},std::vector<unsigned>{65535,65536,65537},
				std::vector<unsigned>{0,1,1048575},std::vector<unsigned>{1048500,1048507,1048572}})
			{
				auto r = range(ids);
				const unsigned width = type == 2 || type == 6 ? 4 : type == 4 || type == 7 ? 1 : 2;
				const unsigned payload = count * width, stride = payload + 7;
				vertex_sparse_inputs input; input.attribute_mask = 1;
				input.attributes[0] = {{stride | (frequency << 8) | (type << 24) | (count << 27),
					(be ? 1u << 29 : 0) | (modulo ? 1u << 31 : 0)},true,4096,{}};
				std::set<std::uint64_t> required, read;
				for (const auto id : *r.source_vertices)
				{
					const auto element = !frequency ? 0 : modulo ? id % frequency : id / frequency;
					for (unsigned b = 0; b < payload; ++b) required.insert(4096ull + element * std::uint64_t(stride) + b);
				}
				input.read_memory = [&](unsigned address, std::span<std::byte> bytes)
				{
					for (std::size_t b = 0; b < bytes.size(); ++b)
					{
						require(required.contains(std::uint64_t(address) + b),"Capture read an unreferenced VM byte");
						read.insert(std::uint64_t(address) + b); bytes[b] = pattern(std::uint64_t(address) + b);
					}
				};
				const auto packed = capture_sparse_vertex_bytes(input,r,256);
				require(read == required,"Capture omitted required VM bytes");
				require(packed.fetch.first_vertex == r.first_vertex && packed.fetch.vertex_count == r.vertex_count &&
					packed.fetch.vertex_base_index == 0 && packed.fetch.vertex_index_offset == 0,"Capture rebased original shader IDs");
				require(((packed.fetch.attributes[0].format >> 8) & 65535) == frequency &&
					(packed.fetch.attributes[0].storage & 0xa0000000u) == (input.attributes[0].descriptor.storage & 0xa0000000u),"Capture changed frequency/endian/modulo");
				modes.insert(packed.fetch.attributes[0].format >> 30);
				for (const auto id : *r.source_vertices)
				{
					const auto element = !frequency ? 0 : modulo ? id % frequency : id / frequency;
					const auto offset = packed_address(packed,0,element);
					for (unsigned b = 0; b < payload; ++b)
					{
						require(packed.bytes[offset + b] == pattern(4096ull + element * std::uint64_t(stride) + b),"Packed source bytes differ from address oracle"); ++checked;
					}
				}
				++profiles;
			}
	require(modes == std::set<unsigned>{0,1,2},"Dense/window/sparse representations were not exercised");
	std::cout << "Sparse capture: " << profiles << " format/endian/frequency/modulo/high/wrap profiles, " << checked << " independent byte oracles passed\n";
}
struct location { unsigned index; bool modulo = false; unsigned frequency = 1; };
struct block
{
	bool interleaved = true, single_vertex = false;
	unsigned base_offset = 0, real_offset_address = 0, memory_location = 0;
	std::uint8_t attribute_stride = 0;
	std::pair<unsigned,unsigned> vertex_range{};
	std::vector<location> locations;
};
struct layout
{
	std::array<block,16> owned{};
	unsigned used = 0, attribute_mask = 0;
	std::array<unsigned,16> attribute_placement{};
	std::vector<block*> interleaved_blocks;
	std::vector<std::pair<unsigned,unsigned>> volatile_blocks;
	std::vector<unsigned> referenced_registers;
	block* alloc_interleaved_block() { require(used < 16,"DMA layout overflows 16 blocks"); return &owned[used++]; }
};
struct metadata { unsigned referenced_inputs_mask = 0; };
struct state
{
	struct array_info { unsigned value = 0; unsigned offset() const { return value; } };
	struct register_info { std::array<unsigned,4> data{}; };
	std::array<array_info,16> vertex_arrays_info{};
	std::array<register_info,16> register_vertex_info{};
	vertex_constant_registers transform_constants{};
	struct { unsigned command = 1; std::vector<unsigned> inline_vertex_array; } current_draw_clause;
};
class processor
{
  protected:
	struct push { std::vector<unsigned> data; };
	std::array<push,16> m_vertex_push_buffers;
  public:
	std::array<vertex_attribute_descriptor,16> descriptors{};
	mutable unsigned fills = 0, reads = 0;
	unsigned expected_first = 65537, expected_count = 1;
	std::vector<std::byte> memory = std::vector<std::byte>(1u << 20);
	void set_push(unsigned unit, std::vector<unsigned> values) { m_vertex_push_buffers[unit].data = std::move(values); }
	void fill_vertex_layout_state(const layout& l, const metadata&, unsigned first, unsigned count, std::int32_t* output, unsigned a, unsigned b) const
	{
		require(first == expected_first && count == expected_count && !a && !b,"Descriptor fill did not retain expected source IDs");
		for (const auto block : l.interleaved_blocks) require(block->vertex_range == std::pair{0u,1u},"Descriptor fill can re-scan original index data");
		std::memcpy(output,descriptors.data(),128); ++fills;
	}
	void write_vertex_data_to_memory(const layout& l, unsigned first, unsigned count, void* output, void* transient) const
	{
		require(first == 0 && count == 1 && output && !transient,"Native adapter called full transient writer");
		auto* dest = static_cast<std::byte*>(output);
		for (const auto block : l.interleaved_blocks)
		{
			require(block->vertex_range == std::pair{0u,1u} && block->attribute_stride,"DMA block can calculate a prefix");
			require(std::uint64_t(block->real_offset_address) + block->attribute_stride <= memory.size(),"DMA oracle out of range");
			std::memcpy(dest,memory.data() + block->real_offset_address,block->attribute_stride); dest += block->attribute_stride;
		}
		++reads;
	}
};
void native_sources()
{
	state s; metadata meta{7}; processor p; layout l;
	const auto original = l.alloc_interleaved_block(); original->base_offset = 100;
	original->real_offset_address = 1024; original->attribute_stride = 12; original->locations.push_back({0});
	original->vertex_range = {17,999}; l.interleaved_blocks.push_back(original);
	l.referenced_registers.push_back(1); s.vertex_arrays_info[0].value = 104;
	p.descriptors[0] = {12u | (1u << 8) | (2u << 24) | (2u << 27),1u << 29};
	p.descriptors[1] = {(2u << 24) | (4u << 27),1u << 30};
	p.descriptors[2] = {4u | (1u << 8) | (2u << 24) | (1u << 27),16u | (1u << 30)};
	for (std::size_t i = 0; i < p.memory.size(); ++i) p.memory[i] = pattern(i);
	s.register_vertex_info[1].data = {0x12345678,0x87654321,0x3f800000,0x80000000};
	auto r = range({65535,65536,65537});
	// Push arrays need real elements at those IDs; the capture must retain
	// only the three required words, without invoking the whole push writer.
	std::vector<unsigned> pushed(65538); pushed[65535] = 111; pushed[65536] = 222; pushed[65537] = 333;
	p.set_push(2,pushed);
	const auto inputs = rsx_vertex_sources(p,l,meta,s,r);
	const auto snapshot = vertex_data_snapshot::capture_sparse(inputs,s,r,128);
	require(snapshot.persistent().size() <= 64 && snapshot.transient().empty(),"Native source adapter captured a prefix");
	require(snapshot.sparse_statistics().memory_bytes == 24 && p.reads == 3 && p.fills == 1,"Native source adapter read unneeded VM bytes");
	require(original->vertex_range == std::pair{17u,999u},"Native source adapter mutated shared range cache");
	const auto bytes = std::vector<std::byte>(snapshot.persistent().begin(),snapshot.persistent().end());
	p.memory.clear(); p.set_push(2,{}); s.register_vertex_info[1].data.fill(0); s.transform_constants[0].fill(0xffffffffu);
	require(std::equal(bytes.begin(),bytes.end(),snapshot.persistent().begin()) && snapshot.constants()[0][0] == 0,"Sparse snapshot retained a mutable producer source");
	// Incomplete inline triangle: five source vertices, only three emitted.
	// The adapter copies their fields directly, never the full inline writer.
	s.current_draw_clause.command = 2; s.current_draw_clause.inline_vertex_array = {10,20,30,40,50};
	meta.referenced_inputs_mask = 4; r = range({0,1,2});
	// Fake fill expects high ID; invoke provider with its descriptor epoch,
	// then capture a different independently bounded inline source envelope.
	const auto inline_inputs = rsx_vertex_sources(p,l,meta,s,range({65535,65536,65537}));
	const auto inline_snapshot = vertex_data_snapshot::capture_sparse(inline_inputs,s,r,12);
	require(inline_snapshot.persistent().size() == 12 && word(inline_snapshot.persistent(),0) == 10 && word(inline_snapshot.persistent(),8) == 30,
		"Incomplete inline capture copied discarded primitive vertices");
	// The reference API also rejects its original undersized inline writer
	// contract before allowing a 20-byte copy into a 12-byte allocation.
	layout legacy; p.expected_first = 0; p.expected_count = 3; p.descriptors[2].storage = 1u << 30;
	const auto reads = p.reads; bool refused = false;
	try { (void)vertex_data_snapshot::capture(p,legacy,meta,s,r,{0,12},12); }
	catch (const std::length_error&) { refused = true; }
	require(refused && p.reads == reads,"Legacy incomplete inline capture reached the full writer");
	std::cout << "Native source adapter: cached descriptor layouts, original DMA, protected push span, register/inline copies and producer mutation;legacy inline20-to12byte writer guard passed\n";
}
void bounds_and_sharing()
{
	auto r = range({65535,65536,65537});
	vertex_sparse_inputs input; input.attribute_mask = 65535;
	for (auto& source : input.attributes) source = {{16u | (1u << 8) | (2u << 24) | (4u << 27),0},true,0,{}};
	unsigned reads = 0;
	input.read_memory = [&](unsigned address,std::span<std::byte> bytes) { ++reads; for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = pattern(address + i); };
	const auto packed = capture_sparse_vertex_bytes(input,r,52);
	require(packed.bytes.size() == 52 && packed.statistics.shared_fields == 15 && reads == 1 && packed.statistics.memory_bytes == 48,"Identical fields were duplicated or adjacent reads not merged");
	unsigned refused = 0;
	const auto refusal = [&](auto action)
	{
		reads = 0;
		try { action(); } catch (const std::exception&) { ++refused; }
		require(reads == 0,"Invalid capture touched source memory before complete preflight");
	};
	refusal([&] { (void)capture_sparse_vertex_bytes(input,r,51); });
	input.attributes[15].base = 0xfffffff0ull;
	refusal([&] { (void)capture_sparse_vertex_bytes(input,r,1024); });
	input.attributes[15] = {{16u | (1u << 8) | (2u << 24) | (4u << 27),0},false,0,{}};
	refusal([&] { (void)capture_sparse_vertex_bytes(input,r,1024); });
	input.attribute_mask = 1; r.vertex_base_index = 65536;
	refusal([&] { (void)capture_sparse_vertex_bytes(input,r,1024); });
	r.vertex_base_index = 0; input.attributes[0].descriptor.storage |= 1u << 31; r.vertex_index_offset = -65536;
	refusal([&] { (void)capture_sparse_vertex_bytes(input,r,1024); });
	r.vertex_index_offset = 0; input.attributes[0].descriptor.format |= 3u << 30;
	refusal([&] { (void)capture_sparse_vertex_bytes(input,r,1024); });
	r.source_vertices.reset(); refusal([&] { (void)capture_sparse_vertex_bytes(input,r,1024); });
	require(refused == 7,"Sparse preflight refusal missing");
	const auto high = plan_array_geometry(guest_topology::triangles,1048572,3,{},provoking_vertex::last,{3,3,3});
	require(high.streams.vertex_count == 3 && high.streams.source_vertices->size() == 3,"High-ID source budget still measures prefix");
	r = range({65535,65536,65537}); r.vertex_index_offset = -65532;
	input.attributes[0].descriptor = {16u | (7u << 8) | (2u << 24) | (4u << 27),1u << 31};
	const auto negative = capture_sparse_vertex_bytes(input,r,52);
	for (unsigned element : {3u,4u,5u})
		require(negative.bytes[packed_address(negative,0,element)] == pattern(element * 16),"Valid signed modulo offset lost its original phase");
	std::cout << "Sparse bounds: shared fields, coalesced DMA, high-ID unique-source budget and seven no-read refusals passed\n";
}
void bounded_dma_runs()
{
	state s; metadata meta{1}; processor p; layout l;
	p.expected_first = 1024;
	p.descriptors[0] = {16u | (1u << 8) | (2u << 24) | (4u << 27),0};
	const auto block = l.alloc_interleaved_block(); block->attribute_stride = 16; block->locations.push_back({0}); l.interleaved_blocks.push_back(block);
	for (std::size_t i = 0; i < p.memory.size(); ++i) p.memory[i] = pattern(i);
	std::vector<unsigned> ids(1025); for (unsigned i = 0; i < ids.size(); ++i) ids[i] = i;
	const auto r = range(std::move(ids));
	const auto packed = capture_sparse_vertex_bytes(rsx_vertex_sources(p,l,meta,s,r),r,16400);
	require(packed.bytes.size() == 16400 && packed.statistics.memory_bytes == 16400 && packed.statistics.memory_calls == 5 && p.reads == 5,
		"Bounded coalesced DMA runs duplicated data or exceeded their scratch/block budget");
	for (unsigned i = 0; i < 16400; ++i) require(packed.bytes[i] == pattern(i),"Multi-block DMA run changed source bytes");
	std::cout << "Bounded native DMA adapter:1025 vertices/16400 exact bytes,five <=4080byte runs with <=16 cached <=255byte blocks passed\n";
}
}
int main()
{
	try { storage_profiles(); native_sources(); bounds_and_sharing(); bounded_dma_runs(); return 0; }
	catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
