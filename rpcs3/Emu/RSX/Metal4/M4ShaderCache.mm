#include "M4ShaderCache.h"

#if !__has_feature(objc_arc)
#error "The native Metal 4 backend requires ARC."
#endif

#include <algorithm>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace metal4
{
struct shader_cache::implementation
{
	struct vertex_entry
	{
		std::vector<std::array<std::uint32_t, 4>> code;
		std::vector<std::uint8_t> live;
		vertex_program_options options;
		std::uint64_t fingerprint = 0;
		generated_vertex_program generated;
		std::shared_ptr<const vertex_shader> shader;
		std::optional<compiler::prepared_library> library;
		std::size_t bytes() const noexcept
		{
			return sizeof(*this) + sizeof(vertex_shader) + code.capacity() * sizeof(code[0]) + live.capacity() + generated.source.capacity() + 1 +
				(library ? library->retained_bytes() : 0);
		}
		bool matches_options(const vertex_program_options& value) const
		{
			return options.entry == value.entry && options.base_address == value.base_address &&
				   options.flat_colors == value.flat_colors && options.point_output == value.point_output && options.packed_inputs == value.packed_inputs && options.flat_result_inputs == value.flat_result_inputs && options.border_textures == value.border_textures && options.textures == value.textures &&
				   std::ranges::equal(live, value.live_instructions);
		}
		bool matches(std::span<const std::array<std::uint32_t, 4>> input, const vertex_program_options& value) const
		{
			return matches_options(value) && std::ranges::equal(code, input);
		}
	};
	struct fragment_entry
	{
		decoded_fragment_program image;
		std::vector<std::byte> code;
		std::uint64_t fingerprint = 0;
		fragment_program_options options;
		generated_fragment_program generated;
		std::shared_ptr<const fragment_shader> shader;
		std::optional<compiler::prepared_library> library;
		std::size_t bytes() const noexcept
		{
			return sizeof(*this) + sizeof(fragment_shader) + code.capacity() + 2 * image.instructions.capacity() * sizeof(fragment_instruction) +
				generated.source.capacity() + 1 + generated.constant_offsets.capacity() * sizeof(std::size_t) +
				(library ? library->retained_bytes() : 0);
		}
		bool matches(std::span<const std::byte> input, const fragment_program_options& value) const
		{
			if (options != value || input.size() < image.byte_length)
				return false;
			// The stored validated layout identifies instruction blocks exactly.
			// Compare all words, including dead operands/flow overlays. Only the
			// complete intervening literal blocks and trailing bytes are mutable.
			for (const auto& instruction : image.instructions)
				if (decode_fragment_block(input.subspan(instruction.byte_offset, 16).first<16>()) != instruction.words)
					return false;
			return true;
		}
	};
	template <typename Entry>
	struct stage
	{
		using list = std::list<std::shared_ptr<Entry>>;
		list entries;
		std::unordered_multimap<std::uint64_t, typename list::iterator> index;
		void erase_index(typename list::iterator entry)
		{
			const auto [first, last] = index.equal_range((*entry)->fingerprint);
			for (auto it = first; it != last; ++it)
				if (it->second == entry) { index.erase(it); return; }
		}
		shader_cache_statistics stats;
		template <typename Code, typename Options, typename Build>
		std::shared_ptr<Entry> get(Code code, const Options& options, std::uint64_t fingerprint, Build build)
		{
			++stats.lookups;
			const auto [first, last] = index.equal_range(fingerprint);
			for (auto candidate = first; candidate != last; ++candidate)
			{
				++stats.candidates;
				const auto it = candidate->second;
				if ((*it)->matches(code, options))
				{
					const auto result = *it;
					entries.splice(entries.begin(), entries, it);
					++stats.reused;
					return result;
				}
			}
			auto result = build();
			result->fingerprint = fingerprint;
			++stats.translated;
			const auto bytes = result->bytes();
			if (!stats.capacity || bytes > stats.host_budget)
			{
				++stats.bypassed;
				return result;
			}
			entries.push_front(result);
			try { index.emplace(fingerprint, entries.begin()); }
			catch (...) { entries.pop_front(); throw; }
			while (entries.size() > stats.capacity || stats.host_bytes > stats.host_budget - bytes)
			{
				stats.host_bytes -= entries.back()->bytes();
				erase_index(std::prev(entries.end()));
				entries.pop_back();
				++stats.evicted;
			}
			stats.host_bytes += bytes;
			stats.entries = entries.size();
			return result;
		}
		void clear()
		{
			stats.evicted += entries.size();
			index.clear();
			entries.clear();
			stats.entries = stats.host_bytes = 0;
		}
		void refresh_charge(const std::shared_ptr<Entry>& entry, std::size_t before)
		{
			const auto [first, last] = index.equal_range(entry->fingerprint);
			bool retained = false;
			for (auto it = first; it != last; ++it) if (*it->second == entry) { retained = true; break; }
			if (!retained) return;
			stats.host_bytes -= before; stats.host_bytes += entry->bytes();
			while (stats.host_bytes > stats.host_budget)
			{
				stats.host_bytes -= entries.back()->bytes();
				erase_index(std::prev(entries.end())); entries.pop_back(); ++stats.evicted;
			}
			stats.entries = entries.size();
		}
	};
	compiler compile;
	std::shared_ptr<void> identity = std::make_shared<int>(0);
	stage<vertex_entry> vertices;
	stage<fragment_entry> fragments;
	explicit implementation(const compiler& value) : compile(value) {}
	std::shared_ptr<vertex_entry> vertex(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options)
	{
		std::uint64_t hash = 14695981039346656037ull;
		for (const auto& instruction : code)
			for (auto word : instruction) hash = (hash ^ word) * 1099511628211ull;
		return vertices.get(code, options, hash, [&]
							{
			auto result = std::make_shared<vertex_entry>();
			result->generated = generate_vertex_program(code, options);
			result->code.assign(code.begin(), code.end());
			result->live.assign(options.live_instructions.begin(), options.live_instructions.end());
			result->options = options;
			result->options.live_instructions = {}; // Never retain caller spans.
			return result; });
	}
	std::shared_ptr<fragment_entry> fragment(std::span<const std::byte> code, const fragment_program_options& options)
	{
		return fragments.get(code, options, fragment_program_fingerprint(code), [&]
							 {
			auto result = std::make_shared<fragment_entry>();
			result->image = decode_fragment_program(code);
			result->code.assign(code.begin(), code.begin() + result->image.byte_length);
			result->generated = generate_fragment_program(code, options);
			result->options = options;
			return result; });
	}
};

shader_cache::shader_cache(const compiler& compiler, std::size_t capacity, std::size_t host_budget)
{
	if (capacity > 4096 || (capacity && !host_budget))
		throw std::invalid_argument("Metal4 shader cache capacity/host budget is invalid");
	m_impl = std::make_unique<implementation>(compiler);
	for (auto stats : {&m_impl->vertices.stats, &m_impl->fragments.stats})
	{
		stats->capacity = capacity;
		stats->host_budget = host_budget;
	}
}
shader_cache::~shader_cache() = default;
std::uint32_t shader_cache::vertex_mask(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options)
{
	return prepare_vertex(code, options).texture_mask();
}
std::uint32_t shader_cache::fragment_mask(std::span<const std::byte> code, const fragment_program_options& options)
{
	return prepare_fragment(code, options).texture_mask();
}
std::shared_ptr<const vertex_shader> shader_cache::vertex(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options)
{
	return vertex(prepare_vertex(code, options), options);
}
shader_cache::prepared_vertex shader_cache::prepare_vertex(std::span<const std::array<std::uint32_t, 4>> code, const vertex_program_options& options)
{
	const auto entry = m_impl->vertex(code, options);
	return {m_impl->identity, entry, entry->generated.texture_mask};
}
shader_cache::prepared_vertex shader_cache::prewarm_vertex(const prepared_vertex& program, const vertex_program_options& options)
{
	if (program.m_owner != m_impl->identity || !program.m_entry) throw std::invalid_argument("Foreign prepared vertex prewarm");
	auto entry = std::static_pointer_cast<implementation::vertex_entry>(program.m_entry);
	if (!entry->matches_options(options)) entry = m_impl->vertex(entry->code, options);
	if (entry->shader || entry->library) ++m_impl->vertices.stats.prewarm_reused;
	else
	{
		const auto before = entry->bytes();
		entry->library = m_impl->compile.prewarm_library(entry->generated.source, @"RPCS3 native RSX vertex program");
		if (entry->library) ++m_impl->vertices.stats.prewarmed;
		else ++m_impl->vertices.stats.prewarm_refused;
		m_impl->vertices.refresh_charge(entry, before);
	}
	return {m_impl->identity, entry, entry->generated.texture_mask};
}
bool shader_cache::ready(const prepared_vertex& program) const
{
	if (program.m_owner != m_impl->identity || !program.m_entry) throw std::invalid_argument("Foreign prepared vertex readiness");
	const auto entry = std::static_pointer_cast<implementation::vertex_entry>(program.m_entry);
	return entry->shader || (entry->library && entry->library->ready());
}
std::shared_ptr<const vertex_shader> shader_cache::vertex(const prepared_vertex& program, const vertex_program_options& options)
{
	if (program.m_owner != m_impl->identity || !program.m_entry)
		throw std::invalid_argument("Metal4 prepared vertex belongs to another cache");
	auto entry = std::static_pointer_cast<implementation::vertex_entry>(program.m_entry);
	if (entry->matches_options(options)) ++m_impl->vertices.stats.prepared_hits;
	else entry = m_impl->vertex(entry->code, options);
	if (!entry->shader)
	{
		const auto before = entry->bytes();
		if (entry->library && !entry->library->ready()) ++m_impl->vertices.stats.prewarm_waited;
		id<MTLLibrary> library = nil;
		try { if (entry->library) library = m_impl->compile.library(*entry->library); }
		catch (...) { if (entry->library && entry->library->failed()) { entry->library.reset(); m_impl->vertices.refresh_charge(entry, before); } throw; }
		entry->shader = std::shared_ptr<const vertex_shader>(new vertex_shader(m_impl->compile, entry->generated, options, library));
		entry->library.reset(); m_impl->vertices.refresh_charge(entry, before);
		++m_impl->vertices.stats.compiled;
	}
	return entry->shader;
}
std::shared_ptr<const fragment_shader> shader_cache::fragment(std::span<const std::byte> code, const fragment_program_options& options)
{
	return fragment(prepare_fragment(code, options), options);
}
shader_cache::prepared_fragment shader_cache::prepare_fragment(std::span<const std::byte> code, const fragment_program_options& options)
{
	const auto entry = m_impl->fragment(code, options);
	return {m_impl->identity, entry, entry->generated.texture_mask};
}
shader_cache::prepared_fragment shader_cache::prewarm_fragment(const prepared_fragment& program, const fragment_program_options& options)
{
	if (program.m_owner != m_impl->identity || !program.m_entry) throw std::invalid_argument("Foreign prepared fragment prewarm");
	auto entry = std::static_pointer_cast<implementation::fragment_entry>(program.m_entry);
	if (entry->options != options) entry = m_impl->fragment(entry->code, options);
	if (entry->shader || entry->library) ++m_impl->fragments.stats.prewarm_reused;
	else
	{
		const auto before = entry->bytes();
		entry->library = m_impl->compile.prewarm_library(entry->generated.source, @"RPCS3 native RSX fragment program");
		if (entry->library) ++m_impl->fragments.stats.prewarmed;
		else ++m_impl->fragments.stats.prewarm_refused;
		m_impl->fragments.refresh_charge(entry, before);
	}
	return {m_impl->identity, entry, entry->generated.texture_mask};
}
bool shader_cache::ready(const prepared_fragment& program) const
{
	if (program.m_owner != m_impl->identity || !program.m_entry) throw std::invalid_argument("Foreign prepared fragment readiness");
	const auto entry = std::static_pointer_cast<implementation::fragment_entry>(program.m_entry);
	return entry->shader || (entry->library && entry->library->ready());
}
std::shared_ptr<const fragment_shader> shader_cache::fragment(const prepared_fragment& program, const fragment_program_options& options)
{
	if (program.m_owner != m_impl->identity || !program.m_entry)
		throw std::invalid_argument("Metal4 prepared fragment belongs to another cache");
	auto entry = std::static_pointer_cast<implementation::fragment_entry>(program.m_entry);
	if (entry->options == options) ++m_impl->fragments.stats.prepared_hits;
	else entry = m_impl->fragment(entry->code, options);
	if (!entry->shader)
	{
		const auto before = entry->bytes();
		if (entry->library && !entry->library->ready()) ++m_impl->fragments.stats.prewarm_waited;
		id<MTLLibrary> library = nil;
		try { if (entry->library) library = m_impl->compile.library(*entry->library); }
		catch (...) { if (entry->library && entry->library->failed()) { entry->library.reset(); m_impl->fragments.refresh_charge(entry, before); } throw; }
		entry->shader = std::shared_ptr<const fragment_shader>(new fragment_shader(m_impl->compile, entry->image, entry->generated, options, library));
		entry->library.reset(); m_impl->fragments.refresh_charge(entry, before);
		++m_impl->fragments.stats.compiled;
	}
	return entry->shader;
}
shader_cache_statistics shader_cache::vertex_statistics() const noexcept { return m_impl->vertices.stats; }
shader_cache_statistics shader_cache::fragment_statistics() const noexcept { return m_impl->fragments.stats; }
void shader_cache::clear()
{
	m_impl->vertices.clear();
	m_impl->fragments.clear();
}
} // namespace metal4
