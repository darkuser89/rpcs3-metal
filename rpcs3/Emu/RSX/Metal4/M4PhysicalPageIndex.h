#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace metal4
{
// Writer-excluded metadata only. One fixed slot per logical guest page, with
// an intrusive AVL ordered by (allocation control block, physical offset, VA).
// Updates allocate nothing and never retain the source allocation. In
// particular, an expired allocation cannot be identified by a recycled VA,
// source pointer, or independent kernel mapping address.
class physical_page_index final
{
	static constexpr std::uint32_t absent = std::numeric_limits<std::uint32_t>::max();
	struct node
	{
		std::weak_ptr<void> owner;
		std::uint64_t offset = 0;
		std::uint32_t left = absent, right = absent;
		std::uint8_t height = 0;
	};
	std::vector<node> m_nodes;
	std::uint32_t m_root = absent;
	std::size_t m_size = 0;
	[[nodiscard]] unsigned height(std::uint32_t n) const noexcept { return n == absent ? 0 : m_nodes[n].height; }
	void refresh(std::uint32_t n) noexcept
	{
		m_nodes[n].height = static_cast<std::uint8_t>(1 + std::max(height(m_nodes[n].left), height(m_nodes[n].right)));
	}
	[[nodiscard]] bool less(std::uint32_t a, std::uint32_t b) const noexcept
	{
		const auto& x = m_nodes[a]; const auto& y = m_nodes[b];
		if (x.owner.owner_before(y.owner)) return true;
		if (y.owner.owner_before(x.owner)) return false;
		return x.offset != y.offset ? x.offset < y.offset : a < b;
	}
	std::uint32_t rotate_left(std::uint32_t n) noexcept
	{
		const auto next = m_nodes[n].right;
		m_nodes[n].right = m_nodes[next].left; m_nodes[next].left = n;
		refresh(n); refresh(next); return next;
	}
	std::uint32_t rotate_right(std::uint32_t n) noexcept
	{
		const auto next = m_nodes[n].left;
		m_nodes[n].left = m_nodes[next].right; m_nodes[next].right = n;
		refresh(n); refresh(next); return next;
	}
	std::uint32_t balance(std::uint32_t n) noexcept
	{
		refresh(n);
		const auto left = m_nodes[n].left, right = m_nodes[n].right;
		if (height(left) > height(right) + 1)
		{
			if (height(m_nodes[left].right) > height(m_nodes[left].left)) m_nodes[n].left = rotate_left(left);
			return rotate_right(n);
		}
		if (height(right) > height(left) + 1)
		{
			if (height(m_nodes[right].left) > height(m_nodes[right].right)) m_nodes[n].right = rotate_right(right);
			return rotate_left(n);
		}
		return n;
	}
	std::uint32_t insert(std::uint32_t root, std::uint32_t id) noexcept
	{
		if (root == absent) return id;
		if (less(id, root)) m_nodes[root].left = insert(m_nodes[root].left, id);
		else m_nodes[root].right = insert(m_nodes[root].right, id);
		return balance(root);
	}
	std::uint32_t detach_first(std::uint32_t root, std::uint32_t& first) noexcept
	{
		if (m_nodes[root].left == absent) { first = root; return m_nodes[root].right; }
		m_nodes[root].left = detach_first(m_nodes[root].left, first);
		return balance(root);
	}
	std::uint32_t erase(std::uint32_t root, std::uint32_t id) noexcept
	{
		if (root != id)
		{
			if (less(id, root)) m_nodes[root].left = erase(m_nodes[root].left, id);
			else m_nodes[root].right = erase(m_nodes[root].right, id);
			return balance(root);
		}
		const auto left = m_nodes[root].left, right = m_nodes[root].right;
		if (left == absent) return right;
		if (right == absent) return left;
		std::uint32_t next;
		const auto following = detach_first(right, next);
		m_nodes[next].left = left; m_nodes[next].right = following;
		return balance(next);
	}
	template <typename Visit>
	void query(std::uint32_t id, const std::weak_ptr<void>& owner, std::uint64_t first,
		std::uint64_t end, Visit& visit, std::size_t& visits) const
	{
		if (id == absent) return;
		++visits;
		const auto& n = m_nodes[id];
		if (n.owner.owner_before(owner)) { query(n.right, owner, first, end, visit, visits); return; }
		if (owner.owner_before(n.owner)) { query(n.left, owner, first, end, visit, visits); return; }
		if (n.offset + 4096 > first) query(n.left, owner, first, end, visit, visits);
		const auto begin = std::max(first, n.offset), stop = std::min(end, n.offset + 4096);
		if (begin < stop) visit(static_cast<std::uint32_t>(std::uint64_t(id) * 4096 + begin - n.offset), stop - begin);
		if (n.offset < end) query(n.right, owner, first, end, visit, visits);
	}
  public:
	explicit physical_page_index(std::size_t pages = 1u << 20) : m_nodes(checked_pages(pages)) {}
	physical_page_index(const physical_page_index&) = delete;
	physical_page_index& operator=(const physical_page_index&) = delete;
	[[nodiscard]] static std::size_t checked_pages(std::size_t pages)
	{
		if (!pages || pages > (1u << 20)) throw std::invalid_argument("Physical page index exceeds the guest address domain");
		return pages;
	}
	[[nodiscard]] std::size_t size() const noexcept { return m_size; }
	[[nodiscard]] std::size_t capacity() const noexcept { return m_nodes.size(); }
	[[nodiscard]] std::size_t metadata_bytes() const noexcept { return m_nodes.size() * sizeof(node); }
	[[nodiscard]] unsigned tree_height() const noexcept { return height(m_root); }
	// False means invalid metadata; the caller must fail closed, never use a
	// partially trusted index. An absent owner removes the previous mapping.
	[[nodiscard]] bool replace(std::uint32_t page, const std::shared_ptr<void>& owner, std::uint64_t offset) noexcept
	{
		if (page >= m_nodes.size() || (owner && ((offset & 4095) || offset > std::numeric_limits<std::uint64_t>::max() - 4096))) return false;
		auto& n = m_nodes[page];
		if (n.height && owner && !n.owner.owner_before(owner) && !owner.owner_before(n.owner) && n.offset == offset) return true;
		if (n.height) { m_root = erase(m_root, page); --m_size; }
		n = {};
		if (owner)
		{
			n.owner = owner; n.offset = offset; n.height = 1;
			m_root = insert(m_root, page); ++m_size;
		}
		return true;
	}
	// Exact intersections in allocation/offset order. Callback may throw for
	// its output quota but must not mutate this index. Traversal needs no heap.
	template <typename Visit>
	std::size_t aliases(const std::weak_ptr<void>& owner, std::uint64_t offset, std::uint64_t size, Visit&& visit) const
	{
		if (size > std::numeric_limits<std::uint64_t>::max() - offset) throw std::invalid_argument("Physical page query wraps allocation offsets");
		std::size_t visits = 0;
		if (size && !owner.expired()) query(m_root, owner, offset, offset + size, visit, visits);
		return visits;
	}
};
} // namespace metal4
