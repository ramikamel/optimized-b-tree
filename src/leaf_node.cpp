#include "adaptive_btree/leaf_node.hpp"

#include <algorithm>
#include <cstring>

namespace abt
{

    namespace
    {
        // A single thread-local scratch page used as a build target during splits.
        // Reused across calls; SlottedPage::init() resets it cheaply (no 4 KiB memset).
        thread_local SlottedPage tls_scratch;
    }

    void LeafNode::initEmpty(std::string_view lower_fence, std::string_view upper_fence, NodeId next_leaf)
    {
        page_.init(NodeType::kLeaf, lower_fence, upper_fence);
        page_.setLink(next_leaf);
    }

    bool LeafNode::tryInsert(std::string_view key, Value value, bool& inserted_new)
    {
        // Static prefix truncation invariant: any key reaching this leaf must
        // share the (statically computed) prefix of the leaf, because routing
        // and splits guarantee the key lies between the leaf's fences.
        const std::size_t prefix_len = page_.prefixLen();
        const std::string_view suffix(key.data() + prefix_len, key.size() - prefix_len);
        const std::uint32_t head = make_head(suffix);

        // Tail-insert fast path: sorted/sequential workloads hit this almost
        // always. A single head compare (and rarely a suffix compare) beats a
        // full hint+binary-search.
        const std::uint16_t n = page_.slotCount();
        if (__builtin_expect(n > 0, 1))
        {
            const std::uint16_t last_idx = static_cast<std::uint16_t>(n - 1);
            const std::uint32_t last_head = page_.headAt(last_idx);
            if (__builtin_expect(head > last_head ||
                    (head == last_head && suffix > page_.keySuffix(last_idx)), 1))
            {
                if (!page_.hasSpaceForLeaf(suffix.size())) return false;
                page_.appendLeaf(suffix, value, head);
                // The new slot is past every hint sample point unless the slot
                // count crosses a spacing boundary, in which case rebuild.
                if ((n / (kHintCount + 1)) != ((n + 1) / (kHintCount + 1))) page_.rebuildHints();
                inserted_new = true;
                return true;
            }
        }
        // Slow path: out-of-order insert (or first-ever insert into the leaf).
        const std::uint16_t idx = (n == 0) ? std::uint16_t{0} : page_.lowerBoundIndex(head, suffix);
        if (idx < n && page_.keySuffix(idx) == suffix)
        {
            page_.setLeafValue(idx, value);
            inserted_new = false;
            return true;
        }
        if (!page_.insertLeaf(idx, suffix, value, head)) return false;
        page_.rebuildHints();
        inserted_new = true;
        return true;
    }

    std::optional<Value> LeafNode::find(std::string_view key) const
    {
        const std::size_t prefix_len = page_.prefixLen();
        if (key.size() < prefix_len) return std::nullopt;
        const std::string_view suffix(key.data() + prefix_len, key.size() - prefix_len);
        const std::uint32_t head = make_head(suffix);

        const std::uint16_t idx = page_.lowerBoundIndex(head, suffix);
        if (idx < page_.slotCount() && page_.keySuffix(idx) == suffix)
        {
            return page_.leafValue(idx);
        }
        return std::nullopt;
    }

    std::uint16_t LeafNode::lowerBoundIndex(std::string_view key) const
    {
        const std::string_view prefix = page_.prefixView();
        if (key.size() < prefix.size())
        {
            // Compare prefix vs key: if key < prefix, lower_bound is 0.
            const int cmp = key.compare(prefix.substr(0, key.size()));
            return cmp < 0 ? std::uint16_t{0} : page_.slotCount();
        }
        const int pcmp = std::memcmp(key.data(), prefix.data(), prefix.size());
        if (pcmp < 0) return 0;
        if (pcmp > 0) return page_.slotCount();
        const std::string_view suffix = key.substr(prefix.size());
        return page_.lowerBoundIndex(make_head(suffix), suffix);
    }

    std::string LeafNode::keyAt(std::uint16_t i) const
    {
        const std::string_view prefix = page_.prefixView();
        const std::string_view suffix = page_.keySuffix(i);
        std::string out;
        out.reserve(prefix.size() + suffix.size());
        out.append(prefix);
        out.append(suffix);
        return out;
    }

    std::string LeafNode::splitInto(LeafNode& right_node, NodeId right_id)
    {
        SlottedPage& src = page_;
        const std::uint16_t n = src.slotCount();

        // 1. Pick split index by accumulated payload bytes (not just key count)
        //    so variable-length keys are balanced.
        std::size_t total = 0;
        for (std::uint16_t i = 0; i < n; ++i)
            total += src.keySuffix(i).size() + sizeof(Value);

        std::uint16_t mid = static_cast<std::uint16_t>(n / 2);
        if (mid == 0) mid = 1;
        if (mid >= n) mid = static_cast<std::uint16_t>(n - 1);
        {
            const std::size_t target = total / 2;
            std::size_t acc = 0;
            for (std::uint16_t i = 0; i < n; ++i)
            {
                acc += src.keySuffix(i).size() + sizeof(Value);
                if (acc >= target)
                {
                    mid = (i == 0) ? std::uint16_t{1} : i;
                    break;
                }
            }
        }
        if (mid == 0) mid = 1;
        if (mid >= n) mid = static_cast<std::uint16_t>(n - 1);

        // 2. Compute the truncated separator (paper §2 "Separator Selection").
        //    separator = LCP(last_left_full, first_right_full) + first differing byte from right.
        const std::string_view src_prefix = src.prefixView();
        const std::string_view last_left_suffix = src.keySuffix(static_cast<std::uint16_t>(mid - 1));
        const std::string_view first_right_suffix = src.keySuffix(mid);
        const std::size_t suf_common = longest_common_prefix(last_left_suffix, first_right_suffix);

        std::string separator;
        const std::size_t sep_take = std::min(first_right_suffix.size(), suf_common + 1);
        separator.reserve(src_prefix.size() + sep_take);
        separator.append(src_prefix);
        separator.append(first_right_suffix.data(), sep_take);

        // 3. Snapshot fences before we tear down the source page.
        //    Lower/upper fence views point into src.bytes_; we copy them into a
        //    thread-local buffer because the rebuild below overwrites src.
        thread_local std::string lower_buf;
        thread_local std::string upper_buf;
        lower_buf.assign(src.lowerFenceView());
        upper_buf.assign(src.upperFenceView());
        const bool had_lower = src.hasLowerFence();
        const bool had_upper = src.hasUpperFence();
        const NodeId src_next = src.link();
        const std::string_view src_lower = had_lower ? std::string_view(lower_buf) : std::string_view{};
        const std::string_view src_upper = had_upper ? std::string_view(upper_buf) : std::string_view{};

        // 4. Build the right node into the thread-local scratch page, then memcpy.
        //    Right's fences are (separator, src_upper). Its prefix is LCP(separator, src_upper)
        //    which is guaranteed >= old prefix (separator and src_upper both share old prefix
        //    with src_lower, so they share at least old_prefix bytes with each other).
        tls_scratch.init(NodeType::kLeaf, separator, src_upper);
        const std::size_t right_prefix_len = tls_scratch.prefixView().size();
        const std::size_t eat_right = right_prefix_len - src_prefix.size();
        for (std::uint16_t i = mid; i < n; ++i)
        {
            const std::string_view old_suf = src.keySuffix(i);
            const Value v = src.leafValue(i);
            const std::string_view new_suf =
                old_suf.size() > eat_right
                    ? old_suf.substr(eat_right)
                    : std::string_view{};
            tls_scratch.appendLeaf(new_suf, v, make_head(new_suf));
        }
        tls_scratch.setLink(src_next);
        tls_scratch.rebuildHints();
        std::memcpy(right_node.page_.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

        // 5. Rebuild the left node into scratch, then copy over src.
        //    We must read left-half slot data from src BEFORE overwriting, which
        //    is what we do here (the loop reads src; the memcpy at the end writes src).
        tls_scratch.init(NodeType::kLeaf, src_lower, separator);
        const std::size_t left_prefix_len = tls_scratch.prefixView().size();
        const std::size_t eat_left = left_prefix_len - src_prefix.size();
        for (std::uint16_t i = 0; i < mid; ++i)
        {
            const std::string_view old_suf = src.keySuffix(i);
            const Value v = src.leafValue(i);
            const std::string_view new_suf =
                old_suf.size() > eat_left
                    ? old_suf.substr(eat_left)
                    : std::string_view{};
            tls_scratch.appendLeaf(new_suf, v, make_head(new_suf));
        }
        tls_scratch.setLink(right_id);
        tls_scratch.rebuildHints();
        std::memcpy(src.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

        return separator;
    }

} // namespace abt
