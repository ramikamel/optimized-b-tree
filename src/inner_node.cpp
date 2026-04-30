#include "adaptive_btree/inner_node.hpp"

#include <cstring>

namespace abt
{

    namespace
    {
        thread_local SlottedPage tls_scratch;
    }

    void InnerNode::initEmpty(std::string_view lower_fence, std::string_view upper_fence, NodeId left_child)
    {
        page_.init(NodeType::kInner, lower_fence, upper_fence);
        page_.setLink(left_child);
    }

    void InnerNode::initRoot(NodeId left_child, std::string_view separator_full_key, NodeId right_child)
    {
        page_.init(NodeType::kInner, {}, {});
        page_.setLink(left_child);
        // Root has no fences, so prefix is empty; the separator stores its full key.
        page_.appendInner(separator_full_key, right_child, make_head(separator_full_key));
        page_.rebuildHints();
    }

    bool InnerNode::tryInsertSeparator(std::string_view separator_full_key, NodeId right_child)
    {
        const std::size_t prefix_len = page_.prefixLen();
        const std::string_view suffix(separator_full_key.data() + prefix_len,
                                      separator_full_key.size() - prefix_len);
        const std::uint32_t head = make_head(suffix);

        const std::uint16_t n = page_.slotCount();
        // Tail-insert fast path: sorted insertion appends new separators in order.
        if (__builtin_expect(n > 0, 1))
        {
            const std::uint16_t last_idx = static_cast<std::uint16_t>(n - 1);
            const std::uint32_t last_head = page_.headAt(last_idx);
            if (__builtin_expect(head > last_head ||
                    (head == last_head && suffix > page_.keySuffix(last_idx)), 1))
            {
                if (!page_.hasSpaceForInner(suffix.size())) return false;
                page_.appendInner(suffix, right_child, head);
                if ((n / (kHintCount + 1)) != ((n + 1) / (kHintCount + 1))) page_.rebuildHints();
                return true;
            }
        }

        const std::uint16_t pos = (n == 0) ? std::uint16_t{0} : page_.lowerBoundIndex(head, suffix);
        if (!page_.insertInner(pos, suffix, right_child, head)) return false;
        page_.rebuildHints();
        return true;
    }

    std::uint16_t InnerNode::childIndexForKey(std::string_view key) const
    {
        // Under correct B-tree routing, every key reaching this node lies between
        // the node's fences and therefore starts with the node's static prefix.
        // Strip the prefix bytes off and proceed to head/suffix routing.
        const std::size_t prefix_len = page_.prefixLen();
        const std::string_view suffix(key.data() + prefix_len, key.size() - prefix_len);
        const std::uint16_t n = page_.slotCount();
        if (n == 0) return 0;
        const std::uint32_t target_head = make_head(suffix);
        // Sequential-workload fast path: key past the last separator -> rightmost.
        const std::uint16_t last_idx = static_cast<std::uint16_t>(n - 1);
        const std::uint32_t last_head = page_.headAt(last_idx);
        if (target_head > last_head ||
            (target_head == last_head && suffix >= page_.keySuffix(last_idx)))
        {
            return n;
        }
        return page_.upperBoundIndex(target_head, suffix);
    }

    NodeId InnerNode::childAt(std::uint16_t child_index) const
    {
        if (child_index == 0) return page_.link();
        return page_.rightChild(static_cast<std::uint16_t>(child_index - 1));
    }

    std::string InnerNode::splitInto(InnerNode& right_node)
    {
        SlottedPage& src = page_;
        const std::uint16_t n = src.slotCount();
        const std::uint16_t mid = static_cast<std::uint16_t>(n / 2);

        // Snapshot fences/links from src before we tear it down.
        thread_local std::string lower_buf;
        thread_local std::string upper_buf;
        thread_local std::string sep_buf;
        lower_buf.assign(src.lowerFenceView());
        upper_buf.assign(src.upperFenceView());

        const std::string_view src_prefix = src.prefixView();
        const std::string_view mid_suffix = src.keySuffix(mid);

        // Promoted (median) key = prefix + mid_suffix. This key disappears from
        // both halves and is inserted into the parent.
        sep_buf.clear();
        sep_buf.reserve(src_prefix.size() + mid_suffix.size());
        sep_buf.append(src_prefix);
        sep_buf.append(mid_suffix);

        const NodeId mid_right_child = src.rightChild(mid);
        const NodeId orig_left_child = src.link();
        const bool had_lower = src.hasLowerFence();
        const bool had_upper = src.hasUpperFence();
        const std::string_view src_lower = had_lower ? std::string_view(lower_buf) : std::string_view{};
        const std::string_view src_upper = had_upper ? std::string_view(upper_buf) : std::string_view{};
        const std::string_view sep_view = sep_buf;

        // Build right node into scratch: fences (sep, src_upper), leftChild = mid_right_child.
        tls_scratch.init(NodeType::kInner, sep_view, src_upper);
        tls_scratch.setLink(mid_right_child);
        const std::size_t right_prefix_len = tls_scratch.prefixView().size();
        const std::size_t eat_right = right_prefix_len - src_prefix.size();
        for (std::uint16_t i = static_cast<std::uint16_t>(mid + 1); i < n; ++i)
        {
            const std::string_view old_suf = src.keySuffix(i);
            const NodeId rc = src.rightChild(i);
            const std::string_view new_suf =
                old_suf.size() > eat_right
                    ? old_suf.substr(eat_right)
                    : std::string_view{};
            tls_scratch.appendInner(new_suf, rc, make_head(new_suf));
        }
        tls_scratch.rebuildHints();
        std::memcpy(right_node.page_.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

        // Build left node into scratch: fences (src_lower, sep), keep original leftChild.
        tls_scratch.init(NodeType::kInner, src_lower, sep_view);
        tls_scratch.setLink(orig_left_child);
        const std::size_t left_prefix_len = tls_scratch.prefixView().size();
        const std::size_t eat_left = left_prefix_len - src_prefix.size();
        for (std::uint16_t i = 0; i < mid; ++i)
        {
            const std::string_view old_suf = src.keySuffix(i);
            const NodeId rc = src.rightChild(i);
            const std::string_view new_suf =
                old_suf.size() > eat_left
                    ? old_suf.substr(eat_left)
                    : std::string_view{};
            tls_scratch.appendInner(new_suf, rc, make_head(new_suf));
        }
        tls_scratch.rebuildHints();
        std::memcpy(src.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

        return sep_buf;
    }

} // namespace abt
