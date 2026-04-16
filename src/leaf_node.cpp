#include "adaptive_btree/leaf_node.hpp"

#include <algorithm>
#include <utility>

#include "adaptive_btree/common.hpp"

namespace abt
{

    LeafNode::LeafNode(NodeId id) : Node(id, NodeType::kLeaf) {}

    std::vector<LeafEntry> LeafNode::entries() const
    {
        std::vector<LeafEntry> out;
        out.reserve(page_.slotCount());
        const std::string prefix = page_.prefix();

        for (std::uint16_t i = 0; i < page_.slotCount(); ++i)
        {
            std::string key = prefix;
            key.append(page_.keySuffix(i));
            out.push_back(LeafEntry{std::move(key), page_.leafValue(i)});
        }
        return out;
    }

    bool LeafNode::rebuild(const std::vector<LeafEntry> &sorted_entries)
    {
        NodeId old_next = page_.nextLeaf();

        SlottedPage candidate(NodeType::kLeaf);
        const std::string prefix = commonPrefix(sorted_entries);
        candidate.setPrefix(prefix);

        for (std::uint16_t i = 0; i < sorted_entries.size(); ++i)
        {
            const std::string_view key = sorted_entries[i].key;
            const std::string_view suffix = key.substr(prefix.size());
            const std::uint16_t hint = make_prefix_hint(suffix);
            if (!candidate.insertLeaf(i, suffix, sorted_entries[i].value, hint))
            {
                return false;
            }
        }

        candidate.setNextLeaf(old_next);
        page_ = std::move(candidate);
        return true;
    }

    std::optional<Value> LeafNode::find(std::string_view key) const
    {
        const std::size_t index = lowerBoundIndex(key);
        if (index >= page_.slotCount())
        {
            return std::nullopt;
        }

        if (compareKeyAt(static_cast<std::uint16_t>(index), key) == 0)
        {
            return page_.leafValue(static_cast<std::uint16_t>(index));
        }
        return std::nullopt;
    }

    std::size_t LeafNode::lowerBoundIndex(std::string_view key) const
    {
        std::size_t lo = 0;
        std::size_t hi = page_.slotCount();

        const std::string prefix = page_.prefix();
        const bool can_use_hint = starts_with(key, prefix);
        std::uint16_t target_hint = 0;
        std::string_view target_suffix;
        if (can_use_hint)
        {
            target_suffix = key.substr(prefix.size());
            target_hint = make_prefix_hint(target_suffix);
        }

        while (lo < hi)
        {
            const std::size_t mid = lo + ((hi - lo) / 2);

            int cmp = 0;
            if (can_use_hint)
            {
                const std::uint16_t slot_hint = page_.hintAt(static_cast<std::uint16_t>(mid));
                if (target_hint < slot_hint)
                {
                    cmp = 1;
                }
                else if (target_hint > slot_hint)
                {
                    cmp = -1;
                }
                else
                {
                    cmp = compareKeyAt(static_cast<std::uint16_t>(mid), key);
                }
            }
            else
            {
                cmp = compareKeyAt(static_cast<std::uint16_t>(mid), key);
            }

            if (cmp < 0)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return lo;
    }

    NodeId LeafNode::nextLeaf() const
    {
        return page_.nextLeaf();
    }

    void LeafNode::setNextLeaf(NodeId id)
    {
        page_.setNextLeaf(id);
    }

    int LeafNode::compareKeyAt(std::uint16_t slot_index, std::string_view key) const
    {
        const std::string prefix = page_.prefix();

        const std::string_view prefix_view(prefix);
        const std::size_t prefix_common = std::min(prefix_view.size(), key.size());
        const int prefix_cmp = std::char_traits<char>::compare(prefix_view.data(), key.data(), prefix_common);
        if (prefix_cmp < 0)
        {
            return -1;
        }
        if (prefix_cmp > 0)
        {
            return 1;
        }

        if (key.size() < prefix_view.size())
        {
            return 1;
        }

        const std::string_view suffix = page_.keySuffix(slot_index);
        const std::string_view key_suffix = key.substr(prefix_view.size());
        return lexical_compare(suffix, key_suffix);
    }

    std::string LeafNode::commonPrefix(const std::vector<LeafEntry> &sorted_entries)
    {
        if (sorted_entries.empty())
        {
            return "";
        }

        std::string prefix = sorted_entries.front().key;
        for (std::size_t i = 1; i < sorted_entries.size() && !prefix.empty(); ++i)
        {
            const std::string &key = sorted_entries[i].key;
            std::size_t len = 0;
            const std::size_t max_len = std::min(prefix.size(), key.size());
            while (len < max_len && prefix[len] == key[len])
            {
                ++len;
            }
            prefix.resize(len);
        }
        return prefix;
    }

} // namespace abt
