#include "adaptive_btree/inner_node.hpp"

#include <algorithm>
#include <utility>

#include "adaptive_btree/common.hpp"

namespace abt
{

    InnerNode::InnerNode(NodeId id) : Node(id, NodeType::kInner) {}

    InnerMaterialized InnerNode::materialize() const
    {
        InnerMaterialized out;
        out.left_child = page_.leftChild();
        out.entries.reserve(page_.slotCount());

        const std::string prefix = page_.prefix();
        for (std::uint16_t i = 0; i < page_.slotCount(); ++i)
        {
            std::string key = prefix;
            key.append(page_.keySuffix(i));
            out.entries.push_back(InnerEntry{std::move(key), page_.rightChild(i)});
        }
        return out;
    }

    bool InnerNode::rebuild(NodeId left_child, const std::vector<InnerEntry> &sorted_entries)
    {
        SlottedPage candidate(NodeType::kInner);
        const std::string prefix = commonPrefix(sorted_entries);
        candidate.setPrefix(prefix);
        candidate.setLeftChild(left_child);

        for (std::uint16_t i = 0; i < sorted_entries.size(); ++i)
        {
            const std::string_view key = sorted_entries[i].key;
            const std::string_view suffix = key.substr(prefix.size());
            const std::uint16_t hint = make_prefix_hint(suffix);
            if (!candidate.insertInner(i, suffix, sorted_entries[i].right_child, hint))
            {
                return false;
            }
        }

        page_ = std::move(candidate);
        return true;
    }

    std::size_t InnerNode::childIndexForKey(std::string_view key) const
    {
        std::size_t lo = 0;
        std::size_t hi = page_.slotCount();

        const std::string prefix = page_.prefix();
        const bool can_use_hint = starts_with(key, prefix);
        std::uint16_t target_hint = 0;
        if (can_use_hint)
        {
            target_hint = make_prefix_hint(key.substr(prefix.size()));
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

            // upper_bound: first key greater than search key.
            if (cmp <= 0)
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

    NodeId InnerNode::childAt(std::size_t child_index) const
    {
        if (child_index == 0)
        {
            return page_.leftChild();
        }
        return page_.rightChild(static_cast<std::uint16_t>(child_index - 1));
    }

    NodeId InnerNode::childForKey(std::string_view key) const
    {
        return childAt(childIndexForKey(key));
    }

    int InnerNode::compareKeyAt(std::uint16_t slot_index, std::string_view key) const
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

    std::string InnerNode::commonPrefix(const std::vector<InnerEntry> &sorted_entries)
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
