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

    InnerMaterializedView InnerNode::materializeViews() const {
        InnerMaterializedView out;
        out.left_child = page_.leftChild();
        out.entries.reserve(page_.slotCount() + 1);
        for (std::uint16_t i = 0; i < page_.slotCount(); ++i) {
            out.entries.push_back({page_.keySuffix(i), page_.rightChild(i), false});
        }
        return out;
    }

    std::string InnerNode::commonPrefixFromViews(const std::vector<InnerEntryView> &sorted_entries, std::string_view new_key, std::string_view old_prefix) {
        if (sorted_entries.empty()) return "";
        
        const auto& first_e = sorted_entries.front();
        const auto& last_e = sorted_entries.back();
        
        std::string prefix;
        std::size_t i = 0;
        while (true) {
            char c1, c2;
            if (first_e.is_new) {
                if (i >= new_key.size()) break;
                c1 = new_key[i];
            } else {
                if (i < old_prefix.size()) c1 = old_prefix[i];
                else if (i - old_prefix.size() < first_e.key.size()) c1 = first_e.key[i - old_prefix.size()];
                else break;
            }
            
            if (last_e.is_new) {
                if (i >= new_key.size()) break;
                c2 = new_key[i];
            } else {
                if (i < old_prefix.size()) c2 = old_prefix[i];
                else if (i - old_prefix.size() < last_e.key.size()) c2 = last_e.key[i - old_prefix.size()];
                else break;
            }
            
            if (c1 != c2) break;
            prefix.push_back(c1);
            ++i;
        }
        return prefix;
    }

    bool InnerNode::rebuildFromViews(NodeId left_child, const std::vector<InnerEntryView> &sorted_entries, std::string_view new_key) {
        SlottedPage candidate(NodeType::kInner);
        std::string_view old_prefix = page_.prefixView();
        std::string prefix = commonPrefixFromViews(sorted_entries, new_key, old_prefix);
        candidate.setPrefix(prefix);
        candidate.setLeftChild(left_child);

        for (std::uint16_t i = 0; i < sorted_entries.size(); ++i) {
            std::string_view p1, p2;
            if (sorted_entries[i].is_new) {
                std::size_t start = std::min(prefix.size(), new_key.size());
                p1 = std::string_view(new_key).substr(start);
            } else {
                if (prefix.size() <= old_prefix.size()) {
                    std::size_t start = std::min(prefix.size(), old_prefix.size());
                    p1 = old_prefix.substr(start);
                    p2 = sorted_entries[i].key;
                } else {
                    std::size_t eat = prefix.size() - old_prefix.size();
                    eat = std::min(eat, sorted_entries[i].key.size());
                    p1 = sorted_entries[i].key.substr(eat);
                }
            }
            
            char buf[4] = {0, 0, 0, 0};
            std::size_t n1 = std::min<std::size_t>(4, p1.size());
            std::memcpy(buf, p1.data(), n1);
            std::size_t n2 = std::min<std::size_t>(4 - n1, p2.size());
            std::memcpy(buf + n1, p2.data(), n2);
            std::uint16_t hint = make_prefix_hint(std::string_view(buf, n1 + n2));

            if (!candidate.insertInner(i, p1, p2, sorted_entries[i].right_child, hint)) return false;
        }

        page_ = std::move(candidate);
        return true;
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
            if (!candidate.insertInner(i, suffix, sorted_entries[i].right_child, hint)) return false;
        }

        page_ = std::move(candidate);
        return true;
    }

    std::size_t InnerNode::childIndexForKey(std::string_view key) const {
        std::size_t lo = 0;
        std::size_t hi = page_.slotCount();

        const std::string_view prefix = page_.prefixView();
        
        bool can_use_hint = false;
        if (key.size() >= prefix.size() && prefix.size() > 0) {
            if (std::char_traits<char>::compare(key.data(), prefix.data(), prefix.size()) == 0) {
                can_use_hint = true;
            }
        } else if (prefix.empty()) {
            can_use_hint = true;
        }

        std::uint16_t target_hint = 0;
        if (can_use_hint) {
            target_hint = make_prefix_hint(key.substr(prefix.size()));
        }

        while (lo < hi) {
            const std::size_t mid = lo + ((hi - lo) / 2);
            int cmp = 0;
            if (can_use_hint) {
                const std::uint16_t slot_hint = page_.hintAt(static_cast<std::uint16_t>(mid));
                if (target_hint < slot_hint) cmp = 1;
                else if (target_hint > slot_hint) cmp = -1;
                else cmp = compareKeyAt(static_cast<std::uint16_t>(mid), key);
            } else {
                cmp = compareKeyAt(static_cast<std::uint16_t>(mid), key);
            }
            if (cmp <= 0) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    NodeId InnerNode::childAt(std::size_t child_index) const {
        if (child_index == 0) return page_.leftChild();
        return page_.rightChild(static_cast<std::uint16_t>(child_index - 1));
    }

    NodeId InnerNode::childForKey(std::string_view key) const { return childAt(childIndexForKey(key)); }
    std::string_view InnerNode::prefixView() const { return page_.prefixView(); }

    int InnerNode::compareKeyAt(std::uint16_t slot_index, std::string_view key) const {
        const std::string_view prefix_view = page_.prefixView();
        const std::size_t prefix_common = std::min(prefix_view.size(), key.size());
        
        if (prefix_common > 0) {
            const int prefix_cmp = std::char_traits<char>::compare(prefix_view.data(), key.data(), prefix_common);
            if (prefix_cmp < 0) return -1;
            if (prefix_cmp > 0) return 1;
        }
        if (key.size() < prefix_view.size()) return 1;

        const std::string_view suffix = page_.keySuffix(slot_index);
        const std::string_view key_suffix = key.substr(prefix_view.size());
        return lexical_compare(suffix, key_suffix);
    }

    std::string InnerNode::commonPrefix(const std::vector<InnerEntry> &sorted_entries)
    {
        if (sorted_entries.empty()) return "";
        std::string prefix = sorted_entries.front().key;
        for (std::size_t i = 1; i < sorted_entries.size() && !prefix.empty(); ++i)
        {
            const std::string &key = sorted_entries[i].key;
            std::size_t len = 0;
            const std::size_t max_len = std::min(prefix.size(), key.size());
            while (len < max_len && prefix[len] == key[len]) ++len;
            prefix.resize(len);
        }
        return prefix;
    }

    bool InnerNode::tryInsertInPlace(std::string_view key, NodeId right_child) {
        std::string_view prefix = page_.prefixView();
        if (key.size() >= prefix.size()) {
            bool match = true;
            if (prefix.size() > 0) {
                match = (std::char_traits<char>::compare(key.data(), prefix.data(), prefix.size()) == 0);
            }
            if (match) {
                std::string_view suffix = key.substr(prefix.size());
                std::uint16_t hint = make_prefix_hint(suffix);
                if (page_.insertInner(static_cast<std::uint16_t>(childIndexForKey(key)), suffix, right_child, hint)) return true;
            }
        }
        return false;
    }

} // namespace abt