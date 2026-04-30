#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "adaptive_btree/node.hpp"

namespace abt
{

    struct InnerEntry
    {
        std::string key;
        NodeId right_child;
    };

    struct InnerEntryView {
        std::string_view key;
        NodeId right_child;
        bool is_new;
    };

    struct InnerMaterialized
    {
        NodeId left_child;
        std::vector<InnerEntry> entries;
    };

    struct InnerMaterializedView
    {
        NodeId left_child;
        std::vector<InnerEntryView> entries;
    };

    class InnerNode final : public Node
    {
    public:
        explicit InnerNode(NodeId id);

        InnerMaterialized materialize() const;
        bool rebuild(NodeId left_child, const std::vector<InnerEntry> &sorted_entries);

        InnerMaterializedView materializeViews() const;
        bool rebuildFromViews(NodeId left_child, const std::vector<InnerEntryView> &sorted_entries, std::string_view new_key);

        std::size_t childIndexForKey(std::string_view key) const;
        NodeId childAt(std::size_t child_index) const;
        NodeId childForKey(std::string_view key) const;
        std::string_view prefixView() const;

        bool tryInsertInPlace(std::string_view key, NodeId right_child);

    private:
        int compareKeyAt(std::uint16_t slot_index, std::string_view key) const;
        static std::string commonPrefix(const std::vector<InnerEntry> &sorted_entries);
        static std::string commonPrefixFromViews(const std::vector<InnerEntryView> &sorted_entries, std::string_view new_key, std::string_view old_prefix);
    };

} // namespace abt