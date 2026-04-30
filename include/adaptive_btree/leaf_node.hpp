#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "adaptive_btree/node.hpp"

namespace abt
{

    struct LeafEntry
    {
        std::string key;
        Value value;
    };

    struct LeafEntryView {
        std::string_view key; // Suffix if from page, full key if new
        Value value;
        bool is_new;
    };

    class LeafNode final : public Node
    {
    public:
        explicit LeafNode(NodeId id);

        std::vector<LeafEntry> entries() const;
        bool rebuild(const std::vector<LeafEntry> &sorted_entries);

        std::vector<LeafEntryView> entryViews() const;
        bool rebuildFromViews(const std::vector<LeafEntryView> &sorted_entries, std::string_view new_key);

        std::optional<Value> find(std::string_view key) const;
        std::size_t lowerBoundIndex(std::string_view key) const;

        NodeId nextLeaf() const;
        void setNextLeaf(NodeId id);

        bool tryInsertInPlace(std::string_view key, Value value, bool& inserted_new);
        std::uint16_t slotCount() const;
        std::string keyAt(std::uint16_t index) const;
        Value valueAt(std::uint16_t index) const;
        std::string_view prefixView() const;

    private:
        int compareKeyAt(std::uint16_t slot_index, std::string_view key) const;
        static std::string commonPrefix(const std::vector<LeafEntry> &sorted_entries);
        static std::string commonPrefixFromViews(const std::vector<LeafEntryView> &sorted_entries, std::string_view new_key, std::string_view old_prefix);
    };

} // namespace abt