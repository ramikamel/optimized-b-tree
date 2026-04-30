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

    class LeafNode final : public Node
    {
    public:
        explicit LeafNode(NodeId id);

        std::vector<LeafEntry> entries() const;
        bool rebuild(const std::vector<LeafEntry> &sorted_entries);

        std::optional<Value> find(std::string_view key) const;
        std::size_t lowerBoundIndex(std::string_view key) const;

        NodeId nextLeaf() const;
        void setNextLeaf(NodeId id);

        bool tryInsertInPlace(std::string_view key, Value value, bool& inserted_new);
        std::uint16_t slotCount() const;
        std::string keyAt(std::uint16_t index) const;
        Value valueAt(std::uint16_t index) const;

    private:
        int compareKeyAt(std::uint16_t slot_index, std::string_view key) const;
        static std::string commonPrefix(const std::vector<LeafEntry> &sorted_entries);
    };

} // namespace abt
