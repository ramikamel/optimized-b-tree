#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "adaptive_btree/node.hpp"

namespace abt
{

    class LeafNode final : public Node
    {
    public:
        explicit LeafNode(NodeId id) : Node(id, NodeType::kLeaf) {}

        // Reset to empty with the given fences and forward link.
        void initEmpty(std::string_view lower_fence, std::string_view upper_fence, NodeId next_leaf);

        // Try to insert (key, value) without splitting. Returns false only if
        // the page is out of space. Sets *inserted_new = false on upsert.
        bool tryInsert(std::string_view key, Value value, bool& inserted_new);

        std::optional<Value> find(std::string_view key) const;

        // Lower-bound on full key, used for range scans.
        std::uint16_t lowerBoundIndex(std::string_view key) const;

        NodeId nextLeaf() const { return page_.link(); }
        void setNextLeaf(NodeId id) { page_.setLink(id); }

        std::uint16_t slotCount() const { return page_.slotCount(); }
        std::string_view keySuffix(std::uint16_t i) const { return page_.keySuffix(i); }
        std::string_view prefixView() const { return page_.prefixView(); }
        Value valueAt(std::uint16_t i) const { return page_.leafValue(i); }
        std::string keyAt(std::uint16_t i) const;

        // Split this leaf in half. The right-half entries are written to
        // `right_node` (which must be freshly allocated). Returns the truncated
        // separator that should be promoted to the parent.
        std::string splitInto(LeafNode& right_node, NodeId right_id);
    };

} // namespace abt
