#pragma once

#include <string>
#include <string_view>

#include "adaptive_btree/node.hpp"

namespace abt
{

    class InnerNode final : public Node
    {
    public:
        explicit InnerNode(NodeId id) : Node(id, NodeType::kInner) {}

        // Reset to empty with the given fences. Caller must then setLeftChild()
        // and append separators/right_children.
        void initEmpty(std::string_view lower_fence, std::string_view upper_fence, NodeId left_child);

        // Initialize this node as a fresh root holding (left_child, sep, right_child).
        void initRoot(NodeId left_child, std::string_view separator_full_key, NodeId right_child);

        // Try to insert a separator + right child without splitting.
        bool tryInsertSeparator(std::string_view separator_full_key, NodeId right_child);

        // Routing: returns the index in [0, slotCount] that picks the correct
        // child for `key` (0 -> leftChild, k+1 -> rightChild(k)).
        std::uint16_t childIndexForKey(std::string_view key) const;
        NodeId childAt(std::uint16_t child_index) const;
        NodeId childForKey(std::string_view key) const { return childAt(childIndexForKey(key)); }

        std::uint16_t slotCount() const { return page_.slotCount(); }
        std::string_view keySuffix(std::uint16_t i) const { return page_.keySuffix(i); }
        std::string_view prefixView() const { return page_.prefixView(); }
        NodeId rightChild(std::uint16_t i) const { return page_.rightChild(i); }
        NodeId leftChild() const { return page_.link(); }
        void setLeftChild(NodeId id) { page_.setLink(id); }

        // Split this inner node and promote the median key (paper §2):
        //   - left keeps slots [0, mid) and its current leftChild
        //   - right (freshly allocated) gets leftChild = src.rightChild(mid),
        //     and slots [mid+1, n)
        //   - the median full key is returned and disappears from both children
        std::string splitInto(InnerNode& right_node);
    };

} // namespace abt
