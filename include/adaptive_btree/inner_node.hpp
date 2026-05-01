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

        // Erase the separator at slot `pos`. The slot directory shifts down;
        // the heap is not compacted (rebuilt on the next merge).
        void eraseSeparatorAt(std::uint16_t pos);

        // Merge `right` into `*this`, with `promoted_sep` as the parent's
        // separator that previously sat between us. After this call, `right`
        // is logically empty. Returns false if the union does not fit.
        bool tryMergeFrom(InnerNode& right, std::string_view promoted_sep);

        // True iff this inner node is below the merge underflow threshold.
        bool isUnderflow() const;

        // Routing: returns the index in [0, slotCount] that picks the correct
        // child for `key` (0 -> leftChild, k+1 -> rightChild(k)).
        std::uint16_t childIndexForKey(std::string_view key) const;
        NodeId childAt(std::uint16_t child_index) const;
        NodeId childForKey(std::string_view key) const { return childAt(childIndexForKey(key)); }
        std::string_view lowerFenceView() const { return page_.lowerFenceView(); }
        std::string_view upperFenceView() const { return page_.upperFenceView(); }

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
