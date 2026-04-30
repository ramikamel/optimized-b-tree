#pragma once

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/slotted_page.hpp"

namespace abt
{

    // Thin wrapper carrying a stable NodeId plus a 4 KiB page.
    class Node
    {
    public:
        Node(NodeId id, NodeType type) : id_(id) { page_.init(type, {}, {}); }

        NodeId id() const { return id_; }
        NodeType type() const { return page_.type(); }
        bool isLeaf() const { return type() == NodeType::kLeaf; }

        SlottedPage& page() { return page_; }
        const SlottedPage& page() const { return page_; }

    protected:
        NodeId id_;
        SlottedPage page_;
    };

} // namespace abt
