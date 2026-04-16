#pragma once

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/slotted_page.hpp"

namespace abt
{

    class Node
    {
    public:
        Node(NodeId id, NodeType type);
        virtual ~Node() = default;

        NodeId id() const;
        NodeType type() const;
        bool isLeaf() const;

        const SlottedPage &page() const;
        SlottedPage &page();

    protected:
        NodeId id_;
        SlottedPage page_;
    };

} // namespace abt
