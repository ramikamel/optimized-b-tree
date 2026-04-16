#include "adaptive_btree/node.hpp"

namespace abt
{

    Node::Node(NodeId id, NodeType type) : id_(id), page_(type) {}

    NodeId Node::id() const
    {
        return id_;
    }

    NodeType Node::type() const
    {
        return page_.type();
    }

    bool Node::isLeaf() const
    {
        return type() == NodeType::kLeaf;
    }

    const SlottedPage &Node::page() const
    {
        return page_;
    }

    SlottedPage &Node::page()
    {
        return page_;
    }

} // namespace abt
