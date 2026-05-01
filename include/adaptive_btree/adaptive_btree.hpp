#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/inner_node.hpp"
#include "adaptive_btree/leaf_node.hpp"
#include "adaptive_btree/node.hpp"

namespace abt
{

    class AdaptiveBTree
    {
    public:
        explicit AdaptiveBTree(FeatureFlags features = FeatureFlags{});

        bool insert(std::string_view key, Value value);
        bool insert(const std::string& key, Value value) { return insert(std::string_view(key), value); }

        // Returns true iff a key was removed. Triggers leaf merge + bubble-up
        // inner-merge when the touched leaf falls below the underflow
        // threshold; layout is re-evaluated adaptively after each merge.
        bool erase(std::string_view key);
        bool erase(const std::string& key) { return erase(std::string_view(key)); }

        std::optional<Value> search(std::string_view key) const;
        std::vector<KeyValue> rangeScan(std::string_view start_key, std::size_t max_results) const;

        std::size_t size() const { return size_; }
        std::size_t height() const { return height_; }

        // Per-leaf-kind statistics (Phase 7 observability).
        struct LayoutStats
        {
            std::size_t n_comparison = 0;
            std::size_t n_fdl = 0;
            std::size_t n_sdl = 0;
            std::size_t total_entries = 0;
            std::size_t total_dense_capacity = 0;
        };
        LayoutStats layoutStats() const;

    private:
        NodeId allocateLeaf();
        NodeId allocateInner();

        Node* getNode(NodeId id) { return nodes_[id].get(); }
        const Node* getNode(NodeId id) const { return nodes_[id].get(); }
        LeafNode* getLeaf(NodeId id) { return static_cast<LeafNode*>(getNode(id)); }
        const LeafNode* getLeaf(NodeId id) const { return static_cast<const LeafNode*>(getNode(id)); }
        InnerNode* getInner(NodeId id) { return static_cast<InnerNode*>(getNode(id)); }
        const InnerNode* getInner(NodeId id) const { return static_cast<const InnerNode*>(getNode(id)); }

        NodeId findLeafForKey(std::string_view key) const;

        std::vector<std::unique_ptr<Node>> nodes_;
        NodeId root_id_{0};
        NodeId next_node_id_{0};

        std::size_t size_{0};
        std::size_t height_{1};
        FeatureFlags features_;
    };

} // namespace abt
