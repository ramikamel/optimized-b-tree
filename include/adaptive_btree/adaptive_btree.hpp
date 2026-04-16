#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "adaptive_btree/config.hpp"
#include "adaptive_btree/inner_node.hpp"
#include "adaptive_btree/leaf_node.hpp"

namespace abt
{

    class AdaptiveBTree
    {
    public:
        explicit AdaptiveBTree(FeatureFlags features = {});

        bool insert(const std::string &key, Value value);
        std::optional<Value> search(std::string_view key) const;
        std::vector<KeyValue> rangeScan(std::string_view start_key, std::size_t max_results) const;

        std::size_t size() const;
        std::size_t height() const;

    private:
        struct SplitResult
        {
            std::string separator_key;
            NodeId right_node;
        };

        NodeId allocateLeaf();
        NodeId allocateInner();

        Node *getNode(NodeId id);
        const Node *getNode(NodeId id) const;

        LeafNode *getLeaf(NodeId id);
        const LeafNode *getLeaf(NodeId id) const;

        InnerNode *getInner(NodeId id);
        const InnerNode *getInner(NodeId id) const;

        std::optional<SplitResult> insertRecursive(NodeId node_id,
                                                   const std::string &key,
                                                   Value value,
                                                   bool *inserted_new);

        NodeId findLeafForKey(std::string_view key) const;

        static std::size_t chooseLeafSplitIndex(const std::vector<LeafEntry> &entries);
        static std::size_t chooseInnerSplitIndex(const std::vector<InnerEntry> &entries);

        static std::size_t estimateLeafBytes(const std::vector<LeafEntry> &entries,
                                             std::size_t begin,
                                             std::size_t end);

        static std::size_t estimateInnerBytes(const std::vector<InnerEntry> &entries,
                                              std::size_t begin,
                                              std::size_t end);

        FeatureFlags features_;
        std::unordered_map<NodeId, std::unique_ptr<Node>> nodes_;

        NodeId root_id_ = 0;
        NodeId next_node_id_ = 1;

        std::size_t size_ = 0;
        std::size_t height_ = 1;
    };

} // namespace abt
