#include "adaptive_btree/adaptive_btree.hpp"

#include <stdexcept>
#include <utility>

namespace abt
{

    namespace
    {
        // Fixed-size path stack avoids std::vector bookkeeping in the insert hot
        // path. A 4 KiB page B-tree of 64-bit values can never reach depth 32.
        constexpr std::size_t kMaxTreeDepth = 32;
    }

    AdaptiveBTree::AdaptiveBTree(FeatureFlags features) : features_(features)
    {
        nodes_.reserve(1 << 12);
        root_id_ = allocateLeaf();
        // Root leaf has no fences; tryInsert always succeeds until it splits.
    }

    NodeId AdaptiveBTree::allocateLeaf()
    {
        const NodeId id = next_node_id_++;
        if (id >= nodes_.size()) nodes_.resize(static_cast<std::size_t>(id + 1) * 2);
        nodes_[id] = std::make_unique<LeafNode>(id);
        return id;
    }

    NodeId AdaptiveBTree::allocateInner()
    {
        const NodeId id = next_node_id_++;
        if (id >= nodes_.size()) nodes_.resize(static_cast<std::size_t>(id + 1) * 2);
        nodes_[id] = std::make_unique<InnerNode>(id);
        return id;
    }

    NodeId AdaptiveBTree::findLeafForKey(std::string_view key) const
    {
        NodeId cur = root_id_;
        while (!getNode(cur)->isLeaf())
        {
            cur = getInner(cur)->childForKey(key);
        }
        return cur;
    }

    std::optional<Value> AdaptiveBTree::search(std::string_view key) const
    {
        const NodeId leaf_id = findLeafForKey(key);
        return getLeaf(leaf_id)->find(key);
    }

    std::vector<KeyValue> AdaptiveBTree::rangeScan(std::string_view start_key, std::size_t max_results) const
    {
        std::vector<KeyValue> results;
        if (max_results == 0) return results;
        results.reserve(max_results);

        NodeId leaf_id = findLeafForKey(start_key);
        bool first = true;

        while (results.size() < max_results)
        {
            const LeafNode* leaf = getLeaf(leaf_id);
            const std::uint16_t count = leaf->slotCount();
            std::uint16_t i = 0;
            if (first)
            {
                i = leaf->lowerBoundIndex(start_key);
                first = false;
            }
            for (; i < count && results.size() < max_results; ++i)
            {
                results.push_back(KeyValue{leaf->keyAt(i), leaf->valueAt(i)});
            }
            const NodeId next_id = leaf->nextLeaf();
            if (next_id == 0) break;
            leaf_id = next_id;
        }
        return results;
    }

    bool AdaptiveBTree::insert(std::string_view key, Value value)
    {
        // Iterative descent recording the path on a fixed-size stack.
        InnerNode* path[kMaxTreeDepth];
        std::size_t depth = 0;

        Node* cur_node = getNode(root_id_);
        while (!cur_node->isLeaf())
        {
            InnerNode* inner = static_cast<InnerNode*>(cur_node);
            const std::uint16_t i = inner->childIndexForKey(key);
            path[depth++] = inner;
            cur_node = getNode(inner->childAt(i));
        }

        LeafNode* leaf = static_cast<LeafNode*>(cur_node);
        bool inserted_new = false;
        if (__builtin_expect(leaf->tryInsert(key, value, inserted_new), 1))
        {
            if (inserted_new) ++size_;
            return inserted_new;
        }

        // Leaf full -> split, insert into the appropriate half, bubble up.
        const NodeId right_leaf_id = allocateLeaf();
        LeafNode* right_leaf = getLeaf(right_leaf_id);
        std::string sep_key = leaf->splitInto(*right_leaf, right_leaf_id);

        const bool to_right = std::string_view(key) >= std::string_view(sep_key);
        bool ok = to_right
            ? right_leaf->tryInsert(key, value, inserted_new)
            : leaf->tryInsert(key, value, inserted_new);
        if (!ok)
        {
            throw std::runtime_error("entry too large to fit in a fresh page after split");
        }

        NodeId promoted_right = right_leaf_id;
        while (depth > 0)
        {
            InnerNode* parent = path[--depth];
            if (parent->tryInsertSeparator(sep_key, promoted_right))
            {
                if (inserted_new) ++size_;
                return inserted_new;
            }

            const NodeId right_inner_id = allocateInner();
            InnerNode* right_inner = getInner(right_inner_id);
            std::string new_promoted = parent->splitInto(*right_inner);

            const bool sep_to_right = std::string_view(sep_key) >= std::string_view(new_promoted);
            bool ok2 = sep_to_right
                ? right_inner->tryInsertSeparator(sep_key, promoted_right)
                : parent->tryInsertSeparator(sep_key, promoted_right);
            if (!ok2)
            {
                throw std::runtime_error("could not insert separator after inner split");
            }

            sep_key = std::move(new_promoted);
            promoted_right = right_inner_id;
        }

        const NodeId old_root = root_id_;
        const NodeId new_root_id = allocateInner();
        getInner(new_root_id)->initRoot(old_root, sep_key, promoted_right);
        root_id_ = new_root_id;
        ++height_;

        if (inserted_new) ++size_;
        return inserted_new;
    }

} // namespace abt
