#include "adaptive_btree/adaptive_btree.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "adaptive_btree/fdl_layout.hpp"
#include "adaptive_btree/sdl_layout.hpp"

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
            const std::string_view sk = first ? start_key : std::string_view{};
            first = false;
            leaf->collectScan(sk, max_results, results);
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

        // An empty FDL/SDL whose shape doesn't match the inserting key (e.g.,
        // every entry was previously erased and the key now arrives below the
        // dense leaf's base or with a different ref_key shape) cannot be split
        // by the median-style demote path. Demote the empty leaf to a fresh
        // comparison leaf in place and retry; no allocation, no bubble-up.
        if (leaf->kind() != LeafKind::kComparison && leaf->entryCount() == 0)
        {
            std::string lower(leaf->lowerFenceView());
            std::string upper(leaf->upperFenceView());
            const NodeId next_link = leaf->nextLeaf();
            leaf->initEmpty(lower, upper, next_link);
            if (leaf->tryInsert(key, value, inserted_new))
            {
                if (inserted_new) ++size_;
                return inserted_new;
            }
        }

        // Leaf full -> split, insert into the appropriate half, bubble up.
        // Note: nodes_ is std::vector<std::unique_ptr<Node>>; resizing the
        // vector moves the unique_ptrs but the heap-allocated Node addresses
        // are stable, so `leaf` and `path[]` pointers survive allocateLeaf().
        const NodeId right_leaf_id = allocateLeaf();
        LeafNode* right_leaf = getLeaf(right_leaf_id);
        // Pass the inserting key so FDL/SDL splits can decide partition vs.
        // demote based on whether the key falls in or beyond the dense range.
        LeafSplitResult split = leaf->splitInto(*right_leaf, right_leaf_id, key);
        if (split.separator.empty())
        {
            throw std::runtime_error("split failed: leaf could not be split");
        }
        std::string sep_key = std::move(split.separator);

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

    bool AdaptiveBTree::erase(std::string_view key)
    {
        // Descent: record (parent, child_index) frames for bubble-up.
        struct PathFrame { InnerNode* parent; std::uint16_t child_idx; };
        PathFrame path[kMaxTreeDepth];
        std::size_t depth = 0;

        Node* cur_node = getNode(root_id_);
        while (!cur_node->isLeaf())
        {
            InnerNode* inner = static_cast<InnerNode*>(cur_node);
            const std::uint16_t i = inner->childIndexForKey(key);
            path[depth++] = {inner, i};
            cur_node = getNode(inner->childAt(i));
        }

        LeafNode* leaf = static_cast<LeafNode*>(cur_node);
        if (!leaf->tryErase(key)) return false;
        --size_;

        if (depth == 0 || !leaf->isUnderflow()) return true;

        // Step 1: try to merge the underflow leaf with a sibling under its
        // immediate parent. Pick right sibling when available, else left.
        PathFrame frame = path[depth - 1];
        InnerNode* parent = frame.parent;
        const std::uint16_t my_idx = frame.child_idx;

        std::uint16_t merge_left;
        if (my_idx < parent->slotCount())
            merge_left = my_idx;          // merge me + right sibling
        else if (my_idx > 0)
            merge_left = static_cast<std::uint16_t>(my_idx - 1); // merge left + me
        else
            return true; // single child under parent; cannot merge here

        const std::uint16_t sep_pos = merge_left;
        const NodeId left_id  = parent->childAt(merge_left);
        const NodeId right_id = parent->childAt(static_cast<std::uint16_t>(merge_left + 1));
        LeafNode* left_leaf  = getLeaf(left_id);
        LeafNode* right_leaf = getLeaf(right_id);

        if (!left_leaf->tryMergeFrom(*right_leaf)) return true;
        parent->eraseSeparatorAt(sep_pos);
        nodes_[right_id].reset();
        --depth;

        // Step 2: bubble up inner-node merges while the current `parent` is
        // underflow and a sibling is mergeable.
        while (depth > 0 && parent->isUnderflow())
        {
            PathFrame gframe = path[depth - 1];
            InnerNode* grand = gframe.parent;
            const std::uint16_t pidx = gframe.child_idx;

            std::uint16_t imerge;
            if (pidx < grand->slotCount())
                imerge = pidx;
            else if (pidx > 0)
                imerge = static_cast<std::uint16_t>(pidx - 1);
            else
                break;

            const std::uint16_t isep = imerge;
            const NodeId il = grand->childAt(imerge);
            const NodeId ir = grand->childAt(static_cast<std::uint16_t>(imerge + 1));
            InnerNode* lnode = getInner(il);
            InnerNode* rnode = getInner(ir);

            std::string promoted_sep;
            {
                const std::string_view gpfx = grand->prefixView();
                const std::string_view gsuf = grand->keySuffix(isep);
                promoted_sep.reserve(gpfx.size() + gsuf.size());
                promoted_sep.append(gpfx);
                promoted_sep.append(gsuf);
            }
            if (!lnode->tryMergeFrom(*rnode, promoted_sep)) break;
            grand->eraseSeparatorAt(isep);
            nodes_[ir].reset();
            parent = lnode;
            --depth;
        }

        // Step 3: collapse the root while it is an inner with zero separators.
        Node* root = getNode(root_id_);
        while (!root->isLeaf())
        {
            InnerNode* rinner = static_cast<InnerNode*>(root);
            if (rinner->slotCount() > 0) break;
            const NodeId only_child = rinner->leftChild();
            nodes_[root_id_].reset();
            root_id_ = only_child;
            if (height_ > 1) --height_;
            root = getNode(root_id_);
        }
        return true;
    }

    AdaptiveBTree::LayoutStats AdaptiveBTree::layoutStats() const
    {
        LayoutStats s{};
        // Walk the leaf chain via the leftmost descent.
        if (nodes_.empty()) return s;
        NodeId cur = root_id_;
        while (cur < nodes_.size() && nodes_[cur] && !nodes_[cur]->isLeaf())
        {
            cur = static_cast<const InnerNode*>(nodes_[cur].get())->leftChild();
        }
        while (cur != 0 || (cur < nodes_.size() && nodes_[cur]))
        {
            if (cur >= nodes_.size() || !nodes_[cur]) break;
            const LeafNode* leaf = static_cast<const LeafNode*>(nodes_[cur].get());
            const std::uint16_t entries = leaf->entryCount();
            s.total_entries += entries;
            switch (leaf->kind())
            {
                case LeafKind::kComparison: ++s.n_comparison; break;
                case LeafKind::kFullyDense:
                {
                    ++s.n_fdl;
                    const FdlMeta& meta = fdl_meta(leaf->page().rawBytes());
                    s.total_dense_capacity += meta.capacity;
                    break;
                }
                case LeafKind::kSemiDense:
                {
                    ++s.n_sdl;
                    const SdlMeta& meta = sdl_meta(leaf->page().rawBytes());
                    s.total_dense_capacity += meta.capacity;
                    break;
                }
            }
            const NodeId nxt = leaf->nextLeaf();
            if (nxt == 0) break;
            cur = nxt;
        }
        return s;
    }

} // namespace abt
