#include "adaptive_btree/adaptive_btree.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace abt
{

    AdaptiveBTree::AdaptiveBTree(FeatureFlags features) : features_(features)
    {
        nodes_.resize(100000); 
        root_id_ = allocateLeaf();
    }

    bool AdaptiveBTree::insert(const std::string &key, Value value)
    {
        bool inserted_new = false;
        const std::optional<SplitResult> split = insertRecursive(root_id_, key, value, &inserted_new);

        if (split.has_value())
        {
            const NodeId old_root = root_id_;
            const NodeId new_root = allocateInner();
            InnerNode *root = getInner(new_root);

            std::vector<InnerEntry> entries;
            entries.push_back(InnerEntry{split->separator_key, split->right_node});

            if (!root->rebuild(old_root, entries))
            {
                throw std::runtime_error("new root does not fit in one page");
            }

            root_id_ = new_root;
            ++height_;
        }

        if (inserted_new) ++size_;
        return inserted_new;
    }

    std::optional<Value> AdaptiveBTree::search(std::string_view key) const
    {
        const NodeId leaf_id = findLeafForKey(key);
        const LeafNode *leaf = getLeaf(leaf_id);
        return leaf->find(key);
    }

    std::vector<KeyValue> AdaptiveBTree::rangeScan(std::string_view start_key, std::size_t max_results) const {
        std::vector<KeyValue> results;
        if (max_results == 0) return results;

        NodeId leaf_id = findLeafForKey(start_key);
        bool first_leaf = true;

        while (leaf_id != 0 && results.size() < max_results) {
            const LeafNode *leaf = getLeaf(leaf_id);
            const std::uint16_t count = leaf->slotCount(); 

            std::uint16_t i = 0;
            if (first_leaf) {
                i = static_cast<std::uint16_t>(leaf->lowerBoundIndex(start_key));
                first_leaf = false;
            }

            for (; i < count; ++i) {
                results.push_back(KeyValue{leaf->keyAt(i), leaf->valueAt(i)});
                if (results.size() >= max_results) break;
            }
            leaf_id = leaf->nextLeaf();
        }

        return results;
    }

    std::size_t AdaptiveBTree::size() const { return size_; }
    std::size_t AdaptiveBTree::height() const { return height_; }

    NodeId AdaptiveBTree::allocateLeaf()
    {
        const NodeId id = next_node_id_++;
        if (id >= nodes_.size()) nodes_.resize(id * 2); 
        nodes_[id] = std::make_unique<LeafNode>(id);
        return id;
    }

    NodeId AdaptiveBTree::allocateInner()
    {
        const NodeId id = next_node_id_++;
        if (id >= nodes_.size()) nodes_.resize(id * 2);
        nodes_[id] = std::make_unique<InnerNode>(id);
        return id;
    }

    Node *AdaptiveBTree::getNode(NodeId id) { return nodes_[id].get(); }
    const Node *AdaptiveBTree::getNode(NodeId id) const { return nodes_[id].get(); }
    LeafNode *AdaptiveBTree::getLeaf(NodeId id) { return static_cast<LeafNode *>(getNode(id)); }
    const LeafNode *AdaptiveBTree::getLeaf(NodeId id) const { return static_cast<const LeafNode *>(getNode(id)); }
    InnerNode *AdaptiveBTree::getInner(NodeId id) { return static_cast<InnerNode *>(getNode(id)); }
    const InnerNode *AdaptiveBTree::getInner(NodeId id) const { return static_cast<const InnerNode *>(getNode(id)); }

    std::optional<AdaptiveBTree::SplitResult> AdaptiveBTree::insertRecursive(NodeId node_id,
                                                                             const std::string &key,
                                                                             Value value,
                                                                             bool *inserted_new)
    {
        Node *node = getNode(node_id);

        if (node->isLeaf())
        {
            LeafNode *leaf = static_cast<LeafNode *>(node);

            if (leaf->tryInsertInPlace(key, value, *inserted_new)) {
                return std::nullopt;
            }

            std::vector<LeafEntryView> views = leaf->entryViews();
            std::string_view old_prefix = leaf->prefixView();

            auto it = std::lower_bound(views.begin(), views.end(), key,
                [&](const LeafEntryView &lhs, const std::string &rhs) {
                    std::string_view rhs_view(rhs);
                    std::size_t common_len = std::min(old_prefix.size(), rhs_view.size());
                    
                    if (common_len > 0) {
                        int cmp = std::char_traits<char>::compare(old_prefix.data(), rhs_view.data(), common_len);
                        if (cmp != 0) return cmp < 0;
                    }
                    if (rhs_view.size() < old_prefix.size()) return false;
                    
                    std::string_view lhs_suffix = lhs.key;
                    std::string_view rhs_suffix = rhs_view.substr(old_prefix.size());
                    return lexical_compare(lhs_suffix, rhs_suffix) < 0;
                });

            bool found = false;
            if (it != views.end()) {
                std::string full_lhs;
                if (it->is_new) full_lhs = key;
                else {
                    full_lhs = std::string(old_prefix);
                    full_lhs.append(it->key);
                }
                if (full_lhs == key) {
                    it->value = value;
                    *inserted_new = false;
                    found = true;
                }
            }
            
            if (!found) {
                views.insert(it, LeafEntryView{key, value, true});
                *inserted_new = true;
            }

            if (leaf->rebuildFromViews(views, key)) {
                return std::nullopt;
            }

            std::size_t total_bytes = 0;
            for (const auto& e : views) {
                total_bytes += e.is_new ? key.size() : old_prefix.size() + e.key.size();
            }
            
            std::size_t current_bytes = 0;
            std::size_t split_index = views.size() / 2;
            for (std::size_t i = 0; i < views.size(); ++i) {
                current_bytes += views[i].is_new ? key.size() : old_prefix.size() + views[i].key.size();
                if (current_bytes >= total_bytes / 2) {
                    split_index = std::max<std::size_t>(1, i);
                    break;
                }
            }

            const std::vector<LeafEntryView> left(views.begin(), views.begin() + split_index);
            const std::vector<LeafEntryView> right(views.begin() + split_index, views.end());

            const NodeId right_id = allocateLeaf();
            LeafNode *right_leaf = getLeaf(right_id);
            const NodeId old_next = leaf->nextLeaf();

            leaf->rebuildFromViews(left, key);
            right_leaf->rebuildFromViews(right, key);

            leaf->setNextLeaf(right_id);
            right_leaf->setNextLeaf(old_next);

            std::string sep_key;
            if (right.front().is_new) {
                sep_key = key;
            } else {
                sep_key = std::string(old_prefix);
                sep_key.append(right.front().key);
            }
            return SplitResult{sep_key, right_id};
        }

        InnerNode *inner = static_cast<InnerNode *>(node);
        const std::size_t child_index = inner->childIndexForKey(key);
        const NodeId child_id = inner->childAt(child_index);

        const std::optional<SplitResult> child_split = insertRecursive(child_id, key, value, inserted_new);
        if (!child_split.has_value()) {
            return std::nullopt;
        }

        if (inner->tryInsertInPlace(child_split->separator_key, child_split->right_node)) {
            return std::nullopt;
        }

        InnerMaterializedView state = inner->materializeViews();
        std::string_view old_prefix = inner->prefixView();

        state.entries.insert(state.entries.begin() + child_index,
                             InnerEntryView{child_split->separator_key, child_split->right_node, true});

        if (inner->rebuildFromViews(state.left_child, state.entries, child_split->separator_key)) {
            return std::nullopt;
        }

        std::size_t total_bytes = 0;
        for (const auto& e : state.entries) {
            total_bytes += e.is_new ? child_split->separator_key.size() : old_prefix.size() + e.key.size();
        }
        
        std::size_t current_bytes = 0;
        std::size_t split_index = state.entries.size() / 2;
        for (std::size_t i = 0; i < state.entries.size(); ++i) {
            current_bytes += state.entries[i].is_new ? child_split->separator_key.size() : old_prefix.size() + state.entries[i].key.size();
            if (current_bytes >= total_bytes / 2) {
                split_index = std::max<std::size_t>(0, i);
                break;
            }
        }

        const InnerEntryView promoted = state.entries[split_index];
        const std::vector<InnerEntryView> left(state.entries.begin(), state.entries.begin() + split_index);
        const NodeId right_left_child = promoted.right_child;
        const std::vector<InnerEntryView> right(state.entries.begin() + split_index + 1, state.entries.end());

        const NodeId right_id = allocateInner();
        InnerNode *right_inner = getInner(right_id);

        inner->rebuildFromViews(state.left_child, left, child_split->separator_key);
        right_inner->rebuildFromViews(right_left_child, right, child_split->separator_key);

        std::string sep_key;
        if (promoted.is_new) {
            sep_key = child_split->separator_key;
        } else {
            sep_key = std::string(old_prefix);
            sep_key.append(promoted.key);
        }
        return SplitResult{sep_key, right_id};
    }

    NodeId AdaptiveBTree::findLeafForKey(std::string_view key) const
    {
        NodeId current = root_id_;
        while (!getNode(current)->isLeaf())
        {
            const InnerNode *inner = getInner(current);
            current = inner->childForKey(key);
        }
        return current;
    }

} // namespace abt