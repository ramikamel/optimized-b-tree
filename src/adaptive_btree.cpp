#include "adaptive_btree/adaptive_btree.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace abt
{
    namespace
    {

        template <typename EntryType>
        std::string commonPrefixInRange(const std::vector<EntryType> &entries,
                                        std::size_t begin,
                                        std::size_t end)
        {
            if (begin >= end)
            {
                return "";
            }

            std::string prefix = entries[begin].key;
            for (std::size_t i = begin + 1; i < end && !prefix.empty(); ++i)
            {
                const std::string &key = entries[i].key;
                std::size_t len = 0;
                const std::size_t max_len = std::min(prefix.size(), key.size());
                while (len < max_len && prefix[len] == key[len])
                {
                    ++len;
                }
                prefix.resize(len);
            }
            return prefix;
        }

    } // namespace

    AdaptiveBTree::AdaptiveBTree(FeatureFlags features) : features_(features)
    {
        nodes_.resize(100000); // Pre-allocate capacity to avoid constant reallocations
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

        if (inserted_new)
        {
            ++size_;
        }
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
            // Instantly jump to the target key on the first leaf using binary search
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

    std::size_t AdaptiveBTree::size() const
    {
        return size_;
    }

    std::size_t AdaptiveBTree::height() const
    {
        return height_;
    }

    NodeId AdaptiveBTree::allocateLeaf()
    {
        const NodeId id = next_node_id_++;
        if (id >= nodes_.size()) nodes_.resize(id * 2); // Resize vector if needed
        nodes_[id] = std::make_unique<LeafNode>(id);
        return id;
    }

    NodeId AdaptiveBTree::allocateInner()
    {
        const NodeId id = next_node_id_++;
        if (id >= nodes_.size()) nodes_.resize(id * 2); // Resize vector if needed
        nodes_[id] = std::make_unique<InnerNode>(id);
        return id;
    }

    Node *AdaptiveBTree::getNode(NodeId id)
    {
        if (id >= nodes_.size() || !nodes_[id])
        {
            throw std::out_of_range("node id not found");
        }
        return nodes_[id].get();
    }

    const Node *AdaptiveBTree::getNode(NodeId id) const
    {
        if (id >= nodes_.size() || !nodes_[id])
        {
            throw std::out_of_range("node id not found");
        }
        return nodes_[id].get();
    }

    LeafNode *AdaptiveBTree::getLeaf(NodeId id)
    {
        Node *node = getNode(id);
        if (!node->isLeaf())
        {
            throw std::logic_error("node is not a leaf");
        }
        return static_cast<LeafNode *>(node);
    }

    const LeafNode *AdaptiveBTree::getLeaf(NodeId id) const
    {
        const Node *node = getNode(id);
        if (!node->isLeaf())
        {
            throw std::logic_error("node is not a leaf");
        }
        return static_cast<const LeafNode *>(node);
    }

    InnerNode *AdaptiveBTree::getInner(NodeId id)
    {
        Node *node = getNode(id);
        if (node->isLeaf())
        {
            throw std::logic_error("node is not an inner node");
        }
        return static_cast<InnerNode *>(node);
    }

    const InnerNode *AdaptiveBTree::getInner(NodeId id) const
    {
        const Node *node = getNode(id);
        if (node->isLeaf())
        {
            throw std::logic_error("node is not an inner node");
        }
        return static_cast<const InnerNode *>(node);
    }

    std::optional<AdaptiveBTree::SplitResult> AdaptiveBTree::insertRecursive(NodeId node_id,
                                                                             const std::string &key,
                                                                             Value value,
                                                                             bool *inserted_new)
    {
        Node *node = getNode(node_id);

        if (node->isLeaf())
        {
            LeafNode *leaf = static_cast<LeafNode *>(node);

            // --- OPTIMIZATION: Try fast in-place insert first! ---
            if (leaf->tryInsertInPlace(key, value, *inserted_new))
            {
                return std::nullopt;
            }

            // --- FALLBACK: Rebuild and Split if page is full or prefix doesn't match ---
            std::vector<LeafEntry> entries = leaf->entries();

            auto it = std::lower_bound(entries.begin(), entries.end(), key, [](const LeafEntry &lhs, const std::string &rhs)
                                       { return lhs.key < rhs; });

            if (it != entries.end() && it->key == key)
            {
                it->value = value;
                *inserted_new = false;
            }
            else
            {
                entries.insert(it, LeafEntry{key, value});
                *inserted_new = true;
            }

            if (leaf->rebuild(entries))
            {
                return std::nullopt;
            }

            const std::size_t split_index = chooseLeafSplitIndex(entries);
            const std::vector<LeafEntry> left(entries.begin(), entries.begin() + split_index);
            const std::vector<LeafEntry> right(entries.begin() + split_index, entries.end());

            const NodeId right_id = allocateLeaf();
            LeafNode *right_leaf = getLeaf(right_id);

            const NodeId old_next = leaf->nextLeaf();
            if (!leaf->rebuild(left) || !right_leaf->rebuild(right))
            {
                throw std::runtime_error("leaf split failed: split result does not fit in page");
            }

            leaf->setNextLeaf(right_id);
            right_leaf->setNextLeaf(old_next);

            return SplitResult{right.front().key, right_id};
        }

        InnerNode *inner = static_cast<InnerNode *>(node);
        const std::size_t child_index = inner->childIndexForKey(key);
        const NodeId child_id = inner->childAt(child_index);

        const std::optional<SplitResult> child_split = insertRecursive(child_id, key, value, inserted_new);
        if (!child_split.has_value())
        {
            return std::nullopt;
        }

        // --- OPTIMIZATION: Try fast in-place inner insert! ---
        if (inner->tryInsertInPlace(child_split->separator_key, child_split->right_node))
        {
            return std::nullopt;
        }

        // --- FALLBACK: Rebuild and split inner node ---
        InnerMaterialized state = inner->materialize();
        state.entries.insert(state.entries.begin() + child_index,
                             InnerEntry{child_split->separator_key, child_split->right_node});

        if (inner->rebuild(state.left_child, state.entries))
        {
            return std::nullopt;
        }

        const std::size_t split_index = chooseInnerSplitIndex(state.entries);
        const InnerEntry promoted = state.entries[split_index];

        const std::vector<InnerEntry> left(state.entries.begin(), state.entries.begin() + split_index);
        const NodeId right_left_child = promoted.right_child;
        const std::vector<InnerEntry> right(state.entries.begin() + split_index + 1, state.entries.end());

        const NodeId right_id = allocateInner();
        InnerNode *right_inner = getInner(right_id);

        if (!inner->rebuild(state.left_child, left) || !right_inner->rebuild(right_left_child, right))
        {
            throw std::runtime_error("inner split failed: split result does not fit in page");
        }

        return SplitResult{promoted.key, right_id};
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

    std::size_t AdaptiveBTree::chooseLeafSplitIndex(const std::vector<LeafEntry> &entries) {
        if (entries.size() < 2) return 1;
        
        // O(N) single-pass byte counting instead of O(N^2) string allocations
        std::size_t total_bytes = 0;
        for (const auto& e : entries) total_bytes += e.key.size() + sizeof(Value);
        
        std::size_t current_bytes = 0;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            current_bytes += entries[i].key.size() + sizeof(Value);
            if (current_bytes >= total_bytes / 2) return std::max<std::size_t>(1, i);
        }
        return entries.size() / 2;
    }

    std::size_t AdaptiveBTree::chooseInnerSplitIndex(const std::vector<InnerEntry> &entries) {
        if (entries.size() < 2) return 0;
        
        std::size_t total_bytes = 0;
        for (const auto& e : entries) total_bytes += e.key.size() + sizeof(NodeId);
        
        std::size_t current_bytes = 0;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            current_bytes += entries[i].key.size() + sizeof(NodeId);
            if (current_bytes >= total_bytes / 2) return std::max<std::size_t>(0, i);
        }
        return entries.size() / 2;
    }

    std::size_t AdaptiveBTree::estimateLeafBytes(const std::vector<LeafEntry> &entries,
                                                 std::size_t begin,
                                                 std::size_t end)
    {
        if (begin >= end)
        {
            return std::numeric_limits<std::size_t>::max();
        }

        const std::string prefix = commonPrefixInRange(entries, begin, end);

        // Slotted page sizing math:
        // header + node-prefix + sum(slot metadata + suffix bytes + payload bytes).
        std::size_t bytes = sizeof(SlottedPage::Header) + prefix.size();
        for (std::size_t i = begin; i < end; ++i)
        {
            const std::size_t suffix_len = entries[i].key.size() - prefix.size();
            bytes += sizeof(SlottedPage::LeafSlot) + suffix_len + sizeof(Value);
        }
        return bytes;
    }

    std::size_t AdaptiveBTree::estimateInnerBytes(const std::vector<InnerEntry> &entries,
                                                  std::size_t begin,
                                                  std::size_t end)
    {
        if (begin >= end)
        {
            return sizeof(SlottedPage::Header);
        }

        const std::string prefix = commonPrefixInRange(entries, begin, end);

        // Inner pages omit payload values and store right-child ids in slot metadata.
        std::size_t bytes = sizeof(SlottedPage::Header) + prefix.size();
        for (std::size_t i = begin; i < end; ++i)
        {
            const std::size_t suffix_len = entries[i].key.size() - prefix.size();
            bytes += sizeof(SlottedPage::InnerSlot) + suffix_len;
        }
        return bytes;
    }

} // namespace abt