#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <algorithm>
#include <cstdint>

namespace bpt {

using Value = std::uint64_t;

struct KeyValue {
    std::string key;
    Value value;
};

// A standard B+-Tree modeling traditional unoptimized in-memory indexes.
// It uses std::string, exposing the cache-miss penalty of heap indirection 
// that slotted-pages are designed to fix.
class StandardBPlusTree {
private:
    static constexpr int B = 64; // Standard branching factor

    struct Node {
        bool is_leaf;
        int count;
        std::string keys[B];
        Value values[B];            // Only used if is_leaf == true
        Node* children[B + 1];      // Only used if is_leaf == false
        Node* next_leaf;            // Leaf chaining for range scans

        explicit Node(bool leaf) : is_leaf(leaf), count(0), next_leaf(nullptr) {}
    };

    Node* root = nullptr;

    void splitChild(Node* parent, int index, Node* child) {
        Node* new_node = new Node(child->is_leaf);
        
        if (child->is_leaf) {
            // Leaf Split: Keep middle key in right child, copy it up to parent
            int mid = child->count / 2;
            new_node->count = child->count - mid;
            for (int i = 0; i < new_node->count; ++i) {
                new_node->keys[i] = std::move(child->keys[mid + i]);
                new_node->values[i] = child->values[mid + i];
            }
            child->count = mid;
            
            new_node->next_leaf = child->next_leaf;
            child->next_leaf = new_node;

            for (int i = parent->count; i > index; --i) {
                parent->keys[i] = std::move(parent->keys[i - 1]);
                parent->children[i + 1] = parent->children[i];
            }
            parent->keys[index] = new_node->keys[0]; // Copy up
            parent->children[index + 1] = new_node;
            parent->count++;
        } else {
            // Inner Split: Move middle key up to parent
            int mid = child->count / 2;
            new_node->count = child->count - mid - 1;
            for (int i = 0; i < new_node->count; ++i) {
                new_node->keys[i] = std::move(child->keys[mid + 1 + i]);
                new_node->children[i] = child->children[mid + 1 + i];
            }
            new_node->children[new_node->count] = child->children[child->count];

            std::string up_key = std::move(child->keys[mid]);
            child->count = mid;

            for (int i = parent->count; i > index; --i) {
                parent->keys[i] = std::move(parent->keys[i - 1]);
                parent->children[i + 1] = parent->children[i];
            }
            parent->keys[index] = std::move(up_key);
            parent->children[index + 1] = new_node;
            parent->count++;
        }
    }

    void insertNonFull(Node* node, const std::string& key, Value value) {
        if (node->is_leaf) {
            int i = node->count - 1;
            // Shifting std::string causes deep copies and pointer moving
            while (i >= 0 && key < node->keys[i]) {
                node->keys[i + 1] = std::move(node->keys[i]);
                node->values[i + 1] = node->values[i];
                i--;
            }
            if (i >= 0 && key == node->keys[i]) {
                node->values[i] = value; 
            } else {
                node->keys[i + 1] = key;
                node->values[i + 1] = value;
                node->count++;
            }
        } else {
            int i = node->count - 1;
            while (i >= 0 && key < node->keys[i]) i--;
            i++;
            if (node->children[i]->count == B) {
                splitChild(node, i, node->children[i]);
                if (key >= node->keys[i]) i++;
            }
            insertNonFull(node->children[i], key, value);
        }
    }

    void destroy(Node* node) {
        if (!node) return;
        if (!node->is_leaf) {
            for (int i = 0; i <= node->count; ++i) destroy(node->children[i]);
        }
        delete node;
    }

public:
    StandardBPlusTree() = default;
    ~StandardBPlusTree() { destroy(root); }

    void insert(const std::string& key, Value value) {
        if (!root) {
            root = new Node(true);
            root->keys[0] = key;
            root->values[0] = value;
            root->count = 1;
            return;
        }
        if (root->count == B) {
            Node* new_root = new Node(false);
            new_root->children[0] = root;
            splitChild(new_root, 0, root);
            root = new_root;
        }
        insertNonFull(root, key, value);
    }

    std::optional<Value> search(std::string_view key) const {
        if (!root) return std::nullopt;
        Node* curr = root;
        while (!curr->is_leaf) {
            int i = 0;
            // Searching std::string array triggers cache misses for long strings
            while (i < curr->count && key >= curr->keys[i]) i++;
            curr = curr->children[i];
        }
        for (int i = 0; i < curr->count; ++i) {
            if (curr->keys[i] == key) return curr->values[i];
        }
        return std::nullopt;
    }

    std::vector<KeyValue> rangeScan(std::string_view start_key, std::size_t max_results) const {
        std::vector<KeyValue> results;
        if (!root || max_results == 0) return results;

        Node* curr = root;
        while (!curr->is_leaf) {
            int i = 0;
            while (i < curr->count && start_key >= curr->keys[i]) i++;
            curr = curr->children[i];
        }

        int i = 0;
        while (i < curr->count && curr->keys[i] < start_key) i++;

        while (curr && results.size() < max_results) {
            while (i < curr->count && results.size() < max_results) {
                results.push_back({curr->keys[i], curr->values[i]});
                i++;
            }
            curr = curr->next_leaf;
            i = 0;
        }
        return results;
    }
};

} // namespace bpt