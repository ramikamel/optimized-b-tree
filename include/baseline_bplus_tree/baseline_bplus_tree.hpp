#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

namespace bpt {

using Value = std::uint64_t;

struct KeyValue {
    std::string key;
    Value value;
};

// Represents the unoptimized database-style Slotted Page defined in the paper.
// All variable-sized strings are stored inline inside the node's byte buffer.
// No Prefix Truncation or Hint Arrays are applied here.
class StandardBPlusTree {
private:
    static constexpr std::size_t PAGE_SIZE = 4096;
    
    // Forward declare Node so Slot can use pointers to it
    struct Node;

    struct Slot {
        std::uint16_t offset;
        std::uint16_t key_len;
        Value value;
        Node* right_child;
    };

    struct Node {
        bool is_leaf;
        std::uint16_t slot_count;
        std::uint16_t free_end;
        Node* next_leaf;
        Node* left_child;

        std::vector<Slot> slots;
        std::uint8_t buffer[PAGE_SIZE]; // Represents the inline heap

        explicit Node(bool leaf) : is_leaf(leaf), slot_count(0), free_end(PAGE_SIZE), 
                                   next_leaf(nullptr), left_child(nullptr) {
            slots.reserve(256);
        }

        bool hasSpace(std::size_t key_len) const {
            std::size_t slot_space = (slot_count + 1) * sizeof(Slot);
            // sizeof(Node) includes the PAGE_SIZE buffer, so subtract it to get the header size
            std::size_t header_space = sizeof(Node) - PAGE_SIZE; 
            
            // Prevent unsigned underflow by using addition
            return free_end >= (slot_space + header_space + key_len);
        }

        std::string_view getKey(int index) const {
            return std::string_view(reinterpret_cast<const char*>(&buffer[slots[index].offset]), slots[index].key_len);
        }

        bool insert(int index, std::string_view key, Value val, Node* child) {
            if (!hasSpace(key.size()) && slot_count > 0) return false;

            free_end -= key.size();
            std::memcpy(&buffer[free_end], key.data(), key.size());

            Slot s;
            s.offset = free_end;
            s.key_len = key.size();
            s.value = val;
            s.right_child = child;

            slots.insert(slots.begin() + index, s);
            slot_count++;
            return true;
        }
    };

    Node* root = nullptr;

    void splitChild(Node* parent, int index, Node* child) {
        Node* new_node = new Node(child->is_leaf);
        
        int mid = child->slot_count / 2;
        
        if (child->is_leaf) {
            for (int i = mid; i < child->slot_count; ++i) {
                new_node->insert(i - mid, child->getKey(i), child->slots[i].value, nullptr);
            }
            child->slot_count = mid;
            child->slots.resize(mid);
            
            new_node->next_leaf = child->next_leaf;
            child->next_leaf = new_node;

            parent->insert(index, new_node->getKey(0), 0, new_node);
        } else {
            new_node->left_child = child->slots[mid].right_child;
            for (int i = mid + 1; i < child->slot_count; ++i) {
                new_node->insert(i - (mid + 1), child->getKey(i), 0, child->slots[i].right_child);
            }
            
            std::string up_key = std::string(child->getKey(mid));
            child->slot_count = mid;
            child->slots.resize(mid);

            parent->insert(index, up_key, 0, new_node);
        }
    }

    void insertNonFull(Node* node, const std::string& key, Value value) {
        int i = node->slot_count - 1;
        if (node->is_leaf) {
            while (i >= 0 && key < node->getKey(i)) i--;
            if (i >= 0 && key == node->getKey(i)) {
                node->slots[i].value = value;
            } else {
                if (!node->insert(i + 1, key, value, nullptr)) {
                    // Force split on out of page-bounds
                }
            }
        } else {
            while (i >= 0 && key < node->getKey(i)) i--;
            i++;
            Node* target = (i == 0) ? node->left_child : node->slots[i - 1].right_child;
            
            if (!target->hasSpace(key.size() + 128)) { // Pre-emptive splitting
                splitChild(node, i, target);
                if (key >= node->getKey(i)) {
                    target = node->slots[i].right_child;
                }
            }
            insertNonFull(target, key, value);
        }
    }

    void destroy(Node* node) {
        if (!node) return;
        if (!node->is_leaf) {
            destroy(node->left_child);
            for (int i = 0; i < node->slot_count; ++i) {
                destroy(node->slots[i].right_child);
            }
        }
        delete node;
    }

public:
    StandardBPlusTree() = default;
    ~StandardBPlusTree() { destroy(root); }

    void insert(const std::string& key, Value value) {
        if (!root) {
            root = new Node(true);
            root->insert(0, key, value, nullptr);
            return;
        }
        if (!root->hasSpace(key.size() + 128)) {
            Node* new_root = new Node(false);
            new_root->left_child = root;
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
            while (i < curr->slot_count && key >= curr->getKey(i)) i++;
            curr = (i == 0) ? curr->left_child : curr->slots[i - 1].right_child;
        }
        for (int i = 0; i < curr->slot_count; ++i) {
            if (curr->getKey(i) == key) return curr->slots[i].value;
        }
        return std::nullopt;
    }

    std::vector<KeyValue> rangeScan(std::string_view start_key, std::size_t max_results) const {
        std::vector<KeyValue> results;
        if (!root || max_results == 0) return results;

        Node* curr = root;
        while (!curr->is_leaf) {
            int i = 0;
            while (i < curr->slot_count && start_key >= curr->getKey(i)) i++;
            curr = (i == 0) ? curr->left_child : curr->slots[i - 1].right_child;
        }

        int i = 0;
        while (i < curr->slot_count && curr->getKey(i) < start_key) i++;

        while (curr && results.size() < max_results) {
            while (i < curr->slot_count && results.size() < max_results) {
                results.push_back({std::string(curr->getKey(i)), curr->slots[i].value});
                i++;
            }
            curr = curr->next_leaf;
            i = 0;
        }
        return results;
    }
};

} // namespace bpt