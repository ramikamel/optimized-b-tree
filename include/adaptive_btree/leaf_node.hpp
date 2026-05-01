#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "adaptive_btree/node.hpp"

namespace abt
{

    // Result of a leaf split. `inserting_numeric` is meaningful only for FDL
    // partition splits (Phase 3); other kinds ignore it. The promoted separator
    // is the full untruncated key the parent should route on.
    struct LeafSplitResult
    {
        std::string  separator;
        bool         right_is_new_partition = false;  // true for FDL/SDL partition splits
    };

    class LeafNode final : public Node
    {
    public:
        explicit LeafNode(NodeId id) : Node(id, NodeType::kLeaf) {}

        // Kind-aware accessors -----------------------------------------------
        LeafKind kind() const { return page_.leafKind(); }
        bool isComparison() const { return kind() == LeafKind::kComparison; }
        bool isFullyDense() const { return kind() == LeafKind::kFullyDense; }
        bool isSemiDense() const { return kind() == LeafKind::kSemiDense; }

        // Reset to empty Comparison leaf with the given fences and forward link.
        void initEmpty(std::string_view lower_fence, std::string_view upper_fence, NodeId next_leaf);

        // Reset to empty Fully Dense Leaf with the given fences, reference key
        // (post-prefix shared bytes), uniform numeric suffix length, base
        // numeric, capacity, and forward link. `numeric_mode` selects how the
        // suffix tail is interpreted (raw BE bytes or ASCII decimal). Returns
        // false if the requested configuration cannot fit.
        bool initFdlEmpty(std::string_view lower_fence, std::string_view upper_fence,
                          std::string_view ref_key, std::uint16_t suffix_len,
                          std::uint32_t base_numeric, std::uint16_t capacity,
                          NumericMode numeric_mode, NodeId next_leaf);

        // Reset to empty Semi Dense Leaf. Same fields as FDL except suffix_len
        // is the *maximum* expected suffix length (used only for sizing
        // heuristics; per-entry length is stored on the heap).
        bool initSdlEmpty(std::string_view lower_fence, std::string_view upper_fence,
                          std::string_view ref_key, std::uint16_t suffix_len,
                          std::uint32_t base_numeric, std::uint16_t capacity,
                          NumericMode numeric_mode, NodeId next_leaf);

        // Hot paths ---------------------------------------------------------
        // Try to insert (key, value). Returns false on space exhaustion (Cmp,
        // SDL) or out-of-range / wrong-shape input (FDL, SDL). Sets
        // *inserted_new = false on upsert.
        bool tryInsert(std::string_view key, Value value, bool& inserted_new);

        std::optional<Value> find(std::string_view key) const;

        // Erase a key. Returns true iff a key was removed (i.e. the leaf
        // shrank). Phase 6 semantics: never reorganizes; merges happen at the
        // tree level after the call returns.
        bool tryErase(std::string_view key);

        // Append entries with key >= `start_key` (ascending), up to `max_to_add`
        // entries, into `results`. If `start_key` is empty, emit from the first
        // entry. Caller chains via `nextLeaf()`.
        void collectScan(std::string_view start_key, std::size_t max_to_add,
                         std::vector<KeyValue>& results) const;

        // Number of stored entries (kind-aware: bit count for FDL, slot count
        // for Cmp, present-offset count for SDL).
        std::uint16_t entryCount() const;

        // True iff merging this leaf with a neighbour is worth attempting.
        bool isUnderflow() const;

        NodeId nextLeaf() const { return page_.link(); }
        void setNextLeaf(NodeId id) { page_.setLink(id); }
        std::string_view prefixView() const { return page_.prefixView(); }
        std::string_view lowerFenceView() const { return page_.lowerFenceView(); }
        std::string_view upperFenceView() const { return page_.upperFenceView(); }

        // Comparison-leaf-only accessors (kept for benchmark observability).
        std::uint16_t cmpSlotCount() const { return page_.slotCount(); }
        std::string_view cmpKeySuffix(std::uint16_t i) const { return page_.keySuffix(i); }
        Value cmpValueAt(std::uint16_t i) const { return page_.leafValue(i); }
        std::string cmpKeyAt(std::uint16_t i) const;

        // Split this leaf. For Comparison: median split with adaptive layout
        // choice. For FDL/SDL: partition-detection (out-of-range) or numeric
        // median split. The `inserting_full_key` is required by FDL/SDL to
        // decide which partition the failing insert goes into; Comparison
        // leaves ignore it.
        LeafSplitResult splitInto(LeafNode& right_node, NodeId right_id,
                                  std::string_view inserting_full_key);

        // Merge `right` into `*this` (consuming all of right's entries). After
        // this call, `right` is logically empty and the caller should free its
        // NodeId. The resulting layout is chosen adaptively. Returns false
        // only if the union does not fit; in that case neither side is mutated.
        bool tryMergeFrom(LeafNode& right);

    private:
        // Per-kind hot paths. All operate on the same page_; the dispatch is a
        // single switch in the public methods above.
        bool tryInsertCmp(std::string_view key, Value value, bool& inserted_new);
        bool tryInsertFdl(std::string_view key, Value value, bool& inserted_new);
        bool tryInsertSdl(std::string_view key, Value value, bool& inserted_new);

        std::optional<Value> findCmp(std::string_view key) const;
        std::optional<Value> findFdl(std::string_view key) const;
        std::optional<Value> findSdl(std::string_view key) const;

        bool tryEraseCmp(std::string_view key);
        bool tryEraseFdl(std::string_view key);
        bool tryEraseSdl(std::string_view key);

        void collectScanCmp(std::string_view start_key, std::size_t max_to_add,
                            std::vector<KeyValue>& results) const;
        void collectScanFdl(std::string_view start_key, std::size_t max_to_add,
                            std::vector<KeyValue>& results) const;
        void collectScanSdl(std::string_view start_key, std::size_t max_to_add,
                            std::vector<KeyValue>& results) const;
    };

} // namespace abt
