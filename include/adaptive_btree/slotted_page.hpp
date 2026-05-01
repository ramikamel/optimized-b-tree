#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/config.hpp"

namespace abt
{

    // Slotted page following "B-Trees Are Back" §2 with on-page fences.
    //
    // Layout (4096 bytes total):
    //   [Header]                                        - fixed-size metadata
    //   [Hint array: kHintCount * uint32_t]            - sampled heads for fast pre-filter
    //   [Slot directory: grows forward]                - LeafSlot or InnerSlot (sorted by key)
    //   ...free space...
    //   [Heap: grows backward]                          - payloads + lower fence + upper fence
    //
    // Fences are stored untruncated at the back of the heap. The truncated
    // prefix length equals LCP(lower_fence, upper_fence), set exactly once
    // at split time. Local insert/delete never touch the prefix.
    class SlottedPage
    {
    public:
#pragma pack(push, 1)
        struct Header
        {
            std::uint16_t slot_count;
            std::uint16_t free_begin;       // first free byte after header+hints+slots
            std::uint16_t free_end;         // first byte of the heap (payloads grow down to here)
            std::uint16_t prefix_len;       // = LCP(lower_fence, upper_fence)
            std::uint16_t lower_fence_off;  // 0 == no lower fence (unbounded below)
            std::uint16_t lower_fence_len;
            std::uint16_t upper_fence_off;  // 0 == no upper fence (unbounded above)
            std::uint16_t upper_fence_len;
            std::uint8_t  node_type;        // NodeType  (offset 16)
            std::uint8_t  leaf_kind;        // LeafKind  (offset 17; only meaningful for leaves)
            std::uint8_t  reserved[2];
            NodeId        link;             // left_child for inner, next_leaf for leaf
        };

        struct LeafSlot
        {
            std::uint32_t head;             // big-endian first 4 bytes of stored suffix
            std::uint16_t offset;           // payload offset on heap
            std::uint16_t key_len;          // length of stored suffix
        };

        struct InnerSlot
        {
            std::uint32_t head;             // big-endian first 4 bytes of stored suffix
            std::uint16_t offset;           // payload offset on heap
            std::uint16_t key_len;          // length of stored suffix
            NodeId        right_child;      // child for the right side of this separator
        };
#pragma pack(pop)

        static_assert(sizeof(Header) == 24, "header layout drift");
        static_assert(sizeof(LeafSlot) == 8, "leaf slot layout drift");
        static_assert(sizeof(InnerSlot) == 12, "inner slot layout drift");

        // Default ctor leaves bytes_ uninitialized; caller must call init().
        SlottedPage() = default;

        // Reset the page to empty state with the given fences. prefix_len is set to LCP.
        // Pass empty string_view to indicate "no fence" (unbounded).
        void init(NodeType type, std::string_view lower_fence, std::string_view upper_fence);

        NodeType type() const { return static_cast<NodeType>(header().node_type); }
        LeafKind leafKind() const { return static_cast<LeafKind>(header().leaf_kind); }
        void setLeafKind(LeafKind k) { header().leaf_kind = static_cast<std::uint8_t>(k); }
        std::uint16_t slotCount() const { return header().slot_count; }
        std::size_t freeSpace() const { return static_cast<std::size_t>(header().free_end - header().free_begin); }

        // Bytewise length of the static (LCP) prefix; cheaper than prefixView() in
        // hot paths that only need the strip count, not the bytes.
        std::uint16_t prefixLen() const { return header().prefix_len; }

        std::string_view prefixView() const;
        std::string_view lowerFenceView() const;
        std::string_view upperFenceView() const;
        bool hasLowerFence() const { return header().lower_fence_off != 0; }
        bool hasUpperFence() const { return header().upper_fence_off != 0; }

        NodeId link() const { return header().link; }
        void setLink(NodeId id) { header().link = id; }

        // Capacity probes. Returns true iff a new slot+payload of the given size fits.
        bool hasSpaceForLeaf(std::size_t suffix_len) const
        {
            return sizeof(LeafSlot) + suffix_len + sizeof(Value) <= freeSpace();
        }
        bool hasSpaceForInner(std::size_t suffix_len) const
        {
            return sizeof(InnerSlot) + suffix_len <= freeSpace();
        }

        // Insert a new leaf slot at position pos (shifts later slots right).
        // Caller has already stripped the prefix and computed head.
        // Returns false only on real space exhaustion.
        bool insertLeaf(std::uint16_t pos, std::string_view suffix, Value value, std::uint32_t head);
        bool insertInner(std::uint16_t pos, std::string_view suffix, NodeId right_child, std::uint32_t head);

        // Append an entry at the end (no shift, no space check beyond hasSpaceFor*).
        // Used by bulk rebuild after split. Caller ensures fit.
        void appendLeaf(std::string_view suffix, Value value, std::uint32_t head);
        void appendInner(std::string_view suffix, NodeId right_child, std::uint32_t head);

        std::string_view keySuffix(std::uint16_t index) const;
        std::uint32_t headAt(std::uint16_t index) const;

        Value leafValue(std::uint16_t index) const;
        void setLeafValue(std::uint16_t index, Value value);

        NodeId rightChild(std::uint16_t index) const;

        // Sampled hint array (paper §3.3). Read-only access; rebuild after edits.
        std::uint32_t hintAt(std::uint16_t i) const;
        void rebuildHints();

        // Lower-bound search restricted by the hint array. Returns the first
        // slot index whose (head, suffix) >= (target_head, target_suffix).
        std::uint16_t lowerBoundIndex(std::uint32_t target_head, std::string_view target_suffix) const;
        // Same but strict (>). Used for inner-node routing (upper_bound).
        std::uint16_t upperBoundIndex(std::uint32_t target_head, std::string_view target_suffix) const;

        // Truncate the slot directory (used after split). Heap is not compacted;
        // callers that need free space should rebuild instead.
        void setSlotCount(std::uint16_t count) { header().slot_count = count; }

        // Raw byte buffer access. Used by the split path to memcpy a freshly-built
        // scratch page into a target page in one shot.
        std::uint8_t* rawBytes() { return bytes_; }
        const std::uint8_t* rawBytes() const { return bytes_; }

    private:
        Header& header() { return *reinterpret_cast<Header*>(bytes_); }
        const Header& header() const { return *reinterpret_cast<const Header*>(bytes_); }

        std::uint32_t* hintArray()
        {
            return reinterpret_cast<std::uint32_t*>(bytes_ + sizeof(Header));
        }
        const std::uint32_t* hintArray() const
        {
            return reinterpret_cast<const std::uint32_t*>(bytes_ + sizeof(Header));
        }

        std::uint8_t* slotBase()
        {
            return bytes_ + sizeof(Header) + kHintCount * sizeof(std::uint32_t);
        }
        const std::uint8_t* slotBase() const
        {
            return bytes_ + sizeof(Header) + kHintCount * sizeof(std::uint32_t);
        }

        LeafSlot* leafSlots() { return reinterpret_cast<LeafSlot*>(slotBase()); }
        const LeafSlot* leafSlots() const { return reinterpret_cast<const LeafSlot*>(slotBase()); }
        InnerSlot* innerSlots() { return reinterpret_cast<InnerSlot*>(slotBase()); }
        const InnerSlot* innerSlots() const { return reinterpret_cast<const InnerSlot*>(slotBase()); }

        // Raw uninitialized array - prevents zeroing the entire 4KB on alloc.
        std::uint8_t bytes_[kPageSizeBytes];
    };

} // namespace abt
