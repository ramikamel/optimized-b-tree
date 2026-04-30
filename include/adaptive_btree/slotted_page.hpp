#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/config.hpp"

namespace abt
{

    class SlottedPage
    {
    public:
#pragma pack(push, 1)
        struct Header
        {
            std::uint16_t slot_count;
            std::uint16_t free_begin;
            std::uint16_t free_end;
            std::uint16_t prefix_len;
            std::uint8_t node_type;
            std::uint8_t layout_version;
            std::uint16_t reserved;
            NodeId left_child;
            NodeId next_leaf;
        };

        struct LeafSlot
        {
            std::uint16_t key_len;
            std::uint16_t payload_offset;
            std::uint16_t hint;
            std::uint16_t value_len;
        };

        struct InnerSlot
        {
            std::uint16_t key_len;
            std::uint16_t payload_offset;
            std::uint16_t hint;
            NodeId right_child;
        };
#pragma pack(pop)

        explicit SlottedPage(NodeType type);

        void clear();
        NodeType type() const;

        std::uint16_t slotCount() const;
        std::size_t freeSpace() const;

        std::string prefix() const;
        std::string_view prefixView() const;
        void setPrefix(std::string_view prefix);

        bool hasSpaceForLeaf(std::size_t key_len, std::size_t value_len) const;
        bool hasSpaceForInner(std::size_t key_len) const;

        bool insertLeaf(std::uint16_t pos, std::string_view key_suffix, Value value, std::uint16_t hint);
        bool insertInner(std::uint16_t pos, std::string_view key_suffix, NodeId right_child, std::uint16_t hint);

        std::string_view keySuffix(std::uint16_t index) const;
        std::uint16_t hintAt(std::uint16_t index) const;

        Value leafValue(std::uint16_t index) const;
        void setLeafValue(std::uint16_t index, Value value);

        NodeId rightChild(std::uint16_t index) const;

        NodeId leftChild() const;
        void setLeftChild(NodeId id);

        NodeId nextLeaf() const;
        void setNextLeaf(NodeId id);

    private:
        Header &header();
        const Header &header() const;

        std::uint8_t *slotBase();
        const std::uint8_t *slotBase() const;

        std::uint8_t *payloadAt(std::uint16_t offset);
        const std::uint8_t *payloadAt(std::uint16_t offset) const;

        std::size_t slotSize() const;

        LeafSlot leafSlotAt(std::uint16_t index) const;
        void setLeafSlotAt(std::uint16_t index, const LeafSlot &slot);

        InnerSlot innerSlotAt(std::uint16_t index) const;
        void setInnerSlotAt(std::uint16_t index, const InnerSlot &slot);

        void shiftSlotsRight(std::uint16_t pos);

        template <typename SlotType>
        SlotType readSlot(std::uint16_t index) const
        {
            SlotType out{};
            const std::uint8_t *base = slotBase();
            const std::size_t stride = sizeof(SlotType);
            std::memcpy(&out, base + (index * stride), sizeof(SlotType));
            return out;
        }

        template <typename SlotType>
        void writeSlot(std::uint16_t index, const SlotType &slot)
        {
            std::uint8_t *base = slotBase();
            const std::size_t stride = sizeof(SlotType);
            std::memcpy(base + (index * stride), &slot, sizeof(SlotType));
        }

        std::array<std::uint8_t, kPageSizeBytes> bytes_{};
    };

    static_assert(sizeof(SlottedPage::Header) <= 32, "Header should stay compact");

} // namespace abt
