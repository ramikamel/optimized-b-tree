#include "adaptive_btree/slotted_page.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace abt
{

    SlottedPage::SlottedPage(NodeType type)
    {
        clear();
        header().node_type = static_cast<std::uint8_t>(type);
    }

    void SlottedPage::clear()
    {
        // FATAL BOTTLENECK REMOVED: Do not zero the 4KB array. 
        // The slot/payload boundaries safely protect uninitialized memory.
        Header h{};
        h.slot_count = 0;
        h.free_begin = static_cast<std::uint16_t>(sizeof(Header));
        h.free_end = static_cast<std::uint16_t>(kPageSizeBytes);
        h.prefix_len = 0;
        h.layout_version = 1;
        std::memcpy(bytes_.data(), &h, sizeof(Header));
    }

    NodeType SlottedPage::type() const { return static_cast<NodeType>(header().node_type); }
    std::uint16_t SlottedPage::slotCount() const { return header().slot_count; }
    std::size_t SlottedPage::freeSpace() const { return static_cast<std::size_t>(header().free_end - header().free_begin); }

    std::string SlottedPage::prefix() const
    {
        const Header &h = header();
        const char *begin = reinterpret_cast<const char *>(bytes_.data() + sizeof(Header));
        return std::string(begin, begin + h.prefix_len);
    }

    std::string_view SlottedPage::prefixView() const {
        const Header &h = header();
        const char *begin = reinterpret_cast<const char *>(bytes_.data() + sizeof(Header));
        return std::string_view(begin, h.prefix_len);
    }

    void SlottedPage::setPrefix(std::string_view prefix_value)
    {
        if (header().slot_count != 0) throw std::logic_error("setPrefix requires an empty page");
        if (sizeof(Header) + prefix_value.size() > kPageSizeBytes) throw std::invalid_argument("prefix does not fit in page");

        Header &h = header();
        h.prefix_len = static_cast<std::uint16_t>(prefix_value.size());
        std::memcpy(bytes_.data() + sizeof(Header), prefix_value.data(), prefix_value.size());
        h.free_begin = static_cast<std::uint16_t>(sizeof(Header) + prefix_value.size());
        h.free_end = static_cast<std::uint16_t>(kPageSizeBytes);
    }

    bool SlottedPage::hasSpaceForLeaf(std::size_t key_len, std::size_t value_len) const
    {
        return (sizeof(LeafSlot) + key_len + value_len) <= freeSpace();
    }

    bool SlottedPage::hasSpaceForInner(std::size_t key_len) const
    {
        return (sizeof(InnerSlot) + key_len) <= freeSpace();
    }

    bool SlottedPage::insertLeaf(std::uint16_t pos, std::string_view key_suffix, Value value, std::uint16_t hint) {
        return insertLeaf(pos, key_suffix, "", value, hint);
    }

    bool SlottedPage::insertInner(std::uint16_t pos, std::string_view key_suffix, NodeId right_child, std::uint16_t hint) {
        return insertInner(pos, key_suffix, "", right_child, hint);
    }

    bool SlottedPage::insertLeaf(std::uint16_t pos, std::string_view p1, std::string_view p2, Value value, std::uint16_t hint)
    {
        if (type() != NodeType::kLeaf) return false;
        Header &h = header();
        if (pos > h.slot_count) return false;

        const std::size_t key_len = p1.size() + p2.size();
        constexpr std::size_t value_len = sizeof(Value);
        if (!hasSpaceForLeaf(key_len, value_len)) return false;

        const std::size_t payload_size = key_len + value_len;
        const std::uint16_t payload_offset = static_cast<std::uint16_t>(h.free_end - payload_size);

        std::uint8_t *payload = payloadAt(payload_offset);
        if (!p1.empty()) std::memcpy(payload, p1.data(), p1.size());
        if (!p2.empty()) std::memcpy(payload + p1.size(), p2.data(), p2.size());
        std::memcpy(payload + key_len, &value, sizeof(Value));

        shiftSlotsRight(pos);
        LeafSlot slot{};
        slot.key_len = static_cast<std::uint16_t>(key_len);
        slot.payload_offset = payload_offset;
        slot.hint = hint;
        slot.value_len = static_cast<std::uint16_t>(value_len);
        setLeafSlotAt(pos, slot);

        h.slot_count++;
        h.free_begin += sizeof(LeafSlot);
        h.free_end = payload_offset;
        return true;
    }

    bool SlottedPage::insertInner(std::uint16_t pos, std::string_view p1, std::string_view p2, NodeId right_child, std::uint16_t hint)
    {
        if (type() != NodeType::kInner) return false;
        Header &h = header();
        if (pos > h.slot_count) return false;

        const std::size_t key_len = p1.size() + p2.size();
        if (!hasSpaceForInner(key_len)) return false;

        const std::size_t payload_size = key_len;
        const std::uint16_t payload_offset = static_cast<std::uint16_t>(h.free_end - payload_size);

        std::uint8_t *payload = payloadAt(payload_offset);
        if (!p1.empty()) std::memcpy(payload, p1.data(), p1.size());
        if (!p2.empty()) std::memcpy(payload + p1.size(), p2.data(), p2.size());

        shiftSlotsRight(pos);
        InnerSlot slot{};
        slot.key_len = static_cast<std::uint16_t>(key_len);
        slot.payload_offset = payload_offset;
        slot.hint = hint;
        slot.right_child = right_child;
        setInnerSlotAt(pos, slot);

        h.slot_count++;
        h.free_begin += sizeof(InnerSlot);
        h.free_end = payload_offset;
        return true;
    }

    std::string_view SlottedPage::keySuffix(std::uint16_t index) const
    {
        if (type() == NodeType::kLeaf)
        {
            const LeafSlot slot = leafSlotAt(index);
            const char *ptr = reinterpret_cast<const char *>(payloadAt(slot.payload_offset));
            return std::string_view(ptr, slot.key_len);
        }

        const InnerSlot slot = innerSlotAt(index);
        const char *ptr = reinterpret_cast<const char *>(payloadAt(slot.payload_offset));
        return std::string_view(ptr, slot.key_len);
    }

    std::uint16_t SlottedPage::hintAt(std::uint16_t index) const
    {
        return type() == NodeType::kLeaf ? leafSlotAt(index).hint : innerSlotAt(index).hint;
    }

    Value SlottedPage::leafValue(std::uint16_t index) const
    {
        const LeafSlot slot = leafSlotAt(index);
        Value value = 0;
        const std::uint8_t *ptr = payloadAt(slot.payload_offset) + slot.key_len;
        std::memcpy(&value, ptr, sizeof(Value));
        return value;
    }

    void SlottedPage::setLeafValue(std::uint16_t index, Value value)
    {
        LeafSlot slot = leafSlotAt(index);
        if (slot.value_len != sizeof(Value)) throw std::logic_error("value size mismatch");
        std::uint8_t *ptr = payloadAt(slot.payload_offset) + slot.key_len;
        std::memcpy(ptr, &value, sizeof(Value));
    }

    NodeId SlottedPage::rightChild(std::uint16_t index) const { return innerSlotAt(index).right_child; }
    NodeId SlottedPage::leftChild() const { return header().left_child; }
    void SlottedPage::setLeftChild(NodeId id) { header().left_child = id; }
    NodeId SlottedPage::nextLeaf() const { return header().next_leaf; }
    void SlottedPage::setNextLeaf(NodeId id) { header().next_leaf = id; }

    SlottedPage::Header &SlottedPage::header() { return *reinterpret_cast<Header *>(bytes_.data()); }
    const SlottedPage::Header &SlottedPage::header() const { return *reinterpret_cast<const Header *>(bytes_.data()); }
    std::uint8_t *SlottedPage::slotBase() { return bytes_.data() + sizeof(Header) + header().prefix_len; }
    const std::uint8_t *SlottedPage::slotBase() const { return bytes_.data() + sizeof(Header) + header().prefix_len; }
    std::uint8_t *SlottedPage::payloadAt(std::uint16_t offset) { return bytes_.data() + offset; }
    const std::uint8_t *SlottedPage::payloadAt(std::uint16_t offset) const { return bytes_.data() + offset; }

    std::size_t SlottedPage::slotSize() const { return type() == NodeType::kLeaf ? sizeof(LeafSlot) : sizeof(InnerSlot); }
    SlottedPage::LeafSlot SlottedPage::leafSlotAt(std::uint16_t index) const { return readSlot<LeafSlot>(index); }
    void SlottedPage::setLeafSlotAt(std::uint16_t index, const LeafSlot &slot) { writeSlot<LeafSlot>(index, slot); }
    SlottedPage::InnerSlot SlottedPage::innerSlotAt(std::uint16_t index) const { return readSlot<InnerSlot>(index); }
    void SlottedPage::setInnerSlotAt(std::uint16_t index, const InnerSlot &slot) { writeSlot<InnerSlot>(index, slot); }

    void SlottedPage::shiftSlotsRight(std::uint16_t pos)
    {
        Header &h = header();
        const std::size_t move_count = static_cast<std::size_t>(h.slot_count - pos);
        if (move_count == 0) return;
        const std::size_t stride = slotSize();
        std::uint8_t *base = slotBase();
        const std::size_t src = static_cast<std::size_t>(pos) * stride;
        std::memmove(base + src + stride, base + src, move_count * stride);
    }

} // namespace abt