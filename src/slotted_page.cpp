#include "adaptive_btree/slotted_page.hpp"

#include <algorithm>
#include <cstring>

namespace abt
{

    namespace
    {
        constexpr std::size_t kSlotBaseOffset =
            sizeof(SlottedPage::Header) + kHintCount * sizeof(std::uint32_t);
    }

    void SlottedPage::init(NodeType type, std::string_view lower_fence, std::string_view upper_fence)
    {
        Header h{};
        h.node_type = static_cast<std::uint8_t>(type);
        h.slot_count = 0;
        h.free_begin = static_cast<std::uint16_t>(kSlotBaseOffset);
        h.free_end = static_cast<std::uint16_t>(kPageSizeBytes);
        h.prefix_len = 0;
        h.lower_fence_off = 0;
        h.lower_fence_len = 0;
        h.upper_fence_off = 0;
        h.upper_fence_len = 0;
        h.link = 0;

        // Place upper fence first (highest address), then lower fence, then payloads grow toward slots.
        if (!upper_fence.empty() || /* explicit empty fence flag */ false)
        {
            h.free_end = static_cast<std::uint16_t>(h.free_end - upper_fence.size());
            h.upper_fence_off = h.free_end;
            h.upper_fence_len = static_cast<std::uint16_t>(upper_fence.size());
            std::memcpy(bytes_ + h.upper_fence_off, upper_fence.data(), upper_fence.size());
        }
        if (!lower_fence.empty())
        {
            h.free_end = static_cast<std::uint16_t>(h.free_end - lower_fence.size());
            h.lower_fence_off = h.free_end;
            h.lower_fence_len = static_cast<std::uint16_t>(lower_fence.size());
            std::memcpy(bytes_ + h.lower_fence_off, lower_fence.data(), lower_fence.size());
        }

        // Static prefix truncation: prefix is the LCP of the (untruncated) fences.
        // If either fence is unbounded (empty/missing), prefix is empty.
        if (h.lower_fence_len > 0 && h.upper_fence_len > 0)
        {
            h.prefix_len = static_cast<std::uint16_t>(longest_common_prefix(lower_fence, upper_fence));
        }

        std::memcpy(bytes_, &h, sizeof(Header));

        // Zero the hint array up-front (cheap; 64 bytes).
        std::memset(bytes_ + sizeof(Header), 0, kHintCount * sizeof(std::uint32_t));
    }

    std::string_view SlottedPage::prefixView() const
    {
        const Header& h = header();
        if (h.prefix_len == 0) return {};
        // The prefix is the first prefix_len bytes of either fence; lower exists when prefix > 0.
        const char* base = reinterpret_cast<const char*>(bytes_ + h.lower_fence_off);
        return std::string_view(base, h.prefix_len);
    }

    std::string_view SlottedPage::lowerFenceView() const
    {
        const Header& h = header();
        if (h.lower_fence_off == 0) return {};
        const char* base = reinterpret_cast<const char*>(bytes_ + h.lower_fence_off);
        return std::string_view(base, h.lower_fence_len);
    }

    std::string_view SlottedPage::upperFenceView() const
    {
        const Header& h = header();
        if (h.upper_fence_off == 0) return {};
        const char* base = reinterpret_cast<const char*>(bytes_ + h.upper_fence_off);
        return std::string_view(base, h.upper_fence_len);
    }

    bool SlottedPage::insertLeaf(std::uint16_t pos, std::string_view suffix, Value value, std::uint32_t head)
    {
        Header& h = header();
        if (pos > h.slot_count) return false;
        if (!hasSpaceForLeaf(suffix.size())) return false;

        const std::size_t payload_size = suffix.size() + sizeof(Value);
        const std::uint16_t payload_offset = static_cast<std::uint16_t>(h.free_end - payload_size);

        std::uint8_t* p = bytes_ + payload_offset;
        if (!suffix.empty())
            std::memcpy(p, suffix.data(), suffix.size());
        std::memcpy(p + suffix.size(), &value, sizeof(Value));

        LeafSlot* slots = leafSlots();
        if (pos < h.slot_count)
        {
            std::memmove(slots + pos + 1, slots + pos,
                         (h.slot_count - pos) * sizeof(LeafSlot));
        }
        slots[pos].head = head;
        slots[pos].offset = payload_offset;
        slots[pos].key_len = static_cast<std::uint16_t>(suffix.size());

        ++h.slot_count;
        h.free_begin = static_cast<std::uint16_t>(h.free_begin + sizeof(LeafSlot));
        h.free_end = payload_offset;
        return true;
    }

    bool SlottedPage::insertInner(std::uint16_t pos, std::string_view suffix, NodeId right_child, std::uint32_t head)
    {
        Header& h = header();
        if (pos > h.slot_count) return false;
        if (!hasSpaceForInner(suffix.size())) return false;

        const std::size_t payload_size = suffix.size();
        const std::uint16_t payload_offset = static_cast<std::uint16_t>(h.free_end - payload_size);

        if (!suffix.empty())
            std::memcpy(bytes_ + payload_offset, suffix.data(), suffix.size());

        InnerSlot* slots = innerSlots();
        if (pos < h.slot_count)
        {
            std::memmove(slots + pos + 1, slots + pos,
                         (h.slot_count - pos) * sizeof(InnerSlot));
        }
        slots[pos].head = head;
        slots[pos].offset = payload_offset;
        slots[pos].key_len = static_cast<std::uint16_t>(suffix.size());
        slots[pos].right_child = right_child;

        ++h.slot_count;
        h.free_begin = static_cast<std::uint16_t>(h.free_begin + sizeof(InnerSlot));
        h.free_end = payload_offset;
        return true;
    }

    void SlottedPage::appendLeaf(std::string_view suffix, Value value, std::uint32_t head)
    {
        Header& h = header();
        const std::size_t payload_size = suffix.size() + sizeof(Value);
        const std::uint16_t payload_offset = static_cast<std::uint16_t>(h.free_end - payload_size);

        std::uint8_t* p = bytes_ + payload_offset;
        if (!suffix.empty())
            std::memcpy(p, suffix.data(), suffix.size());
        std::memcpy(p + suffix.size(), &value, sizeof(Value));

        LeafSlot* slots = leafSlots();
        slots[h.slot_count].head = head;
        slots[h.slot_count].offset = payload_offset;
        slots[h.slot_count].key_len = static_cast<std::uint16_t>(suffix.size());

        ++h.slot_count;
        h.free_begin = static_cast<std::uint16_t>(h.free_begin + sizeof(LeafSlot));
        h.free_end = payload_offset;
    }

    void SlottedPage::appendInner(std::string_view suffix, NodeId right_child, std::uint32_t head)
    {
        Header& h = header();
        const std::size_t payload_size = suffix.size();
        const std::uint16_t payload_offset = static_cast<std::uint16_t>(h.free_end - payload_size);

        if (!suffix.empty())
            std::memcpy(bytes_ + payload_offset, suffix.data(), suffix.size());

        InnerSlot* slots = innerSlots();
        slots[h.slot_count].head = head;
        slots[h.slot_count].offset = payload_offset;
        slots[h.slot_count].key_len = static_cast<std::uint16_t>(suffix.size());
        slots[h.slot_count].right_child = right_child;

        ++h.slot_count;
        h.free_begin = static_cast<std::uint16_t>(h.free_begin + sizeof(InnerSlot));
        h.free_end = payload_offset;
    }

    std::string_view SlottedPage::keySuffix(std::uint16_t index) const
    {
        if (type() == NodeType::kLeaf)
        {
            const LeafSlot& s = leafSlots()[index];
            return std::string_view(reinterpret_cast<const char*>(bytes_ + s.offset), s.key_len);
        }
        const InnerSlot& s = innerSlots()[index];
        return std::string_view(reinterpret_cast<const char*>(bytes_ + s.offset), s.key_len);
    }

    std::uint32_t SlottedPage::headAt(std::uint16_t index) const
    {
        return type() == NodeType::kLeaf ? leafSlots()[index].head : innerSlots()[index].head;
    }

    Value SlottedPage::leafValue(std::uint16_t index) const
    {
        const LeafSlot& s = leafSlots()[index];
        Value v = 0;
        std::memcpy(&v, bytes_ + s.offset + s.key_len, sizeof(Value));
        return v;
    }

    void SlottedPage::setLeafValue(std::uint16_t index, Value value)
    {
        const LeafSlot& s = leafSlots()[index];
        std::memcpy(bytes_ + s.offset + s.key_len, &value, sizeof(Value));
    }

    NodeId SlottedPage::rightChild(std::uint16_t index) const
    {
        return innerSlots()[index].right_child;
    }

    std::uint32_t SlottedPage::hintAt(std::uint16_t i) const
    {
        return hintArray()[i];
    }

    void SlottedPage::rebuildHints()
    {
        const Header& h = header();
        std::uint32_t* hints = hintArray();
        if (h.slot_count == 0)
        {
            std::memset(hints, 0, kHintCount * sizeof(std::uint32_t));
            return;
        }
        // Hint at i samples slot at (slot_count / (kHintCount + 1)) * (i + 1).
        // When slot_count <= kHintCount, the hints degenerate but binary search
        // is still correct; we sample by clamping to slot_count - 1.
        const std::uint32_t spacing = h.slot_count / (kHintCount + 1);
        if (spacing == 0)
        {
            // Too few slots to sample meaningfully. Fill with the last head so
            // the hint-search just falls through to the full-range binary search.
            const std::uint32_t last_head = headAt(static_cast<std::uint16_t>(h.slot_count - 1));
            for (std::uint16_t i = 0; i < kHintCount; ++i) hints[i] = last_head;
            return;
        }
        for (std::uint16_t i = 0; i < kHintCount; ++i)
        {
            const std::uint16_t idx = static_cast<std::uint16_t>(spacing * (i + 1));
            hints[i] = headAt(idx);
        }
    }

    std::uint16_t SlottedPage::lowerBoundIndex(std::uint32_t target_head, std::string_view target_suffix) const
    {
        const Header& h = header();
        if (h.slot_count == 0) return 0;

        // Narrow the search range using hints. lo and hi must be SAFE bounds —
        // tighter on lo using the last hint strictly less than target, and tighter
        // on hi using the FIRST hint strictly greater than target (advancing past
        // any equal hints, since hints with equal heads cannot rule the slot out).
        std::uint16_t lo = 0;
        std::uint16_t hi = h.slot_count;
        const std::uint32_t spacing = h.slot_count / (kHintCount + 1);
        if (spacing >= 1)
        {
            const std::uint32_t* hints = hintArray();
            std::uint16_t i = 0;
            while (i < kHintCount && hints[i] < target_head) ++i;
            if (i > 0)
                lo = static_cast<std::uint16_t>(spacing * i + 1);
            std::uint16_t j = i;
            while (j < kHintCount && hints[j] <= target_head) ++j;
            if (j < kHintCount)
                hi = static_cast<std::uint16_t>(spacing * (j + 1) + 1);
            if (hi > h.slot_count) hi = h.slot_count;
            if (lo > hi) lo = hi;
        }

        // Standard lower_bound on (head, suffix).
        if (type() == NodeType::kLeaf)
        {
            const LeafSlot* slots = leafSlots();
            while (lo < hi)
            {
                const std::uint16_t mid = lo + ((hi - lo) >> 1);
                const LeafSlot& s = slots[mid];
                if (s.head < target_head)
                {
                    lo = mid + 1;
                }
                else if (s.head > target_head)
                {
                    hi = mid;
                }
                else
                {
                    std::string_view suf(reinterpret_cast<const char*>(bytes_ + s.offset), s.key_len);
                    if (suf < target_suffix) lo = mid + 1;
                    else hi = mid;
                }
            }
        }
        else
        {
            const InnerSlot* slots = innerSlots();
            while (lo < hi)
            {
                const std::uint16_t mid = lo + ((hi - lo) >> 1);
                const InnerSlot& s = slots[mid];
                if (s.head < target_head)
                {
                    lo = mid + 1;
                }
                else if (s.head > target_head)
                {
                    hi = mid;
                }
                else
                {
                    std::string_view suf(reinterpret_cast<const char*>(bytes_ + s.offset), s.key_len);
                    if (suf < target_suffix) lo = mid + 1;
                    else hi = mid;
                }
            }
        }
        return lo;
    }

    std::uint16_t SlottedPage::upperBoundIndex(std::uint32_t target_head, std::string_view target_suffix) const
    {
        const Header& h = header();
        if (h.slot_count == 0) return 0;

        // Same hint narrowing as lowerBoundIndex; the binary-search comparison
        // below uses <= to implement strict upper_bound.
        std::uint16_t lo = 0;
        std::uint16_t hi = h.slot_count;
        const std::uint32_t spacing = h.slot_count / (kHintCount + 1);
        if (spacing >= 1)
        {
            const std::uint32_t* hints = hintArray();
            std::uint16_t i = 0;
            while (i < kHintCount && hints[i] < target_head) ++i;
            if (i > 0)
                lo = static_cast<std::uint16_t>(spacing * i + 1);
            std::uint16_t j = i;
            while (j < kHintCount && hints[j] <= target_head) ++j;
            if (j < kHintCount)
                hi = static_cast<std::uint16_t>(spacing * (j + 1) + 1);
            if (hi > h.slot_count) hi = h.slot_count;
            if (lo > hi) lo = hi;
        }

        if (type() == NodeType::kLeaf)
        {
            const LeafSlot* slots = leafSlots();
            while (lo < hi)
            {
                const std::uint16_t mid = lo + ((hi - lo) >> 1);
                const LeafSlot& s = slots[mid];
                if (s.head < target_head)
                {
                    lo = mid + 1;
                }
                else if (s.head > target_head)
                {
                    hi = mid;
                }
                else
                {
                    std::string_view suf(reinterpret_cast<const char*>(bytes_ + s.offset), s.key_len);
                    if (suf <= target_suffix) lo = mid + 1;
                    else hi = mid;
                }
            }
        }
        else
        {
            const InnerSlot* slots = innerSlots();
            while (lo < hi)
            {
                const std::uint16_t mid = lo + ((hi - lo) >> 1);
                const InnerSlot& s = slots[mid];
                if (s.head < target_head)
                {
                    lo = mid + 1;
                }
                else if (s.head > target_head)
                {
                    hi = mid;
                }
                else
                {
                    std::string_view suf(reinterpret_cast<const char*>(bytes_ + s.offset), s.key_len);
                    if (suf <= target_suffix) lo = mid + 1;
                    else hi = mid;
                }
            }
        }
        return lo;
    }

} // namespace abt
