#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace abt
{

    using NodeId = std::uint32_t;
    using Value = std::uint64_t;

    enum class NodeType : std::uint8_t
    {
        kInner = 0,
        kLeaf = 1,
    };

    struct KeyValue
    {
        std::string key;
        Value value;
    };

    // Big-endian 4-byte head of a key suffix. Padded with zeros.
    // Comparing two heads as uint32_t gives the correct lexicographic order
    // for the first 4 bytes; ties must fall through to a full suffix compare
    // because heads are zero-padded (so they cannot distinguish "ab" from "ab\0").
    inline std::uint32_t make_head(std::string_view suffix) noexcept
    {
        if (suffix.size() >= 4)
        {
            std::uint32_t v;
            std::memcpy(&v, suffix.data(), 4);
            return __builtin_bswap32(v);
        }
        std::uint32_t head = 0;
        for (std::size_t i = 0; i < suffix.size(); ++i)
        {
            head |= static_cast<std::uint32_t>(static_cast<unsigned char>(suffix[i])) << (24 - 8 * i);
        }
        return head;
    }

    inline std::size_t longest_common_prefix(std::string_view a, std::string_view b)
    {
        const std::size_t n = a.size() < b.size() ? a.size() : b.size();
        std::size_t i = 0;
        while (i < n && a[i] == b[i])
            ++i;
        return i;
    }

} // namespace abt
