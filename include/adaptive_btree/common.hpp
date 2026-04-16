#pragma once

#include <algorithm>
#include <cstdint>
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

    inline bool starts_with(std::string_view value, std::string_view prefix)
    {
        if (prefix.size() > value.size())
        {
            return false;
        }
        return value.substr(0, prefix.size()) == prefix;
    }

    inline int lexical_compare(std::string_view lhs, std::string_view rhs)
    {
        const std::size_t common = std::min(lhs.size(), rhs.size());
        const int cmp = std::char_traits<char>::compare(lhs.data(), rhs.data(), common);
        if (cmp < 0)
        {
            return -1;
        }
        if (cmp > 0)
        {
            return 1;
        }
        if (lhs.size() < rhs.size())
        {
            return -1;
        }
        if (lhs.size() > rhs.size())
        {
            return 1;
        }
        return 0;
    }

    inline std::uint16_t make_prefix_hint(std::string_view suffix)
    {
        const std::uint16_t high = suffix.empty() ? 0 : static_cast<unsigned char>(suffix[0]);
        const std::uint16_t low = suffix.size() < 2 ? 0 : static_cast<unsigned char>(suffix[1]);
        return static_cast<std::uint16_t>((high << 8U) | low);
    }

} // namespace abt
