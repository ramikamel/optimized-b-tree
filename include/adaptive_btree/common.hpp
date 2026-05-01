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

    // Leaf sub-kind. Lives at byte 17 of every leaf page (alongside NodeType
    // at byte 16). Inner nodes leave this field zero. The kind tag drives the
    // single switch() in LeafNode's hot-path facade; no virtual dispatch.
    enum class LeafKind : std::uint8_t
    {
        kComparison = 0,   // classic slotted-page leaf (default)
        kFullyDense = 1,   // bitmap + value array, indexed by numeric part
        kSemiDense  = 2,   // offset table + length-prefixed payload heap
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

    // Returns the big-endian uint32 formed by suffix[0..min(4,len)], zero-padded
    // on the LOW side. Two extractions are memcmp-ordered iff the underlying
    // suffix lengths match - the caller must verify uniform suffix_len before
    // relying on numeric ordering across keys.
    //
    // For the FDL/SDL paths the typical use is suffix_len <= 4 (uniform across
    // the leaf), and the result is the literal numeric value to index by.
    inline std::uint32_t extract_numeric_be4(std::string_view suffix) noexcept
    {
        if (suffix.size() >= 4)
        {
            std::uint32_t v;
            std::memcpy(&v, suffix.data(), 4);
            return __builtin_bswap32(v);
        }
        std::uint32_t out = 0;
        for (std::size_t i = 0; i < suffix.size(); ++i)
        {
            out = (out << 8) | static_cast<std::uint8_t>(suffix[i]);
        }
        // Right-pad: if the suffix is shorter than 4, the unread tail is zero.
        // Shift accumulated bytes into the high end so order matches BE. Guard
        // the shift-by-32 UB when suffix is empty (out is already 0 in that case).
        if (suffix.size() > 0 && suffix.size() < 4)
            out <<= 8 * (4 - suffix.size());
        return out;
    }

    // Inverse of extract_numeric_be4: writes the BE bytes of `numeric` into
    // `out` (which must have at least suffix_len bytes; suffix_len in [1,4]).
    // For suffix_len == 4 this is the standard BE encoding.
    inline void inverse_extract_numeric_be4(char* out, std::uint32_t numeric, std::uint16_t suffix_len) noexcept
    {
        // Big-endian write. For suffix_len < 4 we drop the low (4-suffix_len)
        // bytes which are zero anyway because extract_numeric_be4 left-shifts.
        if (suffix_len == 4)
        {
            const std::uint32_t be = __builtin_bswap32(numeric);
            std::memcpy(out, &be, 4);
            return;
        }
        // Fall back: write the high suffix_len bytes of numeric in BE order.
        for (std::uint16_t i = 0; i < suffix_len; ++i)
        {
            out[i] = static_cast<char>((numeric >> (8 * (3 - i))) & 0xFF);
        }
    }

    // ASCII-decimal numeric extraction. Returns true and sets *out_value to the
    // decimal interpretation of `suffix` iff every byte is in '0'..'9'. This
    // gives a *tight* numeric range for zero-padded ASCII integer keys, where
    // raw byte-BE would over-estimate the range by ~10x per digit (since the
    // ASCII gap between '0' and '9' is only 9 but '0' and the next decade
    // boundary is much wider in byte space).
    inline bool extract_decimal_numeric(std::string_view suffix, std::uint32_t* out_value) noexcept
    {
        std::uint32_t n = 0;
        for (char c : suffix)
        {
            const auto u = static_cast<unsigned char>(c);
            if (u < '0' || u > '9') return false;
            n = n * 10u + static_cast<std::uint32_t>(u - '0');
        }
        *out_value = n;
        return true;
    }

    // Inverse: write `numeric` as zero-padded ASCII digits (`digits` chars).
    inline void inverse_decimal_numeric(char* out, std::uint32_t numeric, std::uint16_t digits) noexcept
    {
        for (int i = static_cast<int>(digits) - 1; i >= 0; --i)
        {
            out[i] = static_cast<char>('0' + (numeric % 10u));
            numeric /= 10u;
        }
    }

    // Numeric extraction mode stored in FDL/SDL meta. Determines whether the
    // numeric tail is interpreted as raw BE bytes or as zero-padded ASCII
    // digits. Single byte; lives in the meta region.
    enum class NumericMode : std::uint8_t
    {
        kByteBE = 0,       // 4-byte big-endian raw bytes
        kAsciiDecimal = 1, // up to 9 ASCII digits decoded as decimal
    };

    // Number of decimal digits required to write `n` (without leading zeros).
    // n == 0 returns 1. Capped at 10 to stay within uint32 range.
    inline std::uint16_t count_decimal_digits(std::uint32_t n) noexcept
    {
        std::uint16_t d = 1;
        std::uint32_t threshold = 10u;
        while (n >= threshold)
        {
            ++d;
            if (threshold > 0xFFFFFFFFu / 10u) break; // would overflow
            threshold *= 10u;
        }
        return d;
    }

} // namespace abt
