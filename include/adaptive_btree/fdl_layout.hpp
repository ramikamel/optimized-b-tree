#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/config.hpp"

namespace abt
{

    // Fully Dense Leaf physical layout (4 KiB page overlay).
    //
    // The first 24 bytes match SlottedPage::Header layout *positionally* so the
    // common dispatch fields (node_type at offset 16, leaf_kind at offset 17,
    // link at offset 20) are read uniformly by callers regardless of leaf kind.
    // The hint-array region of a slotted page (offsets 24..87) is repurposed
    // for FDL-specific metadata. Bitmap and value array begin at offset 88
    // (= sizeof(SlottedPage::Header) + sizeof(FdlMeta)).
    //
    // Layout:
    //   [0..23]      Header-compatible: slot_count(=count), free_begin/free_end,
    //                prefix_len, lower/upper fences, node_type=kLeaf,
    //                leaf_kind=kFullyDense, link=next_leaf.
    //   [24..87]     FdlMeta: capacity, suffix_len, base_numeric, ref_key, ...
    //   [88..147]    bitmap[ kFdlBitmapBytes ]   (1 bit per numeric slot)
    //   [148..3987]  values[ kFdlMaxCapacity * sizeof(Value) ]
    //   [3988..4095] free space + fences (fences live at the back of the page)

    // Offsets are constexpr so the compiler can fold them into immediates.
    inline constexpr std::size_t kFdlHeaderEnd     = 24;        // == sizeof(SlottedPage::Header)
    inline constexpr std::size_t kFdlMetaSize      = 64;        // matches old hint-array span
    inline constexpr std::size_t kFdlBitmapOffset  = kFdlHeaderEnd + kFdlMetaSize;
    inline constexpr std::size_t kFdlValuesOffset  = kFdlBitmapOffset + kFdlBitmapBytes;
    inline constexpr std::size_t kFdlFencesBegin   = kFdlValuesOffset + kFdlValuesBytes;

    static_assert(kFdlFencesBegin <= kPageSizeBytes,
                  "FDL fixed regions must fit in the 4 KiB page");

#pragma pack(push, 1)
    // FdlMeta is overlaid at offset 24 of an FDL page (the slotted-page hint
    // array region). All offsets stored here are page-byte offsets.
    struct FdlMeta
    {
        std::uint16_t capacity;       // numeric range covered by this FDL
        std::uint16_t suffix_len;     // uniform key-suffix width (after stripping prefix + ref_key)
        std::uint16_t ref_key_off;    // 0 = no ref-key (rare)
        std::uint16_t ref_key_len;
        std::uint32_t base_numeric;   // lowest numeric value the leaf accepts
        std::uint16_t bitmap_off;     // = kFdlBitmapOffset (kept explicit for debugging)
        std::uint16_t values_off;     // = kFdlValuesOffset
        std::uint8_t  numeric_mode;   // NumericMode: 0 = byte-BE, 1 = ASCII decimal
        std::uint8_t  reserved[64 - 17];
    };
#pragma pack(pop)
    static_assert(sizeof(FdlMeta) == 64, "FdlMeta must occupy the full meta region");

    // Helpers that read/mutate an FDL page given its raw 4 KiB buffer.
    inline FdlMeta& fdl_meta(std::uint8_t* page) noexcept
    {
        return *reinterpret_cast<FdlMeta*>(page + kFdlHeaderEnd);
    }
    inline const FdlMeta& fdl_meta(const std::uint8_t* page) noexcept
    {
        return *reinterpret_cast<const FdlMeta*>(page + kFdlHeaderEnd);
    }

    inline std::uint8_t* fdl_bitmap(std::uint8_t* page) noexcept
    {
        return page + kFdlBitmapOffset;
    }
    inline const std::uint8_t* fdl_bitmap(const std::uint8_t* page) noexcept
    {
        return page + kFdlBitmapOffset;
    }

    inline Value* fdl_values(std::uint8_t* page) noexcept
    {
        return reinterpret_cast<Value*>(page + kFdlValuesOffset);
    }
    inline const Value* fdl_values(const std::uint8_t* page) noexcept
    {
        return reinterpret_cast<const Value*>(page + kFdlValuesOffset);
    }

    inline bool fdl_bit_test(const std::uint8_t* page, std::uint16_t slot) noexcept
    {
        return (fdl_bitmap(page)[slot >> 3] >> (slot & 7)) & 1u;
    }
    inline void fdl_bit_set(std::uint8_t* page, std::uint16_t slot) noexcept
    {
        fdl_bitmap(page)[slot >> 3] |= static_cast<std::uint8_t>(1u << (slot & 7));
    }
    inline void fdl_bit_clear(std::uint8_t* page, std::uint16_t slot) noexcept
    {
        fdl_bitmap(page)[slot >> 3] &= static_cast<std::uint8_t>(~(1u << (slot & 7)));
    }

} // namespace abt
