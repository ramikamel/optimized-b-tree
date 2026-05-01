#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "adaptive_btree/common.hpp"
#include "adaptive_btree/config.hpp"

namespace abt
{

    // Semi Dense Leaf physical layout (4 KiB page overlay).
    //
    // Like FDL, SDL's first 24 bytes are SlottedPage::Header-compatible so the
    // dispatch path can read node_type/leaf_kind/link without knowing the
    // concrete leaf kind. The hint-array region (offsets 24..87) is repurposed
    // for SDL-specific metadata. After that comes a uint16_t offset table
    // indexed by (numeric - base), where each entry is a byte offset into a
    // length-prefixed payload heap that grows down from the end of the page.
    //
    // Layout:
    //   [0..23]              Header-compatible (slot_count = count of present
    //                        entries; free_begin/free_end track the heap top
    //                        and offset-table top respectively).
    //   [24..87]             SdlMeta: capacity, base_numeric, suffix_len, ...
    //   [88..88+2*capacity)  offsets[capacity]  (0 = absent)
    //   [..heap_top)         free space
    //   [heap_top..page_end) payloads, growing downward, each formatted as
    //                        [uint16_t suffix_len][suffix bytes][Value]
    //
    // The leaf chain link, the fences, and the prefix view are inherited from
    // SlottedPage::Header. SDL fences live in the same back-of-page region the
    // slotted page uses; the `free_end` field doubles as `heap_top`.

    inline constexpr std::size_t kSdlHeaderEnd      = 24;
    inline constexpr std::size_t kSdlMetaSize       = 64;
    inline constexpr std::size_t kSdlOffsetsOffset  = kSdlHeaderEnd + kSdlMetaSize;
    inline constexpr std::size_t kSdlOffsetsBytes   = kSdlMaxCapacity * sizeof(std::uint16_t);
    inline constexpr std::size_t kSdlHeapBegin      = kSdlOffsetsOffset + kSdlOffsetsBytes;

    static_assert(kSdlHeapBegin < kPageSizeBytes,
                  "SDL fixed regions must leave room for heap + fences");

#pragma pack(push, 1)
    struct SdlMeta
    {
        std::uint16_t capacity;       // numeric range covered by this SDL
        std::uint16_t suffix_len;     // 0 == variable-length suffix (length is stored per-entry)
        std::uint16_t ref_key_off;
        std::uint16_t ref_key_len;
        std::uint32_t base_numeric;
        std::uint16_t offsets_off;    // = kSdlOffsetsOffset
        std::uint16_t heap_bottom;    // first byte after the offset table; heap cannot grow below this
        std::uint8_t  numeric_mode;   // NumericMode
        std::uint8_t  reserved[64 - 17];
    };
#pragma pack(pop)
    static_assert(sizeof(SdlMeta) == 64, "SdlMeta must occupy the full meta region");

    inline SdlMeta& sdl_meta(std::uint8_t* page) noexcept
    {
        return *reinterpret_cast<SdlMeta*>(page + kSdlHeaderEnd);
    }
    inline const SdlMeta& sdl_meta(const std::uint8_t* page) noexcept
    {
        return *reinterpret_cast<const SdlMeta*>(page + kSdlHeaderEnd);
    }

    inline std::uint16_t* sdl_offsets(std::uint8_t* page) noexcept
    {
        return reinterpret_cast<std::uint16_t*>(page + kSdlOffsetsOffset);
    }
    inline const std::uint16_t* sdl_offsets(const std::uint8_t* page) noexcept
    {
        return reinterpret_cast<const std::uint16_t*>(page + kSdlOffsetsOffset);
    }

    // Per-entry payload: [uint16_t suffix_len][suffix bytes][Value].
    inline std::size_t sdl_payload_bytes(std::uint16_t suffix_len) noexcept
    {
        return sizeof(std::uint16_t) + suffix_len + sizeof(Value);
    }

} // namespace abt
