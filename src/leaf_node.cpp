#include "adaptive_btree/leaf_node.hpp"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>

#include "adaptive_btree/fdl_layout.hpp"
#include "adaptive_btree/sdl_layout.hpp"

namespace abt
{

    namespace
    {
        // Forward declarations -------------------------------------------------
        struct LayoutChoice;
        LayoutChoice chooseLayoutCmp(const SlottedPage& src, std::uint16_t lo, std::uint16_t hi,
                                     std::string_view extra_suffix = {});

        bool init_fdl_empty(SlottedPage& dst, std::string_view lf, std::string_view uf,
                            std::string_view ref_key, std::uint16_t suffix_len,
                            std::uint32_t base, std::uint16_t capacity,
                            NumericMode mode, NodeId link);
        bool init_sdl_empty(SlottedPage& dst, std::string_view lf, std::string_view uf,
                            std::string_view ref_key, std::uint16_t suffix_len,
                            std::uint32_t base, std::uint16_t capacity,
                            NumericMode mode, NodeId link);

        void materializeSliceInto(SlottedPage& dst, const SlottedPage& src,
                                  std::uint16_t lo, std::uint16_t hi,
                                  std::string_view lower_fence,
                                  std::string_view upper_fence,
                                  NodeId next_link,
                                  const LayoutChoice& choice);

        // ---------------------------------------------------------------------
        // Thread-local scratch pages used as a build target during splits,
        // merges, and adaptive-layout conversions. Reused across calls;
        // SlottedPage::init() resets the metadata cheaply (no 4 KiB memset).
        // ---------------------------------------------------------------------
        thread_local SlottedPage tls_scratch;
        thread_local SlottedPage tls_scratch_aux;

        // ---------------------------------------------------------------------
        // Tiny helpers
        // ---------------------------------------------------------------------
        // Mode-aware encoder: write `numeric` into `out` per `mode` over
        // `suffix_len` bytes. Used for FDL key materialization and separator
        // construction during partition splits.
        inline void write_numeric(char* out, std::uint32_t numeric,
                                  std::uint16_t suffix_len, NumericMode mode) noexcept
        {
            if (mode == NumericMode::kAsciiDecimal)
                inverse_decimal_numeric(out, numeric, suffix_len);
            else
                inverse_extract_numeric_be4(out, numeric, suffix_len);
        }

        // Mode-aware decoder: parses `suffix` into `*out`. Returns false if
        // the bytes do not match the mode (e.g. non-digit when mode is
        // ASCII-decimal). For byte-BE, never fails.
        inline bool read_numeric(std::string_view suffix, NumericMode mode,
                                 std::uint32_t* out) noexcept
        {
            if (mode == NumericMode::kAsciiDecimal)
                return extract_decimal_numeric(suffix, out);
            *out = extract_numeric_be4(suffix);
            return true;
        }

        // Strip the leaf's static prefix and ref_key from a full key. Returns
        // false if the key does not match the leaf's shape (too short or the
        // ref_key bytes do not match). On success, *out_suffix is set to the
        // numeric-only tail.
        inline bool fdl_strip_to_suffix(std::string_view key,
                                        const SlottedPage& page,
                                        const FdlMeta& meta,
                                        const std::uint8_t* raw,
                                        std::string_view& out_suffix) noexcept
        {
            const std::size_t prefix_len = page.prefixLen();
            if (key.size() < prefix_len) return false;
            const std::string_view post_prefix(key.data() + prefix_len,
                                               key.size() - prefix_len);
            if (post_prefix.size() < meta.ref_key_len) return false;
            if (meta.ref_key_len > 0)
            {
                const char* ref = reinterpret_cast<const char*>(raw + meta.ref_key_off);
                if (std::memcmp(post_prefix.data(), ref, meta.ref_key_len) != 0)
                    return false;
            }
            const std::string_view suffix = post_prefix.substr(meta.ref_key_len);
            if (suffix.size() != meta.suffix_len) return false;
            out_suffix = suffix;
            return true;
        }

        inline bool sdl_strip_to_suffix(std::string_view key,
                                        const SlottedPage& page,
                                        const SdlMeta& meta,
                                        const std::uint8_t* raw,
                                        std::string_view& out_suffix) noexcept
        {
            const std::size_t prefix_len = page.prefixLen();
            if (key.size() < prefix_len) return false;
            const std::string_view post_prefix(key.data() + prefix_len,
                                               key.size() - prefix_len);
            if (post_prefix.size() < meta.ref_key_len) return false;
            if (meta.ref_key_len > 0)
            {
                const char* ref = reinterpret_cast<const char*>(raw + meta.ref_key_off);
                if (std::memcmp(post_prefix.data(), ref, meta.ref_key_len) != 0)
                    return false;
            }
            out_suffix = post_prefix.substr(meta.ref_key_len);
            return true;
        }

        inline std::uint32_t sdl_numeric_of(std::string_view suffix) noexcept
        {
            return extract_numeric_be4(suffix.size() <= 4 ? suffix : suffix.substr(0, 4));
        }

        // ---------------------------------------------------------------------
        // Layout choice (Phase 5 helper, lives here so split/merge can call it)
        // ---------------------------------------------------------------------
        struct LayoutChoice
        {
            LeafKind     kind = LeafKind::kComparison;
            std::uint32_t base = 0;
            std::uint16_t capacity = 0;
            std::uint16_t suffix_len = 0;
            std::uint16_t ref_key_len = 0; // post-prefix shared bytes
            NumericMode  numeric_mode = NumericMode::kByteBE;
        };

        LayoutChoice chooseLayoutCmp(const SlottedPage& src, std::uint16_t lo, std::uint16_t hi,
                                     std::string_view extra_suffix)
        {
            LayoutChoice fallback{};
            if (hi <= lo) return fallback;
            const std::uint16_t count = static_cast<std::uint16_t>(hi - lo);
            if (count < 2) return fallback;

            const std::string_view first_suf = src.keySuffix(lo);
            const std::size_t L = first_suf.size();
            if (L == 0 || L > 16) return fallback;

            // 1. Uniform suffix length check. The "extra" suffix (the inserting
            //    key, if provided) must also have the same length to be
            //    representable in a uniform FDL/SDL layout.
            for (std::uint16_t i = static_cast<std::uint16_t>(lo + 1); i < hi; ++i)
            {
                if (src.keySuffix(i).size() != L) return fallback;
            }
            const bool have_extra = !extra_suffix.empty();
            if (have_extra && extra_suffix.size() != L) return fallback;

            // 2. Find the longest common ref_key (post-prefix shared bytes)
            //    among all suffixes (and the extra suffix if provided); the
            //    residual is the numeric tail. Early out: if L - common already
            //    exceeds 9 we can never form a decimal numeric (max 9 digits
            //    fit in uint32) or a byte-BE numeric (max 4 bytes), so abandon
            //    FDL/SDL.
            const char* p0 = first_suf.data();
            std::size_t common = L;
            const std::size_t min_common_for_dense = (L > 9) ? (L - 9) : 0;
            for (std::uint16_t i = static_cast<std::uint16_t>(lo + 1); i < hi; ++i)
            {
                const std::string_view s = src.keySuffix(i);
                std::size_t c = 0;
                while (c < common && c < s.size() && p0[c] == s.data()[c]) ++c;
                common = c;
                if (common < min_common_for_dense) return fallback;
                if (common == 0) break;
            }
            if (have_extra && common > 0)
            {
                std::size_t c = 0;
                while (c < common && p0[c] == extra_suffix.data()[c]) ++c;
                common = c;
                if (common < min_common_for_dense) return fallback;
            }
            // Determine the suffix tail length we'll try numeric extraction
            // over. ASCII-decimal mode supports up to 9 digits; byte-BE caps
            // at 4 raw bytes. We prefer ASCII-decimal whenever the tail is
            // entirely ASCII digits (gives a tight range that respects the
            // decimal interpretation rather than the byte interpretation).
            const std::uint16_t tail_len_byte = static_cast<std::uint16_t>(L - common);
            if (tail_len_byte == 0) return fallback;

            // 2a. Try ASCII-decimal mode first (up to 9 digits).
            if (tail_len_byte <= 9)
            {
                bool all_digits = true;
                for (std::uint16_t i = lo; i < hi && all_digits; ++i)
                {
                    const std::string_view s = src.keySuffix(i);
                    for (std::size_t p = common; p < L; ++p)
                    {
                        const auto u = static_cast<unsigned char>(s.data()[p]);
                        if (u < '0' || u > '9') { all_digits = false; break; }
                    }
                }
                if (all_digits)
                {
                    std::uint32_t n_min = 0, n_max = 0;
                    extract_decimal_numeric(first_suf.substr(common, tail_len_byte), &n_min);
                    extract_decimal_numeric(src.keySuffix(static_cast<std::uint16_t>(hi - 1))
                                                .substr(common, tail_len_byte), &n_max);
                    // Fold in the inserting key's numeric (if provided) so the
                    // chosen cap covers it. This is what prevents the "demote
                    // + median + re-promote" path from producing a layout
                    // tighter than the inserting key requires.
                    if (have_extra)
                    {
                        bool extra_ok = true;
                        for (std::size_t p = common; p < L; ++p)
                        {
                            const auto u = static_cast<unsigned char>(extra_suffix[p]);
                            if (u < '0' || u > '9') { extra_ok = false; break; }
                        }
                        if (!extra_ok) return fallback;
                        std::uint32_t n_extra = 0;
                        extract_decimal_numeric(extra_suffix.substr(common, tail_len_byte), &n_extra);
                        if (n_extra < n_min) n_min = n_extra;
                        if (n_extra > n_max) n_max = n_extra;
                    }
                    if (n_max >= n_min)
                    {
                        const std::uint64_t range = static_cast<std::uint64_t>(n_max) - n_min + 1;

                        // Cap is bounded by what the chosen tail_len_byte width
                        // can represent: at most 10^T - 1. The chosen layout
                        // must keep base + cap - 1 within this window so every
                        // possible in-range numeric has a uniform-width
                        // ASCII-decimal representation. Any insert beyond it
                        // takes the partition-split path into a sibling with a
                        // wider suffix_len.
                        std::uint64_t max_for_T = 1;
                        for (std::uint16_t i = 0; i < tail_len_byte; ++i) max_for_T *= 10;
                        --max_for_T;
                        if (n_max > max_for_T) goto ascii_decimal_skip; // extra widens past T
                        {
                            const std::uint64_t cap_for_T =
                                max_for_T - static_cast<std::uint64_t>(n_min) + 1;
                            const std::uint64_t fdl_cap_capped =
                                std::min<std::uint64_t>(kFdlMaxCapacity, cap_for_T);
                            const std::uint64_t sdl_cap_capped =
                                std::min<std::uint64_t>(kSdlMaxCapacity, cap_for_T);

                            if (range <= fdl_cap_capped &&
                                static_cast<std::uint64_t>(count) * kFdlDensityDenominator >=
                                    range * kFdlDensityNumerator)
                            {
                                LayoutChoice c{};
                                c.kind = LeafKind::kFullyDense;
                                c.base = n_min;
                                c.capacity = static_cast<std::uint16_t>(fdl_cap_capped);
                                c.suffix_len = tail_len_byte;
                                c.ref_key_len = static_cast<std::uint16_t>(common);
                                c.numeric_mode = NumericMode::kAsciiDecimal;
                                return c;
                            }
                            if (range <= sdl_cap_capped)
                            {
                                LayoutChoice c{};
                                c.kind = LeafKind::kSemiDense;
                                c.base = n_min;
                                c.capacity = static_cast<std::uint16_t>(sdl_cap_capped);
                                c.suffix_len = tail_len_byte;
                                c.ref_key_len = static_cast<std::uint16_t>(common);
                                c.numeric_mode = NumericMode::kAsciiDecimal;
                                return c;
                            }
                        }
                    ascii_decimal_skip: ;
                    }
                }
            }

            // 2b. Fall back to byte-BE mode (raw bytes interpreted as BE u32).
            if (tail_len_byte > 4) return fallback;
            const std::uint16_t ref_key_len = static_cast<std::uint16_t>(common);
            const std::uint16_t suffix_len  = tail_len_byte;

            const std::uint32_t n_min = extract_numeric_be4(first_suf.substr(common, suffix_len));
            const std::uint32_t n_max = extract_numeric_be4(src.keySuffix(static_cast<std::uint16_t>(hi - 1))
                                                                .substr(common, suffix_len));
            if (n_max < n_min) return fallback;
            const std::uint64_t range = static_cast<std::uint64_t>(n_max) - n_min + 1;

            if (range <= kFdlMaxCapacity &&
                static_cast<std::uint64_t>(count) * kFdlDensityDenominator >=
                    range * kFdlDensityNumerator)
            {
                LayoutChoice c{};
                c.kind = LeafKind::kFullyDense;
                c.base = n_min;
                c.capacity = kFdlMaxCapacity;
                c.suffix_len = suffix_len;
                c.ref_key_len = ref_key_len;
                c.numeric_mode = NumericMode::kByteBE;
                return c;
            }
            if (range <= kSdlMaxCapacity)
            {
                LayoutChoice c{};
                c.kind = LeafKind::kSemiDense;
                c.base = n_min;
                c.capacity = kSdlMaxCapacity;
                c.suffix_len = suffix_len;
                c.ref_key_len = ref_key_len;
                c.numeric_mode = NumericMode::kByteBE;
                return c;
            }
            return fallback;
        }

        // ---------------------------------------------------------------------
        // Free-function init helpers used by the materialize path.
        // ---------------------------------------------------------------------
        bool init_fdl_empty(SlottedPage& dst, std::string_view lf, std::string_view uf,
                            std::string_view ref_key, std::uint16_t suffix_len,
                            std::uint32_t base, std::uint16_t capacity,
                            NumericMode mode, NodeId link)
        {
            if (capacity == 0 || capacity > kFdlMaxCapacity) return false;
            // Mode-dependent suffix-length constraint: BE caps at 4 raw bytes;
            // ASCII-decimal at 9 digits (so the value fits in uint32).
            const std::uint16_t max_len = (mode == NumericMode::kAsciiDecimal) ? 9 : 4;
            if (suffix_len == 0 || suffix_len > max_len) return false;
            dst.init(NodeType::kLeaf, lf, uf);
            dst.setLeafKind(LeafKind::kFullyDense);
            dst.setLink(link);

            auto& hdr = *reinterpret_cast<SlottedPage::Header*>(dst.rawBytes());
            hdr.slot_count = 0;
            if (hdr.free_end < kFdlFencesBegin) return false;
            std::uint8_t* raw = dst.rawBytes();
            std::uint16_t ref_off = 0;
            std::uint16_t ref_len = 0;
            if (!ref_key.empty())
            {
                if (hdr.free_end < kFdlFencesBegin + ref_key.size()) return false;
                hdr.free_end = static_cast<std::uint16_t>(hdr.free_end - ref_key.size());
                ref_off = hdr.free_end;
                ref_len = static_cast<std::uint16_t>(ref_key.size());
                std::memcpy(raw + ref_off, ref_key.data(), ref_key.size());
            }
            std::memset(raw + kFdlBitmapOffset, 0, kFdlBitmapBytes);
            FdlMeta& meta = fdl_meta(raw);
            meta.capacity = capacity;
            meta.suffix_len = suffix_len;
            meta.ref_key_off = ref_off;
            meta.ref_key_len = ref_len;
            meta.base_numeric = base;
            meta.bitmap_off = static_cast<std::uint16_t>(kFdlBitmapOffset);
            meta.values_off = static_cast<std::uint16_t>(kFdlValuesOffset);
            meta.numeric_mode = static_cast<std::uint8_t>(mode);
            return true;
        }

        bool init_sdl_empty(SlottedPage& dst, std::string_view lf, std::string_view uf,
                            std::string_view ref_key, std::uint16_t suffix_len,
                            std::uint32_t base, std::uint16_t capacity,
                            NumericMode mode, NodeId link)
        {
            if (capacity == 0 || capacity > kSdlMaxCapacity) return false;
            dst.init(NodeType::kLeaf, lf, uf);
            dst.setLeafKind(LeafKind::kSemiDense);
            dst.setLink(link);

            auto& hdr = *reinterpret_cast<SlottedPage::Header*>(dst.rawBytes());
            hdr.slot_count = 0;
            const std::size_t offsets_bytes =
                static_cast<std::size_t>(capacity) * sizeof(std::uint16_t);
            const std::size_t heap_bottom = kSdlOffsetsOffset + offsets_bytes;
            if (hdr.free_end < heap_bottom) return false;
            std::uint8_t* raw = dst.rawBytes();
            std::uint16_t ref_off = 0;
            std::uint16_t ref_len = 0;
            if (!ref_key.empty())
            {
                if (hdr.free_end < heap_bottom + ref_key.size()) return false;
                hdr.free_end = static_cast<std::uint16_t>(hdr.free_end - ref_key.size());
                ref_off = hdr.free_end;
                ref_len = static_cast<std::uint16_t>(ref_key.size());
                std::memcpy(raw + ref_off, ref_key.data(), ref_key.size());
            }
            std::memset(raw + kSdlOffsetsOffset, 0, offsets_bytes);
            SdlMeta& meta = sdl_meta(raw);
            meta.capacity = capacity;
            meta.suffix_len = suffix_len;
            meta.ref_key_off = ref_off;
            meta.ref_key_len = ref_len;
            meta.base_numeric = base;
            meta.offsets_off = static_cast<std::uint16_t>(kSdlOffsetsOffset);
            meta.heap_bottom = static_cast<std::uint16_t>(heap_bottom);
            meta.numeric_mode = static_cast<std::uint8_t>(mode);
            return true;
        }

        // ---------------------------------------------------------------------
        // Materialize a slice [lo, hi) of `src` (a Comparison leaf) into `dst`
        // using the chosen layout. `dst` is freshly init-able. Fences and link
        // are passed by the caller.
        // ---------------------------------------------------------------------
        void materializeSliceInto(SlottedPage& dst, const SlottedPage& src,
                                  std::uint16_t lo, std::uint16_t hi,
                                  std::string_view lower_fence,
                                  std::string_view upper_fence,
                                  NodeId next_link,
                                  const LayoutChoice& choice)
        {
            const std::string_view src_prefix = src.prefixView();

            // Comparison materialization: rebuild slots in dst with possibly
            // shorter suffixes after the new prefix is computed.
            if (choice.kind == LeafKind::kComparison)
            {
                dst.init(NodeType::kLeaf, lower_fence, upper_fence);
                dst.setLeafKind(LeafKind::kComparison);
                dst.setLink(next_link);
                const std::size_t new_prefix_len = dst.prefixView().size();
                if (new_prefix_len < src_prefix.size())
                {
                    // Shouldn't happen: new prefix derives from fences derived
                    // from the source's fences and is always >= src prefix.
                    return;
                }
                const std::size_t eat = new_prefix_len - src_prefix.size();
                for (std::uint16_t i = lo; i < hi; ++i)
                {
                    const std::string_view old_suf = src.keySuffix(i);
                    const Value v = src.leafValue(i);
                    const std::string_view new_suf =
                        old_suf.size() > eat ? old_suf.substr(eat) : std::string_view{};
                    dst.appendLeaf(new_suf, v, make_head(new_suf));
                }
                dst.rebuildHints();
                return;
            }

            // FDL / SDL materialization. Both reuse the same shape probe.
            const std::string_view first_suf = src.keySuffix(lo);

            // Dry-run init the destination to learn the new prefix length.
            dst.init(NodeType::kLeaf, lower_fence, upper_fence);
            const std::size_t new_prefix_len = dst.prefixView().size();
            if (new_prefix_len < src_prefix.size()) return; // pathological
            const std::size_t eat = new_prefix_len - src_prefix.size();
            const std::string_view first_post_new_prefix =
                first_suf.size() > eat ? first_suf.substr(eat) : std::string_view{};

            // If the new prefix consumed past the ref_key boundary, fall back.
            if (eat > choice.ref_key_len)
            {
                LayoutChoice fallback{};
                materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                return;
            }
            const std::size_t new_ref_key_len = choice.ref_key_len - eat;
            const std::string_view new_ref_key(first_post_new_prefix.data(), new_ref_key_len);

            const bool ok = (choice.kind == LeafKind::kFullyDense)
                ? init_fdl_empty(dst, lower_fence, upper_fence, new_ref_key,
                                 choice.suffix_len, choice.base, choice.capacity,
                                 choice.numeric_mode, next_link)
                : init_sdl_empty(dst, lower_fence, upper_fence, new_ref_key,
                                 choice.suffix_len, choice.base, choice.capacity,
                                 choice.numeric_mode, next_link);
            if (!ok)
            {
                LayoutChoice fallback{};
                materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                return;
            }

            std::uint8_t* raw = dst.rawBytes();
            if (choice.kind == LeafKind::kFullyDense)
            {
                FdlMeta& meta = fdl_meta(raw);
                Value* vals = fdl_values(raw);
                auto& hdr = *reinterpret_cast<SlottedPage::Header*>(raw);
                for (std::uint16_t i = lo; i < hi; ++i)
                {
                    const std::string_view s = src.keySuffix(i);
                    if (s.size() < choice.ref_key_len + choice.suffix_len)
                    {
                        // Shape mismatch mid-build: bail to comparison.
                        LayoutChoice fallback{};
                        materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                        return;
                    }
                    const std::string_view tail = s.substr(choice.ref_key_len, choice.suffix_len);
                    std::uint32_t n_val = 0;
                    if (!read_numeric(tail, choice.numeric_mode, &n_val))
                    {
                        LayoutChoice fallback{};
                        materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                        return;
                    }
                    if (n_val < meta.base_numeric || n_val - meta.base_numeric >= meta.capacity)
                    {
                        LayoutChoice fallback{};
                        materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                        return;
                    }
                    const std::uint16_t slot = static_cast<std::uint16_t>(n_val - meta.base_numeric);
                    if (!fdl_bit_test(raw, slot))
                        hdr.slot_count = static_cast<std::uint16_t>(hdr.slot_count + 1);
                    fdl_bit_set(raw, slot);
                    vals[slot] = src.leafValue(i);
                }
            }
            else // SDL
            {
                SdlMeta& meta = sdl_meta(raw);
                std::uint16_t* offsets = sdl_offsets(raw);
                auto& hdr = *reinterpret_cast<SlottedPage::Header*>(raw);
                for (std::uint16_t i = lo; i < hi; ++i)
                {
                    const std::string_view s = src.keySuffix(i);
                    if (s.size() < choice.ref_key_len)
                    {
                        LayoutChoice fallback{};
                        materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                        return;
                    }
                    const std::string_view tail = s.substr(choice.ref_key_len);
                    std::uint32_t n_val = 0;
                    if (choice.numeric_mode == NumericMode::kAsciiDecimal)
                    {
                        if (!extract_decimal_numeric(tail.size() <= 9 ? tail : tail.substr(0, 9), &n_val))
                        {
                            LayoutChoice fallback{};
                            materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                            return;
                        }
                    }
                    else
                    {
                        n_val = sdl_numeric_of(tail);
                    }
                    if (n_val < meta.base_numeric || n_val - meta.base_numeric >= meta.capacity)
                    {
                        LayoutChoice fallback{};
                        materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                        return;
                    }
                    const std::uint16_t slot = static_cast<std::uint16_t>(n_val - meta.base_numeric);
                    const std::size_t need = sdl_payload_bytes(static_cast<std::uint16_t>(tail.size()));
                    if (hdr.free_end < meta.heap_bottom + need)
                    {
                        LayoutChoice fallback{};
                        materializeSliceInto(dst, src, lo, hi, lower_fence, upper_fence, next_link, fallback);
                        return;
                    }
                    hdr.free_end = static_cast<std::uint16_t>(hdr.free_end - need);
                    const std::uint16_t off = hdr.free_end;
                    const std::uint16_t suf_len_u16 = static_cast<std::uint16_t>(tail.size());
                    std::memcpy(raw + off, &suf_len_u16, sizeof(std::uint16_t));
                    if (!tail.empty())
                        std::memcpy(raw + off + sizeof(std::uint16_t), tail.data(), tail.size());
                    Value v = src.leafValue(i);
                    std::memcpy(raw + off + sizeof(std::uint16_t) + tail.size(), &v, sizeof(Value));
                    if (offsets[slot] == 0)
                        hdr.slot_count = static_cast<std::uint16_t>(hdr.slot_count + 1);
                    offsets[slot] = off;
                }
            }
        }
    } // anonymous namespace

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    void LeafNode::initEmpty(std::string_view lower_fence, std::string_view upper_fence, NodeId next_leaf)
    {
        page_.init(NodeType::kLeaf, lower_fence, upper_fence);
        page_.setLeafKind(LeafKind::kComparison);
        page_.setLink(next_leaf);
    }

    bool LeafNode::initFdlEmpty(std::string_view lower_fence, std::string_view upper_fence,
                                std::string_view ref_key, std::uint16_t suffix_len,
                                std::uint32_t base_numeric, std::uint16_t capacity,
                                NumericMode numeric_mode, NodeId next_leaf)
    {
        return init_fdl_empty(page_, lower_fence, upper_fence, ref_key,
                              suffix_len, base_numeric, capacity, numeric_mode, next_leaf);
    }

    bool LeafNode::initSdlEmpty(std::string_view lower_fence, std::string_view upper_fence,
                                std::string_view ref_key, std::uint16_t suffix_len,
                                std::uint32_t base_numeric, std::uint16_t capacity,
                                NumericMode numeric_mode, NodeId next_leaf)
    {
        return init_sdl_empty(page_, lower_fence, upper_fence, ref_key,
                              suffix_len, base_numeric, capacity, numeric_mode, next_leaf);
    }

    // -----------------------------------------------------------------------
    // Top-level dispatch
    // -----------------------------------------------------------------------

    bool LeafNode::tryInsert(std::string_view key, Value value, bool& inserted_new)
    {
        switch (kind())
        {
            case LeafKind::kComparison: return tryInsertCmp(key, value, inserted_new);
            case LeafKind::kFullyDense: return tryInsertFdl(key, value, inserted_new);
            case LeafKind::kSemiDense:  return tryInsertSdl(key, value, inserted_new);
        }
        return false;
    }

    std::optional<Value> LeafNode::find(std::string_view key) const
    {
        switch (kind())
        {
            case LeafKind::kComparison: return findCmp(key);
            case LeafKind::kFullyDense: return findFdl(key);
            case LeafKind::kSemiDense:  return findSdl(key);
        }
        return std::nullopt;
    }

    bool LeafNode::tryErase(std::string_view key)
    {
        switch (kind())
        {
            case LeafKind::kComparison: return tryEraseCmp(key);
            case LeafKind::kFullyDense: return tryEraseFdl(key);
            case LeafKind::kSemiDense:  return tryEraseSdl(key);
        }
        return false;
    }

    void LeafNode::collectScan(std::string_view start_key, std::size_t max_to_add,
                               std::vector<KeyValue>& results) const
    {
        switch (kind())
        {
            case LeafKind::kComparison: collectScanCmp(start_key, max_to_add, results); return;
            case LeafKind::kFullyDense: collectScanFdl(start_key, max_to_add, results); return;
            case LeafKind::kSemiDense:  collectScanSdl(start_key, max_to_add, results); return;
        }
    }

    std::uint16_t LeafNode::entryCount() const { return page_.slotCount(); }

    bool LeafNode::isUnderflow() const
    {
        switch (kind())
        {
            case LeafKind::kComparison:
            {
                const std::size_t used = kPageSizeBytes - page_.freeSpace();
                return used <= kMergeUnderflowBytes;
            }
            case LeafKind::kFullyDense:
            {
                const FdlMeta& meta = fdl_meta(page_.rawBytes());
                if (meta.capacity == 0) return true;
                return entryCount() * 4u < meta.capacity;
            }
            case LeafKind::kSemiDense:
            {
                const SdlMeta& meta = sdl_meta(page_.rawBytes());
                if (meta.capacity == 0) return true;
                return entryCount() * 4u < meta.capacity;
            }
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Comparison-leaf hot paths
    // -----------------------------------------------------------------------

    bool LeafNode::tryInsertCmp(std::string_view key, Value value, bool& inserted_new)
    {
        const std::size_t prefix_len = page_.prefixLen();
        const std::string_view suffix(key.data() + prefix_len, key.size() - prefix_len);
        const std::uint32_t head = make_head(suffix);

        const std::uint16_t n = page_.slotCount();
        if (__builtin_expect(n > 0, 1))
        {
            const std::uint16_t last_idx = static_cast<std::uint16_t>(n - 1);
            const std::uint32_t last_head = page_.headAt(last_idx);
            if (__builtin_expect(head > last_head ||
                    (head == last_head && suffix > page_.keySuffix(last_idx)), 1))
            {
                if (!page_.hasSpaceForLeaf(suffix.size())) return false;
                page_.appendLeaf(suffix, value, head);
                if ((n / (kHintCount + 1)) != ((n + 1) / (kHintCount + 1))) page_.rebuildHints();
                inserted_new = true;
                return true;
            }
        }
        const std::uint16_t idx = (n == 0) ? std::uint16_t{0} : page_.lowerBoundIndex(head, suffix);
        if (idx < n && page_.keySuffix(idx) == suffix)
        {
            page_.setLeafValue(idx, value);
            inserted_new = false;
            return true;
        }
        if (!page_.insertLeaf(idx, suffix, value, head)) return false;
        page_.rebuildHints();
        inserted_new = true;
        return true;
    }

    std::optional<Value> LeafNode::findCmp(std::string_view key) const
    {
        const std::size_t prefix_len = page_.prefixLen();
        if (key.size() < prefix_len) return std::nullopt;
        const std::string_view suffix(key.data() + prefix_len, key.size() - prefix_len);
        const std::uint32_t head = make_head(suffix);

        const std::uint16_t idx = page_.lowerBoundIndex(head, suffix);
        if (idx < page_.slotCount() && page_.keySuffix(idx) == suffix)
        {
            return page_.leafValue(idx);
        }
        return std::nullopt;
    }

    bool LeafNode::tryEraseCmp(std::string_view key)
    {
        const std::size_t prefix_len = page_.prefixLen();
        if (key.size() < prefix_len) return false;
        const std::string_view suffix(key.data() + prefix_len, key.size() - prefix_len);
        const std::uint32_t head = make_head(suffix);

        const std::uint16_t n = page_.slotCount();
        if (n == 0) return false;
        const std::uint16_t idx = page_.lowerBoundIndex(head, suffix);
        if (idx >= n || page_.keySuffix(idx) != suffix) return false;

        // Lazy delete: shift the slot directory down. Heap bytes become
        // unreachable garbage until the next split/merge rebuild compacts.
        auto& hdr = *reinterpret_cast<SlottedPage::Header*>(page_.rawBytes());
        auto* slots = reinterpret_cast<SlottedPage::LeafSlot*>(
            page_.rawBytes() + sizeof(SlottedPage::Header) + kHintCount * sizeof(std::uint32_t));
        if (idx + 1 < n)
        {
            std::memmove(slots + idx, slots + idx + 1,
                         (n - idx - 1) * sizeof(SlottedPage::LeafSlot));
        }
        hdr.slot_count = static_cast<std::uint16_t>(n - 1);
        hdr.free_begin = static_cast<std::uint16_t>(hdr.free_begin - sizeof(SlottedPage::LeafSlot));
        page_.rebuildHints();
        return true;
    }

    void LeafNode::collectScanCmp(std::string_view start_key, std::size_t max_to_add,
                                  std::vector<KeyValue>& results) const
    {
        const std::uint16_t n = page_.slotCount();
        if (n == 0 || max_to_add == 0) return;

        std::uint16_t i = 0;
        if (!start_key.empty())
        {
            const std::string_view prefix = page_.prefixView();
            if (start_key.size() < prefix.size())
            {
                const int cmp = start_key.compare(prefix.substr(0, start_key.size()));
                i = (cmp < 0) ? std::uint16_t{0} : n;
            }
            else
            {
                const int pcmp = std::memcmp(start_key.data(), prefix.data(), prefix.size());
                if (pcmp < 0) i = 0;
                else if (pcmp > 0) i = n;
                else
                {
                    const std::string_view suffix = start_key.substr(prefix.size());
                    i = page_.lowerBoundIndex(make_head(suffix), suffix);
                }
            }
        }

        const std::string_view prefix = page_.prefixView();
        for (; i < n && results.size() < max_to_add; ++i)
        {
            std::string out;
            const std::string_view suf = page_.keySuffix(i);
            out.reserve(prefix.size() + suf.size());
            out.append(prefix);
            out.append(suf);
            results.push_back(KeyValue{std::move(out), page_.leafValue(i)});
        }
    }

    std::string LeafNode::cmpKeyAt(std::uint16_t i) const
    {
        const std::string_view prefix = page_.prefixView();
        const std::string_view suffix = page_.keySuffix(i);
        std::string out;
        out.reserve(prefix.size() + suffix.size());
        out.append(prefix);
        out.append(suffix);
        return out;
    }

    // -----------------------------------------------------------------------
    // Fully Dense Leaf hot paths
    // -----------------------------------------------------------------------

    bool LeafNode::tryInsertFdl(std::string_view key, Value value, bool& inserted_new)
    {
        std::uint8_t* raw = page_.rawBytes();
        FdlMeta& meta = fdl_meta(raw);
        std::string_view suffix;
        if (!fdl_strip_to_suffix(key, page_, meta, raw, suffix)) return false;

        std::uint32_t n_val = 0;
        if (!read_numeric(suffix, static_cast<NumericMode>(meta.numeric_mode), &n_val))
            return false;
        if (n_val < meta.base_numeric) return false;
        const std::uint64_t slot64 = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
        if (slot64 >= meta.capacity) return false;
        const std::uint16_t slot = static_cast<std::uint16_t>(slot64);

        if (fdl_bit_test(raw, slot))
        {
            fdl_values(raw)[slot] = value;
            inserted_new = false;
            return true;
        }
        fdl_bit_set(raw, slot);
        fdl_values(raw)[slot] = value;
        auto& hdr = *reinterpret_cast<SlottedPage::Header*>(raw);
        hdr.slot_count = static_cast<std::uint16_t>(hdr.slot_count + 1);
        inserted_new = true;
        return true;
    }

    std::optional<Value> LeafNode::findFdl(std::string_view key) const
    {
        const std::uint8_t* raw = page_.rawBytes();
        const FdlMeta& meta = fdl_meta(raw);
        std::string_view suffix;
        if (!fdl_strip_to_suffix(key, page_, meta, raw, suffix)) return std::nullopt;

        std::uint32_t n_val = 0;
        if (!read_numeric(suffix, static_cast<NumericMode>(meta.numeric_mode), &n_val))
            return std::nullopt;
        if (n_val < meta.base_numeric) return std::nullopt;
        const std::uint64_t slot64 = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
        if (slot64 >= meta.capacity) return std::nullopt;
        const std::uint16_t slot = static_cast<std::uint16_t>(slot64);

        if (!fdl_bit_test(raw, slot)) return std::nullopt;
        return fdl_values(raw)[slot];
    }

    bool LeafNode::tryEraseFdl(std::string_view key)
    {
        std::uint8_t* raw = page_.rawBytes();
        FdlMeta& meta = fdl_meta(raw);
        std::string_view suffix;
        if (!fdl_strip_to_suffix(key, page_, meta, raw, suffix)) return false;

        std::uint32_t n_val = 0;
        if (!read_numeric(suffix, static_cast<NumericMode>(meta.numeric_mode), &n_val))
            return false;
        if (n_val < meta.base_numeric) return false;
        const std::uint64_t slot64 = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
        if (slot64 >= meta.capacity) return false;
        const std::uint16_t slot = static_cast<std::uint16_t>(slot64);

        if (!fdl_bit_test(raw, slot)) return false;
        fdl_bit_clear(raw, slot);
        auto& hdr = *reinterpret_cast<SlottedPage::Header*>(raw);
        hdr.slot_count = static_cast<std::uint16_t>(hdr.slot_count - 1);
        return true;
    }

    void LeafNode::collectScanFdl(std::string_view start_key, std::size_t max_to_add,
                                  std::vector<KeyValue>& results) const
    {
        if (max_to_add == 0) return;
        const std::uint8_t* raw = page_.rawBytes();
        const FdlMeta& meta = fdl_meta(raw);
        if (meta.capacity == 0) return;

        const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);

        // Determine the numeric slot to start from.
        std::uint16_t start_slot = 0;
        if (!start_key.empty())
        {
            std::string_view suffix;
            if (fdl_strip_to_suffix(start_key, page_, meta, raw, suffix))
            {
                std::uint32_t n_val = 0;
                if (read_numeric(suffix, mode, &n_val) && n_val >= meta.base_numeric)
                {
                    const std::uint64_t s = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
                    if (s >= meta.capacity) return;
                    start_slot = static_cast<std::uint16_t>(s);
                }
            }
            else
            {
                const std::string_view uf = page_.upperFenceView();
                if (!uf.empty() && start_key >= uf) return;
                // start_key sorts before this leaf -> emit from beginning.
            }
        }

        const std::string_view prefix = page_.prefixView();
        const std::string_view ref_key(reinterpret_cast<const char*>(raw + meta.ref_key_off),
                                       meta.ref_key_len);
        const std::uint8_t* bm = fdl_bitmap(raw);
        const Value* vals = fdl_values(raw);
        const std::uint16_t cap = meta.capacity;
        const std::uint16_t suffix_len = meta.suffix_len;
        const std::uint32_t base = meta.base_numeric;

        // Iterate set bits using uint64_t words and __builtin_ctzll for speed.
        const std::uint16_t end_word = static_cast<std::uint16_t>((cap + 63) / 64);
        std::uint16_t word_idx = static_cast<std::uint16_t>(start_slot / 64);
        std::uint64_t word = 0;
        if (word_idx < end_word)
        {
            const std::size_t offset_in_bm = static_cast<std::size_t>(word_idx) * 8;
            const std::size_t to_copy = std::min<std::size_t>(8, kFdlBitmapBytes - offset_in_bm);
            std::memcpy(&word, bm + offset_in_bm, to_copy);
            const std::uint16_t bit_in_word = static_cast<std::uint16_t>(start_slot & 63);
            if (bit_in_word > 0)
                word &= ~((std::uint64_t{1} << bit_in_word) - 1);
        }

        while (word_idx < end_word && results.size() < max_to_add)
        {
            while (word != 0 && results.size() < max_to_add)
            {
                const int b = __builtin_ctzll(word);
                const std::uint16_t slot = static_cast<std::uint16_t>(word_idx * 64 + b);
                if (slot >= cap) { word = 0; break; }

                std::string out;
                out.reserve(prefix.size() + ref_key.size() + suffix_len);
                out.append(prefix);
                out.append(ref_key);
                char tail[16] = {0};
                write_numeric(tail, base + slot, suffix_len, mode);
                out.append(tail, suffix_len);
                results.push_back(KeyValue{std::move(out), vals[slot]});

                word &= word - 1;
            }
            ++word_idx;
            if (word_idx < end_word)
            {
                word = 0;
                const std::size_t offset_in_bm = static_cast<std::size_t>(word_idx) * 8;
                const std::size_t to_copy = std::min<std::size_t>(8, kFdlBitmapBytes - offset_in_bm);
                std::memcpy(&word, bm + offset_in_bm, to_copy);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Semi Dense Leaf hot paths
    // -----------------------------------------------------------------------

    bool LeafNode::tryInsertSdl(std::string_view key, Value value, bool& inserted_new)
    {
        std::uint8_t* raw = page_.rawBytes();
        SdlMeta& meta = sdl_meta(raw);
        std::string_view suffix;
        if (!sdl_strip_to_suffix(key, page_, meta, raw, suffix)) return false;

        const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);
        std::uint32_t n_val = 0;
        if (mode == NumericMode::kAsciiDecimal)
        {
            if (suffix.empty()) return false;
            const std::string_view tail = suffix.size() <= 9 ? suffix : suffix.substr(0, 9);
            if (!extract_decimal_numeric(tail, &n_val)) return false;
        }
        else
        {
            n_val = sdl_numeric_of(suffix);
        }
        if (n_val < meta.base_numeric) return false;
        const std::uint64_t slot64 = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
        if (slot64 >= meta.capacity) return false;
        const std::uint16_t slot = static_cast<std::uint16_t>(slot64);

        std::uint16_t* offsets = sdl_offsets(raw);
        const std::uint16_t cur_off = offsets[slot];
        if (cur_off != 0)
        {
            std::uint16_t cur_len = 0;
            std::memcpy(&cur_len, raw + cur_off, sizeof(std::uint16_t));
            if (cur_len == suffix.size())
            {
                std::memcpy(raw + cur_off + sizeof(std::uint16_t) + cur_len, &value, sizeof(Value));
                inserted_new = false;
                return true;
            }
        }

        auto& hdr = *reinterpret_cast<SlottedPage::Header*>(raw);
        const std::size_t need = sdl_payload_bytes(static_cast<std::uint16_t>(suffix.size()));
        if (hdr.free_end < meta.heap_bottom + need) return false;
        hdr.free_end = static_cast<std::uint16_t>(hdr.free_end - need);
        const std::uint16_t new_off = hdr.free_end;

        const std::uint16_t suf_len_u16 = static_cast<std::uint16_t>(suffix.size());
        std::memcpy(raw + new_off, &suf_len_u16, sizeof(std::uint16_t));
        if (!suffix.empty())
            std::memcpy(raw + new_off + sizeof(std::uint16_t), suffix.data(), suffix.size());
        std::memcpy(raw + new_off + sizeof(std::uint16_t) + suffix.size(), &value, sizeof(Value));

        offsets[slot] = new_off;
        if (cur_off == 0)
        {
            hdr.slot_count = static_cast<std::uint16_t>(hdr.slot_count + 1);
            inserted_new = true;
        }
        else
        {
            inserted_new = false;
        }
        return true;
    }

    std::optional<Value> LeafNode::findSdl(std::string_view key) const
    {
        const std::uint8_t* raw = page_.rawBytes();
        const SdlMeta& meta = sdl_meta(raw);
        std::string_view suffix;
        if (!sdl_strip_to_suffix(key, page_, meta, raw, suffix)) return std::nullopt;

        const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);
        std::uint32_t n_val = 0;
        if (mode == NumericMode::kAsciiDecimal)
        {
            if (suffix.empty()) return std::nullopt;
            const std::string_view tail = suffix.size() <= 9 ? suffix : suffix.substr(0, 9);
            if (!extract_decimal_numeric(tail, &n_val)) return std::nullopt;
        }
        else
        {
            n_val = sdl_numeric_of(suffix);
        }
        if (n_val < meta.base_numeric) return std::nullopt;
        const std::uint64_t slot64 = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
        if (slot64 >= meta.capacity) return std::nullopt;
        const std::uint16_t slot = static_cast<std::uint16_t>(slot64);

        const std::uint16_t* offsets = sdl_offsets(raw);
        const std::uint16_t off = offsets[slot];
        if (off == 0) return std::nullopt;

        std::uint16_t stored_len = 0;
        std::memcpy(&stored_len, raw + off, sizeof(std::uint16_t));
        if (stored_len != suffix.size()) return std::nullopt;
        if (stored_len > 0 &&
            std::memcmp(raw + off + sizeof(std::uint16_t), suffix.data(), stored_len) != 0)
            return std::nullopt;
        Value v = 0;
        std::memcpy(&v, raw + off + sizeof(std::uint16_t) + stored_len, sizeof(Value));
        return v;
    }

    bool LeafNode::tryEraseSdl(std::string_view key)
    {
        std::uint8_t* raw = page_.rawBytes();
        SdlMeta& meta = sdl_meta(raw);
        std::string_view suffix;
        if (!sdl_strip_to_suffix(key, page_, meta, raw, suffix)) return false;

        const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);
        std::uint32_t n_val = 0;
        if (mode == NumericMode::kAsciiDecimal)
        {
            if (suffix.empty()) return false;
            const std::string_view tail = suffix.size() <= 9 ? suffix : suffix.substr(0, 9);
            if (!extract_decimal_numeric(tail, &n_val)) return false;
        }
        else
        {
            n_val = sdl_numeric_of(suffix);
        }
        if (n_val < meta.base_numeric) return false;
        const std::uint64_t slot64 = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
        if (slot64 >= meta.capacity) return false;
        const std::uint16_t slot = static_cast<std::uint16_t>(slot64);

        std::uint16_t* offsets = sdl_offsets(raw);
        const std::uint16_t off = offsets[slot];
        if (off == 0) return false;
        std::uint16_t stored_len = 0;
        std::memcpy(&stored_len, raw + off, sizeof(std::uint16_t));
        if (stored_len != suffix.size()) return false;
        if (stored_len > 0 &&
            std::memcmp(raw + off + sizeof(std::uint16_t), suffix.data(), stored_len) != 0)
            return false;
        offsets[slot] = 0;
        auto& hdr = *reinterpret_cast<SlottedPage::Header*>(raw);
        hdr.slot_count = static_cast<std::uint16_t>(hdr.slot_count - 1);
        return true;
    }

    void LeafNode::collectScanSdl(std::string_view start_key, std::size_t max_to_add,
                                  std::vector<KeyValue>& results) const
    {
        if (max_to_add == 0) return;
        const std::uint8_t* raw = page_.rawBytes();
        const SdlMeta& meta = sdl_meta(raw);
        if (meta.capacity == 0) return;

        const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);
        std::uint16_t start_slot = 0;
        if (!start_key.empty())
        {
            std::string_view suffix;
            if (sdl_strip_to_suffix(start_key, page_, meta, raw, suffix))
            {
                std::uint32_t n_val = 0;
                bool ok = false;
                if (mode == NumericMode::kAsciiDecimal)
                {
                    if (!suffix.empty())
                    {
                        const std::string_view tail = suffix.size() <= 9 ? suffix : suffix.substr(0, 9);
                        ok = extract_decimal_numeric(tail, &n_val);
                    }
                }
                else
                {
                    n_val = sdl_numeric_of(suffix);
                    ok = true;
                }
                if (ok && n_val >= meta.base_numeric)
                {
                    const std::uint64_t s = static_cast<std::uint64_t>(n_val) - meta.base_numeric;
                    if (s >= meta.capacity) return;
                    start_slot = static_cast<std::uint16_t>(s);
                }
            }
            else
            {
                const std::string_view uf = page_.upperFenceView();
                if (!uf.empty() && start_key >= uf) return;
            }
        }

        const std::string_view prefix = page_.prefixView();
        const std::string_view ref_key(reinterpret_cast<const char*>(raw + meta.ref_key_off),
                                       meta.ref_key_len);
        const std::uint16_t* offsets = sdl_offsets(raw);
        const std::uint16_t cap = meta.capacity;
        for (std::uint16_t slot = start_slot; slot < cap && results.size() < max_to_add; ++slot)
        {
            const std::uint16_t off = offsets[slot];
            if (off == 0) continue;
            std::uint16_t stored_len = 0;
            std::memcpy(&stored_len, raw + off, sizeof(std::uint16_t));
            std::string out;
            out.reserve(prefix.size() + ref_key.size() + stored_len);
            out.append(prefix);
            out.append(ref_key);
            if (stored_len > 0)
                out.append(reinterpret_cast<const char*>(raw + off + sizeof(std::uint16_t)),
                           stored_len);
            Value v = 0;
            std::memcpy(&v, raw + off + sizeof(std::uint16_t) + stored_len, sizeof(Value));

            if (!start_key.empty() && out < start_key) continue;

            results.push_back(KeyValue{std::move(out), v});
        }
    }

    // -----------------------------------------------------------------------
    // Split (Phase 3 + Phase 5: kind-aware with adaptive promotion)
    // -----------------------------------------------------------------------

    LeafSplitResult LeafNode::splitInto(LeafNode& right_node, NodeId right_id,
                                        std::string_view inserting_full_key)
    {
        SlottedPage& src = page_;
        thread_local std::string lower_buf;
        thread_local std::string upper_buf;
        lower_buf.assign(src.lowerFenceView());
        upper_buf.assign(src.upperFenceView());
        const bool had_lower = src.hasLowerFence();
        const bool had_upper = src.hasUpperFence();
        const NodeId src_next = src.link();
        const std::string_view src_lower = had_lower ? std::string_view(lower_buf) : std::string_view{};
        const std::string_view src_upper = had_upper ? std::string_view(upper_buf) : std::string_view{};

        // ------------------------------------------------------------------
        // FDL: out-of-range partition split (no rebuild). Otherwise demote.
        // ------------------------------------------------------------------
        if (src.leafKind() == LeafKind::kFullyDense)
        {
            const FdlMeta& meta = fdl_meta(src.rawBytes());
            const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);
            const std::uint64_t base = meta.base_numeric;
            const std::uint64_t end_numeric = base + meta.capacity;

            std::string_view ins_suffix;
            const bool shape_ok = (!inserting_full_key.empty()) &&
                fdl_strip_to_suffix(inserting_full_key, src, meta, src.rawBytes(), ins_suffix);
            std::uint32_t ins_num = 0;
            bool num_ok = false;
            if (shape_ok) num_ok = read_numeric(ins_suffix, mode, &ins_num);

            // For ASCII-decimal mode we must accept partition splits even when
            // the inserting key has a *different* shape than the source FDL,
            // because a decimal carry (999 -> 1000) widens the suffix. Detect
            // this by stripping only the source's static prefix and parsing
            // the post-prefix bytes as decimal.
            std::uint32_t ins_num_loose = 0;
            bool ins_num_loose_ok = false;
            if (mode == NumericMode::kAsciiDecimal && !inserting_full_key.empty())
            {
                const std::size_t pfx_len = src.prefixLen();
                if (inserting_full_key.size() >= pfx_len)
                {
                    const std::string_view post(inserting_full_key.data() + pfx_len,
                                                inserting_full_key.size() - pfx_len);
                    const std::string_view t = post.size() <= 9 ? post : post.substr(0, 9);
                    ins_num_loose_ok = extract_decimal_numeric(t, &ins_num_loose);
                }
            }

            const bool tight_ok = (shape_ok && num_ok && ins_num >= end_numeric);
            const bool loose_ok = (mode == NumericMode::kAsciiDecimal && ins_num_loose_ok &&
                                   ins_num_loose >= end_numeric);
            if ((tight_ok || loose_ok) && end_numeric <= 0xFFFFFFFFu)
            {
                const std::uint32_t in = tight_ok ? ins_num : ins_num_loose;
                std::uint32_t new_base = static_cast<std::uint32_t>(end_numeric);
                if (in >= new_base + kFdlMaxCapacity)
                    new_base = in;

                // Recompute the new sibling's ref_key/suffix_len so that all
                // keys in [new_base, new_base+kFdlMaxCapacity) share a common
                // post-prefix prefix and have a uniform decimal/byte-tail width.
                // For ASCII-decimal mode this handles digit-carry boundaries
                // where the source FDL's ref_key/suffix_len no longer fit.
                const std::string_view src_prefix = src.prefixView();
                std::string new_ref_key;
                std::uint16_t new_suffix_len = meta.suffix_len;
                std::uint64_t new_cap64 = kFdlMaxCapacity;
                if (static_cast<std::uint64_t>(new_base) + kFdlMaxCapacity > 0xFFFFFFFFu)
                {
                    new_cap64 = 0xFFFFFFFFu - new_base;
                }
                if (new_cap64 == 0)
                {
                    goto fdl_partition_fail;
                }
                if (mode == NumericMode::kAsciiDecimal)
                {
                    // Use enough digits for new_base+new_cap-1 (the highest
                    // numeric the new sibling can ever hold).
                    const std::uint32_t new_max = static_cast<std::uint32_t>(new_base + new_cap64 - 1);
                    std::uint16_t digits = 1;
                    std::uint32_t threshold = 10;
                    while (new_max >= threshold && digits < 9)
                    {
                        ++digits;
                        if (threshold > 0xFFFFFFFFu / 10) break;
                        threshold *= 10;
                    }
                    // The full key has total length = src_prefix.size() + (some extra).
                    // For zero-padded ASCII, all keys have the same total length;
                    // we infer it from the inserting key (a real key of the workload).
                    const std::size_t total_len = inserting_full_key.size();
                    if (total_len < src_prefix.size() + digits) goto fdl_partition_fail;
                    const std::size_t new_ref_key_len = total_len - src_prefix.size() - digits;
                    new_suffix_len = digits;
                    if (new_ref_key_len > 0)
                    {
                        new_ref_key.assign(total_len - src_prefix.size() - digits, '0');
                    }
                }
                else
                {
                    // Byte-BE mode: reuse src's ref_key bytes (uniform suffix).
                    new_ref_key.assign(reinterpret_cast<const char*>(src.rawBytes() + meta.ref_key_off),
                                       meta.ref_key_len);
                }

                std::string sep;
                sep.reserve(src_prefix.size() + new_ref_key.size() + new_suffix_len);
                sep.append(src_prefix);
                sep.append(new_ref_key);
                char tail[16] = {0};
                write_numeric(tail, new_base, new_suffix_len, mode);
                sep.append(tail, new_suffix_len);

                // The new sibling's actual page prefix == LCP(sep, src_upper).
                // If that prefix extends past src_prefix + new_ref_key (i.e.
                // into the numeric tail), the chosen ref_key/suffix_len shape
                // is unrepresentable with these fences. Bail to demote+median
                // so chooseLayoutCmp can pick a shape that respects the
                // tighter post-fence prefix.
                const std::size_t actual_prefix =
                    src_upper.empty() ? 0 : longest_common_prefix(sep, src_upper);
                if (actual_prefix > src_prefix.size() + new_ref_key.size())
                    goto fdl_partition_fail;

                const std::uint16_t new_cap = static_cast<std::uint16_t>(new_cap64);
                if (new_cap > 0)
                {
                    const bool ok = right_node.initFdlEmpty(sep, src_upper, new_ref_key,
                                                            new_suffix_len,
                                                            new_base, new_cap, mode, src_next);
                    if (ok)
                    {
                        src.setLink(right_id);
                        return LeafSplitResult{std::move(sep), /*right_is_new_partition=*/true};
                    }
                }
            }
        fdl_partition_fail:
            ; // fall through to demote+median
        }

        // ------------------------------------------------------------------
        // SDL: out-of-range partition split. Otherwise demote.
        // ------------------------------------------------------------------
        if (src.leafKind() == LeafKind::kSemiDense)
        {
            const SdlMeta& meta = sdl_meta(src.rawBytes());
            const NumericMode mode = static_cast<NumericMode>(meta.numeric_mode);
            const std::uint64_t base = meta.base_numeric;
            const std::uint64_t end_numeric = base + meta.capacity;

            std::string_view ins_suffix;
            const bool shape_ok = (!inserting_full_key.empty()) &&
                sdl_strip_to_suffix(inserting_full_key, src, meta, src.rawBytes(), ins_suffix);
            std::uint32_t ins_num = 0;
            bool num_ok = false;
            if (shape_ok)
            {
                if (mode == NumericMode::kAsciiDecimal)
                {
                    if (!ins_suffix.empty())
                    {
                        const std::string_view t = ins_suffix.size() <= 9 ? ins_suffix : ins_suffix.substr(0, 9);
                        num_ok = extract_decimal_numeric(t, &ins_num);
                    }
                }
                else
                {
                    ins_num = sdl_numeric_of(ins_suffix);
                    num_ok = true;
                }
            }

            // ASCII-decimal: also try a "loose" parse where we strip only the
            // static prefix and parse post-prefix bytes as decimal. This lets
            // us recover from digit-carry boundaries (the source SDL had
            // suffix_len=3, the inserting key needs suffix_len=4).
            std::uint32_t ins_num_loose = 0;
            bool ins_num_loose_ok = false;
            if (mode == NumericMode::kAsciiDecimal && !inserting_full_key.empty())
            {
                const std::size_t pfx_len = src.prefixLen();
                if (inserting_full_key.size() >= pfx_len)
                {
                    const std::string_view post(inserting_full_key.data() + pfx_len,
                                                inserting_full_key.size() - pfx_len);
                    const std::string_view t = post.size() <= 9 ? post : post.substr(0, 9);
                    ins_num_loose_ok = extract_decimal_numeric(t, &ins_num_loose);
                }
            }

            const bool tight_ok = (shape_ok && num_ok && ins_num >= end_numeric);
            const bool loose_ok = (mode == NumericMode::kAsciiDecimal && ins_num_loose_ok &&
                                   ins_num_loose >= end_numeric);
            if ((tight_ok || loose_ok) && end_numeric <= 0xFFFFFFFFu)
            {
                const std::uint32_t in = tight_ok ? ins_num : ins_num_loose;
                std::uint32_t new_base = static_cast<std::uint32_t>(end_numeric);
                if (in >= new_base + kSdlMaxCapacity)
                    new_base = in;

                const std::string_view src_prefix = src.prefixView();
                std::string new_ref_key;
                std::uint16_t new_suffix_len = meta.suffix_len;
                std::uint64_t new_cap64 = kSdlMaxCapacity;
                if (static_cast<std::uint64_t>(new_base) + kSdlMaxCapacity > 0xFFFFFFFFu)
                {
                    new_cap64 = 0xFFFFFFFFu - new_base;
                }
                if (new_cap64 == 0) goto sdl_partition_fail;

                if (mode == NumericMode::kAsciiDecimal)
                {
                    const std::uint32_t new_max = static_cast<std::uint32_t>(new_base + new_cap64 - 1);
                    const std::uint16_t digits = count_decimal_digits(new_max);
                    // Re-cap so [new_base, new_base+cap) fits within 10^digits.
                    std::uint64_t max_for_digits = 1;
                    for (std::uint16_t i = 0; i < digits; ++i) max_for_digits *= 10;
                    --max_for_digits;
                    if (static_cast<std::uint64_t>(new_base) > max_for_digits) goto sdl_partition_fail;
                    const std::uint64_t cap_for_digits =
                        max_for_digits - static_cast<std::uint64_t>(new_base) + 1;
                    if (cap_for_digits < new_cap64) new_cap64 = cap_for_digits;

                    const std::size_t total_len = inserting_full_key.size();
                    if (total_len < src_prefix.size() + digits) goto sdl_partition_fail;
                    const std::size_t new_ref_key_len = total_len - src_prefix.size() - digits;
                    new_suffix_len = digits;
                    if (new_ref_key_len > 0)
                        new_ref_key.assign(new_ref_key_len, '0');
                }
                else
                {
                    new_ref_key.assign(reinterpret_cast<const char*>(src.rawBytes() + meta.ref_key_off),
                                       meta.ref_key_len);
                }

                std::string sep;
                sep.reserve(src_prefix.size() + new_ref_key.size() + new_suffix_len);
                sep.append(src_prefix);
                sep.append(new_ref_key);
                char tail[16] = {0};
                write_numeric(tail, new_base, new_suffix_len, mode);
                sep.append(tail, new_suffix_len);

                // Bail if the post-fence prefix would extend past the chosen
                // ref_key boundary; see fdl partition split for details.
                const std::size_t actual_prefix =
                    src_upper.empty() ? 0 : longest_common_prefix(sep, src_upper);
                if (actual_prefix > src_prefix.size() + new_ref_key.size())
                    goto sdl_partition_fail;

                const std::uint16_t new_cap = static_cast<std::uint16_t>(new_cap64);
                if (new_cap > 0)
                {
                    const bool ok = right_node.initSdlEmpty(sep, src_upper, new_ref_key,
                                                            new_suffix_len,
                                                            new_base, new_cap, mode, src_next);
                    if (ok)
                    {
                        src.setLink(right_id);
                        return LeafSplitResult{std::move(sep), /*right_is_new_partition=*/true};
                    }
                }
            }
        sdl_partition_fail:
            ; // fall through to demote+median
        }

        // ------------------------------------------------------------------
        // Comparison-style split. Demote FDL/SDL first if needed.
        // ------------------------------------------------------------------
        const bool needs_demote = (src.leafKind() != LeafKind::kComparison);
        if (needs_demote)
        {
            std::vector<KeyValue> entries;
            entries.reserve(512);
            collectScan(std::string_view{}, std::numeric_limits<std::size_t>::max(), entries);

            // Two-step demote: when a Cmp page can't hold all entries (which
            // is common for FDLs with high capacity and entry count), do a
            // *direct* median split into the two destination pages without
            // routing through a single Cmp page. The right-half goes into
            // tls_scratch first; the left-half into tls_scratch_aux.

            // Decide split point by entry count (numeric-sort is preserved
            // because collectScan emits ascending).
            const std::size_t n_total = entries.size();
            if (n_total < 2)
            {
                return LeafSplitResult{};
            }

            // Try the simple "rebuild as one comparison leaf" path first; if
            // it fits, we proceed with the standard median split below. This
            // gives chooseLayoutCmp a chance to re-promote.
            tls_scratch.init(NodeType::kLeaf, src_lower, src_upper);
            tls_scratch.setLeafKind(LeafKind::kComparison);
            tls_scratch.setLink(src_next);
            const std::size_t new_prefix_len = tls_scratch.prefixView().size();
            bool fits_in_one = true;
            for (const auto& kv : entries)
            {
                if (kv.key.size() < new_prefix_len) { fits_in_one = false; break; }
                const std::string_view suf(kv.key.data() + new_prefix_len,
                                           kv.key.size() - new_prefix_len);
                if (!tls_scratch.hasSpaceForLeaf(suf.size())) { fits_in_one = false; break; }
                tls_scratch.appendLeaf(suf, kv.value, make_head(suf));
            }

            if (fits_in_one)
            {
                tls_scratch.rebuildHints();
                std::memcpy(src.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);
            }
            else
            {
                // Direct two-leaf split: the median entry's key becomes the
                // separator. Build the right half first, then the left half;
                // commit both at the end.
                const std::size_t mid_e = n_total / 2;
                std::string sep_full = entries[mid_e].key;

                // Build right (Comparison; chooseLayoutCmp can re-promote later
                // via materializeSliceInto, but here we go straight to Cmp).
                tls_scratch.init(NodeType::kLeaf, sep_full, src_upper);
                tls_scratch.setLeafKind(LeafKind::kComparison);
                tls_scratch.setLink(src_next);
                const std::size_t r_prefix_len = tls_scratch.prefixView().size();
                for (std::size_t i = mid_e; i < n_total; ++i)
                {
                    const auto& kv = entries[i];
                    if (kv.key.size() < r_prefix_len) return LeafSplitResult{};
                    const std::string_view suf(kv.key.data() + r_prefix_len,
                                               kv.key.size() - r_prefix_len);
                    if (!tls_scratch.hasSpaceForLeaf(suf.size())) return LeafSplitResult{};
                    tls_scratch.appendLeaf(suf, kv.value, make_head(suf));
                }
                tls_scratch.rebuildHints();
                std::memcpy(right_node.page_.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

                // Build left (Comparison) into tls_scratch_aux, link to right_id.
                tls_scratch_aux.init(NodeType::kLeaf, src_lower, sep_full);
                tls_scratch_aux.setLeafKind(LeafKind::kComparison);
                tls_scratch_aux.setLink(right_id);
                const std::size_t l_prefix_len = tls_scratch_aux.prefixView().size();
                for (std::size_t i = 0; i < mid_e; ++i)
                {
                    const auto& kv = entries[i];
                    if (kv.key.size() < l_prefix_len) return LeafSplitResult{};
                    const std::string_view suf(kv.key.data() + l_prefix_len,
                                               kv.key.size() - l_prefix_len);
                    if (!tls_scratch_aux.hasSpaceForLeaf(suf.size())) return LeafSplitResult{};
                    tls_scratch_aux.appendLeaf(suf, kv.value, make_head(suf));
                }
                tls_scratch_aux.rebuildHints();
                std::memcpy(src.rawBytes(), tls_scratch_aux.rawBytes(), kPageSizeBytes);

                return LeafSplitResult{std::move(sep_full), /*right_is_new_partition=*/false};
            }
        }

        const std::uint16_t n = src.slotCount();
        if (n < 2)
        {
            // Cannot split a single-entry leaf. Caller should handle this by
            // creating a fresh empty sibling.
            return LeafSplitResult{};
        }

        // Pick split index by accumulated payload bytes so variable-length
        // keys are balanced.
        std::size_t total = 0;
        for (std::uint16_t i = 0; i < n; ++i)
            total += src.keySuffix(i).size() + sizeof(Value);

        std::uint16_t mid = static_cast<std::uint16_t>(n / 2);
        if (mid == 0) mid = 1;
        if (mid >= n) mid = static_cast<std::uint16_t>(n - 1);
        {
            const std::size_t target = total / 2;
            std::size_t acc = 0;
            for (std::uint16_t i = 0; i < n; ++i)
            {
                acc += src.keySuffix(i).size() + sizeof(Value);
                if (acc >= target)
                {
                    mid = (i == 0) ? std::uint16_t{1} : i;
                    break;
                }
            }
        }
        if (mid == 0) mid = 1;
        if (mid >= n) mid = static_cast<std::uint16_t>(n - 1);

        const std::string_view src_prefix2 = src.prefixView();
        const std::string_view last_left_suffix = src.keySuffix(static_cast<std::uint16_t>(mid - 1));
        const std::string_view first_right_suffix = src.keySuffix(mid);
        const std::size_t suf_common = longest_common_prefix(last_left_suffix, first_right_suffix);

        std::string separator;
        const std::size_t sep_take = std::min(first_right_suffix.size(), suf_common + 1);
        separator.reserve(src_prefix2.size() + sep_take);
        separator.append(src_prefix2);
        separator.append(first_right_suffix.data(), sep_take);

        // Determine which half the inserting key will land in so we can fold
        // its post-prefix bytes into chooseLayoutCmp on that side. Prevents
        // the chosen layout from being too tight to fit the inserting key
        // (which would cause `tryInsert` to fail post-split).
        std::string_view inserting_post_prefix;
        const bool ins_to_right = std::string_view(inserting_full_key) >= std::string_view(separator);
        if (!inserting_full_key.empty() && inserting_full_key.size() >= src_prefix2.size())
        {
            inserting_post_prefix = std::string_view(
                inserting_full_key.data() + src_prefix2.size(),
                inserting_full_key.size() - src_prefix2.size());
        }

        // Build right side.
        const LayoutChoice right_choice = chooseLayoutCmp(src, mid, n,
            ins_to_right ? inserting_post_prefix : std::string_view{});
        materializeSliceInto(tls_scratch, src, mid, n, separator, src_upper, src_next, right_choice);
        std::memcpy(right_node.page_.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

        // Build left side.
        const LayoutChoice left_choice = chooseLayoutCmp(src, 0, mid,
            ins_to_right ? std::string_view{} : inserting_post_prefix);
        materializeSliceInto(tls_scratch, src, 0, mid, src_lower, separator, right_id, left_choice);
        std::memcpy(src.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);

        return LeafSplitResult{std::move(separator), /*right_is_new_partition=*/false};
    }

    // -----------------------------------------------------------------------
    // Merge (Phase 6)
    // -----------------------------------------------------------------------

    bool LeafNode::tryMergeFrom(LeafNode& right)
    {
        thread_local std::string lower_buf;
        thread_local std::string upper_buf;
        lower_buf.assign(page_.lowerFenceView());
        upper_buf.assign(right.page_.upperFenceView());
        const bool had_lower = page_.hasLowerFence();
        const bool had_upper = right.page_.hasUpperFence();
        const std::string_view ml = had_lower ? std::string_view(lower_buf) : std::string_view{};
        const std::string_view mu = had_upper ? std::string_view(upper_buf) : std::string_view{};
        const NodeId merged_link = right.page_.link();

        std::vector<KeyValue> left_kvs;
        std::vector<KeyValue> right_kvs;
        left_kvs.reserve(256);
        right_kvs.reserve(256);
        collectScan(std::string_view{}, std::numeric_limits<std::size_t>::max(), left_kvs);
        right.collectScan(std::string_view{}, std::numeric_limits<std::size_t>::max(), right_kvs);
        std::vector<KeyValue> merged;
        merged.reserve(left_kvs.size() + right_kvs.size());
        merged.insert(merged.end(),
                      std::make_move_iterator(left_kvs.begin()),
                      std::make_move_iterator(left_kvs.end()));
        merged.insert(merged.end(),
                      std::make_move_iterator(right_kvs.begin()),
                      std::make_move_iterator(right_kvs.end()));

        // Build the merged comparison leaf into tls_scratch. If it doesn't fit,
        // bail (no mutation to either side).
        tls_scratch.init(NodeType::kLeaf, ml, mu);
        tls_scratch.setLeafKind(LeafKind::kComparison);
        tls_scratch.setLink(merged_link);
        const std::size_t pfx_len = tls_scratch.prefixView().size();
        for (const auto& kv : merged)
        {
            if (kv.key.size() < pfx_len) return false;
            const std::string_view suf(kv.key.data() + pfx_len, kv.key.size() - pfx_len);
            if (!tls_scratch.hasSpaceForLeaf(suf.size())) return false;
            tls_scratch.appendLeaf(suf, kv.value, make_head(suf));
        }
        tls_scratch.rebuildHints();

        // Adaptive re-promotion of the merged contents.
        const LayoutChoice c = chooseLayoutCmp(tls_scratch, 0, tls_scratch.slotCount());
        if (c.kind != LeafKind::kComparison)
        {
            std::memcpy(tls_scratch_aux.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);
            materializeSliceInto(tls_scratch, tls_scratch_aux, 0, tls_scratch_aux.slotCount(),
                                 ml, mu, merged_link, c);
        }

        std::memcpy(page_.rawBytes(), tls_scratch.rawBytes(), kPageSizeBytes);
        return true;
    }

} // namespace abt
