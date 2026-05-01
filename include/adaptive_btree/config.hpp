#pragma once

#include <cstddef>
#include <cstdint>

namespace abt
{

    constexpr std::size_t kPageSizeBytes = 4096;
    constexpr std::uint16_t kHintCount = 16;

    // Fully Dense Leaf parameters. The numeric range a single FDL covers; with
    // bitmap=ceil(N/8), values=N*8, header+meta=88 bytes, this leaves enough
    // tail room for typical fences/ref-keys (>=400 bytes).
    constexpr std::uint16_t kFdlMaxCapacity = 480;
    constexpr std::size_t   kFdlBitmapBytes = (kFdlMaxCapacity + 7) / 8; // 60
    constexpr std::size_t   kFdlValuesBytes = kFdlMaxCapacity * sizeof(std::uint64_t); // 3840

    // Semi Dense Leaf parameters. Capacity is the numeric span; the heap caps
    // the actual entry count by payload size. Sized so offsets[]+heap+header
    // comfortably fit in 4 KiB even with typical 12-16 byte payloads.
    constexpr std::uint16_t kSdlMaxCapacity = 512;

    // Adaptive promotion threshold. Convert to FDL when count/range >=
    // kFdlDensityNumerator / kFdlDensityDenominator. 50% is conservative; the
    // space-cost crossover sits around 41% for 4-byte numeric suffixes.
    constexpr std::uint16_t kFdlDensityNumerator = 1;
    constexpr std::uint16_t kFdlDensityDenominator = 2;

    // Underflow threshold for merge consideration. A leaf is "underflow" when
    // its used bytes drop below this fraction of the page. 25% is the standard
    // textbook B+-tree value.
    constexpr std::size_t   kMergeUnderflowBytes = kPageSizeBytes / 4;

    struct FeatureFlags
    {
        bool enable_dense_leaves = true;
        bool enable_fingerprinting = false;
    };

} // namespace abt
