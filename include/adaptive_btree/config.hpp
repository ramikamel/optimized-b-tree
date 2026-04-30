#pragma once

#include <cstddef>
#include <cstdint>

namespace abt
{

    constexpr std::size_t kPageSizeBytes = 4096;
    constexpr std::uint16_t kHintCount = 16;

    struct FeatureFlags
    {
        bool enable_dense_leaves = false;
        bool enable_fingerprinting = false;
    };

} // namespace abt
