#pragma once

#include <cstddef>

namespace abt
{

    constexpr std::size_t kPageSizeBytes = 4096;

    struct FeatureFlags
    {
        // Hook for future semi-dense/fully dense leaf encodings.
        bool enable_dense_leaves = false;
        // Hook for future hash-based fingerprinting paths.
        bool enable_fingerprinting = false;
    };

} // namespace abt
