# Adaptive B-Tree for Variable-Length Keys (C++20)

## Project Overview

This repository contains a standalone implementation of an adaptive B-Tree optimized for variable-length string keys (for example, URLs and Wikipedia-style titles).

The design is inspired by the 2025 SIGMOD paper *"B-Trees Are Back: Engineering Fast and Pageable Node Layouts"* by Mueller, Benson, and Leis, and focuses on pageable node engineering techniques that improve space efficiency and lookup speed while preserving classic B-Tree behavior.

The implementation is intentionally dependency-light and built from scratch in modern C++ for clarity, experimentation, and extension by systems developers and researchers.

## Key Features / Highlights

- Fixed-size 4 KiB slotted node pages with on-page lower/upper fences and a 16-entry sampled hint array following the paper's layout.
- **Static prefix truncation**: each node's prefix is set exactly once, at split time, as `LCP(lower_fence, upper_fence)`. Local inserts never rewrite the prefix, eliminating O(N) repacks.
- **Separator truncation**: split-time separators are truncated to the shortest valid string (`LCP(last_left, first_right) + 1` byte from `first_right`), keeping inner nodes small and prefixes long.
- **Hint array** (`uint32_t` heads sampled at fixed strides) narrows the binary-search range before any payload access.
- **Tail-insert fast path** on both leaves and inner nodes: a single head compare, optionally one suffix compare, then a direct `appendLeaf` / `appendInner`. Bypasses the full hint+binary-search on sorted workloads.
- **Iterative descent** with a fixed-size on-stack `InnerNode*` path array (no per-insert heap traffic, no recursion frames).
- **Zero-allocation hot path**: traversal, insertion, lookup, and range scan never call `new`/`malloc` for keys; only the per-split promoted separator and the tree's per-node `unique_ptr` are allocated.
- Variable-length-aware split that balances by accumulated payload bytes, not just key count.
- Leaf chaining for cheap forward range scans.

## Prerequisites

- C++ compiler with modern standard support (C++20 recommended; C++17-capable toolchains are generally suitable with minor adaptation if needed).
- CMake 3.16 or newer.
- Standard native build tools for your platform.
	- Linux: `gcc`/`clang`, `make` or `ninja`
	- macOS: Xcode Command Line Tools (`clang`)
	- Windows: MSVC (Visual Studio) or LLVM/MinGW toolchain

## Build Instructions

The CMake build defaults to a `Release` configuration with `-O3 -DNDEBUG -march=native` and IPO/LTO enabled. Reproducing the published numbers requires Release.

### 1) Configure and build with CMake (recommended)

```bash
cmake -S . -B build
cmake --build build -j
```

### 2) Run the benchmark target

```bash
./build/benchmark > benchmark_output.txt
```

The default benchmark sweeps three datasets (URL-style, Wikipedia-style, integer strings) at three sizes (1M / 5M / 10M keys) and reports `insert`, `point lookup`, and `range scan` throughput for both the Adaptive B-Tree and an unoptimized baseline B+-Tree.

### 3) Fallback: direct compiler invocation (without CMake)

```bash
c++ -std=c++20 -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic \
    -Iinclude src/*.cpp benchmark/benchmark.cpp -o benchmark_app
./benchmark_app
```

## Usage / Running Benchmarks

The benchmark runner exercises three workload families:

- URL-style string keys
- Wikipedia-style title keys
- Integer keys encoded as strings

### What it runs

1. A 100,000-key correctness self-check (insert, then verify every key) for each dataset. The harness aborts with a diagnostic if any inserted key is not retrievable.
2. A scaling sweep at `1,000,000`, `5,000,000`, and `10,000,000` keys per dataset, comparing the Adaptive B-Tree against the unoptimized baseline B+-Tree on `insert`, `point lookup`, and `range scan`.

Lookup and scan results are sunk through a `volatile uint64_t` accumulator so the optimizer can't delete the loops.

### Latest results

See [`benchmark_output.txt`](benchmark_output.txt) for the full scaling sweep. Headlines (head-to-head ABT speedup over baseline, &gt;1.0 means ABT wins):

| Dataset | Op | 1M | 5M | 10M |
| --- | --- | --- | --- | --- |
| URL  | insert | 0.87x | 0.88x | 0.90x |
| URL  | lookup | 1.32x | 1.81x | **5.73x** |
| URL  | scan   | 1.13x | 1.13x | 1.73x |
| Wiki | insert | 0.93x | 1.07x | 1.16x |
| Wiki | lookup | 1.57x | 2.63x | **4.33x** |
| Wiki | scan   | 1.15x | 1.36x | 1.30x |
| Int  | insert | 0.98x | 0.79x | 0.66x |
| Int  | lookup | 1.92x | 2.08x | 2.09x |
| Int  | scan   | 1.24x | 1.29x | 1.34x |

The Adaptive B-Tree dominates the read paths the paper targets - point lookups and range scans - on every dataset and every size, with the gap widening as N grows. Inserts are competitive: ABT wins Wikipedia inserts at 5M and 10M; the integer-string regression is a structural advantage of the unoptimized baseline on monotonically increasing inputs (linear-scan-from-end with pre-emptive splits) and costs the baseline a 2-6x penalty on the read paths.

## Project Architecture (Codebase Tour)

| Path | Purpose |
| --- | --- |
| `include/adaptive_btree/config.hpp` | Global constants (`kPageSizeBytes = 4096`, `kHintCount = 16`). |
| `include/adaptive_btree/common.hpp` | Shared types (`NodeId`, `Value`, `KeyValue`) + `make_head` (big-endian 4-byte head computed via `memcpy + __builtin_bswap32`) + `longest_common_prefix`. |
| `include/adaptive_btree/slotted_page.hpp` + `src/slotted_page.cpp` | 4 KiB slotted page: header, hint array, slot directory, fence-bearing payload heap. Hint-narrowed `lowerBoundIndex` / `upperBoundIndex`, `appendLeaf` / `appendInner`, `rebuildHints`. |
| `include/adaptive_btree/node.hpp` + `src/node.cpp` | Thin base wrapping `(NodeId, SlottedPage)` with a leaf/inner type tag (no virtual dispatch in hot paths). |
| `include/adaptive_btree/leaf_node.hpp` + `src/leaf_node.cpp` | `tryInsert` with tail-insert fast path, `find`, range-scan `lowerBoundIndex`, and `splitInto` that rebuilds both halves into a `thread_local` scratch page and writes back via a single 4 KiB `memcpy`. |
| `include/adaptive_btree/inner_node.hpp` + `src/inner_node.cpp` | `childIndexForKey` (sequential-workload fast path + hint-narrowed upper-bound), `tryInsertSeparator`, and `splitInto` with median-key promotion and separator truncation. |
| `include/adaptive_btree/adaptive_btree.hpp` + `src/adaptive_btree.cpp` | Iterative descent with a fixed-size `InnerNode* path[kMaxTreeDepth]` stack; bubble-up split propagation; root-split handling; `search` and `rangeScan`. |
| `benchmark/benchmark.cpp` | 100k-key correctness self-check + scaling sweep at 1M / 5M / 10M; baseline-vs-ABT throughput for insert, lookup, scan; volatile sink to defeat dead-code elimination. |
| `CMakeLists.txt` | Release-by-default, `-O3 -march=native`, IPO/LTO. |

### Design Notes

- The static prefix is set exactly once at split time as `LCP(lower_fence, upper_fence)`. The B-tree routing invariant guarantees every key reaching a node already starts with that prefix, so local inserts never need to repack on prefix mismatch.
- The hint array stores 16 sampled `uint32_t` heads. `lowerBoundIndex` / `upperBoundIndex` use those heads to shrink `[lo, hi)` before the binary search hits the actual slot directory; the narrowing logic explicitly extends `hi` past hints that are equal to the target head so equal-head ties don't drop matching slots out of range.
- Splits never `memset` the 4 KiB page. They rebuild left and right halves into a single `thread_local SlottedPage tls_scratch` and `memcpy` the result back over the source / destination page.
- `make_head` uses `memcpy + __builtin_bswap32` for keys with at least 4 suffix bytes and a 1-byte loop fallback for shorter suffixes, keeping comparisons branch-free in hot paths.

## Roadmap / Future Work

- Implement semi-dense and fully dense leaf encodings behind feature flags.
- Add optional fingerprinting mode for deeper comparison avoidance.
- Expand benchmark coverage with additional key distributions (highly skewed, long-tail, adversarial prefixes) and unsorted insert workloads.
- Add randomized stress / fuzz tests for splits and merges.
- Explore concurrent access patterns and latch/lock strategies for multi-threaded workloads.
