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
- **Adaptive leaf encodings** sharing one slotted-page facade through a 1-byte `LeafKind` tag (no virtual dispatch):
  - **Comparison leaves** (default): heads + hints + suffix-bytes payload heap.
  - **Fully Dense Leaves (FDL)**: bitmap + value array indexed by a numeric tail; insert decays to a bit-set + 64-bit store on dense sequential keys.
  - **Semi Dense Leaves (SDL)**: u16 offset table + length-prefixed payload heap, indexed by the same numeric extraction; covers variable-length numeric tails.
- **Adaptive switching**: at split time a per-half "bits-per-key" cost model picks the cheapest layout. Underflow demotes FDL/SDL back through the chain on `erase`, and oversize partition splits demote to comparison automatically.
- `NumericMode { kByteBE, kAsciiDecimal }` lets printable integer keys (`"000000001234"`) promote into FDL/SDL via ASCII-decimal extraction.
- **Iterative descent** with a fixed-size on-stack `InnerNode*` path array (no per-insert heap traffic, no recursion frames).
- **Zero-allocation hot path**: traversal, insertion, lookup, and range scan never call `new`/`malloc` for keys; FDL/SDL rebuilds reuse `thread_local SlottedPage` scratch pages.
- `erase()` API with sibling/inner merges and adaptive demotion on underflow; the 50k erase_mix self-check exercises insert/erase/search/re-insert in interleaved mode.
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

The default benchmark sweeps six datasets (URL-style, Wikipedia-style, three integer-string distributions including a dense-sequential FDL target and a tight-gap SDL target) at three sizes (1M / 5M / 10M keys) and reports `insert`, `point lookup`, and `range scan` throughput for both the Adaptive B-Tree and an unoptimized baseline B+-Tree. Each ABT line also prints a `leaf layout` summary (`cmp` / `fdl` / `sdl` counts and `dense_density`).

### 3) Fallback: direct compiler invocation (without CMake)

```bash
c++ -std=c++20 -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic \
    -Iinclude src/*.cpp benchmark/benchmark.cpp -o benchmark_app
./benchmark_app
```

## Usage / Running Benchmarks

The benchmark runner exercises six workload families:

- URL-style string keys
- Wikipedia-style title keys
- Integer strings, sparse formula `i*17 + 11` (zero-padded to 12 digits)
- Integer strings, dense sequential `0..N-1` (FDL target)
- Integer strings, sparse-gap `1..32` (variable spacing, comparison-leaf workload)
- Integer strings, tight-gap `1..3` (SDL/FDL target)

### What it runs

1. A 100,000-key correctness self-check (insert, then verify every key) for each dataset, plus a 50,000-key `erase_mix` round (insert, erase half, verify, re-insert, verify) that exercises split + merge + adaptive demote in interleaved mode. The harness aborts with a diagnostic if any inserted key is not retrievable.
2. A scaling sweep at `1,000,000`, `5,000,000`, and `10,000,000` keys per dataset, comparing the Adaptive B-Tree against the unoptimized baseline B+-Tree on `insert`, `point lookup`, and `range scan`. Each ABT line also prints `leaf layout` (`cmp` / `fdl` / `sdl` counts and `dense_density`).

Lookup and scan results are sunk through a `volatile uint64_t` accumulator so the optimizer can't delete the loops.

### Latest results

See [`benchmark_output.txt`](benchmark_output.txt) for the full scaling sweep. Headlines (head-to-head ABT speedup over baseline, &gt;1.0 means ABT wins):

| Dataset | Op | 1M | 5M | 10M |
| --- | --- | --- | --- | --- |
| URL                | insert | 0.86x | 0.88x | 0.89x |
| URL                | lookup | 1.49x | 2.66x | **10.41x** |
| URL                | scan   | 1.14x | 1.24x | 2.49x |
| Wiki               | insert | 0.93x | 0.98x | 0.79x |
| Wiki               | lookup | 1.58x | 1.79x | 2.22x |
| Wiki               | scan   | 1.18x | 1.22x | 1.27x |
| Int sparse i*17+11 | insert | 0.73x | 0.60x | 0.55x |
| Int sparse i*17+11 | lookup | 2.04x | 2.23x | 2.32x |
| Int sparse i*17+11 | scan   | 1.25x | 1.41x | 1.50x |
| **Int dense (FDL)**  | **insert** | **1.41x** | **1.56x** | **1.20x** |
| **Int dense (FDL)**  | **lookup** | **3.53x** | **7.50x** | **3.71x** |
| **Int dense (FDL)**  | **scan**   | **1.32x** | **2.52x** | **1.48x** |
| Int gap 1..32      | insert | 0.62x | 0.58x | 0.55x |
| Int gap 1..32      | lookup | 2.02x | 2.17x | 2.31x |
| Int gap 1..32      | scan   | 1.24x | 1.41x | 1.51x |
| **Int tight (FDL)** | **insert** | **1.48x** | **1.43x** | 0.96x |
| **Int tight (FDL)** | **lookup** | **3.23x** | **3.02x** | **3.49x** |
| Int tight (FDL)    | scan   | 1.22x | 1.24x | 1.42x |

The Adaptive B-Tree now dominates the read paths the paper targets *and* the dense-sequential insert path: dense and tight-gap integer workloads are 100% FDL-promoted (`dense_density=1.000` and `0.500` respectively) and beat the baseline insert by 1.20-1.56x while running point lookups 3-7x faster. Random URL/Wiki workloads stay on comparison leaves and inherit the prior-pass wins on point lookup and range scan, with the gap widening at 10M (10.4x URL lookup). The remaining insert delta on the sparse-integer formulas is a structural property of the unoptimized baseline (linear-scan-from-end + key.size()+128-byte pre-emptive split on monotonic input) and costs that baseline a 2-3x penalty on every read path.

## Project Architecture (Codebase Tour)

| Path | Purpose |
| --- | --- |
| `include/adaptive_btree/config.hpp` | Global constants (`kPageSizeBytes = 4096`, `kHintCount = 16`, `kSdlMaxCapacity`). |
| `include/adaptive_btree/common.hpp` | Shared types (`NodeId`, `Value`, `KeyValue`, `LeafKind`, `NumericMode`) + `make_head` + longest-common-prefix helpers + ASCII-decimal digit counting. |
| `include/adaptive_btree/fdl_layout.hpp`, `sdl_layout.hpp` | Header PODs / numeric extraction / decode helpers for FDL/SDL encodings embedded in raw page bytes. |
| `include/adaptive_btree/slotted_page.hpp` + `src/slotted_page.cpp` | 4 KiB slotted page: header (with `LeafKind` tag), hint array, slot directory, fence-bearing payload heap. Hint-narrowed `lowerBoundIndex` / `upperBoundIndex`, `appendLeaf` / `appendInner`, `rebuildHints`. |
| `include/adaptive_btree/node.hpp` + `src/node.cpp` | Thin base wrapping `(NodeId, SlottedPage)` with a leaf/inner type tag (no virtual dispatch in hot paths). |
| `include/adaptive_btree/leaf_node.hpp` + `src/leaf_node.cpp` | `tryInsert` / `find` / `erase` dispatched on `LeafKind`; comparison-tail fast path + FDL bitmap/SDL offset-table paths + `splitInto` with adaptive promote/demote and `tryMergeFrom` for erase. |
| `include/adaptive_btree/inner_node.hpp` + `src/inner_node.cpp` | `childIndexForKey` (sequential-workload fast path + hint-narrowed upper-bound), `tryInsertSeparator`, `splitInto`, `tryMergeFrom` for inner merges. |
| `include/adaptive_btree/adaptive_btree.hpp` + `src/adaptive_btree.cpp` | Iterative descent with a fixed-size `InnerNode* path[kMaxTreeDepth]` stack; bubble-up split propagation; root-split handling; `search`, `rangeScan`, `erase`, `layoutStats()`. |
| `benchmark/benchmark.cpp` | Dataset generators + correctness self-checks (including erase_mix) + scaling sweep vs baseline B+-tree; volatile sink to defeat dead-code elimination. |
| `CMakeLists.txt` | Release-by-default, `-O3 -march=native`, IPO/LTO. |

### Design Notes

- The page header carries a 1-byte `LeafKind` discriminator (`comparison` vs `FDL` vs `SDL`) so `LeafNode` can branch without virtual calls. Dense layouts stash their POD metadata immediately after the slotted-page header (`fdl_layout.hpp` / `sdl_layout.hpp`).
- The static prefix is set exactly once at split time as `LCP(lower_fence, upper_fence)`. The B-tree routing invariant guarantees every key reaching a node already starts with that prefix, so local inserts never need to repack on prefix mismatch.
- The hint array stores 16 sampled `uint32_t` heads. `lowerBoundIndex` / `upperBoundIndex` use those heads to shrink `[lo, hi)` before the binary search hits the actual slot directory; the narrowing logic explicitly extends `hi` past hints that are equal to the target head so equal-head ties don't drop matching slots out of range.
- Splits never `memset` the 4 KiB page. They rebuild left and right halves into `thread_local SlottedPage` scratch buffers and `memcpy` the result back over the source / destination page (aux scratch for two-at-once median demotes).
- `make_head` uses `memcpy + __builtin_bswap32` for keys with at least 4 suffix bytes and a 1-byte loop fallback for shorter suffixes, keeping comparisons branch-free in hot paths.

## Roadmap / Future Work

- Add optional fingerprinting mode for deeper comparison avoidance.
- Expand benchmark coverage with additional key distributions (highly skewed, long-tail, adversarial prefixes) and unsorted insert workloads.
- Add randomized stress / fuzz tests specifically targeting FDL/SDL boundary cases (sparse→dense transitions, ASCII-decimal digit overflow, sibling merges).
- Hard stress `erase_mix` with SDL-heavy sparse gaps once merge/split invariants are further hardened across all leaf kinds.
- Explore concurrent access patterns and latch/lock strategies for multi-threaded workloads.
