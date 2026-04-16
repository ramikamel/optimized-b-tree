# Adaptive B-Tree for Variable-Length Keys (C++20)

## Project Overview

This repository contains a standalone implementation of an adaptive B-Tree optimized for variable-length string keys (for example, URLs and Wikipedia-style titles).

The design is inspired by the 2025 SIGMOD paper *"B-Trees Are Back: Engineering Fast and Pageable Node Layouts"* by Mueller, Benson, and Leis, and focuses on pageable node engineering techniques that improve space efficiency and lookup speed while preserving classic B-Tree behavior.

The implementation is intentionally dependency-light and built from scratch in modern C++ for clarity, experimentation, and extension by systems developers and researchers.

## Key Features / Highlights

- Fixed-size 4 KiB node pages for cache- and page-friendly behavior.
- Slotted-page node layout for variable-length record storage.
- Explicit separation between inner (routing) nodes and leaf (key/value) nodes.
- Node-level prefix truncation (head/prefix compression) to reduce key storage and comparison work.
- Dense 16-bit hints (derived from key suffix prefixes) to accelerate binary search and reduce full-string comparisons.
- Variable-length-aware split strategy based on encoded byte estimates (not only key count).
- Leaf chaining for efficient forward range scans.
- Feature flags that keep the architecture open for future dense-leaf and fingerprinting modules.

## Prerequisites

- C++ compiler with modern standard support (C++20 recommended; C++17-capable toolchains are generally suitable with minor adaptation if needed).
- CMake 3.16 or newer.
- Standard native build tools for your platform.
	- Linux: `gcc`/`clang`, `make` or `ninja`
	- macOS: Xcode Command Line Tools (`clang`)
	- Windows: MSVC (Visual Studio) or LLVM/MinGW toolchain

## Build Instructions

### 1) Configure and build with CMake (recommended)

```bash
cmake -S . -B build
cmake --build build -j
```

### 2) Run the benchmark target

```bash
./build/benchmark 100000
```

### 3) Fallback: direct compiler invocation (without CMake)

```bash
c++ -std=c++20 -Wall -Wextra -Wpedantic -Iinclude src/*.cpp benchmark/benchmark.cpp -o benchmark_app
./benchmark_app 100000
```

## Usage / Running Benchmarks

The benchmark runner exercises three workload families:

- URL-style string keys
- Wikipedia-style title keys
- Integer keys encoded as strings

### Command format

```bash
./build/benchmark <row_count> [wiki_titles_file]
```

- `<row_count>`: number of records per dataset (for example `100000`).
- `[wiki_titles_file]` (optional): plain text file with one title per line. If omitted, synthetic titles are generated.

### Example: synthetic datasets only

```bash
./build/benchmark 200000
```

### Example: custom Wikipedia titles file

```bash
./build/benchmark 100000 ./data/wiki_titles.txt
```

The benchmark reports throughput for:

- inserts
- point lookups
- range scans

along with basic tree shape statistics such as logical size and height.

## Project Architecture (Codebase Tour)

| Path | Purpose |
| --- | --- |
| `include/adaptive_btree/config.hpp` | Global constants and feature toggles (future dense-leaf and fingerprinting hooks). |
| `include/adaptive_btree/common.hpp` | Shared types/utilities (`NodeId`, `Value`, key compare helpers, hint derivation). |
| `include/adaptive_btree/slotted_page.hpp` + `src/slotted_page.cpp` | 4 KiB slotted-page core, header/slot/payload management, prefix storage, and record insertion primitives. |
| `include/adaptive_btree/node.hpp` + `src/node.cpp` | Base node abstraction used by inner and leaf nodes. |
| `include/adaptive_btree/leaf_node.hpp` + `src/leaf_node.cpp` | Leaf materialization/rebuild, hint-aware lower-bound search, value access, leaf-next chaining. |
| `include/adaptive_btree/inner_node.hpp` + `src/inner_node.cpp` | Inner node routing logic, child selection, and materialized reconstruction APIs. |
| `include/adaptive_btree/adaptive_btree.hpp` + `src/adaptive_btree.cpp` | Tree operations (`insert`, `search`, `rangeScan`), recursive split propagation, node allocation/ownership. |
| `benchmark/benchmark.cpp` | Benchmark harness, synthetic data generators, optional file-driven input, and throughput timing. |
| `CMakeLists.txt` | Build configuration for library and benchmark executable. |

### Design Notes

- Prefix truncation is applied per node by factoring out the common key prefix and storing only variable suffixes in payload space.
- Hints are stored densely in slot metadata and used as a fast pre-filter before full key comparisons.
- Split decisions account for encoded byte footprint to handle variable-length keys robustly.

## Roadmap / Future Work

- Implement semi-dense and fully dense leaf encodings behind feature flags.
- Add optional fingerprinting mode for deeper comparison avoidance.
- Expand benchmark coverage with larger-scale datasets and additional key distributions (highly skewed, long-tail, adversarial prefixes).
- Add correctness and stress test suites (fuzzing, randomized insert/search/range validation).
- Explore concurrent access patterns and latch/lock strategies for multi-threaded workloads.
