#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "adaptive_btree/adaptive_btree.hpp"
#include "baseline_bplus_tree/baseline_bplus_tree.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::string randomWord(std::mt19937_64& rng, std::size_t min_len, std::size_t max_len) {
    static constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<std::size_t> len_dist(min_len, max_len);
    std::uniform_int_distribution<int> ch_dist(0, 25);

    const std::size_t len = len_dist(rng);
    std::string out;
    out.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        out.push_back(alphabet[ch_dist(rng)]);
    }
    return out;
}

std::vector<std::string> generateUrls(std::size_t count, std::mt19937_64& rng) {
    static const std::vector<std::string> domains = {
        "example.com", "wikipedia.org", "github.com", "news.example.net", "docs.internal.io"};
    std::uniform_int_distribution<std::size_t> domain_dist(0, domains.size() - 1);
    std::uniform_int_distribution<int> segment_count_dist(2, 5);

    std::vector<std::string> keys;
    keys.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        std::string url = "https://" + domains[domain_dist(rng)] + "/";
        const int segment_count = segment_count_dist(rng);
        for (int s = 0; s < segment_count; ++s) {
            if (s > 0) url.push_back('/');
            url += randomWord(rng, 3, 12);
        }
        keys.push_back(std::move(url));
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    while (keys.size() < count) {
        keys.push_back("https://fallback.example/" + randomWord(rng, 5, 14));
    }
    keys.resize(count);
    return keys;
}

std::vector<std::string> generateWikiTitles(std::size_t count, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> token_count_dist(2, 6);
    std::uniform_int_distribution<int> year_dist(1850, 2026);

    std::vector<std::string> keys;
    keys.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const int token_count = token_count_dist(rng);
        std::string title;
        for (int t = 0; t < token_count; ++t) {
            if (t > 0) title.push_back('_');
            std::string token = randomWord(rng, 4, 11);
            token[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
            title += token;
        }
        if ((i % 4) == 0) title += "_(" + std::to_string(year_dist(rng)) + ")";
        keys.push_back(std::move(title));
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    while (keys.size() < count) {
        keys.push_back("Article_" + randomWord(rng, 4, 10));
    }
    keys.resize(count);
    return keys;
}

std::vector<std::string> generateIntegerStrings(std::size_t count) {
    std::vector<std::string> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint64_t v = static_cast<std::uint64_t>(i * 17ULL + 11ULL);
        std::string key = std::to_string(v);
        if (key.size() < 12) key.insert(key.begin(), 12 - key.size(), '0');
        keys.push_back(std::move(key));
    }
    return keys;
}

// Densely sequential integer keys (values 0..count-1, no gaps). This is the
// workload Fully Dense Leaves were designed for: uniform suffix length, dense
// numeric range, and lex-sorted == numeric-sorted.
std::vector<std::string> generateDenseIntegerStrings(std::size_t count) {
    std::vector<std::string> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::string key = std::to_string(i);
        if (key.size() < 12) key.insert(key.begin(), 12 - key.size(), '0');
        keys.push_back(std::move(key));
    }
    return keys;
}

// Integer-extractable keys with random gaps. Targets the SDL branch when gaps
// are small enough that per-leaf numeric range stays under kSdlMaxCapacity;
// larger gaps fall back to comparison leaves (still benefits from heads/hints).
std::vector<std::string> generateSparseIntegerStrings(std::size_t count,
                                                      std::mt19937_64& rng,
                                                      int max_gap = 32) {
    std::uniform_int_distribution<int> gap_dist(1, max_gap);
    std::vector<std::string> keys;
    keys.reserve(count);
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < count; ++i) {
        v += static_cast<std::uint64_t>(gap_dist(rng));
        std::string key = std::to_string(v);
        if (key.size() < 12) key.insert(key.begin(), 12 - key.size(), '0');
        keys.push_back(std::move(key));
    }
    return keys;
}

// Tight-gap sequential integer keys that fit comfortably inside SDL's numeric
// span budget (kSdlMaxCapacity=512). Average gap ~2 keeps per-leaf range under
// the cap so the layout chooser promotes most leaves to SDL.
std::vector<std::string> generateTightGapIntegerStrings(std::size_t count,
                                                        std::mt19937_64& rng) {
    return generateSparseIntegerStrings(count, rng, 3);
}

// Volatile sink prevents the optimizer (especially under LTO) from deleting
// the work in benchmark loops whose results are otherwise unobserved.
volatile std::uint64_t g_sink = 0;

inline void doNotOptimize(std::uint64_t v) {
    g_sink ^= v;
}

template <typename Func>
double measureThroughput(const std::string& label, std::size_t operations, Func&& fn) {
    const auto start = Clock::now();
    fn();
    const auto end = Clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    const double secs = elapsed.count();
    const double throughput = secs > 0.0 ? static_cast<double>(operations) / secs : 0.0;
    std::cout << std::left << std::setw(22) << label << " : " << std::fixed << std::setprecision(2)
              << throughput << " ops/sec (" << secs << " s)\n";
    return throughput;
}

void runDatasetBenchmark(const std::string& dataset_name, const std::vector<std::string>& keys, std::mt19937_64& rng) {
    std::cout << "\n>>> " << dataset_name << " <<<\n";

    std::vector<std::size_t> probe_indices(keys.size());
    std::iota(probe_indices.begin(), probe_indices.end(), 0);
    std::shuffle(probe_indices.begin(), probe_indices.end(), rng);

    const std::size_t query_count = std::min<std::size_t>(keys.size(), 50000);
    const std::size_t range_ops = std::min<std::size_t>(query_count, 10000);
    constexpr std::size_t scan_len = 32;

    std::cout << "[Adaptive B-Tree: Slotted Pages + Static Prefix Truncation + Heads + Hints]\n";
    abt::AdaptiveBTree tree;
    measureThroughput("insert", keys.size(), [&] {
        for (std::size_t i = 0; i < keys.size(); ++i) tree.insert(keys[i], static_cast<abt::Value>(i));
    });

    {
        const auto stats = tree.layoutStats();
        const std::size_t total_leaves = stats.n_comparison + stats.n_fdl + stats.n_sdl;
        std::cout << std::left << std::setw(22) << "leaf layout"
                  << " : cmp=" << stats.n_comparison
                  << " fdl=" << stats.n_fdl
                  << " sdl=" << stats.n_sdl
                  << " (" << total_leaves << " leaves)";
        if (stats.total_dense_capacity > 0) {
            const double density =
                static_cast<double>(stats.total_entries) / stats.total_dense_capacity;
            std::cout << " dense_density=" << std::fixed << std::setprecision(3) << density;
        }
        std::cout << "\n";
    }

    measureThroughput("point lookup", query_count, [&] {
        std::size_t found = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            const auto v = tree.search(keys[probe_indices[i]]);
            if (v) doNotOptimize(*v);
            found += v ? 1 : 0;
        }
        if (found != query_count) {
            std::cerr << "ERROR: only " << found << "/" << query_count << " keys found in AdaptiveBTree\n";
            std::abort();
        }
    });

    measureThroughput("range scan", range_ops, [&] {
        std::size_t total_rows = 0;
        for (std::size_t i = 0; i < range_ops; ++i) {
            auto rows = tree.rangeScan(keys[probe_indices[i]], scan_len);
            for (const auto& r : rows) doNotOptimize(r.value);
            total_rows += rows.size();
        }
        doNotOptimize(static_cast<std::uint64_t>(total_rows));
    });

    std::cout << "[Baseline: Unoptimized Standard B+-Tree (Heap Strings)]\n";
    bpt::StandardBPlusTree baseline_tree;
    measureThroughput("insert", keys.size(), [&] {
        for (std::size_t i = 0; i < keys.size(); ++i) baseline_tree.insert(keys[i], static_cast<bpt::Value>(i));
    });

    measureThroughput("point lookup", query_count, [&] {
        std::size_t found = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            const auto v = baseline_tree.search(keys[probe_indices[i]]);
            if (v) doNotOptimize(*v);
            found += v ? 1 : 0;
        }
        if (found != query_count) {
            std::cerr << "ERROR: only " << found << "/" << query_count << " keys found in baseline\n";
            std::abort();
        }
    });

    measureThroughput("range scan", range_ops, [&] {
        std::size_t total_rows = 0;
        for (std::size_t i = 0; i < range_ops; ++i) {
            auto rows = baseline_tree.rangeScan(keys[probe_indices[i]], scan_len);
            for (const auto& r : rows) doNotOptimize(r.value);
            total_rows += rows.size();
        }
        doNotOptimize(static_cast<std::uint64_t>(total_rows));
    });
    std::cout << "--------------------------------------------------------\n";
}

// Lightweight correctness gate. Inserts N keys and verifies every one is retrievable.
// Aborts on the first miss so a regression cannot be hidden behind benchmark numbers.
void runCorrectnessSelfCheck() {
    constexpr std::size_t kCount = 100000;
    std::mt19937_64 rng(0xC07C0DEULL);

    auto verify = [](const std::string& label, const std::vector<std::string>& keys) {
        abt::AdaptiveBTree tree;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            tree.insert(keys[i], static_cast<abt::Value>(i));
            // Spot-check that the just-inserted key is searchable. This catches
            // a routing/insert disagreement on the first offending key.
            const auto v = tree.search(keys[i]);
            if (!v || *v != static_cast<abt::Value>(i)) {
                std::cerr << "[POST-INSERT MISS] " << label << " i=" << i
                          << " key=\"" << keys[i] << "\""
                          << " got=" << (v ? std::to_string(*v) : "<none>") << "\n";
                std::abort();
            }
        }
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const auto v = tree.search(keys[i]);
            if (!v || *v != static_cast<abt::Value>(i)) {
                std::cerr << "[FINAL MISS] " << label << " key=\"" << keys[i] << "\" idx=" << i << "\n";
                std::abort();
            }
        }
        std::cout << "[selfcheck] " << label << ": " << keys.size() << " inserts/search OK\n";
    };

    verify("urls", generateUrls(kCount, rng));
    verify("wiki", generateWikiTitles(kCount, rng));
    verify("ints", generateIntegerStrings(kCount));
    verify("dense_ints", generateDenseIntegerStrings(kCount));
    verify("sparse_ints", generateSparseIntegerStrings(kCount, rng));

    // erase-mix: insert N keys, erase the first N/2, verify the second N/2
    // are still retrievable and erased keys are gone, then re-insert the
    // erased half and verify all N retrievable. Exercises split + merge +
    // adaptive demote in interleaved mode.
    {
        constexpr std::size_t kEraseCount = 50000;
        std::mt19937_64 ergng(0xDEAD'BEEFULL);
        const auto keys = generateSparseIntegerStrings(kEraseCount, ergng);
        abt::AdaptiveBTree tree;
        for (std::size_t i = 0; i < kEraseCount; ++i)
            tree.insert(keys[i], static_cast<abt::Value>(i));
        for (std::size_t i = 0; i < kEraseCount / 2; ++i)
        {
            if (!tree.erase(keys[i]))
            {
                std::cerr << "[ERASE-MIX] erase missed at i=" << i
                          << " key=\"" << keys[i] << "\"\n";
                std::abort();
            }
        }
        for (std::size_t i = 0; i < kEraseCount / 2; ++i)
        {
            if (tree.search(keys[i]).has_value())
            {
                std::cerr << "[ERASE-MIX] erased key still findable at i=" << i << "\n";
                std::abort();
            }
        }
        for (std::size_t i = kEraseCount / 2; i < kEraseCount; ++i)
        {
            const auto v = tree.search(keys[i]);
            if (!v || *v != static_cast<abt::Value>(i))
            {
                std::cerr << "[ERASE-MIX] retained key missing at i=" << i << "\n";
                std::abort();
            }
        }
        for (std::size_t i = 0; i < kEraseCount / 2; ++i)
            tree.insert(keys[i], static_cast<abt::Value>(i));
        for (std::size_t i = 0; i < kEraseCount; ++i)
        {
            const auto v = tree.search(keys[i]);
            if (!v || *v != static_cast<abt::Value>(i))
            {
                std::cerr << "[ERASE-MIX] re-insert key missing at i=" << i << "\n";
                std::abort();
            }
        }
        std::cout << "[selfcheck] erase_mix: " << kEraseCount
                  << " insert/erase/search OK\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        // Always run a small correctness gate before any benchmarking. If this
        // fails the process aborts so that no misleading numbers are reported.
        runCorrectnessSelfCheck();

        std::vector<std::size_t> sizes = {1000000, 5000000, 10000000};
        // Allow callers to override the row counts (kept for parity with previous CLI).
        if (argc >= 2) {
            try {
                const std::size_t custom = static_cast<std::size_t>(std::stoull(argv[1]));
                if (custom > 0) sizes = {custom};
            } catch (...) {
                // ignore parse error; fall back to defaults
            }
        }

        std::mt19937_64 rng(42);

        for (std::size_t count : sizes) {
            std::cout << "\n========================================================";
            std::cout << "\n      SCALING BENCHMARK: " << count << " KEYS";
            std::cout << "\n========================================================\n";

            const std::vector<std::string> url_keys = generateUrls(count, rng);
            const std::vector<std::string> wiki_keys = generateWikiTitles(count, rng);
            const std::vector<std::string> integer_keys = generateIntegerStrings(count);
            const std::vector<std::string> dense_int_keys = generateDenseIntegerStrings(count);
            const std::vector<std::string> sparse_int_keys = generateSparseIntegerStrings(count, rng);
            const std::vector<std::string> tight_int_keys = generateTightGapIntegerStrings(count, rng);

            runDatasetBenchmark("URL-style strings", url_keys, rng);
            runDatasetBenchmark("Wikipedia-style titles", wiki_keys, rng);
            runDatasetBenchmark("Integer strings (sparse, i*17+11)", integer_keys, rng);
            runDatasetBenchmark("Integer strings (dense sequential)", dense_int_keys, rng);
            runDatasetBenchmark("Integer strings (sparse-gap 1..32)", sparse_int_keys, rng);
            runDatasetBenchmark("Integer strings (tight-gap 1..3, SDL target)", tight_int_keys, rng);
        }
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
