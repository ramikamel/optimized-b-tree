#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "adaptive_btree/adaptive_btree.hpp"
#include "baseline_bplus_tree/baseline_bplus_tree.hpp" // NEW BASELINE

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

    // --- OPTIMIZED B-TREE ---
    std::cout << "[Adaptive B-Tree: Slotted Pages + Truncation + Hints]\n";
    abt::AdaptiveBTree tree;
    measureThroughput("insert", keys.size(), [&] {
        for (std::size_t i = 0; i < keys.size(); ++i) tree.insert(keys[i], static_cast<abt::Value>(i));
    });

    measureThroughput("point lookup", query_count, [&] {
        std::size_t found = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            found += tree.search(keys[probe_indices[i]]).has_value() ? 1 : 0;
        }
    });

    measureThroughput("range scan", range_ops, [&] {
        std::size_t total_rows = 0;
        for (std::size_t i = 0; i < range_ops; ++i) {
            total_rows += tree.rangeScan(keys[probe_indices[i]], scan_len).size();
        }
    });

    // --- BASELINE (Standard B+-Tree) ---
    std::cout << "[Baseline: Unoptimized Standard B+-Tree (Heap Strings)]\n";
    bpt::StandardBPlusTree baseline_tree;
    measureThroughput("insert", keys.size(), [&] {
        for (std::size_t i = 0; i < keys.size(); ++i) baseline_tree.insert(keys[i], static_cast<bpt::Value>(i));
    });

    measureThroughput("point lookup", query_count, [&] {
        std::size_t found = 0;
        for (std::size_t i = 0; i < query_count; ++i) {
            found += baseline_tree.search(keys[probe_indices[i]]).has_value() ? 1 : 0;
        }
    });

    measureThroughput("range scan", range_ops, [&] {
        std::size_t total_rows = 0;
        for (std::size_t i = 0; i < range_ops; ++i) {
            total_rows += baseline_tree.rangeScan(keys[probe_indices[i]], scan_len).size();
        }
    });
    std::cout << "--------------------------------------------------------\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::size_t> sizes = {100000, 500000, 1000000};
        std::mt19937_64 rng(42);

        for (std::size_t count : sizes) {
            std::cout << "\n========================================================";
            std::cout << "\n      SCALING BENCHMARK: " << count << " KEYS";
            std::cout << "\n========================================================\n";

            const std::vector<std::string> url_keys = generateUrls(count, rng);
            const std::vector<std::string> wiki_keys = generateWikiTitles(count, rng);
            const std::vector<std::string> integer_keys = generateIntegerStrings(count);

            runDatasetBenchmark("URL-style strings", url_keys, rng);
            runDatasetBenchmark("Wikipedia-style titles", wiki_keys, rng);
            runDatasetBenchmark("Integer strings", integer_keys, rng);
        }
    } catch (const std::exception& ex) {
        std::cerr << "benchmark failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}