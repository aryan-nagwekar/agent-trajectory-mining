// Exact, all-pairs Jaccard counts on pre-encoded sorted shingle sets.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    int n, numerator, denominator;
    if (!(std::cin >> n >> numerator >> denominator) || n < 0 || numerator <= 0) {
        return 1;
    }
    std::vector<std::vector<int>> sets(n);
    for (auto &set : sets) {
        int size;
        if (!(std::cin >> size) || size < 0) {
            return 1;
        }
        set.resize(size);
        for (auto &item : set) {
            if (!(std::cin >> item)) {
                return 1;
            }
        }
    }

    std::uint64_t matches = 0;
    for (int i = 0; i < n; ++i) {
        const auto &left = sets[i];
        if (left.empty()) {
            continue;
        }
        const int a = static_cast<int>(left.size());
        for (int j = i + 1; j < n; ++j) {
            const auto &right = sets[j];
            const int b = static_cast<int>(right.size());
            // A size-ratio bound excludes pairs that cannot reach the threshold.
            if (std::min(a, b) * denominator < std::max(a, b) * numerator) {
                continue;
            }
            int p = 0, q = 0, shared = 0;
            const int target = ((a + b) * numerator + denominator + numerator - 1) / (denominator + numerator);
            while (p < a && q < b && shared < target) {
                if (shared + std::min(a - p, b - q) < target) {
                    break;
                }
                if (left[p] == right[q]) {
                    ++shared;
                    ++p;
                    ++q;
                } else if (left[p] < right[q]) {
                    ++p;
                } else {
                    ++q;
                }
            }
            if (shared >= target) {
                ++matches;
            }
        }
    }
    std::cout << matches << '\n';
}
