// Minimal reader for the consensus-spec `compute_cells_and_kzg_proofs` test
// vectors.  The files are YAML but with a very regular shape, so rather than
// pull in a YAML dependency we just scan the quoted hex literals in order:
// the first is the blob, then 128 cells, then 128 proofs.  `output: null`
// marks a case that must be rejected.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vkzg_test {

struct Vector {
    std::string name;
    std::vector<uint8_t> blob;
    std::vector<uint8_t> cells;  // 128 * 2048, empty when invalid
    std::vector<uint8_t> proofs; // 128 * 48, empty when invalid
    bool valid = false;
};

inline bool read_file(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)n);
    size_t got = fread(&out[0], 1, (size_t)n, f);
    fclose(f);
    return got == (size_t)n;
}

inline bool parse_vector(const std::string &path, const std::string &name, Vector &v) {
    std::string text;
    if (!read_file(path, text)) return false;
    v.name = name;
    v.valid = text.find("output: null") == std::string::npos;

    // Collect every '0x...' literal in file order.
    std::vector<std::pair<size_t, size_t>> spans; // offset, length in hex chars
    for (size_t i = 0; i + 2 < text.size();) {
        if (text[i] == '0' && text[i + 1] == 'x') {
            size_t j = i + 2;
            while (j < text.size() && isxdigit((unsigned char)text[j])) j++;
            spans.emplace_back(i + 2, j - (i + 2));
            i = j;
        } else {
            i++;
        }
    }
    if (spans.empty()) return false;

    auto to_bytes = [&](size_t off, size_t len, std::vector<uint8_t> &dst) {
        auto nib = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
        dst.resize(len / 2);
        for (size_t k = 0; k < len / 2; k++) {
            dst[k] = (uint8_t)((nib(text[off + 2 * k]) << 4) | nib(text[off + 2 * k + 1]));
        }
    };

    to_bytes(spans[0].first, spans[0].second, v.blob);
    if (!v.valid) return true;
    if (spans.size() != 1 + 128 + 128) return false;

    v.cells.reserve(128 * 2048);
    for (size_t i = 1; i <= 128; i++) {
        std::vector<uint8_t> tmp;
        to_bytes(spans[i].first, spans[i].second, tmp);
        if (tmp.size() != 2048) return false;
        v.cells.insert(v.cells.end(), tmp.begin(), tmp.end());
    }
    v.proofs.reserve(128 * 48);
    for (size_t i = 129; i <= 256; i++) {
        std::vector<uint8_t> tmp;
        to_bytes(spans[i].first, spans[i].second, tmp);
        if (tmp.size() != 48) return false;
        v.proofs.insert(v.proofs.end(), tmp.begin(), tmp.end());
    }
    return true;
}

// Loads every case in `dir` (a directory of <case-name>/data.yaml).
std::vector<Vector> load_all(const std::string &dir);

} // namespace vkzg_test
