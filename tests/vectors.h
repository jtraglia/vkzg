// Minimal reader for the consensus-spec test vectors used by this project:
// `compute_cells_and_kzg_proofs` (a flat sequence of hex literals) and
// `recover_cells_and_kzg_proofs` (which also carries an integer
// `cell_indices` list saying which cells are present). The files are YAML
// but with a very regular shape, so rather than pull in a YAML dependency
// both just scan the quoted hex literals in file order. `output: null`
// marks a case that must be rejected.
//
// For recover_cells_and_kzg_proofs, only the recovered cells (output[0])
// are checked -- vkzg doesn't compute proofs from recovered cells, so
// output[1] is unused.
#pragma once

#include "../include/vkzg.h"
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace vkzg_test {

struct Vector {
    std::string name;
    std::vector<uint8_t> blob;
    std::vector<uint8_t> proofs; // 128 * 48, empty when invalid
    bool valid = false;
};

struct RecoverVector {
    std::string name;
    std::vector<uint8_t> cells;          // num_present * VKZG_BYTES_PER_CELL
    std::vector<uint32_t> cell_indices;  // wire indices the cells above belong to
    std::vector<uint8_t> expected_cells; // VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_CELL, empty when invalid
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

// Every '0x...' literal in file order, as (offset, length-in-hex-chars) spans.
inline std::vector<std::pair<size_t, size_t>> scan_hex_spans(const std::string &text) {
    std::vector<std::pair<size_t, size_t>> spans;
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
    return spans;
}

inline std::vector<uint8_t> hex_span_to_bytes(const std::string &text,
                                               std::pair<size_t, size_t> span) {
    auto nib = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
    std::vector<uint8_t> dst(span.second / 2);
    for (size_t k = 0; k < dst.size(); k++) {
        dst[k] = (uint8_t)((nib(text[span.first + 2 * k]) << 4) | nib(text[span.first + 2 * k + 1]));
    }
    return dst;
}

inline bool parse_vector(const std::string &path, const std::string &name, Vector &v) {
    std::string text;
    if (!read_file(path, text)) return false;
    v.name = name;
    v.valid = text.find("output: null") == std::string::npos;

    auto spans = scan_hex_spans(text);
    if (spans.empty()) return false;

    v.blob = hex_span_to_bytes(text, spans[0]);
    if (!v.valid) return true;
    if (spans.size() != 1 + 128 + 128) return false;

    v.proofs.reserve(128 * 48);
    for (size_t i = 129; i <= 256; i++) {
        auto tmp = hex_span_to_bytes(text, spans[i]);
        if (tmp.size() != 48) return false;
        v.proofs.insert(v.proofs.end(), tmp.begin(), tmp.end());
    }
    return true;
}

inline bool parse_recover_vector(const std::string &path, const std::string &name,
                                  RecoverVector &v) {
    std::string text;
    if (!read_file(path, text)) return false;
    v.name = name;
    v.valid = text.find("output: null") == std::string::npos;

    const size_t idxPos = text.find("cell_indices:");
    if (idxPos == std::string::npos) return false;
    size_t open = text.find('[', idxPos);
    size_t close = text.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return false;
    for (size_t i = open + 1; i < close;) {
        while (i < close && !isdigit((unsigned char)text[i])) i++;
        size_t j = i;
        while (j < close && isdigit((unsigned char)text[j])) j++;
        if (j > i) v.cell_indices.push_back((uint32_t)std::stoul(text.substr(i, j - i)));
        i = j;
    }

    auto spans = scan_hex_spans(text);
    const size_t numPresent = v.cell_indices.size();
    if (spans.size() < numPresent) return false;

    v.cells.reserve(numPresent * VKZG_BYTES_PER_CELL);
    for (size_t i = 0; i < numPresent; i++) {
        auto tmp = hex_span_to_bytes(text, spans[i]);
        if (tmp.size() != VKZG_BYTES_PER_CELL) return false;
        v.cells.insert(v.cells.end(), tmp.begin(), tmp.end());
    }
    if (!v.valid) return true;

    if (spans.size() != numPresent + VKZG_NUM_CELL_PROOFS + VKZG_NUM_CELL_PROOFS) return false;
    v.expected_cells.reserve((size_t)VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_CELL);
    for (size_t i = numPresent; i < numPresent + VKZG_NUM_CELL_PROOFS; i++) {
        auto tmp = hex_span_to_bytes(text, spans[i]);
        if (tmp.size() != VKZG_BYTES_PER_CELL) return false;
        v.expected_cells.insert(v.expected_cells.end(), tmp.begin(), tmp.end());
    }
    return true;
}

// Loads every case in `dir` (a directory of <case-name>/data.yaml).
std::vector<Vector> load_all(const std::string &dir);
std::vector<RecoverVector> load_all_recover(const std::string &dir);

} // namespace vkzg_test
