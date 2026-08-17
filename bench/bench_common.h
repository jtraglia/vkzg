// Shared helpers for bench.cpp and profile.cpp.
#pragma once

#include "../include/vkzg.h"

#include <chrono>
#include <cstdint>

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Deterministic pseudo-random canonical blob.
inline void fill_blob(uint8_t *blob, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (int i = 0; i < VKZG_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * VKZG_BYTES_PER_FIELD_ELEMENT;
        for (int j = 0; j < VKZG_BYTES_PER_FIELD_ELEMENT; j++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            fe[j] = (uint8_t)(s >> 24);
        }
        // The modulus r's top byte is 0x73 (0b0111_0011); clearing just its top
        // two bits (rather than the whole byte) is enough to guarantee every
        // element is below r regardless of the remaining bytes, while keeping
        // 6 bits of real entropy instead of 0.
        fe[0] &= 0x3F;
    }
}

// Deterministic pseudo-random canonical full 128-cell extended array (the
// same shape vkzg_recover_cells_batch expects: cell recovery never sees an actual
// blob, only cells). Real, non-zero data matters here -- an all-zero array
// would make every scalar multiply in the pipeline degenerate and give
// misleadingly fast timings.
inline void fill_cells(uint8_t *cells, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    const int totalElements = VKZG_NUM_CELL_PROOFS * VKZG_FIELD_ELEMENTS_PER_CELL;
    for (int i = 0; i < totalElements; i++) {
        uint8_t *fe = cells + (size_t)i * VKZG_BYTES_PER_FIELD_ELEMENT;
        for (int j = 0; j < VKZG_BYTES_PER_FIELD_ELEMENT; j++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            fe[j] = (uint8_t)(s >> 24);
        }
        // The modulus r's top byte is 0x73 (0b0111_0011); clearing just its top
        // two bits (rather than the whole byte) is enough to guarantee every
        // element is below r regardless of the remaining bytes, while keeping
        // 6 bits of real entropy instead of 0.
        fe[0] &= 0x3F;
    }
}
