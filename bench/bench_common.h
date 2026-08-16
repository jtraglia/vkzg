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
        fe[0] = 0; // keep every element below r
    }
}
