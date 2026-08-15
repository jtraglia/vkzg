// Per-stage GPU timing (development tool).  Runs the real dispatch sequence
// with the command buffer flushed at stage boundaries, so each figure includes
// a little submission overhead but attributes time correctly.
#include "../include/kzgpu.h"
#include "../src/kzgpu_profile.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void fill_blob(uint8_t *blob, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (int i = 0; i < KZGPU_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * 32;
        for (int j = 0; j < 32; j++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; fe[j] = (uint8_t)(s >> 24); }
        fe[0] = 0;
    }
}

int main(int argc, char **argv) {
    const char *setup = argc > 1 ? argv[1] : "data/trusted_setup.txt";
    unsigned batch = argc > 2 ? (unsigned)atoi(argv[2]) : 8;
    int reps = argc > 3 ? atoi(argv[3]) : 6;

    kzgpu_options o;
    kzgpu_options_default(&o);
    o.table_cache_path = "/tmp/kzgpu_tables_v3.cache";
    o.max_batch_size = batch;
    kzgpu_prover *p = nullptr;
    if (kzgpu_prover_new_from_file(&p, setup, &o) != KZGPU_OK) { printf("setup failed\n"); return 1; }

    std::vector<uint8_t> blobs((size_t)batch * KZGPU_BYTES_PER_BLOB);
    for (unsigned i = 0; i < batch; i++) fill_blob(&blobs[(size_t)i * KZGPU_BYTES_PER_BLOB], i + 1);
    std::vector<uint8_t> cells((size_t)batch * 128 * 2048), proofs((size_t)batch * 128 * 48);

    kzgpu::StageTimes best;
    bool first = true;
    for (int r = 0; r < reps; r++) {
        kzgpu::StageTimes t;
        if (kzgpu::profile_batch(p, cells.data(), proofs.data(), blobs.data(), batch, t) != KZGPU_OK) {
            printf("compute failed\n");
            return 1;
        }
        if (first || t.total < best.total) { best = t; first = false; }
    }

    printf("batch = %u   (stage-split; each figure includes ~0.1ms submission)\n\n", batch);
    struct { const char *n; double v; } rows[] = {
        {"scalar stage (NTTs)", best.scalar_stage},
        {"phase A  (sort + MSM)", best.phase_a},
        {"  reduce A", best.reduce_a},
        {"ladder", best.ladder},
        {"  normalize ladder", best.normalize_ladder},
        {"phase B  (circulant)", best.phase_b},
        {"  reduce B", best.reduce_b},
        {"  normalize proofs", best.normalize_proofs},
        {"  compress proofs", best.compress},
    };
    double sum = 0;
    for (auto &r : rows) {
        printf("  %-24s %8.2f ms  %5.1f%%\n", r.n, r.v, 100.0 * r.v / best.total);
        sum += r.v;
    }
    printf("  %-24s %8.2f ms\n", "accounted", sum);
    printf("  %-24s %8.2f ms  (%.2f ms/blob)\n", "TOTAL (wall)", best.total, best.total / batch);
    kzgpu_prover_free(p);
    return 0;
}
