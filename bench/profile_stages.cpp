// Per-stage GPU timing (development tool). Flushes the GPU between every
// dispatch for honest per-stage numbers, at the cost of real submission
// overhead each stage pays that the normal (single-command-buffer) path doesn't.
#include "../include/vkzg.h"
#include "../src/profile.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void fill_blob(uint8_t *blob, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (int i = 0; i < VKZG_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * 32;
        for (int j = 0; j < 32; j++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; fe[j] = (uint8_t)(s >> 24); }
        fe[0] = 0;
    }
}

int main(int argc, char **argv) {
    unsigned batch = argc > 1 ? (unsigned)atoi(argv[1]) : 8;
    int reps = argc > 2 ? atoi(argv[2]) : 6;

    vkzg_options o;
    vkzg_options_default(&o);
    o.max_batch_size = batch;
    vkzg_prover *p = nullptr;
    if (vkzg_prover_new(&p, &o) != VKZG_OK) { printf("setup failed\n"); return 1; }

    std::vector<uint8_t> blobs((size_t)batch * VKZG_BYTES_PER_BLOB);
    for (unsigned i = 0; i < batch; i++) fill_blob(&blobs[(size_t)i * VKZG_BYTES_PER_BLOB], i + 1);
    std::vector<uint8_t> proofs((size_t)batch * 128 * 48);

    vkzg::StageTimes best;
    bool first = true;
    for (int r = 0; r < reps; r++) {
        vkzg::StageTimes t;
        if (vkzg::profile_batch(p, proofs.data(), blobs.data(), batch, t) != VKZG_OK) {
            printf("compute failed\n");
            return 1;
        }
        if (first || t.total < best.total) { best = t; first = false; }
    }

    printf("batch = %u\n\n", batch);
    struct { const char *n; double v; } rows[] = {
        {"scalar stage (NTTs)", best.scalar_stage},
        {"phase A  (sort + MSM)", best.phase_a},
        {"  reduce A", best.reduce_a},
        {"ladder", best.ladder},
        {"  fold ladder (split)", best.fold_ladder},
        {"  normalize ladder", best.normalize_ladder},
        {"phase B  (cyclic + negacyclic)", best.phase_b},
        {"  reduce B", best.reduce_b},
        {"  combine (split)", best.combine},
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
    vkzg_prover_free(p);
    return 0;
}
