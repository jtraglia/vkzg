// Latency and throughput benchmark for the GPU prover.
#include "../include/vkzg.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Deterministic pseudo-random canonical blobs.
static void fill_blob(uint8_t *blob, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
    for (int i = 0; i < VKZG_FIELD_ELEMENTS_PER_BLOB; i++) {
        uint8_t *fe = blob + (size_t)i * 32;
        for (int j = 0; j < 32; j++) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            fe[j] = (uint8_t)(s >> 24);
        }
        fe[0] = 0; // keep every element below r
    }
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    const uint32_t maxBatch = argc > 2 ? (uint32_t)atoi(argv[2]) : 8;

    vkzg_options opts;
    vkzg_options_default(&opts);
    opts.max_batch_size = maxBatch;

    vkzg_prover *p = nullptr;
    double t0 = now_ms();
    vkzg_result rc = vkzg_prover_new(&p, &opts);
    if (rc != VKZG_OK) {
        printf("prover_new failed: %s\n", vkzg_error_string(rc));
        return 1;
    }
    printf("device: %s   setup: %.0f ms\n\n", vkzg_prover_device_name(p), now_ms() - t0);

    const size_t proofBytes = (size_t)VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_PROOF;

    std::vector<uint8_t> blobs((size_t)maxBatch * VKZG_BYTES_PER_BLOB);
    for (uint32_t i = 0; i < maxBatch; i++) {
        fill_blob(&blobs[(size_t)i * VKZG_BYTES_PER_BLOB], i + 1);
    }
    std::vector<uint8_t> proofs((size_t)maxBatch * proofBytes);

    auto run = [&](const char *label, uint32_t n) {
        // warm up
        vkzg_compute_proofs(p, proofs.data(), blobs.data(), n);
        double best = 1e30, total = 0;
        for (int r = 0; r < reps; r++) {
            double a = now_ms();
            vkzg_result e = vkzg_compute_proofs(p, proofs.data(), blobs.data(), n);
            double ms = now_ms() - a;
            if (e != VKZG_OK) {
                printf("  %s: error %s\n", label, vkzg_error_string(e));
                return;
            }
            best = std::min(best, ms);
            total += ms;
        }
        printf("  %-8s  best %7.2f ms   avg %7.2f ms   per blob %6.2f ms\n", label, best,
               total / reps, best / n);
    };

    for (uint32_t n = 1; n <= maxBatch; n *= 2) {
        char label[64];
        snprintf(label, sizeof(label), "%u blob%s", n, n == 1 ? "" : "s");
        run(label, n);
    }

    vkzg_prover_free(p);
    return 0;
}
