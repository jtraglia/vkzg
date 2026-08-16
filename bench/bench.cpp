// Latency and throughput benchmark for the GPU prover.
#include "../include/vkzg.h"
#include "bench_common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    const int reps = 10;
    const uint32_t maxBatch = 256;

    vkzg_options opts;
    vkzg_options_default(&opts);
    opts.max_batch_size = maxBatch;

    vkzg_prover *p = nullptr;
    vkzg_result rc = vkzg_prover_new(&p, &opts);
    if (rc != VKZG_OK) {
        printf("prover_new failed: %s\n", vkzg_error_string(rc));
        return 1;
    }
    printf("device: %s (%u GPU cores)\n\n", vkzg_prover_device_name(p),
           vkzg_prover_gpu_core_count(p));

    const size_t proofBytes = (size_t)VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_PROOF;

    std::vector<uint8_t> blobs((size_t)maxBatch * VKZG_BYTES_PER_BLOB);
    for (uint32_t i = 0; i < maxBatch; i++) {
        fill_blob(&blobs[(size_t)i * VKZG_BYTES_PER_BLOB], i + 1);
    }
    std::vector<uint8_t> proofs((size_t)maxBatch * proofBytes);

    auto run = [&](uint32_t n) {
        // warm up
        vkzg_compute_proofs(p, proofs.data(), blobs.data(), n);
        double best = 1e30, total = 0;
        for (int r = 0; r < reps; r++) {
            double a = now_ms();
            vkzg_result e = vkzg_compute_proofs(p, proofs.data(), blobs.data(), n);
            double ms = now_ms() - a;
            if (e != VKZG_OK) {
                printf("  blobs: %3u   error: %s\n", n, vkzg_error_string(e));
                return;
            }
            best = std::min(best, ms);
            total += ms;
        }
        printf("  blobs: %3u   best: %9.2f ms   avg: %9.2f ms   per blob: %9.2f ms\n", n, best,
               total / reps, best / n);
    };

    for (uint32_t n = 1; n <= maxBatch; n *= 2) {
        run(n);
    }

    vkzg_prover_free(p);
    return 0;
}
