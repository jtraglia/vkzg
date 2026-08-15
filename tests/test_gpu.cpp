// End-to-end GPU tests against the consensus-spec vectors.
#include "../include/kzgpu.h"
#include "vectors.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::string hex(const uint8_t *b, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

int main(int argc, char **argv) {
    const char *setup_path = argc > 1 ? argv[1] : "data/trusted_setup.txt";
    const char *vec_dir = argc > 2 ? argv[2] : "tests/vectors";

    kzgpu_options opts;
    kzgpu_options_default(&opts);
    opts.table_cache_path = "/tmp/kzgpu_tables_v3.cache";
    opts.max_batch_size = 4;

    kzgpu_prover *p = nullptr;
    double t0 = now_ms();
    kzgpu_result rc = kzgpu_prover_new_from_file(&p, setup_path, &opts);
    if (rc != KZGPU_OK) {
        printf("FAIL: prover_new: %s\n", kzgpu_error_string(rc));
        return 1;
    }
    printf("prover ready on %s in %.0f ms\n", kzgpu_device_name(p), now_ms() - t0);

    auto vectors = kzgpu_test::load_all(vec_dir);
    if (vectors.empty()) {
        printf("FAIL: no vectors in %s\n", vec_dir);
        return 1;
    }

    const size_t cellBytes = (size_t)KZGPU_CELLS_PER_EXT_BLOB * KZGPU_BYTES_PER_CELL;
    const size_t proofBytes = (size_t)KZGPU_CELLS_PER_EXT_BLOB * KZGPU_BYTES_PER_PROOF;

    int passed = 0;
    for (const auto &v : vectors) {
        std::vector<uint8_t> cells(cellBytes), proofs(proofBytes);
        kzgpu_result r = v.blob.size() == KZGPU_BYTES_PER_BLOB
                             ? kzgpu_compute_cells_and_proofs(p, cells.data(), proofs.data(),
                                                              v.blob.data())
                             : KZGPU_ERR_BADARGS;
        if (!v.valid) {
            if (r == KZGPU_OK) {
                printf("FAIL %s: expected rejection, got success\n", v.name.c_str());
                g_failures++;
            } else {
                passed++;
            }
            continue;
        }
        if (r != KZGPU_OK) {
            printf("FAIL %s: %s\n", v.name.c_str(), kzgpu_error_string(r));
            g_failures++;
            continue;
        }
        const bool cells_ok = memcmp(cells.data(), v.cells.data(), cellBytes) == 0;
        const bool proofs_ok = memcmp(proofs.data(), v.proofs.data(), proofBytes) == 0;
        if (!cells_ok || !proofs_ok) {
            printf("FAIL %s: cells %s, proofs %s\n", v.name.c_str(), cells_ok ? "ok" : "MISMATCH",
                   proofs_ok ? "ok" : "MISMATCH");
            for (int i = 0; i < 128 && !proofs_ok; i++) {
                if (memcmp(&proofs[i * 48], &v.proofs[i * 48], 48) != 0) {
                    printf("   first bad proof %d\n     got  %s\n     want %s\n", i,
                           hex(&proofs[i * 48], 48).c_str(), hex(&v.proofs[i * 48], 48).c_str());
                    break;
                }
            }
            for (int i = 0; i < 8192 && !cells_ok; i++) {
                if (memcmp(&cells[i * 32], &v.cells[i * 32], 32) != 0) {
                    printf("   first bad field element %d (cell %d, elem %d)\n", i, i / 64, i % 64);
                    printf("     got  %s\n     want %s\n", hex(&cells[i * 32], 32).c_str(),
                           hex(&v.cells[i * 32], 32).c_str());
                    break;
                }
            }
            g_failures++;
        } else {
            printf("ok  : %s\n", v.name.c_str());
            passed++;
        }
    }

    // Batched path must agree with the single-blob path.
    {
        std::vector<const kzgpu_test::Vector *> valid;
        for (const auto &v : vectors) {
            if (v.valid && v.blob.size() == KZGPU_BYTES_PER_BLOB) valid.push_back(&v);
        }
        const size_t n = valid.size();
        if (n >= 2) {
            std::vector<uint8_t> blobs(n * KZGPU_BYTES_PER_BLOB);
            for (size_t i = 0; i < n; i++) {
                memcpy(&blobs[i * KZGPU_BYTES_PER_BLOB], valid[i]->blob.data(), KZGPU_BYTES_PER_BLOB);
            }
            std::vector<uint8_t> cells(n * cellBytes), proofs(n * proofBytes);
            kzgpu_result r = kzgpu_compute_cells_and_proofs_batch(p, cells.data(), proofs.data(),
                                                                  blobs.data(), n);
            bool ok = r == KZGPU_OK;
            for (size_t i = 0; ok && i < n; i++) {
                ok &= memcmp(&cells[i * cellBytes], valid[i]->cells.data(), cellBytes) == 0;
                ok &= memcmp(&proofs[i * proofBytes], valid[i]->proofs.data(), proofBytes) == 0;
            }
            if (!ok) {
                printf("FAIL: batched path (%zu blobs) disagrees with the vectors\n", n);
                g_failures++;
            } else {
                printf("ok  : batched path, %zu blobs\n", n);
                passed++;
            }
        }
    }

    kzgpu_prover_free(p);
    printf("%s: %d checks\n", g_failures ? "FAILED" : "ok", passed);
    return g_failures ? 1 : 0;
}
