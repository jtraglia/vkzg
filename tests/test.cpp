// End-to-end GPU tests against the consensus-spec vectors.
#include "../include/vkzg.h"
#include "vectors.h"

#include <atomic>
#include <chrono>
#include <thread>
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
    const char *vec_dir = argc > 1 ? argv[1] : "tests/vectors";

    vkzg_options opts;
    vkzg_options_default(&opts);
    opts.max_batch_size = 4;

    vkzg_prover *p = nullptr;
    double t0 = now_ms();
    vkzg_result rc = vkzg_prover_new(&p, &opts);
    if (rc != VKZG_OK) {
        printf("FAIL: prover_new: %s\n", vkzg_error_string(rc));
        return 1;
    }
    printf("prover ready on %s in %.0f ms\n", vkzg_prover_device_name(p), now_ms() - t0);

    auto vectors = vkzg_test::load_all(vec_dir);
    if (vectors.empty()) {
        printf("FAIL: no vectors in %s\n", vec_dir);
        return 1;
    }

    const size_t proofBytes = (size_t)VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_PROOF;

    int passed = 0;
    for (const auto &v : vectors) {
        std::vector<uint8_t> proofs(proofBytes);
        vkzg_result r = v.blob.size() == VKZG_BYTES_PER_BLOB
                             ? vkzg_compute_proofs_batch(p, proofs.data(), v.blob.data(), 1)
                             : VKZG_ERR_BADARGS;
        if (!v.valid) {
            if (r == VKZG_OK) {
                printf("FAIL %s: expected rejection, got success\n", v.name.c_str());
                g_failures++;
            } else {
                passed++;
            }
            continue;
        }
        if (r != VKZG_OK) {
            printf("FAIL %s: %s\n", v.name.c_str(), vkzg_error_string(r));
            g_failures++;
            continue;
        }
        const bool proofs_ok = memcmp(proofs.data(), v.proofs.data(), proofBytes) == 0;
        if (!proofs_ok) {
            printf("FAIL %s: proofs MISMATCH\n", v.name.c_str());
            for (int i = 0; i < VKZG_NUM_CELL_PROOFS; i++) {
                const size_t off = (size_t)i * VKZG_BYTES_PER_PROOF;
                if (memcmp(&proofs[off], &v.proofs[off], VKZG_BYTES_PER_PROOF) != 0) {
                    printf("   first bad proof %d\n     got  %s\n     want %s\n", i,
                           hex(&proofs[off], VKZG_BYTES_PER_PROOF).c_str(),
                           hex(&v.proofs[off], VKZG_BYTES_PER_PROOF).c_str());
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
        std::vector<const vkzg_test::Vector *> valid;
        for (const auto &v : vectors) {
            if (v.valid && v.blob.size() == VKZG_BYTES_PER_BLOB) valid.push_back(&v);
        }
        const size_t n = valid.size();
        if (n >= 2) {
            std::vector<uint8_t> blobs(n * VKZG_BYTES_PER_BLOB);
            for (size_t i = 0; i < n; i++) {
                memcpy(&blobs[i * VKZG_BYTES_PER_BLOB], valid[i]->blob.data(), VKZG_BYTES_PER_BLOB);
            }
            std::vector<uint8_t> proofs(n * proofBytes);
            vkzg_result r = vkzg_compute_proofs_batch(p, proofs.data(), blobs.data(), n);
            bool ok = r == VKZG_OK;
            for (size_t i = 0; ok && i < n; i++) {
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

    // ---------------------------------------------------------- API contract
    {
        const vkzg_test::Vector *v = nullptr;
        for (const auto &x : vectors) {
            if (x.valid && x.blob.size() == VKZG_BYTES_PER_BLOB) v = &x;
        }
        std::vector<uint8_t> proofs(proofBytes);

        // Argument validation.
        struct {
            const char *what;
            vkzg_result got;
        } checks[] = {
            {"null prover", vkzg_compute_proofs_batch(nullptr, proofs.data(), v->blob.data(), 1)},
            {"null blob", vkzg_compute_proofs_batch(p, proofs.data(), nullptr, 1)},
            {"null proofs", vkzg_compute_proofs_batch(p, nullptr, v->blob.data(), 1)},
        };
        for (const auto &c : checks) {
            if (c.got != VKZG_ERR_BADARGS) {
                printf("FAIL: %s should be rejected, got %s\n", c.what, vkzg_error_string(c.got));
                g_failures++;
            } else {
                passed++;
            }
        }
        // Zero blobs is a no-op, not an error.
        if (vkzg_compute_proofs_batch(p, proofs.data(), v->blob.data(), 0) != VKZG_OK) {
            printf("FAIL: zero-length batch should succeed\n");
            g_failures++;
        } else {
            passed++;
        }

        // A batch larger than max_batch_size must chunk transparently.
        const size_t big = opts.max_batch_size * 2 + 1;
        std::vector<uint8_t> blobs(big * VKZG_BYTES_PER_BLOB), bp(big * proofBytes);
        for (size_t i = 0; i < big; i++) {
            memcpy(&blobs[i * VKZG_BYTES_PER_BLOB], v->blob.data(), VKZG_BYTES_PER_BLOB);
        }
        bool ok = vkzg_compute_proofs_batch(p, bp.data(), blobs.data(), big) == VKZG_OK;
        for (size_t i = 0; ok && i < big; i++) {
            ok &= memcmp(&bp[i * proofBytes], v->proofs.data(), proofBytes) == 0;
        }
        if (!ok) {
            printf("FAIL: batch of %zu (> max_batch_size) did not chunk correctly\n", big);
            g_failures++;
        } else {
            printf("ok  : chunked batch of %zu\n", big);
            passed++;
        }
    }

    // ------------------------------------------------------------ concurrency
    // The header promises a prover may be shared between threads.
    {
        std::vector<const vkzg_test::Vector *> valid;
        for (const auto &x : vectors) {
            if (x.valid && x.blob.size() == VKZG_BYTES_PER_BLOB) valid.push_back(&x);
        }
        std::atomic<int> bad{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; t++) {
            threads.emplace_back([&, t] {
                std::vector<uint8_t> proofs(proofBytes);
                for (int r = 0; r < 3; r++) {
                    const auto *v = valid[(size_t)(t + r) % valid.size()];
                    if (vkzg_compute_proofs_batch(p, proofs.data(), v->blob.data(), 1) != VKZG_OK ||
                        memcmp(proofs.data(), v->proofs.data(), proofBytes) != 0) {
                        bad++;
                    }
                }
            });
        }
        for (auto &th : threads) th.join();
        if (bad) {
            printf("FAIL: %d concurrent calls returned wrong results\n", bad.load());
            g_failures++;
        } else {
            printf("ok  : 4 threads x 3 calls concurrently\n");
            passed++;
        }
    }

    // ------------------------------------------------------------ cell recovery
    {
        const size_t cellBytes = (size_t)VKZG_NUM_CELL_PROOFS * VKZG_BYTES_PER_CELL;
        auto recoverVectors = vkzg_test::load_all_recover(vec_dir);
        if (recoverVectors.empty()) {
            printf("FAIL: no recover_cells_and_kzg_proofs vectors in %s\n", vec_dir);
            g_failures++;
        }

        std::vector<const vkzg_test::RecoverVector *> validVecs;
        for (const auto &v : recoverVectors) {
            std::vector<uint8_t> cells(cellBytes, 0);
            std::vector<uint8_t> present(VKZG_NUM_CELL_PROOFS, 0);
            for (size_t i = 0; i < v.cell_indices.size(); i++) {
                const uint32_t idx = v.cell_indices[i];
                memcpy(&cells[(size_t)idx * VKZG_BYTES_PER_CELL], &v.cells[i * VKZG_BYTES_PER_CELL],
                       VKZG_BYTES_PER_CELL);
                present[idx] = 1;
            }
            std::vector<uint8_t> out(cellBytes);
            vkzg_result r = vkzg_recover_cells_batch(p, out.data(), cells.data(), present.data(), 1);
            if (!v.valid) {
                if (r == VKZG_OK) {
                    printf("FAIL %s: expected rejection, got success\n", v.name.c_str());
                    g_failures++;
                } else {
                    passed++;
                }
                continue;
            }
            if (r != VKZG_OK) {
                printf("FAIL %s: %s\n", v.name.c_str(), vkzg_error_string(r));
                g_failures++;
                continue;
            }
            if (memcmp(out.data(), v.expected_cells.data(), cellBytes) != 0) {
                printf("FAIL %s: recovered cells MISMATCH\n", v.name.c_str());
                g_failures++;
            } else {
                printf("ok  : %s\n", v.name.c_str());
                passed++;
                validVecs.push_back(&v);
            }
        }

        // Batched path: every valid recover vector's blob in one call.
        if (validVecs.size() >= 2) {
            const size_t n = validVecs.size();
            std::vector<uint8_t> cells(n * cellBytes, 0), present(n * VKZG_NUM_CELL_PROOFS, 0);
            for (size_t b = 0; b < n; b++) {
                const auto &v = *validVecs[b];
                for (size_t i = 0; i < v.cell_indices.size(); i++) {
                    const uint32_t idx = v.cell_indices[i];
                    memcpy(&cells[b * cellBytes + (size_t)idx * VKZG_BYTES_PER_CELL],
                           &v.cells[i * VKZG_BYTES_PER_CELL], VKZG_BYTES_PER_CELL);
                    present[b * VKZG_NUM_CELL_PROOFS + idx] = 1;
                }
            }
            std::vector<uint8_t> out(n * cellBytes);
            vkzg_result r = vkzg_recover_cells_batch(p, out.data(), cells.data(), present.data(), n);
            bool ok = r == VKZG_OK;
            for (size_t b = 0; ok && b < n; b++) {
                ok &= memcmp(&out[b * cellBytes], validVecs[b]->expected_cells.data(), cellBytes) == 0;
            }
            if (!ok) {
                printf("FAIL: batched recovery (%zu blobs) disagrees with the vectors\n", n);
                g_failures++;
            } else {
                printf("ok  : batched recovery, %zu blobs\n", n);
                passed++;
            }
        }
    }

    vkzg_prover_free(p);
    printf("%s: %d checks\n", g_failures ? "FAILED" : "ok", passed);
    return g_failures ? 1 : 0;
}
