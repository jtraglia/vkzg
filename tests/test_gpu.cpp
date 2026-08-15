// End-to-end GPU tests against the consensus-spec vectors.
#include "../include/vulkan_prover.h"
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

    vkp_options opts;
    vkp_options_default(&opts);
    opts.table_cache_path = "/tmp/vkp_prover_tables_v3.cache";
    opts.max_batch_size = 4;

    vkp_prover *p = nullptr;
    double t0 = now_ms();
    vkp_result rc = vkp_prover_new_default(&p, &opts);
    if (rc != VKP_OK) {
        printf("FAIL: prover_new: %s\n", vkp_error_string(rc));
        return 1;
    }
    printf("prover ready on %s in %.0f ms\n", vkp_prover_device_name(p), now_ms() - t0);

    auto vectors = vkp_test::load_all(vec_dir);
    if (vectors.empty()) {
        printf("FAIL: no vectors in %s\n", vec_dir);
        return 1;
    }

    const size_t proofBytes = (size_t)VKP_NUM_CELL_PROOFS * VKP_BYTES_PER_PROOF;

    int passed = 0;
    for (const auto &v : vectors) {
        std::vector<uint8_t> proofs(proofBytes);
        vkp_result r = v.blob.size() == VKP_BYTES_PER_BLOB
                             ? vkp_compute_proofs(p, proofs.data(), v.blob.data())
                             : VKP_ERR_BADARGS;
        if (!v.valid) {
            if (r == VKP_OK) {
                printf("FAIL %s: expected rejection, got success\n", v.name.c_str());
                g_failures++;
            } else {
                passed++;
            }
            continue;
        }
        if (r != VKP_OK) {
            printf("FAIL %s: %s\n", v.name.c_str(), vkp_error_string(r));
            g_failures++;
            continue;
        }
        const bool proofs_ok = memcmp(proofs.data(), v.proofs.data(), proofBytes) == 0;
        if (!proofs_ok) {
            printf("FAIL %s: proofs MISMATCH\n", v.name.c_str());
            for (int i = 0; i < 128; i++) {
                if (memcmp(&proofs[i * 48], &v.proofs[i * 48], 48) != 0) {
                    printf("   first bad proof %d\n     got  %s\n     want %s\n", i,
                           hex(&proofs[i * 48], 48).c_str(), hex(&v.proofs[i * 48], 48).c_str());
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
        std::vector<const vkp_test::Vector *> valid;
        for (const auto &v : vectors) {
            if (v.valid && v.blob.size() == VKP_BYTES_PER_BLOB) valid.push_back(&v);
        }
        const size_t n = valid.size();
        if (n >= 2) {
            std::vector<uint8_t> blobs(n * VKP_BYTES_PER_BLOB);
            for (size_t i = 0; i < n; i++) {
                memcpy(&blobs[i * VKP_BYTES_PER_BLOB], valid[i]->blob.data(), VKP_BYTES_PER_BLOB);
            }
            std::vector<uint8_t> proofs(n * proofBytes);
            vkp_result r = vkp_compute_proofs_batch(p, proofs.data(), blobs.data(), n);
            bool ok = r == VKP_OK;
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
        const vkp_test::Vector *v = nullptr;
        for (const auto &x : vectors) {
            if (x.valid && x.blob.size() == VKP_BYTES_PER_BLOB) v = &x;
        }
        std::vector<uint8_t> proofs(proofBytes);

        // Argument validation.
        struct {
            const char *what;
            vkp_result got;
        } checks[] = {
            {"null prover", vkp_compute_proofs(nullptr, proofs.data(), v->blob.data())},
            {"null blob", vkp_compute_proofs(p, proofs.data(), nullptr)},
            {"null proofs", vkp_compute_proofs(p, nullptr, v->blob.data())},
        };
        for (const auto &c : checks) {
            if (c.got != VKP_ERR_BADARGS) {
                printf("FAIL: %s should be rejected, got %s\n", c.what, vkp_error_string(c.got));
                g_failures++;
            } else {
                passed++;
            }
        }
        // Zero blobs is a no-op, not an error.
        if (vkp_compute_proofs_batch(p, proofs.data(), v->blob.data(), 0) != VKP_OK) {
            printf("FAIL: zero-length batch should succeed\n");
            g_failures++;
        } else {
            passed++;
        }

        // A batch larger than max_batch_size must chunk transparently.
        const size_t big = 9; // max_batch_size is 4 above
        std::vector<uint8_t> blobs(big * VKP_BYTES_PER_BLOB), bp(big * proofBytes);
        for (size_t i = 0; i < big; i++) {
            memcpy(&blobs[i * VKP_BYTES_PER_BLOB], v->blob.data(), VKP_BYTES_PER_BLOB);
        }
        bool ok = vkp_compute_proofs_batch(p, bp.data(), blobs.data(), big) == VKP_OK;
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
        std::vector<const vkp_test::Vector *> valid;
        for (const auto &x : vectors) {
            if (x.valid && x.blob.size() == VKP_BYTES_PER_BLOB) valid.push_back(&x);
        }
        std::atomic<int> bad{0};
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; t++) {
            threads.emplace_back([&, t] {
                std::vector<uint8_t> proofs(proofBytes);
                for (int r = 0; r < 3; r++) {
                    const auto *v = valid[(size_t)(t + r) % valid.size()];
                    if (vkp_compute_proofs(p, proofs.data(), v->blob.data()) != VKP_OK ||
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

    vkp_prover_free(p);
    printf("%s: %d checks\n", g_failures ? "FAILED" : "ok", passed);
    return g_failures ? 1 : 0;
}
