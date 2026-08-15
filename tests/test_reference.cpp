// Validates the CPU reference pipeline (and therefore the algorithm the GPU
// implements) against the consensus-spec test vectors.
//
// Two things get proven here:
//   1. the fused circulant kernel is *equivalent* to c-kzg's
//      "G1 inverse transform, truncate, forward transform", and
//   2. the whole pipeline reproduces the official cells and proofs.
#include "../src/cpu/reference.h"
#include "../src/cpu/setup.h"
#include "../src/setup_data.h"
#include "vectors.h"

#include <chrono>
#include <cstdio>
#include <cstring>

using namespace kzgpu;

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

// Confirms the fused circulant map equals the two-G1-FFT formulation on random
// input. This is the crux of the whole design, so it is checked directly.
static void test_circulant_equivalence(const SetupTables &tables) {
    // Build a deterministic pseudo-random set of 128 G1 points.
    G1 u[kCirculantSize];
    G1 g;
    {
        static const uint8_t gen[48] = {0x97, 0xf1, 0xd3, 0xa7, 0x31, 0x97, 0xd7, 0x94,
                                        0x26, 0x95, 0x63, 0x8c, 0x4f, 0xa9, 0xac, 0x0f,
                                        0xc3, 0x68, 0x8c, 0x4f, 0x97, 0x74, 0xb9, 0x05,
                                        0xa1, 0x4e, 0x3a, 0x3f, 0x17, 0x1b, 0xac, 0x58,
                                        0x6c, 0x55, 0xe8, 0x3f, 0xf9, 0x7a, 0x1a, 0xef,
                                        0xfb, 0x3a, 0xf0, 0x0a, 0xdb, 0x22, 0xc6, 0xbb};
        G1Affine ga;
        if (!g1_decompress(ga, gen)) {
            printf("FAIL: could not decompress generator\n");
            g_failures++;
            return;
        }
        g1_from_affine(g, ga);
    }
    for (int i = 0; i < kCirculantSize; i++) {
        Fr k;
        fr_from_u64(k, (uint64_t)(i * 2654435761u + 12345u));
        g1_mul(u[i], g, k);
    }

    G1 viaFft[kCirculantSize], viaCirc[kCirculantSize];
    phase_b_via_g1_ffts(viaFft, u, tables);
    phase_b_circulant(viaCirc, u, tables);
    int bad = 0;
    for (int i = 0; i < kCirculantSize; i++) {
        if (!g1_eq(viaFft[i], viaCirc[i])) bad++;
    }
    if (bad) {
        printf("FAIL: fused circulant differs from G1-FFT form at %d/%d outputs\n", bad,
               kCirculantSize);
        g_failures++;
    } else {
        printf("ok  : fused circulant == IFFT/truncate/FFT over G1 (128/128 outputs)\n");
    }
}

int main(int argc, char **argv) {
    const char *vec_dir = argc > 1 ? argv[1] : "tests/vectors";

    SetupTables tables;
    double t0 = now_ms();
    // Reuse a cache so repeated runs during development are quick.
    const std::string cache = "/tmp/kzgpu_tables_ref.cache";
    const uint64_t digest = compute_setup_digest(kEmbeddedSetupG1Monomial, kEmbeddedSetupSize);
    bool loaded = load_table_cache(cache, digest, tables) == KZGPU_OK;
    if (!loaded) {
        if (build_setup_tables(kEmbeddedSetupG1Monomial, kEmbeddedSetupSize, true, tables) !=
            KZGPU_OK) {
            printf("FAIL: build_setup_tables\n");
            return 1;
        }
        save_table_cache(cache, tables);
    }
    printf("setup tables %s in %.0f ms (%.1f MiB position table)\n", loaded ? "loaded" : "built",
           now_ms() - t0, tables.position_table.size() * 4.0 / (1024 * 1024));

    test_circulant_equivalence(tables);

    auto vectors = kzgpu_test::load_all(vec_dir);
    if (vectors.empty()) {
        printf("FAIL: no test vectors found in %s\n", vec_dir);
        return 1;
    }

    int passed = 0;
    for (const auto &v : vectors) {
        std::vector<uint8_t> cells(128 * 2048), proofs(128 * 48);
        double a = now_ms();
        // The library takes a fixed-size blob buffer, so a wrong length is the
        // caller's error to catch -- two of the spec vectors exercise exactly
        // that, and the harness stands in for the caller here.
        kzgpu_result rc = v.blob.size() == (size_t)kFieldElementsPerBlob * kBytesPerFieldElement
                              ? reference_compute(tables, v.blob.data(), cells.data(), proofs.data())
                              : KZGPU_ERR_BADARGS;
        double ms = now_ms() - a;

        if (!v.valid) {
            if (rc == KZGPU_OK) {
                printf("FAIL %s: expected rejection, got success\n", v.name.c_str());
                g_failures++;
            } else {
                passed++;
            }
            continue;
        }
        if (rc != KZGPU_OK) {
            printf("FAIL %s: unexpected error %d\n", v.name.c_str(), (int)rc);
            g_failures++;
            continue;
        }
        bool cells_ok = memcmp(cells.data(), v.cells.data(), cells.size()) == 0;
        bool proofs_ok = memcmp(proofs.data(), v.proofs.data(), proofs.size()) == 0;
        if (!cells_ok || !proofs_ok) {
            printf("FAIL %s: cells %s, proofs %s\n", v.name.c_str(), cells_ok ? "ok" : "MISMATCH",
                   proofs_ok ? "ok" : "MISMATCH");
            if (!proofs_ok) {
                for (int i = 0; i < 128; i++) {
                    if (memcmp(&proofs[i * 48], &v.proofs[i * 48], 48) != 0) {
                        printf("   first bad proof %d:\n     got %s\n     want %s\n", i,
                               hex(&proofs[i * 48], 48).c_str(), hex(&v.proofs[i * 48], 48).c_str());
                        break;
                    }
                }
            }
            if (!cells_ok) {
                for (int i = 0; i < 128; i++) {
                    if (memcmp(&cells[i * 2048], &v.cells[i * 2048], 2048) != 0) {
                        printf("   first bad cell %d\n", i);
                        break;
                    }
                }
            }
            g_failures++;
        } else {
            printf("ok  : %-52s (%.0f ms)\n", v.name.c_str(), ms);
            passed++;
        }
    }

    printf("%s: %d/%zu vectors\n", g_failures ? "FAILED" : "ok", passed, vectors.size());
    return g_failures ? 1 : 0;
}
