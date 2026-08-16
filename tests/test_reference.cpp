// Validates the CPU reference pipeline (and therefore the algorithm the GPU
// implements) against the consensus-spec test vectors.
//
// Two things get proven here:
//   1. the fused circulant kernel is *equivalent* to c-kzg's
//      "G1 inverse transform, truncate, forward transform", and
//   2. the whole pipeline reproduces the official cell proofs.
#include "../src/cpu/reference.h"
#include "../src/cpu/setup.h"
#include "../src/setup_data.h"
#include "vectors.h"

#include <chrono>
#include <cstdio>
#include <cstring>

using namespace vkp;

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

// Prototype/validation for a not-yet-implemented optimization: splitting the
// 128-point circulant convolution via X^128-1 = (X^64-1)(X^64+1) into two
// 64-point sub-convolutions (a plain cyclic one and a negacyclic one), which
// in principle roughly halves the tap*output work phase B does. Deliberately
// written as a direct O(n^2) scalar-multiply, with no bucket/digit encoding
// at all, so this checks only the *mathematical* claim -- that the split
// reconstructs the same output as the trusted flat form -- independent of
// any GPU-side bucket-MSM machinery.
static void test_split_circulant_equivalence(const SetupTables &tables) {
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
            printf("FAIL: could not decompress generator (split test)\n");
            g_failures++;
            return;
        }
        g1_from_affine(g, ga);
    }
    for (int i = 0; i < kCirculantSize; i++) {
        Fr k;
        // Different constants from test_circulant_equivalence, so this isn't
        // just re-checking the same input.
        fr_from_u64(k, (uint64_t)(i * 1000000007u + 998244353u));
        g1_mul(u[i], g, k);
    }

    G1 want[kCirculantSize];
    phase_b_circulant(want, u, tables);

    Fr kappa[kCirculantSize];
    compute_circulant_kernel(kappa);
    const int H = kCirculantSize / 2; // 64

    Fr two, half;
    fr_from_u64(two, 2);
    fr_inv(half, two);

    Fr kappa_plus[64], kappa_minus[64];
    for (int i = 0; i < H; i++) {
        Fr sum, diff;
        fr_add(sum, kappa[i], kappa[i + H]);
        fr_sub(diff, kappa[i], kappa[i + H]);
        fr_mul(kappa_plus[i], sum, half);
        fr_mul(kappa_minus[i], diff, half);
    }

    G1 u_plus[64], u_minus[64];
    for (int j = 0; j < H; j++) {
        g1_add(u_plus[j], u[j], u[j + H]);
        g1_sub(u_minus[j], u[j], u[j + H]);
    }

    G1 got[kCirculantSize];
    for (int a = 0; a < H; a++) {
        G1 c_plus = kG1Identity, c_minus = kG1Identity;
        for (int e = 0; e < H; e++) {
            if (!fr_is_zero(kappa_plus[e])) {
                const int src = ((a - e) % H + H) % H;
                G1 term;
                g1_mul(term, u_plus[src], kappa_plus[e]);
                g1_add(c_plus, c_plus, term);
            }
            if (!fr_is_zero(kappa_minus[e])) {
                const int src = ((a - e) % H + H) % H;
                G1 term;
                g1_mul(term, u_minus[src], kappa_minus[e]);
                // Negacyclic: wrap-around (e > a) flips the sign.
                if (e > a) {
                    G1 neg;
                    g1_neg(neg, term);
                    g1_add(c_minus, c_minus, neg);
                } else {
                    g1_add(c_minus, c_minus, term);
                }
            }
        }
        g1_add(got[a], c_plus, c_minus);
        g1_sub(got[a + H], c_plus, c_minus);
    }

    int bad = 0;
    for (int i = 0; i < kCirculantSize; i++) {
        if (!g1_eq(want[i], got[i])) bad++;
    }
    if (bad) {
        printf("FAIL: split circulant differs from flat form at %d/%d outputs\n", bad,
               kCirculantSize);
        g_failures++;
    } else {
        printf("ok  : split circulant (X^128-1 = (X^64-1)(X^64+1)) == flat form (128/128)\n");
    }
}

int main(int argc, char **argv) {
    const char *vec_dir = argc > 1 ? argv[1] : "tests/vectors";

    SetupTables tables;
    double t0 = now_ms();
    // Reuse a cache so repeated runs during development are quick.
    const std::string cache = "/tmp/vkp_prover_tables_ref.cache";
    const uint64_t digest = compute_setup_digest(kEmbeddedSetupG1Monomial, kEmbeddedSetupSize);
    bool loaded = load_table_cache(cache, digest, tables) == VKP_OK;
    if (!loaded) {
        if (build_setup_tables(kEmbeddedSetupG1Monomial, kEmbeddedSetupSize, true, tables) !=
            VKP_OK) {
            printf("FAIL: build_setup_tables\n");
            return 1;
        }
        save_table_cache(cache, tables);
    }
    printf("setup tables %s in %.0f ms (%.1f MiB position table)\n", loaded ? "loaded" : "built",
           now_ms() - t0, tables.position_table.size() * 4.0 / (1024 * 1024));

    test_circulant_equivalence(tables);
    test_split_circulant_equivalence(tables);

    auto vectors = vkp_test::load_all(vec_dir);
    if (vectors.empty()) {
        printf("FAIL: no test vectors found in %s\n", vec_dir);
        return 1;
    }

    int passed = 0;
    for (const auto &v : vectors) {
        std::vector<uint8_t> proofs(128 * 48);
        double a = now_ms();
        // The library takes a fixed-size blob buffer, so a wrong length is the
        // caller's error to catch -- two of the spec vectors exercise exactly
        // that, and the harness stands in for the caller here.
        vkp_result rc = v.blob.size() == (size_t)kFieldElementsPerBlob * kBytesPerFieldElement
                              ? reference_compute(tables, v.blob.data(), proofs.data())
                              : VKP_ERR_BADARGS;
        double ms = now_ms() - a;

        if (!v.valid) {
            if (rc == VKP_OK) {
                printf("FAIL %s: expected rejection, got success\n", v.name.c_str());
                g_failures++;
            } else {
                passed++;
            }
            continue;
        }
        if (rc != VKP_OK) {
            printf("FAIL %s: unexpected error %d\n", v.name.c_str(), (int)rc);
            g_failures++;
            continue;
        }
        bool proofs_ok = memcmp(proofs.data(), v.proofs.data(), proofs.size()) == 0;
        if (!proofs_ok) {
            printf("FAIL %s: proofs MISMATCH\n", v.name.c_str());
            for (int i = 0; i < 128; i++) {
                if (memcmp(&proofs[i * 48], &v.proofs[i * 48], 48) != 0) {
                    printf("   first bad proof %d:\n     got %s\n     want %s\n", i,
                           hex(&proofs[i * 48], 48).c_str(), hex(&v.proofs[i * 48], 48).c_str());
                    break;
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
