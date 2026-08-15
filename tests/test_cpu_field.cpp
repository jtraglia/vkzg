// Unit tests for the host BLS12-381 arithmetic.
//
// The reference values are the ones every BLS12-381 implementation agrees on
// (generator, compressed encodings, group order), so passing these means the
// Montgomery arithmetic, the point formulas and the serialisation all line up
// with the rest of the ecosystem.
#include "../src/cpu/bls12_381.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace kzgpu;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                                                           \
    do {                                                                                           \
        g_checks++;                                                                                \
        if (!(cond)) {                                                                             \
            g_failures++;                                                                          \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                       \
            std::printf(__VA_ARGS__);                                                              \
            std::printf("\n");                                                                     \
        }                                                                                          \
    } while (0)

static std::string hex(const uint8_t *b, size_t n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

static void from_hex(uint8_t *out, const char *h, size_t n) {
    for (size_t i = 0; i < n; i++) {
        auto nib = [](char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; };
        out[i] = (uint8_t)((nib(h[2 * i]) << 4) | nib(h[2 * i + 1]));
    }
}

// xorshift, so the tests are deterministic across machines.
static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static uint64_t rnd() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static Fr rand_fr() {
    uint8_t b[32];
    for (int i = 0; i < 32; i++) b[i] = (uint8_t)rnd();
    b[0] &= 0x0f; // keep it below r
    Fr r;
    bool ok = fr_from_bytes(r, b);
    if (!ok) return kFrZero;
    return r;
}

static void test_fp_basics() {
    Fp a, b, c, d;
    fp_from_u64(a, 5);
    fp_from_u64(b, 7);
    fp_add(c, a, b);
    fp_from_u64(d, 12);
    CHECK(fp_eq(c, d), "5 + 7 != 12");
    fp_mul(c, a, b);
    fp_from_u64(d, 35);
    CHECK(fp_eq(c, d), "5 * 7 != 35");
    fp_sub(c, a, b);
    fp_from_u64(d, 2);
    fp_neg(d, d);
    CHECK(fp_eq(c, d), "5 - 7 != -2");

    // a * a^-1 == 1 for a bunch of values
    for (int i = 1; i < 50; i++) {
        fp_from_u64(a, (uint64_t)i * 1000003);
        fp_inv(b, a);
        fp_mul(c, a, b);
        CHECK(fp_eq(c, kFpOne), "inversion failed for %d", i);
    }

    // squaring agrees with multiplication
    for (int i = 0; i < 200; i++) {
        uint8_t buf[48] = {0};
        for (int j = 8; j < 48; j++) buf[j] = (uint8_t)rnd();
        CHECK(fp_from_bytes(a, buf), "fp_from_bytes rejected a small value");
        fp_sqr(b, a);
        fp_mul(c, a, a);
        CHECK(fp_eq(b, c), "fp_sqr != fp_mul");
    }

    // round-trip serialisation
    for (int i = 0; i < 200; i++) {
        uint8_t buf[48] = {0}, out[48];
        for (int j = 8; j < 48; j++) buf[j] = (uint8_t)rnd();
        fp_from_bytes(a, buf);
        fp_to_bytes(out, a);
        CHECK(memcmp(buf, out, 48) == 0, "fp serialisation round trip");
    }

    // p itself must be rejected
    uint8_t pbytes[48];
    from_hex(pbytes, "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f624"
                     "1eabfffeb153ffffb9feffffffffaaab", 48);
    CHECK(!fp_from_bytes(a, pbytes), "p should not be a valid field element");
}

static void test_fr_basics() {
    Fr a, b, c;
    fr_from_u64(a, 12345);
    fr_inv(b, a);
    fr_mul(c, a, b);
    CHECK(fr_eq(c, kFrOne), "fr inversion");

    // The 2^13-th root of unity really has order 8192.
    Fr root;
    fr_root_of_unity(root, 13);
    Fr acc = kFrOne;
    int order = 0;
    for (int i = 1; i <= 8192; i++) {
        fr_mul(acc, acc, root);
        if (fr_eq(acc, kFrOne)) {
            order = i;
            break;
        }
    }
    CHECK(order == 8192, "root of unity has order %d, expected 8192", order);

    for (int log = 1; log <= 13; log++) {
        fr_root_of_unity(root, (size_t)log);
        Fr p = root;
        for (int i = 1; i < (1 << log); i++) fr_mul(p, p, root);
        CHECK(fr_eq(p, kFrOne), "root^(2^%d) != 1", log);
    }
}

static void test_g1() {
    // The BLS12-381 G1 generator in compressed form.
    static const char *kGenHex =
        "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    uint8_t gen[48];
    from_hex(gen, kGenHex, 48);
    G1Affine g;
    CHECK(g1_decompress(g, gen), "failed to decompress the generator");
    CHECK(g1_affine_is_on_curve(g), "generator not on curve");
    CHECK(g1_affine_in_subgroup(g), "generator not in subgroup");

    uint8_t reenc[48];
    g1_compress(reenc, g);
    CHECK(memcmp(gen, reenc, 48) == 0, "generator re-encoding mismatch: %s", hex(reenc, 48).c_str());

    // Point at infinity round trip.
    uint8_t inf[48] = {0};
    inf[0] = 0xc0;
    G1Affine z;
    CHECK(g1_decompress(z, inf), "failed to decompress infinity");
    g1_compress(reenc, z);
    CHECK(memcmp(inf, reenc, 48) == 0, "infinity re-encoding mismatch");

    G1 G, acc, t1, t2;
    g1_from_affine(G, g);

    // [r]G == O, i.e. the generator really has order r.
    Fr negone;
    fr_neg(negone, kFrOne);
    g1_mul(t1, G, negone); // [r-1]G
    g1_add(t2, t1, G);     // + G == [r]G
    CHECK(g1_is_identity(t2), "[r]G is not the identity");

    // Doubling agrees with addition.
    g1_dbl(t1, G);
    g1_add(t2, G, G);
    CHECK(g1_eq(t1, t2), "dbl != add(P,P)");

    // Mixed addition agrees with general addition.
    acc = G;
    for (int i = 0; i < 20; i++) {
        g1_dbl(acc, acc);
        G1Affine aff;
        g1_to_affine(aff, acc);
        g1_add_mixed(t1, G, aff);
        g1_add(t2, G, acc);
        CHECK(g1_eq(t1, t2), "mixed add != general add at step %d", i);
    }

    // Adding the negation gives the identity, including via the mixed path.
    G1 negG;
    g1_neg(negG, G);
    g1_add(t1, G, negG);
    CHECK(g1_is_identity(t1), "P + (-P) != O");
    G1Affine negAff;
    g1_to_affine(negAff, negG);
    g1_add_mixed(t1, G, negAff);
    CHECK(g1_is_identity(t1), "P + (-P) != O via mixed add");

    // Identity behaves.
    g1_add(t1, kG1Identity, G);
    CHECK(g1_eq(t1, G), "O + P != P");
    g1_add_mixed(t1, kG1Identity, g);
    CHECK(g1_eq(t1, G), "O + P != P (mixed)");

    // Scalar multiplication is linear: [a]G + [b]G == [a+b]G
    for (int i = 0; i < 20; i++) {
        Fr a = rand_fr(), b = rand_fr(), s;
        fr_add(s, a, b);
        G1 pa, pb, ps, sum;
        g1_mul(pa, G, a);
        g1_mul(pb, G, b);
        g1_mul(ps, G, s);
        g1_add(sum, pa, pb);
        CHECK(g1_eq(sum, ps), "[a]G + [b]G != [a+b]G");
    }

    // Batch affine conversion matches the one-at-a-time path.
    const size_t n = 64;
    G1 *pts = new G1[n];
    G1Affine *aff = new G1Affine[n];
    acc = G;
    for (size_t i = 0; i < n; i++) {
        pts[i] = acc;
        g1_dbl(acc, acc);
    }
    pts[7] = kG1Identity; // include an infinity to exercise that path
    g1_batch_to_affine(aff, pts, n);
    for (size_t i = 0; i < n; i++) {
        G1Affine one;
        g1_to_affine(one, pts[i]);
        CHECK(fp_eq(one.x, aff[i].x) && fp_eq(one.y, aff[i].y), "batch affine mismatch at %zu", i);
    }
    delete[] pts;
    delete[] aff;
}


int main() {
    test_fp_basics();
    test_fr_basics();
    test_g1();
    std::printf("%s: %d checks, %d failures\n", g_failures ? "FAILED" : "ok", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
