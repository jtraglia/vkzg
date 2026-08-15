#include "bls12_381.h"

#include "bls12_381_constants.h"

#include <cassert>

namespace vkp {
namespace {

typedef unsigned __int128 u128;

constexpr uint64_t kP[6] = VKP_FP_P;
constexpr uint64_t kFpN0 = VKP_FP_N0;
constexpr uint64_t kFpR2[6] = VKP_FP_R2;
constexpr uint64_t kR[4] = VKP_FR_R_MOD;
constexpr uint64_t kFrN0 = VKP_FR_N0;
constexpr uint64_t kFrR2[4] = VKP_FR_R2;

// ---------------------------------------------------------------- helpers

// r = a + b + carry_in, returns carry out.
inline uint64_t addc(uint64_t a, uint64_t b, uint64_t carry, uint64_t &r) {
    u128 t = (u128)a + b + carry;
    r = (uint64_t)t;
    return (uint64_t)(t >> 64);
}

// r = a - b - borrow_in, returns borrow out.
inline uint64_t subb(uint64_t a, uint64_t b, uint64_t borrow, uint64_t &r) {
    u128 t = (u128)a - b - borrow;
    r = (uint64_t)t;
    return (uint64_t)((t >> 64) & 1);
}

inline uint64_t mulhi(uint64_t a, uint64_t b) { return (uint64_t)(((u128)a * b) >> 64); }


template <int N>
inline int cmp_n(const uint64_t *a, const uint64_t *b) {
    for (int i = N - 1; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    }
    return 0;
}

// Conditional subtraction of the modulus.
template <int N>
inline void csub_mod(uint64_t *a, const uint64_t *mod) {
    uint64_t t[N], borrow = 0;
    for (int i = 0; i < N; i++) borrow = subb(a[i], mod[i], borrow, t[i]);
    if (!borrow) {
        for (int i = 0; i < N; i++) a[i] = t[i];
    }
}

// Montgomery multiplication with separated carry chains.
//
// The obvious form -- `t[j] = t[j] + a[j]*b[i] + carry` with a 128-bit
// accumulator -- reads well but compiles badly on ARM64: clang materialises
// each carry into a register with `cset` instead of chaining `adds/adcs`, and
// spills. Measured 51.5 ns.
//
// Issuing all N products first and then absorbing them with two straight carry
// chains (one for the low halves, one for the high halves) lets the
// multiplies dual-issue and turns the rest into two adds/adcs runs: 32.6 ns,
// a 1.58x improvement, and it matters because the host now carries a real
// share of the MSM work.
#define KZ_ADDC(x, y, cin, cout) __builtin_addcll((x), (y), (cin), (cout))

// 6-limb (Fp). The modulus is below 2^381, so the running value never needs
// more than the seven words held here.
void mont_mul_fp(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    const uint64_t p0 = kP[0], p1 = kP[1], p2 = kP[2], p3 = kP[3], p4 = kP[4], p5 = kP[5];
    unsigned long long c;

    for (int i = 0; i < 6; i++) {
        const uint64_t bi = b[i];
        const uint64_t l0 = a0 * bi, l1 = a1 * bi, l2 = a2 * bi;
        const uint64_t l3 = a3 * bi, l4 = a4 * bi, l5 = a5 * bi;
        const uint64_t h0 = mulhi(a0, bi), h1 = mulhi(a1, bi), h2 = mulhi(a2, bi);
        const uint64_t h3 = mulhi(a3, bi), h4 = mulhi(a4, bi), h5 = mulhi(a5, bi);
        t0 = KZ_ADDC(t0, l0, 0, &c); t1 = KZ_ADDC(t1, l1, c, &c);
        t2 = KZ_ADDC(t2, l2, c, &c); t3 = KZ_ADDC(t3, l3, c, &c);
        t4 = KZ_ADDC(t4, l4, c, &c); t5 = KZ_ADDC(t5, l5, c, &c);
        t6 = KZ_ADDC(t6, 0, c, &c);  t7 = (uint64_t)c;
        t1 = KZ_ADDC(t1, h0, 0, &c); t2 = KZ_ADDC(t2, h1, c, &c);
        t3 = KZ_ADDC(t3, h2, c, &c); t4 = KZ_ADDC(t4, h3, c, &c);
        t5 = KZ_ADDC(t5, h4, c, &c); t6 = KZ_ADDC(t6, h5, c, &c);
        t7 = KZ_ADDC(t7, 0, c, &c);

        const uint64_t m = t0 * kFpN0;
        const uint64_t q0 = m * p0, q1 = m * p1, q2 = m * p2;
        const uint64_t q3 = m * p3, q4 = m * p4, q5 = m * p5;
        const uint64_t g0 = mulhi(m, p0), g1 = mulhi(m, p1), g2 = mulhi(m, p2);
        const uint64_t g3 = mulhi(m, p3), g4 = mulhi(m, p4), g5 = mulhi(m, p5);
        t0 = KZ_ADDC(t0, q0, 0, &c); t1 = KZ_ADDC(t1, q1, c, &c);
        t2 = KZ_ADDC(t2, q2, c, &c); t3 = KZ_ADDC(t3, q3, c, &c);
        t4 = KZ_ADDC(t4, q4, c, &c); t5 = KZ_ADDC(t5, q5, c, &c);
        t6 = KZ_ADDC(t6, 0, c, &c);  t7 = KZ_ADDC(t7, 0, c, &c);
        t1 = KZ_ADDC(t1, g0, 0, &c); t2 = KZ_ADDC(t2, g1, c, &c);
        t3 = KZ_ADDC(t3, g2, c, &c); t4 = KZ_ADDC(t4, g3, c, &c);
        t5 = KZ_ADDC(t5, g4, c, &c); t6 = KZ_ADDC(t6, g5, c, &c);
        t7 = KZ_ADDC(t7, 0, c, &c);
        // t0 is zero by construction; shift the window down one limb.
        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = t6; t6 = t7; t7 = 0;
    }

    uint64_t s0, s1, s2, s3, s4, s5, brw;
    s0 = __builtin_subcll(t0, p0, 0, &c);   brw = (uint64_t)c;
    s1 = __builtin_subcll(t1, p1, brw, &c); brw = (uint64_t)c;
    s2 = __builtin_subcll(t2, p2, brw, &c); brw = (uint64_t)c;
    s3 = __builtin_subcll(t3, p3, brw, &c); brw = (uint64_t)c;
    s4 = __builtin_subcll(t4, p4, brw, &c); brw = (uint64_t)c;
    s5 = __builtin_subcll(t5, p5, brw, &c); brw = (uint64_t)c;
    const bool keep = t6 < brw;
    r[0] = keep ? t0 : s0; r[1] = keep ? t1 : s1; r[2] = keep ? t2 : s2;
    r[3] = keep ? t3 : s3; r[4] = keep ? t4 : s4; r[5] = keep ? t5 : s5;
}

// 4-limb (Fr), same shape.
void mont_mul_fr(uint64_t *r, const uint64_t *a, const uint64_t *b) {
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const uint64_t p0 = kR[0], p1 = kR[1], p2 = kR[2], p3 = kR[3];
    unsigned long long c;

    for (int i = 0; i < 4; i++) {
        const uint64_t bi = b[i];
        const uint64_t l0 = a0 * bi, l1 = a1 * bi, l2 = a2 * bi, l3 = a3 * bi;
        const uint64_t h0 = mulhi(a0, bi), h1 = mulhi(a1, bi);
        const uint64_t h2 = mulhi(a2, bi), h3 = mulhi(a3, bi);
        t0 = KZ_ADDC(t0, l0, 0, &c); t1 = KZ_ADDC(t1, l1, c, &c);
        t2 = KZ_ADDC(t2, l2, c, &c); t3 = KZ_ADDC(t3, l3, c, &c);
        t4 = KZ_ADDC(t4, 0, c, &c);  t5 = (uint64_t)c;
        t1 = KZ_ADDC(t1, h0, 0, &c); t2 = KZ_ADDC(t2, h1, c, &c);
        t3 = KZ_ADDC(t3, h2, c, &c); t4 = KZ_ADDC(t4, h3, c, &c);
        t5 = KZ_ADDC(t5, 0, c, &c);

        const uint64_t m = t0 * kFrN0;
        const uint64_t q0 = m * p0, q1 = m * p1, q2 = m * p2, q3 = m * p3;
        const uint64_t g0 = mulhi(m, p0), g1 = mulhi(m, p1);
        const uint64_t g2 = mulhi(m, p2), g3 = mulhi(m, p3);
        t0 = KZ_ADDC(t0, q0, 0, &c); t1 = KZ_ADDC(t1, q1, c, &c);
        t2 = KZ_ADDC(t2, q2, c, &c); t3 = KZ_ADDC(t3, q3, c, &c);
        t4 = KZ_ADDC(t4, 0, c, &c);  t5 = KZ_ADDC(t5, 0, c, &c);
        t1 = KZ_ADDC(t1, g0, 0, &c); t2 = KZ_ADDC(t2, g1, c, &c);
        t3 = KZ_ADDC(t3, g2, c, &c); t4 = KZ_ADDC(t4, g3, c, &c);
        t5 = KZ_ADDC(t5, 0, c, &c);
        t0 = t1; t1 = t2; t2 = t3; t3 = t4; t4 = t5; t5 = 0;
    }

    uint64_t s0, s1, s2, s3, brw;
    s0 = __builtin_subcll(t0, p0, 0, &c);   brw = (uint64_t)c;
    s1 = __builtin_subcll(t1, p1, brw, &c); brw = (uint64_t)c;
    s2 = __builtin_subcll(t2, p2, brw, &c); brw = (uint64_t)c;
    s3 = __builtin_subcll(t3, p3, brw, &c); brw = (uint64_t)c;
    const bool keep = t4 < brw;
    r[0] = keep ? t0 : s0; r[1] = keep ? t1 : s1;
    r[2] = keep ? t2 : s2; r[3] = keep ? t3 : s3;
}

// Coarsely Integrated Operand Scanning Montgomery multiplication.

template <int N>
void add_mod_n(uint64_t *r, const uint64_t *a, const uint64_t *b, const uint64_t *mod) {
    uint64_t carry = 0;
    for (int i = 0; i < N; i++) carry = addc(a[i], b[i], carry, r[i]);
    // p < 2^(64N-1) for both our moduli, so the sum never overflows N limbs.
    csub_mod<N>(r, mod);
}

template <int N>
void sub_mod_n(uint64_t *r, const uint64_t *a, const uint64_t *b, const uint64_t *mod) {
    uint64_t borrow = 0;
    for (int i = 0; i < N; i++) borrow = subb(a[i], b[i], borrow, r[i]);
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < N; i++) carry = addc(r[i], mod[i], carry, r[i]);
    }
}

template <int N>
bool is_zero_n(const uint64_t *a) {
    uint64_t acc = 0;
    for (int i = 0; i < N; i++) acc |= a[i];
    return acc == 0;
}

} // namespace

// ---------------------------------------------------------------------- Fp

const Fp kFpZero = {{0, 0, 0, 0, 0, 0}};
const Fp kFpOne = {VKP_FP_R};
const Fp kFpBeta = {VKP_FP_BETA};

void fp_add(Fp &r, const Fp &a, const Fp &b) { add_mod_n<6>(r.v, a.v, b.v, kP); }
void fp_sub(Fp &r, const Fp &a, const Fp &b) { sub_mod_n<6>(r.v, a.v, b.v, kP); }
void fp_neg(Fp &r, const Fp &a) { sub_mod_n<6>(r.v, kFpZero.v, a.v, kP); }
void fp_mul(Fp &r, const Fp &a, const Fp &b) { mont_mul_fp(r.v, a.v, b.v); }
void fp_sqr(Fp &r, const Fp &a) { mont_mul_fp(r.v, a.v, a.v); }
void fp_dbl(Fp &r, const Fp &a) { add_mod_n<6>(r.v, a.v, a.v, kP); }
bool fp_is_zero(const Fp &a) { return is_zero_n<6>(a.v); }
bool fp_eq(const Fp &a, const Fp &b) { return cmp_n<6>(a.v, b.v) == 0; }

namespace {
// Exponentiation by a fixed big-endian-limb exponent, used for inversion and
// square roots. Not constant time -- inputs here are public data.
void fp_pow_limbs(Fp &r, const Fp &a, const uint64_t *e, int nlimbs) {
    Fp acc = kFpOne;
    bool started = false;
    for (int i = nlimbs - 1; i >= 0; i--) {
        for (int b = 63; b >= 0; b--) {
            if (started) fp_sqr(acc, acc);
            if ((e[i] >> b) & 1) {
                if (started) {
                    fp_mul(acc, acc, a);
                } else {
                    acc = a;
                    started = true;
                }
            }
        }
    }
    r = acc;
}
} // namespace

void fp_inv(Fp &r, const Fp &a) {
    static const uint64_t e[6] = VKP_FP_P_MINUS_2;
    fp_pow_limbs(r, a, e, 6);
}

void fp_from_u64(Fp &r, uint64_t x) {
    Fp t = {{x, 0, 0, 0, 0, 0}};
    mont_mul_fp(r.v, t.v, kFpR2); // into Montgomery form
}

bool fp_from_bytes(Fp &r, const uint8_t in[48]) {
    uint64_t t[6] = {0};
    for (int i = 0; i < 6; i++) {
        uint64_t w = 0;
        for (int j = 0; j < 8; j++) w = (w << 8) | in[i * 8 + j];
        t[5 - i] = w;
    }
    if (cmp_n<6>(t, kP) >= 0) return false;
    mont_mul_fp(r.v, t, kFpR2);
    return true;
}

namespace {
void fp_to_canonical(uint64_t out[6], const Fp &a) {
    static const uint64_t one[6] = {1, 0, 0, 0, 0, 0};
    mont_mul_fp(out, a.v, one); // multiply by 1 => out of Montgomery
}
} // namespace

void fp_to_bytes(uint8_t out[48], const Fp &a) {
    uint64_t t[6];
    fp_to_canonical(t, a);
    for (int i = 0; i < 6; i++) {
        uint64_t w = t[5 - i];
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (uint8_t)(w >> (56 - 8 * j));
    }
}

bool fp_is_lexicographically_largest(const Fp &a) {
    uint64_t t[6];
    fp_to_canonical(t, a);
    // Compare against (p-1)/2 == p >> 1.
    uint64_t half[6];
    for (int i = 0; i < 6; i++) {
        half[i] = (kP[i] >> 1) | (i + 1 < 6 ? kP[i + 1] << 63 : 0);
    }
    return cmp_n<6>(t, half) > 0;
}

// ---------------------------------------------------------------------- Fr

const Fr kFrZero = {{0, 0, 0, 0}};
const Fr kFrOne = {VKP_FR_ONE};

void fr_add(Fr &r, const Fr &a, const Fr &b) { add_mod_n<4>(r.v, a.v, b.v, kR); }
void fr_sub(Fr &r, const Fr &a, const Fr &b) { sub_mod_n<4>(r.v, a.v, b.v, kR); }
void fr_neg(Fr &r, const Fr &a) { sub_mod_n<4>(r.v, kFrZero.v, a.v, kR); }
void fr_mul(Fr &r, const Fr &a, const Fr &b) { mont_mul_fr(r.v, a.v, b.v); }
void fr_sqr(Fr &r, const Fr &a) { mont_mul_fr(r.v, a.v, a.v); }
bool fr_is_zero(const Fr &a) { return is_zero_n<4>(a.v); }
bool fr_eq(const Fr &a, const Fr &b) { return cmp_n<4>(a.v, b.v) == 0; }

void fr_from_u64(Fr &r, uint64_t x) {
    Fr t = {{x, 0, 0, 0}};
    mont_mul_fr(r.v, t.v, kFrR2);
}

void fr_from_canonical(Fr &r, const uint64_t in[4]) {
    mont_mul_fr(r.v, in, kFrR2);
}

void fr_to_canonical(uint64_t out[4], const Fr &a) {
    static const uint64_t one[4] = {1, 0, 0, 0};
    mont_mul_fr(out, a.v, one);
}

bool fr_from_bytes(Fr &r, const uint8_t in[32]) {
    uint64_t t[4];
    for (int i = 0; i < 4; i++) {
        uint64_t w = 0;
        for (int j = 0; j < 8; j++) w = (w << 8) | in[i * 8 + j];
        t[3 - i] = w;
    }
    if (cmp_n<4>(t, kR) >= 0) return false;
    fr_from_canonical(r, t);
    return true;
}

void fr_to_bytes(uint8_t out[32], const Fr &a) {
    uint64_t t[4];
    fr_to_canonical(t, a);
    for (int i = 0; i < 4; i++) {
        uint64_t w = t[3 - i];
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (uint8_t)(w >> (56 - 8 * j));
    }
}


void fr_inv(Fr &r, const Fr &a) {
    static const uint64_t e[4] = VKP_FR_R_MINUS_2;
    Fr acc = kFrOne;
    bool started = false;
    for (int i = 3; i >= 0; i--) {
        for (int b = 63; b >= 0; b--) {
            if (started) fr_sqr(acc, acc);
            if ((e[i] >> b) & 1) {
                if (started) {
                    fr_mul(acc, acc, a);
                } else {
                    acc = a;
                    started = true;
                }
            }
        }
    }
    r = acc;
}

void fr_root_of_unity(Fr &r, size_t log_order) {
    assert(log_order <= 13);
    Fr root = {VKP_FR_ROOT_8192}; // primitive 2^13-th root
    for (size_t i = log_order; i < 13; i++) fr_sqr(root, root);
    r = root;
}

// ---------------------------------------------------------------------- G1

const G1 kG1Identity = {kFpZero, kFpOne, kFpZero};

void g1_set_identity(G1 &r) { r = kG1Identity; }
bool g1_is_identity(const G1 &p) { return fp_is_zero(p.z); }

void g1_neg(G1 &r, const G1 &p) {
    r = p;
    fp_neg(r.y, p.y);
}

// dbl-2009-l (a == 0). Results go to locals first so that `r` may alias `p`,
// which the double-and-add loops rely on.
void g1_dbl(G1 &r, const G1 &p) {
    if (fp_is_zero(p.z)) {
        r = p;
        return;
    }
    Fp A, B, C, D, E, F, t0, t1, x3, y3, z3;
    fp_sqr(A, p.x);          // A = X^2
    fp_sqr(B, p.y);          // B = Y^2
    fp_sqr(C, B);            // C = B^2
    fp_add(t0, p.x, B);      // X + B
    fp_sqr(t0, t0);          // (X+B)^2
    fp_sub(t0, t0, A);       // - A
    fp_sub(t0, t0, C);       // - C
    fp_dbl(D, t0);           // D = 2*((X+B)^2 - A - C)
    fp_dbl(E, A);            //
    fp_add(E, E, A);         // E = 3A
    fp_sqr(F, E);            // F = E^2
    fp_dbl(t0, D);           // 2D
    fp_sub(x3, F, t0);       // X3 = F - 2D
    fp_sub(t0, D, x3);       // D - X3
    fp_mul(t0, E, t0);       // E*(D - X3)
    fp_dbl(t1, C);           // 2C
    fp_dbl(t1, t1);          // 4C
    fp_dbl(t1, t1);          // 8C
    fp_sub(y3, t0, t1);      // Y3 = E*(D-X3) - 8C
    fp_mul(t0, p.y, p.z);    //
    fp_dbl(z3, t0);          // Z3 = 2*Y*Z
    r.x = x3;
    r.y = y3;
    r.z = z3;
}

// add-2007-bl (a == 0)
void g1_add(G1 &r, const G1 &a, const G1 &b) {
    if (fp_is_zero(a.z)) {
        r = b;
        return;
    }
    if (fp_is_zero(b.z)) {
        r = a;
        return;
    }
    Fp Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, K, t0, t1;
    fp_sqr(Z1Z1, a.z);
    fp_sqr(Z2Z2, b.z);
    fp_mul(U1, a.x, Z2Z2);
    fp_mul(U2, b.x, Z1Z1);
    fp_mul(t0, b.z, Z2Z2);
    fp_mul(S1, a.y, t0);
    fp_mul(t0, a.z, Z1Z1);
    fp_mul(S2, b.y, t0);
    if (fp_eq(U1, U2)) {
        if (fp_eq(S1, S2)) {
            g1_dbl(r, a);
        } else {
            g1_set_identity(r);
        }
        return;
    }
    fp_sub(H, U2, U1);
    fp_dbl(t0, H);
    fp_sqr(I, t0);         // I = (2H)^2
    fp_mul(J, H, I);       // J = H*I
    fp_sub(t0, S2, S1);
    fp_dbl(K, t0);         // K = 2*(S2 - S1)
    fp_mul(t0, U1, I);     // V = U1*I
    Fp x3, y3, z3;
    fp_sqr(x3, K);
    fp_sub(x3, x3, J);
    fp_dbl(t1, t0);
    fp_sub(x3, x3, t1);    // X3 = K^2 - J - 2V
    fp_sub(t1, t0, x3);    // V - X3
    fp_mul(t1, K, t1);
    fp_mul(t0, S1, J);
    fp_dbl(t0, t0);
    fp_sub(y3, t1, t0);    // Y3 = K*(V-X3) - 2*S1*J
    fp_add(t0, a.z, b.z);
    fp_sqr(t0, t0);
    fp_sub(t0, t0, Z1Z1);
    fp_sub(t0, t0, Z2Z2);
    fp_mul(z3, t0, H);     // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    r.x = x3;
    r.y = y3;
    r.z = z3;
}

// madd-2007-bl: a Jacobian, b affine (Z2 == 1).
void g1_add_mixed(G1 &r, const G1 &a, const G1Affine &b) {
    if (fp_is_zero(b.x) && fp_is_zero(b.y)) {
        r = a;
        return;
    }
    if (fp_is_zero(a.z)) {
        r.x = b.x;
        r.y = b.y;
        r.z = kFpOne;
        return;
    }
    Fp Z1Z1, U2, S2, H, HH, I, J, K, V, t0, t1;
    fp_sqr(Z1Z1, a.z);
    fp_mul(U2, b.x, Z1Z1);
    fp_mul(t0, a.z, Z1Z1);
    fp_mul(S2, b.y, t0);
    if (fp_eq(a.x, U2)) {
        if (fp_eq(a.y, S2)) {
            g1_dbl(r, a);
        } else {
            g1_set_identity(r);
        }
        return;
    }
    fp_sub(H, U2, a.x);
    fp_sqr(HH, H);
    fp_dbl(I, HH);
    fp_dbl(I, I);          // I = 4*HH
    fp_mul(J, H, I);
    fp_sub(t0, S2, a.y);
    fp_dbl(K, t0);         // K = 2*(S2 - Y1)
    fp_mul(V, a.x, I);
    Fp x3, y3, z3;
    fp_sqr(x3, K);
    fp_sub(x3, x3, J);
    fp_dbl(t0, V);
    fp_sub(x3, x3, t0);    // X3 = K^2 - J - 2V
    fp_sub(t0, V, x3);
    fp_mul(t0, K, t0);
    fp_mul(t1, a.y, J);
    fp_dbl(t1, t1);
    fp_sub(y3, t0, t1);    // Y3 = K*(V-X3) - 2*Y1*J
    fp_add(t0, a.z, H);
    fp_sqr(t0, t0);
    fp_sub(t0, t0, Z1Z1);
    fp_sub(z3, t0, HH);    // Z3 = (Z1+H)^2 - Z1Z1 - HH
    r.x = x3;
    r.y = y3;
    r.z = z3;
}

void g1_sub(G1 &r, const G1 &a, const G1 &b) {
    G1 nb;
    g1_neg(nb, b);
    g1_add(r, a, nb);
}

void g1_from_affine(G1 &r, const G1Affine &a) {
    if (fp_is_zero(a.x) && fp_is_zero(a.y)) {
        g1_set_identity(r);
        return;
    }
    r.x = a.x;
    r.y = a.y;
    r.z = kFpOne;
}

void g1_mul(G1 &r, const G1 &p, const Fr &k) {
    uint64_t e[4];
    fr_to_canonical(e, k);
    G1 acc = kG1Identity;
    bool started = false;
    for (int i = 3; i >= 0; i--) {
        for (int b = 63; b >= 0; b--) {
            if (started) g1_dbl(acc, acc);
            if ((e[i] >> b) & 1) {
                if (started) {
                    g1_add(acc, acc, p);
                } else {
                    acc = p;
                    started = true;
                }
            }
        }
    }
    r = acc;
}

bool g1_eq(const G1 &a, const G1 &b) {
    bool ai = g1_is_identity(a), bi = g1_is_identity(b);
    if (ai || bi) return ai && bi;
    Fp z1z1, z2z2, u1, u2, s1, s2, t;
    fp_sqr(z1z1, a.z);
    fp_sqr(z2z2, b.z);
    fp_mul(u1, a.x, z2z2);
    fp_mul(u2, b.x, z1z1);
    fp_mul(t, b.z, z2z2);
    fp_mul(s1, a.y, t);
    fp_mul(t, a.z, z1z1);
    fp_mul(s2, b.y, t);
    return fp_eq(u1, u2) && fp_eq(s1, s2);
}

void batch_inverse(Fp *out, const Fp *in, size_t n) {
    if (n == 0) return;
    // Prefix products, skipping zeros (which invert to zero).
    Fp acc = kFpOne;
    for (size_t i = 0; i < n; i++) {
        out[i] = acc;
        if (!fp_is_zero(in[i])) fp_mul(acc, acc, in[i]);
    }
    Fp inv;
    fp_inv(inv, acc);
    for (size_t i = n; i-- > 0;) {
        if (fp_is_zero(in[i])) {
            out[i] = kFpZero;
            continue;
        }
        Fp t;
        fp_mul(t, out[i], inv);       // = 1 / in[i]
        fp_mul(inv, inv, in[i]);      // advance
        out[i] = t;
    }
}

void g1_to_affine(G1Affine &out, const G1 &in) {
    if (g1_is_identity(in)) {
        out.x = kFpZero;
        out.y = kFpZero;
        return;
    }
    Fp zinv, zinv2, zinv3;
    fp_inv(zinv, in.z);
    fp_sqr(zinv2, zinv);
    fp_mul(zinv3, zinv2, zinv);
    fp_mul(out.x, in.x, zinv2);
    fp_mul(out.y, in.y, zinv3);
}

void g1_batch_to_affine(G1Affine *out, const G1 *in, size_t n) {
    if (n == 0) return;
    Fp *zs = new Fp[n];
    Fp *inv = new Fp[n];
    for (size_t i = 0; i < n; i++) zs[i] = in[i].z;
    batch_inverse(inv, zs, n);
    for (size_t i = 0; i < n; i++) {
        if (fp_is_zero(in[i].z)) {
            out[i].x = kFpZero;
            out[i].y = kFpZero;
            continue;
        }
        Fp z2, z3;
        fp_sqr(z2, inv[i]);
        fp_mul(z3, z2, inv[i]);
        fp_mul(out[i].x, in[i].x, z2);
        fp_mul(out[i].y, in[i].y, z3);
    }
    delete[] zs;
    delete[] inv;
}

bool g1_affine_is_on_curve(const G1Affine &p) {
    if (fp_is_zero(p.x) && fp_is_zero(p.y)) return true; // infinity
    Fp lhs, rhs, four;
    fp_sqr(lhs, p.y);
    fp_sqr(rhs, p.x);
    fp_mul(rhs, rhs, p.x);
    fp_from_u64(four, 4);
    fp_add(rhs, rhs, four);
    return fp_eq(lhs, rhs);
}

void g1_affine_endo(G1Affine &out, const G1Affine &in) {
    fp_mul(out.x, in.x, kFpBeta);
    out.y = in.y;
}

bool g1_affine_in_subgroup(const G1Affine &p) {
    if (fp_is_zero(p.x) && fp_is_zero(p.y)) return true;
    // P is in the prime-order subgroup iff phi(P) == [lambda]P.
    G1Affine phi;
    g1_affine_endo(phi, p);
    G1 lhs, rhs, jp;
    g1_from_affine(jp, p);
    g1_from_affine(lhs, phi);
    // lambda is short over Z (~128 bits); multiply by it directly.
    static const uint64_t lam[4] = VKP_GLV_LAMBDA_INT;
    G1 acc = kG1Identity;
    bool started = false;
    for (int i = 3; i >= 0; i--) {
        for (int b = 63; b >= 0; b--) {
            if (started) g1_dbl(acc, acc);
            if ((lam[i] >> b) & 1) {
                if (started) {
                    g1_add(acc, acc, jp);
                } else {
                    acc = jp;
                    started = true;
                }
            }
        }
    }
    rhs = acc;
    return g1_eq(lhs, rhs);
}

bool g1_decompress(G1Affine &out, const uint8_t in[48]) {
    const uint8_t flags = in[0];
    const bool compressed = (flags & 0x80) != 0;
    const bool infinity = (flags & 0x40) != 0;
    const bool sign = (flags & 0x20) != 0;
    if (!compressed) return false; // we only accept the compressed encoding

    uint8_t xb[48];
    memcpy(xb, in, 48);
    xb[0] &= 0x1f;

    if (infinity) {
        // All remaining bits must be zero for a canonical encoding.
        for (int i = 0; i < 48; i++) {
            if (xb[i] != 0) return false;
        }
        if (sign) return false;
        out.x = kFpZero;
        out.y = kFpZero;
        return true;
    }

    if (!fp_from_bytes(out.x, xb)) return false;
    Fp y2, four;
    fp_sqr(y2, out.x);
    fp_mul(y2, y2, out.x);
    fp_from_u64(four, 4);
    fp_add(y2, y2, four);
    // p == 3 (mod 4), so sqrt(a) == a^((p+1)/4).
    static const uint64_t e[6] = VKP_FP_P_MINUS_3_DIV_4;
    Fp y;
    fp_pow_limbs(y, y2, e, 6);
    fp_mul(y, y, y2); // a^((p-3)/4) * a == a^((p+1)/4)
    Fp check;
    fp_sqr(check, y);
    if (!fp_eq(check, y2)) return false; // not a quadratic residue
    if (fp_is_lexicographically_largest(y) != sign) fp_neg(y, y);
    out.y = y;
    return true;
}

void g1_compress(uint8_t out[48], const G1Affine &p) {
    if (fp_is_zero(p.x) && fp_is_zero(p.y)) {
        memset(out, 0, 48);
        out[0] = 0xc0;
        return;
    }
    fp_to_bytes(out, p.x);
    out[0] |= 0x80;
    if (fp_is_lexicographically_largest(p.y)) out[0] |= 0x20;
}


uint32_t bit_reverse(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

} // namespace vkp
