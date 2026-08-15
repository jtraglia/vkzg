// BLS12-381 arithmetic for Metal.
//
// Fp is 12 x 32-bit limbs, Fr is 8 x 32-bit limbs, both little-endian and in
// Montgomery form.  Multiplication is CIOS with a 64-bit accumulator: on Apple7
// that measured ~374M Fp multiplies/s, about 25% faster than hand-rolled 32-bit
// carry chains, because the compiler maps `(ulong)a*b` onto mul/mulhi directly.
//
// The constants block (FP_P, FP_N0, ...) is prepended from
// bls12_381_constants.h.inc, which is generated alongside the host constants.

// ---------------------------------------------------------------------- Fp

struct Fp {
    uint v[FP_NLIMBS];
};

struct Fr {
    uint v[FR_NLIMBS];
};

static inline Fp fp_zero() {
    Fp r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = 0u;
    return r;
}

static inline Fp fp_one() {
    Fp r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = FP_R[i];
    return r;
}

static inline bool fp_is_zero(thread const Fp &a) {
    uint acc = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) acc |= a.v[i];
    return acc == 0u;
}

static inline bool fp_eq(thread const Fp &a, thread const Fp &b) {
    uint acc = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) acc |= (a.v[i] ^ b.v[i]);
    return acc == 0u;
}

static inline Fp fp_add(thread const Fp &a, thread const Fp &b) {
    uint t[FP_NLIMBS];
    uint carry = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + (ulong)carry;
        t[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    // p has room in the top limb, so the sum never overflows 12 limbs.
    uint s2[FP_NLIMBS];
    uint borrow = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) {
        ulong d = (ulong)t[i] - (ulong)FP_P[i] - (ulong)borrow;
        s2[i] = (uint)d;
        borrow = (uint)((d >> 32) & 1u);
    }
    Fp r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = borrow ? t[i] : s2[i];
    return r;
}

static inline Fp fp_sub(thread const Fp &a, thread const Fp &b) {
    uint t[FP_NLIMBS];
    uint borrow = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - (ulong)borrow;
        t[i] = (uint)d;
        borrow = (uint)((d >> 32) & 1u);
    }
    uint carry = 0u;
    uint mask = borrow ? 0xffffffffu : 0u;
    Fp r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) {
        ulong s = (ulong)t[i] + (ulong)(FP_P[i] & mask) + (ulong)carry;
        r.v[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    return r;
}

static inline Fp fp_neg(thread const Fp &a) {
    if (fp_is_zero(a)) return a;
    Fp z = fp_zero();
    return fp_sub(z, a);
}

static inline Fp fp_dbl(thread const Fp &a) { return fp_add(a, a); }

static inline Fp fp_mul(thread const Fp &a, thread const Fp &b) {
    uint t[FP_NLIMBS + 2];
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS + 2; i++) t[i] = 0u;

#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) {
        uint c = 0u;
        const uint bi = b.v[i];
#pragma clang loop unroll(full)
        for (int j = 0; j < FP_NLIMBS; j++) {
            ulong s = (ulong)a.v[j] * (ulong)bi + (ulong)t[j] + (ulong)c;
            t[j] = (uint)s;
            c = (uint)(s >> 32);
        }
        ulong s = (ulong)t[FP_NLIMBS] + (ulong)c;
        t[FP_NLIMBS] = (uint)s;
        t[FP_NLIMBS + 1] = (uint)(s >> 32);

        const uint m = t[0] * FP_N0;
        ulong s0 = (ulong)m * (ulong)FP_P[0] + (ulong)t[0];
        c = (uint)(s0 >> 32);
#pragma clang loop unroll(full)
        for (int j = 1; j < FP_NLIMBS; j++) {
            ulong sj = (ulong)m * (ulong)FP_P[j] + (ulong)t[j] + (ulong)c;
            t[j - 1] = (uint)sj;
            c = (uint)(sj >> 32);
        }
        ulong sn = (ulong)t[FP_NLIMBS] + (ulong)c;
        t[FP_NLIMBS - 1] = (uint)sn;
        t[FP_NLIMBS] = t[FP_NLIMBS + 1] + (uint)(sn >> 32);
    }

    // The CIOS result is below 2p; subtract once when needed.
    uint s2[FP_NLIMBS];
    uint borrow = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) {
        ulong d = (ulong)t[i] - (ulong)FP_P[i] - (ulong)borrow;
        s2[i] = (uint)d;
        borrow = (uint)((d >> 32) & 1u);
    }
    borrow = (t[FP_NLIMBS] < borrow) ? 1u : 0u;
    Fp r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = borrow ? t[i] : s2[i];
    return r;
}

static inline Fp fp_sqr(thread const Fp &a) { return fp_mul(a, a); }

// ---------------------------------------------------------------------- Fr

static inline Fr fr_zero() {
    Fr r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = 0u;
    return r;
}


static inline Fr fr_add(thread const Fr &a, thread const Fr &b) {
    uint t[FR_NLIMBS];
    uint carry = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) {
        ulong s = (ulong)a.v[i] + (ulong)b.v[i] + (ulong)carry;
        t[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    uint s2[FR_NLIMBS];
    uint borrow = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) {
        ulong d = (ulong)t[i] - (ulong)FR_P[i] - (ulong)borrow;
        s2[i] = (uint)d;
        borrow = (uint)((d >> 32) & 1u);
    }
    Fr r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = borrow ? t[i] : s2[i];
    return r;
}

static inline Fr fr_sub(thread const Fr &a, thread const Fr &b) {
    uint t[FR_NLIMBS];
    uint borrow = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) {
        ulong d = (ulong)a.v[i] - (ulong)b.v[i] - (ulong)borrow;
        t[i] = (uint)d;
        borrow = (uint)((d >> 32) & 1u);
    }
    uint mask = borrow ? 0xffffffffu : 0u;
    uint carry = 0u;
    Fr r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) {
        ulong s = (ulong)t[i] + (ulong)(FR_P[i] & mask) + (ulong)carry;
        r.v[i] = (uint)s;
        carry = (uint)(s >> 32);
    }
    return r;
}

static inline Fr fr_mul(thread const Fr &a, thread const Fr &b) {
    uint t[FR_NLIMBS + 2];
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS + 2; i++) t[i] = 0u;

#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) {
        uint c = 0u;
        const uint bi = b.v[i];
#pragma clang loop unroll(full)
        for (int j = 0; j < FR_NLIMBS; j++) {
            ulong s = (ulong)a.v[j] * (ulong)bi + (ulong)t[j] + (ulong)c;
            t[j] = (uint)s;
            c = (uint)(s >> 32);
        }
        ulong s = (ulong)t[FR_NLIMBS] + (ulong)c;
        t[FR_NLIMBS] = (uint)s;
        t[FR_NLIMBS + 1] = (uint)(s >> 32);

        const uint m = t[0] * FR_N0;
        ulong s0 = (ulong)m * (ulong)FR_P[0] + (ulong)t[0];
        c = (uint)(s0 >> 32);
#pragma clang loop unroll(full)
        for (int j = 1; j < FR_NLIMBS; j++) {
            ulong sj = (ulong)m * (ulong)FR_P[j] + (ulong)t[j] + (ulong)c;
            t[j - 1] = (uint)sj;
            c = (uint)(sj >> 32);
        }
        ulong sn = (ulong)t[FR_NLIMBS] + (ulong)c;
        t[FR_NLIMBS - 1] = (uint)sn;
        t[FR_NLIMBS] = t[FR_NLIMBS + 1] + (uint)(sn >> 32);
    }

    uint s2[FR_NLIMBS];
    uint borrow = 0u;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) {
        ulong d = (ulong)t[i] - (ulong)FR_P[i] - (ulong)borrow;
        s2[i] = (uint)d;
        borrow = (uint)((d >> 32) & 1u);
    }
    borrow = (t[FR_NLIMBS] < borrow) ? 1u : 0u;
    Fr r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = borrow ? t[i] : s2[i];
    return r;
}

// Leaves Montgomery form: multiplying by 1 divides out R.
static inline Fr fr_to_canonical(thread const Fr &a) {
    Fr one = fr_zero();
    one.v[0] = 1u;
    return fr_mul(a, one);
}

// Enters Montgomery form.
static inline Fr fr_from_canonical(thread const Fr &a) {
    Fr r2;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) r2.v[i] = FR_R2[i];
    return fr_mul(a, r2);
}

// true when a < r, i.e. the value is a canonical field element.
static inline bool fr_is_canonical(thread const Fr &a) {
    for (int i = FR_NLIMBS - 1; i >= 0; i--) {
        if (a.v[i] != FR_P[i]) return a.v[i] < FR_P[i];
    }
    return false; // exactly r
}

// ---------------------------------------------------------------------- G1

// Jacobian (X : Y : Z) == (X/Z^2, Y/Z^3); Z == 0 is the point at infinity.
struct G1 {
    Fp x, y, z;
};
// Affine, with (0, 0) reserved for the point at infinity.
struct G1A {
    Fp x, y;
};

static inline G1 g1_identity() {
    G1 r;
    r.x = fp_zero();
    r.y = fp_one();
    r.z = fp_zero();
    return r;
}

static inline bool g1a_is_identity(thread const G1A &p) {
    return fp_is_zero(p.x) && fp_is_zero(p.y);
}

// dbl-2009-l, for curves with a == 0.
static inline G1 g1_dbl(thread const G1 &p) {
    if (fp_is_zero(p.z)) return p;
    Fp A = fp_sqr(p.x);
    Fp B = fp_sqr(p.y);
    Fp C = fp_sqr(B);
    Fp t0 = fp_add(p.x, B);
    t0 = fp_sqr(t0);
    t0 = fp_sub(t0, A);
    t0 = fp_sub(t0, C);
    Fp D = fp_dbl(t0);
    Fp E = fp_add(fp_dbl(A), A);
    Fp F = fp_sqr(E);
    G1 r;
    r.x = fp_sub(F, fp_dbl(D));
    Fp t1 = fp_sub(D, r.x);
    t1 = fp_mul(E, t1);
    Fp t2 = fp_dbl(fp_dbl(fp_dbl(C)));
    r.y = fp_sub(t1, t2);
    r.z = fp_dbl(fp_mul(p.y, p.z));
    return r;
}

// add-2007-bl
static inline G1 g1_add(thread const G1 &a, thread const G1 &b) {
    if (fp_is_zero(a.z)) return b;
    if (fp_is_zero(b.z)) return a;
    Fp Z1Z1 = fp_sqr(a.z);
    Fp Z2Z2 = fp_sqr(b.z);
    Fp U1 = fp_mul(a.x, Z2Z2);
    Fp U2 = fp_mul(b.x, Z1Z1);
    Fp S1 = fp_mul(a.y, fp_mul(b.z, Z2Z2));
    Fp S2 = fp_mul(b.y, fp_mul(a.z, Z1Z1));
    if (fp_eq(U1, U2)) {
        if (fp_eq(S1, S2)) return g1_dbl(a);
        return g1_identity();
    }
    Fp H = fp_sub(U2, U1);
    Fp I = fp_sqr(fp_dbl(H));
    Fp J = fp_mul(H, I);
    Fp K = fp_dbl(fp_sub(S2, S1));
    Fp V = fp_mul(U1, I);
    G1 r;
    r.x = fp_sub(fp_sub(fp_sqr(K), J), fp_dbl(V));
    Fp t = fp_mul(K, fp_sub(V, r.x));
    r.y = fp_sub(t, fp_dbl(fp_mul(S1, J)));
    Fp z = fp_add(a.z, b.z);
    z = fp_sqr(z);
    z = fp_sub(z, Z1Z1);
    z = fp_sub(z, Z2Z2);
    r.z = fp_mul(z, H);
    return r;
}

// madd-2007-bl: Jacobian + affine.
static inline G1 g1_madd(thread const G1 &a, thread const G1A &b) {
    if (g1a_is_identity(b)) return a;
    if (fp_is_zero(a.z)) {
        G1 r;
        r.x = b.x;
        r.y = b.y;
        r.z = fp_one();
        return r;
    }
    Fp Z1Z1 = fp_sqr(a.z);
    Fp U2 = fp_mul(b.x, Z1Z1);
    Fp S2 = fp_mul(b.y, fp_mul(a.z, Z1Z1));
    if (fp_eq(a.x, U2)) {
        if (fp_eq(a.y, S2)) return g1_dbl(a);
        return g1_identity();
    }
    Fp H = fp_sub(U2, a.x);
    Fp HH = fp_sqr(H);
    Fp I = fp_dbl(fp_dbl(HH));
    Fp J = fp_mul(H, I);
    Fp K = fp_dbl(fp_sub(S2, a.y));
    Fp V = fp_mul(a.x, I);
    G1 r;
    r.x = fp_sub(fp_sub(fp_sqr(K), J), fp_dbl(V));
    Fp t = fp_mul(K, fp_sub(V, r.x));
    r.y = fp_sub(t, fp_dbl(fp_mul(a.y, J)));
    Fp z = fp_add(a.z, H);
    z = fp_sqr(z);
    z = fp_sub(z, Z1Z1);
    r.z = fp_sub(z, HH);
    return r;
}

// ------------------------------------------------------------ memory access

static inline Fp fp_load(device const uint *p) {
    Fp r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = p[i];
    return r;
}

static inline void fp_store(device uint *p, thread const Fp &a) {
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) p[i] = a.v[i];
}

static inline G1A g1a_load(device const uint *p) {
    G1A r;
    r.x = fp_load(p);
    r.y = fp_load(p + FP_NLIMBS);
    return r;
}


static inline G1 g1_load(device const uint *p) {
    G1 r;
    r.x = fp_load(p);
    r.y = fp_load(p + FP_NLIMBS);
    r.z = fp_load(p + 2 * FP_NLIMBS);
    return r;
}

static inline void g1_store(device uint *p, thread const G1 &a) {
    fp_store(p, a.x);
    fp_store(p + FP_NLIMBS, a.y);
    fp_store(p + 2 * FP_NLIMBS, a.z);
}



static inline Fr fr_load(device const uint *p) {
    Fr r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = p[i];
    return r;
}

static inline void fr_store(device uint *p, thread const Fr &a) {
#pragma clang loop unroll(full)
    for (int i = 0; i < FR_NLIMBS; i++) p[i] = a.v[i];
}

// ---------------------------------------------------- SIMD-group reduction

static inline G1 g1_shuffle_down(thread const G1 &p, ushort delta) {
    G1 r;
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.x.v[i] = simd_shuffle_down(p.x.v[i], delta);
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.y.v[i] = simd_shuffle_down(p.y.v[i], delta);
#pragma clang loop unroll(full)
    for (int i = 0; i < FP_NLIMBS; i++) r.z.v[i] = simd_shuffle_down(p.z.v[i], delta);
    return r;
}

// Collapses 128 buckets into sum_{k=1}^{128} k * B[k-1].
//
// L_REDUCE_LANES lanes cooperate on one output, each first collapsing a
// contiguous run of buckets with a serial running sum, then combining across
// lanes with a shuffle tree.  The split matters: a pure shuffle tree over all
// 128 buckets executes at full SIMD width on every level, so 32 lanes do the
// work of one useful merge -- measured 40ms per phase.  Doing the bulk serially
// inside a lane and only the last log2(L_REDUCE_LANES) levels as a tree keeps
// the wasted width down while keeping the dependency chain short.
//
// Merging two adjacent ranges of h lanes each:
//     rs = rs_lo + rs_hi
//     w  = w_lo + w_hi + h * s_hi
//     s  = s_lo + s_hi
// and h is a power of two, so the scaling is doublings.
static inline G1 reduce_buckets(device const uint *buckets, uint pos) {
    const uint per_lane = L_NUM_BUCKETS / L_REDUCE_LANES;

    // Serial running sum over this lane's slice, from the top down, giving
    // rs = sum_i (i+1) * B[base+i] and s = sum_i B[base+i].
    G1 s = g1_identity();
    G1 rs = g1_identity();
    for (uint i = per_lane; i-- > 0u;) {
        const G1 bk = g1_load(buckets + (ulong)(pos * per_lane + i) * L_JACOBIAN_WORDS);
        s = g1_add(s, bk);
        rs = g1_add(rs, s);
    }
    G1 w = g1_identity();

    for (uint level = 0u; (1u << level) < L_REDUCE_LANES; level++) {
        const ushort h = (ushort)(1u << level);
        // Every lane must reach the shuffle, so compute first and commit after.
        const G1 sb = g1_shuffle_down(s, h);
        const G1 wb = g1_shuffle_down(w, h);
        const G1 rb = g1_shuffle_down(rs, h);
        G1 hs = sb;
        for (uint t = 0u; t < level; t++) hs = g1_dbl(hs);
        const G1 nw = g1_add(w, g1_add(wb, hs));
        const G1 ns = g1_add(s, sb);
        const G1 nr = g1_add(rs, rb);
        if ((pos & ((1u << (level + 1u)) - 1u)) == 0u) {
            w = nw;
            s = ns;
            rs = nr;
        }
    }

    // Lane 0 holds rs = sum_p RS_p and w = sum_p p * GS_p, and
    //     sum_k (k+1) B_k = sum_p [ RS_p + per_lane * p * GS_p ].
    for (uint t = 0u; t < L_LOG_REDUCE_PER_LANE; t++) w = g1_dbl(w);
    return g1_add(rs, w);
}

// ------------------------------------------------------- inversion, encoding

// a^(p-2) by square-and-multiply.  ~381 squarings and ~190 multiplies, so the
// dependency chain is ~570 F_p multiplies deep -- about 2ms on this GPU.  That
// is a fixed cost per dispatch regardless of how many points are being
// normalised, which is why the batched Montgomery trick around it matters: one
// inversion serves a whole chunk.
static inline Fp fp_inv(thread const Fp &a) {
    Fp acc = a;
    bool started = false;
#pragma clang loop unroll(disable)
    for (int i = FP_NLIMBS - 1; i >= 0; i--) {
        const uint limb = FP_P_MINUS_2[i];
#pragma clang loop unroll(disable)
        for (int b = 31; b >= 0; b--) {
            if (started) acc = fp_sqr(acc);
            if ((limb >> b) & 1u) {
                if (started) {
                    acc = fp_mul(acc, a);
                } else {
                    acc = a;
                    started = true;
                }
            }
        }
    }
    return acc;
}

// Leaves Montgomery form: multiplying by 1 divides out R.
static inline Fp fp_to_canonical(thread const Fp &a) {
    Fp one = fp_zero();
    one.v[0] = 1u;
    return fp_mul(a, one);
}

// True when the canonical representative exceeds (p-1)/2, which is the sign bit
// the compressed encoding carries.
static inline bool fp_is_lex_largest(thread const Fp &canonical) {
    for (int i = FP_NLIMBS - 1; i >= 0; i--) {
        if (canonical.v[i] != FP_P_HALF[i]) return canonical.v[i] > FP_P_HALF[i];
    }
    return false;
}
