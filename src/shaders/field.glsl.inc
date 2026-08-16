// BLS12-381 arithmetic for GLSL compute shaders (Vulkan).
//
// Fp is 12 x 32-bit limbs, Fr is 8 x 32-bit limbs, both little-endian and in
// Montgomery form.  Multiplication is CIOS with a 64-bit accumulator, mapping
// onto the same widening-multiply hardware path Metal's `(ulong)a*b` used.
//
// The constants block (FP_P, FP_N0, ...) is included from
// bls12_381_constants.glsl.inc, which is generated alongside the host
// constants so the two limb representations cannot drift.
//
// Buffers are passed between host and shader as raw GPU addresses
// (VK_KHR_buffer_device_address) rather than descriptor-bound arrays, so
// every kernel's buffer arguments are `buffer_reference` pointers -- the
// closest GLSL analogue of Metal's `device T *` arguments, and it keeps the
// shape of the kernels below close to the .metal originals.
//
// Every small fixed-trip-count loop below is marked [[unroll]], matching the
// `#pragma clang loop unroll(full)` the Metal original used throughout: with
// no hint, glslang emits a plain SPIR-V loop (OpLoopMerge ... None) and lets
// the driver's backend decide, and on this driver that turned out to matter
// a lot for fp_inv specifically (see the [[dont_unroll]] note there).

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_control_flow_attributes : require

#include "layout_defs.h"
#include "bls12_381_constants.glsl.inc"

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Buf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16Buf { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8Buf { uint8_t v[]; };

// ---------------------------------------------------------------------- Fp

struct Fp {
    uint v[FP_NLIMBS];
};

struct Fr {
    uint v[FR_NLIMBS];
};

Fp fp_zero() {
    Fp r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = 0u;
    return r;
}

Fp fp_one() {
    Fp r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = FP_R[i];
    return r;
}

bool fp_is_zero(Fp a) {
    uint acc = 0u;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) acc |= a.v[i];
    return acc == 0u;
}

bool fp_eq(Fp a, Fp b) {
    uint acc = 0u;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) acc |= (a.v[i] ^ b.v[i]);
    return acc == 0u;
}

Fp fp_add(Fp a, Fp b) {
    uint t[FP_NLIMBS];
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) {
        uint64_t s = uint64_t(a.v[i]) + uint64_t(b.v[i]) + uint64_t(carry);
        t[i] = uint(s);
        carry = uint(s >> 32);
    }
    // p has room in the top limb, so the sum never overflows 12 limbs.
    uint s2[FP_NLIMBS];
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) {
        uint64_t d = uint64_t(t[i]) - uint64_t(FP_P[i]) - uint64_t(borrow);
        s2[i] = uint(d);
        borrow = uint((d >> 32) & 1ul);
    }
    Fp r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = borrow != 0u ? t[i] : s2[i];
    return r;
}

Fp fp_sub(Fp a, Fp b) {
    uint t[FP_NLIMBS];
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) {
        uint64_t d = uint64_t(a.v[i]) - uint64_t(b.v[i]) - uint64_t(borrow);
        t[i] = uint(d);
        borrow = uint((d >> 32) & 1ul);
    }
    uint carry = 0u;
    uint mask = borrow != 0u ? 0xffffffffu : 0u;
    Fp r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) {
        uint64_t s = uint64_t(t[i]) + uint64_t(FP_P[i] & mask) + uint64_t(carry);
        r.v[i] = uint(s);
        carry = uint(s >> 32);
    }
    return r;
}

Fp fp_neg(Fp a) {
    if (fp_is_zero(a)) return a;
    return fp_sub(fp_zero(), a);
}

Fp fp_dbl(Fp a) { return fp_add(a, a); }

Fp fp_mul(Fp a, Fp b) {
    uint t[FP_NLIMBS + 2];
    [[unroll]] for (int i = 0; i < FP_NLIMBS + 2; i++) t[i] = 0u;

    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) {
        uint c = 0u;
        const uint bi = b.v[i];
        [[unroll]] for (int j = 0; j < FP_NLIMBS; j++) {
            uint64_t s = uint64_t(a.v[j]) * uint64_t(bi) + uint64_t(t[j]) + uint64_t(c);
            t[j] = uint(s);
            c = uint(s >> 32);
        }
        uint64_t s = uint64_t(t[FP_NLIMBS]) + uint64_t(c);
        t[FP_NLIMBS] = uint(s);
        t[FP_NLIMBS + 1] = uint(s >> 32);

        const uint m = t[0] * FP_N0;
        uint64_t s0 = uint64_t(m) * uint64_t(FP_P[0]) + uint64_t(t[0]);
        c = uint(s0 >> 32);
        [[unroll]] for (int j = 1; j < FP_NLIMBS; j++) {
            uint64_t sj = uint64_t(m) * uint64_t(FP_P[j]) + uint64_t(t[j]) + uint64_t(c);
            t[j - 1] = uint(sj);
            c = uint(sj >> 32);
        }
        uint64_t sn = uint64_t(t[FP_NLIMBS]) + uint64_t(c);
        t[FP_NLIMBS - 1] = uint(sn);
        t[FP_NLIMBS] = t[FP_NLIMBS + 1] + uint(sn >> 32);
    }

    // The CIOS result is below 2p; subtract once when needed.
    uint s2[FP_NLIMBS];
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) {
        uint64_t d = uint64_t(t[i]) - uint64_t(FP_P[i]) - uint64_t(borrow);
        s2[i] = uint(d);
        borrow = uint((d >> 32) & 1ul);
    }
    borrow = (t[FP_NLIMBS] < borrow) ? 1u : 0u;
    Fp r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = borrow != 0u ? t[i] : s2[i];
    return r;
}

Fp fp_sqr(Fp a) { return fp_mul(a, a); }

// ---------------------------------------------------------------------- Fr

Fr fr_zero() {
    Fr r;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = 0u;
    return r;
}

Fr fr_add(Fr a, Fr b) {
    uint t[FR_NLIMBS];
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) {
        uint64_t s = uint64_t(a.v[i]) + uint64_t(b.v[i]) + uint64_t(carry);
        t[i] = uint(s);
        carry = uint(s >> 32);
    }
    uint s2[FR_NLIMBS];
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) {
        uint64_t d = uint64_t(t[i]) - uint64_t(FR_P[i]) - uint64_t(borrow);
        s2[i] = uint(d);
        borrow = uint((d >> 32) & 1ul);
    }
    Fr r;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = borrow != 0u ? t[i] : s2[i];
    return r;
}

Fr fr_sub(Fr a, Fr b) {
    uint t[FR_NLIMBS];
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) {
        uint64_t d = uint64_t(a.v[i]) - uint64_t(b.v[i]) - uint64_t(borrow);
        t[i] = uint(d);
        borrow = uint((d >> 32) & 1ul);
    }
    uint mask = borrow != 0u ? 0xffffffffu : 0u;
    uint carry = 0u;
    Fr r;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) {
        uint64_t s = uint64_t(t[i]) + uint64_t(FR_P[i] & mask) + uint64_t(carry);
        r.v[i] = uint(s);
        carry = uint(s >> 32);
    }
    return r;
}

Fr fr_mul(Fr a, Fr b) {
    uint t[FR_NLIMBS + 2];
    [[unroll]] for (int i = 0; i < FR_NLIMBS + 2; i++) t[i] = 0u;

    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) {
        uint c = 0u;
        const uint bi = b.v[i];
        [[unroll]] for (int j = 0; j < FR_NLIMBS; j++) {
            uint64_t s = uint64_t(a.v[j]) * uint64_t(bi) + uint64_t(t[j]) + uint64_t(c);
            t[j] = uint(s);
            c = uint(s >> 32);
        }
        uint64_t s = uint64_t(t[FR_NLIMBS]) + uint64_t(c);
        t[FR_NLIMBS] = uint(s);
        t[FR_NLIMBS + 1] = uint(s >> 32);

        const uint m = t[0] * FR_N0;
        uint64_t s0 = uint64_t(m) * uint64_t(FR_P[0]) + uint64_t(t[0]);
        c = uint(s0 >> 32);
        [[unroll]] for (int j = 1; j < FR_NLIMBS; j++) {
            uint64_t sj = uint64_t(m) * uint64_t(FR_P[j]) + uint64_t(t[j]) + uint64_t(c);
            t[j - 1] = uint(sj);
            c = uint(sj >> 32);
        }
        uint64_t sn = uint64_t(t[FR_NLIMBS]) + uint64_t(c);
        t[FR_NLIMBS - 1] = uint(sn);
        t[FR_NLIMBS] = t[FR_NLIMBS + 1] + uint(sn >> 32);
    }

    uint s2[FR_NLIMBS];
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) {
        uint64_t d = uint64_t(t[i]) - uint64_t(FR_P[i]) - uint64_t(borrow);
        s2[i] = uint(d);
        borrow = uint((d >> 32) & 1ul);
    }
    borrow = (t[FR_NLIMBS] < borrow) ? 1u : 0u;
    Fr r;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = borrow != 0u ? t[i] : s2[i];
    return r;
}

// Leaves Montgomery form: multiplying by 1 divides out R.
Fr fr_to_canonical(Fr a) {
    Fr one = fr_zero();
    one.v[0] = 1u;
    return fr_mul(a, one);
}

// Enters Montgomery form.
Fr fr_from_canonical(Fr a) {
    Fr r2;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) r2.v[i] = FR_R2[i];
    return fr_mul(a, r2);
}

// true when a < r, i.e. the value is a canonical field element.
bool fr_is_canonical(Fr a) {
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

G1 g1_identity() {
    G1 r;
    r.x = fp_zero();
    r.y = fp_one();
    r.z = fp_zero();
    return r;
}

bool g1a_is_identity(G1A p) {
    return fp_is_zero(p.x) && fp_is_zero(p.y);
}

// dbl-2009-l, for curves with a == 0.
G1 g1_dbl(G1 p) {
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
G1 g1_add(G1 a, G1 b) {
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
G1 g1_madd(G1 a, G1A b) {
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

// Negation only flips Y in Jacobian coordinates.
G1 g1_neg(G1 p) {
    G1 r = p;
    r.y = fp_neg(p.y);
    return r;
}

G1 g1_sub(G1 a, G1 b) { return g1_add(a, g1_neg(b)); }

// ------------------------------------------------------------ memory access
//
// Buffers are buffer_reference pointers; `off` is a 32-bit element index
// (word/limb/byte index depending on the buffer's element type), matching
// the word offsets the .metal originals computed before dereferencing.

Fp fp_load(U32Buf buf, uint off) {
    Fp r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.v[i] = buf.v[off + i];
    return r;
}

void fp_store(U32Buf buf, uint off, Fp a) {
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) buf.v[off + i] = a.v[i];
}

G1A g1a_load(U32Buf buf, uint off) {
    G1A r;
    r.x = fp_load(buf, off);
    r.y = fp_load(buf, off + FP_NLIMBS);
    return r;
}

G1 g1_load(U32Buf buf, uint off) {
    G1 r;
    r.x = fp_load(buf, off);
    r.y = fp_load(buf, off + FP_NLIMBS);
    r.z = fp_load(buf, off + 2 * FP_NLIMBS);
    return r;
}

void g1_store(U32Buf buf, uint off, G1 a) {
    fp_store(buf, off, a.x);
    fp_store(buf, off + FP_NLIMBS, a.y);
    fp_store(buf, off + 2 * FP_NLIMBS, a.z);
}

Fr fr_load(U32Buf buf, uint off) {
    Fr r;
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) r.v[i] = buf.v[off + i];
    return r;
}

void fr_store(U32Buf buf, uint off, Fr a) {
    [[unroll]] for (int i = 0; i < FR_NLIMBS; i++) buf.v[off + i] = a.v[i];
}

// ---------------------------------------------------- SIMD-group reduction

G1 g1_shuffle_down(G1 p, uint delta) {
    G1 r;
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.x.v[i] = subgroupShuffleDown(p.x.v[i], delta);
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.y.v[i] = subgroupShuffleDown(p.y.v[i], delta);
    [[unroll]] for (int i = 0; i < FP_NLIMBS; i++) r.z.v[i] = subgroupShuffleDown(p.z.v[i], delta);
    return r;
}

// Collapses 128 buckets into sum_{k=1}^{128} k * B[k-1].
//
// L_REDUCE_LANES lanes cooperate on one output, each first collapsing a
// contiguous run of buckets with a serial running sum, then combining across
// lanes with a shuffle tree.  See kernels/k_bucket_reduce.comp for why the
// split matters; the maths is unchanged from the Metal original.
G1 reduce_buckets(U32Buf buckets, uint base, uint pos) {
    const uint per_lane = L_NUM_BUCKETS / L_REDUCE_LANES;

    // Serial running sum over this lane's slice, from the top down, giving
    // rs = sum_i (i+1) * B[base+i] and s = sum_i B[base+i].
    G1 s = g1_identity();
    G1 rs = g1_identity();
    for (uint i = per_lane; i-- > 0u;) {
        const G1 bk = g1_load(buckets, base + (pos * per_lane + i) * L_JACOBIAN_WORDS);
        s = g1_add(s, bk);
        rs = g1_add(rs, s);
    }
    G1 w = g1_identity();

    for (uint level = 0u; (1u << level) < L_REDUCE_LANES; level++) {
        const uint h = 1u << level;
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
// dependency chain is ~570 F_p multiplies deep.  That is a fixed cost per
// dispatch regardless of how many points are being normalised, which is why
// the batched Montgomery trick around it matters: one inversion serves a
// whole chunk.
//
// [[dont_unroll]] on both loops here is deliberate and matters a lot: fully
// unrolling 12*32 = 384 iterations of branchy code bloats this function
// enormously, and on this driver that showed up as k_normalize's inversion
// pass taking orders of magnitude longer than the point count alone would
// predict (a `normalize_ladder` call at batch 64 dropped from ~1.4s to a few
// ms once this was added) -- almost certainly register-pressure/occupancy
// collapse from the bloated unrolled code, not any change to the algorithm.
Fp fp_inv(Fp a) {
    Fp acc = a;
    bool started = false;
    [[dont_unroll]] for (int i = FP_NLIMBS - 1; i >= 0; i--) {
        const uint limb = FP_P_MINUS_2[i];
        [[dont_unroll]] for (int b = 31; b >= 0; b--) {
            if (started) acc = fp_sqr(acc);
            if (((limb >> b) & 1u) != 0u) {
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
Fp fp_to_canonical(Fp a) {
    Fp one = fp_zero();
    one.v[0] = 1u;
    return fp_mul(a, one);
}

// True when the canonical representative exceeds (p-1)/2, which is the sign bit
// the compressed encoding carries.
bool fp_is_lex_largest(Fp canonical) {
    for (int i = FP_NLIMBS - 1; i >= 0; i--) {
        if (canonical.v[i] != FP_P_HALF[i]) return canonical.v[i] > FP_P_HALF[i];
    }
    return false;
}

uint brev(uint x, uint bits) { return bitfieldReverse(x) >> (32u - bits); }
