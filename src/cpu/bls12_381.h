// BLS12-381 arithmetic for the host.
//
// This is deliberately a small, self-contained implementation rather than a
// dependency on blst: the host only needs it for trusted-setup precomputation,
// the short serialisation tail, and as a reference oracle in tests.  All the
// throughput-critical arithmetic lives in the Metal shaders.
//
// Field elements are held in Montgomery form (R = 2^384 for Fp, R = 2^256 for
// Fr) with little-endian 64-bit limbs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kzgpu {

struct Fp {
    uint64_t v[6];
};
struct Fr {
    uint64_t v[4];
};

// Jacobian coordinates: (X : Y : Z) represents (X/Z^2, Y/Z^3); Z == 0 is the
// point at infinity.
struct G1 {
    Fp x, y, z;
};

// Affine coordinates; the point at infinity is encoded as (0, 0), which is not
// on the curve and so is unambiguous.
struct G1Affine {
    Fp x, y;
};

// ---------------------------------------------------------------------- Fp

extern const Fp kFpZero;
extern const Fp kFpOne;  // == R mod p
extern const Fp kFpBeta; // primitive cube root of unity, for the GLV map

void fp_add(Fp &r, const Fp &a, const Fp &b);
void fp_sub(Fp &r, const Fp &a, const Fp &b);
void fp_neg(Fp &r, const Fp &a);
void fp_mul(Fp &r, const Fp &a, const Fp &b);
void fp_sqr(Fp &r, const Fp &a);
void fp_inv(Fp &r, const Fp &a);
bool fp_is_zero(const Fp &a);
bool fp_eq(const Fp &a, const Fp &b);
// Multiply by 2, 3, 4, 8 -- used by the point formulas.
void fp_dbl(Fp &r, const Fp &a);

// Serialisation. Bytes are big-endian, 48 wide, and represent the *canonical*
// (non-Montgomery) integer.
bool fp_from_bytes(Fp &r, const uint8_t in[48]);
void fp_to_bytes(uint8_t out[48], const Fp &a);
void fp_from_u64(Fp &r, uint64_t x);
// Returns true when the canonical representative is > (p-1)/2, i.e. the
// "lexicographically largest" square root, which is what the compressed point
// encoding signals.
bool fp_is_lexicographically_largest(const Fp &a);

// ---------------------------------------------------------------------- Fr

extern const Fr kFrZero;
extern const Fr kFrOne;

void fr_add(Fr &r, const Fr &a, const Fr &b);
void fr_sub(Fr &r, const Fr &a, const Fr &b);
void fr_neg(Fr &r, const Fr &a);
void fr_mul(Fr &r, const Fr &a, const Fr &b);
void fr_sqr(Fr &r, const Fr &a);
void fr_inv(Fr &r, const Fr &a);
bool fr_is_zero(const Fr &a);
bool fr_eq(const Fr &a, const Fr &b);
void fr_from_u64(Fr &r, uint64_t x);
// Big-endian 32-byte canonical encoding; returns false if >= r.
bool fr_from_bytes(Fr &r, const uint8_t in[32]);
void fr_to_bytes(uint8_t out[32], const Fr &a);
// Canonical (non-Montgomery) little-endian limbs, for scalar recoding.
void fr_to_canonical(uint64_t out[4], const Fr &a);
void fr_from_canonical(Fr &r, const uint64_t in[4]);

// The 2^13-th root of unity, and the root of unity of a given power-of-two
// order (order must divide 2^32, the two-adicity of r is 32).
void fr_root_of_unity(Fr &r, size_t log_order);

// ---------------------------------------------------------------------- G1

extern const G1 kG1Identity;

void g1_set_identity(G1 &r);
bool g1_is_identity(const G1 &p);
void g1_neg(G1 &r, const G1 &p);
void g1_dbl(G1 &r, const G1 &p);
void g1_add(G1 &r, const G1 &a, const G1 &b);
// Mixed addition: b must be affine and not the point at infinity.
void g1_add_mixed(G1 &r, const G1 &a, const G1Affine &b);
void g1_sub(G1 &r, const G1 &a, const G1 &b);
void g1_from_affine(G1 &r, const G1Affine &a);
void g1_mul(G1 &r, const G1 &p, const Fr &k);
bool g1_eq(const G1 &a, const G1 &b);

// Batch conversion to affine: one field inversion for the whole array.
void g1_batch_to_affine(G1Affine *out, const G1 *in, size_t n);
void g1_to_affine(G1Affine &out, const G1 &in);

// Compressed (48 byte) encoding as used by Ethereum.
bool g1_decompress(G1Affine &out, const uint8_t in[48]);
void g1_compress(uint8_t out[48], const G1Affine &p);
bool g1_affine_is_on_curve(const G1Affine &p);
bool g1_affine_in_subgroup(const G1Affine &p);

// GLV: phi(x, y) = (beta*x, y) == [lambda](x, y).
void g1_affine_endo(G1Affine &out, const G1Affine &in);


// ---------------------------------------------------------------- utilities

void batch_inverse(Fp *out, const Fp *in, size_t n);
uint32_t bit_reverse(uint32_t x, uint32_t bits);

} // namespace kzgpu
