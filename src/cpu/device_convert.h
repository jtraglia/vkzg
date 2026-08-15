// Host-side conversions between the device point layout and byte encodings.
//
// The GPU leaves points in Jacobian form because normalising them needs a field
// inversion, which is ~380 sequential squarings -- about 1.5ms of pure latency
// on this GPU, versus ~40us for a threaded batch inversion on the idle CPU.
#pragma once

#include <cstddef>
#include <cstdint>

namespace kzgpu {

// Converts `n` Jacobian points (36 words each) into affine points (24 words
// each) using one batched inversion per worker thread.  Points at infinity
// become (0, 0), which the shaders treat as the identity.
void jacobian_to_affine_device(uint32_t *out_affine, const uint32_t *in_jacobian, size_t n,
                               unsigned threads);

// Builds the phase B doubling ladder: for each of `num_points` input points
// u (Jacobian, device layout) emit L_LADDER_POSITIONS affine points
// 2^(8d) * u, d = 0..31.
//
// This runs on the host on purpose.  It is 248 sequential doublings over only
// 128 points per blob -- almost no parallelism and a long dependency chain,
// which is the worst case for a GPU whose single-thread Fp multiply latency is
// 3.4us and best case for a CPU core.  On the GPU it measured 19.6ms; here it
// is well under a millisecond across the idle cores.
void build_ladder_affine(uint32_t *out_affine, const uint32_t *u_jacobian, size_t num_points,
                         unsigned threads);

// Applies the bit-reversal permutation over each blob's 128 proofs, converts to
// affine, and writes the 48-byte compressed encoding.
void finalize_proofs(uint8_t *out, const uint32_t *jacobian, size_t num_blobs, unsigned threads);

} // namespace kzgpu
