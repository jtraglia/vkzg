// Trusted setup ingestion and FK20 table precomputation.
#pragma once

#include "../../include/vulkan_prover.h"
#include "../internal.h"
#include "bls12_381.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vkp {

// Everything the GPU needs, already in device limb layout so the vectors can
// be handed straight to Vulkan buffers.
struct SetupTables {
    // Phase A bases: affine 2^(8d) * P[j][i], indexed
    //     ((j * 64 + i) * 32 + d) * 24 words
    std::vector<uint32_t> position_table;

    // Fr roots of unity of the 8192-element domain and their inverses, so a
    // transform of size N reads with stride 8192/N.
    std::vector<uint32_t> roots_fwd; // 8192 * 8 words
    std::vector<uint32_t> roots_inv; // 8192 * 8 words

    // Phase B circulant kernel, precomputed as a bucket-sorted item list.
    // Each item packs (tap index e, ladder position d, sign); the taps are
    // shared by all 128 outputs, which only differ by a rotation of the point
    // index, so this is built once here rather than per dispatch.
    std::vector<uint32_t> kernel_items;   // kPhaseBItems entries
    std::vector<uint32_t> kernel_offsets; // kNumBuckets + 1 entries
    // Lane -> bucket, ordered by descending item count.  A SIMD group runs
    // until its slowest lane finishes, so handing the heavy buckets to one
    // group instead of spreading them across all four is worth ~1.3x.
    std::vector<uint32_t> kernel_perm; // kNumBuckets entries

    // 1/4096 in Montgomery form, the scale factor for the inverse transform.
    uint32_t inv_blob[kFrLimbs];

    // Digest of the trusted setup this was derived from.
    uint64_t setup_digest = 0;
};

// Digest of the trusted setup bytes, used to validate the on-disk table cache.
uint64_t compute_setup_digest(const uint8_t *g1_monomial_bytes, size_t len);


// Decompresses the setup and derives every table. `validate` additionally runs
// on-curve and subgroup checks on all 4096 points.
vkp_result build_setup_tables(const uint8_t *g1_monomial_bytes, size_t len, bool validate,
                                SetupTables &out);

// On-disk cache of the derived tables.
vkp_result load_table_cache(const std::string &path, uint64_t expected_digest, SetupTables &out);
vkp_result save_table_cache(const std::string &path, const SetupTables &in);

// Exposed for tests: the fused IFFT/truncate/FFT circulant kernel, kappa[e].
void compute_circulant_kernel(Fr kappa[kCirculantSize]);

// Exposed for tests: an in-place radix-2 forward transform over G1 of size n,
// using the 8192-domain root table with the appropriate stride.
void g1_fft(G1 *data, size_t n, const Fr *roots8192);
void fr_fft(Fr *data, size_t n, const Fr *roots8192);

// Exposed for tests: signed-digit recoding of a canonical scalar.
// digits[d] lands in [-127, +128]; returns them as int32.
void recode_scalar(int32_t digits[kNumDigits], const uint64_t canonical[4]);

} // namespace vkp
