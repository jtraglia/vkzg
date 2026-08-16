// Development-only stage timing.
//
// Not part of the public API.  Recorded by the real compute path.
#pragma once

#include "../include/vulkan_prover.h"

#include <cstddef>

namespace vkp {

struct StageTimes {
    double scalar_stage = 0; // blob deserialise, inverse NTT, circulant
    double phase_a = 0;      // recode/sort + fixed-base bucket MSM
    double reduce_a = 0;
    double ladder = 0;
    double fold_ladder = 0; // L+/L- = L[j] +/- L[j+64]
    double normalize_ladder = 0;
    double phase_b = 0;  // split form: both the cyclic and negacyclic halves
    double reduce_b = 0; // both halves at once (disjoint bucket ranges)
    double combine = 0;  // split form: out[a] = C+[a] +/- C-[a]
    double normalize_proofs = 0;
    double compress = 0;
    double total = 0;    // wall time including the host memcpys
};

// Runs the normal compute path and reports what it measured.
vkp_result profile_batch(vkp_prover *p, unsigned char *proofs, const unsigned char *blobs,
                           unsigned batch, StageTimes &out);

} // namespace vkp
