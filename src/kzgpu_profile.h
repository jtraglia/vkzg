// Development-only stage timing.
//
// Not part of the public API.  These are recorded by the real compute path, so
// they reflect what actually runs, including the concurrent CPU assist: the
// phase entries are wall time for the whole phase (the max of the GPU and CPU
// halves), with the two halves broken out alongside.
#pragma once

#include "../include/kzgpu.h"

#include <cstddef>

namespace kzgpu {

struct StageTimes {
    double scalar_stage = 0; // blob deserialise, both NTTs, cells, circulant
    double phase_a = 0;      // wall time
    double phase_a_gpu = 0;
    double phase_a_cpu = 0;
    double ladder = 0;
    double phase_b = 0;
    double phase_b_gpu = 0;
    double phase_b_cpu = 0;
    double finalize = 0;
    double total = 0;
    int split_a = 0; // outputs given to the GPU
    int split_b = 0;
};

// Runs the normal compute path and reports what it measured.
kzgpu_result profile_batch(kzgpu_prover *p, unsigned char *cells, unsigned char *proofs,
                           const unsigned char *blobs, unsigned batch, StageTimes &out);

} // namespace kzgpu
