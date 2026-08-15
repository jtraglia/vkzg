// Development-only stage timing.
//
// Not part of the public API: each stage runs in its own command buffer so the
// numbers include ~0.1ms of submission overhead per stage, which is fine for
// finding where the milliseconds go but not for reporting throughput.
#pragma once

#include "../include/kzgpu.h"

#include <cstddef>

namespace kzgpu {

struct StageTimes {
    double blob_to_fr = 0;
    double ntt_inverse = 0;
    double ntt_forward = 0;
    double serialize_cells = 0;
    double build_circulant = 0;
    double phase_a = 0;
    double reduce_a = 0;
    double ladder = 0;
    double normalize = 0;
    double phase_b = 0;
    double reduce_b = 0;
    double finalize = 0;
    double total = 0;
};

kzgpu_result profile_batch(kzgpu_prover *p, unsigned char *cells, unsigned char *proofs,
                           const unsigned char *blobs, unsigned batch, StageTimes &out);

} // namespace kzgpu
