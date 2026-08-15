// Development-only stage timing.
//
// Not part of the public API.  Recorded by the real compute path.
#pragma once

#include "../include/metal_prover.h"

#include <cstddef>

namespace mp {

struct StageTimes {
    double scalar_stage = 0; // blob deserialise, both NTTs, cells, circulant
    double phase_a = 0;      // recode/sort + fixed-base bucket MSM
    double reduce_a = 0;
    double ladder = 0;
    double normalize_ladder = 0;
    double phase_b = 0; // the fused circulant map
    double reduce_b = 0;
    double normalize_proofs = 0;
    double compress = 0;
    double total = 0;    // wall time including the host memcpys
};

// Runs the normal compute path and reports what it measured.
mp_result profile_batch(mp_prover *p, unsigned char *cells, unsigned char *proofs,
                           const unsigned char *blobs, unsigned batch, StageTimes &out);

} // namespace mp
