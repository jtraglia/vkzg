// A CPU implementation of the exact pipeline the GPU runs.
//
// This exists for two reasons: it pins down the algorithm (including the fused
// circulant step that replaces c-kzg's two G1 FFTs) in readable code, and it
// gives the shader tests a stage-by-stage oracle to diff against.
#pragma once

#include "../internal.h"
#include "bls12_381.h"
#include "setup.h"

#include <vector>

namespace vkzg {


// Full pipeline: blob -> 128 cell proofs.
vkzg_result reference_compute(const SetupTables &tables, const uint8_t *blob, uint8_t *proofs);
void phase_b_circulant(G1 *out, const G1 *u, const SetupTables &tables);

// The straightforward c-kzg-shaped implementation of phase B (G1 inverse
// transform, truncate, forward transform).  Only used to prove the fused
// circulant form agrees with it.
void phase_b_via_g1_ffts(G1 *out, const G1 *u, const SetupTables &tables);


} // namespace vkzg
