// A CPU implementation of the exact pipeline the GPU runs.
//
// This exists for two reasons: it pins down the algorithm (including the fused
// circulant step that replaces c-kzg's two G1 FFTs) in readable code, and it
// gives the shader tests a stage-by-stage oracle to diff against.
#pragma once

#include "../kzgpu_internal.h"
#include "bls12_381.h"
#include "setup.h"

#include <vector>

namespace kzgpu {

struct ReferenceIntermediates {
    Fr poly[kFieldElementsPerBlob];                       // monomial coefficients
    Fr ext_evals[kFieldElementsPerExtBlob];               // pre-BRP cell data
    Fr coeffs[kCirculantSize][kPhaseATerms];              // transposed circulant FFTs
    G1 u[kCirculantSize];                                 // phase A output
    G1 proofs[kCirculantSize];                            // phase B output, pre-BRP
};

// Full pipeline. `cells` and `proofs` may be null.
kzgpu_result reference_compute(const SetupTables &tables, const uint8_t *blob, uint8_t *cells,
                               uint8_t *proofs, ReferenceIntermediates *dbg = nullptr);

// Individual stages, exposed so tests can bisect a mismatch.
kzgpu_result blob_to_polynomial(Fr *poly, const uint8_t *blob, const SetupTables &tables);
void build_circulant_coeffs(Fr coeffs[kCirculantSize][kPhaseATerms], const Fr *poly,
                            const SetupTables &tables);
void phase_a_msm(G1 *u, const Fr coeffs[kCirculantSize][kPhaseATerms], const SetupTables &tables);
void phase_b_circulant(G1 *out, const G1 *u, const SetupTables &tables);

// The straightforward c-kzg-shaped implementation of phase B (G1 inverse
// transform, truncate, forward transform).  Only used to prove the fused
// circulant form agrees with it.
void phase_b_via_g1_ffts(G1 *out, const G1 *u, const SetupTables &tables);

void fr_ifft(Fr *data, size_t n, const SetupTables &tables);
void fr_fft_fwd(Fr *data, size_t n, const SetupTables &tables);

} // namespace kzgpu
