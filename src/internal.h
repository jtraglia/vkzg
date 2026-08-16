// Layout constants shared by the host code and the GLSL shaders.
//
// The numeric values live in layout_defs.h, which is also concatenated into the
// shader source, so host and device cannot disagree about a size.
#pragma once

#include "layout_defs.h"

#include <cstddef>
#include <cstdint>

namespace vkzg {

// ---------------------------------------------------------------- protocol
constexpr int kFieldElementsPerBlob = L_FIELD_ELEMENTS_PER_BLOB;
constexpr int kFieldElementsPerExtBlob = L_FIELD_ELEMENTS_PER_EXT_BLOB;
constexpr int kFieldElementsPerCell = L_FIELD_ELEMENTS_PER_CELL; // l
constexpr int kCellsPerBlob = L_CELLS_PER_BLOB;                  // r
constexpr int kCirculantSize = L_CIRCULANT_SIZE; // 2r, and the number of cells
constexpr int kBytesPerFieldElement = L_BYTES_PER_FIELD_ELEMENT;
constexpr int kBytesPerProof = L_BYTES_PER_PROOF;

// ---------------------------------------------------------------- windowing
// Signed-digit window for both MSM phases.  w = 8 minimises
//   (#terms) * ceil(256/w) + 2^w
// for the term counts we have (64 for phase A, 65 for phase B).
constexpr int kWindowBits = L_WINDOW_BITS;
constexpr int kNumDigits = L_NUM_DIGITS;
constexpr int kNumBuckets = L_NUM_BUCKETS; // digits land in [-127, +128]

constexpr int kPhaseATerms = L_PHASE_A_TERMS;
constexpr int kPhaseAItems = L_PHASE_A_ITEMS;

// Phase B: the fused IFFT/truncate/FFT circulant kernel has 65 non-zero taps
// (index 0 plus every odd index). kernel_items/offsets/perm below still hold
// this flat form -- only the CPU reference (reference.cpp) uses it now; the
// GPU path uses the split form (kCirculantHalf / kPhaseBHalfTerms /
// kPhaseBHalfItems) instead. See layout_defs.h for why.
constexpr int kPhaseBTerms = L_PHASE_B_TERMS;
constexpr int kPhaseBItems = L_PHASE_B_ITEMS;

constexpr int kCirculantHalf = L_CIRCULANT_HALF;
constexpr int kPhaseBHalfTerms = L_PHASE_B_HALF_TERMS;
constexpr int kPhaseBHalfItems = L_PHASE_B_HALF_ITEMS;

constexpr int kLadderPositions = L_LADDER_POSITIONS;

// ---------------------------------------------------------------- device types
// Field elements on the device are little-endian 32-bit limbs in Montgomery
// form.  These sizes are in uint32 units.
constexpr int kFpLimbs = L_FP_WORDS;
constexpr int kFrLimbs = L_FR_WORDS;
constexpr int kAffineWords = L_AFFINE_WORDS;
constexpr int kJacobianWords = L_JACOBIAN_WORDS;

// Total number of precomputed setup points: one per (output, base, position).
constexpr int kPositionTablePoints = kCirculantSize * kPhaseATerms * kNumDigits; // 262144
constexpr size_t kPositionTableWords = (size_t)kPositionTablePoints * kAffineWords;

// Bump when the on-disk cache layout changes.
constexpr uint32_t kTableCacheVersion = 6;
constexpr uint32_t kTableCacheMagic = 0x475A4B50; // "PKZG"

} // namespace vkzg
