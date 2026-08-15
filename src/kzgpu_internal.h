// Layout constants shared by the host code and (textually) the Metal shaders.
//
// The numbers here define the FK20 decomposition and the MSM windowing.  The
// shader source includes a generated copy of these as #defines, so changing a
// value here changes both sides at once.
#pragma once

#include <cstddef>
#include <cstdint>

namespace kzgpu {

// ---------------------------------------------------------------- protocol
constexpr int kFieldElementsPerBlob = 4096;
constexpr int kFieldElementsPerExtBlob = 8192;
constexpr int kFieldElementsPerCell = 64;   // l
constexpr int kCellsPerBlob = 64;           // r == kFieldElementsPerBlob / l
constexpr int kCellsPerExtBlob = 128;       // n
constexpr int kCirculantSize = 128;         // 2r, the FK20 circulant domain
constexpr int kBytesPerFieldElement = 32;
constexpr int kBytesPerProof = 48;

// ---------------------------------------------------------------- windowing
// Signed-digit window for both MSM phases.  w = 8 minimises
//   (#terms) * ceil(256/w) + 2^w
// for the term counts we have (64 for phase A, 65 for phase B).
constexpr int kWindowBits = 8;
constexpr int kNumDigits = 32;   // ceil(256 / kWindowBits)
constexpr int kNumBuckets = 128; // 2^(w-1); digits land in [-127, +128]

// Phase A: 64 bases x 32 digits per output.
constexpr int kPhaseATerms = kFieldElementsPerCell;
constexpr int kPhaseAItems = kPhaseATerms * kNumDigits; // 2048

// Phase B: the fused IFFT/truncate/FFT circulant kernel has 65 non-zero taps
// (index 0 plus every odd index).
constexpr int kPhaseBTerms = 65;
constexpr int kPhaseBItems = kPhaseBTerms * kNumDigits; // 2080

// Number of ladder positions the phase B kernel needs (2^(8d) * u[j]).
constexpr int kLadderPositions = kNumDigits; // 32

// ---------------------------------------------------------------- device types
// Field elements on the device are little-endian 32-bit limbs in Montgomery
// form.  These sizes are in uint32 units.
constexpr int kFpLimbs = 12;
constexpr int kFrLimbs = 8;
constexpr int kAffineWords = 2 * kFpLimbs;  // 24
constexpr int kJacobianWords = 3 * kFpLimbs; // 36

// Total number of precomputed setup points: one per (output, base, position).
constexpr int kPositionTablePoints = kCirculantSize * kPhaseATerms * kNumDigits; // 262144
constexpr size_t kPositionTableWords = (size_t)kPositionTablePoints * kAffineWords;

// Bump when the on-disk cache layout changes.
constexpr uint32_t kTableCacheVersion = 3;
constexpr uint32_t kTableCacheMagic = 0x475A4B50; // "PKZG"

} // namespace kzgpu
