/*
 * Single source of truth for the sizes shared between host and device.
 *
 * This file is plain #defines so that it can be included by C++ *and*
 * concatenated into the Metal shader source.  internal.h turns these
 * into typed constexprs for the host; the shaders use the macros directly.
 */
#ifndef METAL_PROVER_LAYOUT_DEFS_H
#define METAL_PROVER_LAYOUT_DEFS_H

#define L_FIELD_ELEMENTS_PER_BLOB 4096
#define L_FIELD_ELEMENTS_PER_EXT_BLOB 8192
#define L_FIELD_ELEMENTS_PER_CELL 64
#define L_CELLS_PER_BLOB 64
#define L_CELLS_PER_EXT_BLOB 128
#define L_CIRCULANT_SIZE 128
#define L_BYTES_PER_FIELD_ELEMENT 32
#define L_BYTES_PER_BLOB 131072
#define L_BYTES_PER_PROOF 48

#define L_LOG_BLOB 12
#define L_LOG_EXT_BLOB 13
#define L_LOG_CIRCULANT 7

/* Signed-digit MSM window. */
#define L_WINDOW_BITS 8
#define L_NUM_DIGITS 32
#define L_NUM_BUCKETS 128

#define L_PHASE_A_TERMS 64
#define L_PHASE_A_ITEMS 2048 /* L_PHASE_A_TERMS * L_NUM_DIGITS */
#define L_PHASE_B_TERMS 65
#define L_PHASE_B_ITEMS 2080 /* L_PHASE_B_TERMS * L_NUM_DIGITS */
#define L_LADDER_POSITIONS 32

/* Element sizes in uint32 words. */
#define L_FP_WORDS 12
#define L_FR_WORDS 8
#define L_AFFINE_WORDS 24
#define L_JACOBIAN_WORDS 36

/*
 * Bucket reduction: L_REDUCE_LANES lanes cooperate on one output, each first
 * collapsing L_NUM_BUCKETS / L_REDUCE_LANES buckets serially, then a shuffle
 * tree across the lanes.
 *
 * Fewer lanes wins once there are enough threads to fill the GPU, because the
 * tree executes at full SIMD width on every level however few lanes are
 * actually merging.  Measured at batch 64, interleaved to cancel thermal
 * drift: 4 lanes 46.1 ms/blob, 8 lanes 48.0, 16 lanes 53.4, 32 lanes 71.8.
 * Below 4 the serial chain starts to dominate (2 lanes 48.3, 1 lane 49.6).
 */
#define L_LOAD_CLASSES 64 /* counting-sort bins for the bucket load ordering */

#define L_REDUCE_LANES 4
#define L_LOG_REDUCE_PER_LANE 5
#define L_REDUCE_OUTPUTS_PER_TG 32 /* 128 threads / L_REDUCE_LANES */

#endif /* METAL_PROVER_LAYOUT_DEFS_H */
