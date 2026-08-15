/*
 * Single source of truth for the sizes shared between host and device.
 *
 * This file is plain #defines so that it can be included by C++ *and*
 * included into the GLSL shader sources.  internal.h turns these
 * into typed constexprs for the host; the shaders use the macros directly.
 */
#ifndef VULKAN_PROVER_LAYOUT_DEFS_H
#define VULKAN_PROVER_LAYOUT_DEFS_H

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
 * collapsing L_NUM_BUCKETS / L_REDUCE_LANES buckets serially, then a subgroup
 * shuffle tree across the lanes. At L_REDUCE_LANES == 1 the tree is empty and
 * every thread reduces all 128 buckets on its own, with no shuffles at all --
 * that also means fewer, fatter output-carrying threads per dispatch, which
 * matters below.
 *
 * Metal (Apple GPU, `simd_shuffle_down`) measured 4 lanes as the optimum at
 * batch 64: 4 lanes 46.1 ms/blob, 8 lanes 48.0, 16 lanes 53.4, 32 lanes 71.8,
 * 2 lanes 48.3, 1 lane 49.6 -- cooperation paying for itself down to 2 lanes,
 * only backfiring once you go all the way to 1.
 *
 * On this Vulkan/Honeykrisp driver (`subgroupShuffleDown`) fewer lanes is
 * better at large batch but *worse* at small batch, because L_REDUCE_LANES
 * also sets how many threads a reduce dispatch launches (batch * 128 /
 * L_REDUCE_LANES): fewer lanes means fewer, fatter threads, and at small
 * batch that's not enough threads to hide the GPU's latency, cooperation or
 * not. Full sweep, ms/blob:
 *
 *   lanes   batch 1   batch 8   batch 16   batch 32   batch 64
 *       1     174.2      69.2       60.4       60.7       58.5
 *       2     140.1      63.0       61.3       59.4       59.0
 *       4     120.1      64.7       60.6       60.5       59.5
 *
 * 4 lanes is kept: it is the only setting that doesn't regress the batch >= 8
 * range this library recommends (see the README's Batching section) to chase
 * a ~1.7% win that only shows up at batch 64, and it is dramatically better
 * at batch 1. Worth revisiting if the recommended batch range changes, or on
 * a different Vulkan implementation.
 */
#define L_LOAD_CLASSES 64 /* counting-sort bins for the bucket load ordering */

#define L_REDUCE_LANES 4
#define L_LOG_REDUCE_PER_LANE 5
#define L_REDUCE_OUTPUTS_PER_TG 32 /* 128 threads / L_REDUCE_LANES */

#endif /* VULKAN_PROVER_LAYOUT_DEFS_H */
