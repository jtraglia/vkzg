/*
 * Single source of truth for the sizes shared between host and device.
 *
 * This file is plain #defines so that it can be included by C++ *and*
 * included into the GLSL shader sources.  internal.h turns these
 * into typed constexprs for the host; the shaders use the macros directly.
 */
#ifndef VKZG_LAYOUT_DEFS_H
#define VKZG_LAYOUT_DEFS_H

#define L_FIELD_ELEMENTS_PER_BLOB 4096
#define L_FIELD_ELEMENTS_PER_EXT_BLOB 8192
#define L_FIELD_ELEMENTS_PER_CELL 64
#define L_CIRCULANT_SIZE 128
#define L_BYTES_PER_BLOB 131072
#define L_BYTES_PER_PROOF 48

#define L_LOG_BLOB 12
#define L_LOG_CIRCULANT 7

/* Signed-digit MSM window. */
#define L_WINDOW_BITS 8
#define L_NUM_DIGITS 32
#define L_NUM_BUCKETS 128

#define L_PHASE_A_TERMS 64
#define L_PHASE_A_ITEMS 2048 /* L_PHASE_A_TERMS * L_NUM_DIGITS */
#define L_LADDER_POSITIONS 32

/*
 * Phase B, split form: X^128-1 = (X^64-1)(X^64+1) splits the 128-point
 * circulant convolution into a 64-point cyclic half and a 64-point
 * negacyclic half (sign flips when a tap wraps past the output index --
 * a runtime check, not a table). Halves the tap*output cost (8320 -> 4224).
 * Verified against the flat form during development; the CPU reference
 * used for that check has since been removed, so this is now verified only
 * indirectly, by the split form's proofs matching the consensus-spec
 * vectors (tests/test.cpp).
 *
 * Both halves run as one dispatch (k_phase_b_split.comp, branching per
 * workgroup) sharing one k_bucket_reduce.comp call: splitting into two
 * dispatches each measured worse at small batch. k_ladder.comp and
 * k_fold_ladder.comp stay separate for the opposite reason -- merging those
 * measured worse too (register pressure).
 */
#define L_CIRCULANT_HALF 64
#define L_PHASE_B_HALF_TERMS 33
#define L_PHASE_B_HALF_ITEMS 1056 /* L_PHASE_B_HALF_TERMS * L_NUM_DIGITS */

/* Element sizes in uint32 words. */
#define L_FP_WORDS 12
#define L_FR_WORDS 8
#define L_AFFINE_WORDS 24
#define L_JACOBIAN_WORDS 36

/*
 * Bucket reduction: L_REDUCE_LANES threads cooperate on one output via a
 * subgroup shuffle tree. More lanes shortens the serial per-thread chain
 * (helps when batch size leaves few workgroups per GPU core) at the cost of
 * more shuffle overhead (hurts once there's already enough parallel work) --
 * where that crossover falls depends on the GPU's core count. So the lane
 * count isn't fixed: k_bucket_reduce.comp declares it as a specialization
 * constant, and recordReduce (vkzg.cpp) picks one of two VkPipelines per
 * dispatch from the batch size and the core count queried at prover-creation
 * time (gpu_topology.cpp), not guessed from a device name. Falls back to the
 * default below when topology can't be queried.
 */
#define L_LOAD_CLASSES 64 /* counting-sort bins for the bucket load ordering */

#define L_REDUCE_LANES 4 /* default lane count when GPU topology can't be queried */

#endif /* VKZG_LAYOUT_DEFS_H */
