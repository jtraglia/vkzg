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

/*
 * Phase B, split form: the 128-point circulant convolution factors via
 * X^128-1 = (X^64-1)(X^64+1) into a 64-point cyclic sub-convolution ("plus")
 * and a 64-point negacyclic sub-convolution ("minus"), each over half the
 * ladder (folded by cheap point add/sub: L+[j] = L[j]+L[j+64], L-[j] =
 * L[j]-L[j+64]) and each own half-size kappa. Both halves keep the same
 * "shared item list, only the output index rotates" property the flat form
 * has; the negacyclic half additionally flips sign when a tap's offset
 * wraps past the output index (e > a), a cheap per-item runtime check, not
 * a separate precomputed table. Verified equivalent to the flat form (with
 * real G1 arithmetic) in tests/test_reference.cpp.
 *
 * Reconstruction folds the outputs back: out[a] = C+[a] + C-[a], out[a+64] =
 * C+[a] - C-[a] for a in [0,64) (see k_combine_split.comp).
 *
 * Same math as the flat kernel (65 taps at e=0 and every odd e in [1,128)),
 * just halved: each half kernel is nonzero at e=0 and every odd e in
 * [1,64), 33 taps -- so the *tap density* barely changes (65 -> 33+33), but
 * each half only has to cover 64 outputs instead of 128, which is where the
 * saving comes from: total tap*output work (the bucket-MSM accumulation
 * phase B actually pays for) drops from 65*128 = 8320 to 33*64*2 = 4224, a
 * theoretical ~49% cut.
 *
 * The cyclic and negacyclic halves run as *one* dispatch (128 workgroups,
 * a uniform-per-workgroup branch on `a` picks which half's tables to read;
 * see k_phase_b_split.comp), and both halves' buckets get reduced by a
 * single k_bucket_reduce.comp call too, since they occupy disjoint ranges
 * of the same bucket buffer. An earlier version used two dispatches per
 * stage (one per half) and measured noticeably worse at small batch: on
 * this driver each extra dispatch's fixed overhead (submission, the
 * pipeline barrier in vkzg.cpp) costs more than this algorithmic
 * saving does when there isn't much batch to amortise it over. Collapsing
 * to one dispatch per stage keeps the large-batch win without that
 * small-batch cost -- see the README's Results section for the numbers.
 *
 * (k_ladder.comp and k_fold_ladder.comp stay two separate dispatches,
 * *not* similarly merged: an attempt to fold while doubling -- one thread
 * driving both u[j]'s and u[j+64]'s chains at once, halving thread count
 * but doubling live registers per thread -- measured worse than the two
 * separate dispatches, most likely register-pressure/occupancy loss
 * outweighing the saved dispatch. Not every dispatch merge is a win; check.)
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
 * Bucket reduction: L_REDUCE_LANES lanes cooperate on one output, each first
 * collapsing L_NUM_BUCKETS / L_REDUCE_LANES buckets serially, then a subgroup
 * shuffle tree across the lanes. At L_REDUCE_LANES == 1 the tree is empty and
 * every thread reduces all 128 buckets on its own, with no shuffles at all --
 * that also means fewer, fatter output-carrying threads per dispatch.
 *
 * Metal (Apple GPU, `simd_shuffle_down`) measured 4 lanes as the optimum at
 * batch 64: 4 lanes 46.1 ms/blob, 8 lanes 48.0, 16 lanes 53.4, 32 lanes 71.8,
 * 2 lanes 48.3, 1 lane 49.6 -- cooperation paying for itself down to 2 lanes,
 * only backfiring once you go all the way to 1.
 *
 * On Vulkan/Honeykrisp, more lanes helps at small batch and hurts at large
 * batch, on *every* GPU tested -- but where the crossover falls depends on
 * the GPU's actual core count, not just the batch size. More lanes means a
 * shorter serial per-thread chain (good when a dispatch is latency-bound:
 * few enough workgroups, relative to the GPU's core count, that there's
 * nothing else to hide that latency behind) but also more total shuffle-
 * tree work (bad once there are already enough independent workgroups to
 * keep every core busy, since the extra instructions just add up). An M1
 * Ultra (64 cores) at batch 1 has 8x fewer workgroups per core than an
 * 8-core M1 does at the same batch, so it needs more lanes at a batch size
 * where the 8-core M1 already has enough to switch to fewer: 8 lanes beat 4
 * by ~18% on the Ultra at batch 1 (157.8ms -> 130.1ms) while only barely
 * regressing batch 64 (+2%); the same change on the 8-core M1 helps batch
 * 1-2 (~10%) but measurably hurts batch >= 4 (+5-6%).
 *
 * Because the crossover point scales with core count, L_REDUCE_LANES is not
 * a compile-time constant at all: k_bucket_reduce.comp declares it as a
 * specialization constant, buildPipelines (vkzg.cpp) creates one
 * VkPipeline per lane count from the same SPIR-V module, and recordReduce
 * picks a pipeline per dispatch from the batch size and the GPU's real
 * core/cluster count -- queried live at prover-creation time (see
 * gpu_topology.h/.cpp), not guessed from a device-name string. That keeps
 * this correct on any core count a future GPU (Apple or otherwise) reports,
 * without a per-device table to maintain; unknown/unqueryable topology
 * (any non-Asahi Vulkan driver right now) falls back to the fixed default
 * below, both lanes settings having already been validated safe there.
 */
#define L_LOAD_CLASSES 64 /* counting-sort bins for the bucket load ordering */

#define L_REDUCE_LANES 4 /* default lane count when GPU topology can't be queried */

#endif /* VKZG_LAYOUT_DEFS_H */
