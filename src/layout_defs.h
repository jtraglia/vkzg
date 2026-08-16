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
 * pipeline barrier in vulkan_prover.cpp) costs more than this algorithmic
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
 *
 * Separately from L_REDUCE_LANES: k_bucket_reduce.comp's workgroup size
 * (L_REDUCE_OUTPUTS_PER_TG * L_REDUCE_LANES threads) went from 128 down to
 * 32 -- the smallest that still holds a full subgroup, since a team's
 * shuffles must stay inside one 32-wide subgroup -- after an M1 Ultra
 * measured this stage 2x *slower* than the 8-core M1 at batch 1 despite
 * being the same GPU family with 8x the cores. The cause wasn't thread
 * count or lane count, it was *workgroup* count: at batch 1 there were only
 * 4 workgroups total (count=128 outputs / 32 outputs-per-workgroup at the
 * old size), and a GPU with many more independent compute clusters than
 * the one this was tuned on can't spread 4 workgroups across itself no
 * matter how fast each cluster is -- most of the chip idles regardless of
 * core count. Quartering the workgroup size quadruples the workgroup count
 * for the same output count, which is what actually lets a bigger GPU help.
 * Confirmed harmless on the 8-core M1 across the whole batch range (a small
 * *improvement*, not a tradeoff) before assuming it would also help the
 * Ultra; k_ladder.comp's workgroup size was dropped 128 -> 32 for the same
 * reason and with the same on-the-8-core-M1 validation.
 */
#define L_LOAD_CLASSES 64 /* counting-sort bins for the bucket load ordering */

#define L_REDUCE_LANES 4
#define L_LOG_REDUCE_PER_LANE 5
#define L_REDUCE_OUTPUTS_PER_TG 8 /* 32 threads / L_REDUCE_LANES */

#endif /* VULKAN_PROVER_LAYOUT_DEFS_H */
