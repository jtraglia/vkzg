// Host-side MSM workers that share the GPU's data layout.
//
// Why a CPU path exists in a GPU library: on Apple silicon the integer
// multiplier is the scarce resource, and it is scarcer on the GPU than on the
// CPU.  A 32x32->64 multiply issues about every 8.9 cycles per GPU lane
// (measured), which puts the whole 8-core GPU at ~392M Fp multiplies/s, while
// one CPU core manages ~23M and eight of them land in the same ballpark.  So
// the fastest configuration is not GPU-instead-of-CPU but GPU-and-CPU: both
// phases split their 128 independent outputs between the two, and the split is
// re-balanced from measured throughput.
//
// These functions read and write the same device-layout buffers as the
// shaders, so no conversion is needed at the boundary.
#pragma once

#include "../kzgpu_internal.h"

#include <cstddef>
#include <cstdint>

namespace kzgpu {

// A small persistent pool; spawning threads per call would cost more than the
// work for a single blob.
class ThreadPool {
public:
    explicit ThreadPool(unsigned threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    unsigned size() const { return nthreads_; }
    // Runs fn(i) for i in [0, n), blocking until all have finished.
    void parallel_for(size_t n, void (*fn)(void *, size_t), void *ctx);

private:
    struct Impl;
    Impl *impl_;
    unsigned nthreads_;
};

// Phase A for outputs [j0, j1) of every blob in the batch.
//   out      : Jacobian points, [blob][output], device layout
//   coeffs   : Fr scalars, [blob][output][term]
//   table    : affine 2^(8d) * P[j][i]
void cpu_phase_a(ThreadPool &pool, uint32_t *out, const uint32_t *coeffs, const uint32_t *table,
                 size_t num_blobs, int j0, int j1);

// Phase B for outputs [a0, a1) of every blob in the batch.
void cpu_phase_b(ThreadPool &pool, uint32_t *out, const uint32_t *ladder_affine,
                 const uint32_t *items, const uint32_t *offsets, size_t num_blobs, int a0, int a1);

} // namespace kzgpu
