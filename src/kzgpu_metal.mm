// Metal host layer and public API implementation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/kzgpu.h"
#include "cpu/bls12_381.h"
#include "cpu/device_convert.h"
#include "cpu/cpu_msm.h"
#include "cpu/setup.h"
#include "kzgpu_internal.h"
#include "shaders/shader_source.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

using namespace kzgpu;

namespace {

// Mirrors NttParams in kernels.metal.
struct NttParams {
    uint32_t n;
    uint32_t log_n;
    uint32_t in_stride_t;
    uint32_t in_stride_i;
    uint32_t out_stride_t;
    uint32_t out_stride_i;
    uint32_t root_stride;
    uint32_t twiddle_stride;
    uint32_t full_n;
    uint32_t in_batch;
    uint32_t out_batch;
    uint32_t scale;
    uint32_t scale_val[kFrLimbs];
};

double nowMs() { return (double)clock_gettime_nsec_np(CLOCK_MONOTONIC) / 1.0e6; }

uint32_t ilog2(uint32_t n) {
    uint32_t r = 0;
    while ((1u << r) < n) r++;
    return r;
}

} // namespace

struct kzgpu_prover {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> psoBlobToFr = nil;
    id<MTLComputePipelineState> psoNtt = nil;
    id<MTLComputePipelineState> psoSerializeCells = nil;
    id<MTLComputePipelineState> psoBuildCirculant = nil;
    id<MTLComputePipelineState> psoPhaseASort = nil;
    id<MTLComputePipelineState> psoPhaseA = nil;
    id<MTLComputePipelineState> psoPhaseB = nil;
    id<MTLComputePipelineState> psoReduce = nil;

    // Setup-derived, immutable.
    id<MTLBuffer> bufTable = nil;
    id<MTLBuffer> bufRootsFwd = nil;
    id<MTLBuffer> bufRootsInv = nil;
    id<MTLBuffer> bufKernelItems = nil;
    id<MTLBuffer> bufKernelOffsets = nil;
    id<MTLBuffer> bufKernelPerm = nil;

    // Per-batch working set.
    id<MTLBuffer> bufBlob = nil;
    id<MTLBuffer> bufLagrange = nil;
    id<MTLBuffer> bufWorkA = nil;   // 8192 Fr/blob
    id<MTLBuffer> bufPolyExt = nil; // 8192 Fr/blob, upper half kept at zero
    id<MTLBuffer> bufEvals = nil;
    id<MTLBuffer> bufCells = nil;
    id<MTLBuffer> bufCoeffs = nil;
    id<MTLBuffer> bufBuckets = nil;
    id<MTLBuffer> bufItems = nil;  // bucket-sorted phase A digit lists
    id<MTLBuffer> bufStarts = nil;
    id<MTLBuffer> bufPerm = nil;   // per-output lane -> bucket ordering
    id<MTLBuffer> bufPoints = nil;     // u[j], then reused for the proofs
    id<MTLBuffer> bufProofs = nil;
    id<MTLBuffer> bufLadderAff = nil;
    id<MTLBuffer> bufErr = nil;

    uint32_t maxBatch = 0;
    std::string deviceName;
    std::mutex mutex;

    uint32_t invBlob[kFrLimbs] = {0};
    uint32_t invExtBlob[kFrLimbs] = {0};

    // Concurrent CPU assist. `splitA`/`splitB` are the number of outputs the
    // GPU takes; the rest go to the pool. Both are re-balanced from measured
    // per-output cost so the two finish together.
    std::unique_ptr<ThreadPool> pool;
    std::vector<uint32_t> cpuStaging; // CPU results, copied in once the GPU is done
    int splitA = kCirculantSize;
    int splitB = kCirculantSize;
    double gpuCostA = 0, cpuCostA = 0, gpuCostB = 0, cpuCostB = 0;
};

namespace {
// Re-balance so that split*gpuCost == (128-split)*cpuCost, damped by an EWMA.
//
// The split must stay a multiple of L_REDUCE_OUTPUTS_PER_TG: the reduction
// dispatches split/L_REDUCE_OUTPUTS_PER_TG threadgroups, so any remainder would
// silently leave those outputs unreduced.  That granularity (16 of 128) costs
// at most a few percent of balance.
void rebalance(int &split, double gpuTime, double cpuTime, int gpuCount, int cpuCount,
               double &gpuCost, double &cpuCost) {
    if (gpuCount > 0 && gpuTime > 0) {
        const double c = gpuTime / gpuCount;
        gpuCost = gpuCost > 0 ? 0.7 * gpuCost + 0.3 * c : c;
    }
    if (cpuCount > 0 && cpuTime > 0) {
        const double c = cpuTime / cpuCount;
        cpuCost = cpuCost > 0 ? 0.7 * cpuCost + 0.3 * c : c;
    }
    if (gpuCost <= 0 || cpuCost <= 0) return;
    const double ideal = kCirculantSize * cpuCost / (gpuCost + cpuCost);
    const int quantum = L_REDUCE_OUTPUTS_PER_TG;
    int next = ((int)(ideal / quantum + 0.5)) * quantum;
    if (next < quantum) next = quantum;
    if (next > kCirculantSize) next = kCirculantSize;
    split = next;
}
} // namespace

// --------------------------------------------------------------------- utils

const char *kzgpu_error_string(kzgpu_result r) {
    switch (r) {
        case KZGPU_OK: return "ok";
        case KZGPU_ERR_BADARGS: return "invalid argument";
        case KZGPU_ERR_MALLOC: return "allocation failed";
        case KZGPU_ERR_IO: return "i/o error";
        case KZGPU_ERR_SETUP: return "malformed trusted setup";
        case KZGPU_ERR_GPU: return "gpu error";
        case KZGPU_ERR_INVALID_BLOB: return "blob contains a non-canonical field element";
    }
    return "unknown error";
}

void kzgpu_options_default(kzgpu_options *opts) {
    if (!opts) return;
    opts->table_cache_path = nullptr;
    opts->validate_setup = 0;
    opts->cpu_assist_threads = 0;
    opts->max_batch_size = 0;
}

const char *kzgpu_device_name(const kzgpu_prover *p) {
    return p ? p->deviceName.c_str() : "";
}

namespace {

id<MTLBuffer> makeBuffer(id<MTLDevice> dev, size_t bytes) {
    if (bytes == 0) bytes = 16;
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

id<MTLBuffer> makeBufferFrom(id<MTLDevice> dev, const void *src, size_t bytes) {
    if (bytes == 0) bytes = 16;
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (b && src) memcpy(b.contents, src, bytes);
    return b;
}

bool allocateWorkingSet(kzgpu_prover *p, uint32_t batch) {
    const size_t B = batch;
    p->bufBlob = makeBuffer(p->device, B * KZGPU_BYTES_PER_BLOB);
    p->bufLagrange = makeBuffer(p->device, B * kFieldElementsPerBlob * kFrLimbs * 4);
    p->bufWorkA = makeBuffer(p->device, B * kFieldElementsPerExtBlob * kFrLimbs * 4);
    p->bufPolyExt = makeBuffer(p->device, B * kFieldElementsPerExtBlob * kFrLimbs * 4);
    p->bufEvals = makeBuffer(p->device, B * kFieldElementsPerExtBlob * kFrLimbs * 4);
    p->bufCells = makeBuffer(p->device, B * kFieldElementsPerExtBlob * 32);
    p->bufCoeffs = makeBuffer(p->device, B * kCirculantSize * kPhaseATerms * kFrLimbs * 4);
    p->bufBuckets = makeBuffer(p->device, B * kCirculantSize * kNumBuckets * kJacobianWords * 4);
    p->bufItems = makeBuffer(p->device, B * kCirculantSize * kPhaseAItems * 2);
    p->bufStarts = makeBuffer(p->device, B * kCirculantSize * (kNumBuckets + 1) * 4);
    p->bufPerm = makeBuffer(p->device, B * kCirculantSize * kNumBuckets * 4);
    p->bufPoints = makeBuffer(p->device, B * kCirculantSize * kJacobianWords * 4);
    p->bufProofs = makeBuffer(p->device, B * kCirculantSize * kJacobianWords * 4);
    p->bufLadderAff =
        makeBuffer(p->device, B * kCirculantSize * kLadderPositions * kAffineWords * 4);
    p->bufErr = makeBuffer(p->device, 4);

    if (!p->bufBlob || !p->bufLagrange || !p->bufWorkA || !p->bufPolyExt || !p->bufEvals ||
        !p->bufCells || !p->bufCoeffs || !p->bufBuckets || !p->bufItems || !p->bufStarts || !p->bufPerm ||
        !p->bufPoints || !p->bufProofs ||
        !p->bufLadderAff || !p->bufErr) {
        return false;
    }
    // The forward transform reads a zero-padded polynomial; the upper half is
    // never written, so zero it once here.
    memset(p->bufPolyExt.contents, 0, p->bufPolyExt.length);
    p->cpuStaging.assign((size_t)batch * kCirculantSize * kJacobianWords, 0);
    p->maxBatch = batch;
    return true;
}

kzgpu_result buildPipelines(kzgpu_prover *p) {
    NSError *err = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    // Integer-only code, but be explicit that we want exact semantics.
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
    }
    NSString *src = [NSString stringWithUTF8String:kzgpu::kShaderSource];
    id<MTLLibrary> lib = [p->device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        NSLog(@"kzgpu: shader compilation failed: %@", err.localizedDescription);
        return KZGPU_ERR_GPU;
    }

    struct {
        const char *name;
        id<MTLComputePipelineState> __strong *slot;
    } kernels[] = {
        {"k_blob_to_fr", &p->psoBlobToFr},
        {"k_ntt_pass", &p->psoNtt},
        {"k_serialize_cells", &p->psoSerializeCells},
        {"k_build_circulant", &p->psoBuildCirculant},
        {"k_phase_a_sort", &p->psoPhaseASort},
        {"k_phase_a", &p->psoPhaseA},
        {"k_phase_b", &p->psoPhaseB},
        {"k_bucket_reduce", &p->psoReduce},
    };
    for (auto &k : kernels) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:k.name]];
        if (!fn) {
            NSLog(@"kzgpu: missing kernel %s", k.name);
            return KZGPU_ERR_GPU;
        }
        id<MTLComputePipelineState> pso = [p->device newComputePipelineStateWithFunction:fn
                                                                                  error:&err];
        if (!pso) {
            NSLog(@"kzgpu: pipeline for %s failed: %@", k.name, err.localizedDescription);
            return KZGPU_ERR_GPU;
        }
        *k.slot = pso;
    }
    return KZGPU_OK;
}

kzgpu_result createProver(kzgpu_prover **out, SetupTables &tables, const kzgpu_options *opts) {
    auto *p = new kzgpu_prover();
    p->device = MTLCreateSystemDefaultDevice();
    if (!p->device) {
        delete p;
        return KZGPU_ERR_GPU;
    }
    p->queue = [p->device newCommandQueue];
    p->deviceName = [[p->device name] UTF8String];

    kzgpu_result rc = buildPipelines(p);
    if (rc != KZGPU_OK) {
        delete p;
        return rc;
    }

    p->bufTable = makeBufferFrom(p->device, tables.position_table.data(),
                                 tables.position_table.size() * 4);
    p->bufRootsFwd = makeBufferFrom(p->device, tables.roots_fwd.data(), tables.roots_fwd.size() * 4);
    p->bufRootsInv = makeBufferFrom(p->device, tables.roots_inv.data(), tables.roots_inv.size() * 4);
    p->bufKernelItems =
        makeBufferFrom(p->device, tables.kernel_items.data(), tables.kernel_items.size() * 4);
    p->bufKernelOffsets =
        makeBufferFrom(p->device, tables.kernel_offsets.data(), tables.kernel_offsets.size() * 4);
    p->bufKernelPerm =
        makeBufferFrom(p->device, tables.kernel_perm.data(), tables.kernel_perm.size() * 4);
    memcpy(p->invBlob, tables.inv_blob, sizeof(p->invBlob));
    memcpy(p->invExtBlob, tables.inv_ext_blob, sizeof(p->invExtBlob));

    int32_t cpuThreads = opts ? opts->cpu_assist_threads : 0;
    if (cpuThreads == 0) {
        unsigned hw = std::thread::hardware_concurrency();
        cpuThreads = hw > 1 ? (int32_t)(hw - 1) : 1;
    }
    if (cpuThreads > 0) {
        p->pool.reset(new ThreadPool((unsigned)cpuThreads));
        // Start at an even split and let the controller find the balance.
        p->splitA = kCirculantSize / 2;
        p->splitB = kCirculantSize / 2;
    }

    uint32_t batch = opts && opts->max_batch_size ? opts->max_batch_size : 4;
    if (!allocateWorkingSet(p, batch)) {
        delete p;
        return KZGPU_ERR_MALLOC;
    }
    *out = p;
    return KZGPU_OK;
}

} // namespace

// --------------------------------------------------------------- constructors

kzgpu_result kzgpu_prover_new(kzgpu_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const kzgpu_options *opts) {
    if (!out || !g1_monomial_bytes) return KZGPU_ERR_BADARGS;
    *out = nullptr;

    SetupTables tables;
    const bool validate = opts && opts->validate_setup;
    const char *cache = opts ? opts->table_cache_path : nullptr;

    // The cache records the digest of the setup it was derived from; hashing
    // the input is far cheaper than building the tables just to learn it.
    bool have = false;
    if (cache) {
        const uint64_t digest = compute_setup_digest(g1_monomial_bytes, g1_monomial_len);
        have = load_table_cache(cache, digest, tables) == KZGPU_OK;
    }
    if (!have) {
        kzgpu_result rc = build_setup_tables(g1_monomial_bytes, g1_monomial_len, validate, tables);
        if (rc != KZGPU_OK) return rc;
        if (cache) save_table_cache(cache, tables);
    }

    @autoreleasepool {
        return createProver(out, tables, opts);
    }
}

kzgpu_result kzgpu_prover_new_from_file(kzgpu_prover **out, const char *trusted_setup_path,
                                        const kzgpu_options *opts) {
    if (!out || !trusted_setup_path) return KZGPU_ERR_BADARGS;
    std::vector<uint8_t> g1;
    kzgpu_result rc = read_trusted_setup_file(trusted_setup_path, g1);
    if (rc != KZGPU_OK) return rc;
    return kzgpu_prover_new(out, g1.data(), g1.size(), opts);
}

void kzgpu_prover_free(kzgpu_prover *p) { delete p; }

// -------------------------------------------------------------------- compute

namespace {

void encodeNtt(id<MTLComputeCommandEncoder> enc, kzgpu_prover *p, id<MTLBuffer> out,
               id<MTLBuffer> in, id<MTLBuffer> roots, const NttParams &params, uint32_t count,
               uint32_t batch) {
    [enc setComputePipelineState:p->psoNtt];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:in offset:0 atIndex:1];
    [enc setBuffer:roots offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(count, batch, 1)
        threadsPerThreadgroup:MTLSizeMake(params.n, 1, 1)];
}

// Configures one pass of an N = N1 * N2 four-step transform.
NttParams nttPass1(uint32_t N, uint32_t N1, uint32_t N2, uint32_t inBatch, uint32_t outBatch) {
    NttParams q{};
    q.n = N1;
    q.log_n = ilog2(N1);
    q.in_stride_t = 1;
    q.in_stride_i = N2;
    q.out_stride_t = 1;
    q.out_stride_i = N2;
    q.root_stride = kFieldElementsPerExtBlob / N1;
    q.twiddle_stride = kFieldElementsPerExtBlob / N;
    q.full_n = N;
    q.in_batch = inBatch;
    q.out_batch = outBatch;
    q.scale = 0;
    return q;
}

NttParams nttPass2(uint32_t N, uint32_t N1, uint32_t N2, uint32_t inBatch, uint32_t outBatch) {
    NttParams q{};
    q.n = N2;
    q.log_n = ilog2(N2);
    q.in_stride_t = N2;
    q.in_stride_i = 1;
    q.out_stride_t = 1;
    q.out_stride_i = N1;
    q.root_stride = kFieldElementsPerExtBlob / N2;
    q.twiddle_stride = 0;
    q.full_n = N;
    q.in_batch = inBatch;
    q.out_batch = outBatch;
    q.scale = 0;
    return q;
}

// Copies the CPU-computed outputs [split, 128) into the device buffer.  The
// CPU writes to its own staging array rather than straight into the Metal
// buffer, because the GPU is concurrently writing the [0, split) range of the
// same allocation and Metal makes no promise about that overlap.
void copyStagedOutputs(kzgpu_prover *p, id<MTLBuffer> dst, uint32_t batch, int split) {
    const int n = kCirculantSize - split;
    if (n <= 0) return;
    uint32_t *d = (uint32_t *)dst.contents;
    for (uint32_t b = 0; b < batch; b++) {
        const size_t off = ((size_t)b * kCirculantSize + (size_t)split) * kJacobianWords;
        memcpy(d + off, p->cpuStaging.data() + off, (size_t)n * kJacobianWords * 4);
    }
}

kzgpu_result computeBatch(kzgpu_prover *p, uint8_t *cells, uint8_t *proofs, const uint8_t *blobs,
                          uint32_t batch) {
    @autoreleasepool {
        memcpy(p->bufBlob.contents, blobs, (size_t)batch * KZGPU_BYTES_PER_BLOB);
        *(uint32_t *)p->bufErr.contents = 0;

        // ---- pass 1: scalar work, up to and including the circulant columns.
        id<MTLCommandBuffer> cb1 = [p->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb1 computeCommandEncoder];

        [enc setComputePipelineState:p->psoBlobToFr];
        [enc setBuffer:p->bufLagrange offset:0 atIndex:0];
        [enc setBuffer:p->bufBlob offset:0 atIndex:1];
        [enc setBuffer:p->bufErr offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(kFieldElementsPerBlob, batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        {
            NttParams q = nttPass1(4096, 64, 64, kFieldElementsPerBlob, kFieldElementsPerExtBlob);
            encodeNtt(enc, p, p->bufWorkA, p->bufLagrange, p->bufRootsInv, q, 64, batch);
            NttParams r = nttPass2(4096, 64, 64, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
            r.scale = 1;
            memcpy(r.scale_val, p->invBlob, sizeof(r.scale_val));
            encodeNtt(enc, p, p->bufPolyExt, p->bufWorkA, p->bufRootsInv, r, 64, batch);
        }

        if (cells) {
            NttParams q = nttPass1(8192, 64, 128, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
            encodeNtt(enc, p, p->bufWorkA, p->bufPolyExt, p->bufRootsFwd, q, 128, batch);
            NttParams r = nttPass2(8192, 64, 128, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
            encodeNtt(enc, p, p->bufEvals, p->bufWorkA, p->bufRootsFwd, r, 64, batch);

            [enc setComputePipelineState:p->psoSerializeCells];
            [enc setBuffer:p->bufCells offset:0 atIndex:0];
            [enc setBuffer:p->bufEvals offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(kFieldElementsPerExtBlob, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        if (proofs) {
            [enc setComputePipelineState:p->psoBuildCirculant];
            [enc setBuffer:p->bufCoeffs offset:0 atIndex:0];
            [enc setBuffer:p->bufPolyExt offset:0 atIndex:1];
            [enc setBuffer:p->bufRootsFwd offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(kPhaseATerms, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kCirculantSize, 1, 1)];
        }

        [enc endEncoding];
        [cb1 commit];
        [cb1 waitUntilCompleted];
        if (cb1.error) {
            NSLog(@"kzgpu: command buffer 1 failed: %@", cb1.error);
            return KZGPU_ERR_GPU;
        }
        if (*(uint32_t *)p->bufErr.contents != 0) return KZGPU_ERR_INVALID_BLOB;

        if (cells) {
            memcpy(cells, p->bufCells.contents, (size_t)batch * kFieldElementsPerExtBlob * 32);
        }
        if (!proofs) return KZGPU_OK;

        const int splitA = p->pool ? p->splitA : kCirculantSize;
        const int splitB = p->pool ? p->splitB : kCirculantSize;

        // ---- pass 2: phase A.  The GPU takes outputs [0, splitA); the CPU
        // pool takes the rest and runs while the command buffer is in flight.
        id<MTLCommandBuffer> cb2 = [p->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> e = [cb2 computeCommandEncoder];
            [e setComputePipelineState:p->psoPhaseASort];
            [e setBuffer:p->bufItems offset:0 atIndex:0];
            [e setBuffer:p->bufStarts offset:0 atIndex:1];
            [e setBuffer:p->bufPerm offset:0 atIndex:2];
            [e setBuffer:p->bufCoeffs offset:0 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(splitA, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kPhaseATerms, 1, 1)];

            [e setComputePipelineState:p->psoPhaseA];
            [e setBuffer:p->bufBuckets offset:0 atIndex:0];
            [e setBuffer:p->bufItems offset:0 atIndex:1];
            [e setBuffer:p->bufStarts offset:0 atIndex:2];
            [e setBuffer:p->bufPerm offset:0 atIndex:3];
            [e setBuffer:p->bufTable offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(splitA, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kNumBuckets, 1, 1)];

            [e setComputePipelineState:p->psoReduce];
            [e setBuffer:p->bufPoints offset:0 atIndex:0];
            [e setBuffer:p->bufBuckets offset:0 atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(splitA / L_REDUCE_OUTPUTS_PER_TG, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e endEncoding];
        }
        [cb2 commit];

        double cpuTimeA = 0;
        if (p->pool && splitA < kCirculantSize) {
            const double t0 = nowMs();
            cpu_phase_a(*p->pool, p->cpuStaging.data(), (const uint32_t *)p->bufCoeffs.contents,
                        (const uint32_t *)p->bufTable.contents, batch, splitA, kCirculantSize);
            cpuTimeA = nowMs() - t0;
        }
        const double gpuWaitA0 = nowMs();
        [cb2 waitUntilCompleted];
        const double gpuTimeA = (cb2.GPUEndTime - cb2.GPUStartTime) * 1e3;
        (void)gpuWaitA0;
        if (cb2.error) {
            NSLog(@"kzgpu: command buffer 2 failed: %@", cb2.error);
            return KZGPU_ERR_GPU;
        }
        if (p->pool && splitA < kCirculantSize) {
            copyStagedOutputs(p, p->bufPoints, batch, splitA);
            rebalance(p->splitA, gpuTimeA, cpuTimeA, splitA, kCirculantSize - splitA, p->gpuCostA,
                      p->cpuCostA);
        }

        // Phase B needs 2^(8d) * u[j] in affine form.  Both the doubling chain
        // and the inversion run on the host: 248 sequential doublings over 128
        // points is latency-bound on a GPU whose single-thread Fp multiply
        // costs 3.4us (measured 19.6ms there, well under 1ms here).
        build_ladder_affine((uint32_t *)p->bufLadderAff.contents,
                            (const uint32_t *)p->bufPoints.contents,
                            (size_t)batch * kCirculantSize, 0);

        // ---- pass 3: phase B, split the same way.
        id<MTLCommandBuffer> cb3 = [p->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> e = [cb3 computeCommandEncoder];
            [e setComputePipelineState:p->psoPhaseB];
            [e setBuffer:p->bufBuckets offset:0 atIndex:0];
            [e setBuffer:p->bufLadderAff offset:0 atIndex:1];
            [e setBuffer:p->bufKernelItems offset:0 atIndex:2];
            [e setBuffer:p->bufKernelOffsets offset:0 atIndex:3];
            [e setBuffer:p->bufKernelPerm offset:0 atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake(splitB, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kNumBuckets, 1, 1)];

            [e setComputePipelineState:p->psoReduce];
            [e setBuffer:p->bufProofs offset:0 atIndex:0];
            [e setBuffer:p->bufBuckets offset:0 atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(splitB / L_REDUCE_OUTPUTS_PER_TG, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [e endEncoding];
        }
        [cb3 commit];

        double cpuTimeB = 0;
        if (p->pool && splitB < kCirculantSize) {
            const double t0 = nowMs();
            cpu_phase_b(*p->pool, p->cpuStaging.data(),
                        (const uint32_t *)p->bufLadderAff.contents,
                        (const uint32_t *)p->bufKernelItems.contents,
                        (const uint32_t *)p->bufKernelOffsets.contents, batch, splitB,
                        kCirculantSize);
            cpuTimeB = nowMs() - t0;
        }
        [cb3 waitUntilCompleted];
        const double gpuTimeB = (cb3.GPUEndTime - cb3.GPUStartTime) * 1e3;
        if (cb3.error) {
            NSLog(@"kzgpu: command buffer 3 failed: %@", cb3.error);
            return KZGPU_ERR_GPU;
        }
        if (p->pool && splitB < kCirculantSize) {
            copyStagedOutputs(p, p->bufProofs, batch, splitB);
            rebalance(p->splitB, gpuTimeB, cpuTimeB, splitB, kCirculantSize - splitB, p->gpuCostB,
                      p->cpuCostB);
        }

        finalize_proofs(proofs, (const uint32_t *)p->bufProofs.contents, batch, 0);
        return KZGPU_OK;
    }
}

} // namespace

kzgpu_result kzgpu_compute_cells_and_proofs(kzgpu_prover *p, uint8_t *cells, uint8_t *proofs,
                                            const uint8_t *blob) {
    return kzgpu_compute_cells_and_proofs_batch(p, cells, proofs, blob, 1);
}

kzgpu_result kzgpu_compute_cells_and_proofs_batch(kzgpu_prover *p, uint8_t *cells, uint8_t *proofs,
                                                  const uint8_t *blobs, size_t num_blobs) {
    if (!p || !blobs) return KZGPU_ERR_BADARGS;
    if (!cells && !proofs) return KZGPU_ERR_BADARGS;
    if (num_blobs == 0) return KZGPU_OK;

    std::lock_guard<std::mutex> lock(p->mutex);
    const size_t cellBytes = (size_t)kCirculantSize * KZGPU_BYTES_PER_CELL;
    const size_t proofBytes = (size_t)kCirculantSize * kBytesPerProof;

    for (size_t done = 0; done < num_blobs;) {
        const uint32_t batch = (uint32_t)std::min<size_t>(p->maxBatch, num_blobs - done);
        kzgpu_result rc = computeBatch(p, cells ? cells + done * cellBytes : nullptr,
                                       proofs ? proofs + done * proofBytes : nullptr,
                                       blobs + done * KZGPU_BYTES_PER_BLOB, batch);
        if (rc != KZGPU_OK) return rc;
        done += batch;
    }
    return KZGPU_OK;
}

// ------------------------------------------------------------------ profiling
//
// Development helper: same dispatch sequence, but every stage gets its own
// command buffer so we can attribute time.  See src/kzgpu_profile.h.
#include "kzgpu_profile.h"

namespace kzgpu {

namespace {
// Runs one encoding closure in its own command buffer and returns its duration.
template <typename Fn>
double timedPass(kzgpu_prover *p, Fn fn) {
    const double a = nowMs();
    id<MTLCommandBuffer> cb = [p->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    fn(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return nowMs() - a;
}
} // namespace

kzgpu_result profile_batch(kzgpu_prover *p, unsigned char *cells, unsigned char *proofs,
                           const unsigned char *blobs, unsigned batch, StageTimes &out) {
    @autoreleasepool {
        const double t_begin = nowMs();
        memcpy(p->bufBlob.contents, blobs, (size_t)batch * KZGPU_BYTES_PER_BLOB);
        *(uint32_t *)p->bufErr.contents = 0;

        out.blob_to_fr = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
            [enc setComputePipelineState:p->psoBlobToFr];
            [enc setBuffer:p->bufLagrange offset:0 atIndex:0];
            [enc setBuffer:p->bufBlob offset:0 atIndex:1];
            [enc setBuffer:p->bufErr offset:0 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(kFieldElementsPerBlob, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        });

        out.ntt_inverse = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
            NttParams q = nttPass1(4096, 64, 64, kFieldElementsPerBlob, kFieldElementsPerExtBlob);
            encodeNtt(enc, p, p->bufWorkA, p->bufLagrange, p->bufRootsInv, q, 64, batch);
            NttParams r = nttPass2(4096, 64, 64, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
            r.scale = 1;
            memcpy(r.scale_val, p->invBlob, sizeof(r.scale_val));
            encodeNtt(enc, p, p->bufPolyExt, p->bufWorkA, p->bufRootsInv, r, 64, batch);
        });

        if (cells) {
            out.ntt_forward = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                NttParams q =
                    nttPass1(8192, 64, 128, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
                encodeNtt(enc, p, p->bufWorkA, p->bufPolyExt, p->bufRootsFwd, q, 128, batch);
                NttParams r =
                    nttPass2(8192, 64, 128, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
                encodeNtt(enc, p, p->bufEvals, p->bufWorkA, p->bufRootsFwd, r, 64, batch);
            });
            out.serialize_cells = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoSerializeCells];
                [enc setBuffer:p->bufCells offset:0 atIndex:0];
                [enc setBuffer:p->bufEvals offset:0 atIndex:1];
                [enc dispatchThreads:MTLSizeMake(kFieldElementsPerExtBlob, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            });
            memcpy(cells, p->bufCells.contents, (size_t)batch * kFieldElementsPerExtBlob * 32);
        }

        if (proofs) {
            out.build_circulant = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoBuildCirculant];
                [enc setBuffer:p->bufCoeffs offset:0 atIndex:0];
                [enc setBuffer:p->bufPolyExt offset:0 atIndex:1];
                [enc setBuffer:p->bufRootsFwd offset:0 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(kPhaseATerms, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(kCirculantSize, 1, 1)];
            });
            out.build_circulant += timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoPhaseASort];
                [enc setBuffer:p->bufItems offset:0 atIndex:0];
                [enc setBuffer:p->bufStarts offset:0 atIndex:1];
                [enc setBuffer:p->bufPerm offset:0 atIndex:2];
                [enc setBuffer:p->bufCoeffs offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(kPhaseATerms, 1, 1)];
            });
            out.phase_a = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoPhaseA];
                [enc setBuffer:p->bufBuckets offset:0 atIndex:0];
                [enc setBuffer:p->bufItems offset:0 atIndex:1];
                [enc setBuffer:p->bufStarts offset:0 atIndex:2];
                [enc setBuffer:p->bufPerm offset:0 atIndex:3];
                [enc setBuffer:p->bufTable offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(kNumBuckets, 1, 1)];
            });
            out.reduce_a = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoReduce];
                [enc setBuffer:p->bufPoints offset:0 atIndex:0];
                [enc setBuffer:p->bufBuckets offset:0 atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize / L_REDUCE_OUTPUTS_PER_TG, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            });
            double a = nowMs();
            build_ladder_affine((uint32_t *)p->bufLadderAff.contents,
                                (const uint32_t *)p->bufPoints.contents,
                                (size_t)batch * kCirculantSize, 0);
            out.ladder = nowMs() - a;

            out.phase_b = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoPhaseB];
                [enc setBuffer:p->bufBuckets offset:0 atIndex:0];
                [enc setBuffer:p->bufLadderAff offset:0 atIndex:1];
                [enc setBuffer:p->bufKernelItems offset:0 atIndex:2];
                [enc setBuffer:p->bufKernelOffsets offset:0 atIndex:3];
                [enc setBuffer:p->bufKernelPerm offset:0 atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(kNumBuckets, 1, 1)];
            });

            out.reduce_b = timedPass(p, [&](id<MTLComputeCommandEncoder> enc) {
                [enc setComputePipelineState:p->psoReduce];
                [enc setBuffer:p->bufProofs offset:0 atIndex:0];
                [enc setBuffer:p->bufBuckets offset:0 atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize / L_REDUCE_OUTPUTS_PER_TG, batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            });

            a = nowMs();
            finalize_proofs(proofs, (const uint32_t *)p->bufProofs.contents, batch, 0);
            out.finalize = nowMs() - a;
        }
        out.total = nowMs() - t_begin;
        return KZGPU_OK;
    }
}

} // namespace kzgpu
