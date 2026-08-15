// Metal host layer and public API implementation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/metal_prover.h"
#include "cpu/bls12_381.h"
#include "cpu/setup.h"
#include "internal.h"
#include "setup_data.h"
#include "profile.h"
#include "shaders/shader_source.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

using namespace mp;

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

struct mp_prover {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> psoBlobToFr = nil;
    id<MTLComputePipelineState> psoNtt = nil;
    id<MTLComputePipelineState> psoSerializeCells = nil;
    id<MTLComputePipelineState> psoBuildCirculant = nil;
    id<MTLComputePipelineState> psoPhaseASort = nil;
    id<MTLComputePipelineState> psoPhaseA = nil;
    id<MTLComputePipelineState> psoPhaseB = nil;
    id<MTLComputePipelineState> psoLadder = nil;
    id<MTLComputePipelineState> psoNormalize = nil;
    id<MTLComputePipelineState> psoCompress = nil;
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
    id<MTLBuffer> bufLadderJac = nil;
    id<MTLBuffer> bufLadderAff = nil;
    id<MTLBuffer> bufProofsAff = nil;
    id<MTLBuffer> bufProofBytes = nil;
    id<MTLBuffer> bufNormScratch = nil;
    id<MTLBuffer> bufErr = nil;

    uint32_t maxBatch = 0;
    std::string deviceName;
    std::mutex mutex;

    uint32_t invBlob[kFrLimbs] = {0};

    StageTimes lastStage; // filled by computeBatch, read by profile_batch
};


// --------------------------------------------------------------------- utils

const char *mp_error_string(mp_result r) {
    switch (r) {
        case MP_OK: return "ok";
        case MP_ERR_BADARGS: return "invalid argument";
        case MP_ERR_MALLOC: return "allocation failed";
        case MP_ERR_IO: return "i/o error";
        case MP_ERR_SETUP: return "malformed trusted setup";
        case MP_ERR_GPU: return "gpu error";
        case MP_ERR_INVALID_BLOB: return "blob contains a non-canonical field element";
    }
    return "unknown error";
}

void mp_options_default(mp_options *opts) {
    if (!opts) return;
    opts->table_cache_path = nullptr;
    opts->validate_setup = 0;
    opts->max_batch_size = 0;
}

const char *mp_prover_device_name(const mp_prover *p) {
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

bool allocateWorkingSet(mp_prover *p, uint32_t batch) {
    const size_t B = batch;
    p->bufBlob = makeBuffer(p->device, B * MP_BYTES_PER_BLOB);
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
    p->bufLadderJac =
        makeBuffer(p->device, B * kCirculantSize * kLadderPositions * kJacobianWords * 4);
    p->bufLadderAff =
        makeBuffer(p->device, B * kCirculantSize * kLadderPositions * kAffineWords * 4);
    p->bufProofsAff = makeBuffer(p->device, B * kCirculantSize * kAffineWords * 4);
    p->bufProofBytes = makeBuffer(p->device, B * kCirculantSize * kBytesPerProof);
    // Prefix products for the batched inversion; sized for the larger of the
    // two normalisation passes.
    p->bufNormScratch =
        makeBuffer(p->device, B * kCirculantSize * kLadderPositions * kFpLimbs * 4);
    p->bufErr = makeBuffer(p->device, 4);

    if (!p->bufBlob || !p->bufLagrange || !p->bufWorkA || !p->bufPolyExt || !p->bufEvals ||
        !p->bufCells || !p->bufCoeffs || !p->bufBuckets || !p->bufItems || !p->bufStarts || !p->bufPerm ||
        !p->bufPoints || !p->bufProofs ||
        !p->bufLadderJac || !p->bufLadderAff || !p->bufProofsAff ||
        !p->bufProofBytes || !p->bufNormScratch || !p->bufErr) {
        return false;
    }
    // The forward transform reads a zero-padded polynomial; the upper half is
    // never written, so zero it once here.
    memset(p->bufPolyExt.contents, 0, p->bufPolyExt.length);
    p->maxBatch = batch;
    return true;
}

mp_result buildPipelines(mp_prover *p) {
    NSError *err = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    // Integer-only code, but be explicit that we want exact semantics.
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
    }
    NSString *src = [NSString stringWithUTF8String:mp::kShaderSource];
    id<MTLLibrary> lib = [p->device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        NSLog(@"metal-prover: shader compilation failed: %@", err.localizedDescription);
        return MP_ERR_GPU;
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
        {"k_ladder", &p->psoLadder},
        {"k_normalize", &p->psoNormalize},
        {"k_compress_proofs", &p->psoCompress},
        {"k_bucket_reduce", &p->psoReduce},
    };
    for (auto &k : kernels) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:k.name]];
        if (!fn) {
            NSLog(@"metal-prover: missing kernel %s", k.name);
            return MP_ERR_GPU;
        }
        id<MTLComputePipelineState> pso = [p->device newComputePipelineStateWithFunction:fn
                                                                                  error:&err];
        if (!pso) {
            NSLog(@"metal-prover: pipeline for %s failed: %@", k.name, err.localizedDescription);
            return MP_ERR_GPU;
        }
        *k.slot = pso;
    }
    return MP_OK;
}

mp_result createProver(mp_prover **out, SetupTables &tables, const mp_options *opts) {
    auto *p = new mp_prover();
    p->device = MTLCreateSystemDefaultDevice();
    if (!p->device) {
        delete p;
        return MP_ERR_GPU;
    }
    p->queue = [p->device newCommandQueue];
    p->deviceName = [[p->device name] UTF8String];

    mp_result rc = buildPipelines(p);
    if (rc != MP_OK) {
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

    uint32_t batch = opts && opts->max_batch_size ? opts->max_batch_size : 4;
    if (!allocateWorkingSet(p, batch)) {
        delete p;
        return MP_ERR_MALLOC;
    }
    *out = p;
    return MP_OK;
}

} // namespace

// --------------------------------------------------------------- constructors

mp_result mp_prover_new(mp_prover **out, const uint8_t *g1_monomial_bytes,
                              size_t g1_monomial_len, const mp_options *opts) {
    if (!out || !g1_monomial_bytes) return MP_ERR_BADARGS;
    *out = nullptr;

    SetupTables tables;
    const bool validate = opts && opts->validate_setup;
    const char *cache = opts ? opts->table_cache_path : nullptr;

    // The cache records the digest of the setup it was derived from; hashing
    // the input is far cheaper than building the tables just to learn it.
    bool have = false;
    if (cache) {
        const uint64_t digest = compute_setup_digest(g1_monomial_bytes, g1_monomial_len);
        have = load_table_cache(cache, digest, tables) == MP_OK;
    }
    if (!have) {
        mp_result rc = build_setup_tables(g1_monomial_bytes, g1_monomial_len, validate, tables);
        if (rc != MP_OK) return rc;
        if (cache) save_table_cache(cache, tables);
    }

    @autoreleasepool {
        return createProver(out, tables, opts);
    }
}

mp_result mp_prover_new_default(mp_prover **out, const mp_options *opts) {
    return mp_prover_new(out, kEmbeddedSetupG1Monomial, kEmbeddedSetupSize, opts);
}


void mp_prover_free(mp_prover *p) { delete p; }

// -------------------------------------------------------------------- compute

namespace {

void encodeNtt(id<MTLComputeCommandEncoder> enc, mp_prover *p, id<MTLBuffer> out,
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
// Points per thread in the batched inversion.
//
// Each chunk pays one field inversion, which is ~570 multiplies deep and so
// costs a fixed ~2ms of latency no matter how few points it covers.  Bigger
// chunks amortise that better; smaller chunks give more threads.  Rather than
// pin a constant tuned to an 8-core M1, size the chunk so the dispatch keeps a
// few thousand threads busy, with a floor that keeps the inversion below ~18
// multiplies per point.  A 64-core part running large batches therefore gets
// proportionally more threads instead of inheriting this machine's shape.
uint32_t inversionChunk(uint32_t count, uint32_t floorChunk, uint32_t ceilChunk,
                        uint32_t targetThreads) {
    uint32_t chunk = count / targetThreads;
    if (chunk < floorChunk) chunk = floorChunk;
    if (chunk > ceilChunk) chunk = ceilChunk;
    return chunk;
}

void encodeReduce(id<MTLComputeCommandEncoder> enc, mp_prover *p, id<MTLBuffer> out,
                  uint32_t count, uint32_t batch) {
    [enc setComputePipelineState:p->psoReduce];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:p->bufBuckets offset:0 atIndex:1];
    [enc setBytes:&count length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((count + L_REDUCE_OUTPUTS_PER_TG - 1) /
                                              L_REDUCE_OUTPUTS_PER_TG,
                                          batch, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

struct NormalizeParams {
    uint32_t count;
    uint32_t chunk;
};

void encodeNormalize(id<MTLComputeCommandEncoder> enc, mp_prover *p, id<MTLBuffer> outAffine,
                     id<MTLBuffer> inJacobian, uint32_t count, uint32_t chunk) {
    NormalizeParams np{count, chunk};
    const uint32_t threads = (count + chunk - 1) / chunk;
    [enc setComputePipelineState:p->psoNormalize];
    [enc setBuffer:outAffine offset:0 atIndex:0];
    [enc setBuffer:inJacobian offset:0 atIndex:1];
    [enc setBuffer:p->bufNormScratch offset:0 atIndex:2];
    [enc setBytes:&np length:sizeof(np) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads < 64 ? threads : 64, 1, 1)];
}

// `stageSplit` is for the profiler only: it flushes the command buffer at each
// stage boundary so the stages can be timed individually.  Normal calls encode
// everything into one command buffer.
mp_result computeBatch(mp_prover *p, uint8_t *cells, uint8_t *proofs, const uint8_t *blobs,
                          uint32_t batch, bool stageSplit = false) {
    @autoreleasepool {
        StageTimes &st = p->lastStage;
        st = StageTimes{};
        const double tStart = nowMs();
        memcpy(p->bufBlob.contents, blobs, (size_t)batch * MP_BYTES_PER_BLOB);
        *(uint32_t *)p->bufErr.contents = 0;

        // The whole pipeline is one command buffer: nothing needs the host in
        // the middle, so there is no synchronisation point to pay for.
        id<MTLCommandBuffer> cb = [p->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        bool failed = false;

        // Ends the current stage. In profiling mode that means commit, wait,
        // record the GPU time and open a fresh buffer; otherwise it is a no-op.
        auto stage = [&](double *slot) {
            if (!stageSplit) return;
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) failed = true;
            if (slot) *slot = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            cb = [p->queue commandBuffer];
            enc = [cb computeCommandEncoder];
        };

        // 1. blob bytes -> bit-reversed Lagrange values
        [enc setComputePipelineState:p->psoBlobToFr];
        [enc setBuffer:p->bufLagrange offset:0 atIndex:0];
        [enc setBuffer:p->bufBlob offset:0 atIndex:1];
        [enc setBuffer:p->bufErr offset:0 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(kFieldElementsPerBlob, batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // 2. inverse transform of size 4096 -> monomial coefficients
        {
            NttParams q = nttPass1(4096, 64, 64, kFieldElementsPerBlob, kFieldElementsPerExtBlob);
            encodeNtt(enc, p, p->bufWorkA, p->bufLagrange, p->bufRootsInv, q, 64, batch);
            NttParams r = nttPass2(4096, 64, 64, kFieldElementsPerExtBlob, kFieldElementsPerExtBlob);
            r.scale = 1;
            memcpy(r.scale_val, p->invBlob, sizeof(r.scale_val));
            encodeNtt(enc, p, p->bufPolyExt, p->bufWorkA, p->bufRootsInv, r, 64, batch);
        }

        if (cells) {
            // 3. forward transform of size 8192 over the zero-padded polynomial
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
            stage(&st.scalar_stage);

            // 4. circulant columns and their size-128 transforms
            [enc setComputePipelineState:p->psoBuildCirculant];
            [enc setBuffer:p->bufCoeffs offset:0 atIndex:0];
            [enc setBuffer:p->bufPolyExt offset:0 atIndex:1];
            [enc setBuffer:p->bufRootsFwd offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(kPhaseATerms, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kCirculantSize, 1, 1)];

            // 5a. phase A scalar pass: recode, histogram, load-order, sort
            [enc setComputePipelineState:p->psoPhaseASort];
            [enc setBuffer:p->bufItems offset:0 atIndex:0];
            [enc setBuffer:p->bufStarts offset:0 atIndex:1];
            [enc setBuffer:p->bufPerm offset:0 atIndex:2];
            [enc setBuffer:p->bufCoeffs offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kPhaseATerms, 1, 1)];

            // 5b. phase A curve pass: fixed-base bucket MSM
            [enc setComputePipelineState:p->psoPhaseA];
            [enc setBuffer:p->bufBuckets offset:0 atIndex:0];
            [enc setBuffer:p->bufItems offset:0 atIndex:1];
            [enc setBuffer:p->bufStarts offset:0 atIndex:2];
            [enc setBuffer:p->bufPerm offset:0 atIndex:3];
            [enc setBuffer:p->bufTable offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kNumBuckets, 1, 1)];

            stage(&st.phase_a);
            encodeReduce(enc, p, p->bufPoints, kCirculantSize, batch);
            stage(&st.reduce_a);

            // 6. doubling ladder over u[j], then to affine for the mixed adds
            [enc setComputePipelineState:p->psoLadder];
            [enc setBuffer:p->bufLadderJac offset:0 atIndex:0];
            [enc setBuffer:p->bufPoints offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(kCirculantSize, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];

            stage(&st.ladder);

            const uint32_t ladderPoints = (uint32_t)batch * kCirculantSize * kLadderPositions;
            encodeNormalize(enc, p, p->bufLadderAff, p->bufLadderJac, ladderPoints,
                            inversionChunk(ladderPoints, 32, 128, 8192));
            stage(&st.normalize_ladder);

            // 7. phase B: the fused circulant map
            [enc setComputePipelineState:p->psoPhaseB];
            [enc setBuffer:p->bufBuckets offset:0 atIndex:0];
            [enc setBuffer:p->bufLadderAff offset:0 atIndex:1];
            [enc setBuffer:p->bufKernelItems offset:0 atIndex:2];
            [enc setBuffer:p->bufKernelOffsets offset:0 atIndex:3];
            [enc setBuffer:p->bufKernelPerm offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(kCirculantSize, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(kNumBuckets, 1, 1)];

            stage(&st.phase_b);
            encodeReduce(enc, p, p->bufProofs, kCirculantSize, batch);
            stage(&st.reduce_b);

            // 8. proofs -> affine -> compressed bytes
            const uint32_t proofPoints = (uint32_t)batch * kCirculantSize;
            encodeNormalize(enc, p, p->bufProofsAff, p->bufProofs, proofPoints,
                            inversionChunk(proofPoints, 4, 32, 2048));
            stage(&st.normalize_proofs);

            [enc setComputePipelineState:p->psoCompress];
            [enc setBuffer:p->bufProofBytes offset:0 atIndex:0];
            [enc setBuffer:p->bufProofsAff offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(kCirculantSize, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            stage(&st.compress);
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error || failed) {
            NSLog(@"metal-prover: command buffer failed: %@", cb.error);
            return MP_ERR_GPU;
        }
        if (*(uint32_t *)p->bufErr.contents != 0) return MP_ERR_INVALID_BLOB;

        if (cells) {
            memcpy(cells, p->bufCells.contents, (size_t)batch * kFieldElementsPerExtBlob * 32);
        }
        if (proofs) {
            memcpy(proofs, p->bufProofBytes.contents,
                   (size_t)batch * kCirculantSize * kBytesPerProof);
        }
        st.total = nowMs() - tStart;
        return MP_OK;
    }
}

} // namespace

mp_result mp_compute_cells_and_proofs(mp_prover *p, uint8_t *cells, uint8_t *proofs,
                                            const uint8_t *blob) {
    return mp_compute_cells_and_proofs_batch(p, cells, proofs, blob, 1);
}

mp_result mp_compute_cells_and_proofs_batch(mp_prover *p, uint8_t *cells, uint8_t *proofs,
                                                  const uint8_t *blobs, size_t num_blobs) {
    if (!p || !blobs) return MP_ERR_BADARGS;
    if (!cells && !proofs) return MP_ERR_BADARGS;
    if (num_blobs == 0) return MP_OK;

    std::lock_guard<std::mutex> lock(p->mutex);
    const size_t cellBytes = (size_t)kCirculantSize * MP_BYTES_PER_CELL;
    const size_t proofBytes = (size_t)kCirculantSize * kBytesPerProof;

    for (size_t done = 0; done < num_blobs;) {
        const uint32_t batch = (uint32_t)std::min<size_t>(p->maxBatch, num_blobs - done);
        mp_result rc = computeBatch(p, cells ? cells + done * cellBytes : nullptr,
                                       proofs ? proofs + done * proofBytes : nullptr,
                                       blobs + done * MP_BYTES_PER_BLOB, batch);
        if (rc != MP_OK) return rc;
        done += batch;
    }
    return MP_OK;
}

// ------------------------------------------------------------------ profiling
//
// Development helper: computeBatch records what it measured into the prover,
// and this just runs it and hands the numbers back.  Reporting the real path
// matters here rather than a separate re-implementation of the dispatch order.
#include "profile.h"

namespace mp {

mp_result profile_batch(mp_prover *p, unsigned char *cells, unsigned char *proofs,
                           const unsigned char *blobs, unsigned batch, StageTimes &out) {
    if (!p || !blobs) return MP_ERR_BADARGS;
    std::lock_guard<std::mutex> lock(p->mutex);
    const mp_result rc = computeBatch(p, cells, proofs, blobs, batch, /*stageSplit=*/true);
    out = p->lastStage;
    return rc;
}

} // namespace mp
