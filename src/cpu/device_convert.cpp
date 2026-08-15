#include "device_convert.h"

#include "../kzgpu_internal.h"
#include "bls12_381.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace kzgpu {
namespace {

inline void fp_from_device(Fp &out, const uint32_t *w) {
    for (int i = 0; i < 6; i++) out.v[i] = (uint64_t)w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
}

inline void fp_to_device(uint32_t *out, const Fp &a) {
    for (int i = 0; i < 6; i++) {
        out[2 * i] = (uint32_t)a.v[i];
        out[2 * i + 1] = (uint32_t)(a.v[i] >> 32);
    }
}

inline void g1_from_device(G1 &out, const uint32_t *w) {
    fp_from_device(out.x, w);
    fp_from_device(out.y, w + kFpLimbs);
    fp_from_device(out.z, w + 2 * kFpLimbs);
}

template <typename Fn>
void run_threads(size_t chunks, unsigned threads, Fn fn) {
    if (threads <= 1 || chunks <= 1) {
        for (size_t c = 0; c < chunks; c++) fn(c);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; t++) {
        pool.emplace_back([&, t] {
            for (size_t c = t; c < chunks; c += threads) fn(c);
        });
    }
    for (auto &th : pool) th.join();
}

unsigned pick_threads(unsigned requested) {
    if (requested) return requested;
    unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 4u;
}

} // namespace

void jacobian_to_affine_device(uint32_t *out_affine, const uint32_t *in_jacobian, size_t n,
                               unsigned threads) {
    threads = pick_threads(threads);
    // One batch per thread: the Montgomery trick is sequential within a batch,
    // so splitting costs one extra inversion per thread and buys linear speedup.
    const size_t nchunks = std::min<size_t>(threads, std::max<size_t>(1, n / 256));
    const size_t per = (n + nchunks - 1) / nchunks;

    run_threads(nchunks, (unsigned)nchunks, [&](size_t c) {
        const size_t begin = c * per;
        const size_t end = std::min(n, begin + per);
        if (begin >= end) return;
        const size_t len = end - begin;

        std::vector<Fp> zs(len), inv(len);
        for (size_t i = 0; i < len; i++) {
            fp_from_device(zs[i], in_jacobian + (begin + i) * kJacobianWords + 2 * kFpLimbs);
        }
        batch_inverse(inv.data(), zs.data(), len);

        for (size_t i = 0; i < len; i++) {
            uint32_t *dst = out_affine + (begin + i) * kAffineWords;
            if (fp_is_zero(zs[i])) { // point at infinity
                for (int k = 0; k < kAffineWords; k++) dst[k] = 0;
                continue;
            }
            G1 p;
            g1_from_device(p, in_jacobian + (begin + i) * kJacobianWords);
            Fp z2, z3, x, y;
            fp_sqr(z2, inv[i]);
            fp_mul(z3, z2, inv[i]);
            fp_mul(x, p.x, z2);
            fp_mul(y, p.y, z3);
            fp_to_device(dst, x);
            fp_to_device(dst + kFpLimbs, y);
        }
    });
}

namespace {
// One task per kLadderChunk input points.  The chunk is small enough that the
// P-cores can steal several while an E-core works through one -- with one
// static chunk per thread the whole call ran at E-core speed -- and large
// enough that a single batched inversion still amortises over 8 * 32 entries.
constexpr size_t kLadderChunk = 8;

struct LadderCtx {
    uint32_t *out;
    const uint32_t *in;
    size_t num_points;
};

void ladder_task(void *vctx, size_t chunk) {
    const LadderCtx *c = (const LadderCtx *)vctx;
    const size_t begin = chunk * kLadderChunk;
    const size_t end = std::min(c->num_points, begin + kLadderChunk);
    if (begin >= end) return;
    const size_t len = end - begin;

    G1 jac[kLadderChunk * kLadderPositions];
    for (size_t i = 0; i < len; i++) {
        G1 acc;
        g1_from_device(acc, c->in + (begin + i) * kJacobianWords);
        for (int d = 0; d < kLadderPositions; d++) {
            jac[i * kLadderPositions + d] = acc;
            for (int s = 0; s < kWindowBits; s++) g1_dbl(acc, acc);
        }
    }
    const size_t n = len * kLadderPositions;
    Fp zs[kLadderChunk * kLadderPositions], inv[kLadderChunk * kLadderPositions];
    for (size_t i = 0; i < n; i++) zs[i] = jac[i].z;
    batch_inverse(inv, zs, n);
    for (size_t i = 0; i < n; i++) {
        uint32_t *dst = c->out + (begin * kLadderPositions + i) * kAffineWords;
        if (fp_is_zero(jac[i].z)) {
            for (int k = 0; k < kAffineWords; k++) dst[k] = 0;
            continue;
        }
        Fp z2, z3, x, y;
        fp_sqr(z2, inv[i]);
        fp_mul(z3, z2, inv[i]);
        fp_mul(x, jac[i].x, z2);
        fp_mul(y, jac[i].y, z3);
        fp_to_device(dst, x);
        fp_to_device(dst + kFpLimbs, y);
    }
}
} // namespace

void build_ladder_affine(ThreadPool *pool, uint32_t *out_affine, const uint32_t *u_jacobian,
                         size_t num_points, unsigned threads) {
    LadderCtx ctx{out_affine, u_jacobian, num_points};
    const size_t chunks = (num_points + kLadderChunk - 1) / kLadderChunk;
    if (pool) {
        pool->parallel_for(chunks, ladder_task, &ctx);
        return;
    }
    threads = pick_threads(threads);
    run_threads(chunks, threads, [&](size_t c) { ladder_task(&ctx, c); });
}

namespace {
struct FinalizeCtx {
    uint8_t *out;
    const uint32_t *jacobian;
};

void finalize_task(void *vctx, size_t b) {
    const FinalizeCtx *c = (const FinalizeCtx *)vctx;
    uint8_t *out = c->out;
    const uint32_t *jacobian = c->jacobian;
    {
                    G1 pts[kCirculantSize];
                    for (int i = 0; i < kCirculantSize; i++) {
                        // FK20 emits proofs in natural order; cells are indexed
                        // bit-reversed, so permute on the way out.
                        G1 p;
                        g1_from_device(p, jacobian + (b * kCirculantSize + (size_t)i) * kJacobianWords);
                        pts[bit_reverse((uint32_t)i, 7)] = p;
                    }
                    G1Affine aff[kCirculantSize];
                    g1_batch_to_affine(aff, pts, kCirculantSize);
                    for (int i = 0; i < kCirculantSize; i++) {
                        g1_compress(out + (b * kCirculantSize + (size_t)i) * kBytesPerProof, aff[i]);
                    }
    }
}
} // namespace

void finalize_proofs(ThreadPool *pool, uint8_t *out, const uint32_t *jacobian, size_t num_blobs,
                     unsigned threads) {
    FinalizeCtx ctx{out, jacobian};
    if (pool) {
        pool->parallel_for(num_blobs, finalize_task, &ctx);
        return;
    }
    threads = pick_threads(threads);
    run_threads(num_blobs, (unsigned)std::min<size_t>(threads, std::max<size_t>(num_blobs, 1)),
                [&](size_t b) { finalize_task(&ctx, b); });
}

} // namespace kzgpu
