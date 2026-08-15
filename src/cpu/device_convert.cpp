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

void build_ladder_affine(uint32_t *out_affine, const uint32_t *u_jacobian, size_t num_points,
                         unsigned threads) {
    threads = pick_threads(threads);
    const size_t nchunks = std::min<size_t>(threads, std::max<size_t>(1, num_points / 8));
    const size_t per = (num_points + nchunks - 1) / nchunks;

    run_threads(nchunks, (unsigned)nchunks, [&](size_t c) {
        const size_t begin = c * per;
        const size_t end = std::min(num_points, begin + per);
        if (begin >= end) return;
        const size_t len = end - begin;

        std::vector<G1> jac(len * kLadderPositions);
        for (size_t i = 0; i < len; i++) {
            G1 acc;
            g1_from_device(acc, u_jacobian + (begin + i) * kJacobianWords);
            for (int d = 0; d < kLadderPositions; d++) {
                jac[i * kLadderPositions + d] = acc;
                for (int s = 0; s < kWindowBits; s++) g1_dbl(acc, acc);
            }
        }
        // One batched inversion for this chunk's whole ladder.
        std::vector<Fp> zs(jac.size()), inv(jac.size());
        for (size_t i = 0; i < jac.size(); i++) zs[i] = jac[i].z;
        batch_inverse(inv.data(), zs.data(), zs.size());
        for (size_t i = 0; i < jac.size(); i++) {
            uint32_t *dst = out_affine + (begin * kLadderPositions + i) * kAffineWords;
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
    });
}

void finalize_proofs(uint8_t *out, const uint32_t *jacobian, size_t num_blobs, unsigned threads) {
    threads = pick_threads(threads);
    run_threads(num_blobs, (unsigned)std::min<size_t>(threads, std::max<size_t>(num_blobs, 1)),
                [&](size_t b) {
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
                });
}

} // namespace kzgpu
