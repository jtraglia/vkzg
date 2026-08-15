#include "cpu_msm.h"

#include "bls12_381.h"
#include "setup.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
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

inline void fr_from_device(Fr &out, const uint32_t *w) {
    for (int i = 0; i < 4; i++) out.v[i] = (uint64_t)w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
}

inline void affine_from_device(G1Affine &out, const uint32_t *w) {
    fp_from_device(out.x, w);
    fp_from_device(out.y, w + kFpLimbs);
}

inline void g1_to_device(uint32_t *out, const G1 &p) {
    fp_to_device(out, p.x);
    fp_to_device(out + kFpLimbs, p.y);
    fp_to_device(out + 2 * kFpLimbs, p.z);
}

// sum_{k=1}^{kNumBuckets} k * B[k-1], by running sum from the top.
G1 reduce_buckets(const G1 *buckets) {
    G1 running = kG1Identity, total = kG1Identity;
    for (int k = kNumBuckets - 1; k >= 0; k--) {
        g1_add(running, running, buckets[k]);
        g1_add(total, total, running);
    }
    return total;
}

} // namespace

// ------------------------------------------------------------------ pool

struct ThreadPool::Impl {
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable cv_start, cv_done;
    void (*fn)(void *, size_t) = nullptr;
    void *ctx = nullptr;
    size_t total = 0;
    std::atomic<size_t> next{0};
    size_t finished = 0;
    uint64_t generation = 0;
    bool stop = false;
};

ThreadPool::ThreadPool(unsigned threads) : impl_(new Impl), nthreads_(threads) {
    for (unsigned t = 0; t < threads; t++) {
        impl_->workers.emplace_back([this] {
            Impl *im = impl_;
            uint64_t seen = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(im->m);
                im->cv_start.wait(lk, [&] { return im->stop || im->generation != seen; });
                if (im->stop) return;
                seen = im->generation;
                auto fn = im->fn;
                auto ctx = im->ctx;
                const size_t total = im->total;
                lk.unlock();

                for (;;) {
                    const size_t i = im->next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= total) break;
                    fn(ctx, i);
                }

                lk.lock();
                if (++im->finished == im->workers.size()) im->cv_done.notify_one();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->stop = true;
    }
    impl_->cv_start.notify_all();
    for (auto &w : impl_->workers) w.join();
    delete impl_;
}

void ThreadPool::parallel_for(size_t n, void (*fn)(void *, size_t), void *ctx) {
    if (n == 0) return;
    if (nthreads_ == 0) {
        for (size_t i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->fn = fn;
        impl_->ctx = ctx;
        impl_->total = n;
        impl_->next.store(0, std::memory_order_relaxed);
        impl_->finished = 0;
        impl_->generation++;
    }
    impl_->cv_start.notify_all();
    std::unique_lock<std::mutex> lk(impl_->m);
    impl_->cv_done.wait(lk, [&] { return impl_->finished == impl_->workers.size(); });
}

// ---------------------------------------------------------------- phase A

namespace {
struct PhaseACtx {
    uint32_t *out;
    const uint32_t *coeffs;
    const uint32_t *table;
    int j0;
    int count; // outputs per blob
};

void phase_a_task(void *vctx, size_t idx) {
    const PhaseACtx *c = (const PhaseACtx *)vctx;
    const size_t blob = idx / (size_t)c->count;
    const int j = c->j0 + (int)(idx % (size_t)c->count);

    G1 buckets[kNumBuckets];
    for (int k = 0; k < kNumBuckets; k++) buckets[k] = kG1Identity;

    for (int i = 0; i < kPhaseATerms; i++) {
        Fr s;
        fr_from_device(s, c->coeffs + ((blob * kCirculantSize + (size_t)j) * kPhaseATerms +
                                       (size_t)i) *
                                          kFrLimbs);
        uint64_t canonical[4];
        fr_to_canonical(canonical, s);
        int32_t digits[kNumDigits];
        recode_scalar(digits, canonical);
        const uint32_t *base =
            c->table + ((size_t)j * kPhaseATerms + (size_t)i) * kNumDigits * kAffineWords;
        for (int d = 0; d < kNumDigits; d++) {
            const int v = digits[d];
            if (v == 0) continue;
            G1Affine pt;
            affine_from_device(pt, base + (size_t)d * kAffineWords);
            int mag = v;
            if (mag < 0) {
                mag = -mag;
                fp_neg(pt.y, pt.y);
            }
            g1_add_mixed(buckets[mag - 1], buckets[mag - 1], pt);
        }
    }
    const G1 total = reduce_buckets(buckets);
    g1_to_device(c->out + (blob * kCirculantSize + (size_t)j) * kJacobianWords, total);
}

struct PhaseBCtx {
    uint32_t *out;
    const uint32_t *ladder;
    const uint32_t *items;
    const uint32_t *offsets;
    int a0;
    int count;
};

void phase_b_task(void *vctx, size_t idx) {
    const PhaseBCtx *c = (const PhaseBCtx *)vctx;
    const size_t blob = idx / (size_t)c->count;
    const int a = c->a0 + (int)(idx % (size_t)c->count);

    G1 buckets[kNumBuckets];
    for (int k = 0; k < kNumBuckets; k++) buckets[k] = kG1Identity;

    const uint32_t *lbase = c->ladder + blob * kCirculantSize * kLadderPositions * kAffineWords;
    // The items arrive already grouped by bucket, so walk buckets outermost.
    for (int k = 0; k < kNumBuckets; k++) {
        for (uint32_t it = c->offsets[k]; it < c->offsets[k + 1]; it++) {
            const uint32_t packed = c->items[it];
            const int e = (int)(packed & 0xff);
            const int d = (int)((packed >> 8) & 0xff);
            const bool neg = ((packed >> 16) & 1) != 0;
            const int src = ((a - e) % kCirculantSize + kCirculantSize) % kCirculantSize;
            G1Affine pt;
            affine_from_device(pt,
                               lbase + ((size_t)src * kLadderPositions + (size_t)d) * kAffineWords);
            if (neg) fp_neg(pt.y, pt.y);
            g1_add_mixed(buckets[k], buckets[k], pt);
        }
    }
    const G1 total = reduce_buckets(buckets);
    g1_to_device(c->out + (blob * kCirculantSize + (size_t)a) * kJacobianWords, total);
}
} // namespace

void cpu_phase_a(ThreadPool &pool, uint32_t *out, const uint32_t *coeffs, const uint32_t *table,
                 size_t num_blobs, int j0, int j1) {
    if (j1 <= j0) return;
    PhaseACtx ctx{out, coeffs, table, j0, j1 - j0};
    pool.parallel_for(num_blobs * (size_t)(j1 - j0), phase_a_task, &ctx);
}

void cpu_phase_b(ThreadPool &pool, uint32_t *out, const uint32_t *ladder_affine,
                 const uint32_t *items, const uint32_t *offsets, size_t num_blobs, int a0, int a1) {
    if (a1 <= a0) return;
    PhaseBCtx ctx{out, ladder_affine, items, offsets, a0, a1 - a0};
    pool.parallel_for(num_blobs * (size_t)(a1 - a0), phase_b_task, &ctx);
}

} // namespace kzgpu
