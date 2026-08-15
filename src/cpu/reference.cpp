#include "reference.h"

#include <cstring>

namespace kzgpu {
namespace {

void fr_from_device(Fr &out, const uint32_t *w) {
    for (int i = 0; i < 4; i++) {
        out.v[i] = (uint64_t)w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
    }
}

void fp_from_device(Fp &out, const uint32_t *w) {
    for (int i = 0; i < 6; i++) {
        out.v[i] = (uint64_t)w[2 * i] | ((uint64_t)w[2 * i + 1] << 32);
    }
}

void affine_from_device(G1Affine &out, const uint32_t *w) {
    fp_from_device(out.x, w);
    fp_from_device(out.y, w + kFpLimbs);
}

std::vector<Fr> device_roots(const std::vector<uint32_t> &src) {
    std::vector<Fr> r(src.size() / kFrLimbs);
    for (size_t i = 0; i < r.size(); i++) fr_from_device(r[i], &src[i * kFrLimbs]);
    return r;
}

// Sum over a bucket set: result = sum_{k=1}^{kNumBuckets} k * B[k-1].
// Running-sum form, which is what the GPU does too (just with the groups
// spread over threads).
G1 reduce_buckets(const G1 *buckets) {
    G1 running = kG1Identity, total = kG1Identity;
    for (int k = kNumBuckets - 1; k >= 0; k--) {
        g1_add(running, running, buckets[k]);
        g1_add(total, total, running);
    }
    return total;
}

} // namespace

void fr_fft_fwd(Fr *data, size_t n, const SetupTables &tables) {
    static thread_local std::vector<Fr> roots;
    if (roots.empty()) roots = device_roots(tables.roots_fwd);
    fr_fft(data, n, roots.data());
}

void fr_ifft(Fr *data, size_t n, const SetupTables &tables) {
    static thread_local std::vector<Fr> rootsi;
    if (rootsi.empty()) rootsi = device_roots(tables.roots_inv);
    fr_fft(data, n, rootsi.data());
    Fr inv_n;
    Fr t;
    fr_from_u64(t, (uint64_t)n);
    fr_inv(inv_n, t);
    for (size_t i = 0; i < n; i++) fr_mul(data[i], data[i], inv_n);
}

kzgpu_result blob_to_polynomial(Fr *poly, const uint8_t *blob, const SetupTables &tables) {
    // The blob holds Lagrange-basis evaluations in bit-reversed order.
    std::vector<Fr> lagrange(kFieldElementsPerBlob);
    for (int i = 0; i < kFieldElementsPerBlob; i++) {
        if (!fr_from_bytes(lagrange[i], blob + (size_t)i * kBytesPerFieldElement)) {
            return KZGPU_ERR_INVALID_BLOB;
        }
    }
    // Undo the bit-reversal, then transform to monomial form.
    std::vector<Fr> brp(kFieldElementsPerBlob);
    for (int i = 0; i < kFieldElementsPerBlob; i++) {
        brp[bit_reverse((uint32_t)i, 12)] = lagrange[i];
    }
    fr_ifft(brp.data(), kFieldElementsPerBlob, tables);
    memcpy(poly, brp.data(), sizeof(Fr) * kFieldElementsPerBlob);
    return KZGPU_OK;
}

void build_circulant_coeffs(Fr coeffs[kCirculantSize][kPhaseATerms], const Fr *poly,
                            const SetupTables &tables) {
    // For each offset i, the first column of the circulant matrix embedding the
    // i-th Toeplitz block, transformed and then transposed into per-output
    // scalar vectors.
    const int d = kFieldElementsPerBlob - 1;
    for (int i = 0; i < kPhaseATerms; i++) {
        Fr c[kCirculantSize];
        for (int m = 0; m < kCirculantSize; m++) c[m] = kFrZero;
        c[0] = poly[d - i];
        for (int j = 1; j < kCellsPerBlob - 1; j++) {
            c[kCirculantSize - j] = poly[d - i - j * kFieldElementsPerCell];
        }
        fr_fft_fwd(c, kCirculantSize, tables);
        for (int j = 0; j < kCirculantSize; j++) coeffs[j][i] = c[j];
    }
}

void phase_a_msm(G1 *u, const Fr coeffs[kCirculantSize][kPhaseATerms], const SetupTables &tables) {
    for (int j = 0; j < kCirculantSize; j++) {
        G1 buckets[kNumBuckets];
        for (int k = 0; k < kNumBuckets; k++) buckets[k] = kG1Identity;

        for (int i = 0; i < kPhaseATerms; i++) {
            uint64_t canonical[4];
            fr_to_canonical(canonical, coeffs[j][i]);
            int32_t digits[kNumDigits];
            recode_scalar(digits, canonical);
            for (int dd = 0; dd < kNumDigits; dd++) {
                if (digits[dd] == 0) continue;
                const size_t base = (size_t)j * kPhaseATerms + (size_t)i;
                const size_t idx = (base * kNumDigits + (size_t)dd) * kAffineWords;
                G1Affine pt;
                affine_from_device(pt, &tables.position_table[idx]);
                int mag = digits[dd];
                if (mag < 0) {
                    mag = -mag;
                    fp_neg(pt.y, pt.y);
                }
                g1_add_mixed(buckets[mag - 1], buckets[mag - 1], pt);
            }
        }
        u[j] = reduce_buckets(buckets);
    }
}

void phase_b_circulant(G1 *out, const G1 *u, const SetupTables &tables) {
    // Ladder: L[j][d] = 2^(8d) * u[j], affine so the accumulation can use
    // mixed additions.
    std::vector<G1Affine> ladder((size_t)kCirculantSize * kLadderPositions);
    {
        std::vector<G1> jac((size_t)kCirculantSize * kLadderPositions);
        for (int j = 0; j < kCirculantSize; j++) {
            G1 acc = u[j];
            for (int d = 0; d < kLadderPositions; d++) {
                jac[(size_t)j * kLadderPositions + d] = acc;
                for (int s = 0; s < kWindowBits; s++) g1_dbl(acc, acc);
            }
        }
        g1_batch_to_affine(ladder.data(), jac.data(), jac.size());
    }

    for (int a = 0; a < kCirculantSize; a++) {
        G1 buckets[kNumBuckets];
        for (int k = 0; k < kNumBuckets; k++) buckets[k] = kG1Identity;

        for (int kk = 0; kk < kNumBuckets; kk++) {
            const int k = (int)tables.kernel_perm[kk];
            const uint32_t lo = tables.kernel_offsets[k];
            const uint32_t hi = tables.kernel_offsets[k + 1];
            for (uint32_t it = lo; it < hi; it++) {
                const uint32_t packed = tables.kernel_items[it];
                const int e = (int)(packed & 0xff);
                const int d = (int)((packed >> 8) & 0xff);
                const bool neg = ((packed >> 16) & 1) != 0;
                const int src = ((a - e) % kCirculantSize + kCirculantSize) % kCirculantSize;
                G1Affine pt = ladder[(size_t)src * kLadderPositions + d];
                if (neg) fp_neg(pt.y, pt.y);
                g1_add_mixed(buckets[k], buckets[k], pt);
            }
        }
        out[a] = reduce_buckets(buckets);
    }
}

void phase_b_via_g1_ffts(G1 *out, const G1 *u, const SetupTables &tables) {
    static thread_local std::vector<Fr> rootsf, rootsi;
    if (rootsf.empty()) {
        rootsf = device_roots(tables.roots_fwd);
        rootsi = device_roots(tables.roots_inv);
    }
    std::vector<G1> v(u, u + kCirculantSize);
    g1_fft(v.data(), kCirculantSize, rootsi.data()); // unscaled inverse transform
    Fr inv_n, t;
    fr_from_u64(t, kCirculantSize);
    fr_inv(inv_n, t);
    for (int i = 0; i < kCirculantSize; i++) {
        G1 s;
        g1_mul(s, v[i], inv_n);
        v[i] = s;
    }
    for (int i = kCellsPerBlob; i < kCirculantSize; i++) v[i] = kG1Identity;
    g1_fft(v.data(), kCirculantSize, rootsf.data());
    memcpy(out, v.data(), sizeof(G1) * kCirculantSize);
}

kzgpu_result reference_compute(const SetupTables &tables, const uint8_t *blob, uint8_t *cells,
                               uint8_t *proofs, ReferenceIntermediates *dbg) {
    std::vector<Fr> poly(kFieldElementsPerBlob);
    kzgpu_result rc = blob_to_polynomial(poly.data(), blob, tables);
    if (rc != KZGPU_OK) return rc;
    if (dbg) memcpy(dbg->poly, poly.data(), sizeof(Fr) * kFieldElementsPerBlob);

    if (cells) {
        std::vector<Fr> ext(kFieldElementsPerExtBlob);
        memcpy(ext.data(), poly.data(), sizeof(Fr) * kFieldElementsPerBlob);
        for (int i = kFieldElementsPerBlob; i < kFieldElementsPerExtBlob; i++) ext[i] = kFrZero;
        fr_fft_fwd(ext.data(), kFieldElementsPerExtBlob, tables);
        if (dbg) memcpy(dbg->ext_evals, ext.data(), sizeof(Fr) * kFieldElementsPerExtBlob);
        for (int i = 0; i < kFieldElementsPerExtBlob; i++) {
            uint32_t dst = bit_reverse((uint32_t)i, 13);
            fr_to_bytes(cells + (size_t)dst * kBytesPerFieldElement, ext[i]);
        }
    }

    if (proofs) {
        static thread_local std::vector<Fr> flat;
        std::vector<std::vector<Fr>> tmp;
        auto coeffs = new Fr[kCirculantSize][kPhaseATerms];
        build_circulant_coeffs(coeffs, poly.data(), tables);
        if (dbg) memcpy(dbg->coeffs, coeffs, sizeof(Fr) * kCirculantSize * kPhaseATerms);

        std::vector<G1> u(kCirculantSize);
        phase_a_msm(u.data(), coeffs, tables);
        delete[] coeffs;
        if (dbg) memcpy(dbg->u, u.data(), sizeof(G1) * kCirculantSize);

        std::vector<G1> pr(kCirculantSize);
        phase_b_circulant(pr.data(), u.data(), tables);
        if (dbg) memcpy(dbg->proofs, pr.data(), sizeof(G1) * kCirculantSize);

        // Bit-reverse into cell order, then serialise.
        std::vector<G1> brp(kCirculantSize);
        for (int i = 0; i < kCirculantSize; i++) brp[bit_reverse((uint32_t)i, 7)] = pr[i];
        std::vector<G1Affine> aff(kCirculantSize);
        g1_batch_to_affine(aff.data(), brp.data(), kCirculantSize);
        for (int i = 0; i < kCirculantSize; i++) {
            g1_compress(proofs + (size_t)i * kBytesPerProof, aff[i]);
        }
    }
    return KZGPU_OK;
}

} // namespace kzgpu
