#include "setup.h"

#include "bls12_381_constants.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>

namespace vkp {
namespace {

// Convert host limbs to the device's 32-bit little-endian layout.
void fp_to_device(uint32_t *out, const Fp &a) {
    for (int i = 0; i < 6; i++) {
        out[2 * i] = (uint32_t)a.v[i];
        out[2 * i + 1] = (uint32_t)(a.v[i] >> 32);
    }
}

void fr_to_device(uint32_t *out, const Fr &a) {
    for (int i = 0; i < 4; i++) {
        out[2 * i] = (uint32_t)a.v[i];
        out[2 * i + 1] = (uint32_t)(a.v[i] >> 32);
    }
}

void affine_to_device(uint32_t *out, const G1Affine &p) {
    fp_to_device(out, p.x);
    fp_to_device(out + kFpLimbs, p.y);
}

uint64_t fnv1a(const uint8_t *data, size_t n, uint64_t h = 1469598103934665603ULL) {
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Run `fn(i)` for i in [0, n) across the available cores.
template <typename Fn>
void parallel_for(size_t n, Fn fn) {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    size_t nthreads = std::min<size_t>(hw, n);
    if (nthreads <= 1) {
        for (size_t i = 0; i < n; i++) fn(i);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (size_t t = 0; t < nthreads; t++) {
        threads.emplace_back([&, t] {
            for (size_t i = t; i < n; i += nthreads) fn(i);
        });
    }
    for (auto &th : threads) th.join();
}

} // namespace

void recode_scalar(int32_t digits[kNumDigits], const uint64_t canonical[4]) {
    // Signed window recoding: digit = byte + carry, and if it exceeds the
    // half-window we borrow from the next digit.  Because every scalar is
    // below r < 2^255, the top byte is at most 0x7f and the carry chain always
    // terminates within kNumDigits digits.
    int carry = 0;
    for (int d = 0; d < kNumDigits; d++) {
        int limb = d / 8;
        int shift = (d % 8) * 8;
        int byte = (int)((canonical[limb] >> shift) & 0xff);
        int v = byte + carry;
        if (v > kNumBuckets) {
            v -= 2 * kNumBuckets;
            carry = 1;
        } else {
            carry = 0;
        }
        digits[d] = v;
    }
    // carry must have been absorbed; a non-zero carry here would mean the
    // scalar was >= 2^256.
}

void fr_fft(Fr *data, size_t n, const Fr *roots8192) {
    const size_t stride = (size_t)kFieldElementsPerExtBlob / n;
    // bit-reversal permutation
    uint32_t bits = 0;
    while ((1u << bits) < n) bits++;
    for (size_t i = 0; i < n; i++) {
        size_t j = bit_reverse((uint32_t)i, bits);
        if (j > i) std::swap(data[i], data[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        for (size_t base = 0; base < n; base += len) {
            for (size_t j = 0; j < half; j++) {
                Fr t;
                fr_mul(t, data[base + j + half], roots8192[j * step * stride]);
                fr_sub(data[base + j + half], data[base + j], t);
                fr_add(data[base + j], data[base + j], t);
            }
        }
    }
}

void g1_fft(G1 *data, size_t n, const Fr *roots8192) {
    const size_t stride = (size_t)kFieldElementsPerExtBlob / n;
    uint32_t bits = 0;
    while ((1u << bits) < n) bits++;
    for (size_t i = 0; i < n; i++) {
        size_t j = bit_reverse((uint32_t)i, bits);
        if (j > i) std::swap(data[i], data[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half = len >> 1;
        size_t step = n / len;
        for (size_t base = 0; base < n; base += len) {
            for (size_t j = 0; j < half; j++) {
                G1 t;
                const Fr &w = roots8192[j * step * stride];
                if (fr_eq(w, kFrOne)) {
                    t = data[base + j + half];
                } else {
                    g1_mul(t, data[base + j + half], w);
                }
                g1_sub(data[base + j + half], data[base + j], t);
                g1_add(data[base + j], data[base + j], t);
            }
        }
    }
}

void compute_circulant_kernel(Fr kappa[kCirculantSize]) {
    // out = FFT(truncate_64(IFFT(u))) is a cyclic convolution with
    //     kappa[0]     = 1/2
    //     kappa[e]     = -1 / (64 * (w^e - 1))   for odd e
    //     kappa[e]     = 0                       for even e != 0
    // where w is the primitive 128th root of unity.  Derivation: the composed
    // map is (1/128) * sum_{b<64} w^{b(a-j)}, and that geometric sum collapses
    // because w^64 == -1.
    Fr roots[kCirculantSize];
    Fr w;
    fr_root_of_unity(w, 7); // 2^7 == 128
    roots[0] = kFrOne;
    for (int i = 1; i < kCirculantSize; i++) fr_mul(roots[i], roots[i - 1], w);

    for (int e = 0; e < kCirculantSize; e++) kappa[e] = kFrZero;

    Fr two, half;
    fr_from_u64(two, 2);
    fr_inv(half, two);
    kappa[0] = half;

    Fr sixtyfour, inv64;
    fr_from_u64(sixtyfour, 64);
    fr_inv(inv64, sixtyfour);
    for (int e = 1; e < kCirculantSize; e += 2) {
        Fr denom, t;
        fr_sub(denom, roots[e], kFrOne);
        fr_inv(t, denom);
        fr_mul(t, t, inv64);
        fr_neg(kappa[e], t);
    }
}

uint64_t compute_setup_digest(const uint8_t *g1_monomial_bytes, size_t len) {
    return fnv1a(g1_monomial_bytes, len);
}


vkp_result build_setup_tables(const uint8_t *g1_monomial_bytes, size_t len, bool validate,
                                SetupTables &out) {
    if (!g1_monomial_bytes || len != (size_t)VKP_NUM_SETUP_G1_POINTS * VKP_BYTES_PER_G1) {
        return VKP_ERR_BADARGS;
    }
    out.setup_digest = compute_setup_digest(g1_monomial_bytes, len);

    // ------------------------------------------------------------- decompress
    std::vector<G1Affine> setup_affine(kFieldElementsPerBlob);
    std::vector<uint8_t> ok(kFieldElementsPerBlob, 0);
    parallel_for(kFieldElementsPerBlob, [&](size_t i) {
        G1Affine a;
        if (!g1_decompress(a, g1_monomial_bytes + i * VKP_BYTES_PER_G1)) return;
        if (validate && !(g1_affine_is_on_curve(a) && g1_affine_in_subgroup(a))) return;
        setup_affine[i] = a;
        ok[i] = 1;
    });
    for (int i = 0; i < kFieldElementsPerBlob; i++) {
        if (!ok[i]) return VKP_ERR_SETUP;
    }

    // ---------------------------------------------------------- roots of unity
    std::vector<Fr> roots(kFieldElementsPerExtBlob + 1);
    std::vector<Fr> roots_inv(kFieldElementsPerExtBlob + 1);
    {
        Fr w;
        fr_root_of_unity(w, 13);
        roots[0] = kFrOne;
        for (int i = 1; i <= kFieldElementsPerExtBlob; i++) fr_mul(roots[i], roots[i - 1], w);
        if (!fr_eq(roots[kFieldElementsPerExtBlob], kFrOne)) return VKP_ERR_SETUP;
        for (int i = 0; i <= kFieldElementsPerExtBlob; i++) {
            roots_inv[i] = roots[kFieldElementsPerExtBlob - i];
        }
    }

    out.roots_fwd.resize((size_t)kFieldElementsPerExtBlob * kFrLimbs);
    out.roots_inv.resize((size_t)kFieldElementsPerExtBlob * kFrLimbs);
    for (int i = 0; i < kFieldElementsPerExtBlob; i++) {
        fr_to_device(&out.roots_fwd[(size_t)i * kFrLimbs], roots[i]);
        fr_to_device(&out.roots_inv[(size_t)i * kFrLimbs], roots_inv[i]);
    }
    {
        Fr t, inv;
        fr_from_u64(t, kFieldElementsPerBlob);
        fr_inv(inv, t);
        fr_to_device(out.inv_blob, inv);
    }

    // ----------------------------------------------- FK20 x_ext_fft columns
    // For each offset i, take a strided slice of the monomial setup, extend it
    // with identities to the circulant size, and transform it.  Column j of the
    // result is the base set for output j of phase A.
    std::vector<G1> columns((size_t)kCirculantSize * kPhaseATerms);
    parallel_for(kFieldElementsPerCell, [&](size_t offset) {
        std::vector<G1> x(kCirculantSize);
        for (int i = 0; i < kCirculantSize; i++) x[i] = kG1Identity;
        const int start = kFieldElementsPerBlob - kFieldElementsPerCell - 1 - (int)offset;
        for (int i = 0; i < kCellsPerBlob - 1; i++) {
            int j = start - i * kFieldElementsPerCell;
            g1_from_affine(x[i], setup_affine[j]);
        }
        // x[kCellsPerBlob - 1] and everything above stays at infinity.
        g1_fft(x.data(), kCirculantSize, roots.data());
        for (int row = 0; row < kCirculantSize; row++) {
            columns[(size_t)row * kPhaseATerms + offset] = x[row];
        }
    });

    // ------------------------------------------------------- position tables
    // T[j][i][d] = 2^(8d) * columns[j][i], stored affine.
    out.position_table.assign(kPositionTableWords, 0);
    const size_t bases = (size_t)kCirculantSize * kPhaseATerms;
    parallel_for(bases, [&](size_t b) {
        G1 ladder[kNumDigits];
        G1Affine aff[kNumDigits];
        ladder[0] = columns[b];
        for (int d = 1; d < kNumDigits; d++) {
            G1 acc = ladder[d - 1];
            for (int s = 0; s < kWindowBits; s++) g1_dbl(acc, acc);
            ladder[d] = acc;
        }
        g1_batch_to_affine(aff, ladder, kNumDigits);
        for (int d = 0; d < kNumDigits; d++) {
            size_t idx = (b * kNumDigits + (size_t)d) * kAffineWords;
            affine_to_device(&out.position_table[idx], aff[d]);
        }
    });

    // --------------------------------------------------- phase B kernel items
    //
    // Builds a bucket-sorted signed-digit item list for a circulant kernel of
    // `size` taps, shared by every output of that sub-problem (only a
    // rotation of the point index differs per output -- see k_phase_b.comp /
    // k_phase_b_split.comp). Used for the flat 128-tap kernel (kept for the
    // CPU reference) and, halved, for the split form's two 64-tap kernels
    // that the GPU path actually uses (see layout_defs.h).
    auto build_kernel_items = [](const Fr *kappa, int size, int expected_taps,
                                  std::vector<uint32_t> &items_out,
                                  std::vector<uint32_t> &offsets_out,
                                  std::vector<uint32_t> &perm_out) -> vkp_result {
        struct Item {
            uint32_t bucket; // |digit| - 1
            uint32_t packed; // tap | position << 8 | sign << 16
        };
        std::vector<Item> items;
        items.reserve((size_t)expected_taps * kNumDigits);

        int tap = 0;
        for (int e = 0; e < size; e++) {
            if (fr_is_zero(kappa[e])) continue;
            uint64_t canonical[4];
            fr_to_canonical(canonical, kappa[e]);
            int32_t digits[kNumDigits];
            recode_scalar(digits, canonical);
            for (int d = 0; d < kNumDigits; d++) {
                if (digits[d] == 0) continue;
                uint32_t mag = (uint32_t)(digits[d] < 0 ? -digits[d] : digits[d]);
                uint32_t sign = digits[d] < 0 ? 1u : 0u;
                items.push_back({mag - 1, (uint32_t)e | ((uint32_t)d << 8) | (sign << 16)});
            }
            tap++;
        }
        if (tap != expected_taps) return VKP_ERR_SETUP;

        std::stable_sort(items.begin(), items.end(),
                         [](const Item &a, const Item &b) { return a.bucket < b.bucket; });

        items_out.resize(items.size());
        offsets_out.assign(kNumBuckets + 1, 0);
        for (size_t i = 0; i < items.size(); i++) items_out[i] = items[i].packed;
        std::vector<uint32_t> counts(kNumBuckets, 0);
        for (const auto &it : items) counts[it.bucket]++;
        uint32_t acc = 0;
        for (int k = 0; k < kNumBuckets; k++) {
            offsets_out[k] = acc;
            acc += counts[k];
        }
        offsets_out[kNumBuckets] = acc;

        std::vector<uint32_t> order(kNumBuckets);
        for (int k = 0; k < kNumBuckets; k++) order[k] = (uint32_t)k;
        std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) {
            return counts[x] > counts[y];
        });
        perm_out = order;
        return VKP_OK;
    };

    Fr kappa[kCirculantSize];
    compute_circulant_kernel(kappa);
    vkp_result rc = build_kernel_items(kappa, kCirculantSize, kPhaseBTerms, out.kernel_items,
                                          out.kernel_offsets, out.kernel_perm);
    if (rc != VKP_OK) return rc;

    // Split form: X^128-1 = (X^64-1)(X^64+1). kappa+[i] = (kappa[i] +
    // kappa[i+64])/2, kappa-[i] = (kappa[i] - kappa[i+64])/2 for i in [0,64);
    // the 1/2 is folded in here so the final combine step is pure addition.
    // See the equivalence check in tests/test_reference.cpp.
    {
        Fr half;
        {
            Fr two;
            fr_from_u64(two, 2);
            fr_inv(half, two);
        }
        Fr kappa_plus[kCirculantHalf], kappa_minus[kCirculantHalf];
        for (int i = 0; i < kCirculantHalf; i++) {
            Fr sum, diff;
            fr_add(sum, kappa[i], kappa[i + kCirculantHalf]);
            fr_sub(diff, kappa[i], kappa[i + kCirculantHalf]);
            fr_mul(kappa_plus[i], sum, half);
            fr_mul(kappa_minus[i], diff, half);
        }
        rc = build_kernel_items(kappa_plus, kCirculantHalf, kPhaseBHalfTerms, out.kernel_items_plus,
                                out.kernel_offsets_plus, out.kernel_perm_plus);
        if (rc != VKP_OK) return rc;
        rc = build_kernel_items(kappa_minus, kCirculantHalf, kPhaseBHalfTerms, out.kernel_items_minus,
                                out.kernel_offsets_minus, out.kernel_perm_minus);
        if (rc != VKP_OK) return rc;
    }

    return VKP_OK;
}

// ------------------------------------------------------------------- cache

namespace {
struct CacheHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t setup_digest;
    uint64_t position_words;
    uint64_t roots_words;
    uint64_t kernel_items;
    uint64_t kernel_offsets;
    uint64_t kernel_perm;
    uint64_t kernel_items_plus;
    uint64_t kernel_items_minus;
    uint64_t kernel_offsets_half;  // same for plus and minus
    uint64_t kernel_perm_half;     // same for plus and minus
};

bool read_exact(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }
bool write_exact(FILE *f, const void *p, size_t n) { return fwrite(p, 1, n, f) == n; }

template <typename T>
bool read_vec(FILE *f, std::vector<T> &v, size_t count) {
    v.resize(count);
    return read_exact(f, v.data(), count * sizeof(T));
}
template <typename T>
bool write_vec(FILE *f, const std::vector<T> &v) {
    return write_exact(f, v.data(), v.size() * sizeof(T));
}
} // namespace

vkp_result load_table_cache(const std::string &path, uint64_t expected_digest, SetupTables &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return VKP_ERR_IO;
    CacheHeader h{};
    vkp_result rc = VKP_ERR_IO;
    if (!read_exact(f, &h, sizeof(h))) goto done;
    if (h.magic != kTableCacheMagic || h.version != kTableCacheVersion ||
        h.setup_digest != expected_digest || h.position_words != kPositionTableWords) {
        goto done;
    }
    out.setup_digest = h.setup_digest;
    if (!read_vec(f, out.position_table, h.position_words) ||
        !read_vec(f, out.roots_fwd, h.roots_words) ||
        !read_vec(f, out.roots_inv, h.roots_words) ||
        !read_vec(f, out.kernel_items, h.kernel_items) ||
        !read_vec(f, out.kernel_offsets, h.kernel_offsets) ||
        !read_vec(f, out.kernel_perm, h.kernel_perm) ||
        !read_vec(f, out.kernel_items_plus, h.kernel_items_plus) ||
        !read_vec(f, out.kernel_items_minus, h.kernel_items_minus) ||
        !read_vec(f, out.kernel_offsets_plus, h.kernel_offsets_half) ||
        !read_vec(f, out.kernel_offsets_minus, h.kernel_offsets_half) ||
        !read_vec(f, out.kernel_perm_plus, h.kernel_perm_half) ||
        !read_vec(f, out.kernel_perm_minus, h.kernel_perm_half) ||
        !read_exact(f, out.inv_blob, sizeof(out.inv_blob))) {
        goto done;
    }
    rc = VKP_OK;
done:
    fclose(f);
    return rc;
}

vkp_result save_table_cache(const std::string &path, const SetupTables &in) {
    const std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return VKP_ERR_IO;
    CacheHeader h{};
    h.magic = kTableCacheMagic;
    h.version = kTableCacheVersion;
    h.setup_digest = in.setup_digest;
    h.position_words = in.position_table.size();
    h.roots_words = in.roots_fwd.size();
    h.kernel_items = in.kernel_items.size();
    h.kernel_offsets = in.kernel_offsets.size();
    h.kernel_perm = in.kernel_perm.size();
    h.kernel_items_plus = in.kernel_items_plus.size();
    h.kernel_items_minus = in.kernel_items_minus.size();
    h.kernel_offsets_half = in.kernel_offsets_plus.size();
    h.kernel_perm_half = in.kernel_perm_plus.size();
    bool ok = write_exact(f, &h, sizeof(h)) && write_vec(f, in.position_table) &&
              write_vec(f, in.roots_fwd) && write_vec(f, in.roots_inv) &&
              write_vec(f, in.kernel_items) && write_vec(f, in.kernel_offsets) &&
              write_vec(f, in.kernel_perm) && write_vec(f, in.kernel_items_plus) &&
              write_vec(f, in.kernel_items_minus) && write_vec(f, in.kernel_offsets_plus) &&
              write_vec(f, in.kernel_offsets_minus) && write_vec(f, in.kernel_perm_plus) &&
              write_vec(f, in.kernel_perm_minus) &&
              write_exact(f, in.inv_blob, sizeof(in.inv_blob));
    fclose(f);
    if (!ok) {
        remove(tmp.c_str());
        return VKP_ERR_IO;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        return VKP_ERR_IO;
    }
    return VKP_OK;
}

} // namespace vkp
