#include "setup.h"

#include "bls12_381_constants.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>

namespace kzgpu {
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

kzgpu_result read_trusted_setup_file(const std::string &path, std::vector<uint8_t> &g1_monomial) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return KZGPU_ERR_IO;
    uint64_t n1 = 0, n2 = 0;
    if (fscanf(f, "%llu", (unsigned long long *)&n1) != 1 ||
        fscanf(f, "%llu", (unsigned long long *)&n2) != 1) {
        fclose(f);
        return KZGPU_ERR_SETUP;
    }
    if (n1 != KZGPU_NUM_SETUP_G1_POINTS) {
        fclose(f);
        return KZGPU_ERR_SETUP;
    }

    auto skip_hex = [&](size_t nbytes) {
        for (size_t i = 0; i < nbytes; i++) {
            unsigned v;
            if (fscanf(f, "%2x", &v) != 1) return false;
        }
        return true;
    };

    // Lagrange G1 section, then G2, then the monomial G1 section we want.
    if (!skip_hex((size_t)n1 * KZGPU_BYTES_PER_G1) || !skip_hex((size_t)n2 * 96)) {
        fclose(f);
        return KZGPU_ERR_SETUP;
    }
    g1_monomial.resize((size_t)n1 * KZGPU_BYTES_PER_G1);
    for (size_t i = 0; i < g1_monomial.size(); i++) {
        unsigned v;
        if (fscanf(f, "%2x", &v) != 1) {
            fclose(f);
            return KZGPU_ERR_SETUP;
        }
        g1_monomial[i] = (uint8_t)v;
    }
    fclose(f);
    return KZGPU_OK;
}

kzgpu_result build_setup_tables(const uint8_t *g1_monomial_bytes, size_t len, bool validate,
                                SetupTables &out) {
    if (!g1_monomial_bytes || len != (size_t)KZGPU_NUM_SETUP_G1_POINTS * KZGPU_BYTES_PER_G1) {
        return KZGPU_ERR_BADARGS;
    }
    out.setup_digest = fnv1a(g1_monomial_bytes, len);

    // ------------------------------------------------------------- decompress
    std::vector<G1Affine> setup_affine(kFieldElementsPerBlob);
    std::vector<uint8_t> ok(kFieldElementsPerBlob, 0);
    parallel_for(kFieldElementsPerBlob, [&](size_t i) {
        G1Affine a;
        if (!g1_decompress(a, g1_monomial_bytes + i * KZGPU_BYTES_PER_G1)) return;
        if (validate && !(g1_affine_is_on_curve(a) && g1_affine_in_subgroup(a))) return;
        setup_affine[i] = a;
        ok[i] = 1;
    });
    for (int i = 0; i < kFieldElementsPerBlob; i++) {
        if (!ok[i]) return KZGPU_ERR_SETUP;
    }

    // ---------------------------------------------------------- roots of unity
    std::vector<Fr> roots(kFieldElementsPerExtBlob + 1);
    std::vector<Fr> roots_inv(kFieldElementsPerExtBlob + 1);
    {
        Fr w;
        fr_root_of_unity(w, 13);
        roots[0] = kFrOne;
        for (int i = 1; i <= kFieldElementsPerExtBlob; i++) fr_mul(roots[i], roots[i - 1], w);
        if (!fr_eq(roots[kFieldElementsPerExtBlob], kFrOne)) return KZGPU_ERR_SETUP;
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
        fr_from_u64(t, kFieldElementsPerExtBlob);
        fr_inv(inv, t);
        fr_to_device(out.inv_ext_blob, inv);
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
    {
        Fr kappa[kCirculantSize];
        compute_circulant_kernel(kappa);

        struct Item {
            uint32_t bucket; // |digit| - 1
            uint32_t packed; // tap | position << 8 | sign << 16
        };
        std::vector<Item> items;
        items.reserve(kPhaseBItems);

        int tap = 0;
        for (int e = 0; e < kCirculantSize; e++) {
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
        if (tap != kPhaseBTerms) return KZGPU_ERR_SETUP;

        std::stable_sort(items.begin(), items.end(),
                         [](const Item &a, const Item &b) { return a.bucket < b.bucket; });

        out.kernel_items.resize(items.size());
        out.kernel_offsets.assign(kNumBuckets + 1, 0);
        for (size_t i = 0; i < items.size(); i++) out.kernel_items[i] = items[i].packed;
        // Prefix offsets: offsets[k] is the first item of bucket k.
        std::vector<uint32_t> counts(kNumBuckets, 0);
        for (const auto &it : items) counts[it.bucket]++;
        uint32_t acc = 0;
        for (int k = 0; k < kNumBuckets; k++) {
            out.kernel_offsets[k] = acc;
            acc += counts[k];
        }
        out.kernel_offsets[kNumBuckets] = acc;
    }

    return KZGPU_OK;
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
};

bool read_exact(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }
bool write_exact(FILE *f, const void *p, size_t n) { return fwrite(p, 1, n, f) == n; }
} // namespace

kzgpu_result load_table_cache(const std::string &path, uint64_t expected_digest, SetupTables &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return KZGPU_ERR_IO;
    CacheHeader h{};
    kzgpu_result rc = KZGPU_ERR_IO;
    if (!read_exact(f, &h, sizeof(h))) goto done;
    if (h.magic != kTableCacheMagic || h.version != kTableCacheVersion ||
        h.setup_digest != expected_digest || h.position_words != kPositionTableWords) {
        goto done;
    }
    out.setup_digest = h.setup_digest;
    out.position_table.resize(h.position_words);
    out.roots_fwd.resize(h.roots_words);
    out.roots_inv.resize(h.roots_words);
    out.kernel_items.resize(h.kernel_items);
    out.kernel_offsets.resize(h.kernel_offsets);
    if (!read_exact(f, out.position_table.data(), h.position_words * 4) ||
        !read_exact(f, out.roots_fwd.data(), h.roots_words * 4) ||
        !read_exact(f, out.roots_inv.data(), h.roots_words * 4) ||
        !read_exact(f, out.kernel_items.data(), h.kernel_items * 4) ||
        !read_exact(f, out.kernel_offsets.data(), h.kernel_offsets * 4) ||
        !read_exact(f, out.inv_ext_blob, sizeof(out.inv_ext_blob)) ||
        !read_exact(f, out.inv_blob, sizeof(out.inv_blob))) {
        goto done;
    }
    rc = KZGPU_OK;
done:
    fclose(f);
    return rc;
}

kzgpu_result save_table_cache(const std::string &path, const SetupTables &in) {
    const std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return KZGPU_ERR_IO;
    CacheHeader h{};
    h.magic = kTableCacheMagic;
    h.version = kTableCacheVersion;
    h.setup_digest = in.setup_digest;
    h.position_words = in.position_table.size();
    h.roots_words = in.roots_fwd.size();
    h.kernel_items = in.kernel_items.size();
    h.kernel_offsets = in.kernel_offsets.size();
    bool ok = write_exact(f, &h, sizeof(h)) &&
              write_exact(f, in.position_table.data(), in.position_table.size() * 4) &&
              write_exact(f, in.roots_fwd.data(), in.roots_fwd.size() * 4) &&
              write_exact(f, in.roots_inv.data(), in.roots_inv.size() * 4) &&
              write_exact(f, in.kernel_items.data(), in.kernel_items.size() * 4) &&
              write_exact(f, in.kernel_offsets.data(), in.kernel_offsets.size() * 4) &&
              write_exact(f, in.inv_ext_blob, sizeof(in.inv_ext_blob)) &&
              write_exact(f, in.inv_blob, sizeof(in.inv_blob));
    fclose(f);
    if (!ok) {
        remove(tmp.c_str());
        return KZGPU_ERR_IO;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        return KZGPU_ERR_IO;
    }
    return KZGPU_OK;
}

} // namespace kzgpu
